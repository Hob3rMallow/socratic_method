"""Run the frozen C3-F0 model on a scroll region.

One command turns a scroll name and a region into two things: an OME-NGFF
prediction store in global scroll coordinates that a viewer opens over the CT,
and a ScrollFiesta cube grid that ``grid_pipeline`` and ``socratic-repair``
consume unchanged.

Every model parameter comes from the ``inference`` section of the executable
recipe, not from a flag: eight-way mirror test-time augmentation, the shipped
operating threshold, a 32-voxel raw halo around each 128-voxel cube, bfloat16
autocast.  The checkpoint is verified against the recipe's pinned size and
SHA-256 before anything is downloaded.  The only model choice left to the
operator is the threshold, and only between the two values the inference
preregistration considered.

Without ``--run`` nothing is written: the checked plan is printed instead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from . import hf_export, ome_zarr, repair
from . import recipe as recipe_module

PROFILE = "f0-grid-predict-v1"
PLAN_SCHEMA = "socratic-predict-plan-v1"
RECEIPT_SCHEMA = "socratic-predict-receipt-v1"
REGISTRY_SCHEMA = "socratic-method-scroll-registry-v1"
ENGINE_DIR = "f0"
DELIVERABLE = "prediction.zarr"
LOCK_SUFFIX = ".predict.lock"
PLAN_FILE = "predict_plan.json"
RECEIPT_FILE = "predict_receipt.json"
RAW_INVENTORY = "raw_inventory.jsonl"
DEFAULT_REGISTRY = Path("recipes") / "f0" / "scrolls.json"
DEFAULT_PATHS = Path("recipes") / "f0" / "paths.local.json"
MAX_BAND_BYTES = 4 * 1024**3
_REGION = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*,\s*(\d+)\s*:\s*(\d+)\s*,\s*(\d+)\s*:\s*(\d+)\s*$")
_PITCH = re.compile(r"(\d+\.\d+)um")
_AWS_CREDENTIAL_ENV = (
    "AWS_ACCESS_KEY_ID",
    "AWS_PROFILE",
    "AWS_ROLE_ARN",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
)


# --------------------------------------------------------------------------
# pinned parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InferencePins:
    """The runner parameters the recipe pins. Only the threshold is negotiable."""

    operating_threshold: float
    postprocessor_threshold: float
    test_time_augmentation: str
    mirror_tta: bool
    amp_dtype: str
    chunk_size: int
    halo: int
    context_shape_zyx: tuple[int, int, int]
    device: str
    max_cpu_threads: int
    raw_dtype: str
    levels: int
    zarr_chunk: int
    pitch_range: tuple[float, float]
    pitch_refuse_below: float
    postprocessor_pitch_range: tuple[float, float]

    def permitted_thresholds(self) -> tuple[float, float]:
        return (self.operating_threshold, self.postprocessor_threshold)

    def threshold_role(self, threshold: float) -> str:
        if threshold == self.operating_threshold:
            return "shipped-operating-point"
        if threshold == self.postprocessor_threshold:
            return "postprocessor-pin"
        raise ValueError("threshold is not one of the pinned values")

    def as_dict(self) -> dict[str, Any]:
        return {
            "operating_threshold": self.operating_threshold,
            "postprocessor_threshold": self.postprocessor_threshold,
            "test_time_augmentation": self.test_time_augmentation,
            "mirror_tta": self.mirror_tta,
            "amp_dtype": self.amp_dtype,
            "chunk_size": self.chunk_size,
            "halo_voxels": self.halo,
            "context_shape_zyx": list(self.context_shape_zyx),
            "device": self.device,
            "max_cpu_threads": self.max_cpu_threads,
            "levels": self.levels,
        }


def load_inference_pins(recipe: dict[str, Any]) -> InferencePins:
    if "inference" not in recipe:
        raise ValueError(
            "this recipe has no inference section; socratic-predict runs only "
            "recipes that pin their inference (recipes/f0/recipe.json)"
        )
    section = recipe["inference"]
    if not isinstance(section, dict):
        raise TypeError("recipe inference must be a JSON object")
    required = (
        "operating_threshold",
        "postprocessor_threshold",
        "test_time_augmentation",
        "mirror_tta",
        "amp_dtype",
        "chunk_size",
        "halo_voxels",
        "context_shape_zyx",
        "device",
        "max_cpu_threads",
    )
    missing = [key for key in required if key not in section]
    if missing:
        raise ValueError(f"recipe inference section is missing keys: {', '.join(missing)}")

    operating = float(section["operating_threshold"])
    postprocessor = float(section["postprocessor_threshold"])
    for name, value in (("operating", operating), ("postprocessor", postprocessor)):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name}_threshold must lie in (0, 1), got {value}")
    if operating == postprocessor:
        raise ValueError("the two pinned thresholds must differ")

    augmentation = str(section["test_time_augmentation"])
    mirror = section["mirror_tta"]
    if not isinstance(mirror, bool):
        raise TypeError("inference.mirror_tta must be a boolean")
    if (augmentation == "8-way-mirror") != mirror:
        raise ValueError("inference.test_time_augmentation disagrees with mirror_tta")
    amp = str(section["amp_dtype"])
    if amp not in {"bfloat16", "float16"}:
        raise ValueError("inference.amp_dtype must be bfloat16 or float16")

    chunk = int(section["chunk_size"])
    halo = int(section["halo_voxels"])
    if chunk <= 0 or halo < 0:
        raise ValueError("inference.chunk_size must be positive and halo_voxels non-negative")
    if halo >= chunk:
        raise ValueError("inference.halo_voxels must be smaller than chunk_size")
    context = tuple(int(size) for size in section["context_shape_zyx"])
    if len(context) != 3 or any(size != chunk + 2 * halo for size in context):
        raise ValueError(
            f"inference.context_shape_zyx must equal chunk_size + 2*halo_voxels "
            f"({chunk + 2 * halo}), got {list(context)}"
        )

    threads = int(section["max_cpu_threads"])
    if not 1 <= threads <= 16:
        raise ValueError("inference.max_cpu_threads must lie in [1, 16]")

    release = recipe.get("release", {})
    for key, value in (
        ("operating_threshold", operating),
        ("training_record_operating_threshold", postprocessor),
        ("test_time_augmentation", augmentation),
    ):
        if key in release and release[key] != value:
            raise ValueError(
                f"recipe inference.{key.replace('training_record_', '')} disagrees "
                f"with release.{key}"
            )

    raw = section.get("raw_contract", {})
    pitch = section.get("voxel_size_um", {})
    deliverable = section.get("deliverable", {})
    low, high = pitch.get("accepted_range", [7.5, 10.0])
    post_low, post_high = pitch.get("postprocessor_range", [low, high])
    return InferencePins(
        operating_threshold=operating,
        postprocessor_threshold=postprocessor,
        test_time_augmentation=augmentation,
        mirror_tta=mirror,
        amp_dtype=amp,
        chunk_size=chunk,
        halo=halo,
        context_shape_zyx=context,  # type: ignore[arg-type]
        device=str(section["device"]),
        max_cpu_threads=threads,
        raw_dtype=str(raw.get("dtype", "uint8")),
        levels=int(deliverable.get("levels", 6)),
        zarr_chunk=int((deliverable.get("chunks_zyx") or [chunk])[0]),
        pitch_range=(float(low), float(high)),
        pitch_refuse_below=float(pitch.get("refuse_below", 4.0)),
        postprocessor_pitch_range=(float(post_low), float(post_high)),
    )


def selected_checkpoint_contract(recipe: dict[str, Any]) -> dict[str, Any]:
    contract = recipe.get("release", {}).get("selected_checkpoint")
    if not isinstance(contract, dict) or "sha256" not in contract or "bytes" not in contract:
        raise ValueError("recipe release.selected_checkpoint is missing or incomplete")
    return contract


# --------------------------------------------------------------------------
# scroll registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScrollSpec:
    id: str
    volume_uri: str | None
    uri_source: str | None
    voxel_size_um: float | None
    energy_keV: int | None
    shape_l0_zyx: tuple[int, int, int] | None
    umbilicus_yx: tuple[float, float] | None
    umbilicus_source: str | None
    f0_role: str
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "volume_uri": self.volume_uri,
            "uri_source": self.uri_source,
            "voxel_size_um": self.voxel_size_um,
            "energy_keV": self.energy_keV,
            "shape_l0_zyx": list(self.shape_l0_zyx) if self.shape_l0_zyx else None,
            "umbilicus_yx": list(self.umbilicus_yx) if self.umbilicus_yx else None,
            "umbilicus_source": self.umbilicus_source,
            "f0_role": self.f0_role,
            "notes": self.notes,
        }


def default_registry_path(root: Path | None = None) -> Path:
    return (root or _repository_root()) / DEFAULT_REGISTRY


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_scroll_registry(path: Path | None = None) -> dict[str, ScrollSpec]:
    source = Path(path) if path is not None else default_registry_path()
    document = repair.read_object(source)
    if document.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(f"{source}: expected schema {REGISTRY_SCHEMA}")
    registry: dict[str, ScrollSpec] = {}
    for row in document.get("scrolls", []):
        shape = row.get("shape_l0_zyx")
        umbilicus = row.get("umbilicus_yx")
        spec = ScrollSpec(
            id=str(row["id"]),
            volume_uri=row.get("volume_uri"),
            uri_source=row.get("uri_source"),
            voxel_size_um=row.get("voxel_size_um"),
            energy_keV=row.get("energy_keV"),
            shape_l0_zyx=tuple(int(size) for size in shape) if shape else None,  # type: ignore[arg-type]
            umbilicus_yx=tuple(float(v) for v in umbilicus) if umbilicus else None,  # type: ignore[arg-type]
            umbilicus_source=row.get("umbilicus_source"),
            f0_role=str(row.get("f0_role", "unseen")),
            notes=str(row.get("notes", "")),
        )
        registry[spec.id] = spec
    if not registry:
        raise ValueError(f"{source}: the registry lists no scrolls")
    return registry


def _fold(value: str) -> str:
    return value.strip().replace("-", "").replace("_", "").casefold()


def resolve_scroll(name: str, registry: dict[str, ScrollSpec]) -> ScrollSpec:
    """Accept the name as written, the engine's alias spellings, or a loose form."""

    if name in registry:
        return registry[name]
    from crossres_pred.schema import canonical_scroll_id

    canonical = canonical_scroll_id(name)
    if canonical in registry:
        return registry[canonical]
    folded = {_fold(key): key for key in registry}
    hit = folded.get(_fold(name))
    if hit is not None:
        return registry[hit]
    raise ValueError(
        f"unknown scroll {name!r}; valid ids are {', '.join(sorted(registry))}"
    )


def scroll_for_volume(
    volume: str | None, registry_path: Path | None = None
) -> ScrollSpec | None:
    """Recognise the scroll a recorded volume path belongs to, or return None."""

    if not volume:
        return None
    name = str(volume).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if not name:
        return None
    try:
        registry = load_scroll_registry(registry_path)
    except (FileNotFoundError, ValueError):
        return None
    for spec in registry.values():
        if not spec.volume_uri:
            continue
        recorded = spec.volume_uri.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        if recorded == name:
            return spec
    return None


# --------------------------------------------------------------------------
# region geometry
# --------------------------------------------------------------------------


def parse_region(text: str) -> tuple[int, int, int, int, int, int]:
    """Parse ``z0:z1,y0:y1,x0:x1`` into a half-open level-0 voxel box."""

    match = _REGION.match(text)
    if match is None:
        raise ValueError(
            f"region must be z0:z1,y0:y1,x0:x1 in level-0 voxels, got {text!r}"
        )
    values = tuple(int(group) for group in match.groups())
    return validate_region(values)


def validate_region(values: Sequence[int]) -> tuple[int, int, int, int, int, int]:
    box = tuple(int(value) for value in values)
    if len(box) != 6:
        raise ValueError("region needs six coordinates: Z0 Z1 Y0 Y1 X0 X1")
    for low, high, axis in zip(box[0::2], box[1::2], "zyx", strict=True):
        if low < 0:
            raise ValueError(f"region {axis} start must not be negative")
        if low >= high:
            raise ValueError(
                f"region must be half-open with Z0 < Z1, Y0 < Y1, X0 < X1; "
                f"{axis} is {low}:{high}"
            )
    return box  # type: ignore[return-value]


def align_region(
    region: Sequence[int], *, chunk: int
) -> tuple[tuple[int, ...], bool]:
    """Expand ``region`` outward to the cube lattice. Returns (aligned, changed)."""

    aligned: list[int] = []
    for index, value in enumerate(region):
        if index % 2 == 0:
            aligned.append((value // chunk) * chunk)
        else:
            aligned.append(-(-value // chunk) * chunk)
    return tuple(aligned), tuple(aligned) != tuple(region)


def region_origins(region: Sequence[int], *, chunk: int) -> list[tuple[int, int, int]]:
    z0, z1, y0, y1, x0, x1 = region
    return [
        (z, y, x)
        for z in range(z0, z1, chunk)
        for y in range(y0, y1, chunk)
        for x in range(x0, x1, chunk)
    ]


def format_cube_id(origin: Sequence[int]) -> str:
    return f"z{origin[0]:05d}_y{origin[1]:05d}_x{origin[2]:05d}"


def parse_cube_id(cube_id: str) -> tuple[int, int, int]:
    pieces = cube_id.split("_")
    if len(pieces) != 3 or [piece[:1] for piece in pieces] != ["z", "y", "x"]:
        raise ValueError(f"invalid cube ID: {cube_id!r}")
    try:
        return tuple(int(piece[1:]) for piece in pieces)  # type: ignore[return-value]
    except ValueError as error:
        raise ValueError(f"invalid cube ID: {cube_id!r}") from error


def context_origins(
    origin: Sequence[int], *, chunk: int, halo: int
) -> tuple[tuple[int, int, int], ...]:
    """The cubes one target needs. Mirrors the engine's own context rule."""

    lower = tuple(value - halo for value in origin)
    upper = tuple(value + chunk + halo for value in origin)
    if any(value < 0 for value in lower):
        raise ValueError("context bounds extend below zero")
    starts = [
        range((low // chunk) * chunk, ((high - 1) // chunk) * chunk + 1, chunk)
        for low, high in zip(lower, upper, strict=True)
    ]
    return tuple(product(*starts))  # type: ignore[return-value]


def classify_targets(
    origins: Iterable[tuple[int, int, int]],
    *,
    chunk: int,
    halo: int,
    extent_zyx: Sequence[int] | None = None,
    raw_grid: Path | None = None,
) -> tuple[list[str], list[dict[str, str]], list[tuple[int, int, int]]]:
    """Split targets into predictable and skipped, and collect the cubes to stage.

    A target is predictable only when every one of its context cubes exists in
    full.  Missing context is never padded with zeros: the target is dropped and
    the reason recorded.
    """

    complete: list[str] = []
    skipped: list[dict[str, str]] = []
    carve: set[tuple[int, int, int]] = set()
    for origin in sorted(origins):
        cube_id = format_cube_id(origin)
        try:
            context = context_origins(origin, chunk=chunk, halo=halo)
        except ValueError:
            skipped.append(
                {"cube_id": cube_id, "reason": "context extends below the volume origin"}
            )
            continue
        reason = None
        for neighbour in context:
            if extent_zyx is not None and any(
                value + chunk > size for value, size in zip(neighbour, extent_zyx, strict=True)
            ):
                reason = f"context cube {format_cube_id(neighbour)} lies outside the volume"
                break
            if raw_grid is not None and not (
                raw_grid / "cubes_RAW" / f"{format_cube_id(neighbour)}.tif"
            ).is_file():
                reason = f"context cube {format_cube_id(neighbour)} is missing from the grid"
                break
        if reason is not None:
            skipped.append({"cube_id": cube_id, "reason": reason})
            continue
        complete.append(cube_id)
        carve.update(context)
    return complete, skipped, sorted(carve)


def band_plan(
    carve_origins: Sequence[tuple[int, int, int]],
    *,
    chunk: int,
    max_band_bytes: int = MAX_BAND_BYTES,
) -> list[dict[str, Any]]:
    """Group cubes into slab reads aligned to the store's own chunk lattice.

    One read covers a whole z-band of cubes and as many y rows as fit the byte
    budget, so each 128-voxel chunk of the source volume is fetched exactly once.
    """

    bands: dict[int, list[tuple[int, int, int]]] = {}
    for origin in carve_origins:
        bands.setdefault(origin[0], []).append(origin)
    plan: list[dict[str, Any]] = []
    for z in sorted(bands):
        members = bands[z]
        x_low = min(origin[2] for origin in members)
        x_high = max(origin[2] for origin in members) + chunk
        row_bytes = chunk * (x_high - x_low)
        rows = max(1, int(max_band_bytes // max(1, row_bytes)) // chunk)
        y_values = sorted({origin[1] for origin in members})
        index = 0
        while index < len(y_values):
            group = y_values[index : index + rows]
            y_low, y_high = group[0], group[-1] + chunk
            plan.append(
                {
                    "z": z,
                    "y_range": [y_low, y_high],
                    "x_range": [x_low, x_high],
                    "bytes": chunk * (y_high - y_low) * (x_high - x_low),
                    "cubes": [
                        list(origin) for origin in sorted(members) if y_low <= origin[1] < y_high
                    ],
                }
            )
            index += rows
    return plan


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def _is_remote(spec: str) -> bool:
    return spec.startswith(("s3://", "http://", "https://", "gs://"))


def anonymous_by_default() -> bool:
    return not any(os.environ.get(name) for name in _AWS_CREDENTIAL_ENV)


def open_ct_volume(spec: str, *, anon: bool | None = None) -> tuple[Any, dict[str, Any]]:
    """Open a CT volume, local or remote, without materialising it."""

    if _is_remote(spec):
        uri, separator, key = spec.rpartition("::")
        if not separator:
            uri, key = spec, ""
        try:
            import zarr

            from crossres_pred.voxel.io import _enable_zarr_v2_one_byte_dtype_aliases

            _enable_zarr_v2_one_byte_dtype_aliases()
            options = (
                {"anon": anonymous_by_default() if anon is None else bool(anon)}
                if uri.startswith("s3://")
                else None
            )
            array = zarr.open(uri, path=key or "0", mode="r", storage_options=options)
        except Exception as error:
            raise ValueError(
                f"cannot open volume {spec}: {type(error).__name__}: {error}"
            ) from error
        access = "s3-anonymous" if (options or {}).get("anon") else "remote"
    else:
        from crossres_pred.voxel.io import open_volume

        try:
            array = open_volume(spec)
        except Exception as error:
            raise ValueError(
                f"cannot open volume {spec}: {type(error).__name__}: {error}"
            ) from error
        access = "local"

    shape = tuple(int(size) for size in array.shape)
    if len(shape) != 3:
        raise ValueError(f"volume must be a 3-D z-y-x array; got shape {shape}")
    chunks = getattr(array, "chunks", None)
    info = {
        "spec": spec,
        "access": access,
        "shape_zyx": list(shape),
        "dtype": str(np.dtype(array.dtype)),
        "chunks_zyx": [int(size) for size in chunks] if chunks else None,
    }
    return array, info


def parse_voxel_size(text: str | None) -> float | None:
    if not text:
        return None
    found = _PITCH.findall(str(text))
    return float(found[-1]) if found else None


def resolve_checkpoint(
    *,
    explicit: Path | None,
    model_hf: str | None,
    paths: dict[str, Any] | None,
    cache: Path | None,
    contract: dict[str, Any],
) -> tuple[Path, str]:
    """Locate the frozen checkpoint and prove it is the one the recipe pins."""

    source = "argument"
    if explicit is not None:
        checkpoint = Path(explicit).expanduser().resolve()
    elif model_hf is not None:
        checkpoint = _download_checkpoint(model_hf, contract, cache)
        source = f"hugging-face:{model_hf}"
    elif paths and paths.get("selected_checkpoint"):
        checkpoint = Path(str(paths["selected_checkpoint"])).expanduser().resolve()
        source = "paths-file"
    else:
        raise ValueError(
            "no checkpoint: pass --checkpoint, --model-hf, or set selected_checkpoint "
            "in recipes/f0/paths.local.json"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    hf_export._validate_selected_checkpoint(checkpoint, contract)
    return checkpoint, source


def _download_checkpoint(repo_id: str, contract: dict[str, Any], cache: Path | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ValueError(
            "--model-hf needs huggingface_hub: install this project with the publish extra"
        ) from error
    filename = str(contract.get("file") or "checkpoint.pt")
    try:
        located = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache) if cache else None,
        )
    except Exception as error:
        raise ValueError(
            f"cannot fetch {filename} from Hugging Face repository {repo_id}: "
            f"{type(error).__name__}: {error}"
        ) from error
    return Path(located).resolve()


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def plan_predict(
    *,
    output: Path,
    scroll: str | None = None,
    volume: str | None = None,
    grid: Path | None = None,
    region: Sequence[int] | None = None,
    recipe_path: Path | None = None,
    paths_file: Path | None = None,
    registry_path: Path | None = None,
    checkpoint: Path | None = None,
    model_hf: str | None = None,
    threshold: float | None = None,
    umbilicus: Sequence[float] | None = None,
    voxel_size_um: float | None = None,
    device: str | None = None,
    max_cpu_threads: int | None = None,
    levels: int | None = None,
    copy_raw: bool = False,
    cache: Path | None = None,
    anon: bool | None = None,
    max_band_bytes: int = MAX_BAND_BYTES,
    resume: bool = False,
) -> dict[str, Any]:
    """Check every input and describe the run. Writes nothing."""

    chosen = [name for name, value in (("scroll", scroll), ("volume", volume), ("grid", grid)) if value]
    if len(chosen) != 1:
        raise ValueError("pass exactly one of --scroll, --volume or --grid")

    recipe_path = Path(recipe_path or recipe_module.default_recipe_path()).expanduser().resolve()
    recipe = repair.read_object(recipe_path)
    pins = load_inference_pins(recipe)
    contract = selected_checkpoint_contract(recipe)
    chunk = pins.chunk_size

    if paths_file is None:
        candidate = _repository_root() / DEFAULT_PATHS
        paths_file = candidate if candidate.is_file() else None
    paths = (
        recipe_module.load_paths(
            Path(paths_file).expanduser().resolve(),
            required=(),
            optional=("selected_checkpoint",),
        )
        if paths_file
        else None
    )

    threshold = pins.operating_threshold if threshold is None else float(threshold)
    if threshold not in pins.permitted_thresholds():
        raise ValueError(
            f"threshold {threshold} is not allowed: "
            f"{pins.operating_threshold} is the shipped operating point and "
            f"{pins.postprocessor_threshold} the postprocessor pin; "
            "no other value was preregistered"
        )
    device = device or pins.device
    if device.split(":")[0] != pins.device.split(":")[0]:
        raise ValueError(
            f"recipe pins device type {pins.device.split(':')[0]!r}; {device!r} is not the "
            "qualified inference (an index such as cuda:1 selects a card)"
        )
    threads = pins.max_cpu_threads if max_cpu_threads is None else int(max_cpu_threads)
    if not 1 <= threads <= 16:
        raise ValueError("--max-cpu-threads must lie in [1, 16]")
    levels = pins.levels if levels is None else int(levels)
    if levels < 1:
        raise ValueError("--levels must be at least 1")

    output = Path(output).expanduser().resolve()
    lock = output.with_name(output.name + LOCK_SUFFIX)
    if grid is not None:
        resolved_grid = Path(grid).expanduser().resolve()
        if output == resolved_grid or resolved_grid in output.parents:
            raise ValueError(
                "output must be a new directory outside the input grid; predicting into "
                "the grid would overwrite the predictions it was carved from"
            )
    if output.exists():
        if not resume:
            raise FileExistsError(f"output exists: {output}; outputs are never overwritten")
        if (output / RECEIPT_FILE).is_file():
            raise FileExistsError(f"output is already complete: {output / RECEIPT_FILE}")
        if not (output / PLAN_FILE).is_file():
            raise ValueError(f"nothing to resume: {output / PLAN_FILE} is missing")
    partials = sorted(output.parent.glob(output.name + f"/{ENGINE_DIR}.partial-*"))
    if partials:
        raise FileExistsError(
            f"engine partial output exists: {partials[0]}; it requires inspection, "
            "not automatic recovery"
        )

    # --- source -----------------------------------------------------------
    scroll_spec: ScrollSpec | None = None
    registry_umbilicus = None
    source: dict[str, Any] = {}
    extent: tuple[int, int, int] | None = None
    volume_shape: tuple[int, int, int] | None = None
    grid_manifest: dict[str, Any] | None = None
    input_grid: Path | None = None
    shape_source: str | None = None
    pitch_source = "argument" if voxel_size_um is not None else None

    if scroll:
        registry = load_scroll_registry(registry_path)
        scroll_spec = resolve_scroll(scroll, registry)
        mirrors = (paths or {}).get("scroll_mirrors") or {}
        volume = mirrors.get(scroll_spec.id) or scroll_spec.volume_uri
        if not volume:
            raise ValueError(
                f"scroll {scroll_spec.id} has no recorded volume URI "
                f"({scroll_spec.notes or 'no note'}); pass --volume"
            )
        if scroll_spec.uri_source == "legacy" and _is_remote(volume):
            raise ValueError(
                f"scroll {scroll_spec.id} is a legacy acquisition that is not on the "
                "open-data bucket; pass --volume with a local copy"
            )
        registry_umbilicus = scroll_spec.umbilicus_yx
        if voxel_size_um is None and scroll_spec.voxel_size_um:
            voxel_size_um = float(scroll_spec.voxel_size_um)
            pitch_source = "scroll-registry"

    if grid:
        mode = "grid"
        input_grid = Path(grid).expanduser().resolve()
        grid_manifest = repair.read_object(input_grid / "manifest.json")
        if int(grid_manifest.get("chunk_size", 0)) != chunk:
            raise ValueError(
                f"grid manifest {input_grid / 'manifest.json'} must declare "
                f"chunk_size {chunk} (found {grid_manifest.get('chunk_size')})"
            )
        if not (input_grid / "cubes_RAW").is_dir():
            raise ValueError(f"grid has no cubes_RAW directory: {input_grid}")
        present = input_grid / "cubes_PRED" / "present.json"
        if not present.is_file():
            raise ValueError(
                f"grid has no cubes_PRED/present.json (the target inventory): {input_grid}"
            )
        published = sorted({str(value) for value in json.loads(present.read_text(encoding="utf-8"))})
        if scroll_spec is None:
            # A carved grid records the volume it came from but not that volume's shape.
            # Recognising the scroll recovers the full frame, so the deliverable spans the
            # scroll rather than just the box, and the training-corpus warning still fires.
            scroll_spec = scroll_for_volume(
                (grid_manifest.get("sources") or {}).get("raw"), registry_path
            )
        origins = [parse_cube_id(cube_id) for cube_id in published]
        if region is not None:
            box = validate_region(region)
            aligned, snapped = align_region(box, chunk=chunk)
            origins = [
                origin
                for origin in origins
                if all(
                    low <= value < high
                    for value, low, high in zip(origin, aligned[0::2], aligned[1::2], strict=True)
                )
            ]
            if not origins:
                raise ValueError(f"region selects no cube listed in {present}")
        else:
            box = None
            aligned = None
            snapped = False
        source = {
            "mode": mode,
            "input_grid": str(input_grid),
            "manifest": grid_manifest,
            "published_cubes": len(published),
            "spec": (grid_manifest.get("sources") or {}).get("raw"),
        }
        if scroll_spec is not None:
            source["scroll"] = scroll_spec.as_dict()
        if voxel_size_um is None:
            voxel_size_um = grid_manifest.get("voxel_size_um") or parse_voxel_size(
                str((grid_manifest.get("sources") or {}).get("raw") or "")
            )
            if voxel_size_um is not None:
                pitch_source = "input-manifest"
        if registry_umbilicus is None and grid_manifest.get("umbilicus_yx"):
            registry_umbilicus = tuple(float(v) for v in grid_manifest["umbilicus_yx"])
        complete, skipped, carve = classify_targets(
            origins, chunk=chunk, halo=pins.halo, raw_grid=input_grid
        )
        if aligned is None:
            lows = [min(origin[axis] for origin in origins) for axis in range(3)]
            highs = [max(origin[axis] for origin in origins) + chunk for axis in range(3)]
            aligned = tuple(value for pair in zip(lows, highs, strict=True) for value in pair)
            box = aligned
        shape_hint = grid_manifest.get("volume_shape_zyx")
        shape_source = "input-manifest" if shape_hint else None
        if not shape_hint and scroll_spec is not None and scroll_spec.shape_l0_zyx:
            shape_hint = scroll_spec.shape_l0_zyx
            shape_source = "scroll-registry"
        volume_shape = tuple(int(v) for v in shape_hint) if shape_hint else None
    else:
        mode = "volume"
        if region is None:
            raise ValueError("--region (or --bbox) is required when predicting from a volume")
        box = validate_region(region)
        aligned, snapped = align_region(box, chunk=chunk)
        array, info = open_ct_volume(str(volume), anon=anon)
        source = {"mode": mode, **info}
        if scroll_spec is not None:
            source["scroll"] = scroll_spec.as_dict()
        if np.dtype(array.dtype) != np.uint8:
            raise ValueError(
                f"volume dtype {np.dtype(array.dtype)} is not uint8; F0 expects the public "
                "masked-volume scale (uint8, clipped to [0, 212], mean 87.5, std 47.7) and "
                "will not rescale silently"
            )
        volume_shape = tuple(int(size) for size in array.shape)  # type: ignore[assignment]
        shape_source = "volume"
        extent = tuple((size // chunk) * chunk for size in volume_shape)  # type: ignore[assignment]
        for axis, (low, high, size) in enumerate(
            zip(aligned[0::2], aligned[1::2], volume_shape, strict=True)
        ):
            if high > size:
                raise ValueError(
                    f"region {'zyx'[axis]} {low}:{high} exceeds the volume shape "
                    f"{tuple(volume_shape)}"
                )
        if voxel_size_um is None:
            voxel_size_um = parse_voxel_size(str(volume))
            if voxel_size_um is not None:
                pitch_source = "parsed-from-name"
        complete, skipped, carve = classify_targets(
            region_origins(aligned, chunk=chunk), chunk=chunk, halo=pins.halo, extent_zyx=extent
        )

    if not complete:
        detail = skipped[0]["reason"] if skipped else "no cube in the region"
        raise ValueError(
            f"no target cube has complete {pins.halo}-voxel context ({detail}); "
            "move the region further inside the volume"
        )

    if voxel_size_um is not None:
        pitch = float(voxel_size_um)
        if not math.isfinite(pitch) or pitch <= 0:
            raise ValueError("--voxel-size-um must be a positive number")
        if pitch < pins.pitch_refuse_below:
            raise ValueError(
                f"voxel size {pitch} um is below {pins.pitch_refuse_below} um; F0 is a coarse "
                f"{pins.pitch_range[0]}-{pins.pitch_range[1]} um-class model and this is not "
                "the model for fine volumes"
            )
        if not pins.pitch_range[0] <= pitch <= pins.pitch_range[1]:
            raise ValueError(
                f"voxel size {pitch} um is outside the model's trained range "
                f"{list(pins.pitch_range)} um"
            )
        voxel_size_um = pitch

    if umbilicus is not None:
        umbilicus_value = tuple(float(value) for value in umbilicus)
        umbilicus_source = "argument"
        if (
            grid_manifest
            and grid_manifest.get("umbilicus_yx")
            and tuple(float(v) for v in grid_manifest["umbilicus_yx"]) != umbilicus_value
        ):
            raise ValueError(
                "--umbilicus disagrees with the input grid's manifest; "
                "predict into a new grid rather than relabelling this one"
            )
    elif registry_umbilicus is not None:
        umbilicus_value = tuple(float(value) for value in registry_umbilicus)
        umbilicus_source = (
            scroll_spec.umbilicus_source
            if scroll_spec and scroll_spec.umbilicus_yx
            else "input-manifest"
        )
    else:
        umbilicus_value = None
        umbilicus_source = None

    checkpoint_path, checkpoint_source = resolve_checkpoint(
        explicit=checkpoint,
        model_hf=model_hf,
        paths=paths,
        cache=cache,
        contract=contract,
    )

    # The sealed-pin gate belongs to the repository's own recipe: running one means
    # claiming the frozen engine produced the result. A recipe from elsewhere (a test
    # fixture, a fork) is not making that claim, so the check is skipped and said to be.
    engine_pins_state: Any = "not checked: the recipe is outside this repository"
    root = _repository_root()
    if root in recipe_path.parents and (root / "recipes" / "final_c3_250k_20260907").is_dir():
        from . import engine_pins as engine_pins_module

        try:
            ok, failures = engine_pins_module.check(root)
        except (FileNotFoundError, KeyError) as error:
            engine_pins_state = f"not checked: {error}"
        else:
            if failures:
                raise ValueError(
                    "engine snapshot differs from the sealed pins; run "
                    "socratic-engine-pins --check"
                )
            engine_pins_state = {"ok": len(ok), "failures": 0}

    ring = [origin for origin in carve if format_cube_id(origin) not in set(complete)]
    bands = band_plan(carve, chunk=chunk, max_band_bytes=max_band_bytes) if mode == "volume" else []
    zarr_shape = volume_shape or tuple(int(value) for value in aligned[1::2])

    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "profile": PROFILE,
        "mode": mode,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "recipe": {
            "path": str(recipe_path),
            "version": recipe.get("version"),
            "release_id": recipe.get("release", {}).get("release_id"),
        },
        "pins": pins.as_dict(),
        "threshold": threshold,
        "threshold_role": pins.threshold_role(threshold),
        "device": device,
        "max_cpu_threads": threads,
        "levels": levels,
        "checkpoint": {
            **repair.file_identity(checkpoint_path),
            "samples": contract.get("samples"),
            "source": checkpoint_source,
        },
        "source": source,
        "scroll": scroll_spec.as_dict() if scroll_spec else None,
        "region_requested": list(box) if box else None,
        "region": list(aligned),
        "region_expanded_to_lattice": bool(snapped),
        "volume_shape_zyx": list(volume_shape) if volume_shape else None,
        "target_cube_ids": complete,
        "skipped_target_cube_ids": skipped,
        "carve_cube_ids": [format_cube_id(origin) for origin in carve],
        "ring_cube_ids": [format_cube_id(origin) for origin in ring],
        "umbilicus_yx": list(umbilicus_value) if umbilicus_value else None,
        "umbilicus_source": umbilicus_source,
        "voxel_size_um": voxel_size_um,
        "voxel_size_um_source": pitch_source,
        "raw_transfer": "carve" if mode == "volume" else ("copy" if copy_raw else "hardlink"),
        "copy_raw": bool(copy_raw),
        "cache": str(Path(cache).expanduser().resolve()) if cache else None,
        "bands": bands,
        "output": str(output),
        "deliverable": str(output / DELIVERABLE),
        "lock": str(lock),
        "engine_directory": str(output / ENGINE_DIR),
        "engine_pins": engine_pins_state,
        "estimates": {
            "raw_bytes": len(carve) * chunk**3,
            "probability_bytes": len(complete) * chunk**3 * 2,
            "prediction_bytes": len(complete) * chunk**3,
            "max_band_bytes": max((band["bytes"] for band in bands), default=0),
        },
        "zarr": {
            "shape_zyx": list(zarr_shape),
            "shape_source": shape_source or "region-upper-bound",
            "chunks_zyx": [pins.zarr_chunk] * 3,
            "levels": levels,
        },
        "resume": bool(resume),
        "threshold_note": (
            f"T={pins.operating_threshold:g} is the shipped operating point; "
            "socratic-repair, socratic-continuity and socratic-sheet-patches are qualified "
            f"at T={pins.postprocessor_threshold:g} only"
        ),
        "research_only": True,
        "automatic_deployment": False,
    }
    plan["equivalent_command"] = equivalent_command(plan)
    return plan


def equivalent_command(plan: dict[str, Any]) -> list[str]:
    """The engine invocation this plan is equivalent to. Nothing is hidden."""

    pins = plan["pins"]
    command = [
        "crossres-voxel",
        "infer-grid",
        "--source",
        plan["output"],
        "--checkpoint",
        plan["checkpoint"]["path"],
        "--output",
        plan["engine_directory"],
        "--threshold",
        f"{plan['threshold']:g}",
        "--halo",
        str(pins["halo_voxels"]),
        "--device",
        plan["device"],
        "--amp-dtype",
        pins["amp_dtype"],
        "--tta" if pins["mirror_tta"] else "--no-tta",
        "--max-cpu-threads",
        str(plan["max_cpu_threads"]),
    ]
    return command


# --------------------------------------------------------------------------
# human-readable plan
# --------------------------------------------------------------------------


def _mib(value: int) -> str:
    return f"{value / 1024**2:,.1f} MiB"


def render_plan(plan: dict[str, Any]) -> str:
    pins = plan["pins"]
    lines = ["socratic-predict plan", ""]
    scroll = plan.get("scroll")
    source = plan["source"]
    if plan["mode"] == "grid":
        lines.append(f"  grid       {source['input_grid']}")
        lines.append(f"             {source['published_cubes']} cubes published")
    if scroll:
        pitch = f"{scroll['voxel_size_um']:g} um" if scroll.get("voxel_size_um") else "pitch unknown"
        energy = f", {scroll['energy_keV']} keV" if scroll.get("energy_keV") else ""
        shape = source.get("shape_zyx") or scroll.get("shape_l0_zyx")
        extent = " x ".join(str(size) for size in shape) if shape else "shape unknown"
        lines.append(f"  scroll     {scroll['id']}, {pitch}{energy}, volume {extent}")
        if source.get("spec"):
            lines.append(f"             {source['spec']}")
        role = scroll.get("f0_role")
        if role == "train":
            lines.append("             WARNING: this scroll is in F0's training corpus")
        elif role == "val":
            lines.append("             WARNING: this scroll is F0's validation data")
        else:
            lines.append("             not in F0's training corpus")
    elif plan["mode"] == "volume":
        lines.append(f"  volume     {source.get('spec', '')}")
        shape = source.get("shape_zyx")
        if shape:
            lines.append(
                f"             {' x '.join(str(size) for size in shape)}, "
                f"{source.get('dtype')}, {source.get('access')}"
            )

    region = plan["region"]
    note = "expanded outward to the 128-voxel lattice" if plan["region_expanded_to_lattice"] else "already cube aligned"
    lines.append(
        f"  region     z {region[0]}:{region[1]}, y {region[2]}:{region[3]}, "
        f"x {region[4]}:{region[5]}  ({note})"
    )
    if plan["region_expanded_to_lattice"] and plan.get("region_requested"):
        asked = plan["region_requested"]
        lines.append(
            f"             requested z {asked[0]}:{asked[1]}, y {asked[2]}:{asked[3]}, "
            f"x {asked[4]}:{asked[5]}"
        )

    skipped = plan["skipped_target_cube_ids"]
    lines.append(
        f"  targets    {len(plan['target_cube_ids'])} cubes predicted, {len(skipped)} skipped"
    )
    for item in skipped[:3]:
        lines.append(f"             skipped {item['cube_id']}: {item['reason']}")
    if len(skipped) > 3:
        lines.append(f"             ... and {len(skipped) - 3} more")

    verb = "to fetch" if plan["mode"] == "volume" else f"to {plan['raw_transfer']}"
    lines.append(
        f"  raw        {len(plan['carve_cube_ids'])} cubes {verb}, "
        f"{_mib(plan['estimates']['raw_bytes'])}"
        + (f", {source.get('access')}" if plan["mode"] == "volume" else "")
    )
    lines.append(
        f"  model      C3-F0 {plan['checkpoint'].get('samples')}, T={plan['threshold']:g} with "
        f"{pins['test_time_augmentation']} TTA, halo {pins['halo_voxels']}, "
        f"{pins['amp_dtype']}, {plan['device']}"
    )
    lines.append(
        f"             checkpoint {plan['checkpoint']['sha256'][:8]}... verified "
        f"({plan['checkpoint']['bytes']:,} bytes)"
    )
    pitch = plan.get("voxel_size_um")
    if pitch:
        lines.append(f"  pitch      {pitch:g} um ({plan['voxel_size_um_source']})")
    else:
        lines.append("  pitch      unknown; socratic-repair will need --voxel-size-um")
    if plan.get("umbilicus_yx"):
        lines.append(
            f"  umbilicus  y {plan['umbilicus_yx'][0]:g}, x {plan['umbilicus_yx'][1]:g} "
            f"({plan['umbilicus_source']})"
        )
    else:
        lines.append("  umbilicus  unknown; socratic-repair will need --umbilicus")

    zarr_info = plan["zarr"]
    lines.append(f"  output     {plan['output']}")
    lines.append(
        f"               {DELIVERABLE}   OME-NGFF, {zarr_info['levels']} levels, "
        "global coordinates  <- attach this to a viewer"
    )
    lines.append(
        "               cubes_PRED/       ScrollFiesta grid for grid_pipeline and socratic-repair"
    )
    if plan["threshold_role"] == "shipped-operating-point":
        lines.append(
            "  note       socratic-repair, socratic-continuity and socratic-sheet-patches are"
        )
        lines.append(
            f"             qualified at T={pins['postprocessor_threshold']:g} only; rerun with "
            f"--threshold {pins['postprocessor_threshold']:g} for them."
        )
    else:
        lines.append(
            f"  note       T={plan['threshold']:g} is the postprocessor pin, not the shipped "
            f"operating point (T={pins['operating_threshold']:g})."
        )
    lines.append("")
    lines.append("Nothing written. Add --run to execute.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


def preflight_device(plan: dict[str, Any]) -> dict[str, Any]:
    """Prove the GPU is usable before anything is downloaded or written."""

    device = plan["device"]
    if device.split(":")[0] != "cuda":
        return {"device": device, "power_limit_watts": None}
    import torch

    from crossres_pred.voxel.resources import assert_cuda_power_limit

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested by the recipe but torch.cuda.is_available() is False in "
            f"{sys.executable}; install the CUDA torch wheel recorded in "
            "recipes/f0/environment.lock.txt, or run from an environment that has one"
        )
    handle = torch.device(device)
    limit = assert_cuda_power_limit(handle)
    index = handle.index if handle.index is not None else 0
    return {
        "device": device,
        "name": torch.cuda.get_device_properties(index).name,
        "power_limit_watts": limit,
    }


def _write_cube(path: Path, cube: np.ndarray, chunk: int) -> int:
    import tifffile

    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    tifffile.imwrite(
        temporary,
        np.ascontiguousarray(cube),
        photometric="minisblack",
        compression=None,
        rowsperstrip=chunk,
    )
    os.replace(temporary, path)
    return path.stat().st_size


def read_cube_tif(path: Path, chunk: int) -> np.ndarray:
    import tifffile

    value = np.asarray(tifffile.imread(path))
    if value.shape != (chunk,) * 3:
        raise ValueError(f"{path}: expected {(chunk,) * 3}, got {value.shape}")
    return value


def carve_raw_cubes(
    volume: Any,
    plan: dict[str, Any],
    destination: Path,
    *,
    cache: Path | None = None,
    progress: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Stream the region's cubes out of the volume, one aligned slab at a time."""

    chunk = plan["pins"]["chunk_size"]
    destination.mkdir(parents=True, exist_ok=True)
    wanted = {tuple(parse_cube_id(cube_id)) for cube_id in plan["carve_cube_ids"]}
    records: list[dict[str, Any]] = []
    done = 0
    started = time.time()
    for band in plan["bands"]:
        z = band["z"]
        y_low, y_high = band["y_range"]
        x_low, x_high = band["x_range"]
        pending = [
            tuple(origin)
            for origin in (tuple(value) for value in band["cubes"])
            if tuple(origin) in wanted
        ]
        fresh = [
            origin
            for origin in pending
            if not (destination / f"{format_cube_id(origin)}.tif").is_file()
        ]
        for origin in pending:
            if origin in fresh:
                continue
            path = destination / f"{format_cube_id(origin)}.tif"
            cube = read_cube_tif(path, chunk)
            records.append(
                {
                    "cube_id": format_cube_id(origin),
                    "bytes": path.stat().st_size,
                    "sha256": repair.sha256(path),
                    "transfer": "reused",
                    "all_zero": not bool(cube.any()),
                }
            )
            done += 1
        if not fresh:
            continue
        read_started = time.time()
        slab = np.asarray(volume[z : z + chunk, y_low:y_high, x_low:x_high])
        if slab.dtype != np.uint8:
            raise ValueError(
                f"volume dtype {slab.dtype} is not uint8; F0 expects the public "
                "masked-volume scale and will not rescale silently"
            )
        read_seconds = time.time() - read_started
        for origin in fresh:
            cube = np.ascontiguousarray(
                slab[
                    :,
                    origin[1] - y_low : origin[1] - y_low + chunk,
                    origin[2] - x_low : origin[2] - x_low + chunk,
                ]
            )
            path = destination / f"{format_cube_id(origin)}.tif"
            size = _write_cube(path, cube, chunk)
            records.append(
                {
                    "cube_id": format_cube_id(origin),
                    "bytes": size,
                    "sha256": repair.sha256(path),
                    "transfer": "carved",
                    "all_zero": not bool(cube.any()),
                }
            )
            done += 1
        progress(
            f"carve z{z:05d}: {len(fresh)} cubes, {_mib(int(band['bytes']))} in "
            f"{read_seconds:.1f} s ({done}/{len(wanted)})"
        )
    zeros = sum(1 for record in records if record["all_zero"])
    progress(
        f"carve complete: {len(records)} cubes, "
        f"{_mib(sum(record['bytes'] for record in records))}, "
        f"{time.time() - started:.1f} s, {zeros} all-zero cubes"
    )
    targets = set(plan["target_cube_ids"])
    empty_targets = [r["cube_id"] for r in records if r["all_zero"] and r["cube_id"] in targets]
    if empty_targets:
        progress(
            f"warning: {len(empty_targets)} target cubes are entirely zero; the store "
            "cannot distinguish masked space from an object that was never uploaded"
        )
    records.sort(key=lambda record: record["cube_id"])
    return records


def materialize_raw_cubes(
    input_grid: Path,
    cube_ids: Sequence[str],
    destination: Path,
    *,
    copy: bool = False,
    progress: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Bring an existing grid's raw cubes into the output, by link where possible."""

    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    fell_back = 0
    for cube_id in cube_ids:
        source = input_grid / "cubes_RAW" / f"{cube_id}.tif"
        target = destination / f"{cube_id}.tif"
        transfer = "reused"
        if not target.exists():
            transfer = "copy" if copy else "hardlink"
            if copy:
                shutil.copy2(source, target)
            else:
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
                    transfer = "copy"
                    fell_back += 1
        records.append(
            {
                "cube_id": cube_id,
                "bytes": target.stat().st_size,
                "sha256": repair.sha256(target),
                "transfer": transfer,
            }
        )
    verb = "copied" if copy else "hard-linked"
    detail = f" ({fell_back} copied: hard link unavailable)" if fell_back else ""
    progress(f"raw cubes: {len(records)} {verb}{detail}")
    return records


def _grid_manifest(plan: dict[str, Any]) -> dict[str, Any]:
    chunk = plan["pins"]["chunk_size"]
    region = plan["region"]
    if plan["mode"] == "grid":
        manifest = dict(plan["source"]["manifest"])
        if manifest.get("created_by"):
            manifest["source_manifest_created_by"] = manifest["created_by"]
        manifest["created_by"] = "socratic-predict"
    else:
        carve = [parse_cube_id(cube_id) for cube_id in plan["carve_cube_ids"]]
        lows = [min(origin[axis] for origin in carve) for axis in range(3)]
        highs = [max(origin[axis] for origin in carve) + chunk for axis in range(3)]
        manifest = {
            "chunk_size": chunk,
            "bbox_l0_zyx": list(region),
            "raw_bbox_l0_zyx": [value for pair in zip(lows, highs, strict=True) for value in pair],
            "n_chunks": [(region[2 * axis + 1] - region[2 * axis]) // chunk for axis in range(3)],
            "sources": {"pred": None, "raw": plan["source"].get("spec")},
            "created_by": "socratic-predict",
        }
    manifest["umbilicus_yx"] = plan["umbilicus_yx"]
    manifest["voxel_size_um"] = plan["voxel_size_um"]
    if plan.get("volume_shape_zyx"):
        manifest["volume_shape_zyx"] = plan["volume_shape_zyx"]
    manifest["predicted_with"] = {
        "recipe_version": plan["recipe"]["version"],
        "threshold": plan["threshold"],
        "checkpoint_sha256": plan["checkpoint"]["sha256"],
    }
    return manifest


def _deliverable_metadata(plan: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    pins = plan["pins"]
    return {
        "kind": "surface-inference",
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "model": {
            "name": "C3-F0",
            "recipe_version": plan["recipe"]["version"],
            "release_id": plan["recipe"]["release_id"],
            "checkpoint_sha256": plan["checkpoint"]["sha256"],
            "checkpoint_samples": plan["checkpoint"].get("samples"),
            "composition": "raw-student-only-no-m7-blend-no-teacher",
            "tta_type": pins["test_time_augmentation"],
            "disable_tta": not pins["mirror_tta"],
            "halo_voxels": pins["halo_voxels"],
            "context_shape_zyx": pins["context_shape_zyx"],
            "amp_dtype": pins["amp_dtype"],
        },
        "source": {
            "volume_path": plan["source"].get("spec") or plan["source"].get("input_grid"),
            "input_anon": plan["source"].get("access") == "s3-anonymous",
            "scroll": (plan.get("scroll") or {}).get("id"),
            "region_requested_zyx": plan.get("region_requested"),
            "region_zyx": plan["region"],
            "voxel_size_um": plan["voxel_size_um"],
            "umbilicus_yx": plan["umbilicus_yx"],
        },
        "output": {
            "ome_num_levels": plan["levels"],
            "ome_downsampling_algorithm": "nearest",
            "chunks_zyx": plan["zarr"]["chunks_zyx"],
            "chunk_note": (
                "128-voxel chunks so one prediction cube is exactly one chunk; the published "
                "M7 surface stores use 192"
            ),
            "shape_source": plan["zarr"]["shape_source"],
        },
        "post_processing": {
            "threshold": True,
            "threshold_value": plan["threshold"],
            "threshold_role": plan["threshold_role"],
        },
        "provenance": {
            "engine": provenance.get("schema"),
            "target_cube_ids": provenance.get("target_cube_ids"),
            "research_only": True,
            "deployment_ready": False,
        },
    }


def run_predict(
    plan: dict[str, Any], *, progress: Callable[[str], None] = print
) -> Path:
    """Execute a checked plan. The receipt is the last thing written."""

    output = Path(plan["output"])
    lock = Path(plan["lock"])
    engine_directory = Path(plan["engine_directory"])
    chunk = plan["pins"]["chunk_size"]
    timings: dict[str, float] = {}
    started = time.time()

    if output.exists() and not plan["resume"]:
        raise FileExistsError(f"output exists: {output}; outputs are never overwritten")
    stale = sorted(output.parent.glob(output.name + f"/{ENGINE_DIR}.partial-*"))
    if stale:
        raise FileExistsError(
            f"engine partial output exists: {stale[0]}; it requires inspection"
        )

    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "pid": os.getpid(),
                "output": str(output),
                "started_utc": datetime.now(UTC).isoformat(),
            },
            stream,
        )
    try:
        gpu = preflight_device(plan)
        if gpu.get("power_limit_watts") is not None:
            progress(
                f"gpu {gpu['device']} {gpu.get('name')}, power limit "
                f"{gpu['power_limit_watts']:g} W"
            )

        output.mkdir(parents=True, exist_ok=plan["resume"])
        plan_path = output / PLAN_FILE
        if plan_path.is_file():
            previous = repair.read_object(plan_path)
            if _comparable(previous) != _comparable(plan):
                raise ValueError(
                    "resume plan differs from the recorded one; move the directory aside "
                    "and start again"
                )
        else:
            hf_export._atomic_json(plan_path, plan)
        hf_export._atomic_json(output / "manifest.json", _grid_manifest(plan))

        raw_directory = output / "cubes_RAW"
        carve_started = time.time()
        if plan["mode"] == "volume":
            volume, _ = open_ct_volume(plan["source"]["spec"])
            raw_records = carve_raw_cubes(volume, plan, raw_directory, progress=progress)
        else:
            raw_records = materialize_raw_cubes(
                Path(plan["source"]["input_grid"]),
                plan["carve_cube_ids"],
                raw_directory,
                copy=plan["copy_raw"],
                progress=progress,
            )
        timings["carve"] = time.time() - carve_started
        (raw_directory / RAW_INVENTORY).write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in raw_records),
            encoding="utf-8",
        )

        prediction_directory = output / "cubes_PRED"
        prediction_directory.mkdir(exist_ok=True)
        (prediction_directory / "present.json").write_text(
            json.dumps(sorted(plan["target_cube_ids"]), indent=2) + "\n", encoding="utf-8"
        )

        inference_started = time.time()
        provenance_path = engine_directory / "provenance.json"
        if not provenance_path.is_file():
            from crossres_pred.voxel.grid_inference import infer_voxel_grid

            progress(
                f"inference: {len(plan['target_cube_ids'])} cubes at T={plan['threshold']:g}"
            )
            infer_voxel_grid(
                source_grid=output,
                checkpoint_path=plan["checkpoint"]["path"],
                output_path=engine_directory,
                threshold=plan["threshold"],
                halo=plan["pins"]["halo_voxels"],
                device_name=plan["device"],
                amp_dtype_name=plan["pins"]["amp_dtype"],
                mirror_tta=plan["pins"]["mirror_tta"],
                max_cpu_threads=plan["max_cpu_threads"],
                target_cube_ids=None,
                skip_incomplete_context=False,
            )
        else:
            progress("inference: reusing the completed engine output")
        timings["inference"] = time.time() - inference_started

        provenance = repair.read_object(provenance_path)
        _verify_provenance(provenance, plan)

        assemble_started = time.time()
        _assemble(output, engine_directory, plan)
        timings["assemble"] = time.time() - assemble_started

        verify_started = time.time()
        outputs = _verify_outputs(output, plan)
        repair.check_identity(
            {
                "path": plan["checkpoint"]["path"],
                "bytes": plan["checkpoint"]["bytes"],
                "sha256": plan["checkpoint"]["sha256"],
            }
        )
        timings["verify"] = time.time() - verify_started

        deliverable_started = time.time()
        summary = ome_zarr.write_surface_pyramid(
            output / DELIVERABLE,
            cube_ids=sorted(plan["target_cube_ids"]),
            read_cube=lambda cube_id: read_cube_tif(
                prediction_directory / f"{cube_id}.tif", chunk
            ),
            shape_zyx=plan["zarr"]["shape_zyx"],
            chunk=plan["zarr"]["chunks_zyx"][0],
            levels=plan["levels"],
            metadata=_deliverable_metadata(plan, provenance),
        )
        summary["validation"] = ome_zarr.validate_ome_ngff(output / DELIVERABLE)
        timings["deliverable"] = time.time() - deliverable_started
        timings["total"] = time.time() - started

        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "complete",
            "profile": PROFILE,
            "completed_utc": datetime.now(UTC).isoformat(),
            "plan": plan,
            "gpu": gpu,
            "engine": {
                "module": "crossres_pred.voxel.grid_inference",
                "function": "infer_voxel_grid",
                "provenance_schema": provenance.get("schema"),
                "provenance": {
                    key: provenance.get(key)
                    for key in (
                        "threshold",
                        "halo",
                        "chunk_size",
                        "context_shape_zyx",
                        "mirror_tta",
                        "amp_dtype",
                        "device",
                        "checkpoint_sha256",
                        "source_grid",
                    )
                },
            },
            "raw_cubes": {
                "count": len(raw_records),
                "all_zero": [r["cube_id"] for r in raw_records if r.get("all_zero")],
                "materialized": {
                    key: sum(1 for r in raw_records if r["transfer"] == key)
                    for key in ("carved", "hardlink", "copy", "reused")
                },
                "files": raw_records,
            },
            "deliverable": summary,
            "output_files": outputs,
            "timings_seconds": {key: round(value, 3) for key, value in timings.items()},
            "threshold_note": plan["threshold_note"],
            "research_only": True,
            "automatic_deployment": False,
        }
        hf_export._atomic_json(output / RECEIPT_FILE, receipt)
    finally:
        lock.unlink(missing_ok=True)

    progress(f"published {output}")
    return output


def _comparable(plan: dict[str, Any]) -> dict[str, Any]:
    ignored = {"created_at_utc", "resume", "engine_pins"}
    return {key: value for key, value in plan.items() if key not in ignored}


def _verify_provenance(provenance: dict[str, Any], plan: dict[str, Any]) -> None:
    expected = {
        "threshold": plan["threshold"],
        "halo": plan["pins"]["halo_voxels"],
        "chunk_size": plan["pins"]["chunk_size"],
        "mirror_tta": plan["pins"]["mirror_tta"],
        "amp_dtype": plan["pins"]["amp_dtype"],
        "checkpoint_sha256": plan["checkpoint"]["sha256"],
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(
                f"engine provenance {key} is {provenance.get(key)!r}, expected {value!r}"
            )
    if sorted(provenance.get("target_cube_ids", [])) != sorted(plan["target_cube_ids"]):
        raise ValueError("engine provenance target inventory disagrees with the plan")


def _assemble(output: Path, engine_directory: Path, plan: dict[str, Any]) -> None:
    """Lift the engine's results beside the raw grid they were predicted from."""

    manifest = repair.read_object(output / "manifest.json")
    copied = engine_directory / "source_manifest.json"
    if copied.is_file() and repair.read_object(copied) != manifest:
        raise ValueError("the engine's copy of the grid manifest differs from ours")
    for name in ("cubes_PRED", "probability"):
        source = engine_directory / name
        if not source.is_dir():
            continue
        destination = output / name
        destination.mkdir(exist_ok=True)
        for item in sorted(source.iterdir()):
            os.replace(item, destination / item.name)
        source.rmdir()
    for name in ("provenance.json", "source_manifest.json"):
        item = engine_directory / name
        if item.is_file():
            os.replace(item, output / name)
    if engine_directory.is_dir():
        remaining = sorted(engine_directory.iterdir())
        if remaining:
            raise ValueError(f"engine output has unexpected leftovers: {remaining}")
        engine_directory.rmdir()


def _verify_outputs(output: Path, plan: dict[str, Any]) -> dict[str, Any]:
    chunk = plan["pins"]["chunk_size"]
    predictions: list[dict[str, Any]] = []
    probabilities: list[dict[str, Any]] = []
    for cube_id in sorted(plan["target_cube_ids"]):
        path = output / "cubes_PRED" / f"{cube_id}.tif"
        cube = read_cube_tif(path, chunk)
        if cube.dtype != np.uint8 or not np.isin(cube, (0, 255)).all():
            raise ValueError(f"{path}: prediction cubes must be uint8 with values 0 or 255")
        identity = repair.file_identity(path)
        identity["path"] = f"cubes_PRED/{cube_id}.tif"
        predictions.append(identity)
        probability_path = output / "probability" / f"{cube_id}.tif"
        if probability_path.is_file():
            values = read_cube_tif(probability_path, chunk)
            if values.dtype != np.float16 or not np.isfinite(values.astype(np.float32)).all():
                raise ValueError(f"{probability_path}: expected finite float16 probabilities")
            identity = repair.file_identity(probability_path)
            identity["path"] = f"probability/{cube_id}.tif"
            probabilities.append(identity)
    missing = [
        cube_id
        for cube_id in plan["carve_cube_ids"]
        if not (output / "cubes_RAW" / f"{cube_id}.tif").is_file()
    ]
    if missing:
        raise ValueError(f"raw context cubes are missing after the run: {missing[:3]}")
    files = {"cubes_PRED": predictions, "probability": probabilities}
    for name in ("manifest.json", "source_manifest.json", "provenance.json"):
        path = output / name
        if path.is_file():
            identity = repair.file_identity(path)
            identity["path"] = name
            files[name] = identity
    return files


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="socratic-predict",
        description=(
            "Run the frozen C3-F0 model on a scroll region with the shipped inference "
            "recipe and write one self-contained grid plus an OME-NGFF prediction store."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scroll", help="scroll name resolved through recipes/f0/scrolls.json")
    source.add_argument("--volume", help="CT volume: local .zarr/.npy/.tif, or an s3:// OME-Zarr")
    source.add_argument("--grid", type=Path, help="existing carved grid directory")

    region = parser.add_mutually_exclusive_group()
    region.add_argument("--region", help="z0:z1,y0:y1,x0:x1 in level-0 voxels, half-open")
    region.add_argument(
        "--bbox",
        nargs=6,
        type=int,
        metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"),
        help="the same region as six integers",
    )

    parser.add_argument("--out", "--output", dest="output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, help="the shipped operating point by default")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-hf", help="Hugging Face repository holding the checkpoint")
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--scrolls", type=Path, help="scroll registry JSON")
    parser.add_argument("--umbilicus", nargs=2, type=float, metavar=("Y", "X"))
    parser.add_argument("--voxel-size-um", type=float)
    parser.add_argument("--cache", type=Path, help="reuse downloads across runs")
    parser.add_argument("--device")
    parser.add_argument("--max-cpu-threads", type=int)
    parser.add_argument("--levels", type=int, help="pyramid levels in the deliverable")
    parser.add_argument("--copy-raw", action="store_true", help="copy raw cubes instead of linking")
    parser.add_argument(
        "--s3-anon",
        choices=("auto", "yes", "no"),
        default="auto",
        help="anonymous S3 access; auto signs only when AWS credentials are set",
    )
    parser.add_argument("--check", action="store_true", help="validate and print the plan")
    parser.add_argument("--json", action="store_true", help="print the plan as JSON")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--run", action="store_true", help="execute the checked plan")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--passes", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.passes is not None:
        parser.error(
            "--passes is not supported: F0 is a single-pass model and this recipe has no "
            "pass-2 refinement"
        )
    if args.resume and not args.run:
        parser.error("--resume only applies with --run")
    if args.check and args.run:
        parser.error("--check and --run are mutually exclusive")

    region = None
    try:
        if args.region:
            region = parse_region(args.region)
        elif args.bbox:
            region = validate_region(args.bbox)

        plan = plan_predict(
            output=args.output,
            scroll=args.scroll,
            volume=args.volume,
            grid=args.grid,
            region=region,
            recipe_path=args.recipe,
            paths_file=args.paths,
            registry_path=args.scrolls,
            checkpoint=args.checkpoint,
            model_hf=args.model_hf,
            threshold=args.threshold,
            umbilicus=args.umbilicus,
            voxel_size_um=args.voxel_size_um,
            device=args.device,
            max_cpu_threads=args.max_cpu_threads,
            levels=args.levels,
            copy_raw=args.copy_raw,
            cache=args.cache,
            anon={"auto": None, "yes": True, "no": False}[args.s3_anon],
            resume=args.resume,
        )
        if args.print_command:
            print(" ".join(plan["equivalent_command"]))
        if not args.run:
            print(json.dumps(plan, indent=2) if args.json else render_plan(plan))
            return 0
        print(
            f"plan checked: {len(plan['target_cube_ids'])} targets, "
            f"{len(plan['carve_cube_ids'])} raw cubes, T={plan['threshold']:g} "
            f"({plan['threshold_role'].replace('-', ' ')}), checkpoint "
            f"{plan['checkpoint']['sha256'][:8]}... ok"
        )
        run_predict(plan, progress=lambda line: print(line, flush=True))
        if args.json:
            print(json.dumps(repair.read_object(Path(plan["output"]) / RECEIPT_FILE), indent=2))
        return 0
    except KeyboardInterrupt:
        print(
            "socratic-predict: interrupted; carved cubes are kept, rerun with --resume",
            file=sys.stderr,
        )
        return 130
    except (ValueError, TypeError, FileNotFoundError, FileExistsError, RuntimeError, OSError) as error:
        print(f"socratic-predict: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
