from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import tifffile
import torch
from torch.nn import functional as F

from .carve import load_carved_chunk_ids
from .dataset import normalize_ct_m7
from .model import load_surface_checkpoint
from .provenance import (
    environment_identity,
    require_fresh_directory,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from .volume import open_volume, read_crop

_CUBE_ID = re.compile(r"^z(?P<z>\d+)_y(?P<y>\d+)_x(?P<x>\d+)$")


class InferenceError(RuntimeError):
    """Raised when a source grid cannot satisfy the inference contract."""


@dataclass(frozen=True)
class InferOptions:
    """Options for the drop-in surface prediction over a cube grid.

    The model predicts from raw CT (plus the source baseline prediction for a
    2-channel checkpoint) and emits hard 0/255 cubes. There are no edit
    gates, no calibration branches, and no safety fuse: this is a full
    replacement predictor, and the m7 baseline remains available as a
    config-level fallback producer, not an in-model gate.
    """

    halo: int = 32
    device: str = "cuda"
    amp: bool = True
    amp_dtype: str = "auto"
    threshold: float | None = None
    save_prob: bool = False
    raw_mode: str = "hardlink"

    def validate(self) -> None:
        if self.halo < 0:
            raise InferenceError("halo must be non-negative")
        if self.threshold is not None and (
            not np.isfinite(self.threshold) or not 0.0 < self.threshold < 1.0
        ):
            raise InferenceError("threshold must be in (0, 1)")
        if self.raw_mode not in {"hardlink", "copy", "none"}:
            raise InferenceError("raw_mode must be hardlink, copy, or none")
        if self.amp_dtype not in {"auto", "float16", "bfloat16"}:
            raise InferenceError("amp_dtype must be auto, float16, or bfloat16")


@dataclass(frozen=True)
class TeacherInferOptions:
    """Sliding-window inference over site-confined fine-volume carves.

    The v1 contract deliberately pins a 2x-overlapped tiling: 256^3 model
    patches at stride 128 over the native 128^3 Zarr chunks. Every emitted
    chunk is blended from the available globally aligned tiles with a
    Gaussian importance map. ``retained_margin`` drops chunks too close to a
    site's carve boundary before the bridge sees them.
    """

    patch_shape_zyx: tuple[int, int, int] = (256, 256, 256)
    stride: int = 128
    retained_margin: int = 32
    batch_size: int = 1
    device: str = "cuda"
    amp: bool = True
    amp_dtype: str = "auto"
    array_key: str = "0"

    def validate(self) -> None:
        if len(self.patch_shape_zyx) != 3 or any(
            size < 64 or size % 32 for size in self.patch_shape_zyx
        ):
            raise InferenceError(
                "teacher patch dimensions must be multiples of 32 and at least 64"
            )
        if self.stride <= 0:
            raise InferenceError("teacher stride must be positive")
        if any(size != 2 * self.stride for size in self.patch_shape_zyx):
            raise InferenceError(
                "teacher inference currently requires patch_shape == 2 * stride"
            )
        if self.retained_margin < 0:
            raise InferenceError("retained_margin must be non-negative")
        if self.batch_size <= 0:
            raise InferenceError("batch_size must be positive")
        if self.amp_dtype not in {"auto", "float16", "bfloat16"}:
            raise InferenceError("amp_dtype must be auto, float16, or bfloat16")
        if not self.array_key:
            raise InferenceError("array_key cannot be empty")


def parse_cube_id(cube_id: str) -> tuple[int, int, int]:
    match = _CUBE_ID.fullmatch(cube_id)
    if match is None:
        raise InferenceError(
            f"invalid cube id {cube_id!r}; expected z#####_y#####_x#####"
        )
    return tuple(int(match.group(axis)) for axis in ("z", "y", "x"))


def format_cube_id(origin_zyx: tuple[int, int, int]) -> str:
    z, y, x = origin_zyx
    if min(origin_zyx) < 0:
        raise InferenceError(f"cube origin cannot be negative: {origin_zyx}")
    return f"z{z:05d}_y{y:05d}_x{x:05d}"


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise InferenceError(f"missing required file: {path}") from error
    except json.JSONDecodeError as error:
        raise InferenceError(f"{path}: invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise InferenceError(f"{path}: expected a JSON object")
    return value


def _scan_cube_paths(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise InferenceError(f"missing cube directory: {directory}")
    paths: dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        if _CUBE_ID.fullmatch(path.stem) is None:
            continue
        if path.stem in paths:
            raise InferenceError(f"duplicate TIFFs for cube {path.stem} in {directory}")
        paths[path.stem] = path.resolve()
    return paths


def _load_present_ids(directory: Path) -> list[str]:
    present_path = directory / "present.json"
    if present_path.exists():
        try:
            value = json.loads(present_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise InferenceError(
                f"{present_path}: invalid JSON: {error.msg}"
            ) from error
        # Carved grids write {"emitted": [...], "skipped": [...]}; this
        # tool and older subsets write a bare array.
        if isinstance(value, dict) and isinstance(value.get("emitted"), list):
            value = value["emitted"]
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise InferenceError(f"{present_path}: expected an array of cube ids")
        ids = list(value)
    else:
        ids = sorted(_scan_cube_paths(directory))
    if not ids:
        raise InferenceError(f"{directory}: no cubes")
    if len(ids) != len(set(ids)):
        raise InferenceError(f"{present_path}: duplicate cube ids")
    for cube_id in ids:
        parse_cube_id(cube_id)
    return sorted(ids)


class _CubeReader:
    def __init__(self, paths: dict[str, Path], chunk_size: int, kind: str) -> None:
        self.paths = paths
        self.chunk_size = chunk_size
        self.kind = kind
        self._cache: OrderedDict[tuple[int, int, int], np.ndarray | None] = (
            OrderedDict()
        )

    def _remember(
        self, origin_zyx: tuple[int, int, int], value: np.ndarray | None
    ) -> np.ndarray | None:
        self._cache[origin_zyx] = value
        if len(self._cache) > 64:
            self._cache.popitem(last=False)
        return value

    def read(self, origin_zyx: tuple[int, int, int]) -> np.ndarray | None:
        if origin_zyx in self._cache:
            value = self._cache.pop(origin_zyx)
            self._cache[origin_zyx] = value
            return value
        path = self.paths.get(format_cube_id(origin_zyx))
        if path is None:
            return self._remember(origin_zyx, None)
        try:
            array = np.asarray(tifffile.imread(path))
        except (OSError, ValueError) as error:
            raise InferenceError(
                f"failed to read {self.kind} cube {path}: {error}"
            ) from error
        expected = (self.chunk_size,) * 3
        if tuple(array.shape) != expected:
            raise InferenceError(
                f"{path}: expected {self.kind} cube shape {expected}, got {array.shape}"
            )
        return self._remember(origin_zyx, np.ascontiguousarray(array))


def _assemble_halo(
    reader: _CubeReader,
    center_origin: tuple[int, int, int],
    halo: int,
) -> tuple[np.ndarray, int]:
    center = reader.read(center_origin)
    if center is None:
        cube_id = format_cube_id(center_origin)
        raise InferenceError(f"missing central {reader.kind} cube {cube_id}")
    chunk = reader.chunk_size
    shape = (chunk + 2 * halo,) * 3
    assembled = np.zeros(shape, dtype=center.dtype)
    request_start = tuple(value - halo for value in center_origin)
    request_end = tuple(value + chunk + halo for value in center_origin)
    radius = (halo + chunk - 1) // chunk
    missing = 0

    for offset in product(range(-radius, radius + 1), repeat=3):
        neighbor_origin = tuple(
            center_origin[axis] + offset[axis] * chunk for axis in range(3)
        )
        if min(neighbor_origin) < 0:
            missing += 1
            continue
        neighbor = reader.read(neighbor_origin)
        if neighbor is None:
            missing += 1
            continue
        source_slices: list[slice] = []
        destination_slices: list[slice] = []
        intersects = True
        for axis in range(3):
            lower = max(request_start[axis], neighbor_origin[axis])
            upper = min(request_end[axis], neighbor_origin[axis] + chunk)
            if upper <= lower:
                intersects = False
                break
            source_slices.append(
                slice(lower - neighbor_origin[axis], upper - neighbor_origin[axis])
            )
            destination_slices.append(
                slice(lower - request_start[axis], upper - request_start[axis])
            )
        if intersects:
            assembled[tuple(destination_slices)] = neighbor[tuple(source_slices)]
    return assembled, missing


def _torch_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise InferenceError(
            "CUDA was requested, but torch.cuda.is_available() is false"
        )
    return device


@torch.no_grad()
def _predict_probability(
    model: torch.nn.Module,
    value: np.ndarray,
    device: torch.device,
    amp_dtype: torch.dtype,
    autocast_enabled: bool,
    divisor: int,
) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(value))[None].to(
        device, non_blocking=device.type == "cuda"
    )
    spatial_shape = tuple(int(item) for item in tensor.shape[-3:])
    pad_zyx = tuple((-size) % divisor for size in spatial_shape)
    if any(pad_zyx):
        tensor = F.pad(
            tensor,
            (0, pad_zyx[2], 0, pad_zyx[1], 0, pad_zyx[0]),
        )
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=autocast_enabled,
    ):
        logits = model(tensor)
    probability = torch.sigmoid(logits.float())
    crop = tuple(slice(0, size) for size in spatial_shape)
    return probability[(0, 0) + crop].cpu().numpy()


@torch.no_grad()
def _predict_probability_batch(
    model: torch.nn.Module,
    values: np.ndarray,
    device: torch.device,
    amp_dtype: torch.dtype,
    autocast_enabled: bool,
    divisor: int,
) -> np.ndarray:
    """Predict a ``B,C,Z,Y,X`` batch and return ``B,Z,Y,X`` probabilities."""

    if values.ndim != 5:
        raise InferenceError(
            f"teacher inference expects B,C,Z,Y,X input, got {values.shape}"
        )
    tensor = torch.from_numpy(np.ascontiguousarray(values)).to(
        device, non_blocking=device.type == "cuda"
    )
    spatial_shape = tuple(int(item) for item in tensor.shape[-3:])
    pad_zyx = tuple((-size) % divisor for size in spatial_shape)
    if any(pad_zyx):
        tensor = F.pad(tensor, (0, pad_zyx[2], 0, pad_zyx[1], 0, pad_zyx[0]))
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=autocast_enabled,
    ):
        logits = model(tensor)
    probability = torch.sigmoid(logits.float())
    crop = tuple(slice(0, size) for size in spatial_shape)
    return probability[(slice(None), 0) + crop].cpu().numpy()


def _write_tiff_atomic(path: Path, array: np.ndarray) -> None:
    if array.ndim != 3:
        raise InferenceError(
            f"{path}: prediction must be a 3-D z-y-x array, got {array.shape}"
        )
    if array.dtype != np.uint8:
        raise InferenceError(f"{path}: prediction must be uint8, got {array.dtype}")
    values = np.unique(array)
    if not {int(value) for value in values}.issubset({0, 255}):
        raise InferenceError(
            f"{path}: prediction must contain only 0/255, got {values[:16].tolist()}"
        )
    temporary = path.with_name(path.name + ".tmp.tif")
    tifffile.imwrite(
        temporary,
        np.ascontiguousarray(array),
        byteorder="<",
        photometric="minisblack",
        compression=None,
        metadata=None,
        rowsperstrip=array.shape[-2],
    )
    with temporary.open("rb") as stream:
        if stream.read(4) != b"II*\x00":
            raise InferenceError(f"{temporary}: expected little-endian classic TIFF")
    with tifffile.TiffFile(temporary) as tif:
        if len(tif.pages) != array.shape[0]:
            raise InferenceError(
                f"{temporary}: wrote {len(tif.pages)} pages, expected {array.shape[0]}"
            )
        if any(page.shape != array.shape[1:] for page in tif.pages):
            raise InferenceError(f"{temporary}: TIFF page shape verification failed")
        if any(page.dtype != np.dtype(np.uint8) for page in tif.pages):
            raise InferenceError(f"{temporary}: TIFF dtype verification failed")
        if any(page.compression != 1 for page in tif.pages):
            raise InferenceError(f"{temporary}: TIFF must be uncompressed")
        if any(int(page.photometric) != 1 for page in tif.pages):
            raise InferenceError(f"{temporary}: TIFF must be MINISBLACK")
    os.replace(temporary, path)


def _materialize_raw(source: Path, destination: Path, mode: str) -> None:
    if mode == "none":
        return
    if mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError as error:
            raise InferenceError(
                f"cannot hard-link raw cube {source} to {destination}; "
                "use --raw-mode copy when output is on another filesystem"
            ) from error
    else:
        shutil.copy2(source, destination)


def _event(stream: TextIO, event: str, **values: Any) -> None:
    row = {"timestamp": utc_now(), "event": event, **values}
    stream.write(json.dumps(row, sort_keys=True) + "\n")
    stream.flush()


def _resolve_threshold(
    options: InferOptions, checkpoint_payload: dict[str, Any]
) -> tuple[float, str]:
    if options.threshold is not None:
        return float(options.threshold), "cli"
    recorded = checkpoint_payload.get("deploy_threshold")
    if recorded is not None:
        value = float(recorded)
        if not np.isfinite(value) or not 0.0 < value < 1.0:
            raise InferenceError(
                f"checkpoint deploy_threshold is invalid: {recorded!r}"
            )
        return value, "checkpoint"
    return 0.5, "default"


def _binary_dice(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    total = int(np.count_nonzero(first)) + int(np.count_nonzero(second))
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def infer_grid(
    *,
    checkpoint_path: str | Path,
    source_grid: str | Path,
    output_path: str | Path,
    options: InferOptions,
) -> Path:
    """Predict every listed cube from raw CT without modifying the source grid."""

    options.validate()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    source = Path(source_grid).expanduser().resolve()
    if not checkpoint.is_file():
        raise InferenceError(f"checkpoint does not exist: {checkpoint}")
    manifest_path = source / "manifest.json"
    source_manifest = _read_json_object(manifest_path)
    try:
        chunk_size = int(source_manifest["chunk_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise InferenceError(f"{manifest_path}: invalid chunk_size") from error
    if chunk_size <= 0:
        raise InferenceError(f"{manifest_path}: chunk_size must be positive")
    if chunk_size + 2 * options.halo < 64:
        raise InferenceError(
            f"chunk_size {chunk_size} with halo {options.halo} is below the "
            "64-voxel model minimum; raise --halo"
        )

    source_raw_dir = source / "cubes_RAW"
    source_pred_dir = source / "cubes_PRED"
    raw_paths = _scan_cube_paths(source_raw_dir)
    baseline_paths = (
        _scan_cube_paths(source_pred_dir) if source_pred_dir.is_dir() else {}
    )
    if source_pred_dir.is_dir() and (source_pred_dir / "present.json").exists():
        cube_ids = _load_present_ids(source_pred_dir)
    else:
        cube_ids = _load_present_ids(source_raw_dir)
    missing_raw = [cube_id for cube_id in cube_ids if cube_id not in raw_paths]
    if missing_raw:
        raise InferenceError(f"missing raw cube for {missing_raw[0]}")

    device = _torch_device(options.device)
    autocast_enabled = options.amp and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if autocast_enabled
        and (
            options.amp_dtype == "bfloat16"
            or (options.amp_dtype == "auto" and torch.cuda.is_bf16_supported())
        )
        else torch.float16
    )
    model, checkpoint_payload = load_surface_checkpoint(checkpoint, device)
    checkpoint_profile = checkpoint_payload.get("policy_profile")
    if checkpoint_profile not in {"prize-safe", "research"}:
        raise InferenceError(
            "checkpoint is missing a valid prize-safe/research policy profile"
        )
    needs_baseline = model.config.in_channels == 2
    if needs_baseline:
        missing_baseline = [
            cube_id for cube_id in cube_ids if cube_id not in baseline_paths
        ]
        if missing_baseline:
            raise InferenceError(
                f"a 2-channel checkpoint requires baseline cube "
                f"{missing_baseline[0]} in cubes_PRED"
            )
    threshold, threshold_source = _resolve_threshold(options, checkpoint_payload)

    output = require_fresh_directory(output_path)
    output_pred_dir = output / "cubes_PRED"
    output_pred_dir.mkdir()
    output_raw_dir = output / "cubes_RAW"
    if options.raw_mode != "none":
        output_raw_dir.mkdir()
    prob_dir = output / "prob"
    if options.save_prob:
        prob_dir.mkdir()

    provenance: dict[str, Any] = {
        "schema_version": 2,
        "kind": "crossres-grid-inference",
        "status": "running",
        "created_at": utc_now(),
        "source_grid": str(source),
        "source_manifest_sha256": sha256_file(manifest_path),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "profile": checkpoint_payload.get("profile"),
            "epoch": checkpoint_payload.get("epoch"),
            "val_selection": checkpoint_payload.get("val_selection"),
        },
        "policy_profile": checkpoint_profile,
        "options": asdict(options),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "in_channels": model.config.in_channels,
        "cube_count": len(cube_ids),
        "environment": environment_identity(),
    }
    write_json_atomic(output / "provenance.json", provenance)

    raw_reader = _CubeReader(raw_paths, chunk_size, "raw")
    baseline_reader = (
        _CubeReader(baseline_paths, chunk_size, "baseline")
        if baseline_paths
        else None
    )
    totals = {
        "output_foreground_voxels": 0,
        "baseline_foreground_voxels": 0,
        "compared_cubes": 0,
        "baseline_dice_sum": 0.0,
    }
    started = time.perf_counter()
    events_path = output / "events.jsonl"
    try:
        with events_path.open("x", encoding="utf-8", newline="\n") as events:
            _event(
                events,
                "run_started",
                cube_count=len(cube_ids),
                chunk_size=chunk_size,
                threshold=threshold,
            )
            for cube_index, cube_id in enumerate(cube_ids, 1):
                cube_started = time.perf_counter()
                origin = parse_cube_id(cube_id)
                raw_halo, missing_raw_neighbors = _assemble_halo(
                    raw_reader, origin, options.halo
                )
                channels = [normalize_ct_m7(raw_halo)]
                baseline_center: np.ndarray | None = None
                if baseline_reader is not None:
                    baseline_halo, _ = _assemble_halo(
                        baseline_reader, origin, options.halo
                    )
                    center_slice = tuple(
                        slice(options.halo, options.halo + chunk_size)
                        for _ in range(3)
                    )
                    baseline_center = baseline_halo[center_slice] != 0
                    if needs_baseline:
                        channels.append((baseline_halo != 0).astype(np.float32))
                model_input = np.stack(channels, axis=0).astype(
                    np.float32, copy=False
                )
                probability = _predict_probability(
                    model,
                    model_input,
                    device,
                    amp_dtype,
                    autocast_enabled,
                    model.required_divisor,
                )
                center_slice = tuple(
                    slice(options.halo, options.halo + chunk_size) for _ in range(3)
                )
                center_probability = probability[center_slice]
                prediction = (center_probability >= threshold).astype(
                    np.uint8
                ) * np.uint8(255)
                _write_tiff_atomic(output_pred_dir / f"{cube_id}.tif", prediction)
                _materialize_raw(
                    raw_paths[cube_id],
                    output_raw_dir / f"{cube_id}.tif",
                    options.raw_mode,
                )
                if options.save_prob:
                    temporary = prob_dir / f"{cube_id}.tmp.npz"
                    np.savez_compressed(
                        temporary,
                        prob_u8=np.rint(
                            np.clip(center_probability, 0.0, 1.0) * 255.0
                        ).astype(np.uint8),
                    )
                    os.replace(temporary, prob_dir / f"{cube_id}.npz")

                foreground = int(np.count_nonzero(prediction))
                totals["output_foreground_voxels"] += foreground
                comparison: dict[str, Any] = {}
                if baseline_center is not None:
                    baseline_foreground = int(np.count_nonzero(baseline_center))
                    dice = _binary_dice(prediction != 0, baseline_center)
                    totals["baseline_foreground_voxels"] += baseline_foreground
                    totals["baseline_dice_sum"] += dice
                    totals["compared_cubes"] += 1
                    comparison = {
                        "baseline_foreground_voxels": baseline_foreground,
                        "baseline_dice": dice,
                    }
                _event(
                    events,
                    "cube_complete",
                    cube_id=cube_id,
                    cube_index=cube_index,
                    duration_seconds=time.perf_counter() - cube_started,
                    missing_raw_neighbors=missing_raw_neighbors,
                    foreground_voxels=foreground,
                    foreground_fraction=foreground / prediction.size,
                    mean_probability=float(center_probability.mean()),
                    **comparison,
                )

            write_json_atomic(output_pred_dir / "present.json", cube_ids)
            duration = time.perf_counter() - started
            summary = {
                "schema_version": 2,
                "status": "complete",
                "completed_at": utc_now(),
                "cube_count": len(cube_ids),
                "duration_seconds": duration,
                "threshold": threshold,
                "threshold_source": threshold_source,
                "output_foreground_voxels": totals["output_foreground_voxels"],
                "baseline_foreground_voxels": totals["baseline_foreground_voxels"],
                "compared_cubes": totals["compared_cubes"],
                "mean_baseline_dice": (
                    totals["baseline_dice_sum"] / totals["compared_cubes"]
                    if totals["compared_cubes"]
                    else None
                ),
            }
            write_json_atomic(output / "summary.json", summary)
            output_manifest = dict(source_manifest)
            output_manifest["crossres_pred"] = {
                "schema_version": 2,
                "source_grid": str(source),
                "checkpoint_sha256": provenance["checkpoint"]["sha256"],
                "threshold": threshold,
                "raw_mode": options.raw_mode,
                "cube_count": len(cube_ids),
                "policy_profile": checkpoint_profile,
            }
            output_manifest["n_pred_tiffs_emitted"] = len(cube_ids)
            write_json_atomic(output / "manifest.json", output_manifest)
            provenance.update(
                {
                    "status": "complete",
                    "completed_at": summary["completed_at"],
                    "summary": summary,
                }
            )
            write_json_atomic(output / "provenance.json", provenance)
            _event(events, "run_complete", **summary)
    except Exception as error:
        provenance.update(
            {
                "status": "failed",
                "failed_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        write_json_atomic(output / "provenance.json", provenance)
        if events_path.exists():
            with events_path.open("a", encoding="utf-8", newline="\n") as events:
                _event(
                    events,
                    "run_failed",
                    error_type=type(error).__name__,
                    error=str(error),
                )
        raise
    return output


def _gaussian_importance_map(
    shape_zyx: tuple[int, int, int], sigma_scale: float = 0.125
) -> np.ndarray:
    axes = []
    for size in shape_zyx:
        coordinate = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
        sigma = max(1.0, float(size) * sigma_scale)
        axes.append(np.exp(-0.5 * np.square(coordinate / sigma)))
    weight = (
        axes[0][:, None, None]
        * axes[1][None, :, None]
        * axes[2][None, None, :]
    )
    weight /= float(weight.max())
    return np.maximum(weight, np.float32(1.0e-4)).astype(np.float32)


def _target_chunks_for_site(
    row: dict[str, Any],
    chunk_shape_zyx: tuple[int, int, int],
    chunk_grid_zyx: tuple[int, int, int],
    retained_margin: int,
    available: set[tuple[int, int, int]],
) -> set[tuple[int, int, int]]:
    chunk = np.asarray(chunk_shape_zyx, dtype=np.int64)
    lo = np.asarray(row["fine_bbox_lo_zyx"], dtype=np.int64) + retained_margin
    hi = np.asarray(row["fine_bbox_hi_zyx"], dtype=np.int64) - retained_margin
    if np.any(hi <= lo):
        return set()
    first = np.ceil(lo / chunk).astype(np.int64)
    stop = np.floor(hi / chunk).astype(np.int64)
    first = np.maximum(first, 0)
    stop = np.minimum(stop, np.asarray(chunk_grid_zyx, dtype=np.int64))
    if np.any(stop <= first):
        return set()
    return {
        (z, y, x)
        for z in range(int(first[0]), int(stop[0]))
        for y in range(int(first[1]), int(stop[1]))
        for x in range(int(first[2]), int(stop[2]))
        if (z, y, x) in available
    }


def _tile_has_full_coverage(
    tile_zyx: tuple[int, int, int],
    available: set[tuple[int, int, int]],
) -> bool:
    return all(
        tuple(tile_zyx[axis] + offset[axis] for axis in range(3)) in available
        for offset in product((0, 1), repeat=3)
    )


def _linear_chunk_id(
    chunk_zyx: tuple[int, int, int], chunk_grid_zyx: tuple[int, int, int]
) -> int:
    z, y, x = chunk_zyx
    _, grid_y, grid_x = chunk_grid_zyx
    return (z * grid_y + y) * grid_x + x


def infer_teacher(
    *,
    checkpoint_path: str | Path,
    site_rows: list[dict[str, Any]],
    site_manifest_path: str | Path,
    record_id: str,
    mirror_path: str | Path,
    output_path: str | Path,
    policy_profile: str,
    options: TeacherInferOptions,
) -> Path:
    """Infer a fine teacher into a sparse soft-probability Zarr.

    Output array ``0`` is uint8 probability. Array ``1`` is an exact voxel
    validity mask (predicted and raw CT nonzero), which the bridge combines
    with chunk coverage before its filter-support and retained-interior
    erosion. This prevents masked scan voids from becoming distilled
    background supervision.
    """

    options.validate()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    sites_path = Path(site_manifest_path).expanduser().resolve()
    mirror = Path(mirror_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise InferenceError(f"checkpoint does not exist: {checkpoint}")
    if not sites_path.is_file():
        raise InferenceError(f"site manifest does not exist: {sites_path}")
    if not mirror.is_dir():
        raise InferenceError(f"fine mirror does not exist: {mirror}")
    rows = [row for row in site_rows if str(row.get("record_id")) == record_id]
    if not rows:
        raise InferenceError(f"site manifest has no rows for {record_id}")

    raw_volume = open_volume(f"{mirror}::{options.array_key}")
    raw_chunk_shape, raw_chunk_ids = load_carved_chunk_ids(mirror)
    raw_chunks = tuple(int(item) for item in getattr(raw_volume, "chunks", ()))
    if raw_chunks != raw_chunk_shape:
        raise InferenceError(
            f"mirror chunk metadata disagrees: array={raw_chunks}, "
            f"selection={raw_chunk_shape}"
        )
    if raw_chunk_shape != (options.stride,) * 3:
        raise InferenceError(
            f"teacher stride {options.stride} must match mirror chunks "
            f"{raw_chunk_shape}"
        )
    shape_zyx = tuple(int(item) for item in raw_volume.shape)
    chunk_grid_zyx = tuple(
        (shape_zyx[axis] + raw_chunk_shape[axis] - 1)
        // raw_chunk_shape[axis]
        for axis in range(3)
    )

    device = _torch_device(options.device)
    autocast_enabled = options.amp and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if autocast_enabled
        and (
            options.amp_dtype == "bfloat16"
            or (options.amp_dtype == "auto" and torch.cuda.is_bf16_supported())
        )
        else torch.float16
    )
    model, checkpoint_payload = load_surface_checkpoint(checkpoint, device)
    if checkpoint_payload.get("profile") != "teacher":
        raise InferenceError("infer-teacher requires a teacher-profile checkpoint")
    if checkpoint_payload.get("policy_profile") != policy_profile:
        raise InferenceError(
            "checkpoint policy profile does not match the selected data policy"
        )
    if model.config.in_channels != 1:
        raise InferenceError("fine teachers must be 1-channel raw-CT models")

    try:
        import zarr
    except ImportError as error:
        raise InferenceError(
            "infer-teacher requires `pip install vesuvius-crossres-pred[zarr]`"
        ) from error

    output = require_fresh_directory(output_path)
    root = zarr.open_group(str(output), mode="a", zarr_format=2)
    create_options = {
        "shape": shape_zyx,
        "chunks": raw_chunk_shape,
        "dtype": "u1",
        "compressor": None,
        "fill_value": 0,
        "chunk_key_encoding": {"name": "v2", "separator": "/"},
    }
    probability_array = root.create_array("0", **create_options)
    validity_array = root.create_array("1", **create_options)

    source_manifest = mirror / "crossres_sparse_mirror.json"
    source_selection = mirror / "carve_selected_chunks.json"
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crossres-teacher-inference",
        "status": "running",
        "created_at": utc_now(),
        "record_id": record_id,
        "policy_profile": policy_profile,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "epoch": checkpoint_payload.get("epoch"),
            "val_selection": checkpoint_payload.get("val_selection"),
        },
        "sites": {"path": str(sites_path), "sha256": sha256_file(sites_path)},
        "source_mirror": str(mirror),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_selection_sha256": sha256_file(source_selection),
        "options": asdict(options),
        "zarr": {
            "shape_zyx": list(shape_zyx),
            "chunks_zyx": list(raw_chunk_shape),
            "chunk_grid_zyx": list(chunk_grid_zyx),
            "dimension_separator": "/",
            "probability_array_key": "0",
            "validity_array_key": "1",
        },
        "environment": environment_identity(),
    }
    write_json_atomic(output / "provenance.json", provenance)

    gaussian = _gaussian_importance_map(options.patch_shape_zyx)
    predicted_chunks: set[tuple[int, int, int]] = set()
    total_tiles = 0
    total_requested_chunks = 0
    total_valid_voxels = 0
    probability_sum_valid = 0.0
    started = time.perf_counter()
    events_path = output / "events.jsonl"
    try:
        with events_path.open("x", encoding="utf-8", newline="\n") as events:
            _event(events, "run_started", site_count=len(rows))
            for site_index, row in enumerate(rows, 1):
                site_id = str(row["site_id"])
                targets = _target_chunks_for_site(
                    row,
                    raw_chunk_shape,
                    chunk_grid_zyx,
                    options.retained_margin,
                    raw_chunk_ids,
                ) - predicted_chunks
                total_requested_chunks += len(targets)
                tile_ids = {
                    tuple(target[axis] + delta[axis] for axis in range(3))
                    for target in targets
                    for delta in product((-1, 0), repeat=3)
                }
                tiles = sorted(
                    tile
                    for tile in tile_ids
                    if _tile_has_full_coverage(tile, raw_chunk_ids)
                )
                total_tiles += len(tiles)
                accumulators: dict[tuple[int, int, int], np.ndarray] = {}
                weights: dict[tuple[int, int, int], np.ndarray] = {}

                for batch_start in range(0, len(tiles), options.batch_size):
                    batch_tiles = tiles[
                        batch_start : batch_start + options.batch_size
                    ]
                    model_input = np.stack(
                        [
                            normalize_ct_m7(
                                read_crop(
                                    raw_volume,
                                    tuple(value * options.stride for value in tile),
                                    options.patch_shape_zyx,
                                )
                            )[None]
                            for tile in batch_tiles
                        ],
                        axis=0,
                    ).astype(np.float32, copy=False)
                    probabilities = _predict_probability_batch(
                        model,
                        model_input,
                        device,
                        amp_dtype,
                        autocast_enabled,
                        model.required_divisor,
                    )
                    for tile, probability in zip(
                        batch_tiles, probabilities, strict=True
                    ):
                        for offset in product((0, 1), repeat=3):
                            chunk_id = tuple(
                                tile[axis] + offset[axis] for axis in range(3)
                            )
                            if chunk_id not in targets:
                                continue
                            source = tuple(
                                slice(
                                    offset[axis] * options.stride,
                                    (offset[axis] + 1) * options.stride,
                                )
                                for axis in range(3)
                            )
                            if chunk_id not in accumulators:
                                accumulators[chunk_id] = np.zeros(
                                    raw_chunk_shape, dtype=np.float32
                                )
                                weights[chunk_id] = np.zeros(
                                    raw_chunk_shape, dtype=np.float32
                                )
                            local_weight = gaussian[source]
                            accumulators[chunk_id] += probability[source] * local_weight
                            weights[chunk_id] += local_weight

                emitted = 0
                for chunk_id in sorted(targets):
                    weight = weights.get(chunk_id)
                    if weight is None or not np.all(weight > 0.0):
                        continue
                    probability = np.clip(
                        accumulators[chunk_id] / weight, 0.0, 1.0
                    )
                    origin = tuple(
                        chunk_id[axis] * raw_chunk_shape[axis] for axis in range(3)
                    )
                    destination = tuple(
                        slice(origin[axis], origin[axis] + raw_chunk_shape[axis])
                        for axis in range(3)
                    )
                    raw_chunk = read_crop(raw_volume, origin, raw_chunk_shape)
                    valid = np.asarray(raw_chunk) != 0
                    probability_array[destination] = np.rint(
                        probability * 255.0
                    ).astype(np.uint8)
                    validity_array[destination] = valid.astype(np.uint8)
                    valid_count = int(np.count_nonzero(valid))
                    total_valid_voxels += valid_count
                    if valid_count:
                        probability_sum_valid += float(probability[valid].sum())
                    predicted_chunks.add(chunk_id)
                    emitted += 1

                _event(
                    events,
                    "site_complete",
                    site_id=site_id,
                    site_index=site_index,
                    target_chunks=len(targets),
                    tile_count=len(tiles),
                    emitted_chunks=emitted,
                    skipped_chunks=len(targets) - emitted,
                )
                print(
                    f"site {site_index}/{len(rows)} {site_id}: "
                    f"tiles={len(tiles)} chunks={emitted} "
                    f"total_chunks={len(predicted_chunks)}",
                    file=sys.stderr,
                    flush=True,
                )

            if not predicted_chunks:
                raise InferenceError("no fully covered teacher chunks were emitted")
            selected_linear = sorted(
                _linear_chunk_id(chunk_id, chunk_grid_zyx)
                for chunk_id in predicted_chunks
            )
            write_json_atomic(
                output / "carve_selected_chunks.json",
                {
                    "schema_version": 1,
                    "array_key": "0",
                    "validity_array_key": "1",
                    "chunks_zyx": list(raw_chunk_shape),
                    "chunk_grid_zyx": list(chunk_grid_zyx),
                    "selected_chunk_ids": selected_linear,
                },
            )
            completed_at = utc_now()
            duration = time.perf_counter() - started
            summary = {
                "schema_version": 1,
                "status": "complete",
                "completed_at": completed_at,
                "site_count": len(rows),
                "tile_count": total_tiles,
                "requested_chunk_count": total_requested_chunks,
                "predicted_chunk_count": len(predicted_chunks),
                "valid_voxel_count": total_valid_voxels,
                "mean_probability_in_valid": (
                    probability_sum_valid / total_valid_voxels
                    if total_valid_voxels
                    else 0.0
                ),
                "duration_seconds": duration,
            }
            write_json_atomic(output / "summary.json", summary)
            write_json_atomic(
                output / "crossres_sparse_mirror.json",
                {
                    "schema_version": 1,
                    "kind": "crossres-teacher-inference",
                    "state": "complete",
                    "created_at_utc": provenance["created_at"],
                    "completed_at_utc": completed_at,
                    "output": str(output),
                    "array_key": "0",
                    "validity_array_key": "1",
                    "source_mirror": str(mirror),
                    "checkpoint": provenance["checkpoint"],
                    "selection": {
                        "predicted_chunk_count": len(predicted_chunks),
                        "requested_chunk_count": total_requested_chunks,
                    },
                    "zarr": provenance["zarr"],
                },
            )
            provenance.update(
                {"status": "complete", "completed_at": completed_at, "summary": summary}
            )
            write_json_atomic(output / "provenance.json", provenance)
            _event(events, "run_complete", **summary)
    except Exception as error:
        provenance.update(
            {
                "status": "failed",
                "failed_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        write_json_atomic(output / "provenance.json", provenance)
        if events_path.exists():
            with events_path.open("a", encoding="utf-8", newline="\n") as events:
                _event(
                    events,
                    "run_failed",
                    error_type=type(error).__name__,
                    error=str(error),
                )
        raise
    return output
