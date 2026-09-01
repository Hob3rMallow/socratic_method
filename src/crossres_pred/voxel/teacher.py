from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numcodecs import Blosc
from scipy.ndimage import gaussian_filter

from ..mirror_state import validate_sparse_mirror
from .inference import _predict_logits, _predict_probability
from .io import open_volume, read_crop, split_volume_spec
from .resources import assert_cuda_power_limit, configure_cpu_budget
from .teacher_model import LoadedTeacher, load_teacher_checkpoint, normalize_teacher_ct


@dataclass(frozen=True)
class TeacherOptions:
    chunks: int = 768
    allow_fewer_chunks: bool = False
    seed: int = 1203
    input_shape_zyx: tuple[int, int, int] = (256, 256, 256)
    threshold: float = 0.45
    min_positive_voxels: int = 32
    min_ct_nonzero_fraction: float = 0.05
    max_candidates: int = 20_000
    device: str = "cuda"
    amp_dtype: str = "float16"
    mirror_tta: bool = True
    sliding_blend: bool = True
    sliding_step_size: float = 0.5
    inference_batch_size: int = 1
    candidate_tile_chunks: int = 0
    prediction_cache_entries: int = 0
    candidate_chunk_zyx: tuple[int, int, int] | None = None
    max_cpu_threads: int = 16

    def validate(self) -> None:
        if self.chunks <= 0 or self.max_candidates <= 0:
            raise ValueError("chunk and candidate counts must be positive")
        if len(self.input_shape_zyx) != 3 or any(
            size <= 0 for size in self.input_shape_zyx
        ):
            raise ValueError("teacher input dimensions must be positive")
        if not 0 <= self.threshold <= 1:
            raise ValueError("teacher threshold must be in [0, 1]")
        if self.min_positive_voxels < 0:
            raise ValueError("min_positive_voxels must be non-negative")
        if not 0 <= self.min_ct_nonzero_fraction <= 1:
            raise ValueError("min_ct_nonzero_fraction must be in [0, 1]")
        if self.amp_dtype not in {"auto", "bfloat16", "float16"}:
            raise ValueError("amp_dtype must be auto, bfloat16, or float16")
        if not 0 < self.sliding_step_size <= 1:
            raise ValueError("sliding_step_size must be in (0, 1]")
        if not 1 <= self.inference_batch_size <= 4:
            raise ValueError("inference_batch_size must be in [1, 4]")
        if not 0 <= self.candidate_tile_chunks <= 64:
            raise ValueError("candidate_tile_chunks must be in [0, 64]")
        if not 0 <= self.prediction_cache_entries <= 256:
            raise ValueError("prediction_cache_entries must be in [0, 256]")
        if self.candidate_chunk_zyx is not None and (
            len(self.candidate_chunk_zyx) != 3
            or any(value < 0 for value in self.candidate_chunk_zyx)
        ):
            raise ValueError("candidate_chunk_zyx must be a non-negative z-y-x triplet")
        if not 1 <= self.max_cpu_threads <= 16:
            raise ValueError("max_cpu_threads must be in [1, 16]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_teacher_identity(identity: Any) -> Any:
    """Interpret pre-batching teacher states as explicit serial execution."""

    if not isinstance(identity, dict):
        return identity
    result = json.loads(json.dumps(identity, sort_keys=True))
    options = result.get("options")
    if isinstance(options, dict):
        options.setdefault("inference_batch_size", 1)
        options.setdefault("allow_fewer_chunks", False)
        # This cache changes only scheduling of already-defined global sliding
        # windows.  Cached logits are stored at the same official float16
        # boundary and accumulated in the same order, so capacity is safe to
        # tune across a reboot-resume without changing teacher bytes.
        options.pop("prediction_cache_entries", None)
        result.setdefault("effective_chunks", options.get("chunks"))
    return result


def _identity_difference_paths(
    left: Any, right: Any, *, prefix: str = "", limit: int = 8
) -> list[str]:
    """Return bounded field paths for a fail-closed resume diagnostic."""

    if left == right:
        return []
    if isinstance(left, dict) and isinstance(right, dict):
        result: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                result.append(path)
            else:
                result.extend(
                    _identity_difference_paths(
                        left[key], right[key], prefix=path, limit=limit - len(result)
                    )
                )
            if len(result) >= limit:
                break
        return result[:limit]
    if isinstance(left, list) and isinstance(right, list):
        result = []
        for index, (left_item, right_item) in enumerate(
            zip(left, right, strict=False)
        ):
            result.extend(
                _identity_difference_paths(
                    left_item,
                    right_item,
                    prefix=f"{prefix}[{index}]",
                    limit=limit - len(result),
                )
            )
            if len(result) >= limit:
                break
        if len(left) != len(right) and len(result) < limit:
            result.append(f"{prefix}.length")
        return result[:limit]
    return [prefix or "<root>"]


def _effective_teacher_chunk_target(
    available_chunks: int, options: TeacherOptions
) -> int:
    if available_chunks <= 0:
        raise RuntimeError("fine mirror has no fully supported teacher neighborhoods")
    if available_chunks >= options.chunks:
        return options.chunks
    if not options.allow_fewer_chunks:
        raise RuntimeError(
            f"fine mirror has only {available_chunks:,} full teacher "
            f"neighborhoods for {options.chunks:,} requested chunks"
        )
    return available_chunks


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Tolerate brief Windows sharing locks from status readers."""

    attempts = 100
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _replace_with_retry(temporary, path)


def _parse_chunk_coordinate(
    relative_path: str, array_key: str
) -> tuple[int, int, int] | None:
    parts = tuple(Path(relative_path).parts)
    key_parts = tuple(part for part in array_key.split("/") if part)
    if key_parts and parts[: len(key_parts)] != key_parts:
        return None
    chunk_parts = parts[len(key_parts) :]
    if len(chunk_parts) == 1 and "." in chunk_parts[0]:
        chunk_parts = tuple(chunk_parts[0].split("."))
    if len(chunk_parts) != 3:
        return None
    try:
        return tuple(int(item) for item in chunk_parts)  # type: ignore[return-value]
    except ValueError:
        return None


def load_present_chunk_coordinates(
    inventory_path: str | Path,
    *,
    array_key: str,
    grid_zyx: tuple[int, int, int] | None = None,
) -> np.ndarray:
    source = Path(inventory_path).expanduser().resolve()
    if source.suffix.lower() == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or "selected_chunk_ids" not in value:
            raise ValueError(f"{source}: unsupported chunk-support JSON")
        declared_key = str(value.get("array_key", array_key))
        if declared_key != array_key:
            raise ValueError(
                f"{source}: array key {declared_key!r} does not match {array_key!r}"
            )
        declared_grid = tuple(int(item) for item in value.get("chunk_grid_zyx", ()))
        if len(declared_grid) != 3:
            raise ValueError(f"{source}: missing three-dimensional chunk grid")
        if grid_zyx is not None and declared_grid != grid_zyx:
            raise ValueError(
                f"{source}: chunk grid {declared_grid} does not match {grid_zyx}"
            )
        identifiers = np.asarray(value["selected_chunk_ids"], dtype=np.int64)
        if identifiers.ndim != 1 or identifiers.size == 0:
            raise ValueError(f"{source}: selected_chunk_ids must be non-empty")
        plane = declared_grid[1] * declared_grid[2]
        z = identifiers // plane
        remainder = identifiers % plane
        y = remainder // declared_grid[2]
        x = remainder % declared_grid[2]
        coordinates = np.stack((z, y, x), axis=1)
        if (
            (coordinates < 0)
            | (coordinates >= np.asarray(declared_grid, dtype=np.int64))
        ).any():
            raise ValueError(f"{source}: selected chunk falls outside declared grid")
        return np.unique(coordinates, axis=0)

    coordinates: list[tuple[int, int, int]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from error
            if row.get("kind") != "chunk":
                continue
            coordinate = _parse_chunk_coordinate(
                str(row.get("relative_path", "")), array_key
            )
            if coordinate is not None:
                coordinates.append(coordinate)
    if not coordinates:
        raise ValueError(f"{source}: no chunks found for array {array_key!r}")
    return np.unique(np.asarray(coordinates, dtype=np.int64), axis=0)


def validate_local_support_snapshot(
    *,
    input_path: Path,
    array_key: str,
    support_path: Path,
    shape_zyx: tuple[int, int, int],
    chunks_zyx: tuple[int, int, int],
    grid_zyx: tuple[int, int, int],
    radius_zyx: tuple[int, int, int],
    required_chunks: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Prove that a frozen partial mirror contains its declared neighborhoods."""

    value = json.loads(support_path.read_text(encoding="utf-8"))
    if value.get("schema") != "crossres-local-zarr-support-v1":
        raise ValueError(
            f"{support_path}: partial mirror requires a local-support snapshot"
        )
    declared_zarr = Path(str(value.get("zarr", ""))).expanduser().resolve()
    if declared_zarr != input_path.resolve():
        raise ValueError(
            f"{support_path}: support Zarr {declared_zarr} does not match {input_path}"
        )
    expected = {
        "array_key": array_key,
        "shape_zyx": list(shape_zyx),
        "chunks_zyx": list(chunks_zyx),
        "chunk_grid_zyx": list(grid_zyx),
        "context_radius_chunks_zyx": list(radius_zyx),
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise ValueError(
                f"{support_path}: {name} {value.get(name)!r} != {expected_value!r}"
            )

    array_metadata_path = input_path / Path(array_key) / ".zarray"
    array_metadata = json.loads(array_metadata_path.read_text(encoding="utf-8"))
    for name, expected_value in {
        "shape": list(shape_zyx),
        "chunks": list(chunks_zyx),
    }.items():
        if array_metadata.get(name) != expected_value:
            raise ValueError(
                f"{array_metadata_path}: local Zarr {name} "
                f"{array_metadata.get(name)!r} != {expected_value!r}"
            )
    metadata_grid = tuple(
        math.ceil(size / chunk)
        for size, chunk in zip(shape_zyx, chunks_zyx, strict=True)
    )
    if metadata_grid != grid_zyx:
        raise ValueError(
            f"{support_path}: chunk grid {grid_zyx!r} does not match "
            f"the local Zarr grid {metadata_grid!r}"
        )
    separator = str(array_metadata.get("dimension_separator") or ".")
    if separator not in {".", "/"}:
        raise ValueError(
            f"{array_metadata_path}: unsupported dimension separator {separator!r}"
        )
    if value.get("dimension_separator") != separator:
        raise ValueError(
            f"{support_path}: dimension separator does not match the local Zarr"
        )

    raw_identifiers = value.get("selected_chunk_ids")
    if not isinstance(raw_identifiers, list) or not raw_identifiers:
        raise ValueError(f"{support_path}: selected_chunk_ids must be non-empty")
    identifiers = [int(item) for item in raw_identifiers]
    declared_count = int(value.get("present_chunk_count", -1))
    if len(identifiers) != declared_count or len(set(identifiers)) != declared_count:
        raise ValueError(f"{support_path}: present chunk count is inconsistent")
    present = load_present_chunk_coordinates(
        support_path, array_key=array_key, grid_zyx=grid_zyx
    )
    if len(present) != declared_count:
        raise ValueError(f"{support_path}: decoded present chunk count is inconsistent")

    array_root = input_path / Path(array_key)
    present_bytes = 0
    missing: list[str] = []
    for coordinate in present:
        parts = tuple(str(int(item)) for item in coordinate)
        relative = Path(*parts) if separator == "/" else Path(".".join(parts))
        chunk_path = array_root / relative
        if not chunk_path.is_file():
            if len(missing) < 5:
                missing.append(relative.as_posix())
            continue
        present_bytes += chunk_path.stat().st_size
    if missing:
        raise ValueError(
            f"{support_path}: local-support chunks are missing: {missing}"
        )
    if present_bytes != int(value.get("present_bytes", -1)):
        raise ValueError(
            f"{support_path}: present bytes changed from "
            f"{value.get('present_bytes')!r} to {present_bytes}"
        )

    interior = interior_chunk_coordinates(
        present, grid_zyx=grid_zyx, radius_zyx=radius_zyx
    )
    declared_interior = int(value.get("interior_chunk_count", -1))
    if len(interior) != declared_interior:
        raise ValueError(
            f"{support_path}: interior chunk count changed from "
            f"{declared_interior:,} to {len(interior):,}"
        )
    if len(interior) < required_chunks:
        raise ValueError(
            f"{support_path}: only {len(interior):,} complete neighborhoods for "
            f"{required_chunks:,} requested teacher chunks"
        )
    return (
        present,
        interior,
        {
            "schema": "crossres-local-support-validation-v1",
            "support_sha256": _sha256(support_path),
            "present_chunks": declared_count,
            "present_bytes": present_bytes,
            "interior_chunks": len(interior),
            "radius_zyx": list(radius_zyx),
        },
    )


def interior_chunk_coordinates(
    coordinates_zyx: np.ndarray,
    *,
    grid_zyx: tuple[int, int, int],
    radius_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Return chunks whose entire context neighborhood is materialized."""

    coordinates = np.asarray(coordinates_zyx, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("chunk coordinates must be Nx3 z-y-x")
    if any(radius < 0 for radius in radius_zyx):
        raise ValueError("chunk radii must be non-negative")
    grid = np.asarray(grid_zyx, dtype=np.int64)
    if ((coordinates < 0) | (coordinates >= grid)).any():
        raise ValueError("chunk coordinates fall outside the declared grid")
    encoded = (coordinates[:, 0] * grid[1] + coordinates[:, 1]) * grid[2]
    encoded += coordinates[:, 2]
    present = {int(item) for item in encoded}
    keep = np.ones(coordinates.shape[0], dtype=bool)
    for index, coordinate in enumerate(coordinates):
        for dz in range(-radius_zyx[0], radius_zyx[0] + 1):
            for dy in range(-radius_zyx[1], radius_zyx[1] + 1):
                for dx in range(-radius_zyx[2], radius_zyx[2] + 1):
                    neighbor = coordinate + (dz, dy, dx)
                    if ((neighbor < 0) | (neighbor >= grid)).any():
                        keep[index] = False
                        break
                    neighbor_id = int(
                        (neighbor[0] * grid[1] + neighbor[1]) * grid[2] + neighbor[2]
                    )
                    if neighbor_id not in present:
                        keep[index] = False
                        break
                if not keep[index]:
                    break
            if not keep[index]:
                break
    return coordinates[keep]


def centered_crop_slices(
    outer_zyx: tuple[int, int, int], inner_zyx: tuple[int, int, int]
) -> tuple[slice, slice, slice]:
    slices: list[slice] = []
    for outer, inner in zip(outer_zyx, inner_zyx, strict=True):
        if inner <= 0 or inner > outer or (outer - inner) % 2:
            raise ValueError("inner shape must be centered exactly inside outer shape")
        start = (outer - inner) // 2
        slices.append(slice(start, start + inner))
    return tuple(slices)  # type: ignore[return-value]


def _initialize_target_zarr(
    output: Path,
    *,
    shape_zyx: tuple[int, int, int],
    chunks_zyx: tuple[int, int, int],
    attrs: dict[str, Any],
) -> None:
    output.mkdir(parents=True)
    array_dir = output / "0"
    array_dir.mkdir()
    _atomic_json(output / ".zgroup", {"zarr_format": 2})
    _atomic_json(output / ".zattrs", attrs)
    _atomic_json(
        array_dir / ".zarray",
        {
            "zarr_format": 2,
            "shape": list(shape_zyx),
            "chunks": list(chunks_zyx),
            "dtype": "|u1",
            "compressor": {
                "id": "blosc",
                "cname": "lz4",
                "clevel": 7,
                "shuffle": 1,
                "blocksize": 0,
            },
            "fill_value": 0,
            "order": "C",
            "filters": None,
            "dimension_separator": "/",
        },
    )
    _atomic_json(array_dir / ".zattrs", {"axes": "zyx"})
    (output / "records").mkdir()


def _write_chunk_atomic(
    output: Path, coordinate_zyx: tuple[int, int, int], value: np.ndarray
) -> tuple[Path, int, str]:
    destination = output / "0"
    for coordinate in coordinate_zyx[:-1]:
        destination /= str(coordinate)
    destination.mkdir(parents=True, exist_ok=True)
    destination /= str(coordinate_zyx[-1])
    encoded = Blosc(cname="lz4", clevel=7, shuffle=Blosc.SHUFFLE).encode(
        np.ascontiguousarray(value, dtype=np.uint8).tobytes(order="C")
    )
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return destination, len(encoded), hashlib.sha256(encoded).hexdigest()


def _existing_records(
    output: Path,
    chunks_zyx: tuple[int, int, int],
) -> dict[tuple[int, int, int], dict[str, Any]]:
    records: dict[tuple[int, int, int], dict[str, Any]] = {}
    for path in sorted((output / "records").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema") != "crossres-native-fine-teacher-chunk-v1":
            raise ValueError(f"{path}: unsupported teacher chunk record schema")
        coordinate = tuple(int(item) for item in value["chunk_zyx"])
        if len(coordinate) != 3 or any(item < 0 for item in coordinate):
            raise ValueError(f"{path}: invalid chunk coordinate {coordinate}")
        if coordinate in records:
            raise ValueError(f"{path}: duplicate chunk coordinate {coordinate}")
        expected_relative = (
            Path("0").joinpath(*(str(item) for item in coordinate)).as_posix()
        )
        if value.get("relative_path") != expected_relative:
            raise ValueError(
                f"{path}: relative path {value.get('relative_path')!r} does not "
                f"match {expected_relative!r}"
            )
        if tuple(int(item) for item in value.get("shape_zyx", ())) != chunks_zyx:
            raise ValueError(f"{path}: chunk shape does not match Zarr metadata")
        expected_origin = tuple(
            coordinate[index] * chunks_zyx[index] for index in range(3)
        )
        if tuple(int(item) for item in value.get("origin_zyx", ())) != expected_origin:
            raise ValueError(f"{path}: chunk origin does not match its coordinate")
        chunk_path = output.joinpath(*expected_relative.split("/"))
        if not chunk_path.is_file():
            raise ValueError(f"{path}: missing encoded chunk {chunk_path}")
        encoded = chunk_path.read_bytes()
        if len(encoded) != int(value.get("encoded_bytes", -1)):
            raise ValueError(f"{chunk_path}: encoded byte count does not match record")
        encoded_sha256 = hashlib.sha256(encoded).hexdigest()
        if encoded_sha256 != value.get("encoded_sha256"):
            raise ValueError(f"{chunk_path}: encoded SHA-256 does not match record")
        decoded = Blosc().decode(encoded)
        decoded_array = np.frombuffer(decoded, dtype=np.uint8)
        expected_voxels = math.prod(chunks_zyx)
        if decoded_array.size != expected_voxels:
            raise ValueError(
                f"{chunk_path}: decoded {decoded_array.size:,} voxels, "
                f"expected {expected_voxels:,}"
            )
        if not set(np.unique(decoded_array).tolist()) <= {0, 255}:
            raise ValueError(f"{chunk_path}: teacher chunk has invalid label values")
        positive_voxels = int(np.count_nonzero(decoded_array))
        if positive_voxels != int(value.get("positive_voxels", -1)):
            raise ValueError(
                f"{chunk_path}: positive voxel count does not match record"
            )
        records[coordinate] = value
    return records


def _amp_configuration(name: str, device: torch.device) -> tuple[torch.dtype, bool]:
    if device.type != "cuda":
        return torch.float32, False
    if name == "bfloat16" or (name == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16, True
    return torch.float16, True


def villa_sliding_window_steps(
    image_size: int, patch_size: int, step_size: float
) -> list[int]:
    """Match Villa/nnU-Net's globally anchored sliding-window positions."""

    if image_size < patch_size:
        raise ValueError("image size must be at least the teacher patch size")
    if not 0 < step_size <= 1:
        raise ValueError("step_size must be in (0, 1]")
    target_step = max(1, int(patch_size * step_size))
    step_count = int(np.ceil((image_size - patch_size) / target_step)) + 1
    maximum = image_size - patch_size
    if step_count == 1:
        return [0]
    actual_step = maximum / (step_count - 1)
    return [
        min(int(np.round(actual_step * index)), maximum) for index in range(step_count)
    ]


def villa_gaussian_map(patch_shape_zyx: tuple[int, int, int]) -> np.ndarray:
    """Match Villa's Gaussian logit-blending weights exactly."""

    impulse = np.zeros(patch_shape_zyx, dtype=np.float32)
    center = tuple(size // 2 for size in patch_shape_zyx)
    impulse[center] = 1.0
    result = gaussian_filter(
        impulse,
        tuple(size / 8.0 for size in patch_shape_zyx),
        order=0,
        mode="constant",
        cval=0,
    )
    result /= max(float(result.max()), 1.0e-12)
    return np.clip(result, a_min=0, a_max=None)


class _PredictionLogitCache:
    """Bounded CPU cache for official float16 sliding-window logits."""

    def __init__(self, max_entries: int) -> None:
        if max_entries <= 0:
            raise ValueError("prediction logit cache must have positive capacity")
        self.max_entries = int(max_entries)
        self._values: OrderedDict[
            tuple[int, int, int], np.ndarray | None
        ] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.computed = 0

    def get(
        self, origin: tuple[int, int, int]
    ) -> tuple[bool, np.ndarray | None]:
        if origin not in self._values:
            self.misses += 1
            return False, None
        value = self._values.pop(origin)
        self._values[origin] = value
        self.hits += 1
        return True, value

    def put(
        self, origin: tuple[int, int, int], logits: np.ndarray | None
    ) -> None:
        if origin in self._values:
            self._values.pop(origin)
        self._values[origin] = logits
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)
            self.evictions += 1

    @property
    def entries(self) -> int:
        return len(self._values)


def _candidate_execution_order(
    coordinates_zyx: np.ndarray,
    *,
    seed: int,
    tile_chunks: int,
) -> np.ndarray:
    """Randomize coverage tiles and Morton-walk neighbors for inference reuse."""

    coordinates = np.asarray(coordinates_zyx, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("candidate coordinates must be Nx3 z-y-x")
    if tile_chunks < 0:
        raise ValueError("candidate tile size must be non-negative")
    rng = np.random.default_rng(seed)
    if tile_chunks == 0:
        return coordinates[rng.permutation(coordinates.shape[0])]

    grouped: dict[tuple[int, int, int], list[int]] = {}
    for index, coordinate in enumerate(coordinates):
        key = tuple(int(value) for value in coordinate // tile_chunks)
        grouped.setdefault(key, []).append(index)
    keys = sorted(grouped)
    shuffled_keys = [keys[int(index)] for index in rng.permutation(len(keys))]
    ordered: list[np.ndarray] = []
    for key in shuffled_keys:
        members = coordinates[np.asarray(grouped[key], dtype=np.int64)]
        local = members % tile_chunks
        morton = np.zeros(members.shape[0], dtype=np.int64)
        for bit in range((tile_chunks - 1).bit_length()):
            morton |= ((local[:, 2] >> bit) & 1) << (3 * bit)
            morton |= ((local[:, 1] >> bit) & 1) << (3 * bit + 1)
            morton |= ((local[:, 0] >> bit) & 1) << (3 * bit + 2)
        member_order = np.argsort(morton, kind="stable")
        ordered.append(members[member_order])
    return np.concatenate(ordered, axis=0)


def _overlapping_origins(
    *,
    volume_shape_zyx: tuple[int, int, int],
    patch_shape_zyx: tuple[int, int, int],
    output_origin_zyx: tuple[int, int, int],
    output_shape_zyx: tuple[int, int, int],
    step_size: float,
) -> list[tuple[int, int, int]]:
    per_axis: list[list[int]] = []
    for volume_size, patch_size, output_start, output_size in zip(
        volume_shape_zyx,
        patch_shape_zyx,
        output_origin_zyx,
        output_shape_zyx,
        strict=True,
    ):
        output_end = output_start + output_size
        per_axis.append(
            [
                origin
                for origin in villa_sliding_window_steps(
                    volume_size, patch_size, step_size
                )
                if origin < output_end and origin + patch_size > output_start
            ]
        )
    return [tuple(int(value) for value in values) for values in product(*per_axis)]


def _predict_blended_probability(
    *,
    teacher: LoadedTeacher,
    volume: Any,
    volume_shape_zyx: tuple[int, int, int],
    output_origin_zyx: tuple[int, int, int],
    output_shape_zyx: tuple[int, int, int],
    step_size: float,
    device: torch.device,
    amp_dtype: torch.dtype,
    autocast_enabled: bool,
    mirror_tta: bool,
    inference_batch_size: int,
    prediction_cache: _PredictionLogitCache | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Reproduce Villa's TTA-logit then global Gaussian-blend inference."""

    patch_shape = teacher.patch_shape_zyx
    gaussian = villa_gaussian_map(patch_shape)
    origins = _overlapping_origins(
        volume_shape_zyx=volume_shape_zyx,
        patch_shape_zyx=patch_shape,
        output_origin_zyx=output_origin_zyx,
        output_shape_zyx=output_shape_zyx,
        step_size=step_size,
    )
    logit_sum = np.zeros((2, *output_shape_zyx), dtype=np.float32)
    weight_sum = np.zeros(output_shape_zyx, dtype=np.float32)
    processed: list[tuple[int, int, int]] = []
    output_end = tuple(
        start + size
        for start, size in zip(output_origin_zyx, output_shape_zyx, strict=True)
    )
    logits_by_origin: dict[tuple[int, int, int], np.ndarray | None] = {}
    missing_origins: list[tuple[int, int, int]] = []
    for patch_origin in origins:
        if prediction_cache is None:
            missing_origins.append(patch_origin)
            continue
        found, cached = prediction_cache.get(patch_origin)
        if found:
            logits_by_origin[patch_origin] = cached
        else:
            missing_origins.append(patch_origin)

    for batch_start in range(0, len(missing_origins), inference_batch_size):
        batch: list[tuple[tuple[int, int, int], np.ndarray]] = []
        for patch_origin in missing_origins[
            batch_start : batch_start + inference_batch_size
        ]:
            raw = read_crop(volume, patch_origin, patch_shape)
            if raw.size == 0 or int(raw.min()) == int(raw.max()):
                logits_by_origin[patch_origin] = None
                if prediction_cache is not None:
                    prediction_cache.put(patch_origin, None)
                continue
            batch.append(
                (
                    patch_origin,
                    normalize_teacher_ct(raw, teacher.normalization),
                )
            )
        if batch:
            image = torch.from_numpy(
                np.stack([normalized for _, normalized in batch], axis=0)[:, None]
            ).to(device, non_blocking=True)
            logits = _predict_logits(
                teacher.model,
                image,
                amp_dtype=amp_dtype,
                autocast_enabled=autocast_enabled,
                mirror_tta=mirror_tta,
            )
            # Official inference writes each averaged-TTA patch as float16 before
            # the float32 Gaussian blend. The cache retains that exact boundary.
            logits_batch = logits.to(dtype=torch.float16).cpu().numpy()
            if prediction_cache is not None:
                prediction_cache.computed += len(batch)
            for (patch_origin, _), logits_numpy in zip(
                batch, logits_batch, strict=True
            ):
                logits_by_origin[patch_origin] = logits_numpy
                if prediction_cache is not None:
                    prediction_cache.put(patch_origin, logits_numpy)

    # Accumulate in the original global-origin order. Cache hits therefore alter
    # scheduling only; they cannot alter Villa''s Gaussian reduction order.
    for patch_origin in origins:
        logits_numpy = logits_by_origin.get(patch_origin)
        if logits_numpy is None:
            continue
        patch_end = tuple(
            start + size
            for start, size in zip(patch_origin, patch_shape, strict=True)
        )
        intersection_start = tuple(
            max(a, b) for a, b in zip(output_origin_zyx, patch_origin, strict=True)
        )
        intersection_end = tuple(
            min(a, b) for a, b in zip(output_end, patch_end, strict=True)
        )
        if any(
            a >= b
            for a, b in zip(intersection_start, intersection_end, strict=True)
        ):
            continue
        output_key = tuple(
            slice(a - base, b - base)
            for a, b, base in zip(
                intersection_start,
                intersection_end,
                output_origin_zyx,
                strict=True,
            )
        )
        patch_key = tuple(
            slice(a - base, b - base)
            for a, b, base in zip(
                intersection_start,
                intersection_end,
                patch_origin,
                strict=True,
            )
        )
        weight = gaussian[patch_key]
        logits_float32 = logits_numpy.astype(np.float32, copy=False)
        logit_sum[(slice(None), *output_key)] += (
            logits_float32[(slice(None), *patch_key)] * weight[None]
        )
        weight_sum[output_key] += weight
        processed.append(patch_origin)
    uncovered = weight_sum <= 0
    if not processed or np.any(uncovered):
        output_raw = read_crop(volume, output_origin_zyx, output_shape_zyx)
        if np.any(output_raw[uncovered] != 0):
            raise RuntimeError(
                "teacher sliding blend left nonzero-CT output voxels uncovered"
            )
        if not processed:
            return np.zeros(output_shape_zyx, dtype=np.float32), []
        # Masked fine-volume voxels are encoded as exact zero. Constant all-zero
        # windows are intentionally not sent through instance normalization, so
        # their uncovered contribution is deterministic teacher background.
        weight_sum[uncovered] = 1.0
    logit_sum /= weight_sum[None]
    difference = np.clip(logit_sum[1] - logit_sum[0], -80.0, 80.0)
    probability = 1.0 / (1.0 + np.exp(-difference))
    probability[uncovered] = 0.0
    return probability.astype(np.float32), processed


def _inventory_text(
    records: dict[tuple[int, int, int], dict[str, Any]],
) -> str:
    lines: list[str] = []
    for coordinate, record in sorted(records.items()):
        lines.append(
            json.dumps(
                {
                    "kind": "chunk",
                    "relative_path": "0/"
                    + "/".join(str(item) for item in coordinate),
                    "size": int(record["encoded_bytes"]),
                    "sha256": record["encoded_sha256"],
                    "positive_voxels": int(record["positive_voxels"]),
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )
    return "".join(lines)


def _write_inventory(
    output: Path, records: dict[tuple[int, int, int], dict[str, Any]]
) -> None:
    destination = output / "crossres_sparse_objects.jsonl"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(_inventory_text(records))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def _validate_complete_teacher_output(
    output: Path,
    state: dict[str, Any],
    records: dict[tuple[int, int, int], dict[str, Any]],
) -> dict[str, Any]:
    if state.get("state") != "complete":
        raise ValueError(f"{output}: teacher materialization is not complete")
    identity = state.get("identity", {})
    options = identity.get("options", {})
    requested = int(
        identity.get("effective_chunks", options.get("chunks", -1))
    )
    accepted = len(records)
    allow_fewer = bool(options.get("allow_fewer_chunks", False))
    shortfall = 0 < accepted < requested
    if (
        requested <= 0
        or accepted > requested
        or (accepted != requested and not (allow_fewer and shortfall))
    ):
        raise ValueError(
            f"{output}: found {accepted:,} validated chunks, expected {requested:,}"
        )
    examined = int(state.get("examined", -1))
    if shortfall:
        candidate_chunk = options.get("candidate_chunk_zyx")
        candidate_budget = (
            1
            if candidate_chunk is not None
            else min(
                int(options.get("max_candidates", -1)),
                int(identity.get("eligible_chunks", -1)),
            )
        )
        if candidate_budget <= 0 or examined != candidate_budget:
            raise ValueError(
                f"{output}: accepted-chunk shortfall was recorded before exhausting "
                "the deterministic candidate budget"
            )
    if int(state.get("accepted", -1)) != accepted:
        raise ValueError(f"{output}: teacher state accepted count is inconsistent")
    expected_paths = {
        record["relative_path"] for record in records.values()
    }
    actual_paths = {
        path.relative_to(output).as_posix()
        for path in (output / "0").rglob("*")
        if path.is_file() and not path.name.startswith(".")
    }
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)[:5]
        extra = sorted(actual_paths - expected_paths)[:5]
        raise ValueError(
            f"{output}: encoded chunk set differs from records; "
            f"missing={missing}, extra={extra}"
        )
    inventory_path = output / "crossres_sparse_objects.jsonl"
    if not inventory_path.is_file():
        raise ValueError(f"{output}: sparse support inventory is missing")
    inventory_rows: dict[str, dict[str, Any]] = {}
    with inventory_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            relative_path = str(row.get("relative_path", ""))
            if row.get("kind") != "chunk" or not relative_path:
                raise ValueError(
                    f"{inventory_path}:{line_number}: invalid chunk inventory row"
                )
            if relative_path in inventory_rows:
                raise ValueError(
                    f"{inventory_path}:{line_number}: duplicate {relative_path}"
                )
            inventory_rows[relative_path] = row
    if set(inventory_rows) != expected_paths:
        raise ValueError(f"{output}: sparse support inventory path set is inconsistent")
    for record in records.values():
        row = inventory_rows[record["relative_path"]]
        if (
            int(row.get("size", -1)) != int(record["encoded_bytes"])
            or row.get("sha256") != record["encoded_sha256"]
        ):
            raise ValueError(
                f"{output}: sparse support inventory bytes do not match records"
            )
        if (
            "positive_voxels" in row
            and int(row["positive_voxels"]) != int(record["positive_voxels"])
        ):
            raise ValueError(
                f"{output}: sparse support inventory positives do not match records"
            )
    return {
        "schema": "crossres-native-fine-teacher-validation-v1",
        "output": str(output),
        "state": "complete",
        "requested_chunks": requested,
        "validated_chunks": accepted,
        "examined_candidates": examined,
        "filtered_candidates": examined - accepted,
        "encoded_bytes": sum(int(row["encoded_bytes"]) for row in records.values()),
        "positive_voxels": sum(
            int(row["positive_voxels"]) for row in records.values()
        ),
        "teacher_checkpoint_sha256": state["identity"].get(
            "teacher_checkpoint_sha256"
        ),
    }


def validate_teacher_materialization(
    output_path: str | Path,
) -> dict[str, Any]:
    output = Path(output_path).expanduser().resolve()
    zarray_path = output / "0" / ".zarray"
    state_path = output / "teacher_state.json"
    if not zarray_path.is_file() or not state_path.is_file():
        raise ValueError(f"{output}: teacher metadata is incomplete")
    zarray = json.loads(zarray_path.read_text(encoding="utf-8"))
    chunks = tuple(int(item) for item in zarray.get("chunks", ()))
    if len(chunks) != 3 or any(item <= 0 for item in chunks):
        raise ValueError(f"{zarray_path}: invalid three-dimensional chunk shape")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    records = _existing_records(output, chunks)
    return _validate_complete_teacher_output(output, state, records)


def materialize_teacher(
    *,
    fine_volume: str,
    fine_support_inventory: str | Path,
    output_path: str | Path,
    teacher_checkpoint: str | Path,
    villa_source: str | Path,
    options: TeacherOptions,
) -> Path:
    """Materialize inspectable dense labels from native fine-CT voxel blocks."""

    options.validate()
    configure_cpu_budget(options.max_cpu_threads)
    input_path, array_key_value = split_volume_spec(fine_volume)
    array_key = array_key_value or "0"
    support_path = Path(fine_support_inventory).expanduser().resolve()
    checkpoint_path = Path(teacher_checkpoint).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    mirror_manifest = input_path / "crossres_sparse_mirror.json"
    if not mirror_manifest.is_file():
        raise ValueError(f"fine input has no sparse mirror manifest: {mirror_manifest}")
    mirror_state = json.loads(mirror_manifest.read_text(encoding="utf-8"))
    partial_mirror = mirror_state.get("state") != "complete"
    if (
        not partial_mirror
        and mirror_state.get("kind") == "crossres-sparse-zarr-mirror"
    ):
        mirror_validation = validate_sparse_mirror(input_path)
        print(
            "validated fine mirror: "
            f"{mirror_validation['count']:,} objects, "
            f"{mirror_validation['bytes']:,} bytes, "
            f"plan={mirror_validation['plan_sha256']}",
            flush=True,
        )
    device = torch.device(options.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    assert_cuda_power_limit(device)
    volume = open_volume(fine_volume)
    raw_chunks = getattr(volume, "chunks", None)
    if raw_chunks is None or len(raw_chunks) != 3:
        raise ValueError("fine input must expose three-dimensional Zarr chunks")
    shape = tuple(int(item) for item in volume.shape)
    chunks = tuple(int(item) for item in raw_chunks)
    if any(chunk <= 0 for chunk in chunks):
        raise ValueError("fine input chunk dimensions must be positive")
    centered_crop_slices(options.input_shape_zyx, chunks)
    margin = tuple(
        (outer - inner) // 2
        for outer, inner in zip(options.input_shape_zyx, chunks, strict=True)
    )
    context = options.input_shape_zyx if options.sliding_blend else margin
    radius = tuple(
        math.ceil(value / chunk) for value, chunk in zip(context, chunks, strict=True)
    )
    grid = tuple(
        math.ceil(size / chunk) for size, chunk in zip(shape, chunks, strict=True)
    )
    partial_support_validation = None
    if partial_mirror:
        present, candidates, partial_support_validation = (
            validate_local_support_snapshot(
                input_path=input_path,
                array_key=array_key,
                support_path=support_path,
                shape_zyx=shape,
                chunks_zyx=chunks,
                grid_zyx=grid,
                radius_zyx=radius,
                required_chunks=options.chunks,
            )
        )
    else:
        present = load_present_chunk_coordinates(
            support_path, array_key=array_key, grid_zyx=grid
        )
        candidates = interior_chunk_coordinates(
            present, grid_zyx=grid, radius_zyx=radius
        )
    full_chunk = ((candidates + 1) * np.asarray(chunks) <= np.asarray(shape)).all(
        axis=1
    )
    candidates = candidates[full_chunk]
    if candidates.size == 0:
        raise RuntimeError("fine mirror has no fully supported teacher neighborhoods")
    eligible_chunks = int(candidates.shape[0])
    if options.candidate_chunk_zyx is not None:
        requested = np.asarray(options.candidate_chunk_zyx, dtype=np.int64)
        matches = np.all(candidates == requested[None], axis=1)
        if not np.any(matches):
            raise ValueError(
                f"requested candidate chunk {options.candidate_chunk_zyx} "
                "does not have complete mirrored context"
            )
        candidates = requested[None]
        effective_chunks = 1
    else:
        effective_chunks = _effective_teacher_chunk_target(
            eligible_chunks, options
        )
        if effective_chunks < options.chunks:
            print(
                f"teacher support ceiling: using all {effective_chunks:,} full "
                f"neighborhoods below the requested cap of {options.chunks:,}",
                flush=True,
            )
        if options.max_candidates < effective_chunks:
            raise ValueError(
                f"max_candidates {options.max_candidates:,} cannot satisfy "
                f"{effective_chunks:,} effective chunks"
            )
        candidates = _candidate_execution_order(
            candidates,
            seed=options.seed,
            tile_chunks=options.candidate_tile_chunks,
        )
        candidates = candidates[: min(options.max_candidates, candidates.shape[0])]

    teacher = load_teacher_checkpoint(
        checkpoint_path, villa_source=villa_source, device=device
    )
    if options.input_shape_zyx != teacher.patch_shape_zyx:
        raise ValueError(
            f"teacher checkpoint requires {teacher.patch_shape_zyx}, "
            f"but --input-shape is {options.input_shape_zyx}"
        )
    if any(size % teacher.required_divisor for size in options.input_shape_zyx):
        raise ValueError(
            f"teacher input must be divisible by {teacher.required_divisor}"
        )

    identity = {
        "schema": "crossres-native-fine-teacher-v1",
        "fine_volume": str(input_path) + f"::{array_key}",
        "fine_support_inventory": str(support_path),
        "fine_support_sha256": _sha256(support_path),
        "teacher_checkpoint": str(checkpoint_path),
        "teacher_checkpoint_sha256": _sha256(checkpoint_path),
        "teacher_kind": teacher.kind,
        "teacher_provenance": teacher.provenance,
        "options": asdict(options),
        "effective_chunks": effective_chunks,
        "fine_shape_zyx": list(shape),
        "fine_chunks_zyx": list(chunks),
        "eligible_chunks": eligible_chunks,
    }
    if partial_support_validation is not None:
        identity["partial_support_validation"] = partial_support_validation
    # JSON has no tuple type. Canonicalize before both comparison and writing so
    # a killed process can resume instead of seeing tuple/list drift.
    identity = json.loads(json.dumps(identity, sort_keys=True))
    state_path = output / "teacher_state.json"
    if output.exists():
        if not state_path.is_file():
            raise ValueError(f"{output}: existing output has no teacher state")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        stored_identity = _canonical_teacher_identity(state.get("identity"))
        expected_identity = _canonical_teacher_identity(identity)
        if stored_identity != expected_identity:
            differences = _identity_difference_paths(stored_identity, expected_identity)
            raise ValueError(
                f"{output}: teacher materialization identity changed at {differences}"
            )
    else:
        _initialize_target_zarr(
            output,
            shape_zyx=shape,
            chunks_zyx=chunks,
            attrs={
                "kind": "crossres-native-fine-teacher",
                "encoding": "labels",
                "positive_label": 255,
                "threshold": options.threshold,
                "model": teacher.kind,
                "teacher_checkpoint_sha256": identity["teacher_checkpoint_sha256"],
            },
        )
        _atomic_json(
            state_path,
            {
                "state": "materializing",
                "identity": identity,
                "accepted": 0,
                "examined": 0,
            },
        )

    records = _existing_records(output, chunks)
    accepted = len(records)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    examined = int(state.get("examined", 0))
    if state.get("state") == "complete":
        _validate_complete_teacher_output(output, state, records)
        return output
    if accepted >= effective_chunks:
        _write_inventory(output, records)
        _atomic_json(
            state_path,
            {
                "state": "complete",
                "identity": identity,
                "accepted": accepted,
                "examined": examined,
            },
        )
        return output

    amp_dtype, autocast_enabled = _amp_configuration(options.amp_dtype, device)
    crop_key = centered_crop_slices(options.input_shape_zyx, chunks)
    prediction_cache = (
        _PredictionLogitCache(options.prediction_cache_entries)
        if options.prediction_cache_entries
        else None
    )
    accepted_at_start = accepted
    started = time.perf_counter()
    for candidate_index in range(examined, candidates.shape[0]):
        coordinate = tuple(int(item) for item in candidates[candidate_index])
        examined = candidate_index + 1
        if coordinate in records:
            continue
        chunk_origin = np.asarray(coordinate) * np.asarray(chunks)
        output_origin = tuple(int(item) for item in chunk_origin)
        output_raw = read_crop(
            volume,
            output_origin,
            chunks,
        )
        ct_nonzero_fraction = float(np.count_nonzero(output_raw)) / output_raw.size
        if ct_nonzero_fraction < options.min_ct_nonzero_fraction:
            continue
        if options.sliding_blend:
            output_probability, inference_origins = _predict_blended_probability(
                teacher=teacher,
                volume=volume,
                volume_shape_zyx=shape,
                output_origin_zyx=output_origin,
                output_shape_zyx=chunks,
                step_size=options.sliding_step_size,
                device=device,
                amp_dtype=amp_dtype,
                autocast_enabled=autocast_enabled,
                mirror_tta=options.mirror_tta,
                inference_batch_size=options.inference_batch_size,
                prediction_cache=prediction_cache,
            )
            input_origin_value = None
        else:
            input_origin = chunk_origin - np.asarray(margin)
            raw = read_crop(
                volume,
                tuple(int(item) for item in input_origin),
                options.input_shape_zyx,
            )
            image = torch.from_numpy(normalize_teacher_ct(raw, teacher.normalization))[
                None, None
            ].to(device, non_blocking=True)
            probability = (
                _predict_probability(
                    teacher.model,
                    image,
                    amp_dtype=amp_dtype,
                    autocast_enabled=autocast_enabled,
                    mirror_tta=options.mirror_tta,
                    average_logits=teacher.tta_average_logits,
                )[0]
                .cpu()
                .numpy()
            )
            output_probability = np.asarray(probability[crop_key], dtype=np.float32)
            inference_origins = [tuple(int(item) for item in input_origin)]
            input_origin_value = [int(item) for item in input_origin]
        target = (output_probability >= options.threshold).astype(np.uint8) * 255
        positive_voxels = int(np.count_nonzero(target))
        if positive_voxels < options.min_positive_voxels:
            if examined % 25 == 0:
                _atomic_json(
                    state_path,
                    {
                        "state": "materializing",
                        "identity": identity,
                        "accepted": accepted,
                        "examined": examined,
                    },
                )
            continue
        chunk_path, encoded_bytes, encoded_sha256 = _write_chunk_atomic(
            output, coordinate, target
        )
        record = {
            "schema": "crossres-native-fine-teacher-chunk-v1",
            "chunk_zyx": list(coordinate),
            "origin_zyx": [int(item) for item in chunk_origin],
            "shape_zyx": list(chunks),
            "input_origin_zyx": input_origin_value,
            "input_shape_zyx": list(options.input_shape_zyx),
            "inference_patch_origins_zyx": [
                list(origin) for origin in inference_origins
            ],
            "inference_patch_count": len(inference_origins),
            "ct_nonzero_fraction": ct_nonzero_fraction,
            "positive_voxels": positive_voxels,
            "positive_fraction": positive_voxels / target.size,
            "probability_min": float(output_probability.min()),
            "probability_max": float(output_probability.max()),
            "probability_mean": float(output_probability.mean()),
            "threshold": options.threshold,
            "encoded_bytes": encoded_bytes,
            "encoded_sha256": encoded_sha256,
            "relative_path": chunk_path.relative_to(output).as_posix(),
            "candidate_index": candidate_index,
            "teacher_kind": teacher.kind,
        }
        record_path = (
            output / "records" / ("_".join(str(item) for item in coordinate) + ".json")
        )
        _atomic_json(record_path, record)
        records[coordinate] = record
        accepted += 1
        _atomic_json(
            state_path,
            {
                "state": "materializing",
                "identity": identity,
                "accepted": accepted,
                "examined": examined,
            },
        )
        elapsed = max(time.perf_counter() - started, 1.0e-6)
        cache_text = ""
        if prediction_cache is not None:
            accesses = prediction_cache.hits + prediction_cache.misses
            hit_rate = prediction_cache.hits / max(accesses, 1)
            cache_text = (
                f", cache_hit={hit_rate:.1%}, windows={prediction_cache.computed:,}, "
                f"cache_entries={prediction_cache.entries:,}, "
                f"cache_evictions={prediction_cache.evictions:,}"
            )
        newly_accepted = accepted - accepted_at_start
        print(
            f"teacher {accepted:,}/{effective_chunks:,}: chunk={coordinate}, "
            f"positive={positive_voxels:,}, examined={examined:,}, "
            f"rate={newly_accepted / elapsed:.3f} accepted/s{cache_text}",
            flush=True,
        )
        if accepted >= effective_chunks:
            break
    if accepted < effective_chunks:
        if not options.allow_fewer_chunks or accepted <= 0:
            raise RuntimeError(
                f"only {accepted:,}/{effective_chunks:,} teacher chunks accepted after "
                f"examining {examined:,} candidates"
            )
        print(
            f"teacher label ceiling: accepted {accepted:,}/{effective_chunks:,} "
            f"after exhausting {examined:,} candidates; "
            f"{examined - accepted:,} failed CT/positive-label filters",
            flush=True,
        )
    _write_inventory(output, records)
    _atomic_json(
        state_path,
        {
            "state": "complete",
            "identity": identity,
            "accepted": accepted,
            "examined": examined,
        },
    )
    return output


def materialize_m7_teacher(
    *,
    fine_volume: str,
    fine_support_inventory: str | Path,
    output_path: str | Path,
    m7_checkpoint: str | Path,
    options: TeacherOptions,
) -> Path:
    """Backward-compatible diagnostic alias; production uses materialize_teacher."""

    return materialize_teacher(
        fine_volume=fine_volume,
        fine_support_inventory=fine_support_inventory,
        output_path=output_path,
        teacher_checkpoint=m7_checkpoint,
        villa_source=".",
        options=options,
    )
