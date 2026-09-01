from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from crossres_pred.resample import BridgeOptions

from .io import open_volume, split_volume_spec
from .medial import (
    MEDIAL_MAX_PROJECTION_CONTRACT,
    VILLA_MEDIAL_SURFACE_CONTRACT,
    VILLA_MEDIAL_SURFACE_SOURCE_COMMIT,
    VILLA_MEDIAL_SURFACE_SOURCE_SHA256,
    FineMedialSurfaceReader,
    MedialProjectionOptions,
    medial_provenance,
    project_fine_medial_patch,
)
from .registration import (
    ChunkSupport,
    FineFieldWindowReader,
    affine_matrix,
    antialias_fine_target_patch,
    transform_xyz,
)
from .resources import configure_cpu_budget
from .schema import VoxelPairRecord, load_pair_manifest

ATLAS_SCHEMA = "crossres-coarse-teacher-atlas-v1"
ATLAS_STATE_SCHEMA = "crossres-coarse-teacher-atlas-state-v1"
ATLAS_TILE_SCHEMA = "crossres-coarse-teacher-atlas-tile-v1"
ATLAS_PROJECTION_CONTRACT = "antialias-pullback-gh3-coarse-atlas-v2"
ATLAS_TILE_COMMIT_BATCH_SIZE = 128
MEDIAL_ATLAS_SCHEMA = "crossres-coarse-teacher-medial-atlas-v1"
MEDIAL_ATLAS_STATE_SCHEMA = "crossres-coarse-teacher-medial-atlas-state-v1"
MEDIAL_ATLAS_TILE_SCHEMA = "crossres-coarse-teacher-medial-atlas-tile-v1"
DEFAULT_BRIDGE = BridgeOptions(
    prefilter_sigma_scale=0.5,
    coverage_erosion_fine_vox=0,
    max_fine_window_vox=352,
    maxpool_prefilter=False,
    erode_filter_margin=True,
)


@dataclass(frozen=True)
class CoarseTeacherAtlasOptions:
    tile_shape_zyx: tuple[int, int, int] = (64, 64, 64)
    candidate_margin_coarse_vox: int = 3
    hard_threshold: float = 0.5
    max_cpu_threads: int = 16
    fine_chunk_cache_entries: int = 256
    require_cuda: bool = True
    bridge: BridgeOptions = DEFAULT_BRIDGE

    def validate(self) -> None:
        if len(self.tile_shape_zyx) != 3 or any(
            value <= 0 or value % 8 for value in self.tile_shape_zyx
        ):
            raise ValueError(
                "atlas tile dimensions must be positive multiples of eight"
            )
        if self.candidate_margin_coarse_vox < 0:
            raise ValueError("candidate margin cannot be negative")
        if not 0.0 <= self.hard_threshold <= 1.0:
            raise ValueError("hard threshold must be in [0, 1]")
        if not 1 <= self.max_cpu_threads <= 16:
            raise ValueError("max_cpu_threads must be in [1, 16]")
        if self.fine_chunk_cache_entries <= 0:
            raise ValueError("fine_chunk_cache_entries must be positive")
        self.bridge.validate()
        if self.bridge != DEFAULT_BRIDGE:
            raise ValueError("coarse teacher atlases require the pinned GH3 bridge")


@dataclass(frozen=True)
class CoarseTeacherMedialAtlasOptions:
    max_cpu_threads: int = 16
    fine_chunk_cache_entries: int = 256
    medial: MedialProjectionOptions = field(default_factory=MedialProjectionOptions)

    def validate(self) -> None:
        if not 1 <= self.max_cpu_threads <= 16:
            raise ValueError("max_cpu_threads must be in [1, 16]")
        if self.fine_chunk_cache_entries <= 0:
            raise ValueError("fine_chunk_cache_entries must be positive")
        self.medial.validate()
        if self.medial.skeleton_workers > self.max_cpu_threads:
            raise ValueError("skeleton workers exceed the total CPU budget")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Tolerate brief Windows sharing locks from concurrent atlas-state readers."""

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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(f"{path}: expected JSON objects")
    return rows


def _morton_key(coordinate_zyx: tuple[int, int, int]) -> int:
    result = 0
    for bit in range(21):
        result |= ((coordinate_zyx[2] >> bit) & 1) << (3 * bit)
        result |= ((coordinate_zyx[1] >> bit) & 1) << (3 * bit + 1)
        result |= ((coordinate_zyx[0] >> bit) & 1) << (3 * bit + 2)
    return result


def _inventory_coordinates(path: Path, array_key: str) -> list[tuple[int, int, int]]:
    prefix = tuple(part for part in array_key.split("/") if part)
    coordinates: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") != "chunk":
                continue
            parts = tuple(str(row["relative_path"]).replace("\\", "/").split("/"))
            if prefix and parts[: len(prefix)] != prefix:
                raise ValueError(f"{path}:{line_number}: chunk uses another array key")
            raw = parts[len(prefix) :]
            if len(raw) != 3:
                raise ValueError(f"{path}:{line_number}: expected a 3-D chunk path")
            coordinate = tuple(int(value) for value in raw)
            if any(value < 0 for value in coordinate) or coordinate in seen:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate chunk")
            seen.add(coordinate)
            coordinates.append(coordinate)  # type: ignore[arg-type]
    if not coordinates:
        raise ValueError(f"{path}: sparse inventory contains no chunks")
    return coordinates


def _candidate_coordinates(path: Path) -> list[tuple[int, int, int]]:
    coordinates: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get("accepted", True) is not True:
                raise ValueError(
                    f"{path}:{line_number}: candidate fine chunk is not accepted"
                )
            raw = (
                row.get("chunk_zyx")
                if isinstance(row, dict)
                else row
                if isinstance(row, list)
                else None
            )
            try:
                coordinate = tuple(int(value) for value in raw)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid candidate fine chunk"
                ) from error
            if (
                len(coordinate) != 3
                or any(value < 0 for value in coordinate)
                or coordinate in seen
            ):
                raise ValueError(
                    f"{path}:{line_number}: invalid or duplicate candidate chunk"
                )
            seen.add(coordinate)
            coordinates.append(coordinate)  # type: ignore[arg-type]
    if not coordinates:
        raise ValueError(f"{path}: no candidate fine chunks")
    return coordinates


def candidate_coarse_tiles(
    *,
    fine_chunk_coordinates_zyx: list[tuple[int, int, int]],
    fine_chunks_zyx: tuple[int, int, int],
    fine_shape_zyx: tuple[int, int, int],
    fine_to_coarse_affine_xyz: tuple[tuple[float, float, float, float], ...],
    coarse_shape_zyx: tuple[int, int, int],
    tile_shape_zyx: tuple[int, int, int],
    margin_coarse_vox: int,
) -> list[tuple[int, int, int]]:
    """Return every coarse tile that can contain projected sparse support."""

    chunks = np.asarray(fine_chunks_zyx, dtype=np.int64)
    fine_shape = np.asarray(fine_shape_zyx, dtype=np.int64)
    coarse_shape = np.asarray(coarse_shape_zyx, dtype=np.int64)
    tile_shape = np.asarray(tile_shape_zyx, dtype=np.int64)
    affine = affine_matrix(fine_to_coarse_affine_xyz)
    tiles: set[tuple[int, int, int]] = set()
    for coordinate in fine_chunk_coordinates_zyx:
        lower = np.asarray(coordinate, dtype=np.int64) * chunks
        upper = np.minimum(lower + chunks, fine_shape) - 1
        if (upper < lower).any():
            continue
        fine_corners_zyx = np.asarray(
            list(
                product(
                    (lower[0], upper[0]),
                    (lower[1], upper[1]),
                    (lower[2], upper[2]),
                )
            ),
            dtype=np.float64,
        )
        coarse_corners_zyx = transform_xyz(fine_corners_zyx[:, ::-1], affine)[:, ::-1]
        coarse_lower = (
            np.floor(coarse_corners_zyx.min(axis=0)).astype(np.int64)
            - margin_coarse_vox
        )
        coarse_upper = (
            np.ceil(coarse_corners_zyx.max(axis=0)).astype(np.int64)
            + margin_coarse_vox
            + 1
        )
        coarse_lower = np.maximum(coarse_lower, 0)
        coarse_upper = np.minimum(coarse_upper, coarse_shape)
        if (coarse_upper <= coarse_lower).any():
            continue
        tile_lower = np.floor_divide(coarse_lower, tile_shape)
        tile_upper = np.floor_divide(coarse_upper - 1, tile_shape)
        for z in range(int(tile_lower[0]), int(tile_upper[0]) + 1):
            for y in range(int(tile_lower[1]), int(tile_upper[1]) + 1):
                for x in range(int(tile_lower[2]), int(tile_upper[2]) + 1):
                    tiles.add((z, y, x))
    return sorted(tiles, key=lambda value: (_morton_key(value), value))


def _array_metadata_path(volume_spec: str) -> Path:
    root, key = split_volume_spec(volume_spec)
    return root.joinpath(*((key or "0").split("/"))) / ".zarray"


def _open_or_create_atlas_array(
    path: Path,
    *,
    shape_zyx: tuple[int, int, int],
    chunks_zyx: tuple[int, int, int],
):
    import zarr
    from numcodecs import Blosc

    if path.exists():
        array = zarr.open_array(str(path), mode="r+")
        if (
            tuple(int(value) for value in array.shape) != shape_zyx
            or tuple(int(value) for value in array.chunks) != chunks_zyx
            or np.dtype(array.dtype) != np.dtype(np.uint8)
        ):
            raise ValueError(f"existing atlas array identity differs: {path}")
        return array
    return zarr.open_array(
        str(path),
        mode="w",
        zarr_format=2,
        shape=shape_zyx,
        chunks=chunks_zyx,
        dtype=np.uint8,
        fill_value=0,
        compressor=Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE),
        dimension_separator="/",
    )


def _pair_record(path: Path, record_id: str) -> VoxelPairRecord:
    matches = [
        record for record in load_pair_manifest(path) if record.record_id == record_id
    ]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected exactly one record {record_id!r}")
    return matches[0]


def _qualified_fine_support(
    record: VoxelPairRecord,
    fine_volume: Any,
    candidate_path: Path | None,
) -> ChunkSupport:
    """Recreate the exact support allow-list bound into a parent atlas."""

    support = ChunkSupport.from_field(record.fine.target, fine_volume)
    if candidate_path is None:
        return support
    candidate_coordinates = _candidate_coordinates(candidate_path)
    if any(not support.contains(value) for value in candidate_coordinates):
        raise ValueError("candidate fine chunks are absent from decoded support")
    candidate_ids = np.unique(
        np.asarray(
            [support.encode(value) for value in candidate_coordinates],
            dtype=np.int64,
        )
    )
    return ChunkSupport(
        shape_zyx=support.shape_zyx,
        chunks_zyx=support.chunks_zyx,
        grid_zyx=support.grid_zyx,
        present_ids=candidate_ids,
        sampling_ids=candidate_ids,
    )


def build_coarse_teacher_atlas(
    *,
    pair_manifest_path: str | Path,
    record_id: str,
    output_path: str | Path,
    options: CoarseTeacherAtlasOptions | None = None,
    maximum_tiles: int | None = None,
    candidate_fine_chunks_path: str | Path | None = None,
) -> Path:
    """Project a sparse fine teacher once into a sparse coarse q/valid atlas."""

    options = options or CoarseTeacherAtlasOptions()
    options.validate()
    configure_cpu_budget(options.max_cpu_threads)
    pair_manifest = Path(pair_manifest_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    record = _pair_record(pair_manifest, record_id)
    fine_root, fine_key = split_volume_spec(record.fine.target.volume)
    array_key = fine_key or "0"
    support_spec = record.fine.target.support
    if support_spec is None or support_spec.kind != "present-chunks":
        raise ValueError("coarse teacher atlas requires sparse present-chunk support")
    inventory = Path(support_spec.inventory).expanduser().resolve()
    if not inventory.is_file():
        raise FileNotFoundError(inventory)
    metadata_path = _array_metadata_path(record.fine.target.volume)
    metadata = _read_json(metadata_path)
    fine_shape = tuple(int(value) for value in metadata["shape"])
    fine_chunks = tuple(int(value) for value in metadata["chunks"])
    if len(fine_shape) != 3 or len(fine_chunks) != 3:
        raise ValueError("fine target must be a three-dimensional Zarr array")
    fine_coordinates = _inventory_coordinates(inventory, array_key)
    candidate_path = (
        Path(candidate_fine_chunks_path).expanduser().resolve()
        if candidate_fine_chunks_path is not None
        else None
    )
    candidate_coordinates = (
        _candidate_coordinates(candidate_path)
        if candidate_path is not None
        else fine_coordinates
    )
    unknown_candidates = sorted(set(candidate_coordinates) - set(fine_coordinates))
    if unknown_candidates:
        raise ValueError(
            f"candidate chunks are absent from fine support: {unknown_candidates[:5]}"
        )
    teacher_state_path = fine_root / "teacher_state.json"
    teacher_state_identity: dict[str, Any] | None = None
    if teacher_state_path.is_file():
        teacher_state = _read_json(teacher_state_path)
        if teacher_state.get("state") != "complete" or int(
            teacher_state.get("accepted", -1)
        ) != len(fine_coordinates):
            raise ValueError(f"fine teacher is not a complete commit: {fine_root}")
        teacher_state_identity = {
            "path": str(teacher_state_path),
            "sha256": _sha256(teacher_state_path),
        }

    coarse_volume = open_volume(record.coarse.image)
    coarse_shape = tuple(int(value) for value in coarse_volume.shape)
    tiles = candidate_coarse_tiles(
        fine_chunk_coordinates_zyx=candidate_coordinates,
        fine_chunks_zyx=fine_chunks,  # type: ignore[arg-type]
        fine_shape_zyx=fine_shape,  # type: ignore[arg-type]
        fine_to_coarse_affine_xyz=record.fine.to_coarse_affine_xyz,
        coarse_shape_zyx=coarse_shape,  # type: ignore[arg-type]
        tile_shape_zyx=options.tile_shape_zyx,
        margin_coarse_vox=options.candidate_margin_coarse_vox,
    )
    if maximum_tiles is not None:
        if maximum_tiles <= 0:
            raise ValueError("maximum_tiles must be positive")
        tiles = tiles[:maximum_tiles]
    if not tiles:
        raise ValueError("fine support does not intersect the coarse volume")

    # State is compared after a reboot, when JSON has converted every tuple to
    # a list. Canonicalize before the first comparison/write so an atlas accepts
    # its own persisted identity instead of failing every resume.
    identity = json.loads(
        json.dumps(
            {
                "schema": ATLAS_SCHEMA,
                "projection_contract": ATLAS_PROJECTION_CONTRACT,
                "pair_manifest": str(pair_manifest),
                "pair_manifest_sha256": _sha256(pair_manifest),
                "record_id": record.record_id,
                "scroll_id": record.scroll_id,
                "coarse_image": record.coarse.image,
                "coarse_shape_zyx": list(coarse_shape),
                "fine_target": record.fine.target.volume,
                "fine_target_metadata": str(metadata_path),
                "fine_target_metadata_sha256": _sha256(metadata_path),
                "fine_support_inventory": str(inventory),
                "fine_support_inventory_sha256": _sha256(inventory),
                "fine_support_chunks": len(fine_coordinates),
                "candidate_fine_chunks": len(candidate_coordinates),
                "candidate_fine_chunks_path": (
                    str(candidate_path) if candidate_path else None
                ),
                "candidate_fine_chunks_sha256": (
                    _sha256(candidate_path) if candidate_path is not None else None
                ),
                "fine_support_policy": (
                    "candidate-chunks-only"
                    if candidate_path is not None
                    else "all-present-chunks"
                ),
                "teacher_state": teacher_state_identity,
                "fine_to_coarse_affine_xyz": [
                    list(row) for row in record.fine.to_coarse_affine_xyz
                ],
                "options": {**asdict(options), "bridge": asdict(options.bridge)},
                "candidate_tiles": len(tiles),
                "maximum_tiles": maximum_tiles,
                "tile_commit_batch_size": ATLAS_TILE_COMMIT_BATCH_SIZE,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    rows_dir = output / "rows"
    rows_dir.mkdir(exist_ok=True)
    state_path = output / "atlas_state.json"
    if state_path.is_file():
        state = _read_json(state_path)
        if state.get("identity") != identity:
            raise ValueError("existing coarse teacher atlas identity differs")
    q_array = _open_or_create_atlas_array(
        output / "teacher_q.zarr",
        shape_zyx=coarse_shape,  # type: ignore[arg-type]
        chunks_zyx=options.tile_shape_zyx,
    )
    valid_array = _open_or_create_atlas_array(
        output / "target_valid.zarr",
        shape_zyx=coarse_shape,  # type: ignore[arg-type]
        chunks_zyx=options.tile_shape_zyx,
    )
    completed: dict[int, dict[str, Any]] = {}
    for path in sorted(rows_dir.glob("*.jsonl")):
        batch_index = int(path.stem)
        expected_start = batch_index * ATLAS_TILE_COMMIT_BATCH_SIZE
        expected_end = min(
            expected_start + ATLAS_TILE_COMMIT_BATCH_SIZE,
            len(tiles),
        )
        rows = _read_jsonl(path)
        indices = [int(row.get("index", -1)) for row in rows]
        if indices != list(range(expected_start, expected_end)):
            raise ValueError(f"invalid atlas tile batch sidecar: {path}")
        for index, row in zip(indices, rows, strict=True):
            coordinate = tiles[index]
            if (
                index in completed
                or tuple(row.get("tile_coordinate_zyx", ())) != coordinate
            ):
                raise ValueError(f"invalid atlas tile batch row: {path}:{index}")
            completed[index] = row

    fine_volume = open_volume(record.fine.target.volume)
    support = _qualified_fine_support(record, fine_volume, candidate_path)
    # The published Paris teacher is much larger than the quality-qualified
    # anchor set. Candidate chunks are a supervision allow-list, not merely a
    # way to choose output tiles: leaking neighbouring unqualified chunks into
    # q/valid would silently defeat the teacher quality gate.
    if candidate_path is not None and int(support.present_ids.size) != len(
        candidate_coordinates
    ):
        raise ValueError("candidate fine chunk allow-list changed during build")
    reader = FineFieldWindowReader(
        fine_volume,
        record.fine.target,
        support,
        max_cache_chunks=options.fine_chunk_cache_entries,
    )
    started = time.perf_counter()
    initially_completed = len(completed)
    tile_shape = np.asarray(options.tile_shape_zyx, dtype=np.int64)
    coarse_shape_array = np.asarray(coarse_shape, dtype=np.int64)
    for batch_start in range(0, len(tiles), ATLAS_TILE_COMMIT_BATCH_SIZE):
        batch_end = min(batch_start + ATLAS_TILE_COMMIT_BATCH_SIZE, len(tiles))
        batch_indices = range(batch_start, batch_end)
        present_indices = [index for index in batch_indices if index in completed]
        if len(present_indices) == batch_end - batch_start:
            continue
        if present_indices:
            raise ValueError("atlas tile commit batch is only partially present")
        batch_rows: list[dict[str, Any]] = []
        for index in batch_indices:
            coordinate = tiles[index]
            origin = np.asarray(coordinate, dtype=np.int64) * tile_shape
            shape = np.minimum(tile_shape, coarse_shape_array - origin)
            if (shape <= 0).any():
                raise AssertionError(
                    "candidate atlas tile lies outside the coarse volume"
                )
            _, q, valid_u8, stats = antialias_fine_target_patch(
                fine_volume,
                record.fine.target,
                support,
                record.fine.to_coarse_affine_xyz,
                tuple(int(value) for value in origin),
                tuple(int(value) for value in shape),
                options=options.bridge,
                reader=reader,
                hard_threshold=options.hard_threshold,
                cuda_minimum_output_voxels=1,
            )
            required_backend = "cuda-gauss-hermite3-pullback-linf-validity-v1"
            if (
                options.require_cuda
                and stats.get("projection_backend") != required_backend
            ):
                raise RuntimeError(
                    f"atlas tile requires {required_backend}, got "
                    f"{stats.get('projection_backend')}"
                )
            q_u8 = np.rint(np.clip(q, 0.0, 1.0) * 255.0).astype(np.uint8)
            slices = tuple(
                slice(int(lo), int(lo + size))
                for lo, size in zip(origin, shape, strict=True)
            )
            present = bool(np.any(valid_u8))
            if present:
                q_array[slices] = q_u8
                valid_array[slices] = valid_u8.astype(np.uint8, copy=False)
            batch_rows.append(
                {
                    "schema": ATLAS_TILE_SCHEMA,
                    "index": index,
                    "tile_coordinate_zyx": list(coordinate),
                    "origin_zyx": origin.tolist(),
                    "shape_zyx": shape.tolist(),
                    "present": present,
                    "known_voxels": int(np.count_nonzero(valid_u8)),
                    "positive_voxels": int(
                        np.count_nonzero((q_u8 >= 128) & (valid_u8 > 0))
                    ),
                    "stats": stats,
                }
            )
        batch_path = rows_dir / (
            f"{batch_start // ATLAS_TILE_COMMIT_BATCH_SIZE:06d}.jsonl"
        )
        _atomic_jsonl(batch_path, batch_rows)
        completed.update({int(row["index"]): row for row in batch_rows})
        elapsed = max(time.perf_counter() - started, 1.0e-6)
        newly_completed = len(completed) - initially_completed
        _atomic_json(
            state_path,
            {
                "schema": ATLAS_STATE_SCHEMA,
                "state": "running",
                "completed": len(completed),
                "expected": len(tiles),
                "tiles_per_second": newly_completed / elapsed,
                "identity": identity,
            },
        )
        print(
            f"coarse teacher atlas {len(completed):,}/{len(tiles):,} "
            f"committed_batch={batch_start // ATLAS_TILE_COMMIT_BATCH_SIZE:,}",
            flush=True,
        )

    ordered = [completed[index] for index in range(len(tiles))]
    inventory_path = output / "atlas_tiles.jsonl"
    _atomic_jsonl(inventory_path, ordered)
    present_tiles = sum(bool(row["present"]) for row in ordered)
    known_voxels = sum(int(row["known_voxels"]) for row in ordered)
    positive_voxels = sum(int(row["positive_voxels"]) for row in ordered)
    state = {
        "schema": ATLAS_STATE_SCHEMA,
        "state": "complete",
        "completed": len(tiles),
        "expected": len(tiles),
        "present_tiles": present_tiles,
        "known_voxels": known_voxels,
        "positive_voxels": positive_voxels,
        "identity": identity,
        "teacher_q": str((output / "teacher_q.zarr").resolve()),
        "target_valid": str((output / "target_valid.zarr").resolve()),
        "tile_inventory": str(inventory_path.resolve()),
        "tile_inventory_sha256": _sha256(inventory_path),
    }
    _atomic_json(state_path, state)
    return state_path


def validate_coarse_teacher_atlas(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    state_path = root / "atlas_state.json"
    state = _read_json(state_path)
    identity = state.get("identity")
    inventory = root / "atlas_tiles.jsonl"
    if (
        state.get("schema") != ATLAS_STATE_SCHEMA
        or state.get("state") != "complete"
        or not isinstance(identity, dict)
        or identity.get("schema") != ATLAS_SCHEMA
        or not inventory.is_file()
        or state.get("tile_inventory_sha256") != _sha256(inventory)
        or not (root / "teacher_q.zarr").is_dir()
        or not (root / "target_valid.zarr").is_dir()
    ):
        raise ValueError(f"coarse teacher atlas is not a complete commit: {root}")
    rows = [
        json.loads(line) for line in inventory.read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != int(state.get("completed", -1)):
        raise ValueError("atlas tile inventory count differs from completion state")
    return state


def build_coarse_teacher_medial_atlas(
    *,
    atlas_path: str | Path,
    options: CoarseTeacherMedialAtlasOptions | None = None,
) -> Path:
    """Add a provenance-bound fine-medial sidecar to a completed q atlas.

    The parent occupancy atlas remains immutable.  This stage reconstructs the
    exact Villa slice-wise medial surface in native-fine label space, projects
    its binary indicator with an OR/max pullback, and commits a separate
    per-voxel validity field.  Resume is batch-atomic in the same way as q.
    """

    options = options or CoarseTeacherMedialAtlasOptions()
    options.validate()
    configure_cpu_budget(options.max_cpu_threads)
    root = Path(atlas_path).expanduser().resolve()
    parent = validate_coarse_teacher_atlas(root)
    parent_state_path = root / "atlas_state.json"
    parent_state_sha256 = _sha256(parent_state_path)
    parent_identity = parent["identity"]
    pair_manifest = Path(str(parent_identity["pair_manifest"])).resolve()
    record_id = str(parent_identity["record_id"])
    record = _pair_record(pair_manifest, record_id)
    if (
        record.fine.target.volume != parent_identity["fine_target"]
        or [list(row) for row in record.fine.to_coarse_affine_xyz]
        != parent_identity["fine_to_coarse_affine_xyz"]
    ):
        raise ValueError("parent atlas no longer matches its pair record")

    parent_inventory = Path(str(parent["tile_inventory"])).resolve()
    parent_rows = _read_jsonl(parent_inventory)
    if len(parent_rows) != int(parent["completed"]):
        raise ValueError("parent atlas inventory is incomplete")
    for index, row in enumerate(parent_rows):
        if (
            int(row.get("index", -1)) != index
            or row.get("schema") != ATLAS_TILE_SCHEMA
        ):
            raise ValueError("parent atlas tile inventory order is invalid")

    scientific = medial_provenance(options.medial)
    scientific.pop("skeleton_workers")
    scientific.pop("max_cache_chunks")
    identity = json.loads(
        json.dumps(
            {
                "schema": MEDIAL_ATLAS_SCHEMA,
                "parent_atlas_state": str(parent_state_path),
                "parent_atlas_state_sha256": parent_state_sha256,
                "parent_tile_inventory": str(parent_inventory),
                "parent_tile_inventory_sha256": _sha256(parent_inventory),
                "record_id": record_id,
                "scroll_id": record.scroll_id,
                "fine_target": record.fine.target.volume,
                "fine_to_coarse_affine_xyz": [
                    list(row) for row in record.fine.to_coarse_affine_xyz
                ],
                "scientific_options": scientific,
                "tile_commit_batch_size": ATLAS_TILE_COMMIT_BATCH_SIZE,
                "expected_tiles": len(parent_rows),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )

    rows_dir = root / "medial_rows"
    rows_dir.mkdir(exist_ok=True)
    state_path = root / "medial_state.json"
    if state_path.is_file():
        prior = _read_json(state_path)
        if prior.get("identity") != identity:
            raise ValueError("existing medial atlas identity differs")
        if prior.get("state") == "complete":
            validate_coarse_teacher_medial_atlas(root)
            return state_path

    coarse_shape = tuple(int(value) for value in parent_identity["coarse_shape_zyx"])
    tile_shape = tuple(
        int(value) for value in parent_identity["options"]["tile_shape_zyx"]
    )
    crest_array = _open_or_create_atlas_array(
        root / "teacher_crest.zarr",
        shape_zyx=coarse_shape,
        chunks_zyx=tile_shape,
    )
    crest_valid_array = _open_or_create_atlas_array(
        root / "teacher_crest_valid.zarr",
        shape_zyx=coarse_shape,
        chunks_zyx=tile_shape,
    )
    completed: dict[int, dict[str, Any]] = {}
    for path in sorted(rows_dir.glob("*.jsonl")):
        batch_index = int(path.stem)
        expected_start = batch_index * ATLAS_TILE_COMMIT_BATCH_SIZE
        expected_end = min(
            expected_start + ATLAS_TILE_COMMIT_BATCH_SIZE,
            len(parent_rows),
        )
        rows = _read_jsonl(path)
        indices = [int(row.get("index", -1)) for row in rows]
        if indices != list(range(expected_start, expected_end)):
            raise ValueError(f"invalid medial atlas batch sidecar: {path}")
        for index, row in zip(indices, rows, strict=True):
            if (
                index in completed
                or row.get("schema") != MEDIAL_ATLAS_TILE_SCHEMA
                or row.get("tile_coordinate_zyx")
                != parent_rows[index].get("tile_coordinate_zyx")
            ):
                raise ValueError(f"invalid medial atlas tile row: {path}:{index}")
            completed[index] = row

    candidate_value = parent_identity.get("candidate_fine_chunks_path")
    candidate_path = Path(str(candidate_value)).resolve() if candidate_value else None
    fine_volume = open_volume(record.fine.target.volume)
    support = _qualified_fine_support(record, fine_volume, candidate_path)
    field_reader = FineFieldWindowReader(
        fine_volume,
        record.fine.target,
        support,
        max_cache_chunks=options.fine_chunk_cache_entries,
    )
    started = time.perf_counter()
    initially_completed = len(completed)
    with FineMedialSurfaceReader(field_reader, options=options.medial) as reader:
        for batch_start in range(
            0, len(parent_rows), ATLAS_TILE_COMMIT_BATCH_SIZE
        ):
            batch_end = min(
                batch_start + ATLAS_TILE_COMMIT_BATCH_SIZE,
                len(parent_rows),
            )
            batch_indices = range(batch_start, batch_end)
            present_indices = [index for index in batch_indices if index in completed]
            if len(present_indices) == batch_end - batch_start:
                continue
            if present_indices:
                raise ValueError("medial atlas commit batch is only partially present")
            batch_rows: list[dict[str, Any]] = []
            for index in batch_indices:
                parent_row = parent_rows[index]
                origin = tuple(int(value) for value in parent_row["origin_zyx"])
                shape = tuple(int(value) for value in parent_row["shape_zyx"])
                slices = tuple(
                    slice(lo, lo + size)
                    for lo, size in zip(origin, shape, strict=True)
                )
                crest, crest_valid, stats = project_fine_medial_patch(
                    reader,
                    record.fine.to_coarse_affine_xyz,
                    origin,
                    shape,
                )
                present = bool(np.any(crest_valid))
                if present:
                    crest_array[slices] = crest
                    crest_valid_array[slices] = crest_valid
                batch_rows.append(
                    {
                        "schema": MEDIAL_ATLAS_TILE_SCHEMA,
                        "index": index,
                        "tile_coordinate_zyx": parent_row[
                            "tile_coordinate_zyx"
                        ],
                        "origin_zyx": list(origin),
                        "shape_zyx": list(shape),
                        "present": present,
                        "known_voxels": int(np.count_nonzero(crest_valid)),
                        "crest_voxels": int(np.count_nonzero(crest)),
                        "stats": stats,
                    }
                )
            batch_path = rows_dir / (
                f"{batch_start // ATLAS_TILE_COMMIT_BATCH_SIZE:06d}.jsonl"
            )
            _atomic_jsonl(batch_path, batch_rows)
            completed.update({int(row["index"]): row for row in batch_rows})
            elapsed = max(time.perf_counter() - started, 1.0e-6)
            newly_completed = len(completed) - initially_completed
            _atomic_json(
                state_path,
                {
                    "schema": MEDIAL_ATLAS_STATE_SCHEMA,
                    "state": "running",
                    "completed": len(completed),
                    "expected": len(parent_rows),
                    "tiles_per_second": newly_completed / elapsed,
                    "execution": {
                        "max_cpu_threads": options.max_cpu_threads,
                        "skeleton_workers": options.medial.skeleton_workers,
                        "fine_chunk_cache_entries": (
                            options.fine_chunk_cache_entries
                        ),
                        "medial_chunk_cache_entries": (
                            options.medial.max_cache_chunks
                        ),
                    },
                    "identity": identity,
                },
            )
            print(
                f"coarse medial atlas {len(completed):,}/{len(parent_rows):,} "
                f"committed_batch={batch_start // ATLAS_TILE_COMMIT_BATCH_SIZE:,}",
                flush=True,
            )

    ordered = [completed[index] for index in range(len(parent_rows))]
    inventory_path = root / "medial_tiles.jsonl"
    _atomic_jsonl(inventory_path, ordered)
    state = {
        "schema": MEDIAL_ATLAS_STATE_SCHEMA,
        "state": "complete",
        "completed": len(ordered),
        "expected": len(parent_rows),
        "present_tiles": sum(bool(row["present"]) for row in ordered),
        "known_voxels": sum(int(row["known_voxels"]) for row in ordered),
        "crest_voxels": sum(int(row["crest_voxels"]) for row in ordered),
        "identity": identity,
        "execution": {
            "max_cpu_threads": options.max_cpu_threads,
            "skeleton_workers": options.medial.skeleton_workers,
            "fine_chunk_cache_entries": options.fine_chunk_cache_entries,
            "medial_chunk_cache_entries": options.medial.max_cache_chunks,
        },
        "teacher_crest": str((root / "teacher_crest.zarr").resolve()),
        "teacher_crest_valid": str(
            (root / "teacher_crest_valid.zarr").resolve()
        ),
        "tile_inventory": str(inventory_path.resolve()),
        "tile_inventory_sha256": _sha256(inventory_path),
    }
    _atomic_json(state_path, state)
    return state_path


def validate_coarse_teacher_medial_atlas(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    parent = validate_coarse_teacher_atlas(root)
    state_path = root / "medial_state.json"
    state = _read_json(state_path)
    identity = state.get("identity")
    inventory = root / "medial_tiles.jsonl"
    parent_state_path = root / "atlas_state.json"
    if (
        state.get("schema") != MEDIAL_ATLAS_STATE_SCHEMA
        or state.get("state") != "complete"
        or not isinstance(identity, dict)
        or identity.get("schema") != MEDIAL_ATLAS_SCHEMA
        or identity.get("parent_atlas_state_sha256")
        != _sha256(parent_state_path)
        or identity.get("parent_tile_inventory_sha256")
        != parent.get("tile_inventory_sha256")
        or identity.get("scientific_options", {}).get(
            "medial_surface_contract"
        )
        != VILLA_MEDIAL_SURFACE_CONTRACT
        or identity.get("scientific_options", {}).get("villa_source_commit")
        != VILLA_MEDIAL_SURFACE_SOURCE_COMMIT
        or identity.get("scientific_options", {}).get("villa_source_sha256")
        != VILLA_MEDIAL_SURFACE_SOURCE_SHA256
        or identity.get("scientific_options", {}).get("projection_contract")
        != MEDIAL_MAX_PROJECTION_CONTRACT
        or not inventory.is_file()
        or state.get("tile_inventory_sha256") != _sha256(inventory)
        or not (root / "teacher_crest.zarr").is_dir()
        or not (root / "teacher_crest_valid.zarr").is_dir()
    ):
        raise ValueError(f"coarse teacher medial atlas is not complete: {root}")
    rows = _read_jsonl(inventory)
    if len(rows) != int(state.get("completed", -1)) or len(rows) != int(
        parent["completed"]
    ):
        raise ValueError("medial tile inventory differs from parent completion")
    if any(
        row.get("schema") != MEDIAL_ATLAS_TILE_SCHEMA
        or int(row.get("crest_voxels", -1)) > int(row.get("known_voxels", -1))
        for row in rows
    ):
        raise ValueError("medial tile inventory contains an invalid row")
    crest = open_volume(str(state["teacher_crest"]))
    crest_valid = open_volume(str(state["teacher_crest_valid"]))
    q = open_volume(str(parent["teacher_q"]))
    shapes = {
        tuple(int(value) for value in array.shape)
        for array in (crest, crest_valid, q)
    }
    if len(shapes) != 1:
        raise ValueError("medial and occupancy atlas shapes differ")
    return state
