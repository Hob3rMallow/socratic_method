from __future__ import annotations

import hashlib
import json
import math
import os
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .io import decode_dense_field, open_volume, read_crop, split_volume_spec
from .patches import PATCH_PREPARATION_VERSION, PATCH_SCHEMA
from .registration import (
    ChunkSupport,
    SparseChunkProjectionCache,
    affine_matrix,
    transform_xyz,
    voxelize_fine_target_patch,
)
from .resources import configure_cpu_budget
from .schema import VoxelPairRecord, load_pair_manifest
from .scrollfiesta_metrics import (
    SCROLLFIESTA_PRED_METRICS_CONTRACT,
    ScrollFiestaPredMetrics,
    scrollfiesta_patch_pred_metrics,
)

PREPARE_PROGRESS_DURABILITY_INTERVAL = 16


@dataclass(frozen=True)
class PrepareOptions:
    patches_per_record: int = 64
    patch_shape_zyx: tuple[int, int, int] = (192, 192, 192)
    seed: int = 1203
    min_known_fraction: float = 0.20
    native_teacher_min_known_fraction: float = 0.002
    native_teacher_min_fine_ct_nonzero_fraction: float = 0.95
    min_positive_voxels: int = 32
    min_ct_nonzero_fraction: float = 0.05
    attempts_per_patch: int = 12
    selection_candidates: int = 4
    pathology_fraction: float = 1.0 / 3.0
    positive_density_fraction: float = 1.0 / 6.0
    validity_block: int = 64
    projection_cache_entries: int = 16_384
    max_cpu_threads: int = 16

    def validate(self) -> None:
        if (
            self.patches_per_record <= 0
            or self.attempts_per_patch <= 0
            or self.selection_candidates <= 0
        ):
            raise ValueError("patch counts must be positive")
        if self.selection_candidates > self.attempts_per_patch:
            raise ValueError("selection_candidates cannot exceed attempts_per_patch")
        if len(self.patch_shape_zyx) != 3 or any(
            size <= 0 or size % 32 for size in self.patch_shape_zyx
        ):
            raise ValueError("patch dimensions must be positive multiples of 32")
        if not 0 <= self.min_known_fraction <= 1:
            raise ValueError("min_known_fraction must be in [0, 1]")
        if not 0 <= self.native_teacher_min_known_fraction <= 1:
            raise ValueError(
                "native_teacher_min_known_fraction must be in [0, 1]"
            )
        if not 0 <= self.native_teacher_min_fine_ct_nonzero_fraction <= 1:
            raise ValueError(
                "native_teacher_min_fine_ct_nonzero_fraction must be in [0, 1]"
            )
        if not 0 <= self.min_ct_nonzero_fraction <= 1:
            raise ValueError("min_ct_nonzero_fraction must be in [0, 1]")
        if not 0 <= self.pathology_fraction <= 1:
            raise ValueError("pathology_fraction must be in [0, 1]")
        if not 0 <= self.positive_density_fraction <= 1:
            raise ValueError("positive_density_fraction must be in [0, 1]")
        if self.pathology_fraction + self.positive_density_fraction > 1:
            raise ValueError("ranked sampling fractions cannot sum above 1")
        if self.min_positive_voxels < 0 or self.validity_block <= 0:
            raise ValueError("voxel thresholds and validity_block must be non-negative")
        if self.projection_cache_entries <= 0:
            raise ValueError("projection_cache_entries must be positive")
        if not 1 <= self.max_cpu_threads <= 16:
            raise ValueError("max_cpu_threads must be in [1, 16]")


@dataclass(frozen=True)
class NativeTeacherSupportQuality:
    """Provenance for a native teacher's local fine-CT support gate."""

    min_fine_ct_nonzero_fraction: float
    local_records_applied: bool
    chunks_before: int
    chunks_after: int

    @property
    def chunks_excluded(self) -> int:
        return self.chunks_before - self.chunks_after


@dataclass
class _PrimaryPatchCandidates:
    patch_index: int
    patch_id: str
    rng: np.random.Generator
    strategy: str
    patch_anchor_candidates: np.ndarray | None
    patch_anchor_fallbacks: np.ndarray | None
    acceptable: list[tuple[dict[str, np.ndarray], dict[str, Any]]]
    best: tuple[dict[str, np.ndarray], dict[str, Any]] | None
    candidate_attempts: int


def _bounded_ordered_primary_map(
    function: Callable[[int], _PrimaryPatchCandidates],
    patch_indices: Iterable[int],
    *,
    max_workers: int,
) -> Iterator[_PrimaryPatchCandidates]:
    """Run a bounded number of read-only patch jobs, yielding in index order."""

    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    indices = iter(patch_indices)
    pending: deque[Future[_PrimaryPatchCandidates]] = deque()
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="crossres-patch",
    ) as executor:
        for _ in range(max_workers):
            try:
                patch_index = next(indices)
            except StopIteration:
                break
            pending.append(executor.submit(function, patch_index))
        while pending:
            result = pending.popleft().result()
            try:
                patch_index = next(indices)
            except StopIteration:
                pass
            else:
                pending.append(executor.submit(function, patch_index))
            yield result


def _patch_executor_workers(max_cpu_threads: int) -> int:
    """Overlap chunk I/O without expanding numerical kernels past the CPU budget."""

    return min(16, max(1, max_cpu_threads * 2))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _record_seed(base_seed: int, record_id: str, patch_index: int) -> int:
    digest = hashlib.blake2b(record_id.encode("utf-8"), digest_size=8).digest()
    record_value = int.from_bytes(digest, "little")
    return (base_seed + record_value + patch_index * 1_000_003) % (2**63 - 1)


def _support_anchor_schedule(
    support_coordinates: np.ndarray | None,
    *,
    record_id: str,
    supervision_source: str,
    seed: int,
) -> np.ndarray | None:
    """Return a deterministic without-replacement finite-supervision order.

    Both native fine teachers and official human segmentations expose finite
    chunk inventories. Scheduling those inventories explicitly prevents large
    corpus expansions from repeatedly drawing the same easy neighborhoods while
    untouched labeled chunks remain available.
    """

    if (
        support_coordinates is None
        or not any(
            marker in supervision_source
            for marker in ("native-fine-teacher", "official-human-2um")
        )
    ):
        return None
    coordinates = np.asarray(support_coordinates, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not coordinates.size:
        raise ValueError("finite supervision coordinates must be a non-empty Nx3 array")
    rng = np.random.default_rng(_record_seed(seed, record_id, 0))
    return coordinates[rng.permutation(coordinates.shape[0])]


def _minimum_known_fraction(
    supervision_source: str,
    options: PrepareOptions,
) -> float:
    if "native-fine-teacher" in supervision_source:
        return options.native_teacher_min_known_fraction
    return options.min_known_fraction


def _decode_chunk_id(
    encoded: int,
    grid_zyx: tuple[int, int, int],
) -> tuple[int, int, int]:
    x = encoded % grid_zyx[2]
    yz = encoded // grid_zyx[2]
    y = yz % grid_zyx[1]
    z = yz // grid_zyx[1]
    return int(z), int(y), int(x)


def _quality_filter_native_teacher_support(
    record: VoxelPairRecord,
    support: ChunkSupport,
    options: PrepareOptions,
) -> tuple[ChunkSupport, NativeTeacherSupportQuality | None]:
    """Remove masked-volume boundary chunks from local native teachers.

    Native teacher records contain the nonzero fraction of the fine CT context
    used for inference. When those records are available, chunks below the
    visually audited threshold are removed from both the finite sampling pool
    and present support. They therefore voxelize as unknown rather than false
    surface/background. Published sparse teachers without local records retain
    their already-vetted inventory, while still recording that the local gate
    was unavailable.
    """

    if "native-fine-teacher" not in record.supervision_source:
        return support, None
    if support.present_ids is None:
        raise ValueError(
            f"{record.record_id}: native-fine teacher must declare finite "
            "present-chunk support"
        )

    before = int(support.present_ids.size)
    teacher_root, _ = split_volume_spec(record.fine.target.volume)
    records_dir = teacher_root / "records"
    threshold = options.native_teacher_min_fine_ct_nonzero_fraction
    if not records_dir.exists():
        return support, NativeTeacherSupportQuality(
            min_fine_ct_nonzero_fraction=threshold,
            local_records_applied=False,
            chunks_before=before,
            chunks_after=before,
        )
    if not records_dir.is_dir():
        raise ValueError(f"{records_dir}: native-teacher records path is not a directory")

    retained: list[int] = []
    for encoded_value in support.present_ids.tolist():
        encoded = int(encoded_value)
        coordinate = _decode_chunk_id(encoded, support.grid_zyx)
        path = records_dir / ("_".join(str(item) for item in coordinate) + ".json")
        if not path.is_file():
            raise ValueError(
                f"{record.record_id}: support chunk {coordinate} lacks {path.name}"
            )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: invalid native-teacher record JSON") from error
        if value.get("schema") != "crossres-native-fine-teacher-chunk-v1":
            raise ValueError(f"{path}: unsupported native-teacher record schema")
        try:
            recorded_coordinate = tuple(int(item) for item in value["chunk_zyx"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: invalid native-teacher chunk coordinate") from error
        if recorded_coordinate != coordinate:
            raise ValueError(
                f"{path}: record coordinate {recorded_coordinate} differs from {coordinate}"
            )
        try:
            ct_nonzero_fraction = float(value["ct_nonzero_fraction"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: invalid fine CT nonzero fraction") from error
        if not math.isfinite(ct_nonzero_fraction) or not 0 <= ct_nonzero_fraction <= 1:
            raise ValueError(f"{path}: fine CT nonzero fraction must be finite in [0, 1]")
        if ct_nonzero_fraction >= threshold:
            retained.append(encoded)

    if not retained:
        raise ValueError(
            f"{record.record_id}: no native-teacher support chunks clear fine CT "
            f"nonzero fraction {threshold}"
        )
    present_ids = np.asarray(retained, dtype=np.int64)
    sampling_ids = (
        None
        if support.sampling_ids is None
        else np.intersect1d(
            support.sampling_ids,
            present_ids,
            assume_unique=True,
        )
    )
    if sampling_ids is not None and not sampling_ids.size:
        raise ValueError(
            f"{record.record_id}: no positive native-teacher sampling chunks "
            "remain after the fine CT quality gate"
        )
    filtered = ChunkSupport(
        shape_zyx=support.shape_zyx,
        chunks_zyx=support.chunks_zyx,
        grid_zyx=support.grid_zyx,
        present_ids=present_ids,
        sampling_ids=sampling_ids,
    )
    return filtered, NativeTeacherSupportQuality(
        min_fine_ct_nonzero_fraction=threshold,
        local_records_applied=True,
        chunks_before=before,
        chunks_after=int(present_ids.size),
    )


def _candidate_origin(
    record: VoxelPairRecord,
    support: ChunkSupport,
    support_coordinates: np.ndarray | None,
    coarse_shape: tuple[int, int, int],
    patch_shape: tuple[int, int, int],
    rng: np.random.Generator,
    *,
    anchor_coordinate_zyx: np.ndarray | None = None,
) -> tuple[int, int, int]:
    if anchor_coordinate_zyx is not None:
        coordinate = np.asarray(anchor_coordinate_zyx, dtype=np.int64)
        if coordinate.shape != (3,) or (
            (coordinate < 0) | (coordinate >= np.asarray(support.grid_zyx))
        ).any():
            raise ValueError("support anchor is outside the fine chunk grid")
        fine_center_zyx = (
            coordinate.astype(np.float64) * np.asarray(support.chunks_zyx)
            + np.asarray(support.chunks_zyx) / 2.0
        )
        fine_center_zyx = np.minimum(fine_center_zyx, np.asarray(support.shape_zyx) - 1)
    elif support_coordinates is not None:
        coordinate = support_coordinates[
            int(rng.integers(0, support_coordinates.shape[0]))
        ]
        fine_center_zyx = (
            coordinate.astype(np.float64) * np.asarray(support.chunks_zyx)
            + np.asarray(support.chunks_zyx) / 2.0
        )
        fine_center_zyx = np.minimum(fine_center_zyx, np.asarray(support.shape_zyx) - 1)
    else:
        fine_center_zyx = rng.uniform(
            np.zeros(3), np.asarray(support.shape_zyx, dtype=np.float64) - 1
        )
    coarse_xyz = transform_xyz(
        fine_center_zyx[::-1][None],
        affine_matrix(record.fine.to_coarse_affine_xyz),
    )[0]
    center_zyx = coarse_xyz[::-1]
    jitter_limit = np.maximum(1, np.asarray(patch_shape) // 4)
    jitter = rng.integers(-jitter_limit, jitter_limit + 1)
    origin = np.rint(center_zyx).astype(np.int64) - np.asarray(patch_shape) // 2
    origin += jitter
    maximum = np.asarray(coarse_shape) - np.asarray(patch_shape)
    if (maximum < 0).any():
        raise ValueError(
            f"coarse shape {coarse_shape} is smaller than patch {patch_shape}"
        )
    origin = np.clip(origin, 0, maximum)
    return tuple(int(item) for item in origin)


def _prepare_candidate(
    record: VoxelPairRecord,
    coarse_image: Any,
    fine_target: Any,
    support: ChunkSupport,
    support_coordinates: np.ndarray | None,
    baseline: Any | None,
    projection_cache: SparseChunkProjectionCache | None,
    options: PrepareOptions,
    rng: np.random.Generator,
    *,
    support_anchor_coordinate_zyx: np.ndarray | None = None,
    support_anchor_pool_size: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    coarse_shape = tuple(int(item) for item in coarse_image.shape)
    origin = _candidate_origin(
        record,
        support,
        support_coordinates,
        coarse_shape,
        options.patch_shape_zyx,
        rng,
        anchor_coordinate_zyx=support_anchor_coordinate_zyx,
    )
    image = read_crop(coarse_image, origin, options.patch_shape_zyx)
    target, target_stats = voxelize_fine_target_patch(
        fine_target,
        record.fine.target,
        support,
        record.fine.to_coarse_affine_xyz,
        origin,
        options.patch_shape_zyx,
        validity_block=options.validity_block,
        projection_cache=projection_cache,
    )
    known = target != 2
    ct_nonzero_fraction = float(np.count_nonzero(image)) / image.size
    arrays: dict[str, np.ndarray] = {
        "image": np.ascontiguousarray(image),
        "target_u8": target,
    }
    pathology_score = 0.0
    if baseline is not None:
        assert record.coarse.baseline is not None
        baseline_raw = read_crop(baseline, origin, options.patch_shape_zyx)
        baseline_probability = decode_dense_field(baseline_raw, record.coarse.baseline)
        baseline_u8 = (baseline_probability >= record.coarse.baseline.threshold).astype(
            np.uint8
        )
        arrays["baseline_u8"] = baseline_u8
        if known.any():
            pathology_score = float(
                np.not_equal(baseline_u8[known], target[known]).mean()
            )
    stats = {
        **target_stats,
        "origin_zyx": list(origin),
        "ct_nonzero_fraction": ct_nonzero_fraction,
        "pathology_score": pathology_score,
        "has_baseline": baseline is not None,
        "support_anchor_chunk_zyx": (
            [int(item) for item in support_anchor_coordinate_zyx]
            if support_anchor_coordinate_zyx is not None
            else None
        ),
        "support_anchor_pool_size": support_anchor_pool_size,
    }
    return arrays, stats


def _acceptable(
    stats: dict[str, Any],
    options: PrepareOptions,
    *,
    supervision_source: str,
) -> bool:
    minimum_known_fraction = _minimum_known_fraction(
        supervision_source,
        options,
    )
    return bool(
        stats["known_fraction"] >= minimum_known_fraction
        and stats["positive_voxels"] >= options.min_positive_voxels
        and stats["ct_nonzero_fraction"] >= options.min_ct_nonzero_fraction
    )


def _sampling_strategy(
    patch_index: int,
    *,
    patch_count: int,
    has_baseline: bool,
    options: PrepareOptions,
) -> str:
    """Assign deterministic, evenly-spaced deployment-shaped strata."""

    # A low-discrepancy rotation spreads the ranked cases across the manifest
    # instead of putting every difficult patch in one contiguous block.
    position = ((patch_index * 0.6180339887498949) + 0.5) % 1.0
    if has_baseline and position < options.pathology_fraction:
        return "high-pathology"
    positive_start = options.pathology_fraction if has_baseline else 0.0
    positive_end = positive_start + options.positive_density_fraction
    if positive_start <= position < positive_end:
        return "dense-positive"
    return "random"


def _spatially_ordered_anchors(anchors: np.ndarray) -> np.ndarray:
    """Consume a random anchor sample along a deterministic nearest path."""

    values = np.asarray(anchors, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("anchors must be an Nx3 array")
    if not values.size:
        return values.copy()
    count = values.shape[0]
    if count == 1:
        return values.copy()
    tree = cKDTree(values.astype(np.float64, copy=False))
    used = np.zeros(count, dtype=bool)
    order = np.empty(count, dtype=np.int64)
    current = 0
    used[current] = True
    order[0] = current
    for position in range(1, count):
        remaining = count - position
        if remaining <= 32:
            candidates = np.flatnonzero(~used)
            offsets = values[candidates] - values[current]
            distances = np.einsum("ij,ij->i", offsets, offsets)
            current = int(candidates[np.lexsort((candidates, distances))[0]])
        else:
            neighbors = min(32, count)
            while True:
                distances, indices = tree.query(
                    values[current],
                    k=neighbors,
                    workers=1,
                )
                candidate_pairs = sorted(
                    (
                        (float(distance), int(index))
                        for distance, index in zip(
                            np.atleast_1d(distances),
                            np.atleast_1d(indices),
                            strict=True,
                        )
                        if index < count and not used[int(index)]
                    )
                )
                if candidate_pairs:
                    current = candidate_pairs[0][1]
                    break
                if neighbors == count:
                    raise RuntimeError("nearest-anchor traversal exhausted early")
                neighbors = min(count, neighbors * 2)
        used[current] = True
        order[position] = current
    return values[order]


def _support_anchor_candidate_schedule(
    support_anchor_schedule: np.ndarray | None,
    *,
    patch_count: int,
    has_baseline: bool,
    options: PrepareOptions,
) -> tuple[np.ndarray, ...] | None:
    """Allocate surplus native-teacher anchors to globally ranked strata.

    Every patch receives one deterministic base anchor.  When a materialized
    teacher pool is larger than the record's patch budget, surplus anchors are
    assigned without replacement to high-pathology and dense-positive patches.
    Those patches can then compare genuinely different spatial neighborhoods,
    rather than several jitters around one already-fixed anchor.
    """

    if support_anchor_schedule is None:
        return None
    anchors = np.asarray(support_anchor_schedule, dtype=np.int64)
    if anchors.ndim != 2 or anchors.shape[1] != 3 or not anchors.size:
        raise ValueError("support anchor schedule must be a non-empty Nx3 array")
    if patch_count <= 0:
        raise ValueError("patch_count must be positive")
    pool_size = int(anchors.shape[0])
    base_anchors = anchors[np.arange(patch_count) % pool_size]
    if pool_size > patch_count:
        # Preserve the shuffled random subset while ordering its consumption
        # spatially. Consecutive 192-cubes then reuse most of the same fine
        # chunks instead of decompressing hundreds of random 128-cubes anew.
        base_anchors = _spatially_ordered_anchors(base_anchors)
    groups: list[list[np.ndarray]] = [[anchor] for anchor in base_anchors]
    if pool_size <= patch_count or options.selection_candidates <= 1:
        return tuple(np.stack(group) for group in groups)

    ranked_indices = [
        index
        for index in range(patch_count)
        if _sampling_strategy(
            index,
            patch_count=patch_count,
            has_baseline=has_baseline,
            options=options,
        )
        != "random"
    ]
    cursor = patch_count
    while cursor < pool_size and ranked_indices:
        eligible = [
            index
            for index in ranked_indices
            if len(groups[index]) < options.selection_candidates
        ]
        if not eligible:
            break
        count = min(len(eligible), pool_size - cursor)
        lane = anchors[cursor : cursor + count]
        lane = _spatially_ordered_anchors(lane)
        for index, anchor in zip(eligible, lane, strict=False):
            groups[index].append(anchor)
        cursor += count
    return tuple(np.stack(group) for group in groups)


def _support_anchor_fallback_schedule(
    support_anchor_schedule: np.ndarray | None,
    support_anchor_candidates: tuple[np.ndarray, ...] | None,
) -> np.ndarray | None:
    """Return otherwise-unused native anchors as a deterministic reserve.

    Primary and ranked candidate groups keep their existing assignment.  This
    second tier is consulted only after every attempt around a patch's primary
    group fails acceptance, typically because a valid fine-teacher chunk maps
    into masked/empty coarse CT.  Callers consume reserve anchors globally
    without replacement, preserving the no-overlap contract across candidate
    groups while allowing a difficult patch to search beyond one or two
    pre-partitioned fallbacks.
    """

    if support_anchor_schedule is None or support_anchor_candidates is None:
        return None
    used = {
        tuple(int(item) for item in coordinate)
        for group in support_anchor_candidates
        for coordinate in group
    }
    remaining = [
        coordinate
        for coordinate in np.asarray(support_anchor_schedule, dtype=np.int64)
        if tuple(int(item) for item in coordinate) not in used
    ]
    return (
        np.stack(remaining)
        if remaining
        else np.empty((0, 3), dtype=np.int64)
    )


def _support_anchor_reuse_fallback_schedule(
    support_anchor_schedule: np.ndarray,
    patch_anchor_candidates: np.ndarray,
    *,
    patch_index: int,
    max_candidates: int | None = None,
) -> np.ndarray:
    """Return deterministic alternate anchors when the pool must be reused.

    Some fine-only teachers contain fewer usable chunks than the requested
    patch budget. Their primary schedule necessarily wraps around the same
    finite pool, so there can be no globally unused reserve. If a primary
    anchor cannot produce an acceptable coarse/fine crop, rotate through the
    remaining anchors for that patch. Anchors remain unique within the
    patch's provenance, while reuse between patches is explicit and
    unavoidable for these sparse pools.
    """

    anchors = np.asarray(support_anchor_schedule, dtype=np.int64)
    primary = np.asarray(patch_anchor_candidates, dtype=np.int64)
    if anchors.ndim != 2 or anchors.shape[1] != 3 or not anchors.size:
        raise ValueError("support anchor schedule must be a non-empty Nx3 array")
    if primary.ndim != 2 or primary.shape[1] != 3 or not primary.size:
        raise ValueError("patch anchor candidates must be a non-empty Nx3 array")
    primary_ids = {
        tuple(int(item) for item in coordinate) for coordinate in primary
    }
    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    pool_size = int(anchors.shape[0])
    start = (patch_index + 1) % pool_size
    remaining: list[np.ndarray] = []
    for offset in range(pool_size):
        coordinate = anchors[(start + offset) % pool_size]
        if tuple(int(item) for item in coordinate) in primary_ids:
            continue
        remaining.append(coordinate)
        if max_candidates is not None and len(remaining) >= max_candidates:
            break
    return (
        np.stack(remaining)
        if remaining
        else np.empty((0, 3), dtype=np.int64)
    )


def _support_anchor_pool_requires_reuse(
    support_anchor_schedule: np.ndarray | None,
    *,
    patch_count: int,
) -> bool:
    """Return whether failed primaries must be replaced from the used pool.

    At exact pool exhaustion there are no globally unused reserve anchors either.
    A fine-CT-valid anchor can still map into masked coarse CT, so equality must
    permit a recorded substitution just like a genuinely undersized pool.
    """

    return bool(
        support_anchor_schedule is not None
        and int(support_anchor_schedule.shape[0]) <= patch_count
    )


def _prepare_primary_patch_candidates(
    record: VoxelPairRecord,
    coarse_image: Any,
    fine_target: Any,
    support: ChunkSupport,
    support_coordinates: np.ndarray | None,
    baseline: Any | None,
    projection_cache: SparseChunkProjectionCache | None,
    options: PrepareOptions,
    *,
    patch_index: int,
    patch_count: int,
    support_anchor_schedule: np.ndarray | None,
    support_anchor_candidates: tuple[np.ndarray, ...] | None,
    support_anchor_fallbacks: np.ndarray | None,
) -> _PrimaryPatchCandidates:
    safe_record_id = record.record_id.replace("/", "_").replace("\\", "_")
    patch_id = f"{safe_record_id}-{patch_index:05d}"
    rng = np.random.default_rng(
        _record_seed(options.seed, record.record_id, patch_index)
    )
    strategy = _sampling_strategy(
        patch_index,
        patch_count=patch_count,
        has_baseline=baseline is not None,
        options=options,
    )
    patch_anchor_candidates = (
        support_anchor_candidates[patch_index]
        if support_anchor_candidates is not None
        else None
    )
    acceptable: list[tuple[dict[str, np.ndarray], dict[str, Any]]] = []
    best: tuple[dict[str, np.ndarray], dict[str, Any]] | None = None
    candidate_attempts = 0
    if patch_anchor_candidates is not None and len(patch_anchor_candidates) > 1:
        attempts, remainder = divmod(
            options.attempts_per_patch,
            len(patch_anchor_candidates),
        )
        for anchor_index, support_anchor in enumerate(patch_anchor_candidates):
            anchor_attempts = attempts + (anchor_index < remainder)
            for _ in range(anchor_attempts):
                candidate_attempts += 1
                candidate = _prepare_candidate(
                    record,
                    coarse_image,
                    fine_target,
                    support,
                    support_coordinates,
                    baseline,
                    projection_cache,
                    options,
                    rng,
                    support_anchor_coordinate_zyx=support_anchor,
                    support_anchor_pool_size=int(support_anchor_schedule.shape[0]),
                )
                if (
                    best is None
                    or candidate[1]["known_fraction"] > best[1]["known_fraction"]
                ):
                    best = candidate
                if _acceptable(
                    candidate[1],
                    options,
                    supervision_source=record.supervision_source,
                ):
                    acceptable.append(candidate)
                    break
    else:
        support_anchor = (
            patch_anchor_candidates[0]
            if patch_anchor_candidates is not None
            else None
        )
        for _ in range(options.attempts_per_patch):
            candidate_attempts += 1
            candidate = _prepare_candidate(
                record,
                coarse_image,
                fine_target,
                support,
                support_coordinates,
                baseline,
                projection_cache,
                options,
                rng,
                support_anchor_coordinate_zyx=support_anchor,
                support_anchor_pool_size=(
                    int(support_anchor_schedule.shape[0])
                    if support_anchor_schedule is not None
                    else None
                ),
            )
            if (
                best is None
                or candidate[1]["known_fraction"] > best[1]["known_fraction"]
            ):
                best = candidate
            if _acceptable(
                candidate[1],
                options,
                supervision_source=record.supervision_source,
            ):
                acceptable.append(candidate)
                needed = (
                    1 if strategy == "random" else options.selection_candidates
                )
                if len(acceptable) >= needed:
                    break
    return _PrimaryPatchCandidates(
        patch_index=patch_index,
        patch_id=patch_id,
        rng=rng,
        strategy=strategy,
        patch_anchor_candidates=patch_anchor_candidates,
        patch_anchor_fallbacks=support_anchor_fallbacks,
        acceptable=acceptable,
        best=best,
        candidate_attempts=candidate_attempts,
    )


def _select_candidate(
    candidates: list[tuple[dict[str, np.ndarray], dict[str, Any]]],
    strategy: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    def annotate(
        candidate: tuple[dict[str, np.ndarray], dict[str, Any]],
    ) -> ScrollFiestaPredMetrics | None:
        arrays, stats = candidate
        raw_metrics = stats.get("scrollfiesta_pred_metrics")
        if raw_metrics is not None:
            return ScrollFiestaPredMetrics.from_dict(raw_metrics)
        baseline = arrays.get("baseline_u8")
        if baseline is None:
            stats["scrollfiesta_pred_metrics"] = None
            return None
        metrics = scrollfiesta_patch_pred_metrics(baseline)
        stats["scrollfiesta_pred_metrics"] = metrics.to_dict()
        return metrics

    if strategy == "high-pathology":
        measured = [(candidate, annotate(candidate)) for candidate in candidates]
        selected = max(
            measured,
            key=lambda item: (
                item[1].reject_priority if item[1] is not None else -1,
                item[0][1]["pathology_score"],
            ),
        )[0]
    elif strategy == "dense-positive":
        selected = max(
            candidates,
            key=lambda item: item[1]["positive_fraction_known"],
        )
    else:
        selected = candidates[0]
    annotate(selected)
    return selected


def _truncate_manifest(manifest: Path, length: int) -> None:
    with manifest.open("r+b") as stream:
        stream.truncate(length)
        stream.flush()
        os.fsync(stream.fileno())


def _archive_matches_row(output: Path, row: dict[str, Any]) -> bool:
    relative = Path(str(row.get("path", "")))
    if not relative.as_posix() or relative.is_absolute():
        return False
    archive = (output / relative).resolve()
    try:
        archive.relative_to(output.resolve())
        expected_bytes = int(row["archive_bytes"])
        expected_hash = str(row["archive_sha256"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        expected_bytes >= 0
        and len(expected_hash) == 64
        and archive.is_file()
        and archive.stat().st_size == expected_bytes
        and _sha256(archive) == expected_hash
    )


def _load_existing_rows(
    manifest: Path,
    *,
    output: Path | None = None,
    durable_completed: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Load a manifest and recover only its non-durable crash tail.

    Patch archives are fsynced before their rows are appended. Manifest and
    state fsyncs may be batched, so a hard reset can leave a partial final row
    or valid rows beyond the last durable state checkpoint. A partial final row
    is truncated. Rows after ``durable_completed`` are accepted only after the
    referenced immutable archive still matches its length and SHA-256; the
    first invalid tail row and everything after it are discarded for
    deterministic regeneration.
    """

    if not manifest.exists():
        return {}
    if (output is None) != (durable_completed is None):
        raise ValueError("output and durable_completed must be supplied together")
    if durable_completed is not None and durable_completed < 0:
        raise ValueError("durable_completed cannot be negative")

    payload = manifest.read_bytes()
    raw_lines = payload.splitlines(keepends=True)
    rows: dict[str, dict[str, Any]] = {}
    offsets: list[tuple[str, int]] = []
    offset = 0
    repaired_tail = False
    for line_number, raw_line in enumerate(raw_lines, 1):
        line_start = offset
        offset += len(raw_line)
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise TypeError("row is not an object")
            patch_id = str(row["patch_id"])
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            if any(value.strip() for value in raw_lines[line_number:]):
                raise ValueError(
                    f"{manifest}:{line_number}: invalid non-tail manifest row"
                ) from error
            if durable_completed is not None and len(rows) < durable_completed:
                raise ValueError(
                    f"{manifest}:{line_number}: invalid durable manifest row"
                ) from error
            _truncate_manifest(manifest, line_start)
            repaired_tail = True
            break
        if not patch_id or patch_id in rows:
            raise ValueError(
                f"{manifest}:{line_number}: missing or duplicate patch ID {patch_id!r}"
            )
        rows[patch_id] = row
        offsets.append((patch_id, line_start))

    if not repaired_tail and payload and not payload.endswith(b"\n"):
        with manifest.open("ab") as stream:
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())

    if durable_completed is not None and durable_completed > len(rows):
        raise ValueError(
            f"{manifest}: durable count {durable_completed} exceeds "
            f"the {len(rows)} valid manifest rows"
        )

    if output is not None and durable_completed is not None:
        ordered = list(rows)
        validation_start = max(0, min(durable_completed, len(ordered)) - 1)
        for index in range(validation_start, len(ordered)):
            patch_id = ordered[index]
            if _archive_matches_row(output, rows[patch_id]):
                continue
            if index < durable_completed:
                raise ValueError(
                    f"{manifest}: durable archive no longer matches row {index + 1} "
                    f"({patch_id})"
                )
            _truncate_manifest(manifest, offsets[index][1])
            rows = {key: rows[key] for key in ordered[:index]}
            break
    return rows


def _commit_prepare_progress(
    *,
    manifest: Path,
    state_path: Path,
    identity: dict[str, Any],
    completed: int,
    state: str = "preparing",
) -> None:
    if manifest.exists():
        with manifest.open("ab") as stream:
            stream.flush()
            os.fsync(stream.fileno())
    _write_json_atomic(
        state_path,
        {"state": state, "identity": identity, "completed": completed},
    )


def prepare_patch_corpus(
    *,
    pair_manifest: str | Path,
    output_path: str | Path,
    options: PrepareOptions,
) -> Path:
    """Build or resume an immutable dense voxel patch corpus."""

    options.validate()
    configure_cpu_budget(options.max_cpu_threads)
    source = Path(pair_manifest).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    state_path = output / "prepare_state.json"
    patch_manifest = output / "patches.jsonl"
    expected_identity = json.loads(
        json.dumps(
            {
                "pair_manifest": str(source),
                "pair_manifest_sha256": _sha256(source),
                "preparation_version": PATCH_PREPARATION_VERSION,
                "scrollfiesta_pred_metrics_contract": (
                    SCROLLFIESTA_PRED_METRICS_CONTRACT
                ),
                "options": asdict(options),
            }
        )
    )
    previous_completed = 0
    if output.exists():
        if not state_path.exists():
            raise ValueError(f"{output}: non-empty output has no preparation state")
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("identity") != expected_identity:
            raise ValueError(f"{output}: preparation identity does not match")
        if previous.get("state") == "complete":
            return patch_manifest
        previous_completed = int(previous.get("completed", 0))
    else:
        output.mkdir(parents=True)
        (output / "patches").mkdir()
        _write_json_atomic(
            state_path,
            {"state": "preparing", "identity": expected_identity, "completed": 0},
        )

    records = load_pair_manifest(source)
    existing_rows = _load_existing_rows(
        patch_manifest,
        output=output,
        durable_completed=previous_completed,
    )
    existing_ids = set(existing_rows)
    completed = len(existing_ids)
    if completed != previous_completed:
        _commit_prepare_progress(
            manifest=patch_manifest,
            state_path=state_path,
            identity=expected_identity,
            completed=completed,
        )
    for record in records:
        coarse_image = open_volume(record.coarse.image)
        fine_target = open_volume(record.fine.target.volume)
        support = ChunkSupport.from_field(record.fine.target, fine_target)
        support, support_quality = _quality_filter_native_teacher_support(
            record,
            support,
            options,
        )
        support_coordinates = (
            support.coordinates() if support.present_ids is not None else None
        )
        support_anchor_schedule = _support_anchor_schedule(
            support_coordinates,
            record_id=record.record_id,
            supervision_source=record.supervision_source,
            seed=options.seed,
        )
        baseline = (
            open_volume(record.coarse.baseline.volume)
            if record.coarse.baseline is not None
            else None
        )
        if baseline is not None and tuple(baseline.shape) != tuple(coarse_image.shape):
            raise ValueError(
                f"{record.record_id}: baseline shape does not match coarse image"
            )
        projection_cache = (
            SparseChunkProjectionCache(
                fine_target,
                record.fine.target,
                support,
                record.fine.to_coarse_affine_xyz,
                tuple(int(item) for item in coarse_image.shape),
                max_entries=options.projection_cache_entries,
            )
            if support.present_ids is not None
            else None
        )
        if projection_cache is not None:
            print(
                f"{record.record_id}: sparse projection backend "
                f"{projection_cache.projection_backend}",
                flush=True,
            )
        patch_count = record.patch_count or options.patches_per_record
        support_anchor_candidates = _support_anchor_candidate_schedule(
            support_anchor_schedule,
            patch_count=patch_count,
            has_baseline=baseline is not None,
            options=options,
        )
        support_anchor_fallbacks = _support_anchor_fallback_schedule(
            support_anchor_schedule,
            support_anchor_candidates,
        )
        reuse_sparse_anchor_pool = _support_anchor_pool_requires_reuse(
            support_anchor_schedule,
            patch_count=patch_count,
        )
        fallback_ids = {
            tuple(int(item) for item in coordinate)
            for coordinate in (
                support_anchor_fallbacks
                if support_anchor_fallbacks is not None
                else np.empty((0, 3), dtype=np.int64)
            )
        }
        consumed_fallback_ids = {
            coordinate
            for row in existing_rows.values()
            if row.get("record_id") == record.record_id
            for raw_coordinate in (
                row.get("support_anchor_candidate_chunks_zyx") or []
            )
            if (coordinate := tuple(int(item) for item in raw_coordinate))
            in fallback_ids
        }
        acceptance_min_known_fraction = _minimum_known_fraction(
            record.supervision_source,
            options,
        )
        safe_record_id = record.record_id.replace("/", "_").replace("\\", "_")

        def prepare_primary(
            patch_index: int,
            record: VoxelPairRecord = record,
            coarse_image: Any = coarse_image,
            fine_target: Any = fine_target,
            support: ChunkSupport = support,
            support_coordinates: np.ndarray | None = support_coordinates,
            baseline: Any | None = baseline,
            projection_cache: SparseChunkProjectionCache | None = projection_cache,
            patch_count: int = patch_count,
            support_anchor_schedule: np.ndarray | None = support_anchor_schedule,
            support_anchor_candidates: (
                tuple[np.ndarray, ...] | None
            ) = support_anchor_candidates,
            support_anchor_fallbacks: np.ndarray | None = support_anchor_fallbacks,
        ) -> _PrimaryPatchCandidates:
            return _prepare_primary_patch_candidates(
                record,
                coarse_image,
                fine_target,
                support,
                support_coordinates,
                baseline,
                projection_cache,
                options,
                patch_index=patch_index,
                patch_count=patch_count,
                support_anchor_schedule=support_anchor_schedule,
                support_anchor_candidates=support_anchor_candidates,
                support_anchor_fallbacks=support_anchor_fallbacks,
            )

        remaining_indices = (
            patch_index
            for patch_index in range(patch_count)
            if f"{safe_record_id}-{patch_index:05d}" not in existing_ids
        )
        primary_results = _bounded_ordered_primary_map(
            prepare_primary,
            remaining_indices,
            max_workers=_patch_executor_workers(options.max_cpu_threads),
        )
        for primary in primary_results:
            patch_index = primary.patch_index
            patch_id = primary.patch_id
            rng = primary.rng
            strategy = primary.strategy
            patch_anchor_candidates = primary.patch_anchor_candidates
            patch_anchor_fallbacks = primary.patch_anchor_fallbacks
            acceptable = primary.acceptable
            best = primary.best
            candidate_attempts = primary.candidate_attempts
            attempted_fallbacks: list[np.ndarray] = []
            if not acceptable and reuse_sparse_anchor_pool:
                assert support_anchor_schedule is not None
                assert patch_anchor_candidates is not None
                patch_anchor_fallbacks = _support_anchor_reuse_fallback_schedule(
                    support_anchor_schedule,
                    patch_anchor_candidates,
                    patch_index=patch_index,
                    max_candidates=(
                        options.attempts_per_patch
                        * max(1, options.selection_candidates)
                    ),
                )
            if not acceptable and patch_anchor_fallbacks is not None:
                fallback_limit = (
                    options.attempts_per_patch
                    * max(1, options.selection_candidates)
                )
                needed = (
                    1 if strategy == "random" else options.selection_candidates
                )
                for support_anchor in patch_anchor_fallbacks:
                    fallback_id = tuple(int(item) for item in support_anchor)
                    if (
                        not reuse_sparse_anchor_pool
                        and fallback_id in consumed_fallback_ids
                    ):
                        continue
                    if not reuse_sparse_anchor_pool:
                        consumed_fallback_ids.add(fallback_id)
                    attempted_fallbacks.append(support_anchor)
                    candidate_attempts += 1
                    candidate = _prepare_candidate(
                        record,
                        coarse_image,
                        fine_target,
                        support,
                        support_coordinates,
                        baseline,
                        projection_cache,
                        options,
                        rng,
                        support_anchor_coordinate_zyx=support_anchor,
                        support_anchor_pool_size=int(
                            support_anchor_schedule.shape[0]
                        ),
                    )
                    if (
                        best is None
                        or candidate[1]["known_fraction"]
                        > best[1]["known_fraction"]
                    ):
                        best = candidate
                    if _acceptable(
                        candidate[1],
                        options,
                        supervision_source=record.supervision_source,
                    ):
                        acceptable.append(candidate)
                        if len(acceptable) >= needed:
                            break
                    if len(attempted_fallbacks) >= fallback_limit:
                        break
            if attempted_fallbacks:
                patch_anchor_candidates = np.concatenate(
                    (patch_anchor_candidates, np.stack(attempted_fallbacks)),
                    axis=0,
                )
            if not acceptable:
                assert best is not None
                raise RuntimeError(
                    f"{patch_id}: failed acceptance after {candidate_attempts} "
                    f"attempts; best stats={best[1]}"
                )
            selected = _select_candidate(acceptable, strategy)
            arrays, stats = selected
            relative = Path("patches") / f"{patch_id}.npz"
            destination = output / relative
            temporary = destination.with_name(destination.name + ".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            archive_bytes = destination.stat().st_size
            archive_sha256 = _sha256(destination)
            row = {
                "schema": PATCH_SCHEMA,
                "schema_version": 1,
                "patch_id": patch_id,
                "path": relative.as_posix(),
                "record_id": record.record_id,
                "scroll_id": record.scroll_id,
                "split": record.split,
                "supervision_source": record.supervision_source,
                "sampling_strategy": strategy,
                "preparation_version": PATCH_PREPARATION_VERSION,
                "native_teacher_min_fine_ct_nonzero_fraction": (
                    support_quality.min_fine_ct_nonzero_fraction
                    if support_quality is not None
                    else None
                ),
                "native_teacher_fine_ct_quality_gate_applied": (
                    support_quality.local_records_applied
                    if support_quality is not None
                    else None
                ),
                "native_teacher_support_chunks_before_quality_gate": (
                    support_quality.chunks_before
                    if support_quality is not None
                    else None
                ),
                "native_teacher_support_chunks_after_quality_gate": (
                    support_quality.chunks_after
                    if support_quality is not None
                    else None
                ),
                "native_teacher_support_chunks_excluded_by_quality_gate": (
                    support_quality.chunks_excluded
                    if support_quality is not None
                    else None
                ),
                "support_anchor_chunk_zyx": stats["support_anchor_chunk_zyx"],
                "support_anchor_pool_size": stats["support_anchor_pool_size"],
                "support_anchor_candidate_chunks_zyx": (
                    patch_anchor_candidates.tolist()
                    if patch_anchor_candidates is not None
                    else None
                ),
                "candidate_count": len(acceptable),
                "origin_zyx": stats["origin_zyx"],
                "shape_zyx": list(options.patch_shape_zyx),
                "known_fraction": stats["known_fraction"],
                "acceptance_min_known_fraction": acceptance_min_known_fraction,
                "positive_fraction_known": stats["positive_fraction_known"],
                "pathology_score": stats["pathology_score"],
                "scrollfiesta_pred_metrics": stats[
                    "scrollfiesta_pred_metrics"
                ],
                "has_baseline": stats["has_baseline"],
                "chunks_read": stats["chunks_read"],
                "fine_positive_voxels": stats["fine_positive_voxels"],
                "ct_nonzero_fraction": stats["ct_nonzero_fraction"],
                "archive_bytes": archive_bytes,
                "archive_sha256": archive_sha256,
            }
            with patch_manifest.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                stream.flush()
                if (completed + 1) % PREPARE_PROGRESS_DURABILITY_INTERVAL == 0:
                    os.fsync(stream.fileno())
            existing_ids.add(patch_id)
            existing_rows[patch_id] = row
            completed += 1
            if completed % PREPARE_PROGRESS_DURABILITY_INTERVAL == 0:
                _write_json_atomic(
                    state_path,
                    {
                        "state": "preparing",
                        "identity": expected_identity,
                        "completed": completed,
                    },
                )
        _commit_prepare_progress(
            manifest=patch_manifest,
            state_path=state_path,
            identity=expected_identity,
            completed=completed,
        )
        if projection_cache is not None:
            projection_cache.clear()
    _commit_prepare_progress(
        manifest=patch_manifest,
        state_path=state_path,
        identity=expected_identity,
        completed=completed,
        state="complete",
    )
    return patch_manifest
