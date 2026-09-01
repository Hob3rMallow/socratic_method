from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import ndimage, signal

from .io import open_volume, split_volume_spec
from .registration import (
    ChunkSupport,
    FineFieldWindowReader,
    invert_affine,
    transform_xyz,
)
from .schema import ChunkSupportSpec, VoxelPairRecord

LOCAL_CT_TRANSLATION_CONTRACT = "crossres-local-ct-translation-l0-v1"


@dataclass
class FineCTSource:
    spec: str
    reader: FineFieldWindowReader
    support: ChunkSupport
    support_kind: str
    support_inventory: Path | None
    resolution_method: str
    source_state: Path | None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _decode_chunk_ids(
    ids: np.ndarray,
    grid_zyx: tuple[int, int, int],
) -> np.ndarray:
    x = ids % grid_zyx[2]
    yz = ids // grid_zyx[2]
    y = yz % grid_zyx[1]
    z = yz // grid_zyx[1]
    return np.stack((z, y, x), axis=1)


def _physical_selected_support(
    root: Path,
    *,
    array_key: str,
    shape_zyx: tuple[int, int, int],
    chunks_zyx: tuple[int, int, int],
    inventory_path: Path | None = None,
) -> ChunkSupport | None:
    selected_path = inventory_path or (root / "carve_selected_chunks.json")
    if not selected_path.is_file():
        return None
    selected = _read_json(selected_path)
    grid = tuple(int(value) for value in selected["chunk_grid_zyx"])
    expected_grid = tuple(
        (extent + chunk - 1) // chunk
        for extent, chunk in zip(shape_zyx, chunks_zyx, strict=True)
    )
    if grid != expected_grid:
        raise ValueError(f"{selected_path}: chunk grid differs from fine CT")
    ids = np.asarray(selected["selected_chunk_ids"], dtype=np.int64)
    coordinates = _decode_chunk_ids(ids, grid)
    metadata = _read_json(root / array_key / ".zarray")
    separator = str(metadata.get("dimension_separator", "."))
    present: list[int] = []
    array_root = root.joinpath(*array_key.split("/"))
    for encoded, coordinate in zip(ids, coordinates, strict=True):
        if separator == "/":
            path = array_root.joinpath(*(str(int(value)) for value in coordinate))
        else:
            path = array_root / separator.join(str(int(value)) for value in coordinate)
        try:
            if path.stat().st_size > 0:
                present.append(int(encoded))
        except FileNotFoundError:
            continue
    if not present:
        raise ValueError(f"{selected_path}: no physically present CT chunks")
    return ChunkSupport(
        shape_zyx,
        chunks_zyx,
        grid,
        np.asarray(sorted(present), dtype=np.int64),
    )


def _find_published_fine_ct(pair: VoxelPairRecord) -> str:
    target_root, _ = split_volume_spec(pair.fine.target.volume)
    prefix = target_root.name.split("-surface", 1)[0]
    ancestor = target_root
    while ancestor != ancestor.parent and not ancestor.name.lower().endswith("-full"):
        ancestor = ancestor.parent
    if not ancestor.name.lower().endswith("-full"):
        raise ValueError(f"{pair.record_id}: cannot locate scroll root from fine target")
    candidates = sorted(ancestor.glob(f"{prefix}-*um-*-masked.zarr"))
    if len(candidates) != 1:
        raise ValueError(
            f"{pair.record_id}: expected one published fine CT for prefix {prefix}, "
            f"got {len(candidates)}"
        )
    return f"{candidates[0].resolve()}::0"


def resolve_fine_ct_source(
    pair: VoxelPairRecord,
    *,
    max_cache_chunks: int = 64,
) -> FineCTSource:
    """Resolve the CT that generated a fine teacher without treating holes as zero."""

    target_root, _ = split_volume_spec(pair.fine.target.volume)
    teacher_state_path = target_root / "teacher_state.json"
    support_inventory: Path | None = None
    source_state: Path | None = None
    if teacher_state_path.is_file():
        state = _read_json(teacher_state_path)
        identity = state.get("identity")
        if state.get("state") != "complete" or not isinstance(identity, dict):
            raise ValueError(f"{teacher_state_path}: teacher state is not complete")
        fine_spec = str(identity.get("fine_volume") or "")
        if not fine_spec:
            raise ValueError(f"{teacher_state_path}: identity.fine_volume is missing")
        raw_inventory = identity.get("fine_support_inventory")
        support_inventory = (
            Path(str(raw_inventory)).expanduser().resolve() if raw_inventory else None
        )
        source_state = teacher_state_path
        resolution_method = "teacher-state"
    elif pair.coarse.scan_id == pair.fine.scan_id:
        coarse_root, _ = split_volume_spec(pair.coarse.image)
        fine_spec = f"{coarse_root.resolve()}::0"
        candidate_inventory = coarse_root / "crossres_sparse_objects.jsonl"
        support_inventory = candidate_inventory if candidate_inventory.is_file() else None
        resolution_method = "same-scan-pyramid-root"
    elif pair.supervision_source == "official-native-fine-teacher/published":
        fine_spec = _find_published_fine_ct(pair)
        resolution_method = "published-target-volume-prefix"
    else:
        raise ValueError(
            f"{pair.record_id}: cannot resolve fine CT from "
            f"{pair.supervision_source!r}"
        )
    fine_volume = open_volume(fine_spec)
    raw_chunks = getattr(fine_volume, "chunks", fine_volume.shape)
    shape = tuple(int(value) for value in fine_volume.shape)
    chunks = tuple(int(value) for value in raw_chunks)
    if len(shape) != 3 or len(chunks) != 3:
        raise ValueError(f"{pair.record_id}: fine CT is not three-dimensional")
    fine_root, array_key = split_volume_spec(fine_spec)
    key = array_key or "0"
    if support_inventory is None:
        candidate_inventory = fine_root / "crossres_sparse_objects.jsonl"
        if candidate_inventory.is_file():
            support_inventory = candidate_inventory.resolve()
    ct_field = replace(
        pair.fine.target,
        volume=fine_spec,
        support=(
            ChunkSupportSpec(kind="present-chunks", inventory=support_inventory)
            if support_inventory is not None
            else ChunkSupportSpec()
        ),
    )
    if support_inventory is not None and support_inventory.suffix.lower() == ".jsonl":
        support = ChunkSupport.from_field(ct_field, fine_volume)
        support_kind = "declared-present-chunks-jsonl"
    elif support_inventory is not None:
        physical = _physical_selected_support(
            fine_root,
            array_key=key,
            shape_zyx=shape,
            chunks_zyx=chunks,
            inventory_path=support_inventory,
        )
        if physical is None:
            raise ValueError(
                f"{pair.record_id}: declared fine CT support inventory is missing"
            )
        support = physical
        support_kind = "physically-present-declared-json-selection"
        ct_field = replace(ct_field, support=ChunkSupportSpec())
    else:
        physical = _physical_selected_support(
            fine_root,
            array_key=key,
            shape_zyx=shape,
            chunks_zyx=chunks,
        )
        if physical is None:
            support = ChunkSupport(
                shape,
                chunks,
                tuple(
                    (extent + chunk - 1) // chunk
                    for extent, chunk in zip(shape, chunks, strict=True)
                ),
                None,
            )
            support_kind = "full"
        else:
            support = physical
            support_kind = "physically-present-selected-chunks"
            ct_field = replace(ct_field, support=ChunkSupportSpec())
    return FineCTSource(
        spec=fine_spec,
        reader=FineFieldWindowReader(
            fine_volume,
            ct_field,
            support,
            max_cache_chunks=max_cache_chunks,
        ),
        support=support,
        support_kind=support_kind,
        support_inventory=support_inventory,
        resolution_method=resolution_method,
        source_state=source_state,
    )


def sample_fine_ct_cube(
    source: FineCTSource,
    pair: VoxelPairRecord,
    *,
    coarse_origin_zyx: tuple[int, int, int],
    coarse_shape_zyx: tuple[int, int, int],
    sigma_scale: float,
    affine_xyz: Sequence[Sequence[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    coarse_indices = np.indices(coarse_shape_zyx, dtype=np.float64)
    for axis in range(3):
        coarse_indices[axis] += coarse_origin_zyx[axis]
    coarse_xyz = np.moveaxis(coarse_indices[::-1], 0, -1).reshape(-1, 3)
    declared = pair.fine.to_coarse_affine_xyz if affine_xyz is None else affine_xyz
    matrix = tuple(tuple(float(item) for item in row) for row in declared)
    fine_xyz = transform_xyz(coarse_xyz, invert_affine(matrix))
    fine_zyx = fine_xyz[:, ::-1].T.reshape((3, *coarse_shape_zyx))
    ratio = pair.coarse.voxel_um / pair.fine.voxel_um
    sigma = float(sigma_scale * ratio)
    margin = max(2, int(np.ceil(4.0 * sigma)) + 2)
    flat = fine_zyx.reshape(3, -1)
    lower = np.floor(flat.min(axis=1) - margin).astype(np.int64)
    upper = np.ceil(flat.max(axis=1) + margin).astype(np.int64) + 1
    shape = tuple(int(value) for value in upper - lower)
    raw = source.reader.read_raw(tuple(lower), shape)
    coverage = source.reader.read_coverage(tuple(lower), shape).astype(np.float32)
    if sigma > 0:
        denominator = ndimage.gaussian_filter(
            coverage, sigma=sigma, mode="constant", cval=0.0
        )
        numerator = ndimage.gaussian_filter(
            raw.astype(np.float32) * coverage,
            sigma=sigma,
            mode="constant",
            cval=0.0,
        )
        filtered = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1.0e-4,
        )
    else:
        denominator = coverage
        filtered = raw.astype(np.float32)
    local = fine_zyx - lower[:, None, None, None]
    fine = ndimage.map_coordinates(
        filtered,
        local,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32)
    weight = ndimage.map_coordinates(
        denominator,
        local,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    valid = weight >= 0.995
    return fine, valid, {
        "fine_window_origin_zyx": lower.tolist(),
        "fine_window_shape_zyx": list(shape),
        "gaussian_sigma_fine_voxels": sigma,
        "fine_ct_support_fraction": float(np.mean(valid)),
    }


def robust_unit(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(value, dtype=np.float32)
    sample = values[np.asarray(mask, dtype=bool)]
    if sample.size < 128:
        return np.zeros_like(values, dtype=np.float32)
    lower, upper = np.percentile(sample, (1.0, 99.0))
    if upper <= lower:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)


def _masked_gaussian(value: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(value, dtype=np.float32)
    weights = np.asarray(mask, dtype=np.float32)
    denominator = ndimage.gaussian_filter(weights, sigma, mode="constant", cval=0.0)
    numerator = ndimage.gaussian_filter(
        values * weights, sigma, mode="constant", cval=0.0
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.25,
    )


def structure_field(value: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(value, dtype=np.float32)
    valid = np.ones(values.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    structure = _masked_gaussian(values, valid, 0.8) - _masked_gaussian(
        values, valid, 2.0
    )
    structure[~valid] = 0.0
    return structure


def structure_support(mask: np.ndarray) -> np.ndarray:
    return ndimage.binary_erosion(
        np.asarray(mask, dtype=bool), iterations=6, border_value=0
    )


def masked_ncc_at_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_mask: np.ndarray,
    moving_mask: np.ndarray,
    shift_zyx: Sequence[int],
) -> tuple[float, int]:
    reference_slices: list[slice] = []
    moving_slices: list[slice] = []
    for size, raw_shift in zip(reference.shape, shift_zyx, strict=True):
        shift = int(raw_shift)
        if shift >= 0:
            reference_slices.append(slice(shift, size))
            moving_slices.append(slice(0, size - shift))
        else:
            reference_slices.append(slice(0, size + shift))
            moving_slices.append(slice(-shift, size))
    lhs = np.asarray(reference[tuple(reference_slices)], dtype=np.float64)
    rhs = np.asarray(moving[tuple(moving_slices)], dtype=np.float64)
    mask = np.array(reference_mask[tuple(reference_slices)], dtype=bool, copy=True)
    mask &= np.asarray(moving_mask[tuple(moving_slices)], dtype=bool)
    count = int(np.count_nonzero(mask))
    minimum = 32 if reference.ndim == 2 else 2048
    if count < minimum:
        return float("nan"), count
    lhs = lhs[mask]
    rhs = rhs[mask]
    lhs -= lhs.mean()
    rhs -= rhs.mean()
    denominator = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if denominator <= 1.0e-12:
        return float("nan"), count
    return float(np.dot(lhs, rhs) / denominator), count


def masked_ncc_translation_volume(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_mask: np.ndarray,
    moving_mask: np.ndarray,
    *,
    radius: int,
) -> dict[str, Any]:
    if reference.shape != moving.shape or reference.ndim != 3:
        raise ValueError("translation search requires equal three-dimensional arrays")
    a = np.asarray(reference, dtype=np.float32)
    b = np.asarray(moving, dtype=np.float32)
    ma = np.asarray(reference_mask, dtype=np.float32)
    mb = np.asarray(moving_mask, dtype=np.float32)

    def correlate(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return signal.correlate(lhs, rhs, mode="full", method="fft").astype(
            np.float32, copy=False
        )

    count = correlate(ma, mb)
    sum_a = correlate(a * ma, mb)
    sum_b = correlate(ma, b * mb)
    sum_a2 = correlate(a * a * ma, mb)
    sum_b2 = correlate(ma, b * b * mb)
    sum_ab = correlate(a * ma, b * mb)
    safe_count = np.maximum(count, 1.0)
    covariance = sum_ab - sum_a * sum_b / safe_count
    variance_a = np.maximum(sum_a2 - sum_a * sum_a / safe_count, 0.0)
    variance_b = np.maximum(sum_b2 - sum_b * sum_b / safe_count, 0.0)
    denominator = np.sqrt(variance_a * variance_b)
    ncc = np.divide(
        covariance,
        denominator,
        out=np.full_like(covariance, -np.inf),
        where=denominator > 1.0e-8,
    )
    minimum_overlap = max(
        2048, int(0.20 * min(np.count_nonzero(ma), np.count_nonzero(mb)))
    )
    ncc[count < minimum_overlap] = -np.inf
    center = np.asarray(moving.shape, dtype=np.int64) - 1
    bounds = tuple(
        slice(int(value - radius), int(value + radius + 1)) for value in center
    )
    local = ncc[bounds]
    if not np.isfinite(local).any():
        raise ValueError("registration search has no finite overlap candidate")
    flat_order = np.argsort(local.reshape(-1))[::-1]
    peaks: list[dict[str, Any]] = []
    best_shift: tuple[int, int, int] | None = None
    for flat_index in flat_order:
        coordinate = np.unravel_index(int(flat_index), local.shape)
        shift = tuple(int(value - radius) for value in coordinate)
        score = float(local[coordinate])
        if not math.isfinite(score):
            continue
        if best_shift is None:
            best_shift = shift
        if all(
            max(abs(shift[axis] - other["shift_zyx"][axis]) for axis in range(3))
            > 2
            for other in peaks
        ):
            peaks.append(
                {
                    "shift_zyx": list(shift),
                    "structure_ncc": score,
                    "overlap_voxels": int(count[tuple(center + np.asarray(shift))]),
                }
            )
        if len(peaks) == 5:
            break
    if best_shift is None:
        raise ValueError("registration search has no finite peak")
    nominal = float(ncc[tuple(center)])
    nominal_count = int(count[tuple(center)])
    best_coordinate = tuple(center + np.asarray(best_shift))
    best = float(ncc[best_coordinate])
    peak_margin = best - float(peaks[1]["structure_ncc"]) if len(peaks) > 1 else None
    return {
        "available": True,
        "nominal_structure_ncc": nominal,
        "nominal_overlap_voxels": nominal_count,
        "best_structure_ncc": best,
        "best_shift_zyx": list(best_shift),
        "best_overlap_voxels": int(count[best_coordinate]),
        "structure_ncc_gain": best - nominal,
        "peak_margin": peak_margin,
        "search_radius_coarse_voxels": radius,
        "best_on_search_boundary": max(abs(value) for value in best_shift) == radius,
        "peaks": peaks,
    }


def translate(value: np.ndarray, shift_zyx: Sequence[int]) -> np.ndarray:
    output = np.zeros_like(value)
    destination: list[slice] = []
    source: list[slice] = []
    for size, raw_shift in zip(value.shape, shift_zyx, strict=True):
        shift = int(raw_shift)
        if shift >= 0:
            destination.append(slice(shift, size))
            source.append(slice(0, size - shift))
        else:
            destination.append(slice(0, size + shift))
            source.append(slice(-shift, size))
    output[tuple(destination)] = value[tuple(source)]
    return output


def measure_local_translation(
    coarse: np.ndarray,
    fine: np.ndarray,
    fine_support: np.ndarray,
    *,
    radius: int,
) -> dict[str, Any]:
    coarse_valid = np.asarray(coarse) > 0
    fine_valid = np.asarray(fine_support, dtype=bool) & (np.asarray(fine) > 0)
    coarse_unit = robust_unit(coarse, coarse_valid)
    fine_unit = robust_unit(fine, fine_valid)
    search = masked_ncc_translation_volume(
        structure_field(coarse_unit, coarse_valid),
        structure_field(fine_unit, fine_valid),
        structure_support(coarse_valid),
        structure_support(fine_valid),
        radius=radius,
    )
    shift = tuple(int(value) for value in search["best_shift_zyx"])
    nominal_intensity, nominal_overlap = masked_ncc_at_shift(
        coarse_unit, fine_unit, coarse_valid, fine_valid, (0, 0, 0)
    )
    corrected_intensity, corrected_overlap = masked_ncc_at_shift(
        coarse_unit, fine_unit, coarse_valid, fine_valid, shift
    )
    return {
        "contract": LOCAL_CT_TRANSLATION_CONTRACT,
        "registration_3d": search,
        "intensity_3d": {
            "nominal_ncc": nominal_intensity if math.isfinite(nominal_intensity) else None,
            "nominal_overlap_voxels": nominal_overlap,
            "best_ncc_at_structure_shift": (
                corrected_intensity if math.isfinite(corrected_intensity) else None
            ),
            "best_overlap_voxels": corrected_overlap,
        },
        "coarse_unit": coarse_unit,
        "coarse_valid": coarse_valid,
        "fine_unit": fine_unit,
        "fine_valid": fine_valid,
    }


def translated_affine_xyz(
    affine_xyz: Sequence[Sequence[float]],
    shift_zyx: Sequence[int | float],
) -> list[list[float]]:
    result = np.asarray(affine_xyz, dtype=np.float64).copy()
    if result.shape != (3, 4):
        raise ValueError("affine_xyz must be 3x4")
    result[:, 3] += np.asarray(tuple(shift_zyx)[::-1], dtype=np.float64)
    return result.tolist()
