from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterable
from concurrent.futures import Future
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from threading import Lock

import numpy as np

from crossres_pred.resample import (
    BridgeOptions,
    affine_scale_ratio,
    resample_to_coarse,
)

from .io import ArrayLike3D, decode_dense_field, dense_field_masks, split_volume_spec
from .schema import DenseFieldSpec


def affine_matrix(
    affine_xyz: tuple[tuple[float, float, float, float], ...],
) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :] = np.asarray(affine_xyz, dtype=np.float64)
    return result


def invert_affine(
    affine_xyz: tuple[tuple[float, float, float, float], ...],
) -> np.ndarray:
    return np.linalg.inv(affine_matrix(affine_xyz))


def transform_xyz(points_xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be Nx3 xyz, got {points.shape}")
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _chunk_grid(
    shape: tuple[int, ...], chunks: tuple[int, int, int]
) -> tuple[int, int, int]:
    return tuple(
        (int(extent) + chunk - 1) // chunk
        for extent, chunk in zip(shape, chunks, strict=True)
    )


@dataclass(frozen=True)
class ChunkSupport:
    """Fine voxels known to be present in a full or sparse Zarr mirror."""

    shape_zyx: tuple[int, int, int]
    chunks_zyx: tuple[int, int, int]
    grid_zyx: tuple[int, int, int]
    present_ids: np.ndarray | None
    sampling_ids: np.ndarray | None = None

    @classmethod
    def from_field(cls, field: DenseFieldSpec, volume: ArrayLike3D) -> ChunkSupport:
        raw_chunks = getattr(volume, "chunks", None)
        if raw_chunks is None or len(raw_chunks) != 3:
            chunks = tuple(int(item) for item in volume.shape)
        else:
            chunks = tuple(int(item) for item in raw_chunks)
        shape = tuple(int(item) for item in volume.shape)
        grid = _chunk_grid(shape, chunks)
        if field.support.kind == "all":
            return cls(shape, chunks, grid, None)

        assert field.support.inventory is not None
        _, array_key = split_volume_spec(field.volume)
        key_parts = tuple(part for part in (array_key or "").split("/") if part)
        ids: list[int] = []
        sampling_ids: list[int] = []
        has_sampling_stats = False
        with field.support.inventory.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{field.support.inventory}:{line_number}: invalid JSON"
                    ) from error
                if row.get("kind") != "chunk":
                    continue
                # Sparse label mirrors deliberately materialize zero-byte chunk
                # placeholders for unknown space. They are inventory records,
                # but they are not readable Zarr chunks and must never enter the
                # support set: Blosc correctly rejects an empty payload. Older
                # generic inventories may omit size, so retain their historical
                # behaviour while validating the value whenever it is present.
                if "size" in row:
                    try:
                        encoded_size = int(row["size"])
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"{field.support.inventory}:{line_number}: invalid chunk size "
                            f"{row.get('size')!r}"
                        ) from exc
                    if encoded_size < 0:
                        raise ValueError(
                            f"{field.support.inventory}:{line_number}: negative chunk size "
                            f"{encoded_size}"
                        )
                    if encoded_size == 0:
                        continue
                parts = tuple(Path(str(row["relative_path"])).parts)
                if key_parts and parts[: len(key_parts)] != key_parts:
                    continue
                chunk_parts = parts[len(key_parts) :]
                if len(chunk_parts) == 1 and "." in chunk_parts[0]:
                    chunk_parts = tuple(chunk_parts[0].split("."))
                if len(chunk_parts) != 3:
                    continue
                try:
                    coordinate = tuple(int(item) for item in chunk_parts)
                except ValueError:
                    continue
                if all(
                    0 <= item < extent
                    for item, extent in zip(coordinate, grid, strict=True)
                ):
                    encoded = cls._encode_static(coordinate, grid)
                    ids.append(encoded)
                    if "positive_voxels" in row:
                        has_sampling_stats = True
                        if int(row["positive_voxels"]) > 0:
                            sampling_ids.append(encoded)
        if not ids:
            raise ValueError(
                f"{field.support.inventory}: no chunks found for array {array_key!r}"
            )
        present = np.unique(np.asarray(ids, dtype=np.int64))
        sampling = (
            np.unique(np.asarray(sampling_ids, dtype=np.int64))
            if has_sampling_stats and sampling_ids
            else None
        )
        return cls(shape, chunks, grid, present, sampling)

    @staticmethod
    def _encode_static(
        coordinate_zyx: tuple[int, int, int], grid_zyx: tuple[int, int, int]
    ) -> int:
        z, y, x = coordinate_zyx
        return (z * grid_zyx[1] + y) * grid_zyx[2] + x

    def encode(self, coordinate_zyx: tuple[int, int, int]) -> int:
        return self._encode_static(coordinate_zyx, self.grid_zyx)

    def contains_many(self, coordinates_zyx: np.ndarray) -> np.ndarray:
        coordinates = np.asarray(coordinates_zyx, dtype=np.int64)
        inside = ((coordinates >= 0) & (coordinates < np.asarray(self.grid_zyx))).all(
            axis=-1
        )
        if self.present_ids is None:
            return inside
        ids = coordinates[..., 0] * self.grid_zyx[1] + coordinates[..., 1]
        ids = ids * self.grid_zyx[2] + coordinates[..., 2]
        flat = ids.reshape(-1)
        positions = np.searchsorted(self.present_ids, flat)
        found = positions < self.present_ids.size
        safe = np.minimum(positions, self.present_ids.size - 1)
        found &= self.present_ids[safe] == flat
        return found.reshape(ids.shape) & inside

    def contains(self, coordinate_zyx: tuple[int, int, int]) -> bool:
        return bool(self.contains_many(np.asarray(coordinate_zyx)[None, :])[0])

    def coordinates(self) -> np.ndarray:
        if self.present_ids is None:
            raise ValueError("full support has no finite candidate chunk list")
        ids = self.sampling_ids if self.sampling_ids is not None else self.present_ids
        x = ids % self.grid_zyx[2]
        yz = ids // self.grid_zyx[2]
        y = yz % self.grid_zyx[1]
        z = yz // self.grid_zyx[1]
        return np.stack((z, y, x), axis=1)

    def iter_between(
        self,
        lower_zyx: tuple[int, int, int],
        upper_zyx: tuple[int, int, int],
    ) -> Iterable[tuple[int, int, int]]:
        lower = np.maximum(np.asarray(lower_zyx, dtype=np.int64), 0)
        upper = np.minimum(np.asarray(upper_zyx, dtype=np.int64), self.grid_zyx)
        for coordinate in product(
            range(int(lower[0]), int(upper[0])),
            range(int(lower[1]), int(upper[1])),
            range(int(lower[2]), int(upper[2])),
        ):
            if self.present_ids is None or self.contains(coordinate):
                yield coordinate


class FineFieldWindowReader:
    """Sparse-safe window readers for a fine dense field.

    Missing chunks in a teacher mirror are unknown, not background. Reading a
    rectangular Zarr selection directly can also touch zero-byte placeholders
    deliberately left by sparse mirrors. This reader visits only chunks in the
    declared :class:`ChunkSupport`, decodes their field values, and keeps a small
    raw-chunk LRU shared by the probability and coverage calls made by the
    anti-aliased bridge.
    """

    def __init__(
        self,
        fine_volume: ArrayLike3D,
        field: DenseFieldSpec,
        support: ChunkSupport,
        *,
        max_cache_chunks: int = 8,
    ) -> None:
        if max_cache_chunks <= 0:
            raise ValueError("max_cache_chunks must be positive")
        if tuple(int(item) for item in fine_volume.shape) != support.shape_zyx:
            raise ValueError("fine volume and support shapes differ")
        self.fine_volume = fine_volume
        self.field = field
        self.support = support
        self.max_cache_chunks = int(max_cache_chunks)
        self._cache: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()
        self.chunk_reads = 0
        self.cache_hits = 0

    def _chunk(self, coordinate_zyx: tuple[int, int, int]) -> np.ndarray:
        cached = self._cache.get(coordinate_zyx)
        if cached is not None:
            self._cache.move_to_end(coordinate_zyx)
            self.cache_hits += 1
            return cached
        chunk_shape = np.asarray(self.support.chunks_zyx, dtype=np.int64)
        origin = np.asarray(coordinate_zyx, dtype=np.int64) * chunk_shape
        end = np.minimum(origin + chunk_shape, np.asarray(self.support.shape_zyx))
        slices = tuple(
            slice(int(lo), int(hi)) for lo, hi in zip(origin, end, strict=True)
        )
        value = np.ascontiguousarray(self.fine_volume[slices])
        self.chunk_reads += 1
        self._cache[coordinate_zyx] = value
        self._cache.move_to_end(coordinate_zyx)
        while len(self._cache) > self.max_cache_chunks:
            self._cache.popitem(last=False)
        return value

    def _read(
        self,
        origin_zyx: tuple[int, int, int],
        shape_zyx: tuple[int, int, int],
        *,
        probability: bool,
        compact_probability: bool = False,
    ) -> np.ndarray:
        if any(size <= 0 for size in shape_zyx):
            raise ValueError("fine window shape must be positive")
        dtype = np.uint8 if compact_probability or not probability else np.float32
        output = np.zeros(shape_zyx, dtype=dtype)
        window_lo = np.asarray(origin_zyx, dtype=np.int64)
        window_hi = window_lo + np.asarray(shape_zyx, dtype=np.int64)
        chunks = np.asarray(self.support.chunks_zyx, dtype=np.int64)
        chunk_lo = np.floor_divide(window_lo, chunks)
        chunk_hi = np.floor_divide(window_hi + chunks - 1, chunks)
        for coordinate in self.support.iter_between(tuple(chunk_lo), tuple(chunk_hi)):
            chunk_origin = np.asarray(coordinate, dtype=np.int64) * chunks
            raw = self._chunk(coordinate)
            chunk_end = chunk_origin + np.asarray(raw.shape, dtype=np.int64)
            lower = np.maximum(window_lo, chunk_origin)
            upper = np.minimum(window_hi, chunk_end)
            if (lower >= upper).any():
                continue
            destination = tuple(
                slice(int(lo - base), int(hi - base))
                for lo, hi, base in zip(lower, upper, window_lo, strict=True)
            )
            source = tuple(
                slice(int(lo - base), int(hi - base))
                for lo, hi, base in zip(lower, upper, chunk_origin, strict=True)
            )
            raw_view = raw[source]
            if probability:
                if compact_probability:
                    positive, _ = dense_field_masks(raw_view, self.field)
                    output[destination] = positive.astype(np.uint8, copy=False) * 255
                else:
                    output[destination] = decode_dense_field(raw_view, self.field)
            else:
                _, known = dense_field_masks(raw_view, self.field)
                output[destination] = known.astype(np.uint8, copy=False)
        return output

    def read_probability(
        self,
        origin_zyx: tuple[int, int, int],
        shape_zyx: tuple[int, int, int],
    ) -> np.ndarray:
        return self._read(origin_zyx, shape_zyx, probability=True)

    def read_coverage(
        self,
        origin_zyx: tuple[int, int, int],
        shape_zyx: tuple[int, int, int],
    ) -> np.ndarray:
        return self._read(origin_zyx, shape_zyx, probability=False)

    def read_compact_probability(
        self,
        origin_zyx: tuple[int, int, int],
        shape_zyx: tuple[int, int, int],
    ) -> np.ndarray:
        """Read a binary teacher probability into one byte per fine voxel.

        The production teachers in v11.2 are label fields. Keeping the staging
        window compact avoids allocating a multi-gigabyte float32 array before
        its CUDA Gaussian pullback; conversion happens once on the GPU.
        """

        if self.field.encoding != "labels":
            raise ValueError("compact teacher windows require label encoding")
        return self._read(
            origin_zyx,
            shape_zyx,
            probability=True,
            compact_probability=True,
        )

    def read_raw(
        self,
        origin_zyx: tuple[int, int, int],
        shape_zyx: tuple[int, int, int],
        *,
        fill_value: float = 0,
    ) -> np.ndarray:
        """Read raw voxels while respecting sparse declared chunk support.

        Fine CT mirrors use the same sparse chunk lattice as their teacher
        fields.  A rectangular Zarr read can touch zero-byte placeholders in
        unknown space, so reviewer artifacts need the same chunk-safe access
        contract as target construction even though CT values are not decoded
        as labels or probabilities.
        """

        if any(size <= 0 for size in shape_zyx):
            raise ValueError("fine window shape must be positive")
        output = np.full(
            shape_zyx,
            fill_value,
            dtype=np.dtype(self.fine_volume.dtype),
        )
        window_lo = np.asarray(origin_zyx, dtype=np.int64)
        window_hi = window_lo + np.asarray(shape_zyx, dtype=np.int64)
        chunks = np.asarray(self.support.chunks_zyx, dtype=np.int64)
        chunk_lo = np.floor_divide(window_lo, chunks)
        chunk_hi = np.floor_divide(window_hi + chunks - 1, chunks)
        for coordinate in self.support.iter_between(tuple(chunk_lo), tuple(chunk_hi)):
            chunk_origin = np.asarray(coordinate, dtype=np.int64) * chunks
            raw = self._chunk(coordinate)
            chunk_end = chunk_origin + np.asarray(raw.shape, dtype=np.int64)
            lower = np.maximum(window_lo, chunk_origin)
            upper = np.minimum(window_hi, chunk_end)
            if (lower >= upper).any():
                continue
            destination = tuple(
                slice(int(lo - base), int(hi - base))
                for lo, hi, base in zip(lower, upper, window_lo, strict=True)
            )
            source = tuple(
                slice(int(lo - base), int(hi - base))
                for lo, hi, base in zip(lower, upper, chunk_origin, strict=True)
            )
            output[destination] = raw[source]
        return output


def coarse_patch_fine_bounds(
    origin_zyx: tuple[int, int, int],
    shape_zyx: tuple[int, int, int],
    fine_to_coarse_affine_xyz: tuple[tuple[float, float, float, float], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned fine-voxel bounds enclosing a coarse voxel-cell patch."""

    z0, y0, x0 = origin_zyx
    depth, height, width = shape_zyx
    coarse_corners = np.asarray(
        list(
            product(
                (x0 - 0.5, x0 + width - 0.5),
                (y0 - 0.5, y0 + height - 0.5),
                (z0 - 0.5, z0 + depth - 0.5),
            )
        ),
        dtype=np.float64,
    )
    fine_xyz = transform_xyz(coarse_corners, invert_affine(fine_to_coarse_affine_xyz))
    fine_zyx = fine_xyz[:, ::-1]
    lower = np.floor(fine_zyx.min(axis=0) - 1.0).astype(np.int64)
    upper = np.ceil(fine_zyx.max(axis=0) + 1.0).astype(np.int64) + 1
    return lower, upper


def _valid_centers(
    fine_volume: ArrayLike3D,
    field: DenseFieldSpec,
    support: ChunkSupport,
    coarse_to_fine: np.ndarray,
    origin_zyx: tuple[int, int, int],
    shape_zyx: tuple[int, int, int],
    *,
    block: int,
) -> np.ndarray:
    valid = np.zeros(shape_zyx, dtype=bool)
    for z0 in range(0, shape_zyx[0], block):
        for y0 in range(0, shape_zyx[1], block):
            for x0 in range(0, shape_zyx[2], block):
                z1 = min(shape_zyx[0], z0 + block)
                y1 = min(shape_zyx[1], y0 + block)
                x1 = min(shape_zyx[2], x0 + block)
                zz, yy, xx = np.meshgrid(
                    np.arange(origin_zyx[0] + z0, origin_zyx[0] + z1),
                    np.arange(origin_zyx[1] + y0, origin_zyx[1] + y1),
                    np.arange(origin_zyx[2] + x0, origin_zyx[2] + x1),
                    indexing="ij",
                )
                coarse_xyz = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
                fine_xyz = transform_xyz(coarse_xyz, coarse_to_fine)
                fine_zyx = fine_xyz[:, ::-1]
                fine_index = np.floor(fine_zyx + 0.5).astype(np.int64)
                inside = (
                    (fine_index >= 0) & (fine_index < np.asarray(support.shape_zyx))
                ).all(axis=1)
                chunk_coordinates = np.floor_divide(
                    fine_index,
                    np.asarray(support.chunks_zyx, dtype=np.int64),
                )
                present = support.contains_many(chunk_coordinates) & inside
                known = np.zeros(present.shape, dtype=bool)
                for chunk_coordinate in np.unique(chunk_coordinates[present], axis=0):
                    selected = present & (chunk_coordinates == chunk_coordinate).all(
                        axis=1
                    )
                    chunk_origin = chunk_coordinate * np.asarray(
                        support.chunks_zyx, dtype=np.int64
                    )
                    chunk_end = np.minimum(
                        chunk_origin + np.asarray(support.chunks_zyx),
                        np.asarray(support.shape_zyx),
                    )
                    slices = tuple(
                        slice(int(lo), int(hi))
                        for lo, hi in zip(chunk_origin, chunk_end, strict=True)
                    )
                    raw_chunk = np.asarray(fine_volume[slices])
                    local = fine_index[selected] - chunk_origin
                    sampled = raw_chunk[local[:, 0], local[:, 1], local[:, 2]]
                    _, sampled_known = dense_field_masks(sampled, field)
                    known[selected] = sampled_known
                valid[z0:z1, y0:y1, x0:x1] = known.reshape(z1 - z0, y1 - y0, x1 - x0)
    return valid


def _sparse_chunk_coarse_bounds(
    chunk_origin_zyx: np.ndarray,
    chunk_end_zyx: np.ndarray,
    fine_to_coarse: np.ndarray,
    coarse_origin_zyx: tuple[int, int, int],
    coarse_shape_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Conservative local coarse bounds for one transformed fine chunk."""

    fine_lower = chunk_origin_zyx.astype(np.float64) - 0.5
    fine_upper = chunk_end_zyx.astype(np.float64) - 0.5
    fine_corners_xyz = np.asarray(
        list(
            product(
                (fine_lower[2], fine_upper[2]),
                (fine_lower[1], fine_upper[1]),
                (fine_lower[0], fine_upper[0]),
            )
        ),
        dtype=np.float64,
    )
    coarse_corners_zyx = transform_xyz(
        fine_corners_xyz,
        fine_to_coarse,
    )[:, ::-1]
    local_corners = coarse_corners_zyx - np.asarray(
        coarse_origin_zyx,
        dtype=np.float64,
    )
    lower = np.floor(local_corners.min(axis=0) - 1.0).astype(np.int64)
    upper = np.ceil(local_corners.max(axis=0) + 1.0).astype(np.int64) + 1
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, np.asarray(coarse_shape_zyx, dtype=np.int64))
    return lower, upper


def _sparse_chunk_valid_centers(
    raw_chunk: np.ndarray,
    field: DenseFieldSpec,
    chunk_origin_zyx: np.ndarray,
    chunk_end_zyx: np.ndarray,
    fine_to_coarse: np.ndarray,
    coarse_to_fine: np.ndarray,
    coarse_origin_zyx: tuple[int, int, int],
    coarse_shape_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return coarse local coordinates whose nearest fine voxel is known.

    Sparse teachers only define supervision inside materialized fine chunks.
    The legacy validity pass visits every coarse voxel in the 192-cube and then
    discovers that almost all map to absent chunks.  This equivalent inverse
    mapping restricts that work to the conservative transformed bounding box
    of one present chunk; final integer-index filtering preserves the exact
    nearest-fine-voxel contract.
    """

    lower, upper = _sparse_chunk_coarse_bounds(
        chunk_origin_zyx,
        chunk_end_zyx,
        fine_to_coarse,
        coarse_origin_zyx,
        coarse_shape_zyx,
    )
    if (lower >= upper).any():
        return (
            np.empty((0, 3), dtype=np.int64),
            np.empty((0, 3), dtype=np.int64),
        )

    zz, yy, xx = np.meshgrid(
        np.arange(lower[0], upper[0]),
        np.arange(lower[1], upper[1]),
        np.arange(lower[2], upper[2]),
        indexing="ij",
    )
    local_zyx = np.stack((zz, yy, xx), axis=-1).reshape(-1, 3)
    global_zyx = local_zyx + np.asarray(coarse_origin_zyx, dtype=np.int64)
    fine_xyz = transform_xyz(global_zyx[:, ::-1], coarse_to_fine)
    fine_index_zyx = np.floor(fine_xyz[:, ::-1] + 0.5).astype(np.int64)
    belongs = (
        (fine_index_zyx >= chunk_origin_zyx) & (fine_index_zyx < chunk_end_zyx)
    ).all(axis=1)
    if not belongs.any():
        return (
            np.empty((0, 3), dtype=np.int64),
            np.empty((0, 3), dtype=np.int64),
        )
    selected_local = local_zyx[belongs]
    fine_local = fine_index_zyx[belongs] - chunk_origin_zyx
    if field.encoding == "labels" and not field.ignore_labels:
        return selected_local, fine_local
    sampled = raw_chunk[
        fine_local[:, 0],
        fine_local[:, 1],
        fine_local[:, 2],
    ]
    _, sampled_known = dense_field_masks(sampled, field)
    return selected_local[sampled_known], fine_local[sampled_known]


@dataclass(frozen=True)
class SparseChunkProjection:
    """Packed coarse-space contribution of one sparse fine-label chunk."""

    lower_zyx: tuple[int, int, int]
    shape_zyx: tuple[int, int, int]
    known_bits: np.ndarray
    foreground_bits: np.ndarray
    fine_positive_voxels: int


@dataclass(frozen=True)
class SparseProjectionCacheInfo:
    hits: int
    misses: int
    waits: int
    maxsize: int
    currsize: int


class SparseChunkProjectionCache:
    """Reuse exact fine-to-coarse chunk projections across overlapping patches."""

    def __init__(
        self,
        fine_volume: ArrayLike3D,
        field: DenseFieldSpec,
        support: ChunkSupport,
        fine_to_coarse_affine_xyz: tuple[tuple[float, float, float, float], ...],
        coarse_shape_zyx: tuple[int, int, int],
        *,
        max_entries: int,
        enable_cuda_projection: bool | None = None,
    ) -> None:
        if support.present_ids is None:
            raise ValueError("a sparse projection cache requires finite chunk support")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.fine_volume = fine_volume
        self.field = field
        self.support = support
        self.fine_to_coarse_affine_xyz = fine_to_coarse_affine_xyz
        self.coarse_shape_zyx = coarse_shape_zyx
        self._fine_to_coarse = affine_matrix(fine_to_coarse_affine_xyz)
        self._coarse_to_fine = invert_affine(fine_to_coarse_affine_xyz)
        self._max_entries = max_entries
        self._cache: OrderedDict[tuple[int, int, int], SparseChunkProjection] = (
            OrderedDict()
        )
        self._inflight: dict[tuple[int, int, int], Future[SparseChunkProjection]] = {}
        self._lock = Lock()
        self._cuda_lock = Lock()
        self._hits = 0
        self._misses = 0
        self._waits = 0
        self._torch = None
        self._cuda_device = None
        self._cuda_linear = None
        self._cuda_translation = None
        self._cuda_inverse_linear = None
        self._cuda_inverse_translation = None

        cuda_sized_label_chunk = (
            field.encoding == "labels"
            and int(np.prod(support.chunks_zyx, dtype=np.int64)) >= 128**3
        )
        request_cuda = enable_cuda_projection is True or (
            enable_cuda_projection is None and cuda_sized_label_chunk
        )
        if enable_cuda_projection is True and field.encoding != "labels":
            raise ValueError("CUDA sparse projection currently requires label encoding")
        if request_cuda:
            try:
                import torch
            except ImportError as error:
                if enable_cuda_projection is True:
                    raise RuntimeError(
                        "CUDA sparse projection requires PyTorch"
                    ) from error
            else:
                if torch.cuda.is_available():
                    from .resources import assert_cuda_power_limit

                    device = torch.device("cuda")
                    assert_cuda_power_limit(device)
                    self._torch = torch
                    self._cuda_device = device
                    self._cuda_linear = torch.tensor(
                        self._fine_to_coarse[:3, :3],
                        dtype=torch.float64,
                        device=device,
                    )
                    self._cuda_translation = torch.tensor(
                        self._fine_to_coarse[:3, 3],
                        dtype=torch.float64,
                        device=device,
                    )
                    self._cuda_inverse_linear = torch.tensor(
                        self._coarse_to_fine[:3, :3],
                        dtype=torch.float64,
                        device=device,
                    )
                    self._cuda_inverse_translation = torch.tensor(
                        self._coarse_to_fine[:3, 3],
                        dtype=torch.float64,
                        device=device,
                    )
                elif enable_cuda_projection is True:
                    raise RuntimeError(
                        "CUDA sparse projection requested without a CUDA GPU"
                    )

    @property
    def projection_backend(self) -> str:
        return "cuda-float64" if self._torch is not None else "cpu-float64"

    def get(self, coordinate_zyx: tuple[int, int, int]) -> SparseChunkProjection:
        key = tuple(int(item) for item in coordinate_zyx)
        with self._lock:
            cached = self._cache.pop(key, None)
            if cached is not None:
                self._cache[key] = cached
                self._hits += 1
                return cached
            future = self._inflight.get(key)
            owner = future is None
            if owner:
                future = Future()
                self._inflight[key] = future
                self._misses += 1
            else:
                self._waits += 1
        assert future is not None
        if not owner:
            return future.result()
        try:
            projected = self._project(key)
        except BaseException as error:
            with self._lock:
                self._inflight.pop(key, None)
            future.set_exception(error)
            raise
        with self._lock:
            self._cache[key] = projected
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)
            self._inflight.pop(key, None)
        future.set_result(projected)
        return projected

    def cache_info(self) -> SparseProjectionCacheInfo:
        with self._lock:
            return SparseProjectionCacheInfo(
                hits=self._hits,
                misses=self._misses,
                waits=self._waits,
                maxsize=self._max_entries,
                currsize=len(self._cache),
            )

    def clear(self) -> None:
        with self._lock:
            if self._inflight:
                raise RuntimeError("cannot clear a projection cache with active work")
            self._cache.clear()

    def _project_masks_cuda(
        self,
        raw_chunk: np.ndarray,
        chunk_origin_zyx: np.ndarray,
        chunk_end_zyx: np.ndarray,
        lower_zyx: np.ndarray,
        shape_zyx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Project known and foreground masks using voxel-exact float64 CUDA."""

        torch = self._torch
        if (
            torch is None
            or self._cuda_device is None
            or self._cuda_linear is None
            or self._cuda_translation is None
            or self._cuda_inverse_linear is None
            or self._cuda_inverse_translation is None
        ):
            raise RuntimeError("CUDA sparse projection is not enabled")
        shape = tuple(max(0, int(item)) for item in shape_zyx)
        with self._cuda_lock, torch.inference_mode():
            raw_cuda = torch.as_tensor(
                np.ascontiguousarray(raw_chunk),
                device=self._cuda_device,
            )
            positive = torch.zeros_like(raw_cuda, dtype=torch.bool)
            for label in self.field.positive_labels:
                positive.logical_or_(raw_cuda == int(label))
            for label in self.field.ignore_labels:
                positive.logical_and_(raw_cuda != int(label))
            positions_zyx = torch.nonzero(positive, as_tuple=False)
            fine_positive_voxels = int(positions_zyx.shape[0])
            if not all(shape):
                empty = np.zeros(shape, dtype=bool)
                return empty, empty.copy(), fine_positive_voxels

            shape_cuda = torch.as_tensor(
                shape,
                dtype=torch.int64,
                device=self._cuda_device,
            )
            lower_cuda = torch.as_tensor(
                lower_zyx,
                dtype=torch.int64,
                device=self._cuda_device,
            )
            element_count = int(np.prod(shape, dtype=np.int64))
            foreground_flat = torch.zeros(
                element_count,
                dtype=torch.bool,
                device=self._cuda_device,
            )
            if fine_positive_voxels:
                fine_xyz = positions_zyx[:, [2, 1, 0]].to(dtype=torch.float64)
                fine_xyz += torch.as_tensor(
                    chunk_origin_zyx[::-1].copy(),
                    dtype=torch.float64,
                    device=self._cuda_device,
                )
                coarse_xyz = fine_xyz @ self._cuda_linear.T
                coarse_xyz += self._cuda_translation
                global_zyx = torch.floor(coarse_xyz[:, [2, 1, 0]] + 0.5).to(
                    dtype=torch.int64
                )
                local_zyx = global_zyx - lower_cuda
                inside = ((local_zyx >= 0) & (local_zyx < shape_cuda)).all(dim=1)
                local_zyx = local_zyx[inside]
                if local_zyx.numel():
                    linear = (local_zyx[:, 0] * shape[1] + local_zyx[:, 1]) * shape[
                        2
                    ] + local_zyx[:, 2]
                    foreground_flat[linear] = True
                del fine_xyz, coarse_xyz, global_zyx, local_zyx, inside
            del positive, positions_zyx
            foreground = foreground_flat.reshape(shape).cpu().numpy()
            del foreground_flat

            linear = torch.arange(
                element_count,
                dtype=torch.int64,
                device=self._cuda_device,
            )
            plane = shape[1] * shape[2]
            local_z = torch.div(linear, plane, rounding_mode="floor")
            remainder = linear - local_z * plane
            local_y = torch.div(remainder, shape[2], rounding_mode="floor")
            local_x = remainder - local_y * shape[2]
            coarse_xyz = torch.stack((local_x, local_y, local_z), dim=1).to(
                dtype=torch.float64
            )
            coarse_xyz += lower_cuda[[2, 1, 0]].to(dtype=torch.float64)
            fine_xyz = coarse_xyz @ self._cuda_inverse_linear.T
            fine_xyz += self._cuda_inverse_translation
            fine_index_zyx = torch.floor(fine_xyz[:, [2, 1, 0]] + 0.5).to(
                dtype=torch.int64
            )
            chunk_origin_cuda = torch.as_tensor(
                chunk_origin_zyx,
                dtype=torch.int64,
                device=self._cuda_device,
            )
            chunk_end_cuda = torch.as_tensor(
                chunk_end_zyx,
                dtype=torch.int64,
                device=self._cuda_device,
            )
            belongs = (
                (fine_index_zyx >= chunk_origin_cuda)
                & (fine_index_zyx < chunk_end_cuda)
            ).all(dim=1)
            if self.field.ignore_labels and belongs.any():
                selected = torch.nonzero(belongs, as_tuple=False).flatten()
                fine_local = fine_index_zyx[selected] - chunk_origin_cuda
                sampled = raw_cuda[
                    fine_local[:, 0],
                    fine_local[:, 1],
                    fine_local[:, 2],
                ]
                sampled_known = torch.ones_like(sampled, dtype=torch.bool)
                for label in self.field.ignore_labels:
                    sampled_known.logical_and_(sampled != int(label))
                known_flat = torch.zeros_like(belongs)
                known_flat[selected] = sampled_known
            else:
                known_flat = belongs
            known = known_flat.reshape(shape).cpu().numpy()
        return known, foreground, fine_positive_voxels

    def _project(
        self,
        coordinate_zyx: tuple[int, int, int],
    ) -> SparseChunkProjection:
        chunks = np.asarray(self.support.chunks_zyx, dtype=np.int64)
        chunk_origin = np.asarray(coordinate_zyx, dtype=np.int64) * chunks
        chunk_end = np.minimum(
            chunk_origin + chunks,
            np.asarray(self.support.shape_zyx, dtype=np.int64),
        )
        slices = tuple(
            slice(int(lo), int(hi))
            for lo, hi in zip(chunk_origin, chunk_end, strict=True)
        )
        raw_chunk = np.asarray(self.fine_volume[slices])

        if self._torch is not None:
            lower, upper = _sparse_chunk_coarse_bounds(
                chunk_origin,
                chunk_end,
                self._fine_to_coarse,
                (0, 0, 0),
                self.coarse_shape_zyx,
            )
            shape = np.maximum(upper - lower, 0)
            known, foreground, fine_positive_voxels = self._project_masks_cuda(
                raw_chunk,
                chunk_origin,
                chunk_end,
                lower,
                shape,
            )
            if not all(int(item) for item in shape):
                empty = np.empty(0, dtype=np.uint8)
                return SparseChunkProjection(
                    lower_zyx=(0, 0, 0),
                    shape_zyx=(0, 0, 0),
                    known_bits=empty,
                    foreground_bits=empty,
                    fine_positive_voxels=fine_positive_voxels,
                )
            active = np.argwhere(known | foreground)
            if not active.size:
                empty = np.empty(0, dtype=np.uint8)
                return SparseChunkProjection(
                    lower_zyx=(0, 0, 0),
                    shape_zyx=(0, 0, 0),
                    known_bits=empty,
                    foreground_bits=empty,
                    fine_positive_voxels=fine_positive_voxels,
                )
            active_lower = active.min(axis=0)
            active_upper = active.max(axis=0) + 1
            crop = tuple(
                slice(int(lo), int(hi))
                for lo, hi in zip(active_lower, active_upper, strict=True)
            )
            known = known[crop]
            foreground = foreground[crop]
            tight_lower = lower + active_lower
            tight_shape = active_upper - active_lower
            return SparseChunkProjection(
                lower_zyx=tuple(int(item) for item in tight_lower),
                shape_zyx=tuple(int(item) for item in tight_shape),
                known_bits=np.packbits(known, bitorder="little"),
                foreground_bits=np.packbits(foreground, bitorder="little"),
                fine_positive_voxels=fine_positive_voxels,
            )

        known_global, _ = _sparse_chunk_valid_centers(
            raw_chunk,
            self.field,
            chunk_origin,
            chunk_end,
            self._fine_to_coarse,
            self._coarse_to_fine,
            (0, 0, 0),
            self.coarse_shape_zyx,
        )
        positive, _ = dense_field_masks(raw_chunk, self.field)

        positive_local = np.argwhere(positive)
        fine_positive_voxels = int(positive_local.shape[0])
        if positive_local.size:
            fine_zyx = positive_local.astype(np.float64) + chunk_origin
            coarse_xyz = transform_xyz(fine_zyx[:, ::-1], self._fine_to_coarse)
            foreground_global = np.floor(coarse_xyz[:, ::-1] + 0.5).astype(np.int64)
            foreground_global = foreground_global[
                (
                    (foreground_global >= 0)
                    & (
                        foreground_global
                        < np.asarray(self.coarse_shape_zyx, dtype=np.int64)
                    )
                ).all(axis=1)
            ]
        else:
            foreground_global = np.empty((0, 3), dtype=np.int64)

        coordinate_sets = [
            coordinates
            for coordinates in (known_global, foreground_global)
            if coordinates.size
        ]
        if not coordinate_sets:
            empty = np.empty(0, dtype=np.uint8)
            return SparseChunkProjection(
                lower_zyx=(0, 0, 0),
                shape_zyx=(0, 0, 0),
                known_bits=empty,
                foreground_bits=empty,
                fine_positive_voxels=fine_positive_voxels,
            )

        all_coordinates = np.concatenate(coordinate_sets, axis=0)
        lower = all_coordinates.min(axis=0)
        upper = all_coordinates.max(axis=0) + 1
        shape = upper - lower
        known = np.zeros(tuple(int(item) for item in shape), dtype=bool)
        foreground = np.zeros_like(known)
        if known_global.size:
            local = known_global - lower
            known[local[:, 0], local[:, 1], local[:, 2]] = True
        if foreground_global.size:
            local = foreground_global - lower
            foreground[local[:, 0], local[:, 1], local[:, 2]] = True
        return SparseChunkProjection(
            lower_zyx=tuple(int(item) for item in lower),
            shape_zyx=tuple(int(item) for item in shape),
            known_bits=np.packbits(known, bitorder="little"),
            foreground_bits=np.packbits(foreground, bitorder="little"),
            fine_positive_voxels=fine_positive_voxels,
        )


def _apply_sparse_chunk_projection(
    projection: SparseChunkProjection,
    *,
    origin_zyx: tuple[int, int, int],
    shape_zyx: tuple[int, int, int],
    valid: np.ndarray,
    foreground: np.ndarray,
) -> None:
    if not all(projection.shape_zyx):
        return
    patch_lower = np.asarray(origin_zyx, dtype=np.int64)
    patch_upper = patch_lower + np.asarray(shape_zyx, dtype=np.int64)
    entry_lower = np.asarray(projection.lower_zyx, dtype=np.int64)
    entry_upper = entry_lower + np.asarray(projection.shape_zyx, dtype=np.int64)
    lower = np.maximum(patch_lower, entry_lower)
    upper = np.minimum(patch_upper, entry_upper)
    if (lower >= upper).any():
        return

    element_count = int(np.prod(projection.shape_zyx))
    known = (
        np.unpackbits(
            projection.known_bits,
            count=element_count,
            bitorder="little",
        )
        .reshape(projection.shape_zyx)
        .astype(bool, copy=False)
    )
    projected_foreground = (
        np.unpackbits(
            projection.foreground_bits,
            count=element_count,
            bitorder="little",
        )
        .reshape(projection.shape_zyx)
        .astype(bool, copy=False)
    )
    patch_slices = tuple(
        slice(int(lo - patch_lo), int(hi - patch_lo))
        for lo, hi, patch_lo in zip(lower, upper, patch_lower, strict=True)
    )
    entry_slices = tuple(
        slice(int(lo - entry_lo), int(hi - entry_lo))
        for lo, hi, entry_lo in zip(lower, upper, entry_lower, strict=True)
    )
    valid[patch_slices] |= known[entry_slices]
    foreground[patch_slices] |= projected_foreground[entry_slices]


def voxelize_fine_target_patch(
    fine_volume: ArrayLike3D,
    field: DenseFieldSpec,
    support: ChunkSupport,
    fine_to_coarse_affine_xyz: tuple[tuple[float, float, float, float], ...],
    origin_zyx: tuple[int, int, int],
    shape_zyx: tuple[int, int, int],
    *,
    validity_block: int = 64,
    projection_cache: SparseChunkProjectionCache | None = None,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Forward-voxelize a dense fine label field into a coarse patch.

    Every positive fine voxel is transformed and assigned to its nearest
    coarse voxel. No points, Gaussian splats, distance targets, or synthetic
    topology primitives are involved. Unknown chunks in a sparse mirror are
    emitted as nnU-Net ignore label 2 rather than false background.
    """

    if any(size <= 0 for size in shape_zyx):
        raise ValueError("shape_zyx must be positive")
    if validity_block <= 0:
        raise ValueError("validity_block must be positive")
    coarse_to_fine = invert_affine(fine_to_coarse_affine_xyz)
    sparse_support = support.present_ids is not None
    if projection_cache is not None:
        if not sparse_support:
            raise ValueError("projection_cache is only valid for sparse support")
        if projection_cache.fine_volume is not fine_volume:
            raise ValueError("projection_cache belongs to a different fine volume")
        if projection_cache.support is not support:
            raise ValueError("projection_cache belongs to different chunk support")
        if projection_cache.fine_to_coarse_affine_xyz != fine_to_coarse_affine_xyz:
            raise ValueError("projection_cache belongs to a different affine")
    valid = (
        np.zeros(shape_zyx, dtype=bool)
        if sparse_support
        else _valid_centers(
            fine_volume,
            field,
            support,
            coarse_to_fine,
            origin_zyx,
            shape_zyx,
            block=validity_block,
        )
    )
    foreground = np.zeros(shape_zyx, dtype=bool)
    fine_lower, fine_upper = coarse_patch_fine_bounds(
        origin_zyx, shape_zyx, fine_to_coarse_affine_xyz
    )
    chunks = np.asarray(support.chunks_zyx, dtype=np.int64)
    chunk_lower = np.floor_divide(fine_lower, chunks)
    chunk_upper = np.floor_divide(fine_upper + chunks - 1, chunks)
    fine_to_coarse = affine_matrix(fine_to_coarse_affine_xyz)
    chunks_read = 0
    fine_positive_voxels = 0
    coarse_origin_xyz = np.asarray(origin_zyx[::-1], dtype=np.float64)

    for coordinate in support.iter_between(tuple(chunk_lower), tuple(chunk_upper)):
        if projection_cache is not None:
            projection = projection_cache.get(coordinate)
            _apply_sparse_chunk_projection(
                projection,
                origin_zyx=origin_zyx,
                shape_zyx=shape_zyx,
                valid=valid,
                foreground=foreground,
            )
            chunks_read += 1
            fine_positive_voxels += projection.fine_positive_voxels
            continue
        chunk_origin = np.asarray(coordinate) * chunks
        chunk_end = np.minimum(
            chunk_origin + chunks,
            np.asarray(support.shape_zyx),
        )
        slices = tuple(
            slice(int(lo), int(hi))
            for lo, hi in zip(chunk_origin, chunk_end, strict=True)
        )
        raw_chunk = np.asarray(fine_volume[slices])
        positive, _ = dense_field_masks(raw_chunk, field)
        if sparse_support:
            known_local, _ = _sparse_chunk_valid_centers(
                raw_chunk,
                field,
                chunk_origin,
                chunk_end,
                fine_to_coarse,
                coarse_to_fine,
                origin_zyx,
                shape_zyx,
            )
            if known_local.size:
                valid[
                    known_local[:, 0],
                    known_local[:, 1],
                    known_local[:, 2],
                ] = True
        positive_local = np.argwhere(positive)
        chunks_read += 1
        if not positive_local.size:
            continue
        fine_positive_voxels += int(positive_local.shape[0])
        fine_zyx = positive_local.astype(np.float64) + chunk_origin
        coarse_xyz = transform_xyz(fine_zyx[:, ::-1], fine_to_coarse)
        local_xyz = np.floor(coarse_xyz - coarse_origin_xyz + 0.5).astype(np.int64)
        local_zyx = local_xyz[:, ::-1]
        inside = ((local_zyx >= 0) & (local_zyx < np.asarray(shape_zyx))).all(axis=1)
        selected = local_zyx[inside]
        if selected.size:
            foreground[selected[:, 0], selected[:, 1], selected[:, 2]] = True

    valid |= foreground
    labels = np.full(shape_zyx, 2, dtype=np.uint8)
    labels[valid] = 0
    labels[foreground] = 1
    known_voxels = int(valid.sum())
    positive_voxels = int(foreground.sum())
    stats: dict[str, int | float] = {
        "chunks_read": chunks_read,
        "fine_positive_voxels": fine_positive_voxels,
        "known_voxels": known_voxels,
        "positive_voxels": positive_voxels,
        "known_fraction": known_voxels / labels.size,
        "positive_fraction_known": positive_voxels / max(1, known_voxels),
    }
    return labels, stats


def _analytic_sparse_validity(
    *,
    support: ChunkSupport,
    coarse_to_fine: np.ndarray,
    origin_zyx: tuple[int, int, int],
    shape_zyx: tuple[int, int, int],
    erosion: int,
    block_depth: int = 16,
) -> np.ndarray:
    """Conservatively validate coarse centers against sparse chunk support.

    Production fine teachers are label chunks with no per-voxel ignore values.
    A center is valid only when the complete L-infinity filter cube lies in
    materialized chunks. This is slightly stricter than the legacy Euclidean
    coverage erosion at chunk corners, never more permissive, and avoids a
    second dense fine-window read plus a multi-gigabyte EDT.
    """

    if erosion < 0 or erosion >= min(support.chunks_zyx):
        raise ValueError("analytic validity requires sub-chunk non-negative erosion")
    valid = np.zeros(shape_zyx, dtype=np.uint8)
    fine_shape = np.asarray(support.shape_zyx, dtype=np.int64)
    chunks = np.asarray(support.chunks_zyx, dtype=np.int64)
    for z0 in range(0, shape_zyx[0], block_depth):
        z1 = min(shape_zyx[0], z0 + block_depth)
        zz, yy, xx = np.meshgrid(
            np.arange(origin_zyx[0] + z0, origin_zyx[0] + z1),
            np.arange(origin_zyx[1], origin_zyx[1] + shape_zyx[1]),
            np.arange(origin_zyx[2], origin_zyx[2] + shape_zyx[2]),
            indexing="ij",
        )
        coarse_zyx = np.stack((zz, yy, xx), axis=-1).reshape(-1, 3)
        fine_xyz = transform_xyz(coarse_zyx[:, ::-1], coarse_to_fine)
        centers = np.floor(fine_xyz[:, ::-1] + 0.5).astype(np.int64)
        lower = centers - erosion
        upper = centers + erosion
        selected = ((lower >= 0) & (upper < fine_shape)).all(axis=1)
        if support.present_ids is not None and selected.any():
            lower_chunks = np.floor_divide(lower, chunks)
            upper_chunks = np.floor_divide(upper, chunks)
            # With erosion smaller than a chunk, the support cube touches at
            # most two chunks per axis. Requiring every bounding-box corner is
            # an L-infinity erosion and therefore conservative at diagonals.
            for choose_upper in product((False, True), repeat=3):
                coordinates = np.stack(
                    [
                        upper_chunks[:, axis]
                        if choose_upper[axis]
                        else lower_chunks[:, axis]
                        for axis in range(3)
                    ],
                    axis=1,
                )
                selected &= support.contains_many(coordinates)
        valid[z0:z1] = selected.reshape(z1 - z0, shape_zyx[1], shape_zyx[2])
    return valid


def _cuda_antialias_pullback(
    *,
    provider: FineFieldWindowReader,
    support: ChunkSupport,
    fine_to_coarse_affine_xyz: tuple[tuple[float, float, float, float], ...],
    origin_zyx: tuple[int, int, int],
    shape_zyx: tuple[int, int, int],
    options: BridgeOptions,
    minimum_output_voxels: int = 128**3,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float | str]] | None:
    """Run one whole-patch Gaussian pullback on CUDA when it is safe to do so."""

    if (
        provider.field.encoding != "labels"
        or provider.field.ignore_labels
        or options.maxpool_prefilter
        or int(np.prod(shape_zyx, dtype=np.int64)) < minimum_output_voxels
    ):
        return None
    try:
        import torch
        from torch.nn import functional
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    from .resources import assert_cuda_power_limit

    device = torch.device("cuda")
    assert_cuda_power_limit(device)
    scale_ratio = affine_scale_ratio(fine_to_coarse_affine_xyz)
    sigma = options.prefilter_sigma_scale * scale_ratio
    filter_margin = int(np.ceil(3.0 * sigma)) + 2
    erosion = (
        filter_margin if options.erode_filter_margin else 0
    ) + options.coverage_erosion_fine_vox
    if erosion >= min(support.chunks_zyx):
        return None
    coarse_to_fine = invert_affine(fine_to_coarse_affine_xyz)

    z0, y0, x0 = origin_zyx
    depth, height, width = shape_zyx
    coarse_corners_xyz = np.asarray(
        list(
            product(
                (x0, x0 + width - 1),
                (y0, y0 + height - 1),
                (z0, z0 + depth - 1),
            )
        ),
        dtype=np.float64,
    )
    fine_corners_zyx = transform_xyz(coarse_corners_xyz, coarse_to_fine)[:, ::-1]
    lower = np.floor(fine_corners_zyx.min(axis=0)).astype(np.int64) - filter_margin
    upper = np.ceil(fine_corners_zyx.max(axis=0)).astype(np.int64) + filter_margin + 1
    fine_shape = tuple(int(item) for item in (upper - lower))
    fine_elements = int(np.prod(fine_shape, dtype=np.int64))
    if fine_elements <= 0 or fine_elements > 1_000_000_000:
        return None
    free_bytes, _ = torch.cuda.mem_get_info(device)
    # Compact CPU staging plus one fp32 teacher volume and a third-order
    # Gaussian quadrature grid fit comfortably on the 96 GB production GPU.
    # Fail back to the tiled reference path if another workload has consumed it.
    required_bytes = fine_elements * 5 + int(np.prod(shape_zyx)) * 160
    if free_bytes < required_bytes + 2 * 1024**3:
        return None

    compact = provider.read_compact_probability(tuple(lower), fine_shape)
    valid = _analytic_sparse_validity(
        support=support,
        coarse_to_fine=coarse_to_fine,
        origin_zyx=origin_zyx,
        shape_zyx=shape_zyx,
        erosion=erosion,
    )
    q = np.empty(shape_zyx, dtype=np.float32)
    old_deterministic = torch.backends.cudnn.deterministic
    old_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        with torch.inference_mode():
            teacher = (
                torch.from_numpy(compact)
                .to(device=device, dtype=torch.float32)
                .div_(255.0)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            del compact
            if sigma > 0.0:
                offset_1d = (-np.sqrt(3.0) * sigma, 0.0, np.sqrt(3.0) * sigma)
                weight_1d = (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)
                quadrature = list(product(range(3), repeat=3))
                offsets = torch.as_tensor(
                    [
                        (
                            offset_1d[index[2]],
                            offset_1d[index[1]],
                            offset_1d[index[0]],
                        )
                        for index in quadrature
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                weights = torch.as_tensor(
                    [
                        weight_1d[index[0]] * weight_1d[index[1]] * weight_1d[index[2]]
                        for index in quadrature
                    ],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                offsets = torch.zeros((1, 3), dtype=torch.float32, device=device)
                weights = torch.ones(1, dtype=torch.float32, device=device)
            inverse_linear = torch.as_tensor(
                coarse_to_fine[:3, :3], dtype=torch.float32, device=device
            )
            coarse_origin_xyz = np.asarray(origin_zyx[::-1], dtype=np.float64)
            base_fine_xyz = transform_xyz(coarse_origin_xyz[None, :], coarse_to_fine)[0]
            base = torch.as_tensor(base_fine_xyz, dtype=torch.float32, device=device)
            lower_xyz = torch.as_tensor(
                lower[::-1].copy(), dtype=torch.float32, device=device
            )
            fine_shape_xyz = torch.as_tensor(
                fine_shape[::-1], dtype=torch.float32, device=device
            )
            y_values = torch.arange(height, dtype=torch.float32, device=device)
            x_values = torch.arange(width, dtype=torch.float32, device=device)
            for local_z0 in range(0, depth, 16):
                local_z1 = min(depth, local_z0 + 16)
                z_values = torch.arange(
                    local_z0, local_z1, dtype=torch.float32, device=device
                )
                zz, yy, xx = torch.meshgrid(z_values, y_values, x_values, indexing="ij")
                local_xyz = torch.stack((xx, yy, zz), dim=-1)
                fine_xyz = local_xyz @ inverse_linear.T + base
                quadrature_xyz = fine_xyz.unsqueeze(0) + offsets[:, None, None, None]
                normalized = (
                    2.0 * (quadrature_xyz - lower_xyz) / (fine_shape_xyz - 1.0) - 1.0
                )
                sampled = functional.grid_sample(
                    teacher,
                    normalized.reshape(
                        1,
                        offsets.shape[0] * (local_z1 - local_z0),
                        height,
                        width,
                        3,
                    ),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
                sampled = sampled.reshape(
                    offsets.shape[0], local_z1 - local_z0, height, width
                )
                integrated = (sampled * weights[:, None, None, None]).sum(dim=0)
                q[local_z0:local_z1] = integrated.clamp_(0.0, 1.0).cpu().numpy()
    finally:
        torch.backends.cudnn.deterministic = old_deterministic
        torch.backends.cudnn.benchmark = old_benchmark
    q[valid == 0] = 0.0
    return (
        q,
        valid,
        {
            "projection_backend": "cuda-gauss-hermite3-pullback-linf-validity-v1",
            "fine_window_voxels": fine_elements,
            "fine_window_depth": fine_shape[0],
            "fine_window_height": fine_shape[1],
            "fine_window_width": fine_shape[2],
            "filter_sigma_fine_vox": float(sigma),
            "gaussian_quadrature_order_per_axis": 3,
            "validity_erosion_fine_vox": erosion,
        },
    )


def antialias_fine_target_patch(
    fine_volume: ArrayLike3D,
    field: DenseFieldSpec,
    support: ChunkSupport,
    fine_to_coarse_affine_xyz: tuple[tuple[float, float, float, float], ...],
    origin_zyx: tuple[int, int, int],
    shape_zyx: tuple[int, int, int],
    *,
    options: BridgeOptions | None = None,
    reader: FineFieldWindowReader | None = None,
    hard_threshold: float = 0.5,
    cuda_minimum_output_voxels: int = 128**3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float | str]]:
    """Anti-alias a fine teacher mask onto coarse voxel centers.

    This is the conservative inverse-sampling counterpart to
    :func:`voxelize_fine_target_patch`. It Gaussian-prefilters at the measured
    fine/coarse scale, samples with trilinear interpolation, and erodes sparse
    coverage by filter support. It deliberately performs no maximum pooling or
    morphological thickening.

    Returns ``(hard_target_u8, teacher_q_float32, valid_u8, stats)``. Hard
    labels exist for metrics and backwards-compatible audits; training can use
    the soft field directly.
    """

    if not 0.0 <= hard_threshold <= 1.0:
        raise ValueError("hard_threshold must be in [0, 1]")
    if cuda_minimum_output_voxels <= 0:
        raise ValueError("cuda_minimum_output_voxels must be positive")
    options = options or BridgeOptions(
        prefilter_sigma_scale=0.5,
        coverage_erosion_fine_vox=0,
        maxpool_prefilter=False,
        erode_filter_margin=True,
    )
    options.validate()
    provider = reader or FineFieldWindowReader(fine_volume, field, support)
    if (
        provider.fine_volume is not fine_volume
        or provider.field != field
        or provider.support is not support
    ):
        raise ValueError("fine-field reader belongs to different source data")
    accelerated = _cuda_antialias_pullback(
        provider=provider,
        support=support,
        fine_to_coarse_affine_xyz=fine_to_coarse_affine_xyz,
        origin_zyx=origin_zyx,
        shape_zyx=shape_zyx,
        options=options,
        minimum_output_voxels=cuda_minimum_output_voxels,
    )
    if accelerated is None:
        q, valid_u8 = resample_to_coarse(
            provider.read_probability,
            provider.read_coverage,
            coarse_origin_zyx=origin_zyx,
            coarse_shape_zyx=shape_zyx,
            fine_to_coarse_affine_xyz=fine_to_coarse_affine_xyz,
            options=options,
        )
        backend_stats: dict[str, int | float | str] = {
            "projection_backend": "scipy-tiled-gaussian-pullback-edt-validity-v1"
        }
    else:
        q, valid_u8, backend_stats = accelerated
    valid = valid_u8 > 0
    q = np.asarray(q, dtype=np.float32)
    q[~valid] = 0.0
    hard = np.full(shape_zyx, 2, dtype=np.uint8)
    hard[valid] = (q[valid] >= hard_threshold).astype(np.uint8)
    known_voxels = int(valid.sum())
    positive_voxels = int(np.count_nonzero(hard == 1))
    stats: dict[str, int | float | str] = {
        "known_voxels": known_voxels,
        "positive_voxels": positive_voxels,
        "known_fraction": known_voxels / hard.size,
        "positive_fraction_known": positive_voxels / max(1, known_voxels),
        "soft_positive_mass": float(q[valid].sum()),
        "soft_positive_fraction_known": (
            float(q[valid].mean()) if known_voxels else 0.0
        ),
        "fine_chunk_reads": provider.chunk_reads,
        "fine_chunk_cache_hits": provider.cache_hits,
        **backend_stats,
    }
    return hard, q, valid_u8.astype(np.uint8, copy=False), stats
