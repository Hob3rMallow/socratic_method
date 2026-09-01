from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from .tifxyz import TifxyzError, _find_component, _read_2d


class SparseZarrError(ValueError):
    pass


@dataclass(frozen=True)
class ZarrArraySpec:
    shape_zyx: tuple[int, int, int]
    chunks_zyx: tuple[int, int, int]
    dimension_separator: str

    @classmethod
    def from_metadata(cls, raw: bytes | str | dict[str, object]) -> ZarrArraySpec:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        metadata = json.loads(raw) if isinstance(raw, str) else raw
        try:
            zarr_format = int(metadata["zarr_format"])
            shape = tuple(int(value) for value in metadata["shape"])
            chunks = tuple(int(value) for value in metadata["chunks"])
            separator = str(metadata.get("dimension_separator", "."))
        except (KeyError, TypeError, ValueError) as error:
            raise SparseZarrError(f"invalid .zarray metadata: {error}") from error
        if zarr_format != 2:
            raise SparseZarrError(
                f"only Zarr v2 arrays are supported, got format {zarr_format}"
            )
        if len(shape) != 3 or len(chunks) != 3:
            raise SparseZarrError(
                f"expected a 3-D array, got shape={shape}, chunks={chunks}"
            )
        if any(value <= 0 for value in (*shape, *chunks)):
            raise SparseZarrError(
                f"shape and chunks must be positive, got {shape}, {chunks}"
            )
        if separator not in {".", "/"}:
            raise SparseZarrError(f"unsupported Zarr dimension separator {separator!r}")
        return cls(
            shape_zyx=shape,
            chunks_zyx=chunks,
            dimension_separator=separator,
        )

    @property
    def chunk_grid_zyx(self) -> tuple[int, int, int]:
        return tuple(
            math.ceil(size / chunk)
            for size, chunk in zip(self.shape_zyx, self.chunks_zyx, strict=True)
        )

    @property
    def chunk_count(self) -> int:
        grid_z, grid_y, grid_x = self.chunk_grid_zyx
        return grid_z * grid_y * grid_x

    def encode_chunk(self, chunk_zyx: tuple[int, int, int]) -> int:
        z, y, x = chunk_zyx
        grid_z, grid_y, grid_x = self.chunk_grid_zyx
        if not (0 <= z < grid_z and 0 <= y < grid_y and 0 <= x < grid_x):
            raise SparseZarrError(
                f"chunk {chunk_zyx} is outside chunk grid {self.chunk_grid_zyx}"
            )
        return (z * grid_y + y) * grid_x + x

    def decode_chunk(self, chunk_id: int) -> tuple[int, int, int]:
        _, grid_y, grid_x = self.chunk_grid_zyx
        if not 0 <= chunk_id < self.chunk_count:
            raise SparseZarrError(
                f"chunk id {chunk_id} is outside chunk grid {self.chunk_grid_zyx}"
            )
        z, remainder = divmod(chunk_id, grid_y * grid_x)
        y, x = divmod(remainder, grid_x)
        return z, y, x

    def chunk_key(self, chunk_id: int) -> str:
        return self.dimension_separator.join(
            str(value) for value in self.decode_chunk(chunk_id)
        )


@dataclass(frozen=True)
class TifxyzChunkSelection:
    chunk_ids: frozenset[int]
    map_shape: tuple[int, int]
    sampled_points: int
    valid_points: int
    out_of_bounds_points: int


def select_tifxyz_chunks(
    directory: str | Path,
    spec: ZarrArraySpec,
    *,
    stride: int = 1,
    block_rows: int = 512,
) -> TifxyzChunkSelection:
    """Select chunks containing valid fine-volume TIFXYZ coordinates.

    Coordinates are streamed from memory-mapped component TIFFs in row blocks,
    avoiding the full four-channel materialization used for geometry training.
    """

    source = Path(directory).expanduser()
    if not source.is_dir():
        raise TifxyzError(f"{source}: TIFXYZ directory does not exist")
    if stride < 1:
        raise SparseZarrError("stride must be >= 1")
    if block_rows < 1:
        raise SparseZarrError("block_rows must be >= 1")

    components = [
        _read_2d(_find_component(source, axis, required=True))
        for axis in ("x", "y", "z")
    ]
    shapes = {tuple(component.shape) for component in components}
    if len(shapes) != 1:
        raise TifxyzError(f"{source}: x/y/z shapes do not match: {sorted(shapes)}")
    height, width = components[0].shape

    mask_path = None
    for stem in ("mask", "valid"):
        mask_path = _find_component(source, stem, required=False)
        if mask_path is not None:
            break
    mask = _read_2d(mask_path) if mask_path is not None else None
    if mask is not None and mask.shape != (height, width):
        raise TifxyzError(
            f"{mask_path}: mask shape {mask.shape} does not match "
            f"coordinates {(height, width)}"
        )

    shape_z, shape_y, shape_x = spec.shape_zyx
    chunk_z, chunk_y, chunk_x = spec.chunks_zyx
    _, grid_y, grid_x = spec.chunk_grid_zyx
    selected: set[int] = set()
    sampled_points = 0
    valid_points = 0
    out_of_bounds_points = 0

    row_step = block_rows * stride
    column_slice = slice(0, width, stride)
    for row_start in range(0, height, row_step):
        row_slice = slice(row_start, min(row_start + row_step, height), stride)
        x = np.asarray(components[0][row_slice, column_slice])
        y = np.asarray(components[1][row_slice, column_slice])
        z = np.asarray(components[2][row_slice, column_slice])
        sampled_points += int(x.size)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if mask is not None:
            valid &= np.asarray(mask[row_slice, column_slice]) != 0
        else:
            valid &= (x != 0) | (y != 0) | (z != 0)
        inside = (
            valid
            & (x >= 0)
            & (x < shape_x)
            & (y >= 0)
            & (y < shape_y)
            & (z >= 0)
            & (z < shape_z)
        )
        out_of_bounds_points += int(np.count_nonzero(valid & ~inside))
        valid_points += int(np.count_nonzero(inside))
        if not np.any(inside):
            continue
        chunk_x_values = np.floor(x[inside] / chunk_x).astype(np.int64)
        chunk_y_values = np.floor(y[inside] / chunk_y).astype(np.int64)
        chunk_z_values = np.floor(z[inside] / chunk_z).astype(np.int64)
        linear = (chunk_z_values * grid_y + chunk_y_values) * grid_x
        linear += chunk_x_values
        selected.update(int(value) for value in np.unique(linear))

    return TifxyzChunkSelection(
        chunk_ids=frozenset(selected),
        map_shape=(int(height), int(width)),
        sampled_points=sampled_points,
        valid_points=valid_points,
        out_of_bounds_points=out_of_bounds_points,
    )


def expand_chunk_ids(
    chunk_ids: frozenset[int] | set[int],
    spec: ZarrArraySpec,
    *,
    halo_vox: int,
) -> frozenset[int]:
    """Conservatively dilate selected chunks to cover a voxel-space halo."""

    if halo_vox < 0:
        raise SparseZarrError("halo_vox must be >= 0")
    if not chunk_ids or halo_vox == 0:
        return frozenset(chunk_ids)

    grid_z, grid_y, grid_x = spec.chunk_grid_zyx
    radius_z, radius_y, radius_x = (
        math.ceil(halo_vox / chunk) for chunk in spec.chunks_zyx
    )
    decoded = np.asarray(
        [spec.decode_chunk(chunk_id) for chunk_id in chunk_ids],
        dtype=np.int64,
    )
    expanded: set[int] = set()
    for offset_z, offset_y, offset_x in product(
        range(-radius_z, radius_z + 1),
        range(-radius_y, radius_y + 1),
        range(-radius_x, radius_x + 1),
    ):
        z = decoded[:, 0] + offset_z
        y = decoded[:, 1] + offset_y
        x = decoded[:, 2] + offset_x
        inside = (
            (z >= 0) & (z < grid_z) & (y >= 0) & (y < grid_y) & (x >= 0) & (x < grid_x)
        )
        linear = (z[inside] * grid_y + y[inside]) * grid_x + x[inside]
        expanded.update(int(value) for value in np.unique(linear))
    return frozenset(expanded)
