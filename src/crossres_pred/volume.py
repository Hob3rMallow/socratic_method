from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import tifffile


class VolumeError(ValueError):
    pass


@runtime_checkable
class ArrayLike3D(Protocol):
    shape: tuple[int, ...]

    def __getitem__(self, key: Any) -> Any: ...


def _split_array_key(spec: str) -> tuple[str, str | None]:
    if "::" not in spec:
        return spec, None
    path, key = spec.rsplit("::", 1)
    return path, key or None


def open_volume(spec: str) -> ArrayLike3D:
    """Open a local 3-D array without eagerly materializing it.

    Supported forms are ``file.npy``, a TIFF stack, and
    ``store.zarr[::group/array]``. Remote stores are intentionally not opened
    here: cache or mount them so the exact training bytes can be hashed.
    """

    path_text, key = _split_array_key(spec)
    path = Path(path_text).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        if key:
            raise VolumeError("an .npy volume cannot have an array key")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    elif suffix in {".tif", ".tiff"}:
        if key:
            raise VolumeError("a TIFF volume cannot have an array key")
        try:
            array = tifffile.memmap(path)
        except (ValueError, OSError):
            array = tifffile.imread(path)
    elif suffix == ".zarr" or path.is_dir():
        try:
            import zarr
        except ImportError as error:
            raise VolumeError(
                "Zarr input requires `pip install vesuvius-crossres-pred[zarr]`"
            ) from error
        root = zarr.open(str(path), mode="r")
        if key:
            try:
                array = root[key]
            except KeyError as error:
                raise VolumeError(f"{path}: no Zarr array {key!r}") from error
        elif hasattr(root, "shape"):
            array = root
        elif "0" in root:
            array = root["0"]
        else:
            raise VolumeError(f"{path}: Zarr group needs an explicit ::array/key")
    else:
        raise VolumeError(f"unsupported volume {spec!r}; expected .npy, TIFF, or .zarr")
    shape = tuple(int(item) for item in array.shape)
    if len(shape) != 3:
        raise VolumeError(f"{spec}: expected a 3-D z-y-x array, got shape {shape}")
    return array


def read_crop(
    volume: ArrayLike3D,
    origin_zyx: tuple[int, int, int],
    shape_zyx: tuple[int, int, int],
    *,
    fill_value: float = 0,
) -> np.ndarray:
    """Read a z-y-x crop, padding portions outside the source."""

    if len(volume.shape) != 3:
        raise VolumeError(f"expected a 3-D volume, got {volume.shape}")
    if any(size <= 0 for size in shape_zyx):
        raise VolumeError(f"crop shape must be positive, got {shape_zyx}")
    output = np.full(shape_zyx, fill_value, dtype=np.dtype(volume.dtype))
    source_slices: list[slice] = []
    destination_slices: list[slice] = []
    for origin, size, extent in zip(origin_zyx, shape_zyx, volume.shape):
        source_lo = max(0, origin)
        source_hi = min(int(extent), origin + size)
        if source_hi <= source_lo:
            return output
        destination_lo = source_lo - origin
        destination_hi = destination_lo + (source_hi - source_lo)
        source_slices.append(slice(source_lo, source_hi))
        destination_slices.append(slice(destination_lo, destination_hi))
    output[tuple(destination_slices)] = np.asarray(volume[tuple(source_slices)])
    return np.ascontiguousarray(output)


def read_points(
    volume: ArrayLike3D,
    points_zyx: np.ndarray,
    *,
    fill_value: float = 0,
) -> np.ndarray:
    """Read integer point coordinates, grouping Zarr access by source chunk."""

    points = np.asarray(points_zyx)
    if points.ndim != 2 or points.shape[1] != 3:
        raise VolumeError(f"points must have shape (N, 3), got {points.shape}")
    if not np.issubdtype(points.dtype, np.integer):
        raise VolumeError("point coordinates must be integers")
    coordinates = points.astype(np.int64, copy=False)
    output = np.full(
        coordinates.shape[0],
        fill_value,
        dtype=np.dtype(volume.dtype),
    )
    if not coordinates.size:
        return output

    extent = np.asarray(volume.shape, dtype=np.int64)
    inside = ((coordinates >= 0) & (coordinates < extent)).all(axis=1)
    if not inside.any():
        return output
    destination_indices = np.flatnonzero(inside)
    source_points = coordinates[inside]

    raw_chunks = getattr(volume, "chunks", None)
    if (
        raw_chunks is None
        or len(raw_chunks) != 3
        or any(not isinstance(value, int) or value <= 0 for value in raw_chunks)
    ):
        z, y, x = source_points.T
        output[destination_indices] = np.asarray(volume[z, y, x])
        return output

    chunks = np.asarray(raw_chunks, dtype=np.int64)
    chunk_coordinates = source_points // chunks
    grid = np.ceil(extent / chunks).astype(np.int64)
    linear = (chunk_coordinates[:, 0] * grid[1] + chunk_coordinates[:, 1]) * grid[
        2
    ] + chunk_coordinates[:, 2]
    for chunk_id in np.unique(linear):
        selected = linear == chunk_id
        chunk_coordinate = chunk_coordinates[np.flatnonzero(selected)[0]]
        lower = chunk_coordinate * chunks
        upper = np.minimum(lower + chunks, extent)
        source_slices = tuple(
            slice(int(lo), int(hi)) for lo, hi in zip(lower, upper, strict=True)
        )
        block = np.asarray(volume[source_slices])
        local = source_points[selected] - lower
        z, y, x = local.T
        output[destination_indices[selected]] = block[z, y, x]
    return output


def validate_matching_volumes(
    first: ArrayLike3D, second: ArrayLike3D, first_name: str, second_name: str
) -> None:
    if tuple(first.shape) != tuple(second.shape):
        raise VolumeError(
            f"{first_name} shape {tuple(first.shape)} does not match "
            f"{second_name} shape {tuple(second.shape)}"
        )
