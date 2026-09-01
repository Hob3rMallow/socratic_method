from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import tifffile

from ..pathmap import remap_volume_spec
from .schema import DenseFieldSpec


class VoxelIOError(ValueError):
    pass


def _enable_zarr_v2_one_byte_dtype_aliases() -> None:
    """Accept byte-order markers that are immaterial for one-byte V2 dtypes.

    Zarr V2 stores NumPy dtype strings and therefore permits ``<u1`` even
    though byte order has no meaning for a one-byte value. Zarr 3.3 only
    registers the canonical ``|u1`` spelling. Extend its existing dtype
    wrappers rather than rewriting provenance-bound source metadata.
    """

    try:
        from zarr.core.dtype.npy.int import Int8, UInt8
    except ImportError:  # Zarr 2, or a future public parser without this issue.
        return
    for dtype_class, suffix in ((UInt8, "u1"), (Int8, "i1")):
        aliases = tuple(f"{marker}{suffix}" for marker in ("<", ">", "="))
        dtype_class._zarr_v2_names = tuple(  # type: ignore[attr-defined]
            dict.fromkeys((*dtype_class._zarr_v2_names, *aliases))  # type: ignore[attr-defined]
        )


@runtime_checkable
class ArrayLike3D(Protocol):
    shape: tuple[int, ...]
    dtype: np.dtype[Any]

    def __getitem__(self, key: Any) -> Any: ...


def split_volume_spec(spec: str) -> tuple[Path, str | None]:
    spec = remap_volume_spec(spec)
    path_text, separator, key = spec.rpartition("::")
    if not separator:
        path_text, key = spec, ""
    return Path(path_text).expanduser().resolve(), key or None


def open_volume(spec: str) -> ArrayLike3D:
    path, key = split_volume_spec(spec)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        if key:
            raise VoxelIOError("an NPY volume cannot have an array key")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    elif suffix in {".tif", ".tiff"}:
        if key:
            raise VoxelIOError("a TIFF volume cannot have an array key")
        try:
            array = tifffile.memmap(path)
        except (ValueError, OSError):
            array = tifffile.imread(path)
    elif suffix == ".zarr" or path.is_dir():
        try:
            import zarr
        except ImportError as error:  # pragma: no cover - optional dependency
            raise VoxelIOError("Zarr input requires the zarr extra") from error
        _enable_zarr_v2_one_byte_dtype_aliases()
        root = zarr.open(str(path), mode="r")
        if key:
            try:
                array = root[key]
            except KeyError as error:
                raise VoxelIOError(f"{path}: no array {key!r}") from error
        elif hasattr(root, "shape"):
            array = root
        elif "0" in root:
            array = root["0"]
        else:
            raise VoxelIOError(f"{path}: specify a Zarr array with ::key")
    else:
        raise VoxelIOError(f"unsupported dense volume {spec!r}")
    if len(array.shape) != 3:
        raise VoxelIOError(f"{spec}: expected z-y-x shape, got {array.shape}")
    return array


@dataclass(frozen=True)
class ArrayMetadata:
    """Layout of a dense array, independent of how it is stored on disk.

    Reading ``<store>/<key>/.zarray`` by hand only works for Zarr V2 with one
    file per chunk. Everything that needs a shape or a chunk grid goes through
    here instead, so a store can change physical layout -- V3, sharding -- while
    the logical grid every inventory is keyed on stays the contract.
    """

    shape_zyx: tuple[int, int, int]
    chunks_zyx: tuple[int, int, int]
    dtype: np.dtype[Any]
    fill_value: Any
    dimension_separator: str | None
    zarr_format: int
    shards_zyx: tuple[int, int, int] | None


def array_metadata(spec: str) -> ArrayMetadata:
    """Describe a dense volume's logical grid via the array API, never the files.

    ``chunks_zyx`` is always the *inner* chunk shape: under Zarr V3 sharding
    ``Array.chunks`` reports the inner chunk and ``Array.shards`` the outer one,
    so the chunk grid an inventory was built against survives a re-shard.
    """

    array = open_volume(spec)
    shape = tuple(int(item) for item in array.shape)
    raw_chunks = getattr(array, "chunks", None)
    if raw_chunks is None or len(raw_chunks) != len(shape):
        chunks = shape
    else:
        chunks = tuple(int(item) for item in raw_chunks)
    raw_shards = getattr(array, "shards", None)
    shards = (
        tuple(int(item) for item in raw_shards)
        if raw_shards is not None and len(raw_shards) == len(shape)
        else None
    )
    metadata = getattr(array, "metadata", None)
    zarr_format = int(getattr(metadata, "zarr_format", 0) or 0)
    separator = getattr(metadata, "dimension_separator", None)
    fill_value = getattr(array, "fill_value", None)
    if fill_value is None:
        fill_value = getattr(metadata, "fill_value", None)
    return ArrayMetadata(
        shape_zyx=shape,  # type: ignore[arg-type]
        chunks_zyx=chunks,  # type: ignore[arg-type]
        dtype=np.dtype(array.dtype),
        fill_value=fill_value,
        dimension_separator=str(separator) if separator is not None else None,
        zarr_format=zarr_format,
        shards_zyx=shards,  # type: ignore[arg-type]
    )


def read_crop(
    volume: ArrayLike3D,
    origin_zyx: tuple[int, int, int],
    shape_zyx: tuple[int, int, int],
    *,
    fill_value: float = 0,
) -> np.ndarray:
    if any(size <= 0 for size in shape_zyx):
        raise VoxelIOError(f"crop shape must be positive, got {shape_zyx}")
    output = np.full(shape_zyx, fill_value, dtype=np.dtype(volume.dtype))
    source: list[slice] = []
    destination: list[slice] = []
    for origin, size, extent in zip(origin_zyx, shape_zyx, volume.shape, strict=True):
        lo = max(0, origin)
        hi = min(int(extent), origin + size)
        if hi <= lo:
            return output
        source.append(slice(lo, hi))
        destination.append(slice(lo - origin, hi - origin))
    output[tuple(destination)] = np.asarray(volume[tuple(source)])
    return np.ascontiguousarray(output)


def decode_dense_field(raw: np.ndarray, spec: DenseFieldSpec) -> np.ndarray:
    """Decode a declared label/probability array to foreground probability."""

    value = np.asarray(raw)
    if spec.encoding == "labels":
        return np.isin(value, spec.positive_labels).astype(np.float32)
    result = np.nan_to_num(value.astype(np.float32), copy=False)
    result /= spec.probability_scale
    return np.clip(result, 0.0, 1.0)


def dense_field_masks(
    raw: np.ndarray, spec: DenseFieldSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Return foreground and explicitly-known masks for a dense field.

    Sparse-chunk presence is handled separately by ChunkSupport. This handles
    unknown values inside a materialized chunk, notably the official manual
    surface label value 2.
    """

    value = np.asarray(raw)
    if spec.encoding == "labels":
        positive = np.isin(value, spec.positive_labels)
        known = (
            ~np.isin(value, spec.ignore_labels)
            if spec.ignore_labels
            else np.ones(value.shape, dtype=bool)
        )
        return positive & known, known
    probability = value.astype(np.float32, copy=False) / spec.probability_scale
    known = np.isfinite(probability)
    positive = np.zeros(value.shape, dtype=bool)
    positive[known] = probability[known] >= spec.threshold
    return positive, known
