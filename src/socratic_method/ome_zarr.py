"""Write a surface prediction as an OME-NGFF store in the published M7 shape.

The Vesuvius Challenge publishes every M7 surface prediction as a Zarr v2 group
carrying OME-NGFF 0.4 ``multiscales`` metadata, six nearest-downsampled levels of
binary ``uint8`` (0 or 255), blosc/zstd chunks and a ``metadata.json`` sidecar
describing the inference run.  Those stores are what Volume Cartographer 3D opens
over the CT.  This module writes the same thing for a C3-F0 prediction, so the
result drops into the same tooling.

Two deliberate differences from the published stores, both recorded in the
sidecar: chunks are 128 rather than 192, so one prediction cube is exactly one
chunk and no voxel is ever re-chunked or read back to be modified; and the level
count is configurable.

The store is *sparse and in global scroll coordinates*: level 0 has the shape of
the whole volume, only the chunks a prediction touches are written, and
everything else reads back as the fill value.  A viewer therefore overlays it on
the CT with no offset arithmetic.  The published stores are sparse in exactly the
same way (level 0 of the PHerc1447 M7 prediction holds 74,683 of 245,872 chunks).
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

#: Codec of the published surface stores, verbatim from their ``0/.zarray``:
#: ``{"id": "blosc", "cname": "zstd", "clevel": 1, "shuffle": 1, "blocksize": 0}``.
COMPRESSOR = {"cname": "zstd", "clevel": 1, "shuffle": 1, "blocksize": 0}
NGFF_VERSION = "0.4"
AXES = ("z", "y", "x")
FOREGROUND = 255


class OmeZarrError(ValueError):
    pass


def level_shape(shape_zyx: Sequence[int], level: int) -> tuple[int, int, int]:
    """Shape of pyramid ``level``: ceiling division by ``2**level`` per axis.

    Matches the published stores, whose six level shapes are exactly
    ``ceil(shape0 / 2**k)`` on every axis.
    """

    if level < 0:
        raise OmeZarrError("level must be non-negative")
    if len(shape_zyx) != 3 or any(int(size) <= 0 for size in shape_zyx):
        raise OmeZarrError(f"shape must be three positive sizes, got {tuple(shape_zyx)}")
    factor = 1 << level
    return tuple(-(-int(size) // factor) for size in shape_zyx)  # type: ignore[return-value]


def multiscale_attrs(levels: int) -> dict[str, Any]:
    """The ``.zattrs`` body of the published stores, for ``levels`` datasets."""

    if levels < 1:
        raise OmeZarrError("levels must be at least 1")
    return {
        "multiscales": [
            {
                "axes": [{"name": name, "type": "space"} for name in AXES],
                "datasets": [
                    {
                        "coordinateTransformations": [
                            {"scale": [float(1 << level)] * 3, "type": "scale"}
                        ],
                        "path": str(level),
                    }
                    for level in range(levels)
                ],
                "name": "/",
                "version": NGFF_VERSION,
            }
        ]
    }


def _blosc_compressor():
    try:
        from numcodecs import Blosc
    except ImportError as error:  # pragma: no cover - optional dependency
        raise OmeZarrError(
            "writing an OME-NGFF store requires the zarr extra (numcodecs, zarr)"
        ) from error
    return Blosc(**COMPRESSOR)


def _origin_of(cube_id: str) -> tuple[int, int, int]:
    pieces = cube_id.split("_")
    if len(pieces) != 3 or [piece[:1] for piece in pieces] != ["z", "y", "x"]:
        raise OmeZarrError(f"invalid cube ID: {cube_id!r}")
    try:
        return tuple(int(piece[1:]) for piece in pieces)  # type: ignore[return-value]
    except ValueError as error:
        raise OmeZarrError(f"invalid cube ID: {cube_id!r}") from error


def _chunk_groups(
    origins: Iterable[tuple[int, int, int]], *, chunk: int, level: int
) -> dict[tuple[int, int, int], list[tuple[int, int, int]]]:
    """Map each output chunk of ``level`` to the cube origins feeding it.

    A cube of ``chunk`` voxels at origin ``o`` contributes ``mask[::2**level]``,
    a ``chunk >> level`` block, at output voxel ``o >> level``.  Grouping first
    means every output chunk is assembled once in one buffer and written once:
    no read-modify-write, and peak memory is one chunk plus one cube.
    """

    span = chunk << level  # level-0 voxels spanned by one output chunk
    groups: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for origin in origins:
        key = tuple(value // span for value in origin)
        groups.setdefault(key, []).append(origin)  # type: ignore[arg-type]
    for members in groups.values():
        members.sort()
    return groups


def write_surface_pyramid(
    output_path: str | Path,
    *,
    cube_ids: Sequence[str],
    read_cube: Callable[[str], np.ndarray],
    shape_zyx: Sequence[int],
    chunk: int = 128,
    levels: int = 6,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``cube_ids`` as a sparse OME-NGFF pyramid and return a summary.

    ``read_cube`` returns one cube's binary mask as ``uint8``; values must be 0
    or 255, which is what the engine's ``cubes_PRED`` TIFFs and the published
    stores both hold.  Cube origins must be multiples of ``chunk``.
    """

    try:
        import zarr
    except ImportError as error:  # pragma: no cover - optional dependency
        raise OmeZarrError("writing an OME-NGFF store requires the zarr extra") from error

    if chunk <= 0:
        raise OmeZarrError("chunk must be positive")
    if levels < 1:
        raise OmeZarrError("levels must be at least 1")
    if (chunk >> (levels - 1)) << (levels - 1) != chunk:
        raise OmeZarrError(
            f"chunk {chunk} is not divisible by 2**{levels - 1}; "
            "nearest downsampling would not land on the cube lattice"
        )
    if not cube_ids:
        raise OmeZarrError("no cubes to write")

    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise OmeZarrError(f"prediction store already exists: {output}")
    shape = tuple(int(size) for size in shape_zyx)
    origins = [_origin_of(cube_id) for cube_id in cube_ids]
    for cube_id, origin in zip(cube_ids, origins, strict=True):
        if any(value % chunk for value in origin):
            raise OmeZarrError(f"cube {cube_id} is not aligned to the {chunk}-voxel lattice")
        if any(value + chunk > size for value, size in zip(origin, shape, strict=True)):
            raise OmeZarrError(f"cube {cube_id} extends past the volume shape {shape}")

    by_id = dict(zip(origins, cube_ids, strict=True))
    compressor = _blosc_compressor()
    temporary = output.with_name(output.name + f".partial-{os.getpid()}")
    if temporary.exists():
        raise OmeZarrError(f"stale prediction-store temporary exists: {temporary}")

    written: list[int] = []
    try:
        root = zarr.open_group(str(temporary), mode="w", zarr_format=2)
        arrays = []
        for level in range(levels):
            arrays.append(
                root.create_array(
                    name=str(level),
                    shape=level_shape(shape, level),
                    chunks=(chunk,) * 3,
                    dtype="uint8",
                    fill_value=0,
                    compressor=compressor,
                    chunk_key_encoding={"name": "v2", "separator": "/"},
                )
            )
        root.attrs.update(multiscale_attrs(levels))

        for level, array in enumerate(arrays):
            count = 0
            extent = level_shape(shape, level)
            for key, members in sorted(_chunk_groups(origins, chunk=chunk, level=level).items()):
                buffer = np.zeros((chunk,) * 3, dtype=np.uint8)
                block = chunk >> level
                touched = False
                for origin in members:
                    mask = np.asarray(read_cube(by_id[origin]))
                    if mask.dtype != np.uint8 or mask.shape != (chunk,) * 3:
                        raise OmeZarrError(
                            f"cube {by_id[origin]}: expected uint8 {(chunk,) * 3}, "
                            f"got {mask.dtype} {mask.shape}"
                        )
                    if level:
                        mask = mask[:: 1 << level, :: 1 << level, :: 1 << level]
                    at = tuple(
                        (value >> level) - base * chunk
                        for value, base in zip(origin, key, strict=True)
                    )
                    buffer[
                        at[0] : at[0] + block,
                        at[1] : at[1] + block,
                        at[2] : at[2] + block,
                    ] = mask
                    touched = touched or bool(mask.any())
                if not touched:
                    continue  # leave it absent: it reads back as the fill value
                stop = tuple(
                    min((base + 1) * chunk, size) for base, size in zip(key, extent, strict=True)
                )
                start = tuple(base * chunk for base in key)
                array[
                    start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]
                ] = buffer[
                    : stop[0] - start[0], : stop[1] - start[1], : stop[2] - start[2]
                ]
                count += 1
            written.append(count)

        if metadata is not None:
            (temporary / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "path": str(output),
        "format": f"ome-ngff-{NGFF_VERSION}",
        "zarr_format": 2,
        "shape_zyx": list(shape),
        "chunks_zyx": [chunk] * 3,
        "levels": levels,
        "downsampling": "nearest",
        "dtype": "uint8",
        "values": [0, FOREGROUND],
        "compressor": {"id": "blosc", **COMPRESSOR},
        "chunks_written": {str(level): count for level, count in enumerate(written)},
        "cube_count": len(cube_ids),
    }


def validate_ome_ngff(store_path: str | Path) -> dict[str, Any]:
    """Check a store against the contract this project's own audits enforce.

    The geometry workspace's alignment audit fail-closes unless a store has
    exactly one multiscale, z/y/x axes, a dataset ``0`` and an identity scale on
    that dataset.  Anything we write has to pass the same gate.
    """

    store = Path(store_path).expanduser().resolve()
    attributes_path = store / ".zattrs"
    if not (store / ".zgroup").is_file() or not attributes_path.is_file():
        raise OmeZarrError(f"{store}: not a Zarr v2 group (.zgroup/.zattrs missing)")
    attributes = json.loads(attributes_path.read_text(encoding="utf-8"))
    multiscales = attributes.get("multiscales")
    if not isinstance(multiscales, list) or len(multiscales) != 1:
        raise OmeZarrError(f"{store}: expected one OME-NGFF multiscale")
    multiscale = multiscales[0]
    axes = [axis.get("name") for axis in multiscale.get("axes", [])]
    if axes != list(AXES):
        raise OmeZarrError(f"{store}: expected axes {list(AXES)}, got {axes}")
    datasets = multiscale.get("datasets", [])
    paths = [dataset.get("path") for dataset in datasets]
    if "0" not in paths:
        raise OmeZarrError(f"{store}: OME-NGFF metadata lacks dataset 0")
    identity = datasets[paths.index("0")].get("coordinateTransformations")
    if identity != [{"scale": [1.0, 1.0, 1.0], "type": "scale"}]:
        raise OmeZarrError(f"{store}: L0 must have only identity scale metadata")
    for path in paths:
        if not (store / str(path) / ".zarray").is_file():
            raise OmeZarrError(f"{store}: dataset {path} has no .zarray")
    level_zero = json.loads((store / "0" / ".zarray").read_text(encoding="utf-8"))
    if level_zero.get("dtype") != "|u1":
        raise OmeZarrError(f"{store}: expected |u1, got {level_zero.get('dtype')}")
    if level_zero.get("dimension_separator") != "/":
        raise OmeZarrError(f"{store}: expected nested chunk keys")
    return {
        "levels": len(paths),
        "shape_zyx": list(level_zero["shape"]),
        "chunks_zyx": list(level_zero["chunks"]),
        "dtype": level_zero["dtype"],
        "compressor": level_zero.get("compressor"),
        "version": multiscale.get("version"),
    }
