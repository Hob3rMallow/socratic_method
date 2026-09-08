from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from typing import Any, Self

import numpy as np
from scipy import ndimage
from skimage import __version__ as skimage_version
from skimage.morphology import closing, skeletonize

from .registration import (
    ChunkSupport,
    FineFieldWindowReader,
    affine_matrix,
    coarse_patch_fine_bounds,
    invert_affine,
    transform_xyz,
)

VILLA_MEDIAL_SURFACE_CONTRACT = (
    "villa-f9dacc7-slicewise-skeleton-close-no-tube-v1"
)
VILLA_MEDIAL_SURFACE_SOURCE_COMMIT = "f9dacc741007"
VILLA_MEDIAL_SURFACE_SOURCE_SHA256 = (
    "33e324c8a16be220fe0dc867a9c0ff9bc2b2eaa0265c6a5bcf091c8943297061"
)
VILLA_CENTER_RADIUS_CONTRACT = (
    "villa-f9dacc7-slicewise-centers-with-physical-edt-radius-v1"
)
MEDIAL_MAX_PROJECTION_CONTRACT = (
    "fine-medial-indicator-nearest-coarse-or-max-v1"
)
DEFAULT_MEDIAL_HALO_ZYX = (1, 32, 32)


@dataclass(frozen=True)
class MedialProjectionOptions:
    """Pinned construction options for a fine medial sheet in coarse space."""

    halo_zyx: tuple[int, int, int] = DEFAULT_MEDIAL_HALO_ZYX
    skeleton_workers: int = 1
    max_cache_chunks: int = 64

    def validate(self) -> None:
        if len(self.halo_zyx) != 3 or any(value < 0 for value in self.halo_zyx):
            raise ValueError("medial halo must contain three non-negative values")
        if self.halo_zyx[0] < 1:
            raise ValueError("medial z halo must cover the 3-D closing footprint")
        if self.halo_zyx[1] <= 0 or self.halo_zyx[2] <= 0:
            raise ValueError("medial in-plane halo must be positive")
        if not 1 <= self.skeleton_workers <= 16:
            raise ValueError("skeleton_workers must be in [1, 16]")
        if self.max_cache_chunks <= 0:
            raise ValueError("max_cache_chunks must be positive")


def villa_medial_surface(
    segmentation: np.ndarray,
    *,
    executor: Executor | None = None,
) -> np.ndarray:
    """Reproduce Villa's released medial-surface target for one 3-D label mask.

    Villa deliberately skeletonizes each native z slice in two dimensions,
    stacks those centerlines into a medial *surface*, closes the 3-D stack once,
    and intersects it with the original foreground.  A generic 3-D skeleton is
    not equivalent: it collapses a sheet to a curve or point.
    """

    mask = np.asarray(segmentation, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("fine medial surface requires a 3-D segmentation")
    skeleton = np.zeros(mask.shape, dtype=bool)
    nonempty = [index for index in range(mask.shape[0]) if bool(mask[index].any())]
    if executor is None:
        for index in nonempty:
            skeleton[index] = skeletonize(mask[index])
    else:
        values = executor.map(skeletonize, (mask[index] for index in nonempty))
        for index, value in zip(nonempty, values, strict=True):
            skeleton[index] = value
    # This ordering and default footprint are the exact pinned Villa transform.
    skeleton = closing(skeleton)
    return np.asarray(skeleton & mask, dtype=bool)


def villa_slicewise_center_radius(
    segmentation: np.ndarray,
    *,
    sampling_yx: tuple[float, float] = (1.0, 1.0),
    executor: Executor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair Villa's slice-wise medial centers with inscribed-circle radii.

    The radius and center definitions deliberately use the same two-dimensional
    geometry.  Pairing Villa's slice-wise centers with a three-dimensional EDT
    would mix circles and spheres and would not be a valid medial transform.
    Radii are measured in the physical units supplied by ``sampling_yx`` and
    are zero away from medial centers.
    """

    mask = np.asarray(segmentation, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("fine center-radius transform requires a 3-D mask")
    spacing = tuple(float(value) for value in sampling_yx)
    if len(spacing) != 2 or any(
        not np.isfinite(value) or value <= 0.0 for value in spacing
    ):
        raise ValueError("sampling_yx must contain two finite positive values")

    centers = villa_medial_surface(mask, executor=executor)
    radii = np.zeros(mask.shape, dtype=np.float32)
    for index in np.flatnonzero(centers.reshape(centers.shape[0], -1).any(axis=1)):
        distance = ndimage.distance_transform_edt(
            mask[int(index)], sampling=spacing
        )
        section_centers = centers[int(index)]
        radii[int(index), section_centers] = distance[section_centers].astype(
            np.float32, copy=False
        )
    if bool((radii[centers] <= 0.0).any()):
        raise RuntimeError("medial centers must have positive inscribed radii")
    return centers, radii


def reconstruct_slicewise_center_radius(
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    sampling_yx: tuple[float, float] = (1.0, 1.0),
    radius_margin: float = 0.0,
) -> np.ndarray:
    """Reconstruct a binary volume as a union of slice-wise open disks.

    EDT radii reach the nearest *background voxel center*.  The corresponding
    discrete inscribed disk is therefore open: using ``distance <= radius``
    would include that known background center and systematically add girth.
    ``radius_margin`` is explicit so any later antialias allowance cannot be
    hidden in the transform contract.
    """

    center_mask = np.asarray(centers, dtype=bool)
    radius_values = np.asarray(radii, dtype=np.float32)
    if center_mask.ndim != 3 or radius_values.shape != center_mask.shape:
        raise ValueError("centers and radii must be matching 3-D arrays")
    spacing = tuple(float(value) for value in sampling_yx)
    if len(spacing) != 2 or any(
        not np.isfinite(value) or value <= 0.0 for value in spacing
    ):
        raise ValueError("sampling_yx must contain two finite positive values")
    margin = float(radius_margin)
    if not np.isfinite(margin):
        raise ValueError("radius_margin must be finite")
    if bool((radius_values < 0.0).any()) or bool(
        (radius_values[~center_mask] != 0.0).any()
    ):
        raise ValueError("radii must be non-negative and zero away from centers")

    output = np.zeros(center_mask.shape, dtype=bool)
    sy, sx = spacing
    height, width = center_mask.shape[1:]
    for z_index in np.flatnonzero(
        center_mask.reshape(center_mask.shape[0], -1).any(axis=1)
    ):
        section = output[int(z_index)]
        for center_y, center_x in np.argwhere(center_mask[int(z_index)]):
            stored_radius = radius_values[int(z_index), center_y, center_x]
            radius = float(stored_radius) + margin
            if radius <= 0.0:
                continue
            if margin == 0.0:
                # A float32 representation of sqrt(n) can round upward.  Move
                # one stored-precision ULP inward so the nominally open disk
                # cannot accidentally admit its nearest background center.
                radius = float(
                    np.nextafter(stored_radius, np.float32(0.0), dtype=np.float32)
                )
            y_extent = int(np.ceil(radius / sy))
            x_extent = int(np.ceil(radius / sx))
            y0 = max(0, int(center_y) - y_extent)
            y1 = min(height, int(center_y) + y_extent + 1)
            x0 = max(0, int(center_x) - x_extent)
            x1 = min(width, int(center_x) + x_extent + 1)
            yy = (np.arange(y0, y1, dtype=np.float32) - center_y) * sy
            xx = (np.arange(x0, x1, dtype=np.float32) - center_x) * sx
            # Strict inequality is intentional; see the docstring above.
            disk = yy[:, None] ** 2 + xx[None, :] ** 2 < radius**2
            section[y0:y1, x0:x1] |= disk
    return output


def center_radius_envelope_2d(
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    sampling_yx: tuple[float, float] = (1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return the additively weighted MAT envelope and winning local radius.

    For every pixel ``x`` the envelope is ``max_c(r(c) - distance(x, c))``.
    Its sign separates the union of medial disks from radial spill.  Selecting
    the maximizing center, rather than merely the nearest center, is essential
    when neighboring medial primitives have different radii.
    """

    center_mask = np.asarray(centers, dtype=bool)
    radius_values = np.asarray(radii, dtype=np.float32)
    if center_mask.ndim != 2 or radius_values.shape != center_mask.shape:
        raise ValueError("centers and radii must be matching 2-D arrays")
    spacing = tuple(float(value) for value in sampling_yx)
    if len(spacing) != 2 or any(
        not np.isfinite(value) or value <= 0.0 for value in spacing
    ):
        raise ValueError("sampling_yx must contain two finite positive values")
    if bool((radius_values < 0.0).any()) or bool(
        (radius_values[~center_mask] != 0.0).any()
    ):
        raise ValueError("radii must be non-negative and zero away from centers")

    envelope = np.full(center_mask.shape, -np.inf, dtype=np.float32)
    winning_radius = np.zeros(center_mask.shape, dtype=np.float32)
    yy, xx = np.indices(center_mask.shape, dtype=np.float32)
    sy, sx = spacing
    for center_y, center_x in np.argwhere(center_mask):
        radius = float(radius_values[center_y, center_x])
        distance = np.sqrt(
            ((yy - center_y) * sy) ** 2 + ((xx - center_x) * sx) ** 2
        )
        candidate = radius - distance
        selected = candidate > envelope
        envelope[selected] = candidate[selected]
        winning_radius[selected] = radius
    return envelope, winning_radius


def _required_support_at_coarse_centers(
    *,
    support: ChunkSupport,
    coarse_to_fine: np.ndarray,
    origin_zyx: tuple[int, int, int],
    shape_zyx: tuple[int, int, int],
    halo_zyx: tuple[int, int, int],
    block_depth: int = 16,
) -> np.ndarray:
    """Return centers whose complete anisotropic medial halo is known."""

    valid = np.zeros(shape_zyx, dtype=np.uint8)
    fine_shape = np.asarray(support.shape_zyx, dtype=np.int64)
    chunks = np.asarray(support.chunks_zyx, dtype=np.int64)
    halo = np.asarray(halo_zyx, dtype=np.int64)
    for local_z0 in range(0, shape_zyx[0], block_depth):
        local_z1 = min(shape_zyx[0], local_z0 + block_depth)
        zz, yy, xx = np.meshgrid(
            np.arange(origin_zyx[0] + local_z0, origin_zyx[0] + local_z1),
            np.arange(origin_zyx[1], origin_zyx[1] + shape_zyx[1]),
            np.arange(origin_zyx[2], origin_zyx[2] + shape_zyx[2]),
            indexing="ij",
        )
        coarse_zyx = np.stack((zz, yy, xx), axis=-1).reshape(-1, 3)
        fine_xyz = transform_xyz(coarse_zyx[:, ::-1], coarse_to_fine)
        centers = np.floor(fine_xyz[:, ::-1] + 0.5).astype(np.int64)
        lower = centers - halo
        upper = centers + halo
        selected = ((lower >= 0) & (upper < fine_shape)).all(axis=1)
        if support.present_ids is not None and bool(selected.any()):
            lower_chunks = np.floor_divide(lower, chunks)
            upper_chunks = np.floor_divide(upper, chunks)
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
        valid[local_z0:local_z1] = selected.reshape(
            local_z1 - local_z0,
            shape_zyx[1],
            shape_zyx[2],
        )
    return valid


class FineMedialSurfaceReader:
    """Halo-correct, sparse-safe LRU of exact Villa medial target chunks."""

    def __init__(
        self,
        field_reader: FineFieldWindowReader,
        *,
        options: MedialProjectionOptions | None = None,
    ) -> None:
        options = options or MedialProjectionOptions()
        options.validate()
        self.field_reader = field_reader
        self.support = field_reader.support
        if any(
            2 * halo > chunk
            for halo, chunk in zip(
                options.halo_zyx, self.support.chunks_zyx, strict=True
            )
        ):
            raise ValueError(
                "medial halo may span at most two support chunks per axis"
            )
        self.options = options
        self._cache: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()
        self._executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=options.skeleton_workers)
            if options.skeleton_workers > 1
            else None
        )
        self.chunk_computations = 0
        self.cache_hits = 0

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _binary_window(
        self,
        origin_zyx: tuple[int, int, int],
        shape_zyx: tuple[int, int, int],
    ) -> np.ndarray:
        field = self.field_reader.field
        if field.encoding == "labels":
            return self.field_reader.read_compact_probability(
                origin_zyx, shape_zyx
            ) > 0
        probability = self.field_reader.read_probability(origin_zyx, shape_zyx)
        return probability >= field.threshold

    def chunk(self, coordinate_zyx: tuple[int, int, int]) -> np.ndarray:
        if not self.support.contains(coordinate_zyx):
            raise ValueError("requested medial chunk is outside declared support")
        cached = self._cache.get(coordinate_zyx)
        if cached is not None:
            self._cache.move_to_end(coordinate_zyx)
            self.cache_hits += 1
            return cached

        chunks = np.asarray(self.support.chunks_zyx, dtype=np.int64)
        core_origin = np.asarray(coordinate_zyx, dtype=np.int64) * chunks
        core_end = np.minimum(
            core_origin + chunks,
            np.asarray(self.support.shape_zyx, dtype=np.int64),
        )
        core_shape = core_end - core_origin
        halo = np.asarray(self.options.halo_zyx, dtype=np.int64)
        window_origin = core_origin - halo
        window_shape = core_shape + 2 * halo
        mask = self._binary_window(
            tuple(int(value) for value in window_origin),
            tuple(int(value) for value in window_shape),
        )
        medial = villa_medial_surface(mask, executor=self._executor)
        crop = tuple(
            slice(int(offset), int(offset + size))
            for offset, size in zip(halo, core_shape, strict=True)
        )
        result = np.ascontiguousarray(medial[crop], dtype=bool)
        self.chunk_computations += 1
        self._cache[coordinate_zyx] = result
        self._cache.move_to_end(coordinate_zyx)
        while len(self._cache) > self.options.max_cache_chunks:
            self._cache.popitem(last=False)
        return result


def project_fine_medial_patch(
    reader: FineMedialSurfaceReader,
    fine_to_coarse_affine_xyz: tuple[tuple[float, float, float, float], ...],
    origin_zyx: tuple[int, int, int],
    shape_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float | str | list[int]]]:
    """OR/max-project an exact fine medial sheet into a coarse patch."""

    if any(size <= 0 for size in shape_zyx):
        raise ValueError("coarse medial patch shape must be positive")
    support = reader.support
    fine_lower, fine_upper = coarse_patch_fine_bounds(
        origin_zyx, shape_zyx, fine_to_coarse_affine_xyz
    )
    chunks = np.asarray(support.chunks_zyx, dtype=np.int64)
    chunk_lower = np.floor_divide(fine_lower, chunks)
    chunk_upper = np.floor_divide(fine_upper + chunks - 1, chunks)
    fine_to_coarse = affine_matrix(fine_to_coarse_affine_xyz)
    coarse_origin_xyz = np.asarray(origin_zyx[::-1], dtype=np.float64)
    projected = np.zeros(shape_zyx, dtype=bool)
    chunks_visited = 0
    fine_crest_voxels = 0
    for coordinate in support.iter_between(tuple(chunk_lower), tuple(chunk_upper)):
        fine_chunk = reader.chunk(coordinate)
        positive_local = np.argwhere(fine_chunk)
        chunks_visited += 1
        if not positive_local.size:
            continue
        fine_crest_voxels += int(positive_local.shape[0])
        chunk_origin = np.asarray(coordinate, dtype=np.int64) * chunks
        fine_zyx = positive_local.astype(np.float64) + chunk_origin
        coarse_xyz = transform_xyz(fine_zyx[:, ::-1], fine_to_coarse)
        local_xyz = np.floor(coarse_xyz - coarse_origin_xyz + 0.5).astype(np.int64)
        local_zyx = local_xyz[:, ::-1]
        inside = ((local_zyx >= 0) & (local_zyx < np.asarray(shape_zyx))).all(axis=1)
        selected = local_zyx[inside]
        if selected.size:
            projected[selected[:, 0], selected[:, 1], selected[:, 2]] = True

    crest_valid = _required_support_at_coarse_centers(
        support=support,
        coarse_to_fine=invert_affine(fine_to_coarse_affine_xyz),
        origin_zyx=origin_zyx,
        shape_zyx=shape_zyx,
        halo_zyx=reader.options.halo_zyx,
    )
    projected &= crest_valid > 0
    crest_u8 = projected.astype(np.uint8, copy=False)
    known = int(np.count_nonzero(crest_valid))
    positive = int(np.count_nonzero(crest_u8))
    return (
        crest_u8,
        crest_valid.astype(np.uint8, copy=False),
        {
            "medial_surface_contract": VILLA_MEDIAL_SURFACE_CONTRACT,
            "villa_source_commit": VILLA_MEDIAL_SURFACE_SOURCE_COMMIT,
            "villa_source_sha256": VILLA_MEDIAL_SURFACE_SOURCE_SHA256,
            "projection_contract": MEDIAL_MAX_PROJECTION_CONTRACT,
            "skimage_version": skimage_version,
            "halo_zyx": list(reader.options.halo_zyx),
            "chunks_visited": chunks_visited,
            "fine_crest_voxels": fine_crest_voxels,
            "known_voxels": known,
            "crest_voxels": positive,
            "known_fraction": known / crest_u8.size,
            "crest_fraction_known": positive / max(1, known),
            "reader_chunk_computations": reader.chunk_computations,
            "reader_cache_hits": reader.cache_hits,
        },
    )


def medial_provenance(options: MedialProjectionOptions) -> dict[str, Any]:
    options.validate()
    return {
        "medial_surface_contract": VILLA_MEDIAL_SURFACE_CONTRACT,
        "villa_source_commit": VILLA_MEDIAL_SURFACE_SOURCE_COMMIT,
        "villa_source_sha256": VILLA_MEDIAL_SURFACE_SOURCE_SHA256,
        "projection_contract": MEDIAL_MAX_PROJECTION_CONTRACT,
        "skimage_version": skimage_version,
        "halo_zyx": list(options.halo_zyx),
        "skeleton_workers": options.skeleton_workers,
        "max_cache_chunks": options.max_cache_chunks,
    }
