from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


class ResampleError(ValueError):
    pass


FineReader = Callable[[tuple[int, int, int], tuple[int, int, int]], np.ndarray]


@dataclass(frozen=True)
class BridgeOptions:
    """Anti-aliased pull-back of a fine probability field into the coarse
    lattice through the official fine->coarse affine.

    Thickness is normalized at the *label* level (the physical ~28 um band),
    so no morphological thickening happens here by default; the
    ``maxpool_prefilter`` escape hatch exists for the measured-thickness
    guard, never as an assumption (the recorded majority-downsampling trap).
    """

    prefilter_sigma_scale: float = 0.5
    coverage_erosion_fine_vox: int = 32
    max_fine_window_vox: int = 352
    maxpool_prefilter: bool = False
    # Erode coverage by the filter support before marking validity. Right
    # for carved data (partial context at mirror boundaries); wrong for
    # exact rasterized ground-truth fields, where eroding a thin band+shell
    # slab eats the shell negatives and drives target prevalence to ~1.
    erode_filter_margin: bool = True

    def validate(self) -> None:
        if self.prefilter_sigma_scale < 0.0:
            raise ResampleError("prefilter_sigma_scale must be non-negative")
        if self.coverage_erosion_fine_vox < 0:
            raise ResampleError("coverage_erosion_fine_vox must be non-negative")
        if self.max_fine_window_vox < 64:
            raise ResampleError("max_fine_window_vox must be >= 64")


def invert_affine_xyz(
    affine_xyz: tuple[tuple[float, ...], ...] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert a 3x4 fine->coarse x-y-z affine.

    Returns ``(linear_inverse, translation)`` such that
    ``fine_xyz = linear_inverse @ (coarse_xyz - translation)``.
    """

    matrix = np.asarray(affine_xyz, dtype=np.float64)
    if matrix.shape != (3, 4):
        raise ResampleError(f"affine must be 3x4, got {matrix.shape}")
    linear = matrix[:, :3]
    determinant = np.linalg.det(linear)
    if not np.isfinite(determinant) or abs(determinant) < 1.0e-12:
        raise ResampleError("affine linear part is singular")
    return np.linalg.inv(linear), matrix[:, 3].copy()


def affine_scale_ratio(
    affine_xyz: tuple[tuple[float, ...], ...] | np.ndarray,
) -> float:
    """The isotropic fine->coarse scale implied by the affine (>1 means the
    coarse lattice is coarser, i.e. |det|^(1/3) of the inverse)."""

    matrix = np.asarray(affine_xyz, dtype=np.float64)
    if matrix.shape != (3, 4):
        raise ResampleError(f"affine must be 3x4, got {matrix.shape}")
    determinant = abs(np.linalg.det(matrix[:, :3]))
    if determinant < 1.0e-12:
        raise ResampleError("affine linear part is singular")
    return float(determinant ** (-1.0 / 3.0))


def fine_bbox_for_coarse_box(
    origin_zyx: tuple[int, int, int],
    shape_zyx: tuple[int, int, int],
    affine_xyz: tuple[tuple[float, ...], ...] | np.ndarray,
    *,
    margin_fine_vox: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fine-frame z-y-x bounding box of a coarse box (corner image + margin)."""

    linear_inverse, translation = invert_affine_xyz(affine_xyz)
    corners = []
    for dz in (0, shape_zyx[0]):
        for dy in (0, shape_zyx[1]):
            for dx in (0, shape_zyx[2]):
                coarse_xyz = np.array(
                    [
                        origin_zyx[2] + dx,
                        origin_zyx[1] + dy,
                        origin_zyx[0] + dz,
                    ],
                    dtype=np.float64,
                )
                corners.append(linear_inverse @ (coarse_xyz - translation))
    corner_array = np.stack(corners, axis=0)
    lo_xyz = corner_array.min(axis=0) - margin_fine_vox
    hi_xyz = corner_array.max(axis=0) + margin_fine_vox
    return lo_xyz[::-1].copy(), hi_xyz[::-1].copy()


def _prefilter(
    window: np.ndarray, sigma: float, scale_ratio: float, maxpool: bool
) -> np.ndarray:
    value = window.astype(np.float32, copy=False)
    if maxpool:
        size = max(1, int(round(scale_ratio)))
        if size > 1:
            value = ndimage.maximum_filter(value, size=size, mode="nearest")
    if sigma > 0.0:
        value = ndimage.gaussian_filter(value, sigma=sigma, mode="nearest")
    return value


def resample_to_coarse(
    read_fine_prob: FineReader,
    read_fine_coverage: FineReader | None,
    *,
    coarse_origin_zyx: tuple[int, int, int],
    coarse_shape_zyx: tuple[int, int, int],
    fine_to_coarse_affine_xyz: tuple[tuple[float, ...], ...] | np.ndarray,
    options: BridgeOptions | None = None,
    coarse_shift_zyx: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Pull a fine probability field back onto a coarse block.

    ``read_fine_prob(origin_zyx, shape_zyx)`` returns a float window in
    [0, 1] (zero-filled outside data); ``read_fine_coverage`` returns a
    {0,1} window marking voxels whose fine data actually exists (carved and
    predicted). Coverage is eroded by the filter support plus the boundary
    margin before sampling, so no coarse voxel is marked valid off partial
    context. Returns ``(q float32, valid uint8)`` for the block.
    ``coarse_shift_zyx`` applies a registration correction in the coarse
    frame before the pull-back.
    """

    options = options if options is not None else BridgeOptions()
    options.validate()
    scale_ratio = affine_scale_ratio(fine_to_coarse_affine_xyz)
    sigma = options.prefilter_sigma_scale * scale_ratio
    filter_margin = int(np.ceil(3.0 * sigma)) + 2
    erosion = (
        filter_margin if options.erode_filter_margin else 0
    ) + options.coverage_erosion_fine_vox
    linear_inverse, translation = invert_affine_xyz(fine_to_coarse_affine_xyz)

    q = np.zeros(coarse_shape_zyx, dtype=np.float32)
    valid = np.zeros(coarse_shape_zyx, dtype=np.uint8)
    usable = options.max_fine_window_vox - 2 * (filter_margin + erosion)
    if usable < 32:
        raise ResampleError(
            "max_fine_window_vox is too small for the filter and erosion margins"
        )
    step = max(16, int(usable / max(scale_ratio, 1.0e-6)) // 16 * 16)

    shift = np.asarray(coarse_shift_zyx, dtype=np.float64)
    for z0 in range(0, coarse_shape_zyx[0], step):
        for y0 in range(0, coarse_shape_zyx[1], step):
            for x0 in range(0, coarse_shape_zyx[2], step):
                sub_shape = (
                    min(step, coarse_shape_zyx[0] - z0),
                    min(step, coarse_shape_zyx[1] - y0),
                    min(step, coarse_shape_zyx[2] - x0),
                )
                sub_origin = (
                    coarse_origin_zyx[0] + z0,
                    coarse_origin_zyx[1] + y0,
                    coarse_origin_zyx[2] + x0,
                )
                grid_z, grid_y, grid_x = np.meshgrid(
                    np.arange(sub_shape[0], dtype=np.float64)
                    + sub_origin[0]
                    + shift[0],
                    np.arange(sub_shape[1], dtype=np.float64)
                    + sub_origin[1]
                    + shift[1],
                    np.arange(sub_shape[2], dtype=np.float64)
                    + sub_origin[2]
                    + shift[2],
                    indexing="ij",
                )
                coarse_xyz = np.stack(
                    (grid_x.ravel(), grid_y.ravel(), grid_z.ravel()), axis=0
                )
                fine_xyz = linear_inverse @ (coarse_xyz - translation[:, None])
                fine_zyx = fine_xyz[::-1]
                lo = np.floor(fine_zyx.min(axis=1)).astype(np.int64) - filter_margin
                hi = np.ceil(fine_zyx.max(axis=1)).astype(np.int64) + filter_margin + 1
                window_origin = tuple(int(item) for item in lo)
                window_shape = tuple(int(item) for item in (hi - lo))
                if any(size <= 0 for size in window_shape):
                    continue
                window = np.asarray(
                    read_fine_prob(window_origin, window_shape), dtype=np.float32
                )
                if tuple(window.shape) != window_shape:
                    raise ResampleError(
                        f"fine reader returned {window.shape}, "
                        f"expected {window_shape}"
                    )
                filtered = _prefilter(
                    window, sigma, scale_ratio, options.maxpool_prefilter
                )
                local = fine_zyx - lo[:, None]
                sampled = ndimage.map_coordinates(
                    filtered, local, order=1, mode="constant", cval=0.0
                )
                block_slice = (
                    slice(z0, z0 + sub_shape[0]),
                    slice(y0, y0 + sub_shape[1]),
                    slice(x0, x0 + sub_shape[2]),
                )
                q[block_slice] = np.clip(
                    sampled.reshape(sub_shape), 0.0, 1.0
                ).astype(np.float32)

                if read_fine_coverage is None:
                    valid[block_slice] = 1
                    continue
                coverage_origin = tuple(int(item) - erosion for item in lo)
                coverage_shape = tuple(
                    int(item) + 2 * erosion for item in (hi - lo)
                )
                coverage = (
                    np.asarray(
                        read_fine_coverage(coverage_origin, coverage_shape)
                    )
                    > 0.5
                )
                if not coverage.any():
                    continue
                if erosion == 0 or coverage.all():
                    eroded = coverage
                else:
                    eroded = (
                        ndimage.distance_transform_edt(coverage) > erosion
                    )
                local_cov = fine_zyx - (lo[:, None] - erosion)
                sampled_valid = ndimage.map_coordinates(
                    eroded.astype(np.float32),
                    local_cov,
                    order=0,
                    mode="constant",
                    cval=0.0,
                )
                valid[block_slice] = (
                    sampled_valid.reshape(sub_shape) > 0.5
                ).astype(np.uint8)
    return q, valid


def phase_correlation_shift(
    moving: np.ndarray, reference: np.ndarray
) -> tuple[tuple[float, float, float], float]:
    """Integer-voxel phase-correlation shift aligning ``moving`` to
    ``reference`` (``np.roll(moving, shift)`` best matches ``reference``),
    plus the normalized correlation peak strength in [0, 1]-ish units.
    Used by the registration-residual audit; both inputs must share a shape.
    """

    if moving.shape != reference.shape:
        raise ResampleError(
            f"phase correlation requires equal shapes, got "
            f"{moving.shape} vs {reference.shape}"
        )
    first = np.fft.rfftn(np.asarray(moving, dtype=np.float64))
    second = np.fft.rfftn(np.asarray(reference, dtype=np.float64))
    spectrum = second * np.conj(first)
    magnitude = np.abs(spectrum)
    magnitude[magnitude < 1.0e-12] = 1.0e-12
    correlation = np.fft.irfftn(
        spectrum / magnitude, s=moving.shape, axes=(0, 1, 2)
    )
    peak_index = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    peak = float(correlation[peak_index])
    shift = []
    for index, size in zip(peak_index, moving.shape):
        shift.append(float(index if index <= size // 2 else index - size))
    return (shift[0], shift[1], shift[2]), peak


def registration_action(
    shift_zyx: tuple[float, float, float],
    *,
    accept_vox: float = 0.75,
    correct_vox: float = 2.0,
) -> str:
    """Classify a measured residual: 'accept', 'correct', or 'reject'."""

    magnitude = float(np.linalg.norm(np.asarray(shift_zyx, dtype=np.float64)))
    if magnitude <= accept_vox:
        return "accept"
    if magnitude <= correct_vox:
        return "correct"
    return "reject"


class ChunkCoverage:
    """Voxel coverage derived from a carved chunk-id set.

    A sparse local mirror holds only selected chunks; a chunk absent from the
    selection is *unknown*, not empty, so coverage must come from the carve
    manifest rather than from the array (an absent chunk reads as fill).
    Chunk ids are (cz, cy, cx) grid coordinates.
    """

    def __init__(
        self,
        chunk_shape_zyx: tuple[int, int, int],
        chunk_ids: set[tuple[int, int, int]],
    ) -> None:
        if any(size <= 0 for size in chunk_shape_zyx):
            raise ResampleError("chunk shape must be positive")
        self.chunk_shape = tuple(int(item) for item in chunk_shape_zyx)
        self.chunk_ids = chunk_ids

    def __call__(
        self, origin_zyx: tuple[int, int, int], shape_zyx: tuple[int, int, int]
    ) -> np.ndarray:
        coverage = np.zeros(shape_zyx, dtype=np.uint8)
        lo = [
            int(np.floor(origin_zyx[axis] / self.chunk_shape[axis]))
            for axis in range(3)
        ]
        hi = [
            int(
                np.ceil(
                    (origin_zyx[axis] + shape_zyx[axis]) / self.chunk_shape[axis]
                )
            )
            for axis in range(3)
        ]
        for cz in range(lo[0], hi[0]):
            for cy in range(lo[1], hi[1]):
                for cx in range(lo[2], hi[2]):
                    if (cz, cy, cx) not in self.chunk_ids:
                        continue
                    slices = []
                    inside = True
                    for axis, chunk_coordinate in enumerate((cz, cy, cx)):
                        start = (
                            chunk_coordinate * self.chunk_shape[axis]
                            - origin_zyx[axis]
                        )
                        stop = start + self.chunk_shape[axis]
                        start = max(0, start)
                        stop = min(shape_zyx[axis], stop)
                        if stop <= start:
                            inside = False
                            break
                        slices.append(slice(start, stop))
                    if inside:
                        coverage[tuple(slices)] = 1
        return coverage
