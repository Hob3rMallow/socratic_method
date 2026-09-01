from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .tifxyz import TifxyzMap, resample_parameter_grid

LABEL_BACKGROUND = 0
LABEL_SURFACE = 1
LABEL_IGNORE = 2


class RasterizeError(ValueError):
    pass


@dataclass(frozen=True)
class RasterizeOptions:
    """Physical-band label construction at a given pitch.

    ``surface_radius_vox`` (r1) and ``background_radius_vox`` (r2) express
    the ~25-30 um m7 label convention at the target pitch: voxels within r1
    of a traced surface are label 1, the r1..r2 shell is label 0, everything
    else is label 2 (ignore). Untraced neighbor wraps therefore land in
    ignore by construction, because r2 caps how far "background" is asserted
    from the traced sheet.
    """

    surface_radius_vox: float
    background_radius_vox: float
    point_spacing_vox: float = 0.6

    def validate(self) -> None:
        if not self.surface_radius_vox > 0.0:
            raise RasterizeError("surface_radius_vox must be positive")
        if not self.background_radius_vox > self.surface_radius_vox:
            raise RasterizeError("background_radius_vox must exceed surface_radius_vox")
        if not 0.0 < self.point_spacing_vox <= 1.0:
            raise RasterizeError("point_spacing_vox must be in (0, 1]")

    @property
    def padding_vox(self) -> int:
        return int(np.ceil(self.background_radius_vox)) + 1


def default_options_for_pitch(
    *, pitch_um: float, band_radius_um: float = 14.0
) -> RasterizeOptions:
    """The m7 physical band expressed at an arbitrary pitch.

    ``band_radius_um`` = 14 um gives a ~28 um full band: 1.5 vox at 9.362 um,
    ~1.77 at 7.91 um, ~5.8 at 2.399 um, ~12.4 at 1.129 um -- matching the
    measured TRAIN_LABELS thickness at coarse pitch. The background shell is
    always 2x the surface radius.
    """

    if not pitch_um > 0.0:
        raise RasterizeError("pitch_um must be positive")
    radius = band_radius_um / pitch_um
    return RasterizeOptions(
        surface_radius_vox=radius,
        background_radius_vox=2.0 * radius,
    )


def collect_surface_points_zyx(
    maps: list[TifxyzMap],
    *,
    bbox_lo_zyx: tuple[float, float, float],
    bbox_hi_zyx: tuple[float, float, float],
    point_spacing_vox: float = 0.6,
    margin_vox: float = 0.0,
    max_tile_points: int = 4_000_000,
) -> np.ndarray:
    """Sample every traced surface densely enough for gap-free rasterization.

    TIFXYZ parameter grids are sparse (typically ~20 voxels between cells),
    so each map is upsampled tile-by-tile until adjacent samples are at most
    ``point_spacing_vox`` apart. The tile size in parameter cells adapts to
    the upsample factor so no upsampled tile exceeds ``max_tile_points`` --
    at 30x+ upsampling a fixed cell tile would allocate hundreds of MB per
    tile. Returns an (N, 3) float32 z-y-x array restricted to the bounding
    box expanded by ``margin_vox``.
    """

    lo = np.asarray(bbox_lo_zyx, dtype=np.float32) - margin_vox
    hi = np.asarray(bbox_hi_zyx, dtype=np.float32) + margin_vox
    collected: list[np.ndarray] = []
    for mapping in maps:
        if not mapping.valid.any():
            continue
        spacing = mapping.median_grid_spacing_vox()
        upsample = max(1, int(np.ceil(spacing / point_spacing_vox)))
        tile_cells = max(2, int(np.sqrt(max_tile_points) / upsample))
        height, width = mapping.shape
        # Quick reject: does any valid cell fall near the bbox at all?
        coarse_xyz = mapping.xyz[mapping.valid]
        coarse_zyx = coarse_xyz[:, ::-1]
        near = np.logical_and(
            coarse_zyx >= lo - spacing, coarse_zyx <= hi + spacing
        ).all(axis=1)
        if not near.any():
            continue
        for row_start in range(0, max(1, height - 1), tile_cells):
            row_end = min(height, row_start + tile_cells + 1)
            if row_end - row_start < 2:
                row_start = max(0, row_end - 2)
            for column_start in range(0, max(1, width - 1), tile_cells):
                column_end = min(width, column_start + tile_cells + 1)
                if column_end - column_start < 2:
                    column_start = max(0, column_end - 2)
                tile = TifxyzMap(
                    xyz=mapping.xyz[row_start:row_end, column_start:column_end],
                    valid=mapping.valid[row_start:row_end, column_start:column_end],
                    source=mapping.source,
                )
                if not tile.valid.any():
                    continue
                tile_zyx = tile.xyz[tile.valid][:, ::-1]
                if not (
                    np.logical_and(
                        tile_zyx >= lo - spacing, tile_zyx <= hi + spacing
                    )
                    .all(axis=1)
                    .any()
                ):
                    continue
                dense = resample_parameter_grid(
                    tile,
                    (
                        (tile.shape[0] - 1) * upsample + 1,
                        (tile.shape[1] - 1) * upsample + 1,
                    ),
                )
                points_zyx = dense.xyz[dense.valid][:, ::-1]
                keep = np.logical_and(points_zyx >= lo, points_zyx <= hi).all(
                    axis=1
                )
                if keep.any():
                    collected.append(points_zyx[keep].astype(np.float32))
    if not collected:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(collected, axis=0)


def rasterize_label_block(
    points_zyx: np.ndarray,
    *,
    origin_zyx: tuple[int, int, int],
    shape_zyx: tuple[int, int, int],
    options: RasterizeOptions,
    veto: np.ndarray | None = None,
) -> np.ndarray:
    """Rasterize {0,1,2} labels for one block from dense surface points.

    ``points_zyx`` must cover the block plus ``options.padding_vox`` margin
    so distances near the block boundary are correct. ``veto`` (optional,
    block-shaped boolean) flips would-be background voxels to ignore -- the
    hook for the villa near-band veto on untraced wraps.
    """

    options.validate()
    if any(size <= 0 for size in shape_zyx):
        raise RasterizeError(f"block shape must be positive, got {shape_zyx}")
    if veto is not None and tuple(veto.shape) != tuple(shape_zyx):
        raise RasterizeError(
            f"veto shape {veto.shape} does not match block {shape_zyx}"
        )
    pad = options.padding_vox
    padded_shape = tuple(size + 2 * pad for size in shape_zyx)
    label = np.full(shape_zyx, LABEL_IGNORE, dtype=np.uint8)

    points = np.asarray(points_zyx, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        if points.size == 0:
            return label
        raise RasterizeError(f"points must have shape (N, 3), got {points.shape}")
    local = points - (np.asarray(origin_zyx, dtype=np.float32) - pad)
    indices = np.rint(local).astype(np.int64)
    inside = np.logical_and(
        indices >= 0, indices < np.asarray(padded_shape)
    ).all(axis=1)
    if not inside.any():
        return label
    indices = indices[inside]

    seeds = np.zeros(padded_shape, dtype=bool)
    seeds[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    distance = ndimage.distance_transform_edt(~seeds)
    core = tuple(slice(pad, pad + size) for size in shape_zyx)
    block_distance = distance[core]

    label[block_distance <= options.surface_radius_vox] = LABEL_SURFACE
    shell = (block_distance > options.surface_radius_vox) & (
        block_distance <= options.background_radius_vox
    )
    label[shell] = LABEL_BACKGROUND
    if veto is not None:
        label[(label == LABEL_BACKGROUND) & veto.astype(bool)] = LABEL_IGNORE
    return label


def soft_band_block(
    points_zyx: np.ndarray,
    *,
    origin_zyx: tuple[int, int, int],
    shape_zyx: tuple[int, int, int],
    options: RasterizeOptions,
) -> np.ndarray:
    """A [0,1] float band field: 1 inside the surface tube, 0 outside.

    Used to push rasterized fine ground truth through the same bridge
    operator the teacher predictions use, so validation targets share the
    deployment operator exactly.
    """

    label = rasterize_label_block(
        points_zyx, origin_zyx=origin_zyx, shape_zyx=shape_zyx, options=options
    )
    return (label == LABEL_SURFACE).astype(np.float32)


def band_thickness_stats(label: np.ndarray) -> dict[str, float]:
    """Measure surface-band thickness (2x interior EDT of the band).

    The guard for the resampling thickness check and the rasterizer
    calibration gate G-0a.
    """

    band = label == LABEL_SURFACE
    if not band.any():
        return {"p50": 0.0, "p90": 0.0, "fraction": 0.0}
    interior = ndimage.distance_transform_edt(band)
    positive = interior[band]
    return {
        "p50": float(np.percentile(positive, 50) * 2.0),
        "p90": float(np.percentile(positive, 90) * 2.0),
        "fraction": float(band.mean()),
    }
