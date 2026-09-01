from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


class TifxyzError(ValueError):
    pass


def _find_component(directory: Path, stem: str, *, required: bool) -> Path | None:
    candidates: list[Path] = []
    for child in directory.iterdir():
        if (
            child.is_file()
            and child.suffix.lower() in {".tif", ".tiff"}
            and child.stem.lower() == stem.lower()
        ):
            candidates.append(child)
    if len(candidates) > 1:
        raise TifxyzError(
            f"{directory}: multiple files match TIFXYZ component {stem!r}"
        )
    if not candidates:
        if required:
            raise TifxyzError(f"{directory}: missing {stem}.tif")
        return None
    return candidates[0]


def _read_2d(path: Path) -> np.ndarray:
    try:
        array = tifffile.memmap(path)
    except (ValueError, OSError):
        array = tifffile.imread(path)
    array = np.asarray(array)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise TifxyzError(f"{path}: expected a 2-D image, got {array.shape}")
    return array


@dataclass(frozen=True)
class TifxyzMap:
    """A traced surface as a 2-D parameter grid of 3-D voxel coordinates.

    This is the only surface representation the rewrite keeps, and it is used
    purely offline: locating co-registered regions and rasterizing ground
    truth into voxel labels. It never reaches training or inference.
    """

    xyz: np.ndarray
    valid: np.ndarray
    source: Path

    @classmethod
    def load(cls, directory: str | Path) -> TifxyzMap:
        source = Path(directory).expanduser()
        if not source.is_dir():
            raise TifxyzError(f"{source}: TIFXYZ directory does not exist")
        components = [
            _read_2d(_find_component(source, axis, required=True))
            for axis in ("x", "y", "z")
        ]
        shapes = {tuple(component.shape) for component in components}
        if len(shapes) != 1:
            raise TifxyzError(f"{source}: x/y/z shapes do not match: {sorted(shapes)}")
        xyz = np.stack(components, axis=-1)
        mask_path = None
        for stem in ("mask", "valid"):
            mask_path = _find_component(source, stem, required=False)
            if mask_path is not None:
                break
        valid = np.isfinite(xyz).all(axis=-1)
        # Voxel coordinates are never negative; public exports reserve
        # (-1,-1,-1) for no-surface cells, and the geometric pipeline's
        # degenerate-frame filter used to hide them. Reject them here so the
        # rasterizer never splats a sentinel pile at the volume corner.
        valid &= (xyz >= 0.0).all(axis=-1)
        if mask_path is not None:
            mask = _read_2d(mask_path)
            if mask.shape != xyz.shape[:2]:
                raise TifxyzError(
                    f"{mask_path}: mask shape {mask.shape} does not match "
                    f"coordinates {xyz.shape[:2]}"
                )
            valid &= mask != 0
        else:
            # Exporters also commonly reserve an all-zero triplet. A real
            # boundary coordinate may contain one zero, so test all 3.
            valid &= np.any(xyz != 0, axis=-1)
        return cls(
            xyz=xyz.astype(np.float32, copy=False),
            valid=valid,
            source=source.resolve(),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(item) for item in self.valid.shape)

    def valid_points_xyz(self) -> np.ndarray:
        """All valid surface coordinates as an (N, 3) x-y-z array."""

        return self.xyz[self.valid]

    def median_grid_spacing_vox(self) -> float:
        """Median voxel distance between adjacent valid parameter cells.

        Public TIFXYZ exports rasterize the parameter domain sparsely
        (meta.json ``scale`` around 0.05, i.e. ~20 voxels between cells), so
        rasterization must upsample the grid until adjacent samples are
        sub-voxel. Measuring the spacing from the map itself is robust to
        derived or unusually scaled exports.
        """

        spacings: list[np.ndarray] = []
        along_v = np.linalg.norm(self.xyz[1:] - self.xyz[:-1], axis=-1)
        valid_v = self.valid[1:] & self.valid[:-1]
        if valid_v.any():
            spacings.append(along_v[valid_v])
        along_u = np.linalg.norm(self.xyz[:, 1:] - self.xyz[:, :-1], axis=-1)
        valid_u = self.valid[:, 1:] & self.valid[:, :-1]
        if valid_u.any():
            spacings.append(along_u[valid_u])
        if not spacings:
            raise TifxyzError(f"{self.source}: no adjacent valid grid cells")
        return float(np.median(np.concatenate(spacings)))


def load_meta(directory: str | Path) -> dict[str, Any]:
    """Read a segment's meta.json (scale, bbox) if present; {} otherwise."""

    path = Path(directory).expanduser() / "meta.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TifxyzError(f"{path}: invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise TifxyzError(f"{path}: expected a JSON object")
    return value


def resample_parameter_grid(
    mapping: TifxyzMap,
    shape: tuple[int, int],
) -> TifxyzMap:
    """Bilinearly sample a TIFXYZ map on another normalized parameter grid."""

    if mapping.shape == shape:
        return mapping
    target_height, target_width = shape
    source_height, source_width = mapping.shape
    if min(target_height, target_width, source_height, source_width) < 2:
        raise TifxyzError(
            f"{mapping.source}: cannot resample parameter grid "
            f"{mapping.shape} to {shape}"
        )

    row_coordinates = np.linspace(
        0.0, source_height - 1, target_height, dtype=np.float32
    )
    column_coordinates = np.linspace(
        0.0, source_width - 1, target_width, dtype=np.float32
    )
    row_lower = np.floor(row_coordinates).astype(np.int64)
    column_lower = np.floor(column_coordinates).astype(np.int64)
    row_upper = np.minimum(row_lower + 1, source_height - 1)
    column_upper = np.minimum(column_lower + 1, source_width - 1)
    row_weight = (row_coordinates - row_lower)[:, None, None]
    column_weight = (column_coordinates - column_lower)[None, :, None]

    lower_left = mapping.xyz[row_lower[:, None], column_lower[None, :]]
    lower_right = mapping.xyz[row_lower[:, None], column_upper[None, :]]
    upper_left = mapping.xyz[row_upper[:, None], column_lower[None, :]]
    upper_right = mapping.xyz[row_upper[:, None], column_upper[None, :]]
    lower = lower_left * (1.0 - column_weight) + lower_right * column_weight
    upper = upper_left * (1.0 - column_weight) + upper_right * column_weight
    xyz = lower * (1.0 - row_weight) + upper * row_weight

    valid = (
        mapping.valid[row_lower[:, None], column_lower[None, :]]
        & mapping.valid[row_lower[:, None], column_upper[None, :]]
        & mapping.valid[row_upper[:, None], column_lower[None, :]]
        & mapping.valid[row_upper[:, None], column_upper[None, :]]
        & np.isfinite(xyz).all(axis=-1)
    )
    return TifxyzMap(
        xyz=xyz.astype(np.float32, copy=False),
        valid=valid,
        source=mapping.source,
    )
