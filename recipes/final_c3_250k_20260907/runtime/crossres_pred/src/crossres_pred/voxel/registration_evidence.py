from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy import ndimage

EVIDENCE_SCHEMA = "crossres-pinned-registration-v1"
EVIDENCE_METHOD = "ct-pyramid-affine-registration-v1"
MIN_MEAN_MASK_DICE = 0.95
MIN_MINIMUM_MASK_DICE = 0.85
MIN_MEAN_INTENSITY_CORRELATION = 0.50
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def affine_3x4(value: object, *, context: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 4):
        raise ValueError(f"{context} must be a 3x4 array")
    if not np.isfinite(array).all():
        raise ValueError(f"{context} must contain only finite values")
    if abs(float(np.linalg.det(array[:, :3]))) < 1.0e-12:
        raise ValueError(f"{context} is singular")
    return array


def affine_at_pyramid_level(affine_l0_xyz: np.ndarray, level: int) -> np.ndarray:
    if level < 0:
        raise ValueError("pyramid level must be nonnegative")
    result = affine_3x4(
        affine_l0_xyz, context="fine_l0_to_coarse_l0_affine_xyz"
    ).copy()
    result[:, 3] /= float(2**level)
    return result


def resample_fine_to_coarse(
    fine: np.ndarray,
    *,
    coarse_shape_zyx: Sequence[int],
    affine_l0_xyz: np.ndarray,
    pyramid_level: int,
) -> np.ndarray:
    affine = affine_at_pyramid_level(affine_l0_xyz, pyramid_level)
    reverse_xyz_zyx = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    linear_zyx = reverse_xyz_zyx @ affine[:, :3] @ reverse_xyz_zyx
    translation_zyx = reverse_xyz_zyx @ affine[:, 3]
    inverse = np.linalg.inv(linear_zyx)
    offset = -inverse @ translation_zyx
    return ndimage.affine_transform(
        np.asarray(fine),
        inverse,
        offset=offset,
        output_shape=tuple(int(value) for value in coarse_shape_zyx),
        order=1,
        mode="constant",
        cval=0,
        prefilter=False,
    )


def _slice_metric(
    fixed: np.ndarray, moving: np.ndarray, *, coarse_z: int
) -> dict[str, object]:
    fixed_float = np.asarray(fixed, dtype=np.float32)
    moving_float = np.asarray(moving, dtype=np.float32)
    fixed_mask = fixed_float > 0
    moving_mask = moving_float > 1
    intersection = fixed_mask & moving_mask
    denominator = int(np.count_nonzero(fixed_mask)) + int(
        np.count_nonzero(moving_mask)
    )
    dice = 2.0 * float(np.count_nonzero(intersection)) / max(denominator, 1)
    correlation = float("nan")
    if int(np.count_nonzero(intersection)) > 100:
        fixed_values = fixed_float[intersection]
        moving_values = moving_float[intersection]
        fixed_values = (fixed_values - fixed_values.mean()) / (
            fixed_values.std() + 1.0e-6
        )
        moving_values = (moving_values - moving_values.mean()) / (
            moving_values.std() + 1.0e-6
        )
        correlation = float(np.mean(fixed_values * moving_values))
    return {
        "coarse_z": int(coarse_z),
        "mask_dice": dice,
        "intensity_correlation": correlation,
        "fixed_nonzero_voxels": int(np.count_nonzero(fixed_mask)),
        "moving_nonzero_voxels": int(np.count_nonzero(moving_mask)),
    }


def _summarize_slices(
    coarse: np.ndarray, aligned: np.ndarray, slice_z: Sequence[int]
) -> dict[str, object]:
    rows = [
        _slice_metric(coarse[z], aligned[z], coarse_z=z)
        for z in (int(value) for value in slice_z)
    ]
    correlations = np.asarray(
        [float(row["intensity_correlation"]) for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(correlations).all():
        raise ValueError("registration slice correlation is not finite")
    dice = np.asarray([float(row["mask_dice"]) for row in rows])
    return {
        "slice_z": [int(value) for value in slice_z],
        "count": len(rows),
        "mean_mask_dice": float(dice.mean()),
        "minimum_mask_dice": float(dice.min()),
        "mean_intensity_correlation": float(correlations.mean()),
        "slices": rows,
    }


def build_registration_evidence(
    *,
    sample_id: str,
    fine_volume_id: str,
    coarse_volume_id: str,
    fine_voxel_um: float,
    coarse_voxel_um: float,
    fine: np.ndarray,
    coarse: np.ndarray,
    affine_l0_xyz: np.ndarray,
    pyramid_level: int,
    fit_slice_z: Sequence[int],
    held_out_slice_z: Sequence[int],
    fine_source: dict[str, object],
    coarse_source: dict[str, object],
    notes: Sequence[str] = (),
) -> dict[str, object]:
    fit = tuple(int(value) for value in fit_slice_z)
    held_out = tuple(int(value) for value in held_out_slice_z)
    if len(fit) < 5 or len(held_out) < 5:
        raise ValueError("fit and held-out registration sets need at least 5 slices")
    if set(fit) & set(held_out):
        raise ValueError("fit and held-out registration slices overlap")
    depth = int(np.asarray(coarse).shape[0])
    if any(value < 0 or value >= depth for value in (*fit, *held_out)):
        raise ValueError("registration slice is outside the coarse volume")
    affine = affine_3x4(
        affine_l0_xyz, context="fine_l0_to_coarse_l0_affine_xyz"
    )
    aligned = resample_fine_to_coarse(
        fine,
        coarse_shape_zyx=np.asarray(coarse).shape,
        affine_l0_xyz=affine,
        pyramid_level=pyramid_level,
    )
    payload: dict[str, object] = {
        "schema": EVIDENCE_SCHEMA,
        "schema_version": 1,
        "method": EVIDENCE_METHOD,
        "sample_id": str(sample_id),
        "fine": {
            "volume_id": str(fine_volume_id),
            "voxel_um": float(fine_voxel_um),
            "pyramid_level": int(pyramid_level),
            "shape_zyx": list(np.asarray(fine).shape),
            **fine_source,
        },
        "coarse": {
            "volume_id": str(coarse_volume_id),
            "voxel_um": float(coarse_voxel_um),
            "pyramid_level": int(pyramid_level),
            "shape_zyx": list(np.asarray(coarse).shape),
            **coarse_source,
        },
        "fine_l0_to_coarse_l0_affine_xyz": affine.tolist(),
        "fit": _summarize_slices(coarse, aligned, fit),
        "held_out": _summarize_slices(coarse, aligned, held_out),
        "quality_gates": {
            "minimum_mean_mask_dice": MIN_MEAN_MASK_DICE,
            "minimum_single_slice_mask_dice": MIN_MINIMUM_MASK_DICE,
            "minimum_mean_intensity_correlation": (
                MIN_MEAN_INTENSITY_CORRELATION
            ),
        },
        "notes": [str(note) for note in notes],
    }
    validate_registration_evidence(
        payload,
        sample_id=sample_id,
        fine_volume_id=fine_volume_id,
        coarse_volume_id=coarse_volume_id,
        fine_voxel_um=fine_voxel_um,
        coarse_voxel_um=coarse_voxel_um,
    )
    return payload


def _positive_finite(value: object, *, context: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{context} must be positive and finite")
    return result


def _validate_source(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    for name in ("source_zarr", "mirror_manifest_sha256", "plan_sha256"):
        if not value.get(name):
            raise ValueError(f"{context}.{name} is required")
    for name in ("mirror_manifest_sha256", "plan_sha256"):
        digest = str(value[name])
        if not _SHA256.fullmatch(digest):
            raise ValueError(f"{context}.{name} is not a SHA-256 digest")
    return value


def _validate_summary(value: object, *, context: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    slices = value.get("slice_z")
    if not isinstance(slices, list) or len(slices) < 5:
        raise ValueError(f"{context}.slice_z needs at least 5 slices")
    slice_z = [int(item) for item in slices]
    if len(set(slice_z)) != len(slice_z):
        raise ValueError(f"{context}.slice_z contains duplicates")
    if int(value.get("count", -1)) != len(slices):
        raise ValueError(f"{context}.count does not match slice_z")
    rows = value.get("slices")
    if not isinstance(rows, list) or len(rows) != len(slices):
        raise ValueError(f"{context}.slices does not match slice_z")
    measured_dice: list[float] = []
    measured_correlations: list[float] = []
    for expected_z, row in zip(slice_z, rows, strict=True):
        if not isinstance(row, dict):
            raise TypeError(f"{context}.slices rows must be objects")
        if int(row.get("coarse_z", -1)) != expected_z:
            raise ValueError(f"{context}.slices coarse_z does not match slice_z")
        row_dice = float(row.get("mask_dice", float("nan")))
        row_correlation = float(
            row.get("intensity_correlation", float("nan"))
        )
        if not math.isfinite(row_dice) or not 0.0 <= row_dice <= 1.0:
            raise ValueError(f"{context}.slices mask Dice is invalid")
        if not math.isfinite(row_correlation) or not -1.01 <= row_correlation <= 1.01:
            raise ValueError(f"{context}.slices correlation is invalid")
        for name in ("fixed_nonzero_voxels", "moving_nonzero_voxels"):
            if int(row.get(name, -1)) < 0:
                raise ValueError(f"{context}.slices {name} is invalid")
        measured_dice.append(row_dice)
        measured_correlations.append(row_correlation)
    mean_dice = float(value.get("mean_mask_dice", float("nan")))
    minimum_dice = float(value.get("minimum_mask_dice", float("nan")))
    correlation = float(
        value.get("mean_intensity_correlation", float("nan"))
    )
    if not all(math.isfinite(item) for item in (mean_dice, minimum_dice, correlation)):
        raise ValueError(f"{context} metrics must be finite")
    recomputed = (
        (mean_dice, float(np.mean(measured_dice)), "mean mask Dice"),
        (minimum_dice, min(measured_dice), "minimum mask Dice"),
        (
            correlation,
            float(np.mean(measured_correlations)),
            "mean intensity correlation",
        ),
    )
    for recorded, actual, name in recomputed:
        if not math.isclose(recorded, actual, rel_tol=1.0e-10, abs_tol=1.0e-10):
            raise ValueError(f"{context} {name} does not match slice rows")
    if mean_dice < MIN_MEAN_MASK_DICE:
        raise ValueError(f"{context} mean mask Dice {mean_dice:.4f} failed")
    if minimum_dice < MIN_MINIMUM_MASK_DICE:
        raise ValueError(f"{context} minimum mask Dice {minimum_dice:.4f} failed")
    if correlation < MIN_MEAN_INTENSITY_CORRELATION:
        raise ValueError(
            f"{context} mean intensity correlation {correlation:.4f} failed"
        )


def validate_registration_evidence(
    value: object,
    *,
    sample_id: str,
    fine_volume_id: str,
    coarse_volume_id: str,
    fine_voxel_um: float,
    coarse_voxel_um: float,
) -> np.ndarray:
    if not isinstance(value, dict):
        raise TypeError("registration evidence must be an object")
    if value.get("schema") != EVIDENCE_SCHEMA or value.get("schema_version") != 1:
        raise ValueError("unsupported registration evidence schema")
    if value.get("method") != EVIDENCE_METHOD:
        raise ValueError("unsupported registration evidence method")
    if str(value.get("sample_id")) != str(sample_id):
        raise ValueError("registration evidence sample_id changed")
    fine = _validate_source(value.get("fine"), context="fine")
    coarse = _validate_source(value.get("coarse"), context="coarse")
    if str(fine.get("volume_id")) != str(fine_volume_id):
        raise ValueError("registration evidence fine volume changed")
    if str(coarse.get("volume_id")) != str(coarse_volume_id):
        raise ValueError("registration evidence coarse volume changed")
    actual_fine_um = _positive_finite(fine.get("voxel_um"), context="fine.voxel_um")
    actual_coarse_um = _positive_finite(
        coarse.get("voxel_um"), context="coarse.voxel_um"
    )
    if not math.isclose(actual_fine_um, float(fine_voxel_um), abs_tol=1.0e-6):
        raise ValueError("registration evidence fine voxel size changed")
    if not math.isclose(actual_coarse_um, float(coarse_voxel_um), abs_tol=1.0e-6):
        raise ValueError("registration evidence coarse voxel size changed")
    fine_level = int(fine.get("pyramid_level", -1))
    coarse_level = int(coarse.get("pyramid_level", -1))
    if fine_level < 0 or fine_level != coarse_level:
        raise ValueError("registration evidence pyramid levels do not match")
    for source, name in ((fine, "fine"), (coarse, "coarse")):
        shape = source.get("shape_zyx")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(int(item) <= 0 for item in shape)
        ):
            raise ValueError(f"registration evidence {name} shape is invalid")
    affine = affine_3x4(
        value.get("fine_l0_to_coarse_l0_affine_xyz"),
        context="fine_l0_to_coarse_l0_affine_xyz",
    )
    measured = np.linalg.svd(affine[:, :3], compute_uv=False)
    expected = actual_fine_um / actual_coarse_um
    tolerance = max(0.02, expected * 0.08)
    if float(np.max(np.abs(measured - expected))) > tolerance:
        raise ValueError(
            f"registration affine scales {measured} do not match {expected}"
        )
    _validate_summary(value.get("fit"), context="fit")
    _validate_summary(value.get("held_out"), context="held_out")
    fit_slices = {int(item) for item in value["fit"]["slice_z"]}
    held_out_slices = {int(item) for item in value["held_out"]["slice_z"]}
    if fit_slices & held_out_slices:
        raise ValueError("registration fit and held-out slices overlap")
    coarse_depth = int(coarse["shape_zyx"][0])
    if any(item < 0 or item >= coarse_depth for item in fit_slices | held_out_slices):
        raise ValueError("registration evidence slice is outside coarse shape")
    return affine


def load_registration_evidence(
    path: str | Path,
    **expected: object,
) -> tuple[dict[str, object], np.ndarray, str]:
    import json

    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    affine = validate_registration_evidence(payload, **expected)
    return payload, affine, sha256_file(source)
