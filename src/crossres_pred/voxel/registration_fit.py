from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


GLOBAL_AFFINE_FIT_CONTRACT = "crossres-ct-global-affine-refinement-v1"


@dataclass(frozen=True)
class GlobalAffineFitOptions:
    minimum_fine_ct_support_fraction: float = 0.70
    minimum_structure_ncc: float = 0.15
    minimum_peak_margin: float = 0.01
    minimum_eligible_anchors: int = 12
    holdout_modulus: int = 3
    huber_delta_voxels: float = 1.5
    maximum_holdout_component_error: float = 2.0
    maximum_holdout_norm_error: float = 3.0
    maximum_holdout_median_norm_error: float = 1.75

    def validate(self) -> None:
        for name in ("minimum_fine_ct_support_fraction",):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not -1.0 <= self.minimum_structure_ncc <= 1.0:
            raise ValueError("minimum_structure_ncc must be in [-1, 1]")
        if self.minimum_peak_margin < 0.0:
            raise ValueError("minimum_peak_margin must be non-negative")
        if self.minimum_eligible_anchors < 8:
            raise ValueError("minimum_eligible_anchors must be at least 8")
        if self.holdout_modulus < 3:
            raise ValueError("holdout_modulus must be at least 3")
        for name in (
            "huber_delta_voxels",
            "maximum_holdout_component_error",
            "maximum_holdout_norm_error",
            "maximum_holdout_median_norm_error",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")


def eligible_anchor_rows(
    anchors: Sequence[dict[str, Any]],
    options: GlobalAffineFitOptions,
) -> list[dict[str, Any]]:
    """Select CT-discriminative residual measurements without using labels."""

    options.validate()
    selected: list[dict[str, Any]] = []
    for row in anchors:
        registration = row.get("registration_3d") or {}
        sampling = row.get("sampling") or {}
        margin = registration.get("peak_margin")
        if (
            row.get("status") == "measured"
            and bool(registration.get("available", False))
            and registration.get("best_shift_zyx") is not None
            and not bool(registration.get("best_on_search_boundary", False))
            and float(sampling.get("fine_ct_support_fraction", 0.0))
            >= options.minimum_fine_ct_support_fraction
            and float(registration.get("best_structure_ncc", -np.inf))
            >= options.minimum_structure_ncc
            and margin is not None
            and float(margin) >= options.minimum_peak_margin
        ):
            selected.append(row)
    return selected


def _anchor_arrays(
    anchors: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    coarse: list[np.ndarray] = []
    shifts: list[np.ndarray] = []
    patch_ids: list[str] = []
    for row in anchors:
        origin = np.asarray(row["audit_origin_zyx"], dtype=np.float64)
        shape = np.asarray(row["audit_shape_zyx"], dtype=np.float64)
        coarse.append(origin + 0.5 * shape)
        shifts.append(
            np.asarray(row["registration_3d"]["best_shift_zyx"], dtype=np.float64)
        )
        patch_ids.append(str(row["patch_id"]))
    if not coarse:
        return np.empty((0, 3)), np.empty((0, 3)), []
    return np.stack(coarse), np.stack(shifts), patch_ids


def _split_mask(patch_ids: Sequence[str], modulus: int) -> np.ndarray:
    values = []
    for patch_id in patch_ids:
        digest = hashlib.sha256(patch_id.encode("utf-8")).digest()
        values.append(int.from_bytes(digest[:8], "little") % modulus == 0)
    result = np.asarray(values, dtype=bool)
    # A deterministic hash can occasionally make a small sample lopsided.
    # Preserve independence while guaranteeing enough rows for a 3-D affine.
    minimum = max(4, len(values) // (modulus + 1))
    if int(result.sum()) < minimum:
        order = np.argsort(
            [hashlib.sha256(value.encode("utf-8")).digest() for value in patch_ids]
        )
        result[:] = False
        result[order[:minimum]] = True
    if int((~result).sum()) < 5:
        result[:] = False
        result[::modulus] = True
    return result


def _fit_correction(
    coarse_zyx: np.ndarray,
    shifts_zyx: np.ndarray,
    *,
    huber_delta: float,
) -> np.ndarray:
    """Fit an affine mapping from old mapped coordinates to fixed coordinates."""

    coarse = np.asarray(coarse_zyx, dtype=np.float64)
    shifts = np.asarray(shifts_zyx, dtype=np.float64)
    if coarse.shape != shifts.shape or coarse.ndim != 2 or coarse.shape[1] != 3:
        raise ValueError("coarse coordinates and shifts must both be Nx3")
    if coarse.shape[0] < 5:
        raise ValueError("a 3-D affine correction needs at least five anchors")
    old = coarse - shifts
    center = old.mean(axis=0)
    scale = np.maximum(old.std(axis=0), 1.0)
    design = np.column_stack(((old - center) / scale, np.ones(old.shape[0])))
    delta = coarse - old
    weights = np.ones(old.shape[0], dtype=np.float64)
    coefficients = np.zeros((4, 3), dtype=np.float64)
    for _ in range(25):
        root = np.sqrt(weights)[:, None]
        coefficients = np.linalg.lstsq(design * root, delta * root, rcond=None)[0]
        residual = delta - design @ coefficients
        norms = np.linalg.norm(residual, axis=1)
        updated = np.minimum(1.0, huber_delta / np.maximum(norms, 1.0e-12))
        if np.max(np.abs(updated - weights)) < 1.0e-5:
            weights = updated
            break
        weights = updated
    slope_zyx = (coefficients[:3, :] / scale[:, None]).T
    intercept_zyx = coefficients[3, :] - slope_zyx @ center
    correction_zyx = np.eye(4, dtype=np.float64)
    correction_zyx[:3, :3] += slope_zyx
    correction_zyx[:3, 3] = intercept_zyx
    return correction_zyx


def _zyx_to_xyz_affine(value: np.ndarray) -> np.ndarray:
    reverse = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = reverse @ value[:3, :3] @ reverse
    result[:3, 3] = reverse @ value[:3, 3]
    return result


def _residuals(
    correction_zyx: np.ndarray,
    coarse_zyx: np.ndarray,
    shifts_zyx: np.ndarray,
) -> np.ndarray:
    old = coarse_zyx - shifts_zyx
    predicted = old @ correction_zyx[:3, :3].T + correction_zyx[:3, 3]
    return coarse_zyx - predicted


def _metrics(residuals: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(np.asarray(residuals, dtype=np.float64))
    norms = np.linalg.norm(residuals, axis=1)
    return {
        "count": int(residuals.shape[0]),
        "median_residual_zyx": np.median(residuals, axis=0).tolist(),
        "median_abs_component_zyx": np.median(absolute, axis=0).tolist(),
        "maximum_abs_component": float(np.max(absolute)),
        "median_norm": float(np.median(norms)),
        "p90_norm": float(np.percentile(norms, 90.0)),
        "maximum_norm": float(np.max(norms)),
        "rms": float(np.sqrt(np.mean(np.square(residuals)))),
    }


def fit_global_affine_refinement(
    anchors: Sequence[dict[str, Any]],
    declared_affine_xyz: Sequence[Sequence[float]],
    options: GlobalAffineFitOptions | None = None,
) -> dict[str, Any]:
    """Qualify on a held-out CT split, then refit a proposed global affine."""

    options = options or GlobalAffineFitOptions()
    options.validate()
    eligible = eligible_anchor_rows(anchors, options)
    coarse, shifts, patch_ids = _anchor_arrays(eligible)
    reasons: list[str] = []
    if len(eligible) < options.minimum_eligible_anchors:
        reasons.append(
            f"only {len(eligible)} CT-discriminative anchors; "
            f"need {options.minimum_eligible_anchors}"
        )
        return {
            "contract": GLOBAL_AFFINE_FIT_CONTRACT,
            "status": "local-registration-required",
            "eligible_anchor_count": len(eligible),
            "eligible_patch_ids": patch_ids,
            "reasons": reasons,
        }
    holdout = _split_mask(patch_ids, options.holdout_modulus)
    fit = ~holdout
    if np.linalg.matrix_rank(
        np.column_stack((coarse[fit], np.ones(int(fit.sum()))))
    ) < 4:
        reasons.append("fit anchors do not span a 3-D affine")
    if np.linalg.matrix_rank(
        np.column_stack((coarse[holdout], np.ones(int(holdout.sum()))))
    ) < 4:
        reasons.append("held-out anchors do not span a 3-D affine")
    final_correction_zyx = _fit_correction(
        coarse, shifts, huber_delta=options.huber_delta_voxels
    )
    correction_xyz = _zyx_to_xyz_affine(final_correction_zyx)
    declared = np.eye(4, dtype=np.float64)
    declared[:3, :] = np.asarray(declared_affine_xyz, dtype=np.float64)
    proposed = correction_xyz @ declared
    final_metrics = _metrics(_residuals(final_correction_zyx, coarse, shifts))
    if reasons:
        return {
            "contract": GLOBAL_AFFINE_FIT_CONTRACT,
            "status": "local-registration-required",
            "eligible_anchor_count": len(eligible),
            "eligible_patch_ids": patch_ids,
            "reasons": reasons,
        }
    qualified_correction = _fit_correction(
        coarse[fit], shifts[fit], huber_delta=options.huber_delta_voxels
    )
    fit_metrics = _metrics(_residuals(qualified_correction, coarse[fit], shifts[fit]))
    holdout_metrics = _metrics(
        _residuals(qualified_correction, coarse[holdout], shifts[holdout])
    )
    if (
        holdout_metrics["maximum_abs_component"]
        > options.maximum_holdout_component_error
    ):
        reasons.append("held-out component error exceeds tolerance")
    if holdout_metrics["maximum_norm"] > options.maximum_holdout_norm_error:
        reasons.append("held-out residual norm exceeds tolerance")
    if (
        holdout_metrics["median_norm"]
        > options.maximum_holdout_median_norm_error
    ):
        reasons.append("held-out median residual norm exceeds tolerance")
    if reasons:
        return {
            "contract": GLOBAL_AFFINE_FIT_CONTRACT,
            "status": "local-registration-required",
            "eligible_anchor_count": len(eligible),
            "eligible_patch_ids": patch_ids,
            "fit_patch_ids": [patch_ids[index] for index in np.flatnonzero(fit)],
            "holdout_patch_ids": [
                patch_ids[index] for index in np.flatnonzero(holdout)
            ],
            "fit_metrics": fit_metrics,
            "holdout_metrics": holdout_metrics,
            "final_all_anchor_metrics": final_metrics,
            # A failed global-only gate can still provide a much better
            # initialization for independently gated patch-local registration.
            # This affine is never sufficient on its own and is deliberately
            # named separately from a globally qualified proposal.
            "local_initialization_correction_affine_zyx": (
                final_correction_zyx.tolist()
            ),
            "local_initialization_correction_affine_xyz": correction_xyz.tolist(),
            "local_initialization_to_coarse_affine_xyz": proposed[:3, :].tolist(),
            "reasons": reasons,
        }
    return {
        "contract": GLOBAL_AFFINE_FIT_CONTRACT,
        "status": "global-affine-candidate",
        "eligible_anchor_count": len(eligible),
        "eligible_patch_ids": patch_ids,
        "fit_patch_ids": [patch_ids[index] for index in np.flatnonzero(fit)],
        "holdout_patch_ids": [patch_ids[index] for index in np.flatnonzero(holdout)],
        "fit_metrics": fit_metrics,
        "holdout_metrics": holdout_metrics,
        "final_all_anchor_metrics": final_metrics,
        "correction_affine_zyx": final_correction_zyx.tolist(),
        "correction_affine_xyz": correction_xyz.tolist(),
        "proposed_to_coarse_affine_xyz": proposed[:3, :].tolist(),
        "reasons": [],
    }
