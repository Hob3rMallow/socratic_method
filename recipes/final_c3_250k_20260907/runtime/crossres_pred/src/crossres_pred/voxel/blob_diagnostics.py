from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

_STRUCTURE_4 = ndimage.generate_binary_structure(2, 1)


def _as_mask(value: np.ndarray, *, name: str) -> np.ndarray:
    mask = np.asarray(value, dtype=bool)
    if mask.ndim != 2 or any(size <= 0 for size in mask.shape):
        raise ValueError(f"{name} must be a non-empty 2-D mask")
    return np.ascontiguousarray(mask)


def _component_facts(mask: np.ndarray, minimum_pixels: int) -> dict[str, Any]:
    labels, count = ndimage.label(mask, structure=_STRUCTURE_4)
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    foreground_sizes = sizes[1:]
    return {
        "labels": labels,
        "sizes": sizes,
        "components": int(count),
        "meaningful_components": int(
            np.count_nonzero(foreground_sizes >= minimum_pixels)
        ),
        "largest_component_pixels": int(foreground_sizes.max(initial=0)),
    }


def _taxicab_depth(mask: np.ndarray) -> np.ndarray:
    # Padding supplies background even when a mask reaches the edge of a view.
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    return ndimage.distance_transform_cdt(padded, metric="taxicab")[1:-1, 1:-1]


def _coverage(
    source: np.ndarray,
    target: np.ndarray,
    *,
    tolerance: float,
) -> tuple[int, int, float]:
    source_count = int(np.count_nonzero(source))
    if source_count == 0:
        return 0, 0, 1.0
    if not bool(np.any(target)):
        return 0, source_count, 0.0
    distance = ndimage.distance_transform_edt(~target)
    matched = int(np.count_nonzero(source & (distance <= tolerance)))
    return matched, source_count, matched / source_count


def _reference_relative_merges(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    minimum_reference_component_pixels: int,
) -> dict[str, Any]:
    candidate_facts = _component_facts(candidate, 1)
    reference_facts = _component_facts(reference, 1)
    candidate_labels = candidate_facts["labels"]
    reference_labels = reference_facts["labels"]
    reference_sizes = reference_facts["sizes"]
    eligible = reference_sizes >= minimum_reference_component_pixels
    if eligible.size:
        eligible[0] = False

    touched: dict[int, set[int]] = {}
    overlap = candidate & reference
    if bool(np.any(overlap)):
        pairs = np.unique(
            np.stack((candidate_labels[overlap], reference_labels[overlap]), axis=1),
            axis=0,
        )
        for candidate_label, reference_label in pairs:
            candidate_label = int(candidate_label)
            reference_label = int(reference_label)
            if candidate_label and eligible[reference_label]:
                touched.setdefault(candidate_label, set()).add(reference_label)

    rows: list[dict[str, Any]] = []
    for candidate_label, reference_ids in touched.items():
        if len(reference_ids) < 2:
            continue
        component = candidate_labels == candidate_label
        coordinates = np.argwhere(component)
        lower = coordinates.min(axis=0)
        upper = coordinates.max(axis=0) + 1
        rows.append(
            {
                "candidate_component_label": candidate_label,
                "candidate_component_pixels": int(np.count_nonzero(component)),
                "candidate_only_pixels": int(
                    np.count_nonzero(component & candidate & ~reference)
                ),
                "reference_components_touched": len(reference_ids),
                "reference_component_labels": sorted(reference_ids),
                "reference_component_pixels": int(
                    sum(int(reference_sizes[index]) for index in reference_ids)
                ),
                "bbox_yx": [
                    [int(lower[0]), int(lower[1])],
                    [int(upper[0]), int(upper[1])],
                ],
            }
        )
    rows.sort(
        key=lambda row: (
            int(row["candidate_only_pixels"]),
            int(row["reference_components_touched"]),
        ),
        reverse=True,
    )
    merging_labels = {int(row["candidate_component_label"]) for row in rows}
    merging_mask = np.isin(candidate_labels, list(merging_labels))
    bridge_pixels = int(np.count_nonzero(merging_mask & candidate & ~reference))
    foreground = int(np.count_nonzero(candidate))
    return {
        "eligible_reference_components": int(np.count_nonzero(eligible)),
        "merging_candidate_components": len(rows),
        "merged_reference_component_excess": int(
            sum(int(row["reference_components_touched"]) - 1 for row in rows)
        ),
        "candidate_only_bridge_pixels": bridge_pixels,
        "candidate_only_bridge_fraction": bridge_pixels / max(1, foreground),
        "merging_components": rows,
    }


def slice_blob_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    minimum_component_pixels: int = 16,
    skeleton_tolerance_pixels: float = 1.0,
) -> dict[str, Any]:
    """Measure localized sheet fusion in one 2-D review plane.

    ``reference`` is a morphology comparator, not ground truth.  The merge
    alarm asks whether one candidate component touches two or more meaningful
    reference components.  Skeleton coverage is reported beside it so a
    candidate cannot improve by simply erasing the reference's line work.
    """

    if minimum_component_pixels <= 0:
        raise ValueError("minimum component size must be positive")
    if skeleton_tolerance_pixels < 0:
        raise ValueError("skeleton tolerance cannot be negative")
    predicted = _as_mask(candidate, name="candidate")
    target = _as_mask(reference, name="reference")
    if predicted.shape != target.shape:
        raise ValueError("candidate and reference masks must have matching shapes")

    candidate_facts = _component_facts(predicted, minimum_component_pixels)
    reference_facts = _component_facts(target, minimum_component_pixels)
    foreground = int(np.count_nonzero(predicted))
    reference_foreground = int(np.count_nonzero(target))
    shared = int(np.count_nonzero(predicted & target))
    candidate_only = foreground - shared
    reference_only = reference_foreground - shared

    depth = _taxicab_depth(predicted)
    reference_depth = _taxicab_depth(target)
    interior_r2 = int(np.count_nonzero(depth > 2))
    reference_interior_r2 = int(np.count_nonzero(reference_depth > 2))
    candidate_only_interior_r2 = int(
        np.count_nonzero(predicted & ~target & (depth > 2))
    )
    candidate_skeleton = np.asarray(skeletonize(predicted), dtype=bool)
    reference_skeleton = np.asarray(skeletonize(target), dtype=bool)
    ref_matched, ref_skeleton_pixels, ref_skeleton_recall = _coverage(
        reference_skeleton,
        predicted,
        tolerance=skeleton_tolerance_pixels,
    )
    candidate_matched, candidate_skeleton_pixels, candidate_skeleton_precision = (
        _coverage(
            candidate_skeleton,
            target,
            tolerance=skeleton_tolerance_pixels,
        )
    )
    merges = _reference_relative_merges(
        predicted,
        target,
        minimum_reference_component_pixels=minimum_component_pixels,
    )

    return {
        "shape_yx": [int(value) for value in predicted.shape],
        "minimum_component_pixels": minimum_component_pixels,
        "skeleton_tolerance_pixels": skeleton_tolerance_pixels,
        "candidate_foreground_pixels": foreground,
        "reference_foreground_pixels": reference_foreground,
        "shared_pixels": shared,
        "candidate_only_pixels": candidate_only,
        "reference_only_pixels": reference_only,
        "dice_vs_reference": 2.0 * shared / max(1, foreground + reference_foreground),
        "precision_vs_reference": shared / max(1, foreground),
        "recall_vs_reference": shared / max(1, reference_foreground),
        "foreground_ratio_vs_reference": foreground / max(1, reference_foreground),
        "candidate_components": int(candidate_facts["components"]),
        "candidate_meaningful_components": int(
            candidate_facts["meaningful_components"]
        ),
        "candidate_largest_component_pixels": int(
            candidate_facts["largest_component_pixels"]
        ),
        "candidate_largest_component_fraction": int(
            candidate_facts["largest_component_pixels"]
        )
        / max(1, foreground),
        "reference_components": int(reference_facts["components"]),
        "reference_meaningful_components": int(
            reference_facts["meaningful_components"]
        ),
        "candidate_interior_pixels_r2": interior_r2,
        "candidate_interior_fraction_r2": interior_r2 / max(1, foreground),
        "candidate_max_taxicab_radius": int(depth.max(initial=0)),
        "reference_interior_pixels_r2": reference_interior_r2,
        "reference_interior_fraction_r2": reference_interior_r2
        / max(1, reference_foreground),
        "reference_max_taxicab_radius": int(reference_depth.max(initial=0)),
        "candidate_only_interior_pixels_r2": candidate_only_interior_r2,
        "candidate_interior_fraction_r2_delta_vs_reference": (
            interior_r2 / max(1, foreground)
            - reference_interior_r2 / max(1, reference_foreground)
        ),
        "candidate_max_taxicab_radius_delta_vs_reference": int(
            depth.max(initial=0) - reference_depth.max(initial=0)
        ),
        "candidate_skeleton_pixels": candidate_skeleton_pixels,
        "candidate_sheetness": candidate_skeleton_pixels / max(1, foreground),
        "reference_skeleton_pixels": ref_skeleton_pixels,
        "reference_sheetness": ref_skeleton_pixels / max(1, reference_foreground),
        "reference_skeleton_matched_pixels": ref_matched,
        "reference_skeleton_recall": ref_skeleton_recall,
        "candidate_skeleton_matched_pixels": candidate_matched,
        "candidate_skeleton_precision": candidate_skeleton_precision,
        **merges,
    }


def aggregate_slice_blob_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate counts and recompute rates across a fixed set of planes."""

    if not rows:
        raise ValueError("at least one slice metric row is required")

    def total(key: str) -> int:
        return sum(int(row[key]) for row in rows)

    candidate = total("candidate_foreground_pixels")
    reference = total("reference_foreground_pixels")
    shared = total("shared_pixels")
    ref_skeleton = total("reference_skeleton_pixels")
    candidate_skeleton = total("candidate_skeleton_pixels")
    bridge = total("candidate_only_bridge_pixels")
    candidate_interior = total("candidate_interior_pixels_r2")
    reference_interior = total("reference_interior_pixels_r2")
    return {
        "slice_count": len(rows),
        "candidate_foreground_pixels": candidate,
        "reference_foreground_pixels": reference,
        "shared_pixels": shared,
        "candidate_only_pixels": total("candidate_only_pixels"),
        "reference_only_pixels": total("reference_only_pixels"),
        "foreground_ratio_vs_reference": candidate / max(1, reference),
        "dice_vs_reference": 2.0 * shared / max(1, candidate + reference),
        "precision_vs_reference": shared / max(1, candidate),
        "recall_vs_reference": shared / max(1, reference),
        "candidate_meaningful_components": total(
            "candidate_meaningful_components"
        ),
        "reference_meaningful_components": total(
            "reference_meaningful_components"
        ),
        "candidate_interior_pixels_r2": candidate_interior,
        "candidate_interior_fraction_r2": candidate_interior / max(1, candidate),
        "reference_interior_pixels_r2": reference_interior,
        "reference_interior_fraction_r2": reference_interior / max(1, reference),
        "candidate_only_interior_pixels_r2": total(
            "candidate_only_interior_pixels_r2"
        ),
        "candidate_interior_fraction_r2_delta_vs_reference": (
            candidate_interior / max(1, candidate)
            - reference_interior / max(1, reference)
        ),
        "maximum_candidate_taxicab_radius": max(
            int(row["candidate_max_taxicab_radius"]) for row in rows
        ),
        "maximum_reference_taxicab_radius": max(
            int(row["reference_max_taxicab_radius"]) for row in rows
        ),
        "maximum_candidate_taxicab_radius_delta_vs_reference": max(
            int(row["candidate_max_taxicab_radius_delta_vs_reference"])
            for row in rows
        ),
        "candidate_skeleton_pixels": candidate_skeleton,
        "candidate_sheetness": candidate_skeleton / max(1, candidate),
        "reference_skeleton_pixels": ref_skeleton,
        "reference_sheetness": ref_skeleton / max(1, reference),
        "reference_skeleton_matched_pixels": total(
            "reference_skeleton_matched_pixels"
        ),
        "reference_skeleton_recall": total("reference_skeleton_matched_pixels")
        / max(1, ref_skeleton),
        "candidate_skeleton_matched_pixels": total(
            "candidate_skeleton_matched_pixels"
        ),
        "candidate_skeleton_precision": total("candidate_skeleton_matched_pixels")
        / max(1, candidate_skeleton),
        "eligible_reference_components": total("eligible_reference_components"),
        "merging_candidate_components": total("merging_candidate_components"),
        "merged_reference_component_excess": total(
            "merged_reference_component_excess"
        ),
        "candidate_only_bridge_pixels": bridge,
        "candidate_only_bridge_fraction": bridge / max(1, candidate),
    }
