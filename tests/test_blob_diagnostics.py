from __future__ import annotations

import numpy as np
import pytest

from crossres_pred.voxel.blob_diagnostics import (
    aggregate_slice_blob_metrics,
    slice_blob_metrics,
)


def test_slice_blob_metrics_detects_fusion_without_calling_line_loss() -> None:
    reference = np.zeros((15, 15), dtype=bool)
    reference[4, 2:9] = True
    reference[8, 2:9] = True
    candidate = reference.copy()
    candidate[5:8, 5] = True

    metrics = slice_blob_metrics(
        candidate,
        reference,
        minimum_component_pixels=4,
        skeleton_tolerance_pixels=0,
    )

    assert metrics["reference_meaningful_components"] == 2
    assert metrics["candidate_meaningful_components"] == 1
    assert metrics["merging_candidate_components"] == 1
    assert metrics["merged_reference_component_excess"] == 1
    assert metrics["candidate_only_bridge_pixels"] == 3
    assert metrics["candidate_only_interior_pixels_r2"] == 0
    assert metrics["reference_skeleton_recall"] == 1.0
    assert metrics["candidate_skeleton_precision"] < 1.0
    assert metrics["merging_components"][0]["bbox_yx"] == [[4, 2], [9, 9]]


def test_aggregate_recomputes_weighted_rates() -> None:
    reference = np.zeros((9, 9), dtype=bool)
    reference[4, 2:7] = True
    exact = slice_blob_metrics(reference, reference, minimum_component_pixels=2)
    empty = slice_blob_metrics(
        np.zeros_like(reference), reference, minimum_component_pixels=2
    )

    aggregate = aggregate_slice_blob_metrics([exact, empty])

    assert aggregate["slice_count"] == 2
    assert aggregate["reference_skeleton_recall"] == pytest.approx(0.5)
    assert aggregate["recall_vs_reference"] == pytest.approx(0.5)
    assert aggregate["candidate_only_bridge_pixels"] == 0
    assert aggregate["reference_interior_pixels_r2"] == 0


def test_slice_blob_metrics_rejects_mismatched_or_nonplanar_masks() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        slice_blob_metrics(np.zeros((2, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="2-D"):
        slice_blob_metrics(np.zeros((1, 2, 2)), np.zeros((1, 2, 2)))
