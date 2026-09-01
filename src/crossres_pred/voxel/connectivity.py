from __future__ import annotations

import numpy as np
from scipy import ndimage

_STRUCTURE_6 = ndimage.generate_binary_structure(3, 1)


def component_bridge_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    minimum_reference_component_voxels: int = 500,
) -> dict[str, int | float]:
    """Count predicted components that fuse distinct reference components.

    This is a reference-relative topology alarm, not a truth metric. A bridge
    can be a correct recovered continuation, so flagged components require
    visual review against CT or a privileged fine-resolution teacher.
    """

    if prediction.shape != reference.shape or prediction.ndim != 3:
        raise ValueError("component bridge masks must be matching 3-D arrays")
    if minimum_reference_component_voxels <= 0:
        raise ValueError("minimum reference component size must be positive")
    predicted = np.asarray(prediction, dtype=bool)
    truth = np.asarray(reference, dtype=bool)
    prediction_labels, prediction_count = ndimage.label(
        predicted,
        structure=_STRUCTURE_6,
    )
    reference_labels, _ = ndimage.label(truth, structure=_STRUCTURE_6)
    reference_sizes = np.bincount(reference_labels.reshape(-1))
    eligible_reference = reference_sizes >= minimum_reference_component_voxels
    if eligible_reference.size:
        eligible_reference[0] = False
    overlap = predicted & truth
    prediction_overlap = prediction_labels[overlap]
    reference_overlap = reference_labels[overlap]
    keep = eligible_reference[reference_overlap]
    if bool(np.any(keep)):
        pairs = np.unique(
            np.stack(
                (prediction_overlap[keep], reference_overlap[keep]),
                axis=1,
            ),
            axis=0,
        )
        references_per_prediction = np.bincount(
            pairs[:, 0],
            minlength=prediction_count + 1,
        )
    else:
        references_per_prediction = np.zeros(
            prediction_count + 1,
            dtype=np.int64,
        )
    merging_labels = np.flatnonzero(references_per_prediction >= 2)
    merge_excess = int(
        np.maximum(references_per_prediction[merging_labels] - 1, 0).sum()
    )
    merging_mask = np.isin(prediction_labels, merging_labels)
    prediction_only_bridge_voxels = int(
        np.count_nonzero(merging_mask & predicted & ~truth)
    )
    foreground = int(np.count_nonzero(predicted))
    return {
        "reference_components": int(np.count_nonzero(eligible_reference)),
        "prediction_components": int(prediction_count),
        "merging_prediction_components": int(merging_labels.size),
        "merged_reference_component_excess": merge_excess,
        "prediction_only_bridge_voxels": prediction_only_bridge_voxels,
        "prediction_only_bridge_fraction": (
            prediction_only_bridge_voxels / max(1, foreground)
        ),
        "minimum_reference_component_voxels": minimum_reference_component_voxels,
    }
