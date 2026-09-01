"""Selection metrics for spatially held-out cross-resolution failures.

The helpers in this module are intentionally independent of model and storage
code. They define the two contracts needed by the PHerc0139 reviewer audit:

* a candidate context must be spatially disjoint from every possible training
  patch, with a recorded Euclidean box-to-box margin; and
* a useful slice contains a long teacher crest, a thick/blob-like released-M7
  prediction, and a student prediction that leaves coherent gaps in the crest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

_EIGHT_CONNECTED = np.ones((3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class CrestFailureMetrics:
    """Geometry diagnostics for one two-dimensional review slice."""

    crest_voxels: int
    largest_crest_component: int
    student_crest_recall_r1: float
    student_recovered_segments: int
    missing_crest_voxels: int
    largest_missing_crest_gap: int
    largest_missing_gap_fraction: float
    m7_foreground_voxels: int
    m7_false_positive_known_voxels: int
    m7_interior_fraction_r2: float
    m7_max_radius: float
    student_foreground_voxels: int
    student_missing_crest_probability_mean: float
    student_missing_crest_probability_max: float
    score: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def box_distances_to_many(
    origin_zyx: tuple[int, int, int] | np.ndarray,
    shape_zyx: tuple[int, int, int] | np.ndarray,
    other_origins_zyx: np.ndarray,
    other_shapes_zyx: np.ndarray,
) -> np.ndarray:
    """Euclidean distances between one half-open box and many half-open boxes.

    Touching and overlapping boxes have distance zero. Coordinates are in
    voxel units and may be integer or floating point.
    """

    origin = np.asarray(origin_zyx, dtype=np.float64)
    shape = np.asarray(shape_zyx, dtype=np.float64)
    others = np.asarray(other_origins_zyx, dtype=np.float64)
    other_shapes = np.asarray(other_shapes_zyx, dtype=np.float64)
    if origin.shape != (3,) or shape.shape != (3,):
        raise ValueError("origin_zyx and shape_zyx must contain three values")
    if others.ndim != 2 or others.shape[1] != 3 or other_shapes.shape != others.shape:
        raise ValueError("other origins and shapes must be matching N x 3 arrays")
    if np.any(shape <= 0) or np.any(other_shapes <= 0):
        raise ValueError("box shapes must be positive")

    upper = origin + shape
    other_upper = others + other_shapes
    axis_gap = np.maximum(np.maximum(origin - other_upper, others - upper), 0.0)
    return np.sqrt(np.square(axis_gap).sum(axis=1))


def minimum_box_distance(
    origin_zyx: tuple[int, int, int] | np.ndarray,
    shape_zyx: tuple[int, int, int] | np.ndarray,
    other_origins_zyx: np.ndarray,
    other_shapes_zyx: np.ndarray,
) -> float:
    distances = box_distances_to_many(
        origin_zyx,
        shape_zyx,
        other_origins_zyx,
        other_shapes_zyx,
    )
    return float(distances.min(initial=np.inf))


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, count = ndimage.label(mask, structure=_EIGHT_CONNECTED)
    if count == 0:
        return np.zeros_like(mask, dtype=bool), 0
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0
    label = int(np.argmax(sizes))
    return labels == label, int(sizes[label])


def crest_failure_metrics(
    *,
    m7: np.ndarray,
    student: np.ndarray,
    student_probability: np.ndarray,
    teacher_positive: np.ndarray,
    teacher_crest: np.ndarray,
    valid: np.ndarray,
    tolerance_radius: int = 1,
) -> CrestFailureMetrics:
    """Score a slice for the target failure: blobbed M7, fragmented student.

    Crest recall is tolerant by one coarse voxel by default. The tolerance
    prevents sub-voxel registration or band-thickness differences from being
    mislabeled as fragmentation. Blob measurements remain strict.
    """

    arrays = [m7, student, student_probability, teacher_positive, teacher_crest, valid]
    shape = np.asarray(arrays[0]).shape
    if len(shape) != 2 or any(np.asarray(value).shape != shape for value in arrays):
        raise ValueError("all slice arrays must share one two-dimensional shape")
    if tolerance_radius < 0:
        raise ValueError("tolerance_radius cannot be negative")

    known = np.asarray(valid, dtype=bool)
    crest = np.asarray(teacher_crest, dtype=bool) & known
    teacher = np.asarray(teacher_positive, dtype=bool) & known
    m7_mask = np.asarray(m7, dtype=bool)
    student_mask = np.asarray(student, dtype=bool)
    probability = np.asarray(student_probability, dtype=np.float32)

    largest_crest, largest_size = _largest_component(crest)
    if tolerance_radius:
        student_tolerant = ndimage.binary_dilation(
            student_mask,
            structure=_EIGHT_CONNECTED,
            iterations=tolerance_radius,
        )
    else:
        student_tolerant = student_mask
    recovered = largest_crest & student_tolerant
    missing = largest_crest & ~student_tolerant
    _, recovered_count = ndimage.label(recovered, structure=_EIGHT_CONNECTED)
    _, largest_missing = _largest_component(missing)
    recovered_voxels = int(np.count_nonzero(recovered))
    missing_voxels = int(np.count_nonzero(missing))
    crest_recall = recovered_voxels / max(1, largest_size)
    gap_fraction = largest_missing / max(1, largest_size)

    m7_interior = ndimage.binary_erosion(
        m7_mask,
        structure=_EIGHT_CONNECTED,
        iterations=2,
        border_value=0,
    )
    m7_foreground = int(np.count_nonzero(m7_mask))
    m7_interior_fraction = float(np.count_nonzero(m7_interior)) / max(
        1, m7_foreground
    )
    teacher_tolerant = ndimage.binary_dilation(
        teacher,
        structure=_EIGHT_CONNECTED,
        iterations=1,
    )
    m7_false_positive = int(np.count_nonzero(m7_mask & known & ~teacher_tolerant))
    m7_radius = float(ndimage.distance_transform_edt(m7_mask).max(initial=0.0))

    missing_probabilities = probability[missing]
    missing_probability_mean = (
        float(missing_probabilities.mean()) if missing_probabilities.size else 0.0
    )
    missing_probability_max = (
        float(missing_probabilities.max(initial=0.0))
        if missing_probabilities.size
        else 0.0
    )

    line_strength = float(np.log1p(largest_size))
    fragmentation = max(0, int(recovered_count) - 1)
    missing_force = 1.5 * (1.0 - crest_recall) + 2.5 * gap_fraction
    blob_force = (
        1.0
        + 0.30 * max(0.0, m7_radius - 1.5)
        + 1.5 * m7_interior_fraction
        + 0.05 * np.log1p(m7_false_positive)
    )
    score = line_strength * missing_force * blob_force + 0.05 * fragmentation

    return CrestFailureMetrics(
        crest_voxels=int(np.count_nonzero(crest)),
        largest_crest_component=largest_size,
        student_crest_recall_r1=float(crest_recall),
        student_recovered_segments=int(recovered_count),
        missing_crest_voxels=missing_voxels,
        largest_missing_crest_gap=largest_missing,
        largest_missing_gap_fraction=float(gap_fraction),
        m7_foreground_voxels=m7_foreground,
        m7_false_positive_known_voxels=m7_false_positive,
        m7_interior_fraction_r2=m7_interior_fraction,
        m7_max_radius=m7_radius,
        student_foreground_voxels=int(np.count_nonzero(student_mask)),
        student_missing_crest_probability_mean=missing_probability_mean,
        student_missing_crest_probability_max=missing_probability_max,
        score=float(score),
    )
