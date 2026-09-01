from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

_STRUCTURE_6 = ndimage.generate_binary_structure(3, 1)
_OFFSETS_6 = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)
_LABEL_SENTINEL = np.iinfo(np.int32).max


@dataclass(frozen=True)
class RidgeGrowthResult:
    mask: np.ndarray
    seed_components: int
    final_components: int
    seed_positive: int
    final_positive: int
    eligible_support: int
    completed_steps: int
    added_per_step: tuple[int, ...]
    component_conflict_rejections: int
    thickness_rejections: int
    reached_step_limit: bool

    @property
    def added_positive(self) -> int:
        return self.final_positive - self.seed_positive

    def to_dict(self) -> dict[str, int | bool | list[int] | float]:
        return {
            "seed_components": self.seed_components,
            "final_components": self.final_components,
            "seed_positive": self.seed_positive,
            "final_positive": self.final_positive,
            "added_positive": self.added_positive,
            "foreground_growth_fraction": self.added_positive
            / max(1, self.seed_positive),
            "eligible_support": self.eligible_support,
            "completed_steps": self.completed_steps,
            "added_per_step": list(self.added_per_step),
            "component_conflict_rejections": self.component_conflict_rejections,
            "thickness_rejections": self.thickness_rejections,
            "reached_step_limit": self.reached_step_limit,
        }


def _shift(array: np.ndarray, offset: tuple[int, int, int], fill: int) -> np.ndarray:
    result = np.full(array.shape, fill, dtype=array.dtype)
    source: list[slice] = []
    destination: list[slice] = []
    for delta, size in zip(offset, array.shape, strict=True):
        if delta < 0:
            source.append(slice(-delta, size))
            destination.append(slice(0, size + delta))
        elif delta > 0:
            source.append(slice(0, size - delta))
            destination.append(slice(delta, size))
        else:
            source.append(slice(None))
            destination.append(slice(None))
    result[tuple(destination)] = array[tuple(source)]
    return result


def _frontier_labels(labels: np.ndarray, frontier: np.ndarray) -> np.ndarray:
    minimum = np.full(labels.shape, _LABEL_SENTINEL, dtype=np.int32)
    maximum = np.zeros(labels.shape, dtype=np.int32)
    for offset in _OFFSETS_6:
        neighbour = _shift(labels, offset, 0)
        positive = neighbour > 0
        np.minimum(minimum, neighbour, out=minimum, where=positive)
        np.maximum(maximum, neighbour, out=maximum)
    unique = frontier & (minimum == maximum) & (minimum != _LABEL_SENTINEL)
    assigned = np.zeros(labels.shape, dtype=np.int32)
    assigned[unique] = minimum[unique]
    return assigned


def _remove_component_conflicts(
    labels: np.ndarray,
    assigned: np.ndarray,
) -> tuple[np.ndarray, int]:
    additions = assigned > 0
    if not bool(np.any(additions)):
        return additions, 0
    tentative = labels.copy()
    tentative[additions] = assigned[additions]
    conflict = np.zeros(labels.shape, dtype=bool)
    for offset in _OFFSETS_6:
        neighbour = _shift(tentative, offset, 0)
        conflict |= additions & (neighbour > 0) & (neighbour != assigned)
    rejected = int(np.count_nonzero(conflict))
    additions[conflict] = False
    return additions, rejected


def _remove_new_interior(
    occupied: np.ndarray,
    additions: np.ndarray,
    probability: np.ndarray,
    baseline_interior: np.ndarray,
) -> tuple[np.ndarray, int]:
    additions = additions.copy()
    rejected = 0
    while bool(np.any(additions)):
        proposed = occupied | additions
        new_interior = ndimage.binary_erosion(
            proposed,
            structure=_STRUCTURE_6,
            border_value=0,
        ) & ~baseline_interior
        if not bool(np.any(new_interior)):
            break
        candidate_probability = np.where(additions, probability, np.inf)
        minimum_candidate = ndimage.minimum_filter(
            candidate_probability,
            footprint=_STRUCTURE_6,
            mode="constant",
            cval=np.inf,
        )
        removal_request = np.where(new_interior, minimum_candidate, -np.inf)
        removal_threshold = ndimage.maximum_filter(
            removal_request,
            footprint=_STRUCTURE_6,
            mode="constant",
            cval=-np.inf,
        )
        remove = additions & (probability <= removal_threshold)
        removed = int(np.count_nonzero(remove))
        if removed == 0:
            raise RuntimeError("could not resolve a newly created interior voxel")
        additions[remove] = False
        rejected += removed
    return additions, rejected


def grow_probability_ridges(
    probability: np.ndarray,
    seed: np.ndarray,
    *,
    support_threshold: float,
    max_steps: int,
) -> RidgeGrowthResult:
    """Extend a seed mask along lower-confidence probability support.

    Growth is deliberately more conservative than ordinary hysteresis. Each
    accepted voxel must remain attached to exactly one original 6-connected
    seed component, and the final mask may not create any new one-pass
    6-neighbour erosion interior. The first constraint prevents new component
    fusions; the second lets thin sheets gain reach without acquiring a new
    layer of thickness.
    """

    values = np.asarray(probability)
    initial = np.asarray(seed, dtype=bool)
    if values.ndim != 3 or initial.shape != values.shape:
        raise ValueError("probability and seed must be matching 3-D arrays")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("probability must have a floating dtype")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("probability contains non-finite values")
    if not 0.0 <= support_threshold <= 1.0:
        raise ValueError("support_threshold must be in [0, 1]")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    probability_float = values.astype(np.float32, copy=False)
    support = probability_float >= support_threshold
    support |= initial
    occupied = initial.copy()
    labels, seed_components = ndimage.label(occupied, structure=_STRUCTURE_6)
    labels = labels.astype(np.int32, copy=False)
    baseline_interior = ndimage.binary_erosion(
        occupied,
        structure=_STRUCTURE_6,
        border_value=0,
    )
    added_per_step: list[int] = []
    conflict_rejections = 0
    thickness_rejections = 0

    for _ in range(max_steps):
        frontier = (
            ndimage.binary_dilation(occupied, structure=_STRUCTURE_6)
            & support
            & ~occupied
        )
        if not bool(np.any(frontier)):
            break
        assigned = _frontier_labels(labels, frontier)
        conflict_rejections += int(
            np.count_nonzero(frontier) - np.count_nonzero(assigned)
        )
        additions, rejected_conflicts = _remove_component_conflicts(labels, assigned)
        conflict_rejections += rejected_conflicts
        additions, rejected_thickness = _remove_new_interior(
            occupied,
            additions,
            probability_float,
            baseline_interior,
        )
        thickness_rejections += rejected_thickness
        added = int(np.count_nonzero(additions))
        if added == 0:
            break
        occupied[additions] = True
        labels[additions] = assigned[additions]
        added_per_step.append(added)

    final_labels, final_components = ndimage.label(occupied, structure=_STRUCTURE_6)
    del final_labels
    if int(final_components) != int(seed_components):
        raise RuntimeError(
            "ridge growth changed the number of established seed components"
        )
    new_interior = ndimage.binary_erosion(
        occupied,
        structure=_STRUCTURE_6,
        border_value=0,
    ) & ~baseline_interior
    if bool(np.any(new_interior)):
        raise RuntimeError("ridge growth created new erosion interior")
    return RidgeGrowthResult(
        mask=occupied,
        seed_components=int(seed_components),
        final_components=int(final_components),
        seed_positive=int(np.count_nonzero(initial)),
        final_positive=int(np.count_nonzero(occupied)),
        eligible_support=int(np.count_nonzero(support & ~initial)),
        completed_steps=len(added_per_step),
        added_per_step=tuple(added_per_step),
        component_conflict_rejections=conflict_rejections,
        thickness_rejections=thickness_rejections,
        reached_step_limit=(
            len(added_per_step) == max_steps and bool(added_per_step[-1])
        ),
    )
