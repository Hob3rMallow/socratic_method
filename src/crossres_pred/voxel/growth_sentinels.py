"""Two-dimensional connectivity screening for cross-resolution sentinels.

The screen is intentionally conservative.  A qualifying slice must contain a
thin teacher component which accounts for at least two meaningful M7
components, with explicit teacher geometry in the gap.  Component counts alone
are not sufficient because confetti and incomplete support can create the same
numeric signature.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

STRUCTURE_8 = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class SliceScreenOptions:
    minimum_component_voxels: int = 12
    minimum_valid_fraction: float = 0.95
    maximum_radius: float = 3.0
    maximum_interior_fraction_r2: float = 0.02
    contact_radius: int = 2
    minimum_missing_join_voxels: int = 2


@dataclass(frozen=True)
class GrowthGateOptions:
    threshold: float = 0.40
    minimum_component_voxels: int = 12
    contact_radius: int = 2
    preservation_radius: int = 1
    teacher_match_radius: int = 2
    distance_clip: float = 16.0
    maximum_symmetric_distance: float = 3.0
    minimum_preservation_fraction: float = 0.95
    minimum_growth_recall: float = 0.50
    maximum_radius: float = 3.0
    maximum_interior_fraction_r2: float = 0.02


def _clean_components(mask: np.ndarray, minimum_voxels: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=STRUCTURE_8)
    if count == 0:
        return np.zeros_like(mask, dtype=bool), np.zeros_like(labels), []
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    keep = np.flatnonzero(sizes >= minimum_voxels)
    keep = keep[keep != 0]
    clean = np.isin(labels, keep)
    clean_labels, clean_count = ndimage.label(clean, structure=STRUCTURE_8)
    clean_sizes = np.bincount(clean_labels.reshape(-1), minlength=clean_count + 1)
    return clean, clean_labels, [int(value) for value in clean_sizes[1:]]


def _shape_metrics(mask: np.ndarray) -> tuple[float, float]:
    foreground = int(np.count_nonzero(mask))
    if foreground == 0:
        return 0.0, 0.0
    radius = float(np.max(ndimage.distance_transform_edt(mask)))
    interior = ndimage.binary_erosion(mask, structure=STRUCTURE_8, iterations=2)
    interior_fraction = float(np.count_nonzero(interior)) / foreground
    return radius, interior_fraction


def _surface_distances(
    source: np.ndarray,
    target: np.ndarray,
    *,
    clip: float,
) -> np.ndarray:
    source = np.asarray(source, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not np.any(source):
        return np.empty(0, dtype=np.float32)
    if not np.any(target):
        return np.full(np.count_nonzero(source), clip, dtype=np.float32)
    target_distance = ndimage.distance_transform_edt(~target)
    return np.minimum(target_distance[source], clip).astype(np.float32, copy=False)


def _distance_summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _joined_fragment_count(
    *,
    reference_labels: np.ndarray,
    reference_ids: list[int],
    candidate_labels: np.ndarray,
    contact_radius: int,
) -> int:
    contacts: dict[int, set[int]] = {}
    for reference_id in reference_ids:
        fragment = reference_labels == reference_id
        near = ndimage.binary_dilation(
            fragment,
            structure=STRUCTURE_8,
            iterations=contact_radius,
        )
        for candidate_id in np.unique(candidate_labels[near]):
            if candidate_id != 0:
                contacts.setdefault(int(candidate_id), set()).add(reference_id)
    return max((len(values) for values in contacts.values()), default=0)


def evaluate_growth_gate_slice(
    *,
    m7: np.ndarray,
    candidate_probability: np.ndarray,
    teacher: np.ndarray,
    valid: np.ndarray,
    teacher_confidence: np.ndarray | None = None,
    options: GrowthGateOptions | None = None,
) -> dict[str, object]:
    """Evaluate learned thin growth without allowing erasure or thick bridges."""

    opts = options or GrowthGateOptions()
    arrays = [np.asarray(value) for value in (m7, candidate_probability, teacher, valid)]
    if any(value.ndim != 2 for value in arrays):
        raise ValueError("growth-gate arrays must be two-dimensional")
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("growth-gate arrays must have identical shapes")
    if not 0.0 < opts.threshold < 1.0:
        raise ValueError("threshold must lie in (0, 1)")
    if opts.distance_clip <= 0:
        raise ValueError("distance_clip must be positive")

    domain = arrays[3].astype(bool, copy=False)
    m7_mask = arrays[0].astype(bool, copy=False) & domain
    teacher_mask = arrays[2].astype(bool, copy=False) & domain
    probability = arrays[1].astype(np.float32, copy=False)
    candidate_mask = (probability >= opts.threshold) & domain

    screen = screen_connectivity_slice(
        m7=m7_mask,
        teacher=teacher_mask,
        valid=domain,
        teacher_confidence=teacher_confidence,
        options=SliceScreenOptions(
            minimum_component_voxels=opts.minimum_component_voxels,
            minimum_valid_fraction=0.95,
            maximum_radius=opts.maximum_radius,
            maximum_interior_fraction_r2=opts.maximum_interior_fraction_r2,
            contact_radius=opts.contact_radius,
        ),
    )
    if not screen["qualified"]:
        return {
            "evaluable": False,
            "passed": False,
            "rejection": f"baseline-{screen['rejection']}",
            "screen": screen,
            "options": asdict(opts),
        }

    m7_clean, m7_labels, m7_sizes = _clean_components(
        m7_mask, opts.minimum_component_voxels
    )
    candidate_clean, candidate_labels, candidate_sizes = _clean_components(
        candidate_mask, opts.minimum_component_voxels
    )
    teacher_clean, _teacher_labels, teacher_sizes = _clean_components(
        teacher_mask, opts.minimum_component_voxels
    )

    m7_to_teacher = _surface_distances(
        m7_clean, teacher_clean, clip=opts.distance_clip
    )
    teacher_to_m7 = _surface_distances(
        teacher_clean, m7_clean, clip=opts.distance_clip
    )
    candidate_to_teacher = _surface_distances(
        candidate_clean, teacher_clean, clip=opts.distance_clip
    )
    teacher_to_candidate = _surface_distances(
        teacher_clean, candidate_clean, clip=opts.distance_clip
    )
    baseline_symmetric = float(
        (np.mean(m7_to_teacher) + np.mean(teacher_to_m7)) / 2.0
    )
    candidate_symmetric = float(
        (np.mean(candidate_to_teacher) + np.mean(teacher_to_candidate)) / 2.0
    )

    teacher_near_m7 = ndimage.binary_dilation(
        m7_clean, structure=STRUCTURE_8, iterations=opts.teacher_match_radius
    )
    growth_target = teacher_clean & ~teacher_near_m7
    candidate_near = ndimage.binary_dilation(
        candidate_clean, structure=STRUCTURE_8, iterations=opts.teacher_match_radius
    )
    growth_target_voxels = int(np.count_nonzero(growth_target))
    growth_recall = (
        float(np.count_nonzero(growth_target & candidate_near)) / growth_target_voxels
        if growth_target_voxels
        else 1.0
    )

    teacher_near = ndimage.binary_dilation(
        teacher_clean, structure=STRUCTURE_8, iterations=opts.teacher_match_radius
    )
    new_candidate = candidate_clean & ~ndimage.binary_dilation(
        m7_clean, structure=STRUCTURE_8, iterations=1
    )
    new_candidate_voxels = int(np.count_nonzero(new_candidate))
    growth_precision = (
        float(np.count_nonzero(new_candidate & teacher_near)) / new_candidate_voxels
        if new_candidate_voxels
        else 1.0
    )

    correct_m7 = m7_clean & teacher_near
    preserved = ndimage.binary_dilation(
        candidate_clean,
        structure=STRUCTURE_8,
        iterations=opts.preservation_radius,
    )
    correct_m7_voxels = int(np.count_nonzero(correct_m7))
    preservation = (
        float(np.count_nonzero(correct_m7 & preserved)) / correct_m7_voxels
        if correct_m7_voxels
        else 1.0
    )

    join_event = screen["join_event"]
    reference_ids = [int(value) for value in join_event["m7_component_labels"]]
    joined_fragments = _joined_fragment_count(
        reference_labels=m7_labels,
        reference_ids=reference_ids,
        candidate_labels=candidate_labels,
        contact_radius=opts.contact_radius,
    )
    required_join = int(join_event["m7_components_joined"])
    candidate_radius, candidate_interior = _shape_metrics(candidate_clean)
    component_error = abs(len(candidate_sizes) - len(teacher_sizes))
    baseline_component_error = abs(len(m7_sizes) - len(teacher_sizes))

    growth_probabilities = probability[growth_target]
    reserve_lower = max(0.0, opts.threshold - 0.10)
    reserve_fraction = (
        float(
            np.mean(
                (growth_probabilities >= reserve_lower)
                & (growth_probabilities < opts.threshold)
            )
        )
        if growth_probabilities.size
        else 0.0
    )
    gates = {
        "join_closed": joined_fragments >= required_join,
        "component_count_matches_teacher": component_error == 0,
        "component_error_not_worse": component_error <= baseline_component_error,
        "distance_improves": candidate_symmetric <= baseline_symmetric,
        "distance_within_limit": (
            candidate_symmetric <= opts.maximum_symmetric_distance
        ),
        "preserves_correct_m7": preservation >= opts.minimum_preservation_fraction,
        "recovers_teacher_growth": growth_recall >= opts.minimum_growth_recall,
        "radius_safe": candidate_radius <= opts.maximum_radius,
        "interior_safe": (
            candidate_interior <= opts.maximum_interior_fraction_r2
        ),
    }
    passed = all(gates.values())
    return {
        "evaluable": True,
        "passed": passed,
        "rejection": None if passed else [name for name, value in gates.items() if not value],
        "options": asdict(opts),
        "screen": screen,
        "gates": gates,
        "components": {
            "m7": len(m7_sizes),
            "candidate": len(candidate_sizes),
            "teacher": len(teacher_sizes),
            "baseline_error": baseline_component_error,
            "candidate_error": component_error,
            "joined_fragments": joined_fragments,
            "required_join": required_join,
        },
        "distance": {
            "clip": opts.distance_clip,
            "baseline_symmetric_mean": baseline_symmetric,
            "candidate_symmetric_mean": candidate_symmetric,
            "improvement": baseline_symmetric - candidate_symmetric,
            "m7_to_teacher": _distance_summary(m7_to_teacher),
            "teacher_to_m7": _distance_summary(teacher_to_m7),
            "candidate_to_teacher": _distance_summary(candidate_to_teacher),
            "teacher_to_candidate": _distance_summary(teacher_to_candidate),
        },
        "growth": {
            "target_voxels": growth_target_voxels,
            "recall": growth_recall,
            "new_candidate_voxels": new_candidate_voxels,
            "precision": growth_precision,
            "target_probability_mean": (
                float(np.mean(growth_probabilities))
                if growth_probabilities.size
                else 1.0
            ),
            "subthreshold_reserve_fraction": reserve_fraction,
        },
        "preservation": {
            "correct_m7_voxels": correct_m7_voxels,
            "fraction": preservation,
        },
        "anti_blob": {
            "candidate_max_radius": candidate_radius,
            "candidate_interior_fraction_r2": candidate_interior,
        },
    }


def screen_connectivity_slice(
    *,
    m7: np.ndarray,
    teacher: np.ndarray,
    valid: np.ndarray,
    teacher_confidence: np.ndarray | None = None,
    options: SliceScreenOptions | None = None,
) -> dict[str, object]:
    """Judge one slice for the thin-teacher-joins-fragmented-M7 signature."""

    opts = options or SliceScreenOptions()
    arrays = [np.asarray(value) for value in (m7, teacher, valid)]
    if any(value.ndim != 2 for value in arrays):
        raise ValueError("m7, teacher, and valid must be two-dimensional")
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("m7, teacher, and valid must have identical shapes")
    if opts.minimum_component_voxels < 1:
        raise ValueError("minimum_component_voxels must be positive")

    domain = arrays[2].astype(bool, copy=False)
    valid_fraction = float(np.mean(domain))
    base: dict[str, object] = {
        "qualified": False,
        "valid_fraction": valid_fraction,
        "options": asdict(opts),
    }
    if valid_fraction < opts.minimum_valid_fraction:
        return base | {"rejection": "incomplete-teacher-support"}

    m7_clean, m7_labels, m7_sizes = _clean_components(
        arrays[0].astype(bool, copy=False) & domain,
        opts.minimum_component_voxels,
    )
    teacher_clean, teacher_labels, teacher_sizes = _clean_components(
        arrays[1].astype(bool, copy=False) & domain,
        opts.minimum_component_voxels,
    )
    m7_count = len(m7_sizes)
    teacher_count = len(teacher_sizes)
    base |= {
        "m7_components": m7_count,
        "teacher_components": teacher_count,
        "m7_component_sizes": m7_sizes,
        "teacher_component_sizes": teacher_sizes,
        "m7_foreground_voxels": int(np.count_nonzero(m7_clean)),
        "teacher_foreground_voxels": int(np.count_nonzero(teacher_clean)),
    }
    if teacher_count == 0 or m7_count == 0:
        return base | {"rejection": "missing-meaningful-foreground"}
    if m7_count <= teacher_count:
        return base | {"rejection": "m7-not-more-fragmented"}

    m7_radius, m7_interior = _shape_metrics(m7_clean)
    teacher_radius, teacher_interior = _shape_metrics(teacher_clean)
    base |= {
        "m7_max_radius": m7_radius,
        "teacher_max_radius": teacher_radius,
        "m7_interior_fraction_r2": m7_interior,
        "teacher_interior_fraction_r2": teacher_interior,
    }
    if m7_radius > opts.maximum_radius or teacher_radius > opts.maximum_radius:
        return base | {"rejection": "radius-blob-gate"}
    if (
        m7_interior > opts.maximum_interior_fraction_r2
        or teacher_interior > opts.maximum_interior_fraction_r2
    ):
        return base | {"rejection": "interior-blob-gate"}

    m7_near = ndimage.binary_dilation(m7_clean, structure=STRUCTURE_8, iterations=1)
    best: dict[str, object] | None = None
    for teacher_label in range(1, teacher_count + 1):
        component = teacher_labels == teacher_label
        contact = ndimage.binary_dilation(
            component,
            structure=STRUCTURE_8,
            iterations=opts.contact_radius,
        )
        m7_ids = np.unique(m7_labels[contact])
        m7_ids = m7_ids[m7_ids != 0]
        if m7_ids.size < 2:
            continue
        missing_join = component & ~m7_near
        missing_voxels = int(np.count_nonzero(missing_join))
        if missing_voxels < opts.minimum_missing_join_voxels:
            continue
        confidence = 1.0
        if teacher_confidence is not None:
            values = np.asarray(teacher_confidence, dtype=np.float32)
            if values.shape != component.shape:
                raise ValueError("teacher_confidence must match the masks")
            confidence = float(np.mean(values[component]))
        event = {
            "teacher_component_label": teacher_label,
            "teacher_component_voxels": int(np.count_nonzero(component)),
            "m7_components_joined": int(m7_ids.size),
            "m7_component_labels": [int(value) for value in m7_ids],
            "missing_join_voxels": missing_voxels,
            "teacher_confidence_mean": confidence,
        }
        if best is None or (
            int(event["m7_components_joined"]),
            int(event["missing_join_voxels"]),
            float(event["teacher_confidence_mean"]),
        ) > (
            int(best["m7_components_joined"]),
            int(best["missing_join_voxels"]),
            float(best["teacher_confidence_mean"]),
        ):
            best = event
    if best is None:
        return base | {"rejection": "no-teacher-supported-join"}

    component_excess = m7_count - teacher_count
    merge_excess = int(best["m7_components_joined"]) - 1
    score = (
        1000.0 * merge_excess
        + 100.0 * component_excess
        + 4.0 * int(best["missing_join_voxels"])
        + 20.0 * float(best["teacher_confidence_mean"])
        + 10.0 * valid_fraction
        - 5.0 * (m7_interior + teacher_interior)
    )
    return base | {
        "qualified": True,
        "rejection": None,
        "component_excess": component_excess,
        "merge_excess": merge_excess,
        "score": score,
        "join_event": best,
    }


def cleaned_slice_masks(
    m7: np.ndarray,
    teacher: np.ndarray,
    valid: np.ndarray,
    *,
    minimum_component_voxels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact meaningful masks used by the screen for rendering."""

    domain = np.asarray(valid, dtype=bool)
    m7_clean, _labels, _sizes = _clean_components(
        np.asarray(m7, dtype=bool) & domain,
        minimum_component_voxels,
    )
    teacher_clean, _labels, _sizes = _clean_components(
        np.asarray(teacher, dtype=bool) & domain,
        minimum_component_voxels,
    )
    return m7_clean, teacher_clean
