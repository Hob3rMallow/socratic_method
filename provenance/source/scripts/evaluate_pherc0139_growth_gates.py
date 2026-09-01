#!/usr/bin/env python3
"""Evaluate a student checkpoint on the locked PHerc0139 growth gates.

The sixteen human-approved 64x64 atlas slices remain evaluation-only.  The
reference is the exact q-or-crest mask and joint validity domain rendered in
the review panels.  One global student threshold is selected.  A slice passes
only when meaningful connected-component counts match, symmetric boundary
distance is bounded, and the student satisfies anti-blob limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import generate_pherc0139_heldout_failure_report as heldout
import numpy as np
from crossres_pred.voxel.growth_sentinels import (
    GrowthGateOptions,
    SliceScreenOptions,
    evaluate_growth_gate_slice,
    screen_connectivity_slice,
)
from crossres_pred.voxel.io import read_crop
from crossres_pred.voxel.resources import configure_cpu_budget
from scipy import ndimage

SCHEMA = "crossres-pherc0139-growth-gate-evaluation-v2"
STRUCTURE_8 = np.ones((3, 3), dtype=bool)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _clean(
    mask: np.ndarray, minimum_voxels: int
) -> tuple[np.ndarray, int, list[int]]:
    labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=STRUCTURE_8)
    if count == 0:
        return np.zeros_like(mask, dtype=bool), 0, []
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    keep = np.flatnonzero(sizes >= minimum_voxels)
    keep = keep[keep != 0]
    clean = np.isin(labels, keep)
    labels, count = ndimage.label(clean, structure=STRUCTURE_8)
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    return clean, int(count), [int(value) for value in sizes[1:]]


def _shape(mask: np.ndarray) -> tuple[float, float]:
    foreground = int(np.count_nonzero(mask))
    if foreground == 0:
        return 0.0, 0.0
    radius = float(np.max(ndimage.distance_transform_edt(mask)))
    interior = ndimage.binary_erosion(mask, structure=STRUCTURE_8, iterations=2)
    return radius, float(np.count_nonzero(interior)) / foreground


def _boundary_distance(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left_boundary = left & ~ndimage.binary_erosion(left, structure=STRUCTURE_8)
    right_boundary = right & ~ndimage.binary_erosion(right, structure=STRUCTURE_8)
    if not np.any(left_boundary) or not np.any(right_boundary):
        return float("inf"), float("inf")
    left_to_right = ndimage.distance_transform_edt(~right_boundary)[left_boundary]
    right_to_left = ndimage.distance_transform_edt(~left_boundary)[right_boundary]
    distances = np.concatenate((left_to_right, right_to_left))
    return float(np.mean(distances)), float(np.percentile(distances, 95.0))


def slice_metrics(
    probability: np.ndarray,
    teacher: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    threshold: float,
    minimum_component: int,
    maximum_radius: float,
    maximum_interior: float,
    maximum_assd: float,
    maximum_boundary_p95: float,
) -> dict[str, Any]:
    domain = (
        np.ones_like(teacher, dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool)
    )
    if probability.shape != teacher.shape or domain.shape != teacher.shape:
        raise ValueError("probability, teacher, and valid must have identical shapes")
    student, student_components, student_sizes = _clean(
        (probability >= threshold) & domain, minimum_component
    )
    reference, teacher_components, teacher_sizes = _clean(
        np.asarray(teacher, dtype=bool) & domain, minimum_component
    )
    intersection = int(np.count_nonzero(student & reference))
    denominator = int(np.count_nonzero(student)) + int(np.count_nonzero(reference))
    dice = 2.0 * intersection / denominator if denominator else 1.0
    radius, interior = _shape(student)
    assd, boundary_p95 = _boundary_distance(student, reference)
    component_match = student_components == teacher_components
    anti_blob = radius <= maximum_radius and interior <= maximum_interior
    distance_pass = assd <= maximum_assd and boundary_p95 <= maximum_boundary_p95
    return {
        "passed": bool(component_match and anti_blob and distance_pass),
        "component_match": component_match,
        "student_components": student_components,
        "teacher_components": teacher_components,
        "student_component_sizes": student_sizes,
        "teacher_component_sizes": teacher_sizes,
        "dice": dice,
        "symmetric_boundary_mean_vox": assd,
        "symmetric_boundary_p95_vox": boundary_p95,
        "student_max_radius_vox": radius,
        "student_interior_fraction_r2": interior,
        "anti_blob_passed": anti_blob,
        "distance_passed": distance_pass,
        "student_foreground_voxels": int(np.count_nonzero(student)),
        "teacher_foreground_voxels": int(np.count_nonzero(reference)),
    }


def compose_candidate_probability(
    probability: np.ndarray,
    m7: np.ndarray,
    *,
    preserve_m7_base: bool,
    m7_base_blend: float = 0.0,
) -> np.ndarray:
    """Optionally make the published binary M7 mask an immutable base.

    The student may add foreground probability, but it cannot delete M7
    foreground.  This composition uses no teacher information.
    """

    candidate = np.asarray(probability, dtype=np.float32)
    base = np.asarray(m7, dtype=bool)
    if candidate.shape != base.shape:
        raise ValueError("student probability and M7 base shapes differ")
    blend = 1.0 if preserve_m7_base else float(m7_base_blend)
    if not np.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("M7 base blend must lie in [0, 1]")
    if blend == 0.0:
        return candidate
    # Raise only published-M7 voxels toward one. Outside M7, the trained
    # student's probabilities and therefore its learned additions are intact.
    return candidate + blend * base.astype(np.float32) * (1.0 - candidate)


def compose_growth_residual_probability(
    student_probability: np.ndarray,
    m7_probability: np.ndarray,
    m7_mask: np.ndarray,
) -> np.ndarray:
    """Keep binary M7 and expose only positive learned probability gain.

    Outside the immutable M7 mask, the score is the fraction of foreground
    headroom gained over released M7.  This keeps the locked global threshold
    range meaningful while discarding the student's duplicated full surface.
    """

    student = np.asarray(student_probability, dtype=np.float32)
    reference = np.asarray(m7_probability, dtype=np.float32)
    base = np.asarray(m7_mask, dtype=bool)
    if student.shape != reference.shape or student.shape != base.shape:
        raise ValueError("student, M7 probability, and M7 mask shapes differ")
    if (
        not np.all(np.isfinite(student))
        or not np.all(np.isfinite(reference))
        or np.any(student < 0.0)
        or np.any(student > 1.0)
        or np.any(reference < 0.0)
        or np.any(reference > 1.0)
    ):
        raise ValueError("student and M7 probabilities must be finite in [0, 1]")
    headroom = np.maximum(1.0 - reference, 1.0e-6)
    growth = np.clip((student - reference) / headroom, 0.0, 1.0)
    return np.where(base, 1.0, growth).astype(np.float32, copy=False)


def _assert_reviewed_metrics(
    row: dict[str, Any], reproduced: dict[str, Any]
) -> None:
    expected = row["metrics"]
    exact_keys = (
        "qualified",
        "rejection",
        "m7_components",
        "teacher_components",
        "m7_component_sizes",
        "teacher_component_sizes",
        "m7_foreground_voxels",
        "teacher_foreground_voxels",
        "component_excess",
        "merge_excess",
    )
    for key in exact_keys:
        if reproduced.get(key) != expected.get(key):
            raise ValueError(
                f"{row['candidate_id']}: reviewed metric {key!r} does not reproduce"
            )
    float_keys = (
        "valid_fraction",
        "m7_max_radius",
        "teacher_max_radius",
        "m7_interior_fraction_r2",
        "teacher_interior_fraction_r2",
        "score",
    )
    for key in float_keys:
        if not np.isclose(
            float(reproduced[key]), float(expected[key]), rtol=0.0, atol=1e-9
        ):
            raise ValueError(
                f"{row['candidate_id']}: reviewed metric {key!r} does not reproduce"
            )
    expected_join = expected["join_event"]
    reproduced_join = reproduced["join_event"]
    for key in (
        "teacher_component_label",
        "teacher_component_voxels",
        "m7_components_joined",
        "m7_component_labels",
        "missing_join_voxels",
    ):
        if reproduced_join.get(key) != expected_join.get(key):
            raise ValueError(
                f"{row['candidate_id']}: reviewed join metric {key!r} does not reproduce"
            )
    if not np.isclose(
        float(reproduced_join["teacher_confidence_mean"]),
        float(expected_join["teacher_confidence_mean"]),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            f"{row['candidate_id']}: reviewed join confidence does not reproduce"
        )


def run(args: argparse.Namespace) -> Path:
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    gate_path = args.gates.expanduser().resolve()
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    selected = gate.get("selected_slices")
    if (
        gate.get("slice_count") != 16
        or not isinstance(selected, list)
        or len(selected) != 16
    ):
        raise ValueError("gate input must contain exactly 16 locked slices")

    catalog = args.atlas_catalog.expanduser().resolve()
    source = heldout._source_from_catalog(catalog, args.record_id)
    arrays = heldout._open_arrays(source)
    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_sha256 = _sha256(checkpoint)

    contexts: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in selected:
        origin = tuple(int(value) for value in row["context_origin_zyx"])
        contexts.setdefault(
            origin,
            {
                "candidate_id": "context_" + "_".join(map(str, origin)),
                "context_origin_zyx": origin,
            },
        )
    cache_output = output / "student_cache" / checkpoint_sha256[:16]
    cache_provenance = cache_output / "provenance.json"
    cache_identity = {
        "schema": "crossres-growth-gate-student-cache-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "atlas_catalog": str(catalog),
        "atlas_catalog_sha256": _sha256(catalog),
        "record_id": args.record_id,
        "mirror_tta": not args.no_tta,
        "core_shape_zyx": list(heldout.CORE_SHAPE),
        "contexts": [
            {
                "candidate_id": row["candidate_id"],
                "context_origin_zyx": list(row["context_origin_zyx"]),
            }
            for row in contexts.values()
        ],
    }
    if cache_provenance.is_file():
        prior_cache = json.loads(cache_provenance.read_text(encoding="utf-8"))
        if prior_cache.get("identity") != cache_identity:
            raise ValueError("student probability cache identity differs")
    elif cache_output.exists() and any(cache_output.iterdir()):
        raise ValueError("student probability cache lacks provenance")
    _write_json(
        cache_provenance,
        {"identity": cache_identity, "state": "materializing"},
    )
    heldout._infer_candidates(
        candidates=list(contexts.values()),
        arrays=arrays,
        checkpoint=checkpoint,
        output=cache_output,
        device_name=args.device,
        mirror_tta=not args.no_tta,
    )
    for context in contexts.values():
        values = np.load(
            heldout._cache_path(cache_output, context["candidate_id"]),
            mmap_mode="r",
        )
        if values.shape != heldout.CORE_SHAPE or values.dtype != np.float16:
            raise ValueError(f"invalid student cache for {context['candidate_id']}")
    _write_json(cache_provenance, {"identity": cache_identity, "state": "complete"})

    residual_reference: dict[str, Any] | None = None
    reference_argument = getattr(args, "growth_residual_reference_evaluation", None)
    if reference_argument is not None:
        if bool(getattr(args, "preserve_m7_base", False)) or float(
            getattr(args, "m7_base_blend", 0.0)
        ) != 0.0:
            raise ValueError(
                "growth-residual composition and M7 retention blending are "
                "mutually exclusive"
            )
        reference_path = Path(reference_argument).expanduser().resolve()
        reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
        reference_provenance = Path(
            str(reference_payload.get("student_cache_provenance", ""))
        ).resolve()
        if (
            reference_payload.get("schema") != SCHEMA
            or reference_payload.get("stage") != "complete"
            or Path(str(reference_payload.get("gate_input", ""))).resolve()
            != gate_path
            or reference_payload.get("prediction_composition", {}).get(
                "contract"
            )
            != "student-probability-only-v1"
            or not reference_provenance.is_file()
            or _sha256(reference_provenance)
            != str(reference_payload.get("student_cache_provenance_sha256"))
        ):
            raise ValueError("growth-residual M7 reference is not a plain gate cache")
        reference_cache = json.loads(reference_provenance.read_text(encoding="utf-8"))
        if reference_cache.get("state") != "complete":
            raise ValueError("growth-residual M7 probability cache is incomplete")
        reference_contexts = reference_cache.get("identity", {}).get("contexts")
        if reference_contexts != cache_identity["contexts"]:
            raise ValueError("growth-residual M7 reference contexts differ")
        residual_reference = {
            "evaluation": str(reference_path),
            "evaluation_sha256": _sha256(reference_path),
            "checkpoint": str(reference_payload["checkpoint"]),
            "checkpoint_sha256": str(reference_payload["checkpoint_sha256"]),
            "cache": str(reference_provenance.parent),
            "cache_provenance": str(reference_provenance),
            "cache_provenance_sha256": _sha256(reference_provenance),
        }

    evidence_rows: list[dict[str, Any]] = []
    for row in selected:
        context_origin = tuple(int(value) for value in row["context_origin_zyx"])
        context_id = contexts[context_origin]["candidate_id"]
        probability = np.load(
            heldout._cache_path(cache_output, context_id)
        ).astype(np.float32)
        axis = int(row["axis_index"])
        core_index = int(row["core_local_index"])
        tile_index = int(row["tile_local_index"])
        core_origin = tuple(int(value) for value in row["core_origin_zyx"])
        tile_origin = tuple(int(value) for value in row["tile_origin_zyx"])
        expected_core_index = tile_origin[axis] + tile_index - core_origin[axis]
        if core_index != expected_core_index:
            raise ValueError(f"{row['candidate_id']}: inconsistent slice coordinates")
        probability_core_slice = np.take(probability, core_index, axis=axis)
        plane_axes = [value for value in range(3) if value != axis]
        plane_starts = [
            tile_origin[value] - core_origin[value] for value in plane_axes
        ]
        probability_slice = probability_core_slice[
            plane_starts[0] : plane_starts[0] + heldout.TILE_SHAPE[plane_axes[0]],
            plane_starts[1] : plane_starts[1] + heldout.TILE_SHAPE[plane_axes[1]],
        ]
        if probability_slice.shape != tuple(
            heldout.TILE_SHAPE[value] for value in plane_axes
        ):
            raise ValueError(f"{row['candidate_id']}: student tile is incomplete")
        reference_probability_slice: np.ndarray | None = None
        if residual_reference is not None:
            reference_probability = np.load(
                heldout._cache_path(
                    Path(str(residual_reference["cache"])), context_id
                )
            ).astype(np.float32)
            reference_core_slice = np.take(
                reference_probability, core_index, axis=axis
            )
            reference_probability_slice = reference_core_slice[
                plane_starts[0] : plane_starts[0]
                + heldout.TILE_SHAPE[plane_axes[0]],
                plane_starts[1] : plane_starts[1]
                + heldout.TILE_SHAPE[plane_axes[1]],
            ]
            if reference_probability_slice.shape != probability_slice.shape:
                raise ValueError(
                    f"{row['candidate_id']}: M7 probability tile is incomplete"
                )

        q = read_crop(arrays["q"], tile_origin, heldout.TILE_SHAPE)
        valid = read_crop(arrays["valid"], tile_origin, heldout.TILE_SHAPE) > 0
        crest = read_crop(arrays["crest"], tile_origin, heldout.TILE_SHAPE) > 0
        crest_valid = (
            read_crop(arrays["crest_valid"], tile_origin, heldout.TILE_SHAPE) > 0
        )
        m7 = heldout._m7_mask(
            read_crop(arrays["m7"], tile_origin, heldout.TILE_SHAPE), arrays
        )
        teacher_volume = (
            q.astype(np.float32) / 255.0 >= args.teacher_threshold
        ) | crest
        domain_volume = valid & crest_valid
        teacher = np.take(teacher_volume, tile_index, axis=axis)
        domain = np.take(domain_volume, tile_index, axis=axis)
        m7_slice = np.take(m7, tile_index, axis=axis)
        if reference_probability_slice is not None:
            probability_slice = compose_growth_residual_probability(
                probability_slice,
                reference_probability_slice,
                m7_slice,
            )
        else:
            probability_slice = compose_candidate_probability(
                probability_slice,
                m7_slice,
                preserve_m7_base=bool(getattr(args, "preserve_m7_base", False)),
                m7_base_blend=float(getattr(args, "m7_base_blend", 0.0)),
            )
        confidence = np.take(
            np.maximum(q.astype(np.float32) / 255.0, crest.astype(np.float32)),
            tile_index,
            axis=axis,
        )
        locked_options = SliceScreenOptions(**row["metrics"]["options"])
        if (
            float(row["teacher_threshold"]) != args.teacher_threshold
            or locked_options.minimum_component_voxels != args.minimum_component
            or locked_options.maximum_radius != args.maximum_radius
            or locked_options.maximum_interior_fraction_r2
            != args.maximum_interior_fraction
        ):
            raise ValueError("evaluator limits differ from the human-reviewed gate")
        reproduced = screen_connectivity_slice(
            m7=m7_slice,
            teacher=teacher,
            valid=domain,
            teacher_confidence=confidence,
            options=locked_options,
        )
        _assert_reviewed_metrics(row, reproduced)
        evidence_rows.append(
            {
                "row": row,
                "probability": probability_slice,
                "teacher": teacher,
                "valid": domain,
                "m7": m7_slice,
                "teacher_confidence": confidence,
                "projection": {
                    "contract": "locked-reviewed-coarse-atlas-tile-v1",
                    "tile_origin_zyx": list(tile_origin),
                    "tile_shape_zyx": list(heldout.TILE_SHAPE),
                    "teacher_rule": "(q/255 >= threshold OR crest) AND valid AND crest_valid",
                    "reviewed_metrics_reproduced": True,
                },
            }
        )

    threshold_summaries: list[dict[str, Any]] = []
    for threshold in args.thresholds:
        metrics: list[dict[str, Any]] = []
        for row in evidence_rows:
            geometry = slice_metrics(
                    row["probability"],
                    row["teacher"],
                    valid=row["valid"],
                    threshold=threshold,
                    minimum_component=args.minimum_component,
                    maximum_radius=args.maximum_radius,
                    maximum_interior=args.maximum_interior_fraction,
                    maximum_assd=args.maximum_assd,
                    maximum_boundary_p95=args.maximum_boundary_p95,
                )
            growth = evaluate_growth_gate_slice(
                m7=row["m7"],
                candidate_probability=row["probability"],
                teacher=row["teacher"],
                valid=row["valid"],
                teacher_confidence=row["teacher_confidence"],
                options=GrowthGateOptions(
                    threshold=threshold,
                    minimum_component_voxels=args.minimum_component,
                    maximum_symmetric_distance=args.maximum_assd,
                    minimum_preservation_fraction=args.minimum_preservation,
                    minimum_growth_recall=args.minimum_growth_recall,
                    maximum_radius=args.maximum_radius,
                    maximum_interior_fraction_r2=args.maximum_interior_fraction,
                ),
            )
            geometry["learned_growth_passed"] = bool(growth["passed"])
            geometry["growth_gate"] = growth
            geometry["passed"] = bool(
                geometry["passed"] and geometry["learned_growth_passed"]
            )
            metrics.append(geometry)
        finite_assd = [
            item["symmetric_boundary_mean_vox"]
            for item in metrics
            if np.isfinite(item["symmetric_boundary_mean_vox"])
        ]
        threshold_summaries.append(
            {
                "threshold": threshold,
                "slices_passed": sum(item["passed"] for item in metrics),
                "component_matches": sum(item["component_match"] for item in metrics),
                "learned_growth_passes": sum(
                    item["learned_growth_passed"] for item in metrics
                ),
                "anti_blob_passes": sum(item["anti_blob_passed"] for item in metrics),
                "distance_passes": sum(item["distance_passed"] for item in metrics),
                "mean_assd_vox": (
                    float(np.mean(finite_assd)) if finite_assd else float("inf")
                ),
                "mean_dice": float(np.mean([item["dice"] for item in metrics])),
                "metrics": metrics,
            }
        )
    best = max(
        threshold_summaries,
        key=lambda item: (
            item["slices_passed"],
            item["component_matches"],
            item["learned_growth_passes"],
            item["anti_blob_passes"],
            item["distance_passes"],
            -item["mean_assd_vox"],
            item["mean_dice"],
            item["threshold"],
        ),
    )
    rendered = [
        evidence_row["row"]
        | {"projection": evidence_row["projection"], "metrics": metrics}
        for evidence_row, metrics in zip(evidence_rows, best["metrics"])
    ]
    summary = {
        "passed": best["slices_passed"] == 16,
        "slice_count": 16,
        "slices_passed": best["slices_passed"],
        "component_matches": best["component_matches"],
        "learned_growth_passes": best["learned_growth_passes"],
        "anti_blob_passes": best["anti_blob_passes"],
        "distance_passes": best["distance_passes"],
        "mean_assd_vox": best["mean_assd_vox"],
        "mean_dice": best["mean_dice"],
    }
    atlas_state = Path(str(source["atlas_state"])).resolve()
    medial_state = Path(str(source["medial_state"])).resolve()
    is_negative_control = args.evaluation_role == "negative-control"
    payload = {
        "schema": SCHEMA,
        "stage": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "research_only": True,
        "evaluation_role": args.evaluation_role,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "student_cache_provenance": str(cache_provenance),
        "student_cache_provenance_sha256": _sha256(cache_provenance),
        "prediction_composition": (
            {
                "contract": (
                    "published-binary-m7-plus-positive-relative-"
                    "probability-gain-v1"
                ),
                "preserve_m7_base": True,
                "uses_teacher_at_inference": False,
                "growth_score": "clip((student-m7_probability)/(1-m7_probability),0,1)",
                "m7_probability_reference": residual_reference,
            }
            if residual_reference is not None
            else {
                "contract": (
                    "published-binary-m7-immutable-base-student-additions-v1"
                    if bool(getattr(args, "preserve_m7_base", False))
                    else (
                        "published-m7-voxel-retention-blend-v1"
                        if float(getattr(args, "m7_base_blend", 0.0)) > 0.0
                        else "student-probability-only-v1"
                    )
                ),
                "preserve_m7_base": bool(
                    getattr(args, "preserve_m7_base", False)
                ),
                "m7_base_blend": (
                    1.0
                    if bool(getattr(args, "preserve_m7_base", False))
                    else float(getattr(args, "m7_base_blend", 0.0))
                ),
                "uses_teacher_at_inference": False,
            }
        ),
        "gate_input": str(gate_path),
        "gate_input_sha256": _sha256(gate_path),
        "heldout_split": gate["heldout_split"],
        "locked_teacher": {
            "contract": "human-reviewed-coarse-atlas-q-or-crest-v1",
            "atlas_catalog": str(catalog),
            "atlas_catalog_sha256": _sha256(catalog),
            "record_id": args.record_id,
            "atlas_state": str(atlas_state),
            "atlas_state_sha256": _sha256(atlas_state),
            "medial_state": str(medial_state),
            "medial_state_sha256": _sha256(medial_state),
            "reviewed_metrics_reproduced": True,
            "slice_shape": [64, 64],
        },
        "negative_control": {
            "applicable": is_negative_control,
            "contract": "pre-refinement-m7-xr-must-fail-locked-gate-v1",
            "satisfied": (not summary["passed"]) if is_negative_control else None,
        },
        "selected_threshold": best["threshold"],
        "threshold_sweep": [
            {key: value for key, value in item.items() if key != "metrics"}
            for item in threshold_summaries
        ],
        "gate_limits": {
            "minimum_component_voxels": args.minimum_component,
            "maximum_student_radius_vox": args.maximum_radius,
            "maximum_student_interior_fraction_r2": args.maximum_interior_fraction,
            "maximum_symmetric_boundary_mean_vox": args.maximum_assd,
            "maximum_symmetric_boundary_p95_vox": args.maximum_boundary_p95,
            "minimum_preservation_fraction": args.minimum_preservation,
            "minimum_growth_recall": args.minimum_growth_recall,
            "teacher_threshold": args.teacher_threshold,
            "one_global_student_threshold": True,
        },
        "summary": summary,
        "slices": rendered,
    }
    evaluation_path = output / "evaluation.json"
    _write_json(evaluation_path, payload)
    _write_json(
        output / "status.json",
        {
            "schema": SCHEMA,
            "stage": "complete",
            "summary": summary,
            "evaluation_role": args.evaluation_role,
            "negative_control_satisfied": (
                (not summary["passed"]) if is_negative_control else None
            ),
            "human_report_created": False,
            "report": None,
        },
    )
    return evaluation_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--atlas-catalog", type=Path, required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evaluation-role",
        choices=("negative-control", "candidate"),
        default="candidate",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=tuple(round(value / 100.0, 2) for value in range(10, 61)),
    )
    parser.add_argument("--teacher-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-component", type=int, default=12)
    parser.add_argument("--maximum-radius", type=float, default=3.0)
    parser.add_argument("--maximum-interior-fraction", type=float, default=0.02)
    parser.add_argument("--maximum-assd", type=float, default=1.0)
    parser.add_argument("--maximum-boundary-p95", type=float, default=2.0)
    parser.add_argument("--minimum-preservation", type=float, default=0.95)
    parser.add_argument("--minimum-growth-recall", type=float, default=0.50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preserve-m7-base", action="store_true")
    parser.add_argument("--m7-base-blend", type=float, default=0.0)
    parser.add_argument(
        "--growth-residual-reference-evaluation",
        type=Path,
        help=(
            "compose binary M7 with positive relative probability gain over "
            "this plain released-M7 gate evaluation"
        ),
    )
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--max-cpu-threads", type=int, default=16)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.thresholds or any(
        not 0.0 <= value <= 1.0 for value in args.thresholds
    ):
        raise ValueError("thresholds must lie in [0, 1]")
    if not 1 <= args.max_cpu_threads <= 16:
        raise ValueError("max-cpu-threads must lie in [1, 16]")
    configure_cpu_budget(args.max_cpu_threads)
    report = run(args)
    print(report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
