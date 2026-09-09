"""Derive recipes/f0/selection.json and release_qualification.json from the sealed records.

Every number is copied verbatim from ``recipes/final_c3_250k_20260907/manifest.json``
and ``analysis.json`` (the weights selection) and from
``recipes/f0_inference_20260909/manifest.json`` (the shipped inference recipe:
eight-way mirror TTA at T=0.30 on the same weights); nothing is rounded or
recomputed. Re-run after an intentional record update, never edit the outputs by hand.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RECORD = Path("recipes") / "final_c3_250k_20260907"
INFERENCE_RECORD = Path("recipes") / "f0_inference_20260909"
RECIPE_DIR = Path("recipes") / "f0"
RAW_STUDENT_INFERENCE = "raw-student-only-no-m7-blend-no-teacher"
SELECTION_STATUS = "frozen-model-selected"
DECISION_DATE = "2026-09-09"


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_same_weights(manifest: dict[str, Any], inference: dict[str, Any]) -> None:
    model = manifest["model"]
    weights = inference["weights"]
    if weights["sha256"] != model["sha256"] or int(weights["bytes"]) != int(model["bytes"]):
        raise ValueError("inference record does not describe the frozen training-record weights")
    if int(weights["selected_sample_exposures"]) != int(model["selected_sample_exposures"]):
        raise ValueError("inference record sample exposures differ from the training record")
    if float(inference["inference"]["previous_operating_threshold"]) != float(model["operating_threshold"]):
        raise ValueError("inference record previous threshold differs from the training record")


def build_selection(manifest: dict[str, Any], analysis: dict[str, Any], inference: dict[str, Any]) -> dict[str, Any]:
    _check_same_weights(manifest, inference)
    model = manifest["model"]
    selection = manifest["selection"]
    shipped = inference["inference"]
    benchmark = inference["frozen_v14p2_benchmark"]
    return {
        "schema": "socratic-method-f0-selection-v2",
        "status": SELECTION_STATUS,
        "decision_date": DECISION_DATE,
        "release_id": inference["release_id"],
        "previous_release_id": inference["previous_release_id"],
        "samples": int(model["selected_sample_exposures"]),
        "completed_training_samples": int(model["completed_training_horizon"]),
        "threshold": float(shipped["operating_threshold"]),
        "test_time_augmentation": shipped["test_time_augmentation"],
        "mirror_tta": bool(shipped["mirror_tta"]),
        "training_record_threshold": float(model["operating_threshold"]),
        "checkpoint_sha256": model["sha256"],
        "checkpoint_bytes": int(model["bytes"]),
        "model_composition": "raw-student-probability-only; no teacher, M7 blend, morphology, or line fitting",
        "selection_basis": selection["rule"],
        "inference_basis": shipped["rule"],
        "threshold_selection_contract": shipped["threshold_selection_contract"],
        "selection": selection,
        "inference": {
            "operating_threshold": float(shipped["operating_threshold"]),
            "test_time_augmentation": shipped["test_time_augmentation"],
            "mirror_tta": bool(shipped["mirror_tta"]),
            "amp_dtype": shipped["amp_dtype"],
            "previous_operating_threshold": float(shipped["previous_operating_threshold"]),
            "macro_dice_030": benchmark["macro_dice_030"],
            "ci95_030": benchmark["ci95_030"],
            "macro_dice_035": benchmark["macro_dice_035"],
            "ci95_035": benchmark["ci95_035"],
            "paired_delta_vs_reference": benchmark["paired_delta_vs_reference"],
            "holm_over_family_primaries": benchmark["holm_over_family_primaries"],
            "breaks_0p70": benchmark["breaks_0p70"],
            "verdict": inference["verdict"],
        },
        "results": manifest["results"],
        "candidate_delta_vs_original_c3_30k": analysis["candidate_delta_vs_C3"],
        "H1_late_best_ge_early_best": analysis["H1"],
        "pending": inference["pending"],
        "source_evidence": {
            "manifest": "../final_c3_250k_20260907/manifest.json",
            "analysis": "../final_c3_250k_20260907/analysis.json",
            "review": "../final_c3_250k_20260907/review.md",
            "evaluation_summary_sha256": manifest["provenance"]["summary_sha256"],
            "inference_record": "../f0_inference_20260909/manifest.json",
            "inference_decision_table": "../f0_inference_20260909/evidence/decision_table.json",
            "inference_prereg_sha256": inference["source_evidence"]["prereg_document"]["sha256"],
        },
        "release_note": inference["status"],
    }


def build_qualification(manifest: dict[str, Any], analysis: dict[str, Any], inference: dict[str, Any]) -> dict[str, Any]:
    _check_same_weights(manifest, inference)
    model = manifest["model"]
    selection = manifest["selection"]
    results = manifest["results"]
    candidate = analysis["candidate_fixed_t035"]
    shipped = inference["inference"]
    benchmark = inference["frozen_v14p2_benchmark"]
    six = inference["six_cube_pherc1447"]
    return {
        "schema": "socratic-method-release-qualification-v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection": {
            "status": "frozen-model-selected-not-deployed",
            "checkpoint_samples": int(model["selected_sample_exposures"]),
            "checkpoint_sha256": model["sha256"],
            "checkpoint_bytes": int(model["bytes"]),
            "operating_threshold": float(shipped["operating_threshold"]),
            "test_time_augmentation": shipped["test_time_augmentation"],
            "mirror_tta": bool(shipped["mirror_tta"]),
            "training_record_operating_threshold": float(model["operating_threshold"]),
            "model_composition": RAW_STUDENT_INFERENCE,
            "rule": selection["rule"],
            "inference_rule": shipped["rule"],
            "threshold_selection_contract": shipped["threshold_selection_contract"],
            "selected_checkpoint_is_not_final_endpoint": manifest["training"]["selected_checkpoint_is_not_final_endpoint"],
            "global_best_samples": selection["global_best_samples"],
            "best_early_samples": selection["best_early_samples"],
            "exact_final_samples": selection["exact_final_samples"],
            "all_25_eligible_at_035": selection["all_25_eligible_at_035"],
            "H1": selection["H1"],
            "delta_vs_original_C3_30k": selection["delta_vs_original_C3_30k"],
        },
        "frozen_v14p2_benchmark": {
            "rows": 689,
            "scrolls": ["PHerc0814", "PHerc1451"],
            "tta": True,
            "validation_manifest_sha256": benchmark["validation_manifest_sha256"],
            "audit_report_sha256": benchmark["audit_report_sha256"],
            "F0_200k_tta_030_macro_dice": benchmark["macro_dice_030"],
            "F0_200k_tta_030_ci95": benchmark["ci95_030"],
            "F0_200k_tta_030_scrolls": benchmark["scrolls_030"],
            "F0_200k_tta_035_macro_dice": benchmark["macro_dice_035"],
            "F0_200k_tta_035_ci95": benchmark["ci95_035"],
            "F0_200k_tta_035_scrolls": benchmark["scrolls_035"],
            "paired_delta_vs_reference": benchmark["paired_delta_vs_reference"],
            "holm_over_family_primaries": benchmark["holm_over_family_primaries"],
            "breaks_0p70": benchmark["breaks_0p70"],
            "reference_no_tta": benchmark["reference_no_tta"],
            "m7_comparators_macro_dice": benchmark["m7_comparators_macro_dice"],
            "training_record_no_tta": {
                "tta": False,
                "validation_manifest_sha256": results["validation_manifest_sha256"],
                "F0_200k_035_macro_dice": results["F0_200k_035_macro_dice"],
                "F0_200k_035_scrolls": {
                    scroll: metrics["dice"] for scroll, metrics in candidate["scrolls"].items()
                },
                "M7_raw_035_macro_dice": results["M7_raw_035_macro_dice"],
                "M7_raw_calibrated_020_macro_dice": results["M7_raw_calibrated_020_macro_dice"],
                "M7_8way_TTA_calibrated_020_macro_dice": results["M7_8way_TTA_calibrated_020_macro_dice"],
                "F0_250k_035_macro_dice": results["F0_250k_035_macro_dice"],
                "matched_absolute_dice_gain": results["matched_absolute_dice_gain"],
                "matched_relative_dice_gain": results["matched_relative_dice_gain"],
            },
        },
        "native_scores": {
            "student_tta": True,
            "composite": inference["kaggle_metric"]["composite"],
            "sets": inference["kaggle_metric"]["f0_200k_tta"],
            "reference_no_tta": inference["kaggle_metric"]["reference_f0_200k_no_tta"],
            "flat_m7": inference["kaggle_metric"]["flat_m7"],
            "m7_tta": inference["kaggle_metric"]["m7_tta"],
            "sealed_m7_calibrated_020": inference["kaggle_metric"]["sealed_m7_calibrated_020"],
            "training_record_no_tta": analysis["native_scores"],
        },
        "six_cube_pherc1447_gates": {
            "operating_t030": six["gate_at_030"],
            "fixed_t035": six["gate_at_035"],
            "fixed18_at_030": six["fixed18_at_030"],
            "export": six["export"],
            "training_record": analysis["candidate_gates"],
        },
        "frontier_review": {
            "operating_t030": six["frontier_at_030"],
            "fixed_t035": six["frontier_at_035"],
            "training_record_t035": analysis["frontier_review"],
        },
        "readiness_advisory": inference["readiness_advisory"],
        "gates": {
            "fixed_035_gate_pass": selection["fixed_035_gate_pass"],
            "matched_coverage_frontier_pass": selection["matched_coverage_frontier_pass"],
            "readiness_composite_pass": selection["readiness_composite_pass"],
            "readiness_topology_pass": selection["readiness_topology_pass"],
            "operating_t030_gate_pass": bool(six["gate_at_030"]["passed"]),
            "operating_t030_frontier_pass": bool(six["frontier_at_030"]["passed"]),
            "operating_t030_geometry_pass": bool(six["geometry_pass_030"]),
            "operating_t030_readiness_composite_pass": bool(inference["readiness_advisory"]["composite"]),
            "operating_t030_readiness_topology_pass": bool(inference["readiness_advisory"]["topology"]),
            "p0500p2_floor_pass": bool(inference["readiness_advisory"]["p0500p2_floor_pass"]),
            "holm_pass": bool(benchmark["holm_over_family_primaries"]["pass"]["f0_200k_tta"]),
        },
        "pending": inference["pending"],
        "postprocessor": {
            **manifest["postprocessor"],
            "qualified_operating_threshold": float(model["operating_threshold"]),
            "requalification": inference["pending"]["postprocessor_requalification"],
        },
        "known_limits": inference["known_limits"],
        "source_artifacts": {
            "manifest": "../final_c3_250k_20260907/manifest.json",
            "analysis": "../final_c3_250k_20260907/analysis.json",
            "verification": "../final_c3_250k_20260907/verification.json",
            "inference_record": "../f0_inference_20260909/manifest.json",
            "inference_files": "../f0_inference_20260909/files.json",
        },
    }


def load_records(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _read(root / RECORD / "manifest.json")
    analysis = _read(root / RECORD / "analysis.json")
    inference = _read(root / INFERENCE_RECORD / "manifest.json")
    return manifest, analysis, inference


def main() -> int:
    root = _root()
    manifest, analysis, inference = load_records(root)
    outputs = {
        "selection.json": build_selection(manifest, analysis, inference),
        "release_qualification.json": build_qualification(manifest, analysis, inference),
    }
    for name, value in outputs.items():
        destination = root / RECIPE_DIR / name
        destination.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {destination}")
    print(f"inference record manifest sha256: {_sha256(root / INFERENCE_RECORD / 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
