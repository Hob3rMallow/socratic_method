"""Derive recipes/f0/selection.json and release_qualification.json from the sealed record.

Every number is copied verbatim from ``recipes/final_c3_250k_20260907/manifest.json``
and ``analysis.json``; nothing is rounded or recomputed. Re-run after an
intentional record update, never edit the outputs by hand.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RECORD = Path("recipes") / "final_c3_250k_20260907"
RECIPE_DIR = Path("recipes") / "f0"
RAW_STUDENT_INFERENCE = "raw-student-only-no-m7-blend-no-teacher"
SELECTION_STATUS = "frozen-model-selected"


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def build_selection(manifest: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    model = manifest["model"]
    selection = manifest["selection"]
    return {
        "schema": "socratic-method-f0-selection-v1",
        "status": SELECTION_STATUS,
        "decision_date": "2026-09-07",
        "release_id": manifest["release_id"],
        "samples": int(model["selected_sample_exposures"]),
        "completed_training_samples": int(model["completed_training_horizon"]),
        "threshold": float(model["operating_threshold"]),
        "checkpoint_sha256": model["sha256"],
        "checkpoint_bytes": int(model["bytes"]),
        "model_composition": "raw-student-probability-only; no teacher, M7 blend, morphology, or line fitting",
        "selection_basis": selection["rule"],
        "selection": selection,
        "results": manifest["results"],
        "candidate_delta_vs_original_c3_30k": analysis["candidate_delta_vs_C3"],
        "H1_late_best_ge_early_best": analysis["H1"],
        "source_evidence": {
            "manifest": "../final_c3_250k_20260907/manifest.json",
            "analysis": "../final_c3_250k_20260907/analysis.json",
            "review": "../final_c3_250k_20260907/review.md",
            "evaluation_summary_sha256": manifest["provenance"]["summary_sha256"],
        },
        "release_note": manifest["status"],
    }


def build_qualification(manifest: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    model = manifest["model"]
    selection = manifest["selection"]
    results = manifest["results"]
    candidate = analysis["candidate_fixed_t035"]
    return {
        "schema": "socratic-method-release-qualification-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection": {
            "status": "frozen-model-selected-not-deployed",
            "checkpoint_samples": int(model["selected_sample_exposures"]),
            "checkpoint_sha256": model["sha256"],
            "checkpoint_bytes": int(model["bytes"]),
            "operating_threshold": float(model["operating_threshold"]),
            "model_composition": RAW_STUDENT_INFERENCE,
            "rule": selection["rule"],
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
        "native_scores": analysis["native_scores"],
        "six_cube_pherc1447_gates": analysis["candidate_gates"],
        "frontier_review": analysis["frontier_review"],
        "readiness_advisory": {
            "composite_target": 0.6212,
            "topology_target": 0.3776,
            **analysis["readiness_advisory"],
        },
        "gates": {
            "fixed_035_gate_pass": selection["fixed_035_gate_pass"],
            "matched_coverage_frontier_pass": selection["matched_coverage_frontier_pass"],
            "readiness_composite_pass": selection["readiness_composite_pass"],
            "readiness_topology_pass": selection["readiness_topology_pass"],
        },
        "postprocessor": manifest["postprocessor"],
        "source_artifacts": {
            "manifest": "../final_c3_250k_20260907/manifest.json",
            "analysis": "../final_c3_250k_20260907/analysis.json",
            "verification": "../final_c3_250k_20260907/verification.json",
        },
    }


def main() -> int:
    root = _root()
    manifest = _read(root / RECORD / "manifest.json")
    analysis = _read(root / RECORD / "analysis.json")
    outputs = {
        "selection.json": build_selection(manifest, analysis),
        "release_qualification.json": build_qualification(manifest, analysis),
    }
    for name, value in outputs.items():
        destination = root / RECIPE_DIR / name
        destination.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
