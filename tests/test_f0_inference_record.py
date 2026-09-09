from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "recipes" / "f0_inference_20260909"
TRAINING_RECORD = ROOT / "recipes" / "final_c3_250k_20260907"
RECIPE_DIR = ROOT / "recipes" / "f0"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_copied_evidence_file_matches_its_inventory_hash() -> None:
    inventory = _json(RECORD / "files.json")
    assert inventory["file_count"] == len(inventory["files"]) >= 60
    for entry in inventory["files"]:
        path = RECORD / entry["relative_path"]
        assert path.is_file(), entry["relative_path"]
        assert path.stat().st_size == entry["bytes"], entry["relative_path"]
        assert _sha256(path) == entry["sha256"], entry["relative_path"]


def test_manifest_numbers_are_read_from_the_evidence() -> None:
    manifest = _json(RECORD / "manifest.json")
    assert manifest["schema"] == "socratic-method-inference-record-v1"
    assert manifest["weights"]["unchanged"] is True
    training = _json(TRAINING_RECORD / "manifest.json")
    assert manifest["weights"]["sha256"] == training["model"]["sha256"]
    assert manifest["weights"]["bytes"] == training["model"]["bytes"]
    assert manifest["inference"]["previous_operating_threshold"] == training["model"]["operating_threshold"]
    assert manifest["inference"]["operating_threshold"] == 0.30
    assert manifest["inference"]["test_time_augmentation"] == "8-way-mirror"

    table = _json(RECORD / "evidence" / "decision_table.json")
    row = next(r for r in table["rows"] if r["variant_id"] == "f0_200k_tta")
    benchmark = manifest["frozen_v14p2_benchmark"]
    assert benchmark["macro_dice_030"] == row["macro_0.30"] >= 0.70
    assert benchmark["macro_dice_035"] == row["macro_0.35"] >= 0.70
    assert benchmark["ci95_030"] == row["ci_0.30"]
    assert benchmark["paired_delta_vs_reference"]["0.35"]["delta"] == row["delta_0.35_vs_ref"]
    assert benchmark["paired_delta_vs_reference"]["0.35"]["ci95"][0] > 0
    assert benchmark["holm_over_family_primaries"]["pass"]["f0_200k_tta"] is True
    assert benchmark["breaks_0p70"] is True

    audit = _json(RECORD / "evidence" / "audits" / "f0_200k_tta" / "report.json")
    assert audit["options"]["mirror_tta"] is True
    point = min(audit["sweep"]["points"], key=lambda p: abs(float(p["threshold"]) - 0.30))
    assert point["macro_scroll_dice"] == benchmark["macro_dice_030"]
    assert benchmark["audit_report_sha256"] == _sha256(RECORD / "evidence" / "audits" / "f0_200k_tta" / "report.json")
    assert benchmark["per_patch_counts_sha256"] == audit["per_patch"]["sha256"]
    assert benchmark["reference_no_tta"]["reproduction_of_sealed_value"]["passed"] is True
    assert benchmark["reference_no_tta"]["macro_dice_035"] == training["results"]["F0_200k_035_macro_dice"]

    receipt = _json(RECORD / "evidence" / "geometry" / "54db9a59cb602ff0" / "receipt_t030.json")
    assert manifest["six_cube_pherc1447"]["gate_at_030"] == receipt["gate_at_operating"]
    assert manifest["six_cube_pherc1447"]["frontier_at_030"]["passed"] is True
    metrics = _json(RECORD / "evidence" / "all_metrics" / "all_metrics_table.json")
    shipped = next(e for e in metrics["rows"] if e["id"] == "f0_200k_tta" and abs(e["threshold"] - 0.30) < 1e-6)
    assert manifest["kaggle_metric"]["f0_200k_tta"]["0.30"]["v14p2_val"]["mean_leaderboard"] == shipped["sets"]["v14p2_val"]["mean_leaderboard"]
    assert manifest["readiness_advisory"]["composite"] is True and manifest["readiness_advisory"]["topology"] is True
    assert manifest["pending"]["c3r_replication"]["status"] == "pending"
    assert manifest["pending"]["postprocessor_requalification"]["status"] == "pending"
    prereg = manifest["source_evidence"]["prereg_document"]
    assert prereg["sha256"] == _sha256(RECORD / prereg["path"])


def test_recipe_and_current_model_point_at_the_inference_record() -> None:
    recipe = _json(RECIPE_DIR / "recipe.json")
    release = recipe["release"]
    manifest = _json(RECORD / "manifest.json")
    assert release["inference_record"] == "../f0_inference_20260909/manifest.json"
    assert release["inference_record_sha256"] == _sha256(RECORD / "manifest.json")
    assert release["release_id"] == manifest["release_id"]
    assert release["previous_release_id"] == manifest["previous_release_id"]
    assert release["operating_threshold"] == manifest["inference"]["operating_threshold"]
    assert release["training_record_operating_threshold"] == manifest["inference"]["previous_operating_threshold"]
    assert release["test_time_augmentation"] == manifest["inference"]["test_time_augmentation"]
    assert release["threshold_selection_contract"] == manifest["inference"]["threshold_selection_contract"]
    evaluation = recipe["evaluation"]
    assert evaluation["expected_selected_macro_dice_tta_at_030"] == manifest["frozen_v14p2_benchmark"]["macro_dice_030"]
    assert evaluation["expected_selected_macro_dice_tta_at_035"] == manifest["frozen_v14p2_benchmark"]["macro_dice_035"]
    current = _json(ROOT / "CURRENT_MODEL.json")
    assert current["release_id"] == manifest["release_id"]
    assert current["threshold"] == 0.30 and current["test_time_augmentation"] == "8-way-mirror"
    assert current["inference_record"] == "recipes/f0_inference_20260909/README.md"
    assert current["postprocessor_profiles_qualified_at_threshold"] == 0.35
    selection = _json(RECIPE_DIR / "selection.json")
    assert selection["threshold"] == 0.30 and selection["training_record_threshold"] == 0.35
    assert selection["inference"]["macro_dice_030"] == manifest["frozen_v14p2_benchmark"]["macro_dice_030"]
    qualification = _json(RECIPE_DIR / "release_qualification.json")
    assert qualification["selection"]["operating_threshold"] == 0.30
    assert qualification["selection"]["test_time_augmentation"] == "8-way-mirror"
    assert qualification["gates"]["operating_t030_gate_pass"] and qualification["gates"]["holm_pass"]
    assert qualification["postprocessor"]["qualified_operating_threshold"] == 0.35
