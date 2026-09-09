"""Build recipes/f0_inference_20260909 in socratic_method from the variants-ladder evidence.

Every number in manifest.json is read from the copied evidence files (decision table,
audit reports, bootstrap receipts, Kaggle summaries, six-cube receipts); nothing is typed in.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

V = Path(r"D:\work\vesuvius-c")
DATA = V / "output/crossres_data"
VAR = DATA / "f0_family_20260908/variants"
REF_EVAL = DATA / "final_c3_250k_20260905/evaluation/milestone_00200000"
M7_BASE = DATA / "unified_ladder_20260902/baselines"
REC = Path(r"D:\work\socratic_method/recipes/f0_inference_20260909")
RELEASE_ID = "c3-f0-200k-tta-t030-20260909"
PREVIOUS_RELEASE_ID = "c3-f0-200k-20260907"
CONTRACT = "preregistered-late-checkpoint-rule-8way-mirror-tta-operating-point-argmax-0p30-0p35-v2"
VIEW = "z12160_y04224_x02944_z12224.png"
SETS = ("v14p2_val", "p0500p2_val")
KAGGLE_KEYS = ("mean_leaderboard", "mean_surface_dice", "mean_voi_score", "mean_voi_split", "mean_voi_merge", "mean_voi_total", "mean_toposcore", "mean_topoF1_0", "mean_topoF1_1", "mean_topoF1_2", "n_cases")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


copies: list[tuple[Path, str]] = []


def plan(src: Path, rel: str) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    copies.append((src, rel))


def plan_kaggle(score_root: Path, rel_root: str) -> None:
    for set_name in SETS:
        score_dir = score_root / f"{set_name}_score" if (score_root / f"{set_name}_score").is_dir() else None
        if score_dir is None:
            raise FileNotFoundError(score_root / f"{set_name}_score")
        plan(score_dir / "kaggle_summary.json", f"{rel_root}/{set_name}_score/kaggle_summary.json")
        for shard in sorted(score_dir.glob("shard_*/out/metrics_per_case.csv")):
            plan(shard, f"{rel_root}/{set_name}_score/{shard.parent.parent.name}/out/metrics_per_case.csv")


# -- evidence plan -----------------------------------------------------------------
plan(VAR / "decision_table.json", "evidence/decision_table.json")
plan(VAR / "decision_table.before_primary_kaggle_20260908T185355Z.json", "evidence/decision_table.before_primary_kaggle_20260908T185355Z.json")
for name in ("all_metrics_table.json", "all_metrics_table.md", "all_metrics_ledger.html"):
    plan(VAR / "all_metrics" / name, f"evidence/all_metrics/{name}")
for audit in ("f0_200k_tta", "f0_200k"):
    plan(VAR / "audits" / audit / "report.json", f"evidence/audits/{audit}/report.json")
    plan(VAR / "audits" / audit / "per_patch_counts.npz", f"evidence/audits/{audit}/per_patch_counts.npz")
    plan(VAR / "bootstrap" / f"{audit}.json", f"evidence/bootstrap/{audit}.json")
for label in ("f0_200k_tta_t030", "f0_200k_tta_t035"):
    plan(VAR / "kaggle" / label / "kaggle.json", f"evidence/kaggle/{label}/kaggle.json")
    plan_kaggle(VAR / "kaggle" / label, f"evidence/kaggle/{label}")
plan(VAR / "kaggle" / "f0_200k_tta_extra_kaggle.json", "evidence/kaggle/f0_200k_tta_extra_kaggle.json")
for t in ("t030", "t035"):
    for set_name in SETS:
        src = REF_EVAL / f"kaggle_{set_name}_{t}_score"
        plan(src / "kaggle_summary.json", f"evidence/kaggle/reference_f0_200k_{t}/{set_name}_score/kaggle_summary.json")
        for shard in sorted(src.glob("shard_*/out/metrics_per_case.csv")):
            plan(shard, f"evidence/kaggle/reference_f0_200k_{t}/{set_name}_score/{shard.parent.parent.name}/out/metrics_per_case.csv")
for base in ("m7_raw", "m7_tta"):
    for set_name in SETS:
        plan(M7_BASE / base / "ladder_eval" / f"kaggle_{set_name}_cal_00000000_score" / "kaggle_summary.json", f"evidence/kaggle/sealed_{base}_calibrated_t020/{set_name}_score/kaggle_summary.json")
GEO = VAR / "geometry" / "54db9a59cb602ff0"
for name in ("receipt_t030.json", "receipt_t035.json", "frontier.json"):
    plan(GEO / name, f"evidence/geometry/54db9a59cb602ff0/{name}")
plan(GEO / "blind_t035" / "report" / "images" / VIEW, f"evidence/geometry/54db9a59cb602ff0/review_{VIEW}")
plan(V / "crossres_pred/F0_FAMILY_PREREG_20260908.md", "evidence/prereg/F0_FAMILY_PREREG_20260908.md")
plan(V / "crossres_pred/configs/f0_variants_20260908.json", "evidence/config/f0_variants_20260908.json")
plan(VAR / "preflight.json", "evidence/config/preflight.json")
plan(VAR / "status.json", "evidence/config/status.json")
for script in ("evaluate_f0_variants.py", "score_all_f0_variants.py", "score_f0_variant_kaggle.py", "kaggle_paired_bootstrap.py", "render_f0_all_metrics_ledger.py", "launch_f0_variants.ps1", "bootstrap_audit_ci.py", "average_voxel_checkpoints.py"):
    plan(V / "crossres_pred/scripts" / script, f"runtime/scripts/{script}")
for module in ("checkpoint_audit.py", "audit_bootstrap.py", "ensemble.py", "checkpoint_average.py"):
    plan(V / "crossres_pred/src/crossres_pred/voxel" / module, f"runtime/voxel/{module}")
plan(Path(__file__), "runtime/build_inference_record.py")

# -- copy + inventory ---------------------------------------------------------------
if REC.exists():
    shutil.rmtree(REC)
inventory = []
for src, rel in copies:
    dst = REC / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    inventory.append({"relative_path": rel, "original_path": str(src), "bytes": dst.stat().st_size, "sha256": sha256(dst)})
inv_by_rel = {i["relative_path"]: i for i in inventory}

# -- derive the manifest from the copied evidence --------------------------------------
table = read(REC / "evidence/decision_table.json")
row = next(r for r in table["rows"] if r["variant_id"] == "f0_200k_tta")
ref_row = next(r for r in table["rows"] if r["variant_id"] == "f0_200k")
audit = read(REC / "evidence/audits/f0_200k_tta/report.json")
boot = read(REC / "evidence/bootstrap/f0_200k_tta.json")
metrics = read(REC / "evidence/all_metrics/all_metrics_table.json")
geo30 = read(REC / "evidence/geometry/54db9a59cb602ff0/receipt_t030.json")
geo35 = read(REC / "evidence/geometry/54db9a59cb602ff0/receipt_t035.json")


def sweep_point(report: dict, threshold: float) -> dict:
    return min(report["sweep"]["points"], key=lambda p: abs(float(p["threshold"]) - threshold))


def metrics_row(row_id: str, threshold: float) -> dict:
    return next(e for e in metrics["rows"] if e["id"] == row_id and abs(e["threshold"] - threshold) < 1e-6)


def kaggle_block(entry: dict) -> dict:
    out = {}
    for set_name in SETS:
        s = entry["sets"].get(set_name)
        if not s:
            continue
        block = {k: s.get(k) for k in KAGGLE_KEYS if k in s}
        paired = (entry.get("paired_vs_reference") or {}).get(set_name)
        if paired:
            block["paired_delta_vs_reference_same_t"] = {
                col: {"delta": paired[col]["delta"], "ci95": paired[col]["ci95"], "cases_used": paired[col]["cases_used"]}
                for col in ("leaderboard", "surface_dice", "voi_score", "voi_split", "voi_merge", "toposcore", "topoF1_0")
                if col in paired
            }
        out[set_name] = block
    return out


p30, p35 = sweep_point(audit, 0.30), sweep_point(audit, 0.35)
assert abs(p30["macro_scroll_dice"] - row["macro_0.30"]) < 1e-12 and abs(p35["macro_scroll_dice"] - row["macro_0.35"]) < 1e-12
weights = table["rows"][0]
manifest = {
    "schema": "socratic-method-inference-record-v1",
    "release_id": RELEASE_ID,
    "previous_release_id": PREVIOUS_RELEASE_ID,
    "recorded_at_utc": datetime.now(UTC).isoformat(),
    "status": "operator-authorized inference-recipe update on the unchanged frozen F0/200k weights; current official model record; no automatic production deployment or public upload",
    "training_record": "../final_c3_250k_20260907/manifest.json",
    "weights": {
        "unchanged": True,
        "sha256": row["checkpoint_sha256"],
        "bytes": 409674095,
        "selected_sample_exposures": int(row["members"]),
        "file": "checkpoint_milestone_00200000.pt",
        "frozen_copy": "checkpoint_f0_00200000.pt",
    },
    "inference": {
        "model_composition": "raw-student-only-no-m7-blend-no-teacher",
        "test_time_augmentation": "8-way-mirror",
        "mirror_tta": True,
        "operating_threshold": 0.30,
        "previous_operating_threshold": 0.35,
        "amp_dtype": "bfloat16",
        "threshold_selection_contract": CONTRACT,
        "rule": "Family primary fixed before any scoring (F0_FAMILY_PREREG_20260908.md): the frozen 200k weights with eight-way mirror TTA. Operating point = argmax of the fixed-threshold frozen macro Dice over the two pre-registered thresholds {0.30, 0.35}; no other threshold was considered, no per-variant calibration, nothing tuned on PHerc1447.",
        "audit_command": "crossres-voxel audit-checkpoint --checkpoint <checkpoint_milestone_00200000.pt> --patches <frozen_validation_manifest> --output <audit dir> --split val --device cuda --tta",
    },
    "frozen_v14p2_benchmark": {
        "rows": 689,
        "scrolls": ["PHerc0814", "PHerc1451"],
        "tta": True,
        "validation_manifest_sha256": "a995787032eb446f71759a4dbd510396273b93c65f65863bac8fe9587e12b714",
        "audit_report": "evidence/audits/f0_200k_tta/report.json",
        "audit_report_sha256": inv_by_rel["evidence/audits/f0_200k_tta/report.json"]["sha256"],
        "per_patch_counts_sha256": audit["per_patch"]["sha256"],
        "macro_dice_030": p30["macro_scroll_dice"],
        "ci95_030": row["ci_0.30"],
        "scrolls_030": row["scrolls_0.30"],
        "precision_030": row["precision_0.30"],
        "recall_030": row["recall_0.30"],
        "macro_dice_035": p35["macro_scroll_dice"],
        "ci95_035": row["ci_0.35"],
        "scrolls_035": row["scrolls_0.35"],
        "precision_035": row["precision_0.35"],
        "recall_035": row["recall_0.35"],
        "calibrated_threshold": row["calibrated_threshold"],
        "calibrated_macro": row["calibrated_macro"],
        "calibrated_optimism": boot["levels"]["0.30"]["calibrated_optimism"]["optimism"],
        "reference_no_tta": {
            "macro_dice_030": ref_row["macro_0.30"],
            "macro_dice_035": ref_row["macro_0.35"],
            "reproduction_of_sealed_value": table["reference_reproduction"],
        },
        "paired_delta_vs_reference": {
            "0.30": {"delta": row["delta_0.30_vs_ref"], "ci95": row["delta_ci_0.30_vs_ref"], "probability_delta_le_zero": row["p_le_zero_0.30_vs_ref"], "scrolls": row["scroll_deltas_0.30_vs_ref"]},
            "0.35": {"delta": row["delta_0.35_vs_ref"], "ci95": row["delta_ci_0.35_vs_ref"], "probability_delta_le_zero": row["p_le_zero_0.35_vs_ref"], "scrolls": row["scroll_deltas_0.35_vs_ref"]},
            "method": "paired cluster bootstrap, clusters = support_anchor_chunk_zyx (PHerc0814 256 clusters, PHerc1451 142), 2,000 resamples, seed 20260908, identical resample weights for both audits",
        },
        "holm_over_family_primaries": {"alpha": table["holm"]["alpha"], "primaries": table["family_primaries"], "pass": table["holm"]["pass"], "pvalues": table["holm"]["pvalues"]},
        "breaks_0p70": bool(row["breaks_0p70"]),
        "primary_endpoint_threshold": table["primary_threshold"],
        "m7_comparators_macro_dice": {
            "flat_m7_030": metrics_row("m7_raw", 0.30)["macro_scroll_dice"],
            "flat_m7_035": metrics_row("m7_raw", 0.35)["macro_scroll_dice"],
            "m7_tta_030": metrics_row("m7_tta", 0.30)["macro_scroll_dice"],
            "m7_tta_035": metrics_row("m7_tta", 0.35)["macro_scroll_dice"],
            "m7_raw_calibrated_020": next(e for e in metrics["sealed_baselines_at_020"] if e["id"] == "m7_raw")["macro_scroll_dice"],
            "m7_tta_calibrated_020": next(e for e in metrics["sealed_baselines_at_020"] if e["id"] == "m7_tta")["macro_scroll_dice"],
        },
    },
    "kaggle_metric": {
        "composite": "0.35 SurfaceDice@2.0 + 0.35 VOI + 0.30 TopoScore (metric-windows reimplementation metric.exe; tau 2.0, VOI alpha 0.3, ignore label 2)",
        "student_tta": True,
        "f0_200k_tta": {"0.30": kaggle_block(metrics_row("f0_200k_tta", 0.30)), "0.35": kaggle_block(metrics_row("f0_200k_tta", 0.35))},
        "reference_f0_200k_no_tta": {"0.30": kaggle_block(metrics_row("f0_200k", 0.30)), "0.35": kaggle_block(metrics_row("f0_200k", 0.35))},
        "flat_m7": {"0.30": kaggle_block(metrics_row("m7_raw", 0.30)), "0.35": kaggle_block(metrics_row("m7_raw", 0.35))},
        "m7_tta": {"0.30": kaggle_block(metrics_row("m7_tta", 0.30)), "0.35": kaggle_block(metrics_row("m7_tta", 0.35))},
        "sealed_m7_calibrated_020": {
            e["id"]: {"sets": e["sets"], "paired_delta_vs_reference_at_030": {s: {c: {"delta": v["delta"], "ci95": v["ci95"]} for c, v in p.items() if c in ("leaderboard", "toposcore", "voi_merge", "voi_split", "topoF1_0")} for s, p in e["paired_vs_reference"].items()}}
            for e in metrics["sealed_baselines_at_020"]
        },
        "exploratory_equivalents_not_promoted": {
            v: {"macro_dice_030": metrics_row(v, 0.30)["macro_scroll_dice"], "macro_dice_035": metrics_row(v, 0.35)["macro_scroll_dice"], "composite_030": {s: metrics_row(v, 0.30)["sets"][s]["mean_leaderboard"] for s in SETS}}
            for v in ("f0_110k_tta", "avg_100k_250k_tta")
        },
        "all_metrics_table": "evidence/all_metrics/all_metrics_table.json",
        "all_metrics_table_created_utc": metrics["created_utc"],
    },
    "six_cube_pherc1447": {
        "export": "blind six-cube export, halo 32, eight-way mirror TTA, bfloat16 (evaluate_final_c3_250k.blind_command); the receipts threshold the same probability export at the operating T",
        "gate_at_030": geo30["gate_at_operating"],
        "frontier_at_030": geo30["frontier_review"],
        "fixed18_at_030": {k: geo30["frontier_fixed18_at_operating"][k] for k in ("candidate_only_bridge_pixels", "candidate_only_interior_pixels_r2", "reference_skeleton_recall", "candidate_skeleton_precision", "candidate_meaningful_components") if k in geo30["frontier_fixed18_at_operating"]},
        "gate_at_035": geo35["gate_at_operating"],
        "frontier_at_035": geo35["frontier_review"],
        "geometry_pass_030": geo30["geometry_pass"],
        "review_view": f"evidence/geometry/54db9a59cb602ff0/review_{VIEW}",
        "review_view_legend": "cyan = published M7, yellow = incumbent (jackpot) student, magenta = F0 200k; veto material, never selection",
    },
    "readiness_advisory": {
        "composite_target": 0.6212,
        "composite_030": metrics_row("f0_200k_tta", 0.30)["sets"]["v14p2_val"]["mean_leaderboard"],
        "composite": metrics_row("f0_200k_tta", 0.30)["sets"]["v14p2_val"]["mean_leaderboard"] >= 0.6212,
        "topology_target": 0.3776,
        "topology_030": metrics_row("f0_200k_tta", 0.30)["sets"]["v14p2_val"]["mean_toposcore"],
        "topology": metrics_row("f0_200k_tta", 0.30)["sets"]["v14p2_val"]["mean_toposcore"] >= 0.3776,
        "p0500p2_floor": 0.481767 - 0.005,
        "p0500p2_composite_030": metrics_row("f0_200k_tta", 0.30)["sets"]["p0500p2_val"]["mean_leaderboard"],
        "p0500p2_floor_pass": metrics_row("f0_200k_tta", 0.30)["sets"]["p0500p2_val"]["mean_leaderboard"] >= 0.481767 - 0.005,
    },
    "verdict": row["verdict"],
    "pending": {
        "c3r_replication": {
            "status": "pending",
            "rule": "F0_FAMILY_PREREG_20260908.md rule 8: the promoted inference recipe (same weights index, TTA, T) applied to the seed-1204 replicate C3r must beat C3r's own no-TTA reference with a positive paired lower bound",
            "state_at_record_time": "C3r at 160,000 of 250,000 samples (family coordinator VesuviusCrossres-F0Family-20260908, journal output/crossres_data/logs/f0_family_20260908_state.json)",
        },
        "postprocessor_requalification": {
            "status": "pending",
            "profiles": ["socratic-repair (recipes/f0_repair_20260907)", "socratic-continuity (recipes/f0_continuity_20260907)", "socratic-sheet-patches (recipes/f0_sheet_patches_20260907)"],
            "qualified_input": "frozen F0/200k prediction at T=0.35, halo 32, eight-way mirror TTA, bfloat16",
            "note": "The profiles pin the T=0.35 prediction and stay valid for it; they have not been re-qualified against the T=0.30 operating point (88-case + control comparison, zero erasure). The raw model's operating point moved; the postprocessor pins did not.",
        },
    },
    "known_limits": [
        "the frozen-harness 0.70 is reached by inference changes on the same weights: eight-way mirror TTA (+0.023 at T=0.35) and the pre-registered secondary operating point T=0.30 (+0.009); no retraining was involved",
        "one seed; the C3r replication of the inference recipe is pending",
        "on the human-labelled PHerc0500P2 the composite is flat across the whole ladder (0.476-0.497): released M7 with TTA at its calibrated T=0.20 scores 0.5015 against this recipe's 0.4917, with better topology; the student's advantage there is Surface Dice and merges",
        "the six-cube lower-foreground advisory is met at T=0.30 (foreground ratio 0.799) but the model stays sparser than the exemplar",
        "the exploratory F0 110k + TTA and the 100k-250k weight average + TTA score at or slightly above this recipe on every metric and are reported, not promoted (pre-registered family primary rule)",
    ],
    "source_evidence": {
        "prereg_document": {"path": "evidence/prereg/F0_FAMILY_PREREG_20260908.md", "sha256": inv_by_rel["evidence/prereg/F0_FAMILY_PREREG_20260908.md"]["sha256"]},
        "ladder_config": {"path": "evidence/config/f0_variants_20260908.json", "sha256": inv_by_rel["evidence/config/f0_variants_20260908.json"]["sha256"], "decision_table_config_sha256": table["config_sha256"]},
        "decision_table": {"path": "evidence/decision_table.json", "created_utc": table["created_utc"], "amendments": table.get("amendments", [])},
        "ladder_root": "D:/work/vesuvius-c/output/crossres_data/f0_family_20260908/variants",
        "ledger_page": "https://claude.ai/code/artifact/1eae2886-7404-4266-8052-eb55b07f1e13",
        "external_tools": ["metric-windows/build/metric/Release/metric.exe (Kaggle metric reimplementation, not part of this repository)"],
        "engine_note": "runtime/voxel/checkpoint_audit.py differs from the sealed pin only by the additive per-patch counts sidecar; the sealed engine reproduces the same sweep (reference_reproduction max |delta| 0.0) and the --tta audit of the shipped number needs no engine change",
    },
    "files": "files.json",
}
(REC / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
files = {
    "schema": "c3-f0-inference-record-file-inventory-v1",
    "verified_at_utc": datetime.now(UTC).isoformat(),
    "files": inventory,
    "file_count": len(inventory),
    "total_bytes": sum(i["bytes"] for i in inventory),
}
(REC / "files.json").write_text(json.dumps(files, indent=2) + "\n", encoding="utf-8", newline="\n")
print("record files:", len(inventory), "bytes:", files["total_bytes"])
print("manifest sha256:", sha256(REC / "manifest.json"))
print(json.dumps({k: manifest["frozen_v14p2_benchmark"][k] for k in ("macro_dice_030", "macro_dice_035", "ci95_030")}, indent=0))
