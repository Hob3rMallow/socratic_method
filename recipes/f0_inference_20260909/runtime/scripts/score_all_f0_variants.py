"""Every metric for flat M7 and every variants-ladder variant at T=.30 and .35.

For each row (released M7 with and without TTA, all declared and conditional
ladder variants including the diagnostic ensembles) and each operating
threshold: the frozen macro Dice (existing audits), the Kaggle composite with
its SurfaceDice / VOI split-merge / per-dimension topology F1 on v14p2 (689)
and PHerc0500P2 (320), a paired cluster bootstrap against the reference
(F0@200k, no TTA, same threshold), and for every single checkpoint the
six-cube instance-separation statistics (bridge pixels, components, skeleton
recall/precision, gates, frontier). Reuses the ladder's own dump/score/geometry
code and directory layout; resumable; GPU dumps are serial, Kaggle scoring runs
in a thread pool. Never acquires gpu.lock (set CUDA_VISIBLE_DEVICES before
launching).
"""

from __future__ import annotations

import argparse
import html
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import evaluate_ladder_arm as frozen
import numpy as np
from evaluate_f0_variants import CheckpointSpec, Runner, resolve_checkpoint
from kaggle_paired_bootstrap import clusters_from_manifest, paired, read_cases
from run_final_c3_250k import CROSSRES, DATA, ROOT, atomic_json, now, read_json, sha256

SCHEMA = "crossres-f0-all-metrics-table-v1"
JOURNAL_SCHEMA = "crossres-f0-all-metrics-journal-v1"
THRESHOLDS = (0.30, 0.35)
M7_RAW = DATA / "unified_ladder_20260902/baselines/m7_raw/checkpoint_milestone_00000000.pt"
SEALED_MILESTONE = DATA / "final_c3_250k_20260905/evaluation/milestone_00200000"
SEALED_BASELINES = DATA / "unified_ladder_20260902/baselines"  # released M7 raw / +TTA at their calibrated T (0.20)
SEALED_BASELINE_PAIRED_T = 0.30  # operating-point-paired deltas for the calibrated baselines use the reference at this T
RELEASE = DATA / "releases/c3-f0-200k-20260907"
REFERENCE_ID = "f0_200k"
PRIMARY_ID = "f0_200k_tta"
KAGGLE_COLUMNS = ("mean_leaderboard", "mean_surface_dice", "mean_voi_score", "mean_voi_split", "mean_voi_merge", "mean_voi_total", "mean_toposcore", "mean_topoF1_0", "mean_topoF1_1", "mean_topoF1_2")


# ----------------------------------------------------------------------------
# Rows
# ----------------------------------------------------------------------------


def conditional_variants_from_table(config: dict[str, Any], table: dict[str, Any]) -> list[dict[str, Any]]:
    declared = {v["id"]: v for v in config["variants"]}
    result = []
    for row in table["rows"]:
        vid = row["variant_id"]
        if vid in declared:
            continue
        base = vid.removesuffix("_tta")
        if base not in declared:
            raise ValueError(f"table row {vid} has no declared base variant")
        result.append({**declared[base], "id": vid, "tta": bool(row["tta"]), "role": row.get("role", "exploratory"), "conditional_from": base})
    return result


def rows_to_score(runner: Runner, table: dict[str, Any]) -> list[dict[str, Any]]:
    m7_spec = CheckpointSpec("milestone", M7_RAW, (M7_RAW,), ())
    rows: list[dict[str, Any]] = [
        {"id": "m7_raw", "family": "M7", "role": "baseline", "tta": False, "kind": "milestone", "members": "released M7 (fresh, student container)", "diagnostic_only": False, "spec": m7_spec},
        {"id": "m7_tta", "family": "M7", "role": "baseline", "tta": True, "kind": "milestone", "members": "released M7 (fresh, student container)", "diagnostic_only": False, "spec": m7_spec},
    ]
    variants = list(runner.config["variants"]) + conditional_variants_from_table(runner.config, table)
    for variant in variants:
        spec = resolve_checkpoint(runner.config, variant, runner.averages)
        rows.append({
            "id": variant["id"], "family": variant["family"], "role": variant.get("role", "exploratory"),
            "tta": bool(variant["tta"]), "kind": variant["checkpoint"]["kind"], "members": variant["checkpoint"]["samples"],
            "diagnostic_only": bool(variant.get("diagnostic_only", False)), "spec": spec, "variant": variant,
        })
    rows.sort(key=lambda r: (r["id"] == PRIMARY_ID, r["tta"], r["kind"] == "ensemble"))
    return rows


def label(row_id: str, threshold: float) -> str:
    return f"{row_id}_t{round(threshold * 100):03d}"


def score_dir_for(runner: Runner, row_id: str, set_name: str, threshold: float) -> Path:
    if row_id == REFERENCE_ID:
        return SEALED_MILESTONE / f"kaggle_{set_name}_t{round(threshold * 100):03d}_score"
    return runner.kaggle_dir / label(row_id, threshold) / f"{set_name}_score"


def pred_dir_for(runner: Runner, row_id: str, set_name: str, threshold: float) -> Path:
    if row_id == REFERENCE_ID:
        return SEALED_MILESTONE / f"kaggle_{set_name}_t{round(threshold * 100):03d}"
    return runner.kaggle_dir / label(row_id, threshold) / set_name


# ----------------------------------------------------------------------------
# Dumps (ensemble-aware) and scoring
# ----------------------------------------------------------------------------


def ensemble_dump(members: list[Path], manifest: Path, *, gt_dir: Path, pred_dir: Path, threshold: float, scroll: str | None, mirror_tta: bool) -> int:
    import tifffile
    import torch
    from crossres_pred.voxel.ensemble import ProbabilityEnsemble
    from crossres_pred.voxel.inference import (
        _predict_probability,
        load_voxel_checkpoint,
    )
    from crossres_pred.voxel.patches import VoxelPatchDataset

    device = torch.device("cuda")
    models = [load_voxel_checkpoint(m, device=device)[0].eval() for m in members]
    model = ProbabilityEnsemble(models).eval()
    dataset = VoxelPatchDataset(manifest, split="val", augment=False)
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, row in enumerate(dataset.rows):
        if scroll is not None and row.scroll_id != scroll:
            continue
        name = f"{row.patch_id}.tif"
        target = pred_dir / name
        if not (gt_dir / name).is_file():
            with np.load(Path(row.path)) as payload:
                tifffile.imwrite(gt_dir / name, payload["target_u8"], compression="zlib")
        if not target.is_file():
            item = dataset[index]
            image = item["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                probability = _predict_probability(model, image, amp_dtype=torch.bfloat16, autocast_enabled=True, mirror_tta=mirror_tta)
            field = probability.squeeze().float().cpu().numpy()
            tifffile.imwrite(target, (field >= threshold).astype(np.uint8), compression="zlib")
        written += 1
    return written


class AllMetrics:
    def __init__(self, config: Path, *, shards: int, concurrent: int) -> None:
        self.runner = Runner(config, None)
        self.runner.log_path = ROOT / "output/crossres_data/logs/f0_all_metrics_20260908.log"
        self.runner.config["kaggle"]["metric_shards"] = int(shards)
        self.root = self.runner.root / "all_metrics"
        self.journal_path = ROOT / "output/crossres_data/logs/f0_all_metrics_20260908_state.json"
        self.concurrent = int(concurrent)
        self.table = read_json(self.runner.root / "decision_table.json")
        self.rows = rows_to_score(self.runner, self.table)
        self.kaggle = self.runner.config["kaggle"]

    def say(self, message: str) -> None:
        self.runner.say(f"[all-metrics] {message}")

    def mark(self, stage: str, status: str, **extra: Any) -> None:
        state = read_json(self.journal_path) if self.journal_path.exists() else {"schema": JOURNAL_SCHEMA, "stages": {}}
        state["stages"][stage] = {"status": status, "at_utc": now(), **extra}
        atomic_json(self.journal_path, state)
        self.say(f"{stage}: {status}")

    def wait_for_primary_extra_scoring(self) -> None:
        receipt = self.runner.kaggle_dir / f"{PRIMARY_ID}_extra_kaggle.json"
        started = time.monotonic()
        while not receipt.exists() and any((self.runner.kaggle_dir / label(PRIMARY_ID, t)).exists() for t in THRESHOLDS):
            if time.monotonic() - started > 6 * 3600:
                raise RuntimeError("the primary's extra Kaggle scoring did not finish within 6 h")
            self.say("waiting for the primary's extra Kaggle scoring to finish before touching its directories")
            time.sleep(120)

    def dump(self, row: dict[str, Any], threshold: float) -> dict[str, Path]:
        spec: CheckpointSpec = row["spec"]
        pred_dirs: dict[str, Path] = {}
        for set_name, settings in self.kaggle["sets"].items():
            pred_dir = pred_dir_for(self.runner, row["id"], set_name, threshold)
            pred_dirs[set_name] = pred_dir
            expected = int(self.kaggle["expected_rows"][set_name])
            if pred_dir.exists() and len(list(pred_dir.glob("*.tif"))) == expected:
                continue
            gt_dir = ROOT / self.kaggle["gt_root"] / set_name
            self.say(f"{row['id']}: dump {set_name} at T={threshold:.2f} ({'TTA' if row['tta'] else 'no TTA'}, {spec.kind})")
            if spec.kind == "ensemble":
                count = ensemble_dump(list(spec.members), ROOT / settings["manifest"], gt_dir=gt_dir, pred_dir=pred_dir, threshold=threshold, scroll=settings["scroll"], mirror_tta=row["tta"])
            else:
                count = frozen._dump_kaggle_predictions(spec.primary, ROOT / settings["manifest"], gt_dir=gt_dir, pred_dirs={f"{threshold:.2f}": (pred_dir, threshold)}, scroll=settings["scroll"], device="cuda", mirror_tta=row["tta"])
            if count != expected:
                raise ValueError(f"{row['id']} {set_name}: {count} rows written, expected {expected}")
        return pred_dirs

    def score(self, row: dict[str, Any], threshold: float, pred_dirs: dict[str, Path]) -> dict[str, Any]:
        variant = {"id": row["id"], "tta": row["tta"]}
        return self.runner.kaggle_score(variant, threshold, pred_dirs)

    def geometry(self, row: dict[str, Any], threshold: float) -> dict[str, Any] | None:
        spec: CheckpointSpec = row["spec"]
        if spec.kind == "ensemble":
            return None
        return self.runner.geometry({"id": row["id"], "tta": row["tta"]}, spec, threshold)

    # -- table ---------------------------------------------------------------------

    def macro(self, row: dict[str, Any], threshold: float) -> float | None:
        if row["id"] in ("m7_raw", "m7_tta"):
            report = read_json(RELEASE / "provenance" / f"{row['id']}_audit.json")
        else:
            path = self.runner.audits / row["id"] / "report.json"
            if not path.exists():
                return None
            report = read_json(path)
        point = min(report["sweep"]["points"], key=lambda p: abs(float(p["threshold"]) - threshold))
        return float(point["macro_scroll_dice"]) if abs(float(point["threshold"]) - threshold) < 1e-4 else None

    def compile(self) -> dict[str, Any]:
        v14p2_manifest = ROOT / self.kaggle["sets"]["v14p2_val"]["manifest"]
        p0500_manifest = ROOT / self.kaggle["sets"]["p0500p2_val"]["manifest"]
        entries = []
        for row in self.rows:
            for threshold in THRESHOLDS:
                entry: dict[str, Any] = {
                    "id": row["id"], "family": row["family"], "role": row["role"], "tta": row["tta"], "kind": row["kind"],
                    "members": row["members"], "diagnostic_only": row["diagnostic_only"], "threshold": threshold,
                    "macro_scroll_dice": self.macro(row, threshold), "sets": {}, "paired_vs_reference": {}, "six_cube": None,
                }
                for set_name in self.kaggle["sets"]:
                    summary_path = score_dir_for(self.runner, row["id"], set_name, threshold) / "kaggle_summary.json"
                    if not summary_path.exists():
                        continue
                    summary = read_json(summary_path)
                    entry["sets"][set_name] = {k: summary.get(k) for k in KAGGLE_COLUMNS}
                    if row["id"] != REFERENCE_ID:
                        reference_dir = score_dir_for(self.runner, REFERENCE_ID, set_name, threshold)
                        if (reference_dir / "kaggle_summary.json").exists():
                            candidate = read_cases(summary_path.parent)
                            reference = read_cases(reference_dir)
                            ids = sorted(set(candidate) & set(reference))
                            manifest = v14p2_manifest if set_name == "v14p2_val" else p0500_manifest
                            clusters = clusters_from_manifest(manifest, ids)
                            entry["paired_vs_reference"][set_name] = paired(candidate, reference, clusters, ids, replicates=2000, seed=20260908)["columns"]
                spec: CheckpointSpec = row["spec"]
                if spec.kind != "ensemble":
                    receipt = self.runner.geometry_dir / sha256(spec.primary)[:16] / f"receipt_t{round(threshold * 100):03d}.json"
                    if receipt.exists():
                        geometry = read_json(receipt)
                        fixed18 = geometry["frontier_fixed18_at_operating"]
                        gate = geometry["gate_at_operating"]
                        entry["six_cube"] = {
                            "bridge_pixels": fixed18["candidate_only_bridge_pixels"],
                            "interior_pixels_r2": fixed18.get("candidate_only_interior_pixels_r2"),
                            "skeleton_recall": fixed18["reference_skeleton_recall"],
                            "skeleton_precision": fixed18.get("candidate_skeleton_precision"),
                            "components": fixed18["candidate_meaningful_components"],
                            "foreground_ratio_vs_reference": gate["foreground_ratio_vs_reference"],
                            "recall_vs_reference": gate["recall_vs_reference"],
                            "gates_pass": gate["passed"], "interior_pass": gate["interior_pass"], "thickness_pass": gate["thickness_pass"],
                            "frontier_pass": geometry["frontier_review"].get("passed"),
                        }
                entries.append(entry)
        sealed = read_json(RELEASE / "analysis.json")
        extra = []
        for row_id, key in (("m7_raw", "m7_raw_calibrated"), ("m7_tta", "m7_tta_calibrated")):
            item: dict[str, Any] = {
                "id": row_id, "threshold": 0.20, "note": "sealed calibrated baseline", "sets": {}, "paired_vs_reference": {},
                "paired_reference_threshold": SEALED_BASELINE_PAIRED_T, "macro_scroll_dice": sealed[key]["macro"],
            }
            for set_name in self.kaggle["sets"]:
                score_dir = SEALED_BASELINES / row_id / "ladder_eval" / f"kaggle_{set_name}_cal_00000000_score"
                summary_path = score_dir / "kaggle_summary.json"
                if not summary_path.exists():
                    if set_name == "v14p2_val":
                        item["sets"][set_name] = {k: sealed[key]["kaggle"].get(k) for k in KAGGLE_COLUMNS}
                    continue
                item["sets"][set_name] = {k: read_json(summary_path).get(k) for k in KAGGLE_COLUMNS}
                reference_dir = score_dir_for(self.runner, REFERENCE_ID, set_name, SEALED_BASELINE_PAIRED_T)
                if (reference_dir / "kaggle_summary.json").exists():
                    candidate = read_cases(score_dir)
                    reference = read_cases(reference_dir)
                    ids = sorted(set(candidate) & set(reference))
                    manifest = v14p2_manifest if set_name == "v14p2_val" else p0500_manifest
                    item["paired_vs_reference"][set_name] = paired(candidate, reference, clusters_from_manifest(manifest, ids), ids, replicates=2000, seed=20260908)["columns"]
                    item["paired_vs_reference"][set_name]["_cases"] = {"candidate": len(candidate), "reference": len(reference), "shared": len(ids)}
            extra.append(item)
        return {"schema": SCHEMA, "created_utc": now(), "thresholds": list(THRESHOLDS), "reference": REFERENCE_ID, "rows": entries, "sealed_baselines_at_020": extra}

    def write_markdown(self, table: dict[str, Any]) -> str:
        def fmt(value: Any, digits: int = 4) -> str:
            return "" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))

        def delta(entry: dict[str, Any], set_name: str, column: str) -> str:
            p = entry["paired_vs_reference"].get(set_name, {}).get(column)
            return "" if not p else f"{p['delta']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]"

        lines = ["# Flat M7 and every F0 variant, all metrics (2026-09-08)", "", f"Reference for paired deltas: {REFERENCE_ID} at the same threshold. Frozen macro Dice = per-scroll pooled Dice averaged over PHerc0814/PHerc1451 (no TTA unless the row says TTA). Kaggle = 0.35 SurfaceDice + 0.35 VOI + 0.30 TopoScore (metric.exe). Six-cube = blind PHerc1447 views against the jackpot reference; bridges are inter-wrap merger pixels.", ""]
        for set_name, title in (("v14p2_val", "v14p2 (689 cases, registered fine-teacher GT)"), ("p0500p2_val", "PHerc0500P2 (320 cases, human GT, absolute only)")):
            lines += [f"## Kaggle metrics on {title}", "", "| model | TTA | T | macro Dice | composite | SurfaceDice | VOI score | VOI split | VOI merge | TopoScore | F1 b0 | F1 b1 | F1 b2 | d composite | d TopoScore | d VOI merge | d VOI split |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|"]
            for entry in table["rows"]:
                s = entry["sets"].get(set_name)
                if not s:
                    continue
                lines.append("| " + " | ".join([
                    entry["id"] + (" (diagnostic ensemble)" if entry["diagnostic_only"] else ""), "yes" if entry["tta"] else "no", f"{entry['threshold']:.2f}",
                    fmt(entry["macro_scroll_dice"]), fmt(s["mean_leaderboard"]), fmt(s["mean_surface_dice"]), fmt(s["mean_voi_score"]), fmt(s["mean_voi_split"], 3), fmt(s["mean_voi_merge"], 3),
                    fmt(s["mean_toposcore"]), fmt(s["mean_topoF1_0"], 3), fmt(s["mean_topoF1_1"], 3), fmt(s["mean_topoF1_2"], 3),
                    delta(entry, set_name, "leaderboard"), delta(entry, set_name, "toposcore"), delta(entry, set_name, "voi_merge"), delta(entry, set_name, "voi_split"),
                ]) + " |")
            for extra in table["sealed_baselines_at_020"]:
                s = extra["sets"].get(set_name)
                if s:
                    lines.append("| " + " | ".join([
                        extra["id"] + " (sealed, calibrated; deltas vs reference at 0.30)", "yes" if extra["id"].endswith("tta") else "no", "0.20",
                        fmt(extra["macro_scroll_dice"]), fmt(s["mean_leaderboard"]), fmt(s["mean_surface_dice"]), fmt(s["mean_voi_score"]), fmt(s["mean_voi_split"], 3), fmt(s["mean_voi_merge"], 3),
                        fmt(s["mean_toposcore"]), fmt(s["mean_topoF1_0"], 3), fmt(s["mean_topoF1_1"], 3), fmt(s["mean_topoF1_2"], 3),
                        delta(extra, set_name, "leaderboard"), delta(extra, set_name, "toposcore"), delta(extra, set_name, "voi_merge"), delta(extra, set_name, "voi_split"),
                    ]) + " |")
            lines.append("")
        lines += ["## Six-cube instance separation (blind PHerc1447, TTA export at halo 32)", "", "| checkpoint | T | bridge px | interior px | skeleton recall | skeleton precision | components | fg ratio | recall vs ref | gates | frontier |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
        seen = set()
        for entry in table["rows"]:
            six = entry["six_cube"]
            base = entry["id"].removesuffix("_tta")
            key = ("m7_raw" if base == "m7" else base, entry["threshold"])  # m7_tta shares the m7_raw checkpoint
            if not six or key in seen:
                continue
            seen.add(key)
            lines.append("| " + " | ".join([key[0], f"{entry['threshold']:.2f}", str(six["bridge_pixels"]), str(six["interior_pixels_r2"]), fmt(six["skeleton_recall"]), fmt(six["skeleton_precision"]), str(six["components"]), fmt(six["foreground_ratio_vs_reference"], 3), fmt(six["recall_vs_reference"], 3), "pass" if six["gates_pass"] else "FAIL", "pass" if six["frontier_pass"] else ("veto" if six["frontier_pass"] is False else "unresolved")]) + " |")
        return "\n".join(lines) + "\n"

    def run(self, *, skip_geometry: bool = False) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.mark("start", "running", rows=[r["id"] for r in self.rows], thresholds=list(THRESHOLDS))
        futures: list[tuple[str, float, Future]] = []
        with ThreadPoolExecutor(max_workers=self.concurrent) as pool:
            for row in self.rows:
                if row["id"] == PRIMARY_ID:
                    self.wait_for_primary_extra_scoring()
                for threshold in THRESHOLDS:
                    if row["id"] == REFERENCE_ID:
                        continue  # sealed scores exist at .30/.35/.45
                    summary_ok = all((score_dir_for(self.runner, row["id"], s, threshold) / "kaggle_summary.json").exists() for s in self.kaggle["sets"])
                    if summary_ok:
                        continue
                    pred_dirs = self.dump(row, threshold)
                    futures.append((row["id"], threshold, pool.submit(self.score, row, threshold, pred_dirs)))
                    self.mark(f"dump:{label(row['id'], threshold)}", "complete")
            if not skip_geometry:
                for row in self.rows:
                    for threshold in THRESHOLDS:
                        if row["spec"].kind == "ensemble":
                            continue
                        receipt = self.runner.geometry_dir / sha256(row["spec"].primary)[:16] / f"receipt_t{round(threshold * 100):03d}.json"
                        if receipt.exists():
                            continue
                        self.geometry(row, threshold)
                        self.mark(f"geometry:{label(row['id'], threshold)}", "complete")
            for row_id, threshold, future in futures:
                future.result()
                self.mark(f"score:{label(row_id, threshold)}", "complete")
        table = self.compile()
        atomic_json(self.root / "all_metrics_table.json", table)
        markdown = self.write_markdown(table)
        (self.root / "all_metrics_table.md").write_text(markdown, encoding="utf-8", newline="\n")
        (self.root / "all_metrics_table.html").write_text("<!doctype html><meta charset='utf-8'><title>F0 variants, all metrics</title><style>body{font:14px system-ui;margin:2rem}pre{white-space:pre-wrap}</style><pre>" + html.escape(markdown) + "</pre>", encoding="utf-8")
        self.mark("table", "complete", path=str(self.root / "all_metrics_table.md"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CROSSRES / "configs/f0_variants_20260908.json")
    parser.add_argument("--shards", type=int, default=6)
    parser.add_argument("--concurrent", type=int, default=4)
    parser.add_argument("--compile-only", action="store_true", help="only rebuild the table from existing receipts")
    parser.add_argument("--skip-geometry", action="store_true")
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    job = AllMetrics(args.config, shards=args.shards, concurrent=args.concurrent)
    if args.compile_only:
        table = job.compile()
        job.root.mkdir(parents=True, exist_ok=True)
        atomic_json(job.root / "all_metrics_table.json", table)
        (job.root / "all_metrics_table.md").write_text(job.write_markdown(table), encoding="utf-8", newline="\n")
        print(job.write_markdown(table))
        return
    job.run(skip_geometry=args.skip_geometry)


if __name__ == "__main__":
    main()
