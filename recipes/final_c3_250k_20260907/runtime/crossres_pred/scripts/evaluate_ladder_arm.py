#!/usr/bin/env python3
"""Frozen evaluation harness for one unified-ladder arm.

Per arm: audit every milestone checkpoint on the FIXED v14p2 val manifest
(17-threshold grid, no TTA), pick the best milestone by calibrated
macro-scroll dice, render ONE blind six-cube PHerc1447 report at that
milestone's calibrated threshold (mirror TTA, probability grids saved), sweep
the probability grids against the re-anchored jackpot reference, evaluate the
anti-blob gate pack at the calibrated threshold, the original fixed 0.45, and
the operator-locked blob-review threshold 0.25, and
(for base arms) run the p-vs-q occupancy-ceiling diagnostic. Everything lands
in <run_dir>/ladder_eval/arm_summary.json.

One look per arm: the blind report renders only at the best milestone.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "crossres_pred/configs/unified_ladder_20260902.json"
SUMMARY_SCHEMA = "crossres-unified-ladder-arm-evaluation-v2"
FIXED_TWO_SIDED_DICE_THRESHOLD = 0.25
FIXED_BLOB_REVIEW_THRESHOLD = 0.25


def _say(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [ladder-eval] {message}", flush=True)


def _run(command: list[str], *, cwd: Path) -> None:
    _say("+ " + " ".join(str(part) for part in command))
    result = subprocess.run([str(part) for part in command], cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise SystemExit(f"subcommand failed with exit {result.returncode}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def milestone_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    found = []
    for path in sorted(run_dir.glob("checkpoint_milestone_*.pt")):
        found.append((int(path.stem.rsplit("_", 1)[1]), path))
    return found


def pick_best_milestone(audits: dict[int, dict[str, Any]]) -> int:
    """Highest calibrated macro; ties to the earlier milestone (cheaper)."""
    if not audits:
        raise ValueError("no milestone audits")
    return min(
        audits,
        key=lambda samples: (-float(audits[samples]["macro_scroll_dice"]), samples),
    )


def blob_score(
    sweep_row: dict[str, Any],
    morphology_rows: list[dict[str, Any]],
    gates: dict[str, Any],
) -> float:
    """Normalized worst gate excess; 0.0 means every anti-blob gate passes."""
    band_high = float(gates["foreground_ratio_band"][1])
    interior_delta = float(gates["interior_fraction_max_delta"])
    thickness_delta = float(gates["max_thickness_max_delta"])
    ratio = float(sweep_row["pherc1447"]["foreground_ratio_vs_reference"])
    score = max(0.0, (ratio - band_high) / band_high)
    for row in morphology_rows:
        student = row["student"]
        reference = row["reference"]
        interior_excess = (
            float(student["interior_fraction"])
            - float(reference["interior_fraction"])
            - interior_delta
        )
        thickness_excess = (
            float(student["max_thickness"])
            - float(reference["max_thickness"])
            - thickness_delta
        )
        score = max(
            score,
            interior_excess / interior_delta,
            thickness_excess / thickness_delta,
        )
    return max(0.0, score)


def gate_verdict(
    sweep_row: dict[str, Any],
    morphology_rows: list[dict[str, Any]],
    gates: dict[str, Any],
    *,
    soft_arm: bool,
) -> dict[str, Any]:
    band_low, band_high = (float(v) for v in gates["foreground_ratio_band"])
    interior_delta = float(gates["interior_fraction_max_delta"])
    thickness_delta = float(gates["max_thickness_max_delta"])
    aggregate = sweep_row["pherc1447"]
    ratio = float(aggregate["foreground_ratio_vs_reference"])
    interior_pass = all(
        float(row["student"]["interior_fraction"])
        <= float(row["reference"]["interior_fraction"]) + interior_delta
        for row in morphology_rows
    )
    thickness_pass = all(
        float(row["student"]["max_thickness"])
        <= float(row["reference"]["max_thickness"]) + thickness_delta
        for row in morphology_rows
    )
    fg_upper = ratio <= band_high
    fg_lower = ratio >= band_low
    return {
        "threshold": float(sweep_row["threshold"]),
        "foreground_ratio_vs_reference": ratio,
        "recall_vs_reference": float(aggregate["recall_vs_reference"]),
        "dice_vs_reference": float(aggregate["dice_vs_reference"]),
        "fg_upper_pass": fg_upper,
        "fg_lower_pass": fg_lower,
        "fg_lower_advisory": soft_arm,
        "interior_pass": interior_pass,
        "thickness_pass": thickness_pass,
        "blob_score": blob_score(sweep_row, morphology_rows, gates),
        "passed": bool(
            fg_upper and interior_pass and thickness_pass and (fg_lower or soft_arm)
        ),
    }


def c2_trigger(c1_gate: dict[str, Any], b0_gate: dict[str, Any]) -> dict[str, Any]:
    """C1 moved >= half the blob gap versus B0 without passing."""
    c1_score = float(c1_gate["blob_score"])
    b0_score = float(b0_gate["blob_score"])
    fires = (
        not c1_gate["passed"]
        and not b0_gate["passed"]
        and c1_score > 0.0
        and b0_score > 0.0
        and c1_score <= 0.5 * b0_score
    )
    return {"fires": fires, "c1_blob_score": c1_score, "b0_blob_score": b0_score}


def _nearest_sweep_row(
    sweep: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    return min(sweep, key=lambda row: abs(float(row["threshold"]) - threshold))


def standard_two_sided_dice_metrics(row: dict[str, Any]) -> dict[str, float | int]:
    """Promote ordinary symmetric Dice and its sufficient statistics.

    ``dice`` in the frozen validation sweep is the conventional
    ``2*TP / (2*TP + FP + FN)`` score pooled over all known voxels.  Macro Dice
    gives each validation scroll one vote.  Keeping both explicit prevents
    either score from being confused with the Kaggle composite/leaderboard
    metric reported elsewhere in the harness.
    """
    return {
        "threshold": float(row["threshold"]),
        "pooled_dice": float(row["dice"]),
        "macro_scroll_dice": float(row["macro_scroll_dice"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "true_positive": int(row["true_positive"]),
        "false_positive": int(row["false_positive"]),
        "false_negative": int(row["false_negative"]),
    }


def morphology_gate_thresholds(
    calibrated_threshold: float, fixed_threshold: float
) -> tuple[float, ...]:
    """Keep the original gates and always include the locked T=0.25 review."""
    return tuple(
        dict.fromkeys(
            (
                round(calibrated_threshold, 2),
                round(fixed_threshold, 2),
                FIXED_BLOB_REVIEW_THRESHOLD,
            )
        )
    )


def _ceiling_diagnostic(
    checkpoint: Path,
    val_manifest: Path,
    *,
    sample_rows: int,
    device: str,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from crossres_pred.voxel.inference import (
        _predict_probability,
        load_voxel_checkpoint,
    )
    from crossres_pred.voxel.patches import VoxelPatchDataset

    torch_device = torch.device(device)
    model, _ = load_voxel_checkpoint(checkpoint, device=torch_device)
    model.to(torch_device).eval()
    dataset = VoxelPatchDataset(val_manifest, split="val", augment=False)
    step = max(1, len(dataset.rows) // max(1, sample_rows))
    q_edges = np.linspace(0.0, 1.0, 21)
    p_edges = np.linspace(0.0, 1.0, 102)
    counts = np.zeros((20, 101), dtype=np.int64)
    used = 0
    for index in range(0, len(dataset.rows), step):
        item = dataset[index]
        image = item["image"].unsqueeze(0).to(torch_device)
        with torch.no_grad():
            probability = _predict_probability(
                model,
                image,
                amp_dtype=torch.bfloat16,
                autocast_enabled=True,
                mirror_tta=False,
            )
        p = probability.squeeze().float().cpu().numpy().ravel()
        q = item["teacher_q"].float().numpy().ravel()
        valid = item["target_valid"].numpy().ravel() > 0
        if not valid.any():
            continue
        histogram, _, _ = np.histogram2d(
            q[valid], np.clip(p[valid], 0.0, 1.0), bins=(q_edges, p_edges)
        )
        counts += histogram.astype(np.int64)
        used += 1
    band = counts[5:9, :].sum(axis=0)  # q in [0.25, 0.45)
    band_total = int(band.sum())
    p_centers = (p_edges[:-1] + p_edges[1:]) / 2.0
    cumulative = np.cumsum(band)

    def band_fraction_at_or_above(threshold: float) -> float:
        if band_total == 0:
            return 0.0
        below = int(cumulative[np.searchsorted(p_centers, threshold) - 1]) if (
            np.searchsorted(p_centers, threshold) > 0
        ) else 0
        return (band_total - below) / band_total

    def band_percentile(fraction: float) -> float:
        if band_total == 0:
            return math.nan
        rank = fraction * band_total
        index = int(np.searchsorted(cumulative, rank))
        return float(p_centers[min(index, len(p_centers) - 1)])

    return {
        "contract": "val-p-vs-q-histogram-20x101-v1",
        "rows_used": used,
        "band_q_range": [0.25, 0.45],
        "band_voxels": band_total,
        "band_p_median": band_percentile(0.5),
        "band_p_p90": band_percentile(0.9),
        "band_fraction_p_ge": {
            f"{threshold:.2f}": band_fraction_at_or_above(threshold)
            for threshold in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
        },
        "q_bin_edges": [round(float(v), 3) for v in q_edges],
        "q_bin_mean_p": [
            float((row * p_centers).sum() / row.sum()) if row.sum() else None
            for row in counts
        ],
        "q_bin_voxels": [int(row.sum()) for row in counts],
    }



def merge_metric_csvs(csv_paths: list[Path]) -> dict[str, Any]:
    """Merge per-shard metrics_per_case.csv files into exact overall means."""
    header: list[str] | None = None
    rows: list[list[str]] = []
    for csv_path in csv_paths:
        lines = [
            line.strip()
            for line in csv_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            continue
        columns = lines[0].split(",")
        if header is None:
            header = columns
        elif columns != header:
            raise ValueError(f"{csv_path}: shard CSV header mismatch")
        rows.extend(line.split(",") for line in lines[1:])
    if header is None or not rows:
        raise ValueError("no metric rows to merge")
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        for name, cell in zip(header[1:], row[1:]):
            try:
                value = float(cell)
            except ValueError:
                continue
            if math.isnan(value):
                continue
            sums[name] = sums.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1
    means = {
        f"mean_{name}": sums[name] / counts[name]
        for name in sums
        if counts.get(name)
    }
    return {"n_cases": len(rows), **means}


def _score_kaggle_sharded(
    metric_exe: Path,
    gt_dir: Path,
    pred_dir: Path,
    out_dir: Path,
    *,
    shards: int,
) -> dict[str, Any]:
    """Run metric.exe over matched gt/pred tif pairs, sharded across processes."""
    import os
    import shutil

    names = sorted(p.name for p in pred_dir.glob("*.tif"))
    if not names:
        raise SystemExit(f"{pred_dir}: no prediction tifs to score")
    missing = [n for n in names if not (gt_dir / n).is_file()]
    if missing:
        raise SystemExit(f"{gt_dir}: missing GT for {missing[:3]}...")
    shards = max(1, min(shards, len(names)))
    out_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for shard in range(shards):
        shard_dir = out_dir / f"shard_{shard}"
        shard_gt = shard_dir / "gt"
        shard_pred = shard_dir / "pred"
        shard_gt.mkdir(parents=True, exist_ok=True)
        shard_pred.mkdir(parents=True, exist_ok=True)
        for name in names[shard::shards]:
            for source, target_dir in (
                (gt_dir / name, shard_gt),
                (pred_dir / name, shard_pred),
            ):
                target = target_dir / name
                if not target.exists():
                    try:
                        os.link(source, target)
                    except OSError:
                        # hard links cannot cross volumes: since 2026-09-03 the
                        # read-limited trees live on G: behind junctions while
                        # the shared ground truth stays on D:
                        shutil.copyfile(source, target)
        processes.append(
            subprocess.Popen(
                [
                    str(metric_exe),
                    "--gt-dir", str(shard_gt),
                    "--pred-dir", str(shard_pred),
                    "--outdir", str(shard_dir / "out"),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
    for process in processes:
        if process.wait() != 0:
            raise SystemExit(f"metric.exe shard failed with exit {process.returncode}")
    summary = merge_metric_csvs(
        [out_dir / f"shard_{s}" / "out" / "metrics_per_case.csv"
         for s in range(shards)]
    )
    (out_dir / "kaggle_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _dump_kaggle_predictions(
    checkpoint: Path,
    manifest: Path,
    *,
    gt_dir: Path,
    pred_dirs: dict[str, tuple[Path, float]],
    scroll: str | None,
    device: str,
    mirror_tta: bool = False,
) -> int:
    """Write zlib-compressed GT and thresholded-prediction tifs for val rows.

    GT comes from the raw archives (target_u8 with ignore label 2 intact);
    predictions from one plain forward pass per row.
    """
    import numpy as np
    import tifffile
    import torch

    from crossres_pred.voxel.inference import (
        _predict_probability,
        load_voxel_checkpoint,
    )
    from crossres_pred.voxel.patches import VoxelPatchDataset

    torch_device = torch.device(device)
    model, _ = load_voxel_checkpoint(checkpoint, device=torch_device)
    model.to(torch_device).eval()
    dataset = VoxelPatchDataset(manifest, split="val", augment=False)
    gt_dir.mkdir(parents=True, exist_ok=True)
    for directory, _threshold in pred_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, row in enumerate(dataset.rows):
        if scroll is not None and row.scroll_id != scroll:
            continue
        name = f"{row.patch_id}.tif"
        targets = [
            (directory, threshold)
            for directory, threshold in pred_dirs.values()
            if not (directory / name).is_file()
        ]
        gt_path = gt_dir / name
        if not targets and gt_path.is_file():
            written += 1
            continue
        if not gt_path.is_file():
            archive = Path(row.path)  # PatchRecord resolves paths absolute
            with np.load(archive) as payload:
                tifffile.imwrite(
                    gt_path, payload["target_u8"], compression="zlib"
                )
        if targets:
            item = dataset[index]
            image = item["image"].unsqueeze(0).to(torch_device)
            with torch.no_grad():
                probability = _predict_probability(
                    model,
                    image,
                    amp_dtype=torch.bfloat16,
                    autocast_enabled=True,
                    mirror_tta=mirror_tta,
                )
            field = probability.squeeze().float().cpu().numpy()
            for directory, threshold in targets:
                tifffile.imwrite(
                    directory / name,
                    (field >= threshold).astype(np.uint8),
                    compression="zlib",
                )
        written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--soft-arm", action="store_true",
                        help="foreground lower bound is advisory (soft targets)")
    parser.add_argument("--diagnostic", action="store_true",
                        help="run the p-vs-q occupancy-ceiling diagnostic")
    parser.add_argument("--diagnostic-rows", type=int, default=160)
    parser.add_argument("--fixed-milestone", type=int, default=0,
                        help="score this milestone instead of best-of (A0 uses 50000)")
    parser.add_argument("--audit-tta", action="store_true",
                        help="8-way mirror TTA during the val audit (M7TTA baseline)")
    parser.add_argument("--report-tta", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="mirror TTA for the blind report (M7RAW passes --no-report-tta)")
    parser.add_argument("--skip-kaggle", action="store_true",
                        help="skip the Kaggle-metric scoring pass")
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    config = _read_json(CONFIG)
    evaluation = config["evaluation"]
    val_manifest = ROOT / evaluation["val_manifest"]
    blind_source = ROOT / evaluation["blind_source"]
    reference_grid = ROOT / evaluation["blind_reference_grid"]
    run_dir = arguments.run_dir.resolve()
    eval_dir = run_dir / "ladder_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    crossres = ROOT / "crossres_pred"

    milestones = milestone_checkpoints(run_dir)
    if arguments.fixed_milestone:
        milestones = [m for m in milestones if m[0] == arguments.fixed_milestone]
    if not milestones:
        raise SystemExit(f"{run_dir}: no milestone checkpoints to evaluate")

    audits: dict[int, dict[str, Any]] = {}
    for samples, checkpoint in milestones:
        audit_dir = eval_dir / f"audit_{samples:08d}"
        report_path = audit_dir / "report.json"
        if not report_path.is_file():
            _run(
                [
                    sys.executable, "-m", "crossres_pred.voxel.cli",
                    "audit-checkpoint",
                    "--checkpoint", checkpoint,
                    "--patches", val_manifest,
                    "--output", audit_dir,
                    "--split", "val",
                    "--device", arguments.device,
                ]
                + (["--tta"] if arguments.audit_tta else []),
                cwd=crossres,
            )
        sweep = _read_json(report_path)["sweep"]
        selected = sweep["selected"]
        fixed_t025 = _nearest_sweep_row(
            sweep["points"], FIXED_TWO_SIDED_DICE_THRESHOLD
        )
        audits[samples] = {
            "calibrated_threshold": float(selected["threshold"]),
            "pooled_dice": float(selected["dice"]),
            "macro_scroll_dice": float(selected["macro_scroll_dice"]),
            "standard_two_sided_dice": {
                "calibrated": standard_two_sided_dice_metrics(selected),
                "fixed_t025": standard_two_sided_dice_metrics(fixed_t025),
            },
            "minimum_scroll_dice_gain": float(
                selected["minimum_scroll_dice_gain_vs_baseline"]
            ),
            "scrolls": {
                scroll: {
                    "dice": float(metrics["dice"]),
                    "dice_gain_vs_baseline": float(
                        metrics.get("dice_gain_vs_baseline", math.nan)
                    ),
                }
                for scroll, metrics in selected["scrolls"].items()
            },
        }
        _say(
            f"{arguments.arm_id} @{samples:,}: macro "
            f"{audits[samples]['macro_scroll_dice']:.4f} at "
            f"T={audits[samples]['calibrated_threshold']:.2f}"
        )

    best = pick_best_milestone(audits)
    calibrated = audits[best]["calibrated_threshold"]
    best_checkpoint = dict(milestones)[best]
    _say(f"best milestone {best:,} (calibrated T={calibrated:.2f})")

    blind_dir = eval_dir / f"blind_report_{best:08d}"
    if not (blind_dir / "inference" / "probability").is_dir():
        _run(
            [
                sys.executable, "scripts/generate_checkpoint_report.py",
                "--source", blind_source,
                "--checkpoint", best_checkpoint,
                "--output", blind_dir,
                "--threshold", f"{calibrated:.2f}",
                "--device", arguments.device,
            ]
            + ([] if arguments.report_tta else ["--no-tta"]),
            cwd=crossres,
        )

    blind_t025_dir = eval_dir / f"blind_report_{best:08d}_t025"
    if abs(calibrated - FIXED_BLOB_REVIEW_THRESHOLD) < 5.0e-3:
        blind_t025_dir = blind_dir
    elif not (blind_t025_dir / "report" / "index.html").is_file():
        _run(
            [
                sys.executable,
                "scripts/render_cached_probability_report.py",
                "--source",
                blind_source,
                "--source-inference",
                blind_dir / "inference",
                "--output",
                blind_t025_dir,
                "--threshold",
                f"{FIXED_BLOB_REVIEW_THRESHOLD:.2f}",
            ],
            cwd=crossres,
        )

    # the v29 audit script's --output is the JSON FILE path (its default ends
    # in audit.json), not a directory
    fixed_threshold = float(evaluation["fixed_report_threshold"])
    morphology_thresholds = morphology_gate_thresholds(calibrated, fixed_threshold)
    joint_path = eval_dir / f"joint_audit_{best:08d}.json"
    if not joint_path.is_file():
        sweep_config = evaluation["joint_sweep"]
        _run(
            [
                sys.executable, "scripts/audit_m7_xr_v29_joint_thresholds.py",
                "--probability-grid", blind_dir / "inference" / "probability",
                "--reference-grid", reference_grid,
                "--output", joint_path,
                "--start", str(sweep_config["start"]),
                "--stop", str(sweep_config["stop"]),
                "--step", str(sweep_config["step"]),
                "--morphology-thresholds",
                *[f"{threshold:.2f}" for threshold in morphology_thresholds],
            ],
            cwd=crossres,
        )
    joint = _read_json(joint_path)
    sweep_rows = joint["sweep"]
    morphology = joint["pherc1447_morphology"]

    def morphology_rows_at(threshold: float) -> list[dict[str, Any]]:
        for entry in morphology:
            if abs(float(entry["threshold"]) - threshold) < 5e-3:
                return entry["cubes"]
        raise SystemExit(f"morphology audit lacks threshold {threshold}")

    gates_config = evaluation["gates"]
    verdicts = {}
    for label, threshold in (
        ("calibrated", calibrated),
        ("fixed", fixed_threshold),
        ("fixed_t025", FIXED_BLOB_REVIEW_THRESHOLD),
    ):
        verdicts[label] = gate_verdict(
            _nearest_sweep_row(sweep_rows, threshold),
            morphology_rows_at(threshold),
            gates_config,
            soft_arm=arguments.soft_arm,
        )

    lowest_passing = None
    for row in sorted(sweep_rows, key=lambda item: float(item["threshold"])):
        ratio = float(row["pherc1447"]["foreground_ratio_vs_reference"])
        if ratio <= float(gates_config["foreground_ratio_band"][1]):
            lowest_passing = float(row["threshold"])
            break

    kaggle_config = evaluation.get("kaggle") or {}
    kaggle_results: dict[str, Any] = {}
    if kaggle_config.get("enabled") and not arguments.skip_kaggle:
        metric_exe = ROOT / kaggle_config["metric_exe"]
        shards = int(kaggle_config.get("shards", 8))
        gt_root = ROOT / kaggle_config["gt_root"]
        for set_name, set_config in kaggle_config["sets"].items():
            set_manifest = ROOT / set_config["manifest"]
            scroll = set_config.get("scroll")
            gt_dir = gt_root / set_name
            pred_dirs = {
                "calibrated": (
                    eval_dir / f"kaggle_{set_name}_cal_{best:08d}",
                    calibrated,
                ),
                "fixed": (
                    eval_dir / f"kaggle_{set_name}_t045_{best:08d}",
                    fixed_threshold,
                ),
            }
            _say(f"kaggle {set_name}: dumping predictions (scroll={scroll})")
            count = _dump_kaggle_predictions(
                best_checkpoint,
                set_manifest,
                gt_dir=gt_dir,
                pred_dirs=pred_dirs,
                scroll=scroll,
                device=arguments.device,
                # the M7TTA baseline is defined WITH 8-way mirror TTA; arms
                # and M7RAW dump plain forwards, matching their calibration
                mirror_tta=arguments.audit_tta,
            )
            kaggle_results[set_name] = {"rows": count}
            for label, (pred_dir, threshold) in pred_dirs.items():
                score_dir = pred_dir.with_name(pred_dir.name + "_score")
                summary_path = score_dir / "kaggle_summary.json"
                if summary_path.is_file():
                    summary = _read_json(summary_path)
                else:
                    _say(
                        f"kaggle {set_name}/{label}: scoring {count} pairs at "
                        f"T={threshold:.2f} across {shards} shards"
                    )
                    summary = _score_kaggle_sharded(
                        metric_exe, gt_dir, pred_dir, score_dir, shards=shards
                    )
                summary["threshold"] = threshold
                kaggle_results[set_name][label] = summary
                _say(
                    f"kaggle {set_name}/{label}: leaderboard "
                    f"{summary.get('mean_leaderboard', float('nan')):.4f} "
                    f"(sdice {summary.get('mean_surface_dice', float('nan')):.4f} "
                    f"voi {summary.get('mean_voi_score', float('nan')):.4f} "
                    f"topo {summary.get('mean_toposcore', float('nan')):.4f})"
                )

    diagnostic = None
    if arguments.diagnostic:
        _say("running p-vs-q ceiling diagnostic")
        diagnostic = _ceiling_diagnostic(
            best_checkpoint,
            val_manifest,
            sample_rows=arguments.diagnostic_rows,
            device=arguments.device,
        )
        diagnostic["lowest_fg_upper_passing_threshold"] = lowest_passing

    summary = {
        "schema": SUMMARY_SCHEMA,
        "arm_id": arguments.arm_id,
        "run_dir": str(run_dir),
        "soft_arm": arguments.soft_arm,
        "audit_tta": arguments.audit_tta,
        "report_tta": arguments.report_tta,
        "milestone_audits": {str(k): v for k, v in sorted(audits.items())},
        "best_milestone": best,
        "best_checkpoint": str(best_checkpoint),
        "calibrated_threshold": calibrated,
        "arm_score_pooled_dice": audits[best]["pooled_dice"],
        "arm_score_macro": audits[best]["macro_scroll_dice"],
        "standard_two_sided_dice": {
            "formula": "2*TP / (2*TP + FP + FN)",
            **audits[best]["standard_two_sided_dice"],
        },
        "blind_report": str(blind_dir),
        "blind_report_t025": str(blind_t025_dir),
        "gate_verdicts": verdicts,
        "fixed_blob_review_threshold": FIXED_BLOB_REVIEW_THRESHOLD,
        "lowest_fg_upper_passing_threshold": lowest_passing,
        "kaggle": kaggle_results,
        "ceiling_diagnostic": diagnostic,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary_path = eval_dir / "arm_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _say(
        f"summary -> {summary_path} | ordinary two-sided Dice pooled "
        f"{summary['arm_score_pooled_dice']:.4f}, macro "
        f"{summary['arm_score_macro']:.4f} "
        f"@{best:,} T={calibrated:.2f} | gates calibrated="
        f"{'PASS' if verdicts['calibrated']['passed'] else 'fail'} "
        f"fixed={'PASS' if verdicts['fixed']['passed'] else 'fail'} "
        f"fixed_t025={'PASS' if verdicts['fixed_t025']['passed'] else 'fail'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
