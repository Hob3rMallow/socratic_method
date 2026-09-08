"""Post-training F0 evaluation: all milestones, fixed .35 eligibility, then review."""

from __future__ import annotations

import argparse
import html
import math
import os
import subprocess
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

import audit_m7_xr_v29_joint_thresholds as morphology
import evaluate_ladder_arm as frozen
import measure_blob_frontier as frontier
import numpy as np
from render_release_milestone_panels import FIXED_SLICES
from run_final_c3_250k import (
    ARM_DIR,
    CONFIG,
    CROSSRES,
    ORIGINAL,
    ROOT,
    RUN_ROOT,
    SNAPSHOTS,
    Coordinator,
    atomic_json,
    now,
    read_json,
    sha256,
)

EVALUATION = RUN_ROOT / "evaluation"
BASELINES = {
    "C3": (
        ORIGINAL / "ladder_eval/blind_report_00030000/inference",
        "e55f44a150936bd172fa7649cd55de36aac25b1d44df54bf35f1655efa9cc273",
    ),
    "human25": (
        ROOT
        / "output/crossres_data/native_c3_human25_20260905/arms/H_c3_human25/ladder_eval/panels/milestone_00015000/pherc1447/inference",
        "f33ebc2f418869a2abbe8a8f6996553a3f5136a3f44741c54d693389752bcf72",
    ),
}


def say(message: str) -> None:
    print(f"{now()} [F0-eval] {message}", flush=True)


def run(command: list[str | Path]) -> None:
    subprocess.run(
        [str(x) for x in command],
        cwd=str(CROSSRES),
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def point_at(points: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    row = min(points, key=lambda r: abs(float(r["threshold"]) - threshold))
    if abs(float(row["threshold"]) - threshold) > 1e-4:
        raise ValueError(f"frozen threshold sweep lacks exact T={threshold}")
    return row


def validate_cached_receipt(
    receipt: dict[str, Any], digest: str, checkpoint_sha: str
) -> None:
    if (
        receipt.get("config_sha256") != digest
        or receipt.get("checkpoint_sha256") != checkpoint_sha
    ):
        raise ValueError("cached milestone/config identity changed")
    for path, expected in receipt["artifact_sha256"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"cached evaluation artifact changed: {path}")


def baseline_frontiers() -> dict[str, Any]:
    result = {}
    for name, (inference, checkpoint_sha) in BASELINES.items():
        path = RUN_ROOT / "preflight_frontiers" / f"{name}.json"
        measured = frontier.measure(inference, checkpoint_sha=checkpoint_sha)
        if path.exists() and read_json(path) != measured:
            raise ValueError(f"cached {name} baseline frontier changed")
        if not path.exists():
            atomic_json(path, measured)
        result[name] = measured
    c3 = result["C3"]["thresholds"]["0.35"]["fixed18"]
    human = result["human25"]["thresholds"]["0.25"]["fixed18"]
    for row, bridges, components, coverage in (
        (c3, 5011, 151, 0.7581977685035346),
        (human, 5854, 238, 0.730176305255089),
    ):
        if (
            row["candidate_only_bridge_pixels"] != bridges
            or row["candidate_meaningful_components"] != components
            or not math.isclose(
                row["reference_skeleton_recall"], coverage, abs_tol=1e-12
            )
        ):
            raise ValueError(
                "promoted frontier does not reproduce the recorded baseline"
            )
    return result


def blind_command(
    checkpoint: Path, report_dir: Path, evaluation: dict[str, Any]
) -> list[str]:
    return [
        sys.executable,
        "scripts/generate_checkpoint_report.py",
        "--source",
        str(ROOT / evaluation["blind_source"]),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(report_dir),
        "--threshold",
        "0.35",
        "--halo",
        "32",
        "--tta",
        "--amp-dtype",
        "bfloat16",
        "--device",
        "cuda",
        "--max-cpu-threads",
        "16",
        "--reference-student",
        str(BASELINES["C3"][0]),
        "--jackpot-student",
        str((ROOT / evaluation["blind_reference_grid"]).parent),
        *[value for spec in FIXED_SLICES for value in ("--fixed-slice", spec)],
    ]


def milestone(
    samples: int, coordinator: Coordinator, evaluation: dict[str, Any]
) -> dict[str, Any]:
    checkpoint = ARM_DIR / f"checkpoint_milestone_{samples:08d}.pt"
    checkpoint_sha = sha256(checkpoint)
    directory = EVALUATION / f"milestone_{samples:08d}"
    receipt_path = directory / "receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        validate_cached_receipt(receipt, coordinator.digest, checkpoint_sha)
        return receipt
    directory.mkdir(parents=True, exist_ok=True)
    audit_dir = directory / "audit"
    audit_path = audit_dir / "report.json"
    if not audit_path.exists():
        run(
            [
                sys.executable,
                "-m",
                "crossres_pred.voxel.cli",
                "audit-checkpoint",
                "--checkpoint",
                checkpoint,
                "--patches",
                ROOT / evaluation["val_manifest"],
                "--output",
                audit_dir,
                "--split",
                "val",
                "--device",
                "cuda",
            ]
        )
    audit = read_json(audit_path)
    points = audit["sweep"]["points"]
    fixed = point_at(points, 0.35)
    calibrated = audit["sweep"]["selected"]
    cal_t = round(float(calibrated["threshold"]), 2)
    report_dir = directory / "blind_t035"
    run(blind_command(checkpoint, report_dir, evaluation))
    inference = report_dir / "inference"
    frontier_result = frontier.measure(inference, checkpoint_sha=checkpoint_sha)
    frontier_path = directory / "frontier.json"
    atomic_json(frontier_path, frontier_result)
    thresholds = sorted({0.25, 0.35, cal_t})
    paths = sorted((inference / "probability").glob("*.tif"))
    expected_ids = read_json(inference / "provenance.json")["target_cube_ids"]
    if len(paths) != 6 or {p.stem for p in paths} != set(expected_ids):
        raise ValueError(
            "blind morphology audit must include exactly the six fixed cubes"
        )
    published = ROOT / evaluation["blind_source"] / "cubes_PRED"
    reference = ROOT / evaluation["blind_reference_grid"]
    per_cube = {
        path.stem: morphology._cube_fast_sweep(
            path, published / path.name, reference / path.name, np.asarray(thresholds)
        )[0]
        for path in paths
    }
    aggregate = morphology._aggregate(per_cube, thresholds, {})
    morphology_rows = morphology._morphology_audit(
        paths, published, reference, thresholds
    )
    verdicts = {}
    for label, threshold in (
        ("fixed_t035", 0.35),
        ("advisory_t025", 0.25),
        ("calibrated_diagnostic", cal_t),
    ):
        cubes = next(r["cubes"] for r in morphology_rows if r["threshold"] == threshold)
        verdicts[label] = frozen.gate_verdict(
            point_at(aggregate, threshold), cubes, evaluation["gates"], soft_arm=True
        )
    gate_path = directory / "six_cube_gates.json"
    atomic_json(
        gate_path,
        {
            "thresholds": thresholds,
            "sweep": aggregate,
            "morphology": morphology_rows,
            "verdicts": verdicts,
        },
    )
    scrolls = fixed["scrolls"]
    if set(scrolls) != {"PHerc0814", "PHerc1451"}:
        raise ValueError("frozen validation scroll set changed")
    guard = all(
        math.isfinite(float(row["dice_gain_vs_baseline"]))
        and float(row["dice_gain_vs_baseline"]) >= 0
        for row in scrolls.values()
    )
    artifacts = [
        audit_path,
        frontier_path,
        gate_path,
        inference / "provenance.json",
        report_dir / "report/index.html",
    ]
    artifacts += (
        paths
        + [reference / p.name for p in paths]
        + [published / p.name for p in paths]
    )
    receipt = {
        "schema": "crossres-F0-milestone-evaluation-v1",
        "config_sha256": coordinator.digest,
        "samples": samples,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "fixed_t035": {
            k: fixed[k]
            for k in (
                "threshold",
                "macro_scroll_dice",
                "dice",
                "minimum_scroll_dice_gain_vs_baseline",
                "scrolls",
            )
        },
        "calibrated_diagnostic": {
            k: calibrated[k] for k in ("threshold", "macro_scroll_dice", "dice")
        },
        "gates": verdicts,
        "scroll_guard_passed": guard,
        "eligible": bool(guard and verdicts["fixed_t035"]["passed"]),
        "frontier": frontier_result,
        "blind_report": str(report_dir / "report/index.html"),
        "artifact_sha256": {str(p): sha256(p) for p in artifacts},
    }
    atomic_json(receipt_path, receipt)
    say(
        f"{samples:,}: fixed .35 macro={fixed['macro_scroll_dice']:.6f}; eligible={receipt['eligible']}"
    )
    return receipt


def choose(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if sorted(r["samples"] for r in rows) != SNAPSHOTS:
        raise ValueError("selection requires every one of the 25 evaluated milestones")

    def best(pool: list[dict[str, Any]]) -> int | None:
        return (
            max(
                pool,
                key=lambda r: (
                    float(r["fixed_t035"]["macro_scroll_dice"]),
                    r["samples"],
                ),
            )["samples"]
            if pool
            else None
        )

    eligible = [row for row in rows if row["eligible"]]
    return {
        "proposed_late_candidate": best(
            [r for r in eligible if r["samples"] >= 150000]
        ),
        "best_early_eligible": best([r for r in eligible if r["samples"] <= 50000]),
        "global_best_eligible": best(eligible),
        "global_best_unrestricted": best(rows),
        "exact_final_checkpoint": 250000,
        "operating_threshold": 0.35,
        "automatic_deployment": False,
        "next_experiment": None,
    }


def kaggle(
    row: dict[str, Any], config: dict[str, Any], evaluation: dict[str, Any]
) -> dict[str, Any]:
    directory = EVALUATION / f"milestone_{row['samples']:08d}"
    destination = directory / "kaggle.json"
    if destination.exists():
        cached = read_json(destination)
        if cached.get("checkpoint_sha256") != row["checkpoint_sha256"]:
            raise ValueError("Kaggle checkpoint identity changed")
        return cached
    thresholds = sorted(
        {0.35, 0.45, round(float(row["calibrated_diagnostic"]["threshold"]), 2)}
    )
    kconfig = evaluation["kaggle"]
    result = {
        "checkpoint_sha256": row["checkpoint_sha256"],
        "samples": row["samples"],
        "student_tta": False,
        "sets": {},
    }
    for name, settings in kconfig["sets"].items():
        pred_dirs = {
            f"{t:.2f}": (directory / f"kaggle_{name}_t{round(t * 100):03d}", t)
            for t in thresholds
        }
        gt_dir = ROOT / kconfig["gt_root"] / name
        say(f"Kaggle {name} at {thresholds}, at most two metric processes")
        count = frozen._dump_kaggle_predictions(
            Path(row["checkpoint"]),
            ROOT / settings["manifest"],
            gt_dir=gt_dir,
            pred_dirs=pred_dirs,
            scroll=settings["scroll"],
            device="cuda",
            mirror_tta=False,
        )
        if count != {"v14p2_val": 689, "p0500p2_val": 320}[name]:
            raise ValueError("Kaggle validation row count changed")
        scores = {}
        for label, (pred_dir, threshold) in pred_dirs.items():
            score_dir = pred_dir.with_name(pred_dir.name + "_score")
            cached = score_dir / "kaggle_summary.json"
            summary = (
                read_json(cached)
                if cached.exists()
                else frozen._score_kaggle_sharded(
                    ROOT / kconfig["metric_exe"],
                    gt_dir,
                    pred_dir,
                    score_dir,
                    shards=config["execution"]["metric_shards"],
                )
            )
            scores[label] = {**summary, "threshold": threshold}
        result["sets"][name] = {"rows": count, "scores": scores}
    atomic_json(destination, result)
    return result


def write_report(summary: dict[str, Any]) -> None:
    def link(path: str, label: str) -> str:
        relative = os.path.relpath(path, EVALUATION).replace("\\", "/")
        return f'<a href="{html.escape(relative, quote=True)}">{html.escape(label)}</a>'

    decision = summary["selection"]
    rows_html = []
    for row in summary["milestones"]:
        metrics = row["frontier"]["thresholds"]["0.35"]["fixed18"]
        rows_html.append(
            f"<tr><td>{row['samples']:,}</td><td>{row['fixed_t035']['macro_scroll_dice']:.6f}</td><td>{row['calibrated_diagnostic']['macro_scroll_dice']:.6f} @ {row['calibrated_diagnostic']['threshold']:.2f}</td><td>{'yes' if row['eligible'] else 'no'}</td><td>{metrics['candidate_only_bridge_pixels']:,}</td><td>{metrics['reference_skeleton_recall']:.4f}</td><td>{metrics['candidate_meaningful_components']}</td><td>{link(row['blind_report'], 'T=.35 report')}</td></tr>"
        )
    proposed = decision["proposed_late_candidate"]
    text = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>F0: C3 through 250k</title><style>body{font:16px/1.5 system-ui;max-width:1300px;margin:32px auto;padding:0 20px;color:#202428}table{border-collapse:collapse;width:100%}th,td{padding:9px;text-align:left;border-bottom:1px solid #ddd}.table{overflow:auto}a{color:#125bb5}code{overflow-wrap:anywhere}.note{background:#f0f3f6;padding:16px}</style><h1>C3 through 250k: F0</h1>"""
    text += f"<p>Fresh released M7; wide15 anti-aliased corpus; CE 1 + Dice 1 + M7-KL .5; SGD .001/poly over 250,000 exposures; no trust ball. Operating T=.35. All 25 checkpoints evaluated after natural training completion.</p><p class='note'>Proposed late candidate: {proposed or 'none eligible'}. Best early eligible: {decision['best_early_eligible']}. Global best eligible: {decision['global_best_eligible']}. The exact 250k checkpoint is retained and listed below. No automatic deployment or next experiment.</p>"
    text += "<p>Original C3@30k: fixed-.35 macro 0.676853; fixed18 bridges 5,011; reference-line coverage .758198; meaningful components 151. Reference-relative geometry is not human-ground-truth accuracy. T=.25 is advisory and cannot veto this run. In-run wide15 validation is telemetry only.</p>"
    text += f"<p>Candidate delta versus C3: {summary['candidate_delta_vs_C3']}. Late-versus-early H1: {summary['H1']}. Historical .0715 variability band is descriptive, not a confidence interval. Frontier review: {html.escape(str(summary['frontier_review']))}.</p>"
    text += (
        "<div class='table'><table><thead><tr><th>Samples</th><th>Macro @.35</th><th>Calibrated diagnostic</th><th>Eligible @.35</th><th>Bridge pixels</th><th>Line coverage</th><th>Components</th><th>Geometry</th></tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table></div>"
    )
    text += f"<p>Kaggle evaluated checkpoint: {summary['kaggle']['samples']:,}. Student uses no TTA; the M7+TTA readiness benchmark is explicitly a stronger inference baseline. PHerc0500P2 scores are absolute only (released M7 saw its labels).</p>"
    for name, result in summary["kaggle"]["sets"].items():
        text += f"<h2>{html.escape(name)} ({result['rows']} rows)</h2><table><tr><th>T</th><th>Composite</th><th>Topology</th><th>Surface Dice</th></tr>"
        for threshold, scores in result["scores"].items():
            text += f"<tr><td>{threshold}</td><td>{scores['mean_leaderboard']:.6f}</td><td>{scores['mean_toposcore']:.6f}</td><td>{scores['mean_surface_dice']:.6f}</td></tr>"
        text += "</table>"
    text += f"<p>Fixed-.35 readiness checks (advisory): {html.escape(str(summary['readiness_advisory']))}.</p><p><a href='summary.json'>Complete machine-readable evidence and provenance</a></p><p>Config SHA256: <code>{summary['config_sha256']}</code></p></html>"
    temporary = EVALUATION / "index.html.tmp"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, EVALUATION / "index.html")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--config-sha256")
    parser.add_argument("--baselines-only", action="store_true")
    args = parser.parse_args()
    if args.baselines_only:
        baseline_frontiers()
        say(
            "C3 and human25 cached baseline reproduction passed; no inference or training"
        )
        return
    if not args.config_sha256:
        raise ValueError("evaluation requires the sealed config SHA256")
    coordinator = Coordinator(args.config, args.config_sha256)
    coordinator.preflight()
    if not coordinator.completed_training()["complete"]:
        raise ValueError("no evaluator may run before natural 250k completion")
    baseline = baseline_frontiers()
    evaluation = read_json(CROSSRES / "configs/unified_ladder_20260902.json")[
        "evaluation"
    ]
    rows = []
    for samples in SNAPSHOTS:
        say(f"Evaluating {samples:,}/250,000 milestone")
        row = milestone(samples, coordinator, evaluation)
        rows.append(row)
        atomic_json(
            EVALUATION / "progress.json",
            {"at_utc": now(), "evaluated_samples": [r["samples"] for r in rows]},
        )
    drift_path = ARM_DIR / "ladder_eval/trust_displacement.json"
    if not drift_path.exists():
        run(
            [
                sys.executable,
                "scripts/measure_trust_displacement.py",
                "--run-dir",
                ARM_DIR,
            ]
        )
    selection = choose(rows)
    by_samples = {row["samples"]: row for row in rows}
    proposed = selection["proposed_late_candidate"]
    selected = by_samples[proposed or 250000]
    for threshold in sorted(
        {0.25, round(float(selected["calibrated_diagnostic"]["threshold"]), 2)} - {0.35}
    ):
        directory = EVALUATION / f"milestone_{selected['samples']:08d}"
        run(
            [
                sys.executable,
                "scripts/render_cached_probability_report.py",
                "--source",
                ROOT / evaluation["blind_source"],
                "--source-inference",
                directory / "blind_t035/inference",
                "--output",
                directory / f"blind_t{round(threshold * 100):03d}_companion",
                "--threshold",
                str(threshold),
                "--fixed-slices",
                *FIXED_SLICES,
            ]
        )
    scores = kaggle(selected, coordinator.config, evaluation)
    coverage = selected["frontier"]["thresholds"]["0.35"]["fixed18"][
        "reference_skeleton_recall"
    ]
    review = frontier.interpolated_bridge_limit(baseline["C3"], coverage)
    if review["resolved"]:
        actual = selected["frontier"]["thresholds"]["0.35"]["fixed18"][
            "candidate_only_bridge_pixels"
        ]
        review.update(
            candidate_bridges=actual,
            passed=actual <= review["maximum_candidate_bridges"],
        )
    early = selection["best_early_eligible"]
    candidate_macro = (
        by_samples[proposed]["fixed_t035"]["macro_scroll_dice"] if proposed else None
    )
    readiness = scores["sets"]["v14p2_val"]["scores"]["0.35"]
    summary = {
        "schema": "crossres-F0-final-evaluation-v1",
        "config_sha256": coordinator.digest,
        "evaluated_samples": SNAPSHOTS,
        "selection": selection,
        "milestones": rows,
        "baseline_frontiers": baseline,
        "drift": read_json(drift_path),
        "candidate_delta_vs_C3": candidate_macro - 0.6768526715087372
        if proposed
        else None,
        "H1": candidate_macro >= by_samples[early]["fixed_t035"]["macro_scroll_dice"]
        if proposed and early
        else None,
        "frontier_review": review,
        "kaggle": scores,
        "readiness_advisory": {
            "composite": readiness["mean_leaderboard"] >= 0.6212,
            "topology": readiness["mean_toposcore"] >= 0.3776,
        },
        "operator_review_required": True,
        "retrospective_low_score_alerts": [
            [previous["samples"], current["samples"]]
            for previous, current in pairwise(rows)
            if previous["samples"] > 100000
            and previous["calibrated_diagnostic"]["macro_scroll_dice"] < 0.5926
            and current["calibrated_diagnostic"]["macro_scroll_dice"] < 0.5926
        ],
    }
    atomic_json(EVALUATION / "summary.json", summary)
    write_report(summary)
    say(
        "All F0 evaluation complete; HTML ready for operator review. No deployment or follow-up run."
    )


if __name__ == "__main__":
    main()
