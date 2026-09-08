#!/usr/bin/env python3
"""Journalled runner for the RELEASE line of the unified campaign.

Runs the shipped Socratic Method recipe (D:/work/socratic_method v31: the
exact training argv and all seven energy terms, with the operator-selected
8x trust radius) on the native-only four-scroll atlas corpus, then the
leave-one-out arms that test whether each energy term still earns its place
on the larger corpus. Stages:

  medial_ready   -> wait for the crest (medial) sidecars of all four atlases
  release_corpus -> replay the v16 250k row stream with fresh crest gates
                    (build_atlas_patch_corpus.py, reuse mode)
  connectivity   -> PHerc0139 bridge audit + dynamic-connectivity state bound
                    to the new manifest (the only record with an M7 volume)
  handover       -> take the GPU from the wide15 chain once B0/B0r are scored
  arms           -> R_full (50k) then leave-one-out arms (25k each), each
                    through the identical frozen harness (evaluate_ladder_arm)
  decision       -> release vs M7 / M7+TTA, per-term leave-one-out verdicts

Recipe: configs/release_ladder_20260902.json. Journal:
output/crossres_data/logs/release_ladder_state.json. Every stage resumes
from on-disk artifacts. Shared helpers come from run_unified_ladder.py.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_unified_ladder import (
    CROSSRES,
    LOGS,
    MAX_TRAIN_ATTEMPTS,
    OBJECTIVE_BOOLEAN_FLAGS,
    OBJECTIVE_FLAGS,
    ROOT,
    _atomic_json,
    _kill_tree,
    _milestone_durable,
    _read_json,
    _sha256,
    acquire_gpu_lock,
    release_gpu_lock,
    standard_two_sided_dice,
    tie_margin,
)

CONFIG_PATH = CROSSRES / "configs/release_ladder_20260902.json"
JOURNAL = LOGS / "release_ladder_state.json"
LOG = LOGS / "release_ladder.log"

RELEASE_FLAGS: dict[str, str] = {
    "m7_preservation_radius": "--loss-m7-preservation-radius",
    "m7_preservation_anchor_threshold": "--loss-m7-preservation-anchor-threshold",
    "dynamic_medial_connectivity_probability_floor": (
        "--loss-dynamic-medial-connectivity-probability-floor"
    ),
    "dynamic_medial_connectivity_steps": "--loss-dynamic-medial-connectivity-steps",
}


def _say(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [release] {message}"
    print(line, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _journal() -> dict[str, Any]:
    if JOURNAL.is_file():
        return _read_json(JOURNAL)
    return {"schema": "crossres-release-ladder-journal-v1", "stages": {}}


def _mark(stage: str, status: str, **extra: Any) -> None:
    journal = _journal()
    journal["stages"][stage] = {
        "status": status,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **extra,
    }
    _atomic_json(JOURNAL, journal)
    _say(f"stage {stage}: {status}")


def _stage_done(stage: str) -> bool:
    return _journal()["stages"].get(stage, {}).get("status") == "complete"


def _run_logged(command: list[str], log_name: str) -> int:
    log_path = LOGS / log_name
    _say("+ " + " ".join(str(part) for part in command) + f" -> {log_name}")
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
        stream.flush()
        result = subprocess.run(
            [str(part) for part in command],
            cwd=str(CROSSRES),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode


def _format(value: Any) -> str:
    if isinstance(value, bool):
        raise TypeError("boolean objective keys use their own flags")
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def build_release_command(
    config: dict[str, Any],
    arm: dict[str, Any],
    output: Path,
    *,
    connectivity_state: Path | None,
) -> list[str]:
    """The v31 training argv with the corpus, budget and one override applied."""
    base = config["base"]
    optimizer = {**base["optimizer"], **arm.get("optimizer_overrides", {})}
    schedule = base["schedule"]
    objective = {**base["objective"], **arm.get("objective_overrides", {})}
    samples = int(arm["samples"])
    samples_per_epoch = int(schedule["samples_per_epoch"])
    command = [
        sys.executable, "-m", "crossres_pred.voxel.cli", "train",
        "--patches", str(ROOT / config["corpus"]["manifest"]),
        "--validation-patches", str(ROOT / config["corpus"]["validation_manifest"]),
        "--output", str(output),
        "--m7-checkpoint", str(ROOT / config["initialization"]["checkpoint"]),
        "--preset", "m7-resenc-l",
        "--epochs", str(math.ceil(samples / samples_per_epoch)),
        "--samples-per-epoch", str(samples_per_epoch),
        "--max-train-samples", str(samples),
        "--snapshot-samples", *[str(v) for v in arm["snapshot_samples"]],
        "--batch-size", str(schedule["batch_size"]),
        "--accumulate", str(schedule["accumulate"]),
        "--amp-dtype", schedule["amp_dtype"],
        "--optimizer", optimizer["optimizer"],
        "--learning-rate", repr(float(optimizer["learning_rate"])),
        "--lr-schedule", optimizer["lr_schedule"],
        "--warmup-samples", str(optimizer["warmup_samples"]),
        "--weight-decay", repr(float(optimizer["weight_decay"])),
    ]
    if optimizer["optimizer"] in {"adam", "adamw"}:
        command += [
            "--adamw-beta1", repr(float(optimizer["adam_beta1"])),
            "--adamw-beta2", repr(float(optimizer["adam_beta2"])),
            "--adamw-eps", repr(float(optimizer["adam_eps"])),
        ]
    else:
        command += ["--momentum", repr(float(optimizer["momentum"]))]
    command += [
        "--stratified-sampling" if schedule["stratified_sampling"]
        else "--no-stratified-sampling",
        "--train-augmentation" if schedule["train_augmentation"]
        else "--no-train-augmentation",
        "--seed", str(base["seed"]),
        "--num-workers", str(base["num_workers"]),
        "--max-cpu-threads", str(base["max_cpu_threads"]),
        "--device", base["device"],
    ]
    for key, flag in {**OBJECTIVE_FLAGS, **RELEASE_FLAGS}.items():
        if key in objective:
            command += [flag, _format(objective[key])]
    for key, (on_flag, off_flag) in OBJECTIVE_BOOLEAN_FLAGS.items():
        if key in objective:
            command.append(on_flag if objective[key] else off_flag)
    if float(objective.get("dynamic_medial_connectivity_weight", 0.0)) > 0.0:
        if connectivity_state is None:
            raise ValueError(f"{arm['arm_id']}: connectivity weight without a state")
        command += ["--dynamic-medial-connectivity-state", str(connectivity_state)]
    return command


def stage_medial_ready(config: dict[str, Any]) -> None:
    atlases = {name: ROOT / path for name, path in config["corpus"]["medial_atlases"].items()}
    while True:
        pending = []
        for name, root in atlases.items():
            state = root / "medial_state.json"
            if not state.is_file() or _read_json(state).get("state") != "complete":
                pending.append(name)
        if not pending:
            _say("medial sidecars complete for " + ", ".join(sorted(atlases)))
            return
        _say(f"waiting for medial sidecars: {', '.join(pending)}")
        time.sleep(300)


def stage_release_corpus(config: dict[str, Any]) -> dict[str, Any]:
    corpus = config["corpus"]
    summary_path = ROOT / corpus["output"] / "summary.json"
    if not (summary_path.is_file() and _read_json(summary_path).get("state") == "complete"):
        code = _run_logged(
            [
                sys.executable, "scripts/build_atlas_patch_corpus.py",
                "--plan", str(ROOT / corpus["plan"]),
            ],
            "release_ladder_corpus.log",
        )
        if code != 0:
            raise SystemExit(f"release corpus build failed ({code})")
    summary = _read_json(summary_path)
    if summary.get("state") != "complete":
        raise SystemExit("release corpus summary is not complete")
    manifest = ROOT / corpus["manifest"]
    facts = {
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "rows": int(summary["patches"]),
        "sources": {
            record: int(value["patches"]) for record, value in summary["sources"].items()
        },
        "sampling_strategies": summary.get("sampling_strategies"),
    }
    _say(f"release corpus: {facts['rows']:,} rows, sha {facts['manifest_sha256'][:16]}...")
    return facts


def stage_connectivity(config: dict[str, Any]) -> Path:
    connectivity = config["connectivity"]
    corpus = config["corpus"]
    audit_path = ROOT / connectivity["audit_output"]
    state_dir = ROOT / connectivity["state_output"]
    state_json = state_dir / "connectivity_state.json"
    catalog = ROOT / corpus["output"] / "atlas_catalog.json"
    if not audit_path.is_file():
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        code = _run_logged(
            [
                sys.executable, "scripts/audit_pherc0139_training_medial_bridges.py",
                "--patches", str(ROOT / corpus["manifest"]),
                "--atlas-catalog", str(catalog),
                "--record-id", connectivity["record_id"],
                "--output", str(audit_path),
                "--halo-yx", str(connectivity["halo_yx"]),
                "--border-margin", str(connectivity["border_margin"]),
                "--schedule-seed", str(connectivity["schedule_seed"]),
                "--schedule-samples", str(connectivity["schedule_samples"]),
            ],
            "release_ladder_connectivity.log",
        )
        if code != 0:
            raise SystemExit(f"bridge audit failed ({code})")
    if not state_json.is_file():
        code = _run_logged(
            [
                sys.executable,
                "scripts/build_pherc0139_training_medial_connectivity_atlas.py",
                "--audit", str(audit_path),
                "--output", str(state_dir),
                "--maximum-propagation-steps",
                str(connectivity["maximum_propagation_steps"]),
            ],
            "release_ladder_connectivity.log",
        )
        if code != 0:
            raise SystemExit(f"connectivity atlas build failed ({code})")
    if not state_json.is_file():
        raise SystemExit(f"connectivity state missing: {state_json}")
    return state_json


def stage_handover(config: dict[str, Any]) -> None:
    """Verify the initializer; the GPU itself is serialized per arm by the
    shared lock (`acquire_gpu_lock`), so the release arms start the moment
    the corpus is ready and the wide15 chain simply waits its turn."""
    checkpoint = ROOT / config["initialization"]["checkpoint"]
    if _sha256(checkpoint) != config["initialization"]["sha256"]:
        raise SystemExit("released M7 SHA mismatch")
    _say("released M7 verified; arms serialize on the shared GPU lock")


def train_arm(
    config: dict[str, Any], arm: dict[str, Any], connectivity_state: Path
) -> Path:
    output = ROOT / config["output_root"] / "arms" / str(arm["arm_id"])
    milestone = output / f"checkpoint_milestone_{int(arm['samples']):08d}.pt"
    if milestone.is_file():
        _say(f"{arm['arm_id']}: final milestone already durable, skipping training")
        return output
    output.mkdir(parents=True, exist_ok=True)
    state = (
        connectivity_state
        if float(
            {**config["base"]["objective"], **arm.get("objective_overrides", {})}.get(
                "dynamic_medial_connectivity_weight", 0.0
            )
        ) > 0.0
        else None
    )
    command = build_release_command(config, arm, output, connectivity_state=state)
    finish_epoch = bool(arm.get("finish_epoch_after_final_milestone", False))
    _atomic_json(
        output / "ladder_recipe.json",
        {
            "schema": "crossres-release-ladder-arm-recipe-v1",
            "arm_id": arm["arm_id"],
            "delta": arm.get("delta"),
            "samples": arm["samples"],
            "finish_epoch_after_final_milestone": finish_epoch,
            "objective_overrides": arm.get("objective_overrides", {}),
            "optimizer_overrides": arm.get("optimizer_overrides", {}),
            "base": config["base"],
            "connectivity_state": str(state) if state else None,
            "command": [str(part) for part in command],
            "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    for attempt in range(1, MAX_TRAIN_ATTEMPTS + 1):
        launch = list(command)
        if (output / "run.json").exists():
            launch.append("--resume")
        log_path = LOGS / f"release_ladder_train_{arm['arm_id']}.log"
        _say(f"{arm['arm_id']}: training attempt {attempt}")
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
            stream.flush()
            process = subprocess.Popen(
                [str(part) for part in launch],
                cwd=str(CROSSRES),
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            announced_final = False
            final_durable_at: float | None = None
            while True:
                if _milestone_durable(milestone):
                    if not announced_final:
                        suffix = (
                            "; waiting for final validation epoch"
                            if finish_epoch
                            else ""
                        )
                        _say(f"{arm['arm_id']}: final milestone durable{suffix}")
                        announced_final = True
                        final_durable_at = time.monotonic()
                    if finish_epoch and process.poll() is None:
                        assert final_durable_at is not None
                        if time.monotonic() - final_durable_at > 30 * 60:
                            _say(
                                f"{arm['arm_id']}: final validation exceeded 30m; "
                                "keeping the durable milestone and stopping the process"
                            )
                            _kill_tree(process.pid)
                            process.wait(timeout=120)
                            return output
                        time.sleep(15)
                        continue
                    if process.poll() is None:
                        _kill_tree(process.pid)
                        process.wait(timeout=120)
                    return output
                code = process.poll()
                if code is not None:
                    if milestone.is_file() and _milestone_durable(milestone):
                        return output
                    _say(f"{arm['arm_id']}: training exited {code} before the milestone")
                    break
                time.sleep(60)
    raise SystemExit(f"{arm['arm_id']}: training failed {MAX_TRAIN_ATTEMPTS} times")


def eval_arm(arm_id: str, run_dir: Path, *, diagnostic: bool) -> dict[str, Any]:
    summary_path = run_dir / "ladder_eval" / "arm_summary.json"
    if not summary_path.is_file():
        command = [
            sys.executable, "scripts/evaluate_ladder_arm.py",
            "--arm-id", arm_id, "--run-dir", str(run_dir), "--soft-arm",
        ]
        if diagnostic:
            command.append("--diagnostic")
        code = _run_logged(command, f"release_ladder_eval_{arm_id}.log")
        if code != 0:
            raise SystemExit(f"evaluation of {arm_id} failed ({code})")
    return _read_json(summary_path)


def wait_for_summary(
    arm_id: str, run_dir: Path, *, patience_seconds: int = 3 * 60 * 60
) -> dict[str, Any]:
    """The arm's frozen-harness summary, waiting for the small-card evaluation.

    ``watch_release_panels.py`` launches each arm's harness on cuda:1 as soon
    as its final milestone is durable.  Wait for that to land rather than
    duplicating the work; if nothing appears (no watcher running), score the
    arm here so the ladder still completes on its own.
    """
    summary_path = run_dir / "ladder_eval" / "arm_summary.json"
    deadline = time.time() + patience_seconds
    announced = False
    while not summary_path.is_file():
        if time.time() > deadline:
            _say(f"{arm_id}: no evaluation appeared; scoring it here")
            return eval_arm(arm_id, run_dir, diagnostic=arm_id == "R_full")
        if not announced:
            _say(f"{arm_id}: waiting for the small-card evaluation")
            announced = True
        time.sleep(60)
    return _read_json(summary_path)


def loo_verdicts(
    summaries: dict[str, dict[str, Any]],
    *,
    milestone: int,
    margin: float,
) -> list[dict[str, Any]]:
    full = summaries["R_full"]
    full_at = full["milestone_audits"].get(str(milestone))
    verdicts = []
    for arm_id, summary in sorted(summaries.items()):
        if not arm_id.startswith("R_minus"):
            continue
        arm_at = summary["milestone_audits"].get(str(milestone))
        if full_at is None or arm_at is None:
            verdicts.append({"arm_id": arm_id, "verdict": "milestone-missing"})
            continue
        delta = float(full_at["macro_scroll_dice"]) - float(arm_at["macro_scroll_dice"])
        arm_gates = summary["gate_verdicts"]["calibrated"]["passed"]
        full_gates = full["gate_verdicts"]["calibrated"]["passed"]
        if delta > margin and not (arm_gates and not full_gates):
            verdict = "earns-its-place"
        elif delta < -margin:
            verdict = "hurts"
        else:
            verdict = "no-measurable-effect"
        verdicts.append(
            {
                "arm_id": arm_id,
                "term": arm_id.replace("R_minus_", ""),
                "macro_full_at_milestone": float(full_at["macro_scroll_dice"]),
                "macro_without_term_at_milestone": float(arm_at["macro_scroll_dice"]),
                "delta_full_minus_without": delta,
                "gates_without_term": arm_gates,
                "verdict": verdict,
            }
        )
    return verdicts


def stage_decision(config: dict[str, Any], summaries: dict[str, dict[str, Any]]) -> None:
    baselines = {}
    for name, directory in (("M7RAW", "m7_raw"), ("M7TTA", "m7_tta")):
        path = ROOT / config["evaluation"]["baselines_root"] / directory / "ladder_eval" / "arm_summary.json"
        if path.is_file():
            baselines[name] = _read_json(path)
    margin = 0.005
    wide_b0 = ROOT / "output/crossres_data/unified_ladder_20260902/arms/B0/ladder_eval/arm_summary.json"
    wide_b0r = ROOT / "output/crossres_data/unified_ladder_20260902/arms/B0r/ladder_eval/arm_summary.json"
    if wide_b0.is_file() and wide_b0r.is_file():
        margin, _ = tie_margin(
            _read_json(wide_b0)["milestone_audits"], _read_json(wide_b0r)["milestone_audits"]
        )

    def kaggle(summary: dict[str, Any]) -> dict[str, Any]:
        block = summary.get("kaggle") or {}
        return {
            set_name: (block.get(set_name, {}).get("calibrated") or {}).get("mean_leaderboard")
            for set_name in ("v14p2_val", "p0500p2_val")
        }

    def row(summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "macro": float(summary["arm_score_macro"]),
            "standard_two_sided_dice": standard_two_sided_dice(summary),
            "best_milestone": summary["best_milestone"],
            "calibrated_threshold": float(summary["calibrated_threshold"]),
            "gates_calibrated": summary["gate_verdicts"]["calibrated"]["passed"],
            "gates_fixed_t025": (
                (summary["gate_verdicts"].get("fixed_t025") or {}).get("passed")
            ),
            "recall_of_reference": summary["gate_verdicts"]["calibrated"]["recall_vs_reference"],
            "kaggle_leaderboard_calibrated": kaggle(summary),
        }

    milestone = int(config["evaluation"]["loo_comparison_milestone"])
    decision = {
        "schema": "crossres-release-ladder-decision-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "release_on_corpus": row(summaries["R_full"]) if "R_full" in summaries else None,
        "baselines": {name: row(summary) for name, summary in baselines.items()},
        "release_minus_baselines_macro": {
            name: float(summaries["R_full"]["arm_score_macro"]) - float(summary["arm_score_macro"])
            for name, summary in baselines.items()
        } if "R_full" in summaries else {},
        "tie_margin": margin,
        "loo_comparison_milestone": milestone,
        "term_verdicts": loo_verdicts(summaries, milestone=milestone, margin=margin),
        "arms": {arm_id: row(summary) for arm_id, summary in sorted(summaries.items())},
        "next": "operator review (PNGs veto, never pick); the final 250k-sample run is NOT launched",
    }
    output_root = ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_root / "decision.json", decision)
    lines = [
        "# Release ladder decision table", "",
        f"tie margin {margin:.4f}; leave-one-out compared at {milestone:,} samples", "",
        "| model | ordinary Dice cal pooled | ordinary Dice cal macro | ordinary Dice @0.25 pooled | ordinary Dice @0.25 macro | milestone | cal T | gates@cal | gates@0.25 | recall-of-ref | Kaggle composite v14p2 | Kaggle composite 0500P2 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, value in [*baselines.items(), *sorted(summaries.items())]:
        r = row(value)
        k = r["kaggle_leaderboard_calibrated"]
        cells = [
            "n/a" if k.get(set_name) is None else f"{float(k[set_name]):.4f}"
            for set_name in ("v14p2_val", "p0500p2_val")
        ]
        gates = "PASS" if r["gates_calibrated"] else "fail"
        gates_t025 = (
            "n/a"
            if r["gates_fixed_t025"] is None
            else ("PASS" if r["gates_fixed_t025"] else "fail")
        )
        dice = r["standard_two_sided_dice"]

        def score(value: Any) -> str:
            return "n/a" if value is None else f"{float(value):.4f}"

        lines.append(
            f"| {name} | {score(dice['calibrated']['pooled_dice'])} "
            f"| {score(dice['calibrated']['macro_scroll_dice'])} "
            f"| {score(dice['fixed_t025']['pooled_dice'])} "
            f"| {score(dice['fixed_t025']['macro_scroll_dice'])} "
            f"| {r['best_milestone']:,} "
            f"| {r['calibrated_threshold']:.2f} | {gates} | {gates_t025} "
            f"| {r['recall_of_reference']:.3f} | {cells[0]} | {cells[1]} |"
        )
    lines += ["", "| term | macro with | macro without | delta | verdict |", "|---|---|---|---|---|"]
    for verdict in decision["term_verdicts"]:
        if "delta_full_minus_without" in verdict:
            lines.append(
                f"| {verdict['term']} | {verdict['macro_full_at_milestone']:.4f} "
                f"| {verdict['macro_without_term_at_milestone']:.4f} "
                f"| {verdict['delta_full_minus_without']:+.4f} | {verdict['verdict']} |"
            )
        else:
            lines.append(f"| {verdict['arm_id']} | | | | {verdict['verdict']} |")
    (output_root / "ladder_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _say("decision written")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the stage plan and every arm command; touch nothing")
    arguments = parser.parse_args()
    config = _read_json(CONFIG_PATH)
    arms_by_id = {arm["arm_id"]: arm for arm in config["arms"]}

    if arguments.dry_run:
        state = ROOT / config["connectivity"]["state_output"] / "connectivity_state.json"
        for arm_id in config["committed_arms"] + config.get("optional_arms", []):
            arm = arms_by_id[arm_id]
            command = build_release_command(
                config, arm, ROOT / config["output_root"] / "arms" / arm_id,
                connectivity_state=state,
            )
            print(f"\n[{arm_id}] {arm.get('delta')}\n  " + " ".join(command))
        return 0

    if not _stage_done("medial_ready"):
        _mark("medial_ready", "running")
        stage_medial_ready(config)
        _mark("medial_ready", "complete")
    if not _stage_done("release_corpus"):
        _mark("release_corpus", "running")
        facts = stage_release_corpus(config)
        _mark("release_corpus", "complete", **facts)
    if not _stage_done("connectivity"):
        _mark("connectivity", "running")
        state_json = stage_connectivity(config)
        _mark("connectivity", "complete", state=str(state_json))
    state_json = ROOT / config["connectivity"]["state_output"] / "connectivity_state.json"
    if not _stage_done("handover"):
        _mark("handover", "running")
        stage_handover(config)
        _mark("handover", "complete")

    summaries: dict[str, dict[str, Any]] = {}
    for arm_id in config["committed_arms"]:
        stage = f"arm_{arm_id}"
        run_dir = ROOT / config["output_root"] / "arms" / arm_id
        if not _stage_done(stage):
            _mark(stage, "running")
            acquire_gpu_lock(f"release:{arm_id}")
            try:
                run_dir = train_arm(config, arms_by_id[arm_id], state_json)
            finally:
                release_gpu_lock()
            # Training stays serialized on the big card. Evaluation is
            # collected only after the arm loop; an operator pause can return
            # before that point without spawning the frozen metric harness.
            _mark(stage, "trained")
        if arm_id == config.get("operator_pause_after_arm"):
            _mark(
                "operator_review_pause",
                "complete",
                after_arm=arm_id,
                reason="fixed-T=0.25 blob review before any further arm or evaluator",
            )
            _say(
                f"PAUSED AFTER {arm_id} -- awaiting fixed-T=0.25 operator review; "
                "no evaluator or later arm launched"
            )
            return 0

    for arm_id in config["committed_arms"]:
        run_dir = ROOT / config["output_root"] / "arms" / arm_id
        summaries[arm_id] = wait_for_summary(arm_id, run_dir)
        _mark(f"arm_{arm_id}", "complete", macro=summaries[arm_id]["arm_score_macro"])

    if not _stage_done("decision"):
        _mark("decision", "running")
        stage_decision(config, summaries)
        _mark("decision", "complete")
    _say("RELEASE LADDER COMPLETE -- awaiting operator review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
