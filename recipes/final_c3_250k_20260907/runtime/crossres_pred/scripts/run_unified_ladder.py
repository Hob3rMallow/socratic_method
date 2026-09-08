#!/usr/bin/env python3
"""Journalled queue runner for the unified Socratic-x-wide ablation ladder.

Owns the whole chain after the A0 100k stop: trust-region displacement
measurement -> full anti-aliased reprojection -> corpus validation -> the
pre-registered arms (train 50k, frozen 250k LR declaration, stop at the
durable milestone, eval pack) -> conditional arms by pre-registered triggers
-> D-phase combo-confirm -> decision report. HARD STOP before the final
100k+ run (operator review; champion rule).

Recipe: configs/unified_ladder_20260902.json. One owner: the scheduled task
`VesuviusCrossres-UnifiedLadder` (see launch_unified_ladder.ps1). The journal
is output/crossres_data/logs/unified_ladder_state.json; every stage is
idempotent and resumes from on-disk artifacts after a crash or reboot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

if os.name == "nt":
    import msvcrt
else:  # pragma: no cover - the production runners are Windows scheduled tasks
    import fcntl

ROOT = Path(__file__).resolve().parents[2]
CROSSRES = ROOT / "crossres_pred"
CONFIG_PATH = CROSSRES / "configs/unified_ladder_20260902.json"
LOGS = ROOT / "output/crossres_data/logs"
JOURNAL = LOGS / "unified_ladder_state.json"
LOG = LOGS / "unified_ladder.log"
A0_RUN = ROOT / "output/crossres_data/m7_xr_wide15_jackpot_recipe_20260901"
WIDE15_TASK = "VesuviusCrossres-Wide15-Pipeline"

COMMITTED_ARMS = ("B0", "B0r", "C1", "C3")
DIAGNOSTIC_ARMS = {"A0", "B0", "B0r"}
MAX_TRAIN_ATTEMPTS = 3

OBJECTIVE_FLAGS: dict[str, str] = {
    "ce_weight": "--loss-ce-weight",
    "dice_weight": "--loss-dice-weight",
    "separation_weight": "--loss-separation-weight",
    "separation_radius": "--loss-separation-radius",
    "separation_max_teacher_q": "--loss-separation-max-teacher-q",
    "m7_anchor_weight": "--loss-m7-anchor-weight",
    "m7_anchor_unknown_corridor_radius": "--loss-m7-anchor-unknown-corridor-radius",
    "m7_preservation_weight": "--loss-m7-preservation-weight",
    "medial_recall_weight": "--loss-medial-recall-weight",
    "medial_tail_floor_weight": "--loss-medial-tail-floor-weight",
    "medial_tail_floor_probability": "--loss-medial-tail-floor-probability",
    "medial_tail_bottom_fraction": "--loss-medial-tail-bottom-fraction",
    "pinned_axial_weight": "--loss-pinned-axial-weight",
    "dynamic_medial_connectivity_weight": (
        "--loss-dynamic-medial-connectivity-weight"
    ),
    "m7_trust_region_relative_l2": "--m7-trust-region-relative-l2",
}
OBJECTIVE_BOOLEAN_FLAGS: dict[str, tuple[str, str]] = {
    "m7_anchor_known_agreement": (
        "--loss-m7-anchor-known-agreement",
        "--no-loss-m7-anchor-known-agreement",
    ),
    "m7_anchor_confident_agreement": (
        "--loss-m7-anchor-confident-agreement",
        "--no-loss-m7-anchor-confident-agreement",
    ),
}
ZERO_UNLESS_SET = (
    "separation_weight",
    "m7_anchor_weight",
    "m7_preservation_weight",
    "medial_recall_weight",
    "medial_tail_floor_weight",
    "pinned_axial_weight",
    "dynamic_medial_connectivity_weight",
    "m7_trust_region_relative_l2",
)


def _say(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [ladder] {message}"
    print(line, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _journal() -> dict[str, Any]:
    if JOURNAL.is_file():
        return _read_json(JOURNAL)
    return {"schema": "crossres-unified-ladder-journal-v1", "stages": {}}


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


def evaluate_trained_arm(
    config: dict[str, Any],
    arm_id: str,
    run_dir: Path,
    *,
    soft: bool,
    diagnostic: bool,
) -> dict[str, Any] | None:
    """Evaluate a trained arm unless the frozen operator pause names it.

    Returning ``None`` is an intentional clean stop: the training milestone is
    durable, the GPU lock is released by the caller's ``finally`` block, and no
    evaluator (therefore no metric.exe shard fan-out) has been launched.
    """
    controls = config.get("operator_controls") or {}
    if controls.get("pause_after_training_arm") == arm_id:
        _mark(
            f"arm_{arm_id}",
            "trained",
            run_dir=str(run_dir),
            next_action="operator review required before evaluation",
        )
        _say(
            f"{arm_id}: operator pause after training; evaluator was not launched"
        )
        return None
    return _eval_arm(arm_id, run_dir, soft=soft, diagnostic=diagnostic)


def pause_after_evaluated_arm(
    config: dict[str, Any],
    arm_id: str,
    run_dir: Path,
    summary: dict[str, Any],
) -> bool:
    """Return true when the operator requested a clean stop after this arm."""
    controls = config.get("operator_controls") or {}
    if controls.get("pause_after_evaluation_arm") != arm_id:
        return False
    _mark(
        "operator_review_pause",
        "complete",
        after_arm=arm_id,
        run_dir=str(run_dir),
        summary=str(run_dir / "ladder_eval" / "arm_summary.json"),
        macro=summary["arm_score_macro"],
        next_action="operator review required before conditional or combo arms",
    )
    _say(
        f"{arm_id}: operator pause after evaluation; no conditional or combo arm launched"
    )
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_logged(command: list[str], log_name: str, *, cwd: Path = CROSSRES) -> int:
    log_path = LOGS / log_name
    _say("+ " + " ".join(str(part) for part in command) + f" -> {log_name}")
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
        stream.flush()
        result = subprocess.run(
            [str(part) for part in command],
            cwd=str(cwd),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode


def build_arm_command(
    config: dict[str, Any], arm: dict[str, Any], output: Path
) -> list[str]:
    """Full train CLI flag set for one arm; every term weight explicit."""
    base = config["base"]
    optimizer = base["optimizer"]
    schedule = base["schedule"]
    objective = dict(arm["objective"])
    for key in ZERO_UNLESS_SET:
        objective.setdefault(key, 0.0)
    command = [
        sys.executable, "-m", "crossres_pred.voxel.cli", "train",
        "--patches", str(ROOT / config["corpus"]["soft_manifest"]),
        "--output", str(output),
        "--m7-checkpoint", str(ROOT / config["initialization"]["checkpoint"]),
        "--preset", "m7-resenc-l",
        "--epochs", str(schedule["epochs"]),
        "--samples-per-epoch", str(schedule["samples_per_epoch"]),
        "--max-train-samples", str(schedule["max_train_samples"]),
        "--snapshot-samples",
        *[str(value) for value in schedule["snapshot_samples"]],
        "--batch-size", str(schedule["batch_size"]),
        "--accumulate", str(schedule["accumulate"]),
        "--amp-dtype", schedule["amp_dtype"],
        "--optimizer", optimizer["optimizer"],
        "--learning-rate", repr(float(optimizer["learning_rate"])),
        "--lr-schedule", optimizer["lr_schedule"],
        "--momentum", repr(float(optimizer["momentum"])),
        "--weight-decay", repr(float(optimizer["weight_decay"])),
        "--warmup-samples", str(optimizer["warmup_samples"]),
        "--no-stratified-sampling",
        "--train-augmentation",
        "--seed", str(arm["seed"]),
        "--num-workers", str(base["num_workers"]),
        "--max-cpu-threads", str(base["max_cpu_threads"]),
        "--device", base["device"],
    ]
    for key, flag in OBJECTIVE_FLAGS.items():
        if key in objective:
            value = objective[key]
            command += [
                flag,
                str(value) if isinstance(value, int) and not isinstance(value, bool)
                else repr(float(value)),
            ]
    for key, (on_flag, off_flag) in OBJECTIVE_BOOLEAN_FLAGS.items():
        if key in objective:
            command.append(on_flag if objective[key] else off_flag)
    return command


def tie_margin(
    b0_audits: dict[str, Any], b0r_audits: dict[str, Any]
) -> tuple[float, float]:
    """max(0.005, 2x RMS of matched-milestone macro deltas)."""
    deltas = []
    for samples, b0_point in b0_audits.items():
        if samples in b0r_audits:
            deltas.append(
                float(b0_point["macro_scroll_dice"])
                - float(b0r_audits[samples]["macro_scroll_dice"])
            )
    rms = math.sqrt(sum(d * d for d in deltas) / len(deltas)) if deltas else 0.0
    return max(0.005, 2.0 * rms), rms


def surviving_terms(
    summaries: dict[str, dict[str, Any]], margin: float
) -> list[str]:
    """A term survives if its arm passes calibrated gates, holds the scroll
    guard, and does not trail B0 by more than the tie margin."""
    b0_macro = float(summaries["B0"]["arm_score_macro"])
    survivors = []
    for arm_id, summary in summaries.items():
        if arm_id in {"B0", "B0r", "A0"}:
            continue
        best = summary["milestone_audits"][str(summary["best_milestone"])]
        scroll_guard = all(
            float(metrics["dice_gain_vs_baseline"]) >= 0.0
            for metrics in best["scrolls"].values()
        )
        if (
            summary["gate_verdicts"]["calibrated"]["passed"]
            and scroll_guard
            and float(summary["arm_score_macro"]) >= b0_macro - margin
        ):
            survivors.append(arm_id)
    return sorted(survivors)


def pick_winner(
    summaries: dict[str, dict[str, Any]]
) -> tuple[str | None, list[str]]:
    """Highest macro among calibrated-gate passers; None if nobody passes."""
    ranked = sorted(
        (arm_id for arm_id in summaries if arm_id != "A0"),
        key=lambda arm_id: -float(summaries[arm_id]["arm_score_macro"]),
    )
    passers = [
        arm_id
        for arm_id in ranked
        if summaries[arm_id]["gate_verdicts"]["calibrated"]["passed"]
    ]
    return (passers[0] if passers else None), ranked


def _milestone_durable(path: Path) -> bool:
    if not path.is_file():
        return False
    size = path.stat().st_size
    time.sleep(20)
    return path.is_file() and path.stat().st_size == size and size > 0


def _kill_tree(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/T", "/F", "/PID", str(pid)], check=False, capture_output=True
    )


GPU_LOCK = LOGS / "gpu.lock"


@contextmanager
def _gpu_lock_transaction() -> Iterator[None]:
    """Serialize marker inspection, stale reclamation, and atomic creation.

    ``Path.open('x')`` makes creation atomic, but creation alone does not make
    stale recovery atomic: two rebooted runners can both inspect the old
    marker, then the slower one can unlink the faster one's replacement.  A
    one-byte OS lock closes that race and is released automatically on crash.
    The observable JSON marker remains compatible with already-running older
    runners.
    """
    guard = GPU_LOCK.with_name(f"{GPU_LOCK.name}.acquire")
    guard.parent.mkdir(parents=True, exist_ok=True)
    with guard.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        while True:
            try:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:  # pragma: no cover
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                time.sleep(0.05)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _pid_alive(pid: int) -> bool:
    probe = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        check=False, capture_output=True, text=True,
    )
    return str(pid) in (probe.stdout or "")


def acquire_gpu_lock(owner: str, *, poll_seconds: int = 60) -> None:
    """One GPU, several journalled runners: serialize training + eval.

    Marker inspection/replacement is transactionally serialized; a marker
    whose pid is dead is stale and is taken over.
    """
    while True:
        acquired = False
        with _gpu_lock_transaction():
            if GPU_LOCK.exists():
                try:
                    holder = _read_json(GPU_LOCK)
                except (OSError, ValueError):
                    holder = {}
                pid = int(holder.get("pid", 0) or 0)
                if not pid or not _pid_alive(pid):
                    GPU_LOCK.unlink(missing_ok=True)
            if not GPU_LOCK.exists():
                try:
                    with GPU_LOCK.open("x", encoding="utf-8") as stream:
                        json.dump(
                            {
                                "owner": owner,
                                "pid": os.getpid(),
                                "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            },
                            stream,
                        )
                    acquired = True
                except FileExistsError:
                    # Compatibility with an already-running pre-transaction
                    # runner which may create the marker outside our guard.
                    pass
        if acquired:
            return
        time.sleep(poll_seconds)


def release_gpu_lock() -> None:
    with _gpu_lock_transaction():
        try:
            holder = _read_json(GPU_LOCK)
        except (OSError, ValueError):
            return
        if int(holder.get("pid", 0) or 0) == os.getpid():
            GPU_LOCK.unlink(missing_ok=True)


def stage_preflight() -> None:
    stamp = A0_RUN / "early_termination.json"
    while not stamp.is_file():
        _say("waiting for the A0 100k stop (early_termination.json) ...")
        time.sleep(120)
    record = _read_json(stamp)
    _say(
        f"A0 stopped at {record.get('terminated_at_samples'):,} samples "
        f"({record.get('terminated_utc')})"
    )
    subprocess.run(
        ["schtasks", "/Change", "/TN", WIDE15_TASK, "/DISABLE"],
        check=False, capture_output=True,
    )
    _say(f"disabled scheduled task {WIDE15_TASK} (AtLogOn resurrection guard)")
    for _ in range(30):
        probe = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "(Get-Process python,pythonw -ErrorAction SilentlyContinue | "
                "Where-Object { $_.Path -like '*vesuvius-c*' -and $_.Id -ne "
                # the venv pythonw is a redirector: the runner is its CHILD, so
                # exclude both our pid and the parent shim's pid or the probe
                # counts the runner's own shell forever (bounded 15 min wait)
                f"{os.getpid()} -and $_.Id -ne {os.getppid()}" + " }).Count",
            ],
            check=False, capture_output=True, text=True,
        )
        count = int((probe.stdout or "0").strip() or 0)
        if count == 0:
            break
        _say(f"waiting for {count} vesuvius python process(es) to exit ...")
        time.sleep(30)
    checkpoint = ROOT / _read_json(CONFIG_PATH)["initialization"]["checkpoint"]
    actual = _sha256(checkpoint)
    expected = _read_json(CONFIG_PATH)["initialization"]["sha256"]
    if actual != expected:
        raise SystemExit(f"released M7 SHA mismatch: {actual}")
    _say("released M7 verified")


def stage_trust_measurement() -> None:
    code = _run_logged(
        [sys.executable, "scripts/measure_trust_displacement.py"],
        "unified_ladder_trust.log",
    )
    if code != 0:
        raise SystemExit(f"trust displacement measurement failed ({code})")


def stage_reprojection() -> None:
    code = _run_logged(
        [sys.executable, "scripts/run_wide15_antialias_reprojection.py"],
        "unified_ladder_reprojection.log",
    )
    if code != 0:
        raise SystemExit(f"reprojection failed ({code})")


def stage_corpus_validate(config: dict[str, Any]) -> None:
    corpus = config["corpus"]
    soft_manifest = ROOT / corpus["soft_manifest"]
    hard_manifest = ROOT / corpus["hard_manifest"]
    with hard_manifest.open(encoding="utf-8") as stream:
        hard_ids = [json.loads(line)["patch_id"] for line in stream if line.strip()]
    with soft_manifest.open(encoding="utf-8") as stream:
        soft_ids = [json.loads(line)["patch_id"] for line in stream if line.strip()]
    if hard_ids != soft_ids:
        raise SystemExit(
            "soft manifest order diverges from hard manifest "
            f"({len(soft_ids):,} vs {len(hard_ids):,} rows)"
        )
    _say(f"manifest order identical across {len(soft_ids):,} rows")
    code = _run_logged(
        [
            sys.executable, "-m", "crossres_pred.voxel.cli", "validate-patches",
            "--patches", str(soft_manifest),
            "--expected-count", str(corpus["expected_rows"]),
            "--expected-train", str(corpus["expected_train"]),
            "--expected-val", str(corpus["expected_val"]),
            "--expected-test", "0",
            "--expected-source-corpora", str(corpus["expected_source_corpora"]),
            "--require-complete-antialias-lineage",
            "--require-hashes",
            "--workers", "8", "--max-cpu-threads", "16",
            "--summary-output",
            str(soft_manifest.parent / "validation_summary.json"),
        ],
        "unified_ladder_validate.log",
    )
    if code != 0:
        raise SystemExit(f"soft corpus validation failed ({code})")


def _eval_arm(
    arm_id: str,
    run_dir: Path,
    *,
    soft: bool,
    diagnostic: bool,
    fixed_milestone: int = 0,
) -> dict[str, Any]:
    summary_path = run_dir / "ladder_eval" / "arm_summary.json"
    if not summary_path.is_file():
        command = [
            sys.executable, "scripts/evaluate_ladder_arm.py",
            "--arm-id", arm_id, "--run-dir", str(run_dir),
        ]
        if soft:
            command.append("--soft-arm")
        if diagnostic:
            command.append("--diagnostic")
        if fixed_milestone:
            command += ["--fixed-milestone", str(fixed_milestone)]
        code = _run_logged(command, f"unified_ladder_eval_{arm_id}.log")
        if code != 0:
            raise SystemExit(f"evaluation of {arm_id} failed ({code})")
    return _read_json(summary_path)


def stage_a0_harness(config: dict[str, Any]) -> dict[str, Any]:
    summary = _eval_arm(
        "A0", A0_RUN, soft=False, diagnostic=True, fixed_milestone=50000
    )
    checkpoint_100k = A0_RUN / "checkpoint_milestone_00100000.pt"
    audit_dir = A0_RUN / "ladder_eval" / "audit_00100000"
    if checkpoint_100k.is_file() and not (audit_dir / "report.json").is_file():
        _run_logged(
            [
                sys.executable, "-m", "crossres_pred.voxel.cli",
                "audit-checkpoint",
                "--checkpoint", str(checkpoint_100k),
                "--patches", str(ROOT / config["evaluation"]["val_manifest"]),
                "--output", str(audit_dir),
                "--split", "val",
            ],
            "unified_ladder_eval_A0.log",
        )
    return summary


def train_arm(config: dict[str, Any], arm: dict[str, Any]) -> Path:
    output_root = ROOT / config["output_root"] / "arms"
    output = output_root / str(arm["arm_id"])
    milestone = output / (
        f"checkpoint_milestone_{config['base']['schedule']['stop_at_samples']:08d}.pt"
    )
    if milestone.is_file():
        _say(f"{arm['arm_id']}: 50k milestone already durable, skipping training")
        return output
    output.mkdir(parents=True, exist_ok=True)
    command = build_arm_command(config, arm, output)
    recipe = {
        "schema": "crossres-unified-ladder-arm-recipe-v1",
        "arm_id": arm["arm_id"],
        "delta": arm.get("delta"),
        "seed": arm["seed"],
        "objective": arm["objective"],
        "base": config["base"],
        "command": [str(part) for part in command],
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _atomic_json(output / "ladder_recipe.json", recipe)
    for attempt in range(1, MAX_TRAIN_ATTEMPTS + 1):
        launch = list(command)
        if (output / "run.json").exists():
            launch.append("--resume")
        log_path = LOGS / f"unified_ladder_train_{arm['arm_id']}.log"
        _say(
            f"{arm['arm_id']}: training attempt {attempt} "
            f"({'resume' if launch[-1] == '--resume' else 'fresh'})"
        )
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
            stream.flush()
            process = subprocess.Popen(
                [str(part) for part in launch],
                cwd=str(CROSSRES),
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            while True:
                if _milestone_durable(milestone):
                    _say(f"{arm['arm_id']}: 50k milestone durable; stopping training")
                    _kill_tree(process.pid)
                    process.wait(timeout=120)
                    _atomic_json(
                        output / "arm_termination.json",
                        {
                            "schema": "crossres-run-early-termination-v1",
                            "declared_total_samples": config["base"]["schedule"][
                                "max_train_samples"
                            ],
                            "terminated_at_samples": config["base"]["schedule"][
                                "stop_at_samples"
                            ],
                            "reason": "ladder arm: frozen 250k LR declaration, "
                                      "stop at the 50k milestone",
                            "milestone_sha256": _sha256(milestone),
                        },
                    )
                    return output
                code = process.poll()
                if code is not None:
                    _say(f"{arm['arm_id']}: training exited {code} before milestone")
                    break
                time.sleep(60)
    raise SystemExit(
        f"{arm['arm_id']}: training failed {MAX_TRAIN_ATTEMPTS} times without "
        f"reaching the 50k milestone"
    )


def config_arms_by_id(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {arm["arm_id"]: arm for arm in config["arms"]}


def standard_two_sided_dice(summary: dict[str, Any]) -> dict[str, Any]:
    """Return explicit ordinary Dice metrics, including for legacy summaries.

    Arm summaries created before evaluation-v2 did not promote pooled Dice or
    the fixed-T=0.25 row.  Their frozen checkpoint-audit report still contains
    both, so report generation reads that immutable result instead of rerunning
    inference.  Synthetic/partial summaries retain their known calibrated
    macro and use ``None`` for unavailable values rather than inventing them.
    """
    promoted = summary.get("standard_two_sided_dice")
    if promoted:
        return promoted

    result: dict[str, Any] = {
        "formula": "2*TP / (2*TP + FP + FN)",
        "calibrated": {
            "threshold": summary.get("calibrated_threshold"),
            "pooled_dice": summary.get("arm_score_pooled_dice"),
            "macro_scroll_dice": summary.get("arm_score_macro"),
        },
        "fixed_t025": {
            "threshold": 0.25,
            "pooled_dice": None,
            "macro_scroll_dice": None,
        },
    }
    run_dir = summary.get("run_dir")
    best = summary.get("best_milestone")
    if run_dir is None or best is None:
        return result
    report = (
        Path(run_dir)
        / "ladder_eval"
        / f"audit_{int(best):08d}"
        / "report.json"
    )
    if not report.is_file():
        return result
    sweep = _read_json(report)["sweep"]
    calibrated = sweep["selected"]
    fixed_t025 = min(
        sweep["points"], key=lambda row: abs(float(row["threshold"]) - 0.25)
    )

    def metrics(row: dict[str, Any]) -> dict[str, Any]:
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

    result["calibrated"] = metrics(calibrated)
    result["fixed_t025"] = metrics(fixed_t025)
    return result


def fixed_t025_gate(
    summary: dict[str, Any], gates: dict[str, Any]
) -> dict[str, Any] | None:
    """Return the locked T=0.25 gate, recovering it from legacy cached audits."""
    promoted = (summary.get("gate_verdicts") or {}).get("fixed_t025")
    if promoted is not None:
        return promoted
    run_dir = summary.get("run_dir")
    best = summary.get("best_milestone")
    if run_dir is None or best is None:
        return None
    eval_dir = Path(run_dir) / "ladder_eval"
    candidates = (
        eval_dir / f"joint_audit_{int(best):08d}.json",
        eval_dir / f"joint_audit_{int(best):08d}_t025.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        joint = _read_json(path)
        sweep = [
            row
            for row in joint.get("sweep", [])
            if abs(float(row["threshold"]) - 0.25) < 5.0e-3
        ]
        morphology = [
            row
            for row in joint.get("pherc1447_morphology", [])
            if abs(float(row["threshold"]) - 0.25) < 5.0e-3
        ]
        if sweep and morphology:
            from evaluate_ladder_arm import gate_verdict

            return gate_verdict(
                sweep[0],
                morphology[0]["cubes"],
                gates,
                soft_arm=bool(summary.get("soft_arm", False)),
            )
    return None


def build_final_ablation(
    summaries: dict[str, dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
    winner: str | None,
    arms_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """The operator-directed final ablation: winner vs vanilla M7, vs M7+TTA,
    and per-component isolation from the arms already measured.

    For a <=2-term winner the leave-one-out arms ARE the single-term arms; a
    >=3-component winner marks the missing leave-one-out runs explicitly.
    """

    def kaggle_leaderboard(summary: dict[str, Any]) -> float | None:
        block = (summary.get("kaggle") or {}).get("v14p2_val") or {}
        value = (block.get("calibrated") or {}).get("mean_leaderboard")
        return None if value is None else float(value)

    def row(summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "macro": float(summary["arm_score_macro"]),
            "calibrated_threshold": float(summary["calibrated_threshold"]),
            "standard_two_sided_dice": standard_two_sided_dice(summary),
            "kaggle_leaderboard_calibrated": kaggle_leaderboard(summary),
        }

    result: dict[str, Any] = {
        "baselines": {name: row(summary) for name, summary in baselines.items()},
    }
    if winner is None or winner not in summaries:
        result["status"] = "no-gate-passing-winner; ablation table awaits one"
        return result
    winner_row = row(summaries[winner])
    result["winner"] = {"arm_id": winner, **winner_row}
    for name, baseline in baselines.items():
        result[f"winner_minus_{name.lower()}_macro"] = winner_row["macro"] - float(
            baseline["arm_score_macro"]
        )
    isolation: list[dict[str, Any]] = []
    if "A0" in summaries and "B0" in summaries:
        isolation.append(
            {
                "component": "target_operator_soft_swap",
                "comparison": "B0 vs A0 (fixed-50k view is reported alongside)",
                "delta_macro": float(summaries["B0"]["arm_score_macro"])
                - float(summaries["A0"]["arm_score_macro"]),
            }
        )
    winner_terms = sorted(
        key
        for key in arms_by_id.get(winner, {}).get("objective", {})
        if key not in {"ce_weight", "dice_weight"}
    ) if winner != "D0" else sorted(
        key
        for arm_id in summaries
        if arm_id.startswith("C")
        for key in arms_by_id.get(arm_id, {}).get("objective", {})
        if key not in {"ce_weight", "dice_weight"}
    )
    for arm_id, summary in sorted(summaries.items()):
        if arm_id in {"A0", "B0", "B0r", winner} or not arm_id.startswith("C"):
            continue
        if "B0" in summaries:
            isolation.append(
                {
                    "component": arms_by_id.get(arm_id, {}).get("delta", arm_id),
                    "comparison": f"{arm_id} vs B0",
                    "delta_macro": float(summary["arm_score_macro"])
                    - float(summaries["B0"]["arm_score_macro"]),
                    "gates_calibrated": summary["gate_verdicts"]["calibrated"][
                        "passed"
                    ],
                }
            )
    term_arm_count = sum(
        1 for arm_id in summaries if arm_id.startswith("C")
    )
    if winner == "D0" and term_arm_count >= 3:
        isolation.append(
            {
                "component": "leave-one-out",
                "note": ">=3-component winner: leave-one-out arms beyond the "
                        "single-term set must be trained (config "
                        "d_phase.final_ablation.component_isolation)",
            }
        )
    result["component_isolation"] = isolation
    result["winner_terms"] = winner_terms
    return result


def stage_decision(
    config: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
    extras: dict[str, Any],
) -> None:
    margin, rms = tie_margin(
        summaries["B0"]["milestone_audits"], summaries["B0r"]["milestone_audits"]
    )
    winner, ranked = pick_winner(summaries)
    survivors = surviving_terms(summaries, margin)
    baselines = {}
    for name, directory in (
        ("M7RAW", "baselines/m7_raw"),
        ("M7TTA", "baselines/m7_tta"),
    ):
        baseline_summary = (
            ROOT / config["output_root"] / directory / "ladder_eval"
            / "arm_summary.json"
        )
        if baseline_summary.is_file():
            baselines[name] = _read_json(baseline_summary)
    gates_config = config["evaluation"]["gates"]

    def decision_arm_score(summary: dict[str, Any]) -> dict[str, Any]:
        t025_gate = fixed_t025_gate(summary, gates_config)
        return {
            "macro": summary["arm_score_macro"],
            "standard_two_sided_dice": standard_two_sided_dice(summary),
            "best_milestone": summary["best_milestone"],
            "calibrated_threshold": summary["calibrated_threshold"],
            "gates_calibrated": summary["gate_verdicts"]["calibrated"]["passed"],
            "gates_fixed": summary["gate_verdicts"]["fixed"]["passed"],
            "gates_fixed_t025": None if t025_gate is None else t025_gate["passed"],
            "recall_of_reference_calibrated": summary["gate_verdicts"][
                "calibrated"
            ]["recall_vs_reference"],
        }

    decision = {
        "schema": "crossres-unified-ladder-decision-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tie_margin": margin,
        "b0_replicate_rms": rms,
        "ranked_by_macro": ranked,
        "ranking_metric": "standard two-sided macro-scroll Dice",
        "winner_gate_passing": winner,
        "surviving_terms": survivors,
        "arm_scores": {
            arm_id: decision_arm_score(summary)
            for arm_id, summary in sorted(summaries.items())
        },
        **extras,
        "final_ablation": build_final_ablation(
            summaries, baselines, winner, config_arms_by_id(config)
        ),
        "next": "operator review; the final 100k+ run is NOT launched "
                "(user-chosen autonomy: auto through combo)",
    }
    output_root = ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_root / "decision.json", decision)
    headline = (
        f"tie margin {margin:.4f} (B0 replicate RMS {rms:.4f}); "
        f"winner (gate-passing): {winner or 'NONE'}; "
        f"surviving terms: {', '.join(survivors) or 'none'}"
    )
    header = (
        "| arm | ordinary Dice cal pooled | ordinary Dice cal macro "
        "| ordinary Dice @0.25 pooled | ordinary Dice @0.25 macro "
        "| milestone | cal T | gates@cal | gates@0.25 | gates@0.45 "
        "| recall-of-ref |"
    )
    lines = [
        "# Unified ladder decision table", "",
        headline, "",
        header,
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def format_score(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.4f}"

    for arm_id in ranked + (["A0"] if "A0" in summaries else []):
        summary = summaries[arm_id]
        verdicts = summary["gate_verdicts"]
        dice = standard_two_sided_dice(summary)
        fixed_t025 = fixed_t025_gate(summary, gates_config)
        lines.append(
            f"| {arm_id} | {format_score(dice['calibrated']['pooled_dice'])} "
            f"| {format_score(dice['calibrated']['macro_scroll_dice'])} "
            f"| {format_score(dice['fixed_t025']['pooled_dice'])} "
            f"| {format_score(dice['fixed_t025']['macro_scroll_dice'])} "
            f"| {summary['best_milestone']:,} "
            f"| {summary['calibrated_threshold']:.2f} "
            f"| {'PASS' if verdicts['calibrated']['passed'] else 'fail'} "
            f"| {('PASS' if fixed_t025['passed'] else 'fail') if fixed_t025 else 'n/a'} "
            f"| {'PASS' if verdicts['fixed']['passed'] else 'fail'} "
            f"| {verdicts['calibrated']['recall_vs_reference']:.3f} |"
        )
    (output_root / "ladder_table.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    _say(f"decision written: winner={winner or 'NONE'}, survivors={survivors}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all")
    parser.parse_args()
    config = _read_json(CONFIG_PATH)
    arms_by_id = {arm["arm_id"]: arm for arm in config["arms"]}

    if not _stage_done("preflight"):
        _mark("preflight", "running")
        stage_preflight()
        _mark("preflight", "complete")
    if not _stage_done("trust_measurement"):
        _mark("trust_measurement", "running")
        stage_trust_measurement()
        _mark("trust_measurement", "complete")
    # A0 harness first (~1h): exercises the whole eval pack end-to-end a day
    # before any arm needs it, while the reprojection has not yet claimed the
    # GPU for its long build.
    summaries: dict[str, dict[str, Any]] = {}
    if not _stage_done("a0_harness"):
        _mark("a0_harness", "running")
        summaries["A0"] = stage_a0_harness(config)
        _mark("a0_harness", "complete")
    else:
        summaries["A0"] = _read_json(A0_RUN / "ladder_eval" / "arm_summary.json")

    if not _stage_done("reprojection"):
        _mark("reprojection", "running")
        stage_reprojection()
        _mark("reprojection", "complete")
    if not _stage_done("corpus_validate"):
        _mark("corpus_validate", "running")
        stage_corpus_validate(config)
        _mark("corpus_validate", "complete")

    for arm_id in COMMITTED_ARMS:
        stage = f"arm_{arm_id}"
        arm = arms_by_id[arm_id]
        run_dir = ROOT / config["output_root"] / "arms" / arm_id
        if not _stage_done(stage):
            _mark(stage, "running")
            acquire_gpu_lock(f"wide15:{arm_id}")
            try:
                run_dir = train_arm(config, arm)
                summary = evaluate_trained_arm(
                    config,
                    arm_id,
                    run_dir,
                    soft=True,
                    diagnostic=arm_id in DIAGNOSTIC_ARMS,
                )
                if summary is None:
                    return 0
                summaries[arm_id] = summary
            finally:
                release_gpu_lock()
            _mark(stage, "complete", macro=summaries[arm_id]["arm_score_macro"])
        else:
            summaries[arm_id] = _read_json(
                run_dir / "ladder_eval" / "arm_summary.json"
            )
        if pause_after_evaluated_arm(
            config, arm_id, run_dir, summaries[arm_id]
        ):
            return 0

    from evaluate_ladder_arm import c2_trigger

    extras: dict[str, Any] = {}
    trigger = c2_trigger(
        summaries["C1"]["gate_verdicts"]["calibrated"],
        summaries["B0"]["gate_verdicts"]["calibrated"],
    )
    extras["c2_trigger"] = trigger
    if trigger["fires"] and not _stage_done("arm_C2"):
        _mark("arm_C2", "running")
        acquire_gpu_lock("wide15:C2")
        try:
            run_dir = train_arm(config, arms_by_id["C2"])
            summary = evaluate_trained_arm(
                config, "C2", run_dir, soft=True, diagnostic=False
            )
            if summary is None:
                return 0
            summaries["C2"] = summary
        finally:
            release_gpu_lock()
        _mark("arm_C2", "complete", macro=summaries["C2"]["arm_score_macro"])
    elif _stage_done("arm_C2"):
        summaries["C2"] = _read_json(
            ROOT / config["output_root"] / "arms/C2/ladder_eval/arm_summary.json"
        )

    diagnostic = summaries["B0"].get("ceiling_diagnostic") or {}
    lowest = diagnostic.get("lowest_fg_upper_passing_threshold")
    ceiling_fires = False
    if lowest is not None:
        fraction = float(
            diagnostic.get("band_fraction_p_ge", {}).get(f"{lowest:.2f}", 1.0)
        )
        ceiling_fires = fraction < 0.5
    extras["c5_ceiling_trigger"] = {
        "fires": ceiling_fires,
        "lowest_fg_upper_passing_threshold": lowest,
        "note": "crest sidecar build is NOT pre-built; if this fires the ladder "
                "records it for operator implementation rather than running "
                "unreviewed build code",
    }
    if ceiling_fires:
        _mark("arm_C5", "triggered_pending_crest_build")

    margin, _ = tie_margin(
        summaries["B0"]["milestone_audits"], summaries["B0r"]["milestone_audits"]
    )
    survivors = surviving_terms(summaries, margin)
    if len(survivors) >= 2 and not _stage_done("combo"):
        _mark("combo", "running", survivors=survivors)
        combo_objective: dict[str, Any] = {"ce_weight": 1.0, "dice_weight": 1.0}
        for arm_id in survivors:
            combo_objective.update(arms_by_id[arm_id]["objective"])
        combo_arm = {
            "arm_id": "D0",
            "delta": f"combo of surviving terms: {', '.join(survivors)}",
            "seed": 1203,
            "objective": combo_objective,
        }
        acquire_gpu_lock("wide15:D0")
        try:
            run_dir = train_arm(config, combo_arm)
            summary = evaluate_trained_arm(
                config, "D0", run_dir, soft=True, diagnostic=False
            )
            if summary is None:
                return 0
            summaries["D0"] = summary
        finally:
            release_gpu_lock()
        _mark("combo", "complete", macro=summaries["D0"]["arm_score_macro"])
    elif _stage_done("combo"):
        combo_summary = (
            ROOT / config["output_root"] / "arms/D0/ladder_eval/arm_summary.json"
        )
        if combo_summary.is_file():
            summaries["D0"] = _read_json(combo_summary)
    else:
        extras["combo"] = (
            f"skipped: {len(survivors)} surviving term arm(s); the winner arm "
            f"IS the combo"
        )

    if not _stage_done("decision"):
        _mark("decision", "running")
        stage_decision(config, summaries, extras)
        _mark("decision", "complete")
    _say("LADDER COMPLETE -- awaiting operator review before the final run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
