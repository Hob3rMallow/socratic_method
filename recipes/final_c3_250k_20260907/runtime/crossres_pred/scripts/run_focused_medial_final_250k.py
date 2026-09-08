#!/usr/bin/env python3
"""Durable R_sep3_tail20 gate and exact full-corpus release run.

This coordinator is intentionally narrower than ``run_release_ladder.py``.  It
waits until the unified B0r/C1/C3 campaign has completed C3's frozen
evaluation, trains only the preregistered medial-tail candidate, applies the
already-frozen T=0.25 blob gate, and then retrains the selected release recipe
from the released M7 initializer over every one of the 249,836 accepted atlas
rows exactly once.  Every long stage is journalled and safe to resume after a
process or machine restart.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from run_release_ladder import build_release_command
from run_unified_ladder import (
    CROSSRES,
    LOGS,
    MAX_TRAIN_ATTEMPTS,
    ROOT,
    _atomic_json,
    _read_json,
    _sha256,
    acquire_gpu_lock,
    release_gpu_lock,
)

DEFAULT_CONFIG = CROSSRES / "configs/focused_medial_final_250k_20260904.json"
JOURNAL_SCHEMA = "crossres-focused-medial-final-250k-journal-v1"
SELECTION_SCHEMA = "crossres-focused-medial-final-250k-selection-v1"
RECIPE_SCHEMA = "crossres-focused-medial-final-250k-recipe-v1"
STANDARD_DICE_FORMULA = "2*TP / (2*TP + FP + FN)"
TAIL_GATE_CONTRACT = "release-t025-blob-candidate-gate-v1"


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: malformed JSON line {number}") from error
        if not isinstance(value, dict):
            raise TypeError(f"{path}: line {number} is not an object")
        rows.append(value)
    return rows


def _history_samples(row: dict[str, Any]) -> int:
    training = row.get("train")
    if not isinstance(training, dict) or "cumulative_samples" not in training:
        raise ValueError("history row has no train.cumulative_samples")
    value = float(training["cumulative_samples"])
    rounded = round(value)
    if not math.isfinite(value) or abs(value - rounded) > 1.0e-6 or rounded < 0:
        raise ValueError(f"invalid cumulative sample count: {value!r}")
    return int(rounded)


def _validation_score(validation: dict[str, Any]) -> float:
    for key in ("calibrated_macro_scroll_dice", "macro_scroll_dice", "dice"):
        if key in validation:
            score = float(validation[key])
            if not math.isfinite(score):
                raise ValueError(f"non-finite validation score {key}={score!r}")
            return score
    raise ValueError("validation record has no Dice score")


def standard_dice_from_validation(validation: dict[str, Any]) -> dict[str, Any]:
    """Return ordinary symmetric Dice at calibrated and fixed T=0.25."""

    return {
        "formula": STANDARD_DICE_FORMULA,
        "calibrated": {
            "threshold": float(validation["calibrated_threshold"]),
            "pooled_dice": float(validation["calibrated_dice"]),
            "macro_scroll_dice": float(validation["calibrated_macro_scroll_dice"]),
            "precision": float(validation["calibrated_precision"]),
            "recall": float(validation["calibrated_recall"]),
        },
        "fixed_t025": {
            "threshold": 0.25,
            "pooled_dice": float(validation["threshold/0.25/dice"]),
            "macro_scroll_dice": float(validation["threshold/0.25/macro_scroll_dice"]),
            "precision": float(validation["threshold/0.25/precision"]),
            "recall": float(validation["threshold/0.25/recall"]),
        },
    }


def validation_at_milestone(run_dir: Path, samples: int) -> dict[str, Any]:
    matches = [
        row
        for row in _read_jsonl(run_dir / "history.jsonl")
        if _history_samples(row) == samples
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{run_dir}: expected one history row at {samples:,}, found {len(matches)}"
        )
    validation = matches[0].get("val")
    if not isinstance(validation, dict):
        raise TypeError(f"{run_dir}: history validation at {samples:,} is malformed")
    return validation


def count_manifest_rows(path: Path) -> int:
    rows = 0
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            rows += block.count(b"\n")
    if path.stat().st_size and rows == 0:
        return 1
    with path.open("rb") as stream:
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) != b"\n":
            rows += 1
    return rows


def expected_final_snapshots(total_samples: int, interval: int) -> list[int]:
    if total_samples <= 0 or interval <= 0:
        raise ValueError("sample counts must be positive")
    return [*range(interval, total_samples, interval), total_samples]


def training_commit_state(
    run_dir: Path,
    *,
    total_samples: int,
    expected_snapshots: list[int] | None = None,
) -> dict[str, Any]:
    """Audit the trainer's durable commit, not merely its early snapshot."""

    reasons: list[str] = []
    final_milestone = run_dir / f"checkpoint_milestone_{total_samples:08d}.pt"
    last_checkpoint = run_dir / "checkpoint_last.pt"
    history_path = run_dir / "history.jsonl"
    snapshot_index = run_dir / "checkpoint_milestones.json"
    for label, path in (
        ("final milestone", final_milestone),
        ("checkpoint_last", last_checkpoint),
        ("history", history_path),
        ("snapshot index", snapshot_index),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            reasons.append(f"missing {label}: {path}")

    rows: list[dict[str, Any]] = []
    history_samples: list[int] = []
    if history_path.is_file():
        rows = _read_jsonl(history_path)
        history_samples = [_history_samples(row) for row in rows]
        if history_samples != sorted(set(history_samples)):
            reasons.append("history cumulative sample counts are not strict and unique")
        if history_samples.count(total_samples) != 1:
            reasons.append(
                f"history has {history_samples.count(total_samples)} rows at "
                f"{total_samples:,}"
            )
        if not history_samples or history_samples[-1] != total_samples:
            reasons.append(f"last history row is not the {total_samples:,} commit")

    indexed: list[int] = []
    if snapshot_index.is_file():
        value = _read_json(snapshot_index)
        records = value.get("records")
        if not isinstance(records, list):
            reasons.append("snapshot index records are malformed")
        else:
            indexed = [int(row["actual_samples"]) for row in records]
            if total_samples not in indexed:
                reasons.append(f"snapshot index omits {total_samples:,}")
            for row in records:
                checkpoint = run_dir / str(row["checkpoint"])
                if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
                    reasons.append(f"indexed snapshot is missing: {checkpoint}")

    if expected_snapshots is not None:
        if indexed != expected_snapshots:
            reasons.append(
                f"snapshot index {indexed!r} does not equal expected "
                f"{expected_snapshots!r}"
            )
        missing_files = [
            samples
            for samples in expected_snapshots
            if not (run_dir / f"checkpoint_milestone_{samples:08d}.pt").is_file()
        ]
        if missing_files:
            reasons.append(f"milestone files missing for {missing_files!r}")

    return {
        "complete": not reasons,
        "reasons": reasons,
        "history_rows": len(rows),
        "history_samples": history_samples,
        "snapshot_samples": indexed,
        "final_milestone": str(final_milestone),
        "checkpoint_last": str(last_checkpoint),
    }


def prerequisite_reasons(
    focus: dict[str, Any],
    unified_journal: dict[str, Any],
    summary_presence: dict[str, bool],
) -> list[str]:
    prerequisites = focus["prerequisites"]
    required_arm = str(prerequisites["required_pause_after_arm"])
    stages = unified_journal.get("stages")
    if not isinstance(stages, dict):
        return ["unified journal has no stages object"]
    reasons: list[str] = []
    pause = stages.get("operator_review_pause") or {}
    if pause.get("status") != "complete" or pause.get("after_arm") != required_arm:
        reasons.append(
            f"unified operator pause after evaluated {required_arm} is pending"
        )
    for arm_id in prerequisites["required_frozen_summaries"]:
        arm_stage = stages.get(f"arm_{arm_id}") or {}
        if arm_stage.get("status") != "complete":
            reasons.append(f"unified arm {arm_id} is not complete")
        if not summary_presence.get(str(arm_id), False):
            reasons.append(f"frozen summary for {arm_id} is missing")
    return reasons


def select_release_arm(
    focus: dict[str, Any], candidate_manifest: dict[str, Any]
) -> tuple[str, str]:
    """Apply the fixed gate literally; aggregate/criterion disagreement is fatal."""

    candidate = candidate_manifest.get("candidate")
    if not isinstance(candidate, dict):
        raise TypeError("candidate diagnostic has no candidate identity")
    release = focus["release"]
    if candidate.get("arm") != release["focused_arm"]:
        raise ValueError("candidate diagnostic names the wrong arm")
    if int(candidate.get("milestone_samples", -1)) != int(
        release["focused_milestone_samples"]
    ):
        raise ValueError("candidate diagnostic names the wrong milestone")
    if not math.isclose(
        float(candidate.get("threshold", -1.0)),
        float(release["threshold"]),
        abs_tol=1.0e-12,
    ):
        raise ValueError("candidate diagnostic does not use fixed T=0.25")

    gate = candidate_manifest.get("candidate_gate")
    if not isinstance(gate, dict) or gate.get("contract") != TAIL_GATE_CONTRACT:
        raise ValueError("candidate diagnostic has no recognized preregistered gate")
    criteria = gate.get("criteria")
    if not isinstance(criteria, dict) or not criteria:
        raise ValueError("candidate gate criteria are missing")
    criterion_passes = []
    for name, criterion in criteria.items():
        if not isinstance(criterion, dict) or not isinstance(
            criterion.get("passed"), bool
        ):
            raise TypeError(f"candidate criterion {name!r} has no boolean verdict")
        criterion_passes.append(bool(criterion["passed"]))
    all_passed = all(criterion_passes)
    if not isinstance(gate.get("passed"), bool) or bool(gate["passed"]) != all_passed:
        raise ValueError("candidate gate aggregate disagrees with its criteria")
    if all_passed:
        return str(
            release["focused_arm"]
        ), "all preregistered fixed-T=0.25 criteria passed"
    return str(
        release["baseline_arm"]
    ), "one or more preregistered fixed-T=0.25 criteria failed"


def build_final_training_spec(
    focus: dict[str, Any],
    release_config: dict[str, Any],
    selected_arm: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct the exact one-pass run while retaining the selected recipe."""

    runtime = copy.deepcopy(release_config)
    final = focus["final_training"]
    resources = focus["resources"]
    total = int(final["samples"])
    interval = int(final["samples_per_validation_interval"])
    snapshots = expected_final_snapshots(total, interval)
    if snapshots != [int(value) for value in final["snapshot_samples"]]:
        raise ValueError("final snapshot schedule is not the exact interval partition")
    if math.ceil(total / interval) != int(final["evaluation_intervals"]):
        raise ValueError("final evaluation interval count is inconsistent")

    runtime["corpus"]["manifest"] = focus["corpus"]["manifest"]
    schedule = runtime["base"]["schedule"]
    schedule["samples_per_epoch"] = interval
    schedule["batch_size"] = int(resources["batch_size"])
    schedule["accumulate"] = int(resources["gradient_accumulation"])
    runtime["base"]["num_workers"] = int(resources["data_loader_workers"])
    runtime["base"]["max_cpu_threads"] = int(resources["maximum_cpu_threads"])
    runtime["base"]["device"] = "cuda"
    final_arm = {
        "arm_id": str(final["arm_id"]),
        "delta": f"full exact corpus pass using selected recipe {selected_arm['arm_id']}",
        "samples": total,
        "snapshot_samples": snapshots,
        "finish_epoch_after_final_milestone": True,
        "objective_overrides": copy.deepcopy(
            selected_arm.get("objective_overrides", {})
        ),
        "optimizer_overrides": copy.deepcopy(
            selected_arm.get("optimizer_overrides", {})
        ),
        "selected_source_arm": str(selected_arm["arm_id"]),
    }
    return runtime, final_arm


def select_internal_best(
    initial_validation: dict[str, Any], history_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Reproduce the trainer's guarded, strict-greater-than best selection."""

    initial_score = _validation_score(initial_validation)
    best_score = initial_score
    selected: dict[str, Any] | None = None
    trajectory: list[dict[str, Any]] = []
    for row in history_rows:
        validation = row.get("val")
        if not isinstance(validation, dict):
            raise TypeError("history row has malformed validation")
        score = _validation_score(validation)
        eligible_value = validation.get("checkpoint_eligible", 0.0)
        eligible = bool(float(eligible_value))
        became_best = eligible and score > best_score
        samples = _history_samples(row)
        if became_best:
            best_score = score
            selected = {
                "milestone_samples": samples,
                "epoch": int(row["epoch"]),
                "score": score,
                "validation": validation,
            }
        trajectory.append(
            {
                "milestone_samples": samples,
                "epoch": int(row["epoch"]),
                "score": score,
                "checkpoint_eligible": eligible,
                "became_best": became_best,
                "standard_two_sided_dice": standard_dice_from_validation(validation),
            }
        )
    if selected is None:
        raise RuntimeError(
            "no guarded trained checkpoint beat the fresh M7 initializer; "
            "there is no internally selected trained milestone to freeze"
        )
    return {
        **selected,
        "initial_score": initial_score,
        "selection_rule": (
            "start from fresh-M7 calibrated macro Dice; among checkpoint-eligible "
            "trained rows, replace only on strict improvement"
        ),
        "trajectory": trajectory,
    }


def _arm_by_id(config: dict[str, Any], arm_id: str) -> dict[str, Any]:
    matches = [arm for arm in config["arms"] if arm.get("arm_id") == arm_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one release arm {arm_id!r}")
    return matches[0]


def _option(command: list[str], flag: str) -> str:
    indexes = [index for index, value in enumerate(command) if value == flag]
    if len(indexes) != 1 or indexes[0] + 1 >= len(command):
        raise ValueError(f"command does not contain exactly one {flag}")
    return command[indexes[0] + 1]


class Coordinator:
    def __init__(self, config_path: Path, *, poll_seconds: int = 300) -> None:
        self.config_path = config_path.resolve()
        self.config = _read_json(self.config_path)
        self.config_sha256 = _sha256(self.config_path)
        self.journal_path = _resolve(self.config["journal"])
        self.log_path = _resolve(self.config["log"])
        self.poll_seconds = poll_seconds

    def say(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [focused-final] {message}"
        print(line, flush=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def journal(self) -> dict[str, Any]:
        if self.journal_path.is_file():
            value = _read_json(self.journal_path)
            if value.get("schema") != JOURNAL_SCHEMA:
                raise ValueError("focused coordinator journal schema changed")
            binding = value.get("config") or {}
            if binding.get("sha256") != self.config_sha256:
                raise ValueError(
                    "focused coordinator config changed after the journal was sealed"
                )
            return value
        return {
            "schema": JOURNAL_SCHEMA,
            "created_at_utc": _now(),
            "config": {
                "path": str(self.config_path),
                "sha256": self.config_sha256,
            },
            "stages": {},
        }

    def mark(self, stage: str, status: str, **extra: Any) -> None:
        journal = self.journal()
        journal["stages"][stage] = {
            "status": status,
            "at_utc": _now(),
            **extra,
        }
        _atomic_json(self.journal_path, journal)
        self.say(f"stage {stage}: {status}")

    def preflight(self) -> dict[str, Any]:
        focus = self.config
        if focus.get("schema") != "crossres-focused-medial-final-250k-v1":
            raise ValueError("focused campaign config schema changed")

        verified: dict[str, str] = {}

        def verify(path_value: str, expected: str, label: str) -> Path:
            path = _resolve(path_value)
            if not path.is_file():
                raise FileNotFoundError(f"{label} missing: {path}")
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(
                    f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
                )
            verified[str(path)] = actual
            return path

        prerequisites = focus["prerequisites"]
        release = focus["release"]
        corpus = focus["corpus"]
        verify(
            prerequisites["unified_config"],
            prerequisites["unified_config_sha256"],
            "unified config",
        )
        release_config_path = verify(
            release["config"], release["config_sha256"], "release config"
        )
        summary_path = verify(
            corpus["summary"], corpus["summary_sha256"], "release corpus summary"
        )
        manifest_path = verify(
            corpus["manifest"], corpus["manifest_sha256"], "release corpus manifest"
        )
        preservation_path = verify(
            prerequisites["b0_preservation_manifest"],
            prerequisites["b0_preservation_manifest_sha256"],
            "B0 preservation manifest",
        )
        verify(
            release["baseline_diagnostic_manifest"],
            release["baseline_diagnostic_manifest_sha256"],
            "fixed-T=0.25 baseline diagnostic",
        )
        for path_value, expected in focus["code_pins"].items():
            verify(path_value, expected, f"pinned code {path_value}")

        summary = _read_json(summary_path)
        exact_rows = int(corpus["exact_rows"])
        if (
            summary.get("state") != "complete"
            or int(summary.get("patches", -1)) != exact_rows
        ):
            raise ValueError(
                "release corpus summary is not complete at the exact row count"
            )
        actual_rows = count_manifest_rows(manifest_path)
        if actual_rows != exact_rows:
            raise ValueError(
                f"release manifest has {actual_rows:,} rows, expected {exact_rows:,}"
            )
        source_rows = {
            str(record["scroll_id"]): int(record["patches"])
            for record in summary["sources"].values()
        }
        if sum(source_rows.values()) != exact_rows:
            raise ValueError("release corpus source counts do not sum to the manifest")

        release_config = _read_json(release_config_path)
        if release_config["corpus"]["manifest"] != corpus["manifest"]:
            raise ValueError("release recipe points at a different corpus manifest")
        baseline = _arm_by_id(release_config, str(release["baseline_arm"]))
        tail = _arm_by_id(release_config, str(release["focused_arm"]))
        if baseline.get("objective_overrides", {}) != {}:
            raise ValueError("release baseline objective unexpectedly changed")
        expected_tail = {
            "separation_weight": 3.0,
            "medial_tail_floor_weight": 0.015625,
            "medial_tail_floor_probability": 0.3,
            "medial_tail_bottom_fraction": 0.2,
        }
        if tail.get("objective_overrides") != expected_tail:
            raise ValueError("the preregistered R_sep3_tail20 objective changed")
        if int(tail["samples"]) != int(release["focused_milestone_samples"]):
            raise ValueError("the preregistered tail endpoint changed")

        initialization = release_config["initialization"]
        verify(
            initialization["checkpoint"],
            initialization["sha256"],
            "released M7 initializer",
        )
        connectivity_state = _resolve(
            Path(release_config["connectivity"]["state_output"])
            / "connectivity_state.json"
        )
        state = _read_json(connectivity_state)
        identity = state.get("identity") or {}
        if state.get("state") != "complete":
            raise ValueError("dynamic medial connectivity state is incomplete")
        if identity.get("training_manifest_sha256") != corpus["manifest_sha256"]:
            raise ValueError("connectivity state is bound to a different corpus")

        preservation = _read_json(preservation_path)
        if (
            preservation.get("arm_id") != "B0"
            or int(preservation.get("milestone_samples", -1)) != 30000
        ):
            raise ValueError("B0 preservation manifest names the wrong checkpoint")
        expected_bytes = int(preservation["bytes"])
        expected_hash = str(preservation["sha256"])
        for role in ("source", "preserved_copy"):
            path = _resolve(preservation[role])
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise ValueError(f"B0 {role} byte length changed")
            if _sha256(path) != expected_hash:
                raise ValueError(f"B0 {role} SHA-256 changed")

        final = focus["final_training"]
        if int(final["samples"]) != exact_rows:
            raise ValueError("final training does not cover the exact corpus")
        expected = expected_final_snapshots(
            exact_rows, int(final["samples_per_validation_interval"])
        )
        if expected != [int(value) for value in final["snapshot_samples"]]:
            raise ValueError("final snapshot schedule changed")
        if len(expected) != int(final["evaluation_intervals"]):
            raise ValueError("final interval count changed")
        resources = focus["resources"]
        if (
            int(resources["gpu_power_limit_watts"]) != 600
            or bool(resources["startup_throttle"])
            or int(resources["batch_size"]) != 3
            or int(resources["gradient_accumulation"]) != 8
            or int(resources["data_loader_workers"]) != 2
            or int(resources["maximum_cpu_threads"]) != 16
        ):
            raise ValueError("focused resource contract changed")

        return {
            "verified_files": verified,
            "corpus_rows": exact_rows,
            "corpus_sources": source_rows,
            "manifest": str(manifest_path),
            "connectivity_state": str(connectivity_state),
            "b0_preserved_sha256": expected_hash,
        }

    def wait_for_unified(self, *, no_wait: bool = False) -> dict[str, dict[str, Any]]:
        focus = self.config
        unified_config = _read_json(_resolve(focus["prerequisites"]["unified_config"]))
        output_root = _resolve(unified_config["output_root"])
        required = [
            str(value) for value in focus["prerequisites"]["required_frozen_summaries"]
        ]
        paths = {
            arm_id: output_root / "arms" / arm_id / "ladder_eval" / "arm_summary.json"
            for arm_id in required
        }
        while True:
            journal_path = _resolve(focus["prerequisites"]["unified_journal"])
            unified_journal = _read_json(journal_path) if journal_path.is_file() else {}
            reasons = prerequisite_reasons(
                focus,
                unified_journal,
                {arm_id: path.is_file() for arm_id, path in paths.items()},
            )
            if not reasons:
                break
            self.mark("unified_prerequisites", "waiting", pending=reasons)
            if no_wait:
                raise RuntimeError(
                    "unified prerequisites are pending: " + "; ".join(reasons)
                )
            self.say("waiting for B0r/C1/C3: " + "; ".join(reasons))
            time.sleep(self.poll_seconds)

        summaries: dict[str, dict[str, Any]] = {}
        facts: dict[str, Any] = {}
        for arm_id, path in paths.items():
            summary = _read_json(path)
            if summary.get("arm_id") != arm_id:
                raise ValueError(f"{path}: frozen summary names the wrong arm")
            standard = summary.get("standard_two_sided_dice")
            if not isinstance(standard, dict) or not all(
                key in standard for key in ("calibrated", "fixed_t025")
            ):
                raise ValueError(f"{path}: standard two-sided Dice is missing")
            summaries[arm_id] = summary
            facts[arm_id] = {
                "path": str(path),
                "sha256": _sha256(path),
                "best_milestone": int(summary["best_milestone"]),
                "standard_two_sided_dice": standard,
            }
        self.mark("unified_prerequisites", "complete", summaries=facts)
        return summaries

    def _run_logged(self, command: list[str], log_name: str) -> None:
        log_path = LOGS / log_name
        self.say("+ " + " ".join(str(value) for value in command) + f" -> {log_name}")
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"\n=== launch {_now()} ===\n")
            stream.flush()
            result = subprocess.run(
                [str(value) for value in command],
                cwd=str(CROSSRES),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"command exited {result.returncode}; inspect {log_path}"
            )

    def _write_recipe(
        self,
        output: Path,
        *,
        arm: dict[str, Any],
        runtime: dict[str, Any],
        command: list[str],
        selected_source_arm: str,
    ) -> None:
        recipe_path = output / "ladder_recipe.json"
        recipe = {
            "schema": RECIPE_SCHEMA,
            "config": {
                "path": str(self.config_path),
                "sha256": self.config_sha256,
            },
            "arm_id": arm["arm_id"],
            "selected_source_arm": selected_source_arm,
            "samples": int(arm["samples"]),
            "snapshot_samples": [int(value) for value in arm["snapshot_samples"]],
            "finish_epoch_after_final_milestone": True,
            "objective_overrides": arm.get("objective_overrides", {}),
            "optimizer_overrides": arm.get("optimizer_overrides", {}),
            "base": runtime["base"],
            "corpus": runtime["corpus"],
            "initialization": runtime["initialization"],
            "command": [str(value) for value in command],
            "written_at_utc": _now(),
        }
        if recipe_path.is_file():
            existing = _read_json(recipe_path)
            for key in (
                "arm_id",
                "selected_source_arm",
                "samples",
                "snapshot_samples",
                "objective_overrides",
                "optimizer_overrides",
                "command",
            ):
                if existing.get(key) != recipe.get(key):
                    raise ValueError(
                        f"{recipe_path}: sealed recipe field {key} changed"
                    )
            return
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(recipe_path, recipe)

    def run_training(
        self,
        *,
        label: str,
        runtime: dict[str, Any],
        arm: dict[str, Any],
        output: Path,
        connectivity_state: Path,
        log_name: str,
        selected_source_arm: str,
    ) -> dict[str, Any]:
        total = int(arm["samples"])
        snapshots = [int(value) for value in arm["snapshot_samples"]]
        state = training_commit_state(
            output, total_samples=total, expected_snapshots=snapshots
        )
        if state["complete"]:
            self.say(f"{label}: exact training commit already durable")
            return state

        objective = {
            **runtime["base"]["objective"],
            **arm.get("objective_overrides", {}),
        }
        dynamic_state = (
            connectivity_state
            if float(objective.get("dynamic_medial_connectivity_weight", 0.0)) > 0.0
            else None
        )
        command = build_release_command(
            runtime, arm, output, connectivity_state=dynamic_state
        )
        self._write_recipe(
            output,
            arm=arm,
            runtime=runtime,
            command=command,
            selected_source_arm=selected_source_arm,
        )
        log_path = LOGS / log_name
        for attempt in range(1, MAX_TRAIN_ATTEMPTS + 1):
            launch = list(command)
            if (output / "run.json").is_file():
                launch.append("--resume")
            self.say(f"{label}: training attempt {attempt}/{MAX_TRAIN_ATTEMPTS}")
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"\n=== launch {_now()} ===\n")
                stream.write(" ".join(str(value) for value in launch) + "\n")
                stream.flush()
                process = subprocess.Popen(
                    [str(value) for value in launch],
                    cwd=str(CROSSRES),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                next_heartbeat = time.monotonic() + 10 * 60
                while process.poll() is None:
                    time.sleep(60)
                    if time.monotonic() >= next_heartbeat:
                        rows = _read_jsonl(output / "history.jsonl")
                        committed = _history_samples(rows[-1]) if rows else 0
                        self.say(
                            f"{label}: alive; last validated commit "
                            f"{committed:,}/{total:,}"
                        )
                        next_heartbeat = time.monotonic() + 10 * 60
                code = int(process.returncode or 0)
            state = training_commit_state(
                output, total_samples=total, expected_snapshots=snapshots
            )
            if code == 0 and state["complete"]:
                self.say(f"{label}: exact {total:,}-sample commit complete")
                return state
            self.say(
                f"{label}: attempt {attempt} exited {code}; "
                + "; ".join(state["reasons"])
            )
        raise RuntimeError(f"{label}: failed {MAX_TRAIN_ATTEMPTS} training attempts")

    def train_tail(
        self, release_config: dict[str, Any], connectivity_state: Path
    ) -> Path:
        arm_id = str(self.config["release"]["focused_arm"])
        arm = _arm_by_id(release_config, arm_id)
        output = _resolve(Path(release_config["output_root"]) / "arms" / arm_id)
        state = training_commit_state(
            output,
            total_samples=int(arm["samples"]),
            expected_snapshots=[int(value) for value in arm["snapshot_samples"]],
        )
        if not state["complete"]:
            self.mark("tail_training", "running", arm_id=arm_id)
            acquire_gpu_lock(f"focused:{arm_id}")
            try:
                state = self.run_training(
                    label=arm_id,
                    runtime=release_config,
                    arm=arm,
                    output=output,
                    connectivity_state=connectivity_state,
                    log_name=f"focused_train_{arm_id}.log",
                    selected_source_arm=arm_id,
                )
            finally:
                release_gpu_lock()
        self.mark("tail_training", "complete", arm_id=arm_id, commit=state)
        return output

    def render_tail_panels(self, tail_dir: Path) -> Path:
        release = self.config["release"]
        arm_id = str(release["focused_arm"])
        wanted = [int(value) for value in release["focused_panel_milestones"]]
        manifest_path = tail_dir / "ladder_eval/panels/manifest.json"
        valid = False
        if manifest_path.is_file():
            manifest = _read_json(manifest_path)
            records = manifest.get("records") or []
            valid = (
                manifest.get("arm_id") == arm_id
                and [int(row["samples"]) for row in records] == wanted
            )
        if not valid:
            self.mark("tail_panels", "running", milestones=wanted)
            command = [
                sys.executable,
                "scripts/render_release_milestone_panels.py",
                "--arm",
                arm_id,
                "--milestones",
                *[str(value) for value in wanted],
                "--device",
                "cuda",
            ]
            acquire_gpu_lock(f"focused:{arm_id}:panels")
            try:
                self._run_logged(command, f"focused_panels_{arm_id}.log")
            finally:
                release_gpu_lock()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"tail panel manifest missing: {manifest_path}")
        self.mark(
            "tail_panels",
            "complete",
            manifest=str(manifest_path),
            sha256=_sha256(manifest_path),
        )
        return manifest_path

    def diagnose_tail(self, tail_dir: Path) -> dict[str, Any]:
        release = self.config["release"]
        arm_id = str(release["focused_arm"])
        milestone = int(release["focused_milestone_samples"])
        manifest_path = _resolve(release["focused_diagnostic_manifest"])
        valid = False
        if manifest_path.is_file():
            try:
                manifest = _read_json(manifest_path)
                candidate = manifest.get("candidate") or {}
                validation = manifest.get("validation") or {}
                valid = (
                    candidate.get("arm") == arm_id
                    and int(candidate.get("milestone_samples", -1)) == milestone
                    and math.isclose(
                        float(candidate.get("threshold", -1)), 0.25, abs_tol=1.0e-12
                    )
                    and isinstance(validation.get("standard_two_sided_dice"), dict)
                    and isinstance(manifest.get("candidate_gate"), dict)
                )
            except (KeyError, TypeError, ValueError):
                valid = False
        if not valid:
            self.mark("tail_diagnostic", "running", threshold=0.25)
            self._run_logged(
                [
                    sys.executable,
                    "scripts/diagnose_release_t025_blobs.py",
                    "--arm",
                    arm_id,
                    "--milestone",
                    str(milestone),
                    "--output",
                    str(manifest_path.parent),
                ],
                f"focused_diagnostic_{arm_id}.log",
            )
        manifest = _read_json(manifest_path)
        select_release_arm(self.config, manifest)
        checkpoint = tail_dir / f"checkpoint_milestone_{milestone:08d}.pt"
        if manifest["candidate"].get("checkpoint_sha256") != _sha256(checkpoint):
            raise ValueError("tail diagnostic is bound to a different checkpoint")
        self.mark(
            "tail_diagnostic",
            "complete",
            manifest=str(manifest_path),
            sha256=_sha256(manifest_path),
            gate_passed=bool(manifest["candidate_gate"]["passed"]),
        )
        return manifest

    def seal_selection(
        self,
        *,
        release_config: dict[str, Any],
        candidate_manifest: dict[str, Any],
        unified_summaries: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        path = _resolve(self.config["selection_manifest"])
        candidate_path = _resolve(self.config["release"]["focused_diagnostic_manifest"])
        candidate_hash = _sha256(candidate_path)
        baseline_path = _resolve(self.config["release"]["baseline_diagnostic_manifest"])
        baseline_hash = _sha256(baseline_path)
        selected_id, reason = select_release_arm(self.config, candidate_manifest)
        selected_arm = _arm_by_id(release_config, selected_id)

        baseline_dir = _resolve(
            Path(release_config["output_root"])
            / "arms"
            / str(self.config["release"]["baseline_arm"])
        )
        tail_dir = _resolve(
            Path(release_config["output_root"])
            / "arms"
            / str(self.config["release"]["focused_arm"])
        )
        baseline_validation = validation_at_milestone(
            baseline_dir, int(self.config["release"]["baseline_milestone_samples"])
        )
        tail_validation = validation_at_milestone(
            tail_dir, int(self.config["release"]["focused_milestone_samples"])
        )
        objective = {
            **release_config["base"]["objective"],
            **selected_arm.get("objective_overrides", {}),
        }
        optimizer = {
            **release_config["base"]["optimizer"],
            **selected_arm.get("optimizer_overrides", {}),
        }
        summary_facts = {
            arm_id: {
                "path": str(
                    _resolve(
                        Path(
                            _read_json(
                                _resolve(self.config["prerequisites"]["unified_config"])
                            )["output_root"]
                        )
                        / "arms"
                        / arm_id
                        / "ladder_eval/arm_summary.json"
                    )
                ),
                "best_milestone": int(summary["best_milestone"]),
                "standard_two_sided_dice": summary["standard_two_sided_dice"],
            }
            for arm_id, summary in unified_summaries.items()
        }
        value = {
            "schema": SELECTION_SCHEMA,
            "sealed_at_utc": _now(),
            "focused_config": {
                "path": str(self.config_path),
                "sha256": self.config_sha256,
            },
            "decision": {
                "selected_arm": selected_id,
                "reason": reason,
                "threshold": 0.25,
                "candidate_gate_passed": bool(
                    candidate_manifest["candidate_gate"]["passed"]
                ),
                "composite_used_for_selection": False,
            },
            "candidate": {
                "manifest": str(candidate_path),
                "sha256": candidate_hash,
                "identity": candidate_manifest["candidate"],
                "gate": candidate_manifest["candidate_gate"],
                "standard_two_sided_dice": standard_dice_from_validation(
                    tail_validation
                ),
            },
            "baseline": {
                "manifest": str(baseline_path),
                "sha256": baseline_hash,
                "standard_two_sided_dice": standard_dice_from_validation(
                    baseline_validation
                ),
            },
            "selected_recipe": {
                "source_arm": selected_id,
                "objective": objective,
                "optimizer": optimizer,
                "initialization": release_config["initialization"],
            },
            "unified_frozen_summaries": summary_facts,
            "corpus": self.config["corpus"],
            "code_pins": self.config["code_pins"],
        }
        if path.is_file():
            existing = _read_json(path)
            if existing.get("schema") != SELECTION_SCHEMA:
                raise ValueError("existing final selection manifest has wrong schema")
            checks = (
                existing.get("focused_config", {}).get("sha256") == self.config_sha256,
                existing.get("candidate", {}).get("sha256") == candidate_hash,
                existing.get("baseline", {}).get("sha256") == baseline_hash,
                existing.get("decision", {}).get("selected_arm") == selected_id,
            )
            if not all(checks):
                raise ValueError(
                    "sealed final selection no longer matches its evidence"
                )
            value = existing
        else:
            _atomic_json(path, value)
        self.mark(
            "recipe_selection",
            "complete",
            selected_arm=selected_id,
            reason=reason,
            manifest=str(path),
            sha256=_sha256(path),
        )
        return value

    def prepare_final(
        self,
        *,
        release_config: dict[str, Any],
        selection: dict[str, Any],
        connectivity_state: Path,
    ) -> tuple[dict[str, Any], dict[str, Any], Path, list[str]]:
        selected_id = str(selection["decision"]["selected_arm"])
        selected_arm = _arm_by_id(release_config, selected_id)
        runtime, arm = build_final_training_spec(
            self.config, release_config, selected_arm
        )
        output = _resolve(self.config["final_training"]["output"])
        objective = {
            **runtime["base"]["objective"],
            **arm.get("objective_overrides", {}),
        }
        state = (
            connectivity_state
            if float(objective.get("dynamic_medial_connectivity_weight", 0.0)) > 0.0
            else None
        )
        command = build_release_command(runtime, arm, output, connectivity_state=state)
        exact = int(self.config["corpus"]["exact_rows"])
        interval = int(self.config["final_training"]["samples_per_validation_interval"])
        assertions = {
            "max_train_samples": int(_option(command, "--max-train-samples")),
            "samples_per_epoch": int(_option(command, "--samples-per-epoch")),
            "epochs": int(_option(command, "--epochs")),
            "batch_size": int(_option(command, "--batch-size")),
            "accumulate": int(_option(command, "--accumulate")),
            "patches": str(Path(_option(command, "--patches")).resolve()),
            "m7_checkpoint": str(Path(_option(command, "--m7-checkpoint")).resolve()),
        }
        if assertions != {
            "max_train_samples": exact,
            "samples_per_epoch": interval,
            "epochs": math.ceil(exact / interval),
            "batch_size": 3,
            "accumulate": 8,
            "patches": str(_resolve(self.config["corpus"]["manifest"])),
            "m7_checkpoint": str(
                _resolve(release_config["initialization"]["checkpoint"])
            ),
        }:
            raise ValueError(
                f"final command violates the sealed contract: {assertions}"
            )
        if "--stratified-sampling" not in command or "--resume" in command:
            raise ValueError(
                "final command is not a fresh deterministic stratified pass"
            )

        output.mkdir(parents=True, exist_ok=True)
        effective_path = output / "focused_effective_config.json"
        effective = {
            "schema": "crossres-focused-medial-final-250k-effective-config-v1",
            "focused_config_sha256": self.config_sha256,
            "selection_manifest_sha256": _sha256(
                _resolve(self.config["selection_manifest"])
            ),
            "selected_source_arm": selected_id,
            "runtime_release_config": runtime,
            "final_arm": arm,
            "command": command,
            "command_assertions": assertions,
            "connectivity_state": str(connectivity_state),
            "written_at_utc": _now(),
        }
        if effective_path.is_file():
            old = _read_json(effective_path)
            for key in (
                "focused_config_sha256",
                "selection_manifest_sha256",
                "selected_source_arm",
                "runtime_release_config",
                "final_arm",
                "command",
                "command_assertions",
                "connectivity_state",
            ):
                if old.get(key) != effective.get(key):
                    raise ValueError(f"effective final config field {key} changed")
        else:
            _atomic_json(effective_path, effective)
        self.mark(
            "final_recipe",
            "complete",
            selected_source_arm=selected_id,
            effective_config=str(effective_path),
            sha256=_sha256(effective_path),
            command_assertions=assertions,
        )
        return runtime, arm, output, command

    def train_final(
        self,
        *,
        runtime: dict[str, Any],
        arm: dict[str, Any],
        output: Path,
        connectivity_state: Path,
        selected_source_arm: str,
    ) -> dict[str, Any]:
        total = int(arm["samples"])
        snapshots = [int(value) for value in arm["snapshot_samples"]]
        state = training_commit_state(
            output, total_samples=total, expected_snapshots=snapshots
        )
        if not state["complete"]:
            self.mark(
                "final_training",
                "running",
                samples=total,
                selected_source_arm=selected_source_arm,
            )
            acquire_gpu_lock("focused:R_final_250k")
            try:
                state = self.run_training(
                    label="R_final_250k",
                    runtime=runtime,
                    arm=arm,
                    output=output,
                    connectivity_state=connectivity_state,
                    log_name="focused_train_R_final_250k.log",
                    selected_source_arm=selected_source_arm,
                )
            finally:
                release_gpu_lock()
        self.mark(
            "final_training",
            "complete",
            samples=total,
            selected_source_arm=selected_source_arm,
            commit=state,
        )
        return state

    def choose_final_milestone(self, output: Path) -> dict[str, Any]:
        initial = _read_json(output / "initial_validation.json")
        rows = _read_jsonl(output / "history.jsonl")
        best = select_internal_best(initial, rows)
        milestone = int(best["milestone_samples"])
        checkpoint = output / f"checkpoint_milestone_{milestone:08d}.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise FileNotFoundError(
                f"selected final milestone is missing: {checkpoint}"
            )
        record = {
            "milestone_samples": milestone,
            "epoch": int(best["epoch"]),
            "score": float(best["score"]),
            "initial_score": float(best["initial_score"]),
            "selection_rule": best["selection_rule"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "standard_two_sided_dice": standard_dice_from_validation(
                best["validation"]
            ),
            "trajectory": best["trajectory"],
        }
        selection_path = output / "internal_best_selection.json"
        if selection_path.is_file():
            old = _read_json(selection_path)
            for key in (
                "milestone_samples",
                "score",
                "initial_score",
                "checkpoint_sha256",
            ):
                if old.get(key) != record.get(key):
                    raise ValueError(f"internal best selection field {key} changed")
            record = old
        else:
            _atomic_json(selection_path, record)
        self.mark(
            "internal_best",
            "complete",
            milestone_samples=milestone,
            calibrated_macro_scroll_dice=record["score"],
            selection=str(selection_path),
            sha256=_sha256(selection_path),
        )
        return record

    @staticmethod
    def _outer_summary_valid(path: Path, milestone: int) -> bool:
        if not path.is_file():
            return False
        try:
            summary = _read_json(path)
            standard = summary.get("standard_two_sided_dice") or {}
            return (
                int(summary.get("best_milestone", -1)) == milestone
                and isinstance(standard.get("calibrated"), dict)
                and isinstance(standard.get("fixed_t025"), dict)
            )
        except (OSError, TypeError, ValueError):
            return False

    def evaluate_final(self, output: Path, best: dict[str, Any]) -> dict[str, Any]:
        milestone = int(best["milestone_samples"])
        summary_path = output / "ladder_eval/arm_summary.json"
        if not self._outer_summary_valid(summary_path, milestone):
            self.mark("final_frozen_evaluation", "running", milestone=milestone)
            command = [
                sys.executable,
                "scripts/evaluate_ladder_arm.py",
                "--arm-id",
                str(self.config["final_training"]["arm_id"]),
                "--run-dir",
                str(output),
                "--fixed-milestone",
                str(milestone),
                "--soft-arm",
                "--diagnostic",
                "--device",
                "cuda",
            ]
            acquire_gpu_lock("focused:R_final_250k:evaluation")
            try:
                self._run_logged(command, "focused_eval_R_final_250k.log")
            finally:
                release_gpu_lock()
        if not self._outer_summary_valid(summary_path, milestone):
            raise RuntimeError("final frozen evaluation did not publish standard Dice")
        summary = _read_json(summary_path)
        self.mark(
            "final_frozen_evaluation",
            "complete",
            milestone=milestone,
            summary=str(summary_path),
            sha256=_sha256(summary_path),
            standard_two_sided_dice=summary["standard_two_sided_dice"],
        )
        return summary

    def write_report(
        self,
        *,
        release_config: dict[str, Any],
        selection: dict[str, Any],
        unified_summaries: dict[str, dict[str, Any]],
        output: Path,
        best: dict[str, Any],
        outer: dict[str, Any],
    ) -> Path:
        report_path = _resolve(self.config["report"])
        standard = outer["standard_two_sided_dice"]
        cal = standard["calibrated"]
        fixed = standard["fixed_t025"]
        source_counts = _read_json(_resolve(self.config["corpus"]["summary"]))[
            "sources"
        ]
        lines = [
            "# Full-corpus release training report",
            "",
            f"Generated: {_now()}",
            "",
            "## Outcome",
            "",
            f"- Selected recipe: **{selection['decision']['selected_arm']}** ({selection['decision']['reason']}).",
            f"- Full training coverage: **{int(self.config['corpus']['exact_rows']):,} / {int(self.config['corpus']['exact_rows']):,} distinct accepted rows** in one deterministic stratified pass; no padding or oversampling.",
            f"- Internally selected trained milestone: **{int(best['milestone_samples']):,} samples** by guarded calibrated macro-scroll Dice.",
            f"- Frozen outer standard two-sided Dice at calibrated T={float(cal['threshold']):.2f}: **pooled {float(cal['pooled_dice']):.6f}; macro-by-scroll {float(cal['macro_scroll_dice']):.6f}**.",
            f"- Frozen outer standard two-sided Dice at fixed T=0.25: **pooled {float(fixed['pooled_dice']):.6f}; macro-by-scroll {float(fixed['macro_scroll_dice']):.6f}**.",
            "",
            f"Ordinary Dice formula: `{STANDARD_DICE_FORMULA}`. Competition composite scores are reported separately below and did not select the recipe.",
            "",
            "## Corpus accounting",
            "",
            "| scroll | accepted distinct rows |",
            "|---|---:|",
        ]
        for record in source_counts.values():
            lines.append(f"| {record['scroll_id']} | {int(record['patches']):,} |")
        lines += [
            f"| **total** | **{int(self.config['corpus']['exact_rows']):,}** |",
            "",
            f"Manifest SHA-256: `{self.config['corpus']['manifest_sha256']}`.",
            "",
            "## Fixed-T=0.25 medial-tail decision",
            "",
            "| criterion | value | verdict |",
            "|---|---:|---|",
        ]
        for name, criterion in selection["candidate"]["gate"]["criteria"].items():
            value = criterion.get("value")
            shown = (
                json.dumps(value, sort_keys=True)
                if isinstance(value, list)
                else f"{value}"
            )
            lines.append(
                f"| {name} | {shown} | {'PASS' if criterion['passed'] else 'fail'} |"
            )
        lines += [
            "",
            "The gate was evaluated only at the preregistered 15k endpoint and T=0.25. The competition composite was prohibited from choosing this recipe.",
            "",
            "## Unified B0r/C1/C3 frozen results",
            "",
            "| arm | milestone | calibrated T | pooled Dice | macro Dice | T=.25 pooled | T=.25 macro |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for arm_id in self.config["prerequisites"]["required_frozen_summaries"]:
            summary = unified_summaries[str(arm_id)]
            arm_standard = summary["standard_two_sided_dice"]
            arm_cal = arm_standard["calibrated"]
            arm_fixed = arm_standard["fixed_t025"]
            lines.append(
                f"| {arm_id} | {int(summary['best_milestone']):,} | "
                f"{float(arm_cal['threshold']):.2f} | {float(arm_cal['pooled_dice']):.6f} | "
                f"{float(arm_cal['macro_scroll_dice']):.6f} | "
                f"{float(arm_fixed['pooled_dice']):.6f} | "
                f"{float(arm_fixed['macro_scroll_dice']):.6f} |"
            )
        lines += [
            "",
            "## Full-pass inner validation trajectory",
            "",
            "| samples | eligible | calibrated T | pooled Dice | macro Dice | T=.25 pooled | T=.25 macro | selected then? |",
            "|---:|:---:|---:|---:|---:|---:|---:|:---:|",
        ]
        for row in best["trajectory"]:
            row_standard = row["standard_two_sided_dice"]
            row_cal = row_standard["calibrated"]
            row_fixed = row_standard["fixed_t025"]
            lines.append(
                f"| {int(row['milestone_samples']):,} | "
                f"{'yes' if row['checkpoint_eligible'] else 'no'} | "
                f"{float(row_cal['threshold']):.2f} | {float(row_cal['pooled_dice']):.6f} | "
                f"{float(row_cal['macro_scroll_dice']):.6f} | "
                f"{float(row_fixed['pooled_dice']):.6f} | "
                f"{float(row_fixed['macro_scroll_dice']):.6f} | "
                f"{'yes' if row['became_best'] else ''} |"
            )
        lines += [
            "",
            "## Frozen outer standard two-sided Dice",
            "",
            "| operating point | threshold | pooled Dice | macro Dice | precision | recall |",
            "|---|---:|---:|---:|---:|---:|",
            f"| calibrated | {float(cal['threshold']):.2f} | {float(cal['pooled_dice']):.6f} | {float(cal['macro_scroll_dice']):.6f} | {float(cal['precision']):.6f} | {float(cal['recall']):.6f} |",
            f"| fixed T=.25 | 0.25 | {float(fixed['pooled_dice']):.6f} | {float(fixed['macro_scroll_dice']):.6f} | {float(fixed['precision']):.6f} | {float(fixed['recall']):.6f} |",
            "",
            "## Competition composite (secondary)",
            "",
            "| set | operating point | leaderboard composite | surface Dice | VOI score | TopoScore |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for set_name, set_result in (outer.get("kaggle") or {}).items():
            for label in ("calibrated", "fixed"):
                result = set_result.get(label)
                if not isinstance(result, dict):
                    continue
                lines.append(
                    f"| {set_name} | {label} T={float(result['threshold']):.2f} | "
                    f"{float(result.get('mean_leaderboard', math.nan)):.6f} | "
                    f"{float(result.get('mean_surface_dice', math.nan)):.6f} | "
                    f"{float(result.get('mean_voi_score', math.nan)):.6f} | "
                    f"{float(result.get('mean_toposcore', math.nan)):.6f} |"
                )
        selected_checkpoint = Path(str(best["checkpoint"]))
        lines += [
            "",
            "## Provenance",
            "",
            f"- Fresh released M7 initializer: `{release_config['initialization']['sha256']}`.",
            f"- Selected trained checkpoint: `{selected_checkpoint}` (`{best['checkpoint_sha256']}`).",
            f"- Final endpoint: `{output / f'checkpoint_milestone_{int(self.config['corpus']['exact_rows']):08d}.pt'}`.",
            f"- Selection manifest: `{_resolve(self.config['selection_manifest'])}`.",
            f"- Frozen evaluation summary: `{output / 'ladder_eval/arm_summary.json'}`.",
            "",
        ]
        _atomic_text(report_path, "\n".join(lines))
        self.mark(
            "report",
            "complete",
            report=str(report_path),
            sha256=_sha256(report_path),
        )
        return report_path

    def verify_stage(self, stage: str) -> dict[str, Any]:
        """Recheck every sealed input immediately before a new long phase."""

        self.mark(stage, "running")
        facts = self.preflight()
        self.mark(stage, "complete", **facts)
        return facts

    def run(self, *, no_wait: bool = False) -> Path:
        facts = self.verify_stage("preflight")
        unified_summaries = self.wait_for_unified(no_wait=no_wait)
        facts = self.verify_stage("post_unified_reverification")
        release_config = _read_json(_resolve(self.config["release"]["config"]))
        connectivity_state = Path(facts["connectivity_state"])
        tail_dir = self.train_tail(release_config, connectivity_state)
        self.verify_stage("pre_tail_diagnostic_reverification")
        self.render_tail_panels(tail_dir)
        candidate_manifest = self.diagnose_tail(tail_dir)
        selection = self.seal_selection(
            release_config=release_config,
            candidate_manifest=candidate_manifest,
            unified_summaries=unified_summaries,
        )
        runtime, final_arm, output, _ = self.prepare_final(
            release_config=release_config,
            selection=selection,
            connectivity_state=connectivity_state,
        )
        selected_id = str(selection["decision"]["selected_arm"])
        self.verify_stage("pre_final_training_reverification")
        self.train_final(
            runtime=runtime,
            arm=final_arm,
            output=output,
            connectivity_state=connectivity_state,
            selected_source_arm=selected_id,
        )
        best = self.choose_final_milestone(output)
        self.verify_stage("pre_frozen_evaluation_reverification")
        outer = self.evaluate_final(output, best)
        report = self.write_report(
            release_config=release_config,
            selection=selection,
            unified_summaries=unified_summaries,
            output=output,
            best=best,
            outer=outer,
        )
        self.mark(
            "campaign",
            "complete",
            selected_source_arm=selected_id,
            final_samples=int(self.config["corpus"]["exact_rows"]),
            best_milestone=int(best["milestone_samples"]),
            report=str(report),
        )
        self.say(f"FOCUSED MEDIAL + FULL CORPUS CAMPAIGN COMPLETE -> {report}")
        return report


def dry_run(config_path: Path) -> None:
    coordinator = Coordinator(config_path)
    facts = coordinator.preflight()
    release_config = _read_json(_resolve(coordinator.config["release"]["config"]))
    connectivity_state = Path(facts["connectivity_state"])
    result: dict[str, Any] = {
        "config": str(config_path.resolve()),
        "config_sha256": coordinator.config_sha256,
        "corpus_rows": facts["corpus_rows"],
        "possible_final_commands": {},
    }
    for arm_id in (
        coordinator.config["release"]["baseline_arm"],
        coordinator.config["release"]["focused_arm"],
    ):
        source_arm = _arm_by_id(release_config, str(arm_id))
        runtime, arm = build_final_training_spec(
            coordinator.config, release_config, source_arm
        )
        result["possible_final_commands"][arm_id] = build_release_command(
            runtime,
            arm,
            _resolve(coordinator.config["final_training"]["output"]),
            connectivity_state=connectivity_state,
        )
    print(json.dumps(result, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="fail immediately when the unified C3 prerequisite is pending",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=300,
        help="prerequisite poll interval (production default: five minutes)",
    )
    arguments = parser.parse_args()
    config_path = arguments.config.expanduser().resolve()
    if arguments.dry_run:
        dry_run(config_path)
        return 0
    coordinator = Coordinator(config_path, poll_seconds=arguments.poll_seconds)
    try:
        coordinator.run(no_wait=arguments.no_wait)
    except Exception as error:
        detail = "".join(traceback.format_exception(error))
        coordinator.say("FATAL: " + str(error))
        try:
            coordinator.mark("failure", "failed", error=str(error), traceback=detail)
        except (KeyError, OSError, TypeError, ValueError):
            coordinator.say(detail)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
