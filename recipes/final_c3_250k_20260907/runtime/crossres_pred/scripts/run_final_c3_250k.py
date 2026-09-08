"""Sealed, singleton F0-only C3/SGD/poly 250k coordinator. No trust ball."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CROSSRES = ROOT / "crossres_pred"
DATA = ROOT / "output/crossres_data"
RUN_ROOT = DATA / "final_c3_250k_20260905"
ARM_DIR = RUN_ROOT / "arms/F0"
CONFIG = CROSSRES / "configs/final_c3_250k_20260905.json"
ORIGINAL = DATA / "unified_ladder_20260902/arms/C3"
SNAPSHOTS = list(range(10000, 250001, 10000))
GPU = "GPU-304375e6-2d2f-b771-b8be-a0c095fe209e"
SCHEMA = "crossres-final-c3-250k-journal-v1"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
PINNED_SCRIPTS = (
    "run_final_c3_250k.py",
    "evaluate_final_c3_250k.py",
    "measure_blob_frontier.py",
    "launch_final_c3_250k.ps1",
    "run_unified_ladder.py",
    "evaluate_ladder_arm.py",
    "run_focused_medial_final_250k.py",
    "run_release_ladder.py",
    "generate_checkpoint_report.py",
    "render_cached_probability_report.py",
    "audit_m7_xr_v29_joint_thresholds.py",
    "measure_trust_displacement.py",
    "diagnose_release_t025_blobs.py",
    "render_release_milestone_panels.py",
)


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def code_pins() -> dict[str, str]:
    paths = [CROSSRES / "scripts" / name for name in PINNED_SCRIPTS]
    paths += sorted((CROSSRES / "src/crossres_pred/voxel").rglob("*.py"))
    paths += [CROSSRES / "configs/unified_ladder_20260902.json"]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def argument_options(command: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    flag = None
    for token in command:
        if token.startswith("--"):
            if token in result:
                raise ValueError(f"duplicate training flag: {token}")
            flag = token
            result[flag] = []
        elif flag:
            result[flag].append(token)
    return result


def training_command(config: dict[str, Any]) -> list[str]:
    command = list(config["arms"][0]["argv"])
    command[0] = str(ROOT / command[0])
    for flag in ("--patches", "--m7-checkpoint", "--output"):
        index = command.index(flag) + 1
        command[index] = str(ROOT / command[index])
    return command


def validate_scope(config: dict[str, Any]) -> None:
    if config.get("schema") != "crossres-final-c3-250k-prereg-v2":
        raise ValueError("unsupported F0 config schema")
    if [arm.get("arm_id") for arm in config.get("arms", [])] != ["F0"]:
        raise ValueError("only F0 is authorized")
    arm = config["arms"][0]
    if (
        arm["samples"] != 250000
        or arm["stop_at_samples"] != 250000
        or arm["snapshot_samples"] != SNAPSHOTS
        or arm["seed"] != 1203
        or (ROOT / arm["output"]).resolve() != ARM_DIR.resolve()
    ):
        raise ValueError("F0 output, seed, or full-horizon budget changed")
    execution = config["execution"]
    for key, expected in (
        ("gpu_uuid", GPU),
        ("power_limit_watts", 450),
        ("operating_threshold", 0.35),
        ("heartbeat_seconds", 1800),
        ("max_train_attempts", 3),
        ("minimum_late_samples", 150000),
        ("task_name", "VesuviusCrossres-FinalC3-250k"),
        ("metric_shards", 2),
    ):
        if execution.get(key) != expected:
            raise ValueError(f"authorized F0 execution setting changed: {key}")
    if (ROOT / execution["output_root"]).resolve() != RUN_ROOT.resolve():
        raise ValueError("F0 root changed")
    command = training_command(config)
    old = argument_options(read_json(ORIGINAL / "ladder_recipe.json")["command"])
    new = argument_options(command)
    for options in (old, new):
        options.pop("--output", None)
        options.pop("--snapshot-samples", None)
        for flag in ("--patches", "--m7-checkpoint"):
            options[flag] = [str(Path(options[flag][0]).resolve()).lower()]
    if new != old:
        changed = [k for k in new.keys() | old.keys() if new.get(k) != old.get(k)]
        raise ValueError(f"F0 no longer matches the original C3 argv: {changed}")
    if argument_options(command)["--snapshot-samples"] != [str(s) for s in SNAPSHOTS]:
        raise ValueError("training argv snapshot list disagrees with F0")
    if command[1:4] != ["-m", "crossres_pred.voxel.cli", "train"]:
        raise ValueError("unexpected trainer entry point")


def verify_run_identity(identity: dict[str, Any], config: dict[str, Any]) -> None:
    if identity.get("patch_manifest_sha256") != config["corpus"]["manifest_sha256"]:
        raise ValueError("live trainer is using the wrong corpus")
    if (
        identity.get("initialization", {}).get("checkpoint_sha256")
        != config["initialization"]["sha256"]
    ):
        raise ValueError("live trainer is not initialized from released M7")
    if identity.get("snapshot_samples") != SNAPSHOTS or identity.get(
        "resolved_schedule"
    ) != {
        "evaluation_interval_samples": 10000,
        "evaluation_intervals": 25,
        "total_samples": 250000,
    }:
        raise ValueError("live trainer full-horizon schedule changed")
    options = identity["options"]
    for key, expected in (
        ("learning_rate", 0.001),
        ("lr_schedule", "poly"),
        ("lr_floor_ratio", 0.0),
        ("warmup_samples", 0),
        ("momentum", 0.99),
        ("weight_decay", 3e-5),
        ("batch_size", 3),
        ("accumulate", 1),
        ("amp_dtype", "bfloat16"),
        ("seed", 1203),
        ("stratified_sampling", False),
        ("early_stopping_patience", None),
        ("max_train_samples", 250000),
    ):
        if options.get(key) != expected:
            raise ValueError(f"live trainer option changed: {key}")
    # The trainer deliberately omits SGD from its backward-compatible identity.
    # Adam/AdamW are explicit and add a separate optimizer contract.
    if options.get("optimizer", "sgd") != "sgd" or "optimizer_contract" in identity:
        raise ValueError("live trainer optimizer is not SGD")
    if not options.get("train_augmentation", True):
        raise ValueError("live trainer disabled C3 augmentation")
    if (
        options.get("m7_trust_region_relative_l2", 0) != 0
        or "m7_parameter_trust_region_contract" in identity
    ):
        raise ValueError("F0 must NEVER run with a trust ball")
    loss = options["loss_options"]
    expected_losses = {
        "cross_entropy_weight": 1.0,
        "dice_weight": 1.0,
        "m7_anchor_weight": 0.5,
        "m7_anchor_known_agreement": True,
        "m7_anchor_confident_agreement": True,
        "m7_anchor_unknown_corridor_radius": 2,
    }
    if any(loss.get(k) != v for k, v in expected_losses.items()):
        raise ValueError("live trainer loss or shipped flag triple changed")
    if any(
        v != 0
        for k, v in loss.items()
        if k.endswith("_weight") and k not in expected_losses
    ):
        raise ValueError("unrequested extra loss is active")


def progress(run_dir: Path = ARM_DIR) -> dict[str, Any]:
    rows = []
    history = run_dir / "history.jsonl"
    if history.exists():
        rows = [
            json.loads(line)
            for line in history.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    last = rows[-1] if rows else {}
    index_path = run_dir / "checkpoint_milestones.json"
    records = read_json(index_path).get("records", []) if index_path.exists() else []
    loss = last.get("train", {}).get("loss_total")
    return {
        "committed_samples": int(last.get("train", {}).get("cumulative_samples", 0)),
        "snapshot_samples": max((int(r["actual_samples"]) for r in records), default=0),
        "total_samples": 250000,
        "validated_intervals": len(rows),
        "learning_rate": last.get("learning_rate"),
        "train_loss": loss,
        "nonfinite_loss": loss is not None and not math.isfinite(float(loss)),
    }


def powershell_json(command: str) -> Any:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-WindowStyle",
            "Hidden",
            "-Command",
            "$ErrorActionPreference='Stop'; " + command,
        ],
        capture_output=True,
        text=True,
        check=True,
        creationflags=NO_WINDOW,
    )
    return json.loads(result.stdout) if result.stdout.strip() else []


def competing_workers() -> list[dict[str, Any]]:
    rows = powershell_json(
        "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$|^metric.*\\.exe$' } | Select-Object ProcessId,Name,CommandLine) | ConvertTo-Json -Compress"
    )
    if isinstance(rows, dict):
        rows = [rows]
    return [
        row
        for row in rows
        if row["ProcessId"] != os.getpid()
        and "run_final_c3_250k.py" not in (row.get("CommandLine") or "")
        and (
            "crossres" in (row.get("CommandLine") or "").lower()
            or row["Name"].lower().startswith("metric")
        )
    ]


def resource_facts() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            GPU,
            "--query-gpu=uuid,name,power.limit,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        creationflags=NO_WINDOW,
    )
    uuid, name, power, memory, utilization = next(
        csv.reader(io.StringIO(result.stdout))
    )
    if uuid.strip() != GPU or float(power) != 450:
        raise ValueError("GPU UUID or retained 450 W power limit changed")
    return {
        "uuid": uuid.strip(),
        "name": name.strip(),
        "power_limit_watts": float(power),
        "memory_used_mib": float(memory),
        "utilization_percent": float(utilization),
        "disk_free_bytes": shutil.disk_usage(DATA.resolve()).free,
    }


@contextmanager
def singleton():
    import msvcrt

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with (RUN_ROOT / "coordinator.lock").open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError(
                "another F0 coordinator already owns this run"
            ) from error
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


class Coordinator:
    def __init__(self, config_path: Path, expected_sha: str | None = None) -> None:
        self.path = config_path
        self.config = read_json(config_path)
        self.digest = sha256(config_path)
        if expected_sha and self.digest != expected_sha:
            raise ValueError("scheduled task's sealed config SHA256 changed")
        validate_scope(self.config)
        self.execution = self.config["execution"]
        self.journal_path = ROOT / self.execution["journal"]
        self.log_path = ROOT / self.execution["log"]
        self.next_heartbeat = time.monotonic()

    def journal(self) -> dict[str, Any]:
        state = (
            read_json(self.journal_path)
            if self.journal_path.exists()
            else {
                "schema": SCHEMA,
                "config_sha256": self.digest,
                "training_attempts": 0,
                "stages": {},
            }
        )
        if state.get("schema") != SCHEMA or state.get("config_sha256") != self.digest:
            raise ValueError("journal/config identity changed")
        return state

    def say(self, message: str) -> None:
        line = f"{now()} [final-c3] {message}"
        print(line, flush=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def mark(self, stage: str, status: str, **extra: Any) -> None:
        state = self.journal()
        state["stages"][stage] = {"status": status, "at_utc": now(), **extra}
        atomic_json(self.journal_path, state)
        self.say(f"{stage}: {status}")

    def heartbeat(self, stage: str, **extra: Any) -> dict[str, Any]:
        facts = progress()
        # Nonfinite values are represented as null in the status receipt, never hidden.
        if facts["nonfinite_loss"]:
            facts["train_loss"] = None
        facts.update(at_utc=now(), stage=stage, config_sha256=self.digest, **extra)
        atomic_json(RUN_ROOT / "status.json", facts)
        if time.monotonic() >= self.next_heartbeat:
            self.say(
                f"HEARTBEAT {stage}: validated {facts['committed_samples']:,}/250,000; snapshot {facts['snapshot_samples']:,}; LR={facts['learning_rate']}; loss={facts['train_loss']}"
            )
            self.next_heartbeat = time.monotonic() + 1800
        return facts

    def preflight(self) -> dict[str, Any]:
        validate_scope(self.config)
        if sha256(self.path) != self.digest:
            raise ValueError("config changed during execution")
        pins = self.execution["code_pins"]
        if not pins or code_pins() != pins:
            raise ValueError("sealed source code changed or pins are missing")
        verified = {}
        for path, expected in (
            (
                ROOT / self.config["initialization"]["checkpoint"],
                self.config["initialization"]["sha256"],
            ),
            (
                ROOT / self.config["corpus"]["manifest"],
                self.config["corpus"]["manifest_sha256"],
            ),
            (
                ROOT / self.config["evaluation"]["val_manifest"],
                self.config["evaluation"]["val_manifest_sha256"],
            ),
        ):
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"input SHA256 mismatch: {path}")
            verified[str(path)] = actual
        splits, scrolls, ids = Counter(), set(), set()
        with (ROOT / self.config["corpus"]["manifest"]).open(
            encoding="utf-8"
        ) as stream:
            for line in stream:
                row = json.loads(line)
                splits[row["split"]] += 1
                scrolls.add(row["scroll_id"])
                if row["patch_id"] in ids or not Path(row["path"]).is_file():
                    raise ValueError("duplicate patch ID or missing corpus archive")
                ids.add(row["patch_id"])
                if row["scroll_id"] == "PHerc0500P2" or (
                    row["split"] == "train"
                    and row["scroll_id"] in {"PHerc0814", "PHerc1451"}
                ):
                    raise ValueError("held-out scroll leaked into the selected corpus")
        # Wide15 has twelve training scrolls PLUS two validation-only scrolls.
        if splits != {"train": 15432, "val": 768} or len(scrolls) != 14:
            raise ValueError("wide15 corpus census changed")
        if (ARM_DIR / "run.json").exists():
            verify_run_identity(read_json(ARM_DIR / "run.json"), self.config)
        gpu = resource_facts()
        if gpu["disk_free_bytes"] < self.execution["minimum_free_bytes"]:
            raise RuntimeError(
                "insufficient free space for retained F0 milestones/evaluation"
            )
        return {
            "verified_inputs": verified,
            "corpus_splits": dict(splits),
            "scrolls": sorted(scrolls),
            "resources": gpu,
            "config_sha256": self.digest,
            "command": training_command(self.config),
        }

    def wait_for_workers(self) -> None:
        while workers := competing_workers():
            self.heartbeat("waiting_for_existing_workers", workers=workers)
            time.sleep(30)

    def child(self, command: list[str], stage: str, log_name: str) -> int:
        log_path = DATA / "logs" / log_name
        self.say("Launching " + stage + " -> " + str(log_path))
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"\n=== {now()} ===\n{subprocess.list2cmdline(command)}\n")
            stream.flush()
            process = subprocess.Popen(
                command,
                cwd=str(CROSSRES),
                stdout=stream,
                stderr=subprocess.STDOUT,
                creationflags=NO_WINDOW,
            )
            try:
                self.mark(
                    stage,
                    "running",
                    pid=process.pid,
                    command=command,
                    log=str(log_path),
                )
                identity_checked = False
                while process.poll() is None:
                    status = self.heartbeat(stage, pid=process.pid)
                    if (
                        stage == "training"
                        and (ARM_DIR / "run.json").exists()
                        and not identity_checked
                    ):
                        verify_run_identity(
                            read_json(ARM_DIR / "run.json"), self.config
                        )
                        self.mark(
                            "live_identity",
                            "verified",
                            run_sha256=sha256(ARM_DIR / "run.json"),
                        )
                        identity_checked = True
                    if stage == "training" and status["nonfinite_loss"]:
                        raise RuntimeError("nonfinite committed training loss")
                    time.sleep(30)
                return int(process.returncode)
            except BaseException:
                # Never release GPU ownership while an owned child survives a
                # coordinator error. Only this exact Popen process tree is stopped.
                if process.poll() is None:
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        check=False,
                        capture_output=True,
                        creationflags=NO_WINDOW,
                    )
                    process.wait()
                raise

    def completed_training(self, *, inspect_checkpoint: bool = False) -> dict[str, Any]:
        from run_focused_medial_final_250k import training_commit_state

        state = training_commit_state(
            ARM_DIR, total_samples=250000, expected_snapshots=SNAPSHOTS
        )
        if state["complete"] and inspect_checkpoint:
            import torch

            payload = torch.load(
                ARM_DIR / "checkpoint_last.pt", map_location="cpu", weights_only=False
            )
            identity = read_json(ARM_DIR / "run.json")
            verify_run_identity(identity, self.config)
            if (
                payload.get("identity") != identity
                or payload.get("cumulative_samples") != 250000
                or payload.get("epoch") != 24
            ):
                raise ValueError(
                    "last optimizer checkpoint is not the sealed 250k commit"
                )
            if not payload.get("metrics", {}).get("val") or "optimizer" not in payload:
                raise ValueError("final checkpoint lacks validation or optimizer state")
            if progress()["nonfinite_loss"]:
                raise ValueError("final training loss is nonfinite")
            state["checkpoint_last_sha256"] = sha256(ARM_DIR / "checkpoint_last.pt")
            state["final_milestone_sha256"] = sha256(
                ARM_DIR / "checkpoint_milestone_00250000.pt"
            )
        return state

    def train(self) -> None:
        if self.completed_training()["complete"]:
            self.mark(
                "training",
                "complete",
                commit=self.completed_training(inspect_checkpoint=True),
            )
            return
        recipe_path = ARM_DIR / "ladder_recipe.json"
        recipe = {
            "schema": "crossres-final-c3-F0-recipe-v1",
            "config_sha256": self.digest,
            "command": training_command(self.config),
            "source_recipe_sha256": sha256(ORIGINAL / "ladder_recipe.json"),
            "fresh_m7": self.config["initialization"],
            "natural_completion": True,
        }
        if recipe_path.exists() and read_json(recipe_path) != recipe:
            raise ValueError("sealed F0 launch recipe changed")
        if not recipe_path.exists():
            if (ARM_DIR / "run.json").exists():
                raise ValueError("unowned pre-existing F0 run")
            atomic_json(recipe_path, recipe)
        while self.journal()["training_attempts"] < 3:
            self.preflight()
            state = self.journal()
            state["training_attempts"] += 1
            atomic_json(self.journal_path, state)
            command = training_command(self.config)
            if (ARM_DIR / "run.json").exists():
                command.append("--resume")
            code = self.child(command, "training", "final_c3_250k_train_F0.log")
            commit = self.completed_training(inspect_checkpoint=code == 0)
            if code == 0 and commit["complete"]:
                self.mark(
                    "training",
                    "complete",
                    commit=commit,
                    attempts=state["training_attempts"],
                )
                return
            self.mark(
                "training_attempt_failure",
                "failed",
                attempt=state["training_attempts"],
                exit_code=code,
                commit=commit,
            )
        raise RuntimeError(
            "three sealed training attempts exhausted; no fresh restart or extra experiment"
        )

    def run(self) -> None:
        from run_unified_ladder import acquire_gpu_lock, release_gpu_lock

        with singleton():
            facts = self.preflight()
            self.mark("preflight", "complete", **facts)
            tasks = powershell_json(
                "@(Get-ScheduledTask -TaskName 'VesuviusCrossres-*' | Select-Object TaskName,@{Name='State';Expression={[string]$_.State}},@{Name='Enabled';Expression={$_.Settings.Enabled}}) | ConvertTo-Json -Compress"
            )
            if isinstance(tasks, dict):
                tasks = [tasks]
            if any(
                t["Enabled"]
                for t in tasks
                if t["TaskName"] != self.execution["task_name"]
            ):
                raise RuntimeError("an obsolete crossres task is still enabled")
            self.wait_for_workers()
            acquire_gpu_lock("final-c3:F0-only:no-trust-ball", poll_seconds=30)
            try:
                self.wait_for_workers()
                self.train()
                self.preflight()
                command = [
                    sys.executable,
                    str(CROSSRES / "scripts/evaluate_final_c3_250k.py"),
                    "--config",
                    str(self.path),
                    "--config-sha256",
                    self.digest,
                ]
                code = self.child(command, "evaluation", "final_c3_250k_evaluation.log")
                if code != 0:
                    raise RuntimeError(
                        f"F0 evaluation exited {code}; training remains complete; no training rerun"
                    )
                summary = read_json(RUN_ROOT / "evaluation/summary.json")
                if (
                    summary.get("config_sha256") != self.digest
                    or summary.get("evaluated_samples") != SNAPSHOTS
                ):
                    raise ValueError(
                        "final evaluation receipt is incomplete or mismatched"
                    )
                self.mark(
                    "operator_review_pause",
                    "complete",
                    report=str(RUN_ROOT / "evaluation/index.html"),
                    automatic_deployment=False,
                    next_run=None,
                )
                self.heartbeat("operator_review_pause")
            finally:
                release_gpu_lock()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--config-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--print-code-pins", action="store_true")
    args = parser.parse_args()
    if args.print_code_pins:
        print(json.dumps(code_pins(), indent=2))
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
        kernel.SetProcessAffinityMask.restype = wintypes.BOOL
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), 0xFFFF):
            raise ctypes.WinError(ctypes.get_last_error())
    coordinator = Coordinator(args.config, args.config_sha256)
    if args.preflight_only:
        print(json.dumps(coordinator.preflight(), indent=2))
        return
    if not args.config_sha256:
        raise ValueError("actual launch requires the authorized config SHA256")
    try:
        coordinator.run()
        powershell_json(
            "Get-ScheduledTask -TaskName 'VesuviusCrossres-FinalC3-250k' -ErrorAction SilentlyContinue | Disable-ScheduledTask | Out-Null"
        )
    except Exception:
        coordinator.mark("failure", "failed", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
