"""Pre-registered F0 inference-variant ladder on the frozen v14p2 harness.

Variants are single checkpoints, uniform weight averages of F0 milestones and
diagnostic probability ensembles, audited with and without eight-way mirror
TTA. The reference (F0@200k, no TTA, fixed T=.35) is re-audited first with the
per-patch instrumentation and must reproduce the sealed sweep. Levels and
paired deltas carry cluster-bootstrap intervals; shippable candidates go
through the six-cube geometry gates, the frontier veto and the Kaggle
composite at their operating point. Journalled, idempotent, GPU-lock
serialized; the lock is released before CPU Kaggle scoring.

Never writes into the F0 run directory or the sealed release directory.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import audit_m7_xr_v29_joint_thresholds as morphology
import evaluate_ladder_arm as frozen
import measure_blob_frontier as frontier
import numpy as np
from evaluate_final_c3_250k import blind_command, point_at
from run_final_c3_250k import (
    CROSSRES,
    DATA,
    ROOT,
    atomic_json,
    now,
    powershell_json,
    read_json,
    sha256,
)
from run_unified_ladder import acquire_gpu_lock, release_gpu_lock

from crossres_pred.voxel.audit_bootstrap import (
    bootstrap_level,
    bootstrap_paired_delta,
    load_per_patch,
)
from crossres_pred.voxel.checkpoint_average import (
    average_checkpoints,
    members_match_receipt,
    write_average_checkpoint,
)

CONFIG_SCHEMA = "crossres-f0-variants-prereg-v1"
JOURNAL_SCHEMA = "crossres-f0-variants-journal-v1"
TABLE_SCHEMA = "crossres-f0-variants-decision-table-v1"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
CHECKPOINT_KINDS = ("milestone", "average", "ensemble")
WORKER_PATTERN = re.compile(
    r"crossres_pred[\\/]scripts|-m crossres_pred\.voxel|generate_checkpoint_report|"
    r"evaluate_f0_variants|run_f0_family|metric",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------------
# Pure helpers (unit-tested on CPU)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointSpec:
    """What the audit loads.

    ``members`` are the checkpoints the audit combines (the primary plus, for
    ensembles only, the extra ``--ensemble-member`` files). ``sources`` are the
    milestones a weight average was built from; they are inputs to the average,
    never audit members.
    """

    kind: str
    primary: Path
    members: tuple[Path, ...]
    samples: tuple[int, ...]
    sources: tuple[Path, ...] = ()

    @property
    def extra_members(self) -> tuple[Path, ...]:
        return self.members[1:]


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported variants config schema")
    variants = config["variants"]
    ids = [variant["id"] for variant in variants]
    if len(set(ids)) != len(ids):
        raise ValueError("variant ids must be unique")
    if not re.fullmatch(r"[a-z0-9_]+", "".join(ids)):
        raise ValueError("variant ids must be lowercase [a-z0-9_]")
    reference = config["reference"]["variant_id"]
    if reference not in ids:
        raise ValueError("the reference variant must be declared")
    for variant in variants:
        spec = variant["checkpoint"]
        if spec.get("kind") not in CHECKPOINT_KINDS:
            raise ValueError(f"{variant['id']}: unknown checkpoint kind")
        if spec["kind"] == "milestone" and not isinstance(spec.get("samples"), int):
            raise ValueError(f"{variant['id']}: milestone needs integer samples")
        if spec["kind"] in ("average", "ensemble"):
            samples = spec.get("samples")
            if not isinstance(samples, list) or len(samples) < 2:
                raise ValueError(f"{variant['id']}: {spec['kind']} needs >= 2 members")
            if len(set(samples)) != len(samples):
                raise ValueError(f"{variant['id']}: duplicate members")
        if spec["kind"] == "ensemble" and not variant.get("diagnostic_only", False):
            raise ValueError(f"{variant['id']}: ensembles must be diagnostic_only")
        if not isinstance(variant.get("tta"), bool):
            raise TypeError(f"{variant['id']}: tta must be a boolean")
        if variant.get("family") not in ("S", "W", "E", "X"):
            raise ValueError(f"{variant['id']}: family must be S, W, E or X")
    reference_variant = next(v for v in variants if v["id"] == reference)
    if reference_variant["tta"] or reference_variant["checkpoint"]["kind"] != "milestone":
        raise ValueError("the reference must be a single no-TTA milestone")
    primaries = config["family_primaries"]
    if len(set(primaries)) != len(primaries):
        raise ValueError("family primaries must be unique")
    for primary in primaries:
        if primary not in ids and not primary.startswith("x_"):
            raise ValueError(f"family primary is not a declared variant: {primary}")
    operating = config["evaluation"]["operating_thresholds"]
    if config["evaluation"]["primary_threshold"] not in operating:
        raise ValueError("the primary threshold must be an operating threshold")
    conditional = config["conditional_variants"]
    if conditional["family"] != "W" or conditional["count"] < 0:
        raise ValueError("conditional variants are the top weight averages only")


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    validate_config(config)
    return config


def resolve_checkpoint(
    config: dict[str, Any], variant: dict[str, Any], averages_dir: Path
) -> CheckpointSpec:
    spec = variant["checkpoint"]
    run_dir = ROOT / spec.get("run_dir", config["reference"]["run_dir"])
    if spec["kind"] == "milestone":
        samples = (int(spec["samples"]),)
        path = run_dir / f"checkpoint_milestone_{samples[0]:08d}.pt"
        return CheckpointSpec("milestone", path, (path,), samples)
    samples = tuple(int(value) for value in spec["samples"])
    members = tuple(run_dir / f"checkpoint_milestone_{s:08d}.pt" for s in samples)
    if spec["kind"] == "average":
        # A weight average is ONE checkpoint; its milestones are sources, not
        # ensemble members. Conditional TTA re-audits reuse the base average.
        average_id = str(variant.get("conditional_from") or variant["id"])
        primary = averages_dir / f"{average_id}.pt"
        return CheckpointSpec("average", primary, (primary,), samples, sources=members)
    return CheckpointSpec("ensemble", members[0], members, samples)


def audit_matches(
    report: dict[str, Any],
    *,
    checkpoint_sha: str,
    manifest_sha: str,
    tta: bool,
    thresholds: list[float],
    member_shas: list[str],
) -> bool:
    options = report.get("options", {})
    recorded = [round(float(v), 4) for v in options.get("thresholds", [])]
    members = [m["sha256"] for m in report.get("ensemble", {}).get("members", [])]
    return (
        report.get("checkpoint", {}).get("sha256") == checkpoint_sha
        and report.get("patch_manifest", {}).get("sha256") == manifest_sha
        and bool(options.get("mirror_tta")) == bool(tta)
        and recorded == [round(float(v), 4) for v in thresholds]
        and isinstance(report.get("per_patch"), dict)
        and members == member_shas
    )


def macro_at(report: dict[str, Any], threshold: float) -> float:
    return float(point_at(report["sweep"]["points"], threshold)["macro_scroll_dice"])


def sweep_difference(candidate: dict[str, Any], sealed: dict[str, Any]) -> float:
    """Max |macro difference| over the shared threshold grid."""
    largest = 0.0
    for point in sealed["sweep"]["points"]:
        threshold = float(point["threshold"])
        other = point_at(candidate["sweep"]["points"], threshold)
        largest = max(
            largest,
            abs(float(other["macro_scroll_dice"]) - float(point["macro_scroll_dice"])),
        )
    return largest


def holm(pvalues: dict[str, float], alpha: float) -> dict[str, bool]:
    """Holm step-down over one-sided bootstrap tail probabilities."""
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    count = len(ordered)
    passed: dict[str, bool] = {}
    still_rejecting = True
    for rank, (name, value) in enumerate(ordered):
        threshold = alpha / (count - rank)
        if still_rejecting and value <= threshold:
            passed[name] = True
        else:
            still_rejecting = False
            passed[name] = False
    return passed


def conditional_tta_variants(
    config: dict[str, Any], rows: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Top-k weight averages by fixed primary-T macro get a TTA audit."""
    rule = config["conditional_variants"]
    primary = config["evaluation"]["primary_threshold"]
    averages = [
        variant
        for variant in config["variants"]
        if variant["family"] == rule["family"]
        and not variant["tta"]
        and variant["id"] in rows
    ]
    ranked = sorted(
        averages,
        key=lambda v: (-float(rows[v["id"]][f"macro_{primary:.2f}"]), v["id"]),
    )
    result = []
    for variant in ranked[: int(rule["count"])]:
        result.append(
            {
                **variant,
                "id": f"{variant['id']}_tta",
                "tta": True,
                "role": "exploratory",
                "conditional_from": variant["id"],
            }
        )
    return result


def select_candidates(
    config: dict[str, Any], rows: dict[str, dict[str, Any]], variants: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Top-k shippable variants by the fixed primary-T macro, plus the reference."""
    primary = config["evaluation"]["primary_threshold"]
    selection = config["candidate_selection"]
    reference = config["reference"]["variant_id"]
    shippable = [
        variant
        for variant in variants
        if not variant.get("diagnostic_only", False) and variant["id"] in rows
    ]
    ranked = sorted(
        shippable,
        key=lambda v: (-float(rows[v["id"]][f"macro_{primary:.2f}"]), v["id"]),
    )
    chosen = ranked[: int(selection["shippable_top_k"])]
    if reference not in [v["id"] for v in chosen]:
        chosen.append(next(v for v in variants if v["id"] == reference))
    result = []
    for variant in chosen:
        row = rows[variant["id"]]
        operating = max(
            (float(t) for t in selection["operating_thresholds"]),
            key=lambda t: (float(row[f"macro_{t:.2f}"]), t),
        )
        result.append({**variant, "operating_threshold": operating})
    return result


def kaggle_candidates(candidates: list[dict[str, Any]], rows: dict[str, dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    primary = config["evaluation"]["primary_threshold"]
    ranked = sorted(
        candidates,
        key=lambda v: (-float(rows[v["id"]][f"macro_{primary:.2f}"]), v["id"]),
    )
    return ranked[: int(config["candidate_selection"]["kaggle_top_k"])]


SELF_SCRIPT = "evaluate_f0_variants.py"


def is_competing_worker(
    row: dict[str, Any], *, exclude_pids: set[int], exclude_scripts: tuple[str, ...]
) -> bool:
    """A python/metric process that may use the GPU and is not this ladder.

    The uv venv ``pythonw.exe`` is a launcher shim whose child is the real
    interpreter, so both our own pid and our parent pid are excluded, as is
    any process running this script (a second launcher instance is refused by
    the scheduled task itself).
    """
    command = row.get("CommandLine") or ""
    if int(row.get("ProcessId", 0)) in exclude_pids or "pytest" in command:
        return False
    if any(script in command for script in exclude_scripts):
        return False
    name = str(row.get("Name", "")).lower()
    return name.startswith("metric") or bool(WORKER_PATTERN.search(command))


def competing_gpu_workers() -> list[dict[str, Any]]:
    rows = powershell_json(
        "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match "
        "'^python(w)?\\.exe$|^metric.*\\.exe$' } | Select-Object ProcessId,Name,CommandLine)"
        " | ConvertTo-Json -Compress"
    )
    if isinstance(rows, dict):
        rows = [rows]
    exclude = {os.getpid(), os.getppid()}
    return [
        row
        for row in rows
        if is_competing_worker(row, exclude_pids=exclude, exclude_scripts=(SELF_SCRIPT,))
    ]


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------


class Runner:
    def __init__(self, config_path: Path, expected_sha: str | None) -> None:
        self.path = config_path.resolve()
        self.digest = sha256(self.path)
        if expected_sha and self.digest != expected_sha:
            raise ValueError("config SHA256 differs from the launched one")
        self.config = load_config(self.path)
        execution = self.config["execution"]
        self.root = ROOT / execution["output_root"]
        self.journal_path = ROOT / execution["journal"]
        self.log_path = ROOT / execution["log"]
        self.audits = self.root / "audits"
        self.averages = self.root / "averages"
        self.bootstrap_dir = self.root / "bootstrap"
        self.geometry_dir = self.root / "geometry"
        self.kaggle_dir = self.root / "kaggle"
        self.next_heartbeat = time.monotonic()
        evaluation = self.config["evaluation"]
        self.manifest = ROOT / evaluation["val_manifest"]
        self.manifest_sha = evaluation["val_manifest_sha256"]
        self.thresholds = [float(t) for t in evaluation["thresholds"]]

    # -- bookkeeping -------------------------------------------------------

    def say(self, message: str) -> None:
        line = f"{now()} [f0-variants] {message}"
        print(line, flush=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def journal(self) -> dict[str, Any]:
        if self.journal_path.exists():
            state = read_json(self.journal_path)
            if state.get("schema") != JOURNAL_SCHEMA or state.get("config_sha256") != self.digest:
                raise ValueError("journal/config identity changed")
            return state
        return {"schema": JOURNAL_SCHEMA, "config_sha256": self.digest, "stages": {}, "ledger": []}

    def mark(self, stage: str, status: str, **extra: Any) -> None:
        state = self.journal()
        state["stages"][stage] = {"status": status, "at_utc": now(), **extra}
        atomic_json(self.journal_path, state)
        self.say(f"{stage}: {status}")

    def ledger(self, variant: dict[str, Any]) -> None:
        """Append-only declaration of every variant before it is scored."""
        state = self.journal()
        if variant["id"] not in [entry["id"] for entry in state["ledger"]]:
            state["ledger"].append({**variant, "declared_utc": now()})
            atomic_json(self.journal_path, state)

    def heartbeat(self, stage: str, **extra: Any) -> None:
        facts = {"at_utc": now(), "stage": stage, "config_sha256": self.digest, **extra}
        atomic_json(self.root / "status.json", facts)
        if time.monotonic() >= self.next_heartbeat:
            self.say(f"HEARTBEAT {stage}: {json.dumps(extra, sort_keys=True)[:300]}")
            self.next_heartbeat = time.monotonic() + float(
                self.config["execution"]["heartbeat_seconds"]
            )

    def child(self, command: list[str | Path], log_name: str) -> None:
        log_path = DATA / "logs" / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"\n=== {now()} ===\n{subprocess.list2cmdline([str(c) for c in command])}\n")
            stream.flush()
            completed = subprocess.run(
                [str(c) for c in command],
                cwd=str(CROSSRES),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                creationflags=NO_WINDOW,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"child failed ({completed.returncode}): {log_path}")

    # -- preflight -----------------------------------------------------------

    def gpu_facts(self) -> dict[str, Any]:
        execution = self.config["execution"]
        result = subprocess.run(
            [
                "nvidia-smi", "-i", execution["gpu_uuid"],
                "--query-gpu=uuid,name,power.limit,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, check=True, creationflags=NO_WINDOW,
        )
        uuid, name, power, memory, utilization = [v.strip() for v in result.stdout.strip().split(",")]
        if uuid != execution["gpu_uuid"]:
            raise ValueError("GPU UUID changed")
        if float(power) > float(execution["max_power_limit_watts"]):
            raise ValueError("GPU power limit exceeds the allowed cap")
        return {
            "uuid": uuid, "name": name, "power_limit_watts": float(power),
            "memory_used_mib": float(memory), "utilization_percent": float(utilization),
            "disk_free_bytes": shutil.disk_usage(DATA.resolve()).free,
        }

    def preflight(self) -> dict[str, Any]:
        if sha256(self.path) != self.digest:
            raise ValueError("config changed during execution")
        reference = self.config["reference"]
        run_dir = ROOT / reference["run_dir"]
        if sha256(self.manifest) != self.manifest_sha:
            raise ValueError("frozen validation manifest changed")
        sealed = ROOT / reference["sealed_audit"]
        if not sealed.is_file():
            raise ValueError("sealed reference audit is missing")
        needed: set[int] = set()
        for variant in self.config["variants"]:
            spec = variant["checkpoint"]
            samples = spec["samples"]
            needed.update([samples] if isinstance(samples, int) else samples)
        missing = [s for s in sorted(needed) if not (run_dir / f"checkpoint_milestone_{s:08d}.pt").is_file()]
        if missing:
            raise ValueError(f"missing F0 milestones: {missing}")
        reference_samples = next(
            v for v in self.config["variants"] if v["id"] == reference["variant_id"]
        )["checkpoint"]["samples"]
        actual = sha256(run_dir / f"checkpoint_milestone_{reference_samples:08d}.pt")
        if actual != reference["sealed_release_sha256"]:
            raise ValueError("reference milestone differs from the sealed release")
        geometry = self.config["geometry"]
        if not (ROOT / geometry["frontier_reference"]).is_file():
            raise ValueError("cached C3 frontier is missing")
        for name in ("blind_source", "blind_reference_grid"):
            if not (ROOT / geometry[name]).exists():
                raise ValueError(f"geometry input missing: {name}")
        kaggle = self.config["kaggle"]
        if not (ROOT / kaggle["metric_exe"]).is_file():
            raise ValueError("metric.exe is missing")
        for name, expected in kaggle["expected_rows"].items():
            count = len(list((ROOT / kaggle["gt_root"] / name).glob("*.tif")))
            if count != int(expected):
                raise ValueError(f"Kaggle GT {name} has {count} rows, expected {expected}")
        facts = self.gpu_facts()
        if facts["disk_free_bytes"] < int(self.config["execution"]["minimum_free_bytes"]):
            raise RuntimeError("insufficient free disk space")
        self.root.mkdir(parents=True, exist_ok=True)
        receipt = {
            "schema": "crossres-f0-variants-preflight-v1",
            "at_utc": now(),
            "config_sha256": self.digest,
            "reference_sha256": actual,
            "resources": facts,
            "variants": [v["id"] for v in self.config["variants"]],
        }
        atomic_json(self.root / "preflight.json", receipt)
        return receipt

    # -- stages ----------------------------------------------------------------

    def materialize(self, variant: dict[str, Any]) -> CheckpointSpec:
        spec = resolve_checkpoint(self.config, variant, self.averages)
        if spec.kind != "average":
            return spec
        receipt_path = spec.primary.with_name(spec.primary.name + ".json")
        if spec.primary.exists():
            if receipt_path.exists() and members_match_receipt(read_json(receipt_path), list(spec.sources)):
                return spec
            raise ValueError(f"existing average does not match its sources: {spec.primary}")
        self.say(f"{variant['id']}: averaging {len(spec.sources)} milestones")
        payload = average_checkpoints(list(spec.sources))
        write_average_checkpoint(payload, spec.primary)
        return spec

    def audit(self, variant: dict[str, Any], spec: CheckpointSpec) -> dict[str, Any]:
        audit_dir = self.audits / variant["id"]
        report_path = audit_dir / "report.json"
        checkpoint_sha = sha256(spec.primary)
        member_shas = [sha256(m) for m in spec.extra_members]
        if report_path.exists():
            report = read_json(report_path)
            if audit_matches(
                report, checkpoint_sha=checkpoint_sha, manifest_sha=self.manifest_sha,
                tta=variant["tta"], thresholds=self.thresholds, member_shas=member_shas,
            ):
                return report
            raise ValueError(f"existing audit does not match the declared variant: {audit_dir}")
        if audit_dir.exists():
            raise ValueError(f"incomplete audit directory present: {audit_dir}")
        evaluation = self.config["evaluation"]
        command: list[str | Path] = [
            sys.executable, "-m", "crossres_pred.voxel.cli", "audit-checkpoint",
            "--checkpoint", spec.primary, "--patches", self.manifest, "--output", audit_dir,
            "--split", "val", "--device", "cuda", "--amp-dtype", evaluation["amp_dtype"],
            "--num-workers", str(evaluation["num_workers"]),
            "--max-cpu-threads", str(evaluation["max_cpu_threads"]),
        ]
        if variant["tta"]:
            command.append("--tta")
        for member in spec.extra_members:
            command += ["--ensemble-member", member]
        self.say(f"{variant['id']}: audit ({'TTA' if variant['tta'] else 'no TTA'}, {spec.kind})")
        started = time.monotonic()
        self.child(command, f"f0_variants_audit_{variant['id']}.log")
        report = read_json(report_path)
        if not audit_matches(
            report, checkpoint_sha=checkpoint_sha, manifest_sha=self.manifest_sha,
            tta=variant["tta"], thresholds=self.thresholds, member_shas=member_shas,
        ):
            raise ValueError(f"fresh audit does not match the declared variant: {audit_dir}")
        self.say(f"{variant['id']}: audited in {(time.monotonic() - started) / 60:.1f} min")
        return report

    def bootstrap(self, variant: dict[str, Any], reference_dir: Path | None) -> dict[str, Any]:
        path = self.bootstrap_dir / f"{variant['id']}.json"
        settings = self.config["bootstrap"]
        audit_dir = self.audits / variant["id"]
        report = read_json(audit_dir / "report.json")
        if path.exists():
            cached = read_json(path)
            if cached.get("checkpoint_sha256") == report["checkpoint"]["sha256"] and cached.get("reference_dir") == (str(reference_dir) if reference_dir else None):
                return cached
        counts = load_per_patch(audit_dir)
        result: dict[str, Any] = {
            "variant_id": variant["id"],
            "checkpoint_sha256": report["checkpoint"]["sha256"],
            "reference_dir": str(reference_dir) if reference_dir else None,
            "levels": {},
            "paired": {},
        }
        for threshold in self.config["evaluation"]["operating_thresholds"]:
            result["levels"][f"{float(threshold):.2f}"] = bootstrap_level(
                counts, float(threshold), replicates=int(settings["replicates"]),
                seed=int(settings["seed"]), confidence=float(settings["confidence"]),
                cluster=bool(settings["cluster"]),
            )
        if reference_dir is not None:
            reference = load_per_patch(reference_dir)
            primary = float(self.config["evaluation"]["primary_threshold"])
            for threshold in self.config["evaluation"]["operating_thresholds"]:
                result["paired"][f"{float(threshold):.2f}_vs_{primary:.2f}"] = bootstrap_paired_delta(
                    counts, reference, threshold=float(threshold), reference_threshold=primary,
                    replicates=int(settings["replicates"]), seed=int(settings["seed"]),
                    confidence=float(settings["confidence"]), cluster=bool(settings["cluster"]),
                )
        atomic_json(path, result)
        return result

    def row(self, variant: dict[str, Any], report: dict[str, Any], boot: dict[str, Any]) -> dict[str, Any]:
        primary = float(self.config["evaluation"]["primary_threshold"])
        row: dict[str, Any] = {
            "variant_id": variant["id"],
            "family": variant["family"],
            "role": variant.get("role", "exploratory"),
            "diagnostic_only": bool(variant.get("diagnostic_only", False)),
            "tta": bool(variant["tta"]),
            "checkpoint_kind": variant["checkpoint"]["kind"],
            "members": variant["checkpoint"]["samples"],
            "checkpoint_sha256": report["checkpoint"]["sha256"],
            "ensemble_member_sha256": [m["sha256"] for m in report.get("ensemble", {}).get("members", [])],
            "calibrated_threshold": float(report["sweep"]["selected"]["threshold"]),
            "calibrated_macro": float(report["sweep"]["selected"]["macro_scroll_dice"]),
        }
        for threshold in self.config["evaluation"]["operating_thresholds"]:
            label = f"{float(threshold):.2f}"
            point = point_at(report["sweep"]["points"], float(threshold))
            row[f"macro_{label}"] = float(point["macro_scroll_dice"])
            row[f"scrolls_{label}"] = {s: float(m["dice"]) for s, m in point["scrolls"].items()}
            row[f"precision_{label}"] = float(point["precision"])
            row[f"recall_{label}"] = float(point["recall"])
            level = boot["levels"][label]["interval"]
            row[f"ci_{label}"] = [level["low"], level["high"]]
            paired = boot["paired"].get(f"{label}_vs_{primary:.2f}")
            if paired is not None:
                row[f"delta_{label}_vs_ref"] = paired["delta"]["point"]
                row[f"delta_ci_{label}_vs_ref"] = [paired["delta"]["low"], paired["delta"]["high"]]
                row[f"p_le_zero_{label}_vs_ref"] = paired["delta"]["probability_delta_le_zero"]
                row[f"scroll_deltas_{label}_vs_ref"] = {
                    s: d["point"] for s, d in paired["scroll_deltas"].items()
                }
        return row

    def geometry(self, variant: dict[str, Any], spec: CheckpointSpec, operating: float) -> dict[str, Any]:
        """Blind six-cube export (halo 32, TTA, bf16) + gates + frontier at operating T."""
        checkpoint_sha = sha256(spec.primary)
        directory = self.geometry_dir / checkpoint_sha[:16]
        receipt_path = directory / f"receipt_t{round(operating * 100):03d}.json"
        if receipt_path.exists():
            cached = read_json(receipt_path)
            if cached.get("checkpoint_sha256") == checkpoint_sha:
                return cached
        geometry = self.config["geometry"]
        evaluation = {
            "blind_source": geometry["blind_source"],
            "blind_reference_grid": geometry["blind_reference_grid"],
            "gates": geometry["gates"],
        }
        report_dir = directory / "blind_t035"
        inference = report_dir / "inference"
        if not (inference / "provenance.json").exists():
            if report_dir.exists():
                shutil.rmtree(report_dir)
            self.say(f"{variant['id']}: blind six-cube export")
            self.child(blind_command(spec.primary, report_dir, evaluation), f"f0_variants_blind_{variant['id']}.log")
        measured = frontier.measure(inference, checkpoint_sha=checkpoint_sha)
        atomic_json(directory / "frontier.json", measured)
        thresholds = sorted({float(t) for t in geometry["gate_thresholds"]} | {float(operating)})
        paths = sorted((inference / "probability").glob("*.tif"))
        expected_ids = read_json(inference / "provenance.json")["target_cube_ids"]
        if len(paths) != 6 or {p.stem for p in paths} != set(expected_ids):
            raise ValueError("blind morphology audit must include exactly the six fixed cubes")
        published = ROOT / geometry["blind_source"] / "cubes_PRED"
        reference = ROOT / geometry["blind_reference_grid"]
        per_cube = {
            path.stem: morphology._cube_fast_sweep(
                path, published / path.name, reference / path.name, np.asarray(thresholds)
            )[0]
            for path in paths
        }
        aggregate = morphology._aggregate(per_cube, thresholds, {})
        rows = morphology._morphology_audit(paths, published, reference, thresholds)
        verdicts = {}
        for threshold in thresholds:
            cubes = next(r["cubes"] for r in rows if abs(float(r["threshold"]) - threshold) < 1e-9)
            verdicts[f"{threshold:.2f}"] = frozen.gate_verdict(
                point_at(aggregate, threshold), cubes, geometry["gates"], soft_arm=True
            )
        label = f"{float(operating):.2f}"
        fixed18 = measured["thresholds"][label]["fixed18"]
        c3 = read_json(ROOT / geometry["frontier_reference"])
        review = frontier.interpolated_bridge_limit(
            c3, float(fixed18["reference_skeleton_recall"]),
            maximum_growth=float(geometry["max_bridge_growth"]),
        )
        if review["resolved"]:
            review.update(
                candidate_bridges=int(fixed18["candidate_only_bridge_pixels"]),
                passed=int(fixed18["candidate_only_bridge_pixels"]) <= review["maximum_candidate_bridges"],
            )
        receipt = {
            "variant_id": variant["id"],
            "checkpoint_sha256": checkpoint_sha,
            "operating_threshold": float(operating),
            "gate_thresholds": thresholds,
            "gates": verdicts,
            "gate_at_operating": verdicts[label],
            "frontier_fixed18_at_operating": fixed18,
            "frontier_review": review,
            "geometry_pass": bool(verdicts[label]["passed"] and review.get("passed", False)),
            "blind_report": str(report_dir / "report/index.html"),
        }
        atomic_json(receipt_path, receipt)
        return receipt

    def kaggle_dump(self, variant: dict[str, Any], spec: CheckpointSpec, operating: float) -> dict[str, Path]:
        kaggle = self.config["kaggle"]
        label = f"{variant['id']}_t{round(operating * 100):03d}"
        pred_dirs: dict[str, Path] = {}
        for name, settings in kaggle["sets"].items():
            pred_dir = self.kaggle_dir / label / name
            pred_dirs[name] = pred_dir
            gt_dir = ROOT / kaggle["gt_root"] / name
            if pred_dir.exists() and len(list(pred_dir.glob("*.tif"))) == int(kaggle["expected_rows"][name]):
                continue
            self.say(f"{variant['id']}: Kaggle prediction dump {name} at T={operating:.2f}")
            count = frozen._dump_kaggle_predictions(
                spec.primary, ROOT / settings["manifest"], gt_dir=gt_dir,
                pred_dirs={f"{operating:.2f}": (pred_dir, float(operating))},
                scroll=settings["scroll"], device="cuda", mirror_tta=bool(variant["tta"]),
            )
            if count != int(kaggle["expected_rows"][name]):
                raise ValueError(f"Kaggle {name} row count {count} != {kaggle['expected_rows'][name]}")
        return pred_dirs

    def kaggle_score(self, variant: dict[str, Any], operating: float, pred_dirs: dict[str, Path]) -> dict[str, Any]:
        kaggle = self.config["kaggle"]
        label = f"{variant['id']}_t{round(operating * 100):03d}"
        summary_path = self.kaggle_dir / label / "kaggle.json"
        if summary_path.exists():
            return read_json(summary_path)
        result: dict[str, Any] = {"variant_id": variant["id"], "operating_threshold": float(operating), "student_tta": bool(variant["tta"]), "sets": {}}
        for name, pred_dir in pred_dirs.items():
            score_dir = pred_dir.with_name(pred_dir.name + "_score")
            cached = score_dir / "kaggle_summary.json"
            self.say(f"{variant['id']}: Kaggle scoring {name} ({kaggle['metric_shards']} shards)")
            summary = read_json(cached) if cached.exists() else frozen._score_kaggle_sharded(
                ROOT / kaggle["metric_exe"], ROOT / kaggle["gt_root"] / name, pred_dir, score_dir,
                shards=int(kaggle["metric_shards"]),
            )
            result["sets"][name] = summary
        atomic_json(summary_path, result)
        return result

    # -- decision table ------------------------------------------------------

    def decision_table(self, rows: dict[str, dict[str, Any]], geometry: dict[str, Any], kaggle: dict[str, Any], reproduction: dict[str, Any]) -> dict[str, Any]:
        primary = float(self.config["evaluation"]["primary_threshold"])
        label = f"{primary:.2f}"
        reference_id = self.config["reference"]["variant_id"]
        primaries = [p for p in self.config["family_primaries"] if p in rows]
        pvalues = {p: float(rows[p][f"p_le_zero_{label}_vs_ref"]) for p in primaries}
        holm_pass = holm(pvalues, float(self.config["holm_alpha"])) if pvalues else {}
        table = []
        for variant_id, row in rows.items():
            entry = dict(row)
            entry["holm_pass"] = holm_pass.get(variant_id)
            entry["is_primary"] = variant_id in primaries
            geo = geometry.get(variant_id)
            entry["operating_threshold"] = geo["operating_threshold"] if geo else None
            entry["gates_pass"] = geo["gate_at_operating"]["passed"] if geo else None
            entry["fg_ratio"] = geo["gate_at_operating"]["foreground_ratio_vs_reference"] if geo else None
            entry["frontier_pass"] = geo["frontier_review"].get("passed") if geo else None
            entry["frontier_bridges"] = geo["frontier_fixed18_at_operating"]["candidate_only_bridge_pixels"] if geo else None
            entry["frontier_coverage"] = geo["frontier_fixed18_at_operating"]["reference_skeleton_recall"] if geo else None
            entry["geometry_pass"] = geo["geometry_pass"] if geo else None
            scores = kaggle.get(variant_id)
            if scores:
                for name, summary in scores["sets"].items():
                    entry[f"composite_{name}"] = summary.get("mean_leaderboard")
                    entry[f"topo_{name}"] = summary.get("mean_toposcore")
                    entry[f"surface_dice_{name}"] = summary.get("mean_surface_dice")
            operating = entry["operating_threshold"]
            claim = None
            if variant_id != reference_id and operating is not None:
                op = f"{float(operating):.2f}"
                macro = float(row[f"macro_{op}"])
                delta_low = float(row[f"delta_ci_{op}_vs_ref"][0])
                per_scroll = row[f"scroll_deltas_{op}_vs_ref"]
                claim = bool(
                    macro >= 0.700000
                    and delta_low > 0.0
                    and (holm_pass.get(variant_id, False) if variant_id in primaries else False)
                    and all(float(v) >= 0.0 for v in per_scroll.values())
                )
                entry["exploratory_note"] = None if variant_id in primaries else "exploratory: not promotable on v14p2 alone (rule 7)"
            entry["breaks_0p70"] = claim
            if entry.get("geometry_pass") is False:
                entry["verdict"] = "vetoed-by-geometry"
            elif claim:
                entry["verdict"] = "breaks-0.70-candidate (needs C3r replication, 0500P2 floor, repair re-qualification)"
            elif variant_id == reference_id:
                entry["verdict"] = "reference"
            elif entry["diagnostic_only"]:
                entry["verdict"] = "diagnostic-only"
            else:
                entry["verdict"] = "no-claim"
            table.append(entry)
        table.sort(key=lambda e: -float(e[f"macro_{label}"]))
        return {
            "schema": TABLE_SCHEMA,
            "config_sha256": self.digest,
            "created_utc": now(),
            "primary_threshold": primary,
            "reference": reference_id,
            "reference_reproduction": reproduction,
            "family_primaries": primaries,
            "holm": {"alpha": self.config["holm_alpha"], "pvalues": pvalues, "pass": holm_pass},
            "rows": table,
            "prereg_document": self.config["prereg_document"],
        }

    def write_html(self, table: dict[str, Any]) -> None:
        primary = f"{table['primary_threshold']:.2f}"
        columns = [
            "variant_id", "family", "role", "tta", "members", f"macro_{primary}", f"ci_{primary}",
            f"delta_{primary}_vs_ref", f"delta_ci_{primary}_vs_ref", "holm_pass", "macro_0.30",
            "calibrated_threshold", "calibrated_macro", "operating_threshold", "gates_pass", "fg_ratio",
            "frontier_pass", "composite_v14p2_val", "topo_v14p2_val", "composite_p0500p2_val",
            "breaks_0p70", "verdict",
        ]
        cells = []
        for row in table["rows"]:
            cells.append(
                "<tr>" + "".join(
                    f"<td>{html.escape(json.dumps(row.get(c)) if not isinstance(row.get(c), str) else row.get(c))}</td>"
                    for c in columns
                ) + "</tr>"
            )
        text = (
            "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            "<title>F0 variants ladder</title><style>body{font:15px system-ui;margin:2rem}table{border-collapse:collapse}"
            "th,td{border:1px solid #bbb;padding:.35rem .5rem;text-align:right;font-variant-numeric:tabular-nums}"
            "td:first-child{text-align:left}</style><h1>F0 inference-variant ladder (frozen v14p2, 689 rows)</h1>"
            f"<p>Reference {html.escape(table['reference'])}; reproduction of the sealed sweep: "
            f"{html.escape(json.dumps(table['reference_reproduction']))}. Holm over primaries "
            f"{html.escape(json.dumps(table['holm']))}. Rules: {html.escape(table['prereg_document'])}.</p>"
            "<table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in columns) + "</tr></thead><tbody>"
            + "".join(cells) + "</tbody></table></html>"
        )
        temporary = self.root / "index.html.tmp"
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, self.root / "index.html")

    # -- main flow ---------------------------------------------------------------

    def run(self) -> None:
        facts = self.preflight()
        self.mark("preflight", "complete", **facts)
        variants = list(self.config["variants"])
        reference_id = self.config["reference"]["variant_id"]
        reference_variant = next(v for v in variants if v["id"] == reference_id)
        ordered = [reference_variant] + [
            v for v in sorted(variants, key=lambda v: (v["tta"], v["family"] == "E")) if v["id"] != reference_id
        ]
        for variant in ordered:
            self.ledger(variant)
        while workers := competing_gpu_workers():
            self.heartbeat("waiting_for_existing_gpu_workers", workers=[w["ProcessId"] for w in workers])
            time.sleep(30)
        acquire_gpu_lock(self.config["execution"]["gpu_lock_owner"], poll_seconds=30)
        rows: dict[str, dict[str, Any]] = {}
        specs: dict[str, CheckpointSpec] = {}
        geometry: dict[str, Any] = {}
        kaggle_pending: list[tuple[dict[str, Any], float, dict[str, Path]]] = []
        reproduction: dict[str, Any] = {}
        try:
            reference_dir = self.audits / reference_id
            for index, variant in enumerate(ordered):
                self.heartbeat("audits", done=index, total=len(ordered), current=variant["id"])
                spec = self.materialize(variant)
                specs[variant["id"]] = spec
                report = self.audit(variant, spec)
                if variant["id"] == reference_id:
                    sealed = read_json(ROOT / self.config["reference"]["sealed_audit"])
                    difference = sweep_difference(report, sealed)
                    reproduction = {
                        "max_abs_macro_difference": difference,
                        "tolerance": float(self.config["reference"]["reproduction_tolerance"]),
                        "sealed_macro_at_035": float(self.config["reference"]["sealed_macro_at_035"]),
                        "reaudited_macro_at_035": macro_at(report, 0.35),
                        "passed": difference <= float(self.config["reference"]["reproduction_tolerance"]),
                    }
                    self.mark("reference_reproduction", "complete" if reproduction["passed"] else "failed", **reproduction)
                    if not reproduction["passed"]:
                        raise RuntimeError("re-audited reference does not reproduce the sealed sweep; ladder stopped")
                boot = self.bootstrap(variant, None if variant["id"] == reference_id else reference_dir)
                rows[variant["id"]] = self.row(variant, report, boot)
                self.mark(f"audit:{variant['id']}", "complete", macro_035=rows[variant["id"]]["macro_0.35"])
            # Pre-registered conditional TTA audits of the best weight averages.
            conditional = conditional_tta_variants(self.config, rows)
            for variant in conditional:
                self.ledger(variant)
                spec = self.materialize(variant)
                specs[variant["id"]] = spec
                report = self.audit(variant, spec)
                boot = self.bootstrap(variant, reference_dir)
                rows[variant["id"]] = self.row(variant, report, boot)
                self.mark(f"audit:{variant['id']}", "complete", macro_035=rows[variant["id"]]["macro_0.35"])
            all_variants = variants + conditional
            candidates = select_candidates(self.config, rows, all_variants)
            self.mark("candidates", "selected", candidates=[(c["id"], c["operating_threshold"]) for c in candidates])
            for candidate in candidates:
                self.heartbeat("geometry", current=candidate["id"])
                geometry[candidate["id"]] = self.geometry(candidate, specs[candidate["id"]], candidate["operating_threshold"])
                self.mark(f"geometry:{candidate['id']}", "complete", geometry_pass=geometry[candidate["id"]]["geometry_pass"])
            for candidate in kaggle_candidates(candidates, rows, self.config):
                self.heartbeat("kaggle_dump", current=candidate["id"])
                pred_dirs = self.kaggle_dump(candidate, specs[candidate["id"]], candidate["operating_threshold"])
                kaggle_pending.append((candidate, candidate["operating_threshold"], pred_dirs))
        finally:
            release_gpu_lock()
            self.say("GPU lock released")
        kaggle: dict[str, Any] = {}
        for candidate, operating, pred_dirs in kaggle_pending:
            self.heartbeat("kaggle_score", current=candidate["id"])
            kaggle[candidate["id"]] = self.kaggle_score(candidate, operating, pred_dirs)
            self.mark(f"kaggle:{candidate['id']}", "complete")
        table = self.decision_table(rows, geometry, kaggle, reproduction)
        atomic_json(self.root / "decision_table.json", table)
        self.write_html(table)
        self.mark("operator_review_pause", "complete", table=str(self.root / "decision_table.json"), automatic_deployment=False)
        self.heartbeat("operator_review_pause")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CROSSRES / "configs/f0_variants_20260908.json")
    parser.add_argument("--config-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    runner = Runner(args.config, args.config_sha256)
    if args.preflight_only:
        print(json.dumps(runner.preflight(), indent=2, sort_keys=True))
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = runner.config["execution"]["gpu_uuid"]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        runner.run()
    except BaseException as error:
        runner.mark("failure", "failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
