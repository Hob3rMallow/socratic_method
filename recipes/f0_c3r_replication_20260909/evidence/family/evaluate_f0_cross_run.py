"""Cross-run stage of the F0 family, run by the coordinator after C3r.

Evaluates the pre-registered X family (F0 x C3r probability ensembles, one
exploratory weight soup) and the within-run C3r averages on the frozen
harness, applies the replication rule (the winning variants-ladder recipe
re-built on C3r must beat C3r's own reference with a positive paired lower
bound), merges everything with the variants decision table for a family-wide
Holm pass, and runs the geometry gates and Kaggle composite for the final
winner. Idempotent; receipts per variant; runs inside the coordinator's GPU
lock as a child process.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from evaluate_f0_variants import CheckpointSpec, Runner
from run_f0_family import Coordinator
from run_final_c3_250k import CROSSRES, ROOT, atomic_json, now, read_json

from crossres_pred.voxel.checkpoint_average import (
    average_checkpoints,
    members_match_receipt,
    write_average_checkpoint,
)

JOURNAL_SCHEMA = "crossres-f0-cross-run-journal-v1"
FINAL_SCHEMA = "crossres-f0-family-final-v1"


# ----------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ----------------------------------------------------------------------------


def describe_member(member: dict[str, Any], selected: dict[str, int]) -> str:
    run = str(member["run"])
    if "average" in member:
        samples = [int(s) for s in member["average"]]
        return f"{run}:avg[{min(samples)}-{max(samples)}x{len(samples)}]"
    samples = member["samples"]
    if samples == "selected":
        return f"{run}:{selected[run]}(selected)"
    return f"{run}:{int(samples)}"


def resolve_member_samples(member: dict[str, Any], selected: dict[str, int]) -> list[int]:
    if "average" in member:
        return [int(s) for s in member["average"]]
    samples = member["samples"]
    return [selected[str(member["run"])] if samples == "selected" else int(samples)]


def c3r_analogue(winner: dict[str, Any], c3r_selected: int) -> dict[str, Any] | None:
    """Rebuild the winning variants-ladder recipe on C3r (replication rule).

    Singles map to C3r's own selected milestone; weight averages keep the same
    sample set; diagnostic ensembles are never winners and return None.
    """
    kind = winner["checkpoint_kind"]
    tta = bool(winner["tta"])
    suffix = "_tta" if tta else ""
    if kind == "milestone":
        return {
            "id": f"c3r_sel{suffix}", "family": "X", "role": "replication", "kind": "milestone",
            "members": [{"run": "C3r", "samples": c3r_selected}], "tta": tta,
            "replicates": winner["variant_id"],
        }
    if kind == "average":
        samples = [int(s) for s in winner["members"]]
        label = re.sub(r"[^0-9a-z_]", "", f"c3r_avg_{min(samples) // 1000}k_{max(samples) // 1000}k_{len(samples)}")
        return {
            "id": f"{label}{suffix}", "family": "X", "role": "replication", "kind": "average",
            "members": [{"run": "C3r", "average": samples}], "tta": tta,
            "replicates": winner["variant_id"],
        }
    return None


def best_shippable(rows: list[dict[str, Any]], *, primary: str, exclude: set[str]) -> dict[str, Any] | None:
    candidates = [
        r for r in rows
        if not r.get("diagnostic_only") and r["variant_id"] not in exclude and r.get("verdict") != "vetoed-by-geometry"
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (float(r[f"macro_{primary}"]), r["variant_id"]))


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------


class CrossRunner(Runner):
    def __init__(self, family: Coordinator) -> None:
        super().__init__(CROSSRES / "configs/f0_variants_20260908.json", None)
        self.family = family
        self.digest = family.digest
        self.root = family.root / "cross_run"
        self.audits = self.root / "audits"
        self.averages = self.root / "averages"
        self.bootstrap_dir = self.root / "bootstrap"
        self.geometry_dir = self.root / "geometry"
        self.kaggle_dir = self.root / "kaggle"
        self.journal_path = ROOT / "output/crossres_data/logs/f0_cross_run_20260908_state.json"
        self.log_path = ROOT / "output/crossres_data/logs/f0_cross_run_20260908.log"
        self.variants_root = family.root / "variants"
        self.next_heartbeat = 0.0

    def journal(self) -> dict[str, Any]:
        if self.journal_path.exists():
            state = read_json(self.journal_path)
            if state.get("schema") != JOURNAL_SCHEMA or state.get("config_sha256") != self.digest:
                raise ValueError("cross-run journal/config identity changed")
            return state
        return {"schema": JOURNAL_SCHEMA, "config_sha256": self.digest, "stages": {}, "ledger": []}

    def say(self, message: str) -> None:
        super().say(f"[cross-run] {message}")

    # -- member resolution -------------------------------------------------------

    def run_dir(self, run: str) -> Path:
        if run == "F0":
            return ROOT / self.family.config["base"]["reference_run_dir"]
        return ROOT / self.family.arms[run]["output"]

    def selected_milestones(self) -> dict[str, int]:
        summary_path = self.family.root / "evaluation" / "C3r" / "summary.json"
        if not summary_path.exists():
            raise RuntimeError("C3r summary is missing; the cross-run stage needs the evaluated replicate")
        chosen = read_json(summary_path)["selection"]["proposed_late_candidate"]
        if chosen is None:
            raise RuntimeError("C3r has no eligible late milestone; cross-run selection unresolved")
        return {"F0": 200000, "C3r": int(chosen)}

    def within_run_average(self, run: str, samples: list[int]) -> Path:
        output = self.averages / f"{run.lower()}_avg_{min(samples) // 1000}k_{max(samples) // 1000}k_{len(samples)}.pt"
        members = [self.run_dir(run) / f"checkpoint_milestone_{s:08d}.pt" for s in samples]
        receipt = output.with_name(output.name + ".json")
        if output.exists():
            if receipt.exists() and members_match_receipt(read_json(receipt), members):
                return output
            raise ValueError(f"existing average does not match its members: {output}")
        self.say(f"averaging {run} milestones {samples[0]}..{samples[-1]} ({len(samples)})")
        write_average_checkpoint(average_checkpoints(members), output)
        return output

    def member_checkpoint(self, member: dict[str, Any], selected: dict[str, int]) -> Path:
        run = str(member["run"])
        samples = resolve_member_samples(member, selected)
        if "average" in member:
            return self.within_run_average(run, samples)
        return self.run_dir(run) / f"checkpoint_milestone_{samples[0]:08d}.pt"

    def materialize_cross(self, variant: dict[str, Any], selected: dict[str, int]) -> CheckpointSpec:
        members = [self.member_checkpoint(m, selected) for m in variant["members"]]
        if variant["kind"] == "ensemble":
            return CheckpointSpec("ensemble", members[0], tuple(members), ())
        if variant["kind"] == "milestone":
            return CheckpointSpec("milestone", members[0], (members[0],), ())
        if len(members) == 1:
            return CheckpointSpec("average", members[0], (members[0],), (), sources=tuple(members))
        output = self.averages / f"{variant['id']}.pt"
        receipt = output.with_name(output.name + ".json")
        if output.exists():
            if receipt.exists() and members_match_receipt(read_json(receipt), members):
                return CheckpointSpec("average", output, (output,), (), sources=tuple(members))
            raise ValueError(f"existing soup does not match its sources: {output}")
        self.say(f"{variant['id']}: cross-seed soup of {len(members)} sources")
        write_average_checkpoint(average_checkpoints(members, allowed_identity_keys=("options.seed",)), output)
        # A soup is ONE checkpoint: its sources are never ensemble members.
        return CheckpointSpec("average", output, (output,), (), sources=tuple(members))

    # -- flow -----------------------------------------------------------------------

    def evaluate_variant(self, variant: dict[str, Any], spec: CheckpointSpec, reference_dir: Path) -> dict[str, Any]:
        self.ledger(variant)
        report = self.audit(variant, spec)
        boot = self.bootstrap(variant, reference_dir)
        shaped = {**variant, "checkpoint": {"kind": variant["kind"], "samples": [describe_member(m, self.selected_milestones()) for m in variant["members"]]}}
        row = self.row(shaped, report, boot)
        self.mark(f"audit:{variant['id']}", "complete", macro_035=row["macro_0.35"])
        return row

    def run(self) -> None:
        selected = self.selected_milestones()
        self.mark("selection", "resolved", **{k: v for k, v in selected.items()})
        reference_dir = self.variants_root / "audits" / "f0_200k"
        c3r_reference = self.family.root / "evaluation" / "C3r" / f"milestone_{selected['C3r']:08d}" / "audit"
        if not (reference_dir / "report.json").exists() or not (c3r_reference / "report.json").exists():
            raise RuntimeError("reference audits for F0 and C3r are required before the cross-run stage")
        variants_table = read_json(self.variants_root / "decision_table.json")
        rows: dict[str, dict[str, Any]] = {r["variant_id"]: r for r in variants_table["rows"]}
        specs: dict[str, CheckpointSpec] = {}
        cross_variants = list(self.family.config["cross_run"]["variants"])
        # Replication rule: the winning variants-ladder recipe rebuilt on C3r.
        winner = best_shippable(variants_table["rows"], primary="0.35", exclude={self.config["reference"]["variant_id"]})
        analogue = c3r_analogue(winner, selected["C3r"]) if winner else None
        if analogue is not None and analogue["id"] not in {v["id"] for v in cross_variants}:
            cross_variants.append(analogue)
        c3r_reference_variant = {"id": "c3r_sel", "family": "X", "role": "reference", "kind": "milestone", "members": [{"run": "C3r", "samples": "selected"}], "tta": False}
        if "c3r_sel" not in {v["id"] for v in cross_variants}:
            cross_variants.insert(0, c3r_reference_variant)
        for variant in cross_variants:
            spec = self.materialize_cross(variant, selected)
            specs[variant["id"]] = spec
            rows[variant["id"]] = self.evaluate_variant(variant, spec, reference_dir)
        # Barrier check for soups and averages: below min(member) - .005 is a barrier failure.
        for variant in cross_variants:
            if variant["kind"] != "average":
                continue
            member_macros = []
            for member in variant["members"]:
                run = str(member["run"])
                for samples in resolve_member_samples(member, selected):
                    receipt = (self.variants_root / "audits" / f"f0_{samples // 1000}k" / "report.json") if run == "F0" else (
                        self.family.root / "evaluation" / run / f"milestone_{samples:08d}" / "audit" / "report.json")
                    if receipt.exists():
                        member_macros.append(float(next(p for p in read_json(receipt)["sweep"]["points"] if abs(float(p["threshold"]) - 0.35) < 1e-4)["macro_scroll_dice"]))
            if member_macros:
                rows[variant["id"]]["barrier_check"] = "barrier-failure" if rows[variant["id"]]["macro_0.35"] < min(member_macros) - 0.005 else "ok"
                rows[variant["id"]]["member_macro_035"] = member_macros
        # Replication verdict: the analogue against C3r's own reference.
        replication: dict[str, Any] = {"winner": winner["variant_id"] if winner else None, "analogue": analogue["id"] if analogue else None}
        if analogue is not None:
            paired = self.bootstrap(analogue, c3r_reference)
            operating = float(winner.get("operating_threshold") or 0.35)
            key = f"{operating:.2f}_vs_0.35"
            delta = paired["paired"][key]["delta"]
            replication.update(operating_threshold=operating, delta=delta["point"], ci=[delta["low"], delta["high"]], passed=delta["low"] > 0.0)
        self.mark("replication", "complete", **replication)
        # Final winner over the merged table.
        merged_rows = list(rows.values())
        overall = best_shippable(merged_rows, primary="0.35", exclude={self.config["reference"]["variant_id"], "c3r_sel"})
        geometry: dict[str, Any] = {}
        kaggle: dict[str, Any] = {}
        final: dict[str, Any] = {"schema": FINAL_SCHEMA, "winner": None}
        if overall is not None and overall["variant_id"] in specs:
            operating = max((0.30, 0.35), key=lambda t: (float(overall[f"macro_{t:.2f}"]), t))
            variant = next(v for v in cross_variants if v["id"] == overall["variant_id"])
            geometry[overall["variant_id"]] = self.geometry(variant, specs[overall["variant_id"]], operating)
            pred_dirs = self.kaggle_dump(variant, specs[overall["variant_id"]], operating)
            kaggle[overall["variant_id"]] = self.kaggle_score(variant, operating, pred_dirs)
            final.update(winner=overall["variant_id"], operating_threshold=operating)
        elif overall is not None:
            final.update(winner=overall["variant_id"], note="winner comes from the variants ladder; its gates and Kaggle live under variants/")
        table = self.decision_table(rows, {**{k: v for k, v in geometry.items()}}, kaggle, variants_table["reference_reproduction"])
        table["replication"] = replication
        table["c3r_selected"] = selected["C3r"]
        atomic_json(self.root / "decision_table.json", table)
        self.write_html(table)
        final.update(replication=replication, table=str(self.root / "decision_table.json"), created_utc=now())
        atomic_json(self.family.root / "final" / "summary.json", final)
        self.mark("final", "complete", winner=final["winner"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    family = Coordinator(args.config, args.config_sha256)
    CrossRunner(family).run()


if __name__ == "__main__":
    main()
