"""Opt-in, checked probability-corridor continuity for frozen F0/200k.

Default is preflight only. --run starts no inference or native child process.
Requires an existing checked native-repair baseline and its original inference
cache. Anatomical correctness is not established by the geometric checks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

from . import continuity_core as core
from . import continuity_paths as engine
from . import repair

PROFILE = "f0-probability-continuity-v2"
PROPOSAL = engine.ProposalPolicy(probability_floor=0.20)
ACCEPTANCE = engine.AcceptancePolicy()


def json_file(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def probability_cube(path):
    value = tifffile.imread(path)
    if value.shape != (128, 128, 128) or value.dtype not in (np.float16, np.float32):
        raise ValueError(f"expected float16/float32 128-cube probabilities: {path}")
    if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError(f"invalid probabilities: {path}")
    return value


def separate_output(source, baseline, output):
    for input_path in (source, baseline):
        if (
            output == input_path
            or input_path in output.parents
            or output in input_path.parents
        ):
            raise ValueError("output must be new and separate from both input trees")
    if output.exists():
        raise FileExistsError("output already exists")
    if list(output.parent.glob(output.name + ".partial-*")):
        raise ValueError(
            "partial continuity output requires inspection, not timeout recovery"
        )
    if output.with_name(output.name + ".continuity.lock").exists():
        raise FileExistsError(
            "continuity lock already exists; do not reclaim an unknown owner"
        )


def checked_baseline(receipt):
    plan = receipt["plan"]
    if (
        receipt.get("status") != "complete"
        or receipt.get("profile") != repair.PROFILE
        or receipt.get("erased_voxels") != 0
        or plan.get("parameters") != repair.PARAMETERS
        or plan.get("checkpoint_sha256") != repair.MODEL_SHA256
        or plan.get("threshold") != 0.35
        or plan["binary"]["sha256"] != repair.BINARY_SHA256
        or receipt["native"]["params"] != repair.PARAMETERS
        or receipt["native"]["bridges"]["cut"] != 0
    ):
        raise ValueError(
            "baseline is not a completed checked frozen-F0 additive repair"
        )
    return plan


def plan_continuity(source, baseline, output, *, voxel_size_um, max_work_gib=8):
    source, baseline, output = (Path(p).resolve() for p in (source, baseline, output))
    separate_output(source, baseline, output)
    receipt_path = baseline / "repair_receipt.json"
    old = repair.read_object(receipt_path)
    old_plan = checked_baseline(old)
    if Path(old_plan["source"]).resolve() != source:
        raise ValueError("baseline receipt refers to a different inference source")
    # Reuse the native wrapper's dense-grid, model, threshold, unit, umbilicus,
    # binary-identity and file validation. Its binary is checked, never run.
    grid = repair.plan_repair(
        source,
        output,
        Path(old_plan["binary"]["path"]),
        voxel_size_um=voxel_size_um,
        threads=1,
        max_work_gib=max_work_gib,
    )
    for key in ("cube_ids", "origin_zyx", "shape_zyx", "umbilicus_yx", "voxel_size_um"):
        if old_plan[key] != grid[key]:
            raise ValueError(f"baseline grid mismatch: {key}")
    metadata = repair.read_object(source / "provenance.json")
    if (
        metadata.get("halo") != 32
        or metadata.get("mirror_tta") is not True
        or metadata.get("amp_dtype") != "bfloat16"
    ):
        raise ValueError(
            "probability policy requires the measured halo32 / mirror-TTA / BF16 inference profile"
        )
    estimated = (
        math.prod(grid["shape_zyx"]) * 16 + math.prod(grid["shape_zyx"][1:]) * 256
    ) / 1024**3
    if estimated > max_work_gib:
        raise ValueError("estimated continuity working set exceeds max-work-gib")
    if (
        sorted(p.stem for p in (source / "probability").glob("*.tif"))
        != grid["cube_ids"]
    ):
        raise ValueError("probability cube inventory mismatch")
    if (
        sorted(p.stem for p in (baseline / "cubes_PRED").glob("*.tif"))
        != grid["cube_ids"]
    ):
        raise ValueError("baseline cube inventory mismatch")
    expected_names = {f"cubes_PRED/{key}.tif" for key in grid["cube_ids"]}
    outputs = old["output_files"]
    if (
        len(outputs) != len(expected_names)
        or {r["path"] for r in outputs} != expected_names
    ):
        raise ValueError("baseline receipt output inventory mismatch")
    identities = [
        *grid["input_files"],
        grid["binary"],
        repair.file_identity(receipt_path),
    ]
    # Do not trust a receipt alone: check its old inputs and current mask hashes.
    for item in old_plan["input_files"]:
        repair.check_identity(item)
    current_ids = {
        str(Path(r["path"]).resolve()): r["sha256"] for r in grid["input_files"]
    }
    old_ids = {
        str(Path(r["path"]).resolve()): r["sha256"] for r in old_plan["input_files"]
    }
    if current_ids != old_ids:
        raise ValueError("baseline input identities do not match this inference cache")
    for row in outputs:
        item = {**row, "path": str(baseline / row["path"])}
        repair.check_identity(item)
        identities.append(item)
    native_added = 0
    probability_dtype = None
    for key in grid["cube_ids"]:
        raw = repair.binary_cube(source / "cubes_PRED" / f"{key}.tif")
        filled = repair.binary_cube(baseline / "cubes_PRED" / f"{key}.tif")
        native_added += repair.verify_additive(raw, filled)["added_voxels"]
        path = source / "probability" / f"{key}.tif"
        probability = probability_cube(path)
        if (
            probability_dtype is not None
            and probability.dtype.name != probability_dtype
        ):
            raise ValueError(
                "mixed probability dtypes require a separately qualified cache"
            )
        probability_dtype = probability.dtype.name
        # Stored float16 P and raw float32 thresholding differ around .35.
        # Reject gross disagreement without rethresholding or changing masks.
        if ((probability > 0.351) & (raw == 0)).any() or (
            (probability < 0.349) & (raw != 0)
        ).any():
            raise ValueError("probability cache disagrees with the original .35 mask")
        identities.append(repair.file_identity(path))
    if native_added != old["added_voxels"]:
        raise ValueError(
            "baseline receipt count disagrees with independently measured additions"
        )
    identities.extend(
        repair.file_identity(Path(module.__file__)) for module in (core, engine, repair)
    )
    identities.append(repair.file_identity(Path(__file__)))
    return {
        "schema": "socratic-continuity-plan-v1",
        "profile": PROFILE,
        "source": str(source),
        "baseline": str(baseline),
        "output": str(output),
        "cube_ids": grid["cube_ids"],
        "origin_zyx": grid["origin_zyx"],
        "shape_zyx": grid["shape_zyx"],
        "umbilicus_yx": grid["umbilicus_yx"],
        "voxel_size_um": voxel_size_um,
        "checkpoint_sha256": repair.MODEL_SHA256,
        "threshold": 0.35,
        "proposal": asdict(PROPOSAL),
        "acceptance": asdict(ACCEPTANCE),
        "input_files": identities,
        "native_added_voxels": native_added,
        "probability_dtype": probability_dtype,
        "estimated_work_gib": estimated,
        "memory_estimate_is_not_a_hard_process_limit": True,
        "processes": 1,
        "child_processes": 0,
        "automatic_deployment": False,
        "native_global_radial_gate_applies_to_new_stage": False,
        "anatomical_correctness_requires_review": True,
        "boundary_note": "Outer 8 Z planes are unchanged; XY skeleton routes require 8px margin. Use interior cubes.",
    }


def window_for(key, lower):
    point = tuple(int(t[1:]) for t in key.split("_"))
    return tuple(slice(p - lo, p - lo + 128) for p, lo in zip(point, lower))


def audit_plane(before, after, ledger):
    """Independent labels from final paint, not proposal claims."""
    if (before & ~after).any():
        raise ValueError("baseline foreground erased")
    added = after & ~before
    labels, _ = ndimage.label(before, np.ones((3, 3), bool))
    strokes, count = ndimage.label(added, np.ones((3, 3), bool))
    after_face, _ = ndimage.label(after, ndimage.generate_binary_structure(2, 1))
    anchor_apron = np.zeros(before.shape, bool)
    for row in ledger:
        if row["accepted"]:
            anchors = np.array(row["path_yx"])[[0, -1]]
            anchor_apron[tuple(anchors.T)] = True
    anchor_apron = ndimage.binary_dilation(
        anchor_apron, ndimage.generate_binary_structure(2, 1)
    )
    apron_voxels = 0
    for number, bounds in enumerate(ndimage.find_objects(strokes), 1):
        area = tuple(
            slice(max(0, s.start - 1), min(before.shape[a], s.stop + 1))
            for a, s in enumerate(bounds)
        )
        touch = labels[area][
            ndimage.binary_dilation(strokes[area] == number, np.ones((3, 3), bool))
        ]
        owners = len(np.unique(touch[touch > 0]))
        if (
            owners == 1
            and (strokes[area] == number).sum() == 1
            and anchor_apron[area][strokes[area] == number].all()
        ):
            # Painting a fixed raw anchor can create a one-pixel apron on its
            # back side. It is attached to raw foreground, not another bridge.
            apron_voxels += 1
        elif owners != 2:
            raise ValueError(
                "addition must touch exactly two components or be a single verified anchor-apron voxel"
            )
    for row in ledger:
        if row["accepted"]:
            points = np.array(row["path_yx"])[[0, -1]]
            initial = labels[tuple(points.T)]
            final = after_face[tuple(points.T)]
            if (
                0 in initial
                or initial[0] == initial[1]
                or final[0] == 0
                or final[0] != final[1]
            ):
                raise ValueError("independent endpoint continuity check failed")
    if int(added.sum()) != sum(
        r.get("added_voxels", 0) for r in ledger if r["accepted"]
    ):
        raise ValueError("ledger paint count mismatch")
    return {
        "added_voxels": int(added.sum()),
        "two_contact_strokes": count - apron_voxels,
        "anchor_apron_voxels": apron_voxels,
    }


def run_continuity(plan):
    if (
        plan["profile"] != PROFILE
        or plan["proposal"] != asdict(PROPOSAL)
        or plan["acceptance"] != asdict(ACCEPTANCE)
    ):
        raise ValueError("plan policy differs from the fixed implementation")
    source, baseline, output = (Path(plan[k]) for k in ("source", "baseline", "output"))
    separate_output(source, baseline, output)
    for item in plan["input_files"]:
        repair.check_identity(item)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + ".continuity.lock")
    partial = output.with_name(output.name + f".partial-{os.getpid()}")
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(
            {"pid": os.getpid(), "output": str(output), "partial": str(partial)}, stream
        )
    try:
        if output.exists() or list(output.parent.glob(output.name + ".partial-*")):
            raise FileExistsError("output or partial appeared after preflight")
        partial.mkdir()
        json_file(partial / "plan.json", plan)
        base = np.zeros(plan["shape_zyx"], bool)
        p = np.empty(plan["shape_zyx"], np.dtype(plan["probability_dtype"]))
        for key in plan["cube_ids"]:
            w = window_for(key, plan["origin_zyx"])
            base[w] = repair.binary_cube(baseline / "cubes_PRED" / f"{key}.tif") != 0
            p[w] = probability_cube(source / "probability" / f"{key}.tif")
        after = base.copy()
        census, totals = Counter(), Counter()
        maximum_length = 0.0
        lower = plan["origin_zyx"]
        umb = tuple(a - b for a, b in zip(plan["umbilicus_yx"], lower[1:]))

        @lru_cache(maxsize=20)
        def nearby(z):
            return ndimage.maximum_filter(p[z].astype(np.float32), size=3)

        with (partial / "corridors.jsonl").open("x", encoding="utf-8") as log:
            for z in range(8, base.shape[0] - 8):
                proposals, counts = engine.find_corridors(base[z], p[z], umb, PROPOSAL)
                census.update({"proposal_" + k: v for k, v in counts.items()})
                for row in proposals:
                    path = np.array(row["path_yx"])
                    row["neighbor_probability_coverage"] = [
                        float((nearby(z + d)[tuple(path.T)] >= 0.20).mean())
                        for d in (-2, -1, 1, 2)
                    ]
                after[z], ledger, counts = engine.select_and_paint(
                    base[z], p[z], proposals, 0.20, ACCEPTANCE
                )
                census.update(counts)
                totals.update(audit_plane(base[z], after[z], ledger))
                for row in ledger:
                    row.update(z_world=z + lower[0], z_local=z)
                    log.write(json.dumps(row) + "\n")
                    if row["accepted"]:
                        totals["accepted_longer_than_21px"] += int(
                            row["length_px"] > 21
                        )
                        maximum_length = max(maximum_length, row["length_px"])
                if (z - 7) % 32 == 0:
                    json_file(
                        partial / "progress.json",
                        {
                            "status": "running",
                            "pid": os.getpid(),
                            "completed_planes": z - 7,
                            "accepted": census["accepted"],
                        },
                    )
                    print(
                        f"Continuity: {z - 7}/{base.shape[0] - 16} planes, {census['accepted']} corridors",
                        flush=True,
                    )
        (partial / "cubes_PRED").mkdir()
        cubes, files = {}, []
        for key in plan["cube_ids"]:
            path = partial / "cubes_PRED" / f"{key}.tif"
            w = window_for(key, lower)
            tifffile.imwrite(
                path, after[w].astype(np.uint8) * 255, photometric="minisblack"
            )
            cubes[key] = repair.verify_additive(
                repair.binary_cube(baseline / "cubes_PRED" / f"{key}.tif"),
                repair.binary_cube(path),
            )
            files.append(
                {**repair.file_identity(path), "path": f"cubes_PRED/{key}.tif"}
            )
        if sum(r["added_voxels"] for r in cubes.values()) != totals["added_voxels"]:
            raise ValueError("written cube count differs from plane verification")
        for item in plan["input_files"]:
            repair.check_identity(item)
        receipt = {
            "schema": "socratic-continuity-receipt-v1",
            "status": "complete",
            "profile": PROFILE,
            "completed_utc": datetime.now(UTC).isoformat(),
            "plan": plan,
            "census": dict(census),
            "accepted_corridors": census["accepted"],
            **dict(totals),
            "maximum_corridor_length_px": maximum_length,
            "erased_voxels": 0,
            "independent_endpoint_and_two_contact_audit": "passed",
            "cubes": cubes,
            "output_files": files,
            "anatomical_correctness_requires_review": True,
            "automatic_deployment": False,
            "ledger": {
                **repair.file_identity(partial / "corridors.jsonl"),
                "path": "corridors.jsonl",
            },
        }
        json_file(partial / "continuity_receipt.json", receipt)
        json_file(
            partial / "progress.json",
            {"status": "complete", "completed_planes": base.shape[0] - 16},
        )
        if output.exists():
            raise FileExistsError("output appeared before publication")
        partial.rename(output)
        return output
    except BaseException as error:
        if partial.exists():
            json_file(
                partial / "failure.json",
                {"status": "failed_or_interrupted", "error": repr(error)},
            )
        raise
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--voxel-size-um", type=float, required=True)
    parser.add_argument("--max-work-gib", type=float, default=8)
    parser.add_argument(
        "--run",
        action="store_true",
        help="execute after preflight; default checks only",
    )
    args = parser.parse_args()
    plan = plan_continuity(
        args.source,
        args.baseline,
        args.output,
        voxel_size_um=args.voxel_size_um,
        max_work_gib=args.max_work_gib,
    )
    print(run_continuity(plan) if args.run else json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
