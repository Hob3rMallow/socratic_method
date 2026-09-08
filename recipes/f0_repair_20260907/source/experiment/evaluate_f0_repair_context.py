"""Run declared additive repair presets sequentially on frozen F0 blocks.

Numbers here describe repairs, not their anatomical correctness. Candidate
selection requires inspecting the newly connected paths and control results.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tifffile

import run_f0_repair_context as context


def additive_counts(before: np.ndarray, after: np.ndarray) -> dict:
    if before.shape != after.shape or before.ndim != 3:
        raise ValueError("repair changed volume geometry")
    if not np.isin(before, [0, 255]).all() or not np.isin(after, [0, 255]).all():
        raise ValueError("expected binary uint8 masks encoded as 0/255")
    original, fixed = before != 0, after != 0
    erased = int(np.count_nonzero(original & ~fixed))
    if erased:
        raise ValueError(f"additive contract violated: {erased} erased voxels")
    return {"foreground_before": int(original.sum()), "added_voxels": int((fixed & ~original).sum()), "erased_voxels": erased}


def command_for(source: Path, target: Path, preset: dict) -> list[str]:
    if set(preset) != {"reach_max", "radial_dr", "min_support"}:
        raise ValueError("unexpected native preset fields")
    return [str(context.EXE), source.as_posix(), target.as_posix(),
            "--umb-y", "3287", "--umb-x", "3784", "--reach-safe", "6",
            "--reach-max", str(preset["reach_max"]), "--radial-dr", str(preset["radial_dr"]),
            "--min-support", str(preset["min_support"]), "--min-score", "0.30",
            "--paint-radius", "1", "--threads", "16", "--png-max", "12",
            "--crop-max", "24", "--rej-crop-max", "24"]


def stage_input(request: dict, block: str) -> Path:
    state = context.read(context.OUTPUT / "inference_state.json")
    if block not in state["blocks"]:
        raise ValueError("this inference block is not committed")
    receipt = state["blocks"][block]
    for item in [receipt["provenance"], *receipt["files"]]:
        context.verify_file(item)
    inference = context.OUTPUT / f"inference_{block}"
    source = context.OUTPUT / f"input_{block}"
    spec = request["blocks"][block]
    source_manifest = {
        "schema": "f0-contiguous-repair-input-v1", "chunk_size": 128,
        "bbox_l0_zyx": [v for start in spec["origin_zyx"] for v in (start, start + 384)],
        "n_chunks": [3, 3, 3], "umbilicus_yx": [3287.0, 3784.0],
        "source_inference": str(inference), "request_sha256": context.sha256(context.OUTPUT / "request.json"),
        "checkpoint_sha256": context.MODEL_SHA, "threshold": 0.35,
    }
    if source.exists():
        if context.read(source / "manifest.json") != source_manifest:
            raise ValueError("staged input identity mismatch")
    else:
        (source / "cubes_PRED").mkdir(parents=True)
        for key in spec["target_cube_ids"]:
            shutil.copy2(inference / "cubes_PRED" / f"{key}.tif", source / "cubes_PRED" / f"{key}.tif")
        context.atomic_json(source / "manifest.json", source_manifest)
    for key in spec["target_cube_ids"]:
        if context.sha256(source / "cubes_PRED" / f"{key}.tif") != context.sha256(inference / "cubes_PRED" / f"{key}.tif"):
            raise ValueError("staging changed frozen predictions")
    return source


def summarize(request: dict, block: str, preset_name: str, target: Path) -> dict:
    native = context.read(target / "fixup_report.json")
    preset = request["presets"][preset_name]
    expected = {**preset, "reach_safe": 6, "min_score": 0.3, "paint_radius": 1, "radial_armed": 1, "tracks": 1}
    if native["params"] != expected or native["bridges"]["cut"] != 0:
        raise ValueError("native run did not honor the additive preset")
    spec = request["blocks"][block]
    inference = context.OUTPUT / f"inference_{block}"
    baseline = context.OUTPUT / "repairs" / block / "report_baseline"
    cubes, added_probabilities, files = {}, [], []
    baseline_repairs_not_retained = new_over_baseline = 0
    for key in spec["target_cube_ids"]:
        before_path = inference / "cubes_PRED" / f"{key}.tif"
        after_path = target / "cubes_PRED" / f"{key}.tif"
        before, after = tifffile.imread(before_path), tifffile.imread(after_path)
        if before.shape != (128, 128, 128):
            raise ValueError("unexpected cube shape")
        cubes[key] = additive_counts(before, after)
        probability = tifffile.imread(inference / "probability" / f"{key}.tif")
        added_probabilities.append(probability[(after != 0) & (before == 0)].astype(np.float32))
        baseline_mask = tifffile.imread(baseline / "cubes_PRED" / f"{key}.tif") != 0
        new_over_baseline += int(((after != 0) & ~baseline_mask).sum())
        baseline_repairs_not_retained += int((baseline_mask & (after == 0)).sum())
        files.append(context.inventory(after_path))
    if sum(c["added_voxels"] for c in cubes.values()) != native["painted_px"]:
        raise ValueError("native painted counter does not match independent output diff")
    if sum(c["foreground_before"] for c in cubes.values()) != native["fg_in"]:
        raise ValueError("native input foreground counter mismatch")
    with (target / "planes.csv").open(newline="", encoding="utf-8") as stream:
        planes = list(csv.DictReader(stream))
    census = {key: sum(int(row[key]) for row in planes) for key in planes[0] if key != "z_world"}
    joins = [json.loads(line) for line in (target / "joins.jsonl").read_text(encoding="utf-8").splitlines()]
    kept = [row for row in joins if row["kept"]]
    if len(kept) != native["kept"]:
        raise ValueError("join ledger and native kept count differ")
    p = np.concatenate(added_probabilities)
    probability_summary = ({"mean": float(p.mean()), "p10": float(np.quantile(p, .1)),
                            "minimum": float(p.min()), "fraction_ge_020": float((p >= .2).mean()),
                            "fraction_ge_025": float((p >= .25).mean()), "fraction_ge_030": float((p >= .3).mean())} if p.size else None)
    for name in ("fixup_report.json", "joins.jsonl", "rejected.jsonl", "planes.csv"):
        files.append(context.inventory(target / name))
    return {
        "schema": "f0-repair-preset-result-v1", "block": block, "preset": preset_name,
        "native": native, "plane_census": census, "cubes": cubes,
        "core_cube": spec["primary_assessment_core_cube"], "core": cubes[spec["primary_assessment_core_cube"]],
        "kept_connection_tracks": len({row["conn_id"] for row in kept}),
        "new_over_baseline_voxels": new_over_baseline,
        "baseline_repair_voxels_not_retained": baseline_repairs_not_retained,
        "added_voxel_probability": probability_summary,
        "probability_is_not_ground_truth": True, "anatomical_quality_review_required": True,
        "artifact_files": files,
    }


def run(block: str, selected: str | None) -> None:
    request = context.read(context.OUTPUT / "request.json")
    context.verify_file(request["native_filler"])
    if request["presets"] != context.PRESETS:
        raise ValueError("declared presets changed")
    if block == "control" and selected is None:
        raise ValueError("control is reserved for one explicitly selected policy versus baseline")
    names = list(context.PRESETS) if block == "development" else ["report_baseline", selected]
    source = stage_input(request, block)
    root = context.OUTPUT / "repairs" / block
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    state = context.read(state_path) if state_path.exists() else {
        "schema": "f0-repair-comparison-state-v1", "request_sha256": context.sha256(context.OUTPUT / "request.json"),
        "runner": context.inventory(Path(__file__)), "pid": os.getpid(), "runs": {},
    }
    if state["runner"]["sha256"] != context.sha256(Path(__file__)):
        raise ValueError("comparison code changed; preserve existing results and declare a new version")
    for name in names:
        target = root / name
        if name in state["runs"]:
            prior = state["runs"][name]
            if prior["status"] != "complete":
                raise RuntimeError(f"inspect prior {name} process/artifacts before restarting")
            result = context.read(root / f"{name}_summary.json")
            for item in result["artifact_files"]:
                context.verify_file(item)
            continue
        if target.exists():
            raise FileExistsError(f"uncommitted output needs inspection: {target}")
        command = command_for(source, target, request["presets"][name])
        log = root / f"{name}.log"
        state["runs"][name] = {"status": "starting", "command": command, "started_utc": datetime.now(UTC).isoformat()}
        context.atomic_json(state_path, state)
        env = {**os.environ, "OMP_NUM_THREADS": "16"}
        with log.open("xb") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env,
                                       cwd=context.OUTPUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            state["runs"][name].update(status="running", pid=process.pid)
            context.atomic_json(state_path, state)
            print(f"repair {block}/{name}: PID {process.pid}; one native process", flush=True)
            code = process.wait()
        state["runs"][name].update(exit_code=code, status="native_complete" if code == 0 else "failed")
        context.atomic_json(state_path, state)
        if code:
            raise RuntimeError(f"native repair failed: {block}/{name}, exit {code}")
        result = summarize(request, block, name, target)
        context.atomic_json(root / f"{name}_summary.json", result)
        state["runs"][name].update(status="complete", completed_utc=datetime.now(UTC).isoformat(), summary=context.inventory(root / f"{name}_summary.json"))
        context.atomic_json(state_path, state)
        print(f"repair {name}: {result['native']['kept']} per-plane joins; {result['kept_connection_tracks']} tracks; {result['native']['painted_px']} added voxels; {result['core']['added_voxels']} in core", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("block", choices=("development", "control"))
    parser.add_argument("--selected", choices=("support_only", "radial_safe", "extended_safe"))
    args = parser.parse_args()
    run(args.block, args.selected)


if __name__ == "__main__":
    main()
