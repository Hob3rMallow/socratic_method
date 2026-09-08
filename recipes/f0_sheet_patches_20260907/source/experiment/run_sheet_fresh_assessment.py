"""Freeze a development-only policy, then prepare the predeclared new block.

--freeze writes selection provenance without fitting. --run performs one GPU
inference, one sequential native baseline, and existing v2 continuity. The 3D
patch stage stays separate. No restart or reclaim of unknown partial work.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import run_f0_repair_context as context
import sheet_patch_chart_v3 as chart
import sheet_patch_growth_v2 as growth
from evaluate_sheet_patch_growth_v2 import SOURCES

ROOT = context.ROOT / "output/crossres_data/f0_sheet_patches_20260907"
CENTER = (12800, 4608, 2688)
OFFICIAL = context.ROOT.parent / "socratic_method"


def declaration():
    ids = [context.cube_id(p) for p in context.block_origins(CENTER)]
    used = []
    for center in [(12160,3968,3072), (12800,4096,3328), (12544,4480,3072)]:
        used.extend(context.cube_id(p) for p in context.block_origins(center))
    if set(ids).intersection(used):
        raise ValueError("assessment targets overlap previous targets")
    raw = [context.inventory(context.SOURCE / "manifest.json")]
    raw.extend(context.inventory(context.SOURCE / "cubes_RAW" / f"{context.cube_id(p)}.tif")
               for p in context.block_origins(CENTER, radius=2))
    return {"center_zyx": list(CENTER), "origin_zyx": [v-128 for v in CENTER],
            "shape_zyx": [384]*3, "target_cube_ids": ids,
            "primary_assessment_core_cube": context.cube_id(CENTER)}, raw


def freeze():
    path = ROOT / "selected_policy.json"
    if path.exists() or (ROOT / "fresh_assessment").exists():
        raise FileExistsError("selection/assessment already exists")
    audit = context.read(ROOT / "development_growth_v2/independent_audit.json")
    if audit["status"] != "passed_declared_checks" or audit["reverse_direction_below_090"]:
        raise ValueError("development audit not ready")
    spec, raw = declaration()
    selected = {"schema": "f0-sheet-selection-v1", "status": "development selected; fresh assessment pending",
                "created_utc": datetime.now(UTC).isoformat(), "patch_policy": asdict(chart.PatchPolicy()),
                "growth_policy": asdict(growth.GrowthPolicy()),
                "code": [context.inventory(Path(__file__).parent / n) for n in SOURCES],
                "model_sha256": context.MODEL_SHA, "threshold": .35, "voxel_size_um": 8.64,
                "development_audit": context.inventory(ROOT / "development_growth_v2/independent_audit.json"),
                "development_summary": context.inventory(ROOT / "development_growth_v2/summary.json"),
                "fresh_assessment_spec": spec, "raw_inputs": raw,
                "no_assessment_based_tuning": True, "automatic_deployment": False,
                "selection_reason": "Explicit connected meshes, joint ordered neighbors, evidence-bounded Z growth; 107 development patches, 75661 additions, exact independent mask/mesh reconstruction. Crowded component 301 refused, not solved.",
                "assessment_contract": ["Zero erased foreground", "finite connected consistently oriented meshes", "mask equals supported raster union", "all additions >= .16 probability and attached to baseline", "audit all interacting patches and Z/Y/X examples", "report all refusals and no local anatomical ground-truth score"],
                "limits": ["local monotone charts only", "CT/P identity not anatomical confidence", "up to three seeds per prior component; minimum five supported planes", "does not remove incorrect pre-existing joins", "same scroll; target-disjoint, not statistically independent scrolls", "sum of patch areas may overlap"]}
    context.atomic_json(path, selected)
    print(f"Selection frozen before new inference: {path}", flush=True)


def run():
    selected = context.read(ROOT / "selected_policy.json")
    for item in selected["code"] + selected["raw_inputs"]:
        context.verify_file(item)
    if selected["patch_policy"] != asdict(chart.PatchPolicy()) or selected["growth_policy"] != asdict(growth.GrowthPolicy()):
        raise ValueError("selected policy changed")
    destination = ROOT / "fresh_assessment"
    if destination.exists() or list(ROOT.glob("fresh_assessment.partial-*")):
        raise FileExistsError("existing assessment requires inspection")
    old = context.read(context.OUTPUT / "request.json")
    for item in (old["model"], old["native_filler"]):
        context.verify_file(item)
    for relative, digest in old["sealed_runtime_pins"].items():
        if context.sha256(context.ROOT / relative) != digest:
            raise ValueError(f"sealed runtime changed: {relative}")
    spec = selected["fresh_assessment_spec"]
    request = {"selection": context.inventory(ROOT / "selected_policy.json"),
               "runner": context.inventory(Path(__file__)), "block": spec,
               "model": old["model"], "raw_inputs": selected["raw_inputs"],
               "inference": old["inference"], "threshold": .35,
               "single_gpu_worker_then_sequential_cpu_stages": True, "no_training": True}
    sys.path.insert(0, str(OFFICIAL / "src"))
    from socratic_method import continuity, repair
    request["postprocessor_code"] = [context.inventory(Path(m.__file__))
                                     for m in (repair, continuity, continuity.core, continuity.engine)]
    destination.mkdir()
    context.atomic_json(destination / "request.json", request)
    state = {"status": "waiting_for_gpu", "pid": os.getpid(), "started_utc": datetime.now(UTC).isoformat()}
    context.atomic_json(destination / "state.json", state)
    os.environ["CUDA_VISIBLE_DEVICES"] = context.GPU
    os.environ["OMP_NUM_THREADS"] = "16"
    sys.path.insert(0, str(context.ROOT / "crossres_pred/src"))
    from run_unified_ladder import acquire_gpu_lock, release_gpu_lock
    from crossres_pred.voxel.grid_inference import infer_voxel_grid
    locked = False
    try:
        acquire_gpu_lock("sheet-patches:F0:fresh-assessment", poll_seconds=30)
        locked = True
        state["status"] = "inference_running"
        context.atomic_json(destination / "state.json", state)
        infer_voxel_grid(source_grid=context.SOURCE, checkpoint_path=context.MODEL,
                         output_path=destination / "inference", threshold=.35, halo=32,
                         device_name="cuda", amp_dtype_name="bfloat16", mirror_tta=True,
                         max_cpu_threads=16, target_cube_ids=spec["target_cube_ids"])
        receipt = context.verify_inference(destination / "inference", {"blocks": {"assessment": spec}}, "assessment")
        state["inference_artifacts"] = [receipt["provenance"], *receipt["files"]]
        import torch
        torch.cuda.empty_cache()
        release_gpu_lock()
        locked = False
        state["status"] = "native_baseline_running"
        context.atomic_json(destination / "state.json", state)
        repair.run_repair(repair.plan_repair(destination / "inference", destination / "native_baseline", context.EXE, voxel_size_um=8.64))
        state["status"] = "v2_continuity_running"
        context.atomic_json(destination / "state.json", state)
        continuity.run_continuity(continuity.plan_continuity(destination / "inference", destination / "native_baseline", destination / "continuity_v2", voxel_size_um=8.64))
        for item in selected["code"] + selected["raw_inputs"] + request["postprocessor_code"]:
            context.verify_file(item)
        state.update(status="complete", completed_utc=datetime.now(UTC).isoformat())
        context.atomic_json(destination / "state.json", state)
        print("Fresh inference/native/v2 complete; selected 3D patch assessment is separate.", flush=True)
    except BaseException as error:
        state.update(status="failed_or_interrupted", error=repr(error))
        context.atomic_json(destination / "state.json", state)
        raise
    finally:
        if locked:
            release_gpu_lock()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run"))
    {"freeze": freeze, "run": run}[parser.parse_args().phase]()
