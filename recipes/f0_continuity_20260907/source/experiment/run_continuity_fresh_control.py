"""One bounded F0 inference worker for a predeclared fresh continuity control.

Requires a development-selected policy first. Existing/partial control work is
never restarted automatically. The original model, threshold and source pins
are unchanged; one native baseline worker follows GPU inference sequentially.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import run_f0_repair_context as context

ROOT = context.ROOT / "output/crossres_data/f0_continuity_20260907"
CENTER = (12544,4480,3072)


def main() -> None:
    selected=context.read(ROOT / "selected_policy.json")
    if selected["status"]!="development selected; fresh control pending":
        raise ValueError("control requires a recorded development-only selection")
    destination=ROOT / "fresh_control"
    if destination.exists():
        raise FileExistsError("inspect the existing control process and artifacts; no automatic fresh restart")
    old=context.read(context.OUTPUT / "request.json")
    for item in (old["model"],old["native_filler"]):
        context.verify_file(item)
    for relative,digest in old["sealed_runtime_pins"].items():
        if context.sha256(context.ROOT / relative)!=digest:
            raise ValueError(f"sealed runtime changed: {relative}")
    points=context.block_origins(CENTER)
    spec={"center_zyx":list(CENTER),"origin_zyx":[x-128 for x in CENTER],"shape_zyx":[384]*3,
          "target_cube_ids":[context.cube_id(p) for p in points],
          "primary_assessment_core_cube":context.cube_id(CENTER)}
    raw_inputs=[context.inventory(context.SOURCE / "manifest.json")]
    raw_inputs.extend(context.inventory(context.SOURCE / "cubes_RAW" / f"{context.cube_id(p)}.tif")
                      for p in context.block_origins(CENTER,radius=2))
    official=context.ROOT.parent / "socratic_method"
    wrapper=official / "src/socratic_method/repair.py"
    request={"schema":"f0-continuity-fresh-control-request-v1","created_utc":datetime.now(UTC).isoformat(),
             "block":spec,"selection":context.inventory(ROOT / "selected_policy.json"),
             "model":old["model"],"native_binary":old["native_filler"],"native_wrapper":context.inventory(wrapper),
             "source_grid":str(context.SOURCE),"raw_inputs":raw_inputs,"runner":context.inventory(Path(__file__)),
             "sealed_runtime_pins":old["sealed_runtime_pins"],"inference":old["inference"],
             "threshold":.35,"voxel_size_um":8.64,
             "selection_basis":"Disjoint local block with genuine raw neighbors; not used in either earlier filler comparison or current development probes.",
             "no_control_based_tuning":True,"training_or_deployment":False}
    destination.mkdir(parents=True)
    context.atomic_json(destination / "request.json",request)
    state={"status":"waiting_for_gpu","pid":os.getpid(),"request_sha256":context.sha256(destination / "request.json"),
           "started_utc":datetime.now(UTC).isoformat()}
    context.atomic_json(destination / "state.json",state)
    os.environ["CUDA_VISIBLE_DEVICES"]=context.GPU
    os.environ["OMP_NUM_THREADS"]="16"
    sys.path.insert(0,str(context.ROOT / "crossres_pred/src"))
    sys.path.append(str(official / "src"))
    from run_unified_ladder import acquire_gpu_lock,release_gpu_lock
    from crossres_pred.voxel.grid_inference import infer_voxel_grid
    from socratic_method import repair
    if Path(repair.__file__).resolve()!=wrapper:
        raise ValueError("unexpected repair wrapper import")
    acquire_gpu_lock("continuity:F0:fresh-control",poll_seconds=30)
    try:
        state["status"]="inference_running"
        context.atomic_json(destination / "state.json",state)
        print(f"F0 fresh control: one GPU worker PID {os.getpid()}, 27 targets; model/threshold fixed.",flush=True)
        infer_voxel_grid(source_grid=context.SOURCE,checkpoint_path=context.MODEL,
                         output_path=destination / "inference",threshold=.35,halo=32,device_name="cuda",
                         amp_dtype_name="bfloat16",mirror_tta=True,max_cpu_threads=16,target_cube_ids=spec["target_cube_ids"])
        receipt=context.verify_inference(destination / "inference",{"blocks":{"fresh_control":spec}},"fresh_control")
        state["input_artifacts"]=[receipt["provenance"],*receipt["files"]]
        state["status"]="inference_complete"
        context.atomic_json(destination / "state.json",state)
    except BaseException as error:
        state.update(status="inference_failed",error=repr(error))
        context.atomic_json(destination / "state.json",state)
        raise
    finally:
        import torch
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        release_gpu_lock()
    print("Fresh inference complete; running one sequential native baseline worker.",flush=True)
    state["status"]="native_baseline_running"
    context.atomic_json(destination / "state.json",state)
    plan=repair.plan_repair(destination / "inference",destination / "native_baseline",context.EXE,voxel_size_um=8.64)
    repair.run_repair(plan)
    for item in [*raw_inputs,request["model"],request["native_binary"],request["native_wrapper"]]:
        context.verify_file(item)
    state.update(status="complete",completed_utc=datetime.now(UTC).isoformat(),
                 baseline_receipt=context.inventory(destination / "native_baseline/repair_receipt.json"))
    context.atomic_json(destination / "state.json",state)
    print("Fresh control inference and native baseline complete; probability-corridor evaluation remains separate.",flush=True)


if __name__=="__main__":
    main()
