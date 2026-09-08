"""Run the fixed packaged patch stage only after exact development parity."""
from pathlib import Path
import sys
from dataclasses import asdict
import run_f0_repair_context as context

ROOT=context.ROOT/"output/crossres_data/f0_sheet_patches_20260907"
sys.path.insert(0,str(context.ROOT.parent/"socratic_method/src"))
from socratic_method import sheet_patches as sheets


def main():
    selected=context.read(ROOT/"selected_policy.json")
    parity=context.read(ROOT/"package_development_verified/receipt.json")
    for item in selected["code"]+parity["code"]: context.verify_file(item)
    if parity["status"]!="verified" or parity["byte_identical_tiffs"]!=27 or parity["exact_meshes"]!=107:
        raise ValueError("package parity incomplete")
    if selected["patch_policy"]!=asdict(sheets.PATCH) or selected["growth_policy"]!=asdict(sheets.GROWTH):
        raise ValueError("selected numerical policy changed")
    if context.read(ROOT/"fresh_assessment/state.json")["status"]!="complete":
        raise ValueError("fresh baseline is incomplete")
    prior=ROOT/"fresh_assessment/continuity_v2"
    output=ROOT/"fresh_assessment_growth"
    plan=sheets.plan_patches(prior,context.SOURCE,output,voxel_size_um=8.64)
    if output.exists(): raise ValueError("check-only created output")
    context.atomic_json(ROOT/"assessment_check_only_plan.json",plan)
    context.atomic_json(ROOT/"assessment_execution_request.json",{
        "selection":context.inventory(ROOT/"selected_policy.json"),"parity":context.inventory(ROOT/"package_development_verified/receipt.json"),
        "runner":context.inventory(Path(__file__)),"preflight":context.inventory(ROOT/"assessment_check_only_plan.json"),
        "no_threshold_model_or_policy_change":True,"no_assessment_based_tuning":True})
    print(sheets.run_patches(plan),flush=True)


if __name__=="__main__": main()
