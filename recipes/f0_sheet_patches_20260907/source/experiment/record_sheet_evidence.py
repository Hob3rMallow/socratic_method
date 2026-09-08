"""Copy final text/source evidence to a separate official record, hash-checking each file."""
from __future__ import annotations
import json
import shutil
from datetime import UTC,datetime
from pathlib import Path
import run_f0_repair_context as context

ROOT=context.ROOT/"output/crossres_data/f0_sheet_patches_20260907"
OFFICIAL=context.ROOT.parent/"socratic_method"
RECORD=OFFICIAL/"recipes/f0_sheet_patches_20260907"


def main():
    ver=context.read(ROOT/"verification.json")
    if ver["status"]!="passed" or ver["regression_tests_passed"]!=151: raise ValueError("verification incomplete")
    context.verify_file(ver["final_report"])
    for item in context.read(ROOT/"external_report_assets.json")["files"]: context.verify_file(item)
    for item in context.read(ROOT/"package_development_verified/receipt.json")["code"]: context.verify_file(item)
    mapping={name:ROOT/name for name in ("PLAN.md","selected_policy.json","decision.json","results.json","visual_review.json","verification.json",
             "independent_contact_audit.json","external_report_assets.json","report_integrity_inputs.json","package_tests.xml","package_tests_final.xml",
             "growth_tests_v2.xml","assessment_check_only_plan.json","assessment_execution_request.json")}
    mapping["README.md"]=RECORD/"README.md"
    for name in ("development_growth_v2","fresh_assessment_growth"):
        for filename in ("request.json","state.json","summary.json","ledger.json","patches.json","independent_audit.json"):
            mapping[f"provenance/{name}/{filename}"]=ROOT/name/filename
    for name in ("development_probe_v1","development_probe_v2","development_probe_v3","competition_probe_v1","competition_probe_v2"):
        for filename in ("request.json","results.json"):
            mapping[f"provenance/{name}/{filename}"]=ROOT/name/filename
    for filename in ("request.json","state.json","failure.json"):
        mapping[f"provenance/development_growth_v1/{filename}"]=ROOT/"development_growth_v1"/filename
    for filename in ("request.json","state.json","inference/provenance.json","inference/source_manifest.json",
                     "native_baseline/repair_receipt.json","continuity_v2/continuity_receipt.json","continuity_v2/corridors.jsonl"):
        mapping[f"provenance/fresh_assessment/{filename}"]=ROOT/"fresh_assessment"/filename
    mapping["provenance/package_development/receipt.json"]=ROOT/"package_development_verified/receipt.json"
    names=("sheet_surface_graph.py","sheet_surface_ordered_graph.py","sheet_surface_ordered_graph_v2.py","sheet_patch_prototype.py",
           "sheet_patch_chart_v2.py","sheet_patch_chart_v3.py","sheet_patch_competition.py","sheet_patch_competition_v2.py","sheet_patch_bundle.py",
           "sheet_patch_growth.py","sheet_patch_growth_v2.py","evaluate_sheet_patch_growth.py","evaluate_sheet_patch_growth_v2.py",
           "probe_sheet_patches.py","probe_sheet_patches_v2.py","probe_sheet_patches_v3.py","probe_sheet_competition.py","probe_sheet_competition_v2.py",
           "audit_sheet_patch_outputs.py","audit_sheet_contacts.py","run_sheet_fresh_assessment.py","run_checked_sheet_assessment.py",
           "verify_sheet_package_development.py","build_sheet_patch_report.py","finalize_sheet_evidence.py","record_sheet_evidence.py")
    for filename in names: mapping[f"source/experiment/{filename}"]=Path(__file__).parent/filename
    for path in sorted((OFFICIAL/"src/socratic_method").glob("sheet_*.py")): mapping[f"source/package/{path.name}"]=path
    for path in sorted((OFFICIAL/"tests").glob("test_sheet*.py")): mapping[f"source/package_tests/{path.name}"]=path
    for filename in ("README.md","docs/sheet_patches.md","pyproject.toml"): mapping[f"source/repository/{filename}"]=OFFICIAL/filename
    rows=[]
    for relative,source in sorted(mapping.items()):
        target=(RECORD/relative).resolve()
        if RECORD.resolve() not in target.parents: raise ValueError("record target escapes scope")
        identity=context.inventory(source)
        if target.exists() and context.sha256(target)!=identity["sha256"]: raise FileExistsError(f"preserve different evidence: {target}")
        rows.append({"relative_path":relative,"original_path":str(source),"bytes":identity["bytes"],"sha256":identity["sha256"]})
    for row in rows:
        target=RECORD/row["relative_path"]
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(row["original_path"],target)
        context.verify_file({"path":str(target),"bytes":row["bytes"],"sha256":row["sha256"]})
    manifest={"schema":"f0-sheet-official-evidence-v1","files":rows,"created_utc":datetime.now(UTC).isoformat(),
              "local_html_report":context.inventory(ROOT/"index.html"),"no_arrays_or_weights_copied":True,"no_commit_or_public_upload":True}
    path=RECORD/"files.json"
    if path.exists():
        if context.read(path)["files"]!=rows: raise FileExistsError("different official manifest exists")
    else:
        with path.open("x",encoding="utf-8",newline="\n") as stream:
            json.dump(manifest,stream,indent=2); stream.write("\n")
    context.atomic_json(ROOT/"official_copy_receipt.json",{"status":"verified","files_copied_and_rechecked":len(rows),"record":str(RECORD),
                        "manifest":context.inventory(path),"local_html":context.inventory(ROOT/"index.html"),"local_only":True})
    print(f"Verified {len(rows)} official text/source evidence files: {RECORD}",flush=True)


if __name__=="__main__": main()
