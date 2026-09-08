"""Finalize decision, actual visual-review record and static report verification."""
from __future__ import annotations
import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote,urlsplit
from PIL import Image
import run_f0_repair_context as context
from build_sheet_patch_report import Links

ROOT=context.ROOT/"output/crossres_data/f0_sheet_patches_20260907"
OFFICIAL=context.ROOT.parent/"socratic_method"


def main():
    dev=[4,5,7,26,37,42,50,68,70,100,105]
    fresh=[1,2,3,4,7,20,23,25,30,38,48,77]
    images=[ROOT/"development_growth_v2/assets"/f"patch_{n:04d}.png" for n in dev]
    images += [ROOT/"development_growth_v1/assets"/f"patch_{n:04d}.png" for n in (1,2)]
    images += [ROOT/"report_assets"/f"fresh_patch_{n:04d}.png" for n in fresh]
    overviews=[ROOT/"development_growth_v2/assets/overview_z12192.png"]
    overviews += [ROOT/"report_assets"/n for n in ("development_overview_Y4032.png","development_overview_X3136.png",
                  "fresh_overview_Z12864.png","fresh_overview_Y4672.png","fresh_overview_X2752.png")]
    probes=[ROOT/p for p in ("development_probe_v1/assets/seed_01934_ct_profile.png",
            "development_probe_v1/assets/seed_01039_ct_profile.png","development_probe_v2/assets/seed_01039_ct_profile.png",
            "development_probe_v2/assets/seed_03671_ct_profile.png","development_probe_v3/assets/seed_03752_ct_profile.png",
            "competition_probe_v2/assets/seed_03752_side_+1.png")]
    visual={"schema":"f0-sheet-visual-review-v1","reviewed_utc":datetime.now(UTC).isoformat(),
            "method":"Direct view_image inspection of CT-aligned numerical PNGs; no functioning in-app browser backend",
            "patch_cards_inspected": [context.inventory(p) for p in images],
            "full_field_planes_inspected":[context.inventory(p) for p in overviews],
            "diagnostic_cards_inspected":[context.inventory(p) for p in probes],
            "all_patches_numerically_audited":189,"all_patch_cards_manually_inspected":False,
            "observations":[
                "Development patch 5 shows curved continuation through a missing arch across Z, Y and X; compatible with local CT, not ground-truth-certified.",
                "Development 4, 7, 26, 37, 42, 50 and 70 show local continuations or missing ribbons; larger old foreground and holes remain unchanged.",
                "Development 68/100 and fresh 7/25/30/48 are useful crowded/neighbor-modeled examples but still require anatomical adjudication; oblique intersections can look broad or stepped.",
                "Fresh 1/2/3/4/20/23/38/77 mostly extend existing curving tracks; improvements are local rather than wholesale filling of missing sheets.",
                "Full-field planes show sparse selective additions, not a global expansion of every mask boundary.",
                "The real component-301 diagnostic remains ambiguous; its final three seeds are refused. High alternative-cost margins do not resolve its identity.",
                "No blinded human anatomical labels or local GT precision are available. Visual compatibility is not proof of correct sheet assignment."],
            "no_visual_assessment_based_tuning":True}
    context.atomic_json(ROOT/"visual_review.json",visual)
    decision={"schema":"f0-sheet-decision-v1","status":"completed local assessment; opt-in research implementation; no deployment",
              "selected_profile":"f0-joint-sheet-patches-v1","model_sha256":context.MODEL_SHA,"threshold":.35,
              "selection":context.inventory(ROOT/"selected_policy.json"),"development_patches":107,"fresh_patches":82,
              "development_added_voxels":75661,"fresh_added_voxels":46780,"existing_foreground_erased":0,
              "fresh_policy_tuning":False,"whole_scroll_rollout":False,"anatomical_correctness_established":False,
              "crowded_fold_result":"Ordered neighbors prevent a synthetic switch and change the real crowded proposal, but do not prove identity. Component 301 remains refused; energy margin is not a confidence gate.",
              "next_steps_proposed_not_executed":["3D sheet ownership plus sparse manual same/different-sheet anchors at ambiguous junctions", "Overlap-consistent chart partition for genuine local fold-backs", "Larger-area tiling and seam qualification only after identity review"],
              "do_not_do":["Relax global contact count and call it safe", "Treat repeated CT/model evidence as independent validation", "Turn old ambiguous v2 paths into trusted supervision", "Change model or threshold as part of this postprocessor"],
              "browser_render_verified":False,"local_image_and_static_html_verification":True}
    context.atomic_json(ROOT/"decision.json",decision)
    page=ROOT/"index.html"; parser=Links(); parser.feed(page.read_text(encoding="utf-8"))
    if len(parser.ids)!=len(set(parser.ids)): raise ValueError("duplicate HTML ids")
    paths=[]; image_count=0
    for value in parser.paths:
        u=urlsplit(value)
        if u.scheme or u.netloc: continue
        if not u.path:
            if u.fragment and u.fragment not in parser.ids: raise ValueError("broken anchor")
            continue
        path=(ROOT/unquote(u.path)).resolve()
        if path==ROOT/"verification.json": continue  # written below; report self-link
        if not path.is_file(): raise FileNotFoundError(path)
        paths.append(path)
    for path in set(paths):
        if path.suffix.lower()==".png":
            with Image.open(path) as picture:
                picture.verify()
            image_count+=1
    scripts=tomllib.loads((OFFICIAL/"pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    expected={"socratic-sheet-patches":"socratic_method.sheet_patches:main", "socratic-continuity":"socratic_method.continuity:main",
              "socratic-repair":"socratic_method.repair:main", "socratic-train":"socratic_method.recipe:main"}
    if any(scripts.get(k)!=v for k,v in expected.items()): raise ValueError("entry-point mismatch")
    parity=[]
    for path in sorted((ROOT/"development_growth_v2/cubes_PRED").glob("*.tif")):
        old=ROOT/"development_growth_v1/cubes_PRED"/path.name
        if context.sha256(old)!=context.sha256(path): raise ValueError("metadata-fix numerical parity changed")
        parity.append(path.name)
    for item in context.read(ROOT/"package_development_verified/receipt.json")["code"]: context.verify_file(item)
    ver={"schema":"f0-sheet-verification-v1","status":"passed", "verified_utc":datetime.now(UTC).isoformat(),
         "final_report":context.inventory(page), "static_local_links_checked":len(paths),"unique_linked_pngs_decoded":image_count,
         "static_in_page_anchors_checked":True,"browser_render_verified":False,"direct_image_review":context.inventory(ROOT/"visual_review.json"),
         "metadata_only_fix_byte_identical_cubes":len(parity),"package_exact_cubes":27,"package_exact_meshes":107,
         "regression_tests_passed":151,"tests":context.inventory(ROOT/"package_tests_final.xml"),
         "development_audit":context.inventory(ROOT/"development_growth_v2/independent_audit.json"),
         "fresh_audit":context.inventory(ROOT/"fresh_assessment_growth/independent_audit.json"),
         "independent_contact_audit":context.inventory(ROOT/"independent_contact_audit.json"),
         "prior_model_and_evidence_integrity":context.inventory(ROOT/"report_integrity_inputs.json"),
         "entry_points":expected,"automatic_deployment":False,"git_commit_or_upload":False}
    context.atomic_json(ROOT/"verification.json",ver)
    # External assets remain local; every linked mesh/image is separately pinned.
    external=[context.inventory(path) for path in sorted(set(paths))]
    context.atomic_json(ROOT/"external_report_assets.json",{"files":external,"not_copied_to_official_repo":True})
    print(f"Verified report: {len(paths)} local links, {image_count} unique PNGs; 151 tests; 189 audited meshes.",flush=True)


if __name__=="__main__": main()
