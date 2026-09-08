"""Verify existing 3D patch evidence and build HTML/CT images, without fitting."""
from __future__ import annotations

import html
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
from PIL import Image, ImageDraw

import build_f0_repair_report as frozen
import run_f0_repair_context as context
from render_f0_repair_review import assemble, overlay, panel_row, FONT

ROOT=context.ROOT/"output/crossres_data/f0_sheet_patches_20260907"
RUNS={"development":ROOT/"development_growth_v2", "fresh":ROOT/"fresh_assessment_growth"}


def rel(path): return Path(os.path.relpath(path,ROOT)).as_posix()
def esc(value): return html.escape(str(value),quote=True)
def table(headers,rows):
    return "<div class='scroll'><table><thead><tr>"+"".join(f"<th>{esc(v)}</th>" for v in headers)+"</tr></thead><tbody>"+"".join("<tr>"+"".join(f"<td>{esc(v)}</td>" for v in row)+"</tr>" for row in rows)+"</tbody></table></div>"
def figure(path,caption): return f"<figure><a href='{esc(rel(path))}'><img loading='lazy' src='{esc(rel(path))}' alt='{esc(caption)}'></a><figcaption>{esc(caption)}</figcaption></figure>"


def render_review(name,folder):
    request=context.read(folder/"request.json"); summary=context.read(folder/"summary.json")
    prior=Path(request["source_continuity"])
    ids,lower=request["spec"]["target_cube_ids"],request["spec"]["origin_zyx"]
    base=assemble(prior/"cubes_PRED",ids,lower)!=0
    after=assemble(folder/"cubes_PRED",ids,lower)!=0; added=after & ~base
    ct=assemble(context.SOURCE/"cubes_RAW",ids,lower).astype(np.float32)
    lo,hi=summary["ct_window"]; gray=np.clip((ct-lo)*255/(hi-lo),0,255).astype(np.uint8)
    rows=context.read(folder/"patches.json")["patches"]
    for row in rows:
        if name=="development":
            row["report_image"]=rel(folder/row["image"]); continue
        center=row["view_center_local"]; pictures=[]
        for axis,label in enumerate("ZYX"):
            w=[slice(None)]*3; w[axis]=int(center[axis])
            for other in range(3):
                if other!=axis:
                    start=max(0,min(264,int(center[other])-60)); w[other]=slice(start,start+120)
            w=tuple(w)
            pictures.append(panel_row([overlay(gray[w]),overlay(gray[w],base[w]),overlay(gray[w],base[w],added[w])],
                                      [f"CT {label}={center[axis]+lower[axis]}","Current v2 mask","Joint 3D patch additions"],scale=2))
        image=Image.new("RGB",(pictures[0].width,32+sum(p.height for p in pictures)),"#10151e")
        ImageDraw.Draw(image).text((8,5),f"Fresh patch {row['patch_id']}, seed {row['seed_id']}: {row['metrics']['mesh_z_span']} planes; {row['unique_added_voxels']} new; {row['metrics']['modeled_surfaces']} surfaces",font=FONT,fill="white")
        y=32
        for picture in pictures: image.paste(picture,(0,y)); y+=picture.height
        path=ROOT/"report_assets"/f"fresh_patch_{row['patch_id']:04d}.png"; image.save(path)
        row["report_image"]=rel(path)
    overviews=[]
    for axis,label in enumerate("ZYX"):
        for position in (128,160,192,224,256):
            w=[slice(None)]*3; w[axis]=position; w=tuple(w)
            path=ROOT/"report_assets"/f"{name}_overview_{label}{position+lower[axis]}.png"
            panel_row([overlay(gray[w]),overlay(gray[w],base[w]),overlay(gray[w],base[w],added[w])],
                      [f"CT {label}={position+lower[axis]}","Current v2 mask","Joint 3D patch additions"]).save(path)
            overviews.append(rel(path))
    return rows,overviews


class Links(HTMLParser):
    def __init__(self): super().__init__(); self.paths=[]; self.ids=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if "id" in a: self.ids.append(a["id"])
        for k in ("src","href"):
            if k in a: self.paths.append(a[k])


def main():
    verification=frozen.verify_sources()
    official=context.ROOT.parent/"socratic_method"
    preserved=[]
    for name in ("f0_repair_20260907","f0_continuity_20260907"):
        record=official/"recipes"/name
        for row in context.read(record/"files.json")["files"]:
            context.verify_file({"path":str(record/row["relative_path"]),"bytes":row["bytes"],"sha256":row["sha256"]})
        preserved.append(context.inventory(record/"files.json"))
    selected=context.read(ROOT/"selected_policy.json")
    parity=context.read(ROOT/"package_development_verified/receipt.json")
    for item in [*selected["code"],*parity["code"],selected["development_audit"],selected["development_summary"]]: context.verify_file(item)
    results={}; galleries={}; overviews={}
    (ROOT/"report_assets").mkdir(exist_ok=True)
    for name,folder in RUNS.items():
        request,summary,audit=(context.read(folder/f"{v}.json") for v in ("request","summary","independent_audit"))
        if context.read(folder/"state.json")["status"]!="complete" or audit["status"]!="passed_declared_checks": raise ValueError("incomplete assessment")
        for item in request["input_files"]+request["code"]+summary["output_files"]: context.verify_file(item)
        if summary["erased_v2_voxels"] or audit["reverse_direction_below_090"]: raise ValueError("acceptance audit failed")
        galleries[name],overviews[name]=render_review(name,folder)
        results[name]={"summary":summary,"audit":audit,"source_summary":context.inventory(folder/"summary.json")}
    suite=ET.parse(ROOT/"package_tests_final.xml").getroot().find("testsuite")
    if int(suite.attrib["tests"])!=151 or any(int(suite.attrib[k]) for k in ("failures","errors","skipped")): raise ValueError("package regression failed")
    d,f=results["development"]["summary"],results["fresh"]["summary"]
    da,fa=results["development"]["audit"],results["fresh"]["audit"]
    rows=[("Target cubes / local shape","27 / 384³","27 / 384³"),
          ("Seed-bearing prior components",d["prior_added_components"],f["prior_added_components"]),
          ("Accepted explicit mesh patches",d["accepted_patches"],f["accepted_patches"]),
          ("Added voxels beyond v2",f"{d['added_voxels_over_v2']:,}",f"{f['added_voxels_over_v2']:,}"),
          ("Central-cube additions",f"{d['core_added_voxels_over_v2']:,}",f"{f['core_added_voxels_over_v2']:,}"),
          ("Median / maximum mesh Z span",f"{da['span_quantiles'][2]:g} / {d['maximum_mesh_z_span']}",f"{fa['span_quantiles'][2]:g} / {f['maximum_mesh_z_span']}"),
          ("Patches fitting neighboring sheets",d["patches_with_modeled_neighbors"],f["patches_with_modeled_neighbors"]),
          ("Sum of supported patch area, px²*",f"{d['summed_patch_area_px2_not_unique']:,.1f}",f"{f['summed_patch_area_px2_not_unique']:,.1f}"),
          ("New-voxel connected components",d["added_voxel_components"],f["added_voxel_components"]),
          ("All new components attach to base",True,True),
          ("Existing foreground erased",0,0),
          ("New official/anatomical Dice","not measured","not measured")]
    metrics=table(["Measure","Development","New disjoint assessment"],rows)
    reasons=sorted(set(d["census"])|set(f["census"]))
    census=table(["Per-seed outcome (up to 3 seeds/component)","Development","New assessment"],[(r,d["census"].get(r,0),f["census"].get(r,0)) for r in reasons])
    probe=context.read(ROOT/"development_probe_v3/results.json")["results"]
    ablation=[]
    for seed in (1934,3637,1993,3752,3671,4127,1039,1245):
        pair=[next(r for r in probe if r["seed"]==seed and r["mode"]==mode) for mode in ("probability_only","ct_profile")]
        ablation.append((seed,pair[0]["component"],pair[0]["supported_z_span"],pair[1]["supported_z_span"],pair[1]["proposed_new_voxels"]))
    gallery_html=""
    for name in ("development","fresh"):
        gallery_html+=f"<details><summary>All {len(galleries[name])} {name} patches — CT Z/Y/X, meshes and seeds</summary><div class='gallery'>"
        for row in galleries[name]:
            m=row["metrics"]
            gallery_html+=f"<figure><a href='{esc(row['report_image'])}'><img loading='lazy' src='{esc(row['report_image'])}' alt='{name} patch {row['patch_id']} CT-aligned three-axis view'></a><figcaption>{name} patch {row['patch_id']} · seed {row['seed_id']} · {row['unique_added_voxels']:,} new voxels · {m['mesh_z_span']} planes · {m['modeled_surfaces']} modeled surfaces. <a href='{esc(rel(RUNS[name]/row['mesh']))}'>PLY mesh</a> / <a href='{esc(rel(RUNS[name]/row['mesh_arrays']))}'>exact arrays</a></figcaption></figure>"
        gallery_html+="</div></details>"
    overview_html="".join(f"<details><summary>{name}: 15 full-block Z/Y/X planes</summary>"+"".join(figure(ROOT/p,Path(p).stem) for p in overviews[name])+"</details>" for name in RUNS)
    notable=""
    for name,ids in (("development",[5,26,70]),("fresh",[r["patch_id"] for r in sorted(galleries["fresh"],key=lambda r:-r["unique_added_voxels"])[:3]])):
        for number in ids:
            row=next(r for r in galleries[name] if r["patch_id"]==number)
            notable+=figure(ROOT/row["report_image"],f"{name} patch {number}: {row['unique_added_voxels']:,} unique additions; {row['metrics']['mesh_z_span']} Z planes. Nearby green paint can belong to other patches.")
    body=f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F0 — From paths to sheet patches</title>
<style>:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#10151e;color:#e7edf5;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1240px;margin:auto;padding:30px 24px 100px}}h1{{font-size:clamp(30px,5vw,54px);line-height:1.12;margin-bottom:18px}}h2{{margin-top:52px;color:#9ff2d1}}a{{color:#89c9ff}}p{{max-width:95ch}}.eyebrow{{color:#9faabd;letter-spacing:.12em;font-size:12px}}.note{{border-left:4px solid #f5bb61;padding:12px 20px;background:#1e2330}}.good{{border-color:#64eaa8}}nav{{display:flex;flex-wrap:wrap;gap:16px;margin:24px 0}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{text-align:left;border-bottom:1px solid #34404f;padding:11px}}th{{color:#c6d7ec;background:#192230}}.scroll{{overflow-x:auto}}figure{{margin:24px 0;background:#151d28;padding:10px}}img{{max-width:100%;height:auto;display:block;margin:auto}}figcaption{{font-size:14px;color:#c0cbd8;padding:10px 4px}}details{{margin:18px 0;border:1px solid #34404f;padding:15px}}summary{{cursor:pointer;font-weight:650}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,490px),1fr));gap:16px}}code,pre{{background:#202a38;padding:3px 6px;overflow:auto}}.legend span{{display:inline-block;margin-right:22px}}.magenta{{color:#e57be5}}.green{{color:#63f3a6}}footer{{margin-top:50px;color:#adb8c7;font-size:14px}}</style><main>
<div class="eyebrow">FROZEN F0 · POST-PROCESSING ONLY · 7 SEPTEMBER 2026</div><h1>From found paths<br>to coherent sheet patches</h1>
<p class="note good">A measurable 3D extension, without changing the model: <strong>{d['accepted_patches']} development patches</strong> and <strong>{f['accepted_patches']} patches on a new target-disjoint block</strong>. These are explicit, connected triangle meshes that paint additional supported foreground across neighboring slices. Crowded-fold identity remains a separate, unresolved problem.</p>
<nav><a href="#results">Results</a><a href="#examples">See the geometry</a><a href="#crowds">Crowded folds</a><a href="#failures">Refusals</a><a href="#use">Use / limits</a><a href="#evidence">Full evidence</a></nav>
<p>The frozen training result remains a substantial win: matched no-TTA / T=.35 macro Dice <strong>0.680787 vs M7 0.554140 (+0.126647)</strong>. This report does not recalculate that benchmark or attribute its gain to the new filler. <a href="{esc(rel(official/'recipes/final_c3_250k_20260907/review.md'))}">Full training review</a>.</p>
<h2 id="results">What actually changed</h2>{metrics}
<p>*Area is the sum of valid mesh faces, including parts already covered by the old mask; compatible patches can overlap. It is <strong>not unique recovered sheet area</strong>. At 8.640 µm, a 33-plane patch spans 276.48 µm between its first and last slice centers. Added-voxel components can be split by already-existing foreground and do not count distinct anatomical sheets.</p>
<p>Development gains {d['added_voxels_over_v2']/da['v2_foreground']*100:.2f}% foreground beyond the already large v2 baseline; the new block gains {f['added_voxels_over_v2']/fa['v2_foreground']*100:.2f}%. This is useful local recovery, not broad completion of every missing sheet. The model, T=.35 raw masks, native repair, and old v2 masks are preserved.</p>
<p>Development origin (12032,3840,2944); new assessment origin (12672,4480,2560), center (12800,4608,2688). Policy and source hashes were frozen <em>before</em> new inference. Targets are disjoint from all previous blocks; both are PHerc1447, so this is not independent-scroll or anatomical-ground-truth validation. The previous continuity control was not reused as a fresh control.</p>
<h2 id="examples">Let the geometry speak</h2><p class="legend"><span>CT: grayscale</span><span class="magenta">Existing v2 foreground: magenta</span><span class="green">New sheet-patch foreground: green</span></p><p>Click for full-size images. Every card has Z, Y and X views. Green shows all nearby accepted additions, not only the named patch. Mesh downloads in the full galleries identify the exact patch.</p>{notable}
<h2>Why these are surfaces, not thicker lines</h2><p>Each existing curve seeds a local monotone chart. A joint height field chooses displacement across both path distance and Z, under hard slope bounds. CT normal-profile similarity and model probability supply local evidence. Strong neighboring ridges, when present, are fitted together with an ordering/separation constraint. Only a contiguous evidence-supported interval around the seed becomes a connected triangle mesh; a probability-clipped mesh raster supplies new mask voxels.</p><p>The optimizer is exact for its <em>quantized, restricted height-field objective</em>, not for arbitrary folded scroll anatomy. This follows the local graph-surface formulation described by <a href="https://www.cs.cmu.edu/~kangli/doc/papers/optnet-pami.pdf">Li et al. (2006)</a>; our specific costs and acceptance rules are experimental hypotheses, not validated by that paper. Integer-capacity max-flow uses <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.maximum_flow.html">SciPy’s primary API</a>.</p>
<p>The independent audit reconstructs every output addition from saved meshes; verifies face connectivity, edge winding, boundary degrees, areas, and attachment; and audits all interacting patches in both directions. Only one accepted patch pair interacts in development and none in the new block; 26 and 20 attempted interactions were refused. This is limited real interaction evidence, not a dense-overlap stress test. The packaged loop reproduces <strong>27 development TIFFs and 107 meshes exactly</strong>, including refusal counts. All <strong>151 combined regression tests</strong> pass (75 new sheet tests, 76 existing repair/continuity tests), including exact small optimizer oracles, bent/moving synthetic sheets, absent evidence, branch-switch competition, checked I/O and simulated interruption recovery. No local ground-truth precision is available.</p>
<h2 id="crowds">Crowded folds: real progress, but no magic identity oracle</h2><p><strong>What worked:</strong> in a known-truth repeated-texture synthetic case, a single surface jumps toward a brighter parallel neighbor (mean error &gt;3 px); joint ordering holds the target to &lt;0.7 px mean error. This establishes a mechanism, not universal safety.</p><p><strong>What the real case says:</strong> component 301 / seed 3752 has strong nearby ridges only when neighbor search widens to ±24 px (target still ±8). Modeling one neighbor changes 40 vertices by at least 2 px and excludes 101 voxels from the independent single-surface proposal; the opposite neighbor leaves that proposal unchanged. These are alternatives, not accepted output or erasure of the baseline. Final construction refuses all three tried seeds in component 301 because they touch more than two existing components on a plane.</p>{figure(ROOT/'competition_probe_v2/assets/seed_03752_side_+1.png','Diagnostic only: crowded component 301, single vs ordered fit. Neither panel is automatically accepted by the final profile.')}
<p><strong>What did not work:</strong> alternative-cost margins do not flag this known visually ambiguous case as less certain than the clean examples. The model/CT objective can be confident about the wrong identity. CT appearance repeats between layers, and all probability evidence comes from the same model. Do not turn high margin or a smooth surface into a trusted anatomical label.</p>
<h2>Mechanisms tried, including failures</h2><p>The first independently transported-normal chart produced actual foldovers and was abandoned. Parallel-column monotone charts removed those caustics; bounded terminal-anchor trimming and per-column real-context bounds allowed all eight selected seed probes to be fitted. Genuine interior fold-backs are still refused, not sorted into a false chart.</p>{table(['Seed','Prior component','Z span without CT unary','Z span with CT unary','CT-fit proposal voxels (not accepted)'],ablation)}<p>These are diagnostic proposals, not the final accepted meshes. “Without CT unary” still uses the common CT-correlation quality gate and CT-derived motion initialization; this is a CT-cost ablation, not a fully CT-free method. CT helps some cases and shortens others. Core seed 1934 supports only four planes; the final five-plane minimum correctly prevents claiming a broad extension there.</p>
<h2 id="failures">Every attempted seed has an outcome</h2>{census}<p>The two-component contact rule is still deliberately conservative. It can reject a useful continuation when one anatomical sheet is fragmented into several binary components: development component 292 is an example requiring adjudication, not proof that relaxing the count is safe. Seed-bearing components exclude small anchor-apron components without a found path. Up to three longest seeds are tried per component; these rows are not counts of independent anatomical failures.</p>
<p>A separate <a href="independent_contact_audit.json">contact reconstruction</a> verifies all 1,393 development and 1,324 fresh patch-plane contact records against full-plane binary labels. None exceeds two existing components. Global 26-connected component counts change 310→300 and 98→96; these are descriptive connectivity changes, not counts of correctly recovered anatomical joins.</p>
<p>The first full development execution completed its masks/meshes but failed serializing a NumPy integer in final metadata. A separate metadata-only correction produced 27 byte-identical TIFFs; the partial evidence is retained. An early ordered-solver diagnostic reported layer separation as maximum Z step; the corrected v2 diagnostic is used in the selected bundle. Neither issue is concealed as an extra successful experiment.</p>
<h2 id="use">Use it as an opt-in local tool</h2><p><code>python -m socratic_method.sheet_patches CONTINUITY_DIR RAW_GRID NEW_OUTPUT --voxel-size-um 8.64</code> performs preflight only. Add <code>--run</code> for one CPU patch process. It requires a completed checked v2 continuity receipt, real aligned uint8 CT and the original probability cache. The qualified profile is exactly a 3×3×3 block of 128³ cubes; use interior outputs and genuine context. It is not a whole-scroll tiler.</p><p>Outputs include new masks, exact local-ZYX mesh arrays, world-voxel XYZ PLYs, incremental seed ledger, input/code hashes, summaries and failure/partial records. PLY units are <strong>voxels</strong> (multiply by 8.64 for µm), not millimeters. Existing output/locks/partials are never silently overwritten or reclaimed. See <a href="{esc(rel(official/'docs/sheet_patches.md'))}">checked-command documentation</a>.</p>
<p class="note"><strong>Next engineering target:</strong> give crowded folds explicit sheet identity, using 3D component ownership and sparse manual same-sheet / different-sheet anchors around ambiguous junctions. That is more defensible than relaxing the global contact count or trusting the energy margin. Then split genuine fold-backs into overlapping valid charts with agreement at overlaps. Continue larger-scale tiling only after those identity decisions and boundary seams are assessed. These next steps are proposals, not work already performed.</p>
<h2 id="evidence">Full-field review and complete patch galleries</h2>{overview_html}{gallery_html}
<p><a href="results.json">Machine-readable results</a> · <a href="selected_policy.json">Pre-assessment selection</a> · <a href="decision.json">Decision / limits</a> · <a href="visual_review.json">Actual visual-review record</a> · <a href="verification.json">Integrity and HTML checks</a> · <a href="package_tests_final.xml">151 regression tests</a> · <a href="development_growth_v2/ledger.json">All development attempts</a> · <a href="fresh_assessment_growth/ledger.json">All new-block attempts</a></p>
<footer>Local research artifact. No training, model/threshold change, automatic deployment, public upload, or new official Dice. Images were rendered directly from numerical CT/mask arrays; no image-generation model or anatomical reconstruction was used. Browser backend unavailable: image-level and static HTML QA are recorded separately, not described as a browser render test.</footer></main></html>"""
    (ROOT/"index.html").write_text(body,encoding="utf-8")
    context.atomic_json(ROOT/"results.json",results)
    context.atomic_json(ROOT/"gallery_index.json",{"galleries":galleries,"overviews":overviews})
    context.atomic_json(ROOT/"report_integrity_inputs.json",{"old_frozen_sources":verification,"prior_official_records":preserved,"package_parity":context.inventory(ROOT/"package_development_verified/receipt.json"),"package_tests":context.inventory(ROOT/"package_tests_final.xml"),"created_utc":datetime.now(UTC).isoformat()})
    print(f"Report and {len(galleries['fresh'])} fresh patch cards rendered: {ROOT/'index.html'}",flush=True)


if __name__=="__main__": main()
