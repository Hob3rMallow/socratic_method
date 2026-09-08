"""Verify continuity evidence and render local HTML; no model/repair runs."""

from __future__ import annotations

import html
import importlib.metadata
import json
import os
import sys
import tomllib
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

import build_f0_repair_report as previous
import run_f0_repair_context as context
from render_f0_repair_review import assemble

ROOT = context.ROOT / "output/crossres_data/f0_continuity_20260907"
OFFICIAL = context.ROOT.parent / "socratic_method"
sys.path.insert(0, str(OFFICIAL / "src"))
from socratic_method import continuity as installed

read, identity, check = context.read, context.inventory, previous.check


def rel(path):
    return Path(os.path.relpath(path, ROOT)).as_posix()


def verify_runs():
    fresh = read(ROOT / "fresh_control/request.json")
    context.verify_file(fresh["selection"])
    selected = read(ROOT / "selected_policy.json")
    check(asdict(installed.PROPOSAL) == selected["proposal"] and asdict(installed.ACCEPTANCE) == selected["acceptance"], "installed policy differs")
    check(read(ROOT / "fresh_control/state.json")["status"] == "complete", "inference incomplete")
    results = {}
    inventories = []
    for name in ("development_candidate_v1", "development_candidate_v2", "fresh_control_candidate_v2"):
        folder = ROOT / name
        request, state, summary = (read(folder / f"{n}.json") for n in ("request", "state", "summary"))
        check(state["status"] == "complete", "unfinished comparison")
        context.verify_file(state["summary"])
        context.verify_file(summary["source_request"])
        for item in request["source_artifacts"]:
            context.verify_file(item)
        for key in ("engine", "base_engine", "runner"):
            item = request.get(key)
            if item:
                if name.endswith("v1") and key == "runner":
                    item = {**item, "path": str(folder / "source/evaluate_probability_continuity.py")}
                context.verify_file(item)
        if name.endswith("v2"):
            check(request["proposal"] == selected["proposal"] and request["acceptance"] == selected["acceptance"], "run policy differs from selection")
            check(request["engine"]["sha256"] == selected["engine_sha256"], "engine selection mismatch")
        for item in summary["output_files"]:
            context.verify_file(item)
        rows = read(folder / "corridors.json")["corridors"]
        accepted = [r for r in rows if r["accepted"]]
        check(len(accepted) == summary["accepted_corridors"], "ledger count mismatch")
        check(sum(r["added_voxels"] for r in accepted) == summary["added_voxels"], "ledger additions mismatch")
        results[name] = {k: summary[k] for k in (
            "accepted_corridors", "accepted_longer_than_21px", "maximum_corridor_length_px",
            "added_voxels", "erased_base_voxels", "base_foreground", "core_added_voxels",
            "added_3d_components", "detached_2d_strokes", "multiway_2d_strokes", "census")}
        results[name]["sum_accepted_path_lengths_px"] = summary["total_recovered_centerline_px"]
        results[name]["source_summary"] = identity(folder / "summary.json")
        results[name]["accepted_with_axial_p_rejection"] = summary["census"].get("insufficient_neighbor_probability_support", 0)
        if name.endswith("v2"):
            # Recompute from final arrays, independent of the original runner's
            # claimed contact counters, including endpoint-before/after labels.
            ids, lower = request["spec"]["target_cube_ids"], request["spec"]["origin_zyx"]
            base = assemble(Path(request["baseline"]) / "cubes_PRED", ids, lower) != 0
            after = assemble(folder / "cubes_PRED", ids, lower) != 0
            totals = Counter()
            for z in range(384):
                ledger = [r for r in accepted if r["z_local"] == z]
                totals.update(installed.audit_plane(base[z], after[z], ledger))
            check(totals["two_contact_strokes"] == len(accepted), "bridge/stroke mismatch")
            check(totals["added_voxels"] == summary["added_voxels"], "independent additions mismatch")
            check(not (base & ~after).any(), "erased baseline")
            check(np.array_equal(base[:8], after[:8]) and np.array_equal(base[-8:], after[-8:]), "Z boundary changed")
            raw = assemble(Path(request["inference"]) / "cubes_PRED", ids, lower) != 0
            results[name].update(independent_audit=dict(totals), native_added_voxels=int((base & ~raw).sum()),
                                 native_core_added_voxels=int((base & ~raw)[128:256,128:256,128:256].sum()),
                                 total_added_over_raw=int((after & ~raw).sum()), raw_foreground=int(raw.sum()))
        inventories.append({"run": name, "request": identity(folder / "request.json"),
                            "state": identity(folder / "state.json"),
                            "corridors": identity(folder / "corridors.json"),
                            "components": identity(folder / "components.json")})
    dev_ids = read(ROOT / "development_candidate_v2/request.json")["spec"]["target_cube_ids"]
    old_control_ids = read(context.OUTPUT / "request.json")["blocks"]["control"]["target_cube_ids"]
    control_ids = fresh["block"]["target_cube_ids"]
    check(not set(control_ids).intersection(dev_ids + old_control_ids), "fresh targets overlap earlier blocks")
    return results, inventories


def verify_package():
    folder = ROOT / "package_control_verified"
    receipt = read(folder / "continuity_receipt.json")
    check(receipt["status"] == "complete", "package integration incomplete")
    check(receipt["accepted_corridors"] == 515 and receipt["added_voxels"] == 31951, "package counters differ")
    for item in receipt["plan"]["input_files"]:
        context.verify_file(item)
    context.verify_file({**receipt["ledger"], "path": str(folder / receipt["ledger"]["path"])})
    matches = []
    for item in receipt["output_files"]:
        context.verify_file({**item, "path": str(folder / item["path"])})
        reference = ROOT / "fresh_control_candidate_v2" / item["path"]
        check(context.sha256(reference) == item["sha256"], "package output differs byte for byte")
        matches.append(item)
    check(len(matches) == 27, "incomplete package integration")
    suite = ET.parse(ROOT / "package_tests.xml").getroot().find("testsuite")
    check(suite is not None and int(suite.attrib["tests"]) == 83, "wrong regression count")
    check(all(int(suite.attrib[k]) == 0 for k in ("errors", "failures", "skipped")), "regressions failed")
    scripts = tomllib.loads((OFFICIAL / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    for name, expected in {"socratic-continuity": "socratic_method.continuity:main", "socratic-repair": "socratic_method.repair:main", "socratic-train": "socratic_method.recipe:main", "socratic-export": "socratic_method.hf_export:main"}.items():
        check(scripts[name] == expected, "entry point changed incorrectly")
    destination = ROOT / "final_check_only_no_output"
    plan = installed.plan_continuity(ROOT / "fresh_control/inference", ROOT / "fresh_control/native_baseline", destination, voxel_size_um=8.64)
    check(not destination.exists(), "check-only wrote output")
    context.atomic_json(ROOT / "final_check_only_plan.json", plan)
    return {"status": "verified", "receipt": identity(folder / "continuity_receipt.json"),
            "byte_identical_tiffs": matches, "tests_passed": 83, "test_receipt": identity(ROOT / "package_tests.xml"),
            "no_output_from_check_only": True, "entry_points": scripts,
            "initial_strict_audit_failure_preserved": str(ROOT / "package_control.partial-57780"),
            "environment": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-image", "tifffile", "Pillow")}}


STYLE = """
:root{color-scheme:dark;--bg:#10151e;--panel:#192230;--rule:#354456;--muted:#b5c3d5;--link:#73d6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#edf2f9;font:17px/1.6 system-ui,sans-serif}
main{max-width:1200px;margin:auto;padding:40px 24px 80px}h1{font-size:clamp(32px,4vw,49px);line-height:1.15;max-width:900px}h2{margin-top:42px;font-size:26px}a{color:var(--link)}a:focus-visible,summary:focus-visible{outline:3px solid var(--link)}
.lead{font-size:21px;color:var(--muted)}.eyebrow{color:var(--link);font-size:13px;text-transform:uppercase;letter-spacing:.1em}
.cards,.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.card,figure,details{padding:16px;border:1px solid var(--rule);border-radius:10px;background:var(--panel)}.card b{display:block;font-size:32px}.muted,figcaption{color:var(--muted)}
figure{margin:24px 0}img{display:block;max-width:100%;height:auto;margin:auto}figcaption{font-size:14px;margin-top:12px}.grid figure{margin:0}summary{cursor:pointer;font-weight:600}details{margin:20px 0}.note{border-left:4px solid #efb965;background:#262520;padding:16px 20px}
.table-wrap,pre{overflow:auto}table{border-collapse:collapse;width:100%;font-size:15px;font-variant-numeric:tabular-nums}th,td{padding:12px;text-align:left;border-bottom:1px solid var(--rule)}th{color:var(--muted)}pre{padding:16px;background:var(--panel)}code{font-size:.9em;color:#b7e8ff}footer{border-top:1px solid var(--rule);margin-top:40px;padding-top:20px;font-size:14px;color:var(--muted)}.magenta{color:#da41d5}.green{color:#3cff80}
@media(max-width:700px){main{padding:24px 14px}.cards,.grid{grid-template-columns:1fr}th,td{padding:8px}}
"""


def page(title, body):
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{STYLE}</style></head><body><main>{body}</main></body></html>'


def fig(path, caption):
    p, c = html.escape(path, quote=True), html.escape(caption, quote=True)
    return f'<figure><a href="{p}"><img loading="lazy" src="{p}" alt="{c}"></a><figcaption>{c}</figcaption></figure>'


def gallery(name):
    folder = ROOT / name
    rows = [r for r in read(folder / "corridors.json")["corridors"] if r["accepted"]]
    cards = "".join(fig(r["image"], f"Corridor {r['id']}; z{r['z_world']}; path length {r['length_px']:.1f}px; {r['added_voxels']} added voxels; added 3D component {r['added_component']}. CT / native baseline / probability-guided additions.") for r in rows)
    body = f'<p><a href="../index.html">Return to report</a></p><h1>{html.escape(name.replace("_", " "))}</h1><p>{len(rows)} accepted per-plane corridors. All cards are available; not all were individually adjudicated. Each image includes nearby additions, not just the named path. <a href="corridors.json">All accepted/refused candidates</a>.</p>{cards}'
    (folder / "index.html").write_text(page(name, body), encoding="utf-8")


def render(results, package):
    dev, control = (results[n] for n in ("development_candidate_v2", "fresh_control_candidate_v2"))
    table = ""
    for label, r in (("Development: skeleton v1", results["development_candidate_v1"]), ("Development: refined v2", dev), ("Fresh control: fixed v2", control)):
        cells = [label, r["accepted_corridors"], r["accepted_longer_than_21px"], f'{r["added_voxels"]:,}', f'{r["core_added_voxels"]:,}', r["added_3d_components"]]
        table += "<tr>" + "".join(f"<td>{v}</td>" for v in cells) + "</tr>"
    risk = ""
    for name in ("development_candidate_v2", "fresh_control_candidate_v2"):
        components = [r for r in read(ROOT / name / "components.json")["components"] if "image" in r]
        risk += f'<details><summary>{name.replace("_", " ")}: all {len(components)} risk-first Z/Y/X cards</summary><div class="grid">'
        risk += "".join(fig(f'{name}/{r["image"]}', f'Component {r["id"]}: {r["voxels"]} voxels, {r["z_span"]} Z planes, radial extent {r["radial_extent_px"]:.1f}px. Extent is not a radial safety certificate.') for r in components) + '</div></details>'
    census = ""
    for name, r in (("Development v2", dev), ("Fresh-control v2", control)):
        census += f'<h3>{name}</h3><pre>{html.escape(json.dumps(r["census"], indent=2))}</pre>'
    body = f"""
<div class="eyebrow">The Socratic Method · Continuity review · 7 September 2026</div>
<h1>Longer paths through the gaps.<br>The same frozen F0 model.</h1>
<p class="lead">Probability-guided curves now recover sheet continuations beyond the native filler's 21-pixel reach. A fixed-policy fresh control confirms the effect. Ambiguous folds remain a review problem, not a solved safety claim.</p>
<div class="cards"><div class="card"><b>357 / 210</b><span class="muted">New paths longer than 21 pixels<br>Development / fresh control</span></div><div class="card"><b>825 / 2,004</b><span class="muted">Central-cube voxels added beyond native repair<br>No existing foreground erased</span></div></div>
<p>F0/200k and T=.35 are unchanged. The earlier raw-model benchmark remains <strong>.6808 versus .5541 matched macro Dice</strong>; no new model or repaired-mask Dice is claimed here. This report tests a postprocessor, not training.</p>
<h2>What changed in real masks?</h2>
<p>Both blocks contain 27 real contiguous 128³ cubes: 384³ context at 8.640 µm. The development-selected policy was frozen before inference on the fresh, disjoint PHerc1447 control. The earlier native-filler control was not reused. This is a new region of the same scroll, not independent-scroll replication.</p>
<div class="table-wrap"><table><thead><tr><th>Stage / block</th><th>New corridors</th><th>Longer than 21 px</th><th>Added voxels</th><th>Core added voxels</th><th>Added 3D components</th></tr></thead><tbody>{table}</tbody></table></div>
<p>All additions in this table are <strong>over the chosen native-filled baseline</strong>, not over raw F0. Corridors are per-plane paths, not unique 3D anatomical joins. The core is the exact central 128³ cube. Native repair contributes {dev['native_added_voxels']:,} development and {control['native_added_voxels']:,} control voxels; total additions over raw F0 become <strong>{dev['total_added_over_raw']:,} and {control['total_added_over_raw']:,}</strong>. Core totals over raw become 867 and 2,128.</p>
<p>The longest accepted v2 paths are {dev['maximum_corridor_length_px']:.1f} and {control['maximum_corridor_length_px']:.1f} pixels. New paint is only {100*dev['added_voxels']/dev['base_foreground']:.3f}% and {100*control['added_voxels']/control['base_foreground']:.3f}% of native-filled foreground. Path length includes the two anchors and is not a physical sheet-coverage score.</p>
<div class="note">This is measurable selective recovery, not merely a higher global threshold mask. It still leaves large unsupported gaps alone. More paint and more connections are not, by themselves, proof of correct anatomy.</div>
<h2>Let the geometry speak</h2>
<p>Shared coordinates and CT window within each block. <span class="magenta">Magenta = existing native-filled foreground.</span> <span class="green">Green = new probability-guided repair.</span> CT is gray. Click for full-size images.</p>
{fig('development_candidate_v2/assets/corridor_01934.png', 'Development central-cube corridor 1934: about 54 pixels along a curved, visible tissue interface. Retained after the v2 geometric refinements.')}
{fig('fresh_control_candidate_v2/assets/corridor_01123.png', 'Fresh-control central-cube corridor 1123: 65.1 pixels, versus the native 21-pixel maximum. The visible tissue boundary continues through the previously open interval.')}
{fig('fresh_control_candidate_v2/assets/component_0137.png', 'Fresh-control component 137: a curved continuation in Z, with Y/X views showing the local patch. Its 1,273 added voxels span eight Z planes; nearby additions are also visible.')}
{fig('fresh_control_candidate_v2/assets/overview_z12608.png', 'Fresh-control overview z12608: targeted green additions with the broad field unchanged. Much missing geometry remains visible.')}
<p><a href="development_candidate_v2/index.html">All 764 development corridor cards</a> · <a href="fresh_control_candidate_v2/index.html">All 515 fresh-control cards</a> · <a href="development_candidate_v1/index.html">Preserved v1 development cards</a>.</p>
<h2>Why v2, and where it still needs judgment</h2>
<p>Skeleton-only paths exposed wandering, backward approaches and hooked attachments on development. V2 refines each proposal through the continuous probability field inside a narrow corridor, then adds outward-anchor and local-turn checks. The previously questionable v1 corridors 1116, 2920 and 3110 are rejected by the corresponding v2 cases. The useful central-cube path remains.</p>
<p>The selected policy uses P≥.20 only to propose seeded corridors between existing components; mean gap P≥.25, local endpoint direction and anisotropy, ridge separation, maximum 96-pixel path, detour and turn checks constrain acceptance. It never replaces the raw .35 mask with a .20 mask. Third-fragment contacts, shared/crossing new paint and same-plane duplicate connections are refused.</p>
<div class="note"><strong>The new stage does not retain the native 3-pixel global radial endpoint gate.</strong> Local folded sheets can change radius along correct continuations. Local evidence replaces that restriction; global wrap identity is not certified. The neighboring-probability guard rejected <strong>zero</strong> real proposals in both blocks and should not be described as independently demonstrated anatomical safety.</div>
<p>All 40 risk-first orthogonal cards were inspected. Development component <strong>301 remains ambiguous</strong> in a packed fold; 290 and 333 are crowded/grazing cases. Fresh-control 13, 81 and 88 attach around fragmented or broad existing foreground. Control 8, 9 and 88 span only one plane. A one-plane bridge is useful connectivity, not a reconstructed 3D sheet.</p>
{fig('development_candidate_v2/assets/component_0301.png', 'Keep the uncertain example visible: development component 301 in a crowded fold. The local checks accept it; these cuts do not settle anatomical sheet identity.')}
{risk}
<p><a href="visual_review.json">Inspection record and explicit limits</a>. Forty risk-first cards plus selected core cases were reviewed, not all 1,279 accepted v2 paths. Cards show all nearby additions, not isolated named components.</p>
<details><summary>Every proposal/refusal counter</summary>{census}<p>Oversized/unanchored skeleton regions are counted before route construction. The accepted/refused route ledgers are linked from each gallery. CT contrast is a diagnostic, not a hard brightness gate.</p></details>
<h2>Verified, usable, and opt-in</h2>
<p>The separate <code>socratic-continuity</code> command requires the original probability cache and checked native-repair receipt. It validates fixed F0/.35, halo32/TTA/BF16, coarse units, dense inventory, original source identities, unchanged binary provenance, input probabilities and estimated memory. Check-only is the default. One CPU process; no inference or native child workers; new separate output under an exclusive lock.</p>
<pre><code>python -m socratic_method.continuity INFERENCE_GRID NATIVE_REPAIRED_GRID NEW_OUTPUT `
  --voxel-size-um 8.64
# Check only. Add --run to write a new repaired grid.</code></pre>
<p><a href="{rel(OFFICIAL / 'docs/continuity.md')}">Usage, scope and recovery notes</a>. The console entry point is declared for repository installation; no environment installation or production deployment was performed here.</p>
<p><strong>{package['tests_passed']} tests pass</strong>, including long/bent positives, wrong-sheet proxies, missing evidence, third-fragment/shared paint, validation/transactions and unchanged native-repair/legacy wrappers. The final checked implementation reproduces <strong>all 27 fresh-control TIFFs byte for byte</strong>. Its first integration stopped on one-voxel anchor brush aprons; the corrected independent audit requires those to be single voxels next to declared anchors. That fix changes verification, not the measured policy or masks.</p>
<p>Independent verification across all 768 v2 planes confirms zero baseline erasure, all 1,279 declared endpoint pairs becoming face-connected, and exactly 1,279 two-contact bridge strokes. Attached single-voxel anchor aprons are counted separately ({dev['independent_audit']['anchor_apron_voxels']} development; {control['independent_audit']['anchor_apron_voxels']} control). No detached or third-component strokes were found. These geometric checks do not settle anatomy.</p>
<h2>Next step</h2>
<p><strong>Keep F0 frozen and use the new stage for opt-in geometry review.</strong> The next mechanism should extend reliable repaired paths into coherent 3D sheet patches and explicitly adjudicate ambiguous paired surfaces. Increasing brush thickness or relaxing every threshold would not resolve the current uncertainty. Additional regions and anatomical labels are needed before making this a default.</p>
<footer>Local evidence only; no training, model blend, public upload, or automatic deployment. The existing native profile and frozen model record remain unchanged. <a href="selected_policy.json">Pre-control selection</a> · <a href="decision.json">Decision</a> · <a href="results.json">Results</a> · <a href="verification.json">Verification</a> · <a href="package_tests.xml">Tests</a> · <a href="{rel(OFFICIAL / 'recipes/f0_continuity_20260907/README.md')}">Official record</a>.<br>Visual QA uses direct PNG inspection and static HTML/asset checks. The in-app Browser was unavailable; no browser-render verification is claimed.</footer>
"""
    return page("F0 continuity — longer curves, same frozen model", body)


def verify_html():
    pages = [ROOT / "index.html", *(ROOT / name / "index.html" for name in ("development_candidate_v1", "development_candidate_v2", "fresh_control_candidate_v2"))]
    images, references = set(), 0
    for path in pages:
        parser = previous.AssetParser()
        parser.feed(path.read_text(encoding="utf-8"))
        for reference in parser.references:
            check(not reference.startswith(("http:", "https:", "data:")), "unexpected remote asset")
            target = (path.parent / reference).resolve()
            check(target.is_file(), f"broken local link: {target}")
            references += 1
            if target.suffix == ".png":
                images.add(target)
    for path in images:
        with Image.open(path) as im:
            im.verify()
    return {"pages": len(pages), "local_references": references, "decoded_pngs": len(images), "browser_render_verified": False, "browser_unavailable": True}


def main():
    print("Verifying frozen model, old records, continuity arrays and package parity.", flush=True)
    frozen = previous.verify_sources()
    # The old evidence snapshots remain immutable even as README/entry points grow.
    old_manifest = read(OFFICIAL / "recipes/f0_repair_20260907/files.json")
    for row in old_manifest["files"]:
        context.verify_file({**row, "path": str(OFFICIAL / "recipes/f0_repair_20260907" / row["relative_path"])})
    old_report = read(context.OUTPUT / "verification.json")["final_report"]
    context.verify_file(old_report)
    results, runs = verify_runs()
    package = verify_package()
    decision = {"schema": "f0-continuity-decision-v1", "status": "implemented and locally verified for opt-in review",
                "profile": installed.PROFILE, "pre_control_selection": identity(ROOT / "selected_policy.json"),
                "fresh_control_based_algorithm_tuning": False, "all_27_package_masks_byte_identical": True,
                "official_native_default_changed": False, "model_or_threshold_changed": False,
                "automatic_deployment": False, "anatomical_precision": None,
                "remaining_limit": "Ambiguous folds, one-plane patches and unsupported broad gaps; coherent 3D expansion and anatomical adjudication remain future work.",
                "visual_review": identity(ROOT / "visual_review.json")}
    verification = {"schema": "f0-continuity-verification-v1", "verified_utc": datetime.now(UTC).isoformat(),
                    "frozen_sources": frozen, "old_native_record_files_verified": len(old_manifest["files"]),
                    "old_native_report_unchanged": old_report, "runs": runs, "package": package,
                    "report_generator": identity(Path(__file__)), "status": "checking_html"}
    for filename, value in (("results.json", results), ("decision.json", decision), ("verification.json", verification)):
        context.atomic_json(ROOT / filename, value)
    for name in results:
        gallery(name)
    (ROOT / "index.html").write_text(render(results, package), encoding="utf-8")
    verification.update(html=verify_html(), final_report=identity(ROOT / "index.html"), status="passed")
    context.atomic_json(ROOT / "verification.json", verification)
    print(json.dumps({"report": str(ROOT / "index.html"), "html": verification["html"],
                      "byte_identical_package_tiffs": 27, "tests": 83, "sealed_pins": frozen["sealed_runtime_files_verified"]}), flush=True)


if __name__ == "__main__":
    main()
