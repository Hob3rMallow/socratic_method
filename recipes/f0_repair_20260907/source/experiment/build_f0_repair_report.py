"""Verify existing F0 repair evidence and render its final local report.

No inference, native repair, training, scoring or deployment is launched here.
Only this new repair report's generated text/JSON artifacts are regenerated.
"""

from __future__ import annotations

import html
import os
import stat
import sys
import tomllib
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import run_f0_repair_context as context
from PIL import Image

ROOT = context.OUTPUT
OFFICIAL = context.ROOT.parent / "socratic_method"
OLD_RUN = context.ROOT / "output/crossres_data/final_c3_250k_20260905"
sys.path.insert(0, str(OFFICIAL / "src"))
from socratic_method import repair


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def link(path: Path) -> str:
    return Path(os.path.relpath(path, ROOT)).as_posix()


def artifact(path: Path) -> dict:
    return context.inventory(path)


def read(path: Path) -> dict:
    return context.read(path)


def verify_sources() -> dict:
    request = read(ROOT / "request.json")
    state = read(ROOT / "inference_state.json")
    check(state["status"] == "complete", "inference is incomplete")
    check(state["request_sha256"] == context.sha256(ROOT / "request.json"), "request changed")
    for item in [request["model"], request["native_filler"], request["runner"], *request["input_files"]]:
        context.verify_file(item)
    for name, expected in request["sealed_runtime_pins"].items():
        check(context.sha256(context.ROOT / name) == expected, f"sealed source changed: {name}")
    model_manifest = read(context.RELEASE / "manifest.json")
    check(context.sha256(Path(model_manifest["model"]["original_artifact"])) == context.MODEL_SHA,
          "original model changed")
    check(context.MODEL.stat().st_file_attributes & stat.FILE_ATTRIBUTE_READONLY,
          "frozen model is no longer read-only")
    frozen_files = read(context.RELEASE / "files.json")["files"]
    for row in frozen_files:
        for directory in (context.RELEASE, OFFICIAL / "recipes/final_c3_250k_20260907"):
            path = directory / row["relative_path"]
            check(path.stat().st_size == row["bytes"] and context.sha256(path) == row["sha256"],
                  f"frozen evidence copy changed: {path}")
    preserved = {
        OLD_RUN / "operator_report/index.html": "5c95c6514a819bae4419457016620627a4990710b80a99b5462f5398b8f32009",
        OLD_RUN / "operator_report/filler/manifest.json": "fedabe4b91efac9291d8ca07d23b3907f17efb5c80d1a97cdf1b02157c4a8e4c",
        OFFICIAL / "docs/line_fitter.md": "bef61cce483f1d42d53a8c52909be8eb4568ff7abff76558163f1a261ef5702e",
        context.ROOT / "crossres_pred/configs/final_c3_250k_20260905.json": "af9d74dbf9f4c0fc978d6b9756fc92f6750fec211705c08be0a05e970c496cbc",
    }
    for path, expected in preserved.items():
        check(context.sha256(path) == expected, f"historical artifact changed: {path}")
    for block, receipt in state["blocks"].items():
        for item in [receipt["provenance"], *receipt["files"]]:
            context.verify_file(item)
        check(context.verify_inference(ROOT / f"inference_{block}", request, block) == receipt,
              f"inference inventory changed: {block}")
    overlaps = []
    overlap_ids = {
        "development": ["z12032_y03968_x03072", "z12160_y03968_x03200"],
        "control": ["z12800_y04096_x03456", "z12928_y04096_x03328"],
    }
    old_cache = OLD_RUN / "evaluation/milestone_00200000/blind_t035/inference/probability"
    for block, ids in overlap_ids.items():
        for cube in ids:
            old, new = old_cache / f"{cube}.tif", ROOT / f"inference_{block}/probability/{cube}.tif"
            check(context.sha256(old) == context.sha256(new), f"overlap probabilities differ: {cube}")
            overlaps.append({"old": str(old), "new": str(new), "sha256": context.sha256(new)})
    return {
        "model": request["model"], "native_binary": request["native_filler"],
        "frozen_model_read_only": True, "sealed_runtime_files_verified": len(request["sealed_runtime_pins"]),
        "frozen_evidence_files_verified_in_each_copy": len(frozen_files),
        "raw_context_cubes_verified": sum(Path(r["path"]).parent.name == "cubes_RAW" for r in request["input_files"]),
        "historical_artifacts_unchanged": [artifact(p) for p in preserved],
        "byte_identical_prior_probability_cubes": overlaps,
    }


def collect_results() -> dict:
    blocks = {}
    selected = read(ROOT / "selected_policy.json")
    check(selected["selected_preset"] == "extended_safe", "selection changed")
    selected_time = (ROOT / "selected_policy.json").stat().st_mtime
    for block in ("development", "control"):
        state = read(ROOT / f"repairs/{block}/state.json")
        context.verify_file(state["runner"])
        expected_names = list(context.PRESETS) if block == "development" else ["report_baseline", "extended_safe"]
        check(set(state["runs"]) == set(expected_names), "unexpected comparison arms")
        results = {}
        for name in expected_names:
            run = state["runs"][name]
            check(run["status"] == "complete" and run["exit_code"] == 0, "unfinished native comparison")
            context.verify_file(run["summary"])
            if block == "control":
                check(selected_time <= datetime.fromisoformat(run["started_utc"]).timestamp(),
                      "selection must predate control evaluation")
            summary = read(ROOT / f"repairs/{block}/{name}_summary.json")
            for item in summary["artifact_files"]:
                context.verify_file(item)
            check(sum(c["erased_voxels"] for c in summary["cubes"].values()) == 0, "raw erasure")
            native = summary["native"]
            check(sum(c["added_voxels"] for c in summary["cubes"].values()) == native["painted_px"], "paint count mismatch")
            results[name] = {
                "source": artifact(ROOT / f"repairs/{block}/{name}_summary.json"),
                "parameters": native["params"], "per_plane_joins": native["kept"],
                "connection_tracks": summary["kept_connection_tracks"], "added_voxels": native["painted_px"],
                "raw_foreground_voxels": native["fg_in"], "erased_raw_voxels": 0,
                "core_added_voxels": summary["core"]["added_voxels"],
                "new_over_old_filled_voxels": summary["new_over_baseline_voxels"],
                "old_repair_voxels_not_retained": summary["baseline_repair_voxels_not_retained"],
                "added_voxel_probability": summary["added_voxel_probability"],
                "census": summary["plane_census"],
            }
        audit = read(ROOT / f"contact_audit/{block}/summary.json")
        check(audit["detached_2d_strokes"] == audit["multiway_2d_strokes"] == 0, "contact audit needs review")
        check(audit["added_2d_strokes"] == results["extended_safe"]["per_plane_joins"], "stroke/ledger discrepancy")
        blocks[block] = {"presets": results, "contact_audit": audit}
    pooled = {}
    for name in ("report_baseline", "extended_safe"):
        pooled[name] = {key: sum(blocks[b]["presets"][name][key] for b in blocks) for key in (
            "per_plane_joins", "connection_tracks", "added_voxels", "core_added_voxels",
            "erased_raw_voxels", "raw_foreground_voxels", "old_repair_voxels_not_retained")}
    return {"schema": "f0-repair-final-results-v1", "blocks": blocks, "pooled": pooled,
            "unit_note": "Per-plane joins and connection tracks are distinct counts, not anatomical correctness labels.",
            "new_official_filled_mask_scores": None}


def verify_wrapper() -> dict:
    check(Path(repair.__file__).resolve() == OFFICIAL / "src/socratic_method/repair.py", "wrong installed module")
    scripts = tomllib.loads((OFFICIAL / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    check(scripts["socratic-repair"] == "socratic_method.repair:main", "missing repair entry point")
    check(scripts["socratic-train"] == "socratic_method.recipe:main", "legacy trainer retargeted")
    check(scripts["socratic-export"] == "socratic_method.hf_export:main", "legacy exporter retargeted")
    destination = ROOT / "installed_wrapper_preflight_only"
    plan = repair.plan_repair(ROOT / "inference_control", destination, context.EXE, voxel_size_um=8.64)
    check(not destination.exists(), "check-only unexpectedly wrote output")
    context.atomic_json(ROOT / "installed_wrapper_plan.json", plan)
    integration = ROOT / "wrapper_control_v2"
    receipt = read(integration / "repair_receipt.json")
    check(receipt["status"] == "complete" and receipt["added_voxels"] == 1174 and receipt["erased_voxels"] == 0,
          "integration receipt mismatch")
    for item in [receipt["plan"]["binary"], *receipt["plan"]["input_files"]]:
        context.verify_file(item)
    old_command = read(integration / "process.json")["command"]
    current_command = repair.native_command(plan, Path(old_command[2]))
    check(current_command == old_command, "installed wrapper changed measured native command")
    matches = []
    for row in receipt["output_files"]:
        path = integration / row["path"]
        digest = context.sha256(path)
        comparison = ROOT / "repairs/control/extended_safe/cubes_PRED" / path.name
        check(digest == row["sha256"] == context.sha256(comparison), "wrapper differs from control")
        matches.append({"cube": path.stem, "sha256": digest})
    check(len(matches) == 27, "incomplete real integration")
    suite = ET.parse(ROOT / "tests.xml").getroot().find("testsuite")
    check(suite is not None and int(suite.attrib["tests"]) == 52, "wrong focused regression receipt")
    check(all(int(suite.attrib[k]) == 0 for k in ("errors", "failures", "skipped")), "tests did not all pass")
    return {"installed_module": artifact(Path(repair.__file__)), "entry_points": scripts,
            "check_only_creates_no_output": True, "native_command_identical_to_real_integration": True,
            "real_integration_receipt": artifact(integration / "repair_receipt.json"),
            "byte_identical_control_outputs": matches, "focused_tests_passed": 52,
            "test_receipt": artifact(ROOT / "tests.xml"),
            "failed_first_integration_preserved": str(ROOT / "wrapper_control.partial-47716"),
            "native_integration_note": "Real integration precedes final transaction/cancellation hardening; native command and all 27 masks match, and installed code passed focused regression tests."}


def figure(path: str, caption: str) -> str:
    return (f"<figure><a href='{html.escape(path, quote=True)}'><img loading='lazy' src='{html.escape(path, quote=True)}' "
            f"alt='{html.escape(caption, quote=True)}'></a><figcaption>{html.escape(caption)}</figcaption></figure>")


def render(results: dict, verification: dict) -> str:
    before, after = (results["pooled"][key] for key in ("report_baseline", "extended_safe"))
    ratio = after["per_plane_joins"] / before["per_plane_joins"]
    percent = 100 * after["added_voxels"] / after["raw_foreground_voxels"]
    rows = []
    preset_rows = []
    for block, data in results["blocks"].items():
        old, new = (data["presets"][key] for key in ("report_baseline", "extended_safe"))
        cells = [block.title()] + [f"{old[k]:,} &rarr; <strong>{new[k]:,}</strong>" for k in (
            "per_plane_joins", "connection_tracks", "added_voxels", "core_added_voxels")]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
        if block == "development":
            for name, row in data["presets"].items():
                p = row["parameters"]
                cells = [name, f"{p['reach_max']:g} / {p['radial_dr']:g} / {p['min_support']}",
                         row["per_plane_joins"], row["connection_tracks"], row["added_voxels"],
                         row["core_added_voxels"], row["old_repair_voxels_not_retained"]]
                preset_rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    cases = "".join(figure(f"review/{block}/extended_safe/assets/track_{track:04d}.png", caption) for block, track, caption in [
        ("development", 39, "Development track 39: a longer continuation retained after the development-only reach review."),
        ("development", 96, "Development track 96: a short interruption in the central assessment cube; the hard support veto previously rejected it."),
        ("control", 33, "Reserved control track 33: a central-cube continuation, with no control-based parameter tuning."),
    ])
    chains = []
    for block, data in results["blocks"].items():
        for row in data["contact_audit"]["worst_case_chain_images"]:
            chains.append(figure(f"contact_audit/{block}/{row['image']}",
                f"{block.title()} added component {row['component']}: {row['voxels']} voxels, "
                f"Z span {row['z_span']}, radial span {row['radial_span_px']:.2f} px. "
                "Geometric extent is not the endpoint radial gate or proof of sheet identity."))
    model_review = link(OFFICIAL / "recipes/final_c3_250k_20260907/review.md")
    command_doc = link(OFFICIAL / "docs/repair.md")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>F0 repair — more continuity, same frozen model</title>
<style>
:root{{color-scheme:dark;--bg:#10151e;--panel:#192230;--ink:#edf2f9;--muted:#b5c3d5;--rule:#354456;--green:#3cff80;--magenta:#da41d5;--cyan:#73d6ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:17px/1.6 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:42px 26px 80px}}h1{{font-size:clamp(30px,4vw,48px);line-height:1.15;max-width:850px;margin:12px 0 20px}}
h2{{font-size:26px;margin:42px 0 14px}}h3{{font-size:20px}}p{{max-width:1000px}}a{{color:var(--cyan)}}a:focus-visible,summary:focus-visible{{outline:3px solid var(--cyan);outline-offset:3px}}
.eyebrow{{color:var(--cyan);letter-spacing:.12em;text-transform:uppercase;font-size:13px}}.lead{{font-size:21px;color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:28px 0}}.card{{padding:22px;background:var(--panel);border:1px solid var(--rule);border-radius:12px}}.card b{{display:block;font-size:32px}}.card span{{color:var(--muted)}}
.note{{border-left:4px solid #efb965;background:#262520;padding:16px 20px}}.table-wrap{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:15px}}th,td{{text-align:left;padding:12px;border-bottom:1px solid var(--rule)}}th{{color:var(--muted)}}
figure{{margin:24px 0;padding:14px;background:var(--panel);border:1px solid var(--rule);border-radius:10px}}img{{display:block;max-width:100%;height:auto;margin:auto}}figcaption{{font-size:14px;color:var(--muted);margin-top:12px}}
.chain-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.chain-grid figure{{margin:0}}details{{margin:20px 0;padding:16px;border:1px solid var(--rule);border-radius:8px}}summary{{cursor:pointer;font-weight:600}}code{{font-size:.9em;color:#b7e8ff}}pre{{overflow:auto;background:var(--panel);padding:20px;border-radius:8px}}.magenta{{color:var(--magenta)}}.green{{color:var(--green)}}small,.muted{{color:var(--muted)}}footer{{margin-top:42px;padding-top:20px;border-top:1px solid var(--rule);font-size:14px;color:var(--muted)}}
@media(max-width:700px){{main{{padding:24px 16px}}.cards,.chain-grid{{grid-template-columns:1fr}}th,td{{padding:9px}}}}
</style></head><body><main>
<div class="eyebrow">The Socratic Method · Repair review · 7 September 2026</div>
<h1>More continuity.<br>The same frozen F0 model.</h1>
<p class="lead">Real neighboring context and a less restrictive persistence policy recover useful local gaps. The reserved control confirms more joins. Broad missing-sheet reconstruction remains the next problem.</p>
<div class="cards"><div class="card"><b>{ratio:.2f}×</b><span>per-plane joins, pooled<br>{before['per_plane_joins']} → {after['per_plane_joins']}; not a precision score</span></div>
<div class="card"><b>0</b><span>original F0 voxels erased<br>All output kept separate</span></div>
<div class="card"><b>F0 / 200k</b><span>T = 0.35 unchanged<br>No new training or model blend</span></div></div>
<p>The model win remains substantial: <strong>0.6808 versus 0.5541 matched macro Dice</strong>, a 12.7-point improvement over raw M7 at the same threshold and without TTA. These are the existing raw-model benchmark results, <em>not</em> scores for the new repaired masks. <a href="{model_review}">Full 25-checkpoint training review</a>.</p>
<h2>The measured comparison</h2>
<p>Two disjoint PHerc1447 <strong>384³</strong> blocks, 27 cubes each, at <strong>8.640 µm</strong>. Four presets on development; old versus development-selected only on the reserved control. The two blocks are from one scroll, not independent-scroll replication. F0 probability inference uses BF16, halo 32 and eight-way TTA; 250 real raw context cubes were pinned. Four overlaps reproduce the previous probability cache byte for byte.</p>
<div class="table-wrap"><table><thead><tr><th>Block</th><th>Per-plane joins</th><th>Connection tracks</th><th>Added voxels</th><th>Central cube additions</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p><strong>Interpretation:</strong> a connection track can contain several per-plane joins. Neither count is a ground-truth success rate. The same raw masks are used on both sides of each comparison.</p>
<div class="note"><strong>Keep the scale honest.</strong> {after['added_voxels']:,} added voxels are only {percent:.3f}% of pooled raw foreground. The central cubes gain 42 and 25 voxels. This improves short gaps; it does not yet supply the broad missing geometry visible in the overview.</div>
<h2>What changed, and what did not</h2>
<p><code>reach-max 12 → 21 · radial-dr 4 → 3 · min-support 3 → 0</code>. Near reach 6, minimum score .30 and paint radius 1 remain fixed. Tracking, tangents, radial and other geometric pair gates stay active. Bridge cutting and slab splitting stay off. The existing native binary and algorithm are unchanged.</p>
<p>Removing the hard support cutoff is the main recovery lever. Tightening the radial limit deliberately gives some of that gain back; source history records a chained fusion motivating this guard. Increasing reach then adds a smaller number of longer development continuations. This is not a uniform relaxation of all safeguards.</p>
<details><summary>Development ablation: every declared preset</summary><div class="table-wrap"><table><thead><tr><th>Preset</th><th>Reach / radial / support</th><th>Joins</th><th>Tracks</th><th>Added voxels</th><th>Core additions</th><th>Old repair voxels not retained</th></tr></thead><tbody>{''.join(preset_rows)}</tbody></table></div></details>
<p><strong>Tradeoff:</strong> the new filled masks omit 319 development and 309 control voxels painted by the old preset. Those are old <em>repair additions</em>, never original F0 foreground. The new preset is additive relative to raw F0, not a superset of the old filled mask. Its internal name <code>extended_safe</code> is an identifier, not anatomical certification.</p>
<h2>Let the geometry speak</h2>
<p>Shared CT window and coordinates. <span class="magenta">Magenta = unchanged F0 foreground.</span> <span class="green">Green = added repair.</span> Click any image to inspect the native-size PNG. Probability is diagnostic model evidence, not ground truth.</p>
{cases}
<p><a href="review/development/extended_safe/index.html">All 195 development track cards and overviews</a> · <a href="review/control/extended_safe/index.html">All 55 reserved-control track cards and overviews</a>. Each card uses the longest join in its track. All tracks are rendered; not every track/plane was manually adjudicated.</p>
<details><summary>Full-field development view: most geometry remains unchanged</summary>{figure('review/development/extended_safe/assets/overview_z12224.png', 'Development z12224. The modest green additions should not be mistaken for broad sheet reconstruction.')}</details>
<h2>Contact and 3D checks</h2>
<p>An independent 8-connected audit across all <strong>768 planes</strong> found <strong>zero detached added strokes</strong> and <strong>zero strokes touching more than two raw 2D components</strong>. Added foreground forms 206 development and 65 control 3D components. Their largest axial spans are 8 and 7 planes; largest radial extents are 7.79 and 8.54 pixels. These extents include paint and axial motion; they are not the endpoint radial-displacement gate.</p>
<p>The 17 risk-first orthogonal cards below cover the union of the five largest radial extents, axial spans and paint volumes in each block. Direct inspection supports local continuations in the displayed cuts; some were already repaired by the old preset. <strong>Development components 108 and 73 are crowded/broad-foreground cases to retain for operator caution.</strong> No global wrap-identity guarantee follows from this audit.</p>
<details><summary>All 17 risk-first Z / Y / X cards</summary><div class="chain-grid">{''.join(chains)}</div></details>
<p><a href="visual_review.json">Review observations and limits</a> · <a href="contact_audit/development/details.json">All development contacts/components</a> · <a href="contact_audit/control/details.json">All control contacts/components</a>.</p>
<h2>A checked, opt-in repair command</h2>
<p>The official repository now has a separate <code>socratic-repair</code> entry point. It checks the pinned model and native binary, units, umbilicus, dense cube inventory, input hashes and estimated memory before work. One native CPU worker; exclusive output lock; no cutting; new separate output only. It verifies zero erasure and commits a receipt only after checking the result.</p>
<pre><code>socratic-repair INPUT_GRID NEW_OUTPUT_GRID `
  --binary PATH_TO_PINNED_PRED_FIXUP.exe `
  --voxel-size-um 8.64
# Prints the checked plan. Add --run to execute.</code></pre>
<p>See the <a href="{command_doc}">PowerShell usage and recovery notes</a>. After installing the repository the console entry point is available; <code>python -m socratic_method.repair</code> is equivalent. No environment installation or automatic production promotion was performed here.</p>
<p><strong>{verification['wrapper']['focused_tests_passed']} focused tests pass</strong>, including the unchanged legacy recipe/export wrappers. The final module's check-only path creates no output. Its native command matches the successful integration exactly; all 27 integration TIFFs are byte-identical to the selected control result. The first integration exposed a native Windows-path JSON escape bug; forward-slash arguments fixed it, and the failed partial remains preserved.</p>
<h2>What should happen next?</h2>
<ol><li><strong>Keep F0/200k at .35 frozen.</strong> Use this measured, checked profile for opt-in local repair review; do not restart training to solve a downstream refusal pattern.</li>
<li><strong>Target genuine unproposed gaps.</strong> Use cached probabilities, CT and orientation/radial constraints to propose continuations between confident sheet segments. Simply lowering score or increasing paint thickness does not address the main remaining bottleneck.</li>
<li><strong>Judge useful continuous paths and wrong-sheet joins together.</strong> Use fresh repair regions and explicit ambiguous cases. More paint or more joins alone is not success. Longer-range probability-guided recovery is the next mechanism to build, not a capability claimed by this profile.</li></ol>
<footer>Local, uncommitted evidence record; no uploads, deployment, model or threshold changes. Model/63 sealed sources, both frozen evidence copies, original report and historical hashed filler document reverified unchanged. <a href="request.json">Request</a> · <a href="selected_policy.json">Pre-control selection</a> · <a href="decision.json">Final decision</a> · <a href="profile.json">Profile</a> · <a href="results.json">Results</a> · <a href="verification.json">Verification receipt</a> · <a href="tests.xml">Tests</a>.<br>Visual QA: direct PNG inspection and static HTML/asset checks. The in-app Browser was unavailable; no browser-render verification is claimed.</footer>
</main></body></html>"""


class AssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references = []
        self.ids = set()

    def handle_starttag(self, tag: str, attrs: list) -> None:
        values = dict(attrs)
        if "id" in values:
            self.ids.add(values["id"])
        for name in ("src", "href"):
            if name in values:
                self.references.append(values[name])


def verify_html() -> dict:
    pages = [ROOT / "index.html", *(ROOT / f"review/{b}/extended_safe/index.html" for b in ("development", "control"))]
    image_paths, reference_count = set(), 0
    for page in pages:
        parser = AssetParser()
        parser.feed(page.read_text(encoding="utf-8"))
        for value in parser.references:
            reference_count += 1
            parts = urlsplit(value)
            check(not parts.scheme and not parts.netloc, "report unexpectedly requires external assets")
            target = (page.parent / unquote(parts.path)).resolve() if parts.path else page
            check(target.is_file(), f"broken local report link: {target}")
            if parts.fragment and target.suffix == ".html":
                other = AssetParser()
                other.feed(target.read_text(encoding="utf-8"))
                check(unquote(parts.fragment) in other.ids, "broken report fragment")
            if target.suffix == ".png":
                image_paths.add(target)
    for path in image_paths:
        with Image.open(path) as picture:
            check(min(picture.size) > 0, "empty image")
            picture.verify()
    return {"html_pages_checked": len(pages), "local_references_checked": reference_count,
            "png_assets_decoded": len(image_paths), "external_dependencies": 0,
            "browser_render_verified": False, "browser_unavailable": True}


def main() -> None:
    print("Verifying frozen model, source pins and cached repair artifacts; no new runs.", flush=True)
    source_verification = verify_sources()
    results = collect_results()
    wrapper_verification = verify_wrapper()
    profile = {"schema": "socratic-repair-profile-v1", "name": repair.PROFILE,
               "checkpoint_sha256": repair.MODEL_SHA256, "threshold": .35,
               "native_binary_sha256": repair.BINARY_SHA256, "parameters": repair.PARAMETERS,
               "measured_pitch_um": 8.64, "guarded_pitch_range_um": [8, 10],
               "model_changed": False, "native_algorithm_changed": False,
               "support_zero_keeps_tracking": True, "cutting": False,
               "scope": "Opt-in contiguous coarse-grid local repair; no global topology or reconstruction certification."}
    decision = {"schema": "f0-repair-final-decision-v1", "selected_preset": "extended_safe",
                "profile": repair.PROFILE, "status": "Implemented and locally verified; reserved control confirms increased local repair activity.",
                "pre_control_selection": artifact(ROOT / "selected_policy.json"),
                "control_used_for_tuning": False, "anatomical_ground_truth_precision": None,
                "automatic_deployment": False, "new_training": False,
                "visual_review": artifact(ROOT / "visual_review.json"),
                "residual_work": "Broad missing-sheet recovery remains a separate probability-guided proposal problem."}
    verification = {"schema": "f0-repair-completion-verification-v1", "verified_utc": datetime.now(UTC).isoformat(),
                    "sources": source_verification, "wrapper": wrapper_verification,
                    "report_generator": artifact(Path(__file__)),
                    "official_text_and_code": [artifact(OFFICIAL / p) for p in (
                        "src/socratic_method/repair.py", "tests/test_repair.py", "pyproject.toml",
                        "docs/repair.md", "README.md", "native/line_fitter/README.md")],
                    "comparison_jobs_complete": True, "no_training_or_native_job_launched_by_report_builder": True}
    for name, value in (("profile.json", profile), ("results.json", results), ("decision.json", decision), ("verification.json", verification)):
        context.atomic_json(ROOT / name, value)
    (ROOT / "index.html").write_text(render(results, verification), encoding="utf-8")
    verification["html"] = verify_html()
    verification["final_report"] = artifact(ROOT / "index.html")
    verification["status"] = "passed"
    context.atomic_json(ROOT / "verification.json", verification)
    print(f"Report ready: {ROOT / 'index.html'}", flush=True)
    print(f"{verification['html']}; 52 tests, 27 byte-identical integration outputs, 63 source pins preserved.", flush=True)


if __name__ == "__main__":
    main()
