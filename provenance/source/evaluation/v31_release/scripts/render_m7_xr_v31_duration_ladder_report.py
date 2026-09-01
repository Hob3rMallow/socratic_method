#!/usr/bin/env python3
"""Render the five-model duration ladder as plain side-by-side HTML."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import render_pherc0139_growth_failure_diagnostics as gate_diagnostics
import render_voxel_grid_report as grid_render
import tifffile
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
EVALUATION = (
    ROOT
    / "output/crossres_data/m7_xr_v31_duration_ladder_joint_evaluation_20260831/"
    "result.json"
)
OUTPUT = (
    ROOT
    / "output/crossres_data/m7_xr_v31_duration_ladder_human_report_20260831"
)
GATE_INPUT = (
    ROOT / "output/crossres_data/pherc0139_growth_gates16_20260831/gate_plan_input.json"
)
GATE_ATLAS = (
    ROOT
    / "output/crossres_data/voxel_atlas_native_v19_medial_micro_pherc0139_4096_20260830/"
    "atlas_catalog.json"
)
GATE_RECORD_ID = "pherc0139-native-fine-teacher-2p399-to-9p362-v11p1"
BLIND_SOURCE = (
    ROOT / "output/crossres_data/pherc1447_blind_audit/six_cube_subset"
)
BLIND_REFERENCE = (
    ROOT
    / "output/crossres_data/pherc1447_blind_audit/"
    "m7_xr_v15_relaxed_trust4x_terminal_report/inference/cubes_PRED"
)
THRESHOLDS = (0.25, *(round(value / 100, 2) for value in range(38, 51)))
DEFAULT_THRESHOLD = 0.42


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected an object")
    return value


def _relative(path: Path) -> str:
    return os.path.relpath(path, OUTPUT).replace(os.sep, "/")


def _threshold_key(value: float) -> str:
    return f"{value:.2f}"


def _plain_mask(mask: np.ndarray) -> Image.Image:
    value = np.where(np.asarray(mask, dtype=bool), 242, 5).astype(np.uint8)
    return Image.fromarray(np.repeat(value[..., None], 3, axis=-1), mode="RGB")


def _plain_ct(raw: np.ndarray) -> Image.Image:
    value = grid_render._normalise_raw(np.asarray(raw))
    return Image.fromarray(np.repeat(value[..., None], 3, axis=-1), mode="RGB")


def _save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _stats(mask: np.ndarray, *, minimum_component: int) -> dict[str, int]:
    value = np.asarray(mask, dtype=bool)
    labels, count = ndimage.label(value, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)[1:]
    return {
        "foreground": int(np.count_nonzero(value)),
        "components": int(np.count_nonzero(sizes >= minimum_component)),
    }


def _dynamic_cell(
    *,
    cell_id: str,
    panels: dict[str, dict[str, str]],
    panel_data: dict[str, dict[str, dict[str, str]]],
) -> str:
    panel_data[cell_id] = panels
    initial = panels[_threshold_key(DEFAULT_THRESHOLD)]
    return f"""
      <td class="visual-cell model-cell">
        <a data-panel-link="{html.escape(cell_id)}" href="{html.escape(initial['src'])}">
          <img loading="lazy" data-panel-image="{html.escape(cell_id)}"
               src="{html.escape(initial['src'])}" alt="plain binary model mask">
        </a>
        <div class="cell-detail" data-panel-detail="{html.escape(cell_id)}">{html.escape(initial['detail'])}</div>
      </td>"""


def _fixed_cell(*, source: Path, detail: str, alt: str) -> str:
    relative = _relative(source)
    return f"""
      <td class="visual-cell reference-cell">
        <a href="{html.escape(relative)}"><img loading="lazy" src="{html.escape(relative)}" alt="{html.escape(alt)}"></a>
        <div class="cell-detail">{html.escape(detail)}</div>
      </td>"""


def _render_locked(
    records: list[dict[str, Any]],
    panel_data: dict[str, dict[str, dict[str, str]]],
) -> tuple[str, list[dict[str, Any]]]:
    gate = _read(GATE_INPUT)
    rows = gate.get("selected_slices")
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError("duration report requires the exact locked 16")
    models = [
        (
            int(record["samples"]),
            gate_diagnostics._load_evaluation(
                f"n={int(record['samples']):,}",
                Path(str(record["locked_evaluation"])),
                GATE_INPUT.resolve(),
            ),
        )
        for record in records
    ]
    source = gate_diagnostics.heldout._source_from_catalog(
        GATE_ATLAS.resolve(), GATE_RECORD_ID
    )
    arrays = gate_diagnostics.heldout._open_arrays(source)
    html_rows: list[str] = []
    manifest: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: int(value["review_rank"])):
        evidence = gate_diagnostics._slice_evidence(row, arrays)
        rank = int(row["review_rank"])
        root = OUTPUT / "assets/locked16" / f"rank_{rank:03d}"
        ct = root / "ct.png"
        teacher = root / "teacher.png"
        m7 = root / "m7.png"
        domain = evidence["domain"]
        teacher_mask = evidence["teacher"] & domain
        m7_mask = evidence["m7"] & domain
        _save(Image.fromarray(evidence["gray"], mode="L").convert("RGB"), ct)
        _save(_plain_mask(teacher_mask), teacher)
        _save(_plain_mask(m7_mask), m7)
        fixed = [
            _fixed_cell(source=ct, detail="plain coarse CT", alt="coarse CT"),
            _fixed_cell(
                source=teacher,
                detail=(
                    f"{_stats(teacher_mask, minimum_component=12)['foreground']} px"
                ),
                alt="teacher mask",
            ),
            _fixed_cell(
                source=m7,
                detail=f"{_stats(m7_mask, minimum_component=12)['foreground']} px",
                alt="published M7 mask",
            ),
        ]
        model_cells: list[str] = []
        model_manifest: list[dict[str, Any]] = []
        for samples, model in models:
            probability = gate_diagnostics._probability_slice(
                model, row, evidence["m7"]
            ).astype(np.float32, copy=False)
            panels: dict[str, dict[str, str]] = {}
            for threshold in THRESHOLDS:
                mask = (probability >= threshold) & domain
                stats = _stats(mask, minimum_component=12)
                path = root / f"n{samples:06d}" / f"t{threshold:.2f}.png"
                _save(_plain_mask(mask), path)
                panels[_threshold_key(threshold)] = {
                    "src": _relative(path),
                    "detail": (
                        f"{stats['foreground']} px · {stats['components']} CC"
                    ),
                }
            cell_id = f"locked-r{rank:03d}-n{samples:06d}"
            model_cells.append(
                _dynamic_cell(
                    cell_id=cell_id, panels=panels, panel_data=panel_data
                )
            )
            model_manifest.append({"samples": samples, "panels": panels})
        label = f"rank {rank} · {row['axis']}={int(row['global_coordinate'])}"
        html_rows.append(
            f"<tr><th class=\"row-label\">{html.escape(label)}</th>"
            f"{''.join(fixed)}{''.join(model_cells)}</tr>"
        )
        manifest.append(
            {
                "rank": rank,
                "candidate_id": row["candidate_id"],
                "label": label,
                "models": model_manifest,
            }
        )
    return "".join(html_rows), manifest


def _blind_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_inference = Path(str(records[0]["blind_inference"]))
    return _read(baseline_inference.parent / "report/report.json")


def _render_blind(
    records: list[dict[str, Any]],
    panel_data: dict[str, dict[str, dict[str, str]]],
) -> tuple[str, list[dict[str, Any]]]:
    report = _blind_report(records)
    fixed = [
        (cube["cube_id"], view)
        for cube in report["cubes"]
        for view in cube["views"]
        if bool(view["fixed_review_slice"])
    ]
    if len(fixed) != 18:
        raise ValueError(f"expected 18 fixed PHerc1447 views, found {len(fixed)}")
    html_rows: list[str] = []
    manifest: list[dict[str, Any]] = []
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]] = {}
    for row_index, (cube_id, view) in enumerate(fixed):
        if cube_id not in cache:
            name = f"{cube_id}.tif"
            probabilities = [
                np.asarray(
                    tifffile.imread(
                        Path(str(record["blind_inference"])) / "probability" / name
                    )
                ).astype(np.float32, copy=False)
                for record in records
            ]
            cache[cube_id] = (
                np.asarray(tifffile.imread(BLIND_SOURCE / "cubes_RAW" / name)),
                np.asarray(tifffile.imread(BLIND_REFERENCE / name)) != 0,
                np.asarray(tifffile.imread(BLIND_SOURCE / "cubes_PRED" / name)) != 0,
                probabilities,
            )
        raw, reference, published, probabilities = cache[cube_id]
        axis = {"z": 0, "y": 1, "x": 2}[view["axis"]]
        index = int(view["local_index"])
        root = OUTPUT / "assets/pherc1447" / f"view_{row_index:02d}_{cube_id}_{view['axis']}"
        ct = root / "ct.png"
        v15 = root / "v15.png"
        m7 = root / "m7.png"
        reference_slice = np.take(reference, index, axis=axis)
        published_slice = np.take(published, index, axis=axis)
        _save(_plain_ct(np.take(raw, index, axis=axis)), ct)
        _save(_plain_mask(reference_slice), v15)
        _save(_plain_mask(published_slice), m7)
        fixed_cells = [
            _fixed_cell(source=ct, detail="plain 9 µm CT", alt="9 micrometre CT"),
            _fixed_cell(
                source=v15,
                detail=f"{int(np.count_nonzero(reference_slice))} px",
                alt="v15 anti-blob reference",
            ),
            _fixed_cell(
                source=m7,
                detail=f"{int(np.count_nonzero(published_slice))} px",
                alt="published M7 mask",
            ),
        ]
        model_cells: list[str] = []
        model_manifest: list[dict[str, Any]] = []
        for record, probability in zip(records, probabilities, strict=True):
            samples = int(record["samples"])
            plane = np.take(probability, index, axis=axis)
            panels: dict[str, dict[str, str]] = {}
            for threshold in THRESHOLDS:
                mask = plane >= threshold
                stats = _stats(mask, minimum_component=1)
                path = root / f"n{samples:06d}" / f"t{threshold:.2f}.png"
                _save(_plain_mask(mask), path)
                panels[_threshold_key(threshold)] = {
                    "src": _relative(path),
                    "detail": (
                        f"{stats['foreground']} px · {stats['components']} 2-D CC"
                    ),
                }
            cell_id = f"blind-v{row_index:02d}-n{samples:06d}"
            model_cells.append(
                _dynamic_cell(
                    cell_id=cell_id, panels=panels, panel_data=panel_data
                )
            )
            model_manifest.append({"samples": samples, "panels": panels})
        label = f"{cube_id} · {view['axis']}={int(view['global_coordinate'])}"
        html_rows.append(
            f"<tr><th class=\"row-label\">{html.escape(label)}</th>"
            f"{''.join(fixed_cells)}{''.join(model_cells)}</tr>"
        )
        manifest.append(
            {
                "cube_id": cube_id,
                "axis": view["axis"],
                "global_coordinate": int(view["global_coordinate"]),
                "label": label,
                "models": model_manifest,
            }
        )
    return "".join(html_rows), manifest


def _headers(records: list[dict[str, Any]], *, locked: bool) -> str:
    references = ("CT", "teacher", "published M7") if locked else (
        "CT",
        "v15 anti-blob",
        "published M7",
    )
    labels = [*references, *(f"{int(row['samples']):,} samples" for row in records)]
    return "".join(f"<th>{html.escape(label)}</th>" for label in labels)


def _metric_tables(records: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for threshold in THRESHOLDS:
        rows: list[str] = []
        missing = False
        for record in records:
            match = next(
                (
                    row
                    for row in record["threshold_summary"]
                    if np.isclose(float(row["threshold"]), threshold)
                ),
                None,
            )
            if match is None:
                missing = True
                continue
            passed = bool(match["pherc1447_anti_blob_passed"])
            rows.append(
                f"""
                <tr><th>{int(record['samples']):,}</th>
                  <td>{int(match['locked_component_matches'])}/16</td>
                  <td>{int(match['locked_anti_blob_passes'])}/16</td>
                  <td>{float(match['locked_mean_dice']):.3f}</td>
                  <td>{float(match['pherc1447_foreground_ratio_vs_v15']):.3f}×</td>
                  <td>{float(match['pherc1447_reference_recall']):.3f}</td>
                  <td>{int(match['pherc1447_interior_regression_cubes'])}/6</td>
                  <td>{int(match['pherc1447_thickness_regression_cubes'])}/6</td>
                  <td class="{'pass' if passed else 'fail'}">{'PASS' if passed else 'FAIL'}</td>
                </tr>"""
            )
        key = _threshold_key(threshold)
        if missing:
            body = (
                "<p class=\"control-note\">T=0.25 is retained as a visual "
                "control. The common machine-comparison band begins at T=0.38.</p>"
            )
        else:
            body = f"""
              <div class="table-scroll"><table class="metrics-table">
                <thead><tr><th>training samples</th><th>locked CC</th>
                  <th>locked anti-blob</th><th>locked Dice</th>
                  <th>PHerc1447 foreground / v15</th><th>v15 recall</th>
                  <th>interior regressions</th><th>thickness regressions</th>
                  <th>3-D anti-blob</th></tr></thead>
                <tbody>{''.join(rows)}</tbody>
              </table></div>"""
        blocks.append(
            f"<div class=\"metric-block\" data-threshold-block=\"{key}\" "
            f"{'hidden' if not np.isclose(threshold, DEFAULT_THRESHOLD) else ''}>"
            f"{body}</div>"
        )
    return "".join(blocks)


def _write_report(
    *,
    result: dict[str, Any],
    locked_rows: str,
    blind_rows: str,
    panel_data: dict[str, dict[str, dict[str, str]]],
) -> Path:
    records = result["records"]
    options = "".join(
        f"<option value=\"{value:.2f}\" "
        f"{'selected' if np.isclose(value, DEFAULT_THRESHOLD) else ''}>"
        f"T={value:.2f}</option>"
        for value in THRESHOLDS
    )
    payload = json.dumps(panel_data, separators=(",", ":")).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>M7-XR duration ladder review</title>
<style>
:root {{ color-scheme:dark; --bg:#071019; --panel:#0d1a25; --line:#294153; --text:#edf4fa; --muted:#9fb2c1; --good:#69dda5; --bad:#ff778b; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.45 system-ui,sans-serif; }}
main {{ width:100%; padding:18px; }}
h1 {{ margin:0 0 6px; font-size:30px; }} h2 {{ margin:30px 0 8px; }}
p {{ max-width:1200px; color:var(--muted); }}
.callout {{ max-width:none; padding:12px 16px; border:1px solid #31516a; border-radius:8px; background:#102232; color:var(--text); }}
.controls {{ position:sticky; top:0; z-index:20; display:flex; align-items:center; gap:12px; padding:10px 12px; margin:14px 0; background:#102232; border:1px solid #31516a; border-radius:8px; }}
select {{ font:inherit; padding:6px 10px; color:var(--text); background:#071019; border:1px solid #52748d; border-radius:5px; }}
.table-scroll {{ width:100%; overflow:auto; border:1px solid var(--line); border-radius:8px; }}
table {{ border-collapse:separate; border-spacing:0; width:max-content; min-width:100%; background:var(--panel); }}
th,td {{ padding:7px; border-right:1px solid var(--line); border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
thead th {{ position:static; background:#142737; white-space:nowrap; }}
.visual-table {{ min-width:1580px; table-layout:fixed; }}
.visual-table .row-label {{ position:sticky; left:0; z-index:4; width:180px; background:#10202d; overflow-wrap:anywhere; }}
.visual-table thead .row-label {{ position:static; z-index:auto; }}
.visual-cell {{ width:175px; min-width:175px; padding:5px; }}
.visual-cell img {{ display:block; width:100%; aspect-ratio:1; object-fit:contain; image-rendering:pixelated; background:#05090d; }}
.cell-detail {{ color:var(--muted); font-size:11px; padding:5px 1px 1px; }}
.reference-cell {{ background:#0a151e; }}
.metrics-table th,.metrics-table td {{ white-space:nowrap; }}
.pass {{ color:var(--good); font-weight:700; }} .fail {{ color:var(--bad); font-weight:700; }}
.control-note {{ padding:10px 12px; border:1px solid var(--line); border-radius:6px; }}
.watch-list {{ columns:2; max-width:1250px; color:var(--muted); }}
code {{ color:#c9e6ff; }} a {{ color:#7fc9ff; }}
</style></head><body><main>
<h1>M7-XR raw duration ladder</h1>
<p class="callout"><strong>Human review set.</strong> Every model is the raw dynamic-medial student: no M7 blend, no teacher at inference, and no postprocessing. The only training variable is cumulative samples. All panels are separate plain masks at identical crop and scale.</p>
<div class="controls"><label for="threshold"><strong>Display threshold</strong></label><select id="threshold">{options}</select><span id="threshold-label">Showing T={DEFAULT_THRESHOLD:.2f}</span></div>
<h2>Machine measurements at the displayed threshold</h2>
<p>The legacy locked metrics remain diagnostics rather than human truth. The PHerc1447 three-dimensional anti-blob rule remains a hard constraint.</p>
{_metric_tables(records)}
<h2>Review watch list</h2>
<ul class="watch-list"><li>rank 26: retain the top strip without producing two top blobs or touching strokes</li><li>rank 64: thicken the bottom-left strip and watch inherited drift</li><li>z=12032/12043/12045/12050: hard de-blobbed structure</li><li>z=12153 and z=12161: avoid broken downward/middle lines</li><li>z=12193 and z=12256: undercooked but human-visible structure</li><li>z=12171 and z=12224: preserve the strong T=0.40 solutions</li></ul>
<h2>Locked PHerc0139 — all 16 slices</h2>
<p>CT, teacher, and published M7 stay fixed. The five duration models update together when the threshold changes.</p>
<div class="table-scroll"><table class="visual-table"><thead><tr><th class="row-label">slice</th>{_headers(records, locked=True)}</tr></thead><tbody>{locked_rows}</tbody></table></div>
<h2>PHerc1447 — all six cubes, standard fixed 18 views</h2>
<p>v15 is the prior anti-blob reference, not an answer key. Published M7 is shown separately. Inspect whether longer training recovers legible lines without refilling blobs.</p>
<div class="table-scroll"><table class="visual-table"><thead><tr><th class="row-label">view</th>{_headers(records, locked=False)}</tr></thead><tbody>{blind_rows}</tbody></table></div>
<h2>Model provenance</h2><ul>{''.join(f"<li>{int(row['samples']):,} samples · <code>{html.escape(str(row['checkpoint_sha256']))}</code></li>" for row in records)}</ul>
<script>
const panelData={payload};
const selector=document.getElementById('threshold');
function applyThreshold() {{
  const value=selector.value;
  document.getElementById('threshold-label').textContent=`Showing T=${{value}}`;
  for (const [id, values] of Object.entries(panelData)) {{
    const panel=values[value]; if (!panel) continue;
    const image=document.querySelector(`[data-panel-image="${{id}}"]`);
    const link=document.querySelector(`[data-panel-link="${{id}}"]`);
    const detail=document.querySelector(`[data-panel-detail="${{id}}"]`);
    image.src=panel.src; link.href=panel.src; detail.textContent=panel.detail;
  }}
  for (const block of document.querySelectorAll('[data-threshold-block]')) {{
    block.hidden=block.dataset.thresholdBlock!==value;
  }}
}}
selector.addEventListener('change',applyThreshold); applyThreshold();
</script></main></body></html>"""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def main() -> int:
    result = _read(EVALUATION)
    if (
        result.get("state") != "complete"
        or result.get("model_composition")
        != "raw-student-probability-only; no M7 blend"
    ):
        raise ValueError("duration evaluation is incomplete or blended")
    records = sorted(result["records"], key=lambda row: int(row["samples"]))
    if tuple(int(row["samples"]) for row in records) != (1024, 2048, 3072, 4096, 8192):
        raise ValueError("duration report requires all five milestones")
    panel_data: dict[str, dict[str, dict[str, str]]] = {}
    locked_html, locked_manifest = _render_locked(records, panel_data)
    blind_html, blind_manifest = _render_blind(records, panel_data)
    index = _write_report(
        result=result,
        locked_rows=locked_html,
        blind_rows=blind_html,
        panel_data=panel_data,
    )
    manifest = {
        "schema": "crossres-m7-xr-v31-duration-ladder-human-report-v1",
        "research_only": True,
        "model_composition": result["model_composition"],
        "samples": [int(row["samples"]) for row in records],
        "thresholds": list(THRESHOLDS),
        "default_threshold": DEFAULT_THRESHOLD,
        "locked16_rows": locked_manifest,
        "pherc1447_fixed18_rows": blind_manifest,
        "dynamic_panel_count": len(panel_data),
        "index": str(index.resolve()),
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(index.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
