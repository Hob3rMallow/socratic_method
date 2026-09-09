"""Render the F0-family all-metrics ledger (flat M7 + every variant) as a self-contained HTML page."""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(r"D:\work\vesuvius-c")
VARIANTS = ROOT / "output/crossres_data/f0_family_20260908/variants"
TABLE = json.loads((VARIANTS / "all_metrics/all_metrics_table.json").read_text(encoding="utf-8"))
CONFIG = json.loads((ROOT / "crossres_pred/configs/f0_variants_20260908.json").read_text(encoding="utf-8"))
OUT = VARIANTS / "all_metrics" / "all_metrics_ledger.html"
VIEW = "z12160_y04224_x02944_z12224.png"

SETS = [
    ("v14p2_val", "v14p2 validation", "689 cases, PHerc0814 + PHerc1451, registered fine-teacher ground truth"),
    ("p0500p2_val", "PHerc0500P2", "320 cases, human-labelled ground truth, absolute scores only"),
]
COLS = [  # (key in sets mean_*, label, decimals, lower_is_better)
    ("leaderboard", "Composite", 4, False),
    ("surface_dice", "Surface Dice", 4, False),
    ("voi_score", "VOI score", 4, False),
    ("voi_split", "VOI split", 3, True),
    ("voi_merge", "VOI merge", 3, True),
    ("toposcore", "Topo", 4, False),
    ("topoF1_0", "F1 β0", 3, False),
    ("topoF1_1", "F1 β1", 3, False),
    ("topoF1_2", "F1 β2", 3, False),
]
DELTAS = [  # (paired column, label, lower_is_better)
    ("leaderboard", "Δ composite", False),
    ("surface_dice", "Δ Surface Dice", False),
    ("toposcore", "Δ Topo", False),
    ("voi_merge", "Δ VOI merge", True),
    ("voi_split", "Δ VOI split", True),
]

variant_cfg = {v["id"]: v for v in CONFIG["variants"]}


def k_of(samples: int) -> str:
    return f"{samples // 1000}k"


def describe(row_id: str) -> tuple[str, str]:
    """Return (display name, one-line description)."""
    base = row_id.removesuffix("_tta")
    tta = row_id.endswith("_tta")
    if base == "m7_raw" or base == "m7":
        name, desc = "Flat M7", "Released M7 nnU-Net, raw prediction thresholded at T"
    else:
        cfg = variant_cfg.get(base) or variant_cfg.get(row_id)
        ck = cfg["checkpoint"]
        if ck["kind"] == "milestone":
            name = f"F0 {k_of(ck['samples'])}"
            desc = f"F0 checkpoint at {ck['samples']:,} samples"
            if ck["samples"] == 200000:
                desc += " (the frozen model)"
        elif ck["kind"] == "average":
            s = ck["samples"]
            name = f"avg {k_of(min(s))}–{k_of(max(s))}" if base not in ("avg_top3_150k", "avg_last3") else {"avg_top3_150k": "avg top-3", "avg_last3": "avg last-3"}[base]
            desc = f"uniform fp32 weight average of {len(s)} milestones ({', '.join(k_of(x) for x in s)})"
        else:
            s = ck["samples"]
            name = "ens {" + ", ".join(k_of(x) for x in s) + "}"
            desc = f"mean-probability ensemble of {len(s)} milestones, diagnostic only, never ships"
    if tta:
        name += " + TTA"
        desc += "; 8-way mirror test-time augmentation"
    return name, desc


def arrow(lower: bool | None) -> str:
    """Direction mark: which way is better for this metric."""
    if lower is None:
        return ""
    return " <span class='dir' title='lower is better'>&darr;</span>" if lower else " <span class='dir' title='higher is better'>&uarr;</span>"


def fmt(v, d=4):
    return "" if v is None else f"{v:.{d}f}"


def sfmt(v, d=4):
    return "" if v is None else f"{v:+.{d}f}"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def group_of(row_id: str) -> str:
    if row_id.endswith("_tta"):
        return "tta"
    if row_id.startswith("m7"):
        return "m7"
    if row_id.startswith("f0_"):
        return "single"
    if row_id.startswith("avg_"):
        return "avg"
    return "ens"


GROUP_TITLES = {
    "m7": "M7 baselines",
    "single": "F0 single milestones",
    "avg": "F0 weight averages",
    "ens": "Probability ensembles (diagnostic only)",
    "tta": "With 8-way mirror TTA",
}
TTA_ORDER = ["m7_tta", "f0_200k_tta", "f0_110k_tta", "f0_250k_tta", "avg_100k_250k_tta", "avg_top3_150k_tta"]

rows = TABLE["rows"]
reference_id = TABLE["reference"]


def ordered_rows():
    groups: dict[str, list] = {g: [] for g in GROUP_TITLES}
    for r in rows:
        groups[group_of(r["id"])].append(r)
    groups["tta"].sort(key=lambda r: (TTA_ORDER.index(r["id"]) if r["id"] in TTA_ORDER else 99, r["threshold"]))
    return groups


def delta_cell(pair: dict | None, col: str, lower_better: bool) -> str:
    if not pair or col not in pair:
        return "<td class='num delta'></td>"
    d = pair[col]
    lo, hi = d["ci95"]
    good = (hi < 0) if lower_better else (lo > 0)
    bad = (lo > 0) if lower_better else (hi < 0)
    cls = "good" if good else "bad" if bad else "flat"
    return (f"<td class='num delta {cls}' data-v='{d['delta']:.6f}'><span class='pt'>{sfmt(d['delta'])}</span>"
            f"<span class='ci'>[{sfmt(lo)}, {sfmt(hi)}]</span></td>")


def set_table(set_key: str) -> str:
    groups = ordered_rows()
    head = ["<th class='model'>model</th><th>TTA</th><th class='num sortable' data-col='t'>T</th>",
            f"<th class='num sortable' data-col='macro'>macro Dice{arrow(False)}</th>"]
    for key, label, _, lower in COLS:
        head.append(f"<th class='num sortable' data-col='{key}'>{label}{arrow(lower)}</th>")
    for key, label, lower in DELTAS:
        head.append(f"<th class='num sortable delta-h' data-col='d_{key}'>{label}{arrow(lower)}</th>")
    body = []
    for g, title in GROUP_TITLES.items():
        members = groups[g]
        if not members:
            continue
        body.append(f"<tr class='group'><td colspan='{4 + len(COLS) + len(DELTAS)}'>{title}</td></tr>")
        for r in members:
            s = r["sets"].get(set_key)
            if not s:
                continue
            name, desc = describe(r["id"])
            role = r.get("role", "")
            cls = ["row"]
            if r["id"] == reference_id:
                cls.append("reference")
            if role == "primary" and r["id"].endswith("_tta"):
                cls.append("primary")
            if r.get("diagnostic_only"):
                cls.append("diag")
            if r["id"].startswith("m7"):
                cls.append("m7")
            tag = ""
            if r["id"] == reference_id:
                tag = "<span class='tag'>reference</span>"
            elif r["id"] == "f0_200k_tta":
                tag = "<span class='tag accent'>primary</span>"
            elif r.get("diagnostic_only"):
                tag = "<span class='tag'>diagnostic</span>"
            cells = [f"<td class='model'><span class='name'>{esc(name)}</span>{tag}<span class='desc'>{esc(desc)}</span></td>",
                     f"<td>{'yes' if r['tta'] else 'no'}</td>",
                     f"<td class='num' data-v='{r['threshold']:.2f}'>{r['threshold']:.2f}</td>",
                     f"<td class='num' data-v='{r['macro_scroll_dice']:.6f}'>{fmt(r['macro_scroll_dice'])}</td>"]
            for key, _, dec, _ in COLS:
                v = s.get(f"mean_{key}")
                cells.append(f"<td class='num' data-v='{'' if v is None else f'{v:.6f}'}'>{fmt(v, dec)}</td>")
            pair = (r.get("paired_vs_reference") or {}).get(set_key)
            if r["id"] == reference_id:
                cells += ["<td class='num delta ref'>ref</td>"] * len(DELTAS)
            else:
                for key, _, lower in DELTAS:
                    cells.append(delta_cell(pair, key, lower))
            body.append(f"<tr class='{' '.join(cls)}' data-t='{r['threshold']:.2f}'>{''.join(cells)}</tr>")
    sealed = [e for e in TABLE["sealed_baselines_at_020"] if e["sets"].get(set_key)]
    if sealed:
        body.append(f"<tr class='group'><td colspan='{4 + len(COLS) + len(DELTAS)}'>Sealed M7 baselines at their calibrated T = 0.20 (release record; deltas are operating-point-paired against the reference at 0.30)</td></tr>")
        for e in sealed:
            s = e["sets"][set_key]
            name = "Flat M7" + (" + TTA" if e["id"].endswith("tta") else "")
            cells = [f"<td class='model'><span class='name'>{name}</span><span class='tag'>sealed</span><span class='desc'>{esc(e['note'])}; T calibrated in-sample on v14p2; deltas vs F0 200k at T 0.30</span></td>",
                     f"<td>{'yes' if e['id'].endswith('tta') else 'no'}</td>",
                     "<td class='num' data-v='0.20'>0.20</td>",
                     f"<td class='num' data-v='{e['macro_scroll_dice']:.6f}'>{fmt(e['macro_scroll_dice'])}</td>"]
            for key, _, dec, _ in COLS:
                v = s.get(f"mean_{key}")
                cells.append(f"<td class='num' data-v='{'' if v is None else f'{v:.6f}'}'>{fmt(v, dec)}</td>")
            pair = (e.get("paired_vs_reference") or {}).get(set_key)
            for key, _, lower in DELTAS:
                cells.append(delta_cell(pair, key, lower))
            body.append(f"<tr class='row m7 sealed' data-t='0.20'>{''.join(cells)}</tr>")
    return f"<table class='ledger'><thead><tr>{''.join(head)}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def six_cube_table() -> str:
    seen = set()
    out = []
    for r in rows:
        six = r.get("six_cube")
        if not six:
            continue
        base = r["id"].removesuffix("_tta")
        base = "m7_raw" if base == "m7" else base
        key = (base, r["threshold"])
        if key in seen:
            continue
        seen.add(key)
        name, _ = describe(base)
        if base == "m7_raw":
            name = "M7 (flat and + TTA share this export)"
        gates = "pass" if six["gates_pass"] else "FAIL (" + ", ".join(n for n, ok in (("interior", six["interior_pass"]), ("thickness", six["thickness_pass"])) if not ok) + ")"
        frontier = {True: "pass", False: "veto", None: "unresolved"}[six["frontier_pass"]]
        cls = "m7" if base == "m7_raw" else ""
        out.append(
            f"<tr class='row {cls}' data-t='{r['threshold']:.2f}'><td class='model'><span class='name'>{esc(name)}</span></td>"
            f"<td class='num'>{r['threshold']:.2f}</td><td class='num'>{six['bridge_pixels']:,}</td><td class='num'>{six['interior_pixels_r2']:,}</td>"
            f"<td class='num'>{fmt(six['skeleton_recall'], 3)}</td><td class='num'>{fmt(six['skeleton_precision'], 3)}</td><td class='num'>{six['components']}</td>"
            f"<td class='num'>{fmt(six['foreground_ratio_vs_reference'], 3)}</td><td class='num'>{fmt(six['recall_vs_reference'], 3)}</td>"
            f"<td class='{'ok' if six['gates_pass'] else 'fail'}'>{gates}</td><td class='{ {'pass': 'ok', 'veto': 'fail', 'unresolved': 'na'}[frontier] }'>{frontier}</td></tr>")
    head = (f"<th class='model'>checkpoint</th><th class='num'>T</th><th class='num'>bridge px{arrow(True)}</th><th class='num'>interior px{arrow(True)}</th><th class='num'>skeleton recall{arrow(False)}</th>"
            f"<th class='num'>skeleton precision{arrow(False)}</th><th class='num'>components <span class='dir' title='no single direction: read with bridges (mergers) and skeleton recall (fragments)'>&middot;</span></th>"
            f"<th class='num'>fg ratio <span class='dir' title='closest to one is better'>&rarr;1</span></th><th class='num'>recall vs ref{arrow(False)}</th><th>gates <span class='dir'>pass</span></th><th>frontier <span class='dir'>pass</span></th>")
    return f"<table class='ledger six'><thead><tr>{head}</tr></thead><tbody>{''.join(out)}</tbody></table>"


def panels() -> str:
    geometry = VARIANTS / "geometry"
    by_variant = {}
    for d in geometry.iterdir():
        rec = d / "receipt_t030.json"
        if rec.exists():
            by_variant[str(json.loads(rec.read_text(encoding="utf-8"))["variant_id"]).removesuffix("_tta")] = d
    figs = []
    for vid, caption in (("f0_200k", "F0 200k, the weights behind the reference and the primary"), ("f0_110k", "F0 110k, the best composite on both Kaggle sets with TTA")):
        d = by_variant.get(vid)
        img = d / "blind_t035" / "report" / "images" / VIEW if d else None
        if img and img.exists():
            data = base64.b64encode(img.read_bytes()).decode("ascii")
            figs.append(f"<figure><img alt='six-cube review panel for {esc(vid)}' src='data:image/png;base64,{data}'><figcaption>{esc(caption)}</figcaption></figure>")
    return "".join(figs)


def row_at(row_id: str, t: float) -> dict:
    return next(r for r in rows if r["id"] == row_id and abs(r["threshold"] - t) < 1e-6)


def tile(row_id: str, label: str) -> str:
    r = row_at(row_id, 0.30)
    v = r["sets"]["v14p2_val"]
    p = r["sets"]["p0500p2_val"]
    name, _ = describe(row_id)
    return (f"<div class='tile'><div class='eyebrow'>{esc(label)}</div><div class='tname'>{esc(name)}</div>"
            f"<dl><div><dt>macro Dice{arrow(False)}</dt><dd>{fmt(r['macro_scroll_dice'])}</dd></div>"
            f"<div><dt>composite{arrow(False)}</dt><dd>{fmt(v['mean_leaderboard'])}</dd></div>"
            f"<div><dt>Topo{arrow(False)}</dt><dd>{fmt(v['mean_toposcore'])}</dd></div>"
            f"<div><dt>VOI merge{arrow(True)}</dt><dd>{fmt(v['mean_voi_merge'], 3)}</dd></div>"
            f"<div><dt>0500P2 composite{arrow(False)}</dt><dd>{fmt(p['mean_leaderboard'])}</dd></div></dl></div>")


CSS = """
<title>F0 Family Metrics Ledger</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --bg:#f2f3ee; --surface:#fbfbf8; --surface-2:#e9ebe4; --ink:#1a1d21; --muted:#5c6470; --rule:#d3d7cd; --rule-strong:#aeb4a8;
  --accent:#245e63; --accent-ink:#ffffff; --accent-soft:#d9e8e8; --m7:#efe9dc; --m7-strong:#c9b993;
  --good:#1d7a5b; --good-soft:#dcefe5; --bad:#a8382c; --bad-soft:#f3dedb; --flat:#7a828c; --flat-soft:#e6e8e3;
  --primary:#fff4d6; --primary-edge:#c99a2e;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#14171a; --surface:#1c2025; --surface-2:#242930; --ink:#e6e8e3; --muted:#9aa3ad; --rule:#2d333a; --rule-strong:#4a525b;
    --accent:#6fc1c4; --accent-ink:#0e1d1e; --accent-soft:#1f3436; --m7:#2a2723; --m7-strong:#8a7a55;
    --good:#4fbf95; --good-soft:#173328; --bad:#e0705f; --bad-soft:#3a1f1b; --flat:#8d949c; --flat-soft:#262b31;
    --primary:#33301f; --primary-edge:#d9b04a;
  }
}
:root[data-theme="dark"]{
  --bg:#14171a; --surface:#1c2025; --surface-2:#242930; --ink:#e6e8e3; --muted:#9aa3ad; --rule:#2d333a; --rule-strong:#4a525b;
  --accent:#6fc1c4; --accent-ink:#0e1d1e; --accent-soft:#1f3436; --m7:#2a2723; --m7-strong:#8a7a55;
  --good:#4fbf95; --good-soft:#173328; --bad:#e0705f; --bad-soft:#3a1f1b; --flat:#8d949c; --flat-soft:#262b31;
  --primary:#33301f; --primary-edge:#d9b04a;
}
body{background:var(--bg);color:var(--ink);font-family:"Source Sans 3","Segoe UI",system-ui,sans-serif;font-size:15px;line-height:1.5;margin:0}
.page{max-width:1480px;margin:0 auto;padding:32px 28px 64px}
header.top{display:grid;grid-template-columns:1fr auto;gap:24px;align-items:end;border-bottom:2px solid var(--ink);padding-bottom:18px;margin-bottom:22px}
.eyebrow{font-family:"Barlow Condensed","Arial Narrow",sans-serif;font-weight:600;text-transform:uppercase;letter-spacing:.12em;font-size:13px;color:var(--muted)}
h1{font-family:"Barlow Condensed","Arial Narrow",sans-serif;font-weight:700;font-size:44px;line-height:1;margin:6px 0 10px;letter-spacing:.005em;text-wrap:balance}
h2{font-family:"Barlow Condensed","Arial Narrow",sans-serif;font-weight:600;font-size:26px;margin:0 0 4px;text-wrap:balance}
h3{font-family:"Barlow Condensed","Arial Narrow",sans-serif;font-weight:600;font-size:19px;margin:0 0 6px;text-transform:uppercase;letter-spacing:.06em}
.lede{max-width:70ch;margin:0;color:var(--ink)}
.meta{font-family:"IBM Plex Mono",Consolas,monospace;font-size:12px;color:var(--muted);text-align:right;line-height:1.7}
.tiles{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px;margin:0 0 28px}
.tile{background:var(--surface);border:1px solid var(--rule);border-top:3px solid var(--rule-strong);padding:14px 16px 12px}
.tile.m7{border-top-color:var(--m7-strong);background:var(--m7)}
.tile.primary{border-top-color:var(--primary-edge);background:var(--primary)}
.tile.accent{border-top-color:var(--accent)}
.tname{font-family:"Barlow Condensed","Arial Narrow",sans-serif;font-size:24px;font-weight:600;margin:2px 0 8px}
.tile dl{margin:0;display:grid;grid-template-columns:1fr 1fr;gap:4px 14px}
.tile dl div{display:flex;justify-content:space-between;gap:8px;border-bottom:1px dotted var(--rule);padding:2px 0}
.tile dl div:last-child{grid-column:1 / -1}
.tile dt{color:var(--muted);font-size:13px}
.tile dd{margin:0;font-family:"IBM Plex Mono",Consolas,monospace;font-variant-numeric:tabular-nums;font-size:14px;font-weight:500}
section{margin:0 0 36px}
.sechead{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:8px 24px;margin-bottom:10px}
.sechead p{margin:0;color:var(--muted);font-size:14px}
.controls{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;margin:0 0 14px;font-size:14px}
.seg{display:inline-flex;border:1px solid var(--rule-strong);border-radius:3px;overflow:hidden}
.seg button{appearance:none;border:0;background:var(--surface);color:var(--ink);font:inherit;font-family:"IBM Plex Mono",Consolas,monospace;font-size:13px;padding:5px 12px;cursor:pointer;border-right:1px solid var(--rule)}
.seg button:last-child{border-right:0}
.seg button[aria-pressed="true"]{background:var(--accent);color:var(--accent-ink)}
.seg button:focus-visible,button.plain:focus-visible,label input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button.plain{appearance:none;border:1px solid var(--rule-strong);border-radius:3px;background:var(--surface);color:var(--ink);font:inherit;font-size:13px;padding:5px 10px;cursor:pointer}
label.chk{display:inline-flex;align-items:center;gap:6px;cursor:pointer}
.scroll{overflow-x:auto;border:1px solid var(--rule);background:var(--surface)}
table.ledger{border-collapse:separate;border-spacing:0;width:100%;min-width:1180px;font-size:13.5px}
table.ledger th{position:sticky;top:0;background:var(--surface-2);color:var(--ink);font-family:"Barlow Condensed","Arial Narrow",sans-serif;font-weight:600;font-size:14px;letter-spacing:.04em;text-transform:uppercase;text-align:left;padding:8px 10px;border-bottom:1px solid var(--rule-strong);white-space:nowrap;z-index:1}
table.ledger th.num{text-align:right}
table.ledger th.sortable{cursor:pointer}
table.ledger th.sortable:hover{color:var(--accent)}
table.ledger th[aria-sort="descending"]::after{content:" ▾ sorted"}
table.ledger th[aria-sort="ascending"]::after{content:" ▴ sorted"}
.dir{font-family:"IBM Plex Mono",Consolas,monospace;font-size:11px;color:var(--accent);font-weight:500;letter-spacing:0;text-transform:none}
.legend{display:inline-flex;gap:14px;align-items:center;color:var(--muted);font-size:13px;border-left:1px solid var(--rule-strong);padding-left:14px}
table.ledger th.delta-h{border-left:1px solid var(--rule)}
table.ledger td{padding:6px 10px;border-bottom:1px solid var(--rule);vertical-align:top;white-space:nowrap}
table.ledger td.num{text-align:right;font-family:"IBM Plex Mono",Consolas,monospace;font-variant-numeric:tabular-nums;font-size:13px}
table.ledger td.delta{border-left:1px solid var(--rule)}
table.ledger td.delta .pt{display:block;font-weight:500}
table.ledger td.delta .ci{display:block;font-size:11px;color:var(--muted)}
table.ledger td.delta.good .pt{color:var(--good)}
table.ledger td.delta.bad .pt{color:var(--bad)}
table.ledger td.delta.flat .pt{color:var(--flat)}
table.ledger td.delta.good{background:var(--good-soft)}
table.ledger td.delta.bad{background:var(--bad-soft)}
table.ledger td.delta.ref{color:var(--muted);font-style:italic}
table.ledger tr.group td{background:var(--surface-2);font-family:"Barlow Condensed","Arial Narrow",sans-serif;font-weight:600;letter-spacing:.06em;text-transform:uppercase;font-size:13px;color:var(--muted);padding:6px 10px}
table.ledger tr.m7 td.model,table.ledger tr.m7 td:not(.delta){background:var(--m7)}
table.ledger tr.primary td:not(.delta){background:var(--primary)}
table.ledger tr.primary td.model{box-shadow:inset 3px 0 0 var(--primary-edge)}
table.ledger tr.reference td.model{box-shadow:inset 3px 0 0 var(--accent)}
table.ledger tr.diag td{color:var(--muted)}
table.ledger td.model{white-space:normal;min-width:230px}
table.ledger td.model .name{font-weight:600;display:inline-block;margin-right:6px}
table.ledger td.model .desc{display:block;font-size:12px;color:var(--muted);line-height:1.35}
.tag{display:inline-block;font-family:"Barlow Condensed","Arial Narrow",sans-serif;font-size:11px;letter-spacing:.08em;text-transform:uppercase;border:1px solid var(--rule-strong);border-radius:2px;padding:0 5px;color:var(--muted);vertical-align:middle}
.tag.accent{border-color:var(--primary-edge);color:var(--ink)}
.hide-ci td.delta .ci{display:none}
table.six td.ok{color:var(--good);font-weight:600}
table.six td.fail{color:var(--bad);font-weight:600}
table.six td.na{color:var(--muted)}
.findings{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px 22px;margin:0;padding:0;list-style:none}
.findings li{border-top:1px solid var(--rule-strong);padding-top:8px;max-width:64ch}
.findings li b{font-family:"Barlow Condensed","Arial Narrow",sans-serif;font-weight:600;font-size:17px;display:block;margin-bottom:2px}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px}
figure{margin:0;background:var(--surface);border:1px solid var(--rule);padding:8px}
figure img{width:100%;height:auto;display:block;image-rendering:auto}
figcaption{font-size:12.5px;color:var(--muted);padding:6px 2px 0}
.notes{max-width:76ch;color:var(--ink);font-size:14px}
.notes p{margin:0 0 8px}
code{font-family:"IBM Plex Mono",Consolas,monospace;font-size:12.5px;background:var(--surface-2);padding:1px 4px;border-radius:2px}
@media (max-width:1240px){.tiles{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media (max-width:960px){.tiles{grid-template-columns:repeat(2,minmax(0,1fr))}header.top{grid-template-columns:1fr}.meta{text-align:left}}
@media (prefers-reduced-motion:no-preference){.seg button,button.plain{transition:background .12s ease}}
</style>
"""

JS = """
<script>
(function(){
  var ci = document.getElementById('ci');
  ci.addEventListener('change', function(){ document.body.classList.toggle('hide-ci', !ci.checked); });
  var segs = document.querySelectorAll('.seg[data-filter="t"] button');
  function applyT(val){
    segs.forEach(function(b){ b.setAttribute('aria-pressed', b.dataset.t === val ? 'true' : 'false'); });
    document.querySelectorAll('tr.row').forEach(function(tr){
      var t = tr.dataset.t;
      tr.hidden = !(val === 'both' || t === val || t === '0.20');
    });
  }
  segs.forEach(function(b){ b.addEventListener('click', function(){ applyT(b.dataset.t); }); });
  document.querySelectorAll('table.ledger').forEach(function(table){
    var tbody = table.tBodies[0];
    var original = Array.prototype.slice.call(tbody.rows);
    var heads = table.querySelectorAll('th.sortable');
    heads.forEach(function(th, idx){
      th.addEventListener('click', function(){
        var col = Array.prototype.indexOf.call(th.parentNode.children, th);
        var dir = th.getAttribute('aria-sort') === 'descending' ? 'ascending' : 'descending';
        heads.forEach(function(h){ h.removeAttribute('aria-sort'); });
        th.setAttribute('aria-sort', dir);
        var rows = original.filter(function(r){ return r.classList.contains('row'); });
        rows.sort(function(a, b){
          var va = parseFloat(a.children[col].dataset.v), vb = parseFloat(b.children[col].dataset.v);
          if (isNaN(va)) va = -Infinity; if (isNaN(vb)) vb = -Infinity;
          return dir === 'descending' ? vb - va : va - vb;
        });
        original.forEach(function(r){ if (r.classList.contains('group')) r.hidden = true; });
        rows.forEach(function(r){ tbody.appendChild(r); });
      });
    });
    var reset = table.parentNode.parentNode.querySelector('button.reset');
    if (reset) reset.addEventListener('click', function(){
      heads.forEach(function(h){ h.removeAttribute('aria-sort'); });
      original.forEach(function(r){ if (r.classList.contains('group')) r.hidden = false; tbody.appendChild(r); });
    });
  });
})();
</script>
"""


def main() -> None:
    created = TABLE["created_utc"][:16].replace("T", " ") + " UTC"
    sections = []
    for key, title, sub in SETS:
        sections.append(
            f"<section><div class='sechead'><div><h2>Kaggle metrics on {esc(title)}</h2><p>{esc(sub)}. Paired deltas are against F0 200k at the same threshold, 95% cluster-bootstrap intervals (clusters = support anchor chunks, 2,000 resamples); a shaded delta has an interval that excludes zero.</p></div>"
            f"<button class='plain reset'>restore grouping</button></div><div class='scroll'>{set_table(key)}</div></section>")
    findings = """
<ul class='findings'>
<li><b>Flat M7 to the primary, same threshold</b>At T 0.30 on v14p2 the composite goes 0.5972 → 0.6521, Topo 0.3575 → 0.4016, VOI merge 1.242 → 1.044, VOI split 1.136 → 0.953, and the frozen-harness macro Dice 0.5575 → 0.7130. Both paired intervals (M7 below the reference, the primary above it) are tight relative to the gap.</li>
<li><b>Matched inference: M7 + TTA to the primary</b>Same threshold, same TTA. On v14p2 at 0.30 the composite goes 0.6095 → 0.6521, Topo 0.3709 → 0.4016, VOI merge 1.197 → 1.044, VOI split 1.127 → 0.953, macro Dice 0.5875 → 0.7130. On PHerc0500P2 the order flips: M7 + TTA scores 0.4942 against the primary's 0.4917, with better Topo (0.1586 vs 0.1437) and F1 β0 (0.320 vs 0.284), and at its calibrated 0.20 it reaches 0.5015, +0.012 [+0.009, +0.015] over the reference at 0.30 and the best composite on that set.</li>
<li><b>Two inference levers, on every model</b>Every row scores higher at 0.30 than at 0.35 on both Kaggle sets, and TTA adds +0.009 to +0.014 composite on v14p2 for each student (+0.012 for M7). Neither lever changes the weights.</li>
<li><b>Weight averages do not move the level</b>No average beats the reference on composite except the wide 100k–250k one (+0.003 at 0.30, +0.005 at 0.35); the narrow late averages sit 0.002 to 0.009 below it, matching the frozen-harness reading that averaging only reduces variance.</li>
<li><b>Human ground truth tells a different story</b>On PHerc0500P2 the composite is nearly flat across the table (0.476 to 0.497). Flat M7 at 0.30 ties the reference on composite and beats it on topology (F1 β0 0.288 vs 0.238); the students win Surface Dice and merges. TTA lifts each student's Topo back to M7's level and adds +0.001 to +0.006 composite, at the price of +0.05 to +0.11 VOI split. The best rows here are F0 110k + TTA (0.4965) and M7 + TTA (0.4942, with the best Topo and F1 β0 on this set); the sealed M7 + TTA at its calibrated 0.20 remains the tabled best composite at 0.5015.</li>
<li><b>Six-cube separation</b>Flat M7 at 0.30 carries 9,004 bridge pixels at skeleton recall 0.606 across 179 components and fails the thickness gate; F0 200k at 0.30 carries 4,945 at recall 0.782 across 123 components with gates and frontier passing. At 0.35 M7 drops to 2,021 bridges but recovers under half the reference skeleton (0.490), so its frontier position cannot be resolved against the C3 curve.</li>
</ul>"""
    html = CSS + f"""
<div class='page'>
<header class='top'>
  <div>
    <div class='eyebrow'>crossres · F0 family · all-metrics ledger</div>
    <h1>Flat M7 against every F0 variant</h1>
    <p class='lede'>Every declared variant of the 2026-09-08 ladder, plus the released M7 scored raw at the same thresholds, on the frozen macro Dice harness, both Kaggle metric sets, and the blind six-cube separation views. Two operating thresholds per row.</p>
  </div>
  <div class='meta'>table built {esc(created)}<br>reference for deltas: F0 200k, no TTA, same T<br>Kaggle composite = 0.35 Surface Dice + 0.35 VOI + 0.30 Topo<br>{len(rows)} rows · thresholds 0.30 and 0.35</div>
</header>

<div class='tiles'>
{tile('m7_raw', 'baseline · T 0.30')}
{tile('m7_tta', 'baseline + TTA · T 0.30')}
{tile('f0_200k', 'reference · T 0.30')}
{tile('f0_200k_tta', 'family primary · T 0.30')}
{tile('f0_110k_tta', 'best composite · T 0.30')}
</div>

<section>
  <h3>Readings</h3>
  {findings}
</section>

<div class='controls'>
  <span class='eyebrow'>threshold</span>
  <span class='seg' data-filter='t' role='group' aria-label='threshold filter'><button data-t='both' aria-pressed='true'>both</button><button data-t='0.30' aria-pressed='false'>0.30</button><button data-t='0.35' aria-pressed='false'>0.35</button></span>
  <label class='chk'><input type='checkbox' id='ci' checked> show 95% intervals</label>
  <span class='eyebrow'>click a numeric header to sort</span>
  <span class='legend'><span><span class='dir'>&uarr;</span> higher is better</span><span><span class='dir'>&darr;</span> lower is better</span><span><span class='dir'>&rarr;1</span> closest to one is better</span><span>shaded delta = interval excludes zero, green in the better direction</span></span>
</div>

{''.join(sections)}

<section>
  <div class='sechead'><div><h2>Six-cube instance separation</h2><p>Blind PHerc1447 cubes, TTA export at halo 32, thresholded at T (every row is a TTA export, so a checkpoint's flat and TTA variants share one row); bridges are inter-wrap merger pixels against the jackpot reference (fewer is better), skeleton recall is coverage of the reference sheet skeleton (more is better); gates = foreground ratio, interior and thickness; frontier = bridge count against the C3 curve interpolated at matched coverage.</p></div></div>
  <div class='scroll'>{six_cube_table()}</div>
</section>

<section>
  <div class='sechead'><div><h2>One review view</h2><p>Cube z12160_y04224_x02944 at z = 12,224: cyan = published M7, yellow = the incumbent (jackpot) student, magenta = the checkpoint named in the caption. The panels are veto material, never selection.</p></div></div>
  <div class='panels'>{panels()}</div>
</section>

<section class='notes'>
  <h3>Provenance</h3>
  <p>Frozen macro Dice: <code>crossres-voxel audit-checkpoint</code> on the v14p2 689-row validation manifest, per-scroll pooled Dice averaged over PHerc0814 and PHerc1451. Kaggle scores: <code>metric.exe</code> (tau 2.0, VOI alpha 0.3, ignore label 2) over per-case predictions dumped by <code>scripts/score_all_f0_variants.py</code>; the reference rows reuse the sealed release scoring. Six-cube receipts: <code>evaluate_final_c3_250k.blind_command</code> exports.</p>
  <p>Source table: <code>output/crossres_data/f0_family_20260908/variants/all_metrics/all_metrics_table.json</code> (schema <code>{esc(TABLE['schema'])}</code>). The family primary's Kaggle rows were scored after the ladder's top-k rule and folded into the decision table with a dated copy of the previous version.</p>
</section>
</div>
""" + JS
    OUT.write_text(html, encoding="utf-8")
    print(OUT, len(html.encode("utf-8")), "bytes")


if __name__ == "__main__":
    main()
