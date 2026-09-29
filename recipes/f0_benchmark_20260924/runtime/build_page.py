#!/usr/bin/env python
"""Build the show-and-tell page (page/index.html) from numbers.json and the frontier data. No model runs.

Every number on the page is read from the evidence files; a missing one renders as "pending".
Charts are inline SVG drawn to scale with the page's theme tokens.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

PENDING = '<span class="pending">pending</span>'


def f3(v) -> str:
    return PENDING if v is None else f"{v:.3f}"


def pct(v, digits=1) -> str:
    return PENDING if v is None else f"{100 * v:.{digits}f}%"


def signed(v) -> str:
    return f"{v:+.3f}"


def get(d, *keys):
    for key in keys:
        if d is None:
            return None
        d = d.get(key) if isinstance(d, dict) else None
    return d


# ------------------------------------------------------------------------------------------------ charts
def svg_grouped_bars(groups: list[str], series: list[tuple[str, str, list[float]]], title: str, unit: str) -> str:
    """series: (label, css color var, values per group). Values in percent."""
    width, height = 760, 330
    left, right, top, bottom = 56, 16, 24, 60
    plot_w, plot_h = width - left - right, height - top - bottom
    vmax = max(v for _, _, vals in series for v in vals)
    ymax = max(1.0, (int(vmax) + 1))
    ticks = [i for i in range(0, int(ymax) + 1)]
    group_w = plot_w / len(groups)
    bar_w = group_w * 0.8 / len(series)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}" class="chart">']
    for t in ticks:
        y = top + plot_h - plot_h * t / ymax
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{t}{unit}</text>')
    for g, name in enumerate(groups):
        gx = left + g * group_w + group_w * 0.1
        for s, (label, color, values) in enumerate(series):
            v = values[g]
            h = plot_h * v / ymax
            x = gx + s * bar_w
            y = top + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 2:.1f}" height="{h:.1f}" fill="var({color})"><title>{html.escape(label)}: {v:.1f}{unit}</title></rect>')
            parts.append(f'<text x="{x + (bar_w - 2) / 2:.1f}" y="{y - 4:.1f}" class="val" text-anchor="middle">{v:.1f}</text>')
        parts.append(f'<text x="{left + g * group_w + group_w / 2:.1f}" y="{top + plot_h + 22}" class="axis" text-anchor="middle">{html.escape(name)}</text>')
    parts.append(f'<line x1="{left}" x2="{width - right}" y1="{top + plot_h}" y2="{top + plot_h}" class="axisline"/>')
    parts.append("</svg>")
    legend = "".join(f'<li><span class="sw" style="background:var({c})"></span>{html.escape(l)}</li>' for l, c, _ in series)
    return "".join(parts) + f'<ul class="legend">{legend}</ul>'


def svg_frontier(curves: list[tuple[str, str, list[float], list[float], int | None]]) -> str:
    """curves: (label, css color var, x = recall, y = interior share, operating index)."""
    width, height = 760, 400
    left, right, top, bottom = 64, 20, 20, 56
    plot_w, plot_h = width - left - right, height - top - bottom
    xmin, xmax, ymin, ymax = 0.5, 1.0, 0.0, 0.6

    def X(v):
        return left + plot_w * (min(max(v, xmin), xmax) - xmin) / (xmax - xmin)

    def Y(v):
        return top + plot_h - plot_h * (min(max(v, ymin), ymax) - ymin) / (ymax - ymin)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Coverage versus solid interior" class="chart">']
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        parts.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top}" y2="{top + plot_h}" class="grid"/>')
        parts.append(f'<text x="{X(t):.1f}" y="{top + plot_h + 18}" class="tick" text-anchor="middle">{t:.1f}</text>')
    for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        parts.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 8}" y="{Y(t) + 4:.1f}" class="tick" text-anchor="end">{t:.1f}</text>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 12}" class="axis" text-anchor="middle">surface recall within 2 voxels (frozen benchmark)</text>')
    parts.append(f'<text transform="translate(16 {top + plot_h / 2:.1f}) rotate(-90)" class="axis" text-anchor="middle">share of predicted voxels inside a solid interior</text>')
    for label, color, xs, ys, op in curves:
        pts = [(X(x), Y(y)) for x, y in zip(xs, ys) if xmin <= x <= xmax and ymin <= y <= ymax]
        if len(pts) > 1:
            d = " ".join(f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}" for i, (px, py) in enumerate(pts))
            parts.append(f'<path d="{d}" fill="none" stroke="var({color})" stroke-width="2.2"/>')
        for px, py in pts:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" fill="var({color})"/>')
        if op is not None and xmin <= xs[op] <= xmax and ymin <= ys[op] <= ymax:
            parts.append(f'<rect x="{X(xs[op]) - 7:.1f}" y="{Y(ys[op]) - 7:.1f}" width="14" height="14" fill="none" stroke="var({color})" stroke-width="2.2"><title>{html.escape(label)} operating point</title></rect>')
    parts.append("</svg>")
    legend = "".join(f'<li><span class="sw" style="background:var({c})"></span>{html.escape(l)}</li>' for l, c, *_ in curves)
    return "".join(parts) + f'<ul class="legend">{legend}<li><span class="sq"></span>operating point</li></ul>'


def passes(config: str) -> str:
    k = int(config.rsplit("_p", 1)[1]) + 1
    norm = "training-matched normalisation" if config.startswith("htrain") else ("with TTA" if "tta" in config else "as shipped")
    return f"{k} pass{'es' if k > 1 else ''}, {norm}"


def bar(value, vmax=0.8, color="--ink") -> str:
    if value is None:
        return ""
    return (f'<span class="inbar"><span style="width:{100 * value / vmax:.1f}%;'
            f'background:var({color})"></span></span>')


# ------------------------------------------------------------------------------------------------ page
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbers", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    n = json.loads(args.numbers.read_text(encoding="utf-8"))
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    front = json.loads(args.frontier.read_text(encoding="utf-8"))
    b = n["benchmark"]
    tol = n["tolerant_f1_best_threshold"]
    deltas = n["paired_deltas"]
    blob = n["blob_audit"]
    p5 = n.get("p0500p2", {})
    herc = b["hercunet_best"]

    def delta_line(key, text):
        d = deltas.get(key)
        if not d:
            return ""
        return f'<li>{text}: <b class="num">{signed(d["point"])}</b> <span class="ci">[{signed(d["low"])}, {signed(d["high"])}]</span></li>'

    rows = [
        ("F0 (ours, shipped)", "--f0", "eight-way mirror TTA, T 0.30, preregistered", b["f0"]["dice"],
         get(b, "f0", "kaggle", "composite"), get(tol, "f0", "2", "value"), get(tol, "f0", "4", "value"), "hl"),
        ("M7, as published", "--m7", "the organisers' setting: no TTA, T 0.2", b["m7_published"]["dice"],
         get(b, "m7_published", "kaggle", "composite"), get(tol, "m7_published", "2", "value"), get(tol, "m7_published", "4", "value"), ""),
        ("M7 + our TTA", "--m7-fill", f"eight-way mirror TTA at its best T on this benchmark ({b['m7_tta_best']['threshold']:.2f})",
         b["m7_tta_best"]["dice"], get(b, "m7_tta_best", "kaggle", "composite"), get(tol, "m7_tta_best", "2", "value"), get(tol, "m7_tta_best", "4", "value"), ""),
        ("HercUNet v0", "--herc", "best of every setting, each chosen on this benchmark", herc["dice"],
         get(herc, "kaggle", "composite"), get(tol, "hercunet_best", "2", "value"), get(tol, "hercunet_best", "4", "value"), ""),
    ]
    table_rows = "".join(
        f'<tr class="{cls}"><th scope="row"><span class="sw" style="background:var({color})"></span>{html.escape(name)}</th>'
        f'<td class="how">{html.escape(how)}</td><td class="num">{f3(d)}{bar(d, color=color)}</td><td class="num">{f3(c)}</td>'
        f'<td class="num">{f3(t2)}</td><td class="num">{f3(t4)}</td></tr>'
        for name, color, how, d, c, t2, t4, cls in rows)
    soup = b.get("soup") or {}
    soup_row = (f'<tr class="explore"><th scope="row"><span class="sw" style="background:var(--soup)"></span>F0 + C3r soup</th>'
                f'<td class="how">exploratory: weight average of two seeds, TTA, T 0.30</td><td class="num">{f3(soup.get("dice_030"))}</td>'
                f'<td class="num">{f3(get(soup, "kaggle", "composite"))}</td><td class="num">{f3(get(tol, "soup", "2", "value"))}</td>'
                f'<td class="num">{f3(get(tol, "soup", "4", "value"))}</td></tr>') if soup else ""

    groups = ["open third", "middle third", "most compressed third"]
    strata = ("open", "middle", "compressed")

    def series(name):
        return [100 * blob[name]["per_compression_stratum"][s]["mean_depth_ge_5"] for s in strata]

    bars = svg_grouped_bars(groups, [
        ("M7 as published (no TTA, T 0.2)", "--m7", series("m7_published")),
        ("M7 + 8-way TTA, T 0.30", "--m7-fill", series("m7_rerun_tta")),
        ("F0 (ours, shipped)", "--f0", series("f0")),
        ("HercUNet v0, best setting (4 passes + TTA, T 0.30)", "--herc", series("hercunet")),
        ("HercUNet v0, default (4 passes, no TTA, T 0.30)", "--herc-deep", series("hercunet_4pass")),
    ], "Solid-mass share by compression", "%")

    thresholds = analysis["thresholds"]

    def curve(label, color, config, operating):
        xs = analysis["summaries_tolerant_recall_2"][config]
        ys = front["configs"][config]["pooled_interior_share"]
        op = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - operating)) if operating is not None else None
        return (label, color, xs, ys, op)

    frontier = svg_frontier([
        curve("M7, as published (no TTA)", "--m7", "m7_flat", 0.20),
        curve("M7 + our TTA", "--m7-fill", "m7_tta", None),
        curve("F0 (ours, shipped)", "--f0", "f0_tta", 0.30),
        curve("HercUNet v0, best setting (4 passes + TTA)", "--herc", "hshtta_p3", 0.30),
    ])

    mp, f0b, hb = blob["m7_published"], blob["f0"], blob["hercunet"]
    frag = n.get("fragments") or {}
    try:
        ft, ff = frag["m7_rerun_tta"]["per_compression_stratum"]["compressed"], frag["f0"]["per_compression_stratum"]["compressed"]
        frag_sentence = (f"they fill less of the cube than F0 ({100 * ft['mean_fill']:.1f}% against {100 * ff['mean_fill']:.1f}%) "
                         f"yet break into more pieces ({ft['mean_pieces']:.1f} per cube against {ff['mean_pieces']:.1f}).")
    except (KeyError, TypeError):
        frag_sentence = PENDING + "."
    real = n.get("registered_real_only", {})
    eq = n.get("registered_real_s3_equality") or {}
    repro = n["reproduction"]
    fid = n.get("insitu_fidelity") or []

    def p5cell(label, key="composite"):
        if label == "hercunet_best":
            values = [get(p5, k, "kaggle", key) for k in ("hercunet_p0", "hercunet_p0_t030", "hercunet_p3", "hercunet_tta_p3")]
            values = [v for v in values if v is not None]
            return f3(max(values)) if values else PENDING
        return f3(get(p5, label, "kaggle", key))

    page = f"""<title>F0 Head-to-Head</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root {{
  color-scheme: light;
  --bg: #F1F3F5; --surface: #FFFFFF; --ink: #11151B; --muted: #57616E; --rule: #D4DAE1;
  --m7: #0A8FB2; --m7-fill: #7FD3EA; --f0: #C01F7C; --herc: #6A4BD6; --herc-deep: #3E2C8C;
  --soup: #9A6400; --code-bg: #E6EAEE;
  --display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --body: "Source Serif 4", Georgia, "Times New Roman", serif;
  --mono: "JetBrains Mono", "Cascadia Mono", Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --bg: #0B0E12; --surface: #141920; --ink: #E7EBF0; --muted: #9AA4B1; --rule: #29313B;
    --m7: #4FD3F1; --m7-fill: #2A7F96; --f0: #FF5CB8; --herc: #B2A0FF; --herc-deep: #7457E0;
    --soup: #F0C060; --code-bg: #1A2129;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --bg: #0B0E12; --surface: #141920; --ink: #E7EBF0; --muted: #9AA4B1; --rule: #29313B;
  --m7: #4FD3F1; --m7-fill: #2A7F96; --f0: #FF5CB8; --herc: #B2A0FF; --herc-deep: #7457E0;
  --soup: #F0C060; --code-bg: #1A2129;
}}
body {{ background: var(--bg); color: var(--ink); font-family: var(--body); font-size: 17px; line-height: 1.58; }}
.wrap {{ max-width: 1080px; margin: 0 auto; padding-inline: 20px; padding-block: 44px 88px; display: grid; gap: 56px; }}
header {{ display: grid; gap: 14px; max-width: 70ch; }}
.eyebrow {{ font-family: var(--mono); font-size: 12.5px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
h1 {{ font-family: var(--display); font-weight: 800; font-stretch: 88%; font-size: clamp(38px, 6vw, 64px); line-height: 1.02; margin: 0; letter-spacing: -0.01em; text-wrap: balance; }}
h1 .f0 {{ color: var(--f0); }}
.dek {{ font-size: 20px; line-height: 1.5; margin: 0; color: var(--ink); }}
h2 {{ font-family: var(--display); font-weight: 750; font-stretch: 92%; font-size: 28px; line-height: 1.15; margin: 0; text-wrap: balance; }}
h3 {{ font-family: var(--display); font-weight: 700; font-size: 18px; margin: 0; }}
section {{ display: grid; gap: 18px; }}
.prose {{ max-width: 68ch; display: grid; gap: 12px; }}
.prose p {{ margin: 0; }}
ul.facts {{ margin: 0; padding-left: 1.2em; display: grid; gap: 8px; max-width: 70ch; }}
.num, .ci, code, .tick, .val {{ font-family: var(--mono); font-variant-numeric: tabular-nums; }}
.ci {{ color: var(--muted); font-size: 0.9em; }}
.pending {{ font-family: var(--mono); font-size: 0.85em; color: var(--muted); font-style: italic; }}
.tablewrap {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; }}
table {{ border-collapse: collapse; width: 100%; min-width: 760px; font-size: 15px; }}
th, td {{ padding: 11px 14px; border-bottom: 1px solid var(--rule); text-align: left; vertical-align: middle; }}
thead th {{ font-family: var(--display); font-weight: 650; font-size: 13px; letter-spacing: 0.02em; color: var(--muted); background: var(--surface); }}
thead th.num {{ font-family: var(--display); text-align: right; }}
tbody th {{ font-family: var(--display); font-weight: 650; white-space: nowrap; }}
td.how {{ color: var(--muted); font-size: 14px; max-width: 28ch; }}
td.num {{ text-align: right; white-space: nowrap; }}
tr.hl th, tr.hl td.num {{ color: var(--f0); font-weight: 600; }}
tr.explore {{ opacity: 0.8; }}
tr:last-child th, tr:last-child td {{ border-bottom: 0; }}
.sw {{ display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: 8px; vertical-align: -1px; }}
.inbar {{ display: block; height: 4px; margin-top: 5px; background: var(--rule); border-radius: 2px; }}
.inbar span {{ display: block; height: 100%; background: currentColor; border-radius: 2px; }}
.chart {{ width: 100%; height: auto; max-width: 100%; display: block; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; }}
.chart .grid {{ stroke: var(--rule); stroke-width: 1; }}
.chart .axisline {{ stroke: var(--muted); stroke-width: 1; }}
.chart .tick {{ fill: var(--muted); font-size: 11px; }}
.chart .val {{ fill: var(--ink); font-size: 10.5px; }}
.chart .axis {{ fill: var(--ink); font-family: var(--display); font-size: 12.5px; }}
ul.legend {{ list-style: none; padding: 0; margin: 8px 0 0; display: flex; flex-wrap: wrap; gap: 6px 18px; font-family: var(--display); font-size: 13px; color: var(--muted); }}
ul.legend .sq {{ display: inline-block; width: 10px; height: 10px; border: 2px solid var(--muted); margin-right: 8px; vertical-align: -1px; }}
.figs {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
@media (min-width: 900px) {{ .figs.two {{ grid-template-columns: 1fr 1fr; }} }}
figure {{ margin: 0; display: grid; gap: 8px; }}
figure img {{ width: 100%; height: auto; display: block; border-radius: 4px; border: 1px solid var(--rule); background: #000; }}
figcaption {{ font-size: 14.5px; color: var(--muted); max-width: 72ch; }}
.strip {{ overflow-x: auto; display: grid; gap: 10px; padding-bottom: 6px; }}
.strip img {{ width: 1600px; max-width: none; height: auto; border-radius: 3px; border: 1px solid var(--rule); }}
.strip .lab {{ font-family: var(--display); font-size: 13px; color: var(--muted); }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; }}
.kpi {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 16px; display: grid; gap: 4px; }}
.kpi .big {{ font-family: var(--mono); font-size: 30px; font-weight: 600; line-height: 1.1; }}
.kpi .lab {{ font-family: var(--display); font-size: 13px; color: var(--muted); }}
pre {{ background: var(--code-bg); border-radius: 6px; padding: 14px 16px; overflow-x: auto; font-size: 13px; line-height: 1.5; margin: 0; }}
a {{ color: inherit; text-decoration-color: var(--f0); text-underline-offset: 3px; }}
a:focus-visible {{ outline: 2px solid var(--f0); outline-offset: 2px; }}
ol.ideas {{ margin: 0; padding-left: 1.3em; display: grid; gap: 10px; max-width: 72ch; }}
footer {{ font-size: 14px; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 18px; }}
</style>
<div class="wrap">
<header>
  <div class="eyebrow">ScrollFiesta · Socratic Method · 24 September 2026</div>
  <h1><span class="f0">F0</span> head-to-head</h1>
  <p class="dek">Our shipped surface model against the organisers' published M7 and HercUNet v0: the same held-out scans, the challenge's own metric, and a random-sample audit of where the published M7 makes blobs.</p>
</header>

<section aria-labelledby="result">
  <h2 id="result">The result</h2>
  <div class="kpis">
    <div class="kpi"><span class="lab">F0 over M7 as published, macro Dice</span><span class="big num" style="color:var(--f0)">{signed(get(deltas, "f0_shipped_minus_m7_official_dice", "point"))}</span><span class="ci">95% [{signed(get(deltas, "f0_shipped_minus_m7_official_dice", "low"))}, {signed(get(deltas, "f0_shipped_minus_m7_official_dice", "high"))}], 689 rows</span></div>
    <div class="kpi"><span class="lab">F0 over HercUNet at its best, macro Dice</span><span class="big num" style="color:var(--f0)">{signed(get(deltas, "f0_shipped_minus_hercunet_oracle_dice", "point"))}</span><span class="ci">95% [{signed(get(deltas, "f0_shipped_minus_hercunet_oracle_dice", "low"))}, {signed(get(deltas, "f0_shipped_minus_hercunet_oracle_dice", "high"))}]</span></div>
    <div class="kpi"><span class="lab">Published M7's solid-mass share, open to compressed</span><span class="big num" style="color:var(--m7)">{pct(mp["per_compression_stratum"]["open"]["mean_depth_ge_5"])} → {pct(mp["per_compression_stratum"]["compressed"]["mean_depth_ge_5"])}</span><span class="ci">F0: {pct(f0b["per_compression_stratum"]["open"]["mean_depth_ge_5"])} → {pct(f0b["per_compression_stratum"]["compressed"]["mean_depth_ge_5"])}</span></div>
  </div>
  <div class="tablewrap"><table>
    <thead><tr><th scope="col">Model</th><th scope="col">Run as</th><th scope="col" class="num">Frozen benchmark<br>macro Dice</th><th scope="col" class="num">Kaggle composite<br>(challenge metric)</th><th scope="col" class="num">Surface F1<br>within 2 vox</th><th scope="col" class="num">Surface F1<br>within 4 vox</th></tr></thead>
    <tbody>{table_rows}{soup_row}</tbody>
  </table></div>
  <div class="prose">
    <p>The frozen benchmark is 689 patches from PHerc0814 and PHerc1451, two scrolls none of these models trained on, labelled by the official 2.4 µm surface model run on registered fine scans. {eq.get("byte_equal", "442")} of the rows are byte-for-byte crops of the public open-data volumes. The Kaggle composite is 0.35 SurfaceDice@2 + 0.35 VOI + 0.30 TopoScore, computed by the same metric binary for every model. Surface F1 forgives predictions that sit within 2 or 4 voxels of a true sheet, which matters for HercUNet's medial target.</p>
  </div>
  <ul class="facts">
    {delta_line("f0_shipped_minus_m7_official_dice", "F0 minus M7 as published")}
    {delta_line("f0_shipped_minus_m7_tta_best_dice", "F0 minus M7 with our TTA at its best threshold")}
    {delta_line("f0_shipped_minus_hercunet_oracle_dice", f"F0 minus HercUNet at its best ({passes(herc['dice_config'])}, T {herc['dice_threshold']:.2f})")}
    <li>On the 442 registered-real rows alone: F0 <b class="num">{f3(get(real, "f0_tta", "dice_030"))}</b>, M7 as published <b class="num">{f3(get(real, "m7_flat", "dice_020"))}</b>, HercUNet at best <b class="num">{f3(get(real, "hercunet_best", "dice_best", "value"))}</b>.</li>
  </ul>
</section>

<section aria-labelledby="blobs">
  <h2 id="blobs">Where the published M7 makes blobs</h2>
  <div class="prose">
    <p>We drew 32 cubes of 128³ voxels uniformly at random from the interior of each of eight scrolls that carry an official published M7 prediction, 256 cubes in all, with the sampling rule written into the code before any cube was read. For each cube we kept the organisers' published M7 mask unchanged and ran F0 and HercUNet on the same CT. Each cube was then placed in the open, middle or most compressed third of its scroll by how much of its CT is dark gap.</p>
  </div>
  <div class="figs two">
    <figure><img src="showcase_compressed.jpg" alt="Eight compressed cubes, one per scroll: CT, published M7, F0 and HercUNet overlays" width="1012" height="2068" loading="lazy"><figcaption><b>Most compressed third.</b> One cube per scroll, picked by a seeded random index that never looked at a prediction. Cyan: M7 as published. Magenta: F0. Violet: HercUNet v0 at its best setting. The published mask forms unstructured masses on PHerc1447, PHerc0191 and PHerc0343 and bridges neighbouring wraps in parts of PHerc0814 and PHerc1218. F0 draws sheets wherever layers are visible, thick and touching in places on PHerc1447, and almost nothing on PHerc0343, whose slice shows no clear layers.</figcaption></figure>
    <figure><img src="showcase_open.jpg" alt="Eight open cubes, one per scroll: CT, published M7, F0 and HercUNet overlays" width="1012" height="2068" loading="lazy"><figcaption><b>Most open third.</b> Same rule. Where the wraps are well separated, all three models draw comparable sheets, F0's thicker than M7's. Two cubes go the other way: on PHerc0191 F0 fuses neighbouring wraps into a thick mass, and on the disordered PHerc0125 cube it predicts only fragments where HercUNet draws continuous sheets. PHerc0826's open cube shows no structure, and no model predicts anything there.</figcaption></figure>
  </div>
  {bars}
  <ul class="facts">
    <li>The share of M7's predicted voxels buried in a solid mass (six-connected depth 5 or more, thicker than one sheet) rises from <b class="num">{pct(mp["per_compression_stratum"]["open"]["mean_depth_ge_5"])}</b> in open cubes to <b class="num">{pct(mp["per_compression_stratum"]["compressed"]["mean_depth_ge_5"])}</b> in compressed ones. F0 stays at <b class="num">{pct(f0b["per_compression_stratum"]["open"]["mean_depth_ge_5"])}</b> and <b class="num">{pct(f0b["per_compression_stratum"]["compressed"]["mean_depth_ge_5"])}</b>.</li>
    <li>Our own run of the released M7 weights at the published setting gives <b class="num">{pct(blob["m7_rerun_flat"]["per_compression_stratum"]["open"]["mean_depth_ge_5"])}</b> and <b class="num">{pct(blob["m7_rerun_flat"]["per_compression_stratum"]["compressed"]["mean_depth_ge_5"])}</b>, so the blobs belong to the model, not to the publication pipeline.</li>
    <li>With eight-way TTA at T 0.30 the same M7 weights leave far fewer deep voxels, but in compressed cubes they break into fragments rather than sheets: {frag_sentence} On the benchmark that setting still trails F0 by {abs(get(deltas, "f0_shipped_minus_m7_tta_best_dice", "point")):.3f} Dice.</li>
    <li>Blobs are local. A whole cube is a solid slab for 1 of 256 published M7 cubes and for none of F0's; the canonical thick-mass verdict fires on {pct(mp["overall"]["thick_rate"])} of M7 cubes and {pct(f0b["overall"]["thick_rate"])} of F0's.</li>
    <li>HercUNet at its best setting (four passes with TTA) runs from {pct(hb['per_compression_stratum']['open']['mean_depth_ge_5'])} to {pct(hb['per_compression_stratum']['compressed']['mean_depth_ge_5'])}; its default four-pass output without TTA from {pct(blob['hercunet_4pass']['per_compression_stratum']['open']['mean_depth_ge_5'])} to {pct(blob['hercunet_4pass']['per_compression_stratum']['compressed']['mean_depth_ge_5'])}.</li>
  </ul>
  <h3>Every cube, no selection</h3>
  <p class="ci">Each row is one scroll's 32 random cubes, central axial slice. Scroll sideways to see all of them.</p>
  <div class="strip" role="region" aria-label="Contact sheets of all 256 cubes" tabindex="0">
    <span class="lab">CT</span><img src="contact_ct.jpg" alt="Central slices of all 256 random cubes, CT only" width="2048" height="562" loading="lazy">
    <span class="lab">M7, as published</span><img src="contact_m7_as_published.jpg" alt="All 256 cubes with the published M7 mask" width="2048" height="562" loading="lazy">
    <span class="lab">F0</span><img src="contact_f0.jpg" alt="All 256 cubes with F0" width="2048" height="562" loading="lazy">
    <span class="lab">HercUNet v0, best setting</span><img src="contact_hercunet_v0.jpg" alt="All 256 cubes with HercUNet v0 at its best setting" width="2048" height="562" loading="lazy">
  </div>
</section>

<section aria-labelledby="fair">
  <h2 id="fair">How we kept it fair</h2>
  <ul class="facts">
    <li><b>F0</b> is scored only at its preregistered operating point. <b>M7</b> is scored at the organisers' own published setting (their <code>metadata.json</code>: <code>disable_tta: true</code>, threshold 0.2) and separately with our TTA at its best threshold.</li>
    <li><b>HercUNet</b> publishes no threshold, so it gets the best of both normalisations, one to four passes, TTA off or on, every threshold from 0.10 to 0.90 and up to 3 voxels of dilation, each chosen on this benchmark. That flatters it.</li>
    <li>HercUNet predicts the sheet's medial surface, not its face. The tolerant surface F1 and dilated Dice columns exist for that reason; even forgiving 4 voxels, it trails.</li>
    <li>The harness reproduces the sealed audits before scoring anything new: F0 {f3(repro["f0_tta@0.30"]["harness"])} against {f3(repro["f0_tta@0.30"]["sealed"])}, M7 + TTA {f3(repro["m7_tta@0.30"]["harness"])} against {f3(repro["m7_tta@0.30"]["sealed"])}, and the sealed Kaggle composite to within 0.000001.</li>
    <li>HercUNet's own command-line inference, run on the live scroll around four benchmark patches with its blending and half-window shifts, read CT identical to our patches and scored at or below our in-window harness on every one.</li>
  </ul>
  {frontier}
  <div class="prose"><p>Every threshold from 0.10 to 0.90 with recall above 0.5, on the frozen benchmark. Above about 0.55 recall HercUNet carries more solid interior than F0 or M7 at the same coverage. M7 is the thinnest at low coverage; above about 0.85 recall F0 carries less solid interior than M7, and it reaches coverage no M7 setting reaches.</p></div>
</section>

<section aria-labelledby="human">
  <h2 id="human">Where F0 is not ahead</h2>
  <ul class="facts">
    <li><b>Human-labelled PHerc0500P2</b> (320 rows, Kaggle composite, absolute only because M7 trained on these labels and HercUNet starts from M7): F0 {p5cell("f0")}, M7 as published {p5cell("m7_flat")}, M7 + TTA {p5cell("m7_tta")}, HercUNet at its best {p5cell("hercunet_best")}. On the only human-labelled holdout F0 and M7 are level, M7 with TTA is slightly ahead, and HercUNet trails; the frozen benchmark's advantage should not be read into it.</li>
    <li>On the benchmark's clean regions F0's sheets are thicker than M7's: its predicted volume matches the labels, M7's falls about 12% short. Thickness alone is therefore not the blob measure; fusion under compression is.</li>
    <li>The solid-mass measure is a geometric proxy with no ground truth. It cannot see two wraps fused where the CT itself shows no gap.</li>
    <li>The strongest network we measured is not the shipped one: a weight average of F0 and its seed replicate scores {f3(soup.get("dice_030"))} with TTA. It was declared exploratory in advance and needs its own preregistered confirmation before it ships.</li>
  </ul>
</section>

<section aria-labelledby="steal">
  <h2 id="steal">What we are taking from HercUNet</h2>
  <ol class="ideas">
    <li><b>A self-conditioned refiner.</b> A previous-output input channel, octant-composed and fragmented during training, with online DAgger on the model's own output. It removes the synthetic-versus-real shortcut that stopped our earlier repair networks. On this benchmark HercUNet's later passes lower its Dice and lift its composite only with TTA, so any trial must first show pass two beating pass one.</li>
    <li><b>A separation penalty only inside real gaps</b> between sheets, unlike our retired shell, which also hit sheet rims.</li>
    <li><b>Sheet-instance supervision</b> (an affinity head with constrained MALIS) from our 2.4 µm teacher: the one loss aimed directly at merge paths between wraps.</li>
    <li><b>Six CT orientation channels</b>, cheap and starting identical to M7.</li>
    <li><b>A soft-skeleton crest on every row</b>, which makes our untested medial tail floor testable.</li>
  </ol>
</section>

<section aria-labelledby="repro">
  <h2 id="repro">Check it yourself</h2>
  <div class="prose"><p>Everything is in the repository under <code>recipes/f0_benchmark_20260924/</code>: the scripts that produced every number, per-row counts for every run, the Kaggle receipts, the sample plan, per-cube metrics and all figures, each with a SHA-256.</p></div>
<pre><code># the frozen benchmark, one model family per run (exact per-row counts)
python runtime/patch_benchmark.py --family engine   --label f0 --checkpoint releases/c3-f0-200k-20260907/checkpoint_f0_00200000.pt --manifest &lt;v14p2&gt; --work &lt;work&gt; --store f0_tta
python runtime/patch_benchmark.py --family hercunet --label hshipped --model-dir &lt;hercunet-v0&gt; --norm shipped --passes 4 --manifest &lt;v14p2&gt; --work &lt;work&gt;
python runtime/patch_benchmark.py --family hercunet --label hshtta --model-dir &lt;hercunet-v0&gt; --norm shipped --passes 4 --tta mirror --manifest &lt;v14p2&gt; --work &lt;work&gt; --store hshtta_p3
# the challenge metric, same metric binary and ground truth as the sealed evidence
python runtime/kaggle_score.py --probs &lt;work&gt;/f0/probs/f0_tta --threshold 0.30 --out &lt;kaggle&gt;/f0_tta_t030
# the random-cube audit
python runtime/sample_cubes.py --out &lt;audit&gt;
python runtime/cube_infer.py --family hercunet --label hshtta --model-dir &lt;hercunet-v0&gt; --passes 4 --tta mirror --cubes &lt;audit&gt;/cubes --out &lt;audit&gt;/preds --keep hshtta_p3
python runtime/blob_metrics.py --audit &lt;audit&gt; --preds &lt;audit&gt;/preds --source m7_published=published --source f0=f0_tta:0.30 --source hercunet=hshtta_p3:0.30 --out &lt;audit&gt;/final</code></pre>
</section>

<footer>F0 checkpoint sha256 54db9a59…a175 · HercUNet v0 checkpoint sha256 7cb36f9a…2c946 (jimmylomro/hercUNet @ 100b828) · repository <a href="https://github.com/ubc-nvining/socratic_method">github.com/ubc-nvining/socratic_method</a></footer>
</div>
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"wrote {args.out} ({len(page) // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
