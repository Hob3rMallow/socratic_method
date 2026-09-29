#!/usr/bin/env python
"""Fill the record README's {{...}} tables from numbers.json (the README keeps its prose; numbers come from data)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def f4(v) -> str:
    return "pending" if v is None else f"{v:.4f}"


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def passes(config: str) -> str:
    k = int(config.rsplit("_p", 1)[1]) + 1
    norm = ("training-matched normalisation, no TTA" if config.startswith("htrain")
            else ("mirror TTA" if "tta" in config else "no TTA"))
    return f"{k} pass{'es' if k > 1 else ''}, {norm}"


def setting(kaggle_name: str) -> str:
    """hshtta_p3_t030 -> '4 passes, mirror TTA, T 0.30'."""
    config, t = kaggle_name.rsplit("_t", 1)
    return f"{passes(config)}, T {int(t) / 100:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbers", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True, help="README with {{...}} slots")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    n = json.loads(args.numbers.read_text(encoding="utf-8"))
    b, tol, d = n["benchmark"], n["tolerant_f1_best_threshold"], n["paired_deltas"]
    h = b["hercunet_best"]
    rows = [
        ("**F0, shipped** (TTA, T 0.30)", b["f0"]["dice"], get(b, "f0", "kaggle", "composite"),
         get(tol, "f0", "2", "value"), get(tol, "f0", "4", "value")),
        ("M7, as published (no TTA, T 0.2)", b["m7_published"]["dice"], get(b, "m7_published", "kaggle", "composite"),
         get(tol, "m7_published", "2", "value"), get(tol, "m7_published", "4", "value")),
        (f"M7 + our TTA (T {b['m7_tta_best']['threshold']:.2f})", b["m7_tta_best"]["dice"],
         get(b, "m7_tta_best", "kaggle", "composite"), get(tol, "m7_tta_best", "2", "value"), get(tol, "m7_tta_best", "4", "value")),
        ("HercUNet v0, best of every setting (below)", h["dice"],
         get(h, "kaggle", "composite"), get(tol, "hercunet_best", "2", "value"), get(tol, "hercunet_best", "4", "value")),
    ]
    soup = b.get("soup")
    if soup:
        rows.append(("F0 + C3r weight soup, exploratory (TTA, T 0.30)", soup["dice_030"], get(soup, "kaggle", "composite"),
                     get(tol, "soup", "2", "value"), get(tol, "soup", "4", "value")))
    bench = ["| model | macro Dice | Kaggle composite | surface F1 within 2 vox | within 4 vox |", "|---|---:|---:|---:|---:|"]
    bench += [f"| {name} | {f4(dv)} | {f4(c)} | {f4(t2)} | {f4(t4)} |" for name, dv, c, t2, t4 in rows]
    kag = h.get("kaggle_all") or {}

    def order(name):
        config, t = name.rsplit("_t", 1)
        return ("tta" in config, int(config.rsplit("_p", 1)[1]), int(t))

    bench += ["",
              f"HercUNet's best Dice comes from {passes(h['dice_config'])} at T {h['dice_threshold']:.2f}, and its best "
              f"Kaggle composite from {setting(h['kaggle_best_name'])}. Its best dilated Dice is "
              f"{h['dice_any_dilation']:.4f} ({passes(h['dice_any_dilation_config'])}, dilation {h['dice_any_dilation_r']}). "
              "Surface F1 is read at each model's declared threshold (F0 0.30, M7 0.20) and at HercUNet's best. "
              "M7 composites at T 0.20 are the sealed exact-threshold receipts; our re-scores from stored uint8 maps "
              "agree within 0.0004.",
              "",
              "| HercUNet v0 setting scored on the Kaggle metric | composite | SurfaceDice | VOI score | TopoScore |",
              "|---|---:|---:|---:|---:|"]
    bench += [f"| {setting(name)} | {v['composite']:.4f} | {v['surface_dice']:.4f} | {v['voi_score']:.4f} | "
              f"{v['toposcore']:.4f} |" for name, v in sorted(kag.items(), key=lambda kv: order(kv[0]))]
    deltas = ["| comparison | delta | 95% interval |", "|---|---:|---|"]
    names = {
        "f0_shipped_minus_m7_official_dice": "F0 shipped minus M7 as published",
        "f0_shipped_minus_m7_tta_best_dice": "F0 shipped minus M7 + TTA at its best threshold",
        "f0_shipped_minus_hercunet_oracle_dice": "F0 shipped minus HercUNet at its best",
        "f0_shipped_minus_hercunet_oracle_dilated_dice": "F0 shipped minus HercUNet at its best with dilation",
    }
    for key, label in names.items():
        v = d.get(key)
        if v:
            deltas.append(f"| {label} | {v['point']:+.4f} | [{v['low']:+.4f}, {v['high']:+.4f}] |")
    blob = n["blob_audit"]
    labels = [("m7_published", "M7, as published (no TTA, T 0.2)"), ("m7_rerun_flat", "M7, our rerun of the released weights, same setting"),
              ("m7_rerun_tta", "M7 + 8-way TTA, T 0.30"), ("f0", "F0, shipped (TTA, T 0.30)"),
              ("hercunet", "HercUNet v0, best setting (4 passes, mirror TTA, T 0.30)"),
              ("hercunet_4pass", "HercUNet v0, 4 passes, no TTA, T 0.30"), ("hercunet_1pass", "HercUNet v0, 1 pass, no TTA, T 0.50")]
    expected = {"m7_rerun_flat": ("m7_flat", 0.20), "m7_rerun_tta": ("m7_tta", 0.30), "f0": ("f0_tta", 0.30),
                "hercunet": ("hshtta_p3", 0.30), "hercunet_4pass": ("hshipped_p3", 0.30), "hercunet_1pass": ("hshipped_p0", 0.50)}
    for key, (config, threshold) in expected.items():
        assert blob[key]["config"] == config and abs(blob[key]["threshold"] - threshold) < 1e-6, (key, blob[key]["config"])
    bt = ["| source | solid-mass share, open | middle | compressed | all | thick-mass cubes | slab cubes | on dark CT |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, label in labels:
        s = blob[key]
        ps = s["per_compression_stratum"]
        bt.append(f"| {label} | {100 * ps['open']['mean_depth_ge_5']:.1f}% | {100 * ps['middle']['mean_depth_ge_5']:.1f}% | "
                  f"{100 * ps['compressed']['mean_depth_ge_5']:.1f}% | {100 * s['overall']['mean_depth_ge_5']:.1f}% | "
                  f"{100 * s['overall']['thick_rate']:.1f}% | {100 * s['overall']['solid_slab_rate']:.1f}% | "
                  f"{100 * s['overall']['mean_dark_fraction']:.1f}% |")
    p5 = n.get("p0500p2", {})
    pt = ["| model | Kaggle composite | SurfaceDice | VOI score | TopoScore | pooled Dice at its threshold |",
          "|---|---:|---:|---:|---:|---:|"]
    for key, label, dkey in (("f0", "F0, shipped (TTA, T 0.30)", "dice_030"), ("m7_flat", "M7, as published (no TTA, T 0.2)", "dice_020"),
                             ("m7_tta", "M7 + TTA, T 0.20", "dice_020"), ("hercunet_p0_t030", "HercUNet v0, 1 pass, no TTA, T 0.30", None),
                             ("hercunet_p0", "HercUNet v0, 1 pass, no TTA, T 0.50", None), ("hercunet_p3", "HercUNet v0, 4 passes, no TTA, T 0.30", None),
                             ("hercunet_tta_p3", "HercUNet v0, 4 passes, mirror TTA, T 0.30", None),
                             ("soup", "F0 + C3r soup, exploratory (TTA, T 0.30)", "dice_030")):
        e = p5.get(key, {})
        k = e.get("kaggle") or {}
        dv = e.get(dkey) if dkey else None
        pt.append(f"| {label} | {f4(k.get('composite'))} | {f4(k.get('surface_dice'))} | {f4(k.get('voi_score'))} | "
                  f"{f4(k.get('toposcore'))} | {f4(dv) if dkey else 'n/a'} |")
    soup_extra = ""
    if soup and get(soup, "kaggle", "composite") is not None:
        soup_extra = (f"Its Kaggle composite on the same rows is {get(soup, 'kaggle', 'composite'):.4f} against the shipped "
                      f"{get(b, 'f0', 'kaggle', 'composite'):.4f}, and {f4(get(p5, 'soup', 'kaggle', 'composite'))} on the "
                      "human-labelled PHerc0500P2.")
    rr = n["registered_real_only"]
    hr = rr["hercunet_best"]
    real_line = (f"On the 442 registered-real rows alone the order is unchanged: F0 {rr['f0_tta']['dice_030']:.4f}, M7 as"
                 + chr(10)
                 + f"published {rr['m7_flat']['dice_020']:.4f}, M7 + TTA at best {rr['m7_tta']['dice_best']['value']:.4f}, "
                 f"HercUNet at best {hr['dice_best']['value']:.4f} ({passes(hr['config'])}, "
                 f"T {hr['dice_best']['threshold']:.2f}).")
    text = args.template.read_text(encoding="utf-8")
    for slot, value in (("{{BENCHMARK_TABLE}}", "\n".join(bench)), ("{{DELTA_TABLE}}", "\n".join(deltas)),
                        ("{{BLOB_TABLE}}", "\n".join(bt)), ("{{P0500_TABLE}}", "\n".join(pt)),
                        ("{{SOUP_EXTRA}}", soup_extra), ("{{REAL_ONLY_LINE}}", real_line), ("{{STATUS_LINE}}\n\n", "")):
        text = text.replace(slot, value)
    if "{{" in text:
        i = text.index("{{")
        raise SystemExit("unfilled slot remains: " + text[i:i + 40])
    args.out.write_text(text, encoding="utf-8")
    print("filled", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
