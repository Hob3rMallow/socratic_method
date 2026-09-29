#!/usr/bin/env python
"""Build the benchmark evidence tables from the finished runs (no model is run here).

Comparison rules, fixed before any full HercUNet number existed (after the 41-row pilot):
  * F0: the shipped recipe exactly as preregistered on 2026-09-08: eight-way mirror TTA, T = 0.30.
  * M7 (official): the organisers' own published operating point, read from their metadata.json:
    TTA disabled, threshold 0.2. Also reported: M7 with our TTA at its best threshold on this benchmark.
  * HercUNet v0: its authors publish no operating threshold. It receives the most favourable choice of
    normalisation (as shipped / training-matched), pass count (1-4), TTA (off / eight-way mirror), threshold
    (0.10-0.90) and, for overlap metrics, dilation (0-3 voxels), each selected ON THIS BENCHMARK. That
    selection is optimistic for HercUNet and therefore conservative for any claim that F0 is ahead.
  * For symmetry every table also shows F0's and M7's best threshold on this benchmark.
Uncertainty: paired cluster bootstrap over support_anchor_chunk_zyx within scroll, B = 2000, seed 20260908.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_counts import (  # noqa: E402
    SCROLLS, config_arrays, load_run, macro_from, paired_delta, summarize, tolerant_from)

SEALED = {
    ("f0_tta", 0.30): 0.713025708376333,
    ("f0_tta", 0.35): 0.7038261039837073,
    ("f0_flat", 0.35): 0.680787428136385,
    ("m7_tta", 0.30): 0.5874874255181086,
    ("m7_flat", 0.35): 0.5541400563365637,
}


def subset(arrays: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[keep] for k, v in arrays.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True, help="work/full")
    parser.add_argument("--kaggle", type=Path, required=True, help="work/kaggle")
    parser.add_argument("--insitu", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for label in ("f0", "m7", "soup", "hshipped", "htrain", "hshtta", "htta1"):
        if (args.work / label / "rows.jsonl").exists():
            runs[label] = load_run(args.work / label)
    thresholds = np.asarray(next(iter(runs.values()))[0]["thresholds"])
    ti = {round(float(t), 2): i for i, t in enumerate(thresholds)}
    arrays, summaries = {}, {}
    for label, (identity, rows) in runs.items():
        if len(rows) != 689:
            print(f"[analyze] skipping {label}: {len(rows)} of 689 rows (run incomplete)", flush=True)
            continue
        for config in identity["configs"]:
            arrays[config] = config_arrays(rows, config)
            summaries[config] = summarize(arrays[config], identity["thresholds"])
            summaries[config]["model"] = identity["model"]

    out = {"schema": "socratic-f0-benchmark-analysis-v1", "rules": __doc__}
    # ---- reproduction of the sealed audits
    repro = {}
    for (config, t), sealed in SEALED.items():
        if config in summaries:
            value = summaries[config]["strict_macro_dice"][ti[t]]
            repro[f"{config}@{t:.2f}"] = {"harness": value, "sealed": sealed, "difference": value - sealed}
    out["reproduction"] = repro

    # ---- per-config headline numbers
    def best(config, series):
        v = np.asarray(series)
        i = int(np.argmax(v))
        return {"value": float(v[i]), "threshold": float(thresholds[i]), "index": i}

    table = {}
    for config, s in summaries.items():
        entry = {
            "dice_030": s["strict_macro_dice"][ti[0.30]],
            "dice_035": s["strict_macro_dice"][ti[0.35]],
            "dice_020": s["strict_macro_dice"][ti[0.20]],
            "dice_best": best(config, s["strict_macro_dice"]),
            "tolerant_f1_best": {k: best(config, v["macro_f1"]) for k, v in s["tolerant"].items()},
            "tolerant_f05_best": {k: best(config, v["macro_f0_5"]) for k, v in s["tolerant"].items()},
            "dilated_dice_best": {k: best(config, v) for k, v in s["dilated_macro_dice"].items()},
            "foreground_ratio_030": {sc: s["strict_per_scroll"][sc]["foreground_ratio"][ti[0.30]] for sc in SCROLLS},
        }
        dil = [entry["dice_best"]] + list(entry["dilated_dice_best"].values())
        entry["dice_best_any_dilation"] = max(dil, key=lambda d: d["value"])
        entry["dice_best_any_dilation"]["dilation"] = int(np.argmax([d["value"] for d in dil]))
        table[config] = entry
    out["table"] = table
    out["thresholds"] = [float(t) for t in thresholds]
    out["summaries_tolerant_recall_2"] = {c: s["tolerant"]["2"]["macro_recall"] for c, s in summaries.items()}
    out["summaries_tolerant_f1"] = {c: {t: s["tolerant"][t]["macro_f1"] for t in s["tolerant"]} for c, s in summaries.items()}
    out["summaries_strict_macro_dice"] = {c: s["strict_macro_dice"] for c, s in summaries.items()}
    out["summaries_per_scroll"] = {c: s["strict_per_scroll"] for c, s in summaries.items()}

    # ---- HercUNet oracle selections
    herc = [c for c in table if c.startswith("h")]
    if herc:
        out["hercunet_oracle"] = {
            "dice": max(((c, table[c]["dice_best"]) for c in herc), key=lambda x: x[1]["value"]),
            "dice_any_dilation": max(((c, table[c]["dice_best_any_dilation"]) for c in herc), key=lambda x: x[1]["value"]),
            "tolerant_f1_2": max(((c, table[c]["tolerant_f1_best"]["2"]) for c in herc), key=lambda x: x[1]["value"]),
            "tolerant_f1_4": max(((c, table[c]["tolerant_f1_best"]["4"]) for c in herc), key=lambda x: x[1]["value"]),
        }

    # ---- paired bootstraps
    deltas = {}
    if "f0_tta" in arrays and "m7_flat" in arrays:
        deltas["f0_shipped_minus_m7_official_dice"] = paired_delta(arrays["f0_tta"], arrays["m7_flat"], ti[0.30], ti[0.20])
    if "f0_tta" in arrays and "m7_tta" in arrays:
        b = table["m7_tta"]["dice_best"]["index"]
        deltas["f0_shipped_minus_m7_tta_best_dice"] = paired_delta(arrays["f0_tta"], arrays["m7_tta"], ti[0.30], b)
    if herc and "f0_tta" in arrays:
        config, pick = out["hercunet_oracle"]["dice"]
        deltas["f0_shipped_minus_hercunet_oracle_dice"] = dict(
            paired_delta(arrays["f0_tta"], arrays[config], ti[0.30], pick["index"]), hercunet_config=config)
        config, pick = out["hercunet_oracle"]["dice_any_dilation"]
        r = pick["dilation"]
        deltas["f0_shipped_minus_hercunet_oracle_dilated_dice"] = dict(
            paired_delta(arrays["f0_tta"], arrays[config], ti[0.30], pick["index"],
                         prefix_b="strict" if r == 0 else f"dil{r}"), hercunet_config=config, dilation=r)
    out["paired_deltas"] = deltas

    # ---- registered-real rows only (their image is the public coarse volume byte for byte)
    real = {}
    for config, a in arrays.items():
        keep = np.asarray(["synthetic" not in pid for pid in a["patch_id"]])
        sa = subset(a, keep)
        s = summarize(sa, list(map(float, thresholds)))
        real[config] = {"rows": int(keep.sum()), "dice_030": s["strict_macro_dice"][ti[0.30]],
                        "dice_020": s["strict_macro_dice"][ti[0.20]],
                        "dice_best": best(config, s["strict_macro_dice"]),
                        "tolerant_f1_2_best": best(config, s["tolerant"]["2"]["macro_f1"])}
    out["registered_real_only"] = real

    # ---- in-situ fidelity rows
    path = args.insitu / "insitu_rows.jsonl"
    if path.exists():
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        fidelity = []
        for row in rows:
            pid = row["patch_id"]
            entry = {"patch_id": pid, "ct_equal_to_benchmark_image": row.get("ct_equal_to_benchmark_image")}
            for p in range(4):
                c_in = row["configs"][f"hinsitu_p{p}"]
                tp, fp, fn = (np.asarray(c_in["strict"][k]) for k in ("tp", "fp", "fn"))
                d_in = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
                w = arrays.get(f"hshipped_p{p}")
                if w is None:
                    continue
                j = list(w["patch_id"]).index(pid)
                tp, fp, fn = w["strict_tp"][j], w["strict_fp"][j], w["strict_fn"][j]
                d_w = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
                entry[f"p{p}"] = {"insitu_best": float(d_in.max()), "inwindow_best": float(d_w.max()),
                                  "insitu_030": float(d_in[ti[0.30]]), "inwindow_030": float(d_w[ti[0.30]])}
            fidelity.append(entry)
        out["insitu_fidelity"] = fidelity

    # ---- Kaggle receipts
    kaggle = {}
    for receipt in sorted(args.kaggle.glob("*/kaggle.json")):
        data = json.loads(receipt.read_text(encoding="utf-8"))
        kaggle[receipt.parent.name] = {"threshold": data["threshold"], "dilate": data["dilate"],
                                       **{k: v for k, v in data["summary"].items() if k.startswith("mean_") or k == "n_cases"}}
    out["kaggle"] = kaggle
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print("reproduction:", json.dumps({k: round(v["difference"], 9) for k, v in repro.items()}))
    print(f"{'config':18s} {'D@.20':>7s} {'D@.30':>7s} {'Dbest':>12s} {'Ddil':>14s} {'tF1@2':>12s} {'tF1@4':>12s}")
    for c, e in table.items():
        print(f"{c:18s} {e['dice_020']:7.4f} {e['dice_030']:7.4f} {e['dice_best']['value']:.4f}@{e['dice_best']['threshold']:.2f} "
              f"{e['dice_best_any_dilation']['value']:.4f}r{e['dice_best_any_dilation']['dilation']}@{e['dice_best_any_dilation']['threshold']:.2f} "
              f"{e['tolerant_f1_best']['2']['value']:.4f}@{e['tolerant_f1_best']['2']['threshold']:.2f} "
              f"{e['tolerant_f1_best']['4']['value']:.4f}@{e['tolerant_f1_best']['4']['threshold']:.2f}")
    for k, v in deltas.items():
        print(f"{k}: {v['point']:+.4f} [{v['low']:+.4f}, {v['high']:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
