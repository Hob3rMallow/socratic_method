#!/usr/bin/env python
"""Collect every headline number for the record and the show-and-tell page into one JSON (no model runs).

Inputs are the finished work tree: analysis.json (analyze.py), Kaggle receipts (kaggle_score.py), the
PHerc0500P2 runs (patch_benchmark.py --scroll PHerc0500P2) and their Kaggle receipts, the random-cube
blob-audit summary (blob_metrics.py), the frontier (frontier.py) and the fidelity rows (insitu_check.py).
Missing inputs are recorded as null so the page can say "pending" rather than invent a number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_counts import config_arrays, load_run, summarize  # noqa: E402


def kaggle(root: Path, name: str) -> dict | None:
    path = root / name / "kaggle.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    s = data["summary"]
    return {"threshold": data["threshold"], "dilate": data["dilate"], "cases": s.get("n_cases"),
            "composite": s["mean_leaderboard"], "surface_dice": s["mean_surface_dice"],
            "voi_score": s["mean_voi_score"], "voi_split": s["mean_voi_split"], "voi_merge": s["mean_voi_merge"],
            "toposcore": s["mean_toposcore"]}


SEALED_KAGGLE = Path("D:/work/socratic_method/recipes/f0_inference_20260909/evidence/kaggle")


def sealed(name: str, set_name: str) -> dict | None:
    """Sealed exact-threshold receipts (float probabilities) for M7 at T 0.20. Stored uint8 maps can only
    threshold 0.20 to within half a quantum, so these receipts are preferred where they exist."""
    path = SEALED_KAGGLE / name / f"{set_name}_score" / "kaggle_summary.json"
    if not path.exists():
        return None
    s = json.loads(path.read_text(encoding="utf-8"))
    return {"threshold": 0.20, "dilate": 0, "cases": s.get("n_cases"), "source": f"sealed receipt {name}/{set_name}",
            "composite": s["mean_leaderboard"], "surface_dice": s["mean_surface_dice"], "voi_score": s["mean_voi_score"],
            "voi_split": s["mean_voi_split"], "voi_merge": s["mean_voi_merge"], "toposcore": s["mean_toposcore"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True, help="the benchmark work root")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    w = args.work
    analysis = json.loads((w / "analysis" / "analysis_final.json").read_text(encoding="utf-8"))
    table = analysis["table"]
    k = w / "kaggle"
    out: dict = {"schema": "socratic-f0-benchmark-numbers-v1"}

    # ---- frozen benchmark
    herc_configs = [c for c in table if c.startswith("h")]
    herc_best_dice = max(herc_configs, key=lambda c: table[c]["dice_best"]["value"])
    herc_best_dil = max(herc_configs, key=lambda c: table[c]["dice_best_any_dilation"]["value"])
    herc_best_tf1 = max(herc_configs, key=lambda c: table[c]["tolerant_f1_best"]["2"]["value"])
    herc_best_tf4 = max(herc_configs, key=lambda c: table[c]["tolerant_f1_best"]["4"]["value"])
    kaggle_herc = {name: kaggle(k, name) for name in sorted(p.name for p in k.iterdir() if p.name.startswith("h") and p.is_dir())}
    kaggle_herc = {n: v for n, v in kaggle_herc.items() if v}
    herc_kaggle_best = max(kaggle_herc.items(), key=lambda kv: kv[1]["composite"]) if kaggle_herc else (None, None)
    out["benchmark"] = {
        "f0": {"dice": table["f0_tta"]["dice_030"], "kaggle": kaggle(k, "f0_tta_t030")},
        "m7_published": {"dice": table["m7_flat"]["dice_020"],
                         "kaggle": sealed("sealed_m7_raw_calibrated_t020", "v14p2_val") or kaggle(k, "m7_flat_t020"),
                         "kaggle_rescored_uint8": kaggle(k, "m7_flat_t020")},
        "m7_tta_best": {"dice": table["m7_tta"]["dice_best"]["value"], "threshold": table["m7_tta"]["dice_best"]["threshold"],
                        "kaggle": sealed("sealed_m7_tta_calibrated_t020", "v14p2_val") or kaggle(k, "m7_tta_t020"),
                        "kaggle_rescored_uint8": kaggle(k, "m7_tta_t020"), "kaggle_t030": kaggle(k, "m7_tta_t030")},
        "hercunet_best": {
            "dice": table[herc_best_dice]["dice_best"]["value"], "dice_config": herc_best_dice,
            "dice_threshold": table[herc_best_dice]["dice_best"]["threshold"],
            "dice_any_dilation": table[herc_best_dil]["dice_best_any_dilation"]["value"],
            "dice_any_dilation_config": herc_best_dil,
            "dice_any_dilation_r": table[herc_best_dil]["dice_best_any_dilation"]["dilation"],
            "kaggle_best_name": herc_kaggle_best[0], "kaggle": herc_kaggle_best[1],
            "kaggle_all": kaggle_herc,
        },
        "soup": {"dice_030": table["soup_tta"]["dice_030"], "dice_035": table["soup_tta"]["dice_035"],
                 "kaggle": kaggle(k, "soup_tta_t030")} if "soup_tta" in table else None,
    }
    # tolerant surface F1 at 2 and 4 voxels: F0 and M7 at their declared thresholds (F0 0.30, M7 as published
    # 0.20, M7 + TTA at its best Dice threshold), HercUNet at its best threshold and configuration (oracle).
    thresholds = analysis["thresholds"]
    f1 = analysis["summaries_tolerant_f1"]

    def at(config, threshold):
        i = min(range(len(thresholds)), key=lambda j: abs(thresholds[j] - threshold))
        return {tol_: {"value": f1[config][tol_][i], "threshold": thresholds[i]} for tol_ in ("2", "4")}

    tol = {
        "f0": at("f0_tta", 0.30),
        "m7_published": at("m7_flat", 0.20),
        "m7_tta_best": at("m7_tta", table["m7_tta"]["dice_best"]["threshold"]),
        "hercunet_best": {"2": table[herc_best_tf1]["tolerant_f1_best"]["2"] | {"config": herc_best_tf1},
                          "4": table[herc_best_tf4]["tolerant_f1_best"]["4"] | {"config": herc_best_tf4}},
    }
    if "soup_tta" in f1:
        tol["soup"] = at("soup_tta", 0.30)
    out["tolerant_f1_best_threshold"] = tol
    out["paired_deltas"] = analysis["paired_deltas"]
    out["registered_real_only"] = {c: analysis["registered_real_only"][c] for c in
                                   ("f0_tta", "m7_flat", "m7_tta", "hshipped_p0", "soup_tta") if c in analysis["registered_real_only"]}
    herc_real = max((c for c in herc_configs if c in analysis["registered_real_only"]),
                    key=lambda c: analysis["registered_real_only"][c]["dice_best"]["value"])
    out["registered_real_only"]["hercunet_best"] = dict(analysis["registered_real_only"][herc_real], config=herc_real)
    out["reproduction"] = analysis["reproduction"]
    out["hercunet_passes"] = {c: table[c]["dice_best"] for c in herc_configs}
    out["insitu_fidelity"] = analysis.get("insitu_fidelity")

    # ---- PHerc0500P2 (human labels, absolute only)
    p5 = {}
    root5 = w / "full_p0500"
    for label, run, config in (("f0", "f0", "f0_tta"), ("m7_flat", "m7", "m7_flat"), ("m7_tta", "m7", "m7_tta"),
                               ("hercunet_p0", "hshipped", "hshipped_p0"), ("hercunet_p3", "hshipped", "hshipped_p3"),
                               ("soup", "soup", "soup_tta")):
        if (root5 / run / "rows.jsonl").exists():
            identity, rows = load_run(root5 / run)
            if len(rows) == 320 and config in identity["configs"]:
                s = summarize(config_arrays(rows, config), identity["thresholds"])
                d = np.asarray(s["strict_macro_dice"])
                p5[label] = {"dice_best": float(d.max()), "dice_best_threshold": float(identity["thresholds"][int(d.argmax())]),
                             "dice_030": float(d[4]), "dice_020": float(d[2])}
    for label, name in (("f0", "p0500_f0_tta_t030"), ("m7_flat", "p0500_m7_flat_t020"), ("m7_tta", "p0500_m7_tta_t020"),
                        ("hercunet_p0", "p0500_hshipped_p0_t050"), ("hercunet_p0_t030", "p0500_hshipped_p0_t030"),
                        ("hercunet_p3", "p0500_hshipped_p3_t030"), ("hercunet_tta_p3", "p0500_hshtta_p3_t030"),
                        ("soup", "p0500_soup_tta_t030")):
        rec = kaggle(k, name)
        exact = {"m7_flat": sealed("sealed_m7_raw_calibrated_t020", "p0500p2_val"),
                 "m7_tta": sealed("sealed_m7_tta_calibrated_t020", "p0500p2_val")}.get(label)
        if exact:
            p5.setdefault(label, {})["kaggle_rescored_uint8"] = rec
            rec = exact
        if rec:
            p5.setdefault(label, {})["kaggle"] = rec
    out["p0500p2"] = p5

    # ---- blob audit
    blob = json.loads((w / "blob_audit" / "final" / "summary.json").read_text(encoding="utf-8"))
    out["blob_audit"] = {name: {"overall": v["overall"], "per_compression_stratum": v["per_compression_stratum"],
                                "config": v["config"], "threshold": v["threshold"]} for name, v in blob["sources"].items()}
    plan = json.loads((w / "blob_audit" / "sample_plan.json").read_text(encoding="utf-8"))
    out["blob_audit_plan"] = {s: {"eligible": i["eligible_cubes"], "voxel_size_um": i["voxel_size_um"]}
                              for s, i in plan["scrolls"].items()}
    frag = w / "blob_audit" / "final" / "fragments.json"
    out["fragments"] = json.loads(frag.read_text(encoding="utf-8"))["sources"] if frag.exists() else None
    eq = w / "analysis" / "registered_real_s3_equality.json"
    out["registered_real_s3_equality"] = json.loads(eq.read_text(encoding="utf-8")) if eq.exists() else None
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"f0": out["benchmark"]["f0"]["dice"], "m7": out["benchmark"]["m7_published"]["dice"],
                      "herc": out["benchmark"]["hercunet_best"]["dice"], "p0500": list(p5)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
