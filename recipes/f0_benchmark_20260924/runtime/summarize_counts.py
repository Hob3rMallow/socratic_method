#!/usr/bin/env python
"""Aggregate patch_benchmark.py row counts into per-configuration metric sweeps.

Macro Dice follows the frozen harness: per-scroll pooled Dice (sum TP/FP/FN over the scroll's rows), then the
mean over PHerc0814 and PHerc1451. Tolerant surface metrics follow the audit's pooled precision/recall; F0.5 is
the audit's summary statistic and F1 (surface Dice) is added. Paired deltas use a cluster bootstrap over
support_anchor_chunk_zyx within scroll (B = 2000, seed 20260908, the family preregistration's design).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SCROLLS = ("PHerc0814", "PHerc1451")


def scrolls_of(arrays: dict[str, np.ndarray]) -> tuple[str, ...]:
    present = tuple(sorted(set(arrays["scroll"].tolist())))
    return SCROLLS if set(present) == set(SCROLLS) else present


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    identity = json.loads((run_dir / "identity.json").read_text(encoding="utf-8"))
    rows = []
    with (run_dir / "rows.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return identity, rows


def config_arrays(rows: list[dict], config: str) -> dict[str, np.ndarray]:
    """Per-row count arrays for one configuration (rows in manifest order)."""
    out: dict[str, list] = defaultdict(list)
    for row in rows:
        c = row["configs"][config]
        out["scroll"].append(row["scroll"])
        out["cluster"].append(f"{row['scroll']}:{tuple(row['cluster'] or ())}")
        out["patch_id"].append(row["patch_id"])
        out["known"].append(c["known"])
        out["positive"].append(c["positive"])
        for key in ("tp", "fp", "fn"):
            out[f"strict_{key}"].append(c["strict"][key])
        for tol, t in c["tolerant"].items():
            for key in ("matched_prediction", "predicted", "matched_truth"):
                out[f"tol{tol}_{key}"].append(t[key])
            out[f"tol{tol}_truth"].append(t["truth"])
        for r, d in c["dilated"].items():
            for key in ("tp", "fp", "fn"):
                out[f"dil{r}_{key}"].append(d[key])
    return {k: np.asarray(v) for k, v in out.items()}


def dice(tp, fp, fn):
    return (2.0 * tp) / np.maximum(1, 2 * tp + fp + fn)


def macro_from(arrays: dict[str, np.ndarray], prefix: str, weights: np.ndarray | None = None) -> np.ndarray:
    """Macro (mean over scrolls) of per-scroll pooled Dice for strict_/dilR_ prefixes -> (17,)."""
    per = []
    for scroll in scrolls_of(arrays):
        mask = arrays["scroll"] == scroll
        w = np.ones(mask.sum()) if weights is None else weights[mask]
        tp = (arrays[f"{prefix}_tp"][mask] * w[:, None]).sum(0)
        fp = (arrays[f"{prefix}_fp"][mask] * w[:, None]).sum(0)
        fn = (arrays[f"{prefix}_fn"][mask] * w[:, None]).sum(0)
        per.append(dice(tp, fp, fn))
    return np.mean(per, axis=0)


def tolerant_from(arrays: dict[str, np.ndarray], tol: str, weights: np.ndarray | None = None) -> dict[str, np.ndarray]:
    precision, recall = [], []
    for scroll in scrolls_of(arrays):
        mask = arrays["scroll"] == scroll
        w = np.ones(mask.sum()) if weights is None else weights[mask]
        mp = (arrays[f"tol{tol}_matched_prediction"][mask] * w[:, None]).sum(0)
        pr = (arrays[f"tol{tol}_predicted"][mask] * w[:, None]).sum(0)
        mt = (arrays[f"tol{tol}_matched_truth"][mask] * w[:, None]).sum(0)
        tr = (arrays[f"tol{tol}_truth"][mask] * w).sum()
        precision.append(mp / np.maximum(1, pr))
        recall.append(mt / max(1, tr))
    precision, recall = np.asarray(precision), np.asarray(recall)
    f05 = 1.25 * precision * recall / np.maximum(0.25 * precision + recall, 1e-12)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "macro_precision": precision.mean(0),
        "macro_recall": recall.mean(0),
        "macro_f0_5": f05.mean(0),
        "macro_f1": f1.mean(0),
    }


def summarize(arrays: dict[str, np.ndarray], thresholds: list[float]) -> dict:
    result = {"rows": int(len(arrays["scroll"])), "thresholds": thresholds}
    result["strict_macro_dice"] = macro_from(arrays, "strict").tolist()
    per_scroll = {}
    for scroll in scrolls_of(arrays):
        mask = arrays["scroll"] == scroll
        tp = arrays["strict_tp"][mask].sum(0)
        fp = arrays["strict_fp"][mask].sum(0)
        fn = arrays["strict_fn"][mask].sum(0)
        pos = arrays["positive"][mask].sum()
        per_scroll[scroll] = {
            "rows": int(mask.sum()),
            "dice": dice(tp, fp, fn).tolist(),
            "precision": (tp / np.maximum(1, tp + fp)).tolist(),
            "recall": (tp / np.maximum(1, tp + fn)).tolist(),
            "foreground_ratio": ((tp + fp) / max(1, pos)).tolist(),
        }
    result["strict_per_scroll"] = per_scroll
    result["tolerant"] = {}
    for key in sorted({k.split("_")[0][3:] for k in arrays if k.startswith("tol")}):
        result["tolerant"][key] = {k: v.tolist() for k, v in tolerant_from(arrays, key).items()}
    result["dilated_macro_dice"] = {}
    for key in sorted({k.split("_")[0][3:] for k in arrays if k.startswith("dil")}):
        result["dilated_macro_dice"][key] = macro_from(arrays, f"dil{key}").tolist()
    return result


def cluster_weights(arrays: dict[str, np.ndarray], rng: np.random.Generator, draws: int) -> np.ndarray:
    """(draws, rows) multiplicities: clusters resampled with replacement within each scroll."""
    rows = len(arrays["scroll"])
    weights = np.zeros((draws, rows))
    for scroll in scrolls_of(arrays):
        idx = np.flatnonzero(arrays["scroll"] == scroll)
        clusters = arrays["cluster"][idx]
        unique, inverse = np.unique(clusters, return_inverse=True)
        members = [idx[inverse == c] for c in range(len(unique))]
        for b in range(draws):
            picks = rng.integers(0, len(unique), len(unique))
            counts = np.bincount(picks, minlength=len(unique))
            for c, n in enumerate(counts):
                if n:
                    weights[b, members[c]] += n
    return weights


def paired_delta(a: dict[str, np.ndarray], b: dict[str, np.ndarray], ia: int, ib: int, prefix_a="strict",
                 prefix_b="strict", draws=2000, seed=20260908) -> dict:
    """macro(a at threshold index ia) - macro(b at threshold index ib), paired cluster bootstrap."""
    if list(a["patch_id"]) != list(b["patch_id"]):
        raise ValueError("paired runs must cover identical rows in identical order")
    rng = np.random.default_rng(seed)
    w = cluster_weights(a, rng, draws)
    point = macro_from(a, prefix_a)[ia] - macro_from(b, prefix_b)[ib]
    boot = np.asarray([macro_from(a, prefix_a, w[k])[ia] - macro_from(b, prefix_b, w[k])[ib] for k in range(draws)])
    return {
        "point": float(point),
        "low": float(np.percentile(boot, 2.5)),
        "high": float(np.percentile(boot, 97.5)),
        "probability_delta_le_zero": float(np.mean(boot <= 0)),
        "draws": draws,
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories written by patch_benchmark.py")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = {"schema": "socratic-f0-benchmark-summary-v1", "configs": {}}
    for run in args.runs:
        identity, rows = load_run(run)
        for config in identity["configs"]:
            arrays = config_arrays(rows, config)
            entry = summarize(arrays, identity["thresholds"])
            entry["model"] = identity["model"]
            summary["configs"][config] = entry
            best = int(np.argmax(entry["strict_macro_dice"]))
            print(f"{config:28s} rows {entry['rows']:4d}  T.30 {entry['strict_macro_dice'][4]:.4f}  "
                  f"T.35 {entry['strict_macro_dice'][5]:.4f}  best {entry['strict_macro_dice'][best]:.4f}"
                  f"@{identity['thresholds'][best]:.2f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
