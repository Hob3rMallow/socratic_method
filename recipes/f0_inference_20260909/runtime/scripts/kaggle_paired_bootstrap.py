"""Paired cluster bootstrap over per-case Kaggle metrics (metrics_per_case.csv).

Compares a candidate scoring directory with a reference scoring directory on
the same cases: per-column paired deltas of the per-case means with a cluster
bootstrap (clusters = support_anchor_chunk_zyx from the manifest when present,
else the case itself). NaN cells (e.g. topoF1_2 where no 2-cycles exist) are
dropped pairwise, matching the official NaN-skipping means.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

COLUMNS = ("leaderboard", "surface_dice", "voi_score", "voi_split", "voi_merge", "voi_total", "toposcore", "topoF1_0", "topoF1_1", "topoF1_2")


def read_cases(score_dir: Path) -> dict[str, dict[str, float]]:
    cases: dict[str, dict[str, float]] = {}
    for path in sorted(score_dir.glob("shard_*/out/metrics_per_case.csv")):
        with path.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                cases[row["id"]] = {c: float(row[c]) for c in COLUMNS if c in row}
    if not cases:
        raise ValueError(f"{score_dir}: no per-case metrics found")
    return cases


def clusters_from_manifest(manifest: Path, ids: list[str]) -> np.ndarray:
    anchors: dict[str, str] = {}
    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            anchor = row.get("support_anchor_chunk_zyx")
            anchors[str(row["patch_id"])] = ",".join(str(v) for v in anchor) if anchor else str(row["patch_id"])
    return np.asarray([anchors.get(i, i) for i in ids])


def paired(candidate: dict[str, dict[str, float]], reference: dict[str, dict[str, float]], clusters: np.ndarray, ids: list[str], *, replicates: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(clusters, return_inverse=True)
    draws = rng.integers(0, unique.size, size=(replicates, unique.size))
    cluster_weights = np.zeros((replicates, unique.size), dtype=np.int64)
    np.add.at(cluster_weights, (np.repeat(np.arange(replicates), unique.size), draws.ravel()), 1)
    weights = cluster_weights[:, inverse].astype(np.float64)
    result: dict[str, Any] = {"cases": len(ids), "clusters": int(unique.size), "replicates": replicates, "seed": seed, "columns": {}}
    for column in COLUMNS:
        a = np.asarray([candidate[i].get(column, math.nan) for i in ids])
        b = np.asarray([reference[i].get(column, math.nan) for i in ids])
        keep = ~(np.isnan(a) | np.isnan(b))
        if keep.sum() == 0:
            continue
        w = weights[:, keep]
        delta = a[keep] - b[keep]
        point = float(delta.mean())
        totals = w.sum(axis=1)
        # A resample may draw no scorable case for a sparse column (e.g. the
        # 2-cycle F1 on a dozen cases); such replicates carry no information.
        valid = totals > 0
        if valid.sum() < 2:
            continue
        boot = (w[valid] @ delta) / totals[valid]
        low, high = np.percentile(boot, [2.5, 97.5])
        result["columns"][column] = {
            "candidate_mean": float(a[keep].mean()), "reference_mean": float(b[keep].mean()),
            "delta": point, "ci95": [float(low), float(high)], "cases_used": int(keep.sum()),
            "replicates_used": int(valid.sum()),
            "probability_delta_le_zero": float(np.mean(boot <= 0.0)),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    candidate = read_cases(args.candidate)
    reference = read_cases(args.reference)
    ids = sorted(set(candidate) & set(reference))
    if len(ids) != len(candidate) or len(ids) != len(reference):
        raise ValueError(f"case sets differ: candidate {len(candidate)}, reference {len(reference)}, shared {len(ids)}")
    clusters = clusters_from_manifest(args.manifest, ids) if args.manifest else np.asarray(ids)
    result = paired(candidate, reference, clusters, ids, replicates=args.replicates, seed=args.seed)
    result.update(candidate=str(args.candidate), reference=str(args.reference))
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    for column, value in result["columns"].items():
        print(f"{column:13s} cand {value['candidate_mean']:.4f} ref {value['reference_mean']:.4f} delta {value['delta']:+.4f} CI [{value['ci95'][0]:+.4f}, {value['ci95'][1]:+.4f}] n={value['cases_used']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
