#!/usr/bin/env python
"""How fragmented each model's output is on the random-cube audit (descriptive; added after the audit).

For every drawn cube and source, the central 128^3 prediction is split into 26-connected pieces. Per compression
stratum we report the mean fill, the mean number of pieces of at least 50 voxels, and the mean share of predicted
voxels that sit in pieces under 2,000 voxels. It backs one sentence: that M7 with TTA, which has few deep voxels,
leaves broken pieces rather than sheets where the published mask fuses wraps. It was not preregistered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

SMALL_PIECE = 2000
MIN_PIECE = 50
S26 = np.ones((3, 3, 3), bool)


def mask_of(preds: Path, cube_id: str, config: str, threshold: float | None, cube_archive) -> np.ndarray:
    if config == "published":
        return cube_archive["m7_published"] > 0
    with np.load(preds / config / f"{cube_id}.npz") as archive:
        return archive["p8"].astype(np.float32) / 255.0 >= threshold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True, help="blob_audit directory (cubes/, final/cube_metrics.jsonl)")
    parser.add_argument("--preds", type=Path, required=True, help="cube_infer.py output directory")
    parser.add_argument("--source", action="append", required=True, help="name=config:threshold or name=published")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    for spec in args.source:
        name, rhs = spec.split("=", 1)
        sources[name] = ("published", None) if rhs == "published" else (rhs.split(":")[0], float(rhs.split(":")[1]))
    rows = [json.loads(line) for line in (args.audit / "final" / "cube_metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    per = {name: {} for name in sources}
    for row in rows:
        cube_id, stratum = row["cube_id"], row["stratum"]
        with np.load(args.audit / "cubes" / f"{cube_id}.npz") as cube_archive:
            for name, (config, threshold) in sources.items():
                mask = mask_of(args.preds, cube_id, config, threshold, cube_archive)
                fg = int(mask.sum())
                if fg:
                    labels, _ = ndimage.label(mask, structure=S26)
                    sizes = np.bincount(labels.ravel())[1:]
                    small = float(sizes[sizes < SMALL_PIECE].sum() / fg)
                    pieces = int((sizes >= MIN_PIECE).sum())
                else:
                    small, pieces = 0.0, 0
                per[name].setdefault(stratum, []).append((fg / mask.size, small, pieces))
    out = {"schema": "socratic-f0-benchmark-fragments-v1", "preregistered": False,
           "definition": {"connectivity": 26, "small_piece_voxels_below": SMALL_PIECE, "piece_counted_from_voxels": MIN_PIECE,
                          "volume": "central 128^3 of each drawn cube"},
           "sources": {name: {"config": sources[name][0], "threshold": sources[name][1], "per_compression_stratum": {}}
                       for name in sources}}
    for name, strata in per.items():
        for stratum, values in strata.items():
            v = np.asarray(values, dtype=np.float64)
            out["sources"][name]["per_compression_stratum"][stratum] = {
                "cubes": len(values), "mean_fill": float(v[:, 0].mean()), "mean_small_piece_share": float(v[:, 1].mean()),
                "mean_pieces": float(v[:, 2].mean())}
            print(f"{name:14s} {stratum:10s} fill {100 * v[:, 0].mean():5.1f}%  small-piece share {100 * v[:, 1].mean():5.1f}%  "
                  f"pieces {v[:, 2].mean():5.1f}")
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
