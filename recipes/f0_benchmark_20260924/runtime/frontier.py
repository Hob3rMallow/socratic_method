#!/usr/bin/env python
"""Threshold-free blob frontier on the frozen benchmark: recall versus solid-mass fraction, per threshold.

For each stored configuration (uint8 probability maps from patch_benchmark.py) and each of the 17 audit
thresholds, measure on every row's central 128^3 window (the ScrollFiesta deploy window of a 192^3 patch):
  * the canonical erosion interior: predicted voxels more than 2 six-connected steps from background
    (crossres_pred.voxel.scrollfiesta_metrics, GARBAGE_ERODE_R = 2), as a share of predicted voxels;
  * the canonical thick predicate (interior share >= 0.50 with >= 2,000 interior voxels);
  * voxels at L1 depth >= 5 (inside a solid mass at least ~9 voxels thick) as a share of predicted voxels.
Recall is taken from the exact counts (tolerant surface recall at 2 voxels and strict recall), so each model
traces a curve; a model that dominates sits at higher recall for the same solid-mass share at every threshold.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy import ndimage

THRESHOLDS = [float(v) for v in np.linspace(0.10, 0.90, 17)]
WINDOW = slice(32, 160)


def one(args):
    path, = args
    with np.load(path) as archive:
        p8 = archive["p8"][WINDOW, WINDOW, WINDOW]
    out = []
    for t in THRESHOLDS:
        k = int(np.ceil(255.0 * t + 0.5 - 1e-9))
        mask = p8 >= k
        fg = int(mask.sum())
        if fg == 0:
            out.append((0, 0, 0))
            continue
        padded = np.pad(mask, 1)
        depth = ndimage.distance_transform_cdt(padded, metric="taxicab")[1:-1, 1:-1, 1:-1]
        out.append((fg, int((depth > 2).sum()), int((depth >= 5).sum())))
    return path.stem, out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probs", type=Path, nargs="+", required=True, help="probs/<config> directories")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()
    result = {"schema": "socratic-f0-benchmark-frontier-v1", "thresholds": THRESHOLDS, "window": "central 128^3",
              "configs": {}}
    with mp.Pool(args.workers) as pool:
        for directory in args.probs:
            files = sorted(directory.glob("*.npz"))
            rows = dict(pool.map(one, [(f,) for f in files], chunksize=4))
            fg = np.asarray([[r[0] for r in rows[f.stem]] for f in files], dtype=np.int64)
            interior = np.asarray([[r[1] for r in rows[f.stem]] for f in files], dtype=np.int64)
            deep = np.asarray([[r[2] for r in rows[f.stem]] for f in files], dtype=np.int64)
            share = np.where(fg > 0, interior / np.maximum(fg, 1), 0.0)
            thick = (share >= 0.5) & (interior >= 2000)
            result["configs"][directory.name] = {
                "rows": len(files),
                "pooled_interior_share": (interior.sum(0) / np.maximum(1, fg.sum(0))).tolist(),
                "pooled_depth_ge5_share": (deep.sum(0) / np.maximum(1, fg.sum(0))).tolist(),
                "mean_interior_share": share.mean(0).tolist(),
                "thick_window_rate": thick.mean(0).tolist(),
                "mean_fill": (fg / 128 ** 3).mean(0).tolist(),
                "per_row_patch_ids": [f.stem for f in files],
                "per_row_interior": interior.tolist(),
                "per_row_fg": fg.tolist(),
            }
            print(f"[frontier] {directory.name}: rows {len(files)} thick@T " +
                  " ".join(f"{t:.2f}:{r:.2f}" for t, r in zip(THRESHOLDS, thick.mean(0))), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
