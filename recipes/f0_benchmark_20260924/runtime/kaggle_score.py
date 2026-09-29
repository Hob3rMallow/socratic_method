#!/usr/bin/env python
"""Kaggle surface-detection metric (0.35 SurfaceDice@2 + 0.35 VOI + 0.30 TopoScore) for any stored config.

Thresholds the uint8 probability maps written by patch_benchmark.py (p8 >= ceil(255*T + 0.5), i.e. p >= T to
within 1/510), optionally dilates the mask by r six-connected steps, writes zlib TIFF masks named <patch_id>.tif
and scores them with the SAME metric binary, ground-truth directory and sharded runner the sealed F0 evidence
used (vesuvius-c/crossres_pred/scripts/evaluate_ladder_arm._score_kaggle_sharded -> metric.exe, tau 2.0, VOI
alpha 0.3, ignore label 2).
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import math
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

VC = Path("D:/work/vesuvius-c")
sys.path.insert(0, str(VC / "crossres_pred" / "scripts"))
from evaluate_ladder_arm import _score_kaggle_sharded  # noqa: E402

METRIC_EXE = VC / "metric-windows/build/metric/Release/metric.exe"
GT_ROOT = VC / "output/crossres_data/unified_ladder_20260902/kaggle_gt"


def threshold_u8(threshold: float) -> int:
    return int(math.ceil(255.0 * threshold + 0.5 - 1e-9))


def dump_one(src: Path, dst: Path, k: int, dilate: int) -> None:
    if dst.exists():
        return
    with np.load(src) as archive:
        mask = archive["p8"] >= k
    if dilate:
        mask = ndimage.binary_dilation(mask, structure=ndimage.generate_binary_structure(3, 1), iterations=dilate)
    tmp = dst.with_name(dst.stem + ".partial.tif")
    tifffile.imwrite(tmp, mask.astype(np.uint8), compression="zlib")
    tmp.replace(dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probs", type=Path, required=True, help="directory of <patch_id>.npz uint8 probabilities")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--dilate", type=int, default=0)
    parser.add_argument("--set", default="v14p2_val")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=24)
    parser.add_argument("--only", type=Path, help="optional text file of patch ids to restrict to")
    args = parser.parse_args()
    k = threshold_u8(args.threshold)
    pred_dir = args.out / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(args.probs.glob("*.npz"))
    if args.only:
        keep = {l.strip() for l in args.only.read_text(encoding="utf-8").splitlines() if l.strip()}
        sources = [s for s in sources if s.stem in keep]
    gt_dir = GT_ROOT / args.set
    with futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda s: dump_one(s, pred_dir / f"{s.stem}.tif", k, args.dilate), sources))
    summary = _score_kaggle_sharded(METRIC_EXE, gt_dir, pred_dir, args.out / "score", shards=args.shards)
    receipt = {
        "schema": "socratic-f0-benchmark-kaggle-v1",
        "probs": str(args.probs),
        "threshold": args.threshold,
        "threshold_u8": k,
        "dilate": args.dilate,
        "set": args.set,
        "gt_dir": str(gt_dir),
        "metric_exe": str(METRIC_EXE),
        "cases": len(sources),
        "summary": summary,
    }
    (args.out / "kaggle.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k2: v for k2, v in summary.items() if k2.startswith("mean_") or k2 == "n_cases"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
