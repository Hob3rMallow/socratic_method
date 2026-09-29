#!/usr/bin/env python
"""Fidelity check: HercUNet's own CLI on the live scroll versus the in-window harness, on the same rows.

For registered-real benchmark rows (their image is byte-identical to the public coarse volume at origin_zyx),
run ``hercunet infer single-instance`` exactly as documented (4 passes, overlap 0.25, Gaussian blend,
half-stride shift between passes, TTA off, checkpoint_best) over the patch box expanded by --margin voxels,
reading the exact coarse volume through ``--local-vol`` so their catalog cannot silently choose a finer scan.
Then crop every pass to the patch and score it with patch_benchmark.RowScorer against the same target.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_benchmark import RowScorer, load_rows  # noqa: E402

VOLUMES = {
    "PHerc0814": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0814/volumes/20250804134230-9.362um-1.2m-113keV-masked.zarr",
    "PHerc1451": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1451/volumes/20250521151225-8.640um-1.2m-116keV-masked.zarr",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--patch-ids", nargs="+", required=True)
    parser.add_argument("--margin", type=int, default=96)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    rows = {r["patch_id"]: r for r in load_rows(args.manifest)}
    args.work.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    results_path = args.work / "insitu_rows.jsonl"
    done = set()
    if results_path.exists():
        done = {json.loads(l)["patch_id"] for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    for pid in args.patch_ids:
        if pid in done:
            continue
        row = rows[pid]
        if row["scroll"] not in VOLUMES or "synthetic" in row["record"]:
            raise SystemExit(f"{pid}: not a registered-real row")
        z, y, x = row["origin_zyx"]
        m = args.margin
        region = f"{z - m}:{z + 192 + m},{y - m}:{y + 192 + m},{x - m}:{x + 192 + m}"
        prefix = args.work / pid
        command = [
            sys.executable, "-m", "hercunet", "infer", "single-instance",
            "--model", str(args.model_dir),
            "--scroll", row["scroll"],
            "--local-vol", VOLUMES[row["scroll"]],
            "--region", region,
            "--out", str(prefix),
            "--passes", str(args.passes),
            "--keep-buffers", "--finalise", "no", "--gpus", "0",
        ]
        print("[insitu]", " ".join(command), flush=True)
        t0 = time.time()
        if not all((args.work / f"{pid}_pass{p}.zarr.done" / "_complete").exists() for p in range(args.passes)):
            subprocess.run(command, check=True)
        elapsed = time.time() - t0
        with np.load(row["path"], allow_pickle=False) as archive:
            image = np.asarray(archive["image"])
            target = np.asarray(archive["target_u8"])
        scorer = RowScorer(target, device)
        record = {"patch_id": pid, "scroll": row["scroll"], "record": row["record"], "cluster": row["cluster"],
                  "region": region, "seconds": elapsed, "configs": {}}
        for p in range(args.passes):
            arr = zarr.open_group(str(args.work / f"{pid}_pass{p}.zarr"), mode="r")["0"]
            crop = np.asarray(arr[z:z + 192, y:y + 192, x:x + 192])
            record["configs"][f"hinsitu_p{p}"] = scorer.measure(torch.from_numpy(crop.astype(np.float32) / 255.0).to(device))
            np.savez_compressed(args.work / f"{pid}_pass{p}_crop.npz", p8=crop)
        # sanity: the CT they read equals the benchmark image
        from hercunet.data import ZarrSegment, parse_voxel_um
        seg = ZarrSegment(VOLUMES[row["scroll"]], parse_voxel_um(VOLUMES[row["scroll"]]))
        block, _ = seg.read_window(0, z, z + 192, y, y + 192, x, x + 192)
        record["ct_equal_to_benchmark_image"] = bool(np.array_equal(np.asarray(block), image))
        with results_path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(record) + "\n")
        print(f"[insitu] {pid}: {elapsed:.0f}s ct_equal={record['ct_equal_to_benchmark_image']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
