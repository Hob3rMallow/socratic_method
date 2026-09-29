#!/usr/bin/env python
"""Pre-registered random-cube sample for the published-M7 blob audit (fixed before any cube is read).

Rule, stated here and not changed after looking at data:
  * Scrolls: eight scrolls that carry an official published M7 L0 surface prediction
    (<volume>-surface-20260413222639-surface-m7-L0-th0.2.zarr), chosen to be unseen by F0's training data:
    PHerc1447 (F0's blind gate scroll), PHerc0814 (a validation scroll, never trained on), and six scrolls
    no model of ours has seen: PHerc0125, PHerc0191, PHerc0343, PHerc0800, PHerc1218, PHerc0826.
  * Unit: a 128^3 cube on the CT's own 128-voxel chunk grid.
  * Eligible: the cube AND a 32-voxel halo around it lie inside the scroll, judged at pyramid level 5
    (32x): every level-5 voxel covering the 192^3 context is nonzero (the volumes are masked to 0 outside
    the scroll).
  * Draw: 32 cubes per scroll, uniformly without replacement, numpy default_rng seeded from
    SeedSequence(20260924).spawn(8) in the scroll order above.
For every drawn cube this writes a local cache: the 192^3 CT context (uint8) and the 128^3 published M7 mask
(the organisers' own artefact, read unchanged), so the metric and inference steps never depend on S3 again.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import s3fs
import zarr

BUCKET = "vesuvius-challenge-open-data"
SCROLLS = ["PHerc1447", "PHerc0814", "PHerc0125", "PHerc0191", "PHerc0343", "PHerc0800", "PHerc1218", "PHerc0826"]
PER_SCROLL = 32
SEED = 20260924
CUBE, HALO, LEVEL = 128, 32, 5


def registry(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {s["id"]: s for s in data["scrolls"]}


def open_array(fs: s3fs.S3FileSystem, uri: str, level: int):
    root = uri.replace("s3://", "")
    return zarr.open(s3fs.S3Map(root=f"{root}/{level}", s3=fs, check=False), mode="r")


def eligible_origins(ct5: np.ndarray, shape0: tuple[int, int, int]) -> np.ndarray:
    """L0 cube origins (multiples of 128) whose 192^3 context is fully nonzero at level 5."""
    scale = 2 ** LEVEL
    nz = ct5 > 0
    origins = []
    grid = [range(0, (s - CUBE) // CUBE * CUBE + 1, CUBE) for s in shape0]
    # context [o - HALO, o + CUBE + HALO) in L0 -> level-5 index range
    for z in grid[0]:
        z0, z1 = z - HALO, z + CUBE + HALO
        if z0 < 0 or z1 > shape0[0]:
            continue
        zs = slice(z0 // scale, -(-z1 // scale))
        slab = nz[zs]
        if not slab.size:
            continue
        slab_all = slab.all(axis=0)
        for y in grid[1]:
            y0, y1 = y - HALO, y + CUBE + HALO
            if y0 < 0 or y1 > shape0[1]:
                continue
            ys = slice(y0 // scale, -(-y1 // scale))
            row = slab_all[ys]
            if not row.size:
                continue
            row_all = row.all(axis=0)
            for x in grid[2]:
                x0, x1 = x - HALO, x + CUBE + HALO
                if x0 < 0 or x1 > shape0[2]:
                    continue
                xs = slice(x0 // scale, -(-x1 // scale))
                if row_all[xs].size and row_all[xs].all():
                    origins.append((z, y, x))
    return np.asarray(origins, dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, default=Path("D:/work/socratic_method/recipes/f0/scrolls.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fs = s3fs.S3FileSystem(anon=True)
    reg = registry(args.registry)
    seeds = np.random.SeedSequence(SEED).spawn(len(SCROLLS))
    plan_path = args.out / "sample_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else {
        "schema": "socratic-m7-blob-audit-sample-v1", "rule": __doc__, "seed": SEED, "per_scroll": PER_SCROLL,
        "cube": CUBE, "halo": HALO, "eligibility_level": LEVEL, "scrolls": {}}
    for scroll, seed in zip(SCROLLS, seeds):
        entry = reg[scroll]
        if scroll not in plan["scrolls"]:
            t0 = time.time()
            ct5 = np.asarray(open_array(fs, entry["volume_uri"], LEVEL)[:])
            shape0 = tuple(int(s) for s in open_array(fs, entry["volume_uri"], 0).shape)
            origins = eligible_origins(ct5, shape0)
            rng = np.random.default_rng(seed)
            pick = rng.choice(len(origins), size=PER_SCROLL, replace=False)
            plan["scrolls"][scroll] = {
                "volume_uri": entry["volume_uri"],
                "prediction_uri": entry["surface_prediction_uri"],
                "voxel_size_um": entry["voxel_size_um"],
                "shape_l0_zyx": list(shape0),
                "eligible_cubes": int(len(origins)),
                "drawn_origins_zyx": origins[np.sort(pick)].tolist(),
            }
            plan_path.write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
            print(f"[sample] {scroll}: {len(origins)} eligible cubes, drew {PER_SCROLL} ({time.time() - t0:.0f}s)", flush=True)
        info = plan["scrolls"][scroll]
        ct = open_array(fs, info["volume_uri"], 0)
        m7 = open_array(fs, info["prediction_uri"], 0)
        if tuple(m7.shape) != tuple(ct.shape):
            raise SystemExit(f"{scroll}: prediction shape {m7.shape} != CT shape {ct.shape}")
        for z, y, x in info["drawn_origins_zyx"]:
            cube_id = f"{scroll}_z{z:05d}_y{y:05d}_x{x:05d}"
            path = args.out / "cubes" / f"{cube_id}.npz"
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            context = np.asarray(ct[z - HALO:z + CUBE + HALO, y - HALO:y + CUBE + HALO, x - HALO:x + CUBE + HALO])
            published = np.asarray(m7[z:z + CUBE, y:y + CUBE, x:x + CUBE])
            temporary = path.with_name(path.stem + ".partial.npz")
            np.savez_compressed(temporary, ct_context=context, m7_published=published,
                                origin_zyx=np.asarray([z, y, x]), halo=np.asarray(HALO))
            temporary.replace(path)
            print(f"[sample] cached {cube_id} m7_fg={float((published > 0).mean()):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
