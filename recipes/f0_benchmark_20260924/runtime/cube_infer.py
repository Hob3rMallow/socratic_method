#!/usr/bin/env python
"""Run one model family on every cached random cube (192^3 CT context -> central 128^3 probability).

Geometry is F0's shipped geometry for everyone: a 128^3 target cube inside a 32-voxel halo, one 192^3 window.
  engine    eight mirror forwards (bf16); writes <label>_flat and <label>_tta uint8 probabilities
  hercunet  in-window Jacobi passes on the same 192^3 window with their inference functions; writes <label>_p{k}
Outputs: <work>/<config>/<cube_id>.npz with p8 (uint8, 128^3).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_benchmark import EngineFamily, HercUNetFamily  # noqa: E402

HALO, CUBE = 32, 128


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("engine", "hercunet"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--norm", default="shipped")
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--tta", default="none")
    parser.add_argument("--cubes", type=Path, required=True, help="blob_audit/cubes directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep", nargs="+", required=True, help="configs to write")
    args = parser.parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if args.family == "engine":
        family = EngineFamily(args.checkpoint, args.label, device)
    else:
        family = HercUNetFamily(args.model_dir, args.label, device, args.norm, args.passes, args.tta)
    for config in args.keep:
        if config not in family.configs:
            raise SystemExit(f"unknown config {config}; available {family.configs}")
        (args.out / config).mkdir(parents=True, exist_ok=True)
    (args.out / f"identity_{args.label}.json").write_text(json.dumps(family.identity, indent=1) + "\n", encoding="utf-8")
    paths = sorted(args.cubes.glob("*.npz"))
    for index, path in enumerate(paths, 1):
        targets = [args.out / c / path.name for c in args.keep]
        if all(t.exists() for t in targets):
            continue
        with np.load(path) as archive:
            context = np.asarray(archive["ct_context"])
        if context.shape != (CUBE + 2 * HALO,) * 3:
            raise SystemExit(f"{path.name}: context shape {context.shape}")
        fields = family.predict(context)
        for config, target in zip(args.keep, targets):
            field = fields[config][HALO:HALO + CUBE, HALO:HALO + CUBE, HALO:HALO + CUBE]
            p8 = field.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            np.savez_compressed(target, p8=p8)
        if index % 16 == 0 or index == len(paths):
            print(f"[cube-infer] {args.label}: {index}/{len(paths)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
