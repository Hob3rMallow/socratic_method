#!/usr/bin/env python3
"""Measure parameter displacement from released M7 for existing checkpoints.

The free replacement for a trust-region training arm: if the wide-corpus run's
relative-L2 displacement is far outside the shipped 16x ball
(0.0027535421730275947), the shipped projection would have strangled the
best-ever run -- no 5.75h arm needed to conclude it. Radii ladder from the
v13..v31 lineage. CPU-only, seconds per checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "output/crossres_data/m7_xr_wide15_jackpot_recipe_20260901"
CONFIG = ROOT / "crossres_pred/configs/unified_ladder_20260902.json"

RADII = {
    "1x_v13": 0.00017209638581422467,
    "4x_v15": 0.0006883855432568987,
    "16x_v16_to_v31_shipped": 0.0027535421730275947,
    "32x": 0.0055070843460551894,
}


def relative_l2(student_model, reference_model) -> float:
    import torch

    delta_sq = 0.0
    reference_sq = 0.0
    with torch.no_grad():
        for (name_s, p_s), (name_r, p_r) in zip(
            student_model.named_parameters(), reference_model.named_parameters()
        ):
            if name_s != name_r:
                raise ValueError(f"parameter order mismatch: {name_s} vs {name_r}")
            delta_sq += float(
                (p_s.double() - p_r.double()).pow(2).sum()
            )
            reference_sq += float(p_r.double().pow(2).sum())
    return math.sqrt(delta_sq) / math.sqrt(reference_sq)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    arguments = parser.parse_args()
    run_dir = arguments.run_dir.resolve()
    checkpoints = sorted(run_dir.glob("checkpoint_milestone_*.pt"))
    if not checkpoints:
        raise SystemExit(f"{run_dir}: no milestone checkpoints")

    import torch

    from crossres_pred.voxel.inference import load_voxel_checkpoint
    from crossres_pred.voxel.model import initialize_from_m7

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    m7_path = ROOT / config["initialization"]["checkpoint"]
    device = torch.device("cpu")

    reference_model = None
    rows = []
    for checkpoint in checkpoints:
        student, payload = load_voxel_checkpoint(checkpoint, device=device)
        if reference_model is None:
            reference_model = type(student)(student.config)
            initialize_from_m7(reference_model, m7_path)
        displacement = relative_l2(student, reference_model)
        samples = int(checkpoint.stem.rsplit("_", 1)[1])
        rows.append(
            {
                "checkpoint": checkpoint.name,
                "samples": samples,
                "relative_l2_vs_m7": displacement,
                "ratio_to_shipped_16x_ball": displacement
                / RADII["16x_v16_to_v31_shipped"],
                "cumulative_samples": payload.get("cumulative_samples"),
            }
        )
        print(
            f"{checkpoint.name}: relative-L2 {displacement:.6f} "
            f"({displacement / RADII['16x_v16_to_v31_shipped']:.1f}x the shipped "
            f"16x ball)",
            flush=True,
        )

    output = run_dir / "ladder_eval"
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "crossres-trust-displacement-measurement-v1",
        "contract": "projected-global-relative-l2-to-m7-v1 (measurement only)",
        "run_dir": str(run_dir),
        "m7_checkpoint": str(m7_path),
        "radii": RADII,
        "checkpoints": rows,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (output / "trust_displacement.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"-> {output / 'trust_displacement.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
