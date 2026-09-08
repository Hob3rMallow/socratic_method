"""Inspect development-only curved probability corridors using existing caches."""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
from scipy import ndimage

import probability_continuity as continuity
import run_f0_repair_context as context
from render_f0_repair_review import assemble, overlay, panel_row

OUTPUT = context.ROOT / "output/crossres_data/f0_continuity_20260907"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floor", type=float, required=True)
    args = parser.parse_args()
    request = context.read(context.OUTPUT / "request.json")
    spec = request["blocks"]["development"]
    lower, ids = spec["origin_zyx"], spec["target_cube_ids"]
    out = OUTPUT / f"development_probe_p{args.floor:.2f}"
    if out.exists():
        raise FileExistsError("preserve existing diagnostic output")
    base = assemble(context.OUTPUT / "repairs/development/extended_safe/cubes_PRED", ids, lower) != 0
    probability = assemble(context.OUTPUT / "inference_development/probability", ids, lower)
    raw_ct = assemble(context.SOURCE / "cubes_RAW", ids, lower)
    lo, hi = np.percentile(raw_ct[raw_ct > 0], [1, 99.5])
    ct = np.clip((raw_ct.astype(np.float32) - lo) * (255 / (hi - lo)), 0, 255).astype(np.uint8)
    out.mkdir(parents=True)
    (out / "assets").mkdir()
    rows, total = [], Counter()
    policy = continuity.ProposalPolicy(probability_floor=args.floor)
    for z in range(16, 369, 16):
        proposals, census = continuity.find_corridors(base[z], probability[z],
                                                     (3287-lower[1], 3784-lower[2]), policy)
        neighbors = [ndimage.distance_transform_edt(~base[z+dz]) for dz in (-8, -4, -2, -1, 1, 2, 4, 8)]
        total.update(census)
        labels, _ = ndimage.label(base[z], continuity.EIGHT)
        for row in proposals:
            path = np.array(row["path_yx"])
            row.update(continuity.evidence_features(path, probability[z], ct[z], neighbors))
            paint = continuity.paint_corridor(path, probability[z], args.floor)
            touched = labels[ndimage.binary_dilation(paint & ~base[z], continuity.EIGHT)]
            row["paint_contact_owners"] = np.unique(touched[touched > 0]).tolist()
            row["z_world"] = z + lower[0]
            row["core"] = 128 <= z < 256 and ((path.mean(axis=0) >= 128) & (path.mean(axis=0) < 256)).all().item()
            center = path.mean(axis=0).astype(int)
            y0, x0 = np.maximum(0, np.minimum(288, center-48))
            window = (slice(y0, y0+96), slice(x0, x0+96))
            p = np.clip(probability[z][window].astype(float) / .5, 0, 1)
            from PIL import Image
            p_image = Image.fromarray(np.stack((p*255, np.sqrt(p)*185, (1-p)*110), -1).astype(np.uint8))
            picture = panel_row([overlay(ct[z][window]), overlay(ct[z][window], base[z][window]),
                                 overlay(ct[z][window], base[z][window], (paint & ~base[z])[window]), p_image],
                                ["CT", "Current filler", "PROPOSAL only", "Probability 0–.5"], scale=2)
            row["image"] = f"assets/proposal_{len(rows):04d}.png"
            row["proposal_id"] = len(rows)
            picture.save(out / row["image"])
            rows.append(row)
        print(f"z={z+lower[0]}: {len(proposals)} proposed corridors; {sum(r['length_px']>21 for r in proposals)} longer than 21px", flush=True)
    context.atomic_json(out / "proposals.json", {"schema": "f0-continuity-development-probe-v1",
                        "probability_floor": args.floor, "assessment_planes": list(range(16,369,16)),
                        "census": dict(total), "proposals": rows, "mask_writes": False,
                        "warning": "Unaccepted diagnostic proposals; not validated repairs or a chosen profile."})
    print(f"Wrote {len(rows)} diagnostic cards to {out}; no prediction masks changed.", flush=True)


if __name__ == "__main__":
    main()
