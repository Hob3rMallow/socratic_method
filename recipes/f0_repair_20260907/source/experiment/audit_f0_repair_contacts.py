"""Independent contact/chain diagnostics for the selected additive repairs.

These checks detect detached paint and multi-component stroke contacts. They
do not identify anatomical sheet identity or certify all cross-wrap topology.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

import run_f0_repair_context as context
from render_f0_repair_review import FONT, assemble, overlay, panel_row

EIGHT = np.ones((3, 3), dtype=bool)


def plane_contacts(raw: np.ndarray, filled: np.ndarray) -> dict:
    if (raw & ~filled).any():
        raise ValueError("raw foreground erased")
    labels, _ = ndimage.label(raw, structure=EIGHT)
    added = filled & ~raw
    strokes, n_strokes = ndimage.label(added, structure=EIGHT)
    objects = ndimage.find_objects(strokes)
    rows = []
    for number, bounds in enumerate(objects, 1):
        if bounds is None:
            continue
        window = tuple(slice(max(0, s.start - 1), min(raw.shape[i], s.stop + 1)) for i, s in enumerate(bounds))
        stroke = strokes[window] == number
        contact = labels[window][ndimage.binary_dilation(stroke, structure=EIGHT)]
        touched = np.unique(contact[contact > 0]).tolist()
        rows.append({"stroke": number, "added_pixels": int(stroke.sum()), "touched_raw_components": touched,
                     "bounds_yx": [[s.start, s.stop] for s in bounds]})
    return {"added_strokes": n_strokes, "detached_strokes": sum(not r["touched_raw_components"] for r in rows),
            "multiway_strokes": sum(len(r["touched_raw_components"]) > 2 for r in rows), "strokes": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("block", choices=("development", "control"))
    args = parser.parse_args()
    request = context.read(context.OUTPUT / "request.json")
    block = request["blocks"][args.block]
    lower, ids = block["origin_zyx"], block["target_cube_ids"]
    base = context.OUTPUT / "repairs" / args.block
    raw = assemble(context.OUTPUT / f"inference_{args.block}/cubes_PRED", ids, lower) != 0
    before = assemble(base / "report_baseline/cubes_PRED", ids, lower) != 0
    after = assemble(base / "extended_safe/cubes_PRED", ids, lower) != 0
    ct_raw = assemble(context.SOURCE / "cubes_RAW", ids, lower)
    lo, hi = np.percentile(ct_raw[ct_raw > 0], [1, 99.5])
    ct = np.clip((ct_raw.astype(np.float32) - lo) * (255 / max(1, hi - lo)), 0, 255).astype(np.uint8)
    output = context.OUTPUT / "contact_audit" / args.block
    if output.exists():
        raise FileExistsError(f"preserve existing contact audit: {output}")
    output.mkdir(parents=True)
    plane_rows = []
    for z in range(raw.shape[0]):
        row = plane_contacts(raw[z], after[z])
        row["z_world"] = z + lower[0]
        plane_rows.append(row)
    added = after & ~raw
    labels, count = ndimage.label(added, structure=np.ones((3, 3, 3), dtype=bool))
    objects = ndimage.find_objects(labels)
    chains = []
    for number, bounds in enumerate(objects, 1):
        if bounds is None:
            continue
        local = np.argwhere(labels[bounds] == number)
        starts = np.array([s.start for s in bounds])
        points = local + starts
        world = points + np.array(lower)
        radius = np.hypot(world[:, 1] - 3287, world[:, 2] - 3784)
        point = points[len(points) // 2]
        chains.append({"component": number, "voxels": len(points), "z_span": bounds[0].stop - bounds[0].start,
                       "radial_span_px": float(radius.max() - radius.min()),
                       "bounds_world_zyx": [[s.start + lower[i], s.stop + lower[i]] for i, s in enumerate(bounds)],
                       "view_center_local": point.tolist()})
    # Inspect the union of largest radial excursion, longest axial chain and most paint.
    chosen = {}
    for key in ("radial_span_px", "z_span", "voxels"):
        for row in sorted(chains, key=lambda r: (-r[key], r["component"]))[:5]:
            chosen[row["component"]] = row
    assets = []
    for number, row in chosen.items():
        point = row["view_center_local"]
        pictures = []
        for axis, name in enumerate(("Z", "Y", "X")):
            remaining = [i for i in range(3) if i != axis]
            slices = [slice(None)] * 3
            slices[axis] = point[axis]
            for i in remaining:
                start = max(0, min(304, point[i] - 40))
                slices[i] = slice(start, start + 80)
            slices = tuple(slices)
            pictures.append(panel_row([overlay(ct[slices]), overlay(ct[slices], raw[slices]),
                                       overlay(ct[slices], raw[slices], before[slices] & ~raw[slices]),
                                       overlay(ct[slices], raw[slices], after[slices] & ~raw[slices])],
                                      [f"CT {name}={point[axis]+lower[axis]}", "F0 raw", "Old preset +", "Candidate +"], scale=2))
        picture = Image.new("RGB", (pictures[0].width, 32 + sum(p.height for p in pictures)), "#10151e")
        ImageDraw.Draw(picture).text((8, 5), f"added component {number}: {row['voxels']} voxels, Z span {row['z_span']}, radial span {row['radial_span_px']:.2f}px", font=FONT, fill="white")
        y = 32
        for panel in pictures:
            picture.paste(panel, (0, y))
            y += panel.height
        name = f"chain_{number:04d}.png"
        picture.save(output / name)
        assets.append({**row, "image": name})
    summary = {"schema": "f0-independent-repair-contact-audit-v1", "block": args.block,
               "planes": len(plane_rows), "added_2d_strokes": sum(r["added_strokes"] for r in plane_rows),
               "detached_2d_strokes": sum(r["detached_strokes"] for r in plane_rows),
               "multiway_2d_strokes": sum(r["multiway_strokes"] for r in plane_rows),
               "added_3d_components": count, "maximum_radial_span_px": max((r["radial_span_px"] for r in chains), default=0),
               "maximum_axial_span_planes": max((r["z_span"] for r in chains), default=0),
               "worst_case_chain_images": assets,
               "interpretation": "Independent geometric contact diagnostics, not anatomical labels or a proof that all multi-join paths preserve wrap identity."}
    context.atomic_json(output / "summary.json", summary)
    context.atomic_json(output / "details.json", {"planes": plane_rows, "added_components": chains})
    print(json.dumps({k: v for k, v in summary.items() if k != "worst_case_chain_images"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
