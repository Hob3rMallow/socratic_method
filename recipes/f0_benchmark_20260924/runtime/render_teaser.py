#!/usr/bin/env python
"""Paper teaser: every seeded-random compressed cube of the blob audit (one per scroll, seed 20260925).

Rows: CT, the organisers' published M7 mask, F0 (shipped recipe), HercUNet v0 (its best configuration).
Columns: the eight audited scrolls in sample-plan order. No cube is chosen by outcome; the picks are the
showcase picks written by render_figures.py (showcase_picks.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HALO, CUBE = 32, 128
PANEL = (8, 13, 24)
TEXT = (248, 250, 252)
M7_CYAN = (55, 220, 255)
STUDENT_PINK = (255, 55, 190)
HERC_VIOLET = (167, 139, 250)


def gray(ct: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(ct, 1), np.percentile(ct, 99.5)
    return np.clip((ct.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


def tint(base: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    rgb = np.stack([base] * 3, -1) * 0.78
    rgb[mask] = 0.28 * rgb[mask] + 0.72 * (np.asarray(color) / 255.0)
    return rgb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--preds", type=Path, required=True)
    parser.add_argument("--picks", type=Path, required=True)
    parser.add_argument("--herc-config", default="hshipped_p0")
    parser.add_argument("--herc-threshold", type=float, default=0.50)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cell", type=int, default=256)
    args = parser.parse_args()
    picks = [c for c, stratum in json.loads(args.picks.read_text(encoding="utf-8"))["picks"] if stratum == "compressed"]
    cell, gap, label_w, head_h = args.cell, 6, 150, 44
    rows = ["CT", "M7, as published", "F0 (ours)", "HercUNet v0"]
    width = label_w + len(picks) * (cell + gap)
    height = head_h + len(rows) * (cell + gap)
    canvas = Image.new("RGB", (width, height), PANEL)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arialbd.ttf", 22)
        small = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = small = ImageFont.load_default()
    for c, cube_id in enumerate(picks):
        with np.load(args.audit / "cubes" / f"{cube_id}.npz") as archive:
            ct = archive["ct_context"][HALO:HALO + CUBE, HALO:HALO + CUBE, HALO:HALO + CUBE][CUBE // 2]
            published = archive["m7_published"][CUBE // 2] > 0
        with np.load(args.preds / "f0_tta" / f"{cube_id}.npz") as archive:
            f0 = archive["p8"][CUBE // 2].astype(np.float32) / 255.0 >= 0.30
        with np.load(args.preds / args.herc_config / f"{cube_id}.npz") as archive:
            herc = archive["p8"][CUBE // 2].astype(np.float32) / 255.0 >= args.herc_threshold
        base = gray(ct)
        panels = [np.stack([base] * 3, -1), tint(base, published, M7_CYAN), tint(base, f0, STUDENT_PINK),
                  tint(base, herc, HERC_VIOLET)]
        x = label_w + c * (cell + gap)
        draw.text((x + 6, 10), cube_id.split("_")[0], fill=TEXT, font=small)
        for r, panel in enumerate(panels):
            image = Image.fromarray((np.clip(panel, 0, 1) * 255).astype(np.uint8)).resize((cell, cell), Image.NEAREST)
            canvas.paste(image, (x, head_h + r * (cell + gap)))
    colors = [TEXT, M7_CYAN, STUDENT_PINK, HERC_VIOLET]
    for r, (name, color) in enumerate(zip(rows, colors)):
        y = head_h + r * (cell + gap) + cell // 2 - 14
        words = name.split(", ")
        for i, word in enumerate(words):
            draw.text((10, y + i * 26), word + ("," if i < len(words) - 1 else ""), fill=color, font=font)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(args.out, canvas.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
