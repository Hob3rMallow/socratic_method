#!/usr/bin/env python
"""Relabel the paper teaser with the release name.

Reads the head-to-head record's teaser, rendered by
recipes/f0_benchmark_20260924/runtime/render_teaser.py with the row label
"F0 (ours)", and repaints only that row's label cell with "Socratic Method
September 2026", the name the September 2026 progress video uses. Every image
panel keeps its pixels; the record's own copy is left unchanged.

Run from the repository root:

    python submissions/2026-09/relabel_teaser.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SOURCE = Path("recipes/f0_benchmark_20260924/evidence/figures/teaser_f0_compressed.png")
TARGET = Path("submissions/2026-09/figures/teaser_f0_compressed.png")

# render_teaser.py layout at its default --cell 256: eight columns, four rows
# (CT, M7 as published, the shipped student, HercUNet v0).
CELL, GAP, LABEL_W, HEAD_H = 256, 6, 150, 44
PANEL = (8, 13, 24)
STUDENT_PINK = (255, 55, 190)
STUDENT_ROW = 2
LINES = ("Socratic", "Method", "September", "2026")
LINE_H = 26


def main() -> int:
    image = Image.open(SOURCE).convert("RGB")
    expected = (LABEL_W + 8 * (CELL + GAP), HEAD_H + 4 * (CELL + GAP))
    if image.size != expected:
        raise SystemExit(f"{SOURCE} is {image.size}, expected the render_teaser layout {expected}")
    top = HEAD_H + STUDENT_ROW * (CELL + GAP)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, top, LABEL_W - 1, top + CELL - 1), fill=PANEL)
    font = ImageFont.truetype("arialbd.ttf", 22)
    # render_teaser.py centres a one-line label at cell // 2 - 14; keep that centre.
    y = top + CELL // 2 - 14 - (len(LINES) - 1) * LINE_H // 2
    for i, line in enumerate(LINES):
        draw.text((10, y + i * LINE_H), line, fill=STUDENT_PINK, font=font)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    image.save(TARGET, optimize=True)
    print(TARGET, image.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
