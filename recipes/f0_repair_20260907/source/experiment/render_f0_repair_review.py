"""Render CT-aligned repair evidence without changing predictions or scores."""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont

import run_f0_repair_context as context

FONT = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 15)
SMALL = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 12)


def origin(cube: str) -> tuple[int, int, int]:
    return tuple(int(token[1:]) for token in cube.split("_"))


def assemble(directory: Path, ids: list[str], lower: list[int]) -> np.ndarray:
    result = None
    occupied = np.zeros((3, 3, 3), dtype=bool)
    for key in ids:
        value = tifffile.imread(directory / f"{key}.tif")
        if value.shape != (128, 128, 128):
            raise ValueError(f"wrong cube shape: {key}")
        start = tuple(a - b for a, b in zip(origin(key), lower))
        index = tuple(value // 128 for value in start)
        if occupied[index]:
            raise ValueError("duplicate cube")
        occupied[index] = True
        if result is None:
            result = np.zeros((384, 384, 384), dtype=value.dtype)
        result[tuple(slice(v, v + 128) for v in start)] = value
    if not occupied.all():
        raise ValueError("missing contiguous context")
    return result


def overlay(ct: np.ndarray, raw: np.ndarray | None = None, additions: np.ndarray | None = None) -> Image.Image:
    rgb = np.repeat(ct[..., None], 3, axis=2).astype(np.float32)
    if raw is not None:
        rgb[raw] = .48 * rgb[raw] + .52 * np.array([218, 65, 213])
    if additions is not None:
        rgb[additions] = np.array([60, 255, 128])
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def panel_row(pictures: list[Image.Image], labels: list[str], scale: int = 1) -> Image.Image:
    width, height = pictures[0].size
    result = Image.new("RGB", (len(pictures) * (width * scale + 8) + 8, height * scale + 42), "#10151e")
    draw = ImageDraw.Draw(result)
    for index, (picture, label) in enumerate(zip(pictures, labels)):
        left = 8 + index * (width * scale + 8)
        draw.text((left, 6), label, font=FONT, fill="#e5e9f0")
        result.paste(picture.resize((width * scale, height * scale), Image.Resampling.NEAREST), (left, 33))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("block", choices=("development", "control"))
    parser.add_argument("--preset", default="extended_safe", choices=tuple(context.PRESETS))
    args = parser.parse_args()
    request = context.read(context.OUTPUT / "request.json")
    spec = request["blocks"][args.block]
    ids, lower = spec["target_cube_ids"], spec["origin_zyx"]
    inference = context.OUTPUT / f"inference_{args.block}"
    comparison = context.OUTPUT / "repairs" / args.block
    target = comparison / args.preset
    destination = context.OUTPUT / "review" / args.block / args.preset
    if destination.exists():
        raise FileExistsError(f"preserve prior review: {destination}")
    raw = assemble(inference / "cubes_PRED", ids, lower) != 0
    probability = assemble(inference / "probability", ids, lower)
    baseline = assemble(comparison / "report_baseline/cubes_PRED", ids, lower) != 0
    fixed = assemble(target / "cubes_PRED", ids, lower) != 0
    ct_raw = assemble(context.SOURCE / "cubes_RAW", ids, lower)
    lo, hi = np.percentile(ct_raw[ct_raw > 0], [1, 99.5])
    ct = np.clip((ct_raw.astype(np.float32) - lo) * (255 / max(1, hi - lo)), 0, 255).astype(np.uint8)
    if (raw & ~fixed).any() or (raw & ~baseline).any():
        raise ValueError("non-additive masks cannot be rendered as repairs")
    tracks = defaultdict(list)
    for line in (target / "joins.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["kept"]:
            tracks[row["conn_id"]].append(row)
    destination.mkdir(parents=True)
    assets = destination / "assets"
    assets.mkdir()
    cases = []
    for connection, rows in tracks.items():
        representative = max(rows, key=lambda r: (r["dist"], -r["score"], -r["z"]))
        r = representative
        z = r["z"] - lower[0]
        ay, ax = r["a"]["y"] - lower[1], r["a"]["x"] - lower[2]
        by, bx = r["b"]["y"] - lower[1], r["b"]["x"] - lower[2]
        cy, cx = (ay + by) // 2, (ax + bx) // 2
        y0, x0 = max(0, min(384 - 80, cy - 40)), max(0, min(384 - 80, cx - 40))
        window = (z, slice(y0, y0 + 80), slice(x0, x0 + 80))
        additions = fixed[window] & ~raw[window]
        extra = fixed[window] & ~baseline[window]
        # Straight-chord probabilities are diagnostics, not a replica of the native Bezier.
        ys = np.rint(np.linspace(ay, by, 21)).astype(int)
        xs = np.rint(np.linspace(ax, bx, 21)).astype(int)
        p_line = probability[z, ys, xs].astype(float)
        p = np.clip(probability[window].astype(float) / .5, 0, 1)
        p_rgb = np.stack((p * 255, np.sqrt(p) * 185, (1 - p) * 110), axis=-1).astype(np.uint8)
        picture = panel_row(
            [overlay(ct[window]), overlay(ct[window], raw[window]),
             overlay(ct[window], raw[window], baseline[window] & ~raw[window]),
             overlay(ct[window], raw[window], additions), Image.fromarray(p_rgb)],
            ["CT", "F0 raw", "Old preset +", "Candidate +", "Probability 0–.5"], scale=2,
        )
        asset = f"assets/track_{connection:04d}.png"
        picture.save(destination / asset)
        cases.append({"connection": connection, "planes": len(rows), "z": r["z"],
                      "distance": r["dist"], "score": r["score"], "support": r["support"],
                      "center_yx": [cy + lower[1], cx + lower[2]],
                      "core": 128 <= z < 256 and 128 <= cy < 256 and 128 <= cx < 256,
                      "chord_probability_mean": float(p_line.mean()), "chord_probability_min": float(p_line.min()),
                      "extra_pixels_in_crop": int(extra.sum()), "image": asset})
    overview_planes = [0, 11, 18, 128, 161, 192, 224, 288]
    overview_assets = []
    for z in overview_planes:
        picture = panel_row([overlay(ct[z]), overlay(ct[z], raw[z]),
                             overlay(ct[z], raw[z], baseline[z] & ~raw[z]),
                             overlay(ct[z], raw[z], fixed[z] & ~raw[z])],
                            [f"CT z={z + lower[0]}", "F0 raw @.35", "Old preset additions", f"{args.preset} additions"])
        asset = f"assets/overview_z{z + lower[0]}.png"
        picture.save(destination / asset)
        overview_assets.append(asset)
    cases.sort(key=lambda row: (not row["core"], row["chord_probability_min"], -row["distance"], row["connection"]))
    body = ["<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
            "<title>F0 contiguous repair review</title><style>body{background:#10151e;color:#e5e9f0;font:16px system-ui;margin:24px}img{max-width:100%;height:auto}article{border-top:1px solid #465166;padding:14px 0}code{color:#73d6ff}summary{cursor:pointer}p{max-width:1000px}</style></head><body>",
            f"<h1>F0 repair: {args.block} / {args.preset}</h1>",
            "<p>Magenta: unchanged frozen F0 foreground. Green: additive repair. Shared CT window/coordinates in every comparison. Probability is model evidence, not ground truth. All accepted connection tracks are shown; the longest join represents each track. Core cases first, then lowest chord probability. The control is not used to tune the preset.</p>"]
    summary = context.read(comparison / f"{args.preset}_summary.json")
    body.append(f"<p>{summary['native']['kept']} per-plane joins; {summary['kept_connection_tracks']} tracks; {summary['native']['painted_px']} added voxels; {summary['core']['added_voxels']} added in the central assessment cube. Zero raw foreground erased. Anatomical/chain safety is not certified by these counts.</p>")
    for asset in overview_assets:
        body.append(f"<details><summary>{html.escape(asset)}</summary><img src='{asset}'></details>")
    for case in cases:
        body.append(f"<article id='track-{case['connection']}'><h3>Track {case['connection']} {'— CORE' if case['core'] else ''}</h3><p>z={case['z']}, y/x={case['center_yx']}; distance={case['distance']:.2f}, support={case['support']}, score={case['score']:.3f}, planes={case['planes']}; chord P min={case['chord_probability_min']:.3f}, mean={case['chord_probability_mean']:.3f}</p><img loading='lazy' src='{case['image']}'></article>")
    body.append("</body></html>")
    (destination / "index.html").write_text("\n".join(body), encoding="utf-8")
    context.atomic_json(destination / "cases.json", {"schema": "f0-repair-track-review-v1", "block": args.block,
                        "preset": args.preset, "ct_window": [float(lo), float(hi)], "cases": cases,
                        "overview_images": overview_assets, "source_summary": context.inventory(comparison / f"{args.preset}_summary.json")})
    # Overview contact sheet makes risk-first inspection possible without opening every HTML card.
    selected = cases[:12]
    thumbs = []
    for case in selected:
        picture = Image.open(destination / case["image"]).convert("RGB")
        thumb = Image.new("RGB", (picture.width, picture.height + 26), "#10151e")
        thumb.paste(picture, (0, 26))
        ImageDraw.Draw(thumb).text((8, 3), f"track {case['connection']} z{case['z']} d{case['distance']:.1f} Pmin{case['chord_probability_min']:.3f} core={case['core']}", font=FONT, fill="white")
        thumbs.append(thumb)
    if thumbs:
        contact = Image.new("RGB", (thumbs[0].width, sum(im.height for im in thumbs)), "#10151e")
        y = 0
        for picture in thumbs:
            contact.paste(picture, (0, y))
            y += picture.height
        contact.save(destination / "risk_first_contact.png")
    print(f"rendered {len(cases)} track cards and {len(overview_assets)} overview planes: {destination}", flush=True)


if __name__ == "__main__":
    main()
