#!/usr/bin/env python
"""Reference-free blob metrics on the pre-registered random cubes, plus the unedited gallery.

For every cube and every mask source this records
  * the canonical ScrollFiesta per-cube verdict (crossres_pred.voxel.scrollfiesta_metrics, contract
    scrollfiesta-pred-reject-2026-06-03-v1, fixed on 2026-06-03, months before this audit): keep / empty /
    solid-slab, erosion-interior fraction (voxels more than 2 six-connected steps from background),
    max erosion depth, fill fraction, largest component;
  * the canonical "thick" predicate on its own (interior fraction >= 0.50 and >= 2,000 interior voxels);
  * the share of predicted voxels at L1 depth >= 3, 4, 5 and 7 (a sheet k voxels thick has no voxel deeper
    than ceil(k/2), so depth >= 5 means a solid mass at least ~9 voxels thick, i.e. more than one wrap);
  * the share of predicted voxels lying on dark CT (below the cube's Otsu threshold): surface predicted in
    the gaps between wraps.
Sources are given as name=config:threshold, where config is a directory of uint8 probabilities written by
cube_infer.py, or the literal 'published' for the organisers' own published M7 mask (read unchanged).
The gallery renders, for the first --gallery cubes of every scroll IN THE PRE-REGISTERED DRAW ORDER (never a
selection by outcome), the central axial and coronal slices of CT and every source.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

HALO, CUBE = 32, 128


def depth_l1(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    return ndimage.distance_transform_cdt(padded, metric="taxicab")[1:-1, 1:-1, 1:-1]


def measure(mask: np.ndarray, ct: np.ndarray, otsu: float, sf) -> dict:
    metrics = sf.scrollfiesta_pred_metrics(mask.astype(np.uint8)).to_dict()
    fg = int(mask.sum())
    out = {k: metrics[k] for k in ("reject_kind", "interior_fraction", "interior_voxels", "max_thickness",
                                   "fill_fraction", "largest_component_voxels", "foreground_voxels")}
    out["thick"] = bool(metrics["interior_fraction"] >= sf.GARBAGE_INTERIOR_FRAC
                        and metrics["interior_voxels"] >= sf.GARBAGE_INTERIOR_MIN)
    if fg:
        depth = depth_l1(mask)[mask]
        for d in (3, 4, 5, 7):
            out[f"depth_ge_{d}"] = float(np.mean(depth >= d))
        out["dark_fraction"] = float(np.mean(ct[mask] < otsu))
    else:
        for d in (3, 4, 5, 7):
            out[f"depth_ge_{d}"] = 0.0
        out["dark_fraction"] = 0.0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit", type=Path, required=True, help="blob_audit directory (sample_plan.json, cubes/)")
    parser.add_argument("--preds", type=Path, required=True, help="cube_infer.py output directory")
    parser.add_argument("--source", action="append", required=True, help="name=config:threshold or name=published")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gallery", type=int, default=4)
    args = parser.parse_args()
    sys.path.insert(0, r"D:\work\socratic_method\src")
    from crossres_pred.voxel import scrollfiesta_metrics as sf

    sources = []
    for spec in args.source:
        name, rhs = spec.split("=", 1)
        if rhs == "published":
            sources.append((name, None, None))
        else:
            config, threshold = rhs.split(":")
            sources.append((name, config, float(threshold)))
    plan = json.loads((args.audit / "sample_plan.json").read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    gallery = []
    for scroll, info in plan["scrolls"].items():
        # Compression stratifier, fixed before any model comparison: the share of a cube's CT darker than the
        # scroll-level Otsu threshold (Otsu over the material voxels of all of that scroll's drawn cubes).
        # A compressed cube has little dark space between wraps; strata are terciles within each scroll.
        pooled = []
        for z, y, x in info["drawn_origins_zyx"]:
            with np.load(args.audit / "cubes" / f"{scroll}_z{z:05d}_y{y:05d}_x{x:05d}.npz") as archive:
                c = archive["ct_context"][HALO:HALO + CUBE, HALO:HALO + CUBE, HALO:HALO + CUBE]
            pooled.append(c[c > 0][::97])
        scroll_otsu = float(threshold_otsu(np.concatenate(pooled)))
        air = {}
        for z, y, x in info["drawn_origins_zyx"]:
            with np.load(args.audit / "cubes" / f"{scroll}_z{z:05d}_y{y:05d}_x{x:05d}.npz") as archive:
                c = archive["ct_context"][HALO:HALO + CUBE, HALO:HALO + CUBE, HALO:HALO + CUBE]
            air[(z, y, x)] = float(np.mean(c < scroll_otsu))
        cuts = np.quantile(list(air.values()), [1 / 3, 2 / 3])
        for order, (z, y, x) in enumerate(info["drawn_origins_zyx"]):
            cube_id = f"{scroll}_z{z:05d}_y{y:05d}_x{x:05d}"
            with np.load(args.audit / "cubes" / f"{cube_id}.npz") as archive:
                context = np.asarray(archive["ct_context"])
                published = np.asarray(archive["m7_published"])
            ct = context[HALO:HALO + CUBE, HALO:HALO + CUBE, HALO:HALO + CUBE]
            material = ct[ct > 0]
            otsu = float(threshold_otsu(material)) if material.size else 0.0
            a = air[(z, y, x)]
            stratum = "compressed" if a <= cuts[0] else ("open" if a > cuts[1] else "middle")
            row = {"cube_id": cube_id, "scroll": scroll, "draw_order": order, "origin_zyx": [z, y, x],
                   "otsu": otsu, "scroll_otsu": scroll_otsu, "air_fraction": a, "stratum": stratum, "sources": {}}
            masks = {}
            for name, config, threshold in sources:
                if config is None:
                    mask = published > 0
                else:
                    with np.load(args.preds / config / f"{cube_id}.npz") as archive:
                        mask = archive["p8"].astype(np.float32) / 255.0 >= threshold
                masks[name] = mask
                row["sources"][name] = measure(mask, ct, otsu, sf)
            rows.append(row)
            if order < args.gallery:
                gallery.append((cube_id, ct, masks))
            print(f"[blob] {cube_id} " + " ".join(
                f"{n}:{r['reject_kind'][:4]}/{r['interior_fraction']:.2f}" for n, r in row["sources"].items()), flush=True)
    (args.out / "cube_metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    summary = {"schema": "socratic-m7-blob-audit-metrics-v1", "cubes": len(rows), "sources": {}}
    for name, config, threshold in sources:
        per_scroll = {}
        for scroll in plan["scrolls"]:
            sel = [r["sources"][name] for r in rows if r["scroll"] == scroll]
            per_scroll[scroll] = aggregate(sel)
        per_stratum = {st: aggregate([r["sources"][name] for r in rows if r["stratum"] == st])
                       for st in ("compressed", "middle", "open")}
        summary["sources"][name] = {"config": config, "threshold": threshold,
                                    "overall": aggregate([r["sources"][name] for r in rows]),
                                    "per_scroll": per_scroll, "per_compression_stratum": per_stratum}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    render(gallery, [s[0] for s in sources], args.out / "gallery")
    for name in summary["sources"]:
        o = summary["sources"][name]["overall"]
        print(f"{name:18s} thick {o['thick_rate']:.3f} slab {o['solid_slab_rate']:.3f} empty {o['empty_rate']:.3f} "
              f"interior {o['mean_interior_fraction']:.3f} depth>=5 {o['mean_depth_ge_5']:.3f} dark {o['mean_dark_fraction']:.3f}")
    return 0


def aggregate(items: list[dict]) -> dict:
    n = max(1, len(items))
    return {
        "cubes": len(items),
        "any_depth_ge_6_rate": sum(i["max_thickness"] >= 6 for i in items) / n,
        "any_depth_ge_8_rate": sum(i["max_thickness"] >= 8 for i in items) / n,
        "thick_rate": sum(i["thick"] for i in items) / n,
        "solid_slab_rate": sum(i["reject_kind"] == "solid-slab" for i in items) / n,
        "empty_rate": sum(i["reject_kind"] == "empty" for i in items) / n,
        "mean_interior_fraction": float(np.mean([i["interior_fraction"] for i in items])) if items else 0.0,
        "mean_fill_fraction": float(np.mean([i["fill_fraction"] for i in items])) if items else 0.0,
        "mean_depth_ge_3": float(np.mean([i["depth_ge_3"] for i in items])) if items else 0.0,
        "mean_depth_ge_5": float(np.mean([i["depth_ge_5"] for i in items])) if items else 0.0,
        "mean_depth_ge_7": float(np.mean([i["depth_ge_7"] for i in items])) if items else 0.0,
        "mean_dark_fraction": float(np.mean([i["dark_fraction"] for i in items])) if items else 0.0,
        "median_max_thickness": float(np.median([i["max_thickness"] for i in items])) if items else 0.0,
    }


def render(gallery, names, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    colors = {"m7_published": (1.0, 0.25, 0.2), "f0": (0.2, 1.0, 0.35), "hercunet": (1.0, 0.2, 1.0)}
    for cube_id, ct, masks in gallery:
        fig, axes = plt.subplots(2, 1 + len(names), figsize=(3.2 * (1 + len(names)), 6.6))
        for row, axis in enumerate((0, 1)):
            index = [slice(None)] * 3
            index[axis] = CUBE // 2
            index = tuple(index)
            base = ct[index].astype(np.float32)
            base = (base - base.min()) / max(1e-6, float(base.max() - base.min()))
            axes[row, 0].imshow(base, cmap="gray")
            axes[row, 0].set_title(f"CT {'axial' if axis == 0 else 'coronal'}", fontsize=9)
            for col, name in enumerate(names, 1):
                rgb = np.stack([base] * 3, axis=-1) * 0.75
                color = np.asarray(colors.get(name.split("@")[0], (1.0, 0.8, 0.1)))
                mask = masks[name][index]
                rgb[mask] = 0.35 * rgb[mask] + 0.65 * color
                axes[row, col].imshow(rgb)
                axes[row, col].set_title(name, fontsize=9)
        for a in axes.ravel():
            a.axis("off")
        fig.suptitle(cube_id, fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"{cube_id}.png", dpi=80)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
