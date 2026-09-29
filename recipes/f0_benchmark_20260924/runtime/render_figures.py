#!/usr/bin/env python
"""Figures for the benchmark record and the show-and-tell page (no model is run here).

  showcase_random_cubes.png  the FIRST pre-registered cube of every audited scroll (draw order, never chosen by
                             outcome): CT, the organisers' published M7 mask, F0 (shipped recipe), HercUNet v0
  frontier.png               tolerant surface recall at 2 voxels versus the canonical solid-interior share,
                             one point per threshold, operating points marked
  blob_rates.png             share of random interior cubes whose prediction is a thick mass (canonical
                             ScrollFiesta predicate), per scroll, for every source
  headline.png               frozen-benchmark macro Dice and Kaggle composite for the compared configurations
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HALO, CUBE = 32, 128
COLORS = {"m7": (0.22, 0.86, 1.0), "f0": (1.0, 0.22, 0.75), "herc": (0.66, 0.55, 0.98), "m7tta": (0.10, 0.55, 0.75)}


def overlay(ct: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    base = ct.astype(np.float32)
    lo, hi = np.percentile(base, 1), np.percentile(base, 99.5)
    base = np.clip((base - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.stack([base] * 3, axis=-1) * 0.8
    rgb[mask] = 0.25 * rgb[mask] + 0.75 * np.asarray(color)
    return rgb


SHOWCASE_SEED = 20260925


def showcase_picks(cube_metrics: Path, strata=("compressed", "open")) -> list[tuple[str, str]]:
    """One cube per scroll and stratum, by a seeded random index that never looks at any prediction.
    (The drawn origins are stored sorted by position, so 'the first k drawn' would favour the scroll ends.)"""
    rows = [json.loads(l) for l in cube_metrics.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = np.random.default_rng(SHOWCASE_SEED)
    picks = []
    for scroll in dict.fromkeys(r["scroll"] for r in rows):
        for stratum in strata:
            ids = sorted(r["cube_id"] for r in rows if r["scroll"] == scroll and r["stratum"] == stratum)
            if ids:
                picks.append((ids[int(rng.integers(len(ids)))], stratum))
    return picks


def showcase(audit: Path, preds: Path, sources: dict, out: Path, picks: list[tuple[str, str]]) -> None:
    cols = 1 + len(sources)
    fig, axes = plt.subplots(len(picks), cols, figsize=(2.3 * cols, 2.35 * len(picks)))
    for r, (cube_id, stratum) in enumerate(picks):
        with np.load(audit / "cubes" / f"{cube_id}.npz") as archive:
            ct = archive["ct_context"][HALO:HALO + CUBE, HALO:HALO + CUBE, HALO:HALO + CUBE]
            published = archive["m7_published"] > 0
        sl = CUBE // 2
        base = ct[sl]
        lo, hi = np.percentile(base, 1), np.percentile(base, 99.5)
        axes[r, 0].imshow(np.clip((base - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray")
        axes[r, 0].set_ylabel(cube_id.split("_")[0] + chr(10) + stratum, fontsize=8)
        if r == 0:
            axes[r, 0].set_title("CT (axial)", fontsize=9)
        for c, (title, (config, threshold, color)) in enumerate(sources.items(), 1):
            if config == "published":
                mask = published
            else:
                with np.load(preds / config / f"{cube_id}.npz") as archive:
                    mask = archive["p8"].astype(np.float32) / 255.0 >= threshold
            axes[r, c].imshow(overlay(base, mask[sl], color))
            if r == 0:
                axes[r, c].set_title(title, fontsize=9)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def contact_sheets(audit: Path, preds: Path, sources: dict, out: Path, cell: int = 64) -> None:
    """Every drawn cube, no selection: one row per scroll (32 cubes), central axial slice, one sheet per source."""
    plan = json.loads((audit / "sample_plan.json").read_text(encoding="utf-8"))
    scrolls = list(plan["scrolls"])
    sheets = {name: np.zeros((len(scrolls) * cell, 32 * cell, 3), np.float32) for name in ["CT", *sources]}
    for r, scroll in enumerate(scrolls):
        for c, (z, y, x) in enumerate(plan["scrolls"][scroll]["drawn_origins_zyx"]):
            cube_id = f"{scroll}_z{z:05d}_y{y:05d}_x{x:05d}"
            with np.load(audit / "cubes" / f"{cube_id}.npz") as archive:
                ct = archive["ct_context"][HALO:HALO + CUBE, HALO:HALO + CUBE, HALO:HALO + CUBE][CUBE // 2]
                published = archive["m7_published"][CUBE // 2] > 0
            step = CUBE // cell
            base = overlay(ct, np.zeros_like(published), (0, 0, 0))[::step, ::step]
            sheets["CT"][r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = base
            for name, (config, threshold, color) in sources.items():
                if config == "published":
                    mask = published
                else:
                    with np.load(preds / config / f"{cube_id}.npz") as archive:
                        mask = archive["p8"][CUBE // 2].astype(np.float32) / 255.0 >= threshold
                sheets[name][r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = overlay(ct, mask, color)[::step, ::step]
    for name, image in sheets.items():
        fig, ax = plt.subplots(figsize=(32 * cell / 100, len(scrolls) * cell / 100 + 0.5))
        ax.imshow(np.clip(image, 0, 1))
        ax.set_yticks([cell * (i + 0.5) for i in range(len(scrolls))])
        ax.set_yticklabels(scrolls, fontsize=7)
        ax.set_xticks([])
        ax.set_title(f"{name}: all 256 randomly drawn interior cubes, central axial slice, no selection", fontsize=9)
        fig.tight_layout(pad=0.3)
        safe = name.split(" (")[0].replace(" ", "_").replace(",", "").lower()
        fig.savefig(out / f"contact_{safe}.png", dpi=100)
        plt.close(fig)


def frontier(analysis: dict, front: dict, out: Path, curves: dict) -> None:
    thresholds = np.asarray(front["thresholds"])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for label, (config, color, operating) in curves.items():
        if config not in front["configs"] or config not in analysis["summaries_tolerant_recall_2"]:
            continue
        x = np.asarray(analysis["summaries_tolerant_recall_2"][config])
        y = np.asarray(front["configs"][config]["pooled_interior_share"])
        ax.plot(x, y, "-o", ms=3, color=color, label=label)
        if operating is not None:
            i = int(np.argmin(np.abs(thresholds - operating)))
            ax.plot([x[i]], [y[i]], "s", ms=9, mfc="none", mec=color, mew=2)
    ax.set_xlabel("surface recall within 2 voxels (frozen benchmark)")
    ax.set_ylabel("solid-interior share of predicted voxels\n(more than 2 steps from background)")
    ax.set_title("Coverage versus blobs, every threshold 0.10-0.90\n(squares: operating points)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def blob_rates(summary: dict, out: Path, names: dict) -> None:
    """Deep-mass share (predicted voxels at L1 depth >= 5, i.e. inside a solid mass ~9+ voxels thick) by
    compression stratum, pooled over all eight scrolls (one third of each scroll's cubes per stratum)."""
    strata = ("open", "middle", "compressed")
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    width = 0.8 / len(names)
    for k, (source, (label, color)) in enumerate(names.items()):
        if source not in summary["sources"]:
            continue
        per = summary["sources"][source]["per_compression_stratum"]
        values = [100 * per[st]["mean_depth_ge_5"] for st in strata]
        bars = ax.bar(np.arange(len(strata)) + k * width, values, width, color=color, label=label,
                      edgecolor="black", linewidth=0.3)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.12, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(np.arange(len(strata)) + 0.4 - width / 2)
    ax.set_xticklabels(["open third", "middle third", "most compressed third"], fontsize=9)
    ax.set_ylabel("% of predicted voxels in a solid mass (depth >= 5)", fontsize=8)
    ax.set_title("256 random interior cubes, 8 scrolls, split by how compressed the CT is", fontsize=10)
    ax.set_ylim(0, 1.45 * max(ax.get_ylim()[1], 1.0))
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--preds", type=Path, required=True)
    parser.add_argument("--blob-summary", type=Path, required=True)
    parser.add_argument("--blob-metrics", type=Path, required=True, help="cube_metrics.jsonl (strata)")
    parser.add_argument("--herc-config", required=True, help="e.g. hshipped_p3")
    parser.add_argument("--herc-threshold", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    front = json.loads(args.frontier.read_text(encoding="utf-8"))
    blob = json.loads(args.blob_summary.read_text(encoding="utf-8"))
    display = {
        "M7, as published": ("published", None, COLORS["m7"]),
        "F0 (ours, shipped)": ("f0_tta", 0.30, COLORS["f0"]),
        "HercUNet v0 (best config)": (args.herc_config, args.herc_threshold, COLORS["herc"]),
    }
    picks = showcase_picks(args.blob_metrics)
    (args.out / "showcase_picks.json").write_text(json.dumps({"seed": SHOWCASE_SEED, "picks": picks}, indent=1) + chr(10), encoding="utf-8")
    for stratum in ("compressed", "open"):
        showcase(args.audit, args.preds, display, args.out / f"showcase_{stratum}.png",
                 [p for p in picks if p[1] == stratum])
    contact_sheets(args.audit, args.preds, display, args.out)
    frontier(analysis, front, args.out / "frontier.png", {
        "M7, no TTA (as published)": ("m7_flat", COLORS["m7"], 0.20),
        "M7 + our TTA": ("m7_tta", COLORS["m7tta"], None),
        "F0 (ours, shipped)": ("f0_tta", COLORS["f0"], 0.30),
        "HercUNet v0": (args.herc_config, COLORS["herc"], args.herc_threshold),
    })
    blob_rates(blob, args.out / "deep_mass_by_compression.png", {
        "m7_published": ("M7 as published (no TTA, T 0.2)", COLORS["m7"]),
        "m7_rerun_flat": ("M7, our rerun of the released weights, same setting", (0.55, 0.9, 1.0)),
        "m7_rerun_tta": ("M7 + 8-way TTA, T 0.30", COLORS["m7tta"]),
        "f0": ("F0 (ours, shipped: TTA, T 0.30)", COLORS["f0"]),
        "hercunet": ("HercUNet v0, best setting: 4 passes + TTA, T 0.30", COLORS["herc"]),
        "hercunet_4pass": ("HercUNet v0, default: 4 passes, no TTA, T 0.30", (0.45, 0.35, 0.8)),
        "hercunet_1pass": ("HercUNet v0, 1 pass, no TTA, T 0.50", (0.82, 0.76, 1.0)),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
