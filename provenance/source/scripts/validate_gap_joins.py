"""Validate pred_fixup gap joins against the co-registered 2.399 um CT.

Usage (from the repo root, crossres venv):

  crossres_pred/.venv/Scripts/python.exe crossres_pred/scripts/validate_gap_joins.py \
      --joins output/fixup_4x5x5/run1/joins.jsonl \
      --rejected output/fixup_4x5x5/run1/rejected.jsonl \
      --out output/crossres_data/gapfix_validation_20260818

Phases: plan (chunk set + bytes) -> fetch (local carve first, then anonymous
S3, into a PRIVATE sparse store under --out; the official carve mirror is
never written) -> verdicts -> report + side-by-side PNGs.

CPU only. Touches nothing the 250k reconciler owns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "crossres_pred" / "src"))

from crossres_pred.voxel import gap_join_validation as gjv  # noqa: E402
from crossres_pred.voxel.io import open_volume  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joins", required=True, help="pred_fixup joins.jsonl")
    parser.add_argument("--rejected", help="pred_fixup rejected.jsonl (recall probes)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--store", help="sparse fine store (default <out>/fine_ct.zarr)")
    parser.add_argument("--transform", default=str(gjv.DEFAULT_TRANSFORM))
    parser.add_argument("--source-uri", default=gjv.DEFAULT_SOURCE_URI)
    parser.add_argument("--local-mirror", default=str(gjv.DEFAULT_LOCAL_MIRROR))
    parser.add_argument("--coarse-raw", default=str(gjv.DEFAULT_COARSE_RAW),
                        help="local L0 raw zarr for the left PNG panel ('' to skip)")
    parser.add_argument("--limit", type=int, default=0, help="cap joins (0 = all)")
    parser.add_argument("--sample-rejects", type=int, default=40)
    parser.add_argument("--controls", type=int, default=30)
    parser.add_argument("--umb-y", type=float, default=3405.0)
    parser.add_argument("--umb-x", type=float, default=2878.0)
    parser.add_argument("--pitch", type=float, default=9.5)
    parser.add_argument("--corridor-fine", type=float, default=14.0,
                        help="corridor radius (fine vox) around the join segment")
    parser.add_argument("--seed-fine", type=float, default=10.0,
                        help="endpoint seed radius (fine vox)")
    parser.add_argument("--no-fetch", action="store_true",
                        help="judge with whatever the store already has")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--png-max", type=int, default=120)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    store = Path(args.store) if args.store else out / "fine_ct.zarr"

    affine = gjv.load_fine_to_coarse_affine(args.transform)

    rows = gjv.load_joins(args.joins, kept_only=True)
    if args.limit > 0:
        rows = rows[: args.limit]
    n_joins = len(rows)
    if args.rejected and args.sample_rejects > 0:
        rows += gjv.load_rejects(args.rejected, limit=args.sample_rejects)
    if args.controls > 0:
        rows += gjv.make_crosswrap_controls(
            rows[:n_joins], umb_y=args.umb_y, umb_x=args.umb_x,
            pitch=args.pitch, limit=args.controls,
        )
    print(f"rows: {n_joins} joins + {len(rows) - n_joins} probes/controls")

    # fine volume shape from the local mirror's .zarray (no network needed)
    zarray = json.loads(
        (Path(args.local_mirror) / "0" / ".zarray").read_text(encoding="utf-8")
    )
    fine_shape = tuple(int(v) for v in zarray["shape"])

    plan = gjv.plan_rows(rows, affine, fine_shape=fine_shape)
    print(f"plan: {len(plan['chunks'])} fine chunks, {plan['total_mib']:.0f} MiB")
    (out / "plan.json").write_text(
        json.dumps(
            {k: plan[k] for k in ("schema", "rows", "total_mib")}
            | {"n_chunks": len(plan["chunks"])},
            indent=1,
        ),
        encoding="utf-8",
    )

    stats = gjv.fetch_chunks(
        plan["chunks"], store_dir=store, source_uri=args.source_uri,
        local_mirror=args.local_mirror or None,
        workers=args.workers, allow_network=not args.no_fetch,
    )
    print(f"fetch: {stats}")

    fine_vol = open_volume(f"{store}::0")
    coarse_vol = open_volume(f"{args.coarse_raw}::0") if args.coarse_raw else None

    cfg = gjv.JudgeConfig(
        seed_radius_fine=args.seed_fine, corridor_radius_fine=args.corridor_fine
    )
    verdicts = []
    png_dir = out / "png"
    n_png = 0
    for i, row in enumerate(rows):
        verdict = gjv.judge_row(
            fine_vol, row, affine, store_dir=store, cfg=cfg,
            fine_shape=fine_shape,
        )
        verdicts.append(verdict)
        if n_png < args.png_max and verdict["verdict"] != "NO_DATA":
            try:
                gjv.render_row_png(
                    png_dir / f"{verdict['verdict']}_{row.row_id}.png",
                    row, verdict, fine_vol, affine, coarse_vol=coarse_vol,
                )
                n_png += 1
            except Exception as error:  # noqa: BLE001 - PNGs are best-effort
                print(f"  png failed for {row.row_id}: {error}")
        if (i + 1) % 50 == 0:
            print(f"  judged {i + 1}/{len(rows)}")

    with (out / "verdicts.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for verdict in verdicts:
            fh.write(json.dumps(verdict, separators=(",", ":")) + "\n")

    summary = gjv.summarize(verdicts)
    summary["fetch"] = stats
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"artifacts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
