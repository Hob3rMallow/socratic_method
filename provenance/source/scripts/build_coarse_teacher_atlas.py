#!/usr/bin/env python3
"""Build a sparse antialiased fine-teacher atlas in coarse voxel space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crossres_pred.voxel.coarse_teacher_atlas import (
    CoarseTeacherAtlasOptions,
    CoarseTeacherMedialAtlasOptions,
    build_coarse_teacher_atlas,
    build_coarse_teacher_medial_atlas,
    validate_coarse_teacher_atlas,
    validate_coarse_teacher_medial_atlas,
)
from crossres_pred.voxel.medial import MedialProjectionOptions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", type=Path)
    parser.add_argument("--record-id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-shape", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--candidate-margin", type=int, default=3)
    parser.add_argument("--fine-chunk-cache-entries", type=int, default=256)
    parser.add_argument("--max-cpu-threads", type=int, default=16)
    parser.add_argument("--maximum-tiles", type=int)
    parser.add_argument("--candidate-fine-chunks", type=Path)
    parser.add_argument("--with-medial", action="store_true")
    parser.add_argument("--medial-only", action="store_true")
    parser.add_argument("--medial-halo", type=int, nargs=3, default=(1, 32, 32))
    parser.add_argument("--medial-skeleton-workers", type=int, default=8)
    parser.add_argument("--medial-chunk-cache-entries", type=int, default=64)
    parser.add_argument(
        "--require-cuda", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.with_medial and args.medial_only:
        parser.error("--with-medial and --medial-only are mutually exclusive")
    if args.validate_only:
        state = (
            validate_coarse_teacher_medial_atlas(args.output)
            if args.with_medial or args.medial_only
            else validate_coarse_teacher_atlas(args.output)
        )
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0
    if not args.medial_only:
        if args.pair_manifest is None or args.record_id is None:
            parser.error("base atlas build requires --pair-manifest and --record-id")
        state_path = build_coarse_teacher_atlas(
            pair_manifest_path=args.pair_manifest,
            record_id=args.record_id,
            output_path=args.output,
            options=CoarseTeacherAtlasOptions(
                tile_shape_zyx=tuple(args.tile_shape),
                candidate_margin_coarse_vox=args.candidate_margin,
                fine_chunk_cache_entries=args.fine_chunk_cache_entries,
                max_cpu_threads=args.max_cpu_threads,
                require_cuda=args.require_cuda,
            ),
            maximum_tiles=args.maximum_tiles,
            candidate_fine_chunks_path=args.candidate_fine_chunks,
        )
    else:
        state_path = args.output / "atlas_state.json"
    if args.with_medial or args.medial_only:
        state_path = build_coarse_teacher_medial_atlas(
            atlas_path=args.output,
            options=CoarseTeacherMedialAtlasOptions(
                max_cpu_threads=args.max_cpu_threads,
                fine_chunk_cache_entries=args.fine_chunk_cache_entries,
                medial=MedialProjectionOptions(
                    halo_zyx=tuple(args.medial_halo),
                    skeleton_workers=args.medial_skeleton_workers,
                    max_cache_chunks=args.medial_chunk_cache_entries,
                ),
            ),
        )
    print(state_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
