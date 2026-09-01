from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from crossres_pred.voxel.coarse_teacher_atlas import (
    _pair_record,
    _qualified_fine_support,
    validate_coarse_teacher_atlas,
)
from crossres_pred.voxel.io import open_volume
from crossres_pred.voxel.medial import (
    FineMedialSurfaceReader,
    MedialProjectionOptions,
    medial_provenance,
    project_fine_medial_patch,
)
from crossres_pred.voxel.registration import FineFieldWindowReader
from crossres_pred.voxel.resources import configure_cpu_budget

AUDIT_SCHEMA = "crossres-medial-halo-convergence-audit-v1"
DEFAULT_CANDIDATE_HALO = (1, 32, 32)
DEFAULT_REFERENCE_HALO = (1, 64, 64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def select_density_stratified_tiles(
    rows: Sequence[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Deterministically cover sparse/dense positive parent-atlas tiles."""

    if count <= 0:
        raise ValueError("samples per source must be positive")
    eligible = [
        row
        for row in rows
        if bool(row.get("present"))
        and int(row.get("known_voxels", 0)) > 0
        and int(row.get("positive_voxels", 0)) > 0
    ]
    if not eligible:
        raise ValueError("atlas inventory has no known positive tiles to audit")

    def density_key(row: dict[str, Any]) -> tuple[float, float, int]:
        known = int(row["known_voxels"])
        positive = int(row["positive_voxels"])
        shape = tuple(int(value) for value in row["shape_zyx"])
        voxels = int(np.prod(shape))
        return (
            known / max(1, voxels),
            positive / max(1, known),
            int(row["index"]),
        )

    ranked = sorted(eligible, key=density_key)
    selected_count = min(count, len(ranked))
    if selected_count == 1:
        return [ranked[len(ranked) // 2]]
    positions = [
        index * (len(ranked) - 1) // (selected_count - 1)
        for index in range(selected_count)
    ]
    return [ranked[position] for position in positions]


def density_stratified_tile_order(
    rows: Sequence[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Order initial strata and nearby deterministic replacements.

    A positive parent-atlas tile can still have no support that is jointly valid
    under both medial halos (for example, beside a sparse fine-volume edge).
    The audit must record and replace such a tile, rather than treating absence
    of a comparison domain as either agreement or disagreement.
    """

    selected = select_density_stratified_tiles(rows, count)
    eligible = [
        row
        for row in rows
        if bool(row.get("present"))
        and int(row.get("known_voxels", 0)) > 0
        and int(row.get("positive_voxels", 0)) > 0
    ]

    def density_key(row: dict[str, Any]) -> tuple[float, float, int]:
        known = int(row["known_voxels"])
        positive = int(row["positive_voxels"])
        shape = tuple(int(value) for value in row["shape_zyx"])
        voxels = int(np.prod(shape))
        return (
            known / max(1, voxels),
            positive / max(1, known),
            int(row["index"]),
        )

    ranked = sorted(eligible, key=density_key)
    positions_by_index = {
        int(row["index"]): position for position, row in enumerate(ranked)
    }
    anchor_positions = [positions_by_index[int(row["index"])] for row in selected]
    ordered: list[dict[str, Any]] = []
    seen: set[int] = set()
    for radius in range(len(ranked)):
        offsets = (0,) if radius == 0 else (radius, -radius)
        for anchor in anchor_positions:
            for offset in offsets:
                position = anchor + offset
                if not 0 <= position < len(ranked):
                    continue
                row = ranked[position]
                index = int(row["index"])
                if index not in seen:
                    seen.add(index)
                    ordered.append(row)
        if len(ordered) == len(ranked):
            break
    return ordered


def _raw_comparison_counts(
    candidate: np.ndarray,
    candidate_valid: np.ndarray,
    reference: np.ndarray,
    reference_valid: np.ndarray,
) -> dict[str, int]:
    arrays = (candidate, candidate_valid, reference, reference_valid)
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("candidate/reference medial arrays differ in shape")
    candidate_mask = np.asarray(candidate > 0, dtype=bool)
    reference_mask = np.asarray(reference > 0, dtype=bool)
    candidate_known = np.asarray(candidate_valid > 0, dtype=bool)
    reference_known = np.asarray(reference_valid > 0, dtype=bool)
    common = candidate_known & reference_known
    candidate_common = candidate_mask & common
    reference_common = reference_mask & common
    intersection = candidate_common & reference_common
    return {
        "voxels": int(candidate_mask.size),
        "candidate_valid_voxels": int(np.count_nonzero(candidate_known)),
        "reference_valid_voxels": int(np.count_nonzero(reference_known)),
        "common_valid_voxels": int(np.count_nonzero(common)),
        "agreement_voxels": int(
            np.count_nonzero((candidate_mask == reference_mask) & common)
        ),
        "candidate_crest_voxels": int(np.count_nonzero(candidate_common)),
        "reference_crest_voxels": int(np.count_nonzero(reference_common)),
        "intersection_crest_voxels": int(np.count_nonzero(intersection)),
    }


def _finalize_counts(counts: dict[str, int]) -> dict[str, int | float]:
    common = counts["common_valid_voxels"]
    candidate = counts["candidate_crest_voxels"]
    reference = counts["reference_crest_voxels"]
    intersection = counts["intersection_crest_voxels"]
    if common <= 0:
        raise ValueError("candidate/reference halos have no common valid voxels")
    crest_denominator = candidate + reference
    return {
        **counts,
        "candidate_valid_fraction": counts["candidate_valid_voxels"]
        / counts["voxels"],
        "reference_valid_fraction": counts["reference_valid_voxels"]
        / counts["voxels"],
        "common_valid_fraction": common / counts["voxels"],
        "voxel_agreement": counts["agreement_voxels"] / common,
        "crest_precision": intersection / max(1, candidate),
        "crest_recall": intersection / max(1, reference),
        "crest_dice": (
            2.0 * intersection / crest_denominator
            if crest_denominator
            else 1.0
        ),
    }


def compare_medial_projections(
    candidate: np.ndarray,
    candidate_valid: np.ndarray,
    reference: np.ndarray,
    reference_valid: np.ndarray,
) -> dict[str, int | float]:
    return _finalize_counts(
        _raw_comparison_counts(
            candidate,
            candidate_valid,
            reference,
            reference_valid,
        )
    )


def _sum_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "voxels",
        "candidate_valid_voxels",
        "reference_valid_voxels",
        "common_valid_voxels",
        "agreement_voxels",
        "candidate_crest_voxels",
        "reference_crest_voxels",
        "intersection_crest_voxels",
    )
    return {key: sum(int(row[key]) for row in rows) for key in keys}


def audit_atlas(
    atlas_path: Path,
    *,
    samples_per_source: int,
    candidate_options: MedialProjectionOptions,
    reference_options: MedialProjectionOptions,
    fine_chunk_cache_entries: int,
    minimum_voxel_agreement: float,
    minimum_crest_dice: float,
) -> dict[str, Any]:
    root = atlas_path.expanduser().resolve()
    parent = validate_coarse_teacher_atlas(root)
    identity = parent["identity"]
    inventory_path = Path(str(parent["tile_inventory"])).resolve()
    candidates = density_stratified_tile_order(
        _read_jsonl(inventory_path), samples_per_source
    )
    pair_manifest = Path(str(identity["pair_manifest"])).resolve()
    record = _pair_record(pair_manifest, str(identity["record_id"]))
    fine_volume = open_volume(record.fine.target.volume)
    candidate_value = identity.get("candidate_fine_chunks_path")
    candidate_path = Path(str(candidate_value)).resolve() if candidate_value else None
    support = _qualified_fine_support(record, fine_volume, candidate_path)
    field_reader = FineFieldWindowReader(
        fine_volume,
        record.fine.target,
        support,
        max_cache_chunks=fine_chunk_cache_entries,
    )

    comparisons: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    with (
        FineMedialSurfaceReader(
            field_reader, options=candidate_options
        ) as candidate_reader,
        FineMedialSurfaceReader(
            field_reader, options=reference_options
        ) as reference_reader,
    ):
        for attempt, row in enumerate(candidates, 1):
            started = time.perf_counter()
            crest, valid, _ = project_fine_medial_patch(
                candidate_reader,
                record.fine.to_coarse_affine_xyz,
                tuple(int(value) for value in row["origin_zyx"]),
                tuple(int(value) for value in row["shape_zyx"]),
            )
            candidate_seconds = time.perf_counter() - started
            started = time.perf_counter()
            reference, reference_valid, _ = project_fine_medial_patch(
                reference_reader,
                record.fine.to_coarse_affine_xyz,
                tuple(int(value) for value in row["origin_zyx"]),
                tuple(int(value) for value in row["shape_zyx"]),
            )
            reference_seconds = time.perf_counter() - started
            counts = _raw_comparison_counts(
                crest,
                valid,
                reference,
                reference_valid,
            )
            if counts["common_valid_voxels"] <= 0:
                skipped.append(
                    {
                        "index": int(row["index"]),
                        "tile_coordinate_zyx": row["tile_coordinate_zyx"],
                        "origin_zyx": row["origin_zyx"],
                        "shape_zyx": row["shape_zyx"],
                        "parent_known_voxels": int(row["known_voxels"]),
                        "parent_positive_voxels": int(row["positive_voxels"]),
                        "candidate_seconds": candidate_seconds,
                        "reference_seconds": reference_seconds,
                        "reason": "no-common-valid-voxels",
                        **counts,
                    }
                )
                print(
                    f"{record.scroll_id} skipped attempt={attempt} "
                    f"tile={int(row['index'])} reason=no-common-valid-voxels",
                    flush=True,
                )
                continue
            comparison = _finalize_counts(counts)
            comparisons.append(
                {
                    "index": int(row["index"]),
                    "tile_coordinate_zyx": row["tile_coordinate_zyx"],
                    "origin_zyx": row["origin_zyx"],
                    "shape_zyx": row["shape_zyx"],
                    "parent_known_voxels": int(row["known_voxels"]),
                    "parent_positive_voxels": int(row["positive_voxels"]),
                    "candidate_seconds": candidate_seconds,
                    "reference_seconds": reference_seconds,
                    **comparison,
                }
            )
            print(
                f"{record.scroll_id} accepted {len(comparisons)}/"
                f"{samples_per_source} attempt={attempt} "
                f"tile={int(row['index'])} "
                f"agreement={float(comparison['voxel_agreement']):.6f} "
                f"crest_dice={float(comparison['crest_dice']):.6f}",
                flush=True,
            )
            if len(comparisons) >= samples_per_source:
                break

    if len(comparisons) < min(samples_per_source, len(candidates)):
        raise ValueError(
            f"{record.scroll_id} has only {len(comparisons)} comparable medial "
            f"tiles for {samples_per_source} requested samples"
        )

    aggregate = _finalize_counts(_sum_counts(comparisons))
    passed = (
        aggregate["reference_crest_voxels"] > 0
        and aggregate["voxel_agreement"] >= minimum_voxel_agreement
        and aggregate["crest_dice"] >= minimum_crest_dice
    )
    return {
        "record_id": record.record_id,
        "scroll_id": record.scroll_id,
        "atlas": str(root),
        "atlas_state_sha256": _sha256(root / "atlas_state.json"),
        "atlas_tile_inventory_sha256": _sha256(inventory_path),
        "fine_support_chunks": (
            int(np.prod(support.grid_zyx))
            if support.present_ids is None
            else int(support.present_ids.size)
        ),
        "selected_tiles": len(comparisons),
        "skipped_tiles": len(skipped),
        "selection": "positive-density-stratified-with-nearest-replacement-v2",
        "tiles": comparisons,
        "skipped": skipped,
        "aggregate": aggregate,
        "passed": bool(passed),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether the production fine-medial halo agrees with a "
            "larger-context reference on deterministic real teacher tiles."
        )
    )
    parser.add_argument("--atlas", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-source", type=int, default=8)
    parser.add_argument(
        "--candidate-halo", type=int, nargs=3, default=DEFAULT_CANDIDATE_HALO
    )
    parser.add_argument(
        "--reference-halo", type=int, nargs=3, default=DEFAULT_REFERENCE_HALO
    )
    parser.add_argument("--minimum-voxel-agreement", type=float, default=0.999)
    parser.add_argument("--minimum-crest-dice", type=float, default=0.99)
    parser.add_argument("--max-cpu-threads", type=int, default=16)
    parser.add_argument("--skeleton-workers", type=int, default=8)
    parser.add_argument("--fine-chunk-cache-entries", type=int, default=128)
    parser.add_argument("--medial-chunk-cache-entries", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.max_cpu_threads <= 16:
        raise ValueError("max CPU threads must be in [1, 16]")
    if args.skeleton_workers > args.max_cpu_threads:
        raise ValueError("skeleton workers exceed max CPU threads")
    if not 0.0 <= args.minimum_voxel_agreement <= 1.0:
        raise ValueError("minimum voxel agreement must be in [0, 1]")
    if not 0.0 <= args.minimum_crest_dice <= 1.0:
        raise ValueError("minimum crest Dice must be in [0, 1]")
    candidate_halo = tuple(int(value) for value in args.candidate_halo)
    reference_halo = tuple(int(value) for value in args.reference_halo)
    if not all(
        reference >= candidate
        for candidate, reference in zip(
            candidate_halo, reference_halo, strict=True
        )
    ) or candidate_halo == reference_halo:
        raise ValueError("reference halo must strictly contain candidate halo")
    configure_cpu_budget(args.max_cpu_threads)
    candidate_options = MedialProjectionOptions(
        halo_zyx=candidate_halo,
        skeleton_workers=args.skeleton_workers,
        max_cache_chunks=args.medial_chunk_cache_entries,
    )
    reference_options = MedialProjectionOptions(
        halo_zyx=reference_halo,
        skeleton_workers=args.skeleton_workers,
        max_cache_chunks=args.medial_chunk_cache_entries,
    )
    candidate_options.validate()
    reference_options.validate()

    sources = [
        audit_atlas(
            atlas,
            samples_per_source=args.samples_per_source,
            candidate_options=candidate_options,
            reference_options=reference_options,
            fine_chunk_cache_entries=args.fine_chunk_cache_entries,
            minimum_voxel_agreement=args.minimum_voxel_agreement,
            minimum_crest_dice=args.minimum_crest_dice,
        )
        for atlas in args.atlas
    ]
    overall = _finalize_counts(
        _sum_counts([source["aggregate"] for source in sources])
    )
    passed = all(bool(source["passed"]) for source in sources) and (
        overall["voxel_agreement"] >= args.minimum_voxel_agreement
        and overall["crest_dice"] >= args.minimum_crest_dice
    )
    report = {
        "schema": AUDIT_SCHEMA,
        "candidate": medial_provenance(candidate_options),
        "reference": medial_provenance(reference_options),
        "thresholds": {
            "minimum_voxel_agreement": args.minimum_voxel_agreement,
            "minimum_crest_dice": args.minimum_crest_dice,
        },
        "sources": sources,
        "overall": overall,
        "passed": bool(passed),
    }
    _atomic_json(args.output.expanduser().resolve(), report)
    print(
        f"medial halo audit {'PASS' if passed else 'FAIL'} "
        f"agreement={float(overall['voxel_agreement']):.6f} "
        f"crest_dice={float(overall['crest_dice']):.6f} "
        f"output={args.output.expanduser().resolve()}",
        flush=True,
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
