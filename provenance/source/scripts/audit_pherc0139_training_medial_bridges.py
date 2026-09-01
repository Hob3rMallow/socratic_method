from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from crossres_pred.voxel.io import decode_dense_field, open_volume, read_crop
from crossres_pred.voxel.medial_bridges import (
    PINNED_MEDIAL_BRIDGE_CONTRACT,
    PinnedMedialBridgeOptions,
    extract_pinned_medial_bridge,
)
from crossres_pred.voxel.patches import load_patch_manifest
from crossres_pred.voxel.schema import DenseFieldSpec
from crossres_pred.voxel.train import StratifiedEpochPartitionSampler

AUDIT_SCHEMA = "crossres-training-pinned-medial-bridge-audit-v2"
TILE_SHAPE = (64, 64, 64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _aligned_tile_origins(
    origin: tuple[int, int, int],
    shape: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    ranges: list[range] = []
    for start, size, tile_size in zip(origin, shape, TILE_SHAPE, strict=True):
        first = ((start + tile_size - 1) // tile_size) * tile_size
        last = start + size - tile_size
        ranges.append(range(first, last + 1, tile_size))
    return [
        (z, y, x)
        for z in ranges[0]
        for y in ranges[1]
        for x in ranges[2]
    ]


def _contains_expanded_tile(
    row: Any,
    tile_origin: tuple[int, int, int],
    halo_yx: int,
) -> bool:
    read_origin = (
        tile_origin[0],
        tile_origin[1] - halo_yx,
        tile_origin[2] - halo_yx,
    )
    read_end = (
        tile_origin[0] + TILE_SHAPE[0],
        tile_origin[1] + TILE_SHAPE[1] + halo_yx,
        tile_origin[2] + TILE_SHAPE[2] + halo_yx,
    )
    row_end = tuple(
        start + size for start, size in zip(row.origin_zyx, row.shape_zyx, strict=True)
    )
    return all(
        row_start <= read_start and read_stop <= row_stop
        for row_start, row_stop, read_start, read_stop in zip(
            row.origin_zyx,
            row_end,
            read_origin,
            read_end,
            strict=True,
        )
    )


def _tile_owners(
    rows: list[Any],
    halo_yx: int,
    *,
    record_id: str,
) -> dict[tuple[int, int, int], list[int]]:
    central_owners: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.record_id != record_id:
            continue
        for tile_origin in _aligned_tile_origins(row.origin_zyx, row.shape_zyx):
            central_owners[tile_origin].append(index)
    return {
        origin: [
            index
            for index in indices
            if _contains_expanded_tile(rows[index], origin, halo_yx)
        ]
        for origin, indices in central_owners.items()
        if any(_contains_expanded_tile(rows[index], origin, halo_yx) for index in indices)
    }


def _present_tiles(path: Path) -> set[tuple[int, int, int]]:
    return {
        tuple(int(value) for value in row["origin_zyx"])
        for row in _read_jsonl(path)
        if bool(row["present"])
    }


def _choose_tiles(
    values: list[tuple[int, int, int]], maximum_tiles: int | None
) -> list[tuple[int, int, int]]:
    if maximum_tiles is None or maximum_tiles >= len(values):
        return values
    if maximum_tiles <= 0:
        raise ValueError("maximum_tiles must be positive")
    indices = np.linspace(0, len(values) - 1, maximum_tiles, dtype=np.int64)
    return [values[int(index)] for index in indices]


def _touches_border(mask: np.ndarray, margin: int) -> bool:
    if margin <= 0 or not bool(mask.any()):
        return False
    return bool(
        mask[:margin].any()
        or mask[-margin:].any()
        or mask[:, :margin].any()
        or mask[:, -margin:].any()
    )


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest = Path(args.patches).resolve()
    catalog_path = Path(args.atlas_catalog).resolve()
    rows = load_patch_manifest(manifest)
    if any(row.split != "train" for row in rows):
        raise ValueError("bridge audit requires a training-only manifest")
    if any(not row.has_baseline for row in rows):
        raise ValueError("every bridge-audit row requires an M7 baseline")

    catalog = json.loads(catalog_path.read_text())
    source = catalog["sources"][args.record_id]
    atlas_state_path = Path(source["atlas_state"]).resolve()
    medial_state_path = Path(source["medial_state"]).resolve()
    atlas_state = json.loads(atlas_state_path.read_text())
    medial_state = json.loads(medial_state_path.read_text())
    if atlas_state["state"] != "complete" or medial_state["state"] != "complete":
        raise ValueError("teacher or medial atlas is incomplete")

    target_row_count = sum(row.record_id == args.record_id for row in rows)
    if target_row_count == 0:
        raise ValueError("requested bridge record has no training rows")
    owners = _tile_owners(rows, args.halo_yx, record_id=args.record_id)
    occupancy_tiles = _present_tiles(Path(atlas_state["tile_inventory"]))
    medial_tiles = _present_tiles(Path(medial_state["tile_inventory"]))
    eligible = sorted(set(owners) & occupancy_tiles & medial_tiles)
    selected = _choose_tiles(eligible, args.maximum_tiles)

    q_volume = open_volume(str(source["teacher_q"]))
    valid_volume = open_volume(str(source["target_valid"]))
    crest_volume = open_volume(str(source["teacher_crest"]))
    crest_valid_volume = open_volume(str(source["teacher_crest_valid"]))
    baseline_spec = DenseFieldSpec.from_dict(
        source["coarse_baseline"],
        context="bridge-audit baseline",
        base=catalog_path.parent,
    )
    baseline_volume = open_volume(baseline_spec.volume)

    options = PinnedMedialBridgeOptions()
    screen_rejections: Counter[str] = Counter()
    geometry_rejections: Counter[str] = Counter()
    event_rows: list[dict[str, Any]] = []
    route_owner_rows: set[int] = set()
    interior_route_owner_rows: set[int] = set()
    qualified_slices = 0
    interior_slices = 0
    supervised_voxels = 0
    interior_supervised_voxels = 0
    dilation_counts: Counter[int] = Counter()
    halo = int(args.halo_yx)
    read_shape = (64, 64 + 2 * halo, 64 + 2 * halo)
    center = (slice(None), slice(halo, halo + 64), slice(halo, halo + 64))

    for tile_index, tile_origin in enumerate(selected):
        read_origin = (tile_origin[0], tile_origin[1] - halo, tile_origin[2] - halo)
        if not any(
            _contains_expanded_tile(rows[index], tile_origin, halo)
            for index in owners[tile_origin]
        ):
            raise RuntimeError("teacher read escaped every training row")
        q = read_crop(q_volume, read_origin, read_shape).astype(np.uint8, copy=False)
        valid = read_crop(valid_volume, read_origin, read_shape) > 0
        crest = read_crop(crest_volume, read_origin, read_shape) > 0
        crest_valid = read_crop(crest_valid_volume, read_origin, read_shape) > 0
        baseline_raw = read_crop(baseline_volume, read_origin, read_shape)
        m7 = decode_dense_field(baseline_raw, baseline_spec) >= baseline_spec.threshold
        teacher = (q >= 128) | crest

        q = q[center]
        domain = (valid & crest_valid)[center]
        crest = crest[center]
        m7 = m7[center]
        teacher = teacher[center]
        for z_index in range(64):
            result = extract_pinned_medial_bridge(
                m7=m7[z_index],
                teacher=teacher[z_index],
                centers=crest[z_index],
                valid=domain[z_index],
                teacher_confidence=q[z_index].astype(np.float32) / 255.0,
                options=options,
            )
            if not bool(result.screen["qualified"]):
                screen_rejections[str(result.screen["rejection"])] += 1
                continue
            if not result.qualified:
                geometry_rejections[str(result.rejection)] += 1
                continue
            qualified_slices += 1
            count = int(np.count_nonzero(result.supervision))
            supervised_voxels += count
            assert result.corridor_dilation is not None
            dilation_counts[result.corridor_dilation] += 1
            route_owner_rows.update(owners[tile_origin])
            border = _touches_border(result.supervision, args.border_margin)
            if not border:
                interior_slices += 1
                interior_supervised_voxels += count
                interior_route_owner_rows.update(owners[tile_origin])
            event_rows.append(
                {
                    "tile_origin_zyx": list(tile_origin),
                    "global_z": tile_origin[0] + z_index,
                    "supervision_voxels": count,
                    "route_voxels": int(np.count_nonzero(result.route)),
                    "corridor_dilation": result.corridor_dilation,
                    "m7_components_joined": len(result.reference_ids),
                    "supervision_touches_border": border,
                    "owner_row_count": len(owners[tile_origin]),
                }
            )
        if (tile_index + 1) % 50 == 0 or tile_index + 1 == len(selected):
            print(
                f"audited {tile_index + 1}/{len(selected)} training-only tiles; "
                f"qualified={qualified_slices} interior={interior_slices}",
                flush=True,
            )

    sampler = StratifiedEpochPartitionSampler(
        rows,
        args.schedule_samples,
        args.schedule_seed,
        total_samples=args.schedule_samples,
    )
    scheduled = list(sampler)
    scheduled_route_rows = sum(index in route_owner_rows for index in scheduled)
    scheduled_interior_rows = sum(
        index in interior_route_owner_rows for index in scheduled
    )
    payload = {
        "schema": AUDIT_SCHEMA,
        "state": "complete",
        "changes_training": False,
        "bridge_contract": PINNED_MEDIAL_BRIDGE_CONTRACT,
        "inputs": {
            "patches": str(manifest),
            "patches_sha256": _sha256(manifest),
            "atlas_catalog": str(catalog_path),
            "atlas_catalog_sha256": _sha256(catalog_path),
            "atlas_state": str(atlas_state_path),
            "atlas_state_sha256": _sha256(atlas_state_path),
            "medial_state": str(medial_state_path),
            "medial_state_sha256": _sha256(medial_state_path),
            "record_id": args.record_id,
        },
        "split_contract": {
            "construction_inputs": "training-manifest-boxes-only",
            "coordinate_frame_bound_to_record_id": args.record_id,
            "tile_shape_zyx": list(TILE_SHAPE),
            "read_halo_yx": halo,
            "expanded_tile_must_be_contained_in_training_row": True,
            "heldout_gate_used_for_construction": False,
        },
        "options": {
            "bridge": {
                "screen": asdict(options.screen),
                "maximum_corridor_dilation": options.maximum_corridor_dilation,
            },
            "border_margin": args.border_margin,
            "maximum_tiles": args.maximum_tiles,
            "schedule_seed": args.schedule_seed,
            "schedule_samples": args.schedule_samples,
        },
        "population": {
            "training_rows": len(rows),
            "target_record_training_rows": target_row_count,
            "training_contained_tiles": len(owners),
            "teacher_present_eligible_tiles": len(eligible),
            "audited_tiles": len(selected),
            "audited_slices": len(selected) * 64,
        },
        "summary": {
            "qualified_bridge_slices": qualified_slices,
            "interior_bridge_slices": interior_slices,
            "supervised_voxels": supervised_voxels,
            "interior_supervised_voxels": interior_supervised_voxels,
            "route_bearing_training_rows": len(route_owner_rows),
            "interior_route_bearing_training_rows": len(interior_route_owner_rows),
            "scheduled_rows": len(scheduled),
            "scheduled_route_bearing_rows": scheduled_route_rows,
            "scheduled_interior_route_bearing_rows": scheduled_interior_rows,
            "screen_rejections": dict(sorted(screen_rejections.items())),
            "geometry_rejections": dict(sorted(geometry_rejections.items())),
            "corridor_dilation_counts": {
                str(key): value for key, value in sorted(dilation_counts.items())
            },
        },
        "training_scope": {
            "route_owner_patch_ids": sorted(
                rows[index].patch_id for index in route_owner_rows
            ),
            "interior_route_owner_patch_ids": sorted(
                rows[index].patch_id for index in interior_route_owner_rows
            ),
        },
        "events": event_rows,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit M7-pinned teacher-medial bridge targets in training boxes"
    )
    parser.add_argument("--patches", required=True)
    parser.add_argument("--atlas-catalog", required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--halo-yx", type=int, default=4)
    parser.add_argument("--border-margin", type=int, default=2)
    parser.add_argument("--maximum-tiles", type=int)
    parser.add_argument("--schedule-seed", type=int, default=1203)
    parser.add_argument("--schedule-samples", type=int, default=1024)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.halo_yx < 0 or args.border_margin < 0:
        raise ValueError("halo and border margin must be non-negative")
    payload = run_audit(args)
    print(json.dumps(payload["summary"], indent=2))
    print(Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
