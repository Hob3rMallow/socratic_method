#!/usr/bin/env python3
"""Materialize training-only dynamic medial-connectivity corridors and pins."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc
from scipy import ndimage

from crossres_pred.voxel.growth_sentinels import STRUCTURE_8
from crossres_pred.voxel.io import decode_dense_field, open_volume, read_crop
from crossres_pred.voxel.medial_bridges import (
    DYNAMIC_MEDIAL_CONNECTIVITY_ATLAS_SCHEMA,
    DYNAMIC_MEDIAL_CONNECTIVITY_CONTRACT,
    PINNED_MEDIAL_BRIDGE_CONTRACT,
    extract_pinned_medial_bridge,
)
from crossres_pred.voxel.patches import load_patch_manifest
from crossres_pred.voxel.schema import DenseFieldSpec
from crossres_pred.voxel.train import StratifiedEpochPartitionSampler

AUDIT_SCHEMA = "crossres-training-pinned-medial-bridge-audit-v2"
INVENTORY_SCHEMA = "crossres-training-dynamic-medial-connectivity-tile-v1"
TILE_SHAPE = (64, 64, 64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _pin_bits(pin_membership: np.ndarray) -> tuple[int, ...]:
    union = int(np.bitwise_or.reduce(pin_membership.ravel(), initial=np.uint8(0)))
    return tuple(bit for bit in range(8) if union & (1 << bit))


def _required_connectivity_steps(
    corridor: np.ndarray,
    pin_membership: np.ndarray,
    *,
    maximum_steps: int,
) -> int | None:
    bits = _pin_bits(pin_membership)
    if len(bits) < 2 or bits[0] != 0:
        return None
    reached = (pin_membership & np.uint8(1)) > 0
    targets = [(pin_membership & np.uint8(1 << bit)) > 0 for bit in bits[1:]]
    for step in range(maximum_steps + 1):
        if all(bool((reached & target).any()) for target in targets):
            return step
        reached = ndimage.binary_dilation(reached, structure=STRUCTURE_8) & corridor
    return None


def _create_array(path: Path, shape: tuple[int, ...], dtype: Any) -> Any:
    return zarr.open_array(
        str(path),
        mode="w",
        zarr_format=2,
        shape=shape,
        chunks=TILE_SHAPE,
        dtype=dtype,
        fill_value=0,
        compressor=Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE),
        dimension_separator="/",
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    audit_path = Path(args.audit).resolve()
    output = Path(args.output).resolve()
    state_path = output / "connectivity_state.json"
    if output.exists():
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if (
                state.get("schema") == DYNAMIC_MEDIAL_CONNECTIVITY_ATLAS_SCHEMA
                and state.get("state") == "complete"
                and state.get("identity", {}).get("audit_sha256") == _sha256(audit_path)
                and state.get("maximum_propagation_steps")
                == args.maximum_propagation_steps
            ):
                return state
        raise FileExistsError(
            f"refusing to replace incomplete or different connectivity atlas: {output}"
        )

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    identity = audit.get("inputs", {})
    split = audit.get("split_contract", {})
    if (
        audit.get("schema") != AUDIT_SCHEMA
        or audit.get("state") != "complete"
        or audit.get("changes_training") is not False
        or audit.get("bridge_contract") != PINNED_MEDIAL_BRIDGE_CONTRACT
        or split.get("heldout_gate_used_for_construction") is not False
        or split.get("coordinate_frame_bound_to_record_id") != identity.get("record_id")
    ):
        raise ValueError("input is not the owner-bound training-only bridge audit")

    manifest_path = Path(identity["patches"])
    catalog_path = Path(identity["atlas_catalog"])
    if (
        _sha256(manifest_path) != identity["patches_sha256"]
        or _sha256(catalog_path) != identity["atlas_catalog_sha256"]
    ):
        raise ValueError("connectivity-atlas parent inputs changed")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    source = catalog["sources"][identity["record_id"]]
    baseline_spec = DenseFieldSpec.from_dict(
        source["coarse_baseline"],
        context="connectivity-atlas baseline",
        base=catalog_path.parent,
    )
    arrays = {
        "q": open_volume(str(source["teacher_q"])),
        "valid": open_volume(str(source["target_valid"])),
        "crest": open_volume(str(source["teacher_crest"])),
        "crest_valid": open_volume(str(source["teacher_crest_valid"])),
        "m7": open_volume(baseline_spec.volume),
    }
    shape = tuple(int(value) for value in arrays["q"].shape)
    if any(
        tuple(int(value) for value in array.shape) != shape for array in arrays.values()
    ):
        raise ValueError("connectivity-atlas source arrays differ in shape")

    selected_events = [
        event
        for event in audit["events"]
        if not bool(event["supervision_touches_border"])
    ]
    selected_events.sort(
        key=lambda event: (
            tuple(int(value) for value in event["tile_origin_zyx"]),
            int(event["global_z"]),
        )
    )
    if len(selected_events) != int(audit["summary"]["interior_bridge_slices"]):
        raise ValueError("audit interior-event count changed")
    audit_eligible_patch_ids = audit.get("training_scope", {}).get(
        "interior_route_owner_patch_ids"
    )
    if (
        not isinstance(audit_eligible_patch_ids, list)
        or not all(
            isinstance(value, str) and value for value in audit_eligible_patch_ids
        )
        or len(audit_eligible_patch_ids) != len(set(audit_eligible_patch_ids))
        or len(audit_eligible_patch_ids)
        != int(audit["summary"]["interior_route_bearing_training_rows"])
    ):
        raise ValueError("audit has an invalid owner-bound patch scope")
    if len(selected_events) >= np.iinfo(np.uint16).max:
        raise ValueError("uint16 connectivity event identifiers are exhausted")

    by_tile: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in selected_events:
        by_tile[tuple(int(value) for value in event["tile_origin_zyx"])].append(event)

    output.mkdir(parents=True)
    event_path = output / "event_ids.zarr"
    pin_path = output / "pin_membership.zarr"
    free_path = output / "free_anchors.zarr"
    event_array = _create_array(event_path, shape, np.uint16)
    pin_array = _create_array(pin_path, shape, np.uint8)
    free_array = _create_array(free_path, shape, np.uint8)

    inventory_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    next_identifier = 1
    total_corridor_voxels = 0
    total_pin_voxels = 0
    total_free_voxels = 0
    required_steps: list[int] = []
    for tile_origin, events in sorted(by_tile.items()):
        q = read_crop(arrays["q"], tile_origin, TILE_SHAPE).astype(np.uint8, copy=False)
        valid = read_crop(arrays["valid"], tile_origin, TILE_SHAPE) > 0
        crest = read_crop(arrays["crest"], tile_origin, TILE_SHAPE) > 0
        crest_valid = read_crop(arrays["crest_valid"], tile_origin, TILE_SHAPE) > 0
        baseline_raw = read_crop(arrays["m7"], tile_origin, TILE_SHAPE)
        m7 = decode_dense_field(baseline_raw, baseline_spec) >= baseline_spec.threshold
        domain = valid & crest_valid
        teacher = ((q >= 128) | crest) & domain
        m7 &= domain

        tile_events = np.zeros(TILE_SHAPE, dtype=np.uint16)
        tile_pins = np.zeros(TILE_SHAPE, dtype=np.uint8)
        tile_free = np.zeros(TILE_SHAPE, dtype=np.uint8)
        tile_event_ids: list[int] = []
        for event in events:
            z_index = int(event["global_z"]) - tile_origin[0]
            result = extract_pinned_medial_bridge(
                m7=m7[z_index],
                teacher=teacher[z_index],
                centers=crest[z_index],
                valid=domain[z_index],
                teacher_confidence=q[z_index].astype(np.float32) / 255.0,
            )
            if not result.qualified:
                raise ValueError("audited connectivity event no longer qualifies")
            if bool(tile_events[z_index][result.corridor].any()):
                raise ValueError("two dynamic connectivity corridors overlap")
            bits = _pin_bits(result.pin_membership)
            if len(bits) != len(result.reference_ids) or bits != tuple(
                range(len(bits))
            ):
                raise ValueError("connectivity pin membership is incomplete")
            steps = _required_connectivity_steps(
                result.corridor,
                result.pin_membership,
                maximum_steps=args.maximum_propagation_steps,
            )
            if steps is None:
                raise ValueError(
                    "teacher-medial corridor does not connect every pin within the "
                    "configured propagation limit"
                )

            identifier = next_identifier
            next_identifier += 1
            tile_events[z_index][result.corridor] = identifier
            tile_pins[z_index][result.corridor] = result.pin_membership[result.corridor]
            tile_free[z_index][result.free_anchors] = 1
            coordinates = np.argwhere(result.corridor)
            corridor_count = int(np.count_nonzero(result.corridor))
            pin_count = int(np.count_nonzero(result.pin_membership))
            free_count = int(np.count_nonzero(result.free_anchors))
            local_bbox_yx = [
                [
                    int(coordinates[:, 0].min()),
                    int(coordinates[:, 0].max()) + 1,
                ],
                [
                    int(coordinates[:, 1].min()),
                    int(coordinates[:, 1].max()) + 1,
                ],
            ]
            event_rows.append(
                {
                    "event_id": identifier,
                    "tile_origin_zyx": list(tile_origin),
                    "global_z": int(event["global_z"]),
                    "pin_count": len(bits),
                    "corridor_voxels": corridor_count,
                    "pin_voxels": pin_count,
                    "free_anchor_voxels": free_count,
                    "required_connectivity_steps": steps,
                    "corridor_dilation": int(event["corridor_dilation"]),
                    "local_bbox_yx": local_bbox_yx,
                    "global_bbox_zyx": [
                        [int(event["global_z"]), int(event["global_z"]) + 1],
                        [
                            tile_origin[1] + local_bbox_yx[0][0],
                            tile_origin[1] + local_bbox_yx[0][1],
                        ],
                        [
                            tile_origin[2] + local_bbox_yx[1][0],
                            tile_origin[2] + local_bbox_yx[1][1],
                        ],
                    ],
                }
            )
            tile_event_ids.append(identifier)
            total_corridor_voxels += corridor_count
            total_pin_voxels += pin_count
            total_free_voxels += free_count
            required_steps.append(steps)

        slices = tuple(
            slice(start, min(start + size, limit))
            for start, size, limit in zip(tile_origin, TILE_SHAPE, shape, strict=True)
        )
        stored_shape = tuple(section.stop - section.start for section in slices)
        local = tuple(slice(0, size) for size in stored_shape)
        event_array[slices] = tile_events[local]
        pin_array[slices] = tile_pins[local]
        free_array[slices] = tile_free[local]
        inventory_rows.append(
            {
                "schema": INVENTORY_SCHEMA,
                "tile_origin_zyx": list(tile_origin),
                "tile_shape_zyx": list(stored_shape),
                "event_ids": tile_event_ids,
                "corridor_voxels": int(np.count_nonzero(tile_events)),
                "event_ids_decoded_sha256": _array_sha256(tile_events),
                "pin_membership_decoded_sha256": _array_sha256(tile_pins),
                "free_anchors_decoded_sha256": _array_sha256(tile_free),
            }
        )

    training_rows = [
        row for row in load_patch_manifest(manifest_path) if row.split == "train"
    ]
    rows_by_id = {row.patch_id: row for row in training_rows}
    record_id = str(identity["record_id"])
    if any(
        patch_id not in rows_by_id or rows_by_id[patch_id].record_id != record_id
        for patch_id in audit_eligible_patch_ids
    ):
        raise ValueError("audit owner patches differ from the training manifest")
    patch_event_ids: dict[str, list[int]] = {}
    owner_counts = {int(event["event_id"]): 0 for event in event_rows}
    for patch_id in sorted(audit_eligible_patch_ids):
        row = rows_by_id[patch_id]
        patch_stops = tuple(
            origin + size
            for origin, size in zip(row.origin_zyx, row.shape_zyx, strict=True)
        )
        contained = []
        for event in event_rows:
            bounds = event["global_bbox_zyx"]
            if all(
                row.origin_zyx[axis] <= int(bounds[axis][0])
                and int(bounds[axis][1]) <= patch_stops[axis]
                for axis in range(3)
            ):
                identifier = int(event["event_id"])
                contained.append(identifier)
                owner_counts[identifier] += 1
        if contained:
            patch_event_ids[patch_id] = contained
    orphaned = [identifier for identifier, count in owner_counts.items() if count == 0]
    if orphaned:
        raise ValueError(
            "dynamic connectivity events have no full-corridor owner patches: "
            f"{orphaned[:10]}"
        )
    eligible_patch_ids = sorted(patch_event_ids)
    schedule_seed = int(audit["options"]["schedule_seed"])
    schedule_samples = int(audit["options"]["schedule_samples"])
    schedule = list(
        StratifiedEpochPartitionSampler(
            training_rows,
            schedule_samples,
            schedule_seed,
            total_samples=schedule_samples,
        )
    )
    if len(schedule) != schedule_samples:
        raise RuntimeError("exact connectivity schedule has the wrong length")
    scheduled_event_rows = sum(
        training_rows[index].patch_id in patch_event_ids for index in schedule
    )

    inventory_path = output / "connectivity_tiles.jsonl"
    inventory_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in inventory_rows),
        encoding="utf-8",
    )
    events_path = output / "connectivity_events.json"
    _write_json(events_path, event_rows)
    state = {
        "schema": DYNAMIC_MEDIAL_CONNECTIVITY_ATLAS_SCHEMA,
        "state": "complete",
        "identity": {
            "contract": DYNAMIC_MEDIAL_CONNECTIVITY_CONTRACT,
            "parent_bridge_contract": PINNED_MEDIAL_BRIDGE_CONTRACT,
            "audit": str(audit_path),
            "audit_sha256": _sha256(audit_path),
            "training_manifest": str(manifest_path),
            "training_manifest_sha256": _sha256(manifest_path),
            "atlas_catalog": str(catalog_path),
            "atlas_catalog_sha256": _sha256(catalog_path),
            "record_id": identity["record_id"],
            "construction_inputs": "training-manifest-boxes-only",
            "heldout_gate_used_for_construction": False,
        },
        "event_ids": str(event_path),
        "pin_membership": str(pin_path),
        "free_anchors": str(free_path),
        "events": str(events_path),
        "events_sha256": _sha256(events_path),
        "tile_inventory": str(inventory_path),
        "tile_inventory_sha256": _sha256(inventory_path),
        "shape_zyx": list(shape),
        "chunks_zyx": list(TILE_SHAPE),
        "dtypes": {
            "event_ids": "uint16",
            "pin_membership": "uint8-bitset",
            "free_anchors": "uint8-binary",
        },
        "eligible_patch_ids": eligible_patch_ids,
        "patch_event_ids": patch_event_ids,
        "event_count": len(event_rows),
        "fully_owned_event_count": sum(count > 0 for count in owner_counts.values()),
        "full_owner_observations": sum(owner_counts.values()),
        "corridor_voxels": total_corridor_voxels,
        "pin_voxels": total_pin_voxels,
        "free_anchor_voxels": total_free_voxels,
        "present_tiles": len(inventory_rows),
        "maximum_propagation_steps": args.maximum_propagation_steps,
        "maximum_required_connectivity_steps": max(required_steps),
        "exact_schedule": {
            "seed": schedule_seed,
            "samples": schedule_samples,
            "event_bearing_rows": scheduled_event_rows,
        },
    }
    _write_json(state_path, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-propagation-steps", type=int, default=96)
    args = parser.parse_args()
    if args.maximum_propagation_steps <= 0:
        raise ValueError("maximum propagation steps must be positive")
    state = build(args)
    print(
        json.dumps(
            {
                key: state[key]
                for key in (
                    "event_count",
                    "corridor_voxels",
                    "pin_voxels",
                    "free_anchor_voxels",
                    "maximum_required_connectivity_steps",
                )
            },
            indent=2,
        )
    )
    print(Path(args.output).resolve() / "connectivity_state.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
