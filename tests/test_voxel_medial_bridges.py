from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import zarr

from crossres_pred.voxel.growth_sentinels import SliceScreenOptions
from crossres_pred.voxel.medial_bridges import (
    DYNAMIC_MEDIAL_CONNECTIVITY_ATLAS_SCHEMA,
    PINNED_MEDIAL_BRIDGE_ATLAS_SCHEMA,
    PinnedMedialBridgeOptions,
    extract_pinned_medial_bridge,
)
from crossres_pred.voxel.patches import VoxelPatchDataset


def _options(**changes: object) -> PinnedMedialBridgeOptions:
    values = {
        "minimum_component_voxels": 2,
        "minimum_valid_fraction": 0.95,
        "maximum_radius": 3.0,
        "maximum_interior_fraction_r2": 0.02,
        "contact_radius": 2,
        "minimum_missing_join_voxels": 2,
    }
    values.update(changes)
    return PinnedMedialBridgeOptions(screen=SliceScreenOptions(**values))


def test_extracts_novel_center_route_between_two_m7_pins() -> None:
    m7 = np.zeros((32, 32), dtype=bool)
    teacher = np.zeros_like(m7)
    teacher[16, 4:28] = True
    centers = teacher.copy()
    m7[16, 4:10] = True
    m7[16, 22:28] = True

    result = extract_pinned_medial_bridge(
        m7=m7,
        teacher=teacher,
        centers=centers,
        valid=np.ones_like(m7),
        options=_options(),
    )

    assert result.qualified
    assert result.reference_ids == (1, 2)
    assert result.corridor_dilation == 0
    assert np.all(result.route <= teacher)
    assert np.all(result.supervision <= centers)
    assert np.count_nonzero(result.supervision) >= 2
    assert not np.any(result.supervision & m7)
    assert set(np.unique(result.pin_membership)) == {0, 1, 2}
    assert np.all((result.pin_membership > 0) <= result.free_anchors)
    assert np.all(result.free_anchors <= result.corridor)


def test_route_prefers_longer_medial_center_over_short_off_axis_shortcut() -> None:
    m7 = np.zeros((24, 24), dtype=bool)
    teacher = np.zeros_like(m7)
    teacher[4:17, 3:21] = True
    centers = np.zeros_like(m7)
    centers[4:17, 4] = True
    centers[4:17, 19] = True
    centers[16, 4:20] = True
    m7[4:7, 3:6] = True
    m7[4:7, 18:21] = True

    result = extract_pinned_medial_bridge(
        m7=m7,
        teacher=teacher,
        centers=centers,
        valid=np.ones_like(m7),
        options=_options(
            maximum_radius=20.0,
            maximum_interior_fraction_r2=1.0,
        ),
    )

    assert result.qualified
    assert np.count_nonzero(result.route & centers) >= 20
    assert not np.any(result.route[8:14, 8:16])


def test_three_m7_pins_are_joined_by_a_minimum_spanning_route() -> None:
    m7 = np.zeros((32, 40), dtype=bool)
    teacher = np.zeros_like(m7)
    teacher[16, 3:37] = True
    centers = teacher.copy()
    m7[16, 3:8] = True
    m7[16, 17:22] = True
    m7[16, 32:37] = True

    result = extract_pinned_medial_bridge(
        m7=m7,
        teacher=teacher,
        centers=centers,
        valid=np.ones_like(m7),
        options=_options(),
    )

    assert result.qualified
    assert result.reference_ids == (1, 2, 3)
    assert result.route[16, 10]
    assert result.route[16, 29]
    assert int(np.bitwise_or.reduce(result.pin_membership.ravel())) == 0b111


def test_blob_geometry_is_rejected_before_route_extraction() -> None:
    m7 = np.zeros((32, 32), dtype=bool)
    teacher = np.zeros_like(m7)
    teacher[8:24, 4:28] = True
    centers = np.zeros_like(m7)
    centers[16, 4:28] = True
    m7[8:24, 4:8] = True
    m7[8:24, 24:28] = True

    result = extract_pinned_medial_bridge(
        m7=m7,
        teacher=teacher,
        centers=centers,
        valid=np.ones_like(m7),
        options=_options(),
    )

    assert not result.qualified
    assert result.rejection == "radius-blob-gate"
    assert not result.route.any()


def test_dataset_loads_provenance_bound_pinned_bridge_ids(tmp_path: Path) -> None:
    shape = (8, 8, 8)
    patch = tmp_path / "patch.npz"
    np.savez(
        patch,
        image=np.zeros(shape, dtype=np.uint8),
        target_u8=np.zeros(shape, dtype=np.uint8),
    )
    manifest = tmp_path / "patches.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema": "crossres-voxel-patch-v1",
                "patch_id": "train",
                "path": str(patch),
                "record_id": "record",
                "scroll_id": "PHerc0139",
                "split": "train",
                "origin_zyx": [0, 0, 0],
                "shape_zyx": list(shape),
                "known_fraction": 1.0,
                "positive_fraction_known": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bridge_path = tmp_path / "bridge_ids.zarr"
    bridge = zarr.open_array(
        str(bridge_path),
        mode="w",
        zarr_format=2,
        shape=shape,
        chunks=(4, 4, 4),
        dtype=np.uint16,
        fill_value=0,
    )
    bridge[2, 3, 4:7] = 7
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    state = tmp_path / "bridge_state.json"
    state.write_text(
        json.dumps(
            {
                "schema": PINNED_MEDIAL_BRIDGE_ATLAS_SCHEMA,
                "state": "complete",
                "identity": {
                    "training_manifest": str(manifest.resolve()),
                    "training_manifest_sha256": manifest_sha256,
                    "construction_inputs": "training-manifest-boxes-only",
                    "heldout_gate_used_for_construction": False,
                    "record_id": "record",
                },
                "bridge_ids": str(bridge_path.resolve()),
                "shape_zyx": list(shape),
                "chunks_zyx": [4, 4, 4],
                "dtype": "uint16",
                "eligible_patch_ids": ["train"],
            }
        ),
        encoding="utf-8",
    )

    sample = VoxelPatchDataset(
        manifest,
        split="train",
        pinned_medial_bridge_state=state,
    )[0]

    ids = sample["pinned_medial_bridge"]
    assert ids.dtype == torch.int64
    assert ids.shape == (1, *shape)
    assert ids[0, 2, 3, 4:7].tolist() == [7, 7, 7]
    assert int((ids != 0).sum()) == 3


def test_dataset_never_applies_bridge_coordinates_to_another_record(
    tmp_path: Path,
) -> None:
    shape = (8, 8, 8)
    rows = []
    for record_id in ("target-record", "other-record"):
        patch = tmp_path / f"{record_id}.npz"
        np.savez(
            patch,
            image=np.zeros(shape, dtype=np.uint8),
            target_u8=np.zeros(shape, dtype=np.uint8),
        )
        rows.append(
            {
                "schema": "crossres-voxel-patch-v1",
                "patch_id": record_id,
                "path": str(patch),
                "record_id": record_id,
                "scroll_id": record_id,
                "split": "train",
                "origin_zyx": [0, 0, 0],
                "shape_zyx": list(shape),
                "known_fraction": 1.0,
                "positive_fraction_known": 0.0,
            }
        )
    manifest = tmp_path / "patches.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    bridge_path = tmp_path / "bridge_ids.zarr"
    bridge = zarr.open_array(
        str(bridge_path),
        mode="w",
        zarr_format=2,
        shape=shape,
        chunks=(4, 4, 4),
        dtype=np.uint16,
        fill_value=0,
    )
    bridge[2, 3, 4:7] = 7
    state = tmp_path / "bridge_state.json"
    state.write_text(
        json.dumps(
            {
                "schema": PINNED_MEDIAL_BRIDGE_ATLAS_SCHEMA,
                "state": "complete",
                "identity": {
                    "training_manifest": str(manifest.resolve()),
                    "training_manifest_sha256": hashlib.sha256(
                        manifest.read_bytes()
                    ).hexdigest(),
                    "construction_inputs": "training-manifest-boxes-only",
                    "heldout_gate_used_for_construction": False,
                    "record_id": "target-record",
                },
                "bridge_ids": str(bridge_path.resolve()),
                "shape_zyx": list(shape),
                "chunks_zyx": [4, 4, 4],
                "dtype": "uint16",
                "eligible_patch_ids": ["target-record"],
            }
        ),
        encoding="utf-8",
    )
    dataset = VoxelPatchDataset(
        manifest,
        split="train",
        pinned_medial_bridge_state=state,
    )

    target_ids = dataset[0]["pinned_medial_bridge"]
    other_ids = dataset[1]["pinned_medial_bridge"]

    assert int((target_ids != 0).sum()) == 3
    assert not bool(other_ids.any())


def test_dataset_loads_dynamic_connectivity_fields_together(tmp_path: Path) -> None:
    patch_shape = (8, 8, 8)
    atlas_shape = (8, 8, 12)
    patches = [tmp_path / "patch-a.npz", tmp_path / "patch-b.npz"]
    for patch in patches:
        np.savez(
            patch,
            image=np.zeros(patch_shape, dtype=np.uint8),
            target_u8=np.zeros(patch_shape, dtype=np.uint8),
        )
    manifest = tmp_path / "patches.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "schema": "crossres-voxel-patch-v1",
                    "patch_id": f"train-{label}",
                    "path": str(patch),
                    "record_id": "record",
                    "scroll_id": "PHerc0139",
                    "split": "train",
                    "origin_zyx": [0, 0, offset],
                    "shape_zyx": list(patch_shape),
                    "known_fraction": 1.0,
                    "positive_fraction_known": 0.0,
                }
            )
            + "\n"
            for label, offset, patch in zip(("a", "b"), (0, 4), patches, strict=True)
        ),
        encoding="utf-8",
    )
    paths = {
        "event_ids": tmp_path / "event_ids.zarr",
        "pin_membership": tmp_path / "pin_membership.zarr",
        "free_anchors": tmp_path / "free_anchors.zarr",
    }
    arrays = {
        name: zarr.open_array(
            str(path),
            mode="w",
            zarr_format=2,
            shape=atlas_shape,
            chunks=(4, 4, 4),
            dtype=np.uint16 if name == "event_ids" else np.uint8,
            fill_value=0,
        )
        for name, path in paths.items()
    }
    arrays["event_ids"][2, 3, 1:7] = 1
    arrays["pin_membership"][2, 3, 1] = 1
    arrays["pin_membership"][2, 3, 6] = 2
    arrays["free_anchors"][2, 3, [1, 6]] = 1
    arrays["event_ids"][4, 5, 8:11] = 2
    arrays["pin_membership"][4, 5, 8] = 1
    arrays["pin_membership"][4, 5, 10] = 2
    arrays["free_anchors"][4, 5, [8, 10]] = 1
    state = tmp_path / "connectivity_state.json"
    state.write_text(
        json.dumps(
            {
                "schema": DYNAMIC_MEDIAL_CONNECTIVITY_ATLAS_SCHEMA,
                "state": "complete",
                "identity": {
                    "training_manifest": str(manifest.resolve()),
                    "training_manifest_sha256": hashlib.sha256(
                        manifest.read_bytes()
                    ).hexdigest(),
                    "construction_inputs": "training-manifest-boxes-only",
                    "heldout_gate_used_for_construction": False,
                    "record_id": "record",
                },
                **{name: str(path.resolve()) for name, path in paths.items()},
                "shape_zyx": list(atlas_shape),
                "chunks_zyx": [4, 4, 4],
                "dtypes": {
                    "event_ids": "uint16",
                    "pin_membership": "uint8-bitset",
                    "free_anchors": "uint8-binary",
                },
                "eligible_patch_ids": ["train-a", "train-b"],
                "patch_event_ids": {"train-a": [1], "train-b": [2]},
                "event_count": 2,
                "fully_owned_event_count": 2,
                "maximum_propagation_steps": 96,
                "maximum_required_connectivity_steps": 5,
            }
        ),
        encoding="utf-8",
    )

    samples = VoxelPatchDataset(
        manifest,
        split="train",
        dynamic_medial_connectivity_state=state,
    )
    first = samples[0]
    second = samples[1]

    assert first["dynamic_connectivity_event"][0, 2, 3, 1:7].tolist() == [
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    assert first["dynamic_connectivity_pins"][0, 2, 3, [1, 6]].tolist() == [1, 2]
    assert first["dynamic_connectivity_free"][0, 2, 3, [1, 6]].tolist() == [1, 1]
    assert not bool(second["dynamic_connectivity_event"][0, 2, 3].any())
    assert second["dynamic_connectivity_event"][0, 4, 5, 4:7].tolist() == [2, 2, 2]
    assert second["dynamic_connectivity_pins"][0, 4, 5, [4, 6]].tolist() == [1, 2]


def test_dataset_without_bridge_state_preserves_the_baseline_sample(
    tmp_path: Path,
) -> None:
    shape = (8, 8, 8)
    patch = tmp_path / "patch.npz"
    np.savez(
        patch,
        image=np.zeros(shape, dtype=np.uint8),
        target_u8=np.zeros(shape, dtype=np.uint8),
    )
    manifest = tmp_path / "patches.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema": "crossres-voxel-patch-v1",
                "patch_id": "train",
                "path": str(patch),
                "record_id": "record",
                "scroll_id": "PHerc0139",
                "split": "train",
                "origin_zyx": [0, 0, 0],
                "shape_zyx": list(shape),
                "known_fraction": 1.0,
                "positive_fraction_known": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sample = VoxelPatchDataset(manifest, split="train")[0]

    assert "pinned_medial_bridge" not in sample
    assert "dynamic_connectivity_event" not in sample
