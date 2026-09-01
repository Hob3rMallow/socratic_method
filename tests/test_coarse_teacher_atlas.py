from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from crossres_pred.voxel import coarse_teacher_atlas as atlas
from crossres_pred.voxel.patches import (
    MEDIAL_ATLAS_PATCH_PREPARATION_VERSION,
    VoxelPatchDataset,
)
from crossres_pred.voxel.registration import antialias_fine_target_patch
from crossres_pred.voxel.scrollfiesta_metrics import scrollfiesta_patch_pred_metrics

IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
)


def test_atlas_state_replace_retries_sharing_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_text("ready", encoding="utf-8")
    real_replace = atlas.os.replace
    attempts = 0

    def flaky_replace(left: Path, right: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("simulated reader lock")
        real_replace(left, right)

    monkeypatch.setattr(atlas.os, "replace", flaky_replace)
    monkeypatch.setattr(atlas.time, "sleep", lambda _seconds: None)
    atlas._replace_with_retry(source, destination)

    assert attempts == 3
    assert destination.read_text(encoding="utf-8") == "ready"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _zarr_array(path: Path, value: np.ndarray, chunks: tuple[int, int, int]) -> None:
    import zarr

    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    array = root.create_array("0", shape=value.shape, chunks=chunks, dtype=value.dtype)
    array[:] = value


def test_candidate_tiles_cover_transformed_sparse_chunks() -> None:
    tiles = atlas.candidate_coarse_tiles(
        fine_chunk_coordinates_zyx=[(0, 0, 0), (1, 1, 1)],
        fine_chunks_zyx=(8, 8, 8),
        fine_shape_zyx=(16, 16, 16),
        fine_to_coarse_affine_xyz=IDENTITY,
        coarse_shape_zyx=(16, 16, 16),
        tile_shape_zyx=(8, 8, 8),
        margin_coarse_vox=0,
    )
    assert set(tiles) == {(0, 0, 0), (1, 1, 1)}


def test_antialias_rejects_invalid_cuda_threshold_before_reading() -> None:
    with pytest.raises(ValueError, match="cuda_minimum_output_voxels"):
        antialias_fine_target_patch(  # type: ignore[arg-type]
            None,
            None,
            None,
            IDENTITY,
            (0, 0, 0),
            (8, 8, 8),
            cuda_minimum_output_voxels=0,
        )


def test_build_atlas_commits_sparse_q_and_valid_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fine_root = tmp_path / "fine.zarr"
    coarse_root = tmp_path / "coarse.zarr"
    fine = np.zeros((16, 16, 16), dtype=np.uint8)
    fine[2:6, 2:6, 2:6] = 255
    _zarr_array(fine_root, fine, (8, 8, 8))
    _zarr_array(coarse_root, np.full(fine.shape, 100, dtype=np.uint8), (8, 8, 8))
    inventory = fine_root / "crossres_sparse_objects.jsonl"
    inventory.write_text(
        "\n".join(
            json.dumps({"kind": "chunk", "relative_path": path})
            for path in ("0/0/0/0", "0/1/0/0")
        )
        + "\n",
        encoding="utf-8",
    )
    (fine_root / "teacher_state.json").write_text(
        json.dumps({"state": "complete", "accepted": 2}), encoding="utf-8"
    )
    candidate_chunks = tmp_path / "candidate_chunks.jsonl"
    candidate_chunks.write_text(
        json.dumps({"chunk_zyx": [0, 0, 0]}) + "\n", encoding="utf-8"
    )
    pair_manifest = tmp_path / "pairs.jsonl"
    pair_manifest.write_text(
        json.dumps(
            {
                "schema": "crossres-voxel-pair-v1",
                "schema_version": 1,
                "record_id": "synthetic-native-fine-teacher",
                "scroll_id": "Synthetic",
                "split": "train",
                "supervision_source": "official-native-fine-teacher/test",
                "coarse": {
                    "scan_id": "coarse",
                    "voxel_um": 8.0,
                    "image": f"{coarse_root}::0",
                },
                "fine": {
                    "scan_id": "fine",
                    "voxel_um": 2.0,
                    "target": {
                        "volume": f"{fine_root}::0",
                        "encoding": "labels",
                        "positive_labels": [255],
                        "support": {
                            "kind": "present-chunks",
                            "inventory": str(inventory),
                        },
                    },
                    "to_coarse_affine_xyz": [list(row) for row in IDENTITY],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    observed_support: list[set[tuple[int, int, int]]] = []

    def fake_projector(*args, **kwargs):
        observed_support.append(set(map(tuple, args[2].coordinates())))
        shape = tuple(int(value) for value in args[5])
        q = np.full(shape, 0.75, dtype=np.float32)
        valid = np.ones(shape, dtype=np.uint8)
        hard = np.ones(shape, dtype=np.uint8)
        return (
            hard,
            q,
            valid,
            {
                "projection_backend": "cuda-gauss-hermite3-pullback-linf-validity-v1",
                "known_fraction": 1.0,
                "positive_voxels": int(np.prod(shape)),
            },
        )

    monkeypatch.setattr(atlas, "antialias_fine_target_patch", fake_projector)
    output = tmp_path / "atlas"
    options = atlas.CoarseTeacherAtlasOptions(
        tile_shape_zyx=(16, 16, 16), max_cpu_threads=1
    )
    state_path = atlas.build_coarse_teacher_atlas(
        pair_manifest_path=pair_manifest,
        record_id="synthetic-native-fine-teacher",
        output_path=output,
        options=options,
        candidate_fine_chunks_path=candidate_chunks,
    )
    state = atlas.validate_coarse_teacher_atlas(output)
    assert state_path == output / "atlas_state.json"
    assert state["present_tiles"] == 1
    assert state["identity"]["fine_support_chunks"] == 2
    assert state["identity"]["candidate_fine_chunks"] == 1
    assert state["identity"]["fine_support_policy"] == "candidate-chunks-only"
    assert observed_support == [{(0, 0, 0)}]
    q = atlas.open_volume(str(output / "teacher_q.zarr"))
    valid = atlas.open_volume(str(output / "target_valid.zarr"))
    assert np.all(np.asarray(q[:]) == 191)
    assert np.all(np.asarray(valid[:]) == 1)

    medial_path = atlas.build_coarse_teacher_medial_atlas(
        atlas_path=output,
        options=atlas.CoarseTeacherMedialAtlasOptions(
            max_cpu_threads=1,
            fine_chunk_cache_entries=4,
            medial=atlas.MedialProjectionOptions(
                halo_zyx=(1, 2, 2),
                skeleton_workers=1,
                max_cache_chunks=4,
            ),
        ),
    )
    medial_state = atlas.validate_coarse_teacher_medial_atlas(output)
    crest = atlas.open_volume(str(output / "teacher_crest.zarr"))
    crest_valid = atlas.open_volume(str(output / "teacher_crest_valid.zarr"))
    assert medial_path == output / "medial_state.json"
    assert medial_state["crest_voxels"] > 0
    assert medial_state["known_voxels"] >= medial_state["crest_voxels"]
    assert np.all(np.asarray(crest[:]) <= np.asarray(crest_valid[:]))

    def unexpected_medial_projector(*_args, **_kwargs):
        raise AssertionError("a committed medial atlas batch must not be recomputed")

    monkeypatch.setattr(
        atlas, "project_fine_medial_patch", unexpected_medial_projector
    )
    atlas.build_coarse_teacher_medial_atlas(
        atlas_path=output,
        options=atlas.CoarseTeacherMedialAtlasOptions(
            max_cpu_threads=1,
            fine_chunk_cache_entries=4,
            medial=atlas.MedialProjectionOptions(
                halo_zyx=(1, 2, 2),
                skeleton_workers=1,
                max_cache_chunks=4,
            ),
        ),
    )

    def unexpected_projector(*_args, **_kwargs):
        raise AssertionError("a committed atlas batch must not be recomputed")

    monkeypatch.setattr(atlas, "antialias_fine_target_patch", unexpected_projector)
    atlas.build_coarse_teacher_atlas(
        pair_manifest_path=pair_manifest,
        record_id="synthetic-native-fine-teacher",
        output_path=output,
        options=options,
        candidate_fine_chunks_path=candidate_chunks,
    )

    (output / "rows" / "000000.jsonl").unlink()
    recomputed = 0

    def recompute_projector(*args, **kwargs):
        nonlocal recomputed
        recomputed += 1
        return fake_projector(*args, **kwargs)

    monkeypatch.setattr(atlas, "antialias_fine_target_patch", recompute_projector)
    atlas.build_coarse_teacher_atlas(
        pair_manifest_path=pair_manifest,
        record_id="synthetic-native-fine-teacher",
        output_path=output,
        options=options,
        candidate_fine_chunks_path=candidate_chunks,
    )
    assert recomputed == 1


def test_dataset_reads_shifted_medial_atlas_backed_patch(tmp_path: Path) -> None:
    image_root = tmp_path / "image.zarr"
    baseline_root = tmp_path / "baseline.zarr"
    q_root = tmp_path / "q.zarr"
    valid_root = tmp_path / "valid.zarr"
    crest_root = tmp_path / "crest.zarr"
    crest_valid_root = tmp_path / "crest_valid.zarr"
    image = np.full((8, 8, 8), 100, dtype=np.uint8)
    baseline = np.zeros_like(image)
    q = np.zeros_like(image)
    valid = np.zeros_like(image)
    q[2, 2, 2] = 255
    valid[2, 2, 2] = 1
    crest = np.zeros_like(image)
    crest_valid = np.zeros_like(image)
    crest[2, 2, 2] = 1
    crest_valid[2, 2, 2] = 1
    for path, value in (
        (image_root, image),
        (baseline_root, baseline),
        (q_root, q),
        (valid_root, valid),
        (crest_root, crest),
        (crest_valid_root, crest_valid),
    ):
        _zarr_array(path, value, (4, 4, 4))
    atlas_state = tmp_path / "atlas_state.json"
    atlas_state.write_text(json.dumps({"state": "complete"}), encoding="utf-8")
    medial_state = tmp_path / "medial_state.json"
    medial_state.write_text(
        json.dumps(
            {
                "state": "complete",
                "identity": {
                    "parent_atlas_state_sha256": _sha256(atlas_state),
                },
            }
        ),
        encoding="utf-8",
    )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "schema": "crossres-coarse-teacher-atlas-catalog-v2",
                "sources": {
                    "source": {
                        "coarse_image": f"{image_root}::0",
                        "coarse_baseline": {
                            "volume": f"{baseline_root}::0",
                            "encoding": "labels",
                            "positive_labels": [255],
                            "threshold": 0.5,
                        },
                        "teacher_q": f"{q_root}::0",
                        "target_valid": f"{valid_root}::0",
                        "teacher_crest": f"{crest_root}::0",
                        "teacher_crest_valid": f"{crest_valid_root}::0",
                        "atlas_state": str(atlas_state),
                        "atlas_state_sha256": _sha256(atlas_state),
                        "medial_state": str(medial_state),
                        "medial_state_sha256": _sha256(medial_state),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    metrics = scrollfiesta_patch_pred_metrics(baseline).to_dict()
    manifest = tmp_path / "patches.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema": "crossres-voxel-patch-v1",
                "schema_version": 1,
                "patch_id": "source-00000",
                "path": "patches/source-00000.atlas",
                "record_id": "source",
                "scroll_id": "Synthetic",
                "split": "train",
                "origin_zyx": [0, 0, 0],
                "shape_zyx": [8, 8, 8],
                "known_fraction": 1.0 / 512.0,
                "acceptance_min_known_fraction": 0.001,
                "positive_fraction_known": 1.0,
                "pathology_score": 1.0,
                "sampling_pathology_score": 1.0,
                "scrollfiesta_pred_metrics": metrics,
                "has_baseline": True,
                "supervision_source": "official-native-fine-teacher/atlas",
                "sampling_strategy": "high-pathology",
                "preparation_version": MEDIAL_ATLAS_PATCH_PREPARATION_VERSION,
                "native_teacher_min_fine_ct_nonzero_fraction": 0.95,
                "native_teacher_fine_ct_quality_gate_applied": True,
                "native_teacher_support_chunks_before_quality_gate": 1,
                "native_teacher_support_chunks_after_quality_gate": 1,
                "native_teacher_support_chunks_excluded_by_quality_gate": 0,
                "support_anchor_chunk_zyx": [0, 0, 0],
                "support_anchor_pool_size": 1,
                "support_anchor_candidate_chunks_zyx": [[0, 0, 0]],
                "ct_nonzero_fraction": 1.0,
                "target_projection": {
                    "contract": atlas.ATLAS_PROJECTION_CONTRACT,
                    "prefilter_sigma_scale": 0.5,
                    "coverage_erosion_fine_vox": 0,
                    "maxpool_prefilter": False,
                    "erode_filter_margin": True,
                    "hard_threshold": 0.5,
                    "projection_backend": "cuda-gauss-hermite3-pullback-linf-validity-v1",
                    "gaussian_quadrature_order_per_axis": 3,
                    "validity_erosion_metric": "linf",
                    "atlas_state_sha256": _sha256(atlas_state),
                    "teacher_shift_coarse_zyx": [1, 0, 0],
                    "medial_surface_contract": (
                        atlas.VILLA_MEDIAL_SURFACE_CONTRACT
                    ),
                    "medial_projection_contract": (
                        atlas.MEDIAL_MAX_PROJECTION_CONTRACT
                    ),
                    "medial_state_sha256": _sha256(medial_state),
                },
                "array_source": {
                    "schema": "crossres-coarse-teacher-atlas-patch-source-v2",
                    "catalog": str(catalog),
                    "catalog_sha256": _sha256(catalog),
                    "source_id": "source",
                    "teacher_shift_coarse_zyx": [1, 0, 0],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sample = VoxelPatchDataset(manifest, split="train")[0]
    assert float(sample["target_valid"][0, 3, 2, 2]) == 1.0
    assert float(sample["teacher_q"][0, 3, 2, 2]) == 1.0
    assert int(sample["target"][0, 3, 2, 2]) == 1
    assert bool(sample["has_teacher_crest"])
    assert float(sample["teacher_crest"][0, 3, 2, 2]) == 1.0
    assert float(sample["teacher_crest_valid"][0, 3, 2, 2]) == 1.0
