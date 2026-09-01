from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from crossres_pred.resample import BridgeOptions
from crossres_pred.voxel import antialias_corpus as antialias_module
from crossres_pred.voxel.antialias_corpus import (
    AntialiasCorpusOptions,
    reproject_patch_corpus,
)
from crossres_pred.voxel.antiblob_qualification import (
    AntiblobQualificationOptions,
    morphology_counts,
    select_antiblob_operating_point,
)
from crossres_pred.voxel.checkpoint_audit import TolerantThresholdCounts
from crossres_pred.voxel.loss import (
    CONFIDENT_CONSERVATIVE_LOSS_CONTRACT,
    CONSERVATIVE_LOSS_CONTRACT,
    LOSS_CONTRACT,
    VoxelLossOptions,
    deep_supervision_loss,
    dice_ce_loss,
    loss_contract,
)
from crossres_pred.voxel.patches import (
    ANTIALIAS_PATCH_PREPARATION_VERSION,
    VoxelPatchDataset,
    load_patch_manifest,
    validate_patch_corpus,
)
from crossres_pred.voxel.registration import (
    ChunkSupport,
    FineFieldWindowReader,
    antialias_fine_target_patch,
)
from crossres_pred.voxel.schema import DenseFieldSpec

IDENTITY_AFFINE = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_sparse_fine_reader_marks_absent_chunks_unknown() -> None:
    fine = np.zeros((32, 32, 32), dtype=np.uint8)
    fine[6:10, :16, :16] = 1
    support = ChunkSupport(
        shape_zyx=fine.shape,
        chunks_zyx=(16, 16, 16),
        grid_zyx=(2, 2, 2),
        present_ids=np.asarray([0], dtype=np.int64),
    )
    field = DenseFieldSpec(volume="unused.npy", encoding="labels", positive_labels=(1,))
    reader = FineFieldWindowReader(fine, field, support, max_cache_chunks=2)
    probability = reader.read_probability((0, 0, 0), fine.shape)
    coverage = reader.read_coverage((0, 0, 0), fine.shape)
    raw = reader.read_raw((0, 0, 0), fine.shape, fill_value=7)
    assert probability[6:10, :16, :16].all()
    assert coverage[:16, :16, :16].all()
    assert not coverage[16:].any()
    assert not coverage[:, 16:].any()
    assert not coverage[:, :, 16:].any()
    assert np.array_equal(raw[:16, :16, :16], fine[:16, :16, :16])
    assert np.all(raw[16:] == 7)
    assert np.all(raw[:, 16:] == 7)
    assert np.all(raw[:, :, 16:] == 7)
    assert reader.cache_hits > 0


def test_antialias_pullback_is_deterministic_and_does_not_maxpool() -> None:
    fine = np.zeros((96, 96, 96), dtype=np.uint8)
    fine[44:48] = 1
    field = DenseFieldSpec(volume="unused.npy", encoding="labels", positive_labels=(1,))
    support = ChunkSupport.from_field(field, fine)
    options = BridgeOptions(
        prefilter_sigma_scale=0.5,
        coverage_erosion_fine_vox=0,
        maxpool_prefilter=False,
        erode_filter_margin=True,
    )
    first = antialias_fine_target_patch(
        fine,
        field,
        support,
        IDENTITY_AFFINE,
        (32, 32, 32),
        (32, 32, 32),
        options=options,
    )
    second = antialias_fine_target_patch(
        fine,
        field,
        support,
        IDENTITY_AFFINE,
        (32, 32, 32),
        (32, 32, 32),
        options=options,
    )
    for left, right in zip(first[:3], second[:3], strict=True):
        assert np.array_equal(left, right)
    hard, q, valid, _ = first
    assert valid.all()
    assert np.flatnonzero((hard == 1).any(axis=(1, 2))).tolist() == [12, 13, 14, 15]
    assert float(q[8].max()) < 0.01


def test_two_voxel_surface_metric_rewards_a_nearby_thin_prediction() -> None:
    target = np.zeros((16, 16, 16), dtype=np.uint8)
    target[8, 3:13, 3:13] = 1
    probability = np.zeros_like(target, dtype=np.float32)
    probability[9, 3:13, 3:13] = 0.9
    counts = TolerantThresholdCounts.create((0.5,))
    counts.update(probability, target)
    metrics = counts.metrics_at(0)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f0_5"] == 1.0


def _audit_point(
    threshold: float,
    *,
    f0_5: float,
    precision: float,
    foreground_ratio: float,
    dice: float,
) -> dict[str, object]:
    return {
        "threshold": threshold,
        "foreground_ratio": foreground_ratio,
        "macro_scroll_dice": dice,
        "macro_surface_f0_5_at_2vox": f0_5,
        "surface_at_2vox": {"f0_5": f0_5, "precision": precision},
        "scrolls": {
            "PHerc0814": {"dice": dice},
            "PHerc1451": {"dice": dice - 0.01},
        },
    }


def test_antiblob_selection_uses_f0_5_subject_to_regression_gates() -> None:
    initial = {
        "points": [
            _audit_point(0.4, f0_5=0.6, precision=0.7, foreground_ratio=1.0, dice=0.7),
            _audit_point(0.5, f0_5=0.6, precision=0.7, foreground_ratio=1.0, dice=0.7),
        ]
    }
    trained = {
        "points": [
            _audit_point(0.4, f0_5=0.9, precision=0.5, foreground_ratio=1.4, dice=0.8),
            _audit_point(
                0.5, f0_5=0.8, precision=0.72, foreground_ratio=1.0, dice=0.72
            ),
        ]
    }
    selected, candidates = select_antiblob_operating_point(
        initial,
        trained,
        options=AntiblobQualificationOptions(device="cpu", num_workers=0),
    )
    assert selected["threshold"] == 0.5
    assert selected["preliminary_qualified"] is True
    assert candidates[0]["preliminary_qualified"] is False


def test_morphology_counts_detects_blob_interior() -> None:
    target = np.zeros((24, 24, 24), dtype=np.uint8)
    target[12, 4:20, 4:20] = 1
    thin = morphology_counts(target == 1, target)
    blob = np.zeros_like(target, dtype=bool)
    blob[9:16, 4:20, 4:20] = True
    thick = morphology_counts(blob, target)
    assert thin["predicted_two_erode_interior_voxels"] == 0
    assert thick["predicted_two_erode_interior_voxels"] > 0


@pytest.mark.torch
def test_soft_dice_ce_uses_fractional_teacher_occupancy() -> None:
    logits = torch.zeros((1, 2, 1, 2, 2), requires_grad=True)
    target = torch.zeros((1, 1, 2, 2), dtype=torch.long)
    teacher_q = torch.full((1, 1, 2, 2), 0.25)
    valid = torch.ones_like(teacher_q)
    result = dice_ce_loss(
        logits,
        target,
        teacher_q=teacher_q,
        target_valid=valid,
    )
    assert LOSS_CONTRACT == "soft-dice-ce-per-sample-known-v3"
    assert float(result.cross_entropy.detach()) == pytest.approx(np.log(2.0), rel=1e-6)
    assert float(result.dice.detach()) == pytest.approx(2.0 / 3.0, rel=1e-4)
    result.total.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.torch
def test_conservative_loss_penalizes_confident_background_bridge_voxels() -> None:
    target = torch.zeros((1, 1, 1, 7), dtype=torch.long)
    target[:, :, :, 1] = 1
    target[:, :, :, 5] = 1
    teacher_q = target.float()
    valid = torch.ones_like(teacher_q)
    logits = torch.zeros((1, 2, 1, 1, 7))
    logits[:, 1, 0, 0, 3] = 5.0
    options = VoxelLossOptions(
        dice_weight=0.25,
        separation_weight=2.0,
        separation_radius=2,
    )

    result = dice_ce_loss(
        logits,
        target,
        teacher_q=teacher_q,
        target_valid=valid,
        options=options,
    )

    assert loss_contract(options) == CONSERVATIVE_LOSS_CONTRACT
    assert float(result.separation) > 1.0


@pytest.mark.torch
def test_m7_anchor_kl_applies_where_fine_teacher_is_unknown() -> None:
    target = torch.full((1, 1, 1, 2), 2, dtype=torch.long)
    valid = torch.zeros_like(target, dtype=torch.float32)
    student = torch.zeros((1, 2, 1, 1, 2))
    student[:, 1, 0, 0, 0] = 5.0
    anchor = torch.zeros_like(student)
    anchor[:, 0] = 5.0
    options = VoxelLossOptions(dice_weight=0.0, m7_anchor_weight=1.0)

    result = dice_ce_loss(
        student,
        target,
        target_valid=valid,
        m7_anchor_logits=anchor,
        options=options,
    )

    assert float(result.cross_entropy) == 0.0
    assert float(result.m7_anchor_kl) > 1.0


@pytest.mark.torch
def test_m7_anchor_kl_preserves_known_agreement_but_not_disagreement() -> None:
    target = torch.tensor([[[[0, 1]]]], dtype=torch.long)
    anchor = torch.zeros((1, 2, 1, 1, 2))
    anchor[:, 0] = 5.0
    options = VoxelLossOptions(dice_weight=0.0, m7_anchor_weight=1.0)

    disagreement_only = anchor.clone()
    disagreement_only[:, 0, 0, 0, 1] = 0.0
    disagreement_only[:, 1, 0, 0, 1] = 5.0
    disagreement_result = dice_ce_loss(
        disagreement_only,
        target,
        m7_anchor_logits=anchor,
        options=options,
    )

    agreement_changed = anchor.clone()
    agreement_changed[:, 0, 0, 0, 0] = 0.0
    agreement_changed[:, 1, 0, 0, 0] = 5.0
    agreement_result = dice_ce_loss(
        agreement_changed,
        target,
        m7_anchor_logits=anchor,
        options=options,
    )

    assert float(disagreement_result.m7_anchor_kl) == pytest.approx(0.0, abs=1.0e-6)
    assert float(agreement_result.m7_anchor_kl) > 1.0


@pytest.mark.torch
def test_m7_anchor_confident_agreement_frees_partial_volume_band() -> None:
    # q = [confident background, partial-volume sheet x2, confident sheet];
    # the anchor is confident background everywhere, and the student raises
    # foreground exactly on the partial-volume band the teacher wants grown.
    teacher_q = torch.tensor([[[[0.02, 0.30, 0.30, 0.80]]]], dtype=torch.float32)
    target = (teacher_q >= 0.5).long()
    valid = torch.ones_like(teacher_q)
    anchor = torch.zeros((1, 2, 1, 1, 4))
    anchor[:, 0] = 5.0
    student = torch.zeros((1, 2, 1, 1, 4))
    student[:, 0] = 5.0
    student[:, 0, 0, 0, 1:3] = 0.0
    student[:, 1, 0, 0, 1:3] = 5.0
    known = VoxelLossOptions(dice_weight=0.25, m7_anchor_weight=1.0)
    confident = VoxelLossOptions(
        dice_weight=0.25,
        m7_anchor_weight=1.0,
        m7_anchor_confident_agreement=True,
    )

    known_result = dice_ce_loss(
        student,
        target,
        teacher_q=teacher_q,
        target_valid=valid,
        m7_anchor_logits=anchor,
        options=known,
    )
    confident_result = dice_ce_loss(
        student,
        target,
        teacher_q=teacher_q,
        target_valid=valid,
        m7_anchor_logits=anchor,
        options=confident,
    )

    assert float(known_result.m7_anchor_kl) > 1.0
    assert float(confident_result.m7_anchor_kl) == pytest.approx(0.0, abs=1.0e-6)
    assert float(confident_result.cross_entropy) == pytest.approx(
        float(known_result.cross_entropy), rel=1.0e-6
    )


@pytest.mark.torch
def test_m7_anchor_confident_agreement_still_anchors_confident_background() -> None:
    teacher_q = torch.tensor([[[[0.02, 0.30]]]], dtype=torch.float32)
    target = (teacher_q >= 0.5).long()
    valid = torch.ones_like(teacher_q)
    anchor = torch.zeros((1, 2, 1, 1, 2))
    anchor[:, 0] = 5.0
    student = torch.zeros((1, 2, 1, 1, 2))
    student[:, 1, 0, 0, 0] = 5.0
    options = VoxelLossOptions(
        dice_weight=0.25,
        m7_anchor_weight=1.0,
        m7_anchor_confident_agreement=True,
    )

    result = dice_ce_loss(
        student,
        target,
        teacher_q=teacher_q,
        target_valid=valid,
        m7_anchor_logits=anchor,
        options=options,
    )

    assert float(result.m7_anchor_kl) > 1.0


def test_confident_agreement_contract_and_validation() -> None:
    confident = VoxelLossOptions(
        dice_weight=0.25,
        separation_weight=2.0,
        m7_anchor_weight=0.5,
        m7_anchor_confident_agreement=True,
    )
    assert loss_contract(confident) == CONFIDENT_CONSERVATIVE_LOSS_CONTRACT
    assert (
        loss_contract(VoxelLossOptions(dice_weight=0.25, separation_weight=2.0))
        == CONSERVATIVE_LOSS_CONTRACT
    )
    with pytest.raises(ValueError, match="confident-agreement"):
        VoxelLossOptions(
            m7_anchor_weight=0.5,
            m7_anchor_known_agreement=False,
            m7_anchor_confident_agreement=True,
        ).validate()


@pytest.mark.torch
def test_soft_deep_supervision_resizes_q_and_ignores_invalid_voxels() -> None:
    full = torch.zeros((1, 2, 32, 32, 32), requires_grad=True)
    half = torch.zeros((1, 2, 16, 16, 16), requires_grad=True)
    target = torch.full((1, 1, 32, 32, 32), 2, dtype=torch.long)
    teacher_q = torch.zeros((1, 1, 32, 32, 32))
    valid = torch.zeros_like(teacher_q)
    target[:, :, 8:24, 8:24, 8:24] = 1
    teacher_q[:, :, 8:24, 8:24, 8:24] = 0.75
    valid[:, :, 8:24, 8:24, 8:24] = 1
    loss, _ = deep_supervision_loss(
        [full, half], target, teacher_q=teacher_q, target_valid=valid
    )
    loss.backward()
    assert torch.isfinite(full.grad).all()
    assert torch.isfinite(half.grad).all()


def test_reprojected_corpus_round_trips_soft_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fine = np.zeros((96, 96, 96), dtype=np.uint8)
    fine[44:48, 24:72, 24:72] = 1
    fine_path = tmp_path / "fine.npy"
    coarse_path = tmp_path / "coarse.npy"
    np.save(fine_path, fine)
    np.save(coarse_path, np.full((96, 96, 96), 100, dtype=np.uint16))
    pair_manifest = tmp_path / "pairs.jsonl"
    pair_manifest.write_text(
        json.dumps(
            {
                "schema": "crossres-voxel-pair-v1",
                "schema_version": 1,
                "record_id": "synthetic-native-fine-teacher",
                "scroll_id": "SyntheticTrain",
                "split": "train",
                "supervision_source": "official-native-fine-teacher/test",
                "coarse": {
                    "scan_id": "coarse",
                    "voxel_um": 9.0,
                    "image": str(coarse_path),
                },
                "fine": {
                    "scan_id": "fine",
                    "voxel_um": 2.25,
                    "target": {
                        "volume": str(fine_path),
                        "encoding": "labels",
                        "positive_labels": [1],
                    },
                    "to_coarse_affine_xyz": [list(row) for row in IDENTITY_AFFINE],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_root = tmp_path / "source"
    source_patches = source_root / "patches"
    source_patches.mkdir(parents=True)
    patch_id = "synthetic-native-fine-teacher-00000"
    source_archive = source_patches / f"{patch_id}.npz"
    source_target = np.zeros((32, 32, 32), dtype=np.uint8)
    source_target[12:16] = 1
    np.savez_compressed(
        source_archive,
        image=np.full((32, 32, 32), 100, dtype=np.uint16),
        target_u8=source_target,
    )
    source_manifest = source_root / "patches.jsonl"
    source_manifest.write_text(
        json.dumps(
            {
                "schema": "crossres-voxel-patch-v1",
                "schema_version": 1,
                "patch_id": patch_id,
                "path": f"patches/{patch_id}.npz",
                "record_id": "synthetic-native-fine-teacher",
                "scroll_id": "SyntheticTrain",
                "split": "train",
                "origin_zyx": [32, 32, 32],
                "shape_zyx": [32, 32, 32],
                "known_fraction": 1.0,
                "acceptance_min_known_fraction": 0.002,
                "positive_fraction_known": float((source_target == 1).mean()),
                "pathology_score": 0.0,
                "scrollfiesta_pred_metrics": None,
                "has_baseline": False,
                "supervision_source": "official-native-fine-teacher/test",
                "sampling_strategy": "random",
                "preparation_version": "projection-cache-locality-v9",
                "native_teacher_min_fine_ct_nonzero_fraction": 0.95,
                "native_teacher_fine_ct_quality_gate_applied": True,
                "native_teacher_support_chunks_before_quality_gate": 1,
                "native_teacher_support_chunks_after_quality_gate": 1,
                "native_teacher_support_chunks_excluded_by_quality_gate": 0,
                "support_anchor_chunk_zyx": [0, 0, 0],
                "support_anchor_pool_size": 1,
                "support_anchor_candidate_chunks_zyx": [[0, 0, 0]],
                "ct_nonzero_fraction": 1.0,
                "archive_bytes": source_archive.stat().st_size,
                "archive_sha256": _sha256(source_archive),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "antialias"
    reference_pullback = antialias_module.antialias_fine_target_patch

    def production_contract_stub(*args, **kwargs):
        hard, q, valid, stats = reference_pullback(*args, **kwargs)
        stats.update(
            projection_backend="cuda-gauss-hermite3-pullback-linf-validity-v1",
            gaussian_quadrature_order_per_axis=3,
        )
        return hard, q, valid, stats

    monkeypatch.setattr(
        antialias_module,
        "antialias_fine_target_patch",
        production_contract_stub,
    )
    manifest = reproject_patch_corpus(
        source_manifest_path=source_manifest,
        pair_manifest_paths=[pair_manifest],
        output_path=output,
        options=AntialiasCorpusOptions(max_cpu_threads=4),
    )
    row = load_patch_manifest(manifest)[0]
    assert row.preparation_version == ANTIALIAS_PATCH_PREPARATION_VERSION
    dataset_row = VoxelPatchDataset(manifest, split="train")[0]
    assert dataset_row["teacher_q"].shape == dataset_row["target"].shape
    assert bool(((dataset_row["teacher_q"] > 0) & (dataset_row["teacher_q"] < 1)).any())
    summary = validate_patch_corpus(
        manifest,
        expected_count=1,
        expected_split_counts={"train": 1, "val": 0, "test": 0},
        expected_train_scrolls={"SyntheticTrain"},
        expected_val_scrolls=set(),
        expected_test_scrolls=set(),
        expected_record_counts={"synthetic-native-fine-teacher": 1},
        expected_source_corpora=1,
        workers=1,
        max_cpu_threads=4,
    )
    assert summary["patches"] == 1

    registration_root = tmp_path / "registration"
    registration_root.mkdir()
    registration_manifest = registration_root / "registrations.jsonl"
    translated_affine = [list(row) for row in IDENTITY_AFFINE]
    translated_affine[0][3] = 4.0
    registration_manifest.write_text(
        json.dumps(
            {
                "schema": "crossres-patch-registration-row-v1",
                "patch_id": patch_id,
                "record_id": "synthetic-native-fine-teacher",
                "scroll_id": "SyntheticTrain",
                "origin_zyx": [32, 32, 32],
                "accepted": True,
                "contract": "crossres-local-ct-translation-l0-v1",
                "method": "local-ct-translation",
                "shift_coarse_zyx": [0, 0, 4],
                "effective_to_coarse_affine_xyz": translated_affine,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    registration_state = {
        "schema": "crossres-patch-registration-state-v1",
        "state": "complete",
        "completed": 1,
        "manifest_sha256": _sha256(registration_manifest),
        "identity": {"source_manifest_sha256": _sha256(source_manifest)},
    }
    (registration_root / "state.json").write_text(
        json.dumps(registration_state), encoding="utf-8"
    )
    observed_affines: list[tuple[tuple[float, ...], ...]] = []

    def registered_pullback(*args, **kwargs):
        observed_affines.append(
            tuple(tuple(float(value) for value in row) for row in args[3])
        )
        return production_contract_stub(*args, **kwargs)

    monkeypatch.setattr(
        antialias_module,
        "antialias_fine_target_patch",
        registered_pullback,
    )
    registered_output = tmp_path / "antialias-registered"
    registered_manifest = reproject_patch_corpus(
        source_manifest_path=source_manifest,
        pair_manifest_paths=[pair_manifest],
        output_path=registered_output,
        options=AntialiasCorpusOptions(max_cpu_threads=4),
        patch_registration_manifest_path=registration_manifest,
    )
    assert observed_affines == [tuple(tuple(row) for row in translated_affine)]
    registered_row = json.loads(registered_manifest.read_text(encoding="utf-8"))
    assert registered_row["target_projection"]["patch_registration"][
        "shift_coarse_zyx"
    ] == [0, 0, 4]

    source_value = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_value["balanced_source_quality_gate"] = {
        "schema": "crossres-balanced-source-quality-gate-v1",
        "accepted": True,
    }
    source_manifest.write_text(
        json.dumps(source_value, sort_keys=True) + "\n", encoding="utf-8"
    )

    def forbidden_pullback(*args, **kwargs):
        raise AssertionError("compatible target should have been reused")

    monkeypatch.setattr(
        antialias_module,
        "antialias_fine_target_patch",
        forbidden_pullback,
    )
    reused_output = tmp_path / "antialias-reused"
    reused_manifest = reproject_patch_corpus(
        source_manifest_path=source_manifest,
        pair_manifest_paths=[pair_manifest],
        output_path=reused_output,
        options=AntialiasCorpusOptions(max_cpu_threads=4),
        reuse_corpus_path=output,
    )
    reused_row = json.loads(reused_manifest.read_text(encoding="utf-8"))
    assert reused_row["balanced_source_quality_gate"]["accepted"] is True
    assert reused_row["archive_sha256"] == row.archive_sha256
    assert os.path.samefile(
        output / row.path.relative_to(output),
        reused_output / reused_row["path"],
    )
    reused_summary = json.loads(
        (reused_output / "summary.json").read_text(encoding="utf-8")
    )
    assert reused_summary["reuse"]["archives_reused"] == 1
