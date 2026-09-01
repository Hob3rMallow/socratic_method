from __future__ import annotations

import json
import os
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from crossres_pred.voxel import model as voxel_model
from crossres_pred.voxel import train as voxel_train
from crossres_pred.voxel.loss import LOSS_CONTRACT, VoxelLossOptions
from crossres_pred.voxel.train import (
    ADAM_OPTIMIZER_CONTRACT,
    ADAMW_OPTIMIZER_CONTRACT,
    CHECKPOINT_DURABILITY_CONTRACT,
    CHECKPOINT_SELECTION_CONTRACT,
    EPOCH_PARTITION_CONTRACT,
    FINAL_FIT_CHECKPOINT_CONTRACT,
    LEARNING_RATE_CONTRACT,
    SAMPLING_CONTRACT,
    SNAPSHOT_CHECKPOINT_CONTRACT,
    EpochPartitionSampler,
    StratifiedEpochPartitionSampler,
    ThresholdHistogram,
    TrainOptions,
    _atomic_torch_save,
    _bounded_partition_samples,
    _build_optimizer,
    _capture_rng_state,
    _history_row_score,
    _learning_rate_for_samples,
    _normalize_snapshot_samples,
    _options_identity,
    _reconcile_best_checkpoint_artifacts,
    _reconcile_history,
    _reconcile_snapshot_records,
    _resolve_training_schedule,
    _restore_rng_state,
    _select_best_checkpoint,
    _sha256,
    train_model,
)


def test_validation_supplies_frozen_m7_logits_to_preservation_loss() -> None:
    class StaticModel(torch.nn.Module):
        def __init__(self, foreground_logit: float) -> None:
            super().__init__()
            self.foreground_logit = foreground_logit

        def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
            background = torch.zeros_like(image)
            foreground = torch.full_like(image, self.foreground_logit)
            return [torch.cat((background, foreground), dim=1)]

    shape = (1, 1, 4, 4, 4)
    batch = {
        "image": torch.zeros(shape),
        "target": torch.ones(shape, dtype=torch.long),
        "teacher_q": torch.ones(shape),
        "target_valid": torch.ones(shape),
        "baseline": torch.zeros(shape, dtype=torch.long),
        "has_baseline": torch.tensor([False]),
        "pathology_score": torch.tensor([0.0]),
        "scrollfiesta_pred_reject_kind": torch.tensor([0]),
        "scroll_id": ["PHerc0139"],
        "supervision_source": ["teacher"],
        "sampling_strategy": ["uniform"],
    }
    options = VoxelLossOptions(
        m7_preservation_weight=1.0,
        m7_preservation_radius=1,
    )

    with pytest.raises(ValueError, match="requires the frozen M7 model"):
        voxel_train.validate_model(
            StaticModel(-1.0),
            [batch],
            torch.device("cpu"),
            torch.float32,
            False,
            loss_options=options,
        )

    metrics = voxel_train.validate_model(
        StaticModel(-1.0),
        [batch],
        torch.device("cpu"),
        torch.float32,
        False,
        loss_options=options,
        m7_anchor_model=StaticModel(2.0),
    )
    assert metrics["loss_m7_preservation_loss"] > 1.0


def test_training_artifact_replace_retries_sharing_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_text("ready", encoding="utf-8")
    real_replace = voxel_train.os.replace
    attempts = 0

    def flaky_replace(left: Path, right: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("simulated reader lock")
        real_replace(left, right)

    monkeypatch.setattr(voxel_train.os, "replace", flaky_replace)
    monkeypatch.setattr(voxel_train.time, "sleep", lambda _seconds: None)
    voxel_train._replace_with_retry(source, destination)

    assert attempts == 3
    assert destination.read_text(encoding="utf-8") == "ready"


def test_epoch_partition_sampler_covers_a_shuffle_before_reuse() -> None:
    sampler = EpochPartitionSampler(dataset_size=7, samples_per_epoch=3, seed=41)
    epochs: list[list[int]] = []
    for epoch in range(4):
        sampler.set_epoch(epoch)
        epochs.append(list(sampler))
    first_pass = epochs[0] + epochs[1] + epochs[2][:1]
    assert len(first_pass) == 7
    assert set(first_pass) == set(range(7))
    assert epochs[2][1:] + epochs[3] == list(
        EpochPartitionSampler(dataset_size=7, samples_per_epoch=5, seed=42)
    )


def test_epoch_partition_sampler_caps_the_final_microepoch() -> None:
    sampler = EpochPartitionSampler(
        dataset_size=7,
        samples_per_epoch=3,
        seed=41,
        total_samples=7,
    )
    epochs: list[list[int]] = []
    for epoch in range(3):
        sampler.set_epoch(epoch)
        epochs.append(list(sampler))
    assert [len(indices) for indices in epochs] == [3, 3, 1]
    assert len([index for indices in epochs for index in indices]) == 7
    assert {index for indices in epochs for index in indices} == set(range(7))


def test_one_pass_schedule_is_bounded_only_in_its_final_epoch() -> None:
    assert (
        _bounded_partition_samples(
            250_000,
            epochs=10,
            samples_per_epoch=25_000,
        )
        == 250_000
    )
    assert (
        _bounded_partition_samples(
            251_088,
            epochs=10,
            samples_per_epoch=25_109,
        )
        == 251_088
    )
    assert _bounded_partition_samples(7, epochs=4, samples_per_epoch=3) is None
    assert _bounded_partition_samples(13, epochs=4, samples_per_epoch=3) is None


def test_train_options_reject_non_positive_samples_per_epoch() -> None:
    with pytest.raises(ValueError, match="samples_per_epoch"):
        TrainOptions(
            preset="tiny-test",
            device="cpu",
            amp=False,
            max_cpu_threads=1,
            num_workers=0,
            samples_per_epoch=0,
        ).validate()


def test_no_augmentation_is_explicit_in_run_identity() -> None:
    common = {
        "preset": "tiny-test",
        "device": "cpu",
        "amp": False,
        "max_cpu_threads": 1,
        "num_workers": 0,
    }
    assert "train_augmentation" not in _options_identity(TrainOptions(**common))
    assert (
        _options_identity(TrainOptions(**common, train_augmentation=False))[
            "train_augmentation"
        ]
        is False
    )


def test_confident_agreement_is_explicit_in_run_identity() -> None:
    common = {
        "preset": "tiny-test",
        "device": "cpu",
        "amp": False,
        "max_cpu_threads": 1,
        "num_workers": 0,
    }
    conservative = VoxelLossOptions(
        dice_weight=0.25,
        separation_weight=2.0,
        m7_anchor_weight=0.5,
    )
    identity = _options_identity(TrainOptions(**common, loss_options=conservative))
    assert "m7_anchor_confident_agreement" not in identity["loss_options"]
    confident_identity = _options_identity(
        TrainOptions(
            **common,
            loss_options=replace(conservative, m7_anchor_confident_agreement=True),
        )
    )
    assert confident_identity["loss_options"]["m7_anchor_confident_agreement"] is True


def test_pinned_axial_configuration_is_explicit_and_requires_its_atlas(
    tmp_path: Path,
) -> None:
    common = {
        "preset": "tiny-test",
        "device": "cpu",
        "amp": False,
        "max_cpu_threads": 1,
        "num_workers": 0,
    }
    assert "pinned_medial_bridge_state" not in _options_identity(TrainOptions(**common))
    with pytest.raises(ValueError, match="requires a bridge atlas state"):
        TrainOptions(
            **common,
            loss_options=VoxelLossOptions(pinned_axial_weight=0.25),
        ).validate()

    state = tmp_path / "bridge_state.json"
    state.write_text("{}", encoding="utf-8")
    options = TrainOptions(
        **common,
        loss_options=VoxelLossOptions(
            pinned_axial_weight=0.25,
            pinned_axial_probability_floor=0.20,
            pinned_axial_bottom_fraction=0.10,
        ),
        pinned_medial_bridge_state=str(state),
    )
    options.validate()
    identity = _options_identity(options)
    assert identity["pinned_medial_bridge_state"] == str(state)
    assert identity["loss_options"]["pinned_axial_weight"] == 0.25


def test_dynamic_connectivity_configuration_requires_its_atlas(tmp_path: Path) -> None:
    common = {
        "preset": "tiny-test",
        "device": "cpu",
        "amp": False,
        "max_cpu_threads": 1,
        "num_workers": 0,
    }
    assert "dynamic_medial_connectivity_state" not in _options_identity(
        TrainOptions(**common)
    )
    with pytest.raises(ValueError, match="requires an atlas state"):
        TrainOptions(
            **common,
            loss_options=VoxelLossOptions(dynamic_medial_connectivity_weight=0.125),
        ).validate()

    state = tmp_path / "connectivity_state.json"
    state.write_text("{}", encoding="utf-8")
    options = TrainOptions(
        **common,
        loss_options=VoxelLossOptions(
            dynamic_medial_connectivity_weight=0.125,
            dynamic_medial_connectivity_probability_floor=0.20,
            dynamic_medial_connectivity_steps=96,
        ),
        dynamic_medial_connectivity_state=str(state),
    )
    options.validate()
    identity = _options_identity(options)
    assert identity["dynamic_medial_connectivity_state"] == str(state)
    assert identity["loss_options"]["dynamic_medial_connectivity_weight"] == 0.125


def test_training_accepts_a_separate_pinned_validation_manifest(tmp_path: Path) -> None:
    image = np.arange(8**3, dtype=np.uint16).reshape(8, 8, 8) % 213
    target = np.zeros((8, 8, 8), dtype=np.uint8)
    target[3:5] = 1

    def write_manifest(name: str, split: str, scroll: str) -> Path:
        archive = tmp_path / f"{name}.npz"
        np.savez(archive, image=image, target_u8=target)
        manifest = tmp_path / f"{name}.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "crossres-voxel-patch-v1",
                    "patch_id": name,
                    "path": str(archive),
                    "record_id": name,
                    "scroll_id": scroll,
                    "split": split,
                    "origin_zyx": [0, 0, 0],
                    "shape_zyx": [8, 8, 8],
                    "known_fraction": 1.0,
                    "positive_fraction_known": 0.25,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest

    train_manifest = write_manifest("train", "train", "scroll-train")
    validation_manifest = write_manifest("validation", "val", "scroll-val")
    output = tmp_path / "run"
    checkpoint = train_model(
        patch_manifest=train_manifest,
        validation_patch_manifest=validation_manifest,
        output_path=output,
        options=TrainOptions(
            epochs=1,
            batch_size=1,
            accumulate=1,
            num_workers=0,
            seed=1203,
            device="cpu",
            amp=False,
            preset="tiny-test",
            max_cpu_threads=1,
            train_augmentation=False,
        ),
    )
    assert checkpoint.is_file()
    identity = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert identity["validation_patch_manifest"] == str(validation_manifest.resolve())
    assert identity["validation_patch_manifest_sha256"] == _sha256(validation_manifest)


def test_released_m7_identity_is_exactly_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint_best.pth"
    checkpoint.write_bytes(b"released-m7-fixture")
    digest = _sha256(checkpoint)
    monkeypatch.setattr(voxel_model, "RELEASED_M7_CHECKPOINT_SHA256", digest)

    identity = voxel_model.verify_released_m7_checkpoint(checkpoint)

    assert identity == {
        "contract": voxel_model.FRESH_M7_INITIALIZATION_CONTRACT,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest,
        "checkpoint_bytes": checkpoint.stat().st_size,
    }
    checkpoint.write_bytes(b"student-checkpoint")
    with pytest.raises(ValueError, match="exact released M7"):
        voxel_model.verify_released_m7_checkpoint(checkpoint)


def test_adamw_is_explicit_and_uses_recorded_hyperparameters() -> None:
    options = TrainOptions(
        preset="tiny-test",
        device="cpu",
        amp=False,
        max_cpu_threads=1,
        num_workers=0,
        optimizer="adamw",
        learning_rate=3.0e-5,
        weight_decay=1.0e-4,
        adamw_beta1=0.85,
        adamw_beta2=0.97,
        adamw_eps=1.0e-7,
    )
    options.validate()
    parameter = torch.nn.Parameter(torch.ones(()))

    optimizer = _build_optimizer(torch.nn.ParameterList([parameter]), options)

    assert ADAMW_OPTIMIZER_CONTRACT == "torch-adamw-explicit-betas-eps-v1"
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3.0e-5)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1.0e-4)
    assert optimizer.param_groups[0]["betas"] == pytest.approx((0.85, 0.97))
    assert optimizer.param_groups[0]["eps"] == pytest.approx(1.0e-7)


def test_adam_is_explicit_and_uses_recorded_hyperparameters() -> None:
    options = TrainOptions(
        preset="tiny-test",
        device="cpu",
        amp=False,
        max_cpu_threads=1,
        num_workers=0,
        optimizer="adam",
        learning_rate=1.0e-6,
        weight_decay=0.0,
        adamw_beta1=0.85,
        adamw_beta2=0.97,
        adamw_eps=1.0e-7,
    )
    options.validate()
    parameter = torch.nn.Parameter(torch.ones(()))

    optimizer = _build_optimizer(torch.nn.ParameterList([parameter]), options)

    assert ADAM_OPTIMIZER_CONTRACT == "torch-adam-explicit-betas-eps-v1"
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-6)
    assert optimizer.param_groups[0]["weight_decay"] == 0.0


def test_train_options_reject_invalid_optimizer_settings() -> None:
    common = {
        "preset": "tiny-test",
        "device": "cpu",
        "amp": False,
        "max_cpu_threads": 1,
        "num_workers": 0,
    }
    with pytest.raises(ValueError, match="optimizer"):
        TrainOptions(**common, optimizer="rmsprop").validate()
    with pytest.raises(ValueError, match="betas"):
        TrainOptions(**common, adamw_beta2=1.0).validate()
    with pytest.raises(ValueError, match="epsilon"):
        TrainOptions(**common, adamw_eps=0.0).validate()


def test_explicit_sample_budget_is_independent_of_evaluation_interval() -> None:
    first = _resolve_training_schedule(
        250_000,
        epochs=999,
        samples_per_epoch=10_000,
        max_train_samples=50_000,
    )
    second = _resolve_training_schedule(
        250_000,
        epochs=1,
        samples_per_epoch=25_000,
        max_train_samples=50_000,
    )
    assert first.total_samples == second.total_samples == 50_000
    assert first.evaluation_intervals == 5
    assert second.evaluation_intervals == 2


def test_snapshot_samples_are_sorted_unique_and_within_budget() -> None:
    assert _normalize_snapshot_samples((50, 100, 250), total_samples=250) == (
        50,
        100,
        250,
    )
    with pytest.raises(ValueError, match="sorted and unique"):
        _normalize_snapshot_samples((100, 50), total_samples=250)
    with pytest.raises(ValueError, match="positive"):
        _normalize_snapshot_samples((0,), total_samples=250)
    with pytest.raises(ValueError, match="exceed"):
        _normalize_snapshot_samples((251,), total_samples=250)


def test_learning_rate_depends_on_sample_progress_not_interval_size() -> None:
    options = TrainOptions(
        preset="tiny-test",
        device="cpu",
        amp=False,
        max_cpu_threads=1,
        num_workers=0,
        learning_rate=1.0e-3,
        lr_schedule="cosine",
        lr_floor_ratio=0.1,
    )
    at_20k = _learning_rate_for_samples(
        options, samples_seen=20_000, total_samples=50_000
    )
    assert at_20k == pytest.approx(
        _learning_rate_for_samples(options, samples_seen=20_000, total_samples=50_000)
    )
    assert at_20k < options.learning_rate
    assert _learning_rate_for_samples(
        options, samples_seen=50_000, total_samples=50_000
    ) == pytest.approx(options.learning_rate * 0.1)


def test_stratified_sampler_is_proportional_without_replacement() -> None:
    rows = [
        SimpleNamespace(
            scroll_id="large" if index < 80 else "small",
            supervision_source="human" if index < 80 else "native",
            pathology_score=0.2 if index < 80 else 0.0,
        )
        for index in range(100)
    ]
    sampler = StratifiedEpochPartitionSampler(
        rows, samples_per_epoch=50, seed=41, total_samples=100
    )
    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert len(set(first + second)) == 100
    assert 39 <= sum(index < 80 for index in first) <= 41


def test_threshold_histogram_computes_exact_counts() -> None:
    histogram = ThresholdHistogram((0.25, 0.5, 0.75))
    histogram.update_probability(
        torch.tensor([0.10, 0.40, 0.60, 0.90]),
        torch.tensor([0, 1, 0, 1]),
    )
    assert histogram.metrics(0)["dice"] == pytest.approx(0.8)
    assert histogram.metrics(1)["dice"] == pytest.approx(0.5)
    assert histogram.metrics(2)["dice"] == pytest.approx(2.0 / 3.0)


def _row(epoch: int, *, dice: float) -> dict[str, object]:
    return {
        "epoch": epoch,
        "learning_rate": 0.001,
        "train": {"loss_total": 1.0 - dice},
        "val": {"dice": dice},
    }


def test_checkpoint_score_prefers_macro_scroll_dice() -> None:
    row = _row(0, dice=0.99)
    row["val"]["macro_scroll_dice"] = 0.25  # type: ignore[index]
    assert _history_row_score(row) == pytest.approx(0.25)


def test_final_fit_checkpoint_always_advances_to_last_completed_epoch() -> None:
    assert FINAL_FIT_CHECKPOINT_CONTRACT == ("selected-sample-budget-last-completed-v2")
    assert _select_best_checkpoint(-2.0, -1.0, final_fit=True) == (True, -2.0)
    assert _select_best_checkpoint(0.2, 0.3, final_fit=False) == (False, 0.3)


def test_checkpoint_is_fsynced_before_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "checkpoint.pt"
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def recording_fsync(descriptor: int) -> None:
        events.append("fsync")
        real_fsync(descriptor)

    def recording_replace(source: str | Path, target: str | Path) -> None:
        events.append("replace")
        real_replace(source, target)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "replace", recording_replace)
    _atomic_torch_save(destination, {"epoch": 7, "tensor": torch.arange(4)})

    assert events == ["fsync", "replace"]
    assert not destination.with_name(destination.name + ".tmp").exists()
    loaded = torch.load(destination, map_location="cpu", weights_only=False)
    assert loaded["epoch"] == 7
    assert torch.equal(loaded["tensor"], torch.arange(4))


def test_resume_reconciles_stale_and_torn_history_from_checkpoint(
    tmp_path: Path,
) -> None:
    history = tmp_path / "history.jsonl"
    history.write_text(
        "\n".join(
            (
                json.dumps(_row(0, dice=0.1)),
                json.dumps(_row(1, dice=0.2)),
                json.dumps(_row(2, dice=0.0)),
                '{"epoch":',
            )
        ),
        encoding="utf-8",
    )
    authoritative = _row(2, dice=0.3)

    rows = _reconcile_history(
        history,
        checkpoint_epoch=2,
        checkpoint_metrics=authoritative,
    )

    assert [row["epoch"] for row in rows] == [0, 1, 2]
    assert rows[-1] == authoritative
    persisted = [json.loads(line) for line in history.read_text().splitlines()]
    assert persisted == rows


def test_resume_reconciles_best_artifacts_from_last_checkpoint(tmp_path: Path) -> None:
    best = tmp_path / "checkpoint_best.pt"
    best_trained = tmp_path / "checkpoint_best_trained.pt"
    torch.save({"epoch": 0, "model": {"weight": torch.tensor([0.0])}}, best)
    payload = {
        "epoch": 1,
        "metrics": _row(1, dice=0.3),
        "best_score": 0.3,
        "best_trained_score": 0.3,
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {"state": "large"},
        "checkpoint_roles": {"best": True, "best_trained": True},
    }

    _reconcile_best_checkpoint_artifacts(
        payload,
        best_checkpoint=best,
        best_trained_checkpoint=best_trained,
        final_fit=False,
    )

    for path in (best, best_trained):
        repaired = torch.load(path, map_location="cpu", weights_only=False)
        assert repaired["epoch"] == 1
        assert torch.equal(repaired["model"]["weight"], torch.tensor([1.0]))
        assert "optimizer" not in repaired


def test_resume_discards_only_uncommitted_milestone_suffix(tmp_path: Path) -> None:
    first = tmp_path / "checkpoint_milestone_00000001.pt"
    second = tmp_path / "checkpoint_milestone_00000002.pt"
    first.write_bytes(b"committed")
    second.write_bytes(b"written-before-power-loss")
    index_path = tmp_path / "checkpoint_milestones.json"
    index_path.write_text(
        json.dumps(
            {
                "schema": SNAPSHOT_CHECKPOINT_CONTRACT,
                "records": [
                    {
                        "requested_samples": 1,
                        "actual_samples": 1,
                        "checkpoint": first.name,
                        "bytes": first.stat().st_size,
                    },
                    {
                        "requested_samples": 2,
                        "actual_samples": 2,
                        "checkpoint": second.name,
                        "bytes": second.stat().st_size,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    records = _reconcile_snapshot_records(
        tmp_path,
        index_path,
        snapshots=(1, 2),
        cumulative_samples=1,
    )

    assert [row["requested_samples"] for row in records] == [1]
    persisted = json.loads(index_path.read_text(encoding="utf-8"))
    assert persisted["records"] == records


def test_resume_rejects_a_gap_before_the_committed_checkpoint(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    history.write_text(json.dumps(_row(0, dice=0.1)) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="committed history epochs"):
        _reconcile_history(
            history,
            checkpoint_epoch=2,
            checkpoint_metrics=_row(2, dice=0.3),
        )


def test_checkpoint_rng_state_round_trips() -> None:
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(41)
    random.seed(42)
    np.random.seed(43)
    torch.manual_seed(44)
    state = _capture_rng_state(generator, device)
    expected = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
        float(torch.rand((), generator=generator)),
    )

    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    generator.manual_seed(4)
    _restore_rng_state(state, loader_generator=generator, device=device)
    actual = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
        float(torch.rand((), generator=generator)),
    )

    assert actual == expected


def test_precheckpoint_resume_and_committed_history_repair(tmp_path: Path) -> None:
    image = np.arange(8**3, dtype=np.uint16).reshape(8, 8, 8) % 213
    target = np.zeros((8, 8, 8), dtype=np.uint8)
    target[3:5] = 1
    rows = []
    for split in ("train", "val"):
        patch = tmp_path / f"{split}.npz"
        np.savez(patch, image=image, target_u8=target)
        rows.append(
            {
                "schema": "crossres-voxel-patch-v1",
                "patch_id": split,
                "path": str(patch),
                "record_id": split,
                "scroll_id": f"scroll-{split}",
                "split": split,
                "origin_zyx": [0, 0, 0],
                "shape_zyx": [8, 8, 8],
                "known_fraction": 1.0,
                "positive_fraction_known": 0.25,
            }
        )
    manifest = tmp_path / "patches.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    options = TrainOptions(
        epochs=1,
        batch_size=1,
        accumulate=1,
        num_workers=0,
        seed=1203,
        device="cpu",
        amp=False,
        preset="tiny-test",
        max_cpu_threads=1,
    )
    output = tmp_path / "run"
    output.mkdir()
    identity = json.loads(
        json.dumps(
            {
                "patch_manifest": str(manifest.resolve()),
                "patch_manifest_sha256": _sha256(manifest),
                "loss_contract": LOSS_CONTRACT,
                "checkpoint_selection_contract": CHECKPOINT_SELECTION_CONTRACT,
                "checkpoint_durability_contract": CHECKPOINT_DURABILITY_CONTRACT,
                "epoch_partition_contract": EPOCH_PARTITION_CONTRACT,
                "sampling_contract": SAMPLING_CONTRACT,
                "learning_rate_contract": LEARNING_RATE_CONTRACT,
                "snapshot_checkpoint_contract": SNAPSHOT_CHECKPOINT_CONTRACT,
                "snapshot_samples": [1],
                "effective_partition_samples": 1,
                "resolved_schedule": {
                    "evaluation_interval_samples": 1,
                    "total_samples": 1,
                    "evaluation_intervals": 1,
                },
                "options": _options_identity(options),
            },
            default=str,
        )
    )
    (output / "run.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    best = train_model(
        patch_manifest=manifest,
        output_path=output,
        options=options,
        resume=True,
        snapshot_samples=(1,),
    )
    assert best.is_file()
    milestone = output / "checkpoint_milestone_00000001.pt"
    index = json.loads((output / "checkpoint_milestones.json").read_text())
    assert index == {
        "schema": SNAPSHOT_CHECKPOINT_CONTRACT,
        "records": [
            {
                "requested_samples": 1,
                "actual_samples": 1,
                "checkpoint": milestone.name,
                "bytes": milestone.stat().st_size,
            }
        ],
    }
    snapshot = torch.load(milestone, map_location="cpu", weights_only=False)
    assert snapshot["checkpoint_kind"] == "sample-milestone"
    assert snapshot["requested_samples"] == snapshot["cumulative_samples"] == 1
    assert "optimizer" not in snapshot
    committed = torch.load(
        output / "checkpoint_last.pt", map_location="cpu", weights_only=False
    )
    assert set(committed["checkpoint_roles"]) == {"best", "best_trained"}
    assert committed["checkpoint_roles"]["best_trained"] is True
    history = output / "history.jsonl"
    authoritative = json.loads(history.read_text())
    with history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_row(0, dice=0.0)) + "\n")
        stream.write('{"epoch":')

    assert (
        train_model(
            patch_manifest=manifest,
            output_path=output,
            options=options,
            resume=True,
            snapshot_samples=(1,),
        )
        == best
    )
    repaired = [json.loads(line) for line in history.read_text().splitlines()]
    assert repaired == [authoritative]
