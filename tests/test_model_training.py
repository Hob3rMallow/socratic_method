from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from crossres_pred.model import (
    SurfaceModelConfig,
    SurfaceNet,
    initialize_from_m7_checkpoint,
    initialize_from_surface_checkpoint,
    load_surface_checkpoint,
)
from crossres_pred.train import TrainOptions, train_model

from _synthetic import make_student_patch, student_row, write_student_manifest

M7_CHECKPOINT = Path(
    "D:/work/vesuvius-c/output/crossres_data/models/surface_m7_nnunet/"
    "fold_0/checkpoint_best.pth"
)


@pytest.fixture(scope="module")
def surface_net() -> SurfaceNet:
    model = SurfaceNet(SurfaceModelConfig(in_channels=1))
    model.eval()
    return model


def test_forward_shape_and_divisor(surface_net: SurfaceNet) -> None:
    with torch.no_grad():
        logits = surface_net(torch.zeros((1, 1, 64, 64, 64)))
    assert logits.shape == (1, 1, 64, 64, 64)
    with pytest.raises(ValueError, match="divisible"):
        surface_net(torch.zeros((1, 1, 96, 96, 65)))
    with pytest.raises(ValueError, match="at least 64"):
        surface_net(torch.zeros((1, 1, 32, 32, 32)))
    with pytest.raises(ValueError, match="expected"):
        surface_net(torch.zeros((1, 2, 64, 64, 64)))


def _corpus(tmp_path: Path) -> Path:
    rows = []
    for index in range(3):
        patch_id = f"train{index}"
        make_student_patch(tmp_path, patch_id=patch_id, seed=index)
        rows.append(student_row(patch_id, scroll_id="PHerc1667", split="train"))
    make_student_patch(tmp_path, patch_id="val0", seed=9)
    rows.append(student_row("val0", scroll_id="PHerc0814", split="val"))
    return write_student_manifest(tmp_path, rows=rows)


def _options(**overrides) -> TrainOptions:
    values = dict(
        profile="student",
        epochs=2,
        batch_size=1,
        accumulate=1,
        learning_rate=1.0e-4,
        warmup_steps=0,
        num_workers=0,
        seed=11,
        device="cpu",
        amp=False,
        init_mode="none",
        eval_interior_margin=4,
    )
    values.update(overrides)
    return TrainOptions(**values)


def test_student_training_selects_on_distill_ap(tmp_path: Path) -> None:
    manifest = _corpus(tmp_path)
    output = tmp_path / "run"
    best = train_model(
        patch_manifest=manifest,
        output_path=output,
        options=_options(),
    )
    assert best == output / "best.pt"
    assert (output / "last.pt").is_file()
    payload = torch.load(best, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == 2
    assert payload["profile"] == "student"
    assert payload["policy_profile"] == "research"
    assert payload["deploy_threshold"] is None
    selection = payload["val_selection"]
    assert selection["selection"] == "distill-target-ap"
    assert selection["average_precision"] is not None
    history = [
        json.loads(line)
        for line in (output / "history.jsonl").read_text().splitlines()
        if line
    ]
    assert [row["epoch"] for row in history] == [1, 2]
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["kind"] == "crossres-surface-training"
    assert provenance["rot90_mode"] == "z-only"


def test_resume_extends_history_strictly(tmp_path: Path) -> None:
    manifest = _corpus(tmp_path)
    output = tmp_path / "run"
    train_model(
        patch_manifest=manifest, output_path=output, options=_options()
    )
    train_model(
        patch_manifest=manifest,
        output_path=output,
        options=_options(epochs=3),
        resume_checkpoint=output / "last.pt",
    )
    history = [
        json.loads(line)
        for line in (output / "history.jsonl").read_text().splitlines()
        if line
    ]
    assert [row["epoch"] for row in history] == [1, 2, 3]
    provenance = json.loads((output / "provenance.json").read_text())
    assert len(provenance["resume_events"]) == 1

    with pytest.raises(ValueError, match="mismatch"):
        train_model(
            patch_manifest=manifest,
            output_path=output,
            options=_options(epochs=4, learning_rate=5.0e-4),
            resume_checkpoint=output / "last.pt",
        )


def test_profile_kind_mismatch_is_rejected(tmp_path: Path) -> None:
    manifest = _corpus(tmp_path)
    with pytest.raises(ValueError, match="kind"):
        train_model(
            patch_manifest=manifest,
            output_path=tmp_path / "run",
            options=_options(profile="teacher"),
        )


def test_final_fit_requires_no_validation_rows(tmp_path: Path) -> None:
    rows = []
    for index in range(2):
        patch_id = f"train{index}"
        make_student_patch(tmp_path, patch_id=patch_id, seed=index)
        rows.append(student_row(patch_id, scroll_id="PHerc1667", split="train"))
    manifest = write_student_manifest(tmp_path, rows=rows)
    output = tmp_path / "run"
    result = train_model(
        patch_manifest=manifest,
        output_path=output,
        options=_options(final_fit=True, epochs=1),
    )
    assert result == output / "last.pt"

    with pytest.raises(ValueError, match="no validation"):
        train_model(
            patch_manifest=manifest,
            output_path=tmp_path / "run2",
            options=_options(epochs=1),
        )


def test_surface_checkpoint_initialization_widens_the_stem(
    tmp_path: Path, surface_net: SurfaceNet
) -> None:
    checkpoint = tmp_path / "teacher.pt"
    torch.save(
        {
            "schema_version": 2,
            "model_config": surface_net.config.as_dict(),
            "model_state": surface_net.state_dict(),
            "epoch": 5,
            "train_options": {"profile": "teacher"},
        },
        checkpoint,
    )
    same = SurfaceNet(SurfaceModelConfig(in_channels=1))
    info = initialize_from_surface_checkpoint(same, checkpoint)
    assert info["adapted_stem_keys"] == []
    stem_key = next(
        key for key in surface_net.state_dict() if key.endswith("stem.convs.0.conv.weight")
    )
    assert torch.equal(
        same.state_dict()[stem_key], surface_net.state_dict()[stem_key]
    )

    widened = SurfaceNet(SurfaceModelConfig(in_channels=2))
    info = initialize_from_surface_checkpoint(widened, checkpoint)
    # One stem parameter, four aliased state-dict keys (encoder + the
    # decoder's encoder reference, each with conv/all_modules views).
    assert len(info["adapted_stem_keys"]) == 4
    assert all("stem.convs.0" in key for key in info["adapted_stem_keys"])
    stem = widened.state_dict()[stem_key]
    assert torch.equal(stem[:, 0:1], surface_net.state_dict()[stem_key])
    assert float(stem[:, 1:2].abs().sum()) == 0.0


def test_load_surface_checkpoint_round_trip(
    tmp_path: Path, surface_net: SurfaceNet
) -> None:
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "schema_version": 2,
            "model_config": surface_net.config.as_dict(),
            "model_state": surface_net.state_dict(),
        },
        checkpoint,
    )
    model, payload = load_surface_checkpoint(checkpoint, torch.device("cpu"))
    assert model.config == surface_net.config
    with torch.no_grad():
        value = torch.randn((1, 1, 64, 64, 64))
        assert torch.allclose(model(value), surface_net(value))


@pytest.mark.skipif(
    not M7_CHECKPOINT.is_file(), reason="released m7 checkpoint not present"
)
def test_m7_weight_surgery_on_the_released_checkpoint() -> None:
    model = SurfaceNet(SurfaceModelConfig(in_channels=1))
    info = initialize_from_m7_checkpoint(model, M7_CHECKPOINT)
    assert info["copied_state_keys"] > 900
    assert info["adapted_stem_keys"] == []
    # The converted head is w[1]-w[0]: a forward pass must produce finite,
    # non-degenerate logits on plausible normalized input.
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn((1, 1, 64, 64, 64)) * 0.5)
    assert torch.isfinite(logits).all()
    assert float(logits.std()) > 1.0e-4
