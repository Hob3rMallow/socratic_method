from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from crossres_pred.voxel.final_fit import verify_final_fit
from crossres_pred.voxel.loss import LOSS_CONTRACT
from crossres_pred.voxel.train import (
    CHECKPOINT_DURABILITY_CONTRACT,
    CHECKPOINT_SELECTION_CONTRACT,
    EPOCH_PARTITION_CONTRACT,
    FINAL_FIT_CHECKPOINT_CONTRACT,
    LEARNING_RATE_CONTRACT,
    SAMPLING_CONTRACT,
    TrainOptions,
    _options_identity,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, mismatch: bool = False) -> dict[str, Path]:
    manifest = tmp_path / "patches.jsonl"
    manifest.write_text(
        json.dumps({"split": "train", "scroll_id": "PHerc0139"})
        + "\n"
        + json.dumps({"split": "val", "scroll_id": "PHerc0814"})
        + "\n",
        encoding="utf-8",
    )
    m7 = tmp_path / "m7.pth"
    m7.write_bytes(b"m7")
    options = TrainOptions(
        epochs=2,
        batch_size=3,
        accumulate=1,
        num_workers=0,
        device="cuda",
        amp_dtype="bfloat16",
        preset="m7-resenc-l",
        pretrained_m7_checkpoint=str(m7),
        final_fit=True,
        max_cpu_threads=16,
        samples_per_epoch=1,
        max_train_samples=2,
    )
    run_identity = {
        "patch_manifest": str(manifest.resolve()),
        "patch_manifest_sha256": _sha256(manifest),
        "loss_contract": LOSS_CONTRACT,
        "checkpoint_selection_contract": CHECKPOINT_SELECTION_CONTRACT,
        "checkpoint_durability_contract": CHECKPOINT_DURABILITY_CONTRACT,
        "epoch_partition_contract": EPOCH_PARTITION_CONTRACT,
        "sampling_contract": SAMPLING_CONTRACT,
        "learning_rate_contract": LEARNING_RATE_CONTRACT,
        "effective_partition_samples": 2,
        "resolved_schedule": {
            "evaluation_interval_samples": 1,
            "total_samples": 2,
            "evaluation_intervals": 2,
        },
        "final_fit_checkpoint_contract": FINAL_FIT_CHECKPOINT_CONTRACT,
        "options": _options_identity(options),
    }
    run_identity = json.loads(json.dumps(run_identity))
    run = tmp_path / "run.json"
    run.write_text(json.dumps(run_identity), encoding="utf-8")
    rows = [
        {
            "epoch": epoch,
            "learning_rate": 0.001,
            "train": {"loss_total": 1.0 + epoch, "samples": 1.0},
            "val": {},
        }
        for epoch in range(2)
    ]
    history = tmp_path / "history.jsonl"
    history.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    model = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    best_model = {
        "weight": model["weight"] + (1.0 if mismatch else 0.0),
    }
    common = {
        "epoch": 1,
        "best_score": -2.0,
        "model_config": {"preset": "m7-resenc-l"},
        "scaler": {},
        "initialization": {"checkpoint": str(m7.resolve()), "strict": True},
        "identity": run_identity,
        "metrics": rows[-1],
        "rng_state": {"test": True},
    }
    last = tmp_path / "checkpoint_last.pt"
    best = tmp_path / "checkpoint_best.pt"
    torch.save({**common, "model": model, "optimizer": {"state": {}}}, last)
    torch.save({**common, "model": best_model}, best)

    tuning = tmp_path / "tuning.pt"
    tuning.write_bytes(b"tuning")
    audit = tmp_path / "audit.json"
    audit.write_text("{}", encoding="utf-8")
    qualification = tmp_path / "qualification.json"
    qualification.write_text(
        json.dumps(
            {
                "schema": "crossres-voxel-tuning-qualification-v4",
                "qualified": True,
                "patch_manifest_sha256": _sha256(manifest),
                "checkpoint": str(tuning),
                "checkpoint_sha256": _sha256(tuning),
                "audit_report": str(audit),
                "audit_report_sha256": _sha256(audit),
                "threshold": 0.45,
                "selected_train_samples": 2,
                "inference_policy": {
                    "split": "val",
                    "amp_dtype": "bfloat16",
                    "mirror_tta": True,
                    "qualification_scroll": "PHerc0814",
                    "baseline_comparison_policy": "matched-rows-only",
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "qualification": qualification,
        "best": best,
        "last": last,
        "history": history,
        "manifest": manifest,
    }


def _verify(paths: dict[str, Path], output: Path) -> Path:
    return verify_final_fit(
        qualification_path=paths["qualification"],
        best_checkpoint_path=paths["best"],
        last_checkpoint_path=paths["last"],
        history_path=paths["history"],
        patch_manifest_path=paths["manifest"],
        output_path=output,
        expected_records=2,
        expected_epochs=2,
        expected_samples_per_epoch=1,
    )


def test_final_fit_verification_binds_last_epoch_and_reuses(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "verification.json"
    assert _verify(paths, output) == output.resolve()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["state"] == "verified"
    assert payload["summary"]["best_equals_last_model"]
    assert payload["summary"]["scheduled_samples"] == 2
    assert payload["summary"]["epoch_samples"] == [1, 1]
    assert _verify(paths, output) == output.resolve()


def test_final_fit_verification_rejects_earlier_best_weights(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="model state 'weight' differs"):
        _verify(paths, tmp_path / "verification.json")
