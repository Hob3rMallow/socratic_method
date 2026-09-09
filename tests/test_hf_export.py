from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from socratic_method.hf_export import (
    _preprocessor_config,
    _release_contract,
    _validate_selected_checkpoint,
    _validate_selection_summary,
    deduplicate_shared_tensors,
    load_exported_state_dict,
)


def test_shared_tensors_are_stored_once_and_reexpanded(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    shared = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    weights = {
        "network.encoder.stem.weight": shared,
        "network.decoder.encoder.stem.weight": shared,
        "network.encoder.stem.all_modules.0.weight": shared,
        "network.head.bias": torch.ones(3),
        "copy.weight": shared.clone(),
    }
    stored, tied = deduplicate_shared_tensors(weights)
    # The canonical key of a shared group is its lexicographically first name.
    assert sorted(stored) == ["copy.weight", "network.decoder.encoder.stem.weight", "network.head.bias"]
    assert tied == {
        "network.encoder.stem.all_modules.0.weight": "network.decoder.encoder.stem.weight",
        "network.encoder.stem.weight": "network.decoder.encoder.stem.weight",
    }
    path = tmp_path / "model.safetensors"
    safetensors_torch.save_file(
        {k: v.contiguous() for k, v in stored.items()},
        str(path),
        metadata={"tied_keys": json.dumps(tied, sort_keys=True)},
    )
    loaded = load_exported_state_dict(path)
    assert sorted(loaded) == sorted(weights)
    for name, tensor in weights.items():
        assert torch.equal(loaded[name], tensor)


def _write_contract(tmp_path: Path) -> tuple[Path, Path, Path]:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"selected checkpoint fixture")
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "version": "fixture-recipe-v1",
                "release": {
                    "status": "selected",
                    "selected_checkpoint": {
                        "samples": 8192,
                        "sha256": checksum,
                        "bytes": checkpoint.stat().st_size,
                    },
                    "operating_threshold": 0.45,
                    "threshold_selection_contract": "registered-review-v1",
                    "inference": "raw-student-only-no-m7-blend-no-teacher",
                }
            }
        ),
        encoding="utf-8",
    )
    qualification = tmp_path / "release_qualification.json"
    qualification.write_text(
        json.dumps(
            {
                "selection": {
                    "checkpoint_samples": 8192,
                    "checkpoint_sha256": checksum,
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "operating_threshold": 0.45,
                    "model_composition": "raw-student-only-no-m7-blend-no-teacher",
                }
            }
        ),
        encoding="utf-8",
    )
    return checkpoint, recipe, qualification


def test_selected_release_contract_is_fail_closed(tmp_path: Path) -> None:
    checkpoint, recipe, qualification = _write_contract(tmp_path)
    contract = _release_contract(recipe, qualification)
    assert _validate_selected_checkpoint(checkpoint, contract) == contract["sha256"]
    assert contract["samples"] == 8192
    assert _preprocessor_config(contract)["operating_threshold"] == 0.45

    checkpoint.write_bytes(b"tampered checkpoint fixture")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _validate_selected_checkpoint(checkpoint, contract)


def test_qualification_must_match_recipe(tmp_path: Path) -> None:
    _, recipe, qualification = _write_contract(tmp_path)
    value = json.loads(qualification.read_text(encoding="utf-8"))
    value["selection"]["operating_threshold"] = 0.42
    qualification.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="operating_threshold"):
        _release_contract(recipe, qualification)


def test_f0_shaped_recipe_drives_version_status_and_model_type(tmp_path: Path) -> None:
    _, recipe, qualification = _write_contract(tmp_path)
    value = json.loads(recipe.read_text(encoding="utf-8"))
    value["version"] = "c3-f0-200k-20260907"
    value["model"] = {"hf_model_type": "socratic-c3-f0"}
    value["release"].update(
        selection_status="frozen-model-selected",
        threshold_status="fixed by preregistration",
    )
    recipe.write_text(json.dumps(value), encoding="utf-8")
    contract = _release_contract(recipe, qualification)
    assert contract["version"] == "c3-f0-200k-20260907"
    assert contract["model_type"] == "socratic-c3-f0"
    assert contract["selection_status"] == "frozen-model-selected"
    assert _preprocessor_config(contract)["threshold_status"] == "fixed by preregistration"
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "status": "frozen-model-selected",
                "samples": 8192,
                "threshold": 0.45,
                "checkpoint_sha256": contract["sha256"],
            }
        ),
        encoding="utf-8",
    )
    _validate_selection_summary(selection, contract)


def test_v31_shaped_recipe_keeps_the_historical_defaults(tmp_path: Path) -> None:
    _, recipe, qualification = _write_contract(tmp_path)
    value = json.loads(recipe.read_text(encoding="utf-8"))
    value["version"] = "v31.1-pherc0139-dynamic-medial-duration-8192"
    recipe.write_text(json.dumps(value), encoding="utf-8")
    contract = _release_contract(recipe, qualification)
    assert contract["model_type"] == "socratic-m7-xr"
    assert contract["selection_status"] == "release-candidate-selected"
    assert _preprocessor_config(contract)["threshold_status"].startswith("selected by registered morphology")
    value.pop("version")
    recipe.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        _release_contract(recipe, qualification)


def test_selection_summary_must_match_recipe(tmp_path: Path) -> None:
    _, recipe, qualification = _write_contract(tmp_path)
    contract = _release_contract(recipe, qualification)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "status": "release-candidate-selected",
                "samples": 8192,
                "threshold": 0.45,
                "checkpoint_sha256": contract["sha256"],
            }
        ),
        encoding="utf-8",
    )
    _validate_selection_summary(selection, contract)
    value = json.loads(selection.read_text(encoding="utf-8"))
    value["threshold"] = 0.42
    selection.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="threshold"):
        _validate_selection_summary(selection, contract)


def test_test_time_augmentation_flows_from_the_recipe(tmp_path: Path) -> None:
    _, recipe, qualification = _write_contract(tmp_path)
    contract = _release_contract(recipe, qualification)
    assert contract["test_time_augmentation"] == "none"
    assert _preprocessor_config(contract)["test_time_augmentation"] == "none"
    value = json.loads(recipe.read_text(encoding="utf-8"))
    value["release"]["test_time_augmentation"] = "8-way-mirror"
    recipe.write_text(json.dumps(value), encoding="utf-8")
    contract = _release_contract(recipe, qualification)
    assert contract["test_time_augmentation"] == "8-way-mirror"
    assert _preprocessor_config(contract)["test_time_augmentation"] == "8-way-mirror"
    qualified = json.loads(qualification.read_text(encoding="utf-8"))
    qualified["selection"]["test_time_augmentation"] = "none"
    qualification.write_text(json.dumps(qualified), encoding="utf-8")
    with pytest.raises(ValueError, match="test_time_augmentation"):
        _release_contract(recipe, qualification)
    value["release"]["test_time_augmentation"] = "sixteen-way"
    recipe.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="test_time_augmentation"):
        _release_contract(recipe, qualification)

