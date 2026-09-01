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
)


def _write_contract(tmp_path: Path) -> tuple[Path, Path, Path]:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"selected checkpoint fixture")
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
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
