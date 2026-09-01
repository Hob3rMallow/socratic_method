from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossres_pred.pathmap import remap_embedded_path, remap_volume_spec
from socratic_method.recipe import build_command, load_paths


def test_path_mapping_preserves_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOCRATIC_ORIGINAL_ROOT", r"D:\work\vesuvius-c")
    monkeypatch.setenv("SOCRATIC_ARTIFACT_ROOT", "/data/vesuvius-c")
    mapped = remap_embedded_path(
        r"D:\work\vesuvius-c\output\crossres_data\atlas\state.json"
    )
    assert mapped == Path("/data/vesuvius-c/output/crossres_data/atlas/state.json")
    assert remap_volume_spec(
        r"D:\work\vesuvius-c\output\volume.zarr::0"
    ) == "/data/vesuvius-c/output/volume.zarr::0"


def test_path_mapping_requires_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOCRATIC_ORIGINAL_ROOT", r"D:\work\vesuvius-c")
    monkeypatch.delenv("SOCRATIC_ARTIFACT_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="must be set together"):
        remap_embedded_path(r"D:\work\vesuvius-c\output\x")


def test_command_substitution_and_resume() -> None:
    recipe = {
        "training": {
            "argv": ["-m", "crossres_pred.voxel", "train", "--patches", "{train_manifest}"]
        }
    }
    command = build_command(
        recipe,
        {"train_manifest": "/data/train.jsonl"},
        python="python-test",
        resume=True,
    )
    assert command == [
        "python-test",
        "-m",
        "crossres_pred.voxel",
        "train",
        "--patches",
        "/data/train.jsonl",
        "--resume",
    ]


def test_load_paths_resolves_machine_local_values(tmp_path: Path) -> None:
    value = {
        "train_manifest": "data/train.jsonl",
        "validation_manifest": "data/val.jsonl",
        "m7_checkpoint": "data/m7.pth",
        "dynamic_medial_connectivity_state": "data/state.json",
        "output": "runs/v31",
    }
    path = tmp_path / "paths.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    loaded = load_paths(path)
    assert loaded["train_manifest"] == str((tmp_path / "data/train.jsonl").resolve())
    assert loaded["output"] == str((tmp_path / "runs/v31").resolve())
