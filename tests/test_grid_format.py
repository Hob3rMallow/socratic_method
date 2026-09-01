from __future__ import annotations

import json
from pathlib import Path

import crossres_pred.inference as inference_module
import numpy as np
import pytest
import tifffile
import torch
from _synthetic import write_local_fine_store
from crossres_pred.extract import coverage_for_mirror
from crossres_pred.inference import (
    InferenceError,
    InferOptions,
    TeacherInferOptions,
    format_cube_id,
    infer_grid,
    infer_teacher,
    parse_cube_id,
)
from crossres_pred.model import SurfaceModelConfig, SurfaceNet


def _write_cube(path: Path, array: np.ndarray) -> None:
    tifffile.imwrite(
        path,
        np.ascontiguousarray(array),
        byteorder="<",
        photometric="minisblack",
        compression=None,
        metadata=None,
        rowsperstrip=array.shape[-2],
    )


def _make_grid(tmp_path: Path, *, with_baseline: bool = False) -> Path:
    grid = tmp_path / "grid"
    raw_dir = grid / "cubes_RAW"
    raw_dir.mkdir(parents=True)
    (grid / "manifest.json").write_text(
        json.dumps({"chunk_size": 64, "n_chunks": 2}), encoding="utf-8"
    )
    rng = np.random.default_rng(4)
    for cube_id in ("z00000_y00000_x00000", "z00000_y00000_x00064"):
        raw = rng.integers(20, 90, size=(64, 64, 64), dtype=np.int64).astype(
            np.uint8
        )
        raw[10:13] = 170
        _write_cube(raw_dir / f"{cube_id}.tif", raw)
        if with_baseline:
            pred_dir = grid / "cubes_PRED"
            pred_dir.mkdir(exist_ok=True)
            baseline = np.zeros((64, 64, 64), dtype=np.uint8)
            baseline[10:13] = 255
            _write_cube(pred_dir / f"{cube_id}.tif", baseline)
    return grid


def _make_checkpoint(
    tmp_path: Path, *, in_channels: int = 1, deploy_threshold: float | None = None
) -> Path:
    model = SurfaceNet(SurfaceModelConfig(in_channels=in_channels))
    payload = {
        "schema_version": 2,
        "kind": "crossres-surface-training",
        "profile": "student",
        "epoch": 1,
        "model_config": model.config.as_dict(),
        "model_state": model.state_dict(),
        "policy_profile": "research",
        "deploy_threshold": deploy_threshold,
    }
    checkpoint = tmp_path / "model.pt"
    torch.save(payload, checkpoint)
    return checkpoint


def test_cube_id_round_trip() -> None:
    assert parse_cube_id("z04480_y03328_x02816") == (4480, 3328, 2816)
    assert format_cube_id((4480, 3328, 2816)) == "z04480_y03328_x02816"
    with pytest.raises(InferenceError, match="invalid cube id"):
        parse_cube_id("cube_1")


def test_infer_grid_emits_the_c_pipeline_contract(tmp_path: Path) -> None:
    grid = _make_grid(tmp_path)
    checkpoint = _make_checkpoint(tmp_path)
    output = infer_grid(
        checkpoint_path=checkpoint,
        source_grid=grid,
        output_path=tmp_path / "out",
        options=InferOptions(halo=0, device="cpu", amp=False, raw_mode="copy"),
    )
    pred_dir = output / "cubes_PRED"
    cube_ids = json.loads((pred_dir / "present.json").read_text())
    assert cube_ids == ["z00000_y00000_x00000", "z00000_y00000_x00064"]
    for cube_id in cube_ids:
        path = pred_dir / f"{cube_id}.tif"
        with path.open("rb") as stream:
            assert stream.read(4) == b"II*\x00"
        with tifffile.TiffFile(path) as tif:
            assert len(tif.pages) == 64
            assert all(page.shape == (64, 64) for page in tif.pages)
            assert all(page.dtype == np.dtype(np.uint8) for page in tif.pages)
            assert all(page.compression == 1 for page in tif.pages)
        values = np.unique(tifffile.imread(path))
        assert {int(item) for item in values}.issubset({0, 255})
        assert (output / "cubes_RAW" / f"{cube_id}.tif").is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["crossres_pred"]["schema_version"] == 2
    assert manifest["crossres_pred"]["policy_profile"] == "research"
    assert manifest["n_pred_tiffs_emitted"] == 2
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["threshold_source"] == "default"
    assert summary["threshold"] == 0.5
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["status"] == "complete"


def test_threshold_resolution_prefers_cli_then_checkpoint(tmp_path: Path) -> None:
    grid = _make_grid(tmp_path)
    checkpoint = _make_checkpoint(tmp_path, deploy_threshold=0.7)
    output = infer_grid(
        checkpoint_path=checkpoint,
        source_grid=grid,
        output_path=tmp_path / "out_checkpoint",
        options=InferOptions(halo=0, device="cpu", amp=False, raw_mode="none"),
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["threshold"] == pytest.approx(0.7)
    assert summary["threshold_source"] == "checkpoint"

    output = infer_grid(
        checkpoint_path=checkpoint,
        source_grid=grid,
        output_path=tmp_path / "out_cli",
        options=InferOptions(
            halo=0, device="cpu", amp=False, raw_mode="none", threshold=0.9
        ),
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["threshold"] == pytest.approx(0.9)
    assert summary["threshold_source"] == "cli"


def test_baseline_comparison_is_logged_when_available(tmp_path: Path) -> None:
    grid = _make_grid(tmp_path, with_baseline=True)
    checkpoint = _make_checkpoint(tmp_path)
    output = infer_grid(
        checkpoint_path=checkpoint,
        source_grid=grid,
        output_path=tmp_path / "out",
        options=InferOptions(halo=0, device="cpu", amp=False, raw_mode="none"),
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["compared_cubes"] == 2
    assert summary["mean_baseline_dice"] is not None
    events = [
        json.loads(line)
        for line in (output / "events.jsonl").read_text().splitlines()
        if line
    ]
    cube_events = [row for row in events if row["event"] == "cube_complete"]
    assert all("baseline_dice" in row for row in cube_events)


def test_two_channel_checkpoint_requires_baseline_cubes(tmp_path: Path) -> None:
    grid = _make_grid(tmp_path)
    checkpoint = _make_checkpoint(tmp_path, in_channels=2)
    with pytest.raises(InferenceError, match="2-channel"):
        infer_grid(
            checkpoint_path=checkpoint,
            source_grid=grid,
            output_path=tmp_path / "out",
            options=InferOptions(halo=0, device="cpu", amp=False),
        )


def test_missing_policy_profile_is_rejected(tmp_path: Path) -> None:
    grid = _make_grid(tmp_path)
    model = SurfaceNet(SurfaceModelConfig(in_channels=1))
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "schema_version": 2,
            "model_config": model.config.as_dict(),
            "model_state": model.state_dict(),
        },
        checkpoint,
    )
    with pytest.raises(InferenceError, match="policy profile"):
        infer_grid(
            checkpoint_path=checkpoint,
            source_grid=grid,
            output_path=tmp_path / "out",
            options=InferOptions(halo=0, device="cpu", amp=False),
        )


def test_infer_teacher_writes_sparse_soft_probs_and_voxel_validity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mirror = write_local_fine_store(
        tmp_path, shape=(128, 128, 128), chunks=(32, 32, 32)
    )
    # One selected chunk is a real masked scan void. It remains selected at
    # chunk level but must be invalid at voxel level in the bridge contract.
    (mirror / "0" / "0" / "0" / "0").write_bytes(
        np.zeros((32, 32, 32), dtype=np.uint8).tobytes()
    )
    (mirror / "carve_selected_chunks.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "array_key": "0",
                "chunks_zyx": [32, 32, 32],
                "chunk_grid_zyx": [4, 4, 4],
                "selected_chunk_ids": list(range(64)),
            }
        ),
        encoding="utf-8",
    )
    (mirror / "crossres_sparse_mirror.json").write_text(
        json.dumps({"schema_version": 1, "state": "complete"}),
        encoding="utf-8",
    )
    sites = tmp_path / "sites.jsonl"
    site_row = {
        "record_id": "teacher-test",
        "site_id": "teacher-test_s0000",
        "fine_bbox_lo_zyx": [0, 0, 0],
        "fine_bbox_hi_zyx": [128, 128, 128],
    }
    sites.write_text(json.dumps(site_row) + "\n", encoding="utf-8")
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"synthetic teacher checkpoint")

    class TinyTeacher(torch.nn.Module):
        required_divisor = 1
        config = type("Config", (), {"in_channels": 1})()

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value[:, :1]

    monkeypatch.setattr(
        inference_module,
        "load_surface_checkpoint",
        lambda _path, _device: (
            TinyTeacher(),
            {
                "profile": "teacher",
                "policy_profile": "research",
                "epoch": 2,
                "val_selection": {"average_precision": 0.8},
            },
        ),
    )
    output = infer_teacher(
        checkpoint_path=checkpoint,
        site_rows=[site_row],
        site_manifest_path=sites,
        record_id="teacher-test",
        mirror_path=mirror,
        output_path=tmp_path / "teacher_pred.zarr",
        policy_profile="research",
        options=TeacherInferOptions(
            patch_shape_zyx=(64, 64, 64),
            stride=32,
            retained_margin=0,
            batch_size=4,
            device="cpu",
            amp=False,
        ),
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["tile_count"] == 27
    assert summary["predicted_chunk_count"] == 64

    import zarr

    root = zarr.open(str(output), mode="r")
    assert tuple(root["0"].shape) == (128, 128, 128)
    assert root["0"].dtype == np.dtype(np.uint8)
    valid = np.asarray(root["1"][:64, :64, :64])
    assert not valid[:32, :32, :32].any()
    assert valid[32:, 32:, 32:].all()
    coverage = coverage_for_mirror(output)
    covered = coverage((0, 0, 0), (64, 64, 64))
    assert not covered[:32, :32, :32].any()
    assert covered[32:, 32:, 32:].all()
