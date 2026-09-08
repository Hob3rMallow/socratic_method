from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from socratic_method import repair


@pytest.fixture
def grid(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "cubes_PRED").mkdir(parents=True)
    metadata = {
        "checkpoint_sha256": repair.MODEL_SHA256,
        "threshold": 0.35,
        "chunk_size": 128,
        "umbilicus_yx": [3287, 3784],
    }
    (source / "manifest.json").write_text(json.dumps(metadata))
    for z, y, x in product(range(3), repeat=3):
        (
            source / "cubes_PRED" / f"z{z * 128:05d}_y{y * 128:05d}_x{x * 128:05d}.tif"
        ).write_bytes(b"test fixture")
    binary = tmp_path / "native.exe"
    binary.write_bytes(b"identity fixture, never executed")
    monkeypatch.setattr(repair, "BINARY_SHA256", repair.sha256(binary))
    monkeypatch.setattr(
        repair, "binary_cube", lambda path: np.zeros((2, 2, 2), np.uint8)
    )
    return source, tmp_path / "output", binary


def test_dense_plan_is_checked_without_creating_output(grid):
    plan = repair.plan_repair(*grid, voxel_size_um=8.64)
    assert plan["shape_zyx"] == [384, 384, 384]
    assert len(plan["cube_ids"]) == 27
    assert plan["parameters"]["min_support"] == 0
    assert plan["parameters"]["radial_dr"] == 3
    assert not grid[1].exists()


def test_sparse_grid_is_not_filled_with_fabricated_neighbors(grid):
    next((grid[0] / "cubes_PRED").glob("*.tif")).unlink()
    with pytest.raises(ValueError, match="contiguous"):
        repair.plan_repair(*grid, voxel_size_um=8.64)


@pytest.mark.parametrize("pitch", [2.399, float("nan"), float("inf"), -1])
def test_wrong_grid_units_fail_closed(grid, pitch):
    with pytest.raises(ValueError, match="voxel spacing"):
        repair.plan_repair(*grid, voxel_size_um=pitch)


def test_wrong_model_or_threshold_is_not_silently_retargeted(grid):
    path = grid[0] / "manifest.json"
    value = json.loads(path.read_text())
    value["threshold"] = 0.25
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="F0/200k"):
        repair.plan_repair(*grid, voxel_size_um=8.64)


def test_changed_binary_is_rejected(grid):
    grid[2].write_bytes(b"different implementation")
    with pytest.raises(ValueError, match="binary SHA"):
        repair.plan_repair(*grid, voxel_size_um=8.64)


def test_umbilicus_is_mandatory(grid):
    path = grid[0] / "manifest.json"
    value = json.loads(path.read_text())
    del value["umbilicus_yx"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="umbilicus"):
        repair.plan_repair(*grid, voxel_size_um=8.64)


def test_existing_or_nested_output_is_not_overwritten(grid):
    grid[1].mkdir()
    with pytest.raises(ValueError, match="new and separate"):
        repair.plan_repair(*grid, voxel_size_um=8.64)
    with pytest.raises(ValueError, match="new and separate"):
        repair.plan_repair(grid[0], grid[0] / "nested", grid[2], voxel_size_um=8.64)


def test_partial_job_needs_inspection_not_timeout_recovery(grid):
    grid[1].with_name("output.partial-12345").mkdir()
    with pytest.raises(ValueError, match="partial repair"):
        repair.plan_repair(*grid, voxel_size_um=8.64)


def test_mismatched_declared_cube_inventory_is_rejected(grid):
    path = grid[0] / "manifest.json"
    value = json.loads(path.read_text())
    value["target_cube_ids"] = ["z00000_y00000_x00000"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="target inventory"):
        repair.plan_repair(*grid, voxel_size_um=8.64)


def test_working_set_is_bounded_before_loading_native(grid):
    with pytest.raises(ValueError, match="working set"):
        repair.plan_repair(*grid, voxel_size_um=8.64, max_work_gib=0.01)


def test_command_keeps_tracking_and_never_cuts(grid):
    plan = repair.plan_repair(*grid, voxel_size_um=8.64)
    command = repair.native_command(plan, grid[1])
    assert all("\\" not in argument for argument in command[:3])
    assert "--no-tracks" not in command and "--cut-bridges" not in command
    assert command[command.index("--reach-max") + 1] == "21"
    assert command[command.index("--radial-dr") + 1] == "3"
    assert command[command.index("--min-support") + 1] == "0"


def test_voxel_verification_rejects_erasure_despite_net_growth():
    before = np.zeros((2, 2, 2), np.uint8)
    before[0, 0, 0] = 255
    after = np.full_like(before, 255)
    after[0, 0, 0] = 0
    with pytest.raises(ValueError, match="erased"):
        repair.verify_additive(before, after)


def test_voxel_verification_counts_real_additions():
    before = np.zeros((2, 2, 2), np.uint8)
    after = before.copy()
    after[1, 1, 1] = 255
    assert repair.verify_additive(before, after) == {
        "foreground_before": 0,
        "added_voxels": 1,
        "erased_voxels": 0,
    }


def test_input_mutation_after_preflight_is_rejected(grid):
    plan = repair.plan_repair(*grid, voxel_size_um=8.64)
    Path(plan["input_files"][0]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        repair.check_identity(plan["input_files"][0])


def test_binary_cube_enforces_shape_dtype_and_encoding(tmp_path, monkeypatch):
    for value in [
        np.zeros((2, 2, 2), np.uint8),
        np.zeros((128, 128, 128), np.float32),
        np.ones((128, 128, 128), np.uint8),
    ]:
        monkeypatch.setattr(repair.tifffile, "imread", lambda path, value=value: value)
        with pytest.raises(ValueError):
            repair.binary_cube(tmp_path / "not_read.tif")


def test_plan_does_not_expose_the_global_profile_to_mutation(grid):
    plan = repair.plan_repair(*grid, voxel_size_um=8.64)
    plan["parameters"]["min_support"] = 99
    assert repair.PARAMETERS["min_support"] == 0


def test_existing_lock_never_starts_a_second_native_worker(grid, monkeypatch):
    plan = repair.plan_repair(*grid, voxel_size_um=8.64)
    lock = grid[1].with_name("output.repair.lock")
    lock.write_text("unknown owner")
    monkeypatch.setattr(
        repair.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch")
    )
    with pytest.raises(FileExistsError):
        repair.run_repair(plan)
    assert lock.read_text() == "unknown owner"
    assert not list(grid[1].parent.glob("output.partial-*"))


def test_partial_created_after_preflight_is_preserved_without_launch(grid, monkeypatch):
    plan = repair.plan_repair(*grid, voxel_size_um=8.64)
    partial = grid[1].with_name("output.partial-42")
    partial.mkdir()
    monkeypatch.setattr(
        repair.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch")
    )
    with pytest.raises(ValueError, match="partial repair"):
        repair.run_repair(plan)
    assert partial.is_dir() and not grid[1].exists()
    assert not grid[1].with_name("output.repair.lock").exists()


class FakeNative:
    pid = 12345

    def wait(self, timeout=None):
        return 0


def fake_native_result(monkeypatch, *, cut=0, painted=0, foreground=0):
    def launch(command, **kwargs):
        destination = Path(command[2])
        (destination / "cubes_PRED").mkdir()
        for path in (Path(command[1]) / "cubes_PRED").glob("*.tif"):
            (destination / "cubes_PRED" / path.name).write_bytes(
                b"verified output fixture"
            )
        (destination / "fixup_report.json").write_text(
            json.dumps(
                {
                    "params": dict(repair.PARAMETERS),
                    "bridges": {"cut": cut},
                    "painted_px": painted,
                    "fg_in": foreground,
                }
            )
        )
        return FakeNative()

    monkeypatch.setattr(repair.subprocess, "Popen", launch)


def test_success_publishes_a_verified_receipt(grid, monkeypatch):
    plan = repair.plan_repair(*grid, voxel_size_um=8.64)
    fake_native_result(monkeypatch)
    assert repair.run_repair(plan) == grid[1]
    receipt = json.loads((grid[1] / "repair_receipt.json").read_text())
    assert receipt["status"] == "complete" and receipt["erased_voxels"] == 0
    assert len(receipt["output_files"]) == 27
    assert not grid[1].with_name("output.repair.lock").exists()
    assert not list(grid[1].parent.glob("output.partial-*"))


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"cut": 1}, "additive profile"),
        ({"painted": 1}, "counters"),
        ({"foreground": 1}, "counters"),
    ],
)
def test_native_profile_or_counter_mismatch_is_not_published(
    grid, monkeypatch, overrides, message
):
    plan = repair.plan_repair(*grid, voxel_size_um=8.64)
    fake_native_result(monkeypatch, **overrides)
    with pytest.raises(ValueError, match=message):
        repair.run_repair(plan)
    assert not grid[1].exists()
    assert len(list(grid[1].parent.glob("output.partial-*"))) == 1
    assert not grid[1].with_name("output.repair.lock").exists()


def test_native_failure_preserves_diagnostics_and_does_not_publish(grid, monkeypatch):
    class Failure(FakeNative):
        def wait(self, timeout=None):
            return 9

    plan = repair.plan_repair(*grid, voxel_size_um=8.64)
    monkeypatch.setattr(repair.subprocess, "Popen", lambda *a, **k: Failure())
    with pytest.raises(RuntimeError, match="native repair failed"):
        repair.run_repair(plan)
    assert not grid[1].exists()
    assert len(list(grid[1].parent.glob("output.partial-*"))) == 1
    assert not grid[1].with_name("output.repair.lock").exists()


def test_cancellation_stops_only_the_owned_worker_and_preserves_partial(
    grid, monkeypatch
):
    class Interrupted(FakeNative):
        terminated = False

        def wait(self, timeout=None):
            if not self.terminated:
                raise KeyboardInterrupt
            return -1

        def poll(self):
            return None if not self.terminated else -1

        def terminate(self):
            self.terminated = True

    process = Interrupted()
    plan = repair.plan_repair(*grid, voxel_size_um=8.64)
    monkeypatch.setattr(repair.subprocess, "Popen", lambda *a, **k: process)
    with pytest.raises(KeyboardInterrupt):
        repair.run_repair(plan)
    assert process.terminated
    assert not grid[1].exists()
    assert len(list(grid[1].parent.glob("output.partial-*"))) == 1
    assert not grid[1].with_name("output.repair.lock").exists()
