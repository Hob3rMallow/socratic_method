from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from socratic_method import continuity as c
from socratic_method import repair
from test_repair import grid  # noqa: F401 - shared pytest fixture


@pytest.fixture
def inputs(grid, monkeypatch):  # noqa: F811 - pytest fixture injection
    source, output, binary = grid
    metadata = repair.read_object(source / "manifest.json")
    metadata.update(halo=32, mirror_tta=True, amp_dtype="bfloat16")
    (source / "provenance.json").write_text(json.dumps(metadata))
    baseline = output.with_name("baseline")
    (source / "probability").mkdir()
    plan = repair.plan_repair(source, baseline, binary, voxel_size_um=8.64)
    (baseline / "cubes_PRED").mkdir(parents=True)
    files = []
    for key in plan["cube_ids"]:
        path = baseline / "cubes_PRED" / f"{key}.tif"
        path.write_bytes(b"baseline fixture")
        files.append({**repair.file_identity(path), "path": f"cubes_PRED/{key}.tif"})
        (source / "probability" / f"{key}.tif").write_bytes(b"probability fixture")
    receipt = {
        "status": "complete",
        "profile": repair.PROFILE,
        "erased_voxels": 0,
        "added_voxels": 0,
        "plan": plan,
        "output_files": files,
        "native": {"params": repair.PARAMETERS, "bridges": {"cut": 0}},
    }
    c.json_file(baseline / "repair_receipt.json", receipt)
    monkeypatch.setattr(
        c, "probability_cube", lambda path: np.full((2, 2, 2), 0.1, np.float16)
    )
    return source, baseline, output


def make_plan(inputs):
    return c.plan_continuity(*inputs, voxel_size_um=8.64)


def test_check_only_validates_without_writing(inputs):
    plan = make_plan(inputs)
    assert not inputs[2].exists()
    assert plan["probability_dtype"] == "float16"
    assert plan["shape_zyx"] == [384, 384, 384]
    assert plan["processes"] == 1 and plan["child_processes"] == 0
    assert plan["proposal"]["maximum_length"] == 96


def test_baseline_receipt_with_cutting_is_refused(inputs):
    path = inputs[1] / "repair_receipt.json"
    receipt = repair.read_object(path)
    receipt["native"]["bridges"]["cut"] = 1
    c.json_file(path, receipt)
    with pytest.raises(ValueError, match="completed checked"):
        make_plan(inputs)


def test_baseline_hash_mutation_is_detected(inputs):
    next((inputs[1] / "cubes_PRED").glob("*.tif")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        make_plan(inputs)


def test_probability_inventory_cannot_have_missing_context(inputs):
    next((inputs[0] / "probability").glob("*.tif")).unlink()
    with pytest.raises(ValueError, match="probability cube inventory"):
        make_plan(inputs)


def test_different_inference_profile_is_refused(inputs):
    path = inputs[0] / "provenance.json"
    metadata = repair.read_object(path)
    metadata["mirror_tta"] = False
    c.json_file(path, metadata)
    with pytest.raises(ValueError, match="mirror-TTA"):
        make_plan(inputs)


def test_non_additive_baseline_is_not_trusted(inputs, monkeypatch):
    monkeypatch.setattr(
        repair,
        "binary_cube",
        lambda p: np.full((2, 2, 2), 0 if inputs[1] in p.parents else 255, np.uint8),
    )
    with pytest.raises(ValueError, match="erased"):
        make_plan(inputs)


def test_wrong_probability_field_is_refused(inputs, monkeypatch):
    monkeypatch.setattr(c, "probability_cube", lambda p: np.ones((2, 2, 2), np.float16))
    with pytest.raises(ValueError, match="disagrees"):
        make_plan(inputs)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1, 1.1])
def test_probability_value_validation(monkeypatch, bad):
    value = np.zeros((128, 128, 128), np.float32)
    value[0, 0, 0] = bad
    monkeypatch.setattr(c.tifffile, "imread", lambda p: value)
    with pytest.raises(ValueError, match="invalid probabilities"):
        c.probability_cube(Path("not_read"))


@pytest.mark.parametrize(
    "value", [np.zeros((2, 2, 2), np.float16), np.zeros((128, 128, 128), np.uint8)]
)
def test_probability_shape_dtype_validation(monkeypatch, value):
    monkeypatch.setattr(c.tifffile, "imread", lambda p: value)
    with pytest.raises(ValueError, match="expected float"):
        c.probability_cube(Path("not_read"))


def test_nested_baseline_output_refused(inputs):
    with pytest.raises(ValueError, match="separate"):
        c.plan_continuity(
            inputs[0], inputs[1], inputs[1] / "nested", voxel_size_um=8.64
        )


def test_memory_preflight_refuses_oversize(inputs):
    with pytest.raises(ValueError, match="working set"):
        c.plan_continuity(*inputs, voxel_size_um=8.64, max_work_gib=0.01)


def test_mutated_plan_policy_refused(inputs):
    plan = make_plan(inputs)
    plan["proposal"]["probability_floor"] = 0.1
    with pytest.raises(ValueError, match="plan policy"):
        c.run_continuity(plan)


def test_mutated_input_after_preflight_refused(inputs):
    plan = make_plan(inputs)
    next((inputs[0] / "probability").glob("*.tif")).write_bytes(b"different")
    with pytest.raises(ValueError, match="input changed"):
        c.run_continuity(plan)
    assert not inputs[2].exists()


def test_unknown_lock_is_preserved(inputs):
    plan = make_plan(inputs)
    lock = inputs[2].with_name(inputs[2].name + ".continuity.lock")
    lock.write_text("unknown owner")
    with pytest.raises(FileExistsError, match="lock"):
        c.run_continuity(plan)
    assert lock.read_text() == "unknown owner"


def test_partial_job_is_not_reclaimed(inputs):
    plan = make_plan(inputs)
    partial = inputs[2].with_name(inputs[2].name + ".partial-42")
    partial.mkdir()
    with pytest.raises(ValueError, match="partial"):
        c.run_continuity(plan)
    assert partial.is_dir()


def test_independent_contact_and_endpoint_checks():
    before = np.zeros((32, 32), bool)
    before[16, 2:9] = before[16, 20:29] = True
    after = before.copy()
    after[16, 9:20] = True
    ledger = [{"accepted": True, "path_yx": [[16, 8], [16, 20]], "added_voxels": 11}]
    assert c.audit_plane(before, after, ledger) == {
        "added_voxels": 11,
        "two_contact_strokes": 1,
        "anchor_apron_voxels": 0,
    }
    after[3, 3] = True
    with pytest.raises(ValueError, match="exactly two"):
        c.audit_plane(before, after, ledger)


def test_independent_audit_catches_erasure_even_with_net_growth():
    before = np.zeros((8, 8), bool)
    before[0, 0] = True
    after = ~before
    with pytest.raises(ValueError, match="erased"):
        c.audit_plane(before, after, [])


def test_anchor_apron_is_distinguished_from_detached_or_arbitrary_growth():
    before = np.zeros((32, 32), bool)
    before[16, 2:9] = before[16, 20:29] = True
    before[15, 8] = True
    after = before.copy()
    after[16, 9:20] = True
    after[14, 8] = True
    ledger = [
        {"accepted": True, "path_yx": [[15, 8], [16, 8], [16, 20]], "added_voxels": 12}
    ]
    result = c.audit_plane(before, after, ledger)
    assert result["anchor_apron_voxels"] == 1 and result["two_contact_strokes"] == 1
    after[14, 8] = False
    after[15, 3] = True
    with pytest.raises(ValueError, match="verified anchor-apron"):
        c.audit_plane(before, after, ledger)


def runtime_fixture(inputs, monkeypatch):
    plan = make_plan(inputs)
    # Bounded transaction test; full real 27-cube parity is a separate integration.
    plan["shape_zyx"] = [128, 128, 128]
    plan["cube_ids"] = plan["cube_ids"][:1]
    monkeypatch.setattr(
        repair, "binary_cube", lambda p: np.zeros((128, 128, 128), np.uint8)
    )
    monkeypatch.setattr(
        c, "probability_cube", lambda p: np.full((128, 128, 128), 0.1, np.float16)
    )
    return plan


def test_success_publishes_receipt_and_preserves_dtype(inputs, monkeypatch):
    plan = runtime_fixture(inputs, monkeypatch)

    def find(mask, p, *args):
        assert p.dtype == np.float16
        return [], {}

    monkeypatch.setattr(c.engine, "find_corridors", find)
    output = c.run_continuity(plan)
    receipt = repair.read_object(output / "continuity_receipt.json")
    assert receipt["status"] == "complete" and receipt["added_voxels"] == 0
    assert not list(output.parent.glob(output.name + ".partial-*"))
    assert not output.with_name(output.name + ".continuity.lock").exists()
    assert repair.read_object(output / "progress.json")["status"] == "complete"


@pytest.mark.parametrize(
    "error", [RuntimeError("fixture failure"), KeyboardInterrupt()]
)
def test_failure_or_cancellation_preserves_partial_without_publishing(
    inputs, monkeypatch, error
):
    plan = runtime_fixture(inputs, monkeypatch)

    def fail(*args):
        raise error

    monkeypatch.setattr(c.engine, "find_corridors", fail)
    with pytest.raises(type(error)):
        c.run_continuity(plan)
    assert not inputs[2].exists()
    partials = list(inputs[2].parent.glob(inputs[2].name + ".partial-*"))
    assert len(partials) == 1 and (partials[0] / "failure.json").is_file()
    assert not inputs[2].with_name(inputs[2].name + ".continuity.lock").exists()
