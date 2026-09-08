from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from socratic_method import sheet_patches as sheets


@pytest.mark.parametrize("spacing", [0, 2.399, 9.362, float("nan")])
def test_refuse_unqualified_pitch_before_reading_inputs(spacing):
    with pytest.raises(ValueError, match="8.640"):
        sheets.plan_patches(
            Path("missing"), Path("raw"), Path("out"), voxel_size_um=spacing
        )


@pytest.mark.parametrize("limit", [0, -1, float("inf"), float("nan"), 3])
def test_memory_limit_must_be_finite_and_sufficient(limit):
    with pytest.raises(ValueError, match="max-work"):
        sheets.plan_patches(
            Path("missing"),
            Path("raw"),
            Path("out"),
            voxel_size_um=8.64,
            max_work_gib=limit,
        )


def test_no_output_inside_or_above_input(tmp_path):
    source = tmp_path / "input"
    with pytest.raises(ValueError, match="separate"):
        sheets.check_output([source], source / "nested")
    with pytest.raises(ValueError, match="separate"):
        sheets.check_output([source], tmp_path)


@pytest.mark.parametrize("kind", ["existing", "partial", "lock"])
def test_never_overwrite_or_reclaim_work(tmp_path, kind):
    output = tmp_path / "out"
    if kind == "existing":
        output.mkdir()
    elif kind == "partial":
        (tmp_path / "out.partial-999").mkdir()
    else:
        (tmp_path / "out.sheets.lock").write_text("unknown owner")
    with pytest.raises(FileExistsError):
        sheets.check_output([tmp_path / "input"], output)


@pytest.mark.parametrize(
    "path,z", [([[1.2, 2]], 2), ([[1, 999]], 2), ([[1, 2]], -1), ([[1, 2]], True)]
)
def test_ledger_rejects_invalid_seed_coordinates(tmp_path, path, z):
    f = tmp_path / "ledger.jsonl"
    f.write_text(
        json.dumps(
            {"accepted": True, "path_yx": path, "z_local": z, "z_world": 100 + z}
        )
    )
    with pytest.raises(ValueError, match="coordinates"):
        sheets.validated_rows(f, (128, 128, 128), (100, 0, 0))


def test_default_cli_only_calls_preflight(tmp_path, capsys):
    with (
        patch.object(
            sheets, "plan_patches", return_value={"status": "checked"}
        ) as plan,
        patch.object(sheets, "run_patches") as execute,
        patch(
            "sys.argv",
            [
                "socratic-sheet-patches",
                "prior",
                "raw",
                "out",
                "--voxel-size-um",
                "8.64",
            ],
        ),
    ):
        sheets.main()
    plan.assert_called_once()
    execute.assert_not_called()
    assert "checked" in capsys.readouterr().out


def test_component_labels_derived_from_paint_not_trusted_seed_ids():
    base = np.zeros((8, 8, 8), bool)
    native = base.copy()
    base[3, 3, 2:6] = True
    rows = [{"path_yx": [[3, 2], [3, 3], [3, 4]], "z_local": 3, "added_component": 999}]
    assert sheets.assign_components(rows, base, native)[0]["added_component"] == 1


def test_geometry_rejection_is_logged_without_erasing_base():
    p = np.ones((8, 8, 8), np.float32)
    base = np.zeros_like(p, dtype=bool)
    base[2, 2, 2] = True
    before = base.copy()
    log = []
    rows = [
        {
            "added_component": 1,
            "id": 0,
            "length_px": 3,
            "probability_mean": 0.3,
            "z_world": 3,
            "z_local": 3,
            "path_yx": [[3, 2], [3, 3]],
        }
    ]
    added, accepted, census, count = sheets.fit_arrays(
        p, p, base, rows, lambda r, m: log.append(r), lambda *a: None
    )
    assert np.array_equal(base, before) and not added.any() and not accepted
    assert log[0]["accepted"] is False and count == 1
    assert census["seed too short for this patch profile"] == 1


def test_changed_plan_rejected_before_output_creation(tmp_path):
    plan = {
        "profile": sheets.PROFILE,
        "patch_policy": sheets.asdict(sheets.PATCH),
        "growth_policy": sheets.asdict(sheets.GROWTH),
        "source_continuity": str(tmp_path / "prior"),
        "raw_grid": str(tmp_path / "raw"),
        "output": str(tmp_path / "out"),
        "source": str(tmp_path / "source"),
        "native": str(tmp_path / "native"),
        "voxel_size_um": 8.64,
        "max_work_gib": 8,
    }
    with (
        patch.object(
            sheets, "plan_patches", return_value={**plan, "unexpected": "tamper"}
        ),
        pytest.raises(ValueError, match="plan changed"),
    ):
        sheets.run_patches(plan)
    assert not (tmp_path / "out").exists()


def test_execution_failure_preserves_partial_and_refuses_restart(tmp_path):
    plan = {
        "profile": sheets.PROFILE,
        "patch_policy": sheets.asdict(sheets.PATCH),
        "growth_policy": sheets.asdict(sheets.GROWTH),
        "source_continuity": str(tmp_path / "prior"),
        "raw_grid": str(tmp_path / "raw"),
        "output": str(tmp_path / "out"),
        "source": str(tmp_path / "source"),
        "native": str(tmp_path / "native"),
        "voxel_size_um": 8.64,
        "max_work_gib": 8,
        "input_files": [],
        "code": [],
        "spec": {"target_cube_ids": [], "origin_zyx": [0, 0, 0]},
    }
    with (
        patch.object(sheets, "plan_patches", return_value=plan),
        patch.object(sheets, "assemble", side_effect=OSError("simulated read failure")),
        pytest.raises(OSError, match="simulated"),
    ):
        sheets.run_patches(plan)
    partials = list(tmp_path.glob("out.partial-*"))
    assert len(partials) == 1
    assert "simulated" in json.loads((partials[0] / "failure.json").read_text())["error"]
    assert json.loads((partials[0] / "state.json").read_text())["status"] == "failed_or_interrupted"
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "out.sheets.lock").exists()
    with pytest.raises(FileExistsError, match="partial"):
        sheets.check_output([tmp_path / "prior"], tmp_path / "out")
