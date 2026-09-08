from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_f0_repair_context as repair


def test_blocks_are_dense_and_core_has_real_neighbors():
    for center in repair.BLOCKS.values():
        points = repair.block_origins(center)
        assert len(points) == len(set(points)) == 27
        assert center in points
        assert all(len({p[axis] for p in points}) == 3 for axis in range(3))
        assert all(max(p[axis] for p in points) - min(p[axis] for p in points) == 256 for axis in range(3))


@pytest.mark.parametrize("center", [(1, 128, 128), (-128, 128, 128), (0, 128, 128)])
def test_invalid_or_negative_context_fails_closed(center):
    with pytest.raises(ValueError):
        repair.block_origins(center)


def test_presets_isolate_support_then_strengthen_radial_guard():
    baseline = repair.PRESETS["report_baseline"]
    support = repair.PRESETS["support_only"]
    safer = repair.PRESETS["radial_safe"]
    extended = repair.PRESETS["extended_safe"]
    assert {k for k in baseline if baseline[k] != support[k]} == {"min_support"}
    assert safer == {**support, "radial_dr": 3}
    assert extended == {**safer, "reach_max": 21}


def test_cube_coordinates_keep_world_axes():
    assert repair.cube_id((12160, 3968, 3072)) == "z12160_y03968_x03072"


def test_frozen_output_is_not_a_repair_destination():
    assert repair.OUTPUT != repair.RELEASE
    assert repair.RELEASE not in repair.OUTPUT.parents
    assert repair.OUTPUT not in repair.RELEASE.parents
