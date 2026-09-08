from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_f0_repair_context import additive_counts, command_for
from run_f0_repair_context import PRESETS


def test_added_geometry_is_counted_exactly():
    before = np.zeros((3, 3, 3), np.uint8)
    before[1, 1, 0] = 255
    after = before.copy()
    after[1, 1, 1] = 255
    assert additive_counts(before, after) == {"foreground_before": 1, "added_voxels": 1, "erased_voxels": 0}


def test_noop_is_valid_but_is_not_counted_as_improvement():
    a = np.zeros((3, 3, 3), np.uint8)
    assert additive_counts(a, a)["added_voxels"] == 0


def test_erasure_is_rejected_even_if_net_foreground_increases():
    before = np.zeros((3, 3, 3), np.uint8)
    before[0, 0, 0] = 255
    after = np.full_like(before, 255)
    after[0, 0, 0] = 0
    with pytest.raises(ValueError, match="erased"):
        additive_counts(before, after)


def test_geometry_and_nonbinary_outputs_fail():
    with pytest.raises(ValueError, match="geometry"):
        additive_counts(np.zeros((2, 2, 2)), np.zeros((2, 2, 3)))
    with pytest.raises(ValueError, match="binary"):
        additive_counts(np.zeros((2, 2, 2)), np.ones((2, 2, 2)))


@pytest.mark.parametrize("preset", PRESETS.values())
def test_commands_keep_safety_flags_and_never_enable_cutting(preset):
    command = command_for(Path("source"), Path("target"), preset)
    assert "--cut-bridges" not in command and "--no-tracks" not in command
    assert command[command.index("--paint-radius") + 1] == "1"
    assert command[command.index("--min-support") + 1] == str(preset["min_support"])
    assert command[command.index("--radial-dr") + 1] == str(preset["radial_dr"])


def test_extra_preset_field_cannot_inject_native_options():
    with pytest.raises(ValueError):
        command_for(Path("source"), Path("target"), {**PRESETS["radial_safe"], "cut_bridges": True})
