from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_f0_repair_contacts import plane_contacts


def test_two_ended_gap_is_distinguished_from_detached_paint():
    raw = np.zeros((20, 20), bool)
    raw[10, 1:7] = True
    raw[10, 13:19] = True
    fixed = raw.copy()
    fixed[10, 7:13] = True
    good = plane_contacts(raw, fixed)
    assert good["multiway_strokes"] == good["detached_strokes"] == 0
    assert len(good["strokes"][0]["touched_raw_components"]) == 2
    fixed[1, 1] = True
    assert plane_contacts(raw, fixed)["detached_strokes"] == 1


def test_a_three_way_join_is_flagged_not_misreported_as_simple_gap():
    raw = np.zeros((20, 20), bool)
    raw[10, 1:7] = True
    raw[10, 13:19] = True
    raw[1:7, 10] = True
    fixed = raw.copy()
    fixed[10, 7:13] = True
    fixed[7:10, 10] = True
    assert plane_contacts(raw, fixed)["multiway_strokes"] == 1
