from __future__ import annotations


import numpy as np
import pytest
from scipy import ndimage

from socratic_method import continuity_paths as continuity
from test_continuity_core import long_gap


def supported_rows(mask, p):
    rows, _ = continuity.find_corridors(mask, p, (0, 0))
    for row in rows:
        row["neighbor_probability_coverage"] = [1, 1, 1, 1]
    return rows


@pytest.mark.parametrize("bend", [False, True])
def test_refinement_keeps_long_and_curved_real_continuations(bend):
    mask, p = long_gap(bend)
    rows = supported_rows(mask, p)
    after, ledger, _ = continuity.select_and_paint(mask, p, rows, 0.2)
    assert sum(r["accepted"] for r in ledger) == 1
    assert max(r["length_px"] for r in ledger) > 21
    assert ndimage.label(after)[1] == 1
    assert not (mask & ~after).any()


def test_wrong_side_attachment_is_explicitly_rejected():
    mask, p = long_gap()
    row = supported_rows(mask, p)[0]
    row["endpoint_outward"] = [1, -0.9]
    assert continuity.refusal_reason(row) == "backwards_anchor_approach"


def test_abrupt_zigzag_is_not_treated_as_a_smooth_sheet_curve():
    path = np.array(
        [[64, x] for x in range(30, 45)] + [[64 - y, 44] for y in range(1, 12)]
    )
    assert continuity.local_turn_degrees(path) > 60
    mask, p = long_gap()
    row = supported_rows(mask, p)[0]
    row["maximum_local_turn_degrees"] = 90
    assert continuity.refusal_reason(row) == "abrupt_local_turn"


def test_refinement_does_not_cross_a_third_raw_component():
    mask, p = long_gap()
    rows = supported_rows(mask, p)
    mask[60:69, 64] = True
    path = continuity.refine_route(np.array(rows[0]["path_yx"]), mask, p, 0.2)
    assert path is None


def test_refinement_prefers_probability_peak_over_offset_skeleton():
    mask, p = long_gap()
    rows = supported_rows(mask, p)
    path = np.array(rows[0]["path_yx"])
    rough = path.copy()
    rough[3:-3, 0] += 1
    refined = continuity.refine_route(rough, mask, p, 0.2)
    assert refined is not None
    assert p[tuple(refined.T)].mean() >= p[tuple(rough.T)].mean()
