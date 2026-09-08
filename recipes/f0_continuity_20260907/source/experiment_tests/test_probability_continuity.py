from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probability_continuity as continuity


def long_gap(bend=False):
    y, x = np.indices((128, 128))
    center = 64 + (8 * np.sin((x - 64) / 32) if bend else 0)
    amplitude = np.where((x >= 47) & (x <= 85), .28, .65)
    p = (.01 + amplitude * np.exp(-.5 * ((y-center)/1.2)**2)).astype(np.float32)
    p[:, :16] = p[:, 113:] = .01
    return p >= .35, p


def propose_and_support(mask, p):
    rows, _ = continuity.find_corridors(mask, p, (0, 0), continuity.ProposalPolicy(probability_floor=.2))
    for row in rows:
        row["neighbor_probability_coverage"] = continuity.probability_support(np.array(row["path_yx"]), [p, p, p, p], .2)
    return rows


@pytest.mark.parametrize("bend", [False, True])
def test_real_long_low_probability_continuation_is_recovered(bend):
    mask, p = long_gap(bend)
    before = mask.copy()
    rows = propose_and_support(mask, p)
    after, ledger, _ = continuity.select_and_paint(mask, p, rows, .2)
    accepted = [r for r in ledger if r["accepted"]]
    assert len(accepted) == 1 and accepted[0]["length_px"] > 21
    assert np.array_equal(before, mask) and not (mask & ~after).any()
    assert ndimage.label(mask)[1] == 2 and ndimage.label(after)[1] == 1
    assert not (after & (p < .16)).any()


def test_no_probability_corridor_means_no_fabricated_bridge():
    mask, p = long_gap()
    p[:, 48:85] = .01
    assert not propose_and_support(mask, p)


def test_unseeded_weak_sheet_is_not_globally_thresholded_into_output():
    mask, p = long_gap()
    p[100, 20:100] = .28
    rows = propose_and_support(mask, p)
    after, _, _ = continuity.select_and_paint(mask, p, rows, .2)
    assert not after[100].any()


def test_two_parallel_sheets_with_perpendicular_soft_bridge_are_not_joined():
    y, x = np.indices((128, 128))
    p = np.maximum(.7 * np.exp(-.5*((y-50)/1.2)**2), .7 * np.exp(-.5*((y-70)/1.2)**2))
    p[:, :16] = p[:, 113:] = 0
    p[50:71, 61:64] = np.maximum(p[50:71, 61:64], .28)
    mask = p >= .35
    rows = propose_and_support(mask, p.astype(np.float32))
    after, ledger, _ = continuity.select_and_paint(mask, p, rows, .2)
    assert np.array_equal(mask, after)
    assert not any(r["accepted"] for r in ledger)


def test_paint_near_third_fragment_is_rejected():
    mask, p = long_gap()
    rows = propose_and_support(mask, p)
    mask[66, 63] = True
    after, ledger, _ = continuity.select_and_paint(mask, p, rows, .2)
    assert np.array_equal(mask, after)
    assert any(r["reason"] == "third_component_or_missing_anchor" for r in ledger)


def test_duplicate_and_crossing_corridors_cannot_reinforce_each_other():
    mask, p = long_gap()
    rows = propose_and_support(mask, p)
    after, ledger, _ = continuity.select_and_paint(mask, p, rows + rows, .2)
    assert sum(r["accepted"] for r in ledger) == 1
    assert any(r["reason"] == "crossing_or_shared_gap" for r in ledger)
    assert not (mask & ~after).any()


def test_missing_axial_support_is_not_fabricated_from_repair_output():
    mask, p = long_gap()
    rows = propose_and_support(mask, p)
    for row in rows:
        row["neighbor_probability_coverage"] = [0, 0, 0, 0]
    after, ledger, _ = continuity.select_and_paint(mask, p, rows, .2)
    assert np.array_equal(mask, after)
    assert all(r["reason"] == "insufficient_neighbor_probability_support" for r in ledger)


@pytest.mark.parametrize("reason,change", [
    ("endpoint_direction", {"endpoint_alignment": [.1, 1]}),
    ("ambiguous_endpoint_shape", {"endpoint_anisotropy": [1, 10]}),
    ("excessive_detour", {"arc_ratio": 2}),
    ("weak_mean_probability", {"probability_mean": .21}),
    ("off_probability_ridge", {"probability_ridge_positive_fraction": .1}),
])
def test_geometric_and_probability_guard_reasons(reason, change):
    mask, p = long_gap()
    row = propose_and_support(mask, p)[0]
    row.update(change)
    assert continuity.refusal_reason(row) == reason


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, 1.1])
def test_invalid_probabilities_fail_closed(value):
    mask, p = long_gap()
    p[0, 0] = value
    with pytest.raises(ValueError, match="probability"):
        continuity.find_corridors(mask, p, (0, 0))


def test_completed_sheet_is_a_noop():
    mask, p = long_gap()
    mask |= p >= .2
    assert not propose_and_support(mask, p)
