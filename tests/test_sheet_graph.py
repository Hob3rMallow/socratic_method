from __future__ import annotations

import itertools

import numpy as np
import pytest

from socratic_method.sheet_graph import SurfacePolicy, energy, solve_surface


@pytest.mark.parametrize("seed", range(15))
def test_matches_exhaustive_constrained_surface_oracle(seed):
    rng = np.random.default_rng(seed)
    costs = rng.integers(0, 20, size=(2, 2, 3)).astype(float) / 10
    fixed = np.full((2, 2), -1, int)
    fixed[0, 0] = seed % 3
    policy = SurfacePolicy(maximum_z_step=1, maximum_s_step=1, z_tv=0.2, s_tv=0.1)
    candidates = []
    for raw in itertools.product(range(3), repeat=4):
        labels = np.array(raw).reshape(2, 2)
        if (
            labels[0, 0] != fixed[0, 0]
            or (np.abs(np.diff(labels, axis=0)) > 1).any()
            or (np.abs(np.diff(labels, axis=1)) > 1).any()
        ):
            continue
        candidates.append(energy(labels, costs, policy))
    result = solve_surface(costs, fixed=fixed, policy=policy)
    assert result["energy"] == pytest.approx(min(candidates), abs=1e-9)


def test_joint_surface_recovers_weak_row_from_both_neighbors():
    costs = np.ones((9, 16, 7)) * 3
    costs[:, :, 3] = 0
    costs[4, :, 3] = 0.5
    costs[4, :, 0] = 0
    fixed = np.full((9, 16), -1, int)
    fixed[0] = 3
    result = solve_surface(costs, fixed=fixed)
    assert (result["labels"] == 3).all()


def test_hard_local_sheet_order_bounds_override_brighter_competing_surface():
    costs = np.ones((7, 12, 9)) * 2
    costs[:, :, 2] = 0.3
    costs[:, :, 7] = 0
    upper = np.full((7, 12), 4, int)
    result = solve_surface(costs, upper=upper)
    assert (result["labels"] == 2).all()


def test_infeasible_anchor_slopes_do_not_silently_relax():
    costs = np.zeros((2, 2, 5))
    anchors = np.array([[0, 0], [4, 4]])
    with pytest.raises(ValueError, match="infeasible"):
        solve_surface(costs, fixed=anchors, policy=SurfacePolicy(maximum_z_step=1))


def test_zero_slope_means_one_shared_height():
    rng = np.random.default_rng(1)
    costs = rng.random((4, 5, 7))
    result = solve_surface(costs, policy=SurfacePolicy(0, 0))
    assert np.unique(result["labels"]).size == 1
    assert result["labels"][0, 0] == costs.sum(axis=(0, 1)).argmin()


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_costs_refused(bad):
    costs = np.zeros((3, 3, 3))
    costs[0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        solve_surface(costs)


def test_quantization_is_explicit_and_energy_is_independently_verified():
    costs = np.random.default_rng(2).random((3, 3, 5))
    result = solve_surface(costs)
    assert result["energy"] == pytest.approx(energy(result["labels"], costs))
    assert isinstance(result["quantized_energy"], int)


def test_surface_seed_is_preserved_when_neighbors_prefer_other_sheet():
    costs = np.ones((9, 11, 9)) * 3
    costs[:, :, 6] = 0
    fixed = np.full((9, 11), -1, int)
    fixed[4] = 2
    result = solve_surface(costs, fixed=fixed)
    assert (result["labels"][4] == 2).all()
    assert result["maximum_z_step"] <= 2
