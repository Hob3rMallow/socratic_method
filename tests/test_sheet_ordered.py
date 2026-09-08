from __future__ import annotations

import itertools

import numpy as np
import pytest

from socratic_method.sheet_ordered import SurfacePolicy, energy, solve_surfaces


@pytest.mark.parametrize("seed", range(12))
def test_ordered_solver_matches_exhaustive_oracle(seed):
    costs = np.random.default_rng(seed).integers(0, 20, size=(2, 1, 2, 4)) / 10
    policy = SurfacePolicy(1, 1, 0.2, 0.1, 1000)
    energies = []
    for raw in itertools.product(range(4), repeat=4):
        labels = np.array(raw).reshape(2, 1, 2)
        if (np.diff(labels, axis=0) < 1).any() or (
            np.abs(np.diff(labels, axis=2)) > 1
        ).any():
            continue
        energies.append(energy(labels, costs, policy))
    result = solve_surfaces(costs, minimum_separation=1, policy=policy)
    assert result["energy"] == pytest.approx(min(energies), abs=1e-9)


def test_order_is_preserved_when_independent_preferences_cross():
    costs = np.full((2, 9, 12, 11), 3.0)
    costs[0, :4, :, 3] = 0
    costs[1, :4, :, 7] = 0
    costs[0, 4:, :, 7] = 0
    costs[1, 4:, :, 3] = 0
    anchors = np.full((2, 9, 12), -1, int)
    anchors[0, 0] = 3
    anchors[1, 0] = 7
    result = solve_surfaces(costs, fixed=anchors, minimum_separation=3)
    assert (np.diff(result["labels"], axis=0) >= 3).all()
    assert (result["labels"][0, 0] == 3).all()
    assert (result["labels"][1, 0] == 7).all()


def test_three_surfaces_cannot_collapse_onto_one_bright_probability_ridge():
    costs = np.full((3, 5, 6, 13), 2.0)
    costs[:, :, :, 6] = 0
    costs[0, :, :, 2] = 0.1
    costs[2, :, :, 10] = 0.1
    result = solve_surfaces(costs, minimum_separation=3)
    assert (result["labels"][0] == 2).all()
    assert (result["labels"][1] == 6).all()
    assert (result["labels"][2] == 10).all()


def test_contradictory_seed_order_fails_closed():
    fixed = np.array([[[3, 3]], [[1, 1]]])
    with pytest.raises(ValueError, match="anchor|infeasible"):
        solve_surfaces(np.zeros((2, 1, 2, 6)), fixed=fixed, minimum_separation=2)


def test_reported_axial_step_is_not_inter_surface_separation():
    costs = np.full((2, 5, 6, 13), 2.0)
    costs[0, :, :, 2] = 0
    costs[1, :, :, 10] = 0
    result = solve_surfaces(costs, minimum_separation=3)
    assert result["maximum_z_step"] == 0
    assert result["maximum_s_step"] == 0
