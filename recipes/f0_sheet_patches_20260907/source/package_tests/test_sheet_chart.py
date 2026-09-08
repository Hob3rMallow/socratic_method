from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from socratic_method import sheet_chart as patch
from socratic_method.sheet_bundle import fit_bundle


def volume(bend=True, missing_axial=False):
    z, y, x = np.indices((49, 128, 128))
    base = 64 + (3 * np.sin((x - 64) / 32) if bend else 0)
    sheet = base + 0.25 * (z - 24)
    gap = (x >= 43) & (x <= 85)
    amplitude = np.where(gap, 0.30, 0.70)
    if missing_axial:
        amplitude = np.where(gap & (np.abs(z - 24) > 2), 0.02, amplitude)
    p = (0.01 + amplitude * np.exp(-0.5 * ((y - sheet) / 1.1) ** 2)).astype(np.float32)
    ct = (20 + 190 * np.exp(-0.5 * ((y - sheet + 0.8) / 1.3) ** 2)).astype(np.float32)
    xs = np.arange(38, 92)
    ys = 64 + (3 * np.sin((xs - 64) / 32) if bend else np.zeros_like(xs))
    path = np.column_stack([ys, xs])
    return p, ct, path


@pytest.mark.parametrize("bend", [False, True])
def test_joint_fit_extends_seed_into_moving_ground_truth_sheet(bend):
    p, ct, path = volume(bend)
    fitted = patch.fit_patch(p, ct, path, 24)
    points = fitted["points"]
    truth = (
        64
        + (3 * np.sin((points[..., 2] - 64) / 32) if bend else 0)
        + 0.25 * (points[..., 0] - 24)
    )
    assert np.abs(points[..., 1] - truth).mean() < 0.65
    assert np.quantile(np.abs(points[..., 1] - truth), 0.99) < 1.1
    assert fitted["supported_z_span"] >= 25
    assert fitted["geometry"]["chart_foldover_steps"] == 0
    mask = patch.voxelize_diagnostic(fitted["supported_points"], p.shape, p)
    assert ndimage.label(mask, np.ones((3, 3, 3), bool))[1] == 1
    assert len(np.unique(np.argwhere(mask)[:, 0])) >= 25


def test_missing_probability_evidence_stops_patch_instead_of_filling_whole_chart():
    p, ct, path = volume(missing_axial=True)
    fitted = patch.fit_patch(p, ct, path, 24)
    assert fitted["supported_z_span"] <= 5


def test_local_chart_cannot_sort_a_real_fold_back_into_another_surface():
    path = np.array(
        [[60, x] for x in range(20, 50)] + [[60 + i, 50 - i] for i in range(1, 22)]
    )
    with pytest.raises(ValueError, match="monotone"):
        patch.curve_frame(path)


def test_parallel_chart_has_no_caustic_foldover_on_curved_shifted_sheet():
    p, ct, path = volume()
    fitted = patch.fit_patch(p, ct, path, 24)
    assert fitted["geometry"]["minimum_chart_forward_step"] > 0.9


def test_voxelization_refuses_out_of_volume_geometry():
    points = np.zeros((5, 5, 3))
    points[..., 0] = -5
    with pytest.raises(ValueError, match="outside"):
        patch.voxelize_diagnostic(points, (8, 8, 8), np.ones((8, 8, 8)))


def test_ordered_neighbor_prevents_switch_to_brighter_parallel_sheet():
    z, y, x = np.indices((49, 128, 128))
    target = 64 + 0.15 * (z - 24)
    neighbor = target + 6
    amplitude = np.where(np.abs(z - 24) < 3, 0.3, 0.16)
    p = (
        0.01
        + amplitude * np.exp(-0.5 * ((y - target) / 1.1) ** 2)
        + 0.75 * np.exp(-0.5 * ((y - neighbor) / 1.1) ** 2)
    ).astype(np.float32)
    # Identical repeated tissue appearance is deliberately not an identity cue.
    ct = (120 + 95 * np.cos(2 * np.pi * (y - target) / 6)).astype(np.float32)
    path = np.column_stack((np.full(54, 64), np.arange(38, 92)))
    policy = patch.PatchPolicy(
        use_tensor_motion=False, ct_profile_weight=0, displacement_weight=0
    )
    plain = patch.fit_patch(p, ct, path, 24, policy)
    ordered = fit_bundle(p, ct, path, 24, policy=policy, minimum_separation=5)
    points = ordered["points"]
    expected = 64 + 0.15 * (points[..., 0] - 24)
    assert np.abs(points[..., 1] - expected).mean() < 0.7
    plain_error = np.abs(
        plain["points"][..., 1] - (64 + 0.15 * (plain["points"][..., 0] - 24))
    ).mean()
    assert plain_error > 3
    assert ordered["minimum_normal_separation_px"] >= 5
