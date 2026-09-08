from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage
from test_sheet_chart import volume

from socratic_method import sheet_growth as growth


def test_surface_mesh_is_connected_manifold_and_has_real_area():
    z, s = np.indices((15, 30))
    points = np.stack([z + 10, 50 + 0.2 * z, s + 20], axis=-1).astype(float)
    mesh = growth.mesh_from_grid(points, np.ones((15, 30), bool), 7)
    assert mesh["manifold_with_boundary"]
    assert len(mesh["faces"]) == 2 * 14 * 29
    assert mesh["area_px2"] == pytest.approx(14 * 29 * np.sqrt(1.04))
    assert mesh["supported_z_span"] == 15


def test_disconnected_surface_faces_are_not_claimed_as_one_patch():
    z, s = np.indices((15, 30))
    points = np.stack([z + 10, np.full_like(z, 50), s + 20], axis=-1).astype(float)
    valid = np.ones((15, 30), bool)
    valid[8:10] = False
    mesh = growth.mesh_from_grid(points, valid, 3)
    assert mesh["discarded_face_components"] == 1
    assert mesh["vertices"][:, 0].max() == 17


def plane(zheight, normal=(0, 1, 0)):
    z, s = np.indices((15, 30))
    vertices = (
        np.stack([z + 10, np.full_like(z, zheight), s + 20], axis=-1)
        .reshape(-1, 3)
        .astype(float)
    )
    return {
        "vertices": vertices,
        "normals": np.broadcast_to(normal, vertices.shape).copy(),
    }


def test_overlapping_same_sheet_is_compatible():
    assert growth.compatible_surfaces(plane(50), plane(50))[0]


def test_close_parallel_different_sheets_cannot_merge_through_brush_contact():
    assert not growth.compatible_surfaces(plane(50), plane(52))[0]


def test_crossing_surface_normals_are_incompatible():
    assert not growth.compatible_surfaces(plane(50), plane(50, normal=(0, 0, 1)))[0]


def test_real_patch_output_preserves_existing_mask_and_recovers_axial_sheet_area():
    p, ct, path = volume()
    base = p >= 0.35
    seed = np.rint(path).astype(int)
    base[24, seed[:, 0], seed[:, 1]] = True
    unchanged = base.copy()
    proposal = growth.propose_patch(
        p,
        ct,
        base,
        path,
        24,
        lambda z: ndimage.label(base[z], np.ones((3, 3), bool))[0],
    )
    after = base.copy()
    after[proposal["window"]] |= proposal["paint"]
    assert np.array_equal(base, unchanged)
    assert not (base & ~after).any()
    assert proposal["metrics"]["mesh_z_span"] >= 25
    assert proposal["metrics"]["added_voxels"] > 1000
    assert proposal["metrics"]["manifold_with_boundary"]
    import json

    json.dumps(proposal["metrics"], allow_nan=False)


def test_absent_evidence_cannot_be_promoted_to_a_full_patch():
    p, ct, path = volume(missing_axial=True)
    base = p >= 0.35
    seed = np.rint(path).astype(int)
    base[24, seed[:, 0], seed[:, 1]] = True
    try:
        proposal = growth.propose_patch(
            p,
            ct,
            base,
            path,
            24,
            lambda z: ndimage.label(base[z], np.ones((3, 3), bool))[0],
        )
    except ValueError as exc:
        assert "span" in str(exc) or "seed" in str(exc)
    else:
        assert proposal["metrics"]["mesh_z_span"] <= 5
