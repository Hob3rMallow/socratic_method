from __future__ import annotations

import numpy as np
from crossres_pred.voxel.ridge_growth import grow_probability_ridges
from scipy import ndimage

STRUCTURE_6 = ndimage.generate_binary_structure(3, 1)


def test_ridge_growth_extends_a_supported_thin_line() -> None:
    probability = np.zeros((15, 15, 15), dtype=np.float32)
    seed = np.zeros_like(probability, dtype=bool)
    seed[7, 7, 2:5] = True
    probability[seed] = 0.9
    probability[7, 7, 5:11] = 0.44

    result = grow_probability_ridges(
        probability,
        seed,
        support_threshold=0.4,
        max_steps=8,
    )

    assert np.all(result.mask[7, 7, 2:11])
    assert result.added_positive == 6
    assert result.seed_components == result.final_components == 1
    assert not np.any(ndimage.binary_erosion(result.mask, structure=STRUCTURE_6))


def test_ridge_growth_does_not_join_established_components() -> None:
    probability = np.zeros((15, 15, 15), dtype=np.float32)
    seed = np.zeros_like(probability, dtype=bool)
    seed[7, 7, 2:5] = True
    seed[7, 7, 10:13] = True
    probability[seed] = 0.9
    probability[7, 7, 5:10] = 0.44

    result = grow_probability_ridges(
        probability,
        seed,
        support_threshold=0.4,
        max_steps=8,
    )

    assert result.seed_components == result.final_components == 2
    assert result.component_conflict_rejections > 0
    assert not np.all(result.mask[7, 7, 5:10])


def test_ridge_growth_does_not_create_new_erosion_interior() -> None:
    probability = np.full((11, 11, 11), 0.8, dtype=np.float32)
    seed = np.zeros_like(probability, dtype=bool)
    seed[4:7, 4:7, 4:7] = True
    baseline_interior = ndimage.binary_erosion(seed, structure=STRUCTURE_6)

    result = grow_probability_ridges(
        probability,
        seed,
        support_threshold=0.4,
        max_steps=4,
    )

    final_interior = ndimage.binary_erosion(result.mask, structure=STRUCTURE_6)
    assert not np.any(final_interior & ~baseline_interior)
    assert result.thickness_rejections > 0
    assert result.seed_components == result.final_components == 1
