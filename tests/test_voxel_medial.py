from __future__ import annotations

import numpy as np
import pytest
import torch
from skimage.morphology import closing, skeletonize

from crossres_pred.voxel.loss import (
    CORRIDOR_CONSERVATIVE_LOSS_CONTRACT,
    DYNAMIC_MEDIAL_CONNECTIVITY_LOSS_CONTRACT,
    MEDIAL_CONSERVATIVE_LOSS_CONTRACT,
    PINNED_AXIAL_MEDIAL_CONSERVATIVE_LOSS_CONTRACT,
    PRESERVATION_CUTOFF_MEDIAL_CONSERVATIVE_LOSS_CONTRACT,
    PRESERVATION_MEDIAL_CONSERVATIVE_LOSS_CONTRACT,
    PRESERVATION_SOFT_FLOOR_MEDIAL_CONSERVATIVE_LOSS_CONTRACT,
    VoxelLossOptions,
    deep_supervision_loss,
    dice_ce_loss,
    dynamic_medial_connectivity_loss,
    loss_contract,
    pinned_axial_floor_loss,
    resize_medial_target,
)
from crossres_pred.voxel.medial import (
    FineMedialSurfaceReader,
    MedialProjectionOptions,
    center_radius_envelope_2d,
    project_fine_medial_patch,
    reconstruct_slicewise_center_radius,
    villa_medial_surface,
    villa_slicewise_center_radius,
)
from crossres_pred.voxel.registration import ChunkSupport, FineFieldWindowReader
from crossres_pred.voxel.schema import DenseFieldSpec

IDENTITY_AFFINE = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
)


def test_villa_medial_surface_is_slice_wise_then_closed_and_masked() -> None:
    mask = np.zeros((7, 33, 35), dtype=bool)
    mask[:, 11:20, 4:31] = True
    mask[3, 8:23, 15:20] = True

    expected = np.stack([skeletonize(section) for section in mask])
    expected = closing(expected) & mask
    actual = villa_medial_surface(mask)

    assert np.array_equal(actual, expected)
    assert np.all(actual <= mask)
    # A medial *surface* retains a centerline on every z section.  A generic
    # 3-D skeleton would incorrectly collapse this slab along z as well.
    assert np.all(actual.sum(axis=(1, 2)) > 0)
    assert actual.sum() < mask.sum() / 4


def test_villa_center_radius_reconstructs_without_adding_girth() -> None:
    mask = np.zeros((5, 35, 41), dtype=bool)
    mask[:, 12:23, 5:36] = True
    mask[:, 8:27, 18:23] = True

    centers, radii = villa_slicewise_center_radius(mask)
    reconstructed = reconstruct_slicewise_center_radius(centers, radii)

    assert np.array_equal(centers, villa_medial_surface(mask))
    assert np.all(radii[centers] > 0.0)
    assert np.count_nonzero(radii[~centers]) == 0
    # An EDT radius reaches the nearest background voxel center.  Open disks
    # must never include that center and therefore cannot add radial spill.
    assert not np.any(reconstructed & ~mask)
    assert np.count_nonzero(reconstructed) / np.count_nonzero(mask) > 0.95


def test_center_radius_transform_uses_physical_in_plane_sampling() -> None:
    mask = np.zeros((3, 11, 15), dtype=bool)
    mask[:, 3:8, 2:13] = True

    centers, unit_radii = villa_slicewise_center_radius(mask)
    physical_centers, physical_radii = villa_slicewise_center_radius(
        mask, sampling_yx=(2.0, 1.0)
    )

    assert np.array_equal(physical_centers, centers)
    assert float(physical_radii[physical_centers].max()) > float(
        unit_radii[centers].max()
    )
    reconstructed = reconstruct_slicewise_center_radius(
        physical_centers,
        physical_radii,
        sampling_yx=(2.0, 1.0),
    )
    assert not np.any(reconstructed & ~mask)


def test_center_radius_envelope_uses_weighted_not_nearest_center() -> None:
    centers = np.zeros((7, 9), dtype=bool)
    radii = np.zeros(centers.shape, dtype=np.float32)
    centers[3, 2] = True
    radii[3, 2] = 1.0
    centers[3, 6] = True
    radii[3, 6] = 5.0

    envelope, winning_radius = center_radius_envelope_2d(centers, radii)

    # Pixel (3, 3) is nearer the radius-one center, but the larger primitive
    # contains it with a greater r-distance margin and must own the envelope.
    assert float(envelope[3, 3]) == pytest.approx(2.0)
    assert float(winning_radius[3, 3]) == pytest.approx(5.0)
    assert float(envelope[3, 8]) == pytest.approx(3.0)


@pytest.mark.torch
def test_pinned_axial_loss_gives_each_route_one_weakest_tail_vote() -> None:
    floor_margin = float(torch.logit(torch.tensor(0.20)))
    margins = torch.tensor(
        [
            floor_margin - 1.0,
            floor_margin + 2.0,
            floor_margin - 2.0,
            floor_margin - 0.5,
            floor_margin + 1.0,
            floor_margin + 1.1,
            floor_margin + 1.2,
            floor_margin + 1.3,
            floor_margin + 1.4,
            floor_margin + 1.5,
        ]
    )
    logits = torch.zeros((1, 2, 1, 1, 10), requires_grad=True)
    with torch.no_grad():
        logits[0, 1, 0, 0] = margins
    bridge_ids = torch.tensor([[[[[1, 1, 2, 2, 2, 2, 2, 2, 2, 2]]]]])

    result = pinned_axial_floor_loss(
        logits,
        bridge_ids,
        probability_floor=0.20,
        bottom_fraction=0.25,
    )
    result.loss.backward()

    # Route one: 1.0. Route two: mean(2.0, 0.5) = 1.25. Routes vote equally.
    assert float(result.loss.detach()) == pytest.approx(1.125, abs=1.0e-6)
    assert float(result.groups) == 2.0
    assert float(result.target_voxels) == 10.0
    foreground_gradient = logits.grad[0, 1, 0, 0]
    assert torch.count_nonzero(foreground_gradient) == 3
    assert torch.all(foreground_gradient[[0, 2, 3]] < 0)


@pytest.mark.torch
def test_pinned_axial_floor_is_one_sided_and_uses_new_contract() -> None:
    logits = torch.zeros((1, 2, 1, 1, 3), requires_grad=True)
    with torch.no_grad():
        logits[:, 1] = 0.0  # foreground probability 0.5, safely above 0.20
    bridge_ids = torch.tensor([[[[[4, 4, 4]]]]])
    options = VoxelLossOptions(pinned_axial_weight=1.0)

    result = pinned_axial_floor_loss(logits, bridge_ids)
    result.loss.backward()

    assert loss_contract(options) == PINNED_AXIAL_MEDIAL_CONSERVATIVE_LOSS_CONTRACT
    assert float(result.loss.detach()) == 0.0
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.torch
def test_dynamic_connectivity_chooses_the_stronger_route_and_not_girth() -> None:
    probabilities = torch.full((1, 1, 5, 9), 0.01)
    probabilities[0, 0, 1, :] = 0.05
    probabilities[0, 0, 3, :] = 0.15
    logits = torch.zeros((1, 2, 1, 5, 9), requires_grad=True)
    with torch.no_grad():
        logits[:, 1] = torch.logit(probabilities)
    events = torch.zeros((1, 1, 1, 5, 9), dtype=torch.long)
    events[0, 0, 0, 1, :] = 1
    events[0, 0, 0, 3, :] = 1
    pins = torch.zeros_like(events)
    pins[0, 0, 0, 1, 0] = 1
    pins[0, 0, 0, 3, 0] = 1
    pins[0, 0, 0, 1, -1] = 2
    pins[0, 0, 0, 3, -1] = 2
    free = (pins > 0).long()

    result = dynamic_medial_connectivity_loss(
        logits,
        events,
        pins,
        free,
        probability_floor=0.20,
        propagation_steps=8,
    )
    result.loss.backward()

    expected = float(torch.logit(torch.tensor(0.20)) - torch.logit(torch.tensor(0.15)))
    assert float(result.loss.detach()) == pytest.approx(expected, abs=1.0e-5)
    assert float(result.events) == 1.0
    assert float(result.targets) == 1.0
    assert float(result.mean_bottleneck_probability) == pytest.approx(0.15)
    foreground_gradient = logits.grad[0, 1, 0]
    assert torch.count_nonzero(foreground_gradient[3, 1:-1]) > 0
    assert torch.count_nonzero(foreground_gradient[1]) == 0
    assert torch.count_nonzero(foreground_gradient[0]) == 0
    assert torch.count_nonzero(foreground_gradient[2]) == 0
    assert torch.count_nonzero(foreground_gradient[4]) == 0
    assert foreground_gradient[3, 0] == 0
    assert foreground_gradient[3, -1] == 0


@pytest.mark.torch
def test_dynamic_connectivity_is_existential_and_one_sided() -> None:
    probabilities = torch.full((1, 1, 3, 7), 0.01)
    probabilities[0, 0, 1, :] = 0.30
    logits = torch.zeros((1, 2, 1, 3, 7), requires_grad=True)
    with torch.no_grad():
        logits[:, 1] = torch.logit(probabilities)
    events = torch.zeros((1, 1, 1, 3, 7), dtype=torch.long)
    events[0, 0, 0, 1, :] = 4
    pins = torch.zeros_like(events)
    pins[0, 0, 0, 1, 0] = 1
    pins[0, 0, 0, 1, -1] = 2
    free = (pins > 0).long()
    options = VoxelLossOptions(dynamic_medial_connectivity_weight=1.0)

    result = dynamic_medial_connectivity_loss(
        logits, events, pins, free, propagation_steps=6
    )
    result.loss.backward()

    assert loss_contract(options) == DYNAMIC_MEDIAL_CONNECTIVITY_LOSS_CONTRACT
    assert float(result.loss.detach()) == 0.0
    assert torch.count_nonzero(logits.grad) == 0
    with pytest.raises(ValueError, match="mutually exclusive"):
        VoxelLossOptions(
            pinned_axial_weight=1.0,
            dynamic_medial_connectivity_weight=1.0,
        ).validate()


def test_fine_medial_projection_is_binary_or_and_has_independent_validity() -> None:
    fine = np.zeros((40, 48, 48), dtype=np.uint8)
    fine[8:32, 19:27, 8:40] = 1
    field = DenseFieldSpec(volume="unused.npy", encoding="labels", positive_labels=(1,))
    support = ChunkSupport(
        shape_zyx=fine.shape,
        chunks_zyx=(20, 24, 24),
        grid_zyx=(2, 2, 2),
        present_ids=None,
    )
    field_reader = FineFieldWindowReader(fine, field, support)
    options = MedialProjectionOptions(
        halo_zyx=(1, 8, 8), skeleton_workers=1, max_cache_chunks=16
    )
    with FineMedialSurfaceReader(field_reader, options=options) as reader:
        crest, crest_valid, stats = project_fine_medial_patch(
            reader,
            IDENTITY_AFFINE,
            (8, 12, 12),
            (24, 24, 24),
        )

    expected = villa_medial_surface(fine)[8:32, 12:36, 12:36]
    assert np.array_equal(crest.astype(bool), expected & (crest_valid > 0))
    assert set(np.unique(crest)) <= {0, 1}
    assert set(np.unique(crest_valid)) <= {0, 1}
    assert np.all(crest <= crest_valid)
    assert int(stats["crest_voxels"]) == int(crest.sum()) > 0
    assert stats["projection_contract"].endswith("nearest-coarse-or-max-v1")


def test_medial_validity_is_not_coupled_to_occupancy_validity() -> None:
    fine = np.zeros((24, 32, 32), dtype=np.uint8)
    fine[:, 10:22, 7:25] = 1
    field = DenseFieldSpec(volume="unused.npy", encoding="labels", positive_labels=(1,))
    support = ChunkSupport(
        shape_zyx=fine.shape,
        chunks_zyx=fine.shape,
        grid_zyx=(1, 1, 1),
        present_ids=None,
    )
    reader = FineFieldWindowReader(fine, field, support)
    options = MedialProjectionOptions(
        halo_zyx=(1, 4, 4), skeleton_workers=1, max_cache_chunks=2
    )
    with FineMedialSurfaceReader(reader, options=options) as medial_reader:
        crest, crest_valid, _ = project_fine_medial_patch(
            medial_reader,
            IDENTITY_AFFINE,
            (4, 4, 4),
            (16, 24, 24),
        )

    # Occupancy q applies a separate Gaussian-support erosion.  A complete
    # medial halo is independently sufficient: this target must survive even
    # at a location an occupancy-valid mask could mark unknown.
    assert int(crest.sum()) > 0
    assert np.all(crest <= crest_valid)


def test_reference_halo_64_is_supported_for_128_voxel_chunks() -> None:
    fine = np.zeros((128, 128, 128), dtype=np.uint8)
    field = DenseFieldSpec(volume="unused.npy", encoding="labels", positive_labels=(1,))
    support = ChunkSupport(
        shape_zyx=fine.shape,
        chunks_zyx=fine.shape,
        grid_zyx=(1, 1, 1),
        present_ids=None,
    )
    field_reader = FineFieldWindowReader(fine, field, support)

    reader = FineMedialSurfaceReader(
        field_reader,
        options=MedialProjectionOptions(halo_zyx=(1, 64, 64)),
    )
    reader.close()
    with pytest.raises(ValueError, match="at most two support chunks"):
        FineMedialSurfaceReader(
            field_reader,
            options=MedialProjectionOptions(halo_zyx=(1, 65, 65)),
        )


@pytest.mark.torch
def test_medial_recall_pushes_only_the_thin_crest_toward_foreground() -> None:
    logits = torch.zeros((1, 2, 1, 1, 7), requires_grad=True)
    target = torch.zeros((1, 1, 1, 7), dtype=torch.long)
    q = torch.zeros_like(target, dtype=torch.float32)
    valid = torch.ones_like(q)
    crest = torch.zeros_like(q)
    crest[:, :, :, 3] = 1
    options = VoxelLossOptions(
        cross_entropy_weight=0.0,
        dice_weight=1.0,
        medial_recall_weight=1.0,
    )

    result = dice_ce_loss(
        logits,
        target,
        teacher_q=q,
        target_valid=valid,
        teacher_crest=crest,
        teacher_crest_valid=valid,
        teacher_crest_available=torch.tensor([True]),
        options=options,
    )
    result.medial_recall.backward()

    assert loss_contract(options) == MEDIAL_CONSERVATIVE_LOSS_CONTRACT
    # Villa's recall smoothing is exactly one: (0.5 + 1) / (1 + 1) = 0.75.
    assert float(result.medial_recall.detach()) == pytest.approx(0.25, abs=1.0e-5)
    foreground_gradient = logits.grad[0, 1, 0, 0]
    assert float(foreground_gradient[3]) < 0
    assert torch.count_nonzero(foreground_gradient) == 1


@pytest.mark.torch
def test_m7_preservation_pushes_only_teacher_near_m7_foreground() -> None:
    logits = torch.zeros((1, 2, 1, 1, 9), requires_grad=True)
    target = torch.zeros((1, 1, 1, 9), dtype=torch.long)
    q = torch.zeros_like(target, dtype=torch.float32)
    valid = torch.ones_like(q)
    crest = torch.zeros_like(q)
    crest[:, :, :, 4] = 1
    anchor = torch.zeros_like(logits)
    anchor[:, 0] = 5.0
    anchor[:, 1, 0, 0, 3] = 6.0
    anchor[:, 1, 0, 0, 8] = 6.0
    options = VoxelLossOptions(
        cross_entropy_weight=0.0,
        dice_weight=1.0,
        m7_preservation_weight=1.0,
        m7_preservation_radius=2,
    )

    result = dice_ce_loss(
        logits,
        target,
        teacher_q=q,
        target_valid=valid,
        teacher_crest=crest,
        teacher_crest_valid=valid,
        teacher_crest_available=torch.tensor([True]),
        m7_anchor_logits=anchor,
        options=options,
    )
    result.m7_preservation.backward()

    assert loss_contract(options) == PRESERVATION_MEDIAL_CONSERVATIVE_LOSS_CONTRACT
    foreground_gradient = logits.grad[0, 1, 0, 0]
    assert float(foreground_gradient[3]) < 0
    assert float(foreground_gradient[8]) == pytest.approx(0.0, abs=1.0e-7)
    assert torch.count_nonzero(foreground_gradient) == 1


@pytest.mark.torch
def test_m7_preservation_anchor_cutoff_includes_low_confidence_m7() -> None:
    target = torch.zeros((1, 1, 1, 5), dtype=torch.long)
    q = torch.zeros_like(target, dtype=torch.float32)
    valid = torch.ones_like(q)
    crest = torch.zeros_like(q)
    crest[:, :, :, 2] = 1
    anchor = torch.zeros((1, 2, 1, 1, 5))
    anchor[:, 1] = -1.0  # foreground probability 0.269, below legacy 0.50

    legacy_logits = torch.zeros((1, 2, 1, 1, 5), requires_grad=True)
    legacy = dice_ce_loss(
        legacy_logits,
        target,
        teacher_q=q,
        target_valid=valid,
        teacher_crest=crest,
        teacher_crest_valid=valid,
        teacher_crest_available=torch.tensor([True]),
        m7_anchor_logits=anchor,
        options=VoxelLossOptions(
            cross_entropy_weight=0.0,
            dice_weight=1.0,
            m7_preservation_weight=1.0,
            m7_preservation_radius=1,
        ),
    )
    legacy.m7_preservation.backward()
    assert torch.count_nonzero(legacy_logits.grad) == 0

    cutoff_logits = torch.zeros((1, 2, 1, 1, 5), requires_grad=True)
    options = VoxelLossOptions(
        cross_entropy_weight=0.0,
        dice_weight=1.0,
        m7_preservation_weight=1.0,
        m7_preservation_radius=1,
        m7_preservation_anchor_threshold=0.2,
    )
    cutoff = dice_ce_loss(
        cutoff_logits,
        target,
        teacher_q=q,
        target_valid=valid,
        teacher_crest=crest,
        teacher_crest_valid=valid,
        teacher_crest_available=torch.tensor([True]),
        m7_anchor_logits=anchor,
        options=options,
    )
    cutoff.m7_preservation.backward()

    assert loss_contract(options) == (
        PRESERVATION_CUTOFF_MEDIAL_CONSERVATIVE_LOSS_CONTRACT
    )
    foreground_gradient = cutoff_logits.grad[0, 1, 0, 0]
    assert float(foreground_gradient[1]) < 0
    assert float(foreground_gradient[2]) < 0
    assert float(foreground_gradient[3]) < 0
    assert torch.count_nonzero(foreground_gradient) == 3


@pytest.mark.torch
def test_m7_preservation_soft_floor_only_resists_probability_decreases() -> None:
    target = torch.zeros((1, 1, 1, 5), dtype=torch.long)
    q = torch.zeros_like(target, dtype=torch.float32)
    valid = torch.ones_like(q)
    crest = torch.zeros_like(q)
    crest[:, :, :, 2] = 1
    anchor = torch.zeros((1, 2, 1, 1, 5))
    anchor[:, 1] = -1.0  # foreground probability 0.269
    options = VoxelLossOptions(
        cross_entropy_weight=0.0,
        dice_weight=1.0,
        m7_preservation_weight=1.0,
        m7_preservation_radius=1,
        m7_preservation_anchor_threshold=0.2,
        m7_preservation_soft_floor=True,
    )

    unchanged_logits = anchor.clone().requires_grad_(True)
    unchanged = dice_ce_loss(
        unchanged_logits,
        target,
        teacher_q=q,
        target_valid=valid,
        teacher_crest=crest,
        teacher_crest_valid=valid,
        teacher_crest_available=torch.tensor([True]),
        m7_anchor_logits=anchor,
        options=options,
    )
    unchanged.m7_preservation.backward()
    assert float(unchanged.m7_preservation.detach()) == pytest.approx(0.0, abs=1.0e-7)
    assert torch.count_nonzero(unchanged_logits.grad) == 0

    moved_logits = anchor.clone()
    moved_logits[:, 1, 0, 0, 1] = -2.0  # below M7: floor must push upward
    moved_logits[:, 1, 0, 0, 2] = 0.0  # above M7: floor must do nothing
    moved_logits = moved_logits.requires_grad_(True)
    moved = dice_ce_loss(
        moved_logits,
        target,
        teacher_q=q,
        target_valid=valid,
        teacher_crest=crest,
        teacher_crest_valid=valid,
        teacher_crest_available=torch.tensor([True]),
        m7_anchor_logits=anchor,
        options=options,
    )
    moved.m7_preservation.backward()

    assert loss_contract(options) == (
        PRESERVATION_SOFT_FLOOR_MEDIAL_CONSERVATIVE_LOSS_CONTRACT
    )
    foreground_gradient = moved_logits.grad[0, 1, 0, 0]
    assert float(foreground_gradient[1]) < 0
    assert float(foreground_gradient[2]) == pytest.approx(0.0, abs=1.0e-7)
    assert torch.count_nonzero(foreground_gradient) == 1


@pytest.mark.torch
def test_separation_shell_is_seeded_by_crest_and_never_penalizes_crest() -> None:
    target = torch.zeros((1, 1, 1, 7), dtype=torch.long)
    q = torch.zeros_like(target, dtype=torch.float32)
    valid = torch.ones_like(q)
    crest = torch.zeros_like(q)
    crest[:, :, :, 3] = 1
    options = VoxelLossOptions(
        cross_entropy_weight=0.0,
        dice_weight=1.0,
        separation_weight=1.0,
        separation_radius=2,
    )
    crest_foreground = torch.zeros((1, 2, 1, 1, 7))
    crest_foreground[:, 1, 0, 0, 3] = 7.0
    shell_foreground = torch.zeros_like(crest_foreground)
    shell_foreground[:, 1, 0, 0, 4] = 7.0

    def separation(logits: torch.Tensor) -> float:
        return float(
            dice_ce_loss(
                logits,
                target,
                teacher_q=q,
                target_valid=valid,
                teacher_crest=crest,
                teacher_crest_valid=valid,
                teacher_crest_available=torch.tensor([True]),
                options=options,
            ).separation
        )

    assert separation(crest_foreground) < 1.0
    assert separation(shell_foreground) > 1.0


@pytest.mark.torch
def test_separation_shell_falls_back_to_occupancy_in_medial_invalid_rim() -> None:
    target = torch.zeros((1, 1, 1, 9), dtype=torch.long)
    q = torch.zeros_like(target, dtype=torch.float32)
    valid = torch.ones_like(q)
    q[:, :, :, 2] = 1
    q[:, :, :, 6] = 1
    crest = torch.zeros_like(q)
    crest[:, :, :, 2] = 1
    crest_valid = torch.zeros_like(q)
    crest_valid[:, :, :, :5] = 1
    logits = torch.zeros((1, 2, 1, 1, 9))
    # This false foreground lies beside the q-positive voxel at index 6, in
    # the medial-invalid rim. The occupancy fallback must fence it.
    logits[:, 1, 0, 0, 7] = 7.0
    options = VoxelLossOptions(
        cross_entropy_weight=0.0,
        dice_weight=1.0,
        separation_weight=1.0,
        separation_radius=1,
    )

    result = dice_ce_loss(
        logits,
        target,
        teacher_q=q,
        target_valid=valid,
        teacher_crest=crest,
        teacher_crest_valid=crest_valid,
        teacher_crest_available=torch.tensor([True]),
        options=options,
    )

    assert float(result.separation) > 1.0


@pytest.mark.torch
def test_unknown_kl_corridor_frees_growth_near_a_known_crest_only() -> None:
    target = torch.full((1, 1, 1, 7), 2, dtype=torch.long)
    target[:, :, :, 3] = 0
    q = torch.zeros_like(target, dtype=torch.float32)
    valid = torch.zeros_like(q)
    valid[:, :, :, 3] = 1
    crest = torch.zeros_like(q)
    crest[:, :, :, 3] = 1
    crest_valid = valid.clone()
    anchor = torch.zeros((1, 2, 1, 1, 7))
    anchor[:, 0] = 5.0
    options = VoxelLossOptions(
        dice_weight=0.0,
        m7_anchor_weight=1.0,
        m7_anchor_known_agreement=False,
        m7_anchor_unknown_corridor_radius=1,
    )

    def anchored_at(index: int) -> float:
        student = anchor.clone()
        student[:, 0, 0, 0, index] = 0.0
        student[:, 1, 0, 0, index] = 5.0
        return float(
            dice_ce_loss(
                student,
                target,
                teacher_q=q,
                target_valid=valid,
                teacher_crest=crest,
                teacher_crest_valid=crest_valid,
                teacher_crest_available=torch.tensor([True]),
                m7_anchor_logits=anchor,
                options=options,
            ).m7_anchor_kl
        )

    assert anchored_at(4) == pytest.approx(0.0, abs=1.0e-6)
    assert anchored_at(6) > 0.5


@pytest.mark.torch
def test_deep_supervision_medial_resize_preserves_crest_and_erodes_validity() -> None:
    crest = torch.zeros((1, 1, 8, 8, 8))
    valid = torch.ones_like(crest)
    crest[:, :, 3, 3, 3] = 1
    valid[:, :, 0, 0, 0] = 0

    resized_crest, resized_valid = resize_medial_target(crest, valid, (4, 4, 4))

    assert int(torch.count_nonzero(resized_crest)) == 1
    assert float(resized_crest[0, 0, 1, 1, 1]) == 1.0
    assert float(resized_valid[0, 0, 0, 0, 0]) == 0.0
    assert torch.all(resized_crest <= resized_valid)


@pytest.mark.torch
def test_medial_target_requires_its_own_validity_mask() -> None:
    logits = torch.zeros((1, 2, 1, 1, 1))
    target = torch.zeros((1, 1, 1, 1), dtype=torch.long)
    with pytest.raises(ValueError, match="own voxel-valid"):
        dice_ce_loss(
            logits,
            target,
            teacher_crest=torch.ones_like(target, dtype=torch.float32),
        )


@pytest.mark.torch
def test_corridor_only_contract_does_not_claim_medial_recall() -> None:
    options = VoxelLossOptions(m7_anchor_unknown_corridor_radius=2)

    assert loss_contract(options) == CORRIDOR_CONSERVATIVE_LOSS_CONTRACT
    assert "medial" not in loss_contract(options)


@pytest.mark.torch
def test_deep_supervision_reports_medial_and_shell_per_scale() -> None:
    outputs = [
        torch.zeros((1, 2, 8, 8, 8)),
        torch.zeros((1, 2, 4, 4, 4)),
        torch.zeros((1, 2, 2, 2, 2)),
    ]
    target = torch.zeros((1, 1, 8, 8, 8), dtype=torch.long)
    q = torch.zeros_like(target, dtype=torch.float32)
    valid = torch.ones_like(q)
    crest = torch.zeros_like(q)
    crest[:, :, 3, 3, 3] = 1
    options = VoxelLossOptions(
        medial_recall_weight=1.0,
        separation_weight=2.0,
    )

    _, components = deep_supervision_loss(
        outputs,
        target,
        q,
        valid,
        teacher_crest=crest,
        teacher_crest_valid=valid,
        teacher_crest_available=torch.tensor([True]),
        options=options,
    )

    for index in range(3):
        assert f"medial_recall_ds{index}" in components
        assert f"separation_ds{index}" in components


@pytest.mark.torch
def test_deep_supervision_adds_pinned_axial_loss_once_at_full_resolution() -> None:
    floor_margin = float(torch.logit(torch.tensor(0.20)))
    outputs = [
        torch.zeros((1, 2, 4, 4, 4)),
        torch.zeros((1, 2, 2, 2, 2)),
    ]
    outputs[0][:, 1, 1, 1, 1] = floor_margin - 1.0
    target = torch.zeros((1, 1, 4, 4, 4), dtype=torch.long)
    bridge_ids = torch.zeros_like(target)
    bridge_ids[:, :, 1, 1, 1] = 9
    base_options = VoxelLossOptions()
    axial_options = VoxelLossOptions(pinned_axial_weight=2.0)

    base_total, _ = deep_supervision_loss(
        outputs,
        target,
        options=base_options,
    )
    axial_total, components = deep_supervision_loss(
        outputs,
        target,
        pinned_medial_bridge=bridge_ids,
        options=axial_options,
    )

    assert float(components["pinned_axial_loss"]) == pytest.approx(1.0)
    assert float(components["pinned_axial_groups"]) == 1.0
    assert float(axial_total - base_total) == pytest.approx(2.0)


@pytest.mark.torch
def test_deep_supervision_adds_dynamic_connectivity_once_at_full_resolution() -> None:
    outputs = [
        torch.zeros((1, 2, 1, 3, 5)),
        torch.zeros((1, 2, 1, 2, 3)),
    ]
    with torch.no_grad():
        outputs[0][:, 1, 0, 1, :] = float(torch.logit(torch.tensor(0.10)))
    target = torch.zeros((1, 1, 1, 3, 5), dtype=torch.long)
    events = torch.zeros_like(target)
    events[:, :, 0, 1, :] = 7
    pins = torch.zeros_like(target)
    pins[:, :, 0, 1, 0] = 1
    pins[:, :, 0, 1, -1] = 2
    free = (pins > 0).long()
    base_total, _ = deep_supervision_loss(outputs, target)
    options = VoxelLossOptions(
        dynamic_medial_connectivity_weight=2.0,
        dynamic_medial_connectivity_steps=4,
    )

    connectivity_total, components = deep_supervision_loss(
        outputs,
        target,
        dynamic_connectivity_event=events,
        dynamic_connectivity_pins=pins,
        dynamic_connectivity_free=free,
        options=options,
    )

    expected = float(torch.logit(torch.tensor(0.20)) - torch.logit(torch.tensor(0.10)))
    assert float(components["dynamic_medial_connectivity_loss"]) == pytest.approx(
        expected, abs=1.0e-5
    )
    assert float(components["dynamic_medial_connectivity_events"]) == 1.0
    assert float(connectivity_total - base_total) == pytest.approx(
        2.0 * expected, abs=1.0e-5
    )
