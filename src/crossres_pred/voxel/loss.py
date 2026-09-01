from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch.nn import functional as F

IGNORE_LABEL = 2
LOSS_CONTRACT = "soft-dice-ce-per-sample-known-v3"
CONSERVATIVE_LOSS_CONTRACT = (
    "soft-ce-weighted-dice-separation-shell-conservative-m7-kl-per-sample-v2"
)
CONFIDENT_CONSERVATIVE_LOSS_CONTRACT = (
    "soft-ce-weighted-dice-separation-shell-confident-m7-kl-per-sample-v3"
)
CORRIDOR_CONSERVATIVE_LOSS_CONTRACT = (
    "soft-ce-weighted-dice-separation-shell-m7-kl-unknown-corridor-v4"
)
MEDIAL_CONSERVATIVE_LOSS_CONTRACT = (
    "soft-occupancy-ce-dice-villa-medial-recall-crest-shell-kl-corridor-v4"
)
PRESERVATION_MEDIAL_CONSERVATIVE_LOSS_CONTRACT = (
    "soft-occupancy-ce-dice-villa-medial-crest-shell-kl-corridor-m7-preservation-v5"
)
PRESERVATION_CUTOFF_MEDIAL_CONSERVATIVE_LOSS_CONTRACT = (
    "soft-occupancy-ce-dice-villa-medial-crest-shell-kl-corridor-"
    "m7-preservation-anchor-cutoff-v6"
)
PRESERVATION_SOFT_FLOOR_MEDIAL_CONSERVATIVE_LOSS_CONTRACT = (
    "soft-occupancy-ce-dice-villa-medial-crest-shell-kl-corridor-"
    "m7-preservation-one-sided-soft-floor-v7"
)
PINNED_AXIAL_MEDIAL_CONSERVATIVE_LOSS_CONTRACT = (
    "soft-occupancy-ce-dice-villa-medial-crest-shell-kl-corridor-"
    "m7-preservation-pinned-axial-floor-v8"
)
DYNAMIC_MEDIAL_CONNECTIVITY_LOSS_CONTRACT = (
    "soft-occupancy-ce-dice-villa-medial-crest-shell-kl-corridor-"
    "m7-preservation-dynamic-widest-path-v9"
)
VILLA_MEDIAL_RECALL_SMOOTH = 1.0
M7_PINNED_AXIAL_PROBABILITY_FLOOR = 0.20
M7_PINNED_AXIAL_BOTTOM_FRACTION = 0.10
DYNAMIC_MEDIAL_CONNECTIVITY_PROBABILITY_FLOOR = 0.20
DYNAMIC_MEDIAL_CONNECTIVITY_STEPS = 96


@dataclass(frozen=True)
class VoxelLossOptions:
    cross_entropy_weight: float = 1.0
    dice_weight: float = 1.0
    medial_recall_weight: float = 0.0
    separation_weight: float = 0.0
    separation_radius: int = 2
    separation_max_teacher_q: float = 0.1
    m7_anchor_weight: float = 0.0
    m7_anchor_known_agreement: bool = True
    m7_anchor_confident_agreement: bool = False
    m7_anchor_unknown_corridor_radius: int = 0
    m7_preservation_weight: float = 0.0
    m7_preservation_radius: int = 2
    m7_preservation_anchor_threshold: float = 0.5
    m7_preservation_soft_floor: bool = False
    pinned_axial_weight: float = 0.0
    pinned_axial_probability_floor: float = M7_PINNED_AXIAL_PROBABILITY_FLOOR
    pinned_axial_bottom_fraction: float = M7_PINNED_AXIAL_BOTTOM_FRACTION
    dynamic_medial_connectivity_weight: float = 0.0
    dynamic_medial_connectivity_probability_floor: float = (
        DYNAMIC_MEDIAL_CONNECTIVITY_PROBABILITY_FLOOR
    )
    dynamic_medial_connectivity_steps: int = DYNAMIC_MEDIAL_CONNECTIVITY_STEPS

    def validate(self) -> None:
        for name, value in (
            ("cross_entropy_weight", self.cross_entropy_weight),
            ("dice_weight", self.dice_weight),
            ("medial_recall_weight", self.medial_recall_weight),
            ("separation_weight", self.separation_weight),
            ("m7_anchor_weight", self.m7_anchor_weight),
            ("m7_preservation_weight", self.m7_preservation_weight),
            ("pinned_axial_weight", self.pinned_axial_weight),
            (
                "dynamic_medial_connectivity_weight",
                self.dynamic_medial_connectivity_weight,
            ),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.cross_entropy_weight + self.dice_weight <= 0:
            raise ValueError("cross-entropy and Dice weights cannot both be zero")
        if self.separation_radius <= 0:
            raise ValueError("separation radius must be positive")
        if self.m7_anchor_unknown_corridor_radius < 0:
            raise ValueError("M7 unknown-corridor radius must be non-negative")
        if self.m7_preservation_radius <= 0:
            raise ValueError("M7 preservation radius must be positive")
        if not 0 < self.m7_preservation_anchor_threshold <= 1:
            raise ValueError("M7 preservation anchor threshold must be in (0, 1]")
        if not 0 <= self.separation_max_teacher_q < 0.5:
            raise ValueError("separation teacher-q ceiling must be in [0, 0.5)")
        if not isinstance(self.m7_anchor_known_agreement, bool):
            raise TypeError("M7 known-agreement anchoring flag must be boolean")
        if not isinstance(self.m7_anchor_confident_agreement, bool):
            raise TypeError("M7 confident-agreement anchoring flag must be boolean")
        if not isinstance(self.m7_preservation_soft_floor, bool):
            raise TypeError("M7 preservation soft-floor flag must be boolean")
        if not 0 < self.pinned_axial_probability_floor < 1:
            raise ValueError("pinned axial probability floor must be in (0, 1)")
        if not 0 < self.pinned_axial_bottom_fraction <= 1:
            raise ValueError("pinned axial bottom fraction must be in (0, 1]")
        if not 0 < self.dynamic_medial_connectivity_probability_floor < 1:
            raise ValueError(
                "dynamic medial connectivity probability floor must be in (0, 1)"
            )
        if self.dynamic_medial_connectivity_steps <= 0:
            raise ValueError("dynamic medial connectivity steps must be positive")
        if self.pinned_axial_weight > 0 and self.dynamic_medial_connectivity_weight > 0:
            raise ValueError(
                "fixed axial and dynamic medial connectivity losses are mutually exclusive"
            )
        if self.m7_anchor_confident_agreement and not self.m7_anchor_known_agreement:
            raise ValueError(
                "confident-agreement anchoring refines the known-agreement mask "
                "and requires it"
            )

    @property
    def is_legacy(self) -> bool:
        return self == VoxelLossOptions()


DEFAULT_VOXEL_LOSS_OPTIONS = VoxelLossOptions()


def loss_contract(options: VoxelLossOptions) -> str:
    options.validate()
    if options.is_legacy:
        return LOSS_CONTRACT
    if options.dynamic_medial_connectivity_weight > 0:
        return DYNAMIC_MEDIAL_CONNECTIVITY_LOSS_CONTRACT
    if options.pinned_axial_weight > 0:
        return PINNED_AXIAL_MEDIAL_CONSERVATIVE_LOSS_CONTRACT
    if options.m7_preservation_weight > 0:
        if options.m7_preservation_soft_floor:
            return PRESERVATION_SOFT_FLOOR_MEDIAL_CONSERVATIVE_LOSS_CONTRACT
        if options.m7_preservation_anchor_threshold != 0.5:
            return PRESERVATION_CUTOFF_MEDIAL_CONSERVATIVE_LOSS_CONTRACT
        return PRESERVATION_MEDIAL_CONSERVATIVE_LOSS_CONTRACT
    if options.medial_recall_weight > 0:
        return MEDIAL_CONSERVATIVE_LOSS_CONTRACT
    if options.m7_anchor_unknown_corridor_radius > 0:
        return CORRIDOR_CONSERVATIVE_LOSS_CONTRACT
    if options.m7_anchor_confident_agreement:
        return CONFIDENT_CONSERVATIVE_LOSS_CONTRACT
    return CONSERVATIVE_LOSS_CONTRACT


@dataclass(frozen=True)
class LossResult:
    total: torch.Tensor
    cross_entropy: torch.Tensor
    dice: torch.Tensor
    medial_recall: torch.Tensor
    separation: torch.Tensor
    m7_anchor_kl: torch.Tensor
    m7_preservation: torch.Tensor
    pinned_axial: torch.Tensor
    dynamic_medial_connectivity: torch.Tensor


@dataclass(frozen=True)
class PinnedAxialLossResult:
    loss: torch.Tensor
    groups: torch.Tensor
    target_voxels: torch.Tensor


@dataclass(frozen=True)
class DynamicMedialConnectivityLossResult:
    loss: torch.Tensor
    events: torch.Tensor
    targets: torch.Tensor
    mean_bottleneck_probability: torch.Tensor


def _per_sample_masked_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    spatial = tuple(range(1, value.ndim))
    counts = mask.sum(dim=spatial)
    selected = counts > 0
    if not bool(selected.any()):
        return value.sum() * 0.0
    means = (value * mask.to(value.dtype)).sum(dim=spatial) / counts.clamp_min(1).to(
        value.dtype
    )
    return means[selected].mean()


def pinned_axial_floor_loss(
    logits: torch.Tensor,
    bridge_ids: torch.Tensor,
    *,
    probability_floor: float = M7_PINNED_AXIAL_PROBABILITY_FLOOR,
    bottom_fraction: float = M7_PINNED_AXIAL_BOTTOM_FRACTION,
) -> PinnedAxialLossResult:
    """Raise only the weakest part of each fixed medial route to M7's floor.

    Every nonzero integer identifies one M7-pinned teacher-medial route.  Each
    route receives one vote regardless of length.  A one-sided logit hinge is
    evaluated on its lowest-probability fraction and is exactly zero once that
    route has reached the independently published M7 foreground threshold.
    """

    if logits.ndim != 5 or logits.shape[1] != 2:
        raise ValueError(f"expected Bx2xDxHxW logits, got {tuple(logits.shape)}")
    values = bridge_ids
    if values.ndim == 5 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 4 or values.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("pinned medial bridge IDs must match the logits")
    if not 0 < probability_floor < 1:
        raise ValueError("pinned axial probability floor must be in (0, 1)")
    if not 0 < bottom_fraction <= 1:
        raise ValueError("pinned axial bottom fraction must be in (0, 1]")
    if values.is_floating_point() and (
        not bool(torch.isfinite(values).all())
        or not bool(torch.equal(values, values.round()))
    ):
        raise ValueError("pinned medial bridge IDs must be finite integers")
    identifiers = values.long()
    if bool((identifiers < 0).any()):
        raise ValueError("pinned medial bridge IDs must be non-negative")

    foreground_margin = logits.float()[:, 1] - logits.float()[:, 0]
    floor_margin = math.log(probability_floor / (1.0 - probability_floor))
    sample_losses: list[torch.Tensor] = []
    group_count = 0
    target_voxels = 0
    for sample_index in range(identifiers.shape[0]):
        group_losses: list[torch.Tensor] = []
        for identifier in torch.unique(identifiers[sample_index]).tolist():
            if identifier == 0:
                continue
            group_values = foreground_margin[sample_index][
                identifiers[sample_index] == int(identifier)
            ]
            count = int(group_values.numel())
            if count == 0:
                continue
            selected_count = max(1, math.ceil(bottom_fraction * count))
            weakest = torch.topk(
                group_values,
                selected_count,
                largest=False,
                sorted=False,
            ).values
            group_losses.append(F.relu(floor_margin - weakest).mean())
            group_count += 1
            target_voxels += count
        if group_losses:
            sample_losses.append(torch.stack(group_losses).mean())
    zero = logits.sum() * 0.0
    loss = torch.stack(sample_losses).mean() if sample_losses else zero
    return PinnedAxialLossResult(
        loss=loss,
        groups=logits.new_tensor(float(group_count)),
        target_voxels=logits.new_tensor(float(target_voxels)),
    )


def dynamic_medial_connectivity_loss(
    logits: torch.Tensor,
    event_ids: torch.Tensor,
    pin_membership: torch.Tensor,
    free_anchors: torch.Tensor,
    *,
    probability_floor: float = DYNAMIC_MEDIAL_CONNECTIVITY_PROBABILITY_FLOOR,
    propagation_steps: int = DYNAMIC_MEDIAL_CONNECTIVITY_STEPS,
) -> DynamicMedialConnectivityLossResult:
    """Raise the bottleneck of the best path connecting all M7 pins.

    Each nonzero event ID defines one teacher-medial corridor. Pin membership is
    an uint8-style bitset whose consecutive bits identify two to eight M7
    component contact sets. The recurrence is exact max-min reachability for
    paths no longer than ``propagation_steps``. It therefore asks for one viable
    path, rather than supervising a preselected route. Free anchors have unit
    capacity and carry no direct gradient.
    """

    if logits.ndim != 5 or logits.shape[1] != 2:
        raise ValueError(f"expected Bx2xDxHxW logits, got {tuple(logits.shape)}")

    expected = logits.shape[:1] + logits.shape[2:]

    def labels(values: torch.Tensor, name: str) -> torch.Tensor:
        if values.ndim == 5 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 4 or values.shape != expected:
            raise ValueError(f"{name} must match the logits")
        if values.is_floating_point() and (
            not bool(torch.isfinite(values).all())
            or not bool(torch.equal(values, values.round()))
        ):
            raise ValueError(f"{name} must contain finite integers")
        return values.long()

    events = labels(event_ids, "dynamic medial connectivity event IDs")
    pins = labels(pin_membership, "dynamic medial connectivity pin membership")
    free = labels(free_anchors, "dynamic medial connectivity free anchors")
    if bool((events < 0).any()):
        raise ValueError("dynamic medial connectivity event IDs must be non-negative")
    if bool(((pins < 0) | (pins > 255)).any()):
        raise ValueError("dynamic medial connectivity pin membership must fit uint8")
    if bool(((free != 0) & (free != 1)).any()):
        raise ValueError("dynamic medial connectivity free anchors must be binary")
    if bool(((pins > 0) & (events == 0)).any()) or bool(
        ((free > 0) & (events == 0)).any()
    ):
        raise ValueError("dynamic medial connectivity metadata escapes its corridor")
    if not 0 < probability_floor < 1:
        raise ValueError(
            "dynamic medial connectivity probability floor must be in (0, 1)"
        )
    if propagation_steps <= 0:
        raise ValueError("dynamic medial connectivity steps must be positive")

    probability = torch.softmax(logits.float(), dim=1)[:, 1]
    floor_margin = math.log(probability_floor / (1.0 - probability_floor))
    event_losses: list[torch.Tensor] = []
    bottlenecks: list[torch.Tensor] = []
    event_count = 0
    target_count = 0
    for sample_index in range(events.shape[0]):
        for identifier in torch.unique(events[sample_index]).tolist():
            if identifier == 0:
                continue
            corridor = events[sample_index] == int(identifier)
            coordinates = torch.nonzero(corridor, as_tuple=False)
            if coordinates.numel() == 0:
                continue
            starts = coordinates.amin(dim=0).tolist()
            stops = (coordinates.amax(dim=0) + 1).tolist()
            crop = tuple(
                slice(int(start), int(stop))
                for start, stop in zip(starts, stops, strict=True)
            )
            corridor_crop = corridor[crop]
            pins_crop = pins[sample_index][crop]
            free_crop = free[sample_index][crop] > 0
            union = 0
            for value in torch.unique(pins_crop[corridor_crop]).tolist():
                union |= int(value)
            bits = tuple(bit for bit in range(8) if union & (1 << bit))
            if len(bits) < 2 or bits != tuple(range(len(bits))):
                raise ValueError(
                    f"dynamic medial connectivity event {identifier} has invalid pins"
                )
            source = (pins_crop & 1) > 0
            if not bool(source.any()):
                raise ValueError(
                    f"dynamic medial connectivity event {identifier} has no source pin"
                )
            event_probability = probability[sample_index][crop]
            free_crop = free_crop | (pins_crop > 0)
            capacity = torch.where(
                free_crop,
                torch.ones_like(event_probability),
                event_probability,
            ) * corridor_crop.to(event_probability.dtype)
            reach = source.to(event_probability.dtype)
            for _ in range(propagation_steps):
                expanded = F.max_pool3d(
                    reach[None, None], kernel_size=3, stride=1, padding=1
                )[0, 0]
                reach = torch.maximum(reach, torch.minimum(expanded, capacity))
                reach = reach * corridor_crop.to(reach.dtype)
            target_scores: list[torch.Tensor] = []
            for bit in bits[1:]:
                target = (pins_crop & (1 << bit)) > 0
                if not bool(target.any()):
                    raise ValueError(
                        f"dynamic medial connectivity event {identifier} lost pin {bit}"
                    )
                target_scores.append(reach[target].max())
            bottleneck = torch.stack(target_scores).min()
            safe_bottleneck = bottleneck.clamp(1.0e-7, 1.0 - 1.0e-7)
            event_losses.append(F.relu(floor_margin - torch.logit(safe_bottleneck)))
            bottlenecks.append(bottleneck)
            event_count += 1
            target_count += len(target_scores)

    zero = logits.sum() * 0.0
    return DynamicMedialConnectivityLossResult(
        loss=torch.stack(event_losses).mean() if event_losses else zero,
        events=logits.new_tensor(float(event_count)),
        targets=logits.new_tensor(float(target_count)),
        mean_bottleneck_probability=(
            torch.stack(bottlenecks).mean() if bottlenecks else zero
        ),
    )


def dice_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    teacher_q: torch.Tensor | None = None,
    target_valid: torch.Tensor | None = None,
    teacher_crest: torch.Tensor | None = None,
    teacher_crest_valid: torch.Tensor | None = None,
    teacher_crest_available: torch.Tensor | None = None,
    m7_anchor_logits: torch.Tensor | None = None,
    pinned_medial_bridge: torch.Tensor | None = None,
    dynamic_connectivity_event: torch.Tensor | None = None,
    dynamic_connectivity_pins: torch.Tensor | None = None,
    dynamic_connectivity_free: torch.Tensor | None = None,
    options: VoxelLossOptions = DEFAULT_VOXEL_LOSS_OPTIONS,
    smooth: float = 1.0e-5,
) -> LossResult:
    """Foreground soft Dice + soft CE over known voxels, averaged per sample.

    The released nnU-Net objective averages Dice per sample but reduces CE over
    every known voxel in the physical batch.  That makes a dense human-label
    patch outweigh a native-teacher patch with one known fine chunk by roughly
    its valid-voxel ratio.  Here both terms give every sample containing known
    supervision one vote; unknown-only samples contribute neither loss.
    """

    options.validate()
    if logits.ndim != 5 or logits.shape[1] != 2:
        raise ValueError(f"expected Bx2xDxHxW logits, got {tuple(logits.shape)}")
    if target.ndim == 5 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 4 or target.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError(
            f"target shape {tuple(target.shape)} does not match {tuple(logits.shape)}"
        )
    target = target.long()
    if teacher_q is None:
        truth = (target == 1).to(logits.dtype)
        valid = target != IGNORE_LABEL
    else:
        if teacher_q.ndim == 5 and teacher_q.shape[1] == 1:
            teacher_q = teacher_q[:, 0]
        if teacher_q.ndim != 4 or teacher_q.shape != target.shape:
            raise ValueError(
                f"teacher_q shape {tuple(teacher_q.shape)} does not match "
                f"target {tuple(target.shape)}"
            )
        truth = teacher_q.to(dtype=logits.dtype)
        if not bool(torch.isfinite(truth).all()) or bool(
            ((truth < 0.0) | (truth > 1.0)).any()
        ):
            raise ValueError("teacher_q must be finite and in [0, 1]")
        if target_valid is None:
            valid = target != IGNORE_LABEL
        else:
            if target_valid.ndim == 5 and target_valid.shape[1] == 1:
                target_valid = target_valid[:, 0]
            if target_valid.ndim != 4 or target_valid.shape != target.shape:
                raise ValueError(
                    f"target_valid shape {tuple(target_valid.shape)} does not "
                    f"match target {tuple(target.shape)}"
                )
            valid = target_valid > 0.5
    spatial = tuple(range(1, target.ndim))
    valid_counts = valid.sum(dim=spatial)
    valid_samples = valid_counts > 0
    log_probability = F.log_softmax(logits, dim=1)
    if bool(valid_samples.any()):
        voxel_cross_entropy = -(
            (1.0 - truth) * log_probability[:, 0] + truth * log_probability[:, 1]
        )
        per_sample_cross_entropy = (
            voxel_cross_entropy * valid.to(voxel_cross_entropy.dtype)
        ).sum(dim=spatial) / valid_counts.clamp_min(1).to(voxel_cross_entropy.dtype)
        cross_entropy = per_sample_cross_entropy[valid_samples].mean()
    else:
        cross_entropy = logits.sum() * 0.0

    probability = torch.softmax(logits, dim=1)[:, 1]
    truth = truth.to(probability.dtype)
    mask = valid.to(probability.dtype)
    intersection = (probability * truth * mask).sum(dim=spatial)
    denominator = (probability * mask).sum(dim=spatial) + (truth * mask).sum(
        dim=spatial
    )
    dice_score = (2.0 * intersection + smooth) / (denominator + smooth)
    dice = (
        1.0 - dice_score[valid_samples].mean()
        if bool(valid_samples.any())
        else logits.sum() * 0.0
    )
    crest = torch.zeros_like(valid)
    crest_valid = torch.zeros_like(valid)
    crest_available = torch.zeros(
        target.shape[0], dtype=torch.bool, device=target.device
    )
    if teacher_crest is not None:
        if teacher_crest.ndim == 5 and teacher_crest.shape[1] == 1:
            teacher_crest = teacher_crest[:, 0]
        if teacher_crest.ndim != 4 or teacher_crest.shape != target.shape:
            raise ValueError(
                f"teacher_crest shape {tuple(teacher_crest.shape)} does not match "
                f"target {tuple(target.shape)}"
            )
        if not bool(torch.isfinite(teacher_crest).all()) or bool(
            ((teacher_crest < 0.0) | (teacher_crest > 1.0)).any()
        ):
            raise ValueError("teacher_crest must be finite and in [0, 1]")
        if teacher_crest_valid is None:
            raise ValueError("teacher_crest requires its own voxel-valid mask")
        if teacher_crest_valid.ndim == 5 and teacher_crest_valid.shape[1] == 1:
            teacher_crest_valid = teacher_crest_valid[:, 0]
        if teacher_crest_valid.ndim != 4 or teacher_crest_valid.shape != target.shape:
            raise ValueError(
                "teacher_crest_valid shape must match the segmentation target"
            )
        crest_valid = teacher_crest_valid > 0.5
        crest = (teacher_crest > 0.5) & crest_valid
        if teacher_crest_available is None:
            crest_available = torch.ones_like(crest_available)
        else:
            available = teacher_crest_available
            if available.ndim > 1:
                available = available.reshape(available.shape[0], -1)
                if available.shape[1] != 1:
                    raise ValueError(
                        "teacher_crest_available must have one value per sample"
                    )
                available = available[:, 0]
            if available.ndim != 1 or available.shape[0] != target.shape[0]:
                raise ValueError(
                    "teacher_crest_available must have one value per sample"
                )
            crest_available = available > 0.5
        crest_valid &= crest_available[:, None, None, None]
        crest &= crest_available[:, None, None, None]
    elif teacher_crest_valid is not None or (
        teacher_crest_available is not None
        and bool((teacher_crest_available > 0.5).any())
    ):
        raise ValueError("crest availability was declared without a crest target")

    medial_recall = logits.sum() * 0.0
    crest_samples = crest.flatten(1).any(dim=1)
    if options.medial_recall_weight > 0 and bool(crest_samples.any()):
        crest_float = crest.to(probability.dtype)
        crest_count = crest_float.sum(dim=spatial)
        recall = (
            (probability * crest_float).sum(dim=spatial) + VILLA_MEDIAL_RECALL_SMOOTH
        ) / (crest_count + VILLA_MEDIAL_RECALL_SMOOTH)
        # Villa returns -recall.  Subtracting from one preserves its exact
        # gradient while making the logged component a conventional loss.
        medial_recall = (1.0 - recall)[crest_samples].mean()

    separation = logits.sum() * 0.0
    if options.separation_weight > 0:
        occupancy_positive = (truth >= 0.5) & valid
        # Crest supersedes thick occupancy positives only where medial
        # supervision is actually known. Falling back voxelwise keeps the
        # anti-blob fence armed in sparse-support/volume rims instead of
        # disarming it for the whole sample merely because a crest sidecar is
        # present somewhere in the patch. Shell negatives always require
        # occupancy validity because the q <= ceiling test is otherwise not
        # meaningful; independently valid crests may still seed that shell.
        positive = crest | (occupancy_positive & ~crest_valid)
        separation_domain = valid
        kernel = 2 * options.separation_radius + 1
        near_positive = (
            F.max_pool3d(
                positive[:, None].to(logits.dtype),
                kernel_size=kernel,
                stride=1,
                padding=options.separation_radius,
            )[:, 0]
            > 0.5
        )
        separation_mask = (
            near_positive
            & separation_domain
            & (truth <= options.separation_max_teacher_q)
            & ~positive
        )
        separation = _per_sample_masked_mean(
            -log_probability[:, 0],
            separation_mask,
        )

    anchor_probability: torch.Tensor | None = None
    if options.m7_anchor_weight > 0 or options.m7_preservation_weight > 0:
        if m7_anchor_logits is None:
            raise ValueError("M7 anchor logits are required by the loss options")
        if m7_anchor_logits.shape != logits.shape:
            raise ValueError("M7 anchor logits and student logits differ in shape")
        anchor_probability = torch.softmax(
            m7_anchor_logits.detach().float(),
            dim=1,
        )

    m7_anchor_kl = logits.sum() * 0.0
    if options.m7_anchor_weight > 0:
        assert anchor_probability is not None
        anchor_log_probability = torch.log(anchor_probability.clamp_min(1.0e-7))
        student_log_probability = F.log_softmax(logits.float(), dim=1)
        voxel_kl = (
            anchor_probability * (anchor_log_probability - student_log_probability)
        ).sum(dim=1)
        anchor_mask = ~valid
        if options.m7_anchor_unknown_corridor_radius > 0:
            corridor_seed = ((truth >= 0.5) & valid) | crest
            radius = options.m7_anchor_unknown_corridor_radius
            corridor = (
                F.max_pool3d(
                    corridor_seed[:, None].to(logits.dtype),
                    kernel_size=2 * radius + 1,
                    stride=1,
                    padding=radius,
                )[:, 0]
                > 0.5
            )
            anchor_mask &= ~corridor
        if options.m7_anchor_known_agreement:
            # Crest is a hard sheet-existence positive even where anti-aliased
            # occupancy is sub-threshold.  Calling that voxel an agreeing
            # background would make the KL anchor fight medial supervision.
            teacher_hard = (truth >= 0.5) | crest
            anchor_hard = anchor_probability[:, 1] >= 0.5
            agreement = valid & (teacher_hard == anchor_hard)
            if options.m7_anchor_confident_agreement:
                # Anti-aliased thin sheets carry partial-volume q below the 0.5
                # hard vote; hard-vote "agreement on background" there anchors
                # exactly the voxels the teacher wants grown. Only voxels where
                # the teacher is confident may agree: at or below the
                # separation background ceiling, or at or above the hard vote.
                teacher_confident = (
                    truth <= options.separation_max_teacher_q
                ) | teacher_hard
                agreement = agreement & teacher_confident
            anchor_mask = anchor_mask | agreement
        m7_anchor_kl = _per_sample_masked_mean(voxel_kl, anchor_mask)

    m7_preservation = logits.sum() * 0.0
    if options.m7_preservation_weight > 0:
        assert anchor_probability is not None
        teacher_positive = ((truth >= 0.5) & valid) | crest
        radius = options.m7_preservation_radius
        teacher_near = (
            F.max_pool3d(
                teacher_positive[:, None].to(logits.dtype),
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            )[:, 0]
            > 0.5
        )
        correct_m7 = (
            (anchor_probability[:, 1] >= options.m7_preservation_anchor_threshold)
            & teacher_near
            & valid
        )
        if options.m7_preservation_soft_floor:
            # Preserve M7's calibrated foreground probability, rather than
            # turning every selected low-confidence boundary voxel into a hard
            # positive.  The one-sided mask activates only after the student
            # moves below M7, so this term cannot itself thicken a sheet.
            anchor_log_probability = torch.log(anchor_probability.clamp_min(1.0e-7))
            student_log_probability = F.log_softmax(logits.float(), dim=1)
            voxel_kl = (
                anchor_probability * (anchor_log_probability - student_log_probability)
            ).sum(dim=1)
            student_foreground = torch.softmax(logits.float(), dim=1)[:, 1]
            shrunk = student_foreground < anchor_probability[:, 1]
            m7_preservation = _per_sample_masked_mean(
                voxel_kl,
                correct_m7 & shrunk,
            )
        else:
            m7_preservation = _per_sample_masked_mean(
                -log_probability[:, 1],
                correct_m7,
            )

    pinned_axial = logits.sum() * 0.0
    if options.pinned_axial_weight > 0:
        if pinned_medial_bridge is None:
            raise ValueError("pinned axial loss requires medial bridge IDs")
        pinned_axial = pinned_axial_floor_loss(
            logits,
            pinned_medial_bridge,
            probability_floor=options.pinned_axial_probability_floor,
            bottom_fraction=options.pinned_axial_bottom_fraction,
        ).loss

    dynamic_connectivity = logits.sum() * 0.0
    if options.dynamic_medial_connectivity_weight > 0:
        if (
            dynamic_connectivity_event is None
            or dynamic_connectivity_pins is None
            or dynamic_connectivity_free is None
        ):
            raise ValueError(
                "dynamic medial connectivity loss requires events, pins, and anchors"
            )
        dynamic_connectivity = dynamic_medial_connectivity_loss(
            logits,
            dynamic_connectivity_event,
            dynamic_connectivity_pins,
            dynamic_connectivity_free,
            probability_floor=options.dynamic_medial_connectivity_probability_floor,
            propagation_steps=options.dynamic_medial_connectivity_steps,
        ).loss

    total = (
        options.cross_entropy_weight * cross_entropy
        + options.dice_weight * dice
        + options.medial_recall_weight * medial_recall
        + options.separation_weight * separation
        + options.m7_anchor_weight * m7_anchor_kl
        + options.m7_preservation_weight * m7_preservation
        + options.pinned_axial_weight * pinned_axial
        + options.dynamic_medial_connectivity_weight * dynamic_connectivity
    )
    return LossResult(
        total,
        cross_entropy,
        dice,
        medial_recall,
        separation,
        m7_anchor_kl,
        m7_preservation,
        pinned_axial,
        dynamic_connectivity,
    )


def deep_supervision_weights(output_count: int) -> tuple[float, ...]:
    if output_count <= 0:
        raise ValueError("output_count must be positive")
    weights = [1.0 / (2**index) for index in range(output_count)]
    if output_count > 1:
        weights[-1] = 0.0
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def resize_target(target: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    if target.ndim == 4:
        target = target[:, None]
    if target.ndim != 5 or target.shape[1] != 1:
        raise ValueError("target must be Bx1xDxHxW or BxDxHxW")
    if target.shape[-3:] == shape:
        return target
    return F.interpolate(target.float(), size=shape, mode="nearest").long()


def resize_soft_target(
    teacher_q: torch.Tensor,
    target_valid: torch.Tensor,
    shape: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize soft occupancy with trilinear q and nearest validity."""

    if teacher_q.ndim == 4:
        teacher_q = teacher_q[:, None]
    if target_valid.ndim == 4:
        target_valid = target_valid[:, None]
    if (
        teacher_q.ndim != 5
        or target_valid.ndim != 5
        or teacher_q.shape[1] != 1
        or target_valid.shape[1] != 1
        or teacher_q.shape != target_valid.shape
    ):
        raise ValueError("teacher_q and target_valid must be matching Bx1xDxHxW")
    if teacher_q.shape[-3:] == shape:
        return teacher_q, target_valid
    return (
        F.interpolate(
            teacher_q.float(), size=shape, mode="trilinear", align_corners=False
        ),
        F.interpolate(target_valid.float(), size=shape, mode="nearest"),
    )


def resize_medial_target(
    teacher_crest: torch.Tensor,
    teacher_crest_valid: torch.Tensor,
    shape: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coverage-preserving resize for a sparse binary medial sheet."""

    if teacher_crest.ndim == 4:
        teacher_crest = teacher_crest[:, None]
    if teacher_crest_valid.ndim == 4:
        teacher_crest_valid = teacher_crest_valid[:, None]
    if (
        teacher_crest.ndim != 5
        or teacher_crest_valid.ndim != 5
        or teacher_crest.shape[1] != 1
        or teacher_crest.shape != teacher_crest_valid.shape
    ):
        raise ValueError(
            "teacher_crest and teacher_crest_valid must be matching Bx1xDxHxW"
        )
    if teacher_crest.shape[-3:] == shape:
        return teacher_crest, teacher_crest_valid
    if all(
        output_size <= input_size
        for output_size, input_size in zip(shape, teacher_crest.shape[-3:], strict=True)
    ):
        crest = F.adaptive_max_pool3d(
            teacher_crest.float() * (teacher_crest_valid > 0.5), shape
        )
        # A zero crest target is trustworthy only if the entire pooled cell is
        # covered.  Adaptive average equals one exactly for a binary mask iff
        # all contributing voxels are known.
        valid = (F.adaptive_avg_pool3d(teacher_crest_valid.float(), shape) >= 1.0).to(
            teacher_crest_valid.dtype
        )
        return crest * valid, valid
    return (
        F.interpolate(teacher_crest.float(), size=shape, mode="nearest"),
        F.interpolate(teacher_crest_valid.float(), size=shape, mode="nearest"),
    )


def deep_supervision_loss(
    outputs: list[torch.Tensor] | tuple[torch.Tensor, ...],
    target: torch.Tensor,
    teacher_q: torch.Tensor | None = None,
    target_valid: torch.Tensor | None = None,
    teacher_crest: torch.Tensor | None = None,
    teacher_crest_valid: torch.Tensor | None = None,
    teacher_crest_available: torch.Tensor | None = None,
    m7_anchor_outputs: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
    pinned_medial_bridge: torch.Tensor | None = None,
    dynamic_connectivity_event: torch.Tensor | None = None,
    dynamic_connectivity_pins: torch.Tensor | None = None,
    dynamic_connectivity_free: torch.Tensor | None = None,
    options: VoxelLossOptions = DEFAULT_VOXEL_LOSS_OPTIONS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not outputs:
        raise ValueError("outputs cannot be empty")
    weights = deep_supervision_weights(len(outputs))
    options.validate()
    if m7_anchor_outputs is not None and len(m7_anchor_outputs) != len(outputs):
        raise ValueError("M7 anchor output count differs from student outputs")
    total = outputs[0].sum() * 0.0
    base_options = replace(
        options,
        pinned_axial_weight=0.0,
        dynamic_medial_connectivity_weight=0.0,
    )
    full: LossResult | None = None
    scale_diagnostics: dict[str, torch.Tensor] = {}
    for index, (logits, weight) in enumerate(zip(outputs, weights, strict=True)):
        resized = resize_target(target, tuple(logits.shape[-3:]))
        resized_q = None
        resized_valid = None
        resized_crest = None
        resized_crest_valid = None
        if teacher_q is not None:
            if target_valid is None:
                target_valid = (target != IGNORE_LABEL).to(torch.float32)
            resized_q, resized_valid = resize_soft_target(
                teacher_q, target_valid, tuple(logits.shape[-3:])
            )
        if teacher_crest is not None:
            if teacher_crest_valid is None:
                raise ValueError("teacher_crest requires teacher_crest_valid")
            resized_crest, resized_crest_valid = resize_medial_target(
                teacher_crest,
                teacher_crest_valid,
                tuple(logits.shape[-3:]),
            )
        result = dice_ce_loss(
            logits,
            resized,
            teacher_q=resized_q,
            target_valid=resized_valid,
            teacher_crest=resized_crest,
            teacher_crest_valid=resized_crest_valid,
            teacher_crest_available=teacher_crest_available,
            m7_anchor_logits=(
                None if m7_anchor_outputs is None else m7_anchor_outputs[index]
            ),
            options=base_options,
        )
        if index == 0:
            full = result
        scale_diagnostics[f"medial_recall_ds{index}"] = result.medial_recall.detach()
        scale_diagnostics[f"separation_ds{index}"] = result.separation.detach()
        total = total + result.total * weight
    assert full is not None
    pinned_axial = PinnedAxialLossResult(
        loss=outputs[0].sum() * 0.0,
        groups=outputs[0].new_tensor(0.0),
        target_voxels=outputs[0].new_tensor(0.0),
    )
    if options.pinned_axial_weight > 0:
        if pinned_medial_bridge is None:
            raise ValueError("pinned axial loss requires medial bridge IDs")
        pinned_axial = pinned_axial_floor_loss(
            outputs[0],
            pinned_medial_bridge,
            probability_floor=options.pinned_axial_probability_floor,
            bottom_fraction=options.pinned_axial_bottom_fraction,
        )
        total = total + options.pinned_axial_weight * pinned_axial.loss
    dynamic_connectivity = DynamicMedialConnectivityLossResult(
        loss=outputs[0].sum() * 0.0,
        events=outputs[0].new_tensor(0.0),
        targets=outputs[0].new_tensor(0.0),
        mean_bottleneck_probability=outputs[0].new_tensor(0.0),
    )
    if options.dynamic_medial_connectivity_weight > 0:
        if (
            dynamic_connectivity_event is None
            or dynamic_connectivity_pins is None
            or dynamic_connectivity_free is None
        ):
            raise ValueError(
                "dynamic medial connectivity loss requires events, pins, and anchors"
            )
        dynamic_connectivity = dynamic_medial_connectivity_loss(
            outputs[0],
            dynamic_connectivity_event,
            dynamic_connectivity_pins,
            dynamic_connectivity_free,
            probability_floor=options.dynamic_medial_connectivity_probability_floor,
            propagation_steps=options.dynamic_medial_connectivity_steps,
        )
        total = (
            total
            + options.dynamic_medial_connectivity_weight * dynamic_connectivity.loss
        )
    return total, {
        "total": total.detach(),
        "cross_entropy": full.cross_entropy.detach(),
        "dice_loss": full.dice.detach(),
        "medial_recall_loss": full.medial_recall.detach(),
        "separation_loss": full.separation.detach(),
        "m7_anchor_kl": full.m7_anchor_kl.detach(),
        "m7_preservation_loss": full.m7_preservation.detach(),
        "pinned_axial_loss": pinned_axial.loss.detach(),
        "pinned_axial_groups": pinned_axial.groups.detach(),
        "pinned_axial_target_voxels": pinned_axial.target_voxels.detach(),
        "dynamic_medial_connectivity_loss": dynamic_connectivity.loss.detach(),
        "dynamic_medial_connectivity_events": dynamic_connectivity.events.detach(),
        "dynamic_medial_connectivity_targets": dynamic_connectivity.targets.detach(),
        "dynamic_medial_connectivity_bottleneck": (
            dynamic_connectivity.mean_bottleneck_probability.detach()
        ),
        **scale_diagnostics,
    }


@torch.no_grad()
def segmentation_metrics(
    logits: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    if target.ndim == 5:
        target = target[:, 0]
    valid = target != IGNORE_LABEL
    prediction = torch.argmax(logits, dim=1) == 1
    truth = target == 1
    true_positive = int((prediction & truth & valid).sum())
    false_positive = int((prediction & ~truth & valid).sum())
    false_negative = int((~prediction & truth & valid).sum())
    return {
        "dice": (2.0 * true_positive)
        / max(1, 2 * true_positive + false_positive + false_negative),
        "precision": true_positive / max(1, true_positive + false_positive),
        "recall": true_positive / max(1, true_positive + false_negative),
        "known_voxels": float(valid.sum()),
        "positive_prevalence": float((truth & valid).sum()) / max(1, int(valid.sum())),
    }
