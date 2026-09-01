from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SurfaceObjective:
    """Per-voxel partitioned surface objective.

    Every voxel contributes to at most one term, so no two supervision
    sources ever push the same logit in opposite directions (the recorded
    v4.3 shared-scalar contradiction):

    - P1 (label != 2): supervised BCE + Dice against rasterized ground truth.
      Ground truth always overrides the teacher.
    - P2 (label == 2 and distill_valid): BCE against the soft resampled
      teacher probability, weight ``distill_weight``.
    - P3 (label == 2, not distill_valid, rehearsal_valid): BCE against the
      m7 baseline band, weight ``rehearsal_weight`` -- a weak anti-forgetting
      prior, never a gate.
    """

    dice_weight: float = 0.5
    distill_weight: float = 1.0
    rehearsal_weight: float = 0.25

    def validate(self) -> None:
        for name in ("dice_weight", "distill_weight", "rehearsal_weight"):
            value = getattr(self, name)
            if not value >= 0.0:
                raise ValueError(f"{name} must be non-negative")

    @classmethod
    def from_dict(cls, value: Any) -> SurfaceObjective:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise TypeError("surface objective must be an object")
        objective = cls(**value)
        objective.validate()
        return objective

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def masked_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return masked_mean(loss, mask)


def masked_soft_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """1 - Dice over the masked region; 0 when the mask is empty."""

    if mask.sum() < 1.0:
        return logits.sum() * 0.0
    probability = torch.sigmoid(logits) * mask
    masked_target = target * mask
    intersection = (probability * masked_target).sum()
    denominator = probability.sum() + masked_target.sum()
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice


def partition_masks(
    label: torch.Tensor,
    distill_valid: torch.Tensor,
    rehearsal_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exclusive P1/P2/P3 voxel partition.

    ``label`` holds {0, 1, 2} (2 = ignore/unknown); validity masks are 0/1
    fields. The returned float masks are provably disjoint.
    """

    known = label < 1.5
    distill = distill_valid > 0.5
    rehearsal = rehearsal_valid > 0.5
    p1 = known
    p2 = (~known) & distill
    p3 = (~known) & (~distill) & rehearsal
    return p1.float(), p2.float(), p3.float()


def compute_surface_losses(
    logits: torch.Tensor,
    batch: dict[str, Any],
    objective: SurfaceObjective,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the partitioned objective for one batch.

    ``batch`` must contain ``label`` ({0,1,2}), ``distill`` (soft target in
    [0,1]), ``distill_valid``, ``rehearsal`` (target in [0,1]), and
    ``rehearsal_valid`` -- all shaped like ``logits``.
    """

    label = batch["label"].float()
    p1, p2, p3 = partition_masks(
        label, batch["distill_valid"].float(), batch["rehearsal_valid"].float()
    )
    supervised_target = (label > 0.5).float() * p1

    supervised_bce = masked_bce(logits, supervised_target, p1)
    supervised_dice = masked_soft_dice(logits, supervised_target, p1)
    distill_bce = masked_bce(logits, batch["distill"].float(), p2)
    rehearsal_bce = masked_bce(logits, batch["rehearsal"].float(), p3)

    total = (
        supervised_bce
        + objective.dice_weight * supervised_dice
        + objective.distill_weight * distill_bce
        + objective.rehearsal_weight * rehearsal_bce
    )
    components = {
        "total": total,
        "supervised_bce": supervised_bce,
        "supervised_dice": supervised_dice,
        "distill_bce": distill_bce,
        "rehearsal_bce": rehearsal_bce,
        "p1_voxels": p1.sum(),
        "p2_voxels": p2.sum(),
        "p3_voxels": p3.sum(),
    }
    return total, components
