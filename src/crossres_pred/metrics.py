from __future__ import annotations

from typing import Any

import torch


def interior_slices(
    shape_zyx: tuple[int, ...], margin: int
) -> tuple[slice, slice, slice]:
    """Central-crop slices implementing the retained-interior discipline.

    Deployment keeps only the central region of each padded block, so every
    quality number must be computed on that same region (the recorded
    192-vs-128 audit bug). A margin larger than the block collapses to the
    full block rather than an empty slice.
    """

    if margin < 0:
        raise ValueError("margin must be non-negative")
    result: list[slice] = []
    for size in shape_zyx[-3:]:
        if 2 * margin >= size:
            result.append(slice(0, size))
        else:
            result.append(slice(margin, size - margin))
    return tuple(result)


class StreamingBinaryMetrics:
    """Histogram-based streaming AP / AUROC / Dice for probability fields.

    Probabilities are quantized into ``bins`` tie groups; AP and AUROC use
    the tie-aware block formulation, so results are exact for quantized
    scores and deterministic across batch orderings.
    """

    def __init__(self, bins: int = 2048) -> None:
        if bins < 2:
            raise ValueError("bins must be >= 2")
        self.bins = bins
        self._positive = torch.zeros(bins, dtype=torch.int64)
        self._negative = torch.zeros(bins, dtype=torch.int64)

    def update(
        self,
        probabilities: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        probability = probabilities.detach().reshape(-1).float()
        positive = target.detach().reshape(-1) > 0.5
        if mask is not None:
            keep = mask.detach().reshape(-1) > 0.5
            probability = probability[keep]
            positive = positive[keep]
        if probability.numel() == 0:
            return
        index = (
            (probability.clamp(0.0, 1.0) * (self.bins - 1)).round().to(torch.int64)
        )
        self._positive += torch.bincount(
            index[positive], minlength=self.bins
        ).cpu()
        self._negative += torch.bincount(
            index[~positive], minlength=self.bins
        ).cpu()

    @property
    def positive_count(self) -> int:
        return int(self._positive.sum())

    @property
    def negative_count(self) -> int:
        return int(self._negative.sum())

    def _descending(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._positive.flip(0).double(), self._negative.flip(0).double()

    def result(self) -> dict[str, Any]:
        total_positive = self.positive_count
        total_negative = self.negative_count
        value: dict[str, Any] = {
            "positive_count": total_positive,
            "negative_count": total_negative,
            "average_precision": None,
            "auroc": None,
            "dice_at_half": None,
            "best_dice": None,
            "best_dice_threshold": None,
            "prevalence": None,
        }
        if total_positive == 0 or total_negative == 0:
            return value
        positive, negative = self._descending()
        true_positive = positive.cumsum(0)
        false_positive = negative.cumsum(0)

        proposals = (true_positive + false_positive).clamp_min(1.0)
        precision = true_positive / proposals
        value["average_precision"] = float(
            (positive * precision).sum() / total_positive
        )

        true_positive_before = true_positive - positive
        value["auroc"] = float(
            (negative * (true_positive_before + positive / 2.0)).sum()
            / (float(total_positive) * float(total_negative))
        )

        false_negative = total_positive - true_positive
        dice = (2.0 * true_positive) / (
            2.0 * true_positive + false_positive + false_negative
        ).clamp_min(1.0)
        # Descending index k keeps bins [bins-1-k, bins-1]; its threshold is
        # the lowest included bin's probability.
        thresholds = torch.arange(self.bins - 1, -1, -1).double() / (self.bins - 1)
        half_index = int((self.bins - 1) - round(0.5 * (self.bins - 1)))
        value["dice_at_half"] = float(dice[half_index])
        best_index = int(dice.argmax())
        value["best_dice"] = float(dice[best_index])
        value["best_dice_threshold"] = float(thresholds[best_index])
        value["prevalence"] = total_positive / float(
            total_positive + total_negative
        )
        return value
