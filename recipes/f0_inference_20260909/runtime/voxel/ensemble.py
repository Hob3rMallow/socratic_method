"""Diagnostic mean-probability ensemble behind the single-model inference surface.

The frozen audit calls ``model.full_resolution_logits`` and takes the softmax
of the result. Returning the logarithm of the mean member probability makes
that softmax reproduce the mean probability exactly, so eight-way mirror TTA
and every downstream consumer compose unchanged. Ensembles are diagnostic
only: the shipped inference contract is one student network.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import torch
from torch import nn

from .model import VoxelNNUNet

ENSEMBLE_CONTRACT = "mean-probability-ensemble-v1"
_PROBABILITY_FLOOR = 1.0e-12


class ProbabilityEnsemble(nn.Module):
    def __init__(self, members: Sequence[VoxelNNUNet]) -> None:
        super().__init__()
        if not members:
            raise ValueError("an ensemble needs at least one member")
        configs = {
            json.dumps(member.config.as_dict(), sort_keys=True) for member in members
        }
        if len(configs) != 1:
            raise ValueError("ensemble members must share one model_config")
        self.members = nn.ModuleList(list(members))
        self.config = members[0].config
        self.contract = ENSEMBLE_CONTRACT

    def mean_probability(self, image: torch.Tensor) -> torch.Tensor:
        total: torch.Tensor | None = None
        for member in self.members:
            probability = torch.softmax(
                member.full_resolution_logits(image).float(), dim=1
            )
            total = probability if total is None else total + probability
        assert total is not None
        return total / len(self.members)

    def full_resolution_logits(self, image: torch.Tensor) -> torch.Tensor:
        return torch.log(self.mean_probability(image).clamp_min(_PROBABILITY_FLOOR))

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        return [self.full_resolution_logits(image)]
