from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

M7_PARAMETER_TRUST_REGION_CONTRACT = "projected-global-relative-l2-to-m7-v1"


@dataclass(frozen=True)
class TrustRegionProjection:
    relative_l2_before: float
    relative_l2_after: float
    projection_scale: float
    active: bool


class M7ParameterTrustRegion:
    """Project a student onto a global relative-L2 ball around released M7."""

    def __init__(
        self,
        student: nn.Module,
        reference: nn.Module,
        *,
        relative_l2_radius: float,
    ) -> None:
        if not math.isfinite(relative_l2_radius) or relative_l2_radius <= 0:
            raise ValueError("M7 trust-region radius must be finite and positive")
        student_parameters = dict(student.named_parameters())
        reference_parameters = dict(reference.named_parameters())
        if student_parameters.keys() != reference_parameters.keys():
            raise ValueError("student and M7 reference parameter names differ")
        self._pairs = tuple(
            (student_parameters[name], reference_parameters[name])
            for name in student_parameters
        )
        if not self._pairs:
            raise ValueError("M7 trust region requires model parameters")
        if any(
            student_parameter.shape != reference_parameter.shape
            for student_parameter, reference_parameter in self._pairs
        ):
            raise ValueError("student and M7 reference parameter shapes differ")
        self.relative_l2_radius = float(relative_l2_radius)
        reference_l2 = self._reference_l2()
        if not math.isfinite(reference_l2) or reference_l2 <= 0:
            raise ValueError("M7 reference parameter norm must be positive")
        self.reference_l2 = reference_l2
        self.absolute_l2_radius = reference_l2 * self.relative_l2_radius

    @torch.no_grad()
    def _reference_l2(self) -> float:
        total: torch.Tensor | None = None
        for _, reference in self._pairs:
            squared = reference.detach().float().square().sum()
            total = squared if total is None else total + squared
        assert total is not None
        return math.sqrt(float(total))

    @torch.no_grad()
    def _delta_l2(self) -> float:
        total: torch.Tensor | None = None
        for student, reference in self._pairs:
            squared = (student.detach().float() - reference.detach().float()).square().sum()
            total = squared if total is None else total + squared
        assert total is not None
        return math.sqrt(float(total))

    @torch.no_grad()
    def measure_relative_l2(self) -> float:
        return self._delta_l2() / self.reference_l2

    @torch.no_grad()
    def project(self) -> TrustRegionProjection:
        delta_l2 = self._delta_l2()
        relative_before = delta_l2 / self.reference_l2
        if delta_l2 <= self.absolute_l2_radius:
            return TrustRegionProjection(
                relative_l2_before=relative_before,
                relative_l2_after=relative_before,
                projection_scale=1.0,
                active=False,
            )
        scale = self.absolute_l2_radius / delta_l2
        for student, reference in self._pairs:
            student.copy_(reference + (student - reference) * scale)
        return TrustRegionProjection(
            relative_l2_before=relative_before,
            relative_l2_after=self.relative_l2_radius,
            projection_scale=scale,
            active=True,
        )
