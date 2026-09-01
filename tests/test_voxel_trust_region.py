from __future__ import annotations

import pytest
import torch
from crossres_pred.voxel.trust_region import (
    M7_PARAMETER_TRUST_REGION_CONTRACT,
    M7ParameterTrustRegion,
)


def _linear(weight: tuple[float, float]) -> torch.nn.Linear:
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([weight]))
    return model


def test_projection_caps_global_relative_l2_departure() -> None:
    reference = _linear((3.0, 4.0))
    student = _linear((3.6, 4.8))
    region = M7ParameterTrustRegion(
        student,
        reference,
        relative_l2_radius=0.1,
    )

    result = region.project()

    assert M7_PARAMETER_TRUST_REGION_CONTRACT == (
        "projected-global-relative-l2-to-m7-v1"
    )
    assert result.active is True
    assert result.relative_l2_before == pytest.approx(0.2)
    assert result.relative_l2_after == pytest.approx(0.1)
    assert result.projection_scale == pytest.approx(0.5)
    assert region.measure_relative_l2() == pytest.approx(0.1)
    assert student.weight.detach().flatten().tolist() == pytest.approx([3.3, 4.4])


def test_projection_is_exact_noop_inside_region() -> None:
    reference = _linear((3.0, 4.0))
    student = _linear((3.03, 4.04))
    before = student.weight.detach().clone()
    region = M7ParameterTrustRegion(
        student,
        reference,
        relative_l2_radius=0.1,
    )

    result = region.project()

    assert result.active is False
    assert result.projection_scale == 1.0
    assert torch.equal(student.weight, before)


@pytest.mark.parametrize("radius", [0.0, -0.1, float("nan"), float("inf")])
def test_projection_rejects_invalid_radius(radius: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        M7ParameterTrustRegion(
            _linear((3.0, 4.0)),
            _linear((3.0, 4.0)),
            relative_l2_radius=radius,
        )
