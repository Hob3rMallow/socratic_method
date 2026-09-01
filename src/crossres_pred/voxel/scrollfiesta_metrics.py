from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

# Read-only parity contract with ScrollFiesta's pre-Step-0 prediction gate:
#   src/extract/pred_reject.c
#   src/common/pipeline_constants.h
# calibrated on 2026-06-03. crossres_pred intentionally does not import, edit,
# or invoke ScrollFiesta at runtime.
SCROLLFIESTA_PRED_METRICS_SCHEMA = "crossres-scrollfiesta-pred-metrics-v1"
SCROLLFIESTA_PRED_METRICS_CONTRACT = "scrollfiesta-pred-reject-2026-06-03-v1"
SCROLLFIESTA_DEPLOY_CUBE = 128

GARBAGE_ERODE_R = 2
GARBAGE_ERODE_MAXPASS = 16
GARBAGE_INTERIOR_FRAC = 0.50
GARBAGE_INTERIOR_MIN = 2_000
GARBAGE_RECT_AREA_FRAC = 0.10
GARBAGE_RECT_FILL = 0.75
GARBAGE_RECT_FRAC = 0.20
GARBAGE_RECT_RUN = 12
MIN_CC_SIZE = 500

_STRUCTURE_6 = ndimage.generate_binary_structure(3, 1)
_REASONS = {
    "empty": "reject: empty (no meshable component)",
    "solid-slab": "reject: solid slab",
    "keep-thin": "keep: thin (low erosion interior)",
    "keep-nonrectangular": "keep: not a persistent rectangle",
}


@dataclass(frozen=True)
class ScrollFiestaPredMetrics:
    window_origin_zyx: tuple[int, int, int]
    window_shape_zyx: tuple[int, int, int]
    foreground_voxels: int
    largest_component_voxels: int
    fill_fraction: float
    interior_voxels: int
    interior_fraction: float
    max_thickness: int
    rectangle_axis: int
    rectangle_fraction: float
    max_rectangle_run: int
    reject_kind: str
    reason: str

    @property
    def rejected(self) -> bool:
        return self.reject_kind != "keep"

    @property
    def reject_priority(self) -> int:
        return {"keep": 0, "empty": 1, "solid-slab": 2}[self.reject_kind]

    def validate(self) -> None:
        if (
            len(self.window_origin_zyx) != 3
            or len(self.window_shape_zyx) != 3
            or any(item < 0 for item in self.window_origin_zyx)
            or any(item <= 0 for item in self.window_shape_zyx)
        ):
            raise ValueError("ScrollFiesta metric window must be a positive 3-D crop")
        voxels = math.prod(self.window_shape_zyx)
        if not 0 <= self.foreground_voxels <= voxels:
            raise ValueError("ScrollFiesta foreground count is outside its window")
        if not 0 <= self.largest_component_voxels <= self.foreground_voxels:
            raise ValueError("ScrollFiesta largest-component count is invalid")
        if not 0 <= self.interior_voxels <= self.foreground_voxels:
            raise ValueError("ScrollFiesta erosion-interior count is invalid")
        if self.max_thickness < 0 or not 0 <= self.rectangle_axis <= 2:
            raise ValueError("ScrollFiesta thickness or rectangle axis is invalid")
        for name, value in (
            ("fill_fraction", self.fill_fraction),
            ("interior_fraction", self.interior_fraction),
            ("rectangle_fraction", self.rectangle_fraction),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"ScrollFiesta {name} must be finite in [0, 1]")
        expected_fill = self.foreground_voxels / voxels
        expected_interior = self.interior_voxels / max(1, self.foreground_voxels)
        if not math.isclose(self.fill_fraction, expected_fill, abs_tol=1.0e-12):
            raise ValueError("ScrollFiesta fill fraction differs from voxel count")
        if not math.isclose(
            self.interior_fraction,
            expected_interior,
            abs_tol=1.0e-12,
        ):
            raise ValueError("ScrollFiesta interior fraction differs from voxel count")
        if self.reject_kind not in {"keep", "empty", "solid-slab"}:
            raise ValueError("ScrollFiesta reject kind is invalid")
        valid_reasons = (
            {_REASONS["empty"]}
            if self.reject_kind == "empty"
            else {_REASONS["solid-slab"]}
            if self.reject_kind == "solid-slab"
            else {_REASONS["keep-thin"], _REASONS["keep-nonrectangular"]}
        )
        if self.reason not in valid_reasons:
            raise ValueError("ScrollFiesta reason disagrees with reject kind")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCROLLFIESTA_PRED_METRICS_SCHEMA,
            "contract": SCROLLFIESTA_PRED_METRICS_CONTRACT,
            "window_origin_zyx": list(self.window_origin_zyx),
            "window_shape_zyx": list(self.window_shape_zyx),
            "foreground_voxels": self.foreground_voxels,
            "largest_component_voxels": self.largest_component_voxels,
            "fill_fraction": self.fill_fraction,
            "interior_voxels": self.interior_voxels,
            "interior_fraction": self.interior_fraction,
            "max_thickness": self.max_thickness,
            "rectangle_axis": self.rectangle_axis,
            "rectangle_fraction": self.rectangle_fraction,
            "max_rectangle_run": self.max_rectangle_run,
            "rejected": self.rejected,
            "reject_kind": self.reject_kind,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ScrollFiestaPredMetrics:
        if value.get("schema") != SCROLLFIESTA_PRED_METRICS_SCHEMA:
            raise ValueError("unsupported ScrollFiesta prediction-metric schema")
        if value.get("contract") != SCROLLFIESTA_PRED_METRICS_CONTRACT:
            raise ValueError("unsupported ScrollFiesta prediction-metric contract")
        try:
            result = cls(
                window_origin_zyx=tuple(
                    int(item) for item in value["window_origin_zyx"]
                ),
                window_shape_zyx=tuple(
                    int(item) for item in value["window_shape_zyx"]
                ),
                foreground_voxels=int(value["foreground_voxels"]),
                largest_component_voxels=int(
                    value["largest_component_voxels"]
                ),
                fill_fraction=float(value["fill_fraction"]),
                interior_voxels=int(value["interior_voxels"]),
                interior_fraction=float(value["interior_fraction"]),
                max_thickness=int(value["max_thickness"]),
                rectangle_axis=int(value["rectangle_axis"]),
                rectangle_fraction=float(value["rectangle_fraction"]),
                max_rectangle_run=int(value["max_rectangle_run"]),
                reject_kind=str(value["reject_kind"]),
                reason=str(value["reason"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid ScrollFiesta prediction metrics") from error
        result.validate()
        if bool(value.get("rejected")) != result.rejected:
            raise ValueError("ScrollFiesta rejected flag disagrees with reject kind")
        return result


def _largest_component_voxels(mask: np.ndarray) -> int:
    labels, count = ndimage.label(mask, structure=_STRUCTURE_6)
    if count == 0:
        return 0
    sizes = np.bincount(labels.reshape(-1))
    return int(sizes[1:].max(initial=0))


def _slice_structure(axis: int) -> np.ndarray:
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1, 1, 1] = True
    for dimension in range(3):
        if dimension == axis:
            continue
        lower = [1, 1, 1]
        upper = [1, 1, 1]
        lower[dimension] = 0
        upper[dimension] = 2
        structure[tuple(lower)] = True
        structure[tuple(upper)] = True
    return structure


def _axis_rectangle_metrics(mask: np.ndarray, axis: int) -> tuple[float, int]:
    labels, _ = ndimage.label(mask, structure=_slice_structure(axis))
    rectangle_frames = 0
    run = 0
    maximum_run = 0
    for index in range(mask.shape[axis]):
        label_slice = np.take(labels, index, axis=axis)
        sizes = np.bincount(label_slice.reshape(-1))
        if sizes.size <= 1:
            area = 0
            bounding_fill = 0.0
        else:
            sizes[0] = 0
            component = int(np.argmax(sizes))
            area = int(sizes[component])
            coordinates = np.argwhere(label_slice == component)
            extent = np.ptp(coordinates, axis=0) + 1
            bounding_fill = area / int(np.prod(extent))
        area_fraction = area / label_slice.size
        is_rectangle = (
            area_fraction >= GARBAGE_RECT_AREA_FRAC
            and bounding_fill >= GARBAGE_RECT_FILL
        )
        if is_rectangle:
            rectangle_frames += 1
            run += 1
            maximum_run = max(maximum_run, run)
        else:
            run = 0
    return rectangle_frames / mask.shape[axis], maximum_run


def scrollfiesta_pred_metrics(
    volume: np.ndarray,
    *,
    window_origin_zyx: tuple[int, int, int] = (0, 0, 0),
) -> ScrollFiestaPredMetrics:
    """Evaluate ScrollFiesta's canonical per-cube prediction measurements."""

    raw = np.asarray(volume)
    if raw.ndim != 3 or any(size <= 0 for size in raw.shape):
        raise ValueError("ScrollFiesta prediction metrics require a non-empty 3-D array")
    mask = np.ascontiguousarray(raw != 0)
    shape = tuple(int(item) for item in mask.shape)
    foreground = int(np.count_nonzero(mask))
    largest_component = _largest_component_voxels(mask)
    fill_fraction = foreground / mask.size

    if largest_component < MIN_CC_SIZE:
        result = ScrollFiestaPredMetrics(
            window_origin_zyx=window_origin_zyx,
            window_shape_zyx=shape,
            foreground_voxels=foreground,
            largest_component_voxels=largest_component,
            fill_fraction=fill_fraction,
            interior_voxels=0,
            interior_fraction=0.0,
            max_thickness=0,
            rectangle_axis=0,
            rectangle_fraction=0.0,
            max_rectangle_run=0,
            reject_kind="empty",
            reason=_REASONS["empty"],
        )
        result.validate()
        return result

    # Repeated 6-neighbour erosion is exactly L1 distance from background.
    # Padding supplies the out-of-volume background used by the C implementation.
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    distance = ndimage.distance_transform_cdt(padded, metric="taxicab")[
        1:-1, 1:-1, 1:-1
    ]
    interior = int(np.count_nonzero(distance > GARBAGE_ERODE_R))
    interior_fraction = interior / foreground
    max_thickness = min(int(distance.max(initial=0)), GARBAGE_ERODE_MAXPASS + 1)

    rectangle_axis = 0
    rectangle_fraction = 0.0
    max_rectangle_run = -1
    any_rectangle_pass = False
    for axis in range(3):
        fraction, maximum_run = _axis_rectangle_metrics(mask, axis)
        if fraction >= GARBAGE_RECT_FRAC and maximum_run >= GARBAGE_RECT_RUN:
            any_rectangle_pass = True
        if maximum_run > max_rectangle_run:
            rectangle_axis = axis
            rectangle_fraction = fraction
            max_rectangle_run = maximum_run

    thick = (
        interior_fraction >= GARBAGE_INTERIOR_FRAC
        and interior >= GARBAGE_INTERIOR_MIN
    )
    if thick and any_rectangle_pass:
        reject_kind = "solid-slab"
        reason = _REASONS["solid-slab"]
    elif not thick:
        reject_kind = "keep"
        reason = _REASONS["keep-thin"]
    else:
        reject_kind = "keep"
        reason = _REASONS["keep-nonrectangular"]

    result = ScrollFiestaPredMetrics(
        window_origin_zyx=window_origin_zyx,
        window_shape_zyx=shape,
        foreground_voxels=foreground,
        largest_component_voxels=largest_component,
        fill_fraction=fill_fraction,
        interior_voxels=interior,
        interior_fraction=interior_fraction,
        max_thickness=max_thickness,
        rectangle_axis=rectangle_axis,
        rectangle_fraction=rectangle_fraction,
        max_rectangle_run=max_rectangle_run,
        reject_kind=reject_kind,
        reason=reason,
    )
    result.validate()
    return result


def scrollfiesta_patch_pred_metrics(volume: np.ndarray) -> ScrollFiestaPredMetrics:
    """Measure the centered deploy-sized cube inside a training patch."""

    raw = np.asarray(volume)
    if raw.ndim != 3:
        raise ValueError("ScrollFiesta patch metrics require a 3-D array")
    shape = np.asarray(raw.shape, dtype=np.int64)
    window_shape = np.minimum(shape, SCROLLFIESTA_DEPLOY_CUBE)
    origin = (shape - window_shape) // 2
    slices = tuple(
        slice(int(lower), int(lower + extent))
        for lower, extent in zip(origin, window_shape, strict=True)
    )
    return scrollfiesta_pred_metrics(
        raw[slices],
        window_origin_zyx=tuple(int(item) for item in origin),
    )


def scrollfiesta_metrics_close(
    left: ScrollFiestaPredMetrics,
    right: ScrollFiestaPredMetrics,
) -> bool:
    left_value = left.to_dict()
    right_value = right.to_dict()
    for key in ("fill_fraction", "interior_fraction", "rectangle_fraction"):
        if not math.isclose(
            float(left_value.pop(key)),
            float(right_value.pop(key)),
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        ):
            return False
    return left_value == right_value
