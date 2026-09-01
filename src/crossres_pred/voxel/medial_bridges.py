from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
from scipy import ndimage

from .growth_sentinels import (
    STRUCTURE_8,
    SliceScreenOptions,
    cleaned_slice_masks,
    screen_connectivity_slice,
)

PINNED_MEDIAL_BRIDGE_CONTRACT = (
    "teacher-medial-minimum-off-axis-mst-between-m7-components-v1"
)
PINNED_MEDIAL_BRIDGE_ATLAS_SCHEMA = "crossres-training-pinned-medial-bridge-atlas-v2"
DYNAMIC_MEDIAL_CONNECTIVITY_CONTRACT = (
    "teacher-medial-dynamic-widest-path-connectivity-v1"
)
DYNAMIC_MEDIAL_CONNECTIVITY_ATLAS_SCHEMA = (
    "crossres-training-dynamic-medial-connectivity-atlas-v2"
)


@dataclass(frozen=True)
class PinnedMedialBridgeOptions:
    screen: SliceScreenOptions = field(default_factory=SliceScreenOptions)
    maximum_corridor_dilation: int = 3

    def validate(self) -> None:
        if self.maximum_corridor_dilation < 0:
            raise ValueError("maximum corridor dilation must be non-negative")


@dataclass(frozen=True)
class PinnedMedialBridgeResult:
    qualified: bool
    rejection: str | None
    route: np.ndarray
    supervision: np.ndarray
    corridor: np.ndarray
    pin_membership: np.ndarray
    free_anchors: np.ndarray
    corridor_dilation: int | None
    reference_ids: tuple[int, ...]
    route_cost: int | None
    screen: dict[str, Any]


def _empty_result(
    shape: tuple[int, int],
    *,
    rejection: str,
    screen: dict[str, Any],
    reference_ids: tuple[int, ...] = (),
) -> PinnedMedialBridgeResult:
    empty = np.zeros(shape, dtype=bool)
    return PinnedMedialBridgeResult(
        qualified=False,
        rejection=rejection,
        route=empty,
        supervision=empty.copy(),
        corridor=empty.copy(),
        pin_membership=np.zeros(shape, dtype=np.uint8),
        free_anchors=empty.copy(),
        corridor_dilation=None,
        reference_ids=reference_ids,
        route_cost=None,
        screen=screen,
    )


def _minimum_off_axis_path(
    corridor: np.ndarray,
    centers: np.ndarray,
    start: np.ndarray,
    target: np.ndarray,
) -> tuple[int | None, np.ndarray]:
    """Find a deterministic path that prefers medial centers before length.

    One off-center step costs more than the longest possible simple path in the
    window.  This makes the optimization lexicographic: first minimize total
    distance from the teacher medial set, then choose the shortest route.
    """

    allowed = np.asarray(corridor, dtype=bool)
    center_mask = np.asarray(centers, dtype=bool) & allowed
    source = np.asarray(start, dtype=bool) & allowed
    destination = np.asarray(target, dtype=bool) & allowed
    if not (
        allowed.ndim == 2
        and center_mask.shape == allowed.shape
        and source.shape == allowed.shape
        and destination.shape == allowed.shape
    ):
        raise ValueError("corridor, centers, start, and target must match in 2-D")
    path = np.zeros(allowed.shape, dtype=bool)
    if not bool(source.any()) or not bool(destination.any()):
        return None, path

    distance_from_center = ndimage.distance_transform_cdt(
        ~center_mask,
        metric="chessboard",
    ).astype(np.int64, copy=False)
    off_axis_multiplier = int(allowed.size) + 1
    node_cost = 1 + off_axis_multiplier * distance_from_center
    infinity = np.iinfo(np.int64).max
    best = np.full(allowed.shape, infinity, dtype=np.int64)
    parent_y = np.full(allowed.shape, -1, dtype=np.int32)
    parent_x = np.full(allowed.shape, -1, dtype=np.int32)
    queue: list[tuple[int, int, int]] = []
    for y, x in np.argwhere(source):
        best[y, x] = 0
        heapq.heappush(queue, (0, int(y), int(x)))

    end: tuple[int, int] | None = None
    height, width = allowed.shape
    while queue:
        cost, y, x = heapq.heappop(queue)
        if cost != int(best[y, x]):
            continue
        if destination[y, x]:
            end = (y, x)
            break
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if not (0 <= ny < height and 0 <= nx < width and allowed[ny, nx]):
                    continue
                candidate = cost + int(node_cost[ny, nx])
                if candidate < int(best[ny, nx]):
                    best[ny, nx] = candidate
                    parent_y[ny, nx] = y
                    parent_x[ny, nx] = x
                    heapq.heappush(queue, (candidate, ny, nx))

    if end is None:
        return None, path
    y, x = end
    while True:
        path[y, x] = True
        py, px = int(parent_y[y, x]), int(parent_x[y, x])
        if py < 0 or px < 0:
            break
        y, x = py, px
    return int(best[end]), path


def _minimum_spanning_routes(
    reference_ids: tuple[int, ...],
    contacts: dict[int, np.ndarray],
    corridor: np.ndarray,
    centers: np.ndarray,
) -> tuple[int | None, np.ndarray]:
    edges: list[tuple[int, int, int, np.ndarray]] = []
    for left, right in combinations(reference_ids, 2):
        cost, path = _minimum_off_axis_path(
            corridor,
            centers,
            contacts[left],
            contacts[right],
        )
        if cost is not None:
            edges.append((cost, left, right, path))
    edges.sort(key=lambda value: (value[0], value[1], value[2]))

    parent = {reference_id: reference_id for reference_id in reference_ids}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    route = np.zeros(corridor.shape, dtype=bool)
    total_cost = 0
    selected_edges = 0
    for cost, left, right, path in edges:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            continue
        parent[right_root] = left_root
        route |= path
        total_cost += cost
        selected_edges += 1
        if selected_edges == len(reference_ids) - 1:
            break
    if selected_edges != len(reference_ids) - 1:
        return None, np.zeros(corridor.shape, dtype=bool)
    return total_cost, route


def extract_pinned_medial_bridge(
    *,
    m7: np.ndarray,
    teacher: np.ndarray,
    centers: np.ndarray,
    valid: np.ndarray,
    teacher_confidence: np.ndarray | None = None,
    options: PinnedMedialBridgeOptions | None = None,
) -> PinnedMedialBridgeResult:
    """Extract one fixed, thin teacher route connecting fragmented M7.

    M7 components are the pins.  The teacher medial set supplies the permitted
    longitudinal geometry.  The result never depends on a student prediction,
    which prevents an already-weak student route from moving its own target.
    Only route voxels outside a one-voxel M7 neighborhood are supervised.
    """

    opts = options or PinnedMedialBridgeOptions()
    opts.validate()
    arrays = [np.asarray(value) for value in (m7, teacher, centers, valid)]
    if (
        any(value.ndim != 2 for value in arrays)
        or len({value.shape for value in arrays}) != 1
    ):
        raise ValueError("m7, teacher, centers, and valid must match in 2-D")
    shape = arrays[0].shape
    domain = arrays[3].astype(bool, copy=False)
    center_mask = arrays[2].astype(bool, copy=False) & domain
    screen = screen_connectivity_slice(
        m7=arrays[0],
        teacher=arrays[1],
        valid=domain,
        teacher_confidence=teacher_confidence,
        options=opts.screen,
    )
    if not bool(screen["qualified"]):
        return _empty_result(
            shape,
            rejection=str(screen["rejection"]),
            screen=screen,
        )

    m7_clean, teacher_clean = cleaned_slice_masks(
        arrays[0],
        arrays[1],
        domain,
        minimum_component_voxels=opts.screen.minimum_component_voxels,
    )
    m7_labels, _ = ndimage.label(m7_clean, structure=STRUCTURE_8)
    teacher_labels, _ = ndimage.label(teacher_clean, structure=STRUCTURE_8)
    event = screen["join_event"]
    if not isinstance(event, dict):
        raise TypeError("qualified screen has no join event")
    teacher_component = teacher_labels == int(event["teacher_component_label"])
    reference_ids = tuple(int(value) for value in event["m7_component_labels"])
    center_component = center_mask & teacher_component
    if not bool(center_component.any()):
        return _empty_result(
            shape,
            rejection="teacher-component-has-no-medial-center",
            screen=screen,
            reference_ids=reference_ids,
        )

    selected_corridor: np.ndarray | None = None
    selected_contacts: dict[int, np.ndarray] | None = None
    selected_dilation: int | None = None
    for dilation in range(opts.maximum_corridor_dilation + 1):
        corridor = center_component.copy()
        if dilation:
            corridor = ndimage.binary_dilation(
                corridor,
                structure=STRUCTURE_8,
                iterations=dilation,
            )
        corridor &= teacher_component
        contacts: dict[int, np.ndarray] = {}
        for reference_id in reference_ids:
            near = ndimage.binary_dilation(
                m7_labels == reference_id,
                structure=STRUCTURE_8,
                iterations=opts.screen.contact_radius,
            )
            contacts[reference_id] = near & corridor
        if not all(bool(contact.any()) for contact in contacts.values()):
            continue
        corridor_labels, _ = ndimage.label(corridor, structure=STRUCTURE_8)
        touched = [
            {int(value) for value in np.unique(corridor_labels[contact]) if value}
            for contact in contacts.values()
        ]
        if touched and set.intersection(*touched):
            selected_corridor = corridor
            selected_contacts = contacts
            selected_dilation = dilation
            break
    if selected_corridor is None or selected_contacts is None:
        return _empty_result(
            shape,
            rejection="teacher-medial-corridor-does-not-connect-m7-pins",
            screen=screen,
            reference_ids=reference_ids,
        )

    route_cost, route = _minimum_spanning_routes(
        reference_ids,
        selected_contacts,
        selected_corridor,
        center_component,
    )
    if route_cost is None:
        return _empty_result(
            shape,
            rejection="teacher-medial-route-search-failed",
            screen=screen,
            reference_ids=reference_ids,
        )
    m7_near = ndimage.binary_dilation(
        m7_clean,
        structure=STRUCTURE_8,
        iterations=1,
    )
    if len(reference_ids) > 8:
        raise ValueError("dynamic medial connectivity supports at most eight pins")
    pin_membership = np.zeros(shape, dtype=np.uint8)
    for bit_index, reference_id in enumerate(reference_ids):
        pin_membership[selected_contacts[reference_id]] |= np.uint8(1 << bit_index)
    free_anchors = (m7_near & selected_corridor) | (pin_membership > 0)
    supervision = route & ~m7_near
    if int(np.count_nonzero(supervision)) < opts.screen.minimum_missing_join_voxels:
        return _empty_result(
            shape,
            rejection="fixed-route-has-no-novel-medial-span",
            screen=screen,
            reference_ids=reference_ids,
        )
    return PinnedMedialBridgeResult(
        qualified=True,
        rejection=None,
        route=route,
        supervision=supervision,
        corridor=selected_corridor,
        pin_membership=pin_membership,
        free_anchors=free_anchors,
        corridor_dilation=selected_dilation,
        reference_ids=reference_ids,
        route_cost=route_cost,
        screen=screen,
    )
