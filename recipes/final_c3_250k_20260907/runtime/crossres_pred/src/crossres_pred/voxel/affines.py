from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations
from typing import Any

import numpy as np


def homogeneous_affine(value: object, context: str) -> np.ndarray:
    """Validate a metadata 3x4 affine and return a homogeneous 4x4 matrix."""

    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(not isinstance(row, list) or len(row) != 4 for row in value)
    ):
        raise ValueError(f"{context} transformation matrix is not 3x4")
    result = np.eye(4, dtype=np.float64)
    result[:3, :] = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{context} transformation matrix is not finite")
    if abs(float(np.linalg.det(result[:3, :3]))) < 1.0e-12:
        raise ValueError(f"{context} transformation matrix is singular")
    return result


def _coordinate_equivalence_key(volume: dict[str, Any]) -> tuple[Any, ...] | None:
    """Identify re-windowed exports that retain an identical scan voxel grid."""

    scan_id = str(volume.get("scan_id") or "")
    properties = volume.get("properties") or {}
    shape = properties.get("shape")
    pixel_size = properties.get("pixel_size_um")
    if (
        not scan_id
        or not isinstance(shape, list)
        or len(shape) != 3
        or pixel_size is None
    ):
        return None
    return (
        scan_id,
        tuple(int(value) for value in shape),
        round(float(pixel_size), 9),
        properties.get("left_handed_coordinates"),
        properties.get("z_direction_is_top_to_bottom"),
        properties.get("data_format"),
    )


def find_volume_affine(
    volumes: dict[str, Any],
    source_id: str,
    target_id: str,
) -> np.ndarray:
    """Compose an official volume transform graph in either direction.

    Metadata records only one direction for some valid pairs (for example,
    coarse-to-fine for PHerc0343P and PHerc0009B). Every nonsingular edge is
    therefore made bidirectional before the shortest deterministic path is
    composed. Re-windowed exports of the same scan with identical shape, voxel
    size, handedness, direction, and data format are identity-linked because
    intensity remapping does not change their voxel coordinates.
    """

    if source_id not in volumes:
        raise KeyError(f"unknown source volume {source_id}")
    if target_id not in volumes:
        raise KeyError(f"unknown target volume {target_id}")
    if source_id == target_id:
        return np.eye(4, dtype=np.float64)

    edges: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    equivalent: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for current_id, volume in sorted(volumes.items()):
        equivalence_key = _coordinate_equivalence_key(volume)
        if equivalence_key is not None:
            equivalent[equivalence_key].append(current_id)
        transforms = (volume.get("properties") or {}).get("transforms") or []
        for transform in transforms:
            next_id = str(transform.get("to_volume_id") or "")
            if not next_id or next_id not in volumes:
                continue
            direct = homogeneous_affine(
                transform.get("transformation_matrix"),
                f"{current_id}->{next_id}",
            )
            edges[current_id].append((next_id, direct))
            edges[next_id].append((current_id, np.linalg.inv(direct)))
    identity = np.eye(4, dtype=np.float64)
    for volume_ids in equivalent.values():
        for left, right in combinations(sorted(volume_ids), 2):
            edges[left].append((right, identity))
            edges[right].append((left, identity))

    queue: deque[tuple[str, np.ndarray]] = deque(
        [(source_id, np.eye(4, dtype=np.float64))]
    )
    visited = {source_id}
    while queue:
        current_id, source_to_current = queue.popleft()
        for next_id, current_to_next in sorted(
            edges.get(current_id, ()), key=lambda item: item[0]
        ):
            if next_id in visited:
                continue
            source_to_next = current_to_next @ source_to_current
            if next_id == target_id:
                return source_to_next
            visited.add(next_id)
            queue.append((next_id, source_to_next))
    raise ValueError(
        f"no transform chain from volume {source_id} to volume {target_id}"
    )
