"""Probability-supported skeleton-corridor proposals for sheet gap recovery.

This module proposes curved connections; it does not globally lower a mask's
threshold. Anatomical validity is not implied by a path through model evidence.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

EIGHT = np.ones((3, 3), bool)
STEPS = [(y, x, math.hypot(y, x)) for y in (-1, 0, 1) for x in (-1, 0, 1) if y or x]


@dataclass(frozen=True)
class ProposalPolicy:
    probability_floor: float = 0.15
    maximum_length: float = 96
    boundary_margin: int = 8
    maximum_region_nodes: int = 6000
    maximum_anchors: int = 24
    minimum_component_size: int = 12


@dataclass(frozen=True)
class AcceptancePolicy:
    minimum_mean_probability: float = 0.25
    minimum_axis_alignment: float = 0.70
    minimum_axis_anisotropy: float = 2
    maximum_arc_ratio: float = 1.5
    minimum_ridge_fraction: float = 0.60
    minimum_gap_pixels: int = 3
    minimum_neighbor_probability_coverage: float = 0.80
    minimum_supported_neighbor_planes: int = 2


def local_axis(
    labels: np.ndarray, point: tuple[int, int], owner: int, radius: int = 8
) -> dict:
    y, x = point
    y0, x0 = max(0, y - radius), max(0, x - radius)
    window = labels[y0 : y + radius + 1, x0 : x + radius + 1] == owner
    points = np.argwhere(window) + [y0, x0]
    if len(points) < 5:
        return {"axis": [0.0, 0.0], "anisotropy": 0.0, "outward": [0.0, 0.0]}
    cov = np.cov(points.T)
    values, vectors = np.linalg.eigh(cov)
    axis = vectors[:, -1]
    outward = np.array(point) - points.mean(axis=0)
    outward /= max(1e-8, np.linalg.norm(outward))
    return {
        "axis": axis.tolist(),
        "anisotropy": float(values[-1] / max(0.25, values[0])),
        "outward": outward.tolist(),
    }


def route_features(
    path: np.ndarray,
    probability: np.ndarray,
    labels: np.ndarray,
    umbilicus_yx: tuple[float, float],
) -> dict:
    a, b = tuple(path[0]), tuple(path[-1])
    owners = [int(labels[a]), int(labels[b])]
    p = probability[tuple(path.T)].astype(float)
    missing = labels[tuple(path.T)] == 0
    gap_p = p[missing]
    jump = np.diff(path, axis=0)
    length = float(np.linalg.norm(jump, axis=1).sum())
    chord = float(np.linalg.norm(path[-1] - path[0]))
    steps = min(5, len(path) - 1)
    departures = [path[steps] - path[0], path[-steps - 1] - path[-1]]
    orientations = [
        local_axis(labels, point, owner) for point, owner in zip((a, b), owners)
    ]
    alignment, outward = [], []
    for direction, orientation in zip(departures, orientations):
        direction = direction / max(1e-8, np.linalg.norm(direction))
        alignment.append(abs(float(np.dot(direction, orientation["axis"]))))
        outward.append(float(np.dot(direction, orientation["outward"])))
    smoothed = ndimage.gaussian_filter1d(
        path.astype(float), 1.2, axis=0, mode="nearest"
    )
    tangent = np.gradient(smoothed, axis=0)
    tangent /= np.maximum(1e-8, np.linalg.norm(tangent, axis=1))[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    lateral = []
    for offset in (-4, 4):
        coords = (path + offset * normal).T
        lateral.append(
            ndimage.map_coordinates(
                probability.astype(np.float32), coords, order=1, mode="nearest"
            )
        )
    ridge = p - np.maximum(lateral[0], lateral[1])
    radius = np.linalg.norm(path - np.array(umbilicus_yx), axis=1)
    return {
        "owners": owners,
        "length_px": length,
        "chord_px": chord,
        "arc_ratio": length / max(1.0, chord),
        "gap_pixels": int(missing.sum()),
        "probability_mean": float(gap_p.mean()),
        "probability_min": float(gap_p.min()),
        "probability_p10": float(np.quantile(gap_p, 0.1)),
        "endpoint_alignment": alignment,
        "endpoint_outward": outward,
        "endpoint_anisotropy": [v["anisotropy"] for v in orientations],
        "radial_endpoint_difference_px": float(abs(radius[-1] - radius[0])),
        "radial_extent_px": float(np.ptp(radius)),
        "probability_ridge_margin_mean": float(ridge[missing].mean()),
        "probability_ridge_positive_fraction": float((ridge[missing] > 0).mean()),
    }


def find_corridors(
    mask: np.ndarray,
    probability: np.ndarray,
    umbilicus_yx: tuple[float, float],
    policy: ProposalPolicy = ProposalPolicy(),
) -> tuple[list[dict], dict]:
    if mask.ndim != 2 or probability.shape != mask.shape:
        raise ValueError("matching 2D mask and probability arrays required")
    if (
        not np.isfinite(probability).all()
        or probability.min() < 0
        or probability.max() > 1
    ):
        raise ValueError("invalid probability field")
    if not 0 < policy.probability_floor < 0.35 or policy.maximum_length <= 0:
        raise ValueError("invalid corridor policy")
    mask = mask.astype(bool)
    labels, _ = ndimage.label(mask, EIGHT)
    sizes = np.bincount(labels.ravel())
    skeleton = skeletonize((probability >= policy.probability_floor) | mask)
    missing_labels, count = ndimage.label(skeleton & ~mask, EIGHT)
    regions = ndimage.find_objects(missing_labels)
    pairs = {}
    census = defaultdict(int)
    census["missing_skeleton_regions"] = count
    for number, bounds in enumerate(regions, 1):
        if bounds is None:
            continue
        start = np.array([s.start for s in bounds])
        points = np.argwhere(missing_labels[bounds] == number) + start
        if len(points) > policy.maximum_region_nodes:
            census["oversized_regions"] += 1
            continue
        nodes = {tuple(p) for p in points}
        anchors = defaultdict(set)
        for y, x in nodes:
            for dy, dx, _ in STEPS:
                q = (y + dy, x + dx)
                if (
                    0 <= q[0] < mask.shape[0]
                    and 0 <= q[1] < mask.shape[1]
                    and skeleton[q]
                    and labels[q]
                ):
                    owner = int(labels[q])
                    if sizes[owner] >= policy.minimum_component_size:
                        anchors[owner].add(q)
        if len(anchors) < 2:
            census["fewer_than_two_anchor_components"] += 1
            continue
        if len(anchors) > policy.maximum_anchors:
            census["ambiguous_large_anchor_fanout"] += 1
            continue
        all_nodes = nodes | {point for values in anchors.values() for point in values}
        for owner, seeds in sorted(anchors.items()):
            distance = {q: 0.0 for q in seeds}
            parents = {q: None for q in seeds}
            heap = [(0.0, 0.0, q) for q in sorted(seeds)]
            heapq.heapify(heap)
            reached = set()
            while heap:
                cost, length, point = heapq.heappop(heap)
                if cost != distance[point]:
                    continue
                other = int(labels[point])
                if other and other != owner:
                    if other not in reached:
                        reached.add(other)
                        reverse = [point]
                        while parents[reverse[-1]] is not None:
                            reverse.append(parents[reverse[-1]])
                        path = np.array(reverse[::-1], dtype=int)
                        key = tuple(sorted((owner, other)))
                        if len(path) > 2 and (
                            key not in pairs or cost < pairs[key]["cost"]
                        ):
                            if owner > other:
                                path = path[::-1].copy()
                            pairs[key] = {
                                "path_yx": path.tolist(),
                                "cost": cost,
                                "region": number,
                                "region_anchor_components": len(anchors),
                            }
                    continue
                for dy, dx, step in STEPS:
                    q = (point[0] + dy, point[1] + dx)
                    if q not in all_nodes or length + step > policy.maximum_length:
                        continue
                    p = max(policy.probability_floor, float(probability[q]))
                    new_cost = cost + step * (1 + 0.4 * max(0, 0.35 / p - 1))
                    if new_cost < distance.get(q, float("inf")):
                        distance[q] = new_cost
                        parents[q] = point
                        heapq.heappush(heap, (new_cost, length + step, q))
    result = []
    for row in pairs.values():
        path = np.array(row["path_yx"])
        if (path.min(axis=0) < policy.boundary_margin).any() or (
            path.max(axis=0) >= np.array(mask.shape) - policy.boundary_margin
        ).any():
            census["boundary_routes"] += 1
            continue
        row.update(route_features(path, probability, labels, umbilicus_yx))
        result.append(row)
    result.sort(key=lambda r: (-r["length_px"], r["path_yx"][0]))
    census["proposed_corridors"] = len(result)
    return result, dict(census)


def paint_corridor(
    path: np.ndarray, probability: np.ndarray, probability_floor: float
) -> np.ndarray:
    center = np.zeros(probability.shape, bool)
    center[tuple(path.T)] = True
    # Face-connected 3-pixel-wide strokes, clipped to probability support.
    paint = ndimage.binary_dilation(center, ndimage.generate_binary_structure(2, 1))
    return paint & ((probability >= probability_floor * 0.8) | center)


def contact_owners(mask: np.ndarray, paint: np.ndarray) -> list[int]:
    labels, _ = ndimage.label(mask, EIGHT)
    touched = labels[ndimage.binary_dilation(paint & ~mask, EIGHT)]
    return np.unique(touched[touched > 0]).tolist()


def evidence_features(
    path: np.ndarray,
    probability: np.ndarray,
    ct: np.ndarray,
    neighbor_distance: list[np.ndarray],
) -> dict:
    smoothed = ndimage.gaussian_filter1d(
        path.astype(float), 1.2, axis=0, mode="nearest"
    )
    tangent = np.gradient(smoothed, axis=0)
    tangent /= np.maximum(1e-8, np.linalg.norm(tangent, axis=1))[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    tissue = ndimage.gaussian_filter(ct.astype(np.float32), 0.8)
    central = tissue[tuple(path.T)]
    flanks = [
        ndimage.map_coordinates(tissue, (path + d * normal).T, order=1, mode="nearest")
        for d in (-4, 4)
    ]
    difference = central - np.maximum(flanks[0], flanks[1])
    return {
        "ct_ridge_positive_fraction": float((difference > 0).mean()),
        "ct_ridge_margin_mean": float(difference.mean()),
        "neighbor_raw_coverage": [
            float((d[tuple(path.T)] <= 2).mean()) for d in neighbor_distance
        ],
    }


def probability_support(
    path: np.ndarray, neighboring_probability: list[np.ndarray], floor: float
) -> list[float]:
    # Real nearby fields, not repaired output, provide support; no propagation
    # or iterative self-reinforcement. One-pixel transverse tolerance.
    return [
        float(
            (
                ndimage.maximum_filter(p.astype(np.float32), size=3)[tuple(path.T)]
                >= floor
            ).mean()
        )
        for p in neighboring_probability
    ]


def refusal_reason(
    row: dict, policy: AcceptancePolicy = AcceptancePolicy()
) -> str | None:
    checks = (
        (row["gap_pixels"] >= policy.minimum_gap_pixels, "tiny_gap"),
        (
            min(row["endpoint_alignment"]) >= policy.minimum_axis_alignment,
            "endpoint_direction",
        ),
        (
            min(row["endpoint_anisotropy"]) >= policy.minimum_axis_anisotropy,
            "ambiguous_endpoint_shape",
        ),
        (row["arc_ratio"] <= policy.maximum_arc_ratio, "excessive_detour"),
        (
            row["probability_mean"] >= policy.minimum_mean_probability,
            "weak_mean_probability",
        ),
        (
            row["probability_ridge_positive_fraction"] >= policy.minimum_ridge_fraction,
            "off_probability_ridge",
        ),
        (
            sum(
                v >= policy.minimum_neighbor_probability_coverage
                for v in row["neighbor_probability_coverage"]
            )
            >= policy.minimum_supported_neighbor_planes,
            "insufficient_neighbor_probability_support",
        ),
    )
    return next((reason for accepted, reason in checks if not accepted), None)


def select_and_paint(
    mask: np.ndarray,
    probability: np.ndarray,
    proposals: list[dict],
    floor: float,
    policy: AcceptancePolicy = AcceptancePolicy(),
) -> tuple[np.ndarray, list[dict], dict]:
    mask = mask.astype(bool)
    repaired = mask.copy()
    labels, n_labels = ndimage.label(mask, EIGHT)
    parent = list(range(n_labels + 1))
    accepted_paint = np.zeros(mask.shape, bool)
    reasons = defaultdict(int)
    ledger = []

    def representative(owner: int) -> int:
        while parent[owner] != owner:
            owner = parent[owner]
        return owner

    ordered = sorted(
        proposals,
        key=lambda r: (
            -r["probability_mean"],
            -r["probability_ridge_positive_fraction"],
            r["length_px"],
            r["path_yx"][0],
        ),
    )
    for proposal in ordered:
        row = dict(proposal)
        path = np.array(row["path_yx"], dtype=int)
        reason = refusal_reason(row, policy)
        if reason is None:
            paint = paint_corridor(path, probability, floor)
            added = paint & ~mask
            neighborhood = ndimage.binary_dilation(added, EIGHT)
            touched = np.unique(labels[neighborhood])
            contacts = touched[touched > 0].tolist()
            row["paint_contact_owners"] = contacts
            if sorted(contacts) != sorted(row["owners"]):
                reason = "third_component_or_missing_anchor"
            elif (neighborhood & accepted_paint).any():
                reason = "crossing_or_shared_gap"
            else:
                a, b = [representative(owner) for owner in row["owners"]]
                if a == b:
                    reason = "already_connected_by_selected_corridors"
                else:
                    trial_labels, _ = ndimage.label(
                        repaired | paint, ndimage.generate_binary_structure(2, 1)
                    )
                    endpoints = trial_labels[tuple(path[[0, -1]].T)]
                    if endpoints[0] == 0 or endpoints[0] != endpoints[1]:
                        reason = "not_face_connected"
                    else:
                        parent[b] = a
                        repaired |= paint
                        accepted_paint |= added
                        row["added_voxels"] = int(added.sum())
        row["accepted"] = reason is None
        row["reason"] = reason or "accepted"
        reasons[row["reason"]] += 1
        ledger.append(row)
    if (mask & ~repaired).any():
        raise RuntimeError("continuity repair erased source foreground")
    if int((repaired & ~mask).sum()) != sum(
        r.get("added_voxels", 0) for r in ledger if r["accepted"]
    ):
        raise RuntimeError("independent paint count disagrees with corridor ledger")
    return repaired, ledger, dict(reasons)
