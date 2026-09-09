"""Patch-level bootstrap for frozen checkpoint audits (CPU only).

The frozen harness reports macro-scroll Dice: the unweighted mean over
validation scrolls of the per-scroll POOLED Dice. This module rebuilds that
statistic from the per-patch sidecar written by ``checkpoint_audit`` and
resamples patches (by default whole ``support_anchor_chunk_zyx`` clusters,
because several PHerc1451 rows share one anchor) to give confidence intervals
for a level and, with identical resample weights, for a PAIRED delta between
two audits scored on the same rows. Bootstraps describe patch sampling only;
they say nothing about seed noise.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

PER_PATCH_SCHEMA = "crossres-voxel-audit-per-patch-v1"
BOOTSTRAP_SCHEMA = "crossres-voxel-audit-bootstrap-v1"
DEFAULT_REPLICATES = 2000
DEFAULT_SEED = 20260908
DEFAULT_CONFIDENCE = 0.95
CLUSTER_KEY = "support_anchor_chunk_zyx"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PerPatchCounts:
    report_path: Path
    checkpoint_sha256: str
    mirror_tta: bool
    patch_id: np.ndarray
    scroll_id: np.ndarray
    cluster_id: np.ndarray
    has_baseline: np.ndarray
    known: np.ndarray
    positive: np.ndarray
    thresholds: np.ndarray
    true_positive: np.ndarray
    false_positive: np.ndarray
    false_negative: np.ndarray

    @property
    def records(self) -> int:
        return int(self.patch_id.size)

    @property
    def scrolls(self) -> tuple[str, ...]:
        return tuple(sorted({str(value) for value in self.scroll_id.tolist()}))

    def threshold_index(self, threshold: float) -> int:
        differences = np.abs(self.thresholds.astype(np.float64) - float(threshold))
        index = int(np.argmin(differences))
        if differences[index] > 1.0e-4:
            raise ValueError(f"audit sweep lacks threshold {threshold}")
        return index

    def cluster_labels(self, *, cluster: bool) -> np.ndarray:
        """Cluster label per patch; missing anchors fall back to the patch id."""
        patches = self.patch_id.astype(str)
        if not cluster:
            return patches
        labels = self.cluster_id.astype(str)
        return np.where(labels == "", patches, labels)

    def reordered(self, order: np.ndarray) -> PerPatchCounts:
        return replace(
            self,
            patch_id=self.patch_id[order],
            scroll_id=self.scroll_id[order],
            cluster_id=self.cluster_id[order],
            has_baseline=self.has_baseline[order],
            known=self.known[order],
            positive=self.positive[order],
            true_positive=self.true_positive[order],
            false_positive=self.false_positive[order],
            false_negative=self.false_negative[order],
        )


def _verify_against_sweep(counts: PerPatchCounts, sweep: dict[str, Any]) -> None:
    points = sweep["points"]
    if len(points) != counts.thresholds.size:
        raise ValueError("per-patch thresholds differ from the sweep")
    if int(counts.known.sum()) != int(points[0]["known_voxels"]):
        raise ValueError("per-patch known voxels differ from the sweep")
    if int(counts.positive.sum()) != int(points[0]["positive_voxels"]):
        raise ValueError("per-patch positive voxels differ from the sweep")
    arrays = {
        "true_positive": counts.true_positive,
        "false_positive": counts.false_positive,
        "false_negative": counts.false_negative,
    }
    for index, point in enumerate(points):
        if abs(float(point["threshold"]) - float(counts.thresholds[index])) > 1.0e-4:
            raise ValueError("per-patch threshold order differs from the sweep")
        for name, array in arrays.items():
            if int(array[:, index].sum()) != int(point[name]):
                raise ValueError(
                    f"per-patch {name} sum differs from the sweep at "
                    f"T={float(point['threshold']):.2f}"
                )
        for scroll, metrics in point["scrolls"].items():
            mask = counts.scroll_id == scroll
            for name, array in arrays.items():
                if int(array[mask, index].sum()) != int(metrics[name]):
                    raise ValueError(
                        f"per-patch {name} sum differs from the sweep for {scroll}"
                    )


def load_per_patch(report_path: str | Path) -> PerPatchCounts:
    path = Path(report_path).expanduser().resolve()
    if path.is_dir():
        path = path / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    block = report.get("per_patch")
    if not isinstance(block, dict) or block.get("schema") != PER_PATCH_SCHEMA:
        raise ValueError(f"{path}: audit report carries no per-patch counts")
    sidecar = path.parent / str(block["file"])
    if not sidecar.is_file():
        raise ValueError(f"{sidecar}: per-patch counts file is missing")
    if _sha256(sidecar) != str(block["sha256"]):
        raise ValueError(f"{sidecar}: per-patch counts file changed")
    with np.load(sidecar, allow_pickle=False) as payload:
        if str(payload["schema"]) != PER_PATCH_SCHEMA:
            raise ValueError(f"{sidecar}: unexpected per-patch schema")
        counts = PerPatchCounts(
            report_path=path,
            checkpoint_sha256=str(report["checkpoint"]["sha256"]),
            mirror_tta=bool(report["options"]["mirror_tta"]),
            patch_id=payload["patch_id"].astype(str),
            scroll_id=payload["scroll_id"].astype(str),
            cluster_id=payload["cluster_id"].astype(str),
            has_baseline=payload["has_baseline"].astype(bool),
            known=payload["known"].astype(np.int64),
            positive=payload["positive"].astype(np.int64),
            thresholds=payload["thresholds"].astype(np.float32),
            true_positive=payload["true_positive"].astype(np.int64),
            false_positive=payload["false_positive"].astype(np.int64),
            false_negative=payload["false_negative"].astype(np.int64),
        )
    if int(block["records"]) != counts.records:
        raise ValueError(f"{sidecar}: record count differs from the report")
    shape = (counts.records, counts.thresholds.size)
    for array in (counts.true_positive, counts.false_positive, counts.false_negative):
        if array.shape != shape:
            raise ValueError(f"{sidecar}: count array shape {array.shape} != {shape}")
    if len(set(counts.patch_id.tolist())) != counts.records:
        raise ValueError(f"{sidecar}: duplicate patch ids")
    _verify_against_sweep(counts, report["sweep"])
    return counts


def resample_weights(
    counts: PerPatchCounts,
    *,
    replicates: int,
    seed: int,
    cluster: bool = True,
) -> tuple[np.ndarray, dict[str, int]]:
    """Multinomial resample weights per patch, scroll by scroll.

    Sampling units are clusters (anchor chunks) within each scroll; every
    patch of a drawn cluster receives that cluster's draw count. The scroll
    order is sorted, so equal inputs and seeds give equal weights.
    """
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, counts.records), dtype=np.int64)
    labels = counts.cluster_labels(cluster=cluster)
    clusters: dict[str, int] = {}
    for scroll in counts.scrolls:
        rows = np.flatnonzero(counts.scroll_id == scroll)
        unique, inverse = np.unique(labels[rows], return_inverse=True)
        size = int(unique.size)
        draws = rng.integers(0, size, size=(replicates, size))
        cluster_weights = np.zeros((replicates, size), dtype=np.int64)
        np.add.at(
            cluster_weights,
            (np.repeat(np.arange(replicates), size), draws.ravel()),
            1,
        )
        weights[:, rows] = cluster_weights[:, inverse]
        clusters[scroll] = size
    return weights, clusters


def weighted_macro_dice(
    counts: PerPatchCounts, weights: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Macro-scroll Dice [replicates, thresholds] for integer patch weights."""
    if weights.ndim != 2 or weights.shape[1] != counts.records:
        raise ValueError("weights must be [replicates, patches]")
    per_scroll: dict[str, np.ndarray] = {}
    for scroll in counts.scrolls:
        rows = counts.scroll_id == scroll
        local = weights[:, rows].astype(np.float64)
        true_positive = local @ counts.true_positive[rows].astype(np.float64)
        false_positive = local @ counts.false_positive[rows].astype(np.float64)
        false_negative = local @ counts.false_negative[rows].astype(np.float64)
        denominator = 2.0 * true_positive + false_positive + false_negative
        per_scroll[scroll] = 2.0 * true_positive / np.maximum(1.0, denominator)
    macro = np.mean(np.stack([per_scroll[scroll] for scroll in counts.scrolls]), axis=0)
    return macro, per_scroll


def point_estimate(counts: PerPatchCounts) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ones = np.ones((1, counts.records), dtype=np.int64)
    macro, per_scroll = weighted_macro_dice(counts, ones)
    return macro[0], {scroll: value[0] for scroll, value in per_scroll.items()}


def _interval(values: np.ndarray, confidence: float) -> dict[str, float]:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    tail = 100.0 * (1.0 - confidence) / 2.0
    low, high = np.percentile(values, [tail, 100.0 - tail])
    return {
        "low": float(low),
        "high": float(high),
        "half_width": float((high - low) / 2.0),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "mean": float(np.mean(values)),
    }


def bootstrap_level(
    counts: PerPatchCounts,
    threshold: float,
    *,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
    cluster: bool = True,
) -> dict[str, Any]:
    index = counts.threshold_index(threshold)
    point_macro, point_scrolls = point_estimate(counts)
    weights, clusters = resample_weights(
        counts, replicates=replicates, seed=seed, cluster=cluster
    )
    macro, per_scroll = weighted_macro_dice(counts, weights)
    fixed = macro[:, index]
    reselected = macro.max(axis=1)
    return {
        "schema": BOOTSTRAP_SCHEMA,
        "kind": "level",
        "report": str(counts.report_path),
        "checkpoint_sha256": counts.checkpoint_sha256,
        "mirror_tta": counts.mirror_tta,
        "threshold": float(counts.thresholds[index]),
        "replicates": int(replicates),
        "seed": int(seed),
        "confidence": float(confidence),
        "cluster_mode": CLUSTER_KEY if cluster else "patch",
        "clusters_per_scroll": clusters,
        "patches_per_scroll": {
            scroll: int(np.count_nonzero(counts.scroll_id == scroll))
            for scroll in counts.scrolls
        },
        "point": {
            "macro_scroll_dice": float(point_macro[index]),
            "scrolls": {
                scroll: float(value[index]) for scroll, value in point_scrolls.items()
            },
        },
        "interval": _interval(fixed, confidence),
        "scroll_intervals": {
            scroll: _interval(value[:, index], confidence)
            for scroll, value in per_scroll.items()
        },
        "calibrated_optimism": {
            "note": "best-threshold macro re-selected inside each replicate "
            "minus the fixed-threshold macro; the in-sample calibrated value "
            "inherits this optimism",
            "point_calibrated_macro": float(point_macro.max()),
            "point_calibrated_threshold": float(
                counts.thresholds[int(np.argmax(point_macro))]
            ),
            "replicate_mean_reselected": float(np.mean(reselected)),
            "replicate_mean_fixed": float(np.mean(fixed)),
            "optimism": float(np.mean(reselected - fixed)),
        },
    }


def _aligned(a: PerPatchCounts, b: PerPatchCounts) -> tuple[PerPatchCounts, PerPatchCounts]:
    order_a = np.argsort(a.patch_id, kind="stable")
    order_b = np.argsort(b.patch_id, kind="stable")
    a_sorted = a.reordered(order_a)
    b_sorted = b.reordered(order_b)
    if a_sorted.records != b_sorted.records or not np.array_equal(
        a_sorted.patch_id, b_sorted.patch_id
    ):
        raise ValueError("paired bootstrap needs identical patch id sets")
    for name in ("scroll_id", "cluster_id"):
        if not np.array_equal(getattr(a_sorted, name), getattr(b_sorted, name)):
            raise ValueError(f"paired bootstrap needs identical {name} per patch")
    for name in ("known", "positive"):
        if not np.array_equal(getattr(a_sorted, name), getattr(b_sorted, name)):
            raise ValueError(
                f"paired bootstrap needs identical {name} voxels per patch "
                "(same rows and labels)"
            )
    return a_sorted, b_sorted


def bootstrap_paired_delta(
    candidate: PerPatchCounts,
    reference: PerPatchCounts,
    *,
    threshold: float,
    reference_threshold: float | None = None,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
    cluster: bool = True,
) -> dict[str, Any]:
    """Paired (identical resample) CI for candidate minus reference macro Dice."""
    a, b = _aligned(candidate, reference)
    index_a = a.threshold_index(threshold)
    index_b = b.threshold_index(
        threshold if reference_threshold is None else reference_threshold
    )
    weights, clusters = resample_weights(
        a, replicates=replicates, seed=seed, cluster=cluster
    )
    macro_a, scrolls_a = weighted_macro_dice(a, weights)
    macro_b, scrolls_b = weighted_macro_dice(b, weights)
    delta = macro_a[:, index_a] - macro_b[:, index_b]
    point_a, point_scrolls_a = point_estimate(a)
    point_b, point_scrolls_b = point_estimate(b)
    scroll_deltas = {
        scroll: scrolls_a[scroll][:, index_a] - scrolls_b[scroll][:, index_b]
        for scroll in a.scrolls
    }
    return {
        "schema": BOOTSTRAP_SCHEMA,
        "kind": "paired-delta",
        "candidate": {
            "report": str(a.report_path),
            "checkpoint_sha256": a.checkpoint_sha256,
            "mirror_tta": a.mirror_tta,
            "threshold": float(a.thresholds[index_a]),
            "macro_scroll_dice": float(point_a[index_a]),
        },
        "reference": {
            "report": str(b.report_path),
            "checkpoint_sha256": b.checkpoint_sha256,
            "mirror_tta": b.mirror_tta,
            "threshold": float(b.thresholds[index_b]),
            "macro_scroll_dice": float(point_b[index_b]),
        },
        "pairing": "identical cluster resample weights for both audits"
        + (
            "; operating-point-paired (different thresholds)"
            if index_a != index_b or not np.isclose(
                a.thresholds[index_a], b.thresholds[index_b]
            )
            else ""
        ),
        "replicates": int(replicates),
        "seed": int(seed),
        "confidence": float(confidence),
        "cluster_mode": CLUSTER_KEY if cluster else "patch",
        "clusters_per_scroll": clusters,
        "delta": {
            "point": float(point_a[index_a] - point_b[index_b]),
            **_interval(delta, confidence),
            "probability_delta_le_zero": float(np.mean(delta <= 0.0)),
        },
        "scroll_deltas": {
            scroll: {
                "point": float(
                    point_scrolls_a[scroll][index_a] - point_scrolls_b[scroll][index_b]
                ),
                **_interval(values, confidence),
            }
            for scroll, values in scroll_deltas.items()
        },
    }
