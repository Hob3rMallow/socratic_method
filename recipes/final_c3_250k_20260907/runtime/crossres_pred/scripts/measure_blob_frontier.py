"""CPU-only, provenance-bound fixed-view blob frontier; never trains a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import diagnose_release_t025_blobs as frozen
import numpy as np
from render_release_milestone_panels import FIXED_SLICES

from crossres_pred.voxel.blob_diagnostics import (
    aggregate_slice_blob_metrics,
    slice_blob_metrics,
)

THRESHOLDS = (0.25, 0.30, 0.35, 0.40)
SCHEMA = "crossres-fixed18-blob-frontier-v1"


def summarize_planes(
    probabilities: dict[str, np.ndarray],
    references: dict[str, np.ndarray],
    *,
    thresholds: tuple[float, ...] = THRESHOLDS,
    named: tuple[str, ...] = frozen.WATCHLIST,
) -> dict[str, Any]:
    if set(probabilities) != set(references) or not probabilities:
        raise ValueError("candidate and reference views differ or are empty")
    if not set(named).issubset(probabilities):
        raise ValueError("named views missing from the fixed view set")
    if tuple(sorted(set(thresholds))) != thresholds or any(
        not 0 < t < 1 for t in thresholds
    ):
        raise ValueError("thresholds must be sorted, unique, and inside (0, 1)")
    for spec, field in probabilities.items():
        if field.shape != references[spec].shape or field.ndim != 2:
            raise ValueError("candidate and reference plane shapes differ")
        if not np.isfinite(field).all() or np.any((field < 0) | (field > 1)):
            raise ValueError("invalid probability field")
    result = {}
    for threshold in thresholds:
        rows = []
        for spec, field in probabilities.items():
            row = slice_blob_metrics(field >= threshold, references[spec] != 0)
            row["spec"] = spec
            rows.append(row)
        result[f"{threshold:.2f}"] = {
            "fixed18": aggregate_slice_blob_metrics(rows),
            "named3": aggregate_slice_blob_metrics(
                [row for row in rows if row["spec"] in named]
            ),
            "views": rows,
        }
    return result


def validate_provenance(provenance: dict[str, Any], checkpoint_sha: str | None) -> None:
    for key, expected in (
        ("mirror_tta", True),
        ("halo", 32),
        ("amp_dtype", "bfloat16"),
        ("chunk_size", 128),
        ("context_shape_zyx", [192, 192, 192]),
    ):
        if provenance.get(key) != expected:
            raise ValueError(f"inference provenance mismatch: {key}")
    if provenance.get("skipped_incomplete_context_ids"):
        raise ValueError("incomplete-context export is not a fixed-cube comparison")
    actual = provenance.get("checkpoint_sha256")
    if not actual or actual != provenance.get("checkpoint", {}).get("sha256"):
        raise ValueError("inference checkpoint identity is inconsistent")
    if checkpoint_sha and actual != checkpoint_sha:
        raise ValueError("inference checkpoint SHA256 differs from the selected model")


def measure(inference: Path, *, checkpoint_sha: str | None = None) -> dict[str, Any]:
    provenance_path = inference / "provenance.json"
    provenance = frozen._read(provenance_path)
    validate_provenance(provenance, checkpoint_sha)
    views = [frozen._parse_view(spec) for spec in FIXED_SLICES]
    probabilities, references = {}, {}
    inputs = {str(provenance_path): frozen._sha256(provenance_path)}
    for cube_id in sorted({view["cube_id"] for view in views}):
        field = frozen._read_volume(inference, "probability", cube_id)
        reference = frozen._read_volume(frozen.JACKPOT, "cubes_PRED", cube_id)
        for path in (
            inference / "probability" / f"{cube_id}.tif",
            frozen.JACKPOT / "cubes_PRED" / f"{cube_id}.tif",
        ):
            inputs[str(path)] = frozen._sha256(path)
        for view in views:
            if view["cube_id"] == cube_id:
                spec = view["specification"]
                probabilities[spec] = frozen._plane(field, view).copy()
                references[spec] = frozen._plane(reference, view).copy()
    # Restore the frozen display order after grouping reads by cube.
    probabilities = {spec: probabilities[spec] for spec in FIXED_SLICES}
    return {
        "schema": SCHEMA,
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "inference": str(inference),
        "reference": str(frozen.JACKPOT),
        "reference_threshold": 0.35,
        "fixed_views": list(FIXED_SLICES),
        "named_views": list(frozen.WATCHLIST),
        "input_sha256": inputs,
        "thresholds": summarize_planes(probabilities, references),
        "interpretation": "Model-reference agreement, not human-ground-truth accuracy. No threshold tuning or checkpoint selection on these views.",
    }


def interpolated_bridge_limit(
    reference_frontier: dict[str, Any], coverage: float, *, maximum_growth: float = 1.20
) -> dict[str, Any]:
    points = sorted(
        (
            float(row["fixed18"]["reference_skeleton_recall"]),
            float(row["fixed18"]["candidate_only_bridge_pixels"]),
        )
        for row in reference_frontier["thresholds"].values()
    )
    if not np.isfinite(coverage) or not points[0][0] <= coverage <= points[-1][0]:
        return {
            "resolved": False,
            "reason": "outside measured coverage overlap; no extrapolation",
        }
    # Duplicate coverages use the lowest observed bridge count conservatively.
    unique = {}
    for x, y in points:
        unique[x] = min(unique.get(x, y), y)
    xs = sorted(unique)
    interpolated = float(np.interp(coverage, xs, [unique[x] for x in xs]))
    return {
        "resolved": True,
        "coverage": coverage,
        "reference_bridges_interpolated": interpolated,
        "maximum_candidate_bridges": maximum_growth * interpolated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    args = parser.parse_args()
    result = measure(args.inference, checkpoint_sha=args.checkpoint_sha256)
    if args.output.exists():
        if frozen._read(args.output) != result:
            raise ValueError("refusing to overwrite a different frontier receipt")
    else:
        frozen._atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fixed_t035": result["thresholds"]["0.35"]["fixed18"],
            }
        )
    )


if __name__ == "__main__":
    main()
