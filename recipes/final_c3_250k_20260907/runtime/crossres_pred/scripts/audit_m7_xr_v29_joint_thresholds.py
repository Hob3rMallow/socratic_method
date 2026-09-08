#!/usr/bin/env python3
"""Joint raw-probability threshold audit for PHerc0139 and PHerc1447.

This is deliberately read-only with respect to inference: it reuses the six
cached PHerc1447 probability cubes and the locked-16 PHerc0139 threshold sweep.
The intent is to test threshold calibration as one isolated model-definition
change before changing training or the medial-connectivity loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from crossres_pred.voxel.scrollfiesta_metrics import scrollfiesta_pred_metrics

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROBABILITY_GRID = (
    ROOT
    / "output/crossres_data/pherc1447_blind_audit/"
    "m7_xr_v29_dynconn_w0p03125_n1024_raw_t025_report_20260831/"
    "inference/probability"
)
DEFAULT_PUBLISHED_GRID = (
    ROOT
    / "output/crossres_data/pherc1447_blind_audit/"
    "six_cube_subset/cubes_PRED"
)
# Anti-blob reference. This was pinned to the v15 grid (1,388,516 six-cube
# foreground voxels) until 2026-09-01, which made the gate a ratchet: each
# generation was measured against the previous, already-shrunken model, and
# ratio_max 1.10 capped every candidate at 39% of the 2026-08-15 all-sources
# student (3,907,334 voxels) that human review has preferred since v11.2.
# The reference is now that student, regenerated bit-exact on 2026-09-01.
# Pass --reference-grid (where available) to restore the v15 comparison.
DEFAULT_REFERENCE_GRID = (
    ROOT
    / "output/crossres_data/pherc1447_blind_audit/"
    "jackpot_all_sources_20260815_t035_report_20260901/inference/cubes_PRED"
)
LEGACY_V15_REFERENCE_GRID = (
    ROOT
    / "output/crossres_data/pherc1447_blind_audit/"
    "m7_xr_v15_relaxed_trust4x_terminal_report/inference/cubes_PRED"
)
DEFAULT_LOCKED_EVALUATION = (
    ROOT
    / "output/crossres_data/"
    "pherc0139_growth_gate_candidate_v28_dynconn_w0p03125_n1024_"
    "raw_strict_20260831/evaluation.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "output/crossres_data/"
    "m7_xr_v29_joint_raw_threshold_audit_20260831/audit.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / max(int(denominator), 1)


def _thresholds(start: float, stop: float, step: float) -> list[float]:
    if not (0.0 < start <= stop < 1.0) or step <= 0.0:
        raise ValueError("threshold range must lie in (0, 1) with positive step")
    count = int(np.floor((stop - start) / step + 0.5)) + 1
    values = [round(start + index * step, 6) for index in range(count)]
    if values[-1] > stop + 1.0e-6:
        values.pop()
    return values


def _counts_at_thresholds(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    if not bool(np.all(np.isfinite(flat))):
        raise ValueError("probability cube contains non-finite values")
    ordered = np.sort(flat)
    return ordered.size - np.searchsorted(ordered, thresholds, side="left")


def _cube_fast_sweep(
    probability_path: Path,
    published_path: Path,
    reference_path: Path,
    thresholds: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    probability = np.asarray(tifffile.imread(probability_path))
    published = np.asarray(tifffile.imread(published_path)) != 0
    reference = np.asarray(tifffile.imread(reference_path)) != 0
    if probability.shape != published.shape or probability.shape != reference.shape:
        raise ValueError(f"shape mismatch for {probability_path.stem}")
    if probability.ndim != 3:
        raise ValueError(f"{probability_path}: expected a 3-D cube")

    student_counts = _counts_at_thresholds(probability, thresholds)
    published_shared = _counts_at_thresholds(probability[published], thresholds)
    reference_shared = _counts_at_thresholds(probability[reference], thresholds)
    published_positive = int(np.count_nonzero(published))
    reference_positive = int(np.count_nonzero(reference))
    rows: list[dict[str, Any]] = []
    for index, threshold in enumerate(thresholds):
        student_positive = int(student_counts[index])
        pub_shared = int(published_shared[index])
        ref_shared = int(reference_shared[index])
        rows.append(
            {
                "threshold": float(threshold),
                "student_positive": student_positive,
                "published_positive": published_positive,
                "published_shared_positive": pub_shared,
                "dice_vs_published": _safe_ratio(
                    2 * pub_shared, student_positive + published_positive
                ),
                "precision_vs_published": _safe_ratio(pub_shared, student_positive),
                "recall_vs_published": _safe_ratio(pub_shared, published_positive),
                "reference_positive": reference_positive,
                "reference_shared_positive": ref_shared,
                "dice_vs_reference": _safe_ratio(
                    2 * ref_shared, student_positive + reference_positive
                ),
                "precision_vs_reference": _safe_ratio(ref_shared, student_positive),
                "recall_vs_reference": _safe_ratio(ref_shared, reference_positive),
                "foreground_ratio_vs_reference": _safe_ratio(
                    student_positive, reference_positive
                ),
                "student_added_vs_reference": student_positive - ref_shared,
                "reference_missing_vs_student": reference_positive - ref_shared,
            }
        )
    distribution = {
        "shape_zyx": [int(value) for value in probability.shape],
        "dtype": str(probability.dtype),
        "minimum": float(np.min(probability)),
        "maximum": float(np.max(probability)),
        "mean": float(np.mean(probability, dtype=np.float64)),
        "quantiles": {
            str(quantile): float(np.quantile(probability, quantile))
            for quantile in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        },
    }
    return rows, distribution


def _aggregate(
    per_cube: dict[str, list[dict[str, Any]]],
    thresholds: list[float],
    locked_by_threshold: dict[float, dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    cube_ids = sorted(per_cube)
    for index, threshold in enumerate(thresholds):
        cubes = [per_cube[cube_id][index] for cube_id in cube_ids]
        student = sum(int(row["student_positive"]) for row in cubes)
        published = sum(int(row["published_positive"]) for row in cubes)
        published_shared = sum(
            int(row["published_shared_positive"]) for row in cubes
        )
        reference = sum(int(row["reference_positive"]) for row in cubes)
        reference_shared = sum(
            int(row["reference_shared_positive"]) for row in cubes
        )
        ratios = [float(row["foreground_ratio_vs_reference"]) for row in cubes]
        row: dict[str, Any] = {
            "threshold": threshold,
            "pherc1447": {
                "student_positive": student,
                "published_positive": published,
                "published_shared_positive": published_shared,
                "dice_vs_published": _safe_ratio(
                    2 * published_shared, student + published
                ),
                "precision_vs_published": _safe_ratio(published_shared, student),
                "recall_vs_published": _safe_ratio(published_shared, published),
                "reference_positive": reference,
                "reference_shared_positive": reference_shared,
                "dice_vs_reference": _safe_ratio(
                    2 * reference_shared, student + reference
                ),
                "precision_vs_reference": _safe_ratio(reference_shared, student),
                "recall_vs_reference": _safe_ratio(reference_shared, reference),
                "foreground_ratio_vs_reference": _safe_ratio(student, reference),
                "maximum_cube_foreground_ratio_vs_reference": max(ratios),
                "minimum_cube_foreground_ratio_vs_reference": min(ratios),
                "student_added_vs_reference": student - reference_shared,
                "reference_missing_vs_student": reference - reference_shared,
                "aggregate_foreground_within_10pct": student <= 1.10 * reference,
                "every_cube_foreground_within_25pct": max(ratios) <= 1.25,
            },
            "per_cube": {
                cube_id: per_cube[cube_id][index] for cube_id in cube_ids
            },
        }
        locked = locked_by_threshold.get(round(threshold, 6))
        if locked is not None:
            row["pherc0139_locked16"] = locked
        result.append(row)
    return result


def _parse_cube_id(cube_id: str) -> tuple[int, int, int]:
    pieces = cube_id.split("_")
    if len(pieces) != 3 or tuple(piece[0] for piece in pieces) != ("z", "y", "x"):
        raise ValueError(f"invalid cube identifier: {cube_id!r}")
    return tuple(int(piece[1:]) for piece in pieces)


def _morphology_audit(
    probability_paths: list[Path],
    published_grid: Path,
    reference_grid: Path,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    by_threshold: dict[float, list[dict[str, Any]]] = {
        threshold: [] for threshold in thresholds
    }
    for probability_path in probability_paths:
        cube_id = probability_path.stem
        origin = _parse_cube_id(cube_id)
        probability = np.asarray(tifffile.imread(probability_path)).astype(
            np.float32
        )
        published = np.asarray(
            tifffile.imread(published_grid / probability_path.name)
        ) != 0
        reference = np.asarray(
            tifffile.imread(reference_grid / probability_path.name)
        ) != 0
        published_metrics = scrollfiesta_pred_metrics(
            published, window_origin_zyx=origin
        )
        reference_metrics = scrollfiesta_pred_metrics(
            reference, window_origin_zyx=origin
        )
        for threshold in thresholds:
            student = probability >= threshold
            student_metrics = scrollfiesta_pred_metrics(
                student, window_origin_zyx=origin
            )
            by_threshold[threshold].append(
                {
                    "cube_id": cube_id,
                    "student": student_metrics.to_dict(),
                    "reference": reference_metrics.to_dict(),
                    "published_m7": published_metrics.to_dict(),
                    "foreground_ratio_vs_reference": _safe_ratio(
                        student_metrics.foreground_voxels,
                        reference_metrics.foreground_voxels,
                    ),
                    "interior_fraction_delta_vs_reference": (
                        student_metrics.interior_fraction
                        - reference_metrics.interior_fraction
                    ),
                    "max_thickness_delta_vs_reference": (
                        student_metrics.max_thickness
                        - reference_metrics.max_thickness
                    ),
                    "interior_regression_over_0p10": (
                        student_metrics.interior_fraction
                        > reference_metrics.interior_fraction + 0.10
                    ),
                    "thickness_regression_over_2vox": (
                        student_metrics.max_thickness
                        > reference_metrics.max_thickness + 2
                    ),
                }
            )

    result: list[dict[str, Any]] = []
    for threshold in thresholds:
        cubes = by_threshold[threshold]
        student_positive = sum(
            int(row["student"]["foreground_voxels"]) for row in cubes
        )
        reference_positive = sum(
            int(row["reference"]["foreground_voxels"]) for row in cubes
        )
        foreground_pass = student_positive <= 1.10 * reference_positive
        interior_pass = all(
            not bool(row["interior_regression_over_0p10"]) for row in cubes
        )
        thickness_pass = all(
            not bool(row["thickness_regression_over_2vox"]) for row in cubes
        )
        result.append(
            {
                "threshold": threshold,
                "aggregate_student_positive": student_positive,
                "aggregate_reference_positive": reference_positive,
                "aggregate_foreground_ratio_vs_reference": _safe_ratio(
                    student_positive, reference_positive
                ),
                "gates": {
                    "contract": "pherc1447-blind-antiblob-regression-v1",
                    "foreground_not_over_incumbent_by_10pct": foreground_pass,
                    "no_cube_interior_over_incumbent_by_0p10": interior_pass,
                    "no_cube_max_thickness_over_incumbent_by_2vox": thickness_pass,
                    "passed": foreground_pass and interior_pass and thickness_pass,
                },
                "interior_regression_cube_count": sum(
                    bool(row["interior_regression_over_0p10"]) for row in cubes
                ),
                "thickness_regression_cube_count": sum(
                    bool(row["thickness_regression_over_2vox"]) for row in cubes
                ),
                "cubes": cubes,
            }
        )
    return result


def _first_row(
    rows: list[dict[str, Any]], predicate: Any
) -> dict[str, Any] | None:
    return next((row for row in rows if predicate(row)), None)


def run(args: argparse.Namespace) -> dict[str, Any]:
    probability_grid = Path(args.probability_grid).resolve()
    published_grid = Path(args.published_grid).resolve()
    reference_grid = Path(args.reference_grid).resolve()
    locked_path = Path(args.locked_evaluation).resolve()
    output = Path(args.output).resolve()
    thresholds = _thresholds(args.start, args.stop, args.step)
    threshold_array = np.asarray(thresholds, dtype=np.float64)
    morphology_thresholds = sorted(
        {round(float(value), 6) for value in args.morphology_thresholds}
    )
    if any(not 0.0 < value < 1.0 for value in morphology_thresholds):
        raise ValueError("morphology thresholds must lie in (0, 1)")

    probability_paths = sorted(probability_grid.glob("*.tif"))
    if len(probability_paths) != 6:
        raise ValueError(
            f"joint audit requires the exact six probability cubes, found "
            f"{len(probability_paths)}"
        )
    per_cube: dict[str, list[dict[str, Any]]] = {}
    distributions: dict[str, dict[str, Any]] = {}
    input_files: list[dict[str, Any]] = []
    for probability_path in probability_paths:
        cube_id = probability_path.stem
        published_path = published_grid / probability_path.name
        reference_path = reference_grid / probability_path.name
        if not published_path.is_file() or not reference_path.is_file():
            raise FileNotFoundError(f"missing comparison cube for {cube_id}")
        rows, distribution = _cube_fast_sweep(
            probability_path,
            published_path,
            reference_path,
            threshold_array,
        )
        per_cube[cube_id] = rows
        distributions[cube_id] = distribution
        for role, path in (
            ("probability", probability_path),
            ("published_m7", published_path),
            ("anti_blob_reference", reference_path),
        ):
            input_files.append(
                {
                    "cube_id": cube_id,
                    "role": role,
                    "path": str(path),
                    "sha256": _sha256(path),
                }
            )

    locked = _read_json(locked_path)
    if len(locked.get("slices", [])) != 16:
        raise ValueError("locked evaluation does not contain exactly 16 slices")
    locked_by_threshold = {
        round(float(row["threshold"]), 6): row
        for row in locked["threshold_sweep"]
    }
    sweep = _aggregate(per_cube, thresholds, locked_by_threshold)
    crossing_125 = _first_row(
        sweep,
        lambda row: row["pherc1447"]["foreground_ratio_vs_reference"] <= 1.25,
    )
    crossing_110 = _first_row(
        sweep,
        lambda row: row["pherc1447"]["aggregate_foreground_within_10pct"],
    )
    every_cube_125 = _first_row(
        sweep,
        lambda row: row["pherc1447"]["every_cube_foreground_within_25pct"],
    )
    joint_candidates = [
        row
        for row in sweep
        if row["pherc1447"]["aggregate_foreground_within_10pct"]
        and row["pherc1447"]["every_cube_foreground_within_25pct"]
        and row["pherc1447"]["recall_vs_reference"] >= 0.90
        and row.get("pherc0139_locked16", {}).get("anti_blob_passes") == 16
    ]
    # With false-negative growth repair available downstream, choose the lowest
    # foreground inflation first, then locked Dice. This ranking is diagnostic;
    # it does not promote a threshold without morphology and visual review.
    ranked = sorted(
        joint_candidates,
        key=lambda row: (
            abs(row["pherc1447"]["foreground_ratio_vs_reference"] - 1.0),
            -float(row["pherc1447"]["recall_vs_reference"]),
            -float(row["pherc0139_locked16"]["mean_dice"]),
        ),
    )
    morphology = _morphology_audit(
        probability_paths,
        published_grid,
        reference_grid,
        morphology_thresholds,
    )
    payload = {
        "schema": "crossres-m7-xr-v29-joint-raw-threshold-audit-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "changes_inference": False,
        "changes_training": False,
        "model_composition": "raw-student-probability-only; no M7 blend",
        "selection_policy": {
            "risk_asymmetry": (
                "prefer undergrowth over foreground inflation because a later "
                "tangential cross-slice linker can repair gaps"
            ),
            "diagnostic_constraints": {
                "pherc1447_aggregate_foreground_ratio_max": 1.10,
                "pherc1447_every_cube_foreground_ratio_max": 1.25,
                "pherc1447_reference_recall_min": 0.90,
                "pherc0139_locked16_anti_blob_passes": 16,
            },
            "promotion_requires": "3-D morphology plus side-by-side human review",
        },
        "inputs": {
            "probability_grid": str(probability_grid),
            "published_grid": str(published_grid),
            "anti_blob_reference_grid": str(reference_grid),
            "locked_evaluation": str(locked_path),
            "locked_evaluation_sha256": _sha256(locked_path),
            "files": input_files,
        },
        "threshold_range": {
            "start": thresholds[0],
            "stop": thresholds[-1],
            "step": args.step,
            "count": len(thresholds),
        },
        "probability_distributions": distributions,
        "crossings": {
            "aggregate_foreground_ratio_at_most_1p25": crossing_125,
            "aggregate_foreground_ratio_at_most_1p10": crossing_110,
            "every_cube_foreground_ratio_at_most_1p25": every_cube_125,
        },
        "ranked_diagnostic_candidates": ranked[:10],
        "pherc1447_morphology": morphology,
        "sweep": sweep,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _compact(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    blind = row["pherc1447"]
    locked = row.get("pherc0139_locked16")
    return {
        "threshold": row["threshold"],
        "pherc1447_foreground_ratio": blind["foreground_ratio_vs_reference"],
        "pherc1447_max_cube_ratio": blind[
            "maximum_cube_foreground_ratio_vs_reference"
        ],
        "pherc1447_reference_recall": blind["recall_vs_reference"],
        "pherc1447_dice_vs_published_m7": blind["dice_vs_published"],
        "locked16": locked,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probability-grid", default=str(DEFAULT_PROBABILITY_GRID))
    parser.add_argument("--published-grid", default=str(DEFAULT_PUBLISHED_GRID))
    parser.add_argument("--reference-grid", default=str(DEFAULT_REFERENCE_GRID))
    parser.add_argument("--locked-evaluation", default=str(DEFAULT_LOCKED_EVALUATION))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start", type=float, default=0.10)
    parser.add_argument("--stop", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument(
        "--morphology-thresholds",
        nargs="+",
        type=float,
        default=(0.25, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args)
    interesting = {
        "crossings": {
            key: _compact(value) for key, value in payload["crossings"].items()
        },
        "ranked_diagnostic_candidates": [
            _compact(row) for row in payload["ranked_diagnostic_candidates"][:5]
        ],
        "pherc1447_morphology": [
            {
                "threshold": row["threshold"],
                "foreground_ratio": row[
                    "aggregate_foreground_ratio_vs_reference"
                ],
                "interior_regression_cubes": row[
                    "interior_regression_cube_count"
                ],
                "thickness_regression_cubes": row[
                    "thickness_regression_cube_count"
                ],
                "gates": row["gates"],
            }
            for row in payload["pherc1447_morphology"]
        ],
        "output": str(Path(args.output).resolve()),
    }
    print(json.dumps(interesting, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
