#!/usr/bin/env python3
"""Diagnose release-arm low-threshold PHerc1447 blobs at fixed T=0.25.

This is a CPU-only, cached-probability audit.  Candidate masks are always
measured at the operator-fixed T=0.25.  Jackpot 08-15 at its recorded T=0.35
is the primary morphology comparator; jackpot at T=0.25 is retained as a
calibration control.  The three operator-named planes are the diagnostic set,
and all eighteen pre-existing PHerc1447 views are a fixed regression guard.
R_full at 15k is the frozen baseline; later arms are evaluated against its
preregistered gate rather than selecting a threshold, view, or checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image

from crossres_pred.voxel.blob_diagnostics import (
    aggregate_slice_blob_metrics,
    slice_blob_metrics,
)
from crossres_pred.voxel.scrollfiesta_metrics import scrollfiesta_pred_metrics

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "output/crossres_data"
RELEASE = DATA / "release_ladder_20260902"
COMPARISON = RELEASE / "comparisons/trust_ball_8x_vs_off_15k"
SOURCE = DATA / "pherc1447_blind_audit/six_cube_subset"
JACKPOT = (
    DATA
    / "pherc1447_blind_audit/"
    "jackpot_all_sources_20260815_t035_report_20260901/inference"
)
BASELINE_OUTPUT = RELEASE / "diagnostics/t025_blob_structure_20260903"

CANDIDATE_THRESHOLD = 0.25
JACKPOT_THRESHOLD = 0.35
MINIMUM_COMPONENT_PIXELS = 16
SKELETON_TOLERANCE_PIXELS = 1.0
BASELINE_ARM_ID = "R_full"
BASELINE_MILESTONE = 15_000
WATCHLIST = (
    "z12032_y03968_x03072:z:12032",
    "z12032_y03968_x03072:z:12043",
    "z12032_y03968_x03072:z:12050",
)
_CUBE = re.compile(r"^z(?P<z>\d+)_y(?P<y>\d+)_x(?P<x>\d+)$")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _parse_view(specification: str) -> dict[str, Any]:
    try:
        cube_id, axis, coordinate_text = specification.split(":")
    except ValueError as error:
        raise ValueError(f"invalid fixed-view specification: {specification}") from error
    match = _CUBE.fullmatch(cube_id)
    if match is None or axis not in {"z", "y", "x"}:
        raise ValueError(f"invalid fixed-view specification: {specification}")
    origin = {key: int(value) for key, value in match.groupdict().items()}
    coordinate = int(coordinate_text)
    local_index = coordinate - origin[axis]
    if not 0 <= local_index < 128:
        raise ValueError(f"fixed view lies outside its cube: {specification}")
    return {
        "specification": specification,
        "cube_id": cube_id,
        "axis": axis,
        "axis_index": {"z": 0, "y": 1, "x": 2}[axis],
        "global_coordinate": coordinate,
        "local_index": local_index,
        "label": f"{cube_id} · {axis}={coordinate}",
    }


def _plane(volume: np.ndarray, view: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        np.take(volume, int(view["local_index"]), axis=int(view["axis_index"]))
    )


def _read_volume(root: Path, role: str, cube_id: str) -> np.ndarray:
    path = root / role / f"{cube_id}.tif"
    value = np.asarray(tifffile.imread(path))
    if value.shape != (128, 128, 128):
        raise ValueError(f"{path}: expected 128-cube, found {value.shape}")
    return value


def _probability_provenance(root: Path) -> dict[str, Any]:
    provenance = _read(root / "provenance.json")
    if provenance.get("mirror_tta") is not True:
        raise ValueError(f"{root}: diagnostic probability must use mirror TTA")
    return provenance


def _selected_grid(arm: Path, milestone: int) -> Path:
    return (
        arm
        / f"ladder_eval/panels/milestone_{milestone:08d}/"
        "pherc1447/inference"
    )


def _effective_objective(arm: Path) -> dict[str, Any]:
    recipe = _read(arm / "ladder_recipe.json")
    base = recipe.get("base")
    if not isinstance(base, dict) or not isinstance(base.get("objective"), dict):
        raise TypeError(f"{arm}: ladder recipe lacks its base objective")
    overrides = recipe.get("objective_overrides", {})
    if not isinstance(overrides, dict):
        raise TypeError(f"{arm}: objective overrides must be an object")
    return {**base["objective"], **overrides}


def _verify_inputs(
    comparison: dict[str, Any], arm: Path, milestone: int
) -> dict[str, Any]:
    if comparison.get("schema") != "crossres-release-trust-ball-ab-review-v1":
        raise ValueError("the trust-ball comparison manifest has an unexpected schema")
    if comparison.get("comparison_contract") != (
        "pure-trust-radius-delta-matched-15000-v1"
    ):
        raise ValueError("the trust-ball comparison is not the matched pure A/B")
    radius = float(comparison["trust_radii"]["with_8x"])
    if not np.isclose(radius / float(comparison["trust_radii"]["shipped"]), 8.0):
        raise ValueError("the selected candidate is not the 8x trust-ball arm")
    decision = comparison.get("binding_decision", {})
    if decision.get("chosen") != "with_8x":
        raise ValueError("the trust-ball comparison lacks the binding 8x choice")
    effective_radius = float(
        _effective_objective(arm).get("m7_trust_region_relative_l2", -1.0)
    )
    if not np.isclose(effective_radius, radius):
        raise ValueError(
            f"{arm.name}: candidate changed the binding 8x trust radius"
        )

    selected = _selected_grid(arm, milestone)
    selected_provenance = _probability_provenance(selected)
    jackpot_provenance = _probability_provenance(JACKPOT)
    selected_checkpoint = selected_provenance.get("checkpoint")
    if not isinstance(selected_checkpoint, dict):
        raise TypeError("selected probability provenance lacks its checkpoint")
    if str(selected_provenance.get("checkpoint_sha256", "")) != str(
        selected_checkpoint.get("sha256", "")
    ):
        raise ValueError("selected checkpoint identity is inconsistent")
    if float(jackpot_provenance.get("threshold", -1)) != JACKPOT_THRESHOLD:
        raise ValueError("jackpot morphology reference threshold changed")
    return {
        "selected": selected_provenance,
        "jackpot": jackpot_provenance,
        "effective_trust_radius": effective_radius,
    }


def _discover_milestones(
    arm: Path, selected_milestone: int
) -> list[tuple[int, Path]]:
    records: list[tuple[int, Path]] = []
    panels = arm / "ladder_eval/panels"
    for directory in sorted(panels.glob("milestone_*")):
        try:
            samples = int(directory.name.rsplit("_", 1)[1])
        except ValueError:
            continue
        inference = directory / "pherc1447/inference"
        if (inference / "provenance.json").is_file():
            records.append((samples, inference))
    if selected_milestone not in {samples for samples, _ in records}:
        raise FileNotFoundError(
            f"{arm.name}: selected {selected_milestone:,} probability grid is missing"
        )
    return records


def _view_metrics(
    probability: np.ndarray,
    jackpot_probability: np.ndarray,
    jackpot_selected_volume: np.ndarray,
    m7: np.ndarray,
    view: dict[str, Any],
) -> dict[str, Any]:
    candidate = _plane(probability, view) >= CANDIDATE_THRESHOLD
    jackpot_plane = _plane(jackpot_probability, view)
    jackpot_selected = _plane(jackpot_selected_volume, view) != 0
    jackpot_same_threshold = jackpot_plane >= CANDIDATE_THRESHOLD
    published = _plane(m7, view) != 0
    return {
        "view": view,
        "candidate_vs_jackpot_selected": slice_blob_metrics(
            candidate,
            jackpot_selected,
            minimum_component_pixels=MINIMUM_COMPONENT_PIXELS,
            skeleton_tolerance_pixels=SKELETON_TOLERANCE_PIXELS,
        ),
        "candidate_vs_jackpot_t025": slice_blob_metrics(
            candidate,
            jackpot_same_threshold,
            minimum_component_pixels=MINIMUM_COMPONENT_PIXELS,
            skeleton_tolerance_pixels=SKELETON_TOLERANCE_PIXELS,
        ),
        "candidate_vs_published_m7": slice_blob_metrics(
            candidate,
            published,
            minimum_component_pixels=MINIMUM_COMPONENT_PIXELS,
            skeleton_tolerance_pixels=SKELETON_TOLERANCE_PIXELS,
        ),
        "jackpot_t025_vs_jackpot_selected": slice_blob_metrics(
            jackpot_same_threshold,
            jackpot_selected,
            minimum_component_pixels=MINIMUM_COMPONENT_PIXELS,
            skeleton_tolerance_pixels=SKELETON_TOLERANCE_PIXELS,
        ),
    }


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return aggregate_slice_blob_metrics([row[key] for row in rows])


def _evaluate_grid(
    probability_root: Path,
    views: list[dict[str, Any]],
    jackpot_cache: dict[str, np.ndarray],
    jackpot_selected_cache: dict[str, np.ndarray],
    m7_cache: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    candidate_cache: dict[str, np.ndarray] = {}
    rows = []
    for view in views:
        cube_id = str(view["cube_id"])
        if cube_id not in candidate_cache:
            candidate_cache[cube_id] = _read_volume(
                probability_root, "probability", cube_id
            ).astype(np.float32, copy=False)
        rows.append(
            _view_metrics(
                candidate_cache[cube_id],
                jackpot_cache[cube_id],
                jackpot_selected_cache[cube_id],
                m7_cache[cube_id],
                view,
            )
        )
    return rows


def _full_cube_morphology(
    probability_cache: dict[str, np.ndarray],
    jackpot_selected_cache: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Run the pre-existing 3-D anti-blob geometry at fixed T=0.25."""

    cubes: list[dict[str, Any]] = []
    for cube_id in sorted(probability_cache):
        match = _CUBE.fullmatch(cube_id)
        if match is None:
            raise ValueError(f"invalid cube identifier: {cube_id}")
        origin = tuple(int(match.group(axis)) for axis in ("z", "y", "x"))
        candidate = probability_cache[cube_id] >= CANDIDATE_THRESHOLD
        reference = jackpot_selected_cache[cube_id]
        candidate_metrics = scrollfiesta_pred_metrics(
            candidate, window_origin_zyx=origin
        )
        reference_metrics = scrollfiesta_pred_metrics(
            reference, window_origin_zyx=origin
        )
        interior_regression = (
            candidate_metrics.interior_fraction
            > reference_metrics.interior_fraction + 0.10
        )
        thickness_regression = (
            candidate_metrics.max_thickness > reference_metrics.max_thickness + 2
        )
        cubes.append(
            {
                "cube_id": cube_id,
                "candidate": candidate_metrics.to_dict(),
                "reference": reference_metrics.to_dict(),
                "foreground_ratio_vs_reference": (
                    candidate_metrics.foreground_voxels
                    / max(1, reference_metrics.foreground_voxels)
                ),
                "interior_fraction_delta_vs_reference": (
                    candidate_metrics.interior_fraction
                    - reference_metrics.interior_fraction
                ),
                "max_thickness_delta_vs_reference": (
                    candidate_metrics.max_thickness
                    - reference_metrics.max_thickness
                ),
                "interior_regression_over_0p10": interior_regression,
                "thickness_regression_over_2vox": thickness_regression,
            }
        )
    candidate_positive = sum(
        int(row["candidate"]["foreground_voxels"]) for row in cubes
    )
    reference_positive = sum(
        int(row["reference"]["foreground_voxels"]) for row in cubes
    )
    foreground_pass = candidate_positive <= 1.10 * reference_positive
    interior_pass = all(
        not bool(row["interior_regression_over_0p10"]) for row in cubes
    )
    thickness_pass = all(
        not bool(row["thickness_regression_over_2vox"]) for row in cubes
    )
    return {
        "threshold": CANDIDATE_THRESHOLD,
        "aggregate_candidate_positive": candidate_positive,
        "aggregate_reference_positive": reference_positive,
        "aggregate_foreground_ratio_vs_reference": (
            candidate_positive / max(1, reference_positive)
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


def _morphology_failures(block: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    gates = block["gates"]
    if not bool(gates["foreground_not_over_incumbent_by_10pct"]):
        failures.append("aggregate:foreground_over_reference_by_10pct")
    for row in block["cubes"]:
        cube_id = str(row["cube_id"])
        if bool(row["interior_regression_over_0p10"]):
            failures.append(f"{cube_id}:interior_over_reference_by_0p10")
        if bool(row["thickness_regression_over_2vox"]):
            failures.append(f"{cube_id}:max_thickness_over_reference_by_2vox")
    return failures


def _validation_metrics(arm: Path, milestone: int) -> dict[str, Any]:
    history_path = arm / "history.jsonl"
    if history_path.is_file():
        matches = []
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if round(float(row["train"]["cumulative_samples"])) == milestone:
                matches.append(row)
        if len(matches) != 1:
            raise ValueError(
                f"{history_path}: expected one row at {milestone:,}, found {len(matches)}"
            )
        validation = matches[0]["val"]
        calibrated = {
            "threshold": float(validation["calibrated_threshold"]),
            "pooled_dice": float(validation["calibrated_dice"]),
            "macro_scroll_dice": float(
                validation["calibrated_macro_scroll_dice"]
            ),
            "precision": float(validation["calibrated_precision"]),
            "recall": float(validation["calibrated_recall"]),
        }
        fixed_t025 = {
            "threshold": CANDIDATE_THRESHOLD,
            "pooled_dice": float(validation["threshold/0.25/dice"]),
            "macro_scroll_dice": float(
                validation["threshold/0.25/macro_scroll_dice"]
            ),
            "precision": float(validation["threshold/0.25/precision"]),
            "recall": float(validation["threshold/0.25/recall"]),
        }
        return {
            "source": "training history; same frozen validation manifest",
            "report": str(history_path.resolve()),
            "report_sha256": _sha256(history_path),
            "fixed_t025_macro_scroll_dice": fixed_t025["macro_scroll_dice"],
            "calibrated_threshold": calibrated["threshold"],
            "calibrated_macro_scroll_dice": calibrated["macro_scroll_dice"],
            "standard_two_sided_dice": {
                "formula": "2*TP / (2*TP + FP + FN)",
                "calibrated": calibrated,
                "fixed_t025": fixed_t025,
            },
        }

    report_path = arm / f"ladder_eval/audit_{milestone:08d}/report.json"
    report = _read(report_path)
    sweep = report.get("sweep")
    if not isinstance(sweep, dict) or not isinstance(sweep.get("points"), list):
        raise TypeError(f"{report_path}: malformed validation sweep")
    fixed = next(
        (
            row
            for row in sweep["points"]
            if np.isclose(float(row["threshold"]), CANDIDATE_THRESHOLD)
        ),
        None,
    )
    calibrated = sweep.get("selected")
    if fixed is None or not isinstance(calibrated, dict):
        raise ValueError(f"{report_path}: fixed or calibrated point is missing")

    def standard_point(row: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "threshold": float(row["threshold"]),
            "pooled_dice": float(row["dice"]),
            "macro_scroll_dice": float(row["macro_scroll_dice"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
        }
        for key in ("true_positive", "false_positive", "false_negative"):
            if key in row:
                result[key] = int(row[key])
        return result

    calibrated_standard = standard_point(calibrated)
    fixed_standard = standard_point(fixed)
    return {
        "report": str(report_path.resolve()),
        "report_sha256": _sha256(report_path),
        "fixed_t025_macro_scroll_dice": fixed_standard["macro_scroll_dice"],
        "calibrated_threshold": calibrated_standard["threshold"],
        "calibrated_macro_scroll_dice": calibrated_standard[
            "macro_scroll_dice"
        ],
        "standard_two_sided_dice": {
            "formula": "2*TP / (2*TP + FP + FN)",
            "calibrated": calibrated_standard,
            "fixed_t025": fixed_standard,
        },
    }


def _fractional_reduction(candidate: int, baseline: int) -> float:
    if baseline <= 0:
        return 0.0 if candidate > 0 else 1.0
    return 1.0 - candidate / baseline


def _candidate_gate(
    *,
    candidate_named: dict[str, Any],
    candidate_guard: dict[str, Any],
    candidate_validation: dict[str, Any],
    candidate_full_cube: dict[str, Any],
    baseline_manifest: dict[str, Any],
) -> dict[str, Any]:
    contract = baseline_manifest["preregistered_candidate_gate"]
    baseline_named = baseline_manifest["selected_named_three"]
    baseline_guard = baseline_manifest["selected_fixed_eighteen"]
    baseline_validation = baseline_manifest["validation"]
    baseline_full_cube = baseline_manifest["full_cube_t025_morphology"]

    named_bridge_reduction = _fractional_reduction(
        int(candidate_named["candidate_only_bridge_pixels"]),
        int(baseline_named["candidate_only_bridge_pixels"]),
    )
    named_deep_reduction = _fractional_reduction(
        int(candidate_named["candidate_only_interior_pixels_r2"]),
        int(baseline_named["candidate_only_interior_pixels_r2"]),
    )
    named_recall_drop = float(baseline_named["reference_skeleton_recall"]) - float(
        candidate_named["reference_skeleton_recall"]
    )
    guard_recall_drop = float(baseline_guard["reference_skeleton_recall"]) - float(
        candidate_guard["reference_skeleton_recall"]
    )
    maximum_guard_bridge = (
        int(baseline_guard["candidate_only_bridge_pixels"])
        * (1.0 + float(contract["maximum_fixed18_bridge_pixel_fractional_increase"]))
    )
    maximum_guard_deep = (
        int(baseline_guard["candidate_only_interior_pixels_r2"])
        * (
            1.0
            + float(
                contract["maximum_fixed18_deep_interior_pixel_fractional_increase"]
            )
        )
    )
    maximum_validation_drop = float(contract["maximum_validation_macro_dice_drop"])
    fixed_delta = float(candidate_validation["fixed_t025_macro_scroll_dice"]) - float(
        baseline_validation["fixed_t025_macro_scroll_dice"]
    )
    calibrated_delta = float(
        candidate_validation["calibrated_macro_scroll_dice"]
    ) - float(baseline_validation["calibrated_macro_scroll_dice"])
    baseline_failures = set(_morphology_failures(baseline_full_cube))
    candidate_failures = set(_morphology_failures(candidate_full_cube))
    new_failures = sorted(candidate_failures - baseline_failures)

    criteria = {
        "named_bridge_reduction": {
            "value": named_bridge_reduction,
            "minimum": float(
                contract["required_named_bridge_pixel_reduction_vs_selected_15k"]
            ),
        },
        "named_deep_interior_reduction": {
            "value": named_deep_reduction,
            "minimum": float(
                contract[
                    "required_named_deep_interior_pixel_reduction_vs_selected_15k"
                ]
            ),
        },
        "named_reference_skeleton_recall_drop": {
            "value": named_recall_drop,
            "maximum": float(
                contract["maximum_named_reference_skeleton_recall_drop"]
            ),
        },
        "fixed18_bridge_pixels": {
            "value": int(candidate_guard["candidate_only_bridge_pixels"]),
            "maximum": maximum_guard_bridge,
        },
        "fixed18_deep_interior_pixels": {
            "value": int(candidate_guard["candidate_only_interior_pixels_r2"]),
            "maximum": maximum_guard_deep,
        },
        "fixed18_reference_skeleton_recall_drop": {
            "value": guard_recall_drop,
            "maximum": float(
                contract["maximum_fixed18_reference_skeleton_recall_drop"]
            ),
        },
        "fixed_t025_macro_scroll_dice_delta": {
            "value": fixed_delta,
            "minimum": -maximum_validation_drop,
        },
        "calibrated_macro_scroll_dice_delta": {
            "value": calibrated_delta,
            "minimum": -maximum_validation_drop,
        },
        "new_full_cube_t025_morphology_failures": {
            "value": new_failures,
            "maximum_count": 0,
        },
    }
    for criterion in criteria.values():
        value = criterion["value"]
        if "minimum" in criterion:
            criterion["passed"] = float(value) >= float(criterion["minimum"])
        elif "maximum" in criterion:
            criterion["passed"] = float(value) <= float(criterion["maximum"])
        else:
            criterion["passed"] = len(value) <= int(criterion["maximum_count"])
    return {
        "contract": "release-t025-blob-candidate-gate-v1",
        "baseline_arm": BASELINE_ARM_ID,
        "baseline_milestone_samples": BASELINE_MILESTONE,
        "criteria": criteria,
        "baseline_full_cube_failures": sorted(baseline_failures),
        "candidate_full_cube_failures": sorted(candidate_failures),
        "passed": all(bool(row["passed"]) for row in criteria.values()),
    }


def _plain_mask(mask: np.ndarray) -> Image.Image:
    value = np.where(np.asarray(mask, dtype=bool), 242, 5).astype(np.uint8)
    return Image.fromarray(value, mode="L").convert("RGB")


def _plain_ct(raw: np.ndarray) -> Image.Image:
    value = np.asarray(raw, dtype=np.float32)
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        gray = np.zeros(value.shape, dtype=np.uint8)
    else:
        low, high = np.quantile(finite, (0.01, 0.99))
        if high <= low:
            gray = np.zeros(value.shape, dtype=np.uint8)
        else:
            gray = np.clip((value - low) / (high - low), 0.0, 1.0)
            gray = np.asarray(np.round(gray * 255), dtype=np.uint8)
    return Image.fromarray(gray, mode="L").convert("RGB")


def _difference(candidate: np.ndarray, reference: np.ndarray) -> Image.Image:
    candidate = np.asarray(candidate, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    value = np.full((*candidate.shape, 3), 5, dtype=np.uint8)
    value[candidate & reference] = (225, 225, 225)
    value[candidate & ~reference] = (255, 82, 48)
    value[~candidate & reference] = (32, 205, 255)
    return Image.fromarray(value, mode="RGB")


def _save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((512, 512), resample=Image.Resampling.NEAREST).save(
        path, format="PNG", optimize=True
    )


def _render_named_assets(
    rows: list[dict[str, Any]],
    selected_probability: dict[str, np.ndarray],
    jackpot_cache: dict[str, np.ndarray],
    jackpot_selected_cache: dict[str, np.ndarray],
    m7_cache: dict[str, np.ndarray],
    raw_cache: dict[str, np.ndarray],
    output: Path,
) -> list[dict[str, Any]]:
    rendered = []
    for row in rows:
        view = row["view"]
        cube_id = str(view["cube_id"])
        candidate = _plane(selected_probability[cube_id], view) >= CANDIDATE_THRESHOLD
        jackpot_probability = _plane(jackpot_cache[cube_id], view)
        jackpot_selected = _plane(jackpot_selected_cache[cube_id], view) != 0
        jackpot_t025 = jackpot_probability >= CANDIDATE_THRESHOLD
        published = _plane(m7_cache[cube_id], view) != 0
        root = output / "assets" / (
            f"{cube_id}_{view['axis']}{int(view['global_coordinate'])}"
        )
        assets = {
            "ct": root / "ct.png",
            "jackpot_selected": root / "jackpot_t035.png",
            "jackpot_t025": root / "jackpot_t025.png",
            "published_m7": root / "published_m7.png",
            "candidate_t025": root / "candidate_t025.png",
            "difference": root / "difference_vs_jackpot_t035.png",
        }
        _save(_plain_ct(_plane(raw_cache[cube_id], view)), assets["ct"])
        _save(_plain_mask(jackpot_selected), assets["jackpot_selected"])
        _save(_plain_mask(jackpot_t025), assets["jackpot_t025"])
        _save(_plain_mask(published), assets["published_m7"])
        _save(_plain_mask(candidate), assets["candidate_t025"])
        _save(_difference(candidate, jackpot_selected), assets["difference"])
        rendered.append(
            {
                "view": view,
                "assets": {
                    key: path.relative_to(output).as_posix()
                    for key, path in assets.items()
                },
                "metrics": {
                    key: value for key, value in row.items() if key != "view"
                },
            }
        )
    return rendered


def _metric_table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        metric = row["metrics"]["candidate_vs_jackpot_selected"]
        body.append(
            "<tr>"
            f"<th>{html.escape(str(row['view']['label']))}</th>"
            f"<td>{int(metric['candidate_foreground_pixels']):,}</td>"
            f"<td>{int(metric['reference_foreground_pixels']):,}</td>"
            f"<td>{int(metric['merging_candidate_components'])}</td>"
            f"<td>{int(metric['merged_reference_component_excess'])}</td>"
            f"<td>{int(metric['candidate_only_bridge_pixels']):,}</td>"
            f"<td>{int(metric['candidate_only_interior_pixels_r2']):,}</td>"
            f"<td>{float(metric['candidate_interior_fraction_r2']):.3f} / "
            f"{float(metric['reference_interior_fraction_r2']):.3f}</td>"
            f"<td>{int(metric['candidate_max_taxicab_radius'])} / "
            f"{int(metric['reference_max_taxicab_radius'])}</td>"
            f"<td>{float(metric['reference_skeleton_recall']):.3f}</td>"
            f"<td>{float(metric['candidate_skeleton_precision']):.3f}</td>"
            "</tr>"
        )
    return "".join(body)


def _trend_table(trends: list[dict[str, Any]]) -> str:
    body = []
    for trend in trends:
        named = trend["named_three"]
        guard = trend["fixed_eighteen"]
        selected = " class=\"selected\"" if bool(trend["selected_baseline"]) else ""
        body.append(
            f"<tr{selected}><th>{int(trend['samples']):,}</th>"
            f"<td>{int(named['candidate_only_bridge_pixels']):,}</td>"
            f"<td>{int(named['candidate_only_interior_pixels_r2']):,}</td>"
            f"<td>{int(named['merged_reference_component_excess'])}</td>"
            f"<td>{float(named['reference_skeleton_recall']):.3f}</td>"
            f"<td>{float(named['candidate_skeleton_precision']):.3f}</td>"
            f"<td>{float(named['candidate_interior_fraction_r2']):.3f}</td>"
            f"<td>{int(named['maximum_candidate_taxicab_radius'])}</td>"
            f"<td>{int(guard['candidate_only_bridge_pixels']):,}</td>"
            f"<td>{int(guard['candidate_only_interior_pixels_r2']):,}</td>"
            f"<td>{float(guard['reference_skeleton_recall']):.3f}</td>"
            f"<td>{float(guard['candidate_skeleton_precision']):.3f}</td></tr>"
        )
    return "".join(body)


def _gate_table(gate: dict[str, Any] | None) -> str:
    if gate is None:
        return (
            "<p>This is the frozen baseline that defines the candidate gate; "
            "it is not judged against itself.</p>"
        )
    rows = []
    for name, criterion in gate["criteria"].items():
        value = criterion["value"]
        if isinstance(value, float):
            shown = f"{value:+.4f}"
        elif isinstance(value, list):
            shown = "none" if not value else ", ".join(str(item) for item in value)
        else:
            shown = f"{int(value):,}"
        if "minimum" in criterion:
            limit = f">= {float(criterion['minimum']):.4f}"
        elif "maximum" in criterion:
            limit = f"<= {float(criterion['maximum']):.4f}"
        else:
            limit = f"count <= {int(criterion['maximum_count'])}"
        verdict = "PASS" if bool(criterion["passed"]) else "FAIL"
        rows.append(
            f"<tr><th>{html.escape(name)}</th><td>{html.escape(shown)}</td>"
            f"<td>{html.escape(limit)}</td><td>{verdict}</td></tr>"
        )
    overall = "PASS" if bool(gate["passed"]) else "FAIL"
    return (
        f"<p><strong>Preregistered overall verdict: {overall}</strong></p>"
        "<div class=\"scroll\"><table><thead><tr><th>criterion</th>"
        "<th>candidate value</th><th>limit</th><th>verdict</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _write_html(
    output: Path,
    named: list[dict[str, Any]],
    trends: list[dict[str, Any]],
    selected_named: dict[str, Any],
    selected_guard: dict[str, Any],
    validation: dict[str, Any],
    *,
    arm_id: str,
    selected_milestone: int,
    gate: dict[str, Any] | None,
) -> Path:
    candidate_label = f"{arm_id} · {selected_milestone:,} · T=0.25"
    ordinary_dice = validation["standard_two_sided_dice"]
    calibrated_dice = ordinary_dice["calibrated"]
    fixed_t025_dice = ordinary_dice["fixed_t025"]
    figures = []
    columns = (
        ("ct", "9 µm CT"),
        ("jackpot_selected", "jackpot 08-15 · T=0.35"),
        ("jackpot_t025", "jackpot 08-15 · T=0.25 control"),
        ("published_m7", "published M7"),
        ("candidate_t025", candidate_label),
        ("difference", "difference vs jackpot T=0.35"),
    )
    for row in named:
        cells = "".join(
            f'<figure><a href="{html.escape(row["assets"][key])}">'
            f'<img src="{html.escape(row["assets"][key])}" alt="{html.escape(label)}"></a>'
            f"<figcaption>{html.escape(label)}</figcaption></figure>"
            for key, label in columns
        )
        figures.append(
            f"<section><h2>{html.escape(str(row['view']['label']))}</h2>"
            f'<div class="panels">{cells}</div></section>'
        )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(arm_id)} — fixed-T=0.25 blob diagnosis</title>
<style>
body{{font:15px system-ui,sans-serif;margin:24px;background:#15171a;color:#e9edf1}}
h1,h2{{line-height:1.2}} .callout{{max-width:1100px;background:#22272e;padding:14px;border-left:5px solid #f0b429}}
.panels{{display:grid;grid-template-columns:repeat(6,minmax(150px,1fr));gap:12px}}
figure{{margin:0}} img{{display:block;width:100%;image-rendering:pixelated;border:1px solid #59636f}}
figcaption{{padding-top:5px;color:#bac4ce}} table{{border-collapse:collapse;margin:12px 0 30px;min-width:1050px}}
th,td{{border:1px solid #4d5660;padding:6px 8px;text-align:right}} th:first-child{{text-align:left}}
tr.selected{{background:#313b25}} .legend b:first-child{{color:#ff5230}} .legend b:last-child{{color:#20cdff}}
.scroll{{overflow-x:auto}} code{{color:#ffe08a}}
@media(max-width:1100px){{.panels{{grid-template-columns:repeat(3,1fr)}}}}
</style></head><body>
<h1>{html.escape(arm_id)} at {selected_milestone:,} — low-threshold blob diagnosis</h1>
<p class="callout"><strong>Threshold contract:</strong> every release-student measurement is fixed at <code>T=0.25</code>.
The three planes below were named by the operator before this audit; the other fifteen fixed views are guards.
Jackpot at its recorded <code>T=0.35</code> is a morphology exemplar, not ground truth or a candidate threshold.</p>
<p class="legend">Difference panels: <b>orange = candidate only</b>, white = shared, <b>cyan = jackpot only</b>.</p>
<h2>What the fixed candidate checkpoint is doing</h2>
<p>Named-three bridge pixels: <strong>{int(selected_named['candidate_only_bridge_pixels']):,}</strong>;
reference-component merge excess: <strong>{int(selected_named['merged_reference_component_excess'])}</strong>;
candidate-only deep-interior pixels: <strong>{int(selected_named['candidate_only_interior_pixels_r2']):,}</strong>;
jackpot skeleton retained within one pixel: <strong>{float(selected_named['reference_skeleton_recall']):.1%}</strong>.
 Across all eighteen guards: {int(selected_guard['candidate_only_bridge_pixels']):,} bridge pixels and
 {int(selected_guard['candidate_only_interior_pixels_r2']):,} candidate-only deep-interior pixels, with
 {float(selected_guard['reference_skeleton_recall']):.1%} skeleton retention.</p>
<p><strong>Standard two-sided Dice</strong> (<code>2TP/(2TP+FP+FN)</code>):
calibrated T={float(calibrated_dice['threshold']):.2f} pooled
<strong>{float(calibrated_dice['pooled_dice']):.6f}</strong>, macro-by-scroll
<strong>{float(calibrated_dice['macro_scroll_dice']):.6f}</strong>;
fixed T=0.25 pooled <strong>{float(fixed_t025_dice['pooled_dice']):.6f}</strong>,
macro-by-scroll <strong>{float(fixed_t025_dice['macro_scroll_dice']):.6f}</strong>.</p>
<div class="scroll"><table><thead><tr><th>fixed plane</th><th>candidate px</th><th>jackpot px</th><th>merging CC</th>
<th>merge excess</th><th>bridge px</th><th>deep-only px</th><th>interior r2 (candidate / jackpot)</th><th>max radius (candidate / jackpot)</th><th>line recall</th><th>line precision</th></tr></thead>
<tbody>{_metric_table(named)}</tbody></table></div>
{''.join(figures)}
<h2>Preregistered candidate gate</h2>
{_gate_table(gate)}
<h2>Within-arm duration control at the same fixed T=0.25</h2>
<p>The highlighted row is the fixed {selected_milestone:,}-sample checkpoint. The row was declared before inspection.</p>
<div class="scroll"><table><thead><tr><th>samples</th><th>named bridge px</th><th>named deep-only px</th><th>named merge excess</th>
<th>named line recall</th><th>named line precision</th><th>named interior r2</th><th>named max radius</th>
<th>18-view bridge px</th><th>18-view deep-only px</th><th>18-view line recall</th><th>18-view line precision</th></tr></thead>
<tbody>{_trend_table(trends)}</tbody></table></div>
</body></html>"""
    index = output / "index.html"
    _atomic_text(index, page)
    return index


def build(
    output: Path,
    *,
    arm_id: str = BASELINE_ARM_ID,
    selected_milestone: int = BASELINE_MILESTONE,
) -> Path:
    config = _read(ROOT / "crossres_pred/configs/release_ladder_20260902.json")
    known_arms = {str(row["arm_id"]) for row in config["arms"]}
    if arm_id not in known_arms:
        raise ValueError(f"unknown release arm: {arm_id}")
    if selected_milestone <= 0:
        raise ValueError("selected milestone must be positive")
    arm = RELEASE / "arms" / arm_id
    comparison_path = COMPARISON / "manifest.json"
    comparison = _read(comparison_path)
    provenances = _verify_inputs(comparison, arm, selected_milestone)
    views = [_parse_view(value) for value in comparison["fixed_slices"]]
    specifications = {str(view["specification"]) for view in views}
    if len(views) != 18 or len(specifications) != 18:
        raise ValueError("diagnostic guard must be the exact fixed eighteen")
    if not set(WATCHLIST).issubset(specifications):
        raise ValueError("operator-named views are absent from the fixed eighteen")

    cube_ids = sorted({str(view["cube_id"]) for view in views})
    jackpot_cache = {
        cube_id: _read_volume(JACKPOT, "probability", cube_id).astype(
            np.float32, copy=False
        )
        for cube_id in cube_ids
    }
    jackpot_selected_cache = {
        cube_id: _read_volume(JACKPOT, "cubes_PRED", cube_id) != 0
        for cube_id in cube_ids
    }
    m7_cache = {
        cube_id: _read_volume(SOURCE, "cubes_PRED", cube_id) for cube_id in cube_ids
    }
    raw_cache = {
        cube_id: _read_volume(SOURCE, "cubes_RAW", cube_id) for cube_id in cube_ids
    }
    # The binary was cut from float32 inference before probabilities were stored
    # as float16. Record the tiny threshold-boundary rounding delta, but use the
    # authoritative binary rather than silently regenerating a different mask.
    jackpot_quantization_audit = []
    for cube_id in cube_ids:
        recorded = jackpot_selected_cache[cube_id]
        regenerated = jackpot_cache[cube_id] >= JACKPOT_THRESHOLD
        mismatch = recorded ^ regenerated
        mismatch_count = int(np.count_nonzero(mismatch))
        mismatch_values = jackpot_cache[cube_id][mismatch]
        if mismatch_count > recorded.size // 10_000 or (
            mismatch_count
            and float(np.max(np.abs(mismatch_values - JACKPOT_THRESHOLD))) > 0.0005
        ):
            raise ValueError(
                f"{cube_id}: jackpot mask differs beyond float16 threshold rounding"
            )
        jackpot_quantization_audit.append(
            {
                "cube_id": cube_id,
                "authoritative_binary_foreground": int(np.count_nonzero(recorded)),
                "float16_regenerated_foreground": int(
                    np.count_nonzero(regenerated)
                ),
                "mismatched_voxels": mismatch_count,
                "maximum_probability_delta_from_threshold": (
                    0.0
                    if mismatch_count == 0
                    else float(
                        np.max(np.abs(mismatch_values - JACKPOT_THRESHOLD))
                    )
                ),
            }
        )

    milestones = _discover_milestones(arm, selected_milestone)
    trend_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] | None = None
    selected_probability: dict[str, np.ndarray] | None = None
    input_files: list[dict[str, Any]] = []
    for samples, inference in milestones:
        provenance = _probability_provenance(inference)
        checkpoint_sha = str(provenance.get("checkpoint_sha256", ""))
        rows = _evaluate_grid(
            inference,
            views,
            jackpot_cache,
            jackpot_selected_cache,
            m7_cache,
        )
        named = [
            next(
                row
                for row in rows
                if row["view"]["specification"] == specification
            )
            for specification in WATCHLIST
        ]
        trend_rows.append(
            {
                "samples": samples,
                "checkpoint_sha256": checkpoint_sha,
                "selected_baseline": samples == selected_milestone,
                "named_three": _aggregate(
                    named, "candidate_vs_jackpot_selected"
                ),
                "fixed_eighteen": _aggregate(
                    rows, "candidate_vs_jackpot_selected"
                ),
            }
        )
        for cube_id in cube_ids:
            path = inference / "probability" / f"{cube_id}.tif"
            input_files.append(
                {
                    "role": f"candidate_probability_{samples}",
                    "cube_id": cube_id,
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                }
            )
        if samples == selected_milestone:
            selected_rows = rows
            selected_probability = {
                cube_id: _read_volume(inference, "probability", cube_id).astype(
                    np.float32, copy=False
                )
                for cube_id in cube_ids
            }
    assert selected_rows is not None and selected_probability is not None

    selected_named_rows = [
        next(
            row
            for row in selected_rows
            if row["view"]["specification"] == specification
        )
        for specification in WATCHLIST
    ]
    rendered = _render_named_assets(
        selected_named_rows,
        selected_probability,
        jackpot_cache,
        jackpot_selected_cache,
        m7_cache,
        raw_cache,
        output,
    )
    selected_named = _aggregate(
        selected_named_rows, "candidate_vs_jackpot_selected"
    )
    selected_guard = _aggregate(
        selected_rows, "candidate_vs_jackpot_selected"
    )
    validation = _validation_metrics(arm, selected_milestone)
    full_cube = _full_cube_morphology(
        selected_probability, jackpot_selected_cache
    )
    baseline_record: dict[str, Any] | None = None
    gate: dict[str, Any] | None = None
    if arm_id != BASELINE_ARM_ID or selected_milestone != BASELINE_MILESTONE:
        baseline_path = BASELINE_OUTPUT / "manifest.json"
        baseline_record = _read(baseline_path)
        baseline_candidate = baseline_record.get("candidate", {})
        if (
            baseline_candidate.get("arm") != BASELINE_ARM_ID
            or int(baseline_candidate.get("milestone_samples", -1))
            != BASELINE_MILESTONE
            or float(baseline_candidate.get("threshold", -1.0))
            != CANDIDATE_THRESHOLD
        ):
            raise ValueError("the frozen T=0.25 baseline manifest changed")
        gate = _candidate_gate(
            candidate_named=selected_named,
            candidate_guard=selected_guard,
            candidate_validation=validation,
            candidate_full_cube=full_cube,
            baseline_manifest=baseline_record,
        )
    index = _write_html(
        output,
        rendered,
        trend_rows,
        selected_named,
        selected_guard,
        validation,
        arm_id=arm_id,
        selected_milestone=selected_milestone,
        gate=gate,
    )

    for cube_id in cube_ids:
        for role, path in (
            ("jackpot_probability", JACKPOT / "probability" / f"{cube_id}.tif"),
            ("jackpot_selected_mask", JACKPOT / "cubes_PRED" / f"{cube_id}.tif"),
            ("published_m7", SOURCE / "cubes_PRED" / f"{cube_id}.tif"),
            ("raw_ct", SOURCE / "cubes_RAW" / f"{cube_id}.tif"),
        ):
            input_files.append(
                {
                    "role": role,
                    "cube_id": cube_id,
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                }
            )
    input_files.sort(key=lambda row: (str(row["role"]), str(row["cube_id"])))

    manifest = {
        "schema": "crossres-release-t025-blob-diagnostic-v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "research_only": True,
        "quality_claim": False,
        "candidate": {
            "arm": arm_id,
            "milestone_samples": selected_milestone,
            "trust_radius": provenances["effective_trust_radius"],
            "trust_ratio_to_shipped": float(
                comparison["trust_radii"]["with_ratio_to_shipped"]
            ),
            "threshold": CANDIDATE_THRESHOLD,
            "checkpoint_sha256": str(
                provenances["selected"]["checkpoint_sha256"]
            ),
            "inference": "raw student; 8-way mirror TTA; no blend or postprocessing",
        },
        "reference": {
            "name": "jackpot 08-15",
            "role": "operator-nominated morphology exemplar; not ground truth",
            "primary_threshold": JACKPOT_THRESHOLD,
            "same_threshold_control": CANDIDATE_THRESHOLD,
            "primary_mask_source": (
                "authoritative cubes_PRED cut from pre-quantization float32"
            ),
            "float16_probability_rounding_audit": jackpot_quantization_audit,
            "checkpoint_sha256": str(
                provenances["jackpot"]["checkpoint_sha256"]
            ),
        },
        "measurement_contract": {
            "candidate_threshold": CANDIDATE_THRESHOLD,
            "threshold_is_fixed_not_selected": True,
            "minimum_2d_component_pixels": MINIMUM_COMPONENT_PIXELS,
            "connectivity": "4-neighbour within each plane",
            "line_proxy": "jackpot 2-D skeleton coverage within one pixel",
            "blob_proxy": (
                "candidate-only pixels in candidate components that touch at "
                "least two >=16-pixel jackpot components"
            ),
            "warning": (
                "reference-relative bridges can be legitimate continuations; "
                "the three CT panels remain binding visual evidence"
            ),
        },
        "preregistered_candidate_gate": {
            "threshold": CANDIDATE_THRESHOLD,
            "named_views": list(WATCHLIST),
            "guard_views": [str(view["specification"]) for view in views],
            "required_named_bridge_pixel_reduction_vs_selected_15k": 0.15,
            "required_named_deep_interior_pixel_reduction_vs_selected_15k": 0.15,
            "maximum_named_reference_skeleton_recall_drop": 0.01,
            "maximum_fixed18_bridge_pixel_fractional_increase": 0.05,
            "maximum_fixed18_deep_interior_pixel_fractional_increase": 0.05,
            "maximum_fixed18_reference_skeleton_recall_drop": 0.01,
            "maximum_validation_macro_dice_drop": 0.005,
            "validation_guards": (
                "fixed-T=0.25 and calibrated macro Dice each no worse than "
                "0.005; no new full-cube T=0.25 morphology failure"
            ),
            "selection_prohibition": (
                "do not choose another threshold, view subset, or checkpoint "
                "after looking at the candidate"
            ),
        },
        "selected_named_three": selected_named,
        "selected_fixed_eighteen": selected_guard,
        "validation": validation,
        "full_cube_t025_morphology": full_cube,
        "candidate_gate": gate,
        "named_views": rendered,
        "duration_control": trend_rows,
        "inputs": {
            "trust_ball_comparison_manifest": {
                "path": str(comparison_path.resolve()),
                "sha256": _sha256(comparison_path),
            },
            "frozen_baseline_manifest": (
                None
                if baseline_record is None
                else {
                    "path": str((BASELINE_OUTPUT / "manifest.json").resolve()),
                    "sha256": _sha256(BASELINE_OUTPUT / "manifest.json"),
                }
            ),
            "files": input_files,
        },
        "report": {
            "index": str(index.resolve()),
            "sha256": _sha256(index),
        },
    }
    _atomic_json(output / "manifest.json", manifest)
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default=BASELINE_ARM_ID)
    parser.add_argument("--milestone", type=int, default=BASELINE_MILESTONE)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    output = arguments.output
    if output is None:
        if (
            arguments.arm == BASELINE_ARM_ID
            and arguments.milestone == BASELINE_MILESTONE
        ):
            output = BASELINE_OUTPUT
        else:
            output = (
                RELEASE
                / "diagnostics/t025_blob_candidates_20260903"
                / f"{arguments.arm}_{arguments.milestone:08d}"
            )
    print(
        build(
            output.expanduser().resolve(),
            arm_id=arguments.arm,
            selected_milestone=arguments.milestone,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
