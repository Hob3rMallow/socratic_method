"""Audit the v31 duration ladder against the locked teacher with NVIDIA FLIP.

The audit consumes the exact plain binary PNGs rendered for human review.  It
keeps every slice/model/threshold score, and joins the independent PHerc1447
anti-blob result so that a perceptual optimum cannot select a blob-producing
operating point.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / (
    "output/crossres_data/"
    "m7_xr_v31_duration_ladder_human_report_20260831"
)
JOINT_RESULT = REPO / (
    "output/crossres_data/"
    "m7_xr_v31_duration_ladder_joint_evaluation_20260831/result.json"
)
OUTPUT = REPO / (
    "output/crossres_data/"
    "m7_xr_v31_duration_ladder_flip_audit_20260901"
)
DEFAULT_FLIP = Path(r"D:\work\flip.exe")
EXPECTED_SAMPLES = (1024, 2048, 3072, 4096, 8192)
EXPECTED_THRESHOLDS = (
    0.25,
    0.38,
    0.39,
    0.40,
    0.41,
    0.42,
    0.43,
    0.44,
    0.45,
    0.46,
    0.47,
    0.48,
    0.49,
    0.50,
)
FLOAT_PATTERN = r"([0-9]+(?:\.[0-9]+)?)"
MEAN_RE = re.compile(rf"^\s*Mean:\s*{FLOAT_PATTERN}\s*$", re.MULTILINE)
WEIGHTED_MEDIAN_RE = re.compile(
    rf"^\s*Weighted median:\s*{FLOAT_PATTERN}\s*$", re.MULTILINE
)
PPD_RE = re.compile(rf"Resulting PPD\s*=\s*{FLOAT_PATTERN}")


@dataclass(frozen=True)
class Task:
    rank: int
    label: str
    samples: int
    threshold: float
    reference: Path
    candidate: Path


@dataclass(frozen=True)
class Score:
    rank: int
    label: str
    samples: int
    threshold: float
    flip_mean: float
    flip_weighted_median: float
    ppd: float
    teacher_foreground: int
    candidate_foreground: int
    intersection: int
    false_negative_erosion: int
    false_positive_addition: int
    symmetric_difference: int
    dice: float
    reference: str
    candidate: str


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _threshold_key(value: float) -> str:
    return f"{value:.2f}"


def _tasks(manifest: dict[str, Any]) -> list[Task]:
    samples = tuple(int(value) for value in manifest["samples"])
    thresholds = tuple(float(value) for value in manifest["thresholds"])
    rows = manifest["locked16_rows"]
    if samples != EXPECTED_SAMPLES:
        raise ValueError(f"unexpected duration ladder: {samples}")
    if thresholds != EXPECTED_THRESHOLDS:
        raise ValueError(f"unexpected threshold ladder: {thresholds}")
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError("FLIP audit requires the exact locked 16")

    result: list[Task] = []
    for row in rows:
        rank = int(row["rank"])
        label = str(row["label"])
        reference = REPORT / "assets/locked16" / f"rank_{rank:03d}" / "teacher.png"
        models = row["models"]
        if tuple(int(model["samples"]) for model in models) != samples:
            raise ValueError(f"model order changed for rank {rank}")
        for model in models:
            model_samples = int(model["samples"])
            for threshold in thresholds:
                panel = model["panels"][_threshold_key(threshold)]
                candidate = REPORT / str(panel["src"])
                if not reference.is_file() or not candidate.is_file():
                    raise FileNotFoundError(reference if not reference.is_file() else candidate)
                result.append(
                    Task(
                        rank=rank,
                        label=label,
                        samples=model_samples,
                        threshold=threshold,
                        reference=reference,
                        candidate=candidate,
                    )
                )
    if len(result) != 16 * len(samples) * len(thresholds):
        raise AssertionError("incomplete FLIP task matrix")
    return result


def _mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) >= 128


def _score(task: Task, flip: Path) -> Score:
    completed = subprocess.run(
        [str(flip), str(task.reference), str(task.candidate), "-v", "2"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout = completed.stdout
    mean_match = MEAN_RE.search(stdout)
    weighted_match = WEIGHTED_MEDIAN_RE.search(stdout)
    ppd_match = PPD_RE.search(stdout)
    if mean_match is None or weighted_match is None or ppd_match is None:
        raise ValueError(f"could not parse FLIP output for {task.candidate}:\n{stdout}")

    teacher = _mask(task.reference)
    candidate = _mask(task.candidate)
    if teacher.shape != candidate.shape:
        raise ValueError(f"shape mismatch for {task.candidate}")
    intersection = int(np.count_nonzero(teacher & candidate))
    teacher_count = int(np.count_nonzero(teacher))
    candidate_count = int(np.count_nonzero(candidate))
    erosion = int(np.count_nonzero(teacher & ~candidate))
    addition = int(np.count_nonzero(candidate & ~teacher))
    denominator = teacher_count + candidate_count
    dice = 1.0 if denominator == 0 else (2.0 * intersection) / denominator
    return Score(
        rank=task.rank,
        label=task.label,
        samples=task.samples,
        threshold=task.threshold,
        flip_mean=float(mean_match.group(1)),
        flip_weighted_median=float(weighted_match.group(1)),
        ppd=float(ppd_match.group(1)),
        teacher_foreground=teacher_count,
        candidate_foreground=candidate_count,
        intersection=intersection,
        false_negative_erosion=erosion,
        false_positive_addition=addition,
        symmetric_difference=erosion + addition,
        dice=dice,
        reference=str(task.reference.resolve()),
        candidate=str(task.candidate.resolve()),
    )


def _joint_lookup(joint: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    lookup: dict[tuple[int, str], dict[str, Any]] = {}
    for record in joint["records"]:
        samples = int(record["samples"])
        for row in record["threshold_summary"]:
            lookup[(samples, _threshold_key(float(row["threshold"])))] = row
    return lookup


def _summaries(
    scores: list[Score], joint: dict[str, Any]
) -> list[dict[str, Any]]:
    lookup = _joint_lookup(joint)
    grouped: dict[tuple[int, str], list[Score]] = {}
    for score in scores:
        grouped.setdefault(
            (score.samples, _threshold_key(score.threshold)), []
        ).append(score)

    summaries: list[dict[str, Any]] = []
    for (samples, threshold_key), rows in sorted(grouped.items()):
        if len(rows) != 16:
            raise ValueError(f"incomplete score group: {samples}, {threshold_key}")
        threshold = float(threshold_key)
        gate = lookup.get((samples, threshold_key))
        summary: dict[str, Any] = {
            "samples": samples,
            "threshold": threshold,
            "slice_count": len(rows),
            "flip_mean_macro": fmean(row.flip_mean for row in rows),
            "flip_mean_median_slice": median(row.flip_mean for row in rows),
            "flip_weighted_median_macro": fmean(
                row.flip_weighted_median for row in rows
            ),
            "dice_macro": fmean(row.dice for row in rows),
            "erosion_pixels_total": sum(row.false_negative_erosion for row in rows),
            "addition_pixels_total": sum(row.false_positive_addition for row in rows),
            "symmetric_difference_total": sum(row.symmetric_difference for row in rows),
            "pherc1447_anti_blob_passed": (
                None if gate is None else bool(gate["pherc1447_anti_blob_passed"])
            ),
            "pherc1447_foreground_ratio_vs_v15": (
                None
                if gate is None
                else float(gate["pherc1447_foreground_ratio_vs_v15"])
            ),
            "pherc1447_reference_recall": (
                None if gate is None else float(gate["pherc1447_reference_recall"])
            ),
            "locked_component_matches": (
                None if gate is None else int(gate["locked_component_matches"])
            ),
        }
        summaries.append(summary)
    return summaries


def _rank_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (row["flip_mean_macro"], -row["dice_macro"]))


def _slice_comparison(
    scores: list[Score], *, threshold: float, left: int, right: int
) -> list[dict[str, Any]]:
    key = _threshold_key(threshold)
    selected = {
        (row.rank, row.samples): row
        for row in scores
        if _threshold_key(row.threshold) == key and row.samples in {left, right}
    }
    result: list[dict[str, Any]] = []
    for rank in sorted({row.rank for row in scores}):
        a = selected[(rank, left)]
        b = selected[(rank, right)]
        result.append(
            {
                "rank": rank,
                "label": a.label,
                "threshold": threshold,
                "left_samples": left,
                "right_samples": right,
                "flip_mean_left": a.flip_mean,
                "flip_mean_right": b.flip_mean,
                "flip_mean_delta_right_minus_left": b.flip_mean - a.flip_mean,
                "erosion_left": a.false_negative_erosion,
                "erosion_right": b.false_negative_erosion,
                "erosion_delta_right_minus_left": (
                    b.false_negative_erosion - a.false_negative_erosion
                ),
                "addition_left": a.false_positive_addition,
                "addition_right": b.false_positive_addition,
                "addition_delta_right_minus_left": (
                    b.false_positive_addition - a.false_positive_addition
                ),
                "dice_left": a.dice,
                "dice_right": b.dice,
                "dice_delta_right_minus_left": b.dice - a.dice,
                "right_better_by_flip": b.flip_mean < a.flip_mean,
            }
        )
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flip", type=Path, default=DEFAULT_FLIP)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(12, (os.cpu_count() or 4) // 2))
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    flip = args.flip.resolve()
    if not flip.is_file():
        raise FileNotFoundError(flip)
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    manifest_path = REPORT / "report.json"
    manifest = _read(manifest_path)
    joint = _read(JOINT_RESULT)
    tasks = _tasks(manifest)
    print(f"FLIP matrix: {len(tasks):,} pairs with {args.workers} workers", flush=True)

    scores: list[Score] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_score, task, flip): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            scores.append(future.result())
            if index % 100 == 0 or index == len(tasks):
                print(f"FLIP {index:,}/{len(tasks):,}", flush=True)
    scores.sort(key=lambda row: (row.samples, row.threshold, row.rank))

    score_rows = [asdict(row) for row in scores]
    summaries = _summaries(scores, joint)
    ranked = _rank_summaries(summaries)
    eligible = [row for row in ranked if row["pherc1447_anti_blob_passed"] is True]
    if not eligible:
        raise ValueError("no FLIP candidates pass the PHerc1447 anti-blob rule")
    t045 = [row for row in summaries if _threshold_key(row["threshold"]) == "0.45"]
    t045.sort(key=lambda row: row["samples"])
    erosion = _slice_comparison(scores, threshold=0.45, left=2048, right=8192)
    human_candidate = next(
        row
        for row in summaries
        if row["samples"] == 8192 and _threshold_key(row["threshold"]) == "0.45"
    )
    eligible_rank = next(
        index
        for index, row in enumerate(eligible, start=1)
        if row["samples"] == 8192 and _threshold_key(row["threshold"]) == "0.45"
    )

    output = {
        "schema": "crossres-m7-xr-v31-duration-flip-audit-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "research_only": True,
        "comparison": "plain binary model mask versus locked plain binary teacher mask",
        "primary_metric": "macro mean across the 16 per-image FLIP means; lower is better",
        "flip": {
            "path": str(flip),
            "sha256": _sha256(flip),
            "version": "FLIP v1.0",
            "ppd": scores[0].ppd,
        },
        "sources": {
            "human_report_manifest": str(manifest_path.resolve()),
            "human_report_manifest_sha256": _sha256(manifest_path),
            "joint_gate_result": str(JOINT_RESULT.resolve()),
            "joint_gate_result_sha256": _sha256(JOINT_RESULT),
        },
        "matrix": {
            "slices": 16,
            "samples": list(EXPECTED_SAMPLES),
            "thresholds": list(EXPECTED_THRESHOLDS),
            "pair_count": len(scores),
        },
        "all_candidate_winner": ranked[0],
        "anti_blob_eligible_winner": eligible[0],
        "human_candidate": {
            **human_candidate,
            "rank_among_anti_blob_eligible": eligible_rank,
            "eligible_candidate_count": len(eligible),
        },
        "t0p45_duration_summary": t045,
        "t0p45_2048_to_8192_slice_comparison": erosion,
        "t0p45_2048_to_8192_counts": {
            "flip_improved_slices": sum(row["right_better_by_flip"] for row in erosion),
            "flip_worsened_or_tied_slices": sum(
                not row["right_better_by_flip"] for row in erosion
            ),
            "erosion_increased_slices": sum(
                row["erosion_delta_right_minus_left"] > 0 for row in erosion
            ),
            "erosion_decreased_slices": sum(
                row["erosion_delta_right_minus_left"] < 0 for row in erosion
            ),
        },
        "ranked_summary": ranked,
        "scores": score_rows,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT / "audit.json"
    result_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    _write_csv(OUTPUT / "scores.csv", score_rows)
    _write_csv(OUTPUT / "summary.csv", summaries)
    _write_csv(OUTPUT / "t0p45_2048_to_8192.csv", erosion)
    print(json.dumps({
        "all_candidate_winner": ranked[0],
        "anti_blob_eligible_winner": eligible[0],
        "human_candidate": output["human_candidate"],
        "erosion_counts": output["t0p45_2048_to_8192_counts"],
        "output": str(result_path.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
