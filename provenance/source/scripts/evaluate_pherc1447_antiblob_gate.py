#!/usr/bin/env python3
"""Run the standard PHerc1447 anti-blob contract without rendering a report.

This promotion gate writes inference plus JSON diagnostics only. Human-facing
HTML and PNG artifacts are deliberately left to ``generate_checkpoint_report``
after both the growth and anti-blob gates have passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import render_voxel_grid_report as render

from crossres_pred.voxel.grid_inference import infer_voxel_grid
from crossres_pred.voxel.resources import configure_cpu_budget
from crossres_pred.voxel.scrollfiesta_metrics import scrollfiesta_pred_metrics

SCHEMA = "crossres-pherc1447-machine-antiblob-gate-v1"
CONTRACT = "pherc1447-blind-antiblob-regression-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _cube_ids(source: Path) -> list[str]:
    value = json.loads(
        (source / "cubes_PRED" / "present.json").read_text(encoding="utf-8")
    )
    if not isinstance(value, list) or len(value) != 6 or len(set(value)) != 6:
        raise ValueError("standard anti-blob corpus must contain exactly six cubes")
    return sorted(str(item) for item in value)


def _regression_gates(
    aggregate: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    foreground_pass = not bool(aggregate["foreground_regression_over_10pct"])
    interior_pass = all(
        not bool(row["interior_regression_over_0p10"]) for row in rows
    )
    thickness_pass = all(
        not bool(row["thickness_regression_over_2vox"]) for row in rows
    )
    return {
        "contract": CONTRACT,
        "foreground_not_over_incumbent_by_10pct": foreground_pass,
        "no_cube_interior_over_incumbent_by_0p10": interior_pass,
        "no_cube_max_thickness_over_incumbent_by_2vox": thickness_pass,
        "passed": foreground_pass and interior_pass and thickness_pass,
    }


def _metrics(
    *, source: Path, inference: Path, reference: Path, cube_id: str, chunk_size: int
) -> dict[str, Any]:
    student = render._read_cube(
        inference / "cubes_PRED" / f"{cube_id}.tif", chunk_size
    ).astype(bool)
    published = render._read_cube(
        source / "cubes_PRED" / f"{cube_id}.tif", chunk_size
    ).astype(bool)
    incumbent = render._read_cube(
        reference / "cubes_PRED" / f"{cube_id}.tif", chunk_size
    ).astype(bool)
    row = render._metrics(student, published)
    student_topology = scrollfiesta_pred_metrics(student)
    reference_topology = scrollfiesta_pred_metrics(incumbent)
    shared = int(np.count_nonzero(student & incumbent))
    student_positive = int(np.count_nonzero(student))
    reference_positive = int(np.count_nonzero(incumbent))
    row.update(
        reference_positive=reference_positive,
        reference_shared_positive=shared,
        student_added_vs_reference=student_positive - shared,
        reference_missing_vs_student=reference_positive - shared,
        dice_vs_reference=(2.0 * shared)
        / max(1, student_positive + reference_positive),
        student_interior_fraction=student_topology.interior_fraction,
        reference_interior_fraction=reference_topology.interior_fraction,
        student_max_thickness=student_topology.max_thickness,
        reference_max_thickness=reference_topology.max_thickness,
        interior_regression_over_0p10=(
            student_topology.interior_fraction
            > reference_topology.interior_fraction + 0.10
        ),
        thickness_regression_over_2vox=(
            student_topology.max_thickness > reference_topology.max_thickness + 2
        ),
    )
    return row


def _validate_inference(
    *,
    inference: Path,
    source: Path,
    checkpoint: Path,
    cube_ids: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    provenance_path = inference / "provenance.json"
    provenance = _read_object(provenance_path)
    options = provenance.get("options", {})
    expected = {
        "checkpoint_sha256": _sha256(checkpoint),
        "source_grid": str(source),
        "target_cube_ids": cube_ids,
        "threshold": args.threshold,
        "halo": args.halo,
        "device": args.device,
        "amp_dtype": args.amp_dtype,
        "mirror_tta": not args.no_tta,
        "max_cpu_threads": args.max_cpu_threads,
    }
    observed = {
        "checkpoint_sha256": provenance.get("checkpoint_sha256"),
        "source_grid": provenance.get("source_grid"),
        "target_cube_ids": provenance.get("target_cube_ids"),
        "threshold": provenance.get("threshold"),
        "halo": provenance.get("halo"),
        "device": provenance.get("device"),
        "amp_dtype": provenance.get("amp_dtype"),
        "mirror_tta": provenance.get("mirror_tta"),
        "max_cpu_threads": options.get("max_cpu_threads"),
    }
    if observed != expected:
        raise ValueError("existing standard-corpus inference identity differs")
    present = json.loads(
        (inference / "cubes_PRED" / "present.json").read_text(encoding="utf-8")
    )
    if sorted(str(item) for item in present) != cube_ids:
        raise ValueError("standard-corpus inference is incomplete")
    return provenance


def run(args: argparse.Namespace) -> Path:
    source = args.source.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    reference = args.reference_student.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path in (source / "manifest.json", checkpoint):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    cube_ids = _cube_ids(source)
    reference_ids = set(
        json.loads(
            (reference / "cubes_PRED" / "present.json").read_text(
                encoding="utf-8"
            )
        )
    )
    if not set(cube_ids).issubset(reference_ids):
        raise ValueError("incumbent reference is missing standard cubes")
    chunk_size = int(_read_object(source / "manifest.json")["chunk_size"])
    output.mkdir(parents=True, exist_ok=True)
    inference = output / "inference"
    if not inference.exists():
        infer_voxel_grid(
            source_grid=source,
            checkpoint_path=checkpoint,
            output_path=inference,
            threshold=args.threshold,
            halo=args.halo,
            device_name=args.device,
            amp_dtype_name=args.amp_dtype,
            mirror_tta=not args.no_tta,
            max_cpu_threads=args.max_cpu_threads,
            target_cube_ids=cube_ids,
        )
    provenance = _validate_inference(
        inference=inference,
        source=source,
        checkpoint=checkpoint,
        cube_ids=cube_ids,
        args=args,
    )
    cubes = [
        {
            "cube_id": cube_id,
            "metrics": _metrics(
                source=source,
                inference=inference,
                reference=reference,
                cube_id=cube_id,
                chunk_size=chunk_size,
            ),
        }
        for cube_id in cube_ids
    ]
    metric_rows = [row["metrics"] for row in cubes]
    aggregate = render._aggregate_metrics(metric_rows)
    gates = _regression_gates(aggregate, metric_rows)
    payload = {
        "schema": SCHEMA,
        "stage": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "research_only": True,
        "human_report_created": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "source": str(source),
        "source_manifest_sha256": _sha256(source / "manifest.json"),
        "reference_student": str(reference),
        "inference_provenance": str(inference / "provenance.json"),
        "inference_provenance_sha256": _sha256(inference / "provenance.json"),
        "inference_options": provenance["options"],
        "aggregate": aggregate,
        "blind_regression_gates": gates,
        "cubes": cubes,
    }
    evaluation = output / "evaluation.json"
    _write_json(evaluation, payload)
    _write_json(
        output / "status.json",
        {
            "schema": SCHEMA,
            "stage": "complete",
            "passed": gates["passed"],
            "human_report_ready": gates["passed"],
            "human_report_created": False,
            "evaluation": str(evaluation),
        },
    )
    return evaluation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-student", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--halo", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", default="bfloat16")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--max-cpu-threads", type=int, default=16)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    if not 1 <= args.max_cpu_threads <= 16:
        raise ValueError("max-cpu-threads must lie in [1, 16]")
    configure_cpu_budget(args.max_cpu_threads)
    print(run(args), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
