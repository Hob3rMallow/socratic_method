#!/usr/bin/env python3
"""Render a new threshold from a provenance-bound cached probability grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from render_voxel_grid_report import REVIEW_LAYOUT, render_report

SCHEMA = "crossres-cached-probability-threshold-report-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _artifacts(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.name != "derivation.json"
    ]


def _validate_artifacts(root: Path, records: Sequence[dict[str, Any]]) -> None:
    expected = {str(record["path"]): record for record in records}
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.name != "derivation.json"
    }
    if not expected or set(expected) != set(actual):
        raise ValueError("cached-threshold report artifact inventory differs")
    for relative, record in expected.items():
        path = actual[relative]
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"cached-threshold artifact length changed: {relative}")
        if _sha256(path) != str(record["sha256"]).lower():
            raise ValueError(f"cached-threshold artifact hash changed: {relative}")


def request_identity(
    source: Path,
    source_inference: Path,
    cube_ids: Sequence[str],
    *,
    threshold: float,
    fixed_slices: Sequence[str],
) -> dict[str, Any]:
    probability = source_inference / "probability"
    provenance = source_inference / "provenance.json"
    return {
        "source_grid": str(source),
        "source_manifest_sha256": _sha256(source / "manifest.json"),
        "source_inference": str(source_inference),
        "source_inference_provenance_sha256": _sha256(provenance),
        "probability": [
            {
                "cube_id": cube_id,
                "bytes": (probability / f"{cube_id}.tif").stat().st_size,
                "sha256": _sha256(probability / f"{cube_id}.tif"),
            }
            for cube_id in cube_ids
        ],
        "threshold": threshold,
        "fixed_slices": list(fixed_slices),
        "layout": REVIEW_LAYOUT,
    }


def materialize_threshold_grid(
    source_inference: Path,
    output_inference: Path,
    cube_ids: Sequence[str],
    *,
    threshold: float,
) -> None:
    prediction = output_inference / "cubes_PRED"
    probability_output = output_inference / "probability"
    prediction.mkdir(parents=True)
    probability_output.mkdir()
    for cube_id in cube_ids:
        probability_path = source_inference / "probability" / f"{cube_id}.tif"
        values = np.asarray(tifffile.imread(probability_path))
        if values.ndim != 3 or not np.issubdtype(values.dtype, np.floating):
            raise ValueError(f"invalid probability cube: {probability_path}")
        if not bool(np.isfinite(values).all()) or bool(
            ((values < 0.0) | (values > 1.0)).any()
        ):
            raise ValueError(f"probability cube is not finite in [0, 1]: {cube_id}")
        tifffile.imwrite(
            prediction / f"{cube_id}.tif",
            (values >= threshold).astype(np.uint8) * 255,
            photometric="minisblack",
        )
        shutil.copy2(probability_path, probability_output / probability_path.name)
    (prediction / "present.json").write_text(
        json.dumps(list(cube_ids), indent=2) + "\n", encoding="utf-8"
    )
    source_provenance = _read_json(source_inference / "provenance.json")
    source_options = source_provenance.get("options") or {}
    if not isinstance(source_options, dict):
        raise TypeError("source probability provenance options must be an object")
    _atomic_json(
        output_inference / "provenance.json",
        {
            "schema": "crossres-derived-threshold-grid-v1",
            "kind": "crossres-derived-threshold-grid-v1",
            "threshold": threshold,
            "checkpoint": source_provenance.get("checkpoint"),
            "options": {
                **source_options,
                "threshold": threshold,
                "derived_from_cached_probability": True,
            },
            "source_probability_grid": str(source_inference),
            "source_probability_provenance": source_provenance,
        },
    )
    source_manifest = source_inference / "source_manifest.json"
    if source_manifest.is_file():
        shutil.copy2(source_manifest, output_inference / "source_manifest.json")


def generate(
    *,
    source_path: str | Path,
    source_inference_path: str | Path,
    output_path: str | Path,
    threshold: float,
    fixed_slices: Sequence[str] = (),
) -> tuple[Path, bool]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    source = Path(source_path).resolve()
    source_inference = Path(source_inference_path).resolve()
    output = Path(output_path).resolve()
    present = source_inference / "cubes_PRED" / "present.json"
    cube_ids = sorted({str(value) for value in _read_json(present)})
    if not cube_ids:
        raise ValueError("source inference has no cube IDs")
    identity = request_identity(
        source,
        source_inference,
        cube_ids,
        threshold=threshold,
        fixed_slices=fixed_slices,
    )
    if output.exists():
        derivation = _read_json(output / "derivation.json")
        if derivation.get("state") != "complete" or derivation.get("request") != identity:
            raise ValueError("existing cached-threshold report has a different identity")
        _validate_artifacts(output, derivation["artifacts"])
        report = output / str(derivation["html_report"])
        if not report.is_file():
            raise FileNotFoundError("cached-threshold HTML report is missing")
        return report, True

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".partial-{os.getpid()}")
    if temporary.exists():
        raise ValueError(f"stale cached-threshold temporary exists: {temporary}")
    temporary.mkdir()
    try:
        inference = temporary / "inference"
        materialize_threshold_grid(
            source_inference,
            inference,
            cube_ids,
            threshold=threshold,
        )
        report = render_report(
            source_path=source,
            student_path=inference,
            output_path=temporary / "report",
            cube_ids=list(cube_ids),
            slices_per_axis=3,
            student_identity_path=output / "inference",
            fixed_slices=tuple(fixed_slices),
            layout=REVIEW_LAYOUT,
        )
        derivation = {
            "schema": SCHEMA,
            "state": "complete",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "request": identity,
            "html_report": report.relative_to(temporary).as_posix(),
            "artifacts": _artifacts(temporary),
        }
        _atomic_json(temporary / "derivation.json", derivation)
        _validate_artifacts(temporary, derivation["artifacts"])
        os.replace(temporary, output)
    except BaseException:  # noqa: TRY203 - retain partial evidence for diagnosis
        raise
    return output / "report" / "index.html", False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-inference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--fixed-slices", nargs="*", default=[])
    args = parser.parse_args()
    report, reused = generate(
        source_path=args.source,
        source_inference_path=args.source_inference,
        output_path=args.output,
        threshold=args.threshold,
        fixed_slices=args.fixed_slices,
    )
    print(json.dumps({"report": str(report), "reused": reused}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
