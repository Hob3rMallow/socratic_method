#!/usr/bin/env python3
"""Generate and validate a provenance-bound PHerc1447 checkpoint report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from crossres_pred.voxel.grid_inference import infer_voxel_grid
from render_voxel_grid_report import RENDER_LAYOUTS, REVIEW_LAYOUT, render_report

REQUEST_SCHEMA = "crossres-checkpoint-report-request-v3"
GENERATION_SCHEMA = "crossres-checkpoint-report-generation-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing checkpoint-report file: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


@dataclass(frozen=True)
class CheckpointReportOptions:
    threshold: float = 0.5
    halo: int = 32
    device: str = "cuda"
    amp_dtype: str = "bfloat16"
    mirror_tta: bool = True
    max_cpu_threads: int = 16
    slices_per_axis: int = 3
    fixed_slices: tuple[str, ...] = ()
    layout: str = REVIEW_LAYOUT

    def validate(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.halo < 0:
            raise ValueError("halo must be non-negative")
        if self.amp_dtype not in {"bfloat16", "float16"}:
            raise ValueError("amp_dtype must be bfloat16 or float16")
        if not 1 <= self.max_cpu_threads <= 16:
            raise ValueError("max_cpu_threads must be in [1, 16]")
        if self.slices_per_axis <= 0:
            raise ValueError("slices_per_axis must be positive")
        if len(set(self.fixed_slices)) != len(self.fixed_slices):
            raise ValueError("fixed_slices must be unique")
        if self.layout not in RENDER_LAYOUTS:
            raise ValueError(f"layout must be one of {RENDER_LAYOUTS}")


def _source_cube_ids(source: Path) -> list[str]:
    value = json.loads(
        (source / "cubes_PRED" / "present.json").read_text(encoding="utf-8")
    )
    if not isinstance(value, list) or not value:
        raise ValueError(f"{source}: present.json must be a non-empty array")
    ids = sorted({str(item) for item in value})
    if any(not item for item in ids):
        raise ValueError(f"{source}: present.json contains an empty cube ID")
    return ids


def _source_records(source: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        resolved = path.resolve()
        if not resolved.is_relative_to(source):
            raise ValueError(f"source-grid artifact escapes its root: {path}")
        records.append(
            {
                "path": path.relative_to(source).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not records:
        raise ValueError(f"{source}: source grid contains no files")
    return records


def _request_identity(
    *,
    source: Path,
    checkpoint: Path,
    cube_ids: list[str],
    options: CheckpointReportOptions,
    reference_student: Path | None,
    jackpot_student: Path | None,
    surface_evidence: Path | None,
) -> dict[str, Any]:
    source_manifest = source / "manifest.json"
    present = source / "cubes_PRED" / "present.json"
    for path in (source_manifest, present, checkpoint):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(
                f"checkpoint-report input is missing or empty: {path}"
            )
    reference_identity = None
    if reference_student is not None:
        reference_present = reference_student / "cubes_PRED" / "present.json"
        if not reference_present.is_file():
            raise FileNotFoundError(
                f"incumbent student grid lacks present.json: {reference_present}"
            )
        reference_ids = set(json.loads(reference_present.read_text(encoding="utf-8")))
        missing = sorted(set(cube_ids) - reference_ids)
        if missing:
            raise ValueError(
                f"incumbent student grid is missing requested cubes: {missing}"
            )
        reference_identity = {
            "path": str(reference_student),
            "artifacts": _source_records(reference_student),
        }
    jackpot_identity = None
    if jackpot_student is not None:
        jackpot_present = jackpot_student / "cubes_PRED" / "present.json"
        if not jackpot_present.is_file():
            raise FileNotFoundError(
                f"jackpot student grid lacks present.json: {jackpot_present}"
            )
        jackpot_ids = set(json.loads(jackpot_present.read_text(encoding="utf-8")))
        missing = sorted(set(cube_ids) - jackpot_ids)
        if missing:
            raise ValueError(
                f"jackpot student grid is missing requested cubes: {missing}"
            )
        jackpot_identity = {
            "path": str(jackpot_student),
            "artifacts": _source_records(jackpot_student),
        }
    surface_identity = None
    if surface_evidence is not None:
        for name in ("x.tif", "y.tif", "z.tif"):
            path = surface_evidence / name
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(
                    f"surface evidence lacks a non-empty {name}: {path}"
                )
        surface_identity = {
            "path": str(surface_evidence),
            "artifacts": _source_records(surface_evidence),
        }
    identity = {
        "schema": REQUEST_SCHEMA,
        "source_grid": str(source),
        "source_manifest_sha256": _sha256(source_manifest),
        "source_present_sha256": _sha256(present),
        "source_artifacts": _source_records(source),
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha256(checkpoint),
        },
        "cube_ids": cube_ids,
        "options": json.loads(json.dumps(asdict(options))),
        "reference_student": reference_identity,
    }
    # Preserve the identity of existing v3 reports when this optional visual
    # comparison is absent.
    if jackpot_identity is not None:
        identity["jackpot_student"] = jackpot_identity
    if surface_identity is not None:
        identity["surface_evidence"] = surface_identity
    return identity


def _artifact_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "generation.json":
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not records:
        raise ValueError("checkpoint report produced no artifacts")
    return records


def _validate_artifacts(root: Path, records: Any) -> None:
    if not isinstance(records, list) or not records:
        raise ValueError("checkpoint-report artifact inventory is missing")
    recorded_paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("checkpoint-report artifact row must be an object")
        relative = str(record.get("path", ""))
        path = (root / relative).resolve()
        if not relative or relative in recorded_paths or not path.is_relative_to(root):
            raise ValueError(f"invalid checkpoint-report artifact path: {relative!r}")
        recorded_paths.add(relative)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint-report artifact is missing: {path}")
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise ValueError(f"checkpoint-report artifact length changed: {path}")
        if _sha256(path) != str(record.get("sha256", "")).lower():
            raise ValueError(f"checkpoint-report artifact hash changed: {path}")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "generation.json"
    }
    if actual_paths != recorded_paths:
        raise ValueError("checkpoint-report artifact inventory differs from disk")


def validate_checkpoint_report(
    *,
    output_path: str | Path,
    expected_identity: dict[str, Any] | None = None,
) -> Path:
    output = Path(output_path).expanduser().resolve()
    generation = _read_object(output / "generation.json")
    if generation.get("schema") != GENERATION_SCHEMA:
        raise ValueError(f"{output}: invalid checkpoint-report generation schema")
    if generation.get("state") != "complete":
        raise ValueError(f"{output}: checkpoint report is not complete")
    if expected_identity is not None and generation.get("request") != expected_identity:
        raise ValueError(
            f"{output}: checkpoint-report request identity differs from this invocation"
        )
    _validate_artifacts(output, generation.get("artifacts"))
    report = (output / str(generation.get("html_report", ""))).resolve()
    if not report.is_file() or not report.is_relative_to(output):
        raise FileNotFoundError(f"{output}: generated HTML report is missing")
    return report


def generate_checkpoint_report(
    *,
    source_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    cube_ids: list[str] | None = None,
    options: CheckpointReportOptions | None = None,
    reference_student_path: str | Path | None = None,
    jackpot_student_path: str | Path | None = None,
    surface_evidence_path: str | Path | None = None,
) -> tuple[Path, bool]:
    options = options or CheckpointReportOptions()
    options.validate()
    source = Path(source_path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    reference_student = (
        Path(reference_student_path).expanduser().resolve()
        if reference_student_path is not None
        else None
    )
    jackpot_student = (
        Path(jackpot_student_path).expanduser().resolve()
        if jackpot_student_path is not None
        else None
    )
    surface_evidence = (
        Path(surface_evidence_path).expanduser().resolve()
        if surface_evidence_path is not None
        else None
    )
    available_ids = _source_cube_ids(source)
    selected_ids = sorted(set(cube_ids or available_ids))
    if not selected_ids:
        raise ValueError("checkpoint report requires at least one cube")
    missing = sorted(set(selected_ids) - set(available_ids))
    if missing:
        raise ValueError(f"requested cubes are absent from source grid: {missing}")
    identity = _request_identity(
        source=source,
        checkpoint=checkpoint,
        cube_ids=selected_ids,
        options=options,
        reference_student=reference_student,
        jackpot_student=jackpot_student,
        surface_evidence=surface_evidence,
    )

    if output.exists():
        return (
            validate_checkpoint_report(
                output_path=output,
                expected_identity=identity,
            ),
            True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".partial-{os.getpid()}")
    if temporary.exists():
        raise ValueError(f"stale checkpoint-report temporary exists: {temporary}")
    temporary.mkdir()
    inference = temporary / "inference"
    report_root = temporary / "report"
    try:
        infer_voxel_grid(
            source_grid=source,
            checkpoint_path=checkpoint,
            output_path=inference,
            threshold=options.threshold,
            halo=options.halo,
            device_name=options.device,
            amp_dtype_name=options.amp_dtype,
            mirror_tta=options.mirror_tta,
            max_cpu_threads=options.max_cpu_threads,
            target_cube_ids=selected_ids,
        )
        report = render_report(
            source_path=source,
            student_path=inference,
            output_path=report_root,
            cube_ids=selected_ids,
            slices_per_axis=options.slices_per_axis,
            student_identity_path=output / "inference",
            fixed_slices=options.fixed_slices,
            reference_student_path=reference_student,
            jackpot_student_path=jackpot_student,
            surface_evidence_path=surface_evidence,
            layout=options.layout,
        )
        generation = {
            "schema": GENERATION_SCHEMA,
            "state": "complete",
            "request": identity,
            "grid_provenance": "inference/provenance.json",
            "visual_report": "report/report.json",
            "html_report": report.relative_to(temporary).as_posix(),
            "artifacts": _artifact_records(temporary),
        }
        _atomic_json(temporary / "generation.json", generation)
        validate_checkpoint_report(
            output_path=temporary,
            expected_identity=identity,
        )
        os.replace(temporary, output)
    except BaseException:  # noqa: TRY203 - preserve partial evidence on failure
        # Preserve a non-empty partial directory for diagnosis and explicit
        # operator recovery; never silently bless or overwrite it.
        raise
    return (
        validate_checkpoint_report(
            output_path=output,
            expected_identity=identity,
        ),
        False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cube-id", action="append", default=[])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--halo", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-cpu-threads", type=int, default=16)
    parser.add_argument("--slices-per-axis", type=int, default=3)
    parser.add_argument("--layout", choices=RENDER_LAYOUTS, default=REVIEW_LAYOUT)
    parser.add_argument(
        "--fixed-slice",
        action="append",
        default=[],
        help="always include CUBE_ID:AXIS:GLOBAL_COORDINATE",
    )
    parser.add_argument(
        "--reference-student",
        type=Path,
        help="provenance-bound incumbent grid for blind anti-blob regression flags",
    )
    parser.add_argument(
        "--jackpot-student",
        type=Path,
        help="provenance-bound original jackpot grid for visual comparison",
    )
    parser.add_argument(
        "--surface-evidence",
        type=Path,
        help="optional official TIFXYZ directory for sparse visual comparison",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    report, reused = generate_checkpoint_report(
        source_path=args.source,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        cube_ids=args.cube_id,
        options=CheckpointReportOptions(
            threshold=args.threshold,
            halo=args.halo,
            device=args.device,
            amp_dtype=args.amp_dtype,
            mirror_tta=args.tta,
            max_cpu_threads=args.max_cpu_threads,
            slices_per_axis=args.slices_per_axis,
            fixed_slices=tuple(args.fixed_slice),
            layout=args.layout,
        ),
        reference_student_path=args.reference_student,
        jackpot_student_path=args.jackpot_student,
        surface_evidence_path=args.surface_evidence,
    )
    print(json.dumps({"report": str(report), "reused": reused}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
