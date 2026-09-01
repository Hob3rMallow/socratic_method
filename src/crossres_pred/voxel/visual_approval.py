from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VISUAL_APPROVAL_SCHEMA = "crossres-pherc1447-visual-approval-v1"
LEGACY_GRID_KIND = "crossres-grid-inference"
VOXEL_GRID_KIND = "crossres-voxel-grid-inference-v1"
VOXEL_STUDENT_REPORT_SCHEMA = "crossres-pherc1447-voxel-student-report-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
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
        raise FileNotFoundError(f"missing visual-audit evidence: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _recorded_path(value: Any, source: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = source.parent / path
    return path.resolve()


def _evidence_file(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"visual evidence escapes report directory: {resolved}")
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"missing or empty visual evidence: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _build_identity(
    *,
    grid_provenance_path: Path,
    visual_audit_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    grid_provenance = grid_provenance_path.resolve()
    visual_audit = visual_audit_path.resolve()
    checkpoint = checkpoint_path.resolve()
    provenance = _read_object(grid_provenance)
    audit = _read_object(visual_audit)

    grid_kind = str(provenance.get("kind", ""))
    grid_cube_ids: list[str] | None = None
    grid_options: dict[str, Any] | None = None
    if grid_kind == VOXEL_GRID_KIND:
        if provenance.get("schema") != VOXEL_GRID_KIND:
            raise ValueError(f"{grid_provenance}: invalid voxel-grid schema")
        raw_cube_ids = provenance.get("target_cube_ids")
        if not isinstance(raw_cube_ids, list) or not raw_cube_ids:
            raise ValueError(f"{grid_provenance}: target cube IDs are missing")
        grid_cube_ids = [str(value) for value in raw_cube_ids]
        if any(not value for value in grid_cube_ids) or len(set(grid_cube_ids)) != len(
            grid_cube_ids
        ):
            raise ValueError(f"{grid_provenance}: target cube IDs are invalid")
        present_path = grid_provenance.parent / "cubes_PRED" / "present.json"
        try:
            present_cube_ids = json.loads(present_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"missing completed-grid inventory: {present_path}"
            ) from error
        if present_cube_ids != grid_cube_ids:
            raise ValueError(
                f"{present_path}: completed-grid inventory differs from provenance"
            )
        grid_options = provenance.get("options")
        if not isinstance(grid_options, dict):
            raise TypeError(f"{grid_provenance}: inference options are missing")
        if provenance.get("research_only") is not True or provenance.get(
            "deployment_ready"
        ) is not False:
            raise ValueError(f"{grid_provenance}: invalid research-only contract")
        grid_cube_count = len(grid_cube_ids)
    elif grid_kind == LEGACY_GRID_KIND:
        if provenance.get("status") != "complete":
            raise ValueError(f"{grid_provenance}: grid inference is not complete")
        summary = provenance.get("summary")
        if not isinstance(summary, dict) or summary.get("status") != "complete":
            raise ValueError(f"{grid_provenance}: grid summary is not complete")
        grid_cube_count = int(summary.get("cube_count", -1))
    else:
        raise ValueError(f"{grid_provenance}: not a grid-inference provenance file")
    checkpoint_record = provenance.get("checkpoint")
    if not isinstance(checkpoint_record, dict):
        raise TypeError(f"{grid_provenance}: checkpoint provenance is missing")
    recorded_checkpoint = _recorded_path(checkpoint_record.get("path"), grid_provenance)
    if recorded_checkpoint != checkpoint:
        raise ValueError(
            f"grid checkpoint {recorded_checkpoint} does not match {checkpoint}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"visual checkpoint is missing: {checkpoint}")
    checkpoint_digest = _sha256(checkpoint)
    if str(checkpoint_record.get("sha256", "")).lower() != checkpoint_digest:
        raise ValueError(f"{grid_provenance}: checkpoint digest changed")

    pretty_voxel_report = (
        str(audit.get("schema", "")) == VOXEL_STUDENT_REPORT_SCHEMA
    )
    corrected_grid = _recorded_path(
        audit.get("student_grid") if pretty_voxel_report else audit.get("corrected_grid"),
        visual_audit,
    )
    if corrected_grid != grid_provenance.parent:
        raise ValueError(
            f"visual report uses {corrected_grid}, not grid {grid_provenance.parent}"
        )
    audit_checkpoint = audit.get("checkpoint")
    if not isinstance(audit_checkpoint, dict):
        raise TypeError(f"{visual_audit}: checkpoint provenance is missing")
    if str(audit_checkpoint.get("sha256", "")).lower() != checkpoint_digest:
        raise ValueError(f"{visual_audit}: checkpoint digest does not match the grid")

    source_grid = _recorded_path(provenance.get("source_grid"), grid_provenance)
    audit_source_grid = _recorded_path(audit.get("source_grid"), visual_audit)
    if source_grid != audit_source_grid:
        raise ValueError("grid inference and visual report use different source grids")

    rows = audit.get("cubes")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{visual_audit}: visual report contains no slices")
    cube_ids: list[str] = []
    relative_evidence: list[str] = []
    slice_count = 0
    for row in rows:
        if not isinstance(row, dict) or not row.get("cube_id"):
            raise ValueError(f"{visual_audit}: invalid cube audit row")
        cube_ids.append(str(row["cube_id"]))
        if pretty_voxel_report:
            views = row.get("views")
            if not isinstance(views, list) or not views:
                raise ValueError(f"{visual_audit}: cube has no rendered views")
            for view in views:
                if not isinstance(view, dict):
                    raise TypeError(f"{visual_audit}: invalid rendered view")
                relative_evidence.append(str(view.get("image", "")))
            slice_count += len(views)
        else:
            relative_evidence.append(str(row.get("comparison_image", "")))
            slice_count += 1
    unique_cube_count = int(audit.get("unique_cube_count", len(set(cube_ids))))
    if unique_cube_count <= 0 or unique_cube_count != len(set(cube_ids)):
        raise ValueError(f"{visual_audit}: inconsistent unique cube count")
    if grid_cube_count != unique_cube_count:
        raise ValueError("grid inference and visual report cube counts differ")
    if grid_cube_ids is not None and set(grid_cube_ids) != set(cube_ids):
        raise ValueError("grid inference and visual report cube IDs differ")
    if grid_kind == VOXEL_GRID_KIND:
        audit_options = (
            audit.get("inference_options")
            if pretty_voxel_report
            else audit.get("corrected_options")
        )
        if (
            not pretty_voxel_report
            and audit.get("corrected_provenance_kind") != grid_kind
        ):
            raise ValueError("visual report does not identify the voxel-grid provenance")
        if pretty_voxel_report and (
            audit.get("research_only") is not True
            or audit.get("quality_claim") is not False
        ):
            raise ValueError("voxel-student report lacks its research-only contract")
        if audit_options != grid_options:
            raise ValueError("grid inference and visual report options differ")

    if pretty_voxel_report:
        relative_evidence.append("index.html")
    else:
        relative_evidence.extend(
            [str(audit.get("html_report", "")), str(audit.get("atlas_image", ""))]
        )
    report_root = visual_audit.parent
    evidence_files = []
    for relative in sorted(set(relative_evidence)):
        if not relative:
            raise ValueError(f"{visual_audit}: visual evidence filename is missing")
        evidence_files.append(_evidence_file(report_root / relative, report_root))

    aggregate = audit.get("aggregate")
    if pretty_voxel_report and not isinstance(aggregate, dict):
        raise TypeError(f"{visual_audit}: aggregate comparison is missing")
    added_voxels = (
        int(aggregate.get("student_only", 0))
        if pretty_voxel_report
        else int(audit.get("total_added_voxels", 0))
    )
    removed_voxels = (
        int(aggregate.get("published_only", 0))
        if pretty_voxel_report
        else int(audit.get("total_removed_voxels", 0))
    )

    return {
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_digest,
        },
        "grid_inference": {
            "path": str(grid_provenance),
            "sha256": _sha256(grid_provenance),
            "kind": grid_kind,
            "source_grid": str(source_grid),
            "corrected_grid": str(corrected_grid),
            "cube_count": unique_cube_count,
            "cube_ids": sorted(set(cube_ids)),
            "options": grid_options,
        },
        "visual_evidence": {
            "audit_path": str(visual_audit),
            "audit_sha256": _sha256(visual_audit),
            "slice_count": slice_count,
            "cube_ids": sorted(set(cube_ids)),
            "total_added_voxels": added_voxels,
            "total_removed_voxels": removed_voxels,
            "files": evidence_files,
        },
    }


def record_visual_approval(
    *,
    grid_provenance_path: str | Path,
    visual_audit_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    decision: str,
    reviewer: str,
    notes: str = "",
) -> Path:
    """Record an immutable human/agent decision over hashed visual evidence."""

    normalized_decision = decision.strip().lower()
    if normalized_decision not in {"approved", "rejected"}:
        raise ValueError("decision must be 'approved' or 'rejected'")
    normalized_reviewer = reviewer.strip()
    if not normalized_reviewer:
        raise ValueError("reviewer cannot be empty")
    destination = Path(output_path).expanduser().resolve()
    identity = _build_identity(
        grid_provenance_path=Path(grid_provenance_path).expanduser().resolve(),
        visual_audit_path=Path(visual_audit_path).expanduser().resolve(),
        checkpoint_path=Path(checkpoint_path).expanduser().resolve(),
    )
    stable = {
        "schema": VISUAL_APPROVAL_SCHEMA,
        "state": f"visual-{normalized_decision}",
        "decision": normalized_decision,
        "reviewer": normalized_reviewer,
        "notes": notes.strip(),
        "identity": identity,
    }
    if destination.is_file():
        existing = _read_object(destination)
        comparable = {key: existing.get(key) for key in stable}
        if comparable != stable:
            raise ValueError(
                f"{destination}: visual decision or evidence identity changed"
            )
        return destination
    _atomic_json(
        destination,
        {**stable, "created_at": datetime.now(UTC).isoformat()},
    )
    return destination


def validate_visual_evidence(
    *,
    grid_provenance_path: str | Path,
    visual_audit_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Validate production visual evidence before soliciting a decision."""

    return _build_identity(
        grid_provenance_path=Path(grid_provenance_path).expanduser().resolve(),
        visual_audit_path=Path(visual_audit_path).expanduser().resolve(),
        checkpoint_path=Path(checkpoint_path).expanduser().resolve(),
    )


def validate_visual_approval(
    approval_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    require_approved: bool = True,
) -> dict[str, Any]:
    """Re-hash every recorded artifact and validate an immutable decision."""

    source = Path(approval_path).expanduser().resolve()
    approval = _read_object(source)
    if approval.get("schema") != VISUAL_APPROVAL_SCHEMA:
        raise ValueError(f"{source}: invalid visual-approval schema")
    decision = str(approval.get("decision", ""))
    if require_approved and decision != "approved":
        raise RuntimeError(f"{source}: PHerc1447 visual evidence was not approved")
    identity = approval.get("identity")
    if not isinstance(identity, dict):
        raise TypeError(f"{source}: visual-approval identity is missing")
    try:
        recorded_checkpoint = Path(identity["checkpoint"]["path"]).resolve()
        grid_provenance = Path(identity["grid_inference"]["path"]).resolve()
        visual_audit = Path(identity["visual_evidence"]["audit_path"]).resolve()
    except (KeyError, TypeError) as error:
        raise ValueError(f"{source}: incomplete visual-approval identity") from error
    expected_checkpoint = (
        Path(checkpoint_path).expanduser().resolve()
        if checkpoint_path is not None
        else recorded_checkpoint
    )
    if recorded_checkpoint != expected_checkpoint:
        raise ValueError(
            f"approved checkpoint {recorded_checkpoint} does not match "
            f"{expected_checkpoint}"
        )
    current_identity = _build_identity(
        grid_provenance_path=grid_provenance,
        visual_audit_path=visual_audit,
        checkpoint_path=expected_checkpoint,
    )
    if current_identity != identity:
        raise ValueError(f"{source}: approved visual evidence changed")
    return approval
