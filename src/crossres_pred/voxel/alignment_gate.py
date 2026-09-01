from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_alignment_metadata_gate(
    specification: dict[str, Any], *, base: Path
) -> dict[str, Any]:
    """Validate a passed alignment-metadata audit and all pinned inputs."""

    path = Path(str(specification["path"])).expanduser()
    path = (path if path.is_absolute() else base / path).resolve()
    if not path.is_file():
        raise RuntimeError(f"required alignment-metadata gate is missing: {path}")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(gate, dict):
        raise TypeError(f"{path}: expected an object")
    if (
        gate.get("schema") != "crossres-alignment-metadata-gate-v1"
        or gate.get("state") != "passed"
    ):
        raise RuntimeError(f"alignment-metadata gate is not passed: {path}")
    for key in (
        "config_sha256",
        "catalog_sha256",
        "pair_manifest_sha256",
        "pair_count",
    ):
        expected = specification.get(key)
        if expected is not None and gate.get(key) != expected:
            raise RuntimeError(
                f"alignment-metadata gate {key} changed: expected {expected!r}, "
                f"got {gate.get(key)!r}"
            )
    report = Path(str(gate["report"])).expanduser().resolve()
    if not report.is_file() or _sha256(report) != str(gate["report_sha256"]):
        raise RuntimeError("alignment-metadata gate report is missing or changed")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "report": str(report),
        "report_sha256": gate["report_sha256"],
        "config_sha256": gate["config_sha256"],
        "catalog_sha256": gate["catalog_sha256"],
        "pair_manifest_sha256": gate["pair_manifest_sha256"],
        "pair_count": gate["pair_count"],
    }
