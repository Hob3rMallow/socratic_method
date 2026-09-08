from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

AUDIT_SCHEMA = "crossres-official-catalog-maximality-audit-v1"
MAX_CATALOG_BYTES = 64 * 1024 * 1024


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(value: bytes, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label}: invalid UTF-8 JSON") from error
    if not isinstance(parsed, dict):
        raise TypeError(f"{label}: expected a JSON object")
    return parsed


def load_catalog_source(
    *, catalog_path: str | Path | None = None, catalog_url: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load an official catalog from exactly one local or HTTPS source."""

    if (catalog_path is None) == (catalog_url is None):
        raise ValueError("provide exactly one of catalog_path or catalog_url")
    if catalog_path is not None:
        source = Path(catalog_path).expanduser().resolve()
        payload = source.read_bytes()
        label = str(source)
        provenance = {"kind": "file", "value": label}
    else:
        assert catalog_url is not None
        if not catalog_url.startswith("https://"):
            raise ValueError("catalog_url must use HTTPS")
        request = urllib.request.Request(
            catalog_url,
            headers={"User-Agent": "vesuvius-crossres-pred/catalog-audit-v1"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read(MAX_CATALOG_BYTES + 1)
        label = catalog_url
        provenance = {"kind": "url", "value": catalog_url}
    if not payload or len(payload) > MAX_CATALOG_BYTES:
        raise ValueError(f"{label}: empty or oversized catalog")
    provenance.update({"bytes": len(payload), "sha256": _sha256_bytes(payload)})
    return _read_json_object(payload, label), provenance


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _scan_inventory(
    catalog: dict[str, Any],
    *,
    fine_max_um: float,
    coarse_min_um: float,
    coarse_max_um: float,
) -> dict[str, dict[str, Any]]:
    scrolls = catalog.get("scrolls")
    if not isinstance(scrolls, list) or not scrolls:
        raise ValueError("official catalog contains no scrolls")
    inventory: dict[str, dict[str, Any]] = {}
    for position, value in enumerate(scrolls):
        if not isinstance(value, dict):
            raise TypeError(f"catalog scroll {position} is not an object")
        sample_id = str(value.get("id", ""))
        if not sample_id or sample_id in inventory:
            raise ValueError(f"missing or duplicate catalog sample {sample_id!r}")
        scans = value.get("scans") or []
        if not isinstance(scans, list):
            raise TypeError(f"{sample_id}: scans must be a list")
        scan_px = sorted(
            {
                float(scan["px"])
                for scan in scans
                if isinstance(scan, dict) and scan.get("px") is not None
            }
        )
        if any(not (0.0 < px < 1000.0) for px in scan_px):
            raise ValueError(f"{sample_id}: invalid scan pixel size")
        fine_px = [px for px in scan_px if px <= fine_max_um]
        coarse_px = [px for px in scan_px if coarse_min_um <= px <= coarse_max_um]
        predictions = value.get("predictions") or []
        surface_prediction_px = sorted(
            {
                float(prediction["px"])
                for prediction in predictions
                if isinstance(prediction, dict)
                and prediction.get("purpose") == "surface-prediction"
                and prediction.get("px") is not None
            }
        )
        inventory[sample_id] = {
            "sample_id": sample_id,
            "label": str(value.get("label", sample_id)),
            "type": str(value.get("type", "")),
            "scan_px_um": scan_px,
            "fine_px_um": fine_px,
            "coarse_px_um": coarse_px,
            "surface_prediction_px_um": surface_prediction_px,
            "segments": int(value.get("n_segments") or 0),
            "has_fine": bool(fine_px),
            "has_deployment_coarse": bool(coarse_px),
        }
    return inventory


def _read_jsonl_index(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: row is not an object")
            record_id = str(value.get("record_id", ""))
            if not record_id or record_id in rows:
                raise ValueError(
                    f"{path}:{line_number}: missing or duplicate record_id"
                )
            rows[record_id] = value
    return rows


def _plan_record_inventory(
    plan: dict[str, Any], plan_path: Path
) -> dict[str, list[dict[str, Any]]]:
    values = plan.get("records")
    if not isinstance(values, list) or not values:
        raise ValueError("corpus plan contains no records")
    manifest_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    manifest_hashes: dict[Path, str] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    for position, summary in enumerate(values):
        if not isinstance(summary, dict):
            raise TypeError(f"plan record {position} is not an object")
        record_id = str(summary.get("record_id", ""))
        sample_id = str(summary.get("scroll_id", ""))
        source_value = str(summary.get("source_manifest", ""))
        if not record_id or not sample_id or not source_value:
            raise ValueError(f"plan record {position} is incomplete")
        source = Path(source_value).expanduser()
        if not source.is_absolute():
            source = plan_path.parent / source
        source = source.resolve()
        expected_hash = str(summary.get("source_manifest_sha256", ""))
        if source not in manifest_cache:
            manifest_hashes[source] = _sha256(source)
            manifest_cache[source] = _read_jsonl_index(source)
        if expected_hash and manifest_hashes[source] != expected_hash:
            raise ValueError(f"{source}: hash no longer matches the corpus plan")
        try:
            row = manifest_cache[source][record_id]
        except KeyError as error:
            raise ValueError(f"{source}: missing planned record {record_id}") from error
        if str(row.get("scroll_id", "")) != sample_id:
            raise ValueError(f"{record_id}: plan/source scroll mismatch")
        coarse = row.get("coarse") or {}
        fine = row.get("fine") or {}
        coarse_scan = str(coarse.get("scan_id", ""))
        fine_scan = str(fine.get("scan_id", ""))
        record = {
            "record_id": record_id,
            "split": str(summary.get("split", "")),
            "category": str(summary.get("category", "")),
            "patch_count": int(summary.get("patch_count", 0)),
            "supervision_source": str(row.get("supervision_source", "")),
            "coarse_scan_id": coarse_scan,
            "fine_scan_id": fine_scan,
            "coarse_voxel_um": float(coarse.get("voxel_um", 0.0)),
            "fine_voxel_um": float(fine.get("voxel_um", 0.0)),
            "cross_scan": bool(coarse_scan and fine_scan and coarse_scan != fine_scan),
        }
        records.setdefault(sample_id, []).append(record)
    return records


def audit_catalog_maximality(
    *,
    catalog: dict[str, Any],
    catalog_provenance: dict[str, Any],
    plan_path: str | Path,
    output_path: str | Path,
    source_commit: str,
    expected_paired_holdouts: set[str],
    required_catalog_samples: set[str] | None = None,
    fine_max_um: float = 3.5,
    coarse_min_um: float = 7.0,
    coarse_max_um: float = 12.0,
) -> Path:
    """Prove that every official fine-space source is used or deliberately sealed."""

    if not source_commit:
        raise ValueError("source_commit is required")
    if not (0.0 < fine_max_um < coarse_min_um < coarse_max_um):
        raise ValueError("invalid fine/coarse resolution thresholds")
    plan_source = Path(plan_path).expanduser().resolve()
    plan = _read_json_object(plan_source.read_bytes(), str(plan_source))
    inventory = _scan_inventory(
        catalog,
        fine_max_um=fine_max_um,
        coarse_min_um=coarse_min_um,
        coarse_max_um=coarse_max_um,
    )
    required = required_catalog_samples or set()
    unknown_required = sorted(required - inventory.keys())
    if unknown_required:
        raise ValueError(f"required catalog samples are absent: {unknown_required}")

    records = _plan_record_inventory(plan, plan_source)
    planned = set(records)
    holdouts = {str(value) for value in plan.get("holdout_scrolls", [])}
    collision = sorted(planned & holdouts)
    if collision:
        raise ValueError(f"holdouts entered the corpus plan: {collision}")
    unknown_planned = sorted(planned - inventory.keys())
    if unknown_planned:
        raise ValueError(
            f"planned samples absent from official catalog: {unknown_planned}"
        )

    fine_samples = {
        sample_id for sample_id, item in inventory.items() if item["has_fine"]
    }
    paired = {
        sample_id
        for sample_id in fine_samples
        if inventory[sample_id]["has_deployment_coarse"]
    }
    fine_only = fine_samples - paired
    missing = sorted(fine_samples - planned - holdouts)
    if missing:
        raise ValueError(f"official fine-space sources are unaccounted for: {missing}")

    actual_paired_holdouts = paired & holdouts
    if actual_paired_holdouts != expected_paired_holdouts:
        raise ValueError(
            "paired holdout mismatch: "
            f"{sorted(actual_paired_holdouts)} != {sorted(expected_paired_holdouts)}"
        )
    missing_cross_scan = sorted(
        sample_id
        for sample_id in paired & planned
        if not any(record["cross_scan"] for record in records[sample_id])
    )
    if missing_cross_scan:
        raise ValueError(
            "paired samples have no actual cross-scan plan record: "
            f"{missing_cross_scan}"
        )

    def sample_rows(sample_ids: set[str]) -> list[dict[str, Any]]:
        return [
            {
                **inventory[sample_id],
                "disposition": (
                    "represented" if sample_id in planned else "sealed-holdout"
                ),
                "records": records.get(sample_id, []),
            }
            for sample_id in sorted(sample_ids)
        ]

    excluded = set(inventory) - fine_samples
    report = {
        "schema": AUDIT_SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "maximal": True,
        "definition": {
            "fine_max_um": fine_max_um,
            "deployment_coarse_min_um": coarse_min_um,
            "deployment_coarse_max_um": coarse_max_um,
            "require_actual_cross_scan_for_represented_paired_sample": True,
            "accounting_rule": "every official fine sample is represented or sealed",
        },
        "official_catalog": {
            **catalog_provenance,
            "source_commit": source_commit,
            "catalog_updated": catalog.get("updated"),
            "samples": len(inventory),
        },
        "corpus_plan": {
            "path": str(plan_source),
            "sha256": _sha256(plan_source),
            "records": sum(len(value) for value in records.values()),
            "represented_samples": sorted(planned),
            "declared_holdouts": sorted(holdouts),
        },
        "counts": {
            "official_samples": len(inventory),
            "official_fine_samples": len(fine_samples),
            "native_paired_candidates": len(paired),
            "fine_only_candidates": len(fine_only),
            "represented_fine_samples": len(fine_samples & planned),
            "sealed_fine_samples": len(fine_samples & holdouts),
            "excluded_without_fine_scan": len(excluded),
            "unaccounted_fine_samples": 0,
        },
        "native_paired_candidates": sample_rows(paired),
        "fine_only_candidates": sample_rows(fine_only),
        "excluded_without_fine_scan": [inventory[value] for value in sorted(excluded)],
        "required_catalog_samples": sorted(required),
        "expected_paired_holdouts": sorted(expected_paired_holdouts),
    }
    destination = Path(output_path).expanduser().resolve()
    _atomic_json(destination, report)
    return destination
