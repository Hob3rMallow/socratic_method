from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .visual_approval import validate_visual_approval

SEAL_SCHEMA = "crossres-voxel-sealed-test-v1"
QUALIFICATION_SCHEMA = "crossres-voxel-tuning-qualification-v3"
QUALIFICATION_SCHEMA_V4 = "crossres-voxel-tuning-qualification-v4"
QUALIFICATION_SCHEMAS = frozenset((QUALIFICATION_SCHEMA, QUALIFICATION_SCHEMA_V4))
AUDIT_SCHEMA = "crossres-voxel-checkpoint-audit-v3"
REQUIRED_BASELINE_SCROLLS = ("PHerc0500P2", "PHerc0814", "PHerc1451")


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


def _manifest_inventory(path: Path) -> tuple[int, Counter[str], dict[str, set[str]]]:
    count = 0
    splits: Counter[str] = Counter()
    scrolls: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                split = str(row["split"])
                scroll = str(row["scroll_id"])
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{line_number}: invalid patch row") from error
            count += 1
            splits[split] += 1
            scrolls.setdefault(split, set()).add(scroll)
    if count == 0:
        raise ValueError(f"{path}: empty patch manifest")
    return count, splits, scrolls


def _history_epochs(path: Path) -> tuple[list[int], int]:
    epochs: list[int] = []
    rows = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                epoch = int(json.loads(line)["epoch"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid history row"
                ) from error
            epochs.append(epoch)
            rows += 1
    return epochs, rows


def _same_float(left: object, right: object) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-9)


def _validate_qualification_audit(
    qualification: dict[str, Any],
    *,
    tuning_digest: str,
    patch_digest: str,
) -> tuple[Path, str, dict[str, Any]]:
    audit_source = Path(str(qualification["audit_report"])).expanduser().resolve()
    if not audit_source.is_file():
        raise FileNotFoundError(f"qualification audit is missing: {audit_source}")
    audit_digest = _sha256(audit_source)
    if audit_digest.lower() != str(qualification["audit_report_sha256"]).lower():
        raise ValueError("qualification audit digest changed")
    if patch_digest.lower() != str(qualification["patch_manifest_sha256"]).lower():
        raise ValueError("qualification patch-manifest digest changed")
    audit = json.loads(audit_source.read_text(encoding="utf-8"))
    if audit.get("schema") != AUDIT_SCHEMA:
        raise ValueError(f"{audit_source}: invalid checkpoint-audit schema")
    if str(audit["checkpoint"]["sha256"]).lower() != tuning_digest.lower():
        raise ValueError("qualification audit checkpoint digest changed")
    if str(audit["patch_manifest"]["sha256"]).lower() != patch_digest.lower():
        raise ValueError("qualification audit patch-manifest digest changed")

    options = audit.get("options")
    policy = qualification.get("inference_policy")
    if not isinstance(options, dict) or not isinstance(policy, dict):
        raise TypeError("qualification audit/policy options must be objects")
    required_policy = {
        "split": "val",
        "amp_dtype": "bfloat16",
        "mirror_tta": True,
        "qualification_scroll": "PHerc0814",
    }
    for name, expected in required_policy.items():
        if options.get(name) != expected or policy.get(name) != expected:
            raise ValueError(
                f"qualification inference policy {name!r} is not {expected!r}"
            )
    audit_thresholds = [float(value) for value in options.get("thresholds", [])]
    policy_thresholds = [float(value) for value in policy.get("thresholds", [])]
    if not audit_thresholds or audit_thresholds != policy_thresholds:
        raise ValueError("qualification threshold-grid policy changed")

    sweep = audit.get("sweep")
    if not isinstance(sweep, dict) or not bool(sweep.get("any_qualified")):
        raise RuntimeError("qualification audit contains no deployable threshold")
    audit_required_scrolls = tuple(sweep.get("required_baseline_scrolls", ()))
    policy_required_scrolls = tuple(policy.get("required_baseline_scrolls", ()))
    if (
        audit_required_scrolls != REQUIRED_BASELINE_SCROLLS
        or policy_required_scrolls != REQUIRED_BASELINE_SCROLLS
        or sweep.get("selection_metric") != "macro_scroll_dice"
        or policy.get("selection_metric") != "macro_scroll_dice"
        or sweep.get("baseline_comparison_policy") != "matched-rows-only"
        or policy.get("baseline_comparison_policy") != "matched-rows-only"
    ):
        raise ValueError("qualification source-balanced selection policy changed")
    selected = sweep.get("selected")
    if not isinstance(selected, dict) or not bool(selected.get("qualified")):
        raise RuntimeError("qualification selected an unqualified threshold")
    if not _same_float(selected["threshold"], qualification["threshold"]):
        raise ValueError("qualification threshold differs from its audit")
    if not any(
        _same_float(value, qualification["threshold"]) for value in audit_thresholds
    ):
        raise ValueError("qualification threshold is absent from its audit grid")
    comparison = selected.get("baseline_comparison")
    scrolls = selected.get("scrolls")
    if not isinstance(comparison, dict) or not isinstance(scrolls, dict):
        raise TypeError("qualification selected metrics are incomplete")
    overall_gain = float(comparison["dice_gain_vs_baseline"])
    scroll_gains: dict[str, float] = {}
    for scroll_name in REQUIRED_BASELINE_SCROLLS:
        scroll = scrolls.get(scroll_name)
        if not isinstance(scroll, dict):
            raise TypeError(f"qualification audit has no {scroll_name} metrics")
        matched = scroll.get("baseline_comparison")
        if not isinstance(matched, dict):
            raise TypeError(
                f"qualification audit has no matched {scroll_name} comparison"
            )
        scroll_gains[scroll_name] = float(matched["dice_gain_vs_baseline"])
        if not _same_float(scroll["dice_gain_vs_baseline"], scroll_gains[scroll_name]):
            raise ValueError(f"qualification audit {scroll_name} gain alias changed")
    minimum_scroll_gain = min(scroll_gains.values())
    if overall_gain <= 0 or minimum_scroll_gain <= 0:
        raise RuntimeError("qualification audit regresses on a held-out scroll")
    for recorded, measured in (
        (qualification["dice"], selected["dice"]),
        (qualification["precision"], selected["precision"]),
        (qualification["recall"], selected["recall"]),
        (qualification["overall_dice_gain_vs_baseline"], overall_gain),
        (
            qualification["macro_scroll_dice"],
            selected["macro_scroll_dice"],
        ),
        (
            qualification["macro_scroll_dice_gain_vs_baseline"],
            selected["macro_scroll_dice_gain_vs_baseline"],
        ),
        (
            qualification["minimum_scroll_dice_gain_vs_baseline"],
            minimum_scroll_gain,
        ),
    ):
        if not _same_float(recorded, measured):
            raise ValueError("qualification summary differs from its audit")
    recorded_scroll_gains = qualification.get("scroll_dice_gains_vs_baseline")
    if not isinstance(recorded_scroll_gains, dict) or set(recorded_scroll_gains) != set(
        REQUIRED_BASELINE_SCROLLS
    ):
        raise ValueError("qualification summary has incomplete scroll gains")
    for scroll_name, measured in scroll_gains.items():
        if not _same_float(recorded_scroll_gains[scroll_name], measured):
            raise ValueError("qualification scroll gain differs from its audit")
    return audit_source, audit_digest, policy


def seal_test_checkpoint(
    *,
    qualification_path: str | Path,
    final_checkpoint_path: str | Path,
    final_history_path: str | Path,
    patch_manifest_path: str | Path,
    visual_approval_path: str | Path,
    output_path: str | Path,
    test_scroll: str,
    expected_records: int = 5856,
    expected_epochs: int = 25,
) -> Path:
    """Freeze a completed qualified model before any sealed-test evidence opens."""

    qualification_source = Path(qualification_path).expanduser().resolve()
    final_checkpoint = Path(final_checkpoint_path).expanduser().resolve()
    final_history = Path(final_history_path).expanduser().resolve()
    patch_manifest = Path(patch_manifest_path).expanduser().resolve()
    visual_approval_source = Path(visual_approval_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if expected_records <= 0 or expected_epochs <= 0:
        raise ValueError("expected_records and expected_epochs must be positive")
    if not test_scroll:
        raise ValueError("test_scroll cannot be empty")

    qualification = json.loads(qualification_source.read_text(encoding="utf-8"))
    if qualification.get("schema") not in QUALIFICATION_SCHEMAS:
        raise ValueError(f"{qualification_source}: invalid qualification schema")
    if not bool(qualification.get("qualified")):
        raise RuntimeError(
            "tuning did not qualify; the sealed test must remain unopened while "
            "the model is repaired"
        )
    threshold = float(qualification["threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"{qualification_source}: invalid calibrated threshold")
    if not final_checkpoint.is_file():
        raise FileNotFoundError(
            f"qualified all-source final checkpoint is missing: {final_checkpoint}"
        )
    if not final_history.is_file():
        raise FileNotFoundError(f"final training history is missing: {final_history}")

    epochs, history_rows = _history_epochs(final_history)
    if epochs != list(range(expected_epochs)) or history_rows != expected_epochs:
        raise ValueError(
            f"{final_history}: final training epochs {epochs} are not exactly "
            f"0..{expected_epochs - 1}"
        )
    records, splits, scrolls = _manifest_inventory(patch_manifest)
    if records != expected_records:
        raise ValueError(
            f"{patch_manifest}: {records} records != expected {expected_records}"
        )
    if splits.get("test", 0):
        raise ValueError(f"{patch_manifest}: fitting corpus contains test rows")
    if any(test_scroll in values for values in scrolls.values()):
        raise ValueError(
            f"{patch_manifest}: sealed scroll {test_scroll} entered model fitting"
        )
    patch_digest = _sha256(patch_manifest)

    tuning_checkpoint = Path(str(qualification["checkpoint"])).expanduser().resolve()
    if not tuning_checkpoint.is_file():
        raise FileNotFoundError(
            f"qualified tuning checkpoint is missing: {tuning_checkpoint}"
        )
    tuning_digest = _sha256(tuning_checkpoint)
    recorded_tuning_digest = str(qualification["checkpoint_sha256"]).lower()
    if tuning_digest.lower() != recorded_tuning_digest:
        raise ValueError("qualified tuning checkpoint digest changed")
    audit_source, audit_digest, inference_policy = _validate_qualification_audit(
        qualification,
        tuning_digest=tuning_digest,
        patch_digest=patch_digest,
    )

    visual_approval = validate_visual_approval(
        visual_approval_source,
        checkpoint_path=final_checkpoint,
    )

    identity = {
        "test_scroll": test_scroll,
        "calibrated_threshold": threshold,
        "threshold_source": {
            "path": str(qualification_source),
            "sha256": _sha256(qualification_source),
            "tuning_checkpoint": str(tuning_checkpoint),
            "tuning_checkpoint_sha256": tuning_digest,
            "audit_report": str(audit_source),
            "audit_report_sha256": audit_digest,
            "inference_policy": inference_policy,
        },
        "frozen_checkpoint": {
            "path": str(final_checkpoint),
            "bytes": final_checkpoint.stat().st_size,
            "sha256": _sha256(final_checkpoint),
            "role": "qualified-all-source-final-fit",
        },
        "final_history": {
            "path": str(final_history),
            "sha256": _sha256(final_history),
            "rows": history_rows,
            "epochs": sorted(set(epochs)),
        },
        "fitting_corpus": {
            "path": str(patch_manifest),
            "sha256": patch_digest,
            "records": records,
            "splits": dict(sorted(splits.items())),
            "scrolls": {
                split: sorted(values) for split, values in sorted(scrolls.items())
            },
            "sealed_scroll_absent": True,
        },
        "visual_approval": {
            "path": str(visual_approval_source),
            "sha256": _sha256(visual_approval_source),
            "schema": visual_approval["schema"],
            "decision": visual_approval["decision"],
            "reviewer": visual_approval["reviewer"],
            "grid_provenance_sha256": visual_approval["identity"]["grid_inference"][
                "sha256"
            ],
            "visual_audit_sha256": visual_approval["identity"]["visual_evidence"][
                "audit_sha256"
            ],
        },
    }
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if (
            existing.get("schema") != SEAL_SCHEMA
            or existing.get("identity") != identity
        ):
            raise ValueError(f"{destination}: sealed-test identity changed")
        return destination
    _atomic_json(
        destination,
        {
            "schema": SEAL_SCHEMA,
            "state": "sealed-before-test-access",
            "created_at": datetime.now(UTC).isoformat(),
            "identity": identity,
        },
    )
    return destination
