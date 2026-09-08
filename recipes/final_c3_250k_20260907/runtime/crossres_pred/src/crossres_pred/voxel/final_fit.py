from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from .loss import LOSS_CONTRACT
from .sealed_test import QUALIFICATION_SCHEMAS
from .train import (
    CHECKPOINT_DURABILITY_CONTRACT,
    CHECKPOINT_SELECTION_CONTRACT,
    EPOCH_PARTITION_CONTRACT,
    FINAL_FIT_CHECKPOINT_CONTRACT,
    LEARNING_RATE_CONTRACT,
    SAMPLING_CONTRACT,
    _history_row_score,
)

FINAL_FIT_VERIFICATION_SCHEMA = "crossres-voxel-final-fit-verification-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"final-fit input is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


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


def _manifest_inventory(path: Path) -> tuple[int, Counter[str], set[str]]:
    records = 0
    splits: Counter[str] = Counter()
    scrolls: set[str] = set()
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
            records += 1
            splits[split] += 1
            scrolls.add(scroll)
    if records == 0:
        raise ValueError(f"{path}: patch manifest is empty")
    return records, splits, scrolls


def _history(path: Path, expected_epochs: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid history row"
                ) from error
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: history row is not an object")
            rows.append(row)
    epochs = [row.get("epoch") for row in rows]
    if epochs != list(range(expected_epochs)):
        raise ValueError(
            f"{path}: final-fit epochs {epochs} are not exactly 0..{expected_epochs - 1}"
        )
    return rows


def _assert_models_equal(
    best: dict[str, Any], last: dict[str, Any], *, best_path: Path, last_path: Path
) -> tuple[int, int]:
    best_model = best.get("model")
    last_model = last.get("model")
    if not isinstance(best_model, dict) or not isinstance(last_model, dict):
        raise TypeError("final-fit checkpoint model state is missing")
    if set(best_model) != set(last_model):
        raise ValueError("final-fit best/last model state keys differ")
    parameters = 0
    for name in sorted(best_model):
        best_tensor = best_model[name]
        last_tensor = last_model[name]
        if not isinstance(best_tensor, torch.Tensor) or not isinstance(
            last_tensor, torch.Tensor
        ):
            raise TypeError(f"final-fit model state {name!r} is not a tensor")
        if not torch.equal(best_tensor, last_tensor):
            raise ValueError(
                f"final-fit model state {name!r} differs between "
                f"{best_path} and {last_path}"
            )
        parameters += best_tensor.numel()
    return len(best_model), parameters


def verify_final_fit(
    *,
    qualification_path: str | Path,
    best_checkpoint_path: str | Path,
    last_checkpoint_path: str | Path,
    history_path: str | Path,
    patch_manifest_path: str | Path,
    output_path: str | Path,
    expected_records: int,
    expected_epochs: int,
    expected_samples_per_epoch: int,
    expected_total_samples: int | None = None,
    expected_batch_size: int = 3,
    expected_max_cpu_threads: int = 16,
) -> Path:
    """Verify that a qualified final fit completed and froze its last epoch."""

    if (
        min(
            expected_records,
            expected_epochs,
            expected_samples_per_epoch,
            expected_batch_size,
            expected_max_cpu_threads,
        )
        <= 0
    ):
        raise ValueError("final-fit expectations must be positive")
    if expected_max_cpu_threads > 16:
        raise ValueError("final-fit CPU expectation exceeds the safety limit")
    if expected_total_samples is not None and expected_total_samples <= 0:
        raise ValueError("expected_total_samples must be positive")

    qualification_source = Path(qualification_path).expanduser().resolve()
    best_path = Path(best_checkpoint_path).expanduser().resolve()
    last_path = Path(last_checkpoint_path).expanduser().resolve()
    history_source = Path(history_path).expanduser().resolve()
    manifest = Path(patch_manifest_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if best_path.parent != last_path.parent:
        raise ValueError("final-fit best/last checkpoints are in different runs")
    run_path = best_path.parent / "run.json"
    required_files = (
        qualification_source,
        best_path,
        last_path,
        history_source,
        manifest,
        run_path,
    )
    for path in required_files:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"final-fit input is missing or empty: {path}")

    manifest_digest = _sha256(manifest)
    qualification = _read_object(qualification_source)
    if qualification.get("schema") not in QUALIFICATION_SCHEMAS:
        raise ValueError(f"{qualification_source}: invalid qualification schema")
    if not bool(qualification.get("qualified")):
        raise RuntimeError("final fit has no successful tuning qualification")
    if str(qualification.get("patch_manifest_sha256", "")).lower() != manifest_digest:
        raise ValueError("qualification and final-fit patch manifests differ")
    policy = qualification.get("inference_policy")
    if not isinstance(policy, dict) or (
        policy.get("split") != "val"
        or policy.get("amp_dtype") != "bfloat16"
        or policy.get("mirror_tta") is not True
        or policy.get("qualification_scroll") != "PHerc0814"
        or policy.get("baseline_comparison_policy") != "matched-rows-only"
    ):
        raise ValueError("qualification inference policy is not deployable")
    threshold = float(qualification.get("threshold"))
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("qualification threshold is invalid")
    tuning_checkpoint = Path(str(qualification.get("checkpoint", ""))).resolve()
    audit_report = Path(str(qualification.get("audit_report", ""))).resolve()
    for path, hash_field in (
        (tuning_checkpoint, "checkpoint_sha256"),
        (audit_report, "audit_report_sha256"),
    ):
        if (
            not path.is_file()
            or _sha256(path) != str(qualification.get(hash_field, "")).lower()
        ):
            raise ValueError(f"qualification dependency changed: {path}")

    records, splits, scrolls = _manifest_inventory(manifest)
    if records != expected_records:
        raise ValueError(f"{manifest}: {records} rows != expected {expected_records}")
    if splits.get("test", 0):
        raise ValueError("final-fit manifest contains test rows")
    sealed_scrolls = {"PHerc0846A", "PHerc1203"}
    leaked = sorted(scrolls & sealed_scrolls)
    if leaked:
        raise ValueError(f"sealed scrolls entered final fitting: {leaked}")

    run_identity = _read_object(run_path)
    if (
        Path(str(run_identity.get("patch_manifest", ""))).resolve() != manifest
        or str(run_identity.get("patch_manifest_sha256", "")).lower() != manifest_digest
        or run_identity.get("loss_contract") != LOSS_CONTRACT
        or run_identity.get("checkpoint_selection_contract")
        != CHECKPOINT_SELECTION_CONTRACT
        or run_identity.get("checkpoint_durability_contract")
        != CHECKPOINT_DURABILITY_CONTRACT
        or run_identity.get("epoch_partition_contract") != EPOCH_PARTITION_CONTRACT
        or run_identity.get("sampling_contract") != SAMPLING_CONTRACT
        or run_identity.get("learning_rate_contract") != LEARNING_RATE_CONTRACT
        or run_identity.get("final_fit_checkpoint_contract")
        != FINAL_FIT_CHECKPOINT_CONTRACT
    ):
        raise ValueError("final-fit run identity or training contract changed")
    options = run_identity.get("options")
    if not isinstance(options, dict):
        raise TypeError("final-fit run options are missing")
    expected_options = {
        "final_fit": True,
        "epochs": expected_epochs,
        "samples_per_epoch": expected_samples_per_epoch,
        "batch_size": expected_batch_size,
        "max_cpu_threads": expected_max_cpu_threads,
        "preset": "m7-resenc-l",
        "amp_dtype": "bfloat16",
        "device": "cuda",
    }
    for name, expected in expected_options.items():
        if options.get(name) != expected:
            raise ValueError(f"final-fit option {name!r} is not {expected!r}")
    total_samples = expected_total_samples or records
    if run_identity.get("effective_partition_samples") != total_samples:
        raise ValueError("final-fit schedule does not match the selected sample budget")
    if options.get("max_train_samples") != total_samples:
        raise ValueError("final-fit max_train_samples does not match qualification")
    expected_schedule = {
        "evaluation_interval_samples": expected_samples_per_epoch,
        "total_samples": total_samples,
        "evaluation_intervals": expected_epochs,
    }
    if run_identity.get("resolved_schedule") != expected_schedule:
        raise ValueError("final-fit resolved schedule changed")
    if int(qualification.get("selected_train_samples", -1)) != total_samples:
        raise ValueError("qualification and final-fit sample budgets differ")
    m7_path = Path(str(options.get("pretrained_m7_checkpoint", ""))).resolve()
    if not m7_path.is_file():
        raise FileNotFoundError(f"final-fit m7 initializer is missing: {m7_path}")

    history_rows = _history(history_source, expected_epochs)
    remaining_samples = total_samples
    epoch_samples: list[int] = []
    for epoch, row in enumerate(history_rows):
        expected_samples = min(expected_samples_per_epoch, remaining_samples)
        if expected_samples <= 0:
            raise ValueError("final-fit sample budget ended before the final epoch")
        train_metrics = row.get("train")
        if not isinstance(train_metrics, dict) or float(
            train_metrics.get("samples", float("nan"))
        ) != float(expected_samples):
            raise ValueError(
                f"final-fit epoch {epoch} did not consume {expected_samples} samples"
            )
        epoch_samples.append(expected_samples)
        remaining_samples -= expected_samples
    if remaining_samples != 0:
        raise ValueError(
            f"final-fit history left {remaining_samples} manifest samples unseen"
        )
    final_row = history_rows[-1]
    input_identity = {
        "verification_contract": FINAL_FIT_VERIFICATION_SCHEMA,
        "qualification": {
            "path": str(qualification_source),
            "sha256": _sha256(qualification_source),
        },
        "run": {"path": str(run_path), "sha256": _sha256(run_path)},
        "best_checkpoint": {"path": str(best_path), "sha256": _sha256(best_path)},
        "last_checkpoint": {"path": str(last_path), "sha256": _sha256(last_path)},
        "history": {"path": str(history_source), "sha256": _sha256(history_source)},
        "patch_manifest": {"path": str(manifest), "sha256": manifest_digest},
        "expectations": {
            "records": expected_records,
            "epochs": expected_epochs,
            "samples_per_epoch": expected_samples_per_epoch,
            "total_samples": total_samples,
            "batch_size": expected_batch_size,
            "max_cpu_threads": expected_max_cpu_threads,
        },
    }
    if output.is_file():
        existing = _read_object(output)
        if (
            existing.get("schema") != FINAL_FIT_VERIFICATION_SCHEMA
            or existing.get("state") != "verified"
            or existing.get("identity") != input_identity
        ):
            raise ValueError(f"{output}: final-fit verification identity changed")
        return output

    last = torch.load(last_path, map_location="cpu", weights_only=False)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if not isinstance(last, dict) or not isinstance(best, dict):
        raise TypeError("final-fit checkpoint payload is not an object")
    expected_epoch = expected_epochs - 1
    for label, payload in (("best", best), ("last", last)):
        if payload.get("epoch") != expected_epoch:
            raise ValueError(
                f"final-fit {label} checkpoint is not epoch {expected_epoch}"
            )
        if payload.get("identity") != run_identity:
            raise ValueError(f"final-fit {label} checkpoint identity changed")
        if payload.get("metrics") != final_row:
            raise ValueError(f"final-fit {label} metrics differ from history")
        initialization = payload.get("initialization")
        if not isinstance(initialization, dict) or (
            initialization.get("strict") is not True
            or Path(str(initialization.get("checkpoint", ""))).resolve() != m7_path
        ):
            raise ValueError(f"final-fit {label} checkpoint is not strict m7-init")
    if "optimizer" not in last or "optimizer" in best:
        raise ValueError("final-fit checkpoint optimizer contract changed")
    if set(best) != set(last) - {"optimizer"}:
        raise ValueError("final-fit best/last checkpoint fields differ unexpectedly")
    final_score = _history_row_score(final_row)
    if not math.isclose(
        float(last.get("best_score")), final_score, rel_tol=0.0, abs_tol=0.0
    ) or not math.isclose(
        float(best.get("best_score")), final_score, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("final-fit checkpoint does not bind the last-epoch score")
    tensor_count, parameter_count = _assert_models_equal(
        best, last, best_path=best_path, last_path=last_path
    )

    _atomic_json(
        output,
        {
            "schema": FINAL_FIT_VERIFICATION_SCHEMA,
            "state": "verified",
            "verified_at": datetime.now(UTC).isoformat(),
            "identity": input_identity,
            "summary": {
                "records": records,
                "splits": dict(sorted(splits.items())),
                "scrolls": sorted(scrolls),
                "epochs": expected_epochs,
                "scheduled_samples": total_samples,
                "epoch_samples": epoch_samples,
                "final_epoch": expected_epoch,
                "calibrated_threshold": threshold,
                "model_state_tensors": tensor_count,
                "model_state_values": parameter_count,
                "best_equals_last_model": True,
                "sealed_scrolls_absent": True,
            },
        },
    )
    return output
