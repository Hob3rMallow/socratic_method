from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .provenance import sha256_file, utc_now, write_json_atomic

TENSORBOARD_DIRECTORY = "tensorboard"
_EXPORT_MARKER = "history_export.json"
_LOSS_COMPONENTS = (
    "total",
    "supervised_bce",
    "supervised_dice",
    "distill_bce",
    "rehearsal_bce",
    "p1_voxels",
    "p2_voxels",
    "p3_voxels",
)


def _log_numeric_tree(writer: Any, prefix: str, value: Any, epoch: int) -> None:
    if isinstance(value, dict):
        for name, child in value.items():
            _log_numeric_tree(writer, f"{prefix}/{name}", child, epoch)
    elif isinstance(value, bool):
        writer.add_scalar(prefix, int(value), epoch)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        writer.add_scalar(prefix, float(value), epoch)


def load_history_rows(path: str | Path) -> list[dict[str, Any]]:
    history_path = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    try:
        lines = history_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ValueError(f"training history is missing: {history_path}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{history_path}:{line_number}: invalid training history JSON"
            ) from error
        if not isinstance(row, dict):
            raise TypeError(
                f"{history_path}:{line_number}: history row is not an object"
            )
        expected_epoch = len(rows) + 1
        if int(row.get("epoch", 0)) != expected_epoch:
            raise ValueError(
                f"{history_path}:{line_number}: expected epoch {expected_epoch}, "
                f"found {row.get('epoch')!r}"
            )
        rows.append(row)
    if not rows:
        raise ValueError(f"training history is empty: {history_path}")
    return rows


def _summary_writer(log_dir: Path, *, purge_step: int | None = None) -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError(
            "TensorBoard logging requires the 'tensorboard' package; install the "
            "crossres_pred project dependencies"
        ) from error
    return SummaryWriter(log_dir=str(log_dir), purge_step=purge_step)


def _add_custom_scalar_layout(writer: Any) -> None:
    paired = {
        name: [
            "Multiline",
            [f"loss/train/{name}", f"loss/validation/{name}"],
        ]
        for name in _LOSS_COMPONENTS
    }
    writer.add_custom_scalars({"Loss": paired})


def create_training_writer(
    run_directory: str | Path, *, purge_step: int | None = None
) -> Any:
    run = Path(run_directory).expanduser().resolve()
    writer = _summary_writer(run / TENSORBOARD_DIRECTORY, purge_step=purge_step)
    _add_custom_scalar_layout(writer)
    return writer


def log_history_row(writer: Any, row: dict[str, Any]) -> None:
    epoch = int(row["epoch"])
    train = row.get("train")
    if not isinstance(train, dict):
        raise TypeError(f"epoch {epoch}: training metrics are missing")
    for name, value in train.items():
        writer.add_scalar(f"loss/train/{name}", float(value), epoch)

    validation = row.get("val")
    if validation is not None:
        if not isinstance(validation, dict):
            raise TypeError(f"epoch {epoch}: validation metrics are not an object")
        for name, value in validation.items():
            writer.add_scalar(f"loss/validation/{name}", float(value), epoch)
    best_validation = row.get("best_val_loss")
    if best_validation is not None:
        writer.add_scalar("loss/validation/best_total", float(best_validation), epoch)
    selection = row.get("val_selection")
    if isinstance(selection, dict):
        _log_numeric_tree(writer, "selection/validation", selection, epoch)
    best_selection = row.get("best_selection")
    if best_selection is not None:
        writer.add_scalar("selection/best", float(best_selection), epoch)
    writer.add_scalar("optimization/learning_rate", float(row["learning_rate"]), epoch)


def export_history_to_tensorboard(run_directory: str | Path) -> Path:
    run = Path(run_directory).expanduser().resolve()
    if not run.is_dir():
        raise ValueError(f"training run directory does not exist: {run}")
    history_path = run / "history.jsonl"
    rows = load_history_rows(history_path)
    history_hash = sha256_file(history_path)
    log_dir = run / TENSORBOARD_DIRECTORY
    marker_path = log_dir / _EXPORT_MARKER
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    if event_files:
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise ValueError(
                f"{log_dir}: TensorBoard events already exist and were not created "
                "by the history exporter"
            ) from error
        if marker.get("history_sha256") != history_hash:
            raise ValueError(
                f"{log_dir}: exported TensorBoard history is stale; use a fresh "
                "directory rather than mixing event streams"
            )
        return log_dir

    writer = _summary_writer(log_dir)
    try:
        _add_custom_scalar_layout(writer)
        for row in rows:
            log_history_row(writer, row)
        writer.flush()
    finally:
        writer.close()
    write_json_atomic(
        marker_path,
        {
            "schema_version": 1,
            "kind": "crossres-tensorboard-history-export",
            "created_at": utc_now(),
            "history": str(history_path),
            "history_sha256": history_hash,
            "epochs": len(rows),
        },
    )
    return log_dir
