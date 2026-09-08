from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from .io import open_volume, read_crop, split_volume_spec
from .resources import configure_cpu_budget


@dataclass(frozen=True)
class TeacherAuditOptions:
    max_records: int = 8
    slices_per_axis: int = 3
    tolerance_voxels: int = 2
    max_cpu_threads: int = 16

    def validate(self) -> None:
        if self.max_records <= 0 or self.slices_per_axis <= 0:
            raise ValueError("audit record and slice counts must be positive")
        if self.tolerance_voxels < 0:
            raise ValueError("tolerance_voxels must be non-negative")
        if not 1 <= self.max_cpu_threads <= 16:
            raise ValueError("max_cpu_threads must be in [1, 16]")


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / max(int(denominator), 1)


def _mask_metrics(
    prediction: np.ndarray, reference: np.ndarray, *, tolerance: int
) -> dict[str, Any]:
    true_positive = int(np.count_nonzero(prediction & reference))
    predicted = int(np.count_nonzero(prediction))
    expected = int(np.count_nonzero(reference))
    union = int(np.count_nonzero(prediction | reference))
    result: dict[str, Any] = {
        "voxels": int(prediction.size),
        "predicted_positive": predicted,
        "reference_positive": expected,
        "true_positive": true_positive,
        "false_positive": predicted - true_positive,
        "false_negative": expected - true_positive,
        "dice": _safe_ratio(2 * true_positive, predicted + expected),
        "iou": _safe_ratio(true_positive, union),
        "precision": _safe_ratio(true_positive, predicted),
        "recall": _safe_ratio(true_positive, expected),
        "agreement": float(np.mean(prediction == reference)),
    }
    if tolerance > 0:
        dilated_prediction = binary_dilation(prediction, iterations=tolerance)
        dilated_reference = binary_dilation(reference, iterations=tolerance)
        result[f"precision_at_{tolerance}vox"] = _safe_ratio(
            int(np.count_nonzero(prediction & dilated_reference)), predicted
        )
        result[f"recall_at_{tolerance}vox"] = _safe_ratio(
            int(np.count_nonzero(reference & dilated_prediction)), expected
        )
    return result


def _grayscale(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    nonzero = values[values > 0]
    sample = nonzero if nonzero.size >= 32 else values.reshape(-1)
    lower, upper = np.percentile(sample, (1.0, 99.0))
    if upper <= lower:
        upper = lower + 1.0
    normalized = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    return (normalized * 255).astype(np.uint8)


def _overlay(
    gray: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]
) -> np.ndarray:
    rgb = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)
    color_value = np.asarray(color, dtype=np.float32)
    rgb[mask] = 0.35 * rgb[mask] + 0.65 * color_value
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _difference_panel(
    gray: np.ndarray, prediction: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    rgb = np.repeat((gray // 3)[..., None], 3, axis=-1)
    true_positive = prediction & reference
    false_positive = prediction & ~reference
    false_negative = ~prediction & reference
    rgb[true_positive] = (80, 255, 130)
    rgb[false_positive] = (255, 55, 190)
    rgb[false_negative] = (55, 220, 255)
    return rgb


def _take_slice(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(array, index, axis=axis)


def _slice_indices(disagreement: np.ndarray, axis: int, count: int) -> list[int]:
    reduction_axes = tuple(value for value in range(3) if value != axis)
    score = disagreement.sum(axis=reduction_axes)
    order = np.argsort(score)[::-1]
    chosen: list[int] = [int(disagreement.shape[axis] // 2)]
    for value in order:
        index = int(value)
        if index not in chosen:
            chosen.append(index)
        if len(chosen) >= count:
            break
    return sorted(chosen[:count])


def _render_slice(
    raw: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    title: str,
) -> Image.Image:
    gray = _grayscale(raw)
    panels = [
        np.repeat(gray[..., None], 3, axis=-1),
        _overlay(gray, prediction, (255, 55, 190)),
        _overlay(gray, reference, (55, 220, 255)),
        _difference_panel(gray, prediction, reference),
    ]
    labels = ("fine CT", "local teacher", "published", "TP / FP / FN")
    scale = 2
    label_height = 22
    panel_images: list[Image.Image] = []
    for panel, label in zip(panels, labels, strict=True):
        image = Image.fromarray(panel, mode="RGB").resize(
            (panel.shape[1] * scale, panel.shape[0] * scale),
            Image.Resampling.NEAREST,
        )
        framed = Image.new("RGB", (image.width, image.height + label_height), "#111827")
        framed.paste(image, (0, label_height))
        ImageDraw.Draw(framed).text((6, 4), label, fill="white")
        panel_images.append(framed)
    result = Image.new(
        "RGB",
        (
            sum(image.width for image in panel_images),
            panel_images[0].height + label_height,
        ),
        "#030712",
    )
    ImageDraw.Draw(result).text((8, 4), title, fill="white")
    x = 0
    for panel in panel_images:
        result.paste(panel, (x, label_height))
        x += panel.width
    return result


def audit_materialized_teacher(
    *,
    fine_volume: str,
    candidate_volume: str,
    reference_volume: str,
    output_path: str | Path,
    options: TeacherAuditOptions,
) -> Path:
    """Compare local native-teacher chunks with a published voxel mask."""

    options.validate()
    configure_cpu_budget(options.max_cpu_threads)
    candidate_path, _candidate_key = split_volume_spec(candidate_volume)
    state_path = candidate_path / "teacher_state.json"
    if not state_path.is_file():
        raise ValueError(f"candidate has no teacher state: {candidate_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("state") != "complete":
        raise ValueError(f"candidate teacher is not complete: {state.get('state')!r}")
    record_paths = sorted((candidate_path / "records").glob("*.json"))[
        : options.max_records
    ]
    if not record_paths:
        raise ValueError("candidate teacher has no chunk records")

    fine = open_volume(fine_volume)
    candidate = open_volume(candidate_volume)
    reference = open_volume(reference_volume)
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError(f"audit output already exists: {output}")
    output.mkdir(parents=True)
    images = output / "images"
    images.mkdir()

    records: list[dict[str, Any]] = []
    aggregate_prediction: list[np.ndarray] = []
    aggregate_reference: list[np.ndarray] = []
    axis_names = ("z", "y", "x")
    for record_path in record_paths:
        source_record = json.loads(record_path.read_text(encoding="utf-8"))
        origin = tuple(int(value) for value in source_record["origin_zyx"])
        shape = tuple(int(value) for value in source_record["shape_zyx"])
        raw = read_crop(fine, origin, shape)
        prediction = read_crop(candidate, origin, shape) > 0
        expected = read_crop(reference, origin, shape) > 0
        metrics = _mask_metrics(
            prediction, expected, tolerance=options.tolerance_voxels
        )
        disagreement = prediction != expected
        views: list[dict[str, Any]] = []
        coordinate_label = "_".join(str(value) for value in source_record["chunk_zyx"])
        for axis, axis_name in enumerate(axis_names):
            for index in _slice_indices(disagreement, axis, options.slices_per_axis):
                filename = f"{coordinate_label}_{axis_name}{index:03d}.png"
                title = (
                    f"chunk {coordinate_label} | {axis_name}={index} | "
                    "green=TP magenta=local-only cyan=published-only"
                )
                rendered = _render_slice(
                    _take_slice(raw, axis, index),
                    _take_slice(prediction, axis, index),
                    _take_slice(expected, axis, index),
                    title=title,
                )
                rendered.save(images / filename, optimize=True)
                views.append(
                    {"axis": axis_name, "index": index, "image": f"images/{filename}"}
                )
        records.append(
            {
                "chunk_zyx": source_record["chunk_zyx"],
                "origin_zyx": source_record["origin_zyx"],
                "metrics": metrics,
                "views": views,
            }
        )
        aggregate_prediction.append(prediction)
        aggregate_reference.append(expected)

    combined_prediction = np.stack(aggregate_prediction)
    combined_reference = np.stack(aggregate_reference)
    report = {
        "schema": "crossres-native-teacher-audit-v1",
        "fine_volume": fine_volume,
        "candidate_volume": candidate_volume,
        "reference_volume": reference_volume,
        "teacher_identity": state["identity"],
        "options": {
            "max_records": options.max_records,
            "slices_per_axis": options.slices_per_axis,
            "tolerance_voxels": options.tolerance_voxels,
            "max_cpu_threads": options.max_cpu_threads,
        },
        "aggregate": _mask_metrics(
            combined_prediction,
            combined_reference,
            tolerance=options.tolerance_voxels,
        ),
        "records": records,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    cards: list[str] = []
    for record in records:
        metrics = record["metrics"]
        rows = "".join(
            f"<tr><th>{html.escape(str(key))}</th><td>{value:.6f}</td></tr>"
            if isinstance(value, float)
            else f"<tr><th>{html.escape(str(key))}</th><td>{value:,}</td></tr>"
            for key, value in metrics.items()
        )
        views = "".join(
            f'<figure><img src="{html.escape(view["image"])}" '
            f'alt="{html.escape(view["axis"])} {view["index"]}"></figure>'
            for view in record["views"]
        )
        cards.append(
            f"<section><h2>Chunk {record['chunk_zyx']}</h2>"
            f"<table>{rows}</table><div class=views>{views}</div></section>"
        )
    aggregate_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{value:.6f}</td></tr>"
        if isinstance(value, float)
        else f"<tr><th>{html.escape(str(key))}</th><td>{value:,}</td></tr>"
        for key, value in report["aggregate"].items()
    )
    document = f"""<!doctype html>
<html lang=en><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>Native teacher voxel audit</title>
<style>
body{{font:15px system-ui;background:#030712;color:#e5e7eb;margin:24px}}
h1,h2{{color:#f9fafb}} section{{border-top:1px solid #374151;padding-top:20px}}
table{{border-collapse:collapse;margin:12px 0 20px}}th,td{{padding:5px 10px;border:1px solid #374151;text-align:right}}
th{{text-align:left;color:#93c5fd}}.views{{display:grid;gap:14px}}figure{{margin:0;overflow:auto}}
img{{max-width:none;border:1px solid #374151}}code{{color:#f0abfc}}
</style>
<h1>Native 2 µm teacher: local vs published Paris4</h1>
<p>This is a direct voxel audit. Green is agreement on surface, magenta is local-only, cyan is published-only.</p>
<h2>Aggregate</h2><table>{aggregate_rows}</table>
{"".join(cards)}
</html>"""
    (output / "index.html").write_text(document, encoding="utf-8")
    return output / "index.html"
