from __future__ import annotations

import hashlib
import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

from .inference import _predict_probability, load_voxel_checkpoint
from .patches import VoxelPatchDataset
from .resources import assert_cuda_power_limit, configure_cpu_budget

DEFAULT_THRESHOLDS = tuple(float(value) for value in np.linspace(0.10, 0.90, 17))
CHECKPOINT_AUDIT_SCHEMA = "crossres-voxel-checkpoint-audit-v3"


@dataclass(frozen=True)
class CheckpointAuditOptions:
    split: str = "val"
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    qualification_scroll: str = "PHerc0814"
    device: str = "cuda"
    amp_dtype: str = "bfloat16"
    mirror_tta: bool = False
    num_workers: int = 2
    max_cpu_threads: int = 16

    def validate(self) -> None:
        if not self.split:
            raise ValueError("audit split cannot be empty")
        if not self.thresholds:
            raise ValueError("at least one audit threshold is required")
        if any(not 0 <= value <= 1 for value in self.thresholds):
            raise ValueError("audit thresholds must be in [0, 1]")
        if len(set(self.thresholds)) != len(self.thresholds):
            raise ValueError("audit thresholds must be unique")
        if self.amp_dtype not in {"bfloat16", "float16"}:
            raise ValueError("amp_dtype must be bfloat16 or float16")
        if self.num_workers < 0 or self.num_workers >= self.max_cpu_threads:
            raise ValueError("num_workers must be in [0, max_cpu_threads)")
        if not 1 <= self.max_cpu_threads <= 16:
            raise ValueError("max_cpu_threads must be in [1, 16]")


@dataclass
class ScalarCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    known: int = 0
    positive: int = 0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        valid = target != 2
        truth = target == 1
        predicted = prediction.astype(bool, copy=False)
        self.true_positive += int(np.count_nonzero(predicted & truth & valid))
        self.false_positive += int(np.count_nonzero(predicted & ~truth & valid))
        self.false_negative += int(np.count_nonzero(~predicted & truth & valid))
        self.known += int(np.count_nonzero(valid))
        self.positive += int(np.count_nonzero(truth & valid))

    def metrics(self) -> dict[str, float | int]:
        return _metrics(
            self.true_positive,
            self.false_positive,
            self.false_negative,
            self.known,
            self.positive,
        )


@dataclass
class ThresholdCounts:
    thresholds: np.ndarray
    true_positive: np.ndarray
    false_positive: np.ndarray
    false_negative: np.ndarray
    known: int = 0
    positive: int = 0

    @classmethod
    def create(cls, thresholds: tuple[float, ...]) -> ThresholdCounts:
        values = np.asarray(thresholds, dtype=np.float32)
        zeros = np.zeros(values.shape, dtype=np.int64)
        return cls(values, zeros.copy(), zeros.copy(), zeros.copy())

    def update(
        self,
        probability: np.ndarray,
        target: np.ndarray,
        *,
        chunk_voxels: int = 1_000_000,
    ) -> None:
        if probability.shape != target.shape:
            raise ValueError("probability and target shapes differ")
        valid = target != 2
        truth = target[valid] == 1
        scores = probability[valid].astype(np.float32, copy=False)
        self.known += int(scores.size)
        self.positive += int(np.count_nonzero(truth))
        threshold_column = self.thresholds[:, None]
        for start in range(0, scores.size, chunk_voxels):
            stop = min(scores.size, start + chunk_voxels)
            local_truth = truth[start:stop][None]
            predicted = scores[start:stop][None] >= threshold_column
            self.true_positive += np.count_nonzero(predicted & local_truth, axis=1)
            self.false_positive += np.count_nonzero(predicted & ~local_truth, axis=1)
            self.false_negative += np.count_nonzero(~predicted & local_truth, axis=1)

    def metrics_at(self, index: int) -> dict[str, float | int]:
        result = _metrics(
            int(self.true_positive[index]),
            int(self.false_positive[index]),
            int(self.false_negative[index]),
            self.known,
            self.positive,
        )
        result["threshold"] = float(self.thresholds[index])
        return result


@dataclass
class TolerantThresholdCounts:
    """Asymmetric surface precision/recall with a fixed voxel tolerance."""

    thresholds: np.ndarray
    matched_prediction: np.ndarray
    predicted: np.ndarray
    matched_truth: np.ndarray
    truth: int = 0
    tolerance_voxels: int = 2

    @classmethod
    def create(
        cls, thresholds: tuple[float, ...], *, tolerance_voxels: int = 2
    ) -> TolerantThresholdCounts:
        values = np.asarray(thresholds, dtype=np.float32)
        zeros = np.zeros(values.shape, dtype=np.int64)
        return cls(
            values,
            zeros.copy(),
            zeros.copy(),
            zeros.copy(),
            tolerance_voxels=tolerance_voxels,
        )

    def update(self, probability: np.ndarray, target: np.ndarray) -> None:
        if probability.shape != target.shape:
            raise ValueError("probability and target shapes differ")
        valid = target != 2
        truth = (target == 1) & valid
        structure = ndimage.generate_binary_structure(3, 1)
        near_truth = ndimage.binary_dilation(
            truth,
            structure=structure,
            iterations=self.tolerance_voxels,
            mask=valid,
        )
        local_probability = np.where(valid, probability, -1.0).astype(
            np.float32, copy=False
        )
        recovered_probability = local_probability
        for _ in range(self.tolerance_voxels):
            recovered_probability = ndimage.maximum_filter(
                recovered_probability,
                footprint=structure,
                mode="constant",
                cval=-1.0,
            )
        scores = probability[valid].astype(np.float32, copy=False)
        precision_truth = near_truth[valid]
        recall_scores = recovered_probability[truth]
        threshold_column = self.thresholds[:, None]
        predicted = scores[None] >= threshold_column
        self.predicted += np.count_nonzero(predicted, axis=1)
        self.matched_prediction += np.count_nonzero(
            predicted & precision_truth[None], axis=1
        )
        self.matched_truth += np.count_nonzero(
            recall_scores[None] >= threshold_column, axis=1
        )
        self.truth += int(recall_scores.size)

    def metrics_at(self, index: int) -> dict[str, float | int]:
        precision = int(self.matched_prediction[index]) / max(
            1, int(self.predicted[index])
        )
        recall = int(self.matched_truth[index]) / max(1, self.truth)
        beta_squared = 0.25
        f0_5 = (
            (1.0 + beta_squared)
            * precision
            * recall
            / max(
                beta_squared * precision + recall,
                1.0e-12,
            )
        )
        return {
            "tolerance_voxels": self.tolerance_voxels,
            "matched_prediction_voxels": int(self.matched_prediction[index]),
            "predicted_voxels": int(self.predicted[index]),
            "matched_truth_voxels": int(self.matched_truth[index]),
            "truth_voxels": self.truth,
            "precision": precision,
            "recall": recall,
            "f0_5": f0_5,
        }


def _metrics(
    true_positive: int,
    false_positive: int,
    false_negative: int,
    known: int,
    positive: int,
) -> dict[str, float | int]:
    predicted = true_positive + false_positive
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "known_voxels": known,
        "positive_voxels": positive,
        "positive_prevalence": positive / max(1, known),
        "dice": (2.0 * true_positive)
        / max(1, 2 * true_positive + false_positive + false_negative),
        "precision": true_positive / max(1, true_positive + false_positive),
        "recall": true_positive / max(1, true_positive + false_negative),
        "predicted_voxels": predicted,
        "foreground_ratio": predicted / max(1, positive),
    }


def add_tolerant_surface_metrics(
    sweep: dict[str, Any],
    *,
    overall: TolerantThresholdCounts,
    by_scroll: dict[str, TolerantThresholdCounts],
) -> None:
    points = sweep["points"]
    if len(points) != overall.thresholds.size:
        raise ValueError("strict and tolerant threshold counts differ")
    for index, point in enumerate(points):
        tolerant = overall.metrics_at(index)
        point["surface_at_2vox"] = tolerant
        scroll_f0_5: list[float] = []
        for scroll, counts in sorted(by_scroll.items()):
            scroll_metrics = counts.metrics_at(index)
            point["scrolls"][scroll]["surface_at_2vox"] = scroll_metrics
            scroll_f0_5.append(float(scroll_metrics["f0_5"]))
        point["macro_surface_f0_5_at_2vox"] = sum(scroll_f0_5) / max(
            1, len(scroll_f0_5)
        )
    sweep["surface_tolerance_voxels"] = overall.tolerance_voxels
    sweep["surface_selection_metric"] = "macro_surface_f0_5_at_2vox"


def summarize_thresholds(
    *,
    overall: ThresholdCounts,
    comparison: ThresholdCounts,
    baseline: ScalarCounts,
    by_scroll: dict[str, ThresholdCounts],
    comparison_by_scroll: dict[str, ThresholdCounts],
    baseline_by_scroll: dict[str, ScalarCounts],
    qualification_scroll: str,
) -> dict[str, Any]:
    if not np.array_equal(overall.thresholds, comparison.thresholds):
        raise ValueError("overall/comparison thresholds differ")
    if set(comparison_by_scroll) != set(baseline_by_scroll):
        raise ValueError("model/baseline comparison scroll domains differ")
    for scroll, counts in by_scroll.items():
        if not np.array_equal(overall.thresholds, counts.thresholds):
            raise ValueError(f"overall/{scroll} thresholds differ")
    for scroll, counts in comparison_by_scroll.items():
        if scroll not in by_scroll:
            raise ValueError(
                f"comparison scroll is absent from all-domain metrics: {scroll}"
            )
        if not np.array_equal(overall.thresholds, counts.thresholds):
            raise ValueError(f"overall/{scroll} comparison thresholds differ")
        baseline_counts = baseline_by_scroll[scroll]
        if (
            counts.known != baseline_counts.known
            or counts.positive != baseline_counts.positive
        ):
            raise ValueError(f"model/baseline voxel domains differ for {scroll}")
    if (
        comparison.known
        != sum(counts.known for counts in comparison_by_scroll.values())
        or comparison.positive
        != sum(counts.positive for counts in comparison_by_scroll.values())
        or baseline.known != sum(counts.known for counts in baseline_by_scroll.values())
        or baseline.positive
        != sum(counts.positive for counts in baseline_by_scroll.values())
    ):
        raise ValueError("aggregate/per-scroll comparison domains differ")
    points: list[dict[str, Any]] = []
    baseline_metrics = baseline.metrics()
    required_baseline_scrolls = tuple(sorted(baseline_by_scroll))
    for index in range(overall.thresholds.size):
        metrics = overall.metrics_at(index)
        comparison_metrics = comparison.metrics_at(index)
        comparison_metrics["dice_gain_vs_baseline"] = float(
            comparison_metrics["dice"]
        ) - float(baseline_metrics["dice"])
        scrolls: dict[str, dict[str, Any]] = {}
        scroll_dice: list[float] = []
        scroll_gains: dict[str, float] = {}
        for scroll, counts in sorted(by_scroll.items()):
            scroll_metrics = counts.metrics_at(index)
            scroll_dice.append(float(scroll_metrics["dice"]))
            if scroll in baseline_by_scroll:
                scroll_baseline = baseline_by_scroll[scroll].metrics()
                matched_metrics = comparison_by_scroll[scroll].metrics_at(index)
                matched_metrics["baseline_dice"] = scroll_baseline["dice"]
                matched_metrics["dice_gain_vs_baseline"] = float(
                    matched_metrics["dice"]
                ) - float(scroll_baseline["dice"])
                scroll_metrics["baseline_comparison"] = matched_metrics
                # Keep these aliases for report consumers, but derive them only
                # from the exactly matched model/baseline voxel domain.
                scroll_metrics["baseline_dice"] = scroll_baseline["dice"]
                scroll_metrics["dice_gain_vs_baseline"] = matched_metrics[
                    "dice_gain_vs_baseline"
                ]
                scroll_gains[scroll] = float(scroll_metrics["dice_gain_vs_baseline"])
            scrolls[scroll] = scroll_metrics
        qualification = scrolls.get(qualification_scroll)
        macro_scroll_dice = sum(scroll_dice) / max(1, len(scroll_dice))
        macro_scroll_gain = sum(scroll_gains.values()) / max(1, len(scroll_gains))
        minimum_scroll_gain = min(scroll_gains.values(), default=0.0)
        qualified = (
            float(comparison_metrics["dice_gain_vs_baseline"]) > 0
            and qualification is not None
            and float(qualification.get("dice_gain_vs_baseline", 0.0)) > 0
            and bool(required_baseline_scrolls)
            and all(
                scroll_gains.get(scroll, 0.0) > 0
                for scroll in required_baseline_scrolls
            )
        )
        points.append(
            {
                **metrics,
                "baseline_comparison": comparison_metrics,
                "scrolls": scrolls,
                "macro_scroll_dice": macro_scroll_dice,
                "macro_scroll_dice_gain_vs_baseline": macro_scroll_gain,
                "minimum_scroll_dice_gain_vs_baseline": minimum_scroll_gain,
                "qualified": qualified,
            }
        )
    qualified_points = [point for point in points if point["qualified"]]
    pool = qualified_points or points
    selected = max(
        pool,
        key=lambda point: (
            float(point["macro_scroll_dice"]),
            float(point["minimum_scroll_dice_gain_vs_baseline"]),
            float(point["dice"]),
            float(point["precision"]),
            -abs(float(point["threshold"]) - 0.5),
        ),
    )
    default_point = min(points, key=lambda point: abs(float(point["threshold"]) - 0.5))
    return {
        "baseline": baseline_metrics,
        "baseline_by_scroll": {
            scroll: counts.metrics()
            for scroll, counts in sorted(baseline_by_scroll.items())
        },
        "points": points,
        "selected": selected,
        "default": default_point,
        "any_qualified": bool(qualified_points),
        "qualification_scroll": qualification_scroll,
        "required_baseline_scrolls": list(required_baseline_scrolls),
        "selection_metric": "macro_scroll_dice",
        "baseline_comparison_policy": "matched-rows-only",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_report_html(report: dict[str, Any], path: Path) -> None:
    sweep = report["sweep"]
    qualification_scroll = html.escape(str(sweep["qualification_scroll"]))
    required_scrolls = ", ".join(
        html.escape(str(value)) for value in sweep["required_baseline_scrolls"]
    )
    rows = []
    for point in sweep["points"]:
        comparison = point["baseline_comparison"]
        tolerant = point.get("surface_at_2vox", {})
        qualification = point["scrolls"].get(sweep["qualification_scroll"], {})
        qualification_comparison = qualification.get("baseline_comparison", {})
        rows.append(
            "<tr>"
            f"<td>{float(point['threshold']):.2f}</td>"
            f"<td>{float(point['dice']):.5f}</td>"
            f"<td>{float(point['precision']):.5f}</td>"
            f"<td>{float(point['recall']):.5f}</td>"
            f"<td>{float(tolerant.get('f0_5', 0.0)):.5f}</td>"
            f"<td>{float(point.get('foreground_ratio', 0.0)):.3f}</td>"
            f"<td>{float(point['macro_scroll_dice']):.5f}</td>"
            f"<td>{float(comparison['dice_gain_vs_baseline']):+.5f}</td>"
            f"<td>{float(point['minimum_scroll_dice_gain_vs_baseline']):+.5f}</td>"
            f"<td>{float(qualification.get('dice', 0.0)):.5f}</td>"
            f"<td>{float(qualification_comparison.get('dice_gain_vs_baseline', 0.0)):+.5f}</td>"
            f"<td>{'yes' if point['qualified'] else 'no'}</td>"
            "</tr>"
        )
    selected = sweep["selected"]
    document = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Voxel checkpoint audit</title>
<style>
body{{font:16px system-ui;margin:2rem;max-width:1100px}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:.45rem;border:1px solid #bbb;text-align:right}}
th:first-child,td:first-child{{text-align:center}}.pass{{color:#087a37;font-weight:700}}.fail{{color:#ad2633;font-weight:700}}
</style>
<h1>Voxel checkpoint audit</h1>
<p>Checkpoint: {html.escape(str(report["checkpoint"]["path"]))}<br>
Patches: {html.escape(str(report["patch_manifest"]["path"]))}<br>
Split: {html.escape(str(report["options"]["split"]))}</p>
<p class="{"pass" if sweep["any_qualified"] else "fail"}">
Qualification: {"PASS" if sweep["any_qualified"] else "FAIL"}.
Selected threshold {float(selected["threshold"]):.2f}, macro-scroll Dice
{float(selected["macro_scroll_dice"]):.5f}. Requires positive baseline-relative
Dice gain overall and on every baseline-equipped validation scroll
({required_scrolls}); gains compare model and baseline on exactly matched rows.
All-domain Dice still includes registered-real rows without a baseline.
{qualification_scroll} is the designated true-pair gate.
</p>
<table><thead><tr><th>threshold</th><th>Dice</th><th>precision</th><th>recall</th>
<th>surface F0.5 @2 vox</th><th>foreground ratio</th>
<th>macro-scroll Dice</th><th>overall gain</th><th>minimum scroll gain</th>
<th>{qualification_scroll} all-domain Dice</th><th>{qualification_scroll} matched gain</th><th>qualified</th>
</tr></thead><tbody>{"".join(rows)}</tbody></table>
</html>"""
    path.write_text(document, encoding="utf-8", newline="\n")


@torch.no_grad()
def audit_voxel_checkpoint(
    *,
    checkpoint_path: str | Path,
    patch_manifest: str | Path,
    output_path: str | Path,
    options: CheckpointAuditOptions,
) -> Path:
    options.validate()
    configure_cpu_budget(options.max_cpu_threads, reserve_processes=options.num_workers)
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    manifest = Path(patch_manifest).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError(f"audit output already exists: {output}")
    device = torch.device(options.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    assert_cuda_power_limit(device)
    model, payload = load_voxel_checkpoint(checkpoint, device=device)
    dataset = VoxelPatchDataset(manifest, split=options.split, augment=False)
    if not dataset.rows:
        raise ValueError(f"patch manifest has no {options.split!r} rows")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=options.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=options.num_workers > 0,
    )
    thresholds = tuple(sorted(options.thresholds))
    overall = ThresholdCounts.create(thresholds)
    tolerant_overall = TolerantThresholdCounts.create(thresholds)
    comparison = ThresholdCounts.create(thresholds)
    baseline = ScalarCounts()
    by_scroll: dict[str, ThresholdCounts] = {}
    tolerant_by_scroll: dict[str, TolerantThresholdCounts] = {}
    comparison_by_scroll: dict[str, ThresholdCounts] = {}
    baseline_by_scroll: dict[str, ScalarCounts] = {}
    amp_dtype = torch.bfloat16 if options.amp_dtype == "bfloat16" else torch.float16
    model.eval()
    for index, batch in enumerate(loader, 1):
        image = batch["image"].to(device, non_blocking=True)
        probability = (
            _predict_probability(
                model,
                image,
                amp_dtype=amp_dtype,
                autocast_enabled=device.type == "cuda",
                mirror_tta=options.mirror_tta,
            )[0]
            .cpu()
            .numpy()
        )
        target = batch["target"][0, 0].numpy()
        scroll = str(batch["scroll_id"][0])
        overall.update(probability, target)
        tolerant_overall.update(probability, target)
        by_scroll.setdefault(scroll, ThresholdCounts.create(thresholds)).update(
            probability, target
        )
        tolerant_by_scroll.setdefault(
            scroll, TolerantThresholdCounts.create(thresholds)
        ).update(probability, target)
        if bool(batch["has_baseline"][0]):
            baseline_prediction = batch["baseline"][0, 0].numpy() >= 0.5
            comparison.update(probability, target)
            comparison_by_scroll.setdefault(
                scroll, ThresholdCounts.create(thresholds)
            ).update(probability, target)
            baseline.update(baseline_prediction, target)
            baseline_by_scroll.setdefault(scroll, ScalarCounts()).update(
                baseline_prediction, target
            )
        if index % 10 == 0 or index == len(dataset):
            print(
                f"checkpoint audit {index:,}/{len(dataset):,} patches",
                flush=True,
            )
    sweep = summarize_thresholds(
        overall=overall,
        comparison=comparison,
        baseline=baseline,
        by_scroll=by_scroll,
        comparison_by_scroll=comparison_by_scroll,
        baseline_by_scroll=baseline_by_scroll,
        qualification_scroll=options.qualification_scroll,
    )
    add_tolerant_surface_metrics(
        sweep,
        overall=tolerant_overall,
        by_scroll=tolerant_by_scroll,
    )
    report = {
        "schema": CHECKPOINT_AUDIT_SCHEMA,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "epoch": payload.get("epoch"),
            "metrics": payload.get("metrics"),
        },
        "patch_manifest": {
            "path": str(manifest),
            "sha256": _sha256(manifest),
            "records": len(dataset),
        },
        "options": {
            "split": options.split,
            "thresholds": list(thresholds),
            "qualification_scroll": options.qualification_scroll,
            "device": options.device,
            "amp_dtype": options.amp_dtype,
            "mirror_tta": options.mirror_tta,
            "num_workers": options.num_workers,
            "max_cpu_threads": options.max_cpu_threads,
        },
        "sweep": sweep,
    }
    temporary = output.with_name(output.name + f".partial-{os.getpid()}")
    temporary.mkdir(parents=True)
    (temporary / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report_html(report, temporary / "index.html")
    os.replace(temporary, output)
    return output / "index.html"
