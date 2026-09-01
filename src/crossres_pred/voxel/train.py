from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .loss import (
    DEFAULT_VOXEL_LOSS_OPTIONS,
    VoxelLossOptions,
    deep_supervision_loss,
    loss_contract,
)
from .model import (
    NNUNetConfig,
    VoxelNNUNet,
    initialize_from_m7,
    verify_released_m7_checkpoint,
)
from .patches import VoxelPatchDataset, load_patch_manifest
from .resources import assert_cuda_power_limit, configure_cpu_budget
from .trust_region import (
    M7_PARAMETER_TRUST_REGION_CONTRACT,
    M7ParameterTrustRegion,
)

CHECKPOINT_SELECTION_CONTRACT = "calibrated-macro-scroll-guarded-v3"
CHECKPOINT_DURABILITY_CONTRACT = "checkpoint-last-commit-reconcile-v2"
SNAPSHOT_CHECKPOINT_CONTRACT = "post-optimizer-model-only-sample-milestone-v1"
FINAL_FIT_CHECKPOINT_CONTRACT = "selected-sample-budget-last-completed-v2"
EPOCH_PARTITION_CONTRACT = "sample-budget-evaluation-interval-v2"
SAMPLING_CONTRACT = "proportional-scroll-source-pathology-v1"
LEARNING_RATE_CONTRACT = "sample-progress-v1"
ADAMW_OPTIMIZER_CONTRACT = "torch-adamw-explicit-betas-eps-v1"
ADAM_OPTIMIZER_CONTRACT = "torch-adam-explicit-betas-eps-v1"
DEFAULT_VALIDATION_THRESHOLDS = tuple(round(0.05 * index, 2) for index in range(1, 20))


@dataclass(frozen=True)
class TrainOptions:
    epochs: int = 100
    batch_size: int = 1
    accumulate: int = 3
    learning_rate: float = 1.0e-3
    weight_decay: float = 3.0e-5
    momentum: float = 0.99
    optimizer: str = "sgd"
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    adamw_eps: float = 1.0e-8
    m7_trust_region_relative_l2: float = 0.0
    loss_options: VoxelLossOptions = DEFAULT_VOXEL_LOSS_OPTIONS
    num_workers: int = 2
    seed: int = 1203
    device: str = "cuda"
    amp: bool = True
    amp_dtype: str = "auto"
    preset: str = "m7-resenc-l"
    pretrained_m7_checkpoint: str | None = None
    pinned_medial_bridge_state: str | None = None
    dynamic_medial_connectivity_state: str | None = None
    final_fit: bool = False
    allow_spatial_validation: bool = False
    gradient_clip_norm: float = 12.0
    max_cpu_threads: int = 16
    samples_per_epoch: int | None = None
    max_train_samples: int | None = None
    lr_schedule: str = "poly"
    lr_floor_ratio: float = 0.0
    warmup_samples: int = 0
    stratified_sampling: bool = False
    train_augmentation: bool = True
    validation_thresholds: tuple[float, ...] = DEFAULT_VALIDATION_THRESHOLDS
    minimum_scroll_gain: float = -0.01
    checkpoint_min_delta: float = 0.0
    early_stopping_patience: int | None = None

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.accumulate <= 0:
            raise ValueError("epochs, batch_size, and accumulate must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning rate/weight decay are invalid")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if self.optimizer not in {"sgd", "adam", "adamw"}:
            raise ValueError("optimizer must be sgd, adam, or adamw")
        if not 0 <= self.adamw_beta1 < 1 or not 0 <= self.adamw_beta2 < 1:
            raise ValueError("AdamW betas must be in [0, 1)")
        if self.adamw_eps <= 0 or not math.isfinite(self.adamw_eps):
            raise ValueError("AdamW epsilon must be finite and positive")
        if (
            not math.isfinite(self.m7_trust_region_relative_l2)
            or self.m7_trust_region_relative_l2 < 0
        ):
            raise ValueError("M7 trust-region radius must be finite and non-negative")
        self.loss_options.validate()
        if self.num_workers < 0 or self.gradient_clip_norm < 0:
            raise ValueError("worker count/gradient clipping are invalid")
        if self.samples_per_epoch is not None and self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        if self.max_train_samples is not None and self.max_train_samples <= 0:
            raise ValueError("max_train_samples must be positive")
        if self.lr_schedule not in {"constant", "poly", "cosine"}:
            raise ValueError("lr_schedule must be constant, poly, or cosine")
        if not 0 <= self.lr_floor_ratio <= 1:
            raise ValueError("lr_floor_ratio must be in [0, 1]")
        if self.warmup_samples < 0:
            raise ValueError("warmup_samples must be non-negative")
        if self.early_stopping_patience is not None and (
            self.early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be positive")
        if not self.validation_thresholds:
            raise ValueError("validation_thresholds must not be empty")
        if tuple(sorted(set(self.validation_thresholds))) != self.validation_thresholds:
            raise ValueError("validation_thresholds must be sorted and unique")
        if any(not 0 < value < 1 for value in self.validation_thresholds):
            raise ValueError("validation_thresholds must be in (0, 1)")
        if 0.5 not in self.validation_thresholds:
            raise ValueError("validation_thresholds must include 0.5")
        if not math.isfinite(self.minimum_scroll_gain):
            raise ValueError("minimum_scroll_gain must be finite")
        if self.checkpoint_min_delta < 0 or not math.isfinite(
            self.checkpoint_min_delta
        ):
            raise ValueError("checkpoint_min_delta must be finite and non-negative")
        if not 1 <= self.max_cpu_threads <= 16:
            raise ValueError("max_cpu_threads must be in [1, 16]")
        if self.num_workers >= self.max_cpu_threads:
            raise ValueError("num_workers must be smaller than max_cpu_threads")
        if self.amp_dtype not in {"auto", "bfloat16", "float16"}:
            raise ValueError("amp_dtype must be auto, bfloat16, or float16")
        NNUNetConfig(preset=self.preset)
        if self.preset == "m7-resenc-l" and not self.pretrained_m7_checkpoint:
            raise ValueError("m7-resenc-l training requires the released checkpoint")
        if (
            self.loss_options.m7_anchor_weight > 0
            or self.loss_options.m7_preservation_weight > 0
        ) and not self.pretrained_m7_checkpoint:
            raise ValueError("M7-anchored loss requires the released checkpoint")
        if self.m7_trust_region_relative_l2 > 0 and not self.pretrained_m7_checkpoint:
            raise ValueError(
                "M7 parameter trust region requires the released checkpoint"
            )
        if self.loss_options.pinned_axial_weight > 0:
            if not self.pinned_medial_bridge_state:
                raise ValueError("pinned axial loss requires a bridge atlas state")
            if not Path(self.pinned_medial_bridge_state).expanduser().is_file():
                raise ValueError("pinned medial bridge atlas state is missing")
        if self.loss_options.dynamic_medial_connectivity_weight > 0:
            if not self.dynamic_medial_connectivity_state:
                raise ValueError(
                    "dynamic medial connectivity loss requires an atlas state"
                )
            if not Path(self.dynamic_medial_connectivity_state).expanduser().is_file():
                raise ValueError("dynamic medial connectivity atlas state is missing")


class EpochPartitionSampler(Sampler[int]):
    """Deterministically partition shuffled corpus passes into short epochs."""

    def __init__(
        self,
        dataset_size: int,
        samples_per_epoch: int,
        seed: int,
        *,
        total_samples: int | None = None,
    ) -> None:
        if dataset_size <= 0 or samples_per_epoch <= 0:
            raise ValueError("dataset_size and samples_per_epoch must be positive")
        if total_samples is not None and total_samples <= 0:
            raise ValueError("total_samples must be positive")
        self.dataset_size = dataset_size
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.total_samples = total_samples
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        if self.total_samples is not None:
            remaining = self.total_samples - self.epoch * self.samples_per_epoch
            return max(0, min(self.samples_per_epoch, remaining))
        return self.samples_per_epoch

    def __iter__(self):
        absolute = self.epoch * self.samples_per_epoch
        remaining = len(self)
        while remaining:
            cycle, offset = divmod(absolute, self.dataset_size)
            generator = torch.Generator().manual_seed(self.seed + cycle)
            permutation = torch.randperm(
                self.dataset_size,
                generator=generator,
            )
            take = min(remaining, self.dataset_size - offset)
            for index in permutation[offset : offset + take].tolist():
                yield int(index)
            absolute += take
            remaining -= take


def _pathology_band(score: float) -> str:
    return "high" if score >= 0.10 else ("low" if score > 0 else "zero")


class StratifiedEpochPartitionSampler(EpochPartitionSampler):
    """Visit every row once per pass while keeping prefixes representative.

    Rows are independently shuffled within scroll/source/pathology strata, then
    interleaved by normalized position. Small strata are not oversampled, and a
    complete cycle remains an exact permutation of the corpus.
    """

    def __init__(
        self,
        rows: Sequence[Any],
        samples_per_epoch: int,
        seed: int,
        *,
        total_samples: int | None = None,
    ) -> None:
        super().__init__(
            len(rows),
            samples_per_epoch,
            seed,
            total_samples=total_samples,
        )
        grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            grouped[
                (
                    str(row.scroll_id),
                    str(row.supervision_source),
                    _pathology_band(float(row.pathology_score)),
                )
            ].append(index)
        self._groups = tuple(
            (key, tuple(indices)) for key, indices in sorted(grouped.items())
        )

    def _cycle_order(self, cycle: int) -> list[int]:
        ranked: list[tuple[float, str, int]] = []
        for key, indices in self._groups:
            key_text = "\x1f".join(key)
            key_seed = int.from_bytes(
                hashlib.sha256(key_text.encode("utf-8")).digest()[:8], "little"
            )
            generator = torch.Generator().manual_seed(
                (self.seed + cycle * 1_000_003 + key_seed) % (2**63 - 1)
            )
            permutation = torch.randperm(len(indices), generator=generator).tolist()
            jitter = torch.rand(len(indices), generator=generator).tolist()
            size = len(indices)
            for rank, local_index in enumerate(permutation):
                position = (rank + 0.25 + 0.5 * jitter[rank]) / size
                ranked.append((position, key_text, indices[local_index]))
        ranked.sort()
        return [index for _, _, index in ranked]

    def __iter__(self):
        absolute = self.epoch * self.samples_per_epoch
        remaining = len(self)
        while remaining:
            cycle, offset = divmod(absolute, self.dataset_size)
            order = self._cycle_order(cycle)
            take = min(remaining, self.dataset_size - offset)
            yield from order[offset : offset + take]
            absolute += take
            remaining -= take


@dataclass
class BinaryCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    known: int = 0
    positive: int = 0

    def update_prediction(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if target.ndim == 5:
            target = target[:, 0]
        if prediction.ndim == 5:
            prediction = (
                prediction[:, 0] >= 0.5
                if prediction.shape[1] == 1
                else torch.argmax(prediction, dim=1)
            )
        predicted = prediction == 1
        truth = target == 1
        valid = target != 2
        self.true_positive += int((predicted & truth & valid).sum())
        self.false_positive += int((predicted & ~truth & valid).sum())
        self.false_negative += int((~predicted & truth & valid).sum())
        self.known += int(valid.sum())
        self.positive += int((truth & valid).sum())

    def metrics(self) -> dict[str, float]:
        return {
            "dice": (2.0 * self.true_positive)
            / max(
                1, 2 * self.true_positive + self.false_positive + self.false_negative
            ),
            "precision": self.true_positive
            / max(1, self.true_positive + self.false_positive),
            "recall": self.true_positive
            / max(1, self.true_positive + self.false_negative),
            "known_voxels": float(self.known),
            "positive_prevalence": self.positive / max(1, self.known),
        }


class ThresholdHistogram:
    """Exact binary counts for many thresholds without retaining voxel scores."""

    def __init__(self, thresholds: Sequence[float]) -> None:
        self.thresholds = tuple(float(value) for value in thresholds)
        self.positive_bins = torch.zeros(len(self.thresholds) + 1, dtype=torch.int64)
        self.negative_bins = torch.zeros(len(self.thresholds) + 1, dtype=torch.int64)

    def update_probability(
        self, probability: torch.Tensor, target: torch.Tensor
    ) -> None:
        if target.ndim == 5:
            target = target[:, 0]
        if probability.ndim == 5:
            probability = probability[:, 0]
        valid = target != 2
        if not bool(valid.any()):
            return
        values = probability[valid].float()
        truth = target[valid] == 1
        boundaries = torch.tensor(
            self.thresholds, dtype=values.dtype, device=values.device
        )
        # right=True puts values exactly equal to a threshold in its positive bin.
        bins = torch.bucketize(values, boundaries, right=True)
        positive = torch.bincount(bins[truth], minlength=len(self.thresholds) + 1).cpu()
        negative = torch.bincount(
            bins[~truth], minlength=len(self.thresholds) + 1
        ).cpu()
        self.positive_bins += positive
        self.negative_bins += negative

    def metrics(self, threshold_index: int) -> dict[str, float]:
        if not 0 <= threshold_index < len(self.thresholds):
            raise IndexError(threshold_index)
        first_positive_bin = threshold_index + 1
        true_positive = int(self.positive_bins[first_positive_bin:].sum())
        false_positive = int(self.negative_bins[first_positive_bin:].sum())
        positive = int(self.positive_bins.sum())
        false_negative = positive - true_positive
        known = positive + int(self.negative_bins.sum())
        return {
            "dice": (2.0 * true_positive)
            / max(1, 2 * true_positive + false_positive + false_negative),
            "precision": true_positive / max(1, true_positive + false_positive),
            "recall": true_positive / max(1, positive),
            "known_voxels": float(known),
            "positive_prevalence": positive / max(1, known),
        }


def _calibrated_validation_metrics(
    global_histogram: ThresholdHistogram,
    scroll_histograms: dict[str, ThresholdHistogram],
) -> dict[str, float]:
    candidates: list[tuple[tuple[float, float, float, float], int]] = []
    curve: list[tuple[float, dict[str, float], list[float]]] = []
    for index, threshold in enumerate(global_histogram.thresholds):
        metrics = global_histogram.metrics(index)
        scroll_dice = [
            histogram.metrics(index)["dice"] for histogram in scroll_histograms.values()
        ]
        macro = sum(scroll_dice) / max(1, len(scroll_dice))
        minimum = min(scroll_dice, default=metrics["dice"])
        candidates.append(
            ((macro, minimum, metrics["dice"], -abs(threshold - 0.5)), index)
        )
        curve.append((threshold, metrics, scroll_dice))
    selected = max(candidates)[1]
    threshold, metrics, scroll_dice = curve[selected]
    result = {
        "calibrated_threshold": threshold,
        "calibrated_dice": metrics["dice"],
        "calibrated_precision": metrics["precision"],
        "calibrated_recall": metrics["recall"],
        "calibrated_macro_scroll_dice": sum(scroll_dice) / max(1, len(scroll_dice)),
        "calibrated_minimum_scroll_dice": min(scroll_dice, default=metrics["dice"]),
    }
    for index, (
        candidate_threshold,
        candidate_metrics,
        candidate_scroll_dice,
    ) in enumerate(curve):
        prefix = f"threshold/{candidate_threshold:.2f}"
        result[f"{prefix}/dice"] = candidate_metrics["dice"]
        result[f"{prefix}/precision"] = candidate_metrics["precision"]
        result[f"{prefix}/recall"] = candidate_metrics["recall"]
        result[f"{prefix}/macro_scroll_dice"] = sum(candidate_scroll_dice) / max(
            1, len(candidate_scroll_dice)
        )
        if index == selected:
            for scroll, histogram in sorted(scroll_histograms.items()):
                result[f"stratum/scroll/{scroll}/calibrated_dice"] = histogram.metrics(
                    index
                )["dice"]
    return result


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device(name: str) -> torch.device:
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but this Python environment has CPU-only torch"
        )
    return result


def _build_optimizer(
    model: torch.nn.Module,
    options: TrainOptions,
) -> torch.optim.Optimizer:
    if options.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=options.learning_rate,
            weight_decay=options.weight_decay,
            betas=(options.adamw_beta1, options.adamw_beta2),
            eps=options.adamw_eps,
        )
    if options.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=options.learning_rate,
            weight_decay=options.weight_decay,
            betas=(options.adamw_beta1, options.adamw_beta2),
            eps=options.adamw_eps,
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
        momentum=options.momentum,
        nesterov=True,
    )


def _options_identity(options: TrainOptions) -> dict[str, Any]:
    value = asdict(options)
    if options.optimizer == "sgd":
        # Preserve the identity of runs created before AdamW became an explicit
        # option. SGD and its momentum field were already the only contract.
        for key in ("optimizer", "adamw_beta1", "adamw_beta2", "adamw_eps"):
            value.pop(key)
    if options.loss_options.is_legacy:
        value.pop("loss_options")
    elif not options.loss_options.m7_anchor_confident_agreement:
        # Preserve the identity of runs created before confident-agreement
        # anchoring became an explicit option.
        value["loss_options"].pop("m7_anchor_confident_agreement")
    if "loss_options" in value:
        if options.loss_options.medial_recall_weight == 0:
            value["loss_options"].pop("medial_recall_weight")
        if options.loss_options.m7_anchor_unknown_corridor_radius == 0:
            value["loss_options"].pop("m7_anchor_unknown_corridor_radius")
        if options.loss_options.m7_preservation_weight == 0:
            value["loss_options"].pop("m7_preservation_weight")
            value["loss_options"].pop("m7_preservation_radius")
            value["loss_options"].pop("m7_preservation_anchor_threshold")
            value["loss_options"].pop("m7_preservation_soft_floor")
        elif not options.loss_options.m7_preservation_soft_floor:
            # Preserve identities created before the optional calibrated
            # one-sided preservation floor existed.
            value["loss_options"].pop("m7_preservation_soft_floor")
        if options.loss_options.pinned_axial_weight == 0:
            value["loss_options"].pop("pinned_axial_weight")
            value["loss_options"].pop("pinned_axial_probability_floor")
            value["loss_options"].pop("pinned_axial_bottom_fraction")
        if options.loss_options.dynamic_medial_connectivity_weight == 0:
            value["loss_options"].pop("dynamic_medial_connectivity_weight")
            value["loss_options"].pop("dynamic_medial_connectivity_probability_floor")
            value["loss_options"].pop("dynamic_medial_connectivity_steps")
    if options.m7_trust_region_relative_l2 == 0:
        value.pop("m7_trust_region_relative_l2")
    if options.pinned_medial_bridge_state is None:
        value.pop("pinned_medial_bridge_state")
    if options.dynamic_medial_connectivity_state is None:
        value.pop("dynamic_medial_connectivity_state")
    if options.train_augmentation:
        # Preserve the identity of runs created before augmentation became an
        # explicit diagnostic switch.  Production behavior remains unchanged;
        # only no-augmentation controls acquire a new identity field.
        value.pop("train_augmentation")
    return value


def _amp_configuration(
    options: TrainOptions, device: torch.device
) -> tuple[torch.dtype, bool, bool]:
    enabled = options.amp and device.type == "cuda"
    if not enabled:
        return torch.float32, False, False
    if options.amp_dtype == "bfloat16" or (
        options.amp_dtype == "auto" and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16, True, False
    return torch.float16, True, True


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Tolerate brief Windows sharing locks from concurrent training readers."""

    attempts = 100
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_history_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _history_epoch(row: dict[str, Any]) -> int:
    epoch = row.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"invalid history epoch: {epoch!r}")
    return epoch


def _read_history_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            if index == len(lines) - 1:
                # A killed legacy append may leave only its final line torn.
                break
            raise ValueError(f"{path}: invalid history line {index + 1}") from error
        if not isinstance(value, dict):
            raise TypeError(f"{path}: history line {index + 1} is not an object")
        _history_epoch(value)
        rows.append(value)
    return rows


def _reconcile_history(
    path: Path,
    *,
    checkpoint_epoch: int,
    checkpoint_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    if checkpoint_epoch < 0 or _history_epoch(checkpoint_metrics) != checkpoint_epoch:
        raise ValueError("checkpoint epoch and metrics row disagree")
    existing = _read_history_rows(path)
    committed_prefix = [
        row for row in existing if _history_epoch(row) < checkpoint_epoch
    ]
    prefix_epochs = [_history_epoch(row) for row in committed_prefix]
    if prefix_epochs != list(range(checkpoint_epoch)):
        raise ValueError(
            f"{path}: committed history epochs {prefix_epochs} are not "
            f"0..{checkpoint_epoch - 1}"
        )
    rows = [*committed_prefix, dict(checkpoint_metrics)]
    _write_history_rows(path, rows)
    return rows


def _model_only_checkpoint_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("optimizer", None)
    return result


def _reconcile_best_checkpoint_artifacts(
    payload: dict[str, Any],
    *,
    best_checkpoint: Path,
    best_trained_checkpoint: Path,
    final_fit: bool,
) -> None:
    """Repair derived best artifacts from the authoritative last checkpoint."""

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError("resume checkpoint has no metrics row")
    score = _history_row_score(metrics)
    best_score = float(payload.get("best_score", float("-inf")))
    best_trained_score = float(payload.get("best_trained_score", float("-inf")))
    roles = payload.get("checkpoint_roles")
    if roles is not None and not isinstance(roles, dict):
        raise TypeError("resume checkpoint roles must be an object")

    if isinstance(roles, dict):
        committed_as_best = bool(roles.get("best", False))
        committed_as_best_trained = bool(roles.get("best_trained", False))
    else:
        # Backward-compatible recovery for checkpoints written before roles were
        # explicit. Validation eligibility prevents an exact-score tie with the
        # initial M7 checkpoint from being promoted accidentally.
        validation = metrics.get("val")
        eligible = final_fit or (
            isinstance(validation, dict)
            and bool(validation.get("checkpoint_eligible", False))
        )
        committed_as_best = eligible and score == best_score
        committed_as_best_trained = score == best_trained_score

    model_only = _model_only_checkpoint_payload(payload)
    if committed_as_best:
        _atomic_torch_save(best_checkpoint, model_only)
    elif not best_checkpoint.is_file():
        raise ValueError("resume run is missing its committed best checkpoint")
    if committed_as_best_trained:
        _atomic_torch_save(best_trained_checkpoint, model_only)
    elif not final_fit and not best_trained_checkpoint.is_file():
        raise ValueError("resume run is missing its committed best-trained checkpoint")


def _reconcile_snapshot_records(
    output: Path,
    index_path: Path,
    *,
    snapshots: tuple[int, ...],
    cumulative_samples: int,
) -> list[dict[str, Any]]:
    """Keep only milestone records covered by the last committed interval."""

    committed = tuple(value for value in snapshots if value <= cumulative_samples)
    if not index_path.is_file():
        if committed:
            raise ValueError("resume run is missing its checkpoint milestone index")
        return []

    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema") != SNAPSHOT_CHECKPOINT_CONTRACT:
        raise ValueError("checkpoint milestone index contract changed")
    raw_records = index.get("records")
    if not isinstance(raw_records, list):
        raise TypeError("checkpoint milestone index has no records")
    if len(raw_records) < len(committed):
        raise ValueError("checkpoint milestone index is behind committed samples")

    records = list(raw_records[: len(committed)])
    if tuple(int(row["requested_samples"]) for row in records) != committed:
        raise ValueError("checkpoint milestone index differs from committed samples")
    for row in records:
        path = output / str(row["checkpoint"])
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"checkpoint milestone artifact is missing: {path}")
        actual_samples = int(row["actual_samples"])
        requested_samples = int(row["requested_samples"])
        if not requested_samples <= actual_samples <= cumulative_samples:
            raise ValueError("checkpoint milestone sample count is invalid")

    if len(raw_records) != len(records):
        # A milestone is written before checkpoint_last. A power loss in that
        # narrow window leaves an uncommitted suffix, which will be regenerated
        # after the optimizer/RNG state resumes from checkpoint_last.
        _write_json(
            index_path,
            {"schema": SNAPSHOT_CHECKPOINT_CONTRACT, "records": records},
        )
    return records


def _history_row_score(row: dict[str, Any]) -> float:
    validation = row.get("val")
    training = row.get("train")
    if isinstance(validation, dict) and "calibrated_macro_scroll_dice" in validation:
        score = float(validation["calibrated_macro_scroll_dice"])
    elif isinstance(validation, dict) and "macro_scroll_dice" in validation:
        score = float(validation["macro_scroll_dice"])
    elif isinstance(validation, dict) and "dice" in validation:
        score = float(validation["dice"])
    elif isinstance(training, dict) and "loss_total" in training:
        score = -float(training["loss_total"])
    else:
        raise ValueError("history row has neither validation Dice nor training loss")
    if not math.isfinite(score):
        raise ValueError(f"history score is not finite: {score}")
    return score


def _validation_score(validation: dict[str, Any]) -> float:
    return _history_row_score(
        {"epoch": 0, "train": {}, "val": validation, "learning_rate": 0.0}
    )


def _calibrated_scroll_scores(validation: dict[str, Any]) -> dict[str, float]:
    prefix = "stratum/scroll/"
    suffix = "/calibrated_dice"
    return {
        key[len(prefix) : -len(suffix)]: float(value)
        for key, value in validation.items()
        if key.startswith(prefix) and key.endswith(suffix)
    }


def _checkpoint_guard(
    validation: dict[str, Any],
    initial_validation: dict[str, Any],
    *,
    minimum_scroll_gain: float,
    checkpoint_min_delta: float,
) -> tuple[bool, float]:
    score_gain = _validation_score(validation) - _validation_score(initial_validation)
    initial_scrolls = _calibrated_scroll_scores(initial_validation)
    candidate_scrolls = _calibrated_scroll_scores(validation)
    common = sorted(initial_scrolls.keys() & candidate_scrolls.keys())
    if not common:
        raise ValueError("calibrated validation has no common per-scroll scores")
    minimum_gain = min(
        candidate_scrolls[scroll] - initial_scrolls[scroll] for scroll in common
    )
    return (
        score_gain >= checkpoint_min_delta and minimum_gain >= minimum_scroll_gain,
        minimum_gain,
    )


def _select_best_checkpoint(
    score: float,
    best_score: float,
    *,
    final_fit: bool,
) -> tuple[bool, float]:
    """Select by held-out score during tuning and by completion during final fit."""

    if not math.isfinite(score):
        raise ValueError(f"checkpoint score is not finite: {score}")
    if final_fit:
        # There is deliberately no validation split after qualification. Every
        # later epoch has seen strictly more of the fixed fitting schedule, so
        # a noisy microepoch training loss must not roll the production model
        # back to an earlier, partially fitted state.
        return True, score
    return score > best_score, max(best_score, score)


def _log_tensorboard_row(writer: Any, row: dict[str, Any]) -> None:
    epoch = _history_epoch(row)
    training = row.get("train")
    validation = row.get("val")
    if not isinstance(training, dict) or not isinstance(validation, dict):
        raise TypeError("history row train/val metrics must be objects")
    for name, value in training.items():
        label = str(name)
        category = "loss" if label.startswith("loss_") else "metric"
        writer.add_scalar(
            f"{category}/train/{label.removeprefix('loss_')}", float(value), epoch
        )
    for name, value in validation.items():
        label = str(name)
        category = "loss" if label.startswith("loss_") else "metric"
        writer.add_scalar(
            f"{category}/val/{label.removeprefix('loss_')}", float(value), epoch
        )
    writer.add_scalar("optimizer/learning_rate", float(row["learning_rate"]), epoch)


def _log_tensorboard_initial_validation(
    writer: Any,
    validation: dict[str, Any],
    *,
    total_samples: int,
    evaluation_interval_samples: int,
    learning_rate: float,
) -> None:
    """Make a fresh run observable before its first evaluation interval ends."""

    for name, value in validation.items():
        label = str(name)
        category = "loss" if label.startswith("loss_") else "metric"
        writer.add_scalar(
            f"{category}/initial/{label.removeprefix('loss_')}", float(value), 0
        )
    writer.add_scalar("progress/cumulative_samples", 0.0, 0)
    writer.add_scalar("progress/total_samples", float(total_samples), 0)
    writer.add_scalar(
        "progress/evaluation_interval_samples",
        float(evaluation_interval_samples),
        0,
    )
    writer.add_scalar("optimizer/initial_learning_rate", float(learning_rate), 0)


def _capture_rng_state(
    loader_generator: torch.Generator, device: torch.device
) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        "loader_generator": loader_generator.get_state(),
    }


def _restore_rng_state(
    state: object,
    *,
    loader_generator: torch.Generator,
    device: torch.device,
) -> None:
    if not isinstance(state, dict):
        return
    required = {"python", "numpy", "torch", "cuda", "loader_generator"}
    if set(state) != required:
        raise ValueError("checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    loader_generator.set_state(state["loader_generator"])
    if device.type == "cuda":
        if state["cuda"] is None:
            raise ValueError("CUDA checkpoint has no CUDA RNG state")
        torch.cuda.set_rng_state_all(state["cuda"])


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _validate_scroll_split(
    manifest: Path,
    validation_manifest: Path | None,
    allow_spatial_validation: bool,
    final_fit: bool,
) -> int:
    rows = load_patch_manifest(manifest)
    train_scrolls = {row.scroll_id for row in rows if row.split == "train"}
    validation_rows = (
        load_patch_manifest(validation_manifest)
        if validation_manifest is not None
        else rows
    )
    val_scrolls = {row.scroll_id for row in validation_rows if row.split == "val"}
    overlap = train_scrolls & val_scrolls
    if overlap and not allow_spatial_validation:
        raise ValueError(
            "train/val scroll leakage; use a different held-out scroll or explicitly "
            f"allow spatial validation: {sorted(overlap)}"
        )
    if not final_fit and not val_scrolls:
        raise ValueError("validation patches are required unless final_fit is enabled")
    return len(rows) if final_fit else sum(row.split == "train" for row in rows)


def _bounded_partition_samples(
    dataset_size: int,
    *,
    epochs: int,
    samples_per_epoch: int | None,
) -> int | None:
    """Cap a schedule that brackets exactly one corpus pass.

    The final microepoch is allowed to be shorter so a nominal ceiling such as
    10 x 25,109 does not repeat two rows from a 251,088-row corpus. Schedules
    that clearly request less than or more than one pass retain their original
    cyclic behavior.
    """

    if samples_per_epoch is None:
        return None
    preceding_capacity = (epochs - 1) * samples_per_epoch
    total_capacity = epochs * samples_per_epoch
    if preceding_capacity < dataset_size <= total_capacity:
        return dataset_size
    return None


@dataclass(frozen=True)
class TrainingSchedule:
    evaluation_interval_samples: int
    total_samples: int
    evaluation_intervals: int


def _resolve_training_schedule(
    dataset_size: int,
    *,
    epochs: int,
    samples_per_epoch: int | None,
    max_train_samples: int | None,
) -> TrainingSchedule:
    interval = samples_per_epoch or dataset_size
    if max_train_samples is not None:
        total = max_train_samples
    else:
        bounded = _bounded_partition_samples(
            dataset_size,
            epochs=epochs,
            samples_per_epoch=samples_per_epoch,
        )
        total = bounded if bounded is not None else epochs * interval
    return TrainingSchedule(
        evaluation_interval_samples=interval,
        total_samples=total,
        evaluation_intervals=math.ceil(total / interval),
    )


def _learning_rate_for_samples(
    options: TrainOptions,
    *,
    samples_seen: int,
    total_samples: int,
) -> float:
    if total_samples <= 0 or not 0 <= samples_seen <= total_samples:
        raise ValueError("invalid sample progress for learning-rate schedule")
    if options.warmup_samples and samples_seen < options.warmup_samples:
        warmup_fraction = max(
            1.0 / options.warmup_samples,
            samples_seen / options.warmup_samples,
        )
        return options.learning_rate * warmup_fraction
    scheduled_samples = max(1, total_samples - options.warmup_samples)
    progress = min(
        1.0,
        max(0.0, (samples_seen - options.warmup_samples) / scheduled_samples),
    )
    if options.lr_schedule == "constant":
        multiplier = 1.0
    elif options.lr_schedule == "cosine":
        multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        multiplier = (1.0 - progress) ** 0.9
    multiplier = options.lr_floor_ratio + (1.0 - options.lr_floor_ratio) * multiplier
    return options.learning_rate * multiplier


def _normalize_snapshot_samples(
    values: Sequence[int], *, total_samples: int
) -> tuple[int, ...]:
    snapshots = tuple(int(value) for value in values)
    if tuple(sorted(set(snapshots))) != snapshots:
        raise ValueError("snapshot_samples must be sorted and unique")
    if any(value <= 0 for value in snapshots):
        raise ValueError("snapshot_samples must be positive")
    if snapshots and snapshots[-1] > total_samples:
        raise ValueError("snapshot_samples exceed the training sample budget")
    return snapshots


def _snapshot_checkpoint_path(output: Path, requested_samples: int) -> Path:
    return output / f"checkpoint_milestone_{requested_samples:08d}.pt"


@torch.no_grad()
def validate_model(
    model: VoxelNNUNet,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    amp_dtype: torch.dtype,
    autocast_enabled: bool,
    thresholds: Sequence[float] = DEFAULT_VALIDATION_THRESHOLDS,
    loss_options: VoxelLossOptions = DEFAULT_VOXEL_LOSS_OPTIONS,
    m7_anchor_model: VoxelNNUNet | None = None,
) -> dict[str, float]:
    validation_loss_options = replace(
        loss_options,
        m7_anchor_weight=0.0,
        pinned_axial_weight=0.0,
        dynamic_medial_connectivity_weight=0.0,
    )
    if validation_loss_options.m7_preservation_weight > 0 and m7_anchor_model is None:
        raise ValueError("M7 preservation validation requires the frozen M7 model")
    model.eval()
    if m7_anchor_model is not None:
        m7_anchor_model.eval()
    sums: dict[str, float] = {}
    batches = 0
    model_counts = BinaryCounts()
    baseline_counts = BinaryCounts()
    baseline_batches = 0
    model_strata: dict[str, BinaryCounts] = {}
    baseline_strata: dict[str, BinaryCounts] = {}
    threshold_histogram = ThresholdHistogram(thresholds)
    scroll_threshold_histograms: dict[str, ThresholdHistogram] = {}
    probability_min = float("inf")
    probability_max = float("-inf")
    probability_sum = 0.0
    probability_count = 0
    crest_probability_min = float("inf")
    crest_probability_max = float("-inf")
    crest_probability_sum = 0.0
    crest_probability_count = 0
    crest_probability_above_0_4 = 0
    crest_probability_above_0_5 = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled
        ):
            m7_anchor_outputs = (
                m7_anchor_model(batch["image"])
                if validation_loss_options.m7_preservation_weight > 0
                and m7_anchor_model is not None
                else None
            )
            outputs = model(batch["image"])
            _, components = deep_supervision_loss(
                outputs,
                batch["target"],
                batch.get("teacher_q"),
                batch.get("target_valid"),
                teacher_crest=batch.get("teacher_crest"),
                teacher_crest_valid=batch.get("teacher_crest_valid"),
                teacher_crest_available=batch.get("has_teacher_crest"),
                m7_anchor_outputs=m7_anchor_outputs,
                pinned_medial_bridge=batch.get("pinned_medial_bridge"),
                dynamic_connectivity_event=batch.get("dynamic_connectivity_event"),
                dynamic_connectivity_pins=batch.get("dynamic_connectivity_pins"),
                dynamic_connectivity_free=batch.get("dynamic_connectivity_free"),
                options=validation_loss_options,
            )
        for name, value in components.items():
            sums[name] = sums.get(name, 0.0) + float(value)
        model_counts.update_prediction(outputs[0], batch["target"])
        probability = torch.softmax(outputs[0], dim=1)[:, 1]
        threshold_histogram.update_probability(probability, batch["target"])
        valid = batch["target"][:, 0] != 2
        if bool(valid.any()):
            known_probability = probability[valid]
            probability_min = min(probability_min, float(known_probability.min()))
            probability_max = max(probability_max, float(known_probability.max()))
            probability_sum += float(known_probability.double().sum())
            probability_count += int(known_probability.numel())
        if "teacher_crest" in batch:
            crest = batch["teacher_crest"][:, 0] > 0.5
            crest &= batch["teacher_crest_valid"][:, 0] > 0.5
            crest &= batch["has_teacher_crest"].bool()[:, None, None, None]
            if bool(crest.any()):
                crest_probability = probability[crest]
                crest_probability_min = min(
                    crest_probability_min, float(crest_probability.min())
                )
                crest_probability_max = max(
                    crest_probability_max, float(crest_probability.max())
                )
                crest_probability_sum += float(crest_probability.double().sum())
                crest_probability_count += int(crest_probability.numel())
                crest_probability_above_0_4 += int(
                    torch.count_nonzero(crest_probability >= 0.4)
                )
                crest_probability_above_0_5 += int(
                    torch.count_nonzero(crest_probability >= 0.5)
                )
        has_baseline = batch["has_baseline"].bool()
        for index in range(batch["target"].shape[0]):
            score = float(batch["pathology_score"][index])
            pathology_band = _pathology_band(score)
            scroll = str(batch["scroll_id"][index])
            scroll_threshold_histograms.setdefault(
                scroll, ThresholdHistogram(thresholds)
            ).update_probability(
                probability[index : index + 1],
                batch["target"][index : index + 1],
            )
            scrollfiesta_kind = {
                -1: "unavailable",
                0: "keep",
                1: "empty",
                2: "solid_slab",
            }[int(batch["scrollfiesta_pred_reject_kind"][index])]
            strata = (
                f"scroll/{batch['scroll_id'][index]}",
                f"source/{batch['supervision_source'][index]}",
                f"sampling/{batch['sampling_strategy'][index]}",
                f"pathology/{pathology_band}",
                f"scrollfiesta_pred/{scrollfiesta_kind}",
            )
            for stratum in strata:
                model_strata.setdefault(stratum, BinaryCounts()).update_prediction(
                    outputs[0][index : index + 1],
                    batch["target"][index : index + 1],
                )
            if bool(has_baseline[index]):
                baseline_counts.update_prediction(
                    batch["baseline"][index : index + 1].long(),
                    batch["target"][index : index + 1],
                )
                for stratum in strata:
                    baseline_strata.setdefault(
                        stratum, BinaryCounts()
                    ).update_prediction(
                        batch["baseline"][index : index + 1].long(),
                        batch["target"][index : index + 1],
                    )
                baseline_batches += 1
        batches += 1
    result = {f"loss_{name}": value / max(1, batches) for name, value in sums.items()}
    result.update(model_counts.metrics())
    if probability_count:
        result["probability_min"] = probability_min
        result["probability_max"] = probability_max
        result["probability_mean"] = probability_sum / probability_count
    if crest_probability_count:
        result.update(
            {
                "crest_voxels": float(crest_probability_count),
                "crest_probability_min": crest_probability_min,
                "crest_probability_max": crest_probability_max,
                "crest_probability_mean": (
                    crest_probability_sum / crest_probability_count
                ),
                "crest_recall_at_0_4": (
                    crest_probability_above_0_4 / crest_probability_count
                ),
                "crest_recall_at_0_5": (
                    crest_probability_above_0_5 / crest_probability_count
                ),
            }
        )
    if baseline_batches:
        baseline_metrics = baseline_counts.metrics()
        result.update(
            {f"baseline_{key}": value for key, value in baseline_metrics.items()}
        )
        result["dice_gain"] = result["dice"] - result["baseline_dice"]
    for stratum, counts in sorted(model_strata.items()):
        metrics = counts.metrics()
        for name, value in metrics.items():
            result[f"stratum/{stratum}/{name}"] = value
        if stratum in baseline_strata:
            baseline_metrics = baseline_strata[stratum].metrics()
            for name, value in baseline_metrics.items():
                result[f"stratum/{stratum}/baseline_{name}"] = value
            result[f"stratum/{stratum}/dice_gain"] = (
                metrics["dice"] - baseline_metrics["dice"]
            )
    scroll_dice = [
        counts.metrics()["dice"]
        for stratum, counts in model_strata.items()
        if stratum.startswith("scroll/")
    ]
    if scroll_dice:
        result["macro_scroll_dice"] = sum(scroll_dice) / len(scroll_dice)
    scroll_gains = [
        model_strata[stratum].metrics()["dice"]
        - baseline_strata[stratum].metrics()["dice"]
        for stratum in model_strata
        if stratum.startswith("scroll/") and stratum in baseline_strata
    ]
    if scroll_gains:
        result["macro_scroll_dice_gain"] = sum(scroll_gains) / len(scroll_gains)
        result["minimum_scroll_dice_gain"] = min(scroll_gains)
    result.update(
        _calibrated_validation_metrics(
            threshold_histogram,
            scroll_threshold_histograms,
        )
    )
    return result


def train_model(
    *,
    patch_manifest: str | Path,
    output_path: str | Path,
    options: TrainOptions,
    validation_patch_manifest: str | Path | None = None,
    resume: bool = False,
    snapshot_samples: Sequence[int] = (),
) -> Path:
    options.validate()
    configure_cpu_budget(options.max_cpu_threads, reserve_processes=options.num_workers)
    _seed_everything(options.seed)
    manifest = Path(patch_manifest).expanduser().resolve()
    validation_manifest = (
        Path(validation_patch_manifest).expanduser().resolve()
        if validation_patch_manifest is not None
        else None
    )
    output = Path(output_path).expanduser().resolve()
    dataset_size = _validate_scroll_split(
        manifest,
        validation_manifest,
        options.allow_spatial_validation,
        options.final_fit,
    )
    schedule = _resolve_training_schedule(
        dataset_size,
        epochs=options.epochs,
        samples_per_epoch=options.samples_per_epoch,
        max_train_samples=options.max_train_samples,
    )
    snapshots = _normalize_snapshot_samples(
        snapshot_samples,
        total_samples=schedule.total_samples,
    )
    output_preexisted = output.exists()
    if output_preexisted and not resume:
        raise ValueError(f"output already exists; use resume: {output}")
    if resume and not output_preexisted:
        raise ValueError(f"resume output does not exist: {output}")
    output.mkdir(parents=True, exist_ok=True)
    identity_value = {
        "patch_manifest": str(manifest),
        "patch_manifest_sha256": _sha256(manifest),
        "loss_contract": loss_contract(options.loss_options),
        "checkpoint_selection_contract": CHECKPOINT_SELECTION_CONTRACT,
        "checkpoint_durability_contract": CHECKPOINT_DURABILITY_CONTRACT,
        "epoch_partition_contract": EPOCH_PARTITION_CONTRACT,
        "sampling_contract": SAMPLING_CONTRACT,
        "learning_rate_contract": LEARNING_RATE_CONTRACT,
        "effective_partition_samples": schedule.total_samples,
        "resolved_schedule": asdict(schedule),
        "options": _options_identity(options),
    }
    if validation_manifest is not None:
        identity_value.update(
            validation_patch_manifest=str(validation_manifest),
            validation_patch_manifest_sha256=_sha256(validation_manifest),
        )
    released_m7_identity: dict[str, Any] | None = None
    if options.pretrained_m7_checkpoint:
        released_m7_identity = verify_released_m7_checkpoint(
            options.pretrained_m7_checkpoint
        )
        identity_value["initialization"] = released_m7_identity
    if options.optimizer in {"adam", "adamw"}:
        identity_value["optimizer_contract"] = (
            ADAM_OPTIMIZER_CONTRACT
            if options.optimizer == "adam"
            else ADAMW_OPTIMIZER_CONTRACT
        )
    if options.m7_trust_region_relative_l2 > 0:
        identity_value["m7_parameter_trust_region_contract"] = (
            M7_PARAMETER_TRUST_REGION_CONTRACT
        )
    if snapshots:
        identity_value.update(
            snapshot_checkpoint_contract=SNAPSHOT_CHECKPOINT_CONTRACT,
            snapshot_samples=list(snapshots),
        )
    if options.final_fit:
        identity_value["final_fit_checkpoint_contract"] = FINAL_FIT_CHECKPOINT_CONTRACT
    identity = json.loads(json.dumps(identity_value, default=str))
    identity_path = output / "run.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("resume configuration or patch manifest changed")
    elif resume:
        raise ValueError(f"resume run identity is missing: {identity_path}")
    else:
        _write_json(identity_path, identity)

    device = _device(options.device)
    assert_cuda_power_limit(device)
    amp_dtype, autocast_enabled, use_scaler = _amp_configuration(options, device)
    torch.set_float32_matmul_precision("high")
    config = NNUNetConfig(preset=options.preset)
    model = VoxelNNUNet(config)
    initialization: dict[str, Any] | None = None
    if options.pretrained_m7_checkpoint:
        initialization = initialize_from_m7(
            model,
            options.pretrained_m7_checkpoint,
            verified_identity=released_m7_identity,
        )
    m7_anchor_model: VoxelNNUNet | None = None
    if (
        options.loss_options.m7_anchor_weight > 0
        or options.loss_options.m7_preservation_weight > 0
        or options.m7_trust_region_relative_l2 > 0
    ):
        m7_anchor_model = copy.deepcopy(model)
        m7_anchor_model.requires_grad_(False)
        m7_anchor_model.eval()
    model.to(device)
    if m7_anchor_model is not None:
        m7_anchor_model.to(device)
    trust_region = (
        M7ParameterTrustRegion(
            model,
            m7_anchor_model,
            relative_l2_radius=options.m7_trust_region_relative_l2,
        )
        if options.m7_trust_region_relative_l2 > 0 and m7_anchor_model is not None
        else None
    )
    optimizer = _build_optimizer(model, options)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    loader_generator = torch.Generator().manual_seed(options.seed)
    train_dataset = VoxelPatchDataset(
        manifest,
        split=None if options.final_fit else "train",
        augment=options.train_augmentation,
        pinned_medial_bridge_state=(
            options.pinned_medial_bridge_state
            if options.loss_options.pinned_axial_weight > 0
            else None
        ),
        dynamic_medial_connectivity_state=(
            options.dynamic_medial_connectivity_state
            if options.loss_options.dynamic_medial_connectivity_weight > 0
            else None
        ),
    )
    if (
        options.loss_options.medial_recall_weight > 0
        and not train_dataset.has_complete_teacher_crest
    ):
        raise ValueError(
            "medial recall training requires every training row to use the "
            "provenance-bound medial atlas patch format"
        )
    if len(train_dataset) != dataset_size:
        raise RuntimeError("training dataset size changed after split validation")
    sampler_type = (
        StratifiedEpochPartitionSampler
        if options.stratified_sampling
        else EpochPartitionSampler
    )
    sampler_rows: Any = (
        train_dataset.rows if options.stratified_sampling else len(train_dataset)
    )
    train_sampler = sampler_type(
        sampler_rows,
        schedule.evaluation_interval_samples,
        options.seed,
        total_samples=schedule.total_samples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=options.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=options.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        generator=loader_generator,
    )
    val_loader: DataLoader[dict[str, Any]] | None = None
    if not options.final_fit:
        effective_validation_manifest = validation_manifest or manifest
        val_loader = DataLoader(
            VoxelPatchDataset(
                effective_validation_manifest, split="val", augment=False
            ),
            batch_size=1,
            shuffle=False,
            num_workers=options.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
        )

    start_epoch = 0
    best_score = float("-inf")
    best_trained_score = float("-inf")
    intervals_since_best = 0
    cumulative_samples = 0
    last_checkpoint = output / "checkpoint_last.pt"
    best_checkpoint = output / "checkpoint_best.pt"
    best_trained_checkpoint = output / "checkpoint_best_trained.pt"
    initial_checkpoint = output / "checkpoint_initial.pt"
    history_path = output / "history.jsonl"
    snapshot_index_path = output / "checkpoint_milestones.json"
    resume_payload: dict[str, Any] | None = None
    history_rows: list[dict[str, Any]] = []
    checkpoint_epoch: int | None = None
    if resume and last_checkpoint.exists():
        resume_payload = torch.load(
            last_checkpoint, map_location="cpu", weights_only=False
        )
        if resume_payload.get("identity") != identity:
            raise ValueError("resume checkpoint identity does not match run identity")
        if resume_payload.get("model_config") != config.as_dict():
            raise ValueError("resume checkpoint model configuration changed")
        checkpoint_epoch = int(resume_payload["epoch"])
        if not 0 <= checkpoint_epoch < schedule.evaluation_intervals:
            raise ValueError(f"invalid resume checkpoint epoch: {checkpoint_epoch}")
        metrics = resume_payload.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("resume checkpoint has no metrics row")
        history_rows = _reconcile_history(
            history_path,
            checkpoint_epoch=checkpoint_epoch,
            checkpoint_metrics=metrics,
        )
        model.load_state_dict(resume_payload["model"], strict=True)
        if trust_region is not None and (
            trust_region.measure_relative_l2()
            > options.m7_trust_region_relative_l2 * (1.0 + 1.0e-5)
        ):
            raise ValueError("resume checkpoint lies outside its M7 trust region")
        optimizer.load_state_dict(resume_payload["optimizer"])
        if "scaler" in resume_payload:
            scaler.load_state_dict(resume_payload["scaler"])
        start_epoch = checkpoint_epoch + 1
        best_score = float(resume_payload.get("best_score", float("-inf")))
        if not math.isfinite(best_score):
            raise ValueError(f"invalid checkpoint best score: {best_score}")
        best_trained_score = float(
            resume_payload.get("best_trained_score", float("-inf"))
        )
        intervals_since_best = int(resume_payload.get("intervals_since_best", 0))
        cumulative_samples = int(
            resume_payload.get(
                "cumulative_samples",
                sum(int(row["train"]["samples"]) for row in history_rows),
            )
        )
        _reconcile_best_checkpoint_artifacts(
            resume_payload,
            best_checkpoint=best_checkpoint,
            best_trained_checkpoint=best_trained_checkpoint,
            final_fit=options.final_fit,
        )
    elif resume:
        if _read_history_rows(history_path):
            raise ValueError("pre-checkpoint resume contains committed epoch artifacts")

    snapshot_records = _reconcile_snapshot_records(
        output,
        snapshot_index_path,
        snapshots=snapshots,
        cumulative_samples=cumulative_samples,
    )
    next_snapshot_index = len(snapshot_records)

    initial_validation: dict[str, float] = {}
    initial_metrics_path = output / "initial_validation.json"
    if val_loader is not None:
        if initial_metrics_path.exists():
            initial_validation = json.loads(
                initial_metrics_path.read_text(encoding="utf-8")
            )
        elif resume_payload is not None:
            raise ValueError("resume checkpoint exists without initial validation")
        else:
            initial_validation = validate_model(
                model,
                val_loader,
                device,
                amp_dtype,
                autocast_enabled,
                options.validation_thresholds,
                loss_options=options.loss_options,
                m7_anchor_model=m7_anchor_model,
            )
            _write_json(initial_metrics_path, initial_validation)
        if resume_payload is None:
            best_score = _validation_score(initial_validation)
            initial_payload = {
                "epoch": -1,
                "best_score": best_score,
                "best_trained_score": best_trained_score,
                "intervals_since_best": 0,
                "cumulative_samples": 0,
                "model_config": config.as_dict(),
                "model": model.state_dict(),
                "initialization": initialization,
                "identity": identity,
                "metrics": {
                    "epoch": -1,
                    "learning_rate": 0.0,
                    "train": {},
                    "val": initial_validation,
                },
            }
            _atomic_torch_save(initial_checkpoint, initial_payload)
            _atomic_torch_save(best_checkpoint, initial_payload)
        elif not initial_checkpoint.is_file():
            raise ValueError("resume run is missing its initial m7 checkpoint")

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("TensorBoard is required for voxel training") from error
    purge_step = None if not resume else (checkpoint_epoch or 0)
    writer = SummaryWriter(log_dir=str(output / "tensorboard"), purge_step=purge_step)
    try:
        if resume_payload is not None:
            _log_tensorboard_row(writer, history_rows[-1])
            writer.flush()
            _restore_rng_state(
                resume_payload.get("rng_state"),
                loader_generator=loader_generator,
                device=device,
            )
        elif initial_validation:
            _log_tensorboard_initial_validation(
                writer,
                initial_validation,
                total_samples=schedule.total_samples,
                evaluation_interval_samples=schedule.evaluation_interval_samples,
                learning_rate=options.learning_rate,
            )
            writer.flush()
        for epoch in range(start_epoch, schedule.evaluation_intervals):
            train_sampler.set_epoch(epoch)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            interval_start_samples = cumulative_samples
            learning_rate_start = _learning_rate_for_samples(
                options,
                samples_seen=interval_start_samples,
                total_samples=schedule.total_samples,
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            sums: dict[str, float] = {}
            trust_region_updates = 0
            trust_region_active = 0
            trust_region_preprojection_sum = 0.0
            trust_region_preprojection_max = 0.0
            trust_region_scale_min = 1.0
            trust_region_relative_l2_last = 0.0
            batches = 0
            interval_samples_seen = 0
            epoch_started = time.perf_counter()
            for batch_index, raw_batch in enumerate(train_loader):
                batch = _move_batch(raw_batch, device)
                interval_samples_seen += int(batch["image"].shape[0])
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=autocast_enabled,
                ):
                    m7_anchor_outputs = None
                    if m7_anchor_model is not None:
                        with torch.no_grad():
                            m7_anchor_outputs = m7_anchor_model(batch["image"])
                    outputs = model(batch["image"])
                    loss, components = deep_supervision_loss(
                        outputs,
                        batch["target"],
                        batch.get("teacher_q"),
                        batch.get("target_valid"),
                        teacher_crest=batch.get("teacher_crest"),
                        teacher_crest_valid=batch.get("teacher_crest_valid"),
                        teacher_crest_available=batch.get("has_teacher_crest"),
                        m7_anchor_outputs=m7_anchor_outputs,
                        pinned_medial_bridge=batch.get("pinned_medial_bridge"),
                        dynamic_connectivity_event=batch.get(
                            "dynamic_connectivity_event"
                        ),
                        dynamic_connectivity_pins=batch.get(
                            "dynamic_connectivity_pins"
                        ),
                        dynamic_connectivity_free=batch.get(
                            "dynamic_connectivity_free"
                        ),
                        options=options.loss_options,
                    )
                    scaled_loss = loss / options.accumulate
                if use_scaler:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                update = (batch_index + 1) % options.accumulate == 0 or (
                    batch_index + 1 == len(train_loader)
                )
                if update:
                    learning_rate = _learning_rate_for_samples(
                        options,
                        samples_seen=min(
                            schedule.total_samples,
                            interval_start_samples + interval_samples_seen,
                        ),
                        total_samples=schedule.total_samples,
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                    if use_scaler:
                        scaler.unscale_(optimizer)
                    if options.gradient_clip_norm:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), options.gradient_clip_norm
                        )
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if trust_region is not None:
                        projection = trust_region.project()
                        trust_region_updates += 1
                        trust_region_active += int(projection.active)
                        trust_region_preprojection_sum += projection.relative_l2_before
                        trust_region_preprojection_max = max(
                            trust_region_preprojection_max,
                            projection.relative_l2_before,
                        )
                        trust_region_scale_min = min(
                            trust_region_scale_min,
                            projection.projection_scale,
                        )
                        trust_region_relative_l2_last = projection.relative_l2_after
                    optimizer.zero_grad(set_to_none=True)
                    actual_samples = interval_start_samples + interval_samples_seen
                    while (
                        next_snapshot_index < len(snapshots)
                        and snapshots[next_snapshot_index] <= actual_samples
                    ):
                        requested_samples = snapshots[next_snapshot_index]
                        snapshot_path = _snapshot_checkpoint_path(
                            output,
                            requested_samples,
                        )
                        snapshot_payload = {
                            "checkpoint_kind": "sample-milestone",
                            "snapshot_checkpoint_contract": (
                                SNAPSHOT_CHECKPOINT_CONTRACT
                            ),
                            "requested_samples": requested_samples,
                            "cumulative_samples": actual_samples,
                            "epoch": epoch,
                            "best_score": best_score,
                            "best_trained_score": best_trained_score,
                            "model_config": config.as_dict(),
                            "model": model.state_dict(),
                            "initialization": initialization,
                            "identity": identity,
                            "metrics": {
                                "epoch": epoch,
                                "train": {
                                    "cumulative_samples": float(actual_samples),
                                },
                                "val": {},
                            },
                        }
                        _atomic_torch_save(snapshot_path, snapshot_payload)
                        snapshot_records.append(
                            {
                                "requested_samples": requested_samples,
                                "actual_samples": actual_samples,
                                "checkpoint": snapshot_path.name,
                                "bytes": snapshot_path.stat().st_size,
                            }
                        )
                        _write_json(
                            snapshot_index_path,
                            {
                                "schema": SNAPSHOT_CHECKPOINT_CONTRACT,
                                "records": snapshot_records,
                            },
                        )
                        print(
                            "saved sample milestone "
                            f"{requested_samples:,} at {actual_samples:,} samples: "
                            f"{snapshot_path.name}",
                            flush=True,
                        )
                        next_snapshot_index += 1
                for name, value in components.items():
                    sums[name] = sums.get(name, 0.0) + float(value)
                batches += 1
                if batches % 250 == 0 or batches == len(train_loader):
                    elapsed = max(time.perf_counter() - epoch_started, 1.0e-6)
                    remaining_batches = len(train_loader) - batches
                    print(
                        f"interval {epoch + 1}/{schedule.evaluation_intervals} "
                        f"batches {batches:,}/{len(train_loader):,} "
                        f"({batches / elapsed:.2f}/s, "
                        f"ETA {remaining_batches * elapsed / batches / 60:.1f}m)",
                        flush=True,
                    )
            train_metrics = {
                f"loss_{name}": value / max(1, batches) for name, value in sums.items()
            }
            interval_samples = len(train_sampler)
            if interval_samples_seen != interval_samples:
                raise RuntimeError(
                    f"sampler yielded {interval_samples_seen}, expected {interval_samples}"
                )
            cumulative_samples += interval_samples
            train_metrics["samples"] = float(interval_samples)
            train_metrics["cumulative_samples"] = float(cumulative_samples)
            if trust_region is not None:
                if trust_region_updates <= 0:
                    raise RuntimeError("M7 trust region observed no optimizer updates")
                train_metrics.update(
                    {
                        "m7_trust_region_relative_l2": (trust_region_relative_l2_last),
                        "m7_trust_region_preprojection_relative_l2_mean": (
                            trust_region_preprojection_sum / trust_region_updates
                        ),
                        "m7_trust_region_preprojection_relative_l2_max": (
                            trust_region_preprojection_max
                        ),
                        "m7_trust_region_projection_active_fraction": (
                            trust_region_active / trust_region_updates
                        ),
                        "m7_trust_region_projection_scale_min": (
                            trust_region_scale_min
                        ),
                    }
                )
            if device.type == "cuda":
                train_metrics["cuda_peak_allocated_gib"] = (
                    torch.cuda.max_memory_allocated(device) / 2**30
                )
                train_metrics["cuda_peak_reserved_gib"] = (
                    torch.cuda.max_memory_reserved(device) / 2**30
                )
            validation = (
                validate_model(
                    model,
                    val_loader,
                    device,
                    amp_dtype,
                    autocast_enabled,
                    options.validation_thresholds,
                    loss_options=options.loss_options,
                    m7_anchor_model=m7_anchor_model,
                )
                if val_loader is not None
                else {}
            )
            if validation and "dice" in initial_validation:
                validation["m7_initial_dice"] = initial_validation["dice"]
                validation["dice_gain_vs_m7_initial"] = (
                    validation["dice"] - initial_validation["dice"]
                )
                validation["calibrated_dice_gain_vs_m7_initial"] = (
                    validation["calibrated_dice"]
                    - initial_validation["calibrated_dice"]
                )
                validation["calibrated_macro_gain_vs_m7_initial"] = (
                    validation["calibrated_macro_scroll_dice"]
                    - initial_validation["calibrated_macro_scroll_dice"]
                )
            row = {
                "epoch": epoch,
                "learning_rate_start": learning_rate_start,
                "learning_rate": learning_rate,
                "train": train_metrics,
                "val": validation,
            }
            score = _history_row_score(row)
            trained_is_best = score > best_trained_score
            best_trained_score = max(best_trained_score, score)
            if options.final_fit:
                is_best, best_score = _select_best_checkpoint(
                    score, best_score, final_fit=True
                )
                checkpoint_eligible = True
                minimum_scroll_gain = 0.0
            else:
                checkpoint_eligible, minimum_scroll_gain = _checkpoint_guard(
                    validation,
                    initial_validation,
                    minimum_scroll_gain=options.minimum_scroll_gain,
                    checkpoint_min_delta=options.checkpoint_min_delta,
                )
                validation["checkpoint_minimum_scroll_gain_vs_m7_initial"] = (
                    minimum_scroll_gain
                )
                validation["checkpoint_eligible"] = float(checkpoint_eligible)
                if checkpoint_eligible:
                    is_best, best_score = _select_best_checkpoint(
                        score, best_score, final_fit=False
                    )
                else:
                    is_best = False
            intervals_since_best = 0 if is_best else intervals_since_best + 1
            checkpoint = {
                "epoch": epoch,
                "best_score": best_score,
                "best_trained_score": best_trained_score,
                "intervals_since_best": intervals_since_best,
                "cumulative_samples": cumulative_samples,
                "model_config": config.as_dict(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "initialization": initialization,
                "identity": identity,
                "metrics": row,
                "rng_state": _capture_rng_state(loader_generator, device),
                "checkpoint_roles": {
                    "best": is_best,
                    "best_trained": trained_is_best,
                },
            }
            # checkpoint_last is the epoch commit point. Every other artifact
            # can be deterministically reconciled from this payload.
            _atomic_torch_save(last_checkpoint, checkpoint)
            if is_best:
                best_payload = _model_only_checkpoint_payload(checkpoint)
                _atomic_torch_save(best_checkpoint, best_payload)
            if trained_is_best and not options.final_fit:
                trained_payload = _model_only_checkpoint_payload(checkpoint)
                _atomic_torch_save(best_trained_checkpoint, trained_payload)
            history_rows.append(row)
            _write_history_rows(history_path, history_rows)
            _log_tensorboard_row(writer, row)
            writer.flush()
            if (
                options.early_stopping_patience is not None
                and intervals_since_best >= options.early_stopping_patience
            ):
                break
        if next_snapshot_index != len(snapshots):
            missing = snapshots[next_snapshot_index:]
            raise RuntimeError(
                f"training ended before checkpoint milestones were saved: {missing}"
            )
    finally:
        writer.close()
    return best_checkpoint
