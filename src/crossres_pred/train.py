from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import PatchDataset, load_patch_rows, validate_patch_splits
from .losses import SurfaceObjective, compute_surface_losses
from .metrics import StreamingBinaryMetrics, interior_slices
from .model import (
    SurfaceModelConfig,
    SurfaceNet,
    initialize_from_m7_checkpoint,
    initialize_from_surface_checkpoint,
    load_surface_checkpoint,
)
from .provenance import (
    environment_identity,
    require_fresh_directory,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from .telemetry import create_training_writer, load_history_rows, log_history_row

PROFILES = ("teacher", "student")
INIT_MODES = ("m7-nnunet", "surface-checkpoint", "none")


@dataclass(frozen=True)
class TrainOptions:
    """Shared trainer options for the fine teachers and the coarse student.

    The two profiles differ only in which loss partitions carry weight, the
    rotation augmentation group, and the selection target: the teacher is
    selected on ground-truth AP, the student on distillation-target AP --
    never on a weighted validation loss (the recorded v3-v4.6 lesson).
    """

    profile: str = "student"
    epochs: int = 40
    batch_size: int = 1
    accumulate: int = 4
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    warmup_steps: int = 500
    num_workers: int = 0
    seed: int = 1203
    device: str = "cuda"
    amp: bool = True
    amp_dtype: str = "auto"
    in_channels: int = 1
    init_mode: str = "m7-nnunet"
    pretrained_checkpoint: str | None = None
    dice_weight: float = 0.5
    distill_weight: float | None = None
    rehearsal_weight: float | None = None
    rot90_mode: str | None = None
    eval_interior_margin: int = 32
    final_fit: bool = False

    def validate(self) -> None:
        if self.profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}")
        if self.init_mode not in INIT_MODES:
            raise ValueError(f"init_mode must be one of {INIT_MODES}")
        if self.epochs <= 0 or self.batch_size <= 0 or self.accumulate <= 0:
            raise ValueError("epochs, batch_size, and accumulate must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.amp_dtype not in {"auto", "float16", "bfloat16"}:
            raise ValueError("amp_dtype must be auto, float16, or bfloat16")
        if self.eval_interior_margin < 0:
            raise ValueError("eval_interior_margin must be non-negative")
        if self.init_mode != "none" and self.pretrained_checkpoint is None:
            raise ValueError(f"init_mode {self.init_mode!r} requires a checkpoint")
        self.model_config()
        self.objective().validate()
        if self.resolved_rot90() not in {"none", "z-only", "all"}:
            raise ValueError(f"invalid rot90_mode {self.rot90_mode!r}")

    def model_config(self) -> SurfaceModelConfig:
        return SurfaceModelConfig(in_channels=self.in_channels)

    def objective(self) -> SurfaceObjective:
        if self.profile == "teacher":
            distill = 0.0 if self.distill_weight is None else self.distill_weight
            rehearsal = 0.0 if self.rehearsal_weight is None else self.rehearsal_weight
        else:
            distill = 1.0 if self.distill_weight is None else self.distill_weight
            rehearsal = (
                0.25 if self.rehearsal_weight is None else self.rehearsal_weight
            )
        return SurfaceObjective(
            dice_weight=self.dice_weight,
            distill_weight=distill,
            rehearsal_weight=rehearsal,
        )

    def resolved_rot90(self) -> str:
        if self.rot90_mode is not None:
            return self.rot90_mode
        # Fine-pitch lattices are isotropic; the coarse student keeps scan
        # z-anisotropy honest by rotating only about z.
        return "all" if self.profile == "teacher" else "z-only"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def _amp_dtype(
    requested: str, device: torch.device, enabled: bool
) -> tuple[torch.dtype, bool]:
    if not enabled or device.type != "cuda":
        return torch.float32, False
    if requested == "bfloat16" or (
        requested == "auto" and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16, True
    return torch.float16, True


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (
            value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


def _checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _mean_components(sums: dict[str, float], batches: int) -> dict[str, float]:
    return {name: value / max(1, batches) for name, value in sums.items()}


def _restore_random_state(
    checkpoint: dict[str, Any], generator: torch.Generator
) -> None:
    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])
    if "numpy_random_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_random_state"])
    if "torch_random_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_random_state"].cpu())
    if torch.cuda.is_available() and "cuda_random_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
    if "loader_generator_state" in checkpoint:
        generator.set_state(checkpoint["loader_generator_state"].cpu())


def _resume_output_directory(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if not output.is_dir():
        raise ValueError(f"resume output directory does not exist: {output}")
    return output


def _comparable_options(value: Any) -> Any:
    """Resume equality ignores only ``epochs``: extending a finished run is a
    deliberate feature; every other option stays strictly pinned."""

    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    normalized.pop("epochs", None)
    return normalized


def _warmup_cosine_lambda(
    warmup_steps: int, total_steps: int, floor: float = 0.01
):
    def scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        span = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / span))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return scale


@torch.no_grad()
def _validate(
    model: SurfaceNet,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    autocast_enabled: bool,
    objective: SurfaceObjective,
    profile: str,
    interior_margin: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    model.eval()
    sums: dict[str, float] = {}
    batches = 0
    selection = StreamingBinaryMetrics()
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=autocast_enabled,
        ):
            logits = model(batch["input"])
            _, components = compute_surface_losses(logits, batch, objective)
        for name, value in components.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach())
        interior = (slice(None), slice(None)) + interior_slices(
            tuple(logits.shape[-3:]), interior_margin
        )
        probability = torch.sigmoid(logits.float())[interior]
        label = batch["label"][interior]
        if profile == "teacher":
            mask = label < 1.5
            target = label > 0.5
        else:
            mask = batch["distill_valid"][interior] > 0.5
            target = batch["distill"][interior] >= 0.5
        selection.update(probability, target, mask)
        batches += 1
    metrics = selection.result()
    metrics["selection"] = (
        "ground-truth-ap" if profile == "teacher" else "distill-target-ap"
    )
    return _mean_components(sums, batches), metrics


def train_model(
    *,
    patch_manifest: str | Path,
    output_path: str | Path,
    options: TrainOptions,
    resume_checkpoint: str | Path | None = None,
) -> Path:
    options.validate()
    model_config = options.model_config()
    objective = options.objective()
    rot90_mode = options.resolved_rot90()
    rows = load_patch_rows(patch_manifest)
    validate_patch_splits(rows)
    patch_shapes = {row.shape_zyx for row in rows}
    if len(patch_shapes) != 1:
        raise ValueError(f"patch manifest mixes shapes: {sorted(patch_shapes)}")
    if any(size % 32 or size < 64 for size in next(iter(patch_shapes))):
        raise ValueError(
            "patch dimensions must be multiples of 32 and at least 64"
        )
    policy_profiles = {row.policy_profile for row in rows}
    if len(policy_profiles) != 1:
        raise ValueError(
            f"patch manifest mixes policy profiles: {sorted(policy_profiles)}"
        )
    policy_profile = next(iter(policy_profiles))
    kinds = {row.kind for row in rows}
    if kinds != {options.profile}:
        raise ValueError(
            f"{options.profile} training requires kind={options.profile!r} patches, "
            f"manifest has {sorted(kinds)}"
        )

    train_scrolls = {row.scroll_id for row in rows if row.split == "train"}
    val_scrolls = {row.scroll_id for row in rows if row.split == "val"}
    if not train_scrolls:
        raise ValueError("patch manifest has no training scrolls")
    if not val_scrolls and not options.final_fit:
        raise ValueError("patch manifest has no validation scrolls")
    if val_scrolls and options.final_fit:
        raise ValueError("final-fit training requires no validation rows")
    if train_scrolls & val_scrolls:
        raise ValueError("training and validation scrolls overlap")

    resume_path = (
        Path(resume_checkpoint).expanduser().resolve()
        if resume_checkpoint is not None
        else None
    )
    output = (
        _resume_output_directory(output_path)
        if resume_path is not None
        else require_fresh_directory(output_path)
    )
    device = _device(options.device)
    amp_dtype, autocast_enabled = _amp_dtype(options.amp_dtype, device, options.amp)
    _seed_everything(options.seed)
    train_dataset = PatchDataset(
        patch_manifest,
        split="train",
        kind=options.profile,
        augment=True,
        rot90_mode=rot90_mode,
        in_channels=options.in_channels,
    )
    generator = torch.Generator().manual_seed(options.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=options.batch_size,
        shuffle=True,
        num_workers=options.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=options.num_workers > 0,
        generator=generator,
    )
    val_loader = None
    if val_scrolls:
        val_dataset = PatchDataset(
            patch_manifest,
            split="val",
            kind=options.profile,
            augment=False,
            rot90_mode="none",
            in_channels=options.in_channels,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=max(0, options.num_workers // 2),
            pin_memory=device.type == "cuda",
        )
    model = SurfaceNet(model_config)
    initialization = None
    if resume_path is None and options.init_mode != "none":
        if options.init_mode == "m7-nnunet":
            initialization = initialize_from_m7_checkpoint(
                model, options.pretrained_checkpoint
            )
        else:
            initialization = initialize_from_surface_checkpoint(
                model, options.pretrained_checkpoint
            )
        initialization["checkpoint_sha256"] = sha256_file(
            options.pretrained_checkpoint
        )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / options.accumulate))
    total_steps = options.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        _warmup_cosine_lambda(
            min(options.warmup_steps, max(0, total_steps - 1)), total_steps
        ),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=autocast_enabled and amp_dtype == torch.float16
    )
    best_loss = float("inf")
    best_selection = float("-inf")
    selection_available = False
    best_path = output / "best.pt"
    best_loss_path = output / "best_loss.pt"
    history_path = output / "history.jsonl"

    provenance_path = output / "provenance.json"
    patch_manifest_path = Path(patch_manifest).expanduser().resolve()
    manifest_sha256 = sha256_file(patch_manifest_path)
    current_options = asdict(options)
    current_objective = objective.as_dict()
    expected_train_scrolls = sorted(train_scrolls)
    expected_val_scrolls = sorted(val_scrolls)
    start_epoch = 1
    history_mode = "x"

    if resume_path is None:
        provenance = {
            "schema_version": 2,
            "kind": "crossres-surface-training",
            "profile": options.profile,
            "created_at": utc_now(),
            "patch_manifest": {
                "path": str(patch_manifest_path),
                "sha256": manifest_sha256,
            },
            "policy_profile": policy_profile,
            "train_scrolls": expected_train_scrolls,
            "val_scrolls": expected_val_scrolls,
            "options": current_options,
            "model": model_config.as_dict(),
            "objective": current_objective,
            "rot90_mode": rot90_mode,
            "resolved_amp_dtype": str(amp_dtype).removeprefix("torch."),
            "initialization": initialization,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "environment": environment_identity(),
        }
        write_json_atomic(provenance_path, provenance)
    else:
        if resume_path.parent != output:
            raise ValueError(
                "resume checkpoint must be inside the selected output directory"
            )
        if not resume_path.is_file():
            raise ValueError(f"resume checkpoint does not exist: {resume_path}")
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise ValueError(
                f"resume provenance is missing or invalid: {provenance_path}"
            ) from error
        if not isinstance(provenance, dict):
            raise ValueError(f"resume provenance is not an object: {provenance_path}")

        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if int(checkpoint.get("schema_version", 0)) != 2:
            raise ValueError(f"{resume_path}: unsupported checkpoint schema")
        comparable_current_options = _comparable_options(current_options)
        compatibility = (
            (
                "training options",
                _comparable_options(checkpoint.get("train_options")),
                comparable_current_options,
            ),
            (
                "model configuration",
                checkpoint.get("model_config"),
                model_config.as_dict(),
            ),
            ("objective", checkpoint.get("objective"), current_objective),
            ("policy profile", checkpoint.get("policy_profile"), policy_profile),
            (
                "training scrolls",
                checkpoint.get("train_scrolls"),
                expected_train_scrolls,
            ),
            ("validation scrolls", checkpoint.get("val_scrolls"), expected_val_scrolls),
            (
                "provenance options",
                _comparable_options(provenance.get("options")),
                comparable_current_options,
            ),
            (
                "provenance model configuration",
                provenance.get("model"),
                model_config.as_dict(),
            ),
            ("provenance objective", provenance.get("objective"), current_objective),
            ("provenance policy", provenance.get("policy_profile"), policy_profile),
            (
                "provenance training scrolls",
                provenance.get("train_scrolls"),
                expected_train_scrolls,
            ),
            (
                "provenance validation scrolls",
                provenance.get("val_scrolls"),
                expected_val_scrolls,
            ),
            (
                "patch manifest hash",
                provenance.get("patch_manifest", {}).get("sha256"),
                manifest_sha256,
            ),
        )
        for label, actual, expected in compatibility:
            if actual != expected:
                raise ValueError(
                    f"resume {label} mismatch: expected {expected!r}, found {actual!r}"
                )

        history_rows = load_history_rows(history_path)
        checkpoint_epoch = int(checkpoint.get("epoch", 0))
        if not history_rows or int(history_rows[-1]["epoch"]) != checkpoint_epoch:
            raise ValueError(
                f"resume history must end at the checkpoint epoch {checkpoint_epoch}"
            )
        if checkpoint_epoch >= options.epochs:
            raise ValueError(
                f"checkpoint epoch {checkpoint_epoch} already reached the configured "
                f"{options.epochs} epochs"
            )
        if val_scrolls:
            recorded_best = history_rows[-1].get("best_val_loss")
            if (
                recorded_best is None
                or not best_path.is_file()
                or not best_loss_path.is_file()
            ):
                raise ValueError(
                    "resume validation history or best checkpoint is missing"
                )
            best_loss = float(recorded_best)
            recorded_selection = history_rows[-1].get("best_selection")
            if recorded_selection is not None:
                best_selection = float(recorded_selection)
                selection_available = True

        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        _restore_random_state(checkpoint, generator)
        start_epoch = checkpoint_epoch + 1
        history_mode = "a"

        resume_events = provenance.setdefault("resume_events", [])
        if not isinstance(resume_events, list):
            raise ValueError("resume provenance events must be a list")
        resume_events.append(
            {
                "resumed_at": utc_now(),
                "from_epoch": checkpoint_epoch,
                "checkpoint": str(resume_path),
                "checkpoint_sha256": sha256_file(resume_path),
                "environment": environment_identity(),
            }
        )
        write_json_atomic(provenance_path, provenance)

    with (
        create_training_writer(
            output,
            purge_step=(start_epoch if resume_path is not None else None),
        ) as tensorboard,
        history_path.open(history_mode, encoding="utf-8", newline="\n") as history,
    ):
        for epoch in range(start_epoch, options.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            sums: dict[str, float] = {}
            batches = 0
            for batch_index, raw_batch in enumerate(train_loader):
                batch = _move_batch(raw_batch, device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=autocast_enabled,
                ):
                    logits = model(batch["input"])
                    loss, components = compute_surface_losses(
                        logits, batch, objective
                    )
                    group_start = (
                        batch_index // options.accumulate
                    ) * options.accumulate
                    group_size = min(
                        options.accumulate, len(train_loader) - group_start
                    )
                    scaled_loss = loss / group_size
                scaler.scale(scaled_loss).backward()
                should_step = (
                    batch_index + 1
                ) % options.accumulate == 0 or batch_index + 1 == len(train_loader)
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                for name, value in components.items():
                    sums[name] = sums.get(name, 0.0) + float(value.detach())
                batches += 1
            train_metrics = _mean_components(sums, batches)
            if val_loader is not None:
                val_metrics, val_selection = _validate(
                    model,
                    val_loader,
                    device,
                    amp_dtype,
                    autocast_enabled,
                    objective,
                    options.profile,
                    options.eval_interior_margin,
                )
            else:
                val_metrics = None
                val_selection = None
            val_loss = val_metrics["total"] if val_metrics is not None else None
            is_best_loss = val_loss is not None and val_loss < best_loss
            if is_best_loss:
                best_loss = val_loss
            selection_score = (
                val_selection.get("average_precision")
                if val_selection is not None
                else None
            )
            is_best_selection = selection_score is not None and (
                not selection_available or float(selection_score) > best_selection
            )
            if is_best_selection:
                best_selection = float(selection_score)
                selection_available = True
            select_best = is_best_selection or (
                not selection_available and is_best_loss
            )
            row = {
                "epoch": epoch,
                "learning_rate": scheduler.get_last_lr()[0],
                "train": train_metrics,
                "val": val_metrics,
                "val_selection": val_selection,
                "best_val_loss": (best_loss if val_metrics is not None else None),
                "best_selection": (
                    best_selection if selection_available else None
                ),
                "checkpoint_selection": (
                    (val_selection or {}).get("selection")
                    if selection_available
                    else "validation-loss-fallback"
                ),
            }
            checkpoint = {
                "schema_version": 2,
                "kind": "crossres-surface-training",
                "profile": options.profile,
                "epoch": epoch,
                "model_config": model_config.as_dict(),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "train_options": current_options,
                "objective": current_objective,
                "rot90_mode": rot90_mode,
                "policy_profile": policy_profile,
                "train_scrolls": expected_train_scrolls,
                "val_scrolls": expected_val_scrolls,
                "val_loss": val_loss,
                "val_selection": val_selection,
                "best_val_loss": (best_loss if val_metrics is not None else None),
                "best_selection": (
                    best_selection if selection_available else None
                ),
                "checkpoint_selection": row["checkpoint_selection"],
                "deploy_threshold": None,
                "history_row": row,
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "loader_generator_state": generator.get_state(),
            }
            if device.type == "cuda":
                checkpoint["cuda_random_state"] = torch.cuda.get_rng_state_all()
            _checkpoint_atomic(output / "last.pt", checkpoint)
            if is_best_loss:
                _checkpoint_atomic(best_loss_path, checkpoint)
            if select_best:
                _checkpoint_atomic(best_path, checkpoint)
            history.write(json.dumps(row, sort_keys=True) + "\n")
            history.flush()
            log_history_row(tensorboard, row)
            tensorboard.flush()
            val_text = f"{val_loss:.6f}" if val_loss is not None else "n/a"
            selection_text = (
                f"{best_selection:.6f}" if selection_available else "n/a"
            )
            print(
                f"epoch {epoch}/{options.epochs} "
                f"train={train_metrics['total']:.6f} val={val_text} "
                f"best_loss={best_loss if val_loss is not None else 'n/a'} "
                f"best_ap={selection_text}",
                file=sys.stderr,
                flush=True,
            )
    return best_path if val_scrolls else output / "last.pt"


def load_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[SurfaceNet, dict[str, Any]]:
    return load_surface_checkpoint(path, device)
