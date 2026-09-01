#!/usr/bin/env python3
"""Measure dynamic medial-connectivity loss on the exact training schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from crossres_pred.voxel.inference import load_voxel_checkpoint
from crossres_pred.voxel.loss import (
    DYNAMIC_MEDIAL_CONNECTIVITY_PROBABILITY_FLOOR,
    DYNAMIC_MEDIAL_CONNECTIVITY_STEPS,
    dynamic_medial_connectivity_loss,
)
from crossres_pred.voxel.patches import VoxelPatchDataset
from crossres_pred.voxel.train import StratifiedEpochPartitionSampler

SCHEMA = "crossres-pherc0139-dynamic-medial-connectivity-loss-scale-audit-v1"
DEFAULT_WEIGHTS = (0.015625, 0.03125, 0.0625, 0.125, 0.25)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            key: None for key in ("minimum", "p10", "median", "p90", "maximum", "mean")
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _sample_result(
    logits: torch.Tensor,
    batch: dict[str, Any],
    index: int,
    *,
    probability_floor: float,
    propagation_steps: int,
) -> dict[str, float | int]:
    result = dynamic_medial_connectivity_loss(
        logits[index : index + 1],
        batch["dynamic_connectivity_event"][index : index + 1],
        batch["dynamic_connectivity_pins"][index : index + 1],
        batch["dynamic_connectivity_free"][index : index + 1],
        probability_floor=probability_floor,
        propagation_steps=propagation_steps,
    )
    events = int(result.events.item())
    if events <= 0:
        raise RuntimeError("prefiltered connectivity sample lost all event IDs")
    return {
        "loss": float(result.loss.item()),
        "events": events,
        "targets": int(result.targets.item()),
        "mean_bottleneck_probability": float(result.mean_bottleneck_probability.item()),
    }


def _autograd_probe(
    *,
    model: torch.nn.Module,
    dataset: VoxelPatchDataset,
    dataset_index: int,
    device: torch.device,
    probability_floor: float,
    propagation_steps: int,
) -> dict[str, Any]:
    raw_batch = next(
        iter(
            DataLoader(
                Subset(dataset, [dataset_index]),
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
        )
    )
    batch = _move(raw_batch, device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(batch["image"])[0]
    logits = logits.float().detach().requires_grad_(True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = dynamic_medial_connectivity_loss(
        logits,
        batch["dynamic_connectivity_event"],
        batch["dynamic_connectivity_pins"],
        batch["dynamic_connectivity_free"],
        probability_floor=probability_floor,
        propagation_steps=propagation_steps,
    )
    result.loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert logits.grad is not None
    gradient = logits.grad.detach()
    event_mask = batch["dynamic_connectivity_event"][:, 0] > 0
    free_mask = batch["dynamic_connectivity_free"][:, 0] > 0
    active_mask = gradient.abs().amax(dim=1) > 0
    foreground_gradient = gradient[:, 1]

    def maximum(mask: torch.Tensor) -> float:
        selected = gradient.abs().amax(dim=1)[mask]
        return float(selected.max().item()) if selected.numel() else 0.0

    return {
        "patch_id": str(raw_batch["patch_id"][0]),
        "loss": float(result.loss.item()),
        "events": int(result.events.item()),
        "targets": int(result.targets.item()),
        "mean_bottleneck_probability": float(result.mean_bottleneck_probability.item()),
        "elapsed_seconds_forward_and_backward": elapsed,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "nonzero_gradient_voxels": int(active_mask.sum().item()),
        "nonzero_gradient_voxels_in_corridor": int(
            (active_mask & event_mask).sum().item()
        ),
        "nonzero_gradient_voxels_outside_corridor": int(
            (active_mask & ~event_mask).sum().item()
        ),
        "nonzero_gradient_voxels_on_free_anchors": int(
            (active_mask & free_mask).sum().item()
        ),
        "maximum_absolute_gradient_outside_corridor": maximum(~event_mask),
        "maximum_absolute_gradient_on_free_anchors": maximum(free_mask),
        "foreground_gradient_minimum": float(foreground_gradient.min().item()),
        "foreground_gradient_maximum": float(foreground_gradient.max().item()),
        "raises_foreground_on_at_least_one_corridor_voxel": bool(
            (foreground_gradient[event_mask] < 0).any().item()
        ),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("the loss-scale audit requires CUDA")
    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.patches).resolve()
    connectivity_state = Path(args.connectivity_state).resolve()
    recipe_path = Path(args.recipe).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing audit: {output}")
    state = json.loads(connectivity_state.read_text(encoding="utf-8"))
    _recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if state.get("identity", {}).get("heldout_gate_used_for_construction") is not False:
        raise ValueError("connectivity atlas is not explicitly held-out-blind")
    target_record = str(state["identity"]["record_id"])
    schedule_samples = int(args.schedule_samples)
    batch_size = int(args.batch_size)
    probability_floor = float(args.probability_floor)
    propagation_steps = int(args.propagation_steps)
    if not 0 < probability_floor < 1 or propagation_steps <= 0:
        raise ValueError("invalid connectivity floor or propagation steps")

    dataset = VoxelPatchDataset(
        manifest,
        split="train",
        augment=False,
        dynamic_medial_connectivity_state=connectivity_state,
    )
    sampler = StratifiedEpochPartitionSampler(
        dataset.rows,
        schedule_samples,
        int(args.seed),
        total_samples=schedule_samples,
    )
    schedule = list(sampler)
    if len(schedule) != schedule_samples:
        raise RuntimeError("exact schedule has the wrong length")

    event_flags: list[bool] = []
    event_indices: list[int] = []
    for index in schedule:
        row = dataset.rows[index]
        if row.record_id != target_record:
            event_flags.append(False)
            continue
        metadata = dataset._load_dynamic_medial_connectivity(row)
        present = metadata is not None and bool(np.any(metadata[0]))
        event_flags.append(present)
        if present:
            event_indices.append(index)
    expected_rows = int(state["exact_schedule"]["event_bearing_rows"])
    if len(event_indices) != expected_rows:
        raise RuntimeError(
            "event-bearing exact-schedule count differs from atlas state: "
            f"loaded {len(event_indices)}, expected {expected_rows}"
        )

    event_batches = [
        event_flags[start : start + batch_size]
        for start in range(0, len(event_flags), batch_size)
    ]
    event_batch_count = sum(any(values) for values in event_batches)
    device = torch.device("cuda")
    model, _payload = load_voxel_checkpoint(checkpoint, device=device)
    model.eval()
    loader = DataLoader(
        Subset(dataset, event_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    sample_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _move(raw_batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(batch["image"])[0]
            for sample_index, patch_id in enumerate(raw_batch["patch_id"]):
                values = _sample_result(
                    logits,
                    batch,
                    sample_index,
                    probability_floor=probability_floor,
                    propagation_steps=propagation_steps,
                )
                event_ids = batch["dynamic_connectivity_event"][sample_index, 0]
                pins = batch["dynamic_connectivity_pins"][sample_index, 0]
                free = batch["dynamic_connectivity_free"][sample_index, 0]
                sample_rows.append(
                    {
                        "patch_id": str(patch_id),
                        **values,
                        "corridor_voxels": int((event_ids > 0).sum().item()),
                        "pin_voxels": int((pins > 0).sum().item()),
                        "free_anchor_voxels": int((free > 0).sum().item()),
                    }
                )

    by_patch = {str(row["patch_id"]): row for row in sample_rows}
    if len(by_patch) != len(sample_rows):
        raise RuntimeError("exact schedule unexpectedly repeats an event-bearing patch")
    batch_losses: list[float] = []
    batch_events: list[int] = []
    for start in range(0, len(schedule), batch_size):
        rows = [
            by_patch[dataset.rows[index].patch_id]
            for index in schedule[start : start + batch_size]
            if dataset.rows[index].patch_id in by_patch
        ]
        events = sum(int(row["events"]) for row in rows)
        weighted_loss = sum(float(row["loss"]) * int(row["events"]) for row in rows)
        batch_losses.append(weighted_loss / events if events else 0.0)
        batch_events.append(events)
    schedule_mean_loss = float(np.mean(batch_losses))
    historical_total = float(args.historical_mean_train_loss)
    weight_projection = [
        {
            "weight": float(weight),
            "estimated_schedule_mean_contribution": float(weight) * schedule_mean_loss,
            "fraction_of_historical_mean_train_loss": (
                float(weight) * schedule_mean_loss / historical_total
            ),
        }
        for weight in args.weights
    ]
    worst = max(sample_rows, key=lambda row: float(row["loss"]))
    worst_index = next(
        index
        for index in event_indices
        if dataset.rows[index].patch_id == worst["patch_id"]
    )
    autograd_probe = _autograd_probe(
        model=model,
        dataset=dataset,
        dataset_index=worst_index,
        device=device,
        probability_floor=probability_floor,
        propagation_steps=propagation_steps,
    )
    if (
        autograd_probe["nonzero_gradient_voxels_outside_corridor"] != 0
        or autograd_probe["nonzero_gradient_voxels_on_free_anchors"] != 0
        or not autograd_probe["raises_foreground_on_at_least_one_corridor_voxel"]
    ):
        raise RuntimeError("dynamic connectivity autograd scope check failed")

    return {
        "schema": SCHEMA,
        "state": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "changes_training": False,
        "heldout_gate_used": False,
        "inputs": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "patches": str(manifest),
            "patches_sha256": _sha256(manifest),
            "connectivity_state": str(connectivity_state),
            "connectivity_state_sha256": _sha256(connectivity_state),
            "recipe": str(recipe_path),
            "recipe_sha256": _sha256(recipe_path),
            "target_record_id": target_record,
        },
        "options": {
            "seed": int(args.seed),
            "schedule_samples": schedule_samples,
            "batch_size": batch_size,
            "probability_floor": probability_floor,
            "propagation_steps": propagation_steps,
            "m7_base_blend": float(args.m7_base_blend),
            "final_threshold": float(args.final_threshold),
            "required_student_probability_where_m7_is_absent": (
                float(args.final_threshold) / (1.0 - float(args.m7_base_blend))
            ),
            "historical_mean_train_loss": historical_total,
            "cuda_device": torch.cuda.get_device_name(0),
        },
        "summary": {
            "event_bearing_samples": len(event_indices),
            "event_bearing_sample_fraction": len(event_indices) / schedule_samples,
            "training_batches": len(event_batches),
            "event_bearing_batches": event_batch_count,
            "event_bearing_batch_fraction": event_batch_count / len(event_batches),
            "event_observations": sum(int(row["events"]) for row in sample_rows),
            "unique_event_ids_in_schedule": len(
                {
                    int(identifier)
                    for index in event_indices
                    for identifier in np.unique(
                        dataset._load_dynamic_medial_connectivity(dataset.rows[index])[
                            0
                        ]
                    )
                    if identifier != 0
                }
            ),
            "sample_loss": _quantiles([float(row["loss"]) for row in sample_rows]),
            "sample_mean_bottleneck_probability": _quantiles(
                [float(row["mean_bottleneck_probability"]) for row in sample_rows]
            ),
            "events_per_event_bearing_sample": _quantiles(
                [float(row["events"]) for row in sample_rows]
            ),
            "events_per_training_batch": _quantiles(
                [float(value) for value in batch_events]
            ),
            "schedule_mean_unweighted_connectivity_loss": schedule_mean_loss,
            "weight_projection": weight_projection,
            "autograd_probe": autograd_probe,
        },
        "samples": sample_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patches", required=True)
    parser.add_argument("--connectivity-state", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1203)
    parser.add_argument("--schedule-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument(
        "--probability-floor",
        type=float,
        default=DYNAMIC_MEDIAL_CONNECTIVITY_PROBABILITY_FLOOR,
    )
    parser.add_argument(
        "--propagation-steps",
        type=int,
        default=DYNAMIC_MEDIAL_CONNECTIVITY_STEPS,
    )
    parser.add_argument("--m7-base-blend", type=float, default=0.12)
    parser.add_argument("--final-threshold", type=float, default=0.17)
    parser.add_argument(
        "--historical-mean-train-loss", type=float, default=1.9886225369590067
    )
    parser.add_argument("--weights", nargs="+", type=float, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "16")
    result = audit(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(result["summary"], indent=2))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
