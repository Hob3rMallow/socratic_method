from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import PatchDataset
from .metrics import StreamingBinaryMetrics, interior_slices
from .model import load_surface_checkpoint
from .provenance import sha256_file, utc_now, write_json_atomic
from .rasterize import (
    LABEL_IGNORE,
    LABEL_SURFACE,
    collect_surface_points_zyx,
    default_options_for_pitch,
    rasterize_label_block,
)
from .schema import PairRecord
from .tifxyz import TifxyzMap
from .volume import open_volume, read_crop


class AuditError(RuntimeError):
    pass


@torch.no_grad()
def audit_checkpoint(
    *,
    checkpoint_path: str | Path,
    patch_manifest: str | Path,
    output_path: str | Path,
    split: str = "val",
    device: str = "cuda",
    amp: bool = True,
    interior_margin: int = 32,
    stamp_threshold: bool = False,
) -> dict[str, Any]:
    """Evaluate a surface checkpoint on a patch-manifest split.

    Reports ground-truth metrics (label != ignore) and distillation-target
    metrics (V-masked, binarized q) on the retained interior. With
    ``stamp_threshold``, the best-Dice threshold of the selection target is
    written as ``deploy_threshold`` into a ``.calibrated.pt`` copy -- the
    original checkpoint is never modified.
    """

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise AuditError("CUDA was requested, but torch.cuda.is_available() is false")
    model, payload = load_surface_checkpoint(checkpoint, torch_device)
    profile = str(payload.get("profile", "student"))
    dataset = PatchDataset(
        patch_manifest,
        split=split,
        kind=profile,
        augment=False,
        rot90_mode="none",
        in_channels=model.config.in_channels,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    autocast_enabled = amp and torch_device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if autocast_enabled and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    ground_truth = StreamingBinaryMetrics()
    distill = StreamingBinaryMetrics()
    for batch in loader:
        tensors = {
            key: value.to(torch_device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        with torch.autocast(
            device_type=torch_device.type,
            dtype=amp_dtype,
            enabled=autocast_enabled,
        ):
            logits = model(tensors["input"])
        interior = (slice(None), slice(None)) + interior_slices(
            tuple(logits.shape[-3:]), interior_margin
        )
        probability = torch.sigmoid(logits.float())[interior]
        label = tensors["label"][interior]
        ground_truth.update(probability, label > 0.5, label < 1.5)
        distill.update(
            probability,
            tensors["distill"][interior] >= 0.5,
            tensors["distill_valid"][interior] > 0.5,
        )

    ground_truth_result = ground_truth.result()
    distill_result = distill.result()
    selection_kind = (
        "distill" if distill_result["average_precision"] is not None else "ground-truth"
    )
    selection = distill_result if selection_kind == "distill" else ground_truth_result

    report: dict[str, Any] = {
        "schema_version": 2,
        "kind": "crossres-checkpoint-audit",
        "created_at": utc_now(),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "profile": profile,
            "epoch": payload.get("epoch"),
            "policy_profile": payload.get("policy_profile"),
        },
        "patch_manifest": {
            "path": str(Path(patch_manifest).resolve()),
            "sha256": sha256_file(patch_manifest),
            "split": split,
            "patch_count": len(dataset),
        },
        "interior_margin": interior_margin,
        "ground_truth": ground_truth_result,
        "distill_target": distill_result,
        "selection_kind": selection_kind,
    }

    if stamp_threshold:
        threshold = selection["best_dice_threshold"]
        if threshold is None or not 0.0 < float(threshold) < 1.0:
            raise AuditError(
                "cannot stamp deploy_threshold: no usable best-Dice threshold"
            )
        stamped = dict(payload)
        stamped["deploy_threshold"] = float(threshold)
        stamped["deploy_threshold_provenance"] = {
            "stamped_at": utc_now(),
            "patch_manifest_sha256": report["patch_manifest"]["sha256"],
            "split": split,
            "selection_kind": selection_kind,
            "best_dice": selection["best_dice"],
        }
        calibrated = checkpoint.with_suffix(".calibrated.pt")
        temporary = calibrated.with_name(calibrated.name + ".tmp")
        torch.save(stamped, temporary)
        os.replace(temporary, calibrated)
        report["calibrated_checkpoint"] = str(calibrated)
        report["deploy_threshold"] = float(threshold)

    write_json_atomic(output_path, report)
    return report


def audit_bridge_targets(
    *,
    distill_dir: str | Path,
    site_rows: list[dict[str, Any]],
    record: PairRecord,
    coarse_baseline_spec: str,
    output_path: str | Path,
    interior_margin: int = 32,
    band_radius_um: float = 14.0,
) -> dict[str, Any]:
    """T-G1 clause 3: compare bridged teacher q with m7 on identical voxels."""

    distill = Path(distill_dir).expanduser().resolve()
    if not distill.is_dir():
        raise AuditError(f"distill directory does not exist: {distill}")
    rows = [row for row in site_rows if row["record_id"] == record.record_id]
    if not rows:
        raise AuditError(f"no sites for record {record.record_id}")
    baseline_volume = open_volume(coarse_baseline_spec)
    maps = [TifxyzMap.load(surface.coarse_tifxyz) for surface in record.surfaces]
    raster_options = default_options_for_pitch(
        pitch_um=float(record.coarse.voxel_um), band_radius_um=band_radius_um
    )

    teacher_metrics = StreamingBinaryMetrics()
    baseline_metrics = StreamingBinaryMetrics()
    valid_voxels = 0
    evaluated_sites = 0
    for row in rows:
        site_id = str(row["site_id"])
        target_path = distill / f"{site_id}.npz"
        if not target_path.is_file():
            raise AuditError(f"missing bridge target: {target_path}")
        origin = tuple(int(item) for item in row["coarse_origin_zyx"])
        shape = tuple(int(item) for item in row["site_shape_zyx"])
        points = collect_surface_points_zyx(
            maps,
            bbox_lo_zyx=tuple(float(item) for item in origin),
            bbox_hi_zyx=tuple(
                float(origin[axis] + shape[axis]) for axis in range(3)
            ),
            margin_vox=float(raster_options.padding_vox),
        )
        label = rasterize_label_block(
            points,
            origin_zyx=origin,
            shape_zyx=shape,
            options=raster_options,
        )
        with np.load(target_path, allow_pickle=False) as archive:
            teacher_q = archive["distill_u8"].astype(np.float32) / 255.0
            bridge_valid = archive["distill_valid_u8"] > 0
        if teacher_q.shape != shape or bridge_valid.shape != shape:
            raise AuditError(
                f"{target_path}: bridge arrays do not match site shape {shape}"
            )
        baseline = read_crop(baseline_volume, origin, shape) != 0
        retained = interior_slices(shape, interior_margin)
        target = (label[retained] == LABEL_SURFACE)
        mask = bridge_valid[retained] & (label[retained] != LABEL_IGNORE)
        if not mask.any():
            continue
        teacher_metrics.update(
            torch.from_numpy(teacher_q[retained]),
            torch.from_numpy(target),
            torch.from_numpy(mask),
        )
        baseline_metrics.update(
            torch.from_numpy(baseline[retained].astype(np.float32)),
            torch.from_numpy(target),
            torch.from_numpy(mask),
        )
        valid_voxels += int(np.count_nonzero(mask))
        evaluated_sites += 1

    teacher_result = teacher_metrics.result()
    baseline_result = baseline_metrics.result()
    teacher_dice = teacher_result["dice_at_half"]
    baseline_dice = baseline_result["dice_at_half"]
    passes = (
        teacher_dice is not None
        and baseline_dice is not None
        and float(teacher_dice) > float(baseline_dice)
    )
    audit_manifest = distill / "distill_audit.json"
    report = {
        "schema_version": 1,
        "kind": "crossres-coarse-bridge-audit",
        "created_at": utc_now(),
        "record_id": record.record_id,
        "scroll_id": record.scroll_id,
        "distill_dir": str(distill),
        "distill_audit_sha256": (
            sha256_file(audit_manifest) if audit_manifest.is_file() else None
        ),
        "coarse_baseline": coarse_baseline_spec,
        "site_count": len(rows),
        "evaluated_site_count": evaluated_sites,
        "valid_voxel_count": valid_voxels,
        "interior_margin": interior_margin,
        "teacher_bridge": teacher_result,
        "m7_baseline": baseline_result,
        "gate": {
            "name": "T-G1-clause-3",
            "criterion": "teacher_bridge.dice_at_half > m7_baseline.dice_at_half",
            "passes": passes,
            "dice_delta": (
                float(teacher_dice) - float(baseline_dice)
                if teacher_dice is not None and baseline_dice is not None
                else None
            ),
        },
    }
    write_json_atomic(output_path, report)
    return report
