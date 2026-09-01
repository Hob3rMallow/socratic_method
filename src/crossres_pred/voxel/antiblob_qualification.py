from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

from .checkpoint_audit import CHECKPOINT_AUDIT_SCHEMA
from .inference import _predict_probability, load_voxel_checkpoint
from .patches import VoxelPatchDataset
from .resources import assert_cuda_power_limit, configure_cpu_budget

QUALIFICATION_SCHEMA = "crossres-antiblob-pilot-qualification-v1"
MORPHOLOGY_CONTRACT = "six-neighbor-two-erosion-known-domain-v1"


@dataclass(frozen=True)
class AntiblobQualificationOptions:
    minimum_scroll_dice_gain: float = -0.01
    minimum_precision_gain: float = -0.02
    minimum_foreground_ratio: float = 0.75
    maximum_foreground_ratio: float = 1.25
    maximum_interior_excess: float = 0.05
    device: str = "cuda"
    amp_dtype: str = "bfloat16"
    mirror_tta: bool = True
    num_workers: int = 2
    max_cpu_threads: int = 16

    def validate(self) -> None:
        if self.minimum_foreground_ratio <= 0:
            raise ValueError("minimum_foreground_ratio must be positive")
        if self.maximum_foreground_ratio < self.minimum_foreground_ratio:
            raise ValueError("foreground ratio bounds are reversed")
        if self.maximum_interior_excess < 0:
            raise ValueError("maximum_interior_excess cannot be negative")
        if self.amp_dtype not in {"bfloat16", "float16"}:
            raise ValueError("amp_dtype must be bfloat16 or float16")
        if not 1 <= self.max_cpu_threads <= 16:
            raise ValueError("max_cpu_threads must be in [1, 16]")
        if not 0 <= self.num_workers < self.max_cpu_threads:
            raise ValueError("num_workers must be in [0, max_cpu_threads)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected an object")
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


def _points_by_threshold(sweep: dict[str, Any]) -> dict[float, dict[str, Any]]:
    points = sweep.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("checkpoint audit has no threshold points")
    result = {round(float(point["threshold"]), 8): point for point in points}
    if len(result) != len(points):
        raise ValueError("checkpoint audit thresholds are not unique")
    return result


def _candidate_metrics(
    initial: dict[str, Any],
    trained: dict[str, Any],
    *,
    options: AntiblobQualificationOptions,
) -> dict[str, Any]:
    initial_surface = initial["surface_at_2vox"]
    trained_surface = trained["surface_at_2vox"]
    initial_scrolls = initial["scrolls"]
    trained_scrolls = trained["scrolls"]
    if set(initial_scrolls) != set(trained_scrolls):
        raise ValueError("initial/trained audit scroll sets differ")
    scroll_gains = {
        scroll: float(trained_scrolls[scroll]["dice"])
        - float(initial_scrolls[scroll]["dice"])
        for scroll in sorted(initial_scrolls)
    }
    precision_gain = float(trained_surface["precision"]) - float(
        initial_surface["precision"]
    )
    foreground_ratio = float(trained["foreground_ratio"])
    gates = {
        "per_scroll_dice": all(
            gain >= options.minimum_scroll_dice_gain for gain in scroll_gains.values()
        ),
        "surface_precision": precision_gain >= options.minimum_precision_gain,
        "foreground_ratio": (
            options.minimum_foreground_ratio
            <= foreground_ratio
            <= options.maximum_foreground_ratio
        ),
    }
    return {
        "threshold": float(trained["threshold"]),
        "macro_surface_f0_5_at_2vox": float(trained["macro_surface_f0_5_at_2vox"]),
        "surface_f0_5_gain_vs_initial": float(trained_surface["f0_5"])
        - float(initial_surface["f0_5"]),
        "surface_precision": float(trained_surface["precision"]),
        "surface_precision_gain_vs_initial": precision_gain,
        "foreground_ratio": foreground_ratio,
        "macro_scroll_dice": float(trained["macro_scroll_dice"]),
        "scroll_dice_gain_vs_initial": scroll_gains,
        "gates": gates,
        "preliminary_qualified": all(gates.values()),
    }


def select_antiblob_operating_point(
    initial_sweep: dict[str, Any],
    trained_sweep: dict[str, Any],
    *,
    options: AntiblobQualificationOptions,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select by tolerant macro F0.5 subject to anti-regression gates."""

    options.validate()
    initial_points = _points_by_threshold(initial_sweep)
    trained_points = _points_by_threshold(trained_sweep)
    if set(initial_points) != set(trained_points):
        raise ValueError("initial/trained threshold sweeps differ")
    candidates = [
        _candidate_metrics(
            initial_points[threshold],
            trained_points[threshold],
            options=options,
        )
        for threshold in sorted(trained_points)
    ]
    qualified = [row for row in candidates if row["preliminary_qualified"]]
    pool = qualified or candidates
    selected = max(
        pool,
        key=lambda row: (
            row["macro_surface_f0_5_at_2vox"],
            row["surface_precision"],
            row["macro_scroll_dice"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    return selected, candidates


def morphology_counts(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, int]:
    valid = target != 2
    truth = (target == 1) & valid
    predicted = np.asarray(prediction, dtype=bool) & valid
    structure = ndimage.generate_binary_structure(3, 1)
    eligible = ndimage.binary_erosion(valid, structure=structure, iterations=2)
    truth_interior = (
        ndimage.binary_erosion(truth, structure=structure, iterations=2) & eligible
    )
    predicted_interior = (
        ndimage.binary_erosion(predicted, structure=structure, iterations=2) & eligible
    )
    return {
        "target_positive_voxels": int(truth.sum()),
        "predicted_positive_voxels": int(predicted.sum()),
        "target_two_erode_interior_voxels": int(truth_interior.sum()),
        "predicted_two_erode_interior_voxels": int(predicted_interior.sum()),
    }


def _summarize_morphology(counts: dict[str, int]) -> dict[str, int | float]:
    target_positive = counts["target_positive_voxels"]
    predicted_positive = counts["predicted_positive_voxels"]
    return {
        **counts,
        "foreground_ratio": predicted_positive / max(1, target_positive),
        "target_two_erode_retained_fraction": counts["target_two_erode_interior_voxels"]
        / max(1, target_positive),
        "predicted_two_erode_retained_fraction": counts[
            "predicted_two_erode_interior_voxels"
        ]
        / max(1, predicted_positive),
    }


@torch.no_grad()
def qualify_antiblob_checkpoint(
    *,
    checkpoint_path: str | Path,
    initial_audit_path: str | Path,
    trained_audit_path: str | Path,
    patch_manifest_path: str | Path,
    output_path: str | Path,
    options: AntiblobQualificationOptions | None = None,
) -> Path:
    options = options or AntiblobQualificationOptions()
    options.validate()
    configure_cpu_budget(options.max_cpu_threads, reserve_processes=options.num_workers)
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    manifest = Path(patch_manifest_path).expanduser().resolve()
    initial_audit_file = Path(initial_audit_path).expanduser().resolve()
    trained_audit_file = Path(trained_audit_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError(f"qualification output already exists: {output}")
    initial_audit = _load_object(initial_audit_file)
    trained_audit = _load_object(trained_audit_file)
    for label, audit in (("initial", initial_audit), ("trained", trained_audit)):
        if audit.get("schema") != CHECKPOINT_AUDIT_SCHEMA:
            raise ValueError(f"{label} checkpoint audit schema changed")
        if audit["patch_manifest"]["sha256"] != _sha256(manifest):
            raise ValueError(f"{label} checkpoint audit uses another corpus")
    if trained_audit["checkpoint"]["sha256"] != _sha256(checkpoint):
        raise ValueError("trained checkpoint audit uses another checkpoint")
    selected, candidates = select_antiblob_operating_point(
        initial_audit["sweep"], trained_audit["sweep"], options=options
    )
    threshold = float(selected["threshold"])

    device = torch.device(options.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    assert_cuda_power_limit(device)
    model, _ = load_voxel_checkpoint(checkpoint, device=device)
    dataset = VoxelPatchDataset(manifest, split="val", augment=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=options.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=options.num_workers > 0,
    )
    amp_dtype = torch.bfloat16 if options.amp_dtype == "bfloat16" else torch.float16
    totals = {
        "target_positive_voxels": 0,
        "predicted_positive_voxels": 0,
        "target_two_erode_interior_voxels": 0,
        "predicted_two_erode_interior_voxels": 0,
    }
    by_scroll: dict[str, dict[str, int]] = {}
    model.eval()
    for index, batch in enumerate(loader, 1):
        probability = (
            _predict_probability(
                model,
                batch["image"].to(device, non_blocking=True),
                amp_dtype=amp_dtype,
                autocast_enabled=device.type == "cuda",
                mirror_tta=options.mirror_tta,
            )[0]
            .cpu()
            .numpy()
        )
        target = batch["target"][0, 0].numpy()
        counts = morphology_counts(probability >= threshold, target)
        scroll = str(batch["scroll_id"][0])
        scroll_counts = by_scroll.setdefault(scroll, {name: 0 for name in totals})
        for name, value in counts.items():
            totals[name] += value
            scroll_counts[name] += value
        if index % 10 == 0 or index == len(dataset):
            print(f"anti-blob morphology {index:,}/{len(dataset):,}", flush=True)
    morphology = _summarize_morphology(totals)
    morphology["contract"] = MORPHOLOGY_CONTRACT
    morphology["scrolls"] = {
        scroll: _summarize_morphology(counts)
        for scroll, counts in sorted(by_scroll.items())
    }
    interior_excess = float(
        morphology["predicted_two_erode_retained_fraction"]
    ) - float(morphology["target_two_erode_retained_fraction"])
    morphology_gate = interior_excess <= options.maximum_interior_excess
    final_gates = {
        **selected["gates"],
        "two_erode_interior": morphology_gate,
    }
    output.mkdir(parents=True)
    report = {
        "schema": QUALIFICATION_SCHEMA,
        "qualified": all(final_gates.values()),
        "selection_metric": "macro_surface_f0_5_at_2vox",
        "selected": selected,
        "candidates": candidates,
        "gates": final_gates,
        "morphology": morphology,
        "morphology_interior_excess": interior_excess,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
        },
        "patch_manifest": {
            "path": str(manifest),
            "sha256": _sha256(manifest),
        },
        "initial_audit": {
            "path": str(initial_audit_file),
            "sha256": _sha256(initial_audit_file),
        },
        "trained_audit": {
            "path": str(trained_audit_file),
            "sha256": _sha256(trained_audit_file),
        },
        "options": asdict(options),
    }
    result = output / "qualification.json"
    _atomic_json(result, report)
    return result
