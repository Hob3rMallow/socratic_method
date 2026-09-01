from __future__ import annotations

import json
import os
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .io import ArrayLike3D, read_crop
from .model import NNUNetConfig, VoxelNNUNet
from .patches import normalize_m7_ct


def sliding_window_steps(image_size: int, patch_size: int, overlap: float) -> list[int]:
    if image_size <= 0 or patch_size <= 0:
        raise ValueError("image and patch sizes must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if image_size <= patch_size:
        return [0]
    target_step = max(1, round(patch_size * (1.0 - overlap)))
    step_count = int(np.ceil((image_size - patch_size) / target_step)) + 1
    return [
        round(index * (image_size - patch_size) / (step_count - 1))
        for index in range(step_count)
    ]


def gaussian_importance_map(
    patch_shape_zyx: tuple[int, int, int], *, sigma_scale: float = 1.0 / 8.0
) -> np.ndarray:
    axes: list[np.ndarray] = []
    for size in patch_shape_zyx:
        coordinate = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
        sigma = max(1.0, size * sigma_scale)
        axes.append(np.exp(-0.5 * (coordinate / sigma) ** 2))
    result = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    result /= result.max()
    return np.maximum(result, 1.0e-4).astype(np.float32)


@torch.no_grad()
def _predict_logits(
    model: VoxelNNUNet,
    image: torch.Tensor,
    *,
    amp_dtype: torch.dtype,
    autocast_enabled: bool,
    mirror_tta: bool,
) -> torch.Tensor:
    mirror_options = (
        list(product((False, True), repeat=3)) if mirror_tta else [(False,) * 3]
    )
    total: torch.Tensor | None = None
    for flips in mirror_options:
        dimensions = tuple(index + 2 for index, enabled in enumerate(flips) if enabled)
        value = torch.flip(image, dimensions) if dimensions else image
        with torch.autocast(
            device_type=image.device.type,
            dtype=amp_dtype,
            enabled=autocast_enabled,
        ):
            prediction = model.full_resolution_logits(value)
        if dimensions:
            prediction = torch.flip(prediction, dimensions)
        total = prediction.float() if total is None else total + prediction.float()
    assert total is not None
    return total / len(mirror_options)


@torch.no_grad()
def _predict_probability(
    model: VoxelNNUNet,
    image: torch.Tensor,
    *,
    amp_dtype: torch.dtype,
    autocast_enabled: bool,
    mirror_tta: bool,
    average_logits: bool = False,
) -> torch.Tensor:
    if average_logits:
        logits = _predict_logits(
            model,
            image,
            amp_dtype=amp_dtype,
            autocast_enabled=autocast_enabled,
            mirror_tta=mirror_tta,
        )
        return torch.softmax(logits, dim=1)[:, 1]

    mirror_options = (
        list(product((False, True), repeat=3)) if mirror_tta else [(False,) * 3]
    )
    total: torch.Tensor | None = None
    for flips in mirror_options:
        dimensions = tuple(index + 2 for index, enabled in enumerate(flips) if enabled)
        value = torch.flip(image, dimensions) if dimensions else image
        with torch.autocast(
            device_type=image.device.type,
            dtype=amp_dtype,
            enabled=autocast_enabled,
        ):
            probability = torch.softmax(model.full_resolution_logits(value), dim=1)[
                :, 1
            ]
        if dimensions:
            probability = torch.flip(
                probability, tuple(dimension - 1 for dimension in dimensions)
            )
        total = probability.float() if total is None else total + probability.float()
    assert total is not None
    return total / len(mirror_options)


@torch.no_grad()
def predict_roi(
    model: VoxelNNUNet,
    volume: ArrayLike3D,
    *,
    bounds_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    device: torch.device,
    patch_shape_zyx: tuple[int, int, int] = (192, 192, 192),
    overlap: float = 0.5,
    amp_dtype: torch.dtype = torch.bfloat16,
    autocast_enabled: bool = True,
    mirror_tta: bool = True,
) -> np.ndarray:
    """Sliding-window inference for an explicit coarse-grid region of interest."""

    origin = tuple(int(pair[0]) for pair in bounds_zyx)
    shape = tuple(int(pair[1] - pair[0]) for pair in bounds_zyx)
    if any(size <= 0 for size in shape):
        raise ValueError("bounds must define a positive region")
    if any(size % model.config.required_divisor for size in patch_shape_zyx):
        raise ValueError(
            f"patch shape must be divisible by {model.config.required_divisor}"
        )
    steps = [
        sliding_window_steps(size, patch, overlap)
        for size, patch in zip(shape, patch_shape_zyx, strict=True)
    ]
    importance = gaussian_importance_map(patch_shape_zyx)
    probability_sum = np.zeros(shape, dtype=np.float32)
    weight_sum = np.zeros(shape, dtype=np.float32)
    model.eval()
    for local_origin in product(*steps):
        global_origin = tuple(
            base + local for base, local in zip(origin, local_origin, strict=True)
        )
        raw = read_crop(volume, global_origin, patch_shape_zyx)
        image = torch.from_numpy(normalize_m7_ct(raw))[None, None].to(
            device, non_blocking=True
        )
        probability = (
            _predict_probability(
                model,
                image,
                amp_dtype=amp_dtype,
                autocast_enabled=autocast_enabled and device.type == "cuda",
                mirror_tta=mirror_tta,
            )[0]
            .cpu()
            .numpy()
        )
        source_slices: list[slice] = []
        destination_slices: list[slice] = []
        for local, patch, extent in zip(
            local_origin, patch_shape_zyx, shape, strict=True
        ):
            usable = min(patch, extent - local)
            source_slices.append(slice(0, usable))
            destination_slices.append(slice(local, local + usable))
        source_key = tuple(source_slices)
        destination_key = tuple(destination_slices)
        probability_sum[destination_key] += (
            probability[source_key] * importance[source_key]
        )
        weight_sum[destination_key] += importance[source_key]
    if not np.all(weight_sum > 0):
        raise RuntimeError("sliding-window inference left uncovered voxels")
    return probability_sum / weight_sum


def load_voxel_checkpoint(
    checkpoint_path: str | Path, *, device: torch.device
) -> tuple[VoxelNNUNet, dict[str, Any]]:
    source = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    config_value = payload.get("model_config")
    if not isinstance(config_value, dict):
        raise TypeError(f"{source}: missing model_config")
    model = VoxelNNUNet(NNUNetConfig.from_dict(config_value))
    weights = payload.get("model")
    if not isinstance(weights, dict):
        raise TypeError(f"{source}: missing model state")
    model.load_state_dict(weights, strict=True)
    model.to(device)
    return model, payload


def write_prediction_zarr(
    output_path: str | Path,
    probability: np.ndarray,
    *,
    bounds_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    checkpoint: str | Path,
    threshold: float = 0.5,
) -> Path:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    try:
        import zarr
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Zarr output requires the zarr extra") from error
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError(f"prediction output already exists: {output}")
    temporary = output.with_name(output.name + f".partial-{os.getpid()}")
    if temporary.exists():
        raise ValueError(f"prediction temporary output already exists: {temporary}")
    root = zarr.open_group(str(temporary), mode="w")
    chunks = tuple(min(128, int(size)) for size in probability.shape)
    root.create_array(
        "probability",
        data=probability.astype(np.float16),
        chunks=chunks,
    )
    segmentation = (probability >= threshold).astype(np.uint8)
    root.create_array(
        "0",
        data=segmentation,
        chunks=chunks,
    )
    root.attrs.update(
        {
            "kind": "crossres-voxel-img2img-prediction",
            "checkpoint": str(Path(checkpoint).expanduser().resolve()),
            "threshold": float(threshold),
            "bounds_zyx": json.loads(json.dumps(bounds_zyx)),
            "axes": "zyx",
        }
    )
    os.replace(temporary, output)
    return output
