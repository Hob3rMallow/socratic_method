from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch

from .inference import load_voxel_checkpoint, predict_roi
from .resources import assert_cuda_power_limit, configure_cpu_budget


def _replace_directory_with_retry(
    source: Path,
    target: Path,
    *,
    attempts: int = 20,
    initial_delay_seconds: float = 0.05,
) -> None:
    """Commit an inference directory despite transient Windows file scanners."""

    if attempts <= 0 or initial_delay_seconds < 0:
        raise ValueError("invalid directory-replace retry policy")
    delay = initial_delay_seconds
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(delay)
            delay = min(0.5, delay * 1.5)


def parse_cube_id(cube_id: str) -> tuple[int, int, int]:
    pieces = cube_id.split("_")
    if len(pieces) != 3 or any(len(piece) < 2 for piece in pieces):
        raise ValueError(f"invalid cube ID: {cube_id!r}")
    if [piece[0] for piece in pieces] != ["z", "y", "x"]:
        raise ValueError(f"invalid cube ID: {cube_id!r}")
    try:
        return tuple(int(piece[1:]) for piece in pieces)  # type: ignore[return-value]
    except ValueError as error:
        raise ValueError(f"invalid cube ID: {cube_id!r}") from error


def format_cube_id(origin_zyx: tuple[int, int, int]) -> str:
    return f"z{origin_zyx[0]:05d}_y{origin_zyx[1]:05d}_x{origin_zyx[2]:05d}"


def _read_cube(path: Path, chunk_size: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing raw context cube: {path}")
    value = np.asarray(tifffile.imread(path))
    if value.shape != (chunk_size,) * 3:
        raise ValueError(f"{path}: expected {(chunk_size,) * 3}, got {value.shape}")
    return value


def _required_context_origins(
    target_origin_zyx: tuple[int, int, int],
    *,
    chunk_size: int,
    halo: int,
) -> tuple[tuple[int, int, int], ...]:
    lower = tuple(value - halo for value in target_origin_zyx)
    upper = tuple(value + chunk_size + halo for value in target_origin_zyx)
    if any(value < 0 for value in lower):
        raise ValueError("context bounds extend below zero")
    starts = [
        range(
            (lo // chunk_size) * chunk_size,
            ((hi - 1) // chunk_size) * chunk_size + 1,
            chunk_size,
        )
        for lo, hi in zip(lower, upper, strict=True)
    ]
    return tuple(product(*starts))


def _has_complete_raw_context(
    grid: Path,
    target_origin_zyx: tuple[int, int, int],
    *,
    chunk_size: int,
    halo: int,
) -> bool:
    try:
        origins = _required_context_origins(
            target_origin_zyx,
            chunk_size=chunk_size,
            halo=halo,
        )
    except ValueError:
        return False
    return all(
        (grid / "cubes_RAW" / f"{format_cube_id(origin)}.tif").is_file()
        for origin in origins
    )


def assemble_raw_context(
    grid_path: str | Path,
    target_origin_zyx: tuple[int, int, int],
    *,
    chunk_size: int,
    halo: int,
) -> np.ndarray:
    if chunk_size <= 0 or halo < 0:
        raise ValueError("chunk_size must be positive and halo non-negative")
    grid = Path(grid_path)
    lower = tuple(value - halo for value in target_origin_zyx)
    upper = tuple(value + chunk_size + halo for value in target_origin_zyx)
    if any(value < 0 for value in lower):
        raise ValueError("context bounds extend below zero")
    shape = tuple(hi - lo for lo, hi in zip(lower, upper, strict=True))
    context: np.ndarray | None = None
    coverage = np.zeros(shape, dtype=bool)
    for cube_origin in _required_context_origins(
        target_origin_zyx,
        chunk_size=chunk_size,
        halo=halo,
    ):
        cube_id = format_cube_id(cube_origin)
        cube = _read_cube(grid / "cubes_RAW" / f"{cube_id}.tif", chunk_size)
        if context is None:
            context = np.zeros(shape, dtype=cube.dtype)
        overlap_lower = tuple(
            max(lo, origin) for lo, origin in zip(lower, cube_origin, strict=True)
        )
        overlap_upper = tuple(
            min(hi, origin + chunk_size)
            for hi, origin in zip(upper, cube_origin, strict=True)
        )
        source = tuple(
            slice(lo - origin, hi - origin)
            for lo, hi, origin in zip(
                overlap_lower, overlap_upper, cube_origin, strict=True
            )
        )
        destination = tuple(
            slice(lo - base, hi - base)
            for lo, hi, base in zip(overlap_lower, overlap_upper, lower, strict=True)
        )
        context[destination] = cube[source]
        coverage[destination] = True
    if context is None or not bool(np.all(coverage)):
        raise RuntimeError("raw grid context is incomplete")
    return context


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def infer_voxel_grid(
    *,
    source_grid: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    threshold: float = 0.5,
    halo: int = 32,
    device_name: str = "cuda",
    amp_dtype_name: str = "bfloat16",
    mirror_tta: bool = True,
    max_cpu_threads: int = 16,
    target_cube_ids: Sequence[str] | None = None,
    skip_incomplete_context: bool = False,
) -> Path:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if amp_dtype_name not in {"bfloat16", "float16"}:
        raise ValueError("amp_dtype_name must be bfloat16 or float16")
    configure_cpu_budget(max_cpu_threads)
    source = Path(source_grid).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError(f"grid inference output already exists: {output}")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    chunk_size = int(manifest["chunk_size"])
    present_path = source / "cubes_PRED" / "present.json"
    published_target_ids = json.loads(present_path.read_text(encoding="utf-8"))
    if not isinstance(published_target_ids, list) or not published_target_ids:
        raise ValueError(f"{present_path}: expected a non-empty cube ID list")
    published_target_ids = sorted({str(value) for value in published_target_ids})
    if target_cube_ids is None:
        target_ids = published_target_ids
    else:
        target_ids = sorted({str(value) for value in target_cube_ids})
        if not target_ids:
            raise ValueError("target_cube_ids must not be empty")
        unknown = sorted(set(target_ids) - set(published_target_ids))
        if unknown:
            raise ValueError(
                "requested target cubes are absent from present.json: "
                + ", ".join(unknown)
            )
    requested_target_ids = target_ids
    skipped_incomplete_context_ids: list[str] = []
    if skip_incomplete_context:
        target_ids = []
        for cube_id in requested_target_ids:
            if _has_complete_raw_context(
                source,
                parse_cube_id(cube_id),
                chunk_size=chunk_size,
                halo=halo,
            ):
                target_ids.append(cube_id)
            else:
                skipped_incomplete_context_ids.append(cube_id)
        if not target_ids:
            raise ValueError("no target cubes have complete raw context")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    assert_cuda_power_limit(device)
    model, payload = load_voxel_checkpoint(checkpoint, device=device)
    context_shape = (chunk_size + 2 * halo,) * 3
    if any(size % model.config.required_divisor for size in context_shape):
        raise ValueError(
            f"context shape {context_shape} is not divisible by "
            f"{model.config.required_divisor}"
        )
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
    temporary = output.with_name(output.name + f".partial-{os.getpid()}")
    if temporary.exists():
        raise ValueError(f"stale grid inference temporary exists: {temporary}")
    prediction_root = temporary / "cubes_PRED"
    probability_root = temporary / "probability"
    prediction_root.mkdir(parents=True)
    probability_root.mkdir()
    for index, cube_id in enumerate(target_ids, 1):
        origin = parse_cube_id(cube_id)
        raw = assemble_raw_context(
            source,
            origin,
            chunk_size=chunk_size,
            halo=halo,
        )
        probability = predict_roi(
            model,
            raw,
            bounds_zyx=tuple((0, size) for size in raw.shape),  # type: ignore[arg-type]
            device=device,
            patch_shape_zyx=context_shape,
            overlap=0.5,
            amp_dtype=amp_dtype,
            autocast_enabled=device.type == "cuda",
            mirror_tta=mirror_tta,
        )
        center = tuple(slice(halo, halo + chunk_size) for _ in range(3))
        central_probability = probability[center]
        segmentation = (central_probability >= threshold).astype(np.uint8) * 255
        tifffile.imwrite(prediction_root / f"{cube_id}.tif", segmentation)
        tifffile.imwrite(
            probability_root / f"{cube_id}.tif",
            central_probability.astype(np.float16),
        )
        print(f"grid inference {index}/{len(target_ids)}: {cube_id}", flush=True)
    (prediction_root / "present.json").write_text(
        json.dumps(target_ids, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(source / "manifest.json", temporary / "source_manifest.json")
    checkpoint_sha256 = _sha256(checkpoint)
    checkpoint_metrics = payload.get("metrics")
    if not isinstance(checkpoint_metrics, dict):
        checkpoint_metrics = {}
    validation_metrics = checkpoint_metrics.get("val")
    if not isinstance(validation_metrics, dict):
        validation_metrics = {}
    checkpoint_provenance = {
        "path": str(checkpoint),
        "sha256": checkpoint_sha256,
        "epoch": payload.get("epoch"),
        "best_score": payload.get("best_score"),
        "val_dice": validation_metrics.get("dice"),
        "val_loss": validation_metrics.get("loss_total"),
    }
    inference_options = {
        "threshold": threshold,
        "halo": halo,
        "device": device_name,
        "amp_dtype": amp_dtype_name,
        "mirror_tta": mirror_tta,
        "max_cpu_threads": max_cpu_threads,
        "skip_incomplete_context": skip_incomplete_context,
        "requested_target_count": len(requested_target_ids),
        "selected_target_count": len(target_ids),
        "skipped_incomplete_context_count": len(skipped_incomplete_context_ids),
    }
    provenance: dict[str, Any] = {
        "schema": "crossres-voxel-grid-inference-v1",
        "kind": "crossres-voxel-grid-inference-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_grid": str(source),
        "checkpoint": checkpoint_provenance,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": payload.get("epoch"),
        "options": inference_options,
        "threshold": threshold,
        "halo": halo,
        "chunk_size": chunk_size,
        "context_shape_zyx": list(context_shape),
        "target_cube_ids": target_ids,
        "skipped_incomplete_context_ids": skipped_incomplete_context_ids,
        "device": device_name,
        "amp_dtype": amp_dtype_name,
        "mirror_tta": mirror_tta,
        "research_only": True,
        "deployment_ready": False,
    }
    (temporary / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _replace_directory_with_retry(temporary, output)
    return output
