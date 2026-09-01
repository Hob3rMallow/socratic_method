from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"tensor_shape": list(value.shape), "tensor_dtype": str(value.dtype)}
    return repr(value)


def export_checkpoint(
    checkpoint: Path,
    destination: Path,
    *,
    recipe: Path,
    metrics: Path,
    selection: Path,
    model_card: Path,
    force: bool = False,
) -> dict[str, Any]:
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - dependency guard
        raise RuntimeError("install the publish extra before exporting") from error

    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    selection = selection.expanduser().resolve()
    if not selection.is_file():
        raise FileNotFoundError(selection)
    selection_record = json.loads(selection.read_text(encoding="utf-8"))
    if not isinstance(selection_record, dict):
        raise TypeError("selection record must be an object")
    if selection_record.get("status") != "release-candidate-selected":
        raise ValueError("selection record does not declare a release candidate")
    selected_checkpoint_sha256 = str(selection_record["checkpoint_sha256"])
    selected_threshold = float(selection_record["threshold"])
    selected_samples = int(selection_record["samples"])
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != selected_checkpoint_sha256:
        raise ValueError(
            "checkpoint SHA-256 does not match recipes/v31/selection.json: "
            f"expected {selected_checkpoint_sha256}, got {checkpoint_sha256}"
        )
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    outputs = [
        destination / "model.safetensors",
        destination / "config.json",
        destination / "preprocessor_config.json",
        destination / "checkpoint_metadata.json",
        destination / "training_recipe.json",
        destination / "observed_metrics.json",
        destination / "selection.json",
        destination / "README.md",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"export would replace existing files: {names}")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be an object")
    raw_weights = payload.get("model")
    model_config = payload.get("model_config")
    if not isinstance(raw_weights, dict) or not raw_weights:
        raise TypeError("checkpoint is missing its model state")
    if not isinstance(model_config, dict):
        raise TypeError("checkpoint is missing model_config")
    weights = {
        str(name): tensor.detach().cpu().contiguous()
        for name, tensor in raw_weights.items()
    }
    if not all(isinstance(tensor, torch.Tensor) for tensor in weights.values()):
        raise TypeError("model state contains a non-tensor value")

    safetensor_path = destination / "model.safetensors"
    save_file(
        weights,
        str(safetensor_path),
        metadata={
            "format": "pt",
            "architecture": "crossres_pred.voxel.model.VoxelNNUNet",
            "recipe": "v31.1-pherc0139-dynamic-medial-duration-8192",
        },
    )
    model_sha256 = _sha256(safetensor_path)
    config = {
        "model_type": "socratic-m7-xr",
        "architectures": ["VoxelNNUNet"],
        "implementation": "crossres_pred.voxel.model.VoxelNNUNet",
        "model_config": model_config,
        "parameter_count": sum(int(tensor.numel()) for tensor in weights.values()),
        "torch_dtype": "float32",
        "student_only_inference": True,
        "teacher_required_at_inference": False,
        "m7_blend_at_inference": False,
        "selected_training_samples": selected_samples,
        "operating_threshold": selected_threshold,
        "source_checkpoint_sha256": checkpoint_sha256,
        "model_safetensors_sha256": model_sha256,
    }
    _atomic_json(destination / "config.json", config)
    _atomic_json(
        destination / "preprocessor_config.json",
        {
            "input_layout": "NCDHW",
            "input_channels": 1,
            "ct_clip": [0.0, 212.0],
            "ct_mean": 87.54424285888672,
            "ct_std": 47.74376678466797,
            "spatial_divisor": 32,
            "output": "softmax surface probability from class index 1",
            "operating_threshold": selected_threshold,
            "threshold_status": "selected by locked and blinded morphology review",
        },
    )
    excluded = {"model", "optimizer", "scaler", "rng_state"}
    metadata = {
        key: _json_safe(value)
        for key, value in payload.items()
        if key not in excluded
    }
    metadata.update(
        {
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": checkpoint_sha256,
            "model_safetensors_sha256": model_sha256,
            "exported_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    _atomic_json(destination / "checkpoint_metadata.json", metadata)
    shutil.copy2(recipe, destination / "training_recipe.json")
    shutil.copy2(metrics, destination / "observed_metrics.json")
    shutil.copy2(selection, destination / "selection.json")
    card = model_card.read_text(encoding="utf-8")
    card = card.replace("{{CHECKPOINT_SHA256}}", checkpoint_sha256)
    card = card.replace("{{MODEL_SHA256}}", model_sha256)
    card = card.replace("{{EXPORT_DATE}}", datetime.now(UTC).date().isoformat())
    (destination / "README.md").write_text(card, encoding="utf-8", newline="\n")
    return config


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Export a selected Socratic Method checkpoint for Hugging Face"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--recipe", type=Path, default=root / "recipes" / "v31" / "recipe.json"
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=root / "recipes" / "v31" / "observed_metrics.json",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=root / "recipes" / "v31" / "selection.json",
    )
    parser.add_argument(
        "--model-card", type=Path, default=root / "huggingface" / "README.md"
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = export_checkpoint(
        args.checkpoint,
        args.destination,
        recipe=args.recipe.expanduser().resolve(),
        metrics=args.metrics.expanduser().resolve(),
        selection=args.selection.expanduser().resolve(),
        model_card=args.model_card.expanduser().resolve(),
        force=args.force,
    )
    print(json.dumps(config, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
