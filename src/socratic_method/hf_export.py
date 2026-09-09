from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .recipe import default_recipe_path

RAW_STUDENT_INFERENCE = "raw-student-only-no-m7-blend-no-teacher"
DEFAULT_SELECTION_STATUS = "release-candidate-selected"
DEFAULT_THRESHOLD_STATUS = (
    "selected by registered morphology and blind anti-blob review"
)
DEFAULT_MODEL_TYPE = "socratic-m7-xr"
TEST_TIME_AUGMENTATIONS = ("none", "8-way-mirror")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _storage_key(tensor: Any) -> tuple[Any, ...]:
    return (
        tensor.untyped_storage().data_ptr(),
        int(tensor.storage_offset()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
    )


def deduplicate_shared_tensors(
    weights: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Store each parameter once; map every aliased key to its stored key.

    The nnU-Net state dictionary exposes the encoder under several prefixes
    (``network.encoder.*``, ``network.decoder.encoder.*``, ``conv.*`` and
    ``all_modules.*``) that reference the same storage, which safetensors
    refuses to write twice. The canonical key of each group is its
    lexicographically first name, so the choice is deterministic.
    """
    groups: dict[tuple[Any, ...], list[str]] = {}
    for name in sorted(weights):
        groups.setdefault(_storage_key(weights[name]), []).append(name)
    canonical: dict[str, Any] = {}
    aliases: dict[str, str] = {}
    for names in groups.values():
        canonical[names[0]] = weights[names[0]]
        for alias in names[1:]:
            aliases[alias] = names[0]
    return canonical, aliases


def load_exported_state_dict(path: Path) -> dict[str, Any]:
    """Load ``model.safetensors`` and re-expand the tied keys for strict loading."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = Path(path)
    state = load_file(str(path))
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
    for alias, canonical in json.loads(metadata.get("tied_keys", "{}")).items():
        state[alias] = state[canonical]
    return state


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _release_contract(recipe: Path, qualification: Path) -> dict[str, Any]:
    recipe_value = _read_json(recipe)
    release = recipe_value.get("release")
    if not isinstance(release, dict) or release.get("status") != "selected":
        raise ValueError("training recipe does not declare a selected release")
    selected = release.get("selected_checkpoint")
    if not isinstance(selected, dict):
        raise TypeError("training recipe is missing selected_checkpoint")
    model = recipe_value.get("model")
    model = model if isinstance(model, dict) else {}
    contract = {
        "samples": int(selected["samples"]),
        "sha256": str(selected["sha256"]),
        "bytes": int(selected["bytes"]),
        "operating_threshold": float(release["operating_threshold"]),
        "threshold_selection_contract": str(release["threshold_selection_contract"]),
        "inference": str(release["inference"]),
        "version": str(recipe_value.get("version", "")),
        "selection_status": str(
            release.get("selection_status", DEFAULT_SELECTION_STATUS)
        ),
        "threshold_status": str(
            release.get("threshold_status", DEFAULT_THRESHOLD_STATUS)
        ),
        "model_type": str(model.get("hf_model_type", DEFAULT_MODEL_TYPE)),
        "test_time_augmentation": str(release.get("test_time_augmentation", "none")),
    }
    if contract["samples"] <= 0 or contract["bytes"] <= 0:
        raise ValueError("selected checkpoint samples and bytes must be positive")
    if len(contract["sha256"]) != 64 or any(
        character not in "0123456789abcdef" for character in contract["sha256"]
    ):
        raise ValueError("selected checkpoint SHA-256 is invalid")
    if not 0.0 < contract["operating_threshold"] < 1.0:
        raise ValueError("operating threshold must be in (0, 1)")
    if contract["inference"] != RAW_STUDENT_INFERENCE:
        raise ValueError("release contract must export the raw student only")
    if not contract["version"]:
        raise ValueError("training recipe must declare a version")
    if contract["test_time_augmentation"] not in TEST_TIME_AUGMENTATIONS:
        raise ValueError("release test_time_augmentation must be one of none, 8-way-mirror")

    qualification_value = _read_json(qualification)
    qualification_selection = qualification_value.get("selection")
    if not isinstance(qualification_selection, dict):
        raise TypeError("release qualification is missing its selection")
    checks = {
        "checkpoint_samples": contract["samples"],
        "checkpoint_sha256": contract["sha256"],
        "checkpoint_bytes": contract["bytes"],
        "operating_threshold": contract["operating_threshold"],
        "model_composition": contract["inference"],
    }
    for name, expected in checks.items():
        if qualification_selection.get(name) != expected:
            raise ValueError(
                f"release qualification {name} does not match the recipe"
            )
    if "test_time_augmentation" in qualification_selection and qualification_selection["test_time_augmentation"] != contract["test_time_augmentation"]:
        raise ValueError("release qualification test_time_augmentation does not match the recipe")
    return contract


def _validate_selected_checkpoint(checkpoint: Path, contract: dict[str, Any]) -> str:
    actual_bytes = checkpoint.stat().st_size
    if actual_bytes != int(contract["bytes"]):
        raise ValueError(
            f"selected checkpoint byte size mismatch: {actual_bytes} != "
            f"{contract['bytes']}"
        )
    actual_sha256 = _sha256(checkpoint)
    if actual_sha256 != str(contract["sha256"]):
        raise ValueError(
            "selected checkpoint SHA-256 mismatch: "
            f"{actual_sha256} != {contract['sha256']}"
        )
    return actual_sha256


def _validate_selection_summary(selection: Path, contract: dict[str, Any]) -> None:
    value = _read_json(selection)
    checks = {
        "status": contract.get("selection_status", DEFAULT_SELECTION_STATUS),
        "samples": contract["samples"],
        "threshold": contract["operating_threshold"],
        "checkpoint_sha256": contract["sha256"],
    }
    for name, expected in checks.items():
        if value.get(name) != expected:
            raise ValueError(f"selection summary {name} does not match the recipe")


def _preprocessor_config(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_layout": "NCDHW",
        "input_channels": 1,
        "ct_clip": [0.0, 212.0],
        "ct_mean": 87.54424285888672,
        "ct_std": 47.74376678466797,
        "spatial_divisor": 32,
        "output": "softmax surface probability from class index 1",
        "operating_threshold": float(contract["operating_threshold"]),
        "threshold_status": contract.get("threshold_status", DEFAULT_THRESHOLD_STATUS),
        "threshold_selection_contract": contract["threshold_selection_contract"],
        "test_time_augmentation": contract.get("test_time_augmentation", "none"),
        "inference": "threshold the class-1 softmax probability; with 8-way-mirror TTA average the softmax over the eight axis-flip passes before thresholding",
    }


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


def default_model_card(recipe: Path, root: Path | None = None) -> Path:
    """``huggingface/<recipe directory name>/README.md`` for the given recipe."""
    root = root or Path(__file__).resolve().parents[2]
    return root / "huggingface" / recipe.resolve().parent.name / "README.md"


def export_checkpoint(
    checkpoint: Path,
    destination: Path,
    *,
    recipe: Path,
    metrics: Path | None = None,
    model_card: Path | None = None,
    qualification: Path | None = None,
    selection: Path | None = None,
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
    recipe = recipe.expanduser().resolve()
    metrics = (
        metrics.expanduser().resolve()
        if metrics is not None
        else recipe.with_name("observed_metrics.json")
    )
    qualification = (
        qualification.expanduser().resolve()
        if qualification is not None
        else recipe.with_name("release_qualification.json")
    )
    selection = (
        selection.expanduser().resolve()
        if selection is not None
        else recipe.with_name("selection.json")
    )
    if not selection.is_file():
        selection = None
    model_card = (
        model_card.expanduser().resolve()
        if model_card is not None
        else default_model_card(recipe)
    )
    if not model_card.is_file():
        raise FileNotFoundError(f"model card template is missing: {model_card}")
    contract = _release_contract(recipe, qualification)
    if selection is not None:
        _validate_selection_summary(selection, contract)
    checkpoint_sha256 = _validate_selected_checkpoint(checkpoint, contract)
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    outputs = [
        destination / "model.safetensors",
        destination / "config.json",
        destination / "preprocessor_config.json",
        destination / "checkpoint_metadata.json",
        destination / "training_recipe.json",
        destination / "observed_metrics.json",
        destination / "release_qualification.json",
        destination / "README.md",
    ]
    if selection is not None:
        outputs.append(destination / "selection.json")
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
    if int(payload.get("cumulative_samples", -1)) != int(contract["samples"]):
        raise ValueError("checkpoint cumulative_samples does not match release recipe")
    if int(payload.get("requested_samples", -1)) != int(contract["samples"]):
        raise ValueError("checkpoint requested_samples does not match release recipe")
    weights = {
        str(name): tensor.detach().cpu().contiguous()
        for name, tensor in raw_weights.items()
    }
    if not all(isinstance(tensor, torch.Tensor) for tensor in weights.values()):
        raise TypeError("model state contains a non-tensor value")
    stored, tied_keys = deduplicate_shared_tensors(weights)

    safetensor_path = destination / "model.safetensors"
    save_file(
        stored,
        str(safetensor_path),
        metadata={
            "format": "pt",
            "architecture": "crossres_pred.voxel.model.VoxelNNUNet",
            "recipe": contract["version"],
            "tied_keys": json.dumps(tied_keys, sort_keys=True),
        },
    )
    model_sha256 = _sha256(safetensor_path)
    config = {
        "model_type": contract["model_type"],
        "architectures": ["VoxelNNUNet"],
        "implementation": "crossres_pred.voxel.model.VoxelNNUNet",
        "model_config": model_config,
        "parameter_count": sum(int(tensor.numel()) for tensor in stored.values()),
        "stored_tensors": len(stored),
        "tied_keys": len(tied_keys),
        "state_dict_loader": "socratic_method.hf_export.load_exported_state_dict",
        "torch_dtype": "float32",
        "student_only_inference": True,
        "teacher_required_at_inference": False,
        "m7_blend_at_inference": False,
        "recipe_version": contract["version"],
        "selected_checkpoint_samples": int(contract["samples"]),
        "operating_threshold": float(contract["operating_threshold"]),
        "test_time_augmentation": contract.get("test_time_augmentation", "none"),
        "threshold_selection_contract": contract["threshold_selection_contract"],
        "source_checkpoint_sha256": checkpoint_sha256,
        "model_safetensors_sha256": model_sha256,
    }
    _atomic_json(destination / "config.json", config)
    _atomic_json(destination / "preprocessor_config.json", _preprocessor_config(contract))
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
            "tied_keys": tied_keys,
            "recipe_version": contract["version"],
            "selected_checkpoint_samples": int(contract["samples"]),
            "operating_threshold": float(contract["operating_threshold"]),
            "test_time_augmentation": contract.get("test_time_augmentation", "none"),
            "threshold_selection_contract": contract[
                "threshold_selection_contract"
            ],
            "exported_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    _atomic_json(destination / "checkpoint_metadata.json", metadata)
    shutil.copy2(recipe, destination / "training_recipe.json")
    shutil.copy2(metrics, destination / "observed_metrics.json")
    shutil.copy2(qualification, destination / "release_qualification.json")
    if selection is not None:
        shutil.copy2(selection, destination / "selection.json")
    card = model_card.read_text(encoding="utf-8")
    card = card.replace("{{CHECKPOINT_SHA256}}", checkpoint_sha256)
    card = card.replace("{{MODEL_SHA256}}", model_sha256)
    card = card.replace("{{EXPORT_DATE}}", datetime.now(UTC).date().isoformat())
    card = card.replace("{{CHECKPOINT_SAMPLES}}", f"{contract['samples']:,}")
    card = card.replace(
        "{{OPERATING_THRESHOLD}}", f"{contract['operating_threshold']:.2f}"
    )
    card = card.replace(
        "{{TEST_TIME_AUGMENTATION}}",
        "eight-way mirror TTA" if contract.get("test_time_augmentation") == "8-way-mirror" else "no TTA",
    )
    (destination / "README.md").write_text(card, encoding="utf-8", newline="\n")
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a selected Socratic Method checkpoint for Hugging Face"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=default_recipe_path(),
        help="recipe JSON (default: the executable recipe named by CURRENT_MODEL.json)",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="observed metrics JSON (default: observed_metrics.json beside the recipe)",
    )
    parser.add_argument(
        "--qualification",
        type=Path,
        default=None,
        help="release qualification JSON (default: beside the recipe)",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=None,
        help="selection summary JSON (default: selection.json beside the recipe)",
    )
    parser.add_argument(
        "--model-card",
        type=Path,
        default=None,
        help="model card template (default: huggingface/<recipe dir>/README.md)",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = export_checkpoint(
        args.checkpoint,
        args.destination,
        recipe=args.recipe.expanduser().resolve(),
        metrics=args.metrics,
        qualification=args.qualification,
        selection=args.selection,
        model_card=args.model_card,
        force=args.force,
    )
    print(json.dumps(config, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
