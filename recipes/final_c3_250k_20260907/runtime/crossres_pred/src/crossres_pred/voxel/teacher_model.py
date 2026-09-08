from __future__ import annotations

import importlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

from .model import NNUNetConfig, VoxelNNUNet, initialize_from_m7

OFFICIAL_SURFACE_TEACHER_THRESHOLD = 0.45
PINNED_VILLA_COMMIT = "f9dacc7410075cdb56d81993962ff34d11377366"


@dataclass
class LoadedTeacher:
    """A segmentation teacher plus its inference contract."""

    model: nn.Module
    kind: str
    patch_shape_zyx: tuple[int, int, int]
    required_divisor: int
    normalization: str
    default_threshold: float
    default_mirror_tta: bool
    tta_average_logits: bool
    provenance: dict[str, Any]


class _VillaSurfaceAdapter(nn.Module):
    """Expose Villa's named task output as ordinary two-channel logits."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def full_resolution_logits(self, image: torch.Tensor) -> torch.Tensor:
        result = self.network(image)
        if not isinstance(result, dict) or "surface" not in result:
            raise TypeError("Villa surface teacher must return a 'surface' tensor")
        logits = result["surface"]
        if isinstance(logits, (list, tuple)):
            if not logits:
                raise ValueError("Villa surface teacher returned no logits")
            logits = logits[0]
        if not isinstance(logits, torch.Tensor) or logits.ndim != 5:
            raise TypeError("Villa surface logits must be a B-C-Z-Y-X tensor")
        if logits.shape[1] != 2:
            raise ValueError(
                f"Villa surface teacher must emit two channels, got {logits.shape[1]}"
            )
        return logits

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.full_resolution_logits(image)


def normalize_instance_zscore(image: np.ndarray) -> np.ndarray:
    """Match Villa inference's per-patch, per-channel instance z-score."""

    result = np.asarray(image, dtype=np.float32).copy()
    mean = float(np.mean(result))
    standard_deviation = max(float(np.std(result)), 1.0e-8)
    result -= mean
    result /= standard_deviation
    return result


def normalize_teacher_ct(image: np.ndarray, normalization: str) -> np.ndarray:
    if normalization == "instance_zscore":
        return normalize_instance_zscore(image)
    if normalization == "m7_fixed":
        from .patches import normalize_m7_ct

        return normalize_m7_ct(image)
    raise ValueError(f"unsupported teacher normalization: {normalization!r}")


def _strip_wrapper_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    prefixes = ("module.", "_orig_mod.")

    def strip(key: str) -> str:
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        return key

    return {strip(key): value for key, value in state_dict.items()}


def _unavailable_optional_component(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("this optional Villa component is not used by the ps256 teacher")


class _UnavailableOptionalModule(nn.Module):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        super().__init__()
        _unavailable_optional_component()


def _install_villa_import_shims(villa_source: Path) -> None:
    """Import only Villa's nnU-Net source, without its unrelated optional stack."""

    package_path = villa_source / "vesuvius"
    if not package_path.is_dir():
        raise FileNotFoundError(f"Villa source must contain vesuvius/: {villa_source}")
    existing = sys.modules.get("vesuvius")
    if existing is None:
        namespace = types.ModuleType("vesuvius")
        namespace.__path__ = [str(package_path)]  # type: ignore[attr-defined]
        namespace.__package__ = "vesuvius"
        sys.modules["vesuvius"] = namespace
    else:
        paths = [Path(value).resolve() for value in getattr(existing, "__path__", [])]
        if package_path.resolve() not in paths:
            raise RuntimeError(
                "a different vesuvius package is already imported; start a clean process"
            )

    primus_name = "vesuvius.models.build.primus_wrapper"
    if primus_name not in sys.modules:
        primus = types.ModuleType(primus_name)
        primus.PrimusEncoder = _UnavailableOptionalModule
        primus.PrimusDecoder = _UnavailableOptionalModule
        sys.modules[primus_name] = primus

    backbone_package_name = "vesuvius.models.build.pretrained_backbones"
    if backbone_package_name not in sys.modules:
        backbone_package = types.ModuleType(backbone_package_name)
        backbone_package.__path__ = []  # type: ignore[attr-defined]
        sys.modules[backbone_package_name] = backbone_package
    dinov2_name = backbone_package_name + ".dinov2"
    if dinov2_name not in sys.modules:
        dinov2 = types.ModuleType(dinov2_name)
        dinov2.build_dinov2_backbone = _unavailable_optional_component
        dinov2.build_dinov2_decoder = _unavailable_optional_component
        sys.modules[dinov2_name] = dinov2


def _read_villa_commit(villa_source: Path) -> str | None:
    repository = villa_source.parent.parent
    head_path = repository / ".git" / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    reference = repository / ".git" / head[5:]
    if reference.is_file():
        return reference.read_text(encoding="utf-8").strip()
    packed_refs = repository / ".git" / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + head[5:]):
                return line.split(" ", 1)[0]
    return None


def _validate_villa_config(config: dict[str, Any]) -> tuple[int, int, int]:
    targets = config.get("targets")
    if not isinstance(targets, dict) or set(targets) != {"surface"}:
        raise ValueError("native teacher must contain exactly the surface task")
    surface = targets["surface"]
    if not isinstance(surface, dict) or int(surface.get("out_channels", 0)) != 2:
        raise ValueError("native teacher surface task must have two output channels")
    patch_value = config.get("train_patch_size", config.get("patch_size"))
    if not isinstance(patch_value, (list, tuple)) or len(patch_value) != 3:
        raise ValueError("native teacher has no three-dimensional patch size")
    patch_shape = tuple(int(value) for value in patch_value)
    if any(value <= 0 or value % 64 for value in patch_shape):
        raise ValueError("native teacher patch size must be divisible by 64")
    return patch_shape  # type: ignore[return-value]


def _load_villa_teacher(
    payload: dict[str, Any], *, villa_source: Path, device: torch.device
) -> LoadedTeacher:
    config = payload.get("model_config")
    if not isinstance(config, dict):
        raise TypeError("Villa teacher checkpoint has no model_config")
    patch_shape = _validate_villa_config(config)
    sidecar_path = Path(payload.get("_checkpoint_path", "")).with_name("config.json")
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar != config:
            raise ValueError(
                "teacher config.json does not match checkpoint model_config"
            )

    commit = _read_villa_commit(villa_source)
    if commit is not None and commit != PINNED_VILLA_COMMIT:
        raise ValueError(
            f"Villa source commit is {commit}, expected {PINNED_VILLA_COMMIT}"
        )
    _install_villa_import_shims(villa_source)
    module = importlib.import_module("vesuvius.models.build.build_network_from_config")
    network_class = module.NetworkFromConfig
    manager = SimpleNamespace(
        targets=config["targets"],
        train_patch_size=patch_shape,
        train_batch_size=int(config.get("train_batch_size", 1)),
        in_channels=int(config.get("in_channels", 1)),
        autoconfigure=bool(config.get("autoconfigure", False)),
        enable_deep_supervision=bool(config.get("enable_deep_supervision", False)),
        model_name=str(config.get("model_name", "surface-teacher")),
        model_config=config,
        spacing=(1.0, 1.0, 1.0),
    )
    network = network_class(manager)
    weights = payload.get("model")
    if not isinstance(weights, dict):
        raise TypeError("Villa teacher checkpoint has no model state")
    network.load_state_dict(_strip_wrapper_prefixes(weights), strict=True)
    adapter = _VillaSurfaceAdapter(network).to(device).eval()
    unique_parameters = sum(parameter.numel() for parameter in adapter.parameters())
    normalization_scheme = payload.get("normalization_scheme")
    intensity_properties = payload.get("intensity_properties") or {}
    if normalization_scheme != "zscore" or intensity_properties:
        raise ValueError(
            "official native teacher must use instance z-score without global properties"
        )
    return LoadedTeacher(
        model=adapter,
        kind="villa-native-2um-ps256",
        patch_shape_zyx=patch_shape,
        required_divisor=64,
        normalization="instance_zscore",
        default_threshold=OFFICIAL_SURFACE_TEACHER_THRESHOLD,
        default_mirror_tta=True,
        tta_average_logits=True,
        provenance={
            "architecture": "villa-NetworkFromConfig-resenc-unet-scse",
            "model_name": config.get("model_name"),
            "parameter_count": unique_parameters,
            "normalization_scheme": normalization_scheme,
            "intensity_properties": intensity_properties,
            "villa_source": str(villa_source),
            "villa_commit": commit or "unavailable",
            "model_config": config,
        },
    )


def _load_m7_teacher(checkpoint_path: Path, *, device: torch.device) -> LoadedTeacher:
    model = VoxelNNUNet(NNUNetConfig(preset="m7-resenc-l"))
    initialization = initialize_from_m7(model, checkpoint_path)
    model.to(device).eval()
    return LoadedTeacher(
        model=model,
        kind="m7-diagnostic",
        patch_shape_zyx=(192, 192, 192),
        required_divisor=32,
        normalization="m7_fixed",
        default_threshold=0.2,
        default_mirror_tta=False,
        tta_average_logits=False,
        provenance={
            "architecture": "m7-resenc-l",
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "initialization": initialization,
        },
    )


def load_teacher_checkpoint(
    checkpoint_path: str | Path,
    *,
    villa_source: str | Path,
    device: torch.device,
) -> LoadedTeacher:
    """Load the official native-fine teacher, with m7 retained for diagnostics."""

    source = Path(checkpoint_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict):
        raise TypeError(f"{source}: checkpoint must contain a mapping")
    config = payload.get("model_config")
    if isinstance(config, dict) and "targets" in config:
        payload["_checkpoint_path"] = str(source)
        loaded = _load_villa_teacher(
            payload,
            villa_source=Path(villa_source).expanduser().resolve(),
            device=device,
        )
        del payload
        return loaded
    del payload
    return _load_m7_teacher(source, device=device)
