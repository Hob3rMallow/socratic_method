from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


class ModelError(ValueError):
    """Raised when a model configuration or checkpoint violates the contract."""


@dataclass(frozen=True)
class SurfaceModelConfig:
    """Configuration for the single-head surface network.

    The trunk is always the released m7 ResidualEncoderUNet (ResEnc-L). The
    only degrees of freedom are the input channel count (1 = raw CT drop-in,
    2 = raw CT + baseline prediction refiner) and, implicitly, the pitch the
    weights were trained at, which lives in training provenance rather than
    here: the network itself is pitch-agnostic.
    """

    in_channels: int = 1
    architecture: str = "m7-resenc-l"
    input_normalization: str = "m7-ct"

    def __post_init__(self) -> None:
        if self.in_channels not in {1, 2}:
            raise ModelError("in_channels must be 1 or 2")
        if self.architecture != "m7-resenc-l":
            raise ModelError("architecture must be m7-resenc-l")
        if self.input_normalization != "m7-ct":
            raise ModelError("input_normalization must be m7-ct")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SurfaceModelConfig:
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value[key] for key in allowed if key in value})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_m7_trunk(input_channels: int, *, num_classes: int = 1) -> nn.Module:
    """Instantiate the exact released m7 ResEnc-L architecture."""

    try:
        from dynamic_network_architectures.architectures.unet import (
            ResidualEncoderUNet,
        )
    except ImportError as error:  # pragma: no cover - optional m7 dependency
        raise ModelError(
            "m7-resenc-l requires the 'baseline' optional dependencies"
        ) from error
    return ResidualEncoderUNet(
        input_channels=input_channels,
        n_stages=6,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        conv_op=nn.Conv3d,
        kernel_sizes=((3, 3, 3),) * 6,
        strides=(
            (1, 1, 1),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
        ),
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
        num_classes=num_classes,
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1.0e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
    )


def _is_stem_conv_weight(name: str) -> bool:
    """The first stem convolution's weight appears under two aliased keys in
    dynamic-network-architectures state dicts; both must be widened."""

    return name.endswith("stem.convs.0.conv.weight") or name.endswith(
        "stem.convs.0.all_modules.0.weight"
    )


class SurfaceNet(nn.Module):
    """The m7 trunk with a single foreground-surface logit head.

    Used at every pitch in the pipeline: the fine-resolution teachers and the
    coarse student are the same class with different weights and provenance.
    """

    def __init__(self, config: SurfaceModelConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else SurfaceModelConfig()
        self.task_network = build_m7_trunk(self.config.in_channels, num_classes=1)

    @property
    def required_divisor(self) -> int:
        return 32

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5 or value.shape[1] != self.config.in_channels:
            raise ModelError(
                f"expected Bx{self.config.in_channels}xDxHxW input, "
                f"got {tuple(value.shape)}"
            )
        if any(size % self.required_divisor for size in value.shape[-3:]):
            raise ModelError(
                f"spatial dimensions must be divisible by {self.required_divisor}"
            )
        # The 6-stage encoder reduces each axis 32x; InstanceNorm needs more
        # than one element at the bottleneck, so 64 is the hard minimum.
        if any(size < 2 * self.required_divisor for size in value.shape[-3:]):
            raise ModelError(
                f"spatial dimensions must be at least {2 * self.required_divisor}"
            )
        return self.task_network(value)


def initialize_from_m7_checkpoint(
    model: SurfaceNet,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Load the released m7 trunk and convert its softmax head to one logit.

    The released checkpoint stores a 1-channel stem and 2-class softmax
    segmentation layers. The stem is copied exactly (or widened by zero
    padding for a 2-channel model) and every ``decoder.seg_layers`` pair is
    collapsed to a single foreground logit via ``w[1]-w[0]``, ``b[1]-b[0]``.
    """

    source_path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    source = payload.get("network_weights")
    if not isinstance(source, dict):
        raise ModelError(f"{source_path}: missing nnU-Net network_weights")

    target = model.task_network.state_dict()
    copied_keys: list[str] = []
    adapted_stem_keys: list[str] = []
    skipped_keys: list[str] = []
    for name, source_value in source.items():
        if name not in target:
            skipped_keys.append(name)
            continue
        target_value = target[name]
        if source_value.shape == target_value.shape:
            target[name] = source_value
            copied_keys.append(name)
            continue
        if (
            _is_stem_conv_weight(name)
            and source_value.ndim == 5
            and source_value.shape[1] == 1
            and target_value.shape[1] == model.config.in_channels
            and source_value.shape[0] == target_value.shape[0]
            and source_value.shape[2:] == target_value.shape[2:]
        ):
            adapted = torch.zeros_like(target_value)
            adapted[:, 0:1] = source_value
            target[name] = adapted
            adapted_stem_keys.append(name)
            continue
        skipped_keys.append(name)
    model.task_network.load_state_dict(target, strict=True)

    with torch.no_grad():
        for index, layer in enumerate(model.task_network.decoder.seg_layers):
            weight_name = f"decoder.seg_layers.{index}.weight"
            bias_name = f"decoder.seg_layers.{index}.bias"
            source_weight = source.get(weight_name)
            source_bias = source.get(bias_name)
            if (
                source_weight is None
                or source_bias is None
                or source_weight.shape[0] != 2
                or source_bias.shape != (2,)
            ):
                raise ModelError(
                    f"{source_path}: incompatible m7 segmentation head {index}"
                )
            layer.weight[0].copy_(source_weight[1] - source_weight[0])
            layer.bias[0].copy_(source_bias[1] - source_bias[0])

    return {
        "kind": "m7-nnunet",
        "checkpoint": str(source_path),
        "source_epoch": payload.get("current_epoch"),
        "source_trainer": payload.get("trainer_name"),
        "copied_state_keys": len(copied_keys),
        "adapted_stem_keys": adapted_stem_keys,
        "skipped_state_keys": skipped_keys,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def initialize_from_surface_checkpoint(
    model: SurfaceNet,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Initialize from another SurfaceNet checkpoint (e.g. teacher-1p1 from
    teacher-2p4). Identical configurations load strictly; a 1-channel source
    may initialize a 2-channel model by zero-padding the stem.
    """

    source_path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", 0)) != 2:
        raise ModelError(f"{source_path}: expected a schema-2 surface checkpoint")
    source_config = SurfaceModelConfig.from_dict(payload["model_config"])
    source = payload["model_state"]
    if source_config == model.config:
        model.load_state_dict(source, strict=True)
        copied = len(source)
        adapted: list[str] = []
    else:
        if (source_config.in_channels, model.config.in_channels) != (1, 2):
            raise ModelError(
                f"{source_path}: cannot initialize in_channels="
                f"{model.config.in_channels} from in_channels="
                f"{source_config.in_channels}"
            )
        target = model.state_dict()
        copied = 0
        adapted = []
        for name, source_value in source.items():
            if name not in target:
                continue
            target_value = target[name]
            if source_value.shape == target_value.shape:
                target[name] = source_value
                copied += 1
            elif (
                _is_stem_conv_weight(name)
                and source_value.ndim == 5
                and source_value.shape[1] == 1
                and target_value.shape[1] == 2
            ):
                widened = torch.zeros_like(target_value)
                widened[:, 0:1] = source_value
                target[name] = widened
                adapted.append(name)
            else:
                raise ModelError(
                    f"{source_path}: unexpected shape mismatch for {name}"
                )
        model.load_state_dict(target, strict=True)
    return {
        "kind": "surface-checkpoint",
        "checkpoint": str(source_path),
        "source_epoch": payload.get("epoch"),
        "source_profile": payload.get("train_options", {}).get("profile"),
        "copied_state_keys": copied,
        "adapted_stem_keys": adapted,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def load_surface_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[SurfaceNet, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if int(payload.get("schema_version", 0)) != 2:
        raise ModelError(f"{path}: unsupported checkpoint schema")
    config = SurfaceModelConfig.from_dict(payload["model_config"])
    model = SurfaceNet(config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return model, payload
