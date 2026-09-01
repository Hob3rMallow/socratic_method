from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

# Exact released Scroll Prize surface M7 artifact used by this project.  A new
# cross-resolution experiment may resume its own optimizer state after a crash,
# but it may never use a previous student (or a look-alike checkpoint) as its
# learned initializer.
RELEASED_M7_CHECKPOINT_SHA256 = (
    "17465b77591b794638e671f1a9f79c4cf1e79821f302e6fc235e3725e5da7d7e"
)
FRESH_M7_INITIALIZATION_CONTRACT = "fresh-released-m7-sha256-strict-v1"


@dataclass(frozen=True)
class NNUNetConfig:
    preset: str = "m7-resenc-l"
    input_channels: int = 1
    num_classes: int = 2
    deep_supervision: bool = True

    def __post_init__(self) -> None:
        if self.preset not in {"m7-resenc-l", "tiny-test"}:
            raise ValueError("preset must be m7-resenc-l or tiny-test")
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if self.num_classes != 2:
            raise ValueError("the surface model requires background/surface classes")

    @property
    def required_divisor(self) -> int:
        return 32 if self.preset == "m7-resenc-l" else 4

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> NNUNetConfig:
        return cls(
            **{
                key: value[key]
                for key in (
                    "preset",
                    "input_channels",
                    "num_classes",
                    "deep_supervision",
                )
                if key in value
            }
        )


def _architecture_arguments(config: NNUNetConfig) -> dict[str, Any]:
    if config.preset == "m7-resenc-l":
        return {
            "n_stages": 6,
            "features_per_stage": (32, 64, 128, 256, 320, 320),
            "kernel_sizes": ((3, 3, 3),) * 6,
            "strides": (
                (1, 1, 1),
                (2, 2, 2),
                (2, 2, 2),
                (2, 2, 2),
                (2, 2, 2),
                (2, 2, 2),
            ),
            "n_blocks_per_stage": (1, 3, 4, 6, 6, 6),
            "n_conv_per_stage_decoder": (1, 1, 1, 1, 1),
        }
    return {
        "n_stages": 3,
        "features_per_stage": (8, 16, 32),
        "kernel_sizes": ((3, 3, 3),) * 3,
        "strides": ((1, 1, 1), (2, 2, 2), (2, 2, 2)),
        "n_blocks_per_stage": (1, 1, 1),
        "n_conv_per_stage_decoder": (1, 1),
    }


def build_network(config: NNUNetConfig) -> nn.Module:
    try:
        from dynamic_network_architectures.architectures.unet import (
            ResidualEncoderUNet,
        )
    except ImportError as error:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "voxel nnU-Net requires the baseline optional dependencies"
        ) from error
    return ResidualEncoderUNet(
        input_channels=config.input_channels,
        conv_op=nn.Conv3d,
        num_classes=config.num_classes,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1.0e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=config.deep_supervision,
        **_architecture_arguments(config),
    )


class VoxelNNUNet(nn.Module):
    """Ordinary 3-D nnU-Net mapping coarse CT voxels to surface logits."""

    def __init__(self, config: NNUNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or NNUNetConfig()
        self.network = build_network(self.config)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        if image.ndim != 5 or image.shape[1] != self.config.input_channels:
            raise ValueError(
                f"expected Bx{self.config.input_channels}xDxHxW, got {tuple(image.shape)}"
            )
        if any(size % self.config.required_divisor for size in image.shape[-3:]):
            raise ValueError(
                f"spatial sizes must be divisible by {self.config.required_divisor}"
            )
        result = self.network(image)
        outputs = list(result) if isinstance(result, (list, tuple)) else [result]
        if outputs[0].shape[-3:] != image.shape[-3:]:
            raise RuntimeError("nnU-Net did not return full-resolution logits first")
        return outputs

    def full_resolution_logits(self, image: torch.Tensor) -> torch.Tensor:
        return self(image)[0]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_released_m7_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """Fail closed unless *checkpoint_path* is the exact pinned M7 artifact."""

    source_path = Path(checkpoint_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"released M7 checkpoint is missing: {source_path}")
    checkpoint_sha256 = _sha256_file(source_path)
    if checkpoint_sha256 != RELEASED_M7_CHECKPOINT_SHA256:
        raise ValueError(
            "M7 initializer SHA-256 mismatch: "
            f"expected {RELEASED_M7_CHECKPOINT_SHA256}, got {checkpoint_sha256}; "
            "cross-resolution students must start from the exact released M7"
        )
    return {
        "contract": FRESH_M7_INITIALIZATION_CONTRACT,
        "checkpoint": str(source_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": source_path.stat().st_size,
    }


def initialize_from_m7(
    model: VoxelNNUNet,
    checkpoint_path: str | Path,
    *,
    verified_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly load the released m7 network, including its two-class heads."""

    if model.config.preset != "m7-resenc-l":
        raise ValueError("released m7 initialization requires preset m7-resenc-l")
    if model.config.input_channels != 1 or model.config.num_classes != 2:
        raise ValueError("released m7 is exactly one CT channel and two classes")
    source_path = Path(checkpoint_path).expanduser().resolve()
    identity = verified_identity or verify_released_m7_checkpoint(source_path)
    if (
        identity.get("contract") != FRESH_M7_INITIALIZATION_CONTRACT
        or identity.get("checkpoint") != str(source_path)
        or identity.get("checkpoint_sha256") != RELEASED_M7_CHECKPOINT_SHA256
        or identity.get("checkpoint_bytes") != source_path.stat().st_size
    ):
        raise ValueError("released M7 verification identity is invalid or stale")
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    weights = payload.get("network_weights")
    if not isinstance(weights, dict):
        raise TypeError(f"{source_path}: missing network_weights")
    model.network.load_state_dict(weights, strict=True)
    return {
        **identity,
        "source_epoch": payload.get("current_epoch"),
        "source_trainer": payload.get("trainer_name"),
        "state_tensor_count": len(weights),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "strict": True,
        "copied_segmentation_heads": len(model.network.decoder.seg_layers),
    }
