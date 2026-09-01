from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .schema import (
    SchemaError,
    canonical_scroll_id,
    iter_jsonl,
    normalized_split,
)

PATCH_KINDS = ("teacher", "student")
ROT90_MODES = ("none", "z-only", "all")


@dataclass(frozen=True)
class PatchRow:
    patch_id: str
    path: Path
    record_id: str
    scroll_id: str
    split: str
    kind: str
    origin_zyx: tuple[int, int, int]
    shape_zyx: tuple[int, int, int]
    policy_profile: str
    pitch_um: float = 0.0
    sampling_stratum: str = "uniform"


def load_patch_rows(path: str | Path) -> list[PatchRow]:
    manifest = Path(path).resolve()
    rows: list[PatchRow] = []
    for value in iter_jsonl(manifest):
        required_fields = {
            "patch_id",
            "path",
            "record_id",
            "scroll_id",
            "split",
            "kind",
            "origin_zyx",
            "shape_zyx",
            "policy_profile",
        }
        missing_fields = required_fields.difference(value)
        if missing_fields:
            raise SchemaError(
                f"{manifest}: patch row is missing {sorted(missing_fields)}"
            )
        try:
            version = int(value.get("schema_version", 1))
        except (TypeError, ValueError) as error:
            raise SchemaError("patch schema_version must be an integer") from error
        if version != 2:
            raise SchemaError(
                f"unsupported patch schema_version {version}; the voxel-domain "
                "rewrite reads only schema 2 corpora"
            )
        kind = str(value["kind"]).strip().lower()
        if kind not in PATCH_KINDS:
            raise SchemaError(f"invalid patch kind {kind!r}")
        relative = Path(str(value["path"]))
        patch_path = relative if relative.is_absolute() else manifest.parent / relative
        try:
            origin = tuple(int(item) for item in value["origin_zyx"])
            shape = tuple(int(item) for item in value["shape_zyx"])
        except (TypeError, ValueError) as error:
            raise SchemaError(
                "origin_zyx and shape_zyx must be integer arrays"
            ) from error
        if len(origin) != 3:
            raise SchemaError("origin_zyx must contain three integers")
        if len(shape) != 3 or any(item <= 0 for item in shape):
            raise SchemaError("shape_zyx must contain three positive integers")
        split = normalized_split(str(value["split"]).lower())
        if split not in {"train", "val", "test"}:
            raise SchemaError(f"invalid patch split {split!r}")
        policy_profile = str(value["policy_profile"]).strip().lower()
        if policy_profile not in {"prize-safe", "research"}:
            raise SchemaError(f"invalid policy_profile {policy_profile!r}")
        rows.append(
            PatchRow(
                patch_id=str(value["patch_id"]),
                path=patch_path,
                record_id=str(value["record_id"]),
                scroll_id=canonical_scroll_id(str(value["scroll_id"])),
                split=split,
                kind=kind,
                origin_zyx=origin,
                shape_zyx=shape,
                policy_profile=policy_profile,
                pitch_um=float(value.get("pitch_um", 0.0)),
                sampling_stratum=str(value.get("sampling_stratum", "uniform")),
            )
        )
    if not rows:
        raise SchemaError(f"{manifest}: no patch rows")
    ids = [row.patch_id for row in rows]
    if len(ids) != len(set(ids)):
        raise SchemaError(f"{manifest}: patch_id values are not unique")
    return rows


def validate_patch_splits(rows: list[PatchRow]) -> None:
    splits: dict[str, set[str]] = {}
    for row in rows:
        splits.setdefault(row.scroll_id, set()).add(row.split)
    leakage = {
        scroll: sorted(values) for scroll, values in splits.items() if len(values) > 1
    }
    if leakage:
        detail = ", ".join(
            f"{scroll}={value}" for scroll, value in sorted(leakage.items())
        )
        raise SchemaError(f"scroll-level patch leakage: {detail}")


M7_CT_LOWER = 0.0
M7_CT_UPPER = 212.0
M7_CT_MEAN = 87.54424285888672
M7_CT_STD = 47.74376678466797


def normalize_ct_m7(image: np.ndarray) -> np.ndarray:
    """Apply the released m7 nnU-Net CT normalization exactly."""

    value = np.nan_to_num(
        image.astype(np.float32),
        nan=M7_CT_LOWER,
        posinf=M7_CT_UPPER,
        neginf=M7_CT_LOWER,
    )
    value = np.clip(value, M7_CT_LOWER, M7_CT_UPPER)
    return (value - M7_CT_MEAN) / M7_CT_STD


_OPTIONAL_PAIRS = (
    ("distill_u8", "distill_valid_u8"),
    ("rehearsal_u8", "rehearsal_valid_u8"),
)

_SPATIAL_KEYS = (
    "input",
    "label",
    "distill",
    "distill_valid",
    "rehearsal",
    "rehearsal_valid",
)


class PatchDataset(Dataset[dict[str, Any]]):
    """Schema-2 voxel patches for teacher and student training.

    Every patch carries ``image`` and ``label_u8`` ({0,1,2}); student patches
    optionally add soft distillation targets and rehearsal targets, each with
    an explicit validity mask so the loss partition stays exclusive.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        kind: str,
        augment: bool = False,
        rot90_mode: str = "none",
        in_channels: int = 1,
    ) -> None:
        if kind not in PATCH_KINDS:
            raise SchemaError(f"invalid dataset kind {kind!r}")
        if rot90_mode not in ROT90_MODES:
            raise SchemaError(f"invalid rot90_mode {rot90_mode!r}")
        if in_channels not in {1, 2}:
            raise SchemaError("in_channels must be 1 or 2")
        all_rows = load_patch_rows(manifest_path)
        validate_patch_splits(all_rows)
        kinds = {row.kind for row in all_rows}
        if kinds != {kind}:
            raise SchemaError(
                f"patch manifest mixes kinds {sorted(kinds)}; expected only {kind!r}"
            )
        split = normalized_split(split.lower())
        self.rows = [row for row in all_rows if row.split == split]
        if not self.rows:
            raise SchemaError(f"patch manifest has no {split!r} rows")
        self.kind = kind
        self.augment = augment
        self.rot90_mode = rot90_mode
        self.in_channels = in_channels

    def __len__(self) -> int:
        return len(self.rows)

    def _decode(self, path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as archive:
            required = {"image", "label_u8"}
            missing = required.difference(archive.files)
            if missing:
                raise SchemaError(f"{path}: missing arrays {sorted(missing)}")
            arrays = {name: np.asarray(archive[name]) for name in required}
            for value_name, valid_name in _OPTIONAL_PAIRS:
                present = {value_name, valid_name}.intersection(archive.files)
                if present and len(present) != 2:
                    raise SchemaError(
                        f"{path}: {value_name} and {valid_name} must appear together"
                    )
                if present:
                    arrays[value_name] = np.asarray(archive[value_name])
                    arrays[valid_name] = np.asarray(archive[valid_name])
            if self.in_channels == 2:
                if "baseline_u8" not in archive.files:
                    raise SchemaError(
                        f"{path}: a 2-channel model requires baseline_u8"
                    )
                arrays["baseline_u8"] = np.asarray(archive["baseline_u8"])
        image_shape = tuple(arrays["image"].shape)
        if len(image_shape) != 3:
            raise SchemaError(f"{path}: image must be 3-D, got {image_shape}")
        for name, value in arrays.items():
            if name != "image" and tuple(value.shape) != image_shape:
                raise SchemaError(
                    f"{path}: {name} shape {value.shape} != {image_shape}"
                )
        if int(arrays["label_u8"].max(initial=0)) > 2:
            raise SchemaError(f"{path}: label_u8 must contain only 0, 1, 2")
        return arrays

    @staticmethod
    def _flip(sample: dict[str, Any], spatial_axis: int) -> None:
        tensor_axis = spatial_axis + 1
        for name in _SPATIAL_KEYS:
            sample[name] = torch.flip(sample[name], dims=(tensor_axis,))

    @staticmethod
    def _rot90(sample: dict[str, Any], k: int, plane: tuple[int, int]) -> None:
        if k % 4 == 0:
            return
        for name in _SPATIAL_KEYS:
            sample[name] = torch.rot90(sample[name], k=k, dims=plane)

    @staticmethod
    def _augment_intensity(image: torch.Tensor) -> torch.Tensor:
        gain = float(torch.empty(()).uniform_(0.8, 1.2))
        bias = float(torch.empty(()).uniform_(-0.2, 0.2))
        image = image * gain + bias

        if bool(torch.rand(()) < 0.35):
            gamma = float(torch.empty(()).uniform_(0.7, 1.5))
            lower = image.amin()
            upper = image.amax()
            span = upper - lower
            if float(span) > 1.0e-6:
                image = ((image - lower) / span).clamp(0.0, 1.0).pow(
                    gamma
                ) * span + lower

        if bool(torch.rand(()) < 0.25):
            image = F.avg_pool3d(image[None, None], kernel_size=3, stride=1, padding=1)[
                0, 0
            ]

        if bool(torch.rand(()) < 0.25):
            scale = float(torch.empty(()).uniform_(0.5, 0.9))
            source_shape = image.shape
            reduced_shape = tuple(max(2, round(size * scale)) for size in source_shape)
            reduced = F.interpolate(
                image[None, None],
                size=reduced_shape,
                mode="trilinear",
                align_corners=False,
            )
            image = F.interpolate(
                reduced,
                size=source_shape,
                mode="trilinear",
                align_corners=False,
            )[0, 0]

        if bool(torch.rand(()) < 0.35):
            noise_sigma = float(torch.empty(()).uniform_(0.0, 0.08))
            image = image + torch.randn_like(image) * noise_sigma
        return image.clamp(-3.0, 4.0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        arrays = self._decode(row.path)
        if tuple(arrays["image"].shape) != row.shape_zyx:
            raise SchemaError(
                f"{row.path}: data shape does not match manifest {row.shape_zyx}"
            )
        image = normalize_ct_m7(arrays["image"])
        channels = [image]
        if self.in_channels == 2:
            channels.append(
                (arrays["baseline_u8"] != 0).astype(np.float32)
            )
        shape = arrays["image"].shape
        zeros = np.zeros(shape, dtype=np.float32)

        def optional(name: str, scale: float) -> np.ndarray:
            if name in arrays:
                return arrays[name].astype(np.float32) / scale
            return zeros

        sample: dict[str, Any] = {
            "input": torch.from_numpy(np.stack(channels, axis=0)).float(),
            "label": torch.from_numpy(arrays["label_u8"][None].copy()),
            "distill": torch.from_numpy(optional("distill_u8", 255.0)[None]).float(),
            "distill_valid": torch.from_numpy(
                np.clip(optional("distill_valid_u8", 1.0), 0.0, 1.0)[None]
            ).float(),
            "rehearsal": torch.from_numpy(
                optional("rehearsal_u8", 255.0)[None]
            ).float(),
            "rehearsal_valid": torch.from_numpy(
                np.clip(optional("rehearsal_valid_u8", 1.0), 0.0, 1.0)[None]
            ).float(),
            "patch_id": row.patch_id,
            "scroll_id": row.scroll_id,
            "sampling_stratum": row.sampling_stratum,
            "kind": row.kind,
        }
        if self.augment:
            for axis in range(3):
                if bool(torch.rand(()) < 0.5):
                    self._flip(sample, axis)
            if self.rot90_mode != "none":
                planes = (
                    ((2, 3),)
                    if self.rot90_mode == "z-only"
                    else ((2, 3), (1, 2), (1, 3))
                )
                plane = planes[int(torch.randint(len(planes), ()))]
                sizes = sample["input"].shape
                k = int(torch.randint(4, ()))
                # A quarter-turn in an anisotropic plane would change the
                # spatial shape; restrict to half-turns there.
                if sizes[plane[0]] != sizes[plane[1]] and k % 2 == 1:
                    k = (k + 1) % 4
                self._rot90(sample, k, plane)
            sample["input"][0] = self._augment_intensity(sample["input"][0])
        return sample
