from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .registration_evidence import load_registration_evidence

PAIR_SCHEMA = "crossres-voxel-pair-v1"


class VoxelSchemaError(ValueError):
    """A dense voxel manifest violates the img2img data contract."""


def _required(value: dict[str, Any], key: str, context: str) -> Any:
    if key not in value:
        raise VoxelSchemaError(f"{context} is missing required field {key!r}")
    return value[key]


def _resolve_path(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def resolve_volume_spec(value: Any, base: Path) -> str:
    text = str(value).strip()
    if not text:
        raise VoxelSchemaError("volume path cannot be empty")
    if "://" in text:
        raise VoxelSchemaError(
            "training volumes must be local or mounted; remote URLs are not reproducible"
        )
    path_text, separator, key = text.rpartition("::")
    if not separator:
        path_text, key = text, ""
    path = _resolve_path(path_text, base)
    return f"{path}::{key}" if separator else str(path)


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise VoxelSchemaError(f"{name} must be numeric") from error
    if not math.isfinite(result) or result <= 0:
        raise VoxelSchemaError(f"{name} must be finite and positive")
    return result


def _affine(value: Any) -> tuple[tuple[float, float, float, float], ...]:
    if not isinstance(value, list) or len(value) != 3:
        raise VoxelSchemaError("fine.to_coarse_affine_xyz must be a 3x4 array")
    rows: list[tuple[float, float, float, float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise VoxelSchemaError("fine.to_coarse_affine_xyz must be a 3x4 array")
        try:
            converted = tuple(float(item) for item in row)
        except (TypeError, ValueError) as error:
            raise VoxelSchemaError(
                "fine.to_coarse_affine_xyz must contain numbers"
            ) from error
        if not all(math.isfinite(item) for item in converted):
            raise VoxelSchemaError(
                "fine.to_coarse_affine_xyz must contain finite numbers"
            )
        rows.append(converted)  # type: ignore[arg-type]
    linear = np.asarray(rows, dtype=np.float64)[:, :3]
    if abs(float(np.linalg.det(linear))) < 1.0e-12:
        raise VoxelSchemaError("fine.to_coarse_affine_xyz is singular")
    return tuple(rows)


@dataclass(frozen=True)
class ChunkSupportSpec:
    """Which fine Zarr chunks are known rather than implicit missing data."""

    kind: str = "all"
    inventory: Path | None = None

    @classmethod
    def from_dict(cls, value: Any, *, base: Path) -> ChunkSupportSpec:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise VoxelSchemaError("target.support must be an object")
        kind = str(value.get("kind", "all")).strip().lower()
        if kind not in {"all", "present-chunks"}:
            raise VoxelSchemaError(
                "target.support.kind must be 'all' or 'present-chunks'"
            )
        inventory = value.get("inventory")
        if kind == "present-chunks" and not inventory:
            raise VoxelSchemaError(
                "present-chunks support requires target.support.inventory"
            )
        return cls(
            kind=kind,
            inventory=_resolve_path(inventory, base) if inventory else None,
        )


@dataclass(frozen=True)
class DenseFieldSpec:
    volume: str
    encoding: str
    positive_labels: tuple[int, ...] = (1,)
    ignore_labels: tuple[int, ...] = ()
    probability_scale: float = 1.0
    threshold: float = 0.5
    support: ChunkSupportSpec = ChunkSupportSpec()

    @classmethod
    def from_dict(cls, value: Any, *, context: str, base: Path) -> DenseFieldSpec:
        if not isinstance(value, dict):
            raise VoxelSchemaError(f"{context} must be an object, not a path string")
        encoding = str(_required(value, "encoding", context)).strip().lower()
        if encoding not in {"labels", "probability"}:
            raise VoxelSchemaError(
                f"{context}.encoding must be 'labels' or 'probability'"
            )
        raw_labels = value.get("positive_labels", [1])
        if not isinstance(raw_labels, list) or not raw_labels:
            raise VoxelSchemaError(f"{context}.positive_labels must be non-empty")
        try:
            labels = tuple(int(item) for item in raw_labels)
        except (TypeError, ValueError) as error:
            raise VoxelSchemaError(
                f"{context}.positive_labels must contain integers"
            ) from error
        raw_ignore_labels = value.get("ignore_labels", [])
        if not isinstance(raw_ignore_labels, list):
            raise VoxelSchemaError(f"{context}.ignore_labels must be an array")
        try:
            ignore_labels = tuple(int(item) for item in raw_ignore_labels)
        except (TypeError, ValueError) as error:
            raise VoxelSchemaError(
                f"{context}.ignore_labels must contain integers"
            ) from error
        if encoding == "probability" and ignore_labels:
            raise VoxelSchemaError(
                f"{context}.ignore_labels is only valid for label arrays"
            )
        overlap = set(labels) & set(ignore_labels)
        if overlap:
            raise VoxelSchemaError(
                f"{context} labels cannot be both positive and ignored: "
                f"{sorted(overlap)}"
            )
        scale = _positive_float(
            value.get("probability_scale", 1.0),
            f"{context}.probability_scale",
        )
        threshold = float(value.get("threshold", 0.5))
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise VoxelSchemaError(f"{context}.threshold must be in [0, 1]")
        return cls(
            volume=resolve_volume_spec(_required(value, "volume", context), base),
            encoding=encoding,
            positive_labels=labels,
            ignore_labels=ignore_labels,
            probability_scale=scale,
            threshold=threshold,
            support=ChunkSupportSpec.from_dict(value.get("support"), base=base),
        )


@dataclass(frozen=True)
class CoarseScanSpec:
    scan_id: str
    voxel_um: float
    image: str
    baseline: DenseFieldSpec | None = None

    @classmethod
    def from_dict(cls, value: Any, *, base: Path) -> CoarseScanSpec:
        if not isinstance(value, dict):
            raise VoxelSchemaError("coarse must be an object")
        scan_id = str(_required(value, "scan_id", "coarse")).strip()
        if not scan_id:
            raise VoxelSchemaError("coarse.scan_id cannot be empty")
        baseline = value.get("baseline")
        return cls(
            scan_id=scan_id,
            voxel_um=_positive_float(
                _required(value, "voxel_um", "coarse"), "coarse.voxel_um"
            ),
            image=resolve_volume_spec(_required(value, "image", "coarse"), base),
            baseline=(
                DenseFieldSpec.from_dict(baseline, context="coarse.baseline", base=base)
                if baseline is not None
                else None
            ),
        )


@dataclass(frozen=True)
class FineScanSpec:
    scan_id: str
    voxel_um: float
    target: DenseFieldSpec
    to_coarse_affine_xyz: tuple[tuple[float, float, float, float], ...]

    @classmethod
    def from_dict(cls, value: Any, *, base: Path) -> FineScanSpec:
        if not isinstance(value, dict):
            raise VoxelSchemaError("fine must be an object")
        scan_id = str(_required(value, "scan_id", "fine")).strip()
        if not scan_id:
            raise VoxelSchemaError("fine.scan_id cannot be empty")
        return cls(
            scan_id=scan_id,
            voxel_um=_positive_float(
                _required(value, "voxel_um", "fine"), "fine.voxel_um"
            ),
            target=DenseFieldSpec.from_dict(
                _required(value, "target", "fine"), context="fine.target", base=base
            ),
            to_coarse_affine_xyz=_affine(
                _required(value, "to_coarse_affine_xyz", "fine")
            ),
        )


def _validate_registration_provenance(
    value: Any,
    *,
    base: Path,
    scroll_id: str,
    coarse: CoarseScanSpec,
    fine: FineScanSpec,
) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise VoxelSchemaError("registration_evidence must be an object")
    kind = str(value.get("kind", "")).strip()
    if kind == "official-metadata-transform":
        return
    if kind != "pinned-ct-pyramid-affine":
        raise VoxelSchemaError(f"unsupported registration evidence kind {kind!r}")
    fine_source = value.get("fine_source")
    coarse_source = value.get("coarse_source")
    if not isinstance(fine_source, dict) or not isinstance(coarse_source, dict):
        raise VoxelSchemaError(
            "pinned registration evidence needs fine_source and coarse_source"
        )
    evidence_path = _resolve_path(
        _required(value, "evidence", "registration_evidence"), base
    )
    try:
        payload, affine, evidence_sha256 = load_registration_evidence(
            evidence_path,
            sample_id=scroll_id,
            fine_volume_id=str(_required(fine_source, "volume_id", "fine_source")),
            coarse_volume_id=str(
                _required(coarse_source, "volume_id", "coarse_source")
            ),
            fine_voxel_um=fine.voxel_um,
            coarse_voxel_um=coarse.voxel_um,
        )
    except (OSError, TypeError, ValueError) as error:
        raise VoxelSchemaError(
            f"invalid pinned registration evidence {evidence_path}: {error}"
        ) from error
    if str(value.get("evidence_sha256", "")) != evidence_sha256:
        raise VoxelSchemaError("registration evidence SHA-256 changed")
    for name, payload_name in (
        ("method", "method"),
        ("fit", "fit"),
        ("held_out", "held_out"),
        ("fine_source", "fine"),
        ("coarse_source", "coarse"),
    ):
        if value.get(name) != payload[payload_name]:
            raise VoxelSchemaError(
                f"registration evidence snapshot changed for {name}"
            )
    record_affine = np.asarray(fine.to_coarse_affine_xyz, dtype=np.float64)
    if not np.allclose(record_affine, affine, rtol=0.0, atol=1.0e-12):
        raise VoxelSchemaError(
            "pair affine does not match pinned registration evidence"
        )


@dataclass(frozen=True)
class VoxelPairRecord:
    record_id: str
    scroll_id: str
    split: str
    coarse: CoarseScanSpec
    fine: FineScanSpec
    patch_count: int | None = None
    supervision_source: str = "unspecified"

    @classmethod
    def from_dict(cls, value: Any, *, base: Path) -> VoxelPairRecord:
        if not isinstance(value, dict):
            raise VoxelSchemaError("pair record must be an object")
        if value.get("schema") != PAIR_SCHEMA:
            raise VoxelSchemaError(
                f"pair schema must be {PAIR_SCHEMA!r}; legacy point manifests are invalid"
            )
        if int(value.get("schema_version", 1)) != 1:
            raise VoxelSchemaError("unsupported voxel pair schema_version")
        record_id = str(_required(value, "record_id", "record")).strip()
        scroll_id = str(_required(value, "scroll_id", "record")).strip()
        split = str(_required(value, "split", "record")).strip().lower()
        split = {"development_holdout": "val", "sealed_prize_target": "test"}.get(
            split, split
        )
        if not record_id or not scroll_id:
            raise VoxelSchemaError("record_id and scroll_id cannot be empty")
        if split not in {"train", "val", "test"}:
            raise VoxelSchemaError("split must be train, val, or test")
        if "surfaces" in value:
            raise VoxelSchemaError(
                "voxel pair records cannot contain TIFXYZ/point surfaces"
            )
        raw_patch_count = value.get("patch_count")
        patch_count = int(raw_patch_count) if raw_patch_count is not None else None
        if patch_count is not None and patch_count <= 0:
            raise VoxelSchemaError("patch_count must be positive when provided")
        supervision_source = str(
            value.get("supervision_source", "unspecified")
        ).strip()
        if not supervision_source:
            raise VoxelSchemaError("supervision_source cannot be empty")
        coarse = CoarseScanSpec.from_dict(
            _required(value, "coarse", "record"), base=base
        )
        fine = FineScanSpec.from_dict(_required(value, "fine", "record"), base=base)
        _validate_registration_provenance(
            value.get("registration_evidence"),
            base=base,
            scroll_id=scroll_id,
            coarse=coarse,
            fine=fine,
        )
        return cls(
            record_id=record_id,
            scroll_id=scroll_id,
            split=split,
            coarse=coarse,
            fine=fine,
            patch_count=patch_count,
            supervision_source=supervision_source,
        )

    @property
    def expected_linear_scale(self) -> float:
        return self.fine.voxel_um / self.coarse.voxel_um

    @property
    def measured_linear_scales(self) -> tuple[float, float, float]:
        linear = np.asarray(self.fine.to_coarse_affine_xyz, dtype=np.float64)[:, :3]
        return tuple(float(item) for item in np.linalg.svd(linear, compute_uv=False))


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as error:
                raise VoxelSchemaError(
                    f"{source}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise VoxelSchemaError(f"{source}:{line_number}: expected an object")
            yield value


def load_pair_manifest(path: str | Path) -> list[VoxelPairRecord]:
    source = Path(path).expanduser().resolve()
    records = [
        VoxelPairRecord.from_dict(value, base=source.parent)
        for value in iter_jsonl(source)
    ]
    if not records:
        raise VoxelSchemaError(f"{source}: no voxel pair records")
    ids = [record.record_id for record in records]
    if len(ids) != len(set(ids)):
        raise VoxelSchemaError(f"{source}: duplicate record_id values")
    return records
