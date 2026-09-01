from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SchemaError(ValueError):
    """Raised when a data or patch manifest violates the public contract."""


def canonical_scroll_id(value: str) -> str:
    compact = value.strip().replace("-", "").replace("_", "")
    aliases = {
        "PHERC139": "PHerc0139",
        "PHERC0139": "PHerc0139",
        "PHERC332": "PHerc0332",
        "PHERC0332": "PHerc0332",
        "PHERC814": "PHerc0814",
        "PHERC0814": "PHerc0814",
        "PHERC846A": "PHerc0846A",
        "PHERC0846A": "PHerc0846A",
        "PHERC1203": "PHerc1203",
        "PHERC1451": "PHerc1451",
        "PHERC1667": "PHerc1667",
        "PHERCPARIS4": "PHercParis4",
    }
    key = compact.upper()
    if key in aliases:
        return aliases[key]
    if not value.strip():
        raise SchemaError("scroll_id cannot be empty")
    return value.strip()


def _required(mapping: dict[str, Any], name: str, context: str) -> Any:
    if name not in mapping:
        raise SchemaError(f"{context} is missing required field {name!r}")
    return mapping[name]


def _positive_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise SchemaError(f"{name} must be a number") from error
    if not number > 0.0:
        raise SchemaError(f"{name} must be > 0")
    return number


def _resolve_path(value: Any, base_directory: Path | None) -> Path:
    path = Path(str(value)).expanduser()
    if base_directory is not None and not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def _resolve_volume_spec(value: Any, base_directory: Path | None) -> str:
    spec = str(value)
    if "://" in spec:
        return spec
    path_text, separator, key = spec.rpartition("::")
    if not separator:
        path_text = spec
        key = ""
    resolved = _resolve_path(path_text, base_directory)
    return f"{resolved}::{key}" if separator else str(resolved)


def _affine_3x4(value: Any, name: str) -> tuple[tuple[float, ...], ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise SchemaError(f"{name} must be a 3x4 numeric array")
    rows: list[tuple[float, ...]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise SchemaError(f"{name} must be a 3x4 numeric array")
        try:
            converted = tuple(float(item) for item in row)
        except (TypeError, ValueError) as error:
            raise SchemaError(f"{name} must be a 3x4 numeric array") from error
        if not all(math.isfinite(item) for item in converted):
            raise SchemaError(f"{name} must contain only finite values")
        rows.append(converted)
    return tuple(rows)


def normalized_split(value: str) -> str:
    return {
        "development_holdout": "val",
        "sealed_prize_target": "test",
    }.get(value, value)


@dataclass(frozen=True)
class ScanSpec:
    scan_id: str
    voxel_um: float
    volume: str | None = None
    baseline: str | None = None
    teacher: str | None = None
    to_coarse_affine_xyz: tuple[tuple[float, ...], ...] | None = None

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        *,
        context: str,
        base_directory: Path | None = None,
    ) -> ScanSpec:
        if not isinstance(value, dict):
            raise SchemaError(f"{context} must be an object")
        scan_id = str(_required(value, "scan_id", context)).strip()
        if not scan_id:
            raise SchemaError(f"{context}.scan_id cannot be empty")
        return cls(
            scan_id=scan_id,
            voxel_um=_positive_float(
                _required(value, "voxel_um", context), f"{context}.voxel_um"
            ),
            volume=(
                _resolve_volume_spec(value["volume"], base_directory)
                if value.get("volume")
                else None
            ),
            baseline=(
                _resolve_volume_spec(value["baseline"], base_directory)
                if value.get("baseline")
                else None
            ),
            teacher=(
                _resolve_volume_spec(value["teacher"], base_directory)
                if value.get("teacher")
                else None
            ),
            to_coarse_affine_xyz=_affine_3x4(
                value.get("to_coarse_affine_xyz"),
                f"{context}.to_coarse_affine_xyz",
            ),
        )


@dataclass(frozen=True)
class SurfacePair:
    surface_id: str
    sheet_id: str
    coarse_tifxyz: Path
    fine_tifxyz: Path
    supervision_source: str = "manual-tifxyz"
    supervision_weight: float = 1.0

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        *,
        context: str,
        base_directory: Path | None = None,
    ) -> SurfacePair:
        if not isinstance(value, dict):
            raise SchemaError(f"{context} must be an object")
        surface_id = str(_required(value, "surface_id", context)).strip()
        if not surface_id:
            raise SchemaError(f"{context}.surface_id cannot be empty")
        sheet_id = str(value.get("sheet_id", surface_id)).strip()
        if not sheet_id:
            raise SchemaError(f"{context}.sheet_id cannot be empty")
        supervision_source = str(
            value.get("supervision_source", "manual-tifxyz")
        ).strip()
        if not supervision_source:
            raise SchemaError(f"{context}.supervision_source cannot be empty")
        supervision_weight = _positive_float(
            value.get("supervision_weight", 1.0),
            f"{context}.supervision_weight",
        )
        return cls(
            surface_id=surface_id,
            sheet_id=sheet_id,
            coarse_tifxyz=_resolve_path(
                _required(value, "coarse_tifxyz", context), base_directory
            ),
            fine_tifxyz=_resolve_path(
                _required(value, "fine_tifxyz", context), base_directory
            ),
            supervision_source=supervision_source,
            supervision_weight=supervision_weight,
        )


@dataclass(frozen=True)
class PairRecord:
    schema_version: int
    record_id: str
    scroll_id: str
    split: str
    coarse: ScanSpec
    fine: ScanSpec
    surfaces: tuple[SurfacePair, ...]

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, base_directory: Path | None = None
    ) -> PairRecord:
        if not isinstance(value, dict):
            raise SchemaError("pair record must be an object")
        try:
            version = int(value.get("schema_version", 1))
        except (TypeError, ValueError) as error:
            raise SchemaError("pair schema_version must be an integer") from error
        if version != 1:
            raise SchemaError(f"unsupported pair schema_version {version}")
        record_id = str(_required(value, "record_id", "record")).strip()
        if not record_id:
            raise SchemaError("record_id cannot be empty")
        split = str(_required(value, "split", "record")).strip().lower()
        if split not in {
            "train",
            "val",
            "test",
            "development_holdout",
            "sealed_prize_target",
        }:
            raise SchemaError(
                "split must be train, val, test, development_holdout, "
                "or sealed_prize_target"
            )
        raw_surfaces = _required(value, "surfaces", "record")
        if not isinstance(raw_surfaces, list) or not raw_surfaces:
            raise SchemaError("surfaces must be a non-empty array")
        surfaces = tuple(
            SurfacePair.from_dict(
                item,
                context=f"surfaces[{index}]",
                base_directory=base_directory,
            )
            for index, item in enumerate(raw_surfaces)
        )
        if len({item.surface_id for item in surfaces}) != len(surfaces):
            raise SchemaError(f"record {record_id!r} has duplicate surface_id values")
        return cls(
            schema_version=version,
            record_id=record_id,
            scroll_id=canonical_scroll_id(str(_required(value, "scroll_id", "record"))),
            split=split,
            coarse=ScanSpec.from_dict(
                _required(value, "coarse", "record"),
                context="coarse",
                base_directory=base_directory,
            ),
            fine=ScanSpec.from_dict(
                _required(value, "fine", "record"),
                context="fine",
                base_directory=base_directory,
            ),
            surfaces=surfaces,
        )


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise SchemaError(
                    f"{source}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise SchemaError(
                    f"{source}:{line_number}: each JSONL row must be an object"
                )
            yield value


def load_pair_records(path: str | Path) -> list[PairRecord]:
    source = Path(path).expanduser().resolve()
    records = [
        PairRecord.from_dict(item, base_directory=source.parent)
        for item in iter_jsonl(source)
    ]
    if not records:
        raise SchemaError(f"{path}: manifest contains no records")
    ids = [record.record_id for record in records]
    if len(ids) != len(set(ids)):
        raise SchemaError(f"{path}: record_id values must be unique")
    return records


def ensure_scroll_disjoint(records: Iterable[PairRecord]) -> None:
    scroll_splits: dict[str, set[str]] = {}
    for record in records:
        scroll_splits.setdefault(record.scroll_id, set()).add(
            normalized_split(record.split)
        )
    collisions = {
        scroll: sorted(splits)
        for scroll, splits in scroll_splits.items()
        if len(splits) > 1
    }
    if collisions:
        detail = ", ".join(
            f"{scroll}={splits}" for scroll, splits in sorted(collisions.items())
        )
        raise SchemaError(f"scroll-level split leakage: {detail}")
