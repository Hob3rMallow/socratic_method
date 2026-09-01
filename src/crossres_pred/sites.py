from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .policy import DataPolicy
from .provenance import sha256_file, utc_now, write_json_atomic
from .resample import fine_bbox_for_coarse_box
from .schema import PairRecord, SchemaError, load_pair_records, normalized_split
from .tifxyz import TifxyzMap


class SiteError(ValueError):
    pass


@dataclass(frozen=True)
class SiteOptions:
    """Planning of coarse ROI sites on traced-segment neighborhoods.

    One site manifest drives everything downstream: the fine carve fetches
    only chunks under each site's affine-image footprint, teacher inference
    and distillation targets are confined to the same footprints, and the
    student's distillation stratum is one patch per site. Origins are drawn
    uniformly over the segment bounding region -- never centered on surface
    points (the recorded surface-following sampling bias) -- and accepted
    only if the box actually contains traced surface.
    """

    site_shape_zyx: tuple[int, int, int] = (192, 192, 192)
    sites_per_record: int = 150
    seed: int = 1203
    min_surface_points: int = 64
    min_separation_vox: int = 64
    fine_halo_vox: int = 160
    max_attempt_multiplier: int = 200

    def validate(self) -> None:
        if len(self.site_shape_zyx) != 3 or any(
            size <= 0 or size % 32 for size in self.site_shape_zyx
        ):
            raise SiteError("site_shape_zyx must be three positive multiples of 32")
        if self.sites_per_record <= 0:
            raise SiteError("sites_per_record must be positive")
        if self.min_surface_points <= 0:
            raise SiteError("min_surface_points must be positive")
        if self.min_separation_vox < 0:
            raise SiteError("min_separation_vox must be non-negative")
        if self.fine_halo_vox < 0:
            raise SiteError("fine_halo_vox must be non-negative")
        if self.max_attempt_multiplier < 1:
            raise SiteError("max_attempt_multiplier must be positive")


def _record_surface_points_zyx(
    record: PairRecord, *, max_points_per_surface: int = 400_000
) -> np.ndarray:
    collected: list[np.ndarray] = []
    for surface in record.surfaces:
        mapping = TifxyzMap.load(surface.coarse_tifxyz)
        count = int(mapping.valid.sum())
        if count == 0:
            continue
        stride = max(1, int(np.ceil(np.sqrt(count / max_points_per_surface))))
        xyz = mapping.xyz[::stride, ::stride]
        valid = mapping.valid[::stride, ::stride]
        collected.append(xyz[valid][:, ::-1].astype(np.float32))
    if not collected:
        raise SiteError(f"{record.record_id}: no valid coarse surface points")
    return np.concatenate(collected, axis=0)


def _plan_record_sites(
    record: PairRecord,
    options: SiteOptions,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if record.fine.to_coarse_affine_xyz is None:
        raise SiteError(
            f"{record.record_id}: pair record is missing fine.to_coarse_affine_xyz"
        )
    points = _record_surface_points_zyx(record)
    shape = np.asarray(options.site_shape_zyx, dtype=np.int64)
    lo = np.maximum(0, np.floor(points.min(axis=0)) - 32).astype(np.int64)
    hi = (np.ceil(points.max(axis=0)) + 32).astype(np.int64)
    origin_lo = np.maximum(0, lo - shape // 2)
    origin_hi = np.maximum(origin_lo + 1, hi - shape // 2)

    accepted: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = options.sites_per_record * options.max_attempt_multiplier
    while len(rows) < options.sites_per_record and attempts < max_attempts:
        attempts += 1
        origin = np.array(
            [
                rng.integers(origin_lo[axis], origin_hi[axis] + 1)
                for axis in range(3)
            ],
            dtype=np.int64,
        )
        origin = (origin // 16) * 16
        if accepted and options.min_separation_vox > 0:
            separation = np.abs(np.stack(accepted) - origin).max(axis=1)
            if int(separation.min()) < options.min_separation_vox:
                continue
        inside = np.logical_and(
            points >= origin, points < origin + shape
        ).all(axis=1)
        surface_points = int(np.count_nonzero(inside))
        if surface_points < options.min_surface_points:
            continue
        accepted.append(origin)
        fine_lo, fine_hi = fine_bbox_for_coarse_box(
            tuple(int(item) for item in origin),
            tuple(int(item) for item in shape),
            record.fine.to_coarse_affine_xyz,
            margin_fine_vox=float(options.fine_halo_vox),
        )
        rows.append(
            {
                "schema_version": 1,
                "site_id": f"{record.record_id}_s{len(rows):04d}",
                "record_id": record.record_id,
                "scroll_id": record.scroll_id,
                "split": normalized_split(record.split),
                "coarse_origin_zyx": [int(item) for item in origin],
                "site_shape_zyx": [int(item) for item in shape],
                "surface_point_count": surface_points,
                "coarse_scan_id": record.coarse.scan_id,
                "coarse_voxel_um": record.coarse.voxel_um,
                "fine_scan_id": record.fine.scan_id,
                "fine_voxel_um": record.fine.voxel_um,
                "fine_to_coarse_affine_xyz": [
                    list(row) for row in record.fine.to_coarse_affine_xyz
                ],
                "fine_bbox_lo_zyx": [max(0, int(np.floor(item))) for item in fine_lo],
                "fine_bbox_hi_zyx": [max(0, int(np.ceil(item))) for item in fine_hi],
            }
        )
    if len(rows) < options.sites_per_record:
        raise SiteError(
            f"{record.record_id}: only placed {len(rows)} of "
            f"{options.sites_per_record} sites after {attempts} attempts; "
            "lower sites_per_record or min_surface_points"
        )
    return rows


def load_site_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise SiteError(
                    f"{source}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(row, dict):
                raise SiteError(f"{source}:{line_number}: site row must be an object")
            required = {
                "site_id",
                "record_id",
                "scroll_id",
                "split",
                "coarse_origin_zyx",
                "site_shape_zyx",
                "fine_bbox_lo_zyx",
                "fine_bbox_hi_zyx",
                "fine_to_coarse_affine_xyz",
            }
            missing = required.difference(row)
            if missing:
                raise SiteError(
                    f"{source}:{line_number}: site row missing {sorted(missing)}"
                )
            rows.append(row)
    if not rows:
        raise SiteError(f"{source}: no site rows")
    ids = [row["site_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SiteError(f"{source}: site_id values are not unique")
    return rows


def plan_sites(
    *,
    manifest_path: str | Path,
    policy_path: str | Path,
    output_path: str | Path,
    options: SiteOptions,
    reuse_sites: str | Path | None = None,
) -> Path:
    """Plan sites for every record in a pair manifest.

    ``reuse_sites`` locks the coarse boxes of an existing site manifest
    (matched by scroll) and re-derives only the fine-frame footprints from
    this manifest's records -- how the 1.129 um wave reuses the 2.4 um
    site placement.
    """

    options.validate()
    policy = DataPolicy.load(policy_path)
    records = load_pair_records(manifest_path)
    policy.validate_records(records)

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(options.seed)

    all_rows: list[dict[str, Any]] = []
    if reuse_sites is not None:
        source_rows = load_site_rows(reuse_sites)
        records_by_scroll: dict[str, PairRecord] = {}
        for record in records:
            if record.scroll_id in records_by_scroll:
                raise SiteError(
                    f"reuse-sites requires one record per scroll; "
                    f"{record.scroll_id} appears twice"
                )
            records_by_scroll[record.scroll_id] = record
        for row in source_rows:
            record = records_by_scroll.get(row["scroll_id"])
            if record is None:
                continue
            if record.fine.to_coarse_affine_xyz is None:
                raise SiteError(
                    f"{record.record_id}: pair record is missing "
                    "fine.to_coarse_affine_xyz"
                )
            origin = tuple(int(item) for item in row["coarse_origin_zyx"])
            shape = tuple(int(item) for item in row["site_shape_zyx"])
            fine_lo, fine_hi = fine_bbox_for_coarse_box(
                origin,
                shape,
                record.fine.to_coarse_affine_xyz,
                margin_fine_vox=float(options.fine_halo_vox),
            )
            new_row = dict(row)
            new_row.update(
                {
                    "site_id": f"{record.record_id}_{row['site_id']}",
                    "record_id": record.record_id,
                    "split": normalized_split(record.split),
                    "coarse_scan_id": record.coarse.scan_id,
                    "coarse_voxel_um": record.coarse.voxel_um,
                    "fine_scan_id": record.fine.scan_id,
                    "fine_voxel_um": record.fine.voxel_um,
                    "fine_to_coarse_affine_xyz": [
                        list(item) for item in record.fine.to_coarse_affine_xyz
                    ],
                    "fine_bbox_lo_zyx": [
                        max(0, int(np.floor(item))) for item in fine_lo
                    ],
                    "fine_bbox_hi_zyx": [
                        max(0, int(np.ceil(item))) for item in fine_hi
                    ],
                    "reused_from": str(Path(reuse_sites).resolve()),
                }
            )
            all_rows.append(new_row)
        if not all_rows:
            raise SiteError("reuse-sites matched no scrolls in the manifest")
    else:
        for record in records:
            all_rows.extend(_plan_record_sites(record, options, rng))

    try:
        splits = {
            (row["scroll_id"], row["split"]) for row in all_rows
        }
        by_scroll: dict[str, set[str]] = {}
        for scroll, split in splits:
            by_scroll.setdefault(scroll, set()).add(split)
        leaking = {k: sorted(v) for k, v in by_scroll.items() if len(v) > 1}
        if leaking:
            raise SchemaError(f"scroll-level site leakage: {leaking}")
    except SchemaError:
        raise

    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, output)

    write_json_atomic(
        output.with_suffix(output.suffix + ".provenance.json"),
        {
            "schema_version": 1,
            "kind": "crossres-site-plan",
            "created_at": utc_now(),
            "pair_manifest": {
                "path": str(Path(manifest_path).resolve()),
                "sha256": sha256_file(manifest_path),
            },
            "policy": {
                "path": str(Path(policy_path).resolve()),
                "profile": policy.profile,
            },
            "options": asdict(options),
            "reuse_sites": (
                str(Path(reuse_sites).resolve()) if reuse_sites is not None else None
            ),
            "site_count": len(all_rows),
            "sites_by_scroll": {
                scroll: sum(1 for row in all_rows if row["scroll_id"] == scroll)
                for scroll in sorted({row["scroll_id"] for row in all_rows})
            },
        },
    )
    return output
