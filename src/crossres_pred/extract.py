from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .carve import load_carved_chunk_ids
from .provenance import sha256_file, utc_now, write_json_atomic
from .rasterize import (
    LABEL_IGNORE,
    LABEL_SURFACE,
    RasterizeOptions,
    collect_surface_points_zyx,
    default_options_for_pitch,
    rasterize_label_block,
)
from .resample import (
    BridgeOptions,
    ChunkCoverage,
    affine_scale_ratio,
    phase_correlation_shift,
    registration_action,
    resample_to_coarse,
)
from .schema import PairRecord
from .tifxyz import TifxyzMap
from .volume import open_volume, read_crop


class ExtractError(RuntimeError):
    pass


def _write_manifest(
    output: Path, rows: list[dict[str, Any]], provenance: dict[str, Any]
) -> Path:
    if not rows:
        raise ExtractError("no patches were extracted")
    manifest = output / "patches.jsonl"
    temporary = manifest.with_name(manifest.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, manifest)
    provenance = dict(provenance)
    provenance.update(
        {
            "created_at": utc_now(),
            "patch_count": len(rows),
            "manifest_sha256": sha256_file(manifest),
        }
    )
    write_json_atomic(output / "provenance.json", provenance)
    return manifest


def _save_patch_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def chunk_ids_from_object_plan(
    mirror_path: str | Path, *, array_key: str = "0"
) -> tuple[tuple[int, int, int], set[tuple[int, int, int]]]:
    """Coverage chunk ids for a legacy sparse mirror (villa pred).

    Mirrors made by ``mirror_sparse_teacher.py`` recorded only the object
    plan, not the selected-chunk list, so coverage falls back to *existing*
    chunks -- conservative (selected-but-empty chunks read as uncovered).
    """

    path = Path(mirror_path).expanduser().resolve()
    manifest = json.loads(
        (path / "crossres_sparse_mirror.json").read_text(encoding="utf-8")
    )
    chunks = tuple(int(item) for item in manifest["zarr"]["chunks_zyx"])
    ids: set[tuple[int, int, int]] = set()
    plan = path / "crossres_sparse_objects.jsonl"
    prefix = f"{array_key}/"
    with plan.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            relative = str(value.get("relative_path", ""))
            if value.get("kind") != "chunk" or not relative.startswith(prefix):
                continue
            parts = relative[len(prefix) :].split("/")
            if len(parts) != 3:
                continue
            try:
                ids.add(tuple(int(item) for item in parts))
            except ValueError:
                continue
    if not ids:
        raise ExtractError(f"{plan}: no chunk objects found")
    return chunks, ids


def coverage_for_mirror(mirror_path: str | Path):
    """Coverage from a carve/prediction mirror or legacy sparse mirror.

    Teacher-inference stores also carry a voxel validity array so masked scan
    voids remain unknown through the bridge. Raw carves and legacy villa
    mirrors retain the original chunk-selection-only behavior.
    """

    path = Path(mirror_path).expanduser().resolve()
    if (path / "carve_selected_chunks.json").is_file():
        chunks, ids = load_carved_chunk_ids(path)
    else:
        chunks, ids = chunk_ids_from_object_plan(path)
    chunk_coverage = ChunkCoverage(chunks, ids)
    manifest_path = path / "crossres_sparse_mirror.json"
    if not manifest_path.is_file():
        return chunk_coverage
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validity_key = manifest.get("validity_array_key")
    if not validity_key:
        return chunk_coverage
    validity = open_volume(f"{path}::{validity_key}")

    def combined(origin, shape):
        selected = chunk_coverage(origin, shape)
        voxel_valid = read_crop(validity, origin, shape) > 0
        return (selected & voxel_valid).astype(np.uint8)

    return combined


def _fine_maps(record: PairRecord) -> list[TifxyzMap]:
    return [TifxyzMap.load(surface.fine_tifxyz) for surface in record.surfaces]


def affine_scale_ratio_hint(
    site_rows: list[dict[str, Any]], record: PairRecord
) -> float:
    for row in site_rows:
        if row["record_id"] == record.record_id:
            return affine_scale_ratio(row["fine_to_coarse_affine_xyz"])
    raise ExtractError(f"no sites for record {record.record_id}")


def _coarse_maps(record: PairRecord) -> list[TifxyzMap]:
    return [TifxyzMap.load(surface.coarse_tifxyz) for surface in record.surfaces]


# ---------------------------------------------------------------------------
# Distillation targets (the bridge stage)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistillOptions:
    target_source: str = "teacher-pred"  # or "gt-bridge"
    band_radius_um: float = 14.0
    bridge: BridgeOptions = BridgeOptions()
    registration_reference: str | None = None
    registration_accept_vox: float = 0.75
    registration_correct_vox: float = 2.0

    def validate(self) -> None:
        if self.target_source not in {"teacher-pred", "gt-bridge"}:
            raise ExtractError("target_source must be teacher-pred or gt-bridge")
        self.bridge.validate()


def build_distill_targets(
    *,
    site_rows: list[dict[str, Any]],
    record: PairRecord,
    output_path: str | Path,
    options: DistillOptions,
    teacher_prob_spec: str | None = None,
    teacher_mirror_path: str | Path | None = None,
) -> Path:
    """Produce per-site coarse soft targets q and validity masks V.

    ``teacher-pred`` pulls back a predicted fine probability volume (teacher
    or villa), with coverage from the mirror's selected/existing chunks.
    ``gt-bridge`` rasterizes the fine ground-truth band on the fly and pushes
    it through the *same* operator, restricting validity to the band+shell
    known region -- producing truth-grade validation targets with operator
    parity. An optional coarse reference volume drives the per-site
    registration-residual audit (accept / correct / reject).
    """

    options.validate()
    output = Path(output_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [row for row in site_rows if row["record_id"] == record.record_id]
    if not rows:
        raise ExtractError(f"no sites for record {record.record_id}")

    fine_pitch = float(rows[0]["fine_voxel_um"])
    raster_options = default_options_for_pitch(
        pitch_um=fine_pitch, band_radius_um=options.band_radius_um
    )

    reference_volume = None
    if options.registration_reference is not None:
        reference_volume = open_volume(options.registration_reference)

    if options.target_source == "teacher-pred":
        if teacher_prob_spec is None or teacher_mirror_path is None:
            raise ExtractError(
                "teacher-pred targets require teacher_prob_spec and "
                "teacher_mirror_path"
            )
        probability_volume = open_volume(teacher_prob_spec)
        prob_dtype = np.dtype(probability_volume.dtype)
        prob_scale = 255.0 if prob_dtype == np.uint8 else 1.0
        coverage = coverage_for_mirror(teacher_mirror_path)

        def make_providers(row: dict[str, Any]):
            def read_prob(origin, shape):
                crop = read_crop(probability_volume, origin, shape)
                return crop.astype(np.float32) / prob_scale

            return read_prob, coverage

    else:
        fine_maps = _fine_maps(record)
        # The bridge asks for a prob window and then a coverage window that
        # is the same window expanded by the erosion margin. Rasterizing once
        # on the expanded window and slicing serves both from a single EDT.
        filter_margin = (
            int(
                np.ceil(
                    3.0
                    * options.bridge.prefilter_sigma_scale
                    * affine_scale_ratio_hint(site_rows, record)
                )
            )
            + 2
        )
        erosion = (
            filter_margin if options.bridge.erode_filter_margin else 0
        ) + options.bridge.coverage_erosion_fine_vox

        def make_providers(row: dict[str, Any]):
            lo = row["fine_bbox_lo_zyx"]
            hi = row["fine_bbox_hi_zyx"]
            points = collect_surface_points_zyx(
                fine_maps,
                bbox_lo_zyx=tuple(float(item) for item in lo),
                bbox_hi_zyx=tuple(float(item) for item in hi),
                margin_vox=float(raster_options.padding_vox) + erosion,
            )
            cache: dict[
                tuple[tuple[int, int, int], tuple[int, int, int]], np.ndarray
            ] = {}

            def expanded_label(origin, shape):
                key = (tuple(origin), tuple(shape))
                if key not in cache:
                    cache.clear()
                    cache[key] = rasterize_label_block(
                        points,
                        origin_zyx=key[0],
                        shape_zyx=key[1],
                        options=raster_options,
                    )
                return cache[key]

            def read_prob(origin, shape):
                expanded_origin = tuple(int(item) - erosion for item in origin)
                expanded_shape = tuple(
                    int(item) + 2 * erosion for item in shape
                )
                label = expanded_label(expanded_origin, expanded_shape)
                core = tuple(
                    slice(erosion, erosion + int(item)) for item in shape
                )
                return (label[core] == LABEL_SURFACE).astype(np.float32)

            def read_cov(origin, shape):
                label = expanded_label(
                    tuple(int(item) for item in origin),
                    tuple(int(item) for item in shape),
                )
                return (label != LABEL_IGNORE).astype(np.uint8)

            return read_prob, read_cov

    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        site_id = str(row["site_id"])
        origin = tuple(int(item) for item in row["coarse_origin_zyx"])
        shape = tuple(int(item) for item in row["site_shape_zyx"])
        affine = row["fine_to_coarse_affine_xyz"]
        read_prob, read_cov = make_providers(row)
        q, valid = resample_to_coarse(
            read_prob,
            read_cov,
            coarse_origin_zyx=origin,
            coarse_shape_zyx=shape,
            fine_to_coarse_affine_xyz=affine,
            options=options.bridge,
        )
        action = "accept"
        shift = (0.0, 0.0, 0.0)
        peak = None
        if reference_volume is not None and valid.any():
            reference = (
                read_crop(reference_volume, origin, shape).astype(np.float32)
            )
            reference_band = (reference > 0).astype(np.float32)
            moving = (q >= 0.5).astype(np.float32) * valid
            if moving.any() and reference_band.any():
                shift, peak = phase_correlation_shift(moving, reference_band)
                action = registration_action(
                    shift,
                    accept_vox=options.registration_accept_vox,
                    correct_vox=options.registration_correct_vox,
                )
                if action == "correct":
                    q, valid = resample_to_coarse(
                        read_prob,
                        read_cov,
                        coarse_origin_zyx=origin,
                        coarse_shape_zyx=shape,
                        fine_to_coarse_affine_xyz=affine,
                        options=options.bridge,
                        coarse_shift_zyx=tuple(-value for value in shift),
                    )
                elif action == "reject":
                    valid = np.zeros_like(valid)
        _save_patch_npz(
            output / f"{site_id}.npz",
            {
                "distill_u8": np.rint(np.clip(q, 0.0, 1.0) * 255.0).astype(
                    np.uint8
                ),
                "distill_valid_u8": valid.astype(np.uint8),
            },
        )
        band = (q >= 0.5) & (valid > 0)
        audit_rows.append(
            {
                "site_id": site_id,
                "coverage_fraction": float(valid.mean()),
                "band_fraction": float(band.mean()),
                "mean_q_in_valid": (
                    float(q[valid > 0].mean()) if valid.any() else 0.0
                ),
                "registration_action": action,
                "registration_shift_zyx": list(shift),
                "registration_peak": peak,
            }
        )

    write_json_atomic(
        output / "distill_audit.json",
        {
            "schema_version": 1,
            "kind": "crossres-distill-targets",
            "created_at": utc_now(),
            "record_id": record.record_id,
            "scroll_id": record.scroll_id,
            "target_source": options.target_source,
            "options": {
                "band_radius_um": options.band_radius_um,
                "bridge": asdict(options.bridge),
                "registration_reference": options.registration_reference,
            },
            "site_count": len(audit_rows),
            "median_coverage_fraction": float(
                np.median([row["coverage_fraction"] for row in audit_rows])
            ),
            "registration_actions": {
                action: sum(
                    1
                    for row in audit_rows
                    if row["registration_action"] == action
                )
                for action in ("accept", "correct", "reject")
            },
            "sites": audit_rows,
        },
    )
    return output


# ---------------------------------------------------------------------------
# Teacher corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractTeacherOptions:
    patch_shape_zyx: tuple[int, int, int] = (256, 256, 256)
    patches_per_site: int = 6
    seed: int = 1203
    min_supervised_fraction: float = 0.10
    band_radius_um: float = 14.0
    max_attempts_per_site: int = 80

    def validate(self) -> None:
        if any(size < 64 or size % 32 for size in self.patch_shape_zyx):
            raise ExtractError(
                "patch_shape_zyx must be multiples of 32 and at least 64"
            )
        if self.patches_per_site <= 0:
            raise ExtractError("patches_per_site must be positive")
        if not 0.0 <= self.min_supervised_fraction < 1.0:
            raise ExtractError("min_supervised_fraction must be in [0, 1)")


def extract_teacher_patches(
    *,
    site_rows: list[dict[str, Any]],
    record: PairRecord,
    mirror_path: str | Path,
    output_path: str | Path,
    options: ExtractTeacherOptions,
    policy_profile: str,
    veto_volume_spec: str | None = None,
) -> Path:
    """Materialize fine-pitch teacher patches for one record.

    Patch origins are uniform within each site's fine footprint; a patch is
    accepted only if every overlapping chunk was carved (no partial-context
    patches -- the recorded padded-context trap) and enough of it is
    supervised (label != ignore). An optional veto volume (villa prediction)
    flips would-be background near untraced wraps to ignore.
    """

    options.validate()
    output = Path(output_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [row for row in site_rows if row["record_id"] == record.record_id]
    if not rows:
        raise ExtractError(f"no sites for record {record.record_id}")
    fine_pitch = float(rows[0]["fine_voxel_um"])
    raster_options = default_options_for_pitch(
        pitch_um=fine_pitch, band_radius_um=options.band_radius_um
    )
    mirror = Path(mirror_path).expanduser().resolve()
    fine_volume = open_volume(f"{mirror}::0")
    coverage = coverage_for_mirror(mirror)
    veto_volume = (
        open_volume(veto_volume_spec) if veto_volume_spec is not None else None
    )
    fine_maps = _fine_maps(record)
    rng = np.random.default_rng(options.seed)
    shape = np.asarray(options.patch_shape_zyx, dtype=np.int64)

    patch_rows: list[dict[str, Any]] = []
    for row in rows:
        lo = np.asarray(row["fine_bbox_lo_zyx"], dtype=np.int64)
        hi = np.asarray(row["fine_bbox_hi_zyx"], dtype=np.int64)
        if np.any(hi - lo < shape):
            continue
        points = collect_surface_points_zyx(
            fine_maps,
            bbox_lo_zyx=tuple(float(item) for item in lo),
            bbox_hi_zyx=tuple(float(item) for item in hi),
            margin_vox=float(raster_options.padding_vox),
        )
        if points.shape[0] == 0:
            continue
        accepted = 0
        for _ in range(options.max_attempts_per_site):
            if accepted >= options.patches_per_site:
                break
            origin = np.array(
                [
                    rng.integers(lo[axis], hi[axis] - shape[axis] + 1)
                    for axis in range(3)
                ],
                dtype=np.int64,
            )
            origin_tuple = tuple(int(item) for item in origin)
            shape_tuple = tuple(int(item) for item in shape)
            if not coverage(origin_tuple, shape_tuple).all():
                continue
            veto = None
            if veto_volume is not None:
                veto = read_crop(veto_volume, origin_tuple, shape_tuple) > 0
            label = rasterize_label_block(
                points,
                origin_zyx=origin_tuple,
                shape_zyx=shape_tuple,
                options=raster_options,
                veto=veto,
            )
            if float((label != LABEL_IGNORE).mean()) < options.min_supervised_fraction:
                continue
            image = read_crop(fine_volume, origin_tuple, shape_tuple)
            # Masked volumes store exact zero where the scan has no data.
            # Traced segments can extend past the fine scan's footprint;
            # supervising there would teach surfaces over emptiness.
            label[np.asarray(image) == 0] = LABEL_IGNORE
            supervised = float((label != LABEL_IGNORE).mean())
            if supervised < options.min_supervised_fraction:
                continue
            patch_id = f"{row['site_id']}_t{accepted:02d}"
            _save_patch_npz(
                output / f"{patch_id}.npz",
                {"image": image, "label_u8": label},
            )
            patch_rows.append(
                {
                    "schema_version": 2,
                    "patch_id": patch_id,
                    "path": f"{patch_id}.npz",
                    "record_id": record.record_id,
                    "scroll_id": record.scroll_id,
                    "split": str(row["split"]),
                    "kind": "teacher",
                    "origin_zyx": [int(item) for item in origin],
                    "shape_zyx": [int(item) for item in shape],
                    "policy_profile": policy_profile,
                    "pitch_um": fine_pitch,
                    "sampling_stratum": "site-uniform",
                }
            )
            accepted += 1

    return _write_manifest(
        output,
        patch_rows,
        {
            "schema_version": 2,
            "kind": "crossres-teacher-corpus",
            "record_id": record.record_id,
            "scroll_id": record.scroll_id,
            "pitch_um": fine_pitch,
            "options": asdict(options),
            "mirror": str(mirror),
            "veto_volume": veto_volume_spec,
            "site_count": len(rows),
        },
    )


# ---------------------------------------------------------------------------
# Student corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractStudentOptions:
    patch_shape_zyx: tuple[int, int, int] = (192, 192, 192)
    anchor_patches: int = 200
    rehearsal_patches: int = 150
    seed: int = 1203
    min_supervised_fraction: float = 0.02
    min_image_nonzero_fraction: float = 0.05
    band_radius_um: float = 14.0
    max_attempt_multiplier: int = 40
    holdout_segment_fraction: float = 0.0

    def validate(self) -> None:
        if any(size < 64 or size % 32 for size in self.patch_shape_zyx):
            raise ExtractError(
                "patch_shape_zyx must be multiples of 32 and at least 64"
            )
        if self.anchor_patches < 0 or self.rehearsal_patches < 0:
            raise ExtractError("patch counts must be non-negative")
        if not 0.0 <= self.holdout_segment_fraction < 1.0:
            raise ExtractError("holdout_segment_fraction must be in [0, 1)")


def extract_student_patches(
    *,
    site_rows: list[dict[str, Any]],
    record: PairRecord,
    distill_dir: str | Path | None,
    output_path: str | Path,
    options: ExtractStudentOptions,
    policy_profile: str,
) -> Path:
    """Materialize coarse-pitch student patches for one record.

    Strata: ``distill`` (one patch per site, with the bridge's q/V fields),
    ``anchor`` (uniform patches over the traced region with rasterized
    coarse ground truth), ``rehearsal`` (volume-uniform patches whose only
    supervision is the m7 baseline band as a weak prior). Origins are always
    uniform draws -- never surface-point-centered.
    """

    options.validate()
    output = Path(output_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [row for row in site_rows if row["record_id"] == record.record_id]
    coarse_pitch = float(record.coarse.voxel_um)
    raster_options = default_options_for_pitch(
        pitch_um=coarse_pitch, band_radius_um=options.band_radius_um
    )
    if record.coarse.volume is None:
        raise ExtractError(f"{record.record_id}: coarse volume is required")
    coarse_volume = open_volume(record.coarse.volume)
    baseline_volume = (
        open_volume(record.coarse.baseline)
        if record.coarse.baseline is not None
        else None
    )
    rng = np.random.default_rng(options.seed)
    shape = np.asarray(options.patch_shape_zyx, dtype=np.int64)
    shape_tuple = tuple(int(item) for item in shape)
    split = record.split
    from .schema import normalized_split

    split = normalized_split(split)

    coarse_maps = _coarse_maps(record)
    holdout_count = int(
        round(len(coarse_maps) * options.holdout_segment_fraction)
    )
    holdout_indices = set(
        rng.choice(len(coarse_maps), size=holdout_count, replace=False).tolist()
        if holdout_count
        else []
    )
    training_maps = [
        mapping
        for index, mapping in enumerate(coarse_maps)
        if index not in holdout_indices
    ]
    if not training_maps:
        raise ExtractError(f"{record.record_id}: all segments held out")

    # Raw parameter-grid cells are ~20 voxels apart: fine for region bounds,
    # never for rasterization (a sparse splat makes a polka-dot tube).
    sparse_points = np.concatenate(
        [mapping.xyz[mapping.valid][:, ::-1] for mapping in training_maps],
        axis=0,
    ).astype(np.float32)
    region_lo = np.maximum(0, np.floor(sparse_points.min(axis=0)) - 32).astype(
        np.int64
    )
    region_hi = (np.ceil(sparse_points.max(axis=0)) + 32).astype(np.int64)

    def block_label(origin_tuple: tuple[int, int, int]) -> np.ndarray:
        dense = collect_surface_points_zyx(
            training_maps,
            bbox_lo_zyx=tuple(float(item) for item in origin_tuple),
            bbox_hi_zyx=tuple(
                float(origin_tuple[axis] + shape_tuple[axis]) for axis in range(3)
            ),
            margin_vox=float(raster_options.padding_vox),
        )
        return rasterize_label_block(
            dense,
            origin_zyx=origin_tuple,
            shape_zyx=shape_tuple,
            options=raster_options,
        )

    patch_rows: list[dict[str, Any]] = []

    def emit(
        patch_id: str,
        origin: np.ndarray,
        arrays: dict[str, np.ndarray],
        stratum: str,
    ) -> None:
        _save_patch_npz(output / f"{patch_id}.npz", arrays)
        patch_rows.append(
            {
                "schema_version": 2,
                "patch_id": patch_id,
                "path": f"{patch_id}.npz",
                "record_id": record.record_id,
                "scroll_id": record.scroll_id,
                "split": split,
                "kind": "student",
                "origin_zyx": [int(item) for item in origin],
                "shape_zyx": [int(item) for item in shape],
                "policy_profile": policy_profile,
                "pitch_um": coarse_pitch,
                "sampling_stratum": stratum,
            }
        )

    # --- distill stratum: one patch per site -------------------------------
    if distill_dir is not None:
        distill = Path(distill_dir).expanduser().resolve()
        for row in rows:
            site_id = str(row["site_id"])
            target_path = distill / f"{site_id}.npz"
            if not target_path.is_file():
                raise ExtractError(f"missing distill targets: {target_path}")
            origin = np.asarray(row["coarse_origin_zyx"], dtype=np.int64)
            site_shape = tuple(int(item) for item in row["site_shape_zyx"])
            if site_shape != shape_tuple:
                raise ExtractError(
                    f"{site_id}: site shape {site_shape} != patch shape "
                    f"{shape_tuple}"
                )
            origin_tuple = tuple(int(item) for item in origin)
            with np.load(target_path, allow_pickle=False) as archive:
                distill_u8 = np.asarray(archive["distill_u8"])
                distill_valid = np.asarray(archive["distill_valid_u8"])
            image = read_crop(coarse_volume, origin_tuple, shape_tuple)
            label = block_label(origin_tuple)
            label[np.asarray(image) == 0] = LABEL_IGNORE
            arrays = {
                "image": image,
                "label_u8": label,
                "distill_u8": distill_u8,
                "distill_valid_u8": distill_valid,
            }
            if baseline_volume is not None:
                arrays["baseline_u8"] = read_crop(
                    baseline_volume, origin_tuple, shape_tuple
                ).astype(np.uint8)
            emit(f"{site_id}_d", origin, arrays, "distill")

    # --- anchor stratum: uniform over the traced region --------------------
    attempts_cap = options.anchor_patches * options.max_attempt_multiplier
    accepted = 0
    attempts = 0
    while accepted < options.anchor_patches and attempts < attempts_cap:
        attempts += 1
        origin = np.array(
            [
                rng.integers(
                    max(0, region_lo[axis] - shape[axis] // 2),
                    max(1, region_hi[axis] - shape[axis] // 2),
                )
                for axis in range(3)
            ],
            dtype=np.int64,
        )
        origin_tuple = tuple(int(item) for item in origin)
        # Cheap reject on the sparse cloud before paying for densification.
        near = np.logical_and(
            sparse_points
            >= np.asarray(origin_tuple) - raster_options.background_radius_vox,
            sparse_points
            < np.asarray(origin_tuple)
            + np.asarray(shape_tuple)
            + raster_options.background_radius_vox,
        ).all(axis=1)
        if not near.any():
            continue
        label = block_label(origin_tuple)
        if float((label != LABEL_IGNORE).mean()) < options.min_supervised_fraction:
            continue
        image = read_crop(coarse_volume, origin_tuple, shape_tuple)
        label[np.asarray(image) == 0] = LABEL_IGNORE
        if float((label != LABEL_IGNORE).mean()) < options.min_supervised_fraction:
            continue
        arrays = {"image": image, "label_u8": label}
        if baseline_volume is not None:
            arrays["baseline_u8"] = read_crop(
                baseline_volume, origin_tuple, shape_tuple
            ).astype(np.uint8)
        emit(f"{record.record_id}_a{accepted:04d}", origin, arrays, "anchor")
        accepted += 1

    # --- rehearsal stratum: volume-uniform with the m7 band as weak prior --
    if baseline_volume is not None and options.rehearsal_patches > 0:
        volume_shape = np.asarray(coarse_volume.shape, dtype=np.int64)
        attempts_cap = options.rehearsal_patches * options.max_attempt_multiplier
        accepted = 0
        attempts = 0
        while accepted < options.rehearsal_patches and attempts < attempts_cap:
            attempts += 1
            origin = np.array(
                [
                    rng.integers(0, max(1, volume_shape[axis] - shape[axis]))
                    for axis in range(3)
                ],
                dtype=np.int64,
            )
            origin_tuple = tuple(int(item) for item in origin)
            image = read_crop(coarse_volume, origin_tuple, shape_tuple)
            if (
                float((np.asarray(image) != 0).mean())
                < options.min_image_nonzero_fraction
            ):
                continue
            baseline = read_crop(baseline_volume, origin_tuple, shape_tuple)
            rehearsal = (np.asarray(baseline) != 0).astype(np.uint8) * np.uint8(
                255
            )
            arrays = {
                "image": image,
                "label_u8": np.full(
                    shape_tuple, LABEL_IGNORE, dtype=np.uint8
                ),
                "rehearsal_u8": rehearsal,
                "rehearsal_valid_u8": np.ones(shape_tuple, dtype=np.uint8),
                "baseline_u8": (np.asarray(baseline) != 0).astype(np.uint8),
            }
            emit(
                f"{record.record_id}_r{accepted:04d}", origin, arrays, "rehearsal"
            )
            accepted += 1

    return _write_manifest(
        output,
        patch_rows,
        {
            "schema_version": 2,
            "kind": "crossres-student-corpus",
            "record_id": record.record_id,
            "scroll_id": record.scroll_id,
            "pitch_um": coarse_pitch,
            "options": asdict(options),
            "distill_dir": (str(distill_dir) if distill_dir is not None else None),
            "site_count": len(rows),
            "holdout_segments": sorted(
                coarse_maps[index].source.as_posix() for index in holdout_indices
            ),
            "strata": {
                stratum: sum(
                    1
                    for row in patch_rows
                    if row["sampling_stratum"] == stratum
                )
                for stratum in ("distill", "anchor", "rehearsal")
            },
        },
    )
