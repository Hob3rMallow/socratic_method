#!/usr/bin/env python3
"""Build a reboot-resumable atlas-backed training corpus without patch archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from crossres_pred.voxel.coarse_teacher_atlas import (
    ATLAS_PROJECTION_CONTRACT,
    validate_coarse_teacher_atlas,
    validate_coarse_teacher_medial_atlas,
)
from crossres_pred.voxel.io import decode_dense_field, open_volume, read_crop
from crossres_pred.voxel.medial import (
    MEDIAL_MAX_PROJECTION_CONTRACT,
    VILLA_MEDIAL_SURFACE_CONTRACT,
)
from crossres_pred.voxel.patches import (
    ATLAS_PATCH_PREPARATION_VERSION,
    MEDIAL_ATLAS_PATCH_PREPARATION_VERSION,
    PATCH_SCHEMA,
)
from crossres_pred.voxel.registration import affine_matrix, transform_xyz
from crossres_pred.voxel.resources import configure_cpu_budget
from crossres_pred.voxel.schema import (
    DenseFieldSpec,
    VoxelPairRecord,
    load_pair_manifest,
)
from crossres_pred.voxel.scrollfiesta_metrics import (
    SCROLLFIESTA_PRED_METRICS_CONTRACT,
    scrollfiesta_patch_pred_metrics,
)

PLAN_SCHEMA = "crossres-atlas-patch-corpus-plan-v1"
STATE_SCHEMA = "crossres-atlas-patch-corpus-state-v1"
SUMMARY_SCHEMA = "crossres-atlas-patch-corpus-summary-v1"
CATALOG_SCHEMA = "crossres-coarse-teacher-atlas-catalog-v1"
MEDIAL_CATALOG_SCHEMA = "crossres-coarse-teacher-atlas-catalog-v2"
REGISTRATION_SCHEMA = "crossres-qualified-global-registration-commit-v1"
ANCHOR_GATE_SCHEMA = "crossres-atlas-anchor-quality-gate-v1"
BATCH_SIZE = 128


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Tolerate brief Windows sharing locks from concurrent corpus-state readers."""

    attempts = 100
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(f"{path}: expected JSON objects")
    return rows


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _dense_field_dict(value: DenseFieldSpec | None) -> dict[str, Any] | None:
    if value is None:
        return None
    result: dict[str, Any] = {
        "volume": value.volume,
        "encoding": value.encoding,
        "positive_labels": list(value.positive_labels),
        "ignore_labels": list(value.ignore_labels),
        "probability_scale": value.probability_scale,
        "threshold": value.threshold,
    }
    return result


def _inventory_coordinates(path: Path, array_key: str) -> list[tuple[int, int, int]]:
    prefix = tuple(part for part in array_key.split("/") if part)
    result: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") != "chunk":
            continue
        parts = tuple(str(row["relative_path"]).replace("\\", "/").split("/"))
        raw = parts[len(prefix) :] if parts[: len(prefix)] == prefix else ()
        if len(raw) != 3:
            raise ValueError(f"{path}:{line_number}: invalid sparse chunk path")
        coordinate = tuple(int(value) for value in raw)
        if coordinate in seen:
            raise ValueError(f"{path}:{line_number}: duplicate sparse chunk")
        seen.add(coordinate)
        result.append(coordinate)  # type: ignore[arg-type]
    return result


def _candidate_coordinates(state: dict[str, Any]) -> list[tuple[int, int, int]]:
    identity = state["identity"]
    candidate_path = identity.get("candidate_fine_chunks_path")
    if candidate_path:
        rows = _read_jsonl(Path(str(candidate_path)))
        if any(row.get("accepted", True) is not True for row in rows):
            raise ValueError("atlas candidate fine chunks include rejected rows")
        result = [tuple(int(value) for value in row["chunk_zyx"]) for row in rows]
    else:
        target = str(identity["fine_target"])
        _, separator, key = target.rpartition("::")
        result = _inventory_coordinates(
            Path(str(identity["fine_support_inventory"])), key if separator else "0"
        )
    if not result or len(result) != len(set(result)):
        raise ValueError("atlas candidate fine chunks are empty or duplicated")
    return result


def _fine_chunks_zyx(state: dict[str, Any]) -> tuple[int, int, int]:
    metadata = _read_json(Path(str(state["identity"]["fine_target_metadata"])))
    chunks = tuple(int(value) for value in metadata["chunks"])
    if len(chunks) != 3:
        raise ValueError("fine target chunks are not three-dimensional")
    return chunks  # type: ignore[return-value]


def _hash_numbers(text: str, count: int) -> list[int]:
    digest = hashlib.sha256(text.encode()).digest()
    values: list[int] = []
    while len(values) < count:
        for offset in range(0, len(digest), 4):
            values.append(int.from_bytes(digest[offset : offset + 4], "little"))
            if len(values) == count:
                break
        digest = hashlib.sha256(digest).digest()
    return values


def _logical_origin(
    *,
    center_zyx: np.ndarray,
    coarse_shape_zyx: tuple[int, int, int],
    patch_shape_zyx: tuple[int, int, int],
    key: str,
) -> tuple[int, int, int]:
    numbers = _hash_numbers(key, 3)
    patch = np.asarray(patch_shape_zyx, dtype=np.int64)
    coarse = np.asarray(coarse_shape_zyx, dtype=np.int64)
    # Keep the teacher anchor inside ScrollFiesta's centered 128-cube while
    # varying surrounding coarse context reproducibly.
    low = np.maximum((patch - 128) // 2 + 16, 0)
    high = np.minimum((patch + 128) // 2 - 16, patch - 1)
    local = np.asarray(
        [
            int(lo + number % max(1, int(hi - lo + 1)))
            for lo, hi, number in zip(low, high, numbers, strict=True)
        ],
        dtype=np.int64,
    )
    origin = np.rint(center_zyx).astype(np.int64) - local
    origin = np.minimum(np.maximum(origin, 0), coarse - patch)
    if (origin < 0).any():
        raise ValueError("coarse volume is smaller than the requested patch")
    return tuple(int(value) for value in origin)


class SourceContext:
    def __init__(
        self,
        *,
        pair: VoxelPairRecord,
        atlas_root: Path,
        atlas_state: dict[str, Any],
        medial_state: dict[str, Any] | None,
        catalog_path: Path,
        catalog_sha256: str,
        registration_sha256: str,
        patch_shape_zyx: tuple[int, int, int],
        minimum_ct_nonzero_fraction: float,
        minimum_known_fraction: float,
        minimum_positive_voxels: int,
        minimum_crest_voxels: int,
        seed: int,
    ) -> None:
        self.pair = pair
        self.atlas_root = atlas_root
        self.atlas_state = atlas_state
        self.atlas_state_path = atlas_root / "atlas_state.json"
        self.atlas_state_sha256 = _sha256(self.atlas_state_path)
        self.medial_state = medial_state
        self.medial_state_path = atlas_root / "medial_state.json"
        self.medial_state_sha256 = (
            _sha256(self.medial_state_path) if medial_state is not None else None
        )
        self.catalog_path = catalog_path
        self.catalog_sha256 = catalog_sha256
        self.registration_sha256 = registration_sha256
        self.patch_shape_zyx = patch_shape_zyx
        self.minimum_ct_nonzero_fraction = minimum_ct_nonzero_fraction
        self.minimum_known_fraction = minimum_known_fraction
        self.minimum_positive_voxels = minimum_positive_voxels
        self.minimum_crest_voxels = minimum_crest_voxels
        self.seed = seed
        self.anchors = _candidate_coordinates(atlas_state)
        self.fine_chunks_zyx = _fine_chunks_zyx(atlas_state)
        self.image = open_volume(pair.coarse.image)
        self.baseline_spec = pair.coarse.baseline
        self.baseline = (
            open_volume(pair.coarse.baseline.volume)
            if pair.coarse.baseline is not None
            else None
        )
        self.teacher_q = open_volume(str(atlas_state["teacher_q"]))
        self.target_valid = open_volume(str(atlas_state["target_valid"]))
        self.teacher_crest = (
            open_volume(str(medial_state["teacher_crest"]))
            if medial_state is not None
            else None
        )
        self.teacher_crest_valid = (
            open_volume(str(medial_state["teacher_crest_valid"]))
            if medial_state is not None
            else None
        )
        self.coarse_shape_zyx = tuple(int(value) for value in self.image.shape)
        self.affine = affine_matrix(pair.fine.to_coarse_affine_xyz)

    def center(self, coordinate: tuple[int, int, int]) -> np.ndarray:
        fine_center = (
            np.asarray(coordinate, dtype=np.float64) + 0.5
        ) * np.asarray(self.fine_chunks_zyx, dtype=np.float64)
        return transform_xyz(fine_center[::-1][None], self.affine)[0][::-1]

    def evaluate(
        self,
        *,
        candidate_index: int,
        anchor_gate: dict[str, Any] | None,
    ) -> dict[str, Any]:
        coordinate = self.anchors[candidate_index % len(self.anchors)]
        cycle = candidate_index // len(self.anchors)
        origin = _logical_origin(
            center_zyx=self.center(coordinate),
            coarse_shape_zyx=self.coarse_shape_zyx,
            patch_shape_zyx=self.patch_shape_zyx,
            key=(
                f"{self.seed}\x1f{self.pair.record_id}\x1f{coordinate}\x1f{cycle}"
            ),
        )
        image = read_crop(self.image, origin, self.patch_shape_zyx)
        q = read_crop(self.teacher_q, origin, self.patch_shape_zyx).astype(
            np.uint8, copy=False
        )
        valid_u8 = (read_crop(self.target_valid, origin, self.patch_shape_zyx) > 0).astype(
            np.uint8
        )
        valid = valid_u8 > 0
        known_voxels = int(np.count_nonzero(valid))
        positive = (q >= 128) & valid
        positive_voxels = int(np.count_nonzero(positive))
        crest_known_voxels = 0
        crest_voxels = 0
        if self.teacher_crest is not None and self.teacher_crest_valid is not None:
            crest_valid = read_crop(
                self.teacher_crest_valid, origin, self.patch_shape_zyx
            ) > 0
            crest = (
                read_crop(self.teacher_crest, origin, self.patch_shape_zyx) > 0
            ) & crest_valid
            crest_known_voxels = int(np.count_nonzero(crest_valid))
            crest_voxels = int(np.count_nonzero(crest))
        ct_nonzero_fraction = float(np.count_nonzero(image)) / image.size
        known_fraction = known_voxels / valid.size
        center = np.rint(self.center(coordinate)).astype(np.int64)
        local_center = center - np.asarray(origin, dtype=np.int64)
        contained = bool(
            ((local_center >= 0) & (local_center < np.asarray(self.patch_shape_zyx))).all()
        )
        if anchor_gate is None:
            teacher_metrics = scrollfiesta_patch_pred_metrics(positive)
            anchor_gate = {
                "schema": ANCHOR_GATE_SCHEMA,
                "chunk_zyx": list(coordinate),
                "source_candidate_index": candidate_index,
                "teacher_scrollfiesta_metrics": teacher_metrics.to_dict(),
                "accepted": teacher_metrics.reject_kind == "keep",
            }
        accepted = bool(anchor_gate["accepted"])
        reasons: list[str] = []
        if not accepted:
            reasons.append("teacher-scrollfiesta-reject")
        if not contained:
            reasons.append("support-anchor-outside-patch")
        if ct_nonzero_fraction + 1.0e-12 < self.minimum_ct_nonzero_fraction:
            reasons.append("coarse-ct-support")
        if known_fraction + 1.0e-12 < self.minimum_known_fraction:
            reasons.append("known-teacher-support")
        if positive_voxels < self.minimum_positive_voxels:
            reasons.append("teacher-positive-support")
        if self.medial_state is not None:
            if crest_known_voxels <= 0:
                reasons.append("medial-teacher-support")
            if crest_voxels < self.minimum_crest_voxels:
                reasons.append("medial-teacher-positive-support")
        pathology_score = 0.0
        if self.baseline is not None and self.baseline_spec is not None and known_voxels:
            baseline_raw = read_crop(self.baseline, origin, self.patch_shape_zyx)
            baseline_probability = decode_dense_field(baseline_raw, self.baseline_spec)
            baseline = baseline_probability >= self.baseline_spec.threshold
            pathology_score = float(np.not_equal(baseline[valid], positive[valid]).mean())
        return {
            "candidate_index": candidate_index,
            "coordinate": coordinate,
            "origin": origin,
            "known_fraction": known_fraction,
            "positive_fraction_known": positive_voxels / max(1, known_voxels),
            "positive_voxels": positive_voxels,
            "crest_known_voxels": crest_known_voxels,
            "crest_voxels": crest_voxels,
            "ct_nonzero_fraction": ct_nonzero_fraction,
            "pathology_score": pathology_score,
            "anchor_gate": anchor_gate,
            "accepted": not reasons,
            "reasons": reasons,
        }

    def row(self, result: dict[str, Any], accepted_index: int) -> dict[str, Any]:
        coordinate = result["coordinate"]
        patch_id = f"{self.pair.record_id}-{accepted_index:06d}"
        quality = self.atlas_state["identity"]
        support_before = int(quality["fine_support_chunks"])
        support_after = len(self.anchors)
        preparation_version = (
            MEDIAL_ATLAS_PATCH_PREPARATION_VERSION
            if self.medial_state is not None
            else ATLAS_PATCH_PREPARATION_VERSION
        )
        row = {
            "schema": PATCH_SCHEMA,
            "schema_version": 1,
            "patch_id": patch_id,
            "path": f"patches/{patch_id}.atlas",
            "record_id": self.pair.record_id,
            "scroll_id": self.pair.scroll_id,
            "split": "train",
            "origin_zyx": list(result["origin"]),
            "shape_zyx": list(self.patch_shape_zyx),
            "known_fraction": result["known_fraction"],
            "acceptance_min_known_fraction": self.minimum_known_fraction,
            "positive_fraction_known": result["positive_fraction_known"],
            "pathology_score": result["pathology_score"],
            "sampling_pathology_score": result["pathology_score"],
            "scrollfiesta_pred_metrics": None,
            "has_baseline": self.baseline is not None,
            "supervision_source": "official-native-fine-teacher/coarse-atlas",
            "sampling_strategy": "random",
            "preparation_version": preparation_version,
            "native_teacher_min_fine_ct_nonzero_fraction": 0.95,
            "native_teacher_fine_ct_quality_gate_applied": True,
            "native_teacher_support_chunks_before_quality_gate": support_before,
            "native_teacher_support_chunks_after_quality_gate": support_after,
            "native_teacher_support_chunks_excluded_by_quality_gate": (
                support_before - support_after
            ),
            "support_anchor_chunk_zyx": list(coordinate),
            "support_anchor_pool_size": len(self.anchors),
            "support_anchor_candidate_chunks_zyx": [list(coordinate)],
            "ct_nonzero_fraction": result["ct_nonzero_fraction"],
            "archive_bytes": None,
            "archive_sha256": None,
            "registration_decision": {
                "contract": "crossres-local-ct-translation-l0-v1",
                "method": "identity",
                "shift_coarse_zyx": [0, 0, 0],
                "registration_manifest_sha256": self.registration_sha256,
                "base_method": "alignment-metadata-qualified-global-affine",
            },
            "registered_source_quality_gate": {
                "accepted": True,
                "ct_nonzero_fraction": result["ct_nonzero_fraction"],
                "minimum_ct_nonzero_fraction": self.minimum_ct_nonzero_fraction,
                "fine_support_anchor_contained": True,
                "teacher_scrollfiesta_metrics": result["anchor_gate"][
                    "teacher_scrollfiesta_metrics"
                ],
            },
            "target_projection": {
                "contract": ATLAS_PROJECTION_CONTRACT,
                "prefilter_sigma_scale": 0.5,
                "coverage_erosion_fine_vox": 0,
                "maxpool_prefilter": False,
                "erode_filter_margin": True,
                "hard_threshold": 0.5,
                "projection_backend": "cuda-gauss-hermite3-pullback-linf-validity-v1",
                "gaussian_quadrature_order_per_axis": 3,
                "validity_erosion_metric": "linf",
                "atlas_state": str(self.atlas_state_path),
                "atlas_state_sha256": self.atlas_state_sha256,
                "teacher_shift_coarse_zyx": [0, 0, 0],
            },
            "array_source": {
                "schema": (
                    "crossres-coarse-teacher-atlas-patch-source-v2"
                    if self.medial_state is not None
                    else "crossres-coarse-teacher-atlas-patch-source-v1"
                ),
                "catalog": str(self.catalog_path),
                "catalog_sha256": self.catalog_sha256,
                "source_id": self.pair.record_id,
                "teacher_shift_coarse_zyx": [0, 0, 0],
            },
            "atlas_schedule": {
                "candidate_index": result["candidate_index"],
                "anchor_gate_schema": ANCHOR_GATE_SCHEMA,
            },
        }
        if self.medial_state is not None:
            assert self.medial_state_sha256 is not None
            row["target_projection"].update(
                {
                    "medial_surface_contract": VILLA_MEDIAL_SURFACE_CONTRACT,
                    "medial_projection_contract": MEDIAL_MAX_PROJECTION_CONTRACT,
                    "medial_state": str(self.medial_state_path),
                    "medial_state_sha256": self.medial_state_sha256,
                }
            )
            row["atlas_schedule"].update(
                {
                    "crest_known_voxels": result["crest_known_voxels"],
                    "crest_voxels": result["crest_voxels"],
                    "crest_fraction_known": (
                        result["crest_voxels"]
                        / max(1, result["crest_known_voxels"])
                    ),
                }
            )
        return row


def _load_batches(
    path: Path, *, record_id: str, requested: int
) -> list[dict[str, Any]]:
    if requested <= 0:
        raise ValueError("requested batch row count must be positive")
    rows: list[dict[str, Any]] = []
    origins: set[tuple[int, int, int]] = set()
    previous_candidate = -1
    for batch_index, batch in enumerate(sorted(path.glob("*.jsonl"))):
        expected_name = f"{batch_index:06d}.jsonl"
        if batch.name != expected_name:
            raise ValueError(
                f"{path}: corpus batch sequence expected {expected_name}, got {batch.name}"
            )
        batch_rows = _read_jsonl(batch)
        expected_count = min(BATCH_SIZE, requested - len(rows))
        if expected_count <= 0 or len(batch_rows) != expected_count:
            raise ValueError(
                f"{batch}: expected {expected_count} committed rows, got "
                f"{len(batch_rows)}"
            )
        for offset, row in enumerate(batch_rows):
            accepted_index = len(rows) + offset
            candidate_index = int(row.get("atlas_schedule", {}).get("candidate_index", -1))
            origin = tuple(int(value) for value in row.get("origin_zyx", ()))
            if (
                row.get("schema") != PATCH_SCHEMA
                or int(row.get("schema_version", -1)) != 1
                or row.get("record_id") != record_id
                or row.get("patch_id") != f"{record_id}-{accepted_index:06d}"
                or candidate_index <= previous_candidate
                or len(origin) != 3
                or origin in origins
            ):
                raise ValueError(f"{batch}: invalid committed row {accepted_index}")
            previous_candidate = candidate_index
            origins.add(origin)  # type: ignore[arg-type]
        rows.extend(batch_rows)
    return rows


def _assign_sampling_strategies(rows: list[dict[str, Any]]) -> None:
    order = sorted(
        range(len(rows)),
        key=lambda index: (
            -float(rows[index]["pathology_score"]),
            hashlib.sha256(str(rows[index]["patch_id"]).encode()).hexdigest(),
        ),
    )
    high_count = math.ceil(len(rows) / 3)
    high = set(order[:high_count])
    remaining = [index for index in order if index not in high]
    dense = set(
        sorted(
            remaining,
            key=lambda index: (
                -float(rows[index]["positive_fraction_known"]),
                str(rows[index]["patch_id"]),
            ),
        )[: math.ceil(len(rows) / 6)]
    )
    for index, row in enumerate(rows):
        row["sampling_strategy"] = (
            "high-pathology"
            if index in high
            else "dense-positive"
            if index in dense
            else "random"
        )


def build_corpus(plan_path: Path) -> Path:
    plan_path = plan_path.expanduser().resolve()
    plan = _read_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"unsupported atlas corpus plan: {plan_path}")
    base = plan_path.parent
    pair_manifest = _resolve(base, str(plan["pair_manifest"]))
    output = _resolve(base, str(plan.get("output", ".")))
    seed = int(plan.get("seed", 1203))
    patch_shape = tuple(int(value) for value in plan.get("patch_shape_zyx", [192] * 3))
    if len(patch_shape) != 3 or any(value <= 0 for value in patch_shape):
        raise ValueError("patch_shape_zyx must contain three positive values")
    minimum_ct = float(plan.get("minimum_ct_nonzero_fraction", 0.95))
    minimum_known = float(plan.get("minimum_known_fraction", 0.001))
    minimum_positive = int(plan.get("minimum_positive_voxels", 32))
    raw_require_teacher_crest = plan.get("require_teacher_crest", False)
    if not isinstance(raw_require_teacher_crest, bool):
        raise TypeError("require_teacher_crest must be boolean")
    require_teacher_crest = raw_require_teacher_crest
    minimum_crest = int(plan.get("minimum_crest_voxels", 1))
    if minimum_crest <= 0:
        raise ValueError("minimum_crest_voxels must be positive")
    max_cpu_threads = int(plan.get("max_cpu_threads", 16))
    workers = int(plan.get("workers", min(8, max_cpu_threads)))
    if not 1 <= workers <= max_cpu_threads <= 16:
        raise ValueError("workers/max_cpu_threads violate the 16-thread ceiling")
    configure_cpu_budget(
        max_cpu_threads,
        reserve_processes=min(workers, max_cpu_threads - 1),
    )
    pairs = {record.record_id: record for record in load_pair_manifest(pair_manifest)}
    source_plans = plan.get("sources")
    if not isinstance(source_plans, list) or not source_plans:
        raise ValueError("atlas corpus plan requires sources")

    output.mkdir(parents=True, exist_ok=True)
    catalog_path = output / "atlas_catalog.json"
    registration_path = output / "global_registration_commit.json"
    source_inputs: list[
        tuple[
            dict[str, Any],
            VoxelPairRecord,
            Path,
            dict[str, Any],
            dict[str, Any] | None,
        ]
    ] = []
    catalog_sources: dict[str, Any] = {}
    atlas_commits: dict[str, Any] = {}
    for item in source_plans:
        record_id = str(item["record_id"])
        if record_id not in pairs:
            raise ValueError(f"pair manifest lacks atlas source {record_id}")
        pair = pairs[record_id]
        if pair.split != "train":
            raise ValueError(f"atlas source is not train split: {record_id}")
        atlas_root = _resolve(base, str(item.get("atlas", ".")))
        atlas_state = validate_coarse_teacher_atlas(atlas_root)
        medial_state = (
            validate_coarse_teacher_medial_atlas(atlas_root)
            if require_teacher_crest
            else None
        )
        identity = atlas_state["identity"]
        if (
            identity.get("record_id") != record_id
            or identity.get("pair_manifest_sha256") != _sha256(pair_manifest)
        ):
            raise ValueError(f"atlas identity differs from pair record: {record_id}")
        state_path = atlas_root / "atlas_state.json"
        state_sha256 = _sha256(state_path)
        catalog_sources[record_id] = {
            "scroll_id": pair.scroll_id,
            "coarse_image": pair.coarse.image,
            "coarse_baseline": _dense_field_dict(pair.coarse.baseline),
            "teacher_q": str(Path(str(atlas_state["teacher_q"])).resolve()),
            "target_valid": str(Path(str(atlas_state["target_valid"])).resolve()),
            "atlas_state": str(state_path),
            "atlas_state_sha256": state_sha256,
        }
        if medial_state is not None:
            medial_state_path = atlas_root / "medial_state.json"
            catalog_sources[record_id].update(
                {
                    "teacher_crest": str(
                        Path(str(medial_state["teacher_crest"])).resolve()
                    ),
                    "teacher_crest_valid": str(
                        Path(str(medial_state["teacher_crest_valid"])).resolve()
                    ),
                    "medial_state": str(medial_state_path),
                    "medial_state_sha256": _sha256(medial_state_path),
                }
            )
        atlas_commits[record_id] = {
            "state": str(state_path),
            "state_sha256": state_sha256,
        }
        if medial_state is not None:
            atlas_commits[record_id].update(
                {
                    "medial_state": str(atlas_root / "medial_state.json"),
                    "medial_state_sha256": _sha256(
                        atlas_root / "medial_state.json"
                    ),
                }
            )
        source_inputs.append((item, pair, atlas_root, atlas_state, medial_state))
    catalog = {
        "schema": MEDIAL_CATALOG_SCHEMA if require_teacher_crest else CATALOG_SCHEMA,
        "state": "complete",
        "pair_manifest": str(pair_manifest),
        "pair_manifest_sha256": _sha256(pair_manifest),
        "sources": catalog_sources,
    }
    _atomic_json(catalog_path, catalog)
    catalog_sha256 = _sha256(catalog_path)
    registration_commit = {
        "schema": REGISTRATION_SCHEMA,
        "state": "complete",
        "method": "alignment-metadata-qualified-global-affine",
        "pair_manifest": str(pair_manifest),
        "pair_manifest_sha256": _sha256(pair_manifest),
        "atlas_commits": atlas_commits,
        "shift_coarse_zyx": [0, 0, 0],
    }
    _atomic_json(registration_path, registration_commit)
    registration_sha256 = _sha256(registration_path)
    preparation_version = (
        MEDIAL_ATLAS_PATCH_PREPARATION_VERSION
        if require_teacher_crest
        else ATLAS_PATCH_PREPARATION_VERSION
    )
    identity = {
        "preparation_version": preparation_version,
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "pair_manifest": str(pair_manifest),
        "pair_manifest_sha256": _sha256(pair_manifest),
        "catalog": str(catalog_path),
        "catalog_sha256": catalog_sha256,
        "registration_commit": str(registration_path),
        "registration_commit_sha256": registration_sha256,
        "scrollfiesta_pred_metrics_contract": SCROLLFIESTA_PRED_METRICS_CONTRACT,
        "options": {
            "min_known_fraction": minimum_known,
            "native_teacher_min_known_fraction": minimum_known,
            "native_teacher_min_fine_ct_nonzero_fraction": 0.95,
            "minimum_ct_nonzero_fraction": minimum_ct,
            "minimum_positive_voxels": minimum_positive,
            "require_teacher_crest": require_teacher_crest,
            "minimum_crest_voxels": minimum_crest,
            "patch_shape_zyx": list(patch_shape),
            "seed": seed,
            "max_cpu_threads": max_cpu_threads,
            "workers": workers,
        },
    }
    if not require_teacher_crest:
        identity["options"].pop("require_teacher_crest")
        identity["options"].pop("minimum_crest_voxels")
    state_path = output / "prepare_state.json"
    if state_path.is_file():
        state = _read_json(state_path)
        if state.get("identity") != identity:
            raise ValueError("existing atlas corpus identity differs")
    _atomic_json(
        state_path,
        {"schema": STATE_SCHEMA, "state": "running", "completed": 0, "identity": identity},
    )

    all_rows: list[dict[str, Any]] = []
    source_summaries: dict[str, Any] = {}
    for item, pair, atlas_root, atlas_state, medial_state in source_inputs:
        requested = int(item["patch_count"])
        if requested <= 0:
            raise ValueError(f"{pair.record_id}: patch_count must be positive")
        batch_root = output / "rows" / pair.record_id
        batch_root.mkdir(parents=True, exist_ok=True)
        rows = _load_batches(batch_root, record_id=pair.record_id, requested=requested)
        if len(rows) > requested:
            raise ValueError(f"{pair.record_id}: committed batches exceed requested count")
        used_origins = {tuple(int(value) for value in row["origin_zyx"]) for row in rows}
        candidate_index = max(
            (int(row["atlas_schedule"]["candidate_index"]) for row in rows),
            default=-1,
        ) + 1
        gate_path = output / "anchor_gates" / f"{pair.record_id}.jsonl"
        gates = {
            tuple(int(value) for value in row["chunk_zyx"]): row
            for row in (_read_jsonl(gate_path) if gate_path.is_file() else [])
        }
        context = SourceContext(
            pair=pair,
            atlas_root=atlas_root,
            atlas_state=atlas_state,
            medial_state=medial_state,
            catalog_path=catalog_path,
            catalog_sha256=catalog_sha256,
            registration_sha256=registration_sha256,
            patch_shape_zyx=patch_shape,  # type: ignore[arg-type]
            minimum_ct_nonzero_fraction=minimum_ct,
            minimum_known_fraction=minimum_known,
            minimum_positive_voxels=minimum_positive,
            minimum_crest_voxels=minimum_crest,
            seed=seed,
        )
        attempts = candidate_index
        while len(rows) < requested:
            needed = min(BATCH_SIZE, requested - len(rows))
            accepted: list[dict[str, Any]] = []
            while len(accepted) < needed:
                width = max(workers * 4, needed - len(accepted))
                indices = list(range(candidate_index, candidate_index + width))

                def evaluate(
                    index: int,
                    *,
                    source_context: SourceContext = context,
                    anchor_gates: dict[tuple[int, int, int], dict[str, Any]] = gates,
                ) -> dict[str, Any]:
                    coordinate = source_context.anchors[
                        index % len(source_context.anchors)
                    ]
                    return source_context.evaluate(
                        candidate_index=index,
                        anchor_gate=anchor_gates.get(coordinate),
                    )

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    results = list(executor.map(evaluate, indices))
                cutoff = indices[-1] + 1
                for result in results:
                    coordinate = result["coordinate"]
                    if coordinate not in gates:
                        gates[coordinate] = result["anchor_gate"]
                    if not result["accepted"] or result["origin"] in used_origins:
                        continue
                    accepted_index = len(rows) + len(accepted)
                    accepted.append(context.row(result, accepted_index))
                    used_origins.add(result["origin"])
                    cutoff = int(result["candidate_index"]) + 1
                    if len(accepted) == needed:
                        break
                candidate_index = cutoff
                attempts = max(attempts, candidate_index)
                if candidate_index > requested * 100:
                    raise RuntimeError(
                        f"{pair.record_id}: could not fill atlas corpus after "
                        f"{candidate_index:,} deterministic candidates"
                    )
            batch_index = len(rows) // BATCH_SIZE
            batch_path = batch_root / f"{batch_index:06d}.jsonl"
            _atomic_jsonl(batch_path, accepted)
            rows.extend(accepted)
            _atomic_jsonl(
                gate_path,
                [gates[key] for key in sorted(gates)],
            )
            _atomic_json(
                state_path,
                {
                    "schema": STATE_SCHEMA,
                    "state": "running",
                    "completed": len(all_rows) + len(rows),
                    "active_record_id": pair.record_id,
                    "identity": identity,
                },
            )
            print(
                f"atlas patches {pair.record_id}: {len(rows):,}/{requested:,} "
                f"anchors_gated={len(gates):,}",
                flush=True,
            )
        all_rows.extend(rows)
        source_summary = {
            "scroll_id": pair.scroll_id,
            "patches": len(rows),
            "distinct_origins": len(used_origins),
            "fine_anchor_pool": len(context.anchors),
            "fine_anchors_gated": len(gates),
            "fine_anchors_accepted": sum(bool(value["accepted"]) for value in gates.values()),
            "deterministic_candidates_examined": attempts,
            "atlas_state_sha256": context.atlas_state_sha256,
        }
        if context.medial_state_sha256 is not None:
            source_summary["medial_state_sha256"] = context.medial_state_sha256
        source_summaries[pair.record_id] = source_summary

    _assign_sampling_strategies(all_rows)
    manifest = output / "patches.jsonl"
    _atomic_jsonl(manifest, all_rows)
    manifest_sha256 = _sha256(manifest)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "state": "complete",
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "patches": len(all_rows),
        "distinct_coarse_origins": len(
            {
                (str(row["record_id"]), *tuple(row["origin_zyx"]))
                for row in all_rows
            }
        ),
        "sources": source_summaries,
        "sampling_strategies": {
            name: sum(row["sampling_strategy"] == name for row in all_rows)
            for name in ("high-pathology", "dense-positive", "random")
        },
        "identity": identity,
    }
    _atomic_json(output / "summary.json", summary)
    _atomic_json(
        state_path,
        {
            "schema": STATE_SCHEMA,
            "state": "complete",
            "completed": len(all_rows),
            "expected": sum(int(item["patch_count"]) for item in source_plans),
            "manifest_sha256": manifest_sha256,
            "summary_sha256": _sha256(output / "summary.json"),
            "identity": identity,
        },
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    print(build_corpus(args.plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
