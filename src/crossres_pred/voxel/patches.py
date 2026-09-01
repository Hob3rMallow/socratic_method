from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from ..pathmap import remap_embedded_path
from .coarse_teacher_atlas import ATLAS_PROJECTION_CONTRACT
from .io import decode_dense_field, open_volume, read_crop
from .medial import (
    MEDIAL_MAX_PROJECTION_CONTRACT,
    VILLA_MEDIAL_SURFACE_CONTRACT,
)
from .medial_bridges import (
    DYNAMIC_MEDIAL_CONNECTIVITY_ATLAS_SCHEMA,
    PINNED_MEDIAL_BRIDGE_ATLAS_SCHEMA,
)
from .resources import configure_cpu_budget
from .schema import DenseFieldSpec
from .scrollfiesta_metrics import (
    SCROLLFIESTA_PRED_METRICS_CONTRACT,
    ScrollFiestaPredMetrics,
    scrollfiesta_metrics_close,
    scrollfiesta_patch_pred_metrics,
)

PATCH_SCHEMA = "crossres-voxel-patch-v1"
PATCH_PREPARATION_VERSION = "projection-cache-locality-v9"
ANTIALIAS_PATCH_PREPARATION_VERSION = "antialias-pullback-gh3-v11"
ATLAS_PATCH_PREPARATION_VERSION = "coarse-atlas-antialias-gh3-v1"
MEDIAL_ATLAS_PATCH_PREPARATION_VERSION = "coarse-atlas-antialias-gh3-villa-medial-v2"
ATLAS_PATCH_PREPARATION_VERSIONS = {
    ATLAS_PATCH_PREPARATION_VERSION,
    MEDIAL_ATLAS_PATCH_PREPARATION_VERSION,
}
SUPPORTED_PATCH_PREPARATION_VERSIONS = {
    PATCH_PREPARATION_VERSION,
    ANTIALIAS_PATCH_PREPARATION_VERSION,
    ATLAS_PATCH_PREPARATION_VERSION,
    MEDIAL_ATLAS_PATCH_PREPARATION_VERSION,
}
ANTIALIAS_TARGET_PROJECTION_CONTRACT = "antialias-pullback-gh3-v2"
M7_PATHOLOGY_OVERLAY_SCHEMA = "crossres-m7-pathology-overlay-v1"
M7_CT_LOWER = 0.0
M7_CT_UPPER = 212.0
M7_CT_MEAN = 87.54424285888672
M7_CT_STD = 47.74376678466797


@dataclass(frozen=True)
class PatchRecord:
    patch_id: str
    path: Path
    record_id: str
    scroll_id: str
    split: str
    origin_zyx: tuple[int, int, int]
    shape_zyx: tuple[int, int, int]
    known_fraction: float
    acceptance_min_known_fraction: float
    positive_fraction_known: float
    pathology_score: float
    sampling_pathology_score: float | None
    scrollfiesta_pred_metrics: ScrollFiestaPredMetrics | None
    has_baseline: bool
    supervision_source: str
    sampling_strategy: str
    preparation_version: str
    native_teacher_min_fine_ct_nonzero_fraction: float | None
    native_teacher_fine_ct_quality_gate_applied: bool | None
    native_teacher_support_chunks_before_quality_gate: int | None
    native_teacher_support_chunks_after_quality_gate: int | None
    native_teacher_support_chunks_excluded_by_quality_gate: int | None
    support_anchor_chunk_zyx: tuple[int, int, int] | None
    support_anchor_pool_size: int | None
    support_anchor_candidate_chunks_zyx: tuple[tuple[int, int, int], ...] | None
    ct_nonzero_fraction: float
    archive_bytes: int | None
    archive_sha256: str | None
    pathology_mining: dict[str, Any] | None
    target_projection: dict[str, Any] | None
    registration_decision: dict[str, Any] | None
    registered_source_quality_gate: dict[str, Any] | None
    array_source: dict[str, Any] | None

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, base: Path) -> PatchRecord:
        if value.get("schema") != PATCH_SCHEMA:
            raise ValueError(f"patch schema must be {PATCH_SCHEMA!r}")
        path = remap_embedded_path(str(value["path"]))
        if not path.is_absolute():
            path = base / path
        origin = tuple(int(item) for item in value["origin_zyx"])
        shape = tuple(int(item) for item in value["shape_zyx"])
        if len(origin) != 3 or len(shape) != 3 or any(item <= 0 for item in shape):
            raise ValueError("origin_zyx/shape_zyx must be three-element arrays")
        split = str(value["split"]).lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"invalid patch split {split!r}")
        supervision_source = str(value.get("supervision_source", "unspecified"))
        has_baseline = bool(value.get("has_baseline", False))
        raw_anchor = value.get("support_anchor_chunk_zyx")
        support_anchor = (
            tuple(int(item) for item in raw_anchor) if raw_anchor is not None else None
        )
        if support_anchor is not None and (
            len(support_anchor) != 3 or any(item < 0 for item in support_anchor)
        ):
            raise ValueError("support_anchor_chunk_zyx must be three non-negative ints")
        raw_pool_size = value.get("support_anchor_pool_size")
        support_anchor_pool_size = (
            int(raw_pool_size) if raw_pool_size is not None else None
        )
        if support_anchor_pool_size is not None and support_anchor_pool_size <= 0:
            raise ValueError("support_anchor_pool_size must be positive")
        if (support_anchor is None) != (support_anchor_pool_size is None):
            raise ValueError(
                "support anchor coordinate and pool size must appear together"
            )
        preparation_version = str(value.get("preparation_version", "legacy"))
        raw_quality_threshold = value.get("native_teacher_min_fine_ct_nonzero_fraction")
        quality_threshold = (
            float(raw_quality_threshold) if raw_quality_threshold is not None else None
        )
        raw_quality_applied = value.get("native_teacher_fine_ct_quality_gate_applied")
        quality_applied = (
            raw_quality_applied if isinstance(raw_quality_applied, bool) else None
        )
        raw_quality_counts = (
            value.get("native_teacher_support_chunks_before_quality_gate"),
            value.get("native_teacher_support_chunks_after_quality_gate"),
            value.get("native_teacher_support_chunks_excluded_by_quality_gate"),
        )
        quality_counts = tuple(
            int(item) if item is not None else None for item in raw_quality_counts
        )
        is_current_native = (
            preparation_version in SUPPORTED_PATCH_PREPARATION_VERSIONS
            and "native-fine-teacher" in supervision_source
        )
        if is_current_native:
            if quality_threshold is None or not 0 <= quality_threshold <= 1:
                raise ValueError(
                    "current native patch must record a fine-CT quality threshold"
                )
            if quality_applied is None:
                raise ValueError(
                    "current native patch must record whether its fine-CT gate ran"
                )
            before, after, excluded = quality_counts
            if (
                before is None
                or after is None
                or excluded is None
                or before <= 0
                or after <= 0
                or excluded < 0
                or before != after + excluded
                or (not quality_applied and excluded != 0)
            ):
                raise ValueError(
                    "current native patch has inconsistent fine-CT support counts"
                )
        raw_scrollfiesta_metrics = value.get("scrollfiesta_pred_metrics")
        if raw_scrollfiesta_metrics is None:
            scrollfiesta_metrics = None
        elif isinstance(raw_scrollfiesta_metrics, dict):
            scrollfiesta_metrics = ScrollFiestaPredMetrics.from_dict(
                raw_scrollfiesta_metrics
            )
        else:
            raise ValueError("ScrollFiesta prediction metrics must be an object")
        raw_pathology_mining = value.get("pathology_mining")
        if raw_pathology_mining is None:
            pathology_mining = None
        elif isinstance(raw_pathology_mining, dict):
            pathology_mining = json.loads(
                json.dumps(raw_pathology_mining, sort_keys=True)
            )
            required_hashes = (
                "mining_identity_sha256",
                "score_journal_sha256",
                "score_sha256",
            )
            if (
                pathology_mining.get("schema") != M7_PATHOLOGY_OVERLAY_SCHEMA
                or pathology_mining.get("score_schema")
                != "crossres-m7-pathology-score-v2"
                or pathology_mining.get("inference_contract")
                != "released-m7-local-192-fp16-no-tta-th0.2-center128-v2"
                or any(
                    len(str(pathology_mining.get(name, ""))) != 64
                    for name in required_hashes
                )
                or not Path(
                    str(pathology_mining.get("source_manifest", ""))
                ).is_absolute()
                or int(pathology_mining.get("source_row_index", -1)) < 0
                or pathology_mining.get("archive_sha256") != value.get("archive_sha256")
                or not isinstance(pathology_mining.get("selected_high_pathology"), bool)
                or not str(pathology_mining.get("base_sampling_strategy", ""))
            ):
                raise ValueError("invalid local-m7 pathology mining provenance")
            teacher_metrics = ScrollFiestaPredMetrics.from_dict(
                pathology_mining["scrollfiesta_teacher_metrics"]
            )
            if (
                scrollfiesta_metrics is None
                or teacher_metrics.window_origin_zyx
                != scrollfiesta_metrics.window_origin_zyx
                or teacher_metrics.window_shape_zyx
                != scrollfiesta_metrics.window_shape_zyx
                or not _close_fraction(
                    float(pathology_mining["pathology_score"]),
                    float(value.get("pathology_score", math.nan)),
                )
            ):
                raise ValueError("local-m7 pathology overlay differs from patch row")
            selected = pathology_mining["selected_high_pathology"]
            sampling = str(value.get("sampling_strategy", "unspecified"))
            base_sampling = str(pathology_mining["base_sampling_strategy"])
            if selected != (sampling == "mined-high-pathology") or (
                not selected and sampling != base_sampling
            ):
                raise ValueError("local-m7 pathology selection flag is inconsistent")
        else:
            raise ValueError("pathology_mining must be an object")
        if has_baseline and pathology_mining is not None:
            raise ValueError("published-baseline patch cannot have a local-m7 overlay")
        if (
            preparation_version in SUPPORTED_PATCH_PREPARATION_VERSIONS
            and preparation_version not in ATLAS_PATCH_PREPARATION_VERSIONS
            and (
                (has_baseline and scrollfiesta_metrics is None)
                or (
                    not has_baseline
                    and scrollfiesta_metrics is not None
                    and pathology_mining is None
                )
            )
        ):
            raise ValueError(
                "current patch ScrollFiesta metrics must match baseline availability"
            )
        raw_candidates = value.get("support_anchor_candidate_chunks_zyx")
        support_anchor_candidates = (
            tuple(
                tuple(int(item) for item in coordinate) for coordinate in raw_candidates
            )
            if raw_candidates is not None
            else None
        )
        if support_anchor_candidates is not None and (
            not support_anchor_candidates
            or any(
                len(coordinate) != 3 or any(item < 0 for item in coordinate)
                for coordinate in support_anchor_candidates
            )
            or len(set(support_anchor_candidates)) != len(support_anchor_candidates)
        ):
            raise ValueError(
                "support_anchor_candidate_chunks_zyx must contain unique "
                "non-negative 3-D coordinates"
            )
        if (
            preparation_version in SUPPORTED_PATCH_PREPARATION_VERSIONS
            and support_anchor is not None
            and (
                support_anchor_candidates is None
                or support_anchor not in support_anchor_candidates
            )
        ):
            raise ValueError(
                "current patch anchor must belong to its candidate anchors"
            )
        if support_anchor is None and support_anchor_candidates is not None:
            raise ValueError("candidate anchors require a selected support anchor")
        if (
            support_anchor_candidates is not None
            and support_anchor_pool_size is not None
            and len(support_anchor_candidates) > support_anchor_pool_size
        ):
            raise ValueError("candidate anchor count exceeds support anchor pool size")
        raw_acceptance = value.get("acceptance_min_known_fraction")
        if (
            preparation_version in SUPPORTED_PATCH_PREPARATION_VERSIONS
            and raw_acceptance is None
        ):
            raise ValueError(
                "current patch rows must record acceptance_min_known_fraction"
            )
        acceptance_min_known_fraction = (
            float(raw_acceptance) if raw_acceptance is not None else 0.0
        )
        if not 0 <= acceptance_min_known_fraction <= 1:
            raise ValueError("acceptance_min_known_fraction must be in [0, 1]")
        raw_projection = value.get("target_projection")
        if raw_projection is None:
            target_projection = None
        elif isinstance(raw_projection, dict):
            target_projection = json.loads(json.dumps(raw_projection, sort_keys=True))
        else:
            raise ValueError("target_projection must be an object")
        if preparation_version == ANTIALIAS_PATCH_PREPARATION_VERSION:
            if (
                target_projection is None
                or target_projection.get("contract")
                != ANTIALIAS_TARGET_PROJECTION_CONTRACT
                or float(target_projection.get("prefilter_sigma_scale", -1.0)) != 0.5
                or int(target_projection.get("coverage_erosion_fine_vox", -1)) != 0
                or target_projection.get("maxpool_prefilter") is not False
                or target_projection.get("erode_filter_margin") is not True
                or float(target_projection.get("hard_threshold", -1.0)) != 0.5
                or target_projection.get("projection_backend")
                != "cuda-gauss-hermite3-pullback-linf-validity-v1"
                or int(target_projection.get("gaussian_quadrature_order_per_axis", -1))
                != 3
                or target_projection.get("validity_erosion_metric") != "linf"
                or len(str(target_projection.get("source_archive_sha256", ""))) != 64
            ):
                raise ValueError("invalid anti-aliased target projection provenance")
        elif preparation_version in ATLAS_PATCH_PREPARATION_VERSIONS:
            if (
                target_projection is None
                or target_projection.get("contract") != ATLAS_PROJECTION_CONTRACT
                or float(target_projection.get("prefilter_sigma_scale", -1.0)) != 0.5
                or int(target_projection.get("coverage_erosion_fine_vox", -1)) != 0
                or target_projection.get("maxpool_prefilter") is not False
                or target_projection.get("erode_filter_margin") is not True
                or float(target_projection.get("hard_threshold", -1.0)) != 0.5
                or target_projection.get("projection_backend")
                != "cuda-gauss-hermite3-pullback-linf-validity-v1"
                or int(target_projection.get("gaussian_quadrature_order_per_axis", -1))
                != 3
                or target_projection.get("validity_erosion_metric") != "linf"
                or len(str(target_projection.get("atlas_state_sha256", ""))) != 64
            ):
                raise ValueError("invalid coarse-atlas target projection provenance")
            if preparation_version == MEDIAL_ATLAS_PATCH_PREPARATION_VERSION and (
                target_projection.get("medial_surface_contract")
                != VILLA_MEDIAL_SURFACE_CONTRACT
                or target_projection.get("medial_projection_contract")
                != MEDIAL_MAX_PROJECTION_CONTRACT
                or len(str(target_projection.get("medial_state_sha256", ""))) != 64
            ):
                raise ValueError("invalid coarse-atlas medial provenance")
        elif target_projection is not None:
            raise ValueError("legacy patch cannot declare anti-aliased projection")
        raw_registration = value.get("registration_decision")
        raw_registered_gate = value.get("registered_source_quality_gate")
        if (raw_registration is None) != (raw_registered_gate is None):
            raise ValueError(
                "registration decision and registered-source quality gate must "
                "appear together"
            )
        registration_decision: dict[str, Any] | None = None
        registered_source_quality_gate: dict[str, Any] | None = None
        if raw_registration is not None:
            if not isinstance(raw_registration, dict) or not isinstance(
                raw_registered_gate, dict
            ):
                raise ValueError("registered-source provenance must contain objects")
            registration_decision = json.loads(
                json.dumps(raw_registration, sort_keys=True)
            )
            registered_source_quality_gate = json.loads(
                json.dumps(raw_registered_gate, sort_keys=True)
            )
            method = str(registration_decision.get("method", ""))
            raw_shift = registration_decision.get("shift_coarse_zyx")
            manifest_sha256 = str(
                registration_decision.get("registration_manifest_sha256", "")
            )
            try:
                shift = tuple(int(item) for item in raw_shift)
            except (TypeError, ValueError):
                shift = ()
            if (
                registration_decision.get("contract")
                != "crossres-local-ct-translation-l0-v1"
                or method not in {"identity", "local-ct-translation"}
                or len(shift) != 3
                or len(manifest_sha256) != 64
                or registered_source_quality_gate.get("accepted") is not True
                or registered_source_quality_gate.get("fine_support_anchor_contained")
                is not True
            ):
                raise ValueError("invalid registered-source provenance")
            try:
                registered_ct = float(
                    registered_source_quality_gate["ct_nonzero_fraction"]
                )
                registered_minimum = float(
                    registered_source_quality_gate["minimum_ct_nonzero_fraction"]
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("invalid registered-source CT quality gate") from error
            row_ct = float(value.get("ct_nonzero_fraction", math.nan))
            if (
                not math.isfinite(registered_ct)
                or not math.isfinite(registered_minimum)
                or not 0 <= registered_minimum <= registered_ct <= 1
                or not _close_fraction(registered_ct, row_ct)
            ):
                raise ValueError("invalid registered-source CT quality gate")
            raw_center = registered_source_quality_gate.get(
                "fine_support_anchor_local_center_zyx"
            )
            raw_source_shape = registered_source_quality_gate.get("source_shape_zyx")
            if method == "local-ct-translation":
                try:
                    center = tuple(int(item) for item in raw_center)
                    source_shape = tuple(int(item) for item in raw_source_shape)
                except (TypeError, ValueError):
                    center = ()
                    source_shape = ()
                if (
                    len(center) != 3
                    or len(source_shape) != 3
                    or any(size <= 0 for size in source_shape)
                    or any(
                        item < 0 or item >= size
                        for item, size in zip(center, source_shape, strict=True)
                    )
                ):
                    raise ValueError("invalid registered-source anchor containment")
            elif raw_center is not None or raw_source_shape is not None:
                raise ValueError("identity registration cannot declare a local anchor")
            if preparation_version == ANTIALIAS_PATCH_PREPARATION_VERSION:
                projected_registration = (
                    target_projection.get("patch_registration")
                    if isinstance(target_projection, dict)
                    else None
                )
                if (
                    not isinstance(projected_registration, dict)
                    or projected_registration.get("contract")
                    != registration_decision.get("contract")
                    or projected_registration.get("method") != method
                    or tuple(projected_registration.get("shift_coarse_zyx", ()))
                    != shift
                    or projected_registration.get("map_manifest_sha256")
                    != manifest_sha256
                ):
                    raise ValueError(
                        "target projection differs from registered-source provenance"
                    )
        pathology_score = float(value.get("pathology_score", 0.0))
        raw_sampling_pathology_score = value.get("sampling_pathology_score")
        sampling_pathology_score = (
            float(raw_sampling_pathology_score)
            if raw_sampling_pathology_score is not None
            else None
        )
        if sampling_pathology_score is not None and (
            not math.isfinite(sampling_pathology_score)
            or not 0 <= sampling_pathology_score <= 1
        ):
            raise ValueError("sampling_pathology_score must be finite and in [0, 1]")
        raw_array_source = value.get("array_source")
        array_source: dict[str, Any] | None = None
        if raw_array_source is not None:
            if not isinstance(raw_array_source, dict):
                raise ValueError("array_source must be an object")
            array_source = json.loads(json.dumps(raw_array_source, sort_keys=True))
            catalog = remap_embedded_path(str(array_source.get("catalog", "")))
            if not catalog.is_absolute():
                catalog = base / catalog
            raw_shift = array_source.get("teacher_shift_coarse_zyx")
            try:
                array_shift = tuple(int(item) for item in raw_shift)
            except (TypeError, ValueError):
                array_shift = ()
            if (
                preparation_version not in ATLAS_PATCH_PREPARATION_VERSIONS
                or array_source.get("schema")
                != (
                    "crossres-coarse-teacher-atlas-patch-source-v2"
                    if preparation_version == MEDIAL_ATLAS_PATCH_PREPARATION_VERSION
                    else "crossres-coarse-teacher-atlas-patch-source-v1"
                )
                or not str(array_source.get("source_id", ""))
                or len(str(array_source.get("catalog_sha256", ""))) != 64
                or len(array_shift) != 3
            ):
                raise ValueError("invalid coarse-atlas patch array source")
            array_source["catalog"] = str(catalog.resolve())
            array_source["teacher_shift_coarse_zyx"] = list(array_shift)
        elif preparation_version in ATLAS_PATCH_PREPARATION_VERSIONS:
            raise ValueError("coarse-atlas patch has no array source")
        if array_source is not None and registration_decision is not None:
            declared_shift = tuple(
                int(item) for item in array_source["teacher_shift_coarse_zyx"]
            )
            registration_shift = tuple(
                int(item) for item in registration_decision["shift_coarse_zyx"]
            )
            projection_shift = tuple(
                int(item)
                for item in (target_projection or {}).get(
                    "teacher_shift_coarse_zyx", ()
                )
            )
            if (
                declared_shift != registration_shift
                or projection_shift != registration_shift
            ):
                raise ValueError("coarse-atlas source shift differs from registration")
        return cls(
            patch_id=str(value["patch_id"]),
            path=path.resolve(),
            record_id=str(value["record_id"]),
            scroll_id=str(value["scroll_id"]),
            split=split,
            origin_zyx=origin,  # type: ignore[arg-type]
            shape_zyx=shape,  # type: ignore[arg-type]
            known_fraction=float(value["known_fraction"]),
            acceptance_min_known_fraction=acceptance_min_known_fraction,
            positive_fraction_known=float(value["positive_fraction_known"]),
            pathology_score=pathology_score,
            sampling_pathology_score=sampling_pathology_score,
            scrollfiesta_pred_metrics=scrollfiesta_metrics,
            has_baseline=has_baseline,
            supervision_source=supervision_source,
            sampling_strategy=str(value.get("sampling_strategy", "unspecified")),
            preparation_version=preparation_version,
            native_teacher_min_fine_ct_nonzero_fraction=quality_threshold,
            native_teacher_fine_ct_quality_gate_applied=quality_applied,
            native_teacher_support_chunks_before_quality_gate=quality_counts[0],
            native_teacher_support_chunks_after_quality_gate=quality_counts[1],
            native_teacher_support_chunks_excluded_by_quality_gate=quality_counts[2],
            support_anchor_chunk_zyx=support_anchor,  # type: ignore[arg-type]
            support_anchor_pool_size=support_anchor_pool_size,
            support_anchor_candidate_chunks_zyx=support_anchor_candidates,
            ct_nonzero_fraction=float(value.get("ct_nonzero_fraction", 0.0)),
            archive_bytes=(
                int(value["archive_bytes"])
                if value.get("archive_bytes") is not None
                else None
            ),
            archive_sha256=(
                str(value["archive_sha256"])
                if value.get("archive_sha256") is not None
                else None
            ),
            pathology_mining=pathology_mining,
            target_projection=target_projection,
            registration_decision=registration_decision,
            registered_source_quality_gate=registered_source_quality_gate,
            array_source=array_source,
        )


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected an object")
            yield value


def load_patch_manifest(path: str | Path) -> list[PatchRecord]:
    source = Path(path).expanduser().resolve()
    rows = [
        PatchRecord.from_dict(value, base=source.parent)
        for value in _iter_jsonl(source)
    ]
    if not rows:
        raise ValueError(f"{source}: no patches")
    ids = [row.patch_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{source}: duplicate patch IDs")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _open_pinned_medial_bridge_atlas(
    state_path_text: str,
    state_sha256: str,
    manifest_path_text: str,
    manifest_sha256: str,
) -> tuple[Any, str, frozenset[str]]:
    state_path = Path(state_path_text)
    manifest_path = Path(manifest_path_text)
    if not state_path.is_file() or _sha256(state_path) != state_sha256:
        raise ValueError(f"pinned medial bridge state changed: {state_path}")
    if not manifest_path.is_file() or _sha256(manifest_path) != manifest_sha256:
        raise ValueError(
            f"pinned medial bridge training manifest changed: {manifest_path}"
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    identity = state.get("identity")
    eligible_patch_ids = state.get("eligible_patch_ids")
    if (
        state.get("schema") != PINNED_MEDIAL_BRIDGE_ATLAS_SCHEMA
        or state.get("state") != "complete"
        or not isinstance(identity, dict)
        or identity.get("heldout_gate_used_for_construction") is not False
        or identity.get("construction_inputs") != "training-manifest-boxes-only"
        or remap_embedded_path(str(identity.get("training_manifest", ""))).resolve()
        != manifest_path.resolve()
        or identity.get("training_manifest_sha256") != manifest_sha256
        or not str(identity.get("record_id", ""))
        or not isinstance(eligible_patch_ids, list)
        or not eligible_patch_ids
        or not all(isinstance(value, str) and value for value in eligible_patch_ids)
        or len(eligible_patch_ids) != len(set(eligible_patch_ids))
        or state.get("dtype") != "uint16"
    ):
        raise ValueError(f"invalid pinned medial bridge state: {state_path}")
    volume = open_volume(str(state["bridge_ids"]))
    if (
        tuple(int(value) for value in volume.shape)
        != tuple(int(value) for value in state["shape_zyx"])
        or tuple(int(value) for value in volume.chunks)
        != tuple(int(value) for value in state["chunks_zyx"])
        or np.dtype(volume.dtype) != np.dtype(np.uint16)
    ):
        raise ValueError(f"pinned medial bridge array changed: {state_path}")
    return volume, str(identity["record_id"]), frozenset(eligible_patch_ids)


@lru_cache(maxsize=8)
def _open_dynamic_medial_connectivity_atlas(
    state_path_text: str,
    state_sha256: str,
    manifest_path_text: str,
    manifest_sha256: str,
) -> tuple[tuple[Any, Any, Any], str, dict[str, frozenset[int]]]:
    state_path = Path(state_path_text)
    manifest_path = Path(manifest_path_text)
    if not state_path.is_file() or _sha256(state_path) != state_sha256:
        raise ValueError(f"dynamic medial connectivity state changed: {state_path}")
    if not manifest_path.is_file() or _sha256(manifest_path) != manifest_sha256:
        raise ValueError(
            f"dynamic medial connectivity training manifest changed: {manifest_path}"
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    identity = state.get("identity")
    eligible_patch_ids = state.get("eligible_patch_ids")
    patch_event_ids = state.get("patch_event_ids")
    if (
        state.get("schema") != DYNAMIC_MEDIAL_CONNECTIVITY_ATLAS_SCHEMA
        or state.get("state") != "complete"
        or not isinstance(identity, dict)
        or identity.get("heldout_gate_used_for_construction") is not False
        or identity.get("construction_inputs") != "training-manifest-boxes-only"
        or remap_embedded_path(str(identity.get("training_manifest", ""))).resolve()
        != manifest_path.resolve()
        or identity.get("training_manifest_sha256") != manifest_sha256
        or not str(identity.get("record_id", ""))
        or not isinstance(eligible_patch_ids, list)
        or not eligible_patch_ids
        or not all(isinstance(value, str) and value for value in eligible_patch_ids)
        or len(eligible_patch_ids) != len(set(eligible_patch_ids))
        or not isinstance(patch_event_ids, dict)
        or set(patch_event_ids) != set(eligible_patch_ids)
        or not isinstance(state.get("event_count"), int)
        or state["event_count"] <= 0
        or state.get("fully_owned_event_count") != state["event_count"]
        or state.get("dtypes")
        != {
            "event_ids": "uint16",
            "pin_membership": "uint8-bitset",
            "free_anchors": "uint8-binary",
        }
        or not isinstance(state.get("maximum_propagation_steps"), int)
        or state["maximum_propagation_steps"] <= 0
        or state.get("maximum_required_connectivity_steps", 0)
        > state["maximum_propagation_steps"]
    ):
        raise ValueError(f"invalid dynamic medial connectivity state: {state_path}")
    normalized_patch_event_ids: dict[str, frozenset[int]] = {}
    for patch_id, raw_identifiers in patch_event_ids.items():
        if (
            not isinstance(raw_identifiers, list)
            or not raw_identifiers
            or not all(
                isinstance(identifier, int) and 0 < identifier <= state["event_count"]
                for identifier in raw_identifiers
            )
            or len(raw_identifiers) != len(set(raw_identifiers))
        ):
            raise ValueError(
                f"invalid dynamic medial connectivity ownership: {state_path}"
            )
        normalized_patch_event_ids[patch_id] = frozenset(raw_identifiers)
    if set().union(*normalized_patch_event_ids.values()) != set(
        range(1, state["event_count"] + 1)
    ):
        raise ValueError(
            f"dynamic medial connectivity ownership is incomplete: {state_path}"
        )
    volumes = (
        open_volume(str(state["event_ids"])),
        open_volume(str(state["pin_membership"])),
        open_volume(str(state["free_anchors"])),
    )
    expected_shape = tuple(int(value) for value in state["shape_zyx"])
    expected_chunks = tuple(int(value) for value in state["chunks_zyx"])
    expected_dtypes = (np.dtype(np.uint16), np.dtype(np.uint8), np.dtype(np.uint8))
    if any(
        tuple(int(value) for value in volume.shape) != expected_shape
        or tuple(int(value) for value in volume.chunks) != expected_chunks
        or np.dtype(volume.dtype) != dtype
        for volume, dtype in zip(volumes, expected_dtypes, strict=True)
    ):
        raise ValueError(f"dynamic medial connectivity arrays changed: {state_path}")
    return volumes, str(identity["record_id"]), normalized_patch_event_ids


@lru_cache(maxsize=32)
def _open_atlas_patch_source(
    catalog_path_text: str,
    catalog_sha256: str,
    source_id: str,
) -> tuple[Any, Any | None, DenseFieldSpec | None, Any, Any, Any | None, Any | None]:
    catalog_path = Path(catalog_path_text)
    if not catalog_path.is_file() or _sha256(catalog_path) != catalog_sha256:
        raise ValueError(f"coarse-atlas catalog changed: {catalog_path}")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        raise TypeError(f"invalid coarse-atlas catalog: {catalog_path}")
    sources = catalog.get("sources")
    source = sources.get(source_id) if isinstance(sources, dict) else None
    catalog_schema = catalog.get("schema")
    if catalog_schema not in {
        "crossres-coarse-teacher-atlas-catalog-v1",
        "crossres-coarse-teacher-atlas-catalog-v2",
    } or not isinstance(source, dict):
        raise ValueError(f"invalid coarse-atlas catalog source {source_id!r}")
    state_path = remap_embedded_path(str(source.get("atlas_state", "")))
    if not state_path.is_absolute():
        state_path = catalog_path.parent / state_path
    expected_state_sha256 = str(source.get("atlas_state_sha256", ""))
    if (
        not state_path.is_file()
        or len(expected_state_sha256) != 64
        or _sha256(state_path) != expected_state_sha256
    ):
        raise ValueError(f"coarse teacher atlas state changed: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("state") != "complete":
        raise ValueError(f"coarse teacher atlas is incomplete: {state_path}")
    image = open_volume(str(source["coarse_image"]))
    raw_baseline = source.get("coarse_baseline")
    baseline_spec = (
        DenseFieldSpec.from_dict(
            raw_baseline,
            context="coarse-atlas catalog baseline",
            base=catalog_path.parent,
        )
        if raw_baseline is not None
        else None
    )
    baseline = open_volume(baseline_spec.volume) if baseline_spec is not None else None
    teacher_q = open_volume(str(source["teacher_q"]))
    target_valid = open_volume(str(source["target_valid"]))
    teacher_crest = None
    teacher_crest_valid = None
    if catalog_schema == "crossres-coarse-teacher-atlas-catalog-v2":
        medial_state_path = remap_embedded_path(str(source.get("medial_state", "")))
        if not medial_state_path.is_absolute():
            medial_state_path = catalog_path.parent / medial_state_path
        expected_medial_sha256 = str(source.get("medial_state_sha256", ""))
        if (
            not medial_state_path.is_file()
            or len(expected_medial_sha256) != 64
            or _sha256(medial_state_path) != expected_medial_sha256
        ):
            raise ValueError(f"coarse medial atlas state changed: {medial_state_path}")
        medial_state = json.loads(medial_state_path.read_text(encoding="utf-8"))
        if (
            medial_state.get("state") != "complete"
            or medial_state.get("identity", {}).get("parent_atlas_state_sha256")
            != expected_state_sha256
        ):
            raise ValueError(f"coarse medial atlas is incomplete: {medial_state_path}")
        teacher_crest = open_volume(str(source["teacher_crest"]))
        teacher_crest_valid = open_volume(str(source["teacher_crest_valid"]))
    shapes = {
        tuple(int(value) for value in array.shape)
        for array in (
            image,
            baseline,
            teacher_q,
            target_valid,
            teacher_crest,
            teacher_crest_valid,
        )
        if array is not None
    }
    if len(shapes) != 1:
        raise ValueError(
            f"coarse-atlas source arrays have different shapes: {source_id}"
        )
    return (
        image,
        baseline,
        baseline_spec,
        teacher_q,
        target_valid,
        teacher_crest,
        teacher_crest_valid,
    )


def _load_atlas_patch_arrays(
    row: PatchRecord,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
]:
    source = row.array_source
    if source is None:
        raise ValueError(f"{row.patch_id}: no coarse-atlas array source")
    (
        image_volume,
        baseline_volume,
        baseline_spec,
        q_volume,
        valid_volume,
        crest_volume,
        crest_valid_volume,
    ) = _open_atlas_patch_source(
        str(source["catalog"]),
        str(source["catalog_sha256"]),
        str(source["source_id"]),
    )
    if str(source["source_id"]) != row.record_id:
        raise ValueError(f"{row.patch_id}: coarse-atlas source record differs")
    if (baseline_volume is not None) != row.has_baseline:
        raise ValueError(f"{row.patch_id}: coarse-atlas baseline availability differs")
    shift = tuple(int(value) for value in source["teacher_shift_coarse_zyx"])
    teacher_origin = tuple(
        origin - offset for origin, offset in zip(row.origin_zyx, shift, strict=True)
    )
    image = read_crop(image_volume, row.origin_zyx, row.shape_zyx)
    teacher_q = read_crop(q_volume, teacher_origin, row.shape_zyx).astype(
        np.uint8, copy=False
    )
    target_valid = (read_crop(valid_volume, teacher_origin, row.shape_zyx) > 0).astype(
        np.uint8
    )
    has_teacher_crest = crest_volume is not None and crest_valid_volume is not None
    expects_teacher_crest = (
        row.preparation_version == MEDIAL_ATLAS_PATCH_PREPARATION_VERSION
    )
    if has_teacher_crest != expects_teacher_crest:
        raise ValueError(
            f"{row.patch_id}: patch and atlas disagree on medial supervision"
        )
    if has_teacher_crest:
        teacher_crest = (
            read_crop(crest_volume, teacher_origin, row.shape_zyx) > 0
        ).astype(np.uint8)
        teacher_crest_valid = (
            read_crop(crest_valid_volume, teacher_origin, row.shape_zyx) > 0
        ).astype(np.uint8)
        if np.any(teacher_crest > teacher_crest_valid):
            raise ValueError(f"{row.patch_id}: crest lies outside medial validity")
    else:
        teacher_crest = np.zeros(row.shape_zyx, dtype=np.uint8)
        teacher_crest_valid = np.zeros(row.shape_zyx, dtype=np.uint8)
    if baseline_volume is not None and baseline_spec is not None:
        baseline_raw = read_crop(baseline_volume, row.origin_zyx, row.shape_zyx)
        baseline_probability = decode_dense_field(baseline_raw, baseline_spec)
        baseline = (baseline_probability >= baseline_spec.threshold).astype(np.uint8)
    else:
        baseline = np.zeros(row.shape_zyx, dtype=np.uint8)
    target = np.full(row.shape_zyx, 2, dtype=np.uint8)
    valid = target_valid > 0
    target[valid] = (teacher_q[valid] >= 128).astype(np.uint8)
    return (
        image,
        target,
        baseline,
        teacher_q,
        target_valid,
        teacher_crest,
        teacher_crest_valid,
        has_teacher_crest,
    )


def _close_fraction(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1.0e-9, abs_tol=1.0e-9)


def _validate_patch_file(
    row: PatchRecord,
    *,
    require_hashes: bool,
) -> dict[str, int]:
    if row.array_source is not None:
        (
            image,
            target,
            baseline,
            teacher_q,
            target_valid,
            teacher_crest,
            teacher_crest_valid,
            has_teacher_crest,
        ) = _load_atlas_patch_arrays(row)
        archive_bytes = 0
    else:
        teacher_crest = None
        teacher_crest_valid = None
        has_teacher_crest = False
        if not row.path.is_file():
            raise ValueError(f"{row.path}: patch archive is missing")
        archive_bytes = row.path.stat().st_size
        if require_hashes and (row.archive_bytes is None or row.archive_sha256 is None):
            raise ValueError(f"{row.path}: patch manifest has no archive hash")
        if row.archive_bytes is not None and archive_bytes != row.archive_bytes:
            raise ValueError(f"{row.path}: archive byte count does not match manifest")
        if row.archive_sha256 is not None and (
            len(row.archive_sha256) != 64 or _sha256(row.path) != row.archive_sha256
        ):
            raise ValueError(f"{row.path}: archive SHA-256 does not match manifest")

        with np.load(row.path, allow_pickle=False) as archive:
            names = set(archive.files)
            expected_names = {"image", "target_u8"}
            if row.preparation_version == ANTIALIAS_PATCH_PREPARATION_VERSION:
                expected_names.update({"teacher_q_u8", "target_valid_u8"})
            if row.has_baseline:
                expected_names.add("baseline_u8")
            if names != expected_names:
                raise ValueError(
                    f"{row.path}: arrays {sorted(names)} do not match "
                    f"{sorted(expected_names)}"
                )
            image = np.asarray(archive["image"])
            target = np.asarray(archive["target_u8"])
            teacher_q = (
                np.asarray(archive["teacher_q_u8"])
                if "teacher_q_u8" in archive
                else None
            )
            target_valid = (
                np.asarray(archive["target_valid_u8"])
                if "target_valid_u8" in archive
                else None
            )
            baseline = (
                np.asarray(archive["baseline_u8"])
                if row.has_baseline
                else np.zeros_like(target, dtype=np.uint8)
            )
    if (
        tuple(image.shape) != row.shape_zyx
        or target.shape != image.shape
        or baseline.shape != image.shape
    ):
        raise ValueError(f"{row.path}: array shapes do not match manifest")
    if image.dtype.kind not in {"u", "i", "f"} or not np.isfinite(image).all():
        raise ValueError(f"{row.path}: CT image is non-numeric or non-finite")
    if target.dtype != np.uint8 or not np.isin(target, (0, 1, 2)).all():
        raise ValueError(f"{row.path}: target must be uint8 labels 0/1/2")
    if teacher_q is not None or target_valid is not None:
        if (
            teacher_q is None
            or target_valid is None
            or teacher_q.shape != image.shape
            or target_valid.shape != image.shape
            or teacher_q.dtype != np.uint8
            or target_valid.dtype != np.uint8
            or not np.isin(target_valid, (0, 1)).all()
        ):
            raise ValueError(f"{row.path}: invalid soft-target arrays")
        if not np.array_equal(target_valid > 0, target != 2):
            raise ValueError(f"{row.path}: target validity differs from hard labels")
        expected_hard = np.full(target.shape, 2, dtype=np.uint8)
        valid_soft = target_valid > 0
        expected_hard[valid_soft] = (teacher_q[valid_soft] >= 128).astype(np.uint8)
        if not np.array_equal(expected_hard, target):
            raise ValueError(f"{row.path}: hard target differs from q >= 0.50")
    if has_teacher_crest and (
        teacher_crest is None
        or teacher_crest_valid is None
        or teacher_crest.shape != image.shape
        or teacher_crest_valid.shape != image.shape
        or teacher_crest.dtype != np.uint8
        or teacher_crest_valid.dtype != np.uint8
        or not np.isin(teacher_crest, (0, 1)).all()
        or not np.isin(teacher_crest_valid, (0, 1)).all()
        or np.any(teacher_crest > teacher_crest_valid)
    ):
        raise ValueError(f"{row.path}: invalid medial target arrays")
    if (
        row.preparation_version == MEDIAL_ATLAS_PATCH_PREPARATION_VERSION
        and not has_teacher_crest
    ):
        raise ValueError(f"{row.path}: medial atlas patch has no crest arrays")
    if row.has_baseline and (
        baseline.dtype != np.uint8 or not np.isin(baseline, (0, 1)).all()
    ):
        raise ValueError(f"{row.path}: baseline must be uint8 labels 0/1")

    known = target != 2
    known_voxels = int(np.count_nonzero(known))
    positive_voxels = int(np.count_nonzero(target == 1))
    known_fraction = known_voxels / target.size
    positive_fraction = positive_voxels / max(1, known_voxels)
    ct_nonzero_fraction = float(np.count_nonzero(image)) / image.size
    pathology_score = (
        float(np.not_equal(baseline[known], target[known]).mean())
        if row.has_baseline and known_voxels
        else row.pathology_score
        if row.pathology_mining is not None
        else 0.0
    )
    if row.scrollfiesta_pred_metrics is not None and row.has_baseline:
        recomputed_scrollfiesta = scrollfiesta_patch_pred_metrics(baseline)
        if not scrollfiesta_metrics_close(
            recomputed_scrollfiesta,
            row.scrollfiesta_pred_metrics,
        ):
            raise ValueError(
                f"{row.path}: recomputed ScrollFiesta prediction metrics differ "
                "from manifest"
            )
    for name, actual, expected in (
        ("known fraction", known_fraction, row.known_fraction),
        ("positive fraction", positive_fraction, row.positive_fraction_known),
        ("CT nonzero fraction", ct_nonzero_fraction, row.ct_nonzero_fraction),
        ("pathology score", pathology_score, row.pathology_score),
    ):
        if not _close_fraction(actual, expected):
            raise ValueError(
                f"{row.path}: recomputed {name} {actual} does not match {expected}"
            )
    if row.known_fraction + 1.0e-12 < row.acceptance_min_known_fraction:
        raise ValueError(f"{row.path}: patch does not clear its known-fraction gate")
    return {
        "archive_bytes": archive_bytes,
        "voxels": int(target.size),
        "known_voxels": known_voxels,
        "positive_voxels": positive_voxels,
    }


def _validate_source_manifests(rows: list[PatchRecord]) -> int:
    by_root: dict[Path, list[PatchRecord]] = {}
    for row in rows:
        if row.path.parent.name != "patches":
            raise ValueError(f"{row.path}: patch is not inside a patches directory")
        by_root.setdefault(row.path.parent.parent, []).append(row)
    for root, included in by_root.items():
        state_path = root / "prepare_state.json"
        manifest_path = root / "patches.jsonl"
        if not state_path.is_file() or not manifest_path.is_file():
            raise ValueError(f"{root}: source preparation metadata is incomplete")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("state") != "complete":
            raise ValueError(f"{root}: source patch preparation is not complete")
        identity = state.get("identity")
        root_version = (
            identity.get("preparation_version") if isinstance(identity, dict) else None
        )
        if root_version not in SUPPORTED_PATCH_PREPARATION_VERSIONS:
            raise ValueError(f"{root}: unsupported patch preparation {root_version!r}")
        if identity.get("scrollfiesta_pred_metrics_contract") != (
            SCROLLFIESTA_PRED_METRICS_CONTRACT
        ):
            raise ValueError(
                f"{root}: ScrollFiesta prediction-metric contract is inconsistent"
            )
        source_rows = load_patch_manifest(manifest_path)
        if any(row.preparation_version != root_version for row in source_rows):
            raise ValueError(f"{root}: patch rows differ from root preparation version")
        options = identity.get("options")
        if not isinstance(options, dict):
            raise TypeError(f"{root}: source preparation options are missing")
        for row in source_rows:
            option_name = (
                "native_teacher_min_known_fraction"
                if "native-fine-teacher" in row.supervision_source
                else "min_known_fraction"
            )
            if option_name not in options:
                raise ValueError(f"{root}: source-aware known gate is missing")
            expected_gate = float(options[option_name])
            if not _close_fraction(
                row.acceptance_min_known_fraction,
                expected_gate,
            ):
                raise ValueError(
                    f"{row.patch_id}: acceptance known-fraction gate differs "
                    "from source preparation options"
                )
            if "native-fine-teacher" in row.supervision_source:
                quality_option = "native_teacher_min_fine_ct_nonzero_fraction"
                if quality_option not in options:
                    raise ValueError(
                        f"{root}: native-teacher fine-CT quality gate is missing"
                    )
                expected_quality_gate = float(options[quality_option])
                if (
                    row.native_teacher_min_fine_ct_nonzero_fraction is None
                    or not _close_fraction(
                        row.native_teacher_min_fine_ct_nonzero_fraction,
                        expected_quality_gate,
                    )
                ):
                    raise ValueError(
                        f"{row.patch_id}: fine-CT quality gate differs from "
                        "source preparation options"
                    )
        if int(state.get("completed", -1)) != len(source_rows):
            raise ValueError(f"{root}: completed patch count is inconsistent")
        source_by_id = {row.patch_id: row for row in source_rows}
        included_by_id = {row.patch_id: row for row in included}
        if set(included_by_id) != set(source_by_id):
            raise ValueError(
                f"{root}: merged manifest does not contain the exact source corpus"
            )
        for patch_id, source_row in source_by_id.items():
            included_row = included_by_id[patch_id]
            normalized = replace(
                included_row,
                pathology_score=source_row.pathology_score,
                scrollfiesta_pred_metrics=source_row.scrollfiesta_pred_metrics,
                sampling_strategy=source_row.sampling_strategy,
                pathology_mining=None,
            )
            if normalized != source_row:
                raise ValueError(
                    f"{patch_id}: pathology overlay changed immutable source fields"
                )
    return len(by_root)


def _validate_finite_anchor_coverage(
    record_id: str,
    included: list[PatchRecord],
    *,
    pool_size: int,
    registration_filtered: bool = False,
) -> tuple[set[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """Validate finite-supervision coverage without rejecting valid fallbacks.

    Before pool exhaustion, candidate groups are disjoint and therefore every
    selected anchor must be unique.  At or after exhaustion, every scheduled
    primary anchor must instead have been attempted.  A primary can legitimately
    remain unselected when its crop fails acceptance and the recorded fallback
    is selected, as happens near masked coarse-volume boundaries.
    """

    anchors = {
        row.support_anchor_chunk_zyx
        for row in included
        if row.support_anchor_chunk_zyx is not None
    }
    candidate_groups = [
        row.support_anchor_candidate_chunks_zyx
        for row in included
        if row.support_anchor_candidate_chunks_zyx is not None
    ]
    candidate_anchors = [
        coordinate for group in candidate_groups for coordinate in group
    ]
    if len(included) < pool_size and not registration_filtered:
        if len(anchors) != len(included):
            raise ValueError(
                f"{record_id}: {len(anchors)} unique support anchors, "
                f"expected {len(included)}"
            )
        if len(candidate_anchors) != len(set(candidate_anchors)):
            raise ValueError(f"{record_id}: support anchor candidate groups overlap")
        return anchors, candidate_anchors

    primary_anchors = {group[0] for group in candidate_groups}
    if len(primary_anchors) > pool_size or (
        len(primary_anchors) != pool_size and not registration_filtered
    ):
        raise ValueError(
            f"{record_id}: {len(primary_anchors)} primary support anchors, "
            f"expected full pool {pool_size}"
        )
    substituted_primaries = {
        group[0]
        for row, group in zip(included, candidate_groups, strict=True)
        if row.support_anchor_chunk_zyx != group[0]
    }
    unexplained_missing = primary_anchors - anchors - substituted_primaries
    if unexplained_missing:
        raise ValueError(
            f"{record_id}: {len(unexplained_missing)} unselected primary "
            "support anchors lack a recorded fallback substitution"
        )
    return anchors, candidate_anchors


def _validate_native_teacher_anchors(
    rows: list[PatchRecord],
) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[PatchRecord]] = {}
    for row in rows:
        if "native-fine-teacher" in row.supervision_source:
            grouped.setdefault(row.record_id, []).append(row)

    result: dict[str, dict[str, int]] = {}
    for record_id, included in sorted(grouped.items()):
        if any(
            row.preparation_version not in SUPPORTED_PATCH_PREPARATION_VERSIONS
            or row.support_anchor_chunk_zyx is None
            or row.support_anchor_pool_size is None
            or row.support_anchor_candidate_chunks_zyx is None
            for row in included
        ):
            raise ValueError(
                f"{record_id}: native-teacher patches lack versioned support anchors"
            )
        pool_sizes = {int(row.support_anchor_pool_size) for row in included}  # type: ignore[arg-type]
        if len(pool_sizes) != 1:
            raise ValueError(f"{record_id}: inconsistent support anchor pool sizes")
        pool_size = pool_sizes.pop()
        quality_rows = {
            (
                row.native_teacher_min_fine_ct_nonzero_fraction,
                row.native_teacher_fine_ct_quality_gate_applied,
                row.native_teacher_support_chunks_before_quality_gate,
                row.native_teacher_support_chunks_after_quality_gate,
                row.native_teacher_support_chunks_excluded_by_quality_gate,
            )
            for row in included
        }
        if len(quality_rows) != 1:
            raise ValueError(f"{record_id}: inconsistent fine-CT quality provenance")
        _, _, _, support_chunks_after, _ = quality_rows.pop()
        if support_chunks_after is None or pool_size > support_chunks_after:
            raise ValueError(
                f"{record_id}: anchor pool exceeds quality-filtered support"
            )
        filtered_flags = {
            row.registration_decision is not None
            and row.registered_source_quality_gate is not None
            for row in included
        }
        if len(filtered_flags) != 1:
            raise ValueError(
                f"{record_id}: inconsistent registered-source filtering provenance"
            )
        registration_filtered = filtered_flags.pop()
        anchors, candidate_anchors = _validate_finite_anchor_coverage(
            record_id,
            included,
            pool_size=pool_size,
            registration_filtered=registration_filtered,
        )
        primary_anchors = {
            row.support_anchor_candidate_chunks_zyx[0]  # type: ignore[index]
            for row in included
        }
        result[record_id] = {
            "patches": len(included),
            "anchor_pool_size": pool_size,
            "unique_anchors": len(anchors),
            "primary_anchors": len(primary_anchors),
            "primary_anchor_retained_fraction": len(primary_anchors) / pool_size,
            "registration_filtered": registration_filtered,
            "candidate_anchors": len(set(candidate_anchors)),
            "candidate_evaluations": len(candidate_anchors),
        }
    return result


def _validate_human_label_anchors(
    rows: list[PatchRecord],
) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[PatchRecord]] = {}
    for row in rows:
        if "official-human-2um" in row.supervision_source:
            grouped.setdefault(row.record_id, []).append(row)

    result: dict[str, dict[str, int]] = {}
    for record_id, included in sorted(grouped.items()):
        if any(
            row.preparation_version not in SUPPORTED_PATCH_PREPARATION_VERSIONS
            or row.support_anchor_chunk_zyx is None
            or row.support_anchor_pool_size is None
            or row.support_anchor_candidate_chunks_zyx is None
            for row in included
        ):
            raise ValueError(
                f"{record_id}: human-label patches lack versioned support anchors"
            )
        pool_sizes = {int(row.support_anchor_pool_size) for row in included}  # type: ignore[arg-type]
        if len(pool_sizes) != 1:
            raise ValueError(f"{record_id}: inconsistent support anchor pool sizes")
        pool_size = pool_sizes.pop()
        anchors, candidate_anchors = _validate_finite_anchor_coverage(
            record_id,
            included,
            pool_size=pool_size,
        )
        result[record_id] = {
            "patches": len(included),
            "anchor_pool_size": pool_size,
            "unique_anchors": len(anchors),
            "candidate_anchors": len(set(candidate_anchors)),
            "candidate_evaluations": len(candidate_anchors),
        }
    return result


def validate_patch_corpus(
    manifest_path: str | Path,
    *,
    expected_count: int | None = None,
    expected_split_counts: dict[str, int] | None = None,
    expected_train_scrolls: set[str] | None = None,
    expected_val_scrolls: set[str] | None = None,
    expected_test_scrolls: set[str] | None = None,
    expected_train_scroll_counts: dict[str, int] | None = None,
    expected_val_scroll_counts: dict[str, int] | None = None,
    expected_test_scroll_counts: dict[str, int] | None = None,
    expected_record_counts: dict[str, int] | None = None,
    expected_source_corpora: int | None = None,
    require_hashes: bool = True,
    voxel_check_count: int | None = None,
    workers: int = 8,
    max_cpu_threads: int = 16,
) -> dict[str, Any]:
    if not 1 <= workers <= max_cpu_threads:
        raise ValueError("workers must be in [1, max_cpu_threads]")
    if voxel_check_count is not None and voxel_check_count <= 0:
        raise ValueError("voxel_check_count must be positive")
    configure_cpu_budget(max_cpu_threads)
    source = Path(manifest_path).expanduser().resolve()
    rows = load_patch_manifest(source)
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(
            f"{source}: found {len(rows):,} patches, expected {expected_count:,}"
        )
    paths = [row.path for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{source}: duplicate patch archive paths")
    split_counts = Counter(row.split for row in rows)
    record_counts = Counter(row.record_id for row in rows)
    if expected_record_counts is not None:
        if any(
            not record_id or count <= 0
            for record_id, count in expected_record_counts.items()
        ):
            raise ValueError("expected record counts must be positive")
        actual = dict(sorted(record_counts.items()))
        canonical = dict(sorted(expected_record_counts.items()))
        if actual != canonical:
            raise ValueError(
                f"{source}: record counts {actual} do not match {canonical}"
            )
    if expected_split_counts is not None:
        for split, expected in expected_split_counts.items():
            if split_counts[split] != expected:
                raise ValueError(
                    f"{source}: {split} count {split_counts[split]:,} "
                    f"does not match {expected:,}"
                )
    train_scrolls = {row.scroll_id for row in rows if row.split == "train"}
    val_scrolls = {row.scroll_id for row in rows if row.split == "val"}
    test_scrolls = {row.scroll_id for row in rows if row.split == "test"}
    scroll_counts = {
        split: Counter(row.scroll_id for row in rows if row.split == split)
        for split in ("train", "val", "test")
    }
    overlaps = {
        "train/validation": train_scrolls & val_scrolls,
        "train/test": train_scrolls & test_scrolls,
        "validation/test": val_scrolls & test_scrolls,
    }
    for label, overlap in overlaps.items():
        if overlap:
            raise ValueError(f"{source}: {label} scroll leakage {sorted(overlap)}")
    if expected_train_scrolls is not None and train_scrolls != expected_train_scrolls:
        raise ValueError(
            f"{source}: train scroll set {sorted(train_scrolls)} does not match "
            f"{sorted(expected_train_scrolls)}"
        )
    if expected_val_scrolls is not None and val_scrolls != expected_val_scrolls:
        raise ValueError(
            f"{source}: val scroll set {sorted(val_scrolls)} does not match "
            f"{sorted(expected_val_scrolls)}"
        )
    if expected_test_scrolls is not None and test_scrolls != expected_test_scrolls:
        raise ValueError(
            f"{source}: test scroll set {sorted(test_scrolls)} does not match "
            f"{sorted(expected_test_scrolls)}"
        )
    for split, expected in (
        ("train", expected_train_scroll_counts),
        ("val", expected_val_scroll_counts),
        ("test", expected_test_scroll_counts),
    ):
        if expected is None:
            continue
        if any(not scroll or count <= 0 for scroll, count in expected.items()):
            raise ValueError(f"expected {split} scroll counts must be positive")
        actual = dict(sorted(scroll_counts[split].items()))
        canonical = dict(sorted(expected.items()))
        if actual != canonical:
            raise ValueError(
                f"{source}: {split} scroll counts {actual} do not match {canonical}"
            )
    source_corpora = _validate_source_manifests(rows)
    native_teacher_anchors = _validate_native_teacher_anchors(rows)
    human_label_anchors = _validate_human_label_anchors(rows)
    if (
        expected_source_corpora is not None
        and source_corpora != expected_source_corpora
    ):
        raise ValueError(
            f"{source}: {source_corpora} source corpora do not match "
            f"{expected_source_corpora}"
        )
    check_rows = rows
    if voxel_check_count is not None and voxel_check_count < len(rows):
        # The corpus builder has already read and quality-gated every dynamic
        # atlas-backed row.  This second pass exercises the training loader on
        # a deterministic, record-stratified audit sample without spending a
        # second full corpus pass on identical volume reads.
        grouped: dict[str, list[PatchRecord]] = {}
        for row in rows:
            grouped.setdefault(row.record_id, []).append(row)
        selected_ids: set[str] = set()
        if voxel_check_count >= len(grouped):
            for included in grouped.values():
                selected_ids.add(
                    min(
                        included,
                        key=lambda row: hashlib.sha256(row.patch_id.encode()).digest(),
                    ).patch_id
                )
        remaining = voxel_check_count - len(selected_ids)
        if remaining > 0:
            candidates = sorted(
                (row for row in rows if row.patch_id not in selected_ids),
                key=lambda row: hashlib.sha256(
                    f"{row.record_id}\x1f{row.patch_id}".encode()
                ).digest(),
            )
            selected_ids.update(row.patch_id for row in candidates[:remaining])
        check_rows = [row for row in rows if row.patch_id in selected_ids]
        if len(check_rows) != voxel_check_count:
            raise RuntimeError("deterministic voxel-audit sampling is inconsistent")
    started = time.perf_counter()
    totals: Counter[str] = Counter()
    validate = partial(_validate_patch_file, require_hashes=require_hashes)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(validate, check_rows), 1):
            totals.update(result)
            if index % 100 == 0 or index == len(check_rows):
                elapsed = max(time.perf_counter() - started, 1.0e-6)
                print(
                    f"validated patch voxels {index:,}/{len(check_rows):,} "
                    f"({index / elapsed:.2f}/s)",
                    flush=True,
                )
    return {
        "schema": "crossres-voxel-patch-validation-v1",
        "manifest": str(source),
        "manifest_sha256": _sha256(source),
        "patches": len(rows),
        "splits": dict(sorted(split_counts.items())),
        "record_counts": dict(sorted(record_counts.items())),
        "train_scrolls": sorted(train_scrolls),
        "val_scrolls": sorted(val_scrolls),
        "test_scrolls": sorted(test_scrolls),
        "scroll_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in scroll_counts.items()
        },
        "source_corpora": source_corpora,
        "voxel_checked_patches": len(check_rows),
        "voxel_check_mode": "full" if len(check_rows) == len(rows) else "sampled",
        "voxel_check_sample_sha256": hashlib.sha256(
            "\n".join(row.patch_id for row in check_rows).encode()
        ).hexdigest(),
        "native_teacher_anchor_records": native_teacher_anchors,
        "human_label_anchor_records": human_label_anchors,
        "archive_bytes": totals["archive_bytes"],
        "voxels": totals["voxels"],
        "known_voxels": totals["known_voxels"],
        "positive_voxels": totals["positive_voxels"],
        "supervision_sources": dict(
            sorted(Counter(row.supervision_source for row in rows).items())
        ),
        "sampling_strategies": dict(
            sorted(Counter(row.sampling_strategy for row in rows).items())
        ),
        "scrollfiesta_pred_reject_kinds": dict(
            sorted(
                Counter(
                    row.scrollfiesta_pred_metrics.reject_kind
                    if row.scrollfiesta_pred_metrics is not None
                    else "unavailable"
                    for row in rows
                ).items()
            )
        ),
        "pathology_mining_records": sum(
            row.pathology_mining is not None for row in rows
        ),
        "pathology_mining_identities": sorted(
            {
                str(row.pathology_mining["mining_identity_sha256"])
                for row in rows
                if row.pathology_mining is not None
            }
        ),
    }


def normalize_m7_ct(image: np.ndarray) -> np.ndarray:
    value = np.nan_to_num(
        image.astype(np.float32),
        nan=M7_CT_LOWER,
        posinf=M7_CT_UPPER,
        neginf=M7_CT_LOWER,
    )
    value = np.clip(value, M7_CT_LOWER, M7_CT_UPPER)
    return (value - M7_CT_MEAN) / M7_CT_STD


class VoxelPatchDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str | None,
        augment: bool = False,
        pinned_medial_bridge_state: str | Path | None = None,
        dynamic_medial_connectivity_state: str | Path | None = None,
    ) -> None:
        manifest = Path(manifest_path).expanduser().resolve()
        self.rows = [
            row
            for row in load_patch_manifest(manifest)
            if (row.split in {"train", "val"} if split is None else row.split == split)
        ]
        if not self.rows:
            label = "train-or-val" if split is None else split
            raise ValueError(f"patch manifest has no {label!r} rows")
        self.augment = augment
        self.has_complete_teacher_crest = all(
            row.preparation_version == MEDIAL_ATLAS_PATCH_PREPARATION_VERSION
            for row in self.rows
        )
        self.pinned_medial_bridge_state: str | None = None
        self.pinned_medial_bridge_state_sha256: str | None = None
        self.pinned_medial_bridge_record_id: str | None = None
        self.pinned_medial_bridge_eligible_patch_ids: frozenset[str] | None = None
        self.dynamic_medial_connectivity_state: str | None = None
        self.dynamic_medial_connectivity_state_sha256: str | None = None
        self.dynamic_medial_connectivity_record_id: str | None = None
        self.dynamic_medial_connectivity_eligible_patch_ids: frozenset[str] | None = (
            None
        )
        self.manifest_path = str(manifest)
        self.manifest_sha256 = _sha256(manifest)
        if (
            pinned_medial_bridge_state is not None
            and dynamic_medial_connectivity_state is not None
        ):
            raise ValueError(
                "fixed axial and dynamic medial connectivity atlases are mutually exclusive"
            )
        if pinned_medial_bridge_state is not None:
            state_path = Path(pinned_medial_bridge_state).expanduser().resolve()
            state_sha256 = _sha256(state_path)
            _, record_id, eligible_patch_ids = _open_pinned_medial_bridge_atlas(
                str(state_path),
                state_sha256,
                self.manifest_path,
                self.manifest_sha256,
            )
            if not any(row.record_id == record_id for row in self.rows):
                raise ValueError(
                    "pinned medial bridge record is absent from the dataset split"
                )
            rows_by_id = {row.patch_id: row for row in self.rows}
            if any(
                patch_id not in rows_by_id
                or rows_by_id[patch_id].record_id != record_id
                for patch_id in eligible_patch_ids
            ):
                raise ValueError(
                    "pinned medial bridge patch scope differs from the dataset split"
                )
            self.pinned_medial_bridge_state = str(state_path)
            self.pinned_medial_bridge_state_sha256 = state_sha256
            self.pinned_medial_bridge_record_id = record_id
            self.pinned_medial_bridge_eligible_patch_ids = eligible_patch_ids
        if dynamic_medial_connectivity_state is not None:
            state_path = Path(dynamic_medial_connectivity_state).expanduser().resolve()
            state_sha256 = _sha256(state_path)
            _, record_id, event_ids_by_patch = _open_dynamic_medial_connectivity_atlas(
                str(state_path),
                state_sha256,
                self.manifest_path,
                self.manifest_sha256,
            )
            eligible_patch_ids = frozenset(event_ids_by_patch)
            if not any(row.record_id == record_id for row in self.rows):
                raise ValueError(
                    "dynamic medial connectivity record is absent from the dataset split"
                )
            rows_by_id = {row.patch_id: row for row in self.rows}
            if any(
                patch_id not in rows_by_id
                or rows_by_id[patch_id].record_id != record_id
                for patch_id in eligible_patch_ids
            ):
                raise ValueError(
                    "dynamic medial connectivity patch scope differs from the dataset split"
                )
            self.dynamic_medial_connectivity_state = str(state_path)
            self.dynamic_medial_connectivity_state_sha256 = state_sha256
            self.dynamic_medial_connectivity_record_id = record_id
            self.dynamic_medial_connectivity_eligible_patch_ids = eligible_patch_ids

    def _load_pinned_medial_bridge(self, row: PatchRecord) -> np.ndarray | None:
        if self.pinned_medial_bridge_state is None:
            return None
        if (
            row.record_id != self.pinned_medial_bridge_record_id
            or row.patch_id not in self.pinned_medial_bridge_eligible_patch_ids
        ):
            # The manifest contains multiple scrolls whose voxel coordinates are
            # unrelated.  Keep a collatable zero label for non-target records,
            # but never read the PHerc0139 atlas in another record's frame.
            return np.zeros(row.shape_zyx, dtype=np.uint16)
        assert self.pinned_medial_bridge_state_sha256 is not None
        volume, record_id, eligible_patch_ids = _open_pinned_medial_bridge_atlas(
            self.pinned_medial_bridge_state,
            self.pinned_medial_bridge_state_sha256,
            self.manifest_path,
            self.manifest_sha256,
        )
        if record_id != row.record_id:
            raise RuntimeError("pinned medial bridge record identity changed")
        if row.patch_id not in eligible_patch_ids:
            raise RuntimeError("pinned medial bridge patch scope changed")
        values = read_crop(volume, row.origin_zyx, row.shape_zyx)
        if values.shape != row.shape_zyx or np.dtype(values.dtype) != np.dtype(
            np.uint16
        ):
            raise ValueError(f"{row.patch_id}: invalid pinned medial bridge patch")
        return np.asarray(values, dtype=np.uint16)

    def _load_dynamic_medial_connectivity(
        self, row: PatchRecord
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if self.dynamic_medial_connectivity_state is None:
            return None
        if (
            row.record_id != self.dynamic_medial_connectivity_record_id
            or row.patch_id not in self.dynamic_medial_connectivity_eligible_patch_ids
        ):
            return (
                np.zeros(row.shape_zyx, dtype=np.uint16),
                np.zeros(row.shape_zyx, dtype=np.uint8),
                np.zeros(row.shape_zyx, dtype=np.uint8),
            )
        assert self.dynamic_medial_connectivity_state_sha256 is not None
        volumes, record_id, event_ids_by_patch = (
            _open_dynamic_medial_connectivity_atlas(
                self.dynamic_medial_connectivity_state,
                self.dynamic_medial_connectivity_state_sha256,
                self.manifest_path,
                self.manifest_sha256,
            )
        )
        if record_id != row.record_id:
            raise RuntimeError("dynamic medial connectivity record identity changed")
        if row.patch_id not in event_ids_by_patch:
            raise RuntimeError("dynamic medial connectivity patch scope changed")
        values = tuple(
            np.asarray(read_crop(volume, row.origin_zyx, row.shape_zyx)).copy()
            for volume in volumes
        )
        expected_dtypes = (np.dtype(np.uint16), np.dtype(np.uint8), np.dtype(np.uint8))
        if any(
            value.shape != row.shape_zyx or np.dtype(value.dtype) != dtype
            for value, dtype in zip(values, expected_dtypes, strict=True)
        ):
            raise ValueError(
                f"{row.patch_id}: invalid dynamic medial connectivity patch"
            )
        allowed = event_ids_by_patch[row.patch_id]
        present = {
            int(identifier) for identifier in np.unique(values[0]) if identifier != 0
        }
        if not allowed.issubset(present):
            raise ValueError(
                f"{row.patch_id}: fully owned dynamic connectivity event is missing"
            )
        unexpected = present - allowed
        if unexpected:
            discard = np.isin(values[0], tuple(unexpected))
            for value in values:
                value[discard] = 0
        for identifier in allowed:
            selected = values[0] == identifier
            union = int(
                np.bitwise_or.reduce(
                    values[1][selected], initial=np.asarray(0, dtype=np.uint8)
                )
            )
            bits = tuple(bit for bit in range(8) if union & (1 << bit))
            if len(bits) < 2 or bits != tuple(range(len(bits))):
                raise ValueError(
                    f"{row.patch_id}: fully owned dynamic connectivity pins are incomplete"
                )
        return values

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _load(
        row: PatchRecord,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        bool,
    ]:
        if row.array_source is not None:
            return _load_atlas_patch_arrays(row)
        with np.load(row.path, allow_pickle=False) as archive:
            if "image" not in archive or "target_u8" not in archive:
                raise ValueError(f"{row.path}: missing image or target_u8")
            image = np.asarray(archive["image"])
            target = np.asarray(archive["target_u8"])
            teacher_q = (
                np.asarray(archive["teacher_q_u8"])
                if "teacher_q_u8" in archive
                else (target == 1).astype(np.uint8) * 255
            )
            target_valid = (
                np.asarray(archive["target_valid_u8"])
                if "target_valid_u8" in archive
                else (target != 2).astype(np.uint8)
            )
            baseline = (
                np.asarray(archive["baseline_u8"])
                if "baseline_u8" in archive
                else np.zeros_like(target, dtype=np.uint8)
            )
            teacher_crest = np.zeros_like(target, dtype=np.uint8)
            teacher_crest_valid = np.zeros_like(target, dtype=np.uint8)
            has_teacher_crest = False
        if (
            image.ndim != 3
            or target.shape != image.shape
            or baseline.shape != image.shape
            or teacher_q.shape != image.shape
            or target_valid.shape != image.shape
        ):
            raise ValueError(f"{row.path}: patch arrays must share one 3-D shape")
        if not np.isin(target, (0, 1, 2)).all():
            raise ValueError(f"{row.path}: target contains labels outside 0/1/2")
        if teacher_q.dtype != np.uint8 or not np.isin(target_valid, (0, 1)).all():
            raise ValueError(f"{row.path}: invalid soft target arrays")
        if not np.array_equal(target_valid > 0, target != 2):
            raise ValueError(f"{row.path}: target validity differs from labels")
        return (
            image,
            target,
            baseline,
            teacher_q,
            target_valid,
            teacher_crest,
            teacher_crest_valid,
            has_teacher_crest,
        )

    @staticmethod
    def _spatial_augment(
        image: torch.Tensor,
        target: torch.Tensor,
        baseline: torch.Tensor,
        teacher_q: torch.Tensor,
        target_valid: torch.Tensor,
        teacher_crest: torch.Tensor,
        teacher_crest_valid: torch.Tensor,
        pinned_medial_bridge: torch.Tensor | None,
        dynamic_connectivity_event: torch.Tensor | None,
        dynamic_connectivity_pins: torch.Tensor | None,
        dynamic_connectivity_free: torch.Tensor | None,
    ) -> tuple[Any, ...]:
        values: list[torch.Tensor] = [
            image,
            target,
            baseline,
            teacher_q,
            target_valid,
            teacher_crest,
            teacher_crest_valid,
        ]
        auxiliary = (
            pinned_medial_bridge,
            dynamic_connectivity_event,
            dynamic_connectivity_pins,
            dynamic_connectivity_free,
        )
        values.extend(value for value in auxiliary if value is not None)
        for spatial_axis in range(3):
            if bool(torch.rand(()) < 0.5):
                values = [
                    torch.flip(value, dims=(spatial_axis + 1,)) for value in values
                ]
        if image.shape[-3] == image.shape[-2] == image.shape[-1]:
            axis_pairs = ((1, 2), (1, 3), (2, 3))
            pair = axis_pairs[int(torch.randint(0, len(axis_pairs), ()).item())]
            rotations = int(torch.randint(0, 4, ()).item())
            if rotations:
                values = [torch.rot90(value, rotations, pair) for value in values]
        transformed_auxiliary: list[torch.Tensor | None] = []
        cursor = 7
        for original in auxiliary:
            if original is None:
                transformed_auxiliary.append(None)
            else:
                transformed_auxiliary.append(values[cursor])
                cursor += 1
        return (*values[:7], *transformed_auxiliary)

    @staticmethod
    def _intensity_augment(image: torch.Tensor) -> torch.Tensor:
        if bool(torch.rand(()) < 0.8):
            image = image * float(torch.empty(()).uniform_(0.75, 1.25))
            image = image + float(torch.empty(()).uniform_(-0.25, 0.25))
        if bool(torch.rand(()) < 0.3):
            image = image + torch.randn_like(image) * float(
                torch.empty(()).uniform_(0.0, 0.08)
            )
        if bool(torch.rand(()) < 0.25):
            image = F.avg_pool3d(image[None], kernel_size=3, stride=1, padding=1)[0]
        if bool(torch.rand(()) < 0.3):
            lower, upper = image.amin(), image.amax()
            span = upper - lower
            if float(span) > 1.0e-6:
                gamma = float(torch.empty(()).uniform_(0.7, 1.5))
                image = ((image - lower) / span).clamp(0, 1).pow(gamma) * span + lower
        return image

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        (
            raw_image,
            raw_target,
            raw_baseline,
            raw_teacher_q,
            raw_target_valid,
            raw_teacher_crest,
            raw_teacher_crest_valid,
            has_teacher_crest,
        ) = self._load(row)
        raw_pinned_medial_bridge = self._load_pinned_medial_bridge(row)
        raw_dynamic_connectivity = self._load_dynamic_medial_connectivity(row)
        image = torch.from_numpy(normalize_m7_ct(raw_image))[None]
        target = torch.from_numpy(raw_target.astype(np.int64, copy=False))[None]
        baseline = torch.from_numpy(raw_baseline.astype(np.float32, copy=False))[None]
        teacher_q = torch.from_numpy(raw_teacher_q.astype(np.float32) / 255.0)[None]
        target_valid = torch.from_numpy(
            raw_target_valid.astype(np.float32, copy=False)
        )[None]
        teacher_crest = torch.from_numpy(
            raw_teacher_crest.astype(np.float32, copy=False)
        )[None]
        teacher_crest_valid = torch.from_numpy(
            raw_teacher_crest_valid.astype(np.float32, copy=False)
        )[None]
        pinned_medial_bridge = (
            torch.from_numpy(raw_pinned_medial_bridge.astype(np.int64, copy=False))[
                None
            ]
            if raw_pinned_medial_bridge is not None
            else None
        )
        dynamic_connectivity_event = (
            torch.from_numpy(raw_dynamic_connectivity[0].astype(np.int64, copy=False))[
                None
            ]
            if raw_dynamic_connectivity is not None
            else None
        )
        dynamic_connectivity_pins = (
            torch.from_numpy(raw_dynamic_connectivity[1].astype(np.int64, copy=False))[
                None
            ]
            if raw_dynamic_connectivity is not None
            else None
        )
        dynamic_connectivity_free = (
            torch.from_numpy(raw_dynamic_connectivity[2].astype(np.int64, copy=False))[
                None
            ]
            if raw_dynamic_connectivity is not None
            else None
        )
        if self.augment:
            (
                image,
                target,
                baseline,
                teacher_q,
                target_valid,
                teacher_crest,
                teacher_crest_valid,
                pinned_medial_bridge,
                dynamic_connectivity_event,
                dynamic_connectivity_pins,
                dynamic_connectivity_free,
            ) = self._spatial_augment(
                image,
                target,
                baseline,
                teacher_q,
                target_valid,
                teacher_crest,
                teacher_crest_valid,
                pinned_medial_bridge,
                dynamic_connectivity_event,
                dynamic_connectivity_pins,
                dynamic_connectivity_free,
            )
            image = self._intensity_augment(image)
        sample = {
            "image": image.contiguous(),
            "target": target.contiguous(),
            "teacher_q": teacher_q.contiguous(),
            "target_valid": target_valid.contiguous(),
            "teacher_crest": teacher_crest.contiguous(),
            "teacher_crest_valid": teacher_crest_valid.contiguous(),
            "has_teacher_crest": torch.tensor(has_teacher_crest),
            "baseline": baseline.contiguous(),
            "has_baseline": torch.tensor(row.has_baseline),
            "pathology_score": torch.tensor(
                row.pathology_score
                if row.sampling_pathology_score is None
                else row.sampling_pathology_score,
                dtype=torch.float32,
            ),
            "scrollfiesta_pred_reject_kind": torch.tensor(
                row.scrollfiesta_pred_metrics.reject_priority
                if row.scrollfiesta_pred_metrics is not None
                else -1,
                dtype=torch.int64,
            ),
            "patch_id": row.patch_id,
            "scroll_id": row.scroll_id,
            "supervision_source": row.supervision_source,
            "sampling_strategy": row.sampling_strategy,
        }
        if pinned_medial_bridge is not None:
            sample["pinned_medial_bridge"] = pinned_medial_bridge.contiguous()
        if dynamic_connectivity_event is not None:
            assert dynamic_connectivity_pins is not None
            assert dynamic_connectivity_free is not None
            sample["dynamic_connectivity_event"] = (
                dynamic_connectivity_event.contiguous()
            )
            sample["dynamic_connectivity_pins"] = dynamic_connectivity_pins.contiguous()
            sample["dynamic_connectivity_free"] = dynamic_connectivity_free.contiguous()
        return sample
