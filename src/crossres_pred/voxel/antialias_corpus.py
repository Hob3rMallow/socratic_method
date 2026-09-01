from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from crossres_pred.resample import BridgeOptions

from .io import open_volume
from .loss import LOSS_CONTRACT
from .patches import (
    ANTIALIAS_PATCH_PREPARATION_VERSION,
    ANTIALIAS_TARGET_PROJECTION_CONTRACT,
    PATCH_SCHEMA,
)
from .registration import (
    ChunkSupport,
    FineFieldWindowReader,
    antialias_fine_target_patch,
)
from .resources import configure_cpu_budget
from .schema import VoxelPairRecord, load_pair_manifest
from .scrollfiesta_metrics import SCROLLFIESTA_PRED_METRICS_CONTRACT

STATE_SCHEMA = "crossres-antialias-corpus-state-v1"
SUMMARY_SCHEMA = "crossres-antialias-corpus-summary-v1"
DEFAULT_BRIDGE = BridgeOptions(
    prefilter_sigma_scale=0.5,
    coverage_erosion_fine_vox=0,
    max_fine_window_vox=352,
    maxpool_prefilter=False,
    erode_filter_margin=True,
)


@dataclass(frozen=True)
class AntialiasCorpusOptions:
    hard_threshold: float = 0.5
    min_known_fraction: float = 0.001
    min_positive_voxels: int = 32
    max_cpu_threads: int = 16
    fine_chunk_cache_entries: int = 64
    bridge: BridgeOptions = DEFAULT_BRIDGE

    def validate(self) -> None:
        if not 0.0 <= self.hard_threshold <= 1.0:
            raise ValueError("hard_threshold must be in [0, 1]")
        if not 0.0 <= self.min_known_fraction <= 1.0:
            raise ValueError("min_known_fraction must be in [0, 1]")
        if self.min_positive_voxels < 0:
            raise ValueError("min_positive_voxels cannot be negative")
        if not 1 <= self.max_cpu_threads <= 16:
            raise ValueError("max_cpu_threads must be in [1, 16]")
        if self.fine_chunk_cache_entries <= 0:
            raise ValueError("fine_chunk_cache_entries must be positive")
        self.bridge.validate()
        if (
            self.bridge.prefilter_sigma_scale != 0.5
            or self.bridge.coverage_erosion_fine_vox != 0
            or self.bridge.maxpool_prefilter
            or not self.bridge.erode_filter_margin
        ):
            raise ValueError("v11.2 requires the pinned anti-alias bridge contract")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no patch rows")
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}.npz")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_row_path(row: dict[str, Any], manifest: Path) -> Path:
    path = Path(str(row["path"])).expanduser()
    return (path if path.is_absolute() else manifest.parent / path).resolve()


def _pair_records(
    paths: list[Path],
) -> tuple[
    dict[str, VoxelPairRecord],
    list[dict[str, str]],
    dict[str, str],
]:
    records: dict[str, VoxelPairRecord] = {}
    identities: list[dict[str, str]] = []
    record_manifest_sha256: dict[str, str] = {}
    for path in paths:
        resolved = path.expanduser().resolve()
        manifest_sha256 = _sha256(resolved)
        identities.append({"path": str(resolved), "sha256": manifest_sha256})
        for record in load_pair_manifest(resolved):
            if record.record_id in records:
                raise ValueError(
                    f"duplicate pair record across manifests: {record.record_id}"
                )
            records[record.record_id] = record
            record_manifest_sha256[record.record_id] = manifest_sha256
    return records, identities, record_manifest_sha256


def _morphology(target: np.ndarray, valid: np.ndarray) -> dict[str, int | float]:
    positive = target == 1
    structure = ndimage.generate_binary_structure(3, 1)
    eligible = ndimage.binary_erosion(valid, structure=structure, iterations=2)
    interior = ndimage.binary_erosion(positive, structure=structure, iterations=2)
    interior &= eligible
    positive_voxels = int(positive.sum())
    return {
        "positive_voxels": positive_voxels,
        "two_erode_interior_voxels": int(interior.sum()),
        "two_erode_retained_fraction": float(interior.sum()) / max(1, positive_voxels),
    }


def _source_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        if "image" not in names or "target_u8" not in names:
            raise ValueError(f"{path}: source archive lacks image/target_u8")
        allowed = {
            "image",
            "target_u8",
            "baseline_u8",
            "teacher_q_u8",
            "target_valid_u8",
        }
        if not names <= allowed:
            raise ValueError(
                f"{path}: unexpected source arrays {sorted(names - allowed)}"
            )
        return {name: np.asarray(archive[name]) for name in names}


def _identity(
    *,
    source_manifest: Path,
    pair_identities: list[dict[str, str]],
    source_rows: list[dict[str, Any]],
    options: AntialiasCorpusOptions,
    patch_registration_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    quality_values = {
        float(row["native_teacher_min_fine_ct_nonzero_fraction"]) for row in source_rows
    }
    if len(quality_values) != 1:
        raise ValueError("source rows disagree on fine-CT quality threshold")
    return {
        "preparation_version": ANTIALIAS_PATCH_PREPARATION_VERSION,
        "target_projection_contract": ANTIALIAS_TARGET_PROJECTION_CONTRACT,
        "loss_contract": LOSS_CONTRACT,
        "scrollfiesta_pred_metrics_contract": SCROLLFIESTA_PRED_METRICS_CONTRACT,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "pair_manifests": pair_identities,
        "patch_registration": patch_registration_identity,
        "source_rows": len(source_rows),
        "options": {
            **asdict(options),
            "bridge": asdict(options.bridge),
            "native_teacher_min_known_fraction": options.min_known_fraction,
            "native_teacher_min_fine_ct_nonzero_fraction": quality_values.pop(),
        },
    }


def _load_patch_registrations(
    path: Path | None,
    source_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    if path is None:
        return {}, None
    manifest = path.expanduser().resolve()
    state_path = manifest.parent / "state.json"
    if not manifest.is_file() or not state_path.is_file():
        raise FileNotFoundError(f"patch-registration map is incomplete: {manifest}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    manifest_sha256 = _sha256(manifest)
    if (
        state.get("schema") != "crossres-patch-registration-state-v1"
        or state.get("state") != "complete"
        or state.get("manifest_sha256") != manifest_sha256
    ):
        raise ValueError(f"patch-registration map has no valid completion commit: {manifest}")
    registrations = {
        str(row["patch_id"]): row for row in _read_rows(manifest)
    }
    if len(registrations) != int(state.get("completed", -1)):
        raise ValueError("patch-registration manifest count differs from state")
    required = {str(row["patch_id"]) for row in source_rows}
    missing = sorted(required - set(registrations))
    if missing:
        raise ValueError(f"patch-registration map lacks source patches: {missing[:5]}")
    rejected = sorted(
        patch_id
        for patch_id in required
        if not bool(registrations[patch_id].get("accepted", False))
    )
    if rejected:
        raise ValueError(
            "source manifest contains rejected patch registrations: "
            f"{rejected[:5]}"
        )
    return registrations, {
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "state": str(state_path),
        "state_sha256": _sha256(state_path),
        "contract": "crossres-local-ct-translation-l0-v1",
        "source_rows": len(source_rows),
    }


def _completed_rows(rows_dir: Path, expected: int) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not rows_dir.exists():
        return completed
    for path in rows_dir.glob("*.json"):
        try:
            index = int(path.stem)
        except ValueError as error:
            raise ValueError(f"invalid reproject row sidecar: {path}") from error
        if not 0 <= index < expected or index in completed:
            raise ValueError(f"invalid or duplicate reproject row index: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"{path}: expected an object")
        completed[index] = value
    return completed


def _hardlink_archive(source: Path, destination: Path) -> None:
    """Install an immutable transformed archive without silently copying it."""

    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def require_same_file() -> None:
        try:
            same = os.path.samefile(source, destination)
        except OSError as error:
            raise FileExistsError(
                f"reprojected archive destination is unrelated: {destination}"
            ) from error
        if not same:
            raise FileExistsError(
                f"reprojected archive destination is unrelated: {destination}"
            )

    if destination.exists() or destination.is_symlink():
        require_same_file()
        return
    try:
        os.link(source, destination)
    except FileExistsError:
        require_same_file()


def _load_reuse_corpus(
    value: Path | None,
) -> tuple[dict[str, tuple[dict[str, Any], Path]], dict[str, Any] | None]:
    if value is None:
        return {}, None
    root = Path(value).expanduser().resolve()
    manifest = root / "patches.jsonl"
    state_path = root / "prepare_state.json"
    if not manifest.is_file() or not state_path.is_file():
        raise FileNotFoundError(f"reuse corpus is incomplete: {root}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    rows = _read_rows(manifest)
    manifest_sha256 = _sha256(manifest)
    if (
        state.get("schema") != STATE_SCHEMA
        or state.get("state") != "complete"
        or int(state.get("completed", -1)) != len(rows)
        or int(state.get("expected", -1)) != len(rows)
        or state.get("manifest_sha256") != manifest_sha256
    ):
        raise ValueError(f"reuse corpus does not have a valid completion commit: {root}")
    result: dict[str, tuple[dict[str, Any], Path]] = {}
    for row in rows:
        patch_id = str(row["patch_id"])
        if patch_id in result:
            raise ValueError(f"reuse corpus has duplicate patch ID: {patch_id}")
        if row.get("preparation_version") != ANTIALIAS_PATCH_PREPARATION_VERSION:
            raise ValueError(f"reuse row is not anti-aliased: {patch_id}")
        archive = _resolve_row_path(row, manifest)
        result[patch_id] = (row, archive)
    return result, {
        "corpus": str(root),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "candidate_rows": len(rows),
    }


def _validate_reuse_candidate(
    *,
    source_row: dict[str, Any],
    source_archive_sha256: str,
    record: VoxelPairRecord,
    pair_manifest_sha256: str,
    options: AntialiasCorpusOptions,
    reused_row: dict[str, Any],
    reused_archive: Path,
) -> None:
    patch_id = str(source_row["patch_id"])
    for key in ("patch_id", "record_id", "scroll_id", "split", "origin_zyx", "shape_zyx"):
        if reused_row.get(key) != source_row.get(key):
            raise ValueError(f"{patch_id}: reuse row differs on {key}")
    projection = reused_row.get("target_projection")
    expected = {
        "contract": ANTIALIAS_TARGET_PROJECTION_CONTRACT,
        "prefilter_sigma_scale": options.bridge.prefilter_sigma_scale,
        "coverage_erosion_fine_vox": options.bridge.coverage_erosion_fine_vox,
        "maxpool_prefilter": options.bridge.maxpool_prefilter,
        "erode_filter_margin": options.bridge.erode_filter_margin,
        "hard_threshold": options.hard_threshold,
        "projection_backend": "cuda-gauss-hermite3-pullback-linf-validity-v1",
        "gaussian_quadrature_order_per_axis": 3,
        "validity_erosion_metric": "linf",
        "pair_manifest_sha256": pair_manifest_sha256,
        "record_id": record.record_id,
        "source_archive_sha256": source_archive_sha256,
    }
    if not isinstance(projection, dict) or any(
        projection.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise ValueError(f"{patch_id}: reused target projection contract differs")
    expected_sha256 = str(reused_row.get("archive_sha256", ""))
    if (
        len(expected_sha256) != 64
        or not reused_archive.is_file()
        or reused_archive.stat().st_size != int(reused_row.get("archive_bytes", -1))
        or _sha256(reused_archive) != expected_sha256
    ):
        raise ValueError(f"{patch_id}: reused transformed archive changed")


def _merge_reused_row(
    *,
    source_row: dict[str, Any],
    reused_row: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Carry current source provenance while retaining identical target facts."""

    row = dict(source_row)
    row.pop("pathology_mining", None)
    for key in (
        "preparation_version",
        "acceptance_min_known_fraction",
        "known_fraction",
        "positive_fraction_known",
        "pathology_score",
        "sampling_pathology_score",
        "archive_bytes",
        "archive_sha256",
        "target_projection",
        "antialias_stats",
    ):
        row[key] = reused_row[key]
    row["path"] = f"patches/{destination.name}"
    return row


def _morton_key(coordinate_zyx: tuple[int, int, int]) -> int:
    if any(item < 0 for item in coordinate_zyx):
        raise ValueError("anti-alias locality coordinates must be non-negative")
    result = 0
    for bit in range(21):
        result |= ((coordinate_zyx[2] >> bit) & 1) << (3 * bit)
        result |= ((coordinate_zyx[1] >> bit) & 1) << (3 * bit + 1)
        result |= ((coordinate_zyx[0] >> bit) & 1) << (3 * bit + 2)
    return result


def _processing_order(rows: list[dict[str, Any]]) -> list[int]:
    """Group records and traverse their immutable origins in spatial order.

    Final manifest order remains byte-for-byte tied to the source indices. The
    processing-only Morton order lets consecutive pullbacks reuse decompressed
    fine chunks without changing a single training origin or sampler index.
    """

    def key(index: int) -> tuple[str, int, int]:
        row = rows[index]
        raw_anchor = row.get("support_anchor_chunk_zyx")
        if raw_anchor is None:
            raw_anchor = [int(item) // 128 for item in row["origin_zyx"]]
        coordinate = tuple(int(item) for item in raw_anchor)
        if len(coordinate) != 3:
            raise ValueError(f"{row['patch_id']}: invalid locality anchor")
        return str(row["record_id"]), _morton_key(coordinate), index

    return sorted(range(len(rows)), key=key)


def reproject_patch_corpus(
    *,
    source_manifest_path: str | Path,
    pair_manifest_paths: list[str | Path],
    output_path: str | Path,
    options: AntialiasCorpusOptions | None = None,
    maximum_rows: int | None = None,
    reuse_corpus_path: str | Path | None = None,
    patch_registration_manifest_path: str | Path | None = None,
) -> Path:
    """Rebuild a patch corpus at exactly the source origins with soft targets."""

    options = options or AntialiasCorpusOptions()
    options.validate()
    configure_cpu_budget(options.max_cpu_threads)
    source_manifest = Path(source_manifest_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    pair_paths = [Path(path).expanduser().resolve() for path in pair_manifest_paths]
    source_rows = _read_rows(source_manifest)
    if maximum_rows is not None:
        if maximum_rows <= 0:
            raise ValueError("maximum_rows must be positive")
        source_rows = source_rows[:maximum_rows]
    if any(row.get("schema") != PATCH_SCHEMA for row in source_rows):
        raise ValueError("source manifest has a non-voxel patch row")
    if len({str(row["patch_id"]) for row in source_rows}) != len(source_rows):
        raise ValueError("source manifest has duplicate patch IDs")
    pair_records, pair_identities, record_manifest_sha256 = _pair_records(pair_paths)
    missing = sorted({str(row["record_id"]) for row in source_rows} - set(pair_records))
    if missing:
        raise ValueError(f"pair manifests lack selected records: {missing}")
    patch_registrations, patch_registration_identity = _load_patch_registrations(
        None
        if patch_registration_manifest_path is None
        else Path(patch_registration_manifest_path),
        source_rows,
    )
    if patch_registrations and reuse_corpus_path is not None:
        raise ValueError("registered reprojection does not reuse pre-registration archives")
    identity = _identity(
        source_manifest=source_manifest,
        pair_identities=pair_identities,
        source_rows=source_rows,
        options=options,
        patch_registration_identity=patch_registration_identity,
    )
    reuse_rows, reuse_info = _load_reuse_corpus(
        None if reuse_corpus_path is None else Path(reuse_corpus_path)
    )

    output.mkdir(parents=True, exist_ok=True)
    patches_dir = output / "patches"
    rows_dir = output / "rows"
    patches_dir.mkdir(exist_ok=True)
    rows_dir.mkdir(exist_ok=True)
    state_path = output / "prepare_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("identity") != identity:
            raise ValueError("existing anti-alias corpus identity differs")
    completed = _completed_rows(rows_dir, len(source_rows))

    def count_existing_reuse() -> int:
        count = 0
        for row in completed.values():
            candidate = reuse_rows.get(str(row.get("patch_id", "")))
            if candidate is None:
                continue
            _, reused_archive = candidate
            destination = patches_dir / f"{row['patch_id']}.npz"
            try:
                count += int(os.path.samefile(reused_archive, destination))
            except OSError:
                continue
        return count

    reuse_archives = count_existing_reuse()

    def save_running(started_at: float) -> None:
        elapsed = max(time.perf_counter() - started_at, 1.0e-6)
        payload: dict[str, Any] = {
            "schema": STATE_SCHEMA,
            "state": "running",
            "completed": len(completed),
            "expected": len(source_rows),
            "patches_per_second": len(completed) / elapsed,
            "identity": identity,
        }
        if reuse_info is not None:
            payload["reuse"] = {**reuse_info, "archives_reused": reuse_archives}
        _atomic_json(state_path, payload)

    active_record_id: str | None = None
    active_context: (
        tuple[VoxelPairRecord, Any, ChunkSupport, FineFieldWindowReader] | None
    ) = None
    started = time.perf_counter()
    for index in _processing_order(source_rows):
        source_row = source_rows[index]
        patch_id = str(source_row["patch_id"])
        destination = patches_dir / f"{patch_id}.npz"
        sidecar = rows_dir / f"{index:06d}.json"
        if index in completed:
            row = completed[index]
            if row.get("patch_id") != patch_id:
                raise ValueError(f"{sidecar}: patch order changed")
            if not destination.is_file() or _sha256(destination) != row.get(
                "archive_sha256"
            ):
                raise ValueError(f"{destination}: committed archive changed")
            continue

        record = pair_records[str(source_row["record_id"])]
        registration_row = patch_registrations.get(patch_id)
        if registration_row is not None:
            if (
                registration_row.get("record_id") != record.record_id
                or registration_row.get("origin_zyx") != source_row.get("origin_zyx")
            ):
                raise ValueError(f"{patch_id}: patch-registration identity differs")
            effective_affine = tuple(
                tuple(float(item) for item in affine_row)
                for affine_row in registration_row["effective_to_coarse_affine_xyz"]
            )
        else:
            effective_affine = record.fine.to_coarse_affine_xyz
        if (
            record.scroll_id != source_row["scroll_id"]
            or record.split != source_row["split"]
        ):
            raise ValueError(f"{patch_id}: pair identity differs from selected row")
        source_path = _resolve_row_path(source_row, source_manifest)
        source_sha256 = _sha256(source_path)
        if source_sha256 != str(source_row.get("archive_sha256", "")):
            raise ValueError(f"{patch_id}: source archive hash changed")
        source_arrays = _source_arrays(source_path)
        image = source_arrays["image"]
        shape = tuple(int(item) for item in source_row["shape_zyx"])
        if tuple(image.shape) != shape:
            raise ValueError(f"{patch_id}: source image shape changed")
        reuse = reuse_rows.get(patch_id)
        if reuse is not None:
            reused_row, reused_archive = reuse
            _validate_reuse_candidate(
                source_row=source_row,
                source_archive_sha256=source_sha256,
                record=record,
                pair_manifest_sha256=record_manifest_sha256[record.record_id],
                options=options,
                reused_row=reused_row,
                reused_archive=reused_archive,
            )
            _hardlink_archive(reused_archive, destination)
            row = _merge_reused_row(
                source_row=source_row,
                reused_row=reused_row,
                destination=destination,
            )
            _atomic_json(sidecar, row)
            completed[index] = row
            reuse_archives += 1
            save_running(started)
            print(
                f"anti-alias patches {len(completed):,}/{len(source_rows):,} "
                f"{patch_id} reused",
                flush=True,
            )
            continue

        if active_record_id != record.record_id or active_context is None:
            fine_volume = open_volume(record.fine.target.volume)
            support = ChunkSupport.from_field(record.fine.target, fine_volume)
            reader = FineFieldWindowReader(
                fine_volume,
                record.fine.target,
                support,
                max_cache_chunks=options.fine_chunk_cache_entries,
            )
            active_record_id = record.record_id
            active_context = (record, fine_volume, support, reader)
        _, fine_volume, support, reader = active_context
        reads_before = reader.chunk_reads
        hits_before = reader.cache_hits
        target, q, valid_u8, stats = antialias_fine_target_patch(
            fine_volume,
            record.fine.target,
            support,
            effective_affine,
            tuple(int(item) for item in source_row["origin_zyx"]),
            shape,
            options=options.bridge,
            reader=reader,
            hard_threshold=options.hard_threshold,
        )
        required_backend = "cuda-gauss-hermite3-pullback-linf-validity-v1"
        if stats.get("projection_backend") != required_backend:
            raise RuntimeError(
                f"{patch_id}: required {required_backend}, got "
                f"{stats.get('projection_backend')}"
            )
        stats["fine_chunk_reads"] = reader.chunk_reads - reads_before
        stats["fine_chunk_cache_hits"] = reader.cache_hits - hits_before
        known_fraction = float(stats["known_fraction"])
        positive_voxels = int(stats["positive_voxels"])
        if known_fraction + 1.0e-12 < options.min_known_fraction:
            raise ValueError(
                f"{patch_id}: anti-aliased known fraction {known_fraction:.6f} "
                f"is below {options.min_known_fraction:.6f}"
            )
        if positive_voxels < options.min_positive_voxels:
            raise ValueError(
                f"{patch_id}: anti-aliased positives {positive_voxels} are below "
                f"{options.min_positive_voxels}"
            )
        q_u8 = np.rint(np.clip(q, 0.0, 1.0) * 255.0).astype(np.uint8)
        valid = valid_u8 > 0
        target = np.full(shape, 2, dtype=np.uint8)
        target[valid] = (q_u8[valid] >= 128).astype(np.uint8)
        arrays = {
            "image": image,
            "target_u8": target,
            "teacher_q_u8": q_u8,
            "target_valid_u8": valid_u8,
        }
        if "baseline_u8" in source_arrays:
            arrays["baseline_u8"] = source_arrays["baseline_u8"]
        _atomic_npz(destination, arrays)
        archive_sha256 = _sha256(destination)
        pathology_score = 0.0
        if "baseline_u8" in arrays and valid.any():
            pathology_score = float(
                np.not_equal(arrays["baseline_u8"][valid], target[valid]).mean()
            )
        morphology = _morphology(target, valid)
        projection = {
            "contract": ANTIALIAS_TARGET_PROJECTION_CONTRACT,
            "prefilter_sigma_scale": options.bridge.prefilter_sigma_scale,
            "coverage_erosion_fine_vox": options.bridge.coverage_erosion_fine_vox,
            "maxpool_prefilter": options.bridge.maxpool_prefilter,
            "erode_filter_margin": options.bridge.erode_filter_margin,
            "hard_threshold": options.hard_threshold,
            "projection_backend": stats["projection_backend"],
            "gaussian_quadrature_order_per_axis": stats.get(
                "gaussian_quadrature_order_per_axis"
            ),
            "validity_erosion_metric": "linf",
            "pair_manifest_sha256": record_manifest_sha256[record.record_id],
            "record_id": record.record_id,
            "fine_field_encoding": record.fine.target.encoding,
            "fine_positive_labels": list(record.fine.target.positive_labels),
            "fine_declared_threshold": record.fine.target.threshold,
            "source_archive_sha256": source_sha256,
            "source_preparation_version": source_row.get("preparation_version"),
            "source_known_fraction": source_row.get("known_fraction"),
            "source_positive_fraction_known": source_row.get("positive_fraction_known"),
            "source_pathology_score": source_row.get("pathology_score", 0.0),
            "patch_registration": (
                {
                    "contract": registration_row["contract"],
                    "method": registration_row["method"],
                    "shift_coarse_zyx": registration_row["shift_coarse_zyx"],
                    "best_structure_ncc": (
                        registration_row.get("registration_3d", {}).get(
                            "best_structure_ncc"
                        )
                    ),
                    "peak_margin": registration_row.get("registration_3d", {}).get(
                        "peak_margin"
                    ),
                    "map_manifest_sha256": (
                        patch_registration_identity["manifest_sha256"]
                        if patch_registration_identity is not None
                        else None
                    ),
                }
                if registration_row is not None
                else None
            ),
            "soft_positive_fraction_known": stats["soft_positive_fraction_known"],
            **morphology,
        }
        row = dict(source_row)
        row.pop("pathology_mining", None)
        row.update(
            path=f"patches/{patch_id}.npz",
            preparation_version=ANTIALIAS_PATCH_PREPARATION_VERSION,
            acceptance_min_known_fraction=options.min_known_fraction,
            known_fraction=known_fraction,
            positive_fraction_known=float(stats["positive_fraction_known"]),
            pathology_score=pathology_score,
            sampling_pathology_score=float(source_row.get("pathology_score", 0.0)),
            archive_bytes=destination.stat().st_size,
            archive_sha256=archive_sha256,
            target_projection=projection,
            antialias_stats=stats,
        )
        _atomic_json(sidecar, row)
        completed[index] = row
        save_running(started)
        print(
            f"anti-alias patches {len(completed):,}/{len(source_rows):,} "
            f"{patch_id} known={known_fraction:.4f} "
            f"positive={positive_voxels:,}",
            flush=True,
        )

    ordered = [completed[index] for index in range(len(source_rows))]
    manifest = output / "patches.jsonl"
    _atomic_jsonl(manifest, ordered)
    total_positive = sum(
        int(row["target_projection"]["positive_voxels"]) for row in ordered
    )
    total_interior = sum(
        int(row["target_projection"]["two_erode_interior_voxels"]) for row in ordered
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "patches": len(ordered),
        "splits": {
            split: sum(row["split"] == split for row in ordered)
            for split in ("train", "val", "test")
        },
        "scrolls": {
            scroll: sum(row["scroll_id"] == scroll for row in ordered)
            for scroll in sorted({str(row["scroll_id"]) for row in ordered})
        },
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "two_erode_retained_fraction": total_interior / max(1, total_positive),
        "identity": identity,
    }
    if reuse_info is not None:
        summary["reuse"] = {**reuse_info, "archives_reused": reuse_archives}
    _atomic_json(output / "summary.json", summary)
    final_state: dict[str, Any] = {
        "schema": STATE_SCHEMA,
        "state": "complete",
        "completed": len(ordered),
        "expected": len(ordered),
        "identity": identity,
        "manifest_sha256": summary["manifest_sha256"],
        "summary_sha256": _sha256(output / "summary.json"),
    }
    if reuse_info is not None:
        final_state["reuse"] = {**reuse_info, "archives_reused": reuse_archives}
    _atomic_json(state_path, final_state)
    return manifest
