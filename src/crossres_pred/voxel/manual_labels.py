from __future__ import annotations

import hashlib
import json
import os
import threading
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from crossres_pred.concurrency import bounded_thread_map_ordered

from .resources import configure_cpu_budget

EXTRACT_STATE = "crossres_manual_extract.json"
INDEX_STATE = "crossres_label_index.json"
INDEX_ROWS = "crossres_label_chunks.jsonl"
EXTRACT_RESUME_VALIDATION = "size+zip-crc32"
# Written by voxel.label_store_compaction; imported by name to avoid a cycle.
COMPACTION_STATE = "crossres_label_compaction.json"
ZERO_LENGTH_CHUNK_POLICY = "explicit-unknown-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_catalog(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema") != "crossres-official-manual-label-corpus-v1":
        raise ValueError(f"{source}: invalid manual-label catalog schema")
    datasets = value.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"{source}: catalog contains no datasets")
    names = [str(item["zarr"]) for item in datasets]
    ids = [str(item["dataset_id"]) for item in datasets]
    if len(names) != len(set(names)) or len(ids) != len(set(ids)):
        raise ValueError(f"{source}: duplicate dataset IDs or Zarr names")
    value["_path"] = str(source)
    value["_sha256"] = _sha256(source)
    return value


def verify_archive(
    archive_path: str | Path,
    catalog_path: str | Path,
    *,
    compute_sha256: bool = False,
    check_crc: bool = False,
) -> dict[str, Any]:
    archive = Path(archive_path).expanduser().resolve()
    catalog = load_catalog(catalog_path)
    expected_size = int(catalog["source"]["size_bytes"])
    actual_size = archive.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{archive}: incomplete archive {actual_size} != {expected_size} bytes"
        )
    result: dict[str, Any] = {
        "archive": str(archive),
        "size_bytes": actual_size,
        "expected_size_bytes": expected_size,
        "catalog": catalog["_path"],
        "catalog_sha256": catalog["_sha256"],
        "source_xet_hash": catalog["source"]["xet_hash"],
    }
    if compute_sha256:
        result["archive_sha256"] = _sha256(archive)
    with zipfile.ZipFile(archive) as source:
        # Opening the central directory catches truncation/format failures.  A
        # full testzip() pass decompresses the entire 89 GB corpus and would be
        # duplicated by extraction, so reserve it for an explicit deep audit.
        result["zip_members"] = len(source.infolist())
        if check_crc:
            bad_member = source.testzip()
            if bad_member is not None:
                raise ValueError(f"{archive}: CRC failure in {bad_member}")
            result["crc_checked"] = True
    return result


def _selected_member(
    name: str, zarr_names: frozenset[str]
) -> tuple[str, ...] | None:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    parts = path.parts
    zarr_index = next(
        (index for index, part in enumerate(parts) if part in zarr_names), None
    )
    if zarr_index is None:
        return None
    relative = parts[zarr_index:]
    if len(relative) == 2 and relative[1] in {".zattrs", ".zgroup", "meta.json"}:
        return relative
    if len(relative) == 3 and relative[1] == "0" and relative[2] in {
        ".zarray",
        ".zattrs",
    }:
        return relative
    if (
        len(relative) >= 3
        and relative[1] == "0"
        and not relative[2].startswith(".")
    ):
        return relative
    return None


def _matches_zip_member(path: Path, info: zipfile.ZipInfo) -> bool:
    if not path.is_file() or path.stat().st_size != info.file_size:
        return False
    checksum = 0
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            checksum = zlib.crc32(block, checksum)
    return checksum & 0xFFFFFFFF == info.CRC


def extract_level0(
    *,
    archive_path: str | Path,
    output_path: str | Path,
    catalog_path: str | Path,
    max_cpu_threads: int = 16,
) -> Path:
    """Resume-safe extraction of only training-resolution label data."""

    if not 1 <= max_cpu_threads <= 16:
        raise ValueError("max_cpu_threads must be in [1, 16]")
    configure_cpu_budget(max_cpu_threads)
    archive = Path(archive_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    catalog = load_catalog(catalog_path)
    expected_size = int(catalog["source"]["size_bytes"])
    if archive.stat().st_size != expected_size:
        raise ValueError(
            f"{archive}: archive is still downloading "
            f"({archive.stat().st_size}/{expected_size} bytes)"
        )
    identity = {
        "archive": str(archive),
        "archive_size": archive.stat().st_size,
        "archive_mtime_ns": archive.stat().st_mtime_ns,
        "catalog": catalog["_path"],
        "catalog_sha256": catalog["_sha256"],
        "mode": "level0-plus-root-metadata",
    }
    state_path = output / EXTRACT_STATE
    # Arming gate. Extraction re-materializes every published chunk, and a store
    # we compacted has had its all-unknown chunks deliberately deleted and its
    # fill_value flipped. Re-extracting would restore millions of chunks into a
    # store whose metadata says they are absent -- wrong, and invisible. Checked
    # only when extraction would actually run, so a finished corpus still
    # short-circuits normally.
    extraction_complete = False
    if state_path.is_file():
        try:
            extraction_complete = (
                json.loads(state_path.read_text(encoding="utf-8")).get("state")
                == "complete"
            )
        except (OSError, ValueError):
            extraction_complete = False
    if not extraction_complete:
        for item in catalog["datasets"]:
            record = output / str(item["zarr"]) / COMPACTION_STATE
            if record.is_file():
                raise ValueError(
                    f"{record.parent}: store has been compacted "
                    f"({COMPACTION_STATE} present); refusing to re-extract over it"
                )
    if output.exists():
        if not state_path.is_file():
            raise ValueError(f"{output}: non-empty extraction has no state file")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("identity") != identity:
            raise ValueError(f"{output}: extraction identity changed")
        resume_validation = state.get("resume_validation")
        if resume_validation not in {None, EXTRACT_RESUME_VALIDATION}:
            raise ValueError(
                f"{output}: unsupported extraction resume validation "
                f"{resume_validation!r}"
            )
        if state.get("state") == "complete":
            return state_path
    else:
        output.mkdir(parents=True)
        _atomic_json(
            state_path,
            {
                "state": "extracting",
                "identity": identity,
                "completed_files": 0,
                "completed_bytes": 0,
                "resume_validation": EXTRACT_RESUME_VALIDATION,
            },
        )

    names = frozenset(str(item["zarr"]) for item in catalog["datasets"])
    completed_files = 0
    completed_bytes = 0
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            relative = _selected_member(info.filename, names)
            if relative is None or info.is_dir():
                continue
            destination = output.joinpath(*relative)
            if _matches_zip_member(destination, info):
                completed_files += 1
                completed_bytes += info.file_size
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".part")
            checksum = 0
            with source.open(info) as input_stream, temporary.open("wb") as stream:
                while block := input_stream.read(8 * 1024 * 1024):
                    stream.write(block)
                    checksum = zlib.crc32(block, checksum)
                stream.flush()
            if temporary.stat().st_size != info.file_size:
                temporary.unlink(missing_ok=True)
                raise OSError(f"{info.filename}: extracted size mismatch")
            if checksum & 0xFFFFFFFF != info.CRC:
                temporary.unlink(missing_ok=True)
                raise OSError(f"{info.filename}: extracted CRC mismatch")
            os.replace(temporary, destination)
            completed_files += 1
            completed_bytes += info.file_size
            if completed_files % 1000 == 0:
                _atomic_json(
                    state_path,
                    {
                        "state": "extracting",
                        "identity": identity,
                        "completed_files": completed_files,
                        "completed_bytes": completed_bytes,
                        "resume_validation": EXTRACT_RESUME_VALIDATION,
                    },
                )
                print(
                    f"extracted {completed_files:,} files, "
                    f"{completed_bytes / 2**30:.2f} GiB",
                    flush=True,
                )
    _atomic_json(
        state_path,
        {
            "state": "complete",
            "identity": identity,
            "completed_files": completed_files,
            "completed_bytes": completed_bytes,
            "resume_validation": EXTRACT_RESUME_VALIDATION,
        },
    )
    return state_path


@dataclass(frozen=True)
class ChunkTask:
    coordinate_zyx: tuple[int, int, int]
    path: Path
    relative_path: str


def _chunk_coordinate(
    relative: Path,
    *,
    separator: str,
    grid_zyx: tuple[int, int, int],
) -> tuple[int, int, int]:
    parts = relative.parts
    raw = tuple(parts[0].split(".")) if len(parts) == 1 else parts
    if separator == "/" and len(parts) != 3:
        raise ValueError(f"invalid slash-separated chunk key {relative}")
    if len(raw) != 3:
        raise ValueError(f"invalid three-dimensional chunk key {relative}")
    try:
        coordinate = tuple(int(item) for item in raw)
    except ValueError as error:
        raise ValueError(f"invalid chunk coordinate {relative}") from error
    if any(
        item < 0 or item >= extent
        for item, extent in zip(coordinate, grid_zyx, strict=True)
    ):
        raise ValueError(f"out-of-grid chunk coordinate {relative}")
    return coordinate  # type: ignore[return-value]


def _discover_chunks(
    array_root: Path,
    *,
    separator: str,
    grid_zyx: tuple[int, int, int],
) -> list[ChunkTask]:
    tasks: list[ChunkTask] = []
    pending_directories = [array_root]
    while pending_directories:
        parent = pending_directories.pop()
        with os.scandir(parent) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    pending_directories.append(Path(entry.path))
                    continue
                name = entry.name
                if name.startswith(".") or name.endswith(".part"):
                    continue
                if not entry.is_file(follow_symlinks=False):
                    raise ValueError(f"unexpected non-file Zarr entry {entry.path}")
                path = Path(entry.path)
                relative = path.relative_to(array_root)
                coordinate = _chunk_coordinate(
                    relative, separator=separator, grid_zyx=grid_zyx
                )
                tasks.append(
                    ChunkTask(
                        coordinate_zyx=coordinate,
                        path=path,
                        relative_path=f"0/{relative.as_posix()}",
                    )
                )
                if len(tasks) % 100_000 == 0:
                    print(
                        f"discovered {len(tasks):,} present chunks under {array_root}",
                        flush=True,
                    )
    print(
        f"discovered {len(tasks):,} present chunks under {array_root}; sorting",
        flush=True,
    )
    tasks.sort(key=lambda task: task.coordinate_zyx)
    print(f"coordinate-ordered {len(tasks):,} present chunks", flush=True)
    return tasks


def _load_completed_rows(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            text = line.strip()
            if not text:
                continue
            try:
                completed.add(str(json.loads(text)["relative_path"]))
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"{path}:{line_number}: corrupt index row") from error
    return completed


def _accept_legacy_index_identity(
    root: Path,
    stored: dict[str, Any],
    current: dict[str, Any],
    *,
    zarray_sha256: str,
    rows_path: Path,
) -> dict[str, Any] | None:
    """Bridge the old byte-hash pin to the semantic pin, or refuse.

    The pre-2026-08-30 pin hashed the ``.zarray`` file's bytes, which makes any
    metadata change -- including a deliberate, certified one -- indistinguishable
    from corruption. Two bridges are allowed, and nothing else:

    * the live ``.zarray`` still hashes to what the pin recorded, so only the
      pin's own schema moved and the store never did; or
    * a completed compaction record chains the recorded hash to the live one,
      its certificate is present and hashes as recorded, and the inventory is
      byte-identical to the one the certificate signed.

    Returning ``None`` means the caller raises, so an unexplained change still
    fails closed.
    """

    if "zarray_sha256" not in stored:
        return None
    for key in (
        "positive_labels",
        "ignore_labels",
        "row_order",
        "zero_length_chunk_policy",
    ):
        if stored.get(key) != current.get(key):
            return None
    array_now = current["array_identity_v2"]
    if (
        stored.get("shape_zyx") != array_now["shape_zyx"]
        or stored.get("chunks_zyx") != array_now["chunks_zyx"]
        or stored.get("dimension_separator") != array_now["dimension_separator"]
    ):
        return None
    if stored["zarray_sha256"] == zarray_sha256:
        return dict(current)
    record_path = root / COMPACTION_STATE
    if not record_path.is_file():
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("state") != "complete" or not record.get("transform"):
        return None
    if str((record.get("from") or {}).get("zarray_sha256")) != stored["zarray_sha256"]:
        return None
    if str((record.get("to") or {}).get("zarray_sha256")) != zarray_sha256:
        return None
    certificate = Path(str(record.get("certificate") or ""))
    if not certificate.is_file():
        return None
    if _sha256(certificate) != record.get("certificate_sha256"):
        return None
    if _sha256(rows_path) != record.get("inventory_sha256"):
        return None
    return dict(current)


def index_label_zarr(
    *,
    zarr_path: str | Path,
    positive_labels: tuple[int, ...],
    ignore_labels: tuple[int, ...] = (2,),
    workers: int = 16,
    max_cpu_threads: int = 16,
) -> Path:
    """Audit every present L0 chunk and create a positive-aware support index."""

    if not 1 <= workers <= 16 or not 1 <= max_cpu_threads <= 16:
        raise ValueError("workers and max_cpu_threads must be in [1, 16]")
    if workers > max_cpu_threads:
        raise ValueError("workers cannot exceed max_cpu_threads")
    if set(positive_labels) & set(ignore_labels):
        raise ValueError("positive_labels and ignore_labels overlap")
    # The workers are threads performing independent chunk decompressions, not
    # Torch loader processes.  Leave Torch at one thread while the pool is
    # active, including for the workers == max_cpu_threads case.
    configure_cpu_budget(
        max_cpu_threads,
        reserve_processes=min(workers, max_cpu_threads - 1),
    )
    root = Path(zarr_path).expanduser().resolve()
    array_root = root / "0"
    zarray_path = array_root / ".zarray"
    metadata = json.loads(zarray_path.read_text(encoding="utf-8"))
    shape = tuple(int(item) for item in metadata["shape"])
    chunks = tuple(int(item) for item in metadata["chunks"])
    if len(shape) != 3 or len(chunks) != 3:
        raise ValueError(f"{zarray_path}: expected a three-dimensional array")
    grid = tuple(
        (extent + chunk - 1) // chunk
        for extent, chunk in zip(shape, chunks, strict=True)
    )
    separator = str(metadata.get("dimension_separator") or ".")
    if separator not in {".", "/"}:
        raise ValueError(f"{zarray_path}: unsupported dimension separator")
    dtype = np.dtype(metadata["dtype"])
    compressor_config = metadata.get("compressor")
    allowed = {0, *positive_labels, *ignore_labels}
    zarray_sha256 = _sha256(zarray_path)
    # The pin describes the array's *logical* identity plus how its chunks are
    # materialized, not the bytes of one metadata file. A store whose physical
    # layout changes under a signed certificate stays recognizable; a store that
    # changes for any other reason does not.
    identity = {
        "array_identity_v2": {
            "zarr_format": int(metadata.get("zarr_format", 2)),
            "shape_zyx": list(shape),
            "chunks_zyx": list(chunks),
            "dtype": str(metadata["dtype"]),
            "fill_value": metadata.get("fill_value"),
            "dimension_separator": separator,
        },
        "positive_labels": list(positive_labels),
        "ignore_labels": list(ignore_labels),
        "row_order": "coordinate-zyx",
        "zero_length_chunk_policy": ZERO_LENGTH_CHUNK_POLICY,
        "materialization_policy": (
            "unknown-fill-pruned-v1"
            if (root / COMPACTION_STATE).is_file()
            else "as-published-v1"
        ),
    }
    state_path = root / INDEX_STATE
    rows_path = root / INDEX_ROWS
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        stored_identity = dict(state.get("identity") or {})
        stored_identity.setdefault(
            "zero_length_chunk_policy", ZERO_LENGTH_CHUNK_POLICY
        )
        if stored_identity != identity:
            migrated = _accept_legacy_index_identity(
                root,
                stored_identity,
                identity,
                zarray_sha256=zarray_sha256,
                rows_path=rows_path,
            )
            if migrated is None:
                raise ValueError(f"{root}: label-index identity changed")
            if state.get("state") != "complete":
                raise ValueError(
                    f"{root}: identity migration requires a complete index, "
                    f"found state {state.get('state')!r}"
                )
            state["identity"] = migrated
            state["lineage"] = [
                *(state.get("lineage") or []),
                {
                    "migrated_from": stored_identity,
                    "zarray_sha256_before": stored_identity["zarray_sha256"],
                    "zarray_sha256_after": zarray_sha256,
                    "compaction_record": (
                        str(root / COMPACTION_STATE)
                        if (root / COMPACTION_STATE).is_file()
                        else None
                    ),
                },
            ]
            _atomic_json(state_path, state)
            return rows_path
        if state.get("state") == "complete":
            return rows_path
    elif rows_path.exists():
        raise ValueError(f"{root}: index rows exist without an index state")
    else:
        _atomic_json(
            state_path,
            {"state": "indexing", "identity": identity, "completed_chunks": 0},
        )

    tasks = _discover_chunks(
        array_root, separator=separator, grid_zyx=grid
    )
    completed = _load_completed_rows(rows_path)
    pending = [task for task in tasks if task.relative_path not in completed]
    local = threading.local()

    def inspect(task: ChunkTask) -> dict[str, Any]:
        if not hasattr(local, "codec"):
            if compressor_config is None:
                local.codec = None
            else:
                from numcodecs import get_codec

                local.codec = get_codec(compressor_config)
        encoded = task.path.read_bytes()
        if not encoded:
            return {
                "kind": "chunk",
                "relative_path": task.relative_path,
                "coordinate_zyx": list(task.coordinate_zyx),
                "size": 0,
                "decoded_voxels": 0,
                "known_voxels": 0,
                "positive_voxels": 0,
                "ignored_voxels": 0,
                "background_voxels": 0,
                "observed_labels": {},
                "storage": "zero-byte-unknown-placeholder",
            }
        try:
            decoded = (
                local.codec.decode(encoded) if local.codec is not None else encoded
            )
        except Exception as error:
            raise ValueError(
                f"{task.path}: failed to decode compressed label chunk"
            ) from error
        array = np.frombuffer(decoded, dtype=dtype)
        labels, counts = np.unique(array, return_counts=True)
        observed = {
            int(label): int(count)
            for label, count in zip(labels.tolist(), counts.tolist(), strict=True)
        }
        unexpected = set(observed) - allowed
        if unexpected:
            raise ValueError(
                f"{task.path}: unexpected labels {sorted(unexpected)}; "
                f"allowed={sorted(allowed)}"
            )
        positive = sum(observed.get(label, 0) for label in positive_labels)
        ignored = sum(observed.get(label, 0) for label in ignore_labels)
        known = int(array.size) - ignored
        return {
            "kind": "chunk",
            "relative_path": task.relative_path,
            "coordinate_zyx": list(task.coordinate_zyx),
            "size": len(encoded),
            "decoded_voxels": int(array.size),
            "known_voxels": known,
            "positive_voxels": positive,
            "ignored_voxels": ignored,
            "background_voxels": known - positive,
            "observed_labels": {
                str(label): count for label, count in sorted(observed.items())
            },
        }

    completed_count = len(completed)
    total_positive = 0
    total_known = 0
    if completed:
        with rows_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    total_positive += int(row["positive_voxels"])
                    total_known += int(row["known_voxels"])
    with ThreadPoolExecutor(max_workers=workers) as executor, rows_path.open(
        "a", encoding="utf-8", newline="\n"
    ) as stream:
        for row in bounded_thread_map_ordered(
            executor,
            inspect,
            pending,
            max_pending=max(32, workers * 4),
        ):
            stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
            completed_count += 1
            total_positive += int(row["positive_voxels"])
            total_known += int(row["known_voxels"])
            if completed_count % 1000 == 0:
                stream.flush()
                os.fsync(stream.fileno())
                _atomic_json(
                    state_path,
                    {
                        "state": "indexing",
                        "identity": identity,
                        "completed_chunks": completed_count,
                        "total_chunks": len(tasks),
                        "positive_voxels": total_positive,
                        "known_voxels": total_known,
                    },
                )
                print(
                    f"indexed {completed_count:,}/{len(tasks):,} chunks; "
                    f"positive={total_positive:,}",
                    flush=True,
                )
        stream.flush()
        os.fsync(stream.fileno())
    if total_positive <= 0:
        raise ValueError(f"{root}: no positive manual-label voxels found")
    _atomic_json(
        state_path,
        {
            "state": "complete",
            "identity": identity,
            "completed_chunks": completed_count,
            "total_chunks": len(tasks),
            "positive_voxels": total_positive,
            "known_voxels": total_known,
        },
    )
    return rows_path


def index_catalog(
    *,
    labels_root: str | Path,
    catalog_path: str | Path,
    dataset_ids: set[str] | None = None,
    workers: int = 16,
    max_cpu_threads: int = 16,
) -> list[Path]:
    catalog = load_catalog(catalog_path)
    selected = [
        item
        for item in catalog["datasets"]
        if dataset_ids is None or str(item["dataset_id"]) in dataset_ids
    ]
    if dataset_ids is not None:
        found = {str(item["dataset_id"]) for item in selected}
        if found != dataset_ids:
            raise ValueError(f"unknown dataset IDs: {sorted(dataset_ids - found)}")
    root = Path(labels_root).expanduser().resolve()
    return [
        index_label_zarr(
            zarr_path=root / str(item["zarr"]),
            positive_labels=tuple(int(value) for value in item["positive_labels"]),
            ignore_labels=tuple(int(value) for value in item["ignore_labels"]),
            workers=workers,
            max_cpu_threads=max_cpu_threads,
        )
        for item in selected
    ]
