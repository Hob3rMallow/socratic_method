from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import s3fs

from crossres_pred.concurrency import ByteRateLimiter, bounded_thread_map
from crossres_pred.sparse_zarr import ZarrArraySpec

from .official_corpus import load_manual_pair_plan
from .registration import transform_xyz
from .resources import configure_cpu_budget


@dataclass(frozen=True)
class RemoteObject:
    key: str
    relative_path: str
    size: int
    kind: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _split_s3(uri: str) -> str:
    if not uri.startswith("s3://"):
        raise ValueError(f"selective mirroring currently requires s3://, got {uri}")
    return uri.removeprefix("s3://").rstrip("/")


def _positive_label_chunks(inventory: Path) -> list[tuple[int, int, int]]:
    coordinates: list[tuple[int, int, int]] = []
    with inventory.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") != "chunk":
                continue
            # Indexed human-label inventories provide exact counts. Native
            # teacher mirrors historically omitted the count, but every
            # materialized teacher chunk passed a positive-voxel acceptance
            # gate and is therefore a valid center.
            if "positive_voxels" in row and int(row["positive_voxels"]) <= 0:
                continue
            raw_coordinate = row.get("coordinate_zyx")
            if raw_coordinate is None:
                relative_path = row.get("relative_path")
                if not isinstance(relative_path, str):
                    raise ValueError(
                        f"{inventory}:{line_number}: chunk has no coordinate"
                    )
                path_parts = PurePosixPath(relative_path).parts
                if len(path_parts) < 4:
                    raise ValueError(
                        f"{inventory}:{line_number}: invalid chunk path "
                        f"{relative_path!r}"
                    )
                raw_coordinate = path_parts[-3:]
            try:
                coordinate = tuple(int(value) for value in raw_coordinate)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{inventory}:{line_number}: invalid coordinate"
                ) from error
            if len(coordinate) != 3:
                raise ValueError(f"{inventory}:{line_number}: invalid coordinate")
            if any(value < 0 for value in coordinate):
                raise ValueError(
                    f"{inventory}:{line_number}: coordinate must be non-negative"
                )
            coordinates.append(coordinate)  # type: ignore[arg-type]
    if not coordinates:
        raise ValueError(f"{inventory}: no positive label chunks")
    return coordinates


def select_registered_chunks(
    *,
    label_zarray: str | Path,
    label_inventory: str | Path,
    fine_to_source_affine_xyz: list[list[float]],
    source_spec: ZarrArraySpec,
    halo_voxels: int,
) -> frozenset[int]:
    """Map every positive fine-label chunk into a haloed source chunk set."""

    if halo_voxels < 0:
        raise ValueError("halo_voxels must be non-negative")
    metadata = json.loads(Path(label_zarray).read_text(encoding="utf-8"))
    fine_shape = np.asarray(metadata["shape"], dtype=np.int64)
    fine_chunks = np.asarray(metadata["chunks"], dtype=np.int64)
    if fine_shape.shape != (3,) or fine_chunks.shape != (3,):
        raise ValueError("manual labels must be three-dimensional")
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :] = np.asarray(fine_to_source_affine_xyz, dtype=np.float64)
    source_chunks = np.asarray(source_spec.chunks_zyx, dtype=np.int64)
    source_grid = np.asarray(source_spec.chunk_grid_zyx, dtype=np.int64)
    selected: set[int] = set()

    for coordinate in _positive_label_chunks(Path(label_inventory)):
        lower = np.asarray(coordinate, dtype=np.int64) * fine_chunks
        upper = np.minimum(lower + fine_chunks, fine_shape) - 1
        corners_zyx = np.asarray(
            list(
                product(
                    (lower[0] - 0.5, upper[0] + 0.5),
                    (lower[1] - 0.5, upper[1] + 0.5),
                    (lower[2] - 0.5, upper[2] + 0.5),
                )
            ),
            dtype=np.float64,
        )
        source_xyz = transform_xyz(corners_zyx[:, ::-1], affine)
        source_zyx = source_xyz[:, ::-1]
        voxel_lower = np.floor(source_zyx.min(axis=0) - halo_voxels).astype(np.int64)
        voxel_upper = np.ceil(source_zyx.max(axis=0) + halo_voxels).astype(np.int64)
        chunk_lower = np.maximum(np.floor_divide(voxel_lower, source_chunks), 0)
        chunk_upper = np.minimum(
            np.floor_divide(voxel_upper, source_chunks) + 1,
            source_grid,
        )
        for chunk_coordinate in product(
            range(int(chunk_lower[0]), int(chunk_upper[0])),
            range(int(chunk_lower[1]), int(chunk_upper[1])),
            range(int(chunk_lower[2]), int(chunk_upper[2])),
        ):
            selected.add(source_spec.encode_chunk(chunk_coordinate))
    return frozenset(selected)


def _probe(
    fs: s3fs.S3FileSystem,
    key: str,
    relative_path: str,
    kind: str,
) -> RemoteObject | None:
    for attempt in range(5):
        try:
            info = fs.info(key)
            if info.get("type") != "file":
                return None
            return RemoteObject(
                key=str(info.get("name") or key),
                relative_path=relative_path,
                size=int(info.get("size") or 0),
                kind=kind,
            )
        except FileNotFoundError:
            return None
        except Exception:
            if attempt == 4:
                raise
            time.sleep(0.25 * (attempt + 1))
    return None


def _inventory_chunks(
    fs: s3fs.S3FileSystem,
    *,
    remote_root: str,
    array_key: str,
    spec: ZarrArraySpec,
    selected: frozenset[int],
    workers: int,
) -> list[RemoteObject]:
    array_root = f"{remote_root}/{array_key}"
    if spec.dimension_separator == ".":
        def inspect(chunk_id: int) -> RemoteObject | None:
            relative = f"{array_key}/{spec.chunk_key(chunk_id)}"
            return _probe(fs, f"{remote_root}/{relative}", relative, "chunk")

        objects: list[RemoteObject] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for result in bounded_thread_map(
                executor, inspect, sorted(selected), max_pending=workers * 4
            ):
                if result is not None:
                    objects.append(result)
        return objects

    selected_by_z: dict[int, set[int]] = {}
    for chunk_id in selected:
        z_index, _, _ = spec.decode_chunk(chunk_id)
        selected_by_z.setdefault(z_index, set()).add(chunk_id)

    def list_prefix(z_index: int, wanted: set[int]) -> list[RemoteObject]:
        prefix = f"{array_root}/{z_index}/"
        listing: dict[str, dict[str, Any]] | None = None
        for attempt in range(5):
            try:
                listing = fs.find(prefix, detail=True)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(0.5 * (attempt + 1))
        assert listing is not None
        result: list[RemoteObject] = []
        for key, info in listing.items():
            relative_chunk = key[len(array_root) + 1 :]
            parts = relative_chunk.split("/")
            if len(parts) != 3 or info.get("type") != "file":
                continue
            try:
                chunk_id = spec.encode_chunk(tuple(int(value) for value in parts))
            except ValueError:
                continue
            if chunk_id in wanted:
                result.append(
                    RemoteObject(
                        key=key,
                        relative_path=f"{array_key}/{relative_chunk}",
                        size=int(info.get("size") or 0),
                        kind="chunk",
                    )
                )
        fs.invalidate_cache(prefix)
        return result

    objects: list[RemoteObject] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(list_prefix, z_index, wanted)
            for z_index, wanted in sorted(selected_by_z.items())
        ]
        for index, future in enumerate(as_completed(futures), 1):
            objects.extend(future.result())
            if index % 10 == 0 or index == len(futures):
                print(
                    f"  listed {index:,}/{len(futures):,} z-prefixes; "
                    f"{len(objects):,} selected objects",
                    flush=True,
                )
    return objects


def _metadata_objects(
    fs: s3fs.S3FileSystem, remote_root: str, array_key: str
) -> list[RemoteObject]:
    objects: list[RemoteObject] = []
    for relative, required in (
        (".zgroup", True),
        (".zattrs", False),
        (f"{array_key}/.zarray", True),
        (f"{array_key}/.zattrs", False),
    ):
        item = _probe(fs, f"{remote_root}/{relative}", relative, "metadata")
        if item is None and required:
            raise FileNotFoundError(f"s3://{remote_root}/{relative}")
        if item is not None:
            objects.append(item)
    return objects


def _download(
    fs: s3fs.S3FileSystem,
    task: RemoteObject,
    output: Path,
    rate_limiter: ByteRateLimiter | None = None,
) -> tuple[int, str]:
    destination = output.joinpath(*task.relative_path.split("/"))
    if destination.is_file() and destination.stat().st_size == task.size:
        return 0, "skip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    for attempt in range(5):
        try:
            if rate_limiter is None:
                fs.get_file(task.key, str(temporary))
            else:
                with fs.open(task.key, "rb") as source, temporary.open("wb") as sink:
                    while block := source.read(64 * 1024):
                        rate_limiter.wait_for(len(block))
                        sink.write(block)
            if temporary.stat().st_size != task.size:
                raise OSError(
                    f"size mismatch {temporary.stat().st_size} != {task.size}"
                )
            os.replace(temporary, destination)
            return task.size, "ok"
        except Exception as error:  # noqa: BLE001 - retry transient public S3 failures
            temporary.unlink(missing_ok=True)
            if attempt == 4:
                return 0, f"FAIL {type(error).__name__}: {error}"
            time.sleep(0.5 * (attempt + 1))
    return 0, "FAIL unreachable"


def _plan_hash(objects: list[RemoteObject]) -> str:
    digest = hashlib.sha256()
    for task in sorted(objects, key=lambda item: item.relative_path):
        digest.update(f"{task.relative_path}\t{task.size}\n".encode())
    return digest.hexdigest()


def validate_registered_mirror(
    mirror_path: str | Path, *, array_key: str = "0"
) -> dict[str, Any]:
    """Prove a selective registered mirror's plan and physical objects."""

    output = Path(mirror_path).expanduser().resolve()
    key = array_key.strip("/")
    if not key:
        raise ValueError("array_key cannot be empty")
    safe_key = key.replace("/", "_")
    state_path = output / f"crossres_registered_mirror_{safe_key}.json"
    rows_path = output / f"crossres_registered_objects_{safe_key}.jsonl"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("state") != "complete":
        raise ValueError(
            f"{state_path}: mirror state is {state.get('state')!r}, not complete"
        )
    identity = state.get("identity")
    if not isinstance(identity, dict):
        raise TypeError(f"{state_path}: identity must be an object")
    if identity.get("array_key") != key:
        raise ValueError(
            f"{state_path}: identity.array_key {identity.get('array_key')!r} "
            f"!= {key!r}"
        )
    transfer = state.get("transfer")
    if not isinstance(transfer, dict) or transfer.get("failures") != []:
        raise ValueError(f"{state_path}: transfer failures are not empty")

    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    chunk_count = 0
    chunk_bytes = 0
    previous_path: str | None = None
    physical_errors: list[str] = []
    with rows_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.rstrip("\r\n")
            if not line:
                raise ValueError(f"{rows_path}:{line_number}: blank object-plan row")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{rows_path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(row, dict):
                raise TypeError(f"{rows_path}:{line_number}: row must be an object")
            kind = row.get("kind")
            relative_text = row.get("relative_path")
            size = row.get("size")
            if kind not in {"metadata", "chunk"}:
                raise ValueError(
                    f"{rows_path}:{line_number}: invalid object kind {kind!r}"
                )
            if not isinstance(relative_text, str) or not relative_text:
                raise ValueError(
                    f"{rows_path}:{line_number}: relative_path must be non-empty"
                )
            relative = PurePosixPath(relative_text)
            if (
                relative.is_absolute()
                or relative.as_posix() != relative_text
                or "\\" in relative_text
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                raise ValueError(
                    f"{rows_path}:{line_number}: unsafe relative path "
                    f"{relative_text!r}"
                )
            if previous_path is not None and relative_text <= previous_path:
                raise ValueError(
                    f"{rows_path}:{line_number}: object plan is not strictly sorted"
                )
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(
                    f"{rows_path}:{line_number}: size must be a non-negative integer"
                )
            previous_path = relative_text
            digest.update(f"{relative_text}\t{size}\n".encode())
            count += 1
            total_bytes += size
            if kind == "chunk":
                chunk_count += 1
                chunk_bytes += size

            object_path = output.joinpath(*relative.parts)
            try:
                object_stat = object_path.stat()
            except FileNotFoundError:
                if len(physical_errors) < 8:
                    physical_errors.append(f"{relative_text}: missing")
                continue
            if not stat.S_ISREG(object_stat.st_mode):
                if len(physical_errors) < 8:
                    physical_errors.append(f"{relative_text}: not a regular file")
            elif object_stat.st_size != size and len(physical_errors) < 8:
                physical_errors.append(
                    f"{relative_text}: size {object_stat.st_size} != {size}"
                )

    if count == 0:
        raise ValueError(f"{rows_path}: object plan is empty")
    if physical_errors:
        raise ValueError(
            f"{output}: registered mirror physical validation failed: "
            + "; ".join(physical_errors)
        )
    actual = {
        "objects": count,
        "bytes": total_bytes,
        "plan_sha256": digest.hexdigest(),
    }
    for name, actual_value in actual.items():
        if identity.get(name) != actual_value:
            raise ValueError(
                f"{state_path}: identity.{name} {identity.get(name)!r} "
                f"!= {actual_value!r}"
            )
    return {
        "schema": "crossres-registered-mirror-validation-v1",
        "root": str(output),
        "array_key": key,
        "manifest_sha256": _sha256(state_path),
        "object_plan_sha256": _sha256(rows_path),
        "count": count,
        "bytes": total_bytes,
        "chunk_count": chunk_count,
        "chunk_bytes": chunk_bytes,
        "plan_sha256": digest.hexdigest(),
    }


def validate_full_sparse_mirror(
    mirror_path: str | Path, *, array_key: str = "0"
) -> dict[str, Any]:
    """Prove a full sparse selector mirror, including reconstructed metadata."""

    output = Path(mirror_path).expanduser().resolve()
    key = array_key.strip("/")
    if not key:
        raise ValueError("array_key cannot be empty")
    safe_key = key.replace("/", "_")
    state_path = output / f"crossres_full_sparse_mirror_{safe_key}.json"
    rows_path = output / "crossres_sparse_objects.jsonl"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("state") != "complete":
        raise ValueError(
            f"{state_path}: mirror state is {state.get('state')!r}, not complete"
        )
    identity = state.get("identity")
    if not isinstance(identity, dict):
        raise TypeError(f"{state_path}: identity must be an object")
    if identity.get("array_key") != key:
        raise ValueError(
            f"{state_path}: identity.array_key {identity.get('array_key')!r} "
            f"!= {key!r}"
        )
    transfer = state.get("transfer")
    if not isinstance(transfer, dict) or transfer.get("failures") != []:
        raise ValueError(f"{state_path}: transfer failures are not empty")

    objects: list[RemoteObject] = []
    seen: set[str] = set()
    previous_path: str | None = None
    with rows_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.rstrip("\r\n")
            if not line:
                raise ValueError(f"{rows_path}:{line_number}: blank object-plan row")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{rows_path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(row, dict):
                raise TypeError(f"{rows_path}:{line_number}: row must be an object")
            if row.get("kind") != "chunk":
                raise ValueError(
                    f"{rows_path}:{line_number}: full sparse plan must contain chunks"
                )
            relative_text = row.get("relative_path")
            size = row.get("size")
            if not isinstance(relative_text, str) or not relative_text:
                raise ValueError(
                    f"{rows_path}:{line_number}: relative_path must be non-empty"
                )
            relative = PurePosixPath(relative_text)
            if (
                relative.is_absolute()
                or relative.as_posix() != relative_text
                or "\\" in relative_text
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                raise ValueError(
                    f"{rows_path}:{line_number}: unsafe relative path "
                    f"{relative_text!r}"
                )
            if not relative.parts or relative.parts[0] != key:
                raise ValueError(
                    f"{rows_path}:{line_number}: chunk is outside array {key!r}"
                )
            if previous_path is not None and relative_text <= previous_path:
                raise ValueError(
                    f"{rows_path}:{line_number}: object plan is not strictly sorted"
                )
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(
                    f"{rows_path}:{line_number}: size must be a non-negative integer"
                )
            previous_path = relative_text
            seen.add(relative_text)
            objects.append(
                RemoteObject(
                    key="",
                    relative_path=relative_text,
                    size=size,
                    kind="chunk",
                )
            )
    if not objects:
        raise ValueError(f"{rows_path}: object plan is empty")

    for relative_text, required in (
        (".zgroup", True),
        (".zattrs", False),
        (f"{key}/.zarray", True),
        (f"{key}/.zattrs", False),
    ):
        metadata_path = output.joinpath(*PurePosixPath(relative_text).parts)
        if not metadata_path.is_file():
            if required:
                raise ValueError(
                    f"{output}: required metadata is missing: {relative_text}"
                )
            continue
        if relative_text in seen:
            raise ValueError(f"{rows_path}: metadata path duplicated in chunk plan")
        objects.append(
            RemoteObject(
                key="",
                relative_path=relative_text,
                size=metadata_path.stat().st_size,
                kind="metadata",
            )
        )

    physical_errors: list[str] = []
    for item in objects:
        object_path = output.joinpath(*PurePosixPath(item.relative_path).parts)
        try:
            object_stat = object_path.stat()
        except FileNotFoundError:
            if len(physical_errors) < 8:
                physical_errors.append(f"{item.relative_path}: missing")
            continue
        if not stat.S_ISREG(object_stat.st_mode):
            if len(physical_errors) < 8:
                physical_errors.append(f"{item.relative_path}: not a regular file")
        elif object_stat.st_size != item.size and len(physical_errors) < 8:
            physical_errors.append(
                f"{item.relative_path}: size {object_stat.st_size} != {item.size}"
            )
    if physical_errors:
        raise ValueError(
            f"{output}: full sparse mirror physical validation failed: "
            + "; ".join(physical_errors)
        )

    chunk_count = sum(item.kind == "chunk" for item in objects)
    actual = {
        "objects": len(objects),
        "chunks": chunk_count,
        "bytes": sum(item.size for item in objects),
        "plan_sha256": _plan_hash(objects),
    }
    for name, actual_value in actual.items():
        if identity.get(name) != actual_value:
            raise ValueError(
                f"{state_path}: identity.{name} {identity.get(name)!r} "
                f"!= {actual_value!r}"
            )
    return {
        "schema": "crossres-full-sparse-mirror-validation-v1",
        "root": str(output),
        "array_key": key,
        "manifest_sha256": _sha256(state_path),
        "object_plan_sha256": _sha256(rows_path),
        "count": len(objects),
        "bytes": actual["bytes"],
        "chunk_count": chunk_count,
        "plan_sha256": actual["plan_sha256"],
    }


def _mirror_group(
    *,
    group: dict[str, Any],
    workers: int,
    halo_voxels: int,
    dry_run: bool,
    rate_limiter: ByteRateLimiter | None,
) -> dict[str, Any]:
    output = Path(group["local_zarr"])
    if group["preexisting"]:
        print(f"reuse complete local input: {output}", flush=True)
        return {"state": "preexisting", "output": str(output)}
    remote_root = _split_s3(group["uri"])
    array_key = str(group["array_key"])
    fs = s3fs.S3FileSystem(anon=True)
    raw_zarray = fs.cat_file(f"{remote_root}/{array_key}/.zarray")
    spec = ZarrArraySpec.from_metadata(raw_zarray)
    selected: set[int] = set()
    uses: list[dict[str, Any]] = []
    for use in group["uses"]:
        fine = use["fine"]
        inventory = Path(fine["label_inventory"])
        chunks = select_registered_chunks(
            label_zarray=Path(fine["label_zarr"])
            / fine["label_array_key"]
            / ".zarray",
            label_inventory=inventory,
            fine_to_source_affine_xyz=fine["to_coarse_affine_xyz"],
            source_spec=spec,
            halo_voxels=halo_voxels,
        )
        selected.update(chunks)
        uses.append(
            {
                "pair_id": use["pair_id"],
                "label_inventory": str(inventory),
                "label_inventory_sha256": _sha256(inventory),
                "selected_chunks": len(chunks),
            }
        )
    selected_ids = frozenset(selected)
    print(
        f"{group['kind']} {group['uri']}::{array_key}: "
        f"{len(selected_ids):,} haloed chunks",
        flush=True,
    )
    chunks = _inventory_chunks(
        fs,
        remote_root=remote_root,
        array_key=array_key,
        spec=spec,
        selected=selected_ids,
        workers=workers,
    )
    objects = _metadata_objects(fs, remote_root, array_key) + chunks
    identity = {
        "source": group["uri"],
        "array_key": array_key,
        "shape_zyx": list(spec.shape_zyx),
        "chunks_zyx": list(spec.chunks_zyx),
        "halo_voxels": halo_voxels,
        "uses": uses,
        "selected_chunks": len(selected_ids),
        "existing_chunks": len(chunks),
        "objects": len(objects),
        "bytes": sum(item.size for item in objects),
        "plan_sha256": _plan_hash(objects),
    }
    print(
        f"  plan {len(chunks):,}/{len(selected_ids):,} existing chunks, "
        f"{identity['bytes'] / 2**30:.2f} GiB",
        flush=True,
    )
    if dry_run:
        return {"state": "planned", **identity, "output": str(output)}

    safe_key = array_key.replace("/", "_")
    state_path = output / f"crossres_registered_mirror_{safe_key}.json"
    rows_path = output / f"crossres_registered_objects_{safe_key}.jsonl"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("identity") != identity:
            raise ValueError(f"{output}: registered mirror identity changed")
        if state.get("state") == "complete":
            validate_registered_mirror(output, array_key=array_key)
            return state
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(state_path, {"state": "downloading", "identity": identity})
        with rows_path.open("w", encoding="utf-8", newline="\n") as stream:
            for item in sorted(objects, key=lambda value: value.relative_path):
                stream.write(
                    json.dumps(
                        {
                            "kind": item.kind,
                            "relative_path": item.relative_path,
                            "size": item.size,
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )

    downloaded = 0
    skipped = 0
    failures: list[str] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(
            bounded_thread_map(
                executor,
                lambda item: (item, _download(fs, item, output, rate_limiter)),
                objects,
                max_pending=workers * 4,
            ),
            1,
        ):
            item, (size, status) = result
            downloaded += size
            if status == "skip":
                skipped += 1
            elif status != "ok":
                failures.append(f"{item.relative_path}: {status}")
            if index % 200 == 0 or index == len(objects):
                elapsed = max(time.perf_counter() - started, 1.0e-6)
                print(
                    f"  {index:,}/{len(objects):,} new={downloaded / 2**30:.2f} GiB "
                    f"skip={skipped:,} fail={len(failures):,} "
                    f"rate={downloaded / elapsed / 2**20:.2f} MiB/s",
                    flush=True,
                )
    state = {
        "state": "failed" if failures else "complete",
        "identity": identity,
        "transfer": {
            "downloaded_bytes_this_run": downloaded,
            "skipped_objects": skipped,
            "failures": failures,
            "seconds": time.perf_counter() - started,
        },
    }
    _atomic_json(state_path, state)
    if failures:
        raise RuntimeError(f"{output}: {len(failures)} downloads failed")
    validate_registered_mirror(output, array_key=array_key)
    return state


def _input_groups(plan: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for pair in plan["pairs"]:
        coarse = pair["coarse"]
        entries = [("image", coarse)]
        if coarse.get("baseline") is not None:
            entries.append(("baseline", coarse["baseline"]))
        for kind, entry in entries:
            key = (
                kind,
                str(entry["uri"]),
                str(entry["array_key"]),
                str(entry["local_zarr"]),
            )
            group = groups.setdefault(
                key,
                {
                    "kind": kind,
                    "uri": entry["uri"],
                    "array_key": entry["array_key"],
                    "local_zarr": entry["local_zarr"],
                    "preexisting": bool(entry["preexisting"]),
                    "uses": [],
                },
            )
            group["uses"].append(pair)
    return list(groups.values())


def mirror_manual_pair_inputs(
    *,
    plan_path: str | Path,
    workers: int = 8,
    halo_voxels: int = 192,
    max_cpu_threads: int = 16,
    max_mib_per_second: float | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    if not 1 <= workers <= 16:
        raise ValueError("workers must be in [1, 16]")
    if not 1 <= max_cpu_threads <= 16 or workers > max_cpu_threads:
        raise ValueError("CPU budget must be in [workers, 16]")
    configure_cpu_budget(
        max_cpu_threads, reserve_processes=min(workers, max_cpu_threads - 1)
    )
    if max_mib_per_second is not None and max_mib_per_second <= 0:
        raise ValueError("max_mib_per_second must be positive when supplied")
    rate_limiter = (
        ByteRateLimiter(max_mib_per_second)
        if max_mib_per_second is not None
        else None
    )
    plan = load_manual_pair_plan(plan_path)
    return [
        _mirror_group(
            group=group,
            workers=workers,
            halo_voxels=halo_voxels,
            dry_run=dry_run,
            rate_limiter=rate_limiter,
        )
        for group in _input_groups(plan)
    ]


def mirror_full_sparse_zarr(
    *,
    source_uri: str,
    output_path: str | Path,
    array_key: str = "0",
    workers: int = 8,
    max_cpu_threads: int = 16,
    max_mib_per_second: float | None = None,
) -> Path:
    """Resume-safe mirror of one complete sparse Zarr array and its metadata."""

    if not 1 <= workers <= 16:
        raise ValueError("workers must be in [1, 16]")
    if not 1 <= max_cpu_threads <= 16 or workers > max_cpu_threads:
        raise ValueError("CPU budget must be in [workers, 16]")
    configure_cpu_budget(
        max_cpu_threads, reserve_processes=min(workers, max_cpu_threads - 1)
    )
    if max_mib_per_second is not None and max_mib_per_second <= 0:
        raise ValueError("max_mib_per_second must be positive when supplied")
    rate_limiter = (
        ByteRateLimiter(max_mib_per_second)
        if max_mib_per_second is not None
        else None
    )
    remote_root = _split_s3(source_uri)
    key = array_key.strip("/")
    if not key:
        raise ValueError("array_key cannot be empty")
    output = Path(output_path).expanduser().resolve()
    safe_key = key.replace("/", "_")
    state_path = output / f"crossres_full_sparse_mirror_{safe_key}.json"
    rows_path = output / "crossres_sparse_objects.jsonl"
    fs = s3fs.S3FileSystem(anon=True)
    listing = fs.find(f"{remote_root}/{key}", detail=True)
    chunks: list[RemoteObject] = []
    prefix_length = len(remote_root) + 1
    for remote_key, info in listing.items():
        relative = remote_key[prefix_length:]
        if info.get("type") != "file" or Path(relative).name.startswith("."):
            continue
        chunks.append(
            RemoteObject(
                key=remote_key,
                relative_path=relative,
                size=int(info.get("size") or 0),
                kind="chunk",
            )
        )
    objects = _metadata_objects(fs, remote_root, key) + chunks
    identity = {
        "source": source_uri.rstrip("/"),
        "array_key": key,
        "objects": len(objects),
        "chunks": len(chunks),
        "bytes": sum(item.size for item in objects),
        "plan_sha256": _plan_hash(objects),
    }
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("identity") != identity:
            raise ValueError(f"{output}: full sparse mirror identity changed")
        if state.get("state") == "complete":
            validate_full_sparse_mirror(output, array_key=key)
            return state_path
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(state_path, {"state": "downloading", "identity": identity})
        with rows_path.open("w", encoding="utf-8", newline="\n") as stream:
            for item in sorted(chunks, key=lambda value: value.relative_path):
                stream.write(
                    json.dumps(
                        {
                            "kind": "chunk",
                            "relative_path": item.relative_path,
                            "size": item.size,
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )

    print(
        f"full sparse plan: {len(chunks):,} chunks, "
        f"{identity['bytes'] / 2**30:.2f} GiB",
        flush=True,
    )
    downloaded = 0
    skipped = 0
    failures: list[str] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(
            bounded_thread_map(
                executor,
                lambda item: (item, _download(fs, item, output, rate_limiter)),
                objects,
                max_pending=workers * 4,
            ),
            1,
        ):
            item, (size, status) = result
            downloaded += size
            if status == "skip":
                skipped += 1
            elif status != "ok":
                failures.append(f"{item.relative_path}: {status}")
            if index % 200 == 0 or index == len(objects):
                elapsed = max(time.perf_counter() - started, 1.0e-6)
                print(
                    f"  {index:,}/{len(objects):,} new={downloaded / 2**30:.2f} GiB "
                    f"skip={skipped:,} fail={len(failures):,} "
                    f"rate={downloaded / elapsed / 2**20:.2f} MiB/s",
                    flush=True,
                )
    state = {
        "state": "failed" if failures else "complete",
        "identity": identity,
        "transfer": {
            "downloaded_bytes_this_run": downloaded,
            "skipped_objects": skipped,
            "failures": failures,
            "seconds": time.perf_counter() - started,
        },
    }
    _atomic_json(state_path, state)
    if failures:
        raise RuntimeError(f"{output}: {len(failures)} downloads failed")
    validate_full_sparse_mirror(output, array_key=key)
    return state_path
