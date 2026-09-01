from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .concurrency import bounded_thread_map
from .provenance import utc_now, write_json_atomic
from .sparse_zarr import ZarrArraySpec, expand_chunk_ids, select_tifxyz_chunks


class CarveError(RuntimeError):
    pass


@dataclass(frozen=True)
class CarveOptions:
    """Site-footprint carve of a remote fine raw volume.

    Full surface-tube carving of the uncompressed fine raw zarrs is
    infeasible (~15-25 TiB across the pair scrolls), so the carve fetches
    only chunks under the affine-image footprints of planned sites,
    intersected with a dilated surface-tube selection to drop papyrus-free
    footprint corners. ``max_bytes`` is a hard refusal ceiling evaluated on
    the upper-bound estimate before any transfer.
    """

    array_key: str = "0"
    workers: int = 8
    tube_intersect: bool = True
    tube_stride: int = 1
    tube_dilate_vox: int = 128
    max_bytes: int = 350 * 1024**3
    retries: int = 5

    def validate(self) -> None:
        if not 1 <= self.workers <= 8:
            raise CarveError("workers must be in [1, 8] (S3 politeness cap)")
        if self.tube_stride < 1:
            raise CarveError("tube_stride must be >= 1")
        if self.tube_dilate_vox < 0:
            raise CarveError("tube_dilate_vox must be non-negative")
        if self.max_bytes <= 0:
            raise CarveError("max_bytes must be positive")
        if self.retries < 1:
            raise CarveError("retries must be >= 1")


@dataclass(frozen=True)
class RemoteObject:
    key: str
    relative_path: str
    size: int
    kind: str


def human_bytes(size: float) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"


class Store:
    """Minimal read-only object-store view over S3 or a local directory.

    The local implementation exists so the carve logic is unit-testable
    without network access; the S3 implementation is anonymous.
    """

    def __init__(self, source: str) -> None:
        self.source = source.rstrip("/")
        if self.source.startswith("s3://"):
            import s3fs

            self._fs = s3fs.S3FileSystem(anon=True)
            self._root = self.source.removeprefix("s3://")
            self._local: Path | None = None
        else:
            self._fs = None
            self._root = ""
            self._local = Path(self.source).expanduser().resolve()
            if not self._local.is_dir():
                raise CarveError(f"local store does not exist: {self._local}")

    def cat(self, relative_path: str) -> bytes:
        if self._local is not None:
            return (self._local / relative_path).read_bytes()
        return self._fs.cat_file(f"{self._root}/{relative_path}")

    def info(self, relative_path: str) -> dict[str, Any] | None:
        if self._local is not None:
            path = self._local / relative_path
            if not path.is_file():
                return None
            return {"size": path.stat().st_size, "etag": None}
        try:
            value = self._fs.info(f"{self._root}/{relative_path}")
        except FileNotFoundError:
            return None
        if value.get("type") != "file":
            return None
        etag = value.get("ETag")
        return {
            "size": int(value.get("size") or 0),
            "etag": str(etag).strip('"') if etag else None,
        }

    def list_metadata(self) -> list[RemoteObject]:
        objects: list[RemoteObject] = []
        if self._local is not None:
            for child in self._local.iterdir():
                if child.is_file():
                    objects.append(
                        RemoteObject(
                            key=str(child),
                            relative_path=child.name,
                            size=child.stat().st_size,
                            kind="metadata",
                        )
                    )
            return objects
        for value in self._fs.ls(self._root, detail=True):
            if value.get("type") != "file":
                continue
            key = str(value["name"])
            objects.append(
                RemoteObject(
                    key=key,
                    relative_path=key[len(self._root) + 1 :],
                    size=int(value.get("size") or 0),
                    kind="metadata",
                )
            )
        return objects

    def find_selected_chunks(
        self,
        array_key: str,
        selected: frozenset[int],
        spec: ZarrArraySpec,
        workers: int,
    ) -> list[RemoteObject]:
        """Locate selected chunks with bounded-memory z/y-prefix listings.

        A z-only S3 ``find`` can materialize millions of unrelated chunk
        records for a large tomography volume.  Grouping by both z and y keeps
        each remote listing narrow while retaining the important distinction
        between absent (fill-value) and present chunks.
        """

        if self._local is not None:
            objects = []
            for chunk_id in sorted(selected):
                relative = f"{array_key}/{spec.chunk_key(chunk_id)}"
                path = self._local / relative
                if path.is_file():
                    objects.append(
                        RemoteObject(
                            key=str(path),
                            relative_path=relative,
                            size=path.stat().st_size,
                            kind="chunk",
                        )
                    )
            return objects
        if spec.dimension_separator != "/":
            raise CarveError(
                "S3 carve requires '/'-separated chunk keys for prefix listing"
            )
        array_root = f"{self._root}/{array_key}"
        selected_by_zy: dict[tuple[int, int], set[int]] = {}
        for chunk_id in selected:
            z_index, y_index, _ = spec.decode_chunk(chunk_id)
            selected_by_zy.setdefault((z_index, y_index), set()).add(chunk_id)

        def list_zy(
            coordinate: tuple[int, int], wanted: set[int]
        ) -> list[RemoteObject]:
            z_index, y_index = coordinate
            prefix = f"{array_root}/{z_index}/{y_index}/"
            last_error: Exception | None = None
            for attempt in range(5):
                try:
                    listing = self._fs.find(prefix, detail=True)
                    break
                except Exception as error:  # noqa: BLE001 - transient S3
                    last_error = error
                    time.sleep(0.5 * (attempt + 1))
            else:
                raise CarveError(
                    f"failed to list s3://{prefix}: {last_error}"
                ) from last_error
            found: list[RemoteObject] = []
            for key, value in listing.items():
                if value.get("type") != "file":
                    continue
                relative_chunk = key[len(array_root) + 1 :]
                parts = relative_chunk.split("/")
                if len(parts) != 3:
                    continue
                try:
                    chunk_id = spec.encode_chunk(
                        tuple(int(item) for item in parts)
                    )
                except (ValueError, Exception):
                    continue
                if chunk_id not in wanted:
                    continue
                found.append(
                    RemoteObject(
                        key=key,
                        relative_path=f"{array_key}/{relative_chunk}",
                        size=int(value.get("size") or 0),
                        kind="chunk",
                    )
                )
            self._fs.invalidate_cache(prefix)
            return found

        objects: list[RemoteObject] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for result in bounded_thread_map(
                executor,
                lambda item: list_zy(item[0], item[1]),
                sorted(selected_by_zy.items()),
                max_pending=workers * 2,
            ):
                objects.extend(result)
        return objects

    def get_file(self, key_or_relative: str, destination: Path) -> None:
        if self._local is not None:
            data = Path(key_or_relative).read_bytes()
            destination.write_bytes(data)
            return
        self._fs.get_file(key_or_relative, str(destination))


def load_array_spec(store: Store, array_key: str) -> tuple[ZarrArraySpec, int]:
    """Read .zarray metadata; returns the spec plus the item size in bytes."""

    raw = store.cat(f"{array_key}/.zarray")
    metadata = json.loads(raw.decode("utf-8"))
    spec = ZarrArraySpec.from_metadata(metadata)
    try:
        item_size = int(np.dtype(str(metadata["dtype"])).itemsize)
    except (KeyError, TypeError) as error:
        raise CarveError(f"invalid .zarray dtype: {error}") from error
    return spec, item_size


def chunks_for_sites(
    site_rows: list[dict[str, Any]],
    spec: ZarrArraySpec,
    *,
    fine_scan_id: str | None = None,
) -> frozenset[int]:
    """All chunk ids covered by the sites' fine-frame bounding boxes."""

    grid = spec.chunk_grid_zyx
    chunks = spec.chunks_zyx
    selected: set[int] = set()
    for row in site_rows:
        if fine_scan_id is not None and row.get("fine_scan_id") != fine_scan_id:
            continue
        lo = [int(item) for item in row["fine_bbox_lo_zyx"]]
        hi = [int(item) for item in row["fine_bbox_hi_zyx"]]
        chunk_lo = [
            max(0, lo[axis] // chunks[axis]) for axis in range(3)
        ]
        chunk_hi = [
            min(grid[axis], -(-hi[axis] // chunks[axis])) for axis in range(3)
        ]
        for cz in range(chunk_lo[0], chunk_hi[0]):
            for cy in range(chunk_lo[1], chunk_hi[1]):
                for cx in range(chunk_lo[2], chunk_hi[2]):
                    selected.add(spec.encode_chunk((cz, cy, cx)))
    return frozenset(selected)


def tube_chunks(
    fine_tifxyz_dirs: list[str | Path],
    spec: ZarrArraySpec,
    *,
    stride: int = 1,
    dilate_vox: int = 128,
) -> frozenset[int]:
    """Chunks near any traced fine surface, dilated by a voxel halo."""

    center: set[int] = set()
    for directory in fine_tifxyz_dirs:
        selection = select_tifxyz_chunks(directory, spec, stride=stride)
        center.update(selection.chunk_ids)
    return expand_chunk_ids(center, spec, halo_vox=dilate_vox)


@dataclass(frozen=True)
class CarvePlan:
    spec: ZarrArraySpec
    item_size: int
    footprint_chunks: int
    tube_chunks: int
    selected: frozenset[int]
    upper_bound_bytes: int

    def summary(self) -> dict[str, Any]:
        chunk_bytes = int(np.prod(self.spec.chunks_zyx)) * self.item_size
        return {
            "footprint_chunk_count": self.footprint_chunks,
            "tube_chunk_count": self.tube_chunks,
            "selected_chunk_count": len(self.selected),
            "chunk_bytes_each": chunk_bytes,
            "upper_bound_bytes": self.upper_bound_bytes,
            "upper_bound_human": human_bytes(self.upper_bound_bytes),
        }


def plan_carve(
    store: Store,
    site_rows: list[dict[str, Any]],
    *,
    options: CarveOptions,
    fine_scan_id: str | None = None,
    fine_tifxyz_dirs: list[str | Path] | None = None,
) -> CarvePlan:
    options.validate()
    spec, item_size = load_array_spec(store, options.array_key)
    footprint = chunks_for_sites(site_rows, spec, fine_scan_id=fine_scan_id)
    if not footprint:
        raise CarveError("site footprints select no chunks")
    tube: frozenset[int] | None = None
    if options.tube_intersect:
        if not fine_tifxyz_dirs:
            raise CarveError(
                "tube_intersect requires the record's fine TIFXYZ directories"
            )
        tube = tube_chunks(
            fine_tifxyz_dirs,
            spec,
            stride=options.tube_stride,
            dilate_vox=options.tube_dilate_vox,
        )
        selected = frozenset(footprint & tube)
    else:
        selected = footprint
    if not selected:
        raise CarveError("carve selection is empty after tube intersection")
    chunk_bytes = int(np.prod(spec.chunks_zyx)) * item_size
    upper_bound = len(selected) * chunk_bytes
    if upper_bound > options.max_bytes:
        raise CarveError(
            f"carve upper bound {human_bytes(upper_bound)} exceeds the "
            f"{human_bytes(options.max_bytes)} ceiling; reduce sites or raise "
            "max_bytes deliberately"
        )
    return CarvePlan(
        spec=spec,
        item_size=item_size,
        footprint_chunks=len(footprint),
        tube_chunks=(len(tube) if tube is not None else 0),
        selected=selected,
        upper_bound_bytes=upper_bound,
    )


def _download_object(
    store: Store, task: RemoteObject, output: Path, retries: int
) -> tuple[str, int, str]:
    destination = output / Path(task.relative_path)
    if destination.is_file() and destination.stat().st_size == task.size:
        return task.relative_path, 0, "skip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    last_error: Exception | str = "unknown error"
    for attempt in range(retries):
        try:
            store.get_file(task.key, temporary)
            actual = temporary.stat().st_size
            if actual != task.size:
                temporary.unlink(missing_ok=True)
                last_error = f"size mismatch {actual} != {task.size}"
                continue
            os.replace(temporary, destination)
            return task.relative_path, actual, "ok"
        except Exception as error:  # noqa: BLE001 - retry transient failures
            last_error = error
            time.sleep(0.5 * (attempt + 1))
    temporary.unlink(missing_ok=True)
    return (
        task.relative_path,
        0,
        f"FAIL {type(last_error).__name__}: {str(last_error)[:200]}",
    )


def execute_carve(
    store: Store,
    plan: CarvePlan,
    *,
    options: CarveOptions,
    output_path: str | Path,
    provenance: dict[str, Any] | None = None,
    progress_every: int = 500,
) -> dict[str, Any]:
    """Transfer the planned selection into a sparse local mirror.

    Resumable by object size; absent-on-source chunks are masked-empty and
    remain represented by the fill value. The *selected* chunk-id list (not
    just the existing objects) is persisted because selection is what
    defines downstream voxel coverage.
    """

    output = Path(output_path).expanduser().resolve()
    manifest_path = output / "crossres_sparse_mirror.json"
    if output.exists() and any(output.iterdir()) and not manifest_path.is_file():
        raise CarveError(
            f"{output} is non-empty and has no crossres sparse-mirror manifest"
        )
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("source_zarr") != store.source:
            raise CarveError(
                f"{output}: existing mirror describes a different source"
            )
    output.mkdir(parents=True, exist_ok=True)

    metadata_objects = store.list_metadata()
    zarray = store.info(f"{options.array_key}/.zarray")
    if zarray is not None:
        metadata_objects.append(
            RemoteObject(
                key=(
                    f"{store.source.removeprefix('s3://')}/"
                    f"{options.array_key}/.zarray"
                    if store.source.startswith("s3://")
                    else str(
                        Path(store.source) / options.array_key / ".zarray"
                    )
                ),
                relative_path=f"{options.array_key}/.zarray",
                size=int(zarray["size"]),
                kind="metadata",
            )
        )
    chunk_objects = store.find_selected_chunks(
        options.array_key, plan.selected, plan.spec, options.workers
    )
    objects = sorted(
        {task.relative_path: task for task in metadata_objects + chunk_objects}
        .values(),
        key=lambda task: task.relative_path,
    )
    total_bytes = sum(task.size for task in objects)

    selected_path = output / "carve_selected_chunks.json"
    write_json_atomic(
        selected_path,
        {
            "schema_version": 1,
            "array_key": options.array_key,
            "chunks_zyx": list(plan.spec.chunks_zyx),
            "chunk_grid_zyx": list(plan.spec.chunk_grid_zyx),
            "selected_chunk_ids": sorted(plan.selected),
        },
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crossres-fine-raw-carve",
        "state": "planned",
        "created_at_utc": utc_now(),
        "source_zarr": store.source,
        "array_key": options.array_key,
        "output": output.as_posix(),
        "options": asdict(options),
        "zarr": {
            "shape_zyx": list(plan.spec.shape_zyx),
            "chunks_zyx": list(plan.spec.chunks_zyx),
            "chunk_grid_zyx": list(plan.spec.chunk_grid_zyx),
            "dimension_separator": plan.spec.dimension_separator,
        },
        "selection": plan.summary(),
        "objects": {
            "count": len(objects),
            "chunk_count": len(chunk_objects),
            "bytes": total_bytes,
            "fill_value_chunk_count": len(plan.selected) - len(chunk_objects),
        },
        "provenance": provenance or {},
    }
    write_json_atomic(manifest_path, summary)

    downloaded = 0
    skipped = 0
    failures: list[str] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=options.workers) as executor:
        results = bounded_thread_map(
            executor,
            lambda task: _download_object(store, task, output, options.retries),
            objects,
            max_pending=options.workers * 4,
        )
        for index, (relative_path, size, status) in enumerate(results, 1):
            if status == "ok":
                downloaded += size
            elif status == "skip":
                skipped += 1
            else:
                failures.append(f"{relative_path}: {status}")
            if index % progress_every == 0 or index == len(objects):
                elapsed = max(time.perf_counter() - started, 1.0e-6)
                print(
                    f"  {index:,}/{len(objects):,} "
                    f"new={human_bytes(downloaded)} skip={skipped:,} "
                    f"fail={len(failures):,} "
                    f"rate={human_bytes(downloaded / elapsed)}/s",
                    flush=True,
                )

    summary["state"] = "failed" if failures else "complete"
    summary["completed_at_utc"] = utc_now()
    summary["transfer"] = {
        "downloaded_bytes_this_run": downloaded,
        "skipped_objects": skipped,
        "failures": failures,
        "seconds": time.perf_counter() - started,
    }
    write_json_atomic(manifest_path, summary)
    if failures:
        raise CarveError(
            f"{len(failures)} objects failed; first: {failures[0]}"
        )
    return summary


def load_carved_chunk_ids(
    mirror_path: str | Path,
) -> tuple[tuple[int, int, int], set[tuple[int, int, int]]]:
    """Read the carve's selected chunk set for coverage computation.

    Returns ``(chunk_shape_zyx, {(cz, cy, cx), ...})``. Selection -- not
    on-disk existence -- defines coverage, because masked-empty chunks are
    legitimately zero while unselected chunks are unknown.
    """

    path = Path(mirror_path).expanduser().resolve()
    selected_path = path / "carve_selected_chunks.json"
    value = json.loads(selected_path.read_text(encoding="utf-8"))
    chunks = tuple(int(item) for item in value["chunks_zyx"])
    grid = tuple(int(item) for item in value["chunk_grid_zyx"])
    grid_y, grid_x = grid[1], grid[2]
    ids: set[tuple[int, int, int]] = set()
    for chunk_id in value["selected_chunk_ids"]:
        z, remainder = divmod(int(chunk_id), grid_y * grid_x)
        y, x = divmod(remainder, grid_x)
        ids.add((z, y, x))
    return chunks, ids


def verify_carve(
    store: Store,
    mirror_path: str | Path,
    *,
    sample_fraction: float = 0.01,
    full: bool = False,
) -> dict[str, Any]:
    """MD5-verify a deterministic sample of carved objects against S3 ETags.

    Uncompressed single-part objects have ETag == MD5, so this is a
    conclusive integrity check -- the answer to the recorded
    "byte-size-complete Zarr objects can still be corrupt" incident. Local
    (test) stores fall back to byte comparison.
    """

    output = Path(mirror_path).expanduser().resolve()
    manifest = json.loads(
        (output / "crossres_sparse_mirror.json").read_text(encoding="utf-8")
    )
    array_key = str(manifest["array_key"])
    candidates = sorted(
        path
        for path in (output / array_key).rglob("*")
        if path.is_file() and not path.name.startswith(".")
    )
    if not candidates:
        raise CarveError(f"{output}: no carved objects to verify")
    if full:
        sample = candidates
    else:
        step = max(1, int(1.0 / max(sample_fraction, 1.0e-6)))
        sample = candidates[::step]
    mismatches: list[str] = []
    checked = 0
    for path in sample:
        relative = path.relative_to(output).as_posix()
        info = store.info(relative)
        if info is None:
            mismatches.append(f"{relative}: missing at source")
            continue
        if info["size"] != path.stat().st_size:
            mismatches.append(f"{relative}: size mismatch")
            continue
        digest = hashlib.md5()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        local_md5 = digest.hexdigest()
        etag = info.get("etag")
        if etag is not None and "-" not in etag:
            if etag != local_md5:
                mismatches.append(f"{relative}: md5 {local_md5} != etag {etag}")
        elif etag is None:
            remote = store.cat(relative)
            if hashlib.md5(remote).hexdigest() != local_md5:
                mismatches.append(f"{relative}: bytes differ from source")
        checked += 1
    report = {
        "schema_version": 1,
        "kind": "crossres-carve-verification",
        "verified_at": utc_now(),
        "objects_total": len(candidates),
        "objects_checked": checked,
        "mismatches": mismatches,
        "full": full,
    }
    write_json_atomic(output / "carve_verification.json", report)
    if mismatches:
        raise CarveError(
            f"carve verification failed on {len(mismatches)} objects; "
            f"first: {mismatches[0]}"
        )
    return report
