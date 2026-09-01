"""Compact official manual-label Zarr stores that materialize unknown space.

Three of the eight published 2 um label stores declare ``fill_value: 0`` while
using label 2 for "unknown". ``fill_value`` is exactly what a *missing* chunk
reads back as, so with 0 the publisher could not express "unknown" by omission
and had to write an explicit all-2 chunk for every unimaged cell: 91.7% of their
7.1M chunk files carry no information, at roughly 5.2 kB of NTFS footprint each
(a 4 kB cluster plus a 1 kB MFT record for a ~2.3 kB payload).

The transform is ``fill_value: 0 -> 2`` plus deletion of the all-unknown chunks.
It is semantically a no-op: an absent chunk in a ``fill_value: 2`` store reads
back exactly what a materialized all-2 chunk reads back. The four healthy stores
in the same release already use ``fill_value: 2`` and are sparse.

The logical chunk grid is deliberately untouched. ``ChunkSupport.present_ids``
is built *solely* from ``crossres_label_chunks.jsonl``, so leaving the grid and
that inventory byte-identical makes the sampled training corpus unchanged by
construction rather than by re-derivation.
"""

from __future__ import annotations

import array
import gc
import hashlib
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .grid_inference import _replace_directory_with_retry
from .io import array_metadata, dense_field_masks, open_volume
from .manual_labels import INDEX_ROWS, INDEX_STATE, load_catalog
from .resources import configure_cpu_budget
from .schema import DenseFieldSpec

COMPACTION_STATE = "crossres_label_compaction.json"
COMPACTION_SCHEMA = "crossres-label-store-compaction-v1"
TRANSFORM = "unknown-fill-prune-v1"
STAGING_SUFFIX = ".compacting"
SUPERSEDED_SUFFIX = ".superseded"
ARRAY_KEY = "0"
STAGES = ("planning", "verifying", "linking", "certifying", "swapping", "complete")


class CompactionError(ValueError):
    pass


# --------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
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


def chunk_grid(
    shape_zyx: tuple[int, int, int], chunks_zyx: tuple[int, int, int]
) -> tuple[int, int, int]:
    return tuple(  # type: ignore[return-value]
        (extent + chunk - 1) // chunk
        for extent, chunk in zip(shape_zyx, chunks_zyx, strict=True)
    )


def encode_chunk_id(
    coordinate_zyx: tuple[int, int, int], grid_zyx: tuple[int, int, int]
) -> int:
    """Row-major chunk id, identical to ``ChunkSupport._encode_static``."""

    z, y, x = coordinate_zyx
    return (z * grid_zyx[1] + y) * grid_zyx[2] + x


def decode_chunk_id(
    chunk_id: int, grid_zyx: tuple[int, int, int]
) -> tuple[int, int, int]:
    x = chunk_id % grid_zyx[2]
    yz = chunk_id // grid_zyx[2]
    return yz // grid_zyx[1], yz % grid_zyx[1], x


def chunk_relative_key(coordinate_zyx: tuple[int, int, int], separator: str) -> str:
    """Chunk key relative to the array root, honouring the V2 separator."""

    if separator not in {".", "/"}:
        raise CompactionError(f"unsupported dimension separator {separator!r}")
    return separator.join(str(value) for value in coordinate_zyx)


# --------------------------------------------------------------------------
# inventory scan -- the plan is the inventory, never a directory walk
# --------------------------------------------------------------------------


@dataclass
class InventoryPlan:
    """What the existing inventory says about every present chunk."""

    rows: int
    grid_zyx: tuple[int, int, int]
    unknown_ids: np.ndarray
    retained_ids: np.ndarray
    zero_byte_ids: np.ndarray
    positive_ids: np.ndarray
    unknown_bytes: int
    retained_bytes: int
    total_positive: int
    total_known: int
    unknown_size_histogram: dict[int, int] = field(default_factory=dict)

    @property
    def unknown(self) -> int:
        return int(self.unknown_ids.size)

    @property
    def retained(self) -> int:
        return int(self.retained_ids.size)

    @property
    def readable_retained_ids(self) -> np.ndarray:
        """Retained chunks that are decodable Zarr payloads.

        A zero-byte placeholder is an inventory record, not a chunk: Blosc
        correctly rejects an empty payload, which is why ChunkSupport drops
        them from the support set. They are carried through the compaction
        verbatim but must never be read.
        """

        if self.zero_byte_ids.size == 0:
            return self.retained_ids
        return np.setdiff1d(self.retained_ids, self.zero_byte_ids)


_COORD_MARKER = '"coordinate_zyx":['


def _int_field(line: str, key: str) -> int:
    marker = '"' + key + '":'
    start = line.find(marker)
    if start < 0:
        raise CompactionError("inventory row is missing " + repr(key))
    cursor = start + len(marker)
    end = cursor
    if end < len(line) and line[end] == "-":
        end += 1
    while end < len(line) and line[end].isdigit():
        end += 1
    return int(line[cursor:end])


def scan_inventory(rows_path: Path, grid_zyx: tuple[int, int, int]) -> InventoryPlan:
    """Stream the inventory once, classifying every chunk. No filesystem walk.

    A chunk is all-unknown exactly when ``known_voxels == 0``: the indexer writes
    ``known = decoded_voxels - ignored_voxels``, so zero known voxels means every
    voxel in the chunk carries an ignore label (2 = unknown).
    """

    unknown = array.array("q")
    retained = array.array("q")
    zero_byte = array.array("q")
    positive = array.array("q")
    unknown_bytes = 0
    retained_bytes = 0
    total_positive = 0
    total_known = 0
    rows = 0
    histogram: dict[int, int] = {}
    with rows_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if '"kind":"chunk"' not in line and '"kind": "chunk"' not in line:
                continue
            start = line.find(_COORD_MARKER)
            if start < 0:
                raise CompactionError(
                    f"{rows_path}:{line_number}: row is missing coordinate_zyx"
                )
            start += len(_COORD_MARKER)
            end = line.find("]", start)
            parts = line[start:end].split(",")
            if len(parts) != 3:
                raise CompactionError(
                    f"{rows_path}:{line_number}: coordinate_zyx is not 3-D"
                )
            coordinate = (int(parts[0]), int(parts[1]), int(parts[2]))
            if any(
                value < 0 or value >= extent
                for value, extent in zip(coordinate, grid_zyx, strict=True)
            ):
                raise CompactionError(
                    f"{rows_path}:{line_number}: out-of-grid coordinate {coordinate}"
                )
            chunk_id = encode_chunk_id(coordinate, grid_zyx)
            known = _int_field(line, "known_voxels")
            size = _int_field(line, "size")
            positive_voxels = _int_field(line, "positive_voxels")
            rows += 1
            total_positive += positive_voxels
            total_known += known
            if size == 0:
                # A zero-byte placeholder is an inventory record, not a readable
                # chunk. Never a prune candidate; carried through verbatim.
                zero_byte.append(chunk_id)
                retained.append(chunk_id)
                continue
            if known == 0:
                unknown.append(chunk_id)
                unknown_bytes += size
                histogram[size] = histogram.get(size, 0) + 1
            else:
                retained.append(chunk_id)
                retained_bytes += size
                if positive_voxels > 0:
                    positive.append(chunk_id)
    unknown_ids = np.sort(np.frombuffer(unknown, dtype=np.int64).copy())
    retained_ids = np.sort(np.frombuffer(retained, dtype=np.int64).copy())
    if unknown_ids.size and np.any(np.diff(unknown_ids) == 0):
        raise CompactionError(f"{rows_path}: duplicate all-unknown chunk rows")
    if retained_ids.size and np.any(np.diff(retained_ids) == 0):
        raise CompactionError(f"{rows_path}: duplicate retained chunk rows")
    return InventoryPlan(
        rows=rows,
        grid_zyx=grid_zyx,
        unknown_ids=unknown_ids,
        retained_ids=retained_ids,
        zero_byte_ids=np.sort(np.frombuffer(zero_byte, dtype=np.int64).copy()),
        positive_ids=np.sort(np.frombuffer(positive, dtype=np.int64).copy()),
        unknown_bytes=unknown_bytes,
        retained_bytes=retained_bytes,
        total_positive=total_positive,
        total_known=total_known,
        unknown_size_histogram=histogram,
    )


# --------------------------------------------------------------------------
# the canonical all-unknown blob
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalUnknown:
    payload: bytes
    sha256: str
    size: int


def canonical_unknown_chunk(
    array_root: Path,
    plan: InventoryPlan,
    *,
    separator: str,
    compressor_config: dict[str, Any] | None,
    dtype: np.dtype[Any],
    chunk_voxels: int,
    target_fill_value: int,
) -> CanonicalUnknown:
    """Establish and *prove* this store's all-unknown chunk payload.

    Established per store, never globally: two of the three targets share a blob
    but the third differs, and keying on a shared hash would apply the wrong
    payload. The proof is that it decodes to ``chunk_voxels`` values all equal to
    the new fill value. Size alone is not a discriminator -- a data-bearing chunk
    can compress to exactly the canonical length.
    """

    if plan.unknown_ids.size == 0:
        raise CompactionError(f"{array_root}: inventory lists no all-unknown chunks")
    if len(plan.unknown_size_histogram) != 1:
        sizes = sorted(plan.unknown_size_histogram)
        raise CompactionError(
            f"{array_root}: all-unknown chunks have {len(sizes)} distinct encoded "
            f"sizes {sizes[:8]}; a single canonical payload cannot be established"
        )
    coordinate = decode_chunk_id(int(plan.unknown_ids[0]), plan.grid_zyx)
    path = array_root / chunk_relative_key(coordinate, separator)
    payload = path.read_bytes()
    if compressor_config is None:
        decoded_bytes: Any = payload
    else:
        from numcodecs import get_codec

        decoded_bytes = get_codec(compressor_config).decode(payload)
    decoded = np.frombuffer(decoded_bytes, dtype=dtype)
    if decoded.size != chunk_voxels:
        raise CompactionError(
            f"{path}: decoded {decoded.size} voxels, expected {chunk_voxels}"
        )
    if not np.array_equiv(decoded, target_fill_value):
        raise CompactionError(
            f"{path}: the inventory calls this chunk all-unknown but it decodes to "
            f"{np.unique(decoded).tolist()[:8]}, not {target_fill_value}"
        )
    return CanonicalUnknown(
        payload=payload, sha256=hashlib.sha256(payload).hexdigest(), size=len(payload)
    )


# --------------------------------------------------------------------------
# C0 -- on-disk / inventory set reconciliation
# --------------------------------------------------------------------------


def reconcile_on_disk(array_root: Path, plan: InventoryPlan) -> dict[str, Any]:
    """The one directory walk we pay for: prove the inventory is not stale."""

    found = array.array("q")
    pending = [array_root]
    while pending:
        parent = pending.pop()
        with os.scandir(parent) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                    continue
                name = entry.name
                if name.startswith(".") or name.endswith(".part"):
                    continue
                relative = Path(entry.path).relative_to(array_root)
                parts = relative.parts
                raw = tuple(parts[0].split(".")) if len(parts) == 1 else parts
                if len(raw) != 3:
                    raise CompactionError(
                        f"{entry.path}: not a three-dimensional chunk key"
                    )
                try:
                    coordinate = (int(raw[0]), int(raw[1]), int(raw[2]))
                except ValueError as error:
                    raise CompactionError(
                        f"{entry.path}: invalid chunk coordinate"
                    ) from error
                if any(
                    value < 0 or value >= extent
                    for value, extent in zip(coordinate, plan.grid_zyx, strict=True)
                ):
                    # Without this the id encoding would alias an out-of-grid
                    # key onto a legitimate chunk and the set difference would
                    # silently come out empty.
                    raise CompactionError(
                        f"{entry.path}: chunk coordinate {coordinate} is outside "
                        f"the {plan.grid_zyx} chunk grid"
                    )
                found.append(encode_chunk_id(coordinate, plan.grid_zyx))
    on_disk = np.sort(np.frombuffer(found, dtype=np.int64).copy())
    inventory = np.sort(np.concatenate((plan.unknown_ids, plan.retained_ids)))
    missing = np.setdiff1d(inventory, on_disk)
    extra = np.setdiff1d(on_disk, inventory)
    if missing.size:
        sample = [decode_chunk_id(int(i), plan.grid_zyx) for i in missing[:5]]
        raise CompactionError(
            f"{array_root}: the inventory lists {missing.size} chunks that are not "
            f"on disk (stale inventory); e.g. {sample}"
        )
    if extra.size:
        sample = [decode_chunk_id(int(i), plan.grid_zyx) for i in extra[:5]]
        raise CompactionError(
            f"{array_root}: {extra.size} chunk files are absent from the inventory "
            f"(stale inventory); e.g. {sample}"
        )
    return {
        "on_disk_chunks": int(on_disk.size),
        "inventory_chunks": int(inventory.size),
    }


# --------------------------------------------------------------------------
# C1 -- byte-wise prune proof
# --------------------------------------------------------------------------


def _verify_block(
    arguments: tuple[np.ndarray, bool, Path, str, tuple[int, int, int], bytes, int],
) -> tuple[int, list[tuple[int, int, int]]]:
    ids, expect_canonical, array_root, separator, grid, payload, size = arguments
    violations: list[tuple[int, int, int]] = []
    for raw_id in ids.tolist():
        coordinate = decode_chunk_id(int(raw_id), grid)
        path = array_root / chunk_relative_key(coordinate, separator)
        if expect_canonical:
            if path.read_bytes() != payload:
                violations.append(coordinate)
            continue
        try:
            if path.stat().st_size != size:
                continue
        except OSError:
            violations.append(coordinate)
            continue
        # Same encoded length as the canonical unknown payload. Only the bytes
        # can settle it -- this is exactly the case a size-only gate deletes.
        if path.read_bytes() == payload:
            violations.append(coordinate)
    return len(ids), violations


def verify_prune_set(
    array_root: Path,
    plan: InventoryPlan,
    canonical: CanonicalUnknown,
    *,
    separator: str,
    workers: int,
    block: int = 4096,
    progress: bool = True,
) -> dict[str, Any]:
    """Require ``{known_voxels == 0} == {bytes == canonical}`` as sets.

    A chunk in the first set but not the second is one the inventory calls
    unknown while it holds different bytes. A chunk in the second but not the
    first is byte-identical to the unknown payload while the inventory credits
    it with known voxels. Either way the store and the inventory disagree and
    nothing may be deleted.
    """

    tasks: list[tuple[Any, ...]] = []
    for start in range(0, plan.unknown_ids.size, block):
        tasks.append(
            (
                plan.unknown_ids[start : start + block],
                True,
                array_root,
                separator,
                plan.grid_zyx,
                canonical.payload,
                canonical.size,
            )
        )
    unknown_blocks = len(tasks)
    for start in range(0, plan.retained_ids.size, block):
        tasks.append(
            (
                plan.retained_ids[start : start + block],
                False,
                array_root,
                separator,
                plan.grid_zyx,
                canonical.payload,
                canonical.size,
            )
        )
    not_canonical: list[tuple[int, int, int]] = []
    canonical_but_known: list[tuple[int, int, int]] = []
    checked = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (count, violations) in enumerate(executor.map(_verify_block, tasks)):
            checked += count
            if index < unknown_blocks:
                not_canonical.extend(violations)
            else:
                canonical_but_known.extend(violations)
            if progress and (index + 1) % 250 == 0:
                print(f"  verified {checked:,}/{plan.rows:,} chunks", flush=True)
    if not_canonical:
        raise CompactionError(
            f"{array_root}: {len(not_canonical)} chunks are recorded as all-unknown "
            f"but do not match the canonical payload; e.g. {not_canonical[:5]}"
        )
    if canonical_but_known:
        raise CompactionError(
            f"{array_root}: {len(canonical_but_known)} chunks are byte-identical to "
            f"the all-unknown payload but the inventory credits them with known "
            f"voxels; e.g. {canonical_but_known[:5]}"
        )
    return {
        "chunks_checked": checked,
        "prune_set": int(plan.unknown_ids.size),
        "retain_set": int(plan.retained_ids.size),
        "prune_not_canonical": 0,
        "canonical_outside_prune_set": 0,
    }


# --------------------------------------------------------------------------
# .zarray rewrite -- one value, textually, so everything else stays byte-exact
# --------------------------------------------------------------------------


_FILL_VALUE = re.compile(r'("fill_value"\s*:\s*)(-?\d+)')


def rewrite_fill_value(text: str, target: int) -> str:
    before = json.loads(text)
    matches = list(_FILL_VALUE.finditer(text))
    if len(matches) != 1:
        raise CompactionError(
            f"expected exactly one fill_value in .zarray, found {len(matches)}"
        )
    match = matches[0]
    result = text[: match.start(2)] + str(int(target)) + text[match.end(2) :]
    after = json.loads(result)
    if int(after["fill_value"]) != int(target):
        raise CompactionError("fill_value rewrite did not take effect")
    if {k: v for k, v in after.items() if k != "fill_value"} != {
        k: v for k, v in before.items() if k != "fill_value"
    }:
        raise CompactionError("fill_value rewrite altered another metadata field")
    return result


# --------------------------------------------------------------------------
# staging materialisation
# --------------------------------------------------------------------------


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            source_inode = source.stat().st_ino
            if source_inode and destination.stat().st_ino == source_inode:
                return "hardlink"
        except OSError:
            pass
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copyfile(source, destination)
        return "copy"


def _link_block(
    arguments: tuple[np.ndarray, Path, Path, str, tuple[int, int, int]],
) -> tuple[int, int]:
    ids, source_root, staging_root, separator, grid = arguments
    copies = 0
    for raw_id in ids.tolist():
        key = chunk_relative_key(decode_chunk_id(int(raw_id), grid), separator)
        if _link_or_copy(source_root / key, staging_root / key) == "copy":
            copies += 1
    return len(ids), copies


def materialize_staging(
    store: Path,
    staging: Path,
    plan: InventoryPlan,
    *,
    separator: str,
    target_fill_value: int,
    workers: int,
    block: int = 4096,
    progress: bool = True,
) -> dict[str, Any]:
    """Build the compacted store beside the original, out of place.

    Out of place is not stylistic: NTFS never shrinks a directory's index
    allocation, so pruning in place would pay the whole cost and keep the whole
    problem. Retained chunks are hardlinked -- same volume, zero bytes moved,
    and copy corruption stops being a failure class.
    """

    source_root = store / ARRAY_KEY
    staging_root = staging / ARRAY_KEY
    staging_root.mkdir(parents=True, exist_ok=True)

    for name in (".zgroup", ".zattrs", "meta.json", INDEX_ROWS, INDEX_STATE):
        candidate = store / name
        if candidate.is_file():
            _link_or_copy(candidate, staging / name)
    for name in (".zattrs",):
        candidate = source_root / name
        if candidate.is_file():
            _link_or_copy(candidate, staging_root / name)

    original = (source_root / ".zarray").read_text(encoding="utf-8")
    (staging_root / ".zarray").write_text(
        rewrite_fill_value(original, target_fill_value), encoding="utf-8", newline=""
    )

    tasks = [
        (plan.retained_ids[start : start + block], source_root, staging_root, separator, plan.grid_zyx)
        for start in range(0, plan.retained_ids.size, block)
    ]
    linked = 0
    copies = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (count, block_copies) in enumerate(
            executor.map(_link_block, tasks)
        ):
            linked += count
            copies += block_copies
            if progress and (index + 1) % 50 == 0:
                print(
                    f"  materialized {linked:,}/{plan.retained_ids.size:,} chunks",
                    flush=True,
                )
    return {"materialized_chunks": linked, "copied_chunks": copies}


def verify_retention(
    store: Path,
    staging: Path,
    plan: InventoryPlan,
    *,
    separator: str,
    sample: int = 4096,
    seed: int = 0,
) -> dict[str, Any]:
    """C2: every retained chunk is present in staging, and is the same bytes."""

    source_root = store / ARRAY_KEY
    staging_root = staging / ARRAY_KEY
    missing = 0
    for raw_id in plan.retained_ids.tolist():
        key = chunk_relative_key(decode_chunk_id(int(raw_id), plan.grid_zyx), separator)
        if not (staging_root / key).exists():
            missing += 1
            if missing <= 5:
                print(f"  MISSING retained chunk {key}", flush=True)
    if missing:
        raise CompactionError(
            f"{staging}: {missing} retained chunks were not materialized"
        )
    rng = np.random.default_rng(seed)
    picks = (
        plan.retained_ids
        if plan.retained_ids.size <= sample
        else rng.choice(plan.retained_ids, size=sample, replace=False)
    )
    hardlinks = 0
    for raw_id in np.asarray(picks).tolist():
        key = chunk_relative_key(decode_chunk_id(int(raw_id), plan.grid_zyx), separator)
        source = source_root / key
        target = staging_root / key
        source_inode = source.stat().st_ino
        if source_inode and target.stat().st_ino == source_inode:
            hardlinks += 1
        elif source.read_bytes() != target.read_bytes():
            raise CompactionError(f"{target}: materialized bytes differ from source")
    return {
        "retained_verified": int(plan.retained_ids.size),
        "sampled": int(np.asarray(picks).size),
        "hardlinked_in_sample": hardlinks,
    }


# --------------------------------------------------------------------------
# C4 -- semantic read-back A/B against the real decode path
# --------------------------------------------------------------------------


def _stratified_sample(
    plan: InventoryPlan,
    shape_zyx: tuple[int, int, int],
    chunks_zyx: tuple[int, int, int],
    *,
    samples: int,
    seed: int,
) -> list[tuple[str, tuple[int, int, int]]]:
    rng = np.random.default_rng(seed)
    per = max(1, samples // 4)
    picked: list[tuple[str, tuple[int, int, int]]] = []

    def take(label: str, ids: np.ndarray, count: int) -> None:
        if ids.size == 0:
            return
        chosen = (
            ids
            if ids.size <= count
            else rng.choice(ids, size=count, replace=False)
        )
        for raw in np.asarray(chosen).tolist():
            picked.append((label, decode_chunk_id(int(raw), plan.grid_zyx)))

    readable = plan.readable_retained_ids
    take("pruned-unknown", plan.unknown_ids, per)
    take("data-bearing", readable, per)
    take("positive", plan.positive_ids, per)
    # Chunks the array shape truncates: the partial-chunk edge of the grid.
    edge = []
    for axis in range(3):
        if shape_zyx[axis] % chunks_zyx[axis]:
            edge.append(plan.grid_zyx[axis] - 1)
        else:
            edge.append(-1)
    boundary = array.array("q")
    for raw in np.concatenate((plan.unknown_ids, readable)).tolist():
        coordinate = decode_chunk_id(int(raw), plan.grid_zyx)
        if any(coordinate[a] == edge[a] for a in range(3) if edge[a] >= 0):
            boundary.append(int(raw))
        if len(boundary) >= 4 * per:
            break
    take("boundary", np.frombuffer(boundary, dtype=np.int64).copy(), per)
    return picked


def readback_ab(
    store: Path,
    staging: Path,
    plan: InventoryPlan,
    *,
    positive_labels: tuple[int, ...],
    ignore_labels: tuple[int, ...],
    shape_zyx: tuple[int, int, int],
    chunks_zyx: tuple[int, int, int],
    samples: int,
    seed: int,
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Read both stores through the real decode path and require equality.

    ``dense_field_masks`` is what ``FineFieldWindowReader`` feeds, so comparing
    its output -- not just the raw voxels -- is what proves the training signal
    is unchanged.
    """

    before = open_volume(f"{store}::{ARRAY_KEY}")
    after = open_volume(f"{staging}::{ARRAY_KEY}")
    spec = DenseFieldSpec(
        volume=f"{store}::{ARRAY_KEY}",
        encoding="labels",
        positive_labels=positive_labels,
        ignore_labels=ignore_labels,
    )
    picks = _stratified_sample(
        plan, shape_zyx, chunks_zyx, samples=samples, seed=seed
    )
    counts: dict[str, int] = {}
    stream = log_path.open("w", encoding="utf-8", newline="\n") if log_path else None
    try:
        for label, coordinate in picks:
            slices = tuple(
                slice(
                    coordinate[axis] * chunks_zyx[axis],
                    min(
                        (coordinate[axis] + 1) * chunks_zyx[axis], shape_zyx[axis]
                    ),
                )
                for axis in range(3)
            )
            old = np.asarray(before[slices])
            new = np.asarray(after[slices])
            if not np.array_equal(old, new):
                raise CompactionError(
                    f"{staging}: chunk {coordinate} ({label}) reads back differently "
                    f"-- old uniques {np.unique(old).tolist()[:8]}, "
                    f"new uniques {np.unique(new).tolist()[:8]}"
                )
            old_positive, old_known = dense_field_masks(old, spec)
            new_positive, new_known = dense_field_masks(new, spec)
            if not np.array_equal(old_positive, new_positive) or not np.array_equal(
                old_known, new_known
            ):
                raise CompactionError(
                    f"{staging}: chunk {coordinate} ({label}) yields different "
                    f"positive/known masks"
                )
            counts[label] = counts.get(label, 0) + 1
            if stream is not None:
                stream.write(
                    json.dumps(
                        {
                            "stratum": label,
                            "coordinate_zyx": list(coordinate),
                            "voxels": int(old.size),
                            "positive": int(np.count_nonzero(old_positive)),
                            "known": int(np.count_nonzero(old_known)),
                            "equal": True,
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
    finally:
        if stream is not None:
            stream.close()
    # Windows will not rename a directory that still has open handles, and the
    # swap is the very next thing that happens. Drop the arrays explicitly
    # rather than leaving it to whenever the collector next runs.
    del before, after
    gc.collect()
    return {"samples": len(picks), "by_stratum": counts, "mismatches": 0}


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CompactionConfig:
    path: Path
    sha256: str
    catalog: Path
    labels_root: Path
    evidence_root: Path
    transform: str
    target_fill_value: int
    workers: int
    readback_samples: int
    readback_seed: int
    allow_variant_unknown_chunks: bool
    datasets: dict[str, dict[str, Any]]


def load_compaction_config(path: str | Path) -> CompactionConfig:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema") != COMPACTION_SCHEMA:
        raise CompactionError(f"{source}: invalid compaction config schema")
    if value.get("transform") != TRANSFORM:
        raise CompactionError(
            f"{source}: unsupported transform {value.get('transform')!r}"
        )
    datasets = value.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        # No "all datasets" mode: the transform is only sound for stores whose
        # ignore labels contain the target fill value, so each is named.
        raise CompactionError(f"{source}: config names no datasets")
    # Paths in the config are repo-root relative, anchored off the config's own
    # location so the tool does not depend on the caller's working directory.
    base = source.parents[2]
    if not (base / "crossres_pred").is_dir():
        raise CompactionError(
            f"{source}: expected the config under <repo>/crossres_pred/configs/"
        )
    by_id: dict[str, dict[str, Any]] = {}
    for item in datasets:
        dataset_id = str(item["dataset_id"])
        if dataset_id in by_id:
            raise CompactionError(f"{source}: duplicate dataset {dataset_id!r}")
        by_id[dataset_id] = item
    return CompactionConfig(
        path=source,
        sha256=_sha256_file(source),
        catalog=(base / str(value["catalog"])).resolve(),
        labels_root=(base / str(value["labels_root"])).resolve(),
        evidence_root=(base / str(value["evidence_root"])).resolve(),
        transform=str(value["transform"]),
        target_fill_value=int(value["target_fill_value"]),
        workers=int(value["workers"]),
        readback_samples=int(value["readback_samples"]),
        readback_seed=int(value["readback_seed"]),
        allow_variant_unknown_chunks=bool(
            value.get("allow_variant_unknown_chunks", False)
        ),
        datasets=by_id,
    )


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def _swap_directories(store: Path, staging: Path, superseded: Path) -> None:
    """Move the original aside and the compacted tree into its place.

    A directory rename loses to any process holding a handle inside it -- a
    virus scanner walking a freshly written tree of this size routinely does.
    The repo's existing retry policy is tuned for one inference directory and
    gives up after ~8 s, so this uses a longer budget. If the second rename
    fails the first is undone, leaving the original store exactly where it was.
    """

    gc.collect()
    _replace_directory_with_retry(
        store, superseded, attempts=120, initial_delay_seconds=0.1
    )
    try:
        _replace_directory_with_retry(
            staging, store, attempts=120, initial_delay_seconds=0.1
        )
    except OSError:
        _replace_directory_with_retry(
            superseded, store, attempts=120, initial_delay_seconds=0.1
        )
        raise


def _resume_swap(
    dataset_id: str,
    store: Path,
    staging: Path,
    superseded: Path,
    *,
    rows_path: Path,
) -> dict[str, Any] | None:
    """Finish a run that was certified but interrupted before the swap landed.

    Only the swap is skipped ahead to, and only when the signed certificate and
    every hash it recorded still hold. Re-deriving C0/C1/C4 here would take
    tens of minutes to re-prove exactly what the certificate already proves.
    """

    state_path = staging / COMPACTION_STATE
    if not state_path.is_file():
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("state") not in {"swapping", "certified"}:
        return None
    certificate_path = Path(str(state.get("certificate") or ""))
    if not certificate_path.is_file():
        raise CompactionError(f"{staging}: certified state names no readable certificate")
    if _sha256_file(certificate_path) != state.get("certificate_sha256"):
        raise CompactionError(f"{certificate_path}: certificate hash does not match")
    if _sha256_file(rows_path) != state.get("inventory_sha256"):
        raise CompactionError(f"{rows_path}: inventory changed since certification")
    staged_zarray = staging / ARRAY_KEY / ".zarray"
    if _sha256_file(staged_zarray) != (state.get("to") or {}).get("zarray_sha256"):
        raise CompactionError(f"{staged_zarray}: staged metadata changed since certification")
    print(
        f"[{dataset_id}] resuming a certified staging tree; swapping only",
        flush=True,
    )
    _swap_directories(store, staging, superseded)
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    certificate["state"] = "complete"
    certificate["superseded"] = str(superseded)
    _atomic_json(certificate_path, certificate)
    state["state"] = "complete"
    state["superseded"] = str(superseded)
    state["certificate_sha256"] = _sha256_file(certificate_path)
    _atomic_json(store / COMPACTION_STATE, state)
    print(
        f"[{dataset_id}] complete: {state['pruned_chunks']:,} chunks pruned, "
        f"{state['retained_chunks']:,} retained; old tree at {superseded.name}",
        flush=True,
    )
    return certificate


def compact_store(
    *,
    config: CompactionConfig,
    dataset_id: str,
    stop_before_linking: bool = False,
    swap: bool = True,
    max_cpu_threads: int = 16,
) -> dict[str, Any]:
    """Run the compaction for one named store, fail-closed at every gate."""

    if dataset_id not in config.datasets:
        raise CompactionError(f"{config.path}: dataset {dataset_id!r} is not named")
    entry = config.datasets[dataset_id]
    catalog = load_catalog(config.catalog)
    catalog_entry = next(
        (
            item
            for item in catalog["datasets"]
            if str(item["dataset_id"]) == dataset_id
        ),
        None,
    )
    if catalog_entry is None:
        raise CompactionError(f"{config.catalog}: no dataset {dataset_id!r}")

    ignore_labels = tuple(int(v) for v in catalog_entry.get("ignore_labels", []))
    positive_labels = tuple(int(v) for v in catalog_entry["positive_labels"])
    if config.target_fill_value not in ignore_labels:
        raise CompactionError(
            f"{dataset_id}: target fill value {config.target_fill_value} is not an "
            f"ignore label {list(ignore_labels)}; pruning would destroy known voxels"
        )

    configure_cpu_budget(max_cpu_threads, reserve_processes=min(config.workers, 8))
    store = (config.labels_root / str(catalog_entry["zarr"])).resolve()
    staging = store.with_name(store.name + STAGING_SUFFIX)
    superseded = store.with_name(store.name + SUPERSEDED_SUFFIX)
    evidence = config.evidence_root / dataset_id
    evidence.mkdir(parents=True, exist_ok=True)

    rows_path = store / INDEX_ROWS
    if swap and not stop_before_linking and staging.is_dir():
        resumed = _resume_swap(
            dataset_id, store, staging, superseded, rows_path=rows_path
        )
        if resumed is not None:
            return resumed

    metadata = array_metadata(f"{store}::{ARRAY_KEY}")
    if metadata.zarr_format != 2:
        raise CompactionError(f"{store}: expected a Zarr V2 store")
    separator = metadata.dimension_separator or "."
    zarray_path = store / ARRAY_KEY / ".zarray"
    raw_zarray = json.loads(zarray_path.read_text(encoding="utf-8"))
    if int(raw_zarray["fill_value"]) == config.target_fill_value:
        raise CompactionError(
            f"{store}: fill_value is already {config.target_fill_value}; this store "
            f"does not carry the materialized-unknown pathology"
        )
    grid = chunk_grid(metadata.shape_zyx, metadata.chunks_zyx)
    if not rows_path.is_file():
        raise CompactionError(f"{store}: no label inventory at {INDEX_ROWS}")
    index_state = json.loads((store / INDEX_STATE).read_text(encoding="utf-8"))
    if index_state.get("state") != "complete":
        raise CompactionError(f"{store}: label index is not complete")

    print(f"[{dataset_id}] scanning inventory {rows_path.name}", flush=True)
    plan = scan_inventory(rows_path, grid)
    expected = entry.get("expected") or {}
    observed = {
        "rows": plan.rows,
        "unknown": plan.unknown,
        "retained": plan.retained,
        "zero_byte": int(plan.zero_byte_ids.size),
    }
    for key, value in expected.items():
        if key == "unknown_chunk_bytes":
            continue
        if int(value) != observed[key]:
            raise CompactionError(
                f"{dataset_id}: expected {key}={value} but the inventory has "
                f"{observed[key]}; refusing to run on unverified counts"
            )
    if plan.total_positive != int(index_state["positive_voxels"]) or plan.total_known != int(
        index_state["known_voxels"]
    ):
        raise CompactionError(
            f"{store}: inventory aggregates disagree with {INDEX_STATE} "
            f"(positive {plan.total_positive} vs {index_state['positive_voxels']}, "
            f"known {plan.total_known} vs {index_state['known_voxels']})"
        )
    print(
        f"[{dataset_id}] rows={plan.rows:,} unknown={plan.unknown:,} "
        f"retained={plan.retained:,} zero_byte={plan.zero_byte_ids.size} "
        f"unknown_bytes={plan.unknown_bytes / 2**30:.2f} GiB "
        f"retained_bytes={plan.retained_bytes / 2**30:.2f} GiB",
        flush=True,
    )

    chunk_voxels = int(np.prod(metadata.chunks_zyx))
    canonical = canonical_unknown_chunk(
        store / ARRAY_KEY,
        plan,
        separator=separator,
        compressor_config=raw_zarray.get("compressor"),
        dtype=metadata.dtype,
        chunk_voxels=chunk_voxels,
        target_fill_value=config.target_fill_value,
    )
    expected_bytes = expected.get("unknown_chunk_bytes")
    if expected_bytes is not None and int(expected_bytes) != canonical.size:
        raise CompactionError(
            f"{dataset_id}: expected an unknown chunk of {expected_bytes} bytes, "
            f"found {canonical.size}"
        )
    print(
        f"[{dataset_id}] canonical unknown chunk {canonical.size} B "
        f"sha256={canonical.sha256[:16]} -> decodes to all {config.target_fill_value}",
        flush=True,
    )

    print(f"[{dataset_id}] C0 reconciling on-disk chunks against the inventory", flush=True)
    c0 = reconcile_on_disk(store / ARRAY_KEY, plan)
    print(f"[{dataset_id}] C0 ok: {c0['on_disk_chunks']:,} chunks", flush=True)

    print(f"[{dataset_id}] C1 byte-wise prune proof over {plan.rows:,} chunks", flush=True)
    c1 = verify_prune_set(
        store / ARRAY_KEY,
        plan,
        canonical,
        separator=separator,
        workers=config.workers,
    )
    print(f"[{dataset_id}] C1 ok: prune set proven byte-identical", flush=True)

    c5 = {
        "inventory_positive_voxels": plan.total_positive,
        "inventory_known_voxels": plan.total_known,
        "index_state_positive_voxels": int(index_state["positive_voxels"]),
        "index_state_known_voxels": int(index_state["known_voxels"]),
    }
    inventory_sha256 = _sha256_file(rows_path)
    zarray_sha256_before = _sha256_file(zarray_path)

    certificate: dict[str, Any] = {
        "schema": "crossres-label-compaction-certificate-v1",
        "dataset_id": dataset_id,
        "store": str(store),
        "transform": TRANSFORM,
        "config": str(config.path),
        "config_sha256": config.sha256,
        "catalog": str(config.catalog),
        "catalog_sha256": catalog["_sha256"],
        "array": {
            "shape_zyx": list(metadata.shape_zyx),
            "chunks_zyx": list(metadata.chunks_zyx),
            "grid_zyx": list(grid),
            "dtype": metadata.dtype.str,
            "dimension_separator": separator,
            "fill_value_before": int(raw_zarray["fill_value"]),
            "fill_value_after": config.target_fill_value,
        },
        "counts": {**observed, "unknown_bytes": plan.unknown_bytes,
                   "retained_bytes": plan.retained_bytes},
        "canonical_unknown_sha256": canonical.sha256,
        "canonical_unknown_bytes": canonical.size,
        "zarray_sha256_before": zarray_sha256_before,
        "inventory_sha256": inventory_sha256,
        "C0_set_reconciliation": c0,
        "C1_prune_proof": c1,
        "C5_aggregate_invariants": c5,
    }

    if stop_before_linking:
        certificate["state"] = "dry-run"
        _atomic_json(evidence / "dry_run.json", certificate)
        print(f"[{dataset_id}] dry run complete; nothing was modified", flush=True)
        return certificate

    if superseded.exists():
        raise CompactionError(
            f"{superseded}: a previous run left a superseded tree; resolve it first"
        )
    print(f"[{dataset_id}] materializing staging tree {staging.name}", flush=True)
    link_stats = materialize_staging(
        store,
        staging,
        plan,
        separator=separator,
        target_fill_value=config.target_fill_value,
        workers=config.workers,
    )
    c2 = verify_retention(store, staging, plan, separator=separator)
    print(
        f"[{dataset_id}] C2 ok: {c2['retained_verified']:,} retained chunks present "
        f"({link_stats['copied_chunks']} copied, rest hardlinked)",
        flush=True,
    )

    staged_rows_sha256 = _sha256_file(staging / INDEX_ROWS)
    if staged_rows_sha256 != inventory_sha256:
        raise CompactionError(f"{staging}: the inventory changed during staging")

    print(f"[{dataset_id}] C4 read-back A/B ({config.readback_samples} samples)", flush=True)
    c4 = readback_ab(
        store,
        staging,
        plan,
        positive_labels=positive_labels,
        ignore_labels=ignore_labels,
        shape_zyx=metadata.shape_zyx,
        chunks_zyx=metadata.chunks_zyx,
        samples=config.readback_samples,
        seed=config.readback_seed,
        log_path=evidence / "readback.jsonl",
    )
    print(f"[{dataset_id}] C4 ok: {c4['samples']} samples, 0 mismatches", flush=True)

    certificate["zarray_sha256_after"] = _sha256_file(staging / ARRAY_KEY / ".zarray")
    certificate["C2_retention"] = c2
    certificate["C3_inventory_invariance"] = {
        "inventory_sha256_before": inventory_sha256,
        "inventory_sha256_after": staged_rows_sha256,
        "identical": True,
    }
    certificate["C4_readback"] = c4
    certificate["materialization"] = link_stats
    certificate["state"] = "certified"
    certificate_path = evidence / "certificate.json"
    _atomic_json(certificate_path, certificate)

    _atomic_json(
        staging / COMPACTION_STATE,
        {
            "schema": COMPACTION_SCHEMA,
            "state": "swapping" if swap else "certified",
            "transform": TRANSFORM,
            "dataset_id": dataset_id,
            "from": {
                "fill_value": int(raw_zarray["fill_value"]),
                "zarray_sha256": zarray_sha256_before,
            },
            "to": {
                "fill_value": config.target_fill_value,
                "zarray_sha256": certificate["zarray_sha256_after"],
            },
            "pruned_chunks": plan.unknown,
            "retained_chunks": plan.retained,
            "canonical_unknown_sha256": canonical.sha256,
            "inventory_sha256": inventory_sha256,
            "certificate": str(certificate_path),
            "certificate_sha256": _sha256_file(certificate_path),
        },
    )

    if not swap:
        print(f"[{dataset_id}] certified; staging left at {staging}", flush=True)
        return certificate

    print(f"[{dataset_id}] swapping", flush=True)
    _swap_directories(store, staging, superseded)
    # Seal the certificate first, then record its final hash in the state. The
    # identity migration verifies that hash, so recording it before the last
    # write would leave every migration failing closed on a file we ourselves
    # changed afterwards.
    certificate["state"] = "complete"
    certificate["superseded"] = str(superseded)
    _atomic_json(certificate_path, certificate)
    state_path = store / COMPACTION_STATE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["state"] = "complete"
    state["superseded"] = str(superseded)
    state["certificate_sha256"] = _sha256_file(certificate_path)
    _atomic_json(state_path, state)
    print(
        f"[{dataset_id}] complete: {plan.unknown:,} chunks pruned, "
        f"{plan.retained:,} retained; old tree at {superseded.name}",
        flush=True,
    )
    return certificate


def compaction_record(store: str | Path) -> dict[str, Any] | None:
    """Return the completed compaction record for a store, if it has one."""

    path = Path(store).expanduser().resolve() / COMPACTION_STATE
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != COMPACTION_SCHEMA:
        raise CompactionError(f"{path}: unrecognized compaction record schema")
    return value
