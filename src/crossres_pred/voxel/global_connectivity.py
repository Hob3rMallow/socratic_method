from __future__ import annotations

import hashlib
import html
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy import ndimage

from .grid_growth import _assemble_context
from .grid_inference import parse_cube_id
from .resources import configure_cpu_budget
from .scrollfiesta_metrics import GARBAGE_ERODE_MAXPASS, GARBAGE_ERODE_R

SCHEMA = "crossres-global-grid-connectivity-audit-v1"
_STRUCTURE_6 = ndimage.generate_binary_structure(3, 1)


@dataclass(frozen=True)
class _LocalComponents:
    count: int
    sizes: np.ndarray
    low_faces: tuple[np.ndarray, np.ndarray, np.ndarray]
    high_faces: tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class _ComponentIndex:
    cube_ids: tuple[str, ...]
    offsets: dict[str, int]
    counts: dict[str, int]
    roots: np.ndarray
    sizes: np.ndarray

    @property
    def component_count(self) -> int:
        return int(np.count_nonzero(self.sizes))

    @property
    def foreground(self) -> int:
        return int(self.sizes.sum())

    def local_roots(self, cube_id: str, labels: np.ndarray) -> np.ndarray:
        count = self.counts[cube_id]
        if int(labels.max(initial=0)) != count:
            raise RuntimeError(f"component relabel count changed for {cube_id}")
        lookup = np.full(count + 1, -1, dtype=np.int64)
        if count:
            offset = self.offsets[cube_id]
            lookup[1:] = self.roots[offset : offset + count]
        return lookup[labels]


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.uint8)

    def find(self, value: int) -> int:
        parent = self.parent
        root = value
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[value]) != value:
            next_value = int(parent[value])
            parent[value] = root
            value = next_value
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        rank = self.rank
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _read_ids(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: expected a non-empty array")
    ids = [str(item) for item in value]
    if len(ids) != len(set(ids)) or any(not item for item in ids):
        raise ValueError(f"{path}: cube IDs must be unique and non-empty")
    return sorted(ids)


def _read_mask(root: Path, cube_id: str, chunk_size: int) -> np.ndarray:
    path = root / "cubes_PRED" / f"{cube_id}.tif"
    value = np.asarray(tifffile.imread(path))
    if value.shape != (chunk_size,) * 3:
        raise ValueError(f"{path}: unexpected cube shape {value.shape}")
    return value != 0


def _local_components(mask: np.ndarray) -> tuple[np.ndarray, _LocalComponents]:
    labels, count = ndimage.label(mask, structure=_STRUCTURE_6)
    labels = labels.astype(np.int32, copy=False)
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)[1:].astype(
        np.int64,
        copy=False,
    )
    low = (
        labels[0].copy(),
        labels[:, 0].copy(),
        labels[:, :, 0].copy(),
    )
    high = (
        labels[-1].copy(),
        labels[:, -1].copy(),
        labels[:, :, -1].copy(),
    )
    return labels, _LocalComponents(int(count), sizes, low, high)


def _build_component_index(
    root: Path,
    cube_ids: list[str],
    *,
    chunk_size: int,
    workers: int,
) -> _ComponentIndex:
    def work(cube_id: str) -> _LocalComponents:
        _, local = _local_components(_read_mask(root, cube_id, chunk_size))
        return local

    with ThreadPoolExecutor(max_workers=workers) as executor:
        local_rows = list(executor.map(work, cube_ids))
    offsets: dict[str, int] = {}
    counts: dict[str, int] = {}
    component_sizes: list[np.ndarray] = []
    total = 0
    for cube_id, local in zip(cube_ids, local_rows, strict=True):
        offsets[cube_id] = total
        counts[cube_id] = local.count
        total += local.count
        component_sizes.append(local.sizes)
    union_find = _UnionFind(total)
    origin_to_index = {parse_cube_id(cube_id): index for index, cube_id in enumerate(cube_ids)}
    for index, cube_id in enumerate(cube_ids):
        origin = parse_cube_id(cube_id)
        local = local_rows[index]
        current_offset = offsets[cube_id]
        for axis in range(3):
            neighbour_origin = list(origin)
            neighbour_origin[axis] -= chunk_size
            neighbour_index = origin_to_index.get(tuple(neighbour_origin))
            if neighbour_index is None:
                continue
            neighbour_id = cube_ids[neighbour_index]
            neighbour = local_rows[neighbour_index]
            current_face = local.low_faces[axis]
            neighbour_face = neighbour.high_faces[axis]
            overlap = (current_face > 0) & (neighbour_face > 0)
            if not bool(np.any(overlap)):
                continue
            left = current_offset + current_face[overlap].astype(np.int64) - 1
            right = (
                offsets[neighbour_id]
                + neighbour_face[overlap].astype(np.int64)
                - 1
            )
            pair_base = max(total, 1)
            encoded = np.unique(left * pair_base + right)
            for value in encoded:
                union_find.union(int(value // pair_base), int(value % pair_base))
    roots = np.fromiter(
        (union_find.find(index) for index in range(total)),
        dtype=np.int64,
        count=total,
    )
    local_sizes = (
        np.concatenate(component_sizes)
        if component_sizes
        else np.zeros(0, dtype=np.int64)
    )
    global_sizes = np.zeros(total, dtype=np.int64)
    np.add.at(global_sizes, roots, local_sizes)
    return _ComponentIndex(
        cube_ids=tuple(cube_ids),
        offsets=offsets,
        counts=counts,
        roots=roots,
        sizes=global_sizes,
    )


def _global_bridge_metrics(
    reference: Path,
    candidate: Path,
    cube_ids: list[str],
    *,
    chunk_size: int,
    workers: int,
    minimum_reference_component_voxels: int,
) -> dict[str, int | float]:
    reference_index = _build_component_index(
        reference,
        cube_ids,
        chunk_size=chunk_size,
        workers=workers,
    )
    candidate_index = _build_component_index(
        candidate,
        cube_ids,
        chunk_size=chunk_size,
        workers=workers,
    )
    eligible_reference = (
        reference_index.sizes >= minimum_reference_component_voxels
    )
    reference_root_base = max(reference_index.roots.size, 1)

    def work(cube_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reference_mask = _read_mask(reference, cube_id, chunk_size)
        candidate_mask = _read_mask(candidate, cube_id, chunk_size)
        reference_labels, _ = ndimage.label(reference_mask, structure=_STRUCTURE_6)
        candidate_labels, _ = ndimage.label(candidate_mask, structure=_STRUCTURE_6)
        reference_roots = reference_index.local_roots(cube_id, reference_labels)
        candidate_roots = candidate_index.local_roots(cube_id, candidate_labels)
        overlap = reference_mask & candidate_mask
        keep = overlap & eligible_reference[np.maximum(reference_roots, 0)]
        if bool(np.any(keep)):
            pair_codes = np.unique(
                candidate_roots[keep] * reference_root_base + reference_roots[keep]
            )
        else:
            pair_codes = np.zeros(0, dtype=np.int64)
        candidate_only = candidate_mask & ~reference_mask
        if bool(np.any(candidate_only)):
            roots, counts = np.unique(
                candidate_roots[candidate_only],
                return_counts=True,
            )
        else:
            roots = np.zeros(0, dtype=np.int64)
            counts = np.zeros(0, dtype=np.int64)
        return pair_codes, roots, counts

    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(work, cube_ids))
    pair_arrays = [row[0] for row in rows if row[0].size]
    pairs = (
        np.unique(np.concatenate(pair_arrays))
        if pair_arrays
        else np.zeros(0, dtype=np.int64)
    )
    candidate_only_counts = np.zeros(candidate_index.roots.size, dtype=np.int64)
    for _, roots, counts in rows:
        candidate_only_counts[roots] += counts
    if pairs.size:
        pair_candidate_roots = pairs // reference_root_base
        references_per_candidate = np.bincount(
            pair_candidate_roots,
            minlength=candidate_index.roots.size,
        )
    else:
        references_per_candidate = np.zeros(
            candidate_index.roots.size,
            dtype=np.int64,
        )
    merging_roots = np.flatnonzero(references_per_candidate >= 2)
    merge_excess = int(
        np.maximum(references_per_candidate[merging_roots] - 1, 0).sum()
    )
    bridge_voxels = int(candidate_only_counts[merging_roots].sum())
    return {
        "reference_components": reference_index.component_count,
        "eligible_reference_components": int(np.count_nonzero(eligible_reference)),
        "candidate_components": candidate_index.component_count,
        "merging_candidate_components": int(merging_roots.size),
        "merged_reference_component_excess": merge_excess,
        "candidate_only_bridge_voxels": bridge_voxels,
        "candidate_only_bridge_fraction": bridge_voxels
        / max(1, candidate_index.foreground),
        "reference_positive": reference_index.foreground,
        "candidate_positive": candidate_index.foreground,
        "foreground_ratio_vs_reference": candidate_index.foreground
        / max(1, reference_index.foreground),
        "minimum_reference_component_voxels": minimum_reference_component_voxels,
    }


def _global_thickness_metrics(
    reference: Path,
    candidate: Path,
    cube_ids: list[str],
    *,
    chunk_size: int,
    workers: int,
) -> dict[str, int | float]:
    halo = GARBAGE_ERODE_MAXPASS + 1
    center = tuple(slice(halo, halo + chunk_size) for _ in range(3))

    def work(cube_id: str) -> dict[str, int]:
        origin = parse_cube_id(cube_id)
        reference_context = _assemble_context(
            reference,
            origin,
            subdirectory="cubes_PRED",
            chunk_size=chunk_size,
            halo=halo,
            probability=False,
        )
        candidate_context = _assemble_context(
            candidate,
            origin,
            subdirectory="cubes_PRED",
            chunk_size=chunk_size,
            halo=halo,
            probability=False,
        )
        reference_distance = ndimage.distance_transform_cdt(
            reference_context,
            metric="taxicab",
        )[center]
        candidate_distance = ndimage.distance_transform_cdt(
            candidate_context,
            metric="taxicab",
        )[center]
        return {
            "reference_interior": int(
                np.count_nonzero(reference_distance > GARBAGE_ERODE_R)
            ),
            "candidate_interior": int(
                np.count_nonzero(candidate_distance > GARBAGE_ERODE_R)
            ),
            "new_first_erosion_interior": int(
                np.count_nonzero(
                    (candidate_distance > 1) & ~(reference_distance > 1)
                )
            ),
            "new_scrollfiesta_interior": int(
                np.count_nonzero(
                    (candidate_distance > GARBAGE_ERODE_R)
                    & ~(reference_distance > GARBAGE_ERODE_R)
                )
            ),
            "reference_max_thickness": int(reference_distance.max(initial=0)),
            "candidate_max_thickness": int(candidate_distance.max(initial=0)),
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(work, cube_ids))
    reference_interior = sum(row["reference_interior"] for row in rows)
    candidate_interior = sum(row["candidate_interior"] for row in rows)
    return {
        "reference_interior_voxels": reference_interior,
        "candidate_interior_voxels": candidate_interior,
        "interior_voxel_delta": candidate_interior - reference_interior,
        "new_first_erosion_interior_voxels": sum(
            row["new_first_erosion_interior"] for row in rows
        ),
        "new_scrollfiesta_interior_voxels": sum(
            row["new_scrollfiesta_interior"] for row in rows
        ),
        "reference_max_thickness": max(
            row["reference_max_thickness"] for row in rows
        ),
        "candidate_max_thickness": max(
            row["candidate_max_thickness"] for row in rows
        ),
        "maximum_per_cube_thickness_excess": max(
            row["candidate_max_thickness"] - row["reference_max_thickness"]
            for row in rows
        ),
        "context_halo": halo,
    }


def _render_html(report: dict[str, Any]) -> str:
    passed = report["passed"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Global grid connectivity audit</title><style>
:root{{color-scheme:dark;font-family:system-ui,sans-serif}}body{{margin:2rem;background:#07101c;color:#e5e7eb}}
section{{border:1px solid #334155;border-radius:12px;padding:1rem;margin:1rem 0}}pre{{overflow:auto}}
.state{{color:{'#86efac' if passed else '#fca5a5'};font-size:1.4rem}}code{{color:#f0abfc}}</style></head><body>
<h1>Stitched-grid connectivity and thickness audit</h1>
<p class="state"><strong>{'PASS' if passed else 'FLAGGED'}</strong></p>
<p>Unlike a cube-local alarm, this audit unions component labels across every shared
cube face and measures thickness with neighbour halos.</p>
<section><h2>Global components</h2><pre>{html.escape(json.dumps(report['components'], indent=2))}</pre></section>
<section><h2>Halo-correct thickness</h2><pre>{html.escape(json.dumps(report['thickness'], indent=2))}</pre></section>
<p>Candidate <code>{html.escape(str(report['candidate_grid']))}</code><br>
Reference <code>{html.escape(str(report['reference_grid']))}</code><br>
Machine-readable report: <a href="report.json">report.json</a></p></body></html>"""


def audit_global_grid_connectivity(
    *,
    reference_path: str | Path,
    candidate_path: str | Path,
    output_path: str | Path,
    minimum_reference_component_voxels: int = 500,
    workers: int = 4,
    max_cpu_threads: int = 16,
    reference_label: str = "reference",
) -> Path:
    if minimum_reference_component_voxels <= 0:
        raise ValueError("minimum reference component voxels must be positive")
    if not 1 <= workers <= max_cpu_threads <= 16:
        raise ValueError("workers/max_cpu_threads must satisfy 1 <= workers <= threads <= 16")
    configure_cpu_budget(max_cpu_threads, reserve_processes=workers - 1)
    reference = Path(reference_path).expanduser().resolve()
    candidate = Path(candidate_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError(f"global connectivity output already exists: {output}")
    reference_present = reference / "cubes_PRED" / "present.json"
    candidate_present = candidate / "cubes_PRED" / "present.json"
    reference_ids = _read_ids(reference_present)
    candidate_ids = _read_ids(candidate_present)
    if reference_ids != candidate_ids:
        raise ValueError("reference and candidate cube inventories differ")
    manifest_path = reference / "source_manifest.json"
    manifest = _read_object(manifest_path)
    chunk_size = int(manifest["chunk_size"])
    components = _global_bridge_metrics(
        reference,
        candidate,
        candidate_ids,
        chunk_size=chunk_size,
        workers=workers,
        minimum_reference_component_voxels=minimum_reference_component_voxels,
    )
    thickness = _global_thickness_metrics(
        reference,
        candidate,
        candidate_ids,
        chunk_size=chunk_size,
        workers=workers,
    )
    passed = (
        int(components["merging_candidate_components"]) == 0
        and int(components["candidate_only_bridge_voxels"]) == 0
        and int(thickness["new_first_erosion_interior_voxels"]) == 0
        and int(thickness["new_scrollfiesta_interior_voxels"]) == 0
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "reference_label": reference_label,
        "reference_grid": str(reference),
        "candidate_grid": str(candidate),
        "cube_count": len(candidate_ids),
        "chunk_size": chunk_size,
        "reference_present_sha256": _sha256(reference_present),
        "candidate_present_sha256": _sha256(candidate_present),
        "components": components,
        "thickness": thickness,
        "passed": passed,
    }
    output.mkdir(parents=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "index.html").write_text(
        _render_html(report),
        encoding="utf-8",
        newline="\n",
    )
    return output / "index.html"
