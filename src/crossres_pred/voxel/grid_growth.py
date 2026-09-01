from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy import ndimage

from .grid_inference import (
    _replace_directory_with_retry,
    _required_context_origins,
    format_cube_id,
    parse_cube_id,
)
from .resources import configure_cpu_budget
from .ridge_growth import grow_probability_ridges

SCHEMA = "crossres-probability-ridge-growth-v1"
_STRUCTURE_6 = ndimage.generate_binary_structure(3, 1)
_CROSS_OFFSETS = (
    (0, 0, 0),
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)


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


def _read_cube(path: Path, chunk_size: int) -> np.ndarray:
    value = np.asarray(tifffile.imread(path))
    expected = (chunk_size,) * 3
    if value.shape != expected:
        raise ValueError(f"{path}: expected {expected}, got {value.shape}")
    return value


def _assemble_context(
    grid: Path,
    target_origin_zyx: tuple[int, int, int],
    *,
    subdirectory: str,
    chunk_size: int,
    halo: int,
    probability: bool,
) -> np.ndarray:
    lower = tuple(value - halo for value in target_origin_zyx)
    upper = tuple(value + chunk_size + halo for value in target_origin_zyx)
    shape = tuple(hi - lo for lo, hi in zip(lower, upper, strict=True))
    context = np.zeros(shape, dtype=np.float32 if probability else bool)
    for cube_origin in _required_context_origins(
        target_origin_zyx,
        chunk_size=chunk_size,
        halo=halo,
    ):
        cube_path = grid / subdirectory / f"{format_cube_id(cube_origin)}.tif"
        if not cube_path.is_file():
            continue
        cube = _read_cube(cube_path, chunk_size)
        overlap_lower = tuple(
            max(lo, origin) for lo, origin in zip(lower, cube_origin, strict=True)
        )
        overlap_upper = tuple(
            min(hi, origin + chunk_size)
            for hi, origin in zip(upper, cube_origin, strict=True)
        )
        source = tuple(
            slice(lo - origin, hi - origin)
            for lo, hi, origin in zip(
                overlap_lower,
                overlap_upper,
                cube_origin,
                strict=True,
            )
        )
        destination = tuple(
            slice(lo - base, hi - base)
            for lo, hi, base in zip(overlap_lower, overlap_upper, lower, strict=True)
        )
        if probability:
            context[destination] = cube[source].astype(np.float32, copy=False)
        else:
            context[destination] = cube[source] != 0
    return context


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _reconcile_grid_seams(
    *,
    seed_grid: Path,
    candidate_grid: Path,
    cube_ids: list[str],
    chunk_size: int,
) -> dict[str, Any]:
    """Remove rare cross-cube additions that create new first erosion interior."""

    target_set = set(cube_ids)
    removals: dict[str, set[tuple[int, int, int]]] = {}
    violations = 0
    for cube_id in cube_ids:
        origin = parse_cube_id(cube_id)
        seed = _assemble_context(
            seed_grid,
            origin,
            subdirectory="cubes_PRED",
            chunk_size=chunk_size,
            halo=1,
            probability=False,
        )
        candidate = _assemble_context(
            candidate_grid,
            origin,
            subdirectory="cubes_PRED",
            chunk_size=chunk_size,
            halo=1,
            probability=False,
        )
        probability = _assemble_context(
            seed_grid,
            origin,
            subdirectory="probability",
            chunk_size=chunk_size,
            halo=1,
            probability=True,
        )
        seed_interior = ndimage.binary_erosion(
            seed,
            structure=_STRUCTURE_6,
            border_value=0,
        )
        candidate_interior = ndimage.binary_erosion(
            candidate,
            structure=_STRUCTURE_6,
            border_value=0,
        )
        center = tuple(slice(1, 1 + chunk_size) for _ in range(3))
        local_violations = np.argwhere(
            candidate_interior[center] & ~seed_interior[center]
        )
        violations += int(local_violations.shape[0])
        added = candidate & ~seed
        for local in local_violations:
            context_position = tuple(int(value) + 1 for value in local)
            choices: list[
                tuple[float, tuple[int, int, int], str, tuple[int, int, int]]
            ] = []
            for offset in _CROSS_OFFSETS:
                position = tuple(
                    value + delta
                    for value, delta in zip(context_position, offset, strict=True)
                )
                if not bool(added[position]):
                    continue
                global_position = tuple(
                    base + value - 1
                    for base, value in zip(origin, position, strict=True)
                )
                cube_origin = tuple(
                    (value // chunk_size) * chunk_size for value in global_position
                )
                removal_cube_id = format_cube_id(cube_origin)
                if removal_cube_id not in target_set:
                    continue
                removal_local = tuple(
                    value - base
                    for value, base in zip(
                        global_position,
                        cube_origin,
                        strict=True,
                    )
                )
                choices.append(
                    (
                        float(probability[position]),
                        global_position,
                        removal_cube_id,
                        removal_local,
                    )
                )
            if not choices:
                raise RuntimeError(
                    f"new seam interior at {cube_id}:{tuple(local)} has no added neighbour"
                )
            _, _, removal_cube_id, removal_local = min(choices)
            removals.setdefault(removal_cube_id, set()).add(removal_local)
    for cube_id, local_positions in sorted(removals.items()):
        path = candidate_grid / "cubes_PRED" / f"{cube_id}.tif"
        mask = _read_cube(path, chunk_size).copy()
        for position in sorted(local_positions):
            mask[position] = 0
        tifffile.imwrite(path, mask)

    remaining = 0
    for cube_id in cube_ids:
        origin = parse_cube_id(cube_id)
        seed = _assemble_context(
            seed_grid,
            origin,
            subdirectory="cubes_PRED",
            chunk_size=chunk_size,
            halo=1,
            probability=False,
        )
        candidate = _assemble_context(
            candidate_grid,
            origin,
            subdirectory="cubes_PRED",
            chunk_size=chunk_size,
            halo=1,
            probability=False,
        )
        center = tuple(slice(1, 1 + chunk_size) for _ in range(3))
        seed_interior = ndimage.binary_erosion(
            seed,
            structure=_STRUCTURE_6,
            border_value=0,
        )
        candidate_interior = ndimage.binary_erosion(
            candidate,
            structure=_STRUCTURE_6,
            border_value=0,
        )
        remaining += int(
            np.count_nonzero(candidate_interior[center] & ~seed_interior[center])
        )
    if remaining:
        raise RuntimeError(
            f"seam reconciliation left {remaining} new first-interior voxels"
        )
    removed_positions = [
        {
            "cube_id": cube_id,
            "local_zyx": list(position),
        }
        for cube_id, positions in sorted(removals.items())
        for position in sorted(positions)
    ]
    return {
        "initial_new_first_interior_voxels": violations,
        "removed_added_voxels": len(removed_positions),
        "remaining_new_first_interior_voxels": remaining,
        "removed_positions": removed_positions,
    }


def grow_probability_grid(
    *,
    input_grid: str | Path,
    output_path: str | Path,
    support_threshold: float,
    max_steps: int,
    halo: int | None = None,
    workers: int = 4,
    max_cpu_threads: int = 16,
    target_cube_ids: Sequence[str] | None = None,
) -> Path:
    """Grow a thresholded inference grid without new thickness or fusions."""

    if not 0.0 <= support_threshold <= 1.0:
        raise ValueError("support_threshold must be in [0, 1]")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not 1 <= workers <= 16 or workers > max_cpu_threads:
        raise ValueError("workers must be in [1, max_cpu_threads]")
    if halo is None:
        halo = max_steps
    if halo < max_steps:
        raise ValueError("halo must be at least max_steps")
    configure_cpu_budget(max_cpu_threads, reserve_processes=workers - 1)

    input_root = Path(input_grid).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError(f"ridge-growth output already exists: {output}")
    input_provenance_path = input_root / "provenance.json"
    input_provenance = _read_object(input_provenance_path)
    source_manifest_path = input_root / "source_manifest.json"
    source_manifest = _read_object(source_manifest_path)
    chunk_size = int(source_manifest["chunk_size"])
    if not 0 <= halo < chunk_size:
        raise ValueError("halo must be non-negative and smaller than chunk_size")
    input_present_path = input_root / "cubes_PRED" / "present.json"
    input_ids = _read_ids(input_present_path)
    if input_provenance.get("target_cube_ids") != input_ids:
        raise ValueError("input present.json and provenance target IDs differ")
    if target_cube_ids is None:
        target_ids = input_ids
    else:
        target_ids = sorted({str(item) for item in target_cube_ids})
        if not target_ids:
            raise ValueError("target_cube_ids must not be empty")
        missing = sorted(set(target_ids) - set(input_ids))
        if missing:
            raise ValueError(f"target cubes are absent from input grid: {missing}")
    input_options = input_provenance.get("options")
    if not isinstance(input_options, dict):
        raise TypeError("input provenance has no inference options")
    seed_threshold = float(input_options["threshold"])
    if support_threshold >= seed_threshold:
        raise ValueError("support_threshold must be below the seed threshold")

    temporary = output.with_name(output.name + f".partial-{os.getpid()}")
    if temporary.exists():
        raise ValueError(f"stale ridge-growth temporary exists: {temporary}")
    prediction_root = temporary / "cubes_PRED"
    probability_root = temporary / "probability"
    prediction_root.mkdir(parents=True)
    probability_root.mkdir()

    def work(cube_id: str) -> dict[str, Any]:
        origin = parse_cube_id(cube_id)
        probability_context = _assemble_context(
            input_root,
            origin,
            subdirectory="probability",
            chunk_size=chunk_size,
            halo=halo,
            probability=True,
        )
        seed_context = _assemble_context(
            input_root,
            origin,
            subdirectory="cubes_PRED",
            chunk_size=chunk_size,
            halo=halo,
            probability=False,
        )
        result = grow_probability_ridges(
            probability_context,
            seed_context,
            support_threshold=support_threshold,
            max_steps=max_steps,
        )
        center = tuple(slice(halo, halo + chunk_size) for _ in range(3))
        central_seed = seed_context[center]
        central_mask = result.mask[center]
        if bool(np.any(central_seed & ~central_mask)):
            raise RuntimeError(f"ridge growth removed seed voxels from {cube_id}")
        output_mask_path = prediction_root / f"{cube_id}.tif"
        tifffile.imwrite(output_mask_path, central_mask.astype(np.uint8) * 255)
        input_probability_path = input_root / "probability" / f"{cube_id}.tif"
        output_probability_path = probability_root / f"{cube_id}.tif"
        _link_or_copy(input_probability_path, output_probability_path)
        context_stats = result.to_dict()
        return {
            "cube_id": cube_id,
            "seed_positive": int(np.count_nonzero(central_seed)),
            "final_positive": int(np.count_nonzero(central_mask)),
            "added_positive": int(np.count_nonzero(central_mask & ~central_seed)),
            "output_sha256": _sha256(output_mask_path),
            "probability_sha256": _sha256(output_probability_path),
            "context": context_stats,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = []
        for index, row in enumerate(executor.map(work, target_ids), 1):
            rows.append(row)
            print(
                f"ridge growth {index}/{len(target_ids)}: {row['cube_id']} "
                f"(+{row['added_positive']:,})",
                flush=True,
            )
    reconciliation = _reconcile_grid_seams(
        seed_grid=input_root,
        candidate_grid=temporary,
        cube_ids=target_ids,
        chunk_size=chunk_size,
    )
    for row in rows:
        cube_id = str(row["cube_id"])
        seed_path = input_root / "cubes_PRED" / f"{cube_id}.tif"
        output_mask_path = prediction_root / f"{cube_id}.tif"
        seed_mask = _read_cube(seed_path, chunk_size) != 0
        final_mask = _read_cube(output_mask_path, chunk_size) != 0
        row["seed_positive"] = int(np.count_nonzero(seed_mask))
        row["final_positive"] = int(np.count_nonzero(final_mask))
        row["added_positive"] = int(np.count_nonzero(final_mask & ~seed_mask))
        row["output_sha256"] = _sha256(output_mask_path)
    seed_positive = sum(int(row["seed_positive"]) for row in rows)
    final_positive = sum(int(row["final_positive"]) for row in rows)
    aggregate = {
        "cube_count": len(rows),
        "seed_positive": seed_positive,
        "final_positive": final_positive,
        "added_positive": final_positive - seed_positive,
        "foreground_growth_fraction": (final_positive - seed_positive)
        / max(1, seed_positive),
        "cubes_with_growth": sum(int(row["added_positive"] > 0) for row in rows),
        "context_component_conflict_rejections": sum(
            int(row["context"]["component_conflict_rejections"]) for row in rows
        ),
        "context_thickness_rejections": sum(
            int(row["context"]["thickness_rejections"]) for row in rows
        ),
        "seam_reconciliation_removed_voxels": reconciliation[
            "removed_added_voxels"
        ],
    }
    growth_report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input_grid": str(input_root),
        "input_provenance_sha256": _sha256(input_provenance_path),
        "options": {
            "seed_threshold": seed_threshold,
            "support_threshold": support_threshold,
            "max_steps": max_steps,
            "halo": halo,
            "workers": workers,
            "max_cpu_threads": max_cpu_threads,
        },
        "aggregate": aggregate,
        "seam_reconciliation": reconciliation,
        "cubes": rows,
    }
    (temporary / "growth.json").write_text(
        json.dumps(growth_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (prediction_root / "present.json").write_text(
        json.dumps(target_ids, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(source_manifest_path, temporary / "source_manifest.json")
    provenance = dict(input_provenance)
    provenance.update(
        schema=SCHEMA,
        kind=SCHEMA,
        created_at_utc=datetime.now(UTC).isoformat(),
        parent_grid=str(input_root),
        parent_provenance_sha256=_sha256(input_provenance_path),
        growth_options=growth_report["options"],
        growth_aggregate=aggregate,
        seam_reconciliation=reconciliation,
        target_cube_ids=target_ids,
        research_only=True,
        deployment_ready=False,
    )
    (temporary / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _replace_directory_with_retry(temporary, output)
    return output
