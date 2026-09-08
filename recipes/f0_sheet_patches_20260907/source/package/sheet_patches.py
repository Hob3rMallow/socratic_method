"""Checked opt-in sheet patches for frozen F0 and completed v2 continuity.

Default: validate only. --run starts one CPU process, no inference or training.
The qualified profile is 384-cubed local context at 8.640 um, not a whole-scroll
executor. Existing foreground is preserved, including any erroneous old joins.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

from . import (
    continuity,
    repair,
    sheet_bundle,
    sheet_chart,
    sheet_graph,
    sheet_growth,
    sheet_identity,
    sheet_ordered,
)

PROFILE = "f0-joint-sheet-patches-v1"
PATCH = sheet_chart.PatchPolicy()
GROWTH = sheet_growth.GrowthPolicy()


def save_json(path, value):
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def check_output(inputs, output):
    for source in inputs:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("output must be new and separate from every input tree")
    if output.exists():
        raise FileExistsError("output already exists")
    if list(output.parent.glob(output.name + ".partial-*")):
        raise FileExistsError(
            "partial output requires inspection; no automatic recovery"
        )
    if output.with_name(output.name + ".sheets.lock").exists():
        raise FileExistsError(
            "sheet lock already exists; never reclaim an unknown owner"
        )


def validated_rows(path, shape, lower):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream):
            row = json.loads(line)
            if row.get("accepted") is not True:
                continue
            z = row.get("z_local")
            curve = np.asarray(row.get("path_yx"))
            if (
                type(z) is not int
                or not 0 <= z < shape[0]
                or row.get("z_world") != z + lower[0]
                or curve.ndim != 2
                or curve.shape[1] != 2
                or len(curve) < 2
                or not np.issubdtype(curve.dtype, np.integer)
                or (curve < 0).any()
                or (curve >= np.array(shape[1:])).any()
            ):
                raise ValueError("invalid accepted seed coordinates")
            if not math.isfinite(row.get("length_px", math.nan)) or not math.isfinite(
                row.get("probability_mean", math.nan)
            ):
                raise ValueError("invalid seed evidence")
            rows.append({**row, "id": number})
    return rows


def plan_patches(prior, raw_grid, output, *, voxel_size_um, max_work_gib=8):
    prior, raw_grid, output = (Path(p).resolve() for p in (prior, raw_grid, output))
    if voxel_size_um != 8.64:
        raise ValueError("this local profile is qualified only at 8.640 um")
    if not math.isfinite(max_work_gib) or max_work_gib < 4:
        raise ValueError(
            "finite max-work-gib >= 4 required (estimate, not an OS limit)"
        )
    receipt_path = prior / "continuity_receipt.json"
    receipt = repair.read_object(receipt_path)
    old = receipt["plan"]
    if (
        receipt.get("status") != "complete"
        or receipt.get("profile") != continuity.PROFILE
        or receipt.get("erased_voxels") != 0
        or old.get("checkpoint_sha256") != repair.MODEL_SHA256
        or old.get("threshold") != 0.35
        or old.get("proposal") != asdict(continuity.PROPOSAL)
        or old.get("acceptance") != asdict(continuity.ACCEPTANCE)
        or old.get("voxel_size_um") != 8.64
    ):
        raise ValueError("requires a completed checked frozen-F0 v2 continuity result")
    source, native = Path(old["source"]).resolve(), Path(old["baseline"]).resolve()
    check_output((prior, raw_grid, source, native), output)
    fresh = continuity.plan_continuity(
        source, native, output, voxel_size_um=8.64, max_work_gib=max_work_gib
    )
    for key in (
        "cube_ids",
        "origin_zyx",
        "shape_zyx",
        "umbilicus_yx",
        "probability_dtype",
    ):
        if old[key] != fresh[key]:
            raise ValueError(f"continuity grid mismatch: {key}")
    if fresh["shape_zyx"] != [384, 384, 384]:
        raise ValueError(
            "qualified local context is exactly 3x3x3 cubes; tile explicitly with real context"
        )
    for item in old["input_files"]:
        repair.check_identity(item)
    metadata = repair.read_object(source / "provenance.json")
    if Path(metadata["source_grid"]).resolve() != raw_grid:
        raise ValueError("raw CT grid differs from inference source")
    manifest = repair.read_object(raw_grid / "manifest.json")
    if manifest != repair.read_object(
        source / "source_manifest.json"
    ) or "8.640um" not in manifest.get("sources", {}).get("raw", ""):
        raise ValueError("raw grid provenance or declared CT spacing mismatch")
    files = [
        *fresh["input_files"],
        repair.file_identity(receipt_path),
        repair.file_identity(raw_grid / "manifest.json"),
    ]
    if receipt["ledger"]["path"] != "corridors.jsonl":
        raise ValueError("unexpected ledger path")
    ledger = {**receipt["ledger"], "path": str(prior / "corridors.jsonl")}
    repair.check_identity(ledger)
    files.append(ledger)
    expected = {f"cubes_PRED/{key}.tif" for key in fresh["cube_ids"]}
    if (
        len(receipt["output_files"]) != 27
        or {r["path"] for r in receipt["output_files"]} != expected
    ):
        raise ValueError("continuity output receipt inventory mismatch")
    if (
        sorted(p.stem for p in (prior / "cubes_PRED").glob("*.tif"))
        != fresh["cube_ids"]
    ):
        raise ValueError("extra or missing continuity mask cube")
    for row in receipt["output_files"]:
        item = {**row, "path": str(prior / row["path"])}
        repair.check_identity(item)
        files.append(item)
    total = 0
    for key in fresh["cube_ids"]:
        total += repair.verify_additive(
            repair.binary_cube(native / "cubes_PRED" / f"{key}.tif"),
            repair.binary_cube(prior / "cubes_PRED" / f"{key}.tif"),
        )["added_voxels"]
        path = raw_grid / "cubes_RAW" / f"{key}.tif"
        ct = tifffile.imread(path)
        if ct.shape != (128, 128, 128) or ct.dtype != np.uint8:
            raise ValueError(
                "requires aligned uint8 raw CT cubes, without fabricated context"
            )
        files.append(repair.file_identity(path))
    if total != receipt["added_voxels"]:
        raise ValueError("continuity additions disagree with receipt")
    rows = validated_rows(
        prior / "corridors.jsonl", fresh["shape_zyx"], fresh["origin_zyx"]
    )
    if len(rows) != receipt["accepted_corridors"]:
        raise ValueError("seed count differs from continuity receipt")
    code = [
        repair.file_identity(Path(m.__file__))
        for m in (
            continuity,
            repair,
            sheet_chart,
            sheet_growth,
            sheet_bundle,
            sheet_graph,
            sheet_ordered,
            sheet_identity,
        )
    ]
    code.append(repair.file_identity(Path(__file__)))
    return {
        "schema": "socratic-sheet-plan-v1",
        "profile": PROFILE,
        "source_continuity": str(prior),
        "source": str(source),
        "native": str(native),
        "raw_grid": str(raw_grid),
        "output": str(output),
        "spec": {
            "target_cube_ids": fresh["cube_ids"],
            "origin_zyx": fresh["origin_zyx"],
            "shape_zyx": fresh["shape_zyx"],
        },
        "patch_policy": asdict(PATCH),
        "growth_policy": asdict(GROWTH),
        "code": code,
        "input_files": files,
        "model_sha256": repair.MODEL_SHA256,
        "threshold": 0.35,
        "voxel_size_um": 8.64,
        "accepted_seed_count": len(rows),
        "estimated_work_gib": 4,
        "max_work_gib": max_work_gib,
        "memory_estimate_is_not_a_hard_process_limit": True,
        "processes": 1,
        "child_processes": 0,
        "automatic_deployment": False,
        "anatomical_correctness_requires_review": True,
    }


def assemble(directory, ids, lower, dtype):
    value = np.empty((384, 384, 384), dtype)
    for key in ids:
        value[continuity.window_for(key, lower)] = tifffile.imread(
            directory / f"{key}.tif"
        ).astype(dtype)
    return value


def assign_components(rows, base, native):
    labels, _ = ndimage.label(base & ~native, np.ones((3, 3, 3), bool))
    for row in rows:
        path = np.array(row["path_yx"])
        ids = labels[row["z_local"], path[:, 0], path[:, 1]]
        ids = np.unique(ids[ids > 0])
        if len(ids) != 1:
            raise ValueError(
                "accepted path does not identify exactly one v2 addition component"
            )
        row["added_component"] = int(ids[0])
    return rows


def fit_arrays(p, ct, base, rows, emit, progress):
    """Selected deterministic growth loop; emit persists each decision immediately."""
    if p.dtype != np.float32 or p.shape != ct.shape or base.shape != p.shape:
        raise ValueError("incompatible arrays")
    members = defaultdict(list)
    for row in rows:
        members[row["added_component"]].append(row)
    components = sorted(
        members, key=lambda n: (-max(r["length_px"] for r in members[n]), n)
    )
    owners = np.zeros(base.shape, np.uint16)
    meshes, accepted, census = {}, [], Counter()

    @lru_cache(maxsize=96)
    def plane_labels(z):
        return ndimage.label(base[z], np.ones((3, 3), bool))[0]

    for count, component in enumerate(components, 1):
        candidates = sorted(
            members[component],
            key=lambda r: (-r["length_px"], -r["probability_mean"], r["id"]),
        )[:3]
        for row in candidates:
            entry = {
                "seed_id": row["id"],
                "prior_added_component": component,
                "seed_z_world": row["z_world"],
                "seed_length_px": row["length_px"],
            }
            proposal = None
            try:
                if row["length_px"] < 12:
                    raise ValueError("seed too short for this patch profile")
                proposal = sheet_growth.propose_patch(
                    p,
                    ct,
                    base,
                    np.array(row["path_yx"]),
                    row["z_local"],
                    plane_labels,
                    PATCH,
                    GROWTH,
                )
                w, new = proposal["window"], proposal["added"]
                touched = owners[w][
                    ndimage.binary_dilation(new, np.ones((3, 3, 3), bool))
                ]
                interactions = []
                for other in np.unique(touched[touched > 0]).tolist():
                    compatible, metrics = sheet_growth.compatible_surfaces(
                        proposal["mesh"], meshes[other], GROWTH
                    )
                    interactions.append({"prior_patch": other, **metrics})
                    if not compatible:
                        raise ValueError("incompatible interacting surface patches")
                unique = new & (owners[w] == 0)
                if unique.sum() < GROWTH.minimum_new_voxels:
                    raise ValueError(
                        "patch already supplied by an earlier compatible surface"
                    )
                number = len(accepted) + 1
                if number >= np.iinfo(np.uint16).max:
                    raise ValueError("patch ID capacity exceeded")
                owners[w][unique] = number
                meshes[number] = proposal["mesh"]
                positions = np.argwhere(unique) + [s.start for s in w]
                entry.update(
                    accepted=True,
                    patch_id=number,
                    reason="accepted",
                    metrics=proposal["metrics"],
                    unique_added_voxels=int(unique.sum()),
                    compatible_interactions=interactions,
                    view_center_local=positions[len(positions) // 2].tolist(),
                    mesh=f"meshes/patch_{number:04d}.ply",
                    mesh_arrays=f"meshes/patch_{number:04d}.npz",
                )
            except ValueError as error:
                entry.update(accepted=False, reason=str(error))
            census[entry["reason"]] += 1
            emit(entry, proposal["mesh"] if entry["accepted"] else None)
            if entry["accepted"]:
                accepted.append(entry)
                break
        if count % 16 == 0:
            progress(count, len(components), len(accepted), int((owners > 0).sum()))
    return owners > 0, accepted, dict(census), len(components)


def write_mesh(folder, row, mesh, lower):
    np.savez_compressed(
        folder / row["mesh_arrays"],
        vertices_local_zyx=mesh["vertices"],
        faces=mesh["faces"],
        normals_zyx=mesh["normals"],
    )
    vertices = mesh["vertices"] + lower
    with (folder / row["mesh"]).open("x", encoding="ascii", newline="\n") as stream:
        stream.write(
            f"ply\nformat ascii 1.0\ncomment world voxel XYZ; pitch 8.640 um\nelement vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(mesh['faces'])}\nproperty list uchar int vertex_indices\nend_header\n"
        )
        for z, y, x in vertices:
            stream.write(f"{x:.5f} {y:.5f} {z:.5f}\n")
        for a, b, c in mesh["faces"]:
            stream.write(f"3 {a} {b} {c}\n")


def run_patches(plan):
    if (
        plan["profile"] != PROFILE
        or plan["patch_policy"] != asdict(PATCH)
        or plan["growth_policy"] != asdict(GROWTH)
    ):
        raise ValueError("plan differs from fixed profile")
    prior, raw, output, source, native = (
        Path(plan[k])
        for k in ("source_continuity", "raw_grid", "output", "source", "native")
    )
    current = plan_patches(
        prior,
        raw,
        output,
        voxel_size_um=plan["voxel_size_um"],
        max_work_gib=plan["max_work_gib"],
    )
    if current != plan:
        raise ValueError(
            "plan changed or differs from current independently checked inputs"
        )
    check_output((prior, raw, source, native), output)
    for item in plan["input_files"] + plan["code"]:
        repair.check_identity(item)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + ".sheets.lock")
    partial = output.with_name(output.name + f".partial-{os.getpid()}")
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(
            {"pid": os.getpid(), "output": str(output), "partial": str(partial)}, stream
        )
    try:
        if output.exists() or list(output.parent.glob(output.name + ".partial-*")):
            raise FileExistsError("output or partial appeared after preflight")
        partial.mkdir()
        (partial / "meshes").mkdir()
        (partial / "cubes_PRED").mkdir()
        save_json(partial / "request.json", plan)
        save_json(partial / "state.json", {"status": "running", "pid": os.getpid()})
        ids, lower = plan["spec"]["target_cube_ids"], plan["spec"]["origin_zyx"]
        base = assemble(prior / "cubes_PRED", ids, lower, bool)
        baseline = assemble(native / "cubes_PRED", ids, lower, bool)
        p = assemble(source / "probability", ids, lower, np.float32)
        ct = assemble(raw / "cubes_RAW", ids, lower, np.float32)
        if not (ct > 0).any():
            raise ValueError("empty CT context")
        lo, hi = np.percentile(ct[ct > 0], [1, 99.5])
        if hi <= lo:
            raise ValueError("degenerate CT contrast window")
        ct = np.clip((ct - lo) * 255 / (hi - lo), 0, 255)
        rows = assign_components(
            validated_rows(prior / "corridors.jsonl", base.shape, lower), base, baseline
        )
        # Verify accepted 2D paint endpoints and accounting again from arrays.
        by_z = defaultdict(list)
        for row in rows:
            by_z[row["z_local"]].append(row)
        for z in range(384):
            continuity.audit_plane(baseline[z], base[z], by_z[z])
        del baseline
        ledger = []
        with (partial / "ledger.jsonl").open("x", encoding="utf-8") as log:

            def emit(row, mesh):
                if mesh is not None:
                    write_mesh(partial, row, mesh, np.array(lower))
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                ledger.append(row)

            def progress(count, total, accepted, voxels):
                save_json(
                    partial / "state.json",
                    {
                        "status": "running",
                        "pid": os.getpid(),
                        "processed_components": count,
                        "total_components": total,
                        "accepted_patches": accepted,
                        "added_voxels": voxels,
                    },
                )
                print(
                    f"Sheet patches: {count}/{total} components, {accepted} patches, {voxels} additions",
                    flush=True,
                )

            added, accepted, census, count = fit_arrays(
                p, ct, base, rows, emit, progress
            )
        after = base | added
        if (
            (base & ~after).any()
            or int(added.sum()) != sum(r["unique_added_voxels"] for r in accepted)
            or (p[added] < 0.16).any()
        ):
            raise ValueError("final addition accounting or probability check failed")
        files, cubes = [], {}
        for key in ids:
            path = partial / "cubes_PRED" / f"{key}.tif"
            w = continuity.window_for(key, lower)
            tifffile.imwrite(
                path, after[w].astype(np.uint8) * 255, photometric="minisblack"
            )
            actual = repair.binary_cube(path) != 0
            if not np.array_equal(actual, after[w]) or (base[w] & ~actual).any():
                raise ValueError("written output verification failed")
            files.append(
                {
                    **repair.file_identity(path),
                    "path": str(output / "cubes_PRED" / f"{key}.tif"),
                }
            )
            cubes[key] = {"added_voxels": int(added[w].sum()), "erased_voxels": 0}
        for item in plan["input_files"] + plan["code"]:
            repair.check_identity(item)
        summary = {
            "status": "checked_local_candidate_complete",
            "profile": PROFILE,
            "accepted_patches": len(accepted),
            "attempted_seeds": len(ledger),
            "prior_added_components": count,
            "added_voxels_over_v2": int(added.sum()),
            "core_added_voxels_over_v2": int(added[128:256, 128:256, 128:256].sum()),
            "erased_v2_voxels": 0,
            "summed_patch_area_px2_not_unique": sum(
                r["metrics"]["mesh_area_px2"] for r in accepted
            ),
            "maximum_mesh_z_span": max(
                (r["metrics"]["mesh_z_span"] for r in accepted), default=0
            ),
            "patches_with_modeled_neighbors": sum(
                r["metrics"]["modeled_surfaces"] > 1 for r in accepted
            ),
            "all_meshes_manifold_with_boundary": all(
                r["metrics"]["manifold_with_boundary"] for r in accepted
            ),
            "added_voxel_components": int(
                ndimage.label(added, np.ones((3, 3, 3), bool))[1]
            ),
            "census": census,
            "cubes": cubes,
            "output_files": files,
            "ct_window": [float(lo), float(hi)],
            "automatic_deployment": False,
            "anatomical_correctness_requires_review": True,
            "summed_area_overcounts_compatible_overlap": True,
            "no_new_official_dice": True,
        }
        save_json(partial / "ledger.json", {"seeds": ledger})
        save_json(partial / "patches.json", {"patches": accepted})
        save_json(partial / "summary.json", summary)
        save_json(
            partial / "state.json",
            {
                "status": "complete",
                "pid": os.getpid(),
                "completed_utc": datetime.now(UTC).isoformat(),
            },
        )
        if output.exists():
            raise FileExistsError("output appeared before publication")
        partial.rename(output)
        return output
    except BaseException as error:
        if partial.exists():
            save_json(
                partial / "failure.json",
                {"status": "failed_or_interrupted", "error": repr(error)},
            )
            save_json(
                partial / "state.json",
                {"status": "failed_or_interrupted", "pid": os.getpid()},
            )
        raise
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("continuity", type=Path)
    parser.add_argument("raw_grid", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--voxel-size-um", type=float, required=True)
    parser.add_argument("--max-work-gib", type=float, default=8)
    parser.add_argument(
        "--run", action="store_true", help="execute; default preflight writes no output"
    )
    args = parser.parse_args()
    plan = plan_patches(
        args.continuity,
        args.raw_grid,
        args.output,
        voxel_size_um=args.voxel_size_um,
        max_work_gib=args.max_work_gib,
    )
    print(run_patches(plan) if args.run else json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
