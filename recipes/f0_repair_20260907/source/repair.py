"""Checked, additive gap repair for contiguous frozen-F0 prediction grids.

No training/inference, threshold search, erosion, or automatic deployment.
The initial profile pins the measured Windows native build; a different native
build requires a separately reviewed profile, not a silent executable swap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

PROFILE = "f0-contiguous-gap-repair-v1"
MODEL_SHA256 = "54db9a59cb602ff08e3dda0acc1195ad980ab05fb660332d32fb436a0d49a175"
BINARY_SHA256 = "54fe434f76beb8c1eff58cdb1bd72200e4630497e23539cd467e4e64360f1e13"
PARAMETERS = {
    "reach_safe": 6,
    "reach_max": 21,
    "radial_dr": 3,
    "min_support": 0,
    "min_score": 0.30,
    "paint_radius": 1,
    "radial_armed": 1,
    "tracks": 1,
}
CUBE_NAME = re.compile(r"^z(\d+)_y(\d+)_x(\d+)\.tif$", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def file_identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def check_identity(item: dict[str, Any]) -> None:
    path = Path(item["path"])
    if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        raise ValueError(f"input changed during repair: {path}")


def binary_cube(path: Path) -> np.ndarray:
    value = tifffile.imread(path)
    if value.shape != (128, 128, 128) or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 128-cube: {path}")
    if not np.isin(value, [0, 255]).all():
        raise ValueError(f"non-binary foreground encoding: {path}")
    return value


def verify_additive(before: np.ndarray, after: np.ndarray) -> dict[str, int]:
    if before.shape != after.shape:
        raise ValueError("repair changed cube shape")
    original, repaired = before != 0, after != 0
    erased = int((original & ~repaired).sum())
    if erased:
        raise ValueError(f"repair erased {erased} original foreground voxels")
    return {
        "foreground_before": int(original.sum()),
        "added_voxels": int((repaired & ~original).sum()),
        "erased_voxels": 0,
    }


def plan_repair(
    source: Path,
    output: Path,
    binary: Path,
    *,
    voxel_size_um: float,
    threads: int = 16,
    max_work_gib: float = 32,
) -> dict[str, Any]:
    source, output, binary = source.resolve(), output.resolve(), binary.resolve()
    if (
        output.exists()
        or source == output
        or source in output.parents
        or output in source.parents
    ):
        raise ValueError("output must be new and separate from the input tree")
    if list(output.parent.glob(output.name + ".partial-*")):
        raise ValueError(
            "partial repair exists; inspect its process/artifacts before recovery"
        )
    if not 1 <= threads <= 16:
        raise ValueError("threads must be between 1 and 16")
    if not math.isfinite(voxel_size_um) or not 8 <= voxel_size_um <= 10:
        raise ValueError(
            "this coarse-grid profile requires explicit voxel spacing in [8, 10] um; do not transfer fine-grid pixel settings"
        )
    if not math.isfinite(max_work_gib) or max_work_gib <= 0:
        raise ValueError("max-work-gib must be finite and positive")
    if sha256(binary) != BINARY_SHA256:
        raise ValueError(
            "native binary SHA-256 does not match the measured repair profile"
        )
    metadata_path = source / "provenance.json"
    if not metadata_path.is_file():
        metadata_path = source / "manifest.json"
    metadata = read_object(metadata_path)
    model_sha = metadata.get(
        "checkpoint_sha256", metadata.get("checkpoint", {}).get("sha256")
    )
    if model_sha != MODEL_SHA256 or metadata.get("threshold") != 0.35:
        raise ValueError(
            "this profile requires the frozen F0/200k prediction at T=0.35"
        )
    grid_manifest_path = source / "manifest.json"
    if not grid_manifest_path.exists():
        grid_manifest_path = source / "source_manifest.json"
    manifest = read_object(grid_manifest_path)
    if manifest.get("chunk_size") != 128:
        raise ValueError("unsupported prediction chunk size")
    umb = manifest.get("umbilicus_yx")
    if (
        not isinstance(umb, list)
        or len(umb) != 2
        or any(
            not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0
            for v in umb
        )
    ):
        raise ValueError(
            "an explicit finite umbilicus_yx is required to arm the radial guard"
        )
    paths = sorted((source / "cubes_PRED").glob("*.tif"))
    if not paths:
        raise ValueError("no prediction cubes")
    origins = []
    for path in paths:
        match = CUBE_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid cube filename: {path.name}")
        point = tuple(int(v) for v in match.groups())
        if any(v % 128 for v in point):
            raise ValueError("prediction cube coordinates are not chunk aligned")
        origins.append(point)
    if len(set(origins)) != len(origins):
        raise ValueError("duplicate world-coordinate cubes")
    declared = metadata.get("target_cube_ids")
    if declared is not None and sorted(declared) != [p.stem for p in paths]:
        raise ValueError(
            "prediction files disagree with the inference target inventory"
        )
    lower = [min(p[axis] for p in origins) for axis in range(3)]
    counts = [
        (max(p[axis] for p in origins) - lower[axis]) // 128 + 1 for axis in range(3)
    ]
    if min(counts) < 3 or math.prod(counts) != len(origins):
        raise ValueError(
            "repair requires a dense contiguous grid with at least 3 cubes per axis; missing context must not be fabricated"
        )
    dims = [v * 128 for v in counts]
    estimated_bytes = math.prod(dims) * 12 + threads * dims[1] * dims[2] * 48
    if estimated_bytes > max_work_gib * 1024**3:
        raise ValueError(
            "estimated native working set exceeds max-work-gib; use an explicitly planned contiguous region"
        )
    identities = [file_identity(metadata_path)]
    if grid_manifest_path != metadata_path:
        identities.append(file_identity(grid_manifest_path))
    foreground = 0
    for path in paths:
        foreground += int((binary_cube(path) != 0).sum())
        identities.append(file_identity(path))
    return {
        "schema": "socratic-repair-plan-v1",
        "profile": PROFILE,
        "source": str(source),
        "output": str(output),
        "binary": file_identity(binary),
        "checkpoint_sha256": MODEL_SHA256,
        "threshold": 0.35,
        "voxel_size_um": voxel_size_um,
        "umbilicus_yx": umb,
        "parameters": dict(PARAMETERS),
        "threads": threads,
        "cube_ids": [p.stem for p in paths],
        "origin_zyx": lower,
        "shape_zyx": dims,
        "foreground_before": foreground,
        "estimated_work_gib": estimated_bytes / 1024**3,
        "memory_estimate_is_not_a_hard_process_limit": True,
        "input_files": identities,
        "automatic_deployment": False,
        "context_boundary_note": "Outer edges remain clipped; evaluate/use interior regions with real neighboring predictions.",
    }


def native_command(plan: dict[str, Any], destination: Path) -> list[str]:
    umb = plan["umbilicus_yx"]
    # The native reporter embeds grid paths in JSON without escaping backslashes.
    # Windows accepts forward slashes, preserving valid native provenance JSON.
    result = [
        Path(plan["binary"]["path"]).as_posix(),
        Path(plan["source"]).as_posix(),
        destination.as_posix(),
        "--umb-y",
        str(umb[0]),
        "--umb-x",
        str(umb[1]),
    ]
    for key in (
        "reach_safe",
        "reach_max",
        "radial_dr",
        "min_support",
        "min_score",
        "paint_radius",
    ):
        result.extend(["--" + key.replace("_", "-"), str(PARAMETERS[key])])
    result.extend(
        [
            "--threads",
            str(plan["threads"]),
            "--png-max",
            "24",
            "--crop-max",
            "48",
            "--rej-crop-max",
            "24",
        ]
    )
    return result


def run_repair(plan: dict[str, Any]) -> Path:
    output, source = Path(plan["output"]), Path(plan["source"])
    if output.exists():
        raise FileExistsError("repair output was created after preflight")
    for item in [plan["binary"], *plan["input_files"]]:
        check_identity(item)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + f".partial-{os.getpid()}")
    lock = output.with_name(output.name + ".repair.lock")
    # Refuse unknown/interrupted owners; never reclaim a lock by timeout.
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(
            {"pid": os.getpid(), "partial": str(partial), "output": str(output)}, stream
        )
    try:
        # Another invocation may have finished after this invocation's preflight.
        if output.exists():
            raise FileExistsError("repair output was created before lock acquisition")
        if list(output.parent.glob(output.name + ".partial-*")):
            raise ValueError(
                "partial repair exists; inspect its process/artifacts before recovery"
            )
        partial.mkdir()
        (partial / "repair_plan.json").write_text(
            json.dumps(plan, indent=2) + "\n", encoding="utf-8"
        )
        command = native_command(plan, partial)
        with (partial / "native.log").open("xb") as stream:
            process = subprocess.Popen(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": str(plan["threads"])},
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            (partial / "process.json").write_text(
                json.dumps({"pid": process.pid, "command": command}) + "\n",
                encoding="utf-8",
            )
            try:
                code = process.wait()
            except BaseException:
                # Cancellation must not leave our native worker running after
                # releasing the lock. Never touch processes we did not start.
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise
        if code:
            raise RuntimeError(
                f"native repair failed ({code}); preserved diagnostics at {partial}"
            )
        native = read_object(partial / "fixup_report.json")
        if native["params"] != PARAMETERS or native["bridges"]["cut"] != 0:
            raise ValueError("native result violates the declared additive profile")
        observed = sorted(p.stem for p in (partial / "cubes_PRED").glob("*.tif"))
        if observed != plan["cube_ids"]:
            raise ValueError("native repair changed the cube inventory")
        cubes, outputs = {}, []
        for key in plan["cube_ids"]:
            path = partial / "cubes_PRED" / f"{key}.tif"
            cubes[key] = verify_additive(
                binary_cube(source / "cubes_PRED" / f"{key}.tif"), binary_cube(path)
            )
            outputs.append(
                {
                    "path": f"cubes_PRED/{key}.tif",
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        added = sum(row["added_voxels"] for row in cubes.values())
        if (
            added != native["painted_px"]
            or plan["foreground_before"] != native["fg_in"]
        ):
            raise ValueError(
                "native counters disagree with independent voxel verification"
            )
        for item in [plan["binary"], *plan["input_files"]]:
            check_identity(item)
        receipt = {
            "schema": "socratic-additive-repair-v1",
            "status": "complete",
            "profile": PROFILE,
            "completed_utc": datetime.now(UTC).isoformat(),
            "plan": plan,
            "native": native,
            "cubes": cubes,
            "added_voxels": added,
            "erased_voxels": 0,
            "output_files": outputs,
            "anatomical_correctness_not_implied_by_additivity": True,
        }
        (partial / "repair_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        # Publish the validated directory, never overwrite an existing result.
        if output.exists():
            raise FileExistsError("repair output appeared before commit")
        partial.rename(output)
        return output
    finally:
        lock.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--voxel-size-um", type=float, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-work-gib", type=float, default=32)
    parser.add_argument(
        "--run",
        action="store_true",
        help="execute after preflight; default prints the plan only",
    )
    args = parser.parse_args()
    plan = plan_repair(
        args.source,
        args.output,
        args.binary,
        voxel_size_um=args.voxel_size_um,
        threads=args.threads,
        max_work_gib=args.max_work_gib,
    )
    if args.run:
        print(run_repair(plan))
    else:
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
