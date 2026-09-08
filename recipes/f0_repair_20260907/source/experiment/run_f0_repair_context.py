"""Prepare and infer a bounded, provenance-pinned F0 repair test bed.

This is not a trainer or the final-run launcher. It never edits the selected
model, old reports, source CT, or native filler. One process owns the shared
GPU lock; two blocks are inferred sequentially. Interrupted partial outputs
are preserved and require inspection, not an automatic fresh restart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "output/crossres_data/releases/c3-f0-200k-20260907"
OUTPUT = ROOT / "output/crossres_data/f0_repair_20260907"
SOURCE = ROOT / "PHerc1447-10x10x10-overnight"
MODEL = RELEASE / "checkpoint_f0_00200000.pt"
MODEL_SHA = "54db9a59cb602ff08e3dda0acc1195ad980ab05fb660332d32fb436a0d49a175"
EXE = ROOT / "build/c3-human-line-fitter_20260905/extracted/Release/pred_fixup.exe"
EXE_SHA = "54fe434f76beb8c1eff58cdb1bd72200e4630497e23539cd467e4e64360f1e13"
GPU = "GPU-304375e6-2d2f-b771-b8be-a0c095fe209e"
BLOCKS = {"development": (12160, 3968, 3072), "control": (12800, 4096, 3328)}
PRESETS = {
    "report_baseline": {"reach_max": 12, "radial_dr": 4, "min_support": 3},
    "support_only": {"reach_max": 12, "radial_dr": 4, "min_support": 0},
    "radial_safe": {"reach_max": 12, "radial_dr": 3, "min_support": 0},
    "extended_safe": {"reach_max": 21, "radial_dr": 3, "min_support": 0},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def cube_id(origin: tuple[int, int, int]) -> str:
    return f"z{origin[0]:05d}_y{origin[1]:05d}_x{origin[2]:05d}"


def block_origins(center: tuple[int, int, int], radius: int = 1) -> list[tuple[int, int, int]]:
    if radius < 0 or any(value < 0 or value % 128 for value in center):
        raise ValueError("block centers must be non-negative and chunk aligned")
    points = [tuple(value + 128 * offset for value, offset in zip(center, delta))
              for delta in product(range(-radius, radius + 1), repeat=3)]
    if any(min(point) < 0 for point in points):
        raise ValueError("block extends below zero")
    return sorted(points)


def inventory(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_file(item: dict) -> None:
    path = Path(item["path"])
    if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        raise ValueError(f"pinned input changed: {path}")


def verify_inference(path: Path, request: dict, name: str) -> dict:
    provenance = read(path / "provenance.json")
    expected_ids = request["blocks"][name]["target_cube_ids"]
    if (provenance["checkpoint_sha256"] != MODEL_SHA
            or provenance["target_cube_ids"] != expected_ids
            or provenance["threshold"] != 0.35 or provenance["halo"] != 32
            or provenance["mirror_tta"] is not True
            or provenance["amp_dtype"] != "bfloat16"):
        raise ValueError(f"inference identity mismatch: {name}")
    files = []
    for group in ("cubes_PRED", "probability"):
        observed = sorted(p.stem for p in (path / group).glob("*.tif"))
        if observed != expected_ids:
            raise ValueError(f"incomplete/extra inference cubes: {name}/{group}")
        files.extend(inventory(path / group / f"{key}.tif") for key in expected_ids)
    return {"provenance": inventory(path / "provenance.json"), "files": files}


def prepare() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"inspect existing repair work instead of overwriting: {OUTPUT}")
    if sha256(MODEL) != MODEL_SHA or MODEL.stat().st_size != 409674095:
        raise ValueError("frozen model mismatch")
    if sha256(EXE) != EXE_SHA:
        raise ValueError("native filler mismatch")
    manifest = read(SOURCE / "manifest.json")
    if manifest["chunk_size"] != 128 or manifest["umbilicus_yx"] != [3287.0, 3784.0]:
        raise ValueError("unexpected grid geometry/umbilicus")
    published = set(read(SOURCE / "cubes_PRED/present.json"))
    required_raw: set[tuple[int, int, int]] = set()
    blocks = {}
    for name, center in BLOCKS.items():
        targets = block_origins(center)
        ids = [cube_id(origin) for origin in targets]
        if not set(ids) <= published:
            raise ValueError(f"target cubes missing: {name}")
        # Each 192^3 inference context intersects the neighboring 128^3 cubes.
        for origin in targets:
            required_raw.update(block_origins(origin))
        blocks[name] = {
            "center_zyx": list(center), "target_cube_ids": ids,
            "shape_zyx": [384, 384, 384],
            "origin_zyx": [value - 128 for value in center],
            "primary_assessment_core_cube": cube_id(center),
        }
    pins = read(RELEASE / "provenance/recipe.json")["execution"]["code_pins"]
    for relative, expected in pins.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"sealed inference/runtime source changed: {relative}")
    files = [inventory(SOURCE / "manifest.json"), inventory(SOURCE / "cubes_PRED/present.json")]
    for index, origin in enumerate(sorted(required_raw), 1):
        files.append(inventory(SOURCE / "cubes_RAW" / f"{cube_id(origin)}.tif"))
        if index % 25 == 0:
            print(f"preflight: hashed {index}/{len(required_raw)} CT cubes", flush=True)
    request = {
        "schema": "f0-repair-context-v1", "created_utc": datetime.now(UTC).isoformat(),
        "goal": "Improve post-prediction repair", "runner": inventory(Path(__file__)),
        "source_grid": str(SOURCE), "source_manifest": manifest,
        "source_voxel_pitch_note": "PHerc1447 manifest names the 8.640um raw volume; do not assume the generic M7 training pitch or the high-resolution adjudication pitch.",
        "model": inventory(MODEL), "native_filler": inventory(EXE), "blocks": blocks,
        "threshold": 0.35, "inference": {"halo": 32, "tta": True, "amp_dtype": "bfloat16", "gpu_uuid": GPU},
        "sealed_runtime_pins": pins, "input_files": files, "presets": PRESETS,
        "common_filler_options": {"reach_safe": 6, "min_score": 0.30, "paint_radius": 1, "tracks": True, "cut_bridges": False},
        "evaluation_contract": {
            "development": "Compare the four declared presets on development only; inspect added connections, not only added-voxel totals.",
            "control": "Reserve the control block for baseline-versus-selected-policy verification; do not select the preset on it.",
            "invariants": ["no erased foreground", "no fabricated absent context", "no model/threshold change", "no unreviewed promotion", "no public upload"],
            "needed_evidence": ["additive identity", "boundary/candidate/support census", "CT and probability review of new joins", "cross-sheet and chained-merge review", "control verification"],
            "geometry_reference_is_not_ground_truth": True,
            "production_readiness_not_established_by_join_count": True,
        },
    }
    OUTPUT.mkdir()
    atomic_json(OUTPUT / "request.json", request)
    print(f"prepared {len(blocks)} blocks / 54 targets / {len(required_raw)} raw context cubes", flush=True)


def infer() -> None:
    request = read(OUTPUT / "request.json")
    for item in [request["runner"], request["model"], *request["input_files"]]:
        verify_file(item)
    for relative, expected in request["sealed_runtime_pins"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"sealed runtime changed: {relative}")
    for name in request["blocks"]:
        if list(OUTPUT.glob(f"inference_{name}.partial-*")):
            raise RuntimeError(f"interrupted partial exists for {name}; inspect its owner and output before recovery")
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    os.environ["OMP_NUM_THREADS"] = "16"
    from run_unified_ladder import acquire_gpu_lock, release_gpu_lock
    from crossres_pred.voxel.grid_inference import infer_voxel_grid
    state = {"schema": "f0-repair-context-inference-v1", "pid": os.getpid(), "status": "waiting_for_gpu", "request_sha256": sha256(OUTPUT / "request.json"), "blocks": {}}
    atomic_json(OUTPUT / "inference_state.json", state)
    acquire_gpu_lock("repair:F0-200k:two-contiguous-blocks", poll_seconds=30)
    try:
        for name in ("development", "control"):
            destination = OUTPUT / f"inference_{name}"
            state.update(status="running", active_block=name)
            atomic_json(OUTPUT / "inference_state.json", state)
            if not destination.exists():
                infer_voxel_grid(source_grid=SOURCE, checkpoint_path=MODEL, output_path=destination,
                                 threshold=0.35, halo=32, device_name="cuda", amp_dtype_name="bfloat16",
                                 mirror_tta=True, max_cpu_threads=16,
                                 target_cube_ids=request["blocks"][name]["target_cube_ids"])
            state["blocks"][name] = verify_inference(destination, request, name)
            atomic_json(OUTPUT / "inference_state.json", state)
        state.update(status="complete", completed_utc=datetime.now(UTC).isoformat())
        atomic_json(OUTPUT / "inference_state.json", state)
    except BaseException as error:
        state.update(status="failed", error=repr(error), failed_utc=datetime.now(UTC).isoformat())
        atomic_json(OUTPUT / "inference_state.json", state)
        raise
    finally:
        release_gpu_lock()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "infer"))
    args = parser.parse_args()
    {"prepare": prepare, "infer": infer}[args.phase]()


if __name__ == "__main__":
    main()
