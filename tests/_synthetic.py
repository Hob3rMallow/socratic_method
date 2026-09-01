"""Synthetic fixtures shared by the crossres v2 tests.

Everything here builds a tiny consistent world: a coarse frame and a fine
frame related by the affine ``coarse = 0.25 * fine`` (scale ratio 4), one
traced sheet at coarse z = 10 (fine z = 40), a local fine raw zarr store,
and a coarse volume with an m7-style baseline band.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

FINE_TO_COARSE_AFFINE = [
    [0.25, 0.0, 0.0, 0.0],
    [0.0, 0.25, 0.0, 0.0],
    [0.0, 0.0, 0.25, 0.0],
]
COARSE_PLANE_Z = 10.0
COARSE_UM = 9.0
FINE_UM = 2.25


def write_tifxyz_plane(
    directory: Path,
    *,
    plane_z: float,
    extent: float,
    grid_step: float,
) -> Path:
    """Write x/y/z.tif for a flat sheet z = plane_z covering [0, extent)^2."""

    directory.mkdir(parents=True, exist_ok=True)
    axis = np.arange(0.0, extent + 1.0e-3, grid_step, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    zz = np.full_like(xx, plane_z)
    for name, value in (("x", xx), ("y", yy), ("z", zz)):
        tifffile.imwrite(directory / f"{name}.tif", value)
    return directory


def write_pair_manifest(
    tmp_path: Path,
    *,
    scroll_id: str = "PHerc1667",
    split: str = "train",
    coarse_volume: Path | None = None,
    baseline_volume: Path | None = None,
) -> Path:
    # Deliberately sparse parameter grids (public exports are ~20 voxels
    # between cells): every consumer must densify before rasterizing.
    coarse_dir = write_tifxyz_plane(
        tmp_path / "maps" / "coarse",
        plane_z=COARSE_PLANE_Z,
        extent=48.0,
        grid_step=4.0,
    )
    fine_dir = write_tifxyz_plane(
        tmp_path / "maps" / "fine",
        plane_z=COARSE_PLANE_Z * 4.0,
        extent=192.0,
        grid_step=16.0,
    )
    row = {
        "schema_version": 1,
        "record_id": f"{scroll_id.lower()}-test",
        "scroll_id": scroll_id,
        "split": split,
        "coarse": {
            "scan_id": "coarse-scan",
            "voxel_um": COARSE_UM,
            **(
                {"volume": coarse_volume.as_posix()}
                if coarse_volume is not None
                else {}
            ),
            **(
                {"baseline": baseline_volume.as_posix()}
                if baseline_volume is not None
                else {}
            ),
        },
        "fine": {
            "scan_id": "fine-scan",
            "voxel_um": FINE_UM,
            "to_coarse_affine_xyz": FINE_TO_COARSE_AFFINE,
        },
        "surfaces": [
            {
                "surface_id": "sheet-a",
                "coarse_tifxyz": coarse_dir.as_posix(),
                "fine_tifxyz": fine_dir.as_posix(),
            }
        ],
    }
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest


def write_policy(tmp_path: Path, *, scroll_id: str = "PHerc1667") -> Path:
    policy = tmp_path / "policy.toml"
    policy.write_text(
        f"""schema_version = 1
profile = "research"
[policy]
forbid_fine_for_scrolls = []
allow_scrolls = ["{scroll_id}"]
[splits]
train = ["{scroll_id}"]
""",
        encoding="utf-8",
    )
    return policy


def write_local_fine_store(
    tmp_path: Path,
    *,
    shape: tuple[int, int, int] = (256, 256, 256),
    chunks: tuple[int, int, int] = (64, 64, 64),
    plane_z: int = 40,
    masked_y_from: int | None = None,
) -> Path:
    """A local zarr-v2 u1 store whose chunks contain a bright sheet.

    ``masked_y_from`` zeroes all voxels at global y >= that coordinate,
    imitating a masked scan whose footprint ends mid-segment.
    """

    root = tmp_path / "fine_store"
    array = root / "0"
    array.mkdir(parents=True, exist_ok=True)
    (array / ".zarray").write_text(
        json.dumps(
            {
                "zarr_format": 2,
                "shape": list(shape),
                "chunks": list(chunks),
                "dtype": "|u1",
                "compressor": None,
                "fill_value": 0,
                "filters": None,
                "order": "C",
                "dimension_separator": "/",
            }
        ),
        encoding="utf-8",
    )
    (root / ".zgroup").write_text(
        json.dumps({"zarr_format": 2}), encoding="utf-8"
    )
    grid = tuple(-(-shape[axis] // chunks[axis]) for axis in range(3))
    rng = np.random.default_rng(5)
    for cz in range(grid[0]):
        for cy in range(grid[1]):
            for cx in range(grid[2]):
                block = rng.integers(
                    20, 90, size=chunks, dtype=np.int64
                ).astype(np.uint8)
                z_lo = cz * chunks[0]
                for z in range(chunks[0]):
                    if abs(z_lo + z - plane_z) <= 4:
                        block[z] = 180
                if masked_y_from is not None:
                    y_lo = cy * chunks[1]
                    local_from = max(0, masked_y_from - y_lo)
                    if local_from < chunks[1]:
                        block[:, local_from:, :] = 0
                chunk_dir = array / str(cz) / str(cy)
                chunk_dir.mkdir(parents=True, exist_ok=True)
                (chunk_dir / str(cx)).write_bytes(block.tobytes())
    return root


def write_coarse_volume(
    tmp_path: Path,
    *,
    shape: tuple[int, int, int] = (64, 64, 64),
    plane_z: int = 10,
) -> tuple[Path, Path]:
    """A coarse raw .npy plus an m7-style binary baseline band .npy."""

    rng = np.random.default_rng(9)
    volume = rng.integers(20, 90, size=shape, dtype=np.int64).astype(np.uint8)
    volume[plane_z - 1 : plane_z + 2] = 170
    raw_path = tmp_path / "coarse_raw.npy"
    np.save(raw_path, volume)
    baseline = np.zeros(shape, dtype=np.uint8)
    baseline[plane_z - 1 : plane_z + 2] = 255
    baseline_path = tmp_path / "coarse_baseline.npy"
    np.save(baseline_path, baseline)
    return raw_path, baseline_path


def make_student_patch(
    directory: Path,
    *,
    patch_id: str,
    shape: tuple[int, int, int] = (64, 64, 64),
    plane_z: int = 10,
    with_distill: bool = True,
    seed: int = 3,
) -> None:
    rng = np.random.default_rng(seed)
    image = rng.integers(20, 90, size=shape, dtype=np.int64).astype(np.uint8)
    image[plane_z - 1 : plane_z + 2] = 170
    label = np.full(shape, 2, dtype=np.uint8)
    label[plane_z - 1 : plane_z + 2] = 1
    label[plane_z - 3 : plane_z - 1] = 0
    label[plane_z + 2 : plane_z + 4] = 0
    arrays: dict[str, np.ndarray] = {"image": image, "label_u8": label}
    if with_distill:
        distill = np.zeros(shape, dtype=np.uint8)
        distill[plane_z - 1 : plane_z + 2] = 255
        valid = np.zeros(shape, dtype=np.uint8)
        valid[2:-2] = 1
        arrays["distill_u8"] = distill
        arrays["distill_valid_u8"] = valid
    np.savez(directory / f"{patch_id}.npz", **arrays)


def write_student_manifest(
    directory: Path,
    *,
    rows: list[dict[str, object]],
) -> Path:
    manifest = directory / "patches.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


def student_row(
    patch_id: str,
    *,
    scroll_id: str,
    split: str,
    kind: str = "student",
    shape: tuple[int, int, int] = (64, 64, 64),
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "patch_id": patch_id,
        "path": f"{patch_id}.npz",
        "record_id": f"record-{scroll_id.lower()}",
        "scroll_id": scroll_id,
        "split": split,
        "kind": kind,
        "origin_zyx": [0, 0, 0],
        "shape_zyx": list(shape),
        "policy_profile": "research",
        "pitch_um": COARSE_UM,
        "sampling_stratum": "distill",
    }
