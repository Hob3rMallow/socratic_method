"""Cross-resolution ground truth for the 2D prediction gap fixup.

The C tool ``pred_fixup`` proposes joins between skeleton-endpoint pairs in
9.362 um (L0) prediction planes. Each join is a HYPOTHESIS: "these two
fragments are the same sheet, and the gap between them is a prediction
failure". This module tests that hypothesis against the co-registered
2.399 um CT: map the join's neighbourhood through the official affine, fetch
exactly the fine chunks it needs from the public S3 store (into a private
sparse store — the official carve mirror is NEVER extended: its plan hash
gates the in-flight 250k reconciler), and ask whether the two endpoint
regions are connected by bright material inside a corridor around the join.

Verdicts:
  CONNECTED  the fine CT shows continuous material between the endpoints
             inside the corridor — the join is right, the gap was a
             prediction failure.
  SEPARATE   the fine CT shows a real discontinuity — the join bridged a
             genuine tear (or, if it also moved radially, crossed wraps).
  NO_DATA    a needed fine chunk is not in the store (fetch failed/skipped).
  AMBIGUOUS  mask-edge window (nonzero fraction below the project's 0.95
             gate) or an endpoint lands on no material at fine scale.

The corridor restriction matters: two wraps can physically TOUCH inside a
window, so whole-window connectivity would over-report CONNECTED. The
corridor is the fine-frame analog of the fixup's own local evidence flood.

Read-only with respect to every other crossres asset. CPU only.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import ndimage

from ..resample import affine_scale_ratio, invert_affine_xyz
from .io import open_volume, read_crop

SCHEMA = "crossres-gap-join-validation-v1"

DEFAULT_SOURCE_URI = (
    "s3://vesuvius-challenge-open-data/PHerc0139/volumes/"
    "20260102150214-2.399um-0.2m-78keV-masked.zarr"
)
DEFAULT_LOCAL_MIRROR = Path(
    "D:/work/vesuvius-c/PHerc0139-full/20260102150214-2.399um-0.2m-78keV-masked.zarr"
)
DEFAULT_TRANSFORM = DEFAULT_LOCAL_MIRROR / "transform.json"
DEFAULT_COARSE_RAW = Path(
    "D:/work/vesuvius-c/PHerc0139-full/20250728140407-9.362um-1.2m-113keV-masked.zarr"
)

FINE_CHUNK = 128
NONZERO_MIN_FRACTION = 0.95  # the project's mask-edge gate for fine CT


# ----------------------------------------------------------------------------
# rows


@dataclass(frozen=True)
class JoinRow:
    """One pair to judge, in world L0 voxel coordinates (z is the plane)."""

    z: int
    ay: int
    ax: int
    by: int
    bx: int
    kind: str = "join"  # "join" | "reject:<reason>" | "control_crosswrap"
    dist: float = 0.0
    support: int = 0
    far_tier: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def row_id(self) -> str:
        return f"{self.kind}_z{self.z}_a{self.ay}_{self.ax}_b{self.by}_{self.bx}"


def load_joins(path: str | Path, *, kept_only: bool = True) -> list[JoinRow]:
    rows: list[JoinRow] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if kept_only and not obj.get("kept", 0):
            continue
        rows.append(
            JoinRow(
                z=int(obj["z"]),
                ay=int(obj["a"]["y"]),
                ax=int(obj["a"]["x"]),
                by=int(obj["b"]["y"]),
                bx=int(obj["b"]["x"]),
                kind="join",
                dist=float(obj.get("dist", 0.0)),
                support=int(obj.get("support", 0)),
                far_tier=int(obj.get("far_tier", 0)),
            )
        )
    return rows


def load_rejects(
    path: str | Path,
    *,
    reasons: Iterable[str] = ("evidence", "tangent"),
    limit: int = 40,
) -> list[JoinRow]:
    wanted = set(reasons)
    rows: list[JoinRow] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if len(rows) >= limit:
            break
        if not line.strip():
            continue
        obj = json.loads(line)
        reason = obj.get("reason", "")
        if reason not in wanted:
            continue
        rows.append(
            JoinRow(
                z=int(obj["z"]),
                ay=int(obj["a"]["y"]),
                ax=int(obj["a"]["x"]),
                by=int(obj["b"]["y"]),
                bx=int(obj["b"]["x"]),
                kind=f"reject:{reason}",
            )
        )
    return rows


def make_crosswrap_controls(
    joins: list[JoinRow],
    *,
    umb_y: float,
    umb_x: float,
    pitch: float = 9.5,
    limit: int = 30,
) -> list[JoinRow]:
    """Deliberate cross-wrap pairs: endpoint b pushed one pitch radially.

    The validator MUST read these as SEPARATE (or at worst AMBIGUOUS) — they
    calibrate its ability to see wrap separation at fine scale.
    """

    controls: list[JoinRow] = []
    for row in joins:
        if len(controls) >= limit:
            break
        ry = float(row.by) - umb_y
        rx = float(row.bx) - umb_x
        norm = float(np.hypot(ry, rx))
        if norm < 1.0:
            continue
        controls.append(
            JoinRow(
                z=row.z,
                ay=row.ay,
                ax=row.ax,
                by=int(round(row.by + pitch * ry / norm)),
                bx=int(round(row.bx + pitch * rx / norm)),
                kind="control_crosswrap",
                meta={"from": row.row_id},
            )
        )
    return controls


# ----------------------------------------------------------------------------
# affine + planning


def load_fine_to_coarse_affine(path: str | Path = DEFAULT_TRANSFORM) -> np.ndarray:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    matrix = np.asarray(obj["transformation_matrix"], dtype=np.float64)
    if matrix.shape != (3, 4):
        raise ValueError(f"{path}: transformation_matrix must be 3x4")
    return matrix


def coarse_to_fine_zyx(points_zyx: np.ndarray, affine_xyz: np.ndarray) -> np.ndarray:
    """Map coarse z-y-x points to fine z-y-x through the fine->coarse affine."""

    linear_inverse, translation = invert_affine_xyz(affine_xyz)
    pts = np.atleast_2d(np.asarray(points_zyx, dtype=np.float64))
    fine_xyz = (pts[:, ::-1] - translation) @ linear_inverse.T
    return fine_xyz[:, ::-1]


def coarse_box_for_row(
    row: JoinRow, *, pad_zyx: tuple[int, int, int] = (3, 6, 6)
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    y0 = min(row.ay, row.by) - pad_zyx[1]
    y1 = max(row.ay, row.by) + pad_zyx[1] + 1
    x0 = min(row.ax, row.bx) - pad_zyx[2]
    x1 = max(row.ax, row.bx) + pad_zyx[2] + 1
    z0 = row.z - pad_zyx[0]
    z1 = row.z + pad_zyx[0] + 1
    return (z0, y0, x0), (z1 - z0, y1 - y0, x1 - x0)


def fine_window_for_row(
    row: JoinRow,
    affine_xyz: np.ndarray,
    *,
    fine_shape: tuple[int, int, int],
    pad_zyx: tuple[int, int, int] = (3, 6, 6),
    margin_fine_vox: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Clamped integer fine z-y-x window (lo, hi) for the row's coarse box."""

    from ..resample import fine_bbox_for_coarse_box

    origin, shape = coarse_box_for_row(row, pad_zyx=pad_zyx)
    lo, hi = fine_bbox_for_coarse_box(
        origin, shape, affine_xyz, margin_fine_vox=margin_fine_vox
    )
    lo = np.maximum(np.floor(lo).astype(np.int64), 0)
    hi = np.minimum(np.ceil(hi).astype(np.int64), np.asarray(fine_shape))
    return lo, hi


def chunks_for_window(
    lo: np.ndarray, hi: np.ndarray, *, chunk: int = FINE_CHUNK
) -> set[tuple[int, int, int]]:
    out: set[tuple[int, int, int]] = set()
    c0 = lo // chunk
    c1 = (np.maximum(hi, lo + 1) - 1) // chunk
    for cz in range(int(c0[0]), int(c1[0]) + 1):
        for cy in range(int(c0[1]), int(c1[1]) + 1):
            for cx in range(int(c0[2]), int(c1[2]) + 1):
                out.add((cz, cy, cx))
    return out


def plan_rows(
    rows: list[JoinRow],
    affine_xyz: np.ndarray,
    *,
    fine_shape: tuple[int, int, int],
    chunk: int = FINE_CHUNK,
) -> dict[str, Any]:
    all_chunks: set[tuple[int, int, int]] = set()
    windows: list[tuple[np.ndarray, np.ndarray]] = []
    for row in rows:
        lo, hi = fine_window_for_row(row, affine_xyz, fine_shape=fine_shape)
        windows.append((lo, hi))
        all_chunks |= chunks_for_window(lo, hi, chunk=chunk)
    return {
        "schema": SCHEMA,
        "rows": len(rows),
        "chunks": sorted(all_chunks),
        "chunk_bytes": chunk**3,
        "total_mib": len(all_chunks) * chunk**3 / 2**20,
        "windows": windows,
    }


# ----------------------------------------------------------------------------
# fetch (private sparse store; the official carve is read-only)


def ensure_store_metadata(store_dir: Path, local_mirror: Path | None) -> None:
    array_dir = store_dir / "0"
    array_dir.mkdir(parents=True, exist_ok=True)
    if not (array_dir / ".zarray").exists():
        if local_mirror is None or not (local_mirror / "0" / ".zarray").exists():
            raise FileNotFoundError(
                "no .zarray template — pass a local mirror with 0/.zarray"
            )
        shutil.copy2(local_mirror / "0" / ".zarray", array_dir / ".zarray")
    zgroup = store_dir / ".zgroup"
    if not zgroup.exists():
        zgroup.write_text('{"zarr_format": 2}\n', encoding="utf-8")


def fetch_chunks(
    chunks: Iterable[tuple[int, int, int]],
    *,
    store_dir: str | Path,
    source_uri: str = DEFAULT_SOURCE_URI,
    local_mirror: str | Path | None = DEFAULT_LOCAL_MIRROR,
    workers: int = 8,
    allow_network: bool = True,
) -> dict[str, int]:
    """Materialize chunks into store_dir/0/cz/cy/cx (dimension_separator '/').

    Order of preference: already present, copy from the local carve mirror,
    anonymous S3 download. Never writes anywhere else.
    """

    store = Path(store_dir)
    mirror = Path(local_mirror) if local_mirror else None
    ensure_store_metadata(store, mirror)

    todo: list[tuple[int, int, int]] = []
    stats = {"present": 0, "copied_local": 0, "fetched": 0, "failed": 0}
    for cz, cy, cx in chunks:
        dst = store / "0" / str(cz) / str(cy) / str(cx)
        if dst.exists():
            stats["present"] += 1
            continue
        if mirror is not None:
            src = mirror / "0" / str(cz) / str(cy) / str(cx)
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                stats["copied_local"] += 1
                continue
        todo.append((cz, cy, cx))

    if todo and allow_network:
        import s3fs
        from concurrent.futures import ThreadPoolExecutor

        remote_root = source_uri.removeprefix("s3://").rstrip("/")
        fs = s3fs.S3FileSystem(anon=True)

        def grab(coord: tuple[int, int, int]) -> bool:
            cz, cy, cx = coord
            key = f"{remote_root}/0/{cz}/{cy}/{cx}"
            dst = store / "0" / str(cz) / str(cy) / str(cx)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".part")
            for _ in range(3):
                try:
                    fs.get(key, str(tmp))
                    tmp.replace(dst)
                    return True
                except Exception:  # noqa: BLE001 - retry transient public S3
                    if tmp.exists():
                        tmp.unlink()
            return False

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for ok in pool.map(grab, todo):
                stats["fetched" if ok else "failed"] += 1
    else:
        stats["failed"] += len(todo)
    return stats


# ----------------------------------------------------------------------------
# verdicts


def _otsu_threshold(values: np.ndarray) -> int:
    hist = np.bincount(values.reshape(-1), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 128
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(256)) / total
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom <= 0] = np.nan
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    if np.all(np.isnan(sigma_b)):
        return 128
    return int(np.nanargmax(sigma_b))


def _segment_distance_mask(
    shape: tuple[int, int, int],
    a_zyx: np.ndarray,
    b_zyx: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Boolean corridor: voxels within `radius` of segment a-b (local coords)."""

    grid = np.stack(
        np.meshgrid(
            np.arange(shape[0], dtype=np.float32),
            np.arange(shape[1], dtype=np.float32),
            np.arange(shape[2], dtype=np.float32),
            indexing="ij",
        ),
        axis=-1,
    )
    a = a_zyx.astype(np.float32)
    d = (b_zyx - a_zyx).astype(np.float32)
    dd = float(np.dot(d, d))
    if dd < 1.0e-6:
        dist = np.linalg.norm(grid - a, axis=-1)
    else:
        t = np.clip(((grid - a) @ d) / dd, 0.0, 1.0)
        nearest = a + t[..., None] * d
        dist = np.linalg.norm(grid - nearest, axis=-1)
    return dist <= radius


@dataclass(frozen=True)
class JudgeConfig:
    """corridor_radius_fine must stay BELOW the distance from a wrap's
    centerline to the adjacent wrap's near face — at 2.399 um that is
    pitch*scale - thickness/2 ~= 37 - 17 ~= 20 fine vox. The first
    calibration run used 22 and the cross-wrap controls read CONNECTED
    (the corridor admitted the neighbouring wrap's face). 14 keeps the
    neighbour out while containing the own-band connection path.
    seed_radius_fine absorbs the ~1-2 coarse vox offset between the L0
    prediction band and the actual fine material."""

    seed_radius_fine: float = 10.0
    corridor_radius_fine: float = 14.0
    nonzero_min: float = NONZERO_MIN_FRACTION
    connectivity: int = 26


def judge_row(
    fine_vol: Any,
    row: JoinRow,
    affine_xyz: np.ndarray,
    *,
    store_dir: Path | None = None,
    cfg: JudgeConfig = JudgeConfig(),
    fine_shape: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    shape = tuple(int(s) for s in (fine_shape or fine_vol.shape))
    lo, hi = fine_window_for_row(row, affine_xyz, fine_shape=shape)
    size = hi - lo
    result: dict[str, Any] = {
        "id": row.row_id,
        "kind": row.kind,
        "z": row.z,
        "a": [row.ay, row.ax],
        "b": [row.by, row.bx],
        "dist": row.dist,
        "support": row.support,
        "far_tier": row.far_tier,
        "fine_lo": [int(v) for v in lo],
        "fine_hi": [int(v) for v in hi],
    }
    if np.any(size <= 0):
        result["verdict"] = "NO_DATA"
        result["why"] = "window outside fine volume"
        return result

    if store_dir is not None:
        missing = [
            c
            for c in chunks_for_window(lo, hi)
            if not (store_dir / "0" / str(c[0]) / str(c[1]) / str(c[2])).exists()
        ]
        if missing:
            result["verdict"] = "NO_DATA"
            result["why"] = f"{len(missing)} chunks not in store"
            return result

    window = read_crop(fine_vol, tuple(int(v) for v in lo), tuple(int(v) for v in size))
    nonzero_fraction = float(np.count_nonzero(window)) / float(window.size)
    result["nonzero_fraction"] = round(nonzero_fraction, 4)
    if nonzero_fraction < cfg.nonzero_min:
        result["verdict"] = "AMBIGUOUS"
        result["why"] = "mask-edge window (nonzero fraction below gate)"
        return result

    threshold = _otsu_threshold(window[window > 0])
    result["threshold"] = int(threshold)
    binary = window > threshold

    pts = np.array(
        [[row.z, row.ay, row.ax], [row.z, row.by, row.bx]], dtype=np.float64
    )
    fine_pts = coarse_to_fine_zyx(pts, affine_xyz) - lo[None, :]
    result["fine_a"] = [round(float(v), 1) for v in fine_pts[0]]
    result["fine_b"] = [round(float(v), 1) for v in fine_pts[1]]

    corridor = _segment_distance_mask(
        binary.shape, fine_pts[0], fine_pts[1], cfg.corridor_radius_fine
    )
    material = binary & corridor

    structure = ndimage.generate_binary_structure(3, 3 if cfg.connectivity == 26 else 1)
    labels, _ = ndimage.label(material, structure=structure)

    def seed_labels(p: np.ndarray) -> set[int]:
        ball = _segment_distance_mask(binary.shape, p, p, cfg.seed_radius_fine)
        found = np.unique(labels[ball & material])
        return {int(v) for v in found if v != 0}

    la = seed_labels(fine_pts[0])
    lb = seed_labels(fine_pts[1])
    result["seed_a_labels"] = len(la)
    result["seed_b_labels"] = len(lb)
    if not la or not lb:
        result["verdict"] = "AMBIGUOUS"
        result["why"] = "an endpoint lands on no fine material"
        return result

    result["verdict"] = "CONNECTED" if (la & lb) else "SEPARATE"
    return result


# ----------------------------------------------------------------------------
# rendering + report


def render_row_png(
    path: str | Path,
    row: JoinRow,
    verdict: dict[str, Any],
    fine_vol: Any,
    affine_xyz: np.ndarray,
    *,
    coarse_vol: Any | None = None,
    coarse_origin_zyx: tuple[int, int, int] = (0, 0, 0),
    half_coarse: int = 24,
    scale_coarse: int = 4,
) -> None:
    """Side-by-side: L0 raw crop with the join line | fine mid-slice with
    endpoint markers and the verdict."""

    from PIL import Image, ImageDraw

    panels: list[Image.Image] = []

    if coarse_vol is not None:
        cy = (row.ay + row.by) // 2 - coarse_origin_zyx[1]
        cx = (row.ax + row.bx) // 2 - coarse_origin_zyx[2]
        cz = row.z - coarse_origin_zyx[0]
        crop = read_crop(
            coarse_vol,
            (cz, cy - half_coarse, cx - half_coarse),
            (1, 2 * half_coarse, 2 * half_coarse),
        )[0]
        lo_v, hi_v = np.percentile(crop[crop > 0], [2, 98]) if np.any(crop > 0) else (0, 1)
        norm = np.clip((crop.astype(np.float32) - lo_v) / max(1.0, hi_v - lo_v), 0, 1)
        img = Image.fromarray((norm * 255).astype(np.uint8), "L").convert("RGB")
        img = img.resize(
            (img.width * scale_coarse, img.height * scale_coarse), Image.NEAREST
        )
        draw = ImageDraw.Draw(img)
        ay = (row.ay - coarse_origin_zyx[1] - (cy - half_coarse)) * scale_coarse
        ax = (row.ax - coarse_origin_zyx[2] - (cx - half_coarse)) * scale_coarse
        by = (row.by - coarse_origin_zyx[1] - (cy - half_coarse)) * scale_coarse
        bx = (row.bx - coarse_origin_zyx[2] - (cx - half_coarse)) * scale_coarse
        draw.line([(ax, ay), (bx, by)], fill=(255, 60, 60), width=2)
        panels.append(img)

    lo = np.asarray(verdict.get("fine_lo", [0, 0, 0]))
    hi = np.asarray(verdict.get("fine_hi", [1, 1, 1]))
    size = np.maximum(hi - lo, 1)
    window = read_crop(fine_vol, tuple(int(v) for v in lo), tuple(int(v) for v in size))
    fa = np.asarray(verdict.get("fine_a", [size[0] / 2, 0, 0]))
    fb = np.asarray(verdict.get("fine_b", [size[0] / 2, 0, 0]))
    mid = int(np.clip(round((fa[0] + fb[0]) / 2), 0, size[0] - 1))
    sl = window[mid]
    lo_v, hi_v = np.percentile(sl[sl > 0], [2, 98]) if np.any(sl > 0) else (0, 1)
    norm = np.clip((sl.astype(np.float32) - lo_v) / max(1.0, hi_v - lo_v), 0, 1)
    img = Image.fromarray((norm * 255).astype(np.uint8), "L").convert("RGB")
    draw = ImageDraw.Draw(img)
    for p, color in ((fa, (255, 80, 80)), (fb, (255, 200, 60))):
        y, x = float(p[1]), float(p[2])
        draw.line([(x - 6, y), (x + 6, y)], fill=color, width=2)
        draw.line([(x, y - 6), (x, y + 6)], fill=color, width=2)
    draw.text((4, 4), verdict.get("verdict", "?"), fill=(120, 255, 120))
    panels.append(img)

    height = max(p.height for p in panels)
    width = sum(p.width for p in panels) + 6 * (len(panels) - 1)
    canvas = Image.new("RGB", (width, height), (30, 30, 30))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + 6
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def summarize(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    def bucket(rows: list[dict[str, Any]]) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in rows:
            out[v["verdict"]] = out.get(v["verdict"], 0) + 1
        return out

    kinds = sorted({v["kind"] for v in verdicts})
    by_kind = {
        k: bucket([v for v in verdicts if v["kind"] == k]) for k in kinds
    }
    joins = [v for v in verdicts if v["kind"] == "join"]
    decided = [v for v in joins if v["verdict"] in ("CONNECTED", "SEPARATE")]
    connected = sum(1 for v in decided if v["verdict"] == "CONNECTED")
    near = [v for v in decided if float(v.get("dist", 0.0)) <= 6.0]
    far = [v for v in decided if float(v.get("dist", 0.0)) > 6.0]
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "by_kind": by_kind,
        "join_precision": round(connected / len(decided), 4) if decided else None,
        "join_decided": len(decided),
        "join_connected": connected,
        "tier_safe": {
            "decided": len(near),
            "connected": sum(1 for v in near if v["verdict"] == "CONNECTED"),
        },
        "tier_far": {
            "decided": len(far),
            "connected": sum(1 for v in far if v["verdict"] == "CONNECTED"),
        },
    }
    return summary
