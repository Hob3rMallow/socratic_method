from __future__ import annotations

import json

import numpy as np
import pytest

from crossres_pred.voxel import gap_join_validation as gjv

# fine->coarse quarter affine: the fine lattice is 4x finer, no rotation.
QUARTER = np.array(
    [[0.25, 0.0, 0.0, 0.0], [0.0, 0.25, 0.0, 0.0], [0.0, 0.0, 0.25, 0.0]],
    dtype=np.float64,
)

FINE_SHAPE = (64, 256, 256)


def _row(z: int, ay: int, ax: int, by: int, bx: int, kind: str = "join") -> gjv.JoinRow:
    return gjv.JoinRow(z=z, ay=ay, ax=ax, by=by, bx=bx, kind=kind)


def test_coarse_to_fine_round_trip() -> None:
    pts = np.array([[8.0, 26.0, 10.0], [1.0, 2.0, 3.0]])
    fine = gjv.coarse_to_fine_zyx(pts, QUARTER)
    assert np.allclose(fine, pts * 4.0)


def test_chunks_for_window_covers_and_dedupes() -> None:
    lo = np.array([120, 0, 250])
    hi = np.array([130, 5, 260])
    chunks = gjv.chunks_for_window(lo, hi, chunk=128)
    assert chunks == {(0, 0, 1), (1, 0, 1), (0, 0, 2), (1, 0, 2)}

    rows = [_row(8, 26, 10, 26, 14), _row(8, 26, 12, 26, 16)]
    plan = gjv.plan_rows(rows, QUARTER, fine_shape=FINE_SHAPE)
    # neighbouring joins share their fine chunks
    assert plan["rows"] == 2
    assert len(plan["chunks"]) >= 1
    assert plan["total_mib"] == len(plan["chunks"]) * 128**3 / 2**20


def _fine_volume(gap: bool = False, bypass: bool = False) -> np.ndarray:
    """Dark tissue everywhere (value 60), one bright band (200) along x at
    fine y~100..112, z 24..40. Coarse endpoints at y=26 (fine 104) sit on
    the band's centerline."""

    vol = np.full(FINE_SHAPE, 60, dtype=np.uint8)
    vol[24:40, 100:113, 20:230] = 200
    if gap:
        vol[:, :, 110:131] = np.where(
            vol[:, :, 110:131] == 200, 60, vol[:, :, 110:131]
        ).astype(np.uint8)
    if bypass:
        # a far detour: band -> y=180 rail -> band, entirely outside the
        # corridor around the straight a-b segment
        vol[24:40, 100:181, 100:106] = 200
        vol[24:40, 175:181, 100:150] = 200
        vol[24:40, 100:181, 144:150] = 200
    return vol


def test_judge_connected() -> None:
    vol = _fine_volume()
    verdict = gjv.judge_row(vol, _row(8, 26, 10, 26, 50), QUARTER,
                            fine_shape=FINE_SHAPE)
    assert verdict["verdict"] == "CONNECTED"


def test_judge_separate_on_gap() -> None:
    vol = _fine_volume(gap=True)
    verdict = gjv.judge_row(vol, _row(8, 26, 25, 26, 35), QUARTER,
                            fine_shape=FINE_SHAPE)
    assert verdict["verdict"] == "SEPARATE"


def test_corridor_blocks_far_detours() -> None:
    # the two sides ARE connected through the detour, but not inside the
    # corridor around the join segment -> still SEPARATE
    vol = _fine_volume(gap=True, bypass=True)
    verdict = gjv.judge_row(vol, _row(8, 26, 25, 26, 35), QUARTER,
                            fine_shape=FINE_SHAPE)
    assert verdict["verdict"] == "SEPARATE"


def test_mask_edge_window_is_ambiguous() -> None:
    vol = _fine_volume()
    vol[:, :, :128] = 0  # masked-out half
    verdict = gjv.judge_row(vol, _row(8, 26, 25, 26, 35), QUARTER,
                            fine_shape=FINE_SHAPE)
    assert verdict["verdict"] == "AMBIGUOUS"


def test_endpoint_off_material_is_ambiguous() -> None:
    vol = _fine_volume()
    verdict = gjv.judge_row(vol, _row(8, 50, 25, 50, 35), QUARTER,
                            fine_shape=FINE_SHAPE)
    assert verdict["verdict"] == "AMBIGUOUS"


def test_crosswrap_controls_shift_one_pitch() -> None:
    rows = [_row(8, 26, 100, 26, 110)]
    controls = gjv.make_crosswrap_controls(
        rows, umb_y=26.0, umb_x=0.0, pitch=9.5, limit=10
    )
    assert len(controls) == 1
    c = controls[0]
    assert c.kind == "control_crosswrap"
    assert (c.by, c.bx) == (26, 120)  # pushed radially away from the umbilicus


def test_fetch_local_mirror_only(tmp_path) -> None:
    mirror = tmp_path / "mirror"
    (mirror / "0" / "3" / "4").mkdir(parents=True)
    zarray = {
        "shape": list(FINE_SHAPE), "chunks": [128, 128, 128], "dtype": "|u1",
        "fill_value": 0, "order": "C", "filters": None,
        "dimension_separator": "/", "compressor": None, "zarr_format": 2,
    }
    (mirror / "0" / ".zarray").write_text(json.dumps(zarray), encoding="utf-8")
    (mirror / "0" / "3" / "4" / "5").write_bytes(b"\x01" * 16)

    store = tmp_path / "store"
    stats = gjv.fetch_chunks(
        [(3, 4, 5), (9, 9, 9)], store_dir=store, local_mirror=mirror,
        allow_network=False,
    )
    assert stats == {"present": 0, "copied_local": 1, "fetched": 0, "failed": 1}
    assert (store / "0" / "3" / "4" / "5").read_bytes() == b"\x01" * 16
    assert (store / "0" / ".zarray").exists()

    # second call: chunk already present
    stats2 = gjv.fetch_chunks(
        [(3, 4, 5)], store_dir=store, local_mirror=mirror, allow_network=False
    )
    assert stats2["present"] == 1


def test_summarize_precision_and_tiers() -> None:
    verdicts = [
        {"kind": "join", "verdict": "CONNECTED", "dist": 4.0},
        {"kind": "join", "verdict": "CONNECTED", "dist": 8.0},
        {"kind": "join", "verdict": "SEPARATE", "dist": 9.0},
        {"kind": "join", "verdict": "NO_DATA", "dist": 5.0},
        {"kind": "control_crosswrap", "verdict": "SEPARATE"},
    ]
    summary = gjv.summarize(verdicts)
    assert summary["join_decided"] == 3
    assert summary["join_precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["tier_safe"] == {"decided": 1, "connected": 1}
    assert summary["tier_far"] == {"decided": 2, "connected": 1}
    assert summary["by_kind"]["control_crosswrap"]["SEPARATE"] == 1
