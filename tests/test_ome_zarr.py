from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from socratic_method import ome_zarr

pytest.importorskip("zarr")
pytest.importorskip("numcodecs")

import zarr

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 8

#: Level shapes of the published PHerc1447 and PHerc0139 M7 surface predictions.
PUBLISHED_SHAPES = {
    (24297, 8343, 8343): [
        (24297, 8343, 8343),
        (12149, 4172, 4172),
        (6075, 2086, 2086),
        (3038, 1043, 1043),
        (1519, 522, 522),
        (760, 261, 261),
    ],
    (20974, 6621, 6621): [
        (20974, 6621, 6621),
        (10487, 3311, 3311),
        (5244, 1656, 1656),
        (2622, 828, 828),
        (1311, 414, 414),
        (656, 207, 207),
    ],
}

#: ``0/.zarray`` of the published stores, minus ``shape`` and ``chunks``.
PUBLISHED_ARRAY_METADATA = {
    "dtype": "|u1",
    "fill_value": 0,
    "filters": None,
    "order": "C",
    "zarr_format": 2,
    "dimension_separator": "/",
    "compressor": {"id": "blosc", "cname": "zstd", "clevel": 1, "shuffle": 1, "blocksize": 0},
}


def _cubes(ids, *, seed=5, chunk=CHUNK):
    rng = np.random.default_rng(seed)
    return {
        cube_id: (rng.random((chunk,) * 3) > 0.6).astype(np.uint8) * ome_zarr.FOREGROUND
        for cube_id in ids
    }


def _write(tmp_path, ids, *, shape=(40, 40, 40), levels=3, chunk=CHUNK, cubes=None, **kw):
    cubes = cubes if cubes is not None else _cubes(ids, chunk=chunk)
    summary = ome_zarr.write_surface_pyramid(
        tmp_path / "prediction.zarr",
        cube_ids=list(ids),
        read_cube=cubes.__getitem__,
        shape_zyx=shape,
        chunk=chunk,
        levels=levels,
        **kw,
    )
    return summary, cubes


def test_level_shape_matches_the_published_pyramids() -> None:
    for shape, levels in PUBLISHED_SHAPES.items():
        for level, expected in enumerate(levels):
            assert ome_zarr.level_shape(shape, level) == expected, (shape, level)


def test_multiscale_attrs_match_the_published_metadata() -> None:
    attrs = ome_zarr.multiscale_attrs(6)
    assert list(attrs) == ["multiscales"]
    (multiscale,) = attrs["multiscales"]
    assert multiscale["version"] == "0.4"
    assert multiscale["name"] == "/"
    assert multiscale["axes"] == [
        {"name": "z", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]
    assert [dataset["path"] for dataset in multiscale["datasets"]] == list("012345")
    assert multiscale["datasets"][0]["coordinateTransformations"] == [
        {"scale": [1.0, 1.0, 1.0], "type": "scale"}
    ]
    assert multiscale["datasets"][5]["coordinateTransformations"] == [
        {"scale": [32.0, 32.0, 32.0], "type": "scale"}
    ]


def test_store_metadata_matches_the_published_stores(tmp_path: Path) -> None:
    summary, _ = _write(tmp_path, ["z00008_y00008_x00008"])
    store = tmp_path / "prediction.zarr"
    assert json.loads((store / ".zgroup").read_text(encoding="utf-8")) == {"zarr_format": 2}
    for level in range(summary["levels"]):
        metadata = json.loads((store / str(level) / ".zarray").read_text(encoding="utf-8"))
        for key, value in PUBLISHED_ARRAY_METADATA.items():
            assert metadata[key] == value, (level, key)
        assert metadata["chunks"] == [CHUNK] * 3
        assert metadata["shape"] == list(ome_zarr.level_shape((40, 40, 40), level))


def test_validate_ome_ngff_accepts_what_we_write(tmp_path: Path) -> None:
    _write(tmp_path, ["z00008_y00008_x00008"])
    report = ome_zarr.validate_ome_ngff(tmp_path / "prediction.zarr")
    assert report == {
        "levels": 3,
        "shape_zyx": [40, 40, 40],
        "chunks_zyx": [CHUNK] * 3,
        "dtype": "|u1",
        "compressor": PUBLISHED_ARRAY_METADATA["compressor"],
        "version": "0.4",
    }


def test_validate_ome_ngff_refuses_a_non_identity_level_zero(tmp_path: Path) -> None:
    _write(tmp_path, ["z00008_y00008_x00008"])
    store = tmp_path / "prediction.zarr"
    attrs = json.loads((store / ".zattrs").read_text(encoding="utf-8"))
    attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"] = [
        {"scale": [2.0, 2.0, 2.0], "type": "scale"}
    ]
    (store / ".zattrs").write_text(json.dumps(attrs), encoding="utf-8")
    with pytest.raises(ome_zarr.OmeZarrError, match="identity scale"):
        ome_zarr.validate_ome_ngff(store)


def test_cubes_land_at_their_global_coordinates(tmp_path: Path) -> None:
    ids = ["z00008_y00008_x00008", "z00008_y00008_x00016", "z00024_y00016_x00008"]
    _, cubes = _write(tmp_path, ids)
    level0 = np.asarray(zarr.open(str(tmp_path / "prediction.zarr"), mode="r")["0"][:])
    assert set(np.unique(level0)) <= {0, 255}
    for cube_id, mask in cubes.items():
        z, y, x = (int(piece[1:]) for piece in cube_id.split("_"))
        assert np.array_equal(level0[z : z + CHUNK, y : y + CHUNK, x : x + CHUNK], mask), cube_id


def test_downsampling_is_exactly_nearest(tmp_path: Path) -> None:
    ids = ["z00008_y00008_x00008", "z00016_y00016_x00016", "z00024_y00024_x00024"]
    _write(tmp_path, ids, levels=4)
    root = zarr.open(str(tmp_path / "prediction.zarr"), mode="r")
    level0 = np.asarray(root["0"][:])
    for level in range(1, 4):
        got = np.asarray(root[str(level)][:])
        assert got.shape == ome_zarr.level_shape((40, 40, 40), level)
        grid = np.meshgrid(*[np.arange(size) for size in got.shape], indexing="ij")
        assert np.array_equal(got, level0[tuple(axis << level for axis in grid)]), level


def test_untouched_and_empty_chunks_are_never_written(tmp_path: Path) -> None:
    ids = ["z00008_y00008_x00008", "z00016_y00016_x00016"]
    cubes = _cubes(ids)
    cubes["z00016_y00016_x00016"] = np.zeros((CHUNK,) * 3, dtype=np.uint8)
    summary, _ = _write(tmp_path, ids, cubes=cubes)
    assert summary["chunks_written"]["0"] == 1  # the all-zero cube is left absent
    store = tmp_path / "prediction.zarr"
    assert not (store / "0" / "2" / "2" / "2").exists()
    level0 = np.asarray(zarr.open(str(store), mode="r")["0"][:])
    assert not level0[16:24, 16:24, 16:24].any()  # and reads back as the fill value


def test_a_sparse_store_spans_the_whole_volume(tmp_path: Path) -> None:
    """One cube of a 24297-slice scroll: global coordinates, a handful of chunks."""

    summary, cubes = _write(
        tmp_path, ["z12160_y04224_x02944"], shape=(24297, 8343, 8343), levels=6, chunk=128
    )
    assert summary["shape_zyx"] == [24297, 8343, 8343]
    assert sum(summary["chunks_written"].values()) <= 6
    root = zarr.open(str(tmp_path / "prediction.zarr"), mode="r")
    assert root["0"].shape == (24297, 8343, 8343)
    block = np.asarray(root["0"][12160:12288, 4224:4352, 2944:3072])
    assert np.array_equal(block, cubes["z12160_y04224_x02944"])


def test_refusals(tmp_path: Path) -> None:
    with pytest.raises(ome_zarr.OmeZarrError, match="no cubes"):
        _write(tmp_path, [])
    with pytest.raises(ome_zarr.OmeZarrError, match="not aligned"):
        _write(tmp_path, ["z00003_y00008_x00008"])
    with pytest.raises(ome_zarr.OmeZarrError, match="past the volume shape"):
        _write(tmp_path, ["z00040_y00008_x00008"])
    with pytest.raises(ome_zarr.OmeZarrError, match="invalid cube ID"):
        _write(tmp_path, ["not-a-cube"])
    _write(tmp_path, ["z00008_y00008_x00008"])
    with pytest.raises(ome_zarr.OmeZarrError, match="already exists"):
        _write(tmp_path, ["z00008_y00008_x00008"])


def test_a_failed_write_leaves_nothing_behind(tmp_path: Path) -> None:
    def explode(cube_id: str) -> np.ndarray:
        raise RuntimeError("cube unreadable")

    with pytest.raises(RuntimeError, match="unreadable"):
        ome_zarr.write_surface_pyramid(
            tmp_path / "prediction.zarr",
            cube_ids=["z00008_y00008_x00008"],
            read_cube=explode,
            shape_zyx=(40, 40, 40),
            chunk=CHUNK,
            levels=2,
        )
    assert sorted(tmp_path.iterdir()) == []


def test_metadata_sidecar_is_written(tmp_path: Path) -> None:
    _write(tmp_path, ["z00008_y00008_x00008"], metadata={"kind": "surface-inference"})
    sidecar = tmp_path / "prediction.zarr" / "metadata.json"
    assert json.loads(sidecar.read_text(encoding="utf-8"))["kind"] == "surface-inference"


def test_recipe_deliverable_pins_match_this_writer() -> None:
    recipe = json.loads((ROOT / "recipes" / "f0" / "recipe.json").read_text(encoding="utf-8"))
    deliverable = recipe["inference"]["deliverable"]
    assert deliverable["compressor"] == PUBLISHED_ARRAY_METADATA["compressor"]
    assert deliverable["compressor"] == {"id": "blosc", **ome_zarr.COMPRESSOR}
    assert deliverable["format"] == f"ome-ngff-{ome_zarr.NGFF_VERSION}"
    assert deliverable["values"] == [0, ome_zarr.FOREGROUND]
    assert deliverable["downsampling"] == "nearest"
    assert deliverable["zarr_format"] == 2
