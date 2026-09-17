from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from socratic_method import ome_zarr, predict

torch = pytest.importorskip("torch")
pytest.importorskip("zarr")
import tifffile
import zarr

from crossres_pred.voxel.model import NNUNetConfig, VoxelNNUNet

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 8
HALO = 4
REGION = (8, 32, 8, 32, 8, 32)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def checkpoint(tmp_path) -> tuple[Path, dict]:
    path = tmp_path / "checkpoint.pt"
    config = NNUNetConfig(preset="tiny-test")
    torch.save(
        {
            "epoch": 1,
            "best_score": 0.5,
            "model_config": config.as_dict(),
            "model": VoxelNNUNet(config).state_dict(),
            "metrics": {"val": {"dice": 0.5, "loss_total": 0.5}},
        },
        path,
    )
    return path, {
        "samples": 1,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "file": path.name,
    }


def _recipe_document(contract: dict, *, levels: int = 3, device: str = "cpu") -> dict:
    return {
        "schema": "socratic-method-training-recipe-v1",
        "name": "tiny",
        "version": "tiny-test",
        "release": {
            "status": "selected",
            "release_id": "tiny",
            "selected_checkpoint": contract,
            "operating_threshold": 0.30,
            "training_record_operating_threshold": 0.35,
            "test_time_augmentation": "8-way-mirror",
            "inference": "raw-student-only-no-m7-blend-no-teacher",
        },
        "inference": {
            "operating_threshold": 0.30,
            "postprocessor_threshold": 0.35,
            "test_time_augmentation": "8-way-mirror",
            "mirror_tta": True,
            "amp_dtype": "bfloat16",
            "chunk_size": CHUNK,
            "halo_voxels": HALO,
            "context_shape_zyx": [CHUNK + 2 * HALO] * 3,
            "device": device,
            "max_cpu_threads": 4,
            "raw_contract": {"dtype": "uint8"},
            "voxel_size_um": {
                "accepted_range": [7.5, 10.0],
                "postprocessor_range": [8.0, 10.0],
                "refuse_below": 4.0,
            },
            "deliverable": {"levels": levels, "chunks_zyx": [CHUNK] * 3},
        },
    }


@pytest.fixture
def recipe(tmp_path, checkpoint) -> Path:
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(_recipe_document(checkpoint[1]), indent=2), encoding="utf-8")
    return path


def _volume_array(seed: int = 11, shape=(40, 40, 40)) -> np.ndarray:
    return np.random.default_rng(seed).integers(20, 200, size=shape).astype(np.uint8)


@pytest.fixture
def volume(tmp_path) -> Path:
    store = tmp_path / "scan-8.640um-1.2m-116keV-masked.zarr"
    array = zarr.open_array(
        str(store),
        mode="w",
        zarr_format=2,
        shape=(40, 40, 40),
        chunks=(CHUNK,) * 3,
        dtype="uint8",
        fill_value=0,
        dimension_separator="/",
    )
    array[:] = _volume_array()
    return store


def _plan(tmp_path, recipe, checkpoint, **kw):
    arguments = {
        "output": tmp_path / "out",
        "region": REGION,
        "recipe_path": recipe,
        "checkpoint": checkpoint[0],
    }
    arguments.update(kw)
    return predict.plan_predict(**arguments)


@pytest.fixture
def finished(tmp_path, recipe, checkpoint, volume):
    plan = _plan(tmp_path, recipe, checkpoint, volume=str(volume), umbilicus=(20.0, 21.0))
    predict.run_predict(plan, progress=lambda line: None)
    return plan, Path(plan["output"])


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def test_region_parsing() -> None:
    assert predict.parse_region("12032:12416,4096:4480,2816:3200") == (
        12032, 12416, 4096, 4480, 2816, 3200
    )
    assert predict.parse_region(" 0:128 , 0:128 , 0:128 ") == (0, 128, 0, 128, 0, 128)
    for bad in ("12032-12416,1:2,3:4", "1:2,3:4", "a:b,1:2,3:4", ""):
        with pytest.raises(ValueError, match="region must be"):
            predict.parse_region(bad)
    with pytest.raises(ValueError, match="half-open"):
        predict.parse_region("10:10,0:8,0:8")


def test_region_is_expanded_outward_to_the_lattice() -> None:
    aligned, changed = predict.align_region((10889, 11401, 2848, 3360, 3915, 4427), chunk=128)
    assert aligned == (10880, 11520, 2816, 3456, 3840, 4480)
    assert changed is True
    assert all(value % 128 == 0 for value in aligned)
    aligned, changed = predict.align_region((128, 256, 0, 128, 0, 128), chunk=128)
    assert changed is False and aligned == (128, 256, 0, 128, 0, 128)


def test_context_origins_match_the_engine() -> None:
    from crossres_pred.voxel.grid_inference import _required_context_origins

    for origin in [(128, 128, 128), (256, 384, 512), (128, 256, 128)]:
        assert sorted(predict.context_origins(origin, chunk=128, halo=32)) == sorted(
            _required_context_origins(origin, chunk_size=128, halo=32)
        )
    with pytest.raises(ValueError, match="below zero"):
        predict.context_origins((0, 128, 128), chunk=128, halo=32)


def test_edge_targets_are_skipped_with_a_reason_never_padded() -> None:
    origins = predict.region_origins((0, 24, 0, 24, 0, 24), chunk=CHUNK)
    complete, skipped, carve = predict.classify_targets(
        origins, chunk=CHUNK, halo=HALO, extent_zyx=(24, 24, 24)
    )
    assert complete == ["z00008_y00008_x00008"]
    assert len(skipped) == 26
    reasons = {item["reason"] for item in skipped}
    assert "context extends below the volume origin" in reasons
    assert any("lies outside the volume" in reason for reason in reasons)
    assert len(carve) == 27


def test_band_plan_splits_rows_under_the_byte_budget() -> None:
    origins = [(0, y, x) for y in range(0, 40, CHUNK) for x in range(0, 40, CHUNK)]
    whole = predict.band_plan(origins, chunk=CHUNK, max_band_bytes=1 << 30)
    assert len(whole) == 1 and whole[0]["y_range"] == [0, 40]
    split = predict.band_plan(origins, chunk=CHUNK, max_band_bytes=CHUNK * 40 * CHUNK)
    assert len(split) == 5
    assert [band["y_range"] for band in split] == [[0, 8], [8, 16], [16, 24], [24, 32], [32, 40]]
    assert sorted(tuple(c) for band in split for c in band["cubes"]) == sorted(origins)


# --------------------------------------------------------------------------
# pins, registry, thresholds
# --------------------------------------------------------------------------


def test_pins_round_trip_and_reject_inconsistency(checkpoint) -> None:
    document = _recipe_document(checkpoint[1])
    pins = predict.load_inference_pins(document)
    assert pins.chunk_size == CHUNK and pins.halo == HALO
    assert pins.permitted_thresholds() == (0.30, 0.35)
    assert pins.threshold_role(0.35) == "postprocessor-pin"

    broken = _recipe_document(checkpoint[1])
    broken["inference"]["mirror_tta"] = False
    with pytest.raises(ValueError, match="disagrees with mirror_tta"):
        predict.load_inference_pins(broken)

    broken = _recipe_document(checkpoint[1])
    broken["inference"]["context_shape_zyx"] = [32, 32, 32]
    with pytest.raises(ValueError, match="context_shape_zyx must equal"):
        predict.load_inference_pins(broken)

    broken = _recipe_document(checkpoint[1])
    del broken["inference"]["halo_voxels"]
    with pytest.raises(ValueError, match="missing keys: halo_voxels"):
        predict.load_inference_pins(broken)

    broken = _recipe_document(checkpoint[1])
    broken["release"]["operating_threshold"] = 0.4
    with pytest.raises(ValueError, match="disagrees with release"):
        predict.load_inference_pins(broken)


def test_a_recipe_without_pins_is_refused() -> None:
    with pytest.raises(ValueError, match="no inference section"):
        predict.load_inference_pins({"schema": "socratic-method-training-recipe-v1"})
    legacy = json.loads((ROOT / "recipes" / "v31" / "recipe.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="no inference section"):
        predict.load_inference_pins(legacy)


def test_scroll_resolution_and_aliases() -> None:
    registry = predict.load_scroll_registry()
    assert predict.resolve_scroll("PHerc1447", registry).f0_role == "unseen"
    assert predict.resolve_scroll("pherc1447", registry).id == "PHerc1447"
    assert predict.resolve_scroll("PHerc-1447", registry).id == "PHerc1447"
    assert predict.resolve_scroll("PHerc139", registry).id == "PHerc0139"
    assert predict.resolve_scroll("PHerc0139", registry).f0_role == "train"
    assert predict.resolve_scroll("PHerc0814", registry).f0_role == "val"
    # The eligible scroll and the training fragment must not collapse into one id.
    assert predict.resolve_scroll("PHerc0343", registry).f0_role == "unseen"
    assert predict.resolve_scroll("PHerc0343P", registry).f0_role == "train"
    with pytest.raises(ValueError, match="unknown scroll .*valid ids are"):
        predict.resolve_scroll("PHerc9999", registry)


def test_registry_agrees_with_the_recipe_and_the_uri_pattern() -> None:
    document = json.loads(
        (ROOT / "recipes" / "f0" / "scrolls.json").read_text(encoding="utf-8")
    )
    assert document["schema"] == predict.REGISTRY_SCHEMA
    recipe = json.loads((ROOT / "recipes" / "f0" / "recipe.json").read_text(encoding="utf-8"))
    corpus = recipe["artifacts"]["train_manifest"]
    roles = {row["id"]: row["f0_role"] for row in document["scrolls"]}
    assert set(document["f0_corpus"]["train_scrolls"]) == set(corpus["train_scrolls"])
    assert set(document["f0_corpus"]["val_scrolls"]) == set(corpus["val_scrolls"])
    for scroll in corpus["train_scrolls"]:
        assert roles[scroll] == "train", scroll
    for scroll in corpus["val_scrolls"]:
        assert roles[scroll] == "val", scroll
    assert roles["PHerc1447"] == "unseen"  # the eligibility-gate scroll stayed out

    from crossres_pred.schema import canonical_scroll_id

    canonical = [canonical_scroll_id(name) for name in roles]
    assert len(set(canonical)) == len(canonical), "two ids collapse under canonical_scroll_id"
    folded = [predict._fold(name) for name in roles]
    assert len(set(folded)) == len(folded), "two ids collapse under the loose spelling"

    for row in document["scrolls"]:
        if row["uri_source"] == "pattern" or row["uri_source"] == "recorded":
            assert row["volume_uri"].startswith(document["bucket"] + "/" + row["id"] + "/volumes/")
            assert row["volume_uri"].endswith("-masked.zarr")
            assert predict.parse_voxel_size(row["volume_uri"]) == pytest.approx(
                row["voxel_size_um"]
            ), row["id"]
        elif row["uri_source"] == "legacy":
            assert not row["volume_uri"].startswith("s3://")
        else:
            assert row["volume_uri"] is None and row["f0_role"] == "train"


def test_threshold_outside_the_preregistered_pair_is_refused(
    tmp_path, recipe, checkpoint, volume
) -> None:
    for bad in (0.5, 0.25, 0.3000001, 0.2):
        with pytest.raises(ValueError, match="is not allowed"):
            _plan(tmp_path, recipe, checkpoint, volume=str(volume), threshold=bad)
    shipped = _plan(tmp_path, recipe, checkpoint, volume=str(volume))
    assert shipped["threshold"] == 0.30
    assert shipped["threshold_role"] == "shipped-operating-point"
    pinned = _plan(tmp_path, recipe, checkpoint, volume=str(volume), threshold=0.35)
    assert pinned["threshold_role"] == "postprocessor-pin"


def test_device_type_cannot_deviate_from_the_recipe(tmp_path, recipe, checkpoint, volume) -> None:
    with pytest.raises(ValueError, match="device type"):
        _plan(tmp_path, recipe, checkpoint, volume=str(volume), device="cuda")
    document = _recipe_document(checkpoint[1], device="cuda")
    cuda_recipe = tmp_path / "cuda.json"
    cuda_recipe.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="device type"):
        _plan(tmp_path, cuda_recipe, checkpoint, volume=str(volume), device="cpu")
    plan = _plan(tmp_path, cuda_recipe, checkpoint, volume=str(volume), device="cuda:1")
    assert plan["device"] == "cuda:1"  # an index selects a card; the type is what is pinned


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def test_checkpoint_identity_is_verified_against_the_recipe(
    tmp_path, recipe, checkpoint, volume
) -> None:
    path, contract = checkpoint
    plan = _plan(tmp_path, recipe, checkpoint, volume=str(volume))
    assert plan["checkpoint"]["sha256"] == contract["sha256"]
    assert plan["checkpoint"]["source"] == "argument"

    tampered = tmp_path / "tampered.pt"
    body = bytearray(path.read_bytes())
    body[-1] ^= 0xFF
    tampered.write_bytes(bytes(body))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _plan(tmp_path, recipe, (tampered, contract), volume=str(volume))

    short = tmp_path / "short.pt"
    short.write_bytes(path.read_bytes()[:-16])
    with pytest.raises(ValueError, match="byte size mismatch"):
        _plan(tmp_path, recipe, (short, contract), volume=str(volume))

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        _plan(tmp_path, recipe, (tmp_path / "absent.pt", contract), volume=str(volume))

    with pytest.raises(ValueError, match="no checkpoint: pass --checkpoint"):
        predict.plan_predict(
            output=tmp_path / "out", volume=str(volume), region=REGION,
            recipe_path=recipe, checkpoint=None,
        )


def test_checkpoint_from_the_paths_file(tmp_path, recipe, checkpoint, volume) -> None:
    paths = tmp_path / "paths.local.json"
    paths.write_text(
        json.dumps({"selected_checkpoint": checkpoint[0].name}), encoding="utf-8"
    )
    plan = predict.plan_predict(
        output=tmp_path / "out", volume=str(volume), region=REGION,
        recipe_path=recipe, checkpoint=None, paths_file=paths)
    assert plan["checkpoint"]["source"] == "paths-file"
    assert Path(plan["checkpoint"]["path"]) == checkpoint[0]


def test_model_hf_downloads_and_still_verifies(
    tmp_path, recipe, checkpoint, volume, monkeypatch
) -> None:
    calls = {}

    def fake_download(*, repo_id, filename, cache_dir):
        calls.update(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        return str(checkpoint[0])

    monkeypatch.setitem(
        sys.modules, "huggingface_hub",
        type("m", (), {"hf_hub_download": staticmethod(fake_download)}),
    )
    plan = predict.plan_predict(
        output=tmp_path / "out", volume=str(volume), region=REGION,
        recipe_path=recipe, checkpoint=None, model_hf="someone/f0",
        cache=tmp_path / "cache")
    assert calls["repo_id"] == "someone/f0"
    assert calls["filename"] == checkpoint[0].name
    assert plan["checkpoint"]["source"] == "hugging-face:someone/f0"
    assert plan["checkpoint"]["sha256"] == checkpoint[1]["sha256"]


def test_volume_dtype_must_be_uint8(tmp_path, recipe, checkpoint) -> None:
    wide = tmp_path / "wide-8.640um.npy"
    np.save(wide, _volume_array().astype(np.uint16))
    with pytest.raises(ValueError, match="is not uint8"):
        _plan(tmp_path, recipe, checkpoint, volume=str(wide))


def test_region_must_lie_inside_the_volume(tmp_path, recipe, checkpoint, volume) -> None:
    with pytest.raises(ValueError, match="exceeds the volume shape"):
        _plan(tmp_path, recipe, checkpoint, volume=str(volume), region=(8, 48, 8, 32, 8, 32))
    with pytest.raises(ValueError, match="no target cube has complete"):
        _plan(tmp_path, recipe, checkpoint, volume=str(volume), region=(0, 8, 0, 8, 0, 8))
    with pytest.raises(ValueError, match="--region .*is required"):
        predict.plan_predict(
            output=tmp_path / "out", volume=str(volume), recipe_path=recipe,
            checkpoint=checkpoint[0],
        )


def test_pitch_is_parsed_overridden_and_gated(tmp_path, recipe, checkpoint, volume) -> None:
    plan = _plan(tmp_path, recipe, checkpoint, volume=str(volume))
    assert plan["voxel_size_um"] == 8.64
    assert plan["voxel_size_um_source"] == "parsed-from-name"
    plan = _plan(tmp_path, recipe, checkpoint, volume=str(volume), voxel_size_um=9.362)
    assert plan["voxel_size_um"] == 9.362 and plan["voxel_size_um_source"] == "argument"
    with pytest.raises(ValueError, match="is below 4.0 um"):
        _plan(tmp_path, recipe, checkpoint, volume=str(volume), voxel_size_um=2.399)
    with pytest.raises(ValueError, match="outside the model's trained range"):
        _plan(tmp_path, recipe, checkpoint, volume=str(volume), voxel_size_um=12.0)

    unnamed = tmp_path / "plain.npy"
    np.save(unnamed, _volume_array())
    plan = _plan(tmp_path, recipe, checkpoint, volume=str(unnamed))
    assert plan["voxel_size_um"] is None
    assert "socratic-repair will need --voxel-size-um" in predict.render_plan(plan)


def test_npy_volumes_work(tmp_path, recipe, checkpoint) -> None:
    path = tmp_path / "scan-9.362um.npy"
    np.save(path, _volume_array())
    plan = _plan(tmp_path, recipe, checkpoint, volume=str(path))
    assert len(plan["target_cube_ids"]) == 27
    assert plan["source"]["access"] == "local"


def test_s3_specs_open_anonymously_without_network(monkeypatch) -> None:
    seen = {}

    class Fake:
        shape = (40, 40, 40)
        dtype = np.dtype("uint8")
        chunks = (8, 8, 8)

    def fake_open(uri, *, path, mode, storage_options):
        seen.update(uri=uri, path=path, mode=mode, storage_options=storage_options)
        return Fake()

    monkeypatch.setattr(zarr, "open", fake_open)
    for name in predict._AWS_CREDENTIAL_ENV:
        monkeypatch.delenv(name, raising=False)
    uri = "s3://vesuvius-challenge-open-data/PHerc1447/volumes/v-8.640um-masked.zarr"
    _, info = predict.open_ct_volume(uri)
    assert seen == {"uri": uri, "path": "0", "mode": "r", "storage_options": {"anon": True}}
    assert info["access"] == "s3-anonymous"

    predict.open_ct_volume(uri + "::2")
    assert seen["path"] == "2" and seen["uri"] == uri

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "something")
    predict.open_ct_volume(uri)
    assert seen["storage_options"] == {"anon": False}

    def explode(*args, **kwargs):
        raise PermissionError("403 Forbidden")

    monkeypatch.setattr(zarr, "open", explode)
    with pytest.raises(ValueError, match="cannot open volume s3://") as caught:
        predict.open_ct_volume(uri)
    assert isinstance(caught.value.__cause__, PermissionError)


def test_scroll_mode_uses_the_registry(tmp_path, recipe, checkpoint, volume, monkeypatch) -> None:
    paths = tmp_path / "paths.local.json"
    paths.write_text(
        json.dumps({"scroll_mirrors": {"PHerc1447": str(volume)}}), encoding="utf-8"
    )
    plan = _plan(tmp_path, recipe, checkpoint, scroll="PHerc1447", paths_file=paths)
    assert plan["scroll"]["id"] == "PHerc1447"
    assert plan["scroll"]["f0_role"] == "unseen"
    assert plan["umbilicus_yx"] == [3287.0, 3784.0]
    assert plan["source"]["spec"] == str(volume)
    summary = predict.render_plan(plan)
    assert "PHerc1447" in summary and "not in F0's training corpus" in summary

    paths.write_text(
        json.dumps({"scroll_mirrors": {"PHerc0139": str(volume)}}), encoding="utf-8"
    )
    plan = _plan(tmp_path, recipe, checkpoint, scroll="PHerc0139", paths_file=paths)
    assert "WARNING: this scroll is in F0's training corpus" in predict.render_plan(plan)

    with pytest.raises(ValueError, match="has no recorded volume URI"):
        _plan(tmp_path, recipe, checkpoint, scroll="PHerc1299")
    with pytest.raises(ValueError, match="legacy acquisition"):
        _plan(tmp_path, recipe, checkpoint, scroll="PHerc0332")


def test_exactly_one_source_is_required(tmp_path, recipe, checkpoint, volume) -> None:
    with pytest.raises(ValueError, match="exactly one of --scroll"):
        predict.plan_predict(
            output=tmp_path / "out", volume=str(volume), scroll="PHerc1447",
            region=REGION, recipe_path=recipe, checkpoint=checkpoint[0],
        )


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


def test_check_writes_nothing(tmp_path, recipe, checkpoint, volume) -> None:
    plan = _plan(tmp_path, recipe, checkpoint, volume=str(volume))
    assert not (tmp_path / "out").exists()
    assert plan["schema"] == predict.PLAN_SCHEMA
    assert plan["automatic_deployment"] is False and plan["research_only"] is True
    assert len(plan["target_cube_ids"]) == 27
    assert len(plan["carve_cube_ids"]) == 125
    assert len(plan["ring_cube_ids"]) == 98
    assert "infer-grid" in plan["equivalent_command"]
    assert "--threshold" in plan["equivalent_command"]
    assert "--tta" in plan["equivalent_command"]


def test_run_writes_one_self_contained_grid_and_a_deliverable(finished) -> None:
    plan, output = finished
    assert {item.name for item in output.iterdir()} == {
        "prediction.zarr", "cubes_PRED", "cubes_RAW", "probability",
        "manifest.json", "source_manifest.json", "provenance.json",
        "predict_plan.json", "predict_receipt.json",
    }
    assert not list(output.parent.glob("*.predict.lock"))
    assert not list(output.glob("f0*"))

    predictions = sorted((output / "cubes_PRED").glob("*.tif"))
    assert len(predictions) == 27
    for path in predictions:
        cube = tifffile.imread(path)
        assert cube.dtype == np.uint8 and cube.shape == (CHUNK,) * 3
        assert set(np.unique(cube)) <= {0, 255}
    assert len(sorted((output / "probability").glob("*.tif"))) == 27
    assert len(sorted((output / "cubes_RAW").glob("*.tif"))) == 125

    present = json.loads((output / "cubes_PRED" / "present.json").read_text(encoding="utf-8"))
    assert present == sorted(plan["target_cube_ids"])

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["chunk_size"] == CHUNK
    assert manifest["umbilicus_yx"] == [20.0, 21.0]
    assert manifest["voxel_size_um"] == 8.64
    assert manifest["created_by"] == "socratic-predict"
    assert json.loads((output / "source_manifest.json").read_text(encoding="utf-8")) == manifest


def test_engine_provenance_names_the_self_contained_grid(finished) -> None:
    plan, output = finished
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["source_grid"] == str(output)
    assert provenance["threshold"] == 0.30
    assert provenance["halo"] == HALO
    assert provenance["mirror_tta"] is True
    assert provenance["amp_dtype"] == "bfloat16"
    assert provenance["checkpoint_sha256"] == plan["checkpoint"]["sha256"]
    assert sorted(provenance["target_cube_ids"]) == sorted(plan["target_cube_ids"])


def test_deliverable_matches_the_cubes_at_global_coordinates(finished) -> None:
    plan, output = finished
    ome_zarr.validate_ome_ngff(output / DELIVERABLE_NAME)
    root = zarr.open(str(output / DELIVERABLE_NAME), mode="r")
    level0 = np.asarray(root["0"][:])
    assert level0.shape == (40, 40, 40)
    for cube_id in plan["target_cube_ids"]:
        z, y, x = predict.parse_cube_id(cube_id)
        mask = tifffile.imread(output / "cubes_PRED" / f"{cube_id}.tif")
        assert np.array_equal(level0[z : z + CHUNK, y : y + CHUNK, x : x + CHUNK], mask), cube_id
    sidecar = json.loads(
        (output / DELIVERABLE_NAME / "metadata.json").read_text(encoding="utf-8")
    )
    assert sidecar["kind"] == "surface-inference"
    assert sidecar["post_processing"]["threshold_value"] == 0.30
    assert sidecar["model"]["checkpoint_sha256"] == plan["checkpoint"]["sha256"]
    assert sidecar["output"]["ome_downsampling_algorithm"] == "nearest"
    assert sidecar["provenance"]["deployment_ready"] is False


DELIVERABLE_NAME = predict.DELIVERABLE


def test_receipt_inventory_matches_the_files(finished) -> None:
    _, output = finished
    receipt = json.loads((output / "predict_receipt.json").read_text(encoding="utf-8"))
    assert receipt["schema"] == predict.RECEIPT_SCHEMA
    assert receipt["status"] == "complete"
    assert receipt["automatic_deployment"] is False
    assert receipt["raw_cubes"]["materialized"]["carved"] == 125
    for group in ("cubes_PRED", "probability"):
        entries = receipt["output_files"][group]
        assert len(entries) == 27
        for entry in entries:
            body = (output / entry["path"]).read_bytes()
            assert hashlib.sha256(body).hexdigest() == entry["sha256"]
            assert len(body) == entry["bytes"]
    for entry in receipt["raw_cubes"]["files"]:
        path = output / "cubes_RAW" / f"{entry['cube_id']}.tif"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    assert receipt["deliverable"]["levels"] == 3
    assert receipt["deliverable"]["chunks_written"]["0"] == 27


def test_engine_is_called_on_the_self_contained_grid(
    tmp_path, recipe, checkpoint, volume, monkeypatch
) -> None:
    import crossres_pred.voxel.grid_inference as engine

    seen = {}
    original = engine.infer_voxel_grid

    def spy(**kwargs):
        seen.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(engine, "infer_voxel_grid", spy)
    plan = _plan(tmp_path, recipe, checkpoint, volume=str(volume))
    predict.run_predict(plan, progress=lambda line: None)
    assert str(seen["source_grid"]) == plan["output"]
    assert str(seen["output_path"]) == plan["engine_directory"]
    assert seen["target_cube_ids"] is None
    assert seen["skip_incomplete_context"] is False
    assert seen["threshold"] == 0.30 and seen["halo"] == HALO


def test_existing_output_lock_or_partial_is_refused(
    tmp_path, recipe, checkpoint, volume
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(FileExistsError, match="output exists"):
        _plan(tmp_path, recipe, checkpoint, volume=str(volume))
    with pytest.raises(ValueError, match="nothing to resume"):
        _plan(tmp_path, recipe, checkpoint, volume=str(volume), resume=True)

    (output / predict.PLAN_FILE).write_text("{}", encoding="utf-8")
    (output / f"{predict.ENGINE_DIR}.partial-123").mkdir()
    with pytest.raises(FileExistsError, match="engine partial output exists"):
        _plan(tmp_path, recipe, checkpoint, volume=str(volume), resume=True)


def test_a_held_lock_is_never_reclaimed(tmp_path, recipe, checkpoint, volume) -> None:
    plan = _plan(tmp_path, recipe, checkpoint, volume=str(volume))
    lock = Path(plan["lock"])
    lock.write_text("unknown owner", encoding="utf-8")
    with pytest.raises(FileExistsError):
        predict.run_predict(plan, progress=lambda line: None)
    assert lock.read_text(encoding="utf-8") == "unknown owner"
    assert not Path(plan["output"]).exists()


def test_resume_reuses_verified_raw_cubes(
    tmp_path, recipe, checkpoint, volume, monkeypatch
) -> None:
    import crossres_pred.voxel.grid_inference as engine

    plan = _plan(tmp_path, recipe, checkpoint, volume=str(volume))
    monkeypatch.setattr(
        engine, "infer_voxel_grid", lambda **kw: (_ for _ in ()).throw(RuntimeError("gpu fell over"))
    )
    with pytest.raises(RuntimeError, match="gpu fell over"):
        predict.run_predict(plan, progress=lambda line: None)
    output = Path(plan["output"])
    assert len(sorted((output / "cubes_RAW").glob("*.tif"))) == 125
    assert not (output / predict.RECEIPT_FILE).exists()
    assert not Path(plan["lock"]).exists()

    monkeypatch.undo()
    resumed = _plan(tmp_path, recipe, checkpoint, volume=str(volume), resume=True)
    records: list[str] = []
    predict.run_predict(resumed, progress=records.append)
    receipt = json.loads((output / predict.RECEIPT_FILE).read_text(encoding="utf-8"))
    assert receipt["raw_cubes"]["materialized"]["reused"] == 125
    assert receipt["raw_cubes"]["materialized"]["carved"] == 0


def test_resume_refuses_a_plan_for_a_different_job(
    tmp_path, recipe, checkpoint, volume, monkeypatch
) -> None:
    import crossres_pred.voxel.grid_inference as engine

    plan = _plan(tmp_path, recipe, checkpoint, volume=str(volume))
    monkeypatch.setattr(
        engine, "infer_voxel_grid", lambda **kw: (_ for _ in ()).throw(RuntimeError("stop"))
    )
    with pytest.raises(RuntimeError):
        predict.run_predict(plan, progress=lambda line: None)
    monkeypatch.undo()
    other = _plan(
        tmp_path, recipe, checkpoint, volume=str(volume), threshold=0.35, resume=True
    )
    with pytest.raises(ValueError, match="resume plan differs"):
        predict.run_predict(other, progress=lambda line: None)


def test_gpu_preflight_runs_before_anything_is_written(
    tmp_path, recipe, checkpoint, volume, monkeypatch
) -> None:
    plan = _plan(tmp_path, recipe, checkpoint, volume=str(volume))
    monkeypatch.setattr(
        predict, "preflight_device",
        lambda plan: (_ for _ in ()).throw(RuntimeError("power limit is 700.0 W")),
    )
    with pytest.raises(RuntimeError, match="power limit"):
        predict.run_predict(plan, progress=lambda line: None)
    assert not Path(plan["output"]).exists()
    assert not Path(plan["lock"]).exists()


# --------------------------------------------------------------------------
# grid mode
# --------------------------------------------------------------------------


def test_grid_mode_links_raw_and_filters_by_region(finished, tmp_path, recipe, checkpoint) -> None:
    _, grid = finished
    output = tmp_path / "from-grid"
    plan = predict.plan_predict(
        output=output, grid=grid, recipe_path=recipe, checkpoint=checkpoint[0], threshold=0.35
    )
    assert plan["mode"] == "grid"
    assert plan["raw_transfer"] == "hardlink"
    assert plan["voxel_size_um"] == 8.64
    assert plan["umbilicus_yx"] == [20.0, 21.0]  # inherited from the input manifest
    # The finished grid carries a full raw ring, so every published cube predicts again.
    assert len(plan["target_cube_ids"]) == 27
    predict.run_predict(plan, progress=lambda line: None)
    cube = plan["carve_cube_ids"][0]
    assert (output / "cubes_RAW" / f"{cube}.tif").samefile(grid / "cubes_RAW" / f"{cube}.tif")
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["source_grid"] == str(output)
    assert provenance["threshold"] == 0.35
    receipt = json.loads((output / predict.RECEIPT_FILE).read_text(encoding="utf-8"))
    assert receipt["plan"]["source"]["input_grid"] == str(grid)


def test_a_carved_grid_is_recognised_by_the_volume_it_records(
    finished, tmp_path, recipe, checkpoint
) -> None:
    """A grid naming a known volume gets the whole scroll frame, not just its box."""

    _, grid = finished
    manifest = json.loads((grid / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.pop("volume_shape_zyx") == [40, 40, 40]  # written by volume mode
    (grid / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    plan = predict.plan_predict(
        output=tmp_path / "unknown-frame", grid=grid, recipe_path=recipe,
        checkpoint=checkpoint[0],
    )
    assert plan["scroll"] is None
    assert plan["zarr"]["shape_source"] == "region-upper-bound"

    manifest["sources"]["raw"] = (
        "s3://vesuvius-challenge-open-data/PHerc1447/volumes/"
        "20250521151220-8.640um-1.2m-116keV-masked.zarr"
    )
    (grid / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    plan = predict.plan_predict(
        output=tmp_path / "scroll-frame", grid=grid, recipe_path=recipe,
        checkpoint=checkpoint[0],
    )
    assert plan["scroll"]["id"] == "PHerc1447"
    assert plan["zarr"]["shape_zyx"] == [24297, 8343, 8343]
    assert plan["zarr"]["shape_source"] == "scroll-registry"
    summary = predict.render_plan(plan)
    assert "grid       " in summary and "scroll     PHerc1447" in summary


def test_scroll_for_volume_matches_on_the_store_name() -> None:
    uri = (
        "s3://vesuvius-challenge-open-data/PHerc1447/volumes/"
        "20250521151220-8.640um-1.2m-116keV-masked.zarr"
    )
    assert predict.scroll_for_volume(uri).id == "PHerc1447"
    local = r"D:\local" + "\\" + uri.rsplit("/", 1)[-1]
    assert predict.scroll_for_volume(local).id == "PHerc1447"
    assert predict.scroll_for_volume("some-other-volume.zarr") is None
    assert predict.scroll_for_volume(None) is None
    assert predict.scroll_for_volume("") is None


def test_grid_mode_copy_raw(finished, tmp_path, recipe, checkpoint) -> None:
    _, grid = finished
    output = tmp_path / "copied"
    plan = predict.plan_predict(
        output=output, grid=grid, recipe_path=recipe, checkpoint=checkpoint[0], copy_raw=True
    )
    assert plan["raw_transfer"] == "copy"
    predict.run_predict(plan, progress=lambda line: None)
    cube = plan["carve_cube_ids"][0]
    assert not (output / "cubes_RAW" / f"{cube}.tif").samefile(
        grid / "cubes_RAW" / f"{cube}.tif"
    )


def test_grid_mode_requirements(finished, tmp_path, recipe, checkpoint) -> None:
    _, grid = finished
    with pytest.raises(ValueError, match="region selects no cube"):
        predict.plan_predict(
            output=tmp_path / "a", grid=grid, region=(0, 8, 0, 8, 0, 8),
            recipe_path=recipe, checkpoint=checkpoint[0],
        )
    with pytest.raises(ValueError, match="output must be a new directory outside"):
        predict.plan_predict(
            output=grid, grid=grid, recipe_path=recipe, checkpoint=checkpoint[0]
        )
    with pytest.raises(ValueError, match="output must be a new directory outside"):
        predict.plan_predict(
            output=grid / "nested", grid=grid, recipe_path=recipe,
            checkpoint=checkpoint[0]
        )
    manifest = json.loads((grid / "manifest.json").read_text(encoding="utf-8"))
    manifest["chunk_size"] = 64
    (grid / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="must declare chunk_size"):
        predict.plan_predict(
            output=tmp_path / "b", grid=grid, recipe_path=recipe, checkpoint=checkpoint[0]
        )


def test_grid_mode_refuses_a_conflicting_umbilicus(finished, tmp_path, recipe, checkpoint) -> None:
    _, grid = finished
    with pytest.raises(ValueError, match="disagrees with the input grid"):
        predict.plan_predict(
            output=tmp_path / "c", grid=grid, umbilicus=(1.0, 2.0),
            recipe_path=recipe, checkpoint=checkpoint[0],
        )


def test_a_postprocessor_threshold_run_carries_the_contract_fields(
    tmp_path, recipe, checkpoint, volume
) -> None:
    plan = _plan(
        tmp_path, recipe, checkpoint, volume=str(volume), threshold=0.35,
        umbilicus=(20.0, 21.0), output=tmp_path / "t035",
    )
    output = predict.run_predict(plan, progress=lambda line: None)
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    # The exact fields socratic-repair and socratic-continuity read from an input grid.
    assert provenance["threshold"] == 0.35
    assert provenance["checkpoint_sha256"] == plan["checkpoint"]["sha256"]
    assert provenance["halo"] == HALO and provenance["mirror_tta"] is True
    assert provenance["amp_dtype"] == "bfloat16"
    assert isinstance(manifest["umbilicus_yx"], list) and len(manifest["umbilicus_yx"]) == 2
    assert all(np.isfinite(manifest["umbilicus_yx"]))
    stems = sorted(path.stem for path in (output / "cubes_PRED").glob("*.tif"))
    present = json.loads((output / "cubes_PRED" / "present.json").read_text(encoding="utf-8"))
    assert stems == present == sorted(provenance["target_cube_ids"])


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------


def test_main_check_prints_prose_and_json(tmp_path, recipe, checkpoint, volume, capsys) -> None:
    arguments = [
        "--volume", str(volume), "--region", "8:32,8:32,8:32",
        "--out", str(tmp_path / "cli"), "--recipe", str(recipe),
        "--checkpoint", str(checkpoint[0]),
    ]
    assert predict.main(arguments) == 0
    prose = capsys.readouterr().out
    assert prose.startswith("socratic-predict plan")
    assert "Nothing written" in prose
    assert not (tmp_path / "cli").exists()

    assert predict.main(arguments + ["--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["schema"] == predict.PLAN_SCHEMA

    assert predict.main(arguments + ["--print-command"]) == 0
    assert capsys.readouterr().out.startswith("crossres-voxel infer-grid ")


def test_main_reports_errors_on_stderr(tmp_path, recipe, checkpoint, volume, capsys) -> None:
    assert predict.main([
        "--volume", str(volume), "--region", "8:32,8:32,8:32",
        "--out", str(tmp_path / "cli"), "--recipe", str(recipe),
        "--checkpoint", str(checkpoint[0]), "--threshold", "0.5",
    ]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("socratic-predict: error: ")
    assert "is not allowed" in captured.err


def test_main_rejects_passes_and_resume_without_run(tmp_path, recipe, volume) -> None:
    base = ["--volume", str(volume), "--region", "8:32,8:32,8:32", "--out", str(tmp_path / "x")]
    with pytest.raises(SystemExit) as caught:
        predict.main(base + ["--passes", "3"])
    assert caught.value.code == 2
    with pytest.raises(SystemExit):
        predict.main(base + ["--resume"])
    with pytest.raises(SystemExit):
        predict.main(base + ["--check", "--run"])


def test_bbox_and_region_are_equivalent(tmp_path, recipe, checkpoint, volume) -> None:
    one = _plan(tmp_path, recipe, checkpoint, volume=str(volume))
    two = predict.plan_predict(
        output=tmp_path / "out", volume=str(volume),
        region=predict.parse_region("8:32,8:32,8:32"),
        recipe_path=recipe, checkpoint=checkpoint[0],
    )
    assert one["target_cube_ids"] == two["target_cube_ids"]
    assert one["region"] == two["region"] == list(REGION)


def test_unaligned_regions_are_expanded_and_recorded(tmp_path, recipe, checkpoint, volume) -> None:
    plan = _plan(tmp_path, recipe, checkpoint, volume=str(volume), region=(9, 31, 9, 31, 9, 31))
    assert plan["region_requested"] == [9, 31, 9, 31, 9, 31]
    assert plan["region"] == [8, 32, 8, 32, 8, 32]
    assert plan["region_expanded_to_lattice"] is True
    assert "expanded outward" in predict.render_plan(plan)

