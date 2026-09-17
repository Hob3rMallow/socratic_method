from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from socratic_method.hf_export import (
    _release_contract,
    _validate_selection_summary,
    default_model_card,
)
from socratic_method.recipe import (
    REQUIRED_PATHS,
    build_command,
    default_recipe_path,
    load_paths,
    optional_path_keys,
    required_path_keys,
    verify_recipe,
)

ROOT = Path(__file__).resolve().parents[1]
RECIPE_DIR = ROOT / "recipes" / "f0"
RECORD = ROOT / "recipes" / "final_c3_250k_20260907"
HEX64 = set("0123456789abcdef")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def recipe() -> dict:
    return _json(RECIPE_DIR / "recipe.json")


@pytest.fixture(scope="module")
def sealed() -> dict:
    return _json(RECORD / "provenance" / "recipe.json")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return _json(RECORD / "manifest.json")


def test_recipe_structure_and_constants(recipe: dict) -> None:
    for key in ("schema", "name", "version", "artifacts", "model", "release", "objective", "optimization", "evaluation", "inference", "training"):
        assert key in recipe, key
    assert recipe["schema"] == "socratic-method-training-recipe-v1"
    assert recipe["version"] == "c3-f0-200k-tta-t030-20260909"
    release = recipe["release"]
    assert release["status"] == "selected"
    assert release["inference"] == "raw-student-only-no-m7-blend-no-teacher"
    assert release["selection_status"] == "frozen-model-selected"
    assert 0 < release["operating_threshold"] < 1
    assert release["operating_threshold"] == 0.30 and release["training_record_operating_threshold"] == 0.35
    assert release["test_time_augmentation"] == "8-way-mirror" and release["previous_release_id"] == "c3-f0-200k-20260907"
    assert set(release["selected_checkpoint"]["sha256"]) <= HEX64 and len(release["selected_checkpoint"]["sha256"]) == 64
    assert isinstance(recipe["training"]["argv"], list) and all(isinstance(t, str) for t in recipe["training"]["argv"])
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(recipe, _json(ROOT / "recipes" / "schema.json"))


def test_artifact_pins_match_the_sealed_record(recipe: dict, manifest: dict, sealed: dict) -> None:
    train = recipe["artifacts"]["train_manifest"]
    assert train["sha256"] == manifest["training"]["corpus_manifest_sha256"] == sealed["corpus"]["manifest_sha256"]
    assert train["rows"] == manifest["training"]["corpus_rows"] == sealed["corpus"]["rows"]
    assert train["train_rows"] == manifest["training"]["train_rows"]
    assert train["val_rows"] == manifest["training"]["val_rows"]
    assert len(train["train_scrolls"]) == manifest["training"]["train_scrolls"]
    assert len(train["val_scrolls"]) == manifest["training"]["val_scrolls"]
    assert train["native_soft_rows"] == manifest["training"]["native_soft_rows"]
    assert train["human_hard_passthrough_rows"] == manifest["training"]["human_hard_passthrough_rows"]
    assert set(train["train_record_ids"]) <= set(train["record_ids"])
    assert train["train_scrolls"] == sorted(set(train["scrolls"]) - set(train["val_scrolls"]))
    assert recipe["artifacts"]["m7_checkpoint"]["sha256"] == manifest["training"]["fresh_released_m7_sha256"] == sealed["initialization"]["sha256"]
    frozen = recipe["artifacts"]["frozen_validation_manifest"]
    assert frozen["sha256"] == manifest["results"]["validation_manifest_sha256"] == sealed["evaluation"]["val_manifest_sha256"]
    assert frozen["stage"] == "evaluation"
    selected = recipe["release"]["selected_checkpoint"]
    assert selected["sha256"] == manifest["model"]["sha256"]
    assert selected["bytes"] == manifest["model"]["bytes"]
    assert selected["samples"] == manifest["model"]["selected_sample_exposures"]
    assert recipe["release"]["training_record_operating_threshold"] == manifest["model"]["operating_threshold"]
    assert recipe["release"]["completed_training_samples"] == manifest["model"]["completed_training_horizon"]
    assert recipe["source_recipe_sha256"] == _sha256(RECORD / "provenance" / "recipe.json") == manifest["training"]["config_sha256"]
    assert recipe["source_run_identity_sha256"] == manifest["training"]["run_identity_sha256"]
    assert recipe["objective"]["contract"] == "soft-ce-weighted-dice-separation-shell-m7-kl-unknown-corridor-v4"
    assert recipe["optimization"]["snapshot_samples"] == sealed["arms"][0]["snapshot_samples"]
    assert recipe["evaluation"]["expected_selected_macro_dice_at_035"] == manifest["results"]["F0_200k_035_macro_dice"]


def test_training_argv_is_token_identical_to_the_sealed_command(recipe: dict, sealed: dict) -> None:
    sealed_argv = list(sealed["arms"][0]["argv"])
    options = {}
    for index, token in enumerate(sealed_argv):
        if token in ("--patches", "--output", "--m7-checkpoint"):
            options[token] = sealed_argv[index + 1]
    paths = {
        "train_manifest": options["--patches"],
        "output": options["--output"],
        "m7_checkpoint": options["--m7-checkpoint"],
        "frozen_validation_manifest": "unused-by-training",
    }
    command = build_command(recipe, paths, python=sealed_argv[0])
    assert command == sealed_argv
    assert build_command(recipe, paths, python="py", resume=True)[-1] == "--resume"


def test_required_and_optional_path_keys(recipe: dict) -> None:
    assert required_path_keys(recipe) == ("train_manifest", "m7_checkpoint", "output")
    assert optional_path_keys(recipe) == ("frozen_validation_manifest",)
    v31 = _json(ROOT / "recipes" / "v31" / "recipe.json")
    assert required_path_keys(v31) == REQUIRED_PATHS
    assert optional_path_keys(v31) == ()


def test_default_recipe_follows_current_model(tmp_path: Path) -> None:
    assert default_recipe_path(ROOT) == RECIPE_DIR / "recipe.json"
    (tmp_path / "recipes" / "v31").mkdir(parents=True)
    (tmp_path / "recipes" / "v31" / "recipe.json").write_text("{}", encoding="utf-8")
    assert default_recipe_path(tmp_path) == tmp_path / "recipes" / "v31" / "recipe.json"
    (tmp_path / "CURRENT_MODEL.json").write_text(json.dumps({"executable_recipe": "recipes/x/recipe.json"}), encoding="utf-8")
    assert default_recipe_path(tmp_path) == tmp_path / "recipes" / "v31" / "recipe.json"  # missing file falls back
    (tmp_path / "recipes" / "x").mkdir()
    (tmp_path / "recipes" / "x" / "recipe.json").write_text("{}", encoding="utf-8")
    assert default_recipe_path(tmp_path) == tmp_path / "recipes" / "x" / "recipe.json"


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _wide_shaped(tmp_path: Path) -> tuple[dict, dict]:
    rows = [
        {"scroll_id": "PHerc0139", "record_id": "r-native", "split": "train"},
        {"scroll_id": "PHerc0139", "record_id": "r-human", "split": "train"},
        {"scroll_id": "PHerc1667", "record_id": "r-native", "split": "train"},
        {"scroll_id": "PHerc0814", "record_id": "r-val", "split": "val"},
    ]
    train_manifest = tmp_path / "train.jsonl"
    _write_rows(train_manifest, rows)
    m7 = tmp_path / "m7.pth"
    m7.write_bytes(b"m7 fixture")
    frozen = tmp_path / "frozen.jsonl"
    _write_rows(frozen, [{"scroll_id": "PHerc0814", "split": "val"}, {"scroll_id": "PHerc1451", "split": "val"}, {"scroll_id": "PHerc0139", "split": "train"}])
    recipe = {
        "schema": "socratic-method-training-recipe-v1",
        "artifacts": {
            "train_manifest": {
                "kind": "train-manifest", "sha256": _sha256(train_manifest), "bytes": train_manifest.stat().st_size,
                "rows": 4, "train_rows": 3, "val_rows": 1,
                "scrolls": ["PHerc0139", "PHerc0814", "PHerc1667"], "train_scrolls": ["PHerc0139", "PHerc1667"],
                "val_scrolls": ["PHerc0814"], "record_ids": ["r-human", "r-native", "r-val"],
                "train_record_ids": ["r-human", "r-native"],
            },
            "m7_checkpoint": {"sha256": _sha256(m7), "bytes": m7.stat().st_size},
            "frozen_validation_manifest": {
                "kind": "validation-manifest", "stage": "evaluation", "sha256": _sha256(frozen),
                "bytes": frozen.stat().st_size, "rows": 3, "val_rows": 2, "scrolls": ["PHerc0814", "PHerc1451"],
            },
        },
    }
    paths = {"train_manifest": str(train_manifest), "m7_checkpoint": str(m7), "output": str(tmp_path / "out")}
    return recipe, paths


def test_verify_recipe_on_wide15_shaped_inputs(tmp_path: Path) -> None:
    recipe, paths = _wide_shaped(tmp_path)
    messages = verify_recipe(recipe, paths)
    assert any(m.startswith("ok train_manifest") for m in messages)
    assert any(m.startswith("skip frozen_validation_manifest") for m in messages)
    assert any("4 rows (3 train / 1 val)" in m for m in messages)
    with_frozen = {**paths, "frozen_validation_manifest": str(tmp_path / "frozen.jsonl")}
    messages = verify_recipe(recipe, with_frozen)
    assert any("scope validation: 2 val rows" in m for m in messages)
    broken = json.loads(json.dumps(recipe))
    broken["artifacts"]["train_manifest"]["train_scrolls"] = ["PHerc0139"]
    with pytest.raises(ValueError, match="training split scroll scope changed"):
        verify_recipe(broken, paths)
    broken = json.loads(json.dumps(recipe))
    broken["artifacts"]["train_manifest"]["val_rows"] = 2
    with pytest.raises(ValueError, match="in-corpus validation row count changed"):
        verify_recipe(broken, paths)
    broken = json.loads(json.dumps(recipe))
    broken["artifacts"]["m7_checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_recipe(broken, paths)
    missing = dict(paths)
    missing.pop("m7_checkpoint")
    with pytest.raises(ValueError, match="unresolved artifact contract: m7_checkpoint"):
        verify_recipe(recipe, missing)


def test_verify_recipe_keeps_v31_connectivity_checks(tmp_path: Path) -> None:
    train_manifest = tmp_path / "train.jsonl"
    _write_rows(train_manifest, [{"scroll_id": "PHerc0139", "record_id": "r", "split": "train"}])
    validation = tmp_path / "val.jsonl"
    _write_rows(validation, [{"scroll_id": "PHerc0814", "split": "val"}])
    m7 = tmp_path / "m7.pth"
    m7.write_bytes(b"m7")
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"event_count": 149, "fully_owned_event_count": 149, "maximum_propagation_steps": 96, "maximum_required_connectivity_steps": 44}), encoding="utf-8")
    recipe = {
        "schema": "socratic-method-training-recipe-v1",
        "artifacts": {
            "train_manifest": {"sha256": _sha256(train_manifest), "rows": 1, "scrolls": ["PHerc0139"], "record_ids": ["r"]},
            "validation_manifest": {"sha256": _sha256(validation), "scrolls": ["PHerc0814"]},
            "m7_checkpoint": {"sha256": _sha256(m7)},
            "dynamic_medial_connectivity_state": {"sha256": _sha256(state), "event_count": 149, "fully_owned_event_count": 149, "maximum_propagation_steps": 96, "maximum_required_connectivity_steps": 44},
        },
    }
    paths = {"train_manifest": str(train_manifest), "validation_manifest": str(validation), "m7_checkpoint": str(m7), "dynamic_medial_connectivity_state": str(state), "output": str(tmp_path / "out")}
    assert any(m.startswith("ok dynamic_medial_connectivity_state") for m in verify_recipe(recipe, paths))
    recipe["artifacts"]["dynamic_medial_connectivity_state"]["event_count"] = 150
    with pytest.raises(ValueError, match="event_count changed"):
        verify_recipe(recipe, paths)


def test_load_paths_with_recipe_keys(tmp_path: Path, recipe: dict) -> None:
    value = {"train_manifest": "a.jsonl", "m7_checkpoint": "m7.pth", "output": "runs/f0", "frozen_validation_manifest": "v.jsonl", "environment": {"CUDA_VISIBLE_DEVICES": "GPU-x"}}
    path = tmp_path / "paths.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    loaded = load_paths(path, required=required_path_keys(recipe), optional=optional_path_keys(recipe))
    assert loaded["frozen_validation_manifest"] == str((tmp_path / "v.jsonl").resolve())
    assert loaded["environment"] == {"CUDA_VISIBLE_DEVICES": "GPU-x"}
    value.pop("m7_checkpoint")
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="missing path keys: m7_checkpoint"):
        load_paths(path, required=required_path_keys(recipe), optional=optional_path_keys(recipe))


def test_observed_metrics_and_release_records_are_derived_from_the_record() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_f0_observed_metrics as metrics_builder
    import build_f0_release_records as records_builder

    observed = _json(RECIPE_DIR / "observed_metrics.json")
    assert observed == metrics_builder.build_observed_metrics(ROOT)
    assert len(observed["milestones"]) == 25
    assert observed["release_selection"]["checkpoint_samples"] == 200000
    manifest = _json(RECORD / "manifest.json")
    analysis = _json(RECORD / "analysis.json")
    inference = _json(ROOT / "recipes" / "f0_inference_20260909" / "manifest.json")
    selection = _json(RECIPE_DIR / "selection.json")
    expected_selection = records_builder.build_selection(manifest, analysis, inference)
    assert selection == expected_selection
    qualification = _json(RECIPE_DIR / "release_qualification.json")
    expected_qualification = records_builder.build_qualification(manifest, analysis, inference)
    expected_qualification["created_at_utc"] = qualification["created_at_utc"]
    assert qualification == expected_qualification


def test_export_contract_accepts_the_f0_records(recipe: dict) -> None:
    contract = _release_contract(RECIPE_DIR / "recipe.json", RECIPE_DIR / "release_qualification.json")
    assert contract["samples"] == 200000
    assert contract["operating_threshold"] == 0.30
    assert contract["test_time_augmentation"] == "8-way-mirror"
    assert contract["sha256"] == recipe["release"]["selected_checkpoint"]["sha256"]
    assert contract["version"] == "c3-f0-200k-tta-t030-20260909"
    assert contract["selection_status"] == "frozen-model-selected"
    assert contract["model_type"] == "socratic-c3-f0"
    _validate_selection_summary(RECIPE_DIR / "selection.json", contract)
    assert default_model_card(RECIPE_DIR / "recipe.json", ROOT) == ROOT / "huggingface" / "f0" / "README.md"
    assert (ROOT / "huggingface" / "f0" / "README.md").is_file()
    card_config = _json(ROOT / "huggingface" / "f0" / "config.json")
    assert card_config["source_checkpoint_sha256"] == contract["sha256"]
    assert card_config["operating_threshold"] == 0.30
    assert card_config["test_time_augmentation"] == "8-way-mirror"
    current = _json(ROOT / "CURRENT_MODEL.json")
    assert current["executable_recipe"] == "recipes/f0/recipe.json"
    assert current["checkpoint_sha256"] == contract["sha256"]


def test_inference_pins_match_the_records(recipe: dict, manifest: dict) -> None:
    """The socratic-predict runner reads these; every one must be evidenced."""

    from socratic_method.predict import load_inference_pins

    inference = recipe["inference"]
    release = recipe["release"]
    shipped = _json(ROOT / "recipes" / "f0_inference_20260909" / "manifest.json")["inference"]

    assert inference["operating_threshold"] == release["operating_threshold"] == 0.30
    assert inference["operating_threshold"] == shipped["operating_threshold"]
    assert inference["postprocessor_threshold"] == release["training_record_operating_threshold"]
    assert inference["postprocessor_threshold"] == manifest["model"]["operating_threshold"] == 0.35
    assert inference["mirror_tta"] is shipped["mirror_tta"] is True
    assert inference["amp_dtype"] == shipped["amp_dtype"] == "bfloat16"
    assert inference["test_time_augmentation"] == shipped["test_time_augmentation"]

    current = _json(ROOT / "CURRENT_MODEL.json")
    assert current["postprocessor_profiles_qualified_at_threshold"] == inference["postprocessor_threshold"]
    assert current["inference_pins"] == "recipes/f0/recipe.json#/inference"
    assert current["scroll_registry"] == "recipes/f0/scrolls.json"
    assert current["predict_command"].startswith("socratic-predict ")

    # halo, chunk size, context shape, device and thread budget are only recorded as
    # the geometry the postprocessor evidence was produced with. Both records agree.
    for name in ("development", "control"):
        observed = _json(
            ROOT / "recipes" / "f0_repair_20260907" / "provenance"
            / f"{name}_inference_provenance.json"
        )
        assert inference["halo_voxels"] == observed["halo"] == 32
        assert inference["chunk_size"] == observed["chunk_size"] == 128
        assert inference["context_shape_zyx"] == observed["context_shape_zyx"] == [192, 192, 192]
        assert inference["mirror_tta"] == observed["mirror_tta"]
        assert inference["amp_dtype"] == observed["amp_dtype"]
        assert inference["device"] == observed["device"]
        assert inference["max_cpu_threads"] == observed["options"]["max_cpu_threads"]
        assert inference["postprocessor_threshold"] == observed["threshold"]
        assert observed["checkpoint_sha256"] == release["selected_checkpoint"]["sha256"]

    assert inference["context_shape_zyx"] == [
        inference["chunk_size"] + 2 * inference["halo_voxels"]
    ] * 3
    from crossres_pred.voxel.model import NNUNetConfig

    divisor = NNUNetConfig(preset=recipe["model"]["preset"]).required_divisor
    assert all(size % divisor == 0 for size in inference["context_shape_zyx"])

    # The CT contract is the engine's own normalisation, not a second set of numbers.
    from crossres_pred.voxel.patches import M7_CT_LOWER, M7_CT_MEAN, M7_CT_STD, M7_CT_UPPER

    raw = inference["raw_contract"]
    assert raw["dtype"] == "uint8"
    assert raw["ct_clip"] == [M7_CT_LOWER, M7_CT_UPPER]
    assert raw["ct_mean"] == M7_CT_MEAN and raw["ct_std"] == M7_CT_STD

    # Three training scrolls are 7.910 um volumes, so inference must accept them even
    # though socratic-repair separately pins the narrower range it was measured in.
    pitch = inference["voxel_size_um"]
    assert pitch["accepted_range"][0] <= min(pitch["training_corpus_range"])
    assert pitch["accepted_range"][1] >= max(pitch["training_corpus_range"])
    assert pitch["postprocessor_range"] == [8.0, 10.0]
    assert pitch["qualified"] == [8.64]

    pins = load_inference_pins(recipe)
    assert pins.permitted_thresholds() == (0.30, 0.35)
    assert pins.chunk_size == 128 and pins.halo == 32
    assert pins.context_shape_zyx == (192, 192, 192)
    assert pins.mirror_tta is True and pins.amp_dtype == "bfloat16"


def test_predict_is_a_console_script() -> None:
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = project["project"]["scripts"]
    assert scripts["socratic-predict"] == "socratic_method.predict:main"
