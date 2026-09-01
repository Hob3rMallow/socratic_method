from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch
from torch.utils.data import DataLoader

from crossres_pred.sparse_zarr import ZarrArraySpec
from crossres_pred.voxel.affines import find_volume_affine
from crossres_pred.voxel.checkpoint_audit import (
    CheckpointAuditOptions,
    ScalarCounts,
    ThresholdCounts,
    audit_voxel_checkpoint,
    summarize_thresholds,
)
from crossres_pred.voxel.grid_inference import (
    _replace_directory_with_retry,
    assemble_raw_context,
    infer_voxel_grid,
)
from crossres_pred.voxel.io import dense_field_masks, open_volume
from crossres_pred.voxel.loss import (
    deep_supervision_loss,
    dice_ce_loss,
    segmentation_metrics,
)
from crossres_pred.voxel.manual_labels import (
    extract_level0,
    index_label_zarr,
    verify_archive,
)
from crossres_pred.voxel.model import NNUNetConfig, VoxelNNUNet
from crossres_pred.voxel.patches import (
    PATCH_PREPARATION_VERSION,
    VoxelPatchDataset,
    load_patch_manifest,
    validate_patch_corpus,
)
from crossres_pred.voxel.prepare import (
    PrepareOptions,
    _acceptable,
    _load_existing_rows,
    _patch_executor_workers,
    _quality_filter_native_teacher_support,
    _sampling_strategy,
    _select_candidate,
    _spatially_ordered_anchors,
    _support_anchor_candidate_schedule,
    _support_anchor_fallback_schedule,
    _support_anchor_pool_requires_reuse,
    _support_anchor_reuse_fallback_schedule,
    _support_anchor_schedule,
    prepare_patch_corpus,
)
from crossres_pred.voxel.registered_mirror import select_registered_chunks
from crossres_pred.voxel.registration import (
    ChunkSupport,
    SparseChunkProjectionCache,
    _valid_centers,
    affine_matrix,
    invert_affine,
    voxelize_fine_target_patch,
)
from crossres_pred.voxel.resources import assert_cuda_power_limit
from crossres_pred.voxel.schema import (
    ChunkSupportSpec,
    DenseFieldSpec,
    VoxelSchemaError,
    load_pair_manifest,
)
from crossres_pred.voxel.scrollfiesta_metrics import (
    ScrollFiestaPredMetrics,
    scrollfiesta_patch_pred_metrics,
    scrollfiesta_pred_metrics,
)
from crossres_pred.voxel.teacher import (
    TeacherOptions,
    _atomic_json,
    _candidate_execution_order,
    _canonical_teacher_identity,
    _effective_teacher_chunk_target,
    _initialize_target_zarr,
    _predict_blended_probability,
    _PredictionLogitCache,
    _replace_with_retry,
    _write_chunk_atomic,
    _write_inventory,
    centered_crop_slices,
    interior_chunk_coordinates,
    load_present_chunk_coordinates,
    validate_local_support_snapshot,
    validate_teacher_materialization,
    villa_gaussian_map,
    villa_sliding_window_steps,
)
from crossres_pred.voxel.teacher_model import LoadedTeacher, normalize_instance_zscore
from crossres_pred.voxel.train import validate_model

IDENTITY_AFFINE = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
)


def test_grid_commit_retries_transient_windows_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "partial"
    target = tmp_path / "complete"
    source.mkdir()
    calls = 0
    real_replace = os.replace

    def flaky_replace(old: Path, new: Path) -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError("scanner still owns a handle")
        real_replace(old, new)

    monkeypatch.setattr("crossres_pred.voxel.grid_inference.os.replace", flaky_replace)
    monkeypatch.setattr("crossres_pred.voxel.grid_inference.time.sleep", lambda _: None)

    _replace_directory_with_retry(source, target)

    assert calls == 3
    assert target.is_dir()
    assert not source.exists()


def test_cuda_power_guard_fails_closed_above_approved_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        returncode = 0
        stdout = "600.00\n"
        stderr = ""

    monkeypatch.setattr(
        "crossres_pred.voxel.resources.subprocess.run",
        lambda *args, **kwargs: Result(),
    )
    assert assert_cuda_power_limit(torch.device("cuda")) == pytest.approx(600.0)

    Result.stdout = "601.00\n"
    with pytest.raises(RuntimeError, match="required maximum"):
        assert_cuda_power_limit(torch.device("cuda"))


def test_scrollfiesta_prediction_metrics_match_canonical_cases() -> None:
    volume = np.zeros((64, 64, 64), dtype=np.uint8)
    volume[12:52, 12:52, 12:52] = 255
    solid = scrollfiesta_pred_metrics(volume)
    assert solid.reject_kind == "solid-slab"
    assert solid.foreground_voxels == 40**3
    assert solid.largest_component_voxels == 40**3
    assert solid.interior_voxels == 36**3
    assert solid.interior_fraction == pytest.approx(36**3 / 40**3)
    assert solid.max_thickness == 17
    assert solid.max_rectangle_run == 40
    assert ScrollFiestaPredMetrics.from_dict(solid.to_dict()) == solid

    volume.fill(0)
    volume[:, 20:22, :] = 255
    volume[:, :, 40:42] = 255
    thin = scrollfiesta_pred_metrics(volume)
    assert thin.reject_kind == "keep"
    assert thin.interior_voxels == 0

    volume.fill(0)
    volume[30:32, 7:57, 7:57] = 255
    assert scrollfiesta_pred_metrics(volume).reject_kind == "keep"

    volume.fill(0)
    volume[4:11, 4:11, 4:11] = 255
    volume[4:11, 4:11, 40:47] = 255
    volume[40:47, 40:47, 40:47] = 255
    scattered = scrollfiesta_pred_metrics(volume)
    assert scattered.foreground_voxels == 3 * 7**3
    assert scattered.largest_component_voxels == 7**3
    assert scattered.reject_kind == "empty"


def test_scrollfiesta_patch_metrics_use_centered_deploy_cube() -> None:
    patch = np.zeros((192, 192, 192), dtype=np.uint8)
    patch[52:140, 52:140, 52:140] = 1
    metrics = scrollfiesta_patch_pred_metrics(patch)
    assert metrics.window_origin_zyx == (32, 32, 32)
    assert metrics.window_shape_zyx == (128, 128, 128)
    assert metrics.reject_kind == "solid-slab"


def test_scrollfiesta_metric_port_matches_repository_executable(
    tmp_path: Path,
) -> None:
    tool = Path(__file__).resolve().parents[2] / "build" / "Release" / "pred_reject.exe"
    if not tool.is_file():
        pytest.skip("repository ScrollFiesta pred_reject executable is not built")
    volume = np.zeros((64, 64, 64), dtype=np.uint8)
    volume[12:52, 12:52, 12:52] = 255
    source = tmp_path / "solid.tif"
    tifffile.imwrite(source, volume)
    result = subprocess.run(
        [str(tool), str(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(
        r"fg=(\d+)\s+cc=(\d+)\s+fill=([0-9.]+)\s+"
        r"interior=([0-9.]+)\((\d+)\)\s+thick=(\d+)\s+"
        r"rect\[ax(\d+)\]=([0-9.]+)\s+run=(\d+)",
        result.stdout,
    )
    assert match is not None
    metrics = scrollfiesta_pred_metrics(volume)
    assert int(match.group(1)) == metrics.foreground_voxels
    assert int(match.group(2)) == metrics.largest_component_voxels
    assert float(match.group(3)) == pytest.approx(metrics.fill_fraction, abs=5.0e-5)
    assert float(match.group(4)) == pytest.approx(
        metrics.interior_fraction,
        abs=5.0e-4,
    )
    assert int(match.group(5)) == metrics.interior_voxels
    assert int(match.group(6)) == metrics.max_thickness
    assert int(match.group(7)) == metrics.rectangle_axis
    assert float(match.group(8)) == pytest.approx(
        metrics.rectangle_fraction,
        abs=5.1e-3,
    )
    assert int(match.group(9)) == metrics.max_rectangle_run
    assert metrics.reason in result.stdout


def test_high_pathology_selection_prefers_scrollfiesta_reject() -> None:
    thin = np.zeros((64, 64, 64), dtype=np.uint8)
    thin[30:31, 8:56, 8:56] = 1
    solid = np.zeros_like(thin)
    solid[12:52, 12:52, 12:52] = 1
    candidates = [
        (
            {"baseline_u8": thin},
            {"pathology_score": 0.9, "positive_fraction_known": 0.1},
        ),
        (
            {"baseline_u8": solid},
            {"pathology_score": 0.1, "positive_fraction_known": 0.1},
        ),
    ]
    selected = _select_candidate(candidates, "high-pathology")
    assert selected is candidates[1]
    assert selected[1]["scrollfiesta_pred_metrics"]["reject_kind"] == "solid-slab"


def test_official_manual_label_archive_extracts_and_indexes_level0(
    tmp_path: Path,
) -> None:
    from numcodecs import Blosc

    source = tmp_path / "source" / "tiny.zarr"
    level0 = source / "0"
    level1 = source / "1"
    level0.mkdir(parents=True)
    level1.mkdir()
    codec = Blosc(cname="zstd", clevel=1, shuffle=Blosc.NOSHUFFLE)
    zarray = {
        "zarr_format": 2,
        "shape": [4, 4, 12],
        "chunks": [4, 4, 4],
        "dtype": "|u1",
        "compressor": codec.get_config(),
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "dimension_separator": ".",
    }
    (source / ".zgroup").write_text('{"zarr_format":2}', encoding="utf-8")
    (source / ".zattrs").write_text("{}", encoding="utf-8")
    (level0 / ".zarray").write_text(json.dumps(zarray), encoding="utf-8")
    first = np.zeros((4, 4, 4), dtype=np.uint8)
    first[0] = 1
    first[1] = 2
    (level0 / "0.0.0").write_bytes(codec.encode(first))
    (level0 / "0.0.1").write_bytes(codec.encode(np.full_like(first, 2)))
    (level0 / "0.0.2").write_bytes(b"")
    (level1 / ".zarray").write_text(json.dumps(zarray), encoding="utf-8")
    (level1 / "0.0.0").write_bytes(b"must-not-be-extracted")

    archive = tmp_path / "labels.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                output.write(
                    path, f"labels/tiny.zarr/{path.relative_to(source).as_posix()}"
                )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "schema": "crossres-official-manual-label-corpus-v1",
                "source": {
                    "size_bytes": archive.stat().st_size,
                    "xet_hash": "test",
                },
                "datasets": [
                    {
                        "dataset_id": "tiny",
                        "zarr": "tiny.zarr",
                        "positive_labels": [1],
                        "ignore_labels": [2],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    audit = verify_archive(archive, catalog)
    assert audit["zip_members"] == 8
    assert "crc_checked" not in audit
    labels_root = tmp_path / "labels_l0"
    extract_level0(
        archive_path=archive,
        output_path=labels_root,
        catalog_path=catalog,
        max_cpu_threads=4,
    )
    extracted_chunk = labels_root / "tiny.zarr" / "0" / "0.0.0"
    assert extracted_chunk.is_file()
    extracted_placeholder = labels_root / "tiny.zarr" / "0" / "0.0.2"
    assert extracted_placeholder.is_file()
    assert extracted_placeholder.stat().st_size == 0
    assert not (labels_root / "tiny.zarr" / "1").exists()

    # A power loss can leave a same-size file behind after its atomic rename.
    # Resume must validate content, not trust size alone.
    expected_chunk = extracted_chunk.read_bytes()
    extracted_chunk.write_bytes(b"\0" * len(expected_chunk))
    extract_state = labels_root / "crossres_manual_extract.json"
    state = json.loads(extract_state.read_text(encoding="utf-8"))
    state["state"] = "extracting"
    extract_state.write_text(json.dumps(state), encoding="utf-8")
    extract_level0(
        archive_path=archive,
        output_path=labels_root,
        catalog_path=catalog,
        max_cpu_threads=4,
    )
    assert extracted_chunk.read_bytes() == expected_chunk

    rows_path = index_label_zarr(
        zarr_path=labels_root / "tiny.zarr",
        positive_labels=(1,),
        ignore_labels=(2,),
        workers=4,
        max_cpu_threads=4,
    )
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    assert sum(row["positive_voxels"] for row in rows) == 16
    assert sum(row["known_voxels"] for row in rows) == 48
    assert [row["positive_voxels"] for row in rows] == [16, 0, 0]
    assert [row["size"] for row in rows] == [
        len(expected_chunk),
        len(expected_chunk),
        0,
    ]
    placeholder = rows[-1]
    assert placeholder["decoded_voxels"] == 0
    assert placeholder["known_voxels"] == 0
    assert placeholder["background_voxels"] == 0
    assert placeholder["observed_labels"] == {}
    assert placeholder["storage"] == "zero-byte-unknown-placeholder"

    # Indexes completed before the policy was named remain valid. The policy
    # only makes their existing missing-is-unknown behavior explicit.
    index_state = labels_root / "tiny.zarr" / "crossres_label_index.json"
    legacy_state = json.loads(index_state.read_text(encoding="utf-8"))
    legacy_state["identity"].pop("zero_length_chunk_policy")
    index_state.write_text(json.dumps(legacy_state), encoding="utf-8")
    assert (
        index_label_zarr(
            zarr_path=labels_root / "tiny.zarr",
            positive_labels=(1,),
            ignore_labels=(2,),
            workers=4,
            max_cpu_threads=4,
        )
        == rows_path
    )


def test_registered_chunk_selection_uses_only_positive_label_support(
    tmp_path: Path,
) -> None:
    zarray = tmp_path / ".zarray"
    zarray.write_text(
        json.dumps({"shape": [8, 8, 8], "chunks": [4, 4, 4]}),
        encoding="utf-8",
    )
    inventory = tmp_path / "chunks.jsonl"
    inventory.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "kind": "chunk",
                        "coordinate_zyx": [0, 0, 0],
                        "positive_voxels": 0,
                    }
                ),
                json.dumps(
                    {
                        "kind": "chunk",
                        "relative_path": "0/1/1/1",
                        "positive_voxels": 10,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    source = ZarrArraySpec((8, 8, 8), (4, 4, 4), "/")
    selected = select_registered_chunks(
        label_zarray=zarray,
        label_inventory=inventory,
        fine_to_source_affine_xyz=[list(row) for row in IDENTITY_AFFINE],
        source_spec=source,
        halo_voxels=0,
    )
    assert source.encode_chunk((1, 1, 1)) in selected
    assert source.encode_chunk((0, 0, 0)) in selected
    assert len(selected) == 8


def test_open_volume_accepts_zarr_v2_little_endian_uint8(tmp_path: Path) -> None:
    store = tmp_path / "little_endian_u1.zarr"
    array_root = store / "0"
    array_root.mkdir(parents=True)
    (store / ".zgroup").write_text('{"zarr_format":2}', encoding="utf-8")
    (array_root / ".zarray").write_text(
        json.dumps(
            {
                "zarr_format": 2,
                "shape": [2, 2, 2],
                "chunks": [2, 2, 2],
                "dtype": "<u1",
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
                "dimension_separator": ".",
            }
        ),
        encoding="utf-8",
    )
    expected = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
    (array_root / "0.0.0").write_bytes(expected.tobytes(order="C"))

    volume = open_volume(f"{store}::0")
    assert volume.dtype == np.dtype(np.uint8)
    assert np.array_equal(np.asarray(volume[:]), expected)


def test_teacher_support_ceiling_is_explicit_and_opt_in() -> None:
    strict = TeacherOptions(chunks=256)
    with pytest.raises(RuntimeError, match="only 165 full teacher neighborhoods"):
        _effective_teacher_chunk_target(165, strict)

    bounded = TeacherOptions(chunks=256, allow_fewer_chunks=True)
    assert _effective_teacher_chunk_target(165, bounded) == 165
    assert _effective_teacher_chunk_target(300, bounded) == 256


def test_patch_sampling_strata_do_not_inflate_without_a_baseline() -> None:
    options = PrepareOptions(
        pathology_fraction=1 / 3,
        positive_density_fraction=1 / 6,
    )
    with_baseline = [
        _sampling_strategy(
            index,
            patch_count=600,
            has_baseline=True,
            options=options,
        )
        for index in range(600)
    ]
    without_baseline = [
        _sampling_strategy(
            index,
            patch_count=600,
            has_baseline=False,
            options=options,
        )
        for index in range(600)
    ]
    assert 190 <= with_baseline.count("high-pathology") <= 210
    assert 90 <= with_baseline.count("dense-positive") <= 110
    assert without_baseline.count("high-pathology") == 0
    assert 90 <= without_baseline.count("dense-positive") <= 110
    assert _patch_executor_workers(3) == 6
    assert _patch_executor_workers(16) == 16


def test_native_teacher_support_schedule_is_deterministic_and_non_repeating() -> None:
    coordinates = np.asarray(
        [(index // 100, (index // 10) % 10, index % 10) for index in range(300)],
        dtype=np.int64,
    )
    first = _support_anchor_schedule(
        coordinates,
        record_id="PHerc0841-native-teacher-synthetic-l2",
        supervision_source="official-native-fine-teacher/fine-only",
        seed=1203,
    )
    second = _support_anchor_schedule(
        coordinates,
        record_id="PHerc0841-native-teacher-synthetic-l2",
        supervision_source="official-native-fine-teacher/fine-only",
        seed=1203,
    )
    assert first is not None
    assert second is not None
    assert np.array_equal(first, second)
    assert len(set(map(tuple, first[:256]))) == 256
    assert set(map(tuple, first)) == set(map(tuple, coordinates))
    assert (
        _support_anchor_schedule(
            coordinates,
            record_id="manual",
            supervision_source="official-human-label",
            seed=1203,
        )
        is None
    )
    human = _support_anchor_schedule(
        coordinates,
        record_id="official-human",
        supervision_source="official-human-2um/registered-real",
        seed=1203,
    )
    assert human is not None
    assert np.array_equal(human, first) is False
    assert set(map(tuple, human)) == set(map(tuple, coordinates))


def test_surplus_native_anchors_feed_ranked_strata_without_overlap() -> None:
    coordinates = np.asarray(
        [(index // 100, (index // 10) % 10, index % 10) for index in range(300)],
        dtype=np.int64,
    )
    options = PrepareOptions(selection_candidates=4)
    schedule = _support_anchor_schedule(
        coordinates,
        record_id="ranked-native-teacher",
        supervision_source="official-native-fine-teacher/test",
        seed=1203,
    )
    groups = _support_anchor_candidate_schedule(
        schedule,
        patch_count=100,
        has_baseline=True,
        options=options,
    )
    assert groups is not None
    assert len(groups) == 100
    strategies = [
        _sampling_strategy(
            index,
            patch_count=100,
            has_baseline=True,
            options=options,
        )
        for index in range(100)
    ]
    assert all(
        len(group) == 4
        for group, strategy in zip(groups, strategies, strict=True)
        if strategy != "random"
    )
    assert all(
        len(group) == 1
        for group, strategy in zip(groups, strategies, strict=True)
        if strategy == "random"
    )
    flattened = [tuple(anchor) for group in groups for anchor in group]
    assert len(flattened) == len(set(flattened))
    assert len(flattened) > 100
    base = np.stack([group[0] for group in groups])
    assert set(map(tuple, base)) == set(map(tuple, schedule[:100]))
    assert np.array_equal(base, _spatially_ordered_anchors(schedule[:100]))
    assert np.array_equal(base[0], schedule[0])


def test_unused_native_anchors_form_disjoint_patch_fallbacks() -> None:
    coordinates = np.asarray(
        [(index // 100, (index // 10) % 10, index % 10) for index in range(350)],
        dtype=np.int64,
    )
    options = PrepareOptions(selection_candidates=4)
    schedule = _support_anchor_schedule(
        coordinates,
        record_id="fallback-native-teacher",
        supervision_source="official-native-fine-teacher/test",
        seed=1203,
    )
    primary = _support_anchor_candidate_schedule(
        schedule,
        patch_count=100,
        has_baseline=True,
        options=options,
    )
    fallback = _support_anchor_fallback_schedule(
        schedule,
        primary,
    )
    assert schedule is not None
    assert primary is not None
    assert fallback is not None
    primary_coordinates = {
        tuple(coordinate) for group in primary for coordinate in group
    }
    fallback_coordinates = [tuple(coordinate) for coordinate in fallback]
    assert not primary_coordinates.intersection(fallback_coordinates)
    assert len(fallback_coordinates) == len(set(fallback_coordinates))
    assert primary_coordinates.union(fallback_coordinates) == set(map(tuple, schedule))
    assert np.array_equal(
        fallback,
        _support_anchor_fallback_schedule(
            schedule,
            primary,
        ),
    )


def test_sparse_native_anchor_pool_rotates_through_reusable_fallbacks() -> None:
    schedule = np.asarray(
        [(0, 0, 0), (1, 1, 1), (2, 2, 2)],
        dtype=np.int64,
    )
    options = PrepareOptions(selection_candidates=4)
    primary = _support_anchor_candidate_schedule(
        schedule,
        patch_count=5,
        has_baseline=True,
        options=options,
    )
    assert primary is not None
    assert tuple(primary[4][0]) == (1, 1, 1)

    fallback = _support_anchor_reuse_fallback_schedule(
        schedule,
        primary[4],
        patch_index=4,
    )
    assert [tuple(coordinate) for coordinate in fallback] == [
        (2, 2, 2),
        (0, 0, 0),
    ]
    assert np.array_equal(
        fallback,
        _support_anchor_reuse_fallback_schedule(
            schedule,
            primary[4],
            patch_index=4,
        ),
    )


def test_native_anchor_pool_reuses_fallbacks_at_exact_exhaustion() -> None:
    schedule = np.asarray(
        [(0, 0, 0), (1, 1, 1), (2, 2, 2)],
        dtype=np.int64,
    )
    assert _support_anchor_pool_requires_reuse(schedule, patch_count=3)
    assert _support_anchor_pool_requires_reuse(schedule, patch_count=4)
    assert not _support_anchor_pool_requires_reuse(schedule, patch_count=2)
    assert not _support_anchor_pool_requires_reuse(None, patch_count=3)


def test_ranked_native_patch_selects_best_distinct_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_sparse_native_pair_manifest(tmp_path)
    store = tmp_path / "fine_sparse.zarr"
    inventory = tmp_path / "fine_sparse_chunks.jsonl"
    extra_anchor = (0, 1, 1)
    relative = "0/" + ".".join(str(item) for item in extra_anchor)
    (store / relative).write_bytes(
        np.ones((32, 32, 32), dtype=np.uint8).tobytes(order="C")
    )
    with inventory.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(
                {
                    "kind": "chunk",
                    "relative_path": relative,
                    "positive_voxels": 32**3,
                }
            )
            + "\n"
        )
    baseline = np.zeros((64, 64, 64), dtype=np.uint8)
    np.save(tmp_path / "baseline_sparse.npy", baseline, allow_pickle=False)
    pair = json.loads(manifest.read_text(encoding="utf-8"))
    pair["coarse"]["baseline"] = {
        "volume": "baseline_sparse.npy",
        "encoding": "labels",
        "positive_labels": [1],
    }
    manifest.write_text(json.dumps(pair) + "\n", encoding="utf-8")

    def fake_candidate(
        *_args: object,
        support_anchor_coordinate_zyx: np.ndarray | None = None,
        support_anchor_pool_size: int | None = None,
        **_kwargs: object,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        assert support_anchor_coordinate_zyx is not None
        anchor = tuple(int(item) for item in support_anchor_coordinate_zyx)
        score = float((anchor[0] * 4 + anchor[1] * 2 + anchor[2]) / 7)
        shape = (32, 32, 32)
        target = np.zeros(shape, dtype=np.uint8)
        target[0] = 1
        return (
            {
                "image": np.ones(shape, dtype=np.uint8),
                "target_u8": target,
                "baseline_u8": np.zeros(shape, dtype=np.uint8),
            },
            {
                "origin_zyx": [0, 0, 0],
                "known_fraction": 1.0,
                "positive_voxels": 32**2,
                "positive_fraction_known": 1 / 32,
                "fine_positive_voxels": 32**2,
                "chunks_read": 1,
                "ct_nonzero_fraction": 1.0,
                "pathology_score": score,
                "has_baseline": True,
                "support_anchor_chunk_zyx": list(anchor),
                "support_anchor_pool_size": support_anchor_pool_size,
            },
        )

    monkeypatch.setattr(
        "crossres_pred.voxel.prepare._prepare_candidate", fake_candidate
    )
    patch_manifest = prepare_patch_corpus(
        pair_manifest=manifest,
        output_path=tmp_path / "ranked_sparse_patches",
        options=PrepareOptions(
            patches_per_record=2,
            patch_shape_zyx=(32, 32, 32),
            min_known_fraction=0.20,
            native_teacher_min_known_fraction=0.20,
            min_positive_voxels=16,
            attempts_per_patch=4,
            selection_candidates=2,
            validity_block=16,
        ),
    )
    rows = [json.loads(line) for line in patch_manifest.read_text().splitlines()]
    ranked = next(row for row in rows if row["sampling_strategy"] == "high-pathology")
    candidates = [tuple(item) for item in ranked["support_anchor_candidate_chunks_zyx"]]
    assert len(candidates) == 2
    assert tuple(ranked["support_anchor_chunk_zyx"]) == max(
        candidates,
        key=lambda anchor: anchor[0] * 4 + anchor[1] * 2 + anchor[2],
    )
    assert ranked["candidate_count"] == 2


def test_isolated_factor_four_teacher_chunk_has_subpercent_coverage() -> None:
    known_fraction = (128 / 4) ** 3 / (192**3)
    assert known_fraction == pytest.approx(1 / 216)
    assert 0.002 < known_fraction < 0.20


def test_known_fraction_gate_is_source_aware() -> None:
    options = PrepareOptions(
        min_known_fraction=0.05,
        native_teacher_min_known_fraction=0.002,
    )
    stats = {
        "known_fraction": 1 / 216,
        "positive_voxels": 32,
        "ct_nonzero_fraction": 0.5,
    }
    assert _acceptable(
        stats,
        options,
        supervision_source="official-native-fine-teacher/test",
    )
    assert not _acceptable(
        stats,
        options,
        supervision_source="official-human-label",
    )


def test_final_fit_dataset_includes_train_and_validation_rows(tmp_path: Path) -> None:
    patch = tmp_path / "patch.npz"
    np.savez(
        patch,
        image=np.zeros((32, 32, 32), dtype=np.uint8),
        target_u8=np.zeros((32, 32, 32), dtype=np.uint8),
    )
    manifest = tmp_path / "patches.jsonl"
    rows = []
    for split in ("train", "val", "test"):
        rows.append(
            {
                "schema": "crossres-voxel-patch-v1",
                "patch_id": split,
                "path": str(patch),
                "record_id": split,
                "scroll_id": split,
                "split": split,
                "origin_zyx": [0, 0, 0],
                "shape_zyx": [32, 32, 32],
                "known_fraction": 1.0,
                "positive_fraction_known": 0.0,
            }
        )
    manifest.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    dataset = VoxelPatchDataset(manifest, split=None, augment=False)
    assert [row.split for row in dataset.rows] == ["train", "val"]


def test_validation_reports_probability_and_pathology_strata() -> None:
    target = torch.zeros((1, 4, 4, 4), dtype=torch.long)
    target[:, 1] = 1

    class PerfectModel(torch.nn.Module):
        def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
            truth = target.to(image.device)
            logits = torch.stack(
                (
                    torch.where(truth == 0, 5.0, -5.0),
                    torch.where(truth == 1, 5.0, -5.0),
                ),
                dim=1,
            )
            return [logits]

    loader = DataLoader(
        [
            {
                "image": torch.zeros((1, 4, 4, 4)),
                "target": target,
                "baseline": torch.zeros((1, 4, 4, 4)),
                "has_baseline": torch.tensor(True),
                "pathology_score": torch.tensor(0.2),
                "scrollfiesta_pred_reject_kind": torch.tensor(2),
                "patch_id": "patch",
                "scroll_id": "PHercGold",
                "supervision_source": "human",
                "sampling_strategy": "high-pathology",
            }
        ],
        batch_size=1,
    )
    metrics = validate_model(
        PerfectModel(),
        loader,
        torch.device("cpu"),
        torch.float32,
        False,
    )
    assert metrics["dice"] == 1.0
    assert metrics["probability_min"] < 0.01
    assert metrics["probability_max"] > 0.99
    assert metrics["stratum/pathology/high/dice"] == 1.0
    assert metrics["stratum/pathology/high/dice_gain"] > 0
    assert metrics["stratum/scrollfiesta_pred/solid_slab/dice"] == 1.0


def test_checkpoint_threshold_sweep_qualifies_against_true_pair() -> None:
    thresholds = (0.5, 0.7)
    probability = np.asarray([0.9, 0.6, 0.55, 0.1], dtype=np.float32)
    target = np.asarray([1, 1, 0, 0], dtype=np.uint8)
    baseline_prediction = np.asarray([1, 0, 1, 0], dtype=bool)
    overall = ThresholdCounts.create(thresholds)
    comparison = ThresholdCounts.create(thresholds)
    scroll = ThresholdCounts.create(thresholds)
    baseline = ScalarCounts()
    scroll_baseline = ScalarCounts()
    for counts in (overall, comparison, scroll):
        counts.update(probability, target)
    baseline.update(baseline_prediction, target)
    scroll_baseline.update(baseline_prediction, target)

    report = summarize_thresholds(
        overall=overall,
        comparison=comparison,
        baseline=baseline,
        by_scroll={"PHerc0814": scroll},
        comparison_by_scroll={"PHerc0814": scroll},
        baseline_by_scroll={"PHerc0814": scroll_baseline},
        qualification_scroll="PHerc0814",
    )

    assert report["any_qualified"] is True
    assert report["selected"]["threshold"] == pytest.approx(0.5)
    assert report["selected"]["dice"] == pytest.approx(0.8)
    assert report["selected"]["baseline_comparison"][
        "dice_gain_vs_baseline"
    ] == pytest.approx(0.3)


def test_checkpoint_threshold_sweep_rejects_any_scroll_regression() -> None:
    thresholds = (0.5, 0.7)

    dense = ThresholdCounts.create(thresholds)
    dense.true_positive[:] = (90, 85)
    dense.false_positive[:] = (10, 15)
    dense.false_negative[:] = (10, 15)
    dense.known = 200
    dense.positive = 100
    dense_baseline = ScalarCounts(80, 20, 20, 200, 100)

    sparse = ThresholdCounts.create(thresholds)
    sparse.true_positive[:] = (4, 6)
    sparse.false_positive[:] = (6, 4)
    sparse.false_negative[:] = (6, 4)
    sparse.known = 20
    sparse.positive = 10
    sparse_baseline = ScalarCounts(5, 5, 5, 20, 10)

    overall = ThresholdCounts.create(thresholds)
    overall.true_positive[:] = dense.true_positive + sparse.true_positive
    overall.false_positive[:] = dense.false_positive + sparse.false_positive
    overall.false_negative[:] = dense.false_negative + sparse.false_negative
    overall.known = dense.known + sparse.known
    overall.positive = dense.positive + sparse.positive
    baseline = ScalarCounts(85, 25, 25, 220, 110)

    report = summarize_thresholds(
        overall=overall,
        comparison=overall,
        baseline=baseline,
        by_scroll={"PHerc0500P2": dense, "PHerc0814": sparse},
        comparison_by_scroll={"PHerc0500P2": dense, "PHerc0814": sparse},
        baseline_by_scroll={
            "PHerc0500P2": dense_baseline,
            "PHerc0814": sparse_baseline,
        },
        qualification_scroll="PHerc0814",
    )

    assert report["points"][0]["baseline_comparison"]["dice_gain_vs_baseline"] > 0
    assert report["points"][0]["qualified"] is False
    assert report["points"][0]["minimum_scroll_dice_gain_vs_baseline"] < 0
    assert report["points"][1]["qualified"] is True
    assert report["selected"]["threshold"] == pytest.approx(0.7)
    assert report["selection_metric"] == "macro_scroll_dice"
    assert report["required_baseline_scrolls"] == ["PHerc0500P2", "PHerc0814"]


def test_checkpoint_threshold_sweep_uses_matched_rows_for_scroll_gain() -> None:
    thresholds = (0.5,)
    matched_probability = np.asarray([0.9, 0.8, 0.1, 0.1], dtype=np.float32)
    matched_target = np.asarray([1, 1, 0, 0], dtype=np.uint8)
    unmatched_probability = np.ones(20, dtype=np.float32)
    unmatched_target = np.zeros(20, dtype=np.uint8)
    baseline_prediction = np.asarray([1, 0, 1, 0], dtype=bool)

    overall = ThresholdCounts.create(thresholds)
    comparison = ThresholdCounts.create(thresholds)
    all_scroll = ThresholdCounts.create(thresholds)
    matched_scroll = ThresholdCounts.create(thresholds)
    baseline = ScalarCounts()
    scroll_baseline = ScalarCounts()
    for counts in (overall, all_scroll):
        counts.update(matched_probability, matched_target)
        counts.update(unmatched_probability, unmatched_target)
    for counts in (comparison, matched_scroll):
        counts.update(matched_probability, matched_target)
    baseline.update(baseline_prediction, matched_target)
    scroll_baseline.update(baseline_prediction, matched_target)

    report = summarize_thresholds(
        overall=overall,
        comparison=comparison,
        baseline=baseline,
        by_scroll={"PHerc1451": all_scroll},
        comparison_by_scroll={"PHerc1451": matched_scroll},
        baseline_by_scroll={"PHerc1451": scroll_baseline},
        qualification_scroll="PHerc1451",
    )

    selected = report["selected"]
    scroll = selected["scrolls"]["PHerc1451"]
    assert selected["qualified"] is True
    assert scroll["dice"] == pytest.approx(1.0 / 6.0)
    assert scroll["known_voxels"] == 24
    assert scroll["baseline_comparison"]["dice"] == pytest.approx(1.0)
    assert scroll["baseline_comparison"]["known_voxels"] == 4
    assert scroll["baseline_comparison"]["dice_gain_vs_baseline"] == pytest.approx(0.5)
    assert scroll["dice_gain_vs_baseline"] == pytest.approx(0.5)
    assert report["baseline_comparison_policy"] == "matched-rows-only"


def test_checkpoint_audit_writes_json_and_html(tmp_path: Path) -> None:
    config = NNUNetConfig(preset="tiny-test")
    model = VoxelNNUNet(config)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "epoch": 0,
            "model_config": config.as_dict(),
            "model": model.state_dict(),
            "metrics": {},
        },
        checkpoint,
    )
    patch = tmp_path / "patch.npz"
    target = np.zeros((32, 32, 32), dtype=np.uint8)
    target[12:20, 12:20, 12:20] = 1
    np.savez(
        patch,
        image=np.zeros(target.shape, dtype=np.uint8),
        target_u8=target,
        baseline_u8=np.zeros(target.shape, dtype=np.uint8),
    )
    manifest = tmp_path / "patches.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema": "crossres-voxel-patch-v1",
                "patch_id": "audit",
                "path": str(patch),
                "record_id": "audit",
                "scroll_id": "PHerc0814",
                "split": "val",
                "origin_zyx": [0, 0, 0],
                "shape_zyx": [32, 32, 32],
                "known_fraction": 1.0,
                "positive_fraction_known": float(target.mean()),
                "has_baseline": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "audit"
    report_path = audit_voxel_checkpoint(
        checkpoint_path=checkpoint,
        patch_manifest=manifest,
        output_path=output,
        options=CheckpointAuditOptions(
            thresholds=(0.5,),
            device="cpu",
            num_workers=0,
            max_cpu_threads=4,
        ),
    )

    assert report_path == output / "index.html"
    assert report_path.is_file()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["schema"] == "crossres-voxel-checkpoint-audit-v3"
    assert report["patch_manifest"]["records"] == 1
    assert report["sweep"]["qualification_scroll"] == "PHerc0814"
    assert report["sweep"]["baseline_comparison_policy"] == "matched-rows-only"


def test_grid_context_assembly_is_pure_voxel_mosaic(tmp_path: Path) -> None:
    grid = tmp_path / "grid"
    raw_root = grid / "cubes_RAW"
    raw_root.mkdir(parents=True)
    chunk_size = 4
    for z_index, y_index, x_index in np.ndindex(3, 3, 3):
        origin = (
            z_index * chunk_size,
            y_index * chunk_size,
            x_index * chunk_size,
        )
        cube_id = f"z{origin[0]:05d}_y{origin[1]:05d}_x{origin[2]:05d}"
        value = z_index * 9 + y_index * 3 + x_index
        tifffile.imwrite(
            raw_root / f"{cube_id}.tif",
            np.full((chunk_size,) * 3, value, dtype=np.uint16),
        )

    context = assemble_raw_context(
        grid,
        (chunk_size,) * 3,
        chunk_size=chunk_size,
        halo=2,
    )

    assert context.shape == (8, 8, 8)
    assert np.all(context[2:6, 2:6, 2:6] == 13)
    assert context[0, 0, 0] == 0
    assert context[-1, -1, -1] == 26


def test_grid_inference_writes_auditable_checkpoint_provenance(
    tmp_path: Path,
) -> None:
    grid = tmp_path / "grid"
    raw_root = grid / "cubes_RAW"
    prediction_root = grid / "cubes_PRED"
    raw_root.mkdir(parents=True)
    prediction_root.mkdir()
    chunk_size = 8
    target_id = "z00008_y00008_x00008"
    for z_index, y_index, x_index in np.ndindex(3, 3, 3):
        origin = (
            z_index * chunk_size,
            y_index * chunk_size,
            x_index * chunk_size,
        )
        cube_id = f"z{origin[0]:05d}_y{origin[1]:05d}_x{origin[2]:05d}"
        tifffile.imwrite(
            raw_root / f"{cube_id}.tif",
            np.full((chunk_size,) * 3, 100 + z_index, dtype=np.uint16),
        )
    (grid / "manifest.json").write_text(
        json.dumps({"chunk_size": chunk_size}) + "\n", encoding="utf-8"
    )
    (prediction_root / "present.json").write_text(
        json.dumps([target_id, "z99992_y99992_x99992"]) + "\n",
        encoding="utf-8",
    )

    config = NNUNetConfig(preset="tiny-test")
    model = VoxelNNUNet(config)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "epoch": 3,
            "best_score": 0.75,
            "model_config": config.as_dict(),
            "model": model.state_dict(),
            "metrics": {
                "val": {"dice": 0.75, "loss_total": 0.25},
            },
        },
        checkpoint,
    )
    output = tmp_path / "corrected"
    infer_voxel_grid(
        source_grid=grid,
        checkpoint_path=checkpoint,
        output_path=output,
        threshold=0.4,
        halo=4,
        device_name="cpu",
        mirror_tta=False,
        max_cpu_threads=4,
        target_cube_ids=(target_id,),
    )

    prediction = tifffile.imread(output / "cubes_PRED" / f"{target_id}.tif")
    assert prediction.shape == (chunk_size,) * 3
    assert set(np.unique(prediction)) <= {0, 255}
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["kind"] == "crossres-voxel-grid-inference-v1"
    assert provenance["checkpoint"]["epoch"] == 3
    assert provenance["checkpoint"]["val_dice"] == pytest.approx(0.75)
    assert provenance["checkpoint"]["sha256"] == provenance["checkpoint_sha256"]
    assert provenance["options"]["threshold"] == pytest.approx(0.4)
    assert provenance["options"]["max_cpu_threads"] == 4
    assert provenance["options"]["selected_target_count"] == 1
    assert provenance["target_cube_ids"] == [target_id]

    filtered_output = tmp_path / "filtered"
    infer_voxel_grid(
        source_grid=grid,
        checkpoint_path=checkpoint,
        output_path=filtered_output,
        threshold=0.4,
        halo=4,
        device_name="cpu",
        mirror_tta=False,
        max_cpu_threads=4,
        skip_incomplete_context=True,
    )
    filtered_provenance = json.loads(
        (filtered_output / "provenance.json").read_text(encoding="utf-8")
    )
    assert filtered_provenance["options"]["skip_incomplete_context"] is True
    assert filtered_provenance["options"]["requested_target_count"] == 2
    assert filtered_provenance["options"]["selected_target_count"] == 1
    assert filtered_provenance["options"]["skipped_incomplete_context_count"] == 1
    assert filtered_provenance["target_cube_ids"] == [target_id]
    assert filtered_provenance["skipped_incomplete_context_ids"] == [
        "z99992_y99992_x99992"
    ]


def test_official_affine_graph_inverts_coarse_to_fine_edges() -> None:
    coarse_to_fine = [
        [4.0, 0.0, 0.0, 12.0],
        [0.0, 4.0, 0.0, -8.0],
        [0.0, 0.0, 4.0, 20.0],
    ]
    volumes = {
        "coarse": {
            "properties": {
                "transforms": [
                    {
                        "to_volume_id": "fine",
                        "transformation_matrix": coarse_to_fine,
                    }
                ]
            }
        },
        "fine": {"properties": {"transforms": None}},
    }
    fine_to_coarse = find_volume_affine(volumes, "fine", "coarse")
    expected = np.linalg.inv(
        np.asarray(
            [
                coarse_to_fine[0],
                coarse_to_fine[1],
                coarse_to_fine[2],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )
    assert np.allclose(fine_to_coarse, expected)


def test_official_affine_graph_links_coordinate_equivalent_rewindows() -> None:
    coarse_to_original = [
        [4.0, 0.0, 0.0, 12.0],
        [0.0, 4.0, 0.0, -8.0],
        [0.0, 0.0, 4.0, 20.0],
    ]
    shared = {
        "shape": [100, 200, 300],
        "pixel_size_um": 2.4,
        "left_handed_coordinates": False,
        "z_direction_is_top_to_bottom": None,
        "data_format": "uint8",
    }
    volumes = {
        "coarse": {
            "scan_id": "coarse-scan",
            "properties": {
                "shape": [25, 50, 75],
                "pixel_size_um": 9.6,
                "transforms": [
                    {
                        "to_volume_id": "fine-original",
                        "transformation_matrix": coarse_to_original,
                    }
                ],
            },
        },
        "fine-original": {
            "scan_id": "fine-scan",
            "properties": {**shared, "transforms": None},
        },
        "fine-rewindowed": {
            "scan_id": "fine-scan",
            "properties": {**shared, "transforms": None},
        },
        "fine-cropped": {
            "scan_id": "fine-scan",
            "properties": {
                **shared,
                "shape": [99, 200, 300],
                "transforms": None,
            },
        },
    }
    expected = np.linalg.inv(
        np.asarray(
            [
                coarse_to_original[0],
                coarse_to_original[1],
                coarse_to_original[2],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )
    assert np.allclose(
        find_volume_affine(volumes, "fine-rewindowed", "coarse"),
        expected,
    )
    with pytest.raises(ValueError, match="no transform chain"):
        find_volume_affine(volumes, "fine-cropped", "coarse")


def test_voxel_schema_rejects_legacy_point_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "old.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_id": "old",
                "scroll_id": "PHerc0139",
                "split": "train",
                "coarse": {},
                "fine": {},
                "surfaces": [{"coarse_tifxyz": "bad"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(VoxelSchemaError, match="legacy point manifests"):
        load_pair_manifest(manifest)


def test_dense_fine_voxels_are_forward_voxelized_without_splats() -> None:
    fine = np.zeros((16, 16, 16), dtype=np.uint8)
    fine[6:8, :, :] = 1
    field = DenseFieldSpec(volume="unused.npy", encoding="labels")
    support = ChunkSupport.from_field(field, fine)
    affine = (
        (0.5, 0.0, 0.0, 0.0),
        (0.0, 0.5, 0.0, 0.0),
        (0.0, 0.0, 0.5, 0.0),
    )
    target, stats = voxelize_fine_target_patch(
        fine, field, support, affine, (0, 0, 0), (8, 8, 8), validity_block=4
    )
    assert set(np.unique(target)) == {0, 1}
    assert stats["fine_positive_voxels"] == 512
    assert np.flatnonzero(target.any(axis=(1, 2))).tolist() == [3, 4]
    assert int((target == 1).sum()) == 128


def test_manual_label_ignore_voxels_stay_unknown_after_voxelization() -> None:
    fine = np.full((8, 8, 8), 2, dtype=np.uint8)
    fine[2:6, 2:6, 2:6] = 0
    fine[3, 2:6, 2:6] = 1
    field = DenseFieldSpec(
        volume="unused.npy",
        encoding="labels",
        positive_labels=(1,),
        ignore_labels=(2,),
    )
    support = ChunkSupport.from_field(field, fine)
    target, stats = voxelize_fine_target_patch(
        fine,
        field,
        support,
        IDENTITY_AFFINE,
        (0, 0, 0),
        (8, 8, 8),
        validity_block=4,
    )
    assert target[0, 0, 0] == 2
    assert target[2, 2, 2] == 0
    assert target[3, 2, 2] == 1
    assert stats["known_voxels"] == 64
    assert stats["positive_voxels"] == 16
    positive, known = dense_field_masks(fine, field)
    assert int(positive.sum()) == 16
    assert int(known.sum()) == 64


def test_sparse_chunk_validity_matches_full_inverse_reference() -> None:
    fine = np.full((24, 24, 24), 2, dtype=np.uint8)
    fine[:8, :8, :8] = 0
    fine[8:16, 8:16, 8:16] = 0
    fine[10:12, 8:16, 8:16] = 1
    grid = (3, 3, 3)
    coordinates = ((0, 0, 0), (1, 1, 1))
    present = np.asarray(
        [ChunkSupport._encode_static(coordinate, grid) for coordinate in coordinates],
        dtype=np.int64,
    )
    support = ChunkSupport(
        shape_zyx=fine.shape,
        chunks_zyx=(8, 8, 8),
        grid_zyx=grid,
        present_ids=present,
    )
    field = DenseFieldSpec(
        volume="unused.npy",
        encoding="labels",
        positive_labels=(1,),
        ignore_labels=(2,),
    )
    affine = (
        (0.0, 0.5, 0.0, 0.0),
        (0.5, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.5, 0.0),
    )
    origin = (0, 0, 0)
    shape = (12, 12, 12)
    reference = _valid_centers(
        fine,
        field,
        support,
        invert_affine(affine),
        origin,
        shape,
        block=4,
    )
    labels, _ = voxelize_fine_target_patch(
        fine,
        field,
        support,
        affine,
        origin,
        shape,
        validity_block=4,
    )
    assert np.array_equal(labels != 2, reference | (labels == 1))
    assert np.array_equal(
        affine_matrix(affine),
        np.asarray(
            (
                (0.0, 0.5, 0.0, 0.0),
                (0.5, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.5, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        ),
    )


def test_sparse_projection_cache_is_bit_exact_and_reused() -> None:
    fine = np.full((24, 24, 24), 2, dtype=np.uint8)
    fine[:8, :8, :8] = 0
    fine[3:5, :8, :8] = 1
    fine[8:16, 8:16, 8:16] = 0
    fine[10:12, 8:16, 8:16] = 1
    grid = (3, 3, 3)
    present = np.asarray(
        [
            ChunkSupport._encode_static(coordinate, grid)
            for coordinate in ((0, 0, 0), (1, 1, 1))
        ],
        dtype=np.int64,
    )
    support = ChunkSupport(
        shape_zyx=fine.shape,
        chunks_zyx=(8, 8, 8),
        grid_zyx=grid,
        present_ids=present,
    )
    field = DenseFieldSpec(
        volume="unused.npy",
        encoding="labels",
        positive_labels=(1,),
        ignore_labels=(2,),
    )
    affine = (
        (0.0, 0.5, 0.0, 0.0),
        (0.5, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.5, 0.0),
    )
    cache = SparseChunkProjectionCache(
        fine,
        field,
        support,
        affine,
        (12, 12, 12),
        max_entries=8,
    )
    for origin, shape in (((0, 0, 0), (12, 12, 12)), ((2, 2, 2), (8, 8, 8))):
        legacy, legacy_stats = voxelize_fine_target_patch(
            fine,
            field,
            support,
            affine,
            origin,
            shape,
            validity_block=4,
        )
        cached, cached_stats = voxelize_fine_target_patch(
            fine,
            field,
            support,
            affine,
            origin,
            shape,
            validity_block=4,
            projection_cache=cache,
        )
        assert np.array_equal(cached, legacy)
        assert cached_stats == legacy_stats
    assert cache.cache_info().hits > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_sparse_projection_cuda_is_bit_exact_with_cpu() -> None:
    fine = np.zeros((16, 16, 16), dtype=np.uint8)
    fine[2:14:2, 1:15:3, 3:13:2] = 7
    fine[5:8, 6:10, 4:12] = 7
    fine[6, 7:9, 5:11] = 2
    support = ChunkSupport(
        shape_zyx=fine.shape,
        chunks_zyx=(128, 128, 128),
        grid_zyx=(1, 1, 1),
        present_ids=np.asarray([0], dtype=np.int64),
    )
    field = DenseFieldSpec(
        volume="unused.npy",
        encoding="labels",
        positive_labels=(7,),
        ignore_labels=(2,),
    )
    affine = (
        (0.31, -0.07, 0.02, 2.4),
        (0.05, 0.28, -0.03, 1.2),
        (-0.02, 0.04, 0.30, 0.7),
    )
    cpu_cache = SparseChunkProjectionCache(
        fine,
        field,
        support,
        affine,
        (16, 16, 16),
        max_entries=8,
        enable_cuda_projection=False,
    )
    cuda_cache = SparseChunkProjectionCache(
        fine,
        field,
        support,
        affine,
        (16, 16, 16),
        max_entries=8,
    )

    cpu_projection = cpu_cache.get((0, 0, 0))
    cuda_projection = cuda_cache.get((0, 0, 0))
    assert cpu_cache.projection_backend == "cpu-float64"
    assert cuda_cache.projection_backend == "cuda-float64"
    assert cuda_projection.lower_zyx == cpu_projection.lower_zyx
    assert cuda_projection.shape_zyx == cpu_projection.shape_zyx
    assert cuda_projection.fine_positive_voxels == cpu_projection.fine_positive_voxels
    assert np.array_equal(cuda_projection.known_bits, cpu_projection.known_bits)
    assert np.array_equal(
        cuda_projection.foreground_bits,
        cpu_projection.foreground_bits,
    )

    cpu_labels, cpu_stats = voxelize_fine_target_patch(
        fine,
        field,
        support,
        affine,
        (0, 0, 0),
        (16, 16, 16),
        projection_cache=cpu_cache,
    )
    cuda_labels, cuda_stats = voxelize_fine_target_patch(
        fine,
        field,
        support,
        affine,
        (0, 0, 0),
        (16, 16, 16),
        projection_cache=cuda_cache,
    )
    assert np.array_equal(cuda_labels, cpu_labels)
    assert cuda_stats == cpu_stats


def test_sparse_projection_cache_coalesces_inflight_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fine = np.zeros((8, 8, 8), dtype=np.uint8)
    fine[3:5] = 1
    support = ChunkSupport(
        shape_zyx=fine.shape,
        chunks_zyx=fine.shape,
        grid_zyx=(1, 1, 1),
        present_ids=np.asarray([0], dtype=np.int64),
    )
    field = DenseFieldSpec(
        volume="unused.npy",
        encoding="labels",
        positive_labels=(1,),
    )
    cache = SparseChunkProjectionCache(
        fine,
        field,
        support,
        IDENTITY_AFFINE,
        fine.shape,
        max_entries=8,
    )
    original = cache._project
    calls = 0

    def slow_project(coordinate: tuple[int, int, int]) -> object:
        nonlocal calls
        calls += 1
        time.sleep(0.05)
        return original(coordinate)

    monkeypatch.setattr(cache, "_project", slow_project)
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(cache.get, [(0, 0, 0)] * 3))
    assert calls == 1
    assert results[0] is results[1] is results[2]
    assert cache.cache_info().waits == 2


def test_label_chunk_stats_limit_sampling_but_not_known_support(
    tmp_path: Path,
) -> None:
    inventory = tmp_path / "chunks.jsonl"
    inventory.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "kind": "chunk",
                        "relative_path": "0/0.0.0",
                        "positive_voxels": 0,
                    }
                ),
                json.dumps(
                    {
                        "kind": "chunk",
                        "relative_path": "0/0.0.1",
                        "positive_voxels": 7,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    class ChunkedArray:
        shape = (4, 4, 8)
        chunks = (4, 4, 4)
        dtype = np.dtype(np.uint8)

    field = DenseFieldSpec(
        volume="unused.zarr::0",
        encoding="labels",
        support=ChunkSupportSpec("present-chunks", inventory),
    )
    support = ChunkSupport.from_field(field, ChunkedArray())
    assert support.contains((0, 0, 0))
    assert support.contains((0, 0, 1))
    assert support.coordinates().tolist() == [[0, 0, 1]]


def test_zero_byte_label_placeholders_are_unknown_not_readable_support(
    tmp_path: Path,
) -> None:
    inventory = tmp_path / "chunks.jsonl"
    inventory.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "kind": "chunk",
                        "relative_path": "0/0.0.0",
                        "size": 0,
                    }
                ),
                json.dumps(
                    {
                        "kind": "chunk",
                        "relative_path": "0/0.0.1",
                        "size": 37,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    class ChunkedArray:
        shape = (4, 4, 8)
        chunks = (4, 4, 4)
        dtype = np.dtype(np.uint8)

    field = DenseFieldSpec(
        volume="unused.zarr::0",
        encoding="labels",
        support=ChunkSupportSpec("present-chunks", inventory),
    )
    support = ChunkSupport.from_field(field, ChunkedArray())
    assert not support.contains((0, 0, 0))
    assert support.contains((0, 0, 1))
    assert support.coordinates().tolist() == [[0, 0, 1]]


def test_teacher_candidates_require_a_fully_voxelized_context_neighborhood() -> None:
    coordinates = np.asarray(
        [(z, y, x) for z in range(5) for y in range(5) for x in range(5)],
        dtype=np.int64,
    )
    interior = interior_chunk_coordinates(
        coordinates,
        grid_zyx=(5, 5, 5),
        radius_zyx=(1, 1, 1),
    )
    assert interior.shape == (27, 3)
    assert set(map(tuple, interior)) == {
        (z, y, x) for z in range(1, 4) for y in range(1, 4) for x in range(1, 4)
    }
    assert centered_crop_slices((192, 192, 192), (128, 128, 128)) == (
        slice(32, 160),
        slice(32, 160),
        slice(32, 160),
    )
    assert centered_crop_slices((256, 256, 256), (128, 128, 128)) == (
        slice(64, 192),
        slice(64, 192),
        slice(64, 192),
    )


def test_carve_chunk_ids_decode_to_zyx(tmp_path: Path) -> None:
    support = tmp_path / "carve_selected_chunks.json"
    support.write_text(
        json.dumps(
            {
                "array_key": "0",
                "chunk_grid_zyx": [3, 4, 5],
                "selected_chunk_ids": [0, 19, 20, 59],
            }
        ),
        encoding="utf-8",
    )
    coordinates = load_present_chunk_coordinates(
        support, array_key="0", grid_zyx=(3, 4, 5)
    )
    assert coordinates.tolist() == [
        [0, 0, 0],
        [0, 3, 4],
        [1, 0, 0],
        [2, 3, 4],
    ]


def test_partial_teacher_mirror_requires_a_physical_local_support_snapshot(
    tmp_path: Path,
) -> None:
    store = tmp_path / "fine.zarr"
    array = store / "0"
    array.mkdir(parents=True)
    array_metadata_path = array / ".zarray"
    array_metadata_path.write_text(
        json.dumps(
            {
                "shape": [12, 12, 12],
                "chunks": [4, 4, 4],
                "dimension_separator": "/",
                "dtype": "|u1",
                "compressor": None,
            }
        ),
        encoding="utf-8",
    )
    identifiers: list[int] = []
    present_bytes = 0
    for z in range(3):
        for y in range(3):
            for x in range(3):
                chunk = array / str(z) / str(y) / str(x)
                chunk.parent.mkdir(parents=True, exist_ok=True)
                payload = bytes([z * 9 + y * 3 + x])
                chunk.write_bytes(payload)
                identifiers.append((z * 3 + y) * 3 + x)
                present_bytes += len(payload)
    support = store / "crossres_local_support.json"
    support.write_text(
        json.dumps(
            {
                "schema": "crossres-local-zarr-support-v1",
                "schema_version": 1,
                "zarr": str(store.resolve()),
                "array_key": "0",
                "shape_zyx": [12, 12, 12],
                "chunks_zyx": [4, 4, 4],
                "chunk_grid_zyx": [3, 3, 3],
                "dimension_separator": "/",
                "selected_chunk_ids": identifiers,
                "present_chunk_count": 27,
                "present_bytes": present_bytes,
                "context_voxels": 4,
                "context_radius_chunks_zyx": [1, 1, 1],
                "interior_chunk_count": 1,
            }
        ),
        encoding="utf-8",
    )

    present, interior, audit = validate_local_support_snapshot(
        input_path=store,
        array_key="0",
        support_path=support,
        shape_zyx=(12, 12, 12),
        chunks_zyx=(4, 4, 4),
        grid_zyx=(3, 3, 3),
        radius_zyx=(1, 1, 1),
        required_chunks=1,
    )
    assert present.shape == (27, 3)
    assert interior.tolist() == [[1, 1, 1]]
    assert audit["present_bytes"] == 27

    array_metadata = json.loads(array_metadata_path.read_text(encoding="utf-8"))
    array_metadata["chunks"] = [6, 4, 4]
    array_metadata_path.write_text(json.dumps(array_metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="local Zarr chunks"):
        validate_local_support_snapshot(
            input_path=store,
            array_key="0",
            support_path=support,
            shape_zyx=(12, 12, 12),
            chunks_zyx=(4, 4, 4),
            grid_zyx=(3, 3, 3),
            radius_zyx=(1, 1, 1),
            required_chunks=1,
        )
    array_metadata["chunks"] = [4, 4, 4]
    array_metadata_path.write_text(json.dumps(array_metadata), encoding="utf-8")

    (array / "1" / "1" / "1").unlink()
    with pytest.raises(ValueError, match="local-support chunks are missing"):
        validate_local_support_snapshot(
            input_path=store,
            array_key="0",
            support_path=support,
            shape_zyx=(12, 12, 12),
            chunks_zyx=(4, 4, 4),
            grid_zyx=(3, 3, 3),
            radius_zyx=(1, 1, 1),
            required_chunks=1,
        )


def test_native_teacher_instance_zscore_matches_villa_contract() -> None:
    image = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
    normalized = normalize_instance_zscore(image)
    assert normalized.dtype == np.float32
    assert float(normalized.mean()) == pytest.approx(0.0, abs=1.0e-6)
    assert float(normalized.std()) == pytest.approx(1.0, abs=1.0e-6)
    assert np.array_equal(
        normalize_instance_zscore(np.ones((2, 2, 2), dtype=np.uint8)),
        np.zeros((2, 2, 2), dtype=np.float32),
    )


def test_native_teacher_blend_grid_matches_global_nnunet_contract() -> None:
    steps = villa_sliding_window_steps(75784, 256, 0.5)
    assert steps[0] == 0
    assert steps[-1] == 75784 - 256
    assert len(steps) == 592
    assert max(np.diff(steps)) <= 128
    gaussian = villa_gaussian_map((16, 16, 16))
    assert gaussian.shape == (16, 16, 16)
    assert gaussian[8, 8, 8] == pytest.approx(1.0)
    assert gaussian[0, 0, 0] < gaussian[4, 4, 4] < gaussian[8, 8, 8]


def test_native_teacher_spatial_batching_preserves_blend_order() -> None:
    class DeterministicTeacher(torch.nn.Module):
        def full_resolution_logits(self, image: torch.Tensor) -> torch.Tensor:
            return torch.cat((-image, image), dim=1)

    teacher = LoadedTeacher(
        model=DeterministicTeacher().eval(),
        kind="deterministic-test",
        patch_shape_zyx=(4, 4, 4),
        required_divisor=1,
        normalization="instance_zscore",
        default_threshold=0.45,
        default_mirror_tta=False,
        tta_average_logits=True,
        provenance={},
    )
    volume = np.arange(8**3, dtype=np.uint16).reshape(8, 8, 8)
    arguments = {
        "teacher": teacher,
        "volume": volume,
        "volume_shape_zyx": (8, 8, 8),
        "output_origin_zyx": (2, 2, 2),
        "output_shape_zyx": (4, 4, 4),
        "step_size": 0.5,
        "device": torch.device("cpu"),
        "amp_dtype": torch.float32,
        "autocast_enabled": False,
        "mirror_tta": False,
    }
    serial, serial_origins = _predict_blended_probability(
        **arguments, inference_batch_size=1
    )
    batched, batched_origins = _predict_blended_probability(
        **arguments, inference_batch_size=2
    )
    assert len(serial_origins) == 27
    assert batched_origins == serial_origins
    np.testing.assert_array_equal(batched, serial)
    with pytest.raises(ValueError, match="inference_batch_size"):
        TeacherOptions(inference_batch_size=0).validate()


def test_native_teacher_blend_maps_uncovered_masked_zero_to_background() -> None:
    class UnreachableTeacher(torch.nn.Module):
        def full_resolution_logits(self, image: torch.Tensor) -> torch.Tensor:
            raise AssertionError("constant masked windows must not reach the model")

    teacher = LoadedTeacher(
        model=UnreachableTeacher().eval(),
        kind="masked-zero-test",
        patch_shape_zyx=(4, 4, 4),
        required_divisor=1,
        normalization="instance_zscore",
        default_threshold=0.45,
        default_mirror_tta=False,
        tta_average_logits=True,
        provenance={},
    )
    probability, origins = _predict_blended_probability(
        teacher=teacher,
        volume=np.zeros((8, 8, 8), dtype=np.uint16),
        volume_shape_zyx=(8, 8, 8),
        output_origin_zyx=(2, 2, 2),
        output_shape_zyx=(4, 4, 4),
        step_size=0.5,
        device=torch.device("cpu"),
        amp_dtype=torch.float32,
        autocast_enabled=False,
        mirror_tta=False,
        inference_batch_size=2,
    )
    assert origins == []
    np.testing.assert_array_equal(probability, np.zeros((4, 4, 4), dtype=np.float32))


def test_native_teacher_blend_masks_uncovered_zero_beside_valid_context() -> None:
    class DeterministicTeacher(torch.nn.Module):
        def full_resolution_logits(self, image: torch.Tensor) -> torch.Tensor:
            return torch.cat((-image, image), dim=1)

    teacher = LoadedTeacher(
        model=DeterministicTeacher().eval(),
        kind="masked-edge-test",
        patch_shape_zyx=(4, 4, 4),
        required_divisor=1,
        normalization="instance_zscore",
        default_threshold=0.45,
        default_mirror_tta=False,
        tta_average_logits=True,
        provenance={},
    )
    volume = np.zeros((8, 8, 8), dtype=np.uint16)
    volume[:, 6:, :] = np.arange(1, 8 * 2 * 8 + 1, dtype=np.uint16).reshape(8, 2, 8)
    probability, origins = _predict_blended_probability(
        teacher=teacher,
        volume=volume,
        volume_shape_zyx=(8, 8, 8),
        output_origin_zyx=(2, 2, 2),
        output_shape_zyx=(4, 4, 4),
        step_size=0.5,
        device=torch.device("cpu"),
        amp_dtype=torch.float32,
        autocast_enabled=False,
        mirror_tta=False,
        inference_batch_size=2,
    )
    assert origins
    np.testing.assert_array_equal(
        probability[:, :2, :], np.zeros((4, 2, 4), dtype=np.float32)
    )


def test_native_teacher_blend_still_rejects_uncovered_nonzero_ct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crossres_pred.voxel.teacher as teacher_module

    class DeterministicTeacher(torch.nn.Module):
        def full_resolution_logits(self, image: torch.Tensor) -> torch.Tensor:
            return torch.cat((-image, image), dim=1)

    teacher = LoadedTeacher(
        model=DeterministicTeacher().eval(),
        kind="nonzero-fail-closed-test",
        patch_shape_zyx=(4, 4, 4),
        required_divisor=1,
        normalization="instance_zscore",
        default_threshold=0.45,
        default_mirror_tta=False,
        tta_average_logits=True,
        provenance={},
    )
    monkeypatch.setattr(
        teacher_module,
        "villa_gaussian_map",
        lambda shape: np.zeros(shape, dtype=np.float32),
    )
    with pytest.raises(RuntimeError, match="nonzero-CT output voxels uncovered"):
        _predict_blended_probability(
            teacher=teacher,
            volume=np.arange(8**3, dtype=np.uint16).reshape(8, 8, 8),
            volume_shape_zyx=(8, 8, 8),
            output_origin_zyx=(2, 2, 2),
            output_shape_zyx=(4, 4, 4),
            step_size=0.5,
            device=torch.device("cpu"),
            amp_dtype=torch.float32,
            autocast_enabled=False,
            mirror_tta=False,
            inference_batch_size=2,
        )


def test_native_teacher_prediction_cache_preserves_blend_and_reuses_windows() -> None:
    class CountingTeacher(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def full_resolution_logits(self, image: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return torch.cat((-image, image), dim=1)

    model = CountingTeacher().eval()
    teacher = LoadedTeacher(
        model=model,
        kind="counting-test",
        patch_shape_zyx=(4, 4, 4),
        required_divisor=1,
        normalization="instance_zscore",
        default_threshold=0.45,
        default_mirror_tta=False,
        tta_average_logits=True,
        provenance={},
    )
    volume = np.arange(8**3, dtype=np.uint16).reshape(8, 8, 8)
    arguments = {
        "teacher": teacher,
        "volume": volume,
        "volume_shape_zyx": (8, 8, 8),
        "output_origin_zyx": (2, 2, 2),
        "output_shape_zyx": (4, 4, 4),
        "step_size": 0.5,
        "device": torch.device("cpu"),
        "amp_dtype": torch.float32,
        "autocast_enabled": False,
        "mirror_tta": False,
        "inference_batch_size": 2,
    }
    cache = _PredictionLogitCache(32)
    first, first_origins = _predict_blended_probability(
        **arguments, prediction_cache=cache
    )
    first_calls = model.calls
    second, second_origins = _predict_blended_probability(
        **arguments, prediction_cache=cache
    )
    assert first_calls > 0
    assert model.calls == first_calls
    assert cache.hits == len(first_origins)
    assert second_origins == first_origins
    np.testing.assert_array_equal(second, first)


def test_native_teacher_tile_order_is_deterministic_and_neighbor_local() -> None:
    coordinates = np.asarray(
        [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (8, 8, 8),
            (8, 8, 9),
        ],
        dtype=np.int64,
    )
    ordered = _candidate_execution_order(coordinates, seed=1203, tile_chunks=4)
    repeated = _candidate_execution_order(coordinates, seed=1203, tile_chunks=4)
    np.testing.assert_array_equal(ordered, repeated)
    tile_ids = [tuple(row) for row in ordered // 4]
    for tile_id in set(tile_ids):
        positions = [index for index, value in enumerate(tile_ids) if value == tile_id]
        assert positions == list(range(min(positions), max(positions) + 1))
    assert {tuple(row) for row in ordered} == {tuple(row) for row in coordinates}
    with pytest.raises(ValueError, match="prediction_cache_entries"):
        TeacherOptions(prediction_cache_entries=257).validate()


def test_native_teacher_legacy_identity_is_explicitly_serial() -> None:
    legacy = {"schema": "teacher", "options": {"chunks": 1}}
    serial = {
        "schema": "teacher",
        "options": {
            "chunks": 1,
            "inference_batch_size": 1,
            "allow_fewer_chunks": False,
        },
        "effective_chunks": 1,
    }
    batched = {
        "schema": "teacher",
        "options": {
            "chunks": 1,
            "inference_batch_size": 2,
            "allow_fewer_chunks": False,
        },
        "effective_chunks": 1,
    }
    assert _canonical_teacher_identity(legacy) == serial
    assert _canonical_teacher_identity(legacy) != batched
    cached = json.loads(json.dumps(serial))
    cached["options"]["prediction_cache_entries"] = 256
    assert _canonical_teacher_identity(cached) == serial


def test_teacher_state_replace_retries_transient_reader_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "teacher_state.json.tmp"
    destination = tmp_path / "teacher_state.json"
    source.write_text("new", encoding="utf-8")
    destination.write_text("old", encoding="utf-8")
    real_replace = os.replace
    calls = 0

    def flaky_replace(left: Path, right: Path) -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError("simulated Windows reader lock")
        real_replace(left, right)

    monkeypatch.setattr(os, "replace", flaky_replace)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    _replace_with_retry(source, destination)

    assert calls == 3
    assert destination.read_text(encoding="utf-8") == "new"


def test_teacher_materialization_validation_detects_chunk_corruption(
    tmp_path: Path,
) -> None:
    output = tmp_path / "teacher.zarr"
    chunks = (4, 4, 4)
    _initialize_target_zarr(
        output,
        shape_zyx=(8, 8, 8),
        chunks_zyx=chunks,
        attrs={"kind": "test"},
    )
    target = np.zeros(chunks, dtype=np.uint8)
    target[1:3, 1:3, 1:3] = 255
    chunk_path, encoded_bytes, encoded_sha256 = _write_chunk_atomic(
        output, (0, 0, 0), target
    )
    record = {
        "schema": "crossres-native-fine-teacher-chunk-v1",
        "chunk_zyx": [0, 0, 0],
        "origin_zyx": [0, 0, 0],
        "shape_zyx": list(chunks),
        "relative_path": "0/0/0/0",
        "encoded_bytes": encoded_bytes,
        "encoded_sha256": encoded_sha256,
        "positive_voxels": 8,
    }
    _atomic_json(output / "records" / "0_0_0.json", record)
    records = {(0, 0, 0): record}
    _write_inventory(output, records)
    _atomic_json(
        output / "teacher_state.json",
        {
            "state": "complete",
            "accepted": 1,
            "examined": 1,
            "identity": {
                "options": {"chunks": 1},
                "teacher_checkpoint_sha256": "checkpoint",
            },
        },
    )

    summary = validate_teacher_materialization(output)
    assert summary["validated_chunks"] == 1
    assert summary["positive_voxels"] == 8

    _atomic_json(
        output / "teacher_state.json",
        {
            "state": "complete",
            "accepted": 1,
            "examined": 3,
            "identity": {
                "effective_chunks": 3,
                "eligible_chunks": 3,
                "options": {
                    "chunks": 3,
                    "allow_fewer_chunks": True,
                    "max_candidates": 10,
                    "candidate_chunk_zyx": None,
                },
                "teacher_checkpoint_sha256": "checkpoint",
            },
        },
    )
    bounded = validate_teacher_materialization(output)
    assert bounded["requested_chunks"] == 3
    assert bounded["validated_chunks"] == 1
    assert bounded["examined_candidates"] == 3
    assert bounded["filtered_candidates"] == 2

    state = json.loads((output / "teacher_state.json").read_text(encoding="utf-8"))
    state["examined"] = 2
    _atomic_json(output / "teacher_state.json", state)
    with pytest.raises(ValueError, match="before exhausting"):
        validate_teacher_materialization(output)
    state["examined"] = 3
    _atomic_json(output / "teacher_state.json", state)

    damaged = bytearray(chunk_path.read_bytes())
    damaged[-1] ^= 1
    chunk_path.write_bytes(damaged)
    with pytest.raises(ValueError, match="SHA-256"):
        validate_teacher_materialization(output)


def _write_pair_manifest(tmp_path: Path) -> Path:
    coarse = np.full((64, 64, 64), 100, dtype=np.uint8)
    fine = np.zeros_like(coarse)
    fine[4::8, :, :] = 1
    np.save(tmp_path / "coarse.npy", coarse, allow_pickle=False)
    np.save(tmp_path / "fine.npy", fine, allow_pickle=False)
    manifest = tmp_path / "pairs.jsonl"
    row = {
        "schema": "crossres-voxel-pair-v1",
        "schema_version": 1,
        "record_id": "synthetic-dense",
        "scroll_id": "SyntheticTrain",
        "split": "train",
        "coarse": {
            "scan_id": "coarse",
            "voxel_um": 8.0,
            "image": "coarse.npy",
        },
        "fine": {
            "scan_id": "fine",
            "voxel_um": 2.0,
            "target": {
                "volume": "fine.npy",
                "encoding": "labels",
                "positive_labels": [1],
            },
            "to_coarse_affine_xyz": [list(row) for row in IDENTITY_AFFINE],
        },
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest


def _write_sparse_native_pair_manifest(tmp_path: Path) -> Path:
    coarse = np.full((64, 64, 64), 100, dtype=np.uint8)
    np.save(tmp_path / "coarse_sparse.npy", coarse, allow_pickle=False)

    store = tmp_path / "fine_sparse.zarr"
    array = store / "0"
    array.mkdir(parents=True)
    (store / ".zgroup").write_text('{"zarr_format":2}', encoding="utf-8")
    (store / ".zattrs").write_text("{}", encoding="utf-8")
    (array / ".zattrs").write_text("{}", encoding="utf-8")
    (array / ".zarray").write_text(
        json.dumps(
            {
                "zarr_format": 2,
                "shape": [64, 64, 64],
                "chunks": [32, 32, 32],
                "dtype": "|u1",
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
                "dimension_separator": ".",
            }
        ),
        encoding="utf-8",
    )
    coordinates = ((0, 0, 0), (1, 1, 1))
    positive_chunk = np.ones((32, 32, 32), dtype=np.uint8).tobytes(order="C")
    inventory_rows = []
    for coordinate in coordinates:
        relative = "0/" + ".".join(str(item) for item in coordinate)
        (store / relative).write_bytes(positive_chunk)
        inventory_rows.append(
            {
                "kind": "chunk",
                "relative_path": relative,
                "positive_voxels": 32**3,
            }
        )
    inventory = tmp_path / "fine_sparse_chunks.jsonl"
    inventory.write_text(
        "\n".join(json.dumps(row) for row in inventory_rows) + "\n",
        encoding="utf-8",
    )

    manifest = tmp_path / "pairs_sparse.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema": "crossres-voxel-pair-v1",
                "schema_version": 1,
                "record_id": "synthetic-native-teacher",
                "scroll_id": "SyntheticNative",
                "split": "train",
                "patch_count": 2,
                "supervision_source": "official-native-fine-teacher/test",
                "coarse": {
                    "scan_id": "coarse",
                    "voxel_um": 2.0,
                    "image": "coarse_sparse.npy",
                },
                "fine": {
                    "scan_id": "fine",
                    "voxel_um": 2.0,
                    "target": {
                        "volume": "fine_sparse.zarr::0",
                        "encoding": "labels",
                        "positive_labels": [1],
                        "support": {
                            "kind": "present-chunks",
                            "inventory": inventory.name,
                        },
                    },
                    "to_coarse_affine_xyz": [list(row) for row in IDENTITY_AFFINE],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_native_teacher_fine_ct_gate_removes_boundary_chunks_from_known_support(
    tmp_path: Path,
) -> None:
    manifest = _write_sparse_native_pair_manifest(tmp_path)
    records = tmp_path / "fine_sparse.zarr" / "records"
    records.mkdir()
    for coordinate, ct_nonzero_fraction in (
        ((0, 0, 0), 0.60),
        ((1, 1, 1), 1.0),
    ):
        (records / ("_".join(str(item) for item in coordinate) + ".json")).write_text(
            json.dumps(
                {
                    "schema": "crossres-native-fine-teacher-chunk-v1",
                    "chunk_zyx": list(coordinate),
                    "ct_nonzero_fraction": ct_nonzero_fraction,
                }
            ),
            encoding="utf-8",
        )

    record = load_pair_manifest(manifest)[0]
    fine = open_volume(record.fine.target.volume)
    support = ChunkSupport.from_field(record.fine.target, fine)
    filtered, audit = _quality_filter_native_teacher_support(
        record,
        support,
        PrepareOptions(native_teacher_min_fine_ct_nonzero_fraction=0.95),
    )

    assert audit is not None
    assert audit.local_records_applied
    assert audit.chunks_before == 2
    assert audit.chunks_after == 1
    assert audit.chunks_excluded == 1
    assert not filtered.contains((0, 0, 0))
    assert filtered.contains((1, 1, 1))
    assert filtered.coordinates().tolist() == [[1, 1, 1]]

    labels, stats = voxelize_fine_target_patch(
        fine,
        record.fine.target,
        filtered,
        record.fine.to_coarse_affine_xyz,
        (0, 0, 0),
        (64, 64, 64),
        validity_block=16,
    )
    assert np.all(labels[:32, :32, :32] == 2)
    assert np.all(labels[32:, 32:, 32:] == 1)
    assert stats["chunks_read"] == 1


def test_patch_preparation_is_dense_and_resumable(tmp_path: Path) -> None:
    manifest = _write_pair_manifest(tmp_path)
    output = tmp_path / "patches"
    options = PrepareOptions(
        patches_per_record=2,
        patch_shape_zyx=(32, 32, 32),
        min_known_fraction=0.9,
        min_positive_voxels=16,
        attempts_per_patch=4,
        validity_block=16,
    )
    patch_manifest = prepare_patch_corpus(
        pair_manifest=manifest, output_path=output, options=options
    )
    rows = load_patch_manifest(patch_manifest)
    assert len(rows) == 2
    assert (
        prepare_patch_corpus(
            pair_manifest=manifest, output_path=output, options=options
        )
        == patch_manifest
    )
    sample = VoxelPatchDataset(patch_manifest, split="train")[0]
    assert sample["image"].shape == (1, 32, 32, 32)
    assert sample["target"].shape == (1, 32, 32, 32)
    assert set(torch.unique(sample["target"]).tolist()) == {0, 1}
    summary = validate_patch_corpus(
        patch_manifest,
        expected_count=2,
        expected_split_counts={"train": 2, "val": 0, "test": 0},
        expected_train_scrolls={"SyntheticTrain"},
        expected_val_scrolls=set(),
        expected_test_scrolls=set(),
        expected_train_scroll_counts={"SyntheticTrain": 2},
        expected_val_scroll_counts={},
        expected_test_scroll_counts={},
        expected_record_counts={"synthetic-dense": 2},
        expected_source_corpora=1,
        workers=1,
        max_cpu_threads=4,
    )
    assert summary["patches"] == 2
    assert summary["source_corpora"] == 1
    assert summary["test_scrolls"] == []
    assert summary["scroll_counts"]["train"] == {"SyntheticTrain": 2}
    assert summary["record_counts"] == {"synthetic-dense": 2}

    with pytest.raises(ValueError, match="train scroll counts"):
        validate_patch_corpus(
            patch_manifest,
            expected_train_scroll_counts={"SyntheticTrain": 1},
            workers=1,
            max_cpu_threads=4,
        )

    with pytest.raises(ValueError, match="record counts"):
        validate_patch_corpus(
            patch_manifest,
            expected_record_counts={"synthetic-dense": 1},
            workers=1,
            max_cpu_threads=4,
        )

    first_patch = load_patch_manifest(patch_manifest)[0].path
    damaged = bytearray(first_patch.read_bytes())
    damaged[-1] ^= 1
    first_patch.write_bytes(damaged)
    with pytest.raises(ValueError, match="SHA-256"):
        validate_patch_corpus(
            patch_manifest,
            workers=1,
            max_cpu_threads=4,
        )


def test_prepare_manifest_repairs_only_a_torn_final_row(tmp_path: Path) -> None:
    manifest = tmp_path / "patches.jsonl"
    first = {"patch_id": "first", "path": "patches/first.npz"}
    manifest.write_bytes(json.dumps(first).encode() + b'\n{"patch_id":"torn')

    rows = _load_existing_rows(manifest)

    assert rows == {"first": first}
    assert manifest.read_bytes() == json.dumps(first).encode() + b"\n"


def test_prepare_manifest_discards_invalid_nondurable_archive_tail(
    tmp_path: Path,
) -> None:
    output = tmp_path / "corpus"
    patches = output / "patches"
    patches.mkdir(parents=True)
    good = patches / "good.npz"
    bad = patches / "bad.npz"
    good.write_bytes(b"good")
    bad.write_bytes(b"bad")
    rows = [
        {
            "patch_id": "good",
            "path": "patches/good.npz",
            "archive_bytes": 4,
            "archive_sha256": hashlib.sha256(b"good").hexdigest(),
        },
        {
            "patch_id": "bad",
            "path": "patches/bad.npz",
            "archive_bytes": 3,
            "archive_sha256": hashlib.sha256(b"different").hexdigest(),
        },
    ]
    manifest = output / "patches.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    recovered = _load_existing_rows(
        manifest,
        output=output,
        durable_completed=1,
    )

    assert list(recovered) == ["good"]
    assert manifest.read_text(encoding="utf-8") == json.dumps(rows[0]) + "\n"


def test_prepare_manifest_accepts_hash_valid_nondurable_archive_tail(
    tmp_path: Path,
) -> None:
    output = tmp_path / "corpus"
    patches = output / "patches"
    patches.mkdir(parents=True)
    rows = []
    for patch_id in ("durable", "tail"):
        payload = patch_id.encode()
        archive = patches / f"{patch_id}.npz"
        archive.write_bytes(payload)
        rows.append(
            {
                "patch_id": patch_id,
                "path": f"patches/{patch_id}.npz",
                "archive_bytes": len(payload),
                "archive_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = output / "patches.jsonl"
    original = "".join(json.dumps(row) + "\n" for row in rows)
    manifest.write_text(original, encoding="utf-8")

    recovered = _load_existing_rows(
        manifest,
        output=output,
        durable_completed=1,
    )

    assert list(recovered) == ["durable", "tail"]
    assert manifest.read_text(encoding="utf-8") == original


def test_prepare_manifest_rejects_hash_invalid_durable_archive(
    tmp_path: Path,
) -> None:
    output = tmp_path / "corpus"
    patches = output / "patches"
    patches.mkdir(parents=True)
    archive = patches / "durable.npz"
    archive.write_bytes(b"damaged")
    row = {
        "patch_id": "durable",
        "path": "patches/durable.npz",
        "archive_bytes": 7,
        "archive_sha256": hashlib.sha256(b"original").hexdigest(),
    }
    manifest = output / "patches.jsonl"
    original = json.dumps(row) + "\n"
    manifest.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="durable archive"):
        _load_existing_rows(
            manifest,
            output=output,
            durable_completed=1,
        )

    assert manifest.read_text(encoding="utf-8") == original


def test_prepare_manifest_rejects_malformed_durable_final_row(
    tmp_path: Path,
) -> None:
    output = tmp_path / "corpus"
    output.mkdir()
    manifest = output / "patches.jsonl"
    original = b'{"patch_id":"durable"'
    manifest.write_bytes(original)

    with pytest.raises(ValueError, match="invalid durable manifest row"):
        _load_existing_rows(
            manifest,
            output=output,
            durable_completed=1,
        )

    assert manifest.read_bytes() == original


def test_sparse_native_preparation_anchors_each_patch_to_distinct_support(
    tmp_path: Path,
) -> None:
    manifest = _write_sparse_native_pair_manifest(tmp_path)
    output = tmp_path / "sparse_patches"
    patch_manifest = prepare_patch_corpus(
        pair_manifest=manifest,
        output_path=output,
        options=PrepareOptions(
            patches_per_record=2,
            patch_shape_zyx=(32, 32, 32),
            min_known_fraction=0.20,
            native_teacher_min_known_fraction=0.20,
            min_positive_voxels=16,
            attempts_per_patch=2,
            selection_candidates=1,
            validity_block=16,
        ),
    )
    raw_rows = [json.loads(line) for line in patch_manifest.read_text().splitlines()]
    anchors = {tuple(row["support_anchor_chunk_zyx"]) for row in raw_rows}
    assert anchors == {(0, 0, 0), (1, 1, 1)}
    assert {row["support_anchor_pool_size"] for row in raw_rows} == {2}
    assert {
        tuple(
            tuple(coordinate)
            for coordinate in row["support_anchor_candidate_chunks_zyx"]
        )
        for row in raw_rows
    } == {((0, 0, 0),), ((1, 1, 1),)}
    assert {row["acceptance_min_known_fraction"] for row in raw_rows} == {0.20}
    assert {row["native_teacher_min_fine_ct_nonzero_fraction"] for row in raw_rows} == {
        0.95
    }
    assert {row["native_teacher_fine_ct_quality_gate_applied"] for row in raw_rows} == {
        False
    }
    assert {
        row["native_teacher_support_chunks_before_quality_gate"] for row in raw_rows
    } == {2}
    assert {
        row["native_teacher_support_chunks_after_quality_gate"] for row in raw_rows
    } == {2}
    assert {
        row["native_teacher_support_chunks_excluded_by_quality_gate"]
        for row in raw_rows
    } == {0}
    assert {row["preparation_version"] for row in raw_rows} == {
        PATCH_PREPARATION_VERSION
    }
    summary = validate_patch_corpus(
        patch_manifest,
        expected_count=2,
        expected_source_corpora=1,
        workers=1,
        max_cpu_threads=4,
    )
    assert summary["native_teacher_anchor_records"] == {
        "synthetic-native-teacher": {
            "patches": 2,
            "anchor_pool_size": 2,
            "unique_anchors": 2,
            "primary_anchors": 2,
            "primary_anchor_retained_fraction": 1.0,
            "registration_filtered": False,
            "candidate_anchors": 2,
            "candidate_evaluations": 2,
        }
    }

    # A scheduled primary may fail acceptance at a masked coarse boundary.
    # Exhausted pools are complete when every primary was attempted and any
    # missing selected primary has an explicit fallback substitution.
    first_anchor = raw_rows[0]["support_anchor_chunk_zyx"]
    second_anchor = raw_rows[1]["support_anchor_chunk_zyx"]
    raw_rows[1]["support_anchor_chunk_zyx"] = first_anchor
    raw_rows[1]["support_anchor_candidate_chunks_zyx"] = [
        second_anchor,
        first_anchor,
    ]
    patch_manifest.write_text(
        "\n".join(json.dumps(row) for row in raw_rows) + "\n", encoding="utf-8"
    )
    fallback_summary = validate_patch_corpus(
        patch_manifest,
        expected_count=2,
        expected_source_corpora=1,
        workers=1,
        max_cpu_threads=4,
    )
    assert fallback_summary["native_teacher_anchor_records"][
        "synthetic-native-teacher"
    ] == {
        "patches": 2,
        "anchor_pool_size": 2,
        "unique_anchors": 1,
        "primary_anchors": 2,
        "primary_anchor_retained_fraction": 1.0,
        "registration_filtered": False,
        "candidate_anchors": 2,
        "candidate_evaluations": 3,
    }

    # Registration filtering may remove every occurrence of a scheduled primary
    # after the original finite-anchor schedule has already been materialized.
    # Preserve the declared pool while reporting the retained subset explicitly.
    for row in raw_rows:
        row["support_anchor_chunk_zyx"] = first_anchor
        row["support_anchor_candidate_chunks_zyx"] = [first_anchor]
        row["registration_decision"] = {
            "contract": "crossres-local-ct-translation-l0-v1",
            "method": "identity",
            "registration_manifest_sha256": "a" * 64,
            "shift_coarse_zyx": [0, 0, 0],
        }
        row["registered_source_quality_gate"] = {
            "accepted": True,
            "ct_nonzero_fraction": row["ct_nonzero_fraction"],
            "minimum_ct_nonzero_fraction": 0.95,
            "fine_support_anchor_contained": True,
            "fine_support_anchor_local_center_zyx": None,
            "source_shape_zyx": None,
        }
    patch_manifest.write_text(
        "\n".join(json.dumps(row) for row in raw_rows) + "\n", encoding="utf-8"
    )
    filtered_summary = validate_patch_corpus(
        patch_manifest,
        expected_count=2,
        expected_source_corpora=1,
        workers=1,
        max_cpu_threads=4,
    )
    assert filtered_summary["native_teacher_anchor_records"][
        "synthetic-native-teacher"
    ] == {
        "patches": 2,
        "anchor_pool_size": 2,
        "unique_anchors": 1,
        "primary_anchors": 1,
        "primary_anchor_retained_fraction": 0.5,
        "registration_filtered": True,
        "candidate_anchors": 1,
        "candidate_evaluations": 2,
    }
    for row in raw_rows:
        row.pop("registration_decision")
        row.pop("registered_source_quality_gate")
    raw_rows[1]["support_anchor_chunk_zyx"] = second_anchor
    raw_rows[1]["support_anchor_candidate_chunks_zyx"] = [second_anchor]

    raw_rows[0]["acceptance_min_known_fraction"] = 0.21
    patch_manifest.write_text(
        "\n".join(json.dumps(row) for row in raw_rows) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="acceptance known-fraction gate differs"):
        validate_patch_corpus(
            patch_manifest,
            workers=1,
            max_cpu_threads=4,
        )
    raw_rows[0]["acceptance_min_known_fraction"] = 0.20

    raw_rows[0]["native_teacher_min_fine_ct_nonzero_fraction"] = 0.94
    patch_manifest.write_text(
        "\n".join(json.dumps(row) for row in raw_rows) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="fine-CT quality gate differs"):
        validate_patch_corpus(
            patch_manifest,
            workers=1,
            max_cpu_threads=4,
        )
    raw_rows[0]["native_teacher_min_fine_ct_nonzero_fraction"] = 0.95

    raw_rows[1]["support_anchor_chunk_zyx"] = raw_rows[0]["support_anchor_chunk_zyx"]
    raw_rows[1]["support_anchor_candidate_chunks_zyx"] = [
        raw_rows[0]["support_anchor_chunk_zyx"]
    ]
    patch_manifest.write_text(
        "\n".join(json.dumps(row) for row in raw_rows) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="primary support anchors"):
        validate_patch_corpus(
            patch_manifest,
            workers=1,
            max_cpu_threads=4,
        )

    raw_rows[1]["support_anchor_chunk_zyx"] = [1, 1, 1]
    raw_rows[1]["support_anchor_candidate_chunks_zyx"] = [[0, 0, 0]]
    patch_manifest.write_text(
        "\n".join(json.dumps(row) for row in raw_rows) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must belong to its candidate anchors"):
        validate_patch_corpus(
            patch_manifest,
            workers=1,
            max_cpu_threads=4,
        )


@pytest.mark.torch
def test_tiny_nnunet_deep_supervision_trains_dense_labels() -> None:
    model = VoxelNNUNet(NNUNetConfig(preset="tiny-test"))
    image = torch.randn(1, 1, 32, 32, 32)
    target = torch.zeros(1, 1, 32, 32, 32, dtype=torch.long)
    target[:, :, 12:15] = 1
    target[:, :, :2] = 2
    outputs = model(image)
    loss, components = deep_supervision_loss(outputs, target)
    loss.backward()
    assert [tuple(output.shape[-3:]) for output in outputs] == [
        (32, 32, 32),
        (16, 16, 16),
    ]
    assert float(loss.detach()) > 0
    assert set(components) == {
        "total",
        "cross_entropy",
        "dice_loss",
        "medial_recall_loss",
        "medial_recall_ds0",
        "medial_recall_ds1",
        "separation_loss",
        "separation_ds0",
        "separation_ds1",
        "m7_anchor_kl",
        "m7_preservation_loss",
        "pinned_axial_loss",
        "pinned_axial_groups",
        "pinned_axial_target_voxels",
        "dynamic_medial_connectivity_loss",
        "dynamic_medial_connectivity_events",
        "dynamic_medial_connectivity_targets",
        "dynamic_medial_connectivity_bottleneck",
    }
    metrics = segmentation_metrics(outputs[0], target)
    assert 0 <= metrics["dice"] <= 1


@pytest.mark.torch
def test_sparse_ce_equal_weights_samples_not_known_voxels() -> None:
    logits = torch.zeros((2, 2, 1, 1, 100))
    target = torch.full((2, 1, 1, 100), 2, dtype=torch.long)

    # One difficult known voxel in the sparse sample.
    target[0, 0, 0, 0] = 1
    logits[0, 0, 0, 0, 0] = 5.0
    logits[0, 1, 0, 0, 0] = -5.0

    # One hundred easy known voxels in the dense sample.
    target[1] = 0
    logits[1, 0] = 5.0
    logits[1, 1] = -5.0

    combined = dice_ce_loss(logits, target)
    sparse = dice_ce_loss(logits[:1], target[:1])
    dense = dice_ce_loss(logits[1:], target[1:])
    assert combined.cross_entropy == pytest.approx(
        (sparse.cross_entropy + dense.cross_entropy) / 2
    )
    # A batch-wide voxel mean would be dominated by the 100 easy voxels.
    assert float(combined.cross_entropy) > 4.0


@pytest.mark.torch
def test_unknown_only_sample_does_not_dilute_sparse_loss() -> None:
    logits = torch.zeros((2, 2, 1, 2, 2))
    target = torch.full((2, 1, 2, 2), 2, dtype=torch.long)
    target[0, 0, 0, 0] = 1
    combined = dice_ce_loss(logits, target)
    known_only = dice_ce_loss(logits[:1], target[:1])
    assert combined.cross_entropy == pytest.approx(known_only.cross_entropy)
    assert combined.dice == pytest.approx(known_only.dice)
