from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .inference import load_voxel_checkpoint, predict_roi, write_prediction_zarr
from .io import open_volume, split_volume_spec
from .loss import VoxelLossOptions
from .prepare import PrepareOptions, prepare_patch_corpus
from .registration import ChunkSupport
from .resources import assert_cuda_power_limit, configure_cpu_budget
from .schema import load_pair_manifest
from .teacher import TeacherOptions, materialize_teacher
from .train import DEFAULT_VALIDATION_THRESHOLDS, TrainOptions, train_model


def _shape(value: list[int]) -> tuple[int, int, int]:
    result = tuple(int(item) for item in value)
    if len(result) != 3 or any(item <= 0 for item in result):
        raise ValueError("shape requires three positive integers")
    return result  # type: ignore[return-value]


def _audit(args: argparse.Namespace) -> int:
    rows: list[dict[str, object]] = []
    for record in load_pair_manifest(args.pairs):
        coarse_path, _ = split_volume_spec(record.coarse.image)
        target_path, _ = split_volume_spec(record.fine.target.volume)
        coarse = open_volume(record.coarse.image)
        target = open_volume(record.fine.target.volume)
        support_count: int | str = "not-loaded"
        if args.load_support:
            support = ChunkSupport.from_field(record.fine.target, target)
            support_count = (
                int(support.present_ids.size)
                if support.present_ids is not None
                else "all"
            )
        rows.append(
            {
                "record_id": record.record_id,
                "scroll_id": record.scroll_id,
                "split": record.split,
                "coarse_image": str(coarse_path),
                "coarse_shape_zyx": list(coarse.shape),
                "fine_target": str(target_path),
                "fine_shape_zyx": list(target.shape),
                "support_chunks": support_count,
                "expected_scale": record.expected_linear_scale,
                "measured_scales": record.measured_linear_scales,
                "has_baseline": record.coarse.baseline is not None,
            }
        )
    print(json.dumps(rows, indent=2))
    return 0


def _prepare(args: argparse.Namespace) -> int:
    manifest = prepare_patch_corpus(
        pair_manifest=args.pairs,
        output_path=args.output,
        options=PrepareOptions(
            patches_per_record=args.patches_per_record,
            patch_shape_zyx=_shape(args.patch_shape),
            seed=args.seed,
            min_known_fraction=args.min_known_fraction,
            native_teacher_min_known_fraction=(args.native_teacher_min_known_fraction),
            native_teacher_min_fine_ct_nonzero_fraction=(
                args.native_teacher_min_fine_ct_nonzero_fraction
            ),
            min_positive_voxels=args.min_positive_voxels,
            min_ct_nonzero_fraction=args.min_ct_nonzero_fraction,
            attempts_per_patch=args.attempts_per_patch,
            selection_candidates=args.selection_candidates,
            pathology_fraction=args.pathology_fraction,
            positive_density_fraction=args.positive_density_fraction,
            validity_block=args.validity_block,
            projection_cache_entries=args.projection_cache_entries,
            max_cpu_threads=args.max_cpu_threads,
        ),
    )
    print(manifest)
    return 0


def _train(args: argparse.Namespace) -> int:
    checkpoint = train_model(
        patch_manifest=args.patches,
        validation_patch_manifest=args.validation_patches,
        output_path=args.output,
        options=TrainOptions(
            epochs=args.epochs,
            batch_size=args.batch_size,
            accumulate=args.accumulate,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            optimizer=args.optimizer,
            adamw_beta1=args.adamw_beta1,
            adamw_beta2=args.adamw_beta2,
            adamw_eps=args.adamw_eps,
            m7_trust_region_relative_l2=args.m7_trust_region_relative_l2,
            loss_options=VoxelLossOptions(
                cross_entropy_weight=args.loss_ce_weight,
                dice_weight=args.loss_dice_weight,
                medial_recall_weight=args.loss_medial_recall_weight,
                separation_weight=args.loss_separation_weight,
                separation_radius=args.loss_separation_radius,
                separation_max_teacher_q=args.loss_separation_max_teacher_q,
                m7_anchor_weight=args.loss_m7_anchor_weight,
                m7_anchor_known_agreement=(args.loss_m7_anchor_known_agreement),
                m7_anchor_confident_agreement=(args.loss_m7_anchor_confident_agreement),
                m7_anchor_unknown_corridor_radius=(
                    args.loss_m7_anchor_unknown_corridor_radius
                ),
                m7_preservation_weight=args.loss_m7_preservation_weight,
                m7_preservation_radius=args.loss_m7_preservation_radius,
                m7_preservation_anchor_threshold=(
                    args.loss_m7_preservation_anchor_threshold
                ),
                m7_preservation_soft_floor=(args.loss_m7_preservation_soft_floor),
                pinned_axial_weight=args.loss_pinned_axial_weight,
                pinned_axial_probability_floor=(
                    args.loss_pinned_axial_probability_floor
                ),
                pinned_axial_bottom_fraction=(args.loss_pinned_axial_bottom_fraction),
                dynamic_medial_connectivity_weight=(
                    args.loss_dynamic_medial_connectivity_weight
                ),
                dynamic_medial_connectivity_probability_floor=(
                    args.loss_dynamic_medial_connectivity_probability_floor
                ),
                dynamic_medial_connectivity_steps=(
                    args.loss_dynamic_medial_connectivity_steps
                ),
            ),
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
            amp=not args.no_amp,
            amp_dtype=args.amp_dtype,
            preset=args.preset,
            pretrained_m7_checkpoint=args.m7_checkpoint,
            pinned_medial_bridge_state=args.pinned_medial_bridge_state,
            dynamic_medial_connectivity_state=(args.dynamic_medial_connectivity_state),
            final_fit=args.final_fit,
            allow_spatial_validation=args.allow_spatial_validation,
            max_cpu_threads=args.max_cpu_threads,
            samples_per_epoch=args.samples_per_epoch,
            max_train_samples=args.max_train_samples,
            lr_schedule=args.lr_schedule,
            lr_floor_ratio=args.lr_floor_ratio,
            warmup_samples=args.warmup_samples,
            stratified_sampling=args.stratified_sampling,
            train_augmentation=args.train_augmentation,
            validation_thresholds=tuple(args.validation_thresholds),
            minimum_scroll_gain=args.minimum_scroll_gain,
            checkpoint_min_delta=args.checkpoint_min_delta,
            early_stopping_patience=args.early_stopping_patience,
        ),
        resume=args.resume,
        snapshot_samples=tuple(args.snapshot_samples),
    )
    print(checkpoint)
    return 0


def _audit_checkpoint(args: argparse.Namespace) -> int:
    from .checkpoint_audit import (
        CheckpointAuditOptions,
        audit_voxel_checkpoint,
    )

    output = audit_voxel_checkpoint(
        checkpoint_path=args.checkpoint,
        patch_manifest=args.patches,
        output_path=args.output,
        options=CheckpointAuditOptions(
            split=args.split,
            thresholds=tuple(args.thresholds),
            qualification_scroll=args.qualification_scroll,
            device=args.device,
            amp_dtype=args.amp_dtype,
            mirror_tta=args.tta,
            num_workers=args.num_workers,
            max_cpu_threads=args.max_cpu_threads,
        ),
    )
    print(output)
    return 0


def _infer_grid(args: argparse.Namespace) -> int:
    from .grid_inference import infer_voxel_grid

    output = infer_voxel_grid(
        source_grid=args.source,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        threshold=args.threshold,
        halo=args.halo,
        device_name=args.device,
        amp_dtype_name=args.amp_dtype,
        mirror_tta=args.tta,
        max_cpu_threads=args.max_cpu_threads,
        target_cube_ids=args.target_cubes,
        skip_incomplete_context=args.skip_incomplete_context,
    )
    print(output)
    return 0


def _infer(args: argparse.Namespace) -> int:
    configure_cpu_budget(args.max_cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but this Python environment has CPU-only torch"
        )
    assert_cuda_power_limit(device)
    model, _ = load_voxel_checkpoint(args.checkpoint, device=device)
    volume = open_volume(args.volume)
    bounds = (
        (args.bounds[0], args.bounds[1]),
        (args.bounds[2], args.bounds[3]),
        (args.bounds[4], args.bounds[5]),
    )
    amp_dtype = (
        torch.bfloat16 if args.amp_dtype in {"auto", "bfloat16"} else torch.float16
    )
    probability = predict_roi(
        model,
        volume,
        bounds_zyx=bounds,
        device=device,
        patch_shape_zyx=_shape(args.patch_shape),
        overlap=args.overlap,
        amp_dtype=amp_dtype,
        autocast_enabled=not args.no_amp,
        mirror_tta=not args.no_tta,
    )
    output = write_prediction_zarr(
        args.output,
        probability,
        bounds_zyx=bounds,
        checkpoint=args.checkpoint,
        threshold=args.threshold,
    )
    print(output)
    return 0


def _mine_pathology(args: argparse.Namespace) -> int:
    from .pathology_mining import (
        M7PathologyMiningOptions,
        mine_m7_pathology,
    )

    summary = mine_m7_pathology(
        patch_manifests=args.patches,
        output_path=args.output,
        m7_checkpoint=args.m7_checkpoint,
        corpus_plan=args.corpus_plan,
        options=M7PathologyMiningOptions(
            batch_size=args.batch_size,
            device=args.device,
            amp_dtype=args.amp_dtype,
            threshold=args.threshold,
            splits=tuple(args.splits or ("train", "val")),
            max_cpu_threads=args.max_cpu_threads,
            max_patches=args.max_patches,
            expected_count=args.expected_count,
        ),
    )
    print(json.dumps(summary, indent=2))
    return 0


def _materialize_teacher(args: argparse.Namespace) -> int:
    output = materialize_teacher(
        fine_volume=args.fine_volume,
        fine_support_inventory=args.fine_support_inventory,
        output_path=args.output,
        teacher_checkpoint=args.teacher_checkpoint,
        villa_source=args.villa_source,
        options=TeacherOptions(
            chunks=args.chunks,
            allow_fewer_chunks=args.allow_fewer_chunks,
            seed=args.seed,
            input_shape_zyx=_shape(args.input_shape),
            threshold=args.threshold,
            min_positive_voxels=args.min_positive_voxels,
            min_ct_nonzero_fraction=args.min_ct_nonzero_fraction,
            max_candidates=args.max_candidates,
            device=args.device,
            amp_dtype=args.amp_dtype,
            mirror_tta=args.tta,
            sliding_blend=args.sliding_blend,
            sliding_step_size=args.sliding_step_size,
            inference_batch_size=args.inference_batch_size,
            candidate_tile_chunks=args.candidate_tile_chunks,
            prediction_cache_entries=args.prediction_cache_entries,
            candidate_chunk_zyx=(
                tuple(args.candidate_chunk) if args.candidate_chunk else None
            ),
            max_cpu_threads=args.max_cpu_threads,
        ),
    )
    print(output)
    return 0


def _audit_teacher(args: argparse.Namespace) -> int:
    from .teacher_audit import TeacherAuditOptions, audit_materialized_teacher

    output = audit_materialized_teacher(
        fine_volume=args.fine_volume,
        candidate_volume=args.candidate_volume,
        reference_volume=args.reference_volume,
        output_path=args.output,
        options=TeacherAuditOptions(
            max_records=args.max_records,
            slices_per_axis=args.slices_per_axis,
            tolerance_voxels=args.tolerance_voxels,
            max_cpu_threads=args.max_cpu_threads,
        ),
    )
    print(output)
    return 0


def _validate_teacher(args: argparse.Namespace) -> int:
    from .teacher import validate_teacher_materialization

    summary = validate_teacher_materialization(args.output)
    print(json.dumps(summary, indent=2))
    return 0


def _validate_mirror(args: argparse.Namespace) -> int:
    from ..mirror_state import validate_sparse_mirror
    from .registered_mirror import (
        validate_full_sparse_mirror,
        validate_registered_mirror,
    )

    if (args.mirror / "crossres_sparse_mirror.json").is_file():
        summary = validate_sparse_mirror(args.mirror)
    elif (
        args.mirror
        / f"crossres_registered_mirror_{args.array_key.strip('/').replace('/', '_')}.json"
    ).is_file():
        summary = validate_registered_mirror(args.mirror, array_key=args.array_key)
    else:
        summary = validate_full_sparse_mirror(args.mirror, array_key=args.array_key)
    print(json.dumps(summary, indent=2))
    return 0


def _parse_named_counts(
    values: list[str], option: str, name_label: str
) -> dict[str, int] | None:
    if not values:
        return None
    result: dict[str, int] = {}
    for value in values:
        try:
            name, raw_count = value.rsplit("=", 1)
            count = int(raw_count)
        except ValueError as error:
            raise ValueError(
                f"{option}: expected {name_label}=COUNT, got {value!r}"
            ) from error
        name = name.strip()
        if not name or count <= 0:
            raise ValueError(
                f"{option}: expected a nonempty {name_label.lower()} and positive count"
            )
        if name in result:
            raise ValueError(f"{option}: duplicate {name_label.lower()} {name}")
        result[name] = count
    return result


def _validate_patches(args: argparse.Namespace) -> int:
    from .patches import validate_patch_corpus

    expected_split_counts = {
        split: value
        for split, value in (
            ("train", args.expected_train),
            ("val", args.expected_val),
            ("test", args.expected_test),
        )
        if value is not None
    }
    summary = validate_patch_corpus(
        args.patches,
        expected_count=args.expected_count,
        expected_split_counts=expected_split_counts or None,
        expected_train_scrolls=set(args.expected_train_scroll)
        if args.expected_train_scroll
        else None,
        expected_val_scrolls=set(args.expected_val_scroll)
        if args.expected_val_scroll
        else None,
        expected_test_scrolls=set(args.expected_test_scroll)
        if args.expected_test_scroll
        else None,
        expected_train_scroll_counts=_parse_named_counts(
            args.expected_train_scroll_count,
            "--expected-train-scroll-count",
            "SCROLL",
        ),
        expected_val_scroll_counts=_parse_named_counts(
            args.expected_val_scroll_count,
            "--expected-val-scroll-count",
            "SCROLL",
        ),
        expected_test_scroll_counts=_parse_named_counts(
            args.expected_test_scroll_count,
            "--expected-test-scroll-count",
            "SCROLL",
        ),
        expected_record_counts=_parse_named_counts(
            args.expected_record_count,
            "--expected-record-count",
            "RECORD",
        ),
        expected_source_corpora=args.expected_source_corpora,
        require_hashes=args.require_hashes,
        voxel_check_count=args.voxel_check_count,
        workers=args.workers,
        max_cpu_threads=args.max_cpu_threads,
    )
    serialized = json.dumps(summary, indent=2)
    if args.summary_output is not None:
        destination = args.summary_output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(serialized + "\n", encoding="utf-8", newline="\n")
        os.replace(temporary, destination)
    print(serialized)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crossres-voxel",
        description="Dense 3-D nnU-Net cross-resolution img2img distillation",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit-manifest")
    audit.add_argument("pairs", type=Path)
    audit.add_argument("--load-support", action="store_true")
    audit.set_defaults(function=_audit)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--pairs", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--patches-per-record", type=int, default=64)
    prepare.add_argument("--patch-shape", nargs=3, type=int, default=(192, 192, 192))
    prepare.add_argument("--seed", type=int, default=1203)
    prepare.add_argument("--min-known-fraction", type=float, default=0.20)
    prepare.add_argument(
        "--native-teacher-min-known-fraction",
        type=float,
        default=0.002,
        help=(
            "known-voxel gate for sparse native-fine teacher records; dense "
            "records retain --min-known-fraction"
        ),
    )
    prepare.add_argument(
        "--native-teacher-min-fine-ct-nonzero-fraction",
        type=float,
        default=0.95,
        help=(
            "minimum fine-CT nonzero fraction for locally recorded native "
            "teacher chunks; rejected chunks become unknown support"
        ),
    )
    prepare.add_argument("--min-positive-voxels", type=int, default=32)
    prepare.add_argument("--min-ct-nonzero-fraction", type=float, default=0.05)
    prepare.add_argument("--attempts-per-patch", type=int, default=12)
    prepare.add_argument("--selection-candidates", type=int, default=4)
    prepare.add_argument("--pathology-fraction", type=float, default=1.0 / 3.0)
    prepare.add_argument("--positive-density-fraction", type=float, default=1.0 / 6.0)
    prepare.add_argument("--validity-block", type=int, default=64)
    prepare.add_argument(
        "--projection-cache-entries",
        type=int,
        default=16_384,
        help=(
            "maximum packed sparse fine-chunk projections retained per source record"
        ),
    )
    prepare.add_argument("--max-cpu-threads", type=int, default=16)
    prepare.set_defaults(function=_prepare)

    teacher = commands.add_parser(
        "materialize-teacher",
        help="run the native-fine 3-D teacher and write sparse-Zarr voxel labels",
    )
    teacher.add_argument("--fine-volume", required=True)
    teacher.add_argument("--fine-support-inventory", type=Path, required=True)
    teacher.add_argument("--output", type=Path, required=True)
    teacher.add_argument("--teacher-checkpoint", type=Path, required=True)
    teacher.add_argument(
        "--villa-source",
        type=Path,
        default=Path("output/crossres_data/vendor/villa/vesuvius/src"),
        help="pinned Villa vesuvius/src directory used to instantiate the teacher",
    )
    teacher.add_argument("--chunks", type=int, default=768)
    teacher.add_argument(
        "--allow-fewer-chunks",
        action="store_true",
        help=(
            "materialize every usable fully contextual neighborhood when sparse "
            "support or exhausted post-inference filters cannot reach --chunks; "
            "the strict default remains fail-closed"
        ),
    )
    teacher.add_argument("--seed", type=int, default=1203)
    teacher.add_argument("--input-shape", nargs=3, type=int, default=(256, 256, 256))
    teacher.add_argument("--threshold", type=float, default=0.45)
    teacher.add_argument("--min-positive-voxels", type=int, default=32)
    teacher.add_argument("--min-ct-nonzero-fraction", type=float, default=0.05)
    teacher.add_argument("--max-candidates", type=int, default=20_000)
    teacher.add_argument("--device", default="cuda")
    teacher.add_argument(
        "--amp-dtype",
        choices=("auto", "bfloat16", "float16"),
        default="float16",
        help="teacher autocast dtype (official inference used float16)",
    )
    teacher.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="eight-way mirror TTA (official teacher default: enabled)",
    )
    teacher.add_argument(
        "--sliding-blend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="global-grid Gaussian logit blend (official default: enabled)",
    )
    teacher.add_argument(
        "--sliding-step-size",
        type=float,
        default=0.5,
        help="sliding step as a fraction of patch size (official default: 0.5)",
    )
    teacher.add_argument(
        "--inference-batch-size",
        type=int,
        default=1,
        help="spatial teacher patches per forward pass (recorded in identity)",
    )
    teacher.add_argument(
        "--candidate-tile-chunks",
        type=int,
        default=0,
        help=(
            "shuffle spatial tiles, then Morton-walk neighboring output chunks; "
            "zero preserves fully random legacy ordering"
        ),
    )
    teacher.add_argument(
        "--prediction-cache-entries",
        type=int,
        default=0,
        help=(
            "CPU LRU entries for reusable official float16 sliding-window logits; "
            "one 256-cube surface entry is approximately 64 MiB"
        ),
    )
    teacher.add_argument(
        "--candidate-chunk",
        nargs=3,
        type=int,
        metavar=("Z", "Y", "X"),
        help="force one mirrored fine-CT chunk for a reproducible audit",
    )
    teacher.add_argument("--max-cpu-threads", type=int, default=16)
    teacher.set_defaults(function=_materialize_teacher)

    teacher_audit = commands.add_parser(
        "audit-teacher",
        help="render direct voxel comparisons against a published teacher",
    )
    teacher_audit.add_argument("--fine-volume", required=True)
    teacher_audit.add_argument("--candidate-volume", required=True)
    teacher_audit.add_argument("--reference-volume", required=True)
    teacher_audit.add_argument("--output", type=Path, required=True)
    teacher_audit.add_argument("--max-records", type=int, default=8)
    teacher_audit.add_argument("--slices-per-axis", type=int, default=3)
    teacher_audit.add_argument("--tolerance-voxels", type=int, default=2)
    teacher_audit.add_argument("--max-cpu-threads", type=int, default=16)
    teacher_audit.set_defaults(function=_audit_teacher)

    teacher_validate = commands.add_parser(
        "validate-teacher",
        help="verify every compressed chunk and the final sparse inventory",
    )
    teacher_validate.add_argument("--output", type=Path, required=True)
    teacher_validate.set_defaults(function=_validate_teacher)

    mirror_validate = commands.add_parser(
        "validate-mirror",
        help="verify a complete sparse-mirror plan and every physical object",
    )
    mirror_validate.add_argument("--mirror", type=Path, required=True)
    mirror_validate.add_argument("--array-key", default="0")
    mirror_validate.set_defaults(function=_validate_mirror)

    patch_validate = commands.add_parser(
        "validate-patches",
        help="verify immutable NPZ bytes, arrays, statistics, splits, and sources",
    )
    patch_validate.add_argument("--patches", type=Path, required=True)
    patch_validate.add_argument("--expected-count", type=int)
    patch_validate.add_argument("--expected-train", type=int)
    patch_validate.add_argument("--expected-val", type=int)
    patch_validate.add_argument("--expected-test", type=int)
    patch_validate.add_argument("--expected-train-scroll", action="append", default=[])
    patch_validate.add_argument("--expected-val-scroll", action="append", default=[])
    patch_validate.add_argument("--expected-test-scroll", action="append", default=[])
    patch_validate.add_argument(
        "--expected-train-scroll-count",
        action="append",
        default=[],
        metavar="SCROLL=COUNT",
    )
    patch_validate.add_argument(
        "--expected-val-scroll-count",
        action="append",
        default=[],
        metavar="SCROLL=COUNT",
    )
    patch_validate.add_argument(
        "--expected-test-scroll-count",
        action="append",
        default=[],
        metavar="SCROLL=COUNT",
    )
    patch_validate.add_argument(
        "--expected-record-count",
        action="append",
        default=[],
        metavar="RECORD=COUNT",
    )
    patch_validate.add_argument("--expected-source-corpora", type=int)
    patch_validate.add_argument(
        "--require-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    patch_validate.add_argument(
        "--voxel-check-count",
        type=int,
        help=(
            "validate every manifest/provenance row but recompute voxel arrays for "
            "only this deterministic, record-stratified sample"
        ),
    )
    patch_validate.add_argument("--workers", type=int, default=8)
    patch_validate.add_argument("--max-cpu-threads", type=int, default=16)
    patch_validate.add_argument(
        "--summary-output",
        type=Path,
        help="atomically persist the machine-readable validation summary",
    )
    patch_validate.set_defaults(function=_validate_patches)

    train = commands.add_parser("train")
    train.add_argument("--patches", type=Path, required=True)
    train.add_argument(
        "--validation-patches",
        type=Path,
        help="optional separate immutable validation manifest",
    )
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--m7-checkpoint", type=Path)
    train.add_argument(
        "--preset", choices=("m7-resenc-l", "tiny-test"), default="m7-resenc-l"
    )
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument(
        "--samples-per-epoch",
        type=int,
        help=(
            "use deterministic contiguous partitions of shuffled full-corpus "
            "passes, allowing shorter checkpoint/validation epochs"
        ),
    )
    train.add_argument(
        "--max-train-samples",
        type=int,
        help=(
            "authoritative sample budget; samples-per-epoch then controls only "
            "checkpoint and validation frequency"
        ),
    )
    train.add_argument(
        "--snapshot-samples",
        nargs="+",
        type=int,
        default=[],
        help=(
            "save model-only post-optimizer checkpoints when cumulative training "
            "first reaches each requested sample milestone"
        ),
    )
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--accumulate", type=int, default=3)
    train.add_argument("--learning-rate", type=float, default=1.0e-3)
    train.add_argument(
        "--lr-schedule", choices=("constant", "poly", "cosine"), default="poly"
    )
    train.add_argument("--lr-floor-ratio", type=float, default=0.0)
    train.add_argument("--warmup-samples", type=int, default=0)
    train.add_argument("--weight-decay", type=float, default=3.0e-5)
    train.add_argument("--momentum", type=float, default=0.99)
    train.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="sgd")
    train.add_argument("--adamw-beta1", type=float, default=0.9)
    train.add_argument("--adamw-beta2", type=float, default=0.999)
    train.add_argument("--adamw-eps", type=float, default=1.0e-8)
    train.add_argument(
        "--m7-trust-region-relative-l2",
        type=float,
        default=0.0,
        help=(
            "after every optimizer update, project the full student parameter "
            "vector into this relative-L2 ball around released M7"
        ),
    )
    train.add_argument("--loss-ce-weight", type=float, default=1.0)
    train.add_argument("--loss-dice-weight", type=float, default=1.0)
    train.add_argument("--loss-medial-recall-weight", type=float, default=0.0)
    train.add_argument("--loss-pinned-axial-weight", type=float, default=0.0)
    train.add_argument(
        "--loss-pinned-axial-probability-floor", type=float, default=0.20
    )
    train.add_argument("--loss-pinned-axial-bottom-fraction", type=float, default=0.10)
    train.add_argument("--pinned-medial-bridge-state")
    train.add_argument(
        "--loss-dynamic-medial-connectivity-weight", type=float, default=0.0
    )
    train.add_argument(
        "--loss-dynamic-medial-connectivity-probability-floor",
        type=float,
        default=0.20,
    )
    train.add_argument("--loss-dynamic-medial-connectivity-steps", type=int, default=96)
    train.add_argument("--dynamic-medial-connectivity-state")
    train.add_argument("--loss-separation-weight", type=float, default=0.0)
    train.add_argument("--loss-separation-radius", type=int, default=2)
    train.add_argument("--loss-separation-max-teacher-q", type=float, default=0.1)
    train.add_argument("--loss-m7-anchor-weight", type=float, default=0.0)
    train.add_argument("--loss-m7-preservation-weight", type=float, default=0.0)
    train.add_argument("--loss-m7-preservation-radius", type=int, default=2)
    train.add_argument(
        "--loss-m7-preservation-anchor-threshold", type=float, default=0.5
    )
    train.add_argument("--loss-m7-preservation-soft-floor", action="store_true")
    train.add_argument("--loss-m7-anchor-unknown-corridor-radius", type=int, default=0)
    train.add_argument(
        "--loss-m7-anchor-known-agreement",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument(
        "--loss-m7-anchor-confident-agreement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "restrict known-agreement M7 anchoring to teacher-confident voxels "
            "(q at or below the separation background ceiling, or at or above "
            "the 0.5 hard vote), freeing the partial-volume growth band"
        ),
    )
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--max-cpu-threads", type=int, default=16)
    train.add_argument("--seed", type=int, default=1203)
    train.add_argument("--device", default="cuda")
    train.add_argument(
        "--amp-dtype", choices=("auto", "bfloat16", "float16"), default="auto"
    )
    train.add_argument("--no-amp", action="store_true")
    train.add_argument("--final-fit", action="store_true")
    train.add_argument("--allow-spatial-validation", action="store_true")
    train.add_argument(
        "--stratified-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="proportionally interleave scroll/source/pathology strata",
    )
    train.add_argument(
        "--train-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "apply spatial and intensity augmentation to training patches; "
            "disable only for exact-input diagnostic controls"
        ),
    )
    train.add_argument(
        "--validation-thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_VALIDATION_THRESHOLDS),
    )
    train.add_argument("--minimum-scroll-gain", type=float, default=-0.01)
    train.add_argument("--checkpoint-min-delta", type=float, default=0.0)
    train.add_argument("--early-stopping-patience", type=int)
    train.add_argument("--resume", action="store_true")
    train.set_defaults(function=_train)

    checkpoint_audit = commands.add_parser(
        "audit-checkpoint",
        help="sweep dense voxel thresholds against held-out labels and m7",
    )
    checkpoint_audit.add_argument("--checkpoint", type=Path, required=True)
    checkpoint_audit.add_argument("--patches", type=Path, required=True)
    checkpoint_audit.add_argument("--output", type=Path, required=True)
    checkpoint_audit.add_argument("--split", default="val")
    checkpoint_audit.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[float(value) for value in torch.linspace(0.10, 0.90, 17)],
    )
    checkpoint_audit.add_argument("--qualification-scroll", default="PHerc0814")
    checkpoint_audit.add_argument("--device", default="cuda")
    checkpoint_audit.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    checkpoint_audit.add_argument("--tta", action="store_true")
    checkpoint_audit.add_argument("--num-workers", type=int, default=2)
    checkpoint_audit.add_argument("--max-cpu-threads", type=int, default=16)
    checkpoint_audit.set_defaults(function=_audit_checkpoint)

    grid_infer = commands.add_parser(
        "infer-grid",
        help="run dense voxel inference on target cubes in a local audit grid",
    )
    grid_infer.add_argument("--source", type=Path, required=True)
    grid_infer.add_argument("--checkpoint", type=Path, required=True)
    grid_infer.add_argument("--output", type=Path, required=True)
    grid_infer.add_argument("--threshold", type=float, default=0.5)
    grid_infer.add_argument("--halo", type=int, default=32)
    grid_infer.add_argument("--device", default="cuda")
    grid_infer.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    grid_infer.add_argument(
        "--tta", action=argparse.BooleanOptionalAction, default=True
    )
    grid_infer.add_argument(
        "--target-cube",
        dest="target_cubes",
        action="append",
        help="infer only this published target cube ID (repeatable)",
    )
    grid_infer.add_argument(
        "--skip-incomplete-context",
        action="store_true",
        help=(
            "exclude target cubes whose requested halo is absent from the local "
            "raw grid and record every exclusion in provenance"
        ),
    )
    grid_infer.add_argument("--max-cpu-threads", type=int, default=16)
    grid_infer.set_defaults(function=_infer_grid)

    infer = commands.add_parser("infer")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--volume", required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument(
        "--bounds",
        nargs=6,
        type=int,
        required=True,
        metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"),
    )
    infer.add_argument("--patch-shape", nargs=3, type=int, default=(192, 192, 192))
    infer.add_argument("--overlap", type=float, default=0.5)
    infer.add_argument("--threshold", type=float, default=0.5)
    infer.add_argument("--device", default="cuda")
    infer.add_argument(
        "--amp-dtype", choices=("auto", "bfloat16", "float16"), default="auto"
    )
    infer.add_argument("--no-amp", action="store_true")
    infer.add_argument("--no-tta", action="store_true")
    infer.add_argument("--max-cpu-threads", type=int, default=16)
    infer.set_defaults(function=_infer)

    pathology = commands.add_parser(
        "mine-pathology",
        help=(
            "score baseline-missing hold-in patches with released local m7 "
            "and ScrollFiesta metrics"
        ),
    )
    pathology.add_argument(
        "--patches",
        type=Path,
        action="append",
        required=True,
        help="growing patch manifest (repeat for each preparation shard)",
    )
    pathology.add_argument("--output", type=Path, required=True)
    pathology.add_argument("--m7-checkpoint", type=Path, required=True)
    pathology.add_argument("--corpus-plan", type=Path)
    pathology.add_argument(
        "--split",
        dest="splits",
        action="append",
        choices=("train", "val"),
        default=[],
    )
    pathology.add_argument("--batch-size", type=int, default=3)
    pathology.add_argument("--device", default="cuda")
    pathology.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
    )
    pathology.add_argument("--threshold", type=float, default=0.2)
    pathology.add_argument("--max-cpu-threads", type=int, default=4)
    pathology.add_argument("--max-patches", type=int)
    pathology.add_argument("--expected-count", type=int)
    pathology.set_defaults(function=_mine_pathology)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
