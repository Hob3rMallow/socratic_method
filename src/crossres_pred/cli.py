from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .policy import DataPolicy
from .schema import load_pair_records


def _json_print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _validate_manifest(args: argparse.Namespace) -> int:
    policy = DataPolicy.load(args.policy)
    records = load_pair_records(args.manifest)
    policy.validate_records(records)
    _json_print(
        {
            "status": "valid",
            "profile": policy.profile,
            "record_count": len(records),
            "scrolls": sorted({record.scroll_id for record in records}),
            "splits": {
                split: sum(record.split == split for record in records)
                for split in sorted({record.split for record in records})
            },
            "surface_pair_count": sum(len(record.surfaces) for record in records),
        }
    )
    return 0


def _inspect_pairs(args: argparse.Namespace) -> int:
    records = load_pair_records(args.manifest)
    summaries = []
    for record in records:
        summaries.append(
            {
                "record_id": record.record_id,
                "scroll_id": record.scroll_id,
                "split": record.split,
                "coarse_scan_id": record.coarse.scan_id,
                "coarse_voxel_um": record.coarse.voxel_um,
                "fine_scan_id": record.fine.scan_id,
                "fine_voxel_um": record.fine.voxel_um,
                "surface_count": len(record.surfaces),
                "has_affine": record.fine.to_coarse_affine_xyz is not None,
                "has_baseline": record.coarse.baseline is not None,
                "has_teacher": record.fine.teacher is not None,
            }
        )
    _json_print({"record_count": len(summaries), "records": summaries})
    return 0


def _merge_patches(args: argparse.Namespace) -> int:
    from .merge import merge_patch_manifests

    manifest = merge_patch_manifests(args.input, args.output)
    rows = sum(1 for line in manifest.read_text(encoding="utf-8").splitlines() if line)
    _json_print(
        {"status": "complete", "patch_manifest": str(manifest), "patch_count": rows}
    )
    return 0


def _plan_sites(args: argparse.Namespace) -> int:
    from .sites import SiteOptions, plan_sites

    options = SiteOptions(
        site_shape_zyx=tuple(args.site_shape),
        sites_per_record=args.sites_per_record,
        seed=args.seed,
        min_surface_points=args.min_surface_points,
        min_separation_vox=args.min_separation,
        fine_halo_vox=args.fine_halo,
    )
    output = plan_sites(
        manifest_path=args.manifest,
        policy_path=args.policy,
        output_path=args.output,
        options=options,
        reuse_sites=args.reuse_sites,
    )
    from .sites import load_site_rows

    rows = load_site_rows(output)
    _json_print(
        {
            "status": "complete",
            "sites": str(output),
            "site_count": len(rows),
            "by_scroll": {
                scroll: sum(1 for row in rows if row["scroll_id"] == scroll)
                for scroll in sorted({row["scroll_id"] for row in rows})
            },
        }
    )
    return 0


def _record_for(args: argparse.Namespace):
    records = load_pair_records(args.manifest)
    matching = [record for record in records if record.record_id == args.record_id]
    if not matching:
        raise ValueError(
            f"record {args.record_id!r} not found; manifest has "
            f"{sorted(record.record_id for record in records)}"
        )
    return matching[0]


def _carve_fine(args: argparse.Namespace) -> int:
    from .carve import CarveOptions, Store, execute_carve, plan_carve
    from .sites import load_site_rows

    record = _record_for(args)
    site_rows = [
        row
        for row in load_site_rows(args.sites)
        if row["record_id"] == record.record_id
    ]
    if not site_rows:
        raise ValueError(f"no sites for record {record.record_id}")
    options = CarveOptions(
        array_key=args.array_key,
        workers=args.workers,
        tube_intersect=not args.no_tube,
        tube_dilate_vox=args.tube_dilate,
        max_bytes=int(args.max_gib * 1024**3),
    )
    store = Store(args.source)
    fine_dirs = [surface.fine_tifxyz for surface in record.surfaces]
    plan = plan_carve(
        store,
        site_rows,
        options=options,
        fine_scan_id=record.fine.scan_id,
        fine_tifxyz_dirs=fine_dirs,
    )
    if args.dry_run:
        _json_print({"status": "planned", "dry_run": True, **plan.summary()})
        return 0
    summary = execute_carve(
        store,
        plan,
        options=options,
        output_path=args.output,
        provenance={
            "sites": str(Path(args.sites).resolve()),
            "pair_manifest": str(Path(args.manifest).resolve()),
            "record_id": record.record_id,
        },
    )
    _json_print(
        {
            "status": summary["state"],
            "output": str(args.output),
            "selection": summary["selection"],
            "objects": summary["objects"],
        }
    )
    return 0


def _verify_carve(args: argparse.Namespace) -> int:
    from .carve import Store, verify_carve

    manifest = json.loads(
        (Path(args.mirror) / "crossres_sparse_mirror.json").read_text(
            encoding="utf-8"
        )
    )
    store = Store(str(manifest["source_zarr"]))
    report = verify_carve(
        store,
        args.mirror,
        sample_fraction=args.sample_fraction,
        full=args.full,
    )
    _json_print(
        {
            "status": "verified",
            "objects_total": report["objects_total"],
            "objects_checked": report["objects_checked"],
        }
    )
    return 0


def _build_distill(args: argparse.Namespace) -> int:
    from .extract import DistillOptions, build_distill_targets
    from .resample import BridgeOptions
    from .sites import load_site_rows

    record = _record_for(args)
    options = DistillOptions(
        target_source=args.target_source,
        band_radius_um=args.band_radius_um,
        bridge=BridgeOptions(
            prefilter_sigma_scale=args.prefilter_sigma_scale,
            coverage_erosion_fine_vox=args.coverage_erosion,
            max_fine_window_vox=args.max_fine_window,
            maxpool_prefilter=args.maxpool_prefilter,
            # Exact rasterized ground truth needs no boundary guard, and
            # eroding its thin slab would eat the shell negatives.
            erode_filter_margin=(args.target_source != "gt-bridge"),
        ),
        registration_reference=args.registration_reference,
    )
    output = build_distill_targets(
        site_rows=load_site_rows(args.sites),
        record=record,
        output_path=args.output,
        options=options,
        teacher_prob_spec=args.teacher_prob,
        teacher_mirror_path=args.teacher_mirror,
    )
    audit = json.loads(
        (Path(output) / "distill_audit.json").read_text(encoding="utf-8")
    )
    _json_print(
        {
            "status": "complete",
            "output": str(output),
            "site_count": audit["site_count"],
            "median_coverage_fraction": audit["median_coverage_fraction"],
            "registration_actions": audit["registration_actions"],
        }
    )
    return 0


def _extract_teacher(args: argparse.Namespace) -> int:
    from .extract import ExtractTeacherOptions, extract_teacher_patches
    from .sites import load_site_rows

    policy = DataPolicy.load(args.policy)
    record = _record_for(args)
    policy.validate_records([record])
    options = ExtractTeacherOptions(
        patch_shape_zyx=tuple(args.patch_shape),
        patches_per_site=args.patches_per_site,
        seed=args.seed,
        min_supervised_fraction=args.min_supervised_fraction,
    )
    manifest = extract_teacher_patches(
        site_rows=load_site_rows(args.sites),
        record=record,
        mirror_path=args.mirror,
        output_path=args.output,
        options=options,
        policy_profile=policy.profile,
        veto_volume_spec=args.veto_volume,
    )
    rows = sum(1 for line in manifest.read_text(encoding="utf-8").splitlines() if line)
    _json_print(
        {"status": "complete", "patch_manifest": str(manifest), "patch_count": rows}
    )
    return 0


def _infer_teacher(args: argparse.Namespace) -> int:
    from .inference import TeacherInferOptions, infer_teacher
    from .sites import load_site_rows

    policy = DataPolicy.load(args.policy)
    record = _record_for(args)
    policy.validate_records([record])
    options = TeacherInferOptions(
        patch_shape_zyx=tuple(args.patch_shape),
        stride=args.stride,
        retained_margin=args.retained_margin,
        batch_size=args.batch_size,
        device=args.device,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        array_key=args.array_key,
    )
    output = infer_teacher(
        checkpoint_path=args.checkpoint,
        site_rows=load_site_rows(args.sites),
        site_manifest_path=args.sites,
        record_id=record.record_id,
        mirror_path=args.mirror,
        output_path=args.output,
        policy_profile=policy.profile,
        options=options,
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    _json_print({"output": str(output), **summary})
    return 0


def _extract_student(args: argparse.Namespace) -> int:
    from .extract import ExtractStudentOptions, extract_student_patches
    from .sites import load_site_rows

    policy = DataPolicy.load(args.policy)
    record = _record_for(args)
    policy.validate_records([record])
    options = ExtractStudentOptions(
        patch_shape_zyx=tuple(args.patch_shape),
        anchor_patches=args.anchor_patches,
        rehearsal_patches=args.rehearsal_patches,
        seed=args.seed,
        holdout_segment_fraction=args.holdout_segment_fraction,
    )
    manifest = extract_student_patches(
        site_rows=(load_site_rows(args.sites) if args.sites is not None else []),
        record=record,
        distill_dir=args.distill_dir,
        output_path=args.output,
        options=options,
        policy_profile=policy.profile,
    )
    rows = sum(1 for line in manifest.read_text(encoding="utf-8").splitlines() if line)
    _json_print(
        {"status": "complete", "patch_manifest": str(manifest), "patch_count": rows}
    )
    return 0


def _train(args: argparse.Namespace) -> int:
    from .train import TrainOptions, train_model

    options = TrainOptions(
        profile=args.profile,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulate=args.accumulate,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        in_channels=args.in_channels,
        init_mode=args.init_mode,
        pretrained_checkpoint=(
            str(args.pretrained_checkpoint.resolve())
            if args.pretrained_checkpoint is not None
            else None
        ),
        dice_weight=args.dice_weight,
        distill_weight=args.distill_weight,
        rehearsal_weight=args.rehearsal_weight,
        rot90_mode=args.rot90_mode,
        final_fit=args.final_fit,
    )
    checkpoint = train_model(
        patch_manifest=args.patch_manifest,
        output_path=args.output,
        options=options,
        resume_checkpoint=args.resume_checkpoint,
    )
    selection = "final-epoch" if args.final_fit else "best-selection"
    _json_print(
        {
            "status": "complete",
            "checkpoint": str(checkpoint),
            "selection": selection,
        }
    )
    return 0


def _audit_checkpoint(args: argparse.Namespace) -> int:
    from .audit import audit_checkpoint

    report = audit_checkpoint(
        checkpoint_path=args.checkpoint,
        patch_manifest=args.patch_manifest,
        output_path=args.output,
        split=args.split,
        device=args.device,
        stamp_threshold=args.stamp_threshold,
    )
    _json_print(
        {
            "status": "complete",
            "output": str(args.output),
            "selection_kind": report["selection_kind"],
            "ground_truth": report["ground_truth"],
            "distill_target": report["distill_target"],
            **(
                {"deploy_threshold": report["deploy_threshold"]}
                if "deploy_threshold" in report
                else {}
            ),
        }
    )
    return 0


def _audit_bridge(args: argparse.Namespace) -> int:
    from .audit import audit_bridge_targets
    from .sites import load_site_rows

    policy = DataPolicy.load(args.policy)
    record = _record_for(args)
    policy.validate_records([record])
    report = audit_bridge_targets(
        distill_dir=args.distill_dir,
        site_rows=load_site_rows(args.sites),
        record=record,
        coarse_baseline_spec=args.coarse_baseline,
        output_path=args.output,
        interior_margin=args.interior_margin,
        band_radius_um=args.band_radius_um,
    )
    _json_print(
        {
            "status": "complete",
            "output": str(args.output),
            "teacher_bridge": report["teacher_bridge"],
            "m7_baseline": report["m7_baseline"],
            "gate": report["gate"],
        }
    )
    return 0


def _export_tensorboard(args: argparse.Namespace) -> int:
    from .telemetry import export_history_to_tensorboard

    log_dir = export_history_to_tensorboard(args.run_directory)
    _json_print({"status": "complete", "tensorboard_log_dir": str(log_dir)})
    return 0


def _infer_grid(args: argparse.Namespace) -> int:
    from .inference import InferOptions, infer_grid

    options = InferOptions(
        halo=args.halo,
        device=args.device,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        threshold=args.threshold,
        save_prob=args.save_prob,
        raw_mode=args.raw_mode,
    )
    output = infer_grid(
        checkpoint_path=args.checkpoint,
        source_grid=args.source_grid,
        output_path=args.output,
        options=options,
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    _json_print({"output": str(output), **summary})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crossres-pred",
        description="Voxel-domain teacher/student surface prediction",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--debug", action="store_true", help="show a traceback for command failures"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-manifest", help="validate pair schema, splits, and safety policy"
    )
    validate.add_argument("--policy", required=True, type=Path)
    validate.add_argument("--manifest", required=True, type=Path)
    validate.set_defaults(handler=_validate_manifest)

    inspect = subparsers.add_parser(
        "inspect-pairs", help="summarize pair records and their assets"
    )
    inspect.add_argument("--manifest", required=True, type=Path)
    inspect.set_defaults(handler=_inspect_pairs)

    merge = subparsers.add_parser(
        "merge-patches",
        help="merge completed patch corpora using absolute, provenance-backed paths",
    )
    merge.add_argument("--input", required=True, type=Path, nargs="+")
    merge.add_argument("--output", required=True, type=Path)
    merge.set_defaults(handler=_merge_patches)

    plan = subparsers.add_parser(
        "plan-sites", help="plan coarse ROI sites on traced-segment neighborhoods"
    )
    plan.add_argument("--policy", required=True, type=Path)
    plan.add_argument("--manifest", required=True, type=Path)
    plan.add_argument("--output", required=True, type=Path)
    plan.add_argument("--site-shape", nargs=3, type=int, default=[192, 192, 192])
    plan.add_argument("--sites-per-record", type=int, default=150)
    plan.add_argument("--seed", type=int, default=1203)
    plan.add_argument("--min-surface-points", type=int, default=64)
    plan.add_argument("--min-separation", type=int, default=64)
    plan.add_argument("--fine-halo", type=int, default=160)
    plan.add_argument(
        "--reuse-sites",
        type=Path,
        help="lock coarse boxes from an existing site manifest (the 1.129um wave)",
    )
    plan.set_defaults(handler=_plan_sites)

    carve = subparsers.add_parser(
        "carve-fine", help="carve fine raw chunks under site footprints"
    )
    carve.add_argument("--manifest", required=True, type=Path)
    carve.add_argument("--record-id", required=True)
    carve.add_argument("--sites", required=True, type=Path)
    carve.add_argument("--source", required=True, help="s3:// or local zarr root")
    carve.add_argument("--output", required=True, type=Path)
    carve.add_argument("--array-key", default="0")
    carve.add_argument("--workers", type=int, default=8)
    carve.add_argument("--no-tube", action="store_true")
    carve.add_argument("--tube-dilate", type=int, default=128)
    carve.add_argument("--max-gib", type=float, default=350.0)
    carve.add_argument("--dry-run", action="store_true")
    carve.set_defaults(handler=_carve_fine)

    verify = subparsers.add_parser(
        "verify-carve", help="MD5-verify carved objects against source ETags"
    )
    verify.add_argument("--mirror", required=True, type=Path)
    verify.add_argument("--sample-fraction", type=float, default=0.01)
    verify.add_argument("--full", action="store_true")
    verify.set_defaults(handler=_verify_carve)

    distill = subparsers.add_parser(
        "build-distill", help="pull teacher predictions back into coarse targets"
    )
    distill.add_argument("--manifest", required=True, type=Path)
    distill.add_argument("--record-id", required=True)
    distill.add_argument("--sites", required=True, type=Path)
    distill.add_argument("--output", required=True, type=Path)
    distill.add_argument(
        "--target-source", choices=("teacher-pred", "gt-bridge"),
        default="teacher-pred",
    )
    distill.add_argument("--teacher-prob", help="fine probability volume spec")
    distill.add_argument("--teacher-mirror", type=Path)
    distill.add_argument("--band-radius-um", type=float, default=14.0)
    distill.add_argument("--prefilter-sigma-scale", type=float, default=0.5)
    distill.add_argument("--coverage-erosion", type=int, default=32)
    distill.add_argument("--max-fine-window", type=int, default=352)
    distill.add_argument("--maxpool-prefilter", action="store_true")
    distill.add_argument(
        "--registration-reference",
        help="coarse baseline volume spec for the residual audit",
    )
    distill.set_defaults(handler=_build_distill)

    teacher = subparsers.add_parser(
        "extract-teacher", help="materialize fine-pitch teacher patches"
    )
    teacher.add_argument("--policy", required=True, type=Path)
    teacher.add_argument("--manifest", required=True, type=Path)
    teacher.add_argument("--record-id", required=True)
    teacher.add_argument("--sites", required=True, type=Path)
    teacher.add_argument("--mirror", required=True, type=Path)
    teacher.add_argument("--output", required=True, type=Path)
    teacher.add_argument("--patch-shape", nargs=3, type=int, default=[256, 256, 256])
    teacher.add_argument("--patches-per-site", type=int, default=6)
    teacher.add_argument("--seed", type=int, default=1203)
    teacher.add_argument("--min-supervised-fraction", type=float, default=0.10)
    teacher.add_argument("--veto-volume", help="villa prediction volume spec")
    teacher.set_defaults(handler=_extract_teacher)

    teacher_infer = subparsers.add_parser(
        "infer-teacher",
        help="infer a fine teacher into a sparse soft-probability zarr",
    )
    teacher_infer.add_argument("--policy", required=True, type=Path)
    teacher_infer.add_argument("--manifest", required=True, type=Path)
    teacher_infer.add_argument("--record-id", required=True)
    teacher_infer.add_argument("--sites", required=True, type=Path)
    teacher_infer.add_argument("--mirror", required=True, type=Path)
    teacher_infer.add_argument("--checkpoint", required=True, type=Path)
    teacher_infer.add_argument("--output", required=True, type=Path)
    teacher_infer.add_argument("--array-key", default="0")
    teacher_infer.add_argument(
        "--patch-shape", nargs=3, type=int, default=[256, 256, 256]
    )
    teacher_infer.add_argument("--stride", type=int, default=128)
    teacher_infer.add_argument("--retained-margin", type=int, default=32)
    teacher_infer.add_argument("--batch-size", type=int, default=1)
    teacher_infer.add_argument("--device", default="cuda")
    teacher_infer.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    teacher_infer.add_argument(
        "--amp-dtype",
        choices=("auto", "float16", "bfloat16"),
        default="auto",
    )
    teacher_infer.set_defaults(handler=_infer_teacher)

    student = subparsers.add_parser(
        "extract-student", help="materialize coarse-pitch student patches"
    )
    student.add_argument("--policy", required=True, type=Path)
    student.add_argument("--manifest", required=True, type=Path)
    student.add_argument("--record-id", required=True)
    student.add_argument("--sites", type=Path)
    student.add_argument("--distill-dir", type=Path)
    student.add_argument("--output", required=True, type=Path)
    student.add_argument("--patch-shape", nargs=3, type=int, default=[192, 192, 192])
    student.add_argument("--anchor-patches", type=int, default=200)
    student.add_argument("--rehearsal-patches", type=int, default=150)
    student.add_argument("--seed", type=int, default=1203)
    student.add_argument("--holdout-segment-fraction", type=float, default=0.0)
    student.set_defaults(handler=_extract_student)

    train = subparsers.add_parser(
        "train", help="train a fine teacher or the coarse student"
    )
    train.add_argument("--profile", choices=("teacher", "student"), required=True)
    train.add_argument("--patch-manifest", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="resume this output directory from an existing last.pt checkpoint",
    )
    train.add_argument("--device", default="cuda")
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--accumulate", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=1.0e-4)
    train.add_argument("--weight-decay", type=float, default=1.0e-4)
    train.add_argument("--warmup-steps", type=int, default=500)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--seed", type=int, default=1203)
    train.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument(
        "--amp-dtype",
        choices=("auto", "float16", "bfloat16"),
        default="auto",
    )
    train.add_argument("--in-channels", type=int, choices=(1, 2), default=1)
    train.add_argument(
        "--init-mode",
        choices=("m7-nnunet", "surface-checkpoint", "none"),
        default="m7-nnunet",
    )
    train.add_argument("--pretrained-checkpoint", type=Path)
    train.add_argument("--dice-weight", type=float, default=0.5)
    train.add_argument("--distill-weight", type=float)
    train.add_argument("--rehearsal-weight", type=float)
    train.add_argument("--rot90-mode", choices=("none", "z-only", "all"))
    train.add_argument(
        "--final-fit",
        action="store_true",
        help="train without validation and return the final epoch checkpoint",
    )
    train.set_defaults(handler=_train)

    audit = subparsers.add_parser(
        "audit-checkpoint", help="evaluate a checkpoint on a patch-manifest split"
    )
    audit.add_argument("--checkpoint", required=True, type=Path)
    audit.add_argument("--patch-manifest", required=True, type=Path)
    audit.add_argument("--output", required=True, type=Path)
    audit.add_argument("--split", default="val")
    audit.add_argument("--device", default="cuda")
    audit.add_argument(
        "--stamp-threshold",
        action="store_true",
        help="write deploy_threshold into a .calibrated.pt copy",
    )
    audit.set_defaults(handler=_audit_checkpoint)

    bridge_audit = subparsers.add_parser(
        "audit-bridge",
        help="compare coarse teacher-bridge targets with m7 on identical voxels",
    )
    bridge_audit.add_argument("--policy", required=True, type=Path)
    bridge_audit.add_argument("--manifest", required=True, type=Path)
    bridge_audit.add_argument("--record-id", required=True)
    bridge_audit.add_argument("--sites", required=True, type=Path)
    bridge_audit.add_argument("--distill-dir", required=True, type=Path)
    bridge_audit.add_argument("--coarse-baseline", required=True)
    bridge_audit.add_argument("--output", required=True, type=Path)
    bridge_audit.add_argument("--interior-margin", type=int, default=32)
    bridge_audit.add_argument("--band-radius-um", type=float, default=14.0)
    bridge_audit.set_defaults(handler=_audit_bridge)

    tensorboard = subparsers.add_parser(
        "export-tensorboard",
        help="backfill TensorBoard events from a completed training history",
    )
    tensorboard.add_argument("--run-directory", required=True, type=Path)
    tensorboard.set_defaults(handler=_export_tensorboard)

    infer = subparsers.add_parser(
        "infer-grid", help="predict cubes_PRED from raw CT as a drop-in producer"
    )
    infer.add_argument("--checkpoint", required=True, type=Path)
    infer.add_argument("--source-grid", required=True, type=Path)
    infer.add_argument("--output", required=True, type=Path)
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--halo", type=int, default=32)
    infer.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    infer.add_argument(
        "--amp-dtype",
        choices=("auto", "float16", "bfloat16"),
        default="auto",
    )
    infer.add_argument(
        "--threshold",
        type=float,
        help="binarization threshold; defaults to the checkpoint deploy_threshold",
    )
    infer.add_argument("--save-prob", action="store_true")
    infer.add_argument(
        "--raw-mode", choices=("hardlink", "copy", "none"), default="hardlink"
    )
    infer.set_defaults(handler=_infer_grid)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, RuntimeError) as error:
        if args.debug:
            raise
        parser.exit(2, f"{parser.prog}: error: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
