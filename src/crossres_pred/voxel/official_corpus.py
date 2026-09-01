from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .affines import find_volume_affine
from .manual_labels import INDEX_ROWS, load_catalog
from .schema import PAIR_SCHEMA

PLAN_SCHEMA = "crossres-official-manual-pair-plan-v1"
M7_MODEL_ID = "20260413222639"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _origin_uri(origin: dict[str, Any]) -> str:
    path = str(origin["path"]).lstrip("/")
    roots = origin.get("access_roots") or []
    if not roots:
        raise ValueError(f"data origin has no public access root: {origin}")
    preferred = min(
        roots,
        key=lambda item: 0 if str(item.get("type")) == "s3" else 1,
    )
    return f"{str(preferred['url']).rstrip('/')}/{path}"


def official_data_entry(
    volume: dict[str, Any],
    data_type: str,
    *,
    model_id: str | None = None,
    level: int | None = None,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for entry in volume.get("data") or []:
        if entry.get("type") != data_type:
            continue
        parameters = entry.get("parameters") or {}
        if model_id is not None and str(parameters.get("model_id")) != model_id:
            continue
        if level is not None and int(parameters.get("level", -1)) != level:
            continue
        matches.append(entry)
    if len(matches) != 1:
        raise ValueError(
            f"{volume.get('sample_id')}/{volume.get('id')}: expected one "
            f"{data_type} entry for model={model_id}, level={level}; "
            f"found {len(matches)}"
        )
    origins = matches[0].get("origins") or []
    if not origins:
        raise ValueError(f"{volume.get('id')}: {data_type} has no origins")
    return {**matches[0], "uri": _origin_uri(origins[0])}


def _local_candidates(
    repo_root: Path,
    sample_id: str,
    basename: str,
) -> list[Path]:
    root = repo_root / f"{sample_id}-full"
    return [
        root / basename,
        root / "fine_inputs" / basename,
        root / "representations" / "predictions" / "surfaces" / basename,
    ]


def _local_zarr(
    *,
    repo_root: Path,
    inputs_root: Path,
    sample_id: str,
    uri: str,
    array_key: str,
) -> tuple[Path, bool]:
    basename = Path(uri.rstrip("/")).name
    for candidate in _local_candidates(repo_root, sample_id, basename):
        if (candidate / array_key / ".zarray").is_file():
            return candidate.resolve(), True
    return (inputs_root / sample_id / basename).resolve(), False


def _real_affine(
    volumes: dict[str, Any], fine_volume_id: str, coarse_volume_id: str
) -> list[list[float]]:
    affine = find_volume_affine(volumes, fine_volume_id, coarse_volume_id)
    fine_um = float(volumes[fine_volume_id]["properties"]["pixel_size_um"])
    coarse_um = float(volumes[coarse_volume_id]["properties"]["pixel_size_um"])
    measured = np.linalg.svd(affine[:3, :3], compute_uv=False)
    expected = fine_um / coarse_um
    if np.max(np.abs(measured - expected)) > max(0.02, expected * 0.08):
        raise ValueError(
            f"{fine_volume_id}->{coarse_volume_id}: transform scales {measured} "
            f"do not match voxel-size ratio {expected}"
        )
    return affine[:3, :].tolist()


def build_manual_pair_plan(
    *,
    metadata_path: str | Path,
    catalog_path: str | Path,
    labels_root: str | Path,
    inputs_root: str | Path,
    repo_root: str | Path,
    output_path: str | Path,
    holdout_samples: set[str] | None = None,
) -> Path:
    """Resolve official metadata into deterministic local materialization jobs."""

    metadata_source = Path(metadata_path).expanduser().resolve()
    metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
    catalog = load_catalog(catalog_path)
    labels = Path(labels_root).expanduser().resolve()
    inputs = Path(inputs_root).expanduser().resolve()
    repository = Path(repo_root).expanduser().resolve()
    holdouts = holdout_samples or set()
    pairs: list[dict[str, Any]] = []

    for dataset in catalog["datasets"]:
        sample_id = str(dataset["sample_id"])
        fine_volume_id = str(dataset["volume_id"])
        try:
            volumes = metadata["samples"][sample_id]["volumes"]
            fine_volume = volumes[fine_volume_id]
        except KeyError as error:
            raise KeyError(
                f"official metadata is missing {sample_id}/{fine_volume_id}"
            ) from error
        fine_um = float(fine_volume["properties"]["pixel_size_um"])
        label_root = labels / str(dataset["zarr"])
        pairings = dataset.get("pairings") or []
        if not pairings:
            raise ValueError(f"{dataset['dataset_id']}: no pairings declared")

        for pairing in pairings:
            kind = str(pairing["kind"])
            coarse_volume_id = str(pairing["coarse_volume_id"])
            coarse_volume = volumes[coarse_volume_id]
            array_key = str(pairing.get("array_key", "0")).strip("/")
            if not array_key:
                raise ValueError(f"{dataset['dataset_id']}: empty source array key")
            if kind == "registered-real":
                affine = _real_affine(volumes, fine_volume_id, coarse_volume_id)
                coarse_um = float(coarse_volume["properties"]["pixel_size_um"])
            elif kind == "same-scan-pyramid":
                if coarse_volume_id != fine_volume_id:
                    raise ValueError(
                        f"{dataset['dataset_id']}: same-scan pairing changed volume"
                    )
                factor = int(pairing["downsample_factor"])
                if factor <= 1:
                    raise ValueError("downsample_factor must exceed one")
                scale = 1.0 / factor
                affine = [
                    [scale, 0.0, 0.0, 0.0],
                    [0.0, scale, 0.0, 0.0],
                    [0.0, 0.0, scale, 0.0],
                ]
                coarse_um = fine_um * factor
            else:
                raise ValueError(
                    f"{dataset['dataset_id']}: unsupported pairing kind {kind!r}"
                )

            volume_entry = official_data_entry(coarse_volume, "ome-zarr")
            image_uri = str(volume_entry["uri"])
            image_local, image_preexisting = _local_zarr(
                repo_root=repository,
                inputs_root=inputs,
                sample_id=sample_id,
                uri=image_uri,
                array_key=array_key,
            )
            baseline: dict[str, Any] | None = None
            if bool(pairing.get("published_baseline", False)):
                baseline_level = 0 if kind == "registered-real" else int(
                    pairing["downsample_factor"]
                ).bit_length() - 1
                baseline_entry = official_data_entry(
                    coarse_volume,
                    "surface-prediction-zarr",
                    model_id=M7_MODEL_ID,
                    level=baseline_level,
                )
                baseline_uri = str(baseline_entry["uri"])
                baseline_local, baseline_preexisting = _local_zarr(
                    repo_root=repository,
                    inputs_root=inputs,
                    sample_id=sample_id,
                    uri=baseline_uri,
                    array_key="0",
                )
                baseline = {
                    "uri": baseline_uri,
                    "array_key": "0",
                    "local_zarr": str(baseline_local),
                    "preexisting": baseline_preexisting,
                    "encoding": "labels",
                    "positive_labels": [255],
                    "threshold": 0.5,
                }

            pair_id = f"{dataset['dataset_id']}-{kind}"
            pairs.append(
                {
                    "pair_id": pair_id,
                    "dataset_id": str(dataset["dataset_id"]),
                    "sample_id": sample_id,
                    "split": (
                        "val"
                        if sample_id in holdouts
                        else (
                            "train"
                            if str(dataset["role"]).startswith("train")
                            else str(dataset["role"])
                        )
                    ),
                    "kind": kind,
                    "patch_count": int(pairing["patch_count"]),
                    "fine": {
                        "volume_id": fine_volume_id,
                        "scan_id": str(fine_volume["scan_id"]),
                        "voxel_um": fine_um,
                        "label_zarr": str(label_root),
                        "label_array_key": "0",
                        "label_inventory": str(label_root / INDEX_ROWS),
                        "positive_labels": [int(value) for value in dataset["positive_labels"]],
                        "ignore_labels": [int(value) for value in dataset["ignore_labels"]],
                        "to_coarse_affine_xyz": affine,
                    },
                    "coarse": {
                        "volume_id": coarse_volume_id,
                        "scan_id": str(coarse_volume["scan_id"]),
                        "voxel_um": coarse_um,
                        "uri": image_uri,
                        "array_key": array_key,
                        "local_zarr": str(image_local),
                        "preexisting": image_preexisting,
                        "baseline": baseline,
                    },
                }
            )

    plan = {
        "schema": PLAN_SCHEMA,
        "schema_version": 1,
        "metadata": str(metadata_source),
        "metadata_sha256": _sha256(metadata_source),
        "catalog": str(catalog["_path"]),
        "catalog_sha256": str(catalog["_sha256"]),
        "labels_root": str(labels),
        "inputs_root": str(inputs),
        "repo_root": str(repository),
        "holdout_samples": sorted(holdouts),
        "pairs": pairs,
    }
    destination = Path(output_path).expanduser().resolve()
    _atomic_json(destination, plan)
    return destination


def load_manual_pair_plan(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    plan = json.loads(source.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA or not plan.get("pairs"):
        raise ValueError(f"{source}: invalid or empty manual pair plan")
    plan["_path"] = str(source)
    plan["_sha256"] = _sha256(source)
    return plan


def write_manual_pair_manifest(
    *, plan_path: str | Path, output_path: str | Path
) -> Path:
    """Write an executable pair manifest once all selected inputs exist."""

    plan = load_manual_pair_plan(plan_path)
    rows: list[dict[str, Any]] = []
    for pair in plan["pairs"]:
        fine = pair["fine"]
        coarse = pair["coarse"]
        label_zarr = Path(fine["label_zarr"])
        label_inventory = Path(fine["label_inventory"])
        image_zarr = Path(coarse["local_zarr"])
        if not (label_zarr / fine["label_array_key"] / ".zarray").is_file():
            raise FileNotFoundError(f"missing extracted label array: {label_zarr}")
        if not label_inventory.is_file():
            raise FileNotFoundError(f"missing label index: {label_inventory}")
        if not (image_zarr / coarse["array_key"] / ".zarray").is_file():
            raise FileNotFoundError(f"missing coarse image array: {image_zarr}")
        row: dict[str, Any] = {
            "schema": PAIR_SCHEMA,
            "schema_version": 1,
            "record_id": pair["pair_id"],
            "scroll_id": pair["sample_id"],
            "split": pair["split"],
            "patch_count": pair["patch_count"],
            "supervision_source": pair.get(
                "supervision_source", f"official-human-2um/{pair['kind']}"
            ),
            "coarse": {
                "scan_id": coarse["scan_id"],
                "voxel_um": coarse["voxel_um"],
                "image": f"{image_zarr}::{coarse['array_key']}",
            },
            "fine": {
                "scan_id": fine["scan_id"],
                "voxel_um": fine["voxel_um"],
                "target": {
                    "volume": f"{label_zarr}::{fine['label_array_key']}",
                    "encoding": "labels",
                    "positive_labels": fine["positive_labels"],
                    "ignore_labels": fine["ignore_labels"],
                    "threshold": 0.5,
                    "support": {
                        "kind": "present-chunks",
                        "inventory": str(label_inventory),
                    },
                },
                "to_coarse_affine_xyz": fine["to_coarse_affine_xyz"],
            },
        }
        if pair.get("registration_evidence") is not None:
            row["registration_evidence"] = pair["registration_evidence"]
        baseline = coarse.get("baseline")
        if baseline is not None:
            baseline_zarr = Path(baseline["local_zarr"])
            if not (baseline_zarr / baseline["array_key"] / ".zarray").is_file():
                raise FileNotFoundError(f"missing baseline array: {baseline_zarr}")
            row["coarse"]["baseline"] = {
                "volume": f"{baseline_zarr}::{baseline['array_key']}",
                "encoding": baseline["encoding"],
                "positive_labels": baseline["positive_labels"],
                "threshold": baseline["threshold"],
            }
        rows.append(row)

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    os.replace(temporary, destination)
    return destination
