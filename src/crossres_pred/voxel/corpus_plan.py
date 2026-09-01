from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ALLOCATION_SCHEMA = "crossres-voxel-corpus-allocation-v1"
PLAN_SCHEMA = "crossres-voxel-corpus-plan-v1"
PLAN_BUILDER_VERSION = "finite-supervision-250k-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _supervision_category(source: str) -> str:
    if "official-human-2um" in source:
        return "human"
    if "native-fine-teacher" in source:
        return "native"
    raise ValueError(f"unsupported supervision source: {source!r}")


def _counter(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    result = {str(key): int(count) for key, count in value.items()}
    if any(not key or count < 0 for key, count in result.items()):
        raise ValueError(f"{label} contains invalid counts")
    return result


def build_corpus_plan(
    *,
    config_path: str | Path,
    output_path: str | Path,
    repo_root: str | Path,
) -> Path:
    config_source = Path(config_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    config = json.loads(config_source.read_text(encoding="utf-8"))
    if config.get("schema") != ALLOCATION_SCHEMA:
        raise ValueError(f"invalid allocation schema: {config.get('schema')!r}")
    allocations = config.get("allocations")
    if not isinstance(allocations, list) or not allocations:
        raise ValueError("allocations must be a non-empty list")
    holdouts = {str(value) for value in config.get("holdout_scrolls", [])}

    manifests: dict[str, tuple[Path, str, dict[str, dict[str, Any]]]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    record_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    scroll_counts: dict[str, Counter[str]] = defaultdict(Counter)
    all_categories: Counter[str] = Counter()
    train_categories: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    seen_records: set[str] = set()
    plan_records: list[dict[str, Any]] = []

    for allocation in allocations:
        if not isinstance(allocation, dict):
            raise TypeError("allocation rows must be objects")
        relative_manifest = str(allocation["manifest"])
        record_id = str(allocation["record_id"])
        patch_count = int(allocation["patch_count"])
        group = str(allocation["group"])
        category = str(allocation["category"])
        if patch_count <= 0 or not group or not record_id:
            raise ValueError(f"invalid allocation for {record_id!r}")
        if record_id in seen_records:
            raise ValueError(f"duplicate allocated record: {record_id}")
        seen_records.add(record_id)
        if relative_manifest not in manifests:
            source = (repo / relative_manifest).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            indexed: dict[str, dict[str, Any]] = {}
            for row in _read_jsonl(source):
                source_record_id = str(row.get("record_id", ""))
                if not source_record_id or source_record_id in indexed:
                    raise ValueError(
                        f"{source}: missing or duplicate record_id {source_record_id!r}"
                    )
                indexed[source_record_id] = row
            manifests[relative_manifest] = (source, _sha256(source), indexed)
        source, source_hash, indexed = manifests[relative_manifest]
        if record_id not in indexed:
            raise ValueError(f"{source}: record not found: {record_id}")
        row = json.loads(json.dumps(indexed[record_id]))
        actual_category = _supervision_category(str(row.get("supervision_source", "")))
        if actual_category != category:
            raise ValueError(
                f"{record_id}: category {actual_category!r} does not match {category!r}"
            )
        split = str(row.get("split", ""))
        scroll = str(row.get("scroll_id", ""))
        if split not in {"train", "val"} or not scroll:
            raise ValueError(f"{record_id}: unsupported split/scroll")
        if scroll in holdouts:
            raise ValueError(f"{record_id}: sealed holdout {scroll} entered the corpus")
        row["patch_count"] = patch_count
        grouped[group].append(row)
        record_counts[record_id] += patch_count
        split_counts[split] += patch_count
        scroll_counts[split][scroll] += patch_count
        all_categories[category] += patch_count
        if split == "train":
            train_categories[category] += patch_count
        group_counts[group] += patch_count
        plan_records.append(
            {
                "record_id": record_id,
                "scroll_id": scroll,
                "split": split,
                "category": category,
                "group": group,
                "patch_count": patch_count,
                "source_manifest": str(source),
                "source_manifest_sha256": source_hash,
            }
        )

    expected = config.get("expected")
    if not isinstance(expected, dict):
        raise TypeError("expected must be an object")
    actual_splits = {name: split_counts[name] for name in ("train", "val", "test")}
    checks = (
        (sum(split_counts.values()), int(expected["total"]), "total"),
        (actual_splits, _counter(expected["splits"], "expected.splits"), "splits"),
        (
            dict(sorted(train_categories.items())),
            _counter(
                expected["training_supervision"],
                "expected.training_supervision",
            ),
            "training supervision",
        ),
        (
            dict(sorted(all_categories.items())),
            _counter(expected["all_supervision"], "expected.all_supervision"),
            "all supervision",
        ),
        (
            dict(sorted(group_counts.items())),
            _counter(expected["groups"], "expected.groups"),
            "groups",
        ),
    )
    for actual, canonical, label in checks:
        if actual != canonical:
            raise ValueError(f"{label} mismatch: {actual!r} != {canonical!r}")

    output.mkdir(parents=True, exist_ok=True)
    group_manifests: dict[str, dict[str, Any]] = {}
    for group, rows in sorted(grouped.items()):
        destination = output / f"{group}.pairs.jsonl"
        _write_jsonl_atomic(destination, rows)
        group_manifests[group] = {
            "path": str(destination),
            "sha256": _sha256(destination),
            "records": len(rows),
            "patches": group_counts[group],
        }

    plan = {
        "schema": PLAN_SCHEMA,
        "builder_version": PLAN_BUILDER_VERSION,
        "allocation": {
            "path": str(config_source),
            "sha256": _sha256(config_source),
        },
        "holdout_scrolls": sorted(holdouts),
        "source_manifests": {
            name: {"path": str(value[0]), "sha256": value[1]}
            for name, value in sorted(manifests.items())
        },
        "group_manifests": group_manifests,
        "counts": {
            "total": sum(split_counts.values()),
            "splits": actual_splits,
            "training_supervision": dict(sorted(train_categories.items())),
            "all_supervision": dict(sorted(all_categories.items())),
            "groups": dict(sorted(group_counts.items())),
            "records": dict(sorted(record_counts.items())),
            "scrolls": {
                split: dict(sorted(counts.items()))
                for split, counts in sorted(scroll_counts.items())
            },
        },
        "preparation": config.get("preparation"),
        "training": config.get("training"),
        "records": sorted(plan_records, key=lambda row: str(row["record_id"])),
    }
    destination = output / "plan.json"
    _write_json_atomic(destination, plan)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and validate the exact 250k voxel-corpus allocation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    plan = build_corpus_plan(
        config_path=args.config,
        output_path=args.output,
        repo_root=args.repo_root,
    )
    print(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
