from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

from .dataset import load_patch_rows, validate_patch_splits
from .provenance import sha256_file, utc_now, write_json_atomic
from .schema import iter_jsonl


def merge_patch_manifests(
    input_paths: Sequence[str | Path], output_path: str | Path
) -> Path:
    if len(input_paths) < 2:
        raise ValueError("at least two input patch manifests are required")
    inputs = [Path(path).expanduser().resolve() for path in input_paths]
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError(f"refusing to replace existing patch manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    merged: list[dict[str, object]] = []
    identities: list[dict[str, object]] = []
    for source in inputs:
        values = list(iter_jsonl(source))
        for value in values:
            raw_path = Path(str(value.get("path", "")))
            patch_path = (
                raw_path if raw_path.is_absolute() else source.parent / raw_path
            )
            patch_path = patch_path.resolve()
            if not patch_path.is_file():
                raise FileNotFoundError(f"patch file is missing: {patch_path}")
            row = dict(value)
            row["path"] = patch_path.as_posix()
            row["source_patch_manifest"] = source.as_posix()
            merged.append(row)
        parent_provenance = source.parent / "provenance.json"
        identities.append(
            {
                "path": str(source),
                "sha256": sha256_file(source),
                "row_count": len(values),
                "provenance_path": (
                    str(parent_provenance) if parent_provenance.is_file() else None
                ),
                "provenance_sha256": (
                    sha256_file(parent_provenance)
                    if parent_provenance.is_file()
                    else None
                ),
            }
        )

    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        for row in merged:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    try:
        parsed = load_patch_rows(temporary)
        validate_patch_splits(parsed)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, output)

    splits: dict[str, int] = {}
    scrolls: dict[str, int] = {}
    for row in parsed:
        splits[row.split] = splits.get(row.split, 0) + 1
        scrolls[row.scroll_id] = scrolls.get(row.scroll_id, 0) + 1
    write_json_atomic(
        output.with_suffix(output.suffix + ".provenance.json"),
        {
            "schema_version": 1,
            "kind": "crossres-merged-patch-manifest",
            "created_at": utc_now(),
            "output": str(output),
            "output_sha256": sha256_file(output),
            "row_count": len(parsed),
            "splits": splits,
            "scrolls": scrolls,
            "inputs": identities,
        },
    )
    return output
