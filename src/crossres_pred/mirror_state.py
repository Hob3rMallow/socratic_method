from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path, PurePosixPath
from typing import Any

_MIRROR_MANIFEST_NAMES = (
    "crossres_sparse_mirror.json",
    "http_mirror.json",
    "zarr_mirror.json",
)


def resolved_spec_path(spec: str) -> Path:
    path_text = spec.rsplit("::", 1)[0] if "::" in spec else spec
    return Path(path_text).expanduser().resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mirror_manifest_identity(path: Path) -> dict[str, Any] | None:
    if not path.is_dir():
        return None
    for name in _MIRROR_MANIFEST_NAMES:
        manifest_path = path / name
        if not manifest_path.is_file():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{manifest_path}: mirror manifest must be an object")
        objects = payload.get("objects")
        object_identity = (
            {
                key: objects[key]
                for key in (
                    "count",
                    "bytes",
                    "chunk_count",
                    "chunk_bytes",
                    "plan_sha256",
                )
                if key in objects
            }
            if isinstance(objects, dict)
            else None
        )
        return {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
            "kind": payload.get("kind"),
            "state": payload.get("state"),
            "source": payload.get("source_zarr", payload.get("source")),
            "objects": object_identity,
        }
    return None


def require_complete_mirror(spec: str, *, role: str) -> None:
    identity = mirror_manifest_identity(resolved_spec_path(spec))
    if identity is not None and identity.get("state") != "complete":
        raise ValueError(
            f"{role} mirror is not complete "
            f"(state={identity.get('state')!r}): {identity['path']}"
        )


def validate_sparse_mirror(path: str | Path) -> dict[str, Any]:
    """Validate a completed planned-object sparse mirror against disk.

    This covers mirrors produced by ``mirror_sparse_teacher.py``.  It proves
    that the immutable JSONL plan matches the manifest and that every planned
    object is a regular file with the exact remotely inventoried byte size.
    """

    root = Path(path).expanduser().resolve()
    manifest_path = root / "crossres_sparse_mirror.json"
    plan_path = root / "crossres_sparse_objects.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "crossres-sparse-zarr-mirror":
        raise ValueError(
            f"{manifest_path}: unsupported mirror kind {manifest.get('kind')!r}"
        )
    if manifest.get("state") != "complete":
        raise ValueError(
            f"{manifest_path}: mirror state is {manifest.get('state')!r}, "
            "not complete"
        )
    declared_output = Path(str(manifest.get("output", ""))).expanduser().resolve()
    if declared_output != root:
        raise ValueError(
            f"{manifest_path}: declared output {declared_output} != {root}"
        )
    transfer = manifest.get("transfer")
    if not isinstance(transfer, dict) or transfer.get("failures") != []:
        raise ValueError(f"{manifest_path}: transfer failures are not empty")
    declared = manifest.get("objects")
    if not isinstance(declared, dict):
        raise TypeError(f"{manifest_path}: objects must be an object")

    digest = hashlib.sha256()
    seen: set[str] = set()
    count = 0
    total_bytes = 0
    chunk_count = 0
    chunk_bytes = 0
    physical_errors: list[str] = []
    with plan_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.rstrip("\r\n")
            if not line:
                raise ValueError(f"{plan_path}:{line_number}: blank object-plan row")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{plan_path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(row, dict):
                raise TypeError(f"{plan_path}:{line_number}: row must be an object")
            kind = row.get("kind")
            relative_text = row.get("relative_path")
            size = row.get("size")
            if kind not in {"metadata", "chunk"}:
                raise ValueError(
                    f"{plan_path}:{line_number}: invalid object kind {kind!r}"
                )
            if not isinstance(relative_text, str) or not relative_text:
                raise ValueError(
                    f"{plan_path}:{line_number}: relative_path must be non-empty"
                )
            relative = PurePosixPath(relative_text)
            if (
                relative.is_absolute()
                or relative.as_posix() != relative_text
                or "\\" in relative_text
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                raise ValueError(
                    f"{plan_path}:{line_number}: unsafe relative path "
                    f"{relative_text!r}"
                )
            if relative_text in seen:
                raise ValueError(
                    f"{plan_path}:{line_number}: duplicate path {relative_text!r}"
                )
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(
                    f"{plan_path}:{line_number}: size must be a non-negative integer"
                )
            seen.add(relative_text)
            digest.update(f"{kind}\t{relative_text}\t{size}\n".encode())
            count += 1
            total_bytes += size
            if kind == "chunk":
                chunk_count += 1
                chunk_bytes += size

            object_path = root.joinpath(*relative.parts)
            try:
                object_stat = object_path.stat()
            except FileNotFoundError:
                if len(physical_errors) < 8:
                    physical_errors.append(f"{relative_text}: missing")
                continue
            if not stat.S_ISREG(object_stat.st_mode):
                if len(physical_errors) < 8:
                    physical_errors.append(f"{relative_text}: not a regular file")
            elif object_stat.st_size != size and len(physical_errors) < 8:
                physical_errors.append(
                    f"{relative_text}: size {object_stat.st_size} != {size}"
                )

    if count == 0:
        raise ValueError(f"{plan_path}: object plan is empty")
    if physical_errors:
        raise ValueError(
            f"{root}: sparse mirror physical validation failed: "
            + "; ".join(physical_errors)
        )
    actual = {
        "count": count,
        "bytes": total_bytes,
        "chunk_count": chunk_count,
        "chunk_bytes": chunk_bytes,
        "plan_sha256": digest.hexdigest(),
    }
    for name, actual_value in actual.items():
        if declared.get(name) != actual_value:
            raise ValueError(
                f"{manifest_path}: objects.{name} {declared.get(name)!r} "
                f"!= {actual_value!r}"
            )
    return {
        "schema": "crossres-sparse-mirror-validation-v1",
        "root": str(root),
        "manifest_sha256": _sha256_file(manifest_path),
        "object_plan_sha256": _sha256_file(plan_path),
        **actual,
    }


def seed_sparse_mirror_objects(
    *,
    seed: str | Path,
    output: str | Path,
    source_uri: str,
    array_key: str,
    wanted_sizes: dict[str, int],
) -> dict[str, object]:
    """Reuse exact planned objects from a validated compatible sparse mirror."""

    seed_root = Path(seed).expanduser().resolve()
    output_root = Path(output).expanduser().resolve()
    if seed_root == output_root:
        raise ValueError("seed mirror must differ from output mirror")
    validation = validate_sparse_mirror(seed_root)
    manifest_path = seed_root / "crossres_sparse_mirror.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("source_zarr") != source_uri
        or manifest.get("array_key") != array_key
    ):
        raise ValueError(
            f"{seed_root}: seed mirror source/array differs from the requested mirror"
        )

    declared: dict[str, int] = {}
    with (seed_root / "crossres_sparse_objects.jsonl").open(
        "r", encoding="utf-8"
    ) as stream:
        for raw_line in stream:
            row = json.loads(raw_line)
            relative = str(row["relative_path"])
            if relative in wanted_sizes:
                declared[relative] = int(row["size"])

    hardlinked = 0
    copied = 0
    already_present = 0
    size_mismatches = 0
    for relative, wanted_size in wanted_sizes.items():
        if declared.get(relative) != wanted_size:
            if relative in declared:
                size_mismatches += 1
            continue
        source = seed_root.joinpath(*relative.split("/"))
        destination = output_root.joinpath(*relative.split("/"))
        if destination.is_file() and destination.stat().st_size == wanted_size:
            already_present += 1
            continue
        if not source.is_file() or source.stat().st_size != wanted_size:
            raise ValueError(
                f"{seed_root}: validated seed object disappeared: {relative}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
            hardlinked += 1
        except OSError:
            temporary = destination.with_name(destination.name + ".seed-part")
            try:
                shutil.copy2(source, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            copied += 1

    return {
        "seed": str(seed_root),
        "manifest_sha256": validation["manifest_sha256"],
        "object_plan_sha256": validation["object_plan_sha256"],
        "declared_overlap": len(declared),
        "hardlinked": hardlinked,
        "copied": copied,
        "already_present": already_present,
        "size_mismatches": size_mismatches,
    }
