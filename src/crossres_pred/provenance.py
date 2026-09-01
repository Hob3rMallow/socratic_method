from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .mirror_state import mirror_manifest_identity, resolved_spec_path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, *, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def source_identity(spec: str) -> dict[str, Any]:
    path = resolved_spec_path(spec)
    value: dict[str, Any] = {"spec": spec, "resolved_path": str(path)}
    if path.exists():
        stat = path.stat()
        value.update(
            {
                "size_bytes": stat.st_size if path.is_file() else None,
                "mtime_ns": stat.st_mtime_ns,
                "kind": "file" if path.is_file() else "directory",
            }
        )
        if path.is_file() and stat.st_size <= 64 * 1024 * 1024:
            value["sha256"] = sha256_file(path)
        if path.is_dir():
            mirror = mirror_manifest_identity(path)
            if mirror is not None:
                value["mirror"] = mirror
    else:
        value["missing"] = True
    return value


def environment_identity() -> dict[str, Any]:
    identity: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
    }
    try:
        import numpy

        identity["numpy"] = numpy.__version__
    except ImportError:
        pass
    try:
        import torch

        identity.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "torch_cuda": torch.version.cuda,
            }
        )
        if torch.cuda.is_available():
            identity["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return identity


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, destination)


def require_fresh_directory(path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"output path is not a directory: {destination}")
        if any(destination.iterdir()):
            raise ValueError(
                f"refusing to reuse non-empty output directory: {destination}"
            )
    else:
        destination.mkdir(parents=True)
    return destination
