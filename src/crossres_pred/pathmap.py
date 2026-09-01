"""Read-time relocation for provenance-bound artifact paths.

The original experiments intentionally stored absolute paths in manifests and
state documents.  Rewriting those files invalidates their SHA-256 identities.
When both environment variables below are set, this module maps the original
prefix to a local mirror while leaving the frozen bytes untouched.
"""

from __future__ import annotations

import os
from pathlib import Path

ORIGINAL_ROOT_ENV = "SOCRATIC_ORIGINAL_ROOT"
ARTIFACT_ROOT_ENV = "SOCRATIC_ARTIFACT_ROOT"


def remap_embedded_path(value: str | Path) -> Path:
    text = str(value)
    original = os.environ.get(ORIGINAL_ROOT_ENV)
    replacement = os.environ.get(ARTIFACT_ROOT_ENV)
    if not original and not replacement:
        return Path(text).expanduser()
    if not original or not replacement:
        raise RuntimeError(
            f"{ORIGINAL_ROOT_ENV} and {ARTIFACT_ROOT_ENV} must be set together"
        )

    normalized = text.replace("\\", "/")
    normalized_original = original.replace("\\", "/").rstrip("/")
    folded = normalized.casefold()
    folded_original = normalized_original.casefold()
    if folded == folded_original:
        return Path(replacement).expanduser()
    prefix = folded_original + "/"
    if folded.startswith(prefix):
        relative = normalized[len(normalized_original) :].lstrip("/")
        return Path(replacement).expanduser().joinpath(*relative.split("/"))
    return Path(text).expanduser()


def remap_volume_spec(spec: str) -> str:
    path_text, separator, key = spec.rpartition("::")
    if not separator:
        path_text, key = spec, ""
    # Volume specifications are serialized strings, not paths passed directly
    # to the Windows API.  POSIX separators keep a Linux artifact root usable
    # even when a manifest is being inspected on Windows.
    mapped = remap_embedded_path(path_text).as_posix()
    return f"{mapped}::{key}" if key else mapped
