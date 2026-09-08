"""Verify and re-synchronise the engine snapshot against the sealed F0 code pins.

The current model record (``recipes/final_c3_250k_20260907``) seals 63 source
files by SHA-256 in ``provenance/recipe.json`` under ``execution.code_pins`` and
keeps a byte-exact copy of each under ``runtime/``.  The public engine snapshot
``src/crossres_pred/voxel`` must be those same bytes, with one documented
deviation: the ``crossres_pred.pathmap`` read-time relocation hook, which is
applied to ``io.py`` and ``patches.py`` by the table below.  With the mapping
environment variables absent the hook is a no-op, so the hooked engine behaves
exactly like the sealed one.

``socratic-engine-pins --check`` verifies every pin (runtime tree and snapshot,
the two hooked files modulo the hook); ``--apply`` re-copies the pinned voxel
modules from the record and re-applies the hook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RECORD = Path("recipes") / "final_c3_250k_20260907"
PIN_PREFIX = "crossres_pred/src/crossres_pred/voxel/"
SNAPSHOT_DIR = Path("src") / "crossres_pred" / "voxel"


@dataclass(frozen=True)
class Insert:
    """Insert ``line`` before or after the unique line whose body is ``anchor``."""

    anchor: bytes
    line: bytes
    before: bool


@dataclass(frozen=True)
class Substitute:
    """Replace every line whose body is ``original`` (exactly ``count`` of them)."""

    original: bytes
    hooked: bytes
    count: int


Hook = Insert | Substitute

HOOKS: dict[str, tuple[Hook, ...]] = {
    "io.py": (
        Insert(
            anchor=b"from .schema import DenseFieldSpec",
            line=b"from ..pathmap import remap_volume_spec",
            before=True,
        ),
        Insert(
            anchor=b"def split_volume_spec(spec: str) -> tuple[Path, str | None]:",
            line=b"    spec = remap_volume_spec(spec)",
            before=False,
        ),
    ),
    "patches.py": (
        Insert(
            anchor=b"from .coarse_teacher_atlas import ATLAS_PROJECTION_CONTRACT",
            line=b"from ..pathmap import remap_embedded_path",
            before=True,
        ),
        Substitute(
            b'        path = Path(str(value["path"]))',
            b'        path = remap_embedded_path(str(value["path"]))',
            1,
        ),
        Substitute(
            b'            catalog = Path(str(array_source.get("catalog", ""))).expanduser()',
            b'            catalog = remap_embedded_path(str(array_source.get("catalog", "")))',
            1,
        ),
        Substitute(
            b'        or Path(str(identity.get("training_manifest", ""))).resolve()',
            b'        or remap_embedded_path(str(identity.get("training_manifest", ""))).resolve()',
            2,
        ),
        Substitute(
            b'    state_path = Path(str(source.get("atlas_state", ""))).expanduser()',
            b'    state_path = remap_embedded_path(str(source.get("atlas_state", "")))',
            1,
        ),
        Substitute(
            b'        medial_state_path = Path(str(source.get("medial_state", ""))).expanduser()',
            b'        medial_state_path = remap_embedded_path(str(source.get("medial_state", "")))',
            1,
        ),
        Substitute(
            b"        source_manifest = Path(str(source_text)).expanduser()",
            b"        source_manifest = remap_embedded_path(str(source_text))",
            1,
        ),
    ),
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_code_pins(root: Path) -> dict[str, str]:
    recipe = root / RECORD / "provenance" / "recipe.json"
    value: dict[str, Any] = json.loads(recipe.read_text(encoding="utf-8"))
    pins = value["execution"]["code_pins"]
    if not isinstance(pins, dict) or not pins:
        raise ValueError(f"{recipe}: execution.code_pins is missing")
    return {str(key): str(digest) for key, digest in pins.items()}


def pinned_voxel_modules(pins: dict[str, str]) -> dict[str, str]:
    """Map ``<module>.py`` -> pin for the pinned ``voxel`` package files."""
    return {
        key[len(PIN_PREFIX) :]: digest
        for key, digest in pins.items()
        if key.startswith(PIN_PREFIX) and "/" not in key[len(PIN_PREFIX) :]
    }


def _body(line: bytes) -> bytes:
    return line.rstrip(b"\r\n")


def _terminator(line: bytes) -> bytes:
    return line[len(_body(line)) :]


def _indices(lines: list[bytes], body: bytes) -> list[int]:
    return [index for index, line in enumerate(lines) if _body(line) == body]


def apply_hook(name: str, data: bytes) -> bytes:
    """Apply the documented pathmap hook to the pinned bytes of ``name``."""
    hooks = HOOKS.get(name)
    if hooks is None:
        return data
    lines = data.splitlines(keepends=True)
    for hook in hooks:
        if isinstance(hook, Insert):
            if _indices(lines, hook.line):
                raise ValueError(f"{name}: hook line already present: {hook.line!r}")
            anchors = _indices(lines, hook.anchor)
            if len(anchors) != 1:
                raise ValueError(
                    f"{name}: expected one anchor {hook.anchor!r}, found {len(anchors)}"
                )
            index = anchors[0]
            terminator = _terminator(lines[index]) or b"\n"
            lines.insert(index if hook.before else index + 1, hook.line + terminator)
        else:
            if _indices(lines, hook.hooked):
                raise ValueError(f"{name}: hooked line already present: {hook.hooked!r}")
            matches = _indices(lines, hook.original)
            if len(matches) != hook.count:
                raise ValueError(
                    f"{name}: expected {hook.count} of {hook.original!r}, "
                    f"found {len(matches)}"
                )
            for index in matches:
                lines[index] = hook.hooked + _terminator(lines[index])
    return b"".join(lines)


def remove_hook(name: str, data: bytes) -> bytes:
    """Undo :func:`apply_hook`, recovering the pinned bytes exactly."""
    hooks = HOOKS.get(name)
    if hooks is None:
        return data
    lines = data.splitlines(keepends=True)
    for hook in reversed(hooks):
        if isinstance(hook, Insert):
            matches = _indices(lines, hook.line)
            if len(matches) != 1:
                raise ValueError(
                    f"{name}: expected one hook line {hook.line!r}, found {len(matches)}"
                )
            index = matches[0]
            neighbour = index + 1 if hook.before else index - 1
            if not 0 <= neighbour < len(lines) or _body(lines[neighbour]) != hook.anchor:
                raise ValueError(f"{name}: hook line {hook.line!r} is not at its anchor")
            del lines[index]
        else:
            matches = _indices(lines, hook.hooked)
            if len(matches) != hook.count:
                raise ValueError(
                    f"{name}: expected {hook.count} of {hook.hooked!r}, "
                    f"found {len(matches)}"
                )
            for index in matches:
                lines[index] = hook.original + _terminator(lines[index])
    return b"".join(lines)


def check(root: Path | None = None) -> tuple[list[str], list[str]]:
    """Return ``(ok_messages, failures)`` for the runtime tree and the snapshot."""
    root = root or repository_root()
    pins = load_code_pins(root)
    ok: list[str] = []
    failures: list[str] = []

    runtime = root / RECORD / "runtime"
    for key, digest in sorted(pins.items()):
        path = runtime / key
        if not path.is_file():
            failures.append(f"runtime missing: {key}")
        elif sha256_file(path) != digest:
            failures.append(f"runtime pin mismatch: {key}")
        else:
            ok.append(f"ok runtime {key}")

    snapshot = root / SNAPSHOT_DIR
    modules = pinned_voxel_modules(pins)
    for name, digest in sorted(modules.items()):
        path = snapshot / name
        if not path.is_file():
            failures.append(f"snapshot missing: {SNAPSHOT_DIR.as_posix()}/{name}")
            continue
        data = path.read_bytes()
        try:
            unhooked = remove_hook(name, data)
        except ValueError as error:
            failures.append(f"snapshot hook damaged: {name}: {error}")
            continue
        if sha256_bytes(unhooked) != digest:
            suffix = " (modulo pathmap hook)" if name in HOOKS else ""
            failures.append(f"snapshot pin mismatch: {name}{suffix}")
        else:
            suffix = " (modulo pathmap hook)" if name in HOOKS else ""
            ok.append(f"ok snapshot {SNAPSHOT_DIR.as_posix()}/{name}{suffix}")

    present = {path.name for path in snapshot.glob("*.py")}
    for name in sorted(present - set(modules)):
        failures.append(f"unpinned snapshot module: {SNAPSHOT_DIR.as_posix()}/{name}")
    return ok, failures


def sync(root: Path | None = None) -> list[str]:
    """Copy every pinned voxel module from the record and re-apply the hook."""
    root = root or repository_root()
    pins = load_code_pins(root)
    runtime = root / RECORD / "runtime" / PIN_PREFIX
    snapshot = root / SNAPSHOT_DIR
    snapshot.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for name, digest in sorted(pinned_voxel_modules(pins).items()):
        source = runtime / name
        data = source.read_bytes()
        if sha256_bytes(data) != digest:
            raise ValueError(f"runtime pin mismatch, refusing to sync from it: {name}")
        destination = snapshot / name
        if name in HOOKS:
            hooked = apply_hook(name, data)
            if remove_hook(name, hooked) != data:
                raise AssertionError(f"{name}: hook does not round-trip")
            destination.write_bytes(hooked)
            written.append(f"synced {name} (+ pathmap hook)")
        else:
            shutil.copyfile(source, destination)
            written.append(f"synced {name}")
    _, failures = check(root)
    if failures:
        raise RuntimeError("engine sync left failures:\n" + "\n".join(failures))
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify (default) or re-apply the sealed F0 engine pins"
    )
    parser.add_argument("--root", type=Path, default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="verify every pin")
    group.add_argument(
        "--apply",
        action="store_true",
        help="re-copy the pinned voxel modules from the record and re-apply the hook",
    )
    parser.add_argument("--quiet", action="store_true", help="print failures only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve() if args.root else repository_root()
    if args.apply:
        for line in sync(root):
            print(line)
    ok, failures = check(root)
    if not args.quiet:
        for line in ok:
            print(line)
    for line in failures:
        print(f"FAIL {line}", file=sys.stderr)
    print(f"{len(ok)} ok, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
