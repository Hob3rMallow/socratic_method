from __future__ import annotations

import json
from pathlib import Path

import pytest

from socratic_method import engine_pins
from socratic_method.engine_pins import (
    HOOKS,
    PIN_PREFIX,
    RECORD,
    SNAPSHOT_DIR,
    Insert,
    Substitute,
    apply_hook,
    check,
    load_code_pins,
    pinned_voxel_modules,
    remove_hook,
    sha256_bytes,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]


def test_record_seals_sixty_three_pins() -> None:
    pins = load_code_pins(ROOT)
    assert len(pins) == 63
    assert len(pinned_voxel_modules(pins)) == 48
    assert all(len(digest) == 64 for digest in pins.values())


def test_runtime_tree_matches_every_pin() -> None:
    runtime = ROOT / RECORD / "runtime"
    for key, digest in load_code_pins(ROOT).items():
        assert sha256_file(runtime / key) == digest, key


def test_snapshot_matches_pins_modulo_hook() -> None:
    ok, failures = check(ROOT)
    assert failures == []
    assert len(ok) == 63 + 48


def test_documented_hook_reproduces_the_snapshot_exactly() -> None:
    runtime = ROOT / RECORD / "runtime" / PIN_PREFIX
    for name in HOOKS:
        pinned = (runtime / name).read_bytes()
        hooked = (ROOT / SNAPSHOT_DIR / name).read_bytes()
        assert apply_hook(name, pinned) == hooked, name
        assert remove_hook(name, hooked) == pinned, name


def test_hook_anchor_counts_are_exact() -> None:
    runtime = ROOT / RECORD / "runtime" / PIN_PREFIX
    for name, hooks in HOOKS.items():
        lines = [line.rstrip(b"\r\n") for line in (runtime / name).read_bytes().splitlines()]
        for hook in hooks:
            if isinstance(hook, Insert):
                assert lines.count(hook.anchor) == 1, (name, hook.anchor)
                assert lines.count(hook.line) == 0, (name, hook.line)
            else:
                assert isinstance(hook, Substitute)
                assert lines.count(hook.original) == hook.count, (name, hook.original)
                assert lines.count(hook.hooked) == 0, (name, hook.hooked)


def test_apply_hook_preserves_the_anchor_terminator() -> None:
    sample = (
        b"import x\r\n"
        b"from .schema import DenseFieldSpec\r\n"
        b"\r\n"
        b"def split_volume_spec(spec: str) -> tuple[Path, str | None]:\r\n"
        b"    return spec\r\n"
    )
    hooked = apply_hook("io.py", sample)
    assert b"from ..pathmap import remap_volume_spec\r\n" in hooked
    assert (
        b"def split_volume_spec(spec: str) -> tuple[Path, str | None]:\r\n"
        b"    spec = remap_volume_spec(spec)\r\n"
    ) in hooked
    assert remove_hook("io.py", hooked) == sample


def test_apply_hook_refuses_missing_duplicate_or_repeated_anchors() -> None:
    with pytest.raises(ValueError, match="expected one anchor"):
        apply_hook("io.py", b"nothing here\n")
    duplicated = (
        b"from .schema import DenseFieldSpec\n"
        b"from .schema import DenseFieldSpec\n"
        b"def split_volume_spec(spec: str) -> tuple[Path, str | None]:\n"
    )
    with pytest.raises(ValueError, match="expected one anchor"):
        apply_hook("io.py", duplicated)
    good = (
        b"from .schema import DenseFieldSpec\n"
        b"def split_volume_spec(spec: str) -> tuple[Path, str | None]:\n"
    )
    hooked = apply_hook("io.py", good)
    with pytest.raises(ValueError, match="already present"):
        apply_hook("io.py", hooked)


def test_remove_hook_requires_the_hook_at_its_anchor() -> None:
    misplaced = (
        b"from ..pathmap import remap_volume_spec\n"
        b"import other\n"
        b"from .schema import DenseFieldSpec\n"
        b"def split_volume_spec(spec: str) -> tuple[Path, str | None]:\n"
        b"    spec = remap_volume_spec(spec)\n"
    )
    with pytest.raises(ValueError, match="not at its anchor"):
        remove_hook("io.py", misplaced)


def test_unhooked_files_pass_through() -> None:
    assert apply_hook("model.py", b"x\n") == b"x\n"
    assert remove_hook("model.py", b"x\n") == b"x\n"


def test_check_reports_a_tampered_snapshot(tmp_path: Path) -> None:
    root = tmp_path
    (root / RECORD / "provenance").mkdir(parents=True)
    runtime = root / RECORD / "runtime" / PIN_PREFIX
    runtime.mkdir(parents=True)
    snapshot = root / SNAPSHOT_DIR
    snapshot.mkdir(parents=True)
    plain = b"VALUE = 1\n"
    (runtime / "model.py").write_bytes(plain)
    (snapshot / "model.py").write_bytes(plain)
    recipe = {"execution": {"code_pins": {PIN_PREFIX + "model.py": sha256_bytes(plain)}}}
    (root / RECORD / "provenance" / "recipe.json").write_text(
        json.dumps(recipe), encoding="utf-8"
    )
    ok, failures = check(root)
    assert failures == [] and len(ok) == 2
    (snapshot / "model.py").write_bytes(b"VALUE = 2\n")
    (snapshot / "stray.py").write_bytes(b"\n")
    _, failures = check(root)
    assert any("snapshot pin mismatch: model.py" in line for line in failures)
    assert any("unpinned snapshot module" in line for line in failures)
    assert engine_pins.main(["--root", str(root), "--check", "--quiet"]) == 1


@pytest.mark.torch
def test_sealed_f0_argv_parses_with_the_synced_cli() -> None:
    from crossres_pred.voxel import cli

    recipe = json.loads(
        (ROOT / RECORD / "provenance" / "recipe.json").read_text(encoding="utf-8")
    )
    argv = [str(value) for value in recipe["arms"][0]["argv"]]
    assert argv[1:4] == ["-m", "crossres_pred.voxel.cli", "train"]
    args = cli.build_parser().parse_args(argv[3:])
    assert args.max_train_samples == 250000
    assert args.loss_medial_tail_floor_weight == 0.0
    assert args.loss_m7_anchor_weight == 0.5
    assert args.loss_m7_anchor_confident_agreement is True
    assert args.seed == 1203
