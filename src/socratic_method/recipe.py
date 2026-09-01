from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from crossres_pred.pathmap import ARTIFACT_ROOT_ENV, ORIGINAL_ROOT_ENV

RECIPE_SCHEMA = "socratic-method-training-recipe-v1"
REQUIRED_PATHS = (
    "train_manifest",
    "validation_manifest",
    "m7_checkpoint",
    "dynamic_medial_connectivity_state",
    "output",
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(value: str, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_paths(path: Path) -> dict[str, Any]:
    values = _read_object(path)
    missing = sorted(set(REQUIRED_PATHS) - set(values))
    if missing:
        raise ValueError(f"{path}: missing path keys: {', '.join(missing)}")
    result = dict(values)
    for key in REQUIRED_PATHS:
        result[key] = str(_resolve_path(str(values[key]), base=path.parent))
    if ("original_root" in values) != ("artifact_root" in values):
        raise ValueError("original_root and artifact_root must be supplied together")
    if "artifact_root" in values:
        result["artifact_root"] = str(
            _resolve_path(str(values["artifact_root"]), base=path.parent)
        )
        result["original_root"] = str(values["original_root"])
    return result


def _iter_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def verify_recipe(recipe: dict[str, Any], paths: dict[str, Any]) -> list[str]:
    if recipe.get("schema") != RECIPE_SCHEMA:
        raise ValueError(f"recipe schema must be {RECIPE_SCHEMA!r}")
    artifacts = recipe.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TypeError("recipe artifacts must be an object")

    messages: list[str] = []
    for key, contract in artifacts.items():
        if not isinstance(contract, dict) or key not in paths:
            raise ValueError(f"invalid or unresolved artifact contract: {key}")
        path = Path(str(paths[key]))
        if not path.is_file():
            raise FileNotFoundError(f"{key} is missing: {path}")
        actual_hash = _sha256(path)
        expected_hash = str(contract.get("sha256", ""))
        if actual_hash != expected_hash:
            raise ValueError(
                f"{key} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
            )
        expected_bytes = contract.get("bytes")
        if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
            raise ValueError(
                f"{key} byte-size mismatch: expected {expected_bytes}, "
                f"got {path.stat().st_size}"
            )
        messages.append(f"ok {key}: {actual_hash}")

    train_rows = _iter_manifest(Path(paths["train_manifest"]))
    train_contract = artifacts["train_manifest"]
    if len(train_rows) != int(train_contract["rows"]):
        raise ValueError("training manifest row count changed")
    train_scrolls = sorted({str(row.get("scroll_id")) for row in train_rows})
    if train_scrolls != list(train_contract["scrolls"]):
        raise ValueError(
            f"training scroll scope changed: expected {train_contract['scrolls']}, "
            f"got {train_scrolls}"
        )
    train_records = sorted({str(row.get("record_id")) for row in train_rows})
    if train_records != list(train_contract["record_ids"]):
        raise ValueError("training record identity changed")

    validation_rows = _iter_manifest(Path(paths["validation_manifest"]))
    expected_validation_scrolls = list(artifacts["validation_manifest"]["scrolls"])
    validation_scrolls = sorted(
        {
            str(row.get("scroll_id"))
            for row in validation_rows
            if str(row.get("split", "")).lower() == "val"
        }
    )
    if validation_scrolls != expected_validation_scrolls:
        raise ValueError(
            "validation scroll scope changed: "
            f"expected {expected_validation_scrolls}, got {validation_scrolls}"
        )

    state = _read_object(Path(paths["dynamic_medial_connectivity_state"]))
    state_contract = artifacts["dynamic_medial_connectivity_state"]
    for key in (
        "event_count",
        "fully_owned_event_count",
        "maximum_propagation_steps",
        "maximum_required_connectivity_steps",
    ):
        if int(state.get(key, -1)) != int(state_contract[key]):
            raise ValueError(f"dynamic connectivity {key} changed")
    return messages


def build_command(
    recipe: dict[str, Any],
    paths: dict[str, Any],
    *,
    python: str | None = None,
    resume: bool = False,
) -> list[str]:
    raw_argv = recipe.get("training", {}).get("argv")
    if not isinstance(raw_argv, list) or not all(
        isinstance(value, str) for value in raw_argv
    ):
        raise TypeError("recipe training.argv must be a string array")
    substitutions = {key: str(value) for key, value in paths.items()}
    command = [python or str(paths.get("python", sys.executable))]
    command.extend(value.format_map(substitutions) for value in raw_argv)
    if resume:
        command.append("--resume")
    return command


def build_environment(recipe: dict[str, Any], paths: dict[str, Any]) -> dict[str, str]:
    environment = os.environ.copy()
    for key, value in recipe.get("environment", {}).items():
        environment[str(key)] = str(value)
    if "artifact_root" in paths:
        environment[ORIGINAL_ROOT_ENV] = str(paths["original_root"])
        environment[ARTIFACT_ROOT_ENV] = str(paths["artifact_root"])
    return environment


def _display_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def build_parser() -> argparse.ArgumentParser:
    root = _repository_root()
    parser = argparse.ArgumentParser(
        description="Verify and execute the pinned Socratic Method v31 recipe"
    )
    parser.add_argument(
        "--recipe", type=Path, default=root / "recipes" / "v31" / "recipe.json"
    )
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--python", help="Python executable for the training child")
    parser.add_argument("--check", action="store_true", help="verify pinned inputs")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--resume", action="store_true", help="resume the same output directory"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.check or args.print_command or args.run):
        raise SystemExit("choose at least one of --check, --print-command, or --run")
    recipe_path = args.recipe.expanduser().resolve()
    paths_path = args.paths.expanduser().resolve()
    recipe = _read_object(recipe_path)
    paths = load_paths(paths_path)
    if args.check or args.run:
        for message in verify_recipe(recipe, paths):
            print(message)
    command = build_command(
        recipe, paths, python=args.python, resume=args.resume
    )
    if args.print_command:
        print(_display_command(command))
    if args.run:
        output = Path(paths["output"])
        output.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            command,
            cwd=_repository_root(),
            env=build_environment(recipe, paths),
            check=False,
        )
        return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
