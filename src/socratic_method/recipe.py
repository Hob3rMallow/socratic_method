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
CURRENT_MODEL_FILE = "CURRENT_MODEL.json"
LEGACY_RECIPE = Path("recipes") / "v31" / "recipe.json"

# The historical v31 layout: every artifact is a training input.  Kept as the
# documented default for recipes that predate per-artifact contracts.
REQUIRED_PATHS = (
    "train_manifest",
    "validation_manifest",
    "m7_checkpoint",
    "dynamic_medial_connectivity_state",
    "output",
)
ARTIFACT_KINDS = {
    "train_manifest": "train-manifest",
    "validation_manifest": "validation-manifest",
    "dynamic_medial_connectivity_state": "connectivity-state",
}
TRAINING_STAGE = "training"
PASSTHROUGH_PATH_KEYS = ("original_root", "python", "environment")


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


def default_recipe_path(root: Path | None = None) -> Path:
    """The executable recipe named by CURRENT_MODEL.json, else the v31 recipe."""
    root = root or _repository_root()
    pointer = root / CURRENT_MODEL_FILE
    if pointer.is_file():
        try:
            value = _read_object(pointer)
        except (TypeError, ValueError):
            value = {}
        recipe = value.get("executable_recipe")
        if isinstance(recipe, str) and recipe:
            candidate = root / recipe
            if candidate.is_file():
                return candidate
    return root / LEGACY_RECIPE


def artifact_kind(key: str, contract: dict[str, Any]) -> str:
    kind = contract.get("kind")
    if isinstance(kind, str) and kind:
        return kind
    return ARTIFACT_KINDS.get(key, "file")


def artifact_stage(contract: dict[str, Any]) -> str:
    stage = contract.get("stage", TRAINING_STAGE)
    return str(stage) if stage else TRAINING_STAGE


def _artifacts(recipe: dict[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = recipe.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TypeError("recipe artifacts must be an object")
    for key, contract in artifacts.items():
        if not isinstance(contract, dict):
            raise TypeError(f"invalid artifact contract: {key}")
    return artifacts


def required_path_keys(recipe: dict[str, Any]) -> tuple[str, ...]:
    """Path keys a paths file must supply: training-stage artifacts plus output."""
    keys = [
        key
        for key, contract in _artifacts(recipe).items()
        if artifact_stage(contract) == TRAINING_STAGE
    ]
    keys.append("output")
    return tuple(keys)


def optional_path_keys(recipe: dict[str, Any]) -> tuple[str, ...]:
    """Artifact keys that only later stages (evaluation, export) need."""
    return tuple(
        key
        for key, contract in _artifacts(recipe).items()
        if artifact_stage(contract) != TRAINING_STAGE
    )


def load_paths(
    path: Path,
    required: tuple[str, ...] = REQUIRED_PATHS,
    optional: tuple[str, ...] = (),
) -> dict[str, Any]:
    values = _read_object(path)
    missing = sorted(set(required) - set(values))
    if missing:
        raise ValueError(f"{path}: missing path keys: {', '.join(missing)}")
    result = dict(values)
    for key in (*required, *optional):
        if key in values:
            result[key] = str(_resolve_path(str(values[key]), base=path.parent))
    if ("original_root" in values) != ("artifact_root" in values):
        raise ValueError("original_root and artifact_root must be supplied together")
    if "artifact_root" in values:
        result["artifact_root"] = str(
            _resolve_path(str(values["artifact_root"]), base=path.parent)
        )
        result["original_root"] = str(values["original_root"])
    environment = values.get("environment", {})
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in environment.items()
    ):
        raise TypeError(f"{path}: environment must map strings to strings")
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


def _split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("split", "")).lower() == split]


def _sorted_values(rows: list[dict[str, Any]], field: str) -> list[str]:
    return sorted({str(row.get(field)) for row in rows})


def _expect_count(label: str, actual: int, expected: Any) -> None:
    if expected is not None and actual != int(expected):
        raise ValueError(f"{label} changed: expected {int(expected)}, got {actual}")


def _expect_scope(label: str, actual: list[str], expected: Any) -> None:
    if expected is not None and actual != list(expected):
        raise ValueError(f"{label} changed: expected {list(expected)}, got {actual}")


def _verify_train_manifest(path: Path, contract: dict[str, Any]) -> list[str]:
    rows = _iter_manifest(path)
    _expect_count("training manifest row count", len(rows), contract.get("rows"))
    _expect_scope("training scroll scope", _sorted_values(rows, "scroll_id"), contract.get("scrolls"))
    _expect_scope("training record identity", _sorted_values(rows, "record_id"), contract.get("record_ids"))
    train = _split_rows(rows, "train")
    val = _split_rows(rows, "val")
    _expect_count("training split row count", len(train), contract.get("train_rows"))
    _expect_count("in-corpus validation row count", len(val), contract.get("val_rows"))
    _expect_scope("training split scroll scope", _sorted_values(train, "scroll_id"), contract.get("train_scrolls"))
    _expect_scope("in-corpus validation scroll scope", _sorted_values(val, "scroll_id"), contract.get("val_scrolls"))
    _expect_scope("training split record identity", _sorted_values(train, "record_id"), contract.get("train_record_ids"))
    summary = f"{len(rows)} rows"
    if "train_rows" in contract or "val_rows" in contract:
        summary += f" ({len(train)} train / {len(val)} val)"
    return [f"scope train_manifest: {summary}, {len(_sorted_values(rows, 'scroll_id'))} scrolls"]


def _verify_validation_manifest(path: Path, contract: dict[str, Any]) -> list[str]:
    rows = _iter_manifest(path)
    _expect_count("validation manifest row count", len(rows), contract.get("rows"))
    val = _split_rows(rows, "val")
    _expect_count("validation split row count", len(val), contract.get("val_rows"))
    _expect_scope("validation scroll scope", _sorted_values(val, "scroll_id"), contract.get("scrolls"))
    return [f"scope validation: {len(val)} val rows over {_sorted_values(val, 'scroll_id')}"]


def _verify_connectivity_state(path: Path, contract: dict[str, Any]) -> list[str]:
    state = _read_object(path)
    for key in (
        "event_count",
        "fully_owned_event_count",
        "maximum_propagation_steps",
        "maximum_required_connectivity_steps",
    ):
        if int(state.get(key, -1)) != int(contract[key]):
            raise ValueError(f"dynamic connectivity {key} changed")
    return []


_KIND_CHECKS = {
    "train-manifest": _verify_train_manifest,
    "validation-manifest": _verify_validation_manifest,
    "connectivity-state": _verify_connectivity_state,
}


def verify_recipe(recipe: dict[str, Any], paths: dict[str, Any]) -> list[str]:
    if recipe.get("schema") != RECIPE_SCHEMA:
        raise ValueError(f"recipe schema must be {RECIPE_SCHEMA!r}")
    artifacts = _artifacts(recipe)

    messages: list[str] = []
    staged: list[str] = []
    for key, contract in artifacts.items():
        if key not in paths:
            stage = artifact_stage(contract)
            if stage != TRAINING_STAGE:
                messages.append(f"skip {key}: not staged ({stage}-only)")
                continue
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
        staged.append(key)

    for key in staged:
        contract = artifacts[key]
        check = _KIND_CHECKS.get(artifact_kind(key, contract))
        if check is not None:
            messages.extend(check(Path(str(paths[key])), contract))
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
    substitutions = {
        key: str(value) for key, value in paths.items() if isinstance(value, str)
    }
    command = [python or str(paths.get("python", sys.executable))]
    command.extend(value.format_map(substitutions) for value in raw_argv)
    if resume:
        command.append("--resume")
    return command


def build_environment(recipe: dict[str, Any], paths: dict[str, Any]) -> dict[str, str]:
    environment = os.environ.copy()
    for key, value in recipe.get("environment", {}).items():
        environment[str(key)] = str(value)
    for key, value in paths.get("environment", {}).items():
        environment[str(key)] = str(value)
    if "artifact_root" in paths:
        environment[ORIGINAL_ROOT_ENV] = str(paths["original_root"])
        environment[ARTIFACT_ROOT_ENV] = str(paths["artifact_root"])
    return environment


def _display_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and execute a pinned Socratic Method training recipe"
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        default=default_recipe_path(),
        help="recipe JSON (default: the executable recipe named by CURRENT_MODEL.json)",
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
    paths = load_paths(
        paths_path,
        required=required_path_keys(recipe),
        optional=optional_path_keys(recipe),
    )
    if args.check or args.run:
        print(f"recipe: {recipe_path}")
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
