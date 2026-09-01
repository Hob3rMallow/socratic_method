#!/usr/bin/env python3
"""Train the requested 1024/2048/4096/8192 duration ladder."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_pherc0139_dynamic_medial_connectivity_weight_sweep as v28

SCHEMA = "crossres-pherc0139-dynamic-medial-duration-8192-result-v1"
PLAN_SCHEMA = "crossres-pherc0139-dynamic-medial-duration-8192-plan-v1"
DEFAULT_PLAN = Path(__file__).resolve().parents[1] / (
    "configs/m7_xr_v31_pherc0139_dynamic_medial_duration_8192_20260831.json"
)
MILESTONES = (1024, 2048, 4096, 8192)
WEIGHT = 0.03125


def _io() -> Any:
    return v28.v27.v21.v20.growth.v19


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _pinned_path(repo: Path, spec: dict[str, Any], field: str) -> Path:
    path = _io()._resolve(repo, str(spec[field]))
    hash_field = "sha256" if field == "path" else f"{field}_sha256"
    if _io()._sha256(path) != str(spec[hash_field]):
        raise ValueError(f"pinned artifact changed: {field}")
    return path


def validate_plan(
    repo: Path, plan_path: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    plan = _io()._read_object(plan_path)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("version")
        != "v31.1-pherc0139-dynamic-medial-duration-8192"
        or int(plan.get("seed", -1)) != 1203
    ):
        raise ValueError("unsupported 8192-sample duration plan")
    base_path = _pinned_path(repo, plan["base_weight_sweep"], "path")
    base_plan, v21_plan, v20_plan, ancestor_plan, ancestor_recipe, corpus = (
        v28.validate_plan(repo, base_path)
    )
    base_recipe_path = _pinned_path(repo, plan["base_candidate"], "recipe")
    _pinned_path(repo, plan["base_candidate"], "checkpoint")
    base_recipe = _io()._read_object(base_recipe_path)
    v28.v27.v21.v20.growth.v19.V17.assert_objective_fingerprint(
        base_recipe["objective"]
    )
    axis = plan["single_axis"]
    pilot = plan["pilot"]
    base = plan["base_candidate"]
    if (
        axis.get("field") != "training_samples"
        or int(axis.get("baseline", -1)) != 1024
        or int(axis.get("target", -1)) != 8192
        or tuple(axis.get("observation_milestones", ())) != MILESTONES
        or int(pilot.get("training_samples", -1)) != 8192
        or int(pilot.get("validation_interval_samples", -1)) != 1024
        or tuple(pilot.get("snapshot_samples", ())) != MILESTONES
        or pilot.get("fresh_released_m7_initialization") is not True
        or float(pilot.get("full_corpus_passes", -1)) != 2.0
        or float(pilot.get("dynamic_medial_connectivity_weight", -1)) != WEIGHT
        or base.get("raw_no_blend") is not True
        or float(base.get("dynamic_medial_connectivity_weight", -1)) != WEIGHT
        or int(base.get("training_samples", -1)) != 1024
        or plan["candidate"].get("label")
        != "dynconn_w0p03125_n8192_duration"
        or int(base_recipe["optimization"]["training_samples"]) != 1024
        or tuple(base_recipe["optimization"]["snapshot_samples"]) != (1024,)
        or float(
            base_recipe["objective"]["dynamic_medial_connectivity_weight"]
        )
        != WEIGHT
        or int(base_recipe["corpus"]["expected_train_rows"]) != 4096
    ):
        raise ValueError("8192-sample single-axis contract changed")
    return (
        plan,
        base_plan,
        v21_plan,
        v20_plan,
        ancestor_plan,
        ancestor_recipe,
        corpus,
    )


def build_recipe(
    *,
    repo: Path,
    plan: dict[str, Any],
    base_plan: dict[str, Any],
    v21_plan: dict[str, Any],
    v20_plan: dict[str, Any],
    ancestor_plan: dict[str, Any],
    ancestor_recipe: dict[str, Any],
    corpus: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    adapted = copy.deepcopy(base_plan)
    adapted["output"] = str(plan["output"])
    adapted["pilot"].update(
        training_samples=8192,
        validation_interval_samples=1024,
        snapshot_samples=list(MILESTONES),
    )
    candidate = {
        "label": str(plan["candidate"]["label"]),
        "dynamic_medial_connectivity_weight": WEIGHT,
        "hypothesis": str(plan["candidate"]["hypothesis"]),
    }
    adapted["candidates"] = [candidate]
    recipe, output = v28.candidate_recipe(
        repo=repo,
        plan=adapted,
        v21_plan=v21_plan,
        v20_plan=v20_plan,
        base_plan=ancestor_plan,
        base_recipe=ancestor_recipe,
        corpus=corpus,
        candidate=candidate,
    )
    recipe["version"] = "v31.1-pherc0139-dynamic-medial-duration-8192"
    recipe["scientific_change"] = {
        "contract": "training-duration-only-from-v28-winner-v2",
        "description": (
            "Restart the winning raw weight-0.03125 candidate from released M7 "
            "and vary only cumulative training duration through 8192 samples."
        ),
        "single_axis": copy.deepcopy(plan["single_axis"]),
        "base_candidate": copy.deepcopy(plan["base_candidate"]),
        "candidate": copy.deepcopy(plan["candidate"]),
    }
    return recipe, output


def assert_duration_only(
    *, repo: Path, plan: dict[str, Any], recipe: dict[str, Any]
) -> None:
    baseline = _io()._read_object(
        _io()._resolve(repo, str(plan["base_candidate"]["recipe"]))
    )
    candidate = copy.deepcopy(recipe)
    for document in (baseline, candidate):
        document.pop("version")
        document.pop("scientific_change")
        document.pop("output")
        optimization = document["optimization"]
        optimization.pop("training_samples")
        optimization.pop("snapshot_samples")
        optimization.pop("corpus_passes")
    if candidate != baseline:
        raise ValueError("generated recipe changes more than training duration")
    optimization = recipe["optimization"]
    if (
        int(optimization["training_samples"]) != 8192
        or int(optimization["validation_interval_samples"]) != 1024
        or tuple(optimization["snapshot_samples"]) != MILESTONES
        or float(optimization["corpus_passes"]) != 2.0
        or float(recipe["objective"]["dynamic_medial_connectivity_weight"])
        != WEIGHT
    ):
        raise ValueError("generated 8192-sample duration values are incorrect")


def _status(path: Path, **fields: Any) -> None:
    _io()._atomic_json(
        path,
        {
            "schema": SCHEMA,
            "updated_at_utc": _utc_now(),
            "human_report_created": False,
            **fields,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_PLAN.parents[2])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    plan_path = args.plan.resolve()
    (
        plan,
        base_plan,
        v21_plan,
        v20_plan,
        ancestor_plan,
        ancestor_recipe,
        corpus,
    ) = validate_plan(repo, plan_path)
    recipe, output = build_recipe(
        repo=repo,
        plan=plan,
        base_plan=base_plan,
        v21_plan=v21_plan,
        v20_plan=v20_plan,
        ancestor_plan=ancestor_plan,
        ancestor_recipe=ancestor_recipe,
        corpus=corpus,
    )
    assert_duration_only(repo=repo, plan=plan, recipe=recipe)
    command = v28.v27.v21.v20.growth.v19.V17.build_training_command(
        python=Path(sys.executable),
        repo=repo,
        recipe=recipe,
        output=output,
        resume=(output / "run.json").is_file(),
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "single_axis": plan["single_axis"],
                    "output": str(output),
                    "command": command,
                    "human_report_created": False,
                },
                indent=2,
            )
        )
        return 0

    root = _io()._resolve(repo, str(plan["output"]))
    root.mkdir(parents=True, exist_ok=True)
    recipe_path = root / "recipes" / f"{plan['candidate']['label']}.json"
    status_path = root / "status.json"
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "16"
    os.environ["NUMEXPR_MAX_THREADS"] = "16"
    try:
        _io()._write_immutable_recipe(recipe_path, recipe)
        power_limit = v28.v27.v21.v20.growth.v19.V17._query_gpu_power_limit()
        if power_limit > v28.v27.v21.v20.growth.v19.V17.MAX_GPU_POWER_W:
            raise RuntimeError("GPU power limit exceeds the supported training ceiling")
        completed = _io()._completed_samples(output)
        if completed > 8192:
            raise ValueError("candidate exceeds the pinned sample budget")
        _status(
            status_path,
            state="training" if completed < 8192 else "verifying",
            plan=str(plan_path),
            plan_sha256=_io()._sha256(plan_path),
            recipe=str(recipe_path),
            recipe_sha256=_io()._sha256(recipe_path),
            output=str(output),
            completed_samples=completed,
            gpu_power_limit_w=power_limit,
            single_axis=plan["single_axis"],
        )
        if completed < 8192:
            completed_process = subprocess.run(command, cwd=repo, check=False)
            if completed_process.returncode != 0:
                raise RuntimeError(
                    f"8192-sample duration training exited "
                    f"{completed_process.returncode}"
                )
        completed = _io()._completed_samples(output)
        paths = [
            output / f"checkpoint_milestone_{samples:08d}.pt"
            for samples in MILESTONES
        ]
        if completed != 8192 or any(not path.is_file() for path in paths):
            raise RuntimeError("8192-sample training did not commit every milestone")
        result = {
            "schema": SCHEMA,
            "state": "checkpoints-ready-for-full-validation",
            "created_at_utc": _utc_now(),
            "human_report_created": False,
            "plan": str(plan_path),
            "plan_sha256": _io()._sha256(plan_path),
            "recipe": str(recipe_path),
            "recipe_sha256": _io()._sha256(recipe_path),
            "run_provenance": str(output / "run.json"),
            "run_provenance_sha256": _io()._sha256(output / "run.json"),
            "completed_samples": completed,
            "gpu_power_limit_w": power_limit,
            "milestones": [
                {
                    "samples": samples,
                    "checkpoint": str(path),
                    "checkpoint_sha256": _io()._sha256(path),
                }
                for samples, path in zip(MILESTONES, paths, strict=True)
            ],
        }
        result_path = root / "result.json"
        _io()._atomic_json(result_path, result)
        _status(status_path, **result, result=str(result_path))
    except Exception as error:
        _status(
            status_path,
            state="failed",
            plan=str(plan_path),
            output=str(output),
            completed_samples=_io()._completed_samples(output),
            error=f"{type(error).__name__}: {error}",
        )
        raise
    print(result_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
