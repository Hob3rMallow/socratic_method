"""Human-review panels for release-ladder milestones: locked 16 + six-cube 18.

For every durable ``checkpoint_milestone_*.pt`` of an arm this runs the two
standard visual gates the operator reviews by eye and renders them as the
same side-by-side page the v31 duration ladder used:

* the sixteen locked PHerc0139 growth-gate slices (CT | teacher | published
  M7 | model at the displayed threshold) via
  ``evaluate_pherc0139_growth_gates.py`` (raw student, TTA, no M7 blend);
* the six blind PHerc1447 cubes at their eighteen fixed review views
  (CT | jackpot 08-15 reference | published M7 | model) via
  ``generate_checkpoint_report.py`` (halo 32, TTA, T = 0.45);
* the v29 joint threshold audit (CPU) for the machine table under the
  threshold selector.

Outputs under ``<arm>/ladder_eval/panels/``: ``milestone_<N>/{locked16,
pherc1447,audit.json}``, ``index.html`` (all milestones side by side, the
threshold selector switches every model panel), ``manifest.json``.  Runs
alongside training on purpose (bounded: ~5 min of GPU per milestone).

    python scripts/render_release_milestone_panels.py --arm R_full
    python scripts/render_release_milestone_panels.py --arm R_full --milestones 5000
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CROSSRES = ROOT / "crossres_pred"
SCRIPTS = CROSSRES / "scripts"
DATA = ROOT / "output" / "crossres_data"
CONFIG_PATH = CROSSRES / "configs" / "release_ladder_20260902.json"

GATES = DATA / "pherc0139_growth_gates16_20260831/gate_plan_input.json"
GATE_ATLAS = (
    DATA / "voxel_atlas_native_v19_medial_micro_pherc0139_4096_20260830/atlas_catalog.json"
)
GATE_RECORD_ID = "pherc0139-native-fine-teacher-2p399-to-9p362-v11p1"
BLIND_SOURCE = DATA / "pherc1447_blind_audit/six_cube_subset"
BLIND_REFERENCE_ROOT = (
    DATA / "pherc1447_blind_audit/jackpot_all_sources_20260815_t035_report_20260901/inference"
)
# the eighteen fixed human-review views (the v29 report's list, unchanged)
FIXED_SLICES = (
    "z12032_y03968_x03072:z:12032",
    "z12032_y03968_x03072:z:12043",
    "z12032_y03968_x03072:z:12045",
    "z12032_y03968_x03072:z:12050",
    "z12032_y03968_x03072:z:12056",
    "z12032_y03968_x03072:z:12096",
    "z12032_y03968_x03072:z:12153",
    "z12160_y03968_x03200:z:12193",
    "z12160_y03968_x03200:z:12205",
    "z12160_y03968_x03200:z:12224",
    "z12160_y04224_x02944:z:12161",
    "z12160_y04224_x02944:z:12171",
    "z12160_y04224_x02944:z:12224",
    "z12160_y04224_x02944:z:12256",
    "z12800_y04096_x03456:z:12864",
    "z12928_y04096_x03328:z:12936",
    "z12928_y04096_x03328:z:12938",
    "z12928_y04096_x03328:z:12992",
)
THRESHOLDS = (0.25, *(round(value / 100, 2) for value in range(38, 51)))
DISPLAY_THRESHOLD = 0.45  # the release operating threshold
BLIND_THRESHOLD = 0.45


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _say(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [panels] {message}", flush=True)


def _run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    _say("+ " + " ".join(str(part) for part in command[1:4]) + " ...")
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
        stream.write(" ".join(str(part) for part in command) + "\n")
        stream.flush()
        completed = subprocess.run(
            [str(part) for part in command], cwd=str(CROSSRES),
            stdout=stream, stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode != 0:
        raise SystemExit(f"command failed ({completed.returncode}); see {log}")


def _durable(path: Path) -> bool:
    if not path.is_file():
        return False
    size = path.stat().st_size
    time.sleep(2)
    return size > 0 and path.stat().st_size == size


def milestone_checkpoints(arm_dir: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in sorted(arm_dir.glob("checkpoint_milestone_*.pt")):
        samples = int(path.stem.rsplit("_", 1)[1])
        if _durable(path):
            found[samples] = path
    return found


def locked_command(python: Path, checkpoint: Path, output: Path, device: str) -> list[str]:
    return [
        str(python), str(SCRIPTS / "evaluate_pherc0139_growth_gates.py"),
        "--gates", str(GATES),
        "--atlas-catalog", str(GATE_ATLAS),
        "--record-id", GATE_RECORD_ID,
        "--checkpoint", str(checkpoint),
        "--output", str(output),
        "--evaluation-role", "candidate",
        "--thresholds", *(str(value) for value in THRESHOLDS),
        "--device", device,
        "--max-cpu-threads", "16",
    ]


def blind_command(python: Path, checkpoint: Path, output: Path, device: str) -> list[str]:
    command = [
        str(python), str(SCRIPTS / "generate_checkpoint_report.py"),
        "--source", str(BLIND_SOURCE),
        "--checkpoint", str(checkpoint),
        "--output", str(output),
        "--threshold", f"{BLIND_THRESHOLD:.2f}",
        "--halo", "32",
        "--device", device,
        "--amp-dtype", "bfloat16",
        "--tta",
        "--max-cpu-threads", "16",
        "--slices-per-axis", "3",
        "--reference-student", str(BLIND_REFERENCE_ROOT),
    ]
    for specification in FIXED_SLICES:
        command.extend(("--fixed-slice", specification))
    return command


def audit_command(python: Path, *, blind: Path, locked: Path, output: Path) -> list[str]:
    return [
        str(python), str(SCRIPTS / "audit_m7_xr_v29_joint_thresholds.py"),
        "--probability-grid", str(blind / "inference/probability"),
        "--published-grid", str(BLIND_SOURCE / "cubes_PRED"),
        "--reference-grid", str(BLIND_REFERENCE_ROOT / "cubes_PRED"),
        "--locked-evaluation", str(locked),
        "--output", str(output),
        "--start", "0.38", "--stop", "0.50", "--step", "0.01",
        "--morphology-thresholds", *(str(value) for value in THRESHOLDS),
    ]


def audit_summary(path: Path) -> list[dict[str, Any]]:
    payload = _read(path)
    morphology = {
        round(float(row["threshold"]), 2): row for row in payload["pherc1447_morphology"]
    }
    result: list[dict[str, Any]] = []
    for row in payload["sweep"]:
        threshold = round(float(row["threshold"]), 2)
        if threshold not in morphology or "pherc0139_locked16" not in row:
            continue
        locked = row["pherc0139_locked16"]
        blind = row["pherc1447"]
        shape = morphology[threshold]
        result.append(
            {
                "threshold": threshold,
                "locked_component_matches": int(locked["component_matches"]),
                "locked_anti_blob_passes": int(locked["anti_blob_passes"]),
                "locked_mean_dice": float(locked["mean_dice"]),
                "pherc1447_foreground_ratio_vs_v15": float(
                    blind["foreground_ratio_vs_reference"]
                ),
                "pherc1447_reference_recall": float(blind["recall_vs_reference"]),
                "pherc1447_anti_blob_passed": bool(shape["gates"]["passed"]),
                "pherc1447_interior_regression_cubes": int(
                    shape["interior_regression_cube_count"]
                ),
                "pherc1447_thickness_regression_cubes": int(
                    shape["thickness_regression_cube_count"]
                ),
            }
        )
    return result


def build_record(python: Path, samples: int, checkpoint: Path, root: Path, device: str) -> dict[str, Any]:
    locked_root = root / "locked16"
    blind_root = root / "pherc1447"
    locked_path = locked_root / "evaluation.json"
    blind_report = blind_root / "report/report.json"
    audit_path = root / "audit.json"
    log = root / "panels.log"
    if not locked_path.is_file():
        _run(locked_command(python, checkpoint, locked_root, device), log)
    if not blind_report.is_file():
        _run(blind_command(python, checkpoint, blind_root, device), log)
    if not audit_path.is_file():
        _run(audit_command(python, blind=blind_root, locked=locked_path, output=audit_path), log)
    locked = _read(locked_path)
    composition = locked.get("prediction_composition", {})
    if (
        composition.get("contract") != "student-probability-only-v1"
        or float(composition.get("m7_base_blend", -1)) != 0.0
    ):
        raise ValueError(f"milestone {samples}: locked evaluation is not raw")
    return {
        "samples": samples,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": locked["checkpoint_sha256"],
        "locked_evaluation": str(locked_path),
        "blind_inference": str(blind_root / "inference"),
        "blind_report": str(blind_report),
        "audit": str(audit_path),
        "threshold_summary": audit_summary(audit_path),
    }


def _renderer():
    spec = importlib.util.spec_from_file_location(
        "duration_renderer", SCRIPTS / "render_m7_xr_v31_duration_ladder_report.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS))
    spec.loader.exec_module(module)
    return module


def render(arm_id: str, records: list[dict[str, Any]], output: Path) -> Path:
    renderer = _renderer()
    renderer.OUTPUT = output
    renderer.GATE_INPUT = GATES
    renderer.GATE_ATLAS = GATE_ATLAS
    renderer.GATE_RECORD_ID = GATE_RECORD_ID
    renderer.BLIND_SOURCE = BLIND_SOURCE
    renderer.BLIND_REFERENCE = BLIND_REFERENCE_ROOT / "cubes_PRED"
    renderer.THRESHOLDS = THRESHOLDS
    renderer.DEFAULT_THRESHOLD = DISPLAY_THRESHOLD
    panel_data: dict[str, dict[str, dict[str, str]]] = {}
    locked_html, locked_manifest = renderer._render_locked(records, panel_data)
    blind_html, blind_manifest = renderer._render_blind(records, panel_data)
    index = renderer._write_report(
        result={"records": records}, locked_rows=locked_html,
        blind_rows=blind_html, panel_data=panel_data,
    )
    page = index.read_text(encoding="utf-8")
    page = (
        page.replace("M7-XR duration ladder review", f"Release ladder {arm_id} milestones")
        .replace("M7-XR raw duration ladder", f"Release ladder — {arm_id} milestones (raw student)")
        .replace("v15 anti-blob reference", "jackpot 08-15 reference")
        .replace("v15 anti-blob", "jackpot 08-15")
        .replace("foreground / v15", "foreground / jackpot")
        .replace("v15 recall", "jackpot recall")
        .replace("v15 is the prior anti-blob reference", "The jackpot 08-15 student is the anti-blob reference")
        .replace("The five duration models update together", "The milestone columns update together")
    )
    index.write_text(page, encoding="utf-8")
    _write(
        output / "manifest.json",
        {
            "schema": "crossres-release-ladder-milestone-panels-v1",
            "arm_id": arm_id,
            "research_only": True,
            "model_composition": "raw-student-probability-only; no M7 blend",
            "display_threshold": DISPLAY_THRESHOLD,
            "blind_threshold": BLIND_THRESHOLD,
            "thresholds": list(THRESHOLDS),
            "fixed_slices": list(FIXED_SLICES),
            "records": records,
            "locked16_rows": locked_manifest,
            "pherc1447_fixed18_rows": blind_manifest,
            "index": str(index),
            "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="R_full")
    parser.add_argument("--milestones", type=int, nargs="*", default=None,
                        help="sample milestones to include (default: every durable one)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arm-dir", type=Path, default=None)
    arguments = parser.parse_args()
    config = _read(CONFIG_PATH)
    arm_dir = arguments.arm_dir or ROOT / config["output_root"] / "arms" / arguments.arm
    checkpoints = milestone_checkpoints(arm_dir)
    wanted = sorted(arguments.milestones) if arguments.milestones else sorted(checkpoints)
    missing = [m for m in wanted if m not in checkpoints]
    if missing:
        raise SystemExit(f"{arguments.arm}: milestones not durable yet: {missing}")
    if not wanted:
        raise SystemExit(f"{arguments.arm}: no durable milestones in {arm_dir}")
    output = arm_dir / "ladder_eval" / "panels"
    python = Path(sys.executable)
    records = []
    for samples in wanted:
        _say(f"{arguments.arm} milestone {samples:,}")
        records.append(
            build_record(python, samples, checkpoints[samples],
                         output / f"milestone_{samples:08d}", arguments.device)
        )
    index = render(arguments.arm, records, output)
    print(index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
