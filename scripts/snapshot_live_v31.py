from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

DEFAULT_LIVE = Path(
    r"D:\work\vesuvius-c\output\crossres_data\m7_xr_v31_pherc0139_dynamic_medial_duration_8192_20260831"
)
TARGET_SAMPLES = 8192


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: missing or invalid history rows")
    samples = [int(row["train"]["cumulative_samples"]) for row in rows]
    if samples != sorted(samples) or len(samples) != len(set(samples)):
        raise ValueError(f"{path}: sample intervals are not strictly increasing")
    return rows


def _metric(row: dict[str, Any]) -> dict[str, Any]:
    train = row["train"]
    val = row["val"]
    return {
        "samples": int(train["cumulative_samples"]),
        "calibrated_threshold": float(val["calibrated_threshold"]),
        "calibrated_dice": float(val["calibrated_dice"]),
        "calibrated_macro_scroll_dice": float(val["calibrated_macro_scroll_dice"]),
        "fixed_0p40_macro_scroll_dice": float(
            val["threshold/0.40/macro_scroll_dice"]
        ),
        "PHerc0814_calibrated_dice": float(
            val["stratum/scroll/PHerc0814/calibrated_dice"]
        ),
        "PHerc1451_calibrated_dice": float(
            val["stratum/scroll/PHerc1451/calibrated_dice"]
        ),
        "calibrated_macro_gain_vs_m7": float(
            val["calibrated_macro_gain_vs_m7_initial"]
        ),
        "minimum_scroll_gain_vs_m7": float(
            val["checkpoint_minimum_scroll_gain_vs_m7_initial"]
        ),
        "trust_projection_active_fraction": float(
            train["m7_trust_region_projection_active_fraction"]
        ),
    }


def _signed(value: float) -> str:
    return f"{value:+.5f}".replace("0.", ".")


def _decimal(value: float) -> str:
    return f"{value:.5f}".replace("0.", ".")


def _write_metrics(root: Path, metrics: list[dict[str, Any]]) -> None:
    best = max(metrics, key=lambda row: float(row["calibrated_macro_scroll_dice"]))
    maximum = max(int(row["samples"]) for row in metrics)
    selection_path = root / "recipes" / "v31" / "selection.json"
    selection = (
        json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_path.is_file()
        else None
    )
    selected = bool(selection and selection.get("status") == "release-candidate-selected")
    value = {
        "schema": "socratic-method-observed-milestones-v1",
        "run": "m7_xr_v31_pherc0139_dynamic_medial_duration_8192_20260831/dynconn_w0p03125_n8192_duration",
        "status": (
            "training-complete-release-candidate-selected"
            if maximum >= TARGET_SAMPLES and selected
            else "training-complete-qualification-pending"
            if maximum >= TARGET_SAMPLES
            else "in-progress"
        ),
        "selection_warning": (
            "The calibrated threshold is at the 0.25 lower boundary and is "
            "censored. It is a duration diagnostic, not the selected operating "
            "point; the independent morphology decision is recorded in selection.json."
        ),
        "best_observed_by_calibrated_macro_scroll_dice": int(best["samples"]),
        "milestones": metrics,
    }
    if selected:
        value["selection_record"] = "recipes/v31/selection.json"
        value["selected_samples"] = int(selection["samples"])
        value["selected_threshold"] = float(selection["threshold"])
    destination = root / "recipes" / "v31" / "observed_metrics.json"
    destination.write_text(
        json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    rows = []
    for row in metrics:
        is_best = int(row["samples"]) == int(best["samples"])
        values = [
            f"{int(row['samples']):,}",
            _decimal(float(row["calibrated_macro_scroll_dice"])),
            _decimal(float(row["fixed_0p40_macro_scroll_dice"])),
            _signed(float(row["calibrated_macro_gain_vs_m7"])),
            _signed(float(row["minimum_scroll_gain_vs_m7"])),
        ]
        if is_best:
            values = [values[0], *(f"\\textbf{{{item}}}" for item in values[1:])]
        rows.append(" & ".join(values) + r" \\")
    tex_lines = [
            "% Generated from the live v31 history by scripts/snapshot_live_v31.py.",
            f"\\newcommand{{\\observedSamples}}{{{maximum}}}",
            f"\\newcommand{{\\observedSamplesText}}{{{maximum:,}}}",
            f"\\newcommand{{\\bestSamples}}{{{int(best['samples'])}}}",
            f"\\newcommand{{\\bestSamplesText}}{{{int(best['samples']):,}}}",
            f"\\newcommand{{\\bestMacro}}{{{float(best['calibrated_macro_scroll_dice']):.5f}}}",
    ]
    if selected:
        tex_lines.extend(
            [
                f"\\newcommand{{\\selectedSamples}}{{{int(selection['samples'])}}}",
                f"\\newcommand{{\\selectedSamplesText}}{{{int(selection['samples']):,}}}",
                f"\\newcommand{{\\selectedThreshold}}{{{float(selection['threshold']):.2f}}}",
            ]
        )
    tex = "\n".join(
        [*tex_lines, "\\newcommand{\\durationRows}{%", *rows, "}", ""]
    )
    (root / "submissions" / "2026-09" / "src" / "generated_run.tex").write_text(
        tex, encoding="utf-8", newline="\n"
    )


def snapshot(live_root: Path) -> int:
    root = _root()
    candidate = live_root / "candidates" / "dynconn_w0p03125_n8192_duration"
    history_path = candidate / "history.jsonl"
    rows = _read_rows(history_path)
    metrics = [_metric(row) for row in rows]
    _write_metrics(root, metrics)

    destination = root / "provenance" / "source" / "live_v31"
    destination.mkdir(parents=True, exist_ok=True)
    copies = {
        candidate / "history.jsonl": destination / "history.jsonl",
        candidate / "run.json": destination / "run.json",
        candidate / "initial_validation.json": destination / "initial_validation.json",
        candidate / "checkpoint_milestones.json": destination
        / "checkpoint_milestones.json",
        live_root / "status.json": destination / "status.json",
        live_root / "recipes" / "dynconn_w0p03125_n8192_duration.json": destination
        / "recipe.original.json",
    }
    for source, target in copies.items():
        if source.is_file():
            shutil.copy2(source, target)
    maximum = max(int(row["samples"]) for row in metrics)
    best = max(metrics, key=lambda row: float(row["calibrated_macro_scroll_dice"]))
    print(
        f"snapshotted {len(metrics)} intervals through {maximum:,} samples; "
        f"best={int(best['samples']):,} at "
        f"{float(best['calibrated_macro_scroll_dice']):.5f}"
    )
    return maximum


def main() -> int:
    parser = argparse.ArgumentParser(description="Snapshot the active v31 duration run")
    parser.add_argument("--live-root", type=Path, default=DEFAULT_LIVE)
    args = parser.parse_args()
    snapshot(args.live_root.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
