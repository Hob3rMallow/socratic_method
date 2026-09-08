"""Derive recipes/f0/observed_metrics.json from the sealed C3-F0 record.

Every value is copied verbatim from ``recipes/final_c3_250k_20260907/analysis.json``
(``trajectory``) and ``provenance/checkpoint_milestones.json``; nothing is rounded
or recomputed.  Re-run after an intentional record update, never by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RECORD = Path("recipes") / "final_c3_250k_20260907"
OUTPUT = Path("recipes") / "f0" / "observed_metrics.json"
MILESTONE_FIELDS = (
    "samples",
    "eligible",
    "frozen_macro_dice_at_035",
    "calibrated_macro_dice",
    "calibrated_threshold",
    "train_loss",
    "cross_entropy",
    "dice_loss",
    "anchor_kl",
    "learning_rate_end",
    "wide15_val_dice_at_050",
    "wide15_calibrated_dice",
    "relative_l2_from_m7",
    "fixed18_historical_reference_bridges",
    "fixed18_historical_reference_line_coverage",
    "fixed18_components",
    "checkpoint_sha256",
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def build_observed_metrics(root: Path) -> dict[str, Any]:
    analysis = _read(root / RECORD / "analysis.json")
    manifest = _read(root / RECORD / "manifest.json")
    milestones_record = _read(root / RECORD / "provenance" / "checkpoint_milestones.json")
    bytes_by_samples = {
        int(row["requested_samples"]): int(row["bytes"])
        for row in milestones_record["records"]
    }
    trajectory = analysis["trajectory"]
    if not isinstance(trajectory, list) or len(trajectory) != 25:
        raise ValueError("analysis.json trajectory must list the 25 milestones")
    milestones = []
    for row in trajectory:
        entry = {field: row[field] for field in MILESTONE_FIELDS}
        entry["checkpoint_bytes"] = bytes_by_samples[int(row["samples"])]
        milestones.append(entry)
    samples = [int(row["samples"]) for row in milestones]
    if samples != sorted(samples) or len(set(samples)) != len(samples):
        raise ValueError("milestones are not strictly increasing")
    selection = manifest["selection"]
    best_fixed = max(milestones, key=lambda row: float(row["frozen_macro_dice_at_035"]))
    best_calibrated = max(milestones, key=lambda row: float(row["calibrated_macro_dice"]))
    return {
        "schema": "socratic-method-observed-milestones-v1",
        "run": "final_c3_250k_20260905/arms/F0",
        "status": "training-complete-frozen-model-selected",
        "selection_warning": (
            "The global fixed-T=0.35 maximum (110k) and the calibrated optimum "
            "(T=0.30 at most milestones) are training observations. The frozen "
            "model is the preregistered late-checkpoint choice, 200k at T=0.35."
        ),
        "best_observed_by_fixed_035_macro_scroll_dice": int(best_fixed["samples"]),
        "best_observed_by_calibrated_macro_scroll_dice": int(best_calibrated["samples"]),
        "release_selection": {
            "checkpoint_samples": int(selection["selected_samples"]),
            "operating_threshold": float(manifest["model"]["operating_threshold"]),
            "basis": selection["rule"],
            "qualification": "release_qualification.json",
        },
        "milestones": milestones,
    }


def main() -> int:
    root = _root()
    value = build_observed_metrics(root)
    destination = root / OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {destination} with {len(value['milestones'])} milestones")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
