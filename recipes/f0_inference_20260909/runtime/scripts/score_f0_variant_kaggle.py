"""Score one declared variants-ladder variant on the Kaggle harness at given thresholds.

Reuses the ladder's own dump/score methods and directory layout
(variants/kaggle/<id>_t0xx/), so results are byte-compatible with the decision
table. Meant for variants the pre-registered top-k rule did not score (e.g.
the family primary) and for a second GPU: set CUDA_VISIBLE_DEVICES before
launching to keep the training GPU untouched. Never acquires gpu.lock.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_f0_variants import Runner, resolve_checkpoint
from run_final_c3_250k import CROSSRES, ROOT, atomic_json, now


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CROSSRES / "configs/f0_variants_20260908.json")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()
    runner = Runner(args.config, None)
    runner.log_path = ROOT / "output/crossres_data/logs/f0_variants_extra_kaggle_20260908.log"
    runner.config["kaggle"]["metric_shards"] = int(args.shards)
    variant = next(v for v in runner.config["variants"] if v["id"] == args.variant)
    spec = resolve_checkpoint(runner.config, variant, runner.averages)
    if not spec.primary.is_file():
        raise SystemExit(f"checkpoint missing: {spec.primary}")
    if spec.extra_members:
        raise SystemExit("ensembles are diagnostic only and never scored on Kaggle")
    results = {}
    for threshold in args.thresholds:
        runner.say(f"{variant['id']}: extra Kaggle scoring at T={threshold:.2f} ({args.shards} shards)")
        pred_dirs = runner.kaggle_dump(variant, spec, float(threshold))
        results[f"{threshold:.2f}"] = runner.kaggle_score(variant, float(threshold), pred_dirs)
    receipt = {"schema": "crossres-f0-variants-extra-kaggle-v1", "variant": variant["id"], "at_utc": now(), "results": results}
    atomic_json(runner.kaggle_dir / f"{variant['id']}_extra_kaggle.json", receipt)
    print(json.dumps({t: {s: {k: v for k, v in r["sets"][s].items() if k.startswith("mean_")} for s in r["sets"]} for t, r in results.items()}, indent=1))


if __name__ == "__main__":
    main()
