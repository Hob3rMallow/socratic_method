"""CPU bootstrap CI for a frozen audit level or a paired delta between audits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from crossres_pred.voxel.audit_bootstrap import (
    DEFAULT_CONFIDENCE,
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    bootstrap_level,
    bootstrap_paired_delta,
    load_per_patch,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True,
                        help="audit report.json (or its directory) of the candidate")
    parser.add_argument("--reference", type=Path,
                        help="audit report.json of the reference for a paired delta")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--reference-threshold", type=float,
                        help="reference operating threshold (default: --threshold)")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--patch-level", action="store_true",
                        help="resample patches instead of anchor clusters")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    candidate = load_per_patch(args.report)
    result = {
        "level": bootstrap_level(
            candidate,
            args.threshold,
            replicates=args.replicates,
            seed=args.seed,
            confidence=args.confidence,
            cluster=not args.patch_level,
        )
    }
    if args.reference is not None:
        result["paired_delta"] = bootstrap_paired_delta(
            candidate,
            load_per_patch(args.reference),
            threshold=args.threshold,
            reference_threshold=args.reference_threshold,
            replicates=args.replicates,
            seed=args.seed,
            confidence=args.confidence,
            cluster=not args.patch_level,
        )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, args.output)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
