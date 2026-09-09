"""Write a uniform/weighted state-dict average of sample-milestone checkpoints.

Averages never live inside a training run directory: name --output under the
experiment that owns them (e.g. f0_family_20260908/variants/averages/).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crossres_pred.voxel.checkpoint_average import (
    DEFAULT_ALLOWED_IDENTITY_KEYS,
    average_checkpoints,
    members_match_receipt,
    write_average_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        help="training run holding checkpoint_milestone_*.pt")
    parser.add_argument("--samples", type=int, nargs="*", default=[],
                        help="milestone sample counts taken from --run-dir")
    parser.add_argument("--member", type=Path, action="append", default=[],
                        help="explicit member checkpoint (repeatable)")
    parser.add_argument("--weight", type=float, nargs="*",
                        help="positive weights, one per member (default uniform)")
    parser.add_argument("--allow-differing", action="append", default=None,
                        help="dotted run-identity keys allowed to differ "
                             f"(default {list(DEFAULT_ALLOWED_IDENTITY_KEYS)})")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    members: list[Path] = []
    if args.samples:
        if args.run_dir is None:
            raise SystemExit("--samples requires --run-dir")
        members += [
            args.run_dir / f"checkpoint_milestone_{value:08d}.pt"
            for value in args.samples
        ]
    members += list(args.member)
    if not members:
        raise SystemExit("no members: pass --run-dir/--samples or --member")
    allowed = (
        tuple(args.allow_differing)
        if args.allow_differing is not None
        else DEFAULT_ALLOWED_IDENTITY_KEYS
    )
    receipt_path = args.output.with_name(args.output.name + ".json")
    if args.output.exists():
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if members_match_receipt(receipt, members):
                print(json.dumps({"skipped": "identical average already exists",
                                  **receipt}, indent=2, sort_keys=True))
                return 0
        raise SystemExit(f"refusing to overwrite a different average: {args.output}")
    payload = average_checkpoints(members, args.weight, allowed_identity_keys=allowed)
    receipt = write_average_checkpoint(payload, args.output)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
