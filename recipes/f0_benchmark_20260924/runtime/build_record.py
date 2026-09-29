#!/usr/bin/env python
"""Copy the benchmark evidence from the work tree into the record and write files.json (no model runs).

evidence/
  analysis.json, numbers.json, frontier.json           analysis tables (analyze.py, collect_numbers.py, frontier.py)
  registered_real_s3_equality.json                      byte check of the 442 registered rows against S3
  runs/<set>/<label>/identity.json, rows.jsonl.gz       every run's identity and exact per-row counts
  kaggle/<name>.json, kaggle/<name>_per_case.csv        receipts and merged per-case metric rows
  insitu/insitu_rows.jsonl                              HercUNet own-CLI fidelity rows
  blob_audit/sample_plan.json, summary.json, cube_metrics.jsonl, showcase_picks.json, fragments.json
  figures/*                                             figures used by the README, the paper and the page
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    w, rec = args.work, args.record
    ev = rec / "evidence"
    ev.mkdir(parents=True, exist_ok=True)
    copies = {
        w / "analysis" / "analysis_final.json": ev / "analysis.json",
        w / "analysis" / "numbers.json": ev / "numbers.json",
        w / "analysis" / "frontier_all.json": ev / "frontier.json",
        w / "analysis" / "registered_real_s3_equality.json": ev / "registered_real_s3_equality.json",
        w / "insitu" / "insitu_rows.jsonl": ev / "insitu" / "insitu_rows.jsonl",
        w / "blob_audit" / "sample_plan.json": ev / "blob_audit" / "sample_plan.json",
        w / "blob_audit" / "final" / "summary.json": ev / "blob_audit" / "summary.json",
        w / "blob_audit" / "final" / "cube_metrics.jsonl": ev / "blob_audit" / "cube_metrics.jsonl",
        w / "figures" / "showcase_picks.json": ev / "blob_audit" / "showcase_picks.json",
        w / "blob_audit" / "final" / "fragments.json": ev / "blob_audit" / "fragments.json",
    }
    for name in ("showcase_compressed.jpg", "showcase_open.jpg", "contact_ct.jpg", "contact_m7_as_published.jpg",
                 "contact_f0.jpg", "contact_hercunet_v0.jpg", "deep_mass_by_compression.png", "frontier.png"):
        copies[w / "page" / name] = ev / "figures" / name
    copies[w / "figures" / "teaser_f0_compressed.png"] = ev / "figures" / "teaser_f0_compressed.png"
    for src, dst in copies.items():
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        else:
            print("missing (skipped):", src)
    for set_dir in ("full", "full_p0500", "full_store"):
        for run in sorted((w / set_dir).glob("*")):
            if not (run / "rows.jsonl").exists():
                continue
            out = ev / "runs" / set_dir / run.name
            out.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run / "identity.json", out / "identity.json")
            with (run / "rows.jsonl").open("rb") as src, gzip.open(out / "rows.jsonl.gz", "wb", compresslevel=9) as dst:
                shutil.copyfileobj(src, dst)
    for receipt in sorted((w / "kaggle").glob("*/kaggle.json")):
        name = receipt.parent.name
        (ev / "kaggle").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(receipt, ev / "kaggle" / f"{name}.json")
        shards = sorted((receipt.parent / "score").glob("shard_*/out/metrics_per_case.csv"))
        if shards:
            header, lines = None, []
            for csv in shards:
                rows = [l for l in csv.read_text(encoding="utf-8").splitlines() if l.strip()]
                header = header or rows[0]
                lines += rows[1:]
            (ev / "kaggle" / f"{name}_per_case.csv").write_text("\n".join([header, *sorted(lines)]) + "\n", encoding="utf-8")
    files = []
    for path in sorted(p for p in rec.rglob("*") if p.is_file() and p.name != "files.json" and "__pycache__" not in p.parts):
        files.append({"path": path.relative_to(rec).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    (rec / "files.json").write_text(json.dumps({"schema": "socratic-record-files-v1", "record": rec.name, "files": files},
                                               indent=1) + "\n", encoding="utf-8")
    total = sum(f["bytes"] for f in files)
    print(f"{len(files)} files, {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
