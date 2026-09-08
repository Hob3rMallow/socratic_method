"""Mirror verified continuity text/source evidence to its separate official record."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import run_f0_repair_context as context

ROOT = context.ROOT / "output/crossres_data/f0_continuity_20260907"
OFFICIAL = context.ROOT.parent / "socratic_method"
RECORD = OFFICIAL / "recipes/f0_continuity_20260907"


def main():
    verification = context.read(ROOT / "verification.json")
    if verification["status"] != "passed":
        raise ValueError("continuity verification is incomplete")
    context.verify_file(verification["final_report"])
    for item in context.read(ROOT / "package_control_verified/continuity_receipt.json")["plan"]["input_files"]:
        context.verify_file(item)
    mapping = {name: ROOT / name for name in (
        "PLAN.md", "selected_policy.json", "visual_review.json", "decision.json", "results.json",
        "verification.json", "package_tests.xml", "package_tests_attempt1.xml",
        "package_tests_before_apron_audit.xml", "final_check_only_plan.json")}
    mapping["README.md"] = RECORD / "README.md"
    mapping[".gitattributes"] = RECORD / ".gitattributes"
    for name in ("development_candidate_v1", "development_candidate_v2", "fresh_control_candidate_v2"):
        for filename in ("request.json", "state.json", "summary.json", "corridors.json", "components.json", "contact_audit.json"):
            mapping[f"provenance/{name}/{filename}"] = ROOT / name / filename
    for filename in ("request.json", "state.json"):
        mapping[f"provenance/fresh_control/{filename}"] = ROOT / "fresh_control" / filename
    for filename in ("provenance.json", "source_manifest.json"):
        mapping[f"provenance/fresh_control/inference/{filename}"] = ROOT / "fresh_control/inference" / filename
    mapping["provenance/fresh_control/native_repair_receipt.json"] = ROOT / "fresh_control/native_baseline/repair_receipt.json"
    for filename in ("continuity_receipt.json", "plan.json", "progress.json"):
        mapping[f"provenance/package_control/{filename}"] = ROOT / "package_control_verified" / filename
    for filename in ("failure.json", "plan.json"):
        mapping[f"provenance/initial_package_audit_failure/{filename}"] = ROOT / "package_control.partial-57780" / filename
    for filename in ("probability_continuity.py", "evaluate_probability_continuity.py"):
        mapping[f"source/original_v1/{filename}"] = ROOT / "development_candidate_v1/source" / filename
    for directory in ("development_probe_p0.15", "development_probe_p0.20"):
        mapping[f"provenance/{directory}/proposals.json"] = ROOT / directory / "proposals.json"
    for filename in ("probability_continuity.py", "probability_continuity_v2.py", "probe_probability_continuity.py",
                     "evaluate_probability_continuity.py", "run_continuity_fresh_control.py",
                     "build_f0_continuity_report.py", "record_f0_continuity_evidence.py"):
        mapping[f"source/experiment/{filename}"] = Path(__file__).parent / filename
    for filename in ("test_probability_continuity.py", "test_probability_continuity_v2.py"):
        mapping[f"source/experiment_tests/{filename}"] = Path(__file__).resolve().parents[1] / "tests" / filename
    for filename in ("continuity.py", "continuity_core.py", "continuity_paths.py"):
        mapping[f"source/package/{filename}"] = OFFICIAL / "src/socratic_method" / filename
    for filename in ("test_continuity.py", "test_continuity_core.py", "test_continuity_paths.py"):
        mapping[f"source/package_tests/{filename}"] = OFFICIAL / "tests" / filename
    for filename in ("docs/continuity.md", "README.md", "pyproject.toml"):
        mapping[f"source/repository/{filename}"] = OFFICIAL / filename
    items = []
    for relative, source in sorted(mapping.items()):
        target = (RECORD / relative).resolve()
        if RECORD.resolve() not in target.parents:
            raise ValueError("target escapes declared evidence record")
        identity = context.inventory(source)
        if target.exists() and context.sha256(target) != identity["sha256"]:
            raise FileExistsError(f"preserve existing different evidence: {target}")
        items.append({"relative_path": relative, "original_path": str(source), "bytes": identity["bytes"], "sha256": identity["sha256"]})
    for item in items:
        target = RECORD / item["relative_path"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item["original_path"], target)
        if target.stat().st_size != item["bytes"] or context.sha256(target) != item["sha256"]:
            raise ValueError(f"copy verification failed: {target}")
    manifest = {"schema": "f0-continuity-official-evidence-inventory-v1", "files": items,
                "verified_utc": datetime.now(UTC).isoformat(), "local_report": context.inventory(ROOT / "index.html"),
                "model_and_image_arrays_copied": False, "git_commit_or_public_upload": False}
    path = RECORD / "files.json"
    if path.exists():
        if context.read(path)["files"] != items:
            raise FileExistsError("existing official inventory differs")
    else:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
    context.atomic_json(ROOT / "official_copy_receipt.json", {"status": "verified", "copied_and_reverified_files": len(items),
                        "record": str(RECORD), "manifest": context.inventory(path), "local_only": True})
    print(f"Verified {len(items)} official continuity evidence/source files: {RECORD}", flush=True)


if __name__ == "__main__":
    main()
