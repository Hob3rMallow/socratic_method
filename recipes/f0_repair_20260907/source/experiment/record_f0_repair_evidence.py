"""Copy the completed repair evidence into its new official record, without overwrites."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import run_f0_repair_context as context


def main() -> None:
    official = context.ROOT.parent / "socratic_method"
    record = official / "recipes/f0_repair_20260907"
    report = context.OUTPUT
    stage = context.ROOT / ".tmp/f0-repair-package-20260907"
    verification = context.read(report / "verification.json")
    if verification["status"] != "passed":
        raise ValueError("report verification is incomplete")
    context.verify_file(verification["final_report"])
    mapping = {name: report / name for name in (
        "profile.json", "results.json", "decision.json", "visual_review.json", "request.json",
        "selected_policy.json", "verification.json", "tests.xml")}
    mapping["README.md"] = stage / "README.md"
    mapping[".gitattributes"] = stage / ".gitattributes"
    mapping["provenance/inference_state.json"] = report / "inference_state.json"
    mapping["provenance/wrapper_integration_receipt.json"] = report / "wrapper_control_v2/repair_receipt.json"
    for block in ("development", "control"):
        names = list(context.PRESETS) if block == "development" else ["report_baseline", "extended_safe"]
        for preset in names:
            mapping[f"provenance/{block}_{preset}_summary.json"] = report / f"repairs/{block}/{preset}_summary.json"
        mapping[f"provenance/{block}_comparison_state.json"] = report / f"repairs/{block}/state.json"
        mapping[f"provenance/{block}_inference_provenance.json"] = report / f"inference_{block}/provenance.json"
        mapping[f"provenance/{block}_track_cases.json"] = report / f"review/{block}/extended_safe/cases.json"
        for name in ("summary", "details"):
            mapping[f"provenance/{block}_contacts_{name}.json"] = report / f"contact_audit/{block}/{name}.json"
    mapping["source/repair.py"] = official / "src/socratic_method/repair.py"
    mapping["source/test_repair.py"] = official / "tests/test_repair.py"
    for name in ("run_f0_repair_context.py", "evaluate_f0_repair_context.py", "render_f0_repair_review.py",
                 "audit_f0_repair_contacts.py", "build_f0_repair_report.py", "record_f0_repair_evidence.py"):
        mapping[f"source/experiment/{name}"] = Path(__file__).parent / name
    for name in ("test_f0_repair_context.py", "test_f0_repair_evaluation.py", "test_f0_repair_contacts.py"):
        mapping[f"source/tests/{name}"] = Path(__file__).resolve().parents[1] / "tests" / name
    items = []
    for relative, source in sorted(mapping.items()):
        destination = (record / relative).resolve()
        if record.resolve() not in destination.parents:
            raise ValueError("record target escapes the explicit recipe directory")
        identity = context.inventory(source)
        if destination.exists() and context.sha256(destination) != identity["sha256"]:
            raise FileExistsError(f"preserve different existing official content: {destination}")
        items.append({"relative_path": relative, "original_path": str(source),
                      "bytes": identity["bytes"], "sha256": identity["sha256"]})
    for item in items:
        destination = record / item["relative_path"]
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item["original_path"], destination)
        if destination.stat().st_size != item["bytes"] or context.sha256(destination) != item["sha256"]:
            raise ValueError(f"official copy did not verify: {destination}")
    manifest_path = record / "files.json"
    manifest = {"schema": "f0-repair-official-evidence-inventory-v1", "files": items,
                "verified_at_utc": datetime.now(UTC).isoformat(), "local_report": context.inventory(report / "index.html"),
                "weights_or_image_arrays_copied": False, "git_commit_or_public_upload": False}
    if manifest_path.exists():
        if context.read(manifest_path)["files"] != items:
            raise FileExistsError("preserve the existing official evidence inventory")
    else:
        with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
    context.atomic_json(report / "official_copy_receipt.json", {
        "schema": "f0-repair-official-copy-receipt-v1", "status": "verified",
        "record": str(record), "copied_and_reverified_files": len(items),
        "manifest": context.inventory(manifest_path), "local_only": True,
    })
    print(f"Verified {len(items)} official evidence/source files: {record}", flush=True)


if __name__ == "__main__":
    main()
