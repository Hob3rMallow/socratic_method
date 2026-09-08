"""Execute one declared probability-corridor policy on cached contiguous data.

No training or inference. Inputs are hashed; every accepted/refused proposal is
recorded and existing foreground is independently checked for preservation.
"""

from __future__ import annotations

import argparse
import json
import importlib
import os
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage

import probability_continuity as continuity
import run_f0_repair_context as context
from audit_f0_repair_contacts import plane_contacts
from render_f0_repair_review import FONT, assemble, overlay, panel_row

ROOT = context.ROOT / "output/crossres_data/f0_continuity_20260907"
PROPOSAL = continuity.ProposalPolicy(probability_floor=.20)
ACCEPTANCE = continuity.AcceptancePolicy()


def utc() -> str:
    return datetime.now(UTC).isoformat()


def main() -> None:
    global continuity, PROPOSAL, ACCEPTANCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("block", choices=("development", "fresh_control"))
    parser.add_argument("--version",type=int,choices=(1,2),default=1)
    args = parser.parse_args()
    if args.version == 2:
        continuity=importlib.import_module("probability_continuity_v2")
        PROPOSAL=continuity.ProposalPolicy(probability_floor=.20)
        ACCEPTANCE=continuity.AcceptancePolicy()
    output = ROOT / f"{args.block}_candidate_v{args.version}"
    if output.exists():
        raise FileExistsError("preserve prior experiment output; declare a new version for changes")
    if args.block == "development":
        old_request = context.read(context.OUTPUT / "request.json")
        spec = old_request["blocks"]["development"]
        inference = context.OUTPUT / "inference_development"
        baseline = context.OUTPUT / "repairs/development/extended_safe"
        sources = [old_request["model"], old_request["native_filler"],
                   *context.read(context.OUTPUT / "inference_state.json")["blocks"]["development"]["files"],
                   *context.read(context.OUTPUT / "repairs/development/extended_safe_summary.json")["artifact_files"]]
    else:
        selected = context.read(ROOT / "selected_policy.json")
        if selected["proposal"] != asdict(PROPOSAL) or selected["acceptance"] != asdict(ACCEPTANCE):
            raise ValueError("control must use the development-selected policy")
        fresh = context.read(ROOT / "fresh_control/request.json")
        spec = fresh["block"]
        inference = ROOT / "fresh_control/inference"
        baseline = ROOT / "fresh_control/native_baseline"
        sources = [*context.read(ROOT / "fresh_control/state.json")["input_artifacts"],
                   *context.read(baseline / "repair_receipt.json")["plan"]["input_files"]]
        for item in context.read(baseline / "repair_receipt.json")["output_files"]:
            sources.append({**item, "path": str(baseline / item["path"])})
    for item in sources:
        context.verify_file(item)
    ids, lower = spec["target_cube_ids"], spec["origin_zyx"]
    for key in ids:
        sources.append(context.inventory(context.SOURCE / "cubes_RAW" / f"{key}.tif"))
    request = {"schema": "f0-probability-continuity-run-v1", "created_utc": utc(), "block": args.block,
               "spec": spec, "inference": str(inference), "baseline": str(baseline),
               "model_sha256": context.MODEL_SHA, "threshold": .35, "voxel_size_um": 8.64,
               "proposal": asdict(PROPOSAL), "acceptance": asdict(ACCEPTANCE),
               "engine": context.inventory(Path(continuity.__file__)), "runner": context.inventory(Path(__file__)),
               "base_engine": context.inventory(Path(__file__).with_name("probability_continuity.py")),
               "source_artifacts": sources, "processed_z_local": [8,376], "threads": 1,
               "global_radial_endpoint_gate_applies_to_new_stage": False,
               "local_evidence_not_anatomical_certification": True,
               "control_training_or_deployment": False}
    output.mkdir(parents=True)
    context.atomic_json(output / "request.json", request)
    state = {"status": "running", "pid": os.getpid(), "started_utc": utc(), "completed_planes": 0}
    context.atomic_json(output / "state.json", state)
    base = assemble(baseline / "cubes_PRED", ids, lower) != 0
    p = assemble(inference / "probability", ids, lower)
    raw_ct = assemble(context.SOURCE / "cubes_RAW", ids, lower)
    lo, hi = np.percentile(raw_ct[raw_ct > 0], [1, 99.5])
    ct = np.clip((raw_ct.astype(np.float32) - lo) * 255 / (hi-lo), 0, 255).astype(np.uint8)
    del raw_ct
    repaired = base.copy()
    rows, census, contacts = [], Counter(), []
    started = time.perf_counter()

    @lru_cache(maxsize=20)
    def probability_neighborhood(z):
        return ndimage.maximum_filter(p[z].astype(np.float32), size=3)

    @lru_cache(maxsize=24)
    def raw_distance(z):
        return ndimage.distance_transform_edt(~base[z])

    for z in range(8, 376):
        proposals, proposed_counts = continuity.find_corridors(base[z], p[z],
                                    (3287-lower[1], 3784-lower[2]), PROPOSAL)
        census.update({"proposal_"+k: v for k, v in proposed_counts.items()})
        for row in proposals:
            path = np.array(row["path_yx"])
            row["neighbor_probability_coverage"] = [float((probability_neighborhood(z+dz)[tuple(path.T)] >= PROPOSAL.probability_floor).mean())
                                                       for dz in (-2,-1,1,2)]
        repaired[z], ledger, refused = continuity.select_and_paint(base[z], p[z], proposals,
                                                                    PROPOSAL.probability_floor, ACCEPTANCE)
        census.update(refused)
        for row in ledger:
            row["id"] = len(rows)
            row["z_world"] = z + lower[0]
            row["z_local"] = z
            path = np.array(row["path_yx"])
            row["core"] = bool(128 <= z < 256 and ((path.mean(axis=0) >= 128) & (path.mean(axis=0) < 256)).all())
            if row["accepted"]:
                row.update(continuity.evidence_features(path, p[z], ct[z], [raw_distance(z+dz) for dz in (-8,-4,-2,-1,1,2,4,8)]))
            rows.append(row)
        contact = plane_contacts(base[z], repaired[z])
        contacts.append({"z_world": z+lower[0], **contact})
        if (z-7) % 32 == 0:
            state.update(completed_planes=z-7, accepted=census["accepted"], elapsed_seconds=time.perf_counter()-started)
            context.atomic_json(output / "state.json", state)
            print(f"{args.block}: {z-7}/368 planes; {census['accepted']} accepted corridors; {int((repaired & ~base).sum())} added voxels", flush=True)
    added = repaired & ~base
    if (base & ~repaired).any():
        raise ValueError("source foreground erased")
    if sum(r.get("added_voxels",0) for r in rows if r["accepted"]) != int(added.sum()):
        raise ValueError("ledger and mask counts disagree")
    labels, n_components = ndimage.label(added, np.ones((3,3,3), bool))
    objects = ndimage.find_objects(labels)
    sizes = np.bincount(labels.ravel())
    components = []
    for number, bounds in enumerate(objects, 1):
        local = np.argwhere(labels[bounds] == number) + [s.start for s in bounds]
        middle = local[len(local)//2]
        radius = np.hypot(local[:,1]+lower[1]-3287, local[:,2]+lower[2]-3784)
        components.append({"id": number, "voxels": int(sizes[number]),
                           "z_span": bounds[0].stop-bounds[0].start,
                           "radial_extent_px": float(np.ptp(radius)),
                           "bounds_world_zyx": [[s.start+lower[a],s.stop+lower[a]] for a,s in enumerate(bounds)],
                           "view_center_local": middle.tolist(), "corridor_ids": []})
    for row in rows:
        if row["accepted"]:
            path = np.array(row["path_yx"])
            ids_at_path = labels[row["z_local"], path[:,0], path[:,1]]
            component_ids = np.unique(ids_at_path[ids_at_path>0])
            if len(component_ids) != 1:
                raise ValueError("accepted corridor lacks a single connected addition")
            component = int(component_ids[0])
            row["added_component"] = component
            components[component-1]["corridor_ids"].append(row["id"])
    (output / "cubes_PRED").mkdir()
    cube_counts, files = {}, []
    for key in ids:
        origin = tuple(int(t[1:]) for t in key.split("_"))
        window = tuple(slice(o-lo,o-lo+128) for o,lo in zip(origin,lower))
        path = output / "cubes_PRED" / f"{key}.tif"
        tifffile.imwrite(path, repaired[window].astype(np.uint8)*255, photometric="minisblack")
        cube_counts[key] = {"added": int(added[window].sum()), "base_foreground": int(base[window].sum()), "erased": 0}
        files.append(context.inventory(path))
    assets = output / "assets"
    assets.mkdir()
    accepted = [r for r in rows if r["accepted"]]
    for row in accepted:
        path = np.array(row["path_yx"])
        z = row["z_local"]
        center = path.mean(axis=0).astype(int)
        y0,x0 = np.maximum(0,np.minimum(264,center-60))
        w = (slice(y0,y0+120),slice(x0,x0+120))
        picture = panel_row([overlay(ct[z][w]),overlay(ct[z][w],base[z][w]),overlay(ct[z][w],base[z][w],added[z][w])],
                            ["CT", "Current native filler", "Probability corridors"],scale=2)
        row["image"] = f"assets/corridor_{row['id']:05d}.png"
        picture.save(output / row["image"])
    chosen = {}
    for key in ("voxels","z_span","radial_extent_px"):
        for row in sorted(components,key=lambda r:-r[key])[:8]:
            chosen[row["id"]] = row
    for row in chosen.values():
        point = row["view_center_local"]
        panels=[]
        for axis,name in enumerate(("Z","Y","X")):
            slices=[slice(None)]*3
            slices[axis]=point[axis]
            for other in range(3):
                if other!=axis:
                    start=max(0,min(264,point[other]-60))
                    slices[other]=slice(start,start+120)
            slices=tuple(slices)
            panels.append(panel_row([overlay(ct[slices]),overlay(ct[slices],base[slices]),overlay(ct[slices],base[slices],added[slices])],
                                    [f"CT {name}={point[axis]+lower[axis]}","Current native filler","Probability corridors"],scale=2))
        picture=Image.new("RGB",(panels[0].width,32+sum(p.height for p in panels)),"#10151e")
        ImageDraw.Draw(picture).text((8,5),f"Added component {row['id']}: {row['voxels']} voxels, Z span {row['z_span']}, radial extent {row['radial_extent_px']:.1f}",font=FONT,fill="white")
        y=32
        for panel in panels:
            picture.paste(panel,(0,y)); y+=panel.height
        row["image"]=f"assets/component_{row['id']:04d}.png"
        picture.save(output / row["image"])
    for z in (128,160,192,224,256):
        panel_row([overlay(ct[z]),overlay(ct[z],base[z]),overlay(ct[z],base[z],added[z])],
                  [f"CT z={z+lower[0]}","Current native filler","Probability corridors"]).save(assets / f"overview_z{z+lower[0]}.png")
    summary={"schema":"f0-probability-continuity-result-v1","block":args.block,"accepted_corridors":len(accepted),
             "accepted_longer_than_21px":sum(r["length_px"]>21 for r in accepted),
             "maximum_corridor_length_px":max((r["length_px"] for r in accepted),default=0),
             "total_recovered_centerline_px":sum(r["length_px"] for r in accepted),
             "added_voxels":int(added.sum()),"erased_base_voxels":0,"base_foreground":int(base.sum()),
             "core_added_voxels":int(added[128:256,128:256,128:256].sum()),
             "added_3d_components":n_components,"census":dict(census),"cube_counts":cube_counts,
             "detached_2d_strokes":sum(r["detached_strokes"] for r in contacts),
             "multiway_2d_strokes":sum(r["multiway_strokes"] for r in contacts),
             "source_request":context.inventory(output / "request.json"), "output_files":files,
             "ct_window":[float(lo),float(hi)],"anatomical_correctness_requires_review":True,
             "no_official_filled_mask_scores":True,"automatic_deployment":False}
    for name,value in (("summary.json",summary),("corridors.json",{"corridors":rows}),("components.json",{"components":components}),
                       ("contact_audit.json",{"planes":contacts})):
        context.atomic_json(output / name,value)
    for item in sources:
        context.verify_file(item)
    state.update(status="complete",completed_planes=368,completed_utc=utc(),elapsed_seconds=time.perf_counter()-started,
                 summary=context.inventory(output / "summary.json"))
    context.atomic_json(output / "state.json",state)
    print(json.dumps({k:summary[k] for k in ("accepted_corridors","accepted_longer_than_21px","added_voxels","core_added_voxels","added_3d_components","detached_2d_strokes","multiway_2d_strokes")}),flush=True)


if __name__ == "__main__":
    main()
