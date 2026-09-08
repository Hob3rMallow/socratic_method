"""Bounded development-only joint-sheet probes on saved F0 evidence."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

import run_f0_repair_context as context
import sheet_patch_chart_v3 as patches
from render_f0_repair_review import assemble,overlay,panel_row,FONT

ROOT=context.ROOT/"output/crossres_data/f0_sheet_patches_20260907"
PRIOR=context.ROOT/"output/crossres_data/f0_continuity_20260907/development_candidate_v2"


def main():
    output=ROOT/"development_probe_v3"
    if output.exists(): raise FileExistsError("preserve prior probe")
    request=context.read(PRIOR/"request.json")
    for item in request["source_artifacts"]: context.verify_file(item)
    for item in context.read(PRIOR/"summary.json")["output_files"]: context.verify_file(item)
    ids,lower=request["spec"]["target_cube_ids"],request["spec"]["origin_zyx"]
    base=assemble(PRIOR/"cubes_PRED",ids,lower)!=0
    p=assemble(Path(request["inference"])/"probability",ids,lower).astype(np.float32)
    ct=assemble(context.SOURCE/"cubes_RAW",ids,lower).astype(np.float32)
    lo,hi=np.percentile(ct[ct>0],[1,99.5])
    ct=np.clip((ct-lo)*255/(hi-lo),0,255)
    rows=[r for r in context.read(PRIOR/"corridors.json")["corridors"] if r["accepted"]]
    chosen={1934:next(r for r in rows if r["id"]==1934)}
    for component in (292,144,301,290,333,342,79,93):
        members=[r for r in rows if r["added_component"]==component and 22<=r["z_local"]<362]
        if members:
            row=max(members,key=lambda r:r["length_px"])
            chosen[row["id"]]=row
    output.mkdir(parents=True)
    (output/"assets").mkdir()
    context.atomic_json(output/"request.json",{"source_request":context.inventory(PRIOR/"request.json"),
                         "base_summary":context.inventory(PRIOR/"summary.json"),"seed_ids":list(chosen),
                         "runner":context.inventory(Path(__file__)),"engine":context.inventory(Path(patches.__file__)),
                         "development_only":True,"creates_accepted_masks":False})
    results=[]
    for key,row in chosen.items():
        for mode in ("probability_only","ct_profile"):
            policy=patches.PatchPolicy(ct_profile_weight=0 if mode=="probability_only" else .6)
            try:
                fitted=patches.fit_patch(p,ct,np.array(row["path_yx"]),row["z_local"],policy)
            except ValueError as exc:
                results.append({"seed":key,"component":row["added_component"],"mode":mode,"error":str(exc)})
                continue
            proposed=patches.voxelize_diagnostic(fitted["supported_points"],base.shape,p)
            added=proposed & ~base
            point=np.rint(fitted["points"][fitted["seed_index"],fitted["points"].shape[1]//2]).astype(int)
            pictures=[]
            for axis,name in enumerate(("Z","Y","X")):
                w=[slice(None)]*3; w[axis]=int(point[axis])
                for other in range(3):
                    if axis!=other:
                        start=max(0,min(264,int(point[other])-60)); w[other]=slice(start,start+120)
                w=tuple(w)
                pictures.append(panel_row([overlay(ct[w].astype(np.uint8)),overlay(ct[w].astype(np.uint8),base[w]),overlay(ct[w].astype(np.uint8),base[w],added[w])],
                                         [f"CT {name}={point[axis]+lower[axis]}","Current v2 mask","Proposed 3D patch"],scale=2))
            image=Image.new("RGB",(pictures[0].width,32+sum(v.height for v in pictures)),"#10151e")
            ImageDraw.Draw(image).text((8,5),f"Seed {key}, component {row['added_component']}, {mode}: PROPOSAL, {fitted['supported_z_span']} planes, {added.sum()} new voxels",font=FONT,fill="white")
            y=32
            for pic in pictures: image.paste(pic,(0,y)); y+=pic.height
            name=f"seed_{key:05d}_{mode}"
            image.save(output/"assets"/f"{name}.png")
            np.savez_compressed(output/f"{name}.npz",points=fitted["points"],labels=fitted["labels"],probability=fitted["probability"],correlation=fitted["correlation"],costs=fitted["costs"],coordinates=fitted["coordinates"])
            result={k:v for k,v in fitted.items() if not isinstance(v,np.ndarray)}
            result.update(seed=key,component=row["added_component"],mode=mode,seed_length_px=row["length_px"],
                          proposed_new_voxels=int(added.sum()),image=f"assets/{name}.png")
            results.append(result)
            context.atomic_json(output/"progress.json",{"status":"running","pid":os.getpid(),"completed":len(results)})
            print(json.dumps({k:result[k] for k in ("seed","component","mode","supported_z_span","proposed_new_voxels","geometry")}),flush=True)
    context.atomic_json(output/"results.json",{"results":results,"ct_window":[float(lo),float(hi)],"status":"diagnostic_only"})
    context.atomic_json(output/"progress.json",{"status":"complete","completed":len(results)})


if __name__=="__main__": main()
