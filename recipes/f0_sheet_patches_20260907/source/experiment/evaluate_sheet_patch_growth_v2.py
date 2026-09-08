"""One serial CPU evaluation of coherent patches on a declared cached block.

Writes new arrays and explicit meshes; never overwrites previous masks. This is
a research comparison, not production promotion or a new inference/training run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter,defaultdict
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image,ImageDraw
from scipy import ndimage

import run_f0_repair_context as context
import sheet_patch_chart_v3 as chart
import sheet_patch_growth_v2 as growth
from render_f0_repair_review import assemble,overlay,panel_row,FONT

ROOT=context.ROOT/"output/crossres_data/f0_sheet_patches_20260907"
CONTINUITY=context.ROOT/"output/crossres_data/f0_continuity_20260907"
SOURCES=("sheet_surface_graph.py","sheet_surface_ordered_graph_v2.py","sheet_patch_chart_v3.py",
         "sheet_patch_competition.py","sheet_patch_bundle.py","sheet_patch_growth_v2.py","evaluate_sheet_patch_growth_v2.py")


def write_ply(path,mesh,lower):
    vertices=mesh["vertices"]+lower
    with path.open("x",encoding="ascii",newline="\n") as f:
        f.write(f"ply\nformat ascii 1.0\ncomment world voxel XYZ; pitch 8.640 um\nelement vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(mesh['faces'])}\nproperty list uchar int vertex_indices\nend_header\n")
        for z,y,x in vertices: f.write(f"{x:.5f} {y:.5f} {z:.5f}\n")
        for a,b,c in mesh["faces"]: f.write(f"3 {a} {b} {c}\n")


def render_patch(output,number,seed,metrics,center,ct,base,added,lower):
    pictures=[]
    for axis,name in enumerate(("Z","Y","X")):
        w=[slice(None)]*3; w[axis]=int(center[axis])
        for other in range(3):
            if axis!=other:
                start=max(0,min(264,int(center[other])-60)); w[other]=slice(start,start+120)
        w=tuple(w)
        pictures.append(panel_row([overlay(ct[w]),overlay(ct[w],base[w]),overlay(ct[w],base[w],added[w])],
                                 [f"CT {name}={center[axis]+lower[axis]}","Current v2 mask","Joint 3D patch additions"],scale=2))
    image=Image.new("RGB",(pictures[0].width,32+sum(v.height for v in pictures)),"#10151e")
    ImageDraw.Draw(image).text((8,5),f"Patch {number}, seed {seed}: {metrics['mesh_z_span']} planes; {metrics['added_voxels']} new voxels; {metrics['modeled_surfaces']} modeled surfaces",font=FONT,fill="white")
    y=32
    for picture in pictures: image.paste(picture,(0,y)); y+=picture.height
    name=f"assets/patch_{number:04d}.png"; image.save(output/name)
    return name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("block",choices=("development","transfer"))
    args=parser.parse_args()
    prior=CONTINUITY/("development_candidate_v2" if args.block=="development" else "fresh_control_candidate_v2")
    output=ROOT/f"{args.block}_growth_v2"
    if output.exists(): raise FileExistsError("preserve prior experiment output")
    request=context.read(prior/"request.json")
    old_summary=context.read(prior/"summary.json")
    for item in [*request["source_artifacts"],*old_summary["output_files"]]: context.verify_file(item)
    ids,lower=request["spec"]["target_cube_ids"],request["spec"]["origin_zyx"]
    rows=[r for r in context.read(prior/"corridors.json")["corridors"] if r["accepted"]]
    members=defaultdict(list)
    for row in rows: members[row["added_component"]].append(row)
    components=sorted(members,key=lambda n:(-max(r["length_px"] for r in members[n]),n))
    patch_policy=chart.PatchPolicy(); growth_policy=growth.GrowthPolicy()
    input_files=[context.inventory(prior/name) for name in ("request.json","summary.json","corridors.json","components.json")]
    input_files.extend(context.inventory(context.SOURCE/"cubes_RAW"/f"{key}.tif") for key in ids)
    code=[context.inventory(Path(__file__).parent/name) for name in SOURCES]
    output.mkdir(parents=True)
    for directory in ("meshes","assets","cubes_PRED"): (output/directory).mkdir()
    context.atomic_json(output/"request.json",{"block":args.block,"source_request":context.inventory(prior/"request.json"),"source_continuity":str(prior),
                         "spec":request["spec"],"patch_policy":asdict(patch_policy),"growth_policy":asdict(growth_policy),
                         "input_files":input_files,"code":code,"model_sha256":context.MODEL_SHA,"threshold":.35,"voxel_size_um":8.64,
                         "seed_selection":"Each prior added component, descending maximum path length; up to three longest seeds tried, first passing patch retained.",
                         "processes":1,"child_processes":0,"prior_control_is_known_transfer_evidence":True,"automatic_deployment":False})
    state={"status":"running","pid":os.getpid(),"processed_components":0,"accepted_patches":0}
    context.atomic_json(output/"state.json",state)
    base=assemble(prior/"cubes_PRED",ids,lower)!=0
    p=assemble(Path(request["inference"])/"probability",ids,lower).astype(np.float32)
    ct=assemble(context.SOURCE/"cubes_RAW",ids,lower).astype(np.float32)
    lo,hi=np.percentile(ct[ct>0],[1,99.5]); ct=np.clip((ct-lo)*255/(hi-lo),0,255)
    gray=ct.astype(np.uint8)
    owners=np.zeros(base.shape,np.uint16)
    fitted_meshes={}; ledger=[]; accepted=[]; census=Counter()

    @lru_cache(maxsize=96)
    def plane_labels(z): return ndimage.label(base[z],np.ones((3,3),bool))[0]

    started=time.perf_counter()
    for count,component in enumerate(components,1):
        candidates=sorted(members[component],key=lambda r:(-r["length_px"],-r["probability_mean"],r["id"]))[:3]
        for row in candidates:
            entry={"seed_id":row["id"],"prior_added_component":component,"seed_z_world":row["z_world"],"seed_length_px":row["length_px"]}
            try:
                if row["length_px"]<12: raise ValueError("seed too short for this patch profile")
                proposal=growth.propose_patch(p,ct,base,np.array(row["path_yx"]),row["z_local"],plane_labels,patch_policy,growth_policy)
                w,added=proposal["window"],proposal["added"]
                touched=owners[w][ndimage.binary_dilation(added,np.ones((3,3,3),bool))]
                other_ids=np.unique(touched[touched>0]).tolist()
                interactions=[]
                for other in other_ids:
                    compatible,metrics=growth.compatible_surfaces(proposal["mesh"],fitted_meshes[other],growth_policy)
                    interactions.append({"prior_patch":other,**metrics})
                    if not compatible: raise ValueError("incompatible interacting surface patches")
                unique_new=added & (owners[w]==0)
                if unique_new.sum()<growth_policy.minimum_new_voxels: raise ValueError("patch already supplied by an earlier compatible surface")
                number=len(accepted)+1
                owners[w][unique_new]=number
                fitted_meshes[number]=proposal["mesh"]
                entry.update(accepted=True,patch_id=number,reason="accepted",metrics=proposal["metrics"],
                             unique_added_voxels=int(unique_new.sum()),compatible_interactions=interactions,
                             mesh=f"meshes/patch_{number:04d}.ply",mesh_arrays=f"meshes/patch_{number:04d}.npz")
                write_ply(output/entry["mesh"],proposal["mesh"],np.array(lower))
                np.savez_compressed(output/entry["mesh_arrays"],vertices_local_zyx=proposal["mesh"]["vertices"],
                                    faces=proposal["mesh"]["faces"],normals_zyx=proposal["mesh"]["normals"])
                positions=np.argwhere(unique_new)+[s.start for s in w]
                entry["view_center_local"]=positions[len(positions)//2].tolist()
                accepted.append(entry); ledger.append(entry); census["accepted"]+=1
                break
            except ValueError as error:
                entry.update(accepted=False,reason=str(error)); ledger.append(entry); census[str(error)]+=1
        if count%16==0:
            state.update(processed_components=count,accepted_patches=len(accepted),added_voxels=int((owners>0).sum()),elapsed_seconds=time.perf_counter()-started)
            context.atomic_json(output/"state.json",state)
            print(f"{args.block}: {count}/{len(components)} prior components, {len(accepted)} patches, {state['added_voxels']} new voxels",flush=True)
    added=owners>0
    after=base|added
    if (base & ~after).any() or sum(r["unique_added_voxels"] for r in accepted)!=int(added.sum()): raise ValueError("independent output accounting failed")
    files=[]; cubes={}
    for key in ids:
        point=[int(t[1:]) for t in key.split("_")]
        w=tuple(slice(a-b,a-b+128) for a,b in zip(point,lower))
        path=output/"cubes_PRED"/f"{key}.tif"
        tifffile.imwrite(path,after[w].astype(np.uint8)*255,photometric="minisblack")
        actual=tifffile.imread(path)!=0
        if not np.array_equal(actual,after[w]) or (base[w]&~actual).any(): raise ValueError("written mask verification failed")
        files.append(context.inventory(path)); cubes[key]={"added_voxels":int(added[w].sum()),"erased_voxels":0}
    for row in accepted:
        row["image"]=render_patch(output,row["patch_id"],row["seed_id"],row["metrics"],row["view_center_local"],gray,base,added,lower)
    for z in (128,160,192,224,256):
        panel_row([overlay(gray[z]),overlay(gray[z],base[z]),overlay(gray[z],base[z],added[z])],
                  [f"CT z={z+lower[0]}","Current v2 mask","Joint 3D patch additions"]).save(output/"assets"/f"overview_z{z+lower[0]}.png")
    new_labels,n=ndimage.label(added,np.ones((3,3,3),bool))
    summary={"status":"research_candidate_complete","block":args.block,"accepted_patches":len(accepted),"attempted_seeds":len(ledger),
             "prior_added_components":len(components),"added_voxels_over_v2":int(added.sum()),"core_added_voxels_over_v2":int(added[128:256,128:256,128:256].sum()),
             "erased_v2_voxels":0,"summed_patch_area_px2_not_unique":sum(r["metrics"]["mesh_area_px2"] for r in accepted),
             "maximum_mesh_z_span":max((r["metrics"]["mesh_z_span"] for r in accepted),default=0),
             "patches_with_modeled_neighbors":sum(r["metrics"]["modeled_surfaces"]>1 for r in accepted),
             "all_meshes_manifold_with_boundary":all(r["metrics"]["manifold_with_boundary"] for r in accepted),
             "added_voxel_components":n,"census":dict(census),"cubes":cubes,"output_files":files,
             "anatomical_correctness_requires_review":True,"no_new_official_dice":True,"automatic_deployment":False,
             "ct_window":[float(lo),float(hi)],"summed_area_overcounts_compatible_overlap":True}
    for item in [*input_files,*code]: context.verify_file(item)
    context.atomic_json(output/"ledger.json",{"seeds":ledger})
    context.atomic_json(output/"patches.json",{"patches":accepted})
    context.atomic_json(output/"summary.json",summary)
    state.update(status="complete",processed_components=len(components),accepted_patches=len(accepted),elapsed_seconds=time.perf_counter()-started)
    context.atomic_json(output/"state.json",state)
    print(json.dumps({k:summary[k] for k in ("accepted_patches","added_voxels_over_v2","core_added_voxels_over_v2","maximum_mesh_z_span","patches_with_modeled_neighbors")}),flush=True)


if __name__=="__main__": main()
