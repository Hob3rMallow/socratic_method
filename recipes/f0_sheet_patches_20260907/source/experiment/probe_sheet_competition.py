"""Development-only investigation of alternative and ordered sheet hypotheses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

import run_f0_repair_context as context
import sheet_patch_chart_v3 as chart
import sheet_patch_competition as competition
from render_f0_repair_review import assemble,overlay,panel_row,FONT

ROOT=context.ROOT/"output/crossres_data/f0_sheet_patches_20260907"
PRIOR=context.ROOT/"output/crossres_data/f0_continuity_20260907/development_candidate_v2"


def supported(result,policy):
    rows=(result["probability"]>=policy.probability_floor).mean(axis=1)>=policy.minimum_p_coverage
    rows &= result["probability"].mean(axis=1)>=policy.minimum_mean_probability
    rows &= result["correlation"].mean(axis=1)>=policy.minimum_profile_correlation
    lo=hi=result["seed_index"]
    while lo>0 and rows[lo-1]: lo-=1
    while hi<len(rows)-1 and rows[hi+1]: hi+=1
    return result["points"][lo:hi+1],hi-lo+1


def main():
    output=ROOT/"competition_probe_v1"
    if output.exists(): raise FileExistsError("preserve prior investigation")
    request=context.read(PRIOR/"request.json")
    for item in request["source_artifacts"]: context.verify_file(item)
    ids,lower=request["spec"]["target_cube_ids"],request["spec"]["origin_zyx"]
    base=assemble(PRIOR/"cubes_PRED",ids,lower)!=0
    p=assemble(Path(request["inference"])/"probability",ids,lower).astype(np.float32)
    ct=assemble(context.SOURCE/"cubes_RAW",ids,lower).astype(np.float32)
    lo,hi=np.percentile(ct[ct>0],[1,99.5]); ct=np.clip((ct-lo)*255/(hi-lo),0,255)
    rows=context.read(PRIOR/"corridors.json")["corridors"]
    chosen=[next(r for r in rows if r["id"]==key) for key in (1934,3637,3752,1039)]
    output.mkdir(parents=True); (output/"assets").mkdir()
    source=Path(__file__).parent
    context.atomic_json(output/"request.json",{"seeds":[r["id"] for r in chosen],"source_request":context.inventory(PRIOR/"request.json"),
                         "code":[context.inventory(source/name) for name in ("probe_sheet_competition.py","sheet_patch_chart_v3.py","sheet_patch_competition.py","sheet_surface_graph.py","sheet_surface_ordered_graph.py")],
                         "minimum_normal_separation_px":5,"development_only":True,"proposals_not_accepted_masks":True})
    results=[]
    for row in chosen:
        policy=chart.PatchPolicy()
        single=chart.fit_patch(p,ct,np.array(row["path_yx"]),row["z_local"],policy)
        alternatives=competition.alternative_diagnostics(single,policy)
        plain=chart.voxelize_diagnostic(single["supported_points"],base.shape,p) & ~base
        case={"seed":row["id"],"component":row["added_component"],"single_z_span":single["supported_z_span"],
              "single_new_voxels":int(plain.sum()),"alternatives":alternatives,"ordered":[]}
        for side in (-1,1):
            try: fitted=competition.ordered_fit(p,ct,np.array(row["path_yx"]),row["z_local"],side=side,policy=policy,minimum_separation=5)
            except ValueError as error:
                case["ordered"].append({"side":side,"error":str(error)}); continue
            points,span=supported(fitted,policy)
            added=chart.voxelize_diagnostic(points,base.shape,p)&~base
            center=np.rint(fitted["points"][fitted["seed_index"],fitted["points"].shape[1]//2]).astype(int)
            pictures=[]
            for axis,name in enumerate(("Z","Y","X")):
                w=[slice(None)]*3; w[axis]=int(center[axis])
                for other in range(3):
                    if axis!=other:
                        start=max(0,min(264,int(center[other])-60)); w[other]=slice(start,start+120)
                w=tuple(w)
                pictures.append(panel_row([overlay(ct[w].astype(np.uint8)),overlay(ct[w].astype(np.uint8),base[w]),
                                          overlay(ct[w].astype(np.uint8),base[w],plain[w]),overlay(ct[w].astype(np.uint8),base[w],added[w])],
                                         [f"CT {name}={center[axis]+lower[axis]}","Current v2 mask","Single-surface proposal","Ordered-pair proposal"],scale=2))
            image=Image.new("RGB",(pictures[0].width,32+sum(v.height for v in pictures)),"#10151e")
            ImageDraw.Draw(image).text((8,5),f"Seed {row['id']}, component {row['added_component']}, neighbor side {side}: conditional order hypothesis",font=FONT,fill="white")
            y=32
            for picture in pictures: image.paste(picture,(0,y)); y+=picture.height
            name=f"seed_{row['id']:05d}_side_{side:+d}"
            image.save(output/"assets"/f"{name}.png")
            np.savez_compressed(output/f"{name}.npz",single_points=single["points"],ordered_points=fitted["points"],neighbor_points=fitted["neighbor_points"],
                                ordered_probability=fitted["probability"],ordered_correlation=fitted["correlation"])
            case["ordered"].append({"side":side,"span":span,"proposed_new_voxels":int(added.sum()),"single_proposal_voxels_not_retained":int((plain & ~added).sum()),
                                    "points_moved_at_least_2px":int((np.linalg.norm(single["points"]-fitted["points"],axis=2)>=2).sum()),
                                    "mean_position_change_px":float(np.linalg.norm(single["points"]-fitted["points"],axis=2).mean()),
                                    "neighbor_seed_p_mean":fitted["neighbor_seed_p_mean"],"geometry":fitted["geometry"],
                                    "minimum_normal_separation_px":fitted["minimum_normal_separation_px"],"image":f"assets/{name}.png"})
        results.append(case)
        context.atomic_json(output/"results.json",{"results":results,"status":"running"})
        print(json.dumps(case),flush=True)
    context.atomic_json(output/"results.json",{"results":results,"status":"diagnostic_complete","no_acceptance_or_anatomical_claim":True})


if __name__=="__main__": main()
