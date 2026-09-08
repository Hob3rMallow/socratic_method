"""Prove the packaged numerical loop matches the pre-selection development run."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import tifffile
import run_f0_repair_context as context
from render_f0_repair_review import assemble

ROOT=context.ROOT/"output/crossres_data/f0_sheet_patches_20260907"
sys.path.insert(0,str(context.ROOT.parent/"socratic_method/src"))
from socratic_method import sheet_patches as sheets


def main():
    reference=ROOT/"development_growth_v2"
    output=ROOT/"package_development_verified"
    if output.exists(): raise FileExistsError(output)
    request=context.read(reference/"request.json")
    for item in request["input_files"]+request["code"]: context.verify_file(item)
    prior=Path(request["source_continuity"])
    old=context.read(prior/"request.json")
    ids,lower=request["spec"]["target_cube_ids"],request["spec"]["origin_zyx"]
    base=assemble(prior/"cubes_PRED",ids,lower)!=0
    native=assemble(Path(old["baseline"])/"cubes_PRED",ids,lower)!=0
    p=assemble(Path(old["inference"])/"probability",ids,lower).astype(np.float32)
    ct=assemble(context.SOURCE/"cubes_RAW",ids,lower).astype(np.float32)
    lo,hi=np.percentile(ct[ct>0],[1,99.5]); ct=np.clip((ct-lo)*255/(hi-lo),0,255)
    rows=[r for r in context.read(prior/"corridors.json")["corridors"] if r["accepted"]]
    expected_labels=[r["added_component"] for r in rows]
    sheets.assign_components(rows,base,native)
    if expected_labels!=[r["added_component"] for r in rows]: raise ValueError("seed ownership changed")
    output.mkdir(); (output/"cubes_PRED").mkdir()
    matched=[]
    with (output/"ledger.jsonl").open("x",encoding="utf-8") as log:
        def emit(row,mesh):
            if mesh is not None:
                expected=np.load(reference/row["mesh_arrays"],allow_pickle=False)
                for key,stored in (("vertices","vertices_local_zyx"),("faces","faces"),("normals","normals_zyx")):
                    if not np.array_equal(mesh[key],expected[stored]): raise ValueError("packaged mesh changed")
                matched.append(row["patch_id"])
            log.write(json.dumps(row)+"\n"); log.flush()
        def progress(n,total,count,voxels): print(f"Package parity: {n}/{total}, {count} patches, {voxels} additions",flush=True)
        added,accepted,census,count=sheets.fit_arrays(p,ct,base,rows,emit,progress)
    after=base|added
    files=[]
    for key in ids:
        path=output/"cubes_PRED"/f"{key}.tif"
        tifffile.imwrite(path,after[sheets.continuity.window_for(key,lower)].astype(np.uint8)*255,photometric="minisblack")
        if context.sha256(path)!=context.sha256(reference/"cubes_PRED"/path.name): raise ValueError("packaged mask changed")
        files.append(context.inventory(path))
    expected=context.read(reference/"summary.json")
    if census!=expected["census"] or len(accepted)!=107 or int(added.sum())!=75661: raise ValueError("decision parity failed")
    package=context.ROOT.parent/"socratic_method/src/socratic_method"
    receipt={"status":"verified", "byte_identical_tiffs":len(files),"exact_meshes":len(matched),"identical_refusal_census":True,
             "accepted_patches":len(accepted),"added_voxels":int(added.sum()),"output_files":files,
             "code":[context.inventory(path) for path in sorted(package.glob("sheet_*.py"))],
             "source_selection":context.inventory(ROOT/"selected_policy.json")}
    context.atomic_json(output/"receipt.json",receipt)
    print("Verified 27 exact TIFFs, 107 exact meshes, and every refusal count.",flush=True)


if __name__=="__main__": main()
