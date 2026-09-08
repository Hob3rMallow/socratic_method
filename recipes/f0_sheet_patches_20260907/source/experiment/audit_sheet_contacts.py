"""Independent per-patch contact audit and descriptive global connectivity."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy import ndimage
import run_f0_repair_context as context
from render_f0_repair_review import assemble
from audit_sheet_patch_outputs import raster

ROOT=context.ROOT/"output/crossres_data/f0_sheet_patches_20260907"


def main():
    results={}
    for name in ("development_growth_v2","fresh_assessment_growth"):
        folder=ROOT/name
        request=context.read(folder/"request.json"); prior=Path(request["source_continuity"])
        source=Path(request.get("source") or context.read(prior/"request.json")["inference"])
        ids,lower=request["spec"]["target_cube_ids"],request["spec"]["origin_zyx"]
        base=assemble(prior/"cubes_PRED",ids,lower)!=0
        after=assemble(folder/"cubes_PRED",ids,lower)!=0
        p=assemble(source/"probability",ids,lower).astype(np.float32)
        @lru_cache(maxsize=96)
        def labels(z): return ndimage.label(base[z],np.ones((3,3),bool))[0]
        contact_rows=0; max_contacts=0
        for row in context.read(folder/"patches.json")["patches"]:
            data=np.load(folder/row["mesh_arrays"],allow_pickle=False)
            w,paint=raster({"vertices":data["vertices_local_zyx"],"faces":data["faces"]},p)
            added=paint & ~base[w]; actual=[]
            for z in range(added.shape[0]):
                if not added[z].any(): continue
                values=labels(z+w[0].start)[w[1:]][ndimage.binary_dilation(added[z],np.ones((3,3),bool))]
                contacts=np.unique(values[values>0]).tolist()
                max_contacts=max(max_contacts,len(contacts)); contact_rows+=1
                actual.append({"z_local":z+w[0].start,"existing_component_contacts":contacts})
            if actual!=row["metrics"]["contacts"] or any(len(r["existing_component_contacts"])>2 for r in actual):
                raise ValueError("saved contact ledger differs from independent labels")
        labels.cache_clear()
        before_count=ndimage.label(base,np.ones((3,3,3),bool))[1]
        after_count=ndimage.label(after,np.ones((3,3,3),bool))[1]
        results[name]={"status":"passed", "independently_checked_patch_planes":contact_rows,
                       "maximum_existing_components_contacted_per_patch_plane":max_contacts,
                       "baseline_26_connected_components":before_count,"after_26_connected_components":after_count,
                       "net_component_reduction":before_count-after_count,
                       "component_reduction_is_not_true_sheet_identity_or_anatomical_safety":True}
    context.atomic_json(ROOT/"independent_contact_audit.json",results)
    print(results,flush=True)


if __name__=="__main__": main()
