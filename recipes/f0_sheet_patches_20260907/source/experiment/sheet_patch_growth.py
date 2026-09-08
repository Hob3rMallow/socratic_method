"""Auditable 3D patch construction and compatibility checks for seeded surfaces.

The model and previous binary outputs are not changed. New patch vertices,
triangles and supported paint are returned separately for transaction/review.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import sheet_patch_chart_v3 as chart
from sheet_patch_bundle import fit_bundle


@dataclass(frozen=True)
class GrowthPolicy:
    minimum_z_span: int = 5
    minimum_surface_area_px2: float = 40
    minimum_new_voxels: int = 24
    minimum_normal_separation: int = 5
    maximum_component_contacts_per_plane: int = 2
    maximum_compatible_normal_offset: float = .8
    minimum_compatible_normal_dot: float = .85
    minimum_compatible_fraction: float = .9


def mesh_from_grid(points,valid,seed_index):
    """Extract an edge-connected, non-pinched supported triangle patch."""
    nz,ns=valid.shape
    quads=valid[:-1,:-1]&valid[1:,:-1]&valid[:-1,1:]&valid[1:,1:]
    z,s=np.nonzero(quads)
    if len(z)==0: raise ValueError("no supported surface faces")
    a=z*ns+s; b=(z+1)*ns+s; c=z*ns+s+1; d=(z+1)*ns+s+1
    faces=np.concatenate([np.column_stack([a,b,c]),np.column_stack([b,d,c])])
    vertices=points.reshape(-1,3)
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    unique,inverse,counts=np.unique(edges,axis=0,return_inverse=True,return_counts=True)
    if counts.max()>2: raise ValueError("nonmanifold surface edges")
    face_ids=np.tile(np.arange(len(faces)),3)
    order=np.argsort(inverse,kind="stable")
    offsets=np.r_[0,np.cumsum(counts)]
    twins=np.flatnonzero(counts==2)
    first=face_ids[order[offsets[twins]]]; second=face_ids[order[offsets[twins]+1]]
    graph=coo_matrix((np.ones(len(first)*2),(np.r_[first,second],np.r_[second,first])),shape=(len(faces),len(faces))).tocsr()
    number,labels=connected_components(graph,directed=False)
    touches_seed=(faces//ns==seed_index).any(axis=1)
    choices=np.unique(labels[touches_seed])
    if not len(choices): raise ValueError("surface faces disconnected from seed row")
    sizes=np.bincount(labels)
    chosen=max(choices,key=lambda k:sizes[k])
    faces=faces[labels==chosen]
    used=np.unique(faces); remap=np.full(len(vertices),-1,int); remap[used]=np.arange(len(used))
    faces=remap[faces]; vertices=vertices[used]
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    boundary=unique[counts==1]
    boundary_degree=np.bincount(boundary.ravel(),minlength=len(vertices))
    if ((boundary_degree!=0)&(boundary_degree!=2)).any():
        raise ValueError("pinched/nonmanifold surface boundary")
    crosses=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
    face_area=np.linalg.norm(crosses,axis=1)/2
    if (face_area<1e-6).any(): raise ValueError("degenerate surface triangles")
    normals=np.zeros_like(vertices)
    for column in range(3): np.add.at(normals,faces[:,column],crosses)
    norms=np.linalg.norm(normals,axis=1)
    if (norms<1e-8).any(): raise ValueError("undefined surface normal")
    normals/=norms[:,None]
    return {"vertices":vertices,"faces":faces.astype(np.int32),"normals":normals,
            "area_px2":float(face_area.sum()),"supported_z_span":int(np.ptp(vertices[:,0]))+1,
            "boundary_edges":int(len(boundary)),"surface_components_before_selection":int(number),
            "discarded_face_components":int(number-1),"manifold_with_boundary":True}


def voxelize_mesh(mesh,probability,floor=.16):
    vertices,faces=mesh["vertices"],mesh["faces"]
    lower=np.maximum(0,np.floor(vertices.min(axis=0)-2).astype(int))
    upper=np.minimum(probability.shape,np.ceil(vertices.max(axis=0)+3).astype(int))
    window=tuple(slice(a,b) for a,b in zip(lower,upper))
    a=vertices[faces[:,0]]; ab=vertices[faces[:,1]]-a; ac=vertices[faces[:,2]]-a
    canvas=np.zeros(tuple(upper-lower),bool)
    for i in range(5):
        for j in range(5-i):
            points=np.rint(a+(i/4)*ab+(j/4)*ac).astype(int)-lower
            canvas[tuple(points.T)]=True
    canvas=ndimage.binary_dilation(canvas,ndimage.generate_binary_structure(3,1))
    # Do not extrapolate one more slice beyond the actual supported surface.
    zs=np.arange(lower[0],upper[0])
    canvas[(zs<vertices[:,0].min())|(zs>vertices[:,0].max())]=False
    canvas &= probability[window]>=floor
    labels,count=ndimage.label(canvas,np.ones((3,3,3),bool))
    if count!=1: raise ValueError("voxelized surface is disconnected")
    return window,canvas,{"paint_components":int(count),"paint_voxels":int(canvas.sum())}


def compatible_surfaces(new,old,policy=GrowthPolicy()):
    distance,indices=cKDTree(old["vertices"]).query(new["vertices"],distance_upper_bound=3)
    near=np.isfinite(distance)
    if not near.any(): return False,{"reason":"paint_contact_without_surface_agreement"}
    delta=new["vertices"][near]-old["vertices"][indices[near]]
    a,b=new["normals"][near],old["normals"][indices[near]]
    normal_distance=np.maximum(np.abs(np.sum(delta*a,axis=1)),np.abs(np.sum(delta*b,axis=1)))
    agreement=np.abs(np.sum(a*b,axis=1))
    fraction=float(((normal_distance<=policy.maximum_compatible_normal_offset)&(agreement>=policy.minimum_compatible_normal_dot)).mean())
    return fraction>=policy.minimum_compatible_fraction,{"nearby_vertices":int(near.sum()),"compatible_fraction":fraction,
                                                       "normal_offset_p90":float(np.quantile(normal_distance,.9)),"normal_dot_p10":float(np.quantile(agreement,.1))}


def propose_patch(p,ct,base,path,z,plane_labels,patch_policy=chart.PatchPolicy(),growth_policy=GrowthPolicy()):
    fitted=fit_bundle(p,ct,path,z,patch_policy,minimum_separation=growth_policy.minimum_normal_separation)
    points=fitted["points"]
    rows=[]
    for index,world_z in enumerate(fitted["z_local"]):
        prob=fitted["probability"][index]
        corr=fitted["correlation"][index]
        good=(float((prob>=patch_policy.probability_floor).mean())>=patch_policy.minimum_p_coverage
              and float(prob.mean())>=patch_policy.minimum_mean_probability
              and float(corr.mean())>=patch_policy.minimum_profile_correlation
              and not (np.abs(fitted["labels"][index])>=patch_policy.normal_radius).any())
        # Missing interior, not just the already-strong attachment rails.
        coords=np.rint(points[index]).astype(int)
        missing=~base[tuple(coords.T)]
        missing_mean=float(prob[missing].mean()) if missing.any() else None
        if missing.any() and (missing_mean<.20 or (prob[missing]>=.16).mean()<.85): good=False
        rows.append({"z_local":int(world_z),"supported":bool(good),"p_mean":float(prob.mean()),
                     "p016_fraction":float((prob>=.16).mean()),"ct_correlation":float(corr.mean()),
                     "missing_p_mean":missing_mean})
    seed=fitted["seed_index"]
    if not rows[seed]["supported"]: raise ValueError("seed itself lacks supported surface evidence")
    lo=hi=seed
    while lo>0 and rows[lo-1]["supported"]: lo-=1
    while hi<len(rows)-1 and rows[hi+1]["supported"]: hi+=1
    if hi-lo+1<growth_policy.minimum_z_span: raise ValueError("insufficient coherent axial span")
    points=points[lo:hi+1]
    valid=fitted["probability"][lo:hi+1]>=patch_policy.probability_floor
    mesh=mesh_from_grid(points,valid,seed-lo)
    if mesh["area_px2"]<growth_policy.minimum_surface_area_px2 or mesh["supported_z_span"]<growth_policy.minimum_z_span:
        raise ValueError("insufficient supported sheet area")
    window,paint,voxel_metrics=voxelize_mesh(mesh,p,patch_policy.probability_floor)
    added=paint & ~base[window]
    if added.sum()<growth_policy.minimum_new_voxels: raise ValueError("surface already covered or negligible extension")
    contacts=[]
    for local_z in range(paint.shape[0]):
        if not added[local_z].any(): continue
        labels=plane_labels(local_z+window[0].start)[window[1:]]
        touched=labels[ndimage.binary_dilation(added[local_z],np.ones((3,3),bool))]
        owners=np.unique(touched[touched>0]).tolist()
        if len(owners)>growth_policy.maximum_component_contacts_per_plane:
            raise ValueError("patch touches more than two existing components on a plane")
        contacts.append({"z_local":local_z+window[0].start,"existing_component_contacts":owners})
    # Patch must have an actual source attachment, not merely a copied seed ID.
    if not (paint & base[window]).any(): raise ValueError("patch detached from existing geometry")
    metrics={"added_voxels":int(added.sum()),"mesh_area_px2":mesh["area_px2"],"mesh_z_span":mesh["supported_z_span"],
             "mesh_vertices":len(mesh["vertices"]),"mesh_triangles":len(mesh["faces"]),
             "manifold_with_boundary":True,"modeled_surfaces":fitted["modeled_surfaces"],
             "minimum_normal_separation_px":fitted["minimum_normal_separation_px"],"neighbor_diagnostics":fitted["neighbor_diagnostics"],
             "geometry":fitted["geometry"],"rows":rows,"contacts":contacts,**voxel_metrics}
    return {"mesh":mesh,"window":window,"paint":paint,"added":added,"metrics":metrics}
