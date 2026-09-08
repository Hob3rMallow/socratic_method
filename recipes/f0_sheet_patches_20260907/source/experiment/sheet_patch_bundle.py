"""Fit a seeded patch together with any strongly supported neighboring sheets.

Target search remains within +/-8 pixels. A wider +/-24-pixel chart is used
only to locate neighbors. Ordering is conditional on those observed ridges.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import sheet_patch_chart_v3 as chart
from sheet_patch_competition import normalized_profiles
from sheet_surface_graph import SurfacePolicy, solve_surface
from sheet_surface_ordered_graph_v2 import solve_surfaces


def fit_bundle(p,ct,path,z,policy=chart.PatchPolicy(),minimum_separation=5,
               surface_policy=SurfacePolicy(cost_scale=1000)):
    wide=replace(policy,normal_radius=24)
    fields=chart.chart_fields(p,ct,path,z,wide)
    nz,ns,nk=fields["costs"].shape
    seed=fields["seed_index"]
    center=wide.normal_radius
    seed_p=fields["probability"][seed:seed+1]
    neighbors={}
    diagnostics=[]
    for side in (-1,1):
        lower,upper=fields["lower"][seed:seed+1].copy(),fields["upper"][seed:seed+1].copy()
        if side>0: lower=np.maximum(lower,center+minimum_separation)
        else: upper=np.minimum(upper,center-minimum_separation)
        if (lower>upper).any():
            diagnostics.append({"side":side,"status":"no_context_for_neighbor"}); continue
        costs=-np.log(np.maximum(.025,seed_p))+.002*(np.arange(nk)[None,None,:]-center)**2
        try: labels=solve_surface(costs,lower=lower,upper=upper,policy=surface_policy)["labels"][0]
        except ValueError as error:
            diagnostics.append({"side":side,"status":"no_coherent_neighbor","detail":str(error)}); continue
        values=seed_p[0,np.arange(ns),labels]
        strong=float((values>=.35).mean())
        if strong<.6 or values.mean()<.35:
            diagnostics.append({"side":side,"status":"insufficient_neighbor_evidence","p035_fraction":strong}); continue
        neighbors[side]=labels
        diagnostics.append({"side":side,"status":"modeled","p035_fraction":strong,"p_mean":float(values.mean())})
    profiles=normalized_profiles(ct,fields["coordinates"],fields["normal"]) if neighbors else None
    entries=[]
    for side in (-1,0,1):
        if side==0:
            entries.append({"side":0,"costs":fields["costs"],"seed_labels":np.full(ns,center,int)})
        elif side in neighbors:
            labels=neighbors[side]
            template=profiles[seed,np.arange(ns),labels]
            correlation=(profiles*template[None,:,None,:]).sum(axis=-1)
            costs=-np.log(np.maximum(.025,fields["probability"]))+policy.ct_profile_weight*(1-correlation)
            costs+=policy.displacement_weight*(np.arange(nk)[None,None,:]-labels[None,:,None])**2
            entries.append({"side":side,"costs":costs,"seed_labels":labels})
    target_index=next(i for i,r in enumerate(entries) if r["side"]==0)
    layers=len(entries)
    fixed=np.full((layers,nz,ns),-1,int)
    lower=np.stack([fields["lower"]]*layers)
    upper=np.stack([fields["upper"]]*layers)
    lower[target_index]=np.maximum(lower[target_index],center-policy.normal_radius)
    upper[target_index]=np.minimum(upper[target_index],center+policy.normal_radius)
    for i,entry in enumerate(entries): fixed[i,seed]=entry["seed_labels"]
    if layers==1:
        solution=solve_surface(entries[0]["costs"],fixed=fixed[0],lower=lower[0],upper=upper[0],policy=surface_policy)
        labels=solution.pop("labels")[None]
    else:
        solution=solve_surfaces(np.stack([r["costs"] for r in entries]),fixed=fixed,lower=lower,upper=upper,
                                minimum_separation=minimum_separation,policy=surface_policy)
        labels=solution.pop("labels")
    zi,si=np.indices((nz,ns)); chosen=labels[target_index]
    points=fields["coordinates"][zi,si,chosen]
    geometry=chart.mesh_metrics(points,fields["tangent"])
    if geometry["chart_foldover_steps"]: raise ValueError("chart foldover")
    return {"points":points,"probability":fields["probability"][zi,si,chosen],
            "correlation":fields["correlation"][zi,si,chosen],"labels":chosen-center,
            "seed_index":seed,"z_local":fields["z_local"],"geometry":geometry,
            "modeled_surfaces":layers,"neighbor_diagnostics":diagnostics,
            "minimum_normal_separation_px":int(np.diff(labels,axis=0).min()) if layers>1 else None,
            "optimization":solution,"motion":fields["motion"],
            "neighbor_points":[fields["coordinates"][zi,si,labels[i]] for i in range(layers) if i!=target_index],
            "neighbor_probability":[fields["probability"][zi,si,labels[i]] for i in range(layers) if i!=target_index]}
