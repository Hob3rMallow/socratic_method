"""Competing-sheet diagnostics and ordered fitting for the v3 local chart.

Ordering is a conditional hypothesis inferred from a separately visible nearby
probability ridge. It is not a label identifying the anatomical wrap. Retain
both outcomes when the CT and model evidence do not distinguish them.
"""

from __future__ import annotations

import numpy as np

import sheet_patch_chart_v3 as chart
from sheet_surface_graph import SurfacePolicy, solve_surface
from sheet_surface_ordered_graph import solve_surfaces


def normalized_profiles(ct,coordinates,normal):
    sampled=[]
    for offset in (-6,-4,-2,0,2,4,6):
        points=coordinates.copy(); points[...,1:]+=offset*normal[None,:,None,:]
        sampled.append(chart.sample(ct,points))
    values=np.stack(sampled,axis=-1).astype(float)
    values-=values.mean(axis=-1,keepdims=True)
    return values/np.maximum(5.,np.linalg.norm(values,axis=-1,keepdims=True))


def ordered_fit(p,ct,path,z,*,side,policy=chart.PatchPolicy(),minimum_separation=3,
                surface_policy=SurfacePolicy(cost_scale=1000)):
    if side not in (-1,1): raise ValueError("neighbor side must be -1 or +1")
    fields=chart.chart_fields(p,ct,path,z,policy)
    nz,ns,nk=fields["costs"].shape
    seed=fields["seed_index"]
    seed_p=fields["probability"][seed:seed+1]
    lower,upper=fields["lower"][seed:seed+1].copy(),fields["upper"][seed:seed+1].copy()
    if side>0: lower[:]=np.maximum(lower,policy.normal_radius+minimum_separation)
    else: upper[:]=np.minimum(upper,policy.normal_radius-minimum_separation)
    if (lower>upper).any(): raise ValueError("no room for a separately ordered neighboring ridge")
    # Recover a coherent adjacent seed ridge before assuming an order relation.
    neighbor_seed=solve_surface(-np.log(np.maximum(.025,seed_p)),lower=lower,upper=upper,policy=surface_policy)["labels"][0]
    neighbor_p=seed_p[0,np.arange(ns),neighbor_seed]
    if (neighbor_p>=.35).mean()<.6 or neighbor_p.mean()<.35:
        raise ValueError("no independently strong neighboring seed ridge")
    profiles=normalized_profiles(ct,fields["coordinates"],fields["normal"])
    neighbor_template=profiles[seed,np.arange(ns),neighbor_seed]
    neighbor_correlation=np.sum(profiles*neighbor_template[None,:,None,:],axis=-1)
    labels_axis=np.arange(nk)[None,None,:]
    neighbor_unary=-np.log(np.maximum(.025,fields["probability"]))+policy.ct_profile_weight*(1-neighbor_correlation)
    neighbor_unary+=policy.displacement_weight*(labels_axis-neighbor_seed[None,:,None])**2
    target_index=0 if side>0 else 1
    unary=np.stack([fields["costs"],neighbor_unary] if side>0 else [neighbor_unary,fields["costs"]])
    anchors=np.full((2,nz,ns),-1,int)
    anchors[target_index,seed]=policy.normal_radius
    anchors[1-target_index,seed]=neighbor_seed
    solution=solve_surfaces(unary,minimum_separation=minimum_separation,fixed=anchors,
                            lower=np.stack([fields["lower"]]*2),upper=np.stack([fields["upper"]]*2),policy=surface_policy)
    labels=solution.pop("labels")
    zi,si=np.indices((nz,ns))
    target=labels[target_index]
    points=fields["coordinates"][zi,si,target]
    neighbor_points=fields["coordinates"][zi,si,labels[1-target_index]]
    return {"points":points,"neighbor_points":neighbor_points,"labels":target,
            "neighbor_labels":labels[1-target_index],"probability":fields["probability"][zi,si,target],
            "correlation":fields["correlation"][zi,si,target],"geometry":chart.mesh_metrics(points,fields["tangent"]),
            "seed_index":seed,"z_local":fields["z_local"],"optimization":solution,
            "neighbor_side":side,"neighbor_seed_p_mean":float(neighbor_p.mean()),
            "neighbor_seed_p035_fraction":float((neighbor_p>=.35).mean()),
            "minimum_normal_separation_px":int(np.abs(labels[0]-labels[1]).min()),
            "conditional_anatomical_hypothesis":True}


def alternative_diagnostics(fitted,policy=chart.PatchPolicy(),surface_policy=SurfacePolicy()):
    """Counterfactual surface costs away from the seed, not a calibrated confidence."""
    costs=fitted["costs"]
    nz,ns,nk=costs.shape
    fixed=np.full((nz,ns),-1,int); fixed[fitted["seed_index"]]=policy.normal_radius
    baseline=fitted["labels"]
    probes=[]
    for direction in (-1,1):
        row=int(np.clip(fitted["seed_index"]+direction*8,0,nz-1))
        if row==fitted["seed_index"]: continue
        center=ns//2
        existing=int(baseline[row,center])
        for sign in (-1,1):
            forced=existing+4*sign
            if not fitted["lower"][row,center]<=forced<=fitted["upper"][row,center]: continue
            trial=fixed.copy(); trial[row,center]=forced
            try:
                solution=solve_surface(costs,fixed=trial,lower=fitted["lower"],upper=fitted["upper"],policy=surface_policy)
            except ValueError: continue
            labels=solution["labels"]
            changed=np.abs(labels-baseline)>=2
            energy_delta=solution["energy"]-fitted["optimization"]["energy"]
            probes.append({"z_local":int(fitted["z_local"][row]),"column":center,"forced_label":forced,
                           "normal_change_at_probe_px":forced-existing,"surface_columns_moved_at_least_2px":int(changed.sum()),
                           "energy_delta":float(energy_delta),"energy_delta_per_changed_column":float(energy_delta/max(1,changed.sum()))})
    unconstrained=solve_surface(costs,lower=fitted["lower"],upper=fitted["upper"],policy=surface_policy)
    seed_change=np.abs(unconstrained["labels"][fitted["seed_index"]]-policy.normal_radius)
    return {"forced_alternatives":probes,
            "seed_release_energy_gain":float(fitted["optimization"]["energy"]-unconstrained["energy"]),
            "seed_release_mean_displacement":float(seed_change.mean()),
            "seed_release_fraction_moved_at_least_3px":float((seed_change>=3).mean()),
            "interpretation":"Model/CT objective sensitivity, not anatomical confidence or a validated rejection threshold."}
