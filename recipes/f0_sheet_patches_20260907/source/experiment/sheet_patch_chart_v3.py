"""Joint sheet fitting in a parallel-column, monotone local chart.

Fits a single surface around an existing curve. Supports a probability-only
ablation and CT-profile identity. All outputs are candidates requiring geometry,
contact and competing-sheet review; a global optimum is not anatomical truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from sheet_surface_graph import SurfacePolicy, solve_surface


@dataclass(frozen=True)
class PatchPolicy:
    z_radius: int = 16
    normal_radius: int = 8
    probability_weight: float = 1.
    ct_profile_weight: float = .6
    displacement_weight: float = .01
    use_tensor_motion: bool = True
    minimum_p_coverage: float = .90
    probability_floor: float = .16
    minimum_mean_probability: float = .23
    minimum_profile_correlation: float = .15


def sample(volume, coordinates):
    return ndimage.map_coordinates(volume, coordinates.reshape(-1,3).T, order=1,mode="nearest").reshape(coordinates.shape[:-1])


def curve_frame(path):
    curve=np.asarray(path,dtype=float)
    if curve.ndim!=2 or curve.shape[1]!=2 or len(curve)<5 or not np.isfinite(curve).all():
        raise ValueError("a finite seeded YX curve with at least five points is required")
    # Parallel columns cannot intersect as independently transported normals can.
    # Only resample a genuinely monotone curve; never sort a folded-back path.
    centered=curve-curve.mean(axis=0)
    _,vectors=np.linalg.eigh(centered.T@centered)
    axis=vectors[:,-1]
    if np.dot(axis,curve[-1]-curve[0])<0: axis=-axis
    normal=np.array([-axis[1],axis[0]])
    smooth=ndimage.gaussian_filter1d(curve,.8,axis=0,mode="nearest")
    smooth[[0,-1]]=curve[[0,-1]]
    u=smooth@axis
    # A two-ended corridor may turn one pixel backwards when it enters a thick
    # raw anchor. Remove at most three terminal samples, never reorder a fold.
    for _ in range(3):
        if u[1]<=u[0]+1e-5: smooth,u=smooth[1:],u[1:]
        if u[-1]<=u[-2]+1e-5: smooth,u=smooth[:-1],u[:-1]
    if (np.diff(u)<=1e-5).any():
        raise ValueError("seed is not a monotone local chart; split into overlapping charts")
    distances=np.linspace(u[0],u[-1],max(5,int(np.ceil(u[-1]-u[0]))+1))
    heights=np.interp(distances,u,smooth@normal)
    resampled=distances[:,None]*axis+heights[:,None]*normal
    return resampled,np.broadcast_to(axis,resampled.shape).copy(),np.broadcast_to(normal,resampled.shape).copy()


def tensor_velocity(ct,z,curve,normal):
    lo=np.maximum(0,np.r_[z-8,curve.min(axis=0)-10].astype(int))
    hi=np.minimum(ct.shape,np.r_[z+9,curve.max(axis=0)+11].astype(int))
    window=tuple(slice(a,b) for a,b in zip(lo,hi))
    image=ndimage.gaussian_filter(ct[window].astype(np.float32),.8)
    gradients=np.gradient(image)
    coords=np.column_stack((np.full(len(curve),z),curve))-lo
    tensor=np.zeros((len(curve),3,3),float)
    for a in range(3):
        for b in range(a,3):
            field=ndimage.gaussian_filter(gradients[a]*gradients[b],2.)
            tensor[:,a,b]=tensor[:,b,a]=sample(field,coords)
    values,vectors=np.linalg.eigh(tensor)
    axis=vectors[:,:,-1]
    denominator=np.sum(axis[:,1:]*normal,axis=1)
    reliable=(np.abs(denominator)>.3) & (values[:,-1] > 1.5*np.maximum(values[:,-2],1e-5))
    velocity=np.zeros(len(curve))
    velocity[reliable]=np.clip(-axis[reliable,0]/denominator[reliable],-1.,1.)
    velocity=ndimage.gaussian_filter1d(velocity,2.,mode="nearest")
    return normal*velocity[:,None], {"reliable_fraction":float(reliable.mean()),"maximum_normal_speed":float(np.abs(velocity).max())}


def chart_fields(p,ct,path,z,policy=PatchPolicy()):
    if p.ndim!=3 or ct.shape!=p.shape or p.dtype!=np.float32:
        raise ValueError("matching 3D float32 probability and CT arrays required")
    if not policy.z_radius>=2 or not policy.normal_radius>=2 or z-policy.z_radius<2 or z+policy.z_radius>=p.shape[0]-2:
        raise ValueError("insufficient real axial context")
    curve,tangent,normal=curve_frame(path)
    velocity,motion=tensor_velocity(ct,z,curve,normal) if policy.use_tensor_motion else (np.zeros_like(curve),{"reliable_fraction":0.,"maximum_normal_speed":0.})
    offsets=np.arange(-policy.normal_radius,policy.normal_radius+1,dtype=float)
    zs=np.arange(z-policy.z_radius,z+policy.z_radius+1)
    reference=curve[None,:,:]+(zs-z)[:,None,None]*velocity[None,:,:]
    xy=reference[:,:,None,:]+offsets[None,None,:,None]*normal[None,:,None,:]
    coords=np.concatenate((np.broadcast_to(zs[:,None,None,None],(*xy.shape[:-1],1)),xy),axis=-1)
    # Search labels outside genuine context are forbidden, not clamped into
    # border evidence. Each local normal column intersects the rectangular
    # volume in a contiguous interval of feasible labels.
    valid=np.ones(coords.shape[:-1],bool)
    for d in (-6,6):
        shifted=coords.copy(); shifted[...,1:]+=d*normal[None,:,None,:]
        valid &= ((shifted>=1)&(shifted<np.array(p.shape)-2)).all(axis=-1)
    valid_rows=valid.any(axis=-1).all(axis=-1)
    seed_index=policy.z_radius
    if not valid_rows[seed_index] or not valid[seed_index,:,policy.normal_radius].all():
        raise ValueError("seed CT profile lacks real context")
    zlo=zhi=seed_index
    while zlo>0 and valid_rows[zlo-1]: zlo-=1
    while zhi<len(zs)-1 and valid_rows[zhi+1]: zhi+=1
    if zhi-zlo<4: raise ValueError("insufficient valid axial chart extent")
    coords,valid,zs=coords[zlo:zhi+1],valid[zlo:zhi+1],zs[zlo:zhi+1]
    seed_index-=zlo
    lower=valid.argmax(axis=2)
    upper=valid.shape[2]-1-valid[:,:,::-1].argmax(axis=2)
    if ((np.arange(valid.shape[2])[None,None,:]>=lower[...,None]) & (np.arange(valid.shape[2])[None,None,:]<=upper[...,None]) != valid).any():
        raise ValueError("noncontiguous context label bounds")
    prob=sample(p,coords)
    # CT intensity profile across the sheet, not a brightness-at-center gate.
    profile=[]
    for d in (-6,-4,-2,0,2,4,6):
        shifted=coords.copy()
        shifted[...,1:]+=d*normal[None,:,None,:]
        profile.append(sample(ct,shifted))
    profiles=np.stack(profile,axis=-1).astype(float)
    profiles-=profiles.mean(axis=-1,keepdims=True)
    denom=np.linalg.norm(profiles,axis=-1,keepdims=True)
    profiles/=np.maximum(denom,5.)
    seed_profile=profiles[seed_index,:,policy.normal_radius,:]
    correlation=np.sum(profiles*seed_profile[None,:,None,:],axis=-1)
    unary=-policy.probability_weight*np.log(np.maximum(.025,prob)) + policy.ct_profile_weight*(1-correlation)
    unary+=policy.displacement_weight*offsets[None,None,:]**2
    # Values evaluated outside context cannot affect any feasible cut: bounds
    # forbid them. Replace those sample values for explicit diagnostic hygiene.
    unary[~valid]=0; prob[~valid]=0; correlation[~valid]=0
    return {"coordinates":coords,"probability":prob,"correlation":correlation,"costs":unary,
            "normal":normal,"tangent":tangent,"velocity":velocity,"motion":motion,
            "z_local":zs,"offsets":offsets,"curve":curve,"lower":lower,"upper":upper,"seed_index":seed_index}


def mesh_metrics(points,tangent):
    du=np.diff(points,axis=1)
    dz=np.diff(points,axis=0)
    a,b,c=points[:-1,:-1],points[1:,:-1],points[:-1,1:]
    d=points[1:,1:]
    area=.5*(np.linalg.norm(np.cross(b-a,c-a),axis=-1)+np.linalg.norm(np.cross(d-b,c-b),axis=-1)).sum()
    along=np.sum(du[...,1:]*tangent[None,:-1,:],axis=-1)
    return {"surface_area_px2":float(area),"minimum_chart_forward_step":float(along.min()),
            "maximum_along_edge_px":float(np.linalg.norm(du,axis=-1).max()),
            "maximum_axial_edge_px":float(np.linalg.norm(dz,axis=-1).max()),
            "chart_foldover_steps":int((along<=0).sum())}


def fit_patch(p,ct,path,z,policy=PatchPolicy(),surface_policy=SurfacePolicy()):
    fields=chart_fields(p,ct,path,z,policy)
    nz,ns,nk=fields["costs"].shape
    anchors=np.full((nz,ns),-1,int)
    anchors[fields["seed_index"]]=policy.normal_radius
    solution=solve_surface(fields["costs"],fixed=anchors,lower=fields["lower"],upper=fields["upper"],policy=surface_policy)
    labels=solution.pop("labels")
    zi,si=np.indices(labels.shape)
    points=fields["coordinates"][zi,si,labels]
    probability=fields["probability"][zi,si,labels]
    correlation=fields["correlation"][zi,si,labels]
    # Interval support is assessed on the full tracked curve here. A later
    # acceptance layer must also assess the actually missing patch interior.
    rows=[]
    for i in range(nz):
        row={"z_local":int(fields["z_local"][i]),"p_mean":float(probability[i].mean()),
             "p10":float(np.quantile(probability[i],.1)),"supported_fraction":float((probability[i]>=policy.probability_floor).mean()),
             "ct_profile_correlation_mean":float(correlation[i].mean()),
             "mean_displacement":float((labels[i]-policy.normal_radius).mean()),
             "maximum_abs_displacement":int(np.abs(labels[i]-policy.normal_radius).max()),
             "touches_search_limit":bool(((labels[i]==0)|(labels[i]==nk-1)).any())}
        row["supported"]=(row["supported_fraction"]>=policy.minimum_p_coverage
                          and row["p_mean"]>=policy.minimum_mean_probability
                          and row["ct_profile_correlation_mean"]>=policy.minimum_profile_correlation
                          and not row["touches_search_limit"])
        rows.append(row)
    lo=hi=fields["seed_index"]
    while lo>0 and rows[lo-1]["supported"]: lo-=1
    while hi<nz-1 and rows[hi+1]["supported"]: hi+=1
    accepted_points=points[lo:hi+1]
    metrics=mesh_metrics(points,fields["tangent"])
    if metrics["chart_foldover_steps"]:
        raise ValueError("fitted surface violated parallel-chart monotonicity")
    return {"points":points,"labels":labels,"probability":probability,"correlation":correlation,
            "costs":fields["costs"],"coordinates":fields["coordinates"],"rows":rows,
            "supported_interval_local":[int(fields["z_local"][lo]),int(fields["z_local"][hi])+1],
            "supported_z_span":hi-lo+1,"supported_points":accepted_points,
            "seed_index":fields["seed_index"],"z_local":fields["z_local"],
            "lower":fields["lower"],"upper":fields["upper"],
            "motion":fields["motion"],"optimization":solution,"geometry":metrics,
            "patch_policy":asdict(policy),"surface_policy":asdict(surface_policy)}


def voxelize_diagnostic(points,shape,probability,floor=.16):
    """One-voxel surface samples + face brush, clipped to real P; not an acceptance rule."""
    canvas=np.zeros(shape,bool)
    # Supersample the parametric chart so oblique quads do not become dotted
    # Z rails. The mesh remains the authoritative surface geometry.
    refined=ndimage.zoom(points,(2,2,1),order=1)
    ijk=np.rint(refined.reshape(-1,3)).astype(int)
    if (ijk.min(axis=0)<0).any() or (ijk.max(axis=0)>=shape).any():
        raise ValueError("surface outside voxel canvas")
    canvas[tuple(ijk.T)]=True
    canvas=ndimage.binary_dilation(canvas,ndimage.generate_binary_structure(3,1))
    return canvas & (probability>=floor)
