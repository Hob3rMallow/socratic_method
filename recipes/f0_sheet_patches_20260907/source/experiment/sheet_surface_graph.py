"""Exact integer-cost local height-field optimization with hard slope limits.

A label h at each (z,s) column represents one surface point, not foreground
throughout a volume. Threshold-node closure encodes arbitrary unary costs;
directed implication edges enforce slopes and anchors. Equal-level symmetric
edges encode L1 total variation. Scope: local height fields, not arbitrary folds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow


@dataclass(frozen=True)
class SurfacePolicy:
    maximum_z_step: int = 2
    maximum_s_step: int = 1
    z_tv: float = .20
    s_tv: float = .10
    cost_scale: int = 10000


def energy(labels, costs, policy=SurfacePolicy()):
    unary = np.take_along_axis(costs, labels[..., None], axis=-1)[..., 0].sum()
    return float(unary + policy.z_tv * np.abs(np.diff(labels, axis=0)).sum()
                 + policy.s_tv * np.abs(np.diff(labels, axis=1)).sum())


def solve_surface(costs, *, fixed=None, lower=None, upper=None, policy=SurfacePolicy()):
    """Return the global minimum of the quantized feasible height-field energy.

    Fixed entries are -1 (free) or a label. Lower/upper bounds may vary by column.
    Refuses infeasible constraints or integer ranges unsafe for SciPy's flow.
    Does not know which probability peak is the correct anatomical surface.
    """
    costs = np.asarray(costs, dtype=np.float64)
    if costs.ndim != 3 or min(costs.shape[:2]) < 1 or costs.shape[2] < 2 or not np.isfinite(costs).all():
        raise ValueError("finite (z,s,label) costs with at least two labels required")
    if not (isinstance(policy.cost_scale, int) and policy.cost_scale > 0
            and isinstance(policy.maximum_z_step, int) and policy.maximum_z_step >= 0
            and isinstance(policy.maximum_s_step, int) and policy.maximum_s_step >= 0
            and np.isfinite([policy.z_tv, policy.s_tv]).all()
            and min(policy.z_tv, policy.s_tv) >= 0):
        raise ValueError("invalid surface policy")
    nz, ns, nk = costs.shape
    lo = np.zeros((nz, ns), int) if lower is None else np.asarray(lower)
    hi = np.full((nz, ns), nk-1, int) if upper is None else np.asarray(upper)
    anchors = np.full((nz, ns), -1, int) if fixed is None else np.asarray(fixed)
    for value in (lo, hi, anchors):
        if value.shape != (nz, ns) or value.dtype.kind not in "iu":
            raise ValueError("integer bounds/anchors must match surface columns")
    if (lo < 0).any() or (hi >= nk).any() or (lo > hi).any() or (anchors < -1).any() or (anchors >= nk).any():
        raise ValueError("invalid label bounds")
    lo, hi = lo.copy(), hi.copy()
    pinned = anchors >= 0
    if ((anchors[pinned] < lo[pinned]) | (anchors[pinned] > hi[pinned])).any():
        raise ValueError("anchor outside bounds")
    lo[pinned] = hi[pinned] = anchors[pinned]
    quantized = np.rint((costs-costs.min(axis=2, keepdims=True))*policy.cost_scale)
    if quantized.max() > 10**8:
        raise ValueError("cost range too large")
    quantized = quantized.astype(np.int64)
    increments = np.diff(quantized, axis=2)
    ztv, stv = round(policy.z_tv*policy.cost_scale), round(policy.s_tv*policy.cost_scale)
    # Larger than all finite edges combined, but under the int32 capacity range.
    finite_bound = int(np.abs(increments).sum()) + (nz*max(0,ns-1)*stv + max(0,nz-1)*ns*ztv)*2*(nk-1)
    infinity = finite_bound+1
    if infinity >= 2**30:
        raise ValueError("graph capacity budget exceeded; use a bounded local chart")
    nodes = np.arange(nz*ns*(nk-1)).reshape(nz,ns,nk-1)
    source, sink = nodes.size, nodes.size+1
    rows, cols, vals = [], [], []

    def add(a, b, capacity):
        aa, bb, cc = np.broadcast_arrays(a, b, capacity)
        good = cc.ravel() > 0
        rows.append(aa.ravel()[good].astype(np.int32))
        cols.append(bb.ravel()[good].astype(np.int32))
        vals.append(cc.ravel()[good].astype(np.int64))

    add(source, nodes, np.maximum(0,-increments))
    add(nodes, sink, np.maximum(0,increments))
    # Height>=k implies height>=k-1. Label zero is implicit in every column.
    add(nodes[:,:,1:], nodes[:,:,:-1], infinity)
    thresholds = np.arange(1,nk)[None,None,:]
    add(source, nodes, np.where(thresholds <= lo[...,None], infinity, 0))
    add(nodes, sink, np.where(thresholds > hi[...,None], infinity, 0))
    for axis, maximum, tv in ((0,policy.maximum_z_step,ztv), (1,policy.maximum_s_step,stv)):
        a = np.take(nodes, np.arange(nodes.shape[axis]-1), axis=axis)
        b = np.take(nodes, np.arange(1,nodes.shape[axis]), axis=axis)
        add(a,b,tv)
        add(b,a,tv)
        if maximum < nk-1:
            target = nk-1-maximum
            add(a[...,maximum:],b[...,:target],infinity)
            add(b[...,maximum:],a[...,:target],infinity)
    graph = coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),
                       shape=(sink+1,sink+1), dtype=np.int64).tocsr()
    # Duplicate implication edges can sum capacities; defend the backend range.
    if graph.data.max(initial=0) >= 2**31-1:
        raise ValueError("summed edge capacities exceed backend range")
    flow = maximum_flow(graph, source, sink, method="dinic")
    residual = graph-flow.flow
    residual.data = (residual.data > 0).astype(np.int8)
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, source, directed=True, return_predecessors=False)
    selected = np.zeros(sink+1, bool)
    selected[reachable] = True
    labels = selected[nodes].sum(axis=2).astype(np.int32)
    if ((labels < lo) | (labels > hi)).any() or (np.abs(np.diff(labels,axis=0)) > policy.maximum_z_step).any() or (np.abs(np.diff(labels,axis=1)) > policy.maximum_s_step).any():
        raise ValueError("infeasible surface constraints")
    quant_policy = SurfacePolicy(policy.maximum_z_step,policy.maximum_s_step,ztv,stv,1)
    measured = energy(labels,quantized,quant_policy)
    objective_from_cut = int(flow.flow_value) + int(quantized[:,:,0].sum()) + int(np.minimum(increments,0).sum())
    if measured != objective_from_cut:
        raise RuntimeError("cut and direct quantized surface energy disagree")
    return {"labels": labels, "energy": energy(labels,costs,policy),
            "quantized_energy": int(measured), "flow": int(flow.flow_value),
            "nodes": nodes.size+2, "edges": graph.nnz,
            "maximum_z_step": int(np.abs(np.diff(labels,axis=0)).max(initial=0)),
            "maximum_s_step": int(np.abs(np.diff(labels,axis=1)).max(initial=0))}
