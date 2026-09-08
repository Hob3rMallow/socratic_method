"""Refined probability geodesics with outward-anchor and local-turn checks.

The prior skeleton supplies proposals; continuous probability chooses the
route within a narrow corridor. No free cross-volume threshold growth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.graph import route_through_array

import probability_continuity as base

ProposalPolicy = base.ProposalPolicy
EIGHT = base.EIGHT
evidence_features = base.evidence_features
probability_support = base.probability_support


@dataclass(frozen=True)
class AcceptancePolicy(base.AcceptancePolicy):
    minimum_outward_alignment: float = .25
    maximum_local_turn_degrees: float = 60


def local_turn_degrees(path: np.ndarray, span: int = 4) -> float:
    if len(path) < 2 * span + 1:
        return 0.
    incoming = path[span:-span] - path[:-2*span]
    outgoing = path[2*span:] - path[span:-span]
    denom = np.linalg.norm(incoming,axis=1)*np.linalg.norm(outgoing,axis=1)
    cos = np.sum(incoming*outgoing,axis=1) / np.maximum(1e-8,denom)
    return float(np.degrees(np.arccos(np.clip(cos,-1,1))).max())


def refine_route(path: np.ndarray, mask: np.ndarray, probability: np.ndarray, floor: float) -> np.ndarray | None:
    lo = np.maximum(0,path.min(axis=0)-4)
    hi = np.minimum(mask.shape,path.max(axis=0)+5)
    window = tuple(slice(a,b) for a,b in zip(lo,hi))
    local_path = path-lo
    center = np.zeros(tuple(hi-lo), bool)
    center[tuple(local_path.T)] = True
    distance = ndimage.distance_transform_edt(~center)
    p = probability[window].astype(np.float64)
    allowed = (distance <= 3) & (p >= floor) & ~mask[window]
    allowed[tuple(local_path[[0,-1]].T)] = True
    costs = np.full(p.shape,np.inf)
    costs[allowed] = (.35/np.maximum(.05,p[allowed]))**2 + .08*distance[allowed]
    # Bound anchors to one edge point; never travel backwards through a raw
    # component to find a different, easier attachment behind the declared tip.
    try:
        route,cost = route_through_array(costs,tuple(local_path[0]),tuple(local_path[-1]),
                                        fully_connected=True,geometric=True)
    except ValueError:
        return None
    if not np.isfinite(cost):
        return None
    return np.array(route,dtype=int)+lo


def find_corridors(mask,probability,umbilicus_yx,policy=ProposalPolicy(probability_floor=.2)):
    proposals,census = base.find_corridors(mask,probability,umbilicus_yx,policy)
    labels,_ = ndimage.label(mask,EIGHT)
    result=[]
    for row in proposals:
        original=np.array(row["path_yx"])
        path=refine_route(original,mask,probability,policy.probability_floor)
        if path is None:
            census["refinement_failed"]=census.get("refinement_failed",0)+1
            continue
        features=base.route_features(path,probability,labels,umbilicus_yx)
        if features["length_px"] > policy.maximum_length:
            census["refinement_over_reach"]=census.get("refinement_over_reach",0)+1
            continue
        row={**row,**features,"skeleton_path_yx":row["path_yx"],"path_yx":path.tolist(),
             "maximum_local_turn_degrees":local_turn_degrees(path)}
        result.append(row)
    census["refined_corridors"]=len(result)
    return result,census


def refusal_reason(row,policy=AcceptancePolicy()):
    reason=base.refusal_reason(row,policy)
    if reason:
        return reason
    if min(row["endpoint_outward"]) < policy.minimum_outward_alignment:
        return "backwards_anchor_approach"
    if row["maximum_local_turn_degrees"] > policy.maximum_local_turn_degrees:
        return "abrupt_local_turn"
    return None


def select_and_paint(mask,probability,proposals,floor,policy=AcceptancePolicy()):
    # Apply additional guards before the original transactional per-plane
    # selection, keeping every refusal in the combined ledger.
    eligible=[]
    refused=[]
    for row in proposals:
        reason=refusal_reason(row,policy)
        if reason:
            refused.append({**row,"accepted":False,"reason":reason})
        else:
            eligible.append(row)
    after,ledger,census=base.select_and_paint(mask,probability,eligible,floor,policy)
    for row in refused:
        census[row["reason"]]=census.get(row["reason"],0)+1
    return after,ledger+refused,census
