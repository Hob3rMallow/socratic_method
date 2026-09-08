"""Read-only numerical audit of saved patch geometry and masks; no fitting."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import run_f0_repair_context as context
from render_f0_repair_review import assemble


def verify_mesh(mesh):
    v, f = mesh["vertices"], mesh["faces"]
    if not np.isfinite(v).all() or f.min() < 0 or f.max() >= len(v):
        raise ValueError("invalid mesh coordinates/indices")
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                      return_inverse=True, return_counts=True)
    if counts.max() > 2:
        raise ValueError("nonmanifold edge")
    orientation = np.bincount(inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1))
    if (orientation[counts == 2] != 0).any():
        raise ValueError("inconsistent winding")
    degree = np.bincount(edges[counts == 1].ravel(), minlength=len(v))
    if ((degree != 0) & (degree != 2)).any():
        raise ValueError("pinched boundary")
    face_ids = np.tile(np.arange(len(f)), 3)
    order = np.argsort(inverse, kind="stable")
    offsets = np.r_[0, np.cumsum(counts)]
    twin = np.flatnonzero(counts == 2)
    a, b = face_ids[order[offsets[twin]]], face_ids[order[offsets[twin] + 1]]
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(f), len(f))).tocsr()
    if connected_components(graph, directed=False, return_labels=False) != 1:
        raise ValueError("disconnected face complex")
    cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    areas = np.linalg.norm(cross, axis=1) / 2
    if areas.min() <= 1e-6:
        raise ValueError("degenerate triangle")
    return {"vertices": len(v), "faces": len(f), "area_px2": float(areas.sum()),
            "z_span": int(np.ptp(v[:, 0])) + 1,
            "euler_characteristic": int(len(v) - len(edges) + len(f)),
            "maximum_edge_px": float(np.linalg.norm(v[edges[:, 1]] - v[edges[:, 0]], axis=1).max())}


def raster(mesh, probability):
    """Independent implementation of the declared quarter-step face raster."""
    v, f = mesh["vertices"], mesh["faces"]
    low = np.maximum(0, np.floor(v.min(axis=0) - 2).astype(int))
    high = np.minimum(probability.shape, np.ceil(v.max(axis=0) + 3).astype(int))
    w = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    tri = v[f]
    weights = np.array([(1 - (a+b)/4, a/4, b/4) for a in range(5) for b in range(5-a)])
    pos = np.rint(np.einsum("wv,fvc->fwc", weights, tri)).astype(int).reshape(-1, 3) - low
    paint = np.zeros(tuple(high-low), bool)
    paint[tuple(pos.T)] = True
    paint = ndimage.binary_dilation(paint, ndimage.generate_binary_structure(3, 1))
    z = np.arange(low[0], high[0])
    paint[(z < v[:, 0].min()) | (z > v[:, 0].max())] = False
    paint &= probability[w] >= .16
    return w, paint


def agreement(a, b):
    d, ix = cKDTree(b["vertices"]).query(a["vertices"], distance_upper_bound=3)
    near = np.isfinite(d)
    if not near.any():
        return 0.
    delta = a["vertices"][near] - b["vertices"][ix[near]]
    na, nb = a["normals"][near], b["normals"][ix[near]]
    distance = np.maximum(abs((delta*na).sum(axis=1)), abs((delta*nb).sum(axis=1)))
    return float(((distance <= .8) & (abs((na*nb).sum(axis=1)) >= .85)).mean())


def audit(folder):
    request = context.read(folder / "request.json")
    summary = context.read(folder / "summary.json")
    if context.read(folder / "state.json")["status"] != "complete":
        raise ValueError("incomplete run")
    for item in [*request["input_files"], *request["code"], *summary["output_files"]]:
        context.verify_file(item)
    prior = Path(request["source_continuity"])
    if (prior / "continuity_receipt.json").exists():
        old = {"inference": context.read(prior / "continuity_receipt.json")["plan"]["source"]}
    else:
        old = context.read(prior / "request.json")
    ids, lower = request["spec"]["target_cube_ids"], request["spec"]["origin_zyx"]
    base = assemble(prior / "cubes_PRED", ids, lower) != 0
    after = assemble(folder / "cubes_PRED", ids, lower) != 0
    p = assemble(Path(old["inference"]) / "probability", ids, lower).astype(np.float32)
    raw = assemble(Path(old["inference"]) / "cubes_PRED", ids, lower) != 0
    if (base & ~after).any():
        raise ValueError("erased foreground")
    added = after & ~base
    if int(added.sum()) != summary["added_voxels_over_v2"] or (p[added] < .16).any():
        raise ValueError("probability or output count mismatch")
    union = np.zeros(base.shape, bool)
    owners = np.zeros(base.shape, np.uint16)
    meshes, records, interactions = {}, [], []
    for row in context.read(folder / "patches.json")["patches"]:
        data = np.load(folder / row["mesh_arrays"], allow_pickle=False)
        mesh = {"vertices": data["vertices_local_zyx"], "faces": data["faces"], "normals": data["normals_zyx"]}
        metrics = verify_mesh(mesh)
        if abs(metrics["area_px2"] - row["metrics"]["mesh_area_px2"]) > 1e-5:
            raise ValueError("area mismatch")
        w, paint = raster(mesh, p)
        if ndimage.label(paint, np.ones((3, 3, 3), bool))[1] != 1 or not (paint & base[w]).any():
            raise ValueError("paint disconnected or detached from base")
        new = paint & ~base[w]
        unique = new & ~union[w]
        if int(unique.sum()) != row["unique_added_voxels"]:
            raise ValueError("mesh/mask ownership mismatch")
        touched = owners[w][ndimage.binary_dilation(new, np.ones((3, 3, 3), bool))]
        for other in np.unique(touched[touched > 0]):
            forward, reverse = agreement(mesh, meshes[int(other)]), agreement(meshes[int(other)], mesh)
            interactions.append({"new_patch": row["patch_id"], "old_patch": int(other),
                                 "forward_fraction": forward, "reverse_fraction": reverse})
        union[w] |= new
        owners[w][unique] = row["patch_id"]
        meshes[row["patch_id"]] = mesh
        records.append({"patch_id": row["patch_id"], "seed_id": row["seed_id"],
                        "prior_added_component": row["prior_added_component"], **metrics,
                        "unique_new_voxels": int(unique.sum()), "modeled_surfaces": row["metrics"]["modeled_surfaces"]})
    if not np.array_equal(union, added):
        raise ValueError("final mask differs from union of explicit meshes")
    cc, number = ndimage.label(added, np.ones((3, 3, 3), bool))
    attached = np.unique(cc[ndimage.binary_dilation(base, np.ones((3, 3, 3), bool)) & added])
    if len(attached) != number:
        raise ValueError("detached new component")
    if any(r["forward_fraction"] < .9 for r in interactions):
        raise ValueError("accepted interaction violates directional policy")
    result = {"schema": "sheet-patch-independent-audit-v1", "status": "passed_declared_checks",
              "run": str(folder), "erased_voxels": 0, "added_voxels": int(added.sum()),
              "raw_foreground": int(raw.sum()), "v2_foreground": int(base.sum()),
              "total_added_over_raw": int((after & ~raw).sum()), "added_components": number,
              "all_added_components_touch_base": True, "all_masks_match_explicit_mesh_union": True,
              "added_probability_quantiles": np.quantile(p[added], [0, .1, .5, .9, 1]).tolist() if added.any() else [],
              "patches": records, "interactions": interactions,
              "reverse_direction_below_090": sum(r["reverse_fraction"] < .9 for r in interactions),
              "span_quantiles": np.quantile([r["z_span"] for r in records], [0, .25, .5, .75, 1]).tolist() if records else [],
              "anatomical_correctness_not_established": True}
    context.atomic_json(folder / "independent_audit.json", result)
    print({k: v for k, v in result.items() if k not in ("patches", "interactions")}, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    audit(parser.parse_args().run)
