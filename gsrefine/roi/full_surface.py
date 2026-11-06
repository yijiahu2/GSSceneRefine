"""Full-exterior Poisson topology diagnostics, distinct from LOCAL Gaussian recovery.

Raw exterior mesh provides continuity hypotheses. Confidence is graded for graph
travel and only above-threshold faces count as reliable area; confidence is not
semantic rock/background truth. No exterior face beyond the original bounded
candidate is ever passed to automatic Gaussian recovery.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


def exterior_masks(inside, credible):
    inside=np.asarray(inside,bool)
    credible=np.asarray(credible,bool)
    return (~inside), (~inside)&credible


def full_exterior_graph(mesh, inside, confidence, threshold, penalty=2.0):
    """Outside-only face graph. Untrusted faces are passable at increased cost.

    This is an approximate geodesic on face centroids, not an exact intrinsic
    shortest path. Interior mesh faces never connect otherwise disconnected
    exterior components through the ROI. Disconnected raw mesh stays disconnected.
    """
    n=len(mesh.faces)
    adj=np.asarray(mesh.face_adjacency,dtype=np.int64)
    outside=~np.asarray(inside,bool)
    edges=adj[outside[adj[:,0]] & outside[adj[:,1]]]
    center=np.asarray(mesh.triangles_center)
    if len(edges):
        length=np.linalg.norm(center[edges[:,0]]-center[edges[:,1]],axis=1)
        weak=np.maximum(0.,threshold-np.minimum(confidence[edges[:,0]],confidence[edges[:,1]])) / max(threshold,1e-8)
        cost=np.maximum(length,1e-9)*(1.+float(penalty)*weak)
    else: cost=np.empty(0)
    return coo_matrix((np.r_[cost,cost],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()


def cumulative_area(distances, face_ids, areas, scales):
    distances=np.asarray(distances)[face_ids]
    areas=np.asarray(areas)[face_ids]
    order=np.argsort(distances)
    distances=distances[order]
    areas=areas[order]
    sums=np.cumsum(areas)
    pos=np.searchsorted(distances,scales,side='right')
    return np.where(pos>0,sums[np.maximum(pos-1,0)],0.).astype(float)


def full_exterior_analysis(mesh, inside, credible, candidate, confidence, threshold,
                           component_faces, group_records, scene_diag, study_cfg=None):
    """Measure COMPLETE outside area, and large-scale growth of candidate components.

    Returned face masks are for inspection ONLY. `candidate` remains the original
    confidence+geodesic+distance bounded set and is NEVER enlarged here.
    """
    cfg=study_cfg or {}
    raw, trusted=exterior_masks(inside,credible)
    a=np.asarray(mesh.area_faces,float)
    total_credible=float(a[trusted].sum())
    base_credible=float(a[np.asarray(inside)&credible].sum())
    local_area=float(a[candidate].sum())
    multiple=float(cfg.get('full_growth_max_diag_frac',.65))
    if not 0 < multiple <= 3:raise ValueError('full_growth_max_diag_frac must lie in (0,3]')
    steps=int(cfg.get('full_growth_steps',9))
    if steps<4 or steps>32:raise ValueError('full_growth_steps must be 4..32')
    # Uniform spatial scales, all in unscaled original COLMAP model units.
    maximum=max(float(scene_diag)*multiple,1e-9)
    scales=np.linspace(maximum/steps,maximum,steps)
    stats={'raw_exterior_face_count':int(raw.sum()),
           'credible_exterior_face_count':int(trusted.sum()),
           'raw_exterior_area_centroid_approx':float(a[raw].sum()),
           'credible_exterior_area_centroid_approx':total_credible,
           'credible_internal_area_centroid_approx':base_credible,
           'full_credible_exterior_to_internal_ratio':total_credible/max(base_credible,1e-12),
           'bounded_candidate_area':local_area,
           'bounded_to_full_credible_exterior_fraction':local_area/max(total_credible,1e-12),
           'area_method':'full faces classified by centroid; exact ROI face clipping not yet implemented',
           'topology_method':'original complete open Poisson adjacency, low-confidence faces passable at penalty',
           'low_confidence_faces_not_used_for_gaussian_recovery':True,
           'full_growth_scales_model_units':scales.tolist()}
    # A separate pass on the raw mesh means it stays connected across low-confidence
    # holes for *diagnosis*, without declaring such faces reliable.
    graph=full_exterior_graph(mesh,inside,confidence,threshold,
                              penalty=float(cfg.get('low_confidence_path_penalty',2.)))
    members=defaultdict(list)
    for row, fi in zip(group_records,component_faces):
        members[int(row['component'])].append(np.asarray(fi,dtype=np.int64))
    ranking=sorted(members,key=lambda k:-sum(float(a[v].sum()) for v in members[k]))
    max_comps=int(cfg.get('full_growth_max_components',16))
    if max_comps<1:raise ValueError('full_growth_max_components must be >=1')
    reports=[]
    for rank,comp in enumerate(ranking):
        ids=np.unique(np.concatenate(members[comp]))
        local=float(a[ids].sum())
        result={'component':int(comp),'local_candidate_area':local,
                'local_candidate_face_count':int(len(ids)),
                'full_growth_evaluated':bool(rank<max_comps)}
        if rank>=max_comps:
            result['reason']='skipped_component_count_limit'
            reports.append(result);continue
        dist=dijkstra(graph,directed=False,indices=ids,min_only=True,limit=maximum)
        raw_ids=np.flatnonzero(raw)
        reliable_ids=np.flatnonzero(trusted)
        raw_curve=cumulative_area(dist,raw_ids,a,scales)
        credible_curve=cumulative_area(dist,reliable_ids,a,scales)
        far=np.isfinite(dist) & raw & (dist > scales[0]) & (dist <= scales[-1])
        far_credible=far & trusted
        reliable_far_fraction=float(a[far_credible].sum()/max(a[far].sum(),1e-12))
        # Growth beyond local ROI buffer, i.e. truly use the full exterior.
        end=float(credible_curve[-1]); prior=float(credible_curve[len(scales)//2-1])
        persistence=(end-prior)/max(end,1e-12)
        result.update({'full_raw_growth_area':raw_curve.tolist(),
                       'full_credible_growth_area':credible_curve.tolist(),
                       'full_credible_reached_area':end,
                       'full_credible_to_local_ratio':end/max(local,1e-12),
                       'outer_half_credible_growth_fraction':float(persistence),
                       'far_raw_area':float(a[far].sum()),
                       'far_credible_area':float(a[far_credible].sum()),
                       'far_reliable_area_fraction':reliable_far_fraction,
                       'reached_max_search_distance':bool(np.any(np.isfinite(dist[raw]) & (dist[raw] >= .98*maximum))),
                       'interpretation':'Topology/continuation diagnostics only, not target-ground-truth'})
        reports.append(result)
    return {'stats':stats,'raw_mask':raw,'credible_mask':trusted,
            'components':reports}
