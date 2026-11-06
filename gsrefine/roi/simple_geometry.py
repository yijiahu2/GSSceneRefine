"""Conservative, explainable ROI exterior geometry review (no semantic classifier).

Design contract
---------------
1. Analyze the COMPLETE open Poisson exterior once, by actual mesh connectivity.
2. Identify normal perimeter continuation by joint evidence: exterior surface
   scale, contact on multiple ROI edges, and contact near the base.
3. For remaining LOCAL bounded candidates, require adjacency to ROI interior,
   near-boundary distance, and plausible local surface-area ratio TOGETHER.
4. Gaussian restoration never uses full exterior; only approved LOCAL faces.

No Poisson confidence scoring is performed here. `credible` is the upstream
face mask (all True by default in roi-study). No learned weights or semantic
claims. All areas use original triangle areas with ROI membership determined by
triangle centroid; boundary clipping remains an explicitly reported approximation.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from .core import mesh_external_candidates
from .boundary_analysis import (vertex_crossings, assign_candidate_boundaries,
                                classify_view_edges, angular_coverage)
from .surface_growth import face_graph
from .full_surface import full_exterior_graph


def _test_config(cfg):
    """Small set of interpretable geometric tolerances, NOT a scoring model."""
    defaults = dict(
        background_min_full_to_internal=.30,
        background_min_full_to_local=2.0,
        background_min_contact_edges=2,
        background_min_base_fraction=.12,
        background_min_ring_coverage=.14,
        background_max_local_height=.55,
        local_max_area_to_internal=.20,
        local_relaxed_area_to_internal=.35,
        local_relaxed_distance_fraction=.45,
        local_max_contact_coverage=.50,
    )
    result={k:cfg.get(k,v) for k,v in defaults.items()}
    for k,v in result.items():
        val=float(v)
        if not np.isfinite(val) or val < 0:
            raise ValueError(f'Invalid roi_study.{k}={v}')
        if k.endswith('_fraction') or 'coverage' in k or k.endswith('_height'):
            if val > 1.:
                raise ValueError(f'roi_study.{k} must be <= 1')
    if int(result['background_min_contact_edges']) < 2:
        raise ValueError('background_min_contact_edges must be >= 2')
    return result


def decide_candidate(*, background, boundary_class, height, inside_connected,
                     area_ratio, outside_distance, outside_cap, contact_coverage,
                     params):
    """Simple JOINT logical rules; no area-only decision or blanket recovery.

    Returned states: normal, suspect (eligible for guarded recovery), uncertain.
    """
    if (background and boundary_class != 'top'
            and height <= params['background_max_local_height']):
        return 'normal', 'full_exterior_multiedge_base_continuation'
    if not inside_connected:
        return 'uncertain', 'not_mesh_connected_to_roi_interior'
    if outside_distance > outside_cap * (1. + 1e-8):
        return 'uncertain', 'outside_local_search_range'
    localized=(contact_coverage <= params['local_max_contact_coverage']
               or boundary_class == 'top')
    modest=area_ratio <= params['local_max_area_to_internal']
    narrowly_attached=(area_ratio <= params['local_relaxed_area_to_internal']
                       and outside_distance <= outside_cap * params['local_relaxed_distance_fraction'])
    if localized and (modest or narrowly_attached):
        return 'suspect', 'connected+near_roi+local_area_plausible'
    return 'uncertain', 'joint_geometry_not_sufficient_for_auto_recovery'


def analyze_simple(mesh, roi, conf, threshold, diag, guards, up, *,
                   candidate_labels=None, study_cfg=None):
    """Replace heuristic growth classifier with complete-outside connectivity.

    Return the same public keys as `study.analyze`, so existing visualization,
    reports, CLI, and R0-R3 guard/export remain compatible.
    """
    cfg=_test_config(study_cfg or {})
    up=np.asarray(up,dtype=float);up=up/np.linalg.norm(up)
    centers=np.asarray(mesh.triangles_center,dtype=float)
    areas=np.asarray(mesh.area_faces,dtype=float)
    sd, inside, credible, candidate, ext_diag=mesh_external_candidates(
        mesh,roi,centers,conf,threshold,diag,
        {**guards,'extension_area_policy':'review'})
    roi_internal_area=float(areas[inside & credible].sum())
    if not roi_internal_area > 0:
        raise ValueError('ROI has zero accepted Poisson interior area')
    out_cap=float(ext_diag['outside_search'])
    candidate_ids=np.flatnonzero(candidate)
    trace,trace_labels,trace_parent,edge_meta=vertex_crossings(
        mesh,roi,credible,up)
    assigned,trace_dist,violation=assign_candidate_boundaries(
        centers,candidate_ids,trace,trace_labels,roi,up)
    meta={(r['view_index'],r['edge_index']):r for r in edge_meta}

    # LOCAL components are needed to split multiple, possibly disconnected,
    # patches contacting a given ROI edge. These are NOT full-background units.
    _,local_labels=connected_components(face_graph(mesh,candidate),directed=False)

    # Complete OUTSIDE components are measured ONCE; no separate global Dijkstra
    # from 117 differently sized candidate seed sets. The exterior graph NEVER
    # travels through interior faces to merge distinct outside components.
    full_mask=~inside
    _,global_labels=connected_components(
        full_exterior_graph(mesh,inside,np.ones(len(centers)),0.,penalty=0.),
        directed=False)
    full_component_area=np.bincount(global_labels[full_mask],
                                  weights=areas[full_mask],minlength=len(centers))
    full_component_faces=np.bincount(global_labels[full_mask],minlength=len(centers))
    trusted_component_area=np.bincount(global_labels[full_mask & credible],
         weights=areas[full_mask & credible],minlength=len(centers))
    # Direct Poisson adjacency through the initial ROI boundary.
    adjacency=np.asarray(mesh.face_adjacency,dtype=np.int64)
    cut_edges=adjacency[inside[adjacency[:,0]] ^ inside[adjacency[:,1]]]
    exterior_contact=np.where(inside[cut_edges[:,0]],cut_edges[:,1],cut_edges[:,0])
    direct_contact=np.bincount(global_labels[exterior_contact],minlength=len(centers))

    internal_centers=centers[inside & credible]
    anchor=internal_centers.mean(axis=0)
    h_all=centers@up
    h05,h95=np.quantile(h_all[inside & credible],[.05,.95])
    hspan=max(float(h95-h05),1e-9)

    groups_idx=defaultdict(list)
    for idx,fid in enumerate(candidate_ids):
        vi,ei=map(int,assigned[idx]); local=int(local_labels[fid])
        groups_idx[(vi,ei,local)].append(int(fid))

    # Build stable patch-level records and retain associated actual face IDs.
    records=[];parts=[]
    overrides=candidate_labels or {}
    for (vi,ei,local),fids in sorted(groups_idx.items(),
            key=lambda kv:(-float(areas[kv[1]].sum()),kv[0])):
        fi=np.asarray(fids,dtype=np.int64)
        gcomp=int(global_labels[fi[0]])
        assert np.all(global_labels[fi]==gcomp), 'Local component crosses full mesh components'
        descriptor=meta[vi,ei]
        point_height=float(np.median((h_all[fi]-h05)/hspan))
        point_height=float(np.clip(point_height,0.,1.))
        # An observed ROI/Possion crossing is preferred over all patch centers.
        contact=np.empty((0,3),dtype=float)
        if len(trace):
            matching=trace[np.all(trace_labels==[vi,ei],axis=1)]
            if len(matching):
                distances,_=cKDTree(centers[fi]).query(matching,workers=-1)
                contact=matching[distances<=max(out_cap,1e-5*diag)]
        coverage=angular_coverage(contact if len(contact) else centers[fi],up,anchor)
        area=float(areas[fi].sum())
        outside=np.maximum(0.,-np.min(sd[fi],axis=1))
        record={
            'id':f'v{vi:02d}_e{ei:02d}_c{local:06d}',
            'view':descriptor['view_name'],'view_index':vi,'edge':ei,
            'component':local,'global_exterior_component':gcomp,
            'boundary_class':descriptor['boundary_class'],
            'area':area,'area_to_internal_fraction':area/roi_internal_area,
            'fraction_of_all_exterior':area/max(float(areas[candidate].sum()),1e-12),
            'faces':len(fi),'median_relative_height':point_height,
            'azimuth_coverage':coverage,
            'median_outside_violation':float(np.median(outside)),
            'max_outside_violation':float(np.max(outside)),
            'boundary_trace_samples':int(len(contact)),
            'centroid_xyz':centers[fi].mean(axis=0).tolist(),
            'inside_mesh_connected':bool(direct_contact[gcomp]>0),
            'manual_override':False,
        }
        records.append(record);parts.append(fi)

    grouped_global=defaultdict(list)
    for row in records: grouped_global[row['global_exterior_component']].append(row)
    bg_reports=[];by_global={}
    for gcomp,rows in sorted(grouped_global.items(),
                             key=lambda kv:-float(full_component_area[kv[0]])):
        loc_sum=sum(r['area'] for r in rows)
        edges={(r['view_index'],r['edge']) for r in rows}
        base_rows=[r for r in rows if r['boundary_class']=='bottom'
                   or (r['boundary_class'] in ('side','plan_side')
                       and r['median_relative_height']<=cfg['background_max_local_height'])]
        base_area=sum(r['area'] for r in base_rows)
        # Angle coverage for all base local candidate triangles TOGETHER; unlike
        # single polygon-edge coverage, works across front+top ROI boundaries.
        base_ids={r['id'] for r in base_rows}
        base_parts=[parts[i] for i,r in enumerate(records) if r['id'] in base_ids]
        base_xyz=centers[np.concatenate(base_parts)] if base_parts else np.empty((0,3))
        ring=float(angular_coverage(base_xyz,up,anchor))
        full_area=float(full_component_area[gcomp])
        checks={
            'full_exterior_area_large_enough':full_area/roi_internal_area>=cfg['background_min_full_to_internal'],
            'full_area_exceeds_local_contact_area':full_area/max(loc_sum,1e-12)>=cfg['background_min_full_to_local'],
            'multi_roi_boundary_contact':len(edges)>=int(cfg['background_min_contact_edges']),
            'base_support':base_area/max(loc_sum,1e-12)>=cfg['background_min_base_fraction'],
            'base_angular_spread':ring>=cfg['background_min_ring_coverage'],
            'connects_across_roi_boundary':bool(direct_contact[gcomp]>0),
        }
        is_background=bool(all(checks.values()))
        info={
            'component':int(gcomp),'global_exterior_component':int(gcomp),
            'full_raw_exterior_area':full_area,
            'full_trusted_exterior_area':float(trusted_component_area[gcomp]),
            'full_exterior_faces':int(full_component_faces[gcomp]),
            'local_candidate_area':float(loc_sum),
            'full_to_local_area_ratio':full_area/max(loc_sum,1e-12),
            'full_to_internal_area_ratio':full_area/roi_internal_area,
            'n_crossed_edges':int(len(edges)),'base_supported_area_fraction':base_area/max(loc_sum,1e-12),
            'base_contact_azimuth_coverage':ring,
            'direct_roi_crossing_edges':int(direct_contact[gcomp]),
            'checks':checks,'expected_perimeter_continuation':is_background,
            'classification_reason':('broad_multiedge_base_continuation'
                if is_background else 'insufficient_joint_background_evidence'),
        }
        bg_reports.append(info);by_global[gcomp]=info

    labels={k:np.zeros(len(centers),dtype=bool)
            for k in ('normal','suspect','uncertain')}
    face_group_ids=np.full(len(centers),-1,dtype=np.int32)
    for i,(r,fi) in enumerate(zip(records,parts)):
        evidence=by_global[r['global_exterior_component']]
        automatic,reason=decide_candidate(
            background=evidence['expected_perimeter_continuation'],
            boundary_class=r['boundary_class'],height=r['median_relative_height'],
            inside_connected=r['inside_mesh_connected'],
            area_ratio=r['area_to_internal_fraction'],
            outside_distance=r['max_outside_violation'],outside_cap=out_cap,
            contact_coverage=r['azimuth_coverage'],params=cfg)
        chosen=overrides.get(r['id'],automatic)
        if chosen not in labels:
            raise ValueError(f'Invalid label {chosen} for group {r["id"]}')
        r.update({'background_component_candidate':evidence['expected_perimeter_continuation'],
                  'background_component_area_to_internal_fraction':evidence['full_to_internal_area_ratio'],
                  'background_component_crossed_edges':evidence['n_crossed_edges'],
                  'background_component_base_fraction':evidence['base_supported_area_fraction'],
                  'background_component_ring_coverage':evidence['base_contact_azimuth_coverage'],
                  'classification':automatic,'classification_reason':reason,
                  'effective_label':chosen,'manual_override':r['id'] in overrides,
                  'growth_evaluated':False,
                  'growth_curve_area':None,
                  'full_growth_evaluated':False})
        labels[chosen][fi]=True;face_group_ids[fi]=i
    unexpected=set(overrides)-{r['id'] for r in records}
    if unexpected:
        raise ValueError(f'Unknown group label IDs: {sorted(unexpected)[:5]}')

    full_raw=~inside
    full_credible=full_raw & credible
    ext_area=float(areas[candidate].sum())
    full_area=float(areas[full_credible].sum())
    stat_full={
        'raw_exterior_face_count':int(full_raw.sum()),
        'credible_exterior_face_count':int(full_credible.sum()),
        'raw_exterior_area_centroid_approx':float(areas[full_raw].sum()),
        'credible_exterior_area_centroid_approx':full_area,
        'credible_internal_area_centroid_approx':roi_internal_area,
        'full_credible_exterior_to_internal_ratio':full_area/roi_internal_area,
        'bounded_candidate_area':ext_area,
        'bounded_to_full_credible_exterior_fraction':ext_area/max(full_area,1e-12),
        'area_method':'triangle-centroid classification (approximate at ROI boundary)',
        'topology_method':'full original Poisson outside-only face adjacency, global component measured ONCE',
        'global_components_with_local_candidates':len(bg_reports),
        'no_repeated_full_area_across_local_sources':True,
        'low_confidence_faces_not_used_for_gaussian_recovery':True,
    }
    stats={
        'internal_credible_area_centroid_approx':roi_internal_area,
        'initial_exterior_candidate_area':ext_area,
        'initial_exterior_ratio':ext_area/roi_internal_area,
        'groups':len(records),'boundary_samples':len(trace),
        'connected_components':len(set(int(local_labels[f]) for f in candidate_ids)),
        'global_exterior_components_with_candidates':len(bg_reports),
        'expected_perimeter_components':sum(int(r['expected_perimeter_continuation']) for r in bg_reports),
        'areas_by_label':{k:float(areas[v].sum()) for k,v in labels.items()},
        'full_exterior':stat_full,
        'face_centroid_area_estimate':True,
        'algorithm':'simple_geometry_v1',
        'geometry_limitations':[
            'ROI face-center inclusion, not exact triangle clipping.',
            'ROI boundary crossing is a sampled mesh-edge intersection.',
            'Normal and suspect are geometry-based hypotheses, not semantic labels.',
            'Normal perimeter classification uses complete outside connectivity, not growth curve saturation.',
            'Only limited-distance, connected local candidates can enter Gaussian restoration.',
        ],
    }
    return dict(sd=sd,inside=inside,credible=credible,candidate=candidate,
                groups=records,group_ids=face_group_ids,label_masks=labels,
                traces=trace,trace_labels=trace_labels,stats=stats,
                component_diagnostics=bg_reports,
                full_raw_mask=full_raw,full_credible_mask=full_credible,
                full_component_diagnostics=bg_reports)
