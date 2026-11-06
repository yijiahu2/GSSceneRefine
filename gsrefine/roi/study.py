"""R0-R3 full-exterior topology and LOCAL Gaussian ROI protection experiment.

One project, one environment. Inputs are never edited. Use `python run.py roi-study`.
Method R0: initial hard ROI
Method R1: all distance/geodesic-bounded exterior candidates, global sigma
Method R2: geometry-validated LOCAL truncation candidates, global sigma
Method R3: restore accepted Poisson core THEN apply full-boundary adaptive 5% trimmed Gaussian-axis safety band

Poisson triangle confidence scoring is DISABLED by default (all faces trusted).
Caveat: geometric classification is NOT semantic ground truth. Boundary traces
and clipped surface areas are approximate (face-centroid / mesh-edge based).
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import trimesh
import yaml
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components, dijkstra
from .core import (ROI, GaussianPLY, file_sha256, geometry_confidence,
                   load_sfm, mesh_external_candidates, local_protection,
                   add_guard_gaussians, json_write)
from .boundary_analysis import (vertex_crossings, assign_candidate_boundaries,
                                angular_coverage, classify_view_edges)
from .surface_growth import face_graph, growth_curve, saturated_growth
from .candidate_classification import (triage, component_background_evidence,
                                       apply_component_background_evidence)
from .full_surface import full_exterior_analysis
from .two_stage_sdf import recover_two_stage_from_poisson


def _write_csv(path, rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def _write_faces(mesh, selected, path, color=None):
    if not np.any(selected):
        Path(path).unlink(missing_ok=True)
        return False
    geom=trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                          faces=np.asarray(mesh.faces)[selected],process=False)
    geom.remove_unreferenced_vertices()
    if color is not None:geom.visual.face_colors=np.tile(np.array(color,dtype=np.uint8),(len(geom.faces),1))
    geom.export(path)
    return True



def anchored_local_recovery_faces(mesh, roi, selected_faces, inside_faces,
                                  max_geodesic_distance):
    """Limit Stage-A restoration to locally continuous Poisson extensions.

    This is a *second, stage-A-only* geometry gate. It never changes the
    exterior classification, R0, R1/R2, or Stage-B Gaussian safety rule.

    A starting exterior face must directly share a triangle edge with an
    ROI-interior face, OR itself straddle the ROI boundary. Propagation then
    uses only the selected suspect-face adjacency graph, never jumps across
    nonselected/normal faces or free space. Multi-source intrinsic distance
    is limited using an existing scene guard distance; no positional rules.
    """
    selected = np.asarray(selected_faces, dtype=bool)
    inside = np.asarray(inside_faces, dtype=bool)
    count = len(mesh.faces)
    if selected.shape != (count,) or inside.shape != (count,):
        raise ValueError('Poisson face masks must match mesh.faces')
    if np.any(selected & inside):
        raise ValueError('Stage-A suspect faces must be outside ROI')
    if not np.isfinite(max_geodesic_distance) or max_geodesic_distance <= 0:
        raise ValueError('Positive finite geodesic extent required')
    areas = np.asarray(mesh.area_faces, dtype=float)
    result = np.zeros(count, dtype=bool)
    if not np.any(selected):
        return result, {
            'algorithm': 'roi_anchored_suspect_surface_geodesic',
            'selected_faces_before': 0, 'selected_faces_after': 0,
            'boundary_seed_faces': 0, 'max_geodesic_distance': float(max_geodesic_distance),
            'removed_faces': 0, 'selected_area_before': 0.,
            'selected_area_after': 0., 'removed_area': 0.,
            'unanchored_faces': 0, 'beyond_reach_faces': 0,
        }
    adj = np.asarray(mesh.face_adjacency, dtype=np.int64).reshape(-1, 2)
    crossing = adj[selected[adj[:, 0]] & inside[adj[:, 1]]]
    reverse = adj[selected[adj[:, 1]] & inside[adj[:, 0]]]
    seeds = np.zeros(count, dtype=bool)
    if len(crossing):
        seeds[crossing[:, 0]] = True
    if len(reverse):
        seeds[reverse[:, 1]] = True

    # A triangle intersected by the polygon boundary can have its centroid
    # outside without any neighbouring face-centroid classified as inside.
    # This protects genuine small truncated tips against centroid aliasing.
    vertices = np.asarray(mesh.vertices, dtype=float)
    vertex_inside = np.all(roi.distances(vertices) >= -1e-9, axis=1)
    ids = np.flatnonzero(selected)
    tris = np.asarray(mesh.faces, dtype=np.int64)[ids]
    any_inside = np.any(vertex_inside[tris], axis=1)
    any_outside = np.any(~vertex_inside[tris], axis=1)
    seeds[ids[any_inside & any_outside]] = True
    seeds &= selected

    if np.any(seeds):
        # Graph edges join face centers only through an actual shared edge.
        # Full-scene generic shortest paths can travel via normal perimeter
        # and re-enter another suspect patch; explicitly disallow this.
        graph = face_graph(mesh, selected)
        dist = dijkstra(graph, directed=False, indices=np.flatnonzero(seeds),
                        min_only=True, limit=float(max_geodesic_distance))
        result = selected & np.isfinite(dist) & (dist <= max_geodesic_distance)
        # The capped shortest-path solver returns inf both for disconnected
        # and over-limit faces. Separate these causes using mesh connectivity.
        _, components = connected_components(graph, directed=False)
        has_seed = np.zeros(count, dtype=bool)
        has_seed[components[seeds]] = True
        reached = selected & has_seed[components]
    else:
        # No interior-connected seeds: never recover an isolated suspect island
        # just because its centroid is spatially close to a selected object.
        reached = np.zeros(count, bool)
    removed = selected & ~result
    return result, {
        'algorithm': 'roi_anchored_suspect_surface_geodesic',
        'selected_faces_before': int(selected.sum()),
        'selected_faces_after': int(result.sum()),
        'boundary_seed_faces': int(seeds.sum()),
        'max_geodesic_distance': float(max_geodesic_distance),
        'removed_faces': int(removed.sum()),
        'selected_area_before': float(areas[selected].sum()),
        'selected_area_after': float(areas[result].sum()),
        'removed_area': float(areas[removed].sum()),
        'unanchored_faces': int(np.count_nonzero(selected & ~reached)),
        'beyond_reach_faces': int(np.count_nonzero(reached & ~result)),
        'connectivity': 'original_triangle_edge_adjacency_only',
        'seed_rule': 'ROI_interior_edge_contact_or_boundary_straddling_face',
        'scope': 'R3_stage_A_only',
    }


def analyze(mesh, roi, conf, threshold, diag, guard_cfg, up, *,
            scales_count=6, candidate_labels=None, growth_multiple=3.0,
            min_component_area_fraction=.0001, study_cfg=None):
    """Group finite exterior candidates by boundary edge AND connected component.

    Full exterior context is evaluated independently from bounded local candidates;
    never include farther faces as automatic recovery output.
    """
    study_cfg=study_cfg or {}
    # Default: concise, interpretable complete-exterior + local geometry rules.
    # Older topology-growth classifier remains callable only for reproducible
    # comparison with previous experiments.
    if not bool(study_cfg.get('legacy_growth_classifier', False)):
        from .simple_geometry import analyze_simple
        return analyze_simple(mesh,roi,conf,threshold,diag,guard_cfg,up,
                              candidate_labels=candidate_labels,study_cfg=study_cfg)
    if int(scales_count)<3 or float(growth_multiple)<=0:
        raise ValueError('growth_steps >= 3 and growth_multiple > 0 required')
    if int(study_cfg.get('max_growth_groups',32))<1:
        raise ValueError('max_growth_groups must be >=1')
    center=np.asarray(mesh.triangles_center)
    sd, inside, credible, ext, diag0=mesh_external_candidates(
        mesh, roi, center, conf, threshold, diag,
        {**guard_cfg,'extension_area_policy':'review'})
    # ext here is the PRE-BUDGET bounded set (policy always review).
    candidate_ids=np.flatnonzero(ext)
    area=np.asarray(mesh.area_faces,dtype=float)
    all_boundary=classify_view_edges(roi,up)
    traces,trace_idx,trace_face,edges=vertex_crossings(mesh,roi,credible,up)
    boundary_ids,dist_cross,viol=assign_candidate_boundaries(center,candidate_ids,traces,trace_idx,roi,up)
    boundary_meta={(r['view_index'],r['edge_index']):r for r in edges}
    search=float(diag0['outside_search'])
    multi_cap=max(search*growth_multiple,1e-9)
    # Retain old short-range curves as a local baseline; full-area topology and
    # full-scale growth are independently analyzed in full_surface.py below.
    growth_region=credible & (sd.min(axis=1)>=-multi_cap) & (
        sd.min(axis=1)<=float(guard_cfg['inner_band_frac'])*diag)
    source_graph=face_graph(mesh, growth_region)
    subgraph=face_graph(mesh, ext)
    # Connected components on entire index range, but only candidate ids inspected.
    _, components=connected_components(subgraph,directed=False)
    # Parent keys: bound edge AND candidate adjacency component.
    from collections import defaultdict
    groups=defaultdict(list)
    for ix,fid in enumerate(candidate_ids):
        vi,ei=map(int,boundary_ids[ix]);groups[(vi,ei,int(components[fid]))].append(int(fid))
    source_heights=center@up
    h_low,h_high=np.quantile(source_heights[credible],[.05,.95])
    span=max(h_high-h_low,1e-12)
    origin=np.mean(center[credible & inside],axis=0) if np.any(credible & inside) else center.mean(axis=0)
    scales=np.linspace(0,multi_cap,scales_count+1)[1:]
    inside_band=credible & inside & (sd.min(axis=1) <= float(guard_cfg['inner_band_frac'])*diag)
    source_centers=center[np.flatnonzero(inside_band)]
    inside_ids=np.flatnonzero(inside_band)
    inside_tree=cKDTree(source_centers) if len(source_centers) else None
    total_external=float(area[ext].sum())
    group_items=[]; component_faces=[]
    labels_override=candidate_labels or {}
    # Limit expensive whole-mesh Dijkstra to the largest candidate groups.
    # Tiny components are retained and marked uncertain, never silently lost.
    max_growth_groups=int(study_cfg.get('max_growth_groups',32))
    # Unique group IDs stable for identical geometry/inputs, not under mesh remeshing.
    for rank,(key,fids) in enumerate(sorted(groups.items(),key=lambda kv: -float(area[kv[1]].sum()))):
        fi=np.asarray(fids,dtype=int); local_area=float(area[fi].sum())
        if local_area < min_component_area_fraction*max(float(area[inside&credible].sum()),1e-12):
            # Tiny patches still appear as individual uncertain groups.
            pass
        vi,ei,comp=key; meta=boundary_meta[(vi,ei)]
        # Prefer crossing faces from this exact ROI polygon edge; if none are
        # present nearby, use near-boundary inside seeds closest to candidates.
        matched_trace=trace_face[np.all(trace_idx==[vi,ei],axis=1)] if len(trace_idx) else np.empty(0,int)
        matched_trace=np.intersect1d(matched_trace,np.flatnonzero(credible))
        if len(matched_trace):
            tree=cKDTree(center[fi]); nearest,_=tree.query(center[matched_trace]);
            matched_trace=matched_trace[nearest <= max(search,.05*diag)]
        if len(matched_trace):
            sources=np.unique(matched_trace)
        elif inside_tree is not None:
            _,j=inside_tree.query(center[fi][::max(1,len(fi)//24)],workers=-1)
            sources=np.unique(inside_ids[np.atleast_1d(j)])
        else:sources=np.array([],int)
        if rank < max_growth_groups:
            curve,_=growth_curve(source_graph,sources,credible & ~inside,area,scales)
        else:
            # Not measured is distinct from zero growth or saturation.
            curve=np.full(len(scales),np.nan)
        hit_mesh_boundary=False
        # Candidate face touches mesh edge if any of its three edges has no
        # adjacent face (boundary of raw open Poisson mesh).
        face_edges=np.sort(np.asarray(mesh.faces[fi])[:,[[0,1],[1,2],[2,0]]],axis=-1).reshape(-1,2)
        # Avoid constructing entire-mesh boundary set for each group;
        # use an incidence map provided by trimesh's unique face edges.
        # Boundary-termination flag is approximated via mesh vertex flags below.
        if getattr(mesh,'_roi_boundary_vertex_flags',None) is not None:
            hit_mesh_boundary=bool(np.any(mesh._roi_boundary_vertex_flags[np.asarray(mesh.faces)[fi]]))
        heights=(center[fi]@up-h_low)/span
        hq=float(np.median(heights))
        # Estimate perimeter wrapping from contact traces rather than all
        # exterior patch points (large background patches bias the latter).
        contact=traces[np.all(trace_idx==[vi,ei],axis=1)] if len(trace_idx) else np.empty((0,3))
        if len(contact):
            local_d,_=cKDTree(center[fi]).query(contact)
            contact=contact[local_d <= max(search, .05*diag)]
        az_cov=angular_coverage(contact if len(contact) else center[fi],up,origin)
        local_contact_count=int(len(contact))
        # Candidate area relative to ALL initial reliable interior surface.
        internal_area=max(float(area[inside&credible].sum()),1e-12)
        ratio=local_area/internal_area
        closed=saturated_growth(curve) if np.isfinite(curve).all() else False
        automatic,reason=triage(boundary_class=meta['boundary_class'],
            height_quantile=hq, angular_coverage=az_cov,
            saturated=closed,reaches_mesh_edge=hit_mesh_boundary,
            external_area_ratio=ratio,
            baseline_coverage=float(study_cfg.get('min_ring_coverage',.45)),
            high_height=float(study_cfg.get('high_height_quantile',.65)),
            base_height=float(study_cfg.get('base_height_quantile',.30)),
            max_suspect_area_ratio=float(study_cfg.get('max_suspect_area_fraction',.25)))
        group_id=f'v{vi:02d}_e{ei:02d}_c{comp:06d}'
        if rank >= max_growth_groups:
            automatic,reason='uncertain','growth_not_evaluated_group_limit'
        effective=labels_override.get(group_id,automatic)
        if effective not in ('normal','suspect','uncertain'):
            raise ValueError(f'Invalid label {effective} for {group_id}')
        record={'id':group_id,'view':meta['view_name'],'edge':ei,'component':comp,
            'boundary_class':meta['boundary_class'],'area':local_area,
            'area_to_internal_fraction':ratio,'fraction_of_all_exterior':local_area/max(total_external,1e-12),
            'faces':len(fi),'median_relative_height':hq,'azimuth_coverage':az_cov,
            'growth_curve_area':[float(x) if np.isfinite(x) else None for x in curve],'growth_scales':scales.tolist(),
            'growth_saturated':closed,'touches_open_mesh_boundary':hit_mesh_boundary,
            'classification':automatic,'classification_reason':reason,
            'effective_label':effective,'manual_override':group_id in labels_override,
            'centroid_xyz':center[fi].mean(axis=0).tolist(),
            'boundary_trace_samples':local_contact_count,
            'growth_evaluated':bool(rank < max_growth_groups)}
        group_items.append(record);component_faces.append(fi)
    # Cross-boundary base ring: a continuous surrounding terrain may cross
    # 3-4 different polygon edges. Aggregate spatial angular support for all
    # lower candidates in the SAME connected exterior mesh component.
    base_by_component={}
    for row,fi in zip(group_items,component_faces):
        if row['median_relative_height'] <= .30:
            base_by_component.setdefault(row['component'],[]).append(fi)
    ring_by_component={}
    for comp,parts in base_by_component.items():
        ring_by_component[comp]=angular_coverage(center[np.concatenate(parts)],up,origin)
    for row in group_items:
        ring=ring_by_component.get(row['component'],0.)
        row['base_component_azimuth_coverage']=ring
        # A local ring alone cannot prove background; classification follows
        # from component-level evidence on the complete exterior Poisson.
        if not row['manual_override']:
            row['effective_label']=row['classification']
    # Classify *connected base continuations* after aggregating across all ROI
    # polygon edges.  Critically, keep high/terminating portions of a component
    # eligible for suspect/uncertain rather than blanket background removal.
    internal_area=float(area[inside & credible].sum())
    component_diagnostics=component_background_evidence(group_items,internal_area,
                                                          study_cfg=study_cfg)
    # Full-scale context is DISTINCT from local bounded candidates. The full
    # Poisson exterior is used only as evidence for background continuation.
    full=full_exterior_analysis(mesh,inside,credible,ext,conf,threshold,
                               component_faces,group_items,diag,study_cfg)
    full_by_component={c['component']:c for c in full['components']}
    for cd in component_diagnostics.values():
        measured=full_by_component.get(cd['component'],{})
        cd['full_exterior']=measured
        if measured.get('full_growth_evaluated'):
            cfg=study_cfg
            checks={
                'extensive_full_surface': measured['full_credible_to_local_ratio'] >= float(cfg.get('full_normal_area_multiplier',2.)),
                'sustained_full_growth': measured['outer_half_credible_growth_fraction'] >= float(cfg.get('full_normal_growth_fraction',.12)),
                'credible_far_support': measured['far_reliable_area_fraction'] >= float(cfg.get('full_normal_reliable_far_fraction',.35)),
            }
            cd['full_checks']=checks
            # *Never* accept a base-background label solely from an adjacent
            # finite-width ring. Require the large-scale evidence too.
            cd['expected_perimeter_continuation']=bool(cd['expected_perimeter_continuation'] and all(checks.values()))
            if not all(checks.values()):
                cd['classification_reason'] += '+full_exterior_insufficient:' + ','.join(k for k,v in checks.items() if not v)
        else:
            cd['full_checks']={'evaluated':False}
            cd['expected_perimeter_continuation']=False
            cd['classification_reason'] += '+full_exterior_not_evaluated'
    apply_component_background_evidence(group_items,component_diagnostics,
                                         high_height=float(study_cfg.get('high_height_quantile',.65)))
    for row in group_items:
        full_record=full_by_component.get(row['component'],{})
        row['full_growth_evaluated']=bool(full_record.get('full_growth_evaluated',False))
        row['full_credible_reached_area']=full_record.get('full_credible_reached_area')
        row['full_credible_to_component_local_ratio']=full_record.get('full_credible_to_local_ratio')
        row['full_outer_half_growth_fraction']=full_record.get('outer_half_credible_growth_fraction')
        row['full_far_reliable_area_fraction']=full_record.get('far_reliable_area_fraction')

    missing_labels=set(labels_override)-{row['id'] for row in group_items}
    if missing_labels:
        raise ValueError(f'Unknown group label IDs: {sorted(missing_labels)[:5]}')
    group_ids=np.full(len(mesh.faces),-1,dtype=np.int32)
    label_faces={k:np.zeros(len(mesh.faces),bool) for k in ('normal','suspect','uncertain')}
    for i,(row,fi) in enumerate(zip(group_items,component_faces)):
        group_ids[fi]=i;label_faces[row['effective_label']][fi]=True
    stats={'internal_credible_area_centroid_approx':float(area[inside&credible].sum()),
           'initial_exterior_candidate_area':total_external,
           'initial_exterior_ratio':total_external/max(float(area[inside&credible].sum()),1e-12),
           'groups':len(group_items),'boundary_samples':len(traces),
           'connected_components':len(component_diagnostics),
           'expected_perimeter_components':sum(int(c['expected_perimeter_continuation'])
                                               for c in component_diagnostics.values()),
           'areas_by_label':{k:float(area[v].sum()) for k,v in label_faces.items()},
           'full_exterior':full['stats'],
           'face_centroid_area_estimate':True,
           'geometry_limitations':['Centroid-area integration is approximate near ROI boundary.',
                'Mesh edge crossing traces are samples; not guaranteed complete when both endpoints are exterior.',
                'Surface geodesics approximate face-center adjacency graph.',
                'Class labels are geometrical hypotheses and not semantic ground truth.',
                f'Local growth is evaluated for the {max_growth_groups} largest groups; complete exterior analysis is reported separately.',
                'Low-confidence original faces may carry diagnostic connectivity but never enter the trusted recovery mask.']}
    return {'sd':sd,'inside':inside,'credible':credible,'candidate':ext,
            'groups':group_items,'group_ids':group_ids,'label_masks':label_faces,
            'traces':traces,'trace_labels':trace_idx,'stats':stats,
            'component_diagnostics':list(component_diagnostics.values()),
            'full_raw_mask':full['raw_mask'],'full_credible_mask':full['credible_mask'],
            'full_component_diagnostics':full['components']}


def run(config,roi_file,output_dir, labels_file=None, *, export_ply=False,
        run_ablation=True, up_vector=None, gaussian_path=None,
        export_final=True, final_method='R3', final_only=False, progress_callback=None):
    if final_only and (not run_ablation or final_method != 'R3' or export_ply):
        raise ValueError('final_only requires R3, run_ablation, and no comparison PLY exports')
    config=Path(config).resolve();cfg=yaml.safe_load(config.read_text(encoding='utf8'))
    if not isinstance(cfg,dict):raise ValueError('Config YAML not a mapping')
    out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    roi=ROI.from_file(roi_file)
    if progress_callback: progress_callback('loading', '加载Poisson网格及完整Gaussian PLY')
    paths=cfg['paths'];meshfile=Path(paths['poisson_raw']);sfmfile=Path(paths['clean_sfm_npz'])
    plyfile=Path(gaussian_path or paths['gaussian_ply'])
    mesh=trimesh.load(meshfile,process=False)
    if not isinstance(mesh,trimesh.Trimesh):raise ValueError('Expected a triangle mesh')
    ply=GaussianPLY(plyfile);xyz=ply.positions()
    diag=float(np.linalg.norm(np.ptp(xyz,axis=0)))
    if not diag>0:raise ValueError('Empty or invalid Gaussian bounds')
    # EXPERIMENTAL SIMPLIFICATION: treat every original Poisson face as trusted.
    # Do NOT call geometry_confidence() in the default mode. This avoids hiding
    # geometry behind a heuristic SfM score while we validate topology first.
    # To restore the historical scoring experiment, explicitly set
    # roi_study.poisson_confidence_enabled: true in the scene config.
    study_cfg=cfg.get('roi_study',{}) or {}
    score_enabled=bool(study_cfg.get('poisson_confidence_enabled',False))
    conf_threshold=float(cfg['sfm']['confidence_threshold'])
    sfm=stree=h=nearest=None
    if score_enabled:
        sfm,err,tracks,fields=load_sfm(sfmfile)
        conf,h,nearest,stree,cstats=geometry_confidence(mesh,sfm,err,tracks,
             k=int(cfg['sfm']['knn_spacing']),weights=tuple(cfg['sfm']['confidence_weights']))
    else:
        conf=np.ones(len(mesh.faces),dtype=np.float64)
        # SfM spacing is needed for historical R1/R2 guards only.
        # The validated R3 two-stage algorithm uses Gaussian covariance and
        # the Poisson surface; final-only runs avoid redundant SfM queries.
        if run_ablation and not final_only:
            sfm,_,_,_=load_sfm(sfmfile)
            stree=cKDTree(sfm)
            k=min(len(sfm),max(2,int(cfg['sfm']['knn_spacing'])))
            dd,ii=stree.query(np.asarray(mesh.triangles_center),k=k,workers=-1)
            if np.asarray(dd).ndim==1:
                h=np.maximum(np.asarray(dd,dtype=float),1e-9)
                nearest=np.asarray(ii,dtype=np.int64)
            else:
                h=np.maximum(np.asarray(dd[:,-1],dtype=float),1e-9)
                nearest=np.asarray(ii[:,0],dtype=np.int64)
        cstats={'mode':'all_poisson_faces_trusted','scores_computed':False}

    up=np.array(up_vector if up_vector is not None else json.loads(Path(roi_file).read_text())
        .get('annotation_metadata',{}).get('up_vector',[]),float)
    if up.shape!=(3,) or not np.isfinite(up).all() or np.linalg.norm(up)<.99:
        raise ValueError('No valid COLMAP up_vector in ROI annotation; pass --up x,y,z explicitly')
    up=up/np.linalg.norm(up)
    overrides={}
    if labels_file:
        labeldata=json.loads(Path(labels_file).read_text(encoding='utf8'))
        overrides=labeldata.get('labels',labeldata)
        if not isinstance(overrides,dict):raise ValueError('Expected JSON mapping of group ID to label')
    # Cache raw Poisson open boundary vertex flags once.
    e=np.asarray(mesh.edges_sorted);unique,count=np.unique(e,axis=0,return_counts=True)
    flags=np.zeros(len(mesh.vertices),bool);flags[unique[count==1].reshape(-1)]=True
    mesh._roi_boundary_vertex_flags=flags
    if progress_callback: progress_callback('poisson', 'Poisson外围连通性与误截断候选识别')
    analysis=analyze(mesh,roi,conf,conf_threshold,diag,
                     cfg['guards'],up,candidate_labels=overrides,
                     study_cfg=cfg.get('roi_study',{}),
                     growth_multiple=float(cfg.get('roi_study',{}).get('growth_multiple',3.0)),
                     scales_count=int(cfg.get('roi_study',{}).get('growth_steps',6)))
    base=roi.distances(xyz);basemask=np.all(base>=-1e-9,axis=1)
    np.save(out/'keep_mask_R0.npy',basemask)
    np.savez_compressed(out/'boundary_traces.npz',xyz=analysis['traces'].astype('f4'),
                         view_edge=analysis['trace_labels'])
    np.savez_compressed(out/'full_exterior_faces.npz',
                        raw_exterior_face_indices=np.flatnonzero(analysis['full_raw_mask']).astype('i4'),
                        credible_exterior_face_indices=np.flatnonzero(analysis['full_credible_mask']).astype('i4'),
                        credible_exterior_confidence=conf[analysis['full_credible_mask']].astype('f4'))
    np.savez_compressed(out/'candidate_faces.npz',face_indices=np.flatnonzero(analysis['candidate']).astype('i4'),
                         group_indices=analysis['group_ids'][analysis['candidate']].astype('i4'),
                         confidence=conf[analysis['candidate']].astype('f4'))
    _write_csv(out/'candidate_groups.csv',[{k:(json.dumps(v) if isinstance(v,(list,dict)) else v)
                   for k,v in row.items()} for row in analysis['groups']])
    json_write(out/'candidate_groups.json',analysis['groups'])
    json_write(out/'full_exterior_components.json',analysis['full_component_diagnostics'])
    json_write(out/'background_components.json',analysis['component_diagnostics'])
    _write_csv(out/'background_components.csv',[
        {k:(json.dumps(v) if isinstance(v,(list,dict)) else v) for k,v in row.items()}
        for row in analysis['component_diagnostics']])
    # Diagnostic exports: complete ROI exterior before confidence cutoff (raw)
    # and after confidence cutoff (trusted). Neither is a Gaussian recovery mask.
    _write_faces(mesh,analysis['full_raw_mask'],out/'exterior_raw_full.ply',[180,180,180,255])
    _write_faces(mesh,analysis['full_credible_mask'],out/'exterior_credible_full.ply',[53,175,202,255])
    _write_faces(mesh,analysis['candidate'],out/'exterior_candidate_local.ply',[230,158,30,255])
    palette={'normal':[140,140,140,255],'suspect':[255,68,47,255],
             'uncertain':[255,167,35,255]}
    for label,color in palette.items():
        _write_faces(mesh,analysis['label_masks'][label],out/f'exterior_{label}.ply',color)
    if np.any(analysis['candidate']):
        merged=trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
              faces=np.asarray(mesh.faces)[analysis['candidate']],process=False)
        group_indices=analysis['group_ids'][analysis['candidate']]
        merged.visual.face_colors=np.array([palette[analysis['groups'][int(i)]['effective_label']]
                if i>=0 else palette['uncertain'] for i in group_indices],dtype=np.uint8)
        merged.remove_unreferenced_vertices()
        merged.export(out/'exterior_all_colored.ply')
    else:
        (out/'exterior_all_colored.ply').unlink(missing_ok=True)
    if len(analysis['traces']):
        cloud=trimesh.points.PointCloud(analysis['traces'],colors=[15,212,240,255]);cloud.export(out/'roi_boundary_trace.ply')
    else:
        (out/'NO_BOUNDARY_TRACE.txt').write_text('No trace samples: inspect ROI coordinate system and mesh coverage.\n')
    report={'schema':'GSSceneRefine.roi_study.v1','roi':str(Path(roi_file).resolve()),
      'input_sha256':{'roi':file_sha256(roi_file),'poisson':file_sha256(meshfile),
        'sfm':file_sha256(sfmfile),'gaussian':file_sha256(plyfile)},
      'gaussian_input_path':str(plyfile.resolve()),
      'gaussian_input_count':len(xyz),'R0_count':int(basemask.sum()),
      'orientation_up_vector':up.tolist(),'candidate':analysis['stats'],
      'poisson_face_scoring':{
          'enabled':score_enabled,
          'mode':'sfm_heuristic' if score_enabled else 'all_faces_trusted',
          'all_faces_accepted':not score_enabled,
          'scores_computed':score_enabled,
          'sfm_spacing_used_only_for_gaussian_guards':bool(not score_enabled and run_ablation and not final_only),
      },
      'classifications_are_semantic_ground_truth':False,
      'warning':'All R1-R3 outputs are UNVALIDATED geometry hypotheses; inspect in 3D before conclusions.'}
    ablate=[]
    if run_ablation:
        selected=analysis['label_masks']['suspect']
        r1=analysis['candidate']
        # R1/R2 remain unchanged historical controls; R3 is strictly two-stage.
        methods = [('R3', selected, 'local')] if final_only else [
            ('R1',r1,'global'),('R2',selected,'global'),('R3',selected,'local')]
        for name,exterior,mode in methods:
            if name == 'R3':
                if progress_callback: progress_callback('core', 'ROI内部曲面种子与局部连续恢复范围验证')
                # The existing first-pass candidate classifier is unchanged.
                # This gate limits Stage-A Gaussian recovery to local Poisson
                # extensions that genuinely attach to ROI-interior geometry.
                # The original 0.04D path cap is only an upper bound; use the
                # smaller existing outside-search scale to reject long
                # tangential continuations of otherwise short outside patches.
                geo_limit = min(float(cfg['guards']['geodesic_limit_frac']),
                                float(cfg['guards']['outside_search_frac'])) * diag
                recovery_faces, anchored_info = anchored_local_recovery_faces(
                    mesh, roi, exterior, analysis['inside'], geo_limit)
                _write_faces(mesh, recovery_faces, out/'exterior_r3_recovery_eligible.ply',
                             [35, 188, 88, 255])
                _write_faces(mesh, exterior & ~recovery_faces,
                             out/'exterior_r3_recovery_rejected.ply',
                             [228, 101, 55, 255])
                if progress_callback: progress_callback('core', '恢复有限连通Poisson曲面支持的局部主体')
                core, extra, kept, two_stage_info = recover_two_stage_from_poisson(
                    roi=roi, mesh=mesh, selected_faces=recovery_faces,
                    gaussian_ply=ply, positions=xyz, original_mask=basemask,
                    diag=diag,
                    k_neighbors=int(cfg.get('guards',{}).get('safety_knn',128)),
                    trim_fraction=0.05,
                    safety_factor=1.0, progress_callback=progress_callback)
                two_stage_info['stage_A']['anchored_local_recovery'] = anchored_info
                addition = float(np.count_nonzero(kept & ~basemask)) / max(1, int(basemask.sum()))
                allowed = float(cfg['guards']['max_guard_add_fraction'])
                two_stage_info['review_required'] = bool(addition > allowed)
                two_stage_info['added_to_baseline_fraction'] = addition
                two_stage_info['review_fraction_threshold'] = allowed
                two_stage_info['over_budget_policy'] = 'report_only_no_gaussian_dropped'
                # Keep original Module-4 public masks AND save stage-separated
                # audit masks. No new CLI, output folder or model version.
                np.save(out/'core_mask_R3.npy', core)
                np.save(out/'safety_added_mask_R3.npy', extra)
                np.save(out/'keep_mask_R3.npy', kept)
                combined_extra=kept & ~basemask
                np.save(out/'added_mask_R3.npy', combined_extra)
                if export_ply: ply.export(kept,out/'gaussians_R3_PROVISIONAL.ply')
                ablate.append({'method':'R3', 'gaussian_count':int(kept.sum()),
                    'newly_restored_gaussians':int(combined_extra.sum()),
                    'surface_diagnostics':two_stage_info['stage_A'],
                    'guard_diagnostics':two_stage_info['stage_B'],
                    'two_stage_diagnostics':two_stage_info})
                print('[roi-study] R3 recovered core:',int(np.count_nonzero(core & ~basemask)),
                      'safety additional:',int(extra.sum()),'total:',int(kept.sum()),flush=True)
                continue
            if not np.any(exterior):
                extra=np.zeros(len(xyz),bool)
                diag1={'no_selected_exterior':True}
                diag2={}
            else:
                surface,radii,diag1=local_protection(mesh,roi,ply,xyz,sfm,stree,conf,h,nearest,
                    analysis['sd'],analysis['inside'],analysis['credible'],exterior,
                    diag,cfg['guards'],mode,None)
                if name in ('R2','R3'):
                    # Restrict interior safety samples to near selected EXTERIOR
                    # patches, so unrelated perimeter doesn't trigger recovery.
                    ext_pts=np.asarray(mesh.triangles_center)[exterior]
                    nearest_ext,_=cKDTree(ext_pts).query(surface,workers=-1)
                    take=nearest_ext <= radii+float(cfg['guards']['inner_band_frac'])*diag
                    surface=surface[take];radii=radii[take]
                extra,diag2=add_guard_gaussians(xyz,basemask,surface,radii,
                    float(cfg['guards']['max_guard_add_fraction']),budget_policy='review')
            kept=basemask|extra
            np.save(out/f'keep_mask_{name}.npy',kept)
            np.save(out/f'added_mask_{name}.npy',extra)
            if export_ply:ply.export(kept,out/f'gaussians_{name}_PROVISIONAL.ply')
            ablate.append({'method':name,'gaussian_count':int(kept.sum()),
                 'newly_restored_gaussians':int(extra.sum()),
                 'surface_diagnostics':diag1,'guard_diagnostics':diag2})
    if export_ply:ply.export(basemask,out/'gaussians_R0.ply')
    report['keep_mask_sha256']={k:file_sha256(out/f'keep_mask_{k}.npy')
             for k in ((('R0','R3') if final_only else ('R0','R1','R2','R3')) if run_ablation else ('R0',))}
    report['candidate']['poisson_face_scoring_enabled']=score_enabled
    report['candidate']['all_poisson_faces_trusted']=not score_enabled
    report['ablation']=ablate
    json_write(out/'study_report.json',report)
    json_write(out/'config_snapshot.json',cfg)
    # Stage 1 / Module 4: materialize a real Gaussian PLY from validated masks.
    # The default exports ONE R3 PLY; --export-ply still provides R0--R3.
    if export_final and run_ablation:
        if progress_callback: progress_callback('exporting', '写入最终Gaussian R3 PLY并校验属性')
        from .finalize import finalize
        result=finalize(out,method=final_method,gaussian_path=plyfile,
                        report=report,source_ply=ply,verify_input_hash=False)
        report['final_gaussian_scene']=result
        json_write(out/'study_report.json',report)
    elif not run_ablation:
        report['final_gaussian_scene']=None
    print('[roi-study] Poisson face scoring:', 'ENABLED' if score_enabled else 'DISABLED (all faces trusted)',flush=True)
    print('[roi-study] ROI:',roi_file,flush=True)
    print('[roi-study] R0 retained:',int(basemask.sum()),'of',len(xyz),flush=True)
    print('[roi-study] Initial exterior area / internal:',f"{analysis['stats']['initial_exterior_ratio']:.4f}",flush=True)
    print('[roi-study] groups:',len(analysis['groups']),'classes:',analysis['stats']['areas_by_label'],flush=True)
    print('[roi-study] output:',out,flush=True)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--roi',default=None,help='Explicit ROI JSON; default is latest confirmed scene ROI')
    p.add_argument('--output-dir',default=None)
    p.add_argument('--labels',default=None,help='Optional independent manual group labels JSON')
    p.add_argument('--up',default=None,help='Override COLMAP up vector: x,y,z')
    p.add_argument('--no-ablation',action='store_true',help='Analyze complete exterior, boundary groups and classifications only; no Gaussian guard computation')
    p.add_argument('--export-ply',action='store_true',help='Also export large R0--R3 comparison PLYs')
    p.add_argument('--final-only',action='store_true',help='Production workflow: run only R0 and R3; skip R1/R2 ablations')
    p.add_argument('--no-final-export',action='store_true',help='Do not automatically export default final R3 Gaussian PLY')
    p.add_argument('--final-method',choices=('R0','R1','R2','R3'),default='R3',help='Method chosen as the final export (default R3)')
    args=p.parse_args(argv)
    roi=args.roi
    if not roi:
        from gsrefine.roi_init.settings import InitSettings
        from gsrefine.roi_init.results import current_latest
        roi=current_latest(InitSettings.load(args.config))
        if roi is None:raise FileNotFoundError('No latest ROI for this scene')
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf8'))
    out=args.output_dir or str(Path(cfg['paths'].get('output_dir','runs'))/'boundary_study')
    up=list(map(float,args.up.split(','))) if args.up else None
    run(args.config,roi,out,args.labels,export_ply=args.export_ply,
        run_ablation=not args.no_ablation,up_vector=up,
        export_final=not args.no_final_export,final_method=args.final_method,final_only=args.final_only)

if __name__=='__main__': main()
