"""Stage 2 local-sheet partition, source-backed recovery and scoped publication."""
import csv
from pathlib import Path

import numpy as np

from gsrefine.cleanup import local_partition as geom
from gsrefine.cleanup.local_voxels import RegionTree, classify
from gsrefine.cleanup.roi_barrier_parity import _prepare_geometry, _query_nearby_candidates
from gsrefine.cleanup.sdf_common import robust_gaussian_radius
from gsrefine.cleanup.stage2_runtime import file_digest, measured, snapshot, timed
from gsrefine.cleanup.threshold_distance import exact
from gsrefine.geometry import original_wall_audit as grid

FULL = 'COMPLETED_EXPERIMENTAL_CLEAN_ROI_GAUSSIAN'
PARTIAL = 'PARTIAL_VERIFIED_ROI_CLEANUP'
NONE = 'NO_VERIFIED_CLEANUP_SCOPE'


def _opening_boxes(paths,h):
    # Join nearby rim witnesses, then exclude the enclosed hole as well as its rim.
    # Each proposed restricted domain must independently pass both grid audits.
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    tips=paths[:,1]
    bins, inverse=np.unique(np.floor(tips/(2*h)).astype(int),axis=0,return_inverse=True)
    ij=cKDTree(bins).query_pairs(2.5,output_type='ndarray')
    graph=coo_matrix((np.ones(len(ij)),ij.T),shape=(len(bins),len(bins))).tocsr()
    n, groups=connected_components(graph,directed=False)
    labels=groups[inverse.ravel()]
    boxes=[]
    for i in range(n):
        xyz=paths[labels==i].reshape(-1,3)
        boxes.append([xyz.min(axis=0)-2*h,xyz.max(axis=0)+2*h])
    return np.asarray(boxes)


def _scope(field, paths, anchor, h):
    """Retain an anchored half-space excluding every localized bypass."""
    tips = paths[:,1]
    choices = np.vstack([np.eye(3), -np.eye(3), anchor-tips.mean(axis=0)])
    choices /= np.maximum(np.linalg.norm(choices,axis=1)[:,None],1e-15)
    offsets = (paths.reshape(-1,3) @ choices.T).max(axis=0) + 2*h
    clearance = choices @ anchor - offsets
    i = np.argmax(clearance)
    if clearance[i] <= 1.5*h:
        return None
    return geom.Domain(field,choices[i],offsets[i])


def _attempt(main, domain, up, anchor, lo, hi, h, folder, strict):
    from gsrefine.cleanup.stage2_onepass import _atomic_json
    try:
        return geom.audit(main, domain, up, anchor, lo, hi, h, folder, strict=strict)
    except (ValueError, RuntimeError) as error:
        # A deficient region/grid is inconclusive. File, integrity and backend
        # failures propagate, rather than being counted as legitimate abstentions.
        if not str(error).startswith(('R3 ROI too small','R3 sampling window',
                                      'fine sampled box','coarse sampled box')):
            raise
        reason = {'accepted':False, 'reason':str(error)}
        _atomic_json(folder/'local_partition_report.json',reason)
        return None, anchor, np.empty((0,3,3)), reason


@measured('local_partition_and_recovery')
def establish(data, lo, hi, h, seed, diagnostics):
    from gsrefine.cleanup.direct_r3_pipeline import _load_core_sfm, sfm_fit
    from gsrefine.cleanup.stage2_onepass import _atomic_json

    record, field = data['record'], data['field']
    domain = geom.Domain(field)
    main = data['main']
    aux = record.get('auxiliary_observed_surface')
    main_path = record['main_observed_surface']
    fit = record['sfm_fit']
    trace = []
    def attempt(mesh, scope, anchor, name, strict=True):
        print('[local-region] '+name, flush=True)
        result = _attempt(mesh, scope, data['up'], anchor, lo, hi, h,
                          diagnostics/name, strict)
        trace.append({'audit':name, 'accepted':result[0] is not None})
        return result

    index, anchor, paths, _ = attempt(main, domain, seed, 'local_audit', False)
    if index is None and not len(paths):
        # Rim witnesses have already been checked against the actual R3 and triangles.
        # A coarse graph failure alone still requires refined reconstruction.
        reference = seed if seed is not None else anchor
        index, retry_anchor, paths, _ = attempt(main, domain, reference,'refined_local_audit')
        if anchor is None:
            anchor = retry_anchor
    if index is None and len(paths):
        source = data['global_main']
        ids = geom.source_hits(source, paths)
        trace.append({'bypass_paths':len(paths),'intersected_source_faces':ids.tolist()})
        if len(ids):
            import trimesh
            try:
                candidate = geom.repair_from_source(source, ids, field)
            except ValueError as error:
                if not str(error).startswith(('Largest R3 body component is ambiguous',
                                              'R3 clipping retained no Poisson triangles')):
                    raise
                trace.append({'source_reclip_unusable':str(error)})
                candidate = None
        else:
            candidate = None
        if candidate is not None:
            trial = candidate['main']
            # Restrict repairs to the same main body. Do not silently promote a
            # different clipped component just because local subdivision added faces.
            sc, o3d = grid.make_scene(trial)
            probes = np.asarray(main.triangles_center)[::max(1,len(main.faces)//20000)]
            probes = probes[domain.interior_radius(probes)>h]
            close = exact(sc,o3d,probes) <= max(h*.01,1e-6) if len(probes) else []
            if len(close) and np.mean(close) >= .995:
                # Compare/audit the exact serialized mesh used downstream.
                trial_path = diagnostics/'source_reclipped_main.ply'
                trial.export(trial_path)
                trial = trimesh.load(trial_path, force='mesh', process=False)
                idx, _, _, _ = attempt(trial, domain, seed if seed is not None else anchor,
                                       'source_reclip_audit')
                if idx is not None:
                    xyz, _, _ = _load_core_sfm(record['sfm'])
                    trial_fit = sfm_fit(trial, xyz, max_points=0)
                    if trial_fit['p95'] <= record['sfm_fit_safeguards']['max_by_roi_diagonal']:
                        index, main, fit, main_path = idx, trial, trial_fit, trial_path
                        aux = None
                        if candidate['auxiliary'] is not None:
                            aux = diagnostics/'source_reclipped_auxiliary.ply'
                            candidate['auxiliary'].export(aux)
                        trace.append({'adopted_source_reclip':True,
                                      'source_face_ids_file':'source_repair_faces.npz'})
                        np.savez_compressed(diagnostics/'source_repair_faces.npz', ids=ids)
    scope = 'full' if index is not None else 'none'
    if index is None and len(paths) and anchor is not None:
        reference = np.asarray(seed if seed is not None else anchor)
        restricted = geom.Domain(field,boxes=_opening_boxes(paths,h))
        alternatives = [('excluded_rim_audit',restricted),
                        ('anchored_halfspace_audit',_scope(field,paths,reference,h))]
        for name, partial in alternatives:
            if partial is None or partial.membership(reference[None])[0] <= 0:
                continue
            index, _, _, _ = attempt(main,partial,reference,name)
            if index is not None:
                scope, domain = 'partial', partial
                break
    evidence = dict(scope=scope, attempts=trace, anchor=seed if seed is not None else anchor,
                    main=str(main_path), main_sha256=file_digest(main_path),
                    auxiliary=str(aux) if aux else None,
                    auxiliary_sha256=file_digest(aux) if aux else None,
                    sfm_p95=fit['p95'], sfm_core_count=fit['source_core_sfm_count'],
                    plane_normal=domain.normal, plane_offset=domain.offset,
                    excluded_centers=domain.holes.data if domain.holes is not None else [],
                    exclusion_radius=domain.exclusion_radius, excluded_boxes=domain.boxes)
    _atomic_json(diagnostics/'scope.json', evidence)
    return index, main, aux, fit, domain, evidence


def run(stage1, preseal, out, model, *, seed, chunk, nearest_k, min_agree,
        protect, previews, aux_preview, partition_report=None, original_mode=False, aux_ablation=False):
    from gsrefine.cleanup.stage2_onepass import _atomic_json, _export_native, _auxiliary_protection
    from gsrefine.cleanup.stage2_geometry_validation import GAUSSIAN_DISTANCE_DEFAULTS as defaults
    from gsrefine.roi.core import ROI

    diagnostics = out/'diagnostics'
    data = _prepare_geometry(stage1, preseal, with_interior_radius=True, local_sheet_only=True,
                             defer_global_component=original_mode or partition_report is not None)
    inputs = data['inputs']
    points = np.asarray(model.positions(),float)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError('Empty or nonfinite Gaussian centers')
    original = original_mode or partition_report is not None
    if original:
        from gsrefine.cleanup.original_partition import load
        if partition_report is not None:
            index, wall, domain, scope = load(data, partition_report, seed)
            lo = index['origin']
            hi = lo + np.asarray(index['shape'])*index['voxel']
        else:
            from gsrefine.geometry.original_wall_source import source_groups
            wall, _, sizes = source_groups(data['orig_global'])
            domain = geom.Domain(data['field'])
            h = float(np.ptp(points, axis=0).max()/60.)
            if h <= 0:
                raise ValueError('Degenerate R3 extent')
            lo, hi, _ = grid.bounding_window(ROI.from_file(inputs['roi_json']),points,3*h)
            lo, hi, win = grid.complete_window_for_levels(domain,lo,hi,
                levels=[('coarse',h,250000), ('fine',.72*h,640000)])
            _atomic_json(diagnostics/'window.json',win)
            folder = diagnostics/'original_A_audit'
            if len(sizes)>1 and sizes[0]==sizes[1]:
                index, anchor, evidence = None, seed, {'reason':'ambiguous largest source component'}
            else:
                index, anchor, _, evidence = geom.audit(wall,domain,data['up'],seed,lo,hi,h,
                    folder,strict=True,original_wall=True)
            scope = dict(scope='full' if index is not None else 'none', group='A_largest',
                main=str(inputs['poisson_mesh']), faces=len(wall.faces), anchor=anchor,
                distance_surface=data['record']['main_observed_surface'],historical_ab_seconds=None,
                assessment=str(folder), accepted=index is not None)
            if index is None:
                scope['reason'] = evidence.get('reason',evidence.get('verdict',{}).get('blocking_reasons'))
        scope['partition_reused'] = partition_report is not None
        mesh = data['main']
        aux = data['record'].get('auxiliary_observed_surface')
        fit = data['record']['sfm_fit']
        _atomic_json(diagnostics/'scope.json', scope)
        print(f"[original-region] v7 original wall: {scope['faces']:,} triangles; "
              f"partition={scope['scope']}, reused={scope['partition_reused']}", flush=True)
    else:
        h0 = float(np.ptp(points,axis=0).max()/60.)
        if h0 <= 0:
            raise ValueError('Degenerate R3 extent')
        lo, hi, _ = grid.bounding_window(ROI.from_file(inputs['roi_json']),points,3*h0)
        lo, hi, win = grid.complete_window_for_levels(data['field'], lo, hi,
            levels=[('coarse',h0*1.6,95000),('fine',h0,250000),('retry',h0*.72,640000)])
        _atomic_json(diagnostics/'window.json',win)
        index, mesh, aux, fit, domain, scope = establish(data,lo,hi,h0,seed,diagnostics)
        wall = mesh
    radii = measured('gaussian_radius_statistics')(robust_gaussian_radius)(model, defaults['trim_fraction'], defaults['sigma_multiplier'])
    delta = float(fit['p95']+radii['trimmed_mean_radius'])
    tau = delta+defaults['epsilon']
    info = {}
    if index is not None:
        scene, o3d = grid.make_scene(mesh)
        wall_scene = grid.make_scene(wall)[0] if original else scene
        def distance(p):
            return exact(scene,o3d,p)
        def wall_distance(p):
            return exact(wall_scene,o3d,p)
        def rays(p):
            return _query_nearby_candidates(p,index,domain,wall_scene,o3d,wall.face_normals,
                neighbours=nearest_k,min_agree=min_agree,batch=max(500,chunk//nearest_k),
                roi_link_samples=7)
        guard = 32*np.finfo(np.float32).eps*max(1.,abs(lo).max(),abs(hi).max(),
                                                  np.abs(mesh.bounds).max(), np.abs(wall.bounds).max())
        tree = measured('region_tree_build')(RegionTree)(index,domain,distance,tau,guard,
                          wall_distance=wall_distance if original else None)
        if original and tree.conflicts:
            info['initial_voxel_conflicts'] = tree.conflicts
            tree = measured('region_tree_conflict_rebuild')(RegionTree)(index,domain,distance,tau,guard,wall_distance=wall_distance,
                              refine_keys=tree.conflict_keys)
            info['unresolved_voxel_conflicts'] = tree.conflicts
        if tree.conflicts and not original:
            original_conflicts = tree.conflicts
            index, _, _, _ = _attempt(mesh,domain,data['up'],scope['anchor'],lo,hi,h0*.8,
                                      diagnostics/'voxel_conflict_refinement',True)
            if index is not None:
                tree = RegionTree(index,domain,distance,tau,guard)
                if tree.conflicts:
                    index = None
            info['initial_voxel_conflicts'] = original_conflicts
        if index is not None:
            cover = None
            if original:
                from gsrefine.cleanup.region_cover import RegionCover

                def radius(p):
                    return np.maximum(0., np.minimum(domain.interior_radius(p), wall_distance(p)) - guard)

                cover = measured('region_cover_build')(RegionCover)(index, tree, radius)
            outside, inside, withheld, lookup = classify(
                points, tree, domain, distance, tau, rays,
                batch=chunk, region_cover=cover)
            if original:
                info['classifier_revision'] = 'original_region_residual_v7'
                print(f"[original-region] Gaussian short rays={lookup['short_ray_queries']}; "
                      f"extra region lookups={lookup['cover_queries']:,}, "
                      f"resolved={lookup['cover_resolved']:,}, "
                      f"unverified={int(withheld.sum()):,}", flush=True)
            info.update(lookup)
        info['tree'] = tree.summary
    if index is None:
        outside = np.zeros(len(points),bool)
        inside = outside.copy()
        withheld = ~outside
    protected = np.zeros(len(points),bool)
    shadow_protected = None
    if aux_ablation and not protect and outside.any() and aux:
        shadow_protected, _, _ = _auxiliary_protection(model,outside,delta,aux,chunk_size=chunk)
    if protect and outside.any() and aux:
        protected, _, _ = _auxiliary_protection(model,outside,delta,aux,chunk_size=chunk)
    remove = outside & ~protected
    keep = ~remove
    if (remove & withheld).any() or (outside & inside).any():
        raise RuntimeError('Invalid local region mask')
    status = FULL
    if index is None or withheld.all():
        status = NONE
    elif scope['scope']=='partial' or withheld.any():
        status = PARTIAL
    target = out/('gaussian_roi_clean.ply' if status==FULL else 'gaussian_roi_partial.ply')
    # No-verified-scope: refer directly to the original PLY; no false clean output.
    if status == NONE:
        target = inputs['gaussian_ply']
    else:
        _export_native(model,keep,target)
    masks = diagnostics/'final_classification_masks.npz'
    np.savez_compressed(masks,keep=keep,remove=remove,protected_by_aux=protected,
                        original_parity_outside=outside,unverified_scope=withheld,
                        known_inside=inside,delta=delta)
    if previews:
        for name, mask in [('outside_candidates',outside),('inside_labelled',inside),('unverified',withheld)]:
            if mask.any():
                _export_native(model,mask,diagnostics/(name+'.ply'))
    # Read-only deleted-Gaussian risk, exterior seed and cross-stage attribution.
    from gsrefine.cleanup.engineering_audit import cross_stage, classification_risk
    import json as _json
    with timed('engineering_cross_stage_and_risk_audit'):
        engineering = {
            'cross_stage': cross_stage(Path(stage1).parent.parent, inputs['gaussian_ply'],
                                      remove, input_count=len(points)),
            'risk': classification_risk(points=points, remove=remove, outside=outside,
                        withheld=withheld, protected=protected, delta=delta, scope=scope),
            'auxiliary_ablation': dict(
                        status=('COUNTERFACTUAL_SAME_CLASSIFICATION' if protect or shadow_protected is not None
                                else 'NOT_COMPUTED_NO_AUXILIARY_RUN'),
                        protection_enabled=bool(protect),
                        removed_if_auxiliary_off=int(outside.sum()),
                        removed_with_protection=(int(outside.sum()-(protected if protect else shadow_protected).sum())
                                if protect or shadow_protected is not None else None),
                        extra_removed_without_protection=(int((protected if protect else shadow_protected).sum())
                                if protect or shadow_protected is not None else None),
                        note='Counterfactual reuses same side and distance classifications; does not edit final mask.'),
        }
        _atomic_json(diagnostics/'engineering_audit.json',engineering)
    if aux_preview and protected.any():
        _export_native(model,protected,diagnostics/'aux_protected_gaussians.ply')
    counts = dict(input_gaussians=len(points), outside_before_aux=int(outside.sum()),
                  protected_by_all_aux=int(protected.sum()),final_removed=int(remove.sum()),
                  final_retained=int(keep.sum()),unverified_retained=int(withheld.sum()))
    result = dict(schema='GSSceneRefine.local_region.v1',status=status,engineering_audit=str(diagnostics/'engineering_audit.json'),quality_review_required=bool(withheld.any() or seed is None),
        input=dict(stage1_manifest=str(stage1),stage1_manifest_sha256=file_digest(stage1),
                   original_roi_gaussian=str(inputs['gaussian_ply']),
                   original_roi_gaussian_sha256=file_digest(inputs['gaussian_ply']),
                   original_global_poisson=str(inputs['poisson_mesh']),
                   original_global_poisson_sha256=file_digest(inputs['poisson_mesh']),
                   preseal_report=str(preseal),preseal_report_sha256=file_digest(preseal)),
        method=dict(spatial_classifier='original-region' if original else 'local-region',space_wall=scope['main'],
                    distance_surface=scope.get('distance_surface',scope['main']),aux_enabled=protect,
                    exterior_mode='user_coordinate' if seed is not None else 'accepted_highest_shell'),
        geometry=scope,threshold=dict(delta=delta,epsilon=defaults['epsilon'],sfm_p95=fit['p95'],
                    gaussian_trimmed_mean_radius=radii['trimmed_mean_radius']),
        counts=counts,lookup=info,outputs=dict(final_supersplat_gaussian_ply=str(target),
                    final_mask=str(masks),final_mask_sha256=file_digest(masks),final_ply_sha256=file_digest(target)))
    timing = snapshot()
    result['stage2_timing'] = timing
    with (diagnostics/'stage2_timing.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['项目','次数','含子步骤秒数','独占秒数'])
        writer.writerow(['阶段二主程序至汇总',1,timing['total_s'],''])
        writer.writerow(['未归类开销',1,'',timing['other_s']])
        for name,item in timing['steps'].items():
            writer.writerow([name,item['calls'],item['inclusive_s'],item['exclusive_s']])
    _atomic_json(out/'stage2_final_report.json',result)
    print('[local-region]', status, counts, flush=True)
    return result
