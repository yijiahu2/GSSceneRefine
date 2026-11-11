"""Stage-2 experimental A+B pruning on ONLY the verified open R3 MAIN surface.

A: exact unsigned point-to-triangle distance with the existing full-SfM P95
   plus global P10--P90 trimmed Gaussian long-axis 3sigma scale threshold.
B: generalized winding number and multiple geometrically unoccluded COLMAP
   camera directions. This is an observationally constrained heuristic for an
   OPEN surface, not a mathematically certified inside/outside solid.

No sealing, no original vertex motion, no auxiliary geometry / support, no
implicit addition to official Stage-2 final. Writes only --output-dir.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import tempfile

import numpy as np

ALGORITHM = 'open_ab_main_20261110_v2_camera_relative'
DEFAULTS = {
    'chunk_size': 30000,
    'winding_camera_max_delta': 0.20,
    'winding_inside_min': 0.85,
    'camera_candidates': 12,
    'min_unobstructed_cameras': 2,
    'camera_forward_dot_min': 0.20,
    'trim_fraction': 0.10,
    'sigma_multiplier': 3.0,
    'epsilon': 1e-5,
}


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as src:
        for block in iter(lambda:src.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def _write_json_atomic(path, data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name,suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf8') as f:
            json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.write('\n'); f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def classify_main_open_ab(distances, winding, visible_camera_votes, delta, *,
                          winding_outside_max=0.10,winding_inside_min=0.85,
                          min_unobstructed_cameras=2,epsilon=1e-5,
                          outside_mode='absolute'):
    """Pure-Numpy conservative classifier. No auxiliary mesh is inspected.

    Winding is evaluated only for far candidates; NaN for near-surface is OK.
    OUTSIDE is an *operational* classifier, never ground-truth geometry.
    outside_mode='camera_relative': camera votes already require winding
    agreement with the respective visible outside-reference camera.
    """
    d=np.asarray(distances,dtype=np.float64).reshape(-1)
    w=np.asarray(winding,dtype=np.float64).reshape(-1)
    votes=np.asarray(visible_camera_votes,dtype=np.int64).reshape(-1)
    if not (len(d)==len(w)==len(votes)):
        raise ValueError('Mismatched Gaussian classification array lengths')
    if not (np.isfinite(delta) and delta>=0 and np.isfinite(epsilon) and epsilon>=0):
        raise ValueError('Invalid distance threshold')
    if not (0 < winding_outside_max < winding_inside_min < 1.5):
        raise ValueError('Invalid open-surface winding confidence bands')
    if min_unobstructed_cameras<1:
        raise ValueError('Need at least one independent camera vote')
    far=np.isfinite(d) & (d>delta+epsilon)
    wfinite=np.isfinite(w)
    if outside_mode=='camera_relative':
        # Cameras are outside OBSERVED geometry, but an open sheet can have
        # arbitrarily nonzero signed winding at these outside locations.
        # A camera vote has already been filtered by camera-relative winding.
        outside=far & wfinite & (votes>=min_unobstructed_cameras)
    elif outside_mode=='absolute':
        # Retained for historical classifier unit tests / ablations only.
        outside=far & wfinite & (np.abs(w)<=winding_outside_max) & (votes>=min_unobstructed_cameras)
    else:
        raise ValueError('Unknown exterior classifier mode')
    inside=far & wfinite & (np.abs(w)>=winding_inside_min) & ~outside
    # Uncertain includes ambiguous far points; those are NEVER automatically deleted.
    uncertain=far & ~inside & ~outside
    near_or_nonfinite=~far
    keep=~outside
    return dict(keep=keep,remove=outside,inside=inside,uncertain=uncertain,
                near_or_nonfinite=near_or_nonfinite,far=far)


def count_unobstructed_camera_rays(points,camera_xyz,camera_forward,scene,*,
                                   o3d,k=12,forward_dot_min=.2, margin=1e-5,
                                   tree=None,camera_indices=None,anchor_mask=None):
    """Geometric visibility against main Poisson, NOT proof of empty rays.

    Query each candidate against the nearest k cameras; only cameras pointing
    toward the point are counted. An optional per-candidate anchor_mask
    rejects cameras whose signed winding differs too far from the Gaussian.
    A ray is accepted if NO observed main Poisson triangle lies before the
    Gaussian center. An open Poisson cannot certify unseen occluders.
    """
    from scipy.spatial import cKDTree
    points=np.asarray(points,dtype=np.float64)
    camera_xyz=np.asarray(camera_xyz,dtype=np.float64)
    camera_forward=np.asarray(camera_forward,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=3:
        raise ValueError('Query points must be Nx3')
    if camera_xyz.ndim!=2 or camera_xyz.shape[1]!=3 or camera_forward.shape!=camera_xyz.shape:
        raise ValueError('Camera pose arrays invalid')
    if not np.isfinite(points).all() or not np.isfinite(camera_xyz).all() or not np.isfinite(camera_forward).all():
        raise ValueError('Nonfinite camera poses / points')
    if not len(camera_xyz) or not 1<=k<=len(camera_xyz):
        raise ValueError('Invalid k cameras')
    if not -1<=forward_dot_min<=1:
        raise ValueError('Invalid camera optical dot threshold')
    if not len(points): return np.empty(0,dtype=np.uint8)
    if camera_indices is None:
        if tree is None:tree=cKDTree(camera_xyz)
        _,indices=tree.query(points,k=k,workers=-1)
        indices=np.asarray(indices,dtype=np.int64).reshape(-1,k)
    else:
        indices=np.asarray(camera_indices,dtype=np.int64)
        if indices.shape!=(len(points),k) or np.any(indices<0) or np.any(indices>=len(camera_xyz)):
            raise ValueError('Invalid per-candidate camera indices')
    origin=camera_xyz[indices]
    vector=points[:,None,:]-origin
    length=np.linalg.norm(vector,axis=-1)
    good=length>1e-9
    direction=vector/np.maximum(length[...,None],1e-9)
    forward=camera_forward[indices]
    f_norm=np.linalg.norm(forward,axis=-1)
    good &= (f_norm>1e-8)&(np.sum(direction*forward,axis=-1)/np.maximum(f_norm,1e-9)>=forward_dot_min)
    if anchor_mask is not None:
        anchor_mask=np.asarray(anchor_mask,dtype=bool)
        if anchor_mask.shape!=(len(points),k):
            raise ValueError('Invalid per-candidate camera winding anchor mask')
        good &= anchor_mask
    flat_indices=np.flatnonzero(good.ravel())
    votes=np.zeros(len(points)*k,dtype=np.uint8)
    if len(flat_indices):
        rays=np.concatenate([origin.reshape(-1,3)[flat_indices],
                             direction.reshape(-1,3)[flat_indices]],axis=1)
        out=scene.cast_rays(o3d.core.Tensor(np.ascontiguousarray(rays,np.float32)))
        hit=np.asarray(out['t_hit'].numpy(),dtype=np.float64).reshape(-1)
        end=length.ravel()[flat_indices]
        ok=np.isfinite(hit)&(hit>=end-margin)
        ok |= np.isinf(hit)
        votes[flat_indices]=ok.astype(np.uint8)
    return votes.reshape(-1,k).sum(axis=1,dtype=np.uint8)


def _validated_inputs(manifest,preseal_report,*,colmap_images=None,config_file=None):
    """Read strict immutable Stage-1 and Stage-2 main-surface provenance."""
    from gsrefine.cleanup.sdf_common import _load_stage1_manifest
    from gsrefine.cleanup.direct_r3_pipeline import (
        select_direct_r3_clip,validate_sources,_load_camera,boundary_diagnostics)
    import trimesh
    stage1,upstream,inputs,run_dir=_load_stage1_manifest(manifest)
    run_dir=Path(run_dir).resolve()
    preseal=Path(preseal_report).resolve(strict=True)
    if not preseal.is_relative_to(run_dir/'cleanup/geometry/direct_r3_sdf'):
        raise ValueError('Preseal report must belong to this ROI run')
    record=json.loads(preseal.read_text(encoding='utf8'))
    if record.get('schema')!='GSSceneRefine.direct_r3_stage2.v1':
        raise ValueError('Not a GSSceneRefine authenticated R3 preseal report')
    if Path(record.get('stage1_manifest','')).resolve()!=stage1:
        raise ValueError('Preseal provenance does not match Stage-1 manifest')
    if not record.get('sfm_fit_safeguards_passed') or not record.get('preseal_eligible'):
        raise ValueError('Preseal main surface did not pass existing mesh/SfM safeguards')
    fit=record.get('sfm_fit',{})
    if (not fit.get('is_full_core_fit') or
        int(fit.get('fit_sfm_count',0))!=int(fit.get('source_core_sfm_count',-1)) or
        int(fit.get('fit_sfm_count',0))<100 or
        not np.isfinite(float(fit.get('p95',float('nan'))))):
        raise ValueError('A complete, finite core-SfM P95 fit is required')
    mesh_file=Path(record['main_observed_surface']).resolve(strict=True)
    if not mesh_file.is_relative_to(preseal.parent) or mesh_file.name!='poisson_roi_main.ply':
        raise ValueError('Invalid main surface in preseal report')
    if _sha(mesh_file)!=record.get('main_observed_surface_sha256'):
        raise ValueError('Main observed Poisson surface hash mismatch')
    raw,p,clip=select_direct_r3_clip(run_dir,selected_report=record['direct_r3_clip_report'])
    if raw!=Path(record['observed_direct_r3_mesh']).resolve():
        raise ValueError('Direct R3 original mesh does not match preseal')
    # Original COLMAP, R3 and Poisson immutable provenance already enforced.
    sfm=Path(record['sfm']).resolve(strict=True)
    validate_sources(stage1,upstream,inputs,run_dir,clip,sfm)
    mesh=trimesh.load(mesh_file,process=False,force='mesh')
    topo=boundary_diagnostics(mesh)
    if (topo['mesh_faces']!=record['topology']['mesh_faces'] or
        topo['nonmanifold_edges']!=0 or topo['face_components']!=1 or
        topo['open_boundary_components']!=1 or
        not topo['boundary_component_details'][0]['is_simple_closed_ring'] or
        not topo['winding_consistent']):
        raise ValueError('Main surface changed / invalid; preflight must be rerun')
    if int(record.get('topology_partition',{}).get('auxiliary_component_count',-1))<0:
        raise ValueError('Missing main/aux partition provenance')
    cfg=Path(config_file).resolve(strict=True) if config_file else (run_dir/'metadata/effective_scene.yaml').resolve(strict=True)
    positions,_,camera_path=_load_camera(run_dir,inputs,cfg,colmap_images)
    from gsrefine.geometry.orientation import read_colmap_poses
    centers,rights,forwards=read_colmap_poses(camera_path)
    if len(centers)!=len(positions) or not np.allclose(centers,positions,atol=1e-8,rtol=0):
        raise ValueError('Camera pose disagreement')
    return record, mesh, inputs, run_dir, cfg, camera_path, centers, forwards, preseal


def run_open_ab_main(*,manifest,preseal_report,output_dir,chunk_size=30000,
                     colmap_images=None,config_file=None,
                     winding_camera_max_delta=.20,winding_inside_min=.85,
                     camera_candidates=12,min_unobstructed_cameras=2,
                     camera_forward_dot_min=.2,write_clean_ply=True):
    """Execute one fully verified, preview-only experiment. Never write Stage-1 or Stage-2 official outputs."""
    from gsrefine.cleanup.sdf_common import robust_gaussian_radius,_raycast
    from gsrefine.roi.core import GaussianPLY
    import open3d as o3d
    try:
        import igl
    except ImportError as e:
        raise RuntimeError('Missing libigl (generalized winding number). In gsrefine environment run: python -m pip install libigl') from e
    if not hasattr(igl,'fast_winding_number'):
        raise RuntimeError('Installed igl has no fast_winding_number mesh query API')
    if chunk_size<1 or camera_candidates<min_unobstructed_cameras or min_unobstructed_cameras<1:
        raise ValueError('Invalid Open A+B chunk/camera options')
    if not (0<winding_camera_max_delta<=0.5 and 0<winding_inside_min<1.5):
        raise ValueError('Invalid camera-relative winding tolerance')
    record,main,inputs,run_dir,cfg,camera_path,centers,forwards,preseal=_validated_inputs(
        manifest,preseal_report,colmap_images=colmap_images,config_file=config_file)
    output=Path(output_dir).expanduser().resolve()
    # Never put experimental results in source code, stage1 originals or final output.
    if output==run_dir or output==run_dir/'final' or output==run_dir/'cleanup/final':
        raise ValueError('Use a dedicated output directory, not a final / Stage-1 output folder')
    out_clean=output/'gaussian_roi_clean.ply'
    out_report=output/'open_ab_main_report.json'
    out_mask=output/'classification_masks.npz'
    if any(x.exists() for x in (out_clean,out_report,out_mask)):
        raise FileExistsError('Experiment output exists: select a new --output-dir to avoid overwriting prior results')
    model=GaussianPLY(inputs['gaussian_ply'])
    centers_g=np.asarray(model.positions(),dtype=np.float64)
    if centers_g.ndim!=2 or centers_g.shape[1]!=3 or not np.isfinite(centers_g).all():
        raise ValueError('Invalid Gaussian centers')
    radius=robust_gaussian_radius(model,float(DEFAULTS['trim_fraction']),float(DEFAULTS['sigma_multiplier']))
    p95=float(record['sfm_fit']['p95'])
    delta=p95+float(radius['trimmed_mean_radius'])
    eps=float(DEFAULTS['epsilon'])
    scene=_raycast(o3d,main)
    faces=np.asarray(main.faces,dtype=np.int64)
    verts=np.asarray(main.vertices,dtype=np.float64)
    camera_k=min(int(camera_candidates),len(centers))
    if camera_k<min_unobstructed_cameras:
        raise ValueError('Not enough COLMAP cameras for required multiple-view votes')
    # A baseline to reveal inconsistent orientation / incomplete-camera geometry.
    camera_w=np.asarray(igl.fast_winding_number(verts,faces,np.ascontiguousarray(centers,np.float64)),dtype=np.float64).ravel()
    if camera_w.shape!=(len(centers),) or not np.isfinite(camera_w).all():
        raise ValueError('Camera generalized winding numbers missing or nonfinite')
    # Open surfaces have nonzero solid angles even at genuinely outside cameras.
    # Record their winding values as *local exterior anchors*, never impose
    # the invalid requirement that >=50% have |w| <= 0.10.
    camera_ref={
        'signed_winding_p10':float(np.quantile(camera_w,.10)),
        'signed_winding_p50':float(np.median(camera_w)),
        'signed_winding_p90':float(np.quantile(camera_w,.90)),
        'absolute_winding_p50':float(np.median(np.abs(camera_w))),
        'absolute_winding_p90':float(np.quantile(np.abs(camera_w),.90)),
        'fraction_abs_winding_below_legacy_0p10':float(np.mean(np.abs(camera_w)<=.10)),
        'assumption':'Each COLMAP camera is an exterior reference for the observed main surface; no near-zero open-surface winding assumption.',
    }
    from scipy.spatial import cKDTree
    cam_tree=cKDTree(centers)
    n=len(centers_g)
    distances=np.empty(n,dtype=np.float32)
    winding=np.full(n,np.nan,dtype=np.float32)
    cam_votes=np.zeros(n,dtype=np.uint8)
    candidate_anchor_pass=0
    candidate_ray_tested=0
    for start in range(0,n,int(chunk_size)):
        end=min(n,start+int(chunk_size))
        xyz=np.ascontiguousarray(centers_g[start:end],np.float32)
        query=o3d.core.Tensor(xyz)
        ds=np.asarray(scene.compute_distance(query).numpy(),dtype=np.float32).ravel()
        if len(ds)!=len(xyz):
            raise RuntimeError('Open3D distance output size mismatch')
        distances[start:end]=ds
        distant=np.flatnonzero(np.isfinite(ds)&(ds>delta+eps))
        if len(distant):
            candidates=np.ascontiguousarray(centers_g[start:end][distant],np.float64)
            winds=np.asarray(igl.fast_winding_number(verts,faces,candidates),dtype=np.float64).ravel()
            if winds.shape!=(len(distant),) or not np.isfinite(winds).all():
                raise RuntimeError('Generalized winding number output invalid')
            winding[start+distant]=winds.astype(np.float32)
            # For OPEN surfaces the winding at an exterior camera need not be
            # near zero. Compare signed winding at the Gaussian to that at each
            # nearby camera. Only cameras with a compatible local winding
            # baseline can vote for outside; additionally their direct rays
            # must point towards the Gaussian and avoid the observed surface.
            _,candidate_camera_indices=cam_tree.query(candidates,k=camera_k,workers=-1)
            candidate_camera_indices=np.asarray(candidate_camera_indices,dtype=np.int64).reshape(-1,camera_k)
            anchor_mask=(np.abs(winds[:,None]-camera_w[candidate_camera_indices])
                         <=winding_camera_max_delta)
            anchor_enough=np.count_nonzero(anchor_mask,axis=1)>=min_unobstructed_cameras
            candidate_anchor_pass+=int(np.count_nonzero(anchor_enough))
            if np.any(anchor_enough):
                candidate_ray_tested+=int(np.count_nonzero(anchor_enough))
                votes=count_unobstructed_camera_rays(candidates[anchor_enough],centers,forwards,scene,o3d=o3d,
                         k=camera_k,forward_dot_min=camera_forward_dot_min,
                         margin=eps,tree=cam_tree,
                         camera_indices=candidate_camera_indices[anchor_enough],
                         anchor_mask=anchor_mask[anchor_enough])
                cam_votes[start+distant[anchor_enough]]=votes
        if start==0 or end==n or (start//int(chunk_size))%10==0:
            print(f'[stage2-open-ab-main] Gaussian examined {end:,}/{n:,}; far in batch {len(distant):,}',flush=True)
    labels=classify_main_open_ab(distances,winding,cam_votes,delta,
        winding_inside_min=winding_inside_min,
        min_unobstructed_cameras=min_unobstructed_cameras,epsilon=eps,
        outside_mode='camera_relative')
    keep=labels['keep']
    # Report generated before writing. Any failures leave the original PLY unchanged.
    report={
        'schema':'GSSceneRefine.stage2_open_ab_main_experiment.v1',
        'status':'EXPERIMENTAL_OPEN_SURFACE_CLASSIFICATION_NOT_CERTIFIED_SOLID',
        'algorithm':ALGORITHM,'created_utc':datetime.now(timezone.utc).isoformat(),
        'stage1_manifest':str(Path(manifest).resolve()),'preseal_report':str(preseal),
        'main_observed_surface':record['main_observed_surface'],
        'main_sha256':_sha(record['main_observed_surface']),
        'source_gaussian':str(inputs['gaussian_ply']),
        'source_gaussian_sha256':_sha(inputs['gaussian_ply']),
        'colmap_pose_file':str(camera_path),'colmap_pose_sha256':_sha(camera_path),
        'main_triangles':int(len(main.faces)),
        'auxiliary_component_count_ignored':int(record['topology_partition']['auxiliary_component_count']),
        'auxiliary_used_for_distance_or_classification':False,
        'sfm_fit':record['sfm_fit'],'p95':p95,'gaussian_scale':radius,
        'delta':float(delta),'epsilon':eps,
        'threshold_formula':'delta = full_core_SfM_P95_to_original_main + trimmed_mean(3 * exp(max(log_scales)))',
        'distance_metric':'exact_unsigned_distance_to_observed_main_triangles_open3d',
        'winding_metric':'igl.fast_winding_number_on_open_oriented_main; compare SIGNED winding to local exterior camera reference, not to zero',
        'camera_evidence':'multiple pose-forward-consistent, winding-anchor-compatible and Poisson-unobstructed rays to Gaussian center; not proof of empty unobserved space',
        'camera_winding_baseline':camera_ref,
        'params':{'chunk_size':int(chunk_size),'winding_camera_max_delta':float(winding_camera_max_delta),
                  'winding_inside_min':float(winding_inside_min),'camera_candidates':camera_k,
                  'min_unobstructed_cameras':int(min_unobstructed_cameras),
                  'camera_forward_dot_min':float(camera_forward_dot_min)},
        'counts':{'camera_winding_anchor_pass_candidates':int(candidate_anchor_pass),
                  'camera_ray_tested_candidates':int(candidate_ray_tested),
                  'input':int(n),'within_threshold_or_unknown_distance':int(labels['near_or_nonfinite'].sum()),
                  'over_threshold':int(labels['far'].sum()),
                  'over_threshold_inside_candidate_keep':int(labels['inside'].sum()),
                  'over_threshold_uncertain_keep':int(labels['uncertain'].sum()),
                  'over_threshold_outside_direct_remove':int(labels['remove'].sum()),
                  'kept':int(keep.sum()),'removed':int((~keep).sum()),
                  'removed_fraction':float((~keep).mean())},
        'rule':'Remove exactly when d > delta+epsilon AND >=N forward-looking visible cameras have signed winding within tolerance of query winding AND camera-to-query ray has no observed Poisson hit. No anomaly classifier.',
        'limitations':'Open surface does not define unique solid; camera rays may traverse unobserved empty gaps, and winding can vary within one exterior region. Camera-relative agreement is a conservative heuristic, not independently certified ground truth. All 19 auxiliary components are ignored by experimental request; some real auxiliary structures may be pruned.',
        'output_clean_gaussian':str(out_clean) if write_clean_ply else None,
        'classification_masks':str(out_mask),
    }
    output.mkdir(parents=True,exist_ok=True)
    # Save all classification data in compressed form; Gaussian record attributes unchanged.
    np.savez_compressed(out_mask, keep=keep,remove=labels['remove'],
            over_threshold=labels['far'],inside_candidate=labels['inside'],
            unknown_kept=labels['uncertain'],distance_main=distances,
            winding=winding,visible_camera_votes=cam_votes)
    if write_clean_ply:
        temp=out_clean.with_name('.'+out_clean.stem+'.writing.tmp.ply')
        try:
            written=model.export(keep,temp)
            if int(written)!=int(keep.sum()):
                raise RuntimeError('GaussianPLY.export wrote unexpected Gaussian count')
            written_model=GaussianPLY(temp)
            if written_model.vertex_count!=written or written_model.dtype!=model.dtype:
                raise RuntimeError('Gaussian PLY schema mismatch after export')
            if written:
                probe=np.unique(np.linspace(0,written-1,min(32,written),dtype=np.int64))
                orig_index=np.flatnonzero(keep)[probe]
                if written_model.arr[probe].tobytes()!=model.arr[orig_index].tobytes():
                    raise RuntimeError('PLY attributes failed byte-preservation check')
            del written_model
            os.replace(temp,out_clean)
        finally:
            temp.unlink(missing_ok=True)
        report['output_clean_gaussian_sha256']=_sha(out_clean)
        report['output_clean_gaussian_bytes']=out_clean.stat().st_size
    _write_json_atomic(out_report,report)
    return {'status':report['status'],'report':str(out_report),
            'clean_gaussian':str(out_clean) if write_clean_ply else None,
            'masks':str(out_mask),
            'original_gaussians':int(n),'removed':int((~keep).sum()),
            'kept':int(keep.sum()),'over_threshold':int(labels['far'].sum()),
            'outside_direct_remove':int(labels['remove'].sum()),
            'uncertain_over_threshold_keep':int(labels['uncertain'].sum()),
            'distance_threshold_delta':float(delta)}


# ---------------- V2: global implicit occupancy, not camera voting --------------
# Keep run_open_ab_camera_legacy accessible for an explicit controlled ablation.
run_open_ab_camera_legacy = run_open_ab_main


def classify_implicit_signed_proxy(distances, occupancy, delta, epsilon=1.e-5):
    """Only confident exterior + exact distance outside threshold can be removed."""
    from gsrefine.cleanup.implicit_space import OUTSIDE, INSIDE, UNKNOWN
    d=np.asarray(distances,dtype=np.float64).reshape(-1)
    c=np.asarray(occupancy,dtype=np.int8).reshape(-1)
    if len(d)!=len(c):raise ValueError('Occupancy/distance row order mismatch')
    if not(np.isfinite(delta) and delta>=0 and np.isfinite(epsilon) and epsilon>=0):
        raise ValueError('Invalid threshold')
    if not np.isin(c,[OUTSIDE,INSIDE,UNKNOWN]).all():
        raise ValueError('Unexpected occupancy label')
    far=np.isfinite(d)&(d>delta+epsilon)
    remove=far&(c==OUTSIDE)
    signed=np.full(len(d),np.nan,dtype=np.float32)
    known=(c==OUTSIDE)|(c==INSIDE)
    signed[known]=d[known]*np.where(c[known]==OUTSIDE,1.,-1.)
    return {'keep':~remove,'remove':remove,'far':far,'signed_proxy':signed,
            'outside':c==OUTSIDE,'inside':c==INSIDE,'unknown':c==UNKNOWN}


def _v2_export(source,keep,destination,*,expected_dtype=None):
    """Atomic full-field PLY export with true byte preservation sampling."""
    dst=Path(destination)
    tmp=dst.with_name('.'+dst.stem+'.writing.tmp.ply')
    from gsrefine.roi.core import GaussianPLY
    try:
        n=source.export(keep,tmp)
        if int(n)!=int(np.count_nonzero(keep)):
            raise RuntimeError('Export Gaussian count mismatch')
        check=GaussianPLY(tmp)
        if check.vertex_count!=n or check.dtype!=source.dtype:
            raise RuntimeError('PLY attributes/types changed')
        if n:
            positions=np.unique(np.linspace(0,n-1,min(64,n),dtype=np.int64))
            indexes=np.flatnonzero(keep)[positions]
            if check.arr[positions].tobytes()!=source.arr[indexes].tobytes():
                raise RuntimeError('PLY content corrupted during export')
        del check
        os.replace(tmp,dst)
    finally:
        tmp.unlink(missing_ok=True)
    return {'path':str(dst),'gaussians':int(n),'bytes':int(dst.stat().st_size),
            'sha256':_sha(dst)}


def run_open_ab_main(*,manifest,preseal_report,output_dir,chunk_size=30000,
                     colmap_images=None,config_file=None,
                     b_method='implicit-volume',voxel_size=None,
                     max_volume_nodes=240000,max_sfm_rays=130000,
                     diagnostic_only=False,write_clean_ply=True,
                     winding_camera_max_delta=.2,winding_inside_min=.85,
                     camera_candidates=12,min_unobstructed_cameras=2,
                     camera_forward_dot_min=.2):
    """V2 two-resolution occupancy + original-surface exact distance experiment.

    `camera-legacy` deliberately routes to the old experiment unchanged.
    The full version is observationally constrained but *not ground truth*.
    """
    if b_method=='camera-legacy':
        if diagnostic_only:
            raise ValueError('Diagnostic-only is implemented for implicit-volume only')
        return run_open_ab_camera_legacy(manifest=manifest,preseal_report=preseal_report,
                   output_dir=output_dir,chunk_size=chunk_size,
                   colmap_images=colmap_images,config_file=config_file,
                   winding_camera_max_delta=winding_camera_max_delta,
                   winding_inside_min=winding_inside_min,
                   camera_candidates=camera_candidates,
                   min_unobstructed_cameras=min_unobstructed_cameras,
                   camera_forward_dot_min=camera_forward_dot_min,
                   write_clean_ply=write_clean_ply)
    if b_method!='implicit-volume':raise ValueError('Unknown B method')
    if chunk_size<=0 or max_volume_nodes<10000 or max_sfm_rays<100:
        raise ValueError('Invalid V2 run limits')
    from gsrefine.cleanup.implicit_space import (
        OUTSIDE,INSIDE,UNKNOWN,make_grid,read_colmap_images_index,
        sample_verified_colmap_rays,valid_observation_rays,solve_one_grid,
        boundary_points,query_stable_labels)
    from gsrefine.cleanup.sdf_common import robust_gaussian_radius,_raycast
    from gsrefine.roi.core import GaussianPLY
    import open3d as o3d
    record,main,inputs,run_dir,cfg,camera_path,centers,forwards,preseal = _validated_inputs(
        manifest,preseal_report,colmap_images=colmap_images,config_file=config_file)
    output=Path(output_dir).expanduser().resolve()
    # Do not allow output to replace any existing source or Stage-1/Stage-2 official output.
    if output in (run_dir,run_dir/'final',run_dir/'cleanup/final') or not output.is_relative_to(run_dir/'cleanup'):
        raise ValueError('V2 output must reside in a dedicated ROI cleanup subdirectory')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('V2 output directory already contains files; choose a fresh version directory')
    model=GaussianPLY(inputs['gaussian_ply'])
    xyz=np.asarray(model.positions(),dtype=np.float64)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():
        raise ValueError('Gaussian center array invalid')
    radii=robust_gaussian_radius(model,float(DEFAULTS['trim_fraction']),float(DEFAULTS['sigma_multiplier']))
    p95=float(record['sfm_fit']['p95']);delta=p95+float(radii['trimmed_mean_radius'])
    eps=float(DEFAULTS['epsilon'])
    if not(np.isfinite(delta) and delta>0):raise ValueError('Invalid Gaussian distance threshold')
    meshsha=_sha(record['main_observed_surface'])
    gsha=_sha(inputs['gaussian_ply'])
    scene=_raycast(o3d,main)
    d=np.empty(len(xyz),dtype=np.float32)
    for a in range(0,len(xyz),int(chunk_size)):
        b=min(a+int(chunk_size),len(xyz))
        query=o3d.core.Tensor(np.ascontiguousarray(xyz[a:b],np.float32))
        dist=np.asarray(scene.compute_distance(query).numpy(),dtype=np.float32).reshape(-1)
        if len(dist)!=b-a:raise RuntimeError('Bad exact distance array')
        d[a:b]=dist
        if a==0 or b==len(xyz) or (a//chunk_size)%10==0:
            print(f'[stage2-open-ab-v2] A original-surface distances {b:,}/{len(xyz):,}',flush=True)
    far=np.isfinite(d)&(d>delta+eps)
    output.mkdir(parents=True,exist_ok=True)
    report={'schema':'GSSceneRefine.stage2_open_ab_implicit_v2',
         'algorithm':'implicit_observed_tracks_twoscale_graphcut_v2',
         'created_utc':datetime.now(timezone.utc).isoformat(),
         'status':'A_DISTANCE_DIAGNOSTICS_READY',
         'stages':{'A':'exact unsigned distance to UNMODIFIED observed main Poisson',
                   'B':'two-scale 3D graph cut seeded by actual COLMAP camera-SfM tracks'},
         'main_surface':str(record['main_observed_surface']),
         'main_surface_sha256':meshsha,'main_triangles':int(len(main.faces)),
         'source_gaussian':str(inputs['gaussian_ply']),
         'source_gaussian_sha256':gsha,'source_gaussian_count':int(len(xyz)),
         'preseal_report':str(preseal),'colmap_images':str(camera_path),
         'colmap_images_sha256':_sha(camera_path),
         'auxiliary_components_ignored':int(record['topology_partition']['auxiliary_component_count']),
         'sfm_fit':record['sfm_fit'],'gaussian_scale':radii,
         'delta':float(delta),'epsilon':float(eps),
         'distance_metric':'original observed main triangles, NEVER implicit voxel boundary',
         'counts':{'input':int(len(xyz)),'far':int(far.sum()),'near_or_unknown_distance':int((~far).sum())},
         'warning':'Open surface interior/exterior is ESTIMATED, not a certified real solid. '
                   'The distance-only upper bound can remove real Gaussians and is DIAGNOSTIC ONLY.'}
    # Save A-stage checkpoint/report before any B-stage dependencies. A failed
    # graph cut still leaves useful diagnostic results, no false clean output.
    np.save(output/'distance_main.npy',d)
    report['distance_upper_bound'] = _v2_export(model,~far,
                                       output/'gaussian_distance_only_upper_bound.ply')
    _write_json_atomic(output/'open_ab_v2_report.json',report)
    if diagnostic_only:
        report['status']='DIAGNOSTIC_ONLY_DISTANCE_UPPER_BOUND_NOT_FORMAL_CLEAN'
        _write_json_atomic(output/'open_ab_v2_report.json',report)
        return {'status':report['status'],'report':str(output/'open_ab_v2_report.json'),
                'delta':float(delta),'far':int(far.sum()),
                'distance_upper_bound':report['distance_upper_bound']['path']}
    import igl
    if not hasattr(igl,'fast_winding_number'):
        raise RuntimeError('The installed libigl does not support fast_winding_number')
    # Strictly extract REAL SfM observations from COLMAP image <-> 3D tracks.
    sfm_path=Path(record['sfm']).resolve(strict=True)
    with np.load(sfm_path,allow_pickle=False) as npz:
        core=np.asarray(npz['xyz'],dtype=np.float64)[np.asarray(npz['in_core_roi'],dtype=bool)]
    poses=read_colmap_images_index(camera_path)
    if len(poses)!=len(centers):
        raise ValueError('Image-ID COLMAP parser disagrees with verified camera poses')
    from scipy.spatial import cKDTree
    parsed_centers=np.asarray([v[0] for v in poses.values()],dtype=np.float64)
    alignment_dist=cKDTree(centers).query(parsed_centers,k=1)[0]
    if np.max(alignment_dist)>1.e-6:
        raise ValueError('COLMAP parsed image centers inconsistent with Stage-1 orientation')
    points3d_path=Path(camera_path).with_name('points3D.bin')
    if not points3d_path.is_file():
        raise FileNotFoundError('V2 requires COLMAP points3D.bin ACTUAL tracks near images.bin')
    print('[stage2-open-ab-v2] B0 parse actual COLMAP image-to-SfM track observations',flush=True)
    rays_org,rays_end,track_stats=sample_verified_colmap_rays(
        points3d_path,poses,core,max_points=max_sfm_rays,
        max_rays=max_sfm_rays,xyz_match_tolerance=0.0005)
    # At observed point endpoint, allow SfM-Poisson P95 discrepancy;
    # discard blocked observation paths (not guessed Gaussian rays).
    # Use the ACTUAL adaptive (memory-capped) voxel spacing, not merely the
    # requested spacing, for the near-SfM endpoint safety margin.
    fine=make_grid(xyz,np.asarray(main.vertices),delta,voxel_size,max_nodes=int(max_volume_nodes))
    coarse=make_grid(xyz,np.asarray(main.vertices),delta,voxel_size,max_nodes=max(10000,int(max_volume_nodes*.40)))
    margin=max(delta,2.*fine.voxel)
    good=valid_observation_rays(rays_org,rays_end,scene,o3d,endpoint_margin=margin)
    track_stats['rays_verified_unobstructed']=int(good.sum())
    track_stats['rejected_blocked_rays']=int((~good).sum())
    if good.sum()<200:
        raise RuntimeError('Too few trustworthy SfM camera rays; cannot infer outside')
    rays_org=rays_org[good];rays_end=rays_end[good]
    # Two bounded full grids. Different resolutions are independent checks.
    if abs(fine.voxel-coarse.voxel)<1e-6:
        raise ValueError('Two occupancy grids have indistinguishable resolution; increase --max-volume-nodes')
    grids=[];infos=[]
    try:
        for name,spec in [('coarse',coarse),('fine',fine)]:
            print(f'[stage2-open-ab-v2] B {name} grid {spec.shape} '
                  f'{spec.count:,} voxels, voxel={spec.voxel:.5f}',flush=True)
            labels,info=solve_one_grid(main,scene,o3d,igl,spec,poses,rays_org,rays_end,
                                      endpoint_margin=margin)
            info['name']=name
            grids.append((spec,labels));infos.append(info)
            np.savez_compressed(output/f'implicit_{name}_grid.npz',
                origin=spec.origin,voxel=np.array(spec.voxel),shape=np.asarray(spec.shape),
                labels=labels)
        boundary=boundary_points(main)
        occ=np.full(len(xyz),UNKNOWN,dtype=np.int8)
        indexes=np.flatnonzero(far)
        for a in range(0,len(indexes),chunk_size):
            sl=indexes[a:a+chunk_size]
            occ[sl]=query_stable_labels(xyz[sl],grids,boundary,
                                       boundary_margin=max(2*fine.voxel,2*delta))
        labels=classify_implicit_signed_proxy(d,occ,delta,eps)
        counts=report['counts']
        counts.update({'outside_confident_far':int((far&labels['outside']).sum()),
                       'inside_confident_far':int((far&labels['inside']).sum()),
                       'uncertain_far_kept':int((far&labels['unknown']).sum()),
                       'removed':int(labels['remove'].sum()),
                       'kept':int(labels['keep'].sum()),
                       'removed_fraction':float(labels['remove'].mean())})
        report['B_evidence']={'colmap_points3d_sha256':_sha(points3d_path),
                              'colmap_track_statistics':track_stats,
                              'voxel_grids':infos,
                              'boundary_protection_margin':float(max(2*fine.voxel,2*delta)),
                              'two_scale_agreement_required':True}
        # More than half far points may be truly unknown: explicit and safe.
        np.savez_compressed(output/'classification_masks.npz',
            keep=labels['keep'],remove=labels['remove'],
            over_threshold=far,inside_confident=labels['inside'],
            outside_confident=labels['outside'],uncertain_kept=labels['unknown'],
            distance_main=d,signed_distance_proxy=labels['signed_proxy'],
            occupancy_label=occ)
        if write_clean_ply:
            report['output_clean_gaussian']=_v2_export(model,labels['keep'],
                                             output/'gaussian_roi_clean.ply')
        report['status']='EXPERIMENTAL_IMPLICIT_VOLUME_SPATIAL_CLASSIFICATION'
        report['delete_rule']='confirmed exterior in both voxel resolutions AND exact original main distance > delta + epsilon; else keep'
        _write_json_atomic(output/'open_ab_v2_report.json',report)
        return {'status':report['status'],'report':str(output/'open_ab_v2_report.json'),
                'removed':counts['removed'],'kept':counts['kept'],
                'distance_threshold_delta':float(delta),
                'distance_upper_bound':report['distance_upper_bound']['path'],
                'clean_gaussian':str(output/'gaussian_roi_clean.ply') if write_clean_ply else None}
    except Exception as exc:
        report['status']='BLOCKED_IMPLICIT_SPATIAL_SOLVE'
        report['block_reason']=f'{type(exc).__name__}: {exc}'
        report['B_evidence']={'colmap_track_statistics':track_stats,'completed_grids':infos}
        _write_json_atomic(output/'open_ab_v2_report.json',report)
        return {'status':report['status'],'reason':report['block_reason'],
                'report':str(output/'open_ab_v2_report.json'),
                'distance_upper_bound':report['distance_upper_bound']['path']}


# A completed A-distance upper bound must remain usable even when B fails;
# no partially classified Gaussian PLY is exported on B errors.
_run_open_ab_main_v2_impl = run_open_ab_main


def run_open_ab_main(**kwargs):
    try:
        return _run_open_ab_main_v2_impl(**kwargs)
    except Exception as exc:
        if kwargs.get('b_method','implicit-volume') != 'implicit-volume':
            raise
        output=Path(kwargs.get('output_dir','/nonexistent-open-ab')).expanduser().resolve()
        report_path=output/'open_ab_v2_report.json'
        if not report_path.exists():
            raise  # fail input provenance before allocating any output
        report=json.loads(report_path.read_text(encoding='utf8'))
        if report.get('status','').startswith('BLOCKED'):
            return {'status':report['status'],'report':str(report_path),
                    'reason':report.get('block_reason','unknown')}
        # Avoid masking unexpected programming failures, while preserving
        # diagnostics and explicitly refusing any automatic Gaussian delete.
        report['status']='BLOCKED_IMPLICIT_SPATIAL_PRECHECK'
        report['block_reason']=f'{type(exc).__name__}: {exc}'
        _write_json_atomic(report_path,report)
        return {'status':report['status'],'report':str(report_path),
                'reason':report['block_reason'],
                'distance_upper_bound':report.get('distance_upper_bound',{}).get('path')}
