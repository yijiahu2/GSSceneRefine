"""GSSceneRefine Stage 2: Stage-1 ROI SfM -> local Poisson -> verified SDF.

Stage 1 supplies the immutable Gaussian ROI and exact per-location R3 safety.
Older runs can be supported only if the SAME Stage-1 local safety field
is available; a global scalar d/P95 fallback is explicitly prohibited.
No original SfM/Poisson/Gaussian files are mutated. Local Poisson is a
candidate observed surface, not automatically a certified target solid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

import numpy as np

from gsrefine.roi.core import GaussianPLY, file_sha256, exact_sfm_to_poisson_distance_open3d
from gsrefine.geometry.orientation import read_colmap_poses, horizontal_basis
from gsrefine.geometry.sdf_sealing import choose_main_component, prepare_open_poisson, seal_main
from gsrefine.geometry.roi_sfm_local import resolve_stage2_exact_roi_sfm
from gsrefine.geometry.roi_poisson import build_local_poisson, DEFAULT_POISSON


DEFAULTS = {
    'trim_fraction': .10, 'sigma_multiplier': 3.,
    'nsamples': 3, 'chunk_size': 40000, 'epsilon': 1e-5,
    'min_main_face_fraction': .85, 'bottom_margin_fraction': .10,
    'min_camera_outside_fraction': 1.,
    'min_camera_normal_agreement': .60,
    'max_p95_to_sfm_spacing': 3.,
    'max_p95_to_roi_diagonal': .05,
    'poisson_weld_digits': 8,
    'min_roi_fit_points': 100,
    'allow_inferred_guard_export': False,
}


SEALING_ALGORITHM_REVISION = 'exact_local_r3_sfm_double_safety_poisson_direct_sdf_20261107_v2'



def _iso_now():
    return datetime.now(timezone.utc).isoformat()


def _path(value):
    p = Path(value).expanduser().resolve(strict=True)
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf8') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _read_json(path):
    return json.loads(_path(path).read_text(encoding='utf8'))


def _load_stage1_manifest(path):
    manifest_file = _path(path)
    # The manifest itself is immutable. Stage-2 has a distinct completion record.
    manifest = _read_json(manifest_file)
    if manifest.get('schema') != 'GSSceneRefine.output.v3' or manifest.get('status') != 'completed':
        raise ValueError('Need a verified GSSceneRefine Stage-1 output.v3 completed manifest')
    if manifest.get('result_status') != 'ROI_RESTORED_NOT_ANOMALY_CLEANED':
        raise ValueError('The provided manifest is not the expected Stage-1 R3-only result')
    paths = manifest['stage2_inputs']
    required = ('gaussian_ply', 'roi_json', 'poisson_mesh', 'clean_sfm_npz')
    inputs = {key: _path(paths[key]) for key in required}
    run_dir = manifest_file.parent.parent
    if inputs['gaussian_ply'] != (run_dir / 'final/gaussian_roi.ply').resolve():
        raise ValueError('Manifest Gaussian is not the immutable Stage-1 final/gaussian_roi.ply')
    for k, name in [('poisson','poisson_mesh'), ('sfm','clean_sfm_npz'), ('roi','roi_json')]:
        expected = manifest.get('input_sha256', {}).get(k)
        if not expected or file_sha256(inputs[name]) != expected:
            raise ValueError(f'Stage-1 {name} changed; rebuild the Stage-1 provenance instead of cleaning')
    expected_gaussian = manifest.get('output_sha256')
    if not expected_gaussian or file_sha256(inputs['gaussian_ply']) != expected_gaussian:
        raise ValueError('Stage-1 ROI Gaussian hash mismatch; will not process an altered PLY')
    return manifest_file, manifest, inputs, run_dir


def robust_gaussian_radius(ply, trim_fraction=.1, sigma_multiplier=3., chunk_size=250000):
    """GSPL stores log(std) in scale_0..2. P10/P90 trimmed major 3sigma radius."""
    if not (0 <= trim_fraction < .5 and sigma_multiplier > 0):
        raise ValueError('Invalid robust Gaussian scale parameters')
    n = ply.vertex_count
    if n < 10:
        raise ValueError('At least ten Gaussian records are needed for trimmed statistics')
    radii = np.empty(n, dtype=np.float64)
    for a in range(0, n, chunk_size):
        b = min(n, a+chunk_size)
        log_scales = np.column_stack([ply.arr[f'scale_{i}'][a:b] for i in range(3)]).astype(np.float64)
        if not np.isfinite(log_scales).all() or (abs(log_scales)>80).any():
            raise ValueError('Gaussian PLY contains non-finite or implausible log scales')
        radii[a:b] = sigma_multiplier * np.exp(log_scales.max(axis=1))
    low, high = np.quantile(radii, [trim_fraction, 1-trim_fraction])
    central = (radii>=low)&(radii<=high)
    mean = float(radii[central].mean())
    if not np.isfinite(mean) or mean<=0:
        raise ValueError('Invalid Gaussian radius statistic')
    return {'gaussians':n, 'sigma_multiplier':float(sigma_multiplier),
            'trim_fraction_each_tail':float(trim_fraction),
            'quantile_low':float(low), 'quantile_high':float(high),
            'central_count':int(central.sum()), 'trimmed_mean_radius':mean,
            'radius_p95':float(np.quantile(radii, .95))}


def classify_direct_sdf(distances, delta, epsilon=1e-5):
    """Positive outside, negative inside. Unknown distances remain conservatively kept."""
    distances = np.asarray(distances)
    if not (np.isfinite(delta) and delta>=0 and np.isfinite(epsilon) and epsilon>=0):
        raise ValueError('Invalid SDF threshold or epsilon')
    finite = np.isfinite(distances)
    remove = finite & (distances > delta + epsilon)
    uncertain = ~finite
    return ~remove, remove, uncertain


def _local_basis(up):
    """Reuse existing GSSceneRefine COLMAP-up basis; columns are local axes in world."""
    return np.asarray(horizontal_basis(up), dtype=np.float64).T


def _raycast(o3d, mesh):
    legacy = o3d.geometry.TriangleMesh()
    legacy.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, np.float64))
    legacy.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, np.int32))
    scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    return scene


def _camera_check(o3d, scene, camera_xyz, nsamples):
    q = o3d.core.Tensor(np.ascontiguousarray(camera_xyz,np.float32))
    d = np.asarray(scene.compute_signed_distance(q, nsamples=nsamples).numpy())
    closest = scene.compute_closest_points(q)
    pts = closest['points'].numpy().astype(np.float64)
    normals = closest['primitive_normals'].numpy().astype(np.float64)
    facing = np.sum(normals*(camera_xyz-pts), axis=1)
    return {'total':int(len(camera_xyz)),
            'outside_fraction':float(np.mean(d>0)),
            'outward_normal_agreement':float(np.mean(facing>0)),
            'nonfinite_distances':int(np.sum(~np.isfinite(d)))}


def _validated_local_geometry(raw_mesh, roi_sfm_file, camera_xyz, up, cfg, *, o3d):
    """Fit and seal a locally RECONSTRUCTED Poisson, with no mesh ROI clipping.

    The Stage-1 SfM handoff distinguishes core and build-only context points.
    P95 is estimated from core ONLY, never from the extra SfM support shell or
    the artificial skirt/base. The full mesh still needs manual target review.
    """
    import trimesh
    from scipy.spatial import cKDTree
    from gsrefine.geometry.roi_poisson import read_local_sfm
    if not isinstance(raw_mesh,trimesh.Trimesh) or not len(raw_mesh.faces):
        raise ValueError('Local SfM Poisson must be a nonempty triangle mesh')
    sfm_xyz,core_mask,_=read_local_sfm(roi_sfm_file)
    fit_xyz=sfm_xyz[core_mask]
    if len(fit_xyz)<int(cfg['min_roi_fit_points']):
        raise ValueError('Insufficient core SfM points for local Poisson fit P95')
    fit_dist=exact_sfm_to_poisson_distance_open3d(raw_mesh,fit_xyz,batch=50000)
    p95=float(np.quantile(fit_dist,.95))
    neighbors=cKDTree(fit_xyz).query(fit_xyz,k=min(7,len(fit_xyz)),workers=-1)[0]
    spacing=float(np.median(neighbors[:,-1])) if neighbors.ndim==2 else float(np.median(neighbors))
    fit={'sfm_count':len(fit_xyz),'source_build_sfm_count':len(sfm_xyz),
         'distance_mean':float(np.mean(fit_dist)),
         'distance_p90':float(np.quantile(fit_dist,.90)),
         'distance_p95':p95,'distance_p99':float(np.quantile(fit_dist,.99)),
         'sfm_spacing_median':spacing,
         'definition':'CORE ROI SfM to OBSERVED locally-reconstructed Poisson nearest-triangle distance (no context-only points and no artificial closure faces)'}
    repaired,topology_report=prepare_open_poisson(
        raw_mesh,weld_digits=int(cfg['poisson_weld_digits']))
    B=_local_basis(up)
    local=repaired.copy()
    local.vertices=np.asarray(repaired.vertices)@B
    main,components=choose_main_component(local,float(cfg['min_main_face_fraction']))
    candidates=[]
    for sign in (1,-1):
        item={'cap_direction_relative_to_verified_up':sign}
        try:
            span=np.ptp(np.asarray(main.vertices)[:,2])
            sealed,seal_report,_artificial=seal_main(
                main,2,sign,float(cfg['bottom_margin_fraction'])*float(span))
            sealed.vertices=np.asarray(sealed.vertices)@B.T
            sc=_raycast(o3d,sealed)
            cam=_camera_check(o3d,sc,camera_xyz,int(cfg['nsamples']))
            item.update(sealing=seal_report,cameras=cam,
                        passed=(cam['nonfinite_distances']==0 and
                          cam['outside_fraction']>=float(cfg['min_camera_outside_fraction']) and
                          cam['outward_normal_agreement']>=float(cfg['min_camera_normal_agreement'])))
            candidates.append((item,sealed,sc))
        except Exception as exc:
            item.update(passed=False,error=str(exc))
            candidates.append((item,None,None))
    valid=[r for r in candidates if r[0]['passed']]
    if not valid:
        raise ValueError('Locally reconstructed Poisson did not pass sealing+ALL-camera validation: '+
                         json.dumps([r[0] for r in candidates],ensure_ascii=False))
    valid.sort(key=lambda x:(x[0]['cameras']['outside_fraction'],
                             x[0]['cameras']['outward_normal_agreement']),reverse=True)
    selection=valid[0]
    report={'fit':fit,
            'roi_crop':{'selection_mode':'local_poisson_from_exact_stage1_local_safety_2x',
                        'source':'local_screened_poisson_from_core_plus_context_sfm',
                        'exact_r3_surface_certified':False,
                        'warning':'Local Poisson may include non-target surface; manual geometry review required.'},
            'components':components,'topology_preparation':topology_report,
            'direction_trials':[r[0] for r in candidates],
            'selected_direction':selection[0]['cap_direction_relative_to_verified_up'],
            'selected_camera_checks':selection[0]['cameras'],
            'original_poisson_vertices':int(len(raw_mesh.vertices)),
            'original_poisson_triangles':int(len(raw_mesh.faces)),
            'observed_roi_poisson_triangles':int(len(raw_mesh.faces)),
            'sealed_poisson_triangles':int(len(selection[1].faces)),
            'original_main_triangles_in_sealed':int(selection[0]['sealing'].get('mesh_before_faces',len(main.faces))),
            'mesh_is_watertight':bool(selection[1].is_watertight),
            'mesh_is_volume':bool(selection[1].is_volume),
            'artificial_surface_warning':'Skirt and base are closure aids, NOT observed SfM geometry.'}
    return selection[1],report,raw_mesh


def _camera_path(run_dir, roi_payload, config_file, override=None):
    import yaml
    cfg=yaml.safe_load(_path(config_file).read_text(encoding='utf8')) or {}
    candidate=override or (cfg.get('roi_init',{}).get('orientation',{}) or {}).get('colmap_images')
    if not candidate:
        candidate=(cfg.get('paths',{}) or {}).get('colmap_images')
    if not candidate:
        raise FileNotFoundError('COLMAP images.bin path missing; supply --colmap-images')
    camera=_path(candidate)
    sha=roi_payload['annotation_metadata'].get('orientation_source_sha256')
    if sha and sha!=file_sha256(camera):
        raise ValueError('COLMAP image poses do not match Stage-1 ROI orientation provenance')
    return camera


def _safe_geometry_cache(geom_dir, signature, raw_mesh, sfm_file, cams, up, cfg, *, o3d, observed_file):
    import trimesh
    folder=geom_dir / signature[:24]
    metadata=folder/'sealing_report.json'
    sealed_ply=folder/'sealed_poisson.ply'
    observed_ply=Path(observed_file).resolve(strict=True)
    if sealed_ply.is_file() and metadata.is_file():
        saved=_read_json(metadata)
        if saved.get('cache_signature')==signature and saved.get('validation_passed') is True:
            mesh=trimesh.load(sealed_ply,process=False)
            observed=trimesh.load(observed_ply,process=False)
            if (not isinstance(mesh,trimesh.Trimesh) or not mesh.is_watertight
                or not mesh.is_volume or not isinstance(observed,trimesh.Trimesh)):
                raise ValueError('Cached ROI geometry failed volume/surface validation')
            if (file_sha256(sealed_ply) != saved.get('mesh_sha256') or
                file_sha256(observed_ply) != saved.get('roi_mesh_sha256')):
                raise ValueError('Cached ROI geometry hash mismatch')
            return mesh,saved,True
        raise ValueError('Cached ROI geometry signature invalid; refusing to replace it')
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('Incomplete ROI geometry cache; inspect manually: '+str(folder))
    try:
        mesh, report, roi_surface = _validated_local_geometry(
            raw_mesh,sfm_file,cams,up,cfg,o3d=o3d)
    except Exception as exc:
        # A failed ROI extraction or sealing must not claim a valid cache.
        # Save the message outside the cache to enable local troubleshooting.
        diagnostic_path=geom_dir/'failed_geometry_check.json'
        _write_json(diagnostic_path,{'status':'failed','cache_signature':signature,
                  'error':str(exc),'timestamp_utc':_iso_now()})
        raise
    folder.mkdir(parents=True, exist_ok=True)
    mesh.export(sealed_ply)
    # The single local Poisson file is already cached; do not duplicate its PLY.
    check=trimesh.load(sealed_ply,process=False)
    if not isinstance(check,trimesh.Trimesh) or not check.is_watertight or not check.is_volume:
        raise ValueError('Saved sealed ROI Poisson failed round-trip validation')
    report.update(cache_signature=signature,validation_passed=True,
                  roi_mesh_file=str(observed_ply.resolve()),roi_mesh_sha256=file_sha256(observed_ply),
                  mesh_file=str(sealed_ply.resolve()),mesh_sha256=file_sha256(sealed_ply))
    _write_json(metadata,report)
    return mesh,report,False


def cleanup_from_manifest(manifest, *, colmap_images=None, no_export=False,
                          dry_run=False, chunk_size=None, config_override=None,
                          surface_only=False, keep_diagnostics=False,
                          allow_inferred_guard_export=False, surface_face_mask=None,
                          local_safety_field=None):
    import trimesh
    import open3d as o3d
    import yaml
    path, upstream, inputs, run_dir = _load_stage1_manifest(manifest)
    cfgfile = _path(config_override) if config_override else _path(run_dir/'metadata/effective_scene.yaml')
    cfg=yaml.safe_load(cfgfile.read_text(encoding='utf8')) or {}
    opts={**DEFAULTS, **(cfg.get('stage2_cleanup') or {})}
    if chunk_size is not None:opts['chunk_size']=chunk_size
    if allow_inferred_guard_export:
        opts['allow_inferred_guard_export']=True
    if int(opts['min_roi_fit_points']) < 10:
        raise ValueError('Invalid minimum ROI SfM fitting count')
    if not 1<=int(opts['nsamples'])<=11 or int(opts['nsamples'])%2==0:
        raise ValueError('nsamples must be a positive odd integer no greater than 11')
    if int(opts['chunk_size'])<1 or not np.isfinite(float(opts['epsilon'])) or float(opts['epsilon'])<0:
        raise ValueError('Invalid chunk size or numerical epsilon')
    if int(opts['poisson_weld_digits']) != opts['poisson_weld_digits'] or not 0<=int(opts['poisson_weld_digits'])<=14:
        raise ValueError('poisson_weld_digits must be an integer between 0 and 14')
    if not 0<opts['min_main_face_fraction']<=1:
        raise ValueError('min_main_face_fraction must be in (0,1]')

    output = run_dir/'cleanup/final/gaussian_roi.ply'
    out_meta=run_dir/'cleanup/metadata/manifest.json'
    if (not no_export and not surface_only) and (out_meta.exists() or output.exists()):
        raise FileExistsError('Stage-2 output exists; Stage-1 and published Stage-2 models are immutable: '+str(out_meta))

    roi_payload=_read_json(inputs['roi_json'])
    annotation=roi_payload.get('annotation_metadata',{})
    if annotation.get('orientation_verified') is not True:
        raise ValueError('Stage-1 confirmed ROI orientation is not verified')
    camera_path=_camera_path(run_dir,roi_payload,cfgfile,colmap_images)
    cameras,*_=read_colmap_poses(camera_path)
    up=np.asarray(annotation.get('up_vector'),float)
    input_ply=GaussianPLY(inputs['gaussian_ply'])
    if input_ply.vertex_count==0:
        raise ValueError('Stage-1 Gaussian ROI is empty')
    if surface_face_mask is not None:
        raise ValueError('--surface-face-mask is for the deprecated mesh-cropping method; '
                         'the new route reconstructs a Poisson directly from ROI SfM')
    # Stage-1 exact local safety d(x) ONLY; legacy scalar P95 fallback is banned.
    # Prefer verified Stage-1 v2 SfM; else create v2 from the saved exact field.
    roi_sfm_file,roi_sfm_report,stage1_sfm_reused=resolve_stage2_exact_roi_sfm(
        run_dir,upstream,inputs,local_safety_file=local_safety_field)
    if roi_sfm_report.get('selection_mode') != 'exact_per_location_stage1_r3_safety_1x_2x':
        raise ValueError('Stage-2 requires verified 1x/2x per-location Stage-1 R3 safety; no scalar fallback')
    print('[stage2-cleanup] ROI SfM input:',roi_sfm_file,
          'source='+('Stage-1 verified' if stage1_sfm_reused else 'Stage-2 fallback'),flush=True)
    poisson_params={**DEFAULT_POISSON,**(cfg.get('stage2_roi_poisson') or {})}
    poisson_file,poisson_report,poisson_reused=build_local_poisson(
        roi_sfm_file,run_dir/'cleanup/geometry',parameters=poisson_params,o3d=o3d)
    print('[stage2-cleanup] Local Poisson:',poisson_file,
          'faces='+str(poisson_report['observed_faces']),
          'cache='+('reused' if poisson_reused else 'built'),flush=True)
    raw=trimesh.load(poisson_file,process=False,force='mesh')
    if not isinstance(raw,trimesh.Trimesh):
        raise ValueError('Local Poisson did not load as a triangle mesh')
    mask_origin='stage1_exact_local_roi_sfm' if stage1_sfm_reused else 'stage2_generated_from_exact_stage1_local_safety'
    radius=robust_gaussian_radius(input_ply,float(opts['trim_fraction']),
                                  float(opts['sigma_multiplier']))
    h=hashlib.sha256()
    geometry_opts={k:v for k,v in opts.items() if k not in (
         'allow_inferred_guard_export','chunk_size','epsilon',
         'max_p95_to_sfm_spacing','max_p95_to_roi_diagonal')}
    for value in (SEALING_ALGORITHM_REVISION,file_sha256(cfgfile),
                  file_sha256(poisson_file),file_sha256(roi_sfm_file),
                  file_sha256(inputs['roi_json']),file_sha256(inputs['gaussian_ply']),
                  file_sha256(camera_path),str(up.tolist()),
                  json.dumps(geometry_opts,sort_keys=True,default=str)):
        h.update(value.encode('utf8'))
    signature=h.hexdigest()
    geometry_dir=run_dir/'cleanup/cache'
    if surface_only:
        # The Poisson cache is the true, *single* locally reconstructed mesh.
        # The preview only writes a small report; avoids duplicating huge PLY.
        preview=run_dir/'cleanup/preview'
        preview.mkdir(parents=True,exist_ok=True)
        report_path=preview/f'local_poisson_preview_{signature[:12]}.json'
        payload={'schema':'GSSceneRefine.roi_surface_preview.v2',
                 'status':'LOCAL_POISSON_ONLY_NO_SEAL_NO_SDF',
                 'signature':signature,'local_poisson':str(poisson_file),
                 'local_poisson_sha256':file_sha256(poisson_file),
                 'roi_sfm_file':str(roi_sfm_file),
                 'stage1_roi_sfm_reused':stage1_sfm_reused,
                 'roi_sfm_report':roi_sfm_report,'poisson':poisson_report,
                 'warning':'This is a Poisson reconstruction from expanded SfM; '
                           'not all exterior/context surface is necessarily target geometry.'}
        if report_path.exists():
            existing=_read_json(report_path)
            if existing.get('signature')!=signature or existing.get('local_poisson_sha256')!=payload['local_poisson_sha256']:
                raise FileExistsError('ROI Poisson preview belongs to another input; refusing overwrite')
        else:
            _write_json(report_path,payload)
        print('[stage2-cleanup] Local ROI Poisson preview:',poisson_file,flush=True)
        print('[stage2-cleanup] Preview report:',report_path,flush=True)
        return {'status':'local_poisson_only','roi_mesh_file':str(poisson_file),
                'roi_sfm_file':str(roi_sfm_file),'roi_sfm_report':roi_sfm_report}
    sealed,sealing_report,reused=_safe_geometry_cache(
        geometry_dir,signature,raw,roi_sfm_file,cameras,up,opts,o3d=o3d,
        observed_file=poisson_file)
    inferred=True  # A local raw Poisson is a candidate, not a certified target mask.
    if inferred and not (no_export or dry_run or opts['allow_inferred_guard_export']):
        raise ValueError('Local ROI Poisson is NOT automatically a certified R3 target '
                         'surface. Inspect poisson_roi.ply and the sealing report first; '
                         'then explicitly pass --allow-inferred-guard-export to publish '
                         'a provisional cleaned PLY.')
    p95=float(sealing_report['fit']['distance_p95'])
    roi_xyz=input_ply.positions()
    if not np.isfinite(roi_xyz).all():
        raise ValueError('Gaussian ROI contains non-finite centers; clean input provenance first')
    diag=float(np.linalg.norm(np.max(roi_xyz,axis=0)-np.min(roi_xyz,axis=0)))
    p95_limits={
      'max_by_sfm_spacing':float(opts['max_p95_to_sfm_spacing'])*float(sealing_report['fit']['sfm_spacing_median']),
      'max_by_roi_diagonal':float(opts['max_p95_to_roi_diagonal'])*diag}
    if any(p95>lim for lim in p95_limits.values()):
        raise ValueError('Unusually large SfM-to-Poisson P95; stop automatic cleaning. '+
                         json.dumps({'p95':p95,'limits':p95_limits}))
    delta=float(radius['trimmed_mean_radius']+p95)
    epsilon=float(opts['epsilon'])
    print(f'[stage2-cleanup] Verified Stage-1 ROI Gaussian={input_ply.vertex_count:,}',flush=True)
    print(f'[stage2-cleanup] P95={p95:.9g}, Gaussian trimmed 3sigma={radius["trimmed_mean_radius"]:.9g}, '
          f'delta={delta:.9g}; sealed cache={"reused" if reused else "built"}',flush=True)
    sc=_raycast(o3d,sealed)
    n=input_ply.vertex_count
    distances=np.full(n,np.nan,np.float32)
    artificial=np.zeros(n,bool)
    # Triangle IDs are preserved when exporting the unmodified-face + artificial-face mesh.
    first_artificial_face=int(sealing_report['original_main_triangles_in_sealed'])
    batch=int(opts['chunk_size'])
    for a in range(0,n,batch):
        b=min(a+batch,n)
        points=o3d.core.Tensor(np.ascontiguousarray(roi_xyz[a:b],np.float32))
        ds=sc.compute_signed_distance(points,nsamples=int(opts['nsamples'])).numpy().astype(np.float32)
        distances[a:b]=ds
        # Risk flag only. Do not silently redefine the classification threshold.
        closest=sc.compute_closest_points(points)
        artificial[a:b]=closest['primitive_ids'].numpy().reshape(-1)>=first_artificial_face
        if a==0 or b==n or ((a//batch)%10==0):
            print(f'[stage2-cleanup] Direct SDF Gaussian centers {b:,}/{n:,}',flush=True)
    keep,remove,unknown=classify_direct_sdf(distances,delta,epsilon)
    summary={
        'schema':'GSSceneRefine.stage2_cleanup.v1','status':'diagnostics_only' if (no_export or dry_run) else 'completed',
        'completed_utc':_iso_now(),'stage1_manifest':str(path),'stage1_manifest_sha256':file_sha256(path),
        'stage1_gaussian':str(inputs['gaussian_ply']),'stage1_gaussian_sha256':file_sha256(inputs['gaussian_ply']),
        'source_roi_json':str(inputs['roi_json']),'roi_json_sha256':file_sha256(inputs['roi_json']),
        'source_poisson':str(inputs['poisson_mesh']),'roi_poisson':str(poisson_file),
        'roi_poisson_sha256':file_sha256(poisson_file),
        'roi_surface_mode':sealing_report['roi_crop']['selection_mode'],
        'roi_sfm':str(roi_sfm_file),'roi_sfm_stage1_reused':stage1_sfm_reused,
        'roi_sfm_report':roi_sfm_report,'roi_poisson_report':poisson_report,
        'roi_surface_source':mask_origin,
        'roi_surface_diagnostics':sealing_report['roi_crop'],
        'sealed_poisson':sealing_report['mesh_file'],
        'sealed_poisson_sha256':sealing_report['mesh_sha256'],
        'camera_poses':str(camera_path),'camera_poses_sha256':file_sha256(camera_path),
        'source_model_scope':cfg.get('source_model_scope','unknown'),
        'scene_frame':'original_COLMAP_world','gaussian_index_space':'Stage-1 final/gaussian_roi.ply vertex rows, NOT full checkpoint rows',
        'original_roi_count':n,'removed_count':int(remove.sum()),'kept_count':int(keep.sum()),
        'removed_fraction':float(remove.mean()),'uncertain_nonfinite_kept_count':int(unknown.sum()),
        'removed_nearest_artificial_face_count':int((remove & artificial).sum()),
        'removed_nearest_artificial_face_fraction':float((remove & artificial).sum()/max(int(remove.sum()),1)),
        'delta':delta,'sfm_p95':p95,'gaussian_scale':radius,
        'epsilon':epsilon,'nsamples':int(opts['nsamples']),'p95_safeguards':p95_limits,
        'geometry_cache_reused':reused,
        'geometry_cache_signature':signature,'algorithm_revision':SEALING_ALGORITHM_REVISION,
        'effective_config':str(cfgfile),'effective_config_sha256':file_sha256(cfgfile),
        'criteria':'Keep iff nonfinite SDF (flagged uncertain) OR finite SDF(center) <= delta+epsilon',
        'limitations':['This is center-based SDF pruning, not ellipsoid-support volumetric classification.',
                       'Artificial skirt/base face nearest to a Gaussian is a diagnostic, not observed surface.',
                       'Image quality and free-view reconstruction fidelity require separate testing.'],
        'result_status':'PROVISIONAL_INFERRED_GUARD_REQUIRES_VISUAL_VERIFICATION' if inferred else 'PROVISIONAL_REQUIRES_VISUAL_VERIFICATION',
        'output_ply':str(output) if not (no_export or dry_run) else None}
    # By default avoid multi-GB distance/mask copies; diagnostics are opt-in.
    # A dry run publishes only its small metadata report, and may be re-run
    # with a new source signature without changing Stage-1 outputs.
    stage2_dir=run_dir/'cleanup'
    stage2_dir.mkdir(parents=True,exist_ok=True)
    if keep_diagnostics:
        diagnostics=stage2_dir/('preview/diagnostics' if (no_export or dry_run) else 'diagnostics')
        diagnostics.mkdir(parents=True,exist_ok=True)
        for f in ('signed_distances.npy','classification_masks.npz','removed_roi_local_indices.npy'):
            if (diagnostics/f).exists():
                raise FileExistsError('Diagnostics from previous run exist; not overwriting: '+str(diagnostics/f))
        np.save(diagnostics/'signed_distances.npy',distances)
        np.savez_compressed(diagnostics/'classification_masks.npz',keep=keep,remove=remove,
                            uncertain_kept=unknown,nearest_artificial_face=artificial)
        np.save(diagnostics/'removed_roi_local_indices.npy',np.flatnonzero(remove).astype(np.int64))
    if not (no_export or dry_run):
        output.parent.mkdir(parents=True,exist_ok=True)
        temp=output.with_name(output.name+'.writing.tmp')
        try:
            written=input_ply.export(keep,temp)
            checked=GaussianPLY(temp)
            if written!=int(keep.sum()) or checked.vertex_count!=written or checked.dtype!=input_ply.dtype:
                raise RuntimeError('Stage-2 export changed vertex count or field schema')
            if temp.stat().st_size!=checked.data_offset+written*checked.dtype.itemsize:
                raise RuntimeError('Stage-2 PLY has unexpected binary data length')
            if written:
                probe=np.unique(np.linspace(0,written-1,min(32,written),dtype=int))
                ids=np.flatnonzero(keep)[probe]
                if checked.arr[probe].tobytes()!=input_ply.arr[ids].tobytes():
                    raise RuntimeError('Stage-2 PLY changed source Gaussian attributes')
            del checked
            os.replace(temp,output)
        finally:
            temp.unlink(missing_ok=True)
        summary['output_sha256']=file_sha256(output)
        summary['output_size_bytes']=output.stat().st_size
    if no_export or dry_run:
        _write_json(stage2_dir/'preview/cleanup_summary.json',summary)
    else:
        _write_json(out_meta,summary)  # atomic success marker, Stage-1 unchanged
    print(f'[stage2-cleanup] retained={summary["kept_count"]:,}, removed={summary["removed_count"]:,} '
          f'({summary["removed_fraction"]:.2%}); output={summary["output_ply"]}',flush=True)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,help='Stage-1 metadata/manifest.json (completed R3 run)')
    p.add_argument('--colmap-images',help='COLMAP images.bin, only when absent in effective_scene.yaml')
    p.add_argument('--config',help='Optional stage-1 effective_scene.yaml with stage2_cleanup settings')
    p.add_argument('--chunk-size',type=int,default=None)
    p.add_argument('--no-export',action='store_true',help='Compute SDF statistics without writing cleaned PLY')
    p.add_argument('--surface-only',action='store_true',help='Reconstruct/reuse local Poisson from ROI SfM, no seal or SDF')
    p.add_argument('--keep-diagnostics',action='store_true',help='Keep full SDF distances and ROI-index masks; defaults to compact metadata')
    p.add_argument('--allow-inferred-guard-export',action='store_true',
                   help='After local Poisson inspection, explicitly permit provisional Gaussian cleaning export')
    p.add_argument('--surface-face-mask',help='Deprecated mesh-cropping option; unsupported with local SfM route')
    p.add_argument('--stage1-local-safety-field',help='Verified per-SfM-point d(x) NPZ computed by existing Stage-1 R3 algorithm; never a scalar')
    p.add_argument('--safety-distance',type=float,help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.safety_distance is not None:
        p.error('--safety-distance scalar fallback was removed. Supply the exact Stage-1 local d(x) field instead.')
    cleanup_from_manifest(args.manifest,colmap_images=args.colmap_images,no_export=args.no_export,
                          chunk_size=args.chunk_size,config_override=args.config,
                          surface_only=args.surface_only,keep_diagnostics=args.keep_diagnostics,
                          allow_inferred_guard_export=args.allow_inferred_guard_export,
                          surface_face_mask=args.surface_face_mask,
                          local_safety_field=args.stage1_local_safety_field)


if __name__=='__main__':
    main()
