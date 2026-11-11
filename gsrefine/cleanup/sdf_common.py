"""Shared manifest, camera and Gaussian SDF math for the canonical Direct-R3 pipeline.

Moved verbatim in behavior from the retired local-Poisson ``direct_sdf.py``.
No local SfM Poisson rebuilding, trusted triangle filtering, or alternative
cleaning path exists here. This module is imported by the R3 geometry workflow.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np



def _path(value):
    p = Path(value).expanduser().resolve(strict=True)
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def _read_json(path):
    return json.loads(_path(path).read_text(encoding='utf8'))


def _load_stage1_manifest(path):
    """Verify immutable Stage-1 R3 provenance before any Stage-2 operation."""
    from gsrefine.roi.core import file_sha256
    manifest_file = _path(path)
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
    """GSPL log(std) longest-axis, 3-sigma radius, global P10--P90 trimmed mean."""
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
    """SDF positive outside. Keep unknown signed distances conservatively."""
    distances = np.asarray(distances)
    if not (np.isfinite(delta) and delta>=0 and np.isfinite(epsilon) and epsilon>=0):
        raise ValueError('Invalid SDF threshold or epsilon')
    finite = np.isfinite(distances)
    remove = finite & (distances > delta + epsilon)
    uncertain = ~finite
    return ~remove, remove, uncertain


def _local_basis(up):
    from gsrefine.geometry.orientation import horizontal_basis
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


def _camera_path(run_dir, roi_payload, config_file, override=None):
    import yaml
    from gsrefine.roi.core import file_sha256
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
