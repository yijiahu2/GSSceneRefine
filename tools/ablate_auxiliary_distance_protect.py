"""GSSceneRefine: distance-only ablation for ALL auxiliary R3 Poisson components.

Input: V2 ROI barrier parity classification (read-only), Stage-1 native GSPL Gaussian PLY
and the 19-component *local R3 auxiliary* Poisson mesh (not the full-scene Poisson).

Protect = existing outside-to-remove AND distance_to_auxiliary <= original_delta.
No auxiliary inside/outside or SfM support check. Output PLY retains every original
Gaussian vertex property exactly for SuperSplat and GSPL.

Run: python -m tools.ablate_auxiliary_distance_protect --help
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

REVISION = 'GSSceneRefine.aux_distance_protection_ablation.v1'


def _log(s):
    print('[aux-distance-ablation] ' + str(s), flush=True)


def _read_json(path):
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def _native_json(x):
    if isinstance(x, dict):
        return {str(k): _native_json(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)):
        return [_native_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return _native_json(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def _write_json_atomic(path, obj):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(_native_json(obj), f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _component_summary(mesh):
    """Topology check of local R3 auxiliary triangle mesh; never edits geometry."""
    import trimesh
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError('Need a nonempty auxiliary TRIANGLE mesh')
    parts = trimesh.graph.connected_components(mesh.face_adjacency,
                                                nodes=np.arange(len(mesh.faces)), min_len=1)
    counts = sorted((int(len(p)) for p in parts), reverse=True)
    return {'components':len(counts), 'faces':int(len(mesh.faces)),
            'vertices':int(len(mesh.vertices)), 'faces_per_component':counts}


def _load_auxiliary_mesh(path, expected_components):
    import trimesh
    mesh = trimesh.load(str(path), file_type='ply', process=False, maintain_order=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError('Auxiliary PLY must contain exactly one triangle mesh')
    xyz = np.asarray(mesh.vertices, np.float64)
    faces = np.asarray(mesh.faces, np.int64)
    if len(xyz)==0 or len(faces)==0 or not np.isfinite(xyz).all():
        raise ValueError('Auxiliary Poisson is empty/nonfinite')
    if faces.min() < 0 or faces.max() >= len(xyz):
        raise ValueError('Auxiliary mesh has invalid face indices')
    summary = _component_summary(mesh)
    if expected_components > 0 and summary['components'] != expected_components:
        raise ValueError('Expected %d auxiliary components, got %d. Check that you selected '
                         'poisson_roi_auxiliary.ply from THIS ROI run, not global Poisson.'
                         % (expected_components, summary['components']))
    return mesh, summary


def _distance_scene(mesh):
    # Reuse the SAME Open3D triangle-scene conversion as Stage-2 main-body SDF.
    # This avoids tensor triangle-index dtype incompatibilities across Open3D versions.
    import open3d as o3d
    from gsrefine.cleanup.sdf_common import _raycast
    return _raycast(o3d, mesh), o3d


def _protect_by_distance(points, delta, batch_size, distance_query, progress=None):
    """Exact point-to-triangle unsigned distances for only outside-to-remove points."""
    points = np.asarray(points, np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Invalid Gaussian candidate positions')
    if not np.isfinite(delta) or delta <= 0 or batch_size < 1:
        raise ValueError('Invalid delta or batch size')
    protect = np.zeros(len(points), bool)
    min_dist = np.empty(len(points), np.float32)
    for start in range(0, len(points), batch_size):
        end = min(start + batch_size, len(points))
        d = np.asarray(distance_query(points[start:end]), np.float32).reshape(-1)
        if len(d) != end-start or not np.isfinite(d).all() or np.any(d < 0):
            raise ValueError('Invalid distance-to-auxiliary triangle result')
        min_dist[start:end] = d
        protect[start:end] = d.astype(np.float64) <= delta
        if progress is not None:
            progress(end,len(points))
    return protect,min_dist


def _validated_parity_masks(parity_dir, n_gaussians, source_path):
    report_path = parity_dir / 'roi_parity_report.json'
    masks_path = parity_dir / 'classification_masks.npz'
    report = _read_json(report_path)
    if not str(report.get('schema','')).startswith('GSSceneRefine.roi_barrier_parity_sdf.v'):
        raise ValueError('Expected ROI barrier parity classification report')
    p = Path(report['inputs']['original_R3_Gaussian']).expanduser().resolve(strict=True)
    if p != source_path.resolve():
        raise ValueError('V2 classification references a DIFFERENT Stage-1 Gaussian source')
    t = report.get('threshold', {})
    delta = float(t['delta'])
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('Invalid original V2 delta')
    with np.load(masks_path, allow_pickle=False) as npz:
        required={'remove','over_threshold','distance_main'}
        if not required.issubset(npz.files):
            raise ValueError('V2 masks missing: ' + str(required - set(npz.files)))
        remove = np.asarray(npz['remove'])
        over = np.asarray(npz['over_threshold'])
        if remove.dtype != np.bool_ or over.dtype != np.bool_:
            raise ValueError('Expected boolean remove/over_threshold masks')
        if remove.shape!=(n_gaussians,) or over.shape!=(n_gaussians,) or npz['distance_main'].shape!=(n_gaussians,):
            raise ValueError('V2 classification mask length mismatch')
        if np.any(remove & ~over):
            raise ValueError('V2 remove mask includes below-threshold Gaussians')
        if 'outside_confident' in npz.files and not np.array_equal(remove,npz['outside_confident']):
            raise ValueError('V2 remove mask differs from outside_confident')
        if 'keep' in npz.files and not np.array_equal(~remove,npz['keep']):
            raise ValueError('V2 remove/keep masks conflict')
    counts = report.get('counts',{})
    if int(remove.sum()) != int(counts.get('candidate_outside_remove',-1)):
        raise ValueError('V2 candidate count does not match original report')
    if int(over.sum()) != int(counts.get('over_distance_threshold',-1)):
        raise ValueError('V2 over-threshold count does not match original report')
    if report.get('status') not in ('CLASSIFICATION_ONLY_PREVIEW', 'CLEANED_PLY_EXPORTED'):
        # accept only known diagnostic and release statuses, never silently use a failed run
        raise ValueError('Unrecognized V2 classification status: ' + str(report.get('status')))
    return remove, over, delta, report, report_path, masks_path


def _checked_export(source, mask, dest):
    """Full native 3DGS PLY: source attributes are byte-for-byte preserved."""
    from gsrefine.roi.core import GaussianPLY
    needed = {'x','y','z','f_dc_0','f_dc_1','f_dc_2','opacity',
              'scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3'}
    missing = needed-set(source.dtype.names)
    if missing:
        raise ValueError('Cannot guarantee native SuperSplat Gaussian PLY: missing '+str(sorted(missing)))
    dest = Path(dest)
    temp = dest.with_name('.' + dest.stem + '.writing.tmp.ply')
    try:
        written = source.export(mask,temp)
        verify = GaussianPLY(temp)
        if written != int(mask.sum()) or verify.vertex_count != written or verify.dtype != source.dtype:
            raise RuntimeError('PLY export schema/count verification failed')
        if temp.stat().st_size != verify.data_offset+written*verify.dtype.itemsize:
            raise RuntimeError('PLY export data size verification failed')
        # Sample original binary vertex records, not just positions or RGB.
        if written:
            probes = np.unique(np.linspace(0,written-1,min(32,written),dtype=np.int64))
            original_indices = np.flatnonzero(mask)[probes]
            if verify.arr[probes].tobytes() != source.arr[original_indices].tobytes():
                raise RuntimeError('Gaussian attributes changed during export')
        del verify
        os.replace(temp,dest)
        return written
    finally:
        temp.unlink(missing_ok=True)


def run(*, manifest, parity_dir, preseal_report, output_dir, auxiliary_ply=None,
        expected_components=19, batch_size=30000):
    from gsrefine.cleanup.sdf_common import _load_stage1_manifest
    from gsrefine.roi.core import GaussianPLY

    manifest = Path(manifest).expanduser().resolve(strict=True)
    parity_dir = Path(parity_dir).expanduser().resolve(strict=True)
    preseal_report = Path(preseal_report).expanduser().resolve(strict=True)
    preseal = _read_json(preseal_report)
    manifest_path, upstream, inputs, roi_run = _load_stage1_manifest(manifest)
    source_path = inputs['gaussian_ply']
    output = Path(output_dir).expanduser().resolve()
    allowed = (roi_run / 'cleanup').resolve()
    if not output.is_relative_to(allowed) or output==allowed:
        raise ValueError('--output-dir must be a NEW subdirectory inside this ROI run/cleanup')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refuse to overwrite nonempty output: '+str(output))
    if not parity_dir.is_relative_to(allowed):
        raise ValueError('--parity-dir must belong to this same ROI run')
    if not preseal_report.is_relative_to(allowed):
        raise ValueError('--preseal-report must belong to this same ROI run')

    auxiliary_ply = (Path(auxiliary_ply).expanduser() if auxiliary_ply else
                     preseal_report.parent/'poisson_roi_auxiliary.ply').resolve(strict=True)
    if not auxiliary_ply.is_relative_to(allowed):
        raise ValueError('Auxiliary Poisson must come from this ROI run')
    model = GaussianPLY(source_path)
    candidate_mask, threshold_mask, delta, old_report, old_report_path, mask_path = _validated_parity_masks(
        parity_dir, model.vertex_count, source_path)
    _log(f'Loaded V2 candidate mask: {int(candidate_mask.sum()):,} outside-to-remove Gaussians')
    _log(f'Original adaptive delta={delta:.12f} (unchanged)')
    mesh, geometry = _load_auxiliary_mesh(auxiliary_ply, expected_components)
    _log(f'Original auxiliary R3 Poisson: {geometry["components"]} components, {geometry["faces"]} faces')
    scene, o3d = _distance_scene(mesh)
    ids = np.flatnonzero(candidate_mask)
    selected = model.selected_xyz(ids)
    last_log = [-1]
    def progress(n,total):
        pct = (n * 10) // max(total,1)
        if pct!=last_log[0] or n==total:
            last_log[0]=pct
            _log(f'Actual triangle distance: {n:,}/{total:,}')
    def distance_query(xyz):
        q = o3d.core.Tensor(np.ascontiguousarray(xyz,dtype=np.float32),dtype=o3d.core.Dtype.Float32)
        return scene.compute_distance(q).numpy()
    protected_subset, distances = _protect_by_distance(selected, delta, batch_size, distance_query,progress)
    protected = np.zeros(model.vertex_count, dtype=bool)
    protected[ids[protected_subset]]=True
    remaining = candidate_mask & ~protected
    if int(protected.sum()) + int(remaining.sum()) != int(candidate_mask.sum()):
        raise RuntimeError('Protected and remaining subsets do not cover exactly the V2 outside candidates')
    output.mkdir(parents=True,exist_ok=True)
    native_ply = output/'aux_protected_gaussians.ply'
    count = _checked_export(model,protected,native_ply)
    np.save(output/'aux_protected_roi_indices.npy', ids[protected_subset].astype(np.int64))
    np.savez_compressed(output/'aux_distance_protection_masks.npz',
                        protected=protected, remaining_outside=remaining,
                        candidate_outside=candidate_mask, delta=np.float64(delta),
                        candidate_roi_indices=ids,
                        candidate_aux_distances=distances)
    report = {
        'schema':REVISION,'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'DISTANCE_ONLY_PROTECTION_ABLATION_NOT_TRUTH_VALIDATED',
        'source':{
            'manifest':str(manifest),
            'source_gaussian_ply':str(source_path),
            'source_gaussian_sha256':_sha256(source_path),
            'V2_report':str(old_report_path),
            'V2_masks':str(mask_path),
            'V2_masks_sha256':_sha256(mask_path),
            'auxiliary_poisson_ply':str(auxiliary_ply),
            'auxiliary_poisson_sha256':_sha256(auxiliary_ply),
            'preseal_report':str(preseal_report),
        },
        'geometry':geometry,
        'threshold':{'delta':delta,
                     'candidate_condition':'V2 outside AND original_main_distance > delta+epsilon',
                     'protection_condition':'exact unsigned triangle distance to ANY auxiliary component <= delta',
                     'auxiliary_inside_outside_test':False,
                     'sfm_or_semantic_filter':False},
        'counts':{'all_roi_gaussians':int(model.vertex_count),
                  'original_V2_over_threshold':int(threshold_mask.sum()),
                  'original_V2_outside_candidates':int(candidate_mask.sum()),
                  'protected_by_19_aux':count,
                  'still_removable_V2_outside':int(remaining.sum()),
                  'protected_fraction_of_V2_outside':float(count/max(int(candidate_mask.sum()),1))},
        'distance_stats_on_original_V2_candidates':{
            'minimum':float(np.min(distances)) if len(distances) else None,
            'median':float(np.median(distances)) if len(distances) else None,
            'p95':float(np.quantile(distances,.95)) if len(distances) else None,
            'maximum':float(np.max(distances)) if len(distances) else None},
        'outputs':{
            'supersplat_native_gaussian_ply':str(native_ply),
            'original_roi_indices':str(output/'aux_protected_roi_indices.npy'),
            'ablation_masks':str(output/'aux_distance_protection_masks.npz')},
        'notes':'The protected PLY contains complete native Gaussian records. SuperSplat can visualize the selected Gaussians directly. No original PLY changed. Auxiliary sheet validity not confirmed.',
    }
    _write_json_atomic(output/'aux_protection_report.json',report)
    _log(f'SuperSplat-ready: {native_ply} ({count:,} complete Gaussians)')
    _log(f'V2 candidate remaining after protection: {int(remaining.sum()):,}')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--parity-dir',required=True,help='V2 classification directory with masks+report')
    parser.add_argument('--preseal-report',required=True,help='R3 main/aux geometry preseal report')
    parser.add_argument('--aux-poisson',help='Default: preseal report directory/poisson_roi_auxiliary.ply')
    parser.add_argument('--expected-components',type=int,default=19)
    parser.add_argument('--batch-size',type=int,default=30000)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args(argv)
    run(manifest=args.manifest, parity_dir=args.parity_dir,
        preseal_report=args.preseal_report, auxiliary_ply=args.aux_poisson,
        output_dir=args.output_dir, expected_components=args.expected_components,
        batch_size=args.batch_size)


if __name__=='__main__':
    main()
