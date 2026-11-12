"""Reusable Stage-2 auxiliary Poisson distance protection, independent of scene.

All disconnected auxiliary triangle components are used as unsigned-distance
support surfaces, never as ROI barriers. Original native 3DGS Gaussian vertex
bytes are preserved when generating preview and final models.
"""
from __future__ import annotations

from pathlib import Path
import json
import os
import numpy as np


def load_validated_parity(parity_dir, n_gaussians, source_gaussian):
    """Read a completed V2/V3 parity result; formal V3-only check is in onepass."""
    parity_dir = Path(parity_dir).expanduser().resolve(strict=True)
    report_file = parity_dir/'roi_parity_report.json'
    masks_file = parity_dir/'classification_masks.npz'
    report = json.loads(report_file.read_text(encoding='utf8'))
    if report.get('schema') not in ('GSSceneRefine.roi_barrier_parity_sdf.v2',
                                    'GSSceneRefine.roi_barrier_parity_sdf.v3'):
        raise ValueError('Expected a complete GSSceneRefine V2/V3 parity report')
    if report.get('status') not in ('CLASSIFICATION_ONLY_PREVIEW', 'CLEANED_PLY_EXPORTED',
                                    'CLEANED_EXPERIMENTAL_EXTERIOR_APPROVED'):
        raise ValueError('Parity report was not successfully classified')
    source = Path(report['inputs']['original_R3_Gaussian']).expanduser().resolve(strict=True)
    if source != Path(source_gaussian).expanduser().resolve(strict=True):
        raise ValueError('Parity result belongs to a different Gaussian file')
    delta = float(report['threshold']['delta'])
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('Invalid adaptive distance threshold')
    with np.load(masks_file, allow_pickle=False) as z:
        required = {'remove', 'over_threshold', 'distance_main', 'keep'}
        if not required.issubset(z.files):
            raise ValueError('Incomplete V2 parity classification masks')
        remove = np.asarray(z['remove'])
        over = np.asarray(z['over_threshold'])
        keep = np.asarray(z['keep'])
        distances = np.asarray(z['distance_main'])
        if (remove.dtype != np.bool_ or over.dtype != np.bool_ or keep.dtype != np.bool_
                or remove.shape != (n_gaussians,) or over.shape != remove.shape
                or keep.shape != remove.shape or distances.shape != remove.shape):
            raise ValueError('V2 parity mask dtype / shape mismatch')
        if np.any(remove & ~over) or not np.array_equal(keep,~remove):
            raise ValueError('V2 parity mask violates outside/threshold restrictions')
        if 'outside_confident' in z.files and not np.array_equal(remove, z['outside_confident']):
            raise ValueError('Parity outside classification and remove mask disagree')
    counts = report['counts']
    if (int(remove.sum()) != int(counts['candidate_outside_remove'])
            or int(over.sum()) != int(counts['over_distance_threshold'])
            or int(n_gaussians) != int(counts['all_gaussians'])):
        raise ValueError('V2 parity report counts differ from stored masks')
    return remove, over, delta, report, report_file, masks_file


def load_auxiliary_mesh(path):
    """Load all available auxiliary face components, no hardcoded component count."""
    import trimesh
    mesh = trimesh.load(str(path), file_type='ply', process=False, maintain_order=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError('Expected a nonempty auxiliary triangle mesh')
    v = np.asarray(mesh.vertices, np.float64)
    f = np.asarray(mesh.faces, np.int64)
    if (not len(v) or not np.isfinite(v).all() or f.min()<0 or f.max()>=len(v)):
        raise ValueError('Invalid auxiliary triangle geometry')
    parts = trimesh.graph.connected_components(mesh.face_adjacency,
                                                 nodes=np.arange(len(f)), min_len=1)
    counts = sorted((int(len(p)) for p in parts), reverse=True)
    return mesh, {'components':len(counts), 'faces':len(f), 'vertices':len(v),
                  'faces_per_component':counts}


def distance_protection(model, candidate_mask, delta, mesh_path, *, chunk_size=30000,
                        progress=None):
    """Protect d(center, ANY auxiliary triangle) <= delta, no inside/outside test."""
    if chunk_size < 1 or not np.isfinite(delta) or delta <= 0:
        raise ValueError('Invalid distance protection parameters')
    candidate_mask = np.asarray(candidate_mask, bool)
    if candidate_mask.shape != (model.vertex_count,):
        raise ValueError('Invalid candidate mask length')
    ids = np.flatnonzero(candidate_mask)
    protected = np.zeros(model.vertex_count, dtype=bool)
    if mesh_path is None:
        return protected, {'components':0, 'faces':0, 'vertices':0,
                           'faces_per_component':[]},np.full(len(ids),np.inf,np.float32)
    mesh, info = load_auxiliary_mesh(Path(mesh_path))
    if not len(ids):
        return protected, info, np.empty(0,np.float32)
    from gsrefine.cleanup.sdf_common import _raycast
    import open3d as o3d
    scene = _raycast(o3d, mesh)
    distances = np.empty(len(ids),np.float32)
    for start in range(0,len(ids),chunk_size):
        end=min(start+chunk_size,len(ids))
        xyz = model.selected_xyz(ids[start:end])
        q = o3d.core.Tensor(np.ascontiguousarray(xyz,dtype=np.float32),
                            dtype=o3d.core.Dtype.Float32)
        d = np.asarray(scene.compute_distance(q).numpy(), np.float32).reshape(-1)
        if len(d)!=end-start or not np.isfinite(d).all() or np.any(d<0):
            raise ValueError('Open3D returned invalid point-to-auxiliary-triangle distance')
        distances[start:end]=d
        protected[ids[start:end]]=d.astype(np.float64)<=delta
        if progress is not None:
            progress(end,len(ids))
    return protected, info, distances


def export_native_gaussian(model, mask, destination):
    """Binary-le Gaussian PLY: all original SH, opacity, rotations preserved."""
    from gsrefine.roi.core import GaussianPLY
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (model.vertex_count,):
        raise ValueError('Gaussian export mask length does not match source model')
    required={'x','y','z','f_dc_0','f_dc_1','f_dc_2','opacity',
              'scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3'}
    missing=required-set(model.dtype.names)
    if missing:
        raise ValueError('SuperSplat native Gaussian PLY has missing fields: '+str(sorted(missing)))
    target=Path(destination)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        raise FileExistsError('Refuse overwrite existing Gaussian PLY: '+str(target))
    temp=target.with_name('.'+target.stem+'.writing.tmp.ply')
    try:
        count=model.export(mask,temp)
        reread=GaussianPLY(temp)
        if count != int(mask.sum()) or reread.vertex_count!=count or reread.dtype != model.dtype:
            raise RuntimeError('Native 3DGS PLY schema/count verification failed')
        if temp.stat().st_size!=reread.data_offset+count*reread.dtype.itemsize:
            raise RuntimeError('Native 3DGS PLY binary byte count failed')
        if count:
            probe=np.unique(np.linspace(0,count-1,min(count,32),dtype=int))
            original_idx=np.flatnonzero(mask)[probe]
            if reread.arr[probe].tobytes()!=model.arr[original_idx].tobytes():
                raise RuntimeError('Native Gaussian attributes differ from original model')
        del reread
        os.replace(temp,target)
        return count
    finally:
        temp.unlink(missing_ok=True)
