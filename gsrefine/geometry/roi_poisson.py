"""Local Poisson reconstruction from Stage-1 ROI SfM (core + extra context shell).

Produces only one observed mesh: poisson_roi.ply. No secondary 05/06/07
meshes; no changes to the global trusted Stage-1 Poisson.

The reconstruction band is wider than the expected target: this mesh is a
candidate geometric reference, NOT automatically a certified target surface.
Always inspect and validate topology/camera orientation before direct SDF.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

POISSON_REVISION='local_sfm_reconstruction_v1_20261107'
DEFAULT_POISSON={'depth':8,'scale':1.1,'linear_fit':False,'n_threads':-1,
                 'normal_knn':20,'normal_consistency_knn':20}


def file_sha(path):
    sha=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):
            sha.update(chunk)
    return sha.hexdigest()


def read_local_sfm(path):
    with np.load(path,allow_pickle=False) as source:
        if 'xyz' not in source or 'in_core_roi' not in source:
            raise ValueError('ROI SfM needs xyz and in_core_roi masks')
        xyz=np.asarray(source['xyz'],dtype=np.float64)
        core=np.asarray(source['in_core_roi'],dtype=np.bool_)
        rgb=np.asarray(source['rgb']) if 'rgb' in source else None
        if (xyz.ndim!=2 or xyz.shape[1]!=3 or
            len(xyz)!=len(core) or not np.isfinite(xyz).all()):
            raise ValueError('Invalid ROI SfM data or coordinate frame')
        if len(xyz)<100 or core.sum()<100:
            raise ValueError(f'Insufficient local SfM points: build={len(xyz)}, core={int(core.sum())}')
        if rgb is not None and (rgb.shape != xyz.shape or not np.isfinite(rgb).all()):
            raise ValueError('Invalid ROI SfM RGB')
    return xyz,core,rgb


def estimate_normals(o3d,cloud,core_xyz,params):
    """Historical consistent-outward method; orientation is a heuristic.

    Do not assume global COLMAP Z is up. The Stage-2 camera validation remains
    mandatory before publishing a cleaned model.
    """
    n=len(cloud.points)
    knn=min(int(params['normal_knn']),n-1)
    consistent_knn=min(int(params['normal_consistency_knn']),n-1)
    if knn<3 or consistent_knn<3:
        raise ValueError('Not enough points for local surface normals')
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    method='consistent_outward'
    try:
        cloud.orient_normals_consistent_tangent_plane(consistent_knn)
    except RuntimeError:
        method='radial_fallback'
    xyz=np.asarray(cloud.points)
    center=np.median(core_xyz,axis=0)
    outward=xyz-center[None,:]
    normals=np.asarray(cloud.normals).copy()
    if method=='consistent_outward':
        use=np.linalg.norm(outward,axis=1)>1e-10
        if use.any() and np.median(np.einsum('ij,ij->i',normals[use],outward[use]))<0:
            normals*=-1
    else:
        normals[np.einsum('ij,ij->i',normals,outward)<0]*=-1
    cloud.normals=o3d.utility.Vector3dVector(normals)
    cloud.normalize_normals()
    return {'method':method,'normal_knn':knn,'consistency_knn':consistent_knn,
            'orientation':'heuristic outward relative to local SfM core median; downstream signed volume and cameras required'}


def _write_report(path,obj):
    path=Path(path)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.stem+'_',suffix='.json',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf8') as f:
            json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
            f.write('\n')
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def build_local_poisson(roi_sfm,base_dir,*,parameters=None,o3d=None):
    """Build or verify a hash-addressed, immutable local Poisson cache."""
    p={**DEFAULT_POISSON,**(parameters or {})}
    if (not 5<=int(p['depth'])<=11 or float(p['scale'])<=1 or
        int(p['normal_knn'])<3 or int(p['normal_consistency_knn'])<3):
        raise ValueError('Invalid local Screened-Poisson reconstruction parameters')
    payload=json.dumps(p,sort_keys=True)
    source_sha=file_sha(roi_sfm)
    identity=hashlib.sha256((POISSON_REVISION+source_sha+payload).encode()).hexdigest()
    folder=Path(base_dir)/identity[:24]
    mesh_file=folder/'poisson_roi.ply'
    report_file=folder/'poisson_roi_report.json'
    if folder.exists():
        if not (mesh_file.is_file() and report_file.is_file()):
            raise FileExistsError('Incomplete local Poisson cache; refusing overwrite: '+str(folder))
        report=json.loads(report_file.read_text(encoding='utf8'))
        if (report.get('signature')!=identity or
            report.get('source_sfm_sha256')!=source_sha or
            report.get('poisson_sha256')!=file_sha(mesh_file)):
            raise ValueError('Local Poisson cache provenance changed; refusing reuse')
        return mesh_file.resolve(),report,True
    if o3d is None:
        import open3d as o3d
    xyz,core,rgb=read_local_sfm(roi_sfm)
    cloud=o3d.geometry.PointCloud()
    cloud.points=o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        colors=rgb.astype(np.float64)
        if np.max(colors)>1.:
            colors/=255.
        cloud.colors=o3d.utility.Vector3dVector(np.clip(colors,0.,1.))
    orient=estimate_normals(o3d,cloud,xyz[core],p)
    mesh,density=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud,depth=int(p['depth']),scale=float(p['scale']),
        linear_fit=bool(p['linear_fit']),n_threads=int(p['n_threads']))
    if len(mesh.triangles)==0:
        raise RuntimeError('Local SfM Poisson produced no faces')
    from trimesh import Trimesh
    tm=Trimesh(vertices=np.asarray(mesh.vertices),faces=np.asarray(mesh.triangles),process=False)
    bbox=np.asarray(tm.bounds).tolist()
    folder.mkdir(parents=True,exist_ok=False)
    temp=folder/'poisson_roi.writing.ply'
    try:
        if not o3d.io.write_triangle_mesh(str(temp),mesh,write_ascii=False):
            raise RuntimeError('Failed writing local Poisson to disk')
        os.replace(temp,mesh_file)
        report={'schema':'GSSceneRefine.roi_poisson.v1',
                'signature':identity,'revision':POISSON_REVISION,
                'source_roi_sfm':str(Path(roi_sfm).resolve()),
                'source_sfm_sha256':source_sha,
                'reconstruction_settings':p,'normals':orient,
                'observed_faces':int(len(mesh.triangles)),
                'observed_vertices':int(len(mesh.vertices)),
                'build_sfm_count':int(len(xyz)),'core_sfm_count':int(core.sum()),
                'density_p05':float(np.quantile(np.asarray(density),.05)),
                'bounds':bbox,'poisson_sha256':file_sha(mesh_file),
                'is_certified_target_surface':False,
                'warning':'Build-context SfM can introduce non-target mesh. Inspect and validate trusted target before SDF pruning.'}
        _write_report(report_file,report)
    finally:
        temp.unlink(missing_ok=True)
    return mesh_file.resolve(),report,False
