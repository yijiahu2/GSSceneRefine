"""Review SfM-estimated gravity candidate, with optional Poisson normal diagnostics.

Run: python run.py orientation --config configs/songziguanyin.yaml --check-poisson
No model / mesh coordinates are modified.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from gsrefine.geometry.orientation import read_or_estimate, read_colmap_poses, infer_colmap_images
from gsrefine.roi_init.results import write_atomic

ROOT=Path(__file__).resolve().parents[2]


def describe_camera_path(centers,up):
    """Trajectory flatness is *only* diagnostic, not a gravity assumption."""
    points=np.asarray(centers,float)
    u=np.asarray(up,float)
    cov=np.cov(points.T)
    val,vec=np.linalg.eigh(cov)
    normal=vec[:,0]
    return {'centers':len(points),
            'trajectory_pca_plane_normal_abs_dot_up':float(abs(normal@u)),
            'trajectory_pca_eigenvalues':val.tolist(),
            'trajectory_height_spread_model_units_p90_p10':float(np.ptp(np.percentile(points@u,[10,90]))),
            'warning':'UAV flight paths need not lie on a horizontal plane. Do not use camera-trajectory PCA as hard ground truth.'}


def describe_poisson_mesh(path,up):
    """Area weighted face-normal diagnostic; cliff normals are not gravity."""
    import trimesh
    mesh=trimesh.load(str(path),force='mesh',process=False)
    if len(mesh.faces)==0:raise ValueError('Empty Poisson mesh')
    areas=np.asarray(mesh.area_faces,float)
    normals=np.asarray(mesh.face_normals,float)
    cos=np.abs(normals@np.asarray(up,float))
    valid=np.isfinite(cos)&np.isfinite(areas)&(areas>0)
    areas,cos=areas[valid],cos[valid]
    area=areas.sum()
    if area<=0:raise ValueError('Poisson face area invalid')
    return {'faces':int(len(areas)),
            'area_fraction_near_horizontal_surface_25deg':float(areas[cos>=np.cos(np.deg2rad(25))].sum()/area),
            'area_fraction_near_vertical_surface_25deg':float(areas[cos<=np.sin(np.deg2rad(25))].sum()/area),
            'warning':'Inclined rock/cliffs/Poisson hallucinations can dominate normals. Weak support does NOT prove the camera estimate wrong.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/songziguanyin.yaml')
    p.add_argument('--check-poisson',action='store_true',help='Read optional original open Poisson mesh for supporting diagnostics')
    args=p.parse_args(argv)
    config=yaml.safe_load(Path(args.config).read_text(encoding='utf8'))
    scene=config['scene']['name']
    ori=config.get('roi_init',{}).get('orientation',{})
    frame_id=ori.get('frame_id',f'{scene}_colmap_original')
    result=read_or_estimate(scene,frame_id,ori,config,ROOT/'runs'/'orientation')
    out={'scene':scene,'frame_id':frame_id,'estimate':result}
    if result['method']!='manual_explicit':
        centers,_,_=read_colmap_poses(infer_colmap_images(scene,config))
        out['camera_path_diagnostic']=describe_camera_path(centers,result['up_vector'])
    if args.check_poisson:
        source=(config.get('roi_init',{}).get('poisson_ply') or
                config.get('paths',{}).get('poisson_raw'))
        if source and Path(source).is_file():
            out['poisson_normal_diagnostic']=describe_poisson_mesh(source,result['up_vector'])
        else:out['poisson_normal_diagnostic']={'warning':'Original Poisson file missing; no geometry validation performed'}
    report=ROOT/'runs'/'orientation'/f'{scene}_{frame_id}_diagnostics.json'
    write_atomic(report,out)
    print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
    print('[orientation] diagnostic report:',report)
    if not result['accepted']:
        raise SystemExit('[orientation] Candidate is uncertain; do not create a confirmed ROI from it.')

if __name__=='__main__':main()
