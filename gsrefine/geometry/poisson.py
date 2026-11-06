#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent Stage-1 geometry build: cleaned SfM NPZ -> raw Screened Poisson.

The already validated Songzi Poisson should be reused for initial comparisons.
This script is for reproducible new-scene preparation and cleaning ablations.
It NEVER modifies input SfM or overwrites existing meshes. Do not confuse the
result with the earlier first-stage Poisson until visual/normal checks pass.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from gsrefine.roi.core import file_sha256, load_sfm


def make_poisson(sfm_path:Path, output_dir:Path, depth:int=9,
                 normal_k:int=30, orientation_k:int=30,
                 radius_scale:float=4.0, scale:float=1.1,
                 density_quantile:float=0.0):
    import open3d as o3d
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Output folder must be empty: {output_dir}')
    output_dir.mkdir(parents=True,exist_ok=True)
    xyz,err,tracks,fields=load_sfm(sfm_path)
    if len(xyz)<100:raise ValueError('Too few SfM points for surface reconstruction')
    tree=cKDTree(xyz)
    distances,_=tree.query(xyz,k=min(7,len(xyz)),workers=-1)
    spacing=float(np.median(distances[:,-1]))
    extent=float(np.linalg.norm(xyz.max(axis=0)-xyz.min(axis=0)))
    radius=max(float(radius_scale)*spacing,extent*1e-5)
    pc=o3d.geometry.PointCloud()
    pc.points=o3d.utility.Vector3dVector(xyz)
    with np.load(sfm_path,allow_pickle=False) as dat:
        if 'rgb' in dat and dat['rgb'].shape==(len(xyz),3):
            rgb=dat['rgb'].astype(float)
            if rgb.max()>1:rgb/=255.0
            pc.colors=o3d.utility.Vector3dVector(np.clip(rgb,0,1))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius,max_nn=max(10,normal_k)))
    # A consistent normal field is necessary for Poisson. On large disconnected
    # outdoor scenes MST orientation can be expensive or wrong. Stop if it fails;
    # do not silently output an untrustworthy reference surface.
    pc.orient_normals_consistent_tangent_plane(max(10,orientation_k))
    surface,density=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc,depth=int(depth),width=0,scale=float(scale),linear_fit=False)
    raw=output_dir/'04_poisson_raw.ply'
    if not o3d.io.write_triangle_mesh(str(raw),surface,write_ascii=False):
        raise RuntimeError('Failed writing Poisson PLY')
    density=np.asarray(density,dtype=float)
    np.save(output_dir/'poisson_vertex_density.npy',density)
    report={
      'script':'prepare_geometry.py','input_cleaned_sfm':str(sfm_path.resolve()),
      'input_sha256':file_sha256(sfm_path),
      'output_raw_poisson':str(raw.resolve()), 'output_sha256':file_sha256(raw),
      'sfm_points':int(len(xyz)), 'mesh_vertices':len(surface.vertices),
      'mesh_faces':len(surface.triangles),'knn6_spacing_median':spacing,
      'normal_radius':radius,'normal_k':normal_k,'orientation_k':orientation_k,
      'poisson_depth':depth,'poisson_scale':scale,'fit_info_available':fields,
      'warning':'Check normals, artificial surfaces, isolated components and scene completeness before using as trusted reference.'}
    if density_quantile>0:
        if not 0<density_quantile<1:raise ValueError('density_quantile must be in [0,1)')
        threshold=float(np.quantile(density,density_quantile))
        keep=density>=threshold
        restricted=o3d.geometry.TriangleMesh(surface)
        restricted.remove_vertices_by_mask(~keep)
        cropped=output_dir/'05_poisson_density_preview.ply'
        o3d.io.write_triangle_mesh(str(cropped),restricted,write_ascii=False)
        report['density_preview']={'path':str(cropped),'quantile':density_quantile,
                                   'warning':'For visualization only; core uses original 04_poisson_raw.ply.'}
    (output_dir/'reconstruction_report_poisson.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clean-sfm',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--depth',type=int,default=9)
    p.add_argument('--normal-k',type=int,default=30)
    p.add_argument('--orientation-k',type=int,default=30)
    p.add_argument('--normal-radius-scale',type=float,default=4.0)
    p.add_argument('--poisson-scale',type=float,default=1.1)
    p.add_argument('--density-preview-quantile',type=float,default=0.0)
    a=p.parse_args()
    if not 6<=a.depth<=12:raise SystemExit('depth must be 6..12')
    report=make_poisson(a.clean_sfm,a.out,a.depth,a.normal_k,a.orientation_k,
                       a.normal_radius_scale,a.poisson_scale,a.density_preview_quantile)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
