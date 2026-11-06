"""UAV COLMAP world-up ESTIMATION, never a silent +Z convention.

For (most) roll-stabilised UAV images, each camera-right axis r_i lies in
approximately horizontal planes. Estimate up as the smallest-eigenvalue
unit eigenvector of sum_i w_i r_i r_i^T; IRLS downweights rolled images.
The sign is deduced only when there is sufficient evidence that the optical
axis points downward in a majority of images. The camera-centre trajectory
and geometry are NOT assumed planar (important for cliffs/oblique flights).

This is a conditional inference, not an absolute gravity measurement. Geo-
referenced ENU/GNSS/IMU priors should take precedence when independently
validated. Weak support FAILS CLOSED; the user must not simply tick a checkbox.
"""
from __future__ import annotations
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def _read_images_bin(path):
    cameras=[]
    with open(path,'rb') as file:
        head=file.read(8)
        if len(head)!=8:raise ValueError('Truncated COLMAP images.bin')
        count=struct.unpack('<Q',head)[0]
        if not 1<=count<=2_000_000:raise ValueError('Implausible COLMAP image count')
        for _ in range(count):
            record=file.read(64)
            if len(record)!=64:raise ValueError('Truncated COLMAP image pose')
            row=struct.unpack('<i4d3di',record)
            q=np.array(row[1:5],float);t=np.array(row[5:8],float)
            for idx in range(8193):
                c=file.read(1)
                if c==b'\0':break
                if not c:raise ValueError('Truncated COLMAP image name')
            else:raise ValueError('Invalid COLMAP image name')
            nraw=file.read(8)
            if len(nraw)!=8:raise ValueError('Truncated COLMAP points2D header')
            n=struct.unpack('<Q',nraw)[0]
            if n>100_000_000:raise ValueError('Implausible COLMAP points2D count')
            file.seek(n*24,1)
            cameras.append((q,t))
    return cameras


def _read_images_txt(path):
    cameras=[]
    with open(path,encoding='utf8',errors='replace') as file:
        while True:
            line=file.readline()
            if not line:break
            if not line.strip() or line.lstrip().startswith('#'):continue
            tokens=line.split()
            if len(tokens)<10:raise ValueError('Invalid COLMAP images.txt pose line')
            cameras.append((np.array([float(x) for x in tokens[1:5]]),
                            np.array([float(x) for x in tokens[5:8]])))
            file.readline()  # alternating POINTS2D[] record, possibly empty
    return cameras


def read_colmap_poses(path):
    """Return C, R^T e_x, R^T e_z. COLMAP q describes world->camera.

    Positive camera Z means optical forward. This is NOT PlayCanvas Y-up.
    """
    path=Path(path)
    if path.suffix.lower()=='.bin':raw=_read_images_bin(path)
    elif path.suffix.lower()=='.txt':raw=_read_images_txt(path)
    else:raise ValueError('Expected COLMAP images.bin or images.txt')
    centers=[];rights=[];forwards=[]
    for q,t in raw:
        nq=np.linalg.norm(q)
        if not np.isfinite(q).all() or not np.isfinite(t).all() or nq<1e-8:continue
        w,x,y,z=q/nq
        R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                    [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                    [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        centers.append(-R.T@t)
        rights.append(R[0,:])
        forwards.append(R[2,:])
    if not centers:raise ValueError('COLMAP contains no valid camera pose')
    return np.asarray(centers),np.asarray(rights),np.asarray(forwards)


def _normalize(v):
    n=np.linalg.norm(v,axis=-1)
    if np.any(~np.isfinite(n)) or np.any(n<1e-12):raise ValueError('Degenerate camera basis')
    return v/n[...,None]


def estimate_from_camera_axes(rights,forwards,minimum_cameras=20,
                              max_median_roll_deg=12.,max_p90_roll_deg=28.,
                              min_heading_eigenvalue=.075,min_optical_projection=.08,
                              min_downward_fraction=.60):
    """Estimate arbitrary unit up from a roll-stabilised UAV camera set.

    R.right approximately horizontal, so u minimizes Σ w_i (r_i·u)^2.
    Optical down-looking consensus resolves +/-u but ONLY conditionally.
    The reported checks are heuristic acceptance criteria (not probability).
    """
    rights=np.asarray(rights,float);forwards=np.asarray(forwards,float)
    if rights.ndim!=2 or rights.shape[1]!=3 or rights.shape!=forwards.shape:
        raise ValueError('Camera right/forward arrays must both be Nx3')
    mask=np.isfinite(rights).all(axis=1)&np.isfinite(forwards).all(axis=1)
    rights,forwards=rights[mask],forwards[mask]
    if len(rights)<minimum_cameras:raise ValueError(f'Only {len(rights)} valid camera poses; need {minimum_cameras}')
    r,f=_normalize(rights),_normalize(forwards)
    weights=np.ones(len(r))
    for _ in range(7):
        covariance=(r*weights[:,None]).T@r/weights.sum()
        evals,evecs=np.linalg.eigh(covariance)
        up=evecs[:,0]
        residual=np.abs(r@up)
        weights=np.minimum(1.,.18/np.maximum(residual,1e-9))
    covariance=(r*weights[:,None]).T@r/weights.sum()
    evals,evecs=np.linalg.eigh(covariance)
    up=evecs[:,0]
    rolls=np.degrees(np.arcsin(np.clip(np.abs(r@up),0.,1.)))
    median_roll=float(np.median(rolls));p90_roll=float(np.percentile(rolls,90))
    # Positive opt. dot means the camera is looking toward +u (up).
    optic=f@up
    if np.median(optic)>0:up=-up;optic=-optic
    downward_fraction=float(np.mean(optic<-.03))
    optical_support=float(-np.median(optic))
    diversity=float(evals[1])
    warnings=[]
    if diversity<min_heading_eigenvalue:
        warnings.append('camera headings insufficiently diverse for stable plane normal')
    if median_roll>max_median_roll_deg or p90_roll>max_p90_roll_deg:
        warnings.append('camera right axes are not consistently horizontal (possible roll/flight layout)')
    if optical_support<min_optical_projection or downward_fraction<min_downward_fraction:
        warnings.append('sign not reliably supported by down-looking optical axes')
    accepted=not warnings
    return {
        'up_vector':np.round(_normalize(up),12).tolist(),
        'accepted':bool(accepted), 'method':'uav_camera_right_irls_with_downlooking_sign',
        'camera_count':int(len(r)),
        'median_camera_right_off_horizontal_deg':median_roll,
        'p90_camera_right_off_horizontal_deg':p90_roll,
        'heading_diversity_eigenvalue':diversity,
        'median_downward_optical_projection':optical_support,
        'downward_looking_camera_fraction':downward_fraction,
        'eigenvalues_right_scatter':evals.tolist(),
        'warnings':warnings,
        'limitations':'Assumes cameras mostly roll-stabilised and down-looking; arbitrary SfM rotation is recoverable, but absolute gravity is not guaranteed without external observations.'}


def _sha256_stream(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def estimate_colmap_vertical(path, minimum_cameras=20):
    path=Path(path).expanduser().resolve(strict=True)
    center,right,forward=read_colmap_poses(path)
    result=estimate_from_camera_axes(right,forward,minimum_cameras=minimum_cameras)
    result.update({'colmap_images':str(path),'source_sha256':_sha256_stream(path),
                   'center_bounds':np.stack([center.min(axis=0),center.max(axis=0)]).round(7).tolist()})
    return result


def infer_colmap_images(scene,config):
    roi=config.get('roi_init',{}) or {}
    o=roi.get('orientation',{}) or {}
    candidates=[o.get('colmap_images'),(config.get('paths',{}) or {}).get('colmap_images'),
                f'/home/xth/datasets/danxiashan/{scene}/sparse/images.bin',
                f'/home/xth/datasets/danxiashan/{scene}/sparse/images.txt']
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().is_file():
            return Path(candidate).expanduser().resolve()
    raise FileNotFoundError('No original COLMAP images.bin found. Specify roi_init.orientation.colmap_images in scene configuration.')


def horizontal_basis(up):
    """Return right-handed (h1,h2,u) frame in the original COLMAP world."""
    up=_normalize(np.asarray(up,float).reshape(3))
    preferred=np.eye(3)[int(np.argmin(np.abs(up)))]
    h1=_normalize(preferred-up*np.dot(preferred,up))
    h2=_normalize(np.cross(up,h1))
    return np.array([h1,h2,up]).tolist()


def read_or_estimate(scene,frame_id,orientation,config,cache_root):
    """Resolve auto/manual orientation; leave source model coordinates unchanged.

    Auto cache keyed to COLMAP path/size/mtime; the full SHA is written once.
    Legacy verified/up_axis fields are intentionally ignored in auto mode.
    """
    mode=(orientation or {}).get('mode','auto')
    if mode=='manual':
        if orientation.get('manual_verified') is not True:
            raise ValueError('Explicit manual vector requires manual_verified: true')
        raw=np.asarray(orientation.get('manual_up_vector'),float)
        if raw.shape!=(3,):raise ValueError('Manual up vector must have 3 elements')
        vec=_normalize(raw)
        return {'up_vector':vec.tolist(),'accepted':True,'method':'manual_explicit',
                'source_sha256':None,'frame_id':frame_id,'confidence_label':'manual_not_independently_verified',
                'warnings':['Manual vector is user-specified; no automatic gravity validation.']}
    if mode!='auto':raise ValueError('roi_init.orientation.mode must be auto or manual')
    images=infer_colmap_images(scene,config)
    st=images.stat()
    fingerprint={'colmap_images':str(images),'size':st.st_size,'mtime_ns':st.st_mtime_ns,
                 'frame_id':frame_id,'algorithm_version':2}
    root=Path(cache_root);root.mkdir(parents=True,exist_ok=True)
    output=root/f'{scene}_{frame_id}.json'
    if output.is_file():
        try:
            cached=json.loads(output.read_text(encoding='utf8'))
            if cached.get('fingerprint')==fingerprint:
                return cached['estimate']
        except (ValueError,KeyError,TypeError):pass
    estimate=estimate_colmap_vertical(images)
    estimate['frame_id']=frame_id
    estimate['confidence_label']='heuristic_accepted' if estimate['accepted'] else 'insufficient_evidence'
    from gsrefine.roi_init.results import write_atomic
    write_atomic(output,{'schema_version':1,'fingerprint':fingerprint,'estimate':estimate})
    return estimate
