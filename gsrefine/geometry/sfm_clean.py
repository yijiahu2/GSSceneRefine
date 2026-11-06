#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OPTIONAL, independent COLMAP point filtering module for future SfM ablation.

Do NOT overwrite the previously successful 02_sfm_clean_colmap_metadata.npz.
This utility is not claimed to reproduce the previous unknown-stage1 settings.
It reads points3D.bin and writes a new ID-preserving cleaned SfM NPZ and JSON report.

Example:
  python clean_sfm_optional.py --colmap-points /home/xth/datasets/danxiashan/songziguanyin/sparse/points3D.bin --out /mnt/f/.../sfm_clean_observation_sor.npz
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import struct
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

LOG=logging.getLogger('sfm_clean')


def load_colmap_points3d(path):
    path=Path(path)
    xyz=[];rgb=[];errs=[];tracks=[];pids=[]
    head=struct.Struct('<QdddBBBd')
    with path.open('rb') as f:
        raw=f.read(8)
        if len(raw)!=8:raise ValueError('Incomplete COLMAP points3D.bin header')
        count=struct.unpack('<Q',raw)[0]
        if count>50_000_000:raise ValueError(f'implausible COLMAP point count: {count}')
        for j in range(count):
            chunk=f.read(head.size)
            if len(chunk)!=head.size:raise ValueError(f'Truncated point record {j}')
            i,x,y,z,r,g,b,e=head.unpack(chunk)
            chunk=f.read(8)
            if len(chunk)!=8:raise ValueError('Truncated track-length field')
            n=struct.unpack('<Q',chunk)[0]
            if n>10_000_000:raise ValueError('Implausible track length')
            f.seek(8*n,1)
            pids.append(i);xyz.append((x,y,z));rgb.append((r,g,b));errs.append(e);tracks.append(n)
    return (np.asarray(xyz,dtype=np.float64),np.asarray(rgb,dtype=np.uint8),
        np.asarray(errs,dtype=np.float64),np.asarray(tracks,dtype=np.int32),np.asarray(pids,dtype=np.uint64))


def filter_colmap(xyz,errors,tracks,min_track=3,error_mad_mult=5,sor_neighbors=16,sor_std=2.5,do_sor=True):
    if xyz.ndim!=2 or xyz.shape[1]!=3:raise ValueError('Bad xyz')
    valid=np.isfinite(xyz).all(axis=1)&np.isfinite(errors)&(tracks>=int(min_track))
    good_errors=errors[valid]
    if len(good_errors)<100:raise RuntimeError('Too few observed SfM points after initial quality filter')
    emedian=float(np.median(good_errors))
    mad=float(np.median(np.abs(good_errors-emedian)))
    error_limit=emedian+float(error_mad_mult)*max(1.4826*mad,1e-6)
    valid &= (errors<=error_limit)
    reasons={'original':int(len(valid)),'after_observations':int(valid.sum()),'reprojection_error_limit':error_limit}
    if do_sor:
        idx=np.flatnonzero(valid)
        if len(idx)<sor_neighbors+2:raise ValueError('Too few points for SOR')
        tree=cKDTree(xyz[idx])
        d,_=tree.query(xyz[idx],k=sor_neighbors+1,workers=-1)
        knnmean=d[:,1:].mean(axis=1)
        # Conventional statistical outlier filter, global threshold. It can
        # remove real sparse structures; do not use this as a ground-truth mask.
        upper=float(knnmean.mean()+float(sor_std)*knnmean.std())
        valid[idx[knnmean>upper]]=False
        reasons['sor_mean_distance_upper']=upper
        reasons['after_sor']=int(valid.sum())
    return valid,reasons


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--colmap-points',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--min-track',type=int,default=3)
    p.add_argument('--error-mad-mult',type=float,default=5)
    p.add_argument('--sor-neighbors',type=int,default=16)
    p.add_argument('--sor-std',type=float,default=2.5)
    p.add_argument('--skip-sor',action='store_true')
    a=p.parse_args();logging.basicConfig(level=logging.INFO)
    xyz,rgb,errs,tracks,ids=load_colmap_points3d(a.colmap_points)
    keep,report=filter_colmap(xyz,errs,tracks,a.min_track,a.error_mad_mult,a.sor_neighbors,a.sor_std,not a.skip_sor)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.out.exists():raise FileExistsError(f'Will not overwrite existing cleaned point set: {a.out}')
    np.savez_compressed(a.out,xyz=xyz[keep],rgb=rgb[keep],error=errs[keep],track_length=tracks[keep],point3D_ids=ids[keep],source_row_indices=np.flatnonzero(keep))
    np.save(a.out.with_name(a.out.stem+'_keep_mask.npy'),keep)
    report.update({'source':str(a.colmap_points),'output':str(a.out),'deleted_count':int((~keep).sum()),
        'removed_fraction':float(np.mean(~keep)), 'note':'Independent optional cleaning; not a reproduction of previous unverified cleaning configuration.'})
    a.out.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
