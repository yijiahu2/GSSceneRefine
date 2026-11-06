"""Camera-geometry orientation unit tests; do not require the real Songzi dataset."""
import struct
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from gsrefine.geometry.orientation import (
    estimate_from_camera_axes, read_colmap_poses, read_or_estimate, horizontal_basis)


def camera_dataset(up, n=128, seed=41, err=0.):
    rng=np.random.default_rng(seed)
    up=np.asarray(up,float);up/=np.linalg.norm(up)
    h1,h2,_=np.asarray(horizontal_basis(up))
    rights=[];forwards=[];rotations=[];centers=[]
    for i in range(n):
        a=i*2*np.pi/n
        right=np.cos(a)*h1+np.sin(a)*h2
        side=-np.sin(a)*h1+np.cos(a)*h2
        forward=-.78*up+.626*side
        forward/=np.linalg.norm(forward)
        down=np.cross(forward,right)
        R=np.stack([right,down,forward])
        if err:
            R=Rotation.from_rotvec(rng.normal(size=3)*err).as_matrix()@R
        rights.append(R[0]);forwards.append(R[2]);rotations.append(R)
        centers.append(rng.normal(size=3))
    return np.array(rights),np.array(forwards),np.array(rotations),np.array(centers)


def write_colmap_images_bin(path, Rs,Cs):
    with open(path,'wb') as f:
        f.write(struct.pack('<Q',len(Rs)))
        for i,(R,C) in enumerate(zip(Rs,Cs)):
            q=Rotation.from_matrix(R).as_quat() # xyzw
            q=np.r_[q[3],q[:3]]
            t=-R@C
            f.write(struct.pack('<i4d3di',i+1,*q,*t,1))
            f.write(f'image_{i:04d}.jpg'.encode()+b'\0')
            f.write(struct.pack('<Q',0))


def test_known_arbitrary_sfm_up_without_rescaling():
    u=np.array([.4,-.75,.526]);u/=np.linalg.norm(u)
    r,f,_,_=camera_dataset(u,err=.025)
    result=estimate_from_camera_axes(r,f)
    assert result['accepted'],result
    assert np.dot(u,result['up_vector'])>.999
    assert result['median_camera_right_off_horizontal_deg']<5


def test_azimuth_coverage_and_optical_sign_are_required():
    r,f,_,_=camera_dataset([0,0,1],n=64)
    weak=estimate_from_camera_axes(np.tile(r[:1],(64,1)),np.tile(f[:1],(64,1)))
    assert not weak['accepted']
    # Side-looking cameras cannot give reliable absolute +/- sign.
    horizontal=np.cross(r,np.array([0,0,1]))
    assert not estimate_from_camera_axes(r,horizontal)['accepted']


def test_colmap_binary_parsing_and_cached_provenance(tmp_path):
    u=np.array([.23,-.51,.83]);u/=np.linalg.norm(u)
    _,_,Rs,Cs=camera_dataset(u,n=50,err=.006)
    path=tmp_path/'images.bin';write_colmap_images_bin(path,Rs,Cs)
    centers,right,forward=read_colmap_poses(path)
    assert np.allclose(centers,Cs,atol=1e-10)
    result=estimate_from_camera_axes(right,forward)
    assert result['accepted'] and np.dot(result['up_vector'],u)>.999
    cfg={'roi_init':{'orientation':{'mode':'auto','colmap_images':str(path)}}}
    cachedir=tmp_path/'orientation'
    a=read_or_estimate('synthetic','synthetic_colmap',cfg['roi_init']['orientation'],cfg,cachedir)
    b=read_or_estimate('synthetic','synthetic_colmap',cfg['roi_init']['orientation'],cfg,cachedir)
    assert a==b
    assert len(a['source_sha256'])==64
    assert (cachedir/'synthetic_synthetic_colmap.json').is_file()


def test_manual_mode_requires_explicit_approval(tmp_path):
    cfg={'roi_init':{'orientation':{}}}
    try:
        read_or_estimate('s','s_colmap',{'mode':'manual','manual_up_vector':[0,0,1]},cfg,tmp_path)
    except ValueError as e:
        assert 'manual_verified' in str(e)
    else:raise AssertionError('Manual orientation cannot silently be accepted')


def test_auto_settings_and_roi_validate_same_arbitrary_colmap_frame(tmp_path,monkeypatch):
    import yaml
    import gsrefine.roi_init.settings as mod
    from gsrefine.roi_init.results import require_orientation,gaussian_roi_preview
    from tests.test_demo import build_synthetic_case
    monkeypatch.setattr(mod,'PROJECT_ROOT',tmp_path)
    build_synthetic_case(tmp_path)
    up=np.array([.41,-.62,.67]);up/=np.linalg.norm(up)
    _,_,Rs,Cs=camera_dataset(up,80,err=.01)
    images=tmp_path/'images.bin'
    write_colmap_images_bin(images,Rs,Cs)
    conf=yaml.safe_load((tmp_path/'config.yaml').read_text())
    conf['scene']={'name':'syntheticcliff','method':'absgs'}
    conf['roi_init']={
        'gaussian_ply':str(tmp_path/'model.ply'),
        'output_dir':str(tmp_path/'all_roi'),
        'orientation':{'mode':'auto','colmap_images':str(images),'frame_id':'syntheticcliff_colmap'},
    }
    file=tmp_path/'auto.yaml';file.write_text(yaml.safe_dump(conf))
    setting=mod.InitSettings.load(file)
    assert np.dot(setting.up_vector,up)>.999
    h1,h2,_=np.asarray(horizontal_basis(setting.up_vector))
    # Large polygons include all original synthetic point centers.
    square=[[-1e3,-1e3],[1e3,-1e3],[1e3,1e3],[-1e3,1e3]]
    payload={'coordinate_system':'original_COLMAP_world',
             'polygon_units':'COLMAP_model_units_not_pixels',
             'views':[{'name':'top','origin':[0,0,0],'axes':[h1.tolist(),h2.tolist()],'polygon':square},
                      {'name':'facade_0deg','origin':[0,0,0],
                       'axes':[h1.tolist(),list(setting.up_vector)],'polygon':square}],
             'annotation_metadata':{
                 'scene':'syntheticcliff','method':'absgs','frame_id':setting.frame_id,
                 'source_ply':str(setting.gaussian_ply),'orientation_verified':True,
                 'up_vector':list(setting.up_vector),
                 'orientation_source_sha256':setting.orientation_result['source_sha256']}}
    require_orientation(payload,setting)
    stat=gaussian_roi_preview(payload,setting)
    assert stat['gaussian_roi_count']==stat['gaussian_total']
