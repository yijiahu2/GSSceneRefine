from pathlib import Path
import sys
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gsrefine.roi.core import Projection,ROI,GaussianPLY,run
import json,yaml


def test_distance_math():
    a=Projection.from_json({'name':'top','origin':[0,0,0], 'axes':[[1,0,0],[0,1,0]], 'polygon':[[-1,-1],[1,-1],[1,1],[-1,1]]})
    b=Projection.from_json({'name':'front','origin':[0,0,0], 'axes':[[1,0,0],[0,0,1]], 'polygon':[[-1,-1],[1,-1],[1,1],[-1,1]]})
    r=ROI([a,b]);x=np.array([[0,0,0],[1.2,0,0],[0,0,1.1],[1.05,0,1.08]])
    v=r.distances(x)
    assert np.allclose(v[0],[1.,1.])
    assert np.isclose(v[1,0],-0.2)
    assert np.isclose(v[2,1],-0.1)
    assert np.isclose(v[3].min(),-np.hypot(.05,.08))


def test_quaternion_scale(tmp_path):
    f=tmp_path/'a.ply'
    with f.open('wb') as out:
        out.write(b'ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\nproperty float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nend_header\n')
        out.write(np.array([(0,0,0,np.log(.1),np.log(.2),np.log(.3),1,0,0,0)],dtype=[(n,'<f4') for n in ['x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']]).tobytes())
    p=GaussianPLY(f)
    assert abs(p.directional_sigma([0], [0,1,0])[0]-.2)<1e-6
    p.export(np.array([True]),tmp_path/'keep.ply')
    assert GaussianPLY(tmp_path/'keep.ply').vertex_count==1


def build_synthetic_case(tmp_path):
    # Two DISCONNECTED planes. The first is an intended object with a short
    # surface tip cut by x=1 ROI boundary, the second is unrelated background.
    def plane(x0,x1,y0,y1,nx,ny,z):
        xs=np.linspace(x0,x1,nx); ys=np.linspace(y0,y1,ny)
        pts=np.array([[x,y,z] for y in ys for x in xs],float)
        faces=[]
        for j in range(ny-1):
            for i in range(nx-1):
                a=j*nx+i;b=a+1;c=a+nx;d=c+1
                faces.extend([[a,b,c],[b,d,c]])
        return trimesh.Trimesh(vertices=pts,faces=faces,process=False)
    a=plane(-1.5,1.4,-0.7,0.7,36,17,0)
    bg=plane(3.4,4.6,-0.6,0.6,16,14,0)
    mesh=trimesh.util.concatenate([a,bg])
    mesh.export(tmp_path/'mesh.ply')
    rng=np.random.default_rng(7)
    sfm=np.vstack([a.vertices+rng.normal(0,.003,size=a.vertices.shape),bg.vertices+rng.normal(0,.003,size=bg.vertices.shape)])
    np.savez(tmp_path/'sfm.npz',xyz=sfm,track_length=np.full(len(sfm),8),error=np.full(len(sfm),.1))
    ga=np.vstack([a.vertices+rng.normal(0,.012,size=a.vertices.shape),bg.vertices+rng.normal(0,.012,size=bg.vertices.shape)])
    dtype=[(n,'<f4') for n in ['x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']]
    props=np.zeros(len(ga),dtype=dtype)
    for i,c in enumerate('xyz'):props[c]=ga[:,i]
    for i in range(3):props[f'scale_{i}']=np.log(.035)
    props['rot_0']=1
    with (tmp_path/'model.ply').open('wb') as f:
        f.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(props)}\n'.encode())
        for k in props.dtype.names:f.write(f'property float {k}\n'.encode())
        f.write(b'end_header\n');f.write(props.tobytes())
    obj={'views':[
        {'name':'top','origin':[0,0,0],'axes':[[1,0,0],[0,1,0]],'polygon':[[-1.6,-0.8],[1,-0.8],[1,0.8],[-1.6,0.8]]},
        {'name':'front','origin':[0,0,0],'axes':[[1,0,0],[0,0,1]],'polygon':[[-1.6,-.2],[1,-.2],[1,.2],[-1.6,.2]]},
    ]}
    (tmp_path/'roi.json').write_text(json.dumps(obj))
    cfg={'paths':{'poisson_raw':str(tmp_path/'mesh.ply'),'clean_sfm_npz':str(tmp_path/'sfm.npz'),
          'gaussian_ply':str(tmp_path/'model.ply'),'roi_json':str(tmp_path/'roi.json'),'output_dir':str(tmp_path/'out')},
        'modes':['hard','global','local'],'export_ply':True,'sfm':{'knn_spacing':6,'confidence_weights':[.35,.4,.25],'confidence_threshold':.70},
        'guards':{'seed':2026,'inner_band_frac':.09,'outside_search_frac':.08,'geodesic_limit_frac':.12,
            'max_extension_area_fraction':.25,'max_guard_add_fraction':.5,'max_surface_samples':4000,
            'local_radius_multiplier':4,'local_radius_min_frac':.002,'local_radius_max_frac':.05,
            'max_gaussian_neighbors':64,'min_gaussian_neighbors':16,'global_sample_cap':500,
            'sigma_multiplier':2,'sfm_uncertainty_factor':.2,'b_min_frac':.0001,'b_max_frac':.03}}
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(cfg))
    return a,bg


def test_end_to_end_local_guard_no_distant_background(tmp_path):
    a,bg=build_synthetic_case(tmp_path)
    r=run(tmp_path/'config.yaml')
    assert r['hard']['base_count']<r['global']['final_count']
    assert r['global']['final_count']<=r['local']['final_count'] or r['local']['final_count']>r['hard']['base_count']
    assert r['exterior_search']['accepted_exterior_faces']>0
    hard=np.load(tmp_path/'out/keep_mask_hard.npy')
    loc=np.load(tmp_path/'out/keep_mask_local.npy')
    assert np.all(~loc[len(a.vertices):]),'disconnected background must stay outside ROI'
    assert np.sum(loc[:len(a.vertices)] & ~hard[:len(a.vertices)])>0,'local guard should recover cut tip'
    assert np.all(loc>=hard)
    assert GaussianPLY(tmp_path/'out/gaussians_roi_local.ply').vertex_count==int(loc.sum())
    assert r['local']['guard_diagnostics']['budget_rejected'] is False


def test_fail_closed_if_background_extension_exceeds_budget(tmp_path):
    # Deliberately configure a very strict allowed exterior-area budget.
    # The method must report and reject exterior mesh propagation rather than
    # silently pulling in an arbitrary truncated part of the connected surface.
    build_synthetic_case(tmp_path)
    cfg=yaml.safe_load((tmp_path/'config.yaml').read_text())
    cfg['guards']['max_extension_area_fraction']=0.0001
    cfg['guards']['extension_area_policy']='reject'
    cfg['modes']=['hard','global']
    cfg['export_ply']=False
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(cfg))
    r=run(tmp_path/'config.yaml')
    assert r['exterior_search']['extension_rejected_by_budget'] is True
    assert r['exterior_search']['accepted_exterior_faces']==0
    assert r['global']['final_count'] >= r['hard']['final_count']


def test_rotated_gaussian_directionality(tmp_path):
    f=tmp_path/'rot.ply'
    names=['x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    values=[0,0,0,np.log(.1),np.log(.2),np.log(.3),np.sqrt(.5),0,0,np.sqrt(.5)]
    with f.open('wb') as out:
        out.write(b'ply\nformat binary_little_endian 1.0\nelement vertex 1\n')
        for name in names:out.write(f'property float {name}\n'.encode())
        out.write(b'end_header\n')
        out.write(np.array([tuple(values)],dtype=[(n,'<f4') for n in names]).tobytes())
    p=GaussianPLY(f)
    assert abs(p.directional_sigma([0],[1,0,0])[0]-.2)<1e-5
    assert abs(p.directional_sigma([0],[0,1,0])[0]-.1)<1e-5
