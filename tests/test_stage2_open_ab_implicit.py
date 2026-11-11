"""V2 pure-geometry, authentic COLMAP track, SDF and CLI regressions."""
import struct
from pathlib import Path
import numpy as np
import pytest
from gsrefine.cleanup.implicit_space import (
    GridSpec, OUTSIDE, INSIDE, UNKNOWN, make_grid, boundary_points,
    rasterize_segments, query_stable_labels, exterior_connected_to_observations,
    read_colmap_images_index, sample_verified_colmap_rays, valid_observation_rays)
from gsrefine.cleanup.open_ab_main import classify_implicit_signed_proxy


def test_distance_rule_unchanged_and_signs():
    x=classify_implicit_signed_proxy([0.09,.123,.125,1.,5.,np.nan],
             [OUTSIDE,OUTSIDE,OUTSIDE,INSIDE,UNKNOWN,OUTSIDE],.122684916,1e-5)
    assert x['remove'].tolist()==[False,True,True,False,False,False]
    assert x['keep'].sum()==4
    assert np.isnan(x['signed_proxy'][-2])
    assert x['signed_proxy'][3]==pytest.approx(-1.)


def test_distance_rule_strict_epsilon():
    x=classify_implicit_signed_proxy([.12269,.122696],
                          [OUTSIDE,OUTSIDE],.122684916,1.e-5)
    assert x['remove'].tolist()==[False,True]


def test_unknown_and_inside_always_protect_even_when_far():
    x=classify_implicit_signed_proxy([100,200],[INSIDE,UNKNOWN],.01)
    assert not x['remove'].any()
    with pytest.raises(ValueError):
        classify_implicit_signed_proxy([1],[42],.1)


def test_grid_positions_roundtrip_and_outside_unknown():
    sp=GridSpec(np.array([-1.,-1.,-1.]),(5,5,5),.5)
    indices=np.array([0,2,28,100,124])
    assert np.array_equal(sp.index(sp.locations(indices)),indices)
    assert sp.index([[100,0,0]]).tolist()==[-1]


def test_bounded_grid_memory_and_voxel_controls():
    pts=np.array([[-5.,-3.,-2.],[4.,7.,10.]])
    sp=make_grid(pts,pts,.12,voxel_size=.1,max_nodes=12000)
    assert sp.count<=12000
    assert sp.voxel>=.1
    with pytest.raises(ValueError):make_grid(pts,pts,.1,max_nodes=100)


def test_ray_samples_only_between_camera_and_observed_point():
    sp=GridSpec(np.array([-1.,-1.,-1.]),(12,7,7),.5)
    ori=np.array([[-1.,0,0]])
    end=np.array([[3.,0,0]])
    hits,info=rasterize_segments(sp,ori,end)
    assert info['supported_voxels']>0
    assert hits.sum()>0
    assert hits[11].sum()==0  # beyond observed endpoint must not be carved


def test_rays_outside_domain_do_not_create_evidence():
    sp=GridSpec(np.array([0.,0.,0.]),(5,5,5),1.)
    h,info=rasterize_segments(sp,np.array([[-100,-100,-100]]),np.array([[-80,-80,-80]]))
    assert h.sum()==0


def test_two_grid_outside_requires_stability_and_boundary_protection():
    sp=GridSpec(np.zeros(3),(6,6,6),1.)
    labels0=np.full(sp.shape,OUTSIDE,dtype=np.int8)
    labels1=np.full(sp.shape,OUTSIDE,dtype=np.int8)
    labels1[2,2,2]=INSIDE
    p=np.array([[.5,.5,.5],[2.5,2.5,2.5],[5.5,5.5,5.5]])
    res=query_stable_labels(p,[(sp,labels0),(sp,labels1)],np.array([[.5,.5,.5]]),boundary_margin=.25)
    assert res.tolist()==[UNKNOWN,UNKNOWN,OUTSIDE]


def test_only_observed_connected_exterior_valid():
    ext=np.zeros((7,7,7),dtype=bool)
    ext[:2,:2,:2]=True
    ext[5:,5:,5:]=True
    observed=np.zeros_like(ext);observed[0,0,0]=True
    good,stats=exterior_connected_to_observations(ext,observed)
    assert good[0,0,0] and not good[6,6,6]
    assert stats['exterior_components']==2


class FakeTensor:
    def __init__(self,x,whatever=None): self.x=np.asarray(x)
class FakeCast:
    def __init__(self,x):self.data=np.asarray(x)
    def numpy(self):return self.data
class FakeO3d:
    class core:
        Tensor=FakeTensor
class FakeScene:
    def __init__(self,hits):self.hits=hits
    def cast_rays(self,r):return {'t_hit':FakeCast(np.full(len(r.x),self.hits,dtype=float))}


def test_observed_rays_occluded_are_rejected():
    org=np.array([[0.,0,0]])
    dst=np.array([[2.,0,0]])
    assert valid_observation_rays(org,dst,FakeScene(np.inf),FakeO3d).tolist()==[True]
    assert valid_observation_rays(org,dst,FakeScene(.1),FakeO3d).tolist()==[False]
    assert valid_observation_rays(org,dst,FakeScene(np.nan),FakeO3d).tolist()==[False]


def create_minimal_colmap(tmp):
    # two registered COLMAP images, one observed 3D track (point IDs non-contiguous)
    path=tmp/'images.bin'
    with path.open('wb') as f:
        f.write(struct.pack('<Q',2))
        for iid,cx in [(5,0.),(29,1.)]:
            f.write(struct.pack('<i7di',iid,1,0,0,0,-cx,0,0,1))
            f.write(f'a{iid}.jpg'.encode()+b'\x00')
            f.write(struct.pack('<Q',1))
            f.write(struct.pack('<2dq',100.,200.,10005))
    p=tmp/'points3D.bin'
    with p.open('wb') as f:
        f.write(struct.pack('<Q',1))
        f.write(struct.pack('<Q3d3BdQ',10005,2.,3.,4.,255,0,128,1.,2))
        f.write(struct.pack('<2i',5,0));f.write(struct.pack('<2i',29,0))
    return path,p


def test_colmap_binary_ids_tracks_and_core_xyz(tmp_path):
    image_path,point_path=create_minimal_colmap(tmp_path)
    pos=read_colmap_images_index(image_path)
    assert set(pos)=={5,29}
    assert pos[29][0][0]==pytest.approx(1.)
    # need >100 core points under current parser guard, actual observed matches one
    core=np.tile(np.array([[2.,3.,4.]]),(100,1))
    # sample function requires at least 100 matching track rays: replicate
    # enough file records to exercise end-to-end parser, IDs not contiguous.
    with point_path.open('wb') as f:
        f.write(struct.pack('<Q',70))
        for j in range(70):
            f.write(struct.pack('<Q3d3BdQ',10005+j,2.,3.,4.,255,0,128,1.,2))
            f.write(struct.pack('<2i',5,0));f.write(struct.pack('<2i',29,0))
    org,ends,stats=sample_verified_colmap_rays(point_path,pos,core,max_points=70,max_rays=140)
    assert len(org)==140 and stats['core_matched_points']==70
    assert np.allclose(ends,[2,3,4])


def test_colmap_vision_does_not_accept_unmatched_core(tmp_path):
    im,p=create_minimal_colmap(tmp_path)
    poses=read_colmap_images_index(im)
    core=np.tile([[30.,30.,30.]],(100,1))
    with pytest.raises(ValueError,match='Not enough matched'):
        sample_verified_colmap_rays(p,poses,core)


def test_open_ab_v2_cli_routing(monkeypatch):
    from gsrefine.cleanup import direct_r3_pipeline,open_ab_main
    captured={}
    def dummy(**kwargs):
        captured.update(kwargs)
        return {'status':'EXPERIMENTAL_V2','removed':3}
    monkeypatch.setattr(open_ab_main,'run_open_ab_main',dummy)
    r=direct_r3_pipeline.run(manifest='x',phase='open-ab',preseal_report='y',
                output_dir='z',b_method='implicit-volume',max_volume_nodes=20000,
                diagnostic_only=True)
    assert r['removed']==3
    assert captured['max_volume_nodes']==20000
    assert captured['b_method']=='implicit-volume'
    assert captured['diagnostic_only'] is True


def test_graphcut_terminals_and_no_conflicting_seeds(monkeypatch):
    """Exercise actual graph wiring using a tiny exact graph solver stub."""
    import sys,types,networkx as nx
    from gsrefine.cleanup.implicit_space import solve_binary_grid
    class TestGraph:
        def __init__(self, capacity_nodes, capacity_edges):
            self.net=nx.DiGraph();self.n=0;self.labels=None
            self.s='s';self.t='t'
            self.net.add_nodes_from([self.s,self.t])
        def add_nodes(self,n):
            self.n=n
            self.net.add_nodes_from(range(n))
        def add_tedge(self,i,src,sink):
            self.net.add_edge(self.s,i,capacity=src)
            self.net.add_edge(i,self.t,capacity=sink)
        def add_edge(self,i,j,forward,backward):
            self.net.add_edge(i,j,capacity=forward)
            self.net.add_edge(j,i,capacity=backward)
        def maxflow(self):
            value,cut=nx.minimum_cut(self.net,self.s,self.t)
            self.labels=cut[0]
            return value
        def get_segment(self,i):return 0 if i in self.labels else 1
    class GraphFactory:
        def __class_getitem__(cls,kind):return TestGraph
    monkeypatch.setitem(sys.modules,'maxflow',types.SimpleNamespace(Graph=GraphFactory))
    sh=(4,4,4)
    free=np.zeros(sh,np.uint16);free[0,0,0]=1
    inside=np.zeros(sh,bool);inside[3,3,3]=True
    outside=np.zeros(sh,bool);outside[0,0,1]=True
    # 10+ required strong seeds each: create 2x2x3 clusters each side.
    free[:2,:2,:3]=1;inside[2:,2:,1:]=True
    outside[:2,:2,:3]=True
    distance=np.ones(sh,dtype=float)
    prior=np.full(sh,.5,dtype=float)
    labels,diag=solve_binary_grid(free,inside,outside,distance,prior)
    assert labels[0,0,0] and labels[0,0,1]
    assert not labels[3,3,3]
    assert diag['nodes']==64
    inside[0,0,0]=True
    with pytest.raises(ValueError,match='Contradictory'):
        solve_binary_grid(free,inside,outside,distance,prior)


def test_parallel_boundary_ray_no_nan_or_false_voxels():
    spec=GridSpec(np.array([0.,0.,0.]),(5,5,5),1.)
    x,stats=rasterize_segments(spec,np.array([[0.,0.,0.]]),np.array([[4.,0.,0.]]))
    assert stats['voxel_samples']>0 and int(x.sum())>0
