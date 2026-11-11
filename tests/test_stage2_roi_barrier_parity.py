"""Closed-loop and fail-closed tests for original-Poisson parity; no Open3D required."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from gsrefine.cleanup.roi_barrier_parity import (
    OUTSIDE, INSIDE, UNKNOWN, ParityUnionFind, _crossings_from_hits,
    _validate_intersection_loop, segment_parity, _make_navigation,
    _query_nearby_candidates)


class FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data)
    def numpy(self):
        return self.data


class FakeTriangleScene:
    """Open3D-like list_intersections for finite line crossings of plane z=0."""
    def __init__(self, *, degenerate=False):
        self.degenerate = degenerate

    def list_intersections(self, rays):
        rays = np.asarray(rays, float)
        splits = [0]
        t = []; uv = []; ids = []
        for o, d in zip(rays[:, :3], rays[:, 3:]):
            if abs(d[2]) > 1e-10:
                z = -o[2] / d[2]
                if z >= 0:
                    t.append(z)
                    uv.append([0.0, .5] if self.degenerate else [.25, .25])
                    ids.append(0)
            splits.append(len(t))
        return {k: FakeTensor(v) for k,v in {
            'ray_splits':np.array(splits),
            't_hit':np.array(t),
            'primitive_uvs':np.array(uv, float).reshape(-1,2),
            'primitive_ids':np.array(ids, np.int32)}.items()}

O3D = SimpleNamespace(core=SimpleNamespace(Tensor=lambda a: a))


def test_ray_parity_0_1_and_endpoint_exclusion():
    o = np.array([[0,0,1],[0,0,1],[0,0,0]], float)
    t = np.array([[0,0,.5],[0,0,-1],[0,0,1]], float)
    p, ok = segment_parity(FakeTriangleScene(), O3D, o, t,
                           normals=np.array([[0,0,1.]],float))
    assert p.tolist() == [0,1,0]
    assert ok.tolist() == [True,True,False]


def test_ambiguous_shared_edge_is_protected():
    o=np.array([[0,0,1]],float);t=np.array([[0,0,-1]],float)
    _, ok=segment_parity(FakeTriangleScene(degenerate=True),O3D,o,t)
    assert ok.tolist()==[False]


def test_two_crossings_are_even_and_not_double_counted():
    def hits(tt,uvs=None):
        if uvs is None:uvs=np.tile([.25,.25],(len(tt),1))
        return {'ray_splits':np.array([0,len(tt)]),
                't_hit':np.array(tt,float),
                'primitive_uvs':np.asarray(uvs),
                'primitive_ids':np.zeros(len(tt),np.int32)}
    p, ok=_crossings_from_hits(hits([.1,.7]),[1],[[0,0,1]])
    assert p.tolist()==[0] and ok.tolist()==[True]
    p, ok=_crossings_from_hits(hits([.1,.7,.9]),[1],[[0,0,1]])
    assert p.tolist()==[1] and ok.tolist()==[True]
    _, ok=_crossings_from_hits(hits([.3,.3]),[1],[[0,0,1]])
    assert ok.tolist()==[False]  # shared-edge / coincident triangle not silently even


def test_union_find_parity_on_multiple_routes():
    u=ParityUnionFind(5)
    u.add(0,1,0)
    u.add(1,2,1)
    u.add(2,3,0)
    u.add(0,3,1)
    label, stats=u.labels(np.array([1,1,1,1,0],bool),[0])
    assert label.tolist()==[0,0,1,1,-1]
    assert stats['odd_cycle_conflicts']==0
    u.add(0,3,0)  # wrong alternative path: invalidate component
    label,stats=u.labels(np.array([1,1,1,1,0],bool),[0])
    assert np.all(label==-1)
    assert stats['odd_cycle_conflicts']>=1


def test_exterior_seeds_must_not_cover_both_sides():
    u=ParityUnionFind(2)
    u.add(0,1,1)
    label,stats=u.labels(np.array([1,1],bool),[0,1])
    assert label.tolist()==[-1,-1]
    assert stats['odd_cycle_conflicts']>=1


def test_parse_real_intersection_archive():
    f=Path('/mnt/data/global_poisson_final_r3_intersection.npz')
    if f.exists():
        s=_validate_intersection_loop(f)
        assert s['intersections']==623
        assert s['closed_cycles']==1
        assert s['segments']==623


def test_local_ray_correction_with_navigation_grid():
    # Indexed nodes all above or below z=0, known outside=upper.
    shape=(4,4,4);origin=np.array([-2.,-2.,-2.]);voxel=1.
    ids=np.indices(shape).reshape(3,-1).T
    z=origin[2]+(ids[:,2]+.5)*voxel
    labels=np.where(z>0,OUTSIDE,INSIDE).astype('int8').reshape(shape)
    index={'origin':origin,'voxel':voxel,'shape':np.array(shape),
           'labels':labels,'passable':np.ones(shape,bool)}
    points=np.array([[.1,.1,.25],[.1,.1,-.25]])
    field=lambda p: np.ones(len(p))
    res,stats=_query_nearby_candidates(points,index,field,FakeTriangleScene(),O3D,
                    np.array([[0,0,1]],float),neighbours=12,min_agree=2,batch=2)
    assert res.tolist()==[OUTSIDE,INSIDE], stats


def test_dangerous_geo_unknown_protected():
    shape=(4,4,4);origin=np.array([-2.,-2.,-2.]);voxel=1.
    labels=np.full(shape,UNKNOWN,np.int8)
    labels[2,2,2]=OUTSIDE
    labels[2,2,1]=INSIDE
    index={'origin':origin,'voxel':voxel,'shape':np.array(shape),
           'labels':labels,'passable':labels!=UNKNOWN}
    # A broken/no-wall geometry contradicts the cached inside/outside labels.
    class NoWall:
        def list_intersections(self,rays):
            n=len(rays)
            return {k: FakeTensor(v) for k,v in {
                "ray_splits":np.zeros(n+1,np.int32),"t_hit":np.empty(0),
                "primitive_uvs":np.empty((0,2)),"primitive_ids":np.empty(0,np.int32)}.items()}
    val,info=_query_nearby_candidates(np.array([[.1,.1,.1]]),index,
        lambda p:np.ones(len(p)), NoWall(),O3D,
        np.array([[0,0,1]],float),neighbours=2,min_agree=2,batch=2)
    assert val[0]==UNKNOWN


def test_full_navigation_graph_build_with_true_triangle_parity(tmp_path):
    # Full navigation graph where z=0 is a separating original triangle sheet.
    shape=(5,5,6); origin=np.array([-2.5,-2.5,-3.]); voxel=1.
    passable=np.ones(shape,bool)
    np.savez(tmp_path/'grid.npz',origin=origin,voxel=voxel,
        shape=np.asarray(shape,dtype=np.int32),
        passable=passable,final_r3_inside=passable)
    result=_make_navigation(tmp_path/'grid.npz',lambda x:np.ones(len(x)),
        np.array([0.,0.,1.]),FakeTriangleScene(),O3D,
        np.array([[0.,0.,1.]]),neighbour_rays_batch=128,roi_link_samples=3)
    labels=result['labels']
    assert np.all(labels[:,:,3:] == OUTSIDE), result['summary']
    assert np.all(labels[:,:,:3] == INSIDE), result['summary']
    assert result['summary']['odd_cycle_conflicts']==0
    assert result['summary']['triangle_crossing_edges']>0


def test_navigation_mismatch_fails_closed(tmp_path):
    shape=(4,4,4);origin=np.zeros(3)
    bad=np.ones(shape,bool); bad[0,0,0]=False
    np.savez(tmp_path/'grid.npz', origin=origin, voxel=1.,
        shape=np.asarray(shape),passable=bad,final_r3_inside=bad)
    with pytest.raises(ValueError,match='does not match'):
        _make_navigation(tmp_path/'grid.npz',lambda x:np.ones(len(x)),
            np.array([0,0,1.]),FakeTriangleScene(),O3D,
            np.array([[0,0,1.]]))


def test_v1_navigation_identity_signature_bug_fixed(tmp_path):
    """Regression: V1 crashed after 'Audit accepted' from an extra positional argument."""
    from gsrefine.cleanup.roi_barrier_parity import _navigation_identity
    grid = tmp_path / 'fine_connectivity.npz'
    grid.write_bytes(b'grid-content')
    audit = {'inputs': {'manifest_sha256': 'manifest', 'r3_sha256': 'r3',
                        'original_poisson_sha256': 'global'}}
    identity = _navigation_identity(audit, grid, roi_link_samples=3)
    assert identity['grid_sha256']
    assert identity['r3_sha256'] == 'r3'
    assert identity['roi_link_samples'] == 3
    assert 'local_main_poisson_sha256' not in identity
    assert _navigation_identity(audit, grid, roi_link_samples=3) == identity
    assert _navigation_identity(audit, grid, roi_link_samples=7) != identity


def test_adaptive_radius_recovers_far_labelled_nodes():
    """Previously fixed 2.5 voxel search could never classify this valid point."""
    shape = (6, 6, 6)
    labels = np.full(shape, UNKNOWN, np.int8)
    labels[2, 2, 5] = OUTSIDE
    labels[2, 3, 5] = OUTSIDE
    index = {'origin': np.array([-3., -3., -3.]), 'voxel': 1.,
             'shape': np.array(shape), 'labels': labels,
             'passable': labels != UNKNOWN}
    result, details = _query_nearby_candidates(
        np.array([[.1, .1, -.1]]), index, lambda xyz: np.ones(len(xyz)),
        FakeTriangleScene(), O3D, np.array([[0., 0., 1.]]),
        neighbours=2, min_agree=2, batch=1)
    assert result.tolist() == [INSIDE], details
    assert details['stage_resolved'][0] == 0, details
    assert details['stage_resolved'][1] == 1, details
    assert details['classified_uncertain'] == 0


def test_adaptive_local_query_zero_candidates():
    shape = (2, 2, 2)
    labels = np.full(shape, OUTSIDE, np.int8)
    index = {'origin': np.array([-1., -1., -1.]), 'voxel': 1.,
             'shape': np.array(shape), 'labels': labels,
             'passable': np.ones(shape, bool)}
    result, details = _query_nearby_candidates(
        np.empty((0, 3)), index, lambda xyz: np.ones(len(xyz)),
        FakeTriangleScene(), O3D, np.array([[0., 0., 1.]]))
    assert len(result) == 0 and details['candidate_count'] == 0


def test_formal_export_requires_complete_binary_labels():
    from gsrefine.cleanup.roi_barrier_parity import _formal_export_gate
    assert _formal_export_gate(np.array([OUTSIDE, INSIDE, OUTSIDE])) is None
    assert _formal_export_gate(np.array([], dtype=np.int8)) is None
    with pytest.raises(RuntimeError, match='unresolved'):
        _formal_export_gate(np.array([OUTSIDE, UNKNOWN, INSIDE]))
