"""Stage-2 V2 separation-gate synthetic regression; no Open3D dependency."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gsrefine.cleanup.barrier_separation_gate import evaluate, _region_graph
from gsrefine.cleanup.roi_barrier_parity import _make_navigation


class Tensor:
    def __init__(self, data): self.value=np.asarray(data)
    def numpy(self): return self.value


class Plane:
    def __init__(self, height=0.): self.height=height
    def list_intersections(self, rays):
        rays=np.asarray(rays,float)
        splits=[0];t=[];uv=[];face=[]
        for origin,direction in zip(rays[:,:3],rays[:,3:]):
            if abs(direction[2])>1e-9:
                x=(self.height-origin[2])/direction[2]
                if x>0:
                    t.append(x);uv.append([.25,.25]);face.append(0)
            splits.append(len(t))
        return {'ray_splits':Tensor(np.array(splits, np.int64)),
                't_hit':Tensor(np.array(t, float)),
                'primitive_uvs':Tensor(np.array(uv, float).reshape(-1,2)),
                'primitive_ids':Tensor(np.array(face, np.int32))}


class EmptyWall:
    def list_intersections(self, rays):
        n=len(rays)
        return {'ray_splits':Tensor(np.zeros(n+1,np.int64)),
                't_hit':Tensor(np.empty(0)),
                'primitive_uvs':Tensor(np.empty((0,2))),
                'primitive_ids':Tensor(np.empty(0,np.int32))}


O3D=SimpleNamespace(core=SimpleNamespace(Tensor=lambda x:x))


def fixture(tmp_path):
    details={}
    for label,h,shape in [('coarse',1.,(8,8,8)),('fine',.5,(16,16,16))]:
        xyz=np.indices(shape,dtype=int)[2]
        comp=np.where(xyz<shape[2]//2,0,1).astype(np.int32)
        passing=np.ones(shape,bool)
        np.savez(tmp_path/f'{label}_connectivity.npz',
                 origin=np.array([-4.,-4.,-4.]),voxel=h,
                 shape=np.array(shape,np.int32), final_r3_inside=passing,
                 passable=passing,components=comp)
        details[label]={'status':'TOP_TO_BOTTOM_PATH_FOUND_SAMPLED_GRAPH',
                        'grid':{'shape':list(shape),'voxel':h,'roi_touching_audit_window_samples':0}}
    return {'flood':details}


def get_verdict(tmp_path, scene=None):
    audit=fixture(tmp_path)
    return evaluate(audit_dir=tmp_path,audit_record=audit,
                    field=lambda x:np.ones(len(x),float),
                    wall_scene=scene or Plane(),o3d=O3D,
                    normals=np.array([[0.,0.,1.]]))


def test_valid_original_wall_passes_even_when_old_top_bottom_flag_failed(tmp_path):
    r=get_verdict(tmp_path)
    assert r['status']=='SUPPORTED_NUMERIC_TWO_SIDED_PARTITION_NOT_SEALED',r
    assert not r['blocking_reasons']
    for level in ('coarse','fine'):
        info=r['levels'][level]
        assert set(info['material_ids'])=={0,1}
        assert info['relations'][0]['status']=='OPPOSITE_SIDE_SUPPORTED'
        assert info['relations'][0]['odd']>=8
    assert r['cross_scale']['errors']==[]
    assert 'old_top_bottom_status' not in r


def test_disconnected_grid_without_a_real_triangle_wall_is_rejected(tmp_path):
    r=get_verdict(tmp_path, EmptyWall())
    assert r['status'].startswith('BLOCKED')
    assert any('No pair' in x for x in r['blocking_reasons'])


def test_missing_fine_grid_blocks(tmp_path):
    audit=fixture(tmp_path)
    (tmp_path/'fine_connectivity.npz').unlink()
    r=evaluate(audit_dir=tmp_path,audit_record=audit,
               field=lambda x:np.ones(len(x)),wall_scene=Plane(),o3d=O3D,
               normals=np.array([[0.,0.,1.]]))
    assert r['status'].startswith('BLOCKED')
    assert 'fine flood grid missing' in r['blocking_reasons']


def test_original_triangle_region_xor_cycle_conflict():
    def edge(a,b,bit):
        return {'regions':[a,b], 'status':'OPPOSITE_SIDE_SUPPORTED' if bit else 'SAME_SIDE_SUPPORTED', 'bit':bit}
    r=_region_graph([0,1,2],[edge(0,1,1),edge(1,2,1),edge(0,2,1)])
    assert r['parity_conflicts']==1


def test_auto_seed_is_single_documented_exterior_hypothesis(tmp_path):
    """Highest-shell exterior seed is a hypothesis, not a top-bottom gate."""
    from gsrefine.cleanup.roi_barrier_parity import _make_navigation
    shape=(4,4,6); h=1.; origin=np.array([-2.,-2.,-3.])
    passable=np.ones(shape,bool)
    comp=np.zeros(shape,np.int32); comp[:,:,3:]=1
    np.savez(tmp_path/'grid.npz', origin=origin, voxel=h,
             shape=np.asarray(shape), passable=passable,
             final_r3_inside=passable, components=comp)
    nav=_make_navigation(tmp_path/'grid.npz',lambda p:np.ones(len(p)),
                         np.array([0.,0.,1.]),Plane(),O3D,np.array([[0.,0.,1.]]),
                         neighbour_rays_batch=256)
    summary=nav['summary']
    assert summary['seed_mode']=='single highest R3 shell node (researcher must verify exterior)'
    assert summary['seed_count']==1
    assert np.isclose(summary['seed_info']['chosen_seed_xyz'][2],2.5)
    assert summary['odd_cycle_conflicts']==0

# stage2_region_unified_original_main_v1_20261112
