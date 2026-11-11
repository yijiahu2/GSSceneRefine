"""Synthetic tests: source triangle/root and flood graph support, without Open3D."""
from pathlib import Path
import importlib.util
import numpy as np
import trimesh

MODULE=Path(__file__).resolve().parents[1]/'tools/audit_final_r3_global_poisson_barrier.py'
spec=importlib.util.spec_from_file_location('audit_final_r3_global_poisson_barrier',MODULE)
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def final_roi_plane_field(x):
    return np.asarray(x)[:,2] - 0.4


def test_roots_use_actual_nonlinear_r3_field():
    xyz=np.array([[0,0,0],[1,0,1],[2,0,0],[3,0,1]],float)
    edges=np.array([[0,1],[2,3]])
    fa=final_roi_plane_field(xyz[edges[:,0]])
    fb=final_roi_plane_field(xyz[edges[:,1]])
    loc,res,window=audit.roots_on_crossing_edges(xyz,edges,fa,fb,final_roi_plane_field,
                                                  iterations=32,eval_batch=4)
    np.testing.assert_allclose(loc[:,2],.4,atol=1e-9)
    assert np.max(res)<1e-9
    assert np.max(window)<1e-8


def test_intersection_from_original_triangle_not_clipped(tmp_path):
    raw=trimesh.Trimesh(vertices=[[0,0,0],[1,0,1],[0,1,1]],
                       faces=[[0,1,2]],process=False)
    result=audit.field_edge_crossings(raw,final_roi_plane_field,iterations=26,
                                      batch=8,output_dir=tmp_path,check_same=0)
    assert result['strict_crossing_original_edges']==2
    assert result['original_faces_with_two_intersected_edges']==1
    z=np.load(tmp_path/'global_poisson_final_r3_intersection.npz')
    assert z['segment_endpoints'].shape==(1,2,3)
    np.testing.assert_allclose(z['segment_endpoints'][0,:,2],.4,atol=1e-6)
    assert np.asarray(raw.vertices)[0,2]==0 # not modified


def test_global_largest_component_discards_unrelated_piece():
    big=trimesh.creation.box()
    small=trimesh.Trimesh(vertices=[[100,0,0],[101,0,0],[100,1,0]],faces=[[0,1,2]],process=False)
    source=trimesh.util.concatenate([big,small])
    picked,report=audit.largest_original_component(source)
    assert len(picked.faces)==len(big.faces)
    assert report['other_global_component_faces_ignored']==1


def test_grid_bounds_coverage():
    h,shape=audit.grid_spec([0,0,0],[10,10,10],.1,2000)
    assert h>.1
    assert np.prod(shape)<=2000
    assert all(np.array(shape)*h>=10)


def test_r3_link_filter_handles_nonconvex_path():
    # endpoints inside (two disjoint bands), centre outside -> rejected
    field=lambda x: (np.abs(np.asarray(x)[:,0]-1.)-.3)
    p=np.array([[0,0,0],[0,0,0]])
    q=np.array([[2,0,0],[.2,0,0]])
    allowed=audit.link_r3_filter(p,q,field,batch=2,samples=3)
    assert allowed.tolist()==[False,True]


def test_walkable_graph_components():
    shape=(2,2,2)
    mask=np.ones(np.prod(shape),dtype=bool)
    a,b=audit.neighbour_pairs(shape,mask)
    labels,n=audit._connected_components(mask.size,mask,a,b)
    assert n==1
    assert np.unique(labels).tolist()==[0]
    labels2,n2=audit._connected_components(mask.size,mask,
                      np.empty(0,dtype='int32'),np.empty(0,dtype='int32'))
    assert n2==8
    assert len(set(labels2.tolist()))==8


def test_report_sanitizes_numpy_types(tmp_path):
    f=tmp_path/'audit.json'
    audit.atomic_json(f,{'a':np.int64(7),'b':np.array([1,2],dtype=np.int32)})
    assert '"a": 7' in f.read_text()


def test_no_real_inputs_were_ever_created(tmp_path):
    src=np.array([[1.,2.,3.]])
    sha=src.copy()
    result=audit.eval_r3(lambda x:np.sum(x,axis=1),src,batch=1)
    assert np.array_equal(src,sha)
    np.testing.assert_allclose(result,[6.])


def test_real_triangle_ray_crossing_test_interface():
    # Fake Open3D scene simulating an actual plane z=0.5; the segmentation
    # layer must forbid a traversing segment and allow a parallel segment.
    from types import SimpleNamespace
    class AsArray:
        def __init__(self,a):self.a=a
        def numpy(self):return self.a
    class PlaneScene:
        def cast_rays(self,rays):
            origin=rays[:,:3]
            direction=rays[:,3:]
            t=np.full(len(rays),np.inf)
            z=direction[:,2]
            valid=np.abs(z)>1e-10
            t[valid]=(.5-origin[valid,2])/z[valid]
            t[t<0]=np.inf
            return {'t_hit':AsArray(t)}
    o3d=SimpleNamespace(core=SimpleNamespace(Tensor=lambda a:a))
    start=np.array([[0,0,0],[0,0,0]])
    end=np.array([[0,0,1],[1,0,0]])
    flags=audit._ray_hit_free(PlaneScene(),o3d,start,end,eps=1e-6,batch=2)
    assert flags.tolist()==[False,True]
