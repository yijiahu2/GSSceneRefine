import importlib.util
from pathlib import Path
import numpy as np
import trimesh

P=Path(__file__).resolve().parents[1]/'tools/trace_r3_poisson_origin.py'
spec=importlib.util.spec_from_file_location('r3_origin_audit',P)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def mesh(vertices, faces):
    return trimesh.Trimesh(vertices=vertices,faces=faces,process=False)


def test_inherited_nonmanifold_source_edge():
    src=mesh([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]],
             [[0,1,2],[1,0,3],[0,1,4]])
    edges,counts,_=mod.edge_incidence(src)
    report=mod.trace_nonmanifold(src,src,np.arange(3),edges,counts,tolerance=1e-7)
    assert report['counts']=={'inherited_from_global_nonmanifold_edge':1}
    assert report['items'][0]['original_global_edge_incidence']==3


def test_preexisting_vs_crop_separated_components():
    vertices=[[0,0,0],[1,0,0],[0,1,0],[1,1,0],[2,0,0],[10,0,0],[11,0,0],[10,1,0]]
    original=mesh(vertices,[[0,1,2],[1,3,2],[1,4,3],[5,6,7]])
    comps, labels=mod.face_components(original)
    assert len(comps)==2
    # Remove connecting face 1. Clipped faces 0 and 2 were connected in global mesh.
    clip=mesh(vertices,[[0,1,2],[1,4,3],[5,6,7]])
    result=mod.trace_components(original,clip,np.array([0,2,3]),labels,tolerance=1e-9)
    assert result['counts']['separated_by_R3_crop_or_clip_topology']==1
    assert result['counts']['preexisting_separate_global_component']==1


def test_ambiguous_parent_does_not_get_definite_classification():
    original=mesh([[0,0,0],[1,0,0],[0,1,0],[5,0,0],[6,0,0],[5,1,0]],
                  [[0,1,2],[3,4,5]])
    clipped=original.copy()
    _,labels=mod.face_components(original)
    out=mod.trace_components(original,clipped,np.array([0,-1]),labels,tolerance=1e-8)
    assert out['counts']=={'uncertain_parent_mapping':1}


def test_edge_map_no_extrapolated_classification_on_ambiguous_parents():
    m=mesh([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]],
           [[0,1,2],[1,0,3],[0,1,4]])
    edges,counts,_=mod.edge_incidence(m)
    d=mod.trace_nonmanifold(m,m,np.array([0,-1,2]),edges,counts,tolerance=1e-8)
    assert d['counts']=={'uncertain_parent_mapping':1}


def test_reports_work_on_edge_deleted_component():
    m=mesh([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]])
    edges,counts,incidence=mod.edge_incidence(m)
    assert int(np.sum(counts>2))==0
    assert len(incidence)==0
    assert mod.trace_nonmanifold(m,m,np.array([0]),edges,counts,tolerance=1e-9)['items']==[]


def test_candidate_face_must_be_geometrically_inside_source_triangle():
    src=np.array([[[0.,0,0],[2.,0,0],[0.,2,0]],
                  [[0.,0,0],[2.,0,0],[0.,2,0]]])
    inside=np.array([[[0.2,.2,0],[.5,.1,0],[.1,.5,0]],
                     [[0.2,.2,0],[.5,.1,0],[3.,.5,0]]])
    got=mod._contained_in_parent_triangles(inside,src,tolerance=1e-5)
    assert got.tolist()==[True,False]


def test_out_of_plane_parent_rejected():
    src=np.array([[[0.,0,0],[2.,0,0],[0.,2,0]]])
    clip=np.array([[[0.2,.2,0],[.5,.1,.01],[.1,.5,0]]])
    assert mod._contained_in_parent_triangles(clip,src,tolerance=1e-5).tolist()==[False]


def test_geometry_parent_lookup_three_probe_flow_without_real_open3d(monkeypatch):
    import sys
    from types import SimpleNamespace
    class Tensor:
        def __init__(self, value): self.value=np.asarray(value)
        def numpy(self): return self.value
    class TMesh:
        def __init__(self): self.vertex={}; self.triangle={}
    class Scene:
        def add_triangles(self,mesh): self.mesh=mesh
        def compute_closest_points(self,pts):
            # All test subtriangles lie strictly in sole source triangle.
            n=len(pts.value)
            return {'primitive_ids':Tensor(np.zeros(n,dtype=np.int32)),
                    'points':Tensor(np.asarray(pts.value,dtype=np.float32))}
    fake=SimpleNamespace(core=SimpleNamespace(Tensor=Tensor),
                         t=SimpleNamespace(geometry=SimpleNamespace(TriangleMesh=TMesh,
                                                                    RaycastingScene=Scene)))
    monkeypatch.setitem(sys.modules,'open3d',fake)
    src=mesh([[0,0,0],[2,0,0],[0,2,0]],[[0,1,2]])
    cut=mesh([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]])
    parents, errors=mod.map_faces_to_source(src,cut,tolerance=1e-6,batch=1)
    assert parents.tolist()==[0]
    assert errors.max()<1e-6
