"""Stage-2: main-only sealing geometry, auxiliary persistence and SDF protection.

Synthetic geometry; all tests run without Open3D or real GSSceneRefine data.
"""
from pathlib import Path
import numpy as np
import trimesh
import pytest
from gsrefine.geometry.r3_main_auxiliary import (
    split_direct_r3_mesh, _boundary_edges_xyz, vertex_fan_issues, auxiliary_evidence)
from gsrefine.cleanup import direct_r3_pipeline as stage2


def disk(n=8):
    verts=np.array([(i,j,0) for j in range(n+1) for i in range(n+1)],dtype=np.float32)
    faces=[]
    for j in range(n):
        for i in range(n):
            a=j*(n+1)+i; b=a+1; c=a+n+1; d=c+1
            faces.extend([[a,b,c],[b,d,c]])
    return trimesh.Trimesh(vertices=verts,faces=faces,process=False)


def soup(n=19):
    main=disk()
    aux=[]
    for i in range(n):
        t=trimesh.creation.icosphere(subdivisions=0,radius=.03)
        t.apply_translation([12+i*.22,2.,1.])
        aux.append(t)
    return main,trimesh.util.concatenate([main,*aux])


def test_main_aux_partition_preserves_every_original_triangle_and_ring():
    main,merged=soup(19)
    p=split_direct_r3_mesh(merged)
    assert p['component_count']==20
    assert p['main_face_count']==len(main.faces)
    assert p['auxiliary_component_count']==19
    assert p['auxiliary_face_count']==len(merged.faces)-len(main.faces)
    assert p['original_r3_boundary_preserved']
    assert p['main_vertex_fan_issue_count']==0
    main_topo=stage2.boundary_diagnostics(p['main'])
    assert main_topo['face_components']==1
    assert main_topo['nonmanifold_edges']==0
    assert main_topo['open_boundary_components']==1
    assert main_topo['open_boundary_edges']==4*8
    a=np.sort(np.concatenate([p['main_original_face_ids'],p['auxiliary_original_face_ids']]))
    assert np.array_equal(a,np.arange(len(merged.faces)))
    assert _boundary_edges_xyz(p['main'])==_boundary_edges_xyz(merged)
    assert np.allclose(np.asarray(merged.vertices),np.asarray(merged.vertices).copy())


def test_vertex_fan_detector_rejects_pinched_geometry_without_nonmanifold_edges():
    # Two separate planar triangles touch only at vertex 0, so all edges have
    # at most one face but fan is NOT a disk/half-disk at vertex zero.
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]],float)
    f=np.array([[0,1,2],[0,3,4]],int)
    m=trimesh.Trimesh(vertices=v,faces=f,process=False)
    assert vertex_fan_issues(m)==[0]


def test_aux_evidence_requires_multiple_distinct_observations():
    main=disk(8)
    island=trimesh.Trimesh(vertices=np.array([[20,0,0],[21,0,0],[20,1,0]],np.float32),
                           faces=[[0,1,2]],process=False)
    part=split_direct_r3_mesh(trimesh.util.concatenate([main,island]))
    sfm=np.column_stack([np.arange(40,dtype=float)*.002+20.1,
                         np.arange(40,dtype=float)*.002+0.1,
                         np.zeros(40)])
    # Spatially nearby but only a few distinct nearest points; this does not
    # automatically become strong independent multi-view evidence.
    gauss=np.array([[20.1,.1,0],[20.2,.2,0],[20.3,.3,0]],float)
    rows=auxiliary_evidence(part,sfm,gauss,np.full(3,.04),
                             main_distance_fn=lambda main,xyz,**kw:np.full(len(xyz),4.))
    assert len(rows)==1
    assert rows[0]['component_id']==0
    assert rows[0]['confirmed_real_object'] is False
    assert rows[0]['observation_evidence'] in ('distributed_both_sources_and_separated','insufficient_or_ambiguous','overlap_with_main_ambiguous')


def test_auxiliary_only_rescues_nearby_main_sdf_removals_and_flags_uncertain():
    main_keep=np.array([True,False,False,False,False,False])
    d=np.array([0.,.04,.07,.03,np.inf,.02])
    idx=np.array([-1,0,0,1,1,-1])
    evidence=[{'component_id':0,'observation_evidence':'distributed_both_sources_and_separated'},
              {'component_id':1,'observation_evidence':'insufficient_or_ambiguous'}]
    keep,good,review=stage2.combine_auxiliary_protection(main_keep,d,idx,evidence,.05,1e-5)
    assert keep.tolist()==[True,True,False,True,False,False]
    assert good.tolist()==[False,True,False,False,False,False]
    assert review.tolist()==[False,False,False,True,False,False]


def test_main_aux_cache_creates_and_verifies_face_map_without_mutating_source(tmp_path):
    _,mesh=soup(3)
    original=tmp_path/'poisson_roi_direct_r3.ply'
    mesh.export(original,file_type='ply',encoding='binary')
    original_bytes=original.read_bytes()
    part=split_direct_r3_mesh(trimesh.load(original,force='mesh',process=False))
    entry={'signature':'demo','observed_direct_r3_mesh':str(original),
           'topology_original':stage2.boundary_diagnostics(mesh),
           'topology':stage2.boundary_diagnostics(part['main']),
           'topology_partition':{'auxiliary_component_count':3}}
    main=tmp_path/'poisson_roi_main.ply'
    aux=tmp_path/'poisson_roi_auxiliary.ply'
    meta=tmp_path/'main_aux_partition_report.json'
    fmap=tmp_path/'main_aux_face_map.npz'
    stage2._persist_main_auxiliary_partition(entry,part,main,aux,meta,fmap)
    assert main.exists() and aux.exists() and fmap.exists()
    assert original.read_bytes()==original_bytes
    h=stage2._sha(main)
    stage2._persist_main_auxiliary_partition(entry,part,main,aux,meta,fmap)
    assert h==stage2._sha(main)
    main.write_bytes(main.read_bytes()+b'invalid')
    with pytest.raises(ValueError,match='Main Poisson changed'):
        stage2._persist_main_auxiliary_partition(entry,part,main,aux,meta,fmap)


def test_single_main_component_no_aux():
    m=disk()
    out=split_direct_r3_mesh(m)
    assert out['auxiliary'] is None
    assert out['auxiliary_face_count']==0
    assert out['original_r3_boundary_preserved']


def test_exact_auxiliary_distance_detects_sheet_overlapping_main():
    """Dense observations beside a duplicate sheet never certify it as separate."""
    main=disk(8)
    island=trimesh.Trimesh(vertices=np.array([[20,0,0],[21,0,0],[20,1,0]],np.float32),
                           faces=[[0,1,2]],process=False)
    part=split_direct_r3_mesh(trimesh.util.concatenate([main,island]))
    sfm=np.array([[20+i*.2,j*.2,0] for i in range(5) for j in range(5)],float)
    gauss=sfm.copy()
    results=auxiliary_evidence(part,sfm,gauss,np.full(len(gauss),.2),
                               main_distance_fn=lambda m,xyz,**kw:np.zeros(len(xyz)))
    assert len(results)==1
    assert results[0]['fraction_face_centroids_separated_from_main']==0
    assert results[0]['observation_evidence']!='distributed_both_sources_and_separated'


def test_main_and_aux_sdf_integration_uses_unsigned_auxiliary_only(monkeypatch, tmp_path):
    """An auxiliary sheet can rescue only main-solid removals within its band."""
    import sys
    import types
    from gsrefine.cleanup import sdf_common

    class T:
        def __init__(self,v): self.v=np.asarray(v)
        def numpy(self): return self.v
    class O3Tensor(T):
        pass
    fake_o3d=types.ModuleType('open3d')
    fake_o3d.core=types.SimpleNamespace(Tensor=O3Tensor)
    monkeypatch.setitem(sys.modules,'open3d',fake_o3d)
    class FakePLY:
        def __init__(self, name): self.vertex_count=4
        def positions(self): return np.array([[0.,0,0],[2.,0,0],[3.,0,0],[4.,0,0]])
    fake_core=types.ModuleType('gsrefine.roi.core')
    fake_core.GaussianPLY=FakePLY
    monkeypatch.setitem(sys.modules,'gsrefine.roi.core',fake_core)
    monkeypatch.setattr(sdf_common,'robust_gaussian_radius',
                        lambda *args,**kwargs:{'trimmed_mean_radius':.05})

    solid=object(); auxiliary=type('FakeAuxiliary',(),{'faces':np.zeros((2,3),dtype=np.int64)})()
    class FakeScene:
        def __init__(self,is_main): self.is_main=is_main
        def compute_signed_distance(self,q,nsamples):
            assert self.is_main
            x=q.numpy()[:,0]
            return T(np.where(x==0,-.5, np.ones(len(x))*.25))
        def compute_distance(self,q):
            assert not self.is_main
            d={2:.02,3:.3,4:.02}
            return T([d[int(x)] for x in q.numpy()[:,0]])
        def compute_closest_points(self,q):
            if self.is_main:
                return {'primitive_ids':T(np.zeros(len(q.numpy()),np.int32))}
            return {'primitive_ids':T(np.array([0 if x<4 else 1 for x in q.numpy()[:,0]],np.int32))}
    monkeypatch.setattr(sdf_common,'_raycast',
                        lambda o3d, mesh:FakeScene(mesh is solid))
    evidence=[{'component_id':0,'observation_evidence':'distributed_both_sources_and_separated'},
              {'component_id':1,'observation_evidence':'overlap_with_main_ambiguous'}]
    kept,stat=stage2._compute_sdf(
        solid,'dummy',{'p95':.05},{'original_main_triangles_in_sealed':1},
        {**stage2.DEFAULTS,'chunk_size':2},auxiliary=auxiliary,
        auxiliary_component_ids=np.array([0,1]),auxiliary_evidence=evidence,
        keep_diagnostics=True,output_dir=tmp_path)
    assert kept.tolist()==[True,True,False,True]
    assert stat['main_sdf_only_kept_count']==1
    assert stat['auxiliary_protected_distributed_support_count']==1
    assert stat['auxiliary_protected_review_count']==1
    assert stat['removed_count']==1
    assert (tmp_path/'classification_masks.npz').exists()


@pytest.mark.parametrize('self_intersects,expected_pass',[(True,False),(False,True)])
def test_seal_rejects_self_intersection_even_if_mesh_is_watertight(monkeypatch,
                                                                   self_intersects,expected_pass):
    import sys
    import types
    from gsrefine.cleanup import sdf_common
    from gsrefine.geometry import r3_main_auxiliary
    class O3Mesh:
        def __init__(self,*args): pass
        def is_self_intersecting(self): return self_intersects
        def is_vertex_manifold(self): return True
        def is_edge_manifold(self,allow_boundary_edges=False): return True
    o3d=types.ModuleType('open3d')
    o3d.geometry=types.SimpleNamespace(TriangleMesh=O3Mesh)
    o3d.utility=types.SimpleNamespace(Vector3dVector=lambda x:x,Vector3iVector=lambda x:x)
    monkeypatch.setitem(sys.modules,'open3d',o3d)
    monkeypatch.setattr(sdf_common,'_local_basis',lambda up:np.eye(3))
    monkeypatch.setattr(sdf_common,'_raycast',lambda *args:object())
    monkeypatch.setattr(sdf_common,'_camera_check',lambda *args:{
        'outside_fraction':1.0,'outward_normal_agreement':1.0,'nonfinite_distances':0})
    fake_sealer=types.ModuleType('gsrefine.geometry.sdf_sealing')
    class SimulatedSolid(trimesh.Trimesh):
        @property
        def body_count(self): return 1
    def simulated_seal(mesh,*args):
        block=trimesh.creation.box()
        source=np.asarray(mesh.vertices)
        block_mesh=SimulatedSolid(vertices=np.vstack((source,block.vertices)),
                                  faces=np.asarray(block.faces)+len(source),process=False)
        return block_mesh, {'mesh_before_faces':len(mesh.faces),
                            'mesh_before_vertices':len(mesh.vertices)}, object()
    fake_sealer.seal_main=simulated_seal
    monkeypatch.setitem(sys.modules,'gsrefine.geometry.sdf_sealing',fake_sealer)
    tilted=disk()
    tilted.vertices[:,2]=tilted.vertices[:,0]*.05
    solid,report=stage2.attempt_seal(tilted,np.array([[0.,0.,2.]]),
                                    np.array([0,0,1.]),stage2.DEFAULTS)
    assert bool(solid is not None)==expected_pass
    assert report['passed']==expected_pass
    assert all(item['sealed_mesh_geometry_audit']['self_intersecting']==self_intersects
               for item in report['direction_trials'])
