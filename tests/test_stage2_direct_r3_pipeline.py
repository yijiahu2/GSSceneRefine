"""Synthetic, no Open3D required: topology, source identity, fail-closed behavior."""
from pathlib import Path
import json
import sys
import types
import numpy as np
import pytest
import trimesh
from gsrefine.cleanup import direct_r3_pipeline as mod


def disk():
    verts = np.array([[0,0,0],[1,0,0],[2,0,0],
                      [0,1,0],[1,1,0],[2,1,0],
                      [0,2,0],[1,2,0],[2,2,0]], dtype=float)
    faces=[]
    for j in range(2):
        for i in range(2):
            a=j*3+i; b=a+1; c=a+3; d=c+1
            faces.extend([[a,b,c],[b,d,c]])
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def test_single_contiguous_R3_surface_has_one_boundary_and_is_seal_candidate():
    q = mod.boundary_diagnostics(disk())
    assert q['open_boundary_components'] == 1
    assert q['open_boundary_edges'] == 8
    assert q['nonmanifold_edges'] == 0
    assert q['face_components'] == 1
    assert q['preseal_one_ring_eligible']


def test_interior_hole_is_blocked_not_silently_filled():
    # 4x4 quad grid with a removed central cell: TWO open boundary loops.
    verts=np.array([(i,j,0) for j in range(5) for i in range(5)],float)
    tris=[]
    for j in range(4):
        for i in range(4):
            if (i,j)==(2,2):
                continue
            a=j*5+i;b=a+1;c=a+5;d=c+1
            tris.extend([[a,b,c],[b,d,c]])
    m=trimesh.Trimesh(vertices=verts,faces=tris,process=False)
    q=mod.boundary_diagnostics(m)
    assert q['open_boundary_components']==2
    assert all(x['is_simple_closed_ring'] for x in q['boundary_component_details'])
    assert not q['preseal_one_ring_eligible']
    assert q['face_components']==1


def test_disconnected_surface_is_blocked_even_if_every_part_has_simple_ring():
    merged=trimesh.util.concatenate([disk(),disk().copy().apply_translation([4,0,0])])
    q=mod.boundary_diagnostics(merged)
    assert q['open_boundary_components']==2
    assert not q['preseal_one_ring_eligible']


def test_nonmanifold_edge_is_blocked():
    mesh=disk()
    faces=np.asarray(mesh.faces)
    mesh2=trimesh.Trimesh(vertices=mesh.vertices,
                          faces=np.vstack([faces,[faces[0,0],faces[0,1],8]]),process=False)
    q=mod.boundary_diagnostics(mesh2)
    assert q['nonmanifold_edges']>=1 or not q['preseal_one_ring_eligible']


def test_immutable_clip_report_and_hash_validation(tmp_path):
    run=tmp_path/'a'/'methods'/'absgs'/'roi_001'
    folder=run/'cleanup/geometry/direct_r3_clip/abc'
    folder.mkdir(parents=True)
    candidate=folder/'poisson_roi_direct_r3.ply'
    disk().export(candidate)
    report=folder/'clip_comparison_report.json'
    obj={'schema':'GSSceneRefine.direct_r3_global_poisson_clip.v1',
         'status':'CANDIDATE_ONLY_NOT_SEALED_NOT_GAUSSIAN_CLEANED',
         'candidate_mesh':str(candidate), 'output_sha256':mod._sha(candidate),
         'sources':{}}
    report.write_text(json.dumps(obj))
    assert mod.select_direct_r3_clip(run)[0]==candidate
    candidate.write_bytes(candidate.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='changed'):
        mod.select_direct_r3_clip(run)


def test_ambiguous_clip_reports_fail_closed(tmp_path):
    run=tmp_path/'roi_001'
    for stem in ('one','two'):
        f=run/'cleanup/geometry/direct_r3_clip'/stem/'clip_comparison_report.json'
        f.parent.mkdir(parents=True)
        f.write_text('{}')
    with pytest.raises(ValueError,match='exactly one'):
        mod.select_direct_r3_clip(run)


def test_full_core_fit_required_for_any_sealing(monkeypatch):
    rec={'preseal_eligible':True,'signature':'abc','sfm_fit':{'is_full_core_fit':False}}
    def fake_record(*args,**kwargs):
        raise ValueError('Sealing/SDF requires P95 from ALL R3 core SfM points')
    monkeypatch.setattr(mod,'_geometry_record',fake_record)
    with pytest.raises(ValueError,match='ALL R3 core'):
        mod.run(manifest='/unused',phase='seal')


def test_no_automatic_gaussian_export_from_check(monkeypatch,tmp_path):
    original=tmp_path/'poisson_roi_direct_r3.ply'
    disk().export(original,file_type='ply',encoding='binary')
    from gsrefine.geometry.r3_main_auxiliary import split_direct_r3_mesh
    partition=split_direct_r3_mesh(trimesh.load(original,force='mesh',process=False))
    rec={'signature':'abc','preseal_eligible':False,
         'observed_direct_r3_mesh':str(original),
         'topology_original':mod.boundary_diagnostics(partition['main']),
         'topology_partition':{'auxiliary_component_count':0},
         'topology':{'open_boundary_components':3},
         'sfm_fit':{'fit_sfm_count':60000,'p95':0.04},'sfm_fit_safeguards_passed':True}
    def fake_record(*args,**kwargs):
        return rec,partition['main'],None,None,tmp_path,{},None,partition
    monkeypatch.setattr(mod,'_geometry_record',fake_record)
    result=mod.run(manifest='/unused',phase='check')
    assert result['status']=='BLOCKED_PRESEAL'
    assert (tmp_path/'cleanup/geometry/direct_r3_sdf/abc/preseal_diagnostics.json').is_file()
    assert not (tmp_path/'cleanup/final/gaussian_roi.ply').exists()


def test_source_provenance_rejects_stale_roi(monkeypatch,tmp_path):
    src={}
    for name in ('manifest','poisson','roi','sfm','handoff','gaussian','config'):
        p=tmp_path/name;p.write_text(name);src[name]=p
    fake=types.ModuleType('gsrefine.geometry.r3_direct_poisson_clip')
    fake._source_r3_handoff=lambda *args,**kwargs:src['handoff']
    monkeypatch.setitem(sys.modules,'gsrefine.geometry.r3_direct_poisson_clip',fake)
    class Foo:pass
    # Only mismatch of input hash should block BEFORE loading downstream sources.
    with pytest.raises(ValueError,match='provenance mismatch'):
        mod.validate_sources(src['manifest'],{},
          {'poisson_mesh':src['poisson'],'roi_json':src['roi']},tmp_path,
          {'sources':{'stage1_manifest_sha256':mod._sha(src['manifest']),
                      'stage1_poisson_sha256':mod._sha(src['poisson']),
                      'saved_roi_sha256':'bad'}},src['sfm'])


def test_seal_trial_requires_camera_agreement(monkeypatch):
    fake_d=types.ModuleType('gsrefine.cleanup.sdf_common')
    fake_d._local_basis=lambda up:np.eye(3)
    fake_d._raycast=lambda o3d,mesh:mesh
    fake_d._camera_check=lambda o3d,mesh,cameras,nsamples:{
        'total':len(cameras),'outside_fraction':0.9,'outward_normal_agreement':1.0,
        'nonfinite_distances':0}
    fake_s=types.ModuleType('gsrefine.geometry.sdf_sealing')
    fake_s.prepare_open_poisson=lambda mesh,**kwargs:(mesh,{'no_repair':True})
    fake_s.seal_main=lambda mesh,axis,sign,margin:(
        trimesh.creation.box(), {'mesh_before_faces':len(mesh.faces), 'mesh_before_vertices':len(mesh.vertices)}, trimesh.creation.box())
    monkeypatch.setitem(sys.modules,'gsrefine.cleanup.sdf_common',fake_d)
    monkeypatch.setitem(sys.modules,'gsrefine.geometry.sdf_sealing',fake_s)
    monkeypatch.setitem(sys.modules,'open3d',types.ModuleType('open3d'))
    opts={**mod.DEFAULTS}
    sealed,report=mod.attempt_seal(disk(),np.array([[0,0,2]]),np.array([0,0,1]),opts)
    assert sealed is None
    assert report['status']=='BLOCKED_SEALING'
    assert len(report['direction_trials'])==2
    assert all(not t['passed'] for t in report['direction_trials'])
