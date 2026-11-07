"""Stage-2 unit/contract tests; real-scene Open3D rendering validation is separate."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from gsrefine.cleanup.direct_sdf import (
    _load_stage1_manifest, _local_basis, classify_direct_sdf, robust_gaussian_radius,
)
from gsrefine.geometry.sdf_sealing import boundary_loops, choose_main_component, prepare_open_poisson, seal_main


def test_keep_unknown_delete_only_positive_outside():
    signed=np.asarray([-0.2,0.,.01,.08002,.10,np.nan,np.inf],dtype=np.float32)
    kept,remove,uncertain=classify_direct_sdf(signed, delta=.08,epsilon=.00001)
    assert remove.tolist()==[False,False,False,True,True,False,False]
    assert uncertain.tolist()==[False,False,False,False,False,True,True]
    assert np.all(kept | remove) and np.all(~(kept & remove))


def test_model_specific_trimmed_three_sigma():
    class FakePLY: pass
    f=FakePLY();f.vertex_count=20
    f.arr=np.zeros(20,dtype=[('scale_0','<f4'),('scale_1','<f4'),('scale_2','<f4')])
    f.arr['scale_0']=np.log(.02)
    f.arr['scale_1']=np.log(.01)
    f.arr['scale_2']=np.log(.005)
    stat=robust_gaussian_radius(f,.1,3.)
    assert stat['gaussians']==20
    assert stat['trimmed_mean_radius']==pytest.approx(.06,rel=1e-6)


def test_colmap_up_vector_is_not_assumed_world_z():
    up=np.array([-.0026179,-.6855628,-.7280088])
    up/=np.linalg.norm(up)
    b=_local_basis(up)
    assert np.allclose(b.T@b,np.eye(3),atol=1e-12)
    assert np.allclose(b[:,2],up,atol=1e-12)


def test_sealing_reuses_trusted_surface_and_fails_nontrivial_holes():
    # A mildly sloping open surface; the artificial vertical base is NOT Poisson-observed.
    vertices=np.array([[-1,-1,0],[1,-1,.1],[1,1,.2],[-1,1,.1]],float)
    mesh=trimesh.Trimesh(vertices=vertices,faces=np.array([[0,1,2],[0,2,3]]),process=False)
    main,report=choose_main_component(mesh,min_fraction=.85)
    assert report['main_face_count']==2
    loops,diag=boundary_loops(main)
    assert len(loops)==1 and diag['boundary_edges']==4
    for direction in (1,-1):
        solid,information,artificial=seal_main(main,2,direction,.02)
        assert solid.is_watertight and solid.is_volume
        assert artificial is not None
        assert information['skirt_face_count']==8
        assert np.allclose(np.asarray(solid.vertices)[:len(main.vertices)],np.asarray(main.vertices))
        assert len(solid.faces)==len(main.faces)+information['skirt_face_count']+information['cap_face_count']


def test_upstream_manifest_hash_contract(tmp_path):
    run_dir=tmp_path/'example_scene/methods/absgs/roi_001'
    (run_dir/'metadata').mkdir(parents=True)
    (run_dir/'final').mkdir()
    scene=tmp_path/'example_scene/geometry';scene.mkdir(parents=True)
    gaussian=run_dir/'final/gaussian_roi.ply'
    sfm=scene/'cleaned_sfm.npz';poisson=scene/'poisson_surface.ply'
    roi=run_dir/'metadata/roi.json'
    for path in (gaussian,sfm,poisson,roi):path.write_bytes(path.name.encode())
    sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    path=run_dir/'metadata/manifest.json'
    manifest={'schema':'GSSceneRefine.output.v3','status':'completed',
       'result_status':'ROI_RESTORED_NOT_ANOMALY_CLEANED',
       'stage2_inputs':{'gaussian_ply':str(gaussian),'roi_json':str(roi),
          'poisson_mesh':str(poisson),'clean_sfm_npz':str(sfm)},
       'input_sha256':{'poisson':sha(poisson),'sfm':sha(sfm),'roi':sha(roi)},
       'output_sha256':sha(gaussian)}
    path.write_text(json.dumps(manifest))
    m,md,inputs,rdir=_load_stage1_manifest(path)
    assert rdir==run_dir and inputs['gaussian_ply']==gaussian
    gaussian.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='hash mismatch'):
        _load_stage1_manifest(path)


def test_single_nonmanifold_edge_caused_by_duplicate_face_is_repaired():
    # Edge (0,2) has three incident faces because a triangle is repeated.
    vertices = np.array([[-1,-1,0],[1,-1,.1],[1,1,.2],[-1,1,.1]],float)
    faces = np.array([[0,1,2],[0,2,3],[2,1,0]],dtype=int)
    broken = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    with pytest.raises(ValueError, match='non-manifold edges'):
        boundary_loops(broken)
    repaired, report = prepare_open_poisson(broken)
    assert report['nonmanifold_edges_before']['count'] == 1
    assert report['duplicate_faces_removed'] == 1
    assert report['nonmanifold_edges_after']['count'] == 0
    assert report['input_poisson_modified'] is False
    assert len(broken.faces) == 3  # original remains untouched
    main, _ = choose_main_component(repaired)
    solid, *_ = seal_main(main, 2, 1, .02)
    assert solid.is_watertight and solid.is_volume


def test_three_distinct_faces_on_one_edge_still_fail_closed():
    vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]],float)
    faces = np.array([[0,1,2],[1,0,3],[0,1,4]],dtype=int)
    broken = trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    cleaned, report = prepare_open_poisson(broken)
    assert report['duplicate_faces_removed'] == 0
    assert report['degenerate_faces_removed'] == 0
    assert report['nonmanifold_edges_after']['count'] == 1
    with pytest.raises(ValueError, match='no arbitrary face deletion'):
        boundary_loops(cleaned)


def test_roi_piecewise_clip_splits_boundary_and_preserves_original():
    from gsrefine.geometry.roi_surface import _clip_piecewise_linear
    vertices=np.array([[-2,0,0],[2,0,0],[2,2,0],[-2,2,0]],float)
    faces=np.array([[0,1,2],[0,2,3]],int)
    original=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    # Crop the left halfspace x >= 0, with a shared edge crossing on (0,2).
    signed=vertices[:,0].copy()
    clipped, info=_clip_piecewise_linear(original,signed)
    assert clipped.faces.shape[0] >= 2
    assert clipped.vertices[:,0].min() >= -1e-10
    assert info['partly_clipped_triangles'] == 2
    assert len(original.faces) == 2
    assert np.allclose(original.vertices,vertices)
    # The midpoint of shared original edge 0->2 should be present once.
    shared=np.isclose(clipped.vertices,[0,1,0]).all(axis=1)
    assert shared.sum() == 1


def test_roi_crop_with_certified_mask_does_not_infer_gaussian_guard():
    from gsrefine.geometry.roi_surface import extract_roi_surface
    raw=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0],[3,0,0],[4,0,0],[3,1,0]],
                        faces=[[0,1,2],[3,4,5]],process=False)
    mask=np.array([True,False],dtype=bool)
    cropped, report=extract_roi_surface(raw,None,None,None,{},o3d=None,
        gaussian_radius=.1,certified_face_mask=mask)
    assert len(cropped.faces)==1
    assert report['selection_mode']=='certified_stage1_surface_faces'
    assert report['guard']['is_exact_stage1_r3_surface_mask'] is True


def test_roi_signed_uses_intersection_of_all_projection_constraints():
    from gsrefine.geometry.roi_surface import roi_signed
    class V:
        def __init__(self,axis):self.axis=axis
        def signed_and_normal(self,points):
            sd=1-np.abs(points[:,self.axis]);return sd,np.zeros_like(points)
    class R:views=(V(0),V(1))
    points=np.array([[0,0,0],[1.3,0,0],[0,-1.5,0],[.5,.5,10]])
    sd=roi_signed(points,R())
    assert np.allclose(sd,[1,-.3,-.5,.5])


def test_roi_crop_inferred_guard_provenance_and_conservation(monkeypatch):
    from gsrefine.geometry import roi_surface
    class V:
        def signed_and_normal(self,points):
            return 1-np.abs(points[:,0]),np.zeros_like(points)
    class R:views=(V(),)
    class Gaussian:
        vertex_count=10
        def positions(self):return np.array([[0,0,0],[.2,1,0]])
    monkeypatch.setattr(roi_surface,'_guard_anchors',
        lambda *args,**kwargs:(np.empty((0,3)),np.empty(0),
        {'source':'inferred_from_restored_gaussian_roi',
         'guard_max_outside':.1,'is_exact_stage1_r3_surface_mask':False}))
    raw=trimesh.Trimesh(vertices=[[-1.1,0,0],[.1,0,0],[.1,1,0],[-1.1,1,0]],
         faces=[[0,1,2],[0,2,3]],process=False)
    before=raw.vertices.copy()
    cfg={'guard_max_outside_fraction':.025,'max_unresolved_face_violation':1e-5}
    cropped,report=roi_surface.extract_roi_surface(raw,R(),Gaussian(),np.ones((50,3)),
        cfg,o3d=None,gaussian_radius=.05)
    assert report['selection_mode']=='inferred_r3_gaussian_support'
    assert report['guard']['is_exact_stage1_r3_surface_mask'] is False
    assert np.all(cropped.vertices[:,0] >= -1-1e-8)
    assert np.array_equal(raw.vertices,before)


def test_stage1_handoff_retains_only_existing_certified_faces(tmp_path):
    from gsrefine.geometry.stage1_handoff import (
        SOURCE_NAME, SEMANTICS, read_verified_surface_faces,
        retain_stage1_surface_faces,
    )
    work=tmp_path/'working'; work.mkdir()
    meta=tmp_path/'metadata'; meta.mkdir()
    roi=tmp_path/'roi.json';roi.write_text('{"views": []}',encoding='utf8')
    mesh=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0],[1,1,0]],
                         faces=[[0,1,2],[1,3,2]],process=False)
    poisson=tmp_path/'trusted.ply';mesh.export(poisson)
    assert retain_stage1_surface_faces(work,meta,poisson,roi) is None
    sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    selected=np.array([True,False],dtype=bool)
    np.savez_compressed(work/SOURCE_NAME,keep=selected,
                        poisson_sha256=np.array(sha(poisson)),
                        roi_sha256=np.array(sha(roi)),
                        selection_semantics=np.array(SEMANTICS))
    saved=retain_stage1_surface_faces(work,meta,poisson,roi)
    assert saved.parent==meta
    assert np.array_equal(read_verified_surface_faces(saved,poisson,roi,2),selected)
    # Upstream source mutation MUST cause a hard error, never silent inference.
    roi.write_text('{"views": [1]}',encoding='utf8')
    with pytest.raises(ValueError,match='changed'):
        read_verified_surface_faces(saved,poisson,roi,2)


def test_stage1_handoff_rejects_non_authoritative_mask(tmp_path):
    from gsrefine.geometry.stage1_handoff import (
        SOURCE_NAME, retain_stage1_surface_faces)
    work=tmp_path/'work';work.mkdir()
    metadata=tmp_path/'meta'
    roi=tmp_path/'roi.json';roi.write_text('{}')
    poisson=tmp_path/'raw.ply'
    trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]],
                    faces=[[0,1,2]],process=False).export(poisson)
    sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    np.savez_compressed(work/SOURCE_NAME,keep=np.array([True]),
                        poisson_sha256=np.array(sha(poisson)),
                        roi_sha256=np.array(sha(roi)),
                        selection_semantics=np.array('initial_roi_polygons_only'))
    with pytest.raises(ValueError,match='NOT a final R3'):
        retain_stage1_surface_faces(work,metadata,poisson,roi)
    assert not (metadata/'trusted_roi_surface_faces.npz').exists()


def test_local_stage1_publisher_patch_is_conservative_and_idempotent(tmp_path):
    import importlib.util
    from pathlib import Path
    source=Path(__file__).resolve().parents[1] / 'tools/apply_stage2_handoff.py'
    spec=importlib.util.spec_from_file_location('local_patcher',source)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    project=tmp_path/'repo'
    target=project/'gsrefine/roi_init/output_layout.py'
    target.parent.mkdir(parents=True)
    fixture=("def publish(out, work, roi, config, report):\n"
             "    downstream = {'roi_json': 'foo'}\n"
             "    shutil.copy2(config, meta/'effective_scene.yaml')\n"
             "    manifest = {'stage2_inputs': downstream}\n")
    target.write_text(fixture,encoding='utf8')
    status,backup=mod.apply(project,tmp_path/'backup')
    assert status=='patched' and backup.is_file()
    assert 'retain_stage1_surface_faces' in target.read_text()
    status,other=mod.apply(project,tmp_path/'backup')
    assert status=='already_installed' and other is None
    assert backup.read_text()==fixture
    target.write_text('def publish():\n    pass\n',encoding='utf8')
    with pytest.raises(ValueError,match='not modified'):
        mod.apply(project,tmp_path/'backup')
