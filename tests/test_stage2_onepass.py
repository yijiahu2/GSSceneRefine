"""Portable, no external CUDA/Open3D: regression contracts for stage2 one-pass."""
import json
from pathlib import Path

import numpy as np
import pytest

from gsrefine.cleanup import stage2_onepass as mod


def _native_gaussian(path, n=6):
    props = ['x','y','z','f_dc_0','f_dc_1','f_dc_2','opacity',
             'scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3',
             'f_rest_0','f_rest_1']
    dt = np.dtype([(p, '<f4') for p in props])
    values = np.empty(n, dt)
    for j, p in enumerate(props):
        values[p] = np.arange(n,dtype=np.float32) + j * 0.12
    header = '\n'.join(['ply','format binary_little_endian 1.0',
                         f'element vertex {n}',*(f'property float {p}' for p in props),'end_header',''])
    path.write_bytes(header.encode('ascii')+values.tobytes())
    return values


def test_masks_only_outside_unprotected_are_removed():
    outside = np.array([0,1,1,1,0,0],bool)
    over = np.array([0,1,1,1,1,0],bool)
    protected = np.array([0,1,0,1,0,0],bool)
    keep, remove = mod._combine_removal(outside,over,protected,uncertain_count=0,all_gaussians=6)
    assert remove.tolist() == [False,False,True,False,False,False]
    assert keep.sum() == 5


def test_auxiliary_protection_never_overrides_near_threshold():
    with pytest.raises(ValueError,match='Outside/protected'):
        mod._combine_removal(np.array([1,0],bool),np.array([0,1],bool),
                             np.array([0,0],bool),uncertain_count=0,all_gaussians=2)


def test_uncertain_blocks_final_output():
    with pytest.raises(RuntimeError,match='unresolved'):
        mod._combine_removal(np.array([1,0],bool),np.array([1,1],bool),
                             np.array([0,0],bool),uncertain_count=1,all_gaussians=2)


def test_no_scene_specific_count_assumptions():
    for n in (1,3,19,100):
        outside=np.ones(n,bool)
        keep,remove=mod._combine_removal(outside,outside,np.zeros(n,bool),
                                         uncertain_count=0,all_gaussians=n)
        assert remove.sum()==n and keep.sum()==0


def test_output_restricted_to_roi_cleanup(tmp_path):
    run=tmp_path/'scene/methods/m/roi_001'
    source=run/'final/gaussian_roi.ply'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'raw')
    allowed=run/'cleanup/final_v1'
    assert mod._validate_output(allowed,run,source)==allowed
    with pytest.raises(ValueError,match='subdirectory'):
        mod._validate_output(tmp_path/'elsewhere',run,source)
    allowed.mkdir(parents=True)
    (allowed/'old.ply').write_bytes(b'existing')
    with pytest.raises(FileExistsError,match='not empty'):
        mod._validate_output(allowed,run,source)


def test_auxiliary_absent_is_valid_for_new_scene():
    class Source:
        vertex_count=4
    protect,topo,d=mod._auxiliary_protection(Source(),np.array([1,0,1,0],bool),.1,None)
    assert not protect.any() and topo['components']==0
    assert len(d)==2 and np.isinf(d).all()


def test_native_supersplat_fields_byte_exact(tmp_path):
    from gsrefine.roi.core import GaussianPLY
    original=_native_gaussian(tmp_path/'gaussian_roi.ply')
    model=GaussianPLY(tmp_path/'gaussian_roi.ply')
    mask=np.array([1,0,1,0,0,1],bool)
    target=tmp_path/'gaussian_roi_clean.ply'
    written=mod._export_native(model,mask,target)
    dest=GaussianPLY(target)
    assert written==3 and dest.vertex_count==3
    assert dest.arr.tobytes()==original[mask].tobytes()
    assert any(k.startswith('f_rest') for k in dest.dtype.names)


def test_reuses_source_matched_barrier_audit(tmp_path):
    """Reuse source-identical region graphs, never old top-bottom conclusions."""
    root=tmp_path/'roi_001'
    root.mkdir()
    manifest=root/'manifest.json'; manifest.write_bytes(b'one')
    handoff=root/'handoff.npz'; handoff.write_bytes(b'two')
    poisson=root/'poisson.ply'; poisson.write_bytes(b'three')
    folder=root/'cleanup/diagnostics/barrier'
    folder.mkdir(parents=True)
    for name in ('coarse_connectivity.npz','fine_connectivity.npz','global_poisson_final_r3_intersection.npz'):
        (folder/name).write_bytes(b'x')
    report={'schema':'GSSceneRefine.final_R3_global_Poisson_barrier_audit.v2',
            'version':'final_R3_original_global_poisson_region_graph_v3_window_20261112',
            'status':'REGION_GRAPHS_COMPLETED_NUMERIC_NOT_CERTIFIED',
            'diagnostic_window':{'window_cover':{'status':'COMPLETE_SAMPLED_WINDOW'}},
            'inputs':{'manifest_sha256':mod._sha(manifest),
                      'r3_sha256':mod._sha(handoff),
                      'original_poisson_sha256':mod._sha(poisson)},
            'flood':{'coarse':{'status':'REGION_GRAPH_SAMPLED_NOT_CERTIFIED',
                                  'grid':{'voxel':.3,'r3_boundary_touch_count':0}},
                     'fine':{'status':'REGION_GRAPH_SAMPLED_NOT_CERTIFIED',
                                 'grid':{'voxel':.2,'r3_boundary_touch_count':0}}}}
    report_path=folder/'roi_global_poisson_barrier_report.json'
    report_path.write_text(json.dumps(report))
    assert mod._find_barrier_audit(root,manifest,handoff,poisson)==folder
    # Older sampled graph status is not a rejection when source, grid and
    # post-cache separation verification remain valid.
    report['flood']['coarse']['status']='TOP_TO_BOTTOM_PATH_FOUND_SAMPLED_GRAPH'
    report_path.write_text(json.dumps(report))
    assert mod._find_barrier_audit(root,manifest,handoff,poisson) is None
    report['flood']['coarse']['status']='REGION_GRAPH_SAMPLED_NOT_CERTIFIED'
    report_path.write_text(json.dumps(report))
    # Actual truncated sampling window still invalidates cache reuse.
    report['flood']['coarse']['grid']['r3_boundary_touch_count']=1
    report_path.write_text(json.dumps(report))
    assert mod._find_barrier_audit(root,manifest,handoff,poisson) is None
    report['flood']['coarse']['grid']['r3_boundary_touch_count']=0
    report_path.write_text(json.dumps(report))
    poisson.write_bytes(b'changed')
    assert mod._find_barrier_audit(root,manifest,handoff,poisson) is None


def test_complete_onepass_assembles_full_scene(tmp_path, monkeypatch):
    """Exercise integrated clean export from synthetic parity, without geometry CUDA."""
    from gsrefine.cleanup.sdf_common import _load_stage1_manifest
    from gsrefine.roi.core import GaussianPLY
    run=tmp_path/'s/methods/absgs/roi_001'
    source=run/'final/gaussian_roi.ply'; source.parent.mkdir(parents=True)
    _native_gaussian(source,6)
    manifest=run/'metadata/manifest.json'; manifest.parent.mkdir(parents=True)
    manifest.write_text('{}')
    global_poisson=run/'poisson.ply';global_poisson.write_bytes(b'xyz')
    preseal=run/'cleanup/geometry/direct_r3_sdf/abc/preseal_diagnostics.json'; preseal.parent.mkdir(parents=True)
    preseal.write_text(json.dumps({'auxiliary_observed_surface':None,'sfm_fit':{'p95':.07}}))
    pre_existing_parity=run/'cleanup/preview/parity';pre_existing_parity.mkdir(parents=True)
    rep=pre_existing_parity/'roi_parity_report.json';rep.write_text('{}')
    mask=pre_existing_parity/'classification_masks.npz';np.savez(mask,foo=np.array([1]))
    monkeypatch.setattr('gsrefine.cleanup.sdf_common._load_stage1_manifest',
       lambda path:(manifest,{}, {'gaussian_ply':source,'poisson_mesh':global_poisson},run))
    monkeypatch.setattr(mod,'_prepare_required_geometry',
                        lambda *a,**kw:(preseal,{'status':'REUSED'}))
    fake_outside=np.array([1,0,1,0,1,0],bool)
    fake_over=np.array([1,0,1,0,1,0],bool)
    fake_parity={'inputs':{},'counts':{'candidate_uncertain_kept':0},
                 'geometry':{'distance_poisson_triangles':15,
                             'navigation':{'known_outside_nodes':10,'known_inside_nodes':5,
                                           'conflicting_roots':0,'odd_cycle_conflicts':0},
                             'intersection_topology':{'closed_cycles':1}},
                 'short_ray':{'conflicting_paths':0},
                 'threshold':{'epsilon':.00001,'sfm_p95':.07,
                              'gaussian_scale':{'trimmed_mean_radius':.03}}}
    monkeypatch.setattr(mod,'_validated_parity',lambda *a,**kw:
                        (fake_outside,fake_over,.10,fake_parity,rep,mask))
    monkeypatch.setattr(mod,'_auxiliary_protection',lambda *a,**kw:
                        (np.array([0,0,1,0,0,0],bool),
                         {'components':2,'faces':3},np.array([.3,.02,.4],np.float32)))
    # Explicit audit dir bypasses cache searching; classification is reused.
    audit=run/'cleanup/diagnostics/audit';audit.mkdir(parents=True)
    out=run/'cleanup/final_test'
    result=mod.run(manifest=manifest,output_dir=out,
                   preseal_report=preseal,barrier_audit_dir=audit,
                   parity_dir=pre_existing_parity,accept_auto_exterior=True)
    assert result['counts']['protected_by_all_aux']==1
    assert result['counts']['final_removed']==2
    assert result['counts']['final_retained']==4
    assert (out/'gaussian_roi_clean.ply').is_file()
    assert GaussianPLY(out/'gaussian_roi_clean.ply').vertex_count==4
    assert (out/'stage2_final_report.json').is_file()
    npz=np.load(out/'diagnostics/final_classification_masks.npz')
    assert npz['remove'].tolist()==[True,False,False,False,True,False]
    assert len(npz['keep'])==6
    assert (out/'diagnostics/aux_protected_gaussians.ply').is_file()


def test_requires_exterior_seed_or_explicit_provisional_acknowledgment():
    with pytest.raises(ValueError,match='No certified exterior seed'):
        mod.run(manifest='/none',output_dir='/none')


def test_auxiliary_protection_independent_of_component_count(tmp_path,monkeypatch):
    """Five parts, not nineteen: component count has no algorithmic magic."""
    import types
    import gsrefine.cleanup.auxiliary_protection as aux
    from gsrefine.roi.core import GaussianPLY
    p=tmp_path/'source.ply';_native_gaussian(p,5)
    model=GaussianPLY(p)
    mask=np.array([1,1,0,1,0],bool)
    mesh=object()
    monkeypatch.setattr(aux,'load_auxiliary_mesh',lambda path:
        (mesh,{'components':5,'faces':11,'vertices':20,'faces_per_component':[3,2,2,2,2]}))
    class FakeScene:
        def compute_distance(self,q):
            class Answer:
                def numpy(self):return np.array([.01,.13,.05],dtype=np.float32)
            return Answer()
    o3d=types.SimpleNamespace(core=types.SimpleNamespace(
        Tensor=lambda data,dtype=None:data,Dtype=types.SimpleNamespace(Float32='f4')))
    import sys
    from gsrefine.cleanup import sdf_common
    monkeypatch.setitem(sys.modules,'open3d',o3d)
    monkeypatch.setattr(sdf_common,'_raycast',lambda o3d, mesh: FakeScene())
    path=tmp_path/'aux.ply';path.write_bytes(b'fake')
    protected,geo,d=mod._auxiliary_protection(model,mask,.1,path,chunk_size=10)
    assert geo['components']==5 and protected.tolist()==[True,False,False,True,False]
    assert np.allclose(d,[.01,.13,.05])


def test_no_over_threshold_gaussians_becomes_valid_noop(tmp_path,monkeypatch):
    import gsrefine.cleanup.auxiliary_protection as aux
    from gsrefine.roi.core import GaussianPLY
    p=tmp_path/'source.ply';_native_gaussian(p,3)
    model=GaussianPLY(p)
    monkeypatch.setattr(aux,'load_auxiliary_mesh',lambda p:
                        (object(),{'components':4,'faces':8}))
    class Scene:
        def compute_distance(self,q):
            pytest.fail('Distance lookup should not be called for empty candidate set')
    import types
    o3d=types.SimpleNamespace(core=types.SimpleNamespace(
        Tensor=lambda d,dtype=None:d,Dtype=types.SimpleNamespace(Float32='f4')))
    import sys
    from gsrefine.cleanup import sdf_common
    monkeypatch.setitem(sys.modules,'open3d',o3d)
    monkeypatch.setattr(sdf_common,'_raycast',lambda o3d,mesh: Scene())
    auxfile=tmp_path/'aux.ply';auxfile.write_bytes(b'fake')
    protect,geo,dist=mod._auxiliary_protection(model,np.zeros(3,bool),.2,auxfile)
    assert len(dist)==0 and not protect.any() and geo['components']==4


def test_validated_parity_rejects_different_distance_surface(tmp_path,monkeypatch):
    import gsrefine.cleanup.auxiliary_protection as aux
    a=tmp_path/'parity';a.mkdir()
    original=tmp_path/'gaussian_roi.ply';_native_gaussian(original,2)
    mesh_a=tmp_path/'poisson_a.ply';mesh_a.write_bytes(b'a')
    mesh_b=tmp_path/'poisson_b.ply';mesh_b.write_bytes(b'b')
    preseal=tmp_path/'preseal.json';preseal.write_text(json.dumps({
        'main_observed_surface':str(mesh_a),'sfm_fit':{'p95':.03}}))
    glob=tmp_path/'global.ply';glob.write_bytes(b'foo')
    report={'inputs':{'preseal':str(preseal),'original_full_scene_poisson':str(glob),
                      'main_trusted_poisson':str(mesh_b)},
            'threshold':{'gaussian_scale':{'trimmed_mean_radius':.02}},
            'counts':{'candidate_uncertain_kept':0}}
    parityfile=a/'roi_parity_report.json';parityfile.write_text('{}')
    masks=a/'classification_masks.npz';np.savez(masks,foo=np.array([1]))
    monkeypatch.setattr(aux,'load_validated_parity',lambda *args:
        (np.array([1,0],bool),np.array([1,0],bool),.05,report,parityfile,masks))
    class Source:
        vertex_count=2
    with pytest.raises(ValueError,match='distance surface differs'):
        mod._validated_parity(a,Source(),original,preseal,glob)


def test_preseal_delta_mismatch_blocks_cache_reuse(tmp_path,monkeypatch):
    import gsrefine.cleanup.auxiliary_protection as aux
    a=tmp_path/'parity';a.mkdir()
    original=tmp_path/'gaussian_roi.ply';_native_gaussian(original,2)
    main=tmp_path/'main.ply';main.write_bytes(b'main')
    glob=tmp_path/'global.ply';glob.write_bytes(b'global')
    preseal=tmp_path/'preseal.json';preseal.write_text(json.dumps({
        'main_observed_surface':str(main),'sfm_fit':{'p95':.03}}))
    report={'inputs':{'preseal':str(preseal),'original_full_scene_poisson':str(glob),
                      'main_trusted_poisson':str(main)},
            'threshold':{'gaussian_scale':{'trimmed_mean_radius':.02}},
            'counts':{'candidate_uncertain_kept':0}}
    rf=a/'roi_parity_report.json';rf.write_text('{}')
    mf=a/'classification_masks.npz';np.savez(mf,foo=np.array([1]))
    monkeypatch.setattr(aux,'load_validated_parity',lambda *args:
        (np.array([1,0],bool),np.array([1,0],bool),.08,report,rf,mf))
    class Source:
        vertex_count=2
    with pytest.raises(ValueError,match='delta disagrees'):
        mod._validated_parity(a,Source(),original,preseal,glob)


def test_parity_stage_executes_classification_only(monkeypatch,tmp_path):
    from gsrefine.cleanup import roi_barrier_parity as p
    params={}
    def pretend(**kwargs):
        params.update(kwargs)
        return {'status':'CLASSIFICATION_ONLY_PREVIEW'}
    monkeypatch.setattr(p,'run_roi_barrier_parity',pretend)
    mod._run_parity('manifest','preseal',tmp_path/'classification',
                    nearest_k=16,min_path_agree=2)
    assert params['classification_only'] is True
    assert params['confirm_exterior'] is False
    assert params['nearest_k']==16


def test_geometry_preparation_reuses_validated_existing_records(monkeypatch,tmp_path):
    """Reject old clip-before-main preseal; accept new original-main-first cache."""
    from gsrefine.cleanup import sdf_common
    run=tmp_path/'ROI';run.mkdir()
    manifest=run/'metadata/manifest.json';manifest.parent.mkdir()
    manifest.write_text('{}')
    preseal=run/'cleanup/geometry/direct_r3_sdf/a/preseal_diagnostics.json'
    preseal.parent.mkdir(parents=True)
    preseal.write_text(json.dumps({'stage1_manifest':str(manifest)}))
    monkeypatch.setattr(sdf_common,'_load_stage1_manifest',
                        lambda path:(manifest,{}, {},run))
    import types,sys
    fake=types.ModuleType('gsrefine.cleanup.stage2_geometry_validation')
    fake.validate_parity_geometry=lambda *a,**kw:{'summary':{'status':'VALIDATED_SEALLESS_DISTANCE_GEOMETRY'}}
    monkeypatch.setitem(sys.modules,'gsrefine.cleanup.stage2_geometry_validation',fake)
    with pytest.raises(ValueError,match='Old preseal'):
        mod._prepare_required_geometry(manifest,run,preseal_report=preseal)
    clip=run/'cleanup/geometry/direct_r3_clip/new/clip_comparison_report.json'
    clip.parent.mkdir(parents=True)
    clip.write_text(json.dumps({
        'source_component_selection':
        'largest_original_edge_connected_component_BEFORE_R3_clip'}))
    preseal.write_text(json.dumps({
        'stage1_manifest':str(manifest),
        'direct_r3_clip_report':str(clip)}))
    selected,rec=mod._prepare_required_geometry(manifest,run,preseal_report=preseal)
    assert selected==preseal
    assert rec['status']=='VALIDATED_SEALLESS_DISTANCE_GEOMETRY_REUSED'


def test_core_aux_geometry_counts_are_dynamic(tmp_path):
    """An arbitrary number of detached triangles must work without 19 special-cases."""
    import trimesh
    from gsrefine.cleanup.auxiliary_protection import load_auxiliary_mesh
    components=[]
    for j in range(4):
        q=np.array([[j*3.,0.,0.],[j*3.+1,0,0],[j*3.,1,0]],float)
        components.append(trimesh.Trimesh(vertices=q,faces=[[0,1,2]],process=False))
    combined=trimesh.util.concatenate(components)
    path=tmp_path/'auxiliary.ply'
    combined.export(path,file_type='ply',encoding='binary')
    mesh,stat=load_auxiliary_mesh(path)
    assert stat['components']==4 and stat['faces']==4
    assert stat['faces_per_component']==[1,1,1,1]


def test_production_v2_mask_loader_rejects_source_method_mismatch(tmp_path):
    from gsrefine.cleanup.auxiliary_protection import load_validated_parity
    a=tmp_path/'parity';a.mkdir()
    src=tmp_path/'original_gaussian_roi.ply';_native_gaussian(src,4)
    other=tmp_path/'another_method.ply';_native_gaussian(other,4)
    report={'schema':'GSSceneRefine.roi_barrier_parity_sdf.v2',
            'status':'CLASSIFICATION_ONLY_PREVIEW',
            'inputs':{'original_R3_Gaussian':str(src)},
            'threshold':{'delta':.123},
            'counts':{'candidate_outside_remove':2,
                      'over_distance_threshold':3,'all_gaussians':4}}
    (a/'roi_parity_report.json').write_text(json.dumps(report))
    candidate=np.array([True,False,True,False])
    over=np.array([True,True,True,False])
    np.savez(a/'classification_masks.npz',
             remove=candidate,keep=~candidate,over_threshold=over,
             distance_main=np.array([.3,.2,.5,.01],np.float32),
             outside_confident=candidate)
    mask,above,delta,*_=load_validated_parity(a,4,src)
    assert mask.sum()==2 and above.sum()==3 and delta==.123
    with pytest.raises(ValueError,match='different Gaussian'):
        load_validated_parity(a,4,other)
