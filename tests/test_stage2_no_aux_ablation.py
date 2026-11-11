"""Stage-2 no-aux one-factor ablation regression tests (scene independent)."""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from gsrefine.cleanup import ablate_no_aux as mod


def _masks():
    n=7
    outside=np.array([1,1,0,0,1,0,1],dtype=bool)
    protected=np.array([0,1,0,0,1,0,0],dtype=bool)
    over=np.array([1,1,0,1,1,0,1],dtype=bool)
    removed=outside & ~protected
    fields=dict(keep=~removed,remove=removed,protected_by_aux=protected,
                original_parity_outside=outside,over_distance_threshold=over,
                delta=np.array(.08259647420369177,dtype=np.float64))
    counts=dict(input_gaussians=n,outside_before_aux=int(outside.sum()),
                protected_by_all_aux=int(protected.sum()),
                final_removed=int(removed.sum()),final_retained=int((~removed).sum()),
                over_threshold=int(over.sum()))
    return fields, counts


def test_exact_single_factor_difference():
    f,c=_masks()
    keep,remove,formerly,over,delta=mod._derive_no_aux_masks(f,7,c)
    assert remove.tolist()==[True,True,False,False,True,False,True]
    assert keep.tolist()==[False,False,True,True,False,True,False]
    assert np.array_equal(remove,f['remove']|f['protected_by_aux'])
    assert formerly.sum()==2
    assert delta==pytest.approx(.08259647420369177)


def test_rejects_mask_beyond_outside():
    f,c=_masks()
    f['protected_by_aux']=f['protected_by_aux'].copy()
    f['protected_by_aux'][3]=True
    with pytest.raises(ValueError, match='relations'):
        mod._derive_no_aux_masks(f,7,c)


def test_rejects_count_mismatch_and_dtype():
    f,c=_masks()
    with pytest.raises(ValueError,match='counts'):
        mod._derive_no_aux_masks(f,7,{**c,'final_removed':999})
    f['keep']=f['keep'].astype(np.int8)
    with pytest.raises(ValueError, match='bool'):
        mod._derive_no_aux_masks(f,7,c)


def test_no_scenario_or_component_count_constant():
    for n in (2,3,8,19,150):
        outside=np.zeros(n,bool);outside[:n//2]=True
        prot=outside.copy(); prot[::2]=False
        over=outside.copy()
        old_rm=outside & ~prot
        fields=dict(keep=~old_rm,remove=old_rm,protected_by_aux=prot,
                    original_parity_outside=outside,over_distance_threshold=over,
                    delta=np.array(.2))
        counts=dict(input_gaussians=n,outside_before_aux=int(outside.sum()),
                    protected_by_all_aux=int(prot.sum()),final_removed=int(old_rm.sum()),
                    final_retained=int((~old_rm).sum()),over_threshold=int(over.sum()))
        _,r,_,_,_=mod._derive_no_aux_masks(fields,n,counts)
        assert r.sum()==outside.sum()


def test_fast_ablation_exports_complete_source_subset(tmp_path,monkeypatch):
    f,c=_masks()
    run=tmp_path/'bazhaifeng/methods/absgs/roi_001'
    source=run/'final/gaussian_roi.ply';source.parent.mkdir(parents=True)
    source.write_bytes(b'immutable-GSPL-native-fake')
    baseline=run/'cleanup/stage2_final_v1';diag=baseline/'diagnostics'
    diag.mkdir(parents=True)
    parity=diag/'parity/roi_parity_report.json';parity.parent.mkdir(parents=True)
    parity.write_text('valid mocked parity')
    mask=diag/'final_classification_masks.npz';np.savez_compressed(mask,**f)
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report={
      'schema':'GSSceneRefine.stage2_onepass_r3_parity_aux_protection.v1',
      'status':'COMPLETED_EXPERIMENTAL_CLEAN_ROI_GAUSSIAN',
      'input':{'original_roi_gaussian':str(source),
               'original_roi_gaussian_sha256':sha(source),
               'parity_report':str(parity),'parity_report_sha256':sha(parity)},
      'counts':c,
      'method':{'space_wall':'original global Poisson','distance_surface':'main R3 Poisson',
                'spatial_classifier':'parity short rays'},
      'threshold':{'delta':float(f['delta']), 'epsilon':1e-5},
      'outputs':{'final_mask':str(mask)}
    }
    ref=baseline/'stage2_final_report.json';ref.write_text(json.dumps(report))
    class FakeModel:
        def __init__(self,path):self.vertex_count=7
    monkeypatch.setitem(sys.modules,'gsrefine.roi.core',types.SimpleNamespace(GaussianPLY=FakeModel))
    captured=[]
    def export(model,mask,path):
        captured.append(np.array(mask,dtype=bool,copy=True))
        Path(path).write_bytes(b'full-native-ply-stub')
        return int(mask.sum())
    monkeypatch.setattr(mod,'_export_native',export)
    output=run/'cleanup/stage2_no_aux_v1'
    r=mod.run(baseline_report=ref,output_dir=output,export_diagnostics=True)
    assert r['counts']['final_removed']==4
    assert r['counts']['baseline_aux_protected_now_removed']==2
    assert r['counts']['final_retained']==3
    assert len(captured)==3
    assert np.array_equal(captured[0],~f['original_parity_outside'])
    assert np.array_equal(captured[1],f['original_parity_outside'])
    assert np.array_equal(captured[2],f['over_distance_threshold'] & ~f['original_parity_outside'])
    assert (output/'gaussian_roi_clean.ply').is_file()
    with pytest.raises(FileExistsError):
        mod.run(baseline_report=ref,output_dir=output)


def test_onepass_disable_aux_does_not_run_aux_distance(tmp_path,monkeypatch):
    from gsrefine.cleanup import stage2_onepass as onepass
    root=tmp_path/'another_scene/methods/vanilla/roi_002'
    source=root/'final/gaussian_roi.ply';source.parent.mkdir(parents=True)
    source.write_bytes(b'native mock')
    manifest=root/'metadata/manifest.json';manifest.parent.mkdir(parents=True)
    manifest.write_text('{}')
    global_mesh=root/'global.ply';global_mesh.write_bytes(b'global poisson')
    pseal=root/'cleanup/geometry/direct_r3_sdf/abc/preseal_diagnostics.json'
    pseal.parent.mkdir(parents=True)
    pseal.write_text(json.dumps({'auxiliary_observed_surface':None}))
    parity=root/'cleanup/diagnostics/parity';parity.mkdir(parents=True)
    rfile=parity/'roi_parity_report.json';rfile.write_text('{}')
    mfile=parity/'classification_masks.npz';np.savez(mfile,foo=np.array([1]))
    audit=root/'cleanup/diagnostics/audit';audit.mkdir(parents=True)
    class FakeModel:
        def __init__(self,path):self.vertex_count=5
    monkeypatch.setitem(sys.modules,'gsrefine.roi.core',types.SimpleNamespace(GaussianPLY=FakeModel))
    monkeypatch.setitem(sys.modules,'gsrefine.cleanup.sdf_common',types.SimpleNamespace(
        _load_stage1_manifest=lambda _:(manifest,{},
           {'gaussian_ply':source,'poisson_mesh':global_mesh},root)))
    monkeypatch.setattr(onepass,'_prepare_required_geometry',lambda *a,**kw:(pseal,{'status':'REUSED'}))
    outside=np.array([1,1,0,0,1],bool)
    over=np.array([1,1,1,0,1],bool)
    fake_parity={'inputs':{},'counts':{'candidate_uncertain_kept':0},
        'geometry':{'distance_poisson_triangles':15,
                    'navigation':{'known_outside_nodes':10,'known_inside_nodes':10,
                                  'conflicting_roots':0,'odd_cycle_conflicts':0},
                    'intersection_topology':{'closed_cycles':1}},
        'short_ray':{'conflicting_paths':0},
        'threshold':{'epsilon':.00001,'sfm_p95':.04,
                     'gaussian_scale':{'trimmed_mean_radius':.04}}}
    monkeypatch.setattr(onepass,'_validated_parity',lambda *a,**kw:
           (outside,over,.08,fake_parity,rfile,mfile))
    def no_call(*a,**kw):pytest.fail('Auxiliary proximity function must NOT be called')
    monkeypatch.setattr(onepass,'_auxiliary_protection',no_call)
    def export(model,mask,path):
        Path(path).write_bytes(b'test-native');return int(mask.sum())
    monkeypatch.setattr(onepass,'_export_native',export)
    out=root/'cleanup/stage2_noaux'
    result=onepass.run(manifest=manifest,output_dir=out,preseal_report=pseal,
         barrier_audit_dir=audit,parity_dir=parity,accept_auto_exterior=True,
         auxiliary_protection_enabled=False)
    assert result['counts']['final_removed']==3
    assert result['counts']['final_retained']==2
    assert result['counts']['protected_by_all_aux']==0
    assert result['method']['auxiliary_protection_enabled'] is False
    assert not (out/'diagnostics/aux_protected_gaussians.ply').exists()
    assert (out/'gaussian_roi_clean.ply').exists()
