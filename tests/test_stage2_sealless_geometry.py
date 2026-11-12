"""Regression: complex R3 geometry must not be rejected by OLD sealing gates."""
from __future__ import annotations
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import trimesh

from gsrefine.cleanup.stage2_geometry_validation import (
    GAUSSIAN_DISTANCE_DEFAULTS, STAGE2_GEOMETRY_VALIDATION,
    validate_parity_geometry,
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    run = tmp_path / 'scene' / 'methods' / 'EfficientGS' / 'roi_003'
    base = run / 'cleanup' / 'geometry' / 'direct_r3_sdf' / 'abc'
    base.mkdir(parents=True)
    metadata = run / 'metadata'; metadata.mkdir()
    (metadata / 'effective_scene.yaml').write_text('test: true\n')
    manifest = metadata / 'manifest.json'; manifest.write_text('{}')
    sfm = base / 'sfm_roi_local_v3.npz'; sfm.write_bytes(b'dummy sfm')
    source = base / 'poisson_roi_direct_r3.ply'; source.write_bytes(b'source-ply')
    # A valid positive-area triangle mesh with TWO distinct open boundary rings.
    mesh = trimesh.creation.box()
    keep = [i for i in range(len(mesh.faces)) if i not in (0, 2)]
    mesh.update_faces(keep)
    main = base / 'poisson_roi_main.ply'; mesh.export(main)
    clip = base / 'clip_comparison_report.json'; clip.write_text('{}')
    report = base / 'preseal_diagnostics.json'
    rec = {
        'schema':'GSSceneRefine.direct_r3_stage2.v1',
        'stage1_manifest':str(manifest),
        'sfm_fit':{'p95':0.1,'source_core_sfm_count':150,
                   'fit_sfm_count':150,'is_full_core_fit':True},
        'sfm_fit_safeguards':{'max_by_roi_diagonal':0.5},
        'sfm_fit_safeguards_passed':True,
        'preseal_eligible':False, # MUST NOT gate sealless production.
        'topology':{
            'mesh_faces':len(mesh.faces),'mesh_vertices':len(mesh.vertices),
            'open_boundary_components':2,'preseal_one_ring_eligible':False,
            'winding_consistent':False,
        },
        'topology_partition':{'main_face_count':len(mesh.faces),
                              'auxiliary_component_count':0,
                              'original_r3_boundary_preserved':False,
                              'main_vertex_fan_issue_count':2},
        'main_observed_surface':str(main),
        'main_observed_surface_sha256':sha(main),
        'observed_direct_r3_mesh':str(source),
        'direct_r3_clip_report':str(clip),
        'sfm':str(sfm),
        'auxiliary_observed_surface':None,
    }
    def write(): report.write_text(json.dumps(rec))
    write()

    # Synthetic source/clip context is sufficient; the *real* functions are
    # used automatically in the installed repository on real data.
    sdf = types.ModuleType('gsrefine.cleanup.sdf_common')
    sdf._load_stage1_manifest=lambda p:(manifest, {'dummy':1},{'poisson_mesh':str(source)},run)
    d = types.ModuleType('gsrefine.cleanup.direct_r3_pipeline')
    d.select_direct_r3_clip=lambda _run_dir, selected_report=None:(source,clip,{'sources':'verified'})
    d.validate_sources=lambda *args:{'validated':True}
    monkeypatch.setitem(sys.modules,'gsrefine.cleanup.sdf_common',sdf)
    monkeypatch.setitem(sys.modules,'gsrefine.cleanup.direct_r3_pipeline',d)
    return manifest, report, rec, write, main


def test_multi_ring_nonseal_preflight_is_valid(prepared):
    manifest, report, rec, write, main=prepared
    checked=validate_parity_geometry(manifest,report)
    assert checked['summary']['status']=='VALIDATED_SEALLESS_DISTANCE_GEOMETRY'
    assert checked['summary']['schema']==STAGE2_GEOMETRY_VALIDATION
    assert checked['summary']['legacy_seal_only_diagnostics_not_gates']['preseal_eligible'] is False
    assert checked['summary']['legacy_seal_only_diagnostics_not_gates']['open_boundary_components']==2
    assert checked['summary']['main_faces']==10


def test_rejects_wrong_stage1_roi(prepared):
    manifest, report, rec, write, main=prepared
    rec['stage1_manifest']='/some/other/roi_001/metadata/manifest.json';write()
    with pytest.raises(ValueError,match='DIFFERENT Stage-1'):
        validate_parity_geometry(manifest,report)


def test_rejects_incomplete_sfm(prepared):
    manifest, report, rec, write, main=prepared
    rec['sfm_fit']['is_full_core_fit']=False;write()
    with pytest.raises(ValueError,match='incomplete core-SfM fit'):
        validate_parity_geometry(manifest,report)


def test_rejects_genuinely_bad_sfm_fit_with_metrics(prepared):
    manifest, report, rec, write, main=prepared
    rec['sfm_fit']['p95']=0.9;rec['sfm_fit_safeguards_passed']=False;write()
    with pytest.raises(ValueError,match=r'P95=0.9, max_by_roi_diagonal=0.5'):
        validate_parity_geometry(manifest,report)


def test_rejects_changed_observed_mesh(prepared):
    manifest, report, rec, write, main=prepared
    main.write_bytes(main.read_bytes()+b'tampering')
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        validate_parity_geometry(manifest,report)


def test_rejects_fake_main_aux_partition(prepared):
    manifest, report, rec, write, main=prepared
    rec['topology_partition']['main_face_count']=999;write()
    with pytest.raises(ValueError,match='face count'):
        validate_parity_geometry(manifest,report)


def test_only_stage2_distance_parameters_used():
    assert GAUSSIAN_DISTANCE_DEFAULTS=={'trim_fraction':.1,'sigma_multiplier':3.0,'epsilon':1e-5}


def test_production_files_do_not_import_old_seal_validator():
    root=Path(__file__).resolve().parent.parent/'gsrefine'/'cleanup'
    for path in (root/'roi_barrier_parity.py',root/'stage2_onepass.py'):
        text=path.read_text('utf-8')
        assert 'from gsrefine.cleanup.open_ab_main import _validated_inputs' not in text
        assert 'from gsrefine.cleanup.open_ab_main import DEFAULTS' not in text


def test_new_onepass_geometry_ignores_old_BLOCKED_PRESEAL_status(tmp_path, monkeypatch):
    """The real regression: old skirt/cap may be impossible; parity still proceeds."""
    import gsrefine.cleanup.stage2_onepass as runner
    run = tmp_path / 'results/scene/methods/EfficientGS/roi_003'; run.mkdir(parents=True)
    manifest = run / 'metadata/manifest.json'; manifest.parent.mkdir()
    manifest.write_text('{}')
    preseal = run / 'cleanup/geometry/direct_r3_sdf/a/preseal_diagnostics.json'
    preseal.parent.mkdir(parents=True); preseal.write_text('{}')
    clip_mesh = run / 'cleanup/geometry/direct_r3_clip/xyz/poisson_roi_direct_r3.ply'
    clip_mesh.parent.mkdir(parents=True); clip_mesh.write_text('placeholder')
    (clip_mesh.parent / 'clip_comparison_report.json').write_text('{}')
    sdf = types.ModuleType('gsrefine.cleanup.sdf_common')
    sdf._load_stage1_manifest=lambda p:(manifest,{}, {}, run)
    sfm = types.ModuleType('gsrefine.geometry.r3_sfm_bridge')
    sfm.resolve_stage2_r3_sfm=lambda *a:(run / 'core.npz', {}, False)
    clip = types.ModuleType('gsrefine.geometry.r3_direct_poisson_clip')
    clip.run=lambda **kw:{'candidate_mesh':str(clip_mesh)}
    pipeline = types.ModuleType('gsrefine.cleanup.direct_r3_pipeline')
    pipeline.run=lambda **kw:{'status':'BLOCKED_PRESEAL',
                              'preseal_report':str(preseal),
                              'safeguards_passed':True,
                              'main_surface':str(run/'poisson_roi_main.ply')}
    validation = types.ModuleType('gsrefine.cleanup.stage2_geometry_validation')
    validation.validate_parity_geometry=lambda *a,**kw:{'summary':{'status':'VALIDATED_SEALLESS_DISTANCE_GEOMETRY'}}
    monkeypatch.setitem(sys.modules,'gsrefine.cleanup.sdf_common',sdf)
    monkeypatch.setitem(sys.modules,'gsrefine.geometry.r3_sfm_bridge',sfm)
    monkeypatch.setitem(sys.modules,'gsrefine.geometry.r3_direct_poisson_clip',clip)
    monkeypatch.setitem(sys.modules,'gsrefine.cleanup.direct_r3_pipeline',pipeline)
    monkeypatch.setitem(sys.modules,'gsrefine.cleanup.stage2_geometry_validation',validation)
    # Force the new-geometry branch rather than picking the fixture's ready file.
    monkeypatch.setattr(runner, '_discover_preseal', lambda *a: None)
    report, result = runner._prepare_required_geometry(manifest, run)
    assert report==preseal
    assert result['status']=='VALIDATED_SEALLESS_DISTANCE_GEOMETRY'
    assert result['historical_seal_trial_status']=='BLOCKED_PRESEAL'


def test_duplicate_same_geometry_reports_do_not_block_onepass(tmp_path,monkeypatch):
    import gsrefine.cleanup.stage2_onepass as runner
    run=tmp_path/'roi_001'; root=run/'cleanup/geometry/direct_r3_sdf'
    for n in ('a','b'):
        p=root/n/'preseal_diagnostics.json';p.parent.mkdir(parents=True)
        p.write_text(json.dumps({'main_observed_surface_sha256':'same',
                                 'auxiliary_observed_surface_sha256':None,
                                 'sfm_fit':{'p95':.1,'source_core_sfm_count':101},
                                 'stage1_manifest':str(run/'metadata/manifest.json')}))
    module=types.ModuleType('gsrefine.cleanup.stage2_geometry_validation')
    module.validate_parity_geometry=lambda *a,**kw:{'summary':{'status':'OK'}}
    monkeypatch.setitem(sys.modules,'gsrefine.cleanup.stage2_geometry_validation',module)
    got=runner._discover_preseal(run/'metadata/manifest.json', run)
    assert got==root/'a/preseal_diagnostics.json'
    d=json.loads((root/'b/preseal_diagnostics.json').read_text())
    d['main_observed_surface_sha256']='different'
    (root/'b/preseal_diagnostics.json').write_text(json.dumps(d))
    with pytest.raises(ValueError,match='DIFFERENT authenticated'):
        runner._discover_preseal(run/'metadata/manifest.json', run)
