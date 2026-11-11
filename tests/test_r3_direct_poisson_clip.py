"""Read-only R3 direct clipping experiments on synthetic controlled geometry.

Real Songzi Guanyin model files remain external to this test; mock the costly
R3 replay only, not the polygon intersection and provenance safeguards.
"""
from __future__ import annotations
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

from gsrefine.geometry import r3_direct_poisson_clip as mod


def square_mesh():
    # A single, connected source with one crossing through x=0.  The clipping
    # must yield shared-edge cuts and must NOT delete internal faces by SfM KNN.
    verts = np.array([[-2, -1, 0], [1, -1, 0], [1, 1, 0], [-2, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def test_source_clip_preserves_surface_and_has_open_boundary():
    from gsrefine.geometry.mesh_clip import clip_piecewise_linear as _clip_piecewise_linear
    original = square_mesh()
    field = np.asarray(original.vertices)[:, 0]
    out, report = _clip_piecewise_linear(original, field)
    assert out.area == pytest.approx(2.0, abs=1e-7)
    assert len(out.faces) >= 2
    assert report['partly_clipped_triangles'] == 2
    assert np.min(out.vertices[:, 0]) >= -1e-9
    assert np.max(original.vertices[:, 0]) == 1.0  # unchanged source
    topology = mod._topology(out)
    assert topology['open_boundary_edges'] > 0
    assert topology['nonmanifold_edges'] == 0


def test_fail_missing_postrestoration_handoff(tmp_path):
    with pytest.raises(FileNotFoundError, match='stage2-sfm'):
        mod._source_r3_handoff(tmp_path, {'stage2_inputs':{}})


def test_skips_legacy_safety_record_and_finds_v3(tmp_path):
    folder=tmp_path/'cleanup/inputs'
    folder.mkdir(parents=True)
    old=tmp_path/'metadata/legacy_v2.npz'
    old.parent.mkdir()
    np.savez(old, schema=np.array('GSSceneRefine.r3_local_safety.v2'))
    v3=folder/'stage2_ready_r3_sfm_v3.npz'
    np.savez(v3, schema=np.array('GSSceneRefine.r3_sfm_handoff.v3'))
    assert mod._source_r3_handoff(tmp_path, {'stage2_inputs':{'r3_local_safety_npz':str(old)}})==v3.resolve()


def test_full_mocked_run_only_writes_candidate_and_metadata(tmp_path, monkeypatch):
    root = tmp_path/'scene/methods/absgs/roi_001'
    (root/'metadata').mkdir(parents=True)
    (root/'cleanup/inputs').mkdir(parents=True)
    original = square_mesh()
    source_mesh = tmp_path/'module1_poisson.ply'
    original.export(source_mesh)
    source_gaussian = tmp_path/'original_gaussian.ply'
    source_gaussian.write_bytes(b'original-gaussian-immutable')
    roi_gaussian = root/'final/gaussian_roi.ply'
    roi_gaussian.parent.mkdir(parents=True)
    roi_gaussian.write_bytes(b'R3-gaussian-immutable')
    roi_json = root/'metadata/roi.json'
    roi_json.write_text(json.dumps({'annotation_metadata':{'orientation_verified':True}}))
    clean = tmp_path/'cleaned_sfm.npz'
    np.savez(clean, xyz=np.array([[1.,0.,0.],[0.5,0,0.],[-.5,0,0.]]))
    sfm = root/'cleanup/inputs/sfm_roi_local_v3.npz'
    np.savez(sfm, xyz=np.array([[1.,0.,0.],[.5,0,0.],[-.5,0,0.]]),
             in_core_roi=np.array([True,True,False]))
    handoff = root/'cleanup/inputs/stage2_ready_r3_sfm_v3.npz'
    np.savez(handoff, schema=np.array('GSSceneRefine.r3_sfm_handoff.v3'))
    cfg = {'paths':{'gaussian_ply':str(source_gaussian)}}
    (root/'metadata/effective_scene.yaml').write_text(yaml.safe_dump(cfg))
    manifest_file = root/'metadata/manifest.json'
    manifest_file.write_text(json.dumps({'status':'completed'}))
    upstream={'source_gaussian':str(source_gaussian),'stage2_inputs':{}}
    inputs={'gaussian_ply':roi_gaussian,'poisson_mesh':source_mesh,
            'roi_json':roi_json,'clean_sfm_npz':clean}

    import gsrefine.cleanup.sdf_common as sdf_common
    import gsrefine.geometry.r3_sfm_bridge as v3
    monkeypatch.setattr(sdf_common, '_load_stage1_manifest',
                        lambda x:(manifest_file,upstream,inputs,root))
    monkeypatch.setattr(v3, 'read_verified_handoff', lambda *args, **kwargs:None)
    monkeypatch.setattr(mod,'_restore_r3_field',lambda **kwargs:(
        original, lambda points,**kwargs: np.asarray(points)[:,0],
        {'v3_handoff_audit_samples':512}))
    monkeypatch.setattr(mod,'_fit',lambda mesh,pts:{'distance_type':'mocked','points':len(pts)})
    before={p:mod._sha(p) for p in (source_mesh,source_gaussian,roi_gaussian,roi_json,clean,sfm,handoff)}
    result=mod.run(manifest=manifest_file,fit_max_points=2,audit_samples=32)
    assert result['status']=='CANDIDATE_ONLY_NOT_SEALED_NOT_GAUSSIAN_CLEANED'
    assert result['candidate_topology']['open_boundary_edges']>0
    assert all(mod._sha(p)==digest for p,digest in before.items())
    assert result['nonlinear_boundary_alias_audit']['sampled_original_faces']==2
    assert result['comparison']['global_R3_direct_clip']['fit_status']=='computed'
    assert not (root/'cleanup/final').exists()
    candidate=Path(result['candidate_mesh'])
    assert candidate.is_file()
    assert Path(candidate.parent/'clip_comparison_report.json').is_file()
    assert mod.run(manifest=manifest_file,fit_max_points=2,audit_samples=32)['output_sha256']==result['output_sha256']
    candidate.write_bytes(b'tampered mesh')
    with pytest.raises(ValueError,match='hashes mismatch'):
        mod.run(manifest=manifest_file,fit_max_points=2,audit_samples=32)
