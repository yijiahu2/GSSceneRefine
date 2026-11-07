"""Proves that newly computed Stage-1 exact faces can be handed off without recomputation."""
from pathlib import Path
import numpy as np
import trimesh
import pytest
from gsrefine.geometry.stage1_handoff import (
    save_computed_stage1_surface_faces, retain_stage1_surface_faces,
    read_verified_surface_faces,
)


def test_write_retain_reload_stage1_exact_faces(tmp_path):
    mesh=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]], faces=[[0,1,2]], process=False)
    raw=tmp_path/'raw.ply';mesh.export(raw)
    roi=tmp_path/'roi.json';roi.write_text('{"views":[]}')
    scratch=tmp_path/'scratch';metadata=tmp_path/'metadata'
    original=np.array([True],dtype=bool)
    save_computed_stage1_surface_faces(scratch, original, raw, roi, face_count=1)
    dest=retain_stage1_surface_faces(scratch, metadata, raw, roi)
    assert np.array_equal(read_verified_surface_faces(dest, raw, roi, triangle_count=1), original)
    with pytest.raises(FileExistsError):
        save_computed_stage1_surface_faces(scratch, original, raw, roi)


def test_append_generic_stage2_config_without_overwriting_existing(tmp_path):
    import importlib.util
    project=tmp_path/'checkout'
    path=project/'configs/default.yaml';path.parent.mkdir(parents=True)
    path.write_text('scene:\n  name: template\nworkflow:\n  final_method: R3\n')
    helper=Path(__file__).resolve().parents[1]/'tools/apply_stage2_handoff.py'
    spec=importlib.util.spec_from_file_location('stage2_patcher',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    status,old=module.ensure_default_parameters(project,tmp_path/'backup')
    assert status=='configured' and old.exists()
    content=path.read_text()
    assert content.count('stage2_cleanup:')==1
    assert 'name: template' in content
    status,_=module.ensure_default_parameters(project,tmp_path/'backup')
    assert status=='already_configured' and path.read_text()==content
