"""Exact per-location Stage-1 safety inheritance; no global scalar fallback."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from gsrefine.geometry.roi_sfm_local import (
    SAFETY_SCHEMA, ROI_SFM_SCHEMA, WORK_SAFETY_NAME,
    save_stage1_r3_local_safety, read_verified_local_safety,
    select_clean_sfm_local, create_exact_roi_sfm, validate_exact_roi_sfm,
    publish_stage1_exact_roi_sfm, resolve_stage2_exact_roi_sfm,
)


class View:
    def signed_and_normal(self, xyz):
        normals = np.zeros_like(xyz, dtype=float)
        normals[:, 0] = 1.0
        return xyz[:, 0].astype(float), normals


class FakeROI:
    def __init__(self):
        self.views = [View()]

    @staticmethod
    def from_file(_):
        return FakeROI()


@pytest.fixture
def scene(tmp_path, monkeypatch):
    core = types.ModuleType('gsrefine.roi.core')
    core.ROI = FakeROI
    monkeypatch.setitem(sys.modules, 'gsrefine.roi.core', core)
    xs = np.linspace(-3, 3, 49)
    ys = np.linspace(-2, 2, 49)
    xyz = np.array([(x, y, 8.0) for x in xs for y in ys], dtype=np.float64)
    sfm = tmp_path/'clean_sfm.npz'
    point_id = np.arange(len(xyz), dtype=np.int64)
    rgb = np.tile(np.array([5, 90, 190], dtype=np.uint8), (len(xyz), 1))
    np.savez_compressed(sfm, xyz=xyz, rgb=rgb, colmap_point_id=point_id,
                        reprojection_error=np.linspace(0, .5, len(xyz)))
    roi = tmp_path/'roi.json'; roi.write_text('{"views":[]}', encoding='utf8')
    gaussian = tmp_path/'gaussian_roi.ply'; gaussian.write_bytes(b'gaussian-unchanged')
    poisson = tmp_path/'poisson.ply'; poisson.write_bytes(b'poisson-unchanged')

    def actual_stage1_local_d(points):
        return .45 + .13*(points[:, 1] + 2)

    work = tmp_path/'work'
    record = save_stage1_r3_local_safety(work, clean_sfm_npz=sfm,
        roi_json=roi, safety_at=actual_stage1_local_d, chunk_size=127)
    return dict(tmp_path=tmp_path, sfm=sfm, roi=roi, gaussian=gaussian, poisson=poisson,
                work=work, record=record, xyz=xyz, safety_fn=actual_stage1_local_d)


def test_exact_selection_is_twice_safety_at_every_location(scene):
    s = scene
    arrays, report = select_clean_sfm_local(s['sfm'], s['roi'], s['record'])
    xyz = s['xyz']
    d = s['safety_fn'](xyz)
    core = xyz[:, 0] >= -d
    build = xyz[:, 0] >= -2*d
    assert report['schema'] == ROI_SFM_SCHEMA
    assert report['build_points'] == build.sum()
    assert report['core_points'] == core.sum()
    assert np.array_equal(arrays['colmap_point_id'], np.flatnonzero(build))
    assert np.array_equal(arrays['in_core_roi'], core[build])
    np.testing.assert_array_equal(arrays['stage1_r3_local_safety_distance'], d[build])
    np.testing.assert_array_equal(arrays['build_outward_distance_local'], 2*d[build])
    assert report['context_only_points'] > 0
    assert report['source_is_cleaned_stage1_sfm']
    assert report['gaussian_r3_unchanged'] and report['sdf_threshold_formula_unchanged']


def test_reject_scalar_and_wrong_clean_sfm(scene):
    s = scene
    with pytest.raises(ValueError, match='VECTOR'):
        save_stage1_r3_local_safety(s['tmp_path']/'wrong', clean_sfm_npz=s['sfm'],
            roi_json=s['roi'], per_point_safety=0.55)
    s['sfm'].write_bytes(s['sfm'].read_bytes() + b'changed')
    with pytest.raises(ValueError, match='SHA mismatch'):
        read_verified_local_safety(s['record'], clean_sfm_npz=s['sfm'], roi_json=s['roi'])


def test_publish_stage1_v2_and_stage2_reuse(scene):
    s = scene
    final = s['tmp_path']/'stage1/final'; final.mkdir(parents=True)
    g = final/'gaussian_roi.ply'; g.write_bytes(s['gaussian'].read_bytes())
    meta = s['tmp_path']/'stage1/metadata'; meta.mkdir()
    downstream = {'clean_sfm_npz': str(s['sfm']), 'roi_json': str(s['roi']),
                  'gaussian_ply': str(g), 'poisson_mesh': str(s['poisson'])}
    sfm_roi, report = publish_stage1_exact_roi_sfm(
        downstream=downstream, metadata_dir=meta, work_dir=s['work'])
    assert sfm_roi.is_file() and report['schema'] == ROI_SFM_SCHEMA
    assert 'r3_local_safety_npz' in downstream
    downstream['sfm_roi_npz'] = str(sfm_roi)
    upstream = {'stage2_inputs': downstream}
    inputs = {'clean_sfm_npz': s['sfm'], 'roi_json': s['roi'],
              'gaussian_ply': g, 'poisson_mesh': s['poisson']}
    reuse, rep, yes = resolve_stage2_exact_roi_sfm(s['tmp_path']/'stage1', upstream, inputs)
    assert yes and reuse == sfm_roi and rep == report
    # Old scalar v1 in the same manifest cannot bypass v2 source checks.
    assert validate_exact_roi_sfm(sfm_roi, meta/'roi_sfm_report.json',
        s['sfm'], s['roi'], g, s['poisson'], meta/'stage1_r3_local_safety.npz') == report


def test_old_stage1_can_generate_stage2_if_exact_field_saved(scene):
    s = scene
    run = s['tmp_path']/'oldrun'; (run/'metadata').mkdir(parents=True)
    target = run/'metadata/stage1_r3_local_safety.npz'
    target.write_bytes(s['record'].read_bytes())
    inputs = {'clean_sfm_npz': s['sfm'], 'roi_json': s['roi'],
              'gaussian_ply': s['gaussian'], 'poisson_mesh': s['poisson']}
    v2, report, reused = resolve_stage2_exact_roi_sfm(run, {'stage2_inputs': {}}, inputs)
    assert not reused and v2.name == 'sfm_roi_local_v2.npz'
    assert report['schema'] == ROI_SFM_SCHEMA
    v2b, report2, reused2 = resolve_stage2_exact_roi_sfm(run, {'stage2_inputs': {}}, inputs)
    assert v2b == v2 and report2 == report


def test_missing_field_fails_closed_and_does_not_make_scalar_guess(scene):
    s = scene
    inputs = {'clean_sfm_npz': s['sfm'], 'roi_json': s['roi'],
              'gaussian_ply': s['gaussian'], 'poisson_mesh': s['poisson']}
    run = s['tmp_path']/'no-safety'; run.mkdir()
    with pytest.raises(ValueError, match='global scalar/P95 fallback is disabled'):
        resolve_stage2_exact_roi_sfm(run, {'stage2_inputs': {}}, inputs)
    assert not (run/'cleanup/inputs').exists()


def test_stage1_publisher_guarded_source_edit(scene):
    s = scene
    file = s['tmp_path']/'repo/gsrefine/roi_init/output_layout.py'; file.parent.mkdir(parents=True)
    old = ("def publish():\n    downstream = {}\n    meta = None\n"
           "    # Stage 2 input: retain core SfM and one additional original safety band.\n"
           "    # If the local R3 safety was not published, the scalar-envelope fallback\n"
           "    # is labelled approximate in metadata/roi_sfm_report.json.\n"
           "    from gsrefine.geometry.roi_sfm import prepare_stage1_roi_sfm\n"
           "    _roi_sfm_path, _roi_sfm_report = prepare_stage1_roi_sfm(\n"
           "        downstream=downstream, metadata_dir=meta, work_dir=work)\n"
           "    downstream['sfm_roi_npz'] = str(_roi_sfm_path)\n")
    file.write_text(old)
    location = Path(__file__).resolve().parents[1]/'tools/apply_exact_sfm_handoff.py'
    spec = importlib.util.spec_from_file_location('patcher_exact', location)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    assert tool.install(s['tmp_path']/'repo', dry_run=True)[0] == 'would_patch'
    state, backup = tool.install(s['tmp_path']/'repo', backup_dir=s['tmp_path']/'backups')
    assert state == 'patched' and backup.is_file()
    assert tool.install(s['tmp_path']/'repo')[0] == 'already_installed'
    assert file.read_text().count('publish_stage1_exact_roi_sfm(') == 1


def test_gaussian_and_source_clean_sfm_are_never_modified(scene):
    s = scene
    import hashlib
    before = {key: hashlib.sha256(s[key].read_bytes()).hexdigest()
              for key in ('sfm', 'gaussian', 'poisson')}
    arrays, _ = select_clean_sfm_local(s['sfm'], s['roi'], s['record'])
    assert len(arrays['xyz']) >= 100
    assert all(hashlib.sha256(s[key].read_bytes()).hexdigest() == h for key, h in before.items())
