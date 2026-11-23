import os
from hashlib import sha256

import numpy as np
import pytest

from gsrefine.roi import stage1_runtime as runtime
from gsrefine.roi.core import ROI, Projection
from gsrefine.roi.two_stage_sdf import CoreImplicitField, SafetyConfig, _project_to_boundary, roi_polygon_implicit_field


@pytest.mark.parametrize('count', [1, 29, 2049])
def test_rotated_concave_roi_and_ball_seams_are_bitwise_equal(count):
    rng = np.random.default_rng(583)
    basis, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    poly = np.array([[-1,-1], [1,-1], [1,1], [.2,1], [.2,.1], [-1,1]], float)
    roi = ROI([Projection("top", np.zeros(3), basis[:2], poly),
               Projection("front", np.zeros(3), basis[1:], poly[::-1])])
    pts = rng.uniform(-1.4, 1.4, (count, 3))
    pts[0] = poly[3] @ basis[:2]
    field = CoreImplicitField(roi_polygon_implicit_field(roi), rng.normal(size=(32,3)), .3)
    cfg = SafetyConfig(projection_tolerance=1e-7)
    def compute(version):
        with runtime.session(version):
            d = field.evaluate(pts)
            out, residual = _project_to_boundary(field, pts, cfg, initial_values=d if version==2 else None)
            return [roi.distances(pts), d, out, residual]
    for before, after in zip(compute(1), compute(2)):
        assert before.tobytes() == after.tobytes()


def test_digest_scope_and_input_replacement(tmp_path):
    path = tmp_path/'model.ply'
    path.write_bytes(b'geometry')
    with runtime.session():
        runtime.watch_inputs([path])
        assert runtime.digest(path) == runtime.digest(path)
        assert runtime.report()['hash_cache_hits'] == 1
        old_time = path.stat().st_mtime_ns
        new = tmp_path/'new'
        new.write_bytes(b'GEOMETRY')
        os.utime(new, ns=(old_time,old_time))
        os.replace(new,path)
        assert runtime.digest(path) == sha256(b'GEOMETRY').hexdigest()
        with pytest.raises(ValueError, match="source changed"):
            runtime.check_inputs()
    with runtime.session():
        runtime.digest(path)
        assert runtime.report()['hash_cache_hits'] == 0


def test_fast_publication_verifies_once_and_rejects_corrupt_copy(tmp_path, monkeypatch):
    from gsrefine.geometry import r3_sfm_bridge as b
    from gsrefine.roi.two_stage_sdf import SafetyConfig
    rng = np.random.default_rng(25)
    xyz = rng.normal(0., .05, size=(240,3))
    clean = tmp_path/'clean.npz'
    np.savez(clean, xyz=xyz, ids=np.arange(len(xyz)))
    paths = dict(clean_sfm_npz=clean, roi_json=tmp_path/'roi.json',
                 poisson_mesh=tmp_path/'mesh.ply', source_full_gaussian_ply=tmp_path/'full.ply',
                 gaussian_ply=tmp_path/'final'/'gaussian_roi.ply')
    for key in ('roi_json', 'poisson_mesh','source_full_gaussian_ply'):
        paths[key].write_text(key)
    model = tmp_path/'gaussian_roi.ply'
    model.write_bytes(b'output')
    cfg = SafetyConfig(k_neighbors=16)
    field = CoreImplicitField(lambda p:p[:,0]-.5, np.empty((0,3)), 0.)
    work=tmp_path/'work'
    meta=tmp_path/'meta'
    with runtime.session():
        b.save_r3_sfm_handoff(work_dir=work, clean_sfm_npz=clean, roi_json=paths['roi_json'],
            gaussian_ply=paths['source_full_gaussian_ply'], poisson_mesh=paths['poisson_mesh'],
            core_field=field, safety_config=cfg, gaussian_positions=xyz, major_axes=np.ones(len(xyz)))
        original_copy = b.shutil.copy2
        monkeypatch.setattr(b.shutil,'copy2',lambda src,dst: dst.write_bytes(b'wrong bytes'))
        with pytest.raises(ValueError,match='differs from verified'):
            b.publish_stage1_r3_sfm(downstream=paths, metadata_dir=meta, work_dir=work, gaussian_source_file=model)
        assert not (meta/b.PUBLISHED_NAME).exists()
        monkeypatch.setattr(b.shutil,'copy2', original_copy)
        loads = []
        original_load = b._load_verified_handoff
        def counted(*args, **kwargs):
            loads.append(args[0])
            return original_load(*args, **kwargs)
        monkeypatch.setattr(b,'_load_verified_handoff',counted)
        dest, report = b.publish_stage1_r3_sfm(downstream=paths, metadata_dir=meta, work_dir=work, gaussian_source_file=model)
        assert len(loads)==1 and dest.is_file() and report['core_points']==len(xyz)
        assert runtime.report()['steps']['npz_stream_write']['calls'] > 0
