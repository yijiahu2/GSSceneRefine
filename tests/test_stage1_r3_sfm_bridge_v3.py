"""R3→SfM v3 regression: true restored-core field, 1x/2x, hashes and provenance."""
import hashlib
import sys
import types

import numpy as np
import pytest

from gsrefine.geometry.r3_sfm_bridge import (
    SCHEMA, SFM_SCHEMA, SELECTION, save_r3_sfm_handoff,
    read_verified_handoff, select_verified_clean_sfm, create_roi_sfm_v3)


class SyntheticRecoveredCore:
    """Initial ROI x<=0 PLUS a recovered detached sample around x=2."""
    def evaluate(self, xyz):
        xyz = np.asarray(xyz)
        return np.minimum(xyz[:, 0], np.linalg.norm(xyz - [2, 0, 0], axis=1) - .65)


class Config:
    trim_fraction = .05
    safety_factor = 1.
    distance_tolerance = 1e-6
    projection_tolerance = 1e-4
    k_neighbors = 16
    batch_size = 512
    def validate(self, count):
        assert count >= self.k_neighbors


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    module = types.ModuleType('gsrefine.roi.two_stage_sdf')
    # Stand-ins keep this unit test independent of Open3D / all source models.
    # The production code calls the actual unchanged Stage-1 R3 functions.
    module._project_to_boundary = lambda field, xyz, cfg: (xyz.copy(), np.zeros(len(xyz)))
    module.trimmed_knn_mean = lambda tree, axes, pts, **kwargs: np.full(len(pts), .3)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    x = np.linspace(-.9, 2.7, 73)
    y = np.linspace(-.4, .4, 13)
    xyz = np.array([[xi, yi, 0.] for xi in x for yi in y])
    clean = tmp_path/'clean_sfm.npz'
    ids = np.arange(len(xyz), dtype=np.int64)
    np.savez_compressed(clean, xyz=xyz, point3D_ids=ids,
                        track_length=np.ones(len(xyz), dtype=np.int32)*4,
                        nonrow_scalar=np.array('COLMAP-world'))
    roi = tmp_path/'roi.json'; roi.write_text('{"views":[]}', encoding='utf8')
    gaussian = tmp_path/'source_gaussians.ply'; gaussian.write_bytes(b'synthetic-full-gaussian')
    poisson = tmp_path/'poisson.ply'; poisson.write_bytes(b'synthetic-poisson')
    work = tmp_path/'work'; work.mkdir()
    rec = save_r3_sfm_handoff(work_dir=work, clean_sfm_npz=clean, roi_json=roi,
                             gaussian_ply=gaussian, poisson_mesh=poisson,
                             core_field=SyntheticRecoveredCore(), safety_config=Config(),
                             gaussian_positions=xyz, major_axes=np.ones(len(xyz)))
    return dict(tmp=tmp_path, record=rec, clean=clean, roi=roi,
                gaussian=gaussian, poisson=poisson, xyz=xyz, ids=ids)


def test_twice_safety_is_relative_to_recovered_core_not_initial_roi(synthetic):
    s = synthetic
    d, radii, tolerance, _ = read_verified_handoff(s['record'], clean_sfm_npz=s['clean'],
        roi_json=s['roi'], gaussian_ply=s['gaussian'], poisson_mesh=s['poisson'])
    assert radii.shape == (len(s['xyz']),)
    assert np.array_equal(d <= .3 + tolerance, d <= radii + tolerance)
    arrays, report = select_verified_clean_sfm(s['clean'], s['roi'],
        s['gaussian'], s['poisson'], s['record'])
    core = d <= radii + tolerance
    build = d <= 2*radii + tolerance
    assert report['schema'] == SFM_SCHEMA and report['selection_mode'] == SELECTION
    assert report['core_points'] == core.sum()
    assert report['build_points'] == build.sum()
    assert report['context_only_points'] == np.sum(build & ~core)
    np.testing.assert_array_equal(arrays['point3D_ids'], s['ids'][build])
    np.testing.assert_array_equal(arrays['in_core_roi'], core[build])
    np.testing.assert_array_equal(arrays['source_row_indices_in_clean_sfm'], np.flatnonzero(build))
    assert np.any((s['xyz'][:,0] > 1.5) & core), 'restored core must include points far beyond initial ROI'
    assert np.all(arrays['build_outward_distance_local']==2*arrays['stage1_r3_local_safety_distance'])
    assert arrays['nonrow_scalar'].item() == 'COLMAP-world'


def test_stable_output_reuse_and_source_tampering_is_rejected(synthetic):
    s=synthetic
    dest = s['tmp']/'final/sfm_roi.npz'
    meta = s['tmp']/'metadata/sfm_roi_report.json'
    args=dict(clean_sfm_npz=s['clean'], roi_json=s['roi'], gaussian_ply=s['gaussian'],
              poisson_mesh=s['poisson'], handoff_file=s['record'],
              output_npz=dest, output_report=meta)
    path, report = create_roi_sfm_v3(**args)
    assert path == dest.resolve() and report['sfm_roi_sha256'] == hashlib.sha256(dest.read_bytes()).hexdigest()
    reused, same = create_roi_sfm_v3(**args)
    assert reused == path and same == report
    s['poisson'].write_bytes(b'other-poisson')
    with pytest.raises(ValueError, match='source hashes differ'):
        select_verified_clean_sfm(s['clean'], s['roi'],s['gaussian'],s['poisson'],s['record'])
    with pytest.raises(ValueError, match='different algorithm/source'):
        create_roi_sfm_v3(**args)


def test_historical_backfill_reuses_original_study_geometry_without_gaussian_export(tmp_path, monkeypatch):
    """Exercise only the source reconstruction path, not Poisson/GS training."""
    import json
    import yaml
    import gsrefine.geometry.r3_sfm_bridge as bridge
    from scipy.spatial import cKDTree

    source = tmp_path/'full.ply'; source.write_bytes(b'full')
    final = tmp_path/'final.ply'; final.write_bytes(b'final')
    poisson = tmp_path/'poisson.ply'; poisson.write_bytes(b'poisson')
    roi = tmp_path/'roi.json'
    roi.write_text(json.dumps({'annotation_metadata': {'up_vector': [0,0,1]}}))
    clean = tmp_path/'clean.npz'
    xyz = np.column_stack([np.linspace(-1.,1.,250),np.zeros(250),np.zeros(250)])
    np.savez_compressed(clean,xyz=xyz)
    run = tmp_path/'roi_run'; (run/'metadata').mkdir(parents=True)
    config = {
        'paths':{'gaussian_ply':str(source)},
        'sfm':{'confidence_threshold':.75},
        'guards':{'geodesic_limit_frac':.04,'outside_search_frac':.025,'safety_knn':16},
        'roi_study':{},
    }
    (run/'metadata/effective_scene.yaml').write_text(yaml.safe_dump(config))
    gauss_xyz=np.column_stack([np.linspace(-1,1,400),np.zeros(400),np.zeros(400)])

    class MockGaussian:
        def __init__(self,path):
            self.path=path
            self.arr={f'scale_{k}':np.full(len(gauss_xyz),np.log(.1)) for k in range(3)}
        def positions(self):
            return gauss_xyz if str(self.path)==str(source) else gauss_xyz[gauss_xyz[:,0]<-.5]
    class MockROI:
        def distances(self,pts):return np.asarray(pts)[:,0,None]*-1
        @staticmethod
        def from_file(path):return MockROI()
    class FakeMesh:
        vertices=np.array([[0,0,0],[0,1,0],[1,0,0]],float)
        faces=np.array([[0,1,2]],int)
        edges_sorted=np.array([[0,1],[0,2],[1,2]],int)
    import trimesh
    monkeypatch.setattr(trimesh,'load',lambda *args,**kw:FakeMesh())
    roi_module = types.ModuleType('gsrefine.roi.core')
    roi_module.ROI=MockROI
    roi_module.GaussianPLY=MockGaussian
    roi_module.geometry_confidence=lambda *a,**k:None
    roi_module.load_sfm=lambda *a,**k:None
    monkeypatch.setitem(sys.modules,'gsrefine.roi.core',roi_module)
    study = types.ModuleType('gsrefine.roi.study')
    study.analyze=lambda *args,**kw:{'label_masks':{'suspect':np.array([False])},'inside':np.array([True])}
    study.anchored_local_recovery_faces=lambda *args,**kw:(np.array([False]),{})
    monkeypatch.setitem(sys.modules,'gsrefine.roi.study',study)
    sdf=types.ModuleType('gsrefine.roi.two_stage_sdf')
    sdf.SafetyConfig=lambda **kw: types.SimpleNamespace(**kw,validate=lambda n:None,
                                                         distance_tolerance=1e-6)
    sdf.CoreImplicitField=lambda *args,**kw:SyntheticRecoveredCore()
    sdf.roi_polygon_implicit_field=lambda roi:None
    sdf.poisson_surface_coverage=lambda *args:(np.empty((0,3)), {'surface_cover_error_bound':0.0})
    sdf.estimate_geometric_support_radius=lambda *args:(0.0,{})
    sdf.longest_gaussian_axis=lambda *args,**kw:np.full(len(gauss_xyz),.1)
    sdf._project_to_boundary=lambda field,pts,cfg:(pts.copy(),np.zeros(len(pts)))
    sdf.trimmed_knn_mean=lambda tree,axes,pts,**kw:np.full(len(pts),.3)
    monkeypatch.setitem(sys.modules,'gsrefine.roi.two_stage_sdf',sdf)
    inputs={'roi_json':roi,'clean_sfm_npz':clean,'poisson_mesh':poisson,'gaussian_ply':final}
    before={k:bridge.sha256(v) for k,v in inputs.items()}
    out = bridge._recompute_original_r3_field(run,inputs,{'source_gaussian':str(source)},tmp_path/'backfill')
    assert out.is_file()
    data = np.load(out)
    assert data['schema'].item()==SCHEMA
    assert data['point_count'].item()==len(xyz)
    assert all(bridge.sha256(v)==before[k] for k,v in inputs.items())


def test_legacy_v2_handoff_is_not_mistaken_for_post_restoration_r3(synthetic, monkeypatch):
    import shutil
    import gsrefine.geometry.r3_sfm_bridge as bridge
    s = synthetic
    run = s['tmp']/'historical_roi'
    meta = run/'metadata'; meta.mkdir(parents=True)
    old = meta/'stage1_r3_local_safety.npz'
    np.savez_compressed(old, schema=np.array('GSSceneRefine.r3_local_safety.v2'),
                        safety_distance_per_sfm_point=np.ones(len(s['xyz'])))
    final = run/'final/gaussian_roi.ply'
    final.parent.mkdir(); final.write_bytes(b'old-gaussian-roi-not-modified')
    backfill_called=[]
    def backfill(folder, inputs, upstream, work_dir):
        backfill_called.append(True)
        work_dir.mkdir(parents=True,exist_ok=True)
        p = work_dir/bridge.WORK_NAME
        shutil.copy2(s['record'],p)
        return p
    monkeypatch.setattr(bridge,'_recompute_original_r3_field',backfill)
    upstream = {'source_gaussian':str(s['gaussian']),
                'stage2_inputs':{'r3_local_safety_npz':str(old)}}
    inputs = {'clean_sfm_npz':s['clean'], 'roi_json':s['roi'],
              'poisson_mesh':s['poisson'], 'gaussian_ply':final}
    result, report, reused = bridge.resolve_stage2_r3_sfm(run,upstream,inputs)
    assert backfill_called and not reused
    assert result.is_file()
    assert report['schema']==SFM_SCHEMA
    assert np.load(old)['schema'].item().endswith('.v2')
