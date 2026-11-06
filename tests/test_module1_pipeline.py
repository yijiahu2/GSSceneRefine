"""Module 1 auto integration; no Open3D, Windows service or large real-world data required."""
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

from gsrefine.geometry import prepare as geo
from tests.test_roi_init import prepare as make_scene


def build_auto_config(tmp_path):
    config = make_scene(tmp_path)
    data = yaml.safe_load(config.read_text(encoding='utf8'))
    data['source_model_scope'] = 'full_reconstruction'
    source = tmp_path / 'points3D.bin'
    source.write_bytes(b'input COLMAP synthetic reconstruction')
    data['module1'] = {'mode': 'auto', 'colmap_points': str(source),
                       'output_dir': str(tmp_path / 'geometry_cache'),
                       'poisson': {'depth': 8},
                       'sfm_clean': {'do_sor': False}}
    config.write_text(yaml.safe_dump(data), encoding='utf8')
    return config, source


def mock_geometry(tmp_path, monkeypatch):
    data = np.load(tmp_path / 'sfm.npz')
    xyz = data['xyz']
    rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)
    errors = np.full(len(xyz), .1)
    tracks = np.full(len(xyz), 8, dtype=np.int32)
    ids = np.arange(len(xyz), dtype=np.uint64)
    monkeypatch.setattr(geo, 'load_colmap_points3d', lambda path: (xyz, rgb, errors, tracks, ids))
    called = []
    def fake_poisson(sfm_path, output_dir, **params):
        called.append((sfm_path, params))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / '04_poisson_raw.ply').write_bytes((tmp_path/'mesh.ply').read_bytes())
        np.save(output_dir / 'poisson_vertex_density.npy', np.ones(4))
        (output_dir / 'reconstruction_report_poisson.json').write_text('{}')
        return {'mesh_faces': len(trimesh.load(tmp_path/'mesh.ply').faces)}
    monkeypatch.setattr(geo, 'make_poisson', fake_poisson)
    return called


def test_auto_prepares_and_reuses_identical_geometry(tmp_path, monkeypatch):
    cfg, sfm_source = build_auto_config(tmp_path)
    called = mock_geometry(tmp_path, monkeypatch)
    before = (tmp_path / 'model.ply').read_bytes()
    active = geo.prepare_scene(cfg)
    assert active.is_file() and len(called) == 1
    data = yaml.safe_load(active.read_text())
    assert data['module1']['mode'] == 'prepared'
    assert Path(data['paths']['poisson_raw']).is_file()
    assert Path(data['paths']['clean_sfm_npz']).is_file()
    assert data['paths']['gaussian_ply'] == str(tmp_path / 'model.ply')
    assert (tmp_path / 'model.ply').read_bytes() == before
    assert (tmp_path / 'geometry_cache/poisson_vertex_density.npy').is_file()
    assert (tmp_path / 'geometry_cache/reconstruction_report_poisson.json').is_file()
    assert geo.prepare_scene(cfg) == active
    assert len(called) == 1
    assert geo.prepare_scene(active) == active
    assert json.loads((tmp_path / 'geometry_cache/module1_status.json').read_text())['cached']
    # Replacing SfM with another reconstruction under the same scene would
    # invalidate previously saved ROIs across methods, so fail closed.
    sfm_source.write_bytes(b'changed COLMAP reconstruction')
    with pytest.raises(geo.GeometryPreparationError, match='COLMAP points changed'):
        geo.prepare_scene(cfg)
    assert len(called) == 1


def test_auto_rejects_aabb_and_uses_current_original_config(tmp_path, monkeypatch):
    config, source = build_auto_config(tmp_path)
    data = yaml.safe_load(config.read_text())
    data['source_model_scope'] = 'aabb_baseline_only'
    config.write_text(yaml.safe_dump(data))
    with pytest.raises(geo.GeometryPreparationError, match='full_reconstruction'):
        geo.prepare_scene(config)
    data['module1']['mode'] = 'reuse'
    config.write_text(yaml.safe_dump(data))
    assert geo.prepare_scene(config) == config.resolve()


def test_auto_missing_source_does_not_delete_existing_outputs(tmp_path):
    config, source = build_auto_config(tmp_path)
    source.unlink()
    protected = tmp_path / 'geometry_cache' / '04_poisson_raw.ply'
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b'prior data')
    with pytest.raises(FileNotFoundError):
        geo.prepare_scene(config)
    assert protected.read_bytes() == b'prior data'


def test_cli_raw_inputs_create_runtime_without_overwriting_baseline(tmp_path, monkeypatch):
    """The original AABB YAML remains intact while the full-scene path is active."""
    config, source = build_auto_config(tmp_path)
    data = yaml.safe_load(config.read_text(encoding='utf8'))
    data.pop('module1', None)
    data['source_model_scope'] = 'aabb_baseline_only'
    original_gaussian = data['paths']['gaussian_ply']
    config.write_text(yaml.safe_dump(data))
    full = tmp_path / 'complete.ply'
    full.write_bytes((tmp_path / 'model.ply').read_bytes())
    from gsrefine.geometry.prepare import prepare_input_config
    prepared_config = prepare_input_config(config, colmap_points=source, gaussian_ply=full)
    assert prepared_config.is_file()
    configured = yaml.safe_load(prepared_config.read_text())
    assert configured['module1']['mode'] == 'auto'
    assert configured['paths']['gaussian_ply'] == str(full)
    assert configured['roi_init']['gaussian_ply'] == str(full)
    assert configured['source_model_scope'] == 'full_reconstruction'
    assert yaml.safe_load(config.read_text())['paths']['gaussian_ply'] == original_gaussian
    mock_geometry(tmp_path, monkeypatch)
    active = geo.prepare_scene(prepared_config)
    assert yaml.safe_load(active.read_text())['module1']['mode'] == 'prepared'
    assert yaml.safe_load(active.read_text())['paths']['gaussian_ply'] == str(full)


def test_cli_refuses_configured_aabb_as_full(tmp_path):
    config, source = build_auto_config(tmp_path)
    data = yaml.safe_load(config.read_text())
    data['source_model_scope'] = 'aabb_baseline_only'
    config.write_text(yaml.safe_dump(data))
    with pytest.raises(geo.GeometryPreparationError, match='AABB baseline'):
        geo.prepare_input_config(config, colmap_points=source,
                                 gaussian_ply=tmp_path / 'model.ply')


def test_new_scene_must_supply_both_colmap_sources(tmp_path):
    cfg, old = build_auto_config(tmp_path)
    from gsrefine.geometry.prepare import prepare_input_config
    images = tmp_path / 'images.bin'
    images.write_bytes(b'matching-camera-file')
    with pytest.raises(geo.GeometryPreparationError, match='--colmap-points'):
        prepare_input_config(cfg, scene='anothercliff', method='vanilla',
                             gaussian_ply=tmp_path/'model.ply', colmap_images=images)
    generated = prepare_input_config(cfg, scene='anothercliff', method='vanilla',
                                     gaussian_ply=tmp_path/'model.ply',
                                     colmap_points=old, colmap_images=images)
    data = yaml.safe_load(generated.read_text())
    assert data['scene'] == {'name': 'anothercliff', 'method': 'vanilla'}
    assert data['module1']['output_dir'].endswith('/runs/geometry/anothercliff')
    assert data['roi_init']['orientation']['frame_id'] == 'anothercliff_colmap_original'
    assert data['roi_init']['orientation']['colmap_images'] == str(images)


def test_user_defined_sfm_folder_and_final_output_root(tmp_path, monkeypatch):
    """CLI user paths must travel to module1 and the final ROI result root."""
    sparse = tmp_path / 'external_sparse' / '0'
    sparse.mkdir(parents=True)
    (sparse / 'points3D.bin').write_bytes(b'colmap point source')
    (sparse / 'images.bin').write_bytes(b'colmap camera source')
    gaussian = tmp_path / 'complete_gaussians.ply'
    gaussian.write_bytes(b'ply source')
    root = tmp_path / 'requested_final_outputs'
    from gsrefine.roi_init import server
    from gsrefine.roi_init.workflow import workflow_options, output_for_roi
    from gsrefine.geometry.prepare import expand_colmap_inputs, prepare_input_config

    config, _ = build_auto_config(tmp_path)
    points, images = expand_colmap_inputs(sparse)
    assert Path(points) == sparse / 'points3D.bin'
    assert Path(images) == sparse / 'images.bin'
    with pytest.raises(geo.GeometryPreparationError, match='either --sfm-dir'):
        expand_colmap_inputs(sparse, colmap_points=points)
    with pytest.raises(geo.GeometryPreparationError, match='Missing required COLMAP'):
        expand_colmap_inputs(tmp_path)

    prepared = prepare_input_config(config, colmap_points=points, colmap_images=images,
                                    gaussian_ply=gaussian, final_output_root=root)
    data = yaml.safe_load(prepared.read_text(encoding='utf8'))
    assert data['paths']['gaussian_ply'] == str(gaussian)
    assert data['paths']['colmap_points'] == str(sparse/'points3D.bin')
    assert data['roi_init']['orientation']['colmap_images'] == str(sparse/'images.bin')
    assert workflow_options(prepared)['output_root'] == root
    assert data['module1']['output_dir'] != str(root)
    assert yaml.safe_load(config.read_text(encoding='utf8')).get('workflow', {}).get('output_root') != str(root)

    roi = tmp_path / 'confirmed_roi_007.json'
    roi.write_text(json.dumps({'annotation_metadata': {
        'source_ply': str(gaussian), 'method': data['scene']['method']}}))
    assert output_for_roi(prepared, roi) == root / 'confirmed_roi_007'

    captured = []
    class StopBeforeBuilding(Exception):
        pass
    def capture(cfg):
        captured.append(Path(cfg))
        raise StopBeforeBuilding
    monkeypatch.setattr(server, 'prepare_scene', capture)
    with pytest.raises(StopBeforeBuilding):
        server.main(['--config', str(config), '--sfm-dir', str(sparse),
                     '--gaussian-ply', str(gaussian), '--output-dir', str(root)])
    assert len(captured) == 1
    assert workflow_options(captured[0])['output_root'] == root
