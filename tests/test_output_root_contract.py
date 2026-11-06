"""Regression checks for output isolation, publication, and legacy ROI reuse."""
import json
from pathlib import Path

import pytest
import yaml

from gsrefine.geometry.prepare import relocate_existing_reference_config
from gsrefine.roi_init.output_layout import run_dir, publish
from gsrefine.roi_init import workflow
from tests.test_roi_init import prepare


def test_run_path_is_scene_method_roi_under_user_root(tmp_path):
    roi = tmp_path / 'songziguanyin_absgs_roi_006.json'
    roi.write_text('{}')
    assert run_dir(tmp_path / 'results', 'songziguanyin', 'vanilla', roi) == (
        tmp_path/'results/songziguanyin/methods/vanilla/roi_006')
    assert run_dir(tmp_path / 'results', 'songziguanyin', 'absgs', roi) != run_dir(
        tmp_path / 'results', 'songziguanyin', 'vanilla', roi)


def test_legacy_read_only_geometry_does_not_write_to_repository(tmp_path):
    cfg = prepare(tmp_path)
    output = tmp_path / 'all_outputs'
    runtime = relocate_existing_reference_config(cfg, output)
    data = yaml.safe_load(runtime.read_text(encoding='utf8'))
    assert runtime.is_relative_to(output)
    assert data['module1']['mode'] == 'reuse'
    assert Path(data['roi_init']['output_dir']).is_relative_to(output)
    assert Path(data['roi_init']['orientation']['cache_dir']).is_relative_to(output)
    assert Path(data['workflow']['output_root']) == output
    assert Path(data['paths']['poisson_raw']).is_file()
    assert Path(data['paths']['clean_sfm_npz']).is_file()
    assert str(cfg.resolve()) != str(runtime.resolve())


def test_roi_auto_cli_rehomes_legacy_reference(tmp_path, monkeypatch):
    cfg = prepare(tmp_path)
    output = tmp_path / 'outputs'
    roi = tmp_path / 'roi_002.json'
    roi.write_text('{}')
    captured = []
    monkeypatch.setattr(workflow, 'run_once', lambda config, roi, root, **options:
                        captured.append((Path(config), Path(root), options)))
    workflow.main(['--config', str(cfg), '--roi', str(roi), '--output-dir', str(output)])
    assert captured[0][0].is_relative_to(output)
    assert captured[0][1] == output
    assert captured[0][2]['keep_diagnostics'] is False


def test_publish_never_exposes_final_ply_with_missing_downstream_inputs(tmp_path):
    work = tmp_path / 'scratch'
    work.mkdir()
    model = work / 'gaussians_final_R3.ply'
    model.write_bytes(b'not-a-real-ply-but-nonempty')
    cfg = tmp_path/'scene.yaml'
    cfg.write_text(yaml.safe_dump({'paths': {
        'gaussian_ply': str(tmp_path/'full_model.ply'),
        'poisson_raw': str(tmp_path/'missing_poisson.ply'),
        'clean_sfm_npz': str(tmp_path/'missing_sfm.npz')}}))
    roi = tmp_path/'roi_001.json'
    roi.write_text('{}')
    out = tmp_path/'results/roi_001'
    with pytest.raises(RuntimeError, match='downstream input'):
        publish(out, work, roi, cfg, {
            'R0_count': 2, 'input_sha256': {'roi': 'abc'},
            'final_gaussian_scene': {'output_ply': str(model), 'method': 'R3',
                                     'final_gaussians': 3, 'added_gaussians': 1}})
    assert model.is_file()
    assert not (out/'final/gaussian_roi.ply').exists()
    assert not (out/'metadata/manifest.json').exists()
