"""Generic scene/method contract: never mix COLMAP frames, models or output masks."""
import json
from pathlib import Path

import pytest
import yaml

from gsrefine.roi_init.settings import InitSettings
from gsrefine.roi_init.results import (
    save_confirmed_roi, require_orientation, current_latest_for_source,
)
from gsrefine.roi_init.workflow import output_for_roi, run_once
from tests.test_roi_init import annotate, prepare


def test_reuse_confirmed_full_scene_roi_in_another_method_without_overwriting(tmp_path, monkeypatch):
    config = prepare(tmp_path)
    data = yaml.safe_load(config.read_text(encoding='utf8'))
    data['source_model_scope'] = 'full_reconstruction'
    config.write_text(yaml.safe_dump(data), encoding='utf8')
    first = InitSettings.load(config)
    original = Path(save_confirmed_roi(annotate(first), first)['saved_path'])
    first_output = output_for_roi(config, original)
    first_output.mkdir(parents=True, exist_ok=True)
    preserved = first_output / 'gaussians_final_R3.ply'
    preserved.write_bytes(b'original-method-model')
    new_model = tmp_path / 'method_vanilla.ply'
    new_model.write_bytes(first.gaussian_ply.read_bytes())
    data['scene']['method'] = 'vanilla'
    data['paths']['gaussian_ply'] = str(new_model)
    data['roi_init']['gaussian_ply'] = str(new_model)
    config.write_text(yaml.safe_dump(data), encoding='utf8')
    second = InitSettings.load(config)
    payload = json.loads(original.read_text(encoding='utf8'))
    assert current_latest_for_source(second) is None
    with pytest.raises(ValueError, match='reuse-scene-roi'):
        require_orientation(payload, second)
    assert require_orientation(payload, second, reuse_scene_geometry=True)
    second_output = output_for_roi(config, original)
    assert second_output != first_output
    assert second_output.parent == first_output.parent
    assert 'vanilla' in second_output.name
    seen = []
    def fake_study(cfg, roi, out, **kwargs):
        seen.append((Path(cfg), Path(roi), Path(out), kwargs))
        final = Path(out) / 'gaussians_final_R3.ply'
        final.write_bytes(b'other-method-model')
        from gsrefine.roi_init.workflow import _sha
        return {'roi': str(original.resolve()), 'input_sha256': {'roi': _sha(original)},
                'R0_count': 5, 'final_gaussian_scene': {
                    'output_ply': str(final), 'method': 'R3',
                    'final_gaussians': 7, 'added_gaussians': 2}}
    monkeypatch.setattr('gsrefine.roi.study.run', fake_study)
    result = run_once(config, original, reuse_scene_roi=True)
    assert len(seen) == 1 and seen[0][2] == second_output
    assert result['target_method'] == 'vanilla'
    assert result['reused_roi_geometry'] is True
    assert preserved.read_bytes() == b'original-method-model'
    assert Path(result['final_ply']).read_bytes() == b'other-method-model'
    assert 'save_diagnostics' not in seen[0][3]  # original output behavior preserved


def test_roi_not_reusable_in_other_scene_or_aabb(tmp_path):
    config = prepare(tmp_path)
    data = yaml.safe_load(config.read_text(encoding='utf8'))
    data['source_model_scope'] = 'full_reconstruction'
    config.write_text(yaml.safe_dump(data), encoding='utf8')
    first = InitSettings.load(config)
    roi = json.loads(Path(save_confirmed_roi(annotate(first), first)['saved_path']).read_text())
    data['scene']['name'] = 'unrelated'
    config.write_text(yaml.safe_dump(data), encoding='utf8')
    with pytest.raises(ValueError, match='scene'):
        require_orientation(roi, InitSettings.load(config), reuse_scene_geometry=True)
    data['scene']['name'] = first.scene
    data['scene']['method'] = 'vanilla'
    data['source_model_scope'] = 'aabb_baseline_only'
    config.write_text(yaml.safe_dump(data), encoding='utf8')
    with pytest.raises(ValueError, match='complete'):
        require_orientation(roi, InitSettings.load(config), reuse_scene_geometry=True)
