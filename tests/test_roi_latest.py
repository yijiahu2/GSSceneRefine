"""Module 3 must consume latest *confirmed* scene ROI across methods."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from gsrefine.roi_init.settings import InitSettings
from gsrefine.roi_init.results import (
    current_canonical, current_latest, save_confirmed_roi,
    read_manifest, write_atomic,
)
from gsrefine.roi_init.handoff import main as refine_main
from tests.test_roi_init import annotate, prepare


def test_latest_updates_without_overwriting_scene_reference(tmp_path, capsys):
    config = prepare(tmp_path)
    settings = InitSettings.load(config)
    a = annotate(settings)
    first = save_confirmed_roi(a, settings)
    first_content = Path(first['saved_path']).read_bytes()
    assert current_canonical(settings) == current_latest(settings)

    b = copy.deepcopy(a)
    b['views'][0]['polygon'][0][0] += 0.03
    second = save_confirmed_roi(b, settings)
    assert second['saved_path'].endswith('_002.json')
    assert second['latest_path'] == second['saved_path']
    assert current_latest(settings) == Path(second['saved_path'])
    assert current_canonical(settings) == Path(first['saved_path'])
    assert Path(first['saved_path']).read_bytes() == first_content

    # No full Gaussian computation is launched in dry inspection mode.
    selected = refine_main(['--config', str(config), '--show-roi'])
    assert selected == Path(second['saved_path'])
    assert 'latest confirmed' in capsys.readouterr().out

    # Explicitly pin an earlier version for reproducibility.
    pinned = refine_main(['--config', str(config), '--roi', first['saved_path'], '--show-roi'])
    assert pinned == Path(first['saved_path'])


def test_latest_is_scene_wide_not_highest_method_number(tmp_path):
    config = prepare(tmp_path)
    settings = InitSettings.load(config)
    for i in range(3):
        a = annotate(settings)
        a['views'][0]['polygon'][0][0] += i * 0.01
        first = save_confirmed_roi(a, settings)
    assert first['saved_path'].endswith('absgs_roi_003.json')

    cfg = yaml.safe_load(config.read_text())
    cfg['scene']['method'] = 'edepth'
    config.write_text(yaml.safe_dump(cfg))
    other = InitSettings.load(config)
    saved = save_confirmed_roi(annotate(other), other)
    assert saved['saved_path'].endswith('edepth_roi_001.json')
    assert current_latest(other) == Path(saved['saved_path'])
    assert current_canonical(other).name.endswith('absgs_roi_001.json')
    assert read_manifest(other.out_dir)['scene_latest'][other.scene] == Path(saved['saved_path']).name
    assert refine_main(['--config', str(config), '--show-roi']) == Path(saved['saved_path'])


def test_legacy_manifest_can_resolve_last_record_and_migrate_on_save(tmp_path):
    config = prepare(tmp_path)
    settings = InitSettings.load(config)
    first = save_confirmed_roi(annotate(settings), settings)
    second = save_confirmed_roi(annotate(settings), settings)
    m = read_manifest(settings.out_dir)
    del m['scene_latest']
    write_atomic(settings.out_dir/'manifest.json', m)
    assert current_latest(settings) == Path(second['saved_path'])
    third = save_confirmed_roi(annotate(settings), settings)
    assert current_latest(settings) == Path(third['saved_path'])
    assert current_canonical(settings) == Path(first['saved_path'])


def test_manifest_latest_pointer_must_match_record_order(tmp_path):
    config = prepare(tmp_path)
    settings = InitSettings.load(config)
    first = save_confirmed_roi(annotate(settings), settings)
    save_confirmed_roi(annotate(settings), settings)
    m = read_manifest(settings.out_dir)
    m['scene_latest'][settings.scene] = Path(first['saved_path']).name
    write_atomic(settings.out_dir/'manifest.json', m)
    with pytest.raises(ValueError, match='latest pointer conflicts'):
        current_latest(settings)


def test_empty_scene_has_no_latest(tmp_path):
    settings = InitSettings.load(prepare(tmp_path))
    assert current_latest(settings) is None
