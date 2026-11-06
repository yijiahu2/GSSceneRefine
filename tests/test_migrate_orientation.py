from pathlib import Path
import yaml
import pytest
import tools.migrate_orientation_auto as m


def test_null_orientation_preserves_user_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    f = tmp_path/'custom.yaml'
    original = '''# USER NOTE\nscene:\n  name: songziguanyin\n  method: absgs\npaths:\n  gaussian_ply: /mnt/f/modified/path.ply\nroi_init:\n  orientation:\n  # customized section below\n  web:\n    port: 9999\n  reuse_scene_roi: false\n'''
    f.write_text(original)
    assert m.migrate(f, dry_run=True)
    assert f.read_text() == original
    assert m.migrate(f)
    text=f.read_text()
    assert '# USER NOTE' in text and '/mnt/f/modified/path.ply' in text
    d=yaml.safe_load(text)
    assert d['roi_init']['web']['port']==9999
    assert d['roi_init']['reuse_scene_roi'] is False
    assert d['roi_init']['orientation']['mode']=='auto'
    assert d['roi_init']['orientation']['colmap_images'].endswith('/songziguanyin/sparse/images.bin')
    assert m.migrate(f) is False
    assert len(list((tmp_path/'runs/config_backups').glob('*.yaml'))) == 1


def test_preserve_other_orientation_options(tmp_path, monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    f=tmp_path/'cfg.yaml'
    f.write_text('''scene:\n  name: a_scene\nroi_init:\n  orientation:\n    up_axis: "+z" # legacy\n    frame_id: special_frame\n    verified: true\n    custom_parameter: 123\n  web:\n    port: 8765\n''')
    assert m.migrate(f, '/home/xth/custom/images.bin')
    d=yaml.safe_load(f.read_text())
    assert d['roi_init']['orientation']['frame_id']=='special_frame'
    assert d['roi_init']['orientation']['custom_parameter']==123
    assert d['roi_init']['orientation']['up_axis']=='+z'
    assert d['roi_init']['orientation']['mode']=='auto'
    assert d['roi_init']['orientation']['colmap_images']=='/home/xth/custom/images.bin'


def test_manual_refused_without_force(tmp_path, monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    f=tmp_path/'cfg.yaml'
    source='''scene:\n  name: s\nroi_init:\n  orientation:\n    mode: manual\n    manual_up_vector: [0, 0, 1]\n'''
    f.write_text(source)
    with pytest.raises(ValueError,match='manual'):
        m.migrate(f)
    assert f.read_text()==source


def test_existing_colmap_location_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    f=tmp_path/'cfg.yaml'
    f.write_text('''scene:\n  name: s\nroi_init:\n  orientation:\n    mode: auto\n    frame_id: s_colmap\n    colmap_images: /mnt/f/custom/images.bin\n  web:\n    port: 8765\n''')
    assert m.migrate(f) is False
    assert not (tmp_path/'runs').exists()


@pytest.mark.parametrize("empty_value", ["null", "~", "{}"])
def test_inline_empty_orientation(tmp_path, monkeypatch, empty_value):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    f = tmp_path/'cfg.yaml'
    f.write_text('scene:\n  name: s\nroi_init:\n  orientation: '+empty_value+'\n  web:\n    port: 8765\n')
    assert m.migrate(f)
    d=yaml.safe_load(f.read_text())
    assert d['roi_init']['web']['port']==8765
    assert d['roi_init']['orientation']['mode']=='auto'
