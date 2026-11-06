from pathlib import Path
import yaml
from tools.activate_roi_init import activate


def test_activation_preserves_original_comments_and_is_idempotent(tmp_path,monkeypatch):
    import tools.activate_roi_init as mod
    root=Path(__file__).resolve().parents[1]
    # Root constant is patchable only for the test; actual project never changes.
    monkeypatch.setattr(mod,'ROOT',root)
    original='''# USER COMMENT\npaths:\n  gaussian_ply: /mnt/f/example.ply\n  output_dir: runs/example\n'''
    p=tmp_path/'songziguanyin.yaml'
    p.write_text(original)
    assert activate(p,'songziguanyin','absgs')
    assert '# USER COMMENT' in p.read_text()
    assert yaml.safe_load(p.read_text())['roi_init']['orientation']['mode'] == 'auto'
    assert activate(p,'songziguanyin','absgs') is False
