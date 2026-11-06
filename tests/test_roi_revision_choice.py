"""Each valid annotation saves a numbered ROI; the scene reference stays unchanged."""
import copy
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from gsrefine.roi_init.results import current_canonical, read_manifest, save_confirmed_roi
from gsrefine.roi_init.server import make_handler, build_meta
from gsrefine.roi_init.settings import InitSettings
from tests.test_roi_init import prepare, annotate


def _two_roi(settings):
    first = annotate(settings)
    changed = copy.deepcopy(first)
    changed['views'][0]['polygon'][0][0] += 0.08
    return first, changed


def test_explicit_revision_preserves_canonical(tmp_path):
    settings = InitSettings.load(prepare(tmp_path))
    first, changed = _two_roi(settings)
    original = save_confirmed_roi(first, settings)
    assert Path(original['saved_path']).name.endswith('001.json')
    original_bytes = Path(original['saved_path']).read_bytes()

    revised = save_confirmed_roi(changed, settings)
    assert Path(revised['saved_path']).name.endswith('002.json')
    assert revised['saved_as_revision'] is True
    assert revised['canonical_path'] == original['saved_path']
    assert Path(original['saved_path']).read_bytes() == original_bytes
    assert current_canonical(settings) == Path(original['saved_path'])
    assert 'save_mode' not in json.loads(Path(revised['saved_path']).read_text())
    manifest = read_manifest(settings.out_dir)
    assert len(manifest['records']) == 2
    assert manifest['records'][1]['record_type'] == 'revision'
    assert manifest['scene_canonical'][settings.scene] == Path(original['saved_path']).name


def test_http_auto_revision_never_overwrites(tmp_path):
    settings = InitSettings.load(prepare(tmp_path))
    original, changed = _two_roi(settings)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(
        settings, Path(__file__).resolve().parents[1]/'ui', build_meta(settings)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}/api/confirm-roi'

    def post(data):
        request = urllib.request.Request(url, data=json.dumps(data).encode(),
                                         headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    try:
        saved = post(original)
        revised = post(changed)
        assert revised['ok'] and revised['verified_on_disk']
        assert revised['saved_as_revision'] and revised['saved_path'].endswith('002.json')
        assert Path(saved['saved_path']).exists()
        assert current_canonical(settings) == Path(saved['saved_path'])
        bad = copy.deepcopy(changed)
        bad['save_mode'] = 'force_overwrite'
        with pytest.raises(urllib.error.HTTPError) as error:
            post(bad)
        assert error.value.code == 400
    finally:
        server.shutdown();server.server_close();thread.join(timeout=3)


def test_frontend_saves_new_roi_and_tracks_latest_without_replacing_reference():
    ui = Path(__file__).resolve().parents[1]/'ui'
    app = (ui/'app.mjs').read_text()
    html = (ui/'index.html').read_text()
    assert 'askRevisionChoice' not in app
    assert 'revision-overlay' not in html
    assert 'meta.latest_roi={...obj,path:answer.latest_path||savedPath}' in app
    assert 'if(!answer.saved_as_revision)' in app
