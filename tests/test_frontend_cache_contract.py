"""Regression: browser must not mix legacy +Z frontend with vector SfM metadata."""
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

from gsrefine.roi_init.server import make_handler


class SettingsStub:
    poisson_ply = None
    ui_port = 8765


def test_first_party_code_never_served_as_cached_304():
    root = Path(__file__).resolve().parents[1] / 'ui'
    assert 'app.mjs?v=' in (root / 'index.html').read_text()
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(SettingsStub(), root, {'coordinate_frame': {'up_vector': [0, 1, 0]}}))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f'http://127.0.0.1:{server.server_port}'
        for path in ('/', '/index.html', '/app.mjs', '/geometry.mjs', '/poisson-overlay.mjs'):
            request = Request(base + path, headers={'If-Modified-Since': 'Wed, 04 Nov 2026 00:00:00 GMT'})
            with urlopen(request, timeout=5) as response:
                assert response.status == 200
                assert 'no-store' in response.headers['Cache-Control']
                assert response.read()
        with urlopen(base + '/api/meta', timeout=5) as response:
            assert 'no-store' in response.headers['Cache-Control']
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)
