"""Visual-only SOG cache contracts: no modification to scientific input PLY."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path

import yaml

from gsrefine.web.preview_cache import cache_paths, file_signature, find_sog_cache
from gsrefine.web.server import create_app_state, make_handler
from tests.test_demo import build_synthetic_case


def fake_sog(source, output):
    target, manifest = cache_paths(source, output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target,'w') as z:
        z.writestr('meta.json', json.dumps({'count':200}))
    manifest.write_text(json.dumps({
        'source_signature': file_signature(source),
        'source_ply': str(source),
    }), encoding='utf-8')
    return target


def test_cache_validation_and_reuse(tmp_path):
    build_synthetic_case(tmp_path)
    source=tmp_path/'model.ply'
    out=tmp_path/'runs'
    assert find_sog_cache(source,out) is None
    cache=fake_sog(source,out)
    assert find_sog_cache(source,out)==cache
    old=cache.stat().st_size
    assert old>0
    # Changing the source invalidates the cache even if the Gaussian count is unchanged.
    source.touch()
    assert find_sog_cache(source,out) is None


def test_server_selects_sog_and_http_cache_headers(tmp_path):
    build_synthetic_case(tmp_path)
    src=tmp_path/'model.ply'
    out=tmp_path/'runs'
    target=fake_sog(src,out)
    cfg=yaml.safe_load((tmp_path/'config.yaml').read_text())
    state=create_app_state(cfg,src,tmp_path/'mesh.ply',out,100,
                           sfm_npz=tmp_path/'sfm.npz')
    assert state['display_ply']==target
    assert state['meta']['display_format']=='SOG'
    assert state['meta']['display_lossy'] is True
    assert state['meta']['display_gaussians']==state['meta']['gaussians']
    assert state['meta']['display_url'].startswith('/api/model.sog?v=')
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(state,Path(__file__).resolve().parents[1]/'ui'))
    t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
    url=f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(url+state['meta']['display_url']) as r:
            content=r.read()
            etag=r.headers['ETag']
            assert content==target.read_bytes()
            assert 'max-age=' in r.headers['Cache-Control']
            assert etag
        req=urllib.request.Request(url+state['meta']['display_url'],headers={'If-None-Match':etag})
        try:
            urllib.request.urlopen(req)
            raise AssertionError('Expected 304')
        except urllib.error.HTTPError as e:
            assert e.code==304
        req=urllib.request.Request(url+state['meta']['display_url'],headers={'Range':'bytes=0-3'})
        with urllib.request.urlopen(req) as r:
            assert r.status==206
            assert r.read()==content[:4]
        try:
            urllib.request.urlopen(url+'/api/model.ply')
            raise AssertionError('Inactive raw endpoint must not leak original PLY')
        except urllib.error.HTTPError as e:
            assert e.code==404
    finally:
        server.shutdown();server.server_close();t.join(timeout=3)


def test_raw_ply_force_and_manual_sog(tmp_path):
    build_synthetic_case(tmp_path)
    src=tmp_path/'model.ply'
    out=tmp_path/'runs'
    target=fake_sog(src,out)
    cfg=yaml.safe_load((tmp_path/'config.yaml').read_text())
    raw=create_app_state(cfg,src,tmp_path/'mesh.ply',out,100,
                         sfm_npz=tmp_path/'sfm.npz',raw_ply=True)
    assert raw['display_ply']==src
    assert raw['meta']['display_format']=='PLY'
    manual=create_app_state(cfg,src,tmp_path/'mesh.ply',out,100,
                            sfm_npz=tmp_path/'sfm.npz',display_sog=target)
    assert manual['display_ply']==target
    assert manual['meta']['display_format']=='SOG'


def test_sog_cache_converter_is_lossless_to_input_and_reuses_cache(tmp_path, monkeypatch):
    """Conversion emits visualization cache only; original scientific PLY is untouched."""
    import os
    from gsrefine.web import preview_cache

    build_synthetic_case(tmp_path)
    source = tmp_path / 'model.ply'
    before = source.read_bytes()
    converter = tmp_path / 'mock_splat_transform'
    converter.write_text(
        '#!/usr/bin/env python3\n'
        'import sys, zipfile\n'
        'from pathlib import Path\n'
        'src, dest = sys.argv[-2:]\n'
        'assert Path(src).is_file()\n'
        'with zipfile.ZipFile(dest,"w") as z: z.writestr("meta.json", "{\\"count\\":200}")\n',
        encoding='utf-8',
    )
    converter.chmod(converter.stat().st_mode | 0o111)
    monkeypatch.setattr(preview_cache, 'find_converter', lambda _: str(converter))
    out = tmp_path / 'runs'
    cache = preview_cache.build_sog_cache(source, out, tmp_path, device='cpu')
    assert cache.is_file() and cache.suffix == '.sog'
    assert preview_cache.find_sog_cache(source, out) == cache
    assert source.read_bytes() == before
    _, manifest = preview_cache.cache_paths(source, out)
    record = json.loads(manifest.read_text(encoding='utf-8'))
    assert record['display_only_lossy']
    assert record['source_signature'] == preview_cache.file_signature(source)
    # Second call must not invoke the external converter.
    monkeypatch.setattr(preview_cache, 'find_converter', lambda _: None)
    assert preview_cache.build_sog_cache(source, out, tmp_path) == cache
