"""Contract regression for end-to-end Module 1→2→3→4 orchestration."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gsrefine.roi_init.settings import InitSettings
from gsrefine.roi_init.results import current_latest
from gsrefine.roi_init.server import build_meta, make_handler
from gsrefine.roi_init.workflow import WorkflowManager, output_for_roi, run_once
from tests.test_roi_init import prepare, annotate


def enabled_config(tmp_path):
    path = prepare(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg['workflow'] = {'auto_after_confirm': True, 'final_method': 'R3',
                       'output_root': str(tmp_path / 'roi_tests')}
    path.write_text(yaml.safe_dump(cfg), encoding='utf8')
    return path


class FakeProcess:
    def __init__(self):
        self.code = None

    def poll(self):
        return self.code


def test_confirm_starts_existing_module3_and4_command(tmp_path, monkeypatch):
    cfg = enabled_config(tmp_path)
    settings = InitSettings.load(cfg)
    process = FakeProcess()
    command = []

    def fake_popen(argv, **kwargs):
        command.extend(argv)
        assert kwargs['cwd'] == Path(__file__).resolve().parents[1]
        assert argv[2] == 'roi-auto' and '--roi' in argv
        return process

    import gsrefine.roi_init.workflow as wf
    monkeypatch.setattr(wf.subprocess, 'Popen', fake_popen)
    handler = make_handler(settings, Path(__file__).resolve().parents[1] / 'ui',
                           build_meta(settings))
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    base = f'http://127.0.0.1:{server.server_port}'

    def get(endpoint):
        with urllib.request.urlopen(base + endpoint, timeout=5) as r:
            return json.load(r)

    def post(roi):
        request = urllib.request.Request(
            base + '/api/confirm-roi', data=json.dumps(roi).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=5) as r:
            return json.load(r)

    try:
        assert get('/api/meta')['workflow']['auto_after_confirm'] is True
        first = post(annotate(settings))
        assert first['ok'] and first['verified_on_disk']
        assert first['workflow']['status'] == 'running'
        assert first['workflow']['roi'] == first['saved_path']
        assert Path(first['workflow']['log']).exists()
        assert command[2] == 'roi-auto'
        assert command[command.index('--roi') + 1] == first['saved_path']
        assert get('/api/workflow-status')['status'] == 'running'
        # A second confirmation cannot save a different ROI while the job runs.
        with pytest.raises(urllib.error.HTTPError) as error:
            post(annotate(settings))
        assert error.value.code == 409
        assert current_latest(settings) == Path(first['saved_path'])
        process.code = 0
        # Exit code 0 alone is NOT enough to report completion.
        assert get('/api/workflow-status')['status'] == 'failed'
        out = Path(first['workflow']['output_dir'])
        model = out / 'final/gaussian_roi.ply'
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b'valid-mock-ply')
        from gsrefine.roi_init.workflow import _sha
        (out / 'metadata').mkdir(parents=True,exist_ok=True)
        (out / 'metadata/manifest.json').write_text(json.dumps({
            'status': 'completed', 'stage2_inputs': {'gaussian_ply': str(model)}}))
        (out / 'metadata/workflow_result.json').write_text(json.dumps({
            'final_ply': str(model), 'roi': first['saved_path'],
            'roi_sha256': _sha(first['saved_path']), 'R0_gaussians': 5,
            'final_gaussians': 7, 'added_gaussians': 2,
            'completed_utc': '2026-11-06T00:00:00+00:00'}))
        done = get('/api/workflow-status')
        assert done['status'] == 'completed'
        assert done['final_ply'] == str(model)
        assert done['added_gaussians'] == 2
    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=5)


def test_manual_pipeline_dispatch_and_immutable_source(tmp_path, monkeypatch):
    cfg = enabled_config(tmp_path)
    settings = InitSettings.load(cfg)
    roi = Path(__import__('gsrefine.roi_init.results', fromlist=['save_confirmed_roi'])
               .save_confirmed_roi(annotate(settings), settings)['saved_path'])
    source_sha = __import__('gsrefine.roi_init.workflow', fromlist=['_sha'])._sha(settings.gaussian_ply)
    calls = []
    def fake_study(config, roi_file, output_dir, **options):
        calls.append(options)
        dest = Path(output_dir) / 'gaussians_final_R3.ply'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b'ply-mock')
        from gsrefine.roi_init.workflow import _sha
        return {'roi': str(Path(roi_file).resolve()),
                'input_sha256': {'roi': _sha(roi_file)},
                'R0_count': 10,
                'final_gaussian_scene': {'method': 'R3', 'output_ply': str(dest),
                                         'final_gaussians': 15, 'added_gaussians': 5}}
    import gsrefine.roi.study as study
    monkeypatch.setattr(study, 'run', fake_study)
    result = run_once(cfg, roi)
    assert len(calls) == 1
    assert {key:value for key,value in calls[0].items() if key != 'progress_callback'} == {
        'run_ablation': True, 'final_only': True, 'export_final': True,
        'final_method': 'R3', 'save_diagnostics': False, 'save_indices': False}
    assert callable(calls[0]['progress_callback'])
    assert result['final_gaussians'] == 15
    assert result['final_ply'].endswith('final/gaussian_roi.ply')
    assert not list(Path(result['output_dir']).glob('.work-*'))
    assert Path(result['manifest']).is_file()
    assert _sha_same(settings.gaussian_ply, source_sha)
    assert output_for_roi(cfg, roi).name == 'roi_001'


def _sha_same(path, value):
    from gsrefine.roi_init.workflow import _sha
    return _sha(path) == value


def test_pipeline_rejects_mismatched_input_source(tmp_path):
    cfg = enabled_config(tmp_path)
    settings = InitSettings.load(cfg)
    from gsrefine.roi_init.results import save_confirmed_roi
    roi = save_confirmed_roi(annotate(settings), settings)['saved_path']
    data = yaml.safe_load(cfg.read_text())
    other = tmp_path / 'other.ply'
    other.write_bytes(settings.gaussian_ply.read_bytes())
    data['roi_init']['gaussian_ply'] = str(other)
    cfg.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='source PLY|display Gaussian'):
        run_once(cfg, roi)


def test_real_http_confirm_to_final_R3_ply(tmp_path):
    """Unmocked synthetic run: browser HTTP → Module 2 → Module 3 → Module 4."""
    import time
    import numpy as np
    from gsrefine.roi.core import GaussianPLY

    config = enabled_config(tmp_path)
    settings = InitSettings.load(config)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(
        settings, Path(__file__).resolve().parents[1] / 'ui', build_meta(settings)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        request = urllib.request.Request(
            base + '/api/confirm-roi', data=json.dumps(annotate(settings)).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=10) as resp:
            submitted = json.load(resp)
        assert submitted['verified_on_disk'] is True
        until = time.monotonic() + 40
        while time.monotonic() < until:
            with urllib.request.urlopen(base + '/api/workflow-status', timeout=5) as response:
                status = json.load(response)
            if status['status'] != 'running':
                break
            time.sleep(.2)
        assert status['status'] == 'completed', status
        final = GaussianPLY(status['final_ply'])
        assert final.vertex_count == status['final_gaussians']
        out = Path(status['output_dir'])
        assert status['added_gaussians'] >= 0
        assert status['core_restored_gaussians'] + status['safety_added_gaussians'] == status['added_gaussians']
        assert (out / 'metadata/manifest.json').is_file()
        assert (out / 'metadata/processing_report.json').is_file()
        assert (out / 'logs/processing.log').is_file()
        assert not (out / 'diagnostics').exists()
        assert not list(out.glob('.work-*'))
        assert not (out / 'core_mask_R3.npy').exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
