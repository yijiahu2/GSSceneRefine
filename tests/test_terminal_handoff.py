"""Module 2 browser handoff must leave the scientific subprocess intact."""
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from gsrefine.roi_init.server import build_meta, make_handler
from gsrefine.roi_init.settings import InitSettings
from gsrefine.roi_init.workflow import WorkflowManager, monitor_terminal
from tests.test_roi_init import prepare, annotate


def config_for_test(tmp_path):
    path = prepare(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg['workflow'] = {'auto_after_confirm': True, 'final_method': 'R3',
                       'output_root': str(tmp_path / 'test_results')}
    path.write_text(yaml.safe_dump(cfg), encoding='utf8')
    return path


class Worker:
    def __init__(self):
        self.alive = True

    def poll(self):
        return None if self.alive else 0


def test_successful_roi_confirm_closes_browser_but_not_worker(tmp_path, monkeypatch):
    config = config_for_test(tmp_path)
    settings = InitSettings.load(config)
    worker = Worker()
    started = []

    def fake_popen(args, **kwargs):
        started.append(args)
        return worker

    import gsrefine.roi_init.workflow as workflow_module
    monkeypatch.setattr(workflow_module.subprocess, 'Popen', fake_popen)
    manager = WorkflowManager(config, settings=settings)
    handler = make_handler(settings, Path(__file__).resolve().parents[1] / 'ui',
                           build_meta(settings), workflow=manager, handoff_shutdown=True)
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        req = urllib.request.Request(
            f'http://127.0.0.1:{server.server_port}/api/confirm-roi',
            data=json.dumps(annotate(settings)).encode('utf8'),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=5) as response:
            confirmed = json.load(response)
        assert confirmed['verified_on_disk']
        assert confirmed['workflow']['status'] == 'running'
        thread.join(timeout=4)
        assert not thread.is_alive(), 'Web server should shut down after confirmed handoff'
        assert len(started) == 1 and 'roi-auto' in started[0]
        assert worker.poll() is None, 'Closing the UI must not terminate the worker'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=4)


def test_failed_confirmation_does_not_close_browser(tmp_path):
    config = config_for_test(tmp_path)
    settings = InitSettings.load(config)
    manager = WorkflowManager(config, settings=settings)
    handler = make_handler(settings, Path(__file__).resolve().parents[1] / 'ui',
                           build_meta(settings), workflow=manager, handoff_shutdown=True)
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f'http://127.0.0.1:{server.server_port}/api/meta', timeout=5) as response:
            assert json.load(response)['project'] == 'GSSceneRefine'
        assert thread.is_alive()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=4)


def test_terminal_monitor_prints_stages_and_validated_final_path(tmp_path):
    logfile = tmp_path / 'workflow.log'
    logfile.write_text(
        '[pipeline-stage] 模块三：Poisson局部误截断识别: 检查背景\n'
        '[pipeline-stage] 模块三：局部主体恢复: 核心恢复\n'
        '[pipeline-stage] 模块三：全边界自适应安全外扩: KNN截尾均值\n'
        '[pipeline-stage] 模块四：导出最终Gaussian PLY: 输出模型\n'
        '[roi-auto] Complete: /test/model.ply\n', encoding='utf8')

    class Manager:
        def __init__(self):
            self._lock = threading.Lock()
            self._job = {'log': logfile}
        def status(self, *, announce=True):
            assert announce is False
            return dict(status='completed', roi='/test/roi.json',
                        output_dir=str(tmp_path), final_ply='/test/model.ply',
                        R0_gaussians=8, final_gaussians=13, added_gaussians=5,
                        core_restored_gaussians=4, safety_added_gaussians=1)

    printed = []
    result = monitor_terminal(Manager(), print_fn=lambda *a, **kw: printed.append(str(a[0])),
                              poll_interval=0.001)
    assert result['status'] == 'completed'
    output = '\n'.join(printed)
    assert 'Poisson局部误截断识别' in output
    assert '安全外扩' in output
    assert '模块四' in output
    assert '/test/model.ply' in output
    assert '13' in output
    assert '主体恢复: 4' in output
