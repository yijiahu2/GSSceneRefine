"""Lightweight regression for the new orchestrator; no reconstruction/GPU needed."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from gsrefine.roi_init import full_pipeline as cli
from tools import install_full_pipeline as installer


def args(tmp_path):
    return cli._parser().parse_args([
        '--config', str(tmp_path / 'default.yaml'),
        '--scene', 'biechuanchansi', '--method', 'Mip-Splatting',
        '--sfm-dir', '/sfm/sparse', '--gaussian-ply', '/gs.ply',
        '--output-dir', str(tmp_path / 'results'), '--accept-auto-exterior'])


def successful_stage1(tmp_path):
    roi_run = tmp_path / 'results' / 'biechuanchansi' / 'methods' / 'Mip-Splatting' / 'roi_002'
    (roi_run / 'final').mkdir(parents=True)
    (roi_run / 'metadata').mkdir()
    roi_path = tmp_path / 'results' / 'biechuanchansi' / 'roi' / 'roi_002.json'
    roi_path.parent.mkdir(parents=True)
    roi_path.write_text('{}')
    ply = roi_run / 'final' / 'gaussian_roi.ply'
    ply.write_bytes(b'ply\nDATA')
    (roi_run / 'metadata' / 'manifest.json').write_text(json.dumps({
        'status': 'completed',
        'stage2_inputs': {'roi_json': str(roi_path), 'gaussian_ply': str(ply)}}))
    return {'status': 'completed', 'output_dir': str(roi_run),
            'roi': str(roi_path), 'final_ply': str(ply)}, roi_run


def fake_modules(monkeypatch, stage1_status):
    work = ModuleType('gsrefine.roi_init.workflow')
    work.workflow_options = lambda config: {'enabled': True}
    server = ModuleType('gsrefine.roi_init.server')
    observed = []
    def server_main(argv):
        observed.append(argv)
        return stage1_status
    server.main = server_main
    monkeypatch.setitem(sys.modules, work.__name__, work)
    monkeypatch.setitem(sys.modules, server.__name__, server)
    return observed


def test_integrated_run_selects_exact_completed_roi_and_stage2(monkeypatch, tmp_path):
    arg = args(tmp_path)
    status, roi_run = successful_stage1(tmp_path)
    observed = fake_modules(monkeypatch, status)
    commands = []
    def run_stage2(cmd, check):
        assert check is False
        commands.append(cmd)
        out = Path(cmd[cmd.index('--output-dir') + 1])
        (out / 'gaussian_roi_clean.ply').write_bytes(b'ply\nCLEAN')
        (out / 'stage2_final_report.json').write_text(json.dumps({
            'status': 'COMPLETED_EXPERIMENTAL_CLEAN_ROI_GAUSSIAN',
            'counts': {'final_retained': 7, 'final_removed': 3}}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cli.subprocess, 'run', run_stage2)
    record = cli.run(arg)
    assert record['status'] == 'COMPLETED'
    assert '--method' in observed[0] and 'Mip-Splatting' in observed[0]
    assert commands[0][commands[0].index('--manifest')+1] == str(roi_run/'metadata'/'manifest.json')
    assert '--accept-auto-exterior' in commands[0]
    assert str(roi_run/'cleanup'/'stage2_final_v1') in commands[0]
    assert (roi_run/'metadata'/'full_pipeline_status.json').is_file()


def test_stage1_without_success_never_runs_stage2(monkeypatch, tmp_path):
    arg = args(tmp_path)
    fake_modules(monkeypatch, None)
    monkeypatch.setattr(cli.subprocess, 'run', lambda *args, **kw: pytest.fail('stage2 must not start'))
    with pytest.raises(RuntimeError, match='阶段一未返回'):
        cli.run(arg)


def test_bad_manifest_prevents_stage2(monkeypatch, tmp_path):
    arg = args(tmp_path)
    status, roi_run = successful_stage1(tmp_path)
    p = roi_run/'metadata'/'manifest.json'
    data = json.loads(p.read_text())
    data['stage2_inputs']['gaussian_ply'] = '/wrong/gaussian_roi.ply'
    p.write_text(json.dumps(data))
    fake_modules(monkeypatch, status)
    monkeypatch.setattr(cli.subprocess, 'run', lambda *args, **kw: pytest.fail('stage2 must not start'))
    with pytest.raises(FileNotFoundError):
        cli.run(arg)


def test_second_run_output_does_not_overwrite_existing(monkeypatch, tmp_path):
    _, roi_run = successful_stage1(tmp_path)
    p = roi_run/'cleanup'/'stage2_final_v1'
    p.mkdir(parents=True)
    (p/'previous_result.txt').write_text('DO NOT TOUCH')
    reserved = cli._reserve_stage2_dir(roi_run, 'stage2_final_v1')
    assert reserved.name == 'stage2_final_v2'
    assert (p/'previous_result.txt').read_text() == 'DO NOT TOUCH'


def test_stage2_failure_is_reported_and_never_marked_complete(monkeypatch, tmp_path):
    arg = args(tmp_path)
    status, roi_run = successful_stage1(tmp_path)
    fake_modules(monkeypatch, status)
    monkeypatch.setattr(cli.subprocess, 'run', lambda *args, **kw: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match='退出码=1'):
        cli.run(arg)
    record = json.loads((roi_run/'metadata'/'full_pipeline_status.json').read_text())
    assert record['status'] == 'STAGE2_FAILED'
    assert not (roi_run/'cleanup'/'stage2_final_v1'/'gaussian_roi_clean.ply').exists()


def test_exterior_anchor_required_before_stage1(tmp_path):
    with pytest.raises(SystemExit):
        cli._parser().parse_args(['--output-dir', str(tmp_path)])


def test_safe_directory_name_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        cli._reserve_stage2_dir(tmp_path, '../incorrect')


def test_installer_only_adds_new_route_and_completed_status():
    run_text = "COMMANDS = {\n    'roi-init':'gsrefine.roi_init.server',\n    'roi-auto':'gsrefine.roi_init.workflow',\n}\n"
    expected = installer.change_run(run_text)
    assert expected.count("'roi-init':'gsrefine.roi_init.server'") == 1
    assert expected.count("'full-pipeline':'gsrefine.roi_init.full_pipeline'") == 1
    assert installer.change_run(expected) == expected

    server_text = """def main():
    workflow = 1
    interrupted = False
    if workflow.launched_job() and not interrupted:
        status=monitor_terminal(workflow)
        if status['status'] != 'completed':
            raise SystemExit(1)

if __name__=='__main__':main()
"""
    fixed = installer.change_server(server_text)
    compile(fixed, 'server.py', 'exec')
    assert fixed.count('return completed_status') == 1
    assert installer.change_server(fixed) == fixed


def test_installer_check_apply_idempotent_and_backups(tmp_path, monkeypatch):
    run = tmp_path/'run.py'
    server = tmp_path/'gsrefine'/'roi_init'/'server.py'
    server.parent.mkdir(parents=True)
    run.write_text("COMMANDS = {\n    'roi-init':'gsrefine.roi_init.server',\n    'roi-auto':'gsrefine.roi_init.workflow',\n}\n")
    server.write_text("""def main():
    workflow = object()
    interrupted = False
    if workflow.launched_job() and not interrupted:
        status=monitor_terminal(workflow)
        if status['status'] != 'completed':
            raise SystemExit(1)

if __name__=='__main__':main()
""")
    new = tmp_path/'gsrefine'/'roi_init'/'full_pipeline.py'
    new.write_text('def main(): pass\n')
    (tmp_path/'.git').mkdir()
    monkeypatch.setattr(installer, 'ROOT', tmp_path)
    monkeypatch.setattr(installer, 'RUN', run)
    monkeypatch.setattr(installer, 'SERVER', server)
    monkeypatch.setattr(installer, 'NEW', new)
    before_run, before_server = run.read_bytes(), server.read_bytes()
    installer.apply(check_only=True)
    assert run.read_bytes() == before_run and server.read_bytes() == before_server
    installer.apply(check_only=False)
    after_run, after_server = run.read_bytes(), server.read_bytes()
    assert after_run != before_run and after_server != before_server
    assert "'roi-auto':'gsrefine.roi_init.workflow'" in after_run.decode()
    backups = list((tmp_path/'.git'/'gsrefine_patch_backups').glob('full_pipeline_*'))
    assert len(backups) == 1
    assert (backups[0]/'run.py').read_bytes() == before_run
    assert (backups[0]/'gsrefine'/'roi_init'/'server.py').read_bytes() == before_server
    installer.apply(check_only=False)
    assert run.read_bytes() == after_run and server.read_bytes() == after_server
    assert len(list((tmp_path/'.git'/'gsrefine_patch_backups').glob('full_pipeline_*'))) == 1
