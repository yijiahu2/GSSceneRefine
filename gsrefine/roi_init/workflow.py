"""Persisted orchestration status for the existing Modules 1→2→3→4 pipeline.

The scientific R3 algorithm and finalizer are unchanged. A confirmed ROI starts
one ordinary subprocess. The browser reads durable, validated progress and may
reconnect after a refresh without needing to rerun any Gaussian computation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import traceback
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from gsrefine.roi_init.settings import PROJECT_ROOT, InitSettings
from gsrefine.roi_init.results import current_latest_for_source, require_orientation, write_atomic
from gsrefine.roi_init.output_layout import (run_dir, publish, result_path, progress_path,
                                            log_path, FINAL_REL, MANIFEST_REL)

PROGRESS_FILENAME = 'workflow_progress.json'
PHASES = {
    'queued': (0, '已确认ROI，等待启动'),
    'validating': (1, '校验ROI与输入数据'),
    'loading': (2, '加载Gaussian和Poisson数据'),
    'poisson': (3, '模块三：Poisson局部误截断识别'),
    'core': (4, '模块三：局部主体恢复'),
    'safety': (5, '模块三：全边界自适应安全外扩'),
    'exporting': (6, '模块四：导出最终Gaussian PLY'),
    'verifying': (7, '核对最终模型与恢复统计'),
    'completed': (8, '四模块流程已完成'),
    'failed': (-1, '自动处理失败'),
}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for data in iter(lambda: file.read(1048576), b''):
            h.update(data)
    return h.hexdigest()


def workflow_options(config):
    path = Path(config).expanduser().resolve(strict=True)
    data = yaml.safe_load(path.read_text(encoding='utf8'))
    if not isinstance(data, dict):
        raise ValueError('Scene config must be a YAML mapping')
    opts = data.get('workflow', {}) or {}
    if not isinstance(opts, dict):
        raise ValueError('workflow must be a YAML mapping')
    out = Path(opts.get('output_root', path.parent / 'gsrefine_results')).expanduser()
    if not out.is_absolute():
        out = path.parent / out
    if opts.get('final_method', 'R3') != 'R3':
        raise ValueError('Automatic pipeline final_method must be R3')
    return {'enabled': bool(opts.get('auto_after_confirm', False)),
            'output_root': out.resolve(), 'final_method': 'R3'}


def output_for_roi(config, roi, output_root=None):
    """Public run directory: <root>/<scene>/methods/<method>/roi_NNN/."""
    roi = Path(roi).expanduser().resolve(strict=True)
    cfg = yaml.safe_load(Path(config).read_text(encoding='utf8'))
    scene = cfg.get('scene', {}) or {}
    root = Path(output_root).expanduser().resolve() if output_root else workflow_options(config)['output_root']
    return run_dir(root, scene['name'], scene['method'], roi)


def update_progress(out, phase, *, detail='', roi=None, **extra):
    """Atomically publish a real processing milestone, never a fabricated percent."""
    if phase not in PHASES:
        raise ValueError(f'Unknown workflow phase: {phase}')
    out = Path(out)
    progress = progress_path(out)
    progress.parent.mkdir(parents=True, exist_ok=True)
    try:
        old = json.loads(progress.read_text(encoding='utf8'))
    except (OSError, json.JSONDecodeError, ValueError):
        old = {}
    now = _utc()
    if roi is not None:
        roi = Path(roi).resolve(strict=True)
        roi_hash = _sha(roi)
        if old.get('roi_sha256') != roi_hash or old.get('roi') != str(roi):
            old = {}
        old.update(roi=str(roi), roi_sha256=roi_hash)
    idx, label = PHASES[phase]
    old.update(schema='GSSceneRefine.workflow_progress.v1',
               phase=phase, phase_index=idx, phase_total=8,
               phase_label=label, detail=str(detail),
               status='completed' if phase == 'completed' else ('failed' if phase == 'failed' else 'running'),
               started_utc=old.get('started_utc', now), updated_utc=now)
    old.update(extra)
    write_atomic(progress, old)
    # The subprocess logs this to workflow.log. The WSL parent forwards only
    # stage milestones, never the browser's repetitive HTTP polling requests.
    if phase != 'queued':
        title = f'[pipeline-stage] {label}'
        print(f'{title}: {detail}' if detail else title, flush=True)
    return old


def _read_progress(out):
    try:
        data = json.loads(progress_path(out).read_text(encoding='utf8'))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _checked_result(out, roi, sha):
    """A valid manifest, source identity and nonempty model authorize success."""
    out = Path(out)
    data = json.loads(result_path(out).read_text(encoding='utf8'))
    manifest = json.loads((out / MANIFEST_REL).read_text(encoding='utf8'))
    path = Path(data['final_ply']).resolve(strict=True)
    if (data['roi_sha256'] != sha or Path(data['roi']).resolve() != roi or
            not path.is_file() or path.stat().st_size <= 0 or
            path != (out / FINAL_REL).resolve() or
            manifest.get('status') != 'completed' or
            Path(manifest['stage2_inputs']['gaussian_ply']).resolve() != path):
        raise ValueError('Final model/ROI manifest verification failed')
    return {'final_ply': str(path), 'R0_gaussians': data['R0_gaussians'],
            'final_gaussians': data['final_gaussians'],
            'added_gaussians': data['added_gaussians'],
            'core_restored_gaussians': data.get('core_restored_gaussians'),
            'safety_added_gaussians': data.get('safety_added_gaussians'),
            'completed_utc': data['completed_utc']}


def _pid_alive(pid):
    """Restart recovery only: require the actual roi-auto process, not a reused PID."""
    try:
        pid = int(pid)
        if pid < 1:
            return False
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ')
        return b'roi-auto' in command and b'run.py' in command
    except (ValueError, TypeError, OSError):
        return False


def run_once(config, roi, output_dir=None, *, reuse_scene_roi=False,
             keep_diagnostics=False):
    """Four-module sync workflow; --output-dir always means the output ROOT.

    Module-3's validated algorithm writes into a temporary, isolated study
    directory. Only final outputs and requested diagnostics are published.
    """
    from gsrefine.geometry.prepare import prepare_scene
    config = prepare_scene(config)
    roi = Path(roi).expanduser().resolve(strict=True)
    out = output_for_roi(config, roi, output_root=output_dir)
    out.mkdir(parents=True, exist_ok=True)
    completed = out / FINAL_REL
    if completed.exists() or result_path(out).is_file():
        raise FileExistsError(f'Existing ROI result will not be overwritten: {out}. '
                              'Choose another ROI or remove this specific run after backup.')
    update_progress(out, 'validating', roi=roi, pid=os.getpid(),
                    detail='检查已确认ROI、Gaussian PLY和COLMAP坐标一致性')
    settings = InitSettings.load(config)
    require_orientation(json.loads(roi.read_text(encoding='utf8')), settings,
                        reuse_scene_geometry=reuse_scene_roi)
    scene_config = yaml.safe_load(config.read_text(encoding='utf8'))
    study_source = Path(scene_config['paths']['gaussian_ply']).expanduser().resolve()
    if study_source != settings.gaussian_ply:
        raise ValueError('Module 2 display Gaussian and Module 3 study Gaussian differ')
    update_progress(out, 'loading', detail='读取原始Gaussian和可信Poisson曲面')

    def report_stage(phase, detail=''):
        update_progress(out, phase, detail=detail)

    from gsrefine.roi.study import run as run_study
    # TemporaryDirectory cleans even on Python exceptions. For SIGKILL,
    # stale work directories are removable by the next managed invocation.
    import shutil
    for stale in out.glob('.work-*'):
        if stale.is_dir():
            shutil.rmtree(stale)
    with tempfile.TemporaryDirectory(prefix='.work-', dir=out) as scratch:
        work = Path(scratch)
        report = run_study(config, roi, work, run_ablation=True, final_only=True,
                           export_final=True, final_method='R3', progress_callback=report_stage,
                           save_diagnostics=keep_diagnostics, save_indices=keep_diagnostics)
        update_progress(out, 'verifying', detail='检查PLY、ROI身份和恢复数量')
        final = report.get('final_gaussian_scene') or {}
        ply = Path(final.get('output_ply', ''))
        if not ply.is_file() or final.get('method') != 'R3':
            raise RuntimeError('Module 4 did not produce a validated Gaussian ROI PLY')
        if Path(report['roi']).resolve() != roi or report['input_sha256']['roi'] != _sha(roi):
            raise RuntimeError('Final study ROI does not match confirmed annotation')
        r3 = next((x for x in report.get('ablation', []) if x.get('method') == 'R3'), None)
        stages = r3.get('two_stage_diagnostics', {}) if r3 else {}
        recovered = stages.get('stage_A', {}).get('restored_core_gaussians')
        safety = stages.get('stage_B', {}).get('safety_added_gaussians')
        if recovered is not None and safety is not None:
            if int(recovered) + int(safety) != int(final['added_gaussians']):
                raise RuntimeError('R3 recovery counts do not sum to final added count')
        model, manifest, summary = publish(out, work, roi, config, report,
                                           keep_diagnostics=keep_diagnostics)
    result = {
        'schema': 'GSSceneRefine.workflow.v2', 'status': 'completed',
        'source_scene': settings.scene, 'target_method': settings.method,
        'source_gaussian': str(settings.gaussian_ply),
        'reused_roi_geometry': bool(reuse_scene_roi),
        'completed_utc': _utc(), 'roi': str(roi), 'roi_sha256': _sha(roi),
        'output_dir': str(out), 'final_ply': str(model),
        'R0_gaussians': int(report['R0_count']),
        'final_gaussians': int(final['final_gaussians']),
        'added_gaussians': int(final['added_gaussians']),
        'core_restored_gaussians': int(recovered) if recovered is not None else None,
        'safety_added_gaussians': int(safety) if safety is not None else None,
        'processing_report': str(out/'metadata/processing_report.json'),
        'manifest': str(out/MANIFEST_REL),
    }
    write_atomic(result_path(out), result)
    update_progress(out, 'completed', detail='Gaussian ROI已导出且通过一致性检查')
    print(f'[roi-auto] Complete: {result["final_ply"]}', flush=True)
    return result


class WorkflowManager:
    """Thread-safe handoff; durable status survives tab/browser/server restarts."""
    def __init__(self, config, settings=None, *, keep_diagnostics=False):
        self.config = Path(config).expanduser().resolve(strict=True) if config else None
        self.options = workflow_options(self.config) if self.config else {'enabled': False}
        self.enabled = self.options['enabled']
        self.settings = settings
        self.keep_diagnostics = bool(keep_diagnostics)
        self.confirm_lock = threading.Lock()
        self._lock = threading.Lock()
        self._job = None
        self._last_announced = None

    def _recent(self):
        if not self.config:
            return None
        try:
            settings = self.settings if self.settings is not None else InitSettings.load(self.config)
            roi = current_latest_for_source(settings)
            if roi is None:
                return None
            out = output_for_roi(self.config, roi)
            if not progress_path(out).is_file() and not result_path(out).is_file():
                return None
            return {'roi': roi, 'roi_sha256': _sha(roi), 'output_dir': out,
                    'log': log_path(out), 'process': None, 'started_utc': None}
        except (ValueError, OSError, KeyError, RuntimeError):
            return None

    def is_running(self):
        with self._lock:
            if self._job and self._job['process'].poll() is None:
                return True
        recent = self._recent()
        if recent is None:
            return False
        progress = _read_progress(recent['output_dir'])
        if progress.get('roi_sha256') != recent['roi_sha256']:
            return False
        return (progress.get('status') == 'running' and _pid_alive(progress.get('pid'))
                and not result_path(recent['output_dir']).is_file())

    def launch(self, roi):
        if not self.enabled:
            raise RuntimeError('Automatic workflow is disabled in scene config')
        roi = Path(roi).expanduser().resolve(strict=True)
        out = output_for_roi(self.config, roi)
        with self._lock:
            if self._job and self._job['process'].poll() is None:
                raise RuntimeError('Another ROI is already being processed')
            out.mkdir(parents=True, exist_ok=True)
            # Never destroy a completed model or success marker.
            if (out / FINAL_REL).exists() or result_path(out).is_file():
                raise FileExistsError(f'ROI result already exists: {out}')
            update_progress(out, 'queued', roi=roi, detail='已接受ROI确认，正在创建本地计算进程', pid=None)
            log = log_path(out)
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(PROJECT_ROOT / 'run.py'), 'roi-auto',
                       '--config', str(self.config), '--roi', str(roi),
                       '--output-dir', str(self.options['output_root'])]
            if self.keep_diagnostics:
                command.append('--keep-diagnostics')
            with log.open('w', encoding='utf8') as stream:
                process = subprocess.Popen(command, cwd=PROJECT_ROOT,
                                           stdin=subprocess.DEVNULL,
                                           stdout=stream, stderr=subprocess.STDOUT)
            self._job = {'process': process, 'roi': roi, 'roi_sha256': _sha(roi),
                         'output_dir': out, 'log': log, 'started_utc': _utc()}
            print('[roi-init] Automatic Modules 3→4 started:', roi, flush=True)
        return self.status()

    def launched_job(self):
        with self._lock:
            return self._job is not None

    def status(self, *, announce=True):
        with self._lock:
            job = dict(self._job) if self._job is not None else None
        if job is None:
            job = self._recent()
        if job is None:
            return {'enabled': self.enabled, 'status': 'idle'}
        out, roi, sha = job['output_dir'], job['roi'], job['roi_sha256']
        rc = job['process'].poll() if job['process'] is not None else None
        state = _read_progress(out)
        if state.get('roi_sha256') != sha:
            state = {}
        stage = state.get('phase', 'queued')
        result = {'enabled': self.enabled, 'status': 'running',
                  'roi': str(roi), 'roi_sha256': sha,
                  'output_dir': str(out), 'log': str(job['log']),
                  'started_utc': state.get('started_utc') or job.get('started_utc'),
                  'return_code': rc, 'phase': stage,
                  'phase_index': state.get('phase_index', 0), 'phase_total': 8,
                  'phase_label': state.get('phase_label', PHASES.get(stage, PHASES['queued'])[1]),
                  'phase_detail': state.get('detail', ''),
                  'progress_updated_utc': state.get('updated_utc')}
        # The authoritative success indicator is the final validated manifest.
        # This also recovers a completed ROI when the browser/server is reopened.
        try:
            done = _checked_result(out, roi, sha)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            done = None
        if done is not None:
            result.update(done)
            result.update(status='completed', phase='completed', phase_index=8,
                          phase_label=PHASES['completed'][1], phase_detail='最终模型已验证')
        elif rc is not None or (job['process'] is None and not _pid_alive(state.get('pid'))):
            result.update(status='failed', phase='failed', phase_index=-1,
                          phase_label=PHASES['failed'][1],
                          error=(f'Worker exited with code {rc}; inspect workflow.log' if rc is not None
                                 else 'Interrupted or missing worker; inspect workflow.log'))
        if announce and result['status'] != 'running':
            key = (sha, result['status'], result.get('final_ply'))
            with self._lock:
                if key != self._last_announced:
                    self._last_announced = key
                    print(f'[roi-init] Automatic workflow {result["status"]}: '
                          f'{result.get("final_ply", result.get("error", out))}', flush=True)
        return result


def monitor_terminal(workflow, *, poll_interval=0.4, print_fn=print):
    """Forward subprocess milestones and its verified terminal result to WSL.

    Only the owner process calls this after the browser server has stopped.
    The worker continues independently and writes its real stdout to
    workflow.log. The parent follows stage lines without repeating huge logs.
    """
    with workflow._lock:
        job = dict(workflow._job) if workflow._job else None
    if job is None:
        raise RuntimeError('No confirmed ROI was launched')
    log = job['log']
    position = 0
    partial = ''
    status = None
    while True:
        if log.is_file():
            # Reading the tail is safe even while the worker is writing it.
            with log.open('r', encoding='utf8', errors='replace') as fp:
                fp.seek(position)
                text = fp.read()
                position = fp.tell()
            data = partial + text
            lines = data.split('\n')
            partial = lines.pop()
            for line in lines:
                if line.startswith(('[pipeline-stage]', '[roi-auto]')):
                    print_fn(line, flush=True)
        status = workflow.status(announce=False)
        if status['status'] != 'running':
            # After process exit, Popen.poll() makes stdout fully available.
            if partial and partial.startswith(('[pipeline-stage]', '[roi-auto]')):
                print_fn(partial, flush=True)
            break
        time.sleep(poll_interval)
    if status['status'] == 'completed':
        print_fn('[pipeline] [4/4] 模块四已完成：最终Gaussian ROI通过验证', flush=True)
        print_fn(f'[pipeline] ROI: {status["roi"]}', flush=True)
        print_fn(f'[pipeline] 最终模型: {status["final_ply"]}', flush=True)
        print_fn(f'[pipeline] 输出目录: {status["output_dir"]}', flush=True)
        print_fn(f'[pipeline] Gaussian数量: R0={status["R0_gaussians"]:,}, '
                 f'最终={status["final_gaussians"]:,}, 新增={status["added_gaussians"]:,}',
                 flush=True)
        for k, label in (('core_restored_gaussians','主体恢复'),
                         ('safety_added_gaussians','安全外扩')):
            if status.get(k) is not None:
                print_fn(f'[pipeline] {label}: {status[k]:,}',flush=True)
    else:
        print_fn(f'[pipeline] 处理失败：{status.get("error", "未知错误")}',flush=True)
        print_fn(f'[pipeline] 任务日志: {log}',flush=True)
        if log.is_file():
            tail=log.read_text(encoding='utf8', errors='replace').splitlines()[-15:]
            if tail:
                print_fn('[pipeline] 失败日志末尾：',flush=True)
                for line in tail:
                    print_fn(f'  {line}',flush=True)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description='Modules 3+4 for an already confirmed Module-2 ROI')
    parser.add_argument('--config', required=True)
    parser.add_argument('--roi', required=True, help='Confirmed ROI JSON (never an unsaved draft)')
    parser.add_argument('--output-dir', required=True, help='Root containing all scene-level and method-level results')
    parser.add_argument('--keep-diagnostics', action='store_true', help='Keep detailed research masks, PLYs and tables')
    parser.add_argument('--reuse-scene-roi', action='store_true',
                        help='Explicitly apply a full-scene ROI to another Gaussian model in the identical COLMAP frame')
    parser.add_argument('--scene', help='Scene ID for a new COLMAP reconstruction')
    parser.add_argument('--method', help='3DGS method ID for this Gaussian model')
    parser.add_argument('--gaussian-ply', help='Full Gaussian PLY for the target method')
    parser.add_argument('--colmap-points', help='Original points3D.bin; may be reused from the scene cache')
    parser.add_argument('--colmap-images', help='COLMAP images.bin defining the same camera/world frame')
    parser.add_argument('--sfm-dir', help='COLMAP sparse directory containing points3D.bin and images.bin')
    args = parser.parse_args(argv)
    config = Path(args.config).expanduser().resolve(strict=True)
    root = Path(args.output_dir).expanduser().resolve()
    overrides = any((args.scene, args.method, args.gaussian_ply,
                     args.colmap_points, args.colmap_images, args.sfm_dir))
    # A prepared config produced by roi-init already points exclusively into
    # this root. Reusing it prevents an unnecessary Module-1 rerun in the worker.
    config_data = yaml.safe_load(config.read_text(encoding='utf8')) or {}
    stored_root = (config_data.get('workflow', {}) or {}).get('output_root')
    try:
        managed_config = (
            config_data.get('module1', {}).get('mode') == 'prepared'
            and config.is_relative_to(root)
            and bool(stored_root)
            and Path(stored_root).expanduser().resolve() == root
            and Path(config_data['roi_init']['output_dir']).expanduser().resolve().is_relative_to(root)
            and Path(config_data['module1_runtime']['clean_sfm']).expanduser().resolve().is_relative_to(root)
            and Path(config_data['module1_runtime']['poisson_raw']).expanduser().resolve().is_relative_to(root)
        )
    except (TypeError, ValueError, KeyError, AttributeError):
        managed_config = False
    if overrides or not managed_config:
        # Even --output-dir alone must redirect geometry/orientation/ROI. An
        # old configs/*.yaml may still refer to in-repository runs/ paths.
        from gsrefine.geometry.prepare import prepare_input_config, expand_colmap_inputs
        points, images = expand_colmap_inputs(args.sfm_dir, colmap_points=args.colmap_points,
                                              colmap_images=args.colmap_images)
        if not overrides and not (config_data.get('module1', {}) or {}).get('colmap_points') and not (
                config_data.get('paths', {}) or {}).get('colmap_points'):
            from gsrefine.geometry.prepare import relocate_existing_reference_config
            config = relocate_existing_reference_config(config, root)
        else:
            config = prepare_input_config(config, scene=args.scene, method=args.method,
                                          gaussian_ply=args.gaussian_ply, colmap_points=points,
                                          colmap_images=images, final_output_root=root)
    try:
        run_once(config, args.roi, args.output_dir, reuse_scene_roi=args.reuse_scene_roi,
                 keep_diagnostics=args.keep_diagnostics)
    except Exception as exc:
        try:
            output_dir = output_for_roi(config, args.roi, args.output_dir)
            update_progress(output_dir, 'failed', detail=str(exc))
        except (OSError, ValueError):
            pass
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == '__main__':
    main()
