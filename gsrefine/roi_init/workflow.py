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
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from gsrefine.roi_init.settings import PROJECT_ROOT, InitSettings
from gsrefine.roi_init.results import current_latest_for_source, require_orientation, write_atomic

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
    out = Path(opts.get('output_root', 'runs/roi_tests')).expanduser()
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if opts.get('final_method', 'R3') != 'R3':
        raise ValueError('Automatic pipeline final_method must be R3')
    return {'enabled': bool(opts.get('auto_after_confirm', False)),
            'output_root': out.resolve(), 'final_method': 'R3'}


def output_for_roi(config, roi):
    """One stable experiment directory for each numbered confirmed ROI."""
    p = Path(roi).expanduser().resolve(strict=True)
    if not p.name.endswith('.json') or not p.stem:
        raise ValueError('A confirmed ROI must be a JSON artifact')
    root = workflow_options(config)['output_root']
    meta = json.loads(p.read_text(encoding='utf8')).get('annotation_metadata', {})
    cfg = yaml.safe_load(Path(config).read_text(encoding='utf8'))
    scene = cfg.get('scene', {}) or {}
    current = Path(cfg.get('paths', {}).get('gaussian_ply', '')).expanduser().resolve()
    if meta.get('source_ply') == str(current) and meta.get('method') == scene.get('method'):
        return root / p.stem  # Existing ROI005 path is backwards compatible.
    method = scene.get('method', 'unknown')
    st = current.stat()
    # Include file identity to avoid overwriting another model stored later
    # at the same pathname. Keep the original matching-source output path.
    identity = f'{current}|{st.st_size}|{st.st_mtime_ns}'
    source_key = hashlib.sha256(identity.encode('utf8')).hexdigest()[:8]
    return root / f'{p.stem}__{method}__{source_key}'


def update_progress(out, phase, *, detail='', roi=None, **extra):
    """Atomically publish a real processing milestone, never a fabricated percent."""
    if phase not in PHASES:
        raise ValueError(f'Unknown workflow phase: {phase}')
    out = Path(out)
    progress = out / PROGRESS_FILENAME
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
        data = json.loads((Path(out) / PROGRESS_FILENAME).read_text(encoding='utf8'))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _checked_result(out, roi, sha):
    """Only a matching manifest AND a real nonempty PLY authorize completion."""
    data = json.loads((out / 'workflow_result.json').read_text(encoding='utf8'))
    path = Path(data['final_ply']).resolve(strict=True)
    if (data['roi_sha256'] != sha or Path(data['roi']).resolve() != roi or
            not path.is_file() or not path.stat().st_size > 0 or
            path != (out / 'gaussians_final_R3.ply').resolve()):
        raise ValueError('Final model/ROI verification failed')
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


def run_once(config, roi, output_dir=None, *, reuse_scene_roi=False):
    """Synchronous reproducible pipeline; interactive and direct CLI share it."""
    # Calling roi-auto directly with the original scene YAML must prepare the
    # same geometry as interactive roi-init; subprocesses passed the effective
    # config are a no-op (module1.mode=prepared).
    from gsrefine.geometry.prepare import prepare_scene
    config = prepare_scene(config)
    roi = Path(roi).expanduser().resolve(strict=True)
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else output_for_roi(config, roi)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / 'workflow_result.json'
    result_file.unlink(missing_ok=True)
    update_progress(output_dir, 'validating', roi=roi, pid=os.getpid(),
                    detail='检查已确认ROI、Gaussian PLY和COLMAP坐标一致性')
    settings = InitSettings.load(config)
    require_orientation(json.loads(roi.read_text(encoding='utf8')), settings,
                        reuse_scene_geometry=reuse_scene_roi)
    scene_config = yaml.safe_load(config.read_text(encoding='utf8'))
    study_source = Path(scene_config['paths']['gaussian_ply']).expanduser().resolve()
    if study_source != settings.gaussian_ply:
        raise ValueError('Module 1 display Gaussian and Module 3 study Gaussian differ; configure one common source PLY')
    update_progress(output_dir, 'loading', detail='读取原始PLY和Poisson曲面')

    def report_stage(phase, detail=''):
        update_progress(output_dir, phase, detail=detail)

    from gsrefine.roi.study import run as run_study
    report = run_study(config, roi, output_dir, run_ablation=True, final_only=True,
                       export_final=True, final_method='R3', progress_callback=report_stage)
    update_progress(output_dir, 'verifying', detail='检查最终PLY、ROI身份和两阶段Gaussian数量')
    final = report.get('final_gaussian_scene') or {}
    ply = Path(final.get('output_ply', ''))
    if not ply.is_file() or final.get('method') != 'R3':
        raise RuntimeError('Module 4 did not produce a validated final R3 PLY')
    if Path(report['roi']).resolve() != roi or report['input_sha256']['roi'] != _sha(roi):
        raise RuntimeError('Final study ROI does not match the confirmed annotation')
    r3 = next((x for x in report.get('ablation', []) if x.get('method') == 'R3'), None)
    stages = r3.get('two_stage_diagnostics', {}) if r3 else {}
    recovered = stages.get('stage_A', {}).get('restored_core_gaussians')
    safety = stages.get('stage_B', {}).get('safety_added_gaussians')
    if recovered is not None and safety is not None:
        if int(recovered) + int(safety) != int(final['added_gaussians']):
            raise RuntimeError('R3 two-stage counts do not sum to the final added Gaussian count')
    result = {
        'schema': 'GSSceneRefine.workflow.v1', 'status': 'completed',
        'source_scene': settings.scene, 'target_method': settings.method,
        'source_gaussian': str(settings.gaussian_ply),
        'reused_roi_geometry': bool(reuse_scene_roi),
        'completed_utc': _utc(), 'roi': str(roi), 'roi_sha256': _sha(roi),
        'output_dir': str(output_dir), 'final_ply': str(ply),
        'R0_gaussians': int(report['R0_count']),
        'final_gaussians': int(final['final_gaussians']),
        'added_gaussians': int(final['added_gaussians']),
        'core_restored_gaussians': int(recovered) if recovered is not None else None,
        'safety_added_gaussians': int(safety) if safety is not None else None,
        'study_report': str(output_dir / 'study_report.json'),
        'final_report': str(output_dir / 'final_gaussian_report_R3.json'),
    }
    write_atomic(result_file, result)
    update_progress(output_dir, 'completed', detail='最终PLY已导出且通过一致性检查')
    print(f'[roi-auto] Complete: {result["final_ply"]}', flush=True)
    return result


class WorkflowManager:
    """Thread-safe handoff; durable status survives tab/browser/server restarts."""
    def __init__(self, config, settings=None):
        self.config = Path(config).expanduser().resolve(strict=True) if config else None
        self.options = workflow_options(self.config) if self.config else {'enabled': False}
        self.enabled = self.options['enabled']
        self.settings = settings
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
            if not (out / PROGRESS_FILENAME).is_file() and not (out / 'workflow_result.json').is_file():
                return None
            return {'roi': roi, 'roi_sha256': _sha(roi), 'output_dir': out,
                    'log': out / 'workflow.log', 'process': None, 'started_utc': None}
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
                and not (recent['output_dir'] / 'workflow_result.json').is_file())

    def launch(self, roi):
        if not self.enabled:
            raise RuntimeError('Automatic workflow is disabled in scene config')
        roi = Path(roi).expanduser().resolve(strict=True)
        out = output_for_roi(self.config, roi)
        with self._lock:
            if self._job and self._job['process'].poll() is None:
                raise RuntimeError('Another ROI is already being processed')
            out.mkdir(parents=True, exist_ok=True)
            # A new run must never inherit a previous success marker.
            (out / 'workflow_result.json').unlink(missing_ok=True)
            update_progress(out, 'queued', roi=roi, detail='已接受ROI确认，正在创建本地计算进程', pid=None)
            log = out / 'workflow.log'
            command = [sys.executable, str(PROJECT_ROOT / 'run.py'), 'roi-auto',
                       '--config', str(self.config), '--roi', str(roi),
                       '--output-dir', str(out)]
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
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--reuse-scene-roi', action='store_true',
                        help='Explicitly apply a full-scene ROI to another Gaussian model in the identical COLMAP frame')
    parser.add_argument('--scene', help='Scene ID for a new COLMAP reconstruction')
    parser.add_argument('--method', help='3DGS method ID for this Gaussian model')
    parser.add_argument('--gaussian-ply', help='Full Gaussian PLY for the target method')
    parser.add_argument('--colmap-points', help='Original points3D.bin; may be reused from the scene cache')
    parser.add_argument('--colmap-images', help='COLMAP images.bin defining the same camera/world frame')
    parser.add_argument('--sfm-dir', help='COLMAP sparse directory containing points3D.bin and images.bin')
    args = parser.parse_args(argv)
    config = args.config
    if any((args.scene, args.method, args.gaussian_ply, args.colmap_points,
            args.colmap_images, args.sfm_dir)):
        from gsrefine.geometry.prepare import prepare_input_config, expand_colmap_inputs
        points, images = expand_colmap_inputs(args.sfm_dir, colmap_points=args.colmap_points,
                                              colmap_images=args.colmap_images)
        config = prepare_input_config(config, scene=args.scene, method=args.method,
                                      gaussian_ply=args.gaussian_ply, colmap_points=points,
                                      colmap_images=images)
    try:
        run_once(config, args.roi, args.output_dir, reuse_scene_roi=args.reuse_scene_roi)
    except Exception as exc:
        try:
            output_dir = (Path(args.output_dir).expanduser().resolve() if args.output_dir
                          else output_for_roi(config, args.roi))
            update_progress(output_dir, 'failed', detail=str(exc))
        except (OSError, ValueError):
            pass
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == '__main__':
    main()
