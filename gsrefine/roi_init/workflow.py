"""Module 1-4 orchestration without changing independently tested algorithms.

Browser confirmation saves a signed/verified ROI (Module 2); a single local
subprocess runs existing Poisson/R3 study (Module 3), which itself uses the
unchanged Module 4 PLY finalizer. Only one automatic job runs at a time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

from gsrefine.roi_init.settings import PROJECT_ROOT, InitSettings
from gsrefine.roi_init.results import require_orientation, write_atomic


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
    return workflow_options(config)['output_root'] / p.stem


def run_once(config, roi, output_dir=None):
    """Synchronous reproducible pipeline, usable without the browser."""
    config = Path(config).expanduser().resolve(strict=True)
    roi = Path(roi).expanduser().resolve(strict=True)
    settings = InitSettings.load(config)
    require_orientation(json.loads(roi.read_text(encoding='utf8')), settings)
    scene_config = yaml.safe_load(config.read_text(encoding='utf8'))
    study_source = Path(scene_config['paths']['gaussian_ply']).expanduser().resolve()
    if study_source != settings.gaussian_ply:
        raise ValueError('Module 1 display Gaussian and Module 3 study Gaussian differ; configure one common source PLY')
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else output_for_roi(config, roi)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / 'workflow_result.json'
    result_file.unlink(missing_ok=True)
    # The successful original R3 geometry and safety math are left untouched.
    # final_only skips the R1/R2 ablations in the production workflow.
    from gsrefine.roi.study import run as run_study
    report = run_study(config, roi, output_dir, run_ablation=True, final_only=True,
                       export_final=True, final_method='R3')
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
    print(f'[roi-auto] Complete: {result["final_ply"]}', flush=True)
    return result


class WorkflowManager:
    """Thread-safe browser handoff; no background thread performing scientific work.

    The actual heavy computation is a separate, explicit local Python process.
    Logs and outputs remain in the SAME project tree, separated by ROI ID.
    """
    def __init__(self, config):
        self.config = Path(config).expanduser().resolve(strict=True) if config else None
        self.options = workflow_options(self.config) if self.config else {'enabled':False}
        self.enabled = self.options['enabled']
        self.confirm_lock = threading.Lock()
        self._lock = threading.Lock()
        self._job = None

    def is_running(self):
        with self._lock:
            return self._job is not None and self._job['process'].poll() is None

    def launch(self, roi):
        if not self.enabled:
            raise RuntimeError('Automatic workflow is disabled in scene config')
        roi = Path(roi).expanduser().resolve(strict=True)
        out = output_for_roi(self.config, roi)
        with self._lock:
            if self._job and self._job['process'].poll() is None:
                raise RuntimeError('Another ROI is already being processed')
            out.mkdir(parents=True, exist_ok=True)
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

    def status(self):
        with self._lock:
            if self._job is None:
                return {'enabled': self.enabled, 'status': 'idle'}
            job = dict(self._job)
        rc = job['process'].poll()
        result = {'enabled': self.enabled,
                  'status': 'running' if rc is None else ('completed' if rc == 0 else 'failed'),
                  'roi': str(job['roi']), 'roi_sha256': job['roi_sha256'],
                  'output_dir': str(job['output_dir']), 'log': str(job['log']),
                  'started_utc': job['started_utc'], 'return_code': rc}
        if rc == 0:
            manifest = job['output_dir'] / 'workflow_result.json'
            try:
                data = json.loads(manifest.read_text(encoding='utf8'))
                path = Path(data['final_ply'])
                if (not path.is_file() or not path.stat().st_size > 0 or
                    data['roi_sha256'] != job['roi_sha256'] or
                    Path(data['roi']).resolve() != job['roi']):
                    raise ValueError('Final model/ROI verification failed')
                result.update({'final_ply': str(path),
                               'R0_gaussians': data['R0_gaussians'],
                               'final_gaussians': data['final_gaussians'],
                               'added_gaussians': data['added_gaussians'],
                               'core_restored_gaussians': data.get('core_restored_gaussians'),
                               'safety_added_gaussians': data.get('safety_added_gaussians'),
                               'completed_utc': data['completed_utc']})
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                result.update(status='failed', error=f'Final output validation failed: {e}')
        elif rc is not None:
            result['error'] = f'Automatic processing stopped with exit code {rc}; inspect workflow.log'
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Modules 3+4 for an already confirmed Module-2 ROI')
    parser.add_argument('--config', required=True)
    parser.add_argument('--roi', required=True, help='Confirmed ROI JSON (never an unsaved draft)')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args(argv)
    try:
        run_once(args.config, args.roi, args.output_dir)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == '__main__':
    main()
