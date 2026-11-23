"""Canonical Stage-1 -> Stage-2 orchestration. No reconstruction algorithms here.

Existing command stays unchanged:
  python run.py full-pipeline --config ... --scene ... --method ... --sfm-dir ... \
      --gaussian-ply ... --output-dir ... --accept-auto-exterior

Resume after a completed Stage 1 without opening the browser:
  python run.py full-pipeline --resume-manifest /path/roi_NNN/metadata/manifest.json \
      --accept-auto-exterior
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

SCHEMA = 'GSSceneRefine.full_pipeline.v2'


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _read_json(path):
    result = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise ValueError('Expected JSON object in ' + str(path))
    return result


def _atomic_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            json.dump(record, out, ensure_ascii=False, indent=2, allow_nan=False)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _parser():
    p = argparse.ArgumentParser(description='GSSceneRefine: one CLI, two unchanged scientific stages, manifest-based resume')
    s1 = p.add_argument_group('Stage-1 inputs (ignored in --resume-manifest mode)')
    s1.add_argument('--config', default='configs/default.yaml')
    s1.add_argument('--scene')
    s1.add_argument('--method')
    s1.add_argument('--sfm-dir')
    s1.add_argument('--gaussian-ply')
    s1.add_argument('--colmap-points')
    s1.add_argument('--colmap-images')
    s1.add_argument('--output-dir', help='Stage-1 results ROOT; optional for --resume-manifest')
    s1.add_argument('--keep-diagnostics', action='store_true')
    s1.add_argument('--no-auto-windows', action='store_true')
    p.add_argument('--resume-manifest', metavar='MANIFEST_JSON',
                   help='Only run Stage 2 on this exact verified completed Stage-1 ROI; never redraw ROI')
    stage2 = p.add_argument_group('Stage-2 options (unchanged semantics)')
    anchor = stage2.add_mutually_exclusive_group(required=True)
    anchor.add_argument('--accept-auto-exterior', action='store_true')
    anchor.add_argument('--exterior-seed', nargs=3, type=float, metavar=('X', 'Y', 'Z'))
    from gsrefine.cleanup.adaptive_voxel import CLASSIFIERS
    stage2.add_argument('--classifier', choices=(*CLASSIFIERS, 'local-region', 'original-region'), default='original-region',
                        help='Candidate side query after the Stage-2 distance threshold')
    stage2.add_argument('--compare-stage2', help='Compare A without auxiliary protection to an existing Stage-2 result directory')
    stage2.add_argument('--partition-report', help='Existing original-wall A/B report for this ROI')
    stage2.add_argument('--stage2-name', default='stage2_final_v1')
    stage2.add_argument('--no-aux-protection', action='store_true')
    stage2.add_argument('--audit-aux-ablation', action='store_true')
    stage2.add_argument('--no-aux-preview', action='store_true')
    stage2.add_argument('--keep-classification-previews', action='store_true')
    stage2.add_argument('--reclassify', action='store_true')
    stage2.add_argument('--chunk-size', type=int, default=30000)
    stage2.add_argument('--nearest-k', type=int, default=8)
    stage2.add_argument('--min-path-agree', type=int, default=2)
    stage2.add_argument('--ray-batch', type=int, default=8000)
    return p


def _stage1_argv(args):
    out = []
    for name in ('config', 'scene', 'method', 'sfm_dir', 'gaussian_ply',
                 'colmap_points', 'colmap_images', 'output_dir'):
        val = getattr(args, name)
        if val is not None:
            out += ['--' + name.replace('_', '-'), str(val)]
    if args.keep_diagnostics:
        out.append('--keep-diagnostics')
    if args.no_auto_windows:
        out.append('--no-auto-windows')
    return out


def _validate_manifest(manifest_path, *, expected_run=None, expected_roi=None, expected_ply=None,
                       output_root=None):
    """Single authoritative Stage-1 manifest loader, also supports older snapshot-based runs.

    No latest/filename discovery. Identity is based on Stage-1 verified hashes, not
    on the current ROI named 'latest'.  Never invent or repair a missing source.
    """
    from gsrefine.cleanup.sdf_common import _load_stage1_manifest
    mf, data, inputs, run_dir = _load_stage1_manifest(manifest_path)
    mf = Path(mf).resolve(strict=True)
    run_dir = Path(run_dir).resolve(strict=True)
    if mf != run_dir / 'metadata' / 'manifest.json':
        raise ValueError('Manifest must be the canonical Stage-1 metadata/manifest.json')
    if expected_run is not None and run_dir != Path(expected_run).resolve(strict=True):
        raise ValueError('Stage-1 manifest belongs to another ROI run')
    if expected_ply is not None and inputs['gaussian_ply'] != Path(expected_ply).resolve(strict=True):
        raise ValueError('Stage-1 Gaussian does not match the completed workflow')
    if output_root:
        root = Path(output_root).expanduser().resolve(strict=True)
        if not run_dir.is_relative_to(root) or len(run_dir.relative_to(root).parts) != 4:
            raise ValueError('Manifest ROI run is outside --output-dir or has an invalid directory hierarchy')
        relative = run_dir.relative_to(root).parts
        if relative[1] != 'methods' or not relative[3].startswith('roi_'):
            raise ValueError('Stage-1 manifest does not have the standard scene/method/ROI hierarchy')
    declared_roi = Path(inputs['roi_json']).resolve(strict=True)
    # For historical Stage-1 snapshots, verify the ORIGINAL confirmed ROI too.
    # This matters in resume-only mode, where no WorkflowManager status is present.
    old_snapshot = (run_dir / 'metadata' / 'roi.json').resolve()
    if declared_roi == old_snapshot:
        report = _read_json(run_dir / 'metadata' / 'processing_report.json')
        original_roi = Path(report['source_roi']).resolve(strict=True)
        if _sha(original_roi) != _sha(declared_roi):
            raise ValueError('Historical original ROI changed after Stage-1; refuse resume')
    if expected_roi is not None:
        roi = Path(expected_roi).resolve(strict=True)
        declared = Path(inputs['roi_json']).resolve(strict=True)
        if declared != roi:
            # Historical Stage-1 manifests declared a copied metadata/roi.json.
            # Read-only legacy compatibility: never add a new copy or edit the old manifest.
            if (declared != old_snapshot or
                    Path(report.get('source_roi', '')).resolve() != roi or
                    _sha(declared) != _sha(roi)):
                raise ValueError('Confirmed ROI and manifest ROI identity differ')
    return run_dir, mf, data, inputs


def _completed_manifest(status):
    """Bind the workflow just confirmed in Edge to a validated immutable manifest."""
    if not isinstance(status, dict) or status.get('status') != 'completed':
        raise RuntimeError('阶段一未返回已验证的完成状态；不会启动阶段二')
    run_dir = Path(status['output_dir']).resolve(strict=True)
    roi = Path(status['roi']).resolve(strict=True)
    ply = Path(status['final_ply']).resolve(strict=True)
    if _sha(roi) != status.get('roi_sha256'):
        raise ValueError('Stage-1 completed workflow ROI SHA256 mismatch')
    if ply != (run_dir / 'final' / 'gaussian_roi.ply').resolve(strict=True):
        raise ValueError('Stage-1 completed workflow Gaussian path mismatch')
    return _validate_manifest(run_dir / 'metadata' / 'manifest.json',
                              expected_run=run_dir, expected_roi=roi, expected_ply=ply)


def _reserve_stage2_dir(roi_run, preferred):
    """Each recovery/attempt is isolated. NEVER reuse a failed or successful output directory."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', preferred) or preferred in ('.', '..'):
        raise ValueError('Invalid --stage2-name: expected a single plain folder name')
    root = Path(roi_run) / 'cleanup'
    root.mkdir(parents=True, exist_ok=True)
    match = re.fullmatch(r'(.+)_v([1-9][0-9]*)', preferred)
    for attempt in range(10000):
        if attempt == 0:
            name = preferred
        elif match:
            name = f'{match.group(1)}_v{int(match.group(2))+attempt}'
        else:
            name = f'{preferred}_v{attempt+1}'
        p = root / name
        try:
            p.mkdir(exist_ok=False)
            return p
        except FileExistsError:
            pass
    raise FileExistsError('No unused Stage-2 output directory found')


def _stage2_command(a, manifest, output):
    command = [sys.executable, '-m', 'gsrefine.cleanup.stage2_onepass',
               '--manifest', str(manifest), '--output-dir', str(output),
               '--chunk-size', str(a.chunk_size), '--nearest-k', str(a.nearest_k),
               '--min-path-agree', str(a.min_path_agree), '--ray-batch', str(a.ray_batch)]
    command.extend(['--classifier', a.classifier])
    if getattr(a, 'partition_report', None):
        command.extend(['--partition-report', str(a.partition_report)])
    if a.accept_auto_exterior:
        command.append('--accept-auto-exterior')
    if a.exterior_seed is not None:
        command.extend(['--exterior-seed', *(str(x) for x in a.exterior_seed)])
    for attr in ('no_aux_protection', 'no_aux_preview', 'keep_classification_previews', 'reclassify', 'audit_aux_ablation'):
        if getattr(a, attr):
            command.append('--' + attr.replace('_', '-'))
    return command


def _validate_stage2_completed(output, manifest, inputs):
    report_path = output / 'stage2_final_report.json'
    report = _read_json(report_path)
    status = report.get('status')
    allowed = {'COMPLETED_EXPERIMENTAL_CLEAN_ROI_GAUSSIAN':output/'gaussian_roi_clean.ply'}
    if report.get('schema') == 'GSSceneRefine.local_region.v1':
        allowed.update(PARTIAL_VERIFIED_ROI_CLEANUP=output/'gaussian_roi_partial.ply',
                       NO_VERIFIED_CLEANUP_SCOPE=inputs['gaussian_ply'])
    if status not in allowed:
        raise RuntimeError('Invalid Stage-2 terminal status')
    clean = allowed[status].resolve(strict=True)
    if not clean.stat().st_size:
        raise RuntimeError('Stage 2 did not report a completed nonempty Gaussian ROI')
    upstream = report.get('input', {})
    published = report.get('outputs', {})
    if (Path(upstream.get('stage1_manifest', '')).resolve() != manifest or
            upstream.get('stage1_manifest_sha256') != _sha(manifest) or
            Path(upstream.get('original_roi_gaussian', '')).resolve() != inputs['gaussian_ply'] or
            upstream.get('original_roi_gaussian_sha256') != _sha(inputs['gaussian_ply']) or
            Path(published.get('final_supersplat_gaussian_ply', '')).resolve() != clean or
            published.get('final_ply_sha256') != _sha(clean)):
        raise RuntimeError('Stage-2 report source/output fingerprints do not match this ROI')
    return report, clean


def _state_path(roi_run):
    return roi_run / 'metadata' / 'full_pipeline_status.json'


def _code_identity():
    repo = Path(__file__).resolve().parents[2]
    def git(*args):
        result = subprocess.run(['git','-C',str(repo),*args], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return dict(git_head=git('rev-parse','HEAD'), git_dirty=bool(git('status','--porcelain')),
                implementation='engineering_unified_v8')


def _write_state(roi_run, stage, manifest, output=None, *, extra=None):
    path = _state_path(roi_run)
    previous = _read_json(path) if path.is_file() else {}
    history = previous.get('attempt_history', [])
    old = {k: previous.get(k) for k in ('status', 'stage2_output_dir', 'updated_utc')}
    if previous.get('stage2_output_dir') and old['stage2_output_dir'] != str(output):
        history = [*history, old]
    record = {
        'schema': SCHEMA, 'status': stage, 'updated_utc': _utc(),
        'manifest': str(manifest), 'manifest_sha256': _sha(manifest),
        'stage2_output_dir': str(output) if output is not None else None,
        'attempt_history': history,
        'execution_code': _code_identity(),
    }
    if extra:
        record.update(extra)
    _atomic_json(path, record)
    return record


def run(a):
    if a.chunk_size < 100 or a.nearest_k < a.min_path_agree or a.min_path_agree < 2 or a.ray_batch < 1:
        raise ValueError('Invalid Stage-2 chunk / parity neighbour / ray-batch settings')
    if a.resume_manifest:
        print('[full-pipeline] Resume: read exact completed Stage-1 manifest; do not launch Edge/Stage-1', flush=True)
        roi_run, manifest, _, inputs = _validate_manifest(a.resume_manifest, output_root=a.output_dir)
    else:
        if not a.output_dir:
            raise ValueError('--output-dir is required when starting interactive Stage 1')
        from gsrefine.roi_init.workflow import workflow_options
        if not workflow_options(a.config)['enabled']:
            raise ValueError('Stage-1 config workflow.auto_after_confirm must be enabled')
        from gsrefine.roi_init.server import main as roi_init_main
        print('[full-pipeline] [1/2] Stage 1: Edge ROI confirmation -> verified Gaussian ROI', flush=True)
        status = roi_init_main(_stage1_argv(a))
        roi_run, manifest, _, inputs = _completed_manifest(status)
    # Reverify source identity immediately before Stage 2; no glob/latest scan.
    roi_run, manifest, _, inputs = _validate_manifest(manifest, expected_run=roi_run, output_root=a.output_dir)
    stage2_dir = _reserve_stage2_dir(roi_run, a.stage2_name)
    state = _write_state(roi_run, 'STAGE2_RUNNING', manifest, stage2_dir,
                         extra={'stage1_status': 'COMPLETED', 'resume': bool(a.resume_manifest),
                                'roi_json': str(inputs['roi_json']), 'roi_sha256': _sha(inputs['roi_json'])})
    print('[full-pipeline] [2/2] Stage 2: ' + str(stage2_dir), flush=True)
    try:
        # Scientific Stage-2 implementation/arguments are left intact.
        proc = subprocess.run(_stage2_command(a, manifest, stage2_dir), check=False)
        if proc.returncode:
            raise RuntimeError(f'阶段二进程退出码={proc.returncode}；'
                               '参见本次输出目录下的stage2_failure_report.json')
        report, clean = _validate_stage2_completed(stage2_dir, manifest, inputs)
        terminal = 'COMPLETED' if report['status']=='COMPLETED_EXPERIMENTAL_CLEAN_ROI_GAUSSIAN' else report['status']
        state = _write_state(roi_run, terminal, manifest, stage2_dir,
                             extra={'stage1_status': 'COMPLETED', 'resume': bool(a.resume_manifest),
                                    {'COMPLETED':'cleaned_gaussian', 'PARTIAL_VERIFIED_ROI_CLEANUP':'partially_cleaned_gaussian',
                                     'NO_VERIFIED_CLEANUP_SCOPE':'retained_stage1_gaussian'}[terminal]: str(clean),
                                    'stage2_report': str(stage2_dir/'stage2_final_report.json'),
                                    'counts': report.get('counts', {}),
                                    'quality_review': {
                                        'auto_exterior_seed_is_provisional': bool(a.accept_auto_exterior),
                                        'stage1':_read_json(roi_run/'metadata'/'processing_report.json').get('quality_review',{}),
                                        'stage2_unverified_gaussians': report.get('counts',{}).get('unverified_retained'),
                                        'engineering_audit':report.get('engineering_audit'),
                                        'truth_certified':False}})
        print('[full-pipeline] '+terminal+' '+str(clean), flush=True)
        return state
    except BaseException as exc:
        _write_state(roi_run, 'STAGE2_FAILED', manifest, stage2_dir,
                     extra={'stage1_status':'COMPLETED', 'resume': bool(a.resume_manifest),
                            'error': type(exc).__name__ + ': ' + str(exc)})
        raise


def main(argv=None):
    a = _parser().parse_args(argv)
    try:
        from time import perf_counter
        if a.compare_stage2 and (a.classifier != 'original-region' or not a.no_aux_protection):
            raise ValueError('--compare-stage2 needs --classifier original-region --no-aux-protection')
        start = perf_counter()
        state = run(a)
        if a.compare_stage2:
            from gsrefine.cleanup.compare_original_region import compare
            compare(Path(state['stage2_output_dir']), a.compare_stage2, perf_counter()-start)
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print('[full-pipeline] STOPPED: ' + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()
