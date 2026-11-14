"""Run GSSceneRefine interactive Stage 1 followed by the verified Stage 2.

This is an orchestration-only module: the scientific Stage-1 and Stage-2
algorithms, their CLI interfaces and their output structure remain unchanged.

python run.py full-pipeline --config configs/default.yaml --scene ... --method ... \
    --sfm-dir ... --gaussian-ply ... --output-dir ... --accept-auto-exterior
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


SCHEMA = 'GSSceneRefine.full_pipeline.v1'


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _parser():
    parser = argparse.ArgumentParser(
        description='一条命令：Edge交互ROI → Stage1最终R3 → Stage2异常Gaussian清理。旧命令保持不变。')
    group = parser.add_argument_group('原阶段一 roi-init 参数（原样转发）')
    group.add_argument('--config', default='configs/default.yaml')
    group.add_argument('--scene')
    group.add_argument('--method')
    group.add_argument('--sfm-dir')
    group.add_argument('--gaussian-ply')
    group.add_argument('--colmap-points')
    group.add_argument('--colmap-images')
    group.add_argument('--output-dir', required=True, help='与 roi-init 相同：整个GSSceneRefine_results根目录')
    group.add_argument('--keep-diagnostics', action='store_true')
    group.add_argument('--no-auto-windows', action='store_true')

    stage2 = parser.add_argument_group('阶段二自动清理选项')
    anchor = stage2.add_mutually_exclusive_group(required=True)
    anchor.add_argument('--accept-auto-exterior', action='store_true',
                        help='明确接受阶段二自动选取的临时外部空间种子（需研究者检查结果）')
    anchor.add_argument('--exterior-seed', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                        help='提供已确认的外部空间种子坐标')
    stage2.add_argument('--stage2-name', default='stage2_final_v1',
                        help='阶段二输出子目录名；如已存在，自动选取下一未使用的v编号')
    stage2.add_argument('--no-aux-protection', action='store_true',
                        help='与原阶段二命令同义：仅用于无辅助保护的消融实验')
    stage2.add_argument('--no-aux-preview', action='store_true')
    stage2.add_argument('--keep-classification-previews', action='store_true')
    stage2.add_argument('--reclassify', action='store_true')
    stage2.add_argument('--chunk-size', type=int, default=30000)
    stage2.add_argument('--nearest-k', type=int, default=8)
    stage2.add_argument('--min-path-agree', type=int, default=2)
    stage2.add_argument('--ray-batch', type=int, default=8000)
    return parser


def _stage1_argv(a) -> list[str]:
    # Match the existing server.py argparse exactly; do not invent new flags.
    result = []
    for name in ('config', 'scene', 'method', 'sfm_dir', 'gaussian_ply',
                 'colmap_points', 'colmap_images', 'output_dir'):
        value = getattr(a, name)
        if value is not None:
            result.extend(('--' + name.replace('_', '-'), str(value)))
    if a.keep_diagnostics:
        result.append('--keep-diagnostics')
    if a.no_auto_windows:
        result.append('--no-auto-windows')
    return result


def _completed_manifest(status: dict):
    """Use validated WorkflowManager status, NEVER locate a ROI by glob/latest."""
    if not isinstance(status, dict) or status.get('status') != 'completed':
        raise RuntimeError('阶段一未返回已验证的完成状态；不会启动阶段二')
    roi_run = Path(status['output_dir']).expanduser().resolve(strict=True)
    roi_file = Path(status['roi']).expanduser().resolve(strict=True)
    final_ply = Path(status['final_ply']).expanduser().resolve(strict=True)
    manifest = roi_run / 'metadata' / 'manifest.json'
    data = json.loads(manifest.read_text(encoding='utf-8'))
    inputs = data.get('stage2_inputs') or {}
    if data.get('status') != 'completed':
        raise ValueError('阶段一manifest未标记为completed；停止阶段二')
    if final_ply != (roi_run / 'final' / 'gaussian_roi.ply').resolve(strict=True):
        raise ValueError('阶段一成功状态的Gaussian路径与标准ROI结果不一致')
    if Path(inputs['gaussian_ply']).resolve(strict=True) != final_ply:
        raise ValueError('阶段一manifest中的Gaussian与完成状态不一致')
    if Path(inputs['roi_json']).resolve(strict=True) != roi_file:
        raise ValueError('阶段一manifest中的ROI与实际确认ROI不一致')
    if not final_ply.is_file() or final_ply.stat().st_size <= 0:
        raise ValueError('阶段一最终Gaussian PLY为空或不存在')
    return roi_run, manifest


def _reserve_stage2_dir(roi_run: Path, preferred: str) -> Path:
    """Create an EMPTY destination atomically, never reuse or overwrite a run."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', preferred) or preferred in ('.', '..'):
        raise ValueError('非法--stage2-name：只能是单层目录名称')
    root = roi_run / 'cleanup'
    root.mkdir(parents=True, exist_ok=True)
    m = re.fullmatch(r'(.+)_v([1-9][0-9]*)', preferred)
    for i in range(1, 10000):
        name = preferred if i == 1 else (f'{m.group(1)}_v{int(m.group(2))+i-1}'
                                          if m else f'{preferred}_v{i}')
        folder = root / name
        try:
            folder.mkdir(exist_ok=False)
            return folder
        except FileExistsError:
            continue
    raise FileExistsError('已有太多相同名称的阶段二输出，无法创建独立目录')


def _stage2_command(a, manifest: Path, output: Path) -> list[str]:
    cmd = [sys.executable, '-m', 'gsrefine.cleanup.stage2_onepass',
           '--manifest', str(manifest), '--output-dir', str(output),
           '--chunk-size', str(a.chunk_size), '--nearest-k', str(a.nearest_k),
           '--min-path-agree', str(a.min_path_agree), '--ray-batch', str(a.ray_batch)]
    if a.accept_auto_exterior:
        cmd.append('--accept-auto-exterior')
    if a.exterior_seed is not None:
        cmd.extend(['--exterior-seed', *(str(x) for x in a.exterior_seed)])
    for attr in ('no_aux_protection', 'no_aux_preview',
                 'keep_classification_previews', 'reclassify'):
        if getattr(a, attr):
            cmd.append('--' + attr.replace('_', '-'))
    return cmd


def run(a) -> dict:
    # Fail before launching Edge when it is impossible to proceed to Stage2.
    if a.chunk_size < 100 or a.nearest_k < a.min_path_agree or a.min_path_agree < 2:
        raise ValueError('无效的阶段二参数：chunk-size >=100，nearest-k >= min-path-agree >=2')
    from gsrefine.roi_init.workflow import workflow_options
    if not workflow_options(a.config)['enabled']:
        raise ValueError('config中workflow.auto_after_confirm没有启用，无法实现一键执行；请启用后再运行')
    from gsrefine.roi_init.server import main as roi_init_main
    print('[full-pipeline] [1/2] 启动现有阶段一：Edge ROI确认 → R3恢复 → 高斯ROI导出', flush=True)
    stage1_status = roi_init_main(_stage1_argv(a))
    roi_run, manifest = _completed_manifest(stage1_status)
    output = _reserve_stage2_dir(roi_run, a.stage2_name)
    state_file = roi_run / 'metadata' / 'full_pipeline_status.json'
    state = {'schema': SCHEMA, 'status': 'STAGE2_RUNNING',
             'updated_utc': _utc(), 'roi': str(stage1_status['roi']),
             'manifest': str(manifest), 'stage1_result': stage1_status['final_ply'],
             'stage2_output_dir': str(output)}
    _atomic_json(state_file, state)
    print(f'[full-pipeline] [1/2] 阶段一验证成功：{stage1_status["final_ply"]}', flush=True)
    print(f'[full-pipeline] [2/2] 开始阶段二：{output}', flush=True)
    cmd = _stage2_command(a, manifest, output)
    try:
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f'阶段二进程退出码={proc.returncode}；请查看该目录中的stage2_failure_report.json')
        report_file = output / 'stage2_final_report.json'
        report = json.loads(report_file.read_text(encoding='utf-8'))
        clean = output / 'gaussian_roi_clean.ply'
        if report.get('status') != 'COMPLETED_EXPERIMENTAL_CLEAN_ROI_GAUSSIAN':
            raise RuntimeError('阶段二未返回正式完成状态')
        if not clean.is_file() or clean.stat().st_size == 0:
            raise RuntimeError('阶段二虽退出成功，但没有有效的gaussian_roi_clean.ply')
        state.update(status='COMPLETED', updated_utc=_utc(),
                     cleaned_gaussian=str(clean), stage2_report=str(report_file),
                     counts=report.get('counts', {}))
        _atomic_json(state_file, state)
        print('[full-pipeline] 两阶段均已成功完成', flush=True)
        print('[full-pipeline] 阶段一：' + str(stage1_status['final_ply']), flush=True)
        print('[full-pipeline] 最终干净高斯：' + str(clean), flush=True)
        return state
    except BaseException as exc:
        state.update(status='STAGE2_FAILED', updated_utc=_utc(),
                     error=f'{type(exc).__name__}: {exc}')
        _atomic_json(state_file, state)
        raise


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print('[full-pipeline] 已停止：' + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()
