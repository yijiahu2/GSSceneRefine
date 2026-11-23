#!/usr/bin/env python3
"""在项目根目录运行：python /mnt/f/Edge/time_full_pipeline.py full-pipeline [原命令参数]"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime

SELF = Path(__file__).resolve()
# 每行：阶段、模块、互斥步骤。下面的函数只记录真正执行过的步骤。
STEPS = {
    'init': (1,'启动与交接','启动、参数及输入准备'),
    'g0': (1,'模块一 · 场景几何','输入签名与几何缓存检查'),
    'g1': (1,'模块一 · 场景几何','COLMAP 读取及 SfM 净化'),
    'g2': (1,'模块一 · 场景几何','法向估计与 Poisson 重建'),
    'g3': (1,'模块一 · 场景几何','几何校验、配置与方向准备'),
    'gc': (1,'模块一 · 场景几何','复用缓存、核验当前模型及准备配置'),
    'ui': (1,'模块二 · 交互 ROI','准备模型服务及绘制界面'),
    'draw': (1,'模块二 · 交互 ROI','等待 ROI 确认（含加载、绘制、预览、保存）'),
    'launch': (1,'模块二 · 交互 ROI','确认后启动 R3 工作进程'),
    'validating': (1,'模块三 · R3 恢复','ROI 与输入校验'),
    'loading': (1,'模块三 · R3 恢复','加载 Gaussian、参考曲面与预处理'),
    'poisson': (1,'模块三 · R3 恢复','局部误截断识别'),
    'core': (1,'模块三 · R3 恢复','局部主体恢复'),
    'safety': (1,'模块三 · R3 恢复','自适应安全外扩及 SfM 交接准备'),
    'exporting': (1,'模块四 · R3 导出','导出阶段一 Gaussian PLY'),
    'verifying': (1,'模块四 · R3 导出','恢复统计核对与成果发布'),
    'handoff': (1,'启动与交接','阶段一收尾与 manifest 交接'),
    'boot': (2,'启动与几何准备','启动、manifest 与已有结果检查'),
    'geo': (2,'启动与几何准备','几何来源核验与缓存查找'),
    'paritycache': (2,'启动与几何准备','准备分类输入、查找或核验可用缓存'),
    'sfm': (2,'启动与几何准备','R3 核心 SfM 准备或校验'),
    'clip': (2,'启动与几何准备','按 R3 裁剪原始主体或校验裁剪缓存'),
    'parts': (2,'启动与几何准备','主辅曲面划分、校验与几何记录'),
    'p95': (2,'启动与几何准备','核心 SfM 到局部主体的拟合统计（含 P95）'),
    'prep': (2,'空间分隔与导航','准备分类几何、恢复 R3 场及三角面索引'),
    'audit': (2,'空间分隔与导航','几何审计、最终 R3 场与全场景主体准备'),
    'loops': (2,'空间分隔与导航','原始主体与 R3 交线、采样窗口准备'),
    'coarse': (2,'空间分隔与导航','粗网格连通性检查'),
    'fine': (2,'空间分隔与导航','细网格连通性检查及审计收尾'),
    'gate': (2,'空间分隔与导航','审计来源与内外分隔核验'),
    'nav': (2,'空间分隔与导航','导航网格、来源与节点核验'),
    'graph': (2,'空间分隔与导航','邻边过滤、三角面奇偶判定、图标记与保存'),
    'loadgs': (2,'距离与候选分类','读取 R3 Gaussian 中心'),
    'scale': (2,'距离与候选分类','Gaussian 半径统计'),
    'dist': (2,'距离与候选分类','阈值、全量点面距离及超阈值候选筛选'),
    'side': (2,'距离与候选分类','超阈值空间分类（含准备、索引与回退）'),
    'diag': (2,'距离与候选分类','生成分类掩膜、预览及报告'),
    'check': (2,'距离与候选分类','读取与核验分类结果、准备辅助保护'),
    'aux': (2,'辅助保护与发布','辅助曲面保护（或执行禁用分支）'),
    'mask': (2,'辅助保护与发布','合并并核验最终删除掩膜'),
    'ply': (2,'辅助保护与发布','写出最终 gaussian_roi_clean.ply'),
    'publish': (2,'辅助保护与发布','诊断导出、报告与成果校验'),
}

class Recorder:
    def __init__(self, folder):
        self.path = Path(folder) / f'{os.getpid()}.jsonl'

    def mark(self, code, note=''):
        rec = [time.monotonic_ns(), code, str(note)]
        try:
            with self.path.open('a', encoding='utf8') as f:
                f.write(json.dumps(rec, ensure_ascii=False)+'\n')
        except FileNotFoundError:
            # The original ROI worker may outlive a manually interrupted parent.
            if self.path.parent.exists():
                raise


def tap(module, attr, callback, *, after=None):
    original = getattr(module, attr)
    @functools.wraps(original)
    def observed(*args, **kwargs):
        callback(*args, **kwargs)
        result = original(*args, **kwargs)
        if after:
            after()
        return result
    setattr(module, attr, observed)


def run_worker(role, args, root, folder):
    sys.path.insert(0, str(root))
    rec = Recorder(folder)
    mark = rec.mark
    if role == 'roi-auto':
        from gsrefine.roi_init import workflow
        def progress(out, phase, **kwargs):
            if phase in STEPS:
                mark(phase)
            elif phase == 'completed':
                mark('handoff')
        tap(workflow, 'update_progress', progress)
        return workflow.main(args)

    from gsrefine.cleanup import stage2_onepass as runner, roi_barrier_parity as parity
    from gsrefine.cleanup import direct_r3_pipeline, sdf_common
    from tools import audit_final_r3_global_poisson_barrier as audit
    # 只包装粗粒度函数；点、射线、三角面和 kNN 内循环保持原样。
    def enter(module, name, key, end=None):
        tap(module, name, lambda *a, **k: mark(key), after=(lambda: mark(end)) if end else None)
    enter(runner, '_prepare_required_geometry', 'geo', 'paritycache')
    enter(direct_r3_pipeline, 'sfm_fit', 'p95', 'parts')
    enter(parity, '_audit_identity', 'gate')
    enter(sdf_common, 'robust_gaussian_radius', 'scale', 'dist')
    enter(runner, '_combine_removal', 'mask')
    def export(model, mask, outfile):
        mark('ply' if Path(outfile).name == 'gaussian_roi_clean.ply' else 'publish')
    tap(runner, '_export_native', export, after=lambda: mark('publish'))
    def steps(msg):
        text = str(msg)
        for prefix, key in [('1/5 ','sfm'),('2/5 ','clip'),('3/5 ','parts'),
                            ('4/5 Classifying','prep'),('4/5 Reusing','check'),('5/5 ','aux')]:
            if text.startswith(prefix):
                mark(key)
                if prefix == '4/5 Reusing':
                    mark('@cached_class')
                return
        if text.startswith('Reusing authenticated'):
            mark('@cached_geom')
    tap(runner, '_log', steps)
    def parity_steps(msg):
        for prefix, key in [('No prior geometry audit','audit'),('Original-triangle region separation','nav'),
                            ('Building parity graph:','graph'),
                            ('Navigation:','loadgs'),('Adaptive delta=','side'),('Classifier=','diag'),
                            ('Saved report:','check')]:
            if str(msg).startswith(prefix):
                mark(key)
                return
    tap(parity, '_log', parity_steps)
    def audit_steps(msg):
        if msg in ('Water-flood audit level: coarse','Water-flood audit level: fine'):
            mark(msg.split()[-1])
        elif msg.startswith('Computing actual FINAL R3 x'):
            mark('loops')
    tap(audit, 'log', audit_steps)
    return runner.main(args)


def run_full(args, root, folder, resume):
    sys.path.insert(0, str(root))
    rec = Recorder(folder)
    from gsrefine.roi_init import full_pipeline as cli

    # Preserve the original interpreter, arguments, working directories and stdout
    # redirection. Instrument only the two workers this pipeline actually starts.
    original = subprocess.Popen
    class TimedPopen(original):
        def __init__(self, args, *pos, **kw):
            if isinstance(args, (tuple,list)) and len(args) >= 3 and args[0] == sys.executable:
                role = None
                if tuple(args[1:3]) == ('-m', 'gsrefine.cleanup.stage2_onepass'):
                    role = 'stage2'
                elif Path(args[1]) == root/'run.py' and args[2] == 'roi-auto':
                    role = 'roi-auto'
                if role:
                    args = [sys.executable, str(SELF), '--worker', role, str(root), str(folder), *args[3:]]
            super().__init__(args, *pos, **kw)
    subprocess.Popen = TimedPopen
    def full_print(*values, **kwargs):
        if values and str(values[0]).startswith('[full-pipeline] [2/2] Stage 2: '):
            rec.mark('boot', str(values[0]).split('Stage 2: ',1)[1])
        print(*values, **kwargs)
    cli.print = full_print
    if not resume:
        from gsrefine.roi_init import server
        from gsrefine.geometry import prepare
        def server_print(*values, **kwargs):
            text = str(values[0]) if values else ''
            for prefix, key in [('[pipeline] [1/4] 模块一：','g0'),('[pipeline] [1/4] 模块一完成','ui'),
                                ('[pipeline] [2/4] 模块二：','draw'),('[roi-init] ROI confirmed:','launch')]:
                if text.startswith(prefix):
                    rec.mark(key)
                    break
            print(*values, **kwargs)
        server.print = server_print
        def status(output, phase, message, **values):
            key = {'cleaning':'g1','reconstructing':'g2','verifying':'g3','cached':'gc'}.get(phase)
            if key:
                rec.mark(key)
        tap(prepare, '_status', status)
    try:
        return cli.main(args)
    finally:
        subprocess.Popen = original


def write_report(folder, out, start, end, resume, status, args):
    events = []
    for path in folder.glob('*.jsonl'):
        for line in path.read_bytes().splitlines(keepends=True):
            if line.endswith(b'\n'):
                events.append(json.loads(line))
    durations = defaultdict(int)
    current, last = ('boot' if resume else 'init'), start
    stage2_dir = None
    flags = set()
    for tick, code, note in sorted(events):
        if code.startswith('@'):
            flags.add(code)
            continue
        if tick < last or tick > end:
            continue
        if code == 'boot' and note:
            stage2_dir = Path(note)
        durations[current] += tick - last
        current, last = code, tick
    durations[current] += end-last
    total = end-start
    def seconds(ns):
        return ns / 1e9
    total_seconds = seconds(total)
    rows = []
    def row(kind, stage, group, task, value, note=''):
        rows.append([kind, stage, group, task,
                     '' if value is None else f'{value:.6f}',
                     '' if value is None else f'{value/60:.4f}',
                     '' if value is None or not total else f'{value/total_seconds*100:.3f}', note])
    row('总计','全流程','','本次墙钟耗时',total_seconds,status)
    row('运行信息','全流程','','命令参数',None,shlex.join(['full-pipeline', *args]))
    if stage2_dir:
        row('运行信息','阶段二','','本次阶段二目录',None,str(stage2_dir))
    if 'draw' in durations:
        row('参考','全流程','','扣除等待 ROI 确认区间',seconds(total-durations['draw']),
            '包含启动、I/O 与校验；不等同于 CPU 时间')
    for stage in (1,2):
        seen = [k for k in durations if STEPS[k][0] == stage]
        st = '阶段一' if stage == 1 else '阶段二'
        row('阶段小计',st,'','阶段耗时',seconds(sum(durations[k] for k in seen)) if seen else None,
            ('恢复运行，本次跳过' if stage == 1 and resume else '本次未进入') if not seen else '')
        for group in dict.fromkeys(STEPS[k][1] for k in seen):
            keys = [k for k in seen if STEPS[k][1] == group]
            row('模块小计',st,group,'模块耗时',seconds(sum(durations[k] for k in keys)))
            for key in keys:
                comment = ''
                if key == 'geo' and '@cached_geom' in flags:
                    comment = '已复用主辅曲面及完整 P95，按本次核验耗时计'
                if key == 'check' and '@cached_class' in flags:
                    comment = '复用完整分类；本次未重跑候选分类'
                row('互斥步骤',st,group,STEPS[key][2],seconds(durations[key]),comment)
    if stage2_dir and '@cached_class' not in flags and 'side' in durations:
        # Do not mix historical cached timings into a new run.
        report = stage2_dir/'diagnostics/parity/roi_parity_report.json'
        if report.is_file():
            try:
                perf = json.loads(report.read_text(encoding='utf8')).get('short_ray', {})
            except (ValueError, OSError) as exc:
                row('运行信息','阶段二','','分类器内部明细不可读',None,str(exc))
                perf = {}
            fields = [('total_seconds','分类器内部总耗时'),('index_seconds','构建分界面查询索引'),
                      ('representative_seconds','代表点短射线（包含于细分层）'),
                      ('fallback_seconds','剩余候选逐点短射线')]
            for field, label in fields:
                if field in perf:
                    row('分类内计时','阶段二','距离与候选分类',label,perf[field],'包含在空间分类步骤内，不重复加总')
            for level in perf.get('levels', []):
                row('分类内计时','阶段二','距离与候选分类',f"体素第 {level['depth']} 层",
                    level['seconds'], f"本层直接分类 {level['direct']:,} 个，含本层代表点查询")
    out.mkdir(parents=True, exist_ok=True)
    path = out/'timing.csv'
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['层级','阶段','模块','项目','耗时（秒）','耗时（分钟）','占本次总耗时（%）','说明'])
        writer.writerows(rows)
    print(f'\n[计时] {status}   {path}', flush=True)
    for kind, stage, group, task, sec, _, _, _ in rows:
        if kind in ('总计','阶段小计','模块小计','参考'):
            print(f'  {sec or "—":>13} 秒   {stage}  {group or task}')


def main():
    if sys.argv[1:2] == ['--worker']:
        role, root, folder = sys.argv[2:5]
        return run_worker(role, sys.argv[5:], Path(root), Path(folder))
    args = sys.argv[1:]
    if args[:1] == ['full-pipeline']:
        args = args[1:]
    if not args or args == ['--help']:
        print(__doc__)
        print('命令从 GSSceneRefine 根目录执行。恢复运行同样支持 --resume-manifest。')
        return 0
    settings = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    settings.add_argument('--output-dir')
    settings.add_argument('--resume-manifest')
    opt, _ = settings.parse_known_args(args)
    root = Path.cwd().resolve()
    if not (root/'gsrefine/roi_init/full_pipeline.py').is_file():
        sys.exit('请先 cd /home/xth/GSSceneRefine，再运行此脚本。')
    if opt.output_dir:
        dest = Path(opt.output_dir).expanduser().resolve()
    elif opt.resume_manifest:
        dest = Path(opt.resume_manifest).expanduser().resolve().parent.parent
    else:
        sys.exit('从头运行请传 --output-dir，复用阶段一请传 --resume-manifest。')
    reports = dest/'_timings'
    reports.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S_'), dir=reports))
    print(f'[计时] 本次统计表将写入 {out / "timing.csv"}', flush=True)
    with tempfile.TemporaryDirectory(prefix='gsrefine-timing-') as private:
        start = time.monotonic_ns()
        status = '未完成'
        try:
            run_full(args, root, Path(private), bool(opt.resume_manifest))
            status = '完成'
            return 0
        except KeyboardInterrupt:
            status = '已中断，耗时截至中断'
            return 130
        except SystemExit as exc:
            status = '完成' if exc.code in (None, 0) else f'提前退出（退出码 {exc.code}），已计至退出时'
            raise
        except Exception as exc:
            status = f'失败，已计至退出时（{type(exc).__name__}）'
            raise
        finally:
            end = time.monotonic_ns()
            write_report(Path(private), out, start, end, bool(opt.resume_manifest), status, args)


if __name__ == '__main__':
    sys.exit(main())
