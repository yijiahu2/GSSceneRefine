#!/usr/bin/env python3
"""在 WSL 中补记一次已经启动的 GSSceneRefine full-pipeline 运行。"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

UTC = timezone.utc


def timestamp(value):
    if not value:
        return None
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        # Also accept LC_ALL=C TZ=UTC ps -o lstart output.
        result = datetime.strptime(value.strip(), '%a %b %d %H:%M:%S %Y')
    return result.replace(tzinfo=UTC) if result.tzinfo is None else result.astimezone(UTC)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def active_runs():
    runs = []
    offset = time.time() - time.clock_gettime(time.CLOCK_BOOTTIME)
    ticks_per_second = os.sysconf('SC_CLK_TCK')
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for flag in ('scene', 'method', 'output-dir', 'resume-manifest'):
        parser.add_argument('--' + flag)
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            args = (entry / 'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')
            if (len(args) < 3 or not Path(args[0]).name.startswith('python')
                    or Path(args[1]).name != 'run.py' or args[2] != 'full-pipeline'):
                continue
            opt, _ = parser.parse_known_args(args[3:])
            if opt.resume_manifest or not (opt.output_dir and opt.scene and opt.method):
                continue
            cwd = (entry / 'cwd').resolve(strict=True)
            method_dir = (cwd / opt.output_dir / opt.scene / 'methods' / opt.method).resolve()
            # Fields after the process name begin at stat field 3. Start ticks are field 22.
            ticks = int((entry / 'stat').read_text().rsplit(')', 1)[1].split()[19])
            start = datetime.fromtimestamp(offset + ticks / ticks_per_second, UTC)
            runs.append((method_dir, start))
            print(f'已读取 PID {entry.name}（{opt.scene}/{opt.method}）启动 UTC {start.isoformat()}')
        except (OSError, ValueError, IndexError):
            continue
    return runs


def created(path):
    # GNU stat birth time. Modification time changes as files are added.
    result = subprocess.run(['stat', '-c', '%w', '--', str(path)], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            env={**os.environ, 'LC_ALL': 'C', 'TZ': 'UTC'})
    raw = result.stdout.strip()
    if result.returncode or raw in ('', '-'):
        return None
    date, clock, _zone = raw.split()
    # GNU stat emits nanoseconds. Python 3.9/3.10 accepts six fractional digits.
    whole, _, fraction = clock.partition('.')
    return timestamp(f'{date}T{whole}.{fraction[:6]:0<6}+00:00')


def collect(dest, starts, supplied):
    dest = dest.resolve(strict=True)
    if dest.name == 'gaussian_roi_clean.ply':
        dest = dest.parent
    run = dest.parent.parent
    if dest.parent.name != 'cleanup':
        raise ValueError('请输入本次 cleanup 下的阶段二目录，或最终 gaussian_roi_clean.ply 路径。')
    state = read(run / 'metadata/full_pipeline_status.json')
    if Path(state['stage2_output_dir']).resolve() != dest or state['status'] != 'COMPLETED':
        raise ValueError('该目录尚未完成，或对应另一轮运行。请使用本次 Both stages completed 行中的路径。')
    final = read(dest / 'stage2_final_report.json')
    end = timestamp(state['updated_utc'])
    stage2 = created(dest)
    if stage2 and stage2 > end:
        stage2 = None
    progress = read(run / 'logs/workflow_progress.json') if not state['resume'] else {}
    queued = timestamp(progress.get('started_utc'))
    ready = timestamp(progress.get('updated_utc')) if progress.get('phase') == 'completed' else None
    if not state['resume'] and ready and stage2 and stage2 < ready:
        stage2 = None
    start = supplied
    if start is None and queued and ready:
        matches = [t for path, t in starts if path == run.parent and t <= queued <= ready <= end]
        if len(matches) == 1:
            start = matches[0]
    if start and (start > end or (queued and start > queued) or (stage2 and start > stage2)):
        raise ValueError('启动时间晚于本次任务的后续记录，请检查 UTC 时间与 ROI 路径。')

    rows = []
    def add(label, value, source):
        rows.append([label, '' if value is None else f'{value:.3f}',
                     '' if value is None else f'{value / 60:.3f}', source])
    def span(label, a, b, source):
        seconds = (b - a).total_seconds() if a and b and a <= b else None
        add(label, seconds, source if seconds is not None else '缺少可核验的时间边界')
    span('本次命令总耗时', start, end, '进程启动至主程序写入 COMPLETED；含 ROI 交互')
    if state['resume']:
        add('阶段一', None, '本次恢复运行跳过阶段一')
    else:
        span('阶段一（含交互和阶段交接，近似）', start, stage2, '阶段分界采用本次阶段二目录创建时间')
        span('其中：ROI 确认后模块三、四合计', queued, ready, 'workflow_progress：排队启动至成果发布完成')
        span('确认后任务排队至全流程完成', queued, end, '实有记录覆盖的区间，属于全流程的一部分')
    span('阶段二（近似）', stage2, end, '本次阶段二目录创建至主程序写入 COMPLETED；请在原始输出目录运行')
    report_path = Path(final['input']['parity_report']).resolve()
    if report_path == dest / 'diagnostics/parity/roi_parity_report.json':
        perf = read(report_path).get('short_ray', {})
        for key, title in [('total_seconds', '候选空间分类合计'), ('index_seconds', '构建分界面索引'),
                           ('representative_seconds', '代表点短射线'), ('fallback_seconds', '剩余候选逐点短射线')]:
            add(title, perf.get(key), '分类器内部计时；子项已包含在合计内' if key in perf else '报告未记录此项')
        for level in perf.get('levels', []):
            add(f"体素第 {level['depth']} 层", level['seconds'], '该层包含其代表点查询，已包含在分类合计内')
    else:
        add('候选空间分类合计', None, '本次复用历史分类结果')
    for label in ('ROI 单独绘制', '全量 Gaussian 精确点面距离', '辅助分量保护'):
        add(label, None, '普通入口无此步骤的独立起止记录')
    return dest, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', nargs='?', help='本次最终 PLY 或其父目录')
    parser.add_argument('--start-utc', type=timestamp, help='另行记录的本次进程启动时间（UTC）')
    opt = parser.parse_args()
    starts = active_runs()
    path = opt.result or input('等原终端出现 Both stages completed 后，粘贴该行的完整 PLY 路径并回车：\n').strip()
    path = path.removeprefix('[full-pipeline] Both stages completed: ').strip().strip('"').strip("'")
    dest, rows = collect(Path(path).expanduser(), starts, opt.start_utc)
    # Keep every export, including a later one supplemented with a known start time.
    with tempfile.NamedTemporaryFile(dir=dest, prefix='timing_posthoc_', suffix='.csv',
                                     mode='w', encoding='utf-8-sig', newline='', delete=False) as stream:
        writer = csv.writer(stream)
        writer.writerow(['项目', '耗时（秒）', '耗时（分钟）', '统计口径'])
        writer.writerows(rows)
    for label, sec, _, _ in rows:
        print(f'{sec or "—":>11}  {label}')
    print(f'已保存（秒）：{stream.name}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, EOFError) as exc:
        sys.exit(f'无法统计：{exc}')
