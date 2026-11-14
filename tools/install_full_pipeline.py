"""Install the unified CLI without replacing existing Stage-1/Stage-2 modules.

Only two existing entrypoints require tiny, checked modifications; the new
orchestrator is a separate module. `--check` is completely read-only.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'run.py'
SERVER = ROOT / 'gsrefine' / 'roi_init' / 'server.py'
NEW = ROOT / 'gsrefine' / 'roi_init' / 'full_pipeline.py'


def change_run(text: str) -> str:
    if "'full-pipeline':'gsrefine.roi_init.full_pipeline'," in text:
        return text
    anchor = "    'roi-init':'gsrefine.roi_init.server',"
    if text.count(anchor) != 1:
        raise RuntimeError('run.py入口与GitHub预期结构不一致；没有修改任何文件')
    return text.replace(anchor, anchor + "\n    'full-pipeline':'gsrefine.roi_init.full_pipeline',", 1)


def change_server(text: str) -> str:
    if '    return completed_status\n' in text and '    completed_status = None\n' in text:
        return text
    a = '    if workflow.launched_job() and not interrupted:\n'
    b = '        status=monitor_terminal(workflow)\n        if status[\'status\'] != \'completed\':\n            raise SystemExit(1)\n'
    if text.count(a) != 1 or text.count(b) != 1:
        raise RuntimeError('server.py阶段一终态与GitHub预期结构不一致；没有修改任何文件')
    text = text.replace(a, '    completed_status = None\n' + a, 1)
    text = text.replace(b, "        completed_status = monitor_terminal(workflow)\n"
                        "        if completed_status['status'] != 'completed':\n"
                        "            raise SystemExit(1)\n", 1)
    footer = "if __name__=='__main__':main()"
    if text.count(footer) != 1:
        raise RuntimeError('server.py末尾入口异常；没有修改任何文件')
    text = text.replace(footer, '    return completed_status\n\n' + footer, 1)
    return text


def apply(check_only: bool) -> None:
    for f in (RUN, SERVER, NEW):
        if not f.is_file():
            raise FileNotFoundError(f'必要文件不存在：{f}')
    source = {RUN: RUN.read_text(encoding='utf-8'), SERVER: SERVER.read_text(encoding='utf-8')}
    result = {RUN: change_run(source[RUN]), SERVER: change_server(source[SERVER])}
    for file, content in [*result.items(), (NEW, NEW.read_text(encoding='utf-8'))]:
        compile(content, str(file), 'exec')
    changed = [p for p in result if source[p] != result[p]]
    if check_only:
        print('[full-pipeline install] READ-ONLY CHECK PASSED')
        print('将更新：' + (', '.join(str(p.relative_to(ROOT)) for p in changed) if changed else '无需修改'))
        return
    if not changed:
        print('[full-pipeline install] 已安装，无需重复修改')
        return
    backup_dir = (ROOT / '.git' / 'gsrefine_patch_backups' /
                  ('full_pipeline_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')))
    if not (ROOT / '.git').is_dir():
        backup_dir = ROOT.parent / ('.GSSceneRefine_backup_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    backup_dir.mkdir(parents=True, exist_ok=False)
    for p in changed:
        dest = backup_dir / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    staged = {}
    try:
        for p in changed:
            fd, tmp = tempfile.mkstemp(prefix='.'+p.name+'.', suffix='.tmp', dir=p.parent)
            with open(fd, 'w', encoding='utf-8') as f:
                f.write(result[p])
                f.flush()
            staged[p] = Path(tmp)
        for p in changed:
            staged[p].replace(p)
        print('[full-pipeline install] 安装完成；原命令保持不变')
        print('[full-pipeline install] 备份目录：' + str(backup_dir))
    except BaseException:
        for p in changed:
            original = backup_dir / p.relative_to(ROOT)
            if original.is_file():
                shutil.copy2(original, p)
        raise
    finally:
        for tmp in staged.values():
            tmp.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='只检查不写入')
    args = parser.parse_args()
    try:
        apply(args.check)
    except Exception as exc:
        print('[full-pipeline install] 安装拒绝：' + str(exc), file=sys.stderr)
        raise SystemExit(1)
