"""Compare two completed Stage-2 onepass runs on the same input and cache mode."""
import argparse
import filecmp
import json
from pathlib import Path


def read_run(directory):
    report = json.loads((directory / 'stage2_final_report.json').read_text(encoding='utf-8'))
    times = report['stage2_timing']
    steps = times['steps']
    inputs = report['input']
    identity = (inputs['stage1_manifest_sha256'], inputs['original_roi_gaussian_sha256'],
                inputs['original_global_poisson_sha256'], inputs['preseal_report_sha256'],
                report['method'], report['provenance']['status'], 'audit_run' in steps,
                steps['export_native']['calls'])
    if 'run_roi_barrier_parity' not in steps:
        raise ValueError(f'{directory} 复用了分类结果。分类耗时对照需使用 --reclassify。')
    perf = report['classification_performance']
    rows = {
        '阶段二合计': times['total_s'],
        '几何准备与校验': sum(steps[k]['inclusive_s'] for k in ('prepare_required_geometry', 'prepare_geometry')),
        '候选空间分类合计': perf['total_seconds'],
        '  其中代表点短射线': perf['representative_seconds'],
        '  其中逐点回退': perf['fallback_seconds'],
    }
    return identity, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old', type=Path)
    parser.add_argument('new', type=Path)
    args = parser.parse_args()
    old_id, old = read_run(args.old)
    new_id, new = read_run(args.new)
    if old_id != new_id:
        parser.exit(1, '输入、几何准备、审计复用、方法或导出次数不同，请核对对照条件。\n')
    print('项目\t旧版(秒)\t新版(秒)\t节省(秒)\t耗时下降')
    for name, before in old.items():
        after = new[name]
        saved = before - after
        change = f'{100 * saved / before:.1f}%' if before else '—'
        print(f'{name}\t{before:.2f}\t{after:.2f}\t{saved:.2f}\t{change}')
    if not filecmp.cmp(args.old/'gaussian_roi_clean.ply', args.new/'gaussian_roi_clean.ply', shallow=False):
        parser.exit(1, '两次最终 PLY 内容不同。\n')
    print('两次最终 PLY 逐字节一致。')


if __name__ == '__main__':
    main()
