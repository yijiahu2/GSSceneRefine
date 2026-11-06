"""Reproducible top-only vs dual-orthographic Gaussian ROI ablation.

Creates a top-only ROI JSON from an already confirmed dual-view ROI. The
polygon and all projection coordinates are preserved; only the facade constraint
is omitted. Both conditions use exactly the same GSSceneRefine R3 pipeline.

This is a standalone research command; it never changes the ROI UI or R3 code.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from .core import GaussianPLY, Projection, ROI, json_write
from .study import run as study_run


def make_top_only_roi(roi_json: dict, *, up=None) -> dict:
    """Return a single-prism ROI with exactly the original top polygon."""
    if not isinstance(roi_json, dict) or not isinstance(roi_json.get('views'), list):
        raise ValueError('ROI JSON must contain views array')
    views = roi_json['views']
    if len(views) < 2:
        raise ValueError('A dual-view confirmed ROI is required for this ablation')
    # Never guess the top polygon from list position. Prefer the actual view
    # orientation when an up vector exists; otherwise require a unique 'top'.
    if up is None:
        up = roi_json.get('annotation_metadata', {}).get('up_vector')
    matched = []
    if up is not None:
        up = np.asarray(up, dtype=np.float64)
        if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-6:
            raise ValueError('Invalid up vector')
        up = up / np.linalg.norm(up)
        for i, view in enumerate(views):
            p = Projection.from_json(view)
            vn = np.cross(p.U[0], p.U[1])
            if abs(float(np.dot(vn, up))) >= 0.92:
                matched.append(i)
    else:
        matched = [i for i, v in enumerate(views)
                   if str(v.get('name', '')).strip().lower() in ('top', 'top_view', 'top-view')]
    if len(matched) != 1:
        raise ValueError(f'Cannot uniquely identify the top projection: {matched}')
    idx = matched[0]
    if up is None:
        # User must provide an orientation for study.py (which uses the up vector
        # to characterize Poisson groups). No silent cross-product sign guessing.
        raise ValueError('ROI lacks up_vector; pass --up x,y,z')
    result = json.loads(json.dumps(roi_json))
    result['views'] = [result['views'][idx]]
    result.setdefault('annotation_metadata', {})['up_vector'] = up.tolist()
    result['annotation_metadata']['view_count'] = 1
    result['ablation_note'] = ('Research comparison: original top polygon only. '
                               'No facade/height constraint; original ROI JSON is untouched.')
    return result


def summarize_masks(top_dir: Path, dual_dir: Path, output_dir: Path,
                    gaussian_ply: GaussianPLY, *, export_comparison_ply=True,
                    ground_truth_mask=None) -> dict:
    masks = {}
    for label, path in (('top', top_dir), ('dual', dual_dir)):
        for key in ('R0', 'R3'):
            arr = np.load(path / f'keep_mask_{key}.npy', allow_pickle=False)
            if arr.dtype != np.bool_ or arr.shape != (gaussian_ply.vertex_count,):
                raise ValueError(f'Invalid mask from {path}: {key}')
            masks[label, key] = arr
    a, b = masks['top', 'R0'], masks['dual', 'R0']
    if np.any(b & ~a):
        raise AssertionError('Dual-view R0 must be a subset of top-only R0')
    x, y = masks['top', 'R3'], masks['dual', 'R3']
    n = int(gaussian_ply.vertex_count)
    summary = {
        'experiment': 'top_only_vs_dual_view_same_top_polygon_same_R3',
        'input_gaussians': n,
        'top_R0': int(a.sum()), 'dual_R0': int(b.sum()),
        'R0_extra_top_only': int((a & ~b).sum()),
        'top_R3': int(x.sum()), 'dual_R3': int(y.sum()),
        'R3_common': int((x & y).sum()),
        'R3_only_top': int((x & ~y).sum()),
        'R3_only_dual': int((y & ~x).sum()),
        'R3_union': int((x | y).sum()),
        'R3_jaccard': float((x & y).sum() / max(1, int((x | y).sum()))),
        'top_minus_dual_R0_fraction_of_input': float((a & ~b).sum() / n),
        'top_minus_dual_R3_fraction_of_input': float((x & ~y).sum() / n),
        'note': ('Extra points are not automatically false positives. '
                 'Independent target reference is needed for precision/recall.'),
    }
    for label, folder in (('top', top_dir), ('dual', dual_dir)):
        report = json.loads((folder / 'study_report.json').read_text(encoding='utf-8'))
        rr = next((m for m in report.get('ablation', []) if m.get('method') == 'R3'), None)
        if rr:
            td = rr.get('two_stage_diagnostics', {})
            summary[label + '_R3_core_added'] = int(td.get('stage_A', {}).get('restored_core_gaussians', 0))
            summary[label + '_R3_safety_added'] = int(td.get('stage_B', {}).get('safety_added_gaussians', 0))
    if ground_truth_mask is not None:
        gt = np.load(ground_truth_mask, allow_pickle=False)
        if gt.shape != (n,) or gt.dtype != np.bool_:
            raise ValueError('Ground-truth mask must match original PLY row count and be bool')
        for label in ('top', 'dual'):
            m = masks[label, 'R3']
            tp = int((m & gt).sum())
            fp = int((m & ~gt).sum())
            fn = int((~m & gt).sum())
            summary[label + '_precision'] = tp / max(1, tp + fp)
            summary[label + '_recall'] = tp / max(1, tp + fn)
            summary[label + '_F1'] = 2 * tp / max(1, 2*tp + fp + fn)
        summary['ground_truth_mask'] = str(Path(ground_truth_mask).resolve())
    if export_comparison_ply:
        extras = {
            'R0_only_top_not_dual.ply': a & ~b,
            'R3_only_top_not_dual.ply': x & ~y,
            'R3_only_dual_not_top.ply': y & ~x,
        }
        for filename, m in extras.items():
            if m.any():
                gaussian_ply.export(m, output_dir / filename)
                print('[ablation] exported:', output_dir / filename,flush=True)
            else:
                (output_dir / filename).unlink(missing_ok=True)
    return summary


def run_ablation(config, roi, output_dir, *, gaussian_path=None,
                 up=None, ground_truth_mask=None, export_comparison_ply=True,
                 prepare_only=False):
    config = Path(config).resolve()
    roi = Path(roi).resolve()
    out = Path(output_dir).resolve()
    if not config.exists() or not roi.exists():
        raise FileNotFoundError('Config or ROI does not exist')
    if out == roi.parent or roi.is_relative_to(out):
        raise ValueError('Choose a separate ablation output folder; original ROI must not be overwritten')
    cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
    gaussian = Path(gaussian_path or cfg['paths']['gaussian_ply']).resolve()
    ply = GaussianPLY(gaussian)
    source = json.loads(roi.read_text(encoding='utf-8'))
    if up is not None:
        values = np.asarray([float(x) for x in up.split(',')], dtype=float)
        if values.shape != (3,):
            raise ValueError('--up expects x,y,z')
    else:
        values = None
    simplified = make_top_only_roi(source, up=values)
    dual = json.loads(json.dumps(source))
    if values is not None:
        dual.setdefault('annotation_metadata', {})['up_vector'] = (values/np.linalg.norm(values)).tolist()
    out.mkdir(parents=True, exist_ok=True)
    top_json = out/'roi_top_only.json'
    dual_json = out/'roi_dual_view.json'
    json_write(top_json, simplified)
    json_write(dual_json, dual)
    top_dir = out/'top_only'
    dual_dir = out/'dual_view'
    print('[ablation] Gaussian:', gaussian,flush=True)
    print('[ablation] ROI original:',roi,flush=True)
    print('[ablation] Save folder:',out,flush=True)
    print('[ablation] A: top polygon extruded without any facade restriction',flush=True)
    print('[ablation] B: original top/facade intersection',flush=True)
    # The preview phase computes the exact R0 masks without Poisson/R3 and
    # lets users inspect the top-only expansion before committing to 2 full runs.
    if prepare_only:
        xyz=ply.positions()
        for label, path in (('top_only',top_json), ('dual_view',dual_json)):
            m=np.all(ROI.from_file(path).distances(xyz) >= -1e-9,axis=1)
            print(f'[ablation preview] {label}: {int(m.sum()):,}/{len(xyz):,}',flush=True)
        return None
    for label, roi_path, folder in (('top_only',top_json,top_dir),('dual_view',dual_json,dual_dir)):
        print(f'\n[ablation] Running {label} R0+R3 ...',flush=True)
        study_run(config, roi_path, folder, gaussian_path=gaussian,
                  run_ablation=True, final_only=True, export_final=True)
        # Export R0 for side-by-side review without changing current study output
        # defaults or recompiling Poisson diagnostics.
        base = np.load(folder/'keep_mask_R0.npy',allow_pickle=False)
        ply.export(base,folder/'gaussians_R0.ply')
    summary=summarize_masks(top_dir,dual_dir,out,ply,
                            export_comparison_ply=export_comparison_ply,
                            ground_truth_mask=ground_truth_mask)
    summary.update({'config':str(config),'source_roi':str(roi),'gaussian_ply':str(gaussian),
                    'method':'R3 with unchanged scene configuration'})
    json_write(out/'ablation_summary.json',summary)
    with (out/'ablation_summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['metric','value'])
        for key,value in summary.items():writer.writerow([key,value])
    for key in ('top_R0','dual_R0','R0_extra_top_only','top_R3','dual_R3','R3_only_top','R3_only_dual','R3_jaccard'):
        print(f'[ablation] {key}: {summary[key]}',flush=True)
    print('[ablation] Compare:',top_dir/'gaussians_final_R3.ply', 'vs',dual_dir/'gaussians_final_R3.ply',flush=True)
    print('[ablation] report:',out/'ablation_summary.json',flush=True)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,help='Same resolved scene config for both conditions')
    parser.add_argument('--roi',required=True,help='One confirmed dual-view ROI JSON')
    parser.add_argument('--output-dir',required=True,help='Folder for this ablation comparison')
    parser.add_argument('--gaussian-ply',default=None,help='Override one shared input PLY for both conditions')
    parser.add_argument('--up',default=None,help='COLMAP up x,y,z only if missing from the ROI metadata')
    parser.add_argument('--ground-truth-mask',default=None,help='Optional independent bool .npy reference (same original PLY ordering)')
    parser.add_argument('--prepare-only',action='store_true',help='Only generate two ROI JSON and print R0 counts')
    parser.add_argument('--no-comparison-ply',action='store_true',help='Skip extra R3 difference models')
    a=parser.parse_args(argv)
    run_ablation(a.config,a.roi,a.output_dir,gaussian_path=a.gaussian_ply,up=a.up,
                 ground_truth_mask=a.ground_truth_mask,
                 export_comparison_ply=not a.no_comparison_ply,
                 prepare_only=a.prepare_only)


if __name__=='__main__':
    main()
