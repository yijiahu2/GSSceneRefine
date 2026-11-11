"""Export a *full* native Gaussian ROI without auxiliary-Poisson protection.

This is a strict single-factor ablation derived from an ALREADY COMPLETED
GSSceneRefine Stage-2 one-pass run. It reuses the baseline's FINAL masks and
original Stage-1 Gaussian vertex records, and does not redo Poisson, SfM,
intersection, flood, parity or threshold calculation.

python -m gsrefine.cleanup.ablate_no_aux \\
  --baseline-report .../stage2_final_v1/stage2_final_report.json \\
  --output-dir .../cleanup/stage2_no_aux_v1 \\
  --export-diagnostics
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from gsrefine.cleanup.stage2_onepass import _atomic_json, _export_native, _read, _sha, _validate_output

SCHEMA = 'GSSceneRefine.stage2_ablation_no_aux.v1'


def _derive_no_aux_masks(data: dict, count: int, counts: dict):
    """Validate baseline row-aligned masks; return no-aux keep/remove/contrast.

    Avoid reconstructing a subset from the already cleaned baseline: auxiliary
    Gaussians must be deleted from the original Stage-1 Gaussian records.
    """
    required = {'keep', 'remove', 'protected_by_aux', 'original_parity_outside',
                'over_distance_threshold', 'delta'}
    missing = required - set(data)
    if missing:
        raise ValueError(f'Missing canonical Stage-2 masks: {sorted(missing)}')
    for name in required - {'delta'}:
        arr = np.asarray(data[name])
        if arr.dtype != np.dtype('bool') or arr.shape != (count,):
            raise ValueError(f'{name} mask must be bool with length {count}')
    old_keep = data['keep']; old_remove = data['remove']; protected = data['protected_by_aux']
    outside = data['original_parity_outside']; over = data['over_distance_threshold']
    if (np.any(outside & ~over) or np.any(protected & ~outside)
        or not np.array_equal(old_remove, outside & ~protected)
        or not np.array_equal(old_keep, ~old_remove)):
        raise ValueError('Baseline masks violate parity/threshold/auxiliary relations')
    if (int(counts['input_gaussians']) != count
        or int(counts['outside_before_aux']) != int(outside.sum())
        or int(counts['protected_by_all_aux']) != int(protected.sum())
        or int(counts['final_removed']) != int(old_remove.sum())
        or int(counts['final_retained']) != int(old_keep.sum())
        or int(counts['over_threshold']) != int(over.sum())):
        raise ValueError('Baseline report counts disagree with its own masks')
    d = float(np.asarray(data['delta']).item())
    if not np.isfinite(d) or d <= 0:
        raise ValueError('Invalid baseline threshold')
    new_remove = np.array(outside, dtype=bool, copy=True)
    new_keep = ~new_remove
    if not np.array_equal(new_remove, old_remove | protected):
        raise ValueError('No-aux group must equal baseline removal plus rescued auxiliaries')
    return new_keep, new_remove, protected, over, d


def run(*, baseline_report, output_dir, export_diagnostics=False):
    """One-factor ablation. Outputs a NEW complete SuperSplat-compatible ROI PLY."""
    from gsrefine.roi.core import GaussianPLY

    report_file = Path(baseline_report).expanduser().resolve(strict=True)
    baseline = _read(report_file)
    if (baseline.get('schema') != 'GSSceneRefine.stage2_onepass_r3_parity_aux_protection.v1'
        or baseline.get('status') != 'COMPLETED_EXPERIMENTAL_CLEAN_ROI_GAUSSIAN'):
        raise ValueError('Require a SUCCESSFUL native Stage-2 one-pass baseline report')
    source = Path(baseline['input']['original_roi_gaussian']).expanduser().resolve(strict=True)
    roi_run = source.parent.parent
    if source != (roi_run/'final/gaussian_roi.ply').resolve(strict=True):
        raise ValueError('Baseline source must be the unchanged Stage-1 final Gaussian PLY')
    if not report_file.is_relative_to(roi_run/'cleanup'):
        raise ValueError('Baseline report must belong to the same ROI run')
    if _sha(source) != baseline['input']['original_roi_gaussian_sha256']:
        raise ValueError('Original Stage-1 Gaussian content changed: abort')
    mask_path = Path(baseline['outputs']['final_mask']).expanduser().resolve(strict=True)
    if not mask_path.is_relative_to(report_file.parent):
        raise ValueError('Final masks are outside baseline result directory')
    if _sha(Path(baseline['input']['parity_report']).resolve(strict=True)) != baseline['input']['parity_report_sha256']:
        raise ValueError('Parity report changed after baseline: abort')
    out = _validate_output(output_dir, roi_run, source)
    model = GaussianPLY(source)
    with np.load(mask_path, allow_pickle=False) as z:
        # Copies avoid the NPZ loader closing during masked export.
        data = {k: np.array(z[k], copy=True) for k in z.files
                if k in ('keep', 'remove', 'protected_by_aux',
                         'original_parity_outside', 'over_distance_threshold', 'delta')}
    keep, remove, formerly_protected, over, delta = _derive_no_aux_masks(
        data, model.vertex_count, baseline['counts'])
    if not np.isclose(delta, baseline['threshold']['delta'], rtol=0, atol=1e-12):
        raise ValueError('Baseline threshold differs from classification mask')
    diagnostics = out/'diagnostics'
    out.mkdir(parents=True, exist_ok=True)
    diagnostics.mkdir(exist_ok=False)
    try:
        target = out/'gaussian_roi_clean.ply'
        written = _export_native(model, keep, target)
        if written != int(keep.sum()):
            raise RuntimeError('Gaussian export count differs from no-aux mask')
        if export_diagnostics:
            _export_native(model, remove, diagnostics/'no_aux_removed_gaussians.ply')
            _export_native(model, over & ~remove, diagnostics/'inside_over_threshold.ply')
        np.savez_compressed(
            diagnostics/'final_classification_masks.npz', keep=keep, remove=remove,
            protected_by_aux=np.zeros(model.vertex_count,dtype=bool),
            original_parity_outside=remove, over_distance_threshold=over,
            previously_protected_by_aux=formerly_protected, delta=np.float64(delta))
        record = {
            'schema': SCHEMA,
            'status': 'COMPLETED_CONTROLLED_NO_AUX_ABLATION',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'ablation_factor': 'auxiliary Poisson distance protection ONLY',
            'method': {'space_wall':baseline['method']['space_wall'],
                       'distance_surface':baseline['method']['distance_surface'],
                       'spatial_classifier':baseline['method']['spatial_classifier'],
                       'auxiliary_protection_enabled': False,
                       'final_delete_rule': 'distance_main>delta+epsilon AND Outside'},
            'threshold': baseline['threshold'],
            'baseline': {'report':str(report_file), 'report_sha256':_sha(report_file),
                         'mask':str(mask_path),'mask_sha256':_sha(mask_path),
                         'gaussian_source':str(source),
                         'gaussian_source_sha256':baseline['input']['original_roi_gaussian_sha256']},
            'counts': {'input_gaussians':int(model.vertex_count),
                       'over_threshold':int(over.sum()),
                       'outside_before_aux':int(remove.sum()),
                       'baseline_aux_protected_now_removed':int(formerly_protected.sum()),
                       'baseline_removed':int(baseline['counts']['final_removed']),
                       'final_removed':int(remove.sum()),
                       'final_retained':int(keep.sum())},
            'outputs':{'final_supersplat_gaussian_ply':str(target),
                       'final_ply_sha256':_sha(target),
                       'final_mask':str(diagnostics/'final_classification_masks.npz'),
                       'no_aux_removed_preview':str(diagnostics/'no_aux_removed_gaussians.ply') if export_diagnostics else None,
                       'inside_over_threshold_preview':str(diagnostics/'inside_over_threshold.ply') if export_diagnostics else None},
            'limitations':[
                'Only auxiliary protection is disabled; spatial classifier and threshold are unchanged.',
                'Visual residual artifacts may include outside-independent below-threshold Gaussians and above-threshold inside Gaussians.',
                'No empirical claim that the spatial partition is topologically watertight is made.'
            ]
        }
        _atomic_json(out/'stage2_final_report.json', record)
        print(f'[stage2-no-aux] SAVED: {target}',flush=True)
        print(f'[stage2-no-aux] input={model.vertex_count:,} removed={int(remove.sum()):,} '
              f'retained={int(keep.sum()):,} previously_protected_now_removed={int(formerly_protected.sum()):,}',flush=True)
        return record
    except Exception:
        if not (out/'stage2_final_report.json').is_file():
            (out/'gaussian_roi_clean.ply').unlink(missing_ok=True)
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-report',required=True,
                   help='Stage-2 one-pass WITH auxiliary protection: stage2_final_report.json')
    p.add_argument('--output-dir',required=True,
                   help='New EMPTY subdirectory of the same ROI run/cleanup')
    p.add_argument('--export-diagnostics',action='store_true',
                   help='Also emit no_aux_removed_gaussians.ply and inside_over_threshold.ply')
    a=p.parse_args(argv)
    run(baseline_report=a.baseline_report,output_dir=a.output_dir,
        export_diagnostics=a.export_diagnostics)


if __name__ == '__main__':
    main()
