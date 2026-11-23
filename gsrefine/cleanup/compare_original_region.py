"""Paired masks in native Stage-1 row order. Comparison is timed independently."""
from pathlib import Path
from time import perf_counter

import numpy as np

from gsrefine.cleanup.stage2_onepass import _atomic_json, _export_native, _read
from gsrefine.cleanup.stage2_runtime import file_digest, run_scope
from gsrefine.roi.core import GaussianPLY


def checked_masks(report, source, *, current):
    outputs = report['outputs']
    path = Path(outputs['final_mask'])
    if current and file_digest(path) != outputs['final_mask_sha256']:
        raise ValueError('Current mask digest mismatch.')
    with np.load(path, allow_pickle=False) as z:
        masks = {k:z[k] for k in ('remove','keep','original_parity_outside','protected_by_aux')}
        if current:
            masks['unverified_scope'] = z['unverified_scope']
    if any(v.dtype != np.bool_ or v.shape != (source.vertex_count,) for v in masks.values()):
        raise ValueError('Masks must be boolean arrays in Stage-1 row order.')
    remove, keep, candidate, protected = (
        masks[k] for k in ('remove','keep','original_parity_outside','protected_by_aux'))
    if (not np.array_equal(remove, ~keep) or np.any(protected & ~candidate)
            or not np.array_equal(remove, candidate & ~protected)):
        raise ValueError('Invalid mask accounting.')
    if current and (protected.any() or np.any(masks['unverified_scope'] & remove)):
        raise ValueError('Expected an unprotected experiment retaining unresolved rows.')
    counts = report['counts']
    if (int(keep.sum()), int(remove.sum()), int(candidate.sum()), int(protected.sum())) != (
            counts['final_retained'], counts['final_removed'], counts['outside_before_aux'],
            counts['protected_by_all_aux']):
        raise ValueError('Stage-2 report counts differ from its masks.')
    target = Path(outputs['final_supersplat_gaussian_ply'])
    result = GaussianPLY(target)
    if (file_digest(target) != outputs['final_ply_sha256'] or result.vertex_count != keep.sum()
            or result.dtype != source.dtype):
        raise ValueError('Published native PLY identity or format differs.')
    offset = 0
    for i in range(0, len(keep), 65536):
        rows = source.arr[i:i+65536][keep[i:i+65536]]
        if rows.tobytes() != result.arr[offset:offset+len(rows)].tobytes():
            raise ValueError('PLY rows differ from the source selection, including attributes.')
        offset += len(rows)
    return masks


def compare_masks(a, b):
    both = int(np.count_nonzero(a & b))
    new = int(np.count_nonzero(a & ~b))
    old = int(np.count_nonzero(b & ~a))
    union = both + new + old
    return dict(both_removed=both, A_only_removed=new, v3_only_removed=old,
                both_retained=int(len(a)-union), different=new+old,
                deletion_jaccard=both/union if union else 1.0)


@run_scope
def compare(output, baseline, command_seconds=None):
    start = perf_counter()
    output, baseline = Path(output), Path(baseline)
    curr = _read(output/'stage2_final_report.json')
    old = _read(baseline/'stage2_final_report.json')
    if curr['method']['spatial_classifier'] != 'original-region' or curr['method']['aux_enabled']:
        raise ValueError('Comparison expects original-region with --no-aux-protection.')
    if old['status'] != 'COMPLETED_EXPERIMENTAL_CLEAN_ROI_GAUSSIAN':
        raise ValueError('Baseline must have completed.')
    if old.get('method', {}).get('spatial_classifier') != 'adaptive-voxel':
        raise ValueError('Expected an adaptive-voxel baseline for the v3 comparison.')
    identity = ('stage1_manifest_sha256','original_roi_gaussian_sha256',
                'original_global_poisson_sha256','preseal_report_sha256')
    if any(curr['input'][k] != old['input'][k] for k in identity):
        raise ValueError('Cannot isolate classification: baseline and A input geometry differ.')
    if any(not np.isclose(curr['threshold'][k],old['threshold'][k],rtol=0,atol=1e-12)
           for k in ('delta','epsilon','sfm_p95','gaussian_trimmed_mean_radius')):
        raise ValueError('Baseline and A used different safety thresholds.')
    source_path = Path(curr['input']['original_roi_gaussian'])
    if file_digest(source_path) != curr['input']['original_roi_gaussian_sha256']:
        raise ValueError('Stage-1 source Gaussian PLY changed.')
    source = GaussianPLY(source_path)
    a = checked_masks(curr, source, current=True)
    b = checked_masks(old, source, current=False)
    result = dict(A_status=curr['status'], v3_report=str(baseline/'stage2_final_report.json'),
                  stage1_ply_sha256=curr['input']['original_roi_gaussian_sha256'],
                  native_retained_rows_verified=True, A_counts=curr['counts'],
                  A_vs_v3_before_aux=compare_masks(a['remove'],b['original_parity_outside']),
                  A_vs_v3_final=compare_masks(a['remove'],b['remove']),
                  v3_aux_protected=int(b['protected_by_aux'].sum()),
                  v3_aux_protected_removed_by_A=int((b['protected_by_aux'] & a['remove']).sum()),
                  v3_candidates_unverified_in_A=int((b['original_parity_outside'] & a['unverified_scope']).sum()),
                  A_stage2_seconds=curr['stage2_timing']['total_s'],
                  v3_stage2_seconds=(old.get('stage2_timing') or {}).get('total_s'),
                  A_command_to_validated_output_seconds=command_seconds,
                  A_used_cached_partition=curr['geometry'].get('partition_reused', True),
                  historical_A_and_B_audit_seconds=curr['geometry']['historical_ab_seconds'],
                  difference_previews={})
    for name, mask in [('A_only_removed',a['remove'] & ~b['remove']),
                       ('v3_only_removed',b['remove'] & ~a['remove'])]:
        if mask.any():
            path = output/'diagnostics'/(name+'.ply')
            _export_native(source, mask, path)
            result['difference_previews'][name] = str(path)
    result['comparison_seconds'] = perf_counter()-start
    _atomic_json(output/'comparison_v3.json',result)
    print('[original-region] A vs v3 (before auxiliary protection):',result['A_vs_v3_before_aux'])
    print('[original-region] A vs v3 (final):',result['A_vs_v3_final'])
    print('[original-region] Comparison:',output/'comparison_v3.json')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare previously published Stage-2 masks and native PLY rows.')
    parser.add_argument('output', type=Path, help='Completed original A, no-auxiliary run directory')
    parser.add_argument('baseline', type=Path, help='Completed v3 run directory')
    args = parser.parse_args()
    compare(args.output, args.baseline)
