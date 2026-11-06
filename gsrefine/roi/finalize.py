"""Module 4: extract and validate a Gaussian ROI from R0--R3 keep masks.

The output is a subset of the ORIGINAL binary Gaussian PLY: vertex records
are copied in input order without re-estimating any Gaussian attribute.
Poisson is only a ROI geometry reference. No SDF/internal artifact cleanup.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml

from .core import GaussianPLY, file_sha256, json_write

METHODS = ('R0', 'R1', 'R2', 'R3')


def _mask(path: Path, n: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path.name}. Run roi-study without --no-ablation to compute R1--R3.')
    mask = np.load(path, allow_pickle=False, mmap_mode='r')
    if mask.dtype != np.dtype('bool') or mask.shape != (n,):
        raise ValueError(f'{path.name} must be a bool array with {n} entries; got {mask.dtype}, {mask.shape}')
    return mask


def _resolve_gaussian(study_dir: Path, report: dict, config_path=None, gaussian_path=None):
    if gaussian_path is not None:
        return Path(gaussian_path).expanduser().resolve(), None
    if config_path is not None:
        cfg = yaml.safe_load(Path(config_path).read_text(encoding='utf8'))
        return Path(cfg['paths']['gaussian_ply']).expanduser().resolve(), cfg
    if report.get('gaussian_input_path'):
        return Path(report['gaussian_input_path']).expanduser().resolve(), None
    snapshot = study_dir / 'config_snapshot.json'
    if not snapshot.exists():
        raise FileNotFoundError('No original Gaussian source. Supply --config or --gaussian-ply.')
    cfg = json.loads(snapshot.read_text(encoding='utf8'))
    return Path(cfg['paths']['gaussian_ply']).expanduser().resolve(), cfg


def finalize(study_dir, *, method='R3', config_path=None, gaussian_path=None,
             output_path=None, report=None, source_ply=None,
             verify_input_hash=True) -> dict:
    """Export a selected, validated kept-Gaussian PLY using cached mask only.

    Refuses a modified source PLY, mismatched masks or a keep mask that drops R0.
    The .ply is written atomically; failed validation cannot publish a partial PLY.
    """
    study_dir = Path(study_dir).expanduser().resolve()
    if method not in METHODS:
        raise ValueError(f'method must be one of {METHODS}')
    if report is None:
        path = study_dir / 'study_report.json'
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}; first run roi-study.')
        report = json.loads(path.read_text(encoding='utf8'))
    if report.get('schema') != 'GSSceneRefine.roi_study.v1':
        raise ValueError('Invalid study_report schema; expected a roi-study result')
    source, config = _resolve_gaussian(study_dir, report, config_path, gaussian_path)
    if source_ply is None:
        ply = GaussianPLY(source)
    else:
        ply = source_ply
        if ply.path.resolve() != source:
            raise ValueError('Source PLY object/path mismatch')
    n = len(ply.arr)
    if n != int(report['gaussian_input_count']):
        raise ValueError('Source Gaussian count differs from the recorded roi-study input')
    expected_sha = report.get('input_sha256', {}).get('gaussian')
    if not expected_sha:
        raise ValueError('roi-study report does not provide source Gaussian SHA256')
    if verify_input_hash and file_sha256(source) != expected_sha:
        raise ValueError('Source Gaussian PLY SHA256 mismatch. Refusing to export stale masks.')
    roi_path = report.get('roi')
    roi_sha = report.get('input_sha256', {}).get('roi')
    if roi_path and roi_sha and Path(roi_path).exists() and file_sha256(roi_path) != roi_sha:
        raise ValueError('ROI JSON changed since roi-study. Refusing to export stale masks.')

    base_path = study_dir / 'keep_mask_R0.npy'
    selected_path = study_dir / f'keep_mask_{method}.npy'
    base = _mask(base_path, n)
    selected = base if method == 'R0' else _mask(selected_path, n)
    # New reports record exact mask bytes. Legacy reports are supported after
    # count/superset/added-mask checks so prior finished runs can be exported.
    recorded_mask_hashes = report.get('keep_mask_sha256', {})
    for key, path in (('R0', base_path), (method, selected_path)):
        if recorded_mask_hashes.get(key) and file_sha256(path) != recorded_mask_hashes[key]:
            raise ValueError(f'{key} keep mask SHA256 mismatch with study_report')
    if int(np.count_nonzero(base)) != int(report['R0_count']):
        raise ValueError('R0 mask count differs from study_report')
    if method != 'R0':
        if np.any(base & ~selected):
            raise ValueError(f'{method} drops existing R0 Gaussian(s); refusing to export')
        extra_mask = _mask(study_dir / f'added_mask_{method}.npy', n)
        if not np.array_equal(np.asarray(extra_mask), np.asarray(selected & ~base)):
            raise ValueError(f'added_mask_{method}.npy does not match {method} minus R0')
        matches = [x for x in report.get('ablation', []) if x.get('method') == method]
        if len(matches) != 1:
            raise ValueError(f'No unique {method} record in study_report; cannot validate mask')
        if int(np.count_nonzero(selected)) != int(matches[0]['gaussian_count']):
            raise ValueError(f'{method} mask count differs from study_report')
        if int(np.count_nonzero(extra_mask)) != int(matches[0]['newly_restored_gaussians']):
            raise ValueError(f'{method} added mask count differs from study_report')
    n_base = int(np.count_nonzero(base))
    n_keep = int(np.count_nonzero(selected))
    n_extra = n_keep - n_base

    destination = Path(output_path).expanduser().resolve() if output_path else study_dir / f'gaussians_final_{method}.ply'
    if destination == source:
        raise ValueError('Refusing to overwrite original Gaussian PLY')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + '.writing.tmp')
    try:
        written = ply.export(selected, staging)
        if written != n_keep:
            raise RuntimeError('Gaussian PLY writer returned inconsistent vertex count')
        check = GaussianPLY(staging)
        if check.vertex_count != n_keep or check.dtype != ply.dtype:
            raise RuntimeError('Exported Gaussian PLY has a different count or vertex property layout')
        if staging.stat().st_size != check.data_offset + n_keep * ply.dtype.itemsize:
            raise RuntimeError('Exported Gaussian PLY has incorrect byte length')
        # Verify input records are unchanged at deterministic samples without
        # reading an entire multi-GB PLY back into memory.
        if n_keep:
            ids = np.flatnonzero(selected)
            probes = np.unique(np.linspace(0, n_keep - 1, min(32, n_keep), dtype=int))
            if check.arr[probes].tobytes() != ply.arr[ids[probes]].tobytes():
                raise RuntimeError('Gaussian vertex attributes were modified in export')
        del check
        os.replace(staging, destination)
    finally:
        staging.unlink(missing_ok=True)

    # Row indices are sufficient to replay, audit, and invert the extraction.
    kept_ids = np.flatnonzero(selected)
    restored_ids = np.flatnonzero(selected & ~base)
    index_type = np.uint32 if n <= np.iinfo(np.uint32).max else np.uint64
    kept_index_path = study_dir / f'kept_indices_{method}.npy'
    restored_index_path = study_dir / f'restored_indices_{method}.npy'
    np.save(kept_index_path, kept_ids.astype(index_type))
    np.save(restored_index_path, restored_ids.astype(index_type))

    scope = (config or {}).get('source_model_scope')
    if scope is None:
        snapshot = study_dir / 'config_snapshot.json'
        if snapshot.is_file():
            scope = json.loads(snapshot.read_text(encoding='utf8')).get('source_model_scope')
    summary = {
        'schema': 'GSSceneRefine.gaussian_roi_final.v1',
        'method': method,
        'source_gaussian_ply': str(source),
        'source_gaussian_sha256': expected_sha,
        'source_model_scope': scope,
        'roi': roi_path,
        'roi_sha256': roi_sha,
        'input_gaussians': n,
        'R0_gaussians': n_base,
        'added_gaussians': n_extra,
        'final_gaussians': n_keep,
        'removed_from_source': n - n_keep,
        'retention_fraction': n_keep / max(n, 1),
        'output_ply': str(destination),
        'output_size_bytes': destination.stat().st_size,
        'keep_mask_path': str(selected_path),
        'keep_mask_sha256': file_sha256(selected_path),
        'kept_indices_path': str(kept_index_path),
        'restored_indices_path': str(restored_index_path),
        'output_sha256': file_sha256(destination),
        'validated': {
            'source_sha256_matches_study': True if verify_input_hash else None,
            'source_sha256_previously_recorded_in_current_study': bool(not verify_input_hash),
            'source_gaussian_count_matches_study': True,
            'all_R0_gaussians_retained': True,
            'restored_count_matches_mask': True,
            'original_vertex_property_layout_preserved': True,
            'original_gaussian_vertex_data_preserved': True,
            'original_gaussian_order_preserved': True,
        },
        'sdf_internal_cleanup_applied': False,
        'result_status': 'PROVISIONAL: verify visual ROI and boundary before scientific conclusions',
        'warning': ('AABB-only input cannot recover Gaussians already removed by AABB.'
                    if scope == 'aabb_baseline_only' else None),
    }
    json_write(study_dir / f'final_gaussian_report_{method}.json', summary)
    print(f'[roi-finalize] {method}: {n_keep:,} kept = {n_base:,} R0 + {n_extra:,} restored', flush=True)
    print(f'[roi-finalize] Gaussian scene: {destination}', flush=True)
    print('[roi-finalize] Geometry-only result; SDF internal cleanup NOT applied.', flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-dir', required=True, help='Directory containing study_report.json and keep_mask_R0--R3.npy')
    p.add_argument('--config', default=None, help='Original scene YAML; overrides snapshot Gaussian path')
    p.add_argument('--gaussian-ply', default=None, help='Explicit original binary GSPL PLY override')
    p.add_argument('--method', choices=METHODS, default='R3')
    p.add_argument('--output', default=None, help='Optional output PLY path; defaults to study dir')
    args = p.parse_args(argv)
    return finalize(args.study_dir, method=args.method, config_path=args.config,
                    gaussian_path=args.gaussian_ply, output_path=args.output)


if __name__ == '__main__':
    main()
