"""Single official GSSceneRefine command for full, Stage-2 resume and repeat-from-ROI.

The from-roi mode never copies/modifies the authoritative annotation and never
reconstructs module-1 geometry. Every invocation reserves a unique experiment
root and runs the existing unmodified Stage-1 R3 implementation followed by
Stage-2 on that run's exact manifest. Scientific decisions are not made here.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'GSSceneRefine.repeat_from_roi.v1'
SAFE = re.compile(r'^[A-Za-z][A-Za-z0-9_-]{0,50}$')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _json_atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.'+path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def _existing_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f'{label} is not a file: {path}')
    return path


def _common_stage2(p: argparse.ArgumentParser) -> None:
    exterior = p.add_mutually_exclusive_group(required=True)
    exterior.add_argument('--accept-auto-exterior', action='store_true',
        help='PROVISIONAL outside seed; not independent scientific certification')
    exterior.add_argument('--exterior-seed', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
        help='Externally reviewed outside point in original COLMAP coordinates')
    p.add_argument('--stage2-name', default='stage2_original_region_v8')
    p.add_argument('--no-aux-protection', action='store_true',
                   help='Disable auxiliary protection for controlled ablation')
    p.add_argument('--audit-aux-ablation', action='store_true',
                   help='Evaluate non-decision-changing auxiliary-protection counterfactual')
    p.add_argument('--keep-classification-previews', action='store_true')
    p.add_argument('--no-aux-preview', action='store_true')
    p.add_argument('--reclassify', action='store_true')
    p.add_argument('--chunk-size', type=int, default=30000)
    p.add_argument('--nearest-k', type=int, default=8)
    p.add_argument('--min-path-agree', type=int, default=2)
    p.add_argument('--ray-batch', type=int, default=8000)


def _common_inputs(p: argparse.ArgumentParser) -> None:
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--scene', required=True)
    p.add_argument('--method', required=True)
    sfm = p.add_mutually_exclusive_group(required=True)
    sfm.add_argument('--sfm-dir', help='COLMAP sparse directory (points3D.bin/images.bin)')
    sfm.add_argument('--colmap-points', help='Original COLMAP points3D.bin')
    p.add_argument('--colmap-images', help='Original COLMAP images.bin')
    p.add_argument('--gaussian-ply', required=True, help='Complete ORIGINAL Gaussian PLY')
    p.add_argument('--output-dir', required=True, help='Common results root, not the ROI experiment folder')
    p.add_argument('--keep-diagnostics', action='store_true')


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='GSSceneRefine: full | stage2 | from-roi')
    options = p.add_subparsers(dest='mode', required=True)
    full = options.add_parser('full', help='Module 1 -> Edge ROI -> R3 -> Stage 2')
    _common_inputs(full)
    _common_stage2(full)
    full.add_argument('--no-auto-windows', action='store_true')
    s2 = options.add_parser('stage2', help='Resume Stage 2 from a verified Stage-1 manifest')
    s2.add_argument('--manifest', required=True)
    _common_stage2(s2)
    repeat = options.add_parser('from-roi',
        help='REPEAT confirmed ROI -> Modules 3/4 R3 -> Stage 2 in a NEW experiment folder')
    _common_inputs(repeat)
    _common_stage2(repeat)
    repeat.add_argument('--roi', required=True, help='Original confirmed immutable ROI JSON')
    repeat.add_argument('--reuse-scene-roi', action='store_true',
        help='Explicitly reuse ROI from another method in identical COLMAP coordinates')
    return p


def _run(command: list[str]) -> None:
    print('[GSSceneRefine official]', ' '.join(command), flush=True)
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def _s2_args(a) -> list[str]:
    result = ['--classifier', 'original-region', '--stage2-name', a.stage2_name,
              '--chunk-size', str(a.chunk_size), '--nearest-k', str(a.nearest_k),
              '--min-path-agree', str(a.min_path_agree), '--ray-batch', str(a.ray_batch)]
    if a.accept_auto_exterior:
        result.append('--accept-auto-exterior')
    else:
        result += ['--exterior-seed', *(str(x) for x in a.exterior_seed)]
    for name in ('no_aux_protection', 'audit_aux_ablation', 'keep_classification_previews',
                 'no_aux_preview', 'reclassify'):
        if getattr(a, name):
            result.append('--' + name.replace('_', '-'))
    return result


def _preflight_known_stage2_errors():
    """Reject the known v8.1/v8.2 audit integration errors before expensive work."""
    module = PROJECT_ROOT/'gsrefine/cleanup/local_region_pipeline.py'
    if not module.is_file():
        # Development fixtures may omit Stage-2; installed projects must include it.
        return
    tree = ast.parse(module.read_text(encoding='utf-8'), filename=str(module))
    functions = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == 'run']
    if len(functions) != 1:
        raise RuntimeError('Stage-2 original-region run() entry cannot be verified')
    for node in ast.walk(functions[0]):
        if (isinstance(node, ast.ImportFrom)
                and node.module == 'gsrefine.cleanup.stage2_runtime'
                and any(a.name in ('measured', 'timed') for a in node.names)):
            raise RuntimeError('Known v8.1 Stage-2 local-import shadowing detected. '
                               'Apply hotfix_measured_v8_1.py before an experiment.')
        if isinstance(node, (ast.With, ast.AsyncWith)):
            if any(isinstance(item.context_expr, ast.Call)
                   and isinstance(item.context_expr.func, ast.Name)
                   and item.context_expr.func.id == 'measured'
                   for item in node.items):
                raise RuntimeError('Known v8.2 Stage-2 audit timer bug detected. '
                                   'Apply hotfix_timing_v8_2.py before an experiment.')


def _source_args(a) -> list[str]:
    if not SAFE.fullmatch(a.scene) or not SAFE.fullmatch(a.method):
        raise ValueError('Scene and method must be safe ASCII labels')
    args = ['--config', str(_existing_file(a.config, 'config')),
            '--scene', a.scene, '--method', a.method,
            '--gaussian-ply', str(_existing_file(a.gaussian_ply, 'Gaussian PLY')),
            '--output-dir', str(Path(a.output_dir).expanduser().resolve())]
    if a.sfm_dir:
        directory = Path(a.sfm_dir).expanduser().resolve(strict=True)
        if not directory.is_dir() or not (directory/'points3D.bin').is_file():
            raise ValueError(f'COLMAP sparse directory missing points3D.bin: {directory}')
        args.extend(['--sfm-dir', str(directory)])
    if a.colmap_points:
        args.extend(['--colmap-points', str(_existing_file(a.colmap_points, 'COLMAP points3D'))])
    if a.colmap_images:
        args.extend(['--colmap-images', str(_existing_file(a.colmap_images, 'COLMAP images'))])
    if a.keep_diagnostics:
        args.append('--keep-diagnostics')
    return args


def _confirmed_roi(roi, a):
    """Bind polygon + orientation identity to the original Module-2 registry.

    SHA of the entire ROI is recorded for this invocation. If an earlier Stage-1
    run exists, its SHA is checked as an additional immutable anchor.
    """
    from gsrefine.roi_init.results import canonical_views_sha
    if not re.fullmatch(r'roi_\d{3,4}\.json', roi.name):
        raise ValueError('from-roi requires a numbered, confirmed roi_NNN.json')
    record_path = roi.parent/'manifest.json'
    if not record_path.is_file():
        raise ValueError('Confirmed ROI registry missing: ' + str(record_path))
    registry = json.loads(record_path.read_text(encoding='utf-8'))
    records = [r for r in registry.get('records', []) if r.get('filename') == roi.name]
    if len(records) != 1:
        raise ValueError('ROI must have exactly one entry in its original confirmation registry')
    record = records[0]
    payload = json.loads(roi.read_text(encoding='utf-8'))
    meta = payload.get('annotation_metadata') or {}
    if (not isinstance(payload.get('views'), list) or len(payload['views']) < 2
            or meta.get('orientation_verified') is not True):
        raise ValueError('ROI is not a verified dual-orthographic annotation')
    views_sha = canonical_views_sha(payload)
    if (record.get('view_sha256') != views_sha or meta.get('view_sha256') != views_sha
            or record.get('scene') != a.scene or meta.get('scene') != a.scene
            or record.get('frame_id') != meta.get('frame_id')
            or record.get('orientation_source_sha256') != meta.get('orientation_source_sha256')
            or record.get('source_ply') != meta.get('source_ply')):
        raise ValueError('Confirmed ROI or registry was changed; refusing edited annotation')
    if not a.reuse_scene_roi and (record.get('method') != a.method or
                                 Path(record['source_ply']).resolve() != Path(a.gaussian_ply).expanduser().resolve()):
        raise ValueError('ROI source/method differs; use --reuse-scene-roi ONLY for the same COLMAP frame')
    original_root = roi.parent.parent.parent
    from gsrefine.roi_init.output_layout import run_dir
    previous = run_dir(original_root, record['scene'], record['method'], roi) / 'metadata/workflow_result.json'
    sha = _sha(roi)
    if previous.is_file():
        old = json.loads(previous.read_text(encoding='utf-8'))
        if old.get('roi_sha256') != sha or Path(old.get('roi','')).resolve() != roi:
            raise ValueError('The original ROI JSON differs from its completed Stage-1 SHA256')
    return dict(sha256=sha, views_sha256=views_sha, registry=str(record_path),
                source_roi=str(roi),
                source_method=record['method'], original_root=str(original_root))


def _verify_module1_cache(a, root, roi):
    """Read-only validation; from-roi never calls Module-1 reconstruction.

    The same immutable Module-1 geometry may be shared by many repeat runs.
    For a changed --output-dir, also inspect the original ROI scene's cache.
    """
    import yaml
    from gsrefine.geometry.prepare import _source_signature, _outputs_ok, _options
    source_config = _existing_file(a.config, 'config')
    raw = yaml.safe_load(source_config.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('Invalid YAML source configuration')
    _, _, sfm, poisson = _options(raw)
    pts = (Path(a.sfm_dir).expanduser().resolve(strict=True)/'points3D.bin'
           if a.sfm_dir else _existing_file(a.colmap_points, 'points3D.bin'))
    imgs = (_existing_file(a.colmap_images, 'images.bin') if a.colmap_images else
            (Path(a.sfm_dir).expanduser().resolve(strict=True)/'images.bin' if a.sfm_dir else None))
    if not pts.is_file() or imgs is None or not imgs.is_file():
        raise ValueError('from-roi requires the original points3D.bin and images.bin')
    sfm_opts = dict(min_track=int(sfm.get('min_track', 3)),
        error_mad_mult=float(sfm.get('error_mad_mult', 5.0)),
        sor_neighbors=int(sfm.get('sor_neighbors', 16)),
        sor_std=float(sfm.get('sor_std', 2.5)),
        do_sor=bool(sfm.get('do_sor', True)))
    p_opts = dict(depth=int(poisson.get('depth', 9)),
        normal_k=int(poisson.get('normal_k', 30)),
        orientation_k=int(poisson.get('orientation_k', 30)),
        radius_scale=float(poisson.get('radius_scale', 4.0)),
        scale=float(poisson.get('scale', 1.1)),
        density_quantile=float(poisson.get('density_quantile', 0.0)))
    signature, source_material = _source_signature(pts, sfm_opts, p_opts)
    candidates = [Path(root).expanduser().resolve()/a.scene/'geometry',
                  roi.parent.parent/'geometry']
    used = None
    for geometry_root in dict.fromkeys(candidates):
        mf = geometry_root/'metadata/geometry_manifest.json'
        if not mf.is_file() or not _outputs_ok(geometry_root, signature):
            continue
        manifest = json.loads(mf.read_text(encoding='utf-8'))
        if (manifest.get('signature') == signature
                and (manifest.get('signature_material') or {}).get('colmap_points_sha256')
                    == source_material['colmap_points_sha256']
                and Path(manifest.get('source','')).resolve() == pts):
            used = geometry_root
            break
    if used is None:
        raise ValueError('Module-1 cache is missing, altered or parameter-mismatched; '
                         'from-roi will NOT rerun SfM cleaning/Poisson. Select its original '
                         'output root and config, or use pipeline full first.')
    print('[GSSceneRefine from-roi] Verified existing Module-1 cache; no SfM/Poisson reconstruction', flush=True)
    return dict(source_config=source_config, original_points=pts, original_images=imgs,
        geometry_root=used, sfm=used/'cleaned_sfm.npz', poisson=used/'poisson_surface.ply',
        geometry_signature=signature, source_material=source_material)


def _write_repeat_config(a, roi, repeat_root, geometry):
    """Effective YAML belongs to this attempt; shared caches are NEVER rewritten."""
    import yaml
    source = yaml.safe_load(Path(geometry['source_config']).read_text(encoding='utf-8'))
    if not isinstance(source, dict):
        raise ValueError('Invalid YAML source config')
    source.setdefault('scene', {}).update(name=a.scene, method=a.method)
    source['source_model_scope'] = 'full_reconstruction'
    source.setdefault('paths', {}).update(
        gaussian_ply=str(Path(a.gaussian_ply).expanduser().resolve(strict=True)),
        colmap_points=str(geometry['original_points']),
        clean_sfm_npz=str(geometry['sfm']),
        poisson_raw=str(geometry['poisson']),
        output_dir=str(repeat_root/a.scene/'methods'/a.method))
    block = source.setdefault('module1', {})
    block.update(mode='prepared', colmap_points=str(geometry['original_points']),
                 output_dir=str(geometry['geometry_root']))
    model_key = hashlib.sha256(str(Path(a.gaussian_ply).expanduser().resolve()).encode('utf-8')).hexdigest()[:10]
    quality_path = (geometry['geometry_root']/'metadata/quality'/f'{a.method}__{model_key}.json')
    warnings = []
    if quality_path.is_file():
        cached_quality = json.loads(quality_path.read_text(encoding='utf-8'))
        warnings = cached_quality.get('warnings', [])
    else:
        # A pre-existing Module-1 manifest may predate per-method quality reports.
        cache_manifest = json.loads((geometry['geometry_root']/'metadata/geometry_manifest.json')
                                    .read_text(encoding='utf-8'))
        warnings = (cache_manifest.get('quality') or {}).get('warnings', [])
    source['module1_runtime'] = dict(status='completed', mode='auto',
        cached=True, manifest=str(geometry['geometry_root']/'metadata/geometry_manifest.json'),
        quality_report=str(quality_path) if quality_path.is_file() else None,
        clean_sfm=str(geometry['sfm']), poisson_raw=str(geometry['poisson']),
        warnings=warnings, signature=geometry['geometry_signature'])
    init = source.setdefault('roi_init', {})
    init['gaussian_ply'] = str(Path(a.gaussian_ply).expanduser().resolve(strict=True))
    init['output_dir'] = str(roi.parent)
    orientation = init.setdefault('orientation', {})
    original_meta = json.loads(roi.read_text(encoding='utf-8'))['annotation_metadata']
    orientation['frame_id'] = original_meta['frame_id']
    orientation['cache_dir'] = str(roi.parent.parent/'geometry/orientation')
    orientation['colmap_images'] = str(geometry['original_images'])
    source.setdefault('workflow', {})['output_root'] = str(repeat_root)
    effective = repeat_root/'config/effective_scene.yaml'
    effective.parent.mkdir(parents=True, exist_ok=True)
    # The repeat root is a NEW atomic reservation; this cannot overwrite an older YAML.
    with effective.open('x', encoding='utf-8') as stream:
        yaml.safe_dump(source, stream, allow_unicode=True, sort_keys=False)
    return effective


def _reserve_repeat(root, a, roi):
    """OS-atomic unique reservation: failed runs also remain reserved forever."""
    dest = Path(root)/'experiments'/a.scene/a.method/roi.stem
    dest.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1000000):
        candidate = dest / f'repeat_{index:04d}'
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError('No unused experiment slot is available: ' + str(dest))


def _repeat_from_roi(a):
    from gsrefine.roi_init.workflow import run_once
    from gsrefine.roi_init.output_layout import run_dir
    root = Path(a.output_dir).expanduser().resolve()
    roi = _existing_file(a.roi, 'Confirmed hand-drawn ROI JSON')
    provenance = _confirmed_roi(roi, a)
    # The previous command validates source paths without writing into Stage-1.
    _source_args(a)
    geometry = _verify_module1_cache(a, root, roi)
    repeat_root = _reserve_repeat(root, a, roi)
    result_dir = run_dir(repeat_root, a.scene, a.method, roi)
    control = repeat_root/'experiment.json'
    metadata = dict(schema=SCHEMA, status='RESERVED', created_utc=_utc(),
        mode='from-roi', repeat_root=str(repeat_root),
        result_dir=str(result_dir),
        roi=provenance, full_gaussian=str(Path(a.gaussian_ply).expanduser().resolve()),
        original_colmap_points=(str(Path(a.sfm_dir).resolve()/'points3D.bin')
                                if a.sfm_dir else str(Path(a.colmap_points).resolve())),
        effective_scene_config=str(repeat_root/'config/effective_scene.yaml'),
        reuse_scene_roi=bool(a.reuse_scene_roi),
        algorithm='unchanged R3 + original-global Poisson side + R3-local distance',
        module1='verified cache; no reconstruction',
        geometry_reference=dict(root=str(geometry['geometry_root']),
                                signature=geometry['geometry_signature'],
                                cleaned_sfm=str(geometry['sfm']),
                                original_poisson=str(geometry['poisson'])))
    _json_atomic(control, metadata)
    started = perf_counter()
    try:
        prepared_config = _write_repeat_config(a, roi, repeat_root, geometry)
        print(f'[GSSceneRefine from-roi] Experiment: {repeat_root}', flush=True)
        print(f'[GSSceneRefine from-roi] Original immutable ROI: {roi}', flush=True)
        metadata['status'] = 'STAGE1_RUNNING'
        _json_atomic(control, metadata)
        stage1 = run_once(prepared_config, roi, output_dir=repeat_root,
            reuse_scene_roi=a.reuse_scene_roi, keep_diagnostics=a.keep_diagnostics)
        if _sha(roi) != provenance['sha256']:
            raise ValueError('Original confirmed ROI changed while Stage-1 was running')
        after = _confirmed_roi(roi, a)
        if after['views_sha256'] != provenance['views_sha256']:
            raise ValueError('Original confirmation record changed during Stage-1')
        manifest = _existing_file(stage1['manifest'], 'New Stage-1 manifest')
        if manifest != result_dir/'metadata/manifest.json':
            raise RuntimeError('Stage-1 produced a manifest outside this reserved experiment')
        data = json.loads(manifest.read_text(encoding='utf-8'))
        if (data.get('status') != 'completed'
                or Path(data['stage2_inputs']['roi_json']).resolve() != roi
                or data['input_sha256']['roi'] != provenance['sha256']):
            raise RuntimeError('New Stage-1 manifest did not bind the original confirmed ROI')
        stage1_wall = perf_counter()-started
        metadata.update(status='STAGE1_COMPLETED', stage1_manifest=str(manifest),
                        stage1_manifest_sha256=_sha(manifest), stage1_result=stage1,
                        stage1_wall_s=stage1_wall)
        _json_atomic(control, metadata)
        command = ([sys.executable, str(PROJECT_ROOT/'run.py'), 'full-pipeline',
                    '--resume-manifest', str(manifest)] + _s2_args(a))
        metadata['status'] = 'STAGE2_RUNNING'
        _json_atomic(control, metadata)
        _run(command)
        status_file = result_dir/'metadata/full_pipeline_status.json'
        if not status_file.is_file():
            raise RuntimeError('Stage-2 process returned without its authoritative terminal status')
        status_report = json.loads(status_file.read_text(encoding='utf-8'))
        terminal = status_report.get('status')
        if terminal not in ('COMPLETED', 'PARTIAL_VERIFIED_ROI_CLEANUP', 'NO_VERIFIED_CLEANUP_SCOPE'):
            raise RuntimeError('Unexpected Stage-2 terminal status: '+str(terminal))
        total_wall = perf_counter()-started
        timing_path = result_dir/'metadata/stage1_timing.json'
        stage1_timing = json.loads(timing_path.read_text(encoding='utf-8')) if timing_path.is_file() else {}
        s2_report_file = status_report.get('stage2_report')
        s2_data = (json.loads(Path(s2_report_file).read_text(encoding='utf-8'))
                   if s2_report_file and Path(s2_report_file).is_file() else {})
        metadata.update(status=terminal, finished_utc=_utc(),
                        stage2_pipeline_status=str(status_file),
                        stage2_output_dir=status_report.get('stage2_output_dir'),
                        stage2_report=s2_report_file,
                        elapsed_s=dict(total_wall=total_wall, stage1_wall=stage1_wall,
                                       stage2_subprocess_wall=total_wall-stage1_wall,
                                       stage1_phases=stage1_timing.get('phases_s',{}),
                                       stage2_timers=s2_data.get('stage2_timing',{})),
                        final_counts=s2_data.get('counts',{}))
        _json_atomic(control, metadata)
        print(f'[GSSceneRefine from-roi] COMPLETE: {control}', flush=True)
        return manifest
    except BaseException as error:
        metadata.update(status='FAILED', failed_utc=_utc(),
                        failure_type=type(error).__name__, failure=str(error))
        _json_atomic(control, metadata)
        print(f'[GSSceneRefine from-roi] Preserved failed experiment: {repeat_root}', flush=True)
        raise


def run(a):
    _preflight_known_stage2_errors()
    runner = [sys.executable, str(PROJECT_ROOT/'run.py')]
    if a.mode == 'stage2':
        manifest = _existing_file(a.manifest, 'Stage-1 manifest')
        data = json.loads(manifest.read_text(encoding='utf-8'))
        if data.get('status') != 'completed':
            raise ValueError('Stage-1 manifest is not completed; refuse Stage-2 resume')
        _run(runner + ['full-pipeline', '--resume-manifest', str(manifest)] + _s2_args(a))
        return manifest
    if a.mode == 'full':
        source = _source_args(a)
        flags = ['--no-auto-windows'] if a.no_auto_windows else []
        _run(runner + ['full-pipeline'] + source + flags + _s2_args(a))
        return None
    if a.mode == 'from-roi':
        return _repeat_from_roi(a)
    raise ValueError('Unsupported mode')


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    try:
        run(a)
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError, RuntimeError) as error:
        print(f'[GSSceneRefine official] FAILED: {error}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
