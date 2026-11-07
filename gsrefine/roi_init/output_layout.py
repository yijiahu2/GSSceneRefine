"""Stable public outputs for GSSceneRefine; ephemeral study internals are private."""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from gsrefine.roi_init.results import write_atomic

LAYOUT_VERSION = 'GSSceneRefine.output.v3'
FINAL_REL = Path('final/gaussian_roi.ply')
META_REL = Path('metadata/workflow_result.json')
MANIFEST_REL = Path('metadata/manifest.json')
PROGRESS_REL = Path('logs/workflow_progress.json')
LOG_REL = Path('logs/processing.log')
ROI_ID = re.compile(r'(?:^|_)roi_(\d+)$', flags=re.I)


def scene_dir(output_root, scene):
    return Path(output_root).expanduser().resolve() / str(scene)


def run_dir(output_root, scene, method, roi):
    roi = Path(roi)
    match = ROI_ID.search(roi.stem)
    if match:
        identifier = 'roi_' + match.group(1)
    else:
        # Handles imported ROI files without numbering, stable across methods.
        from gsrefine.roi_init.results import canonical_views_sha
        payload = json.loads(roi.read_text(encoding='utf8'))
        identifier = 'roi_' + canonical_views_sha(payload)[:10]
    return scene_dir(output_root, scene) / 'methods' / str(method) / identifier


def result_path(out):
    out = Path(out)
    return out / META_REL if (out / META_REL).exists() or not (out/'workflow_result.json').exists() else out/'workflow_result.json'


def progress_path(out):
    out = Path(out)
    return out / PROGRESS_REL if (out / PROGRESS_REL).exists() or not (out/'workflow_progress.json').exists() else out/'workflow_progress.json'


def log_path(out):
    out = Path(out)
    return out / LOG_REL if (out / LOG_REL).exists() or not (out/'workflow.log').exists() else out/'workflow.log'


def _diagnostic_subdir(name):
    if name.endswith('.ply') or name == 'NO_BOUNDARY_TRACE.txt':
        return 'surfaces'
    if name.endswith(('.npy', '.npz')):
        return 'masks' if ('mask_' in name or 'indices_' in name) else 'geometry'
    return 'reports'


def _report_summary(report, roi, config, dest, final):
    stages = next((item.get('two_stage_diagnostics', {})
                   for item in report.get('ablation', []) if item.get('method') == 'R3'), {})
    return {
        'schema': 'GSSceneRefine.processing_report.v1',
        'status': 'completed',
        'source_gaussian': report.get('gaussian_input_path') or
                           ( __import__('yaml').safe_load(Path(config).read_text(encoding='utf8'))
                             or {}).get('paths', {}).get('gaussian_ply'),
        'source_roi': str(Path(roi).resolve()),
        'source_config': str(Path(config).resolve()),
        'final_gaussian': str(dest.resolve()),
        'original_gaussians': int(report.get('gaussian_input_count', 0)),
        'initial_roi_gaussians': int(report['R0_count']),
        'retained_gaussians': int(final['final_gaussians']),
        'restored_gaussians': int(final['added_gaussians']),
        'core_restored_gaussians': stages.get('stage_A', {}).get('restored_core_gaussians'),
        'safety_added_gaussians': stages.get('stage_B', {}).get('safety_added_gaussians'),
        'algorithm_stage': 'Poisson-guided ROI restoration and boundary protection',
        'not_cleaned': True,
    }


def publish(out, work, roi, config, report, *, keep_diagnostics=False):
    """Publish a verified Gaussian ROI, then clean all private work artifacts.

    Study runs in a dedicated temporary folder so internal and debug files do
    not leak into public output. The full study report stays available only
    under --keep-diagnostics; the compact report contains valid final paths.
    """
    out, work, roi, config = map(Path, (out, work, roi, config))
    final = dict(report.get('final_gaussian_scene') or {})
    model = Path(final['output_ply']).resolve(strict=True)
    if model.parent != work.resolve() or not model.stat().st_size:
        raise RuntimeError('Cannot publish an unverified or mismatched Gaussian PLY')
    dest = out / FINAL_REL
    if dest.exists():
        raise FileExistsError(f'Already completed: {dest}. Refusing to overwrite a result.')
    # Do not publish a large final PLY until all downstream dependencies are
    # known to exist. In particular, the next cleanup stage requires both
    # the trusted Poisson mesh and cleaned SfM, not only the Gaussian ROI.
    import yaml
    paths = (yaml.safe_load(config.read_text(encoding='utf8')) or {}).get('paths', {})
    downstream = {
        'gaussian_ply': str(dest.resolve()),
        'roi_json': str((out/'metadata/roi.json').resolve()),
        'poisson_mesh': str(Path(paths['poisson_raw']).resolve()),
        'clean_sfm_npz': str(Path(paths['clean_sfm_npz']).resolve()),
    }
    for key in ('poisson_mesh', 'clean_sfm_npz'):
        if not Path(downstream[key]).is_file():
            raise RuntimeError(f'Missing required downstream input before PLY publication: {key}')
    if not roi.is_file() or not config.is_file():
        raise RuntimeError('Cannot publish without source ROI JSON and effective config')
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta = out / 'metadata'
    meta.mkdir(parents=True, exist_ok=True)
    # Copy small records first; a final model is visible only after study
    # validation. The workflow_result + manifest remain the success markers.
    shutil.copy2(roi, meta/'roi.json')
    shutil.copy2(config, meta/'effective_scene.yaml')
    # Stage 2: exact Stage-1 R3 per-location d(x); no scalar fallback.
    from gsrefine.geometry.roi_sfm_local import publish_stage1_exact_roi_sfm
    _roi_sfm_path, _roi_sfm_report = publish_stage1_exact_roi_sfm(
        downstream=downstream, metadata_dir=meta, work_dir=work)
    if _roi_sfm_path is not None:
        downstream['sfm_roi_npz'] = str(_roi_sfm_path)
    # Stage 2 may reuse a certified surface mask already made in Stage 1.
    # This is a no-op if Stage 1 did not produce the optional handoff.
    from gsrefine.geometry.stage1_handoff import retain_stage1_surface_faces
    _stage2_faces = retain_stage1_surface_faces(work, meta, paths['poisson_raw'], roi)
    if _stage2_faces is not None:
        downstream['roi_surface_faces'] = str(_stage2_faces)
    if keep_diagnostics:
        for file in list(work.iterdir()):
            if file.is_file() and file != model:
                target = out/'diagnostics'/ _diagnostic_subdir(file.name)
                target.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file), str(target/file.name))
    os.replace(model, dest)
    final['output_ply'] = str(dest.resolve())
    final['output_size_bytes'] = dest.stat().st_size
    if not keep_diagnostics:
        for key in ('keep_mask_path', 'kept_indices_path', 'restored_indices_path'):
            final[key] = None
    summary = _report_summary(report, roi, config, dest, final)
    write_atomic(meta/'processing_report.json', summary)
    manifest = {
        'schema': LAYOUT_VERSION,
        'status': 'completed',
        'stage2_inputs': downstream,
        'source_gaussian': str(Path(report.get('gaussian_input_path') or
                          paths.get('gaussian_ply', '')).resolve()),
        'input_sha256': report['input_sha256'],
        'output_sha256': final.get('output_sha256'),
        'diagnostics_retained': bool(keep_diagnostics),
        'result_status': 'ROI_RESTORED_NOT_ANOMALY_CLEANED',
    }
    for key in ('gaussian_ply', 'roi_json', 'poisson_mesh', 'clean_sfm_npz'):
        if not Path(manifest['stage2_inputs'][key]).is_file():
            raise RuntimeError(f'Missing required downstream input: {key}')
    write_atomic(meta/'manifest.json', manifest)
    return dest.resolve(), manifest, summary
