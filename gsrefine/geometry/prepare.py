"""Module 1 orchestrator: raw COLMAP SfM -> cleaned points -> Poisson reference.

Prepared geometry artifacts and effective runtime YAML live under the user
selected --output-dir/<scene>/geometry. Original reconstruction, Gaussian PLY and scene YAML are
never modified. Existing-reference mode is kept for previously validated scenes.

The module does NOT claim a generated Poisson mesh is semantically ground truth:
quality diagnostics are explicit, and the user may compare the mesh before use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from gsrefine.geometry.sfm_clean import load_colmap_points3d, filter_colmap
from gsrefine.geometry.poisson import make_poisson
from gsrefine.roi_init.settings import PROJECT_ROOT


class GeometryPreparationError(RuntimeError):
    pass


def _sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _write_json_atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf8') as stream:
            json.dump(content, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _write_yaml_atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf8') as stream:
            yaml.safe_dump(content, stream, allow_unicode=True, sort_keys=False)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _p(path, *, base=PROJECT_ROOT):
    if not path:
        raise GeometryPreparationError('Required Module-1 input path is not configured')
    value = Path(path).expanduser()
    return (value if value.is_absolute() else base / value).resolve()


def _status(output, phase, message, **values):
    data = dict(schema='GSSceneRefine.module1.status.v1', phase=phase,
                detail=message, timestamp_utc=datetime.now(timezone.utc).isoformat(), **values)
    _write_json_atomic(Path(output) / 'logs' / 'module1_status.json', data)
    print(f'[module1] {message}', flush=True)


def _options(data):
    block = data.get('module1', {}) or {}
    if not isinstance(block, dict):
        raise GeometryPreparationError('module1 must be a YAML mapping')
    mode = str(block.get('mode', 'reuse')).lower()
    if mode not in ('auto', 'reuse', 'prepared'):
        raise GeometryPreparationError('module1.mode must be auto or reuse')
    sfm = block.get('sfm_clean', {}) or {}
    poisson = block.get('poisson', {}) or {}
    for name, section in [('sfm_clean', sfm), ('poisson', poisson)]:
        if not isinstance(section, dict):
            raise GeometryPreparationError(f'module1.{name} must be a mapping')
    return mode, block, sfm, poisson


def _source_signature(source, sfm_args, poisson_args):
    material = {'colmap_points_sha256': _sha256_file(source),
                'sfm_clean': sfm_args, 'poisson': poisson_args,
                'pipeline_revision': 'module1-20261106-v1'}
    signature = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
    return signature, material


def _outputs_ok(directory, signature):
    try:
        manifest = json.loads((directory / 'metadata' / 'geometry_manifest.json').read_text(encoding='utf8'))
        if manifest.get('signature') != signature:
            return False
        for filename, props in manifest['outputs'].items():
            path = directory / filename
            stat = path.stat()
            if stat.st_size != props['bytes'] or stat.st_mtime_ns != props['mtime_ns']:
                return False
        return True
    except (FileNotFoundError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _check_geometry(sfm_path, mesh_path, gaussian_path, *, min_faces=100):
    """Conservative, deterministic validation without false 'semantic' certainty."""
    import trimesh
    from scipy.spatial import cKDTree
    from gsrefine.roi.core import GaussianPLY, load_sfm

    points, _, _, _ = load_sfm(sfm_path)
    mesh = trimesh.load(str(mesh_path), process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) < min_faces:
        raise GeometryPreparationError('Poisson output is not a sufficiently sized triangle mesh')
    verts = np.asarray(mesh.vertices)
    if not np.isfinite(verts).all() or not np.isfinite(points).all():
        raise GeometryPreparationError('Nonfinite SfM/Poisson vertices')
    if len(points) < 100:
        raise GeometryPreparationError('Too few cleaned SfM points')
    sfm_extent = np.maximum(np.ptp(points, axis=0), 1e-12)
    sfm_bounds = np.stack((points.min(axis=0), points.max(axis=0)))
    mesh_bounds = np.stack((verts.min(axis=0), verts.max(axis=0)))
    # An entirely disjoint mesh is a coordinate-system error, not a minor warning.
    if np.any(sfm_bounds[0] > mesh_bounds[1]) or np.any(mesh_bounds[0] > sfm_bounds[1]):
        raise GeometryPreparationError('SfM and Poisson bounds are disjoint')
    gp = GaussianPLY(gaussian_path)
    sample_count = min(int(gp.vertex_count), 4000)
    if sample_count < 1:
        raise GeometryPreparationError('Gaussian PLY contains no vertices')
    sampled = gp.selected_xyz(np.linspace(0, gp.vertex_count - 1, sample_count, dtype=np.int64))
    if not np.isfinite(sampled).all():
        raise GeometryPreparationError('Gaussian contains nonfinite centers')
    sfm_tree = cKDTree(points)
    dist = sfm_tree.query(sampled, k=1, workers=-1)[0]
    scale = max(float(np.linalg.norm(sfm_extent)), 1e-9)
    proximity = float(np.quantile(dist, 0.2) / scale)
    if proximity > 1.0:
        raise GeometryPreparationError('Gaussian/SfM likely use incompatible spatial frames: '
                                       f'nearest-neighbor distance P20 / SfM diagonal = {proximity:.3f}')
    notes = []
    if proximity > .1:
        notes.append('Gaussian/SfM spatial proximity is weak; verify COLMAP alignment before ROI extraction.')
    if gp.vertex_count > 20_000_000:
        notes.append('Very large Gaussian scene: GPU/browser and disk performance may be limiting.')
    # Triangle-distance QA (not face confidence scoring) is intentionally
    # independent of the downstream R3 classification and never edits meshes.
    try:
        import open3d as o3d
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))))
        scene.add_triangles(mesh_t)
        sparse = points[np.linspace(0, len(points)-1, min(5000, len(points)), dtype=int)]
        distances = scene.compute_distance(o3d.core.Tensor(sparse.astype(np.float32))).numpy()
        if not np.isfinite(distances).all():
            raise ValueError('Nonfinite point-to-Poisson distances')
        spacing = cKDTree(points).query(sparse, k=min(7, len(points)), workers=-1)[0][:, -1]
        sfm_fit_p95 = float(np.quantile(distances, .95))
        spacing_med = float(np.median(spacing))
        fit_ratio = sfm_fit_p95 / max(spacing_med, 1e-9)
        if fit_ratio > 15:
            notes.append(f'SfM-to-Poisson P95/median-spacing={fit_ratio:.2f}; inspect missing or artificial surfaces.')
        fit_report = dict(sfm_poisson_fit_p95=sfm_fit_p95,
                          sfm_spacing_median=spacing_med, sfm_fit_p95_over_spacing=fit_ratio,
                          fit_samples=int(len(sparse)), fit_method='exact_point_to_triangle_open3d')
    except ImportError:
        fit_report = {'fit_method':'unavailable_open3d', 'warning':'Install open3d for detailed SfM/Poisson fit QA'}
        notes.append('No Open3D: triangle-distance QA unavailable.')
    if not mesh.is_watertight:
        notes.append('Poisson surface is open; this is acceptable for ROI analysis but not a sealed solid.')
    return dict(sfm_points=int(len(points)), mesh_vertices=int(len(verts)),
                mesh_faces=int(len(mesh.faces)), gaussian_count=int(gp.vertex_count),
                bounds_sfm=sfm_bounds.tolist(), bounds_poisson=mesh_bounds.tolist(),
                sampled_gaussian_nn_sfm_p20_fraction=proximity,
                warnings=notes, **fit_report,
                scope='spatial/data integrity checks; not per-face semantic credibility')



def expand_colmap_inputs(sfm_dir=None, *, colmap_points=None, colmap_images=None):
    """Resolve a COLMAP sparse folder into the explicit inputs for Module 1.

    `--sfm-dir` is a convenience for the ordinary COLMAP sparse folder
    containing points3D.bin and images.bin. Do not silently combine paths
    from two different reconstructions.
    """
    if sfm_dir is None:
        return colmap_points, colmap_images
    if colmap_points is not None or colmap_images is not None:
        raise GeometryPreparationError(
            'Use either --sfm-dir OR --colmap-points/--colmap-images, not both')
    folder = Path(sfm_dir).expanduser().resolve(strict=True)
    if not folder.is_dir():
        raise GeometryPreparationError('--sfm-dir must be a COLMAP sparse directory')
    points, images = folder / 'points3D.bin', folder / 'images.bin'
    for item in (points, images):
        if not item.is_file():
            raise GeometryPreparationError(f'Missing required COLMAP file: {item}')
    return str(points), str(images)


def prepare_input_config(config, *, colmap_points=None, gaussian_ply=None,
                         colmap_images=None, scene=None, method=None,
                         final_output_root=None):
    """Prepare a scene/method-specific runtime config without editing the YAML.

    For a second Gaussian method on the *same COLMAP frame*, the original
    SfM source can be reused from the module-1 manifest. Geometry cache is
    shared per scene, while runtime YAML files are per Gaussian source.
    """
    import re
    config = Path(config).expanduser().resolve(strict=True)
    data = yaml.safe_load(config.read_text(encoding='utf8'))
    if not isinstance(data, dict) or not isinstance(data.get('paths'), dict):
        raise GeometryPreparationError('Invalid scene config: paths mapping required')
    sd = data.setdefault('scene', {})
    original_scene = sd.get('name')
    requested_scene = scene or original_scene
    requested_method = method or sd.get('method')
    tag = re.compile(r'^[a-zA-Z][a-zA-Z0-9_-]{0,50}$')
    if not tag.fullmatch(str(requested_scene)) or not tag.fullmatch(str(requested_method)):
        raise GeometryPreparationError('Scene and method require nonblank ASCII identifiers')
    if scene and original_scene != requested_scene and (not colmap_points or not colmap_images):
        raise GeometryPreparationError('A new scene requires its own --colmap-points and --colmap-images; never inherit another COLMAP frame')
    sd['name'], sd['method'] = requested_scene, requested_method
    block = data.setdefault('module1', {})
    if final_output_root is not None:
        from gsrefine.roi_init.output_layout import scene_dir
        output_root = Path(final_output_root).expanduser().resolve()
        out = scene_dir(output_root, requested_scene) / 'geometry'
    else:
        # Old programmatic callers may still use their configured directory.
        out = PROJECT_ROOT / 'runs' / 'geometry' / requested_scene
        if not scene or original_scene == requested_scene:
            out = _p(block.get('output_dir', f'runs/geometry/{requested_scene}'))
    source = colmap_points or block.get('colmap_points') or data['paths'].get('colmap_points')
    if not source:
        manifest = out / 'metadata' / 'geometry_manifest.json'
        if manifest.is_file():
            source = json.loads(manifest.read_text(encoding='utf8')).get('source')
    if not source:
        raise GeometryPreparationError('Raw points3D.bin required on the first run of a scene')
    source = Path(source).expanduser().resolve(strict=True)
    if source.name != 'points3D.bin':
        raise GeometryPreparationError('Raw COLMAP input must be points3D.bin')
    model = gaussian_ply or data['paths'].get('gaussian_ply')
    if not model:
        raise GeometryPreparationError('Full Gaussian PLY path required')
    model = Path(model).expanduser().resolve(strict=True)
    if model.suffix.lower() != '.ply':
        raise GeometryPreparationError('Full Gaussian model must be a PLY; export CKPT first')
    old_model = data['paths'].get('gaussian_ply')
    if (data.get('source_model_scope') == 'aabb_baseline_only' and old_model
            and _p(old_model) == model):
        raise GeometryPreparationError('Requested full Gaussian is the configured AABB baseline')
    data['paths']['gaussian_ply'] = str(model)
    data['paths']['colmap_points'] = str(source)
    data['source_model_scope'] = 'full_reconstruction'
    block.update(mode='auto', colmap_points=str(source), output_dir=str(out))
    init = data.setdefault('roi_init', {})
    init['gaussian_ply'] = str(model)
    orientation = init.setdefault('orientation', {})
    if scene and original_scene != requested_scene:
        orientation['frame_id'] = f'{requested_scene}_colmap_original'
        init['frame_id'] = f'{requested_scene}_colmap_original'
        orientation['mode'] = 'auto'
    if colmap_images is not None:
        image_path = Path(colmap_images).expanduser().resolve(strict=True)
        if image_path.name != 'images.bin':
            raise GeometryPreparationError('COLMAP camera metadata must be images.bin')
        orientation['colmap_images'] = str(image_path)
    if final_output_root is not None:
        final_root = Path(final_output_root).expanduser().resolve()
        if final_root.exists() and not final_root.is_dir():
            raise GeometryPreparationError(
                f'Final output path must be a directory: {final_root}')
        # This is the parent of per-confirmed-ROI output directories. It must
        # not be confused with module1.output_dir (geometry cache).
        data.setdefault('workflow', {})['output_root'] = str(final_root)
        # One authoritative output root. Do not write any run files into source.
        init['output_dir'] = str(final_root / requested_scene / 'roi')
        orientation['cache_dir'] = str(final_root / requested_scene / 'geometry' / 'orientation')
        data['paths']['output_dir'] = str(final_root / requested_scene / 'methods' / requested_method)
    # For the same scene, preserve legacy orientation settings. New scenes
    # explicitly require images.bin, enforced above.
    out.mkdir(parents=True, exist_ok=True)
    source_key = hashlib.sha256(str(model).encode('utf8')).hexdigest()[:10]
    runtime_config = out / 'runtime' / f'input_{requested_method}__{source_key}.yaml'
    _write_yaml_atomic(runtime_config, data)
    return runtime_config

def relocate_existing_reference_config(config, output_root):
    """Run an existing, read-only SfM/Poisson reference under a new output root.

    Legacy configs may have usable precomputed geometry but no raw
    ``points3D.bin``. Only their existing geometry files are read; runtime
    YAML, ROI records, orientation cache, logs and final outputs are relocated.
    Never use this for a new scene or to bypass original-frame verification.
    """
    import re
    config = Path(config).expanduser().resolve(strict=True)
    output_root = Path(output_root).expanduser().resolve()
    data = yaml.safe_load(config.read_text(encoding='utf8')) or {}
    paths = data.get('paths', {})
    scene_def = data.get('scene', {})
    scene, method = scene_def.get('name'), scene_def.get('method')
    pattern = re.compile(r'^[a-zA-Z][a-zA-Z0-9_-]{0,50}$')
    if not pattern.fullmatch(str(scene)) or not pattern.fullmatch(str(method)):
        raise GeometryPreparationError('Legacy reference must specify valid scene.name and scene.method')
    for label, key in [('SfM', 'clean_sfm_npz'), ('Poisson', 'poisson_raw'),
                       ('Gaussian', 'gaussian_ply')]:
        if not paths.get(key) or not _p(paths[key]).is_file():
            raise GeometryPreparationError(
                f'Missing existing {label} reference ({key}); provide original --sfm-dir for Module 1')
    # Legacy read-only reference is allowed for the SAME method.
    # Cross-method reuse still requires the full-scene scope check.
    target = output_root / scene / 'geometry' / 'runtime' / f'legacy_reference_{method}.yaml'
    data.setdefault('module1', {})['mode'] = 'reuse'
    data.setdefault('roi_init', {})['output_dir'] = str(output_root / scene / 'roi')
    data['roi_init'].setdefault('orientation', {})['cache_dir'] = str(
        output_root / scene / 'geometry' / 'orientation')
    data.setdefault('workflow', {})['output_root'] = str(output_root)
    data.setdefault('paths', {})['output_dir'] = str(output_root / scene / 'methods' / method)
    _write_yaml_atomic(target, data)
    return target


def prepare_scene(config):
    """Return an effective config path, preparing stage 1 if explicitly enabled.

    mode=reuse/default: retain existing verified reference paths; do not rebuild.
    mode=auto: input raw COLMAP points3D.bin, run deterministic SfM->Poisson,
    validate and atomically publish cached geometry and runtime config.
    mode=prepared: trust a previously generated effective config (subprocess).
    """
    config = Path(config).expanduser().resolve(strict=True)
    data = yaml.safe_load(config.read_text(encoding='utf8'))
    if not isinstance(data, dict) or not isinstance(data.get('paths'), dict):
        raise GeometryPreparationError('Config must contain a paths mapping')
    mode, block, sfm, poisson = _options(data)
    if mode == 'prepared':
        return config
    if mode == 'reuse':
        # Preserve the exact Songzi validated AABB+Poisson baseline until the
        # researcher changes module1.mode to auto and Gaussian input to FULL.
        return config
    scene = data.get('scene', {}).get('name')
    if not scene or '/' in str(scene) or '..' in str(scene):
        raise GeometryPreparationError('A valid scene.name is required')
    source = _p(block.get('colmap_points') or data['paths'].get('colmap_points'))
    gaussian = _p(data['paths'].get('gaussian_ply'))
    if not source.is_file():
        raise FileNotFoundError(f'Original COLMAP points3D.bin not found: {source}')
    if source.suffix != '.bin':
        raise GeometryPreparationError('Module-1 input must be COLMAP points3D.bin')
    if str(data.get('source_model_scope')) != 'full_reconstruction':
        raise GeometryPreparationError('Full-scene automatic mode requires source_model_scope: full_reconstruction; do not use the Task1 AABB baseline as full input')
    ui_gauss = (data.get('roi_init', {}) or {}).get('gaussian_ply')
    if ui_gauss and _p(ui_gauss) != gaussian:
        raise GeometryPreparationError('roi_init.gaussian_ply must match paths.gaussian_ply; Modules 2 and 3 must use identical original Gaussians')
    if not gaussian.is_file():
        raise FileNotFoundError(f'Gaussian PLY not found: {gaussian}')
    if gaussian.suffix.lower() != '.ply':
        raise GeometryPreparationError('The interactive module requires full Gaussian PLY; convert CKPT to PLY first')
    out_dir = _p(block.get('output_dir', f'runs/geometry/{scene}'))
    out_dir.mkdir(parents=True, exist_ok=True)
    sfm_opts = dict(min_track=int(sfm.get('min_track', 3)),
                    error_mad_mult=float(sfm.get('error_mad_mult', 5.0)),
                    sor_neighbors=int(sfm.get('sor_neighbors', 16)),
                    sor_std=float(sfm.get('sor_std', 2.5)),
                    do_sor=bool(sfm.get('do_sor', True)))
    p_opts = dict(depth=int(poisson.get('depth', 9)), normal_k=int(poisson.get('normal_k', 30)),
                  orientation_k=int(poisson.get('orientation_k', 30)),
                  radius_scale=float(poisson.get('radius_scale', 4.0)),
                  scale=float(poisson.get('scale', 1.1)),
                  density_quantile=float(poisson.get('density_quantile', 0.0)))
    if not 6 <= p_opts['depth'] <= 12:
        raise GeometryPreparationError('Poisson depth must be 6..12')
    signature, material = _source_signature(source, sfm_opts, p_opts)
    # The user may change reconstruction methods, never silently switch a
    # scene to a different COLMAP reference frame.
    old_manifest = out_dir / 'metadata' / 'geometry_manifest.json'
    if old_manifest.is_file():
        old = json.loads(old_manifest.read_text(encoding='utf8'))
        former = old.get('signature_material', {}).get('colmap_points_sha256')
        if former and former != material['colmap_points_sha256']:
            raise GeometryPreparationError('COLMAP points changed under one scene identity. '
                                           'Assign a different scene name or deliberately archive the old geometry cache.')
    clean_path = out_dir / 'cleaned_sfm.npz'
    mesh_path = out_dir / 'poisson_surface.ply'
    cached = _outputs_ok(out_dir, signature)
    if cached:
        _status(out_dir, 'cached', '模块一缓存有效：复用SfM净化与Poisson曲面', cached=True)
    else:
        _status(out_dir, 'cleaning', '模块一：读取原始COLMAP并净化SfM')
        with tempfile.TemporaryDirectory(prefix='.build-', dir=out_dir) as temp:
            stage = Path(temp)
            xyz, rgb, errors, tracks, ids = load_colmap_points3d(source)
            mask, clean_report = filter_colmap(xyz, errors, tracks, **sfm_opts)
            if int(mask.sum()) < 100:
                raise GeometryPreparationError('SfM filtering removed too many points (<100 remain)')
            clean = stage / clean_path.name
            np.savez_compressed(clean, xyz=xyz[mask], rgb=rgb[mask], error=errors[mask],
                                track_length=tracks[mask], point3D_ids=ids[mask],
                                source_row_indices=np.flatnonzero(mask))
            _write_json_atomic(stage / 'sfm_clean_report.json', {
                **clean_report, 'colmap_points': str(source), 'retained': int(mask.sum()),
                'removed': int((~mask).sum()), 'original_count': int(len(mask))})
            _status(out_dir, 'reconstructing', '模块一：法向一致性处理与Poisson曲面重建',
                    clean_points=int(mask.sum()))
            poisson_stage = stage / 'poisson'
            make_poisson(clean, poisson_stage, **p_opts)
            output_mesh = poisson_stage / '04_poisson_raw.ply'
            _status(out_dir, 'verifying', '模块一：核验SfM、Poisson及Gaussian坐标',
                    clean_points=int(mask.sum()))
            quality = _check_geometry(clean, output_mesh, gaussian)
            # Publish only complete validated results; never overwrite original
            # SfM/Poisson outside our dedicated runs/geometry directory.
            for old, new in [(clean, clean_path), (output_mesh, mesh_path),
                             (stage / 'sfm_clean_report.json', out_dir / 'metadata' / 'sfm_clean_report.json'),
                             (poisson_stage / 'reconstruction_report_poisson.json', out_dir / 'metadata' / 'poisson_reconstruction_report.json')]:
                if old.exists():
                    new.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(old, new)
            # Independent per-model QA is written below, even when geometry is cached.
            manifest = dict(schema='GSSceneRefine.module1.manifest.v1', signature=signature,
                            source=str(source), signature_material=material,
                            poisson_parameters=p_opts, sfm_clean_parameters=sfm_opts,
                            generated_utc=datetime.now(timezone.utc).isoformat(),
                            quality=quality,
                            outputs={p.name: {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                                     for p in (clean_path, mesh_path)})
            _write_json_atomic(out_dir / 'metadata' / 'geometry_manifest.json', manifest)
    # Even on a cache hit, check that a changed FULL Gaussian still aligns with
    # SfM and reference mesh. This does not change the scientific R3 algorithm.
    quality = _check_geometry(clean_path, mesh_path, gaussian)
    model_key = hashlib.sha256(str(gaussian).encode('utf8')).hexdigest()[:10]
    method = data.get('scene', {}).get('method', 'method')
    quality_path = out_dir / 'metadata' / 'quality' / f'{method}__{model_key}.json'
    _write_json_atomic(quality_path, quality)
    runtime = json.loads(json.dumps(data))
    runtime['paths']['clean_sfm_npz'] = str(clean_path)
    runtime['paths']['poisson_raw'] = str(mesh_path)
    runtime['paths']['gaussian_ply'] = str(gaussian)
    runtime['source_model_scope'] = 'full_reconstruction'
    runtime['module1'] = {**block, 'mode': 'prepared'}
    runtime['module1_runtime'] = {'status': 'completed', 'mode': 'auto',
                                  'cached': cached, 'manifest': str(out_dir/'metadata/geometry_manifest.json'),
                                  'quality_report': str(quality_path),
                                  'clean_sfm': str(clean_path), 'poisson_raw': str(mesh_path),
                                  'warnings': quality['warnings']}
    # A new Gaussian method must never overwrite the effective YAML of a
    # different method that may still be running or undergoing evaluation.
    effective = out_dir / 'runtime' / f'active_{method}__{model_key}.yaml'
    _write_yaml_atomic(effective, runtime)
    _status(out_dir, 'completed', '模块一已完成：已准备统一SfM/Poisson数据，进入ROI交互',
            cached=cached, geometry_report=str(quality_path),
            effective_config=str(effective))
    return effective


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--scene', help='Scene ID for an independent COLMAP reconstruction')
    parser.add_argument('--method', help='Method ID, e.g. vanilla, absgs, citygs')
    parser.add_argument('--colmap-points', help='Original COLMAP points3D.bin')
    parser.add_argument('--gaussian-ply', help='Original FULL Gaussian model PLY')
    parser.add_argument('--colmap-images', help='Matching COLMAP images.bin (optional when already configured)')
    parser.add_argument('--sfm-dir', help='COLMAP sparse folder containing points3D.bin and images.bin')
    parser.add_argument('--output-dir', required=True, help='All GSSceneRefine outputs, including geometry and cache')
    args = parser.parse_args(argv)
    points, images = expand_colmap_inputs(args.sfm_dir, colmap_points=args.colmap_points,
                                          colmap_images=args.colmap_images)
    config = (prepare_input_config(args.config, scene=args.scene, method=args.method,
                                 colmap_points=points, gaussian_ply=args.gaussian_ply,
                                 colmap_images=images, final_output_root=args.output_dir)
              if any((args.scene, args.method, args.colmap_points,
                      args.gaussian_ply, args.colmap_images, args.sfm_dir,
                      args.output_dir)) else args.config)
    result = prepare_scene(config)
    print(f'[module1] Effective config: {result}')


if __name__ == '__main__':
    main()
