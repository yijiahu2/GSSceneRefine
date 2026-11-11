"""Exact post-restoration R3 core/safety field for cleaned local COLMAP SfM.

The same CoreImplicitField, boundary projection and trimmed kNN used for
Gaussian R3 are queried at CLEAN SfM positions.  This module DOES NOT change
R3 Gaussian selection, Stage-2 Poisson, sealing or SDF pruning formula.

Stage-1 writer is optional for mocked/historical workflows; a real R3 call
must write the v3 handoff. Historical ROI runs can reconstruct ONLY the
required R3 geometric support from their immutable source inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SCHEMA = 'GSSceneRefine.r3_sfm_handoff.v3'
SFM_SCHEMA = 'GSSceneRefine.roi_sfm.v3'
SEMANTICS = 'original_R3_repaired_core_field_and_kNN_5pct_trim_radius_at_clean_SfM_points'
WORK_NAME = 'stage2_ready_r3_sfm_v3.npz'
PUBLISHED_NAME = 'stage1_r3_local_safety_v3.npz'
SELECTION = 'exact_post_restoration_r3_safety_1x_2x'
EPS = 1e-6


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _npz_atomic(path, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + path.stem + '_', suffix='.npz', dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temp, **fields)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def _json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + path.stem + '_', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf8') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def save_r3_sfm_handoff(*, work_dir, clean_sfm_npz, roi_json, gaussian_ply,
                        poisson_mesh, core_field, safety_config,
                        gaussian_positions, major_axes, chunk_size=16384):
    """Query the actual in-memory R3 field at ALL cleaned SfM positions.

    For Stage-B candidate Gaussians, this calls precisely the same functions
    and parameters as the existing R3 selector. Interior points also receive
    local radii for a complete auditable per-position record.
    """
    from gsrefine.roi.two_stage_sdf import _project_to_boundary, trimmed_knn_mean
    clean_sfm_npz, roi_json, gaussian_ply, poisson_mesh = map(
        lambda x: Path(x).resolve(strict=True),
        (clean_sfm_npz, roi_json, gaussian_ply, poisson_mesh))
    with np.load(clean_sfm_npz, allow_pickle=False) as sfm:
        xyz = np.asarray(sfm['xyz'], dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Invalid cleaned SfM xyz; cannot export exact R3 support')
    all_xyz = np.asarray(gaussian_positions, dtype=np.float64)
    axes = np.asarray(major_axes, dtype=np.float64)
    if (all_xyz.ndim != 2 or all_xyz.shape[1] != 3 or axes.shape != (len(all_xyz),)
            or not np.isfinite(all_xyz).all() or not np.isfinite(axes).all()
            or np.any(axes <= 0)):
        raise ValueError('R3 source Gaussian centers/axes are invalid')
    safety_config.validate(len(all_xyz))
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    tree = cKDTree(all_xyz)
    field_distance = np.empty(len(xyz), dtype=np.float64)
    local_r = np.empty(len(xyz), dtype=np.float64)
    projection_error = np.empty(len(xyz), dtype=np.float64)
    for a in range(0, len(xyz), chunk_size):
        b = min(a + chunk_size, len(xyz))
        pts = xyz[a:b]
        dist = np.asarray(core_field.evaluate(pts), dtype=np.float64)
        boundary, residual = _project_to_boundary(core_field, pts, safety_config)
        radii = trimmed_knn_mean(tree, axes, boundary,
                                 k=safety_config.k_neighbors,
                                 trim_fraction=safety_config.trim_fraction,
                                 batch_size=safety_config.batch_size)
        if (dist.shape != (len(pts),) or radii.shape != (len(pts),)
                or not np.isfinite(dist).all() or not np.isfinite(radii).all()
                or np.any(radii <= 0) or not np.isfinite(residual).all()):
            raise ValueError('R3 spatial field query contains invalid values')
        field_distance[a:b] = dist
        local_r[a:b] = radii * safety_config.safety_factor
        projection_error[a:b] = residual
    # Query all points, not just the initial ROI boundary. These are finite,
    # auditable inputs for the SAME R3 rule with 1x or 2x radius.
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    target = work_dir / WORK_NAME
    if target.exists():
        raise FileExistsError('Already have a R3 SfM handoff in this work directory: ' + str(target))
    _npz_atomic(target, schema=np.array(SCHEMA), semantics=np.array(SEMANTICS),
                cleaned_sfm_sha256=np.array(sha256(clean_sfm_npz)),
                roi_sha256=np.array(sha256(roi_json)),
                source_gaussian_sha256=np.array(sha256(gaussian_ply)),
                source_poisson_sha256=np.array(sha256(poisson_mesh)),
                point_count=np.array(len(xyz), dtype=np.int64),
                core_field_signed_distance=field_distance,
                safety_distance_per_sfm_point=local_r,
                projection_residual=projection_error,
                r3_trim_each_tail=np.array(safety_config.trim_fraction),
                r3_knn=np.array(safety_config.k_neighbors),
                r3_safety_factor=np.array(safety_config.safety_factor),
                r3_distance_tolerance=np.array(safety_config.distance_tolerance),
                r3_projection_tolerance=np.array(safety_config.projection_tolerance))
    return target


def read_verified_handoff(record, *, clean_sfm_npz, roi_json, gaussian_ply,
                          poisson_mesh):
    expected = dict(cleaned_sfm_sha256=sha256(clean_sfm_npz), roi_sha256=sha256(roi_json),
                    source_gaussian_sha256=sha256(gaussian_ply),
                    source_poisson_sha256=sha256(poisson_mesh))
    with np.load(record, allow_pickle=False) as data:
        required = ('schema', 'semantics', 'point_count', 'core_field_signed_distance',
                    'safety_distance_per_sfm_point', 'r3_distance_tolerance',
                    'r3_projection_tolerance',
                    'r3_trim_each_tail', 'r3_safety_factor', 'r3_knn',
                    'projection_residual', *expected.keys())
        if any(key not in data for key in required):
            raise ValueError('Incomplete R3 v3 handoff')
        if data['schema'].item() != SCHEMA or data['semantics'].item() != SEMANTICS:
            raise ValueError('Not a post-restoration R3 handoff; old initial-ROI v2 is not accepted')
        if any(str(data[k].item()) != value for k, value in expected.items()):
            raise ValueError('Stage-1 R3 source hashes differ from current ROI/SfM/Gaussian/Poisson')
        d = np.asarray(data['core_field_signed_distance'], dtype=np.float64)
        r = np.asarray(data['safety_distance_per_sfm_point'], dtype=np.float64)
        tol = float(data['r3_distance_tolerance'].item())
        n = int(data['point_count'].item())
        if (not np.isclose(float(data['r3_trim_each_tail'].item()), .05, atol=1e-12)
                or not np.isclose(float(data['r3_safety_factor'].item()), 1., atol=1e-12)
                or int(data['r3_knn'].item()) < 3
                or not np.isfinite(data['projection_residual']).all()):
            raise ValueError('R3 v3 safety metadata differs from Stage-1 fixed 5%/1x scheme')
    with np.load(clean_sfm_npz, allow_pickle=False) as data:
        xyz_n = len(data['xyz'])
    if (n != xyz_n or d.shape != (n,) or r.shape != (n,)
            or not np.isfinite(d).all() or not np.isfinite(r).all()
            or np.any(r <= 0) or not np.isfinite(tol) or tol < 0):
        raise ValueError('Malformed or wrong-length R3 spatial field for cleaned SfM')
    return d, r, tol, expected


def select_verified_clean_sfm(clean_sfm_npz, roi_json, gaussian_ply, poisson_mesh,
                              handoff_file):
    """Subset ONLY cleaned SfM; core=R3 1x; Poisson support=R3 2x."""
    d, r, tol, source_hashes = read_verified_handoff(
        handoff_file, clean_sfm_npz=clean_sfm_npz, roi_json=roi_json,
        gaussian_ply=gaussian_ply, poisson_mesh=poisson_mesh)
    core = d <= r + tol
    build = d <= 2.0 * r + tol
    if not np.all(~core | build):
        raise AssertionError('2x R3 support must contain 1x core')
    if core.sum() < 100:
        raise ValueError(f'Only {int(core.sum())} core cleaned SfM points; need >=100')
    with np.load(handoff_file, allow_pickle=False) as rec:
        residual = np.asarray(rec['projection_residual'])
        projection_tolerance = float(rec['r3_projection_tolerance'].item())
    selected = {}
    with np.load(clean_sfm_npz, allow_pickle=False) as sfm:
        n = len(d)
        for key in sfm.files:
            value = np.asarray(sfm[key])
            if value.ndim and len(value) == n:
                selected[key] = value[build]
            elif value.ndim == 0:
                selected[key] = value
            else:
                raise ValueError(f'Ambiguous cleaned SfM field {key}: {value.shape}')
    selected.update(in_core_roi=core[build],
                    in_r3_geometric_core=(d[build] <= tol),
                    r3_core_field_distance=d[build],
                    stage1_r3_local_safety_distance=r[build],
                    core_outward_distance_local=r[build],
                    build_outward_distance_local=2.0*r[build],
                    source_row_indices_in_clean_sfm=np.flatnonzero(build).astype(np.int64),
                    roi_sfm_schema=np.array(SFM_SCHEMA),
                    source_cleaned_sfm_sha256=np.array(source_hashes['cleaned_sfm_sha256']),
                    source_local_safety_sha256=np.array(sha256(handoff_file)))
    report = dict(schema=SFM_SCHEMA, selection_mode=SELECTION,
                  description='Use exact post-restoration R3 field F(x) with 1x/2x original R3 kNN radius',
                  source_is_cleaned_stage1_sfm=True,
                  source_sfm_points=len(d), core_points=int(core.sum()),
                  build_points=int(build.sum()),
                  context_only_points=int((build & ~core).sum()),
                  r3_geometric_core_sfm_points=int(np.sum(d <= tol)),
                  boundary_projection_failures=int(np.count_nonzero(residual > projection_tolerance)),
                  boundary_projection_tolerance=projection_tolerance,
                  minimum_radius=float(r.min()), median_radius=float(np.median(r)),
                  maximum_radius=float(r.max()),
                  coordinate_frame='unchanged COLMAP world',
                  gaussian_r3_unchanged=True, sdf_threshold_formula_unchanged=True,
                  warning='SfM 1x is spatial R3 query, not Gaussian membership or semantic certification')
    return selected, report


def create_roi_sfm_v3(*, clean_sfm_npz, roi_json, gaussian_ply, poisson_mesh,
                      handoff_file, output_npz, output_report):
    result = Path(output_npz)
    metadata = Path(output_report)
    provenance = dict(clean_sfm_sha256=sha256(clean_sfm_npz), roi_sha256=sha256(roi_json),
                      original_gaussian_sha256=sha256(gaussian_ply),
                      source_poisson_sha256=sha256(poisson_mesh),
                      handoff_sha256=sha256(handoff_file))
    if result.exists() or metadata.exists():
        if not result.is_file() or not metadata.is_file():
            raise FileExistsError('Incomplete v3 ROI SfM publication: ' + str(result))
        old = json.loads(metadata.read_text(encoding='utf8'))
        if (old.get('schema') != SFM_SCHEMA or old.get('sources') != provenance
                or old.get('sfm_roi_sha256') != sha256(result)):
            raise ValueError('Existing ROI SfM belongs to a different algorithm/source')
        return result.resolve(), old
    records, report = select_verified_clean_sfm(clean_sfm_npz, roi_json,
                               gaussian_ply, poisson_mesh, handoff_file)
    report['sources'] = provenance
    _npz_atomic(result, **records)
    report['sfm_roi_sha256'] = sha256(result)
    try:
        _json_atomic(metadata, report)
    except Exception:
        result.unlink(missing_ok=True)
        raise
    return result.resolve(), report


def publish_stage1_r3_sfm(*, downstream, metadata_dir, work_dir, gaussian_source_file):
    """Call BEFORE stage1 os.replace(model, dest) so all SfM files are staged."""
    source = Path(work_dir) / WORK_NAME
    if not source.is_file():
        # Preserve tests / historical Stage-1 entrypoints with no real R3 writer.
        return None, None
    meta = Path(metadata_dir)
    meta.mkdir(parents=True, exist_ok=True)
    clean = downstream['clean_sfm_npz']
    roi = downstream['roi_json']
    # Full Gaussian input, NOT the final Gaussian ROI.
    original = downstream['source_full_gaussian_ply']
    poisson = downstream['poisson_mesh']
    read_verified_handoff(source, clean_sfm_npz=clean, roi_json=roi,
                          gaussian_ply=original, poisson_mesh=poisson)
    record = meta / PUBLISHED_NAME
    if record.exists() and sha256(record) != sha256(source):
        raise FileExistsError('Existing published R3 safety is from another source')
    if not record.exists():
        staging = record.with_name('.' + record.name + '.writing')
        try:
            shutil.copy2(source, staging)
            os.replace(staging, record)
        finally:
            staging.unlink(missing_ok=True)
    # Hash the already-written TEMP final PLY; its bytes will be identical after
    # the publisher's os.replace. The final destination need not exist yet.
    source_final_hash = sha256(gaussian_source_file)
    dest, report = create_roi_sfm_v3(
        clean_sfm_npz=clean, roi_json=roi, gaussian_ply=original,
        poisson_mesh=poisson, handoff_file=record,
        output_npz=Path(downstream['gaussian_ply']).parent / 'sfm_roi.npz',
        output_report=meta / 'roi_sfm_report_v3.json')
    report['stage1_gaussian_roi_sha256_at_publication'] = source_final_hash
    _json_atomic(meta / 'roi_sfm_report_v3.json', report)
    downstream['r3_local_safety_npz'] = str(record.resolve())
    downstream['sfm_roi_npz'] = str(dest.resolve())
    return dest, report


def _recompute_original_r3_field(run_dir, inputs, upstream, work_dir):
    """Recompute just original R3 spatial geometry for an OLD confirmed ROI.

    No ROI editing, no Gaussian training, no Gaussian classification/export, no
    Poisson reconstruction. The exact ORIGINAL stage1 face classifier and R3
    geometric core construction are called, with the same config and sources.
    """
    import trimesh
    import yaml
    from gsrefine.roi.core import ROI, GaussianPLY, geometry_confidence, load_sfm
    from gsrefine.roi.study import analyze, anchored_local_recovery_faces
    from gsrefine.roi.two_stage_sdf import (
        SafetyConfig, CoreImplicitField, roi_polygon_implicit_field,
        poisson_surface_coverage, estimate_geometric_support_radius,
        longest_gaussian_axis)
    cfgfile = Path(run_dir) / 'metadata/effective_scene.yaml'
    cfg = yaml.safe_load(cfgfile.read_text(encoding='utf8'))
    source = Path(upstream.get('source_gaussian') or cfg['paths']['gaussian_ply']).resolve(strict=True)
    roi = ROI.from_file(inputs['roi_json'])
    mesh = trimesh.load(inputs['poisson_mesh'], process=False)
    gaussian = GaussianPLY(source)
    xyz = gaussian.positions()
    diag = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    if not diag > 0:
        raise ValueError('Invalid source Gaussian dimensions')
    info = json.loads(Path(inputs['roi_json']).read_text(encoding='utf8'))
    up = np.asarray(info['annotation_metadata']['up_vector'], dtype=float)
    up /= np.linalg.norm(up)
    edges = np.asarray(mesh.edges_sorted)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    flags = np.zeros(len(mesh.vertices), dtype=bool)
    flags[unique[counts == 1].reshape(-1)] = True
    mesh._roi_boundary_vertex_flags = flags
    study_cfg = cfg.get('roi_study', {}) or {}
    if bool(study_cfg.get('poisson_confidence_enabled', False)):
        sfm_xyz, err, tracks, fields = load_sfm(inputs['clean_sfm_npz'])
        conf, *_ = geometry_confidence(mesh, sfm_xyz, err, tracks,
            k=int(cfg['sfm']['knn_spacing']),
            weights=tuple(cfg['sfm']['confidence_weights']))
    else:
        conf = np.ones(len(mesh.faces), dtype=np.float64)
    analysis = analyze(mesh, roi, conf, float(cfg['sfm']['confidence_threshold']),
        diag, cfg['guards'], up, candidate_labels={}, study_cfg=study_cfg,
        growth_multiple=float(study_cfg.get('growth_multiple', 3.0)),
        scales_count=int(study_cfg.get('growth_steps', 6)))
    exterior = analysis['label_masks']['suspect']
    geo_limit = min(float(cfg['guards']['geodesic_limit_frac']),
                    float(cfg['guards']['outside_search_frac'])) * diag
    recovery, _ = anchored_local_recovery_faces(
        mesh, roi, exterior, analysis['inside'], geo_limit)
    sdf = roi_polygon_implicit_field(roi)
    surface, samples = poisson_surface_coverage(mesh, recovery, diag)
    r_geom, _ = estimate_geometric_support_radius(
        roi, xyz, np.all(roi.distances(xyz) >= -1e-9, axis=1),
        surface, samples['surface_cover_error_bound'], diag)
    support = r_geom + samples['surface_cover_error_bound'] if len(surface) else 0.0
    field = CoreImplicitField(sdf, surface, support)
    scales = np.column_stack([gaussian.arr[f'scale_{j}'] for j in range(3)])
    axes = longest_gaussian_axis(scales, logarithmic=True)
    safety_cfg = SafetyConfig(
        k_neighbors=min(int(cfg['guards'].get('safety_knn',128)),len(xyz)),
        trim_fraction=.05, safety_factor=1.0, batch_size=8192,
        projection_step=max(samples['surface_cover_error_bound']/5.0,diag*1e-6,1e-6),
        projection_tolerance=max(diag*1e-5,1e-6))
    # Reconstructed support should classify already-exported R3 Gaussian
    # centers as kept. Audit a deterministic subset before accepting a backfill.
    from gsrefine.roi.two_stage_sdf import _project_to_boundary, trimmed_knn_mean
    final_xyz = GaussianPLY(inputs['gaussian_ply']).positions()
    sample = final_xyz[np.linspace(0, len(final_xyz)-1,
                  min(2048, len(final_xyz)), dtype=np.int64)]
    dist = np.asarray(field.evaluate(sample))
    outside = dist > safety_cfg.distance_tolerance
    if np.any(outside):
        boundary, _ = _project_to_boundary(field, sample[outside], safety_cfg)
        radii = trimmed_knn_mean(cKDTree(xyz), axes, boundary,
                    k=safety_cfg.k_neighbors, trim_fraction=safety_cfg.trim_fraction,
                    batch_size=safety_cfg.batch_size)
        # A very small extra numerical slack accounts for float32 PLY positions.
        slack = max(safety_cfg.distance_tolerance * 10, diag * 1e-7)
        failures = int(np.count_nonzero(dist[outside] > radii + slack))
        if failures:
            raise ValueError(f'R3 historical field mismatch: {failures} of '
                             f'{len(sample)} sampled previously-kept Gaussians '
                             'fall outside the reconstructed R3 1x field. '
                             'Refusing an unverified SfM backfill.')
    return save_r3_sfm_handoff(
        work_dir=work_dir, clean_sfm_npz=inputs['clean_sfm_npz'],
        roi_json=inputs['roi_json'], gaussian_ply=source,
        poisson_mesh=inputs['poisson_mesh'], core_field=field,
        safety_config=safety_cfg, gaussian_positions=xyz, major_axes=axes)


def resolve_stage2_r3_sfm(run_dir, upstream, inputs, *, local_safety_file=None):
    """Stage-2 only: prefer exact Stage-1 v3, or refit R3 *support* for old runs."""
    run_dir = Path(run_dir)
    # The Stage-1 manifest normally records the full model source.
    if upstream.get('source_gaussian'):
        original = Path(upstream['source_gaussian'])
    else:
        import yaml
        original = Path(yaml.safe_load((run_dir/'metadata/effective_scene.yaml').read_text(encoding='utf8'))['paths']['gaussian_ply'])
    original = original.resolve(strict=True)
    expected_source_sha = (upstream.get('input_sha256') or {}).get('gaussian')
    if expected_source_sha and sha256(original) != expected_source_sha:
        raise ValueError('Historical Stage-1 full Gaussian has changed: refusing R3 support backfill')
    published = upstream.get('stage2_inputs') or {}
    candidates = ([Path(local_safety_file)] if local_safety_file else [])
    if published.get('r3_local_safety_npz'):
        candidates.append(Path(published['r3_local_safety_npz']))
    candidates += [run_dir/'metadata'/PUBLISHED_NAME,
                   run_dir/'cleanup/inputs'/PUBLISHED_NAME,
                   run_dir/'cleanup/inputs'/WORK_NAME]
    record = None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            with np.load(candidate, allow_pickle=False) as existing:
                schema = str(existing['schema'].item()) if 'schema' in existing else ''
        except (ValueError, OSError, KeyError) as error:
            raise ValueError('Unreadable R3 handoff candidate: '+str(candidate)) from error
        if schema == SCHEMA:
            record = candidate
            break
        if local_safety_file and candidate == Path(local_safety_file):
            raise ValueError('Explicit Stage-1 local safety record is not post-restoration R3 v3')
        # Ignore historical v1/v2 *for discovery*, never re-label as R3 v3.
    if record is None:
        # A historical Stage-1 v2 file is deliberately not treated as exact R3.
        folder = run_dir/'cleanup/inputs'
        folder.mkdir(parents=True, exist_ok=True)
        record = _recompute_original_r3_field(run_dir, inputs, upstream, folder)
        # Keep the audit record under the existing cleanup scope.
    d, r, tol, hashes = read_verified_handoff(record,
        clean_sfm_npz=inputs['clean_sfm_npz'], roi_json=inputs['roi_json'],
        gaussian_ply=original, poisson_mesh=inputs['poisson_mesh'])
    listed = published.get('sfm_roi_npz')
    if listed and (run_dir/'metadata/roi_sfm_report_v3.json').exists():
        p = Path(listed)
        report = json.loads((run_dir/'metadata/roi_sfm_report_v3.json').read_text(encoding='utf8'))
        if (p.is_file() and report.get('schema') == SFM_SCHEMA
                and report.get('sfm_roi_sha256') == sha256(p)
                and report.get('sources',{}).get('handoff_sha256') == sha256(record)):
            return p.resolve(), report, True
    out = run_dir / 'cleanup/inputs'
    sfm_path, report = create_roi_sfm_v3(
        clean_sfm_npz=inputs['clean_sfm_npz'], roi_json=inputs['roi_json'],
        gaussian_ply=original, poisson_mesh=inputs['poisson_mesh'],
        handoff_file=record, output_npz=out/'sfm_roi_local_v3.npz',
        output_report=out/'sfm_roi_local_v3_report.json')
    return sfm_path, report, False


def main(argv=None):
    """Validation-only CLI: no Poisson rebuild, sealing, SDF or Gaussian output."""
    import argparse
    from gsrefine.cleanup.sdf_common import _load_stage1_manifest
    parser = argparse.ArgumentParser(description='Build/verify post-restoration R3 1x/2x local CLEAN SfM')
    parser.add_argument('--manifest', required=True, help='Stage-1 metadata/manifest.json')
    parser.add_argument('--stage1-local-safety-field', default=None,
                        help='Optional verified v3 R3 spatial-field NPZ')
    args = parser.parse_args(argv)
    _, manifest, inputs, run_dir = _load_stage1_manifest(args.manifest)
    local, report, reused = resolve_stage2_r3_sfm(
        run_dir, manifest, inputs, local_safety_file=args.stage1_local_safety_field)
    print(json.dumps(dict(status='READY_R3_SFM_1X_2X', source='Stage-1' if reused else 'Stage-2 backfill',
                          sfm_roi_npz=str(local), core_points=report['core_points'],
                          build_points=report['build_points'],
                          context_only_points=report['context_only_points'],
                          selection_mode=report['selection_mode']),
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
