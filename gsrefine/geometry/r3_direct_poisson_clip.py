"""Canonical clipping: FIRST module-1 Poisson by the recovered R3 1x field.

This is NOT a fresh Poisson fit or SfM triangle-support selector,
not a seal, and not Gaussian SDF pruning.  It intentionally creates an OPEN
observed-surface candidate.  It reuses canonical Stage-1 R3 geometry functions
and verifies that reconstructed field values match the saved v3 SfM handoff.

Run: python -m gsrefine.geometry.r3_direct_poisson_clip --manifest /.../roi_001/metadata/manifest.json
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

REVISION = 'global_module1_poisson_direct_r3_1x_clip_v1_20261108'


def _sha(path):
    hash_ = hashlib.sha256()
    with Path(path).open('rb') as inp:
        for block in iter(lambda: inp.read(4 * 1024 * 1024), b''):
            hash_.update(block)
    return hash_.hexdigest()


def _write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.stem, suffix='.json.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            json.dump(record, out, indent=2, ensure_ascii=False, allow_nan=False)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _source_r3_handoff(run_dir, upstream):
    """A *verified v3* spatial field is mandatory; never substitute initial ROI or a scalar."""
    from gsrefine.geometry.r3_sfm_bridge import SCHEMA
    published = upstream.get('stage2_inputs') or {}
    ordered = [published.get('r3_local_safety_npz'),
               run_dir / 'metadata/stage1_r3_local_safety_v3.npz',
               run_dir / 'cleanup/inputs/stage2_ready_r3_sfm_v3.npz',
               run_dir / 'cleanup/inputs/stage1_r3_local_safety_v3.npz']
    for entry in ordered:
        if entry is None:
            continue
        path = Path(entry)
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as record:
            schema = str(record['schema'].item()) if 'schema' in record else ''
            if schema == SCHEMA:
                return path.resolve(strict=True)
    raise FileNotFoundError(
        'No verified R3 post-restoration v3 handoff. First run '
        '`python run.py stage2-sfm --manifest ...`; never infer a continuous '
        'R3 field from the old initial ROI or from per-SfM distances alone.')


def _restore_r3_field(*, inputs, upstream, run_dir, audit_record, audit_samples=768):
    """Build an in-memory R3 field with EXACTLY the v3 backfill geometry recipe.

    No original input or manifest is modified.  We do not re-run Gaussian R3
    classification or export; the preexisting Stage-1 R3 algorithms are called.
    A saved v3 handoff is cross-checked on stratified cleaned SfM samples.
    """
    import trimesh
    import yaml
    from scipy.spatial import cKDTree
    from gsrefine.roi.core import ROI, GaussianPLY, geometry_confidence, load_sfm
    from gsrefine.roi.study import analyze, anchored_local_recovery_faces
    from gsrefine.roi.two_stage_sdf import (
        SafetyConfig, CoreImplicitField, roi_polygon_implicit_field,
        poisson_surface_coverage, estimate_geometric_support_radius,
        longest_gaussian_axis, _project_to_boundary, trimmed_knn_mean)
    from gsrefine.geometry.r3_sfm_bridge import read_verified_handoff

    cfg_path = (run_dir / 'metadata/effective_scene.yaml').resolve(strict=True)
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    if not isinstance(cfg, dict):
        raise ValueError('Invalid Stage-1 effective_scene.yaml')
    source = Path(upstream.get('source_gaussian') or cfg['paths']['gaussian_ply']).resolve(strict=True)
    expected = (upstream.get('input_sha256') or {}).get('gaussian')
    if expected and _sha(source) != expected:
        raise ValueError('Stage-1 source Gaussian SHA mismatch; refusing inaccurate R3 replay')

    stored_d, stored_r, stored_tol, hashes = read_verified_handoff(
        audit_record, clean_sfm_npz=inputs['clean_sfm_npz'],
        roi_json=inputs['roi_json'], gaussian_ply=source,
        poisson_mesh=inputs['poisson_mesh'])
    roi = ROI.from_file(inputs['roi_json'])
    global_mesh = trimesh.load(inputs['poisson_mesh'], force='mesh', process=False)
    if not isinstance(global_mesh, trimesh.Trimesh) or not len(global_mesh.faces):
        raise ValueError('Stage-1 global Poisson must be a nonempty triangle mesh')
    full_gaussian = GaussianPLY(source)
    xyz = full_gaussian.positions()
    diag = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    if not np.isfinite(diag) or diag <= 0:
        raise ValueError('Invalid full Gaussian spatial extent')
    annotation = json.loads(Path(inputs['roi_json']).read_text(encoding='utf-8'))
    up = np.asarray(annotation['annotation_metadata']['up_vector'], dtype=np.float64)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-9:
        raise ValueError('Missing Stage-1 verified orientation')
    up /= np.linalg.norm(up)

    # Exact preconditions of r3_sfm_bridge._recompute_original_r3_field().
    edges = np.asarray(global_mesh.edges_sorted)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    flags = np.zeros(len(global_mesh.vertices), dtype=bool)
    flags[unique[counts == 1].reshape(-1)] = True
    global_mesh._roi_boundary_vertex_flags = flags
    study_cfg = cfg.get('roi_study', {}) or {}
    if bool(study_cfg.get('poisson_confidence_enabled', False)):
        sfm_xyz, err, tracks, _fields = load_sfm(inputs['clean_sfm_npz'])
        confidence, *_ = geometry_confidence(
            global_mesh, sfm_xyz, err, tracks,
            k=int(cfg['sfm']['knn_spacing']),
            weights=tuple(cfg['sfm']['confidence_weights']))
    else:
        confidence = np.ones(len(global_mesh.faces), dtype=np.float64)
    original_analysis = analyze(
        global_mesh, roi, confidence, float(cfg['sfm']['confidence_threshold']),
        diag, cfg['guards'], up, candidate_labels={}, study_cfg=study_cfg,
        growth_multiple=float(study_cfg.get('growth_multiple', 3.0)),
        scales_count=int(study_cfg.get('growth_steps', 6)))
    candidates = original_analysis['label_masks']['suspect']
    local_limit = min(float(cfg['guards']['geodesic_limit_frac']),
                      float(cfg['guards']['outside_search_frac'])) * diag
    recovered, _recovery_info = anchored_local_recovery_faces(
        global_mesh, roi, candidates, original_analysis['inside'], local_limit)
    initial = roi_polygon_implicit_field(roi)
    surface, samples = poisson_surface_coverage(global_mesh, recovered, diag)
    support_radius, _ = estimate_geometric_support_radius(
        roi, xyz, np.all(roi.distances(xyz) >= -1e-9, axis=1),
        surface, samples['surface_cover_error_bound'], diag)
    support = support_radius + samples['surface_cover_error_bound'] if len(surface) else 0.
    core_field = CoreImplicitField(initial, surface, support)
    log_scales = np.column_stack([full_gaussian.arr[f'scale_{j}'] for j in range(3)])
    major_axes = longest_gaussian_axis(log_scales, logarithmic=True)
    safety_cfg = SafetyConfig(
        k_neighbors=min(int(cfg['guards'].get('safety_knn', 128)), len(xyz)),
        trim_fraction=.05, safety_factor=1.0, batch_size=8192,
        projection_step=max(samples['surface_cover_error_bound'] / 5., diag * 1e-6, 1e-6),
        projection_tolerance=max(diag * 1e-5, 1e-6))
    safety_cfg.validate(len(xyz))
    if not np.isclose(float(safety_cfg.distance_tolerance), stored_tol,
                      rtol=0., atol=max(1e-9, diag * 1e-8)):
        raise ValueError('Reconstructed R3 distance tolerance differs from stored v3 record')
    tree = cKDTree(xyz)

    def margin(queries, chunk_size=8192, return_detail=False):
        """Nonnegative = inside original R3 (post-restoration) 1x spatial mask.

        Original V3 rules: F(x)<=r(x)+tol. Both F and r evaluated at actual
        query locations; SfM point values are NEVER interpolated to vertices.
        """
        points = np.asarray(queries, dtype=np.float64).reshape(-1, 3)
        signed = np.empty(len(points), dtype=np.float64)
        projection_failures = 0
        rs = np.empty(len(points), dtype=np.float64) if return_detail else None
        ds = np.empty(len(points), dtype=np.float64) if return_detail else None
        for a in range(0, len(points), chunk_size):
            b = min(a + chunk_size, len(points))
            q = points[a:b]
            distance = np.asarray(core_field.evaluate(q), dtype=np.float64)
            boundary, residual = _project_to_boundary(core_field, q, safety_cfg)
            radii = trimmed_knn_mean(
                tree, major_axes, boundary, k=safety_cfg.k_neighbors,
                trim_fraction=safety_cfg.trim_fraction,
                batch_size=safety_cfg.batch_size) * safety_cfg.safety_factor
            if (not np.isfinite(distance).all() or not np.isfinite(radii).all()
                    or np.any(radii <= 0) or not np.isfinite(residual).all()):
                raise ValueError('Invalid R3 query; no mesh may be exported')
            signed[a:b] = radii + safety_cfg.distance_tolerance - distance
            projection_failures += int(np.sum(residual > safety_cfg.projection_tolerance))
            if return_detail:
                rs[a:b], ds[a:b] = radii, distance
        if return_detail:
            return signed, rs, ds, projection_failures
        return signed

    if audit_samples < 32:
        raise ValueError('audit_samples must be at least 32')
    with np.load(inputs['clean_sfm_npz'], allow_pickle=False) as data:
        sfm_xyz = data['xyz']
        n = len(sfm_xyz)
        # A deterministic mixture of broad and boundary-adjacent positions.
        amount = min(int(audit_samples), n)
        random_rows = np.linspace(0, n - 1, amount // 2 + amount % 2, dtype=np.int64)
        closest = np.argpartition(np.abs(stored_d - stored_r), min(n - 1, amount // 2))[:amount // 2]
        ids = np.unique(np.r_[random_rows, closest])
        _m, computed_r, computed_d, failures = margin(sfm_xyz[ids], return_detail=True)
    dd = np.abs(computed_d - stored_d[ids])
    dr = np.abs(computed_r - stored_r[ids])
    if (float(dd.max(initial=0)) > max(2e-6, diag * 2e-7)
            or float(dr.max(initial=0)) > max(2e-6, diag * 2e-7)):
        raise ValueError('R3 field replay does NOT match verified SfM v3 handoff: '
                         f'max |d| error={dd.max():.6g}; max radius error={dr.max():.6g}. '
                         'Refusing unverified geometric clipping.')

    record = {
        'source_gaussian': str(source),
        'full_gaussian_count': int(len(xyz)),
        'global_poisson_faces': int(len(global_mesh.faces)),
        'r3_recovered_surface_samples': int(len(surface)),
        'r3_geometric_support_radius': float(support),
        'r3_knn': int(safety_cfg.k_neighbors),
        'r3_trim_fraction': float(safety_cfg.trim_fraction),
        'r3_distance_tolerance': float(safety_cfg.distance_tolerance),
        'v3_handoff_audit_samples': int(len(ids)),
        'v3_handoff_max_field_distance_error': float(dd.max(initial=0)),
        'v3_handoff_max_radius_error': float(dr.max(initial=0)),
        'v3_handoff_projection_failures': int(failures),
        'reference_mesh_same_as_stage1': True,
        'source_hashes': hashes,
    }
    return global_mesh, margin, record


def _topology(mesh):
    import trimesh
    from trimesh.graph import connected_components
    faces = np.asarray(mesh.faces, np.int64)
    if not len(faces):
        raise ValueError('Empty output mesh')
    edges = np.asarray(mesh.edges_unique)
    index = np.asarray(mesh.edges_unique_inverse)
    counts = np.bincount(index, minlength=len(edges))
    open_edges = edges[counts == 1]
    if len(open_edges):
        degrees = np.bincount(open_edges.ravel(), minlength=len(mesh.vertices))
        branched = int(np.sum(degrees > 2))
    else:
        branched = 0
    comps = connected_components(mesh.face_adjacency, nodes=np.arange(len(faces)), min_len=1)
    sizes = sorted((len(c) for c in comps), reverse=True)
    return {
        'vertices': int(len(mesh.vertices)), 'faces': int(len(faces)),
        'area': float(mesh.area), 'bounds': np.asarray(mesh.bounds).tolist(),
        'open_boundary_edges': int(len(open_edges)),
        'nonmanifold_edges': int(np.sum(counts > 2)),
        'branching_boundary_vertices': branched,
        'connected_components': int(len(comps)),
        'largest_component_face_fraction': float(sizes[0] / len(faces)),
        'top_component_faces': [int(x) for x in sizes[:10]],
        'watertight': bool(mesh.is_watertight),
    }


def _fit(mesh, reference_points):
    # Exact nearest-TRIANGLE distance, not vertex-distance proxy.
    from gsrefine.roi.core import exact_sfm_to_poisson_distance_open3d
    distances = np.asarray(exact_sfm_to_poisson_distance_open3d(
        mesh, reference_points, batch=50000), dtype=np.float64)
    if len(distances) != len(reference_points) or not np.isfinite(distances).all():
        raise ValueError('Invalid SfM to observed Poisson fit distances')
    return {'distance_type': 'exact_unsigned_nearest_triangle',
            'points': int(len(reference_points)),
            'mean': float(np.mean(distances)),
            'median': float(np.median(distances)),
            'p90': float(np.quantile(distances, .90)),
            'p95': float(np.quantile(distances, .95)),
            'p99': float(np.quantile(distances, .99))}


def run(*, manifest, fit_max_points=60000,
        audit_samples=768, query_batch=8192):
    """Produce ONLY a directly clipped Stage-1 Poisson and reproducible diagnostics."""
    import trimesh
    from gsrefine.cleanup.sdf_common import _load_stage1_manifest
    from gsrefine.geometry.r3_sfm_bridge import read_verified_handoff
    from gsrefine.geometry.mesh_clip import clip_piecewise_linear as _clip_piecewise_linear

    if fit_max_points < 0 or query_batch < 1:
        raise ValueError('fit_max_points must be >=0 and query_batch must be positive')
    stage1_path, upstream, inputs, run_dir = _load_stage1_manifest(manifest)
    run_dir = Path(run_dir)
    record = _source_r3_handoff(run_dir, upstream)
    effective = json.loads(Path(inputs['roi_json']).read_text(encoding='utf-8'))
    if effective.get('annotation_metadata', {}).get('orientation_verified') is not True:
        raise ValueError('ROI orientation is not verified')
    import yaml
    cfg = yaml.safe_load((run_dir / 'metadata/effective_scene.yaml').read_text(encoding='utf-8'))
    original = Path(upstream.get('source_gaussian') or cfg['paths']['gaussian_ply']).resolve(strict=True)
    read_verified_handoff(record, clean_sfm_npz=inputs['clean_sfm_npz'],
        roi_json=inputs['roi_json'], gaussian_ply=original,
        poisson_mesh=inputs['poisson_mesh'])
    sources = {
        'stage1_manifest_sha256': _sha(stage1_path),
        'stage1_poisson_sha256': _sha(inputs['poisson_mesh']),
        'original_gaussian_sha256': _sha(original),
        'saved_roi_sha256': _sha(inputs['roi_json']),
        'v3_handoff_sha256': _sha(record),
        'roi_core_sfm_sha256': None,
    }
    roi_sfm = run_dir / 'cleanup/inputs/sfm_roi_local_v3.npz'
    if not roi_sfm.is_file():
        sfm_declared = (upstream.get('stage2_inputs') or {}).get('sfm_roi_npz')
        roi_sfm = Path(sfm_declared).resolve(strict=True) if sfm_declared else roi_sfm
    if not roi_sfm.is_file():
        raise FileNotFoundError('V3 local SfM missing; first run `python run.py stage2-sfm --manifest ...`')
    sources['roi_core_sfm_sha256'] = _sha(roi_sfm)
    identity = dict(revision=REVISION, sources=sources,
                    mode='post_restoration_R3_1x', fit_max_points=int(fit_max_points),
                    audit_samples=int(audit_samples))
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()
    folder = run_dir / 'cleanup/geometry/direct_r3_clip' / signature[:24]
    result = folder / 'poisson_roi_direct_r3.ply'
    report_path = folder / 'clip_comparison_report.json'
    if folder.exists():
        if not result.is_file() or not report_path.is_file():
            raise FileExistsError('Incomplete existing experiment; refusing overwrite: ' + str(folder))
        saved = json.loads(report_path.read_text(encoding='utf-8'))
        if (saved.get('signature') != signature or saved.get('sources') != sources
                or saved.get('output_sha256') != _sha(result)):
            raise ValueError('Existing clip experiment input/output hashes mismatch')
        print(json.dumps({'status': 'REUSED_DIRECT_R3_POISSON_CANDIDATE',
                          'candidate_mesh': str(result), 'report': str(report_path),
                          'signature': signature}, indent=2, ensure_ascii=False))
        return saved

    global_mesh, margin_fn, field_report = _restore_r3_field(
        inputs=inputs, upstream=upstream, run_dir=run_dir, audit_record=record,
        audit_samples=audit_samples)
    source_verts = np.asarray(global_mesh.vertices, np.float64)
    signed = np.empty(len(source_verts), dtype=np.float64)
    for a in range(0, len(source_verts), query_batch):
        b = min(len(source_verts), a + query_batch)
        signed[a:b] = margin_fn(source_verts[a:b], chunk_size=query_batch)
        if a == 0 or b == len(source_verts) or a // query_batch % 20 == 0:
            print(f'[direct-r3-clip] global Poisson vertex R3 queries {b:,}/{len(source_verts):,}', flush=True)
    # Finite alias audit: a nonlinear R3 union can intersect a triangle even
    # when all three corner signs agree.  Report, do NOT silently fix interiors
    # by deleting faces based on SfM density.
    src_faces = np.asarray(global_mesh.faces, dtype=np.int64)
    sample_face_ids = np.linspace(0, len(src_faces)-1, min(12000, len(src_faces)), dtype=np.int64)
    face_values = signed[src_faces[sample_face_ids]]
    centroids = source_verts[src_faces[sample_face_ids]].mean(axis=1)
    centroid_true = margin_fn(centroids, chunk_size=query_batch)
    corners_all_in = (face_values >= 0).all(axis=1)
    corners_all_out = (face_values < 0).all(axis=1)
    alias_audit = {
        'sampled_original_faces': int(len(sample_face_ids)),
        'all_outside_corners_but_centroid_inside': int(np.sum(corners_all_out & (centroid_true > 0))),
        'all_inside_corners_but_centroid_outside': int(np.sum(corners_all_in & (centroid_true < 0))),
        'warning': 'Nonlinear R3 boundary can have subtriangle aliasing. This is a comparison preview, not a certified watertight cut.',
    }
    # One shared-edge piecewise-linear clip for the R3 route.
    # Original Poisson is NEVER remeshed; each crossing edge is split once.
    clipped, clip_metadata = _clip_piecewise_linear(global_mesh, signed)
    source_ids = clip_metadata.pop('source_face_ids')
    if len(source_ids) != len(clipped.faces):
        raise AssertionError('Original face provenance must match clipped triangles')
    if not np.isfinite(clipped.vertices).all() or clipped.area <= 0:
        raise ValueError('Invalid clipped Poisson mesh')
    # Audit how well interpolated boundary-edge points satisfy true R3 field.
    new_vertices = np.asarray(clipped.vertices)
    # remove_unreferenced_vertices reindexes; instead test a sample of all output
    # vertices close to source ROI, reporting interpolation/probe discrepancies.
    check = new_vertices[np.linspace(0, len(new_vertices)-1,
                       min(4000, len(new_vertices)), dtype=np.int64)]
    truth = margin_fn(check, chunk_size=query_batch)
    interpolation_audit = {
        'samples': int(len(check)),
        'evaluated_membership_outside_count': int(np.sum(truth < -1e-7)),
        'evaluated_membership_outside_fraction': float(np.mean(truth < -1e-7)),
        'min_true_margin': float(np.min(truth)),
        'note': 'Diagnostic only. Linear edge crossings approximate the nonlinear R3 field; '
                'the continuous-field membership is re-queried on output mesh samples.'
    }
    topology = _topology(clipped)
    with np.load(roi_sfm, allow_pickle=False) as src:
        pts = np.asarray(src['xyz'], dtype=np.float64)
        core = np.asarray(src['in_core_roi'], dtype=bool)
    subset = pts[core]
    count = len(subset) if fit_max_points == 0 else min(len(subset), fit_max_points)
    fit_rows = np.linspace(0, len(subset)-1, count, dtype=np.int64)
    validation = subset[fit_rows]
    print(f'[direct-r3-clip] fitting {count:,} R3 core SfM points to direct source', flush=True)
    try:
        fit = _fit(clipped, validation)
        comparison = {'global_R3_direct_clip': {'mesh': str(result),
                          'topology': topology, 'sfm_fit': fit, 'fit_status': 'computed'}}
    except Exception as exc:
        comparison = {'global_R3_direct_clip': {'mesh': str(result),
                'topology': topology, 'fit_status': 'failed_not_certified',
                'fit_error': type(exc).__name__ + ': ' + str(exc)}}

    report = {
        'schema': 'GSSceneRefine.direct_r3_global_poisson_clip.v1',
        'status': 'CANDIDATE_ONLY_NOT_SEALED_NOT_GAUSSIAN_CLEANED',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'signature': signature, 'sources': sources,
        'source_global_poisson': str(inputs['poisson_mesh']),
        'source_roi_json': str(inputs['roi_json']),
        'source_r3_handoff': str(record),
        'original_gaussian': str(original),
        'candidate_mesh': str(result),
        'selection_rule': 'recovered_R3_CoreImplicitField(x) <= original_R3_1x_trimmed_KNN_radius(x) + distance_tolerance',
        'clipping_method': 'shared_edge_piecewise_linear_intersection_with_postrestoration_R3_1x',
        'source_face_lineage': 'Each retained triangle comes from a Stage-1 global Poisson triangle; '
                               'boundary triangles are split; NO unsupported SfM face deletion.',
        'global_mesh_topology': _topology(global_mesh),
        'candidate_topology': topology,
        'clip_counts': clip_metadata,
        'original_source_faces_used': int(len(np.unique(source_ids))),
        'field_audit': field_report,
        'interpolated_boundary_field_audit': interpolation_audit,
        'nonlinear_boundary_alias_audit': alias_audit,
        'fit_sample_rule': 'deterministic_even_spacing_across_R3_1x_core_sfm_rows',
        'fit_sample_total_available': int(len(subset)),
        'fit_sample_used_identically_for_all_meshes': int(count),
        'comparison': comparison,
        'limitations': [
            'R3 1x recovered implicit field is a geometric inclusion rule, not a semantic rock mask.',
            'Piecewise-linear edge intersection approximates nonlinear R3 and can miss subtriangle features.',
            'Source Stage-1 global Poisson can itself contain internal inferred/non-target surfaces.',
            'A trimmed/cut surface is expected to be OPEN; no closure, skirt, cap or SDF pruning is performed.',
            'SfM nearest-triangle fit is not external ground-truth geometry accuracy.',
        ]
    }
    folder.mkdir(parents=True, exist_ok=False)
    temp = folder / 'poisson_roi_direct_r3.writing.ply'
    try:
        clipped.export(temp, file_type='ply', encoding='binary')
        os.replace(temp, result)
        report['output_sha256'] = _sha(result)
        _write_json(report_path, report)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps({
        'status': report['status'], 'candidate_mesh': str(result),
        'report': str(report_path),
        'global_source_faces': len(global_mesh.faces),
        'candidate_faces': len(clipped.faces),
        'candidate_open_boundary_edges': topology['open_boundary_edges'],
        'fit_sample_points': count,
        'signature': signature,
    }, ensure_ascii=False, indent=2))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--fit-max-points', type=int, default=60000,
                    help='Deterministic R3-core SfM sample; 0 = all points')
    ap.add_argument('--audit-samples', type=int, default=768,
                    help='Verify reconstituted R3 field against v3 handoff on this many SfM samples')
    ap.add_argument('--query-batch', type=int, default=8192)
    args = ap.parse_args(argv)
    run(manifest=args.manifest, fit_max_points=args.fit_max_points,
        audit_samples=args.audit_samples, query_batch=args.query_batch)


if __name__ == '__main__':
    main()
