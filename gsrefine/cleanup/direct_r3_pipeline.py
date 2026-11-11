"""GSSceneRefine canonical Stage-2 R3-direct geometry -> guarded seal -> Direct SDF.

This path uses ONLY Stage-1 global Poisson clipped by the verified R3 field.
It never rebuilds local Poisson or applies SfM-based per-face deletion.
Outputs are fail-closed, provenance-bound, separate from immutable Stage-1 files.

CLI: python run.py stage2-cleanup --manifest FILE --phase check|seal|sdf|export|open-ab
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

REVISION = 'stage2_direct_r3_main_auxiliary_safe_tilted_seal_v5_20261109'
# Open A+B uses its own revision and does not affect historical seal cache signatures.
DEFAULTS = {
    'trim_fraction': 0.10, 'sigma_multiplier': 3.0, 'epsilon': 1e-5,
    'nsamples': 3, 'chunk_size': 40000, 'poisson_weld_digits': 8,
    'min_camera_outside_fraction': 1.0, 'min_camera_normal_agreement': 0.60,
    'min_main_face_fraction': 0.85, 'bottom_margin_fraction': 0.10,
    'max_p95_to_sfm_spacing': 3.0, 'max_p95_to_roi_diagonal': 0.05,
    'min_roi_fit_points': 100,
    # Heterogeneous SfM spacing: global 3xh is a DIAGNOSTIC, not a hard veto.
    'aux_sfm_support_factor': 2.5, 'aux_gaussian_support_factor': 1.5,
    'aux_min_unique_points': 3, 'aux_min_face_coverage': 0.5,
    # Cap-plane fallback only if COLMAP Up projection is invalid. Artificial
    # closure geometry may be tilted; the original observed surface never moves.
    'seal_max_tilt_deg': 25.0, 'seal_tilt_step_deg': 5.0,
    'seal_tilt_azimuths': 16, 'seal_max_tilt_trials': 8,
}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def select_direct_r3_clip(run_dir, *, selected_report=None):
    """Do not guess between different authenticated direct-R3 experiments."""
    root = Path(run_dir).resolve(strict=True)
    if selected_report:
        report_paths = [Path(selected_report).resolve(strict=True)]
    else:
        report_paths = sorted((root / 'cleanup/geometry/direct_r3_clip').glob('*/clip_comparison_report.json'))
        if len(report_paths) != 1:
            raise ValueError('Expected exactly one Direct R3 clip report; found %d. '
                             'Use --clip-report to select the validated candidate.' % len(report_paths))
    p = report_paths[0]
    if not p.is_relative_to(root):
        raise ValueError('Clip report must belong to the same ROI run')
    record = _read(p)
    if record.get('schema') != 'GSSceneRefine.direct_r3_global_poisson_clip.v1':
        raise ValueError('Not an authenticated direct R3/global Poisson clip report')
    if record.get('status') != 'CANDIDATE_ONLY_NOT_SEALED_NOT_GAUSSIAN_CLEANED':
        raise ValueError('Invalid clip status')
    mesh = Path(record['candidate_mesh']).resolve(strict=True)
    if not mesh.is_relative_to(root) or mesh != p.parent / 'poisson_roi_direct_r3.ply':
        raise ValueError('Direct R3 candidate path is outside expected experiment directory')
    if _sha(mesh) != record.get('output_sha256'):
        raise ValueError('Direct R3 candidate PLY changed since clip verification')
    if p.name != 'clip_comparison_report.json':
        raise ValueError('Invalid direct R3 report name')
    return mesh, p, record


def validate_sources(manifest_path, manifest, inputs, run_dir, clip_report, roi_sfm):
    sources = clip_report.get('sources') or {}
    expected = {
        'stage1_manifest_sha256': _sha(manifest_path),
        'stage1_poisson_sha256': _sha(inputs['poisson_mesh']),
        'saved_roi_sha256': _sha(inputs['roi_json']),
        'roi_core_sfm_sha256': _sha(roi_sfm),
    }
    for name, digest in expected.items():
        if sources.get(name) != digest:
            raise ValueError('Direct R3 clipping provenance mismatch for ' + name)
    # The clip report also binds the full Gaussian and V3 R3 handoff.
    from gsrefine.geometry.r3_direct_poisson_clip import _source_r3_handoff
    handoff = _source_r3_handoff(Path(run_dir), manifest)
    if sources.get('v3_handoff_sha256') != _sha(handoff):
        raise ValueError('Direct R3 V3 handoff hash changed')
    import yaml
    cfg = yaml.safe_load((Path(run_dir) / 'metadata/effective_scene.yaml').read_text(encoding='utf8'))
    original = Path(manifest.get('source_gaussian') or cfg['paths']['gaussian_ply']).resolve(strict=True)
    if sources.get('original_gaussian_sha256') != _sha(original):
        raise ValueError('Original full Gaussian source does not match the R3 clip')
    return {'source_hashes': expected, 'v3_handoff_sha256': sources['v3_handoff_sha256']}


def boundary_diagnostics(mesh, *, max_loops=25):
    """Audit all open mesh edges WITHOUT discarding or repairing any faces.

    One simple ring is the only topology accepted by the existing safe skirt.
    A single ring is necessary but not sufficient (projected self-crossings can fail).
    """
    import trimesh
    from collections import defaultdict
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError('Expected a nonempty triangle mesh')
    faces = np.asarray(mesh.faces, np.int64)
    edges = np.concatenate((faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]))
    unique, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0,
                                         return_inverse=True, return_counts=True)
    open_edges = unique[counts == 1]
    nonmanifold = unique[counts > 2]
    graph = defaultdict(list)
    for a, b in open_edges:
        a, b = int(a), int(b)
        graph[a].append(b)
        graph[b].append(a)
    degree_one = sum(len(v) == 1 for v in graph.values())
    branching = sum(len(v) > 2 for v in graph.values())
    irregular = sum(len(v) != 2 for v in graph.values())
    components = []
    unseen = set(graph)
    v = np.asarray(mesh.vertices, dtype=float)
    while unseen:
        start = min(unseen)
        todo = [start]
        seen = {start}
        unseen.remove(start)
        while todo:
            u = todo.pop()
            for w in graph[u]:
                if w not in seen:
                    seen.add(w); todo.append(w); unseen.discard(w)
        num_edges = sum(len(graph[u]) for u in seen) // 2
        perimeter = sum(float(np.linalg.norm(v[int(a)]-v[int(b)]))
                        for a, b in open_edges if int(a) in seen)
        components.append({'vertices': len(seen), 'edges': num_edges,
                           'is_simple_closed_ring': all(len(graph[u]) == 2 for u in seen),
                           'perimeter': perimeter})
    components.sort(key=lambda x: x['perimeter'], reverse=True)
    face_components = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1)
    sorted_components = sorted(face_components, key=len, reverse=True)
    sizes = [len(c) for c in sorted_components]
    # Spatial localization for genuine unresolved anomalies; no provenance
    # archaeology into first Poisson is required for safe repair decisions.
    minor_components = []
    for component in sorted_components[1:81]:
        loc_faces = faces[np.asarray(component, dtype=np.int64)]
        xyz_component = v[np.unique(loc_faces.ravel())]
        minor_components.append({
            'faces':int(len(component)),
            'centroid_xyz':np.mean(xyz_component,axis=0).tolist(),
            'bounds_xyz':[np.min(xyz_component,axis=0).tolist(),
                          np.max(xyz_component,axis=0).tolist()],
        })
    nonmanifold_examples=[]
    for a,b in nonmanifold[:80]:
        edges_sorted = np.sort(edges,axis=1)
        mask = (edges_sorted[:,0] == a) & (edges_sorted[:,1] == b)
        nonmanifold_examples.append({
            'vertex_ids':[int(a),int(b)],
            'xyz':[v[int(a)].tolist(),v[int(b)].tolist()],
            'incident_face_count':int(np.count_nonzero(mask)),
        })
    can_attempt = (len(nonmanifold) == 0 and len(components) == 1
                   and components[0]['is_simple_closed_ring']
                   and len(sizes) == 1)
    result = {
        'mesh_vertices': int(len(mesh.vertices)), 'mesh_faces': int(len(faces)),
        'mesh_area': float(mesh.area),
        'open_boundary_edges': int(len(open_edges)),
        'open_boundary_components': len(components),
        'boundary_component_details': components[:max_loops],
        'boundary_degree_one_vertices': degree_one,
        'boundary_branch_vertices': branching,
        'boundary_non_degree_two_vertices': irregular,
        'nonmanifold_edges': int(len(nonmanifold)),
        'nonmanifold_edge_examples':nonmanifold_examples,
        'minor_face_component_locations':minor_components,
        'face_components': len(sizes), 'face_component_face_counts': sizes[:12],
        'winding_consistent': bool(mesh.is_winding_consistent),
        'is_watertight': bool(mesh.is_watertight),
        'preseal_one_ring_eligible': can_attempt,
        'winding_repair_on_copy_required':bool(can_attempt and not mesh.is_winding_consistent),
        'preseal_reason': ('eligible_for_guarded_trial_not_certified' if can_attempt else
                           'topology_not_safe_for_existing_single_ring_skirt'),
        'warning': 'A single valid boundary loop is not proof of a correct rock solid. '
                   'Projected ring, camera, fit, and signed volume validation follow.'
    }
    return result


def sfm_fit(mesh, sfm_xyz, *, max_points=0):
    """Exact point-to-TRIANGLE unsigned residuals on OBSERVED open mesh only."""
    from scipy.spatial import cKDTree
    from gsrefine.roi.core import exact_sfm_to_poisson_distance_open3d
    xyz = np.asarray(sfm_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 100:
        raise ValueError('Need at least 100 R3 core SfM points for geometry fit')
    if max_points < 0:
        raise ValueError('max_points must be >= 0')
    n = len(xyz) if max_points == 0 else min(int(max_points), len(xyz))
    sample = xyz[np.linspace(0, len(xyz) - 1, n, dtype=np.int64)]
    d = np.asarray(exact_sfm_to_poisson_distance_open3d(mesh, sample, batch=50000), dtype=np.float64)
    if d.shape != (n,) or not np.isfinite(d).all():
        raise ValueError('Invalid exact SfM-to-open-Poisson distance output')
    # Native sampling density ALWAYS comes from the complete core SfM,
    # even when only a preview subset is used for geometric residuals.
    tree = cKDTree(xyz)
    dd, _ = tree.query(sample, k=min(7, len(xyz)), workers=-1)
    kth = dd[:, -1]
    positive = kth[kth > 0]
    if len(positive) < max(100, len(kth) // 5):
        raise ValueError('Not enough non-duplicate SfM neighborhoods for reliable P95 normalization')
    spacing = float(np.median(positive))
    normalized = d / np.maximum(kth, 1e-10)
    return {
        'source_core_sfm_count': len(xyz), 'fit_sfm_count': n,
        'is_full_core_fit': (n == len(xyz)),
        'distance_type': 'exact_unsigned_point_to_observed_triangle',
        'mean': float(d.mean()), 'p90': float(np.quantile(d,.90)),
        'p95': float(np.quantile(d,.95)), 'p99': float(np.quantile(d,.99)),
        'sfm_knn6_spacing_median': spacing, 'sfm_spacing_positive_fraction':float(len(positive)/len(kth)),
        'fraction_over_3x_spacing': float(np.mean(d > 3 * spacing)),
        'normalized_local_residual_p50':float(np.quantile(normalized, .50)),
        'normalized_local_residual_p90':float(np.quantile(normalized, .90)),
        'normalized_local_residual_p95':float(np.quantile(normalized, .95)),
        'fraction_local_residual_over_3x_spacing':float(np.mean(normalized > 3)),
        'spacing_calculated_using_full_core':True,
        'note': 'KNN spacing is measured against ALL core SfM points, including in preview. Local-normalized residual ratios are descriptive rather than ground-truth mesh accuracy.',
    }


def _load_core_sfm(path):
    with np.load(path, allow_pickle=False) as npz:
        xyz = np.asarray(npz['xyz'], dtype=np.float64)
        mask = np.asarray(npz['in_core_roi'], dtype=bool)
        if len(mask) != len(xyz) or xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError('Invalid V3 ROI SfM')
        return xyz[mask], int(len(xyz)), int(mask.sum())


def _geometry_record(manifest, clip_report=None, *, phase='check', max_fit_points=0,
                     colmap_images=None, config_file=None):
    import trimesh
    import yaml
    from gsrefine.cleanup.sdf_common import _load_stage1_manifest
    stage1_path, upstream, inputs, run_dir = _load_stage1_manifest(manifest)
    run_dir = Path(run_dir)
    mesh_path, report_path, clip = select_direct_r3_clip(run_dir, selected_report=clip_report)
    sfm_path = run_dir / 'cleanup/inputs/sfm_roi_local_v3.npz'
    if not sfm_path.is_file():
        declared = (upstream.get('stage2_inputs') or {}).get('sfm_roi_npz')
        sfm_path = Path(declared).resolve(strict=True) if declared else sfm_path
    if not sfm_path.is_file():
        raise FileNotFoundError('Verified R3 V3 SfM missing. Run stage2-sfm first.')
    sources = validate_sources(stage1_path, upstream, inputs, run_dir, clip, sfm_path)
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError('Invalid direct R3 candidate mesh')
    original_topology = boundary_diagnostics(mesh)
    known_topology = clip.get('candidate_topology') or {}
    for key, value in (('faces', original_topology['mesh_faces']),
                       ('open_boundary_edges', original_topology['open_boundary_edges']),
                       ('nonmanifold_edges', original_topology['nonmanifold_edges'])):
        if key in known_topology and known_topology[key] != value:
            raise ValueError('Clip output topology differs from authenticated experiment report: '+key)
    core_xyz, sfm_n, core_n = _load_core_sfm(sfm_path)
    # Canonical main/aux partition: no face removal, welding, smoothing,
    # vertex motion, new Poisson or local remeshing.  A separate auxiliary
    # observation layer MUST survive even though it cannot be part of the
    # single-rock sealing body.
    from gsrefine.geometry.r3_main_auxiliary import (
        PARTITION_REVISION, split_direct_r3_mesh, auxiliary_evidence)
    from gsrefine.roi.core import GaussianPLY
    partition = split_direct_r3_mesh(mesh)
    main = partition['main']
    topo = boundary_diagnostics(main)
    if topo['mesh_faces'] != partition['main_face_count']:
        raise RuntimeError('Main component extraction failed')
    model = GaussianPLY(inputs['gaussian_ply'])
    gaussian_xyz = model.positions()
    scale_max = np.maximum.reduce([np.asarray(model.arr['scale_%d' % j],
                  dtype=np.float64) for j in range(3)])
    gaussian_radii = 3.0 * np.exp(scale_max)
    if not np.isfinite(gaussian_radii).all() or np.any(gaussian_radii <= 0):
        raise ValueError('Invalid Gaussian 3-sigma major axes')
    cfgpath = Path(config_file).resolve(strict=True) if config_file else (run_dir/'metadata/effective_scene.yaml').resolve(strict=True)
    opts = {**DEFAULTS, **((yaml.safe_load(cfgpath.read_text(encoding='utf8')) or {}).get('stage2_cleanup') or {})}
    aux_evidence = auxiliary_evidence(
        partition, core_xyz, gaussian_xyz, gaussian_radii,
        sfm_support_factor=float(opts['aux_sfm_support_factor']),
        gaussian_support_factor=float(opts['aux_gaussian_support_factor']),
        min_unique_points=int(opts['aux_min_unique_points']),
        min_face_coverage=float(opts['aux_min_face_coverage']))
    fit = sfm_fit(main, core_xyz, max_points=max_fit_points)
    if int(opts['nsamples']) not in (1,3,5,7,9,11):
        raise ValueError('SDF nsamples must be odd between 1 and 11')
    if int(opts['chunk_size']) < 1:
        raise ValueError('Invalid SDF chunk_size')
    # Preserve the previous SDF safeguard semantics: ROI Gaussian
    # extent, NOT the observed mesh's bounding box.
    if not np.isfinite(gaussian_xyz).all():
        raise ValueError('Nonfinite Stage-1 Gaussian centers')
    diag = float(np.linalg.norm(np.ptp(gaussian_xyz, axis=0)))
    if not diag > 0:
        raise ValueError('Invalid Stage-1 Gaussian ROI diagonal')
    p95 = fit['p95']
    limits = {
        'max_by_sfm_spacing': float(opts['max_p95_to_sfm_spacing']) * fit['sfm_knn6_spacing_median'],
        'max_by_roi_diagonal': float(opts['max_p95_to_roi_diagonal']) * diag,
    }
    # SfM density is extremely nonuniform in steep, occluded rock scenes.
    # A global KNN median is NOT a reconstruction-accuracy certificate.
    # Keep the old 3x value for transparent warning/ablation only.
    global_spacing_warning = p95 > limits['max_by_sfm_spacing']
    fit_pass = bool(np.isfinite(p95) and p95 <= limits['max_by_roi_diagonal'])
    if phase != 'check' and not fit['is_full_core_fit']:
        raise ValueError('Sealing/SDF requires P95 from ALL R3 core SfM points')
    identity = {
        'revision': REVISION, 'stage1_manifest_sha256': _sha(stage1_path),
        'clip_report_sha256': _sha(report_path), 'clip_mesh_sha256': _sha(mesh_path),
        'core_sfm_sha256': _sha(sfm_path), 'effective_config_sha256': _sha(cfgpath),
        'options': {k: opts[k] for k in sorted(DEFAULTS)},
    }
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode('utf8')).hexdigest()
    report = {
        'schema': 'GSSceneRefine.direct_r3_stage2.v1',
        'status': 'PRESEAL_VALIDATION', 'phase': phase,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'signature': signature, 'sources': sources,
        'stage1_manifest': str(stage1_path), 'stage1_gaussian': str(inputs['gaussian_ply']),
        'direct_r3_clip_report': str(report_path), 'observed_direct_r3_mesh': str(mesh_path),
        'sfm': str(sfm_path), 'sfm_build_count': sfm_n, 'sfm_core_count': core_n,
        'topology_original': original_topology,
        'topology_partition': {
            'schema':'GSSceneRefine.r3_main_auxiliary_partition.v1',
            'partition_algorithm':PARTITION_REVISION,
            'source_face_count':partition['main_face_count']+partition['auxiliary_face_count'],
            'main_face_count':partition['main_face_count'],
            'auxiliary_face_count':partition['auxiliary_face_count'],
            'auxiliary_component_count':partition['auxiliary_component_count'],
            'main_vertex_fan_issue_count':partition['main_vertex_fan_issue_count'],
            'main_vertex_fan_issue_ids':partition['main_vertex_fan_issue_ids'],
            'original_r3_boundary_preserved':partition['original_r3_boundary_preserved'],
            'original_open_edge_count':partition['original_open_edge_count'],
            'main_open_edge_count':partition['main_open_edge_count'],
            'auxiliary_observation_evidence':aux_evidence,
            'warning':'Auxiliary features are saved, not rejected; multi-source support is not semantic proof.'},
        'topology': topo, 'sfm_fit': fit, 'sfm_fit_safeguards': limits,
        'global_spacing_warning_not_veto':bool(global_spacing_warning),
        'fit_safeguard_rule':'P95 <= 0.05 * R3 Gaussian ROI diagonal; historical P95 <= 3 * global SfM spacing is WARN ONLY; local-normalized residual distribution reported separately.',
        'stage1_gaussian_roi_diagonal':diag,
        'boundary_origin_diagnostic': 'Boundary loops/branching are measured, but R3-origin vs original-hole origin is not yet certified; do not infer from visualization alone.',
        'sfm_fit_safeguards_passed': fit_pass,
        'preseal_eligible': bool(topo['preseal_one_ring_eligible']
                                and partition['original_r3_boundary_preserved']
                                and partition['main_vertex_fan_issue_count']==0 and fit_pass),
        'warning': 'Geometric preseal eligibility is NOT watertight, physical rock identity, or SDF certification.',
    }
    return report, main, core_xyz, inputs, run_dir, opts, cfgpath, partition


def _load_camera(run_dir, inputs, cfgpath, override):
    from gsrefine.cleanup.sdf_common import _camera_path
    from gsrefine.geometry.orientation import read_colmap_poses
    annotation = _read(inputs['roi_json'])
    if annotation.get('annotation_metadata', {}).get('orientation_verified') is not True:
        raise ValueError('Stage-1 ROI orientation not verified')
    path = _camera_path(run_dir, annotation, cfgpath, override)
    positions, *_ = read_colmap_poses(path)
    up = np.asarray(annotation['annotation_metadata']['up_vector'], dtype=np.float64)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) == 0:
        raise ValueError('Invalid COLMAP verified up vector')
    return positions, up / np.linalg.norm(up), path


def _tilted_sealing_projections(mesh, verified_up, opts):
    """Read-only search for simple projections of an actual 3D open boundary.

    NEVER modifies observed vertices or quietly accepts a crossing polygon.
    The returned directions are only *candidates*: skirt/cap intersection,
    oriented volume and camera placement must still pass attempt_seal().
    """
    from shapely.geometry import Polygon
    from shapely.validation import explain_validity
    from gsrefine.cleanup.sdf_common import _local_basis
    from gsrefine.geometry.sdf_sealing import boundary_loops

    maximum = float(opts['seal_max_tilt_deg'])
    increment = float(opts['seal_tilt_step_deg'])
    azimuths = int(opts['seal_tilt_azimuths'])
    budget = int(opts['seal_max_tilt_trials'])
    if not (0 < maximum <= 30 and 0 < increment <= maximum and
            4 <= azimuths <= 72 and 1 <= budget <= 24):
        raise ValueError('Invalid conservative seal tilt search settings; maximum tilt <= 30 degrees')
    loops, topo = boundary_loops(mesh)
    if len(loops) != 1:
        raise ValueError('Tilted-cap trial requires exactly one oriented open boundary ring')
    ring = loops[0]
    points = np.asarray(mesh.vertices, dtype=np.float64)[ring]
    B0 = _local_basis(verified_up)
    tangent_x, tangent_y = B0[:,0], B0[:,1]
    up = B0[:,2]
    if float(np.dot(up, verified_up)) < .999999:
        # No hidden flip of verified COLMAP up is allowed.
        raise ValueError('COLMAP-up basis disagrees with the verified vertical direction')

    results=[]; failures=[]; valid_count=0; examined=0
    # Only try planes *near* the originally verified vertical.  Larger
    # tilts may create arbitrary artificial volumes and are prohibited.
    theta_values = np.arange(increment, maximum+1e-8, increment)
    for angle in theta_values:
        rad = np.deg2rad(float(angle))
        for j in range(azimuths):
            az = 2*np.pi*j/azimuths
            normal = np.cos(rad)*up + np.sin(rad)*(np.cos(az)*tangent_x + np.sin(az)*tangent_y)
            normal /= np.linalg.norm(normal)
            B = _local_basis(normal)
            uv = points @ B[:,:2]
            poly = Polygon(uv)
            span = np.ptp(uv, axis=0)
            area_limit = 1e-6*float(np.prod(span))
            examined += 1
            if poly.is_empty or not poly.is_valid or not poly.exterior.is_simple or poly.area <= area_limit:
                if len(failures) < 8:
                    failures.append({'tilt_deg':float(angle), 'azimuth_deg':float(np.rad2deg(az)),
                                     'reason':explain_validity(poly)})
                continue
            # Nearly tangent projected edges are unsafe for a vertical skirt.
            clearance = float(poly.minimum_clearance)
            if not np.isfinite(clearance) or clearance <= 1e-7*max(float(np.linalg.norm(span)), 1.0):
                if len(failures) < 8:
                    failures.append({'tilt_deg':float(angle), 'azimuth_deg':float(np.rad2deg(az)),
                                     'reason':'Projected ring is nearly tangent'})
                continue
            valid_count += 1
            results.append({'tilt_deg':float(angle), 'azimuth_deg':float(np.rad2deg(az)),
                            'normal_xyz':normal.tolist(), 'clearance':clearance,
                            'projected_area':float(poly.area), 'basis':B})
    results.sort(key=lambda x:(x['tilt_deg'], -x['clearance']))
    diag = {'reason':'verified_up_projection_is_self_intersecting',
            'search_max_tilt_deg':maximum,'search_step_deg':increment,
            'search_azimuths':azimuths,'directions_examined':examined,
            'simple_projections_found':valid_count,
            'trial_budget':budget,'invalid_projection_examples':failures,
            'warning':'A simple projected boundary is necessary, NOT sufficient; '
                      'the artificial tilted skirt/cap still requires self-intersection, '
                      'watertight volume, and camera validation. '
                      'No original observed vertex is changed.'}
    return results[:budget], diag


def attempt_seal(observed, camera_xyz, up, opts):
    """Fail-closed sealing. Exact verified COLMAP Up first; bounded tilt only
    if the original 2D projection is invalid. Never move observed vertices.
    """
    from gsrefine.cleanup.sdf_common import _local_basis, _raycast, _camera_check
    from gsrefine.geometry.sdf_sealing import seal_main
    import open3d as o3d
    repaired = observed.copy()
    repair_report = {'source':'untouched_largest_face_component_of_direct_r3',
                     'second_welding_or_face_pruning':False}
    if not repaired.is_winding_consistent:
        import trimesh
        trimesh.repair.fix_winding(repaired)
        repair_report['winding_repaired_on_copy'] = True
    after = boundary_diagnostics(repaired)
    if not after['preseal_one_ring_eligible']:
        raise ValueError('Main R3 component topology is not suitable for single-ring seal: '+str(after))
    up = np.asarray(up, dtype=np.float64)
    if up.shape != (3,) or not np.all(np.isfinite(up)) or not np.linalg.norm(up) > 0:
        raise ValueError('Invalid verified COLMAP Up direction')
    up = up / np.linalg.norm(up)
    trials = []

    def try_projection(B, *, tilt_deg, azimuth_deg):
        oriented = repaired.copy()
        oriented.vertices = np.asarray(repaired.vertices) @ B
        try:
            span = float(np.ptp(np.asarray(oriented.vertices)[:,2]))
            if not span > 0:
                raise ValueError('No vertical span in proposed sealing plane')
        except Exception as exc:
            return [({'cap_direction_relative_to_candidate_up':sign,
                     'cap_direction_relative_to_verified_up':sign,
                     'tilt_deg':tilt_deg,'azimuth_deg':azimuth_deg,
                     'passed':False,'failure':type(exc).__name__+': '+str(exc)}, None)
                    for sign in (1,-1)]
        outs = []
        for sign in (1,-1):
            item = {'cap_direction_relative_to_candidate_up':sign,
                    'cap_direction_relative_to_verified_up':sign,
                    'tilt_deg':float(tilt_deg), 'azimuth_deg':float(azimuth_deg),
                    'sealing_plane_normal_xyz':np.asarray(B[:,2]).tolist(),
                    'cap_is_artificial_tilted_plane':bool(tilt_deg>0)}
            try:
                mesh, geom, artificial = seal_main(oriented.copy(), 2, sign,
                           float(opts['bottom_margin_fraction']) * span)
                if artificial is None:
                    raise ValueError('Source should be open, but returned no artificial closure')
                # Numerical rotations may introduce 1e-15 coordinate drift.
                # Restore original, observed vertex coordinates exactly; ONLY
                # artificial skirt/cap vertices are allowed to be new.
                original_vertices = np.asarray(repaired.vertices, dtype=np.float64)
                if int(geom.get('mesh_before_vertices', -1)) != len(original_vertices):
                    raise ValueError('Sealer changed the observed source vertex count')
                if int(geom.get('mesh_before_faces', -1)) != len(repaired.faces):
                    raise ValueError('Sealer changed the observed source triangle count')
                world_vertices = np.asarray(mesh.vertices) @ B.T
                if len(world_vertices) < len(original_vertices):
                    raise ValueError('Sealer deleted original observed vertices')
                world_vertices[:len(original_vertices)] = original_vertices
                mesh.vertices = world_vertices
                # No relaxed checks: an altered skirt direction must pass all
                # the exact same geometry/volume/camera gates as the original.
                check_mesh=o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(np.asarray(mesh.vertices, np.float64)),
                    o3d.utility.Vector3iVector(np.asarray(mesh.faces, np.int32)))
                self_intersects=bool(check_mesh.is_self_intersecting())
                vertex_manifold=bool(check_mesh.is_vertex_manifold())
                edge_manifold=bool(check_mesh.is_edge_manifold(allow_boundary_edges=False))
                item['sealed_mesh_geometry_audit']={
                    'self_intersecting':self_intersects,
                    'vertex_manifold':vertex_manifold,
                    'edge_manifold_closed':edge_manifold,
                    'is_volume':bool(mesh.is_volume),
                    'is_watertight':bool(mesh.is_watertight),
                    'winding_consistent':bool(mesh.is_winding_consistent),
                    'body_count':int(mesh.body_count)}
                scene = _raycast(o3d, mesh)
                cam = _camera_check(o3d, scene, camera_xyz, int(opts['nsamples']))
                passed = (not self_intersects and vertex_manifold and edge_manifold
                          and mesh.is_watertight and mesh.is_winding_consistent and mesh.is_volume
                          and mesh.body_count == 1
                          and cam['nonfinite_distances'] == 0
                          and cam['outside_fraction'] >= float(opts['min_camera_outside_fraction'])
                          and cam['outward_normal_agreement'] >= float(opts['min_camera_normal_agreement']))
                item.update(closure=geom, camera=cam, passed=bool(passed))
                outs.append((item, mesh if passed else None))
            except Exception as exc:
                item.update(passed=False, failure=type(exc).__name__ + ': ' + str(exc))
                outs.append((item, None))
        return outs

    B0 = _local_basis(up)
    primary = try_projection(B0, tilt_deg=0., azimuth_deg=0.)
    trials.extend(primary)
    good = [(a,b) for a,b in trials if a['passed']]
    projection_audit = {'verified_up_projection_trial_count':2,
                        'bounded_tilt_fallback_triggered':False}
    # IMPORTANT: Tilting the artificial bottom is allowed only when the
    # original projection is unsuited to a straight skirt.  It is NOT a
    # workaround for failed camera or true 3D self-intersection tests.
    projection_errors=('Outer boundary projection is invalid or collapses',
                       'Projected boundary is not simple',
                       'Repeated projected boundary vertices')
    invalid_projection = (not good and len(primary)==2 and
                          all(any(msg in x[0].get('failure','') for msg in projection_errors)
                              for x in primary))
    if invalid_projection:
        orientations, scan = _tilted_sealing_projections(repaired, up, opts)
        projection_audit.update(scan)
        projection_audit['bounded_tilt_fallback_triggered']=True
        for candidate in orientations:
            B = candidate['basis']
            attempts = try_projection(B, tilt_deg=candidate['tilt_deg'],
                                      azimuth_deg=candidate['azimuth_deg'])
            trials.extend(attempts)
            good.extend((a,b) for a,b in attempts if a['passed'])
            if good:
                break
    if not good:
        return None, {'topology_preparation': repair_report, 'repaired_topology': after,
                      'projection_audit':projection_audit,
                      'direction_trials': [x for x,_ in trials],
                      'passed': False, 'status': 'BLOCKED_SEALING'}
    good.sort(key=lambda t:(-t[0]['tilt_deg'],
                            t[0]['camera']['outside_fraction'],
                            t[0]['camera']['outward_normal_agreement']),reverse=True)
    item, selected = good[0]
    return selected, {'topology_preparation': repair_report, 'repaired_topology': after,
                      'projection_audit':projection_audit,
                      'direction_trials': [x for x,_ in trials],
                      'selected_direction':item['cap_direction_relative_to_candidate_up'],
                      'selected_sealing_normal_xyz':item['sealing_plane_normal_xyz'],
                      'selected_tilt_degrees':item['tilt_deg'],
                      'camera_validation':item['camera'],
                      'original_main_triangles_in_sealed':int(item['closure']['mesh_before_faces']),
                      'passed': True, 'status': 'SEALED_VALIDATED'}



def _persist_main_auxiliary_partition(record, part, main_file, aux_file,
                                      partition_file, face_map_file):
    """Persist immutable, hash-checked main/aux PLYs with exact source face IDs."""
    present=[p.exists() for p in (main_file,partition_file,face_map_file)]
    if any(present) or aux_file.exists():
        if not all(present) or (part['auxiliary'] is not None)!=aux_file.is_file():
            raise FileExistsError('Incomplete main/aux geometry cache; do not overwrite')
        old=_read(partition_file)
        if old.get('signature')!=record['signature']:
            raise ValueError('Old main/aux partition cache signature differs')
        if old.get('source_direct_r3_sha256')!=_sha(record['observed_direct_r3_mesh']):
            raise ValueError('Main/aux cache input hash mismatch')
        if old.get('main_sha256')!=_sha(main_file):
            raise ValueError('Main Poisson changed in partition cache')
        if part['auxiliary'] is not None and old.get('auxiliary_sha256')!=_sha(aux_file):
            raise ValueError('Auxiliary Poisson changed in partition cache')
        if old.get('face_map_sha256')!=_sha(face_map_file):
            raise ValueError('Main/aux face mapping changed in partition cache')
        with np.load(face_map_file,allow_pickle=False) as f:
            for key in ('main_original_face_ids', 'auxiliary_original_face_ids',
                        'auxiliary_face_component_ids'):
                if not np.array_equal(f[key],part[key]):
                    raise ValueError('Main/aux source face assignment differs from cached partition')
        return
    # No in-place repair; each output preserves the ORIGINAL retained vertices.
    main_file.parent.mkdir(parents=True, exist_ok=True)
    part['main'].export(main_file,file_type='ply',encoding='binary')
    if part['auxiliary'] is not None:
        part['auxiliary'].export(aux_file,file_type='ply',encoding='binary')
    np.savez_compressed(face_map_file,
        main_original_face_ids=part['main_original_face_ids'],
        auxiliary_original_face_ids=part['auxiliary_original_face_ids'],
        auxiliary_face_component_ids=part['auxiliary_face_component_ids'])
    import trimesh
    roundtrip=trimesh.load(main_file,force='mesh',process=False)
    if not np.array_equal(np.asarray(roundtrip.vertices),np.asarray(part['main'].vertices)):
        raise ValueError('Main vertices shifted during PLY serialization')
    if not np.array_equal(np.asarray(roundtrip.faces),np.asarray(part['main'].faces)):
        raise ValueError('Main faces changed during PLY serialization')
    if part['auxiliary'] is not None:
        checked=trimesh.load(aux_file,force='mesh',process=False)
        if not np.array_equal(np.asarray(checked.vertices),np.asarray(part['auxiliary'].vertices)):
            raise ValueError('Auxiliary vertices shifted during PLY serialization')
        if not np.array_equal(np.asarray(checked.faces),np.asarray(part['auxiliary'].faces)):
            raise ValueError('Auxiliary faces changed during PLY serialization')
    payload={
        'schema':'GSSceneRefine.r3_main_auxiliary_partition.v1',
        'signature':record['signature'],
        'source_direct_r3':record['observed_direct_r3_mesh'],
        'source_direct_r3_sha256':_sha(record['observed_direct_r3_mesh']),
        'main':str(main_file),'main_sha256':_sha(main_file),
        'auxiliary':str(aux_file) if part['auxiliary'] is not None else None,
        'auxiliary_sha256':_sha(aux_file) if part['auxiliary'] is not None else None,
        'face_map':str(face_map_file),'face_map_sha256':_sha(face_map_file),
        'topology_original':record['topology_original'],
        'topology_main':record['topology'],
        'partition':record['topology_partition'],
        'source_unchanged':True,
        'warning':'Auxiliary faces are preserved as observation-only geometry, not as part of main sealing solid.'
    }
    _write_json(partition_file,payload)


def combine_auxiliary_protection(base_keep, aux_distances, aux_component_ids,
                                 auxiliary_evidence, delta, epsilon):
    """Retain removed-by-main Gaussians near auxiliary observed geometry.

    Distributed multi-source evidence is reported as corroborated but not as
    proof of semantic correctness.  Weak/ambiguous observations are kept for
    REVIEW rather than silently discarded. Nothing outside the verified
    auxiliary support envelope is rescued.
    """
    orig_keep=np.asarray(base_keep,dtype=bool)
    dist=np.asarray(aux_distances,dtype=np.float64)
    comp=np.asarray(aux_component_ids,dtype=np.int64)
    if orig_keep.shape!=dist.shape or comp.shape!=dist.shape:
        raise ValueError('Auxiliary SDF arrays have inconsistent shapes')
    eligible=(~orig_keep)&np.isfinite(dist)&(dist<=delta+epsilon)&(comp>=0)
    supported=np.zeros(len(orig_keep),dtype=bool)
    review=np.zeros(len(orig_keep),dtype=bool)
    for obj in auxiliary_evidence:
        matching=eligible&(comp==int(obj['component_id']))
        if obj['observation_evidence']=='distributed_both_sources_and_separated':
            supported|=matching
        else:
            review|=matching
    # Unknown component labels MUST not be released as supported.
    uncovered=eligible&~(supported|review)
    review|=uncovered
    return orig_keep|supported|review, supported, review

def _persist_seal(report, sealed, base):
    folder = Path(base)
    folder.mkdir(parents=True, exist_ok=True)
    final = folder/'sealed_poisson.ply'
    meta = folder/'sealing_report.json'
    if final.exists() or meta.exists():
        if not final.is_file() or not meta.is_file():
            raise FileExistsError('Incomplete direct R3 seal cache; cannot overwrite')
        old = _read(meta)
        if old.get('signature') != report['signature'] or _sha(final) != old.get('sealed_poisson_sha256'):
            raise ValueError('Direct R3 seal cache provenance mismatch')
        return final, old, True
    temp = folder/'sealed_poisson.writing.ply'
    try:
        sealed.export(temp, file_type='ply', encoding='binary')
        check = __import__('trimesh').load(temp, process=False, force='mesh')
        if not check.is_watertight or not check.is_volume or check.body_count != 1:
            raise ValueError('Sealed mesh lost topological integrity after PLY export')
        os.replace(temp, final)
        report = {**report, 'sealed_poisson': str(final), 'sealed_poisson_sha256': _sha(final)}
        _write_json(meta, report)
        return final, report, False
    finally:
        temp.unlink(missing_ok=True)


def _compute_sdf(sealed, gaussian_ply, fit, sealing_info, opts, *, auxiliary=None,
                 auxiliary_component_ids=None, auxiliary_evidence=None,
                 keep_diagnostics=False, output_dir=None):
    import open3d as o3d
    from gsrefine.cleanup.sdf_common import _raycast, robust_gaussian_radius, classify_direct_sdf
    from gsrefine.roi.core import GaussianPLY
    model = GaussianPLY(gaussian_ply)
    coords = model.positions()
    if not np.isfinite(coords).all():
        raise ValueError('Nonfinite Gaussian ROI centers')
    radius = robust_gaussian_radius(model, float(opts['trim_fraction']), float(opts['sigma_multiplier']))
    delta = float(fit['p95']) + float(radius['trimmed_mean_radius'])
    epsilon = float(opts['epsilon'])
    scene = _raycast(o3d, sealed)
    aux_evidence=list(auxiliary_evidence or [])
    aux_scene=_raycast(o3d, auxiliary) if auxiliary is not None else None
    aux_face_labels=np.asarray(auxiliary_component_ids if auxiliary_component_ids is not None else [],dtype=np.int32)
    if auxiliary is not None and len(aux_face_labels)!=len(auxiliary.faces):
        raise ValueError('Auxiliary component labels do not match saved auxiliary geometry')
    signed = np.empty(len(coords), dtype=np.float32)
    aux_distances=np.full(len(coords),np.inf,dtype=np.float32)
    aux_nearest_component=np.full(len(coords),-1,dtype=np.int32)
    artificial = np.empty(len(coords), dtype=bool)
    first_artificial = sealing_info['original_main_triangles_in_sealed']
    chunk = int(opts['chunk_size'])
    for start in range(0, len(coords), chunk):
        stop = min(start + chunk, len(coords))
        q = o3d.core.Tensor(np.ascontiguousarray(coords[start:stop], np.float32))
        signed[start:stop] = scene.compute_signed_distance(q, nsamples=int(opts['nsamples'])).numpy().astype(np.float32)
        if aux_scene is not None:
            # Evaluate auxiliary proximity ONLY for Gaussians that the main
            # sealed rock would reject. No auxiliary signed distance is used.
            candidate=(np.isfinite(signed[start:stop]) & (signed[start:stop]>delta+epsilon))
            if candidate.any():
                sub=o3d.core.Tensor(np.ascontiguousarray(coords[start:stop][candidate],np.float32))
                distances=aux_scene.compute_distance(sub).numpy().astype(np.float32)
                nearest=aux_scene.compute_closest_points(sub)['primitive_ids'].numpy().astype(np.int64).ravel()
                safe=(nearest>=0)&(nearest<len(aux_face_labels))
                components=np.full(len(nearest),-1,dtype=np.int32)
                components[safe]=aux_face_labels[nearest[safe]]
                target=np.flatnonzero(candidate)+start
                aux_distances[target]=distances
                aux_nearest_component[target]=components
        closest = scene.compute_closest_points(q)
        artificial[start:stop] = closest['primitive_ids'].numpy().reshape(-1) >= first_artificial
        if start == 0 or stop == len(coords) or start//chunk % 10 == 0:
            print(f'[stage2-cleanup] Direct SDF Gaussian centers {stop:,}/{len(coords):,}', flush=True)
    main_kept, _, uncertain = classify_direct_sdf(signed, delta, epsilon)
    kept, aux_supported, aux_review = combine_auxiliary_protection(
        main_kept, aux_distances, aux_nearest_component, aux_evidence, delta, epsilon)
    removed=~kept
    protected_by_aux=aux_supported|aux_review
    aux_stats=[]
    for obj in aux_evidence:
        cid=int(obj['component_id'])
        aux_stats.append({'component_id':cid,'observation_evidence':obj['observation_evidence'],
                          'protected_gaussians':int((protected_by_aux&(aux_nearest_component==cid)).sum()),
                          'requires_review':obj['observation_evidence']!='distributed_both_sources_and_separated'})
    output = {
        'method': 'direct_r3_main_sealed_sdf_auxiliary_observation_protection',
        'main_sdf_only_kept_count':int(main_kept.sum()),
        'auxiliary_protected_distributed_support_count':int(aux_supported.sum()),
        'auxiliary_protected_review_count':int(aux_review.sum()),
        'auxiliary_component_stats':aux_stats,
        'auxiliary_rule':'A main-SDF removal is retained ONLY within distance delta of preserved auxiliary triangles; multi-sample SfM+Gaussian support is distinguished from unresolved review, never silently deleted.',
        'original_roi_count': len(coords), 'kept_count': int(kept.sum()),
        'removed_count': int(removed.sum()), 'removed_fraction': float(removed.mean()),
        'unknown_distances_kept': int(uncertain.sum()),
        'removed_nearest_artificial_face_count': int((removed & artificial).sum()),
        'removed_nearest_artificial_face_fraction': float((removed & artificial).sum()/max(int(removed.sum()), 1)),
        'sdf_finite_p05': float(np.quantile(signed[np.isfinite(signed)],.05)) if np.isfinite(signed).any() else None,
        'sdf_finite_p50': float(np.median(signed[np.isfinite(signed)])) if np.isfinite(signed).any() else None,
        'sdf_finite_p95': float(np.quantile(signed[np.isfinite(signed)],.95)) if np.isfinite(signed).any() else None,
        'delta': delta, 'sfm_p95': float(fit['p95']), 'gaussian_scale': radius,
        'epsilon':epsilon, 'nsamples': int(opts['nsamples']),
        'criterion': 'Keep when main SDF <= delta+epsilon, nonfinite, or auxiliary observation distance <= delta+epsilon (supported/review separately).',
        'warning': 'Auxiliary support is not semantic truth. Review kept uncertain structures and artificial skirt/cap proximity before provisional export.',
    }
    if keep_diagnostics:
        base = Path(output_dir)
        base.mkdir(parents=True, exist_ok=True)
        np.save(base/'signed_distances.npy',signed)
        np.savez_compressed(base/'classification_masks.npz', keep=kept, remove=removed,
                            uncertain_kept=uncertain, nearest_artificial_face=artificial,
                            auxiliary_supported_keep=aux_supported, auxiliary_review_keep=aux_review,
                            auxiliary_distances=aux_distances,
                            nearest_auxiliary_component_id=aux_nearest_component)
    return kept, output


def run(*, manifest, phase='check', clip_report=None, colmap_images=None,
        config_file=None, fit_max_points=0, keep_diagnostics=False,
        chunk_size=None, accept_provisional=False, preseal_report=None,
        output_dir=None, winding_camera_max_delta=0.20, min_unobstructed_cameras=2,
        b_method='implicit-volume',voxel_size=None,max_volume_nodes=240000,
        max_sfm_rays=130000,diagnostic_only=False,
        barrier_audit_dir=None,index_cache=None,classification_only=False,
        confirm_exterior=False,nearest_k=8,min_path_agree=2,
        exterior_seed=None,roi_link_samples=3,ray_batch=8000):
    if phase == 'open-ab':
        if not preseal_report or not output_dir:
            raise ValueError('open-ab needs --preseal-report and --output-dir')
        if fit_max_points not in (0, None):
            raise ValueError('open-ab requires full SfM P95, never a preview sample')
        if b_method == 'roi-barrier-parity':
            from gsrefine.cleanup.roi_barrier_parity import run_roi_barrier_parity
            return run_roi_barrier_parity(
                manifest=manifest,preseal_report=preseal_report,output_dir=output_dir,
                barrier_audit_dir=barrier_audit_dir,index_cache=index_cache,
                classification_only=classification_only,confirm_exterior=confirm_exterior,
                chunk_size=int(chunk_size or 30000),nearest_k=nearest_k,
                min_agree=min_path_agree,exterior_seed=exterior_seed,
                config_file=config_file,roi_link_samples=roi_link_samples,ray_batch=ray_batch)
        if classification_only or confirm_exterior or barrier_audit_dir or index_cache or exterior_seed:
            raise ValueError('ROI parity options require --b-method roi-barrier-parity')
        from gsrefine.cleanup.open_ab_main import run_open_ab_main
        return run_open_ab_main(
            manifest=manifest, preseal_report=preseal_report,
            output_dir=output_dir, chunk_size=int(chunk_size or 30000),
            colmap_images=colmap_images, config_file=config_file,
            winding_camera_max_delta=float(winding_camera_max_delta),
            min_unobstructed_cameras=int(min_unobstructed_cameras),
            b_method=b_method,voxel_size=voxel_size,max_volume_nodes=max_volume_nodes,
            max_sfm_rays=max_sfm_rays,diagnostic_only=diagnostic_only)
    if phase not in ('check','seal','sdf','export'):
        raise ValueError('Invalid phase')
    if phase != 'check' and fit_max_points not in (0, None):
        # Full core is mandatory for P95 that enters actual SDF.
        fit_max_points = 0
    record, observed, core_xyz, inputs, run_dir, opts, cfgpath, partition = _geometry_record(
        manifest, clip_report, phase=phase, max_fit_points=fit_max_points,
        colmap_images=colmap_images, config_file=config_file)
    if chunk_size is not None:
        opts['chunk_size'] = int(chunk_size)
        if opts['chunk_size'] <= 0:
            raise ValueError('chunk-size must be positive')
    base = Path(run_dir)/'cleanup/geometry/direct_r3_sdf'/record['signature'][:24]
    preflight = base/'preseal_diagnostics.json'
    # Persist BOTH geometry layers and the exact original face assignment in a
    # new revision-specific cache.  Prior repair caches remain untouched.
    base.mkdir(parents=True, exist_ok=True)
    main_file=base/'poisson_roi_main.ply'
    aux_file=base/'poisson_roi_auxiliary.ply'
    partition_file=base/'main_aux_partition_report.json'
    face_map_file=base/'main_aux_face_map.npz'
    _persist_main_auxiliary_partition(record, partition,
                                      main_file, aux_file, partition_file, face_map_file)
    record['main_observed_surface']=str(main_file)
    record['main_observed_surface_sha256']=_sha(main_file)
    record['auxiliary_observed_surface']=str(aux_file) if partition['auxiliary'] is not None else None
    record['auxiliary_observed_surface_sha256']=_sha(aux_file) if aux_file.is_file() else None
    record['main_auxiliary_report']=str(partition_file)
    record['main_auxiliary_face_map']=str(face_map_file)
    # A diagnostics file is saved even if the gate blocks sealing.
    if preflight.exists():
        old = _read(preflight)
        if old.get('signature') != record['signature']:
            raise ValueError('Preflight cache/source mismatch; do not overwrite')
        if old.get('sfm_fit',{}).get('is_full_core_fit') and not record['sfm_fit']['is_full_core_fit']:
            # Never replace a stronger full-core P95 audit with a sampled preview.
            pass
        else:
            _write_json(preflight, record)
    else:
        _write_json(preflight, record)
    if phase == 'check':
        status = 'ELIGIBLE_FOR_SEAL_TRIAL' if record['preseal_eligible'] else 'BLOCKED_PRESEAL'
        return {'status':status, 'preseal_report': str(preflight),
                'topology':record['topology'], 'sfm_fit':record['sfm_fit'],
                'main_surface':record.get('main_observed_surface'),
                'auxiliary_surface':record.get('auxiliary_observed_surface'),
                'partition_report':record.get('main_auxiliary_report'),
                'global_spacing_warning_not_veto':record.get('global_spacing_warning_not_veto'),
                'safeguards_passed':record['sfm_fit_safeguards_passed']}
    if not record['preseal_eligible']:
        return {'status':'BLOCKED_PRESEAL', 'preseal_report':str(preflight),
                'topology':record['topology'], 'sfm_fit':record['sfm_fit'],
                'main_surface':record.get('main_observed_surface'),
                'auxiliary_surface':record.get('auxiliary_observed_surface'),
                'partition_report':record.get('main_auxiliary_report')}
    if phase == 'export' and not accept_provisional:
        raise ValueError('Export of provisional geometry requires explicit --accept-provisional')
    positions, up, camera_path = _load_camera(run_dir, inputs, cfgpath, colmap_images)
    sealed_path = base/'sealed_poisson.ply'
    seal_meta = base/'sealing_report.json'
    if sealed_path.is_file() and seal_meta.is_file():
        sealing = _read(seal_meta)
        if sealing.get('signature') != record['signature'] or _sha(sealed_path) != sealing.get('sealed_poisson_sha256') or not sealing.get('passed') or sealing.get('camera_pose_sha256') != _sha(camera_path):
            raise ValueError('Existing direct R3 sealed mesh cache invalid')
        import trimesh
        solid = trimesh.load(sealed_path, process=False, force='mesh')
        if not solid.is_watertight or not solid.is_volume or solid.body_count != 1:
            raise ValueError('Cached sealed mesh topology invalid')
        cache_reused = True
    else:
        solid, sealing = attempt_seal(observed, positions, up, opts)
        if solid is None:
            _write_json(base/'failed_sealing_report.json', {
                'schema':'GSSceneRefine.direct_r3_sealing_failure.v1', 'signature':record['signature'],
                'preseal_report':str(preflight), **sealing})
            return {'status':'BLOCKED_SEALING', 'preseal_report':str(preflight),
                    'failure_report':str(base/'failed_sealing_report.json'),
                    'projection_audit':sealing.get('projection_audit'),
                    'trials':sealing['direction_trials']}
        sealing = {**sealing, 'signature':record['signature'],
                   'observed_direct_r3_mesh':record['observed_direct_r3_mesh'],
                   'candidate_sha256':_sha(record['observed_direct_r3_mesh']),
                   'main_observed_poisson':record.get('main_observed_surface'),
                   'main_observed_sha256':record.get('main_observed_surface_sha256'),
                   'auxiliary_observed_poisson':record.get('auxiliary_observed_surface'),
                   'sfm_fit_on_observed_surface':record['sfm_fit'],
                   'camera_pose_sha256':_sha(camera_path)}
        sealed_path, sealing, cache_reused = _persist_seal(sealing, solid, base)
    if phase == 'seal':
        return {'status':'SEALED_VALIDATED_READY_FOR_SDF',
                'sealed_poisson':str(sealed_path), 'sealing_report':str(seal_meta),
                'selected_tilt_degrees':sealing.get('selected_tilt_degrees'),
                'selected_sealing_normal_xyz':sealing.get('selected_sealing_normal_xyz'),
                'cache_reused':cache_reused}
    if phase == 'export':
        dest = Path(run_dir)/'cleanup/final/gaussian_roi.ply'
        metadata = Path(run_dir)/'cleanup/metadata/manifest.json'
        if dest.exists() or metadata.exists():
            raise FileExistsError('Stage-2 final output exists; will not replace existing result')
    else:
        dest = None
        metadata = None
    diagnostics_dir = Path(run_dir)/'cleanup/preview'/('direct_r3_'+record['signature'][:12])
    kept, stats = _compute_sdf(solid, inputs['gaussian_ply'], record['sfm_fit'], sealing,
                               opts, auxiliary=partition['auxiliary'],
                               auxiliary_component_ids=partition['auxiliary_face_component_ids'],
                               auxiliary_evidence=record['topology_partition']['auxiliary_observation_evidence'],
                               keep_diagnostics=keep_diagnostics, output_dir=diagnostics_dir)
    summary = {
        'schema':'GSSceneRefine.direct_r3_sdf_result.v1',
        'status':'diagnostics_only' if phase == 'sdf' else 'completed_provisional',
        'result_status':'PROVISIONAL_REQUIRES_VISUAL_VERIFICATION',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'signature':record['signature'], 'source_stage1_manifest':record['stage1_manifest'],
        'stage1_gaussian_sha256':_sha(inputs['gaussian_ply']),
        'observed_direct_r3_poisson':record['observed_direct_r3_mesh'],
        'observed_direct_r3_poisson_sha256':_sha(record['observed_direct_r3_mesh']),
        'main_observed_poisson':record.get('main_observed_surface'),
        'main_observed_poisson_sha256':record.get('main_observed_surface_sha256'),
        'auxiliary_observed_poisson':record.get('auxiliary_observed_surface'),
        'auxiliary_observed_poisson_sha256':record.get('auxiliary_observed_surface_sha256'),
        'main_auxiliary_partition_report':record.get('main_auxiliary_report'),
        'sealed_poisson':str(sealed_path),'sealed_poisson_sha256':_sha(sealed_path),
        'sfm_fit':record['sfm_fit'], 'topology_preseal':record['topology'],
        'auxiliary_component_count':partition['auxiliary_component_count'],
        'sealing_report':str(seal_meta), 'gaussian_sdf':stats,
        'final_gaussian':str(dest) if dest else None,
    }
    if phase == 'export':
        from gsrefine.roi.core import GaussianPLY
        ply = GaussianPLY(inputs['gaussian_ply'])
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_name(dest.name+'.writing.tmp')
        try:
            written = ply.export(kept, temp)
            out = GaussianPLY(temp)
            if written != int(kept.sum()) or out.dtype != ply.dtype or out.vertex_count != written:
                raise RuntimeError('Gaussian original attribute preservation failed')
            if written:
                probe = np.unique(np.linspace(0, written - 1, min(32,written),dtype=int))
                idx = np.flatnonzero(kept)[probe]
                if out.arr[probe].tobytes() != ply.arr[idx].tobytes():
                    raise RuntimeError('Gaussian byte-for-byte sample preservation failed')
            del out
            os.replace(temp,dest)
        finally:
            temp.unlink(missing_ok=True)
        summary['output_sha256'] = _sha(dest)
        summary['output_bytes'] = dest.stat().st_size
        _write_json(metadata, summary)
    else:
        _write_json(diagnostics_dir/'cleanup_report.json',summary)
    return {'status':summary['status'], 'report':str(metadata or diagnostics_dir/'cleanup_report.json'),
            'kept':stats['kept_count'], 'removed':stats['removed_count'],
            'removed_fraction':stats['removed_fraction'],
            'delta':stats['delta'], 'sealed_poisson':str(sealed_path),
            'final_gaussian':str(dest) if dest else None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--phase',choices=['check','seal','sdf','export','open-ab'],default='check')
    p.add_argument('--clip-report',default=None)
    p.add_argument('--preseal-report',default=None,help='Verified main-surface preseal JSON, required by open-ab')
    p.add_argument('--output-dir',default=None,help='Required open-ab preview output root; never writes Stage-1/official final')
    p.add_argument('--b-method',choices=['implicit-volume','camera-legacy','roi-barrier-parity'],default='implicit-volume',
                   help='V2 estimated voxel spatial occupancy (default) or old camera voting ablation')
    p.add_argument('--barrier-audit-dir',default=None,help='Existing verified final R3/global Poisson flood audit; default: create inside --output-dir')
    p.add_argument('--index-cache',default=None,help='Previously built ROI parity index; full source SHA checks required')
    p.add_argument('--classification-only',action='store_true',help='Only masks/diagnostic PLYs; do not write cleaned Gaussian')
    p.add_argument('--confirm-exterior',action='store_true',help='Researcher checked exterior anchors/preview, permit clean PLY export')
    p.add_argument('--nearest-k',type=int,default=8,help='Nearby parity index nodes checked with original-triangle short rays')
    p.add_argument('--min-path-agree',type=int,default=2,help='Independent valid short ray paths required to agree')
    p.add_argument('--exterior-seed',nargs=3,type=float,default=None,metavar=('X','Y','Z'),
                   help='Optional scene-verified ROI exterior point, otherwise highest R3 shell is provisional')
    p.add_argument('--roi-link-samples',type=int,default=3,help='Samples along each path checked against FINAL R3')
    p.add_argument('--ray-batch',type=int,default=8000,help='Original Poisson intersection query batch size')
    p.add_argument('--voxel-size',type=float,default=None,help='Requested V2 voxel spacing; may increase to satisfy grid cap')
    p.add_argument('--max-volume-nodes',type=int,default=240000,help='Max fine-grid voxels, coarser grid uses 40 percent')
    p.add_argument('--max-sfm-rays',type=int,default=130000,help='Max verified camera-to-observed-SfM track rays')
    p.add_argument('--diagnostic-only',action='store_true',help='Export DISTANCE upper bound only; NO official clean result')
    p.add_argument('--winding-camera-max-delta',type=float,default=0.20,help='Max signed winding difference from a visible exterior COLMAP camera (open-ab)')
    p.add_argument('--min-unobstructed-cameras',type=int,default=2,help='Minimum camera ray votes for open-ab exterior')
    p.add_argument('--colmap-images',default=None)
    p.add_argument('--config',default=None)
    p.add_argument('--fit-max-points',type=int,default=0,
                   help='Default 0 uses ALL core SfM in check/seal/sdf/export; check permits preview sampling only')
    p.add_argument('--chunk-size',type=int,default=None)
    p.add_argument('--keep-diagnostics',action='store_true')
    p.add_argument('--accept-provisional',action='store_true',
                   help='Explicit researcher permission to publish provisional cleaned PLY')
    a = p.parse_args(argv)
    result = run(manifest=a.manifest, phase=a.phase, clip_report=a.clip_report,
                 colmap_images=a.colmap_images, config_file=a.config,
                 fit_max_points=a.fit_max_points, chunk_size=a.chunk_size,
                 keep_diagnostics=a.keep_diagnostics,
                 accept_provisional=a.accept_provisional,
                 preseal_report=a.preseal_report,output_dir=a.output_dir,
                 winding_camera_max_delta=a.winding_camera_max_delta,
                 min_unobstructed_cameras=a.min_unobstructed_cameras,
                 b_method=a.b_method,voxel_size=a.voxel_size,
                 max_volume_nodes=a.max_volume_nodes,
                 max_sfm_rays=a.max_sfm_rays,diagnostic_only=a.diagnostic_only,
                 barrier_audit_dir=a.barrier_audit_dir,index_cache=a.index_cache,
                 classification_only=a.classification_only,confirm_exterior=a.confirm_exterior,
                 nearest_k=a.nearest_k,min_path_agree=a.min_path_agree,
                 exterior_seed=a.exterior_seed,roi_link_samples=a.roi_link_samples,
                 ray_batch=a.ray_batch)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if result['status'].startswith('BLOCKED'):
        raise SystemExit(3)

if __name__=='__main__':
    main()
