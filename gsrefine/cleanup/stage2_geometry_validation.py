"""Stage-2 *sealless* Poisson distance-surface provenance and safety checks.

This is the ONLY geometry preflight for the production ROI/global-Poisson
barrier-parity path. The historical ``preseal_diagnostics.json`` name is kept
for compatibility with stage2 geometry caches, not as a requirement to seal.

The legacy skirt/cap/solid and Open-A+B experiments retain their separate
validators. They are never imported or executed by this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

STAGE2_GEOMETRY_VALIDATION = 'GSSceneRefine.sealless_stage2_geometry.v1'
# These are the SAME mathematical parameters previously used by Stage-2.
# They no longer depend on legacy Open A+B sealing modules.
GAUSSIAN_DISTANCE_DEFAULTS = {
    'trim_fraction': 0.10,
    'sigma_multiplier': 3.0,
    'epsilon': 1e-5,
}


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as reader:
        for block in iter(lambda: reader.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _finite_nonnegative(value, name):
    try:
        n = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'Stage-2 SfM geometry: invalid {name}={value!r}') from exc
    if not np.isfinite(n) or n < 0:
        raise ValueError(f'Stage-2 SfM geometry: nonfinite/negative {name}={n!r}')
    return n


def validate_parity_geometry(manifest, geometry_report, *, config_file=None):
    """Validate a genuine R3-clipped observed *distance surface*, not a sealed mesh.

    Returns a dictionary with record/main/inputs/run_dir/config_file/report and a
    diagnostic summary. Never requires exactly one open boundary ring,
    watertightness, an artificial cap, a positive volume, or face winding.

    Mandatory invariants: authenticated Stage-1/R3 source; authenticated original
    Poisson direct clipping; matching unmodified local main/aux PLY fingerprints;
    finite, sufficiently large, *full-core* SfM-to-main P95 satisfying existing
    scale-relative fit safeguard; non-empty finite positive-area triangle mesh.

    The original full-scene Poisson *wall* is separately validated in the
    intersection/flood/region-separation/XOR parity stage, so no 'solid' test
    on the R3-clipped DISTANCE mesh is appropriate here.
    """
    from gsrefine.cleanup.sdf_common import _load_stage1_manifest
    from gsrefine.cleanup.direct_r3_pipeline import select_direct_r3_clip, validate_sources
    import trimesh

    stage1, upstream, inputs, run_dir = _load_stage1_manifest(manifest)
    stage1 = Path(stage1).resolve(strict=True)
    run_dir = Path(run_dir).resolve(strict=True)
    report_file = Path(geometry_report).expanduser().resolve(strict=True)
    trusted_root = run_dir / 'cleanup' / 'geometry' / 'direct_r3_sdf'
    if not report_file.is_relative_to(trusted_root) or report_file.name != 'preseal_diagnostics.json':
        raise ValueError('Sealless Stage-2: geometry report must be an authenticated '
                         'preseal_diagnostics.json inside this ROI run/cleanup/geometry/direct_r3_sdf')
    record = json.loads(report_file.read_text(encoding='utf-8'))
    if record.get('schema') != 'GSSceneRefine.direct_r3_stage2.v1':
        raise ValueError('Sealless Stage-2: unsupported main/aux geometry report schema')
    if Path(record.get('stage1_manifest') or '/invalid/source').resolve() != stage1:
        raise ValueError('Sealless Stage-2: geometry report refers to a DIFFERENT Stage-1 ROI manifest')

    fit = record.get('sfm_fit') or {}
    p95 = _finite_nonnegative(fit.get('p95'), 'full-core SfM P95')
    sampled = int(fit.get('fit_sfm_count') or 0)
    total = int(fit.get('source_core_sfm_count') or -1)
    if not fit.get('is_full_core_fit') or sampled != total or sampled < 100:
        raise ValueError('Sealless Stage-2: incomplete core-SfM fit; '
                         f'fit_count={sampled}, total={total}, full={fit.get("is_full_core_fit")}. '
                         'Rebuild --phase check with --fit-max-points 0.')
    bounds = record.get('sfm_fit_safeguards') or {}
    upper = _finite_nonnegative(bounds.get('max_by_roi_diagonal'), 'P95 ROI-diagonal limit')
    if not record.get('sfm_fit_safeguards_passed') or p95 > upper:
        raise ValueError('Sealless Stage-2: SfM-to-main surface fit FAILED: '
                         f'P95={p95:.9g}, max_by_roi_diagonal={upper:.9g}; '
                         'review main Poisson geometry and R3/SfM consistency. '
                         'No threshold or safeguard was relaxed.')

    mesh_file = Path(record.get('main_observed_surface') or '/invalid/main.ply').resolve(strict=True)
    if mesh_file.parent != report_file.parent or mesh_file.name != 'poisson_roi_main.ply':
        raise ValueError('Sealless Stage-2: main distance Poisson must be the original '
                         'R3-clipped, main-connected body in this report directory')
    if _digest(mesh_file) != record.get('main_observed_surface_sha256'):
        raise ValueError('Sealless Stage-2: main Poisson PLY changed (SHA256 mismatch)')

    # Ensure the local R3 mesh was extracted directly from the SAME original
    # global Poisson, R3 field and cleaned SfM as the manifest, with no rebuild.
    raw, _, clip = select_direct_r3_clip(
        run_dir, selected_report=record['direct_r3_clip_report'])
    if Path(raw).resolve(strict=True) != Path(record['observed_direct_r3_mesh']).resolve(strict=True):
        raise ValueError('Sealless Stage-2: direct-R3 source mesh differs from geometry report')
    sfm_path = Path(record['sfm']).resolve(strict=True)
    validate_sources(stage1, upstream, inputs, run_dir, clip, sfm_path)

    main = trimesh.load(mesh_file, force='mesh', process=False)
    if not isinstance(main, trimesh.Trimesh):
        raise ValueError('Sealless Stage-2: expected an actual triangle mesh')
    vertices = np.asarray(main.vertices)
    faces = np.asarray(main.faces)
    if (len(faces) == 0 or len(vertices) < 3 or faces.ndim != 2 or faces.shape[1] != 3
            or not np.isfinite(vertices).all() or not np.isfinite(float(main.area))
            or float(main.area) <= 0):
        raise ValueError('Sealless Stage-2: main distance mesh has invalid/empty triangles')
    if (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Sealless Stage-2: triangle vertex index outside source mesh')
    topo = record.get('topology') or {}
    if int(topo.get('mesh_faces', -1)) != len(faces) or int(topo.get('mesh_vertices', -1)) != len(vertices):
        raise ValueError('Sealless Stage-2: main distance mesh topology/count changed since recorded check')

    part = record.get('topology_partition') or {}
    if int(part.get('main_face_count', -1)) != len(faces):
        raise ValueError('Sealless Stage-2: main/aux partition face count is inconsistent')
    if int(part.get('auxiliary_component_count', -1)) < 0:
        raise ValueError('Sealless Stage-2: missing canonical main/auxiliary partition provenance')
    auxiliary = record.get('auxiliary_observed_surface')
    if auxiliary:
        aux_file = Path(auxiliary).resolve(strict=True)
        if aux_file.parent != report_file.parent or aux_file.name != 'poisson_roi_auxiliary.ply':
            raise ValueError('Sealless Stage-2: auxiliary Poisson belongs to another geometry run')
        if _digest(aux_file) != record.get('auxiliary_observed_surface_sha256'):
            raise ValueError('Sealless Stage-2: auxiliary Poisson PLY changed (SHA256 mismatch)')
    elif int(part['auxiliary_component_count']) != 0:
        raise ValueError('Sealless Stage-2: auxiliary components reported but auxiliary mesh is absent')

    # The next properties only mattered to the RETIRED single-ring cap/skirt
    # experiment. They are intentionally REPORT-ONLY here (not hard gates).
    legacy_seal_only = {
        'preseal_eligible': bool(record.get('preseal_eligible', False)),
        'open_boundary_components': int(topo.get('open_boundary_components', 0)),
        'preseal_one_ring_eligible': bool(topo.get('preseal_one_ring_eligible', False)),
        'winding_consistent': bool(topo.get('winding_consistent', False)),
        'original_r3_boundary_preserved': bool(part.get('original_r3_boundary_preserved', False)),
        'main_vertex_fan_issue_count': int(part.get('main_vertex_fan_issue_count', 0)),
    }
    cfgpath = (Path(config_file).resolve(strict=True) if config_file
               else (run_dir / 'metadata' / 'effective_scene.yaml').resolve(strict=True))
    return {
        'record': record, 'main': main, 'inputs': inputs, 'run_dir': run_dir,
        'config_file': cfgpath, 'report': report_file,
        'summary': {
            'schema': STAGE2_GEOMETRY_VALIDATION,
            'status': 'VALIDATED_SEALLESS_DISTANCE_GEOMETRY',
            'p95': p95, 'p95_limit': upper,
            'sfm_fit_points': sampled, 'main_faces': len(faces),
            'auxiliary_components': int(part['auxiliary_component_count']),
            'legacy_seal_only_diagnostics_not_gates': legacy_seal_only,
        },
    }
