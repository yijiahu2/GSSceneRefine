"""Regression: the one-pass cache selector rejects obsolete status fields in V3 audits."""
import json

import pytest

from gsrefine.cleanup import stage2_onepass as mod

VALID = 'REGION_GRAPH_SAMPLED_NOT_CERTIFIED'
OLD = 'TOP_TO_BOTTOM_PATH_FOUND_SAMPLED_GRAPH'
SCHEMA = 'GSSceneRefine.final_R3_global_Poisson_barrier_audit.v2'
VERSION = 'final_R3_original_global_poisson_region_graph_v3_window_20261112'


@pytest.mark.parametrize('coarse,fine,reusable', [
    (VALID, VALID, True),
    (OLD, VALID, False),
    (VALID, OLD, False),
    ('WINDOW_TRUNCATED', VALID, False),
    (VALID, 'WINDOW_TRUNCATED', False),
    (VALID, '', False),
])
def test_only_completed_v3_region_grids_are_reused(tmp_path, coarse, fine, reusable):
    run = tmp_path / 'roi_001'
    run.mkdir()
    manifest = run / 'manifest.json'
    manifest.write_bytes(b'manifest')
    handoff = run / 'handoff.npz'
    handoff.write_bytes(b'handoff')
    poisson = run / 'poisson.ply'
    poisson.write_bytes(b'poisson')
    folder = run / 'cleanup/diagnostics/barrier'
    folder.mkdir(parents=True)
    for filename in ('coarse_connectivity.npz', 'fine_connectivity.npz',
                     'global_poisson_final_r3_intersection.npz'):
        (folder / filename).write_bytes(b'placeholder')
    report = {
        'schema': SCHEMA, 'version': VERSION,
        'status': 'REGION_GRAPHS_COMPLETED_NUMERIC_NOT_CERTIFIED',
        'diagnostic_window': {'window_cover': {'status': 'COMPLETE_SAMPLED_WINDOW'}},
        'inputs': {
            'manifest_sha256': mod._sha(manifest),
            'r3_sha256': mod._sha(handoff),
            'original_poisson_sha256': mod._sha(poisson),
        },
        'flood': {
            'coarse': {'status': coarse, 'grid': {'voxel': .3, 'r3_boundary_touch_count': 0}},
            'fine': {'status': fine, 'grid': {'voxel': .2, 'r3_boundary_touch_count': 0}},
        },
    }
    (folder / 'roi_global_poisson_barrier_report.json').write_text(json.dumps(report))
    result = mod._find_barrier_audit(run, manifest, handoff, poisson)
    assert (result == folder) if reusable else (result is None)
