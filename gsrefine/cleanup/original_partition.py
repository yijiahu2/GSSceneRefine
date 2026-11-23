"""Use the accepted group A of an original-wall comparison as spatial labels."""
import json
from pathlib import Path

import numpy as np

from gsrefine.cleanup import barrier_separation_gate as gate
from gsrefine.cleanup.local_partition import Domain
from gsrefine.cleanup.roi_barrier_parity import OUTSIDE, UNKNOWN
from gsrefine.cleanup.stage2_runtime import file_digest, measured
from tools import audit_final_r3_global_poisson_barrier as grid
from tools.test_r3_original_walls import source_groups


def read(path):
    return Path(path).read_text(encoding='utf-8-sig')


def select_report(run_dir, stage1, inputs, explicit):
    root = Path(run_dir)/'cleanup'
    expected = file_digest(stage1), file_digest(inputs['poisson_mesh'])
    paths = [Path(explicit).expanduser().resolve(strict=True)] if explicit else (
        sorted(root.glob('original_wall_ab_*/original_wall_ab.json')))
    matches = []
    for path in paths:
        value = json.loads(read(path))
        if (value.get('stage1_manifest_sha256'), value.get('poisson_sha256')) == expected:
            matches.append(path)
    if not matches and not explicit:
        return None
    if not matches:
        raise ValueError('No A/B report matches the Stage-1 manifest and original Poisson.')
    # Multiple repeated experiments can be used only if they identify the same decision.
    identities = set()
    for path in matches:
        record = json.loads(read(path))
        a = record['groups']['A_largest']
        identities.add((record['outcome'], a['status'], tuple(record['fixed_seed_xyz']),
                        a['fine']['region_of_fixed_seed'], a['coarse']['region_of_fixed_seed'],
                        *(a[level]['geometry']['grid_sha256'] for level in ('coarse','fine'))))
    if len(identities) != 1:
        raise ValueError('Different matching A/B decisions exist; select one with --partition-report.')
    return matches[0]


@measured('load_verified_original_A_partition')
def load(data, report_path, requested_seed):
    path = Path(report_path)
    record = json.loads(read(path))
    a = record['groups']['A_largest']
    if (record['outcome'] != 'LARGEST_SUFFICIENT_AT_TESTED_SCALES'
            or a['status'] != 'TWO_MAIN_REGIONS_SUPPORTED' or a['graph_errors']
            or a['original_rim_witnesses'] or a['v4_rim_witnesses_reproduced']
            or a['cross_scale']['errors']):
        raise ValueError('No accepted original A partition in the supplied report.')
    seed = np.asarray(record['fixed_seed_xyz'], float)
    if seed.shape != (3,) or not np.isfinite(seed).all() or (
            requested_seed is not None and not np.array_equal(seed, requested_seed)):
        raise ValueError('Exterior seed does not match the fixed A/B reference.')
    field = Domain(data['field'])
    if field.membership(seed[None])[0] <= 0:
        raise ValueError('Exterior reference lies outside current R3.')
    if record['equal_largest_count'] != 1:
        raise ValueError('The reference has no unique largest source component.')
    ref = Path(data['run_dir'])/'cleanup'/record['reference']/'stage2_final_report.json'
    v4 = json.loads(read(ref))
    identity = v4['input']
    if (identity['stage1_manifest_sha256'] != record['stage1_manifest_sha256']
            or identity['original_global_poisson_sha256'] != record['poisson_sha256']
            or identity['preseal_report_sha256'] != file_digest(data['preseal'])
            or not np.array_equal(seed, v4['geometry']['anchor'])):
        raise ValueError('Reference and current run differ in geometry or accepted anchor.')
    # Reproduce the experiment's face selection, including its native triangle order.
    wall, _, sizes = source_groups(data['orig_global'])
    if sizes != record['original_component_sizes']:
        raise ValueError('Original mesh components differ from A/B reference.')
    fine_index = None
    seed_regions = {}
    for name in ('coarse', 'fine'):
        level = a[name]
        geom = level['geometry']
        graph = geom['graph']
        seed_id = level['region_of_fixed_seed']
        grid_path = path.parent/'A_largest'/(name+'_connectivity.npz')
        g = gate._load(grid_path, name)
        ids = g['material']
        if (g['sha256'] != geom['grid_sha256'] or ids != geom['material_ids']
                or len(ids) != 2 or seed_id not in ids or geom['errors']
                or graph['parity_conflicts'] or not graph['opposite_side_evidenced']
                or level['confirmed_graph_walk'] or level['failed_walk_rechecks']):
            raise ValueError('Inconsistent group A connectivity: '+name)
        with np.load(grid_path, allow_pickle=False) as z:
            if not np.array_equal(grid.eval_r3(field.membership, g['xyz']) > 0,
                                  z['final_r3_inside'].ravel()):
                raise ValueError('Current R3 differs from the A grid: '+name)
        bit = graph['material_region_labels']
        if {bit[str(i)] for i in ids} != {0, 1}:
            raise ValueError('Grid regions lack opposite crossing parity.')
        seed_regions[name] = seed_id
        if name == 'fine':
            labels = np.full(g['shape'], UNKNOWN, np.int8)
            for i in ids:
                labels[g['regions'] == i] = OUTSIDE ^ bit[str(i)] ^ bit[str(seed_id)]
            fine_index = {k:g[k] for k in ('origin','voxel','shape')}
            fine_index['labels'] = labels
    if a['cross_scale']['correspondence'][str(seed_regions['coarse'])]['fine_region'] != seed_regions['fine']:
        raise ValueError('Fixed exterior region differs across scales.')
    geom = data['record']
    scope = dict(scope='full', main=str(data['inputs']['poisson_mesh']),
                 component='original_global_largest', faces=len(wall.faces),
                 main_sha256=record['poisson_sha256'], anchor=seed,
                 seed_regions=seed_regions, partition_report=str(path),
                 partition_report_sha256=file_digest(path), group='A_largest',
                 fine_grid_sha256=a['fine']['geometry']['grid_sha256'],
                 distance_surface=geom['main_observed_surface'],
                 distance_mesh_sha256=file_digest(geom['main_observed_surface']),
                 sfm_p95=geom['sfm_fit']['p95'], historical_ab_seconds=record.get('elapsed_seconds'))
    return fine_index, wall, field, scope
