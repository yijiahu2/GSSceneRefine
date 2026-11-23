"""Read-only LFLS/v4 geometry experiment. Run with python tools/test_r3_original_walls.py."""
import argparse
import json
import sys
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsrefine.cleanup import barrier_separation_gate as gate
from gsrefine.cleanup.local_partition import Domain, anchor_index, rim_witnesses
from gsrefine.cleanup.roi_barrier_parity import segment_parity
from gsrefine.cleanup.sdf_common import _load_stage1_manifest
from gsrefine.cleanup.stage2_onepass import _atomic_json
from gsrefine.cleanup.stage2_runtime import file_digest, run_scope
from gsrefine.geometry.r3_direct_poisson_clip import _restore_r3_field, _source_r3_handoff
from tools import audit_final_r3_global_poisson_barrier as grid


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def source_groups(mesh):
    groups = trimesh.graph.connected_components(mesh.face_adjacency,
        nodes=np.arange(len(mesh.faces)), min_len=1)
    # Match the v4 global-main rule (largest edge-connected face set).
    groups = sorted(groups, key=len, reverse=True)
    component = np.empty(len(mesh.faces), np.int32)
    for index, rows in enumerate(groups):
        component[rows] = index
    main = np.sort(groups[0])
    # Retain original vertex coordinates and face order, without welding.
    vertices, inverse = np.unique(mesh.faces[main], return_inverse=True)
    wall = trimesh.Trimesh(vertices=mesh.vertices[vertices], faces=inverse.reshape(-1, 3),
                           process=False)
    return wall, component, [len(g) for g in groups]


def shared_grid(domain, spec):
    origin, h, shape, previous = spec
    xyz = grid.grid_centers(origin, h, shape)
    inside = grid.eval_r3(domain.membership, xyz) > 0
    if previous is not None and not np.array_equal(inside.reshape(shape), previous):
        raise ValueError('Replayed R3 membership differs from the v4 reference grid.')
    volume = inside.reshape(shape)
    if any(np.take(volume, [0, -1], axis=i).any() for i in range(3)):
        raise ValueError('R3 touches the sampling window. Comparison is inconclusive.')
    a, b = grid.neighbour_pairs(shape, inside)
    allowed = grid.link_r3_filter(xyz[a], xyz[b], domain, samples=11)
    return xyz, inside, a[allowed], b[allowed]


def distance(scene, o3d, points):
    return scene.compute_distance(o3d.core.Tensor(np.ascontiguousarray(points, np.float32))).numpy()


def valid_walk(scene, o3d, domain, path):
    a, b = path[:-1], path[1:]
    clear = grid._ray_hit_free(scene, o3d, a, b, eps=0., batch=8000)
    clear &= grid._ray_hit_free(scene, o3d, b, a, eps=0., batch=8000)
    within = grid.link_r3_filter(a, b, domain, samples=63)
    return bool(clear.all() and within.all())


def check_grid(folder, name, spec, shared, domain, wall, scene, o3d, seed):
    origin, h, shape, _ = spec
    xyz, inside, a, b = shared
    usable = inside.copy()
    usable[inside] = distance(scene, o3d, xyz[inside]) > max(1e-6, .015*h)
    active = usable[a] & usable[b]
    if not usable.any():
        raise ValueError('No passable R3 grid nodes.')
    free = np.zeros(len(a), bool)
    u, v = xyz[a[active]], xyz[b[active]]
    free[active] = (grid._ray_hit_free(scene, o3d, u, v, eps=0., batch=30000)
                    & grid._ray_hit_free(scene, o3d, v, u, eps=0., batch=30000))
    labels, count = grid._connected_components(len(xyz), usable, a[free], b[free])
    suspect = np.flatnonzero(active & ~free & (labels[a] == labels[b]))
    confirmed, rejected = 0, 0
    if len(suspect):
        bits, ok = segment_parity(scene, o3d, xyz[a[suspect]], xyz[b[suspect]],
                                   normals=wall.face_normals)
        odd = suspect[ok & (bits == 1)]
        # One revalidated walk suffices for a counterexample; save its source chord.
        for k in np.unique(np.linspace(0, len(odd)-1, min(5, len(odd)), dtype=int)):
            i = odd[k]
            path = grid._path_witness(len(xyz), a[free], b[free], [a[i]], [b[i]], xyz)
            if len(path) and valid_walk(scene, o3d, domain, path):
                np.savez_compressed(folder/(name+'_walk.npz'), path=path,
                                    odd_chord=xyz[[a[i], b[i]]])
                confirmed = 1
                break
            rejected += 1
    npz = folder/(name+'_connectivity.npz')
    np.savez_compressed(npz, origin=origin, voxel=h, shape=shape, components=labels.reshape(shape),
                        final_r3_inside=inside.reshape(shape), passable=usable.reshape(shape))
    g = gate._load(npz, name)
    g['gate_summary'] = gate._check_level(g, domain, scene, o3d, wall.face_normals)
    node = anchor_index(g, seed, None, domain, scene, o3d)
    region = None if node is None else int(labels[node])
    info = dict(raw_connected_components=count, region_of_fixed_seed=region,
                seed_link_node_xyz=None if node is None else xyz[node],
                confirmed_graph_walk=confirmed, failed_walk_rechecks=rejected,
                walkable_links=int(free.sum()), near_wall_nodes=int((inside & ~usable).sum()),
                geometry=g['gate_summary'])
    return g, info, active, free


def replay_rims(paths, domain, mesh, scene, o3d):
    if not len(paths):
        return np.empty(0, bool)
    a, tip, b = paths.transpose(1, 0, 2)
    valid = np.ones(len(a), bool)
    for x, y in ((a, tip), (tip, b), (b, a)):
        valid &= domain.membership(x) > 0
        valid &= grid.link_r3_filter(x, y, domain, samples=63)
    for x, y in ((a, tip), (tip, b)):
        valid &= grid._ray_hit_free(scene, o3d, x, y, eps=0., batch=8000)
        valid &= grid._ray_hit_free(scene, o3d, y, x, eps=0., batch=8000)
    guard = 16*np.finfo(np.float32).eps*max(1., abs(mesh.vertices).max())
    odd, reliable = segment_parity(scene, o3d, a, b, normals=mesh.face_normals,
                                    endpoint_tol=guard*.2)
    return valid & reliable & (odd == 1)


def added_blockers(folder, name, xyz, a, b, changed, scene, o3d, component):
    rows = np.flatnonzero(changed)
    face_ids, link_ids = [], []
    for batch in np.array_split(rows, max(1, (len(rows)+8191)//8192)):
        if not len(batch):
            continue
        origins = xyz[a[batch]]
        direction = xyz[b[batch]] - origins
        rays = np.ascontiguousarray(np.c_[origins, direction], np.float32)
        hits = scene.list_intersections(o3d.core.Tensor(rays))
        t = hits['t_hit'].numpy()
        ids = hits['primitive_ids'].numpy()
        keep = (t > 0) & (t < 1) & (component[ids] != 0)
        face_ids.extend(ids[keep])
        link_ids.extend(batch[hits['ray_ids'].numpy()[keep]])
    faces = np.asarray(face_ids, dtype=int)
    links = np.asarray(link_ids, dtype=int)
    np.savez_compressed(folder/(name+'_added_blockers.npz'),
                        original_face=faces, component=component[faces],
                        source_node=a[links], target_node=b[links])
    group, hits = np.unique(component[faces], return_counts=True)
    return dict(new_crossing_blocked_links=len(rows),
                links_with_added_face_evidence=len(np.unique(links)),
                additional_original_faces=len(np.unique(faces)),
                hits_by_added_component=dict(zip(map(str, group), map(int, hits))))


def compare(mesh, domain, specs, seed, old_paths, folder):
    main, component, sizes = source_groups(mesh)
    walls = {'A_largest': main, 'B_all': mesh}
    scenes = {key: grid.make_scene(w) for key, w in walls.items()}
    details = {key: {} for key in walls}
    grids = {key: {} for key in walls}
    paired = {}
    np.savez_compressed(folder/'source_components.npz', component_by_original_face=component)
    for key in walls:
        (folder/key).mkdir()
    for level, spec in specs.items():
        print('[A/B] Shared R3 nodes and links:', level, flush=True)
        shared = shared_grid(domain, spec)
        xyz, inside, a, b = shared
        masks = {}
        for key, wall in walls.items():
            print('[A/B]', level, key, flush=True)
            g, info, active, free = check_grid(folder/key, level, spec, shared, domain,
                                      wall, *scenes[key], seed)
            grids[key][level], details[key][level] = g, info
            masks[key] = active, free
        active_a, free_a = masks['A_largest']
        active_b, free_b = masks['B_all']
        paired[level] = dict(r3_nodes=int(inside.sum()), common_pre_wall_links=len(a),
            lost_to_added_near_surface_nodes=int((free_a & ~active_b).sum()),
            unexpected_new_free_links_in_B=int((free_b & ~free_a).sum()),
            **added_blockers(folder, level, xyz, a, b, free_a & active_b & ~free_b,
                             *scenes['B_all'], component))
    for key, wall in walls.items():
        r = details[key]
        cs, fs = grids[key]['coarse'], grids[key]['fine']
        reasons = []
        for lvl, g in grids[key].items():
            info = r[lvl]
            reasons.extend(lvl+': '+e for e in g['gate_summary']['errors'])
            if len(g['material']) != 2:
                reasons.append(lvl+': expected exactly two material regions')
            if info['region_of_fixed_seed'] not in g['material']:
                reasons.append(lvl+': fixed exterior reference not linked to a material region')
        if not reasons:
            cross = gate._cross_scale(cs, fs)
            reasons.extend(cross['errors'])
            # The same physical seed must lie on corresponding sides across resolutions.
            mapped = cross['correspondence'].get(str(r['coarse']['region_of_fixed_seed']), {})
            if mapped.get('fine_region') != r['fine']['region_of_fixed_seed']:
                reasons.append('Exterior anchor changes side across scales')
            r['cross_scale'] = cross
        scene, o3d = scenes[key]
        paths, n = rim_witnesses(wall, domain, scene, o3d, specs['fine'][1])
        direct = replay_rims(paths, domain, wall, scene, o3d)
        old = replay_rims(old_paths, domain, wall, scene, o3d)
        np.savez_compressed(folder/key/'rim_evidence.npz', original_rims=paths[direct],
                            v4_path_indices=np.flatnonzero(old))
        graph_bypass = any(r[k]['confirmed_graph_walk'] for k in specs)
        if any(r[k]['failed_walk_rechecks'] for k in specs) and not graph_bypass:
            reasons.append('Unresolved sampled graph path; cannot accept this discretization')
        bypass = bool(direct.any() or old.any() or graph_bypass)
        r.update(original_free_edges=n, original_rim_witnesses=int(direct.sum()),
                 v4_rim_witnesses_reproduced=int(old.sum()), graph_errors=reasons,
                 status=('BYPASS_OBSERVED' if bypass else
                         'INCONCLUSIVE' if reasons else 'TWO_MAIN_REGIONS_SUPPORTED'))
    A, B = (details[key]['status'] for key in walls)
    if any(x['unexpected_new_free_links_in_B'] for x in paired.values()):
        outcome = 'INCONCLUSIVE_NONMONOTONE_GRAPH'
    elif A == 'TWO_MAIN_REGIONS_SUPPORTED':
        outcome = 'LARGEST_SUFFICIENT_AT_TESTED_SCALES'
    elif B == 'TWO_MAIN_REGIONS_SUPPORTED':
        outcome = 'ONLY_ALL_SUPPORTED_CHECK_ADDED_FACES'
    elif A == B == 'BYPASS_OBSERVED':
        outcome = 'BOTH_HAVE_BYPASS_CHECK_LOCATIONS'
    else:
        outcome = 'INCONCLUSIVE'
    return dict(outcome=outcome, groups=details, paired=paired, fixed_seed_xyz=seed,
                original_component_sizes=sizes, equal_largest_count=sizes.count(sizes[0]),
                v4_reclip_paths=len(old_paths),
                limits='Finite-resolution geometry only. A local rim on a secondary sheet need not join the two main regions. No Gaussian classifications.')


@run_scope
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roi-run', required=True, type=Path)
    p.add_argument('--reference-name', default='stage2_local_region_v4')
    args = p.parse_args()
    roi = args.roi_run.resolve(strict=True)
    ref = roi/'cleanup'/args.reference_name
    diagnostics = ref/'diagnostics'/'source_reclip_audit'
    record = read(ref/'stage2_final_report.json')
    prev = read(diagnostics/'local_partition_report.json')
    manifest, upstream, inputs, root = _load_stage1_manifest(roi/'metadata/manifest.json')
    if file_digest(manifest) != record['input']['stage1_manifest_sha256'] or (
            file_digest(inputs['poisson_mesh']) != record['input']['original_global_poisson_sha256']):
        raise ValueError('Stage 1 and v4 reference do not share the same source geometry.')
    specs = {}
    for key in ('coarse', 'fine'):
        path = diagnostics/(key+'_connectivity.npz')
        if file_digest(path) != prev['verdict']['levels'][key]['grid_sha256']:
            raise ValueError('Reference grid hash mismatch: '+key)
        with np.load(path, allow_pickle=False) as z:
            specs[key] = z['origin'], float(z['voxel']), tuple(map(int,z['shape'])), z['final_r3_inside']
    with np.load(diagnostics/'bypass_paths.npz', allow_pickle=False) as z:
        paths = z['xyz']
    if paths.shape != (prev['rim_bypasses'], 3, 3) or not np.isfinite(paths).all():
        raise ValueError('Invalid v4 rim witnesses')
    seed = np.asarray(record['geometry']['anchor'], float)
    if seed.shape != (3,) or not np.isfinite(seed).all():
        raise ValueError('Invalid reference seed')
    if specs['coarse'][1] < 1.05*specs['fine'][1]:
        raise ValueError('The reference coarse and fine grids are not sufficiently distinct.')
    start = perf_counter()
    print('[A/B] Restoring original final-R3 field once', flush=True)
    handoff = _source_r3_handoff(root, upstream)
    mesh, field, _ = _restore_r3_field(inputs=inputs, upstream=upstream, run_dir=root,
            audit_record=handoff, with_interior_radius=True)
    domain = Domain(field)
    if domain.membership(seed[None])[0] <= 0:
        raise ValueError('Reference seed outside R3')
    folder = Path(mkdtemp(prefix='original_wall_ab_', dir=roi/'cleanup'))
    print('[A/B] Output:', folder, flush=True)
    result = compare(mesh, domain, specs, seed, paths, folder)
    result.update(stage1_manifest_sha256=file_digest(manifest),
                  poisson_sha256=file_digest(inputs['poisson_mesh']),
                  reference=ref.name, elapsed_seconds=perf_counter()-start)
    _atomic_json(folder/'original_wall_ab.json', result)
    print('[A/B]',result['outcome'], flush=True)
    for key, g in result['groups'].items():
        print(key, g['status'], 'reproduced_v4_paths', g['v4_rim_witnesses_reproduced'])
    print(folder/'original_wall_ab.json')


if __name__ == '__main__':
    main()
