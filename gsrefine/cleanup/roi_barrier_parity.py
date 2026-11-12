"""Stage-2 ROI barrier parity SDF: original global Poisson as wall, local Poisson for distance.

Workflow: audited FINAL R3 geometry -> sampled navigation graph with per-edge ORIGINAL
triangle intersection parities -> exterior-anchored XOR labels -> short exact ray queries
for ONLY over-distance Gaussian centres. No artificial Poisson closure. All inputs are
immutable and results are confined to user-supplied --output-dir.

A sampled 3D graph cannot certify topological watertightness. Adaptive queries
attempt increasingly comprehensive exact ray tests; a formal cleaned output is
blocked if any over-threshold candidate cannot be assigned safely to one side.
A diagnostic classification-only preview always preserves unresolved candidates.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
import hashlib
import os
import tempfile

import numpy as np

VERSION = 'GSSceneRefine.roi_barrier_parity_sdf.v3'
OUTSIDE, INSIDE, UNKNOWN = 0, 1, -1


def _log(message):
    print('[stage2-roi-parity] ' + str(message), flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _json_native(value):
    if isinstance(value, dict):
        return {str(k): _json_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_native(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(_json_native(value), f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _validate_intersection_loop(archive):
    """Check that extracted discrete intersection segments are closed 2-regular cycles."""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    with np.load(archive, allow_pickle=False) as z:
        pts = np.asarray(z['edge_intersection_xyz'], np.float64)
        seg = np.asarray(z['segment_endpoints'], np.float64)
    if len(pts) < 3 or len(seg) < 3:
        raise ValueError('ROI/global Poisson intersection has insufficient segments')
    d, ids = cKDTree(pts).query(seg.reshape(-1, 3), k=1)
    if not np.isfinite(d).all() or float(d.max()) > 1e-5:
        raise ValueError('Intersection segment endpoints do not match extracted roots')
    edges = ids.reshape(-1, 2)
    degree = np.bincount(edges.ravel(), minlength=len(pts))
    if np.any(edges[:, 0] == edges[:, 1]) or not np.all(degree == 2):
        raise ValueError('Intersection line has branches, dangling endpoints or degenerate segments')
    row = np.r_[edges[:, 0], edges[:, 1]]
    col = np.r_[edges[:, 1], edges[:, 0]]
    graph = coo_matrix((np.ones(len(row), np.uint8), (row, col)), shape=(len(pts), len(pts))).tocsr()
    count, labels = connected_components(graph, directed=False)
    if count < 1:
        raise ValueError('No closed intersection boundary components')
    return {'intersections': int(len(pts)), 'segments': int(len(seg)),
            'closed_cycles': int(count), 'cycle_lengths':
            sorted(np.bincount(labels).astype(int).tolist(), reverse=True),
            'segment_endpoint_max_error': float(d.max())}


def _audit_identity(audit_dir, manifest_path, inputs, r3_record, *,
                    field, wall_scene, o3d, normals, output):
    root = Path(audit_dir).expanduser().resolve(strict=True)
    report = root / 'roi_global_poisson_barrier_report.json'
    data = json.loads(report.read_text(encoding='utf-8'))
    if data.get('schema') != 'GSSceneRefine.final_R3_global_Poisson_barrier_audit.v1':
        raise ValueError('Not a verified final R3/global Poisson barrier audit')
    if data.get('status') != 'FLOOD_SAMPLING_COMPLETED_NOT_CERTIFIED_SEALED':
        raise ValueError('Incomplete two-resolution spatial component audit')
    inp = data['inputs']
    required = {'manifest_sha256': _sha(manifest_path),
                'r3_sha256': _sha(r3_record),
                'original_poisson_sha256': _sha(inputs['poisson_mesh'])}
    for key, value in required.items():
        if inp.get(key) != value:
            raise ValueError(f'Barrier audit source mismatch: {key}')
    if str(Path(inp['final_roi_gaussian']).resolve()) != str(Path(inputs['gaussian_ply']).resolve()):
        raise ValueError('Audit Gaussian PLY is from a different ROI run')
    if data.get('r3_verified_replay', {}).get('v3_handoff_max_field_distance_error', float('inf')) > 1e-6:
        raise ValueError('R3 replay consistency check failed')
    loops = _validate_intersection_loop(root / 'global_poisson_final_r3_intersection.npz')
    from gsrefine.cleanup.barrier_separation_gate import evaluate
    verdict = evaluate(audit_dir=root, audit_record=data, field=field,
                       wall_scene=wall_scene, o3d=o3d, normals=normals)
    _write_json(Path(output) / 'region_separation_gate.json', verdict)
    if verdict['status'] != 'SUPPORTED_NUMERIC_TWO_SIDED_PARTITION_NOT_SEALED':
        raise ValueError('R3 / original main Poisson region separation failed: '
                         + '; '.join(verdict['blocking_reasons'])
                         + f'. Details: {Path(output) / "region_separation_gate.json"}')
    return data, loops, verdict


class ParityUnionFind:
    """Union-find with XOR potential, detecting odd-parity contradictions."""
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int32)
        self.to_parent = np.zeros(n, dtype=np.uint8)
        self.bad = np.zeros(n, dtype=bool)
        self.contradictions = 0

    def find(self, a):
        x = int(a)
        if self.parent[x] == x:
            return x, 0
        p = int(self.parent[x]); root, parity = self.find(p)
        parity ^= int(self.to_parent[x])
        self.parent[x] = root
        self.to_parent[x] = parity
        return root, parity

    def add(self, a, b, parity):
        a, b, parity = int(a), int(b), int(parity)
        ra, pa = self.find(a)
        rb, pb = self.find(b)
        if ra == rb:
            if (pa ^ pb) != parity:
                self.bad[ra] = True
                self.contradictions += 1
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.to_parent[rb] = pa ^ pb ^ parity
        self.size[ra] += self.size[rb]
        self.bad[ra] |= self.bad[rb]

    def labels(self, valid, exterior_seed_indices):
        valid = np.asarray(valid, bool)
        root = np.full(len(valid), -1, np.int32)
        xr = np.zeros(len(valid), np.uint8)
        for i in np.flatnonzero(valid):
            root[i], xr[i] = self.find(int(i))
        anchors = {}
        for i in np.asarray(exterior_seed_indices, np.int64):
            if not valid[i]:
                continue
            r = int(root[i]); ref = int(xr[i])
            if r in anchors and anchors[r] != ref:
                self.bad[r] = True
                self.contradictions += 1
            anchors[r] = ref
        out = np.full(len(valid), UNKNOWN, np.int8)
        for r, ref in anchors.items():
            if not self.bad[r]:
                idx = valid & (root == r)
                out[idx] = xr[idx].astype(np.int8) ^ ref
        return out, {'root_count': int(len(set(root[valid]))),
                     'exterior_anchored_roots': int(len(anchors)),
                     'odd_cycle_conflicts': int(self.contradictions),
                     'conflicting_roots': int(sum(bool(self.bad[r]) for r in set(root[valid]))),
                     'known_outside_nodes': int(np.sum(out == OUTSIDE)),
                     'known_inside_nodes': int(np.sum(out == INSIDE)),
                     'unknown_nodes': int(np.sum(valid & (out == UNKNOWN)))}


def _crossings_from_hits(result, lengths, dirs, tri_normals=None, *,
                         endpoint_tol=1e-5, uv_tol=2e-5, tangent_tol=1e-5):
    """Convert all Open3D intersections to conservative segment crossing parities.

    Returns parity int8 and valid bool per ray. Shared-edge, grazing, duplicate,
    near-endpoint intersections are UNKNOWN rather than forcibly assigned.
    """
    lengths = np.asarray(lengths, np.float64).ravel()
    dirs = np.asarray(dirs, np.float64).reshape(-1, 3)
    n = len(lengths)
    parities = np.zeros(n, np.uint8)
    valid = (lengths > 10 * endpoint_tol) & np.isfinite(lengths)
    splits = np.asarray(result['ray_splits'], np.int64).ravel()
    t = np.asarray(result['t_hit'], np.float64).ravel()
    uv = np.asarray(result['primitive_uvs'], np.float64).reshape(-1, 2)
    face = np.asarray(result['primitive_ids'], np.int64).ravel()
    if len(splits) != n + 1 or len(t) != len(face) or len(t) != len(uv) or splits[-1] != len(t):
        raise RuntimeError('Malformed Open3D list_intersections return shape')
    for i in range(n):
        if not valid[i]:
            continue
        start, stop = int(splits[i]), int(splits[i + 1])
        if stop == start:
            continue
        distances = t[start:stop]
        uvs = uv[start:stop]
        fid = face[start:stop]
        if not np.isfinite(distances).all() or not np.isfinite(uvs).all():
            valid[i] = False; continue
        if np.any((distances >= -endpoint_tol) & (distances <= endpoint_tol)) or \
           np.any((distances >= lengths[i] - endpoint_tol) &
                  (distances <= lengths[i] + endpoint_tol)):
            valid[i] = False; continue
        within = (distances > endpoint_tol) & (distances < lengths[i] - endpoint_tol)
        if not np.any(within):
            continue
        tt, uu, ff = distances[within], uvs[within], fid[within]
        bary = np.c_[uu, 1. - uu.sum(axis=1)]
        if np.any(bary < uv_tol) or np.any(bary > 1 + uv_tol):
            valid[i] = False; continue
        if tri_normals is not None:
            normals = np.asarray(tri_normals)[ff]
            if np.any(np.abs(normals @ dirs[i]) < tangent_tol):
                valid[i] = False; continue
        tt = np.sort(tt)
        if len(tt) > 1 and np.any(np.diff(tt) <= max(endpoint_tol, lengths[i] * 1e-5)):
            valid[i] = False; continue
        parities[i] = len(tt) % 2
    return parities, valid


def segment_parity(scene, o3d, origins, targets, *, normals=None,
                   ray_batch=8000, endpoint_tol=1e-5):
    """Original triangles, ALL segment intersections, with degeneracy rejection."""
    origins = np.asarray(origins, np.float64).reshape(-1, 3)
    targets = np.asarray(targets, np.float64).reshape(-1, 3)
    if len(origins) != len(targets):
        raise ValueError('Segment origin/target count mismatch')
    bits = np.zeros(len(origins), np.uint8)
    reliable = np.zeros(len(origins), bool)
    for a in range(0, len(origins), int(ray_batch)):
        b = min(a + int(ray_batch), len(origins))
        dr = targets[a:b] - origins[a:b]
        length = np.linalg.norm(dr, axis=1)
        direction = dr / np.maximum(length[:, None], 1e-20)
        rays = np.ascontiguousarray(np.c_[origins[a:b], direction], np.float32)
        raw = scene.list_intersections(o3d.core.Tensor(rays))
        parsed = {k: raw[k].numpy() for k in ('ray_splits', 't_hit', 'primitive_uvs', 'primitive_ids')}
        bits[a:b], reliable[a:b] = _crossings_from_hits(parsed, length, direction,
                                                         normals, endpoint_tol=endpoint_tol)
    return bits, reliable


def _prepare_geometry(manifest, preseal_report, *, config_file=None):
    # Formal Stage 2 is SEALLESS. Legacy single-ring cap/skirt eligibility is
    # not a geometric prerequisite for original-Poisson parity classification.
    from gsrefine.cleanup.stage2_geometry_validation import validate_parity_geometry
    from gsrefine.geometry.r3_direct_poisson_clip import _restore_r3_field, _source_r3_handoff
    from tools.audit_final_r3_global_poisson_barrier import largest_original_component, make_scene
    checked = validate_parity_geometry(manifest, preseal_report, config_file=config_file)
    record, main, inputs, run_dir = (checked[k] for k in ('record', 'main', 'inputs', 'run_dir'))
    cfg, preseal = checked['config_file'], checked['report']
    camera_path = None  # Historical seal-specific camera/winding checks disabled.
    from gsrefine.cleanup.sdf_common import _load_stage1_manifest
    stage1, upstream, original_inputs, original_run_dir = _load_stage1_manifest(manifest)
    handoff = _source_r3_handoff(Path(original_run_dir), upstream)
    orig_global, field, replay = _restore_r3_field(
        inputs=original_inputs, upstream=upstream, run_dir=Path(original_run_dir),
        audit_record=handoff, audit_samples=768)
    global_main, source_stats = largest_original_component(orig_global)
    wall_scene, o3d = make_scene(global_main)
    from gsrefine.cleanup.sdf_common import _raycast
    distance_scene = _raycast(o3d, main)
    faces = np.asarray(global_main.faces, np.int64)
    vertices = np.asarray(global_main.vertices, np.float64)
    normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                       vertices[faces[:, 2]] - vertices[faces[:, 0]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
    roi_raw = json.loads(Path(inputs['roi_json']).read_text('utf-8'))
    meta = roi_raw.get('annotation_metadata', {})
    if meta.get('orientation_verified') is not True:
        raise ValueError('Missing verified final ROI orientation metadata')
    up = np.asarray(meta.get('up_vector'), np.float64)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-8:
        raise ValueError('Invalid scene up vector')
    up = up / np.linalg.norm(up)
    return locals()


def _navigation_identity(audit_report, grid_path, *, roi_link_samples=3):
    """Only data actually controlling the navigation geometry belong in its identity.

    In V1 an unused positional parameter caused a TypeError before navigation
    was built. Distance-only local ROI mesh is intentionally NOT part of this
    identifier: navigation depends on final R3, ORIGINAL global wall, grid,
    path validation, and exterior anchor, not the Gaussian method's SDF mesh.
    """
    return {
        'schema': VERSION,
        'audit_manifest_sha256': audit_report['inputs']['manifest_sha256'],
        'r3_sha256': audit_report['inputs']['r3_sha256'],
        'global_poisson_sha256': audit_report['inputs']['original_poisson_sha256'],
        'grid_sha256': _sha(grid_path),
        'roi_link_samples': int(roi_link_samples),
        'spatial_classification_method': 'original-triangle XOR navigation and adaptive short rays',
    }


def _make_navigation(grid_path, field, up, wall_scene, o3d, normals, *,
                     exterior_seed=None, neighbour_rays_batch=8000, roi_link_samples=3):
    from tools.audit_final_r3_global_poisson_barrier import (
        grid_centers, neighbour_pairs, link_r3_filter, choose_top_bottom_seeds, eval_r3)
    with np.load(grid_path, allow_pickle=False) as g:
        origin = np.asarray(g['origin'], float)
        voxel = float(g['voxel'])
        shape = tuple(int(x) for x in g['shape'])
        passable = np.asarray(g['passable'], bool).ravel()
        stored_inside = np.asarray(g['final_r3_inside'], bool).ravel()
    if int(np.prod(shape)) != len(passable) or not np.isfinite(voxel) or voxel <= 0:
        raise ValueError('Broken sampled ROI navigation grid')
    xyz = grid_centers(origin, voxel, shape)
    # Compare sampled continuous final R3 field, never substitute the initial polygon/AABB.
    rng = np.random.default_rng(42)
    check = rng.choice(len(xyz), min(len(xyz), 4096), replace=False)
    actual_inside = eval_r3(field, xyz[check], name='R3 navigation cache validation') > 0
    if not np.array_equal(actual_inside, stored_inside[check]):
        raise ValueError('Navigation grid does not match reconstructed FINAL R3 field')
    src, dst = neighbour_pairs(shape, passable)
    _log(f'Building parity graph: {int(passable.sum()):,} nodes, {len(src):,} candidate edges')
    verified = link_r3_filter(xyz[src], xyz[dst], field,
                              samples=int(roi_link_samples), batch=2048)
    src, dst = src[verified], dst[verified]
    bits = np.empty(len(src), np.uint8)
    good = np.empty(len(src), bool)
    block = max(1000, int(neighbour_rays_batch))
    for a in range(0, len(src), block):
        b = min(a + block, len(src))
        bits[a:b], good[a:b] = segment_parity(wall_scene, o3d, xyz[src[a:b]],
                                               xyz[dst[a:b]], normals=normals,
                                               ray_batch=block,
                                               endpoint_tol=max(1e-6, voxel * 1e-5))
        if a == 0 or b == len(src) or a // block % 10 == 0:
            _log(f'Original-triangle edge parity {b:,}/{len(src):,}')
    # Top-shell exterior is a per-scene hypothesis: accept only after user review.
    if exterior_seed is None:
        top_shell, _, seed_description = choose_top_bottom_seeds(passable, xyz, shape, up)
        # One exterior reference, not ALL high shell cells: the latter may include
        # both Poisson sides in vertically oriented / folded structures.
        best = top_shell[np.argmax(xyz[top_shell] @ up)]
        seed_idx = np.asarray([best], np.int64)
        seed_mode = 'single highest R3 shell node (researcher must verify exterior)'
        seed_description['chosen_seed_xyz'] = xyz[best].tolist()
    else:
        from scipy.spatial import cKDTree
        seed = np.asarray(exterior_seed, float)
        valid_ids = np.flatnonzero(passable)
        distance, local_id = cKDTree(xyz[valid_ids]).query(seed)
        if distance > voxel * 1.5:
            raise ValueError('Explicit exterior seed is not within 1.5 cells of navigable ROI')
        seed_idx = np.asarray([valid_ids[local_id]], np.int64)
        seed_description = {'explicit_seed': seed.tolist(), 'nearest_distance': float(distance)}
        seed_mode = 'user-supplied explicit exterior seed'
    if not len(seed_idx):
        raise ValueError('No confirmed exterior seed points for this ROI')
    dsu = ParityUnionFind(len(xyz))
    for u, v, p in zip(src[good], dst[good], bits[good]):
        dsu.add(int(u), int(v), int(p))
    labels, union_stats = dsu.labels(passable, seed_idx)
    summary = {'grid_shape': list(shape), 'grid_voxel': float(voxel),
               'grid_nodes': int(len(passable)), 'passable_nodes': int(passable.sum()),
               'roi_rejected_edges': int(np.count_nonzero(~verified)),
               'ambiguous_triangle_edges': int(np.count_nonzero(~good)),
               'valid_triangle_edges': int(good.sum()),
               'triangle_crossing_edges': int((bits[good] == 1).sum()),
               'seed_mode': seed_mode, 'seed_count': int(len(seed_idx)),
               'seed_info': _json_native(seed_description), **union_stats}
    if union_stats['known_outside_nodes'] < 2:
        raise RuntimeError('Exterior navigation region not established')
    return {'origin': origin, 'voxel': voxel, 'shape': np.asarray(shape, np.int32),
            'labels': labels.reshape(shape), 'passable': passable.reshape(shape),
            'summary': summary}


def _query_nearby_candidates(points, index, field, scene, o3d, normals, *,
                             neighbours=8, min_agree=2, batch=3000, roi_link_samples=3):
    """Adaptive exact parity queries for Gaussian centres beyond the SDF threshold.

    Pass 1 uses a small radius and few nearby *labelled* navigation nodes.
    Only unresolved points go through larger K/radii (up to 96 nodes / 16
    voxels). The expensive R3 line-membership check adapts to the ray length.
    Every accepted label needs multiple consistent exact finite-segment parity
    paths, with NO conflicting path. No Gaussian is classified by voxel lookup
    alone. Genuine contradictions are never resolved by majority voting.

    The third physical region is NOT an output class. UNKNOWN means a failed
    numerical/topological query; the formal-clean export is blocked if any
    over-threshold centre has this state.
    """
    from scipy.spatial import cKDTree
    from tools.audit_final_r3_global_poisson_barrier import grid_centers, eval_r3, link_r3_filter

    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if not np.isfinite(pts).all():
        raise ValueError('Gaussian centres contain non-finite coordinates')
    if int(batch) < 1 or int(neighbours) < 1 or int(min_agree) < 1:
        raise ValueError('Invalid adaptive short-ray query settings')
    labels = np.asarray(index['labels'], np.int8).ravel()
    available = np.flatnonzero(labels != UNKNOWN)
    if len(available) < int(min_agree):
        raise RuntimeError('Too few verified navigation nodes to establish side labels')
    voxel = float(index['voxel'])
    centers = grid_centers(index['origin'], voxel, tuple(int(x) for x in index['shape']))
    if len(labels) != len(centers):
        raise ValueError('Navigation grid label size mismatch')
    tree = cKDTree(centers[available])
    start_k = max(int(min_agree), int(neighbours))
    raw_stages = [(start_k, 2.5), (max(start_k * 3, 24), 6.0),
                  (max(start_k * 8, 96), 16.0)]
    stages = []
    for k, radius in raw_stages:
        spec = (min(k, len(available)), float(radius))
        if spec not in stages:
            stages.append(spec)
    assigned = np.full(len(pts), UNKNOWN, np.int8)
    details = {
        'candidate_count': int(len(pts)),
        'rejected_outside_final_r3': 0,
        'not_enough_valid_paths': 0,
        'conflicting_paths': 0,
        'stage_resolved': [0 for _ in stages],
        'stage_attempted': [0 for _ in stages],
        'stages': [{'nearest_k': int(k), 'max_radius_voxels': float(r)} for k, r in stages],
        'classified_outside': 0, 'classified_inside': 0,
    }
    # Process data in batches so the temporary nearest-node matrices never
    # scale with scene size. Only unresolved points reach expensive later passes.
    for begin in range(0, len(pts), int(batch)):
        end = min(begin + int(batch), len(pts))
        cloud = pts[begin:end]
        inside_roi = eval_r3(field, cloud, batch=2048) > 0
        details['rejected_outside_final_r3'] += int(np.count_nonzero(~inside_roi))
        current = np.full(len(cloud), UNKNOWN, np.int8)
        conflicted = np.zeros(len(cloud), bool)
        for stage_id, (k, radius) in enumerate(stages):
            pending = np.flatnonzero(inside_roi & (current == UNKNOWN) & ~conflicted)
            if len(pending) == 0:
                break
            details['stage_attempted'][stage_id] += int(len(pending))
            p = cloud[pending]
            distances, nearest = tree.query(p, k=k)
            distances = np.asarray(distances, float).reshape(len(pending), k)
            nearest = np.asarray(nearest, np.int64).reshape(len(pending), k)
            node_ids = available[nearest]
            proposed = np.full((len(pending), k), UNKNOWN, np.int8)
            within_radius = distances <= (radius * voxel)
            for j in range(k):
                subset = np.flatnonzero(within_radius[:, j])
                if not len(subset):
                    continue
                from_xyz = centers[node_ids[subset, j]]
                to_xyz = p[subset]
                # Long rays in concave R3 need more samples; three samples are
                # insufficient when distant nodes are used for fallback.
                ray_length = np.linalg.norm(to_xyz - from_xyz, axis=1)
                max_length = float(ray_length.max()) if len(ray_length) else 0.0
                samples = max(int(roi_link_samples), int(np.ceil(max_length / max(voxel * .5, 1e-10))))
                samples = min(samples, 41)
                roi_ok = link_r3_filter(from_xyz, to_xyz, field,
                                        samples=samples, batch=2048)
                subset = subset[roi_ok]
                if not len(subset):
                    continue
                from_xyz = centers[node_ids[subset, j]]
                to_xyz = p[subset]
                parity, ray_ok = segment_parity(
                    scene, o3d, from_xyz, to_xyz, normals=normals,
                    endpoint_tol=max(1e-6, voxel * 1e-5), ray_batch=max(256, int(batch)))
                passed = subset[ray_ok]
                proposed[passed, j] = labels[node_ids[passed, j]] ^ parity[ray_ok]
            positive = (proposed == INSIDE).sum(axis=1)
            negative = (proposed == OUTSIDE).sum(axis=1)
            total = positive + negative
            conflicting = (positive > 0) & (negative > 0)
            conflicted[pending[conflicting]] = True
            ready = (total >= int(min_agree)) & ~conflicting
            current[pending[ready & (negative == total)]] = OUTSIDE
            current[pending[ready & (positive == total)]] = INSIDE
            details['stage_resolved'][stage_id] += int(ready.sum())
        unresolved = inside_roi & (current == UNKNOWN)
        details['not_enough_valid_paths'] += int(np.count_nonzero(unresolved & ~conflicted))
        details['conflicting_paths'] += int(np.count_nonzero(unresolved & conflicted))
        assigned[begin:end] = current
        if begin == 0 or end == len(pts) or (begin // int(batch)) % 10 == 0:
            _log(f'Adaptive short-ray parity {end:,}/{len(pts):,}; '
                 f'outside={int((assigned[:end] == OUTSIDE).sum()):,}; '
                 f'inside={int((assigned[:end] == INSIDE).sum()):,}; '
                 f'unresolved={int((assigned[:end] == UNKNOWN).sum()):,}')
    details['classified_outside'] = int((assigned == OUTSIDE).sum())
    details['classified_inside'] = int((assigned == INSIDE).sum())
    details['classified_uncertain'] = int((assigned == UNKNOWN).sum())
    return assigned, details



def _formal_export_gate(labels):
    """Require binary results for ALL super-threshold candidates before formal export."""
    labels = np.asarray(labels, dtype=np.int8).ravel()
    unresolved = int(np.count_nonzero(labels == UNKNOWN))
    invalid = int(np.count_nonzero((labels != OUTSIDE) & (labels != INSIDE) & (labels != UNKNOWN)))
    if invalid:
        raise RuntimeError(f'Invalid parity label count: {invalid}')
    if unresolved:
        raise RuntimeError(f'{unresolved} super-threshold centres unresolved; formal export is prohibited')


def _export(source, mask, destination):
    # Source is the authenticated original GSPL GaussianPLY; preserve native
    # binary Gaussian records, rather than importing the old Open-A+B exporter.
    return source.export(mask, destination)


def _check_output(output_dir, run_dir):
    output = Path(output_dir).expanduser().resolve()
    allowed = Path(run_dir).resolve() / 'cleanup'
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError('All new results must be in a NEW --output-dir under this ROI run/cleanup')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refuse to overwrite existing results: ' + str(output))
    return output


def run_roi_barrier_parity(*, manifest, preseal_report, output_dir,
                           barrier_audit_dir=None, index_cache=None,
                           classification_only=True, confirm_exterior=False,
                           chunk_size=30000, nearest_k=8, min_agree=2,
                           exterior_seed=None, config_file=None,
                           roi_link_samples=3, ray_batch=8000):
    """Complete reusable Stage-2 experiment, with source auditing and fail-closed unknowns."""
    from gsrefine.cleanup.sdf_common import robust_gaussian_radius
    from gsrefine.roi.core import GaussianPLY
    from gsrefine.cleanup.stage2_geometry_validation import GAUSSIAN_DISTANCE_DEFAULTS as DEFAULTS
    from gsrefine.geometry.r3_direct_poisson_clip import _source_r3_handoff
    from gsrefine.cleanup.sdf_common import _load_stage1_manifest
    from tools.audit_final_r3_global_poisson_barrier import run as audit_run
    import argparse

    if (chunk_size < 100 or nearest_k < min_agree or min_agree < 2 or nearest_k > 32
            or roi_link_samples < 2 or roi_link_samples > 15 or ray_batch < 100):
        raise ValueError('Invalid parity query, ray batch, or ROI path sampling settings')
    if not classification_only and not confirm_exterior:
        raise ValueError('Clean PLY export requires --confirm-exterior after reviewing classification results')
    inputs_data = _prepare_geometry(manifest, preseal_report, config_file=config_file)
    record = inputs_data['record']; inputs = inputs_data['inputs']
    run_dir = inputs_data['run_dir']; preseal = inputs_data['preseal']
    field = inputs_data['field']; up = inputs_data['up']
    original_main = inputs_data['global_main']
    o3d = inputs_data['o3d']; normals = inputs_data['normals']
    wall_scene = inputs_data['wall_scene']
    distance_scene = inputs_data['distance_scene']
    stage1, upstream, _, _ = _load_stage1_manifest(manifest)
    handoff = _source_r3_handoff(Path(run_dir), upstream)
    output = _check_output(output_dir, run_dir)
    if barrier_audit_dir is not None:
        report_dir = Path(barrier_audit_dir).expanduser().resolve(strict=True)
    else:
        # Entirely new scenes are handled by the SAME algorithm, no hand-made index.
        report_dir = output / 'geometry_audit'
        _log('No prior geometry audit supplied: running complete final R3/global Poisson audit')
        audit_run(argparse.Namespace(manifest=str(manifest), preseal_report=str(preseal_report),
            output_dir=str(report_dir), intersections_only=False, masks=None,
            sample_gaussians=7000, audit_samples=768, root_iterations=22,
            edge_zero_epsilon=1e-6, same_sign_samples=4000, eval_batch=2048,
            padding=.5, coarse_voxel=.30, fine_voxel=.17,
            coarse_max_cells=95000, fine_max_cells=250000,
            roi_link_samples=roi_link_samples, top_quantile=.9, only_scale='both'))
    audit, loops, separation = _audit_identity(
        report_dir, stage1, inputs, handoff, field=field, wall_scene=wall_scene,
        o3d=o3d, normals=normals, output=output)
    _log(f'Original-triangle region separation confirmed numerically: '
         f'{loops["closed_cycles"]} closed intersection cycle(s)')
    grid_path = report_dir / 'fine_connectivity.npz'
    if not grid_path.is_file():
        raise FileNotFoundError('Verified fine connectivity NPZ is missing from barrier audit')
    identity = _navigation_identity(audit, grid_path, roi_link_samples=roi_link_samples)
    output.mkdir(parents=True, exist_ok=True)
    if index_cache is not None and exterior_seed is not None:
        raise ValueError('Cached navigation already contains its exterior seed; do not override it')
    if index_cache is not None:
        _log('Verifying reusable navigation index source fingerprints')
        with np.load(index_cache, allow_pickle=False) as z:
            old_identity = json.loads(str(z['identity_json'].item()))
            if old_identity != identity:
                raise ValueError('Cached navigation index belongs to a different ROI / Poisson / audit grid')
            index = {'origin': np.asarray(z['origin']), 'voxel': float(z['voxel']),
                     'shape': np.asarray(z['shape'], np.int32),
                     'labels': np.asarray(z['labels'], np.int8),
                     'passable': np.asarray(z['passable'], bool),
                     'summary': json.loads(str(z['summary_json'].item()))}
    else:
        index = _make_navigation(grid_path, field, up, wall_scene, o3d, normals,
                                 exterior_seed=exterior_seed,
                                 neighbour_rays_batch=ray_batch,
                                 roi_link_samples=roi_link_samples)
    # Ensure a reusable cache can be used by Gaussian methods with SAME ROI and source geometry.
    np.savez_compressed(output / 'roi_navigation_cache.npz', origin=index['origin'],
                        voxel=index['voxel'], shape=index['shape'],
                        labels=index['labels'], passable=index['passable'],
                        identity_json=np.array(json.dumps(identity, sort_keys=True)),
                        summary_json=np.array(json.dumps(_json_native(index['summary']), sort_keys=True)))
    _log('Navigation: ' + str(index['summary']))
    # A valid two-sided partition is mandatory for formal deletion; diagnostic
    # classification can still explain a one-sided or underdetermined result.
    if not classification_only and (
            int(index['summary']['known_outside_nodes']) == 0 or
            int(index['summary']['known_inside_nodes']) == 0):
        raise RuntimeError('No reliable TWO-SIDED ROI/Poisson partition; refuse cleaned PLY export')
    model = GaussianPLY(inputs['gaussian_ply'])
    centers = np.asarray(model.positions(), np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
        raise ValueError('Invalid Gaussian center array')
    radii = robust_gaussian_radius(model, float(DEFAULTS['trim_fraction']),
                                    float(DEFAULTS['sigma_multiplier']))
    delta = float(record['sfm_fit']['p95']) + float(radii['trimmed_mean_radius'])
    epsilon = float(DEFAULTS['epsilon'])
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('Invalid adaptive SDF distance threshold')
    distances = np.empty(len(centers), np.float32)
    for a in range(0, len(centers), chunk_size):
        b = min(a + chunk_size, len(centers))
        q = o3d.core.Tensor(np.ascontiguousarray(centers[a:b], dtype=np.float32))
        result = np.asarray(distance_scene.compute_distance(q).numpy(), np.float32).ravel()
        if not np.isfinite(result).all() or len(result) != b - a:
            raise ValueError('Nonfinite distance-to-local-triangle result')
        distances[a:b] = result
        if a == 0 or b == len(centers) or (a // chunk_size) % 10 == 0:
            _log(f'ALL Gaussian exact unsigned distance {b:,}/{len(centers):,}')
    candidate = distances.astype(np.float64) > delta + epsilon
    candidate_ids = np.flatnonzero(candidate)
    _log(f'Adaptive delta={delta:.12f}; over threshold {len(candidate_ids):,}/{len(centers):,}')
    occupancy = np.full(len(centers), UNKNOWN, np.int8)
    classified, diagnostics = _query_nearby_candidates(
        centers[candidate_ids], index, field, wall_scene, o3d, normals,
        neighbours=nearest_k, min_agree=min_agree,
        batch=max(500, chunk_size // max(1, nearest_k)),
        roi_link_samples=roi_link_samples)
    occupancy[candidate_ids] = classified
    remove = candidate & (occupancy == OUTSIDE)
    keep = ~remove
    if not np.all(~remove | candidate):
        raise RuntimeError('Non-candidate Gaussian was marked for deletion')
    if np.any(remove & (occupancy != OUTSIDE)):
        raise RuntimeError('Gaussian deletion violates exterior-only rule')
    report = {
        'schema': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'CLASSIFICATION_ONLY_PREVIEW' if classification_only else 'CLEANED_EXPERIMENTAL_EXTERIOR_APPROVED',
        'inputs': {'stage1_manifest': str(stage1), 'preseal': str(preseal),
                   'main_trusted_poisson': str(record['main_observed_surface']),
                   'original_full_scene_poisson': str(inputs['poisson_mesh']),
                   'original_R3_Gaussian': str(inputs['gaussian_ply']),
                   'barrier_audit_dir': str(report_dir),
                   'source_fingerprints': identity},
        'geometry': {'intersection_topology': loops, 'region_separation': separation, 'navigation': index['summary'],
                     'original_global_main_triangles': int(len(original_main.faces)),
                     'distance_poisson_triangles': int(len(inputs_data['main'].faces))},
        'threshold': {'sfm_p95': float(record['sfm_fit']['p95']),
                      'gaussian_scale': radii, 'delta': delta, 'epsilon': epsilon,
                      'delete_rule': 'distance_to_original_local_main > delta + epsilon AND outside_from_parity'},
        'counts': {'all_gaussians': int(len(centers)),
                   'over_distance_threshold': int(candidate.sum()),
                   'below_or_equal_threshold_kept': int((~candidate).sum()),
                   'candidate_outside_remove': int(remove.sum()),
                   'candidate_inside_kept': int(np.sum(candidate & (occupancy == INSIDE))),
                   'candidate_uncertain_kept': int(np.sum(candidate & (occupancy == UNKNOWN))),
                   'all_kept': int(keep.sum())},
        'short_ray': diagnostics,
        'limitations': 'ROI Poisson barrier is NUMERICALLY audited, not mathematically watertight-certified. '
                       'Top-shell exterior requires scene verification. An unresolved query BLOCKS formal export.',
    }
    np.savez_compressed(output / 'classification_masks.npz',
                        distance_main=distances, over_threshold=candidate,
                        occupancy_label=occupancy, outside_confident=candidate & (occupancy == OUTSIDE),
                        inside_confident=candidate & (occupancy == INSIDE),
                        uncertain_kept=candidate & (occupancy == UNKNOWN),
                        remove=remove, keep=keep)
    # Source-provenance-preserving diagnostic PLYs; do not publish official Stage-1 /final.
    for name, mask in [
        ('outside_candidates.ply', candidate & (occupancy == OUTSIDE)),
        ('inside_candidates.ply', candidate & (occupancy == INSIDE)),
        ('uncertain_candidates.ply', candidate & (occupancy == UNKNOWN))]:
        if np.any(mask):
            _export(model, mask, output / name)
    # Formal SDF cleanup requires a complete verified binary inside/outside
    # classification of every over-threshold Gaussian. A failed query must
    # NEVER be silently treated as a cleanable exterior point or as a successful
    # overall result. Classification-only mode remains available for diagnosis.
    unresolved_count = int(np.sum(candidate & (occupancy == UNKNOWN)))
    if not classification_only and unresolved_count:
        report['status'] = 'BLOCKED_INCOMPLETE_PARITY_CLASSIFICATION'
        report['blocking_reason'] = (
            f'{unresolved_count} over-threshold Gaussian centres remain geometrically '
            'unresolved after adaptive multi-stage short-ray checks. No cleaned PLY exported.')
    elif not classification_only:
        _formal_export_gate(occupancy[candidate])
        report['cleaned_ply'] = _export(model, keep, output / 'gaussian_roi_clean.ply')
    _write_json(output / 'roi_parity_report.json', report)
    _log('Saved report: ' + str(output / 'roi_parity_report.json'))
    if report['status'].startswith('BLOCKED'):
        _log(report['blocking_reason'])
    return {'status': report['status'], 'report': str(output / 'roi_parity_report.json'),
            'removed': int(remove.sum()), 'kept': int(keep.sum()), 'delta': delta,
            'cleaned_ply': str(output / 'gaussian_roi_clean.ply') if report['status'] == 'CLEANED_EXPERIMENTAL_EXTERIOR_APPROVED' else None}

# stage2_region_unified_original_main_v1_20261112
