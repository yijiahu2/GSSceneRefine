#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only source-backed diagnosis of FINAL R3 x ORIGINAL largest Poisson wall.

This is NOT the Stage-2 parity classifier or a relaxation of its flood gates.
The source audit NPZs contain connected regions and witness paths, and this tool
exports their spatial arrangement for CloudCompare. Optional verification uses
real original largest-component triangle intersections and the reconstituted R3
field; it NEVER changes a classification mask or exports cleaned Gaussian PLY.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

SCHEMA = 'GSSceneRefine.R3_wall_connectivity_diagnosis.v1'
PALETTE = np.asarray([
    [32, 117, 207], [225, 127, 50], [60, 176, 113], [184, 90, 192],
    [213, 183, 64], [47, 179, 195], [213, 93, 107], [137, 154, 71],
    [170, 170, 170], [151, 95, 69], [180, 164, 206], [100, 140, 205]
], dtype=np.uint8)


def sha256(path):
    d = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            d.update(b)
    return d.hexdigest()


def write_point_ply(path, xyz, rgb):
    """Ordinary colored XYZ PLY, visualization-only; NOT a Gaussian file."""
    xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, np.uint8)
    if rgb.shape == (3,):
        rgb = np.broadcast_to(rgb, (len(xyz), 3))
    if rgb.shape != xyz.shape:
        raise ValueError('RGB and xyz shapes differ')
    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                      ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    records = np.empty(len(xyz), dtype=dtype)
    for k, col in enumerate(('x', 'y', 'z')):
        records[col] = xyz[:, k]
    for k, col in enumerate(('red', 'green', 'blue')):
        records[col] = rgb[:, k]
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment GSSceneRefine spatial diagnostics only - ordinary XYZ colored point cloud\n'
              'element vertex %d\nproperty float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n') % len(xyz)
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(records.tobytes())


def centers(origin, voxel, shape):
    indices = np.indices(shape, dtype=np.int32).reshape(3, -1).T
    return origin[None, :] + (indices + 0.5) * voxel


def shell_points(passable):
    from scipy.ndimage import binary_erosion, generate_binary_structure
    return passable & ~binary_erosion(passable, structure=generate_binary_structure(3, 1), border_value=0)


def top_bottom(passable, xyz, up, quantile):
    p = np.flatnonzero(shell_points(passable).ravel())
    if len(p) < 12:
        raise ValueError('Too few ROI shell cells to reconstruct top/bottom probes')
    heights = xyz[p] @ up
    top = p[heights >= np.quantile(heights, quantile)]
    bottom = p[heights <= np.quantile(heights, 1 - quantile)]
    return top, bottom


def counts_by_region(labels, ids):
    vals = np.asarray(labels).reshape(-1)[np.asarray(ids, dtype=np.int64)]
    return {int(i): int(n) for i, n in sorted(Counter(map(int, vals)).items()) if i >= 0}


def summarize_grid(npz_path, up, quantile, path_xyz=None):
    with np.load(npz_path, allow_pickle=False) as z:
        origin = np.asarray(z['origin'], np.float64)
        h = float(z['voxel'])
        shape = tuple(map(int, z['shape']))
        inside = np.asarray(z['final_r3_inside'], bool)
        passed = np.asarray(z['passable'], bool)
        comps = np.asarray(z['components'], np.int32)
    if origin.shape != (3,) or not np.isfinite(origin).all() or h <= 0:
        raise ValueError('Invalid grid geometry')
    if inside.shape != shape or passed.shape != shape or comps.shape != shape:
        raise ValueError('Cached flood grid shape mismatch')
    if not np.array_equal(comps.ravel() >= 0, passed.ravel()):
        raise ValueError('Component labels and passable mask are inconsistent')
    xyz = centers(origin, h, shape)
    top, bottom = top_bottom(passed, xyz, up, quantile)
    labels = comps.ravel()
    valid = labels >= 0
    sizes = Counter(map(int, labels[valid]))
    biggest = sorted(sizes, key=lambda k: (-sizes[k], k))
    summary = {
        'voxel': h, 'shape': list(shape), 'passable_nodes': int(valid.sum()),
        'connected_regions': len(sizes), 'component_sizes': {int(k): int(v) for k, v in sorted(sizes.items())},
        'largest_region_ids': list(map(int, biggest[:2])),
        'upper_seed_counts_by_region': counts_by_region(labels, top),
        'lower_seed_counts_by_region': counts_by_region(labels, bottom),
        'top_bottom_shared_regions': [int(x) for x in sorted((set(labels[top]) & set(labels[bottom])) - {-1})],
        'top_seed_n': len(top), 'bottom_seed_n': len(bottom),
    }
    path = None
    if path_xyz and Path(path_xyz).is_file():
        path = np.loadtxt(path_xyz, ndmin=2, dtype=float).reshape(-1, 3)
        ijk = np.rint((path - origin) / h - 0.5).astype(int)
        within = np.all((ijk >= 0) & (ijk < np.asarray(shape)), axis=1)
        path_labels = []
        for p, ix, ok in zip(path, ijk, within):
            if ok and np.linalg.norm(p - (origin+(ix + 0.5)*h)) < h * 1e-3:
                path_labels.append(int(comps[tuple(ix)]))
            else:
                path_labels.append(None)
        summary['witness_path_vertices'] = len(path)
        summary['witness_component_ids'] = sorted(set(i for i in path_labels if i is not None))
        summary['witness_all_nodes_same_region'] = len(set(path_labels)) == 1 and None not in path_labels
        if path_labels:
            summary['witness_start_region'] = path_labels[0]
            summary['witness_end_region'] = path_labels[-1]
    return {'origin': origin, 'voxel': h, 'shape': shape, 'xyz': xyz,
            'inside': inside, 'passable': passed, 'labels': labels,
            'top': top, 'bottom': bottom, 'largest': biggest,
            'path': path, 'summary': summary}


def sample_segments(segments, spacing):
    """Convert segments into colored CloudCompare point samples."""
    seg = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 3)
    pieces = []
    for a, b in seg:
        n = max(2, min(30, int(np.ceil(np.linalg.norm(b-a)/spacing)) + 1))
        pieces.append(a[None, :] + np.linspace(0, 1, n)[:, None] * (b-a)[None, :])
    return np.concatenate(pieces, axis=0) if pieces else np.empty((0, 3))


def export_grid(item, out, name):
    xyz, labels, passed = item['xyz'], item['labels'], item['passable'].ravel()
    color = PALETTE[labels[passed] % len(PALETTE)]
    write_point_ply(out/f'{name}_connected_regions.ply', xyz[passed], color)
    write_point_ply(out/f'{name}_upper_shell_seeds.ply', xyz[item['top']], [29, 130, 255])
    write_point_ply(out/f'{name}_lower_shell_probes.ply', xyz[item['bottom']], [255, 90, 40])
    s = shell_points(item['inside'])
    write_point_ply(out/f'{name}_sampled_R3_shell.ply', xyz[s.ravel()], [128, 205, 218])
    if item['path'] is not None:
        # Render virtual witness polyline as dense colored points; do not modify geometry.
        p = item['path']
        lines = np.stack([p[:-1], p[1:]], axis=1)
        write_point_ply(out/f'{name}_top_bottom_witness.ply',
                        sample_segments(lines, item['voxel']/8), [254, 41, 150])
        write_point_ply(out/f'{name}_witness_endpoints.ply',
                        np.stack([p[0], p[-1]]), np.asarray([[29, 130, 255], [255, 90, 40]]))


def get_pairs(a, b, xyz, voxel, max_pairs, radius_factor):
    from scipy.spatial import cKDTree
    ai, bi = np.flatnonzero(a), np.flatnonzero(b)
    if not len(ai) or not len(bi):
        return np.empty((0, 2, 3)), {'closest_distance': None}
    b_tree = cKDTree(xyz[bi])
    distances, nearest = b_tree.query(xyz[ai], k=1, workers=-1)
    ordering = np.argsort(distances, kind='stable')
    # Avoid 100 almost identical chords at a single spot on the wall.
    probe_candidates = ordering[distances[ordering] <= radius_factor * voxel]
    chosen = []
    min_spacing = 1.75 * voxel
    for ix in probe_candidates:
        if all(np.linalg.norm(xyz[ai[ix]] - xyz[ai[q]]) >= min_spacing for q in chosen):
            chosen.append(ix)
            if len(chosen) == max_pairs:
                break
    # If wall is very small, use all available distinct nearest pairs without duplicates.
    pairs = np.stack([xyz[ai[chosen]], xyz[bi[nearest[chosen]]]], axis=1) if chosen else np.empty((0, 2, 3))
    return pairs, {'closest_distance': float(np.min(distances)),
                   'pairs_within_radius': int(len(probe_candidates)),
                   'spatially_separated_probes': int(len(chosen)),
                   'radius_limit': float(radius_factor*voxel)}


def verify_with_original_triangles(manifest, audit, item, output, *, max_pairs=120, radius_factor=5.0):
    """Use original largest triangle wall and EXACT stage-1 R3 for cross-region chords.

    This is evidence for whether regions are geometrically opposite sides; it
    cannot establish a strict topological seal or change the Stage-2 gate.
    """
    import trimesh
    from gsrefine.cleanup.sdf_common import _load_stage1_manifest
    from gsrefine.geometry.r3_direct_poisson_clip import _restore_r3_field, _source_r3_handoff
    from gsrefine.cleanup.roi_barrier_parity import segment_parity
    from tools.audit_final_r3_global_poisson_barrier import (
        make_scene, largest_original_component, link_r3_filter, eval_r3)

    mf, upstream, inputs, run_dir = _load_stage1_manifest(str(manifest))
    r3_record = _source_r3_handoff(Path(run_dir), upstream)
    if sha256(r3_record) != audit['inputs']['r3_sha256']:
        raise ValueError('Stage-1 R3 handoff changed since source flood audit')
    global_mesh, r3, replay = _restore_r3_field(inputs=inputs, upstream=upstream,
                        run_dir=Path(run_dir), audit_record=r3_record, audit_samples=128)
    wall, wall_stats = largest_original_component(global_mesh)
    if len(wall.faces) != audit['source_component']['largest_source_component_faces']:
        raise ValueError('Original Poisson largest-component size differs from audit')
    # This is the ORIGINAL full-scene Poisson largest connected component, not R3 clipped Poisson.
    wall.export(str(output/'original_global_poisson_largest_body.ply'))
    scene, o3d = make_scene(wall)
    face = np.asarray(wall.faces, np.int64)
    pos = np.asarray(wall.vertices, np.float64)
    normals = np.cross(pos[face[:, 1]]-pos[face[:, 0]], pos[face[:, 2]]-pos[face[:, 0]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
    summary = {'original_largest_body_faces': int(len(wall.faces)),
               'r3_handoff_max_error': replay.get('v3_handoff_max_field_distance_error', None)}
    if len(item['largest']) < 2:
        summary['region_pair_status'] = 'INSUFFICIENT_CONNECTED_REGIONS'
        return summary
    ia, ib = item['largest'][:2]
    pairs, pair_meta = get_pairs(item['labels']==ia, item['labels']==ib, item['xyz'],
                                item['voxel'], max_pairs, radius_factor)
    summary.update({'region_pair': [int(ia), int(ib)], **pair_meta})
    if not len(pairs):
        summary['region_pair_status'] = 'NO_NEARBY_PAIRS_AT_SELECTED_SPACING'
        return summary
    # Both endpoints are sampled passable points; intermediate R3 is checked anew.
    acceptable = link_r3_filter(pairs[:, 0], pairs[:, 1], r3, batch=1024, samples=11)
    filtered = pairs[acceptable]
    summary['pair_chords_inside_R3_by_sample'] = int(len(filtered))
    if not len(filtered):
        summary['region_pair_status'] = 'NO_VALID_R3_CHORDS'
        return summary
    bits, reliable = segment_parity(scene, o3d, filtered[:,0], filtered[:,1],
                         normals=normals, endpoint_tol=max(1e-6,item['voxel']*1e-5))
    summary['ray_pair_reliable'] = int(np.count_nonzero(reliable))
    summary['ray_pair_odd'] = int(np.count_nonzero(reliable & (bits==1)))
    summary['ray_pair_even'] = int(np.count_nonzero(reliable & (bits==0)))
    summary['ray_pair_unreliable'] = int(np.count_nonzero(~reliable))
    if summary['ray_pair_reliable'] >= 10:
        odd_ratio = summary['ray_pair_odd']/summary['ray_pair_reliable']
        if odd_ratio >= .9:
            summary['region_pair_status'] = 'DIFFERENT_SIDES_SUPPORTED_BY_TRIANGLE_PARITY'
        elif odd_ratio <= .1:
            summary['region_pair_status'] = 'SAME_SIDE_SUPPORTED_BY_TRIANGLE_PARITY'
        else:
            summary['region_pair_status'] = 'MIXED_OR_INCONSISTENT_PARITY_NEEDS_REVIEW'
    else:
        summary['region_pair_status'] = 'TOO_FEW_RELIABLE_CHORDS'
    colors = np.tile(np.asarray([165,165,165],np.uint8),(len(filtered)*2,1))
    colors[0::2] = [26, 117, 219]
    colors[1::2] = [244, 160, 49]
    write_point_ply(output/'fine_cross_region_probe_endpoints.ply',filtered.reshape(-1,3),colors)
    write_point_ply(output/'fine_cross_region_probe_lines.ply',
                    sample_segments(filtered,item['voxel']/6),[178,63,204])
    with open(output/'cross_region_chord_parity.csv','w',encoding='utf-8') as f:
        f.write('ax,ay,az,bx,by,bz,length,R3_sampled_valid,triangle_parity,ray_reliable\n')
        for pair, bit, good in zip(filtered, bits, reliable):
            f.write(','.join(map(str,[*pair[0],*pair[1],float(np.linalg.norm(pair[1]-pair[0])),1,int(bit),int(good)]))+'\n')
    path = item['path']
    if path is not None and len(path)>1:
        a,b=path[:-1],path[1:]
        r3_good=link_r3_filter(a,b,r3,batch=1024,samples=11)
        path_bits,path_reliable=segment_parity(scene,o3d,a,b,normals=normals,
                                          endpoint_tol=max(1e-6,item['voxel']*1e-5))
        summary['witness_R3_sampled_links_valid'] = int(r3_good.sum())
        summary['witness_link_count'] = len(a)
        summary['witness_crossing_odd_links'] = int(np.count_nonzero(path_reliable&(path_bits==1)))
        summary['witness_unreliable_triangle_links'] = int(np.count_nonzero(~path_reliable))
        summary['witness_all_links_no_triangle_crossings'] = bool(np.all(r3_good & path_reliable & (path_bits==0)))
    return summary


def run(args):
    from gsrefine.cleanup.sdf_common import _load_stage1_manifest
    from gsrefine.cleanup.roi_barrier_parity import _validate_intersection_loop
    aud = Path(args.audit_dir).expanduser().resolve(strict=True)
    reportfile = aud/'roi_global_poisson_barrier_report.json'
    report=json.loads(reportfile.read_text(encoding='utf8'))
    if report.get('status') != 'FLOOD_SAMPLING_COMPLETED_NOT_CERTIFIED_SEALED':
        raise ValueError('A completed geometry audit (not classification) is required')
    mf, upstream, inputs, run_dir = _load_stage1_manifest(args.manifest)
    if sha256(mf) != report['inputs']['manifest_sha256']:
        raise ValueError('Audit manifest SHA mismatch')
    if sha256(inputs['poisson_mesh']) != report['inputs']['original_poisson_sha256']:
        raise ValueError('Audit original global Poisson SHA mismatch')
    if Path(inputs['gaussian_ply']).resolve() != Path(report['inputs']['final_roi_gaussian']).resolve():
        raise ValueError('Audit ROI Gaussian source mismatch')
    roi_info=json.loads(Path(inputs['roi_json']).read_text(encoding='utf8'))
    up=np.asarray(roi_info['annotation_metadata']['up_vector'],np.float64)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up)<1e-10:
        raise ValueError('ROI orientation metadata not verified')
    up=up/np.linalg.norm(up)
    out=Path(args.output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Refuse to overwrite earlier diagnostic: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    try:
        loop = _validate_intersection_loop(aud/'global_poisson_final_r3_intersection.npz')
        loop['topology_validated'] = True
    except (ValueError, FileNotFoundError) as exc:
        loop = {'topology_validated': False, 'error': f'{type(exc).__name__}: {exc}'}
        print('[diagnosis] intersection topology cannot be confirmed:', loop['error'], flush=True)
    with np.load(aud/'global_poisson_final_r3_intersection.npz',allow_pickle=False) as z:
        seg=np.asarray(z['segment_endpoints'],float)
    h=float(report['flood']['fine']['grid']['voxel'])
    write_point_ply(out/'actual_R3_wall_intersection_samples.ply',
                    sample_segments(seg, h/8),[250,54,59])
    details={}
    for level in ('coarse','fine'):
        x=summarize_grid(aud/f'{level}_connectivity.npz',up,float(args.quantile),
               aud/f'{level}_leak_path.xyz')
        export_grid(x,out,level)
        details[level]=x
        print(f'[{level}] components={x["summary"]["connected_regions"]}; '
              f'top={x["summary"]["upper_seed_counts_by_region"]}; '
              f'bottom={x["summary"]["lower_seed_counts_by_region"]}',flush=True)
    ray=None
    if not args.no_triangle_probe:
        print('[diagnosis] checking original largest Poisson triangle side relations...',flush=True)
        ray=verify_with_original_triangles(args.manifest, report,details['fine'],out,
                 max_pairs=args.max_probe_pairs,radius_factor=args.max_distance_in_voxels)
        print('[diagnosis] side relation:',ray['region_pair_status'],flush=True)
    result={'schema':SCHEMA,'status':'DIAGNOSTIC_ONLY_NOT_CLEANED_NOT_CERTIFIED_SEALED',
        'original_audit':str(reportfile), 'manifest':str(mf),
        'original_global_poisson_largest_faces':report['source_component']['largest_source_component_faces'],
        'intersection_topology':loop,
        'levels':{k:v['summary'] for k,v in details.items()},
        'original_triangle_region_probe':ray,
        'interpretation':{
            'note':'High and low shell probes can occupy the same space region; '
                   'that alone does NOT show a wall leak nor certify a working barrier.',
            'top_bottom_gate_changed':False,
            'parity_classifier_changed':False,
            'gaussian_model_changed':False,
            'outputs':'Colored XYZ PLYs are ONLY CloudCompare geometry evidence, not SuperSplat Gaussians.',
            'required_next_review':'Overlay largest original Poisson, intersection samples, '
                                   'two main region clouds, upper/lower seeds and existing witness paths.'}}
    (out/'r3_wall_connectivity_diagnosis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('[diagnosis] saved:',out,flush=True)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,help='Stage-1 final R3 manifest.json')
    p.add_argument('--audit-dir',required=True,help='Existing full coarse+fine audit (reuses caches)')
    p.add_argument('--output-dir',required=True,help='NEW empty directory under this ROI cleanup')
    p.add_argument('--no-triangle-probe',action='store_true',help='Only inspect cached connected regions without original mesh ray test')
    p.add_argument('--quantile',type=float,default=.90,help='Original audit top/bottom seed quantile (default .90)')
    p.add_argument('--max-probe-pairs',type=int,default=120)
    p.add_argument('--max-distance-in-voxels',type=float,default=5.0)
    args=p.parse_args(argv)
    if not .5<args.quantile<1:p.error('--quantile must be between .5 and 1')
    if args.max_probe_pairs<10 or args.max_distance_in_voxels<=0:p.error('Invalid probe limits')
    return run(args)

if __name__=='__main__':
    main()
