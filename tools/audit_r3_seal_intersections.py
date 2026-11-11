"""Read-only 2D / 3D intersection localization for GSSceneRefine R3 main sealing.

Usage (from GSSceneRefine root):
    python -m tools.audit_r3_seal_intersections --preseal-report /.../preseal_diagnostics.json

No Poisson/R3/Gaussian files are modified. The tool recreates candidate closure
in memory using the project's canonical seal_main, then labels each true 3D
intersecting triangle pair by observed surface / artificial skirt / cap.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np


def _as_json(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _make_o3d_mesh(o3d, vertices, faces):
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, np.float64))
    m.triangles = o3d.utility.Vector3iVector(np.asarray(faces, np.int32))
    return m


def intersection_pairs(o3d, vertices, faces):
    """Real face-pair IDs, not merely the boolean is_self_intersecting()."""
    mesh = _make_o3d_mesh(o3d, vertices, faces)
    if not hasattr(mesh, 'get_self_intersecting_triangles'):
        raise RuntimeError('Installed Open3D lacks get_self_intersecting_triangles()')
    pairs = np.asarray(mesh.get_self_intersecting_triangles(), dtype=np.int64)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError('Open3D face intersection result is not Nx2')
    n = len(faces)
    if np.any(pairs < 0) or np.any(pairs >= n):
        raise ValueError('Open3D returned face IDs outside the mesh')
    # Open3D should return unique unordered pairs. Defensive de-duplication.
    return np.unique(np.sort(pairs, axis=1), axis=0)


def label_faces(n_main, n_skirt, n_total):
    ids = np.empty(n_total, dtype='U8')
    ids[:n_main] = 'main'
    ids[n_main:n_main+n_skirt] = 'skirt'
    ids[n_main+n_skirt:] = 'cap'
    return ids


def pair_categories(pairs, n_main, n_skirt, n_total):
    ids = label_faces(n_main, n_skirt, n_total)
    a = ids[pairs[:, 0]]
    b = ids[pairs[:, 1]]
    return ['-'.join(sorted((x, y))) for x, y in zip(a, b)]


def projected_segment_crossings(uv, xyz, *, max_examples=30):
    """Find non-adjacent segment intersections; report 2D and SOURCE-3D edges.

    Shapely handles collinear overlaps/endpoint touches and robust predicates;
    AABB vector prefilter avoids a quadratic number of GEOS calls.
    """
    from shapely.geometry import LineString
    uv = np.asarray(uv, np.float64)
    xyz = np.asarray(xyz, np.float64)
    n = len(uv)
    if len(xyz) != n or n < 3:
        raise ValueError('Expected corresponding 3D/2D closed ring of >=3 vertices')
    ends = np.roll(uv, -1, axis=0)
    lows = np.minimum(uv, ends)
    highs = np.maximum(uv, ends)
    lines = [LineString((uv[i], ends[i])) for i in range(n)]
    counts = Counter()
    examples = []
    for i in range(n-1):
        js = np.arange(i + 2, n, dtype=np.int64)
        if i == 0:
            js = js[js != n-1]
        intersects_bbox = np.all(lows[i] <= highs[js], axis=1) & np.all(lows[js] <= highs[i], axis=1)
        for j in js[intersects_bbox]:
            j = int(j)
            if not lines[i].intersects(lines[j]):
                continue
            x = lines[i].intersection(lines[j])
            if x.is_empty:
                continue
            typ = 'overlap' if x.length > 1e-12 else ('point' if x.geom_type == 'Point' else x.geom_type)
            counts[typ] += 1
            if len(examples) >= max_examples:
                continue
            center = (np.asarray(x.coords[0]).tolist() if x.geom_type == 'Point'
                      else list(x.centroid.coords)[0])
            # At a 2D crossing, determine whether the two 3D boundary arcs
            # actually meet or only overlap under projection. This is the
            # key distinction between genuine 3D and projected 2D crossing.
            xy = np.asarray(center, dtype=np.float64)
            d_i, d_j = ends[i]-uv[i], ends[j]-uv[j]
            ti = float(np.dot(xy-uv[i], d_i)/max(np.dot(d_i,d_i), 1e-30))
            tj = float(np.dot(xy-uv[j], d_j)/max(np.dot(d_j,d_j), 1e-30))
            xyz_i = xyz[i] + np.clip(ti, 0.0, 1.0) * (xyz[(i+1)%n]-xyz[i])
            xyz_j = xyz[j] + np.clip(tj, 0.0, 1.0) * (xyz[(j+1)%n]-xyz[j])
            spatial_gap = float(np.linalg.norm(xyz_i-xyz_j))
            examples.append({
                'source_3d_gap_at_projected_crossing':spatial_gap,
                'source_3d_point_on_segment_i':xyz_i.tolist(),
                'source_3d_point_on_segment_j':xyz_j.tolist(),
                'segment_i':i, 'segment_j':j, 'contact_type':typ,
                'intersection_uv':list(center),
                'segment_i_xyz': [xyz[i].tolist(), xyz[(i+1)%n].tolist()],
                'segment_j_xyz': [xyz[j].tolist(), xyz[(j+1)%n].tolist()],
                'segment_i_uv': [uv[i].tolist(), ends[i].tolist()],
                'segment_j_uv': [uv[j].tolist(), ends[j].tolist()],
            })
    return {'count':int(sum(counts.values())), 'by_type':dict(counts),
            'examples':examples, 'max_examples':max_examples}


def _plane_audit(uv, xyz, *, examples=20):
    from shapely.geometry import Polygon
    from shapely.validation import explain_validity
    poly = Polygon(uv)
    span = np.ptp(uv, axis=0)
    clearance=float(poly.minimum_clearance)
    near_tangent=not np.isfinite(clearance) or clearance <= 1e-7*max(float(np.linalg.norm(span)),1.0)
    ok = (not poly.is_empty and poly.is_valid and poly.exterior.is_simple
          and poly.area > 1e-6*float(np.prod(span)) and not near_tangent)
    record = {'valid_polygon':bool(ok), 'validity_reason':explain_validity(poly),
              'polygon_area':float(poly.area), 'projected_span':span.tolist(),
              'minimum_clearance':clearance, 'near_tangent':bool(near_tangent),
              'projected_edge_pairs':None}
    if not ok:
        record['projected_edge_pairs'] = projected_segment_crossings(uv, xyz, max_examples=examples)
    return record


def _generate_candidates(main, ring_xyz, up, opts):
    from gsrefine.cleanup.sdf_common import _local_basis
    B0 = _local_basis(up)
    result = [{'tilt_deg':0.0, 'azimuth_deg':0.0, 'normal_xyz': B0[:,2].tolist(), 'B':B0}]
    source_uv = ring_xyz @ B0[:,:2]
    primary = _plane_audit(source_uv, ring_xyz)
    if primary['valid_polygon']:
        return result, [{'tilt_deg':0.0, 'azimuth_deg':0.0, **primary}]
    # Match canonical fallback: max 25°, step 5°, 16 azimuth directions.
    scored = []
    audits = [{'tilt_deg':0.0, 'azimuth_deg':0.0, **primary}]
    n_az = int(opts['seal_tilt_azimuths'])
    for tilt in np.arange(float(opts['seal_tilt_step_deg']),
                          float(opts['seal_max_tilt_deg'])+1e-8,
                          float(opts['seal_tilt_step_deg'])):
        for idx in range(n_az):
            azi = 360.0*idx/n_az
            a, t = np.deg2rad(azi), np.deg2rad(tilt)
            vec = (np.cos(t)*B0[:,2] + np.sin(t)*(np.cos(a)*B0[:,0] + np.sin(a)*B0[:,1]))
            vec /= np.linalg.norm(vec)
            B = _local_basis(vec)
            check = _plane_audit(ring_xyz@B[:,:2], ring_xyz, examples=5)
            audits.append({'tilt_deg':float(tilt), 'azimuth_deg':azi, **check})
            if check['valid_polygon']:
                scored.append((float(tilt), -check['minimum_clearance'], azi, B))
    scored.sort(key=lambda row: (row[0],row[1]))
    for tilt, _, azi, B in scored:
        result.append({'tilt_deg':tilt, 'azimuth_deg':azi, 'normal_xyz':B[:,2].tolist(), 'B':B})
    return result, audits


def reconstruct_seal(main, B, sign, margin_fraction):
    """Exact production construction; restore all observed vertices without drift."""
    from gsrefine.geometry.sdf_sealing import seal_main
    oriented = main.copy()
    oriented.vertices = np.asarray(main.vertices) @ B
    span = float(np.ptp(np.asarray(oriented.vertices)[:,2]))
    sealed, closure, _ = seal_main(oriented, 2, sign, margin_fraction*span)
    n_old = len(main.vertices)
    if closure.get('mesh_before_vertices') != n_old or closure.get('mesh_before_faces') != len(main.faces):
        raise ValueError('Unexpected mesh count after reconstruction')
    new_xyz = np.asarray(sealed.vertices) @ B.T
    new_xyz[:n_old] = np.asarray(main.vertices)
    sealed.vertices = new_xyz
    return sealed, closure


def _read_roi_up(record):
    stage1_manifest = Path(record['stage1_manifest'])
    m = json.loads(stage1_manifest.read_text(encoding='utf8'))
    roi = Path(m['stage2_inputs']['roi_json'])
    payload = json.loads(roi.read_text(encoding='utf8'))
    meta = payload['annotation_metadata']
    if meta.get('orientation_verified') is not True:
        raise ValueError('Stage-1 Up vector was not verified')
    v = np.asarray(meta['up_vector'], np.float64)
    if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-12:
        raise ValueError('Invalid up vector')
    return v / np.linalg.norm(v)


def _write_csv(path, rows, columns):
    with open(path, 'w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row.get(k), (list, dict))
                             else row.get(k, '') for k in columns})


def analyze(preseal_path, output_dir=None, max_directions=8, max_pair_samples=50,
            write_face_plys=True):
    import open3d as o3d
    import trimesh
    from gsrefine.geometry.sdf_sealing import boundary_loops

    preseal = Path(preseal_path).expanduser().resolve(strict=True)
    record = json.loads(preseal.read_text(encoding='utf8'))
    if record.get('schema') != 'GSSceneRefine.direct_r3_stage2.v1':
        raise ValueError('Not a canonical direct-R3 Stage2 preseal report')
    main_file = preseal.parent / 'poisson_roi_main.ply'
    if not main_file.is_file():
        raise FileNotFoundError('Cannot find main layer alongside preseal report: '+str(main_file))
    main = trimesh.load(main_file, process=False, force='mesh')
    loops, details = boundary_loops(main)
    if len(loops)!=1:
        raise ValueError('Expected exactly one original R3 boundary ring')
    if len(main.faces)!=int(record['topology']['mesh_faces']):
        raise ValueError('Main layer face count is inconsistent with preseal report')
    ring_xyz = np.asarray(main.vertices)[loops[0]]
    up = _read_roi_up(record)
    opts = {
        'seal_max_tilt_deg':25.0,'seal_tilt_step_deg':5.0,'seal_tilt_azimuths':16,
        'bottom_margin_fraction':0.10
    }
    candidates, projections = _generate_candidates(main, ring_xyz, up, opts)
    chosen = candidates[:1] + candidates[1:1+max_directions]

    output = Path(output_dir).expanduser().resolve() if output_dir else preseal.parent / 'intersection_diagnostic'
    output.mkdir(parents=True, exist_ok=True)
    main_pairs = intersection_pairs(o3d, main.vertices, main.faces)
    audited = {'schema':'GSSceneRefine.r3_seal_intersection_audit.v1',
               'status':'READ_ONLY_NOT_REPAIRED',
               'created_utc':datetime.now(timezone.utc).isoformat(),
               'preseal_report':str(preseal), 'main_surface':str(main_file),
               'up_vector':up.tolist(), 'main_mesh_triangles':len(main.faces),
               'original_boundary_edges':len(loops[0]),
               'main_intrinsic_boolean':bool(_make_o3d_mesh(o3d,main.vertices,main.faces).is_self_intersecting()),
               'main_intrinsic_self_intersection_pairs':len(main_pairs),
               'main_intrinsic_pair_samples':main_pairs[:max_pair_samples].tolist(),
               'projection_trial_count':len(projections),
               'simple_projection_count':sum(x['valid_polygon'] for x in projections),
               'closure_trials':[],
               'interpretation':'Shared boundary faces are expected to touch; Open3D returns non-trivial intersecting face pairs. Artificial layers are labeled by appended face-ID ranges.'}

    two_d_rows = []
    for d in projections:
        two_d_rows.append({k:v for k,v in d.items() if k!='projected_edge_pairs'} | {
            'crossing_count': (d.get('projected_edge_pairs') or {}).get('count', 0)})
    _write_csv(output/'projection_2d_trials.csv', two_d_rows,
               ['tilt_deg','azimuth_deg','valid_polygon','validity_reason','polygon_area','minimum_clearance','crossing_count','projected_span'])
    # The first (COLMAP-Up) crossing set includes ORIGINAL 3D endpoints.
    (output/'original_up_2d_crossings.json').write_text(
        json.dumps(projections[0], default=_as_json, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
    (output/'main_intrinsic_3d_pairs.csv').write_text('triangle_a,triangle_b\n' +
        ''.join(f'{a},{b}\n' for a,b in main_pairs), encoding='utf8')
    all_3d_csv = []
    for rank, candidate in enumerate(chosen):
        tilt, az = candidate['tilt_deg'],candidate['azimuth_deg']
        for sign in (1,-1):
            row={'candidate_rank':rank,'tilt_deg':tilt,'azimuth_deg':az,'cap_sign':sign,
                 'normal_xyz':candidate['normal_xyz']}
            if rank == 0 and not projections[0]['valid_polygon']:
                row.update(status='SKIPPED_INVALID_2D', reason=projections[0]['validity_reason'])
                audited['closure_trials'].append(row)
                continue
            try:
                solid, close = reconstruct_seal(main, candidate['B'], sign, opts['bottom_margin_fraction'])
                n_main = len(main.faces)
                n_skirt = int(close['skirt_face_count'])
                pairs = intersection_pairs(o3d, solid.vertices, solid.faces)
                labels = pair_categories(pairs,n_main,n_skirt,len(solid.faces))
                groups = dict(Counter(labels))
                row.update(status='AUDITED', triangle_count=int(len(solid.faces)),
                           total_intersection_pairs=len(pairs),
                           open3d_boolean=bool(_make_o3d_mesh(o3d,solid.vertices,solid.faces).is_self_intersecting()),
                           by_layer_pair=groups,
                           watertight=bool(solid.is_watertight),
                           is_volume=bool(solid.is_volume),
                           pair_examples=[]) 
                tri = np.asarray(solid.faces)
                verts = np.asarray(solid.vertices)
                for (face_a,face_b), cat in zip(pairs,labels):
                    a, b = int(face_a),int(face_b)
                    mid_a = verts[tri[a]].mean(axis=0)
                    mid_b = verts[tri[b]].mean(axis=0)
                    pos = (mid_a + mid_b)*.5
                    example = {'face_a':a,'face_b':b,'category':cat,
                               'approx_midpoint_of_face_centroids_xyz':pos.tolist(),
                               'note':'Approximate localization, not exact intersection point'}
                    if len(row['pair_examples']) < max_pair_samples:
                        row['pair_examples'].append(example)
                    all_3d_csv.append({'rank':rank,'tilt_deg':tilt,'azimuth_deg':az,'cap_sign':sign,
                                       'face_a':a,'face_b':b,'category':cat,
                                       'approx_xyz':pos.tolist()})
                if write_face_plys and rank == 1 and sign == 1 and len(pairs):
                    face_ids = np.unique(pairs.ravel())
                    for category in ('main','skirt','cap'):
                        a = n_main
                        ids = face_ids[(face_ids < n_main) if category=='main'
                                       else ((face_ids>=a)&(face_ids<a+n_skirt) if category=='skirt'
                                             else (face_ids>=a+n_skirt))]
                        if len(ids):
                            subset = solid.submesh([ids], append=True, repair=False)
                            subset.export(output/f'debug_intersecting_faces_{category}.ply')
                audited['closure_trials'].append(row)
            except Exception as e:
                row.update(status='CONSTRUCTION_FAILED',reason=f'{type(e).__name__}: {e}')
                audited['closure_trials'].append(row)
    _write_csv(output/'intersections_3d_pairs.csv', all_3d_csv,
               ['rank','tilt_deg','azimuth_deg','cap_sign','face_a','face_b','category','approx_xyz'])
    report = output/'intersection_audit.json'
    report.write_text(json.dumps(audited,indent=2,ensure_ascii=False,default=_as_json)+'\n',encoding='utf8')
    return {'status':'READ_ONLY_INTERSECTION_AUDIT_COMPLETE','report':str(report),
            'original_up_crossings':str(output/'original_up_2d_crossings.json'),
            'projection_table':str(output/'projection_2d_trials.csv'),
            'intersection_pairs_csv':str(output/'intersections_3d_pairs.csv'),
            'main_intrinsic_intersections':len(main_pairs),
            'sampled_closure_directions':len(chosen),
            'closure_pair_counts':[
                {'tilt':r['tilt_deg'],'azimuth':r['azimuth_deg'],'sign':r['cap_sign'],
                 'pairs':r.get('total_intersection_pairs'), 'by_layer':r.get('by_layer_pair',{}),
                 'status':r['status']} for r in audited['closure_trials']],
            'note':'No observed or Gaussian data modified; please inspect full report before proposing repair.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preseal-report', type=Path, required=True,
                   help='The exact preseal_diagnostics.json from the failed seal signature directory')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--max-directions', type=int, default=8)
    p.add_argument('--max-pair-samples', type=int, default=50)
    p.add_argument('--no-debug-ply', action='store_true')
    args = p.parse_args()
    if not 1 <= args.max_directions <= 24:
        p.error('--max-directions must be 1..24')
    print(json.dumps(analyze(args.preseal_report,args.output_dir,args.max_directions,
                             args.max_pair_samples,not args.no_debug_ply),
                     ensure_ascii=False,indent=2,default=_as_json))


if __name__=='__main__':
    main()
