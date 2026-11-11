"""Read-only provenance audit of Direct R3-clipped mesh topology.

Compare original module-1 global Poisson with the *UNREPAIRED* R3 clip;
do not rerun R3, Poisson, topology repair, sealing or Gaussian pruning.
Maps each clipped triangle to source triangles by exact closest-triangle
geometry, with multi-probe consistency and explicit uncertainty.

Usage:
  python tools/trace_r3_poisson_origin.py --manifest ROI_RUN/metadata/manifest.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REVISION = "r3_global_poisson_topology_lineage_v1_20261109"


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_anomaly_csvs(folder, edges, islands):
    """Compact review tables; no original mesh file is touched."""
    folder=Path(folder)
    edgefile=folder/'nonmanifold_edge_origins.csv'
    compfile=folder/'minor_component_origins.csv'
    if not edgefile.exists():
        with edgefile.open('w',newline='',encoding='utf-8-sig') as f:
            out=csv.writer(f)
            out.writerow(['clip_edge_index','origin','clip_incident_faces','global_edge_incident_faces',
                          'x1','y1','z1','x2','y2','z2','parent_global_faces'])
            for i,item in enumerate(edges['items'],1):
                a,b=item['clipped_edge_xyz']
                out.writerow([i,item['conclusion'],item['clipped_incident_faces'],
                    item['original_global_edge_incidence'],*a,*b,
                    ';'.join(str(x) for x in item['parent_global_face_ids'])])
    if not compfile.exists():
        with compfile.open('w',newline='',encoding='utf-8-sig') as f:
            out=csv.writer(f)
            out.writerow(['component_rank','faces','origin','global_component_ids',
                          'centroid_x','centroid_y','centroid_z',
                          'cut_generated_vertices','unresolved_face_parents'])
            for item in islands['items']:
                out.writerow([item['component_rank'],item['faces'],item['decision'],
                    ';'.join(str(x) for x in item['global_component_labels']),
                    *item['centroid_xyz'],item['clip_generated_vertices'],
                    item['parent_face_mapping_unresolved']])
    return edgefile,compfile


def edge_incidence(mesh):
    faces = np.asarray(mesh.faces, dtype=np.int64)
    oriented = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    owners = np.tile(np.arange(len(faces), dtype=np.int64), 3)
    edges, inverse, counts = np.unique(np.sort(oriented, axis=1), axis=0,
                                        return_inverse=True, return_counts=True)
    by_edge = defaultdict(list)
    for eid, face_id in zip(inverse, owners):
        if counts[eid] > 2:
            by_edge[int(eid)].append(int(face_id))
    return edges, counts, by_edge


def face_components(mesh):
    """Same trimesh face-adjacency definition as existing preseal checks."""
    import trimesh
    comps = trimesh.graph.connected_components(mesh.face_adjacency,
              nodes=np.arange(len(mesh.faces)), min_len=1)
    comps = sorted(comps, key=len, reverse=True)
    labels = np.full(len(mesh.faces), -1, dtype=np.int64)
    for i, component in enumerate(comps):
        labels[np.asarray(component, dtype=np.int64)] = i
    if (labels < 0).any():
        raise AssertionError('Some source Poisson faces have no connectivity label')
    return comps, labels


def _contained_in_parent_triangles(tris, parent_tris, *, tolerance):
    """Check each output subtriangle lies in its proposed original source face.

    Coplanarity alone is insufficient near overlapping or adjacent source faces.
    This test checks all three candidate vertices using stable barycentric
    coordinates, with a model-unit tolerance accounting for binary PLY floats.
    """
    p0=parent_tris[:,0]
    e0=parent_tris[:,1]-p0
    e1=parent_tris[:,2]-p0
    v=tris-p0[:,None,:]
    a=np.einsum('ij,ij->i',e0,e0)
    b=np.einsum('ij,ij->i',e0,e1)
    c=np.einsum('ij,ij->i',e1,e1)
    den=a*c-b*b
    good=den>1e-22
    den_safe=np.where(good,den,1.)
    d=np.einsum('bij,bj->bi',v,e0)
    e=np.einsum('bij,bj->bi',v,e1)
    u=(d*c[:,None]-e*b[:,None])/den_safe[:,None]
    w=(e*a[:,None]-d*b[:,None])/den_safe[:,None]
    s=1-u-w
    longest=np.maximum(np.sqrt(np.maximum(a,c)),1e-12)
    slack=np.minimum(.05,tolerance/longest*2)
    within=(np.minimum(np.minimum(u,w),s)>=-slack[:,None]).all(axis=1)
    nearest=p0[:,None,:]+u[:,:,None]*e0[:,None,:]+w[:,:,None]*e1[:,None,:]
    planar=(np.linalg.norm(nearest-tris,axis=2)<=tolerance).all(axis=1)
    return good & within & planar


def map_faces_to_source(source, clipped, *, tolerance, batch=20000):
    """Nearest observed triangle for 3 strict interior probes per clip triangle.

    All probe locations should lie ON the original global Poisson.  One parent
    must win consistently; otherwise mark ambiguous, rather than assert lineage.
    """
    import open3d as o3d
    orig = o3d.t.geometry.TriangleMesh()
    orig.vertex['positions'] = o3d.core.Tensor(np.asarray(source.vertices, dtype=np.float32))
    orig.triangle['indices'] = o3d.core.Tensor(np.asarray(source.faces, dtype=np.int32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(orig)
    nf = len(clipped.faces)
    parent = np.full(nf, -1, dtype=np.int64)
    max_errors = np.zeros(nf, dtype=np.float64)
    # Non-centroid-only deterministic interior samples reduce ties on shared edges.
    weights = np.asarray([[1/3,1/3,1/3], [.17,.31,.52], [.59,.18,.23]], dtype=np.float64)
    for a in range(0, nf, batch):
        b = min(a + batch, nf)
        triangles = np.asarray(clipped.vertices, dtype=np.float64)[np.asarray(clipped.faces[a:b])]
        samples = np.einsum('ji,bik->bjk', weights, triangles).reshape(-1, 3)
        q = o3d.core.Tensor(np.ascontiguousarray(samples, np.float32))
        hits = scene.compute_closest_points(q)
        ids = np.asarray(hits['primitive_ids'].numpy(), dtype=np.int64).reshape(-1, 3)
        nearest = np.asarray(hits['points'].numpy(), dtype=np.float64).reshape(-1, 3, 3)
        errors = np.linalg.norm(nearest - samples.reshape(-1, 3, 3), axis=2)
        max_errors[a:b] = errors.max(axis=1)
        reliable = (ids[:, 0] == ids[:, 1]) & (ids[:, 0] == ids[:, 2]) & \
                   (max_errors[a:b] <= tolerance) & (ids[:, 0] >= 0) & \
                   (ids[:, 0] < len(source.faces))
        idx = np.flatnonzero(reliable)
        if len(idx):
            proposed = ids[idx,0]
            source_triangles = np.asarray(source.vertices, dtype=np.float64)[
                np.asarray(source.faces, dtype=np.int64)[proposed]]
            geometrically_inside = _contained_in_parent_triangles(
                triangles[idx], source_triangles, tolerance=tolerance)
            parent[a:b][idx[geometrically_inside]] = proposed[geometrically_inside]
    return parent, max_errors


def _point_on_segment(pt, a, b, tolerance):
    v = b - a
    den = float(v @ v)
    if den == 0:
        return False
    t = float((pt-a)@v) / den
    nearest = a + np.clip(t, 0, 1)*v
    return -1e-5 <= t <= 1.00001 and np.linalg.norm(pt-nearest) <= tolerance


def trace_nonmanifold(source, clipped, parents, src_edges, src_counts, *, tolerance):
    """Attribute nonmanifold clip edges only when original-edge lineage is proven."""
    faces = np.asarray(clipped.faces, np.int64)
    srcfaces = np.asarray(source.faces, np.int64)
    xyz = np.asarray(source.vertices, np.float64)
    cutxyz = np.asarray(clipped.vertices, np.float64)
    badedges, cutcounts, incident_map = edge_incidence(clipped)
    edge_count = {tuple(map(int,e)): int(n) for e,n in zip(src_edges,src_counts)}
    items=[]
    tally=Counter()
    for eid in np.flatnonzero(cutcounts>2):
        vids = badedges[eid]
        owners = incident_map[int(eid)]
        pids = parents[np.asarray(owners,np.int64)]
        decision='uncertain_parent_mapping'
        source_edge=None
        source_incidence=None
        if np.all(pids>=0):
            shared = set(map(int,srcfaces[pids[0]]))
            for pid in pids[1:]:
                shared.intersection_update(int(i) for i in srcfaces[pid])
            valid_edges=[]
            for x,y in itertools.combinations(sorted(shared),2):
                a,b=xyz[[x,y]]
                if (_point_on_segment(cutxyz[vids[0]],a,b,tolerance)
                    and _point_on_segment(cutxyz[vids[1]],a,b,tolerance)):
                    num=edge_count.get((x,y),0)
                    valid_edges.append((num,(x,y)))
            if valid_edges:
                num,source_edge=max(valid_edges)
                source_incidence=num
                decision = ('inherited_from_global_nonmanifold_edge' if num>2 else
                            'clip_or_overlapping_sheet_topology')
            else:
                decision='clip_generated_or_geometric_overlap'
        tally[decision]+=1
        items.append({
            'clipped_edge_vertex_ids':list(map(int,vids)),
            'clipped_edge_xyz':cutxyz[vids].tolist(),
            'clipped_incident_faces':len(owners),
            'parent_global_face_ids':pids.tolist(),
            'original_global_edge_vertex_ids':list(source_edge) if source_edge else None,
            'original_global_edge_incidence':source_incidence,
            'conclusion':decision,
        })
    return {'counts':dict(sorted(tally.items())), 'items':items}


def trace_components(source, clipped, parents, global_face_labels, *, tolerance):
    comps, _=face_components(clipped)
    major_global=set()
    # Parents indices may be -1. Avoid unintended last-face indexing.
    if len(comps):
        majorparents=parents[np.asarray(comps[0],np.int64)]
        major_global=set(int(x) for x in global_face_labels[majorparents[majorparents>=0]])
    src_xyz=np.asarray(source.vertices,np.float64)
    cutxyz=np.asarray(clipped.vertices,np.float64)
    # Existing vertices are unchanged; newly generated boundary verts lie along
    # old edges and are normally not exact source vertices.
    from scipy.spatial import cKDTree
    nearest_source=cKDTree(src_xyz)
    cut_nearest=nearest_source.query(cutxyz,k=1,workers=-1)[0]
    items=[];tally=Counter()
    for j, face_ids in enumerate(comps[1:],1):
        ids=np.asarray(face_ids,dtype=np.int64)
        pids=parents[ids]
        valid=pids[pids>=0]
        old_labels=set(int(v) for v in global_face_labels[valid])
        if len(valid)!=len(pids):
            decision='uncertain_parent_mapping'
        elif not old_labels:
            decision='uncertain_parent_mapping'
        elif len(old_labels)==1 and old_labels.isdisjoint(major_global):
            decision='preexisting_separate_global_component'
        elif old_labels.issubset(major_global):
            decision='separated_by_R3_crop_or_clip_topology'
        else:
            decision='mixed_source_components_needs_review'
        tally[decision]+=1
        f=np.asarray(clipped.faces,dtype=np.int64)[ids]
        v=np.unique(f.reshape(-1))
        c=cutxyz[v]
        items.append({
            'component_rank':j+1,
            'faces':len(ids),
            'centroid_xyz':c.mean(axis=0).tolist(),
            'bounds_xyz':[c.min(axis=0).tolist(),c.max(axis=0).tolist()],
            'global_component_labels':sorted(old_labels),
            'parent_face_mapping_unresolved':int(len(pids)-len(valid)),
            'clip_generated_vertices':int(np.sum(cut_nearest[v]>tolerance)),
            'decision':decision,
        })
    return {'main_global_component_labels':sorted(major_global),
            'counts':dict(sorted(tally.items())), 'items':items}


def run(manifest, *, clip_report=None, tolerance=None):
    import trimesh
    from gsrefine.cleanup.sdf_common import _load_stage1_manifest
    from gsrefine.cleanup.direct_r3_pipeline import select_direct_r3_clip, validate_sources
    manifest_path, upstream, inputs, root=_load_stage1_manifest(manifest)
    raw_path, report_path, clip=select_direct_r3_clip(root, selected_report=clip_report)
    roi_sfm=Path(root)/'cleanup/inputs/sfm_roi_local_v3.npz'
    if not roi_sfm.is_file():
        sfm_path=(upstream.get('stage2_inputs') or {}).get('sfm_roi_npz')
        roi_sfm=Path(sfm_path).resolve(strict=True) if sfm_path else roi_sfm
    validate_sources(manifest_path,upstream,inputs,root,clip,roi_sfm)
    source=trimesh.load(inputs['poisson_mesh'],process=False,force='mesh')
    clipped=trimesh.load(raw_path,process=False,force='mesh')
    if not isinstance(source,trimesh.Trimesh) or not isinstance(clipped,trimesh.Trimesh):
        raise ValueError('Expected triangle meshes')
    diag=float(np.linalg.norm(np.ptp(source.vertices,axis=0)))
    tol=float(tolerance) if tolerance is not None else max(1e-5,diag*3e-6)
    if not np.isfinite(tol) or tol<=0 or tol>diag*1e-3:
        raise ValueError('Invalid source triangle coordinate tolerance')
    print('[r3-provenance] Source faces:',len(source.faces),'R3 faces:',len(clipped.faces),flush=True)
    print('[r3-provenance] Mapping R3 faces to global Poisson source geometry...',flush=True)
    parents, errors=map_faces_to_source(source,clipped,tolerance=tol)
    valid=int(np.count_nonzero(parents>=0))
    print(f'[r3-provenance] Unambiguous face parents {valid:,}/{len(parents):,}',flush=True)
    global_components,global_labels=face_components(source)
    src_edges,src_counts,_=edge_incidence(source)
    cut_edges,cut_counts,_=edge_incidence(clipped)
    edges=trace_nonmanifold(source,clipped,parents,src_edges,src_counts,tolerance=tol)
    islands=trace_components(source,clipped,parents,global_labels,tolerance=tol)
    payload={
        'schema':'GSSceneRefine.r3_poisson_topology_provenance.v1',
        'revision':REVISION,
        'status':'READ_ONLY_TOPOLOGY_PROVENANCE_NOT_REPAIRED',
        'inputs':{
            'stage1_manifest':str(manifest_path),
            'global_poisson':str(inputs['poisson_mesh']),
            'global_poisson_sha256':sha256(inputs['poisson_mesh']),
            'original_direct_r3_clip':str(raw_path),
            'original_direct_r3_clip_sha256':sha256(raw_path),
            'clip_report':str(report_path),
        },
        'source_global_poisson':{
            'faces':int(len(source.faces)),
            'nonmanifold_edges':int(np.sum(src_counts>2)),
            'face_components':int(len(global_components)),
            'top_component_face_counts':[int(len(x)) for x in global_components[:12]]
        },
        'unrepaired_r3_clip':{
            'faces':int(len(clipped.faces)),
            'nonmanifold_edges':int(np.sum(cut_counts>2)),
            'face_components':int(1+len(islands['items'])),
        },
        'parent_face_geometry_verification':{
            'method':'open3d_nearest_global_triangle_three_interior_probes',
            'closest_triangle_distance_tolerance':tol,
            'definite_parent_face_count':valid,
            'ambiguous_or_unmatched_parent_count':int(len(parents)-valid),
            'definite_parent_fraction':float(valid/len(parents)),
            'max_probe_error':float(np.max(errors)),
            'p95_probe_error':float(np.quantile(errors,.95)),
            'warning':'Overlapping/coplanar source faces can make closest-triangle parent ambiguous. Any affected anomaly is explicitly unresolved.'
        },
        'nonmanifold_edge_origin':edges,
        'minor_component_origin':islands,
        'interpretation':{
            'inherited_edge':'Original global mesh has 3+ source faces on the same edge; do not attribute it to clipping.',
            'crop_separated_component':'Source face parents were globally connected to the retained main body but become disconnected after R3 clipping; this does not by itself mean the clip code is wrong.',
            'source_separate_component':'Source face parents belonged to a different global mesh component even before R3 clipping.',
            'limits':'Ambiguous geometric parent matches and spatially coincident sheets require case-by-case examination. This audit does not modify meshes or establish truth/quality.'
        },
    }
    identity=hashlib.sha256((REVISION+payload['inputs']['global_poisson_sha256']+payload['inputs']['original_direct_r3_clip_sha256']).encode()).hexdigest()
    dest=Path(root)/'cleanup/geometry/direct_r3_provenance'/identity[:24]
    output=dest/'topology_provenance_report.json'
    if output.exists():
        prev=json.loads(output.read_text(encoding='utf8'))
        if (prev.get('inputs')!=payload['inputs'] or prev.get('revision')!=REVISION):
            raise FileExistsError('Report exists but inputs/version changed: '+str(output))
    else:
        write_json(output,payload)
    edgecsv,compcsv=write_anomaly_csvs(dest,edges,islands)
    print(json.dumps({'status':payload['status'],'report':str(output),
        'edge_details_csv':str(edgecsv),'component_details_csv':str(compcsv),
        'source_nonmanifold':payload['source_global_poisson']['nonmanifold_edges'],
        'clipped_nonmanifold':payload['unrepaired_r3_clip']['nonmanifold_edges'],
        'edge_origins':edges['counts'],'minor_component_origins':islands['counts'],
        'unambiguous_source_face_fraction':payload['parent_face_geometry_verification']['definite_parent_fraction']},
        ensure_ascii=False,indent=2))
    return payload


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--clip-report',help='Required if more than one Direct R3 clip candidate exists')
    p.add_argument('--coordinate-tolerance',type=float,default=None,
                   help='Optional PLY geometry tolerance in model units; normally automatic')
    args=p.parse_args(argv)
    run(args.manifest,clip_report=args.clip_report,tolerance=args.coordinate_tolerance)

if __name__=='__main__':
    main()
