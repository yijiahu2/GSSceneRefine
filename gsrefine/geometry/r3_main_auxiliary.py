"""Stage-2 R3 Poisson face-component partition. Geometry read-only.

The full Stage-1 Poisson and authenticated direct-R3 clipping mesh are never
changed. The main sheet is the largest edge-connected face component; every
other face belongs to an auxiliary observation sheet, not the sealing solid.
No welding, vertex movement, remeshing, Poisson reconstruction or repair.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np

PARTITION_REVISION = 'r3_main_auxiliary_split_v1_20261109'


def _subset(mesh, ids):
    import trimesh
    ids = np.asarray(ids, dtype=np.int64)
    faces = np.asarray(mesh.faces, dtype=np.int64)[ids]
    unique, inverse = np.unique(faces.ravel(), return_inverse=True)
    verts = np.asarray(mesh.vertices)[unique]
    result = trimesh.Trimesh(vertices=verts.copy(), faces=inverse.reshape(-1,3), process=False)
    if not np.array_equal(np.asarray(result.vertices), verts):
        raise RuntimeError('Source vertex coordinates changed during component extraction')
    return result


def _boundary_edges_xyz(mesh):
    """Exact coordinate-defined open-edge set, independent of vertex indices."""
    f = np.asarray(mesh.faces, dtype=np.int64)
    ed = np.sort(np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])), axis=1)
    unique, counts = np.unique(ed, axis=0, return_counts=True)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    ret = set()
    for a,b in unique[counts==1]:
        q1=tuple(v[int(a)].tolist()); q2=tuple(v[int(b)].tolist())
        ret.add((q1,q2) if q1<q2 else (q2,q1))
    return ret


def vertex_fan_issues(mesh):
    """Find non-manifold vertex fans even when no edge has >2 incident faces.

    Each vertex's incident triangles must connect via edges that touch the
    vertex; geometric coincidence alone is not a manifold connection.
    """
    faces=np.asarray(mesh.faces,dtype=np.int64)
    incident=defaultdict(list)
    edge_owner=defaultdict(list)
    for i, tri in enumerate(faces):
        for v in tri:
            incident[int(v)].append(i)
        for j in range(3):
            a,b=int(tri[j]),int(tri[(j+1)%3])
            edge_owner[(min(a,b),max(a,b))].append(i)
    # By construction connected face adjacency through each vertex forms one
    # fan. Two internally disconnected fans touching only a vertex are invalid.
    issues=[]
    for v, owners in incident.items():
        if len(owners)<2:
            continue
        parent={i:i for i in owners}
        def find(x):
            while parent[x]!=x:
                parent[x]=parent[parent[x]];x=parent[x]
            return x
        def join(a,b):
            a,b=find(a),find(b)
            if a!=b:parent[a]=b
        for w in set(int(x) for face_id in owners for x in faces[face_id] if x!=v):
            relevant=edge_owner.get((min(v,w),max(v,w)), [])
            for f in relevant[1:]:
                join(relevant[0],f)
        if len({find(i) for i in owners})>1:
            issues.append(int(v))
    return issues


def split_direct_r3_mesh(mesh):
    import trimesh
    if not isinstance(mesh,trimesh.Trimesh) or len(mesh.faces)==0:
        raise ValueError('Need nonempty original direct R3 triangle mesh')
    parts = trimesh.graph.connected_components(
        mesh.face_adjacency,nodes=np.arange(len(mesh.faces),dtype=np.int64),min_len=1)
    parts = sorted((np.sort(np.asarray(p,dtype=np.int64)) for p in parts),
                   key=lambda x:(-len(x), int(x[0])))
    n=len(mesh.faces)
    if sum(map(len,parts))!=n or len(np.unique(np.concatenate(parts)))!=n:
        raise RuntimeError('Face component partition is not a full disjoint cover')
    if len(parts)>1 and len(parts[0])==len(parts[1]):
        raise ValueError('Largest R3 body component is ambiguous (face-count tie)')
    main_ids=parts[0]
    aux_parts=parts[1:]
    aux_ids=np.concatenate(aux_parts) if aux_parts else np.array([],dtype=np.int64)
    main=_subset(mesh,main_ids)
    aux=_subset(mesh,aux_ids) if len(aux_ids) else None
    original_boundary=_boundary_edges_xyz(mesh)
    main_boundary=_boundary_edges_xyz(main)
    ring_preserved=(main_boundary==original_boundary)
    # Every auxiliary face is saved separately and can be mapped back to the
    # original source. In-memory aux face order equals aux_ids order.
    labels=np.concatenate([np.full(len(p),i,dtype=np.int32)
                          for i,p in enumerate(aux_parts)]) if aux_parts else np.empty(0,dtype=np.int32)
    aux_summary=[]
    xyz=np.asarray(mesh.vertices)
    for i,faceids in enumerate(aux_parts):
        verts=xyz[np.unique(np.asarray(mesh.faces)[faceids].ravel())]
        aux_summary.append({'component_id':i,'face_count':int(len(faceids)),
                            'first_original_face_id':int(faceids[0]),
                            'centroid_xyz':np.mean(verts,axis=0).tolist(),
                            'bounds_xyz':[np.min(verts,axis=0).tolist(),np.max(verts,axis=0).tolist()]})
    vertex_issues=vertex_fan_issues(main)
    return {
        'main':main,'auxiliary':aux,'main_original_face_ids':main_ids,
        'auxiliary_original_face_ids':aux_ids,'auxiliary_face_component_ids':labels,
        'original_open_edge_count':len(original_boundary),
        'main_open_edge_count':len(main_boundary),
        'original_r3_boundary_preserved':bool(ring_preserved),
        'main_vertex_fan_issue_count':len(vertex_issues),
        'main_vertex_fan_issue_ids':vertex_issues[:32],
        'component_count':len(parts), 'main_face_count':len(main_ids),
        'auxiliary_face_count':len(aux_ids), 'auxiliary_component_count':len(aux_parts),
        'auxiliary_components':aux_summary,
    }


def auxiliary_evidence(partition, core_xyz, gaussian_xyz, gaussian_radii, *,
                       sfm_support_factor=2.5, gaussian_support_factor=1.5,
                       min_unique_points=3, min_face_coverage=.5,
                       separation_factor=.5, main_distance_fn=None):
    """Independent, spatially distributed SfM + Gaussian evidence per component.

    Crucially, a single nearest SfM/Gaussian distance is NOT a semantic
    certificate.  An ambiguous or low-evidence component remains an unknown
    observation feature; it is NEVER deleted from the auxiliary PLY.
    """
    from scipy.spatial import cKDTree
    aux=partition['auxiliary']
    if aux is None:
        return []
    core=np.asarray(core_xyz,dtype=np.float64)
    gs=np.asarray(gaussian_xyz,dtype=np.float64)
    radii=np.asarray(gaussian_radii,dtype=np.float64)
    if len(core)<7 or len(gs)<3 or len(gs)!=len(radii):
        raise ValueError('Insufficient SfM/Gaussian data for auxiliary evidence')
    stree=cKDTree(core)
    gtree=cKDTree(gs)
    faces=np.asarray(aux.faces)
    v=np.asarray(aux.vertices)
    labels=partition['auxiliary_face_component_ids']
    # One exact, batched triangle-distance query to the MAIN observed surface
    # for every auxiliary sample: nearby SfM/GS alone can reflect overlapping
    # duplicate sheets rather than independent observed rock.
    if main_distance_fn is None:
        from gsrefine.roi.core import exact_sfm_to_poisson_distance_open3d
        main_distance_fn=exact_sfm_to_poisson_distance_open3d
    tasks=[]
    for item in partition['auxiliary_components']:
        fi=np.flatnonzero(labels==item['component_id'])
        tri=v[faces[fi]]
        tests=np.concatenate([tri.mean(axis=1),np.unique(tri.reshape(-1,3),axis=0)],axis=0)
        tasks.append((item,fi,tests))
    all_tests=np.concatenate([t[2] for t in tasks],axis=0)
    main_d_all=np.asarray(main_distance_fn(partition['main'],all_tests,batch=50000),dtype=np.float64)
    if main_d_all.shape!=(len(all_tests),) or not np.isfinite(main_d_all).all():
        raise ValueError('Invalid auxiliary-to-main exact geometric distance')
    results=[]
    offset=0
    for item,fi,tests in tasks:
        main_dist=main_d_all[offset:offset+len(tests)]
        offset+=len(tests)
        # Independent samples across the WHOLE component rather than a single
        # closest point; 2-face components still have vertex coverage.
        ds,si=stree.query(tests,k=1,workers=-1)
        hd=stree.query(core[si],k=7,workers=-1)[0][:,-1]
        h=np.maximum(hd,1.e-10)
        sfm_ok=ds<=sfm_support_factor*h
        dg,gi=gtree.query(tests,k=1,workers=-1)
        gaussian_ok=dg<=np.maximum(radii[gi],gaussian_support_factor*h)
        # Distinct source points across the tested footprint.
        n_sfm=len(np.unique(si[sfm_ok]))
        n_gauss=len(np.unique(gi[gaussian_ok]))
        coverage_sfm=float(np.mean(sfm_ok[:len(fi)]))
        coverage_gauss=float(np.mean(gaussian_ok[:len(fi)]))
        distributed=bool(n_sfm>=min_unique_points and n_gauss>=min_unique_points
                         and coverage_sfm>=min_face_coverage and coverage_gauss>=min_face_coverage)
        # A component that lies on top of the main surface may be redundant;
        # even dense SfM and Gaussian proximity cannot disambiguate it.
        separated=float(np.mean(main_dist[:len(fi)]>separation_factor*h[:len(fi)]))
        independent=bool(separated>=min_face_coverage)
        evidence=('distributed_both_sources_and_separated' if distributed and independent
                  else 'overlap_with_main_ambiguous' if distributed else 'insufficient_or_ambiguous')
        results.append({**item,
                        'tested_points':int(len(tests)),
                        'unique_supported_sfm':int(n_sfm),
                        'unique_supported_gaussian':int(n_gauss),
                        'sfm_face_centroid_coverage':coverage_sfm,
                        'gaussian_face_centroid_coverage':coverage_gauss,
                        'local_sfm_spacing_median':float(np.median(h)),
                        'nearest_sfm_distance_min':float(ds.min()),
                        'nearest_gaussian_distance_min':float(dg.min()),
                        'main_surface_distance_p50':float(np.median(main_dist)),
                        'fraction_face_centroids_separated_from_main':separated,
                        'observation_evidence':evidence,
                        'confirmed_real_object':False,
                        'note':'Distinct SfM/GS support and separation from the main surface reduce overlap ambiguity but cannot certify true object identity.'})
    return results
