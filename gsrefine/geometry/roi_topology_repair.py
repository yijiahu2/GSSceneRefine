"""Conservative, provenance-preserving topology hygiene for direct R3 ROI Poisson.

The single Stage-1 Poisson reconstruction is never refit, smoothed, or altered.
Only validated degenerate/duplicate faces and VERY SMALL disconnected, twice-
unsupported surface patches may be removed after a reversible topology trial. All original retained vertices
keep their exact coordinates. Genuine ambiguity blocks skirt sealing.

This is NOT a general hole-filler or mesh-remesher. It deliberately does not
repair important/multi-supported islands or arbitrarily select one of the
faces meeting at a non-manifold edge.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import hashlib

import numpy as np


REPAIR_REVISION = 'r3_component_transactional_topology_v2_20261109'


def _edge_occurrences(faces):
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    faceids = np.tile(np.arange(len(faces), dtype=np.int64), 3)
    normalized = np.sort(edges, axis=1)
    _, inv, count = np.unique(normalized, axis=0, return_inverse=True, return_counts=True)
    return inv, count, faceids


def _surface_patches(mesh, *, regular_edges_only=False):
    """Connected face groups, optionally ignoring >2-face nonmanifold edges."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    faces = np.asarray(mesh.faces, np.int64)
    inv, counts, face_ids = _edge_occurrences(faces)
    order = np.argsort(inv, kind='stable')
    keys = inv[order]
    owners = face_ids[order]
    pairstarts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    pairlens = np.diff(np.r_[pairstarts, len(keys)])
    eligible = pairlens == 2 if regular_edges_only else pairlens >= 2
    src = []
    dst = []
    for pos, length in zip(pairstarts[eligible], pairlens[eligible]):
        members = owners[pos:pos + length]
        if regular_edges_only:
            src.append(int(members[0])); dst.append(int(members[1]))
        else:
            for m in members[1:]:
                src.append(int(members[0])); dst.append(int(m))
    n = len(faces)
    if not src:
        return [np.array([i], dtype=np.int64) for i in range(n)]
    graph = coo_matrix((np.ones(len(src) * 2, np.uint8),
                        (np.r_[src, dst], np.r_[dst, src])), shape=(n,n)).tocsr()
    _, lab = connected_components(graph, directed=False)
    sizes = np.bincount(lab)
    orderlab = np.argsort(sizes)[::-1]
    return [np.flatnonzero(lab == label) for label in orderlab]


def _basic_hygiene(mesh):
    """Drop only invalid/identical faces; do not weld vertices or move them."""
    out = mesh.copy()
    oldfaces = len(out.faces)
    original_vertex_count = len(out.vertices)
    # Exact-coordinate vertex reindexing only (bit-identical xyz).  The
    # generated PLY can repeat an intersection vertex with separate indices.
    # Merging identical coordinates is topology hygiene, NOT smoothing/welding
    # close-but-different positions; no retained point is moved at all.
    xyz = np.asarray(out.vertices)
    _, first, inv = np.unique(xyz, axis=0, return_index=True,
                              return_inverse=True)
    if len(first) < len(xyz):
        order = np.argsort(first)
        rank = np.empty(len(order), dtype=np.int64)
        rank[order] = np.arange(len(order), dtype=np.int64)
        out = __import__('trimesh').Trimesh(
            vertices=xyz[first[order]].copy(),
            faces=rank[inv[np.asarray(out.faces)]], process=False)
    valid = out.nondegenerate_faces(height=0.0)
    deg = int((~valid).sum())
    if deg:
        out.update_faces(valid)
    valid = out.unique_faces()
    duplicates = int((~valid).sum())
    if duplicates:
        out.update_faces(valid)
    out.remove_unreferenced_vertices()
    if not len(out.faces):
        raise ValueError('All ROI faces invalid after strict topology hygiene')
    return out, {'input_faces': oldfaces, 'degenerate_removed': deg,
                 'duplicate_faces_removed': duplicates,
                 'exact_identical_coordinate_vertices_merged': int(original_vertex_count-len(first)),
                 'vertices_not_relocated': True, 'nearby_vertex_welding_performed': False}


def boundary_coordinate_signature(mesh):
    """Coordinate-defined open boundary signature (independent of vertex indices).

    Enables a strict check that housekeeping did not shift, trim, or substitute
    the confirmed Direct R3 perimeter.  A change blocks auto sealing.
    """
    import hashlib
    faces = np.asarray(mesh.faces, dtype=np.int64)
    e = np.concatenate((faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]))
    ordered, counts = np.unique(np.sort(e,axis=1), axis=0, return_counts=True)
    outer = ordered[counts == 1]
    xyz = np.asarray(mesh.vertices, dtype=np.float64)
    values=[]
    for a,b in outer:
        one=tuple(float(x) for x in xyz[a]);two=tuple(float(x) for x in xyz[b])
        values.append((*one,*two) if one < two else (*two,*one))
    values.sort()
    data = np.asarray(values,dtype=np.float64).tobytes() if values else b''
    return hashlib.sha256(data).hexdigest()


def _dominant_boundary_faces(mesh):
    """Protect the simple closed ring with the MOST edges (R3 cut perimeter)."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = np.concatenate((faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]))
    owners = np.tile(np.arange(len(faces), dtype=np.int64), 3)
    unique, inv, counts = np.unique(np.sort(edges,axis=1), axis=0,
                                    return_inverse=True, return_counts=True)
    mask = counts[inv] == 1
    if not mask.any():
        return set()
    be = edges[mask]
    bf = owners[mask]
    graph = defaultdict(list)
    for eid,(a,b) in enumerate(be):
        graph[int(a)].append(eid)
        graph[int(b)].append(eid)
    seen_vertices = set()
    best_edge_count, best_perimeter, best_faces = -1, -1., set()
    xyz = np.asarray(mesh.vertices)
    for start in sorted(graph):
        if start in seen_vertices:
            continue
        q = [start]; seen_vertices.add(start)
        eids = set()
        while q:
            u = q.pop()
            for eid in graph[u]:
                eids.add(eid)
                for v in be[eid]:
                    v = int(v)
                    if v not in seen_vertices:
                        seen_vertices.add(v); q.append(v)
        # A dangling fin may have a very long perimeter but few edges.
        # Prefer ring edge count and require every vertex to have degree 2.
        ids = np.fromiter(eids, dtype=np.int64)
        this_vertices = set(int(x) for x in be[ids].ravel())
        if not all(len(graph[u]) == 2 for u in this_vertices):
            continue
        perimeter = float(np.linalg.norm(xyz[be[ids,0]]-xyz[be[ids,1]],axis=1).sum())
        if (len(ids), perimeter) > (best_edge_count, best_perimeter):
            best_edge_count, best_perimeter = len(ids), perimeter
            best_faces = set(bf[ids].tolist())
    # If no certifiably closed ring exists, conservatively protect every
    # open-edge face; the later topology gate will block incorrect sealing.
    return best_faces if best_edge_count >= 0 else set(bf.tolist())


def _candidate_has_evidence(mesh, face_ids, sfm_tree, core_xyz,
                             gaussian_tree, gaussian_radii, *, support_factor=3.5,
                             maximum_test_points=2048):
    """A small patch is PROTECTED if either SfM or Gaussian supports any test point.

    Use local core-SfM kNN spacing, never the sampled spacing or a global
    density-derived deletion rule. Gaussian support uses its own major 3-sigma
    axis OR the local SfM spacing. Conservative OR protects sparse real detail.
    """
    faces = np.asarray(mesh.faces)[face_ids]
    verts = np.asarray(mesh.vertices, np.float64)
    distinct = np.unique(faces.ravel())
    # Shared attachment vertices are NOT independent evidence for an isolated
    # nonmanifold fin: they are automatically supported by the legitimate host
    # sheet.  Use only private vertices plus triangle interiors.  Never split
    # or move an attachment edge in this routine.
    allcounts = np.bincount(np.asarray(mesh.faces).ravel(), minlength=len(verts))
    localcounts = np.bincount(faces.ravel(), minlength=len(verts))
    private = distinct[allcounts[distinct] == localcounts[distinct]]
    queries = np.vstack((verts[private], verts[faces].mean(axis=1)))
    if len(queries) > maximum_test_points:
        # Too large to certify as an innocuous island: preserve automatically.
        return {'sfm_support': True, 'gaussian_support': True,
                'blocked_reason': 'test_point_budget_exceeded',
                'tested': 0}
    sfm_dist, near = sfm_tree.query(queries, k=1, workers=-1)
    # True local native SfM spacing, using all CORE points as neighbors.
    h = sfm_tree.query(core_xyz[np.asarray(near, np.int64)],
                       k=min(7,len(core_xyz)), workers=-1)[0]
    if h.ndim == 2:
        h = h[:, -1]
    h = np.maximum(np.asarray(h, dtype=np.float64), 1e-10)
    sfm = bool(np.any(sfm_dist <= support_factor * h))
    gauss_dist, idx = gaussian_tree.query(queries, k=1, workers=-1)
    # A Gaussian within its actual 3-sigma ellipsoid's bounding ball is support;
    # local-spacing floor conservatively protects less dense legitimate regions.
    radii = gaussian_radii[np.asarray(idx, np.int64)]
    gauss = bool(np.any(gauss_dist <= np.maximum(radii, support_factor * h)))
    return {'sfm_support': sfm, 'gaussian_support': gauss,
            'nearest_sfm_distance_min': float(np.min(sfm_dist)),
            'nearest_gaussian_distance_min': float(np.min(gauss_dist)),
            'sfm_local_spacing_median': float(np.median(h)),
            'tested':len(queries)}


def _topology_state(mesh):
    """Index-accurate mesh topology. No geometry relocation or implicit welding."""
    faces = np.asarray(mesh.faces, np.int64)
    xyz = np.asarray(mesh.vertices, np.float64)
    edges = np.sort(np.concatenate((faces[:,[0,1]], faces[:,[1,2]],
                                    faces[:,[2,0]]), axis=0), axis=1)
    keys, count = np.unique(edges, axis=0, return_counts=True)
    open_pairs = keys[count == 1]
    nonmanifold_pairs = keys[count > 2]
    open_coordinate_edges = set()
    for a,b in open_pairs:
        q1 = tuple(xyz[a].tolist())
        q2 = tuple(xyz[b].tolist())
        open_coordinate_edges.add((q1,q2) if q1 < q2 else (q2,q1))
    # Protect only an existing, simple closed ring.  An unsupported pre-existing
    # isolated open sheet can be removed, but the authentic crop loop cannot.
    graph = defaultdict(list)
    for eid,(a,b) in enumerate(open_pairs):
        graph[int(a)].append(eid)
        graph[int(b)].append(eid)
    visited, rings = set(), []
    for start in sorted(graph):
        if start in visited:
            continue
        stack = [start]; visited.add(start)
        vset, eset = set(), set()
        while stack:
            u = stack.pop()
            vset.add(u)
            for eid in graph[u]:
                eset.add(eid)
                for v in open_pairs[eid]:
                    v=int(v)
                    if v not in visited:
                        visited.add(v); stack.append(v)
        if all(len(graph[v]) == 2 for v in vset):
            coords = set()
            perimeter = 0.
            for eid in eset:
                a,b = open_pairs[eid]
                q1=tuple(xyz[a].tolist()); q2=tuple(xyz[b].tolist())
                coords.add((q1,q2) if q1 < q2 else (q2,q1))
                perimeter += float(np.linalg.norm(xyz[a]-xyz[b]))
            rings.append((perimeter, coords))
    largest_ring = max(rings, default=(0., set()),
                       key=lambda item:(len(item[1]), item[0]))[1]
    components = _surface_patches(mesh, regular_edges_only=True)
    return {'open_edges':open_coordinate_edges,
            'original_main_ring':largest_ring,
            'simple_ring_count':len(rings),
            'open_edge_count':len(open_pairs),
            'nonmanifold_edges':int(len(nonmanifold_pairs)),
            'nonmanifold_pairs':nonmanifold_pairs,
            'nonmanifold_faces':None,
            'face_components':len(components),
            'winding_consistent':bool(mesh.is_winding_consistent)}


def _trial_validation(before, candidate, protected_ring, *, stage):
    """Never accept an edit that exposes a NEW cut edge or harms the R3 perimeter.

    For simple-ring source meshes, preserve the ENTIRE original ring. Source
    standalone island boundaries may be removed, but never create new ones.
    """
    old = _topology_state(before)
    new = _topology_state(candidate)
    exposed = new['open_edges'] - old['open_edges']
    lost_outer = protected_ring - new['open_edges']
    reasons=[]
    if exposed:
        # Safe exception for hygiene that restores a broken outline caused by
        # *exact duplicate triangles*. Only when no protected loop existed.
        restores_single_ring = (stage == 'hygiene' and not protected_ring
                                and new['simple_ring_count'] == 1
                                and new['nonmanifold_edges'] < old['nonmanifold_edges'])
        if not restores_single_ring:
            reasons.append('new_open_boundary_edges')
    if lost_outer:
        reasons.append('damages_verified_original_r3_ring')
    if new['nonmanifold_edges'] > old['nonmanifold_edges']:
        reasons.append('increases_nonmanifold_edges')
    if new['face_components'] > old['face_components']:
        reasons.append('splits_existing_components')
    if old['winding_consistent'] and not new['winding_consistent']:
        reasons.append('introduces_winding_inconsistency')
    if stage == 'nonmanifold_manifold_sheet' and new['nonmanifold_edges'] >= old['nonmanifold_edges']:
        reasons.append('does_not_fix_nonmanifold_connection')
    if stage == 'disconnected_components' and new['face_components'] >= old['face_components']:
        reasons.append('does_not_remove_independent_component')
    if stage == 'hygiene' and new['nonmanifold_edges'] == old['nonmanifold_edges'] and \
       new['face_components'] == old['face_components'] and new['open_edges'] == old['open_edges']:
        # A pure index/face canonicalization is allowed if nothing worsened.
        pass
    detail = {
        'reasons':reasons,
        'new_open_boundary_edges':int(len(exposed)),
        'lost_original_r3_ring_edges':int(len(lost_outer)),
        'nonmanifold_edges_before':old['nonmanifold_edges'],
        'nonmanifold_edges_after':new['nonmanifold_edges'],
        'face_components_before':old['face_components'],
        'face_components_after':new['face_components'],
        'open_boundary_edges_before':old['open_edge_count'],
        'open_boundary_edges_after':new['open_edge_count'],
        'accepted':not reasons,
    }
    return not reasons, detail


def _localized_edge_groups(mesh, tolerance=1e-5):
    """Report nonmanifold edges grouped by repeated constant-coordinate planes.

    Purely diagnostic: automatically finds the A/B/C-like planar clusters
    without hardcoding Songzi coordinates or modifying geometry.
    """
    state=_topology_state(mesh)
    e=state['nonmanifold_pairs']
    if not len(e):
        return []
    xyz=np.asarray(mesh.vertices, np.float64)
    pts=xyz[e]
    possible=[]
    for axis in range(3):
        same=np.abs(pts[:,0,axis]-pts[:,1,axis]) <= tolerance
        levels=np.round(pts[:,0,axis]/tolerance).astype(np.int64)
        for level in np.unique(levels[same]):
            selected=np.flatnonzero(same & (levels == level))
            if len(selected):
                possible.append((len(selected),axis,int(level),set(selected.tolist())))
    possible.sort(key=lambda item:(-item[0], item[1], item[2]))
    remaining=set(range(len(e)))
    groups=[]
    for _,axis,level,members in possible:
        selected=remaining & members
        if len(selected) < 2:
            continue
        remaining -= selected
        indices=sorted(selected)
        region=pts[indices].reshape(-1,3)
        groups.append({'plane_axis':'xyz'[axis],
                       'plane_coordinate':float(np.median(region[:,axis])),
                       'nonmanifold_edge_count':len(indices),
                       'centroid_xyz':np.mean(region,axis=0).tolist(),
                       'bounds_xyz':[np.min(region,axis=0).tolist(),np.max(region,axis=0).tolist()],
                       'edge_vertex_ids':e[indices].astype(int).tolist()})
    for i in sorted(remaining):
        region=pts[i]
        groups.append({'plane_axis':None,'plane_coordinate':None,
                       'nonmanifold_edge_count':1,
                       'centroid_xyz':np.mean(region,axis=0).tolist(),
                       'bounds_xyz':[np.min(region,axis=0).tolist(),np.max(region,axis=0).tolist()],
                       'edge_vertex_ids':[e[i].astype(int).tolist()]})
    return groups


def _drop_patch(mesh, face_ids):
    candidate=mesh.copy()
    remain=np.ones(len(mesh.faces),dtype=bool)
    remain[np.asarray(face_ids,dtype=np.int64)]=False
    candidate.update_faces(remain)
    candidate.remove_unreferenced_vertices()
    return candidate


def repair_r3_roi_topology(mesh, core_xyz, gaussian_xyz, gaussian_long_axis_3sigma,
                           *, max_island_faces=48, max_island_fraction=0.002,
                           support_factor=3.5, permit_nonmanifold_patches=True):
    """Try strictly reversible whole-patch edits; reject ANY new open boundary.

    The reference direct-R3 surface is never modified. An existing genuine
    disconnected island is an atomic unit, not a bag of faces. A tiny regular
    manifold sheet incident to a nonmanifold edge is another candidate, but may
    be removed only if unsupported AND the entire resulting topology improves.
    Ambiguous supported geometry is preserved and blocks downstream sealing.
    """
    import trimesh
    from scipy.spatial import cKDTree

    if not isinstance(mesh,trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError('Nonempty trimesh surface required')
    xyz=np.asarray(core_xyz,dtype=np.float64)
    g=np.asarray(gaussian_xyz,dtype=np.float64)
    r=np.asarray(gaussian_long_axis_3sigma,dtype=np.float64)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz)<7 or
        g.ndim != 2 or g.shape[1] != 3 or len(g)<1 or
        r.shape != (len(g),) or not np.isfinite(xyz).all() or
        not np.isfinite(g).all() or not np.isfinite(r).all() or np.any(r<=0)):
        raise ValueError('All core SfM and Gaussian xyz/major-axis radii are required')
    if not 0 < support_factor <= 10 or not 0 < max_island_fraction < .1 or max_island_faces<1:
        raise ValueError('Invalid conservative topology options')

    source_state=_topology_state(mesh)
    protected_ring=source_state['original_main_ring']
    initial_groups=_localized_edge_groups(mesh)
    # Structural checks are applied BEFORE accepting any zero-distance weld,
    # degenerate removal, or duplicate face removal.
    hygienic,hygiene=_basic_hygiene(mesh)
    candidate_changed=(hygiene['degenerate_removed']>0 or
                       hygiene['duplicate_faces_removed']>0 or
                       hygiene['exact_identical_coordinate_vertices_merged']>0)
    current=mesh.copy()
    actions=[]
    if candidate_changed:
        accept, check=_trial_validation(current,hygienic,protected_ring,stage='hygiene')
        actions.append({'stage':'hygiene','faces':len(mesh.faces)-len(hygienic.faces),
                        'decision':'accepted' if accept else 'rolled_back',**check})
        hygiene['committed']=bool(accept)
        if accept:
            current=hygienic
    else:
        hygiene['committed']=False
        actions.append({'stage':'hygiene','decision':'not_needed',
                        'faces':0,'accepted':True})

    sfm_tree=cKDTree(xyz)
    gaussian_tree=cKDTree(g)
    max_patch=min(int(max_island_faces),max(1,int(np.ceil(len(mesh.faces)*max_island_fraction))))
    stages=['disconnected_components']
    if permit_nonmanifold_patches:
        stages.append('nonmanifold_manifold_sheet')
    # Each successful removal strictly decreases face count. Retry analysis
    # after every accepted edit so indices and connectivity are always current.
    for stage in stages:
        attempted=set()
        for _ in range(min(256, len(mesh.faces))):
            regular=True  # match trimesh.face_adjacency and original 19 minor components
            patches=_surface_patches(current,regular_edges_only=regular)
            if len(patches)<=1:
                break
            protected_faces=_dominant_boundary_faces(current)
            incident=set()
            if stage == 'nonmanifold_manifold_sheet':
                faces=np.asarray(current.faces,np.int64)
                inv,counts,owners=_edge_occurrences(faces)
                incident=set(owners[counts[inv]>2].tolist())
                if not incident:
                    break
            changed=False
            # Small first: prioritize local defect groups and full islands.
            for patch in sorted(patches[1:],key=lambda p:(len(p),int(p.min()))):
                n=len(patch)
                # Faces are not stable after a successful edit. Retry set only
                # within the same topology state, using sorted vertex triples.
                pfaces=np.sort(np.asarray(current.faces)[patch],axis=1)
                patch_key=(stage,hashlib.blake2b(pfaces.tobytes(),digest_size=16).digest())
                if patch_key in attempted:
                    continue
                attempted.add(patch_key)
                vertices=np.asarray(current.vertices)[np.unique(current.faces[patch].ravel())]
                entry={'stage':stage,'faces':int(n),
                       'centroid_xyz':np.mean(vertices,axis=0).tolist(),
                       'bounds_xyz':[np.min(vertices,axis=0).tolist(),
                                     np.max(vertices,axis=0).tolist()]}
                if n>max_patch:
                    entry.update(decision='protected_size_limit');actions.append(entry);continue
                if any(int(fid) in protected_faces for fid in patch):
                    entry.update(decision='protected_r3_perimeter');actions.append(entry);continue
                if stage == 'nonmanifold_manifold_sheet' and not any(int(fid) in incident for fid in patch):
                    # This sheet does not meet one of the anomalous 3+/edge
                    # connections; it is not a local nonmanifold repair target.
                    continue
                support=_candidate_has_evidence(current,patch,sfm_tree,xyz,
                                                gaussian_tree,r,support_factor=support_factor)
                entry.update(support)
                if support['sfm_support'] or support['gaussian_support']:
                    entry['decision']='protected_observation_support'
                    actions.append(entry)
                    continue
                trial=_drop_patch(current,patch)
                accept,check=_trial_validation(current,trial,protected_ring,stage=stage)
                entry.update(check)
                entry['decision']='accepted' if accept else 'rolled_back_topology'
                actions.append(entry)
                if accept:
                    current=trial
                    attempted=set()
                    changed=True
                    break
            if not changed:
                break

    # A four-face nonmanifold edge may have TWO tiny redundant sheets.
    # Removing only one gives a three-face edge (still nonmanifold), so the
    # earlier strict trial will correctly reject it.  Try small pairs of
    # whole regular-adjacency sheets as one reversible transaction instead.
    # Never delete an observed sheet, and never touch the R3 perimeter.
    if permit_nonmanifold_patches:
        attempted_groups=set()
        for _ in range(min(64,len(mesh.faces))):
            faces=np.asarray(current.faces,np.int64)
            inv,counts,owners=_edge_occurrences(faces)
            keys=np.unique(inv[counts[inv]>2])
            if not len(keys):
                break
            sheets=_surface_patches(current,regular_edges_only=True)
            if len(sheets)<3:
                break
            face_to_sheet=np.full(len(faces),-1,dtype=np.int64)
            for sid,sheet in enumerate(sheets):
                face_to_sheet[sheet]=sid
            boundary_faces=_dominant_boundary_faces(current)
            changed=False
            # One local nonmanifold edge at a time; no global remeshing.
            for key in keys:
                edge_faces=np.unique(owners[inv==key])
                affected_sheets=sorted(set(int(face_to_sheet[f]) for f in edge_faces))
                small=[sid for sid in affected_sheets if sid != 0 and
                       len(sheets[sid])<=max_patch and
                       not any(int(fid) in boundary_faces for fid in sheets[sid])]
                # Bound combinations (each edge has 3-4 faces in our study).
                for nchoose in range(2,min(3,len(small))+1):
                    for choice in combinations(small,nchoose):
                        ids=np.unique(np.concatenate([sheets[sid] for sid in choice]))
                        if len(ids)>max_patch:
                            continue
                        if int(np.isin(edge_faces,ids).sum()) < int(counts[key])-2:
                            continue
                        sorted_faces=np.sort(faces[ids],axis=1)
                        keyhash=hashlib.blake2b(sorted_faces.tobytes(),digest_size=16).digest()
                        if keyhash in attempted_groups:
                            continue
                        attempted_groups.add(keyhash)
                        pts=np.asarray(current.vertices)[np.unique(faces[ids].ravel())]
                        entry={'stage':'nonmanifold_edge_group',
                               'faces':int(len(ids)),
                               'component_sheets_combined':int(nchoose),
                               'centroid_xyz':np.mean(pts,axis=0).tolist(),
                               'bounds_xyz':[np.min(pts,axis=0).tolist(),
                                             np.max(pts,axis=0).tolist()]}
                        support=_candidate_has_evidence(current,ids,sfm_tree,xyz,
                                                         gaussian_tree,r,support_factor=support_factor)
                        entry.update(support)
                        if support['sfm_support'] or support['gaussian_support']:
                            entry['decision']='protected_observation_support'
                            actions.append(entry)
                            continue
                        trial=_drop_patch(current,ids)
                        accept,check=_trial_validation(current,trial,protected_ring,
                                                       stage='nonmanifold_manifold_sheet')
                        entry.update(check)
                        entry['decision']='accepted' if accept else 'rolled_back_topology'
                        actions.append(entry)
                        if accept:
                            current=trial
                            attempted_groups=set()
                            changed=True
                            break
                    if changed:
                        break
                if changed:
                    break
            if not changed:
                break

    final_state=_topology_state(current)
    unchanged=(len(mesh.faces)==len(current.faces) and
               np.array_equal(np.asarray(mesh.vertices),np.asarray(current.vertices)) and
               np.array_equal(np.asarray(mesh.faces),np.asarray(current.faces)))
    return current,{
        'schema':'GSSceneRefine.r3_roi_conservative_topology_repair.v2',
        'algorithm':REPAIR_REVISION,
        'strategy':'whole_components_then_local_sheet_groups; accept only after topology checks, else rollback',
        'input_face_count':int(len(mesh.faces)),
        'output_face_count':int(len(current.faces)),
        'total_removed_faces':int(len(mesh.faces)-len(current.faces)),
        'hygiene':hygiene,
        'tiny_patch_face_cap':max_patch,
        'source_nonmanifold_edge_groups':initial_groups,
        'remaining_nonmanifold_edge_groups':_localized_edge_groups(current),
        'topology_before':{'open_edges':source_state['open_edge_count'],
                           'nonmanifold_edges':source_state['nonmanifold_edges'],
                           'face_components':source_state['face_components']},
        'topology_after':{'open_edges':final_state['open_edge_count'],
                          'nonmanifold_edges':final_state['nonmanifold_edges'],
                          'face_components':final_state['face_components']},
        'original_r3_ring_preserved':bool(protected_ring <= final_state['open_edges']),
        'new_open_boundary_edges':int(len(final_state['open_edges']-source_state['open_edges'])),
        'actions':actions[:500],
        'accepted_action_count':int(sum(1 for item in actions if item.get('decision')=='accepted')),
        'rejected_topology_action_count':int(sum(1 for item in actions if item.get('decision')=='rolled_back_topology')),
        'is_identical_mesh':unchanged,
        'needs_topology_gate':True,
        'not_a_semantic_certificate':True,
    }
