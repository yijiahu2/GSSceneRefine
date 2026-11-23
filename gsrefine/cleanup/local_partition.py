"""Numerical separation audit of the authenticated *local* R3 sheet."""
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from gsrefine.cleanup import barrier_separation_gate as gate
from gsrefine.cleanup.roi_barrier_parity import segment_parity, UNKNOWN
from gsrefine.cleanup.stage2_runtime import measured
from gsrefine.cleanup.threshold_distance import exact
from gsrefine.geometry import original_wall_audit as grid


class Domain:
    """Final R3, optionally intersected with a half-space for a fresh local audit."""
    def __init__(self, field, normal=None, offset=0., excluded_centers=None, exclusion_radius=0., boxes=None):
        self.field, self.normal, self.offset = field, normal, offset
        self.holes = None if excluded_centers is None else cKDTree(excluded_centers)
        self.exclusion_radius = exclusion_radius
        self.boxes = boxes

    def plane(self, p):
        if self.boxes is not None:
            result = np.full(len(p),np.inf)
            for lo,hi in self.boxes:
                result = np.minimum(result, np.maximum(lo-p,p-hi).max(axis=1))
            return result
        if self.holes is not None:
            return self.holes.query(p)[0] - self.exclusion_radius
        return np.full(len(p), np.inf) if self.normal is None else p @ self.normal - self.offset

    def __call__(self, p):
        return np.minimum(self.field(p), self.plane(p))

    def membership(self, p):
        return np.minimum(getattr(self.field, 'region_membership', getattr(self.field, 'membership', self.field))(p), self.plane(p))

    def allow_link(self, p, q):
        if self.holes is None and self.boxes is None:
            return np.ones(len(p),bool) if self.normal is None else np.minimum(self.plane(p),self.plane(q))>0
        # Nearest-centre distance minus r is 1-Lipschitz. A positive midpoint
        # margin larger than half-length certifies the entire segment. Reject
        # unresolved grazing segments instead of jumping over an excluded ball.
        def clear(a,b,depth):
            mid=(a+b)*.5
            margin=self.plane(mid)
            safe=margin>np.linalg.norm(b-a,axis=1)*.5+1e-10
            candidates=(margin>0)&~safe
            if depth and candidates.any():
                u,v=a[candidates],b[candidates]
                m=mid[candidates]
                safe[candidates]=clear(u,m,depth-1)&clear(m,v,depth-1)
            return safe
        return clear(p,q,6)

    def link_radius(self, p):
        return self.interior_radius(p)

    def interior_radius(self, p):
        return np.maximum(0., np.minimum(getattr(self.field, 'region_radius', self.field.interior_radius)(p), self.plane(p)))


def rim_witnesses(mesh, domain, scene, o3d, spacing):
    """A local closed loop with odd intersection count witnesses a bypass.
    Test both legs of the detour and the cross-wall chord; report actual XYZ.
    """
    faces = np.asarray(mesh.faces)
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    unique, first, counts = np.unique(edges, axis=0, return_index=True, return_counts=True)
    free = counts == 1
    edges, ids = unique[free], first[free] // 3
    vertices = np.asarray(mesh.vertices)
    ends = vertices[edges]
    mid = np.concatenate([ends[:,0]+t*(ends[:,1]-ends[:,0]) for t in (.25,.5,.75)])
    ids = np.tile(ids,3)
    normal = np.asarray(mesh.face_normals)[ids]
    step = np.asarray(mesh.triangles_center)[ids] - mid
    length = np.linalg.norm(step, axis=1)
    guard = 16*np.finfo(np.float32).eps * max(1., abs(vertices).max())
    found = []
    pending = np.ones(len(ids), bool)
    for fraction in (.25, .05):
        width = np.minimum(length * fraction, spacing * fraction)
        rows = np.flatnonzero(pending & (width > guard * 10))
        if not len(rows):
            continue
        d = width[rows, None]
        tangent = step[rows] * (d / length[rows, None])
        a = mid[rows] + tangent + normal[rows]*d
        b = mid[rows] + tangent - normal[rows]*d
        tip = mid[rows] - tangent
        valid = np.ones(len(rows), bool)
        for p, q in ((a, tip), (tip, b), (b, a)):
            valid &= domain.membership(p) > 0
            valid &= grid.link_r3_filter(p, q, domain, samples=9)
        for p, q in ((a, tip), (tip, b)):
            valid &= grid._ray_hit_free(scene, o3d, p, q, eps=guard*.05, batch=8000)
        odd, reliable = segment_parity(scene, o3d, a, b, normals=mesh.face_normals,
                                       endpoint_tol=guard*.2)
        ok = valid & reliable & (odd == 1)
        pending[rows[ok]] = False
        found.extend(np.stack([a[ok], tip[ok], b[ok]], axis=1))
    return np.asarray(found, float).reshape(-1, 3, 3), int(len(edges))


def source_hits(mesh, paths):
    scene, o3d = grid.make_scene(mesh)
    found = []
    for p, q in ((paths[:, 0], paths[:, 1]), (paths[:, 1], paths[:, 2])):
        length = np.linalg.norm(q-p, axis=1)
        rays = np.asarray(np.c_[p, (q-p)/length[:, None]], np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))
        t = hit['t_hit'].numpy()
        keep = (t > 0) & (t < length)
        found.extend(hit['primitive_ids'].numpy()[keep])
    return np.unique(found).astype(np.int64)


def anchor_index(g, seed, up, domain, scene, o3d):
    available = np.flatnonzero(g['passable'])
    if seed is None:
        top, _, _ = grid.choose_top_bottom_seeds(g['passable'], g['xyz'], g['shape'], up)
        return int(top[np.argmax(g['xyz'][top] @ up)])
    seed = np.asarray(seed, float).reshape(1, 3)
    if domain.membership(seed)[0] <= 0:
        return None
    dist, ix = cKDTree(g['xyz'][available]).query(seed[0], k=min(12, len(available)))
    for d, j in zip(np.atleast_1d(dist), np.atleast_1d(ix)):
        if d > g['voxel']*1.5:
            continue
        i = available[j]
        target = g['xyz'][i:i+1]
        if d == 0 or (grid.link_r3_filter(seed, target, domain, samples=11)[0] and
                grid._ray_hit_free(scene, o3d, seed, target, eps=0., batch=1)[0]):
            return int(i)
    return None


@measured('local_two_resolution_audit')
def audit(mesh, domain, up, seed, lo, hi, h, folder, *, strict=False, original_wall=False):
    from gsrefine.cleanup.stage2_onepass import _atomic_json

    folder.mkdir(parents=True)
    scene, o3d = grid.make_scene(mesh)
    levels = {}
    # Caps scale with resolution; avoid accidentally generating identical grids.
    settings = [('coarse', 1.6*h, 95000), ('fine', h, 250000)]
    if strict:
        settings = [('coarse', h, 250000), ('fine', .72*h, 640000)]
    report = {'wall':'original_global_largest' if original_wall else 'local_main', 'flood':{}}
    normal = np.asarray(mesh.face_normals)
    graph_paths = []
    def observe(xyz, labels, src, dst, a, b):
        same = labels[a] == labels[b]
        a, b = a[same], b[same]
        if not len(a):
            return
        bits, reliable = segment_parity(scene, o3d, xyz[a], xyz[b], normals=normal)
        ids = np.flatnonzero(reliable & (bits == 1))
        if len(ids):
            j = ids[len(ids)//2]
            path = grid._path_witness(len(xyz), src, dst, [a[j]], [b[j]], xyz)
            if len(path):
                graph_paths.append(path)
                np.save(folder/('graph_bypass_%d.npy' % len(graph_paths)), path)

    for name, size, cap in settings:
        report['flood'][name] = grid.flood_test(mesh, domain, up, lo, hi, name=name,
            h_target=size, max_cells=cap, output_dir=folder, link_samples=11 if strict else 3,
            observe_links=observe)
        levels[name] = gate._load(folder/(name+'_connectivity.npz'), name)
    verdict = gate.evaluate(audit_dir=folder, audit_record=report, field=domain,
                            wall_scene=scene, o3d=o3d, normals=normal)
    errors = verdict['blocking_reasons']
    fine = levels['fine']
    fine_id = anchor_index(fine, seed, up, domain, scene, o3d)
    anchor = None if fine_id is None else fine['xyz'][fine_id]
    if original_wall and anchor is not None:
        coarse = levels['coarse']
        ci = anchor_index(coarse, seed if seed is not None else anchor, up, domain, scene, o3d)
        coarse_region = None if ci is None else int(coarse['regions'].ravel()[ci])
        fine_region = int(fine['regions'].ravel()[fine_id])
        mapping = (verdict.get('cross_scale') or {}).get('correspondence',{})
        if (len(coarse['material']) != 2 or len(fine['material']) != 2 or
                coarse_region not in coarse['material'] or fine_region not in fine['material'] or
                mapping.get(str(coarse_region),{}).get('fine_region') != fine_region):
            errors.append('Exterior reference or two main regions not confirmed across scales')
    if graph_paths:
        errors.append('A zero-crossing path connects opposite ends of an odd chord')
    if anchor is None:
        errors.append('Exterior seed could not be linked in the audited domain')
    if levels['coarse']['voxel'] < 1.05*fine['voxel']:
        errors.append('Insufficient separation of grid resolutions')
    index = None
    if not errors:
        labels = np.full(fine['shape'], UNKNOWN, np.int8)
        graph = verdict['levels']['fine']['graph']['material_region_labels']
        region = int(fine['regions'].ravel()[fine_id])
        if str(region) not in graph:
            errors.append('Exterior anchor is not in a verified material region')
        else:
            for key, bit in graph.items():
                labels[fine['regions'] == int(key)] = bit ^ graph[str(region)]
            index = dict(origin=fine['origin'], voxel=fine['voxel'], shape=fine['shape'],
                         labels=labels)
    witnesses, boundary_count = rim_witnesses(mesh, domain, scene, o3d, fine['voxel'])
    if len(witnesses):
        errors.append('Interior free-rim bypass with verified odd chord')
        index = None
    found_rims = witnesses
    route_evidence = []
    verified_walk = []
    for path in graph_paths:
        a,b = path[:-1], path[1:]
        # Recheck the full detour on the unsnapped local sheet, including the
        # reverse direction and much denser sampling of uncertain R3 links.
        free = grid._ray_hit_free(scene,o3d,a,b,eps=0.,batch=8000)
        free &= grid._ray_hit_free(scene,o3d,b,a,eps=0.,batch=8000)
        within = grid.link_r3_filter(a,b,domain,samples=63)
        route_evidence.append(dict(edges=len(a), blocked_by_mesh=int((~free).sum()),
                                   outside_domain=int((~within).sum())))
        if free.all() and within.all():
            verified_walk.append(np.stack([a,(a+b)*.5,b],axis=1))
    np.savez_compressed(folder/'bypass_paths.npz', xyz=witnesses)
    if verified_walk and not len(witnesses) and strict:
        # A route by itself localizes no specific hole. Its segments can still
        # identify original triangles omitted from a verified extraction.
        witnesses = np.concatenate(verified_walk)
    verdict['grid_gate_status'] = verdict['status']
    verdict['status'] = 'LOCAL_PARTITION_SUPPORTED_NUMERICALLY' if index is not None else 'LOCAL_PARTITION_UNVERIFIED'
    report.update(verdict=verdict, accepted=index is not None, seed_xyz=anchor,
                  boundary_edges=boundary_count, rim_bypasses=len(found_rims),
                  recovery_segments=len(witnesses), graph_route_rechecks=route_evidence,
                  verification='finite-resolution/local rim paths and original triangle intersections')
    _atomic_json(folder/'local_partition_report.json', report)
    return index, anchor, witnesses, report


def _refine_faces(source, hit_ids, field, signed):
    """Conforming source-face subdivision, then R3 clipping. No caps or extrapolation."""
    import trimesh

    verts = np.asarray(source.vertices)
    faces = np.asarray(source.faces)
    incidence = source.faces_unique_edges
    selected = np.unique(incidence[hit_ids])
    edge_ids = np.full(len(source.edges_unique), -1, int)
    edge_ids[selected] = np.arange(len(selected)) + len(verts)
    mids = verts[source.edges_unique[selected]].mean(axis=1)
    # Split every incident triangle along the same edge, including neighbors.
    edge_mid = edge_ids[incidence]
    changed = (edge_mid >= 0).any(axis=1)
    rows = np.flatnonzero(changed)
    centroids = verts[faces[rows]].mean(axis=1)
    out = []
    for j, row in enumerate(rows):
        boundary = []
        for k in range(3):
            boundary.append(faces[row, k])
            if edge_mid[row, k] >= 0:
                boundary.append(edge_mid[row, k])
        center = len(verts) + len(mids) + j
        out.extend((u, v, center) for u, v in zip(boundary, boundary[1:] + boundary[:1]))
    triangles = np.vstack([faces[~changed], np.asarray(out)])
    refined = trimesh.Trimesh(vertices=np.vstack([verts, mids, centroids]), faces=triangles, process=False)
    return refined, np.r_[signed, field(mids), field(centroids)]


def repair_from_source(source, hit_ids, field):
    from gsrefine.geometry.mesh_clip import clip_piecewise_linear
    from gsrefine.geometry.r3_main_auxiliary import split_direct_r3_mesh
    mesh = source
    signed = grid.vertex_margins(source,field,2048)
    scale = max(1.,np.abs(source.bounds).max())
    for iteration in range(5):
        mesh, signed = _refine_faces(mesh,hit_ids,field,signed)
        edges = np.asarray(mesh.edges_unique)
        a,b = signed[edges[:,0]],signed[edges[:,1]]
        opposite = (a*b < 0)
        crossing = edges[opposite]
        roots = {}
        if len(crossing):
            xyz, _, _ = grid.roots_on_crossing_edges(np.asarray(mesh.vertices),crossing,
                 a[opposite],b[opposite],field,iterations=22,eval_batch=2048)
            roots = {tuple(sorted(edge)):point for edge,point in zip(crossing,xyz)}
        clipped, provenance = clip_piecewise_linear(mesh,signed,tolerance=0.,edge_roots=roots)
        # Correct a curved R3 intersection on the same source triangle. Only
        # chords introduced by clipping qualify; genuine original rims never do.
        face_edges = clipped.faces_unique_edges.ravel()
        unique, first, counts = np.unique(face_edges,return_index=True,return_counts=True)
        rim = counts==1
        edge = clipped.edges_unique[unique[rim]]
        orig = provenance['source_face_ids'][first[rim]//3]
        mixed = (signed[mesh.faces[orig]].min(axis=1)<0)
        ends = np.asarray(clipped.vertices)[edge[mixed]]
        if not len(ends):
            break
        residual = np.abs(field(ends.mean(axis=1)))
        hit_ids = np.unique(orig[mixed][residual>1e-5*scale])
        if not len(hit_ids) or len(mesh.faces)+16*len(hit_ids)>5_000_000:
            break
    return split_direct_r3_mesh(clipped)
