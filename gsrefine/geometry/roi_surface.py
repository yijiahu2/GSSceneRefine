"""Stage-2 trusted Poisson ROI-surface extraction.

Stage 1 is authoritative for the original orthographic polygons and the R3
Gaussian output. We do NOT rebuild Poisson or change the Stage-1 ROI algorithm.

The exact, fully auditable input is an optional surface-face selection mask
indexed against the ORIGINAL Poisson triangle order. When the Stage-1 output
has no such mask, a *separate, labeled inference* is possible using bounded
Poisson-supported Gaussians from the restored R3 ROI. This inferred mask does
NOT claim to be Stage-1's original R3 surface decisions.

A scalar field over mesh vertices, zero-clipped along shared mesh edges, forms
a continuous piecewise-linear surface cut (not a centroid-only face filter).
The result remains an OPEN observed Poisson surface; sealing is downstream.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


def roi_signed(points, roi, chunk=50000):
    """Positive inside all confirmed orthographic projections, negative outside."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('ROI query points must be Nx3')
    out = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk):
        xyz = points[start:start + chunk]
        value = np.full(len(xyz), np.inf)
        for view in roi.views:
            sd, _ = view.signed_and_normal(xyz)
            value = np.minimum(value, sd)
        out[start:start + len(xyz)] = value
    if not np.isfinite(out).all():
        raise ValueError('ROI signed distances contain nonfinite values')
    return out


def _clip_piecewise_linear(mesh, signed, *, tolerance=1e-10):
    """Split intersected original triangles; share every cut edge across faces.

    The signed field is interpolated linearly along mesh edges. This is an
    explicit piecewise-linear approximation, NOT exact analytical clipping of
    a large triangle against a concave ROI polygon. Its approximation budget
    and center-sample error are reported and enforced upstream.
    """
    import trimesh
    verts = np.asarray(mesh.vertices, np.float64)
    faces = np.asarray(mesh.faces, np.int64)
    s = np.asarray(signed, dtype=np.float64)
    if len(s) != len(verts) or not np.isfinite(s).all():
        raise ValueError('Invalid ROI signed scalar field')
    inside = s >= -tolerance
    all_in = np.all(inside[faces], axis=1)
    any_in = np.any(inside[faces], axis=1)
    mixed = np.flatnonzero(any_in & ~all_in)
    out_vertices = verts.tolist()
    new_faces = faces[all_in].tolist()
    original_face_ids = np.flatnonzero(all_in).tolist()
    edge_cuts = {}
    for face_id in mixed:
        face = faces[face_id]
        poly = []
        for idx in range(3):
            a = int(face[idx]); b = int(face[(idx + 1) % 3])
            a_in = bool(inside[a]); b_in = bool(inside[b])
            if a_in:
                poly.append(a)
            if a_in != b_in:
                edge = (min(a, b), max(a, b))
                if edge not in edge_cuts:
                    t = float(s[a] / (s[a] - s[b]))
                    t = float(np.clip(t, 0., 1.))
                    point = verts[a] + t * (verts[b] - verts[a])
                    edge_cuts[edge] = len(out_vertices)
                    out_vertices.append(point.tolist())
                poly.append(edge_cuts[edge])
        if len(poly) < 3:
            continue
        for j in range(1, len(poly) - 1):
            tri = [poly[0], poly[j], poly[j+1]]
            if len(set(tri)) == 3:
                new_faces.append(tri)
                original_face_ids.append(int(face_id))
    if not new_faces:
        raise ValueError('ROI extraction retained no Poisson triangles')
    # Preserve original winding and use compact vertex indexing without
    # aggressive numerical vertex welding (which can create non-manifold edges).
    cut = trimesh.Trimesh(vertices=np.asarray(out_vertices),
                          faces=np.asarray(new_faces, dtype=np.int64), process=False)
    cut.remove_unreferenced_vertices()
    return cut, {
        'source_triangles': int(len(faces)),
        'all_inside_triangles': int(all_in.sum()),
        'partly_clipped_triangles': int(len(mixed)),
        'outside_triangles': int((~any_in).sum()),
        'retained_triangles_after_cut': int(len(cut.faces)),
        'shared_intersection_vertices': int(len(edge_cuts)),
        'clipping_method': 'shared-edge, piecewise-linear signed implicit ROI',
        'original_face_ids_provenance_count': len(original_face_ids),
        'source_face_ids': np.asarray(original_face_ids, dtype=np.int64),
    }


def _guard_anchors(gaussian_ply, roi, raw_mesh, clean_sfm_xyz, cfg, *, o3d, 
                   gaussian_radius, roi_diagonal, seed=2026):
    """Infer bounded, surface-supported R3 exterior points (NOT a Stage-1 mask)."""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    boundary_limit = float(cfg['guard_max_outside_fraction']) * roi_diagonal
    if boundary_limit <= 0:
        return np.empty((0,3)), np.empty(0), {'candidate_count':0,'accepted_count':0}
    if not np.isfinite(boundary_limit):
        raise ValueError('Invalid guard outside limit')
    n = gaussian_ply.vertex_count
    # Streaming scan avoids allocating the entire full Gaussian XYZ tensor.
    candidates = []
    close_margin = float(cfg['guard_geometry_distance_multiplier']) * gaussian_radius
    max_candidates = int(cfg['guard_max_candidates'])
    if max_candidates < 1:
        raise ValueError('guard_max_candidates must be positive')
    # Deterministic reservoir sampling; process all chunks and do not favor
    # lower input rows. Keep a count independent of the sample budget.
    rng = np.random.default_rng(seed)
    seen = 0
    for start in range(0, n, 50000):
        stop = min(start + 50000, n)
        xyz = gaussian_ply.selected_xyz(np.arange(start, stop, dtype=np.int64))
        sd = roi_signed(xyz, roi)
        idx = np.flatnonzero((sd < -1e-8) & (sd >= -boundary_limit))
        for j in idx:
            seen += 1
            if len(candidates) < max_candidates:
                candidates.append(xyz[j].copy())
            else:
                replacement = rng.integers(0, seen)
                if replacement < max_candidates:
                    candidates[int(replacement)] = xyz[j].copy()
    if not candidates:
        return np.empty((0,3)), np.empty(0), {'candidate_count':0,'accepted_count':0,
             'guard_max_outside':boundary_limit,'source':'inferred_from_restored_gaussian_roi'}
    candidate_xyz=np.asarray(candidates,dtype=np.float64)
    # True triangle closest points, never just nearest centroid distance.
    legacy = o3d.geometry.TriangleMesh()
    legacy.vertices = o3d.utility.Vector3dVector(np.asarray(raw_mesh.vertices,np.float64))
    legacy.triangles = o3d.utility.Vector3iVector(np.asarray(raw_mesh.faces,np.int32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    projected = np.empty_like(candidate_xyz)
    distance = np.empty(len(candidate_xyz))
    face_index = np.empty(len(candidate_xyz),dtype=np.int64)
    for a in range(0, len(candidate_xyz), 35000):
        b = min(a+35000, len(candidate_xyz))
        q = o3d.core.Tensor(np.ascontiguousarray(candidate_xyz[a:b], np.float32))
        query = scene.compute_closest_points(q)
        closest = query['points'].numpy().astype(np.float64)
        face_index[a:b] = query['primitive_ids'].numpy().reshape(-1)
        projected[a:b] = closest
        distance[a:b] = np.linalg.norm(closest-candidate_xyz[a:b],axis=1)
    sfm_tree = cKDTree(clean_sfm_xyz)
    sample_spacing = sfm_tree.query(clean_sfm_xyz[::max(1,len(clean_sfm_xyz)//15000)], k=min(7,len(clean_sfm_xyz)), workers=-1)[0]
    if sample_spacing.ndim == 2:
        spacing=float(np.median(sample_spacing[:,-1]))
    else:
        spacing=float(np.median(sample_spacing))
    # Approximate geometric support test, never a replacement for R3's
    # original visibility/continuity/confidence certification.
    sfm_dist=sfm_tree.query(projected,k=1,workers=-1)[0]
    allowed_sfm = float(cfg['guard_sfm_spacing_multiplier'])*spacing
    projection_sd = roi_signed(projected,roi)
    # Stage-1's essential constraint: an exterior patch must be reachable on
    # the CONTINUOUS Poisson surface from an INSIDE near-boundary seed, within
    # a finite geodesic search budget. Merely being near a Gaussian is NOT
    # enough. Do not cross disconnected background components.
    face_centers=np.asarray(raw_mesh.triangles_center,dtype=np.float64)
    face_signed=roi_signed(face_centers,roi)
    inner_limit=float(cfg['guard_inside_search_fraction'])*roi_diagonal
    geo_limit=float(cfg['guard_geodesic_limit_fraction'])*roi_diagonal
    narrow=(face_signed >= -boundary_limit) & (face_signed <= inner_limit)
    seed_faces=np.flatnonzero(narrow & (face_signed >= 0))
    if len(seed_faces)==0:
        raise ValueError('No interior Poisson seed faces near ROI boundary; cannot infer R3 guard')
    edges=np.asarray(raw_mesh.face_adjacency,dtype=np.int64)
    valid=narrow[edges[:,0]] & narrow[edges[:,1]]
    edges=edges[valid]
    edge_weights=np.linalg.norm(face_centers[edges[:,0]]-face_centers[edges[:,1]],axis=1)
    count=len(face_centers)
    graph=coo_matrix((np.r_[edge_weights,edge_weights],
                      (np.r_[edges[:,0],edges[:,1]],
                       np.r_[edges[:,1],edges[:,0]])),shape=(count,count)).tocsr()
    geodesic=dijkstra(graph,directed=False,indices=seed_faces,
                      min_only=True,limit=geo_limit)
    attached=np.isfinite(geodesic) & (geodesic<=geo_limit)
    # The GPU/CPU raycaster reports face indices of the original, unmodified
    # trusted Poisson. This binds each anchor to the correct surface component.
    primitive_valid=(face_index>=0)&(face_index<len(face_signed))
    attached_anchor=np.zeros(len(candidate_xyz),dtype=bool)
    attached_anchor[primitive_valid]=attached[face_index[primitive_valid]]
    good = (attached_anchor & (distance <= close_margin) & (sfm_dist <= allowed_sfm) &
            (projection_sd >= -boundary_limit) &
            (projection_sd < -1e-8))
    support=projected[good]
    projected_sd=roi_signed(support,roi) if len(support) else np.empty(0)
    radii=np.minimum(boundary_limit,
             np.maximum(float(cfg['guard_sphere_radius_multiplier'])*gaussian_radius,
                        -projected_sd + float(cfg['guard_overlap_multiplier'])*gaussian_radius))
    report={'source':'inferred_from_restored_gaussian_roi',
            'is_exact_stage1_r3_surface_mask':False,
            'candidate_count':int(seen), 'candidate_sampled':int(len(candidate_xyz)),
            'interior_seed_faces':int(len(seed_faces)),
            'connected_boundary_faces':int(np.count_nonzero(attached & narrow)),
            'geodesic_limit':float(geo_limit),
            'anchor_connected_count':int(attached_anchor.sum()),
            'accepted_count':int(len(support)),
            'guard_max_outside':boundary_limit,
            'support_distance_limit':close_margin,
            'sfm_spacing_median':spacing,
            'sfm_support_distance_limit':allowed_sfm,
            'maximum_guard_sphere_radius':float(radii.max()) if len(radii) else 0.,
            'warning':'Inferred geometric support is not the same as the original R3 surface mask; inspect extracted surface before export.'}
    return support, radii, report



def _refine_faces_conforming(mesh, scalar, marked, evaluate, *, maximum_added_faces):
    """Bisect all edges of marked faces, and split their unmarked neighbors too.

    This is a geometric WORKING COPY. Shared edge midpoints are unique globally;
    no T-junctions, no face deletion and no changes to the input Poisson mesh.
    The 1/2/3-edge templates preserve every triangle's original winding.
    """
    import trimesh
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    marked = np.asarray(marked, dtype=np.int64)
    if not len(marked):
        return mesh, np.asarray(scalar), {'added_faces': 0, 'split_edges': 0}
    if np.any(marked < 0) or np.any(marked >= len(faces)):
        raise ValueError('Invalid marked ROI triangle indices')
    n = len(vertices)
    if n <= 0 or n * (n + 1) > np.iinfo(np.int64).max:
        raise ValueError('Poisson vertex index range too large for safe edge indexing')
    stride = n + 1

    def edge_keys(tris):
        adjacent = np.stack((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]), axis=1)
        low = adjacent.min(axis=2)
        high = adjacent.max(axis=2)
        return low * stride + high

    split_keys = np.unique(edge_keys(faces[marked]).reshape(-1))
    all_keys = edge_keys(faces)
    split_flags = np.isin(all_keys, split_keys)
    affected = np.flatnonzero(split_flags.any(axis=1))
    increases = int(split_flags.sum())  # each split edge adds exactly one face in each incident triangle
    if increases > maximum_added_faces:
        raise ValueError(f'ROI refinement needs {increases} additional triangles, exceeding safety budget '
                         f'{maximum_added_faces}; refusing to alter source geometry')
    endpoints = np.column_stack([split_keys // stride, split_keys % stride]).astype(np.int64)
    mid_points = (vertices[endpoints[:, 0]] + vertices[endpoints[:, 1]]) * .5
    mid_ids = np.arange(n, n + len(split_keys), dtype=np.int64)
    lookup = dict(zip(split_keys.tolist(), mid_ids.tolist()))
    result = [faces[~split_flags.any(axis=1)]]
    replacements = []
    for face_id in affected:
        v = faces[face_id]
        split = split_flags[face_id]
        m = [lookup[int(k)] if present else -1 for k, present in zip(all_keys[face_id], split)]
        how_many = int(np.count_nonzero(split))
        if how_many == 1:
            i = int(np.flatnonzero(split)[0])
            a, b, c = int(v[i]), int(v[(i+1)%3]), int(v[(i+2)%3])
            replacements.extend(((a, m[i], c), (m[i], b, c)))
        elif how_many == 2:
            i = next(i for i in range(3) if split[i] and split[(i+1)%3])
            a, b, c = int(v[i]), int(v[(i+1)%3]), int(v[(i+2)%3])
            p, q = m[i], m[(i+1)%3]
            replacements.extend(((b, q, p), (a, p, c), (p, q, c)))
        elif how_many == 3:
            a, b, c = map(int, v)
            p, q, r = m
            replacements.extend(((a, p, r), (p, b, q), (r, q, c), (p, q, r)))
        else:
            raise AssertionError('All affected faces must have at least one split edge')
    if replacements:
        result.append(np.asarray(replacements, dtype=np.int64))
    new_faces = np.concatenate(result, axis=0)
    if len(new_faces) != len(faces) + increases:
        raise RuntimeError('Conforming ROI refinement triangle count mismatch')
    new_verts = np.concatenate((vertices, mid_points), axis=0)
    new_signed = np.concatenate((np.asarray(scalar, dtype=np.float64), evaluate(mid_points)), axis=0)
    if not np.isfinite(new_signed).all():
        raise ValueError('Nonfinite ROI values on refined Poisson vertices')
    refined = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    return refined, new_signed, {'added_faces': int(increases),
                                 'split_edges': int(len(split_keys)),
                                 'affected_faces_including_neighbors': int(len(affected))}


def _find_roi_aliases(mesh, scalar, evaluate, tolerance, *, chunk_size=12000):
    """Detect actual implicit-ROI sign disagreements with linear triangles.

    Inspect the centroid and three edge midpoints. A 1-Lipschitz conservative
    bound skips far-away all-inside or all-outside triangles. A detected
    disagreement triggers conforming local subdivision, never silent deletion.
    These finite probes do not certify every sub-triangle of a concave ROI.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    s = np.asarray(scalar, dtype=np.float64)
    if len(s) != len(verts) or not np.isfinite(s).all():
        raise ValueError('Invalid implicit ROI field for alias detection')
    signs = s[faces]
    all_inside = np.all(signs >= -tolerance, axis=1)
    all_outside = np.all(signs < -tolerance, axis=1)
    # A signed planar-polygon distance is 1-Lipschitz. For bounded R3 guard
    # fields this is used as an acceleration heuristic, not exact certification.
    maybe = (~all_inside) & (~all_outside)
    for beginning in range(0, len(faces), 100000):
        ending = min(beginning + 100000, len(faces))
        tri = verts[faces[beginning:ending]]
        edge_length = np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=2).max(axis=1)
        inside = all_inside[beginning:ending]
        outside = all_outside[beginning:ending]
        value = signs[beginning:ending]
        maybe[beginning:ending] |= (outside & (value.max(axis=1) + edge_length >= -tolerance))
        maybe[beginning:ending] |= (inside & (value.min(axis=1) - edge_length <= tolerance))
    candidates = np.flatnonzero(maybe)
    mismatched_faces = []
    conflicting_probes = 0
    for beginning in range(0, len(candidates), chunk_size):
        ids = candidates[beginning:beginning+chunk_size]
        tri = verts[faces[ids]]
        points = np.stack((tri.mean(axis=1),
                           .5*(tri[:,0]+tri[:,1]),
                           .5*(tri[:,1]+tri[:,2]),
                           .5*(tri[:,2]+tri[:,0])), axis=1)
        truth = evaluate(points.reshape(-1,3)).reshape(-1,4)
        v = signs[ids]
        linear = np.stack((v.mean(axis=1),
                           .5*(v[:,0]+v[:,1]),
                           .5*(v[:,1]+v[:,2]),
                           .5*(v[:,2]+v[:,0])), axis=1)
        mismatch = ((truth > tolerance) & (linear < -tolerance)) | \
                   ((truth < -tolerance) & (linear > tolerance))
        # A triangle whose signed corners all agree but whose probe is inside
        # with the opposite sign must also be refined, even if interpolation is 0.
        mismatch |= (all_outside[ids,None] & (truth > tolerance))
        mismatch |= (all_inside[ids,None] & (truth < -tolerance))
        conflicting_probes += int(mismatch.sum())
        if np.any(mismatch):
            mismatched_faces.append(ids[mismatch.any(axis=1)])
    return (np.concatenate(mismatched_faces) if mismatched_faces else np.empty(0, dtype=np.int64),
            {'probed_faces': int(len(candidates)), 'conflicting_probes': conflicting_probes})


def _adapt_roi_mesh(mesh, scalar, evaluate, cfg):
    """Refine only genuinely aliased ROI triangles, with strict memory limits."""
    tolerance = float(cfg.get('max_unresolved_face_violation', 1e-5))
    levels = int(cfg.get('roi_crop_max_refinement_levels', 6))
    max_marked = int(cfg.get('roi_crop_max_marked_faces_per_level', 10000))
    max_extra = int(cfg.get('roi_crop_max_extra_faces', 200000))
    if not (0 < tolerance < 1 and 0 <= levels <= 12 and
            max_marked > 0 and max_extra > 0):
        raise ValueError('Invalid Stage-2 adaptive ROI crop safety bounds')
    refined = mesh
    signed = np.asarray(scalar, dtype=np.float64)
    added_total = 0
    resolved = []
    first_conflicts = None
    for depth in range(levels + 1):
        marked, diagnostics = _find_roi_aliases(refined, signed, evaluate, tolerance)
        if first_conflicts is None:
            first_conflicts = int(diagnostics['conflicting_probes'])
        if not len(marked):
            return refined, signed, {
                'method':'local_conforming_midpoint_refinement',
                'initial_conflicting_probes':first_conflicts,
                'final_conflicting_probes':0,
                'refinement_levels':len(resolved),
                'added_faces_total':added_total,
                'steps':resolved,
                'residual_limitations':'Four finite probes per candidate triangle; visual validation still required.'}
        if depth == levels or len(marked) > max_marked:
            raise ValueError(f'ROI implicit crop remains ambiguous: {len(marked)} triangles, '
                             f'{diagnostics["conflicting_probes"]} conflicting probes '
                             f'after {depth} refinement passes. Limits: levels={levels}, '
                             f'marked_per_level={max_marked}. Refusing approximate seal.')
        refined, signed, step = _refine_faces_conforming(
            refined, signed, marked, evaluate, maximum_added_faces=max_extra-added_total)
        added_total += step['added_faces']
        resolved.append({'pass':depth+1,'marked_faces':int(len(marked)),
                         'conflicting_probes':diagnostics['conflicting_probes'], **step})
    raise AssertionError('Unreachable ROI refinement loop')

def extract_roi_surface(raw_mesh, roi, gaussian_ply, sfm_xyz, cfg, *, o3d,
                        gaussian_radius, certified_face_mask=None):
    """Extract stage-1-compatible *observed* ROI surface, before sealing.

    Surface mask refers to ORIGINAL Poisson triangle rows; if supplied it
    overrides inference. Otherwise the R3 point support generates a bounded
    implicit envelope outside the original two orthographic polygons.
    """
    import trimesh
    from scipy.spatial import cKDTree
    if not isinstance(raw_mesh, trimesh.Trimesh) or not len(raw_mesh.faces):
        raise ValueError('Stage-1 trusted surface must be a nonempty triangle mesh')
    vertices=np.asarray(raw_mesh.vertices,dtype=np.float64)
    if not np.isfinite(vertices).all():
        raise ValueError('Poisson vertices are non-finite')
    if certified_face_mask is not None:
        mask=np.asarray(certified_face_mask)
        if mask.shape != (len(raw_mesh.faces),) or mask.dtype != bool:
            raise ValueError('Certified Stage-1 Poisson face mask must match ORIGINAL mesh face rows')
        # Directly select certified original-triangle IDs. Stage 1 is authoritative;
        # never mix this with guessed stage-2 guard geometry.
        if not mask.any():
            raise ValueError('Certified Stage-1 face mask is empty')
        cropped=raw_mesh.submesh([np.flatnonzero(mask)], append=True, repair=False)
        cropped.remove_unreferenced_vertices()
        report={'selection_mode':'certified_stage1_surface_faces',
                'certified_faces_kept':int(mask.sum()),
                'total_original_faces':int(len(raw_mesh.faces)),
                'guard':{'source':'stage1_certified_mask','is_exact_stage1_r3_surface_mask':True},
                'cropped_faces':int(len(cropped.faces))}
        return cropped, report

    gauss_points=gaussian_ply.positions()
    diag=float(np.linalg.norm(np.ptp(gauss_points,axis=0)))
    if not np.isfinite(diag) or diag <= 0:
        raise ValueError('Invalid ROI Gaussian bounding diagonal')
    anchors, radii, support_report = _guard_anchors(
        gaussian_ply, roi, raw_mesh, sfm_xyz, cfg,
        o3d=o3d, gaussian_radius=gaussian_radius, roi_diagonal=diag)
    tree = cKDTree(anchors) if len(anchors) else None

    def roi_guard_field(xyz):
        source = roi_signed(xyz, roi)
        if tree is None:
            return source
        output = source.copy()
        for i in range(0,len(source),50000):
            j=min(i+50000,len(source))
            distance, index=tree.query(np.asarray(xyz)[i:j],
                                       k=min(8,len(anchors)), workers=-1)
            if distance.ndim==1:
                distance,index=distance[:,None],index[:,None]
            support=np.max(radii[index]-distance,axis=1)
            bounded=np.minimum(support,source[i:j]+support_report['guard_max_outside'])
            output[i:j]=np.maximum(source[i:j],bounded)
        return output

    original_field=roi_guard_field(vertices)
    working_mesh, working_field, refinement = _adapt_roi_mesh(
        raw_mesh, original_field, roi_guard_field, cfg)
    if refinement['refinement_levels']:
        print('[stage2-cleanup] Adaptive ROI surface clip resolved '
              f'{refinement["initial_conflicting_probes"]} conflicting probes '
              f'with {refinement["refinement_levels"]} conforming pass(es), '
              f'{refinement["added_faces_total"]} added triangles.',flush=True)

    cropped, clip_report=_clip_piecewise_linear(working_mesh,working_field)
    clip_report.pop('source_face_ids', None)  # debug arrays are not part of metadata
    report={'selection_mode':'inferred_r3_gaussian_support',
            'guard':support_report,'clipping':clip_report,
            'adaptive_refinement':refinement,
            'maximum_unresolved_face_violation':float(cfg['max_unresolved_face_violation']),
            'raw_poisson_faces':int(len(raw_mesh.faces)),
            'cropped_faces':int(len(cropped.faces)),
            'input_source_unchanged':True,
            'caveat':'Clipping is a finite-probe approximation of an implicit ROI on a locally refined source mesh, not an analytically exact curved boundary.'}
    return cropped,report
