"""Safe sealing of trusted *open* Poisson surfaces for signed-distance pruning.

Scientific sealing functions adapted from the user-supplied, experimentally
validated stage2_poisson_sdf.py (2026-11-02). No SfM reconstruction or 3DGS
PLY readers are replicated here: use the canonical GSSceneRefine modules.

Only the largest connected surface may be selected, with a strict face-share
guard; secondary components are recorded, never presented as observed solids.
"""
from __future__ import annotations
import numpy as np

def _nonmanifold_edge_samples(mesh, limit=5):
    """Count edges incident to >2 triangles; include short actionable samples.

    This diagnoses topology; it intentionally does *not* guess which of three
    legitimate incident faces is expendable.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return {"count": 0, "samples": []}
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    idx = np.flatnonzero(counts > 2)
    examples = [{"vertex_indices": edges[k].tolist(), "incident_faces": int(counts[k])}
                for k in idx[:limit]]
    return {"count": int(len(idx)), "samples": examples}


def prepare_open_poisson(mesh, *, weld_digits=8):
    """Conservative topology hygiene on a COPY of the trusted open Poisson.

    Mirrors the geometrical preprocessing of the experimentally used stage-2
    script (vertex welding and removal of degenerate faces), with an additional
    removal of exact duplicate triangles. No arbitrary face deletion, inferred
    geometry, mesh reconstruction or large-hole filling is performed here.
    The untouched mesh remains the one used for SfM-to-Poisson P95 fitting.

    One non-manifold edge can arise from two regular incident faces plus a
    duplicate triangle. That *can* be fixed deterministically. Three distinct
    valid faces sharing an edge cannot be repaired safely by choosing one to
    discard; downstream sealing must still fail with diagnostics.
    """
    import trimesh
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError('Trusted Poisson must be a nonempty triangle mesh')
    if not isinstance(weld_digits, int) or not (0 <= weld_digits <= 14):
        raise ValueError('weld_digits must be an integer in [0,14]')
    cleaned = mesh.copy()
    before_faces = int(len(cleaned.faces))
    before_vertices = int(len(cleaned.vertices))
    before_nonmanifold = _nonmanifold_edge_samples(cleaned)

    # The previous verified Poisson stage first merged identical/nearly
    # coincident vertices (world-coordinate rounded to 8 decimal places).
    # Keep this operation in the ORIGINAL COLMAP world coordinate system.
    cleaned.merge_vertices(digits_vertex=weld_digits)
    welded_vertices = int(len(cleaned.vertices))

    valid = cleaned.nondegenerate_faces()
    degenerate_count = int((~valid).sum())
    if degenerate_count:
        cleaned.update_faces(valid)
    unique = cleaned.unique_faces()
    duplicate_count = int((~unique).sum())
    if duplicate_count:
        cleaned.update_faces(unique)
    cleaned.remove_unreferenced_vertices()

    if len(cleaned.faces) == 0:
        raise ValueError('Trusted Poisson became empty after removing invalid faces')
    report = {
        'input_faces': before_faces,
        'input_vertices': before_vertices,
        'vertices_after_welding_before_face_cleanup': welded_vertices,
        'weld_digits_in_original_colmap_coordinates': weld_digits,
        'degenerate_faces_removed': degenerate_count,
        'duplicate_faces_removed': duplicate_count,
        'output_faces': int(len(cleaned.faces)),
        'output_vertices': int(len(cleaned.vertices)),
        'nonmanifold_edges_before': before_nonmanifold,
        'nonmanifold_edges_after': _nonmanifold_edge_samples(cleaned),
        'input_poisson_modified': False,
        'automatic_ambiguous_face_removal': False,
        'hole_filling_applied': False,
    }
    return cleaned, report


def boundary_loops(mesh):
    """Return consistently directed boundary loops of a winding-consistent mesh."""
    f = np.asarray(mesh.faces, dtype=np.int64)
    directed = np.concatenate([f[:, [0,1]], f[:, [1,2]], f[:, [2,0]]])
    sorted_edges = np.sort(directed, axis=1)
    _, indices, counts = np.unique(sorted_edges, axis=0, return_index=True, return_counts=True)
    nonmanifold = int((counts > 2).sum())
    if nonmanifold:
        details = _nonmanifold_edge_samples(mesh)
        raise ValueError(f'Mesh still has {nonmanifold} non-manifold edges (>2 faces) '
                         f'after conservative cleanup; no arbitrary face deletion. '
                         f'Inspect vertex indices / incident faces: {details["samples"]}')
    boundary = directed[indices[counts == 1]]
    if not len(boundary):
        return [], {'boundary_edges':0,'nonmanifold_edges':0}
    start, cnt0 = np.unique(boundary[:,0], return_counts=True)
    end, cnt1 = np.unique(boundary[:,1], return_counts=True)
    if not (np.array_equal(start,end) and (cnt0 == 1).all() and (cnt1 == 1).all()):
        raise ValueError('Boundary is branched or face winding is inconsistent. Run basic topology repair first.')
    next_vertex = dict(zip(boundary[:,0].tolist(), boundary[:,1].tolist()))
    unseen = set(next_vertex)
    loops = []
    while unseen:
        first = next(iter(unseen))
        here = first
        ring = []
        while here in unseen:
            ring.append(here)
            unseen.remove(here)
            here = next_vertex[here]
            if len(ring) > len(next_vertex):
                raise ValueError('Loop traversal failed')
        if here != first:
            raise ValueError('Unclosed or branched boundary loop')
        if len(ring) < 3:
            raise ValueError('Degenerate boundary ring')
        loops.append(np.asarray(ring, dtype=np.int64))
    loops.sort(key=len,reverse=True)
    return loops, {'boundary_edges':int(len(boundary)), 'boundary_loop_count':len(loops),
                   'boundary_loop_vertex_counts':[len(x) for x in loops]}


def choose_main_component(mesh, min_fraction=.85):
    """Keep dominant connected component only; report discarded components explicitly."""
    import trimesh
    if not mesh.is_winding_consistent:
        trimesh.repair.fix_winding(mesh)
    comps = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1)
    sizes = sorted((len(c) for c in comps), reverse=True)
    largest = max(comps, key=len)
    share = float(len(largest)/len(mesh.faces))
    report = {'component_count':len(comps), 'main_face_count':int(len(largest)),
              'original_face_count':int(len(mesh.faces)), 'main_face_fraction':share,
              'other_faces_total':int(len(mesh.faces)-len(largest)),
              'top_component_face_counts':sizes[:12],
              'note':'Only the dominant connected surface is used for sealed SDF; other components are excluded and reported.'}
    if share < min_fraction:
        raise ValueError(f'Largest component only {share:.1%} of faces; other geometry is substantial. '
                         'Do not discard valid rock geometry automatically. Inspect components.')
    result = mesh.submesh([largest], append=True, repair=False)
    result.remove_unreferenced_vertices()
    return result, report


def seal_main(mesh, up_axis: int, up_sign: int, bottom_margin: float,
              max_relative_margin: float=.50, base_triangulation='shapely'):
    """Create side skirt and constrained-triangulated base for a simple projected boundary ring.

    Returns (sealed_mesh, report, skirt_only_mesh). Does not silently fill inner holes.
    """
    import trimesh
    from shapely.geometry import Polygon
    from shapely import constrained_delaunay_triangles

    if up_axis not in (0,1,2) or up_sign not in (-1,1):
        raise ValueError('up_axis must be 0/1/2 and up_sign +/-1')
    loops, diagnostic = boundary_loops(mesh)
    if not loops:
        if not (mesh.is_watertight and mesh.is_volume):
            raise ValueError('Mesh has no open boundary but is not a valid volume; check non-manifold/self-overlap')
        return mesh, {'already_watertight':True, **diagnostic}, None
    if len(loops) != 1:
        raise ValueError(f'Dominant component has {len(loops)} boundary rings, not one. '
                         'Automated skirt could seal genuine cavities or self-intersect. '
                         'Inspect rings; fix small holes before retrying.')
    ring = loops[0]
    plane_axes = [k for k in (0,1,2) if k != up_axis]
    uv = np.asarray(mesh.vertices)[ring][:,plane_axes]
    # A 2D polygon is mandatory: overlapping XY boundary causes vertical skirt intersections.
    poly = Polygon(uv)
    projected_area = float(abs(poly.area))
    span = np.ptp(np.asarray(mesh.vertices)[:,plane_axes],axis=0)
    if not (poly.is_valid and not poly.is_empty and projected_area > 1e-6*float(np.prod(span))):
        from shapely.validation import explain_validity
        raise ValueError('Outer boundary projection is invalid or collapses; skirt would self-cross: '
                         + explain_validity(poly))
    if not poly.exterior.is_simple:
        raise ValueError('Projected boundary is not simple')
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    aligned_height = vertices[:,up_axis]*up_sign
    delta_height = float(aligned_height.max()-aligned_height.min())
    if not (0 < bottom_margin <= max_relative_margin * delta_height):
        raise ValueError(f'bottom_margin={bottom_margin} should be >0 and <= '
                         f'{max_relative_margin:.2f} * vertical scene range={delta_height:.6g}')
    bottom_h = float(aligned_height.min() - bottom_margin)
    new_xyz = vertices[ring].copy()
    new_xyz[:,up_axis] = bottom_h * up_sign
    n_old = len(vertices)
    n_ring = len(ring)
    lower = np.arange(n_old, n_old+n_ring, dtype=np.int64)
    all_xyz = np.vstack([vertices,new_xyz])

    # Directed original boundary edge a->b must be reversed by the skirt's first face.
    skirt_faces = np.empty((2*n_ring,3),dtype=np.int64)
    for i in range(n_ring):
        a = int(ring[i]); b = int(ring[(i+1)%n_ring])
        a_down = int(lower[i]); b_down = int(lower[(i+1)%n_ring])
        skirt_faces[2*i]   = [b,a,a_down]
        skirt_faces[2*i+1] = [b,a_down,b_down]

    triangles = constrained_delaunay_triangles(poly)
    if not hasattr(triangles,'geoms') or not len(triangles.geoms):
        raise ValueError('Bottom triangulation returned no faces')
    # Map the exact projected ring coordinates to the new base vertices.
    lookup = {}
    for i,coord in enumerate(uv):
        key = tuple(np.round(coord,11))
        if key in lookup:
            raise ValueError('Repeated projected boundary vertices; potential skirt self-intersection')
        lookup[key] = int(lower[i])
    cap_faces = []
    for tri in triangles.geoms:
        coords = np.asarray(tri.exterior.coords)[:-1]
        if len(coords)!=3:
            raise ValueError('Constrained triangulation returned non-triangle')
        ids=[]
        for xy in coords:
            key=tuple(np.round(xy,11))
            if key not in lookup:
                raise ValueError('Bottom triangulation introduced an unexpected vertex: '+str(key))
            ids.append(lookup[key])
        cap_faces.append(ids)
    cap_faces=np.asarray(cap_faces,dtype=np.int64)
    # 2D area winding is not enough when up_sign=-1 or up_axis!=2.
    # Force cap triangle oriented opposite to the surface's outward up axis, then fix_normals.
    xyz_tri=all_xyz[cap_faces]
    normals=np.cross(xyz_tri[:,1]-xyz_tri[:,0],xyz_tri[:,2]-xyz_tri[:,0])
    dot=normals[:,up_axis]*up_sign
    if np.any(np.abs(dot)<1e-14):
        raise ValueError('Degenerate bottom triangle')
    flip=dot>0
    cap_faces[flip] = cap_faces[flip][:, [0,2,1]]

    all_faces=np.vstack([np.asarray(mesh.faces,dtype=np.int64),skirt_faces,cap_faces])
    sealed=trimesh.Trimesh(vertices=all_xyz,faces=all_faces,process=False)
    diag = {'already_watertight':False,**diagnostic, 'projected_area':projected_area,
            'vertical_axis':up_axis, 'up_sign':up_sign,
            'bottom_level_along_up_axis':bottom_h,
            'bottom_world_coordinate':bottom_h*up_sign,
            'skirt_face_count':int(len(skirt_faces)),
            'cap_face_count':int(len(cap_faces)),
            'mesh_before_vertices':int(len(mesh.vertices)),
            'mesh_before_faces':int(len(mesh.faces)),
            'skirt_type':'vertical projected edges; preserve original Poisson surface',
            'base_type':'constrained 2D triangulation, planar artificial cap'}
    # repair winding and fix orientation; doesn't modify vertex positions.
    sealed.fix_normals(multibody=True)
    diag['sealed']={'vertices':int(len(sealed.vertices)),'faces':int(len(sealed.faces)),
                   'watertight':bool(sealed.is_watertight),
                   'winding_consistent':bool(sealed.is_winding_consistent),
                   'is_volume':bool(sealed.is_volume),'body_count':int(sealed.body_count)}
    if not sealed.is_watertight or not sealed.is_winding_consistent or not sealed.is_volume:
        raise ValueError('Skirt/base topology failed closure check: '+str(diag['sealed']))
    extra = trimesh.Trimesh(vertices=all_xyz,faces=np.vstack([skirt_faces,cap_faces]),process=False)
    return sealed,diag,extra


