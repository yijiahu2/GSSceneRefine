"""Shared-edge piecewise-linear intersection of a triangle mesh and R3 scalar field.

This is the geometric clip already validated for the direct-R3 experiment.
It does not infer SfM density, reconstruct Poisson, or remove isolated faces.
"""
from __future__ import annotations

import numpy as np


def clip_piecewise_linear(mesh, signed, *, tolerance=1e-10):
    """Clip by interpolated vertex margin >= 0, splitting and sharing cut edges.

    Nonlinear R3 implicit boundaries are approximated linearly on each original
    edge; this helper must not be described as an exact analytical intersection.
    """
    import trimesh
    verts = np.asarray(mesh.vertices, np.float64)
    faces = np.asarray(mesh.faces, np.int64)
    s = np.asarray(signed, dtype=np.float64)
    if len(s) != len(verts) or not np.isfinite(s).all():
        raise ValueError('Invalid R3 signed scalar field for clipping')
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
        raise ValueError('R3 clipping retained no Poisson triangles')
    clipped = trimesh.Trimesh(vertices=np.asarray(out_vertices),
                              faces=np.asarray(new_faces, dtype=np.int64), process=False)
    clipped.remove_unreferenced_vertices()
    return clipped, {
        'source_triangles': int(len(faces)),
        'all_inside_triangles': int(all_in.sum()),
        'partly_clipped_triangles': int(len(mixed)),
        'outside_triangles': int((~any_in).sum()),
        'retained_triangles_after_cut': int(len(clipped.faces)),
        'shared_intersection_vertices': int(len(edge_cuts)),
        'clipping_method': 'shared-edge, piecewise-linear signed implicit ROI',
        'original_face_ids_provenance_count': len(original_face_ids),
        'source_face_ids': np.asarray(original_face_ids, dtype=np.int64),
    }
