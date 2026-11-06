"""ROI/Poisson boundary crossing diagnostics in the native COLMAP frame.

A crossing is assigned to a polygon edge through its projected outward normal.
The intersection geometry is sampled from mesh-edge / polygon-edge crossings;
this is a trace approximation, not exact Boolean triangle clipping. It does not
assume the camera-world z axis points up, and never creates artificial caps.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def _edge_normals(polygon):
    p = np.asarray(polygon, dtype=np.float64)
    edges = np.roll(p, -1, axis=0) - p
    norm = np.linalg.norm(edges, axis=1)
    if np.any(norm < 1e-10):
        raise ValueError('Degenerate polygon edge')
    area2 = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    outward = np.column_stack([edges[:, 1], -edges[:, 0]]) if area2 > 0 else np.column_stack([-edges[:, 1], edges[:, 0]])
    return outward / norm[:, None], edges


def classify_view_edges(roi, up):
    """Per-view polygon edge definitions and outward world normals."""
    up = np.asarray(up, np.float64)
    up /= np.linalg.norm(up)
    out = []
    for vi, v in enumerate(roi.views):
        normals, _ = _edge_normals(v.polygon)
        world = normals @ v.U
        for ei, n in enumerate(world):
            vertical = float(np.dot(n, up))
            name = 'top' if vertical >= .5 else 'bottom' if vertical <= -.5 else 'side'
            # View index 0 does not intrinsically equal top view. Determine
            # whether camera projection plane is perpendicular to gravity.
            view_normal = np.cross(v.U[0], v.U[1])
            if abs(float(np.dot(view_normal, up))) >= .92:
                name = 'plan_side'
            out.append({'view_index': vi, 'view_name': v.name,
                        'edge_index': ei, 'boundary_class': name,
                        'outward': n.tolist(), 'vertical_dot_up': vertical})
    return out


def vertex_crossings(mesh, roi, credible, up, max_points=220000):
    """Find edge-crossing trace samples and associate with actual ROI polygon edges.

    Complexity: mesh edges x views, evaluates ONLY edges with differing ROI
    membership. Crosses for which endpoints are both outside are not recovered,
    therefore this trace is explicitly approximate and audited as such.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    score = roi.distances(verts)
    inside = (score >= -1e-9).all(axis=1)
    crosses = np.any(inside[faces] != np.roll(inside[faces], -1, axis=1), axis=1)
    face_ids = np.flatnonzero(crosses & credible)
    if len(face_ids) > max_points:
        face_ids = face_ids[np.linspace(0, len(face_ids)-1, max_points).astype(int)]
    polygon_edges = classify_view_edges(roi, up)
    samples, labels, parent = [], [], []
    # Construct edge -> view-polygon intersection where a segment actually
    # crosses a projected polygon edge. Exact for segment-segment intersection.
    for fid in face_ids:
        tri = faces[fid]
        for k in range(3):
            aidx, bidx = tri[k], tri[(k+1)%3]
            if inside[aidx] == inside[bidx]:
                continue
            a, b = verts[aidx], verts[bidx]
            for vi, view in enumerate(roi.views):
                a2, b2 = view.plane(np.stack([a,b]))
                vseg = b2 - a2
                for ei in range(len(view.polygon)):
                    p = view.polygon[ei]
                    d = view.polygon[(ei+1)%len(view.polygon)]-p
                    den = vseg[0]*d[1]-vseg[1]*d[0]
                    if abs(den) < 1e-13:continue
                    diff = p - a2
                    t = (diff[0]*d[1]-diff[1]*d[0])/den
                    u = (diff[0]*vseg[1]-diff[1]*vseg[0])/den
                    if not (-1e-9 <= t <= 1+1e-9 and -1e-9 <= u <= 1+1e-9):continue
                    point = a + np.clip(t, 0, 1)*(b-a)
                    # Respect every OTHER projection constraint at the point.
                    if any(v2.signed_and_normal(point[None, :])[0][0] < -1e-6
                           for jj,v2 in enumerate(roi.views) if jj != vi):continue
                    samples.append(point)
                    labels.append((vi,ei))
                    parent.append(fid)
    return (np.asarray(samples,dtype=np.float64).reshape(-1,3),
            np.asarray(labels,dtype=np.int32).reshape(-1,2),
            np.asarray(parent,dtype=np.int32), polygon_edges)


def assign_candidate_boundaries(centers, candidate_ids, traces, trace_labels, roi, up):
    """Map external faces to nearest observed boundary crossing, or local edge.

    Nearest spatial crossing is a heuristic. Each candidate also records its
    actual violated view constraints; the mapping must be reviewed near corners.
    """
    xyz = np.asarray(centers)[candidate_ids]
    if not len(xyz): return np.empty((0,2),np.int32), np.empty(0), np.empty(0)
    if len(traces):
        distances, idx = cKDTree(traces).query(xyz, workers=-1)
        labels = trace_labels[idx].copy()
    else:
        distances = np.full(len(xyz),np.inf)
        labels = np.empty((len(xyz),2),dtype=np.int32)
        for ci, pt in enumerate(xyz):
            closest = (np.inf,0,0)
            for vi,v in enumerate(roi.views):
                point2 = v.plane(pt[None,:])[0]
                for ei in range(len(v.polygon)):
                    a = v.polygon[ei]; b=v.polygon[(ei+1)%len(v.polygon)]
                    edge = b-a
                    t=np.clip(np.dot(point2-a,edge)/(np.dot(edge,edge)+1e-30),0,1)
                    d=np.linalg.norm(point2-a-t*edge)
                    if d<closest[0]:closest=(d,vi,ei)
            labels[ci] = closest[1:]
    sd = roi.distances(xyz)
    violation = np.maximum(0, -sd).max(axis=1)
    return labels, distances, violation


def angular_coverage(points, up, origin, bins=36):
    if len(points)==0:return 0.0
    up=np.asarray(up,float);up/=np.linalg.norm(up)
    reference=np.eye(3)[np.argmin(abs(up))]
    e1=reference-up*np.dot(reference,up);e1/=np.linalg.norm(e1)
    e2=np.cross(up,e1)
    q=np.asarray(points)-np.asarray(origin)
    t=np.mod(np.arctan2(q@e2,q@e1),2*np.pi)
    occupied=np.unique(np.floor(t/(2*np.pi)*bins).astype(int).clip(0,bins-1))
    return float(len(occupied)/bins)
