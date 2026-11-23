"""Read-only, deterministic diagnostics of ORIGINAL Poisson/SfM surface support.

Neither geometry nor threshold decisions may depend on these reports.
"""
import numpy as np
from scipy.spatial import cKDTree


def diagnose(mesh, points, *, sfm_spacing=None, sample_faces=20000):
    pts = np.asarray(points, dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(pts) < 3 or not len(faces) or not np.isfinite(pts).all():
        raise ValueError('Invalid SfM/Poisson diagnostic input')
    chosen = np.linspace(0, len(faces)-1, min(sample_faces, len(faces)), dtype=np.int64)
    centers = vertices[faces[chosen]].mean(axis=1)
    tree = cKDTree(pts)
    dist = tree.query(centers, k=1, workers=-1)[0]
    if sfm_spacing is None or not np.isfinite(sfm_spacing) or sfm_spacing <= 0:
        q = pts[np.linspace(0, len(pts)-1, min(len(pts), 12000), dtype=np.int64)]
        sfm_spacing = float(np.median(tree.query(q, k=2, workers=-1)[0][:, 1]))
    slo, shi = pts.min(axis=0), pts.max(axis=0)
    mlo, mhi = vertices.min(axis=0), vertices.max(axis=0)
    sfm_extent = np.maximum(shi - slo, 1e-12)
    mesh_extent = mhi - mlo
    far = dist > 5.0 * max(float(sfm_spacing), 1e-12)
    outside_box = np.any((centers < slo) | (centers > shi), axis=1)
    return dict(
        status='REVIEW_ONLY_NOT_GROUND_TRUTH', sampled_faces=int(len(chosen)),
        total_faces=int(len(faces)), closest_sfm_p50=float(np.median(dist)),
        closest_sfm_p95=float(np.quantile(dist, .95)),
        sfm_spacing_reference=float(sfm_spacing),
        far_from_sfm_sample_fraction=float(far.mean()),
        sampled_faces_outside_sfm_bbox_fraction=float(outside_box.mean()),
        poisson_to_sfm_extent_ratios=(mesh_extent/sfm_extent).tolist(),
        poisson_bounds=[mlo.tolist(), mhi.tolist()],
        sfm_bounds=[slo.tolist(), shi.tolist()],
        almost_cubic_mesh_extent=bool(np.ptp(mesh_extent) < .005*max(mesh_extent.max(), 1e-12)),
        review_required=bool(far.mean() > .05),
        note='Sparse SfM absence is not proof of spurious geometry; no faces are removed.')
