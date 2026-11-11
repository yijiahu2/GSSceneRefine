"""Two-stage Gaussian ROI recovery: independent geometry core and adaptive safety band.

This file is additive and intentionally has no dependency on GSSceneRefine internals.
The caller supplies a sign-correct continuous implicit field for the original ROI
(negative inside), plus a mask from an independently completed geometric
recovery (stage A). The intersection-of-polygons field is a conservative
exterior distance lower bound, not an exact Euclidean SDF at polygon corners.

The recovered-core support is represented as a union of small balls. Their radius
is a geometric discretization/support tolerance, NOT the safety-band width.

For outside points, min(SDF_ROI, SDF_balls) is the exact distance to the union
only when SDF_ROI is metric. The orthographic polygon ROI proxy is conservative
outside, so extra Gaussian centers can be retained near corners but legitimate
nearby Gaussian centers are not incorrectly excluded by the ROI distance bound.
No signed-distance voxel grid or watertight output mesh is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SafetyConfig:
    k_neighbors: int = 128
    trim_fraction: float = 0.05
    safety_factor: float = 1.0
    # Only numerical tolerance, not a user-defined geometric safety distance.
    distance_tolerance: float = 1e-6
    batch_size: int = 8192
    projection_iterations: int = 8
    projection_step: float = 1e-4
    projection_tolerance: float = 1e-4

    def validate(self, n_gaussians: int):
        if not 3 <= self.k_neighbors <= n_gaussians:
            raise ValueError(f"k_neighbors must be in [3,{n_gaussians}]")
        if not 0 <= self.trim_fraction < .5:
            raise ValueError("trim_fraction must be in [0,0.5)")
        if self.safety_factor <= 0 or not np.isfinite(self.safety_factor):
            raise ValueError("safety_factor must be finite and positive")
        if self.distance_tolerance < 0 or self.batch_size < 1:
            raise ValueError("Invalid tolerance or batch size")
        if self.projection_step <= 0 or self.projection_tolerance <= 0:
            raise ValueError("Projection parameters must be positive")


def longest_gaussian_axis(scale_columns: np.ndarray, *, logarithmic: bool = True) -> np.ndarray:
    """Compute sqrt(lambda_max(Sigma)) exactly for Gaussian scale principal axes.

    `scale_columns` has shape (N,3); standard 3DGS stores *log* standard
    deviations as scale_0/1/2. Rotation is unnecessary for lambda_max.
    """
    a = np.asarray(scale_columns, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError("Expected N x 3 finite scale columns")
    s = np.exp(a) if logarithmic else a
    if not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError("Gaussian scales must be positive finite numbers")
    return np.max(s, axis=1)


def trimmed_knn_mean(tree: cKDTree, lengths: np.ndarray, query: np.ndarray, *,
                     k: int, trim_fraction: float, batch_size: int = 8192) -> np.ndarray:
    """Average the central 90% of the k nearest Gaussian major-axis lengths.

    Includes *all* input Gaussians, not only those inside the ROI or judged normal.
    Uses each query's exactly k neighbors and discards floor(0.05*k) at each tail.
    """
    p = np.asarray(query, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("query must be N x 3 finite positions")
    a = np.asarray(lengths, dtype=np.float64)
    if len(a) != tree.n or (a <= 0).any() or not np.isfinite(a).all():
        raise ValueError("lengths must contain one positive finite value per Gaussian")
    if not 3 <= k <= tree.n or not 0 <= trim_fraction < .5:
        raise ValueError("Invalid k or trim fraction")
    # Exact 5% L-trimming with fractional endpoint weights for k not divisible
    # by 20 (e.g. k=128 trims 6.4 observations from each side).
    cut = int(np.floor(k * trim_fraction))
    fraction = k * trim_fraction - cut
    if 2 * (cut + fraction) >= k:
        raise ValueError("Trim fraction discards every sample")
    ans = np.empty(len(p), dtype=np.float64)
    for j in range(0, len(p), batch_size):
        s = slice(j, min(j + batch_size, len(p)))
        _, idx = tree.query(p[s], k=k, workers=-1)
        vals = np.sort(a[np.asarray(idx).reshape(-1, k)], axis=1)
        weighted_sum = vals[:, cut:k - cut].sum(axis=1)
        if fraction:
            weighted_sum -= fraction * (vals[:, cut] + vals[:, k-cut-1])
        ans[s] = weighted_sum / (k * (1.0 - 2.0 * trim_fraction))
    return ans


class CoreImplicitField:
    """Implicit repaired core = original ROI solid UNION recovered-core samples.

    The added point balls use a common *geometric support* radius to turn
    discrete recovered samples into a continuous core. This is distinct from
    the user-controlled safety band and should be calibrated from mesh/sampling
    discretization, not Gaussian axis lengths.
    """

    def __init__(self, original_roi_sdf: Callable[[np.ndarray], np.ndarray],
                 recovered_points: np.ndarray, geometric_support_radius: float):
        self.roi_sdf = original_roi_sdf
        self.recovered_points = np.asarray(recovered_points, dtype=np.float64).reshape(-1, 3)
        if not np.isfinite(self.recovered_points).all():
            raise ValueError("Invalid geometric recovery centers")
        if not np.isfinite(geometric_support_radius) or geometric_support_radius < 0:
            raise ValueError("Invalid geometric support radius")
        self.radius = float(geometric_support_radius)
        self.tree = cKDTree(self.recovered_points) if len(self.recovered_points) else None

    def evaluate(self, x: np.ndarray, *, return_nearest: bool = False):
        p = np.asarray(x, dtype=np.float64).reshape(-1, 3)
        roi_d = np.asarray(self.roi_sdf(p), dtype=np.float64).reshape(-1)
        if len(roi_d) != len(p) or not np.isfinite(roi_d).all():
            raise ValueError("ROI SDF must return one finite metric distance per point")
        if self.tree is None:
            return (roi_d, np.full(len(p), -1, dtype=np.int64), np.zeros(len(p), bool)) if return_nearest else roi_d
        dist, idx = self.tree.query(p, k=1, workers=-1)
        # Signed distance to a ball (negative inside). Evaluating nearest center
        # is exact for balls with the same support radius.
        balls = dist - self.radius
        from_ball = balls < roi_d
        sdf = np.minimum(roi_d, balls)
        return (sdf, idx, from_ball) if return_nearest else sdf


def _project_to_boundary(field: CoreImplicitField, x: np.ndarray,
                         cfg: SafetyConfig) -> tuple[np.ndarray, np.ndarray]:
    """Find core SDF zero-level near each point by guarded Newton projection.

    Projection affects only WHERE kNN axes are sampled. Actual inclusion uses
    the original query point's direct continuous implicit-field distance.
    Unresolved projections return their candidate point and are counted.
    """
    p = np.asarray(x, dtype=np.float64).copy()
    active = np.ones(len(p), dtype=bool)
    h = cfg.projection_step
    for _ in range(cfg.projection_iterations):
        if not active.any():
            break
        a = np.flatnonzero(active)
        z = field.evaluate(p[a])
        good = np.abs(z) <= cfg.projection_tolerance
        active[a[good]] = False
        a = a[~good]
        if len(a) == 0:
            continue
        z = z[~good]
        grad = np.empty((len(a), 3), dtype=np.float64)
        for d in range(3):
            step = np.zeros(3); step[d] = h
            grad[:, d] = (field.evaluate(p[a] + step) - field.evaluate(p[a] - step)) / (2 * h)
        norms = np.sum(grad * grad, axis=1)
        ok = norms > 1e-14
        if not np.any(ok):
            break
        # Clip Newton steps; avoid runaway near non-differentiable junctions.
        shift = z[ok, None] * grad[ok] / norms[ok, None]
        s_len = np.linalg.norm(shift, axis=1)
        shift *= np.minimum(1.0, np.maximum(np.abs(z[ok]) * 2, h) / np.maximum(s_len, 1e-15))[:, None]
        p[a[ok]] -= shift
    residual = np.abs(field.evaluate(p))
    return p, residual


def safety_band_after_core(*, positions: np.ndarray, major_axes: np.ndarray,
                           original_roi_sdf: Callable[[np.ndarray], np.ndarray],
                           original_mask: np.ndarray, geometric_core_mask: np.ndarray,
                           geometric_support_radius: float, config: SafetyConfig = SafetyConfig(),
                           collect_debug: bool = False,
                           core_field: Optional[CoreImplicitField] = None) -> tuple[np.ndarray, dict, Optional[dict]]:
    """Stage B: independently dilate the *already recovered* core.

    Returns ``safety_added_mask`` only; the caller must retain core points:
      final_mask = geometric_core_mask | safety_added_mask.

    All valid source Gaussians, including suspected outliers, contribute to kNN
    statistics, and **all** Gaussians whose centers lie within the band are kept.

    A metric ROI SDF is preferred. A conservative exterior lower bound is also
    admissible when boundary-corner over-inclusion is explicitly reported, as
    done by recover_two_stage_from_poisson().
    """
    xyz = np.asarray(positions, dtype=np.float64)
    axes = np.asarray(major_axes, dtype=np.float64)
    r0 = np.asarray(original_mask, dtype=bool)
    core = np.asarray(geometric_core_mask, dtype=bool)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Invalid Gaussian centers")
    n = len(xyz)
    if axes.shape != (n,) or not np.isfinite(axes).all() or np.any(axes <= 0):
        raise ValueError("One positive Gaussian major-axis sigma per center is required")
    if r0.shape != (n,) or core.shape != (n,) or np.any(r0 & ~core):
        raise ValueError("Core recovery must contain every original ROI Gaussian")
    config.validate(n)
    added = xyz[core & ~r0]
    field = core_field if core_field is not None else CoreImplicitField(
        original_roi_sdf, added, geometric_support_radius)
    all_tree = cKDTree(xyz)
    safety = np.zeros(n, dtype=bool)
    candidates = np.flatnonzero(~core)
    dbg = {"candidate_indices": [], "core_sdf": [], "local_trimmed_axis_mean": [],
           "projected_boundary_residual": []} if collect_debug else None
    n_projection_failed = 0
    radius_sum = 0.0
    radius_min = np.inf
    radius_max = 0.0
    total_added = 0
    all_radius = np.empty(len(candidates), dtype=np.float64)
    added_radius = []
    for start in range(0, len(candidates), config.batch_size):
        ids = candidates[start:start + config.batch_size]
        pts = xyz[ids]
        distances = field.evaluate(pts)
        # Only points outside the core should ever be in the safety set.
        if np.any(distances < -config.distance_tolerance):
            raise ValueError("Geometric core mask and core SDF disagree: core not closed")
        boundary_pts, residual = _project_to_boundary(field, pts, config)
        # Use all Gaussian scales, including abnormal Gaussians.
        local = trimmed_knn_mean(all_tree, axes, boundary_pts,
                                 k=config.k_neighbors,
                                 trim_fraction=config.trim_fraction,
                                 batch_size=config.batch_size)
        radius = local * config.safety_factor
        all_radius[start:start + len(ids)] = radius
        # Explicit numerical distance tolerance, not a user safety parameter.
        ok = distances <= radius + config.distance_tolerance
        safety[ids[ok]] = True
        total_added += int(ok.sum())
        if ok.any(): added_radius.append(radius[ok])
        n_projection_failed += int(np.count_nonzero(residual > config.projection_tolerance))
        radius_sum += float(radius.sum())
        if len(radius):
            radius_min = min(radius_min, float(radius.min()))
            radius_max = max(radius_max, float(radius.max()))
        if dbg is not None:
            dbg["candidate_indices"].append(ids)
            dbg["core_sdf"].append(distances)
            dbg["local_trimmed_axis_mean"].append(local)
            dbg["projected_boundary_residual"].append(residual)
    retained_radii = np.concatenate(added_radius) if added_radius else np.empty(0)
    report = {
        "scheme": "two_stage_repaired_core_then_adaptive_continuous_SDF_band",
        "input_gaussians": int(n),
        "r0_gaussians": int(r0.sum()),
        "geometry_recovered_gaussians": int(np.count_nonzero(core & ~r0)),
        "core_gaussians": int(core.sum()),
        "safety_added_gaussians": int(total_added),
        "final_gaussians": int((core | safety).sum()),
        "k_neighbors": int(config.k_neighbors),
        "trim_fraction_each_tail": float(config.trim_fraction),
        "safety_factor": float(config.safety_factor),
        "geometric_support_radius": float(geometric_support_radius),
        "projection_failed_count": int(n_projection_failed),
        "boundary_projection_tolerance": float(config.projection_tolerance),
        "safety_radius_mean_all_excluded_candidates": radius_sum / len(candidates) if len(candidates) else None,
        "safety_radius_min": radius_min if len(candidates) else None,
        "safety_radius_max": radius_max if len(candidates) else None,
        "safety_radius_p50_candidates": float(np.median(all_radius)) if len(candidates) else None,
        "safety_radius_p95_candidates": float(np.quantile(all_radius, 0.95)) if len(candidates) else None,
        "safety_radius_p50_added": float(np.median(retained_radii)) if len(retained_radii) else None,
        "safety_radius_p95_added": float(np.quantile(retained_radii, 0.95)) if len(retained_radii) else None,
        "roi_sdf_kind": "continuous_sign_correct_field_metric_or_conservative_lower_bound",
        "distance_evaluated_continuously_at_every_excluded_gaussian_center": True,
    }
    if dbg is not None:
        dbg = {key: np.concatenate(value) if value else np.empty(0)
               for key, value in dbg.items()}
    return safety, report, dbg


def roi_polygon_implicit_field(roi):
    """Analytic implicit field for an intersection of orthographic polygon prisms.

    Negative denotes inside. Inside distance to the union boundary is metric;
    outside, max of projection violations is a *conservative lower bound* on
    true Euclidean distance to the intersection (not a metric exterior SDF).
    This deliberate over-inclusion is reported rather than claiming exactness.
    All polygon edges are evaluated continuously: no voxel sampling error.
    """
    def evaluate(points):
        d = roi.distances(np.asarray(points, dtype=np.float64).reshape(-1, 3))
        return -d.min(axis=1)
    return evaluate


def poisson_surface_coverage(mesh, selected_faces: np.ndarray, diag: float,
                             max_samples: int = 400000):
    """Create a deterministic triangle-surface epsilon-net and error bound.

    Every triangle is barycentrically subdivided so the maximum edge of each
    subtriangle is <= `spacing`. Thus every continuous point of each selected
    Poisson triangle lies within `spacing` of a sample vertex. The bound is
    analytical (independent of random sampling and face ordering).
    """
    selected_faces = np.asarray(selected_faces, dtype=bool)
    if selected_faces.shape != (len(mesh.faces),):
        raise ValueError('Selected Poisson face mask size mismatch')
    if not selected_faces.any():
        return np.empty((0, 3), dtype=np.float64), {
            'surface_samples': 0, 'surface_spacing': 0.0,
            'surface_cover_error_bound': 0.0, 'selected_faces': 0}
    triangles = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces)[selected_faces]]
    e = np.maximum.reduce([np.linalg.norm(triangles[:, 1]-triangles[:, 0], axis=1),
                           np.linalg.norm(triangles[:, 2]-triangles[:, 1], axis=1),
                           np.linalg.norm(triangles[:, 0]-triangles[:, 2], axis=1)])
    base = max(float(np.median(e)), 0.0002 * float(diag), 1e-8)
    spacing = base
    for _ in range(40):
        parts = np.maximum(1, np.ceil(e / spacing)).astype(np.int64)
        n = int(np.sum((parts + 1) * (parts + 2) // 2))
        if n <= max_samples:
            break
        spacing *= max(1.05, (n / max_samples) ** 0.5 * 1.02)
    else:
        raise RuntimeError('Poisson surface sampling could not meet memory budget')
    # Guarantee recorded error bound, not just median mesh edge length.
    cover = float(np.max(e / parts))
    result = np.empty((n, 3), dtype=np.float64)
    at = 0
    for t, m in zip(triangles, parts):
        m = int(m)
        # using combinations: u >= 0, v >= 0 and u+v <= m
        # generate via triangular indexing where u and v grow independently.
        u = np.repeat(np.arange(m + 1), np.arange(m + 1, 0, -1))
        v = np.concatenate([np.arange(m-i+1) for i in range(m+1)])
        count = len(u)
        points = t[0] + (u[:, None] / m) * (t[1]-t[0]) + (v[:, None] / m) * (t[2]-t[0])
        result[at:at+count] = points
        at += count
    assert at == n
    return result, {
        'surface_samples': n, 'surface_spacing': spacing,
        'surface_cover_error_bound': cover, 'selected_faces': int(selected_faces.sum()),
        'sampling': 'deterministic barycentric epsilon-net'}


def estimate_geometric_support_radius(roi, xyz, base_mask, surface_samples,
                                      sample_error_bound, diag, batch_size=50000):
    """Geometry-only support from local Poisson-to-Gaussian center density.

    Stage A must not use 3DGS covariance, rendering scales, opacity, or the
    stage-B adaptive safety distance. A finite-width surface tube is needed
    because Poisson itself is a 2-D surface, not a solid Gaussian selector.
    Use P90 of surface-to-k-th Gaussian-center distance (k<=8). Samples are
    spatially distributed over the accepted surface, not just the R0 edge.
    """
    floor = max(2 * sample_error_bound, 1e-6 * diag)
    if not len(surface_samples):
        return 0.0, {'calibration_points': 0, 'calibration': 'no_selected_patches'}
    tree = cKDTree(np.asarray(xyz, dtype=np.float64))
    k = min(8, max(1, int(np.sqrt(len(xyz)))))
    if len(surface_samples) > 50000:
        # Deterministic spatial coverage of sampled Poisson surface.
        idx = np.linspace(0, len(surface_samples)-1, 50000).astype(np.int64)
        reference = surface_samples[idx]
    else:
        reference = surface_samples
    kth = []
    for start in range(0, len(reference), batch_size):
        distances, _ = tree.query(reference[start:start+batch_size], k=k, workers=-1)
        d = np.asarray(distances)
        kth.append(d if k == 1 else d[:, -1])
    values = np.concatenate(kth)
    spatial = float(np.quantile(values, .90))
    radius = max(floor, spatial)
    return radius, {'calibration_points': int(len(values)),
                    'calibration': 'P90_poisson_to_8th_nearest_gaussian_center',
                    'calibration_gaussian_neighbors': int(k),
                    'calibration_P90': spatial,
                    'geometric_discretization_floor': floor}


def recover_two_stage_from_poisson(*, roi, mesh, selected_faces, gaussian_ply, positions,
                                   original_mask, diag, k_neighbors=128,
                                   trim_fraction=.05, safety_factor=1.0,
                                   batch_size=8192, progress_callback=None, stage2_handoff=None):
    """Stage A: reconstruct core from accepted Poisson triangles; stage B: full boundary band.

    Uses no Poisson confidence scoring, no semantic filtering, and no SDF-based
    internal artifact removal. The implicit field represents the original ROI
    intersection plus the selected surface's geometrical support neighborhood.
    """
    xyz = np.asarray(positions, dtype=np.float64)
    base = np.asarray(original_mask, dtype=bool)
    sdf = roi_polygon_implicit_field(roi)
    surface, sample_info = poisson_surface_coverage(mesh, selected_faces, diag)
    radius, radius_info = estimate_geometric_support_radius(
        roi, xyz, base, surface, sample_info['surface_cover_error_bound'], diag)
    # Increase geometric radius by the worst-case sample covering error to
    # guarantee no loss caused solely by triangle discretization.
    support = radius + sample_info['surface_cover_error_bound'] if len(surface) else 0.0
    field = CoreImplicitField(sdf, surface, support)
    core = np.zeros(len(xyz), dtype=bool)
    for i in range(0, len(xyz), batch_size):
        j = min(i + batch_size, len(xyz))
        core[i:j] = field.evaluate(xyz[i:j]) <= 1e-9
    core |= base
    scale_columns = np.column_stack([gaussian_ply.arr[f'scale_{j}'] for j in range(3)])
    axes = longest_gaussian_axis(scale_columns, logarithmic=True)
    cfg = SafetyConfig(k_neighbors=min(k_neighbors, len(xyz)),
                       trim_fraction=trim_fraction,
                       safety_factor=safety_factor,
                       batch_size=batch_size,
                       projection_step=max(sample_info['surface_cover_error_bound'] / 5.0,
                                           diag * 1e-6, 1e-6),
                       projection_tolerance=max(diag * 1e-5, 1e-6))
    if progress_callback: progress_callback('safety', 'K近邻双侧5%截尾长轴与恢复后全边界隐式安全带')
    added, safety_info, _ = safety_band_after_core(
        positions=xyz, major_axes=axes, original_roi_sdf=sdf,
        original_mask=base, geometric_core_mask=core,
        geometric_support_radius=support, config=cfg,
        core_field=field)
    final = core | added
    if np.any(base & ~final) or np.any(core & added):
        raise AssertionError('Invalid two-stage mask containment')
    if int(final.sum()) != int(base.sum()) + int(np.count_nonzero(core & ~base)) + int(added.sum()):
        raise AssertionError('R3 final mask is not the union of two sequential additions')
    # Query the same original post-restoration R3 field at CLEAN SfM positions.
    # This saves geometry/radii only; it never changes the Gaussian R3 mask.
    if stage2_handoff is not None:
        from gsrefine.geometry.r3_sfm_bridge import save_r3_sfm_handoff
        save_r3_sfm_handoff(
            work_dir=stage2_handoff['work_dir'],
            clean_sfm_npz=stage2_handoff['clean_sfm_npz'],
            roi_json=stage2_handoff['roi_json'],
            gaussian_ply=stage2_handoff['gaussian_ply'],
            poisson_mesh=stage2_handoff['poisson_mesh'],
            core_field=field, safety_config=cfg,
            gaussian_positions=xyz, major_axes=axes)
    info = {'method': 'R3', 'algorithm': 'two_stage_poisson_core_then_knn_trimmed_implicit_band',
            'stage_A': {'core_gaussians': int(core.sum()),
                        'restored_core_gaussians': int(np.count_nonzero(core & ~base)),
                        'geometry_support_radius': radius,
                        'surface_sample_support_radius': support,
                        **radius_info, **sample_info},
            'stage_B': safety_info,
        'full_boundary_band_evaluated': True,
            'stage_B_uses_all_gaussians_in_knn': True,
            'stage_B_retains_all_gaussians_inside_band': True,
            'roi_exterior_distance': 'conservative_lower_bound_from_polygon_projections',
            'roi_exterior_distance_is_exact_metric': False,
            'surface_discretization_error_bound': sample_info['surface_cover_error_bound'],
            'inference_limitations': [
                'Poisson surface does not define a filled solid: stage-A support is a calibrated tube.',
                'At corners of projected-polygon intersections, exterior implicit distance is conservative; can over-include.',
                'Zero-level of recovered Poisson support is an approximation, not a certified semantic boundary.'
            ]}
    return core, added, final, info
