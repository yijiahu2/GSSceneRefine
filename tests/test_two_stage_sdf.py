import numpy as np
import pytest
from scipy.spatial import cKDTree
from gsrefine.roi.two_stage_sdf import (
    SafetyConfig, CoreImplicitField, longest_gaussian_axis,
    trimmed_knn_mean, safety_band_after_core,
)


def sphere_roi(x):
    x = np.asarray(x)
    return np.linalg.norm(x, axis=1) - 1.0


def test_long_axis_is_largest_covariance_standard_deviation():
    log_s = np.log([[0.02, 0.11, 0.07], [0.30, 0.01, 0.09]])
    assert longest_gaussian_axis(log_s) == pytest.approx([.11, .30])


def test_knn_trim_five_percent_each_side():
    n = 40
    xyz = np.stack([np.arange(n), np.zeros(n), np.zeros(n)], axis=1).astype(float)
    vals = np.ones(n)
    vals[:2] = .01
    vals[-2:] = 1e3
    result = trimmed_knn_mean(cKDTree(xyz), vals, xyz[:1], k=n, trim_fraction=.05)
    assert result[0] == pytest.approx(1.)


def test_implicit_field_includes_disconnected_core_patch():
    field = CoreImplicitField(sphere_roi, np.array([[3, 0, 0]]), .2)
    assert field.evaluate([[0, 0, 0]])[0] < 0
    assert field.evaluate([[3, 0, 0]])[0] < 0
    assert field.evaluate([[3.3, 0, 0]])[0] == pytest.approx(.1)


def test_two_stages_are_additive_no_reclassification():
    positions = np.array([
        [0, 0, 0],      # R0
        [3, 0, 0],      # recovered core, island
        [3.65, 0, 0],   # safety shell outside core (0.45 from ball surface)
        [4.50, 0, 0],   # beyond shell (1.3 from ball surface)
        [0, 1.5, 0],    # safety shell on original ROI boundary (0.5)
        [0, 4.0, 0],    # beyond shell
    ], float)
    r0 = np.array([1, 0, 0, 0, 0, 0], bool)
    core = np.array([1, 1, 0, 0, 0, 0], bool)
    axes = np.full(len(positions), .6)
    config = SafetyConfig(k_neighbors=6, trim_fraction=.05,
                          projection_step=1e-4, projection_tolerance=1e-5,
                          batch_size=2)
    extra, report, debug = safety_band_after_core(
        positions=positions, major_axes=axes,
        original_roi_sdf=sphere_roi, original_mask=r0,
        geometric_core_mask=core, geometric_support_radius=.2,
        config=config, collect_debug=True)
    assert np.array_equal(extra, [0, 0, 1, 0, 1, 0])
    assert report['geometry_recovered_gaussians'] == 1
    assert report['safety_added_gaussians'] == 2
    assert not (extra & core).any()
    assert np.array_equal(core | extra, [1, 1, 1, 0, 1, 0])


def test_cannot_discard_original_roi_core():
    with pytest.raises(ValueError, match='Core recovery'):
        safety_band_after_core(
            positions=np.array([[0., 0, 0], [2, 0, 0], [3, 0, 0]]),
            major_axes=np.ones(3), original_roi_sdf=sphere_roi,
            original_mask=np.array([1, 0, 0], bool),
            geometric_core_mask=np.array([0, 1, 0], bool),
            geometric_support_radius=.1,
            config=SafetyConfig(k_neighbors=3))


def test_poisson_surface_sampler_has_bounded_continuous_coverage():
    import trimesh
    from gsrefine.roi.two_stage_sdf import poisson_surface_coverage
    tri = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                          faces=[[0, 1, 2]], process=False)
    pts, meta = poisson_surface_coverage(tri, np.array([True]), diag=2., max_samples=4000)
    assert pts.shape[1] == 3
    assert meta['surface_cover_error_bound'] > 0
    rng = np.random.default_rng(31)
    bary = rng.dirichlet([1, 1, 1], size=1000)
    points = bary @ tri.vertices
    distances = cKDTree(pts).query(points)[0]
    assert distances.max() <= meta['surface_cover_error_bound'] + 1e-12


def test_geometry_recovery_precedes_full_boundary_knn_band():
    import trimesh
    from gsrefine.roi.two_stage_sdf import recover_two_stage_from_poisson

    class SphereROI:
        def distances(self, points):
            points = np.asarray(points)
            return (1. - np.linalg.norm(points, axis=1))[:, None]

    class FakePly:
        def __init__(self, count):
            self.arr = np.empty(count, dtype=[('scale_0','f8'),('scale_1','f8'),('scale_2','f8')])
            for key in self.arr.dtype.names: self.arr[key] = np.log(.35)

    mesh = trimesh.Trimesh(
        vertices=np.array([[1.18, -.015, 0], [1.18, .015, 0], [1.18, 0, .015]]),
        faces=[[0, 1, 2]], process=False)
    xyz = np.array([[0.,0,0], [.95,0,0], [1.18,0,0],
                    [1.47,0,0], [0,1.2,0], [0,2.1,0], [2.5,0,0],
                    [0,-.8,0]], float)
    base = np.linalg.norm(xyz, axis=1) <= 1.
    core, safe, final, report = recover_two_stage_from_poisson(
        roi=SphereROI(), mesh=mesh, selected_faces=np.array([True]),
        gaussian_ply=FakePly(len(xyz)), positions=xyz, original_mask=base,
        diag=3.5, k_neighbors=8, batch_size=4)
    assert core[2]  # local subject geometrically reconstructed before band
    assert not core[3] and safe[3]  # further safety band around reconstructed subject
    assert safe[4]  # old ROI boundary ALSO receives safety
    assert not final[5] and not final[6]
    assert np.array_equal(final, core | safe)
    assert np.all(final[base])
    assert report['stage_A']['restored_core_gaussians'] >= 1
    assert report['stage_B']['safety_added_gaussians'] >= 2
