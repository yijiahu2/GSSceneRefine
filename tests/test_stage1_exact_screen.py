import numpy as np
import pytest

from gsrefine.roi.two_stage_sdf import SafetyConfig, safety_band_after_core


@pytest.mark.parametrize("k,trim", [(3, 0.), (11, .1), (128, .05), (327, .05)])
def test_bounds_match_exhaustive_with_outliers_and_ties(k, trim):
    rng = np.random.default_rng(475)
    xyz = rng.uniform(-2., 2., (327, 3))
    # Near and far candidates, duplicate centers and a single oversized axis.
    xyz[10:70, 0] = rng.uniform(.005, .35, 60)
    xyz[75:78] = xyz[2]
    axes = rng.uniform(.08, .25, len(xyz))
    axes[-1] = 1000.
    base = xyz[:, 0] <= 0.
    args = dict(positions=xyz, major_axes=axes, original_roi_sdf=lambda p: p[:, 0],
                geometric_support_radius=0., original_mask=base, geometric_core_mask=base,
                config=SafetyConfig(k_neighbors=k, trim_fraction=trim, safety_factor=1.4))
    exact = safety_band_after_core(**args)[0]
    fast, info, _ = safety_band_after_core(**args, screen_bounds=True)
    np.testing.assert_array_equal(fast, exact)
    assert info["early_keep"] + info["early_drop"] + info["exact_queries"] == (~base).sum()
    if k == 128:
        assert min(info["early_keep"], info["early_drop"], info["exact_queries"]) > 0
    assert info["safety_radius_p95_candidates"] is None
    _, full, debug = safety_band_after_core(**args, screen_bounds=True, collect_debug=True)
    assert full["exact_queries"] == len(debug["candidate_indices"]) == (~base).sum()
    assert full["safety_radius_p95_candidates"] > 0


def test_threshold_roundoff_falls_back_to_exact():
    cfg = SafetyConfig(k_neighbors=3, trim_fraction=0., safety_factor=2., distance_tolerance=1e-6)
    t = 2*.125 + cfg.distance_tolerance
    x = np.array([0., .01, np.nextafter(t, -np.inf), t, np.nextafter(t, np.inf), 1.])
    keep, info, _ = safety_band_after_core(
        positions=np.column_stack((x, x*0, x*0)), major_axes=np.full(len(x), .125),
        original_roi_sdf=lambda p:p[:, 0], geometric_support_radius=0.,
        original_mask=x==0., geometric_core_mask=x==0., config=cfg, screen_bounds=True)
    np.testing.assert_array_equal(keep, (x>0) & (x<=t))
    assert info["exact_queries"] == 3
