"""Guard API regression: roi-study needs 'budget_policy' in core.py."""
import numpy as np
from gsrefine.roi.core import add_guard_gaussians


def test_roi_study_guard_budget_api_review_vs_reject():
    xyz = np.array([[0., 0., 0.], [0.08, 0., 0.], [10., 0., 0.]])
    baseline = np.array([True, False, False])
    surface = np.array([[0., 0., 0.]])
    radii = np.array([0.15])
    provisional, report = add_guard_gaussians(
        xyz, baseline, surface, radii, max_added_fraction=0.1,
        budget_policy='review')
    assert provisional.tolist() == [False, True, False]
    assert report['review_required'] is True
    assert report['budget_rejected'] is False
    rejected, reject_report = add_guard_gaussians(
        xyz, baseline, surface, radii, max_added_fraction=0.1,
        budget_policy='reject')
    assert rejected.sum() == 0
    assert reject_report['budget_rejected'] is True
    assert baseline.tolist() == [True, False, False]
