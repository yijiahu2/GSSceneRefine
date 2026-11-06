"""Compatibility contract for roi-study and component-aware background analysis."""

def test_study_import_and_component_helpers():
    from gsrefine.roi.study import run as study_run
    from gsrefine.roi.candidate_classification import (
        triage,
        component_background_evidence,
        apply_component_background_evidence,
    )
    assert callable(study_run)
    assert callable(triage)
    assert callable(component_background_evidence)
    assert callable(apply_component_background_evidence)


def test_single_edge_is_not_automatically_full_background():
    from gsrefine.roi.candidate_classification import component_background_evidence
    one = [{
        'component': 7, 'view': 'facade_0deg', 'edge': 2,
        'boundary_class': 'bottom', 'area': 0.2,
        'median_relative_height': 0.1,
        'base_component_azimuth_coverage': 0.5,
        'growth_evaluated': True, 'growth_saturated': False,
        'touches_open_mesh_boundary': False,
    }]
    result = component_background_evidence(one, 1.0)
    assert result[7]['expected_perimeter_continuation'] is False
    assert not result[7]['checks']['multi_edge_contact']
