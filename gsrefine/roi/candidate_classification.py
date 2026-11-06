"""Explainable geometric ROI candidate triage (not semantic recognition).

The *outside face-adjacency component* is the unit used to recognize expected
perimeter continuation.  One continuous base terrain ribbon can cross many
orthographic polygon edges.  Testing each individual edge against a near-closed
ring criterion wrongly labels that ribbon uncertain.

A normal component requires independent evidence of:
(1) multi-edge contact, (2) a material base/bottom attachment, (3) angular
spread, (4) sustained observed surface growth, and (5) enough supported area.
The resulting label is a hypothesis; high/top and terminating patches remain
protected from a background label even within a normal component.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def triage(*, boundary_class, height_quantile, angular_coverage,
           saturated, reaches_mesh_edge, external_area_ratio,
           baseline_coverage=0.45, high_height=.65, base_height=.30,
           max_suspect_area_ratio=.25):
    """Detect strong local truncations; defer normal exclusion to component evidence."""
    if (height_quantile <= base_height and angular_coverage >= baseline_coverage
            and not saturated):
        return 'uncertain', 'single_edge_base_pattern_requires_component_evidence'
    if (height_quantile >= high_height and boundary_class in ('top','side')
            and saturated and not reaches_mesh_edge
            and external_area_ratio <= max_suspect_area_ratio):
        return 'suspect', 'local_high_attachment+finite_surface_growth'
    return 'uncertain', 'insufficient_nonsemantic_evidence'


def component_background_evidence(groups, internal_area, *, study_cfg=None):
    """Summarize reliable exterior *mesh* components across polygon edges.

    `groups` records MUST be based on the same mesh-face connected-component
    index.  This never joins geometrically disconnected surfaces by Euclidean
    proximity and never classifies an entire component solely by large area.

    Returns component-index -> compact explainable diagnostics.  Thresholds are
    explicit experimental hypotheses and can be ablated, not universal truths.
    """
    cfg = study_cfg or {}
    min_area = float(cfg.get('normal_min_component_area_fraction', .02))
    min_azimuth = float(cfg.get('normal_min_contact_azimuth', .20))
    min_edges = int(cfg.get('normal_min_crossed_edges', 3))
    min_base_fraction = float(cfg.get('normal_min_base_support_fraction', .14))
    base_height = float(cfg.get('base_height_quantile', .30))
    high_height = float(cfg.get('high_height_quantile', .65))
    if not (0 <= min_area <= 1 and 0 <= min_azimuth <= 1 and
            0 <= min_base_fraction <= 1 and min_edges >= 2):
        raise ValueError('Invalid normal-continuation thresholds')
    by_component = defaultdict(list)
    for row in groups:
        by_component[int(row['component'])].append(row)
    out = {}
    for component, rows in by_component.items():
        total = sum(float(row['area']) for row in rows)
        edges = {(int(row['view_index']) if 'view_index' in row else str(row['view']),
                  int(row['edge'])) for row in rows}
        views = {str(row['view']) for row in rows}
        bottom_edges = {(str(row['view']), int(row['edge'])) for row in rows
                        if row['boundary_class'] == 'bottom'}
        base_area = sum(float(r['area']) for r in rows
                        if r['boundary_class'] == 'bottom'
                        or (r['boundary_class'] in ('side','plan_side')
                            and float(r['median_relative_height']) <= base_height + .07))
        top_area = sum(float(r['area']) for r in rows
                       if r['boundary_class'] == 'top'
                       or float(r['median_relative_height']) >= high_height)
        # Joint support across edges: the original single-edge angle test was
        # overly demanding. The aggregation comes from real face adjacency.
        azimuth = max((float(r.get('base_component_azimuth_coverage', 0))
                       for r in rows), default=0.0)
        measured = [r for r in rows if r.get('growth_evaluated',
                    all(x is not None for x in r.get('growth_curve_area', [])))]
        # Weight persistence by actual area, not number of arbitrary edge splits.
        continuing_area = sum(float(r['area']) for r in measured
                              if not bool(r['growth_saturated'])
                              and not bool(r['touches_open_mesh_boundary']))
        measured_area = sum(float(r['area']) for r in measured)
        persistent_fraction = continuing_area/max(measured_area,1e-12)
        mesh_edge_area = sum(float(r['area']) for r in rows
                             if bool(r['touches_open_mesh_boundary']))
        # Two distinct ways to support surrounding-base continuity:
        # (i) base perimeter across two different facets, or
        # (ii) one base facet with a cross-view continuation.
        multi_boundary_base = (len(bottom_edges) >= 2 or
            (len(bottom_edges) >= 1 and len(views) >= 2))
        evidence = {
            'component': component,
            'area': total,
            'area_to_internal_fraction': total/max(float(internal_area),1e-12),
            'n_groups': len(rows),
            'n_crossed_edges': len(edges),
            'n_views': len(views),
            'n_bottom_edges': len(bottom_edges),
            'base_supported_area_fraction': base_area/max(total,1e-12),
            'top_or_high_area_fraction': top_area/max(total,1e-12),
            'base_contact_azimuth_coverage': azimuth,
            'measured_growth_area_fraction': measured_area/max(total,1e-12),
            'continuing_growth_measured_fraction': persistent_fraction,
            'open_mesh_edge_area_fraction': mesh_edge_area/max(total,1e-12),
        }
        checks = {
            'sufficient_component_area': evidence['area_to_internal_fraction'] >= min_area,
            'multi_edge_contact': len(edges) >= min_edges,
            'base_cross_boundary_contact': multi_boundary_base,
            'base_area_support': evidence['base_supported_area_fraction'] >= min_base_fraction,
            'base_angular_spread': azimuth >= min_azimuth,
            'growth_well_observed': evidence['measured_growth_area_fraction'] >= .5,
            'mostly_continuing_growth': persistent_fraction >= .65,
            'not_limited_by_open_mesh_boundary': evidence['open_mesh_edge_area_fraction'] <= .2,
        }
        evidence['checks'] = checks
        evidence['expected_perimeter_continuation'] = all(checks.values())
        evidence['classification_reason'] = ('multi_edge_base_continuation+angular_spread+growth'
            if evidence['expected_perimeter_continuation']
            else 'incomplete_perimeter_evidence:' + ','.join(k for k, v in checks.items() if not v))
        out[component] = evidence
    return out


def apply_component_background_evidence(groups, components, *, high_height=.65):
    """Apply background evidence to non-top/non-terminating groups only.

    Preserve a locally suspected clipped top even if nearby background has
    the same exterior mesh component.  Manual overrides always take priority.
    """
    for row in groups:
        comp = components[int(row['component'])]
        row['background_component_candidate'] = bool(comp['expected_perimeter_continuation'])
        row['background_component_area_to_internal_fraction'] = comp['area_to_internal_fraction']
        row['background_component_crossed_edges'] = comp['n_crossed_edges']
        row['background_component_base_fraction'] = comp['base_supported_area_fraction']
        row['background_component_ring_coverage'] = comp['base_contact_azimuth_coverage']
        row['background_component_reason'] = comp['classification_reason']
        protect_high = (row['boundary_class'] == 'top'
                        or float(row['median_relative_height']) >= high_height
                        or bool(row['growth_saturated'])
                        or bool(row['touches_open_mesh_boundary']))
        if (comp['expected_perimeter_continuation']
                and row['classification'] == 'uncertain'
                and not protect_high):
            row['classification'] = 'normal'
            row['classification_reason'] = 'component_multi_edge_expected_perimeter_continuation'
        if not row['manual_override']:
            row['effective_label'] = row['classification']
    return groups
