"""Regression: production code uses region separation, original-main first."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_latest_sealless_region_gate_in_production():
    p=(ROOT/'gsrefine/cleanup/roi_barrier_parity.py').read_text()
    assert 'barrier_separation_gate import evaluate' in p
    assert "r['status'] != 'NO_TOP_TO_BOTTOM_PATH_SEEN_NOT_PROVEN_SEALED'" not in p
    assert 'stage2_geometry_validation' in p


def test_original_main_selected_before_clip_and_postclip_main_preserved():
    p=(ROOT/'gsrefine/geometry/r3_direct_poisson_clip.py').read_text()
    main=p.index('global_mesh, original_main_info = largest_original_component(global_mesh)')
    cut=p.index('clipped, clip_metadata = _clip_piecewise_linear(global_mesh, signed)')
    assert main < cut
    assert 'original_main_selection' in p
    downstream=(ROOT/'gsrefine/cleanup/direct_r3_pipeline.py').read_text()
    assert 'split_direct_r3_mesh(mesh)' in downstream


def test_no_top_bottom_flood_verdict_generated():
    s=(ROOT/'tools/audit_final_r3_global_poisson_barrier.py').read_text()
    assert "status='WINDOW_TRUNCATED' if touches else 'REGION_GRAPH_SAMPLED_NOT_CERTIFIED'" in s
    assert 'bottom_l=set(map(int,np.unique(labels[bottom])))' not in s


def test_onepass_preserves_sealless_and_ignores_old_preseal():
    s=(ROOT/'gsrefine/cleanup/stage2_onepass.py').read_text()
    assert 'stage2_geometry_validation' in s
    assert '_source_order_preseal' in s
    assert "GSSceneRefine.roi_barrier_parity_sdf.v3" in s
