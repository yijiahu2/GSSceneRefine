"""No-Open3D regression for the canonical V3 Stage-2 numeric sampling box."""
from __future__ import annotations

import numpy as np
import pytest

from tools.audit_final_r3_global_poisson_barrier import (
    _r3_shell_probe, complete_window_for_levels, grid_spec,
)


def box_margin(p):
    p=np.asarray(p,float)
    return np.min(np.concatenate((p+1.,1.-p),axis=1),axis=1)


def test_no_touch_means_identical_box_and_no_extra_expansion():
    levels=[('coarse',.4,20000),('fine',.2,80000)]
    low=np.array([-2.,-2.,-2.]);high=np.array([2.,2.,2.])
    a,b,report=complete_window_for_levels(box_margin,low,high,levels=levels)
    np.testing.assert_array_equal(a,low)
    np.testing.assert_array_equal(b,high)
    assert report['status']=='COMPLETE_SAMPLED_WINDOW'
    assert report['expanded'] is False
    assert report['expansion_rounds']==0
    assert all(x['r3_boundary_touch_count']==0 for x in report['levels'].values())


def test_fine_only_touch_triggers_same_box_for_both_grids():
    levels=[('coarse',.5,35000),('fine',.16,100000)]
    low=np.array([-.88,-1.4,-1.4]); high=np.array([1.4,1.4,1.4])
    initial=[_r3_shell_probe(box_margin,low,high,target_voxel=h,max_cells=cap)
             for _,h,cap in levels]
    assert any(x['r3_boundary_touch_count']>0 for x in initial)
    a,b,report=complete_window_for_levels(box_margin,low,high,levels=levels)
    assert a[0]<low[0]
    assert report['expanded'] is True
    assert report['expansion_rounds']>=1
    for _,h,cap in levels:
        check=_r3_shell_probe(box_margin,a,b,target_voxel=h,max_cells=cap)
        assert check['r3_boundary_touch_count']==0
    assert np.allclose(high,b) or np.any(b>high)


def test_unbounded_r3_fails_without_false_acceptance():
    with pytest.raises(RuntimeError,match='still touches'):
        complete_window_for_levels(lambda p:np.ones(len(p)),
            [-1,-1,-1],[1,1,1],levels=[('fine',.25,6000)],max_rounds=2,
            max_voxel_scale=2.)


def test_voxel_resolution_guard_is_not_silently_bypassed():
    # Infinite solid forces bounding expansion. Extremely tight resolution
    # budget must stop the process rather than coarsen the test indefinitely.
    with pytest.raises(RuntimeError,match='resolution would degrade'):
        complete_window_for_levels(lambda p:np.ones(len(p)),
            [-1,-1,-1],[1,1,1],levels=[('fine',.05,1000)],max_rounds=10,
            max_voxel_scale=1.03)


def test_legacy_audit_schema_is_rejected_by_production_code():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    audit=(root/'tools/audit_final_r3_global_poisson_barrier.py').read_text()
    parity=(root/'gsrefine/cleanup/roi_barrier_parity.py').read_text()
    onepass=(root/'gsrefine/cleanup/stage2_onepass.py').read_text()
    gate=(root/'gsrefine/cleanup/barrier_separation_gate.py').read_text()
    assert 'COMPLETE_SAMPLED_WINDOW' in audit
    assert 'REGION_GRAPHS_COMPLETED_NUMERIC_NOT_CERTIFIED' in audit
    assert 'WINDOW_TRUNCATED' not in audit
    assert 'r3_audit_window_preflight import' not in audit
    assert 'barrier_audit.v2' in parity
    assert "'audit_version': audit_report.get('version','unversioned')" in parity
    assert 'REGION_GRAPHS_COMPLETED_NUMERIC_NOT_CERTIFIED' in onepass
    assert 'roi_touching_audit_window_samples' not in gate
