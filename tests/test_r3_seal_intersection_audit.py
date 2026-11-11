import importlib.util
from pathlib import Path
import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/'tools'/'audit_r3_seal_intersections.py'
spec=importlib.util.spec_from_file_location('audit_r3_seal_intersections', SCRIPT)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_projected_bowtie_reports_crossing_and_original_3d_endpoints():
    uv=np.array([[0,0],[2,2],[0,2],[2,0]],dtype=float)
    xyz=np.column_stack([uv, [0,0,1,1]])
    result=mod.projected_segment_crossings(uv, xyz)
    assert result['count']==1
    p=result['examples'][0]
    assert (p['segment_i'],p['segment_j'])==(0,2)
    assert np.allclose(p['intersection_uv'],[1,1])
    assert len(p['segment_i_xyz'])==2
    assert abs(p['source_3d_gap_at_projected_crossing']-1.0)<1e-9


def test_simple_square_has_no_projection_crossing():
    uv=np.array([[0,0],[2,0],[2,2],[0,2]],dtype=float)
    result=mod.projected_segment_crossings(uv,np.column_stack([uv,np.zeros(4)]))
    assert result['count']==0
    assert mod._plane_audit(uv,np.column_stack([uv,np.zeros(4)]))['valid_polygon']


def test_pair_layer_category_counts():
    pairs=np.array([[0,2],[0,5],[4,7],[6,7],[1,3],[4,5]],dtype=np.int64)
    labels=mod.pair_categories(pairs,n_main=3,n_skirt=3,n_total=8)
    assert labels==['main-main','main-skirt','cap-skirt','cap-cap','main-skirt','skirt-skirt']


def test_nonadjacent_end_touch_classified():
    uv=np.array([[0,0],[2,0],[2,2],[1,0],[0,2]],dtype=float)
    xyz=np.column_stack([uv,np.arange(5)])
    result=mod.projected_segment_crossings(uv,xyz)
    assert result['count']>=1
