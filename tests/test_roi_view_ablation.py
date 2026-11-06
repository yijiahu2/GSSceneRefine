import json
import numpy as np
import pytest
from gsrefine.roi.core import ROI, Projection
from gsrefine.roi.view_ablation import make_top_only_roi


def sample():
    return {'views':[
        {'name':'top','origin':[0,0,0],'axes':[[1,0,0],[0,1,0]],'polygon':[[-1,-1],[1,-1],[1,1],[-1,1]]},
        {'name':'front','origin':[0,0,0],'axes':[[1,0,0],[0,0,1]],'polygon':[[-1,-0.5],[1,-0.5],[1,0.5],[-1,0.5]]}],
        'annotation_metadata':{'up_vector':[0,0,1]}}


def test_single_view_is_unbounded_along_up_and_dual_is_subset():
    original=sample()
    top=make_top_only_roi(original)
    assert len(top['views'])==1 and len(original['views'])==2
    p=np.asarray([[0,0,0],[0,0,2],[0,0,-4],[3,0,0]])
    tm=Projection.from_json(top['views'][0]).signed_and_normal(p)[0]>=0
    assert tm.tolist()==[True,True,True,False]


def test_missing_up_requires_explicit_vector():
    original=sample();del original['annotation_metadata']
    with pytest.raises(ValueError,match='up_vector'):
        make_top_only_roi(original)
    assert len(make_top_only_roi(original,up=[0,0,1])['views'])==1


def test_top_view_can_be_second_in_list():
    original=sample();original['views'].reverse()
    reduced=make_top_only_roi(original)
    assert reduced['views'][0]['name']=='top'
