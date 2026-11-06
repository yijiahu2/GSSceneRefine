"""Simplified ROI recovery contract: complete background context, local guards."""
import json
import numpy as np
import pytest
import trimesh
import yaml
from gsrefine.roi.core import ROI, Projection
from gsrefine.roi.simple_geometry import analyze_simple, decide_candidate, _test_config
from gsrefine.roi.study import run as study_run
from .test_demo import build_synthetic_case


def test_no_unconditional_restore_remaining_patches():
    cfg=_test_config({})
    base=dict(background=False,boundary_class='side',height=.5,
              inside_connected=True,area_ratio=.08,outside_distance=.2,
              outside_cap=1.,contact_coverage=.1,params=cfg)
    assert decide_candidate(**base)[0] == 'suspect'
    assert decide_candidate(**{**base,'inside_connected':False})[0] == 'uncertain'
    assert decide_candidate(**{**base,'area_ratio':.9})[0] == 'uncertain'
    assert decide_candidate(**{**base,'outside_distance':1.3})[0] == 'uncertain'
    assert decide_candidate(**{**base,'contact_coverage':.9})[0] == 'uncertain'
    # Strong background evidence excludes only local base/side regions, not top.
    assert decide_candidate(**{**base,'background':True,'height':.2})[0] == 'normal'
    assert decide_candidate(**{**base,'background':True,'boundary_class':'top','height':.9})[0] == 'suspect'
    assert decide_candidate(**{**base,'background':True,'height':.9})[0] == 'suspect'


def make_sheet(size=3.,cells=40):
    s=np.linspace(-size,size,cells+1)
    vv=np.array([(x,y,0.) for y in s for x in s],float)
    ff=[]
    for j in range(cells):
        for i in range(cells):
            a=j*(cells+1)+i; b=a+1; c=a+cells+1; d=c+1
            ff.extend(((a,b,d),(a,d,c)))
    return trimesh.Trimesh(vertices=vv,faces=np.asarray(ff),process=False)


def test_large_outside_perimeter_excluded_but_local_candidates_bounded():
    m=make_sheet()
    roi=ROI([Projection.from_json({'name':'top','origin':[0,0,0],
        'axes':[[1,0,0],[0,1,0]],
        'polygon':[[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]]})])
    guard={'outside_search_frac':.09,'inner_band_frac':.09,
           'geodesic_limit_frac':.18,'max_extension_area_fraction':.01}
    result=analyze_simple(m,roi,np.ones(len(m.faces)),.75,9.,guard,np.array([0,0,1]))
    stats=result['stats']
    assert stats['full_exterior']['credible_exterior_area_centroid_approx'] > stats['initial_exterior_candidate_area']
    assert stats['expected_perimeter_components'] >= 1
    assert stats['areas_by_label']['normal'] > 0
    assert np.all(result['label_masks']['normal'] <= result['candidate'])
    assert not np.any(result['label_masks']['normal'] & result['inside'])
    assert stats['full_exterior']['no_repeated_full_area_across_local_sources']
    assert sum(stats['areas_by_label'].values()) == pytest.approx(stats['initial_exterior_candidate_area'])


def test_cut_tip_still_eligible_and_all_faces_unscored(tmp_path):
    subject,background=build_synthetic_case(tmp_path)
    report=study_run(tmp_path/'config.yaml',tmp_path/'roi.json',tmp_path/'out',
                     up_vector=[0,0,1],run_ablation=True)
    assert report['poisson_face_scoring']['enabled'] is False
    assert report['candidate']['algorithm']=='simple_geometry_v1'
    assert report['candidate']['areas_by_label']['suspect'] > 0
    assert report['candidate']['full_exterior']['raw_exterior_face_count'] == report['candidate']['full_exterior']['credible_exterior_face_count']
    masks={n:np.load(tmp_path/f'out/keep_mask_{n}.npy') for n in ('R0','R1','R2','R3')}
    assert sum(masks['R2']) > sum(masks['R0'])
    assert np.all(masks['R3'] >= masks['R0'])
    assert not masks['R3'][len(subject.vertices):].any()
    full=json.loads((tmp_path/'out/full_exterior_components.json').read_text())
    assert len({x['global_exterior_component'] for x in full}) == len(full)
    for x in full:
        assert x['full_raw_exterior_area'] >= x['local_candidate_area'] - 1e-8


def test_legacy_classifier_still_loadable_without_being_default(tmp_path):
    subject,bg=build_synthetic_case(tmp_path)
    # Historical experiment reproducibility remains explicitly opt-in.
    cfg=yaml.safe_load((tmp_path/'config.yaml').read_text())
    cfg['roi_study']={'legacy_growth_classifier':True}
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(cfg))
    report=study_run(tmp_path/'config.yaml',tmp_path/'roi.json',tmp_path/'legacy',
                     up_vector=[0,0,1],run_ablation=False)
    assert 'algorithm' not in report['candidate']


def test_numerical_guardrails():
    with pytest.raises(ValueError):_test_config({'background_min_ring_coverage':1.3})
    with pytest.raises(ValueError):_test_config({'background_min_contact_edges':1})
