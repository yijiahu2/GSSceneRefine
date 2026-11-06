"""Tests of full raw exterior diagnostics vs bounded reliable Gaussian candidate."""
import json
from pathlib import Path
import numpy as np
import pytest
import trimesh
from gsrefine.roi.full_surface import exterior_masks, full_exterior_graph, full_exterior_analysis
from gsrefine.roi.study import run as study_run
from .test_demo import build_synthetic_case


def make_strip(nx=12,ny=3):
    vertices=np.array([[x,y,0] for y in range(ny+1) for x in range(nx+1)],float)
    faces=[]
    for y in range(ny):
        for x in range(nx):
            a=y*(nx+1)+x;b=a+1;c=a+(nx+1);d=c+1
            faces.extend([[a,b,d],[a,d,c]])
    return trimesh.Trimesh(vertices=vertices,faces=np.array(faces),process=False)


def test_full_surface_includes_far_outside_but_not_recovery():
    mesh=make_strip()
    center=mesh.triangles_center
    inside=center[:,0]<3
    confidence=np.ones(len(mesh.faces))
    credible=np.ones(len(mesh.faces),bool)
    candidate=(center[:,0]>=3)&(center[:,0]<4)
    faces=np.flatnonzero(candidate)
    groups=[{'component':0}]
    result=full_exterior_analysis(mesh,inside,credible,candidate,confidence,.75,
          [faces],groups,scene_diag=20.,study_cfg={'full_growth_max_diag_frac':.9})
    raw, trusted=exterior_masks(inside,credible)
    assert result['stats']['raw_exterior_face_count']==int(raw.sum())
    assert result['stats']['credible_exterior_face_count']==int(trusted.sum())
    assert result['stats']['credible_exterior_area_centroid_approx'] > result['stats']['bounded_candidate_area']
    assert result['components'][0]['full_credible_reached_area'] > result['components'][0]['local_candidate_area']
    np.testing.assert_array_equal(result['raw_mask'],raw)
    np.testing.assert_array_equal(candidate,(center[:,0]>=3)&(center[:,0]<4))


def test_raw_low_confidence_bridge_continuity_without_claiming_area():
    mesh=make_strip()
    c=mesh.triangles_center
    inside=c[:,0]<3
    confidence=np.ones(len(mesh.faces))
    confidence[(c[:,0]>=5)&(c[:,0]<6)]=.1
    credible=confidence>=.75
    bounded=(c[:,0]>=3)&(c[:,0]<4)
    result=full_exterior_analysis(mesh,inside,credible,bounded,confidence,.75,
              [np.flatnonzero(bounded)],[{'component':0}],20.,
              {'full_growth_max_diag_frac':1.})
    assert result['stats']['raw_exterior_area_centroid_approx'] > result['stats']['credible_exterior_area_centroid_approx']
    assert result['components'][0]['far_raw_area'] >= result['components'][0]['far_credible_area']
    graph=full_exterior_graph(mesh,inside,confidence,.75)
    assert graph.nnz > 0


def test_generated_full_ply_and_local_candidate_are_distinct(tmp_path):
    build_synthetic_case(tmp_path)
    report=study_run(tmp_path/'config.yaml',tmp_path/'roi.json',tmp_path/'study',
                 up_vector=[0,0,1],run_ablation=False)
    target=tmp_path/'study'
    for name in ('exterior_raw_full.ply','exterior_credible_full.ply',
                 'exterior_candidate_local.ply','full_exterior_components.json'):
        assert (target/name).exists(),name
    diagnostics=report['candidate']['full_exterior']
    assert diagnostics['credible_exterior_area_centroid_approx'] >= diagnostics['bounded_candidate_area']
    assert diagnostics['low_confidence_faces_not_used_for_gaussian_recovery']
    assert len(json.loads((target/'full_exterior_components.json').read_text()))>=1
