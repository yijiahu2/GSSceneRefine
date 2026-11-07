"""Regression tests for Stage-2 conservative, conforming ROI surface refinement."""
import numpy as np
import pytest
import trimesh

from gsrefine.geometry.roi_surface import (
    _adapt_roi_mesh, _clip_piecewise_linear,
    _find_roi_aliases, _refine_faces_conforming,
)


def _triangle_with_central_roi():
    # All three Poisson vertices lie OUTSIDE the ROI, yet the centroid is inside.
    v = np.array([[-2., -1., 0.], [2., -1., 0.], [0., 2., 0.]])
    mesh = trimesh.Trimesh(vertices=v, faces=[[0, 1, 2]], process=False)

    def field(points):
        points = np.asarray(points, dtype=np.float64)
        return np.minimum(.4 - np.abs(points[:, 0]), .5 - np.abs(points[:, 1]))

    return mesh, field


def test_crossing_from_three_outside_corners_is_recovered():
    mesh, field = _triangle_with_central_roi()
    original_vertices = np.asarray(mesh.vertices).copy()
    original_faces = np.asarray(mesh.faces).copy()
    marked, info = _find_roi_aliases(mesh, field(mesh.vertices), field, 1e-5)
    assert marked.tolist() == [0]
    assert info['conflicting_probes'] > 0

    refined, scalar, report = _adapt_roi_mesh(
        mesh, field(mesh.vertices), field,
        {'max_unresolved_face_violation': 1e-5})
    assert report['initial_conflicting_probes'] > 0
    assert report['final_conflicting_probes'] == 0
    assert report['refinement_levels'] > 0
    assert len(refined.faces) > len(mesh.faces)
    assert np.array_equal(original_vertices, mesh.vertices)
    assert np.array_equal(original_faces, mesh.faces)
    cut, _ = _clip_piecewise_linear(refined, scalar)
    assert len(cut.faces) > 0
    assert np.min(cut.vertices[:,0]) >= -.4 - 1e-9
    assert np.max(cut.vertices[:,0]) <=  .4 + 1e-9
    assert np.min(cut.vertices[:,1]) >= -.5 - 1e-9
    assert np.max(cut.vertices[:,1]) <=  .5 + 1e-9


def test_one_refined_face_splits_shared_neighbor_edge_with_consistent_winding():
    vertices = np.array([[0.,0,0],[2,0,0],[2,2,0],[0,2,0]])
    raw = trimesh.Trimesh(vertices=vertices, faces=[[0,1,2],[0,2,3]], process=False)
    refined, scalar, report = _refine_faces_conforming(
        raw, np.arange(4,dtype=float), np.array([0]),
        lambda pts: np.full(len(pts),1.), maximum_added_faces=20)
    assert report['affected_faces_including_neighbors'] == 2
    assert report['split_edges'] == 3
    assert len(refined.faces) == 6  # marked triangle 4 + unmarked neighbor 2
    assert np.isclose(refined.area,raw.area)
    assert all(n[2] > 0 for n in refined.face_normals)
    # Midpoint of shared diagonal is a SINGLE vertex, and both sides use it.
    shared = np.flatnonzero(np.all(np.isclose(refined.vertices,[1,1,0]),axis=1))
    assert len(shared) == 1
    assert np.sum(np.any(refined.faces == shared[0],axis=1)) >= 4
    # Interface edges split into two; neither original full diagonal survives.
    edges = np.sort(refined.edges,axis=1)
    assert not np.any(np.all(edges==[0,2],axis=1))
    assert len(scalar) == len(refined.vertices)


def test_adaptive_refinement_fails_closed_at_depth_budget():
    mesh, field = _triangle_with_central_roi()
    with pytest.raises(ValueError,match='remains ambiguous'):
        _adapt_roi_mesh(mesh, field(mesh.vertices), field,
                        {'max_unresolved_face_violation':1e-5,
                         'roi_crop_max_refinement_levels':0})
    with pytest.raises(ValueError,match='safety budget'):
        _adapt_roi_mesh(mesh, field(mesh.vertices), field,
                        {'max_unresolved_face_violation':1e-5,
                         'roi_crop_max_extra_faces':1})


def test_no_regression_on_smooth_planar_halfspace():
    vertices = np.array([[-1.,0,0],[1,0,0],[1,1,0],[-1,1,0]])
    mesh = trimesh.Trimesh(vertices=vertices,faces=[[0,1,2],[0,2,3]],process=False)
    field = lambda pts: np.asarray(pts)[:,0]
    refined, scalar, report = _adapt_roi_mesh(mesh,field(vertices),field,
                                               {'max_unresolved_face_violation':1e-5})
    assert report['refinement_levels'] == 0
    assert len(refined.faces) == 2
    cut,_ = _clip_piecewise_linear(refined,scalar)
    assert np.isclose(cut.area,1.0)


def test_full_fallback_extract_adaptively_recovers_interior_crossing(monkeypatch):
    from gsrefine.geometry import roi_surface
    mesh, field = _triangle_with_central_roi()

    class View:
        def signed_and_normal(self, points):
            return field(points), np.zeros((len(points), 3))

    class ROI:
        views = (View(),)

    class Gaussian:
        vertex_count = 10

        def positions(self):
            return np.array([[-.2,-.2,0],[.2,.2,0]])

    monkeypatch.setattr(roi_surface, '_guard_anchors',
        lambda *args, **kwargs: (np.empty((0,3)), np.empty(0),
              {'source':'stage2_inferred_gaussian_guard','is_exact_stage1_r3_surface_mask':False}))
    new_mesh, report = roi_surface.extract_roi_surface(
        mesh, ROI(), Gaussian(), np.ones((100,3)),
        {'max_unresolved_face_violation':1e-5},o3d=None,gaussian_radius=.05)
    assert len(new_mesh.faces) > 0
    assert report['selection_mode'] == 'inferred_r3_gaussian_support'
    assert report['adaptive_refinement']['initial_conflicting_probes'] > 0
    assert report['adaptive_refinement']['final_conflicting_probes'] == 0
    assert len(mesh.faces) == 1


def test_two_split_edges_on_neighbor_have_no_cracks_or_inverted_triangles():
    vertices = np.array([[0.,0,0],[2,0,0],[0,2,0],[1,-1,0],[2,2,0]])
    mesh=trimesh.Trimesh(vertices=vertices,
                         faces=[[0,1,2],[1,0,3],[2,1,4]],process=False)
    refined,_,report=_refine_faces_conforming(
        mesh,np.ones(len(vertices)),np.array([1,2]),
        lambda xyz: np.ones(len(xyz)),maximum_added_faces=100)
    assert report['affected_faces_including_neighbors']==3
    assert np.isclose(refined.area,mesh.area)
    assert np.all(refined.face_normals[:,2]>0)
    assert len(refined.faces)==len(mesh.faces)+report['added_faces']
