"""Stage-A spatial leakage regression. No Poisson scoring or Stage-B changes."""
import numpy as np
import pytest
import trimesh

from gsrefine.roi.study import anchored_local_recovery_faces


class HalfSpaceROI:
    """The interior is x <= 0, with + inside signed projection distance."""

    def distances(self, xyz):
        xyz = np.asarray(xyz)
        return -xyz[:, 0, None]


def _strip_with_island():
    # 12 faces in a planar, edge-connected strip, with an exterior unconnected
    # island that is close in Euclidean space but cannot be surface-reached.
    xs = np.arange(-1.0, 3.6, .5)
    vertices = [[float(x), float(y), 0.] for x in xs for y in (0., .5)]
    faces = []
    for i in range(len(xs)-1):
        a = 2*i
        faces += [[a,a+2,a+1], [a+1,a+2,a+3]]
    first_island_vertex = len(vertices)
    # All island vertices lie outside the ROI and its adjacency is disconnected.
    vertices += [[.25, .60, 0], [.30, .62, 0], [.27, .68, 0]]
    faces += [[first_island_vertex, first_island_vertex+1, first_island_vertex+2]]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False), len(faces)-1


def test_local_geodesic_stops_large_background_continuation():
    mesh, island = _strip_with_island()
    roi = HalfSpaceROI()
    centers = mesh.triangles_center
    inside = centers[:, 0] <= 0
    selected = ~inside
    selected[island] = True

    kept, info = anchored_local_recovery_faces(mesh, roi, selected, inside, 0.9)
    assert info['boundary_seed_faces'] > 0
    assert kept.any()
    assert not kept[island]
    assert not kept[(centers[:, 0] > 2.0)].any()
    assert np.any(kept & (centers[:, 0] > 0))
    assert np.all(~kept | selected)
    assert info['removed_faces'] > 0
    assert info['beyond_reach_faces'] > 0
    assert info['unanchored_faces'] > 0
    assert info['scope'] == 'R3_stage_A_only'
    assert info['selected_area_after'] < info['selected_area_before']


def test_no_seed_never_recovers_disconnected_suspect_surface():
    mesh = trimesh.Trimesh(
        vertices=[[-.2,0,0],[-.1,.2,0],[-.1,0,.2],
                  [.25,0,0],[.35,.2,0],[.25,0,.2]],
        faces=[[0,1,2], [3,4,5]], process=False)
    kept, info = anchored_local_recovery_faces(
        mesh, HalfSpaceROI(), [False, True], [True, False], 1.0)
    assert not kept.any()
    assert info['boundary_seed_faces'] == 0
    assert info['unanchored_faces'] == 1


def test_crossing_face_itself_is_anchor_when_face_center_is_outside():
    mesh = trimesh.Trimesh(
        vertices=[[-.1,0,0],[.4,0,0],[.4,.2,0]],
        faces=[[0,1,2]], process=False)
    kept, info = anchored_local_recovery_faces(
        mesh, HalfSpaceROI(), [True], [False], .2)
    assert kept[0]
    assert info['boundary_seed_faces'] == 1


def test_no_candidates_and_invalid_guards():
    mesh, _ = _strip_with_island()
    zeros = np.zeros(len(mesh.faces), dtype=bool)
    kept, info = anchored_local_recovery_faces(mesh, HalfSpaceROI(), zeros, zeros, .25)
    assert not kept.any() and info['removed_faces'] == 0
    with pytest.raises(ValueError, match='Positive finite'):
        anchored_local_recovery_faces(mesh, HalfSpaceROI(), zeros, zeros, 0)
    with pytest.raises(ValueError, match='match mesh.faces'):
        anchored_local_recovery_faces(mesh, HalfSpaceROI(), zeros[:-1], zeros, 1)
    with pytest.raises(ValueError, match='outside ROI'):
        anchored_local_recovery_faces(mesh, HalfSpaceROI(), np.ones(len(zeros),bool), np.ones(len(zeros),bool), 1)
