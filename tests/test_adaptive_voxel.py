"""Regressions for safe lookup, sparse refinement and the real finite-ray fallback."""
import numpy as np
import pytest
import trimesh

from gsrefine.cleanup.adaptive_voxel import WallClearance, classify, make_interior_radius
from gsrefine.cleanup.roi_barrier_parity import _make_navigation, _query_nearby_candidates
from gsrefine.roi.core import ROI, Projection
from gsrefine.roi.two_stage_sdf import CoreImplicitField, roi_polygon_implicit_field
from tools.audit_final_r3_global_poisson_barrier import grid_centers, make_scene


def plane(x, y):
    return .037 + .31 * x - .21 * y


def toy_navigation(tmp_path):
    xy = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2]])
    verts = np.column_stack((xy, plane(*xy.T)))
    mesh = trimesh.Trimesh(verts, [[0, 1, 2], [0, 2, 3]], process=False)
    scene, o3d = make_scene(mesh)

    def field(p):
        return 1. - np.max(np.abs(p), axis=1)

    origin = np.full(3, -1.25)
    shape = (10, 10, 10)
    h = .25
    centers = grid_centers(origin, h, shape)
    inside = field(centers) > 0
    passable = inside & (np.abs(centers[:, 2]-plane(*centers[:, :2].T)) > .015*h*1.1)
    path = tmp_path/'fine.npz'
    np.savez(path, origin=origin, voxel=h, shape=shape, passable=passable.reshape(shape),
             final_r3_inside=inside.reshape(shape))
    nav = _make_navigation(path, field, [0, 0, 1], scene, o3d, mesh.face_normals,
                           exterior_seed=[.125, .125, .875])
    def rays(q):
        return _query_nearby_candidates(q, nav, field, scene, o3d, mesh.face_normals,
                                        neighbours=8, min_agree=2, batch=3750)
    return nav, field, mesh, rays


def test_voxel_and_real_rays_on_sloping_open_surface(tmp_path):
    nav, field, mesh, rays = toy_navigation(tmp_path)
    points = np.random.default_rng(214).uniform(-.98, .98, (16000, 3))
    points = points[np.abs(points[:, 2] - plane(*points[:, :2].T)) > .02]
    truth = (points[:, 2] < plane(*points[:, :2].T)).astype(np.int8)
    baseline, _ = rays(points)
    np.testing.assert_array_equal(baseline, truth)
    wall = WallClearance(mesh.vertices, mesh.faces)
    for depth in (0, 6):
        labels, stats = classify(points, nav, lambda q: np.maximum(0., field(q)),
                                 wall, rays, max_depth=depth)
        np.testing.assert_array_equal(labels, baseline)
        assert 0 < stats['direct_points'] < len(points)
        assert stats['direct_points'] + stats['fallback_points'] == len(points)
        if depth:
            assert len(stats['levels']) > 1
            assert stats['direct_points'] > base_direct
        else:
            base_direct = stats['direct_points']


def test_wall_bound_on_folded_and_degenerate_triangles():
    rng = np.random.default_rng(81)
    faces = rng.uniform(-.9, .9, (17, 3, 3))
    faces[2, 1:] = faces[2, 0]
    queries = rng.uniform(-1., 1., (600, 3))
    bound = WallClearance(faces.reshape(-1,3), np.arange(faces.size//3).reshape(-1,3))
    actual = np.linalg.norm(queries-faces[2, 0], axis=1)
    for j, face in enumerate(faces):
        if j != 2:
            closest = trimesh.triangles.closest_point(np.broadcast_to(face,(len(queries),3,3)), queries)
            actual = np.minimum(actual, np.linalg.norm(queries-closest, axis=1))
    radius = bound(queries, np.full(len(queries), 2.))
    assert np.all(radius >= 0) and np.all(radius <= actual + 1e-12)
    assert np.any(radius > .1)


def test_thin_sheet_in_a_cell_and_outside_grid():
    # Both sides of a thin slab are inside a single coarse cell.
    z1, z2 = .04, .09
    xy = np.array([[-2,-2],[2,-2],[0,3]])
    tri = np.vstack([np.column_stack([xy, np.full(3,z)]) for z in (z1,z2)])
    wall = WallClearance(tri, [[0,1,2],[3,4,5]])
    nav = dict(origin=np.zeros(3), voxel=1., shape=[1,1,1], labels=np.zeros((1,1,1),np.int8))
    def reference(q):
        out = ((q[:,2] > z1) & (q[:,2] < z2)).astype(np.int8)
        out[(q<0).any(axis=1) | (q>=1).any(axis=1)] = -1
        return out, {'conflicting_paths':0}
    points = np.random.default_rng(19).uniform(.001,.99,(1600,3))
    points[-1] = [-.1,.5,.07]
    got, stats = classify(points,nav,lambda q: np.maximum(0,np.minimum(q,1-q).min(axis=1)),
                          wall,reference)
    np.testing.assert_array_equal(got,reference(points)[0])
    assert (got == 1).any() and got[-1] == -1 and stats['fallback_points'] > 0


def test_concave_r3_radius_and_recovery_ball():
    poly = [[0,0],[3,0],[3,1],[1,1],[1,3],[0,3]]
    views = [Projection.from_json({'name':'top','axes':[[1,0,0],[0,1,0]],'polygon':poly}),
             Projection.from_json({'name':'front','axes':[[1,0,0],[0,0,1]],
                                   'polygon':[[-1,-1],[4,-1],[4,1],[-1,1]]})]
    roi = ROI(views)
    core = CoreImplicitField(roi_polygon_implicit_field(roi), np.array([[2.5,2.5,0]]), .25)
    radius = make_interior_radius(core, roi, 1e-6, 4.)
    points = np.array([[.92,.92,0],[1.08,1.08,0],[2.5,2.5,0]])
    r = radius(points)
    assert r[0]>0 and r[1]==0 and r[2]>.2
    rng = np.random.default_rng(34)
    for p, dist in zip(points[r>0], r[r>0]):
        dr = rng.normal(size=(100,3))
        dr /= np.linalg.norm(dr,axis=1)[:,None]
        assert np.all(core.evaluate(p + .99999 * dist * dr) < 1e-6)


def test_conflicting_navigation_witnesses_block_lookup():
    nav = dict(origin=np.zeros(3), voxel=1., shape=[2,1,1], labels=np.array([[[0]],[[1]]]))
    with pytest.raises(RuntimeError, match='Opposite navigation'):
        classify(np.array([[.5,.5,.5]]),nav,lambda q:np.full(len(q),2.),
                 lambda q,r:r,lambda q:pytest.fail('conflict silently resolved'))


def test_failed_representative_routes_covered_points_to_individual_rays():
    # Anchor is across an unverified interval, outside every candidate ball.
    nav = dict(origin=np.zeros(3), voxel=1., shape=[2,1,1], labels=np.array([[[0]],[[-1]]]))
    rng = np.random.default_rng(88)
    points = np.array([1.5,.5,.5]) + rng.uniform(-.03,.03,(70,3))
    batches = []
    def ambiguous(q):
        batches.append(len(q))
        return np.full(len(q), -1, np.int8), {'conflicting_paths':len(q)}
    got, stats = classify(points, nav, lambda q: np.full(len(q), .10), lambda q,r:r, ambiguous)
    assert batches == [1,70]
    assert (got == -1).all() and stats['representative_conflicts'] == 1
    assert stats['conflicting_paths'] == len(points) and stats['direct_points'] == 0


@pytest.mark.parametrize('option,expected', [([], 'adaptive-voxel'), (['--classifier','short-ray'],'short-ray')])
def test_full_pipeline_selects_requested_classifier(option, expected):
    from gsrefine.roi_init.full_pipeline import _parser, _stage2_command
    args = _parser().parse_args(['--resume-manifest','manifest.json','--accept-auto-exterior',*option])
    cmd = _stage2_command(args,'manifest.json','test-output')
    assert cmd[cmd.index('--classifier')+1] == expected
