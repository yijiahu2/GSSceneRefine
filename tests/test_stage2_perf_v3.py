import os

import numpy as np

from gsrefine.cleanup import stage2_runtime as rt
from gsrefine.cleanup.roi_barrier_parity import INSIDE, OUTSIDE, _query_nearby_candidates
from tests.test_stage2_roi_barrier_parity import FakeTriangleScene, O3D


def test_digest_reuse_and_invalidation(tmp_path):
    path = tmp_path / 'source.ply'
    path.write_bytes(b'abc')

    @rt.run_scope
    def exercise():
        before = rt.file_digest(path)
        assert rt.file_digest(path) == before
        assert rt.snapshot()['steps']['read_file_sha256']['calls'] == 1
        stat = path.stat()
        replacement = tmp_path / 'other'
        replacement.write_bytes(b'xyz')
        os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        os.replace(replacement, path)
        assert rt.file_digest(path) != before
        assert rt.snapshot()['steps']['read_file_sha256']['calls'] == 2

    exercise()

    @rt.run_scope
    def fresh():
        rt.file_digest(path)
        assert rt.snapshot()['steps']['read_file_sha256']['calls'] == 1

    fresh()


def test_navigation_tree_shared_and_ray_costs_attributed():
    shape = (4, 4, 4)
    z = np.indices(shape)[2] - 1.5
    index = dict(shape=shape, origin=np.full(3, -2.), voxel=1.,
                 labels=np.where(z > 0, OUTSIDE, INSIDE))
    xyz = np.array([[.1, .1, .25], [.1, .1, -.25]])

    @rt.run_scope
    def classify():
        for _ in range(2):
            labels, _ = _query_nearby_candidates(
                xyz, index, lambda p: np.ones(len(p)), FakeTriangleScene(), O3D,
                np.array([[0., 0., 1.]]), neighbours=12, batch=2)
            np.testing.assert_array_equal(labels, [OUTSIDE, INSIDE])
        steps = rt.snapshot()['steps']
        assert steps['ray_navigation_tree_build']['calls'] == 1
        assert steps['query_nearby_candidates']['calls'] == 2
        for key in ('ray_origin_r3_check', 'ray_nearest_nodes', 'ray_segment_r3_check',
                    'ray_triangle_intersections', 'ray_intersection_parsing'):
            assert steps[key]['calls'] >= 2
            assert steps[key]['inclusive_s'] <= steps['query_nearby_candidates']['inclusive_s']

    classify()
