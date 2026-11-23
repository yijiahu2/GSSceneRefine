import numpy as np
from gsrefine.cleanup.compare_original_region import compare_masks
from gsrefine.cleanup.local_partition import Domain
from gsrefine.cleanup.local_voxels import RegionTree, classify
from tests.test_local_region_v4 import Box
from tools.audit_final_r3_global_poisson_barrier import grid_centers


def test_original_partition_across_missing_local_patch():
    # Original separator spans z=0. Local distance mesh only covers x>=0.3.
    h = .2
    origin = np.array([-1.41, -1.41, -1.41])
    shape = (14, 14, 14)
    nodes = grid_centers(origin, h, shape)
    field = Box()
    labels = np.where(field(nodes) > 0, (nodes[:, 2] < 0).astype(int), -1)
    index = dict(origin=origin, voxel=h, shape=shape, labels=labels.reshape(shape))
    domain = Domain(field)
    tau = .07

    def local_distance(p):
        return np.hypot(p[:, 2], np.maximum(0., .3-p[:, 0]))

    def wall_distance(p):
        return abs(p[:, 2])

    def ray(p):
        return (p[:, 2] < 0).astype(int), {}

    tree = RegionTree(index, domain, local_distance, tau, 1e-8, wall_distance=wall_distance)
    p = np.random.default_rng(667).uniform(-.995, .995, (12000, 3))
    p[:4] = [[-.6, 0., .001], [-.6, 0., -.001], [.6, 0., tau], [.6, 0., tau+1e-6]]
    actual, _, unresolved, stats = classify(p, tree, domain, local_distance, tau, ray)
    expected = (p[:, 2] > 0) & (local_distance(p) > tau)
    np.testing.assert_array_equal(actual, expected)
    assert not unresolved.any() and tree.conflicts == 0
    assert 0 < stats['short_ray_queries'] < len(p)


def test_comparison_distinguishes_classification_and_auxiliary():
    baseline = np.array([1, 1, 0, 0], bool)
    protected = np.array([0, 1, 0, 0], bool)
    current = np.array([0, 1, 1, 0], bool)
    assert compare_masks(current, baseline)['different'] == 2
    result = compare_masks(current, baseline & ~protected)
    assert result['A_only_removed'] == 2 and result['v3_only_removed'] == 1
    assert result['both_retained'] == 1 and result['deletion_jaccard'] == 0
