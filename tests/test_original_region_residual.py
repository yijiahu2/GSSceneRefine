import numpy as np

from gsrefine.cleanup.local_voxels import classify


def test_only_far_unlocated_points_reach_ray_across_batches():
    # Rows 0..6 are respectively inside, far outside, on threshold, near
    # unknown, far located by cover, far residual, and unresolved residual.
    points = np.c_[np.arange(7), np.zeros((7, 2))]
    class Cells:
        side = np.array([1, 0])
        def locate(self, p):
            i = p[:, 0].astype(int)
            return (np.array([0, 1, 1, -1, -1, -1, -1])[i],
                    np.array([2, 2, 0, 0, 0, 0, 0])[i], np.full(len(i), 3))
    class Domain:
        def plane(self, p):
            return np.ones(len(p))
    measured, covered, cast = [], [], []
    def distance(p):
        i = p[:, 0].astype(int)
        measured.extend(i)
        return np.array([2, 2, 1, .9, 2, 2, 2])[i]
    def cover(p):
        i = p[:, 0].astype(int)
        covered.extend(i)
        return np.where(i == 4, 0, -1)
    def ray(p):
        i = p[:, 0].astype(int)
        cast.append(i.tolist())
        return np.where(i == 5, 0, -1), {}
    remove, inside, withheld, stats = classify(
        points, Cells(), Domain(), distance, 1, ray, region_cover=cover, batch=2)
    assert measured == [2,3,4,5,6] and covered == [4,5,6] and cast == [[5,6]]
    assert np.flatnonzero(remove).tolist() == [1,4,5]
    assert np.flatnonzero(withheld).tolist() == [6]
    assert stats['short_ray_queries'] == 2 and stats['short_ray_unresolved'] == 1
