"""Extend the verified A labels using balls contained in R3 and clear of its wall."""
import numpy as np
from scipy.spatial import cKDTree

from gsrefine.cleanup.roi_barrier_parity import UNKNOWN


class RegionCover:
    def __init__(self, index, tree, radius):
        self.radius = radius
        coords = np.argwhere(index['labels'] >= 0)
        xyz = index['origin'] + (coords + .5) * index['voxel']
        sizes = np.empty(len(xyz))
        for i in range(0, len(xyz), 16384):
            sizes[i:i+16384] = radius(xyz[i:i+16384])
        self.sources = []
        self.width = float(index['voxel'])
        self.add(xyz, sizes, index['labels'][tuple(coords.T)])
        # Fine adaptive leaves provide extra references near the wall and R3 shell.
        centers, radii, sides = [], [], []
        for depth, (keys, state, _, _) in enumerate(tree.levels):
            rows = np.flatnonzero(state >= 0)
            rows = rows[tree.side[state[rows]] >= 0]
            if not len(rows):
                continue
            ijk = np.column_stack(np.unravel_index(keys[rows], tree.shape * 2**depth))
            width = tree.h / 2**depth
            centers.append(tree.origin + (ijk + .5)*width)
            # Every labelled leaf passed clearance > half its body diagonal.
            radii.append(np.full(len(rows), np.sqrt(3.)*.5*width))
            sides.append(tree.side[state[rows]])
        if centers:
            self.add(np.concatenate(centers), np.concatenate(radii), np.concatenate(sides))

    def add(self, xyz, sizes, sides):
        ok = sizes > 0
        # A large certified ball can cover points whose nearest reference
        # centres all have tiny balls. Search comparable radii together.
        xyz, sizes, sides = xyz[ok], sizes[ok], sides[ok]
        bins = np.frexp(sizes / self.width)[1]
        for level in np.unique(bins)[::-1]:
            group = bins == level
            self.sources.append((cKDTree(xyz[group]), sizes[group], sides[group]))

    def __call__(self, points):
        result = np.full(len(points), UNKNOWN, np.int8)
        conflicts = np.zeros(len(points), bool)
        query_radius = self.radius(points)
        for spatial, sizes, sides in self.sources:
            rows = np.flatnonzero((result == UNKNOWN) & ~conflicts)
            if not len(rows):
                break
            # Each ball lies in the same R3 component and on the same side
            # as its centre. Overlapping open balls give a continuous path.
            k = min(8, spatial.n)
            d, ids = spatial.query(points[rows], k=k)
            d, ids = d.reshape(-1, k), ids.reshape(-1, k)
            within = d < sizes[ids]
            pending = ~within.any(axis=1)
            if pending.any():
                rq = query_radius[rows[pending]]
                within[pending] = (rq[:, None] > 0) & (d[pending] < sizes[ids[pending]] + rq[:, None])
            out = (within & (sides[ids] == 0)).any(axis=1)
            inside = (within & (sides[ids] == 1)).any(axis=1)
            result[rows[out & ~inside]] = 0
            result[rows[inside & ~out]] = 1
            conflicts[rows[out & inside]] = True
        return result
