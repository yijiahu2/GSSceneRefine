"""Whole-window adaptive cells with face-connected labels and conservative distance bounds."""
from itertools import product

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from gsrefine.cleanup.roi_barrier_parity import INSIDE, OUTSIDE, UNKNOWN
from gsrefine.cleanup.stage2_runtime import measured
from gsrefine.cleanup.threshold_distance import exact

CHILD = np.asarray(list(product(range(2), repeat=3)))
PENDING, SPLIT = -1, -2


def match(keys, wanted):
    idx = keys.searchsorted(wanted)
    valid = idx < len(keys)
    valid[valid] &= keys[idx[valid]] == wanted[valid]
    return idx, valid


class RegionTree:
    @measured('whole_r3_voxel_construction')
    def __init__(self, index, domain, distance, tau, guard, budget=1200000, *, wall_distance=None, refine_keys=None):
        self.origin = np.asarray(index['origin'])
        base_shape = np.asarray(index['shape'])
        self.shape = (base_shape+3)//4
        self.h = 4*float(index['voxel'])
        self.levels = []
        self.guard = guard
        roots = np.arange(np.prod(self.shape))
        # All roots exist, including empty passages. An inactive terminal root
        # stays unresolved; this optimization can never create a new connection.
        shape = self.shape
        near_roi = np.zeros(self.shape, bool)
        near = np.argwhere(index['labels'] >= 0)//4
        near_roi[tuple(near.T)] = True
        near_roi = maximum_filter(near_roi,size=3).ravel()
        keys = roots
        total = len(keys)
        free_count = 0
        for level in range(12):
            size = self.h/(2**level)
            radius = np.sqrt(3.)*.5*size
            coords = np.column_stack(np.unravel_index(keys, shape))
            pos = self.origin + (coords+.5)*size
            state = np.full(len(keys), PENDING, np.int32)
            lower = np.zeros(len(keys))
            upper = np.full(len(keys), np.inf)
            active = near_roi[np.ravel_multi_index((coords//(2**level)).T, self.shape)]
            rows = np.flatnonzero(active)
            for start in range(0, len(rows), 16384):
                ix = rows[start:start+16384]
                d = distance(pos[ix]).astype(float)
                lower[ix] = np.maximum(0., d-radius-guard)
                upper[ix] = d+radius+guard
                # A different wall may cross a cell far from the local distance mesh.
                # The safety interval always refers to the ROI-local distance mesh.
                wall_clear = (lower[ix] > 0) if wall_distance is None else (
                    wall_distance(pos[ix]).astype(float) > radius + guard)
                clear = wall_clear & (domain.interior_radius(pos[ix]) > radius+guard)
                ids = ix[clear]
                state[ids] = np.arange(free_count, free_count+len(ids))
                free_count += len(ids)
            self.levels.append((keys, state, lower, upper))
            forced = (np.isin(keys, refine_keys[level]) if refine_keys and level < len(refine_keys)
                      else np.zeros(len(keys), bool))
            refine = np.flatnonzero((active & (state == PENDING) & (upper > tau) & (size > tau)) | forced)
            if level == 11 or not len(refine):
                break
            # Keep the complete coarser leaves for anything beyond the budget.
            take = max(0, (budget-total)//8)
            if take < len(refine):
                refine = refine[np.linspace(0, len(refine)-1, take, dtype=int)] if take else refine[:0]
            if not len(refine):
                break
            state[refine] = SPLIT
            children = (coords[refine, None, :]*2+CHILD).reshape(-1,3)
            shape = shape*2
            keys = np.sort(np.ravel_multi_index(children.T, shape))
            total += len(keys)
        self.leaf_nodes = free_count
        self.side = np.full(free_count, UNKNOWN, np.int8)
        self.connect(index)

    @measured('whole_r3_face_components')
    def connect(self, index):
        src, dst = [], []
        for level, (keys, state, _, _) in enumerate(self.levels):
            ids = np.flatnonzero(state >= 0)
            shape = self.shape*2**level
            coords = np.column_stack(np.unravel_index(keys[ids], shape))
            for axis in range(3):
                for sign in (-1, 1):
                    nb = coords.copy()
                    nb[:, axis] += sign
                    rows = np.flatnonzero((nb[:, axis] >= 0) & (nb[:, axis] < shape[axis]))
                    for target_level in range(level, -1, -1):
                        if not len(rows):
                            break
                        want = np.ravel_multi_index(nb[rows].T, self.shape*2**target_level)
                        other, exists = match(self.levels[target_level][0], want)
                        s = self.levels[target_level][1][other[exists]]
                        # Same-level edges only once; coarse/fine pairs found from fine side.
                        keep = (s >= 0) & ((sign > 0) or target_level != level)
                        src.append(state[ids[rows[exists][keep]]])
                        dst.append(s[keep])
                        rows = rows[~exists]
                        nb[rows] //= 2
        a = np.concatenate(src) if src else np.empty(0,int)
        b = np.concatenate(dst) if dst else np.empty(0,int)
        graph = coo_matrix((np.ones(len(a), np.uint8), (a,b)),
                           shape=(self.leaf_nodes,self.leaf_nodes)).tocsr()
        n, component = connected_components(graph, directed=False)
        flags = np.zeros(n, np.uint8)
        labels = index['labels']
        coords = np.argwhere(labels >= 0)
        pts = self.origin + (coords+.5)*float(index['voxel'])
        cells, _, _ = self.locate(pts)
        ok = cells >= 0
        np.bitwise_or.at(flags, component[cells[ok]], (1 << labels[tuple(coords[ok].T)]).astype(np.uint8))
        self.side[flags[component] == 1] = OUTSIDE
        self.side[flags[component] == 2] = INSIDE
        self.conflicts = int(np.count_nonzero(flags == 3))
        self.conflict_keys = []
        if self.conflicts:
            for keys, state, _, _ in self.levels:
                rows = np.flatnonzero(state >= 0)
                self.conflict_keys.append(keys[rows[flags[component[state[rows]]] == 3]])
        self.summary = dict(level_cells=[len(level[0]) for level in self.levels],
                            face_components=int(n), conflicting_components=self.conflicts,
                            face_links=len(a), labelled_cells=int((self.side >= 0).sum()))

    def locate(self, points):
        cells = np.full(len(points), PENDING, np.int32)
        lower = np.zeros(len(points))
        upper = np.full(len(points), np.inf)
        rows = np.arange(len(points))
        for level, (keys, state, lo, hi) in enumerate(self.levels):
            if not len(rows):
                break
            shape = self.shape*2**level
            coords = np.floor((points[rows]-self.origin)/(self.h/2**level)).astype(np.int64)
            ok = ((coords >= 0) & (coords < shape)).all(axis=1)
            rows = rows[ok]
            idx, found = match(keys, np.ravel_multi_index(coords[ok].T,shape))
            idx, rows = idx[found], rows[found]
            cells[rows], lower[rows], upper[rows] = state[idx], lo[idx], hi[idx]
            rows = rows[state[idx] == SPLIT]
        return cells, lower, upper


@measured('region_gaussian_lookup')
def classify(points, tree, domain, distance, tau, ray_query, *, batch=30000, region_cover=None):
    outside = np.zeros(len(points), bool)
    withheld = np.zeros(len(points), bool)
    inside = np.zeros(len(points), bool)
    stats = dict(distance_queries=0, short_ray_queries=0, near_kept=0,
                 outside_scope=0, direct_inside=0, far_outside_by_bound=0,
                 cover_queries=0, cover_resolved=0, cover_unresolved=0)
    residual_batches = []
    for start in range(0, len(points), batch):
        rows = np.arange(start, min(start+batch,len(points)))
        allowed = domain.plane(points[rows]) > 0
        withheld[rows[~allowed]] = True
        stats['outside_scope'] += int((~allowed).sum())
        rows = rows[allowed]
        cell, lower, upper = tree.locate(points[rows])
        side = np.full(len(rows), UNKNOWN, np.int8)
        ok = cell >= 0
        side[ok] = tree.side[cell[ok]]
        inside[rows[side == INSIDE]] = True
        stats['direct_inside'] += int((side == INSIDE).sum())
        far = lower > tau
        stats['far_outside_by_bound'] += int(((side == OUTSIDE) & far).sum())
        measure = (side != INSIDE) & ~far & (upper > tau)
        if measure.any():
            stats['distance_queries'] += int(measure.sum())
            far[measure] = distance(points[rows[measure]]).astype(float) > tau
        outside[rows[(side == OUTSIDE) & far]] = True
        stats['near_kept'] += int(((side != INSIDE) & ~far).sum())
        unknown = rows[(side == UNKNOWN) & far]
        if len(unknown):
            if region_cover is None:
                labels, _ = ray_query(points[unknown])
                stats['short_ray_queries'] += len(unknown)
            else:
                labels = region_cover(points[unknown])
                resolved = int((labels != UNKNOWN).sum())
                stats['cover_queries'] += len(unknown)
                stats['cover_resolved'] += resolved
                stats['cover_unresolved'] += len(unknown) - resolved
                if ray_query is not None and resolved < len(unknown):
                    residual_batches.append(unknown[labels == UNKNOWN])
            outside[unknown[labels == OUTSIDE]] = True
            inside[unknown[labels == INSIDE]] = True
            withheld[unknown[labels == UNKNOWN]] = True
        if region_cover is not None:
            end = min(start + batch, len(points))
            if end == len(points) or end // 1000000 != start // 1000000:
                print(f"[original-region] Voxel lookup {end:,}/{len(points):,}; "
                      f"unverified so far={stats['cover_unresolved']:,}", flush=True)
    if region_cover is not None and ray_query is not None:
        remaining = np.concatenate(residual_batches) if residual_batches else np.empty(0, np.intp)
        print(f"[original-region] Local lookup resolved {stats['cover_resolved']:,}/"
              f"{stats['cover_queries']:,}; final short-ray points={len(remaining):,}", flush=True)
        if len(remaining):
            labels, _ = ray_query(points[remaining])
            outside[remaining] = labels == OUTSIDE
            inside[remaining] = labels == INSIDE
            withheld[remaining] = labels == UNKNOWN
            stats['short_ray_queries'] = len(remaining)
        stats['short_ray_unresolved'] = int(withheld[remaining].sum())
    return outside, inside, withheld, stats
