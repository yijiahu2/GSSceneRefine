"""Occupied adaptive voxels for the Stage-2 R3/Poisson side query."""
from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree

CLASSIFIERS = ('adaptive-voxel', 'base-voxel', 'short-ray')
REVISION = 'adaptive_voxel_v1'
MAX_DEPTH = 6
UNKNOWN = -1


def make_interior_radius(core, roi, tolerance, world_scale):
    """An open ball in the restored core is inside the final, expanded R3.

    Each projected polygon distance is ||U||-Lipschitz; the sphere distances
    are 1-Lipschitz. Max and min preserve their common Lipschitz bound.
    Every valid kNN expansion radius is positive, so zero is a safe lower
    bound even in a model whose scales vary greatly across adjacent cells.
    """
    lipschitz = max(1., *(np.linalg.norm(view.U, ord=2) for view in roi.views))
    scale = max(1., world_scale, *(np.abs(view.origin).max() for view in roi.views),
                *(np.abs(view.polygon).max() for view in roi.views))
    guard = 1e-9 + 256 * np.finfo(float).eps * scale

    def radius(points):
        return np.maximum(0., (tolerance - core.evaluate(points) - guard) / lipschitz)
    return radius


class WallClearance:
    """Capped lower distance bound for the original, unmodified wall triangles.

    A triangle lies in its bounding box and within its projection interval
    on every unit direction. Both distances are lower bounds, including for
    long slivers. Prune only nodes whose box cannot reduce the current bound.
    """
    def __init__(self, vertices, faces):
        tri = np.asarray(vertices, np.float64)[faces]
        if not len(tri) or not np.isfinite(tri).all():
            raise ValueError('Invalid original wall triangles')
        tri = tri[cKDTree(tri.mean(axis=1), leafsize=8).indices]
        low, high = tri.min(axis=1), tri.max(axis=1)
        axis = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        self.axis = axis / np.maximum(np.linalg.norm(axis, axis=1)[:, None], np.finfo(float).tiny)
        projection = np.einsum('nvi,ni->nv', tri, self.axis)
        self.plane_low, self.plane_high = projection.min(axis=1), projection.max(axis=1)
        self.scale = max(1., np.abs(low).max(), np.abs(high).max())
        self.levels = [(low, high)]
        while len(low) > 1:
            a = np.arange(0, len(low), 2)
            b = np.minimum(a + 1, len(low) - 1)
            low, high = np.minimum(low[a], low[b]), np.maximum(high[a], high[b])
            self.levels.append((low, high))

    def __call__(self, points, cap):
        points = np.asarray(points, np.float64)
        radii = np.array(cap, dtype=np.float64, copy=True)
        # Includes rounding when the ray backend represents world vertices in float32.
        pad = 8 * np.finfo(np.float32).eps * max(self.scale, np.abs(points).max(initial=0))
        for start in range(0, len(points), 512):
            rows = np.arange(start, min(start + 512, len(points)))
            stack = [(len(self.levels)-1, rows, np.zeros(len(rows), np.intp))]
            while stack:
                level, row, node = stack.pop()
                lo, hi = self.levels[level]
                q = points[row]
                gap = np.maximum(0., np.maximum(lo[node] - q, q - hi[node]))
                bound = np.linalg.norm(gap, axis=1)
                near = bound < radii[row]
                row, node = row[near], node[near]
                if not len(row):
                    continue
                if level:
                    left = node * 2
                    right = left + 1
                    exists = right < len(self.levels[level-1][0])
                    rows2 = np.concatenate((row, row[exists]))
                    nodes2 = np.concatenate((left, right[exists]))
                    for j in range(0, len(nodes2), 8192):
                        stack.append((level-1, rows2[j:j+8192], nodes2[j:j+8192]))
                else:
                    t = np.einsum('ni,ni->n', points[row], self.axis[node])
                    plane = np.maximum(self.plane_low[node] - t, t - self.plane_high[node])
                    np.minimum.at(radii, row, np.maximum(bound[near], plane))
        return np.maximum(0., radii - pad)


def classify(points, index, interior_radius, wall_clearance, query_rays, *,
             max_depth=MAX_DEPTH, log=print):
    """Classify occupied cells; all residuals retain the original ray checks."""
    started = perf_counter()
    points = np.asarray(points, np.float64)
    shape = np.asarray(index['shape'], np.int64)
    origin = np.asarray(index['origin'], np.float64)
    voxel = float(index['voxel'])
    grid = np.asarray(index['labels'])
    if grid.shape != tuple(shape) or not np.isin(grid, (UNKNOWN, 0, 1)).all():
        raise ValueError('Invalid navigation label array')
    if not np.isfinite(points).all() or not np.isfinite(voxel) or voxel <= 0:
        raise ValueError('Invalid voxel query coordinates or grid')
    if not 0 <= max_depth <= MAX_DEPTH:
        raise ValueError('Unsupported subdivision depth')
    assigned = np.full(len(points), UNKNOWN, dtype=np.int8)
    nav_ids = np.flatnonzero(grid.ravel() != UNKNOWN)
    if not len(nav_ids):
        raise ValueError('No labelled navigation nodes')
    nav_xyz = origin + (np.column_stack(np.unravel_index(nav_ids, shape)) + .5) * voxel
    nav_labels = grid.ravel()[nav_ids]
    nav_tree = cKDTree(nav_xyz)
    stats = dict(candidate_count=len(points), levels=[], direct_points=0, fallback_points=0,
                 representative_queries=0, representative_seconds=0., fallback_seconds=0.,
                 representative_unknown=0, representative_conflicts=0, conflicting_paths=0,
                 classifier_revision=REVISION)
    rows = np.flatnonzero(((points >= origin) & (points < origin + shape * voxel)).all(axis=1))
    parent = None
    for depth in range(max_depth + 1):
        if not len(rows):
            break
        tick = perf_counter()
        h = voxel / (2 ** depth)
        size = shape * (2 ** depth)
        ijk = np.floor((points[rows] - origin) / h).astype(np.int64)
        valid = ((ijk >= 0) & (ijk < size)).all(axis=1)
        if not valid.all():
            rows, ijk = rows[valid], ijk[valid]
            if not len(rows):
                break
        cells, inverse = np.unique(np.ravel_multi_index(ijk.T, size), return_inverse=True)
        inverse = inverse.reshape(-1)
        coords = np.column_stack(np.unravel_index(cells, size))
        centers = origin + (coords + .5) * h
        cap = np.minimum(interior_radius(centers), 1.75 * h)
        radii = wall_clearance(centers, cap)
        if not np.isfinite(radii).all() or np.any(radii < 0) or np.any(radii > cap):
            raise ValueError('Invalid cell clearance')
        k = min(8, len(nav_ids))
        dist, ids = nav_tree.query(centers, k=k)
        within = dist.reshape(-1, k) < radii[:, None]
        observed = nav_labels[ids.reshape(-1, k)]
        outside = (within & (observed == 0)).any(axis=1)
        inside = (within & (observed == 1)).any(axis=1)
        if parent is None:
            outside |= grid.ravel()[cells] == 0
            inside |= grid.ravel()[cells] == 1
        else:
            prev_cells, prev_label, prev_radii, prev_xyz = parent
            pi = np.searchsorted(prev_cells, np.ravel_multi_index((coords // 2).T, size // 2))
            separation = np.linalg.norm(centers - prev_xyz[pi], axis=1)
            # Overlap connects two certified open balls, even on a concave ROI.
            linked = (radii > 0) & (prev_radii[pi] > 0) & (separation < radii + prev_radii[pi])
            outside |= linked & (prev_label[pi] == 0)
            inside |= linked & (prev_label[pi] == 1)
        if (outside & inside).any():
            raise RuntimeError('Opposite navigation anchors in a verified wall-free R3 ball')
        label = np.full(len(cells), UNKNOWN, np.int8)
        label[outside], label[inside] = 0, 1
        covered = np.linalg.norm(points[rows] - centers[inverse], axis=1) < radii[inverse]
        count = np.bincount(inverse[covered], minlength=len(cells))
        ask = np.flatnonzero((label == UNKNOWN) & (count >= 8))
        rejected = np.zeros(len(cells), dtype=bool)
        if len(ask):
            t = perf_counter()
            found, detail = query_rays(centers[ask])
            label[ask] = found
            rejected[ask] = found == UNKNOWN
            stats['representative_queries'] += len(ask)
            stats['representative_seconds'] += perf_counter() - t
            stats['representative_unknown'] += int(np.count_nonzero(found == UNKNOWN))
            stats['representative_conflicts'] += int(detail['conflicting_paths'])
        direct = covered & (label[inverse] != UNKNOWN)
        assigned[rows[direct]] = label[inverse[direct]]
        completed = int(direct.sum())
        stats['direct_points'] += completed
        # Unresolved representative queries never label their neighbourhood.
        # Their covered rows go directly to independent per-Gaussian fallback.
        rest = ~direct & ~(covered & rejected[inverse])
        refine = np.zeros_like(rest)
        if depth < max_depth and rest.any():
            # Split only if there is enough work per occupied child. Point order,
            # results and refinement choices do not depend on a wall-clock timer.
            remaining = np.bincount(inverse[rest], minlength=len(cells))
            upper = points[rows[rest]] >= centers[inverse[rest]]
            octant = 4 * upper[:, 0] + 2 * upper[:, 1] + upper[:, 2]
            children = np.unique(8 * inverse[rest] + octant) // 8
            child_count = np.bincount(children, minlength=len(cells))
            refine = rest & (remaining[inverse] >= np.maximum(4, 2 * child_count[inverse]))
        pending = int(refine.sum())
        stats['levels'].append(dict(depth=depth, edge=h, occupied=len(cells), queries=len(rows),
                                    direct=completed, subdivide_points=pending, seconds=perf_counter()-tick))
        log(f"Adaptive voxels depth={depth} occupied={len(cells):,} direct={completed:,} "
            f"cumulative={stats['direct_points']:,}/{len(points):,} subdivide={pending:,}")
        parent = cells, label, radii, centers
        rows = rows[refine]

    residual = np.flatnonzero(assigned == UNKNOWN)
    stats['fallback_points'] = len(residual)
    log(f"Adaptive voxel lookup {stats['direct_points']:,}/{len(points):,}; short-ray fallback {len(residual):,}")
    if len(residual):
        t = perf_counter()
        found, detail = query_rays(points[residual])
        assigned[residual] = found
        stats['fallback_seconds'] = perf_counter() - t
        stats['fallback_details'] = detail
        stats['conflicting_paths'] = int(detail['conflicting_paths'])
    stats['classified_outside'] = int(np.sum(assigned == 0))
    stats['classified_inside'] = int(np.sum(assigned == 1))
    stats['classified_uncertain'] = int(np.sum(assigned == UNKNOWN))
    stats['seconds'] = perf_counter() - started
    return assigned, stats
