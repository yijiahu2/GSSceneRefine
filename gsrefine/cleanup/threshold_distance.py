"""Distance threshold queries on occupied cells, with exact fallback."""
import os
import numpy as np

from gsrefine.cleanup.stage2_runtime import measured


def exact_requested():
    return os.environ.get('GSREFINE_EXACT_DISTANCES', '') == '1'


def exact(scene, o3d, xyz):
    query = o3d.core.Tensor(np.ascontiguousarray(xyz, dtype=np.float32))
    value = np.asarray(scene.compute_distance(query).numpy(), dtype=np.float32).ravel()
    if len(value) != len(xyz) or not np.isfinite(value).all():
        raise ValueError('Invalid triangle distance')
    return value


@measured('main_distance_threshold')
def classify_distance(scene, o3d, centers, threshold, mesh_bounds, chunk, adaptive, log):
    distances = np.full(len(centers), np.nan, np.float32)
    far = np.zeros(len(centers), bool)
    bulk_near = bulk_far = 0
    guard = 64 * np.finfo(np.float32).eps * max(1., np.abs(mesh_bounds).max(),
                                               abs(centers.min(initial=0.)),
                                               abs(centers.max(initial=0.)), threshold)
    for begin in range(0, len(centers), max(chunk, 240000)):
        xyz = centers[begin:begin + max(chunk, 240000)].astype(np.float32).astype(np.float64)
        pending = np.arange(len(xyz))
        for width in (2 * threshold, threshold * .5):
            if not adaptive or len(pending) < 16 or width <= guard:
                break
            cells, inv, counts = np.unique(np.floor(xyz[pending] / width), axis=0,
                                           return_inverse=True, return_counts=True)
            inv = inv.ravel()
            dense = counts >= 8
            if not dense.any():
                break
            mid = ((cells[dense] + .5) * width).astype(np.float32)
            d = exact(scene, o3d, mid).astype(np.float64)
            radius = np.sqrt(3.) * width / 2 + guard
            near_cell = np.zeros(len(cells), bool)
            far_cell = near_cell.copy()
            near_cell[dense] = d + radius <= threshold
            far_cell[dense] = d - radius > threshold
            near, over = near_cell[inv], far_cell[inv]
            far[begin + pending[over]] = True
            bulk_near += int(near.sum())
            bulk_far += int(over.sum())
            pending = pending[~(near | over)]
        for pos in range(0, len(pending), chunk):
            rows = pending[pos:pos+chunk]
            values = exact(scene, o3d, xyz[rows])
            distances[begin + rows] = values
            far[begin + rows] = values.astype(np.float64) > threshold
        log(f'Distance threshold {begin + len(xyz):,}/{len(centers):,}; cell decisions {bulk_near+bulk_far:,}')
    return distances, far, dict(cell_near=bulk_near, cell_far=bulk_far,
                                exact=len(far)-bulk_near-bulk_far,
                                distance_main_nan='cell decision; exact distance not evaluated')
