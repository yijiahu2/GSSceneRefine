"""Audit ORIGINAL global Poisson x VERIFIED FINAL R3 ROI as a 3-D barrier.

No edits of original mesh, ROI, or Gaussian. No Gaussian deletion.

1. Recompute Stage-1 R3 1x *continuous* implicit field with its original inputs
   and validate against the saved handoff (not initial ROI polygons).
2. Find ROI/ORIGINAL-Poisson intersections ON THE ORIGINAL TRIANGLES by
   actual field root finding on shared mesh edges (not the old clipped PLY).
3. Water-flood the true final R3 bounded volume via a grid graph. For each
   possible crossing between adjacent voxel CENTERS, test the real segment
   against ORIGINAL Poisson triangles, not rasterized Poisson voxels.
4. Report leaks/topology ambiguities; NEVER claim certified watertightness.

IMPORTANT: The nonlinear R3 1x field does not have an analytic, polynomial
surface representation. Root positions are numerical to a reported tolerance;
not exact arithmetic. A grid flood can witness a path but cannot mathematically
prove the absence of arbitrarily thin paths. An R3 ROI is not a solid mesh.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

VERSION = 'final_R3_original_global_poisson_region_graph_v3_window_20261112'


def log(msg):
    print('[final-r3-barrier] ' + str(msg), flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def native(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [native(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): native(v) for k, v in obj.items()}
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    raise TypeError(type(obj).__name__)


def atomic_json(path, contents):
    path = Path(path)
    handle, tmp = tempfile.mkstemp(prefix='.' + path.stem, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(handle, 'w', encoding='utf8') as out:
            json.dump(native(contents), out, indent=2, ensure_ascii=False, allow_nan=False)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def stats(array):
    x = np.asarray(array, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x):
        return {'count': 0}
    return {'count': int(len(x)), 'min': float(x.min()), 'median': float(np.median(x)),
            'p90': float(np.quantile(x, .9)), 'p95': float(np.quantile(x, .95)),
            'p99': float(np.quantile(x, .99)), 'max': float(x.max())}


def eval_r3(fn, pts, batch=2048, name=None):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1,3)
    result = np.empty(len(pts), dtype=np.float64)
    for lo in range(0, len(pts), batch):
        hi = min(lo+batch, len(pts))
        result[lo:hi] = np.asarray(fn(pts[lo:hi]),dtype=np.float64).reshape(-1)
        if name and (lo == 0 or hi == len(pts) or lo // batch % 100 == 0):
            log(f'{name}: {hi:,} / {len(pts):,}')
    if not np.isfinite(result).all():
        raise ValueError('Final R3 function returned nonfinite margins')
    return result


def largest_original_component(original_mesh):
    """Extract from ORIGINAL Poisson without altering its triangles or vertices."""
    import trimesh
    n = len(original_mesh.faces)
    components = trimesh.graph.connected_components(
        original_mesh.face_adjacency, nodes=np.arange(n), min_len=1)
    sizes = sorted((len(c) for c in components), reverse=True)
    biggest = np.asarray(max(components, key=len), dtype=np.int64)
    biggest.sort()
    mesh = original_mesh.submesh([biggest], append=True, repair=False)
    return mesh, {'global_original_faces': int(n),
                  'original_components': int(len(components)),
                  'largest_source_component_faces': int(len(mesh.faces)),
                  'other_global_component_faces_ignored': int(n-len(mesh.faces)),
                  'largest_global_source_face_fraction': float(len(mesh.faces)/n),
                  'top_component_sizes': sizes[:15],
                  'policy': 'Use ONLY largest original global-Poisson connected component. '
                            'Other components excluded, following main-only experiment.'}


def roots_on_crossing_edges(xyz, edges, fa, fb, field, *, iterations, eval_batch):
    """Bisection ON ORIGINAL mesh edges, using REAL nonlinear final-R3 function."""
    a = xyz[edges[:,0]]
    b = xyz[edges[:,1]]
    if np.any(fa*fb >= 0):
        raise ValueError('Expected strictly opposite signs at both endpoints')
    left, right = np.zeros(len(edges), np.float64), np.ones(len(edges), np.float64)
    sign_a = fa > 0
    for _ in range(iterations):
        mid = (left+right)/2
        fm = eval_r3(field, a+(b-a)*mid[:,None], batch=eval_batch)
        same = (fm > 0) == sign_a
        left = np.where(same, mid, left)
        right = np.where(same, right, mid)
    t = (left+right)/2
    roots = a+(b-a)*t[:,None]
    residuals = np.abs(eval_r3(field, roots, batch=eval_batch))
    interval = np.linalg.norm(b-a, axis=1)*(right-left)
    return roots, residuals, interval


def field_edge_crossings(mesh, field, *, iterations=22, batch=2048,
                         near_tolerance=1e-5, output_dir=None, check_same=4000):
    """Numerical continuous-R3/ORIGINAL-Poisson intersection segments.

    With nonlinear R3 there may be multiple intersections on one edge even
    if endpoint signs match; count sampled examples of this aliasing and
    DO NOT assert exhaustiveness or mathematical exactness.
    """
    verts = np.asarray(mesh.vertices, np.float64)
    faces = np.asarray(mesh.faces, np.int64)
    margins = eval_r3(field, verts, batch=batch, name='Original global-Poisson vertices vs final R3')
    edge_idx = np.asarray(mesh.edges_unique, dtype=np.int64)
    ea, eb = margins[edge_idx[:,0]], margins[edge_idx[:,1]]
    cross = (ea < -near_tolerance) & (eb > near_tolerance) | (ea > near_tolerance) & (eb < -near_tolerance)
    near_endpoint = (np.abs(ea) <= near_tolerance) | (np.abs(eb) <= near_tolerance)
    selected = np.flatnonzero(cross)
    log(f'Full original source edges: {len(edge_idx):,}; strict R3-crossing edges: {len(selected):,}')
    roots = np.empty((0,3),dtype=np.float64)
    residual = np.empty(0,dtype=np.float64)
    interval = np.empty(0,dtype=np.float64)
    if len(selected):
        pieces = []
        residues = []
        widths = []
        for lo in range(0, len(selected), batch):
            pick = selected[lo:lo+batch]
            p, r, w = roots_on_crossing_edges(verts, edge_idx[pick], ea[pick], eb[pick],
                                              field, iterations=iterations, eval_batch=batch)
            pieces.append(p); residues.append(r); widths.append(w)
            if lo == 0 or lo+batch >= len(selected) or lo//batch % 10 == 0:
                log(f'Actual R3 root isolation: {min(lo+batch,len(selected)):,} / {len(selected):,}')
        roots=np.vstack(pieces)
        residual=np.concatenate(residues)
        interval=np.concatenate(widths)
    e2r=np.full(len(edge_idx),-1,dtype=np.int32)
    e2r[selected]=np.arange(len(selected),dtype=np.int32)
    face_edge_ids=np.asarray(mesh.faces_unique_edges,dtype=np.int64)
    selected_root_indices=e2r[face_edge_ids]
    count_per_face=(selected_root_indices >= 0).sum(axis=1)
    exactly2=np.flatnonzero(count_per_face==2)
    endpoints = np.empty((len(exactly2),2,3),np.float64)
    for i, face_id in enumerate(exactly2):
        p = selected_root_indices[face_id]
        endpoints[i]=roots[p[p>=0]]
    midpoint_residual = (np.abs(eval_r3(field, endpoints.mean(axis=1), batch=batch, name='R3 intersection segment midpoints'))
                         if len(endpoints) else np.empty(0, dtype=float))
    if output_dir is not None:
        if len(roots):
            import trimesh
            trimesh.points.PointCloud(roots).export(str(Path(output_dir)/'intersection_points.ply'))
        np.savez_compressed(Path(output_dir)/'global_poisson_final_r3_intersection.npz',
                            segment_endpoints=endpoints.astype('f4'),
                            global_component_face_ids=exactly2.astype('i4'),
                            edge_intersection_xyz=roots.astype('f4'),
                            edge_intersection_residual=residual.astype('f4'),
                            intersected_original_edge_ids=selected.astype('i4'))
        with (Path(output_dir)/'intersection_segments.csv').open('w',newline='',encoding='utf8') as stream:
            writer=csv.writer(stream)
            writer.writerow(['component_face_id','ax','ay','az','bx','by','bz'])
            for fi, segment in zip(exactly2,endpoints):
                writer.writerow([int(fi),*segment[0],*segment[1]])
    # Probe possible missed double roots using same-sign endpoints, but do not
    # assume a finite sample certifies no such crossing anywhere.
    same_idx=np.flatnonzero(~cross & ~near_endpoint)
    if check_same and len(same_idx):
        probe=same_idx[np.linspace(0,len(same_idx)-1,min(check_same,len(same_idx)),dtype=np.int64)]
        mids=(verts[edge_idx[probe,0]]+verts[edge_idx[probe,1]])*.5
        mids_f=eval_r3(field,mids,batch=batch,name='Same-sign edge midpoint alias audit')
        missed=np.count_nonzero((mids_f>0)!=(ea[probe]>0))
    else:
        probe=np.empty(0,dtype=np.int64);missed=0
    return {'global_component_vertices': int(len(verts)),
            'global_component_faces':int(len(faces)),
            'global_component_unique_edges':int(len(edge_idx)),
            'vertices_inside_final_r3':int(np.sum(margins>=0)),
            'strict_crossing_original_edges':int(len(selected)),
            'near_R3_zero_original_edge_endpoints':int(np.sum(near_endpoint)),
            'original_faces_with_two_intersected_edges':int(len(exactly2)),
            'original_faces_with_one_intersected_edge':int(np.sum(count_per_face==1)),
            'original_faces_with_three_intersected_edges':int(np.sum(count_per_face==3)),
            'root_field_abs_residual':stats(residual),
            'root_spatial_bracket_width':stats(interval),
            'linear_segment_midpoint_R3_residual':stats(midpoint_residual),
            'segments_with_midpoint_residual_over_1e_minus_3':int(np.sum(midpoint_residual>0.001)),
            'same_sign_edges_midpoint_probed':int(len(probe)),
            'same_sign_midpoint_sign_change_samples':int(missed),
            'intersection_method':'Per ORIGINAL source edge, actual verified final R3 field bisection; '
                                  'faces with exactly 2 crossing edges form intersection-line segments',
            'not_certified':'Final-R3 implicit function is nonlinear; same-sign double crossings, tangencies, '
                             'and subtriangle loops may not be detected. Roots have finite numerical residuals. '
                             'This is NOT mathematically exact Boolean intersection.'}


def make_scene(source_mesh):
    import open3d as o3d
    v=np.asarray(source_mesh.vertices, np.float64)
    f=np.asarray(source_mesh.faces, np.int32)
    lm=o3d.geometry.TriangleMesh()
    lm.vertices=o3d.utility.Vector3dVector(v)
    lm.triangles=o3d.utility.Vector3iVector(f)
    scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(lm))
    return scene,o3d


def _bbox_stage1_roi(roi):
    """Conservative rectangular bound, NOT a replacement for actual final R3."""
    from scipy.optimize import linprog
    a=[]; b=[]
    for view in roi.views:
        for i in range(2):
            row=np.asarray(view.U[i],float)
            shift=float(row @ view.origin)
            mx=float(np.max(view.polygon[:,i])); mn=float(np.min(view.polygon[:,i]))
            a += [row,-row]; b += [mx+shift,-mn-shift]
    lo=np.empty(3);hi=np.empty(3)
    for i in range(3):
        e=np.eye(3)[i]
        r0=linprog(e,A_ub=np.asarray(a),b_ub=np.asarray(b),bounds=[(None,None)]*3,method='highs')
        r1=linprog(-e,A_ub=np.asarray(a),b_ub=np.asarray(b),bounds=[(None,None)]*3,method='highs')
        if not r0.success or not r1.success:
            raise RuntimeError('ROI projected polygon prisms do not form bounded 3D region')
        lo[i]=r0.fun;hi[i]=-r1.fun
    return lo,hi


def bounding_window(initial_roi, r3_gaussian_xyz, padding):
    a,b=_bbox_stage1_roi(initial_roi)
    r=np.asarray(r3_gaussian_xyz, float)
    lo=np.minimum(a,r.min(axis=0));hi=np.maximum(b,r.max(axis=0))
    p=max(float(padding), .03*float(np.linalg.norm(hi-lo)))
    return lo-p,hi+p, {'initial_polygon_prism_enclosing_bbox':[a,b],
                        'final_r3_gaussian_enclosing_bbox':[r.min(axis=0),r.max(axis=0)],
                        'padding':p,
                        'note':'AABB is only a sample-window; actual R3 membership comes from the verified continuous field'}




# Stage-2 audit-window implementation V3. The installer embeds this code directly
# into tools/audit_final_r3_global_poisson_barrier.py; this file is not imported.

def _r3_shell_probe(field, lo, hi, *, target_voxel, max_cells, batch=2048):
    """Check exactly the outermost voxel centres of the production grid.

    Counts touch points once per unique shell centre.  Per-side counts may
    overlap at edges/corners; this is intentional and is not a Gaussian mask.
    """
    low=np.asarray(lo, dtype=np.float64)
    high=np.asarray(hi, dtype=np.float64)
    h, shape=grid_spec(low,high,target_voxel,max_cells)
    shape=tuple(int(n) for n in shape)
    boundary=np.zeros(shape,dtype=bool)
    boundary[0,:,:]=True;boundary[-1,:,:]=True
    boundary[:,0,:]=True;boundary[:,-1,:]=True
    boundary[:,:,0]=True;boundary[:,:,-1]=True
    ijk=np.argwhere(boundary)
    xyz=low+(ijk.astype(np.float64)+0.5)*h
    inside=eval_r3(field,xyz,batch=batch)>0
    hits=ijk[inside]
    sides={}
    for axis,letter in enumerate('xyz'):
        sides[letter+'_min']=int(np.count_nonzero(hits[:,axis]==0))
        sides[letter+'_max']=int(np.count_nonzero(hits[:,axis]==shape[axis]-1))
    return {'voxel':float(h),'shape':list(shape),
            'shell_nodes_checked':int(len(ijk)),
            'r3_boundary_touch_count':int(inside.sum()),'touches_by_side':sides}


def complete_window_for_levels(field, lo, hi, *, levels, batch=2048,
                               max_rounds=12, growth_voxels=3.0,
                               max_voxel_scale=1.25, logger=None):
    """Expand only the *numerical* AABB, preserving ROI and Poisson inputs.

    Both levels share the same final AABB. Normal cases retain bit-identical
    bounds. A grid cap may enlarge actual voxel size: prevent >25% upsampling
    deterioration relative to the original scene-specific grid specification.
    Fail closed if the window is unresolved within the expansion budget.
    """
    lower=np.asarray(lo,dtype=np.float64).copy()
    upper=np.asarray(hi,dtype=np.float64).copy()
    if (lower.shape!=(3,) or upper.shape!=(3,) or
            not np.isfinite(lower).all() or not np.isfinite(upper).all() or
            not np.all(upper>lower)):
        raise ValueError('Invalid initial 3D sampling bounds')
    if not levels or len({str(x[0]) for x in levels})!=len(levels):
        raise ValueError('Sampling levels must be nonempty and uniquely named')
    if not (1<=int(max_rounds)<=64 and 1.0<=float(growth_voxels)<=12.0
            and 1.0<=float(max_voxel_scale)<=2.0):
        raise ValueError('Invalid sampling-window expansion safeguards')
    initial_lower=lower.copy();initial_upper=upper.copy()
    base={name:grid_spec(lower,upper,float(h),int(cap))[0]
          for name,h,cap in levels}
    trace=[]
    for step in range(int(max_rounds)+1):
        grow_lo=np.zeros(3,dtype=np.float64)
        grow_hi=np.zeros(3,dtype=np.float64)
        results={}
        for name,target,cap in levels:
            info=_r3_shell_probe(field,lower,upper,target_voxel=float(target),
                                  max_cells=int(cap),batch=batch)
            h=float(info['voxel'])
            if h>base[name]*float(max_voxel_scale)+1e-10:
                raise RuntimeError(
                    f'{name} grid resolution would degrade from {base[name]:.6g} to '
                    f'{h:.6g} (> {max_voxel_scale:.3g}x). Increase max_cells '
                    'instead of silently weakening spatial validation.')
            results[name]=info
            for axis,letter in enumerate('xyz'):
                if info['touches_by_side'][letter+'_min']:
                    grow_lo[axis]=max(grow_lo[axis],float(growth_voxels)*h)
                if info['touches_by_side'][letter+'_max']:
                    grow_hi[axis]=max(grow_hi[axis],float(growth_voxels)*h)
        touched=sum(x['r3_boundary_touch_count'] for x in results.values())
        trace.append({'attempt':step+1,'low':lower.tolist(),
                      'high':upper.tolist(),'levels':results,'total_touches':touched})
        if logger:
            logger('3D sampling window: '+', '.join(
                f'{name} boundary-R3={results[name]["r3_boundary_touch_count"]}'
                f' (h={results[name]["voxel"]:.6f})'
                for name,_,_ in levels)+f'; expansion round={step}')
        if touched==0:
            return lower,upper,{
                'schema':'GSSceneRefine.window_cover.v3',
                'status':'COMPLETE_SAMPLED_WINDOW',
                'expanded':bool(step),'expansion_rounds':int(step),
                'initial_low':initial_lower.tolist(),
                'initial_high':initial_upper.tolist(),
                'final_low':lower.tolist(),'final_high':upper.tolist(),
                'initial_voxel_sizes':{k:float(v) for k,v in base.items()},
                'levels':results,'trace':trace,
                'scope':'Numerical sampling box only; R3 and Poisson unchanged',
                'limitation':'Zero sampled shell contacts is not an analytic proof of full ROI coverage.'}
        if step==int(max_rounds):
            raise RuntimeError(
                f'R3 still touches the sampling box after {max_rounds} expansions: '
                +str({name:info['touches_by_side'] for name,info in results.items()}))
        lower-=grow_lo;upper+=grow_hi
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise RuntimeError('Sampling window expanded to nonfinite values')


def grid_spec(lo,hi,desired,max_cells):
    ext=np.asarray(hi)-np.asarray(lo)
    if np.any(ext<=0) or desired<=0 or max_cells<200:
        raise ValueError('Invalid grid window or voxel controls')
    h=max(float(desired),float(np.cbrt(np.prod(ext)/max_cells)))
    dims=np.maximum(2,np.ceil(ext/h).astype(np.int64))
    while int(np.prod(dims))>int(max_cells):
        h*=1.02;dims=np.maximum(2,np.ceil(ext/h).astype(np.int64))
    return float(h),tuple(int(i) for i in dims)


def grid_centers(origin,h,shape):
    ids=np.indices(shape,dtype=np.int32).reshape(3,-1).T
    return np.asarray(origin,dtype=float)+(ids+.5)*h


def neighbour_pairs(shape, inside):
    inside=np.asarray(inside,dtype=bool).reshape(shape)
    idx=np.arange(inside.size,dtype=np.int32).reshape(shape)
    first=[];second=[]
    for axis in range(3):
        s0=[slice(None)]*3;s1=[slice(None)]*3
        s0[axis]=slice(0,-1);s1[axis]=slice(1,None)
        a=idx[tuple(s0)].reshape(-1);b=idx[tuple(s1)].reshape(-1)
        good=inside.reshape(-1)[a]&inside.reshape(-1)[b]
        first.append(a[good]);second.append(b[good])
    return np.concatenate(first),np.concatenate(second)


def link_r3_filter(p,q,field,*, batch=2048,samples=3):
    # This tests continuous actual R3 along LINKS at interior sample points;
    # cannot guarantee no tiny re-entry between samples.
    if samples<1: raise ValueError('samples must be positive')
    allow=np.ones(len(p),bool)
    for t in np.linspace(0,1,samples+2)[1:-1]:
        indices=np.flatnonzero(allow)
        if not len(indices): break
        m=eval_r3(field,(1-t)*p[indices]+t*q[indices],batch=batch)
        allow[indices]=m>0
    return allow


def _ray_hit_free(scene, o3d, origins, targets, *, eps, batch):
    ok=np.empty(len(origins),dtype=bool)
    for a in range(0,len(origins),batch):
        b=min(a+batch,len(origins))
        vec=np.asarray(targets[a:b]-origins[a:b],dtype=np.float64)
        dist=np.linalg.norm(vec,axis=1)
        direction=vec/np.maximum(dist[:,None],1e-12)
        start=origins[a:b] + direction*eps
        rays=np.c_[start,direction].astype(np.float32)
        result=scene.cast_rays(o3d.core.Tensor(np.ascontiguousarray(rays)))
        hit=np.asarray(result['t_hit'].numpy(),float)
        ok[a:b]=np.isposinf(hit)|(np.isfinite(hit)&(hit>=dist-2*eps))
        if a == 0 or b == len(origins) or a//batch%20==0:
            log(f'Exact ORIGINAL source triangle link-crossing tests: {b:,} / {len(origins):,}')
    return ok


def _connected_components(n, inside, src, dst):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    ids=np.flatnonzero(inside)
    lookup=np.full(n,-1,dtype=np.int32)
    lookup[ids]=np.arange(len(ids),dtype=np.int32)
    a=lookup[src];b=lookup[dst]
    data=np.ones(len(a)*2,dtype=np.uint8)
    adj=csr_matrix((data,(np.r_[a,b],np.r_[b,a])),shape=(len(ids),len(ids)))
    count, cl=connected_components(adj,directed=False)
    labels=np.full(n,-1,dtype=np.int32)
    labels[ids]=cl
    return labels, int(count)


def _grid_shell(mask, shape):
    from scipy.ndimage import binary_erosion,generate_binary_structure
    interior=binary_erosion(np.asarray(mask,bool).reshape(shape),
                            structure=generate_binary_structure(3,1),border_value=0)
    return (np.asarray(mask,bool).reshape(shape)&~interior).reshape(-1)


def choose_top_bottom_seeds(inside,xyz,shape,up,quantile=.90):
    boundary=np.flatnonzero(_grid_shell(inside,shape))
    if len(boundary)<12:
        raise ValueError('R3 ROI too small to define boundary sampling seeds')
    heights=xyz[boundary]@up
    a=np.quantile(heights,quantile);b=np.quantile(heights,1-quantile)
    top=boundary[heights>=a];bottom=boundary[heights<=b]
    return top,bottom,{'ROI_shell_cell_count':int(len(boundary)),
                      'upper_quantile':float(a),'lower_quantile':float(b),
                      'top_seed_count':int(len(top)),'bottom_probe_count':int(len(bottom)),
                      'hypothesis':'Highest R3 shell = physically exterior. Lowest R3 shell = lower-side PROBE, '
                                   'not proven occupied rock.'}


def _path_witness(n,src,dst,start,end,points):
    from collections import deque
    from scipy.sparse import csr_matrix
    A=csr_matrix((np.ones(2*len(src),np.uint8),
                  (np.r_[src,dst],np.r_[dst,src])),shape=(n,n))
    is_end=np.zeros(n,bool);is_end[end]=True
    seen=np.zeros(n,bool);seen[start]=True
    predecessor=np.full(n,-1,dtype=np.int32)
    queue=deque(map(int,start))
    finish=-1
    while queue:
        u=queue.popleft()
        if is_end[u]: finish=u;break
        for v in A.indices[A.indptr[u]:A.indptr[u+1]]:
            v=int(v)
            if not seen[v]:
                seen[v]=True;predecessor[v]=u;queue.append(v)
    if finish<0:return np.empty((0,3),float)
    path=[]
    while finish>=0:
        path.append(finish);finish=int(predecessor[finish])
    return np.asarray(points[path[::-1]],float)


def flood_test(original_main,roi_fn,up,lo,hi,*,name,h_target,max_cells,
               output_dir,batch=2048,link_samples=3, ray_batch=30000,
               seed_quantile=.90):
    h,shape=grid_spec(lo,hi,h_target,max_cells)
    log(f'{name}: grid shape={shape}, node count={np.prod(shape):,}, voxel={h:.6f}')
    xyz=grid_centers(lo,h,shape)
    margins=eval_r3(roi_fn,xyz,batch=batch,name=f'{name} final R3 membership')
    inside=margins>0
    if int(inside.sum())<40:
        raise ValueError('R3 ROI too small at current diagnostic resolution')
    # Fail open/nonconclusive when R3 ROI touches artificial audit window.
    v=inside.reshape(shape)
    touches=int(v[0].sum()+v[-1].sum()+v[:,0].sum()+v[:,-1].sum()+v[:,:,0].sum()+v[:,:,-1].sum())
    if touches:
        raise RuntimeError(f'{name} sampled box still intersects R3 at {touches} outer nodes; '
                           'refuse to construct an incomplete region graph')
    scene,o3d=make_scene(original_main)
    # Only a NUMERICAL touching exclusion; never old 1.25*h / 2*h
    # over-wide Gaussian deletion protection band.
    distance=np.empty(len(xyz),np.float32)
    for i in range(0,len(xyz),50000):
        j=min(len(xyz),i+50000)
        result=scene.compute_distance(o3d.core.Tensor(np.ascontiguousarray(xyz[i:j],dtype=np.float32)))
        distance[i:j]=np.asarray(result.numpy(),float)
    skin=max(1e-6,h*.015)
    passable=inside & (distance>skin)
    src,dst=neighbour_pairs(shape,passable)
    before=len(src)
    good=link_r3_filter(xyz[src],xyz[dst],roi_fn,batch=batch,samples=link_samples)
    src=src[good];dst=dst[good]
    r3_rejected=before-len(src)
    ray_ok=_ray_hit_free(scene,o3d,xyz[src],xyz[dst],eps=min(max(1e-6,h*1e-4),h*.005),batch=ray_batch)
    poisson_blocked=int((~ray_ok).sum())
    src=src[ray_ok];dst=dst[ray_ok]
    labels,ncomponents=_connected_components(len(xyz),passable,src,dst)
    # True spatial components are the evidence; arbitrary top/bottom probe
    # connectivity is NOT a separation criterion.  Upper-shell reachability
    # remains a diagnostic only, never a pass/fail test.
    top, _, seeds=choose_top_bottom_seeds(passable,xyz,shape,up,quantile=seed_quantile)
    top_l=set(map(int,np.unique(labels[top]))) - {-1}
    reach_top=np.isin(labels,list(top_l))&passable
    status='REGION_GRAPH_SAMPLED_NOT_CERTIFIED'
    np.savez_compressed(Path(output_dir)/f'{name}_connectivity.npz',
                        origin=np.asarray(lo),voxel=np.float64(h),shape=np.asarray(shape,dtype=np.int32),
                        final_r3_inside=inside.reshape(shape),
                        passable=passable.reshape(shape),
                        reachable_from_top=reach_top.reshape(shape),
                        components=labels.reshape(shape))
    return {'status':status,'grid':{'shape':list(shape),'voxel':h,'all_cells':int(len(xyz)),
                                  'actual_R3_cells':int(inside.sum()),'passable_cells':int(passable.sum()),
                                  'near_original_mesh_numeric_excluded':int((inside&~passable).sum()),
                                  'r3_boundary_touch_count':touches},
            'barrier':{'candidate_neighbour_links':int(before),
                       'links_rejected_by_actual_R3_field_samples':int(r3_rejected),
                       'original_global_poisson_triangle_blocked_links':poisson_blocked,
                       'remaining_walkable_links':int(len(src))},
            'connectivity':{'connected_regions':int(ncomponents),
                            'upper_shell_connected_regions':sorted(top_l),
                            'reachable_from_upper':int(reach_top.sum())},
            'seed_definition':seeds,
            'limitations':'Sampled space components do not mathematically certify an exact partition. '
                          'The upper shell is not automatically a validated exterior seed. '
                          'R3 segment membership is probed at finitely many positions.'}


def _sample_gaussian_comparison(mask_path,ply,grid_paths,roi_fn,output,batch=2048,sample=8000):
    import numpy as np
    with np.load(mask_path,allow_pickle=False) as masks:
        far=np.asarray(masks['over_threshold'],bool)
        distance=np.asarray(masks['distance_main'],np.float64)
        old=np.asarray(masks.get('occupancy_label',np.zeros(len(far),np.int8)))
    if len(far)!=ply.vertex_count:
        raise ValueError('Masks row order/count differs from final ROI Gaussian PLY')
    ix=np.flatnonzero(far)
    if not len(ix):return {'far':0,'sample':0}
    rng=np.random.default_rng(101110)
    bins=[0,.2,.3,.5,1,2,3,np.inf]
    quota=max(1,int(sample/7))
    pick=np.concatenate([rng.choice(group,size=min(len(group),quota),replace=False)
                         for lo,hi in zip(bins[:-1],bins[1:])
                         if len(group:=ix[(distance[ix]>=lo)&(distance[ix]<hi)])])
    pick=np.unique(pick)
    if len(pick)>sample:pick=np.sort(rng.choice(pick,sample,replace=False))
    pts=np.asarray(ply.positions()[pick],np.float64)
    true_margin=eval_r3(roi_fn,pts,batch=batch,name='Sampled final-R3 Gaussian membership')
    names=list(grid_paths)
    labels=[]
    for name,path in grid_paths.items():
        with np.load(path,allow_pickle=False) as z:
            origin=np.asarray(z['origin']);h=float(z['voxel']); shape=np.asarray(z['shape'],int)
            exterior=z['reachable_from_top']
            membership=z['final_r3_inside']
        cell=np.floor((pts-origin)/h).astype(int)
        valid=np.all((cell>=0)&(cell<shape),axis=1)
        out=np.full(len(pts),-1,dtype=np.int8)
        q=cell[valid].T
        out[valid]=np.where(membership[tuple(q)],exterior[tuple(q)].astype(np.int8),-2)
        labels.append(out)
    with (Path(output)/'sampled_far_gaussians.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f)
        w.writerow(['row_in_original_roi_ply','x','y','z','exact_original_poisson_distance',
                    'previous_V2_class','actual_final_R3_margin',*[f'{n}_exterior_reachable' for n in names]])
        for i,pt in enumerate(pts):
            w.writerow([int(pick[i]),*pt,float(distance[pick[i]]),int(old[pick[i]]),
                        float(true_margin[i]),*[int(v[i]) for v in labels]])
    return {'candidate_over_threshold_population':int(len(ix)),
            'sampled':int(len(pick)),
            'note':'Sampling stratified by distance; label is approximate grid-cell lookup, '
                   'NOT a point-wise certified internal/external classification. '
                   'This is NOT a new Gaussian deletion mask.'}


def run(args):
    import trimesh
    from gsrefine.cleanup.sdf_common import _load_stage1_manifest
    from gsrefine.geometry.r3_direct_poisson_clip import _restore_r3_field,_source_r3_handoff
    from gsrefine.roi.core import ROI,GaussianPLY
    output=Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refuse to overwrite earlier experiment results: '+str(output))
    output.mkdir(parents=True,exist_ok=True)
    mf,upstream,inputs,run_dir=_load_stage1_manifest(args.manifest)
    p=Path(args.preseal_report).resolve(strict=True)
    pre=json.loads(p.read_text('utf8'))
    if pre.get('status') not in ('ELIGIBLE_FOR_SEAL_TRIAL','PRESEAL_VALIDATION') and not pre.get('preseal_eligible'):
        log('Note: prior preseal report status differs; audit still uses original global source independently')
    if pre.get('stage1_manifest') and Path(pre['stage1_manifest']).resolve()!=Path(mf):
        raise ValueError('Preseal provenance does not match Stage-1 manifest')
    declared=(pre.get('sources',{}).get('source_hashes',{}).get('stage1_poisson_sha256'))
    if declared and declared!=sha256_file(inputs['poisson_mesh']):
        raise ValueError('Original full-scene Poisson SHA differs from preseal report')
    roi_raw=json.loads(Path(inputs['roi_json']).read_text('utf8'))
    if roi_raw.get('annotation_metadata',{}).get('orientation_verified') is not True:
        raise ValueError('Missing verified Stage-1 orientation annotation')
    up=np.asarray(roi_raw['annotation_metadata']['up_vector'],float)
    if up.shape!=(3,) or not np.isfinite(up).all() or np.linalg.norm(up)==0:
        raise ValueError('Invalid verified orientation vector')
    up/=np.linalg.norm(up)
    record=_source_r3_handoff(Path(run_dir),upstream)
    log('Reconstructing *FINAL* R3 continuous field from verified Stage-1 code/record ...')
    orig_global,r3_fn,replay=_restore_r3_field(
        inputs=inputs,upstream=upstream,run_dir=Path(run_dir),
        audit_record=record,audit_samples=args.audit_samples)
    log('Original global Poisson recovered; selecting ORIGINAL largest connected component ...')
    original_main,sel=largest_original_component(orig_global)
    result={'schema':'GSSceneRefine.final_R3_global_Poisson_barrier_audit.v2',
            'version':VERSION,'created_utc':datetime.now(timezone.utc).isoformat(),
            'status':'IN_PROGRESS_READ_ONLY_GEOMETRY_DIAGNOSTIC',
            'inputs':{'manifest':str(mf),'manifest_sha256':sha256_file(mf),
                      'actual_final_R3_field_handoff':str(record),'r3_sha256':sha256_file(record),
                      'original_full_scene_poisson':str(inputs['poisson_mesh']),
                      'original_poisson_sha256':sha256_file(inputs['poisson_mesh']),
                      'final_roi_gaussian':str(inputs['gaussian_ply']),
                      'excluded_local_r3_clipped_poisson':'poisson_roi_main.ply is NOT used as wall'},
            'source_component':sel,'r3_verified_replay':replay,
            'criteria':'FINAL restored R3 1x implicit ROI is the container; REAL original global source Poisson '
                       'triangles are the barrier, without reconstructing/capping/clipping any mesh.'}
    atomic_json(output/'roi_global_poisson_barrier_report.json',result)
    log('Computing actual FINAL R3 x original full-scene Poisson intersection (numerical roots)...')
    intersection=field_edge_crossings(original_main,r3_fn,iterations=args.root_iterations,
                                      batch=args.eval_batch,near_tolerance=args.edge_zero_epsilon,
                                      check_same=args.same_sign_samples,output_dir=output)
    result['intersection']=intersection
    atomic_json(output/'roi_global_poisson_barrier_report.json',result)
    if args.intersections_only:
        result['status']='INTERSECTION_ROOT_DIAGNOSTIC_COMPLETE_NOT_CERTIFIED_TOPOLOGY'
        atomic_json(output/'roi_global_poisson_barrier_report.json',result)
        return result
    ply=GaussianPLY(inputs['gaussian_ply'])
    initial=ROI.from_file(inputs['roi_json'])
    lo,hi,window=bounding_window(initial,ply.positions(),args.padding)
    planned_levels=[('coarse',args.coarse_voxel,args.coarse_max_cells),
                    ('fine',args.fine_voxel,args.fine_max_cells)]
    planned_levels=[x for x in planned_levels if args.only_scale in (x[0],'both')]
    initial_window_low=lo.tolist(); initial_window_high=hi.tolist()
    try:
        lo,hi,window_fit=complete_window_for_levels(
            r3_fn,lo,hi,levels=planned_levels,batch=args.eval_batch,
            max_rounds=getattr(args,'window_max_rounds',12),
            growth_voxels=getattr(args,'window_growth_voxels',3.0),
            max_voxel_scale=getattr(args,'window_max_voxel_scale',1.25),logger=log)
    except Exception as exc:
        result['status']='BLOCKED_INCOMPLETE_SAMPLING_WINDOW'
        result['diagnostic_window']={**window,
            'initial_low':initial_window_low,'initial_high':initial_window_high,
            'status':'BLOCKED_WINDOW_CHECK','error':type(exc).__name__+': '+str(exc)}
        atomic_json(output/'roi_global_poisson_barrier_report.json',result)
        raise
    result['diagnostic_window']={'low':lo.tolist(),'high':hi.tolist(),
                                  **window,'window_cover':window_fit}
    atomic_json(output/'roi_global_poisson_barrier_report.json',result)
    for name,desired,cap in [('coarse',args.coarse_voxel,args.coarse_max_cells),
                             ('fine',args.fine_voxel,args.fine_max_cells)]:
        if args.only_scale not in (name,'both'):continue
        log(f'Water-flood audit level: {name}')
        try:
            test=flood_test(original_main,r3_fn,up,lo,hi,
                            name=name,h_target=desired,max_cells=cap,output_dir=output,
                            batch=args.eval_batch,link_samples=args.roi_link_samples,
                            seed_quantile=args.top_quantile)
            result.setdefault('flood',{})[name]=test
            atomic_json(output/'roi_global_poisson_barrier_report.json',result)
        except Exception as e:
            result['status']='BLOCKED_FLOOD_AUDIT_ERROR'
            result.setdefault('errors',{})[name]=type(e).__name__+': '+str(e)
            atomic_json(output/'roi_global_poisson_barrier_report.json',result)
            raise
    result['status']='REGION_GRAPHS_COMPLETED_NUMERIC_NOT_CERTIFIED'
    if args.masks:
        grid_files={name:output/f'{name}_connectivity.npz' for name in result.get('flood',{})}
        result['gaussian_sample_comparison']=_sample_gaussian_comparison(
            Path(args.masks),ply,grid_files,r3_fn,output,batch=args.eval_batch,
            sample=args.sample_gaussians)
    result['interpretation']={
        'region_graph_only': True,
        'warning':'Sampled connectivity does not prove a closed separator; accept or reject only '
                  'after original-triangle region-parity and cross-scale checks. '
                  'Exterior seed still requires review.'}
    atomic_json(output/'roi_global_poisson_barrier_report.json',result)
    log('Saved: '+str(output/'roi_global_poisson_barrier_report.json'))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--preseal-report',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--intersections-only',action='store_true',help='First run intersection audit without voxel flood')
    p.add_argument('--masks',help='Optional V2 classification_masks.npz for Gaussian stratified comparison')
    p.add_argument('--sample-gaussians',type=int,default=7000)
    p.add_argument('--audit-samples',type=int,default=768)
    p.add_argument('--root-iterations',type=int,default=22)
    p.add_argument('--edge-zero-epsilon',type=float,default=1e-6)
    p.add_argument('--same-sign-samples',type=int,default=4000)
    p.add_argument('--eval-batch',type=int,default=2048)
    p.add_argument('--padding',type=float,default=.5)
    p.add_argument('--coarse-voxel',type=float,default=.30)
    p.add_argument('--fine-voxel',type=float,default=.17)
    p.add_argument('--coarse-max-cells',type=int,default=95000)
    p.add_argument('--fine-max-cells',type=int,default=250000)
    p.add_argument('--roi-link-samples',type=int,default=3)
    p.add_argument('--top-quantile',type=float,default=.90)
    p.add_argument('--only-scale',choices=['both','coarse','fine'],default='both')
    p.add_argument('--window-max-rounds',type=int,default=12)
    p.add_argument('--window-growth-voxels',type=float,default=3.0)
    p.add_argument('--window-max-voxel-scale',type=float,default=1.25)
    a=p.parse_args(argv)
    if a.root_iterations<8 or a.root_iterations>45:p.error('--root-iterations 8..45')
    if a.eval_batch<32:p.error('--eval-batch >=32')
    if a.sample_gaussians<1:p.error('--sample-gaussians>=1')
    if not (0.5<a.top_quantile<1):p.error('--top-quantile in (0.5,1)')
    if a.roi_link_samples<1 or a.roi_link_samples>9:p.error('--roi-link-samples 1..9')
    if a.edge_zero_epsilon<=0:p.error('--edge-zero-epsilon positive')
    return run(a)


if __name__=='__main__':
    main()

# stage2_region_unified_original_main_v1_20261112
