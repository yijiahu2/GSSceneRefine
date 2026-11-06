#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GeoGuard-ROI geometry-first validation prototype (NO GUI, NO SDF cleanup).

- Preserve the input Gaussian binary PLY vertex bytes/properties and row order.
- Project orthographic user polygons in the ORIGINAL COLMAP/GSPL coordinate frame.
- Assess Poisson faces using cleaned COLMAP SfM support.
- Permit bounded, confidence-supported exterior extensions from interior seeds only.
- Protect both near-boundary interior surface and accepted exterior surface with
  Gaussian-scale-adaptive balls; never change original user polygon.
- Report unresolved extension and fail closed on over-expansion; never mutate inputs.

This is an experimental geometry baseline. Poisson confidence is not a calibrated
probability; Poisson continuity does not imply semantic object membership.
"""
from __future__ import annotations
import argparse
import hashlib
import heapq
import json
import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
import trimesh

LOGGER = logging.getLogger("geoguard")


def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x,np.generic) else x.tolist() if isinstance(x,np.ndarray) else str(x)), encoding="utf-8")


def file_sha256(path, block=4 * 1024 * 1024):
    digest=hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda:f.read(block), b""):
            digest.update(b)
    return digest.hexdigest()


def check_xyz(a, name):
    a=np.asarray(a,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite (N,3), received {a.shape}")
    return a


@dataclass
class Projection:
    name: str
    origin: np.ndarray
    U: np.ndarray                 # 2 x 3 rows unit and mutually orthogonal
    polygon: np.ndarray           # vertices in model units, NOT pixels

    @classmethod
    def from_json(cls, x):
        origin=np.asarray(x.get("origin",[0,0,0]),dtype=np.float64)
        U=np.asarray(x["axes"],dtype=np.float64)
        poly=np.asarray(x["polygon"],dtype=np.float64)
        if origin.shape!=(3,) or U.shape!=(2,3) or poly.ndim!=2 or poly.shape[1]!=2 or len(poly)<3:
            raise ValueError("Projection origin (3), axes (2,3), polygon (M,2 with M>=3) required")
        if not np.isfinite(origin).all() or not np.isfinite(U).all() or not np.isfinite(poly).all():
            raise ValueError("Nonfinite projection parameters")
        if not np.allclose(U@U.T,np.eye(2),atol=1e-6):
            raise ValueError("Projection axes must be orthonormal; convert pixel coords to model units first")
        if np.linalg.norm(poly[0]-poly[-1])<1e-12:
            poly=poly[:-1]
        # Optional shapely validation; no dependency for the core calculation.
        try:
            from shapely.geometry import Polygon
            p=Polygon(poly)
            if not p.is_valid or p.area<=0:
                raise ValueError(f"Invalid/self-intersecting polygon {x.get('name','?')}")
        except ImportError:
            pass
        return cls(str(x["name"]),origin,U,poly)

    def plane(self, xyz):
        return (np.asarray(xyz,dtype=np.float64)-self.origin)@self.U.T

    def signed_and_normal(self, xyz):
        """Signed distance in projection model units (+ inside); nearest outward 3D unit normal.

        For external points the signed magnitude is a constraint violation, not
        in general the true Euclidean distance to the 3D intersection ROI.
        """
        q=self.plane(xyz)
        n=len(q)
        dist2=np.full(n,np.inf,dtype=np.float64)
        nn=np.zeros((n,2),dtype=np.float64)
        inside=np.zeros(n,dtype=bool)
        poly=self.polygon
        area2=np.sum(poly[:,0]*np.roll(poly[:,1],-1)-np.roll(poly[:,0],-1)*poly[:,1])
        ccw=area2>0
        for i in range(len(poly)):
            a=poly[i]; b=poly[(i+1)%len(poly)]
            edge=b-a
            L2=float(edge@edge)
            if L2<1e-24:
                continue
            # Standard ray-crossing for polygons (no holes).
            crossing=((a[1]>q[:,1])!=(b[1]>q[:,1])) & (q[:,0] < (b[0]-a[0])*(q[:,1]-a[1])/(b[1]-a[1]+1e-300)+a[0])
            inside ^= crossing
            t=np.clip(((q-a)@edge)/L2,0,1)
            d=q-(a+t[:,None]*edge)
            v=np.einsum('ij,ij->i',d,d)
            use=v<dist2
            dist2[use]=v[use]
            if ccw: en=np.array([edge[1],-edge[0]])/np.sqrt(L2)
            else: en=np.array([-edge[1],edge[0]])/np.sqrt(L2)
            nn[use]=en
        d=np.sqrt(dist2)
        d[inside & (d<1e-10)]=0.0
        signed=np.where(inside,d,-d)
        return signed,nn@self.U


class ROI:
    def __init__(self, projections):
        if not projections: raise ValueError("Need at least one projection")
        names=[x.name for x in projections]
        if len(names)!=len(set(names)):raise ValueError("Duplicate projection names")
        self.views=projections

    @classmethod
    def from_file(cls, path):
        obj=json.loads(Path(path).read_text(encoding="utf-8"))
        if 'WARNING' in obj:
            raise ValueError('ROI JSON is an uncalibrated TEMPLATE; replace with genuine orthographic polygons and remove WARNING')
        return cls([Projection.from_json(v) for v in obj["views"]])

    def distances(self, points, chunk=40000):
        points=check_xyz(points,"points")
        result=[]
        for start in range(0,len(points),chunk):
            xyz=points[start:start+chunk]
            sd=np.stack([v.signed_and_normal(xyz)[0] for v in self.views],axis=1)
            result.append(sd)
        return np.concatenate(result,axis=0) if result else np.empty((0,len(self.views)))


_PLY_TYPES={"char":"i1","uchar":"u1","short":"<i2","ushort":"<u2","int":"<i4","uint":"<u4","float":"<f4","double":"<f8","int8":"i1","uint8":"u1","int16":"<i2","uint16":"<u2","int32":"<i4","uint32":"<u4","float32":"<f4","float64":"<f8"}


class GaussianPLY:
    """Binary little endian, vertex-only Gaussian PLY with scalar properties.
    Keeps unknown original vertex properties BYTE-FOR-BYTE when writing subsets.
    """
    def __init__(self,path):
        self.path=Path(path)
        if not self.path.exists():raise FileNotFoundError(path)
        lines=[]
        with self.path.open("rb") as f:
            while True:
                line=f.readline()
                if not line:raise ValueError("No end_header in PLY")
                lines.append(line)
                if line.strip()==b"end_header":break
            self.data_offset=f.tell()
        self.header=b"".join(lines)
        decoded=[x.decode('ascii').strip() for x in lines]
        if decoded[0]!="ply" or "format binary_little_endian 1.0" not in decoded:
            raise ValueError("GaussianPLY supports binary_little_endian 1.0 only; export GSPL native binary PLY")
        self.vertex_count=None
        props=[]; element=None; other_count=0
        for line in decoded:
            p=line.split()
            if len(p)>=3 and p[0]=="element":
                element=p[1]
                if element=="vertex":self.vertex_count=int(p[2])
                elif int(p[2])!=0:other_count+=int(p[2])
            if p and p[0]=="property" and element=="vertex":
                if len(p)!=3 or p[1] not in _PLY_TYPES:
                    raise ValueError("Unsupported list/nonstandard vertex PLY property")
                props.append((p[2],_PLY_TYPES[p[1]]))
        if self.vertex_count is None or other_count:
            raise ValueError("Gaussian PLY must contain a vertex element and no other nonempty elements")
        self.dtype=np.dtype(props)
        self.arr=np.memmap(self.path,dtype=self.dtype,mode="r",offset=self.data_offset,shape=(self.vertex_count,))
        required={'x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3'}
        if not required.issubset(self.dtype.names):
            raise ValueError(f"Missing GSPL Gaussian fields: {required-set(self.dtype.names)}")

    def positions(self):
        a=self.arr
        return np.column_stack([a['x'],a['y'],a['z']]).astype(np.float64)

    def selected_xyz(self,ids):
        a=self.arr[ids]
        return np.column_stack([a['x'],a['y'],a['z']]).astype(np.float64)

    def directional_sigma(self, ids, normals):
        """Std dev along supplied world-coordinate normal; GSPL PLY rot_0 is quaternion w.
        Supports normals shape(3,) or (K,3); log(scale_*) to activated sigma.
        """
        ids=np.asarray(ids,dtype=np.int64)
        arr=self.arr[ids]
        n=np.asarray(normals,dtype=np.float64)
        if n.shape==(3,):n=np.broadcast_to(n,(len(ids),3))
        if n.shape!=(len(ids),3):raise ValueError("Normals shape mismatch")
        q=np.column_stack([arr[f'rot_{i}'] for i in range(4)]).astype(np.float64)
        length=np.linalg.norm(q,axis=1)
        if np.any(length<1e-12):raise ValueError("Zero quaternion in model")
        q/=length[:,None]
        # n_local = quaternion inverse rotation of n_world.
        w=q[:,0]; vec=q[:,1:]
        v_cross=np.cross(-vec,n)
        rotated=n + 2*w[:,None]*v_cross + 2*np.cross(-vec,v_cross)
        s=np.column_stack([arr[f'scale_{i}'] for i in range(3)]).astype(np.float64)
        if (np.abs(s)>80).any():raise ValueError("Implausible Gaussian PLY log scales")
        return np.sqrt(np.sum(np.exp(2*s)*rotated**2,axis=1))

    def export(self,mask,path):
        mask=np.asarray(mask,dtype=bool)
        if len(mask)!=self.vertex_count:raise ValueError("Mask differs from Gaussian count")
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        n=int(mask.sum())
        header_lines=self.header.decode('ascii').splitlines(keepends=True)
        for i,line in enumerate(header_lines):
            if line.strip().startswith("element vertex "):
                end='\r\n' if line.endswith('\r\n') else '\n'
                header_lines[i]=f"element vertex {n}{end}"
        with path.open('wb') as f:
            f.write(''.join(header_lines).encode('ascii'))
            for start in range(0,len(mask),75000):
                chunk=self.arr[start:start+75000]
                f.write(chunk[mask[start:start+75000]].tobytes())
        return n


def load_sfm(path):
    with np.load(path,allow_pickle=False) as x:
        if "xyz" not in x:raise ValueError("Cleaned SfM NPZ must contain 'xyz'")
        xyz=check_xyz(x["xyz"],"sfm.xyz")
        def maybe(names):
            for k in names:
                if k in x:
                    a=np.asarray(x[k]).reshape(-1)
                    if len(a)==len(xyz):return a.astype(np.float64),k
            return None,None
        err,err_name=maybe(["error","errors","reprojection_error","reprojection_errors"])
        tracks,tracks_name=maybe(["track_length","track_lengths","track_len","track_lens","num_observations"])
    return xyz,err,tracks,{"errors":err_name,"tracks":tracks_name}


def geometry_confidence(mesh,sfm,err,tracks,k=6,weights=(0.35,0.40,0.25)):
    """Deterministic, heuristic support score (NOT a posterior probability)."""
    xyz=np.asarray(mesh.triangles_center,dtype=np.float64)
    sfmtree=cKDTree(sfm)
    dd,ii=sfmtree.query(xyz,k=max(2,k),workers=-1)
    h=np.maximum(dd[:,-1],1e-9)
    median_h=max(float(np.median(h)),1e-9)
    density=np.minimum(1.0,(1.4*median_h/h)**0.7)
    fit=np.exp(-dd[:,0]/(1.5*h))
    obs=np.full(len(xyz),0.5,dtype=np.float64)
    if tracks is not None:
        obs=np.clip((tracks[ii[:,0]]-2.0)/4.0,0,1)
    if err is not None:
        good=err[np.isfinite(err)]
        med=max(float(np.median(good)),1e-6) if len(good) else 1.0
        errscore=np.exp(-np.maximum(err[ii[:,0]],0)/ (2*med))
        if tracks is None:obs=errscore
        else:obs=0.5*obs+0.5*errscore
    C=weights[0]*density + weights[1]*fit + weights[2]*obs
    return np.clip(C,0,1),h,ii[:,0],sfmtree,{"median_spacing":median_h,"missing_observation_quality":err is None and tracks is None}


def exact_sfm_to_poisson_distance_open3d(mesh, sfm, batch=50000):
    """Exact nearest-triangle UNSIGNED distances, including open Poisson meshes.

    Requires Open3D RaycastingScene; called only when enabled. Distances are
    computed on the ORIGINAL observed Poisson, never on artificial SDF caps.
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("local_fit_enabled requires 'open3d'; install in WSL environment or disable this optional term") from exc
    legacy=o3d.geometry.TriangleMesh()
    legacy.vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices,dtype=np.float64))
    legacy.triangles=o3d.utility.Vector3iVector(np.asarray(mesh.faces,dtype=np.int32))
    scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    result=np.empty(len(sfm),dtype=np.float32)
    for start in range(0,len(sfm),batch):
        pts=o3d.core.Tensor(np.asarray(sfm[start:start+batch],dtype=np.float32))
        d=scene.compute_distance(pts).numpy()
        result[start:start+len(d)]=d
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite SfM-to-Poisson distances")
    return result


def mesh_external_candidates(mesh, roi, centers, conf, threshold, diag, cfg):
    """Find bounded credible exterior patches connected to interior boundary seeds.

    The exterior/interior area ratio is a *diagnostic* (a user-drawn ROI may
    deliberately cut away a large fraction of the target), not a proof of
    background leakage. In the default 'review' policy, exceeding its soft
    threshold keeps bounded candidates for comparison and marks the result as
    PROVISIONAL. The opt-in 'reject' policy preserves the older fail-closed
    behavior. No input geometry is modified.
    """
    sd = roi.distances(centers)
    nearest = sd.min(axis=1)
    inside = (sd >= -1e-9).all(axis=1)
    credible = conf >= threshold
    out_cap = float(cfg['outside_search_frac']) * diag
    inner_cap = float(cfg['inner_band_frac']) * diag
    geo_cap = float(cfg['geodesic_limit_frac']) * diag
    # Hard geometric checks are always enforced, irrespective of area ratio:
    # confidence, signed projection strip, mesh connectivity and geodesic cap.
    region = credible & (nearest >= -out_cap) & (nearest <= inner_cap)
    seeds = np.flatnonzero(region & inside)
    if len(seeds) == 0:
        raise RuntimeError('No confident near-boundary INSIDE seed faces; do not auto-extend')
    adjacency = mesh.face_adjacency
    good = region[adjacency[:, 0]] & region[adjacency[:, 1]]
    edges = adjacency[good]
    weights = np.linalg.norm(centers[edges[:, 0]] - centers[edges[:, 1]], axis=1)
    n = len(centers)
    graph = coo_matrix((np.r_[weights, weights],
                        (np.r_[edges[:, 0], edges[:, 1]],
                         np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n)).tocsr()
    geod = dijkstra(graph, directed=False, indices=seeds, min_only=True, limit=geo_cap)
    candidate = (~inside) & region & np.isfinite(geod) & (geod <= geo_cap)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    internal_area = float(areas[credible & inside].sum())
    candidate_area = float(areas[candidate].sum())
    area_ratio = candidate_area / max(internal_area, 1e-12)
    soft_limit = float(cfg.get('max_extension_area_fraction', .10))
    if not 0 <= soft_limit:
        raise ValueError('max_extension_area_fraction must be nonnegative')
    policy = cfg.get('extension_area_policy', 'review')
    if policy not in ('review', 'reject'):
        raise ValueError('extension_area_policy must be review or reject')
    over_budget = area_ratio > soft_limit
    ext = candidate.copy()
    rejected = bool(over_budget and policy == 'reject')
    if rejected:
        ext[:] = False
        LOGGER.warning('Exterior area %.4f exceeds soft limit %.4f; '
                       'explicit reject policy discarded exterior patches.', area_ratio, soft_limit)
    elif over_budget:
        LOGGER.warning('Exterior area %.4f exceeds soft review threshold %.4f. '
                       'Bounded candidates are retained as PROVISIONAL; '
                       'inspect spatial spill and background before accepting.', area_ratio, soft_limit)

    # Component statistics allow checking connected-background leakage, which
    # a single whole-scene percentage cannot diagnose. Components are formed
    # by actual triangle-face adjacency, not by proximity across empty space.
    _, labels = connected_components(graph, directed=False)
    components = []
    cand_ids = np.flatnonzero(candidate)
    ordered = cand_ids[np.argsort(labels[cand_ids], kind='stable')]
    if len(ordered):
        cand_labels = labels[ordered]
        bounds = np.r_[0, np.flatnonzero(np.diff(cand_labels)) + 1, len(ordered)]
        # Count actual confident internal seeds in each adjacency component.
        seed_count = np.bincount(labels[seeds], minlength=n)
        for a, b in zip(bounds[:-1], bounds[1:]):
            face_ids = ordered[a:b]
            component = int(cand_labels[a])
            outside_dist = np.maximum(0, -nearest[face_ids])
            components.append({
                'component_id': component,
                'candidate_exterior_faces': int(len(face_ids)),
                'candidate_exterior_area': float(areas[face_ids].sum()),
                'seed_faces': int(seed_count[component]),
                'median_confidence': float(np.median(conf[face_ids])),
                'max_projection_violation': float(outside_dist.max()),
                'max_geodesic_distance': float(geod[face_ids].max()),
            })
    components.sort(key=lambda item: item['candidate_exterior_area'], reverse=True)
    review = bool(over_budget and policy == 'review')
    return sd, inside, credible, ext, {
        'initial_internal_seed_faces': int(len(seeds)),
        'confidence_faces': int(credible.sum()),
        'potential_exterior_faces': int(((~inside) & region).sum()),
        'bounded_candidate_exterior_faces': int(candidate.sum()),
        'bounded_candidate_exterior_area': candidate_area,
        'accepted_exterior_faces': int(ext.sum()),
        'credible_internal_area': internal_area,
        'outside_area_to_internal_fraction_before_budget': area_ratio,
        'extension_budget_policy': policy,
        'extension_review_threshold': soft_limit,
        'extension_rejected_by_budget': rejected,
        'review_required': review,
        'review_reason': ('Large *bounded* exterior patch needs visual / independent verification; '
                          'area percentage alone does not distinguish target from background') if review else None,
        'exterior_component_count': int(len(components)),
        'largest_exterior_components': components[:12],
        'outside_search': out_cap,
        'geodesic_limit': geo_cap,
    }


def local_protection(mesh,roi,ply,xyz, sfm, sfmtree,conf,h_faces,near_sfm_ids,sd,inside,credible,ext,diag,cfg,mode,sfm_mesh_residuals=None):
    """Build center/radius samples for guarded surfaces, only close to ROI boundary."""
    keep_band=float(cfg['inner_band_frac'])*diag
    candidates=credible & ((inside & (sd.min(axis=1)<=keep_band)) | ext)
    faces=np.flatnonzero(candidates)
    if len(faces)==0:raise RuntimeError("No credible candidate faces near ROI boundary")
    max_samples=int(cfg['max_surface_samples'])
    if len(faces)>max_samples:
        # Deterministic, spatially spread sampling rather than relying on mesh row order.
        rng=np.random.default_rng(int(cfg.get('seed',2026)))
        faces=np.sort(rng.choice(faces,max_samples,replace=False))
    surface=np.asarray(mesh.triangles_center[faces],dtype=np.float64)
    face_h=h_faces[faces]
    nfaces=len(faces)
    n_view=len(roi.views)
    s=sd[faces]
    # Near-surface Gaussian KD-tree; query with variable radius and max k.
    gauss_tree=cKDTree(xyz)
    gaussian_min=int(cfg['min_gaussian_neighbors'])
    gaussian_k=int(cfg['max_gaussian_neighbors'])
    lookup_radius=np.clip(float(cfg['local_radius_multiplier'])*face_h,
        float(cfg['local_radius_min_frac'])*diag,float(cfg['local_radius_max_frac'])*diag)
    b=np.zeros((nfaces,n_view),dtype=np.float64)
    local_used=np.zeros((nfaces,n_view),dtype=bool)
    global_ids=np.flatnonzero((roi.distances(xyz).min(axis=1)>=0))
    if len(global_ids)==0:raise RuntimeError('Gaussian PLY has no centers inside ROI; check coordinates and polygon')
    rng=np.random.default_rng(int(cfg.get('seed',2026)))
    global_size=min(len(global_ids),int(cfg['global_sample_cap']))
    global_ids=np.sort(rng.choice(global_ids,global_size,replace=False)) if len(global_ids)>global_size else global_ids
    # Global fallback P10-P90 trimmed directional sigma, computed per unique
    # query normal; quantize direction to reduce cost while preserving orientation.
    normal_cache={}
    def global_sigma(direction):
        key=tuple(np.round(direction,2).tolist())
        if key not in normal_cache:
            sig=ply.directional_sigma(global_ids, np.asarray(direction,dtype=np.float64))
            lo,hi=np.quantile(sig,[0.1,0.9]); trimmed=sig[(sig>=lo)&(sig<=hi)]
            normal_cache[key]=float(trimmed.mean()) if len(trimmed) else float(sig.mean())
        return normal_cache[key]
    # SfM-to-true-triangle local fit: optional exact Open3D mesh distance.
    # The alternative is an explicitly documented SfM-spacing proxy.
    sfm_uncertainty_factor=float(cfg['sfm_uncertainty_factor'])
    local_fit=np.zeros(nfaces,dtype=np.float64)
    if sfm_mesh_residuals is not None:
        _,sfm_neighbor_ids=sfmtree.query(surface,k=min(int(cfg.get('fit_knn',16)),len(sfm)),workers=-1)
        sfm_neighbor_ids=np.asarray(sfm_neighbor_ids)
        if sfm_neighbor_ids.ndim==1:sfm_neighbor_ids=sfm_neighbor_ids[:,None]
        local_fit=np.quantile(sfm_mesh_residuals[sfm_neighbor_ids],.95,axis=1)
    bmin=float(cfg['b_min_frac'])*diag
    bmax=float(cfg['b_max_frac'])*diag
    scale=float(cfg['sigma_multiplier'])
    for j in range(nfaces):
        d,ids=gauss_tree.query(surface[j],k=gaussian_k,distance_upper_bound=float(lookup_radius[j]))
        ids=np.atleast_1d(ids);d=np.atleast_1d(d)
        good=(ids<len(xyz))&np.isfinite(d)
        ids=ids[good]
        # Reject unusually large Gaussians using global direction-specific P99
        # below; a small number of isolated artifacts must not dominate the P90.
        for vi,view in enumerate(roi.views):
            _,normal=view.signed_and_normal(surface[j:j+1])
            nv=normal[0]
            global_s=global_sigma(nv)
            sigma=global_s
            if mode=='local' and len(ids)>=gaussian_min:
                vals=ply.directional_sigma(ids,nv)
                vals=vals[np.isfinite(vals) & (vals>0) & (vals < 10*max(global_s,1e-12))]
                if len(vals)>=gaussian_min:
                    sigma=float(np.quantile(vals,0.90));local_used[j,vi]=True
            b[j,vi]=np.clip(scale*sigma+max(local_fit[j],sfm_uncertainty_factor*face_h[j]),bmin,bmax)
    # Risk is per view; guard radius uses conservative maximum across views.
    deficit=(b-s)>0
    risk=deficit.any(axis=1)
    radii=np.max(b,axis=1)
    return surface[risk],radii[risk],{
        'candidate_surface_faces_sampled':nfaces,
        'risk_faces_sampled':int(risk.sum()),
        'local_stats_fraction':float(local_used.mean()),
        'guard_radius_p50':float(np.median(radii[risk])) if risk.any() else 0,
        'guard_radius_p95':float(np.quantile(radii[risk],0.95)) if risk.any() else 0,
        'guard_width_min':bmin,'guard_width_max':bmax,
        'global_gaussian_sample_count':int(global_size),
        'uncertainty_source':'exact_local_sfm_mesh_fit_P95_plus_spacing_max' if sfm_mesh_residuals is not None else 'sfm_knn_spacing_proxy_only',
    }


def add_guard_gaussians(xyz, base_mask, surface, radii, max_added_fraction=0.15,
                        budget_policy='review'):
    """Compute guarded Gaussian centers without modifying the original model.

    A large added fraction produces a PROVISIONAL mask for verification by
    default. The opt-in 'reject' mode preserves legacy fail-closed semantics.
    """
    if budget_policy not in ('review', 'reject'):
        raise ValueError('guard_budget_policy must be review or reject')
    added = np.zeros(len(xyz), dtype=bool)
    if not len(surface):
        return added, {'guard_raw_added': 0, 'guard_accepted': 0,
                       'budget_rejected': False, 'review_required': False,
                       'added_to_baseline_fraction': 0.0}
    tree = cKDTree(xyz)
    for start in range(0, len(surface), 250):
        neighborhoods = tree.query_ball_point(surface[start:start + 250],
                                               r=radii[start:start + 250], workers=-1)
        for ids in neighborhoods:
            if len(ids):
                added[np.asarray(ids, dtype=np.int64)] = True
    added &= ~base_mask
    total = int(added.sum())
    baseline = int(base_mask.sum())
    fraction = total / max(1, baseline)
    exceeds = fraction > float(max_added_fraction)
    rejected = bool(exceeds and budget_policy == 'reject')
    if rejected:
        added[:] = False
        LOGGER.warning('Guard added fraction %.4f > %.4f; explicit reject policy '
                       'discarded guard.', fraction, max_added_fraction)
    elif exceeds:
        LOGGER.warning('Guard added fraction %.4f > %.4f: retaining PROVISIONAL '
                       'mask for assessment (not a validated final cleanup).',
                       fraction, max_added_fraction)
    return added, {
        'guard_raw_added': total,
        'guard_accepted': int(added.sum()),
        'budget_rejected': rejected,
        'review_required': bool(exceeds and not rejected),
        'guard_budget_policy': budget_policy,
        'guard_review_threshold': float(max_added_fraction),
        'added_to_baseline_fraction': fraction,
    }


def run(config_path,roi_path=None,modes=None,output_dir=None,export_ply_override=None):
    cfg=yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    p=cfg['paths']
    out=Path(output_dir if output_dir is not None else p['output_dir']);out.mkdir(parents=True,exist_ok=True)
    # Exact config snapshot is part of the scientific record.
    (out/'config_snapshot.yaml').write_text(Path(config_path).read_text(encoding='utf-8'),encoding='utf-8')
    rpath=roi_path or p['roi_json']
    roi=ROI.from_file(rpath)
    meshpath=Path(p['poisson_raw']);sfmpath=Path(p['clean_sfm_npz']);plypath=Path(p['gaussian_ply'])
    for path in (meshpath,sfmpath,plypath):
        if not path.is_file():raise FileNotFoundError(f"Missing input: {path}")
    mesh=trimesh.load(str(meshpath),process=False)
    if not isinstance(mesh,trimesh.Trimesh) or not len(mesh.faces):raise ValueError('Poisson mesh must be a triangle mesh, not a scene')
    sfm,err,tracks,available=load_sfm(sfmpath)
    ply=GaussianPLY(plypath)
    xyz=ply.positions()
    diag=float(np.linalg.norm(np.ptp(xyz,axis=0)))
    if diag<=0:raise ValueError('Gaussian bounding diagonal <= 0')
    # Check mesh and SfM share a coordinate system approximately.
    d,_=cKDTree(sfm).query(np.asarray(mesh.vertices)[::max(1,len(mesh.vertices)//4000)],k=1)
    if np.median(d)>0.15*diag:
        raise RuntimeError('Mesh and cleaned SfM appear misregistered; verify native COLMAP coordinates')
    confidence,h,ids,stree,cdiag=geometry_confidence(mesh,sfm,err,tracks,
        k=int(cfg['sfm']['knn_spacing']),weights=tuple(cfg['sfm']['confidence_weights']))
    np.savez_compressed(out/'surface_confidence.npz',score=confidence.astype('f4'),face_index=np.arange(len(mesh.faces),dtype='i4'),face_spacing=h.astype('f4'))
    LOGGER.info('Poisson confidence: median=%.3f, P10=%.3f, P90=%.3f, above threshold=%.1f%%',float(np.median(confidence)),float(np.quantile(confidence,.1)),float(np.quantile(confidence,.9)),100*float(np.mean(confidence>=float(cfg['sfm']['confidence_threshold']))))
    c0=cfg['guards'];th=float(cfg['sfm']['confidence_threshold'])
    fit_residuals=None
    if bool(cfg['sfm'].get('local_fit_enabled',False)):
        LOGGER.info('Calculating exact cleaned-SfM-to-original-Poisson triangle distances ...')
        fit_residuals=exact_sfm_to_poisson_distance_open3d(mesh,sfm,batch=int(cfg['sfm'].get('fit_batch_size',50000)))
        np.save(out/'sfm_poisson_exact_fit.npy',fit_residuals)
    c0sd,inside,credible,ext,extdiag=mesh_external_candidates(mesh,roi,np.asarray(mesh.triangles_center),confidence,th,diag,c0)
    np.savez_compressed(out/'exterior_candidate_faces.npz',
        face_index=np.flatnonzero(ext).astype('i4'),
        confidence=confidence[ext].astype('f4'),
        projection_violation=np.maximum(0, -c0sd[ext].min(axis=1)).astype('f4'))
    if np.any(ext):
        # A separate diagnostic geometry layer allows visual checking whether
        # a bounded extension follows the target cliff or adjacent background.
        exterior_mesh=trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                                     faces=np.asarray(mesh.faces)[ext], process=False)
        exterior_mesh.remove_unreferenced_vertices()
        exterior_mesh.export(out/'exterior_candidate_surface.ply')
    base_sd=roi.distances(xyz)
    basemask=(base_sd>=-1e-9).all(axis=1)
    mode_list=modes or cfg.get('modes',['hard','global','local'])
    report={'version':'0.1.0','project':'GSSceneRefine / GeoGuard-ROI geometry core',
        'modes':mode_list,'mesh_faces':len(mesh.faces),'sfm_points':len(sfm),'gaussians':len(xyz),
        'confidence_threshold':th,'credible_faces':int(credible.sum()),
        'sfm_fields':available,'geometry_statistics':cdiag,'exterior_search':extdiag,
        'sfm_exact_fit_p95':float(np.quantile(fit_residuals,.95)) if fit_residuals is not None else None,
        'source_model_scope':cfg.get('source_model_scope','unknown'),
        'actual_output_dir':str(out.resolve()),
        'config_sha256':file_sha256(config_path),
        'warnings':['Poisson confidence is an uncalibrated support score, NOT a probability.',
          'Mesh continuity is not semantic membership; exterior restoration is bounded and may be wrong near connected background.',
          'Protection balls are a conservative finite approximation; true Gaussian support is unbounded.',
          'No real-scene correctness claim is made without independent reference validation.'],
        'inputs':{k:{'path':str(v),'sha256':file_sha256(v)} for k,v in [('roi_json',rpath),('poisson_raw',meshpath),('clean_sfm_npz',sfmpath),('gaussian_ply',plypath)]}}
    if 'aabb' in str(cfg.get('source_model_scope','')).lower():
        report['warnings'].append('Input Gaussian PLY is already cropped AABB. No Gaussian deleted by the former AABB can be restored; full original Gaussian PLY is required for research conclusions.')
    for mode in mode_list:
        if mode not in ('hard','global','local'):raise ValueError('Mode must be hard, global, local')
        LOGGER.info('Mode: %s',mode)
        extra=np.zeros(len(xyz),dtype=bool)
        local_diag={}
        add_diag={}
        if mode!='hard':
            surface,radii,local_diag=local_protection(mesh,roi,ply,xyz,sfm,stree,confidence,h,ids,c0sd,inside,credible,ext,diag,c0,mode,fit_residuals)
            extra,add_diag=add_guard_gaussians(
                xyz,basemask,surface,radii,float(c0['max_guard_add_fraction']),
                budget_policy=c0.get('guard_budget_policy','review'))
            np.savez_compressed(out/f'guard_{mode}.npz',points=surface.astype('f4'),radii=radii.astype('f4'))
        mask=basemask|extra
        np.save(out/f'keep_mask_{mode}.npy',mask)
        np.save(out/f'added_mask_{mode}.npy',extra)
        review_needed = bool(extdiag['review_required'] or add_diag.get('review_required',False)) if mode!='hard' else False
        export_ply=bool(cfg.get('export_ply',False) if export_ply_override is None else export_ply_override)
        export_name = f'gaussians_roi_{mode}_PROVISIONAL.ply' if review_needed else f'gaussians_roi_{mode}.ply'
        if export_ply:
            ply.export(mask,out/export_name)
        report[mode]={'review_required':review_needed,
                      'result_status':'PROVISIONAL_REVIEW_REQUIRED' if review_needed else 'GEOMETRIC_CANDIDATE_NOT_VALIDATED',
                      'output_ply_path':str(out/export_name) if export_ply else None,
                      'base_count':int(basemask.sum()),'added_count':int(extra.sum()),'final_count':int(mask.sum()),
                      'removed_count':int(len(mask)-mask.sum()),'retained_fraction':float(mask.mean()),
                      'surface_diagnostics':local_diag,'guard_diagnostics':add_diag,'exported_ply':export_ply}
    report['review_required']=any(v.get('review_required',False) for k,v in report.items() if k in ('global','local') and isinstance(v,dict))
    if report['review_required']:
        report['warnings'].append('PROVISIONAL: at least one area/guard soft threshold exceeded; ' 
            'inspect exterior_candidate_surface.ply and render results before accepting. ' 
            'This is not proof of valid target recovery.')
    json_write(out/'report.json',report)
    json_write(out/'roi_snapshot.json',json.loads(Path(rpath).read_text(encoding='utf-8')))
    LOGGER.info('Done. Results in %s',out)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--roi',default=None,help='Override polygon JSON')
    p.add_argument('--modes',default=None,help='Comma-separated hard,global,local')
    p.add_argument('--output-dir',default=None,help='Optional run output directory; useful for deliberate miscut tests without overwriting baseline')
    p.add_argument('--export-ply',action='store_true',help='Export large Gaussian PLYs; provisional results are marked in filenames')
    a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s: %(message)s')
    run(a.config,a.roi,a.modes.split(',') if a.modes else None,a.output_dir,True if a.export_ply else None)

if __name__=='__main__':main()
