"""Two-scale observation-constrained *estimated* occupancy for an OPEN mesh.

This does NOT convert an open surface into a certified solid.  Exact geometric
Gaussian distance is computed separately, on the original triangle mesh.
Strong exterior evidence comes only from COLMAP image <-> 3D point tracks;
no arbitrary Gaussian-to-camera rays are used.  Classify unsupported or
resolution-unstable locations as UNKNOWN rather than authorizing deletion.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import struct
import numpy as np

OUTSIDE = np.int8(1)
INSIDE = np.int8(-1)
UNKNOWN = np.int8(0)


@dataclass(frozen=True)
class GridSpec:
    origin: np.ndarray
    shape: tuple[int,int,int]
    voxel: float

    @property
    def count(self):
        return math.prod(self.shape)

    @property
    def bounds_max(self):
        return self.origin + self.voxel * np.asarray(self.shape)

    def locations(self, indices):
        ijk=np.asarray(np.unravel_index(indices,self.shape)).T
        return self.origin[None,:] + (ijk + .5) * self.voxel

    def index(self, points):
        xyz=np.asarray(points,dtype=np.float64).reshape(-1,3)
        ijk=np.floor((xyz-self.origin)/self.voxel).astype(np.int64)
        ok=np.all((ijk>=0)&(ijk<np.asarray(self.shape)),axis=1)
        out=np.full(len(xyz),-1,dtype=np.int64)
        if ok.any():
            out[ok]=np.ravel_multi_index(ijk[ok].T,self.shape)
        return out


def make_grid(points, mesh_vertices, delta, voxel_size=None, max_nodes=240000):
    """Uniform bounded 3-D voxel grid. Explicit cap prevents accidental OOM."""
    a=np.asarray(points,dtype=np.float64)
    b=np.asarray(mesh_vertices,dtype=np.float64)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=3 or b.shape[1]!=3 or not len(a) or not len(b):
        raise ValueError('Invalid geometry for implicit grid')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite implicit grid data')
    if max_nodes<4096 or delta<=0 or not np.isfinite(delta):
        raise ValueError('Invalid implicit grid limits')
    lower=np.minimum(a.min(axis=0),b.min(axis=0))
    upper=np.maximum(a.max(axis=0),b.max(axis=0))
    desired=float(voxel_size) if voxel_size is not None else max(.8*delta,.065)
    if not np.isfinite(desired) or desired<=0:
        raise ValueError('Invalid --voxel-size')
    padding=max(2*delta,2*desired)
    lower=lower-padding
    upper=upper+padding
    width=upper-lower
    voxel=max(desired,(float(np.prod(width))/max_nodes)**(1/3))
    for _ in range(96):
        shape=tuple(np.maximum(2,np.ceil(width/voxel).astype(np.int64)+1))
        if math.prod(shape)<=max_nodes:
            return GridSpec(lower,shape,voxel)
        voxel*=1.02
    raise RuntimeError('Cannot bound volume grid to memory cap')


def _read_cstring(f):
    parts=[]
    while True:
        q=f.read(1)
        if not q:
            raise EOFError('Truncated COLMAP images.bin')
        if q==b'\0': return b''.join(parts).decode('utf8','replace')
        parts.append(q)


def read_colmap_images_index(images_bin):
    """Return image ID -> (camera center, direction), skipping huge 2D tracks."""
    path=Path(images_bin)
    poses={}
    with path.open('rb') as f:
        head=f.read(8)
        if len(head)!=8:raise ValueError('Truncated COLMAP images.bin header')
        n,=struct.unpack('<Q',head)
        if not 0<n<1000000:raise ValueError('Invalid COLMAP image count')
        for _ in range(n):
            buf=f.read(64)
            if len(buf)!=64:raise ValueError('Truncated COLMAP image pose')
            image_id,*rest=struct.unpack('<i7di',buf)
            q=np.asarray(rest[:4],dtype=np.float64)
            t=np.asarray(rest[4:7],dtype=np.float64)
            nq=np.linalg.norm(q)
            if nq<=0:raise ValueError('Zero quaternion in COLMAP poses')
            q=q/nq
            w,x,y,z=q
            # COLMAP world->camera R; optical forward is camera +Z in world.
            rot=np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                          [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                          [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
            center=-rot.T@t
            _read_cstring(f)
            count_buf=f.read(8)
            if len(count_buf)!=8:raise ValueError('Truncated COLMAP 2D track count')
            num2d,=struct.unpack('<Q',count_buf)
            if num2d>100000000:raise ValueError('Implausible COLMAP 2D track count')
            f.seek(num2d*24,1)
            if image_id in poses:raise ValueError('Repeated COLMAP image ID')
            poses[image_id]=(center,rot.T[:,2])
    return poses


def sample_verified_colmap_rays(points3d_bin, image_poses, core_xyz, *,
                                max_points=65000, max_rays=130000,
                                max_reprojection_error=4., xyz_match_tolerance=0.0005):
    """Only actual COLMAP image-track pairs at points matching R3 core SfM.

    Deterministically subsample the point3D ID stream; no unsupported camera
    visibility is invented. Fail closed if the COLMAP sparse IDs cannot be
    matched to the verified core SfM locations.
    """
    from scipy.spatial import cKDTree
    core=np.asarray(core_xyz,dtype=np.float64)
    if core.ndim!=2 or core.shape[1]!=3 or len(core)<100:raise ValueError('Invalid core SfM')
    tree=cKDTree(core)
    poses=image_poses
    endpoints=[];origins=[];selected=0;sampled=0;matched=0;tracks=0
    with Path(points3d_bin).open('rb') as f:
        b=f.read(8)
        if len(b)!=8:raise ValueError('Truncated points3D.bin')
        count,=struct.unpack('<Q',b)
        if not 0<count<100000000:raise ValueError('Invalid points3D count')
        modulus=max(1,int(np.ceil(count/max_points)))
        for row in range(count):
            head=f.read(51)
            if len(head)!=51:raise ValueError('Truncated COLMAP point3D')
            pid,x,y,z,_r,_g,_b,error,numtrack=struct.unpack('<Q3d3BdQ',head)
            if numtrack>1000000:raise ValueError('Implausible COLMAP point3D track')
            chosen=(row % modulus == 0) and error<=max_reprojection_error and numtrack>=2
            if chosen:
                sampled+=1
                d,_=tree.query([[x,y,z]],k=1)
                chosen=bool(d[0]<=xyz_match_tolerance)
                if chosen:matched+=1
            if not chosen:
                f.seek(numtrack*8,1)
                continue
            track_bytes=f.read(numtrack*8)
            if len(track_bytes)!=numtrack*8:raise ValueError('Truncated COLMAP track')
            ids=np.frombuffer(track_bytes,dtype=np.dtype([('image','<i4'),('point2d','<i4')]))['image']
            selected+=1
            # Geometrically distributed actual observations (not nearest cameras).
            pick=np.unique(np.linspace(0,len(ids)-1,min(3,len(ids)),dtype=int))
            for j in pick:
                if int(ids[j]) in poses:
                    origins.append(poses[int(ids[j])][0]);endpoints.append((x,y,z));tracks+=1
                    if tracks>=max_rays:break
            if tracks>=max_rays:break
    if len(origins)<100:
        raise ValueError('Not enough matched COLMAP core SfM track rays; refusing unknown exterior')
    return np.asarray(origins,dtype=np.float64),np.asarray(endpoints,dtype=np.float64),{
        'colmap_total_points':int(count),'sampled_points':int(sampled),
        'core_matched_points':int(matched),'selected_points':int(selected),
        'rays_with_actual_track_support':int(len(origins)),
        'xyz_match_tolerance':float(xyz_match_tolerance),
        'max_reprojection_error':float(max_reprojection_error),
    }


def valid_observation_rays(origins, targets, scene, o3d, *, endpoint_margin=.15, batch=8192):
    """Discard rays blocked well before their *observed* SfM endpoint."""
    origins=np.asarray(origins,dtype=np.float64);targets=np.asarray(targets,dtype=np.float64)
    if origins.shape!=targets.shape or origins.ndim!=2 or origins.shape[1]!=3:
        raise ValueError('COLMAP rays have mismatching coordinates')
    accepted=np.zeros(len(origins),dtype=bool)
    for a in range(0,len(origins),batch):
        b=min(len(origins),a+batch)
        v=targets[a:b]-origins[a:b]
        d=np.linalg.norm(v,axis=1)
        valid=d>2*endpoint_margin
        rays=np.column_stack([origins[a:b],v/np.maximum(d[:,None],1.e-12)])
        hit=scene.cast_rays(o3d.core.Tensor(np.ascontiguousarray(rays,np.float32)))
        t=np.asarray(hit['t_hit'].numpy(),dtype=np.float64).reshape(-1)
        accepted[a:b]=valid & (np.isposinf(t) | (np.isfinite(t) & (t>=d-endpoint_margin)))
    return accepted


def rasterize_segments(spec, origins, endpoints, *, max_samples=9000000):
    """Carve only segments CAMERA->OBSERVED SfM, excluding endpoint margin upstream.

    Each grid-cell hit is evidence of observed free space. Voxel sampling
    is limited, bounded and vectorized per segment batch.
    """
    org=np.asarray(origins,dtype=np.float64);end=np.asarray(endpoints,dtype=np.float64)
    if org.shape!=end.shape:raise ValueError('Ray endpoint shape mismatch')
    hits=np.zeros(spec.shape,dtype=np.uint16)
    sampled=0
    lo=spec.origin;hi=spec.bounds_max
    for a in range(0,len(org),512):
        o=org[a:a+512];direction=end[a:a+512]-o
        # Slab clipping of segment parameter t in [0,1] to volume AABB.
        # Robust slab clipping even when a coordinate is exactly on a
        # boundary and the segment is parallel to that axis (avoid 0*inf).
        segment_tmin=np.full(len(o),-np.inf,dtype=np.float64)
        segment_tmax=np.full(len(o),np.inf,dtype=np.float64)
        parallel_invalid=np.zeros(len(o),dtype=bool)
        for axis in range(3):
            dj=direction[:,axis]
            parallel=np.abs(dj)<1e-12
            parallel_invalid|=parallel & ((o[:,axis]<lo[axis])|(o[:,axis]>hi[axis]))
            near=np.full(len(o),-np.inf,dtype=np.float64)
            far=np.full(len(o),np.inf,dtype=np.float64)
            valid=~parallel
            c1=(lo[axis]-o[valid,axis])/dj[valid]
            c2=(hi[axis]-o[valid,axis])/dj[valid]
            near[valid]=np.minimum(c1,c2)
            far[valid]=np.maximum(c1,c2)
            segment_tmin=np.maximum(segment_tmin,near)
            segment_tmax=np.minimum(segment_tmax,far)
        t0=np.maximum(0.,segment_tmin)
        t1=np.minimum(1.,segment_tmax)
        ok=(t1>t0)&~parallel_invalid
        if not ok.any():continue
        length=np.linalg.norm(direction,axis=1)
        n=np.minimum(512,np.maximum(1,np.ceil((t1-t0)*length/(.85*spec.voxel)).astype(int)))
        indices=np.where(ok)[0]
        # ragged, each sampled at no more than ~1 voxel spacing.
        repeat=np.repeat(indices,n[indices])
        if not len(repeat):continue
        if sampled+len(repeat)>max_samples:
            raise RuntimeError('COLMAP ray raster budget exceeded; lower --max-sfm-rays')
        stops=np.cumsum(n[indices]);starts=stops-n[indices]
        relative=(np.arange(len(repeat))-np.repeat(starts,n[indices])+.5)/np.repeat(n[indices],n[indices])
        t=t0[repeat]+relative*(t1[repeat]-t0[repeat])
        xyz=o[repeat]+direction[repeat]*t[:,None]
        idx=spec.index(xyz)
        valid=idx>=0
        if np.any(valid):
            flat=hits.reshape(-1)
            # one vote per intersected ray sample, uint16 saturating.
            cc=np.bincount(idx[valid],minlength=spec.count)
            flat[:]=np.minimum(65535,flat.astype(np.int32)+cc).astype(np.uint16)
        sampled+=len(repeat)
    return hits,{'segments':int(len(org)),'voxel_samples':int(sampled),
                 'supported_voxels':int(np.count_nonzero(hits))}


def select_surface_seeds(mesh, camera_xyz, scene, o3d, spec, *, max_faces=4000, offset=None,
                         observation_origins=None, observation_targets=None):
    """Estimate outward side using nearest COLMAP camera and verified mesh LOS.

    Only high-confidence, non-grazing source triangles supply inner/outer
    terminal seeds.  Original vertices and faces are never modified.
    """
    from scipy.spatial import cKDTree
    vertices=np.asarray(mesh.vertices,dtype=np.float64)
    faces=np.asarray(mesh.faces,dtype=np.int64)
    normal=np.asarray(mesh.face_normals,dtype=np.float64)
    areas=np.asarray(mesh.area_faces,dtype=np.float64)
    tri=vertices[faces]
    bary=tri.mean(axis=1)
    choices=np.argsort(areas)[::-1][:min(int(max_faces),len(faces))]
    bary=bary[choices];norm=normal[choices]
    if observation_origins is None or observation_targets is None:
        raise ValueError('Surface seed orientation must be backed by observed COLMAP track rays')
    observed_points=np.asarray(observation_targets,dtype=np.float64)
    observed_cam=np.asarray(observation_origins,dtype=np.float64)
    if observed_points.shape!=observed_cam.shape or observed_points.ndim!=2 or observed_points.shape[1]!=3:
        raise ValueError('Invalid observed COLMAP rays for surface seed evidence')
    track_distance,idx=cKDTree(observed_points).query(bary,k=1)
    external=observed_cam[idx]-bary
    mag=np.linalg.norm(external,axis=1)
    toward=external/np.maximum(mag[:,None],1e-9)
    orientation=np.einsum('ij,ij->i',norm,toward)
    valid=(mag>spec.voxel)&(np.abs(orientation)>.5)&(track_distance<max(2*spec.voxel,.2))
    # Confirm observed mesh is actually camera-visible; nearest camera alone
    # is not enough to claim which side of a folded surface is outside.
    rays=np.column_stack([observed_cam[idx],-toward])
    hits=scene.cast_rays(o3d.core.Tensor(np.ascontiguousarray(rays,np.float32)))
    length=np.asarray(hits['t_hit'].numpy(),dtype=np.float64).reshape(-1)
    valid &= np.isfinite(length)&(np.abs(length-mag)<=max(2*spec.voxel,0.1))
    outward=norm*np.sign(orientation)[:,None]
    h=float(offset if offset is not None else max(spec.voxel*.85,.02))
    inside=spec.index(bary-h*outward)
    outside=spec.index(bary+h*outward)
    valid &= (inside>=0)&(outside>=0)&(inside!=outside)
    valid &= areas[choices]>1e-10
    return np.unique(inside[valid]),np.unique(outside[valid]),{
        'sampled_faces':int(len(choices)),'visible_oriented_faces':int(valid.sum()),
        'actual_observed_track_near_face_count':int(np.count_nonzero(track_distance<max(2*spec.voxel,.2))),
        'inside_seed_cells':int(len(np.unique(inside[valid]))),
        'outside_seed_cells':int(len(np.unique(outside[valid])))
    }


def _grid_distances(spec, scene, o3d, chunk=32000):
    dist=np.empty(spec.count,dtype=np.float32)
    for a in range(0,spec.count,chunk):
        b=min(spec.count,a+chunk)
        xyz=spec.locations(np.arange(a,b))
        query=o3d.core.Tensor(np.ascontiguousarray(xyz,np.float32))
        dist[a:b]=np.asarray(scene.compute_distance(query).numpy(),dtype=np.float32).reshape(-1)
    return dist.reshape(spec.shape)


def _winding_prior(spec, mesh, interior, exterior, igl, *, chunk=16000):
    # Signed anchor difference calibrates the winding to this particular
    # OPEN surface; no universal 0/0.5/1 threshold is assumed.
    v=np.asarray(mesh.vertices,dtype=np.float64);f=np.asarray(mesh.faces,dtype=np.int64)
    ix=np.concatenate([interior[:3000],exterior[:3000]])
    if len(interior)<10 or len(exterior)<10:
        return np.full(spec.shape,.5,dtype=np.float32),{'calibrated':False,'reason':'too_few_surface_seeds'}
    anchor=np.asarray(igl.fast_winding_number(v,f,np.ascontiguousarray(spec.locations(ix),np.float64)),dtype=np.float64).reshape(-1)
    w_in=anchor[:min(len(interior),3000)];w_out=anchor[min(len(interior),3000):]
    mu_in=float(np.median(w_in));mu_out=float(np.median(w_out))
    if not np.isfinite([mu_in,mu_out]).all() or abs(mu_in-mu_out)<.12:
        return np.full(spec.shape,.5,dtype=np.float32),{'calibrated':False,'median_in':mu_in,'median_out':mu_out}
    prior=np.empty(spec.count,dtype=np.float32)
    for a in range(0,spec.count,chunk):
        b=min(spec.count,a+chunk)
        xyz=np.ascontiguousarray(spec.locations(np.arange(a,b)),np.float64)
        w=np.asarray(igl.fast_winding_number(v,f,xyz),dtype=np.float64).reshape(-1)
        if len(w)!=len(xyz) or not np.isfinite(w).all():raise RuntimeError('Nonfinite winding on implicit grid')
        prior[a:b]=np.clip((w-mu_out)/(mu_in-mu_out),.0,1.)
    return prior.reshape(spec.shape),{'calibrated':True,'median_in':mu_in,'median_out':mu_out}


def solve_binary_grid(free_support,inside_seed,outside_seed,distances,prior,*,
                      smooth_weight=1.5,prior_weight=.35,surface_width=.2):
    """Energy minimization by PyMaxflow (full 6-neighbor grid connectivity).

    Label 1=exterior/source; 0=interior/sink. Distinct source/sink unary
    costs, seed conflicts fail loudly instead of deleting Gaussians.
    """
    try:
        import maxflow
    except ImportError as e:
        raise RuntimeError('Implicit graph cut needs PyMaxflow: python -m pip install PyMaxflow') from e
    free=np.asarray(free_support)
    i=np.asarray(inside_seed,dtype=bool)
    o=np.asarray(outside_seed,dtype=bool)
    ds=np.asarray(distances,dtype=np.float32)
    w=np.asarray(prior,dtype=np.float32)
    if free.shape!=i.shape or free.shape!=o.shape or free.shape!=ds.shape or free.shape!=w.shape or free.ndim!=3:
        raise ValueError('Grid cut input shapes disagree')
    ext=o|(free>0)
    if np.any(ext&i):raise ValueError('Contradictory inside/exterior voxel seeds')
    if np.count_nonzero(ext)<10 or np.count_nonzero(i)<10:
        raise ValueError('Too few trustworthy occupied/free-space seeds to solve occupancy')
    n=ds.size
    if n>800000:raise ValueError('Refusing oversized grid graph')
    g=maxflow.Graph[float](int(n),int(n*3))
    g.add_nodes(int(n))
    # Source = exterior. add_tedge(i, cap_source, cap_sink):
    # when node goes to source, cap_sink is cut; source label requires
    # cap_source high to force source; interior requires cap_sink high.
    conf=np.clip(w.reshape(-1),0,1)
    source_cap=(prior_weight*(1-conf)).astype(np.float64)
    sink_cap=(prior_weight*conf).astype(np.float64)
    # Cut cost source = sink_cap, cut cost sink = source_cap.
    # Strong exterior: source cap high. Strong interior: sink cap high.
    source_cap[ext.reshape(-1)]=1000.
    sink_cap[ext.reshape(-1)]=0.
    source_cap[i.reshape(-1)]=0.
    sink_cap[i.reshape(-1)]=1000.
    for j in range(n):
        g.add_tedge(j,float(source_cap[j]),float(sink_cap[j]))
    # Surface is cheap to cut; elsewhere smoothness discourages fake interfaces.
    grad=np.asarray(ds,dtype=np.float32)
    near=np.maximum(0,1-np.exp(-grad/ max(float(surface_width),1.e-6)))
    pairs=0
    for axis in range(3):
        left=[slice(None)]*3;right=[slice(None)]*3
        left[axis]=slice(None,-1);right[axis]=slice(1,None)
        x=np.arange(n,dtype=np.int32).reshape(ds.shape)
        ii=x[tuple(left)].ravel();jj=x[tuple(right)].ravel()
        # Using small positive barrier near source surface enables split.
        strength=smooth_weight*(.08+.92* np.minimum(near[tuple(left)],near[tuple(right)]).ravel())
        for a,b,c in zip(ii,jj,strength):
            g.add_edge(int(a),int(b),float(c),float(c))
        pairs+=len(ii)
    energy=float(g.maxflow())
    labels=np.fromiter((g.get_segment(j)==0 for j in range(n)),count=n,dtype=bool)
    return labels.reshape(ds.shape),{'energy':energy,'nodes':int(n),'edges_undirected':int(pairs),
                                    'inside_seeds':int(np.count_nonzero(i)),
                                    'exterior_seeds':int(np.count_nonzero(ext))}


def exterior_connected_to_observations(binary_exterior, observed_exterior):
    """Reject exterior components that have no observed free-space evidence."""
    from scipy import ndimage
    ext=np.asarray(binary_exterior,dtype=bool)
    support=np.asarray(observed_exterior,dtype=bool)
    labels,n=ndimage.label(ext,structure=ndimage.generate_binary_structure(3,1))
    tagged=np.unique(labels[ext & support]);tagged=tagged[tagged!=0]
    return np.isin(labels,tagged)&ext,{'exterior_components':int(n),
                                     'observed_exterior_components':int(len(tagged))}


def boundary_points(mesh):
    """Get raw open boundary vertices without changing any mesh geometry."""
    faces=np.asarray(mesh.faces,dtype=np.int64)
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]],axis=0),axis=1)
    uniq,counts=np.unique(edges,axis=0,return_counts=True)
    return np.asarray(mesh.vertices,dtype=np.float64)[np.unique(uniq[counts==1])]


def query_stable_labels(points, grids, boundary_xyz, *, boundary_margin):
    """Require BOTH resolutions to call exterior, protect near open boundaries."""
    from scipy.spatial import cKDTree
    xyz=np.asarray(points,dtype=np.float64).reshape(-1,3)
    decisions=[]
    for spec,field in grids:
        idx=spec.index(xyz)
        label=np.zeros(len(xyz),dtype=np.int8)
        ok=idx>=0
        label[ok]=field.ravel()[idx[ok]]
        decisions.append(label)
    out=(decisions[0]==OUTSIDE)&(decisions[1]==OUTSIDE)
    inside=(decisions[0]==INSIDE)&(decisions[1]==INSIDE)
    if boundary_xyz is not None and len(boundary_xyz):
        sep=cKDTree(np.asarray(boundary_xyz)).query(xyz,k=1,workers=-1)[0]
        out &= sep>boundary_margin
        inside &= sep>boundary_margin
    result=np.full(len(xyz),UNKNOWN,dtype=np.int8)
    result[out]=OUTSIDE;result[inside]=INSIDE
    return result


def solve_one_grid(mesh,scene,o3d,igl,spec,cam_poses,ray_origins,ray_targets,*,
                   endpoint_margin,seed_max_faces=4000,prior_weight=.35,
                   smooth_weight=1.5):
    """One scale of observed free-space constrained voxel graph cut."""
    from scipy.spatial import cKDTree
    ray_end=ray_targets-ray_origins
    ray_len=np.linalg.norm(ray_end,axis=1)
    # Keep only observation-supported free-space BEFORE SfM endpoint.
    free_end=ray_origins+ray_end*np.maximum(0,(ray_len-endpoint_margin)/np.maximum(ray_len,1e-12))[:,None]
    print('[stage2-open-ab-v2]   B1 rasterize verified COLMAP free-space rays',flush=True)
    support,ray_stats=rasterize_segments(spec,ray_origins,free_end)
    camera_xyz=np.asarray([p[0] for p in cam_poses.values()])
    print('[stage2-open-ab-v2]   B2 derive observed Poisson orientation seeds',flush=True)
    inside_idx,outside_idx,seed_stats=select_surface_seeds(mesh,camera_xyz,scene,o3d,spec,
        max_faces=seed_max_faces,observation_origins=ray_origins,observation_targets=ray_targets)
    inside=np.zeros(spec.count,dtype=bool);outside=np.zeros(spec.count,dtype=bool)
    inside[inside_idx]=True;outside[outside_idx]=True
    inside=inside.reshape(spec.shape);outside=outside.reshape(spec.shape)
    # A physically observed free-space voxel cannot be interior. If the
    # constructed inner seed touches observed ray: reject, never overwrite.
    conflict=inside & ((support>0)|outside)
    conflict_count=int(np.count_nonzero(conflict))
    # Ambiguous surface points are NOT strong interior anchors.
    inside[conflict]=False
    if conflict_count>max(50,int(.20*max(len(inside_idx),1))):
        raise ValueError('Too many contradictory surface/observed free-space seeds')
    print('[stage2-open-ab-v2]   B3 compute open-mesh winding soft prior',flush=True)
    prior,calib=_winding_prior(spec,mesh,inside_idx,outside_idx,igl)
    print('[stage2-open-ab-v2]   B4 compute grid-to-original-mesh distance',flush=True)
    dist=_grid_distances(spec,scene,o3d)
    print('[stage2-open-ab-v2]   B5 solve constrained spatial graph cut',flush=True)
    exterior,cut=solve_binary_grid(support,inside,outside,dist,prior,
                    prior_weight=prior_weight,smooth_weight=smooth_weight,
                    surface_width=max(1.3*spec.voxel,.1))
    print('[stage2-open-ab-v2]   B6 require exterior connectivity to observed rays',flush=True)
    exterior_connected,conn=exterior_connected_to_observations(exterior,(support>0)|outside)
    from scipy import ndimage
    # UNKNOWN where observed boundary has gaps, seed conflict or ambiguous
    # very-near-surface discretization. No arbitrary exterior label there.
    protected=dist<=max(1.25*spec.voxel,.06)
    field=np.full(spec.shape,UNKNOWN,dtype=np.int8)
    field[exterior_connected & ~protected]=OUTSIDE
    field[(~exterior) & ~protected]=INSIDE
    info={'grid_shape':list(spec.shape),'voxel':float(spec.voxel),'node_count':int(spec.count),
          'ray_support':ray_stats,'surface_seed_stats':seed_stats,
          'winding_prior':calib,'graph_cut':cut,'connectivity':conn,
          'outside_confident_cells':int((field==OUTSIDE).sum()),
          'inside_confident_cells':int((field==INSIDE).sum()),
          'uncertain_cells':int((field==UNKNOWN).sum()),
          'near_original_surface_protected_cells':int(protected.sum()),
          'rejected_conflicting_inside_seeds':conflict_count}
    return field,info
