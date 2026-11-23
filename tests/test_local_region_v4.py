"""Geometry regressions for the three recovery modes and distance/side decisions."""
import numpy as np
import trimesh

from gsrefine.cleanup import local_partition as geom
from gsrefine.cleanup.local_region_pipeline import establish
from gsrefine.cleanup.local_voxels import RegionTree, classify
from gsrefine.cleanup.stage2_runtime import run_scope
from gsrefine.cleanup.threshold_distance import exact
from tools import audit_final_r3_global_poisson_barrier as grid


class Box:
    def __call__(self,p):
        return 1.-np.abs(p).max(axis=1)

    def interior_radius(self,p):
        return np.maximum(0., self(p))


def sheet(hole=False):
    # Planar triangulated grid covering [-1,1]^3, optional internal opening.
    n=17
    v=np.array([(x,y,0.) for x in np.linspace(-1.,1.,n)
                         for y in np.linspace(-1.,1.,n)])
    f=[]
    for i in range(n-1):
        for j in range(n-1):
            a=i*n+j
            if hole and i in (10,11) and j in (9,10):
                continue
            f.extend([(a,a+n,a+1),(a+1,a+n,a+n+1)])
    return trimesh.Trimesh(vertices=v,faces=f,process=False)


@run_scope
def test_source_evidence_and_repaired_audit(tmp_path):
    bad=sheet(True)
    src=sheet(False)
    # The same missing patch is repairable only when original triangles exist.
    d=geom.Domain(Box())
    sc,o3d=grid.make_scene(bad)
    witnesses,_=geom.rim_witnesses(bad,d,sc,o3d,.12)
    assert len(witnesses)
    assert not len(geom.source_hits(bad,witnesses))
    assert len(geom.source_hits(src,witnesses))
    f=tmp_path/'local.ply';bad.export(f)
    sfm=tmp_path/'sfm.npz'
    xyz=np.random.default_rng(24).uniform(-.9,.9,(500,3))
    xyz[:,2]=.001
    np.savez(sfm, xyz=xyz, in_core_roi=np.ones(len(xyz),bool))
    data=dict(record=dict(sfm_fit={'p95':.001,'source_core_sfm_count':len(xyz)},
        sfm=str(sfm), sfm_fit_safeguards={'max_by_roi_diagonal':.1},
        main_observed_surface=str(f)),field=Box(),main=bad,global_main=src,
        up=np.array([0.,0.,1.]))
    index,mesh,_,fit,_,evidence=establish(data,np.full(3,-1.41),np.full(3,1.37),
        .12,np.array([-.5,-.4,.5]),tmp_path)
    assert index is not None and evidence['scope']=='full'
    assert evidence['main'].endswith('source_reclipped_main.ply')
    assert fit['fit_sfm_count']==len(xyz)
    assert np.abs(mesh.vertices[:,2]).max()==0


def test_independent_scope_excludes_real_gap(tmp_path):
    bad=sheet(True)
    path=tmp_path/'main.ply';bad.export(path)
    data=dict(record=dict(sfm_fit={'p95':.012,'source_core_sfm_count':300},
        main_observed_surface=str(path)),field=Box(),main=bad,global_main=bad,
        up=np.array([0.,0.,1.]))
    index,_,_,_,domain,evidence=establish(data,np.full(3,-1.41),np.full(3,1.37),.09,
        np.array([-.5,-.4,.5]),tmp_path)
    assert index is not None and evidence['scope']=='partial'
    assert domain.plane(np.array([[.43125,.2875,0.]]))[0]<0


def test_lookup_agrees_with_oriented_plane_and_safety_band():
    h=.2
    shape=(14,14,14)
    origin=np.array([-1.4,-1.4,-1.41])
    pts=grid.grid_centers(origin,h,shape)
    labels=np.where(Box()(pts)>0,np.where(pts[:,2]>0,0,1),-1).reshape(shape)
    idx=dict(origin=origin,voxel=h,shape=shape,labels=labels)
    d=geom.Domain(Box())
    distance=lambda p: abs(p[:,2])
    tree=RegionTree(idx,d,distance,.07,1e-8)
    rng=np.random.default_rng(87)
    q=rng.uniform(-.999,.999,(16000,3))
    q[:3,2]=[.07,-.07,.070001]
    def rays(p):
        return np.where(p[:,2]>0,0,1),{}
    remove,_,held,info=classify(q,tree,d,distance,.07,rays,batch=777)
    assert np.array_equal(remove, q[:,2]>.07)
    assert not held.any() and not tree.conflicts
    assert info['distance_queries']<len(q)
    assert info['short_ray_queries']<len(q)//3


def test_partial_domain_links_do_not_cross_excisions():
    d=geom.Domain(Box(),excluded_centers=[[0,0,0]],exclusion_radius=.06)
    p=np.array([[-.5,0.,0.],[-.5,.2,0.]])
    q=np.array([[.5,0.,0.],[.5,.2,0.]])
    assert d.allow_link(p,q).tolist()==[False,True]
