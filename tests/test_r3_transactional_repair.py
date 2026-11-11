"""Transaction gate and local sheet-group regression tests for Stage 2."""
import numpy as np
import trimesh
from gsrefine.geometry.roi_topology_repair import (
    repair_r3_roi_topology, _topology_state, _localized_edge_groups)


def grid(n=8):
    xyz=np.array([(i,j,0.) for j in range(n+1) for i in range(n+1)],float)
    faces=[]
    for j in range(n):
        for i in range(n):
            a=j*(n+1)+i;b=a+1;c=a+n+1;d=c+1
            faces.extend([(a,b,c),(b,d,c)])
    return trimesh.Trimesh(vertices=xyz,faces=faces,process=False)


def test_four_face_joint_two_unsupported_fins_transactionally_deleted():
    base=grid(8)
    v=np.vstack([base.vertices,[55.,55.,55.],[54.,54.,54.]])
    # Edge (10,11) already meets TWO genuine base triangles.
    f=np.vstack([base.faces,[10,11,len(v)-2],[10,11,len(v)-1]])
    m=trimesh.Trimesh(vertices=v,faces=f,process=False)
    before=_topology_state(m)
    assert before['nonmanifold_edges']==1
    out,report=repair_r3_roi_topology(m,base.vertices,base.vertices,
            np.full(len(base.vertices),0.07),max_island_fraction=.06)
    after=_topology_state(out)
    assert after['nonmanifold_edges']==0
    assert after['open_edges']==_topology_state(base)['open_edges']
    assert report['new_open_boundary_edges']==0
    assert report['original_r3_ring_preserved']
    assert len(out.faces)==len(base.faces)
    assert any(a['decision']=='accepted' for a in report['actions'])


def test_four_face_joint_observed_extension_must_remain():
    base=grid(8)
    v=np.vstack([base.vertices,[55.,55.,55.],[54.,54.,54.]])
    f=np.vstack([base.faces,[10,11,len(v)-2],[10,11,len(v)-1]])
    m=trimesh.Trimesh(vertices=v,faces=f,process=False)
    gauss=np.vstack([base.vertices,[54.8,54.8,54.8]])
    radii=np.r_[np.full(len(base.vertices),0.07),1.]
    out,report=repair_r3_roi_topology(m,base.vertices,gauss,radii,
                                     max_island_fraction=.06)
    # At least the supported fin must remain; nothing is silently destroyed.
    assert len(out.faces)>=len(base.faces)+1
    assert report['new_open_boundary_edges']==0
    assert report['original_r3_ring_preserved']


def test_nonmanifold_local_planar_grouping_diagnostic():
    base=grid(8)
    vertex=np.vstack([base.vertices,[55,55,55],[56,56,56]])
    faces=np.vstack([base.faces,[10,11,len(vertex)-2],[10,11,len(vertex)-1]])
    m=trimesh.Trimesh(vertices=vertex,faces=faces,process=False)
    groups=_localized_edge_groups(m)
    assert sum(g['nonmanifold_edge_count'] for g in groups)==1


def test_unsupported_single_island_keeps_verified_r3_main_boundary():
    base=grid(8)
    outside=trimesh.Trimesh(vertices=[[90,90,0],[91,90,0],[90,91,0]],
                             faces=[[0,1,2]],process=False)
    merged=trimesh.util.concatenate([base,outside])
    out,report=repair_r3_roi_topology(merged,base.vertices,base.vertices,
                np.full(len(base.vertices),.07),max_island_fraction=.05)
    assert report['original_r3_ring_preserved']
    assert report['new_open_boundary_edges']==0
    assert report['total_removed_faces']==1
    assert len(out.faces)==len(base.faces)


def test_zero_new_boundary_global_invariant_on_unsupported_fin():
    base=grid(8)
    xyz=np.vstack([base.vertices,[90,90,90]])
    faces=np.vstack([base.faces,[10,11,len(xyz)-1]])
    m=trimesh.Trimesh(vertices=xyz,faces=faces,process=False)
    original=_topology_state(m)
    out,rep=repair_r3_roi_topology(m,base.vertices,base.vertices,
                                  np.full(len(base.vertices),.07),max_island_fraction=.05)
    assert not (_topology_state(out)['open_edges'] - original['open_edges'])
    assert rep['original_r3_ring_preserved']
