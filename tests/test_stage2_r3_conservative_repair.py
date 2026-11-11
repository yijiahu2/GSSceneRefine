"""Deterministic synthetic audits; no real Songzi Guanyin files required."""
import sys
import types
import numpy as np
import pytest
import trimesh
from gsrefine.geometry.roi_topology_repair import repair_r3_roi_topology
from gsrefine.cleanup import direct_r3_pipeline as pipeline


def plane_grid(n=10):
    xyz = np.array([[i,j,0] for j in range(n+1) for i in range(n+1)],float)
    faces = []
    for j in range(n):
        for i in range(n):
            a=j*(n+1)+i; b=a+1;c=a+(n+1);d=c+1
            faces.extend([(a,b,c),(b,d,c)])
    return trimesh.Trimesh(vertices=xyz, faces=faces, process=False)


def source_evidence(grid):
    # Vertex sampling at regular grid spacing; Gaussian spheres preserve core.
    xyz=np.asarray(grid.vertices,float)
    return xyz,xyz.copy(),np.full(len(xyz),.07)


def test_tiny_both_unsupported_island_is_removed_without_warping_main():
    main=plane_grid()
    island=trimesh.Trimesh(vertices=[[80,80,0],[81,80,0],[80,81,0]],faces=[[0,1,2]],process=False)
    merged=trimesh.util.concatenate([main,island])
    sfm,g,r=source_evidence(main)
    out,info=repair_r3_roi_topology(merged,sfm,g,r,max_island_fraction=.05)
    assert len(out.faces)==len(main.faces)
    assert info['total_removed_faces']==1
    assert np.array_equal(np.asarray(out.vertices),np.asarray(main.vertices))
    assert pipeline.boundary_diagnostics(out)['open_boundary_components']==1


def test_supported_island_is_preserved_even_if_tiny():
    main=plane_grid()
    island=trimesh.Trimesh(vertices=[[80,80,0],[81,80,0],[80,81,0]],faces=[[0,1,2]],process=False)
    merged=trimesh.util.concatenate([main,island])
    sfm,g,r=source_evidence(main)
    g=np.vstack([g,[80.2,80.2,0]])
    r=np.append(r,.3)
    out,info=repair_r3_roi_topology(merged,sfm,g,r,max_island_fraction=.05)
    assert len(out.faces)==len(merged.faces)
    assert not pipeline.boundary_diagnostics(out)['preseal_one_ring_eligible']


def test_same_triangle_dedup_and_no_coordinate_move():
    grid=plane_grid(5)
    faces=np.vstack((grid.faces,grid.faces[0]))
    input=trimesh.Trimesh(vertices=grid.vertices,faces=faces,process=False)
    s,g,r=source_evidence(grid)
    out,rep=repair_r3_roi_topology(input,s,g,r)
    assert len(out.faces)==len(grid.faces)
    assert np.array_equal(out.vertices,grid.vertices)
    assert rep['hygiene']['duplicate_faces_removed']==1


def test_large_unsupported_island_not_automatically_deleted():
    a=plane_grid(12)
    b=plane_grid(3).copy().apply_translation([80,80,0])
    both=trimesh.util.concatenate([a,b])
    s,g,r=source_evidence(a)
    out,rep=repair_r3_roi_topology(both,s,g,r,max_island_faces=4,max_island_fraction=.05)
    assert len(out.faces)==len(both.faces)
    assert not pipeline.boundary_diagnostics(out)['preseal_one_ring_eligible']


def test_nonmanifold_with_supported_fin_cannot_be_silently_destroyed():
    plane=plane_grid(6)
    xyz=np.vstack([plane.vertices,[1.5,.6,.3]])
    f=np.vstack([plane.faces,[8,9,len(xyz)-1]])
    raw=trimesh.Trimesh(vertices=xyz,faces=f,process=False)
    s,g,r=source_evidence(plane)
    g=np.vstack([g,[1.5,.6,.3]])
    r=np.r_[r, .2]
    fixed,rep=repair_r3_roi_topology(raw,s,g,r,max_island_fraction=.05)
    assert len(fixed.faces)==len(raw.faces)
    assert pipeline.boundary_diagnostics(fixed)['nonmanifold_edges']>0 or not pipeline.boundary_diagnostics(fixed)['preseal_one_ring_eligible']


def test_density_uses_full_core_even_for_subset(monkeypatch):
    # Full 3-D grid; estimate residuals on sample but KNN on all source points.
    full=np.array([[i*.05,j*.05,k*.05] for i in range(12) for j in range(12) for k in range(12)],float)
    fake=types.ModuleType('gsrefine.roi.core')
    fake.exact_sfm_to_poisson_distance_open3d=lambda m,points,batch:np.full(len(points),.08)
    monkeypatch.setitem(sys.modules,'gsrefine.roi.core',fake)
    subset=pipeline.sfm_fit(None,full,max_points=200)
    complete=pipeline.sfm_fit(None,full,max_points=0)
    assert subset['spacing_calculated_using_full_core']
    assert subset['sfm_knn6_spacing_median']==pytest.approx(complete['sfm_knn6_spacing_median'],rel=.01)
    assert subset['p95']==pytest.approx(complete['p95'])


def test_default_no_sampling_in_every_phase():
    import inspect
    assert inspect.signature(pipeline.run).parameters['fit_max_points'].default==0
    assert inspect.signature(pipeline._geometry_record).parameters['max_fit_points'].default==0


def test_real_nonmanifold_with_unsupported_extension_may_be_repaired():
    plane=plane_grid(8)
    # Interior shared edge [10,11] already has two incident normal faces.
    xyz=np.vstack([plane.vertices,[50.0,50.0,50.0]])
    raw=trimesh.Trimesh(vertices=xyz,
                         faces=np.vstack([plane.faces,[10,11,len(xyz)-1]]),process=False)
    assert pipeline.boundary_diagnostics(raw)['nonmanifold_edges'] == 1
    sfm,g,r=source_evidence(plane)
    fixed,rep=repair_r3_roi_topology(raw,sfm,g,r,max_island_fraction=.02)
    # The long unsupported fin shares the edge with the body but is separated
    # by nonmanifold adjacency and may be removed with no vertex relocation.
    assert rep['total_removed_faces'] == 1
    assert pipeline.boundary_diagnostics(fixed)['nonmanifold_edges'] == 0
    assert pipeline.boundary_diagnostics(fixed)['preseal_one_ring_eligible']


def test_exact_coordinate_duplicates_can_be_welded_without_moving_geometry():
    base=plane_grid(3)
    xyz=np.vstack([base.vertices,base.vertices[0]])
    f=base.faces.copy()
    f[0,0]=len(xyz)-1
    dup=trimesh.Trimesh(vertices=xyz,faces=f,process=False)
    sfm,g,r=source_evidence(base)
    fixed,info=repair_r3_roi_topology(dup,sfm,g,r)
    assert info['hygiene']['exact_identical_coordinate_vertices_merged'] == 1
    assert np.array_equal(fixed.vertices,base.vertices)
    assert np.array_equal(fixed.faces,base.faces)


def test_original_r3_perimeter_is_unchanged_by_safe_island_removal():
    from gsrefine.geometry.roi_topology_repair import boundary_coordinate_signature
    main=plane_grid(10)
    island=trimesh.Trimesh(vertices=[[40,40,0],[41,40,0],[40,41,0]],faces=[[0,1,2]],process=False)
    original=trimesh.util.concatenate([main,island])
    s,g,r=source_evidence(main)
    fixed,_=repair_r3_roi_topology(original,s,g,r,max_island_fraction=.05)
    assert boundary_coordinate_signature(main)==boundary_coordinate_signature(fixed)
