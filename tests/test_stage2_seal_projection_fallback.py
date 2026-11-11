"""Bounded tilted-cap fallback; no observed-surface changes, fail-closed behavior.

These interface tests mock sealing and camera checks. Final geometry requires
real dataset validation via stage2-cleanup --phase seal.
"""
import sys
import types
import numpy as np
import trimesh
from shapely.geometry import Polygon
import pytest
from gsrefine.cleanup import direct_r3_pipeline as mod


def _basis(up):
    up = np.asarray(up, float)
    up /= np.linalg.norm(up)
    pref = np.eye(3)[int(np.argmin(np.abs(up)))]
    t = pref - up * np.dot(pref, up)
    t /= np.linalg.norm(t)
    s = np.cross(up, t)
    return np.column_stack((t, s, up))


def _twisted_quad(h):
    v = np.array([[-1., -1., 0.], [1., 1., h],
                  [-1., 1., 0.], [1., -1., -h]], float)
    m = trimesh.Trimesh(vertices=v, faces=[[0, 1, 2], [0, 2, 3]], process=False)
    assert m.is_winding_consistent
    assert mod.boundary_diagnostics(m)['preseal_one_ring_eligible']
    assert not Polygon(v[:, :2]).is_valid
    return m


def _mock_environment(monkeypatch, *, intersect=False):
    fake_d = types.ModuleType('gsrefine.cleanup.sdf_common')
    fake_d._local_basis = _basis
    fake_d._raycast = lambda *a: object()
    fake_d._camera_check = lambda *a: {'outside_fraction':1.0,
                                     'outward_normal_agreement':1.0,
                                     'nonfinite_distances':0}
    fake_s = types.ModuleType('gsrefine.geometry.sdf_sealing')
    fake_s.boundary_loops = lambda mesh: ([np.array([0, 1, 2, 3])], {'boundary_edges':4})
    class SimulatedSolid(trimesh.Trimesh):
        @property
        def body_count(self): return 1
    def sealer(mesh, axis, sign, margin):
        uv = np.asarray(mesh.vertices)[:, :2]
        if not Polygon(uv).is_valid:
            raise ValueError('Outer boundary projection is invalid or collapses; skirt would self-cross: Self-intersection[test]')
        block=trimesh.creation.box()
        original=np.asarray(mesh.vertices)
        simulated=SimulatedSolid(vertices=np.vstack((original,block.vertices)),
                                  faces=block.faces+len(original),process=False)
        return (simulated, {'mesh_before_faces':len(mesh.faces),
                            'mesh_before_vertices':len(mesh.vertices)}, block)
    fake_s.seal_main = sealer
    class O3Mesh:
        def __init__(self,*a):pass
        def is_self_intersecting(self):return intersect
        def is_vertex_manifold(self):return True
        def is_edge_manifold(self,allow_boundary_edges=False):return True
    fake_o3d = types.ModuleType('open3d')
    fake_o3d.geometry = types.SimpleNamespace(TriangleMesh=O3Mesh)
    fake_o3d.utility = types.SimpleNamespace(Vector3dVector=lambda v:v,
                                           Vector3iVector=lambda v:v)
    monkeypatch.setitem(sys.modules, 'gsrefine.cleanup.sdf_common', fake_d)
    monkeypatch.setitem(sys.modules, 'gsrefine.geometry.sdf_sealing', fake_s)
    monkeypatch.setitem(sys.modules, 'open3d', fake_o3d)


def test_valid_bounded_tilt_recovers_projection_only_failure(monkeypatch):
    _mock_environment(monkeypatch)
    observed=_twisted_quad(5.)
    before=observed.vertices.copy()
    solid,report=mod.attempt_seal(observed,np.array([[0.,0.,10.]]),
                                  np.array([0.,0.,1.]),dict(mod.DEFAULTS))
    assert solid is not None and report['passed']
    assert report['projection_audit']['bounded_tilt_fallback_triggered']
    assert report['projection_audit']['simple_projections_found'] >= 1
    assert 0 < report['selected_tilt_degrees'] <= 25
    assert np.array_equal(observed.vertices,before)
    assert len(report['direction_trials']) > 2
    assert all('Outer boundary projection is invalid' in t['failure']
               for t in report['direction_trials'][:2])
    assert not np.allclose(report['selected_sealing_normal_xyz'], [0,0,1])


def test_no_valid_projection_in_bounded_cone_stays_blocked(monkeypatch):
    _mock_environment(monkeypatch)
    observed=_twisted_quad(1.)
    solid,report=mod.attempt_seal(observed,np.array([[0.,0.,10.]]),
                                  np.array([0.,0.,1.]),dict(mod.DEFAULTS))
    assert solid is None
    assert report['status']=='BLOCKED_SEALING'
    assert report['projection_audit']['simple_projections_found']==0
    assert report['projection_audit']['directions_examined']==5*16
    assert len(report['direction_trials'])==2


def test_valid_original_projection_with_true_intersection_does_not_tilt(monkeypatch):
    _mock_environment(monkeypatch,intersect=True)
    mesh=_twisted_quad(5.)
    # Replace the original quadrilateral with an un-crossed projection.
    mesh.vertices[[1,2]] = mesh.vertices[[2,1]]
    solid,report=mod.attempt_seal(mesh,np.array([[0,0,10]]),np.array([0,0,1]),dict(mod.DEFAULTS))
    assert solid is None
    assert report['projection_audit']['bounded_tilt_fallback_triggered'] is False
    assert len(report['direction_trials'])==2


def test_refuses_nonconservative_tilt_budget(monkeypatch):
    _mock_environment(monkeypatch)
    with pytest.raises(ValueError, match='maximum tilt'):
        mod._tilted_sealing_projections(_twisted_quad(5.),np.array([0,0,1]),
                                       {**mod.DEFAULTS,'seal_max_tilt_deg':90})
