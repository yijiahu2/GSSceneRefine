import numpy as np
import trimesh
from gsrefine.cleanup.threshold_distance import classify_distance, exact
from gsrefine.cleanup.auxiliary_protection import distance_protection
from tools.audit_final_r3_global_poisson_barrier import make_scene


def test_distance_bounds_and_auxiliary_masks_match_exact(tmp_path, monkeypatch):
    mesh = trimesh.Trimesh([[0,0,0],[2,0,0],[0,2,0]], [[0,1,2]], process=False)
    scene, o3d = make_scene(mesh)
    rng = np.random.default_rng(15)
    # Dense clusters on both sides and at a float32 threshold.
    xyz = np.vstack([rng.normal([.4,.4,z], .0001, (512,3)) for z in (.01, .1, 1.)])
    xyz = np.vstack([xyz, [.4,.4,float(np.float32(.1))]])
    threshold = float(np.float32(.1))
    baseline = exact(scene,o3d,xyz)
    d, over, stats = classify_distance(scene,o3d,xyz,threshold,mesh.bounds,128,True,lambda msg: None)
    assert stats['cell_near'] and stats['cell_far']
    assert np.array_equal(over, baseline.astype(float)>threshold)
    assert np.array_equal(d[np.isfinite(d)], baseline[np.isfinite(d)])
    path = tmp_path/'aux.ply'
    mesh.export(path)
    class Model:
        vertex_count = len(xyz)
        def selected_xyz(self, ids):
            return xyz[ids]
    cand = np.ones(len(xyz), bool)
    monkeypatch.delenv('GSREFINE_EXACT_DISTANCES', raising=False)
    protected, info, aux = distance_protection(Model(),cand,threshold,path,chunk_size=100)
    assert np.array_equal(protected, ~over)
    assert np.isnan(aux).sum() == info['aabb_rejected'] > 0
