"""Unit checks for the package's new read-only modules, no GSSceneRefine install required."""
import hashlib
import json
import runpy
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np


src = runpy.run_path(str(Path(__file__).with_name('install.py')))
NEW = src['NEW']


def module(name):
    ns = {'__name__':name}
    exec(compile(NEW[name], name, 'exec'), ns)
    return ns


def run():
    ps = module('gsrefine/geometry/poisson_support.py')
    pts = np.array([[0,0,0],[1,0,0],[0,1,0],[.5,.5,0],[.25,.25,0]],float)
    mesh = SimpleNamespace(vertices=pts[:3], faces=np.array([[0,1,2]],int))
    desc = ps['diagnose'](mesh, pts, sfm_spacing=0.1)
    assert desc['sampled_faces']==1
    assert desc['status']=='REVIEW_ONLY_NOT_GROUND_TRUTH'
    eg = module('gsrefine/cleanup/engineering_audit.py')
    with tempfile.TemporaryDirectory() as d:
        r = Path(d)
        m = r/'metadata'
        m.mkdir()
        file = m/'stage1_masks.npz'
        original = np.array([1,1,0,0,0,0],bool)
        core = np.array([1,1,1,1,0,0],bool)
        safety = np.array([0,0,0,0,1,0],bool)
        r3 = core|safety
        np.savez_compressed(file,R0=original,core=core,safety=safety,R3=r3)
        ply = r/'final'/'gaussian_roi.ply'
        ply.parent.mkdir()
        ply.write_bytes(b'fake-for-fingerprint-binding-only')
        (m/'stage1_masks_integrity.json').write_text(json.dumps({
            'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
            'r3_ply':str(ply.resolve())}),encoding='utf8')
        remove=np.array([0,0,1,0,1],bool)
        audit=eg['cross_stage'](r,ply,remove,input_count=5)
        assert audit['group_counts']['R0']['removed']==0
        assert audit['group_counts']['recovered_core']['removed']==1
        assert audit['group_counts']['safety_added']['removed']==1
        assert audit['removed_total']==2
        risk=eg['classification_risk'](points=np.zeros((5,3)),remove=remove,
            outside=remove,withheld=np.zeros(5,bool),protected=np.zeros(5,bool),
            delta=.1,scope={'anchor':[0,0,0]})
        assert risk['deletion_invariant_passed'] is True
        try:
            eg['classification_risk'](points=np.zeros((5,3)),remove=remove,
                outside=remove,withheld=remove,protected=np.zeros(5,bool),
                delta=.1,scope={})
            assert False, 'invalid deletion should be rejected'
        except ValueError:
            pass
    print('PASS: Poisson support diagnostic, provenance mask cross-tab, conservative deletion gate')


if __name__=='__main__':
    run()
