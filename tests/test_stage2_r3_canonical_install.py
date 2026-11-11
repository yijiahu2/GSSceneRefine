"""A controlled in-place migration must remove only obsolete *code*."""
from pathlib import Path
import json
import shutil

import numpy as np
import pytest

from tools.apply_stage2_r3_canonical import apply, REMOVE_CODE, REQUIRED
from gsrefine.cleanup.sdf_common import classify_direct_sdf, robust_gaussian_radius


def sandbox(tmp_path):
    root=tmp_path/'GSSceneRefine'
    payload=Path(__file__).resolve().parents[1]
    for name in REQUIRED:
        p=root/name
        p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(payload/name,p)
    runner=root/'run.py';runner.write_text('''COMMANDS = {
    'stage2-cleanup':'gsrefine.cleanup.direct_sdf',
    'stage2-sfm':'gsrefine.geometry.r3_sfm_bridge',
    'stage2-r3':'gsrefine.cleanup.direct_r3_pipeline',
}
''')
    publisher=root/'gsrefine/roi_init/output_layout.py';publisher.parent.mkdir(parents=True,exist_ok=True)
    publisher.write_text('''def publish(work,meta,paths,roi,downstream):
    # Stage 2 may reuse a certified surface mask already made in Stage 1.
    # This is a no-op if Stage 1 did not produce the optional handoff.
    from gsrefine.geometry.stage1_handoff import retain_stage1_surface_faces
    _stage2_faces = retain_stage1_surface_faces(work, meta, paths['poisson_raw'], roi)
    if _stage2_faces is not None:
        downstream['roi_surface_faces'] = str(_stage2_faces)
    return downstream
''')
    for name in REMOVE_CODE:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('legacy-script')
    output=root/'results/roi_001/cleanup/geometry/trusted_roi/old/trusted_roi_poisson.ply'
    output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(b'preserve historical geometry')
    return root, runner, output


def test_safe_checked_canonical_migration(tmp_path):
    root,runner,out=sandbox(tmp_path)
    original=runner.read_text()
    pre=apply(root,execute=False)
    assert pre['changes']==2
    assert len(list(root.rglob('trusted_roi.py'))) == 1
    assert runner.read_text()==original and out.read_bytes()==b'preserve historical geometry'
    info=apply(root,execute=True)
    assert info['retired']==len(REMOVE_CODE)
    assert "'stage2-cleanup':'gsrefine.cleanup.direct_r3_pipeline'" in runner.read_text()
    assert "'stage2-r3':" not in runner.read_text()
    assert "'stage2-sfm':" in runner.read_text()
    assert all(not (root/name).exists() for name in REMOVE_CODE)
    assert out.read_bytes()==b'preserve historical geometry'
    assert apply(root,execute=True)=={'changes':0,'retired':0}


def test_unknown_runner_aborts_without_removal(tmp_path):
    root,runner,out=sandbox(tmp_path)
    runner.write_text('COMMANDS = {}\n')
    with pytest.raises(RuntimeError,match='Unrecognized'):
        apply(root,execute=True)
    assert (root/'gsrefine/cleanup/direct_sdf.py').exists()
    assert out.exists()


def test_require_complete_package(tmp_path):
    root,runner,out=sandbox(tmp_path)
    (root/'gsrefine/cleanup/sdf_common.py').unlink()
    with pytest.raises(FileNotFoundError,match='Incomplete'):
        apply(root,execute=True)
    assert runner.read_text().count('direct_sdf') == 1


def test_direct_sdf_decision_same_as_previous():
    keep, remove, unknown=classify_direct_sdf(np.array([-1.,0.,0.11,np.nan]), 0.10, 1.e-5)
    assert keep.tolist()==[True,True,False,True]
    assert remove.tolist()==[False,False,True,False]
    assert unknown.tolist()==[False,False,False,True]


def test_trimmed_major_radius_matches_legacy_formula():
    class PLY:
        vertex_count=100
        arr={f'scale_{i}':np.log(np.linspace(.01,.2,100)).astype('float64') for i in range(3)}
    d=robust_gaussian_radius(PLY())
    radii=3*np.linspace(.01,.2,100)
    lo,hi=np.quantile(radii,[.1,.9])
    expected=radii[(radii>=lo)&(radii<=hi)].mean()
    assert d['trimmed_mean_radius']==pytest.approx(expected)
