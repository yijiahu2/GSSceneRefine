"""Module 4 should export exactly the protected original Gaussian records."""
import json
import subprocess
import sys

import numpy as np
import pytest

from gsrefine.roi.core import GaussianPLY
from gsrefine.roi.finalize import finalize
from gsrefine.roi.study import run as study_run
from .test_demo import build_synthetic_case


def _study(tmp_path, *, ablation=True):
    rock, _ = build_synthetic_case(tmp_path)
    out = tmp_path / 'study'
    report = study_run(tmp_path/'config.yaml', tmp_path/'roi.json', out,
                       up_vector=[0,0,1], run_ablation=ablation)
    return rock, out, report


def test_default_study_produces_final_gaussian_scene_without_opt_in(tmp_path):
    rock, out, report = _study(tmp_path)
    target = out / 'gaussians_final_R3.ply'
    assert target.is_file()
    final = report['final_gaussian_scene']
    assert final['method'] == 'R3'
    assert final['output_ply'] == str(target)
    assert final['sdf_internal_cleanup_applied'] is False
    assert final['final_gaussians'] >= final['R0_gaussians']
    assert final['final_gaussians'] == final['R0_gaussians'] + final['added_gaussians']
    assert final['source_model_scope'] is None  # synthetic config omits scope
    assert GaussianPLY(target).vertex_count == int(np.load(out/'keep_mask_R3.npy').sum())
    assert json.loads((out/'final_gaussian_report_R3.json').read_text()) == final
    assert not any(['sfm_heuristic' in str(final), final['sdf_internal_cleanup_applied']])
    mask = np.load(out/'keep_mask_R3.npy')
    assert not mask[len(rock.vertices):].any()


def test_vertex_records_and_order_preserved_exactly(tmp_path):
    _, out, _ = _study(tmp_path)
    source = GaussianPLY(tmp_path/'model.ply')
    result = GaussianPLY(out/'gaussians_final_R3.ply')
    expected = source.arr[np.load(out/'keep_mask_R3.npy')]
    assert result.dtype == source.dtype
    assert result.arr.tobytes() == expected.tobytes()


def test_cached_mask_exports_another_method_without_recomputing(tmp_path):
    _, out, report = _study(tmp_path)
    summary = finalize(out, method='R2')
    assert summary['final_gaussians'] == int(np.load(out/'keep_mask_R2.npy').sum())
    assert (out/'gaussians_final_R2.ply').is_file()
    assert summary['source_gaussian_sha256'] == report['input_sha256']['gaussian']
    assert summary['validated']['all_R0_gaussians_retained']


def test_missing_added_mask_rejected(tmp_path):
    _, out, _ = _study(tmp_path)
    (out/'added_mask_R2.npy').unlink()
    with pytest.raises(FileNotFoundError, match='added_mask_R2'):
        finalize(out, method='R2')
    assert not (out/'gaussians_final_R2.ply').exists()


def test_dropping_original_R0_is_rejected(tmp_path):
    _, out, _ = _study(tmp_path)
    mask=np.load(out/'keep_mask_R2.npy')
    base=np.load(out/'keep_mask_R0.npy')
    mask[np.flatnonzero(base)[0]]=False
    np.save(out/'keep_mask_R2.npy',mask)
    # Legacy roi-study reports do not carry per-mask SHA256. They must still
    # enforce the invariant that a refined ROI never drops baseline Gaussians.
    record=json.loads((out/'study_report.json').read_text())
    record.pop('keep_mask_sha256', None)
    (out/'study_report.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='drops existing R0'):
        finalize(out, method='R2')


def test_modified_gaussian_source_is_rejected(tmp_path):
    _, out, _ = _study(tmp_path)
    with open(tmp_path/'model.ply','ab') as f:
        f.write(b'\0')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        finalize(out, method='R1')


def test_no_ablation_can_export_R0_but_not_R3(tmp_path):
    _, out, report = _study(tmp_path,ablation=False)
    assert not (out/'gaussians_final_R3.ply').exists()
    assert finalize(out, method='R0')['final_gaussians'] == report['R0_count']
    with pytest.raises(FileNotFoundError,match='keep_mask_R3'):
        finalize(out, method='R3')


def test_roi_finalization_cli_available(tmp_path):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    p=subprocess.run([sys.executable,str(root/'run.py'),'roi-finalize','--help'],
                     capture_output=True,text=True,cwd=root)
    assert p.returncode == 0
    assert '--study-dir' in p.stdout and '--method' in p.stdout
    q=subprocess.run([sys.executable,str(root/'run.py'),'roi-study','--help'],
                     capture_output=True,text=True,cwd=root)
    assert q.returncode == 0
    assert '--final-method' in q.stdout and '--no-final-export' in q.stdout


def test_tampered_mask_fails_hash_validation(tmp_path):
    _,out,_=_study(tmp_path)
    mask=np.load(out/'keep_mask_R2.npy')
    mask[0]=~mask[0]
    np.save(out/'keep_mask_R2.npy',mask)
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        finalize(out,method='R2')


def test_indices_reproduce_the_exact_extracted_rows(tmp_path):
    _,out,_=_study(tmp_path)
    idx=np.load(out/'kept_indices_R3.npy')
    added=np.load(out/'restored_indices_R3.npy')
    assert np.array_equal(idx,np.flatnonzero(np.load(out/'keep_mask_R3.npy')))
    assert np.array_equal(added,np.flatnonzero(np.load(out/'added_mask_R3.npy')))
    assert np.all(np.isin(added,idx))
