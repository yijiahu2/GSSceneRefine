"""CPU-only regressions for Stage-1 -> Stage-2 identity/resume, no Open3D/CUDA."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import hashlib

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    s = importlib.util.spec_from_file_location(name, str(path))
    obj = importlib.util.module_from_spec(s)
    s.loader.exec_module(obj)
    return obj

cli = module(ROOT/'tools/payload/full_pipeline.py', 'staged_full_pipeline')
installer = module(ROOT/'tools/install_engineering_unification_v1.py', 'staged_unification_installer')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def stage1(tmp_path, monkeypatch):
    root = tmp_path/'GSSceneRefine_results'
    run = root/'biechuanchansi'/'methods'/'Mip-Splatting'/'roi_003'
    roi = root/'biechuanchansi'/'roi'/'roi_003.json'
    roi.parent.mkdir(parents=True)
    roi.write_text('{"confirmed":true}', encoding='utf-8')
    (run/'metadata').mkdir(parents=True)
    (run/'final').mkdir()
    ply = run/'final/gaussian_roi.ply'
    ply.write_bytes(b'ply\nheader\nDATA')
    poisson = root/'biechuanchansi/geometry/poisson_surface.ply'
    poisson.parent.mkdir(parents=True)
    poisson.write_bytes(b'poisson')
    sfm = root/'biechuanchansi/geometry/cleaned_sfm.npz'
    sfm.write_bytes(b'sfm')
    mf = run/'metadata/manifest.json'
    original = {
        'schema':'GSSceneRefine.output.v3', 'status':'completed',
        'result_status':'ROI_RESTORED_NOT_ANOMALY_CLEANED',
        'input_sha256':{'roi':sha(roi),'poisson':sha(poisson),'sfm':sha(sfm)},
        'output_sha256':sha(ply),
        'stage2_inputs':{'roi_json':str(roi),'gaussian_ply':str(ply),
                         'poisson_mesh':str(poisson),'clean_sfm_npz':str(sfm)}
    }
    mf.write_text(json.dumps(original))

    # Match the real project's Stage-2 loader behavior: trust only SHA-authenticated
    # manifest, never a staged 'latest' filename or the path name alone.
    def loader(path):
        p = Path(path).resolve(strict=True)
        d = json.loads(p.read_text())
        if d.get('schema') != 'GSSceneRefine.output.v3' or d.get('status') != 'completed':
            raise ValueError('invalid manifest status')
        inp = {k:Path(v).resolve(strict=True) for k,v in d['stage2_inputs'].items()}
        for kind,name in [('roi','roi_json'),('sfm','clean_sfm_npz'),('poisson','poisson_mesh')]:
            if sha(inp[name])!=d['input_sha256'][kind]:
                raise ValueError('changed '+kind)
        if sha(inp['gaussian_ply'])!=d['output_sha256']:
            raise ValueError('changed Gaussian')
        return p,d,inp,p.parent.parent
    sdf = types.ModuleType('gsrefine.cleanup.sdf_common')
    sdf._load_stage1_manifest = loader
    gsrefine = sys.modules.get('gsrefine') or types.ModuleType('gsrefine')
    cleanup = sys.modules.get('gsrefine.cleanup') or types.ModuleType('gsrefine.cleanup')
    monkeypatch.setitem(sys.modules,'gsrefine',gsrefine)
    monkeypatch.setitem(sys.modules,'gsrefine.cleanup',cleanup)
    monkeypatch.setitem(sys.modules,'gsrefine.cleanup.sdf_common',sdf)
    return SimpleNamespace(root=root,run=run,roi=roi,ply=ply,poisson=poisson,sfm=sfm,mf=mf,original=original)


def args(stage1, *, resume=True):
    argv = ['--accept-auto-exterior', '--output-dir',str(stage1.root)]
    if resume:
        argv += ['--resume-manifest',str(stage1.mf)]
    return cli._parser().parse_args(argv)


def fake_stage2(monkeypatch, obj, succeed=True, bad_hash=False):
    issued=[]
    def subrun(cmd, check):
        assert check is False
        issued.append(cmd)
        output = Path(cmd[cmd.index('--output-dir')+1])
        if succeed:
            clean = output/'gaussian_roi_clean.ply'
            clean.write_bytes(b'ply\nCLEAN')
            report = {
                'status': 'COMPLETED_EXPERIMENTAL_CLEAN_ROI_GAUSSIAN',
                'input': {'stage1_manifest':str(obj.mf), 'stage1_manifest_sha256':sha(obj.mf),
                          'original_roi_gaussian':str(obj.ply), 'original_roi_gaussian_sha256':sha(obj.ply)},
                'outputs': {'final_supersplat_gaussian_ply':str(clean),
                            'final_ply_sha256':'broken' if bad_hash else sha(clean)},
                'counts':{'final_retained':4, 'final_removed':1},
            }
            (output/'stage2_final_report.json').write_text(json.dumps(report))
        return SimpleNamespace(returncode=0 if succeed else 1)
    monkeypatch.setattr(cli.subprocess,'run',subrun)
    return issued


def test_new_canonical_roi_identity_and_integrity(stage1):
    run,mf,d,inp = cli._validate_manifest(stage1.mf, output_root=stage1.root,
                                          expected_run=stage1.run, expected_roi=stage1.roi, expected_ply=stage1.ply)
    assert run==stage1.run and mf==stage1.mf and inp['roi_json']==stage1.roi
    assert not (stage1.run/'metadata/roi.json').exists()


def test_manual_edit_canonical_roi_rejects_resume(stage1):
    stage1.roi.write_text('{"edited": true}')
    with pytest.raises(ValueError,match='changed roi'):
        cli._validate_manifest(stage1.mf)


def test_manual_edit_gaussian_rejects_resume(stage1):
    stage1.ply.write_bytes(b'not original')
    with pytest.raises(ValueError,match='changed Gaussian'):
        cli._validate_manifest(stage1.mf)


def test_manifest_outside_requested_root_rejected(stage1, tmp_path):
    other = tmp_path/'not_my_root'
    other.mkdir()
    with pytest.raises(ValueError,match='outside --output-dir'):
        cli._validate_manifest(stage1.mf, output_root=other)


def test_legacy_roi_snapshot_remains_readable(stage1):
    snapshot = stage1.run/'metadata/roi.json'
    snapshot.write_bytes(stage1.roi.read_bytes())
    m=json.loads(stage1.mf.read_text())
    m['stage2_inputs']['roi_json']=str(snapshot)
    stage1.mf.write_text(json.dumps(m))
    (stage1.run/'metadata/processing_report.json').write_text(json.dumps({'source_roi':str(stage1.roi)}))
    status={'status':'completed','output_dir':str(stage1.run),'roi':str(stage1.roi),
            'roi_sha256':sha(stage1.roi),'final_ply':str(stage1.ply)}
    assert cli._completed_manifest(status)[0]==stage1.run


def test_legacy_source_swapped_is_rejected(stage1,tmp_path):
    snapshot = stage1.run/'metadata/roi.json'
    snapshot.write_bytes(stage1.roi.read_bytes())
    m=json.loads(stage1.mf.read_text())
    m['stage2_inputs']['roi_json']=str(snapshot)
    stage1.mf.write_text(json.dumps(m))
    wrong = tmp_path/'other.json';wrong.write_bytes(stage1.roi.read_bytes())
    (stage1.run/'metadata/processing_report.json').write_text(json.dumps({'source_roi':str(wrong)}))
    status={'status':'completed','output_dir':str(stage1.run),'roi':str(stage1.roi),
            'roi_sha256':sha(stage1.roi),'final_ply':str(stage1.ply)}
    with pytest.raises(ValueError,match='identity differ'):
        cli._completed_manifest(status)


def test_resume_only_stage2_and_preserves_stage1(stage1, monkeypatch):
    before=stage1.ply.read_bytes()
    # Strictly prohibit Stage-1 startup from this resume mode.
    def trap(*args,**kwargs):
        pytest.fail('Stage 1 was called from resume mode')
    svc = types.ModuleType('gsrefine.roi_init.server');svc.main=trap
    monkeypatch.setitem(sys.modules,'gsrefine.roi_init.server',svc)
    observed = fake_stage2(monkeypatch,stage1)
    result=cli.run(args(stage1))
    assert result['status']=='COMPLETED'
    assert str(stage1.mf) in observed[0]
    assert stage1.ply.read_bytes()==before
    assert (stage1.run/'cleanup/stage2_final_v1/gaussian_roi_clean.ply').is_file()


def test_stage2_failure_resume_new_version_and_old_diagnostics_untouched(stage1,monkeypatch):
    calls=fake_stage2(monkeypatch,stage1,succeed=False)
    with pytest.raises(RuntimeError,match='退出码=1'):
        cli.run(args(stage1))
    first = stage1.run/'cleanup/stage2_final_v1'
    (first/'stage2_failure_report.json').write_text('original diagnostic')
    assert json.loads((stage1.run/'metadata/full_pipeline_status.json').read_text())['status']=='STAGE2_FAILED'
    calls=fake_stage2(monkeypatch,stage1,succeed=True)
    assert cli.run(args(stage1))['status']=='COMPLETED'
    assert (stage1.run/'cleanup/stage2_final_v2/gaussian_roi_clean.ply').is_file()
    assert (first/'stage2_failure_report.json').read_text()=='original diagnostic'


def test_stage2_forged_report_never_marked_complete(stage1,monkeypatch):
    fake_stage2(monkeypatch,stage1,bad_hash=True)
    with pytest.raises(RuntimeError,match='fingerprints'):
        cli.run(args(stage1))
    state=json.loads((stage1.run/'metadata/full_pipeline_status.json').read_text())
    assert state['status']=='STAGE2_FAILED'


def test_output_dir_traversal_rejected(stage1):
    with pytest.raises(ValueError,match='Invalid --stage2-name'):
        cli._reserve_stage2_dir(stage1.run,'../oops')


def test_no_auto_restore_without_explicit_manifest():
    args=cli._parser().parse_args(['--accept-auto-exterior','--resume-manifest','/fake/manifest.json'])
    assert args.resume_manifest=='/fake/manifest.json' and not args.output_dir


def test_source_transform_contract_canonical_only():
    s="    downstream = {\n        'roi_json': str((out/'metadata/roi.json').resolve()),\n    }\n" \
      "    if not roi.is_file() or not config.is_file():\n        raise RuntimeError('Cannot publish without source ROI JSON and effective config')\n" \
      "    shutil.copy2(roi, meta/'roi.json')\n"
    t=installer.change_layout(s)
    assert "'roi_json': str(roi.resolve(strict=True))" in t
    assert "shutil.copy2(roi, meta/'roi.json')" not in t
    assert 'file_sha256(roi)' in t
    assert installer.change_layout(t)==t


def test_cache_classification_settings_recorded():
    s="        'short_ray': diagnostics,\n"
    result=installer.change_parity(s)
    assert "'nearest_k': int(nearest_k)" in result
    assert "'min_path_agree': int(min_agree)" in result
    assert installer.change_parity(result)==result


def test_history_does_not_announce_stale_jobs():
    s=("        if job is None:\n            job = self._recent()\n"
       "        if job is None:\n            return {'enabled': self.enabled, 'status': 'idle'}\n"
       "        if announce and result['status'] != 'running':\n")
    fixed=installer.change_workflow(s)
    assert 'announce and self.launched_job()' in fixed
    assert "previous.get('status') != 'running'" in fixed
    assert installer.change_workflow(fixed)==fixed


def test_installer_change_requires_exact_anchor():
    with pytest.raises(RuntimeError):
        installer.change_layout("# unrelated repo content")


def test_old_snapshot_original_roi_edit_rejects_resume(stage1):
    snapshot=stage1.run/'metadata/roi.json'
    snapshot.write_bytes(stage1.roi.read_bytes())
    m=json.loads(stage1.mf.read_text())
    m['stage2_inputs']['roi_json']=str(snapshot)
    stage1.mf.write_text(json.dumps(m))
    (stage1.run/'metadata/processing_report.json').write_text(json.dumps({'source_roi':str(stage1.roi)}))
    stage1.roi.write_text('{"tampered": true}')
    with pytest.raises(ValueError,match='original ROI changed'):
        cli._validate_manifest(stage1.mf)


def test_onepass_cache_source_transformation_is_idempotent():
    source = '''def _validated_parity(parity_dir, model, source, preseal_report, original_poisson):
    # Source model fields must match native 3DGS, not an XYZ-only converted PLY.
    return None


def _discover_parity(run_dir, source, preseal, global_poisson, model):
    for report in reports:
                _validated_parity(report.parent,model,source,preseal,global_poisson)
    return None


def run():
            parity_dir = _discover_parity(run_dir, inputs['gaussian_ply'], preseal,
                                          inputs['poisson_mesh'], model)
        outside, over, delta, parity_record, report_file, masks_file = _validated_parity(
            parity_path, model, inputs['gaussian_ply'], preseal, inputs['poisson_mesh'])
'''
    amended = installer.change_onepass(source)
    assert 'classification settings' in amended
    assert 'nearest_k=nearest_k, min_path_agree=min_path_agree' in amended
    assert installer.change_onepass(amended) == amended


def test_previous_orchestrator_test_fixture_migrates_idempotently():
    source = """def successful_stage1(tmp_path):
    return None


def fake_modules(monkeypatch, stage1_status):
    pass


def run_stage2():
        (out / 'gaussian_roi_clean.ply').write_bytes(b'ply\\nCLEAN')
        (out / 'stage2_final_report.json').write_text(json.dumps({
            'status': 'COMPLETED_EXPERIMENTAL_CLEAN_ROI_GAUSSIAN',
            'counts': {'final_retained': 7, 'final_removed': 3}}))
"""
    changed = installer.change_full_pipeline_tests(source)
    compile(changed,'tests/test_full_pipeline.py','exec')
    assert 'roi_sha256' in changed and 'stage1_manifest_sha256' in changed
    assert installer.change_full_pipeline_tests(changed)==changed


def test_install_all_six_files_atomically_and_idempotently(tmp_path):
    root = tmp_path/'repo'
    (root/'.git').mkdir(parents=True)
    samples={
      'gsrefine/roi_init/output_layout.py': """def publish(out, work, roi, config, report):
    downstream = {'roi_json': str((out/'metadata/roi.json').resolve()),}
    if not roi.is_file() or not config.is_file():
        raise RuntimeError('Cannot publish without source ROI JSON and effective config')
    shutil.copy2(roi, meta/'roi.json')
    return downstream
""",
      'gsrefine/roi_init/workflow.py': """class WorkflowManager:
    def status(self, announce=True):
        if job is None:
            job = self._recent()
        if job is None:
            return {'enabled': self.enabled, 'status': 'idle'}
        if announce and result['status'] != 'running':
            pass
""",
      'gsrefine/cleanup/roi_barrier_parity.py': """def run_roi_barrier_parity(nearest_k,min_agree,roi_link_samples,index,diagnostics):
    report = {
        'short_ray': diagnostics,
    }
    return report
""",
      'gsrefine/cleanup/stage2_onepass.py': """def _validated_parity(parity_dir, model, source, preseal_report, original_poisson):
    # Source model fields must match native 3DGS, not an XYZ-only converted PLY.
    return 1

def _discover_parity(run_dir, source, preseal, global_poisson, model):
    for report in reports:
                _validated_parity(report.parent,model,source,preseal,global_poisson)
    return None

def run():
    if True:
            parity_dir = _discover_parity(run_dir, inputs['gaussian_ply'], preseal,
                                          inputs['poisson_mesh'], model)
    if True:
        outside, over, delta, parity_record, report_file, masks_file = _validated_parity(
            parity_path, model, inputs['gaussian_ply'], preseal, inputs['poisson_mesh'])
""",
      'gsrefine/roi_init/full_pipeline.py': """def main():
    pass
""",
      'tests/test_full_pipeline.py': """def successful_stage1(tmp_path):
    return None


def fake_modules(monkeypatch, stage1_status):
    pass


def run_stage2():
        (out / 'gaussian_roi_clean.ply').write_bytes(b'ply\\nCLEAN')
        (out / 'stage2_final_report.json').write_text(json.dumps({
            'status': 'COMPLETED_EXPERIMENTAL_CLEAN_ROI_GAUSSIAN',
            'counts': {'final_retained': 7, 'final_removed': 3}}))
"""
    }
    for rel,content in samples.items():
        p=root/rel
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(content)
    old={name:(root/name).read_bytes() for name in samples}
    assert len(installer.apply(check=True,root=root))==6
    assert all((root/name).read_bytes()==old[name] for name in old)
    assert len(installer.apply(check=False,root=root))==6
    assert installer.apply(check=False,root=root)==[]
    assert len(list((root/'.git/gsrefine_patch_backups').glob('unification_*'))) == 1
    assert 'roi.resolve(strict=True)' in (root/'gsrefine/roi_init/output_layout.py').read_text()
    assert '--resume-manifest' in (root/'gsrefine/roi_init/full_pipeline.py').read_text()
