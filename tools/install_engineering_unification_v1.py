#!/usr/bin/env python3
"""Guarded in-place migration from the synchronized 2026-11-14 GSSceneRefine main.

Only orchestrator, output provenance, status logging, and classification-cache
metadata are changed. No R3/SDF/Poisson/geometry math is modified.

  python tools/install_engineering_unification_v1.py --check
  python tools/install_engineering_unification_v1.py
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = Path(__file__).resolve().parent / 'payload' / 'full_pipeline.py'

CHANGED = (
    'gsrefine/roi_init/output_layout.py',
    'gsrefine/roi_init/workflow.py',
    'gsrefine/cleanup/roi_barrier_parity.py',
    'gsrefine/cleanup/stage2_onepass.py',
)
ENTRY = 'gsrefine/roi_init/full_pipeline.py'
MARKER = '# gsrefine_engineering_manifest_resume_v1_20261114'


def single(s, old, new, desc):
    if old == new or new in s:
        return s
    if old not in s:
        raise RuntimeError(f'{desc}: source does not match expected synchronized GitHub version; NO change made')
    if s.count(old) != 1:
        raise RuntimeError(f'{desc}: source match is ambiguous ({s.count(old)} occurrences); NO change made')
    return s.replace(old, new, 1)


def change_layout(s):
    s = single(s,
        "'roi_json': str((out/'metadata/roi.json').resolve()),",
        "'roi_json': str(roi.resolve(strict=True)),",
        'declare canonical ROI directly in Stage-1 manifest')
    s = single(s,
        "    shutil.copy2(roi, meta/'roi.json')\n",
        "    # One canonical immutable scene/roi/roi_NNN.json; never copy it into metadata.\n",
        'remove new ROI duplicate from Stage-1 publication')
    # The original Stage-1 core already verified roi hash. Before publication,
    # verify again to detect any edit after the in-memory R3 selection.
    s = single(s,
        "    if not roi.is_file() or not config.is_file():\n        raise RuntimeError('Cannot publish without source ROI JSON and effective config')\n",
        "    if not roi.is_file() or not config.is_file():\n        raise RuntimeError('Cannot publish without source ROI JSON and effective config')\n"
        "    from gsrefine.roi.core import file_sha256\n"
        "    if file_sha256(roi) != report['input_sha256']['roi']:\n"
        "        raise ValueError('Confirmed canonical ROI changed during Stage-1 processing')\n",
        'detect ROI file modified while Stage-1 R3 was running')
    return s


def change_workflow(s):
    # A fresh Edge session must not report a historical completed ROI as current.
    # Preserve restoration of a genuinely still-running separate roi-auto worker.
    s = single(s,
        "        if job is None:\n            job = self._recent()\n"
        "        if job is None:\n            return {'enabled': self.enabled, 'status': 'idle'}\n",
        "        if job is None:\n            job = self._recent()\n"
        "            if job is not None:\n"
        "                previous = _read_progress(job['output_dir'])\n"
        "                if (previous.get('status') != 'running' or\n"
        "                        not _pid_alive(previous.get('pid'))):\n"
        "                    return {'enabled': self.enabled, 'status': 'idle'}\n"
        "        if job is None:\n            return {'enabled': self.enabled, 'status': 'idle'}\n",
        'isolate old completed ROI from fresh Edge session')
    s = single(s,
        "        if announce and result['status'] != 'running':\n",
        "        # A historic recovered run must not announce itself as THIS session's job.\n"
        "        if announce and self.launched_job() and result['status'] != 'running':\n",
        'prevent historical Stage-1 completed log before current ROI confirmation')
    return s


def change_parity(s):
    s = single(s,
        "        'short_ray': diagnostics,\n",
        "        'classification_config': {'nearest_k': int(nearest_k),\n"
        "                                  'min_path_agree': int(min_agree),\n"
        "                                  'roi_link_samples': int(roi_link_samples),\n"
        "                                  'seed_mode': str(index['summary'].get('seed_mode', 'unknown'))},\n"
        "        'short_ray': diagnostics,\n",
        'record classification inputs for trustworthy cache reuse')
    return s


def change_onepass(s):
    s = single(s,
        "def _validated_parity(parity_dir, model, source, preseal_report, original_poisson):",
        "def _validated_parity(parity_dir, model, source, preseal_report, original_poisson, *,\n"
        "                      nearest_k=None, min_path_agree=None):",
        'extend cache validation signature without changing old calls')
    s = single(s,
        "    # Source model fields must match native 3DGS, not an XYZ-only converted PLY.\n",
        "    # Classification cache reuse is safe only for matching numeric query parameters.\n"
        "    if nearest_k is not None or min_path_agree is not None:\n"
        "        config = report.get('classification_config') or {}\n"
        "        if (config.get('nearest_k') != int(nearest_k) or\n"
        "                config.get('min_path_agree') != int(min_path_agree) or\n"
        "                config.get('roi_link_samples') != 3):\n"
        "            raise ValueError('Parity cache uses different/unknown classification settings; recompute')\n"
        "    # Source model fields must match native 3DGS, not an XYZ-only converted PLY.\n",
        'enforce cached classification settings')
    s = single(s,
        'def _discover_parity(run_dir, source, preseal, global_poisson, model):',
        'def _discover_parity(run_dir, source, preseal, global_poisson, model, *, nearest_k=8, min_path_agree=2):',
        'pass classification settings into automatic cache search')
    s = single(s,
        "                _validated_parity(report.parent,model,source,preseal,global_poisson)\n",
        "                _validated_parity(report.parent,model,source,preseal,global_poisson,\n"
        "                                  nearest_k=nearest_k,min_path_agree=min_path_agree)\n"
        "                if data.get('classification_config',{}).get('seed_mode') != (\n"
        "                        'single highest R3 shell node (researcher must verify exterior)'):\n"
        "                    continue\n",
        'prevent auto seed from reusing a manual exterior classification')
    s = single(s,
        "            parity_dir = _discover_parity(run_dir, inputs['gaussian_ply'], preseal,\n"
        "                                          inputs['poisson_mesh'], model)\n",
        "            parity_dir = _discover_parity(run_dir, inputs['gaussian_ply'], preseal,\n"
        "                                          inputs['poisson_mesh'], model,\n"
        "                                          nearest_k=nearest_k, min_path_agree=min_path_agree)\n",
        'route requested params into auto classification reuse')
    s = single(s,
        "        outside, over, delta, parity_record, report_file, masks_file = _validated_parity(\n"
        "            parity_path, model, inputs['gaussian_ply'], preseal, inputs['poisson_mesh'])\n",
        "        outside, over, delta, parity_record, report_file, masks_file = _validated_parity(\n"
        "            parity_path, model, inputs['gaussian_ply'], preseal, inputs['poisson_mesh'],\n"
        "            nearest_k=nearest_k, min_path_agree=min_path_agree)\n",
        'enforce classification settings in final Stage-2 export')
    return s


def change_full_pipeline_tests(s):
    """Modernize synthetic fixtures to the verified Stage-1 manifest contract."""
    start = "def successful_stage1(tmp_path):\n"
    stop = "\n\ndef fake_modules(monkeypatch, stage1_status):\n"
    if start in s:
        if s.count(start) != 1 or s.count(stop) != 1:
            raise RuntimeError('tests/test_full_pipeline.py: ambiguous fixture')
        a=s.index(start); b=s.index(stop,a)
        fixture = """def successful_stage1(tmp_path):
    from gsrefine.roi.core import file_sha256
    roi_run = tmp_path / 'results' / 'biechuanchansi' / 'methods' / 'Mip-Splatting' / 'roi_002'
    (roi_run / 'final').mkdir(parents=True)
    (roi_run / 'metadata').mkdir()
    roi_path = tmp_path / 'results' / 'biechuanchansi' / 'roi' / 'roi_002.json'
    roi_path.parent.mkdir(parents=True)
    roi_path.write_text('{}')
    ply = roi_run / 'final' / 'gaussian_roi.ply'
    ply.write_bytes(b'ply\\nDATA')
    poisson = tmp_path/'results'/'biechuanchansi'/'geometry'/'poisson_surface.ply'
    poisson.parent.mkdir(parents=True,exist_ok=True)
    poisson.write_bytes(b'poisson')
    sfm = poisson.parent / 'cleaned_sfm.npz'
    sfm.write_bytes(b'sfm')
    (roi_run / 'metadata' / 'manifest.json').write_text(json.dumps({
        'schema': 'GSSceneRefine.output.v3',
        'status': 'completed',
        'result_status': 'ROI_RESTORED_NOT_ANOMALY_CLEANED',
        'input_sha256': {'roi': file_sha256(roi_path),
                         'poisson': file_sha256(poisson), 'sfm': file_sha256(sfm)},
        'output_sha256': file_sha256(ply),
        'stage2_inputs': {'roi_json': str(roi_path), 'gaussian_ply': str(ply),
                          'poisson_mesh': str(poisson), 'clean_sfm_npz': str(sfm)}}))
    return {'status': 'completed', 'output_dir': str(roi_run),
            'roi': str(roi_path), 'roi_sha256': file_sha256(roi_path),
            'final_ply': str(ply)}, roi_run
"""
        s = s[:a]+fixture+s[b:]
    elif "'roi_sha256': file_sha256(roi_path)" not in s:
        raise RuntimeError('tests/test_full_pipeline.py fixture source does not match')
    previous = """        (out / 'gaussian_roi_clean.ply').write_bytes(b'ply\\nCLEAN')
        (out / 'stage2_final_report.json').write_text(json.dumps({
            'status': 'COMPLETED_EXPERIMENTAL_CLEAN_ROI_GAUSSIAN',
            'counts': {'final_retained': 7, 'final_removed': 3}}))
"""
    current = """        from gsrefine.roi.core import file_sha256
        clean=out/'gaussian_roi_clean.ply'
        clean.write_bytes(b'ply\\nCLEAN')
        manifest=roi_run/'metadata'/'manifest.json'
        source=roi_run/'final'/'gaussian_roi.ply'
        (out / 'stage2_final_report.json').write_text(json.dumps({
            'status': 'COMPLETED_EXPERIMENTAL_CLEAN_ROI_GAUSSIAN',
            'input': {'stage1_manifest':str(manifest),
                      'stage1_manifest_sha256':file_sha256(manifest),
                      'original_roi_gaussian':str(source),
                      'original_roi_gaussian_sha256':file_sha256(source)},
            'outputs':{'final_supersplat_gaussian_ply':str(clean),
                       'final_ply_sha256':file_sha256(clean)},
            'counts': {'final_retained': 7, 'final_removed': 3}}))
"""
    return single(s, previous, current, 'update full_pipeline test fake Stage-2 report')

CHANGES = {
    'gsrefine/roi_init/output_layout.py': change_layout,
    'gsrefine/roi_init/workflow.py': change_workflow,
    'gsrefine/cleanup/roi_barrier_parity.py': change_parity,
    'gsrefine/cleanup/stage2_onepass.py': change_onepass,
    'tests/test_full_pipeline.py': change_full_pipeline_tests,
}


def stage(root=ROOT):
    """Build all transformed files in memory before ANY target is touched."""
    staged = {}
    for rel, transform in CHANGES.items():
        path = root / rel
        original = path.read_text(encoding='utf-8')
        new = transform(original)
        if MARKER not in new:
            new = new.rstrip() + '\n\n' + MARKER + '\n'
        compile(new, str(path), 'exec')
        staged[rel] = (original, new)
    original = (root / ENTRY).read_text(encoding='utf-8')
    payload_path = PAYLOAD
    if not payload_path.is_file():
        raise FileNotFoundError('Missing full_pipeline.py payload: ' + str(payload_path))
    new = payload_path.read_text(encoding='utf-8')
    compile(new, ENTRY, 'exec')
    staged[ENTRY] = (original, new)
    return staged


def apply(*, check=False, root=ROOT):
    staged = stage(root)
    changed = {rel: (old, new) for rel, (old, new) in staged.items() if old != new}
    if not changed:
        print('[unification] Already installed; no changes needed')
        return []
    print('[unification] Preflight OK: ' + ', '.join(changed))
    if check:
        return list(changed)
    backup = root / '.git' / 'gsrefine_patch_backups' / ('unification_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    # Fail safely if this is not a real repository; never write unverified files.
    if not (root / '.git').exists():
        raise ValueError('Project root has no .git; cannot create controlled backups')
    if backup.exists():
        raise FileExistsError('Backup directory exists: ' + str(backup))
    for rel, (old, _) in changed.items():
        dest = backup / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(old, encoding='utf-8')
    temps = {}
    completed = []
    try:
        for rel, (_, new) in changed.items():
            target = root / rel
            fd, name = tempfile.mkstemp(prefix='.'+target.name+'.',suffix='.tmp',dir=target.parent)
            with os.fdopen(fd,'w',encoding='utf-8') as out:
                out.write(new)
                out.flush()
                os.fsync(out.fileno())
            os.chmod(name, target.stat().st_mode)
            temps[rel] = name
        for rel in changed:
            os.replace(temps[rel], root / rel)
            completed.append(rel)
    except BaseException:
        for rel in completed:
            shutil.copy2(backup / rel, root / rel)
        raise
    finally:
        for name in temps.values():
            Path(name).unlink(missing_ok=True)
    print('[unification] Installed; original files backed up at: ' + str(backup))
    return list(changed)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true',help='Preflight only; write nothing')
    args = p.parse_args(argv)
    apply(check=args.check)


if __name__ == '__main__':
    main()
