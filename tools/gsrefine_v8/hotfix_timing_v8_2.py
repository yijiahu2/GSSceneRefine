#!/usr/bin/env python3
"""GSSceneRefine v8.2: surgical fix for invalid Stage-2 audit timer.

This updates ONLY gsrefine/cleanup/local_region_pipeline.py.
The v8.1 local-import repair is required. No geometry, threshold, label,
auxiliary-protection or Gaussian export logic is changed.
"""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import shutil
import tempfile

REL = Path('gsrefine/cleanup/local_region_pipeline.py')
OLD_IMPORT = 'from gsrefine.cleanup.stage2_runtime import file_digest, measured, snapshot'
NEW_IMPORT = 'from gsrefine.cleanup.stage2_runtime import file_digest, measured, snapshot, timed'
OLD_TIMER = "with measured('engineering_cross_stage_and_risk_audit'):"
NEW_TIMER = "with timed('engineering_cross_stage_and_risk_audit'):"


def inspect(source):
    tree = ast.parse(source, filename=str(REL))
    runs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run']
    if len(runs) != 1:
        raise RuntimeError('Expected exactly one run() in the Stage-2 local region module')
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom)
               and n.module == 'gsrefine.cleanup.stage2_runtime']
    if not imports:
        raise RuntimeError('Stage-2 timing import is missing')
    imports_aliases = {x.name for node in imports for x in node.names}
    bad_scopes = [n for n in ast.walk(runs[0]) if isinstance(n, ast.ImportFrom)
                  and n.module == 'gsrefine.cleanup.stage2_runtime'
                  and any(a.name in ('measured', 'timed') for a in n.names)]
    if bad_scopes:
        raise RuntimeError('Another function-local timing import shadows the module import')
    incorrectly_used = [n for n in ast.walk(runs[0])
                       if isinstance(n, (ast.With, ast.AsyncWith))
                       for item in n.items
                       if isinstance(item.context_expr, ast.Call)
                       and isinstance(item.context_expr.func, ast.Name)
                       and item.context_expr.func.id == 'measured']
    return imports_aliases, len(incorrectly_used)


def prepare(path):
    original = path.read_text(encoding='utf-8')
    if original.count(NEW_TIMER) == 1 and original.count(NEW_IMPORT) == 1:
        names, invalid = inspect(original)
        if invalid or not {'measured','timed'}.issubset(names):
            raise RuntimeError('Unexpected pre-existing v8.2 source; refuse to proceed')
        return None
    if original.count(OLD_TIMER) != 1 or original.count(OLD_IMPORT) != 1:
        raise RuntimeError('Code does not match v8.1 baseline; no changes made')
    names, invalid = inspect(original)
    if invalid != 1 or 'measured' not in names:
        raise RuntimeError('Unexpected additional invalid timing contexts; no changes made')
    if "measured('gaussian_radius_statistics')" not in original:
        raise RuntimeError('Expected v8 Stage-2 instrumentation is absent')
    fixed = original.replace(OLD_IMPORT, NEW_IMPORT, 1).replace(OLD_TIMER, NEW_TIMER, 1)
    fixed_names, fixed_invalid = inspect(fixed)
    if fixed_invalid or not {'measured','timed'}.issubset(fixed_names):
        raise RuntimeError('Fixed timing import/context failed AST validation')
    return fixed


def install(root, *, check=False):
    path = root / REL
    if not path.is_file():
        raise FileNotFoundError(path)
    fixed = prepare(path)
    if fixed is None:
        print('[v8.2] ALREADY FIXED; verified AST and timing contexts')
        return
    if check:
        print('[v8.2] CHECK PASSED: one invalid audit timer; replacement is syntax-validated')
        return
    # A stable filename facilitates recovery; never overwrite prior backups.
    backup = root / 'tools/gsrefine_v8/local_region_pipeline_before_v8_2.py.bak'
    if backup.exists():
        raise FileExistsError('Backup already exists, refusing to overwrite: ' + str(backup))
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup)
    fd, staged = tempfile.mkstemp(prefix='.local_region_v8_2_', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(fixed)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(staged, path.stat().st_mode)
        os.replace(staged, path)
    finally:
        Path(staged).unlink(missing_ok=True)
    if prepare(path) is not None:
        raise RuntimeError('Post-install validation unexpectedly failed')
    print('[v8.2] FIXED:', path)
    print('[v8.2] BACKUP:', backup)
    print('[v8.2] VERIFIED: module-level measured decorator and timed context; scientific rules unchanged')


def preflight_manifest(root, manifest):
    """Read-only smoke test using a REAL published Stage-1 mask provenance."""
    import json
    import importlib.util
    import numpy as np

    path = root / REL
    if prepare(path) is not None:
        raise RuntimeError('Apply v8.2 fix before running the audit preflight')
    mf = Path(manifest).expanduser().resolve(strict=True)
    if mf.name != 'manifest.json' or mf.parent.name != 'metadata':
        raise ValueError('Expected the completed Stage-1 metadata/manifest.json')
    record = json.loads(mf.read_text(encoding='utf-8'))
    if record.get('status') != 'completed':
        raise ValueError('Stage-1 manifest not marked completed')
    roi_run = mf.parent.parent
    model_path = Path(record['stage2_inputs']['gaussian_ply']).resolve(strict=True)
    if model_path != (roi_run/'final/gaussian_roi.ply').resolve():
        raise ValueError('Stage-1 manifest does not point to the canonical R3 Gaussian PLY')
    stage1_masks = roi_run/'metadata/stage1_masks.npz'
    if not stage1_masks.exists():
        raise ValueError('This Stage-1 run predates v8 permanent masks; cannot preflight attribution')
    with np.load(stage1_masks, allow_pickle=False) as masks:
        n = int(masks['R3'].sum())
    if n <= 0:
        raise ValueError('Stage-1 masks are empty')
    audit_file = root/'gsrefine/cleanup/engineering_audit.py'
    spec = importlib.util.spec_from_file_location('_gsrefine_v8_2_audit_preflight', audit_file)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot locate the installed v8 engineering audit module')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.cross_stage(roi_run, model_path, np.zeros(n, dtype=np.bool_), input_count=n)
    if result.get('status') != 'AUDITED_COUNTS_NOT_GROUND_TRUTH':
        raise RuntimeError('Stage-1 cross-stage audit unexpectedly unavailable or failed')
    sample = module.classification_risk(
        points=np.asarray([[0.,0.,0.],[1.,1.,1.]]),
        remove=np.asarray([True,False]), outside=np.asarray([True,False]),
        withheld=np.asarray([False,False]), protected=np.asarray([False,False]),
        delta=1., scope={'anchor': np.asarray([0.,0.,0.])})
    if not sample.get('deletion_invariant_passed'):
        raise RuntimeError('Classification risk audit failed synthetic invariant test')
    print('[v8.2] REAL STAGE-1 MASK AUDIT PASSED:', roi_run)
    print('[v8.2] R3 Gaussian rows:', n)
    print('[v8.2] Cross-stage attribution and classification-risk audit smoke tests passed')
    print('[v8.2] This is NOT a successful Stage-2 classification / accuracy test')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', default='/home/xth/GSSceneRefine')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--preflight-manifest', metavar='METADATA_MANIFEST_JSON')
    args = p.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    if args.preflight_manifest:
        preflight_manifest(root, args.preflight_manifest)
    else:
        install(root, check=args.check)


if __name__ == '__main__':
    main()
