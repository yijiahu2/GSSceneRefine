#!/usr/bin/env python3
"""Repair three outdated Stage-2 regression tests without patch/context matching.

Only rewrites three named test functions. Does not touch GSSceneRefine core
algorithms, R3 inputs, Poisson, data, caches or unrelated tests.
"""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
FILES = {
    'tests/test_barrier_separation_gate_v2.py': {
        ('test_auto_seed_requires_modal_top_component',
         'test_auto_seed_is_single_documented_exterior_hypothesis'):
'''def test_auto_seed_is_single_documented_exterior_hypothesis(tmp_path):
    """Highest-shell exterior seed is a hypothesis, not a top-bottom gate."""
    from gsrefine.cleanup.roi_barrier_parity import _make_navigation
    shape=(4,4,6); h=1.; origin=np.array([-2.,-2.,-3.])
    passable=np.ones(shape,bool)
    comp=np.zeros(shape,np.int32); comp[:,:,3:]=1
    np.savez(tmp_path/'grid.npz', origin=origin, voxel=h,
             shape=np.asarray(shape), passable=passable,
             final_r3_inside=passable, components=comp)
    nav=_make_navigation(tmp_path/'grid.npz',lambda p:np.ones(len(p)),
                         np.array([0.,0.,1.]),Plane(),O3D,np.array([[0.,0.,1.]]),
                         neighbour_rays_batch=256)
    summary=nav['summary']
    assert summary['seed_mode']=='single highest R3 shell node (researcher must verify exterior)'
    assert summary['seed_count']==1
    assert np.isclose(summary['seed_info']['chosen_seed_xyz'][2],2.5)
    assert summary['odd_cycle_conflicts']==0
''',
    },
    'tests/test_stage2_onepass.py': {
        ('test_reuses_source_matched_barrier_audit',):
'''def test_reuses_source_matched_barrier_audit(tmp_path):
    """Reuse source-identical region graphs, never old top-bottom conclusions."""
    root=tmp_path/'roi_001'
    root.mkdir()
    manifest=root/'manifest.json'; manifest.write_bytes(b'one')
    handoff=root/'handoff.npz'; handoff.write_bytes(b'two')
    poisson=root/'poisson.ply'; poisson.write_bytes(b'three')
    folder=root/'cleanup/diagnostics/barrier'
    folder.mkdir(parents=True)
    for name in ('coarse_connectivity.npz','fine_connectivity.npz','global_poisson_final_r3_intersection.npz'):
        (folder/name).write_bytes(b'x')
    report={'status':'FLOOD_SAMPLING_COMPLETED_NOT_CERTIFIED_SEALED',
            'inputs':{'manifest_sha256':mod._sha(manifest),
                      'r3_sha256':mod._sha(handoff),
                      'original_poisson_sha256':mod._sha(poisson)},
            'flood':{'coarse':{'status':'REGION_GRAPH_SAMPLED_NOT_CERTIFIED',
                                  'grid':{'voxel':.3,'roi_touching_audit_window_samples':0}},
                     'fine':{'status':'REGION_GRAPH_SAMPLED_NOT_CERTIFIED',
                                 'grid':{'voxel':.2,'roi_touching_audit_window_samples':0}}}}
    report_path=folder/'roi_global_poisson_barrier_report.json'
    report_path.write_text(json.dumps(report))
    assert mod._find_barrier_audit(root,manifest,handoff,poisson)==folder
    # Older sampled graph status is not a rejection when source, grid and
    # post-cache separation verification remain valid.
    report['flood']['coarse']['status']='TOP_TO_BOTTOM_PATH_FOUND_SAMPLED_GRAPH'
    report_path.write_text(json.dumps(report))
    assert mod._find_barrier_audit(root,manifest,handoff,poisson)==folder
    # Actual truncated sampling window still invalidates cache reuse.
    report['flood']['coarse']['grid']['roi_touching_audit_window_samples']=1
    report_path.write_text(json.dumps(report))
    assert mod._find_barrier_audit(root,manifest,handoff,poisson) is None
    report['flood']['coarse']['grid']['roi_touching_audit_window_samples']=0
    report_path.write_text(json.dumps(report))
    poisson.write_bytes(b'changed')
    assert mod._find_barrier_audit(root,manifest,handoff,poisson) is None
''',
        ('test_geometry_preparation_reuses_validated_existing_records',):
'''def test_geometry_preparation_reuses_validated_existing_records(monkeypatch,tmp_path):
    """Reject old clip-before-main preseal; accept new original-main-first cache."""
    from gsrefine.cleanup import sdf_common
    run=tmp_path/'ROI';run.mkdir()
    manifest=run/'metadata/manifest.json';manifest.parent.mkdir()
    manifest.write_text('{}')
    preseal=run/'cleanup/geometry/direct_r3_sdf/a/preseal_diagnostics.json'
    preseal.parent.mkdir(parents=True)
    preseal.write_text(json.dumps({'stage1_manifest':str(manifest)}))
    monkeypatch.setattr(sdf_common,'_load_stage1_manifest',
                        lambda path:(manifest,{}, {},run))
    import types,sys
    fake=types.ModuleType('gsrefine.cleanup.stage2_geometry_validation')
    fake.validate_parity_geometry=lambda *a,**kw:{'summary':{'status':'VALIDATED_SEALLESS_DISTANCE_GEOMETRY'}}
    monkeypatch.setitem(sys.modules,'gsrefine.cleanup.stage2_geometry_validation',fake)
    with pytest.raises(ValueError,match='Old preseal'):
        mod._prepare_required_geometry(manifest,run,preseal_report=preseal)
    clip=run/'cleanup/geometry/direct_r3_clip/new/clip_comparison_report.json'
    clip.parent.mkdir(parents=True)
    clip.write_text(json.dumps({
        'source_component_selection':
        'largest_original_edge_connected_component_BEFORE_R3_clip'}))
    preseal.write_text(json.dumps({
        'stage1_manifest':str(manifest),
        'direct_r3_clip_report':str(clip)}))
    selected,rec=mod._prepare_required_geometry(manifest,run,preseal_report=preseal)
    assert selected==preseal
    assert rec['status']=='VALIDATED_SEALLESS_DISTANCE_GEOMETRY_REUSED'
''',
    },
}


def replace_functions(source: str, spec: dict, filename: str) -> str:
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source, filename=filename)
    functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    replacements = []
    for aliases, new_source in spec.items():
        found = [n for n in functions if n.name in aliases]
        if len(found) != 1:
            raise RuntimeError(f'{filename}: expected exactly one of {aliases}; found {len(found)}. No files changed.')
        node = found[0]
        if node.decorator_list:
            raise RuntimeError(f'{filename}:{node.name} has decorators; refusing unknown structure')
        ast.parse(new_source, filename=f'{filename}:{node.name}')
        replacements.append((node.lineno - 1, node.end_lineno, new_source.rstrip() + '\n'))
    # Process backwards so line indices of earlier functions remain stable.
    for lo, hi, new_source in sorted(replacements, reverse=True):
        lines[lo:hi] = [new_source]
    updated=''.join(lines)
    ast.parse(updated, filename=filename)
    return updated


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate only; write nothing')
    args=parser.parse_args()
    updated={}
    for rel, spec in FILES.items():
        path=REPO/rel
        if not path.is_file():
            raise FileNotFoundError(f'Missing test file {path}; no files changed')
        current=path.read_text(encoding='utf-8')
        candidate=replace_functions(current,spec,rel)
        if candidate!=current:
            updated[path]=candidate
        print(f'[verified] {rel}: {len(spec)} named test functions')
    if args.check:
        print(f'Validation OK: would update {len(updated)} test files; core code untouched.')
        return
    if not updated:
        print('Already aligned. No files changed.')
        return
    backup=Path(os.environ.get('GSREFINE_TEST_BACKUP_DIR',str(Path.home()/'GSSceneRefine_test_backup_20261112')))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup=backup/stamp
    for path in updated:
        dest=backup/path.relative_to(REPO)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
    for path,data in updated.items():
        fd, tmp=tempfile.mkstemp(dir=path.parent,prefix='.stage2-testfix-',suffix='.tmp')
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as f:
                f.write(data)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
        print('[updated]',path.relative_to(REPO))
    print('Backups:',backup)
    print('Updated tests only. No core source or data was changed.')


if __name__=='__main__':
    main()
