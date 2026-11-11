#!/usr/bin/env python3
"""Unify Stage-2 on direct clipping of Stage-1 R3 Poisson (no legacy local fit).

Verify package, make two small in-place entrypoint edits, then remove ONLY an
explicit list of obsolete code/tests/installers. Never touches experimental
PLY/NPZ outputs, Stage-1 Gaussian, SfM, Poisson reference, or Conda setup.

Run with --check first, then --apply. Idempotent when already installed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
import tempfile
from pathlib import Path

REQUIRED = (
    'gsrefine/cleanup/sdf_common.py',
    'gsrefine/cleanup/direct_r3_pipeline.py',
    'gsrefine/geometry/mesh_clip.py',
    'gsrefine/geometry/r3_direct_poisson_clip.py',
    'gsrefine/geometry/r3_sfm_bridge.py',
)
# Source files that implement the retired local-SfM Poisson / per-face trusted mask.
# All old experiment results on disk remain available for manuscript ablation.
REMOVE_CODE = (
    'gsrefine/cleanup/direct_sdf.py',
    'gsrefine/geometry/roi_poisson.py',
    'gsrefine/geometry/trusted_roi.py',
    'gsrefine/geometry/roi_surface.py',
    'gsrefine/geometry/roi_sfm_local.py',
    'gsrefine/geometry/stage1_handoff.py',
    'tools/apply_stage2_trusted_surface.py',
    'tools/apply_stage2_direct_r3.py',
    'tools/apply_stage2_handoff.py',
    'tools/apply_exact_sfm_handoff.py',
    'tools/apply_r3_sfm_handoff.py',
    'tests/test_stage2_cleanup.py',
    'tests/test_stage2_roi_adaptive_clip.py',
    'tests/test_stage2_exact_local_safety.py',
    'tests/test_stage2_stage1_writer.py',
    'tests/test_stage2_trusted_surface.py',
    'tests/test_r3_sfm_install_contract.py',
    'tests/test_stage2_direct_r3_installer.py',
)
OLD_MAP = "'stage2-cleanup':'gsrefine.cleanup.direct_sdf'"
NEW_MAP = "'stage2-cleanup':'gsrefine.cleanup.direct_r3_pipeline'"
EXTRA_MAP = "    'stage2-r3':'gsrefine.cleanup.direct_r3_pipeline',\n"
OLD_HANDOFF = '''    # Stage 2 may reuse a certified surface mask already made in Stage 1.
    # This is a no-op if Stage 1 did not produce the optional handoff.
    from gsrefine.geometry.stage1_handoff import retain_stage1_surface_faces
    _stage2_faces = retain_stage1_surface_faces(work, meta, paths['poisson_raw'], roi)
    if _stage2_faces is not None:
        downstream['roi_surface_faces'] = str(_stage2_faces)
'''
NEW_HANDOFF = '''    # Canonical Stage-2 uses the verified R3 spatial field and the original
    # Module-1 Poisson; no legacy triangle-face-mask handoff is generated.
'''

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _check_required(root):
    missing = [x for x in REQUIRED if not (root/x).is_file()]
    if missing:
        raise FileNotFoundError('Incomplete direct-R3 package: '+', '.join(missing))
    for x in REQUIRED:
        ast.parse((root/x).read_text(encoding='utf8'), filename=x)
    for x in REQUIRED:
        src=(root/x).read_text(encoding='utf8')
        for token in ('from gsrefine.cleanup.direct_sdf import',
                      'from gsrefine.geometry.roi_surface import',
                      'from gsrefine.geometry.roi_poisson import',
                      'from gsrefine.geometry.trusted_roi import'):
            if token in src:
                raise RuntimeError(f'Retired implementation dependency in {x}: {token}')

def _changed_sources(root):
    changes = {}
    cli = root/'run.py'
    src = cli.read_text(encoding='utf8')
    if src.count(OLD_MAP) == 1:
        new = src.replace(OLD_MAP, NEW_MAP)
    elif src.count(NEW_MAP) == 1:
        new = src
    else:
        raise RuntimeError('Unrecognized run.py Stage-2 command mapping; refusing to overwrite')
    new = new.replace(EXTRA_MAP, '')
    if "'stage2-sfm':'gsrefine.geometry.r3_sfm_bridge'" not in new:
        raise RuntimeError('Required V3 stage2-sfm entry missing in run.py')
    if new != src:
        ast.parse(new, filename='run.py')
        changes[cli] = new.encode('utf8')
    pub = root/'gsrefine/roi_init/output_layout.py'
    src = pub.read_text(encoding='utf8')
    if src.count(OLD_HANDOFF) == 1:
        new = src.replace(OLD_HANDOFF, NEW_HANDOFF)
    elif src.count(NEW_HANDOFF) == 1:
        new = src
    else:
        raise RuntimeError('Unrecognized optional legacy Stage-1 face-mask hook; refusing patch')
    if new != src:
        ast.parse(new, filename=str(pub))
        changes[pub] = new.encode('utf8')
    return changes

def apply(root, *, execute=False):
    root = Path(root).expanduser().resolve(strict=True)
    _check_required(root)
    changes = _changed_sources(root)
    obsolete = [root/x for x in REMOVE_CODE if (root/x).is_file()]
    if not execute:
        print('[CHECK] Stage-2 canonical direct-R3 package valid')
        print('[CHECK] Source edits:',*[str(x.relative_to(root)) for x in changes],sep='\n  - ')
        print('[CHECK] Retired script removal:',*[str(x.relative_to(root)) for x in obsolete],sep='\n  - ')
        print('[CHECK] Experiment data outputs are untouched. Use --apply to commit.')
        return {'changes':len(changes),'retired':len(obsolete)}
    previous = {p:p.read_bytes() for p in [*changes,*obsolete]}
    replaced = []
    removed = []
    try:
        for path, data in changes.items():
            fd, name = tempfile.mkstemp(prefix='.gsrefine-stage2-',dir=path.parent)
            try:
                with os.fdopen(fd,'wb') as out:
                    out.write(data)
                    out.flush()
                    os.fsync(out.fileno())
                os.chmod(name, path.stat().st_mode)
                if sha(path) != hashlib.sha256(previous[path]).hexdigest():
                    raise RuntimeError('File changed during Stage-2 installation: '+str(path))
                os.replace(name,path)
                replaced.append(path)
            finally:
                Path(name).unlink(missing_ok=True)
        for path in obsolete:
            if sha(path) != hashlib.sha256(previous[path]).hexdigest():
                raise RuntimeError('Legacy script changed during install: '+str(path))
            path.unlink()
            removed.append(path)
    except Exception:
        for path in reversed([*replaced, *removed]):
            path.write_bytes(previous[path])
        raise
    print('[SUCCESS] Canonical stage2-cleanup -> Direct R3; obsolete code removed:', len(removed))
    print('[SUCCESS] Stage-1 R3, V3 SfM, Module-1 Poisson and all experiment files remain unchanged')
    return {'changes':len(replaced),'retired':len(removed)}

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', default='/home/xth/GSSceneRefine')
    group=ap.add_mutually_exclusive_group()
    group.add_argument('--check',action='store_true')
    group.add_argument('--apply',action='store_true')
    a=ap.parse_args(argv)
    apply(a.project_root,execute=a.apply)

if __name__=='__main__':
    main()
