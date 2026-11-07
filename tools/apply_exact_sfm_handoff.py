#!/usr/bin/env python3
"""Apply the exact SfM publication hook to a known GSSceneRefine Stage-1 publisher.

Accepts either original output_layout.py (with existing ROI surface hook allowed),
the previous approximate ROI-SfM hook, or already-exact hook.
Does not touch Stage-1 R3 math or invent an R3 local safety-distance field.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from pathlib import Path
import shutil

ANCHOR = "    shutil.copy2(config, meta/'effective_scene.yaml')\n"
OLD_START = '    # Stage 2 input: retain core SfM and one additional original safety band.\n'
OLD_END = "    downstream['sfm_roi_npz'] = str(_roi_sfm_path)\n"
HOOK = (
    '    # Stage 2: exact Stage-1 R3 per-location d(x); no scalar fallback.\n'
    '    from gsrefine.geometry.roi_sfm_local import publish_stage1_exact_roi_sfm\n'
    '    _roi_sfm_path, _roi_sfm_report = publish_stage1_exact_roi_sfm(\n'
    '        downstream=downstream, metadata_dir=meta, work_dir=work)\n'
    '    if _roi_sfm_path is not None:\n'
    "        downstream['sfm_roi_npz'] = str(_roi_sfm_path)\n"
)


def proposed(text):
    """Return state and full revised source; zero writes."""
    if 'publish_stage1_exact_roi_sfm(' in text:
        return 'already_installed', text
    if OLD_START in text:
        if text.count(OLD_START) != 1 or text.count(OLD_END) != 1:
            raise ValueError('Unexpected old approximate-SfM handoff block')
        a = text.index(OLD_START)
        b = text.index(OLD_END, a) + len(OLD_END)
        window = text[a:b]
        if (window.count('prepare_stage1_roi_sfm(') != 1 or
            'from gsrefine.geometry.roi_sfm import prepare_stage1_roi_sfm' not in window or
            'def publish(' not in text[:a]):
            raise ValueError('Unrecognized old SfM handoff block')
        modified = text[:a] + HOOK + text[b:]
        kind = 'replace_previous_sfm_handoff'
    else:
        if (text.count(ANCHOR) != 1 or
            'downstream = {' not in text or
            "'stage2_inputs': downstream" not in text or
            'def publish(' not in text or
            not (text.index('downstream = {') < text.index(ANCHOR) <
                 text.index("'stage2_inputs': downstream"))):
            raise ValueError('Stage-1 output_layout.py has unrecognized publication anchors')
        modified = text.replace(ANCHOR, ANCHOR + HOOK, 1)
        kind = 'insert_exact_sfm_handoff'
    compile(modified, 'output_layout.py', 'exec')
    return kind, modified


def install(root, *, backup_dir='/tmp/gsrefine_exact_sfm_source_backups', dry_run=False):
    root = Path(root).expanduser().resolve(strict=True)
    target = root / 'gsrefine/roi_init/output_layout.py'
    if not target.is_file():
        raise FileNotFoundError(target)
    before = target.read_text(encoding='utf8')
    state, after = proposed(before)
    if state == 'already_installed':
        return 'already_installed', None
    if dry_run:
        return 'would_patch', None
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f'output_layout_{sha256(before.encode()).hexdigest()[:16]}.py'
    if not backup.exists():
        shutil.copy2(target, backup)
    if target.read_text(encoding='utf8') != before:
        raise RuntimeError('Stage-1 publisher changed during publication-hook update')
    target.write_text(after, encoding='utf8')
    return 'patched', backup


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', default='/home/xth/GSSceneRefine')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    print(install(args.project_root, dry_run=args.dry_run)[0])
