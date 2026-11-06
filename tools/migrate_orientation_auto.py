"""Conservative, comment-preserving migration of existing ROI orientation settings.

Only update roi_init.orientation.mode and .colmap_images. Never rewrite the
whole YAML document or replace any user-defined paths, comments or other keys.
A timestamped backup is saved before any modification. Use --dry-run first.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _config_block(lines, *, name, indent, start=0, end=None):
    """Return [start, end) for a conventional block mapping (spaces-only YAML)."""
    if end is None:
        end = len(lines)
    pat = re.compile(r'^' + ' ' * indent + re.escape(name) + r':(?:\s*(?:#.*)?)?\r?\n?$')
    hits = [i for i in range(start, end) if pat.match(lines[i])]
    if len(hits) != 1:
        raise ValueError(f'Expected exactly one conventional {name}: block with {indent} spaces; found {len(hits)}. No changes made.')
    i = hits[0]
    j = i + 1
    while j < end:
        line = lines[j]
        if line.strip() and not line.lstrip().startswith('#'):
            leading = len(line) - len(line.lstrip(' '))
            if leading <= indent:
                break
        j += 1
    return i, j


def _direct_key(lines, name, indent, start, end):
    pat = re.compile(r'^' + ' ' * indent + re.escape(name) + r':(?:\s|\r?\n|$)')
    hits = [i for i in range(start, end) if pat.match(lines[i])]
    if len(hits) > 1:
        raise ValueError(f'Duplicate {name} keys found; no changes made')
    return hits[0] if hits else None


def migrate(path, colmap_images=None, *, dry_run=False, force_auto=False):
    path = Path(path).expanduser().resolve(strict=True)
    source = path.read_text(encoding='utf8')
    original = yaml.safe_load(source)
    if not isinstance(original, dict) or not isinstance(original.get('roi_init'), dict):
        raise ValueError('Config must contain a roi_init mapping. No changes made.')
    scene = original.get('scene') or {}
    if not isinstance(scene, dict):
        raise ValueError('scene must be a mapping. No changes made.')
    orient = original['roi_init'].get('orientation')
    if orient is None:
        orient = {}
    if not isinstance(orient, dict):
        raise ValueError('roi_init.orientation must be a mapping or empty. No changes made.')
    if orient.get('mode') == 'manual' and not force_auto:
        raise ValueError('Existing manual orientation detected; refusing automatic overwrite. Use --force-auto if intentional.')
    if orient.get('mode') not in (None, 'auto', 'manual'):
        raise ValueError('Unknown orientation.mode value; refusing to overwrite manual settings.')
    if colmap_images is not None:
        images = str(Path(colmap_images).expanduser())
    elif orient.get('colmap_images'):
        images = str(orient['colmap_images'])
    elif (original.get('paths') or {}).get('colmap_images'):
        images = str(original['paths']['colmap_images'])
    else:
        name = scene.get('name') or original['roi_init'].get('scene')
        if not name or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', str(name)):
            raise ValueError('scene.name missing or unsafe: supply --colmap-images explicitly. No changes made.')
        images = f'/home/xth/datasets/danxiashan/{name}/sparse/images.bin'
    if not Path(images).is_absolute():
        raise ValueError('colmap_images must be an absolute WSL path. No changes made.')
    # No filesystem existence requirement during migration: the orientation
    # estimator itself will validate actual COLMAP input when run.
    expected = copy.deepcopy(original)
    expected['roi_init']['orientation'] = dict(orient)
    expected['roi_init']['orientation']['mode'] = 'auto'
    expected['roi_init']['orientation']['colmap_images'] = images
    if original == expected:
        print('[orientation] Already configured; no change:', path)
        return False

    lines = source.splitlines(keepends=True)
    roi_start, roi_end = _config_block(lines, name='roi_init', indent=0)
    orient_line = _direct_key(lines, 'orientation', 2, roi_start + 1, roi_end)
    if orient_line is None:
        lines.insert(roi_end, f'  orientation:\n    mode: auto\n    colmap_images: {images}\n')
    else:
        # Empty YAML values may be written as orientation:, orientation: null,
        # orientation: ~, or orientation: {}. Normalize only these safe cases.
        if orient == {} and re.match(r'^  orientation:\s*(?:null|~|\{\})\s*(?:#.*)?\r?\n?$', lines[orient_line]):
            lines[orient_line] = '  orientation:\n'
        ori_start, ori_end = _config_block(lines, name='orientation', indent=2, start=roi_start + 1, end=roi_end)
        if not re.match(r'^  orientation:\s*(?:#.*)?\r?\n?$', lines[ori_start]):
            raise ValueError('Unsupported orientation YAML layout; no changes made.')
        mline = _direct_key(lines, 'mode', 4, ori_start + 1, ori_end)
        iline = _direct_key(lines, 'colmap_images', 4, ori_start + 1, ori_end)
        # Work in reverse order to keep indices valid; preserve custom keys.
        changes = []
        if mline is not None:
            comment = ''
            if '#' in lines[mline]:
                comment = '  ' + lines[mline].split('#',1)[1].strip()
                comment = '  #'+comment
            changes.append((mline, '    mode: auto'+comment+'\n'))
        else:
            changes.append((ori_start+1, '    mode: auto\n', 'insert'))
        if iline is not None:
            if colmap_images is not None:
                changes.append((iline, f'    colmap_images: {images}\n'))
        else:
            changes.append((ori_start+1, f'    colmap_images: {images}\n', 'insert'))
        for entry in sorted(changes, key=lambda p:p[0], reverse=True):
            if len(entry)==3:
                lines.insert(entry[0],entry[1])
            else:
                lines[entry[0]] = entry[1]
    revised = ''.join(lines)
    parsed = yaml.safe_load(revised)
    if parsed != expected:
        raise ValueError('Migration safety check rejected unexpected YAML changes; original file not modified.')
    if dry_run:
        print('[orientation] DRY RUN passed, original config unchanged:', path)
        print('[orientation] Would use images:', images)
        return True

    backups = ROOT / 'runs' / 'config_backups'
    backups.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = backups / f'{path.stem}_pre_auto_orientation_{stamp}.yaml'
    shutil.copy2(path, backup)
    fd, temp = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf8') as f:
            f.write(revised)
            f.flush()
            os.fsync(f.fileno())
        shutil.copymode(path, temp)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    print('[orientation] Config updated:', path)
    print('[orientation] Backup:', backup)
    print('[orientation] COLMAP images:', images)
    if 'up_axis' in orient or 'verified' in orient:
        print('[orientation] Legacy up_axis/verified keys preserved but ignored by auto mode')
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/songziguanyin.yaml')
    p.add_argument('--colmap-images', default=None)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--force-auto', action='store_true', help='Explicitly switch a custom manual mode to auto')
    args = p.parse_args()
    migrate(args.config, args.colmap_images, dry_run=args.dry_run, force_auto=args.force_auto)


if __name__ == '__main__':
    main()
