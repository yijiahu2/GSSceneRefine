"""Fix one missing V3 audit cache status check in stage2_onepass.py.

Checks the precise V3 implementation, parses the target function with AST,
refuses unknown versions, and atomically modifies ONLY its cache filter.
No R3/Poisson/Gaussian data or other production module is modified.

python tools/fix_stage2_v3_cache_status.py --check
python tools/fix_stage2_v3_cache_status.py
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

TARGET = Path('gsrefine/cleanup/stage2_onepass.py')
EXPECTED_VERSION = 'final_R3_original_global_poisson_region_graph_v3_window_20261112'
EXPECTED_STATUS = 'REGION_GRAPH_SAMPLED_NOT_CERTIFIED'
ANCHOR = "            coarse,fine=data['flood']['coarse'],data['flood']['fine']\n"
GUARD = (
    "            if any(level.get('status') != 'REGION_GRAPH_SAMPLED_NOT_CERTIFIED'\n"
    "                   for level in (coarse, fine)):\n"
    "                continue\n"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transform(text: str) -> tuple[str, bool]:
    tree = ast.parse(text)
    found = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name == '_find_barrier_audit']
    if len(found) != 1:
        raise RuntimeError('Expected exactly one _find_barrier_audit function')
    fn = found[0]
    lines = text.splitlines(keepends=True)
    original = ''.join(lines[fn.lineno - 1:fn.end_lineno])
    if EXPECTED_VERSION not in original or "REGION_GRAPHS_COMPLETED_NUMERIC_NOT_CERTIFIED" not in original:
        raise RuntimeError('Expected V3 audit schema/status not found; refusing to patch unknown code')
    if "r3_boundary_touch_count" not in original:
        raise RuntimeError('V3 boundary field not found; refusing to patch unknown code')
    if original.count(ANCHOR) != 1:
        raise RuntimeError('Cannot uniquely locate coarse/fine audit records in the cache selector')
    if GUARD in original:
        return text, False
    # Require exactly the old omitted guard; do not stack multiple conditions.
    if "level.get('status')" in original or "x.get('status')" in original:
        raise RuntimeError('An unfamiliar status guard exists; review manually rather than stacking guards')
    updated = original.replace(ANCHOR, ANCHOR + GUARD, 1)
    result = ''.join(lines[:fn.lineno - 1]) + updated + ''.join(lines[fn.end_lineno:])
    compile(result, str(TARGET), 'exec')
    return result, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate without changing files')
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve(strict=True)
    target = root / TARGET
    original_bytes = target.read_bytes()
    original = original_bytes.decode('utf-8')
    revised, needs_change = transform(original)
    if not needs_change:
        print('[stage2-cache-fix] Already fixed; nothing to do.')
        return
    print('[stage2-cache-fix] Valid V3 code; inserting coarse/fine audit status validation.')
    if args.check:
        print('[stage2-cache-fix] --check passed; no files changed.')
        return
    backup_dir = root / '.git' / 'gsrefine_patch_backups' / 'stage2_v3_cache_status'
    if not (root / '.git').is_dir():
        raise RuntimeError('Expected .git directory; backup path is unavailable')
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / ('stage2_onepass_' + sha256(original_bytes)[:16] + '.py')
    if backup.exists():
        if backup.read_bytes() != original_bytes:
            raise RuntimeError('Existing backup has conflicting contents')
    else:
        backup.write_bytes(original_bytes)
    revised_bytes = revised.encode('utf-8')
    fd, temp_path = tempfile.mkstemp(prefix='.stage2_onepass_cache_fix_', suffix='.py', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(revised_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_path, target.stat().st_mode)
        if target.read_bytes() != original_bytes:
            raise RuntimeError('Target changed during preparation; refusing overwrite')
        os.replace(temp_path, target)
    finally:
        Path(temp_path).unlink(missing_ok=True)
    final, changed = transform(target.read_text(encoding='utf-8'))
    if changed:
        raise RuntimeError('Unexpected post-install verification failure')
    print('[stage2-cache-fix] Installed: ' + str(TARGET))
    print('[stage2-cache-fix] Backup: ' + str(backup))


if __name__ == '__main__':
    main()
