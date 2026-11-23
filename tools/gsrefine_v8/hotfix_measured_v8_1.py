#!/usr/bin/env python3
"""GSSceneRefine v8.1 repair: eliminate Python local-import shadowing in stage2.

Only removes the duplicate function-local `measured` import introduced by the
v8 engineering patch. The module-level measured import remains. No scientific
algorithm, threshold, classification rule, or output is changed.
"""
from pathlib import Path
import ast
import argparse
import os
import shutil
import tempfile

REL = 'gsrefine/cleanup/local_region_pipeline.py'
DUPLICATE = '    from gsrefine.cleanup.stage2_runtime import measured\n'
TOP_LEVEL = 'from gsrefine.cleanup.stage2_runtime import file_digest, measured, snapshot'


def assess(source):
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'run']
    if len(functions) != 1:
        raise RuntimeError('Expected exactly one stage2 run() function')
    run = functions[0]
    imports = [node for node in ast.walk(run) if isinstance(node, ast.ImportFrom)
               and node.module == 'gsrefine.cleanup.stage2_runtime'
               and any(alias.name == 'measured' for alias in node.names)]
    if TOP_LEVEL not in source:
        raise RuntimeError('Expected module-level measured import is missing')
    return len(imports)


def repair(root, check=False):
    root = Path(root).expanduser().resolve()
    path = root / REL
    if not path.is_file():
        raise FileNotFoundError(path)
    old = path.read_text(encoding='utf-8')
    count = old.count(DUPLICATE)
    local_imports = assess(old)
    if count == 0 and local_imports == 0:
        print('[v8.1] ALREADY FIXED:', path)
        return
    if count != 1 or local_imports != 1 or "measured('gaussian_radius_statistics')" not in old:
        raise RuntimeError('Unexpected source layout; refusing unsafe replacement. '
                           f'duplicate_lines={count}, local_imports={local_imports}')
    new = old.replace(DUPLICATE, '', 1)
    if assess(new) != 0:
        raise RuntimeError('Repaired run() still has local measured import')
    ast.parse(new, filename=str(path))
    if check:
        print('[v8.1] CHECK PASSED: exactly one duplicate local import can be removed')
        return
    backup = root / 'tools/gsrefine_v8/local_region_pipeline_before_v8_1.py.bak'
    if backup.exists():
        raise FileExistsError(f'Backup already exists; refusing to overwrite: {backup}')
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup)
    descriptor, staged = tempfile.mkstemp(prefix='.local_region_pipeline_', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(new)
        os.chmod(staged, path.stat().st_mode)
        os.replace(staged, path)
    finally:
        Path(staged).unlink(missing_ok=True)
    print('[v8.1] FIXED:', path)
    print('[v8.1] BACKUP:', backup)
    print('[v8.1] VERIFIED: run() uses the module-level measured; scientific rules unchanged')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='/home/xth/GSSceneRefine')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    repair(args.project_root, check=args.check)


if __name__ == '__main__':
    main()
