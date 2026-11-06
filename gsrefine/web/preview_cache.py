#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-time, visual-only SOG cache for the full-precision Gaussian PLY.

The source PLY is NEVER overwritten. ROI/Poisson/SDF computation always uses it.
SOG is visually lossy, intended only for the two orthographic browser views.

One-time converter: PlayCanvas SplatTransform (project-local Node dependency).
Requires Node.js 22 in the existing gsrefine Conda environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import yaml


def file_signature(path: Path) -> str:
    """Cheap cache key: source identity + size + high-resolution change times.

    This is a preview invalidation token, NOT a cryptographic proof of contents.
    Long-running scientific operations continue to track their own source hashes.
    """
    path = Path(path).expanduser().resolve(strict=True)
    st = path.stat()
    msg = f'{path}\0{st.st_size}\0{st.st_mtime_ns}\0{st.st_ctime_ns}'
    return hashlib.blake2s(msg.encode('utf-8'), digest_size=10).hexdigest()


def cache_paths(source: Path, output_dir: Path):
    directory = Path(output_dir) / '_display_cache'
    stem = f'model_{file_signature(source)}'
    return directory / (stem + '.sog'), directory / (stem + '.json')


def find_sog_cache(source: Path, output_dir: Path):
    sog, manifest = cache_paths(source, output_dir)
    if not sog.is_file() or not manifest.is_file():
        return None
    try:
        record = json.loads(manifest.read_text(encoding='utf-8'))
        if record.get('source_signature') != file_signature(source):
            return None
        if not zipfile.is_zipfile(sog) or sog.stat().st_size == 0:
            return None
        with zipfile.ZipFile(sog) as archive:
            if 'meta.json' not in archive.namelist():
                return None
    except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile):
        return None
    return sog


def find_converter(project_root: Path):
    candidates = [project_root / 'ui/node_modules/.bin/splat-transform']
    for p in candidates:
        if p.is_file():
            return str(p)
    return shutil.which('splat-transform')


def build_sog_cache(source: Path, output_dir: Path, project_root: Path,
                    force=False, device='auto'):
    source = Path(source).expanduser().resolve(strict=True)
    if source.suffix.lower() != '.ply':
        raise ValueError('Only native full-precision Gaussian .ply is supported as source')
    target, manifest = cache_paths(source, output_dir)
    if not force:
        existing = find_sog_cache(source, output_dir)
        if existing is not None:
            print(f'[cache] Reusing existing visual SOG: {existing} ({existing.stat().st_size:,} bytes)')
            return existing
    cli = find_converter(project_root)
    if cli is None:
        raise RuntimeError(
            'SplatTransform is missing. Install required project dependencies:\n'
            '  cd /home/xth/GSSceneRefine\n'
            '  bash bootstrap.sh --sync-deps\n'
            'This maintains the existing gsrefine Conda environment in place.\n'
            'SplatTransform 3.x requires Node.js >=22; verify with node --version.\n'
            'Alternatively, export a full-resolution .sog from SuperSplat Editor and '
            'start the UI with --display-sog /absolute/path/scene.sog.'
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.stem + '.working.sog')
    if tmp.exists():
        tmp.unlink()
    cmd = [cli]
    if device != 'auto':
        if device != 'cpu' and not device.isdecimal():
            raise ValueError('device must be auto, cpu, or integer GPU index')
        cmd += ['-g', device]
    cmd += [str(source), str(tmp)]
    print('[cache] One-time SOG conversion; full-res PLY will NOT be modified.')
    print('[cache] ' + ' '.join(cmd))
    started = time.monotonic()
    try:
        subprocess.run(cmd, check=True)
        if not zipfile.is_zipfile(tmp):
            raise ValueError('SplatTransform did not produce a valid bundled SOG archive')
        with zipfile.ZipFile(tmp) as bundle:
            if 'meta.json' not in bundle.namelist():
                raise ValueError('Converted SOG has no meta.json')
        os.replace(tmp, target)
        record = {
            'source_ply': str(source),
            'source_signature': file_signature(source),
            'source_size_bytes': source.stat().st_size,
            'sog_path': str(target),
            'sog_size_bytes': target.stat().st_size,
            'display_only_lossy': True,
            'duration_seconds': round(time.monotonic() - started, 2),
            'converter_command': cmd,
        }
        tmp_manifest = manifest.with_suffix('.json.tmp')
        tmp_manifest.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp_manifest, manifest)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f'[cache] SOG ready: {target} ({target.stat().st_size:,} bytes)')
    return target


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/songziguanyin.yaml')
    p.add_argument('--source-ply', default=None, help='Override paths.gaussian_ply')
    p.add_argument('--output-dir', default=None, help='Override paths.output_dir for cache storage')
    p.add_argument('--device', default='auto', help='auto, cpu, or a GPU index (0, 1, ...)')
    p.add_argument('--force', action='store_true', help='Rebuild and replace cache for the SAME source')
    a = p.parse_args(argv)
    project = Path(__file__).resolve().parents[2]
    config = (project / a.config).resolve() if not Path(a.config).is_absolute() else Path(a.config)
    cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
    src = Path(a.source_ply or cfg['paths']['gaussian_ply'])
    output = Path(a.output_dir or cfg['paths']['output_dir'])
    if not output.is_absolute():
        output = project / output
    build_sog_cache(src, output, project, a.force, a.device)


if __name__ == '__main__':
    main()
