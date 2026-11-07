#!/usr/bin/env python3
"""Conservatively insert a *publication-only* Stage-1 data retention hook.

Runs in the user's existing GSSceneRefine checkout. Never downloads or uses
GitHub. Fails without changes if the local Stage-1 publisher no longer has
its known structural anchors. Saves old publisher outside the source tree.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

ANCHOR = "    shutil.copy2(config, meta/'effective_scene.yaml')\n"
HOOK = (
"    # Stage 2 may reuse a certified surface mask already made in Stage 1.\n"
"    # This is a no-op if Stage 1 did not produce the optional handoff.\n"
"    from gsrefine.geometry.stage1_handoff import retain_stage1_surface_faces\n"
"    _stage2_faces = retain_stage1_surface_faces(work, meta, paths['poisson_raw'], roi)\n"
"    if _stage2_faces is not None:\n"
"        downstream['roi_surface_faces'] = str(_stage2_faces)\n"
)

def apply(project_root, backup_root=None):
    root = Path(project_root).expanduser().resolve(strict=True)
    path = root / 'gsrefine/roi_init/output_layout.py'
    if not path.is_file():
        raise FileNotFoundError(path)
    content = path.read_text(encoding='utf8')
    if "retain_stage1_surface_faces(work, meta," in content:
        return 'already_installed', None
    # Require familiar publication code. Refuse to guess if local source differs.
    if (content.count(ANCHOR) != 1 or
        "downstream = {" not in content or
        "'stage2_inputs': downstream" not in content or
        "def publish(" not in content):
        raise ValueError('Local Stage-1 publisher differs from the inspected interface; '
                         'not modified. Review this file and add hook in publish() after '
                         'effective_scene.yaml is saved, before manifest publication.')
    if backup_root is None:
        backup_root = Path('/tmp/gsrefine_stage2_source_backups')
    backup_root = Path(backup_root).expanduser().resolve()
    backup_root.mkdir(parents=True, exist_ok=True)
    from hashlib import sha256
    digest=sha256(content.encode('utf8')).hexdigest()[:16]
    backup = backup_root / f'output_layout_{digest}.py'
    if not backup.exists():
        shutil.copy2(path, backup)
    changed=content.replace(ANCHOR, ANCHOR+HOOK)
    compile(changed, str(path), 'exec')
    path.write_text(changed, encoding='utf8')
    return 'patched', backup

def ensure_default_parameters(project_root, backup_root=None):
    """Append generic Stage-2 defaults in the EXISTING default.yaml, if absent."""
    root=Path(project_root).expanduser().resolve(strict=True)
    path=root/'configs/default.yaml'
    if not path.is_file():
        raise FileNotFoundError(path)
    import yaml
    original=path.read_text(encoding='utf8')
    doc=yaml.safe_load(original)
    if not isinstance(doc,dict):
        raise ValueError('Existing configs/default.yaml is not a YAML mapping')
    if 'stage2_cleanup' in doc:
        return 'already_configured', None
    backup_root=(Path(backup_root).expanduser().resolve()
                 if backup_root else Path('/tmp/gsrefine_stage2_source_backups'))
    backup_root.mkdir(parents=True,exist_ok=True)
    from hashlib import sha256
    backup=backup_root/f'default_yaml_{sha256(original.encode()).hexdigest()[:16]}.yaml'
    if not backup.is_file():shutil.copy2(path,backup)
    section="""
# Stage-2 ROI trusted-Poisson extraction and direct SDF cleanup.
# These are generic algorithm defaults, not scene coordinates or paths.
stage2_cleanup:
  trim_fraction: 0.10
  sigma_multiplier: 3.0
  epsilon: 0.00001
  chunk_size: 40000
  guard_max_outside_fraction: 0.025
  guard_inside_search_fraction: 0.030
  guard_geodesic_limit_fraction: 0.040
  guard_max_candidates: 15000
  guard_geometry_distance_multiplier: 3.0
  guard_sfm_spacing_multiplier: 3.0
  guard_sphere_radius_multiplier: 2.0
  guard_overlap_multiplier: 1.5
  max_unresolved_face_violation: 0.00001
  sfm_interior_margin_radius_multiplier: 1.0
  min_roi_fit_points: 100
  min_camera_outside_fraction: 1.0
  allow_inferred_guard_export: false
"""
    updated=original.rstrip()+"\n"+section
    checked=yaml.safe_load(updated)
    if not isinstance(checked.get('stage2_cleanup'),dict):
        raise ValueError('Failed to validate appended stage2 config')
    path.write_text(updated, encoding='utf8')
    return 'configured',backup


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='/home/xth/GSSceneRefine')
    parser.add_argument('--backup-dir')
    args=parser.parse_args()
    state, backup = apply(args.project_root, args.backup_dir)
    print(f'[stage2-handoff] {state}' + (f'; backup={backup}' if backup else ''))
    status, cfg_backup = ensure_default_parameters(args.project_root,args.backup_dir)
    print(f'[stage2-defaults] {status}' + (f'; backup={cfg_backup}' if cfg_backup else ''))
