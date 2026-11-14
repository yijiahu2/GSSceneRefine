#!/usr/bin/env python3
"""One command entrypoint for the GSSceneRefine project.

Usage:
  python run.py roi-init --config configs/songziguanyin.yaml
  python run.py roi-auto --config configs/songziguanyin.yaml --roi FILE --output-dir ROOT
  python run.py stage2-cleanup --manifest ROOT/SCENE/methods/METHOD/roi_001/metadata/manifest.json
  python run.py orientation --config configs/songziguanyin.yaml --check-poisson
  python run.py ui --config configs/songziguanyin.yaml
  python run.py roi --config configs/songziguanyin.yaml --roi FILE.json
  python run.py scene-prepare --config configs/songziguanyin.yaml
  python run.py poisson --help
  python run.py sfm-clean --help
  python run.py roi-finalize --study-dir runs/... --method R3
  python run.py roi-debug --help
  python run.py doctor --check-data

All modules run in ONE Conda environment: gsrefine.
"""
from __future__ import annotations
import runpy
import sys

COMMANDS = {
    'ui':'gsrefine.web.server',
    'roi-init':'gsrefine.roi_init.server',
    'full-pipeline':'gsrefine.roi_init.full_pipeline',
    'roi-auto':'gsrefine.roi_init.workflow',
    'roi-status':'gsrefine.roi_init.status',
    'orientation':'gsrefine.geometry.diagnose_orientation',
    'roi-refine':'gsrefine.roi_init.handoff',
    'roi':'gsrefine.roi.core',
    'roi-study':'gsrefine.roi.study',
    'roi-finalize':'gsrefine.roi.finalize',
    'poisson':'gsrefine.geometry.poisson',
    'scene-prepare':'gsrefine.geometry.prepare',
    'sfm-clean':'gsrefine.geometry.sfm_clean',
    'roi-debug':'tools.make_roi_from_aabb',
    'doctor':'gsrefine.doctor',
    'stage2-cleanup':'gsrefine.cleanup.direct_r3_pipeline',
    'stage2-sfm':'gsrefine.geometry.r3_sfm_bridge',
}

def main():
    if len(sys.argv)<2 or sys.argv[1] in ('-h','--help','help'):
        print(__doc__)
        print('Commands: '+', '.join(COMMANDS))
        return 0 if len(sys.argv)>1 else 2
    command=sys.argv.pop(1)
    if command not in COMMANDS:
        print(f'Unknown command: {command}; choose from {", ".join(COMMANDS)}',file=sys.stderr)
        return 2
    runpy.run_module(COMMANDS[command],run_name='__main__')
    return 0

if __name__=='__main__':raise SystemExit(main())
