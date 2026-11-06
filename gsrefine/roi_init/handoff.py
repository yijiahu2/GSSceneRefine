"""Refine latest scene ROI by default; --roi pins an older experimental record."""
from __future__ import annotations
from pathlib import Path
import argparse
from gsrefine.roi.core import run
from gsrefine.roi_init.settings import InitSettings
from gsrefine.roi_init.results import current_latest
import json


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--modes',default='hard,global,local')
    p.add_argument('--roi',default=None,help='Explicit ROI JSON override; default is latest confirmed ROI for the scene')
    p.add_argument('--show-roi',action='store_true',help='Print the ROI that would be used and exit without computation')
    p.add_argument('--output-dir',default=None,help='Separate output for experiment; prevents overwriting prior masks')
    args=p.parse_args(argv)
    settings=InitSettings.load(args.config)
    roi=Path(args.roi).expanduser().resolve(strict=True) if args.roi else current_latest(settings)
    if roi is None:
        raise FileNotFoundError('No confirmed ROI for this scene. Run python run.py roi-init first.')
    meta=json.loads(roi.read_text(encoding='utf8')).get('annotation_metadata',{})
    if (meta.get('scene') != settings.scene or
        meta.get('frame_id') != settings.frame_id or
        meta.get('orientation_source_sha256')!=settings.orientation_result.get('source_sha256') or
        meta.get('up_vector') != list(settings.up_vector)):
        raise ValueError('Selected ROI frame conflicts with configured COLMAP coordinate system')
    print('[roi-refine] Using '+('explicit' if args.roi else 'latest confirmed')+' ROI:',roi,flush=True)
    if args.show_roi:return roi
    run(args.config,str(roi),args.modes.split(','),args.output_dir)

if __name__=='__main__':main()
