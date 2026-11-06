"""Confirm the scene-wide ROI actually exists on disk; no Gaussian processing."""
from __future__ import annotations
import argparse
import hashlib
import json
from gsrefine.roi_init.settings import InitSettings
from gsrefine.roi_init.results import current_canonical, current_latest, read_manifest


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True)
    args=ap.parse_args(argv)
    settings=InitSettings.load(args.config)
    canonical=current_canonical(settings)
    current=current_latest(settings)
    manifest=read_manifest(settings.out_dir)
    output={'scene':settings.scene,'method':settings.method,'output_dir':str(settings.out_dir),
            'canonical_path':str(canonical) if canonical else None,
            'latest_path':str(current) if current else None,'saved_exists':bool(current),
            'records':len(manifest['records'])}
    if current:
        data=json.loads(current.read_text(encoding='utf8'))
        ann=data.get('annotation_metadata',{})
        output.update(sha256=hashlib.sha256(current.read_bytes()).hexdigest(),
                      confirmed_utc=ann.get('confirmed_utc'),
                      gaussian_roi_count=ann.get('initial_gaussian_roi_count'),
                      view_count=len(data.get('views',[])))
    print(json.dumps(output,ensure_ascii=False,indent=2))
    if not current:raise SystemExit('No confirmed ROI found for this scene')

if __name__=='__main__':main()
