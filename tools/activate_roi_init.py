"""Safely append Module 2 configuration without destroying user's existing YAML comments.

One-time configuration helper. It makes a versioned backup of the original YAML
under runs/config_backups/ and refuses to rewrite an already-configured section.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import yaml

ROOT=Path(__file__).resolve().parents[1]


def activate(path,scene,method):
    path=Path(path).resolve(strict=True)
    src=path.read_text(encoding='utf8')
    data=yaml.safe_load(src)
    if not isinstance(data,dict):raise ValueError('Invalid scene YAML root')
    scene_entry=data.get('scene')
    if scene_entry and (scene_entry.get('name')!=scene or scene_entry.get('method')!=method):
        raise ValueError('Existing scene.name/method differs from requested values; edit config explicitly')
    if 'roi_init' in data:
        print('[roi-init config] already configured; not overwriting:',path)
        return False
    backup=ROOT/'runs'/'config_backups'
    backup.mkdir(parents=True,exist_ok=True)
    target=backup/f'{path.stem}_{datetime.now().strftime("%Y%m%dT%H%M%S%f")}.yaml'
    shutil.copy2(path,target)
    block=(ROOT/'configs'/'roi_init.example.yaml').read_text(encoding='utf8')
    block=block.replace('songziguanyin_colmap_original',scene+'_colmap_original')
    add=('\n' if src.endswith('\n') else '\n\n')
    if not scene_entry:
        add+=f'\nscene:\n  name: {scene}\n  method: {method}\n'
    add+='\n'+block
    path.write_text(src+add,encoding='utf8')
    yaml.safe_load(path.read_text(encoding='utf8'))
    print('[roi-init config] added scene ROI configuration; backup:',target)
    print('[roi-init config] vertical will be estimated from COLMAP images.bin automatically; review diagnostic output if rejected')
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/songziguanyin.yaml')
    p.add_argument('--scene',default='songziguanyin')
    p.add_argument('--method',default='absgs')
    a=p.parse_args()
    activate(a.config,a.scene,a.method)

if __name__=='__main__':main()
