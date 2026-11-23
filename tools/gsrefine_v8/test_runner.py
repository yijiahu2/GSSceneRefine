"""Standalone command-router tests, executed without user's large datasets."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

MOD = Path(__file__).resolve().parents[2] / 'gsrefine' / 'roi_init' / 'formal_pipeline.py'
modspec = importlib.util.spec_from_file_location('_formal_pipeline_test', MOD)
mod = importlib.util.module_from_spec(modspec)
modspec.loader.exec_module(mod)


def test_all():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root/'sparse').mkdir()
        (root/'sparse'/'points3D.bin').write_bytes(b'points')
        (root/'sparse'/'images.bin').write_bytes(b'images')
        (root/'cfg.yaml').write_text('paths: {}')
        (root/'src.ply').write_bytes(b'ply\n')
        (root/'roi_001.json').write_text(json.dumps({
            'views':[{'name':'top'},{'name':'front'}],
            'annotation_metadata':{'orientation_verified':True}}))
        captured = []
        original = mod._run
        try:
            mod._run = lambda cmd: captured.append(cmd)
            base = ['--config',str(root/'cfg.yaml'),'--scene','lfls','--method','gsplat_accel',
                    '--sfm-dir',str(root/'sparse'),'--gaussian-ply',str(root/'src.ply'),
                    '--output-dir',str(root/'results'),'--accept-auto-exterior']
            a = mod.parser().parse_args(['full',*base,'--no-aux-protection'])
            assert mod.run(a) is None
            assert len(captured)==1 and 'full-pipeline' in captured[0]
            assert 'original-region' in captured[0] and '--no-aux-protection' in captured[0]
            captured.clear()
            manifest = root/'results'/'lfls'/'methods'/'gsplat_accel'/'roi_001'/'metadata'/'manifest.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({'status':'completed'}))
            a = mod.parser().parse_args(['stage2','--manifest',str(manifest),'--accept-auto-exterior'])
            assert mod.run(a)==manifest and len(captured)==1 and '--resume-manifest' in captured[0]
            captured.clear()
            # Replace only the output-layout calculation during standalone test.
            package = types.ModuleType('gsrefine')
            package.__path__=[]
            sub = types.ModuleType('gsrefine.roi_init')
            sub.__path__=[]
            output = types.ModuleType('gsrefine.roi_init.output_layout')
            output.run_dir = lambda output_root, scene, method, roi: root/'results'/'lfls'/'methods'/'gsplat_accel'/'roi_002'
            saved = {name:sys.modules.get(name) for name in (
                'gsrefine','gsrefine.roi_init','gsrefine.roi_init.output_layout')}
            sys.modules.update({'gsrefine':package,'gsrefine.roi_init':sub,
                                'gsrefine.roi_init.output_layout':output})
            try:
                def fake(cmd):
                    captured.append(cmd)
                    if 'roi-auto' in cmd:
                        m = root/'results'/'lfls'/'methods'/'gsplat_accel'/'roi_002'/'metadata'/'manifest.json'
                        m.parent.mkdir(parents=True,exist_ok=True)
                        m.write_text(json.dumps({'status':'completed'}))
                mod._run = fake
                a = mod.parser().parse_args(['from-roi',*base,'--roi',str(root/'roi_001.json')])
                result = mod.run(a)
                assert len(captured)==2
                assert 'roi-auto' in captured[0] and '--roi' in captured[0]
                assert 'full-pipeline' in captured[1] and '--resume-manifest' in captured[1]
                assert result.name == 'manifest.json'
            finally:
                for key, value in saved.items():
                    if value is None:sys.modules.pop(key,None)
                    else:sys.modules[key]=value
        finally:
            mod._run=original
    print('PASS: full, stage2 resume, from-roi routed to validated original commands')


if __name__ == '__main__':
    test_all()
