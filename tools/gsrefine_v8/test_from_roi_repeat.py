#!/usr/bin/env python3
"""Offline contract tests for repeat-from-ROI, no real Gaussian data needed."""
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import yaml

MODULE = Path(__file__).resolve().parents[2]/'gsrefine/roi_init/formal_pipeline.py'
spec = importlib.util.spec_from_file_location('formal_pipeline_test', MODULE)
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


class RepeatContract(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)/'results'
        self.roi_dir = self.root/'lfls/roi'
        self.roi_dir.mkdir(parents=True)
        self.roi = self.roi_dir/'roi_003.json'
        self.gauss = Path(self.work.name)/'source.ply'
        self.gauss.write_bytes(b'ply\nend_header\n')
        self.sfm = Path(self.work.name)/'sparse/0'
        self.sfm.mkdir(parents=True)
        (self.sfm/'points3D.bin').write_bytes(b'test points')
        (self.sfm/'images.bin').write_bytes(b'test images')
        self.geometry = self.root/'lfls/geometry'
        (self.geometry/'metadata').mkdir(parents=True)
        (self.geometry/'metadata/geometry_manifest.json').write_text(json.dumps({'signature':'sig',
            'signature_material':{'colmap_points_sha256':'pointssha'},
            'source':str(self.sfm/'points3D.bin')}))
        (self.geometry/'cleaned_sfm.npz').write_bytes(b'test clean')
        (self.geometry/'poisson_surface.ply').write_bytes(b'test mesh')
        self.cfg = Path(self.work.name)/'default.yaml'
        self.cfg.write_text('scene:\n  name: lfls\n  method: gsplat_accel\n', encoding='utf8')
        views = [{'name':'top','origin':[0,0,0],'axes':[[1,0,0],[0,1,0]],
                  'polygon':[[0,0],[1,0],[1,1],[0,1]]},
                 {'name':'front','origin':[0,0,0],'axes':[[1,0,0],[0,0,1]],
                  'polygon':[[0,0],[1,0],[1,1],[0,1]]}]
        self.view_sha = self._views_sha({'views':views})
        self.annotation = {'views':views, 'annotation_metadata':{
            'orientation_verified':True,'view_sha256':self.view_sha,
            'scene':'lfls', 'frame_id':'lfls_colmap_original',
            'orientation_source_sha256':'orientationsha',
            'source_ply':str(self.gauss), 'method':'gsplat_accel'}}
        self.roi.write_text(json.dumps(self.annotation), encoding='utf8')
        registry = {'records':[{'filename':'roi_003.json','scene':'lfls',
            'method':'gsplat_accel','source_ply':str(self.gauss),
            'view_sha256':self.view_sha, 'frame_id':'lfls_colmap_original',
            'orientation_source_sha256':'orientationsha'}]}
        (self.roi_dir/'manifest.json').write_text(json.dumps(registry), encoding='utf8')
        self.args = fp.parser().parse_args(['from-roi','--config',str(self.cfg),
            '--scene','lfls','--method','gsplat_accel', '--sfm-dir',str(self.sfm),
            '--gaussian-ply',str(self.gauss),'--output-dir',str(self.root),
            '--roi',str(self.roi),'--accept-auto-exterior'])
        self.calls=[]
        self.fail_stage2=False
        self.modules={
            'gsrefine':module('gsrefine'),
            'gsrefine.roi_init':module('gsrefine.roi_init'),
            'gsrefine.geometry':module('gsrefine.geometry'),
            'gsrefine.roi_init.results':module('gsrefine.roi_init.results', canonical_views_sha=self._views_sha),
            'gsrefine.roi_init.output_layout':module('gsrefine.roi_init.output_layout',run_dir=self._run_dir),
            'gsrefine.roi_init.workflow':module('gsrefine.roi_init.workflow',run_once=self._run_once),
            'gsrefine.geometry.prepare':module('gsrefine.geometry.prepare',
                prepare_input_config=self._prepare_input_config,
                prepare_scene=lambda p:p,
                _source_signature=lambda *args:('sig',{'colmap_points_sha256':'pointssha'}),
                _outputs_ok=lambda *args:True,
                _options=lambda d:('auto',d.get('module1',{}),{},{})),
        }
        self.patcher=patch.dict(sys.modules,self.modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.runnerpatch=patch.object(fp,'_run',self._stage2)
        self.runnerpatch.start()
        self.addCleanup(self.runnerpatch.stop)

    @staticmethod
    def _views_sha(d):
        normalized=[{k:v[k] for k in ('name','origin','axes','polygon')} for v in d['views']]
        return hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    @staticmethod
    def _run_dir(root,scene,method,roi):
        return Path(root)/scene/'methods'/method/Path(roi).stem

    def _prepare_input_config(self, config, **kwargs):
        out=Path(kwargs['final_output_root'])/'lfls/geometry/runtime'
        out.mkdir(parents=True,exist_ok=True)
        effective=out/'active.yaml'
        effective.write_text(yaml.safe_dump({'module1':{'mode':'prepared','output_dir':str(out.parent)},
                  'scene':{'name':'lfls','method':'gsplat_accel'}}),encoding='utf8')
        return effective

    def _run_once(self, config, roi, output_dir, **kwargs):
        output_dir=Path(output_dir)
        sub=self._run_dir(output_dir,'lfls','gsplat_accel',roi)
        manifest=sub/'metadata/manifest.json'
        manifest.parent.mkdir(parents=True)
        (sub/'final').mkdir(parents=True)
        (sub/'final/gaussian_roi.ply').write_bytes(b'ply\nend_header\n')
        manifest.write_text(json.dumps({'status':'completed',
            'stage2_inputs':{'roi_json':str(roi)},'input_sha256':{'roi':sha(roi)}}))
        self.calls.append(('stage1',str(manifest)))
        effective = yaml.safe_load(Path(config).read_text(encoding='utf8'))
        assert effective['paths']['clean_sfm_npz'] == str(self.geometry/'cleaned_sfm.npz')
        assert effective['module1']['mode'] == 'prepared'
        assert effective['roi_init']['output_dir'] == str(self.roi_dir)
        return {'manifest':str(manifest),'output_dir':str(sub),'status':'completed'}

    def _stage2(self, cmd):
        assert cmd[cmd.index('--classifier')+1] == 'original-region'
        assert '--resume-manifest' in cmd
        mf=Path(cmd[cmd.index('--resume-manifest')+1]);assert mf.is_file()
        self.calls.append(('stage2',str(mf)))
        if self.fail_stage2:
            raise RuntimeError('simulated stage2 failure')
        (mf.parent/'full_pipeline_status.json').write_text(json.dumps({'status':'COMPLETED'}))

    def test_repeated_runs_use_two_isolated_roots_and_original_roi_is_unmodified(self):
        before=self.roi.read_bytes(); reg=(self.roi_dir/'manifest.json').read_bytes()
        first=fp.run(self.args); second=fp.run(self.args)
        self.assertNotEqual(first,second)
        for slot,mf in enumerate((first,second),1):
            self.assertIn(f'repeat_{slot:04d}',str(mf))
            self.assertEqual(json.loads(mf.read_text())['stage2_inputs']['roi_json'],str(self.roi))
            p=mf.parents[4]/'experiment.json'
            # Experiment root is the repeat_NNNN directory.
            if not p.is_file():
                p=next(self.root.glob(f'experiments/lfls/gsplat_accel/roi_003/repeat_{slot:04d}/experiment.json'))
            self.assertEqual(json.loads(p.read_text())['status'],'COMPLETED')
        self.assertEqual(self.roi.read_bytes(),before)
        self.assertEqual((self.geometry/'cleaned_sfm.npz').read_bytes(),b'test clean')
        self.assertEqual((self.geometry/'poisson_surface.ply').read_bytes(),b'test mesh')
        self.assertEqual((self.roi_dir/'manifest.json').read_bytes(),reg)
        self.assertEqual([x[0] for x in self.calls],['stage1','stage2','stage1','stage2'])

    def test_failed_run_is_never_reused(self):
        self.fail_stage2=True
        with self.assertRaisesRegex(RuntimeError,'simulated'):
            fp.run(self.args)
        failed=self.root/'experiments/lfls/gsplat_accel/roi_003/repeat_0001/experiment.json'
        self.assertEqual(json.loads(failed.read_text())['status'],'FAILED')
        self.fail_stage2=False
        next_run=fp.run(self.args)
        self.assertIn('repeat_0002',str(next_run))
        self.assertEqual(json.loads(failed.read_text())['status'],'FAILED')

    def test_edited_original_roi_rejected_before_reservation(self):
        self.annotation['annotation_metadata']['view_sha256']='changed'
        self.roi.write_text(json.dumps(self.annotation))
        with self.assertRaisesRegex(ValueError,'changed'):
            fp.run(self.args)
        self.assertFalse((self.root/'experiments').exists())

    def test_polygon_tampering_rejected_by_registered_hash(self):
        self.annotation['views'][0]['polygon'][0]=[0.2, 0.2]
        self.roi.write_text(json.dumps(self.annotation))
        with self.assertRaisesRegex(ValueError,'changed'):
            fp.run(self.args)
        self.assertFalse((self.root/'experiments').exists())

    def test_new_results_root_reuses_original_module1_without_copy(self):
        fresh=Path(self.work.name)/'fresh_results'
        self.args.output_dir=str(fresh)
        manifest=fp.run(self.args)
        self.assertTrue(manifest.is_file())
        control=fresh/'experiments/lfls/gsplat_accel/roi_003/repeat_0001/experiment.json'
        data=json.loads(control.read_text())
        self.assertEqual(data['geometry_reference']['root'],str(self.geometry))
        self.assertFalse((fresh/'lfls/geometry').exists())

    def test_previous_stage1_sha_prevents_manual_edit(self):
        old=self._run_dir(self.root,'lfls','gsplat_accel',self.roi)/'metadata/workflow_result.json'
        old.parent.mkdir(parents=True)
        old.write_text(json.dumps({'roi_sha256':'not-actual','roi':str(self.roi)}))
        with self.assertRaisesRegex(ValueError,'SHA256'):
            fp.run(self.args)

    def test_invalid_module1_cache_refused_before_reservation(self):
        with patch.dict(sys.modules,{'gsrefine.geometry.prepare':module('gsrefine.geometry.prepare',
            _source_signature=lambda *args:('different',{'colmap_points_sha256':'pointssha'}),
            _outputs_ok=lambda *args:False,
            _options=lambda d:('auto',{}, {}, {}))}):
            with self.assertRaisesRegex(ValueError,'Module-1 cache'):
                fp.run(self.args)
        self.assertFalse((self.root/'experiments').exists())

    def test_existing_stage2_mode_is_still_routed_without_stage1(self):
        stage1=fp.run(self.args)
        before=len(self.calls)
        a=fp.parser().parse_args(['stage2','--manifest',str(stage1),
                                  '--accept-auto-exterior'])
        fp.run(a)
        self.assertEqual(len(self.calls),before+1)
        self.assertEqual(self.calls[-1],('stage2',str(stage1)))

    def test_known_stage2_v8_1_and_v8_2_fail_fast(self):
        root = Path(self.work.name)/'project_stub'
        f = root/'gsrefine/cleanup/local_region_pipeline.py'
        f.parent.mkdir(parents=True, exist_ok=True)
        sources = [
            "from gsrefine.cleanup.stage2_runtime import measured, timed\n"
            "def run():\n    from gsrefine.cleanup.stage2_runtime import measured\n"
            "    return measured('x')\n",
            "from gsrefine.cleanup.stage2_runtime import measured, timed\n"
            "def run():\n    with measured('x'):\n        pass\n"
        ]
        with patch.object(fp,'PROJECT_ROOT',root):
            for code in sources:
                f.write_text(code,encoding='utf-8')
                with self.assertRaisesRegex(RuntimeError,'v8.1|v8.2'):
                    fp.run(self.args)
                self.assertFalse((self.root/'experiments').exists())
            f.write_text("from gsrefine.cleanup.stage2_runtime import measured, timed\n"
                         "def run():\n    with timed('x'):\n        pass\n",encoding='utf-8')
            fp._preflight_known_stage2_errors()

    def test_existing_full_mode_is_still_routed(self):
        command=[]
        with patch.object(fp,'_run',lambda c:command.extend(c)):
            a=fp.parser().parse_args(['full','--config',str(self.cfg),
                '--scene','lfls','--method','gsplat_accel','--sfm-dir',str(self.sfm),
                '--gaussian-ply',str(self.gauss),'--output-dir',str(self.root),
                '--accept-auto-exterior'])
            fp.run(a)
        self.assertIn('full-pipeline',command)
        self.assertNotIn('--resume-manifest',command)

    def test_atomic_reservation_parallel_unique(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            slots=list(pool.map(lambda _:fp._reserve_repeat(self.root,self.args,self.roi),range(8)))
        self.assertEqual(len(slots),len(set(slots)))
        self.assertTrue(all(p.is_dir() for p in slots))


if __name__=='__main__':
    unittest.main(verbosity=2)
