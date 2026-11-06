"""Module 2 reproducibility and scene-wide ROI sharing tests (no Windows required)."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml
from gsrefine.roi_init.settings import InitSettings
from gsrefine.roi_init.results import (gaussian_roi_preview,save_confirmed_roi,
                                       current_canonical,require_orientation,read_manifest)
from gsrefine.roi_init.server import build_meta,make_handler
from tests.test_demo import build_synthetic_case
from tests.test_ui_server import sample_annotation


def prepare(tmp_path):
    build_synthetic_case(tmp_path)
    orig=yaml.safe_load((tmp_path/'config.yaml').read_text())
    orig['scene']={'name':'examplecliff','method':'absgs'}
    orig['source_model_scope']='aabb_baseline_only'
    orig['roi_init']={
      'gaussian_ply':str(tmp_path/'model.ply'),
      'output_dir':str(tmp_path/'all_roi'),
      'orientation':{'mode':'manual','manual_up_vector':[0,0,1],'manual_verified':True,
                     'frame_id':'examplecliff_colmap'},
      'web':{'port':8765},'file_service':{'port':8766},
      'reuse_scene_roi':True,'allow_new_revision':False}
    path=tmp_path/'module2.yaml'
    path.write_text(yaml.safe_dump(orig),encoding='utf8')
    return path


def annotate(settings):
    obj=sample_annotation()
    obj['annotation_metadata'].update({
      'up_vector':list(settings.up_vector),'frame_id':settings.frame_id,'orientation_verified':True,
      'orientation_source_sha256':settings.orientation_result.get('source_sha256'),
      'scene':settings.scene,'method':settings.method,
      'source_ply':str(settings.gaussian_ply)})
    return obj


def test_module2_orientation_is_not_guessed(tmp_path):
    path=prepare(tmp_path)
    cfg=yaml.safe_load(path.read_text())
    cfg['roi_init']['orientation']['mode']='auto'
    cfg['roi_init']['orientation']['up_axis']='+z'
    cfg['roi_init']['orientation']['verified']=True
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(FileNotFoundError,match='COLMAP images.bin'):
        InitSettings.load(path)


def test_module2_preview_save_and_method_reuse(tmp_path):
    cfg_file=prepare(tmp_path)
    settings=InitSettings.load(cfg_file)
    obj=annotate(settings)
    require_orientation(obj,settings)
    stats=gaussian_roi_preview(obj,settings)
    assert stats['gaussian_total']>0
    assert 0 < stats['gaussian_roi_count'] <= stats['gaussian_total']
    saved=save_confirmed_roi(obj,settings,stats)
    assert saved['saved_path'].endswith('examplecliff_absgs_roi_001.json')
    assert current_canonical(settings)==Path(saved['saved_path'])
    repeat=save_confirmed_roi(obj,settings,stats)
    assert repeat['saved_as_revision'] and repeat['saved_path'].endswith('_002.json')
    m=read_manifest(settings.out_dir)
    assert len(m['records'])==2
    # Exactly the same spatial ROI is reused across methods in the same scene.
    cfg=yaml.safe_load(cfg_file.read_text())
    cfg['scene']['method']='edepth'
    cfg_file.write_text(yaml.safe_dump(cfg))
    other=InitSettings.load(cfg_file)
    next_roi=annotate(other)
    again=save_confirmed_roi(next_roi,other)
    assert again['saved_as_revision']
    assert again['saved_path'].endswith('examplecliff_edepth_roi_001.json')
    assert current_canonical(other)==Path(saved['saved_path'])
    assert len(read_manifest(settings.out_dir)['records'])==3
    next_roi['views'][0]['polygon'][0][0]+=0.1
    newer=save_confirmed_roi(next_roi,other)
    assert newer['saved_path'].endswith('examplecliff_edepth_roi_002.json')


def test_module2_strict_frame_and_source(tmp_path):
    settings=InitSettings.load(prepare(tmp_path))
    x=annotate(settings)
    x['annotation_metadata']['up_vector']=[0,0,-1]
    with pytest.raises(ValueError,match='up_vector'):
        require_orientation(x,settings)
    x=annotate(settings)
    x['annotation_metadata']['frame_id']='another_coordinate_frame'
    with pytest.raises(ValueError,match='frame_id'):
        require_orientation(x,settings)
    x=annotate(settings)
    x['annotation_metadata']['source_ply']='/tmp/another.ply'
    with pytest.raises(ValueError,match='source PLY'):
        require_orientation(x,settings)
    x=annotate(settings)
    x['views'][0]['axes']=[[1,0,0],[0,-1,0]]
    with pytest.raises(ValueError,match='Top orthographic'):
        require_orientation(x,settings)


def test_module2_http_preview_and_save(tmp_path):
    config=prepare(tmp_path)
    settings=InitSettings.load(config)
    meta=build_meta(settings)
    assert meta['display_url'].startswith('http://127.0.0.1:8766/model.ply?')
    assert not meta['display_is_sampled']
    ui=Path(__file__).resolve().parents[1]/'ui'
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(settings,ui,meta))
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    base=f'http://127.0.0.1:{server.server_port}'
    def req(url,obj):
        request=urllib.request.Request(url,data=json.dumps(obj).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=8) as response:return json.load(response)
    try:
        with urllib.request.urlopen(base+'/api/meta',timeout=8) as response:
            assert json.load(response)['coordinate_frame']['verified']
        payload=annotate(settings)
        count=req(base+'/api/roi-preview',payload)
        assert count['preview']['gaussian_roi_count']>0
        saved=req(base+'/api/confirm-roi',payload)
        assert Path(saved['saved_path']).is_file()
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base+'/api/model.ply',timeout=8)
        assert e.value.code==410
    finally:
        server.shutdown();server.server_close();worker.join(timeout=3)
