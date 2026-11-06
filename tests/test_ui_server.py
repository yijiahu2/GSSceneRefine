"""Integration contract: browser annotations -> original COLMAP coordinates -> core."""
import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from tests.test_demo import build_synthetic_case
from gsrefine.web.server import create_app_state, make_handler, view_record_validate


def sample_annotation():
    return {
      'coordinate_system':'original_COLMAP_world',
      'polygon_units':'COLMAP_model_units_not_pixels',
      'views':[
        {'name':'top','origin':[0,0,0],'axes':[[1,0,0],[0,1,0]],'polygon':[[-1.6,-.8],[1,-.8],[1,.8],[-1.6,.8]]},
        {'name':'facade_0deg','origin':[0,0,0],'axes':[[1,0,0],[0,0,1]],'polygon':[[-1.6,-.2],[1,-.2],[1,.2],[-1.6,.2]]}
      ],
      'annotation_metadata':{'view_count':2},
    }


def test_annotation_geometry_contract():
    valid=sample_annotation()
    assert len(view_record_validate(valid).views)==2
    bad=json.loads(json.dumps(valid))
    bad['views'][1]['axes']=[[1,0,0],[0,1,0]] # front not vertical, duplicates top
    with pytest.raises(ValueError,match='upright orthogonal'):
        view_record_validate(bad)
    bad=json.loads(json.dumps(valid))
    bad['polygon_units']='pixels'
    with pytest.raises(ValueError,match='model units'):
        view_record_validate(bad)


def request_json(url, method='GET', data=None):
    payload=None if data is None else json.dumps(data).encode()
    r=urllib.request.Request(url,method=method,data=payload,
                             headers={'Content-Type':'application/json'} if payload else {})
    with urllib.request.urlopen(r,timeout=10) as f:return json.load(f)


def test_ui_http_and_core_handoff(tmp_path):
    build_synthetic_case(tmp_path)
    cfg=yaml.safe_load((tmp_path/'config.yaml').read_text())
    cfg_file=tmp_path/'config.yaml'
    state=create_app_state(cfg,tmp_path/'model.ply',tmp_path/'mesh.ply',tmp_path/'roi_work',
                           max_mesh_faces=100,sfm_npz=tmp_path/'sfm.npz',
                           config_file=cfg_file,preview_max=100)
    assert state['meta']['gaussians']>100
    assert state['meta']['display_gaussians']==100
    assert state['display_ply'].is_file()
    assert state['meta']['poisson_preview_mode']=='confidence_filtered_poisson'

    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(state,Path(__file__).resolve().parents[1]/'ui'))
    bg=threading.Thread(target=server.serve_forever,daemon=True);bg.start()
    url=f'http://127.0.0.1:{server.server_port}'
    try:
        info=request_json(url+'/api/meta')
        assert info['display_is_sampled'] and info['gaussians']>100
        overlay=request_json(url+'/api/poisson-preview')
        assert len(overlay['indices'])>0
        assert len(overlay['positions'])>0
        annotation=sample_annotation()
        saved=request_json(url+'/api/save-roi','POST',annotation)
        assert Path(saved['saved_path']).exists()
        job=request_json(url+'/api/run','POST',annotation)
        assert job['ok'] and job['status']=='running'
        # Keep the test deterministic; synthetic geometry is tiny.
        for attempt in range(120):
            status=request_json(url+'/api/run-status')
            if status['status']!='running':break
            time.sleep(.10)
        assert status['status']=='completed',status
        assert set(status['summary'])=={'hard','global','local'}
        assert status['summary']['local']['final_count']>=status['summary']['hard']['final_count']
        assert (Path(status['output_dir'])/'report.json').exists()
        req=urllib.request.Request(url+'/api/model.ply',headers={'Range':'bytes=0-2'})
        with urllib.request.urlopen(req) as f:
            assert f.status==206 and f.read()==b'ply'
    finally:
        server.shutdown();server.server_close();bg.join(timeout=3)
