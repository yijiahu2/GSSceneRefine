"""Module 2 must never silently assume a save succeeded or lose an error draft."""
from pathlib import Path
from gsrefine.roi_init.server import build_meta, make_handler
from gsrefine.roi_init.settings import InitSettings
from tests.test_roi_init import prepare, annotate
import json
import urllib.request
import threading
from http.server import ThreadingHTTPServer


def test_status_and_save_ack_on_disk(tmp_path):
    cfg=prepare(tmp_path)
    settings=InitSettings.load(cfg)
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(settings,Path(__file__).resolve().parents[1]/'ui',build_meta(settings)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(base+'/api/roi-status') as res:
            assert json.load(res)=={'ok':True,'exists':False}
        payload=json.dumps(annotate(settings)).encode('utf8')
        req=urllib.request.Request(base+'/api/confirm-roi',data=payload,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req) as res: out=json.load(res)
        assert out['ok'] and out['verified_on_disk']
        assert Path(out['saved_path']).is_file() and len(out['roi_sha256'])==64
        with urllib.request.urlopen(base+'/api/roi-status') as res: status=json.load(res)
        assert status['exists'] and status['saved_path']==out['saved_path']
        assert status['sha256']==out['roi_sha256']
    finally:
        server.shutdown();server.server_close();thread.join(timeout=5)


def test_frontend_save_only_clears_after_verified_ack():
    ui=Path(__file__).resolve().parents[1]/'ui'
    code=(ui/'app.mjs').read_text()
    assert 'answer.verified_on_disk!==true' in code
    assert code.index('clearAnnotationAfterSave();') > code.index('answer.verified_on_disk!==true')
    assert "showFeedback('error','ROI保存未成功'" in code
    html=(ui/'index.html').read_text()
    assert 'id="feedback-overlay"' in html and 'id="feedback-path"' in html
    assert 'id="top-poisson-layer"' in html and 'id="front-poisson-layer"' in html


def test_core_allows_isolated_output_folder_for_misclip(tmp_path):
    from gsrefine.roi.core import run
    from tests.test_demo import build_synthetic_case
    build_synthetic_case(tmp_path)
    experiment=tmp_path/'standalone_test'
    output=run(tmp_path/'config.yaml',modes=['hard'],output_dir=experiment)
    assert output['actual_output_dir']==str(experiment.resolve())
    assert (experiment/'report.json').is_file()
    assert not (tmp_path/'out/report.json').exists()
