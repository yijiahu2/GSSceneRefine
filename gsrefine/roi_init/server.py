"""GSSceneRefine Module 2: independent Windows-native model / WSL ROI metadata services.

The Windows file server exposes ONE configured Gaussian PLY, not the whole disk.
The browser performs 3DGS rendering on Windows WebGPU, never WSL llvmpipe.
The WSL side returns small JSON and saves a checked, reproducible ROI artifact.
"""
from __future__ import annotations
import argparse
import hashlib
import shutil
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import yaml
from gsrefine.roi.core import GaussianPLY
from gsrefine.roi_init.results import (current_canonical_for_source, current_latest_for_source,
                                       gaussian_roi_preview, require_orientation, save_confirmed_roi)
from gsrefine.roi_init.settings import InitSettings, PROJECT_ROOT
from gsrefine.roi_init.windows import start_native_server
from gsrefine.roi_init.workflow import WorkflowManager, workflow_options, monitor_terminal
from gsrefine.geometry.prepare import (prepare_scene, prepare_input_config,
                                      expand_colmap_inputs)
from gsrefine.roi_init.display_filter import PoissonDisplayFilter, validate_multiplier


def build_meta(settings):
    gp=GaussianPLY(settings.gaussian_ply)
    n=min(gp.vertex_count,100_000)
    x=gp.selected_xyz(np.linspace(0,gp.vertex_count-1,n,dtype=np.int64))
    bmin,bmax=np.quantile(x,[.01,.99],axis=0)
    center=(bmin+bmax)*.5
    diag=float(np.linalg.norm(bmax-bmin))
    if not np.isfinite(diag) or diag<=0:raise ValueError('Gaussian bounds are invalid')
    st=settings.gaussian_ply.stat()
    uri=f'http://127.0.0.1:{settings.file_port}/model.ply?v={st.st_size}-{st.st_mtime_ns}'
    existing=current_canonical_for_source(settings)
    saved=None
    if existing:
        old=json.loads(existing.read_text(encoding='utf8'))
        oldmeta=old.get('annotation_metadata',{})
        oldup=np.asarray(oldmeta.get('up_vector',[]),dtype=float)
        if (oldmeta.get('frame_id') != settings.frame_id or
            oldmeta.get('orientation_source_sha256') != settings.orientation_result.get('source_sha256') or
            oldup.shape!=(3,) or
            not np.allclose(oldup,settings.up_vector,rtol=0.,atol=1e-6)):
            raise ValueError('Saved canonical ROI coordinate frame disagrees with current config; do not reuse without coordinate registration')
        saved={'path':str(existing),'views':old.get('views',[]),'annotation_metadata':old.get('annotation_metadata',{})}
    latest_file=current_latest_for_source(settings)
    latest_saved=None
    if latest_file:
        latest_data=json.loads(latest_file.read_text(encoding='utf8'))
        latest_saved={'path':str(latest_file),'views':latest_data.get('views',[]),
                      'annotation_metadata':latest_data.get('annotation_metadata',{})}
    return {
        'project':'GSSceneRefine','ui_version':'module2-orientation-auto-0.5',
        'scene':settings.scene,'method':settings.method,'gaussians':gp.vertex_count,
        'gaussian_name':settings.gaussian_ply.name,
        'source_ply':str(settings.gaussian_ply),'source_ply_bytes':st.st_size,
        'display_format':'PLY','display_gaussians':gp.vertex_count,
        'display_is_sampled':False,'display_lossy':False,'display_url':uri,
        'display_bytes':st.st_size,'source_model_scope':settings.model_scope,
        'model_scope':settings.model_scope,'model_original_coordinates':True,
        'reference_center':center.tolist(),
        'gaussian_robust_bounds':{'min':bmin.tolist(),'max':bmax.tolist()},
        'reference_bounds':{'min':bmin.tolist(),'max':bmax.tolist()},
        'scene_diagonal':diag,'poisson_loaded':settings.poisson_ply is not None,
        'poisson_preview_mode':'lazy_raw_unverified',
        'coordinate_frame':{'up_vector':list(settings.up_vector), 'orientation_method':settings.orientation_result['method'],
                            'orientation_source_sha256':settings.orientation_result.get('source_sha256'),
                            'orientation_diagnostics':{k:v for k,v in settings.orientation_result.items()
                                if k in ('camera_count','median_camera_right_off_horizontal_deg','p90_camera_right_off_horizontal_deg',
                                         'heading_diversity_eigenvalue','median_downward_optical_projection',
                                         'downward_looking_camera_fraction','confidence_label','warnings')},
                            'frame_id':settings.frame_id,'verified':settings.orientation_verified,
                            'coordinate_system':'original_COLMAP_world'},
        'roi_output_dir':str(settings.out_dir),'canonical_roi':saved,'latest_roi':latest_saved,
        'module1': yaml.safe_load(settings.config.read_text(encoding='utf8')).get('module1_runtime',
                    {'status': 'reused_existing', 'mode': 'reuse', 'cached': True,
                     'warnings': ['Existing SfM/Poisson references are reused; their original production settings are not reconstructed.']}),
        'workflow':{'auto_after_confirm':workflow_options(settings.config)['enabled'],
                    'method':'R3','status_endpoint':'/api/workflow-status'},
        'max_elevation_views':settings.max_elevations,
        'warnings':['Initial ROI preview uses full-Gaussian center intersections, NOT ellipsoid coverage.',
                    'Poisson structure recovery is deferred to the following geometry module.']
    }


def make_handler(settings, ui_root:Path, meta, workflow=None, *, handoff_shutdown=False):
    workflow = workflow if workflow is not None else WorkflowManager(getattr(settings, "config", None), settings=settings)
    preview_filter = PoissonDisplayFilter(settings)
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw):super().__init__(*a,directory=str(ui_root),**kw)
        def log_message(self,fmt,*args):
            # Successful browser polling and static assets are not processing progress.
            # Keep actual browser errors visible for troubleshooting.
            if self.command == 'GET' and len(args) >= 2:
                try:
                    if 200 <= int(args[1]) < 400:
                        return
                except (TypeError, ValueError):
                    pass
            print('[roi-ui]',fmt%args,flush=True)
        def end_headers(self):
            # Module updates can change /api/meta from up_axis to up_vector.
            # Never let a stale HTML/ES module survive this protocol change.
            # Large PlayCanvas vendor modules and the Windows PLY service retain
            # their independent caching rules; only GSSceneRefine app code is uncached.
            if urlparse(self.path).path in ('/', '/index.html', '/app.mjs', '/geometry.mjs', '/poisson-overlay.mjs'):
                self.send_header('Cache-Control', 'no-store, max-age=0')
                self.send_header('Pragma', 'no-cache')
            super().end_headers()
        def send_json(self,obj,code=200):
            b=json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf8')
            self.send_response(code)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(b)))
            self.end_headers();self.wfile.write(b)
        def send_preview_ply(self, path):
            size = path.stat().st_size
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(size))
            self.send_header('Cache-Control', 'private, max-age=3600')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            with path.open('rb') as source:
                try:
                    shutil.copyfileobj(source, self.wfile, length=1024*1024)
                except (ConnectionError, BrokenPipeError):
                    pass

        def do_GET(self):
            route=urlparse(self.path).path
            if route in ('/api/display-filter', '/api/display-filter.ply'):
                if not settings.poisson_ply:
                    return self.send_json({'ok':False, 'error':'Poisson is not available for display filtering'}, 404)
                try:
                    raw=parse_qs(urlparse(self.path).query).get('multiplier',['1.00'])
                    if len(raw)!=1:raise ValueError('Exactly one multiplier is required')
                    m=validate_multiplier(raw[0])
                    if route=='/api/display-filter':
                        # First request builds/reuses the distance cache and display-only subset.
                        result=preview_filter.prepare_preview(m)
                        return self.send_json(result)
                    path=preview_filter.get_existing_preview(m)
                    if path is None:
                        return self.send_json({'ok':False,'error':'Display preview not prepared'},404)
                    return self.send_preview_ply(path)
                except (ValueError, RuntimeError, OSError, ImportError) as exc:
                    return self.send_json({'ok':False,'error':str(exc)},400)
            if route=='/api/meta':return self.send_json(meta)
            if route=='/api/workflow-status':return self.send_json(workflow.status())
            if route=='/api/roi-status':
                try:
                    confirmed=current_latest_for_source(settings)
                    reference=current_canonical_for_source(settings)
                    if confirmed is None:return self.send_json({'ok':True,'exists':False})
                    roi=json.loads(confirmed.read_text(encoding='utf8'))
                    annotation=roi.get('annotation_metadata',{})
                    return self.send_json({'ok':True,'exists':True,'saved_path':str(confirmed),
                        'latest_path':str(confirmed),
                        'canonical_path':str(reference) if reference else None,
                        'gaussian_roi_count':annotation.get('initial_gaussian_roi_count'),
                        'confirmed_utc':annotation.get('confirmed_utc'),
                        'sha256':hashlib.sha256(confirmed.read_bytes()).hexdigest()})
                except (ValueError,OSError,KeyError,RuntimeError) as exc:
                    return self.send_json({'ok':False,'error':str(exc)},500)
            if route=='/api/poisson-preview':
                if not settings.poisson_ply:return self.send_json({'error':'No Poisson overlay configured'},404)
                # Lazy loaded: no SfM-confidence cost is paid before user asks for overlay.
                from gsrefine.web.server import mesh_metadata
                _,preview,mode=mesh_metadata(settings.poisson_ply,preview_faces=18000)
                return self.send_json({**preview,'mode':mode})
            if route in ('/api/model.ply','/api/model.sog'):
                return self.send_json({'error':'Model served from the Windows-native allowlisted file service'},410)
            if route in ('/', '/index.html', '/app.mjs', '/geometry.mjs', '/poisson-overlay.mjs'):
                # Disable conditional 304 responses for first-party app modules.
                # This prevents an older up_axis-dependent app.js from being
                # served alongside the new up_vector /api/meta contract.
                if 'If-Modified-Since' in self.headers:
                    del self.headers['If-Modified-Since']
            if route in ('/','/index.html'):self.path='/index.html'
            return super().do_GET()
        def do_POST(self):
            route=urlparse(self.path).path
            if route not in ('/api/roi-preview','/api/confirm-roi'):
                return self.send_json({'error':'Unknown route'},404)
            origin=self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{settings.ui_port}',f'http://localhost:{settings.ui_port}'):
                return self.send_json({'error':'Invalid browser origin'},403)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=1_000_000:raise ValueError('ROI JSON must be at most 1MB')
                payload=json.loads(self.rfile.read(size))
                if not isinstance(payload,dict):raise ValueError('ROI request must be a JSON object')
                save_mode=payload.pop('save_mode','standard')
                if save_mode not in ('standard','new_revision'):
                    raise ValueError('Unknown save_mode: expected standard or new_revision')
                if route=='/api/roi-preview' and save_mode!='standard':
                    raise ValueError('Preview must not request a save mode')
                require_orientation(payload,settings)
                preview=gaussian_roi_preview(payload,settings)
                if route=='/api/roi-preview':
                    return self.send_json({'ok':True,'preview':preview})
                # Reserve confirmation under one lock: never create an additional
                # numbered ROI while a previous final-export job is still active.
                with workflow.confirm_lock:
                    if workflow.enabled and workflow.is_running():
                        return self.send_json({'ok':False, 'error':
                            'Automatic R3 export is still running. Finish it before confirming another ROI.'},409)
                    result=save_confirmed_roi(payload,settings,preview,
                                              new_revision=(save_mode=='new_revision'))
                    target=Path(result['saved_path'])
                    if not target.is_file():raise OSError(f'ROI save reported but file not found: {target}')
                    checksum=hashlib.sha256(target.read_bytes()).hexdigest()
                    print(f'[roi-init] ROI confirmed: {target} (sha256={checksum}, reused={result["reused"]})',flush=True)
                    response={'ok':True,**result,'verified_on_disk':True,'roi_sha256':checksum}
                    if workflow.enabled:
                        try:
                            response['workflow']=workflow.launch(target)
                        except (OSError,ValueError,RuntimeError) as exc:
                            # ROI is saved and must never be described as unsaved.
                            response['workflow']={'enabled':True, 'status':'failed',
                                'error':f'ROI was saved but automatic processing could not start: {exc}'}
                    # Finish the HTTP response before stopping the UI server. The
                    # already-launched roi-auto worker is a separate process and
                    # remains active; the WSL terminal becomes the progress UI.
                    self.send_json(response)
                    if (handoff_shutdown and workflow.enabled and
                            response.get('workflow', {}).get('status') == 'running'):
                        self.wfile.flush()
                        threading.Thread(target=self.server.shutdown,
                                         name='roi-ui-handoff', daemon=True).start()
                    return
            except (ValueError,TypeError,KeyError,IndexError,json.JSONDecodeError) as e:
                return self.send_json({'ok':False,'error':str(e)},400)
            except OSError as e:
                return self.send_json({'ok':False,'error':str(e)},500)
    return Handler


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/songziguanyin.yaml')
    parser.add_argument('--no-auto-windows',action='store_true',help='For debug: manually start Windows service')
    parser.add_argument('--scene', help='Scene ID; do not mix different COLMAP reconstructions')
    parser.add_argument('--method', help='Gaussian reconstruction method label, e.g. absgs or vanilla')
    parser.add_argument('--colmap-points',help='Original SfM points3D.bin; may be reused from geometry cache')
    parser.add_argument('--gaussian-ply',help='Original complete 3DGS Gaussian PLY')
    parser.add_argument('--colmap-images',help='COLMAP images.bin for this coordinate frame (optional if configured)')
    parser.add_argument('--sfm-dir',help='COLMAP sparse directory containing points3D.bin and images.bin')
    parser.add_argument('--output-dir',required=True,help='All run outputs: geometry, ROI, result, logs and temporary files')
    parser.add_argument('--keep-diagnostics',action='store_true',help='Keep all research/debug masks, PLYs and tables')
    args=parser.parse_args(argv)
    points, images = expand_colmap_inputs(args.sfm_dir, colmap_points=args.colmap_points,
                                          colmap_images=args.colmap_images)
    dynamic = any((args.scene, args.method, args.colmap_points,
                   args.gaussian_ply, args.colmap_images, args.sfm_dir, args.output_dir))
    initial_config=(prepare_input_config(args.config, scene=args.scene, method=args.method,
                                        colmap_points=points,
                                        gaussian_ply=args.gaussian_ply,
                                        colmap_images=images,
                                        final_output_root=args.output_dir)
                    if dynamic else args.config)
    print('[pipeline] [1/4] 模块一：准备SfM净化与Poisson曲面',flush=True)
    effective_config=prepare_scene(initial_config)
    settings=InitSettings.load(effective_config)
    runtime=yaml.safe_load(effective_config.read_text(encoding='utf8')) or {}
    module1=runtime.get('module1_runtime', {}) or {}
    cache_text='（复用已有缓存）' if module1.get('cached') else ''
    print(f'[pipeline] [1/4] 模块一完成{cache_text}',flush=True)
    print('[roi-init] Module 1 effective config:', effective_config, flush=True)
    ui_root=PROJECT_ROOT/'ui'
    if not (ui_root/'node_modules/playcanvas/build/playcanvas.mjs').exists():
        raise RuntimeError('Missing PlayCanvas. Run: cd /home/xth/GSSceneRefine/ui && npm install')
    print('[roi-init] COLMAP automatic up:',settings.up_vector,flush=True)
    print('[roi-init] Orientation evidence:',settings.orientation_result['method'],
          settings.orientation_result.get('warnings',[]),flush=True)
    meta=build_meta(settings)
    windows_proc=None
    if not args.no_auto_windows:
        windows_proc,log=start_native_server(settings)
        print('[roi-init] Windows model service log:',log,flush=True)
    else:
        print('[roi-init] Windows service must be started independently on port',settings.file_port,flush=True)
    workflow=WorkflowManager(settings.config,settings=settings,
                             keep_diagnostics=args.keep_diagnostics)
    # The browser is only an annotation surface. After ROI confirmation its
    # HTTP server is handed off to the WSL terminal and permanently closed.
    meta['workflow']['terminal_handoff']=workflow.enabled
    # Only the real interactive entrypoint exits the browser after confirm.
    # Unit tests and external callers using make_handler keep their old contract.
    server=ThreadingHTTPServer((settings.ui_host,settings.ui_port),
                               make_handler(settings,ui_root,meta,workflow=workflow,
                                            handoff_shutdown=workflow.enabled))
    print(f'[roi-init] Windows Edge workspace: http://127.0.0.1:{settings.ui_port}',flush=True)
    print('[roi-init] Direct model URL:',meta['display_url'],flush=True)
    print('[roi-init] Save folder:',settings.out_dir,flush=True)
    print('[roi-init] Final ROI result root:',workflow_options(settings.config)['output_root'],flush=True)
    print('[roi-init] Same scene ROI can be reused across other 3DGS methods.',flush=True)
    if workflow_options(settings.config)['enabled']:
        print('[roi-init] One-click workflow ENABLED: confirm ROI → R3 recovery → validated PLY.',flush=True)
    print('[pipeline] [2/4] 模块二：等待Edge双正交ROI确认',flush=True)
    interrupted=False
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        interrupted=True
        print('[pipeline] 模块二已手动停止。',flush=True)
    finally:
        server.server_close()
        # Model display is not needed after Module 2; do not terminate the
        # separate WSL roi-auto worker when closing the browser service.
        if windows_proc and windows_proc.poll() is None:
            windows_proc.terminate()
            try:windows_proc.wait(timeout=4)
            except Exception:windows_proc.kill()
    # Module-2 preview is temporary and not required by later stages.
    shutil.rmtree(settings.out_dir / '.poisson_display_cache', ignore_errors=True)
    completed_status = None
    if workflow.launched_job() and not interrupted:
        print('[pipeline] [2/4] ROI已保存；网页服务已关闭，后续在终端显示。',flush=True)
        completed_status = monitor_terminal(workflow)
        if completed_status['status'] != 'completed':
            raise SystemExit(1)

    return completed_status

if __name__=='__main__':main()
