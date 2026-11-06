#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local, read-only PlayCanvas UI server for GSSceneRefine GeoGuard-ROI annotations.

Never writes to input PLY/Poisson/SfM. Does not upload model files to a remote service.
The browser loads an optional cached SOG or original Gaussian PLY from localhost. Drawing outputs are downloaded
by the user or saved via the POST /api/save-roi endpoint to the configured output folder.
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import trimesh
import yaml

from gsrefine.roi.core import GaussianPLY, ROI, load_sfm, geometry_confidence
from gsrefine.web.preview_cache import file_signature, find_sog_cache, build_sog_cache


def sample_gaussian_bounds(ply: GaussianPLY, limit=100000):
    n=ply.vertex_count
    ids=np.linspace(0,max(n-1,0),min(n,limit),dtype=np.int64)
    pts=ply.selected_xyz(ids)
    q1,q99=np.quantile(pts,[0.01,0.99],axis=0)
    return {'min':q1.tolist(),'max':q99.tolist(),'sample_count':int(len(ids))}


def mesh_metadata(path, preview_faces=30000, sfm_path=None, sfm_cfg=None):
    if not path or not path.is_file():return None, None, 'none'
    mesh=trimesh.load(str(path),process=False)
    if not isinstance(mesh,trimesh.Trimesh) or len(mesh.faces)==0:
        raise ValueError('Poisson preview must be a triangle mesh')
    bounds={'min':np.asarray(mesh.bounds[0]).tolist(),'max':np.asarray(mesh.bounds[1]).tolist(),
            'faces':int(len(mesh.faces)), 'vertices':int(len(mesh.vertices))}
    mode='unverified_raw_poisson'
    selected=np.arange(len(mesh.faces))
    if sfm_path and Path(sfm_path).is_file():
        sfm,err,track,_=load_sfm(sfm_path)
        conf,_,_,_,_=geometry_confidence(mesh,sfm,err,track,
            k=int((sfm_cfg or {}).get('knn_spacing',6)),
            weights=tuple((sfm_cfg or {}).get('confidence_weights',[.35,.4,.25])))
        threshold=float((sfm_cfg or {}).get('confidence_threshold',.75))
        selected=np.flatnonzero(conf>=threshold)
        mode='confidence_filtered_poisson'
        if not len(selected):
            selected=np.arange(len(mesh.faces))
            mode='unverified_raw_poisson_no_credible_faces'
    ix=selected[np.linspace(0,len(selected)-1,min(preview_faces,len(selected)),dtype=np.int64)]
    face_subset=np.asarray(mesh.faces[ix],dtype=np.int32)
    unique,inv=np.unique(face_subset.reshape(-1),return_inverse=True)
    # Local raw COLMAP-world vertex coordinates. This is a visualization-only subset.
    xyz=np.asarray(mesh.vertices[unique],dtype=np.float32)
    preview={'positions':xyz.reshape(-1).round(6).tolist(),
             'indices':inv.astype(np.int32).tolist(),
             'source_faces':int(len(mesh.faces)), 'preview_faces':int(len(face_subset))}
    return bounds,preview,mode


def view_record_validate(payload):
    """Validate against the geometry-core parser before anything is saved."""
    if not isinstance(payload,dict) or not isinstance(payload.get('views'),list):
        raise ValueError('ROI JSON requires a views array')
    views=payload['views']
    names=[v.get('name') for v in views if isinstance(v,dict)]
    if len(views)<2 or len(views)>5 or len(names)!=len(set(names)):
        raise ValueError('Require 1 top + 1–4 front views with unique names')
    if sum(1 for x in names if x=='top')!=1:
        raise ValueError('Exactly one top view required')
    if not all(isinstance(x,dict) for x in views):raise ValueError('Invalid view')
    # Normalize via existing parser: validates orthonormal axes and polygon shape.
    from gsrefine.roi.core import Projection
    parsed=ROI([Projection.from_json(x) for x in views])
    # Orthographic top and facades must share one vertical axis. A mismatched
    # coordinate frame can silently corrupt millions of Gaussian assignments.
    tv=next(x for x in parsed.views if x.name=='top')
    up=np.cross(tv.U[0],tv.U[1]); up/=np.linalg.norm(up)
    for view in parsed.views:
        if view.name=='top':continue
        facade_normal=np.cross(view.U[0],view.U[1])
        if abs(float(np.dot(up,facade_normal)))>1e-5:
            raise ValueError(f'{view.name} is not an upright orthogonal facade (front/up axes mismatch)')
    # Data origin should match the original model coordinates.
    if payload.get('coordinate_system') not in ('original_COLMAP_world',):
        raise ValueError('Unknown coordinate system: use original_COLMAP_world')
    if payload.get('polygon_units') != 'COLMAP_model_units_not_pixels':
        raise ValueError('ROI vertices must be in original model units, not pixels')
    return parsed


def create_app_state(cfg, plyfile, poisson, output_dir, max_mesh_faces, sfm_npz=None,
                     config_file=None, preview_max=0, display_sog=None, raw_ply=False):
    gp=GaussianPLY(plyfile)
    pbounds,preview,preview_mode=mesh_metadata(poisson,max_mesh_faces,sfm_npz,cfg.get('sfm',{}))
    g_bounds=sample_gaussian_bounds(gp)
    # Display-only source selection. Full-precision PLY remains the authoritative
    # source of all geometry calculations. Cached SOG does not change Gaussian data.
    display_ply=Path(plyfile)
    display_count=gp.vertex_count
    display_kind='ply'
    if display_sog and raw_ply:
        raise ValueError('--display-sog cannot be combined with --raw-ply')
    if display_sog and preview_max:
        raise ValueError('--display-sog cannot be combined with --preview-max-gaussians')
    if display_sog:
        import zipfile
        display_ply=Path(display_sog).expanduser().resolve()
        if not display_ply.is_file() or not zipfile.is_zipfile(display_ply):
            raise ValueError(f'--display-sog is not a valid bundled SOG file: {display_ply}')
        with zipfile.ZipFile(display_ply) as bundle:
            if 'meta.json' not in bundle.namelist():
                raise ValueError(f'SOG archive has no meta.json: {display_ply}')
        display_kind='sog'
    elif preview_max and gp.vertex_count>preview_max:
        # Explicit fallback sampling, never the default for a large Gaussian model.
        preview_dir=Path(output_dir)/'_display_cache'
        preview_dir.mkdir(parents=True,exist_ok=True)
        digest=file_signature(plyfile)
        display_ply=preview_dir/f'preview_{digest}_{int(preview_max)}.ply'
        if not display_ply.is_file():
            ids=np.linspace(0,gp.vertex_count-1,int(preview_max),dtype=np.int64)
            mask=np.zeros(gp.vertex_count,dtype=bool);mask[ids]=True
            gp.export(mask,display_ply)
        display_count=int(preview_max)
    elif not raw_ply:
        cached=find_sog_cache(Path(plyfile),Path(output_dir))
        if cached is not None:
            display_ply=cached
            display_kind='sog'
    visual_bytes=display_ply.stat().st_size
    display_token=file_signature(display_ply)
    display_url=f'/api/model.{display_kind}?v={display_token}'
    b=pbounds or g_bounds
    center=(np.asarray(b['min'])+np.asarray(b['max']))/2
    extent=np.asarray(b['max'])-np.asarray(b['min'])
    diag=float(np.linalg.norm(extent))
    if diag<=0 or not np.isfinite(diag):raise ValueError('Invalid model bounds')
    return {
      'ply':Path(plyfile),'poisson':Path(poisson) if poisson else None,
      'output_dir':Path(output_dir), 'mesh_preview':preview,
      'display_ply':display_ply, 'display_kind':display_kind, 'config_file':Path(config_file) if config_file else None,
      'active_run':None,'run_lock':threading.Lock(),
      'meta':{'project':'GeoGuard-ROI','ui_version':'0.2.0',
              'gaussians':gp.vertex_count,'gaussian_name':Path(plyfile).name,
              'display_gaussians':display_count, 'display_is_sampled':display_count<gp.vertex_count,
              'display_format':display_kind.upper(), 'display_url':display_url,
              'display_bytes':visual_bytes,'source_ply_bytes':Path(plyfile).stat().st_size,
              'display_lossy':display_kind=='sog',
              'poisson_loaded':pbounds is not None,'poisson_bounds':pbounds,'poisson_preview_mode':preview_mode,
              'gaussian_robust_bounds':g_bounds, 'reference_bounds':b,
              'reference_center':center.tolist(), 'scene_diagonal':diag,
              'model_scope':cfg.get('source_model_scope','unknown'),
              'model_original_coordinates':True,
              'orthographic':True,'unit':'COLMAP model unit',
              'warnings':['Candidate Poisson overlay is geometric reference only; it may omit real target details.',
                          'Scene displays only; browser clipping does not modify the full input PLY.']}}


def make_handler(state,root):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(root),**kwargs)
        def log_message(self,format,*args):print('[GeoGuard-UI]',format%args)
        def json_response(self,payload,status=200):
            b=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-cache')
            self.end_headers();self.wfile.write(b)
        def do_GET(self):
            path=urlparse(self.path).path
            if path=='/api/meta':return self.json_response(state['meta'])
            if path=='/api/run-status':return self.run_status()
            if path=='/api/poisson-preview':
                if state['mesh_preview'] is None:return self.json_response({'error':'No Poisson reference loaded'},404)
                return self.json_response(state['mesh_preview'])
            if path in ('/api/model.ply','/api/model.sog'):
                if path!=f"/api/model.{state['display_kind']}":
                    return self.json_response({'error':'Requested display format is not active'},404)
                return self.stream_model()
            if path in ('/','/index.html'):self.path='/index.html'
            # Never allow directory traversal to expose source dataset or output folder.
            return super().do_GET()
        def stream_model(self):
            p=state['display_ply'];size=p.stat().st_size
            etag='"'+file_signature(p)+'"'
            byte_range=self.headers.get('Range')
            cache_headers={'ETag':etag,'Cache-Control':'private, max-age=86400'}
            if self.headers.get('If-None-Match') == etag and not byte_range:
                self.send_response(304)
                for key,value in cache_headers.items():self.send_header(key,value)
                self.end_headers()
                return
            start,end=0,size-1
            if byte_range:
                m=re.fullmatch(r'bytes=(\d+)-(\d*)',byte_range.strip())
                if not m:return self.send_error(416,'Unsupported Range')
                start=int(m.group(1));end=min(int(m.group(2)),size-1) if m.group(2) else end
                if start> end or start>=size:return self.send_error(416,'Invalid range')
            self.send_response(206 if byte_range else 200)
            self.send_header('Content-Type','application/octet-stream')
            self.send_header('Content-Length',str(end-start+1))
            self.send_header('Accept-Ranges','bytes')
            for key,value in cache_headers.items():self.send_header(key,value)
            if byte_range:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            with p.open('rb') as f:
                f.seek(start);remaining=end-start+1
                try:
                    while remaining:
                        buf=f.read(min(1048576,remaining))
                        if not buf:break
                        self.wfile.write(buf);remaining-=len(buf)
                except (BrokenPipeError,ConnectionResetError):pass
        def save_roi_object(self,obj):
            view_record_validate(obj)
            out=state['output_dir'];out.mkdir(parents=True,exist_ok=True)
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            dst=out/f'roi_gui_{stamp}.json'
            dst.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            return dst,stamp

        def run_status(self):
            with state['run_lock']:
                r=state.get('active_run')
                if not r:return self.json_response({'status':'idle'})
                p=r['process']
                rc=p.poll()
                status='running' if rc is None else ('completed' if rc==0 else 'failed')
                result={'status':status,'run_id':r['run_id'],
                        'output_dir':str(r['output_dir']),'log':str(r['log']),
                        'return_code':rc}
                report=r['output_dir']/'report.json'
                if rc==0 and report.is_file():
                    data=json.loads(report.read_text(encoding='utf-8'))
                    result['summary']={mode:{k:data[mode].get(k) for k in ('base_count','added_count','final_count','removed_count')}
                                       for mode in data.get('modes',[]) if mode in data}
                return self.json_response(result)

        def start_run(self,obj):
            if state['config_file'] is None:
                raise ValueError('Need --config to run geometry analysis')
            with state['run_lock']:
                active=state.get('active_run')
                if active and active['process'].poll() is None:
                    return self.json_response({'ok':False,'error':'Another geometry task is still running'},409)
                roi_file,run_id=self.save_roi_object(obj)
                run_root=state['output_dir']/'geometry_runs'/run_id
                run_root.mkdir(parents=True,exist_ok=False)
                config=yaml.safe_load(state['config_file'].read_text(encoding='utf-8'))
                config.setdefault('paths',{})['gaussian_ply']=str(state['ply'])
                config['paths']['roi_json']=str(roi_file)
                config['paths']['output_dir']=str(run_root/'results')
                config['export_ply']=False  # Analysis button must not produce huge files silently.
                config_path=run_root/'config_run.yaml'
                config_path.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
                logfile=run_root/'geometry.log'
                with logfile.open('w',encoding='utf-8') as f:
                    process=subprocess.Popen([sys.executable,'-m','gsrefine.roi.core',
                        '--config',str(config_path),'--roi',str(roi_file),'--modes','hard,global,local'],
                        stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                        cwd=str(Path(__file__).resolve().parents[2]))
                state['active_run']={'process':process,'run_id':run_id,'output_dir':run_root/'results','log':logfile}
            return self.json_response({'ok':True,'status':'running','run_id':run_id,
                                       'saved_roi':str(roi_file), 'output_dir':str(run_root/'results')},202)

        def do_POST(self):
            route=urlparse(self.path).path
            if route not in ('/api/save-roi','/api/run'):return self.send_error(404)
            try:
                n=int(self.headers.get('Content-Length','0'))
                if n<=0 or n>1024*1024:raise ValueError('ROI JSON must be 1 byte–1 MiB')
                obj=json.loads(self.rfile.read(n))
                view_record_validate(obj)
                if route=='/api/run':return self.start_run(obj)
                dst,stamp=self.save_roi_object(obj)
                return self.json_response({'ok':True,'saved_path':str(dst),'views':len(obj['views'])})
            except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
                return self.json_response({'ok':False,'error':str(exc)},400)
            except OSError as exc:
                return self.json_response({'ok':False,'error':str(exc)},500)
    return Handler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/songziguanyin.yaml')
    p.add_argument('--gaussian-ply',default=None)
    p.add_argument('--poisson-ply',default=None)
    p.add_argument('--sfm-npz',default=None)
    p.add_argument('--source-model-scope',default=None,choices=['full_original','aabb_baseline_only'],
                   help='Declare whether the supplied Gaussian PLY contains all original Gaussians')
    p.add_argument('--output-dir',default=None)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--max-mesh-faces',type=int,default=25000)
    p.add_argument('--display-sog',default=None,help='Optional prebuilt full-resolution .sog visual cache')
    p.add_argument('--build-display-cache',action='store_true',help='Create/reuse full-model SOG cache before starting local UI; one-time conversion may take time')
    p.add_argument('--cache-device',default='auto',help='SOG compression device: auto, cpu, or GPU adapter number')
    p.add_argument('--raw-ply',action='store_true',help='Force original uncompressed PLY transfer (debug/compare)')
    p.add_argument('--preview-max-gaussians',type=int,default=0,
                   help='0=full model; e.g. 1500000=deterministic display-only downsample')
    args=p.parse_args()
    config_file=Path(args.config).resolve()
    cfg=yaml.safe_load(config_file.read_text(encoding='utf-8'))
    if args.source_model_scope:cfg['source_model_scope']=args.source_model_scope
    paths=cfg.get('paths',{})
    model=Path(args.gaussian_ply or paths['gaussian_ply']).expanduser()
    mesh=Path(args.poisson_ply or paths.get('poisson_raw','')).expanduser() if (args.poisson_ply or paths.get('poisson_raw')) else None
    out=Path(args.output_dir or paths.get('output_dir','./output')).expanduser()
    if not model.is_file():raise FileNotFoundError('Gaussian PLY not found: '+str(model))
    if args.build_display_cache:
        if args.raw_ply or args.display_sog or args.preview_max_gaussians:
            p.error('--build-display-cache cannot be combined with --raw-ply, --display-sog, or --preview-max-gaussians')
        print('[startup] Ensuring full-resolution SOG display cache exists; original PLY is unchanged.', flush=True)
        cache_output = out if out.is_absolute() else Path(__file__).resolve().parents[2] / out
        build_sog_cache(model,cache_output,Path(__file__).resolve().parents[2],device=args.cache_device)
    if mesh and not mesh.is_file():raise FileNotFoundError('Poisson PLY not found: '+str(mesh))
    root=Path(__file__).resolve().parents[2]/'ui'
    if not (root/'node_modules/playcanvas/build/playcanvas.mjs').is_file():
        raise RuntimeError(f'PlayCanvas missing. Run: cd "{root}" && npm install')
    sfm_npz=Path(args.sfm_npz or paths.get('clean_sfm_npz','')).expanduser() if (args.sfm_npz or paths.get('clean_sfm_npz')) else None
    state=create_app_state(cfg,model,mesh,out,args.max_mesh_faces,sfm_npz,config_file,
                           max(0,args.preview_max_gaussians), args.display_sog, args.raw_ply)
    server=ThreadingHTTPServer((args.host,args.port),make_handler(state,root))
    print(f'GSSceneRefine GeoGuard-ROI workspace: http://{args.host}:{args.port}')
    print('Model:',model)
    print('Browser display:',state['display_ply'],'(',state['meta']['display_gaussians'],'Gaussians,',state['display_kind'].upper(),')')
    if state['display_kind']=='ply' and not args.preview_max_gaussians:
        print('TIP: For faster repeated full-scene loads run: python run.py preview-cache --config',str(config_file))
    print('Poisson:',mesh if mesh else 'not supplied')
    print('Poisson preview:',state['meta']['poisson_preview_mode'])
    print('Saved ROI folder:',out)
    print('Do not expose to public networks; model files are served to the browser locally.')
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
