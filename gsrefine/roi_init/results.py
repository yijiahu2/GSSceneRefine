"""Atomic, comparable, scene-aware ROI artifacts. No edits to source PLY files."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from gsrefine.roi.core import GaussianPLY
from gsrefine.web.server import view_record_validate

_IO_LOCK = threading.Lock()


def canonical_views_sha(payload):
    normalized = [{k: v[k] for k in ('name', 'origin', 'axes', 'polygon')} for v in payload['views']]
    return hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write_atomic(path: Path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + path.name + '.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf8') as out:
            json.dump(data,out,indent=2,ensure_ascii=False,allow_nan=False)
            out.write('\n');out.flush();os.fsync(out.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)


def read_manifest(directory: Path):
    p = directory/'manifest.json'
    if not p.exists():return {'schema_version':1,'scene_canonical':{},'scene_latest':{},'records':[]}
    d = json.loads(p.read_text(encoding='utf8'))
    if d.get('schema_version')!=1:raise ValueError('Unsupported ROI manifest schema')
    return d


def require_orientation(payload, settings, *, reuse_scene_geometry=False):
    """A saved ROI must use the SAME ORIGINAL COLMAP world-up, not a label.

    The camera/gaussian/Poisson models must not be rotated for display. For
    scene-wide reuse, other 3DGS methods may use this ROI only when they share
    the same COLMAP reconstruction and up-direction reference.
    """
    frame=payload.get('annotation_metadata',{})
    if frame.get('frame_id') != settings.frame_id:
        raise ValueError('ROI COLMAP frame_id is inconsistent with configured scene registration')
    if frame.get('orientation_verified') is not True:
        raise ValueError('Unvalidated ROI orientation cannot be saved')
    up=np.asarray(frame.get('up_vector',[]),float)
    expected=np.asarray(settings.up_vector,float)
    if up.shape!=(3,) or not np.allclose(up,expected,rtol=0.,atol=1e-6):
        raise ValueError('ROI up_vector disagrees with SfM-estimated COLMAP vertical')
    expected_hash=settings.orientation_result.get('source_sha256')
    if frame.get('orientation_source_sha256')!=expected_hash:
        raise ValueError('ROI orientation source differs from configured COLMAP model')
    if frame.get('scene') != settings.scene:
        raise ValueError('ROI scene does not match configured scene')
    same_source = (frame.get('source_ply') == str(settings.gaussian_ply))
    same_method = (frame.get('method') == settings.method)
    if not same_source or not same_method:
        if not reuse_scene_geometry:
            raise ValueError('ROI scene/method or source PLY does not match config; '
                             'use explicit --reuse-scene-roi for another method')
        if settings.model_scope != 'full_reconstruction' or frame.get('source_model_scope') != 'full_reconstruction':
            raise ValueError('Cross-method ROI reuse requires two complete, verified full-scene models; '
                             'confirm a fresh full-scene ROI to upgrade legacy metadata')
    views=view_record_validate(payload)
    top=next(v for v in views.views if v.name=='top')
    if not np.allclose(np.cross(top.U[0],top.U[1]),expected,rtol=0.,atol=1e-6):
        raise ValueError('Top orthographic projection must align with estimated up_vector')
    for facade in (v for v in views.views if v.name!='top'):
        if not np.allclose(facade.U[1],expected,rtol=0.,atol=1e-6):
            raise ValueError('Facade screen-up must equal the COLMAP up_vector')
    return views


def gaussian_roi_preview(payload, settings, max_points=450, chunk=100_000):
    roi=require_orientation(payload,settings)
    ply=GaussianPLY(settings.gaussian_ply)
    selected=0; sampled=[]; fallback=[]; step=max(1,ply.vertex_count//max_points)
    for start in range(0,ply.vertex_count,chunk):
        stop=min(start+chunk,ply.vertex_count)
        xyz=ply.selected_xyz(np.arange(start,stop))
        # Real multi-polygon intersection in original COLMAP model units; no 3D mesh Boolean required.
        inside=np.ones(len(xyz),dtype=bool)
        for view in roi.views:
            sd,_=view.signed_and_normal(xyz)
            inside &= (sd>=-1e-9)
        selected+=int(inside.sum())
        if len(fallback)<max_points:
            fallback.extend(xyz[np.flatnonzero(inside)[:max_points-len(fallback)]].round(7).tolist())
        if len(sampled)<max_points:
            ids=np.flatnonzero(inside & (((np.arange(start,stop))%step)==0))
            sampled.extend(xyz[ids[:max_points-len(sampled)]].round(7).tolist())
    if not sampled and fallback:sampled=fallback
    return {'gaussian_total':ply.vertex_count,'gaussian_roi_count':selected,
            'roi_fraction':selected/max(1,ply.vertex_count),'sample_xyz':sampled,
            'preview_type':'exact_center_count_sparse_point_overlay',
            'warnings':['Center-inside test is not complete ellipsoid support; stage-3 Poisson/Gaussian protection remains necessary.']}


def current_canonical(settings):
    m=read_manifest(settings.out_dir)
    rel=m['scene_canonical'].get(settings.scene)
    if not rel:return None
    file=settings.out_dir/rel
    if not file.is_file():raise RuntimeError(f'Manifest references missing ROI: {file}')
    return file


def current_latest(settings):
    """Select the last *confirmed* ROI for this scene and verified COLMAP frame.

    Across 3DGS methods there are independent `_001/_002` numbering streams,
    so lexicographic filenames and filesystem modification times are WRONG.
    The authoritative order is manifest.records append order, or the explicit
    scene_latest pointer added by newer versions. This also supports old manifests.
    """
    m = read_manifest(settings.out_dir)
    records = m.get('records', [])
    expected_sha = settings.orientation_result.get('source_sha256')
    scene_records = [r for r in records if r.get('scene') == settings.scene]
    matching = [r for r in scene_records
                if r.get('frame_id') == settings.frame_id
                and r.get('orientation_source_sha256') == expected_sha]
    if not matching:
        if scene_records:
            raise ValueError('ROI files exist for this scene, but none match the '
                             'configured COLMAP frame/orientation source. '
                             'Verify geometric registration; never silently reuse another frame.')
        # Legacy manifest may only contain a scene_canonical pointer.
        legacy = current_canonical(settings)
        if legacy:
            ann = json.loads(legacy.read_text(encoding='utf8')).get('annotation_metadata', {})
            if (ann.get('frame_id') != settings.frame_id or
                ann.get('orientation_source_sha256') != expected_sha):
                raise ValueError('Legacy canonical ROI has a different COLMAP frame')
        return legacy
    latest_record = matching[-1]
    filename = latest_record['filename']
    pointer = m.get('scene_latest', {}).get(settings.scene)
    # An out-of-sync manifest is an error rather than silently analyzing an
    # older ROI. Legacy manifests without scene_latest are fully supported.
    if pointer is not None and pointer != filename:
        raise ValueError(f'ROI manifest latest pointer conflicts with confirmed record order: '
                         f'{pointer} vs {filename}')
    file = settings.out_dir / filename
    if not file.is_file():
        raise FileNotFoundError(f'Latest confirmed ROI is missing: {file}')
    ann = json.loads(file.read_text(encoding='utf8')).get('annotation_metadata', {})
    if (ann.get('scene') != settings.scene or ann.get('frame_id') != settings.frame_id or
        ann.get('orientation_source_sha256') != expected_sha):
        raise ValueError('Latest ROI JSON metadata differs from the scene/COLMAP manifest')
    return file



def _same_source_roi(path, settings):
    """Require the model used when the annotation was confirmed to match.

    AABB and full PLY may share the same SfM frame, but their saved ROI counts
    and downstream masks MUST NOT silently be treated as interchangeable.
    """
    try:
        meta = json.loads(Path(path).read_text(encoding='utf8')).get('annotation_metadata', {})
        return (meta.get('source_ply') == str(settings.gaussian_ply)
                and meta.get('method') == settings.method
                and meta.get('frame_id') == settings.frame_id
                and meta.get('orientation_source_sha256') == settings.orientation_result.get('source_sha256'))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def current_latest_for_source(settings):
    """Latest confirmed annotation for this exact Gaussian model, not merely scene."""
    manifest = read_manifest(settings.out_dir)
    # Retain original manifest integrity checks even if switching model sources.
    latest = current_latest(settings)
    if latest is None:
        return None
    for record in reversed(manifest.get('records', [])):
        if record.get('scene') != settings.scene:
            continue
        if record.get('source_ply') != str(settings.gaussian_ply):
            continue
        path = settings.out_dir / record['filename']
        if _same_source_roi(path, settings):
            return path
    # Legacy manifest has only canonical pointer.
    if not manifest.get('records') and _same_source_roi(latest, settings):
        return latest
    return None


def current_canonical_for_source(settings):
    """Use the original scene canonical only when it came from this model."""
    path = current_canonical(settings)
    return path if path and _same_source_roi(path, settings) else None

def save_confirmed_roi(payload, settings, preview=None, *, new_revision=False):
    require_orientation(payload,settings)
    if preview is None:preview=gaussian_roi_preview(payload,settings)
    if preview['gaussian_roi_count']==0:
        raise ValueError('Empty ROI intersection: all original Gaussian centers excluded. Redraw before saving.')
    with _IO_LOCK:
        out=settings.out_dir;out.mkdir(parents=True,exist_ok=True)
        m=read_manifest(out)
        canonical=m['scene_canonical'].get(settings.scene)
        # A scene cannot silently aggregate spatially incompatible COLMAP
        # frames. Different 3DGS methods may share a scene only after alignment.
        for prior in m.get('records', []):
            if prior.get('scene') != settings.scene:continue
            if (prior.get('frame_id') != settings.frame_id or
                prior.get('orientation_source_sha256') != settings.orientation_result.get('source_sha256')):
                raise ValueError('Existing ROI records for this scene use another COLMAP '
                                 'frame. Use an explicitly different scene ID or register the models.')
        # Module 2 accepts any geometrically valid initial ROI. Do not judge
        # whether a polygon cuts through the target: that belongs to refinement.
        # Every confirmed annotation gets a new numbered artifact; old files and
        # the scene's first canonical reference are never silently overwritten.
        existing={r['filename'] for r in m['records']}
        for num in range(1,10000):
            filename=f'{settings.scene}_{settings.method}_roi_{num:03d}.json'
            if filename not in existing and not (out/filename).exists():break
        else:raise RuntimeError('More than 9999 ROI records for scene/method')
        saved=datetime.now(timezone.utc).isoformat()
        payload=json.loads(json.dumps(payload))
        payload['annotation_metadata']['confirmed_utc']=saved
        payload['annotation_metadata']['source_model_scope']=settings.model_scope
        payload['annotation_metadata']['view_sha256']=canonical_views_sha(payload)
        payload['annotation_metadata']['initial_gaussian_roi_count']=preview['gaussian_roi_count']
        payload['annotation_metadata']['initial_gaussian_roi_fraction']=preview['roi_fraction']
        target=out/filename
        write_atomic(target,payload)
        record={'scene':settings.scene,'method':settings.method,'filename':filename,
                'confirmed_utc':saved,'view_sha256':canonical_views_sha(payload),
                'source_ply':str(settings.gaussian_ply), 'gaussian_count':preview['gaussian_total'],
                'roi_count':preview['gaussian_roi_count'],'up_vector':list(settings.up_vector),
                'orientation_source_sha256':settings.orientation_result.get('source_sha256'),
                'frame_id':settings.frame_id}
        m['records'].append(record)
        # Preserve the scene-wide reference. A changed/experimental ROI is a
        # separate artifact, not an implicit replacement of the standard ROI.
        if not canonical:
            m['scene_canonical'][settings.scene]=filename
        # Every successful confirmation becomes the DEFAULT input to the next
        # module, regardless of method name. Keep the first canonical for
        # reference; changing it is a separate, deliberate operation.
        m.setdefault('scene_latest', {})[settings.scene]=filename
        record['record_type']='revision' if canonical else 'canonical'
        record['reference_canonical']=canonical
        write_atomic(out/'manifest.json',m)
        canonical_path=out/(canonical or filename)
        return {'saved_path':str(target),'reused':False,
                'saved_as_revision':bool(canonical),'canonical_path':str(canonical_path),
                'latest_path':str(target),
                'preview':preview,'manifest':str(out/'manifest.json')}
