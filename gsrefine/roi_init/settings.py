"""Strict configuration and input identity for reproducible ROI initialization."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import yaml
import numpy as np
from gsrefine.geometry.orientation import read_or_estimate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TAG = re.compile(r'^[a-zA-Z][a-zA-Z0-9_\-]{0,50}$')


@dataclass(frozen=True)
class InitSettings:
    config: Path
    scene: str
    method: str
    gaussian_ply: Path
    poisson_ply: Path | None
    out_dir: Path
    up_vector: tuple[float,float,float]
    orientation_result: dict
    frame_id: str
    orientation_verified: bool
    ui_host: str
    ui_port: int
    file_host: str
    file_port: int
    reuse_scene_roi: bool
    allow_new_revision: bool
    max_elevations: int
    model_scope: str

    @classmethod
    def load(cls, config):
        path = Path(config).expanduser().resolve(strict=True)
        data = yaml.safe_load(path.read_text(encoding='utf8'))
        if not isinstance(data, dict):
            raise ValueError('Config YAML must be a mapping')
        p = data.get('paths', {})
        m = data.get('roi_init', {})
        scene_def = data.get('scene', {})
        if not isinstance(scene_def, dict):
            raise ValueError('scene section must contain name and method')
        scene = scene_def.get('name', m.get('scene', ''))
        method = scene_def.get('method', m.get('method', ''))
        if not _TAG.fullmatch(str(scene)) or not _TAG.fullmatch(str(method)):
            raise ValueError('scene.name and scene.method must be descriptive ASCII identifiers')
        gauss = Path(m.get('gaussian_ply') or p.get('gaussian_ply', '')).expanduser()
        if not gauss.is_file():
            raise FileNotFoundError(f'Gaussian model not found: {gauss}')
        gauss = gauss.resolve()
        poisson_path = m.get('poisson_ply') or p.get('poisson_raw')
        poisson = Path(poisson_path).expanduser().resolve() if poisson_path else None
        # The geometric Poisson overlay is optional and lazy: never a UI startup prerequisite.
        if poisson and not poisson.is_file():
            print(f'[roi-init] Poisson overlay unavailable: {poisson}; continuing without it')
            poisson = None
        out = Path(m.get('output_dir', PROJECT_ROOT / 'runs' / 'roi_init')).expanduser()
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        orientation = m.get('orientation', {})
        frame_id = orientation.get('frame_id', f'{scene}_colmap_original')
        if not _TAG.fullmatch(str(frame_id)):
            raise ValueError('roi_init.orientation.frame_id must name the shared original COLMAP world frame for this scene')
        # Auto mode estimates an ARBITRARY vector in original SfM coordinates.
        # Old +z/verified options cannot override this result.
        cache_dir = Path(orientation.get('cache_dir', out.parent / 'geometry' / 'orientation'))
        if not cache_dir.is_absolute():
            cache_dir = out.parent / cache_dir
        estimate=read_or_estimate(scene,str(frame_id),orientation,data,cache_dir)
        if not estimate.get('accepted'):
            raise ValueError('Automatic COLMAP vertical direction is uncertain: '
                             + '; '.join(estimate.get('warnings',[]))
                             + f'. See {cache_dir} for diagnostics. '
                             + 'Do not force +Z; inspect GNSS/IMU or use an explicitly verified manual vector.')
        up=np.asarray(estimate['up_vector'],float)
        if up.shape!=(3,) or not np.isfinite(up).all() or abs(np.linalg.norm(up)-1)>1e-6:
            raise ValueError('Invalid automatically estimated vertical vector')
        ui = m.get('web', {})
        native = m.get('file_service', {})
        ui_host, native_host = ui.get('host', '127.0.0.1'), native.get('host', '127.0.0.1')
        if ui_host not in ('127.0.0.1','localhost') or native_host not in ('127.0.0.1','localhost'):
            raise ValueError('ROI annotation servers must only bind to localhost')
        port, native_port = int(ui.get('port', 8765)), int(native.get('port', 8766))
        if port == native_port or not (1024 <= min(port,native_port) and max(port,native_port)<65536):
            raise ValueError('web.port and file_service.port must be distinct, nonprivileged ports')
        max_elev = int(m.get('views', {}).get('max_elevation_views', 4))
        if max_elev < 1 or max_elev > 4:
            raise ValueError('Only 1-4 elevation views are supported')
        return cls(path,str(scene),str(method),gauss,poisson,out.resolve(),tuple(up.tolist()),estimate,str(frame_id),True,
                   ui_host,port,native_host,native_port,
                   bool(m.get('reuse_scene_roi',True)), bool(m.get('allow_new_revision',False)),
                   max_elev,str(data.get('source_model_scope','unknown')))
