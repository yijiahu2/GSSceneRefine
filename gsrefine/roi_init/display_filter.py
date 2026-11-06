"""Temporary module-2 visualization filter based on the prepared SfM-Poisson surface.

Only creates DISPLAY-ONLY PLY subsets. It never modifies source PLY,
ROI JSON, module-3 masks, or module-4 output. This deliberately does not
reuse R3's spatially varying safety widths.

Preview exterior buffer (model units) = multiplier * global_5%-trimmed_mean
of the longest activated Gaussian scale.

A watertight Poisson mesh permits reliable parity signed distances.
Non-watertight meshes use closest-face oriented normal as a *preview-only*
proxy; it may not distinguish ambiguous concavities and is explicitly flagged.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np

from gsrefine.roi.core import GaussianPLY


ALLOWED_MULTIPLIERS = tuple(v / 4 for v in range(1, 17))  # 0.25--4x


def validate_multiplier(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or not any(abs(value - p) < 1e-8 for p in ALLOWED_MULTIPLIERS):
        raise ValueError('Allowed preview safety multipliers: 0.25 to 4.00 in steps of 0.25')
    return round(value, 2)


def global_trimmed_long_axis(arr, trim: float = .05, chunk: int = 150000) -> float:
    """Activated one-sigma major axis: max(exp(scale_0..2)), not 3-sigma.

    A true two-sided 5%-trimmed mean over ALL source Gaussians, independent of
    whether they are normal or anomalous. Quaternion rotation does not change
    covariance eigenvalues, so decoding rotations is unnecessary.
    """
    n = len(arr)
    if n < 20:
        raise ValueError('Need >=20 source Gaussians for 5%-trimmed mean')
    vals = np.empty(n, np.float32)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        log_scales = np.stack([arr[f'scale_{i}'][start:end] for i in range(3)], axis=1)
        if not np.isfinite(log_scales).all() or np.any(np.abs(log_scales) > 80):
            raise ValueError('Gaussian source contains nonfinite or implausible log-scales')
        vals[start:end] = np.exp(np.max(log_scales, axis=1).astype(np.float64)).astype(np.float32)
    if not np.isfinite(vals).all() or np.any(vals <= 0):
        raise ValueError('Invalid Gaussian major-axis lengths')
    k = int(np.floor(trim * n))
    if k * 2 >= n:
        raise ValueError('Not enough Gaussians for symmetric trimming')
    vals.partition((k, n-k-1))
    return float(np.mean(vals[k:n-k], dtype=np.float64))


def keep_for_preview(signed_distances, base_width, multiplier) -> np.ndarray:
    """Keep all inside Gaussians and exterior centers within the safety buffer."""
    multiplier = validate_multiplier(multiplier)
    if not np.isfinite(base_width) or base_width <= 0:
        raise ValueError('Invalid global Gaussian safety width')
    return np.asarray(signed_distances) <= float(base_width) * multiplier


class PoissonDisplayFilter:
    """Lazy signed-distance cache and on-demand transient PLY subsets.

    Build once on first use. The cache identity binds both input file sizes,
    modification times and absolute paths. Only the generated display PLY
    files are eligible for this server's streaming endpoint.
    """
    def __init__(self, settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._ply = None
        self._signed = None
        self._base = None
        self._watertight = None
        self._mode = None
        self._cache = None
        self._previews = {}
        self._identity = None

    @staticmethod
    def _stamp(path):
        path = Path(path).resolve(strict=True)
        st = path.stat()
        return [str(path), st.st_size, st.st_mtime_ns]

    def _key(self):
        sources = [self._stamp(self.settings.gaussian_ply), self._stamp(self.settings.poisson_ply)]
        return hashlib.sha256(json.dumps(sources).encode('utf8')).hexdigest()[:20]

    def _compute_signed_distance(self, ply, destination):
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError('The temporary Poisson display filter requires open3d in gsrefine') from exc
        print('[roi-init] Poisson临时显示过滤：首次计算全量Gaussian到曲面距离（仅预览，原始PLY不变）', flush=True)
        mesh = o3d.io.read_triangle_mesh(str(self.settings.poisson_ply))
        if not mesh.has_triangles():
            raise RuntimeError('Module-1 Poisson has no triangles')
        self._watertight = bool(mesh.is_watertight())
        self._mode = 'watertight_signed_distance' if self._watertight else 'oriented_normal_approximation'
        if not self._watertight:
            print('[roi-init] WARNING: Poisson mesh is not watertight; temporary preview uses nearest-face normal sign. This is NOT a reliable inside/outside test.', flush=True)
            mesh.compute_triangle_normals()
            normals = np.asarray(mesh.triangle_normals, dtype=np.float32)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        signed = np.lib.format.open_memmap(destination, mode='w+', dtype='f4', shape=(ply.vertex_count,))
        for lo in range(0, ply.vertex_count, 40000):
            hi = min(lo + 40000, ply.vertex_count)
            xyz = np.asarray(ply.selected_xyz(np.arange(lo, hi)), dtype=np.float32)
            positions = o3d.core.Tensor(xyz, dtype=o3d.core.Dtype.Float32)
            if self._watertight:
                d = scene.compute_signed_distance(positions, nsamples=5).numpy()
            else:
                result = scene.compute_closest_points(positions)
                nearest_xyz = result['points'].numpy()
                face_id = result['primitive_ids'].numpy().astype('i8')
                if np.any(face_id < 0) or np.any(face_id >= len(normals)):
                    raise RuntimeError('Closest Poisson triangle index invalid')
                d = np.einsum('ij,ij->i', xyz - nearest_xyz, normals[face_id])
            if not np.isfinite(d).all():
                raise RuntimeError('Nonfinite Gaussian-to-Poisson distances')
            signed[lo:hi] = d
            if hi == ply.vertex_count or hi % 400000 == 0:
                print(f'[roi-init] Poisson临时过滤距离计算：{hi:,}/{ply.vertex_count:,}', flush=True)
        signed.flush()
        return signed

    def _prepare(self):
        if not self.settings.poisson_ply:
            raise ValueError('No module-1 Poisson mesh is configured')
        key = self._key()
        if self._identity == key and self._signed is not None:
            return
        self._ply = GaussianPLY(self.settings.gaussian_ply)
        self._cache = Path(self.settings.out_dir) / '.poisson_display_cache'
        self._cache.mkdir(parents=True, exist_ok=True)
        distance_file = self._cache / f'distance_{key}.npy'
        meta_file = self._cache / f'distance_{key}.json'
        if distance_file.exists() and meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding='utf8'))
                arr = np.load(distance_file, mmap_mode='r')
                if len(arr) == self._ply.vertex_count and np.isfinite(arr).all() and meta['width'] > 0:
                    self._signed = arr
                    self._base = float(meta['width'])
                    self._watertight = bool(meta['watertight'])
                    self._mode = str(meta['mode'])
                    self._identity = key
                    return
            except (OSError, ValueError, KeyError):
                pass
        self._base = global_trimmed_long_axis(self._ply.arr)
        tmp = self._cache / f'distance_{key}.tmp.npy'
        try:
            signed = self._compute_signed_distance(self._ply, tmp)
            del signed
            os.replace(tmp, distance_file)
        finally:
            if tmp.exists():
                tmp.unlink()
        info = {'width':self._base, 'watertight': self._watertight, 'mode':self._mode,
                'gaussians':self._ply.vertex_count, 'source_key':key, 'trim_each_tail':0.05}
        meta_file.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf8')
        self._signed = np.load(distance_file, mmap_mode='r')
        self._identity = key

    def prepare_preview(self, multiplier):
        multiplier = validate_multiplier(multiplier)
        with self._lock:
            self._prepare()
            key = (self._identity, multiplier)
            outfile = self._cache / f'preview_{key[0]}_{multiplier:.2f}.ply'
            if not outfile.is_file() or outfile.stat().st_size < self._ply.data_offset:
                mask = keep_for_preview(self._signed, self._base, multiplier)
                if not mask.any():
                    raise RuntimeError('Filter would display zero Gaussians; raw display remains available')
                tmp = self._cache / f'preview_{key[0]}_{multiplier:.2f}.tmp.ply'
                try:
                    count = self._ply.export(mask, tmp)
                    os.replace(tmp, outfile)
                finally:
                    if tmp.exists():
                        tmp.unlink()
            else:
                count = None
            # Read header of generated preview rather than re-counting full mask.
            if count is None:
                count = GaussianPLY(outfile).vertex_count
            self._previews[multiplier] = outfile
            print(f'[roi-init] 临时显示过滤：{multiplier:.2f}×，保留 {count:,}/{self._ply.vertex_count:,} Gaussian（仅预览）', flush=True)
            # Preview models can be GB-sized: retain at most the three newest
            # variants per source to avoid filling the archive drive.
            candidates = sorted(self._cache.glob(f'preview_{key[0]}_*.ply'),
                                key=lambda item:item.stat().st_mtime_ns, reverse=True)
            for stale in candidates[3:]:
                if stale != outfile:
                    stale.unlink(missing_ok=True)
                    for old_mul, old_path in list(self._previews.items()):
                        if old_path == stale:
                            del self._previews[old_mul]
            return {'ok':True, 'multiplier':multiplier,
                    'base_width':self._base, 'safe_width':self._base * multiplier,
                    'display_gaussians':int(count),
                    'total_gaussians':int(self._ply.vertex_count),
                    'watertight_poisson':self._watertight, 'signed_distance_mode':self._mode,
                    'url':f'/api/display-filter.ply?multiplier={multiplier:.2f}&source={self._identity}'}

    def get_existing_preview(self, multiplier):
        multiplier = validate_multiplier(multiplier)
        # Do not allow arbitrary file paths; only previously built exact files.
        with self._lock:
            path = self._previews.get(multiplier)
            return path if path is not None and path.is_file() else None
