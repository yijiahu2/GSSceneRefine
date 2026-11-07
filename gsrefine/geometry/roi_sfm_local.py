"""Strict per-location Stage-1 R3 safety -> Stage-2 clean SfM handoff.

Only previously CLEANED Stage-1 SfM is eligible. The Stage-1 R3 safety
function is the sole source of d(x). Per SfM point:
  core  := signed_initial_roi(x) >= -d_R3(x)
  build := signed_initial_roi(x) >= -2*d_R3(x)
This is a spatial SfM support envelope, NOT a redefinition of the Stage-1
Gaussian R3 selection or of the Stage-2 SDF threshold.

No scalar approximation, P95 re-estimation, or raw SfM reload is permitted.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

SAFETY_SCHEMA = 'GSSceneRefine.r3_local_safety.v2'
ROI_SFM_SCHEMA = 'GSSceneRefine.roi_sfm.v2'
SAFETY_SEMANTICS = 'stage1_r3_original_per_location_safety_on_clean_sfm'
BUFFER_RULE = 'initial_confirmed_roi_signed >= -multiplier * exact_stage1_r3_local_safety'
WORK_SAFETY_NAME = 'stage2_ready_r3_local_safety.npz'
PUBLISHED_SAFETY_NAME = 'stage1_r3_local_safety.npz'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.stem + '_', suffix='.npz', dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _atomic_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '_', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _safety_values(safety, count):
    safety = np.asarray(safety)
    if safety.ndim != 1 or safety.shape != (count,):
        raise ValueError('Expected a genuine per-SfM-point R3 safety VECTOR (N,), not a scalar d')
    if safety.dtype.kind not in 'fiub':
        raise ValueError('R3 safety is not numeric')
    safety = safety.astype(np.float64, copy=False)
    if not np.isfinite(safety).all() or np.any(safety < 0):
        raise ValueError('Stage-1 R3 local safety must be finite and nonnegative')
    if not np.any(safety > 0):
        raise ValueError('Stage-1 R3 local safety is uniformly zero')
    return safety


def save_stage1_r3_local_safety(work_dir, *, clean_sfm_npz, roi_json,
                                safety_at=None, per_point_safety=None,
                                chunk_size=100000):
    """Call at the Stage-1 R3 local safety calculation, before work cleanup.

    `safety_at(xyz)` MUST be the exact existing Stage-1 per-location d(x)
    evaluator, with no re-estimation. For an existing computed (N,) array,
    pass `per_point_safety` instead, indexed in CLEAN SfM source row order.
    A scalar is deliberately rejected even for constant-valued geometry.
    """
    if (safety_at is None) == (per_point_safety is None):
        raise ValueError('Supply exactly one of the Stage-1 evaluator or the existing full pointwise safety vector')
    source = Path(clean_sfm_npz).resolve(strict=True)
    roi = Path(roi_json).resolve(strict=True)
    with np.load(source, allow_pickle=False) as sfm:
        if 'xyz' not in sfm:
            raise ValueError('Stage-1 CLEAN SfM must have xyz')
        xyz = np.asarray(sfm['xyz'])
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Invalid Stage-1 CLEAN SfM xyz')
    n = len(xyz)
    if safety_at is not None:
        if not callable(safety_at):
            raise ValueError('Stage-1 safety_at must be callable')
        results = []
        for a in range(0, n, chunk_size):
            block = np.asarray(safety_at(xyz[a:a+chunk_size]))
            # Check each block, do not allow scalar broadcasting.
            results.append(_safety_values(block, min(chunk_size, n-a)))
        safety = np.concatenate(results)
    else:
        safety = _safety_values(per_point_safety, n)
    work_dir = Path(work_dir)
    dest = work_dir / WORK_SAFETY_NAME
    if dest.exists():
        raise FileExistsError(f'Exact Stage-1 safety record already exists: {dest}')
    _atomic_npz(dest, schema=np.array(SAFETY_SCHEMA), semantics=np.array(SAFETY_SEMANTICS),
                cleaned_sfm_sha256=np.array(sha256(source)), roi_sha256=np.array(sha256(roi)),
                point_count=np.array(n, dtype=np.int64),
                safety_distance_per_sfm_point=safety)
    return dest


def read_verified_local_safety(path, *, clean_sfm_npz, roi_json):
    """Validate the exact Stage-1 vector, including source hash and row count."""
    source = Path(clean_sfm_npz).resolve(strict=True)
    roi = Path(roi_json).resolve(strict=True)
    with np.load(source, allow_pickle=False) as src:
        if 'xyz' not in src:
            raise ValueError('Stage-1 CLEAN SfM lacks xyz')
        n = len(src['xyz'])
    with np.load(path, allow_pickle=False) as rec:
        mandatory = ('schema', 'semantics', 'cleaned_sfm_sha256', 'roi_sha256',
                     'point_count', 'safety_distance_per_sfm_point')
        if any(k not in rec for k in mandatory):
            raise ValueError('Missing exact Stage-1 safety record fields')
        if str(rec['schema'].item()) != SAFETY_SCHEMA or str(rec['semantics'].item()) != SAFETY_SEMANTICS:
            raise ValueError('Not an exact per-location Stage-1 R3 safety record')
        if str(rec['cleaned_sfm_sha256'].item()) != sha256(source) or str(rec['roi_sha256'].item()) != sha256(roi):
            raise ValueError('Exact local safety source SHA mismatch; cannot reuse')
        if int(rec['point_count'].item()) != n:
            raise ValueError('Exact local safety point count differs from CLEAN SfM')
        return _safety_values(rec['safety_distance_per_sfm_point'], n)


def select_clean_sfm_local(clean_sfm_npz, roi_json, local_safety_file, *,
                           min_core=100, chunk_size=100000):
    """Only subset Stage-1 cleaned SfM, applying 1x and 2x SAME local d(x)."""
    from gsrefine.roi.core import ROI
    from gsrefine.geometry.roi_surface import roi_signed
    source = Path(clean_sfm_npz).resolve(strict=True)
    roi_json = Path(roi_json).resolve(strict=True)
    safety = read_verified_local_safety(local_safety_file, clean_sfm_npz=source, roi_json=roi_json)
    roi = ROI.from_file(roi_json)
    with np.load(source, allow_pickle=False) as src:
        xyz = np.asarray(src['xyz'])
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError('Invalid cleaned SfM xyz')
        n = len(xyz)
        inside_distance = roi_signed(xyz, roi, chunk=chunk_size)
        core = inside_distance >= -safety
        build = inside_distance >= -2.0*safety
        count = int(core.sum())
        if count < min_core:
            raise ValueError(f'Only {count} core cleaned SfM points, need {min_core}')
        if np.any(core & ~build):
            raise AssertionError('2x SfM safety is not a superset of 1x core')
        selected = {}
        for key in src.files:
            arr = np.asarray(src[key])
            if arr.ndim >= 1 and len(arr) == n:
                selected[key] = arr[build]
            elif arr.ndim == 0:
                selected[key] = arr
            else:
                raise ValueError(f'Ambiguous cleaned SfM field {key} shape {arr.shape}')
    selected.update({
        'in_core_roi': core[build].astype(np.bool_),
        'signed_distance_to_initial_roi': inside_distance[build].astype(np.float64),
        'stage1_r3_local_safety_distance': safety[build].astype(np.float64),
        'core_outward_distance_local': safety[build].astype(np.float64),
        'build_outward_distance_local': (2.0*safety[build]).astype(np.float64),
        'buffer_semantics': np.array(BUFFER_RULE),
        'source_cleaned_sfm_sha256': np.array(sha256(source)),
        'source_local_safety_sha256': np.array(sha256(local_safety_file)),
        'roi_sfm_schema': np.array(ROI_SFM_SCHEMA),
    })
    report = {
        'schema': ROI_SFM_SCHEMA,
        'selection_mode': 'exact_per_location_stage1_r3_safety_1x_2x',
        'source_is_cleaned_stage1_sfm': True,
        'safety_source': {'is_exact_stage1_r3_safety': True,
                          'is_spatially_varying': bool(np.max(safety) > np.min(safety)),
                          'semantics': SAFETY_SEMANTICS},
        'source_sfm_points': int(n),
        'core_points': count,
        'build_points': int(build.sum()),
        'context_only_points': int(np.sum(build & ~core)),
        'local_safety_min': float(np.min(safety)),
        'local_safety_median': float(np.median(safety)),
        'local_safety_max': float(np.max(safety)),
        'local_safety_p95': float(np.quantile(safety, .95)),
        'coordinate_frame': 'unchanged COLMAP world',
        'rule': BUFFER_RULE,
        'gaussian_r3_unchanged': True,
        'sdf_threshold_formula_unchanged': True,
        'boundary_notice': 'Core is a 1x spatial safety envelope, not an exact replay of Gaussian R3 topology/continuity membership.',
    }
    return selected, report


def _identity(*, clean_sfm_npz, roi_json, gaussian_ply, poisson_mesh, safety_file):
    return {
        'clean_sfm_sha256': sha256(clean_sfm_npz),
        'roi_json_sha256': sha256(roi_json),
        'gaussian_roi_sha256': sha256(gaussian_ply),
        'original_poisson_sha256': sha256(poisson_mesh),
        'local_r3_safety_sha256': sha256(safety_file),
    }


def create_exact_roi_sfm(*, clean_sfm_npz, roi_json, gaussian_ply, poisson_mesh,
                         local_safety_file, output_npz, output_report):
    source = Path(clean_sfm_npz).resolve(strict=True)
    roi = Path(roi_json).resolve(strict=True)
    g = Path(gaussian_ply).resolve(strict=True)
    p = Path(poisson_mesh).resolve(strict=True)
    f = Path(local_safety_file).resolve(strict=True)
    result = Path(output_npz)
    report_path = Path(output_report)
    identity = _identity(clean_sfm_npz=source, roi_json=roi, gaussian_ply=g,
                         poisson_mesh=p, safety_file=f)
    if result.exists() or report_path.exists():
        if not (result.is_file() and report_path.is_file()):
            raise FileExistsError('Incomplete exact SfM publication, refusing overwrite')
        old = json.loads(report_path.read_text(encoding='utf8'))
        if (old.get('schema') != ROI_SFM_SCHEMA or old.get('sources') != identity
                or old.get('sfm_roi_sha256') != sha256(result)):
            raise ValueError('Existing ROI SfM is not exact v2 or its source hash changed; refusing overwrite')
        validate_exact_roi_sfm(result, report_path, source, roi, g, p, f)
        return result.resolve(), old
    arrays, detail = select_clean_sfm_local(source, roi, f)
    record = {**detail, 'sources': identity,
              'source_clean_sfm': str(source), 'stage1_local_safety_file': str(f)}
    _atomic_npz(result, **arrays)
    record['sfm_roi_sha256'] = sha256(result)
    try:
        _atomic_json(report_path, record)
    except Exception:
        result.unlink(missing_ok=True)
        raise
    return result.resolve(), record


def validate_exact_roi_sfm(roi_sfm_npz, report_path, clean_sfm_npz, roi_json,
                           gaussian_ply, poisson_mesh, local_safety_file):
    roi_path = Path(roi_sfm_npz).resolve(strict=True)
    report = json.loads(Path(report_path).resolve(strict=True).read_text(encoding='utf8'))
    expected = _identity(clean_sfm_npz=clean_sfm_npz, roi_json=roi_json,
                         gaussian_ply=gaussian_ply, poisson_mesh=poisson_mesh,
                         safety_file=local_safety_file)
    if (report.get('schema') != ROI_SFM_SCHEMA or report.get('sources') != expected
            or report.get('sfm_roi_sha256') != sha256(roi_path)):
        raise ValueError('Exact local SfM source hash / report mismatch')
    with np.load(roi_path, allow_pickle=False) as data:
        needed = ('xyz', 'in_core_roi', 'roi_sfm_schema',
                  'stage1_r3_local_safety_distance', 'build_outward_distance_local',
                  'source_cleaned_sfm_sha256', 'source_local_safety_sha256')
        if any(k not in data for k in needed):
            raise ValueError('Missing per-point local SfM fields')
        if str(data['roi_sfm_schema'].item()) != ROI_SFM_SCHEMA:
            raise ValueError('Not an exact local v2 SfM record')
        if str(data['source_cleaned_sfm_sha256'].item()) != expected['clean_sfm_sha256']:
            raise ValueError('SfM ROI originated from wrong cleaned Stage-1 SfM')
        if str(data['source_local_safety_sha256'].item()) != expected['local_r3_safety_sha256']:
            raise ValueError('SfM ROI originated from wrong R3 safety field')
        xyz = data['xyz']; core = data['in_core_roi']; d = data['stage1_r3_local_safety_distance']
        build_dist = data['build_outward_distance_local']
        if xyz.ndim != 2 or xyz.shape[1] != 3 or core.shape != (len(xyz),) or core.dtype != bool:
            raise ValueError('Malformed exact local SfM xyz/core masks')
        if len(d) != len(xyz) or len(build_dist) != len(xyz):
            raise ValueError('Malformed per-point safety distance vectors')
        if not np.isfinite(d).all() or not np.array_equal(build_dist, 2.0*d):
            raise ValueError('SfM build buffer is not exactly twice its local Stage-1 safety')
    return report


def publish_stage1_exact_roi_sfm(*, downstream, metadata_dir, work_dir=None):
    """Stage-1 publisher hook: preserve only ACTUALLY computed exact R3 field.

    When local safety was never computed/published, returns (None,None) so
    Stage-1 Gaussian export remains unchanged. Never estimates a scalar d.
    """
    if work_dir is None:
        return None, None
    source = Path(work_dir) / WORK_SAFETY_NAME
    if not source.is_file():
        return None, None
    sfm = downstream['clean_sfm_npz']; roi = downstream['roi_json']
    read_verified_local_safety(source, clean_sfm_npz=sfm, roi_json=roi)
    meta = Path(metadata_dir)
    meta.mkdir(parents=True, exist_ok=True)
    saved = meta / PUBLISHED_SAFETY_NAME
    if saved.exists():
        if sha256(saved) != sha256(source):
            raise ValueError('Existing published Stage-1 safety differs; refusing overwrite')
    else:
        temp = saved.with_name('.' + saved.name + '.writing')
        try:
            shutil.copy2(source, temp)
            os.replace(temp, saved)
        finally:
            temp.unlink(missing_ok=True)
    final = Path(downstream['gaussian_ply']).parent
    path, report = create_exact_roi_sfm(clean_sfm_npz=sfm, roi_json=roi,
                  gaussian_ply=downstream['gaussian_ply'], poisson_mesh=downstream['poisson_mesh'],
                  local_safety_file=saved, output_npz=final/'sfm_roi.npz',
                  output_report=meta/'roi_sfm_report.json')
    downstream['r3_local_safety_npz'] = str(saved.resolve())
    return path, report


def resolve_stage2_exact_roi_sfm(run_dir, upstream, inputs, *, local_safety_file=None):
    """Stage-2: verified Stage-1 v2 output, else exact safety -> local v2 SfM.

    A historical v1 scalar-envelope SfM is NOT accepted. Missing Stage-1
    per-location data is reported; never silently reconstructed by a global d.
    """
    run = Path(run_dir)
    published = (upstream.get('stage2_inputs') or {})
    referenced = published.get('sfm_roi_npz')
    possible = [Path(local_safety_file)] if local_safety_file else []
    if published.get('r3_local_safety_npz'):
        possible.append(Path(published['r3_local_safety_npz']))
    possible.append(run/'metadata'/PUBLISHED_SAFETY_NAME)
    possible.append(run/'cleanup/inputs'/PUBLISHED_SAFETY_NAME)
    safety = next((f for f in possible if f.is_file()), None)
    if safety is None:
        raise ValueError('Missing exact Stage-1 per-location R3 safety field. '
                         'The old global scalar/P95 fallback is disabled. '
                         'Stage 1 must publish its actual d(x) for cleaned SfM points, '
                         'or Stage 2 must rerun that same Stage-1 local safety evaluator '
                         'and save an exact field (not refit a scalar).')
    read_verified_local_safety(safety, clean_sfm_npz=inputs['clean_sfm_npz'], roi_json=inputs['roi_json'])
    if referenced:
        reference = Path(referenced)
        report_file = run/'metadata/roi_sfm_report.json'
        if reference.is_file() and report_file.is_file():
            try:
                record = validate_exact_roi_sfm(reference, report_file,
                    inputs['clean_sfm_npz'], inputs['roi_json'], inputs['gaussian_ply'],
                    inputs['poisson_mesh'], safety)
                return reference.resolve(), record, True
            except ValueError as error:
                # A legacy V1 handoff is ignored, but a corrupted V2 must fail.
                existing = json.loads(report_file.read_text(encoding='utf8'))
                if existing.get('schema') == ROI_SFM_SCHEMA:
                    raise ValueError('Existing exact v2 Stage-1 SfM cannot be validated') from error
    outdir = run/'cleanup/inputs'
    dest, report = create_exact_roi_sfm(
        clean_sfm_npz=inputs['clean_sfm_npz'], roi_json=inputs['roi_json'],
        gaussian_ply=inputs['gaussian_ply'], poisson_mesh=inputs['poisson_mesh'],
        local_safety_file=safety,
        output_npz=outdir/'sfm_roi_local_v2.npz',
        output_report=outdir/'roi_sfm_local_v2_report.json')
    return dest, report, False
