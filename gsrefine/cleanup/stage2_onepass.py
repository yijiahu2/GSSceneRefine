"""GSSceneRefine: single-command Stage-2 ROI Gaussian cleanup.

Reuses the repository's established R3 SfM, direct source-Poisson clipping,
main/aux partition, final-R3/global-Poisson barrier audit, and exact parity
classification. Adds unsigned distance protection from ALL local auxiliary
components before writing ONE complete, native Gaussian PLY.

Run:
    python -m gsrefine.cleanup.stage2_onepass --manifest ... --output-dir ... --accept-auto-exterior

All outputs are under the designated Stage-1 ROI result root (no repository
runs/). Newly published files and diagnostics are under --output-dir. Existing
Stage-2 shared geometry caches remain in their canonical ROI-run cleanup paths.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import tempfile
import hashlib

import numpy as np

REVISION = 'GSSceneRefine.stage2_onepass_r3_parity_aux_protection.v1'


def _log(message):
    print('[stage2-onepass] ' + str(message), flush=True)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _native(value):
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_native(v) for v in value]
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf8') as writer:
            json.dump(_native(record), writer, ensure_ascii=False, allow_nan=False, indent=2)
            writer.write('\n')
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def _validate_output(output_dir, run_dir, source_gaussian):
    """Never overwrite original ROI or another prior result."""
    out = Path(output_dir).expanduser().resolve()
    run = Path(run_dir).resolve(strict=True)
    cleanup = run / 'cleanup'
    if not out.is_relative_to(cleanup) or out == cleanup:
        raise ValueError('--output-dir must be a new subdirectory in the same ROI run/cleanup')
    if out == Path(source_gaussian).parent or out == Path(source_gaussian):
        raise ValueError('Refuse writing to original Gaussian directory')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output directory exists and is not empty; choose a new --output-dir: ' + str(out))
    return out


def _find_barrier_audit(roi_run, stage1_manifest, r3_handoff, original_poisson):
    """Reuse an already completed, source-matched audit, never an intersection-only report."""
    root = Path(roi_run) / 'cleanup/diagnostics'
    if not root.is_dir():
        return None
    mh, rh, ph = _sha(stage1_manifest), _sha(r3_handoff), _sha(original_poisson)
    candidates = []
    for report_file in root.glob('**/roi_global_poisson_barrier_report.json'):
        try:
            data = _read(report_file)
            inp = data['inputs']
            if (data['status'] != 'FLOOD_SAMPLING_COMPLETED_NOT_CERTIFIED_SEALED'
                or inp['manifest_sha256'] != mh
                or inp['r3_sha256'] != rh
                or inp['original_poisson_sha256'] != ph):
                continue
            fine = data['flood']['fine']
            if (fine['status'] != 'NO_TOP_TO_BOTTOM_PATH_SEEN_NOT_PROVEN_SEALED'
                or fine['grid']['roi_touching_audit_window_samples']):
                continue
            folder = report_file.parent
            if not (folder/'fine_connectivity.npz').is_file() or not (folder/'global_poisson_final_r3_intersection.npz').is_file():
                continue
            voxel = float(fine['grid']['voxel'])
            candidates.append((voxel, str(folder), folder))
        except (KeyError, TypeError, ValueError, OSError):
            continue
    if not candidates:
        return None
    # Prefer the finest validated sample spacing, deterministic across runs.
    return sorted(candidates, key=lambda a:(a[0],a[1]))[0][2]


def _validated_parity(parity_dir, model, source, preseal_report, original_poisson):
    """Do not combine a spatial result from another ROI / Poisson / Gaussian method."""
    from gsrefine.cleanup.auxiliary_protection import load_validated_parity
    parity_dir = Path(parity_dir).resolve(strict=True)
    outside, over, delta, report, report_file, mask_file = load_validated_parity(
        parity_dir, model.vertex_count, Path(source))
    pin = report['inputs']
    if Path(pin['preseal']).resolve(strict=True) != Path(preseal_report).resolve(strict=True):
        raise ValueError('Parity report belongs to a different main Poisson preseal record')
    if Path(pin['original_full_scene_poisson']).resolve(strict=True) != Path(original_poisson).resolve(strict=True):
        raise ValueError('Parity report belongs to a different full-scene Poisson')
    # Report can be generated by classification-only or formal export.
    if int(report['counts'].get('candidate_uncertain_kept', -1)):
        raise RuntimeError('Parity classification left unresolved over-threshold Gaussian centres; no final clean PLY')
    if not (outside & over).sum() == outside.sum():
        raise RuntimeError('Invalid parity candidate mask')
    preseal = _read(preseal_report)
    if Path(pin['main_trusted_poisson']).resolve(strict=True) != Path(preseal['main_observed_surface']).resolve(strict=True):
        raise ValueError('Parity distance surface differs from authenticated preseal main mesh')
    scale = float(report['threshold']['gaussian_scale']['trimmed_mean_radius'])
    expected_delta = float(preseal['sfm_fit']['p95']) + scale
    if not np.isclose(delta, expected_delta, rtol=1e-8, atol=1e-9):
        raise ValueError('Cached parity delta disagrees with the current complete SfM P95')
    # Source model fields must match V2, not an XYZ point cloud converted from a PLY.
    return outside, over, delta, report, report_file, mask_file


def _auxiliary_protection(model, candidate, delta, auxiliary_path, *, chunk_size=30000):
    """Invoke reusable core protection, not a research-only ablation script."""
    from gsrefine.cleanup.auxiliary_protection import distance_protection
    last=[-1]
    def progress(n,total):
        step=(10*n)//max(total,1)
        if step!=last[0] or n==total:
            last[0]=step
            _log(f'Auxiliary exact triangle distances {n:,}/{total:,}')
    return distance_protection(model, candidate, delta, auxiliary_path,
                               chunk_size=chunk_size, progress=progress)


def _combine_removal(outside, over, protected, *, uncertain_count, all_gaussians):
    """Pure mask calculation; regression tests require disjoint partition."""
    outside = np.asarray(outside, dtype=bool)
    over = np.asarray(over, dtype=bool)
    protected = np.asarray(protected, dtype=bool)
    if (outside.shape != (all_gaussians,) or over.shape != outside.shape
            or protected.shape != outside.shape):
        raise ValueError('Invalid Gaussian classification/protection mask lengths')
    if uncertain_count:
        raise RuntimeError(f'{uncertain_count} Gaussian centres unresolved; cleaned PLY is blocked')
    if np.any(outside & ~over) or np.any(protected & ~outside):
        raise ValueError('Outside/protected mask contradicts threshold geometry')
    remove = outside & ~protected
    keep = ~remove
    if int(keep.sum()) + int(remove.sum()) != all_gaussians:
        raise RuntimeError('Gaussian count conservation failed')
    return keep, remove


def _export_native(model, mask, outfile):
    """Preserve ALL SH/color/opacity/scale/rotation bytes for SuperSplat."""
    from gsrefine.cleanup.auxiliary_protection import export_native_gaussian
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    return export_native_gaussian(model, mask, outfile)


def _existing_preseal(manifest, source_roi_run, *, optional_preseal=None):
    """Validate explicit preseal or require bootstrap; never pick an arbitrary old signature."""
    if optional_preseal is None:
        return None
    report = Path(optional_preseal).expanduser().resolve(strict=True)
    if not report.is_relative_to(Path(source_roi_run)/'cleanup/geometry/direct_r3_sdf'):
        raise ValueError('Preseal report is outside this ROI run')
    data = _read(report)
    if Path(data['stage1_manifest']).resolve(strict=True) != Path(manifest).resolve(strict=True):
        raise ValueError('Preseal report does not belong to current Stage-1 manifest')
    return report


def _discover_preseal(manifest, run_dir):
    """Auto-reuse only a uniquely authenticated full-core canonical R3 preseal."""
    root = Path(run_dir)/'cleanup/geometry/direct_r3_sdf'
    if not root.is_dir():
        return None
    from gsrefine.cleanup.open_ab_main import _validated_inputs
    accepted=[]
    for report in sorted(root.glob('*/preseal_diagnostics.json')):
        try:
            _validated_inputs(manifest, report)
            accepted.append(report)
        except (ValueError, FileNotFoundError, KeyError, RuntimeError, OSError):
            continue
    if len(accepted)>1:
        raise ValueError('Multiple valid preseal candidates exist; explicitly supply '
                         '--preseal-report instead of guessing between geometry revisions')
    return accepted[0] if accepted else None


def _discover_parity(run_dir, source, preseal, global_poisson, model):
    """Reusable V2 classification is selected by provenance, never by a scene name."""
    root = Path(run_dir)/'cleanup'
    matches=[]
    for base in (root/'preview',root/'diagnostics'):
        if not base.is_dir():
            continue
        for report in base.glob('**/roi_parity_report.json'):
            try:
                data = _read(report)
                if data.get('schema') != 'GSSceneRefine.roi_barrier_parity_sdf.v2':
                    continue
                _validated_parity(report.parent,model,source,preseal,global_poisson)
                if (int(data['counts']['candidate_uncertain_kept']) or
                    int(data['geometry']['navigation']['conflicting_roots']) or
                    int(data['short_ray']['conflicting_paths'])):
                    continue
                matches.append(report.parent)
            except (ValueError, KeyError, RuntimeError, OSError, FileNotFoundError):
                continue
    if len(matches)>1:
        # Multiple valid classifications could have different exterior anchors.
        # Do NOT silently guess a winner: the manual --parity-dir makes the choice explicit.
        raise ValueError('Multiple valid parity classifications found; use --parity-dir '
                         'to select a reviewed classification or --reclassify to rebuild')
    return matches[0] if matches else None


def _prepare_required_geometry(manifest, run_dir, *, preseal_report=None):
    """One command: missing SfM, direct R3 clipping and preseal are built automatically."""
    from gsrefine.cleanup.sdf_common import _load_stage1_manifest
    stage1, upstream, inputs, actual_run_dir = _load_stage1_manifest(manifest)
    if Path(actual_run_dir).resolve() != Path(run_dir).resolve():
        raise ValueError('Stage-1 run directory changed')
    preflight = _existing_preseal(stage1, run_dir, optional_preseal=preseal_report)
    if preflight is None:
        preflight = _discover_preseal(stage1, run_dir)
    if preflight is not None:
        # This is mandatory also when the user explicitly supplies a path.
        from gsrefine.cleanup.open_ab_main import _validated_inputs
        _validated_inputs(stage1, preflight)
        _log('Reusing authenticated, full-core R3 main/aux preseal: ' + str(preflight))
        return preflight, {'status':'AUTHENTICATED_PRESEAL_REUSED'}
    # These are the project's canonical shared caches for the current ROI run.
    # They do not write into source repository or override original Stage-1 PLY.
    from gsrefine.geometry.r3_sfm_bridge import resolve_stage2_r3_sfm
    from gsrefine.geometry.r3_direct_poisson_clip import run as run_direct_clip
    from gsrefine.cleanup.direct_r3_pipeline import run as run_r3_check
    _log('1/5 Preparing/validating full post-R3 SfM support')
    sfm_file, sfm_report, sfm_reused = resolve_stage2_r3_sfm(run_dir, upstream, inputs)
    _log('2/5 Preparing/validating R3-clipped ORIGINAL global Poisson')
    clip = run_direct_clip(manifest=stage1, fit_max_points=60000)
    clip_report = Path(clip['candidate_mesh']).parent / 'clip_comparison_report.json'
    _log('3/5 Preparing/validating main and auxiliary Poisson partition + FULL SfM P95')
    result = run_r3_check(manifest=stage1, phase='check', clip_report=str(clip_report))
    preflight = Path(result['preseal_report']).resolve(strict=True)
    if not result.get('safeguards_passed') or not result.get('main_surface'):
        raise RuntimeError('Main surface / complete SfM safeguard gate failed; check ' + str(preflight))
    # The geometry barrier will be independently checked by parity.run.
    return preflight, {'status':result['status'], 'sfm_file':str(sfm_file),
                       'sfm_reused':bool(sfm_reused), 'clip_report':str(clip_report),
                       'preseal':str(preflight)}


def _run_parity(manifest, preseal, output_dir, *, audit_dir=None,
                index_cache=None, exterior_seed=None,
                nearest_k=8, min_path_agree=2, ray_batch=8000,
                chunk_size=30000):
    from gsrefine.cleanup.roi_barrier_parity import run_roi_barrier_parity
    _log('4/5 Classifying all above-threshold Gaussian centres via ROI barrier parity')
    result = run_roi_barrier_parity(
        manifest=str(manifest), preseal_report=str(preseal),
        output_dir=str(output_dir), barrier_audit_dir=str(audit_dir) if audit_dir else None,
        index_cache=str(index_cache) if index_cache else None,
        classification_only=True, confirm_exterior=False,
        exterior_seed=exterior_seed, nearest_k=int(nearest_k),
        min_agree=int(min_path_agree), ray_batch=int(ray_batch),
        chunk_size=int(chunk_size))
    if result.get('status') != 'CLASSIFICATION_ONLY_PREVIEW':
        raise RuntimeError('Parity classification did not finish normally: ' + str(result))
    return result


def run(*, manifest, output_dir, preseal_report=None, barrier_audit_dir=None,
        parity_dir=None, index_cache=None, exterior_seed=None,
        accept_auto_exterior=False, reclassify=False, chunk_size=30000, nearest_k=8,
        min_path_agree=2, ray_batch=8000, export_auxiliary_preview=True,
        keep_classification_previews=False, auxiliary_protection_enabled=True):
    """Publish a single cleaned Gaussian scene, not just detached protected fragments."""
    from gsrefine.cleanup.sdf_common import _load_stage1_manifest
    from gsrefine.roi.core import GaussianPLY
    if int(chunk_size) < 100 or int(nearest_k) < int(min_path_agree) or int(min_path_agree) < 2:
        raise ValueError('Invalid chunk-size / parity neighbour options')
    if exterior_seed is None and not accept_auto_exterior:
        raise ValueError('No certified exterior seed: supply --exterior-seed X Y Z, '
                         'or explicitly use --accept-auto-exterior for a provisional top-shell anchor')
    if reclassify and parity_dir is not None:
        raise ValueError('--reclassify and --parity-dir are mutually exclusive')
    if parity_dir is not None and exterior_seed is not None:
        raise ValueError('A precomputed parity index already has its exterior seed; '
                         'do not supply a different seed without rebuilding the index')
    stage1, upstream, inputs, run_dir = _load_stage1_manifest(manifest)
    dest_dir = _validate_output(output_dir, run_dir, inputs['gaussian_ply'])
    if barrier_audit_dir is not None:
        audit_dir = Path(barrier_audit_dir).expanduser().resolve(strict=True)
    else:
        from gsrefine.geometry.r3_direct_poisson_clip import _source_r3_handoff
        try:
            handoff = _source_r3_handoff(Path(run_dir), upstream)
            audit_dir = _find_barrier_audit(run_dir, stage1, handoff, inputs['poisson_mesh'])
        except FileNotFoundError:
            audit_dir = None
    # Only create final output directory after source & exterior preconditions pass.
    dest_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = dest_dir/'diagnostics'
    diagnostics.mkdir()
    try:
        preseal, geometry_info = _prepare_required_geometry(stage1, run_dir, preseal_report=preseal_report)
        # After geometry generation, a previously missing R3 handoff may now exist.
        if audit_dir is None and barrier_audit_dir is None:
            from gsrefine.geometry.r3_direct_poisson_clip import _source_r3_handoff
            handoff = _source_r3_handoff(Path(run_dir), upstream)
            audit_dir = _find_barrier_audit(run_dir, stage1, handoff, inputs['poisson_mesh'])
        model = GaussianPLY(inputs['gaussian_ply'])
        if parity_dir is None and not reclassify and exterior_seed is None:
            parity_dir = _discover_parity(run_dir, inputs['gaussian_ply'], preseal,
                                          inputs['poisson_mesh'], model)
        parity_path = Path(parity_dir).expanduser().resolve(strict=True) if parity_dir else diagnostics/'parity'
        if parity_dir is None:
            _run_parity(stage1, preseal, parity_path, audit_dir=audit_dir,
                        index_cache=index_cache, exterior_seed=exterior_seed,
                        nearest_k=nearest_k, min_path_agree=min_path_agree,
                        ray_batch=ray_batch, chunk_size=chunk_size)
        else:
            _log('4/5 Reusing source-checked parity classification: ' + str(parity_path))
        outside, over, delta, parity_record, report_file, masks_file = _validated_parity(
            parity_path, model, inputs['gaussian_ply'], preseal, inputs['poisson_mesh'])
        nav = parity_record.get('geometry',{}).get('navigation',{})
        if int(nav.get('known_outside_nodes',0)) == 0 or int(nav.get('known_inside_nodes',0)) == 0:
            raise RuntimeError('Parity classification has no two-sided partition')
        if int(nav.get('conflicting_roots',0)) or int(nav.get('odd_cycle_conflicts',0)):
            raise RuntimeError('Parity navigation has contradictory cycle labeling')
        if int(parity_record.get('short_ray',{}).get('conflicting_paths',0)):
            raise RuntimeError('Parity classification has conflicting short-ray witnesses')
        record = _read(preseal)
        aux_file = record.get('auxiliary_observed_surface')
        if aux_file:
            aux_file = Path(aux_file).resolve(strict=True)
            if not aux_file.is_relative_to(preseal.parent):
                raise ValueError('Auxiliary surface is outside authenticated preseal folder')
            if _sha(aux_file) != record.get('auxiliary_observed_surface_sha256'):
                raise ValueError('Auxiliary Poisson source hash differs from preseal')
        if auxiliary_protection_enabled:
            _log('5/5 Protecting every above-threshold OUTSIDE candidate near ANY auxiliary component')
            protected, component_info, aux_distances = _auxiliary_protection(
                model, outside, delta, aux_file, chunk_size=chunk_size)
        else:
            _log('5/5 AUXILIARY PROTECTION DISABLED (controlled ablation)')
            protected = np.zeros(model.vertex_count, dtype=bool)
            component_info = {'components': 0, 'faces': 0, 'vertices': 0,
                              'faces_per_component': []}
            aux_distances = np.full(int(outside.sum()), np.inf, dtype=np.float32)
        keep, remove = _combine_removal(
            outside, over, protected,
            uncertain_count=int(parity_record['counts']['candidate_uncertain_kept']),
            all_gaussians=model.vertex_count)
        if not np.array_equal(remove, outside & ~protected):
            raise RuntimeError('Final delete mask differs from combined parity+aux rule')
        if len(aux_distances) != int(outside.sum()):
            raise RuntimeError('Auxiliary distances do not align with outside candidate ROI indices')
        count = _export_native(model, keep, dest_dir/'gaussian_roi_clean.ply')
        if count != int(keep.sum()):
            raise RuntimeError('Final Gaussian export did not conserve record count')
        if export_auxiliary_preview and protected.any():
            _export_native(model, protected, diagnostics/'aux_protected_gaussians.ply')
        # Masks are aligned with the immutable Stage-1 ROI Gaussian rows.
        np.savez_compressed(diagnostics/'final_classification_masks.npz',
                            keep=keep, remove=remove, protected_by_aux=protected,
                            original_parity_outside=outside, over_distance_threshold=over,
                            delta=np.float64(delta), candidate_aux_distances=aux_distances,
                            candidate_roi_indices=np.flatnonzero(outside).astype(np.int64))
        result = {
            'schema': REVISION,
            'status': 'COMPLETED_EXPERIMENTAL_CLEAN_ROI_GAUSSIAN',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'scene_method_independent':True,
            'input':{
                'stage1_manifest':str(stage1),
                'stage1_manifest_sha256':_sha(stage1),
                'original_roi_gaussian':str(inputs['gaussian_ply']),
                'original_roi_gaussian_sha256':_sha(inputs['gaussian_ply']),
                'original_global_poisson':str(inputs['poisson_mesh']),
                'original_global_poisson_sha256':_sha(inputs['poisson_mesh']),
                'preseal_report':str(preseal),
                'preseal_report_sha256':_sha(preseal),
                'parity_report':str(report_file),
                'parity_report_sha256':_sha(report_file),
                'auxiliary_poisson':str(aux_file) if aux_file else None,
                'auxiliary_poisson_sha256':_sha(aux_file) if aux_file else None,
            },
            'provenance':geometry_info,
            'method':{
                'stage1_roi':'verified final R3 1x',
                'space_wall':'original full-scene Poisson largest connected body',
                'distance_surface':'R3-clipped main Poisson original triangles',
                'spatial_classifier':'ROI constrained original-triangle parity with short-ray correction',
                'adaptive_delta':'SfM->main Poisson P95 + P10-P90 trimmed mean of 3*major_sigma',
                'auxiliary_rule':('protect any outside/remove candidate with unsigned d_to_any_aux_triangle <= delta' if auxiliary_protection_enabled else 'DISABLED: no auxiliary Poisson protection (ablation)'),
                'auxiliary_protection_enabled':bool(auxiliary_protection_enabled),
                'final_delete_rule':('distance_main>delta+epsilon AND Outside AND distance_to_aux>delta' if auxiliary_protection_enabled else 'distance_main>delta+epsilon AND Outside'),
                'exterior_seed_mode':str(nav.get('seed_mode','unknown')),
                'automatic_exterior_seed_explicitly_accepted':bool(accept_auto_exterior),
            },
            'geometry':{
                'main_faces':int(parity_record['geometry']['distance_poisson_triangles']),
                'auxiliary_components':int(component_info['components']),
                'auxiliary_components_used_for_protection':int(component_info['components']),
                'auxiliary_faces':int(component_info['faces']),
                'intersection_topology':parity_record['geometry']['intersection_topology'],
                'navigation':nav,
            },
            'threshold':{
                'delta':float(delta),
                'epsilon':float(parity_record['threshold']['epsilon']),
                'sfm_p95':float(parity_record['threshold']['sfm_p95']),
                'gaussian_trimmed_mean_radius':float(parity_record['threshold']['gaussian_scale']['trimmed_mean_radius']),
            },
            'counts':{
                'input_gaussians':int(model.vertex_count),
                'at_or_below_threshold_kept':int((~over).sum()),
                'over_threshold':int(over.sum()),
                'outside_before_aux':int(outside.sum()),
                'protected_by_all_aux':int(protected.sum()),
                'final_removed':int(remove.sum()),
                'final_retained':int(keep.sum()),
            },
            'outputs':{
                'final_supersplat_gaussian_ply':str(dest_dir/'gaussian_roi_clean.ply'),
                'final_ply_sha256':_sha(dest_dir/'gaussian_roi_clean.ply'),
                'final_mask':str(diagnostics/'final_classification_masks.npz'),
                'aux_protected_preview':str(diagnostics/'aux_protected_gaussians.ply') if protected.any() and export_auxiliary_preview else None,
                'parity_diagnostics':str(parity_path),
            },
            'limitations':[
                'Auto exterior top shell is a hypothesis until checked against source scene/cameras.',
                'The numerically sampled ROI/Poisson barrier is not a mathematical watertightness certificate.',
                ('Auxiliary components are protected regardless of observational truth; some noise may survive.' if auxiliary_protection_enabled else 'Auxiliary distance protection disabled for controlled ablation; no auxiliary candidates are restored.'),
                'SuperSplat-ready PLY retains the original native Gaussian vertex properties.',
            ]
        }
        # Remove only diagnostic Gaussian PLYs freshly generated INSIDE THIS
        # output-dir. Never delete a pre-existing external parity experiment.
        if parity_dir is None and not keep_classification_previews:
            for name in ('outside_candidates.ply', 'inside_candidates.ply',
                         'uncertain_candidates.ply'):
                (parity_path/name).unlink(missing_ok=True)
            result['outputs']['intermediate_parity_preview_gaussians_removed'] = True
        _atomic_json(dest_dir/'stage2_final_report.json', result)
        _log('SAVED FINAL CLEAN ROI GAUSSIAN: ' + str(dest_dir/'gaussian_roi_clean.ply'))
        _log(f'Gaussians: {model.vertex_count:,} input, {int(protected.sum()):,} aux protected, '
             f'{int(remove.sum()):,} removed, {int(keep.sum()):,} retained')
        return result
    except Exception as err:
        # Never leave an unreported final model looking like a successful result.
        if not (dest_dir/'stage2_final_report.json').is_file():
            (dest_dir/'gaussian_roi_clean.ply').unlink(missing_ok=True)
        _atomic_json(dest_dir/'stage2_failure_report.json',
                     {'schema':REVISION, 'status':'BLOCKED_NOT_EXPORTED',
                      'created_utc':datetime.now(timezone.utc).isoformat(),
                      'error_type':type(err).__name__, 'error':str(err),
                      'note':'Partial intermediate outputs are diagnostics, NOT a certified clean ROI Gaussian.'})
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, help='Completed Stage-1 ROI metadata/manifest.json')
    p.add_argument('--output-dir', required=True, help='NEW stage2 final result directory, inside this ROI run/cleanup')
    p.add_argument('--preseal-report', help='Optional existing authenticated full-SfM main/aux preseal; otherwise auto prepare')
    p.add_argument('--barrier-audit-dir', help='Optional prior source-matched full barrier flood; otherwise auto reuse/build')
    p.add_argument('--parity-dir', help='Optional completed classification directory; otherwise auto classify')
    p.add_argument('--index-cache', help='Optional verified navigation cache (same R3/Poisson)')
    p.add_argument('--reclassify', action='store_true',
                   help='Do not reuse previously validated V2 classification; recompute it for this run')
    p.add_argument('--exterior-seed', nargs=3, type=float, metavar=('X','Y','Z'),
                   help='Externally verified scene outside point inside final R3')
    p.add_argument('--accept-auto-exterior', action='store_true',
                   help='Permit the automatic highest-ROI-shell seed as a PROVISIONAL exterior hypothesis')
    p.add_argument('--chunk-size', type=int, default=30000)
    p.add_argument('--nearest-k', type=int, default=8)
    p.add_argument('--min-path-agree', type=int, default=2)
    p.add_argument('--ray-batch', type=int, default=8000)
    p.add_argument('--no-aux-protection', action='store_true',
                   help='ABlATION: disable auxiliary Poisson distance protection; all outside candidates are removed')
    p.add_argument('--no-aux-preview', action='store_true', help='Skip optional auxiliary protected diagnostics PLY')
    p.add_argument('--keep-classification-previews', action='store_true',
                   help='Keep intermediate outside/inside Gaussian PLYs generated by this run; otherwise save masks only')
    a = p.parse_args(argv)
    result = run(manifest=a.manifest, output_dir=a.output_dir,
                 preseal_report=a.preseal_report, barrier_audit_dir=a.barrier_audit_dir,
                 parity_dir=a.parity_dir, index_cache=a.index_cache,
                 exterior_seed=a.exterior_seed, accept_auto_exterior=a.accept_auto_exterior,
                 reclassify=a.reclassify,
                 chunk_size=a.chunk_size, nearest_k=a.nearest_k,
                 min_path_agree=a.min_path_agree, ray_batch=a.ray_batch,
                 export_auxiliary_preview=not a.no_aux_preview,
                 keep_classification_previews=a.keep_classification_previews,
                 auxiliary_protection_enabled=not a.no_aux_protection)
    print(json.dumps(_native({'status':result['status'],
        'final_gaussian':result['outputs']['final_supersplat_gaussian_ply'],
        'removed':result['counts']['final_removed'],
        'retained':result['counts']['final_retained'],
        'aux_protected':result['counts']['protected_by_all_aux']}), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
