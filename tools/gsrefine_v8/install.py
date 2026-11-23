#!/usr/bin/env python3
"""GSSceneRefine engineering v8: guarded, transactional source update.

Requires exact source HEAD 1ebc9c0. No algorithmic threshold or classifier
changes are made. Read MODIFY_REPORT.md for deployment and validation limits.
"""
import argparse
import ast
from datetime import datetime
from pathlib import Path
import re
import shutil
import subprocess
import sys

BASE = '1ebc9c06013b29f5fa71cfdeb1b0754cbd670123'
PATCH = {}
NEW = {}


def edit(path, old, new, *, count=1):
    PATCH.setdefault(path, []).append((old, new, count))


NEW['gsrefine/geometry/poisson_support.py'] = '''"""Read-only, deterministic diagnostics of ORIGINAL Poisson/SfM surface support.

Neither geometry nor threshold decisions may depend on these reports.
"""
import numpy as np
from scipy.spatial import cKDTree


def diagnose(mesh, points, *, sfm_spacing=None, sample_faces=20000):
    pts = np.asarray(points, dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(pts) < 3 or not len(faces) or not np.isfinite(pts).all():
        raise ValueError('Invalid SfM/Poisson diagnostic input')
    chosen = np.linspace(0, len(faces)-1, min(sample_faces, len(faces)), dtype=np.int64)
    centers = vertices[faces[chosen]].mean(axis=1)
    tree = cKDTree(pts)
    dist = tree.query(centers, k=1, workers=-1)[0]
    if sfm_spacing is None or not np.isfinite(sfm_spacing) or sfm_spacing <= 0:
        q = pts[np.linspace(0, len(pts)-1, min(len(pts), 12000), dtype=np.int64)]
        sfm_spacing = float(np.median(tree.query(q, k=2, workers=-1)[0][:, 1]))
    slo, shi = pts.min(axis=0), pts.max(axis=0)
    mlo, mhi = vertices.min(axis=0), vertices.max(axis=0)
    sfm_extent = np.maximum(shi - slo, 1e-12)
    mesh_extent = mhi - mlo
    far = dist > 5.0 * max(float(sfm_spacing), 1e-12)
    outside_box = np.any((centers < slo) | (centers > shi), axis=1)
    return dict(
        status='REVIEW_ONLY_NOT_GROUND_TRUTH', sampled_faces=int(len(chosen)),
        total_faces=int(len(faces)), closest_sfm_p50=float(np.median(dist)),
        closest_sfm_p95=float(np.quantile(dist, .95)),
        sfm_spacing_reference=float(sfm_spacing),
        far_from_sfm_sample_fraction=float(far.mean()),
        sampled_faces_outside_sfm_bbox_fraction=float(outside_box.mean()),
        poisson_to_sfm_extent_ratios=(mesh_extent/sfm_extent).tolist(),
        poisson_bounds=[mlo.tolist(), mhi.tolist()],
        sfm_bounds=[slo.tolist(), shi.tolist()],
        almost_cubic_mesh_extent=bool(np.ptp(mesh_extent) < .005*max(mesh_extent.max(), 1e-12)),
        review_required=bool(far.mean() > .05),
        note='Sparse SfM absence is not proof of spurious geometry; no faces are removed.')
'''

NEW['gsrefine/cleanup/engineering_audit.py'] = '''"""Read-only scientific and run-integrity audits; never alters any classification.

Stage-1 masks are aligned with full source PLY rows; Stage-2 masks with R3 rows.
Mapping is obtained from the sorted R3 boolean selection of the same native PLY.
"""
from pathlib import Path
import hashlib
import json
import numpy as np


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def cross_stage(run_dir, source_roi_ply, final_masks, *, input_count):
    run_dir = Path(run_dir)
    source = run_dir/'metadata'/'stage1_masks.npz'
    meta = run_dir/'metadata'/'stage1_masks_integrity.json'
    if not source.is_file() or not meta.is_file():
        return dict(status='STAGE1_MASKS_NOT_AVAILABLE', note='Historical run; no inferred membership')
    declaration = json.loads(meta.read_text('utf-8'))
    if declaration['sha256'] != _digest(source):
        raise ValueError('Permanent Stage-1 mask hash mismatch')
    if Path(declaration['r3_ply']).resolve() != Path(source_roi_ply).resolve():
        raise ValueError('Cross-stage audit is bound to a different R3 PLY')
    with np.load(source, allow_pickle=False) as f:
        r0, core, safety, r3 = (np.asarray(f[k], dtype=bool) for k in ('R0','core','safety','R3'))
    if not (len(r0) == len(core) == len(safety) == len(r3)
            and np.array_equal(core|safety,r3) and not np.any(core&safety)
            and not np.any(r0&~core)):
        raise ValueError('Stage-1 mask partitions invalid')
    idx = np.flatnonzero(r3)
    if len(idx) != int(input_count):
        raise ValueError('Stage-1 R3 mask count differs from input Stage-2 Gaussian count')
    deletion = np.asarray(final_masks, dtype=bool)
    if len(deletion) != len(idx):
        raise ValueError('Stage-2 mask is not aligned with the Stage-1 R3 PLY')
    keep = ~deletion
    labels = dict(R0=r0[idx], recovered_core=(core&~r0)[idx], safety_added=safety[idx])
    groups = {}
    for name, mask in labels.items():
        groups[name] = dict(input=int(mask.sum()), removed=int((mask&deletion).sum()),
                            retained=int((mask&keep).sum()))
    assert sum(x['input'] for x in groups.values()) == len(idx)
    return dict(status='AUDITED_COUNTS_NOT_GROUND_TRUTH', group_counts=groups,
                r3_source_rows=int(len(r3)), r3_retained_rows=int(len(idx)),
                removed_total=int(deletion.sum()), note='Risk screening, not semantic correctness assessment')


def classification_risk(*, points, remove, outside, withheld, protected, delta, scope):
    xyz = np.asarray(points, dtype=np.float64)
    removal = np.asarray(remove, dtype=bool)
    outside = np.asarray(outside, dtype=bool)
    withheld = np.asarray(withheld, dtype=bool)
    protected = np.asarray(protected, dtype=bool)
    if not (len(xyz) == len(removal) == len(outside) == len(withheld) == len(protected)):
        raise ValueError('Risk report masks are not aligned')
    if np.any(removal&withheld) or np.any(removal&~outside) or np.any(removal&protected):
        raise ValueError('Conservative deletion invariants violated')
    subset = xyz[removal]
    extent = (np.max(subset,axis=0)-np.min(subset,axis=0)).tolist() if len(subset) else [0.,0.,0.]
    anchor = scope.get('anchor')
    seed = dict(mode='user_coordinate' if scope.get('user_exterior_seed') else 'auto_top_shell',
                value=np.asarray(anchor).tolist() if anchor is not None else None,
                independently_verified=False,
                note='Auto top shell is a provisional hypothesis; finite parity checks cannot prove semantic exterior')
    return dict(status='NUMERIC_ONLY_NOT_TRUTH',
                deletion_invariant_passed=True, input_count=int(len(xyz)),
                outside_count=int(outside.sum()), removed_count=int(removal.sum()),
                withheld_count=int(withheld.sum()), auxiliary_protected=int(protected.sum()),
                delta=float(delta), removed_xyz_extent=extent, exterior_seed=seed,
                possible_failure_modes=['incorrect exterior anchor', 'unsupported Poisson surface',
                    'legitimate Gaussian outside threshold', 'invalid assumed surface correspondence'])
'''

# Module 1: read-only Poisson support diagnostic; no change to geometry or Poisson algorithm.
edit('gsrefine/geometry/prepare.py',
     '    return dict(sfm_points=int(len(points)), mesh_vertices=int(len(verts)),',
     '    from gsrefine.geometry.poisson_support import diagnose\n'
     "    support = diagnose(mesh, points, sfm_spacing=spacing_med if 'spacing_med' in locals() else None)\n"
     "    if support['review_required']:\n"
     "        notes.append('Poisson surface has sparsely SfM-supported sampled faces; review spatial extension.')\n"
     '    return dict(poisson_support_diagnostic=support, sfm_points=int(len(points)), mesh_vertices=int(len(verts)),')

# Stage-1 verified SfM 1x/2x membership versus projection-failure cross-tab.
edit('gsrefine/geometry/r3_sfm_bridge.py',
     '                  boundary_projection_failures=int(np.count_nonzero(residual > projection_tolerance)),',
     "                  boundary_projection_failures=int(np.count_nonzero(residual > projection_tolerance)),\n"
     "                  boundary_projection_failure_cross_tab=dict(\n"
     "                      core_1x=int(np.count_nonzero((residual > projection_tolerance) & core)),\n"
     "                      context_2x=int(np.count_nonzero((residual > projection_tolerance) & build & ~core)),\n"
     "                      outside_2x=int(np.count_nonzero((residual > projection_tolerance) & ~build))),")

# Module 2: retain numerical preview distance cache across browser lifetimes;
# keep the potentially very large preview PLYs transient.
edit('gsrefine/roi_init/display_filter.py',
     '        self._cache = None\n        self._previews = {}',
     '        self._cache = None\n        self._preview_dir = None\n        self._previews = {}')
edit('gsrefine/roi_init/display_filter.py',
     "        self._cache = Path(self.settings.out_dir) / '.poisson_display_cache'\n        self._cache.mkdir(parents=True, exist_ok=True)",
     "        self._cache = Path(self.settings.out_dir).parent / 'geometry' / 'display_distance_cache'\n"
     "        self._cache.mkdir(parents=True, exist_ok=True)\n"
     "        self._preview_dir = Path(self.settings.out_dir) / '.poisson_display_cache'\n"
     "        self._preview_dir.mkdir(parents=True, exist_ok=True)")
edit('gsrefine/roi_init/display_filter.py',
     "            outfile = self._cache / f'preview_{key[0]}_{multiplier:.2f}.ply'",
     "            outfile = self._preview_dir / f'preview_{key[0]}_{multiplier:.2f}.ply'")
edit('gsrefine/roi_init/display_filter.py',
     "                tmp = self._cache / f'preview_{key[0]}_{multiplier:.2f}.tmp.ply'",
     "                tmp = self._preview_dir / f'preview_{key[0]}_{multiplier:.2f}.tmp.ply'")
edit('gsrefine/roi_init/display_filter.py',
     "            candidates = sorted(self._cache.glob(f'preview_{key[0]}_*.ply'),",
     "            candidates = sorted(self._preview_dir.glob(f'preview_{key[0]}_*.ply'),")
# Browser close retains the persistent numeric data by design; existing server.py
# only removes the separate temporary .poisson_display_cache.

# New formal Stage 2 refuses hidden reconstruction from historical v1/v2 handoffs.
# Historic migration CLI retains the old behavior unless explicitly strict.
edit('gsrefine/geometry/r3_sfm_bridge.py',
     'def resolve_stage2_r3_sfm(run_dir, upstream, inputs, *, local_safety_file=None):',
     'def resolve_stage2_r3_sfm(run_dir, upstream, inputs, *, local_safety_file=None, require_published_v3=False):')
edit('gsrefine/geometry/r3_sfm_bridge.py',
     '    if record is None:\n        # A historical Stage-1 v2 file',
     "    if require_published_v3 and (record is None or record.resolve() != (run_dir/'metadata'/PUBLISHED_NAME).resolve()):\n"
     "        raise ValueError('Formal Stage 2 requires exact published Stage-1 R3 v3 handoff; legacy fallback disabled')\n"
     '    if record is None:\n        # A historical Stage-1 v2 file')
edit('gsrefine/cleanup/stage2_onepass.py',
     "(run_dir, upstream, inputs)",
     "(run_dir, upstream, inputs, require_published_v3=True)")

# Stage 1: published permanent compact masks; same numerical results.
edit('gsrefine/roi_init/workflow.py',
     '        model, manifest, summary = publish(out, work, roi, config, report,\n                                           keep_diagnostics=keep_diagnostics)',
     '''        # Persist scientific selection partitions BEFORE publish moves/deletes work.
        import numpy as np
        with np.load(work/'keep_mask_R0.npy', allow_pickle=False) if False else open(work/'keep_mask_R0.npy','rb') as _unused:
            pass
        stage1_masks = {
            'R0': np.load(work/'keep_mask_R0.npy', allow_pickle=False).astype(np.bool_),
            'core': np.load(work/'core_mask_R3.npy', allow_pickle=False).astype(np.bool_),
            'safety': np.load(work/'safety_added_mask_R3.npy', allow_pickle=False).astype(np.bool_),
            'R3': np.load(work/'keep_mask_R3.npy', allow_pickle=False).astype(np.bool_),
        }
        if (any(x.shape != stage1_masks['R0'].shape for x in stage1_masks.values())
                or not np.array_equal(stage1_masks['core']|stage1_masks['safety'],stage1_masks['R3'])
                or np.any(stage1_masks['core']&stage1_masks['safety'])
                or np.any(stage1_masks['R0']&~stage1_masks['core'])):
            raise RuntimeError('Stage-1 R3 scientific mask partition inconsistent')
        metadata = out/'metadata'
        metadata.mkdir(parents=True,exist_ok=True)
        mask_path = metadata/'stage1_masks.npz'
        np.savez_compressed(mask_path, **stage1_masks)
        from gsrefine.cleanup.engineering_audit import _digest
        write_atomic(metadata/'stage1_masks_integrity.json', {
            'schema':'GSSceneRefine.stage1_masks.v1', 'sha256':_digest(mask_path),
            'r3_ply':str((out/FINAL_REL).resolve()),
            'full_input_count':int(len(stage1_masks['R3'])),
            'r3_count':int(stage1_masks['R3'].sum()),
            'note':'Masks preserve full-source Gaussian row order; Stage-2 PLY is R3 subsequence.'})
        model, manifest, summary = publish(out, work, roi, config, report,
                                           keep_diagnostics=keep_diagnostics)''')
# no intermediary bogus open logic
PATCH['gsrefine/roi_init/workflow.py'][-1] = (PATCH['gsrefine/roi_init/workflow.py'][-1][0], PATCH['gsrefine/roi_init/workflow.py'][-1][1].replace("        with np.load(work/'keep_mask_R0.npy', allow_pickle=False) if False else open(work/'keep_mask_R0.npy','rb') as _unused:\n            pass\n", ''), 1)

edit('gsrefine/roi_init/output_layout.py',
     "    manifest = {\n        'schema': LAYOUT_VERSION,",
     "    manifest = {\n        'schema': LAYOUT_VERSION,") # intentionally noop anchor detects exact source
edit('gsrefine/roi_init/output_layout.py',
     "        'result_status': 'ROI_RESTORED_NOT_ANOMALY_CLEANED',\n    }",
     "        'result_status': 'ROI_RESTORED_NOT_ANOMALY_CLEANED',\n"
     "        'stage1_masks': {'file':str(meta/'stage1_masks.npz'),\n"
     "                         'sha256':stage1_runtime.digest(meta/'stage1_masks.npz')},\n    }")
edit('gsrefine/roi_init/output_layout.py',
     "        'algorithm_stage': 'Poisson-guided ROI restoration and boundary protection',",
     "        'quality_review': {\n"
     "            'roi_exterior_area_ratio':report.get('candidate',{}).get('initial_exterior_ratio'),\n"
     "            'roi_recovery_review_required':bool(stages.get('review_required',False)),\n"
     "            'roi_exterior_area_review_required':(float(report.get('candidate',{}).get('initial_exterior_ratio',0)) >\n"
     "                float((__import__('yaml').safe_load(Path(config).read_text(encoding='utf8')) or {}).get('guards',{}).get('max_extension_area_fraction',0.1))),\n"
     "            'poisson_faces_unvalidated':not bool(report.get('poisson_face_scoring',{}).get('enabled',False)),\n"
     "            'scientific_truth_certified':False},\n"
     "        'algorithm_stage': 'Poisson-guided ROI restoration and boundary protection',")

# Register a single official 3-mode command under the EXISTING run.py dispatcher.
edit('run.py',
     "    'full-pipeline':'gsrefine.roi_init.full_pipeline',",
     "    'full-pipeline':'gsrefine.roi_init.full_pipeline',\n"
     "    'pipeline':'gsrefine.roi_init.formal_pipeline',")

# Formal default to the currently confirmed original-region, no global-distance A/B.
edit('gsrefine/roi_init/full_pipeline.py', "default='adaptive-voxel',", "default='original-region',")

# Original wall source in tests is a runtime dependency. Copy only its small
# source_groups() utility into a proper library; do not import test code in production.
NEW['gsrefine/geometry/original_wall_source.py'] = '''"""Source-consistent largest edge-connected original-global Poisson wall."""
import numpy as np
import trimesh


def source_groups(mesh):
    groups = trimesh.graph.connected_components(mesh.face_adjacency,
        nodes=np.arange(len(mesh.faces)), min_len=1)
    groups = sorted(groups, key=len, reverse=True)
    component = np.empty(len(mesh.faces), np.int32)
    for index, rows in enumerate(groups):
        component[rows] = index
    main = np.sort(groups[0])
    vertices, inverse = np.unique(mesh.faces[main], return_inverse=True)
    wall = trimesh.Trimesh(vertices=mesh.vertices[vertices],
                           faces=inverse.reshape(-1, 3), process=False)
    return wall, component, [len(g) for g in groups]
'''
edit('gsrefine/cleanup/local_region_pipeline.py',
     '            from tools.test_r3_original_walls import source_groups',
     '            from gsrefine.geometry.original_wall_source import source_groups')
# Copy existing original wall audit into stable library unchanged; redirect
# production imports only. Existing tools remain as historical CLI comparators.
# This copy is prepared from the local repo at update time, not a guessed script.

# Stage 2 provenance and classification telemetry must survive local-region early returns.
edit('gsrefine/cleanup/local_region_pipeline.py',
     "    _atomic_json(diagnostics/'scope.json', scope)\n        print(f\"[original-region] v7 original wall:",
     "    _atomic_json(diagnostics/'scope.json', scope)\n        print(f\"[original-region] v7 original wall:") # no-op; optional removed if mismatch
# trim that optional no-op: this anchor uses wrong indent and shouldn't block
PATCH['gsrefine/cleanup/local_region_pipeline.py'] = [x for x in PATCH['gsrefine/cleanup/local_region_pipeline.py'] if x[0] != x[1]]

# Explicit alternative auxiliary rule evaluation, never changes final remove mask.
edit('gsrefine/cleanup/local_region_pipeline.py',
     '        protect, previews, aux_preview, partition_report=None, original_mode=False):',
     '        protect, previews, aux_preview, partition_report=None, original_mode=False, aux_ablation=False):')
edit('gsrefine/cleanup/local_region_pipeline.py',
     '    protected = np.zeros(len(points),bool)\n    if protect and outside.any() and aux:\n', '    protected = np.zeros(len(points),bool)\n    shadow_protected = None\n    if aux_ablation and not protect and outside.any() and aux:\n        shadow_protected, _, _ = _auxiliary_protection(model,outside,delta,aux,chunk_size=chunk)\n    if protect and outside.any() and aux:\n')

edit('gsrefine/cleanup/local_region_pipeline.py',
     "    if aux_preview and protected.any():",
     '''    # Read-only deleted-Gaussian risk, exterior seed and cross-stage attribution.
    from gsrefine.cleanup.engineering_audit import cross_stage, classification_risk
    import json as _json
    from gsrefine.cleanup.stage2_runtime import measured
    with measured('engineering_cross_stage_and_risk_audit'):
        engineering = {
            'cross_stage': cross_stage(Path(stage1).parent.parent, inputs['gaussian_ply'],
                                      remove, input_count=len(points)),
            'risk': classification_risk(points=points, remove=remove, outside=outside,
                        withheld=withheld, protected=protected, delta=delta, scope=scope),
            'auxiliary_ablation': dict(
                        status=('COUNTERFACTUAL_SAME_CLASSIFICATION' if protect or shadow_protected is not None
                                else 'NOT_COMPUTED_NO_AUXILIARY_RUN'),
                        protection_enabled=bool(protect),
                        removed_if_auxiliary_off=int(outside.sum()),
                        removed_with_protection=(int(outside.sum()-(protected if protect else shadow_protected).sum())
                                if protect or shadow_protected is not None else None),
                        extra_removed_without_protection=(int((protected if protect else shadow_protected).sum())
                                if protect or shadow_protected is not None else None),
                        note='Counterfactual reuses same side and distance classifications; does not edit final mask.'),
        }
        _atomic_json(diagnostics/'engineering_audit.json',engineering)
    if aux_preview and protected.any():''')
edit('gsrefine/cleanup/local_region_pipeline.py',
     "    result = dict(schema='GSSceneRefine.local_region.v1',status=status,",
     "    result = dict(schema='GSSceneRefine.local_region.v1',status=status,"
     "engineering_audit=str(diagnostics/'engineering_audit.json'),"
     "quality_review_required=bool(withheld.any() or seed is None),")

# Stage 2: canonical geometry entry for new formal code (old phase for historical CLI only).
edit('gsrefine/cleanup/direct_r3_pipeline.py',
     "    if phase not in ('check','seal','sdf','export'):",
     "    if phase not in ('check','sealless','seal','sdf','export'):")
edit('gsrefine/cleanup/direct_r3_pipeline.py',
     "    if phase == 'check':\n        status = 'ELIGIBLE_FOR_SEAL_TRIAL' if record['preseal_eligible'] else 'BLOCKED_PRESEAL'",
     "    if phase in ('check','sealless'):\n        status = ('SEALLESS_GEOMETRY_PREPARED' if phase == 'sealless' else\n                  ('ELIGIBLE_FOR_SEAL_TRIAL' if record['preseal_eligible'] else 'BLOCKED_PRESEAL'))")
# Explicitly redirect formal call to 'sealless'.
edit('gsrefine/cleanup/stage2_onepass.py',
     'keep_classification_previews=False, auxiliary_protection_enabled=True, classifier=None, partition_report=None):',
     'keep_classification_previews=False, auxiliary_protection_enabled=True, classifier=None, partition_report=None, audit_aux_ablation=False):')
edit('gsrefine/cleanup/stage2_onepass.py',
     "original_mode=classifier == 'original-region')",
     "original_mode=classifier == 'original-region', aux_ablation=audit_aux_ablation)")
edit('gsrefine/cleanup/stage2_onepass.py',
     "    p.add_argument('--no-aux-protection', action='store_true',",
     "    p.add_argument('--audit-aux-ablation', action='store_true',\n"
     "                   help='When auxiliary protection is OFF, compute a read-only counterfactual')\n"
     "    p.add_argument('--no-aux-protection', action='store_true',")
edit('gsrefine/cleanup/stage2_onepass.py',
     'auxiliary_protection_enabled=not a.no_aux_protection)',
     'auxiliary_protection_enabled=not a.no_aux_protection, audit_aux_ablation=a.audit_aux_ablation)')
edit('gsrefine/roi_init/full_pipeline.py',
     "    stage2.add_argument('--no-aux-protection', action='store_true')",
     "    stage2.add_argument('--no-aux-protection', action='store_true')\n"
     "    stage2.add_argument('--audit-aux-ablation', action='store_true')")
edit('gsrefine/roi_init/full_pipeline.py',
     "('no_aux_protection', 'no_aux_preview', 'keep_classification_previews', 'reclassify')",
     "('no_aux_protection', 'no_aux_preview', 'keep_classification_previews', 'reclassify', 'audit_aux_ablation')")

# Fine-grained timing with no change to numerical decisions.
edit('gsrefine/cleanup/local_region_pipeline.py',
     "    radii = robust_gaussian_radius(model, defaults['trim_fraction'], defaults['sigma_multiplier'])",
     "    radii = measured('gaussian_radius_statistics')(robust_gaussian_radius)(model, defaults['trim_fraction'], defaults['sigma_multiplier'])")
edit('gsrefine/cleanup/local_region_pipeline.py',
     "        tree = RegionTree(index,domain,distance,tau,guard,\n                          wall_distance=wall_distance if original else None)",
     "        tree = measured('region_tree_build')(RegionTree)(index,domain,distance,tau,guard,\n                          wall_distance=wall_distance if original else None)")
edit('gsrefine/cleanup/local_region_pipeline.py',
     "            tree = RegionTree(index,domain,distance,tau,guard,wall_distance=wall_distance,",
     "            tree = measured('region_tree_conflict_rebuild')(RegionTree)(index,domain,distance,tau,guard,wall_distance=wall_distance,")
edit('gsrefine/cleanup/local_region_pipeline.py',
     "                cover = RegionCover(index, tree, radius)",
     "                cover = measured('region_cover_build')(RegionCover)(index, tree, radius)")
edit('gsrefine/cleanup/local_region_pipeline.py',
     "        writer.writerow(['阶段二主程序至汇总',1,timing['total_s'],''])",
     "        writer.writerow(['阶段二主程序至汇总',1,timing['total_s'],''])\n"
     "        writer.writerow(['未归类开销',1,'',timing['other_s']])")

edit('gsrefine/cleanup/stage2_onepass.py',
     "result = measured('run_r3_check')(run_r3_check)(manifest=stage1, phase='check', clip_report=str(clip_report))",
     "result = measured('run_r3_sealless')(run_r3_check)(manifest=stage1, phase='sealless', clip_report=str(clip_report))")

# Stage-2 source-order geometry is unchanged; only its formal entry is renamed.

# Full pipeline provenance: preserve execution-code revision and dirty indicator.
edit('gsrefine/roi_init/full_pipeline.py',
     'def _write_state(roi_run, stage, manifest, output=None, *, extra=None):',
     "def _code_identity():\n    repo = Path(__file__).resolve().parents[2]\n    def git(*args):\n        result = subprocess.run(['git','-C',str(repo),*args], stdout=subprocess.PIPE,\n                                stderr=subprocess.DEVNULL, text=True, check=False)\n        return result.stdout.strip() if result.returncode == 0 else None\n    return dict(git_head=git('rev-parse','HEAD'), git_dirty=bool(git('status','--porcelain')),\n                implementation='engineering_unified_v8')\n\n\ndef _write_state(roi_run, stage, manifest, output=None, *, extra=None):")
edit('gsrefine/roi_init/full_pipeline.py',
     "        'attempt_history': history,",
     "        'attempt_history': history,\n        'execution_code': _code_identity(),")

# Full pipeline: quality review is visible at the TOP-LEVEL success state.
edit('gsrefine/roi_init/full_pipeline.py',
     "                                    'counts': report.get('counts', {})})",
     "                                    'counts': report.get('counts', {}),\n"
     "                                    'quality_review': {\n"
     "                                        'auto_exterior_seed_is_provisional': bool(a.accept_auto_exterior),\n"
     "                                        'stage1':_read_json(roi_run/'metadata'/'processing_report.json').get('quality_review',{}),\n"
     "                                        'stage2_unverified_gaussians': report.get('counts',{}).get('unverified_retained'),\n"
     "                                        'engineering_audit':report.get('engineering_audit'),\n"
     "                                        'truth_certified':False}})")


def patch(root, dry_run=False):
    root = Path(root).expanduser().resolve()
    if not (root/'run.py').is_file() or not (root/'.git').exists():
        raise SystemExit('Project root must be an existing GSSceneRefine Git working tree')
    head = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if head != BASE:
        raise SystemExit('BLOCKED: HEAD %s != verified baseline %s' % (head, BASE))
    # The zip itself is intentionally unpacked INTO this repository. Ignore ONLY
    # its known untracked installer files; all other dirty state remains fatal.
    dirty = subprocess.check_output(
        ['git','-C',str(root),'status','--porcelain','--untracked-files=all'], text=True)
    packaged = {
        'gsrefine/roi_init/formal_pipeline.py',
        'tools/gsrefine_v8/install.py',
        'tools/gsrefine_v8/compare_runs.py',
        'tools/gsrefine_v8/test_patch_components.py',
        'tools/gsrefine_v8/test_runner.py',
        'tools/gsrefine_v8/README.md',
        'tools/gsrefine_v8/VERSION.json',
    }
    changes = []
    for line in dirty.splitlines():
        if line.startswith('?? ') and line[3:].strip('"') in packaged:
            continue
        changes.append(line)
    if changes:
        raise SystemExit('BLOCKED: unrelated dirty working tree (source protected):\n'+'\n'.join(changes))
    # Check the injected new module too. The repository will not be changed
    # until this and every existing edit anchor have passed validation.
    official_module = root / 'gsrefine/roi_init/formal_pipeline.py'
    if not official_module.is_file():
        raise SystemExit('BLOCKED: formal pipeline entry missing from overlay ZIP')
    ast.parse(official_module.read_text(encoding='utf8'), filename=str(official_module))
    newfiles = {}
    for path, operations in PATCH.items():
        target = root/path
        if not target.is_file():
            raise RuntimeError('Missing required source: '+path)
        content = target.read_text(encoding='utf8')
        for old, new, expected in operations:
            n = content.count(old)
            if n != expected:
                raise RuntimeError('BLOCKED ambiguous/missing anchor: %s, matched %d, expected %d: %r' % (path,n,expected,old[:110]))
            content = content.replace(old, new, expected)
        newfiles[path] = content
    for path, content in NEW.items():
        if (root/path).exists():
            raise RuntimeError('Refusing to overwrite already present new file: '+path)
        newfiles[path] = content
    # The original wall audit is NOT rewritten: its exact implementation is
    # copied into a production package and imported there. Preserve source.
    source = root/'tools'/'audit_final_r3_global_poisson_barrier.py'
    if not source.is_file():
        raise RuntimeError('Missing original global Poisson reference implementation')
    newfiles['gsrefine/geometry/original_wall_audit.py'] = source.read_text(encoding='utf8')
    # production imports only, never redirect historical tools/tests
    to_replace = 'tools.audit_final_r3_global_poisson_barrier'
    from_import = 'from tools import audit_final_r3_global_poisson_barrier'
    for f in (root/'gsrefine').rglob('*.py'):
        relative = str(f.relative_to(root))
        s = newfiles.get(relative, f.read_text(encoding='utf8'))
        if to_replace not in s and from_import not in s:
            continue
        s = s.replace(to_replace, 'gsrefine.geometry.original_wall_audit')
        s = s.replace(from_import, 'from gsrefine.geometry import original_wall_audit')
        newfiles[relative] = s
    for path, content in newfiles.items():
        try:
            ast.parse(content, filename=path)
        except SyntaxError as exc:
            raise RuntimeError('BLOCKED syntax verification for %s: %s' % (path, exc)) from exc
    print('Validated baseline:', head)
    print('Will modify %d files:' % len(newfiles))
    for path in sorted(newfiles):
        print(' ',path)
    if dry_run:
        print('DRY RUN ONLY. No files touched.')
        return
    backup = root.parent/('GSSceneRefine_before_engineering_v8_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    backup.mkdir(parents=True, exist_ok=False)
    applied = []
    try:
        for path, data in newfiles.items():
            target = root/path
            if target.is_file():
                copy = backup/path
                copy.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(target,copy)
        for path, data in newfiles.items():
            target = root/path
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(data, encoding='utf8')
            applied.append(path)
        # Parsing checks were run before commit. Running compilation here is
        # intentionally omitted (will create __pycache__; use syntax-safe ast).
        (backup/'INSTALLED_FILES.txt').write_text('\n'.join(sorted(newfiles)),encoding='utf8')
    except BaseException:
        for path in reversed(applied):
            copy = backup/path
            target = root/path
            if copy.exists():
                shutil.copy2(copy,target)
            else:
                target.unlink(missing_ok=True)
        raise
    print('Update applied. Original files backed up at:',backup)
    print('Next: python -m compileall -q gsrefine && python run.py pipeline --help && git diff --check')

def verify(root):
    root = Path(root).expanduser().resolve()
    require = {
        'run.py': "'pipeline':'gsrefine.roi_init.formal_pipeline'",
        'gsrefine/roi_init/full_pipeline.py': "default='original-region'",
        'gsrefine/roi_init/workflow.py': "stage1_masks_integrity.json",
        'gsrefine/roi_init/output_layout.py': "stage1_masks",
        'gsrefine/cleanup/local_region_pipeline.py': "engineering_audit.json",
        'gsrefine/cleanup/stage2_onepass.py': "require_published_v3=True",
        'gsrefine/geometry/r3_sfm_bridge.py': 'require_published_v3=False',
    }
    import ast
    for name, marker in require.items():
        file = root/name
        text = file.read_text(encoding='utf8')
        if marker not in text:
            raise SystemExit('FAILED verification for '+name+': '+marker)
        ast.parse(text, filename=str(file))
    for name in ('gsrefine/geometry/original_wall_audit.py',
                 'gsrefine/geometry/original_wall_source.py',
                 'gsrefine/geometry/poisson_support.py',
                 'gsrefine/cleanup/engineering_audit.py',
                 'gsrefine/roi_init/formal_pipeline.py'):
        file = root/name
        ast.parse(file.read_text(encoding='utf8'), filename=str(file))
    print('PASS: engineering v8 installed in canonical code paths; syntax and key registrations verified')


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--project-root',default='/home/xth/GSSceneRefine')
    p.add_argument('--check',action='store_true',help='preflight all anchors and AST; no changes')
    p.add_argument('--apply',action='store_true',help='apply source changes after exact preflight')
    p.add_argument('--verify',action='store_true',help='verify v8 sources after installation')
    a=p.parse_args()
    if int(a.check)+int(a.apply)+int(a.verify) != 1:
        p.error('Choose exactly one of --check, --apply or --verify')
    if a.verify:
        verify(a.project_root)
    else:
        patch(a.project_root,dry_run=a.check)
