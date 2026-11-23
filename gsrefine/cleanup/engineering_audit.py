"""Read-only scientific and run-integrity audits; never alters any classification.

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
