"""Conservative, scene-independent R3 / ORIGINAL Poisson wall separation gate.

This replaces ONLY the invalid top-to-bottom flood verdict. Flood *graphs* and
exact original-triangle intersection tests stay unchanged.  Connected sampled
regions need original-triangle parity evidence and a coherent coarse/fine map.

This is a NUMERICAL barrier audit, not a watertightness certificate.  Never
publish a Gaussian cleanup result when this gate returns BLOCKED.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SCHEMA = 'GSSceneRefine.R3_original_wall_region_separation.v2'


def _centers(origin, h, shape):
    from tools.audit_final_r3_global_poisson_barrier import grid_centers
    return grid_centers(origin, h, shape)


def _load(path, audit_level):
    with np.load(path, allow_pickle=False) as z:
        origin = np.asarray(z['origin'], np.float64)
        h = float(z['voxel'])
        shape = tuple(int(t) for t in z['shape'])
        passable = np.asarray(z['passable'], bool)
        region = np.asarray(z['components'], np.int32)
        inside = np.asarray(z['final_r3_inside'], bool)
    if len(shape) != 3 or any(a < 2 for a in shape) or origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('Invalid '+audit_level+' grid geometry')
    if not np.isfinite(h) or h <= 0 or any(a.shape != shape for a in (passable, region, inside)):
        raise ValueError('Inconsistent '+audit_level+' grid shape or spacing')
    if not np.array_equal(region >= 0, passable) or np.any(passable & ~inside):
        raise ValueError('Invalid '+audit_level+' flood component membership')
    ids, counts = np.unique(region[passable], return_counts=True)
    order = np.argsort(-counts, kind='stable')
    ids, counts = ids[order], counts[order]
    n = int(passable.sum())
    # A very small component is not sufficient evidence for a separate side;
    # include *all* material components, rather than assuming exactly two.
    material_threshold = max(16, int(np.ceil(n * .01)))
    material = ids[counts >= material_threshold].astype(int).tolist()
    coverage = int(sum(int(counts[np.flatnonzero(ids == i)[0]]) for i in material))
    return dict(origin=origin, voxel=h, shape=shape, passable=passable,
                regions=region, xyz=_centers(origin, h, shape),
                component_sizes={int(i): int(s) for i, s in zip(ids, counts)},
                material=material, coverage=coverage, n_passable=n,
                material_threshold=material_threshold,
                sha256=_hash(path))


def _hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _select_local_pairs(a, b, xyz, h, *, limit=120, radius=5.):
    a_ids = np.flatnonzero(a)
    b_ids = np.flatnonzero(b)
    if not len(a_ids) or not len(b_ids):
        return np.empty((0, 2, 3), np.float64)
    tree = cKDTree(xyz[b_ids])
    d, near = tree.query(xyz[a_ids], k=1)
    eligible = np.flatnonzero(d <= radius * h)
    eligible = eligible[np.argsort(d[eligible],kind='stable')]
    accepted = []
    # Spread probes along the spatial wall interface; avoid 120 nearly
    # identical samples from one Poisson triangle.
    for idx in eligible:
        point = xyz[a_ids[idx]]
        if all(np.linalg.norm(point - xyz[a_ids[other]]) >= 1.75 * h for other in accepted):
            accepted.append(int(idx))
            if len(accepted) >= limit:
                break
    if not accepted:
        return np.empty((0, 2, 3), np.float64)
    arr = np.asarray(accepted)
    return np.stack([xyz[a_ids[arr]], xyz[b_ids[near[arr]]]], axis=1)


def _pairs_test(grid, a, b, field, wall_scene, o3d, normals, *, max_pairs=120):
    from gsrefine.cleanup.roi_barrier_parity import segment_parity
    from tools.audit_final_r3_global_poisson_barrier import link_r3_filter

    labels = grid['regions'].ravel()
    pairs = _select_local_pairs(labels == a, labels == b, grid['xyz'],grid['voxel'],limit=max_pairs)
    info = {'regions': [int(a), int(b)], 'sampled_pairs': int(len(pairs)),
            'within_final_R3': 0, 'reliable': 0, 'odd': 0, 'even': 0,
            'status': 'NOT_ESTABLISHED'}
    if not len(pairs):
        return info
    # R3 might be nonconvex: a pair of valid endpoints does NOT guarantee a
    # connecting straight segment stays inside R3.
    good = link_r3_filter(pairs[:, 0], pairs[:, 1], field,
                          samples=max(11, int(np.ceil(5.0 / .5))), batch=256)
    pairs = pairs[good]
    info['within_final_R3'] = int(len(pairs))
    if not len(pairs):
        return info
    bits, valid = segment_parity(wall_scene, o3d, pairs[:, 0], pairs[:, 1],
                                 normals=normals, ray_batch=1000,
                                 endpoint_tol=max(1e-6, grid['voxel'] * 1e-5))
    info['reliable'] = int(valid.sum())
    info['odd'] = int(np.count_nonzero(valid & (bits == 1)))
    info['even'] = int(np.count_nonzero(valid & (bits == 0)))
    if info['reliable'] < 8:
        info['status'] = 'INSUFFICIENT_TRIANGLE_EVIDENCE'
    elif info['odd'] and info['even']:
        info['status'] = 'INCONSISTENT_CROSS_REGION_PARITY'
    elif info['odd']:
        info['status'] = 'OPPOSITE_SIDE_SUPPORTED'
        info['bit'] = 1
    else:
        info['status'] = 'SAME_SIDE_SUPPORTED'
        info['bit'] = 0
    return info


def _region_graph(material, relations):
    # Region-level XOR consistency; disconnected components can have same or
    # opposite labels. Never assume that different components imply odd parity.
    parent={i:i for i in material}
    xor={i:0 for i in material}
    def root(i):
        if parent[i] == i:
            return i,0
        r,p=root(parent[i])
        xor[i]^=p
        parent[i]=r
        return r,xor[i]
    conflicts=0
    for x in relations:
        if x['status'] not in ('OPPOSITE_SIDE_SUPPORTED','SAME_SIDE_SUPPORTED'):
            continue
        a,b=x['regions']; bit=x['bit']
        ra,xa=root(a);rb,xb=root(b)
        if ra == rb:
            conflicts += int((xa^xb) != bit)
        else:
            parent[ra]=rb
            xor[ra]=xa^xb^bit
    roots={root(i)[0] for i in material}
    labels=[root(i)[1] for i in material]
    return {'relation_roots':int(len(roots)), 'parity_conflicts':int(conflicts),
            'opposite_side_evidenced': len(set(labels)) >= 2 and len(roots)==1,
            'material_region_labels':{str(i):int(root(i)[1]) for i in material}}


def _check_level(grid, field, wall_scene, o3d, normals):
    ids=grid['material']
    summary = {'voxel': float(grid['voxel']), 'grid_sha256':grid['sha256'],
               'total_passable':grid['n_passable'],
               'component_sizes':grid['component_sizes'],
               'material_ids':ids, 'material_coverage':grid['coverage'],
               'material_fraction':grid['coverage']/max(1,grid['n_passable']),
               'material_threshold':grid['material_threshold']}
    errors=[]
    if len(ids)<2:
        errors.append('Fewer than two material sampled spatial regions')
    if summary['material_fraction'] < .95:
        errors.append('Material regions account for less than 95% of navigable space')
    if len(ids) > 12:
        errors.append('More than 12 material regions: manual geometric review required')
    relations=[]
    if not errors:
        for k,a in enumerate(ids):
            for b in ids[k+1:]:
                item=_pairs_test(grid,a,b,field,wall_scene,o3d,normals)
                relations.append(item)
                if item['status']=='INCONSISTENT_CROSS_REGION_PARITY':
                    errors.append(f'Conflicting original-triangle crossing parity: regions {a}/{b}')
        graph=_region_graph(ids,relations)
        if graph['parity_conflicts']:
            errors.append('Inconsistent region-level XOR cycles')
        if graph['relation_roots'] != 1:
            errors.append('Not all material spatial regions have verified triangle-side relations')
        if not graph['opposite_side_evidenced']:
            errors.append('No pair of material regions demonstrated opposite Poisson sides')
    else:
        graph={}
    summary.update(relations=relations, graph=graph, errors=errors)
    return summary


def _cross_scale(coarse, fine):
    """Map each coarse material region onto a fine material region in XYZ space.

    Component IDs are local to each grid! This prevents assuming coarse 0 is
    inherently the same physical side as fine 0.
    """
    rng=np.random.default_rng(20261111)
    all_fine=np.flatnonzero(fine['passable'].ravel())
    tree=cKDTree(fine['xyz'][all_fine])
    fine_labels=fine['regions'].ravel()
    mapping={}; metrics={}; errors=[]
    for coarse_id in coarse['material']:
        ids=np.flatnonzero(coarse['regions'].ravel()==coarse_id)
        sample=ids if len(ids)<=512 else rng.choice(ids,512,replace=False)
        distance,index=tree.query(coarse['xyz'][sample],k=1)
        # Sampling grids have different origins/resolutions; allow half-diagonal
        # of a coarse voxel, plus a small numerical margin.
        near=distance<=max(.90*coarse['voxel'],fine['voxel'])
        candidates=fine_labels[all_fine[index[near]]]
        candidates=candidates[np.isin(candidates,fine['material'])]
        if len(candidates)<20:
            errors.append(f'Coarse region {coarse_id}: fewer than 20 nearby fine material matches')
            continue
        values,counts=np.unique(candidates,return_counts=True)
        i=np.argmax(counts)
        dominant=int(values[i]); fraction=float(counts[i]/counts.sum())
        metrics[str(coarse_id)]={'fine_region':dominant,'matched':int(len(candidates)),
                                  'agreement':fraction}
        mapping[coarse_id]=dominant
        if fraction<.90:
            errors.append(f'Coarse region {coarse_id}: ambiguous fine-grid correspondence')
    if len(mapping)==len(coarse['material']):
        if set(mapping.values()) != set(fine['material']) or len(set(mapping.values()))!=len(mapping):
            errors.append('Coarse/fine material spatial regions do not have a one-to-one correspondence')
        else:
            cl=coarse['gate_summary']['graph']['material_region_labels']
            fl=fine['gate_summary']['graph']['material_region_labels']
            # Equality of *relative* XOR labels: inversion of all labels is okay.
            pairs=list(mapping.items())
            for c,f in pairs[1:]:
                c0,f0=pairs[0]
                if (int(cl[str(c)])^int(cl[str(c0)])) != (int(fl[str(f)])^int(fl[str(f0)])):
                    errors.append('Coarse/fine original-triangle side labels disagree')
                    break
    return {'correspondence':metrics,'errors':errors}


def evaluate(*, audit_dir, audit_record, field, wall_scene, o3d, normals):
    """Return a fail-closed numeric verdict, with reproducible evidence.

    ``audit_record`` must already be validated against the current manifest,
    R3 handoff and ORIGINAL global Poisson SHA by the caller.
    """
    root=Path(audit_dir)
    reasons=[]; levels={}; grids={}
    for level in ('coarse','fine'):
        r=audit_record.get('flood',{}).get(level)
        if not r or r.get('status') not in (
                'NO_TOP_TO_BOTTOM_PATH_SEEN_NOT_PROVEN_SEALED',
                'TOP_TO_BOTTOM_PATH_FOUND_SAMPLED_GRAPH'):
            reasons.append(f'Missing or incomplete {level} sampled flood audit')
            continue
        if r.get('grid',{}).get('roi_touching_audit_window_samples',1):
            reasons.append(f'{level} R3 touches truncated sampling window')
            continue
        p=root/(level+'_connectivity.npz')
        if not p.is_file():
            reasons.append(f'{level} flood grid missing')
            continue
        g=_load(p,level)
        if tuple(r['grid']['shape'])!=g['shape'] or not np.isclose(float(r['grid']['voxel']),g['voxel'],atol=1e-10,rtol=1e-10):
            reasons.append(f'{level} flood grid does not match geometry report')
            continue
        g['gate_summary']=_check_level(g,field,wall_scene,o3d,normals)
        grids[level]=g
        levels[level]=g['gate_summary']
        reasons.extend(f'{level}: {x}' for x in g['gate_summary']['errors'])
    cross={}
    if len(grids)==2 and not reasons:
        cross=_cross_scale(grids['coarse'],grids['fine'])
        reasons.extend(cross['errors'])
    return {'schema':SCHEMA,
            'status':'SUPPORTED_NUMERIC_TWO_SIDED_PARTITION_NOT_SEALED' if not reasons else 'BLOCKED_INSUFFICIENT_OR_INCONSISTENT_PARTITION',
            'old_top_bottom_status':{l:audit_record.get('flood',{}).get(l,{}).get('status') for l in ('coarse','fine')},
            'levels':levels,'cross_scale':cross,'blocking_reasons':reasons,
            'limitations':('Numeric sampled evidence only, not an analytic watertightness certificate. '
                           'A highest-shell exterior seed is a hypothesis until researcher verification. '
                           'No Gaussian is deleted during this check.')}
