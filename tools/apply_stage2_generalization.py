#!/usr/bin/env python3
"""In-place, guarded patch against the 2026-11-12 sealless GSSceneRefine baseline.

Changes are intentionally applied to installed latest files (not restored from
11 Nov V2 snapshot), protecting the 12 Nov sealless improvements.
"""
from __future__ import annotations
import ast
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
MARK = 'stage2_region_unified_original_main_v1_20261112'


def exact(s, old, new, desc):
    count=s.count(old)
    if count!=1:
        raise RuntimeError(f'{desc}: expected one source match, found {count}; no files changed')
    return s.replace(old,new,1)


def function_replace(s, start, stop, body):
    a=s.find(start)
    b=s.find(stop,a+len(start))
    if a<0 or b<0:
        raise RuntimeError(f'Unable to locate function bounds: {start} ... {stop}')
    return s[:a]+body.rstrip()+'\n\n\n'+s[b:]


def update_audit(s):
    old='''    top,bottom,seeds=choose_top_bottom_seeds(passable,xyz,shape,up,quantile=seed_quantile)
    top_l=set(map(int,np.unique(labels[top]))) - {-1}
    bottom_l=set(map(int,np.unique(labels[bottom]))) - {-1}
    both=sorted(top_l&bottom_l)
    reach_top=np.isin(labels,list(top_l))&passable
    reach_bottom=np.isin(labels,list(bottom_l))&passable
    witness=np.empty((0,3),float)
    if both:
        witness=_path_witness(len(xyz),src,dst,top,bottom,xyz)
        if len(witness):np.savetxt(Path(output_dir)/f'{name}_leak_path.xyz',witness,fmt='%.8f')
    status=('WINDOW_TRUNCATED' if touches else
            'TOP_TO_BOTTOM_PATH_FOUND_SAMPLED_GRAPH' if both else
            'NO_TOP_TO_BOTTOM_PATH_SEEN_NOT_PROVEN_SEALED')'''
    new='''    # True spatial components are the evidence; arbitrary top/bottom probe
    # connectivity is NOT a separation criterion.  Upper-shell reachability
    # remains a diagnostic only, never a pass/fail test.
    top, _, seeds=choose_top_bottom_seeds(passable,xyz,shape,up,quantile=seed_quantile)
    top_l=set(map(int,np.unique(labels[top]))) - {-1}
    reach_top=np.isin(labels,list(top_l))&passable
    status='WINDOW_TRUNCATED' if touches else 'REGION_GRAPH_SAMPLED_NOT_CERTIFIED' '''
    s=exact(s,old,new,'remove top-bottom flood verdict')
    s=exact(s,'''                        reachable_from_top=reach_top.reshape(shape),
                        reachable_from_lower_probe=reach_bottom.reshape(shape),
                        components=labels.reshape(shape))''','''                        reachable_from_top=reach_top.reshape(shape),
                        components=labels.reshape(shape))''','drop lower probe array')
    # Match the known connectivity fields by structure, not whitespace.
    # In the November 12 repository the indentation differs from the original
    # patch's literal (the failure reported by the user).
    import re
    pattern = (r"(?m)^(?P<indent>[ \t]*)'upper_shell_connected_regions':sorted\(top_l\),\n"
               r"(?P=indent)'lower_shell_connected_regions':sorted\(bottom_l\),\n"
               r"(?P=indent)'regions_connecting_upper_and_lower':both,\n"
               r"(?P=indent)'reachable_from_upper':int\(reach_top.sum\(\)\),\n"
               r"(?P=indent)'lower_probe_reached_from_upper':bool\(both\),\n"
               r"(?P=indent)'sampled_path_vertices':int\(len\(witness\)\)\},")
    matches=list(re.finditer(pattern,s))
    if len(matches)!=1:
        raise RuntimeError('remove top/bottom result fields: expected one structural match, '
                           f'found {len(matches)}; no files changed')
    m=matches[0]
    indent=m.group('indent')
    s=(s[:m.start()] + indent+"'upper_shell_connected_regions':sorted(top_l),\n"
       +indent+"'reachable_from_upper':int(reach_top.sum())},"+s[m.end():])
    s=exact(s,'''            'limitations':'A found path is a sampled leak witness after segment-triangle checks. '
                          'No found path does not mathematically certify a sealed ROI. '
                          'The lower shell is a hypothesis, not a validated rock-inside seed. '
                          'R3 segment membership is probed at finitely many positions.'}''','''            'limitations':'Sampled space components do not mathematically certify an exact partition. '
                          'The upper shell is not automatically a validated exterior seed. '
                          'R3 segment membership is probed at finitely many positions.'}''','remove obsolete audit notes')
    s=exact(s,'''    result['interpretation']={
        'found_leak_path_at_any_scale':any(x['connectivity']['lower_probe_reached_from_upper']
                                           for x in result.get('flood',{}).values()),
        'warning':'Even if all levels find no leak, discretized water flooding does NOT prove '
                  'topological sealing, and lower ROI shell is not an observed inside-rock seed. '
                  'Do not use audit as automatic Gaussian deletion without separately confirmed partition.'}''','''    result['interpretation']={
        'region_graph_only': True,
        'warning':'Sampled connectivity does not prove a closed separator; accept or reject only '
                  'after original-triangle region-parity and cross-scale checks. '
                  'Exterior seed still requires review.'}''','remove obsolete interpretation')
    s=s.replace("VERSION = 'final_R3_original_global_poisson_intersection_flood_v1_20261110'", "VERSION = 'final_R3_original_global_poisson_region_graph_v2_20261112'")
    return s


def update_gate(s):
    s=exact(s,'''                'NO_TOP_TO_BOTTOM_PATH_SEEN_NOT_PROVEN_SEALED',
                'TOP_TO_BOTTOM_PATH_FOUND_SAMPLED_GRAPH'):''','''                'REGION_GRAPH_SAMPLED_NOT_CERTIFIED',
                # Historical completed audits: geometry graph is reusable,
                # but no legacy top/bottom judgment is used for acceptance.
                'NO_TOP_TO_BOTTOM_PATH_SEEN_NOT_PROVEN_SEALED',
                'TOP_TO_BOTTOM_PATH_FOUND_SAMPLED_GRAPH'):''','gate accepts new region graph status')
    s=exact(s,"'old_top_bottom_status':{l:audit_record.get('flood',{}).get(l,{}).get('status') for l in ('coarse','fine')},\n            ","",'remove legacy status from verdict')
    return s


def update_parity(s):
    a=s.index('def _audit_identity(')
    b=s.index('\nclass ParityUnionFind:',a)
    block='''def _audit_identity(audit_dir, manifest_path, inputs, r3_record, *,
                    field, wall_scene, o3d, normals, output):
    root = Path(audit_dir).expanduser().resolve(strict=True)
    report = root / 'roi_global_poisson_barrier_report.json'
    data = json.loads(report.read_text(encoding='utf-8'))
    if data.get('schema') != 'GSSceneRefine.final_R3_global_Poisson_barrier_audit.v1':
        raise ValueError('Not a verified final R3/global Poisson barrier audit')
    if data.get('status') != 'FLOOD_SAMPLING_COMPLETED_NOT_CERTIFIED_SEALED':
        raise ValueError('Incomplete two-resolution spatial component audit')
    inp = data['inputs']
    required = {'manifest_sha256': _sha(manifest_path),
                'r3_sha256': _sha(r3_record),
                'original_poisson_sha256': _sha(inputs['poisson_mesh'])}
    for key, value in required.items():
        if inp.get(key) != value:
            raise ValueError(f'Barrier audit source mismatch: {key}')
    if str(Path(inp['final_roi_gaussian']).resolve()) != str(Path(inputs['gaussian_ply']).resolve()):
        raise ValueError('Audit Gaussian PLY is from a different ROI run')
    if data.get('r3_verified_replay', {}).get('v3_handoff_max_field_distance_error', float('inf')) > 1e-6:
        raise ValueError('R3 replay consistency check failed')
    loops = _validate_intersection_loop(root / 'global_poisson_final_r3_intersection.npz')
    from gsrefine.cleanup.barrier_separation_gate import evaluate
    verdict = evaluate(audit_dir=root, audit_record=data, field=field,
                       wall_scene=wall_scene, o3d=o3d, normals=normals)
    _write_json(Path(output) / 'region_separation_gate.json', verdict)
    if verdict['status'] != 'SUPPORTED_NUMERIC_TWO_SIDED_PARTITION_NOT_SEALED':
        raise ValueError('R3 / original main Poisson region separation failed: '
                         + '; '.join(verdict['blocking_reasons'])
                         + f'. Details: {Path(output) / "region_separation_gate.json"}')
    return data, loops, verdict
'''
    s=s[:a]+block+'\n\n'+s[b+1:]
    s=exact(s,'''    audit, loops = _audit_identity(report_dir, stage1, inputs, handoff)
    _log(f'Audit accepted for sampled navigation: {loops["closed_cycles"]} closed intersection cycle(s)')''','''    audit, loops, separation = _audit_identity(
        report_dir, stage1, inputs, handoff, field=field, wall_scene=wall_scene,
        o3d=o3d, normals=normals, output=output)
    _log(f'Original-triangle region separation confirmed numerically: '
         f'{loops["closed_cycles"]} closed intersection cycle(s)')''','wire region separation verdict')
    s=exact(s,"VERSION = 'GSSceneRefine.roi_barrier_parity_sdf.v2'", "VERSION = 'GSSceneRefine.roi_barrier_parity_sdf.v3'",'parity schema bump')
    s=exact(s,"'geometry': {'intersection_topology': loops, 'navigation': index['summary'],", "'geometry': {'intersection_topology': loops, 'region_separation': separation, 'navigation': index['summary'],",'report gate provenance')
    return s


def update_onepass(s):
    a=s.index('def _find_barrier_audit(')
    b=s.index('\ndef _validated_parity(',a)
    new='''def _find_barrier_audit(roi_run, stage1_manifest, r3_handoff, original_poisson):
    """Reuse source-identical sampled region graphs; NEVER test top/bottom."""
    root=Path(roi_run)/'cleanup'
    if not root.is_dir():
        return None
    mh,rh,ph=_sha(stage1_manifest),_sha(r3_handoff),_sha(original_poisson)
    candidates=[]
    for report_file in root.glob('**/roi_global_poisson_barrier_report.json'):
        try:
            data=_read(report_file)
            inp=data['inputs']
            if (data['status']!='FLOOD_SAMPLING_COMPLETED_NOT_CERTIFIED_SEALED'
                or inp['manifest_sha256']!=mh or inp['r3_sha256']!=rh
                or inp['original_poisson_sha256']!=ph):
                continue
            folder=report_file.parent
            required=('coarse_connectivity.npz','fine_connectivity.npz',
                      'global_poisson_final_r3_intersection.npz')
            if not all((folder/name).is_file() for name in required):
                continue
            coarse,fine=data['flood']['coarse'],data['flood']['fine']
            if any(x['grid']['roi_touching_audit_window_samples'] for x in (coarse,fine)):
                continue
            candidates.append((float(fine['grid']['voxel']),str(folder),folder))
        except (KeyError,TypeError,ValueError,OSError):
            continue
    return sorted(candidates,key=lambda x:(x[0],x[1]))[0][2] if candidates else None


'''
    s=s[:a]+new+s[b+1:]
    s=exact(s,"if report.get('schema') != 'GSSceneRefine.roi_barrier_parity_sdf.v2':", "if report.get('schema') != 'GSSceneRefine.roi_barrier_parity_sdf.v3':", 'require v3 parity')
    s=exact(s,"if data.get('schema') != 'GSSceneRefine.roi_barrier_parity_sdf.v2':", "if data.get('schema') != 'GSSceneRefine.roi_barrier_parity_sdf.v3':", 'discover only v3 parity')
    # Existing canonical 12 Nov sealless preflights must not silently reuse
    # previous source-order geometry after switching to ORIGINAL-first main.
    hook='''def _discover_preseal(manifest, run_dir):'''
    inject='''def _source_order_preseal(report):
    """True only if the source clip was produced by original-main-first v2."""
    try:
        data=_read(report)
        clip=Path(data['direct_r3_clip_report']).resolve(strict=True)
        source=_read(clip)
        return (source.get('source_component_selection') ==
                'largest_original_edge_connected_component_BEFORE_R3_clip')
    except (KeyError, OSError, ValueError, TypeError):
        return False


'''+hook
    s=exact(s,hook,inject,'add source-order preseal check')
    s=exact(s,'''    if Path(data['stage1_manifest']).resolve(strict=True) != Path(manifest).resolve(strict=True):
        raise ValueError('Preseal report does not belong to current Stage-1 manifest')
    return report''','''    if Path(data['stage1_manifest']).resolve(strict=True) != Path(manifest).resolve(strict=True):
        raise ValueError('Preseal report does not belong to current Stage-1 manifest')
    if not _source_order_preseal(report):
        raise ValueError('Old preseal uses clip-before-global-main selection; rebuild geometry')
    return report''','reject explicitly supplied old preseal')
    s=exact(s,'''            validate_parity_geometry(manifest, report)
            accepted.append(report)''','''            validate_parity_geometry(manifest, report)
            if not _source_order_preseal(report):
                continue
            accepted.append(report)''','ignore stale preseal in auto discovery')
    return s


def update_clip(s):
    s=exact(s,"REVISION = 'global_module1_poisson_direct_r3_1x_clip_v1_20261108'", "REVISION = 'global_original_main_then_direct_r3_1x_clip_v2_20261112'",'clip revision')
    old='''    global_mesh, margin_fn, field_report = _restore_r3_field(
        inputs=inputs, upstream=upstream, run_dir=run_dir, audit_record=record,
        audit_samples=audit_samples)
    source_verts = np.asarray(global_mesh.vertices, np.float64)'''
    new='''    global_mesh, margin_fn, field_report = _restore_r3_field(
        inputs=inputs, upstream=upstream, run_dir=run_dir, audit_record=record,
        audit_samples=audit_samples)
    # Stage 2 source order: FIRST extract original-global Poisson largest
    # edge-connected face component; SECOND clip ONLY that original main with
    # FINAL R3; THIRD select the largest post-clip component as the distance
    # surface (in direct_r3_pipeline), recording all post-clip auxiliaries.
    from tools.audit_final_r3_global_poisson_barrier import largest_original_component
    full_global_topology = _topology(global_mesh)
    global_mesh, original_main_info = largest_original_component(global_mesh)
    source_verts = np.asarray(global_mesh.vertices, np.float64)'''
    s=exact(s,old,new,'source original-main extraction before clip')
    s=exact(s,"'global_mesh_topology': _topology(global_mesh),", "'global_mesh_topology': _topology(global_mesh),\n        'original_unfiltered_global_topology': full_global_topology,\n        'original_main_selection': original_main_info,\n        'source_component_selection': 'largest_original_edge_connected_component_BEFORE_R3_clip',\n        'postclip_components_are_split_again': True,", 'record selection provenance')
    # Existing stage-2 main/aux partition remains unchanged (second largest).
    return s


def update_gate_test(s):
    old="    assert all('TOP_TO_BOTTOM_PATH_FOUND' in x for x in r['old_top_bottom_status'].values())"
    if old not in s:
        raise RuntimeError('Expected legacy test assertion not found; no files changed')
    return s.replace(old, "    assert 'old_top_bottom_status' not in r", 1)


def main():
    paths={
        'tools/audit_final_r3_global_poisson_barrier.py':update_audit,
        'gsrefine/cleanup/barrier_separation_gate.py':update_gate,
        'gsrefine/cleanup/roi_barrier_parity.py':update_parity,
        'gsrefine/cleanup/stage2_onepass.py':update_onepass,
        'gsrefine/geometry/r3_direct_poisson_clip.py':update_clip,
        'tests/test_barrier_separation_gate_v2.py':update_gate_test,
    }
    contents={}
    for rel,transform in paths.items():
        p=ROOT/rel
        if not p.is_file():
            raise FileNotFoundError(f'Missing current repository file: {p}')
        old=p.read_text(encoding='utf-8')
        if MARK in old:
            print('[already patched]',rel)
            continue
        modified=transform(old)
        modified+='\n# '+MARK+'\n'
        ast.parse(modified,filename=rel)
        contents[p]=modified
    # Cross-file verification before ANY source change
    needed=['gsrefine/cleanup/stage2_geometry_validation.py',
            'gsrefine/geometry/r3_main_auxiliary.py']
    for rel in needed:
        if not (ROOT/rel).exists():
            raise RuntimeError('This updater requires the 12-Nov-2026 sealless baseline: '+rel)
    if not contents:
        print('All source files already patched; no changes.')
        return
    backup=Path(os.environ.get('GSREFINE_PATCH_BACKUP',
               str(Path.home()/'GSSceneRefine_patch_backup_20261112')))
    backup.mkdir(parents=True,exist_ok=True)
    for p in contents:
        dest=backup/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest)
    for p,value in contents.items():
        fd,tmp=tempfile.mkstemp(dir=p.parent,prefix='.stage2patch-',suffix='.tmp')
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as f:
                f.write(value)
            os.replace(tmp,p)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
        print('[patched]',p.relative_to(ROOT))
    print('Original source backup:',backup)
    print('No Stage-1 data, PLY or old geometry caches were changed.')
    print('Old direct-clip/preseal caches will not be reused automatically.')


if __name__=='__main__':
    main()
