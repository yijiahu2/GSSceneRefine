"""Replace Stage-2 sampling-window code and every production consuming interface.

The installer updates files in-place using validated AST function locations,
compiles ALL results before writing, preserves backups outside the source tree,
and invalidates pre-V3 geometry-audit caches. It does not edit any ROI, Gaussian,
SfM or Poisson research data. Usage: python tools/install_stage2_window_v3.py --check
       python tools/install_stage2_window_v3.py
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile

REVISION='GSSceneRefine.stage2_sampling_window.v3_20261112'
OLD_SCHEMA='GSSceneRefine.final_R3_global_Poisson_barrier_audit.v1'
NEW_SCHEMA='GSSceneRefine.final_R3_global_Poisson_barrier_audit.v2'
OLD_COMPLETE='FLOOD_SAMPLING_COMPLETED_NOT_CERTIFIED_SEALED'
NEW_COMPLETE='REGION_GRAPHS_COMPLETED_NUMERIC_NOT_CERTIFIED'
NEW_AUDIT_VERSION='final_R3_original_global_poisson_region_graph_v3_window_20261112'

WINDOW_HELPERS = '# Stage-2 audit-window implementation V3. The installer embeds this code directly\n# into tools/audit_final_r3_global_poisson_barrier.py; this file is not imported.\n\ndef _r3_shell_probe(field, lo, hi, *, target_voxel, max_cells, batch=2048):\n    """Check exactly the outermost voxel centres of the production grid.\n\n    Counts touch points once per unique shell centre.  Per-side counts may\n    overlap at edges/corners; this is intentional and is not a Gaussian mask.\n    """\n    low=np.asarray(lo, dtype=np.float64)\n    high=np.asarray(hi, dtype=np.float64)\n    h, shape=grid_spec(low,high,target_voxel,max_cells)\n    shape=tuple(int(n) for n in shape)\n    boundary=np.zeros(shape,dtype=bool)\n    boundary[0,:,:]=True;boundary[-1,:,:]=True\n    boundary[:,0,:]=True;boundary[:,-1,:]=True\n    boundary[:,:,0]=True;boundary[:,:,-1]=True\n    ijk=np.argwhere(boundary)\n    xyz=low+(ijk.astype(np.float64)+0.5)*h\n    inside=eval_r3(field,xyz,batch=batch)>0\n    hits=ijk[inside]\n    sides={}\n    for axis,letter in enumerate(\'xyz\'):\n        sides[letter+\'_min\']=int(np.count_nonzero(hits[:,axis]==0))\n        sides[letter+\'_max\']=int(np.count_nonzero(hits[:,axis]==shape[axis]-1))\n    return {\'voxel\':float(h),\'shape\':list(shape),\n            \'shell_nodes_checked\':int(len(ijk)),\n            \'r3_boundary_touch_count\':int(inside.sum()),\'touches_by_side\':sides}\n\n\ndef complete_window_for_levels(field, lo, hi, *, levels, batch=2048,\n                               max_rounds=12, growth_voxels=3.0,\n                               max_voxel_scale=1.25, logger=None):\n    """Expand only the *numerical* AABB, preserving ROI and Poisson inputs.\n\n    Both levels share the same final AABB. Normal cases retain bit-identical\n    bounds. A grid cap may enlarge actual voxel size: prevent >25% upsampling\n    deterioration relative to the original scene-specific grid specification.\n    Fail closed if the window is unresolved within the expansion budget.\n    """\n    lower=np.asarray(lo,dtype=np.float64).copy()\n    upper=np.asarray(hi,dtype=np.float64).copy()\n    if (lower.shape!=(3,) or upper.shape!=(3,) or\n            not np.isfinite(lower).all() or not np.isfinite(upper).all() or\n            not np.all(upper>lower)):\n        raise ValueError(\'Invalid initial 3D sampling bounds\')\n    if not levels or len({str(x[0]) for x in levels})!=len(levels):\n        raise ValueError(\'Sampling levels must be nonempty and uniquely named\')\n    if not (1<=int(max_rounds)<=64 and 1.0<=float(growth_voxels)<=12.0\n            and 1.0<=float(max_voxel_scale)<=2.0):\n        raise ValueError(\'Invalid sampling-window expansion safeguards\')\n    initial_lower=lower.copy();initial_upper=upper.copy()\n    base={name:grid_spec(lower,upper,float(h),int(cap))[0]\n          for name,h,cap in levels}\n    trace=[]\n    for step in range(int(max_rounds)+1):\n        grow_lo=np.zeros(3,dtype=np.float64)\n        grow_hi=np.zeros(3,dtype=np.float64)\n        results={}\n        for name,target,cap in levels:\n            info=_r3_shell_probe(field,lower,upper,target_voxel=float(target),\n                                  max_cells=int(cap),batch=batch)\n            h=float(info[\'voxel\'])\n            if h>base[name]*float(max_voxel_scale)+1e-10:\n                raise RuntimeError(\n                    f\'{name} grid resolution would degrade from {base[name]:.6g} to \'\n                    f\'{h:.6g} (> {max_voxel_scale:.3g}x). Increase max_cells \'\n                    \'instead of silently weakening spatial validation.\')\n            results[name]=info\n            for axis,letter in enumerate(\'xyz\'):\n                if info[\'touches_by_side\'][letter+\'_min\']:\n                    grow_lo[axis]=max(grow_lo[axis],float(growth_voxels)*h)\n                if info[\'touches_by_side\'][letter+\'_max\']:\n                    grow_hi[axis]=max(grow_hi[axis],float(growth_voxels)*h)\n        touched=sum(x[\'r3_boundary_touch_count\'] for x in results.values())\n        trace.append({\'attempt\':step+1,\'low\':lower.tolist(),\n                      \'high\':upper.tolist(),\'levels\':results,\'total_touches\':touched})\n        if logger:\n            logger(\'3D sampling window: \'+\', \'.join(\n                f\'{name} boundary-R3={results[name]["r3_boundary_touch_count"]}\'\n                f\' (h={results[name]["voxel"]:.6f})\'\n                for name,_,_ in levels)+f\'; expansion round={step}\')\n        if touched==0:\n            return lower,upper,{\n                \'schema\':\'GSSceneRefine.window_cover.v3\',\n                \'status\':\'COMPLETE_SAMPLED_WINDOW\',\n                \'expanded\':bool(step),\'expansion_rounds\':int(step),\n                \'initial_low\':initial_lower.tolist(),\n                \'initial_high\':initial_upper.tolist(),\n                \'final_low\':lower.tolist(),\'final_high\':upper.tolist(),\n                \'initial_voxel_sizes\':{k:float(v) for k,v in base.items()},\n                \'levels\':results,\'trace\':trace,\n                \'scope\':\'Numerical sampling box only; R3 and Poisson unchanged\',\n                \'limitation\':\'Zero sampled shell contacts is not an analytic proof of full ROI coverage.\'}\n        if step==int(max_rounds):\n            raise RuntimeError(\n                f\'R3 still touches the sampling box after {max_rounds} expansions: \'\n                +str({name:info[\'touches_by_side\'] for name,info in results.items()}))\n        lower-=grow_lo;upper+=grow_hi\n        if not np.isfinite(lower).all() or not np.isfinite(upper).all():\n            raise RuntimeError(\'Sampling window expanded to nonfinite values\')\n'


def change_once(source, old, new, name):
    n=source.count(old)
    if n!=1:
        raise RuntimeError(f'{name}: expected exactly one source anchor; found {n}. No files changed.')
    return source.replace(old,new,1)


def def_bounds(source, name):
    module=ast.parse(source)
    targets=[n for n in module.body if isinstance(n, (ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name]
    if len(targets)!=1:
        raise RuntimeError(f'Expected exactly one top-level function {name}')
    return targets[0].lineno,targets[0].end_lineno


def update_audit(source):
    if NEW_AUDIT_VERSION in source:
        if 'def complete_window_for_levels' not in source:
            raise RuntimeError('Half-installed V3 audit detected: abort')
        return source
    source=change_once(source,
        "VERSION = 'final_R3_original_global_poisson_region_graph_v2_20261112'",
        f"VERSION = '{NEW_AUDIT_VERSION}'",'audit version')
    # Inline the complete implementation. No old standalone preflight import survives.
    insert='\n\n'+WINDOW_HELPERS+'\n\n'
    source=change_once(source,'def grid_spec(lo,hi,desired,max_cells):',
                       insert+'def grid_spec(lo,hi,desired,max_cells):',
                       'single canonical sampling-window helper')
    # Remove the old local run() window handling, including optional V1 wrapper.
    a,b=def_bounds(source,'run')
    lines=source.splitlines(keepends=True)
    starts=[i for i in range(a-1,b) if re.match(r'^    lo,hi,window=bounding_window\(',lines[i])]
    ends=[i for i in range(a-1,b) if re.match(r'^    for name,desired,cap in \[',lines[i])]
    if len(starts)!=1 or len(ends)!=1 or ends[0]<=starts[0]:
        raise RuntimeError('Cannot uniquely locate run() original sampling window: no files changed')
    body="""    lo,hi,window=bounding_window(initial,ply.positions(),args.padding)
    planned_levels=[('coarse',args.coarse_voxel,args.coarse_max_cells),
                    ('fine',args.fine_voxel,args.fine_max_cells)]
    planned_levels=[x for x in planned_levels if args.only_scale in (x[0],'both')]
    initial_window_low=lo.tolist(); initial_window_high=hi.tolist()
    try:
        lo,hi,window_fit=complete_window_for_levels(
            r3_fn,lo,hi,levels=planned_levels,batch=args.eval_batch,
            max_rounds=getattr(args,'window_max_rounds',12),
            growth_voxels=getattr(args,'window_growth_voxels',3.0),
            max_voxel_scale=getattr(args,'window_max_voxel_scale',1.25),logger=log)
    except Exception as exc:
        result['status']='BLOCKED_INCOMPLETE_SAMPLING_WINDOW'
        result['diagnostic_window']={**window,
            'initial_low':initial_window_low,'initial_high':initial_window_high,
            'status':'BLOCKED_WINDOW_CHECK','error':type(exc).__name__+': '+str(exc)}
        atomic_json(output/'roi_global_poisson_barrier_report.json',result)
        raise
    result['diagnostic_window']={'low':lo.tolist(),'high':hi.tolist(),
                                  **window,'window_cover':window_fit}
    atomic_json(output/'roi_global_poisson_barrier_report.json',result)
"""
    source=''.join(lines[:starts[0]])+body+''.join(lines[ends[0]:])
    source=change_once(source,
        '    touches=int(v[0].sum()+v[-1].sum()+v[:,0].sum()+v[:,-1].sum()+v[:,:,0].sum()+v[:,:,-1].sum())\n',
        '''    touches=int(v[0].sum()+v[-1].sum()+v[:,0].sum()+v[:,-1].sum()+v[:,:,0].sum()+v[:,:,-1].sum())
    if touches:
        raise RuntimeError(f'{name} sampled box still intersects R3 at {touches} outer nodes; '
                           'refuse to construct an incomplete region graph')
''','flood_test fail-closed boundary check')
    source=change_once(source,
        "    status='WINDOW_TRUNCATED' if touches else 'REGION_GRAPH_SAMPLED_NOT_CERTIFIED' ",
        "    status='REGION_GRAPH_SAMPLED_NOT_CERTIFIED'",
        'remove obsolete WINDOW_TRUNCATED status')
    source=change_once(source,"'roi_touching_audit_window_samples':touches",
        "'r3_boundary_touch_count':touches",'new grid boundary field')
    source=change_once(source,"'schema':'"+OLD_SCHEMA+"'",
        "'schema':'"+NEW_SCHEMA+"'",'audit schema v2')
    source=change_once(source,"result['status']='"+OLD_COMPLETE+"'",
        "result['status']='"+NEW_COMPLETE+"'",'audit completion status')
    source=change_once(source,'    a=p.parse_args(argv)',
        '''    p.add_argument('--window-max-rounds',type=int,default=12)
    p.add_argument('--window-growth-voxels',type=float,default=3.0)
    p.add_argument('--window-max-voxel-scale',type=float,default=1.25)
    a=p.parse_args(argv)''','CLI option controls')
    ast.parse(source)
    return source


def update_gate(s):
    old="""        if not r or r.get('status') not in (
                'REGION_GRAPH_SAMPLED_NOT_CERTIFIED',
                # Historical completed audits: geometry graph is reusable,
                # but no legacy top/bottom judgment is used for acceptance.
                'NO_TOP_TO_BOTTOM_PATH_SEEN_NOT_PROVEN_SEALED',
                'TOP_TO_BOTTOM_PATH_FOUND_SAMPLED_GRAPH'):
"""
    s=change_once(s,old,
        "        if not r or r.get('status') != 'REGION_GRAPH_SAMPLED_NOT_CERTIFIED':\n",
        'region gate only V3 sampled graph')
    s=change_once(s,"'roi_touching_audit_window_samples',1",
                  "'r3_boundary_touch_count',1",'region gate new grid field')
    ast.parse(s);return s


def update_parity(s):
    s=change_once(s,"if data.get('schema') != '"+OLD_SCHEMA+"':",
        "if data.get('schema') != '"+NEW_SCHEMA+"':",
        'parity audit schema')
    s=change_once(s,"if data.get('status') != '"+OLD_COMPLETE+"':",
        "if data.get('status') != '"+NEW_COMPLETE+"':",
        'parity audit status')
    needle="    inp = data['inputs']"
    replacement="""    if data.get('version') != '"""+NEW_AUDIT_VERSION+"""':
        raise ValueError('Old sampling audit version; regenerate both spatial grids')
    if data.get('diagnostic_window', {}).get('window_cover', {}).get('status') != 'COMPLETE_SAMPLED_WINDOW':
        raise ValueError('R3 sampling window has not passed the new coverage check')
    inp = data['inputs']"""
    # only inside _audit_identity
    a,b=def_bounds(s,'_audit_identity')
    lines=s.splitlines(keepends=True)
    span=''.join(lines[a-1:b])
    span=change_once(span,needle,replacement,'parity window provenance')
    s=''.join(lines[:a-1])+span+''.join(lines[b:])
    # Include audit version in navigation cache identity: even an unchanged
    # numeric grid cannot reuse an index from the old sampling-window contract.
    a,b=def_bounds(s,'_navigation_identity')
    lines=s.splitlines(keepends=True)
    span=''.join(lines[a-1:b])
    span=change_once(span,"        'schema': VERSION,",
                     "        'schema': VERSION,\n        'audit_version': audit_report.get('version','unversioned'),",
                     'navigation index audit version')
    s=''.join(lines[:a-1])+span+''.join(lines[b:])
    # explicit runtime config so there is exactly one active interface.
    s=change_once(s,"roi_link_samples=roi_link_samples, top_quantile=.9, only_scale='both'))",
        "roi_link_samples=roi_link_samples, top_quantile=.9, only_scale='both',\n"
        "            window_max_rounds=12,window_growth_voxels=3.0,\n"
        "            window_max_voxel_scale=1.25))",
        'parity call new window arguments')
    ast.parse(s);return s


def update_onepass(s):
    fragment="""            if (data['status']!='"""+OLD_COMPLETE+"""'
                or inp['manifest_sha256']!=mh or inp['r3_sha256']!=rh
                or inp['original_poisson_sha256']!=ph):
                continue"""
    replacement="""            if (data.get('schema')!='"""+NEW_SCHEMA+"""'
                or data.get('version')!='"""+NEW_AUDIT_VERSION+"""'
                or data.get('status')!='"""+NEW_COMPLETE+"""'
                or data.get('diagnostic_window',{}).get('window_cover',{}).get('status')!='COMPLETE_SAMPLED_WINDOW'
                or inp['manifest_sha256']!=mh or inp['r3_sha256']!=rh
                or inp['original_poisson_sha256']!=ph):
                continue"""
    s=change_once(s,fragment,replacement,'onepass cache source and schema validation')
    s=change_once(s,"x['grid']['roi_touching_audit_window_samples']",
                  "x['grid']['r3_boundary_touch_count']",'onepass cache boundary check')
    s=change_once(s,'"""Reuse source-identical sampled region graphs; NEVER test top/bottom."""',
                  '"""Reuse only source-identical V3 bounded two-resolution region graphs."""',
                  'onepass audit cache description')
    ast.parse(s);return s


def update_contract_test(s):
    s=change_once(s,
        '''    assert "status='WINDOW_TRUNCATED' if touches else 'REGION_GRAPH_SAMPLED_NOT_CERTIFIED'" in s''',
        '''    assert "status='REGION_GRAPH_SAMPLED_NOT_CERTIFIED'" in s
    assert 'def complete_window_for_levels' in s
    assert 'BLOCKED_INCOMPLETE_SAMPLING_WINDOW' in s
    assert "'r3_boundary_touch_count':touches" in s''',
        'legacy source contract test')
    ast.parse(s);return s


def update_gate_test(s):
    s=change_once(s,"'status':'TOP_TO_BOTTOM_PATH_FOUND_SAMPLED_GRAPH'",
                  "'status':'REGION_GRAPH_SAMPLED_NOT_CERTIFIED'",
                  'synthetic gate status')
    s=change_once(s,"'roi_touching_audit_window_samples':0",
                  "'r3_boundary_touch_count':0",'synthetic gate boundary count')
    ast.parse(s);return s


def update_onepass_test(s):
    s=change_once(s,"report={'status':'"+OLD_COMPLETE+"',",
        "report={'schema':'"+NEW_SCHEMA+"',\n"
        "            'version':'"+NEW_AUDIT_VERSION+"',\n"
        "            'status':'"+NEW_COMPLETE+"',\n"
        "            'diagnostic_window':{'window_cover':{'status':'COMPLETE_SAMPLED_WINDOW'}},",
        'onepass synthetic report fixture')
    if s.count("'roi_touching_audit_window_samples'")!=4:
        raise RuntimeError('Synthetic cache tests changed: expected four old boundary fields')
    s=s.replace("'roi_touching_audit_window_samples'","'r3_boundary_touch_count'")
    # V3 must not accept old top-bottom cache status, even with the same hashes.
    s=change_once(s,
        "assert mod._find_barrier_audit(root,manifest,handoff,poisson)==folder\n    # Actual truncated sampling window",
        "assert mod._find_barrier_audit(root,manifest,handoff,poisson) is None\n"
        "    report['flood']['coarse']['status']='REGION_GRAPH_SAMPLED_NOT_CERTIFIED'\n"
        "    report_path.write_text(json.dumps(report))\n"
        "    # Actual truncated sampling window",'cache rejects obsolete top/bottom status')
    ast.parse(s);return s


def sha256(b): return hashlib.sha256(b).hexdigest()


def changes(root):
    changes={}
    transforms={
        'tools/audit_final_r3_global_poisson_barrier.py':update_audit,
        'gsrefine/cleanup/barrier_separation_gate.py':update_gate,
        'gsrefine/cleanup/roi_barrier_parity.py':update_parity,
        'gsrefine/cleanup/stage2_onepass.py':update_onepass,
        'tests/test_stage2_patch_source_contract.py':update_contract_test,
        'tests/test_barrier_separation_gate_v2.py':update_gate_test,
        'tests/test_stage2_onepass.py':update_onepass_test,
    }
    all_done=NEW_AUDIT_VERSION in (root/'tools/audit_final_r3_global_poisson_barrier.py').read_text(encoding='utf8')
    completed_markers={
        'tools/audit_final_r3_global_poisson_barrier.py':'def complete_window_for_levels',
        'gsrefine/cleanup/barrier_separation_gate.py':"'r3_boundary_touch_count',1",
        'gsrefine/cleanup/roi_barrier_parity.py':NEW_SCHEMA,
        'gsrefine/cleanup/stage2_onepass.py':NEW_SCHEMA,
        'tests/test_stage2_patch_source_contract.py':'BLOCKED_INCOMPLETE_SAMPLING_WINDOW',
        'tests/test_barrier_separation_gate_v2.py':"'r3_boundary_touch_count':0",
        'tests/test_stage2_onepass.py':NEW_SCHEMA,
    }
    for name,fn in transforms.items():
        path=root/name
        source=path.read_text(encoding='utf8')
        if all_done:
            if completed_markers[name] not in source:
                raise RuntimeError('Partial V3 installation detected in '+name)
            continue
        updated=fn(source)
        compile(updated,str(path),'exec')
        if updated!=source:
            changes[path]=(source.encode('utf8'),updated.encode('utf8'))
    return changes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    root=args.project_root.expanduser().resolve(strict=True)
    edits=changes(root)
    print('V3 sampling-window integration verified; files to update:',len(edits))
    for path,(before,after) in edits.items():
        print(' ',path.relative_to(root),'(',len(before),'->',len(after),'bytes )')
    if args.check or not edits:
        return
    backup=root/'.git'/'gsrefine_patch_backups'/'stage2_sampling_window_v3'
    backup.mkdir(parents=True,exist_ok=True)
    # All transformed code is compiled above. Save all originals before first edit.
    for path,(before,_) in edits.items():
        destination=backup/path.relative_to(root)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            if destination.read_bytes()!=before:
                raise FileExistsError('Incompatible existing backup: '+str(destination))
        else:
            destination.write_bytes(before)
    installed=[]
    try:
        for path,(_,after) in edits.items():
            fd,temp=tempfile.mkstemp(prefix='.stage2_v3_',suffix='.py',dir=path.parent)
            try:
                with os.fdopen(fd,'wb') as out:
                    out.write(after);out.flush();os.fsync(out.fileno())
                os.chmod(temp,path.stat().st_mode)
                os.replace(temp,path)
                installed.append(path)
            finally:
                Path(temp).unlink(missing_ok=True)
    except BaseException:
        for path in reversed(installed):
            path.write_bytes(edits[path][0])
        raise
    # Old optional standalone preflight is superseded by inline canonical code.
    # Remove only byte-identical copies of our previous distributed files.
    obsolete={'tools/r3_audit_window_preflight.py':None,
              'tools/install_r3_audit_window_preflight.py':None}
    for rel in obsolete:
        source=root/rel
        if source.exists():
            # Do not silently delete unrelated local user modifications.
            ref=OLD_STANDALONE_HASHES.get(rel)
            if ref is not None and sha256(source.read_bytes())==ref:
                dest=backup/rel
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,dest)
                source.unlink()
                print(' Removed obsolete unmodified standalone runtime:',rel)
            else:
                print(' NOTE: obsolete standalone file present but locally modified; not imported by V3:',rel)
    print('Stage-2 V3 sampling window and all consuming interfaces installed.')
    print('Backup:',backup)


OLD_STANDALONE_HASHES={'tools/r3_audit_window_preflight.py': 'f21fd606c054b43fbdf7c16fa8aba5017fd1f30fadcf0843a7213c6fd052f114', 'tools/install_r3_audit_window_preflight.py': 'aa9d2cfbd79da7607bf258d97e231b3efb291770b04c945c4c234e3b475da4d9'}

if __name__=='__main__':
    main()
