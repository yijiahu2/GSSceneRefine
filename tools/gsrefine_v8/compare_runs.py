#!/usr/bin/env python3
"""Compare the same ROI/method before/after engineering v8, without changing runs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def load(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def compare(old,new,old_cleanup,new_cleanup):
    old,new=Path(old),Path(new)
    report={"scope":"exact_same_source_and_roi_only", "stage1":{},"stage2":{}}
    for tag,folder in [('old',old),('new',new)]:
        manifest=load(folder/'metadata'/'manifest.json')
        report['stage1'][tag]={
            'source_gaussian_sha256':manifest['input_sha256']['gaussian'],
            'roi_sha256':manifest['input_sha256']['roi'],
            'poisson_sha256':manifest['input_sha256']['poisson'],
            'r3_ply_sha256':digest(folder/'final'/'gaussian_roi.ply'),
        }
        timing=folder/'metadata'/'stage1_timing.json'
        if timing.is_file():report['stage1'][tag]['timing']=load(timing)
    for name in ('source_gaussian_sha256','roi_sha256','poisson_sha256'):
        if report['stage1']['old'][name]!=report['stage1']['new'][name]:
            raise ValueError(name+' differs: NOT a valid same-input comparison')
    report['stage1']['r3_byte_identical']=(report['stage1']['old']['r3_ply_sha256']==
                                          report['stage1']['new']['r3_ply_sha256'])
    if old_cleanup and new_cleanup:
        for tag,path in [('old',old_cleanup),('new',new_cleanup)]:
            p=Path(path)
            r=load(p/'stage2_final_report.json')
            report['stage2'][tag]={
                'final_ply_sha256':digest(Path(r['outputs']['final_supersplat_gaussian_ply'])),
                'counts':r.get('counts',{}),
                'stage2_timing':r.get('stage2_timing',{}),
                'spatial_classifier':r.get('method',{}).get('spatial_classifier'),
            }
            mask=p/'diagnostics'/'final_classification_masks.npz'
            if mask.is_file():
                with np.load(mask,allow_pickle=False) as f:
                    report['stage2'][tag]['remove_mask_sha256']=hashlib.sha256(
                        f['remove'].tobytes()).hexdigest()
        report['stage2']['clean_ply_byte_identical']=(report['stage2']['old']['final_ply_sha256']==
                                                       report['stage2']['new']['final_ply_sha256'])
        if all('remove_mask_sha256' in report['stage2'][k] for k in ('old','new')):
            report['stage2']['delete_mask_identical']=(report['stage2']['old']['remove_mask_sha256']==
                                                       report['stage2']['new']['remove_mask_sha256'])
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--old-roi-run',required=True)
    parser.add_argument('--new-roi-run',required=True)
    parser.add_argument('--old-cleanup')
    parser.add_argument('--new-cleanup')
    parser.add_argument('--output',default='engineering_v8_comparison.json')
    a=parser.parse_args()
    record=compare(a.old_roi_run,a.new_roi_run,a.old_cleanup,a.new_cleanup)
    Path(a.output).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
    print('Comparison saved:',a.output)
    print('R3 PLY bytes identical:',record['stage1']['r3_byte_identical'])
    print('Stage2:',{k:v for k,v in record['stage2'].items() if not isinstance(v,dict)})
