#!/usr/bin/env python3
"""Generate two explicit orthographic polygons from OLD Task-1 AABB for code sanity checks.

They are NOT user-drawn polygons and NOT independent evidence of ROI quality.
Use --inset-z-top to create a controlled top-cut stress test.
"""
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True,help='Stage-1 reconstruction_report.json with roi_lower/roi_upper')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--inset-z-top',type=float,default=0,help='Reduce front polygon max Z by this fraction of ROI Z range, 0..0.25')
    p.add_argument('--inset-x-right',type=float,default=0,help='Reduce both polygon max X by this fraction of ROI X range, 0..0.25')
    a=p.parse_args()
    if not 0<=a.inset_z_top<=.25 or not 0<=a.inset_x_right<=.25:raise ValueError('Insets must be within 0..0.25')
    rep=json.loads(a.report.read_text(encoding='utf-8'))
    low=np.asarray(rep['roi_lower'],float);high=np.asarray(rep['roi_upper'],float)
    if low.shape!=(3,) or high.shape!=(3,) or not (high>low).all():raise ValueError('Invalid roi_lower/roi_upper')
    x0,y0,z0=low;x1,y1,z1=high
    x1-=a.inset_x_right*(high[0]-low[0]);z1-=a.inset_z_top*(high[2]-low[2])
    def rect(a,b,c,d):return [[a,c],[b,c],[b,d],[a,d]]
    out={'source':'automatic_AABB_debug_only_not_user_drawn',
         'coordinate_system':'original_COLMAP_world','polygon_units':'model_units','views':[
         {'name':'top','origin':[0,0,0],'axes':[[1,0,0],[0,1,0]],'polygon':rect(x0,x1,y0,y1)},
         {'name':'front','origin':[0,0,0],'axes':[[1,0,0],[0,0,1]],'polygon':rect(x0,x1,z0,z1)}]}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print('Created',a.out)

if __name__=='__main__':main()
