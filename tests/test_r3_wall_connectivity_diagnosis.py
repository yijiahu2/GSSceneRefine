from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.audit_r3_wall_connectivity import (summarize_grid, get_pairs,
    sample_segments, write_point_ply, shell_points, counts_by_region)


def test_cached_upper_bottom_contamination(tmp_path):
    shape=(10,8,8)
    labels=np.full(shape,-1,np.int32)
    # x=4 plane is blocked; upper/lower shells on BOTH sides.
    labels[:4,:,:]=0
    labels[5:,:,:]=1
    passed=labels>=0
    with open(tmp_path/'flood.npz','wb') as f:
        np.savez_compressed(f,origin=np.array([0.,0.,0.]),voxel=np.array(.2),
            shape=np.asarray(shape),final_r3_inside=passed,
            passable=passed,components=labels)
    v=summarize_grid(tmp_path/'flood.npz',np.array([0.,0.,1.]),.90)
    r=v['summary']
    assert r['connected_regions']==2
    assert r['upper_seed_counts_by_region'][0]>0
    assert r['upper_seed_counts_by_region'][1]>0
    assert r['lower_seed_counts_by_region'][0]>0
    assert r['lower_seed_counts_by_region'][1]>0
    assert r['top_bottom_shared_regions']==[0,1]


def test_ply_properties_and_record_count(tmp_path):
    pts=np.asarray([[1,2,3],[4,5,6]],float)
    target=tmp_path/'visual.ply'
    write_point_ply(target,pts,[15,20,25])
    data=target.read_bytes()
    assert data.startswith(b'ply\nformat binary_little_endian 1.0\n')
    assert b'element vertex 2\n' in data
    offs=data.index(b'end_header\n')+len(b'end_header\n')
    assert len(data)-offs==2*15
    assert struct.unpack_from('<fff',data,offs)==(1.,2.,3.)
    assert data[offs+12:offs+15]==bytes([15,20,25])


def test_nearest_pairs_across_separated_regions():
    shape=(5,5,5)
    xyz=np.indices(shape).reshape(3,-1).T.astype(float)
    a=xyz[:,0]<2
    b=xyz[:,0]>2
    pairs,stats=get_pairs(a,b,xyz,1.0,30,5.)
    assert len(pairs)>0
    assert len(pairs)<=30
    assert np.all(pairs[:,0,0]<2)
    assert np.all(pairs[:,1,0]>2)
    assert stats['closest_distance']==2.


def test_segment_points_and_shell():
    seg=np.array([[[0,0,0],[1,0,0]],[[0,1,0],[1,1,0]]],float)
    samples=sample_segments(seg,0.2)
    assert len(samples)>=12
    assert np.allclose(samples[0],seg[0,0])
    assert np.allclose(samples[-1],seg[-1,1])
    m=np.ones((5,5,5),bool)
    shell=shell_points(m)
    assert not shell[2,2,2]
    assert shell[0,0,0]
    assert counts_by_region(np.array([0,0,1,-1]),np.arange(4))=={0:2,1:1}
