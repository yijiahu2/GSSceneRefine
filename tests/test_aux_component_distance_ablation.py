"""Regression checks for the minimal 19-component distance-only Gaussian ablation."""
import json
import sys
import types

import numpy as np
import pytest
import trimesh

from tools.ablate_auxiliary_distance_protect import (
    _load_auxiliary_mesh, _protect_by_distance, _validated_parity_masks,
    _checked_export,
)


def test_distance_only_protection_keeps_threshold_boundary_and_no_inside_logic():
    centers = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]],np.float32)
    observed=[]
    def distance(q):
        observed.append(len(q))
        return np.where(q[:,0]==0,0.02,np.where(q[:,0]==1,0.12,np.where(q[:,0]==2,0.121,0.9)))
    protected,distance_aux = _protect_by_distance(centers, .12, 2, distance)
    assert protected.tolist()==[True,True,False,False]
    assert observed==[2,2]
    assert len(distance_aux)==4


def test_auxiliary_19_comp_mesh_exact_faces_preserved(tmp_path):
    # 19 isolated triangles, no connectivity to one another.
    vertices=[]
    for i in range(19):
        vertices.extend([[i*5.0,0.0,0.0],[i*5.0+1.0,0.0,0.0],[i*5.0,1.0,0.0]])
    faces=np.arange(57).reshape(19,3)
    p=tmp_path/'poisson_roi_auxiliary.ply'
    trimesh.Trimesh(vertices=vertices,faces=faces,process=False).export(p)
    mesh,stats=_load_auxiliary_mesh(p,19)
    assert stats['components']==19
    assert stats['faces']==19
    assert np.allclose(mesh.vertices,vertices)
    with pytest.raises(ValueError, match='Expected 20'):
        _load_auxiliary_mesh(p,20)


def test_strict_v2_parity_source_and_mask_provenance(tmp_path):
    source = tmp_path/'final'/'gaussian_roi.ply'
    source.parent.mkdir(); source.write_bytes(b'placeholder')
    directory=tmp_path/'parity'; directory.mkdir()
    report={'schema':'GSSceneRefine.roi_barrier_parity_sdf.v2',
            'status':'CLASSIFICATION_ONLY_PREVIEW',
            'inputs':{'original_R3_Gaussian':str(source)},
            'counts':{'candidate_outside_remove':2,'over_distance_threshold':3},
            'threshold':{'delta':.1226849162}}
    (directory/'roi_parity_report.json').write_text(json.dumps(report))
    rm=np.array([False,True,True,False,False],bool)
    over=np.array([False,True,True,True,False],bool)
    np.savez(directory/'classification_masks.npz',remove=rm,over_threshold=over,
             distance_main=np.zeros(5,np.float32), outside_confident=rm,keep=~rm)
    ret=_validated_parity_masks(directory,5,source)
    assert np.array_equal(ret[0],rm) and ret[2]==.1226849162
    np.savez(directory/'classification_masks.npz',remove=rm,over_threshold=~over,
             distance_main=np.zeros(5,np.float32),outside_confident=rm,keep=~rm)
    with pytest.raises(ValueError,match='below-threshold'):
        _validated_parity_masks(directory,5,source)


class _NativeGaussianPLY:
    """Minimal drop-in exercising actual scalar-binary 3DGS header/record semantics."""
    _names=[('x','<f4'),('y','<f4'),('z','<f4'),
            ('f_dc_0','<f4'),('f_dc_1','<f4'),('f_dc_2','<f4'),
            ('f_rest_0','<f4'),('f_rest_1','<f4'),
            ('opacity','<f4'),('scale_0','<f4'),('scale_1','<f4'),('scale_2','<f4'),
            ('rot_0','<f4'),('rot_1','<f4'),('rot_2','<f4'),('rot_3','<f4')]
    def __init__(self,path):
        self.path=path
        with open(path,'rb') as f:
            lines=[]
            while True:
                line=f.readline();lines.append(line)
                if line.strip()==b'end_header':break
            self.data_offset=f.tell()
        self.header=b''.join(lines)
        header=self.header.decode('ascii')
        self.vertex_count=next(int(x.split()[2]) for x in header.splitlines() if x.startswith('element vertex '))
        self.dtype=np.dtype(self._names)
        self.arr=np.memmap(path,mode='r',dtype=self.dtype,offset=self.data_offset,shape=(self.vertex_count,))
    def export(self,mask,path):
        hdr=self.header.decode('ascii').splitlines(True)
        for i,line in enumerate(hdr):
            if line.startswith('element vertex '):hdr[i]=f'element vertex {int(mask.sum())}\n'
        with open(path,'wb') as f:
            f.write(''.join(hdr).encode('ascii'))
            f.write(self.arr[mask].tobytes())
        return int(mask.sum())


def test_supersplat_full_gaussian_attributes_export_unchanged(tmp_path,monkeypatch):
    from tools import ablate_auxiliary_distance_protect as module
    source=tmp_path/'original.ply'
    values=np.zeros(10,np.dtype(_NativeGaussianPLY._names))
    for i,k in enumerate(values.dtype.names):values[k]=np.arange(10)*(.5+i)
    header='ply\nformat binary_little_endian 1.0\nelement vertex 10\n'
    for name,_ in _NativeGaussianPLY._names:header+=f'property float {name}\n'
    header+='end_header\n'
    source.write_bytes(header.encode('ascii')+values.tobytes())
    fake=types.ModuleType('gsrefine.roi.core'); fake.GaussianPLY=_NativeGaussianPLY
    parent=types.ModuleType('gsrefine.roi'); parent.__path__=[]
    root=types.ModuleType('gsrefine');root.__path__=[]
    monkeypatch.setitem(sys.modules,'gsrefine',root)
    monkeypatch.setitem(sys.modules,'gsrefine.roi',parent)
    monkeypatch.setitem(sys.modules,'gsrefine.roi.core',fake)
    src=_NativeGaussianPLY(source)
    mask=np.arange(10)%3==1
    out=tmp_path/'aux_protected_gaussians.ply'
    assert _checked_export(src,mask,out)==int(mask.sum())
    result=_NativeGaussianPLY(out)
    assert result.vertex_count==int(mask.sum())
    assert 'f_rest_1' in result.dtype.names
    assert 'opacity' in result.dtype.names
    assert 'rot_3' in result.dtype.names
    assert result.arr.tobytes()==values[mask].tobytes()
