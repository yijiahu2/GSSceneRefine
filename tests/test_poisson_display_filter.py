"""Pure numerical and input-validation tests for module-2 temporary filtering."""
import numpy as np
import pytest

from gsrefine.roi_init.display_filter import (
    global_trimmed_long_axis, keep_for_preview, validate_multiplier,
)


def test_exact_symmetric_5pct_trimming_and_activated_long_axes():
    n=100
    dtype=[('scale_0','f4'),('scale_1','f4'),('scale_2','f4')]
    data=np.zeros(n, dtype=dtype)
    # 90 normal samples, plus 5 low and 5 high outliers, both tails removed.
    axis=np.array([0.001]*5+[0.1]*90+[10.0]*5, dtype=np.float64)
    data['scale_0']=np.log(axis).astype('f4')
    data['scale_1']=np.log(axis/2).astype('f4')
    data['scale_2']=np.log(axis/3).astype('f4')
    assert global_trimmed_long_axis(data) == pytest.approx(.1, rel=2e-6)


def test_multiplier_changes_outer_buffer_only():
    distance=np.array([-5., -0.1, 0., .05, .11, .3],dtype='f4')
    assert keep_for_preview(distance,.1,1).tolist()==[True,True,True,True,False,False]
    assert keep_for_preview(distance,.1,2).tolist()==[True,True,True,True,True,False]
    assert keep_for_preview(distance,.1,.25).tolist()==[True,True,True,False,False,False]


def test_slider_multiplier_requires_quarter_steps():
    assert validate_multiplier('1.00')==1.0
    assert validate_multiplier('4')==4.0
    with pytest.raises(ValueError): validate_multiplier('0')
    with pytest.raises(ValueError): validate_multiplier('1.13')
    with pytest.raises(ValueError): validate_multiplier('nan')


def test_reject_nonfinite_source_scales():
    data=np.zeros(32,dtype=[('scale_0','f4'),('scale_1','f4'),('scale_2','f4')])
    data['scale_1'][0]=np.nan
    with pytest.raises(ValueError): global_trimmed_long_axis(data)


def test_transient_display_ply_subset_does_not_change_source(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from pathlib import Path
    from gsrefine.roi.core import GaussianPLY
    from gsrefine.roi_init.display_filter import PoissonDisplayFilter

    n=100
    types=['x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    arr=np.zeros(n,dtype=[(name,'<f4') for name in types])
    arr['x']=np.arange(n,dtype='f4')*.01
    arr['scale_0']=np.log(.1);arr['scale_1']=np.log(.05);arr['scale_2']=np.log(.025)
    arr['rot_0']=1.
    original=tmp_path/'original.ply'
    header='ply\nformat binary_little_endian 1.0\nelement vertex 100\n'
    header+=''.join(f'property float {x}\n' for x in types)+'end_header\n'
    original.write_bytes(header.encode('ascii')+arr.tobytes())
    before=original.read_bytes()
    poisson=tmp_path/'poisson.ply';poisson.write_bytes(b'placeholder; mocked ray caster')
    settings=SimpleNamespace(gaussian_ply=original,poisson_ply=poisson,out_dir=tmp_path)
    worker=PoissonDisplayFilter(settings)

    def mocked_distance(ply, destination):
        worker._watertight=True;worker._mode='watertight_signed_distance'
        sd=np.lib.format.open_memmap(destination,mode='w+',dtype='f4',shape=(ply.vertex_count,))
        sd[:40]=-.1;sd[40:]=.2
        sd.flush()
        return sd
    monkeypatch.setattr(worker,'_compute_signed_distance',mocked_distance)
    narrow=worker.prepare_preview(1)
    assert narrow['display_gaussians']==40
    assert GaussianPLY(worker.get_existing_preview(1)).vertex_count==40
    wide=worker.prepare_preview(4)
    assert wide['display_gaussians']==100
    assert GaussianPLY(worker.get_existing_preview(4)).vertex_count==100
    assert original.read_bytes()==before
    assert narrow['base_width']==pytest.approx(.1,rel=1e-6)
