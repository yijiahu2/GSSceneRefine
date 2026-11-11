"""Open A+B rules do not require Open3D/libigl during unit testing."""
import numpy as np
import pytest
from gsrefine.cleanup.open_ab_main import classify_main_open_ab, count_unobstructed_camera_rays


def test_far_and_confirmed_exterior_removed_immediately():
    d=np.array([0.1,0.7,0.8,0.9,1.,np.nan])
    w=np.array([np.nan,.05,.90,.4,.05,np.nan])
    votes=np.array([0,3,8,6,1,0])
    r=classify_main_open_ab(d,w,votes,.5)
    assert r['remove'].tolist()==[False,True,False,False,False,False]
    assert r['keep'].tolist()==[True,False,True,True,True,True]
    assert r['inside'].tolist()==[False,False,True,False,False,False]
    assert r['uncertain'].tolist()==[False,False,False,True,True,False]


def test_threshold_is_strict_and_epsilon_matches_old():
    d=np.array([.5,.500001,.50002])
    w=np.zeros(3);views=np.full(3,2)
    r=classify_main_open_ab(d,w,views,.5,epsilon=1e-5)
    assert r['remove'].tolist()==[False,False,True]


def test_uncertain_for_missing_winding_far():
    r=classify_main_open_ab([.6],[np.nan],[5],.5)
    assert r['uncertain'][0] and r['keep'][0]


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError):classify_main_open_ab([1],[.05],[2],.5,winding_outside_max=.9,winding_inside_min=.8)
    with pytest.raises(ValueError):classify_main_open_ab([1],[.05],[2],float('nan'))
    with pytest.raises(ValueError):classify_main_open_ab([1],[.05],[2,3],.5)


class FakeTensor:
    def __init__(self,a,dtype=None):self.x=np.asarray(a)

class FakeCast:
    def __init__(self,a):self.data=np.asarray(a)
    def numpy(self):return self.data

class FakeScene:
    def __init__(self,hit):self.hit=hit;self.rays=None
    def cast_rays(self,ray):
        self.rays=ray.x
        return {'t_hit':FakeCast(np.broadcast_to(self.hit,len(ray.x)))}

class FakeO3d:
    class core:
        Tensor=FakeTensor


def test_camera_ray_visibility_votes_only_forward_unoccluded():
    cam=np.array([[0.,0.,0.],[0.,1.,0.]])
    forward=np.array([[1.,0.,0.],[1.,0.,0.]])
    points=np.array([[2.,.5,0.]])
    scene=FakeScene(float('inf'))
    votes=count_unobstructed_camera_rays(points,cam,forward,scene,o3d=FakeO3d,k=2)
    assert votes.tolist()==[2]
    assert scene.rays.shape==(2,6)
    # hit at 0.1 before query => occluded
    scene=FakeScene(0.1)
    assert count_unobstructed_camera_rays(points,cam,forward,scene,o3d=FakeO3d,k=2).tolist()==[0]


def test_camera_rays_backward_no_vote():
    cam=np.array([[0.,0.,0.],[0.,1.,0.]])
    backward=np.array([[-1.,0.,0.],[-1.,0.,0.]])
    points=np.array([[2.,.5,0.]])
    scene=FakeScene(float('inf'))
    assert count_unobstructed_camera_rays(points,cam,backward,scene,o3d=FakeO3d,k=2).tolist()==[0]
    assert scene.rays is None


def test_stage2_cli_routes_open_ab_without_seal(monkeypatch):
    from gsrefine.cleanup import direct_r3_pipeline
    from gsrefine.cleanup import open_ab_main
    calls=[]
    def fake_run(**kwargs):
        calls.append(kwargs)
        return {'status':'EXPERIMENT','kept':10,'removed':1}
    monkeypatch.setattr(open_ab_main,'run_open_ab_main',fake_run)
    result=direct_r3_pipeline.run(manifest='/dummy.json',phase='open-ab',
              preseal_report='/dummy_preseal.json',output_dir='/tmp/dummy',chunk_size=64)
    assert result['status']=='EXPERIMENT'
    assert calls[0]['chunk_size']==64
    assert calls[0]['output_dir']=='/tmp/dummy'
    with pytest.raises(ValueError):
        direct_r3_pipeline.run(manifest='/dummy.json',phase='open-ab',output_dir='/tmp/dummy')


def test_camera_relative_mode_can_classify_outside_when_winding_nonzero():
    # An OPEN surface may subtend a large oriented solid angle at an exterior
    # camera. High absolute winding is not proof that the camera is inside.
    d=np.array([.6,.6,.6,.1])
    w=np.array([.37,-.36,.92,np.nan])
    votes=np.array([2,3,0,0])  # votes already filtered by local winding reference
    r=classify_main_open_ab(d,w,votes,.5,outside_mode='camera_relative')
    assert r['remove'].tolist()==[True,True,False,False]
    assert r['uncertain'].tolist()==[False,False,False,False]
    assert r['inside'].tolist()==[False,False,True,False]


def test_camera_relative_voter_mask_does_not_allow_unanchored_camera():
    cam=np.array([[0.,0.,0.],[0.,1.,0.],[0.,2.,0.]])
    forward=np.array([[1.,0.,0.],[1.,0.,0.],[1.,0.,0.]])
    point=np.array([[2.,.5,0.]])
    ids=np.array([[0,1,2]])
    scene=FakeScene(float('inf'))
    v=count_unobstructed_camera_rays(point,cam,forward,scene,o3d=FakeO3d,k=3,
                  camera_indices=ids,anchor_mask=np.array([[True,False,True]]))
    assert v.tolist()==[2]
    assert len(scene.rays)==2
    with pytest.raises(ValueError):
        count_unobstructed_camera_rays(point,cam,forward,scene,o3d=FakeO3d,k=3,
                  camera_indices=ids,anchor_mask=np.array([[True,False]]))


def test_camera_relative_mode_no_votes_means_keep_even_when_winding_matches():
    r=classify_main_open_ab([.8],[.38],[1],.5,outside_mode='camera_relative')
    assert r['keep'][0] and not r['remove'][0] and r['uncertain'][0]
