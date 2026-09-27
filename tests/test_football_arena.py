"""Referee and independent-control checks for the shared football arena."""
import numpy as np
import pytest

from mjlab_microduck.sim.football_arena import build_world, goal_crossing, navigation


@pytest.mark.parametrize('sign,player', [(1,0),(-1,1)])
def test_whole_ball_crossing(sign, player):
    start = np.array([sign*.90, 0., .035])
    assert goal_crossing(start,np.array([sign*.92,0.,.035])) is None
    assert goal_crossing(start,np.array([sign*.96,0.,.035])) == player
    assert goal_crossing(start,np.array([sign*.96,.3,.035])) is None
    assert goal_crossing(start,np.array([sign*.96,0.,.5])) is None
    assert goal_crossing(np.array([sign*.96,0.,.035]),start) is None


def test_navigation_matches_training():
    import torch
    from mjlab_microduck.tasks.mdp import football_navigation_command
    ball = torch.tensor([[.3,.04],[.1,-.042],[-.2,.1]])
    direction = torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
    expected = football_navigation_command(ball,direction,torch.zeros(3,dtype=torch.bool))
    actual = np.stack([navigation(b.numpy(),d.numpy()) for b,d in zip(ball,direction)])
    np.testing.assert_allclose(actual,expected.numpy(),atol=1e-7)


def test_shared_world_independent_actuators():
    model,_ = build_world()
    assert model.nu == 28
    ids = []
    for i in range(2):
        group = [a for a in range(model.nu) if model.actuator(a).name.startswith(f'p{i}_')]
        assert len(group) == 14
        ids.append(set(model.actuator_trnid[group,0]))
    assert ids[0].isdisjoint(ids[1])
    assert model.geom('ball_geom').id >= 0
    assert model.body('g0_goal').pos[0] == -model.body('g1_goal').pos[0]
