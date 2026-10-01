"""StandUp goal-state rewards pay only on feet-only support (no head-tripod)."""
from types import SimpleNamespace as NS
import torch
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks import microduck_standup_env_cfg as standup


def test_gate_wraps_goal_terms_and_keeps_bootstrap_open():
    cfg = standup.make_microduck_standup_env_cfg()
    assert standup.ENABLE_GROUND_SUPPORT_GATE
    sensor = next(s for s in cfg.scene.sensors if s.name == 'non_foot_ground')
    assert 'force' in sensor.fields and sensor.history_length == cfg.decimation  # History needs the force field.
    assert 'body_pose_tracking' in cfg.rewards
    for name in standup.GROUND_SUPPORT_GATED_REWARDS:
        term = cfg.rewards[name]
        assert term.func is mdp.feet_only_support_gated and term.weight >= 0  # body_pose_tracking ramps in from 0.
        assert term.params['sensor_name'] == 'non_foot_ground'
    for name in ('height_stand', 'upright_linear', 'com_upward_velocity'):
        assert cfg.rewards[name].func is not mdp.feet_only_support_gated


def test_gate_zeroes_reward_when_any_non_foot_body_touches():
    hist = torch.zeros(3, 4, 4, 3)  # [env, body, substep, xyz]
    hist[1, 2, 3, 2] = 5.0          # env 1: head on the floor in one substep
    env = NS(scene=NS(sensors={'non_foot_ground': NS(data=NS(force_history=hist))}))
    out = mdp.feet_only_support_gated(env, lambda env, k: torch.full((3,), k), {'k': 2.0})
    assert out.tolist() == [2.0, 0.0, 2.0]
