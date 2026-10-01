"""Goal-first arena reward budgets, supported-shot gates and no repeat payouts."""
from types import SimpleNamespace as NS
import pytest
import torch
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_arena_env_cfg import (
    make_microduck_arena_env_cfg, GOAL_REWARD, SUPPORT_REWARD_WEIGHTS, EPISODE_LENGTH_S,
)


def arena():
    n=4
    class Scene(dict):
        pass
    robot=NS(root_link_pos_w=torch.tensor([[0.,0.,.12]]).repeat(n,1),
             projected_gravity_b=torch.tensor([[0.,0.,-1.]]).repeat(n,1),
             root_link_lin_vel_w=torch.zeros(n,3))
    ball=NS(root_link_pos_w=torch.tensor([[.3,0.,.035]]).repeat(n,1),root_link_lin_vel_w=torch.zeros(n,3))
    scene=Scene(robot=NS(data=robot),ball=NS(data=ball))
    scene.terrain=NS(env_origins=torch.zeros(n,3))
    scene.sensors={name:NS(data=NS(found=torch.zeros(n,1,dtype=torch.bool)))
        for name in ('kick_ball_contact','kick_support_contact')}
    env=NS(num_envs=n,device='cpu',step_dt=.02,common_step_counter=0,scene=scene,
           episode_length_buf=torch.zeros(n,dtype=torch.long),max_episode_length=500)
    term=mdp.ArenaCommandCfg(resampling_time_range=(11.,11.)).build(env)
    term.previous_ball[:]=ball.root_link_pos_w
    term.initial_goal_distance[:]=.6
    term.initial_distance[:]=mdp.arena_kick_stance(ball.root_link_pos_w[:,:2]).norm(dim=-1)
    env.command_manager=NS(get_term=lambda name:term)
    return env,term


def test_goal_dominates_whole_round_support_budget():
    cfg=make_microduck_arena_env_cfg()
    assert sum(SUPPORT_REWARD_WEIGHTS.values())*EPISODE_LENGTH_S == pytest.approx(8.)
    assert GOAL_REWARD > 10*sum(SUPPORT_REWARD_WEIGHTS.values())*EPISODE_LENGTH_S
    assert cfg.is_finite_horizon
    assert 'time_remaining' in cfg.observations['critic'].terms
    assert 'time_remaining' not in cfg.observations['actor'].terms
    assert cfg.rewards['arena_timeout'].weight < 0


def test_supported_strike_requires_balance_and_forward_motion():
    env,term=arena()
    for sensor in env.scene.sensors.values():
        sensor.data.found[:]=True
    env.scene.sensors['kick_support_contact'].data.found[1]=False
    env.scene['robot'].data.root_link_pos_w[2,2]=.06
    env.scene['ball'].data.root_link_lin_vel_w[:,0]=1.
    env.scene['ball'].data.root_link_lin_vel_w[3,0]=-1.
    term.update_match()
    assert term.struck.tolist()==[True,False,False,False]
    assert (term.strike_delta*env.step_dt).tolist()==[1.,0.,0.,0.]
    assert not term.ball_delta.any()
    env.common_step_counter+=1
    term.update_match()
    assert not term.strike_delta.any()
    assert not term.shot_speed_delta.any()


def test_ball_progress_requires_own_strike_and_cannot_be_farmed():
    env,term=arena()
    ball=env.scene['ball'].data
    ball.root_link_pos_w[:,0]=.5
    term.update_match()
    assert not term.ball_delta.any()  # Opponent movement alone earns nothing.
    for sensor in env.scene.sensors.values():
        sensor.data.found[:]=True
    ball.root_link_lin_vel_w[:,0]=.5
    env.common_step_counter+=1
    term.update_match()
    assert not term.ball_delta.any()  # No retroactive reward at first touch.
    ball.root_link_pos_w[:,0]=.6
    env.common_step_counter+=1
    term.update_match()
    assert torch.allclose(term.ball_delta*env.step_dt,torch.full((4,),.1),atol=1e-6)
    for x in (.5,.6,.6):
        ball.root_link_pos_w[:,0]=x
        env.common_step_counter+=1
        term.update_match()
        assert not term.ball_delta.any()


def test_goalward_distance_rejects_sideways_shot_progress():
    env,term=arena()
    term.struck[:]=True
    env.scene['ball'].data.root_link_pos_w[:,1]=.5
    term.update_match()
    assert not term.ball_delta.any()


def test_timeout_is_once_at_limit_and_never_charged_on_goal():
    env,term=arena()
    env.episode_length_buf[:]=torch.tensor([499,500,500,500])
    term.scored[2]=True
    term.conceded[3]=True
    assert torch.equal(mdp.arena_timeout_penalty(env)*env.step_dt,torch.tensor([0.,1.,0.,0.]))
    assert mdp.arena_time_remaining(env).shape==(4,1)


def test_goal_and_own_goal_pay_once_without_a_strike_requirement():
    env,term=arena()
    term.previous_ball[0,0]=.90
    term.previous_ball[1,0]=-.90
    ball=env.scene['ball'].data.root_link_pos_w
    ball[0,0]=.94
    ball[1,0]=-.94
    term.update_match()
    assert torch.equal(term.goal_delta*env.step_dt,torch.tensor([1.,-1.,0.,0.]))
    env.common_step_counter+=1
    term.update_match()
    assert not term.goal_delta.any()


def test_playback_keeps_goal_rules_without_training_sensors():
    env,term=arena()
    term.cfg.track_training_signals=False
    env.scene.sensors={}
    term.previous_ball[0,0]=.90
    env.scene['ball'].data.root_link_pos_w[0,0]=.94
    term.update_match()
    assert term.scored[0]
    assert env.arena_goal_result[0] == 1
    assert torch.equal(term.previous_ball,env.scene['ball'].data.root_link_pos_w)


def test_fall_event_fires_once_after_persisting_and_rearms_on_recovery():
    env,term=arena()
    height=env.scene['robot'].data.root_link_pos_w[:,2]
    height[0]=.06  # Env 0 falls and stays down; env 1 dips for 3 steps then recovers.
    height[1]=.06
    paid=torch.zeros(4)
    for step in range(25):
        if step == 3:
            height[1]=.12
        env.common_step_counter=step
        term.update_match()
        paid+=term.fall_delta*env.step_dt
        if step == term.cfg.fall_persist_steps-1:
            assert term.fall_delta[0]*env.step_dt == 1
    assert paid.tolist()==[1.,0.,0.,0.]


def test_ball_out_rules():
    env,term=arena()
    ball=env.scene['ball'].data.root_link_pos_w
    ball[:,:2]=torch.tensor([[0.,.70],[.95,.3],[.95,0.],[-.95,.3]])  # Last: behind own line, wide.
    term.update_match()
    term.scored[2]=True  # Through the mouth: a goal, not a dead ball.
    assert mdp.arena_ball_out(env).tolist()==[True,True,False,True]


def test_mode_metrics_split_by_opponent():
    env,term=arena()
    env.arena_opponent_mode=torch.tensor([0,1,2,2])
    term.scored[:]=torch.tensor([True,True,False,True])
    assert mdp.arena_metric(env,'mode_keeper').tolist()==[0.,1.,0.,0.]
    assert mdp.arena_metric(env,'scored_vs_attacker').tolist()==[0.,0.,0.,1.]
    assert mdp.arena_metric(env,'scored_vs_solo').tolist()==[1.,0.,0.,0.]


def test_goal_hold_delays_round_end_but_pays_once():
    env,term=arena()
    term.cfg.goal_hold_s=.1  # 5 steps
    ball=env.scene['ball'].data.root_link_pos_w
    ball[0,:2]=torch.tensor([.95,0.])  # Crosses the mouth from x=.3.
    ends,paid=[],0.
    for step in range(8):
        env.common_step_counter=step
        ends.append(bool(mdp.arena_goal_done(env)[0]))
        paid+=float(term.goal_delta[0])*env.step_dt
    assert paid==1. and ends==[False]*5+[True]*3
    term.cfg.goal_hold_s=0.
