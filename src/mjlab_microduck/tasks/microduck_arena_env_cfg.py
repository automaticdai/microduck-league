"""Microduck Go football: shared physical opponent, center-disk kickoffs, 10 s rounds.

Use train_arena.py: the opponent must be explicitly initialized from a policy.
The actor remains 61D; the critic additionally observes the opponent. No fall or
out-of-bounds termination: only goals, timeout, and numerical-safety failures.
"""
from copy import deepcopy
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from .microduck_football_env_cfg import make_microduck_football_env_cfg, MicroduckFootballRlCfg
from . import mdp

ENABLE_SYMMETRY = False
EPISODE_LENGTH_S = 10.0
REWARD_RECIPE = 'goal-first-v2'
GOAL_REWARD = 100.0
FALL_COST_PER_SECOND = 5.0
TIMEOUT_COST = 5.0
# These bounded auxiliaries pay at most 8 points over a full ten-second round.
SUPPORT_REWARD_WEIGHTS = {
    'track_linear_velocity': .20, 'track_angular_velocity': .10,
    'upright': .30, 'pose': .10, 'head_pose_tracking': .10, 'air_time': 0.,
}


def make_microduck_arena_env_cfg(play=False,rough=False):
    cfg = make_microduck_football_env_cfg(play=play,rough=rough)
    cfg.scene.entities['opponent'] = deepcopy(cfg.scene.entities['robot'])
    cfg.scene.entities['own_goal'] = deepcopy(cfg.scene.entities['goal'])
    cfg.scene.env_spacing = 4.0
    cfg.sim.nconmax = 250
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.is_finite_horizon = True
    cfg.commands['twist'] = mdp.ArenaCommandCfg(resampling_time_range=(11.,11.))
    cfg.actions['joint_pos'] = mdp.ArenaJointPositionActionCfg(
        entity_name='robot',actuator_names=('^(?!passive_).*',),scale=1.0)
    for name in list(cfg.rewards):
        if name.startswith('football_') or name == 'ball_speed_overshoot':
            del cfg.rewards[name]
    for name,weight in SUPPORT_REWARD_WEIGHTS.items():
        cfg.rewards[name].weight = weight
    for name,weight in (('approach_delta',5.),('ball_delta',15.),
                        ('strike_delta',3.),('shot_speed_delta',3.),('goal_delta',GOAL_REWARD)):
        cfg.rewards['arena_'+name] = RewardTermCfg(func=mdp.arena_progress,weight=weight,params={'component':name})
    cfg.rewards['arena_fallen'] = RewardTermCfg(func=mdp.arena_fallen_penalty,weight=-FALL_COST_PER_SECOND)
    cfg.rewards['arena_timeout'] = RewardTermCfg(func=mdp.arena_timeout_penalty,weight=-TIMEOUT_COST)
    cfg.rewards['action_rate_l2'].weight = -.02
    cfg.curriculum.pop('action_rate_weight',None)
    for name in list(cfg.curriculum):
        if name.startswith('football_'):
            del cfg.curriculum[name]
    # Inherited DR remains mature when transferring the trained actor.
    for name in ('com_range','head_com_range'):
        if name in cfg.curriculum:
            stages = cfg.curriculum[name].params['range_stages']
            cfg.curriculum[name].params['range_stages'] = [{**stages[-1],'step':0}]
    cfg.events.pop('reset_football')
    cfg.events.pop('push_robot',None)
    cfg.events['reset_arena'] = EventTermCfg(func=mdp.reset_arena,mode='reset')
    cfg.observations['critic'].terms.pop('football_state')
    cfg.observations['critic'].terms['opponent_state'] = ObservationTermCfg(func=mdp.arena_opponent_state)
    cfg.observations['critic'].terms['time_remaining'] = ObservationTermCfg(func=mdp.arena_time_remaining)
    cfg.terminations.pop('lost_balance',None)
    # Keep the template's numerical guards, but not locomotion fall endings.
    for name in list(cfg.terminations):
        if name not in ('time_out','nan_state','ball_nan'):
            del cfg.terminations[name]
    cfg.terminations['arena_goal'] = TerminationTermCfg(func=mdp.arena_goal_done)
    cfg.terminations['opponent_nan'] = TerminationTermCfg(func=mdp.robot_state_is_nan,
        params={'asset_cfg':SceneEntityCfg('opponent')})
    cfg.metrics = {f'arena_{name}':MetricsTermCfg(func=mdp.arena_metric,params={'name':name},reduce='last')
                   for name in ('scored','conceded','standing','struck')}
    return cfg


MicroduckArenaRlCfg = deepcopy(MicroduckFootballRlCfg)
MicroduckArenaRlCfg.experiment_name = 'football_arena'
MicroduckArenaRlCfg.run_name = 'selfplay'
MicroduckArenaRlCfg.max_iterations = 5000

# At 50 Hz, gamma=.997 gives a ~6.7 s credit horizon (old .99: ~2 s).
MicroduckArenaRlCfg.algorithm.gamma = .997
