"""kick-first-v3 arena: 61D actor contract, staged opponents, one-time fall cost, dead-ball endings."""
import pytest
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_arena_env_cfg import (
    make_microduck_arena_env_cfg, make_microduck_arena_v3_env_cfg, MicroduckArenaRlCfg, MicroduckArenaV3RlCfg,
    V3_MODE_STAGES, V3_FALL_COST, GOAL_REWARD,
)


@pytest.mark.parametrize('play', (False, True))
def test_actor_contract_matches_v2_and_critic_sees_mode(play):
    v2, v3 = make_microduck_arena_env_cfg(play=play), make_microduck_arena_v3_env_cfg(play=play)
    assert list(v3.observations['actor'].terms) == list(v2.observations['actor'].terms)
    assert 'opponent_mode' in v3.observations['critic'].terms
    assert 'opponent_mode' not in v2.observations['critic'].terms  # v2 checkpoints still resume.


def test_v2_recipe_is_untouched():
    v2 = make_microduck_arena_env_cfg()
    assert 'arena_fallen' in v2.rewards and 'arena_fall_event' not in v2.rewards
    assert 'arena_ball_out' not in v2.terminations
    assert 'mode_stages' not in v2.events['reset_arena'].params
    assert v2.commands['twist'].aim == 'center'


def test_v3_rewards_signs_and_budget():
    cfg = make_microduck_arena_v3_env_cfg()
    assert 'arena_fallen' not in cfg.rewards
    fall = cfg.rewards['arena_fall_event']
    assert fall.weight < 0 and fall.params == {'component': 'fall_delta'}  # Cost >= 0, negative weight.
    dead = cfg.rewards['arena_timeout']
    assert dead.weight < 0 and dead.params['include_ball_out']
    assert cfg.rewards['arena_goal_delta'].weight > 0
    assert GOAL_REWARD >= 10 * V3_FALL_COST  # A goal still dwarfs the fall that may follow a hard kick.


def test_v3_endings_and_aim():
    cfg = make_microduck_arena_v3_env_cfg()
    assert set(cfg.terminations) == {'time_out', 'arena_goal', 'arena_ball_out', 'nan_state', 'ball_nan', 'opponent_nan'}
    assert not cfg.terminations['arena_ball_out'].time_out  # Terminal: the round is over.
    assert cfg.commands['twist'].aim == 'open'
    assert list(cfg.events)[-1] == 'reset_arena'


def test_mode_stages_are_a_valid_curriculum():
    steps = [s['step'] for s in V3_MODE_STAGES]
    assert steps[0] == 0 and steps == sorted(steps)
    for stage in V3_MODE_STAGES:
        assert len(stage['probs']) == len(mdp.ARENA_OPPONENT_MODES)
        assert sum(stage['probs']) == pytest.approx(1.)
    assert V3_MODE_STAGES[0]['probs'] == (1., 0., 0.)  # Empty goal first.
    first = {m: next(i for i, s in enumerate(V3_MODE_STAGES) if s['probs'][k] > 0)
             for k, m in enumerate(mdp.ARENA_OPPONENT_MODES)}
    assert first['solo'] < first['keeper'] < first['attacker']


def test_per_mode_metrics_and_runner():
    cfg = make_microduck_arena_v3_env_cfg()
    for mode in mdp.ARENA_OPPONENT_MODES:
        assert cfg.metrics[f'arena_mode_{mode}'].reduce == 'last'
        assert cfg.metrics[f'arena_scored_vs_{mode}'].params['name'] == f'scored_vs_{mode}'
    assert MicroduckArenaV3RlCfg.experiment_name != MicroduckArenaRlCfg.experiment_name
    assert MicroduckArenaV3RlCfg.algorithm.symmetry_cfg is None
