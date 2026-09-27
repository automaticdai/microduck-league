"""Training arena contract: two physical ducks, 61D actors, match-only endings."""
from mjlab_microduck.tasks.microduck_arena_env_cfg import make_microduck_arena_env_cfg, MicroduckArenaRlCfg
from mjlab_microduck.tasks.microduck_football_env_cfg import make_microduck_football_env_cfg
from mjlab_microduck.tasks import mdp


def test_shared_entities_and_policy_contract():
    cfg = make_microduck_arena_env_cfg()
    base = make_microduck_football_env_cfg()
    assert set(cfg.scene.entities) == {'robot','opponent','ball','goal','own_goal'}
    assert cfg.scene.entities['robot'] is not cfg.scene.entities['opponent']
    assert cfg.scene.entities['robot'].spec_fn is cfg.scene.entities['opponent'].spec_fn
    assert list(cfg.observations['actor'].terms) == list(base.observations['actor'].terms)
    assert 'opponent_state' in cfg.observations['critic'].terms
    assert cfg.decimation*cfg.sim.mujoco.timestep == .02
    assert isinstance(cfg.actions['joint_pos'],mdp.ArenaJointPositionActionCfg)
    assert MicroduckArenaRlCfg.actor.obs_normalization
    assert MicroduckArenaRlCfg.algorithm.symmetry_cfg is None


def test_match_endings_rewards_and_reset_order():
    for play in (False,True):
        cfg = make_microduck_arena_env_cfg(play=play)
        assert cfg.episode_length_s == 10
        assert set(cfg.terminations) == {'time_out','arena_goal','nan_state','ball_nan','opponent_nan'}
        assert list(cfg.events)[-1] == 'reset_arena'
        assert cfg.rewards['arena_fallen'].weight < 0
        assert cfg.rewards['arena_goal_delta'].weight > 0
        assert not any(name.startswith('football_') for name in cfg.rewards)
        assert not any(name.startswith('football_') for name in cfg.curriculum)
        assert 'expand_bam_friction_fields' in cfg.events
