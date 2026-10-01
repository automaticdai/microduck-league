"""keeper-v1: the learner keeps the -x goal; conceding costs, a clean round pays, 61D actor unchanged."""
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_arena_env_cfg import (
    make_microduck_arena_env_cfg, make_microduck_arena_keeper_env_cfg, MicroduckArenaKeeperRlCfg, MicroduckArenaRlCfg)


def test_keeper_task_contract():
    cfg, base = make_microduck_arena_keeper_env_cfg(), make_microduck_arena_env_cfg()
    assert list(cfg.observations['actor'].terms) == list(base.observations['actor'].terms)
    assert cfg.commands['twist'].role == 'keeper' and base.commands['twist'].role == 'attacker'
    assert cfg.events['reset_arena'].params['mode_stages'] == [{'step': 0, 'probs': (0., 0., 1.)}]
    assert cfg.rewards['arena_goal_delta'].weight > 0  # goal_delta is -1/dt when conceded: -100.
    assert cfg.rewards['arena_timeout'].weight > 0 and cfg.rewards['arena_timeout'].params['include_ball_out']
    assert cfg.rewards['arena_fall_event'].weight < 0
    assert not any(k in cfg.rewards for k in ('arena_approach_delta', 'arena_ball_delta', 'arena_strike_delta'))
    assert 'arena_ball_out' in cfg.terminations
    assert MicroduckArenaKeeperRlCfg.experiment_name != MicroduckArenaRlCfg.experiment_name


def test_learner_keeper_command_defends_minus_x():
    import math, torch
    goal = torch.tensor([[-.9, 0.]])
    cmd = mdp.arena_keeper_command(torch.tensor([[-.7, 0.]]), torch.tensor([0.]), torch.tensor([[0., 0.]]), goal, -1)
    assert cmd.tolist() == [[0., 0., 0.]]  # On its spot, facing +x at the ball: idle.
    back = mdp.arena_keeper_command(torch.tensor([[-.5, 0.]]), torch.tensor([0.]), torch.tensor([[0., 0.]]), goal, -1)
    assert back[0, 0] < 0  # Backs toward its goal.
