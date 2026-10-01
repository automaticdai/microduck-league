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


def test_opponent_open_aim_is_mirrored_and_keeper_task_uses_it():
    import torch
    cfg = make_microduck_arena_keeper_env_cfg()
    assert cfg.actions['joint_pos'].opponent_aim == 'open'
    assert make_microduck_arena_env_cfg().actions['joint_pos'].opponent_aim == 'center'
    # Mirror check: a blocker left of the -x goal's line (from the attacker's view) sends the aim right.
    ball, keeper = torch.tensor([[0., 0.]]), torch.tensor([[-.7, -.06]])
    aim, side = mdp.arena_open_aim(-ball, -keeper, torch.tensor([0]))
    aim = -aim
    assert aim[0, 0] < -.9 and aim[0, 1] > 0  # On the -x goal line, away from the keeper (y<0).


def test_boards_option_adds_colliding_entity():
    from mjlab_microduck.tasks.microduck_arena_env_cfg import add_arena_boards, get_arena_boards_spec, ARENA_BOARDS
    cfg = add_arena_boards(make_microduck_arena_env_cfg())
    assert 'boards' in cfg.scene.entities and 'boards' not in make_microduck_arena_env_cfg().scene.entities
    geoms = get_arena_boards_spec().worldbody.first_body().geoms
    assert len(geoms) == len(ARENA_BOARDS) and all(g.contype == 1 and g.conaffinity == 1 for g in geoms)
