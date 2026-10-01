"""The shot eval must actually apply its condition flags to the env cfg (a refactor once dropped --aim)."""
import importlib.util, types

spec = importlib.util.spec_from_file_location('eas', 'scripts/eval_arena_shots.py')
eas = importlib.util.module_from_spec(spec); spec.loader.exec_module(eas)


def build(**kw):
    args = dict(num_envs=4, seed=0, round_seconds=10., solo=False, opponent_role='attacker', aim='center', settle=0.,
                boards=False, opponent_aim='center',
                keeper_distance=.2, keeper_clear_radius=0., keeper_max_speed=(.15, .25, .12, .5))
    args.update(kw)
    return eas.build_env(types.SimpleNamespace(**args))[0]


def test_aim_settle_and_modes_reach_the_cfg():
    cfg = build(aim='open', settle=.5, opponent_role='keeper', keeper_distance=.25)
    assert cfg.commands['twist'].aim == 'open'
    assert cfg.commands['twist'].post_strike_settle_s == .5
    assert cfg.events['reset_arena'].params['mode_stages'] == [{'step': 0, 'probs': (0., 1., 0.)}]
    assert cfg.actions['joint_pos'].keeper_distance == .25
    assert build(solo=True).events['reset_arena'].params['mode_stages'][0]['probs'] == (1., 0., 0.)
    assert build().commands['twist'].aim == 'center'


def test_boards_and_opponent_aim_flags_reach_the_cfg():
    cfg = build(boards=True, opponent_aim='open')
    assert 'boards' in cfg.scene.entities and cfg.actions['joint_pos'].opponent_aim == 'open'
