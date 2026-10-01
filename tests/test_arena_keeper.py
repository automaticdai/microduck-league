"""Opt-in keeper opponent: holds the ball-goal line in front of its own (+x) goal."""
import math
import inspect
import torch
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_arena_env_cfg import make_microduck_arena_env_cfg

GOAL = torch.tensor([[.9, 0.]])
FACING_FIELD = torch.tensor([math.pi])


def keeper(robot, ball, yaw=FACING_FIELD):
    return mdp.arena_keeper_command(torch.tensor([robot]), yaw, torch.tensor([ball]), GOAL, 1)[0]


def test_training_default_is_unchanged_attacker():
    cfg = make_microduck_arena_env_cfg()
    assert 'mode_stages' not in cfg.events['reset_arena'].params  # Every v2 arena: attacker.
    assert inspect.signature(mdp.reset_arena).parameters['mode_stages'].default is None
    assert mdp.ARENA_OPPONENT_MODES[mdp.ATTACKER] == 'attacker'


def test_keeper_on_its_spot_facing_ball_idles_exactly():
    assert keeper([.7, 0.], [0., 0.]).tolist() == [0., 0., 0.]


def test_keeper_backs_onto_line_and_tracks_ball_laterally():
    back = keeper([.5, 0.], [0., 0.])
    assert back[0] < 0 and abs(back[1]) < 1e-6 and back[2] == 0  # Yaw pi: +x world is -x body.
    side = keeper([.7, 0.], [.2, .3])
    assert side[1] < 0  # Body -y is world +y when facing -x.
    assert side[2] < 0  # Turns toward the off-center ball (clockwise from yaw pi).
    beside = keeper([.7, 0.], [.6, .6], yaw=torch.tensor([math.pi - math.pi/3]))
    assert beside[2] == 0  # Already at the 60-degree cap: never turns its back to the field.


def test_keeper_turns_back_to_face_field_and_idles_when_ball_out():
    rotated = keeper([.7, 0.], [0., 0.], yaw=torch.tensor([math.pi/2]))
    assert rotated[2] != 0
    for ball in ([1.3, .4], [.2, .9]):
        assert keeper([.5, .1], ball).tolist() == [0., 0., 0.]


def test_keeper_spot_stays_in_front_of_line_and_inside_posts():
    for ball in ([.9, .6], [.9, -.6], [.9, .3]):
        cmd = keeper([.82, math.copysign(.2, ball[1]) if ball[1] else 0.], ball)
        assert cmd[:2].tolist() == [0., 0.]


def test_small_errors_give_exact_zero_not_nudges():
    # 4 cm off the spot: stand still, no stumble-inducing creep.
    assert keeper([.66, 0.], [0., .1]).tolist() == [0., 0., 0.]
    moving = keeper([.62, 0.], [0., 0.])
    assert torch.linalg.vector_norm(moving[:2]) >= .12


def test_clamps_match_the_attack_tracker_limits():
    far = keeper([-.9, .6], [0., 0.])
    assert -.25 <= far[0] <= .35 and -.2 <= far[1] <= .2 and -1 <= far[2] <= 1
