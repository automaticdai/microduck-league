"""Opt-in open-side aim: shoot beside the blocker's shadow, center when the mouth is clear."""
import math
import torch
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_arena_env_cfg import make_microduck_arena_env_cfg

LINE, HALF = .935, .165


def aim(ball, blocker, side=0):
    xy, s = mdp.arena_open_aim(torch.tensor([ball]), torch.tensor([blocker]), torch.tensor([side]))
    return xy[0, 1].item(), s[0].item()


def test_training_default_aims_at_center():
    assert make_microduck_arena_env_cfg().commands['twist'].aim == 'center'


def test_clear_mouth_keeps_exact_center_aim():
    for blocker in ([-.3, 0.], [.4, 1.5], [.5, -.9]):  # Behind the ball, parked, far wide.
        xy, s = mdp.arena_open_aim(torch.tensor([[.1, .05]]), torch.tensor([blocker]), torch.tensor([0]))
        assert s.item() == 0 and torch.equal(xy[0], torch.tensor([.9, 0.]))  # Bit-identical to the default target.


def test_aims_away_from_offcenter_blocker_inside_the_mouth():
    y, s = aim([0., 0.], [.7, .06])  # Keeper drifted to the left post side.
    assert s == -1 and -HALF < y < 0
    y, s = aim([0., 0.], [.7, -.06])
    assert s == 1 and 0 < y < HALF


def test_chosen_gap_clears_the_shadow():
    ball, blocker = [0., 0.], [.7, .07]
    y, _ = aim(ball, blocker)
    shot = math.atan2(y - ball[1], LINE - ball[0])
    to_blocker = math.atan2(blocker[1], blocker[0])
    spread = math.asin(.12 / math.hypot(*blocker))
    assert abs(shot - to_blocker) > spread


def test_hysteresis_holds_side_for_a_centered_keeper():
    assert aim([0., 0.], [.7, 0.], side=1)[1] == 1
    assert aim([0., 0.], [.7, 0.], side=-1)[1] == -1


def test_closed_mouth_aims_at_far_edge_from_blocker():
    y, s = aim([.5, 0.], [.62, .01])  # Blocker right in front of the ball.
    assert s == -1 and y < -.1
