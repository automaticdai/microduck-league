"""Where do the arena learner's touches go? Per-touch ball outcomes and round-end state.

uv run scripts/eval_arena_shots.py --checkpoint path/to/arena/model_1750.pt \
  --opponent-policy logs/football_gcp_model_4999.pt --output logs/arena_shots_1750.json
--solo parks the opponent idle off-pitch (skill vs opposition); --opponent-role keeper spawns
it in front of its goal and holds the ball-goal line (mdp.arena_keeper_command), optionally
clearing balls within --keeper-clear-radius. --round-seconds lengthens rounds (clock
ablation: the actor has no time input, so the first 10 s are unchanged).
Only the FIRST round of every arena is recorded, so long rounds are not under-sampled.
A "flight" starts when a learner/ball contact releases and ends at the next touch, a
goal, the ball stopping or leaving play, or the end of the round.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.tasks import mdp

TASK = 'Mjlab-FootballArena-Flat-MicroDuck'
GOAL_PLANE_X = .935  # Whole ball over the line (mdp.ArenaCommand.update_match).
MOUTH_Y = .165
PITCH_X, PITCH_Y = .9, .65
STOP_SPEED = .03
OUTCOMES = ('goal', 'wide', 'out_side', 'own_end', 'stopped', 'intercepted', 'retouch', 'clock', 'own_goal')


def contact_any(env, name):
    history = env.scene.sensors[name].data.force_history
    return torch.nan_to_num(history).abs().sum(-1).flatten(1).gt(1e-6).any(-1)


def build_env(args):
    cfg, agent = load_env_cfg(TASK), load_rl_cfg(TASK)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.episode_length_s = args.round_seconds
    cfg.sim.nan_guard.enabled = False
    def ball_sensor(name, entity, mode, pattern):
        return ContactSensorCfg(name=name, primary=ContactMatch(mode=mode, pattern=pattern, entity=entity),
            secondary=ContactMatch(mode='geom', pattern='^ball_geom$', entity='ball'),
            fields=('found', 'force'), reduce='netforce', num_slots=1, history_length=cfg.decimation)
    cfg.scene.sensors = (*cfg.scene.sensors,
        ball_sensor('eval_robot_ball', 'robot', 'body', '.*'),
        ball_sensor('eval_opponent_ball', 'opponent', 'body', '.*'),
        ball_sensor('eval_left_foot_ball', 'robot', 'geom', '^left_foot_collision$'))
    probs = {'solo': (1., 0., 0.), 'keeper': (0., 1., 0.), 'attacker': (0., 0., 1.)}
    mode = 'solo' if args.solo else args.opponent_role
    cfg.events['reset_arena'].params['mode_stages'] = [{'step': 0, 'probs': probs[mode]}]
    cfg.commands['twist'].aim = args.aim
    cfg.actions['joint_pos'].opponent_aim = getattr(args, 'opponent_aim', 'center')
    if getattr(args, 'boards', False):
        from mjlab_microduck.tasks.microduck_arena_env_cfg import add_arena_boards
        add_arena_boards(cfg)
    cfg.commands['twist'].post_strike_settle_s = args.settle
    cfg.actions['joint_pos'].keeper_distance = args.keeper_distance
    cfg.actions['joint_pos'].keeper_clear_radius = args.keeper_clear_radius
    cfg.actions['joint_pos'].keeper_max_speed = tuple(args.keeper_max_speed)
    return cfg, agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--opponent-policy', type=Path, default=Path('logs/football_gcp_model_4999.pt'))
    parser.add_argument('--num-envs', type=int, default=2048)
    parser.add_argument('--round-seconds', type=float, default=30.)
    parser.add_argument('--solo', action='store_true')
    parser.add_argument('--aim', choices=('center', 'open'), default='center',
                        help="Learner tracker aim: goal center, or beside the opponent's shadow")
    parser.add_argument('--opponent-role', choices=('attacker', 'keeper'), default='attacker')
    parser.add_argument('--keeper-distance', type=float, default=.2)
    parser.add_argument('--keeper-clear-radius', type=float, default=0.)
    parser.add_argument('--keeper-max-speed', type=float, nargs=4, default=(.15, .25, .12, .5),
                        metavar=('BACK', 'FWD', 'SIDE', 'YAW'))
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--boards', action='store_true', help='Pitch boards keep the ball in play (task change)')
    parser.add_argument('--opponent-aim', choices=('center', 'open'), default='center')
    parser.add_argument('--settle', type=float, default=0., help='Zero command for this long after each supported strike')
    parser.add_argument('--getup-policy', type=Path,
                        help='StandUp checkpoint hot-swapped in after 0.2 s down, until 0.5 s upright (runtime-style)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.solo and args.opponent_role != 'attacker':
        parser.error('--solo parks the opponent; it cannot also keep goal')
    cfg, agent = build_env(args)
    configure_torch_backends()
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = load_runner_cls(TASK)(wrapped, asdict(agent), device='cuda:0')
        runner.load(str(args.checkpoint), load_cfg={'actor': True}, map_location='cuda:0')
        learner = runner.alg.actor.eval().requires_grad_(False)
        opponent = deepcopy(learner)
        opponent.load_state_dict(torch.load(args.opponent_policy, map_location='cuda:0',
                                            weights_only=False)['actor_state_dict'])
        env.arena_opponent_policy = opponent  # Solo arenas zero its command (mdp.opponent_command).
        env.reset(seed=args.seed)
        obs = wrapped.get_observations()
        getup = None
        if args.getup_policy:
            getup = deepcopy(learner)  # Same 61D MLP (512-256-128) with its own normalizer.
            getup.load_state_dict(torch.load(args.getup_policy, map_location='cuda:0',
                                             weights_only=False)['actor_state_dict'])
        rounds, flights = run(env, wrapped, learner, obs, getup)
    finally:
        env.close()
    report = {'args': {k: str(v) for k, v in vars(args).items()}, 'rounds': rounds, 'flights': flights}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report))
    print(json.dumps(summarize(report), indent=2))


@torch.inference_mode()
def run(env, wrapped, learner, obs, getup=None):
    n, dev, dt = env.num_envs, env.device, env.step_dt
    ten_s = round(10. / dt)
    z = lambda dtype=torch.float32: torch.zeros(n, dtype=dtype, device=dev)
    active = torch.ones(n, dtype=torch.bool, device=dev)
    s = {k: z(torch.long) for k in ('touches', 'opp_touches', 'right_touches', 'left_touches',
                                     'supported_touches', 'fallen_steps', 'falls', 'last_toucher', 'opp_fallen_steps')}
    s['opp_goal_dist_sum'] = z()
    s['getups'], s['recoveries'] = z(torch.long), z(torch.long)
    swapped, down, up_steps = z(torch.bool), z(torch.long), z(torch.long)
    for k in ('first_touch', 'first_strike', 'last_touch', 'first_fall', 'fall_since_touch'):
        s[k] = torch.full((n,), -1, dtype=torch.long, device=dev)
    s['fall_opp_dist'] = torch.full((n,), math.nan, device=dev)  # Duck-duck distance at the first fall.
    prev = {'contact_r': z(torch.bool), 'contact_o': z(torch.bool), 'upright': torch.ones(n, dtype=torch.bool, device=dev),
            'ball': torch.zeros(n, 2, device=dev), 'ball_speed': z(), 'robot_ball': z(), 'opp_ball': z(), 't': z(torch.long)}
    snap = {k: torch.full((n,), math.nan, device=dev) for k in
            ('ball_x', 'ball_y', 'ball_speed', 'upright', 'robot_ball', 'opp_ball', 'last_toucher', 'since_touch',
             'opp_x', 'opp_y', 'opp_upright')}
    f = {'on': z(torch.bool), 't0': z(torch.long), 'p0': torch.zeros(n, 2, device=dev), 'v0': torch.zeros(n, 2, device=dev),
         'supported': z(torch.bool), 'max_x': z()}
    rounds, flights = [], []

    def end_flights(mask, reason, ball, t):
        mask = mask & f['on']
        if not mask.any():
            return
        idx = torch.where(mask)[0]
        codes = reason[idx] if torch.is_tensor(reason) else torch.full_like(idx, OUTCOMES.index(reason))
        rows = torch.cat((f['t0'][idx, None].float(), t[idx, None].float(), f['p0'][idx], f['v0'][idx],
                          ball[idx], f['supported'][idx, None].float(), f['max_x'][idx, None], codes[:, None].float()), 1)
        for r in rows.cpu().tolist():
            flights.append({'t0': r[0] * dt, 'duration': (r[1] - r[0]) * dt, 'x0': r[2], 'y0': r[3], 'vx0': r[4], 'vy0': r[5],
                            'x1': r[6], 'y1': r[7], 'supported': bool(r[8]), 'max_x': r[9], 'outcome': OUTCOMES[int(r[10])]})
        f['on'][idx] = False

    for _ in range(env.max_episode_length + 2):
        action = learner(obs)
        if getup is not None and swapped.any():
            stand_obs = obs.clone()
            stand_obs['actor'][:, 48:61] = 0  # Exact-zero command block: "stand up".
            action = torch.where(swapped[:, None], getup(stand_obs), action)
        obs, _, dones, _ = wrapped.step(action)
        done = dones.bool()
        goal = env.arena_goal_result.clone()
        timeout, terminated = env.reset_time_outs.clone(), env.reset_terminated.clone()
        strike = env.arena_step_diagnostics['strike']
        origin = env.scene.terrain.env_origins[:, :2]
        ball = env.scene['ball'].data.root_link_pos_w[:, :2] - origin
        ball_v = env.scene['ball'].data.root_link_lin_vel_w[:, :2]
        speed = ball_v.norm(dim=-1)
        robot = env.scene['robot'].data.root_link_pos_w[:, :2] - origin
        opp = env.scene['opponent'].data.root_link_pos_w[:, :2] - origin
        upright = mdp.arena_upright(env)
        if getup is not None:
            down = torch.where(upright | done, 0, down + 1)
            up_steps = torch.where(upright & ~done, up_steps + 1, 0)
            enter = ~swapped & (down >= 10)
            leave = swapped & (up_steps >= 25)
            s['getups'] += active & ~done & enter
            s['recoveries'] += active & ~done & leave
            swapped = (swapped | enter) & ~leave & ~done
        opp_data = env.scene['opponent'].data
        opp_upright = ((opp_data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2] >= .095)
                       & (opp_data.projected_gravity_b[:, 2] < -math.cos(math.pi / 6)))
        t = env.episode_length_buf.clone()
        contact_r, contact_o = contact_any(env, 'eval_robot_ball'), contact_any(env, 'eval_opponent_ball')
        right, left = contact_any(env, 'kick_ball_contact'), contact_any(env, 'eval_left_foot_ball')
        supported = mdp.arena_supported_strike(env) & upright

        # First-strike latch is computed before auto-reset, so it is valid on the final step too.
        first = active & strike & (s['first_strike'] < 0)
        s['first_strike'][first] = torch.where(done, prev['t'] + 1, t)[first]

        live = active & ~done
        # Flights ended by physics within the round.
        inside = ball[:, 1].abs() <= MOUTH_Y
        reason = torch.full((n,), -1, dtype=torch.long, device=dev)
        for code, cond in (('stopped', (speed < STOP_SPEED) & (t - f['t0'] >= 2)),
                           ('out_side', ball[:, 1].abs() > PITCH_Y), ('own_end', ball[:, 0] < -PITCH_X),
                           ('wide', (ball[:, 0] >= GOAL_PLANE_X) & ~inside),
                           ('retouch', contact_r), ('intercepted', contact_o)):
            reason = torch.where(cond, OUTCOMES.index(code), reason)  # Later entries take precedence.
        f['max_x'] = torch.maximum(f['max_x'], ball[:, 0])
        end_flights(live & (reason >= 0), reason, ball, t)
        # A release starts a flight when nobody is touching the ball.
        release = live & prev['contact_r'] & ~contact_r & ~contact_o
        f['on'] |= release
        f['t0'] = torch.where(release, t, f['t0'])
        f['p0'] = torch.where(release[:, None], ball, f['p0'])
        f['v0'] = torch.where(release[:, None], ball_v, f['v0'])
        f['max_x'] = torch.where(release, ball[:, 0], f['max_x'])
        # Touch onsets and round counters.
        onset = live & contact_r & ~prev['contact_r']
        f['supported'] = torch.where(onset, supported, f['supported'] | (live & contact_r & supported))
        s['touches'] += onset
        s['right_touches'] += onset & right
        s['left_touches'] += onset & left
        s['supported_touches'] += onset & supported
        s['opp_touches'] += live & contact_o & ~prev['contact_o']
        s['first_touch'] = torch.where(onset & (s['first_touch'] < 0), t, s['first_touch'])
        s['last_touch'] = torch.where(live & contact_r, t, s['last_touch'])
        s['last_toucher'] = torch.where(live & contact_o, 2, torch.where(live & contact_r, 1, s['last_toucher']))
        s['fallen_steps'] += live & ~upright
        new_fall = live & prev['upright'] & ~upright
        s['falls'] += new_fall
        first_fall = new_fall & (s['first_fall'] < 0)
        s['first_fall'] = torch.where(first_fall, t, s['first_fall'])
        s['fall_opp_dist'] = torch.where(first_fall, (robot - opp).norm(dim=-1), s['fall_opp_dist'])
        s['fall_since_touch'] = torch.where(first_fall & (s['last_touch'] >= 0), t - s['last_touch'], s['fall_since_touch'])
        s['opp_fallen_steps'] += live & ~opp_upright
        own_goal = torch.tensor([PITCH_X, 0.], device=dev)
        s['opp_goal_dist_sum'] += live * (opp - own_goal).norm(dim=-1)
        at10 = live & (t == ten_s)
        if at10.any():
            for k, v in (('ball_x', ball[:, 0]), ('ball_y', ball[:, 1]), ('ball_speed', speed), ('upright', upright.float()),
                         ('robot_ball', (robot - ball).norm(dim=-1)), ('opp_ball', (opp - ball).norm(dim=-1)),
                         ('last_toucher', s['last_toucher'].float()),
                         ('since_touch', torch.where(s['last_touch'] >= 0, (t - s['last_touch']).float() * dt, math.nan)),
                         ('opp_x', opp[:, 0]), ('opp_y', opp[:, 1]), ('opp_upright', opp_upright.float())):
                snap[k] = torch.where(at10, v, snap[k])
        for k, v in (('contact_r', contact_r), ('contact_o', contact_o), ('upright', upright), ('ball', ball),
                     ('ball_speed', speed), ('robot_ball', (robot - ball).norm(dim=-1)),
                     ('opp_ball', (opp - ball).norm(dim=-1)), ('t', t)):
            prev[k] = torch.where(live[:, None] if v.dim() == 2 else live, v, prev[k])

        # Rounds that ended this step; the post-reset physics state is ignored.
        closing = active & done
        if closing.any():
            code = torch.where(goal > 0, OUTCOMES.index('goal'), torch.where(goal < 0, OUTCOMES.index('own_goal'),
                               torch.full_like(t, OUTCOMES.index('clock'))))
            end_flights(closing, code, prev['ball'], prev['t'] + 1)
            idx = torch.where(closing)[0]
            outcome = torch.where(goal > 0, 1, torch.where(goal < 0, -1, torch.where(terminated & ~timeout, 2, 0)))
            cols = [outcome, prev['t'] + 1] + [s[k] for k in s] + [prev[k] for k in ('ball_speed', 'robot_ball', 'opp_ball')]
            cols += [prev['ball'][:, 0], prev['ball'][:, 1], prev['upright']] + [snap[k] for k in snap]
            names = ['outcome', 'steps'] + list(s) + ['end_ball_speed', 'end_robot_ball', 'end_opp_ball',
                     'end_ball_x', 'end_ball_y', 'end_upright'] + ['at10_' + k for k in snap]
            table = torch.stack([c[idx].float() for c in cols], 1).cpu().tolist()
            for row in table:
                r = dict(zip(names, row))
                r['outcome'] = {1: 'scored', -1: 'conceded', 2: 'numerical', 0: 'timeout'}[int(r['outcome'])]
                r['duration'] = r.pop('steps') * dt
                for k in ('first_touch', 'first_strike', 'last_touch', 'first_fall', 'fall_since_touch'):
                    r[k] = r[k] * dt if r[k] >= 0 else None
                rounds.append(r)
            active &= ~closing
        if not active.any():
            break
    return rounds, flights


def summarize(report):
    rounds, flights = report['rounds'], report['flights']
    n = len(rounds)
    q = lambda xs: [round(float(v), 3) for v in np.percentile(xs, [25, 50, 75])] if len(xs) else None
    frac = lambda k, xs: round(sum(k(x) for x in xs) / max(len(xs), 1), 3)
    out = {'rounds': n, 'condition': report['args']}
    for horizon in (10, 20, 30):
        out[f'by_{horizon}s'] = {
            'scored': frac(lambda r: r['outcome'] == 'scored' and r['duration'] <= horizon + 1e-6, rounds),
            'conceded': frac(lambda r: r['outcome'] == 'conceded' and r['duration'] <= horizon + 1e-6, rounds)}
    out['time_to_goal_s_q'] = q([r['duration'] for r in rounds if r['outcome'] == 'scored'])
    out['first_touch_s_q'] = q([r['first_touch'] for r in rounds if r['first_touch'] is not None])
    out['never_touched'] = frac(lambda r: r['first_touch'] is None, rounds)
    out['first_supported_strike_s_q'] = q([r['first_strike'] for r in rounds if r['first_strike'] is not None])
    out['touches_per_round_q'] = q([r['touches'] for r in rounds])
    out['fallen_fraction'] = round(sum(r['fallen_steps'] for r in rounds) / max(sum(r['duration'] for r in rounds) * 50, 1), 3)
    out['falls_per_round'] = round(sum(r['falls'] for r in rounds) / max(n, 1), 3)
    steps = max(sum(r['duration'] for r in rounds) * 50, 1)
    out['opponent'] = {'fallen_fraction': round(sum(r['opp_fallen_steps'] for r in rounds) / steps, 3),
                       'mean_distance_to_own_goal_m': round(sum(r['opp_goal_dist_sum'] for r in rounds) / steps, 3),
                       'touches_per_round': round(sum(r['opp_touches'] for r in rounds) / max(n, 1), 3)}
    # Rounds still running at 10 s: this is the "timeout" population of the training clock.
    alive = [r for r in rounds if not math.isnan(r['at10_ball_x'])]
    if alive:
        in_pitch = [abs(r['at10_ball_x']) <= PITCH_X and abs(r['at10_ball_y']) <= PITCH_Y for r in alive]
        out['at_10s'] = {
            'rounds': len(alive),
            'ball_in_pitch': round(float(np.mean(in_pitch)), 3),
            'ball_x_q': q([r['at10_ball_x'] for r in alive]),
            'ball_abs_y_q': q([abs(r['at10_ball_y']) for r in alive]),
            'ball_moving(>0.05)': frac(lambda r: r['at10_ball_speed'] > .05, alive),
            'learner_upright': frac(lambda r: r['at10_upright'] > .5, alive),
            'learner_within_18cm_of_ball': frac(lambda r: r['at10_robot_ball'] < .18, alive),
            'opponent_within_18cm_of_ball': frac(lambda r: r['at10_opp_ball'] < .18, alive),
            'learner_ball_dist_q': q([r['at10_robot_ball'] for r in alive]),
            'last_touch_by_learner': frac(lambda r: r['at10_last_toucher'] == 1, alive),
            'last_touch_by_opponent': frac(lambda r: r['at10_last_toucher'] == 2, alive),
            'seconds_since_learner_touch_q': q([r['at10_since_touch'] for r in alive if not math.isnan(r['at10_since_touch'])]),
        }
    out['flights'] = flight_summary([x for x in flights if x['t0'] < 10.], 'first 10 s')
    return out


def flight_summary(flights, label):
    real = [x for x in flights if math.hypot(x['vx0'], x['vy0']) >= .10]
    out = {'window': label, 'all_releases': len(flights), 'touches_launching_ball_>=0.1m/s': len(real)}
    if not real:
        return out
    speed = np.array([math.hypot(x['vx0'], x['vy0']) for x in real])
    heading_err, aim_y, on_target = [], [], []
    for x in real:
        goal_dir = math.atan2(-x['y0'], .9 - x['x0'])
        err = math.atan2(x['vy0'], x['vx0']) - goal_dir
        heading_err.append(abs(math.degrees((err + math.pi) % (2 * math.pi) - math.pi)))
        y_line = x['y0'] + x['vy0'] * (GOAL_PLANE_X - x['x0']) / x['vx0'] if x['vx0'] > 1e-3 else math.inf
        aim_y.append(y_line)
        on_target.append(abs(y_line) <= MOUTH_Y)
    on_target = np.array(on_target)
    outcomes = np.array([x['outcome'] for x in real])
    out['launch_speed_mps_q'] = [round(float(v), 3) for v in np.percentile(speed, [25, 50, 75, 90])]
    out['goalward_launch'] = round(float(np.mean([x['vx0'] > 0 for x in real])), 3)
    out['heading_error_deg_q'] = [round(float(v), 1) for v in np.percentile(heading_err, [25, 50, 75])]
    out['on_target_by_launch_line'] = round(float(on_target.mean()), 3)
    out['supported_strike_touches'] = round(float(np.mean([x['supported'] for x in real])), 3)
    out['outcome_all'] = {k: round(float((outcomes == k).mean()), 3) for k in OUTCOMES if (outcomes == k).any()}
    out['outcome_on_target'] = {k: round(float((outcomes[on_target] == k).mean()), 3)
                                for k in OUTCOMES if (outcomes[on_target] == k).any()}
    stopped = [x for x, o in zip(real, on_target) if o and x['outcome'] == 'stopped']
    if stopped:
        out['on_target_stopped_short'] = {
            'count': len(stopped),
            'needed_m_q': [round(float(v), 3) for v in np.percentile([GOAL_PLANE_X - x['x0'] for x in stopped], [25, 50, 75])],
            'travelled_m_q': [round(float(v), 3) for v in np.percentile(
                [math.hypot(x['x1'] - x['x0'], x['y1'] - x['y0']) for x in stopped], [25, 50, 75])],
            'launch_speed_q': [round(float(v), 3) for v in np.percentile(
                [math.hypot(x['vx0'], x['vy0']) for x in stopped], [25, 50, 75])],
        }
    free = [x for x in real if x['outcome'] == 'stopped' and x['duration'] > .1]
    if free:
        # Distance per launch speed on free rolls fixes the ball's effective reach.
        reach = [math.hypot(x['x1'] - x['x0'], x['y1'] - x['y0']) / math.hypot(x['vx0'], x['vy0']) for x in free]
        out['free_roll_metres_per_mps_q'] = [round(float(v), 3) for v in np.percentile(reach, [25, 50, 75])]
    return out


if __name__ == '__main__':
    main()
