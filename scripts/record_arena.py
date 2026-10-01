"""Record arena rounds to MP4 under the exact eval conditions (scripts/eval_arena_shots.py).

uv run scripts/record_arena.py --checkpoint path/model.pt --opponent-role keeper --aim open \
  --rounds 6 --output logs/videos/attacker_vs_keeper.mp4
Sim metrics can pass while the video fails the eye: check which body touches, how it falls,
and whether a keeper dives or stumbles. One arena, rounds back to back; each frame is labeled.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer.viewer_config import ViewerConfig
from mjlab_microduck.sim.football_arena_gpu import blue_spec, orange_spec, pitch_spec

_spec = importlib.util.spec_from_file_location('eval_arena_shots', Path(__file__).with_name('eval_arena_shots.py'))
eas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eas)


def label(frame, lines):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 18 * len(lines) + 6), fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((6, 4 + 18 * i), line, fill=(255, 255, 255))
    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--opponent-policy', type=Path, default=Path('logs/football_gcp_model_4999.pt'))
    parser.add_argument('--opponent-role', choices=('attacker', 'keeper'), default='attacker')
    parser.add_argument('--solo', action='store_true')
    parser.add_argument('--aim', choices=('center', 'open'), default='open')
    parser.add_argument('--keeper-distance', type=float, default=.2)
    parser.add_argument('--keeper-max-speed', type=float, nargs=4, default=(.15, .25, .12, .5))
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--title', default='')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ns = argparse.Namespace(num_envs=1, seed=args.seed, round_seconds=10., solo=args.solo,
                            opponent_role=args.opponent_role, aim=args.aim, settle=0.,
                            keeper_distance=args.keeper_distance, keeper_clear_radius=0.,
                            keeper_max_speed=tuple(args.keeper_max_speed))
    cfg, agent = eas.build_env(ns)
    cfg.scene.entities['pitch'] = EntityCfg(spec_fn=pitch_spec)  # Markings only: training physics.
    cfg.scene.entities['robot'].spec_fn = blue_spec
    cfg.scene.entities['opponent'].spec_fn = orange_spec
    cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
    cfg.viewer.lookat = (0, 0, .1)
    cfg.viewer.distance, cfg.viewer.azimuth, cfg.viewer.elevation = 2.4, 125, -38
    cfg.viewer.width, cfg.viewer.height = 960, 544
    cfg.viewer.enable_shadows = False
    cfg.viewer.enable_reflections = False
    configure_torch_backends()
    env = ManagerBasedRlEnv(cfg, device='cuda:0', render_mode='rgb_array')
    title = args.title or f'{args.checkpoint.parent.name}/{args.checkpoint.stem} vs ' + (
        'empty goal' if args.solo else f'{args.opponent_role} ({args.opponent_policy.stem})')
    frames, rounds = [], []
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = load_runner_cls(eas.TASK)(wrapped, asdict(agent), device='cuda:0')
        runner.load(str(args.checkpoint), load_cfg={'actor': True}, map_location='cuda:0')
        learner = runner.alg.actor.eval().requires_grad_(False)
        opponent = deepcopy(learner)
        opponent.load_state_dict(torch.load(args.opponent_policy, map_location='cuda:0',
                                            weights_only=False)['actor_state_dict'])
        env.arena_opponent_policy = opponent
        env.reset(seed=args.seed)
        obs = wrapped.get_observations()
        event, t, step = 'Kickoff', 0., 0
        with torch.inference_mode():
            while len(rounds) < args.rounds:
                obs, _, dones, _ = wrapped.step(learner(obs))
                t += env.step_dt
                step += 1
                goal = float(env.arena_goal_result[0])
                fallen = not bool(eas.mdp.arena_upright(env)[0])
                if goal:
                    event = 'BLUE SCORES' if goal > 0 else 'ORANGE SCORES'
                done = bool(dones[0])
                if step % 2 == 0 or done:
                    status = event + ('   (blue down)' if fallen and not done else '')
                    frames.append(label(env.render(), [title, f'round {len(rounds) + 1}/{args.rounds}   t={t:4.1f}s   {status}']))
                if done:
                    outcome = event if goal or event.endswith('SCORES') else ('timeout' if t >= 9.99 else 'ball out / reset')
                    rounds.append({'round': len(rounds) + 1, 'seconds': round(t, 2), 'outcome': outcome})
                    frames.extend([frames[-1]] * 12)  # Hold the final frame ~0.5 s.
                    event, t = 'Kickoff', 0.
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.output, frames, fps=25, macro_block_size=8)
    args.output.with_suffix('.json').write_text(json.dumps({'title': title, 'rounds': rounds}, indent=2))
    print(json.dumps({'video': str(args.output), 'rounds': rounds}))


if __name__ == '__main__':
    main()
