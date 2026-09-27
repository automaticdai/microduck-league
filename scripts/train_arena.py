"""Fine-tune the 61D football actor in the shared Microduck Go arena.

Run --num-envs 64 --iterations 5 before any longer run. A frozen opponent is
initialized from the approved football actor and refreshed every 250 iterations.
Half the worlds keep playing the original opponent to limit forgetting.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
import subprocess
import sys
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
import mjlab_microduck.tasks  # noqa

TASK = 'Mjlab-FootballArena-Flat-MicroDuck'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-policy',type=Path,required=True)
    parser.add_argument('--resume',type=Path,help='Full resume within the same reward recipe only')
    parser.add_argument('--opponent-policy',type=Path,help='Fixed reference opponent; defaults to from-policy')
    parser.add_argument('--eval-every',type=int,default=0,help='Evaluate checkpoints against the fixed opponent every N iterations')
    parser.add_argument('--num-envs',type=int,default=64)
    parser.add_argument('--iterations',type=int,default=5)
    parser.add_argument('--opponent-update-every',type=int,default=250)
    parser.add_argument('--run-name',default='arena-smoke')
    parser.add_argument('--evaluate-seconds',type=float,default=0)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.iterations <= 0 or args.opponent_update_every <= 0:
        parser.error('Environment count, iterations and opponent update interval must be positive')
    cfg,agent = load_env_cfg(TASK),load_rl_cfg(TASK)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = agent.seed
    cfg.sim.nan_guard.enabled = args.iterations <= 5
    agent.logger = 'tensorboard'
    agent.run_name = args.run_name
    agent.save_interval = 50
    out = Path('logs/rsl_rl/football_arena') / (datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_'+args.run_name)
    out.mkdir(parents=True)
    dump_yaml(out/'params/env.yaml',asdict(cfg))
    dump_yaml(out/'params/agent.yaml',asdict(agent))
    (out/'initialization.json').write_text(json.dumps({**vars(args), 'task':TASK,
        'recipe':'goal-first-v2','actor_only_initialization':not bool(args.resume)},default=str,indent=2))
    configure_torch_backends()
    env = ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        wrapped = RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
        runner = load_runner_cls(TASK)(wrapped,asdict(agent),str(out),device='cuda:0')
        runner.load(str(args.from_policy),load_cfg={'actor':True},map_location='cuda:0')
        baseline = deepcopy(runner.alg.actor).eval().requires_grad_(False)
        opponent = deepcopy(baseline).eval().requires_grad_(False)
        if args.opponent_policy:
            baseline.load_state_dict(torch.load(args.opponent_policy,map_location='cuda:0',weights_only=False)['actor_state_dict'])
        if args.resume:
            runner.load(str(args.resume),map_location='cuda:0')
            runner.current_learning_iteration += 1
            opponent_file = args.resume.with_name(args.resume.stem+'_opponent.pt')
            if not opponent_file.exists():
                raise FileNotFoundError(f'Resume requires paired opponent: {opponent_file}')
            opponent.load_state_dict(torch.load(opponent_file,map_location='cuda:0',weights_only=True))
        else:
            env.common_step_counter = 0
            runner.current_learning_iteration = 0
        mask = (torch.arange(env.num_envs,device='cuda:0') % 2 == 0)[:,None]
        def opponent_policy(obs):
            if args.evaluate_seconds:
                return baseline(obs)
            return torch.where(mask,baseline(obs),opponent(obs))
        env.arena_opponent_policy = opponent_policy
        original_save = runner.save
        last_refresh = runner.current_learning_iteration
        evaluated = set()
        def save(path,infos=None):
            nonlocal last_refresh
            if runner.current_learning_iteration-last_refresh >= args.opponent_update_every:
                opponent.load_state_dict(runner.alg.actor.state_dict())
                last_refresh = runner.current_learning_iteration
                print(f'Opponent refreshed at iteration {last_refresh}; 50% retain baseline',flush=True)
            original_save(path,infos)
            torch.save(opponent.state_dict(),Path(path).with_name(Path(path).stem+'_opponent.pt'))
            iteration = runner.current_learning_iteration
            if args.eval_every and iteration > 0 and iteration % args.eval_every == 0 and iteration not in evaluated:
                evaluated.add(iteration)
                command = [sys.executable,'-u','scripts/train_arena.py',
                    '--from-policy',str(path),'--opponent-policy',str(args.opponent_policy or args.from_policy),
                    '--num-envs','64','--evaluate-seconds','20','--run-name',f'{args.run_name}-eval-{iteration}']
                with (out/f'eval_{iteration}.log').open('w') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                print(f'Fixed-opponent evaluation saved: {out / f"eval_{iteration}.log"}',flush=True)
        runner.save = save
        print(f'Training output: {out}',flush=True)
        if args.evaluate_seconds:
            obs = wrapped.get_observations()
            scored = conceded = episodes = 0
            upright_sum = speed_sum = ball_speed_sum = near_ball_sum = 0.
            strikes = timeouts = 0
            with torch.inference_mode():
                for _ in range(round(args.evaluate_seconds/env.step_dt)):
                    action = runner.alg.actor(obs)
                    obs,reward,done,extras = wrapped.step(action)
                    # Termination buffers survive auto-reset; score flags do not.
                    episodes += int(done.sum())
                    scored += int((env.arena_goal_result > 0).sum())
                    conceded += int((env.arena_goal_result < 0).sum())
                    diagnostic = env.arena_step_diagnostics
                    upright_sum += float(diagnostic['standing'].float().mean())
                    speed_sum += float(diagnostic['robot_speed'].mean())
                    ball_speed_sum += float(diagnostic['ball_speed'].mean())
                    near_ball_sum += float(diagnostic['near_ball'].float().mean())
                    strikes += int(diagnostic['strike'].sum())
                    timeouts += int(env.reset_time_outs.sum())
            report = {'episodes':episodes,'scored':scored,'conceded':conceded,
                      'standing_fraction':upright_sum/round(args.evaluate_seconds/env.step_dt),
                      'mean_robot_speed':speed_sum/round(args.evaluate_seconds/env.step_dt),
                      'mean_ball_speed':ball_speed_sum/round(args.evaluate_seconds/env.step_dt),
                      'near_ball_fraction':near_ball_sum/round(args.evaluate_seconds/env.step_dt),
                      'supported_strikes':strikes,'timeouts':timeouts}
            (out/'evaluation.json').write_text(json.dumps(report,indent=2))
            print(report,flush=True)
        else:
            runner.add_git_repo_to_log(__file__)
            runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)

    finally:
        env.close()


if __name__ == '__main__':
    main()
