# Microduck Go football arena

Two independently controlled ducks share one physical ball and two goals.
The pitch is 1.8 × 1.3 m, with a 22 cm radius center circle. Each kickoff
samples the ball center uniformly over that disk. A goal or 10 seconds ends
a round; falling and leaving the pitch do not. Numerical failures remain
training safety terminations and are reported separately from goals.

The initial policy was trained for individual shots. Arena fine-tuning keeps
its 61D actor observation contract and 14 actions. A ball-and-goal tracker
supplies navigation commands. The actor has no explicit opponent-position
input; the training critic additionally observes opponent position/velocity.
This is a first competition recipe, not yet evidence of learned tactics.

## Three-a-side arena

`Mjlab-FootballArena-3v3-Flat-MicroDuck` places six independently actuated ducks on
the existing 1.8 × 1.3 m pitch: two attackers and one goal-line defender per team.
Both teams share one ball. Boards keep it in play; goals and 10-second timeouts
reset all six players to mirrored formations. Numerical failures also reset the round.
Attackers use the ball tracker and aim beside the opposing defender; defenders use
the existing keeper tracker and a separately loaded keeper policy. These are fixed
roles using existing policies, not newly learned passing or coordinated tactics.

Replay with team-colored goals, floating triangles, and role labels:

```bash
uv run scripts/play_football_arena_gpu.py --teams \
  --checkpoint path/to/attacker/model_4699.pt \
  --defender-policy path/to/keeper/model_2699.pt \
  --compile-friction --port 8081
```

The same attacker checkpoint controls all four attackers unless `--opponent-policy`
selects a different checkpoint for the two orange attackers. Both defenders use
`--defender-policy`. Open-gap aiming is always enabled in 3v3.

Training updates only the primary blue attacker; its blue teammate and both
defenders stay frozen. Orange attackers use the existing fixed-reference/self-play
selection. Each player retains its own action and velocity history. The actor
remains 61D with 14 actions; the critic additionally sees all five other players.
The learner inherits the base arena's DR, observation noise, BAM actuators and
reward stack; teammates and opponents use clean inference observations.

```bash
# Required smoke test before any longer training run.
uv run scripts/train_arena.py --task Mjlab-FootballArena-3v3-Flat-MicroDuck \
  --from-policy path/to/attacker/model_4699.pt \
  --keeper-policy path/to/keeper/model_2699.pt \
  --num-envs 64 --iterations 5 --run-name 3v3-smoke
```

For sustained fine-tuning, use `--critic-warmup` as with the other arena tasks.
The 3v3 task has its own `football_arena_3v3` experiment directory and requires
`--keeper-policy`. It does not support the two-player `--mode-probs` switch.

## Train on a CUDA server

Run these from the repository root after `uv sync --locked`:

```bash
# Required smoke test before a long run.
uv run scripts/train_arena.py --from-policy path/to/football/model_4999.pt \
  --num-envs 64 --iterations 5 --run-name arena-smoke

# Establish the source policy's deterministic match baseline.
uv run scripts/train_arena.py --from-policy path/to/football/model_4999.pt \
  --num-envs 64 --evaluate-seconds 30 --run-name arena-baseline

# Fine-tune in 2,048 shared arenas for 5,000 iterations.
uv run scripts/train_arena.py --from-policy path/to/football/model_4999.pt \
  --num-envs 2048 --iterations 5000 --run-name arena-selfplay
```

Actor weights and their normalizer transfer together. The critic, optimizer,
and training counters start fresh. Half the worlds retain a fixed reference opponent selected with `--opponent-policy`
(default: `--from-policy`); the other half refresh their frozen opponent at the
`--opponent-update-every` interval (default: 250 iterations).
Both ducks use BAM actuators, and the learner inherits the walking/football
observation noise and domain randomization stack.

Checkpoints are saved every 50 iterations in
`logs/rsl_rl/football_arena/<timestamp>_<run-name>/`. Each `model_N.pt` has a
paired `model_N_opponent.pt`. Resume with the original `--from-policy` and
`--resume path/to/model_N.pt`; `--iterations` means additional iterations.
Keep both checkpoint files together to restore the opponent correctly.

## Goal-first reward revision

The first arena run learned balance but scored infrequently. Revision 2 starts
from its checkpoint 1,000, transferring the actor and normalizer and rebuilding
the critic/optimizer for the new reward targets. Its clock restarts at zero;
4,000 new iterations means roughly 5,000 total iterations of arena experience.

| Signal | Reward |
|---|---|
| Goal / conceded goal | +100 / −100 once |
| Posture and command tracking | At most +0.8 per second combined |
| First supported right-foot strike moving goalward | +3 once |
| Ball speed toward goal after that strike | Up to +3, paid only for new best speed |
| Ball distance toward goal after that strike | +15 per meter of new progress, capped at 1.2 m |
| Approach to the kick stance | +5 per meter of new progress |
| Fallen | −5 per second |
| Timeout without a goal | −5 once |

Ball progress measures distance to the goal mouth, not just forward X motion.
Only upright, supported strikes unlock shot shaping. Actual goals still count
for either team regardless of who touched the ball last. Standing still and
repeatedly tapping the ball cannot repeat the one-time/high-water bonuses.

The 10-second timeout is a finite episode ending. The critic observes remaining
time, while the actor stays 61D. PPO's discount is 0.997 at 50 Hz, extending the
credit horizon to about 6.7 seconds from the previous 2 seconds.

```bash
# Required short check for the revised rewards.
uv run scripts/train_arena.py --from-policy path/to/arena/model_1000.pt \
  --opponent-policy path/to/football/model_4999.pt \
  --num-envs 64 --iterations 5 --run-name arena-v2-smoke

# Four thousand NEW iterations; initialize weights, do not resume old values.
uv run scripts/train_arena.py --from-policy path/to/arena/model_1000.pt \
  --opponent-policy path/to/football/model_4999.pt \
  --num-envs 2048 --iterations 4000 --opponent-update-every 500 \
  --eval-every 250 --run-name arena-v2-from1000
```

`--eval-every 250` evaluates saved actors against the same fixed reference
opponent for 20 simulated seconds in 64 arenas. Reports include goals, conceded
goals, standing fraction, supported strikes, ball speed and timeouts. Training
pauses during these evaluations; inspect `eval_N.log` in the training directory.
Do not directly compare these fixed-opponent evaluations to mixed-opponent
training metrics. `--resume` is for checkpoints from the same reward recipe;
use actor-only `--from-policy` when switching from revision 1 to revision 2.

Monitor `arena_scored`, `arena_conceded`, `arena_standing`, `arena_struck`, the
negative `arena_fallen` and `arena_timeout` costs, and numerical-safety endings.
A fall costs reward but does not restart the round.

## Inspect a trained contender

Export through the normalizer-aware exporter:

```bash
uv run scripts/export.py Mjlab-FootballArena-Flat-MicroDuck \
  --checkpoint-file path/to/arena/model_4999.pt --num-envs 1 \
  --onnx-file logs/arena_policy.onnx

uv run scripts/play_football_arena.py --policy logs/arena_policy.onnx --port 8081
```

Open <http://localhost:8081>. The CPU viewer uses the selected policy for both
teams and shows the cumulative Blue x:y Orange score. Training uses a learner
against a frozen opponent, so the viewer's match results are a separate test.
The CPU arena's transfer from GPU training is still under validation; inspect
both match metrics and visible balance before treating a checkpoint as better.

## Local NVIDIA GPU viewer

For playback with CUDA physics and CUDA policy inference (rather than the CPU
MuJoCo viewer), use the checkpoint directly:

```bash
uv run scripts/play_football_arena_gpu.py \
  --checkpoint logs/arena_v2_model_250.pt --compile-friction --port 8081
```

This uses the training BAM actuator model on the GPU and keeps the pitch,
center-circle spawns, 10-second rounds and cumulative score. The sidebar names
the CUDA device and reports measured playback speed; 1.00× means real time.
Training reward/critic calculations and diagnostic sensors are omitted during
playback. Browser drawing remains WebGL on the client machine.

`--compile-friction` fuses the BAM friction arithmetic without changing its
formula. The first launch compiles kernels before opening the viewer. To measure
simulation throughput without the browser, add `--benchmark-steps 250`.

## Kick-first v3, learned keeper, and the shot eval (2026-09-30)

`scripts/eval_arena_shots.py` is the arena scoreboard. It records the first round of every
arena (no bias toward short rounds), tracks every learner touch to its outcome (goal, wide,
out, blocked, re-touch), and reports fall timing. Use it against three opponents
(`--solo`, `--opponent-role keeper`, default attacker), with `--aim open` and 2048 rounds.
GPU rollouts are not repeatable run to run, so compare rates with their standard errors.

Findings that shaped v3:
- v2 lost the source kick (solo goals 83% → 37%). The cause was fine-tuning against a
  fresh, unfitted critic, not the reward. `train_arena.py --critic-warmup N` freezes the
  actor while the critic fits; always use it when the critic starts from scratch.
- Rounds were decided by about 2 s, then ran dead: the ball left play with no walls and
  near-zero rolling friction. v3 ends the round when the ball goes out and charges a fall
  once (−10) instead of per second.
- The tracker aimed at the goal center, which is exactly where a keeper stands.
  `ArenaCommandCfg.aim='open'` aims at the widest gap beside the opponent's shadow.

Tasks:
- `Mjlab-FootballArena-V3-Flat-MicroDuck` (kick-first-v3): starts from the football kicker
  and stages opponents per arena (empty goal → keeper → attacker, `V3_MODE_STAGES`).
  `train_arena.py --mode-probs S K A` overrides the mix.
- `Mjlab-FootballArena-Keeper-Flat-MicroDuck` (keeper-v1): the learner keeps the −x goal
  against a frozen attacker (`--no-self-play`). Conceding costs −100; a clean round pays +5.

Results (goals within 10 s, 2048 rounds, open aim; source kicker football@4999 shown first):
empty goal 83% → 88% (attacker @3699); vs scripted keeper 30% → 64%; vs attacker
9.8%/7.5% → 12.9%/8.2% (scored/conceded). The learned keeper @1699 lets in 12.5% of v3@2199
shots, against 54.8% for the scripted keeper at the same speed limits.

Viewer: `play_football_arena_gpu.py` takes `--opponent-role {attacker,keeper,solo}`,
`--opponent-policy`, `--aim`, plus boards around the pitch (`--no-boards` to remove them) and a
2 s play-on after goals (`--goal-hold`). Training uses neither by default.

Open issue: StandUp (3000–9000 iterations) rises from sitting but not from lying down.
From face-down it reaches standing height but stays pitched about 34°, so a get-up swap
(`eval_arena_shots.py --getup-policy`) recovers only 3–4% of falls.
