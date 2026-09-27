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
