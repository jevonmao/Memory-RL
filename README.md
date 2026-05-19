# memory-RL

Infrastructure for the CS224R project **Curiosity-Driven Memory-Augmented RL for Adaptive Robot Tasks**.

This repo contains the environment adapter, vanilla PPO baseline, evaluation
pipeline, trajectory logger, and interface stubs that future memory (PTP)
and curiosity (ICM, M-ICM) modules plug into.

## Layout

```
env/           RoboMME Gymnasium adapter
agents/        (reserved for custom agents)
memory/        BaseMemory interface (PTP implementation TBD)
curiosity/     BaseCuriosity interface (ICM / M-ICM TBD)
metrics/       Evaluation metrics (success, return, redundancy, ...)
training/      train_ppo.py, evaluate.py, shared utils
configs/       YAML run configs
scripts/       inspect_env.py and other CLIs
data/          TrajectoryLogger + saved trajectories
logs/          run directories (created at runtime)
```

## Status

| Phase | Status |
| --- | --- |
| Scaffolding (env adapter, train, eval, logger, metrics) | done |
| CartPole CPU smoke (`inspect_env`, 5k-step PPO, `evaluate`, trajectory NPZ) | ✓ passing (2026-05-19) |
| RoboMME `inspect_env --task BinFill` | blocked on Modal GPU |
| Vanilla PPO baseline (4 tasks × 3 seeds × 100k) | blocked on Modal GPU |
| ICM curiosity module | not started |
| PTP memory + M-ICM | not started |

ManiSkill/SAPIEN requires GPU, so all RoboMME runs are deferred to Modal compute.
Local WSL2 is used only for CPU smoke tests and module development.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The active local env is conda `memrl` at `/home/jevon/miniforge3/envs/memrl/`
(sb3 2.8.0, torch 2.12.0+cu130, gymnasium 1.2.3).

### Getting RoboMME

The benchmark lives at <https://github.com/RoboMME/robomme_benchmark>. It is
not on PyPI; install from source. The repo's recommended path is `uv`:

```bash
git clone https://github.com/RoboMME/robomme_benchmark
cd robomme_benchmark
uv sync && uv pip install -e .
```

Alternatively, point this adapter at a local checkout without installing:

```bash
export ROBOMME_PATH=/abs/path/to/robomme_benchmark
# The adapter prepends $ROBOMME_PATH/src to sys.path so `import robomme` works.
```

RoboMME pulls a CUDA-12.8 ManiSkill fork and a pinned `torch==2.9.1`. If
that conflicts with the `memrl` env, use the upstream Dockerfile in the
RoboMME repo instead.

For pipeline smoke tests against a stock Gymnasium env, pass
`--allow-gym-fallback` and use a Gymnasium env id as the task name
(e.g. `CartPole-v1`). The adapter never silently substitutes — you must
opt in.

### RoboMME tasks (16 total)

| Suite | Focus | Tasks |
| --- | --- | --- |
| Counting | Temporal memory | `BinFill`, `PickXtimes`, `SwingXtimes`, `StopCube` |
| Permanence | Spatial memory | `VideoUnmask`, `VideoUnmaskSwap`, `ButtonUnmask`, `ButtonUnmaskSwap` |
| Reference | Object memory | `PickHighlight`, `VideoRepick`, `VideoPlaceButton`, `VideoPlaceOrder` |
| Imitation | Procedural memory | `MoveCube`, `InsertPeg`, `PatternLock`, `RouteStick` |

Action spaces: `joint_angle` (8-D), `ee_pose` (7-D), `waypoint` (7-D
discrete), `multi_choice` (VideoQA). The MLP-friendly Box wrapper supports
the first three; `multi_choice` needs the raw env (`flatten_obs: false`).

## Inspect a task

```bash
python scripts/inspect_env.py --task spatial_memory
python scripts/inspect_env.py --task CartPole-v1 --allow-gym-fallback
python scripts/inspect_env.py --task any --list-tasks
```

Prints task name, backend, observation / action spaces, sample observation
shapes, a random action, the reward/terminated/truncated after one step,
and any task metadata the env exposes.

## Train PPO

```bash
python training/train_ppo.py --config configs/ppo.yaml \
    --task spatial_memory --seed 0 --total_steps 100000
```

CLI flags (`--task`, `--seed`, `--total_steps`, `--output_dir`,
`--allow-gym-fallback`, `--tag`) override the YAML config. The run writes to
`logs/<task>_seed<seed>_<timestamp>[_<tag>]/` containing:

```
checkpoints/         periodic + best + final PPO .zip
checkpoints/best/    best model from EvalCallback
tb/                  TensorBoard logs
eval/                EvalCallback npz logs
config.yaml          merged config snapshot
```

## Evaluate a checkpoint

```bash
python training/evaluate.py \
    --checkpoint logs/<run>/checkpoints/ppo_final.zip \
    --task spatial_memory --episodes 20 --save_trajectories
```

Use `--random` instead of `--checkpoint` to evaluate a uniform-random
policy (useful for sanity-checking metrics before training).
Outputs `metrics.json` next to the checkpoint and, with
`--save_trajectories`, NPZ trajectories under `trajectories/`.

## Metrics

Implemented in `metrics/evaluation.py`:

- `success_rate` — fraction of episodes with `info["is_success"]` (or a
  positive-return terminal as fallback).
- `average_return`, `average_episode_length`.
- `average_steps_to_completion` — mean step index of first success, over
  successful episodes only.
- `redundancy_score` — `(visits - unique_states) / visits`, using
  `info["state_id"]` when the env exposes one, otherwise a hash of the
  rounded flattened observation (3 decimal places).

## Trajectory format

`data/trajectory_logger.py` writes per-episode NPZ files plus an
`index.jsonl` summary. Observations are stored either as a single array
`obs` or, for dict observations, as `obs__<key>` arrays. Per-step infos are
stored as a JSON-encoded `info_json` blob. Use `load_trajectories(out_dir)`
to stream them back in.

## Where things are saved

- Training: `logs/<run>/checkpoints/*.zip`, TensorBoard in `logs/<run>/tb/`
- Eval: `metrics.json` and `trajectories/` next to the checkpoint
- Random-policy eval: `logs/random_<task>_seed<seed>/`

## Known issues / important caveats

- **RoboMME's extrinsic reward is not used.** Per
  `doc/env_format.md` in the benchmark repo, the scalar reward is reserved
  and the official setup is imitation learning. Vanilla PPO will therefore
  see ~zero learning signal — this is exactly why the project layers
  curiosity (ICM) and memory-conditioned intrinsic rewards on top. Treat
  the PPO baseline here as scaffolding; expect flat returns until intrinsic
  rewards are wired in.
- **Each `make_env_for_episode` is bound to one episode.** This wrapper
  rebuilds the inner env on every `reset()`, cycling `episode_idx` mod
  `episode_num`. Pin a single episode with `env_kwargs.episode_idx`.
- **Observations are dict-of-lists.** With `flatten_obs: true` (default) we
  take the last frame and concatenate `eef_state_list`, `joint_state_list`,
  `gripper_state_list` into a Box vector. To use pixels, set
  `flatten_obs: false` and provide a custom policy.
- `robomme` is not on PyPI; install from source or use `ROBOMME_PATH`.
- `stable-baselines3`, `torch`, `gymnasium` must be installed
  (`pip install -r requirements.txt`).
- Headless rendering / ManiSkill may require system libs (EGL, MuJoCo,
  CUDA 12.8 drivers). The RoboMME repo provides a Dockerfile that bundles
  everything if local install is painful.
