# Cluster scripts — SVL (Titan RTX, native Linux, ~20 GPUs)

Adapted from `cs153-project/scripts/selfplay_array.sbatch`. All cluster
operations live in this dir. Code-level changes are kept eval-compatible
with `challenge_interface/scripts/phase1_eval.py`.

## TL;DR

```bash
# 1) ON YOUR LOCAL MACHINE — push the repo to the cluster
bash scripts/cluster/sync_to_cluster.sh

# 2) ON THE CLUSTER (one-time) — install env + deps + smoke test
ssh jevon@scdt.stanford.edu
cd /vision/u/jevon/robomme_benchmark
bash scripts/cluster/setup.sh

# 3) ON THE CLUSTER — submit the default sweep (4 tasks × 2 baselines × 3 seeds = 24 GPUs)
bash scripts/cluster/submit_sweep.sh

# 4) ON THE CLUSTER, after training — eval everything you trained
bash scripts/cluster/submit_eval_sweep.sh
```

Default sweep uses one task per memory suite (BinFill / VideoUnmask /
PickHighlight / MoveCube) for a representative 24-job grid. Set
`FULL_SWEEP=1` to expand to all 16 tasks (96 jobs, 20 concurrent).

## Files

| File | Purpose |
|---|---|
| `setup.sh` | One-shot installer (conda env, torch+cu128, mani-skill fork, deps, wandb, smoke test). Idempotent. |
| `sync_to_cluster.sh` | `rsync` local repo → cluster, excludes `runs/`, `wandb/`, `.git/`, checkpoints |
| `train_array.sbatch` | Generic training array. `BASELINE`, `TASKS`, `SEEDS`, `TIMESTEPS` all overridable. |
| `eval_array.sbatch` | Generic eval array at the official 1500-step horizon. |
| `submit_sweep.sh` | Submits both `b1` and `b2` training arrays in one call (20-GPU cap). |
| `submit_eval_sweep.sh` | Companion eval submission for everything `submit_sweep.sh` trained. |

## What runs where

| Step | Where | Notes |
|---|---|---|
| `sync_to_cluster.sh` | **local** (WSL or any host with rsync + ssh access to scdt) | does NOT touch the cluster's `runs/` |
| `setup.sh` | **cluster login node** | run once; safe to re-run |
| `submit_sweep.sh` / `submit_eval_sweep.sh` | **cluster login node** | submits to SLURM, returns immediately |
| `train_array.sbatch` / `eval_array.sbatch` | **compute nodes** (via SLURM) | one GPU per array task |

## Submission patterns

### Default sweep — 4 tasks × 2 baselines × 3 seeds, 20 GPUs concurrent

```bash
bash scripts/cluster/submit_sweep.sh
```

This submits two arrays (`robomme-b1` and `robomme-b2`), each with 12 jobs (4 tasks × 3 seeds), `%20` cap.

### Single task, single baseline

```bash
BASELINES="b1" TASKS="BinFill" bash scripts/cluster/submit_sweep.sh
# or directly:
BASELINE=b1 TASKS="BinFill" sbatch --array=0-2 scripts/cluster/train_array.sbatch
```

### Full 16-task sweep — 96 jobs, 20 concurrent

```bash
FULL_SWEEP=1 bash scripts/cluster/submit_sweep.sh
```

5 waves of 20 × ~10h per job = ~50h wall to finish. WandB groups by baseline+task.

### Longer training (2M steps instead of 1M)

```bash
TIMESTEPS=2000000 bash scripts/cluster/submit_sweep.sh
```

Bump `--time=24:00:00` in `train_array.sbatch` to `48:00:00` if you go beyond 2M.

### Subset of seeds

```bash
sbatch --array=0,2 scripts/cluster/train_array.sbatch    # only array idx 0 and 2
```

### Eval only one baseline

```bash
BASELINES="b1" bash scripts/cluster/submit_eval_sweep.sh
```

## Eval-compatibility guardrails

These are baked into the sbatch scripts so all cluster runs produce
leaderboard-comparable checkpoints:

| | Setting | Why |
|---|---|---|
| `max_steps` | 1500 in env constructor, 1500 in eval | Matches `challenge_interface/scripts/phase1_eval.py` |
| `sim_freq` | unset (defaults to 100) | Matches official physics fidelity |
| Camera res | unset (defaults to native 256) | Matches official render distribution |
| `obs_mode` | `rgb` | Eval-neutral — wrapper only reads RGB anyway |
| Action space | `joint_angle` | Only `joint_angle`, `ee_pose`, `waypoint` allowed per submission rules |
| `vec_env` | `subproc` (Linux fork, no Windows workaround needed) | Throughput optimization, no policy impact |

## Resource sizing

| Parameter | Value | Why |
|---|---|---|
| `--gres=gpu:titanrtx:1` | 1 Titan RTX per array task | 24 GB VRAM holds 4 SAPIEN contexts + policy + ICM |
| `--cpus-per-task=8` | 8 CPUs | `n_envs=4` SubprocVecEnv × SAPIEN physx_cpu internal threads |
| `--mem=32G` | 32 GB | Per-worker SAPIEN ~1.5 GB resident + rollout buffer |
| `--time=24:00:00` | 24 h | 1M steps at projected 30-60 FPS = 5-9 h; 2M = 10-18 h |
| `--array=0-N%20` | 20 concurrent | Matches available GPU budget |

## Monitoring

```bash
# All your jobs:
squeue -u jevon -o "%.10i %.9P %.20j %.4T %.10M %.10L %R"

# Tail one running job's log:
tail -F logs/slurm/robomme-b1-<JOB_ID>_0.out

# Check FPS across all training jobs:
grep -H "fps " logs/slurm/robomme-b*_*.out | tail -20

# All checkpoints + completion status:
ls -la runs/cluster/*/ppo*_final.zip 2>/dev/null

# Eval results once they land:
ls runs/cluster/eval_results/*.json
```

## What to expect on Titan RTX

Turing-arch, ~50% the throughput of an RTX 4090. With native Linux SAPIEN
(no Windows-specific workarounds, faster Vulkan):

- ~30-60 FPS sustained at `n_envs=4`, `sim_freq=100`, max_steps=1500.
- 1M steps ≈ 5-9 h. 2M steps ≈ 10-18 h.
- 24-job sweep should comfortably finish in 12-18 h with 20 GPUs concurrent.

If FPS comes in dramatically below that range, ssh to the node and run `nvidia-smi` / `top` — most likely culprit is CPU contention (someone else's job on the same node) or an unexpected physx behavior.
