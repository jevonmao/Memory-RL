#!/bin/bash
# Unified training array: parameterized by (BASELINE, TASKS, SEEDS).
# Array index maps to (task, seed) grid:  i -> (TASKS[i % nT], SEEDS[i // nT])
#
# Submit examples:
#     # B1 vanilla PPO, 1 task × 3 seeds  (3 jobs):
#     BASELINE=b1 TASKS=BinFill sbatch --array=0-2 scripts/cluster/train_array.sbatch
#
#     # B2 PPO+ICM, 4 tasks × 3 seeds  (12 jobs, 12 GPUs):
#     BASELINE=b2 TASKS="BinFill MoveCube VideoUnmask PickHighlight" \
#       sbatch --array=0-11 scripts/cluster/train_array.sbatch
#
#     # full 16-task × 3-seed sweep, 20 concurrent:
#     BASELINE=b1 TASKS="$(./scripts/cluster/all_tasks.sh)" \
#       sbatch --array=0-47%20 scripts/cluster/train_array.sbatch
#
# Env vars (all optional, defaults shown):
#     BASELINE=b1          # 'b1' (vanilla PPO) or 'b2' (PPO+ICM)
#     TASKS="BinFill"      # space-separated task ids
#     SEEDS="7 42 0"       # space-separated seeds
#     TIMESTEPS=1000000
#     N_ENVS=4

#SBATCH --account=vision --partition=svl --qos=normal
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=robomme
#SBATCH --output=logs/slurm/%x-%A_%a.out

set -euo pipefail

# --------- parameters ---------
BASELINE=${BASELINE:-b1}
TASKS=${TASKS:-"BinFill"}
SEEDS=${SEEDS:-"7 42 0"}
TIMESTEPS=${TIMESTEPS:-1000000}
N_ENVS=${N_ENVS:-4}
OUTDIR_ROOT=${OUTDIR_ROOT:-runs/cluster}

read -r -a TASK_ARR <<< "$TASKS"
read -r -a SEED_ARR <<< "$SEEDS"
nT=${#TASK_ARR[@]}
nS=${#SEED_ARR[@]}

S=$SLURM_ARRAY_TASK_ID
TASK=${TASK_ARR[$(( S % nT ))]}
SEED=${SEED_ARR[$(( S / nT ))]}

# --------- dispatch on baseline ---------
case "$BASELINE" in
    b1)
        MODULE="train.train_ppo"
        RUN_NAME="ppo_b1_${TASK}_s${SEED}"
        ;;
    b2)
        MODULE="train.train_ppo_icm"
        RUN_NAME="ppo_b2_icm_${TASK}_s${SEED}"
        ;;
    b3)
        MODULE="train.train_ppo_ptp"
        RUN_NAME="ppo_b3_ptp_${TASK}_s${SEED}"
        ;;
    b4)
        MODULE="train.train_ppo_rnd"
        RUN_NAME="ppo_b4_rnd_${TASK}_s${SEED}"
        ;;
    b5)
        MODULE="train.train_ppo_recurrent"
        RUN_NAME="ppo_b5_recurrent_${TASK}_s${SEED}"
        ;;
    b6)
        MODULE="train.train_ppo_recurrent_rnd"
        RUN_NAME="ppo_b6_recurrent_rnd_${TASK}_s${SEED}"
        ;;
    *)
        echo "[error] unknown BASELINE=$BASELINE (expected b1/b2/b3/b4/b5/b6)" >&2
        exit 1
        ;;
esac

OUTDIR="${OUTDIR_ROOT}/${RUN_NAME}"

# --------- env ---------
source /vision/u/jevon/miniconda3/etc/profile.d/conda.sh
conda activate robomme
export PYTHONNOUSERSITE=1
# Eval-neutral throughput knob: skip depth+segmentation render passes.
# (Wrapper only reads RGB; leaves policy distribution unchanged.)
export ROBOMME_OBS_MODE=rgb
# Force CPU physx: sapien 3.0.3 + torch 2.9 trip a __cuda_array_interface__
# typestr parse error inside articulation_joint.drive_target on GPU sim.
# CPU sim is fast enough (~70-100 FPS aggregate at n_envs=4) and is
# behavior-equivalent for our reward / eval pipeline (eval uses CPU too).
export ROBOMME_SIM_BACKEND=physx_cpu
# Leave ROBOMME_CAMERA_RES + ROBOMME_SIM_FREQ unset so SAPIEN uses defaults
# (camera 256 native, sim_freq=100) — these MUST match phase1_eval.py for the
# trained policy to be leaderboard-comparable.

cd /vision/u/jevon/robomme_benchmark
PY=/vision/u/jevon/miniconda3/envs/robomme/bin/python

mkdir -p "$OUTDIR" logs/slurm

echo "================================================================"
echo "robomme-$BASELINE | task=$TASK seed=$SEED steps=$TIMESTEPS n_envs=$N_ENVS"
echo "array_task=$S  node=$SLURM_JOB_NODELIST  gpu=$CUDA_VISIBLE_DEVICES"
echo "outdir=$OUTDIR  module=$MODULE"
echo "================================================================"

$PY -m "$MODULE" \
    --task "$TASK" \
    --timesteps "$TIMESTEPS" \
    --n_envs "$N_ENVS" \
    --vec_env subproc \
    --seed "$SEED" \
    --outdir "$OUTDIR" \
    --wandb \
    --wandb_project robomme-benchmark \
    --wandb_name "${RUN_NAME}"
