#!/bin/bash
# Two-stage BC pretrain on a single GPU node:
#   1. Collect ~80 planner trajectories on PickXtimes (slow, CPU-bound)
#   2. Train BC policy (~10 min on GPU)
# Output state_dict saved to runs/bc/pickxtimes/bc_policy{,_recurrent}.pt
# Used by train_ppo_recurrent_rnd.py via --bc_warmstart.
#
# Submit:  sbatch scripts/cluster/bc_pretrain_job.sbatch
#SBATCH --account=vision --partition=svl --qos=normal
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=rme-bc
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

source /vision/u/jevon/miniconda3/etc/profile.d/conda.sh
conda activate robomme
export PYTHONNOUSERSITE=1
export ROBOMME_OBS_MODE=rgb
export ROBOMME_SIM_BACKEND=physx_cpu
export WANDB_MODE=disabled

cd /vision/u/jevon/robomme_benchmark
# -u for unbuffered stdout so we see progress live (sbatch redirects pipe is
# block-buffered without it, which makes "stuck?" diagnostics impossible).
PY="/vision/u/jevon/miniconda3/envs/robomme/bin/python -u"

N_EPISODES=${N_EPISODES:-80}
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-256}
OUT=${OUT:-runs/bc/pickxtimes}

mkdir -p "$OUT" logs/slurm

echo "================================================================"
echo "BC pretrain (PickXtimes) | n_ep=$N_EPISODES epochs=$EPOCHS"
echo "node=$SLURM_JOB_NODELIST gpu=$CUDA_VISIBLE_DEVICES"
echo "================================================================"

if [[ ! -f "$OUT/bc_dataset_pickxtimes.pt" ]]; then
    echo ""
    echo "--- Stage 1/2: collecting $N_EPISODES expert episodes ---"
    $PY -m scripts.collect_bc_pickxtimes --n_episodes "$N_EPISODES" --out "$OUT"
else
    echo "[bc] reusing existing $OUT/bc_dataset_pickxtimes.pt"
fi

echo ""
echo "--- Stage 2/2: BC non-recurrent ---"
$PY -m scripts.bc_pretrain_pickxtimes \
    --dataset "$OUT/bc_dataset_pickxtimes.pt" \
    --out "$OUT/bc_policy.pt" \
    --epochs "$EPOCHS" --batch_size "$BATCH_SIZE"

echo ""
echo "--- Stage 2/2: BC recurrent (for headline b6) ---"
$PY -m scripts.bc_pretrain_pickxtimes \
    --dataset "$OUT/bc_dataset_pickxtimes.pt" \
    --out "$OUT/bc_policy_recurrent.pt" \
    --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --recurrent

echo ""
echo "BC PRETRAIN COMPLETE"
ls -la "$OUT"
