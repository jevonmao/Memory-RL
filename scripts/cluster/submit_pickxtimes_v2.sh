#!/bin/bash
# Submit a "v2 reward" sweep for PickXtimes alongside the in-progress v1 sweep.
# V2 reduces the per-step proximity term 10× and boosts subgoal/pickup/terminal
# 5× to break the V1 reward-hacking failure.
#
# Output goes to runs/cluster/v2/* to avoid colliding with the v1 run dirs.

set -euo pipefail

SEEDS=${SEEDS:-"7 42 0"}
BASELINES=${BASELINES:-"b1 b4 b5 b6"}
TIMESTEPS=${TIMESTEPS:-1500000}
CONCURRENT=${CONCURRENT:-20}
N_ENVS=${N_ENVS:-4}
OUTDIR_ROOT=${OUTDIR_ROOT:-runs/cluster/v2}

read -r -a SEED_ARR <<< "$SEEDS"
read -r -a BASELINE_ARR <<< "$BASELINES"

N=${#SEED_ARR[@]}
LAST=$(( N - 1 ))

mkdir -p logs/slurm

for B in "${BASELINE_ARR[@]}"; do
    echo "[sweep-v2] submitting baseline=$B ..."
    # Pass ROBOMME_REWARD_VERSION=v2 through env so train/rewards/__init__.py
    # picks the V2 reward. Also override OUTDIR_ROOT so artifacts land in v2/.
    BASELINE="$B" TASKS="PickXtimes" SEEDS="$SEEDS" TIMESTEPS="$TIMESTEPS" \
        N_ENVS="$N_ENVS" OUTDIR_ROOT="$OUTDIR_ROOT" \
        sbatch --array="0-${LAST}%${CONCURRENT}" \
        --job-name="rme-${B}v2-pickx" \
        --export=ALL,ROBOMME_REWARD_VERSION=v2 \
        scripts/cluster/train_array.sbatch
done
