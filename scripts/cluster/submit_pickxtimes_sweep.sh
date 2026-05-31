#!/bin/bash
# Submit the PickXtimes baseline sweep for the memory+curiosity comparison.
#
# Sends:
#   b1: vanilla PPO                 (control)
#   b4: PPO + RND                   (curiosity only)
#   b5: RecurrentPPO                (memory only)
#   b6: RecurrentPPO + RND          (memory + curiosity → the headline run)
#
# Each × 3 seeds = 12 jobs.
#
# Usage:
#     bash scripts/cluster/submit_pickxtimes_sweep.sh
#     TIMESTEPS=3000000 bash scripts/cluster/submit_pickxtimes_sweep.sh
#     BASELINES="b6" bash scripts/cluster/submit_pickxtimes_sweep.sh   # only headline

set -euo pipefail

TASKS=${TASKS:-"PickXtimes"}
SEEDS=${SEEDS:-"7 42 0"}
BASELINES=${BASELINES:-"b1 b4 b5 b6"}
TIMESTEPS=${TIMESTEPS:-1500000}
CONCURRENT=${CONCURRENT:-20}
N_ENVS=${N_ENVS:-4}

read -r -a TASK_ARR <<< "$TASKS"
read -r -a SEED_ARR <<< "$SEEDS"
read -r -a BASELINE_ARR <<< "$BASELINES"

N=$(( ${#TASK_ARR[@]} * ${#SEED_ARR[@]} ))
LAST=$(( N - 1 ))

echo "[sweep] baselines  : ${BASELINES}"
echo "[sweep] tasks      : ${TASKS}"
echo "[sweep] seeds      : ${SEEDS}"
echo "[sweep] timesteps  : ${TIMESTEPS}"
echo "[sweep] n_envs     : ${N_ENVS}"
echo "[sweep] array      : 0-${LAST}%${CONCURRENT} (= ${N} jobs per baseline)"
echo "[sweep] total      : $(( N * ${#BASELINE_ARR[@]} )) jobs"
echo ""

for B in "${BASELINE_ARR[@]}"; do
    echo "[sweep] submitting baseline=$B ..."
    BASELINE="$B" TASKS="$TASKS" SEEDS="$SEEDS" TIMESTEPS="$TIMESTEPS" N_ENVS="$N_ENVS" \
        sbatch --array="0-${LAST}%${CONCURRENT}" \
        --job-name="rme-${B}-pickx" \
        scripts/cluster/train_array.sbatch
done
