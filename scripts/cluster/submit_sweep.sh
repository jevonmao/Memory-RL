#!/bin/bash
# One-line sweep submission. Picks the (tasks, seeds, baselines) grid,
# computes array sizes, submits both training arrays under a 20-GPU cap.
#
# Defaults: 4-task subset (one per memory suite) × {B1, B2} × 3 seeds = 24 jobs.
# Override TASKS / SEEDS / BASELINES / TIMESTEPS via env vars.
# FULL_SWEEP=1 expands to all 16 tasks (= 96 jobs total).
#
# Usage:
#     bash scripts/cluster/submit_sweep.sh
#     TIMESTEPS=2000000 bash scripts/cluster/submit_sweep.sh
#     FULL_SWEEP=1 bash scripts/cluster/submit_sweep.sh
#     TASKS="BinFill" BASELINES="b1 b2" bash scripts/cluster/submit_sweep.sh
#     CONCURRENT=10 bash scripts/cluster/submit_sweep.sh   # cap to 10 GPUs

set -euo pipefail

# 4-task subset hits one task per memory suite (Counting / Permanence /
# Reference / Imitation) — see episode_config_resolver._DEFAULT_TASK_LIST.
SUBSET_TASKS="BinFill VideoUnmask PickHighlight MoveCube"

FULL_TASK_LIST="PickXtimes StopCube SwingXtimes BinFill \
VideoUnmaskSwap VideoUnmask ButtonUnmaskSwap ButtonUnmask \
VideoRepick VideoPlaceButton VideoPlaceOrder PickHighlight \
InsertPeg MoveCube PatternLock RouteStick"

if [[ "${FULL_SWEEP:-0}" == "1" ]]; then
    TASKS=${TASKS:-$FULL_TASK_LIST}
else
    TASKS=${TASKS:-$SUBSET_TASKS}
fi

SEEDS=${SEEDS:-"7 42 0"}
BASELINES=${BASELINES:-"b1 b2"}
TIMESTEPS=${TIMESTEPS:-1000000}
CONCURRENT=${CONCURRENT:-20}

read -r -a TASK_ARR <<< "$TASKS"
read -r -a SEED_ARR <<< "$SEEDS"
read -r -a BASELINE_ARR <<< "$BASELINES"

N=$(( ${#TASK_ARR[@]} * ${#SEED_ARR[@]} ))
LAST=$(( N - 1 ))

echo "[sweep] baselines : ${BASELINES}"
echo "[sweep] tasks     : ${TASKS}"
echo "[sweep] seeds     : ${SEEDS}"
echo "[sweep] timesteps : ${TIMESTEPS}"
echo "[sweep] array     : 0-${LAST}%${CONCURRENT} (= ${N} jobs per baseline)"
echo "[sweep] total     : $(( N * ${#BASELINE_ARR[@]} )) jobs"
echo ""

for B in "${BASELINE_ARR[@]}"; do
    echo "[sweep] submitting baseline=$B  ..."
    BASELINE="$B" TASKS="$TASKS" SEEDS="$SEEDS" TIMESTEPS="$TIMESTEPS" \
        sbatch --array="0-${LAST}%${CONCURRENT}" \
        --job-name="robomme-${B}" \
        scripts/cluster/train_array.sbatch
done
