#!/bin/bash
# Companion to submit_sweep.sh — evaluates every trained checkpoint at the
# official 1500-step horizon. Same env-var override conventions.
#
# Usage:
#     bash scripts/cluster/submit_eval_sweep.sh
#     FULL_SWEEP=1 bash scripts/cluster/submit_eval_sweep.sh

set -euo pipefail

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
N_EVAL=${N_EVAL:-20}
MAX_STEPS=${MAX_STEPS:-1500}
CONCURRENT=${CONCURRENT:-20}

read -r -a TASK_ARR <<< "$TASKS"
read -r -a SEED_ARR <<< "$SEEDS"
read -r -a BASELINE_ARR <<< "$BASELINES"

N=$(( ${#TASK_ARR[@]} * ${#SEED_ARR[@]} ))
LAST=$(( N - 1 ))

echo "[eval-sweep] baselines : ${BASELINES}"
echo "[eval-sweep] tasks     : ${TASKS}"
echo "[eval-sweep] seeds     : ${SEEDS}"
echo "[eval-sweep] array     : 0-${LAST}%${CONCURRENT} per baseline"
echo ""

for B in "${BASELINE_ARR[@]}"; do
    echo "[eval-sweep] submitting baseline=$B  ..."
    BASELINE="$B" TASKS="$TASKS" SEEDS="$SEEDS" \
    N_EVAL="$N_EVAL" MAX_STEPS="$MAX_STEPS" \
        sbatch --array="0-${LAST}%${CONCURRENT}" \
        --job-name="robomme-eval-${B}" \
        scripts/cluster/eval_array.sbatch
done
