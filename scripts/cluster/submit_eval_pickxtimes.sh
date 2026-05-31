#!/bin/bash
# Eval all PickXtimes baselines after training completes.
#
# Each baseline × 3 seeds = 3 eval jobs at ~20 minutes each.
# Total: 4 × 3 = 12 eval jobs.
#
# By default evaluates the final checkpoint; pass CKPT_PATTERN to override
# (e.g. CKPT_PATTERN="*_500000_steps.zip" for an intermediate snapshot).
#
# Usage:
#     bash scripts/cluster/submit_eval_pickxtimes.sh
#     BASELINES="b6" bash scripts/cluster/submit_eval_pickxtimes.sh   # only headline

set -euo pipefail

TASKS=${TASKS:-"PickXtimes"}
SEEDS=${SEEDS:-"7 42 0"}
BASELINES=${BASELINES:-"b1 b4 b5 b6"}
N_EVAL=${N_EVAL:-20}
MAX_STEPS=${MAX_STEPS:-1500}
CONCURRENT=${CONCURRENT:-20}

read -r -a TASK_ARR <<< "$TASKS"
read -r -a SEED_ARR <<< "$SEEDS"
read -r -a BASELINE_ARR <<< "$BASELINES"

N=$(( ${#TASK_ARR[@]} * ${#SEED_ARR[@]} ))
LAST=$(( N - 1 ))

echo "[eval] baselines  : ${BASELINES}"
echo "[eval] tasks      : ${TASKS}"
echo "[eval] seeds      : ${SEEDS}"
echo "[eval] n_eval     : ${N_EVAL}"
echo "[eval] max_steps  : ${MAX_STEPS}"
echo "[eval] array      : 0-${LAST}%${CONCURRENT} (= ${N} jobs per baseline)"
echo "[eval] total      : $(( N * ${#BASELINE_ARR[@]} )) jobs"
echo ""

for B in "${BASELINE_ARR[@]}"; do
    echo "[eval] submitting baseline=$B ..."
    BASELINE="$B" TASKS="$TASKS" SEEDS="$SEEDS" N_EVAL="$N_EVAL" MAX_STEPS="$MAX_STEPS" \
        sbatch --array="0-${LAST}%${CONCURRENT}" \
        --job-name="rme-eval-${B}" \
        scripts/cluster/eval_array.sbatch
done
