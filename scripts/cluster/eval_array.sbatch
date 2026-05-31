#!/bin/bash
# Eval a (BASELINE × TASKS × SEEDS) grid at the official 1500-step horizon.
# Looks for checkpoints at runs/cluster/<prefix>_<TASK>_s<SEED>/...zip.
#
# Submit examples:
#     # eval B1 on BinFill across all 3 seeds:
#     BASELINE=b1 TASKS=BinFill sbatch --array=0-2 scripts/cluster/eval_array.sbatch
#
#     # eval B2 on the 4-task subset across all 3 seeds:
#     BASELINE=b2 TASKS="BinFill MoveCube VideoUnmask PickHighlight" \
#       sbatch --array=0-11 scripts/cluster/eval_array.sbatch

#SBATCH --account=vision --partition=svl --qos=normal
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=robomme-eval
#SBATCH --output=logs/slurm/eval-%A_%a.out

set -euo pipefail

BASELINE=${BASELINE:-b1}
TASKS=${TASKS:-"BinFill"}
SEEDS=${SEEDS:-"7 42 0"}
N_EVAL=${N_EVAL:-10}      # matches phase1_eval.py Phase 1 episode count
MAX_STEPS=${MAX_STEPS:-1500}
OUTDIR_ROOT=${OUTDIR_ROOT:-runs/cluster}

read -r -a TASK_ARR <<< "$TASKS"
read -r -a SEED_ARR <<< "$SEEDS"
nT=${#TASK_ARR[@]}

S=$SLURM_ARRAY_TASK_ID
TASK=${TASK_ARR[$(( S % nT ))]}
SEED=${SEED_ARR[$(( S / nT ))]}

case "$BASELINE" in
    b1)
        PREFIX="ppo_b1"
        SB3_BASELINE="ppo"
        CKPT_NAME="ppo_${TASK}_final.zip"
        ;;
    b2)
        PREFIX="ppo_b2_icm"
        SB3_BASELINE="icm"
        CKPT_NAME="ppo_icm_${TASK}_final.zip"
        ;;
    b3)
        PREFIX="ppo_b3_ptp"
        SB3_BASELINE="ptp"
        CKPT_NAME="ppo_ptp_${TASK}_final.zip"
        ;;
    b4)
        PREFIX="ppo_b4_rnd"
        SB3_BASELINE="ppo"        # eval treats RND ckpt as plain PPO (rnd is training-only)
        CKPT_NAME="ppo_rnd_${TASK}_final.zip"
        ;;
    b5)
        PREFIX="ppo_b5_recurrent"
        SB3_BASELINE="recurrent"
        CKPT_NAME="ppo_recurrent_${TASK}_final.zip"
        ;;
    b6)
        PREFIX="ppo_b6_recurrent_rnd"
        SB3_BASELINE="recurrent"
        CKPT_NAME="ppo_recurrent_rnd_${TASK}_final.zip"
        ;;
    *)
        echo "[error] unknown BASELINE=$BASELINE (expected b1/b2/b3/b4/b5/b6)" >&2
        exit 1
        ;;
esac

RUN_NAME="${PREFIX}_${TASK}_s${SEED}"
MODEL="${OUTDIR_ROOT}/${RUN_NAME}/${CKPT_NAME}"
[[ -f "$MODEL" ]] || { echo "checkpoint missing: $MODEL" >&2; exit 1; }

EVAL_DIR="${OUTDIR_ROOT}/eval_results"
mkdir -p "$EVAL_DIR" logs/slurm

source /vision/u/jevon/miniconda3/etc/profile.d/conda.sh
conda activate robomme
export PYTHONNOUSERSITE=1
export ROBOMME_OBS_MODE=rgb
export ROBOMME_SIM_BACKEND=physx_cpu

cd /vision/u/jevon/robomme_benchmark
PY=/vision/u/jevon/miniconda3/envs/robomme/bin/python

echo "================================================================"
echo "eval $BASELINE | model=$MODEL task=$TASK seed=$SEED n_eval=$N_EVAL"
echo "max_steps=$MAX_STEPS  node=$SLURM_JOB_NODELIST"
echo "================================================================"

$PY -m train.evaluate_trained \
    --model "$MODEL" \
    --baseline "$SB3_BASELINE" \
    --tasks "$TASK" \
    --n_eval "$N_EVAL" \
    --max_steps "$MAX_STEPS" \
    --outfile "${EVAL_DIR}/${RUN_NAME}.json" 2>&1 | tee "${EVAL_DIR}/${RUN_NAME}.log"
