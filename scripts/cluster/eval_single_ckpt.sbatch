#!/bin/bash
# Eval a single PPO checkpoint on PickXtimes (intermediate or final).
# Usage:
#   MODEL=/path/to/ppo_rnd_PickXtimes_100000_steps.zip BASELINE=ppo \
#     sbatch scripts/cluster/eval_single_ckpt.sbatch
#
#   For RecurrentPPO ckpts pass BASELINE=recurrent.
#SBATCH --account=vision --partition=svl --qos=normal
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --job-name=rme-eval-1
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

source /vision/u/jevon/miniconda3/etc/profile.d/conda.sh
conda activate robomme
export PYTHONNOUSERSITE=1
export ROBOMME_OBS_MODE=rgb
export ROBOMME_SIM_BACKEND=physx_cpu

cd /vision/u/jevon/robomme_benchmark
PY="/vision/u/jevon/miniconda3/envs/robomme/bin/python -u"

MODEL=${MODEL:?MODEL env var required}
BASELINE=${BASELINE:-ppo}
TASK=${TASK:-PickXtimes}
N_EVAL=${N_EVAL:-10}
MAX_STEPS=${MAX_STEPS:-1500}
OUTFILE=${OUTFILE:-${MODEL%.zip}.eval.json}

mkdir -p logs/slurm "$(dirname "$OUTFILE")"

echo "================================================================"
echo "single-ckpt eval | model=$MODEL baseline=$BASELINE task=$TASK n_eval=$N_EVAL"
echo "node=$SLURM_JOB_NODELIST gpu=$CUDA_VISIBLE_DEVICES outfile=$OUTFILE"
echo "================================================================"

$PY -m train.evaluate_trained \
    --model "$MODEL" --baseline "$BASELINE" --tasks "$TASK" \
    --n_eval "$N_EVAL" --max_steps "$MAX_STEPS" \
    --outfile "$OUTFILE"

echo ""
echo "RESULTS:"
python -c "import json; d=json.load(open('$OUTFILE')); print(json.dumps(d, indent=2))"
