#!/bin/bash
# Eval a BC-only policy on PickXtimes.
# Submit:
#   BC=runs/bc/pickxtimes/bc_policy_recurrent.pt RECURRENT=1 \
#     sbatch scripts/cluster/eval_bc_only_job.sbatch
#SBATCH --account=vision --partition=svl --qos=normal
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --job-name=rme-bc-eval
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

source /vision/u/jevon/miniconda3/etc/profile.d/conda.sh
conda activate robomme
export PYTHONNOUSERSITE=1
export ROBOMME_OBS_MODE=rgb
export ROBOMME_SIM_BACKEND=physx_cpu

cd /vision/u/jevon/robomme_benchmark
PY="/vision/u/jevon/miniconda3/envs/robomme/bin/python -u"

BC=${BC:?BC env var required}
RECURRENT=${RECURRENT:-0}
N_EVAL=${N_EVAL:-20}
OUTFILE=${OUTFILE:-${BC%.pt}.eval.json}

REC_FLAG=""
[[ "$RECURRENT" == "1" ]] && REC_FLAG="--recurrent"

$PY -m scripts.eval_bc_only --bc "$BC" $REC_FLAG --n_eval "$N_EVAL" --outfile "$OUTFILE"
