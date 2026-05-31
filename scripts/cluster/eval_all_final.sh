#!/bin/bash
# When all training jobs finish, evaluate every final checkpoint and dump
# results into runs/cluster/eval_results/. Run from the cluster.

set -euo pipefail

declare -A RUNS
RUNS["ppo_b1_PickXtimes_s7"]="ppo|ppo_PickXtimes_final.zip|runs/cluster"
RUNS["ppo_b1_PickXtimes_s42"]="ppo|ppo_PickXtimes_final.zip|runs/cluster"
RUNS["ppo_b1_PickXtimes_s0"]="ppo|ppo_PickXtimes_final.zip|runs/cluster"
RUNS["ppo_b4_rnd_PickXtimes_s7"]="ppo|ppo_rnd_PickXtimes_final.zip|runs/cluster"
RUNS["ppo_b4_rnd_PickXtimes_s42"]="ppo|ppo_rnd_PickXtimes_final.zip|runs/cluster"
RUNS["ppo_b4_rnd_PickXtimes_s0"]="ppo|ppo_rnd_PickXtimes_final.zip|runs/cluster"
RUNS["ppo_b5_recurrent_PickXtimes_s7"]="recurrent|ppo_recurrent_PickXtimes_final.zip|runs/cluster"
RUNS["ppo_b5_recurrent_PickXtimes_s42"]="recurrent|ppo_recurrent_PickXtimes_final.zip|runs/cluster"
RUNS["ppo_b5_recurrent_PickXtimes_s0"]="recurrent|ppo_recurrent_PickXtimes_final.zip|runs/cluster"
RUNS["ppo_b6_recurrent_rnd_PickXtimes_s7"]="recurrent|ppo_recurrent_rnd_PickXtimes_final.zip|runs/cluster"
RUNS["ppo_b6_recurrent_rnd_PickXtimes_s42"]="recurrent|ppo_recurrent_rnd_PickXtimes_final.zip|runs/cluster"
RUNS["ppo_b6_recurrent_rnd_PickXtimes_s0"]="recurrent|ppo_recurrent_rnd_PickXtimes_final.zip|runs/cluster"
# V2 reward
RUNS["v2/ppo_b1_PickXtimes_s7"]="ppo|ppo_PickXtimes_final.zip|runs/cluster"
RUNS["v2/ppo_b1_PickXtimes_s42"]="ppo|ppo_PickXtimes_final.zip|runs/cluster"
RUNS["v2/ppo_b1_PickXtimes_s0"]="ppo|ppo_PickXtimes_final.zip|runs/cluster"
RUNS["v2/ppo_b4_rnd_PickXtimes_s7"]="ppo|ppo_rnd_PickXtimes_final.zip|runs/cluster"
RUNS["v2/ppo_b4_rnd_PickXtimes_s42"]="ppo|ppo_rnd_PickXtimes_final.zip|runs/cluster"
RUNS["v2/ppo_b4_rnd_PickXtimes_s0"]="ppo|ppo_rnd_PickXtimes_final.zip|runs/cluster"
RUNS["v2/ppo_b5_recurrent_PickXtimes_s7"]="recurrent|ppo_recurrent_PickXtimes_final.zip|runs/cluster"
RUNS["v2/ppo_b5_recurrent_PickXtimes_s42"]="recurrent|ppo_recurrent_PickXtimes_final.zip|runs/cluster"
RUNS["v2/ppo_b5_recurrent_PickXtimes_s0"]="recurrent|ppo_recurrent_PickXtimes_final.zip|runs/cluster"
RUNS["v2/ppo_b6_recurrent_rnd_PickXtimes_s7"]="recurrent|ppo_recurrent_rnd_PickXtimes_final.zip|runs/cluster"
RUNS["v2/ppo_b6_recurrent_rnd_PickXtimes_s42"]="recurrent|ppo_recurrent_rnd_PickXtimes_final.zip|runs/cluster"
RUNS["v2/ppo_b6_recurrent_rnd_PickXtimes_s0"]="recurrent|ppo_recurrent_rnd_PickXtimes_final.zip|runs/cluster"

EVAL_DIR=runs/cluster/eval_results
mkdir -p "$EVAL_DIR" logs/slurm

for name in "${!RUNS[@]}"; do
    IFS='|' read -r baseline ckpt_name root <<< "${RUNS[$name]}"
    model="${root}/${name}/${ckpt_name}"
    if [[ ! -f "$model" ]]; then
        echo "[skip] $name — $model not found"
        continue
    fi
    outfile="${EVAL_DIR}/$(basename $name).json"
    echo "[submit] $name → $outfile"
    MODEL="$model" BASELINE="$baseline" N_EVAL=20 OUTFILE="$outfile" \
        sbatch --job-name="rme-eval-${name//\//-}" \
        scripts/cluster/eval_single_ckpt.sbatch
done
