#!/bin/bash
# 5M-step BC+RL run with the three V4 fixes:
#   - V4 reward (pure-terminal, no proximity hack)
#   - log_std = -1.0 (std=0.37 — lets BC-warm-started mean execute reliably)
#   - bc_aux_coef = 0.5 (auxiliary BC loss keeps policy near expert during PPO)
#
# 3 seeds × 5M steps × ~60 FPS aggregate (n_envs=4 Titan RTX) ≈ 23h each.
# Walltime 24h fits with auto-resume safety margin.
#
# Submit:  bash scripts/cluster/submit_5m_bc_v4.sh

set -euo pipefail

SEEDS=${SEEDS:-"7 42 0"}
TIMESTEPS=${TIMESTEPS:-5000000}
N_ENVS=${N_ENVS:-4}
OUTDIR_ROOT=${OUTDIR_ROOT:-runs/cluster/v4_bc_5m}
BC=${BC:-runs/bc/pickxtimes/bc_policy_recurrent.pt}
BC_DATASET=${BC_DATASET:-runs/bc/pickxtimes/bc_dataset_pickxtimes.pt}
BC_AUX_COEF=${BC_AUX_COEF:-0.5}
INIT_LOG_STD=${INIT_LOG_STD:--1.0}

read -r -a SEED_ARR <<< "$SEEDS"
mkdir -p logs/slurm

for SEED in "${SEED_ARR[@]}"; do
    NAME="ppo_b6_recurrent_rnd_bc_v4_5m_PickXtimes_s${SEED}"
    OUTDIR="${OUTDIR_ROOT}/${NAME}"
    JOB_SCRIPT=$(mktemp /tmp/v4bc5m_s${SEED}.XXXXXX.sbatch)
    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --account=vision --partition=svl --qos=normal
#SBATCH --gres=gpu:titanrtx:1 --cpus-per-task=8 --mem=48G --time=24:00:00
#SBATCH --job-name=rme-v4bc5m-pickx
#SBATCH --output=/vision/u/jevon/robomme_benchmark/logs/slurm/%x-%j.out
set -euo pipefail
source /vision/u/jevon/miniconda3/etc/profile.d/conda.sh
conda activate robomme
export PYTHONNOUSERSITE=1
export ROBOMME_OBS_MODE=rgb
export ROBOMME_SIM_BACKEND=physx_cpu
export ROBOMME_REWARD_VERSION=v4
cd /vision/u/jevon/robomme_benchmark
PY=/vision/u/jevon/miniconda3/envs/robomme/bin/python
mkdir -p "$OUTDIR"
\$PY -m train.train_ppo_recurrent_rnd \\
    --task PickXtimes --timesteps $TIMESTEPS --n_envs $N_ENVS --vec_env subproc \\
    --seed $SEED --outdir "$OUTDIR" \\
    --bc_warmstart "$BC" \\
    --init_log_std $INIT_LOG_STD \\
    --bc_aux_dataset "$BC_DATASET" \\
    --bc_aux_coef $BC_AUX_COEF \\
    --bc_aux_n_transitions 20000 \\
    --wandb --wandb_project robomme-benchmark --wandb_name "$NAME"
EOF
    sbatch "$JOB_SCRIPT"
done
echo "[v4_bc_5m] submission complete"
