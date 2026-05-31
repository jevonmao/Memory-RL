#!/bin/bash
# DAPG-style enhanced PPO run on PickXtimes.
#
# Configuration (corrects all lessons learned this iteration):
#   - V4 reward (terminal-only, no proximity hack)
#   - BC warmstart from DAgger-v2 final policy (better expert prior than raw BC)
#   - log_std = -1.0 (std=0.37 — lets BC-warm-started mean actually execute)
#   - bc_aux_coef = 0.05 (10× LOWER than the V4 5M run that pinned policy to BC)
#   - bc_aux_n_transitions = 30k (3× more demo data for the aux loss)
#   - n_envs = 4, n_steps = 512 (proven config)
#   - 5M steps per seed (2 chained 24h jobs each at ~54 FPS = ~26h)
#
# Submits A and B per seed. B depends on A via afterany so it picks up the
# latest ckpt regardless of whether A finished or hit walltime.
#
# Optionally chain after the DAgger v2 job via DEPEND_ON=<jobid>.

set -euo pipefail

SEEDS=${SEEDS:-"7 42 0"}
TIMESTEPS=${TIMESTEPS:-5000000}
N_ENVS=${N_ENVS:-4}
OUTDIR_ROOT=${OUTDIR_ROOT:-runs/cluster/dapg_5m}
BC=${BC:-runs/dagger_v2/iter8/dagger_policy_recurrent.pt}    # use DAgger-improved BC
BC_DATASET=${BC_DATASET:-runs/bc/pickxtimes/bc_dataset_pickxtimes.pt}
BC_AUX_COEF=${BC_AUX_COEF:-0.05}
INIT_LOG_STD=${INIT_LOG_STD:--1.0}
DEPEND_ON=${DEPEND_ON:-}

read -r -a SEED_ARR <<< "$SEEDS"
mkdir -p logs/slurm

DEP_FLAG=""
if [[ -n "$DEPEND_ON" ]]; then
    DEP_FLAG="--dependency=afterany:${DEPEND_ON}"
fi

for SEED in "${SEED_ARR[@]}"; do
    NAME="ppo_dapg_v4_5m_PickXtimes_s${SEED}"
    OUTDIR="${OUTDIR_ROOT}/${NAME}"
    mkdir -p "$OUTDIR"

    # --- Job A ---
    A_SCRIPT=$(mktemp /tmp/dapg_a_s${SEED}.XXXXXX.sbatch)
    cat > "$A_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --account=vision --partition=svl --qos=normal
#SBATCH --gres=gpu:titanrtx:1 --cpus-per-task=8 --mem=48G --time=24:00:00
#SBATCH --job-name=rme-dapg-A
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
\$PY -m train.train_ppo_recurrent_rnd \\
    --task PickXtimes --timesteps $TIMESTEPS --n_envs $N_ENVS --vec_env subproc \\
    --seed $SEED --outdir "$OUTDIR" \\
    --bc_warmstart "$BC" \\
    --init_log_std $INIT_LOG_STD \\
    --bc_aux_dataset "$BC_DATASET" \\
    --bc_aux_coef $BC_AUX_COEF \\
    --bc_aux_n_transitions 30000 \\
    --wandb --wandb_project robomme-benchmark --wandb_name "$NAME"
EOF
    JOB_A=$(sbatch --parsable $DEP_FLAG "$A_SCRIPT")
    echo "submitted A: $JOB_A  (seed=$SEED, deps='${DEP_FLAG}')"

    # --- Job B: resume from latest ckpt after A finishes ---
    B_SCRIPT=$(mktemp /tmp/dapg_b_s${SEED}.XXXXXX.sbatch)
    cat > "$B_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --account=vision --partition=svl --qos=normal
#SBATCH --gres=gpu:titanrtx:1 --cpus-per-task=8 --mem=48G --time=24:00:00
#SBATCH --job-name=rme-dapg-B
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
\$PY -m train.train_ppo_recurrent_rnd \\
    --task PickXtimes --timesteps $TIMESTEPS --n_envs $N_ENVS --vec_env subproc \\
    --seed $SEED --outdir "$OUTDIR" \\
    --resume latest \\
    --bc_aux_dataset "$BC_DATASET" \\
    --bc_aux_coef $BC_AUX_COEF \\
    --bc_aux_n_transitions 30000 \\
    --wandb --wandb_project robomme-benchmark --wandb_name "${NAME}-resume"
EOF
    JOB_B=$(sbatch --parsable --dependency=afterany:${JOB_A} "$B_SCRIPT")
    echo "submitted B: $JOB_B  (deps afterany:$JOB_A, seed=$SEED)"
done

echo ""
echo "[dapg-5m] submission complete"
