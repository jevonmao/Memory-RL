#!/bin/bash
# Chain two 24h jobs per seed so 5M-step runs complete despite slurm walltime.
# Job B depends on Job A via --dependency=afterany so it resumes from latest
# ckpt regardless of whether A finished or hit walltime.

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
    mkdir -p "$OUTDIR"

    # --- Job A: fresh start ---
    A_SCRIPT=$(mktemp /tmp/v4chain_a_s${SEED}.XXXXXX.sbatch)
    cat > "$A_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --account=vision --partition=svl --qos=normal
#SBATCH --gres=gpu:titanrtx:1 --cpus-per-task=8 --mem=48G --time=24:00:00
#SBATCH --job-name=rme-v4bc5m-A
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
    --bc_aux_n_transitions 20000 \\
    --wandb --wandb_project robomme-benchmark --wandb_name "$NAME"
EOF
    JOB_A_ID=$(sbatch --parsable "$A_SCRIPT")
    echo "submitted A: $JOB_A_ID (seed=$SEED)"

    # --- Job B: resume after A (afterany so we pick up even if A walltimed) ---
    B_SCRIPT=$(mktemp /tmp/v4chain_b_s${SEED}.XXXXXX.sbatch)
    cat > "$B_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --account=vision --partition=svl --qos=normal
#SBATCH --gres=gpu:titanrtx:1 --cpus-per-task=8 --mem=48G --time=24:00:00
#SBATCH --job-name=rme-v4bc5m-B
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
    --bc_aux_n_transitions 20000 \\
    --wandb --wandb_project robomme-benchmark --wandb_name "${NAME}-resume"
EOF
    JOB_B_ID=$(sbatch --parsable --dependency=afterany:${JOB_A_ID} "$B_SCRIPT")
    echo "submitted B: $JOB_B_ID (deps afterany:$JOB_A_ID, seed=$SEED)"
done

echo "[v4_bc_5m chain] submission complete"
