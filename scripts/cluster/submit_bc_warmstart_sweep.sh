#!/bin/bash
# Submit BC-warmstarted PickXtimes baselines.
#
# Depends on the BC pretrain job having produced:
#   runs/bc/pickxtimes/bc_policy.pt           (for b4 PPO+RND)
#   runs/bc/pickxtimes/bc_policy_recurrent.pt (for b6 RecurrentPPO+RND)
#
# Submits 6 jobs (b4_bc + b6_bc × 3 seeds). Same hyperparams as
# the non-BC sweep but with --bc_warmstart pointing at the policy file.
# Use BC_JOB_ID=<jobid> to add a dependency on the BC pretrain.
#
# Usage:
#     bash scripts/cluster/submit_bc_warmstart_sweep.sh
#     BC_JOB_ID=15544696 bash scripts/cluster/submit_bc_warmstart_sweep.sh

set -euo pipefail

SEEDS=${SEEDS:-"7 42 0"}
TIMESTEPS=${TIMESTEPS:-1500000}
N_ENVS=${N_ENVS:-4}
OUTDIR_ROOT=${OUTDIR_ROOT:-runs/cluster}
BC_DIR=${BC_DIR:-runs/bc/pickxtimes}

read -r -a SEED_ARR <<< "$SEEDS"
N=${#SEED_ARR[@]}

DEP_FLAG=""
if [[ -n "${BC_JOB_ID:-}" ]]; then
    DEP_FLAG="--dependency=afterok:${BC_JOB_ID}"
fi

mkdir -p logs/slurm

for SEED in "${SEED_ARR[@]}"; do
    for VARIANT in b4 b6; do
        if [[ "$VARIANT" == "b4" ]]; then
            MODULE="train.train_ppo_rnd"
            BC="${BC_DIR}/bc_policy.pt"
            NAME="ppo_b4_rnd_bc_PickXtimes_s${SEED}"
        else
            MODULE="train.train_ppo_recurrent_rnd"
            BC="${BC_DIR}/bc_policy_recurrent.pt"
            NAME="ppo_b6_recurrent_rnd_bc_PickXtimes_s${SEED}"
        fi
        OUTDIR="${OUTDIR_ROOT}/${NAME}"

        # train_ppo_rnd doesn't support --bc_warmstart yet (recurrent does).
        # For b4 we just submit without warmstart for now — the recurrent
        # variant is the headline + has BC; b4 stays as a non-BC reference.
        BC_FLAG=""
        if [[ "$VARIANT" == "b6" ]]; then
            BC_FLAG="--bc_warmstart $BC"
        fi

        JOB_SCRIPT=$(mktemp /tmp/bc_warm_${VARIANT}_s${SEED}.XXXXXX.sbatch)
        cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --account=vision --partition=svl --qos=normal
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=rme-${VARIANT}bc-pickx
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
source /vision/u/jevon/miniconda3/etc/profile.d/conda.sh
conda activate robomme
export PYTHONNOUSERSITE=1
export ROBOMME_OBS_MODE=rgb
export ROBOMME_SIM_BACKEND=physx_cpu

cd /vision/u/jevon/robomme_benchmark
PY=/vision/u/jevon/miniconda3/envs/robomme/bin/python

mkdir -p "$OUTDIR" logs/slurm

\$PY -m "$MODULE" \\
    --task PickXtimes --timesteps $TIMESTEPS --n_envs $N_ENVS --vec_env subproc \\
    --seed $SEED --outdir "$OUTDIR" \\
    $BC_FLAG \\
    --wandb --wandb_project robomme-benchmark --wandb_name "$NAME"
EOF
        sbatch $DEP_FLAG "$JOB_SCRIPT" | tee -a /tmp/bc_warmstart_submit.log
    done
done

echo ""
echo "[bc-warmstart] submission complete"
