#!/bin/bash
# V3 + BC + LOW entropy coefficient (0.001 vs default 0.01).
# Hypothesis: V3 reward has a healthy proximity gradient and strong task
# bonuses, but std=0.97 keeps the policy too noisy to lock onto BC's
# expert-near behavior. Lower ent_coef should let std decay faster and
# the policy commit to the BC mean, then PPO refines.

set -euo pipefail

SEEDS=${SEEDS:-"7 42 0"}
TIMESTEPS=${TIMESTEPS:-1500000}
N_ENVS=${N_ENVS:-4}
ENT_COEF=${ENT_COEF:-0.001}
OUTDIR_ROOT=${OUTDIR_ROOT:-runs/cluster/v3_bc_lowent}
BC=${BC:-runs/bc/pickxtimes/bc_policy_recurrent.pt}

read -r -a SEED_ARR <<< "$SEEDS"
mkdir -p logs/slurm

for SEED in "${SEED_ARR[@]}"; do
    NAME="ppo_b6_recurrent_rnd_bc_v3lowent_PickXtimes_s${SEED}"
    OUTDIR="${OUTDIR_ROOT}/${NAME}"
    JOB_SCRIPT=$(mktemp /tmp/b6bcv3le_s${SEED}.XXXXXX.sbatch)
    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --account=vision --partition=svl --qos=normal
#SBATCH --gres=gpu:titanrtx:1 --cpus-per-task=8 --mem=32G --time=24:00:00
#SBATCH --job-name=rme-b6v3le-pickx
#SBATCH --output=/vision/u/jevon/robomme_benchmark/logs/slurm/%x-%j.out
set -euo pipefail
source /vision/u/jevon/miniconda3/etc/profile.d/conda.sh
conda activate robomme
export PYTHONNOUSERSITE=1
export ROBOMME_OBS_MODE=rgb
export ROBOMME_SIM_BACKEND=physx_cpu
export ROBOMME_REWARD_VERSION=v3
cd /vision/u/jevon/robomme_benchmark
PY=/vision/u/jevon/miniconda3/envs/robomme/bin/python
mkdir -p "$OUTDIR"
\$PY -m train.train_ppo_recurrent_rnd \\
    --task PickXtimes --timesteps $TIMESTEPS --n_envs $N_ENVS --vec_env subproc \\
    --seed $SEED --outdir "$OUTDIR" \\
    --bc_warmstart "$BC" \\
    --ent_coef $ENT_COEF \\
    --wandb --wandb_project robomme-benchmark --wandb_name "$NAME"
EOF
    sbatch "$JOB_SCRIPT"
done
echo "[b6_bc_v3_lowent] submission complete"
