#!/bin/bash
# Submit 3 seeds of b6+BC with V3 reward.
# V3 = V1 proximity gradient + V2-magnitude subgoal/terminal bonuses.

set -euo pipefail

SEEDS=${SEEDS:-"7 42 0"}
TIMESTEPS=${TIMESTEPS:-1500000}
N_ENVS=${N_ENVS:-4}
OUTDIR_ROOT=${OUTDIR_ROOT:-runs/cluster/v3_bc}
BC=${BC:-runs/bc/pickxtimes/bc_policy_recurrent.pt}

read -r -a SEED_ARR <<< "$SEEDS"
mkdir -p logs/slurm

for SEED in "${SEED_ARR[@]}"; do
    NAME="ppo_b6_recurrent_rnd_bc_v3_PickXtimes_s${SEED}"
    OUTDIR="${OUTDIR_ROOT}/${NAME}"

    JOB_SCRIPT=$(mktemp /tmp/b6bcv3_s${SEED}.XXXXXX.sbatch)
    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --account=vision --partition=svl --qos=normal
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=rme-b6bcv3-pickx
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
    --wandb --wandb_project robomme-benchmark --wandb_name "$NAME"
EOF
    sbatch "$JOB_SCRIPT"
done
echo "[b6_bc_v3] submission complete"
