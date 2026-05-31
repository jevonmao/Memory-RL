#!/bin/bash
# Quick smoke check on a real Titan RTX node: verify env imports, instantiate
# every new training stack, run a few hundred steps each. Goal is to catch
# install / driver / SAPIEN issues early before queuing the full sweep.
#
# Submit:  sbatch scripts/cluster/smoke_job.sbatch
#SBATCH --account=vision --partition=svl --qos=normal
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=rme-smoke
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

source /vision/u/jevon/miniconda3/etc/profile.d/conda.sh
conda activate robomme
export PYTHONNOUSERSITE=1
export ROBOMME_OBS_MODE=rgb
# Force CPU physx because sapien 3.0.3 + torch 2.9 trip a __cuda_array_interface__
# typestr parse error inside articulation_joint.drive_target. Scene loading
# uses set_drive_target → must be CPU sim.
export ROBOMME_SIM_BACKEND=physx_cpu
export WANDB_MODE=disabled

cd /vision/u/jevon/robomme_benchmark
PY=/vision/u/jevon/miniconda3/envs/robomme/bin/python

echo "================================================================"
echo "node=$SLURM_JOB_NODELIST gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 | head -3
echo "================================================================"

mkdir -p logs/slurm

echo ""
echo "--- import + offline smoke ---"
$PY -c "import torch, torchvision, sb3_contrib, stable_baselines3, mani_skill, robomme; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('sb3c', sb3_contrib.__version__); print('mani', mani_skill.__version__ if hasattr(mani_skill,'__version__') else '?')"

echo ""
echo "--- offline smoke ---"
$PY -m scripts.smoke_offline

echo ""
echo "--- PickXtimes env smoke (real SAPIEN) ---"
$PY -m scripts.smoke_pickxtimes --device cuda --skip recurrent_rnd recurrent

echo ""
echo "--- PPO+RND learn(2048) on PickXtimes (real env) ---"
$PY -m train.train_ppo_rnd \
    --task PickXtimes --timesteps 2048 --n_envs 2 --n_steps 512 --batch_size 64 \
    --vec_env dummy --seed 7 --outdir runs/smoke_b4_rnd

echo ""
echo "--- RecurrentPPO+RND learn(2048) on PickXtimes (real env) ---"
$PY -m train.train_ppo_recurrent_rnd \
    --task PickXtimes --timesteps 2048 --n_envs 2 --n_steps 512 --batch_size 64 \
    --vec_env dummy --seed 7 --outdir runs/smoke_b6_recurrent_rnd

echo ""
echo "SMOKE JOB COMPLETE"
