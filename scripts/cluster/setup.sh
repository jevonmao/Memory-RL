#!/bin/bash
# One-shot cluster setup for robomme_benchmark on SVL.
# Run ONCE on a login node after cloning / rsyncing the repo.
# Idempotent — safe to re-run; will skip steps that are already done.
#
# Usage:
#     ssh jevon@scdt.stanford.edu
#     cd /vision/u/jevon/robomme_benchmark
#     bash scripts/cluster/setup.sh
#
# Optional env overrides:
#     ENV_NAME=robomme            # conda env name
#     PYTHON_VERSION=3.11
#     CONDA_ROOT=/vision/u/jevon/miniconda3
#     CUDA_TAG=cu128              # PyTorch wheel index suffix

set -euo pipefail

ENV_NAME=${ENV_NAME:-robomme}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
CONDA_ROOT=${CONDA_ROOT:-/vision/u/jevon/miniconda3}
CUDA_TAG=${CUDA_TAG:-cu128}
REPO_ROOT=${REPO_ROOT:-$(pwd)}

log() { printf "\n[setup] %s\n" "$*"; }

# ----------------------------------------------------------------------------
# 0. Sanity — we are on a cluster login node, not localhost.
# ----------------------------------------------------------------------------
if [[ ! -d "$CONDA_ROOT" ]]; then
    echo "[setup] FATAL: $CONDA_ROOT not found." >&2
    echo "         This script is meant for the SVL cluster. Aborting." >&2
    exit 1
fi
if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
    echo "[setup] FATAL: cd into the robomme_benchmark repo first." >&2
    echo "         (expected $REPO_ROOT/pyproject.toml)" >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# 1. Conda env
# ----------------------------------------------------------------------------
source "$CONDA_ROOT/etc/profile.d/conda.sh"

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    log "conda env '$ENV_NAME' already exists — skipping create"
else
    log "creating conda env '$ENV_NAME' (python ${PYTHON_VERSION})"
    conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
fi

conda activate "$ENV_NAME"
export PYTHONNOUSERSITE=1

# ----------------------------------------------------------------------------
# 2. PyTorch (matches pyproject.toml: torch==2.9.1, torchvision==0.24.1)
# ----------------------------------------------------------------------------
if python -c "import torch; assert torch.__version__.startswith('2.9.1')" 2>/dev/null; then
    log "torch 2.9.1 already installed — skipping"
else
    log "installing torch 2.9.1 + torchvision 0.24.1 (${CUDA_TAG})"
    pip install --no-cache-dir \
        torch==2.9.1 torchvision==0.24.1 \
        --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

# ----------------------------------------------------------------------------
# 3. ManiSkill (pinned to the RoboMME fork) + benchmark in editable mode
# ----------------------------------------------------------------------------
if python -c "import mani_skill" 2>/dev/null; then
    log "mani_skill already importable — skipping pinned install"
else
    log "installing pinned mani-skill fork (from pyproject.toml [tool.uv.sources])"
    pip install --no-cache-dir \
        "git+https://github.com/YinpeiDai/ManiSkill.git@07be6fbc66350ddca200abfb0a11b692f078f7fd"
fi

log "installing robomme_benchmark (editable)"
pip install --no-cache-dir -e "$REPO_ROOT" \
    --no-deps  # deps come from the next step so we don't reinstall torch

# ----------------------------------------------------------------------------
# 4. Remaining python deps
# ----------------------------------------------------------------------------
log "installing remaining deps (sb3, wandb, opencv, etc)"
pip install --no-cache-dir \
    "stable-baselines3>=2.3.0" \
    "tensorboard>=2.14.0" \
    "tqdm>=4.65.0" \
    "wandb>=0.27.0" \
    "opencv-python>=4.11.0.86" \
    "imageio>=2.34.0" \
    "msgpack-numpy>=0.4.8" \
    "h5py>=3.10.0" \
    "setuptools==80.9.0"

# ----------------------------------------------------------------------------
# 5. Smoke test (lightweight — no GPU env build)
# ----------------------------------------------------------------------------
log "smoke-testing imports"
python - <<'PY'
import torch
import gymnasium
import stable_baselines3
import mani_skill
import robomme
from robomme.env_record_wrapper import BenchmarkEnvBuilder
from train.envs.rl_env import RobommeRLEnv
from train.models.encoder import RobommeCNNExtractor, RobommeCNNExtractorLegacy
from train.algos.ppo_with_icm import PPOWithICM
from challenge_interface.policy import SB3Policy
print(f"  torch={torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  sb3  ={stable_baselines3.__version__}")
print(f"  mani ={mani_skill.__version__}")
print(f"  benchmark builder loadable; episodes(BinFill train) = "
      f"{BenchmarkEnvBuilder('BinFill', 'train', 'joint_angle', max_steps=1500).get_episode_num()}")
PY

# ----------------------------------------------------------------------------
# 6. Wandb (optional — interactive)
# ----------------------------------------------------------------------------
if [[ -f "$HOME/.netrc" ]] && grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    log "wandb credentials already in ~/.netrc — skipping login"
else
    log "wandb login (paste your API key from https://wandb.ai/authorize)"
    wandb login || echo "[setup] wandb login skipped (you can run 'wandb login' later)"
fi

# ----------------------------------------------------------------------------
# 7. Dirs
# ----------------------------------------------------------------------------
mkdir -p "$REPO_ROOT/logs/slurm" "$REPO_ROOT/runs/cluster" "$REPO_ROOT/runs/cluster/eval_results"

cat <<EOF

[setup] === DONE ===

Conda env:     $ENV_NAME
Python:        $(which python)
Repo:          $REPO_ROOT
Logs:          $REPO_ROOT/logs/slurm/
Runs:          $REPO_ROOT/runs/cluster/

Next steps:
  # submit the default sweep (4 tasks × 2 baselines × 3 seeds = 24 jobs):
  bash scripts/cluster/submit_sweep.sh

  # or a single baseline / single task:
  TASKS=BinFill sbatch scripts/cluster/train_array.sbatch BASELINE=b1
  TASKS=BinFill sbatch scripts/cluster/train_array.sbatch BASELINE=b2

  # full 16-task sweep (96 jobs, 20 concurrent):
  FULL_SWEEP=1 bash scripts/cluster/submit_sweep.sh

EOF
