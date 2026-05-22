# RoboMME Benchmark — Windows Training Setup Guide

This guide walks through setting up the RoboMME benchmark environment on a native Windows system with an NVIDIA GPU (tested on RTX 4090, driver 591.86, CUDA 13.x) and running the three PPO baselines.

---

## System Requirements

| Component | Minimum | Tested |
|-----------|---------|--------|
| OS | Windows 10 21H2+ | Windows 11 |
| GPU | NVIDIA with Vulkan | RTX 4090 |
| NVIDIA Driver | 527+ | 591.86 |
| CUDA Toolkit | Not required* | 13.1 |
| Python | 3.11 | 3.11 |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB | SSD |

> *CUDA Toolkit is **not** needed for training. PyTorch ships its own CUDA runtime. SAPIEN uses Vulkan for rendering (via the GPU driver), not CUDA.

> **Important:** Do **not** install CUDA-based physics (`physx_cuda`) on Windows — it crashes due to a tensor interop bug in SAPIEN 3.0.2. The training pipeline uses `physx_cpu` for simulation and Vulkan GPU for rendering, which works correctly.

---

## Part 1: One-Time System Setup

### 1.1 Install Git

Download and install from https://git-scm.com/download/win. Accept defaults.

### 1.2 Install uv (Python package manager)

Open PowerShell and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell after installation. Verify with:

```powershell
uv --version
```

### 1.3 Install Visual C++ Build Tools (required by some packages)

Download the **Build Tools for Visual Studio** (not the full IDE) from:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

During install select: **C++ build tools** workload. This is needed to compile certain Python extensions.

---

## Part 2: Clone and Configure the Repository

```powershell
git clone <repo-url> robomme_benchmark
cd robomme_benchmark
```

### 2.1 Verify pyproject.toml has CUDA sources

Open `pyproject.toml` and confirm it contains the following sections (they should already be present):

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
mani-skill = { git = "https://github.com/YinpeiDai/ManiSkill.git", rev = "07be6fbc66350ddca200abfb0a11b692f078f7fd" }
torch = [{ index = "pytorch-cu128" }]
torchvision = [{ index = "pytorch-cu128" }]
```

And the `dependencies` list includes:

```toml
dependencies = [
    "mani-skill",
    "opencv-python>=4.11.0.86",
    "setuptools==80.9.0",
    "torch==2.9.1",
    "torchvision==0.24.1",
    "stable-baselines3>=2.3.0",
    "tensorboard>=2.14.0",
    "tqdm>=4.65.0",
]
```

### 2.2 Create the virtual environment and install packages

```powershell
uv sync
```

This will:
- Create `.venv\` in the project root
- Install PyTorch 2.9.1 with CUDA 12.8 support
- Install ManiSkill from the pinned fork
- Install Stable-Baselines3, TensorBoard, OpenCV, and all other dependencies

The first run downloads ~4 GB and takes several minutes.

Verify CUDA torch was installed (not the CPU-only version):

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.9.1+cu128 True
```

---

## Part 3: Apply Required Patches

These patches fix platform-specific incompatibilities. They only need to be applied once after `uv sync`.

### 3.1 Stub mplib (no Windows wheels available)

`mplib` is a Linux-only motion planning library. Create a stub so imports don't crash:

```powershell
New-Item -ItemType Directory -Force ".venv\Lib\site-packages\mplib"
Set-Content ".venv\Lib\site-packages\mplib\__init__.py" @"
"""
Stub mplib for Windows — mplib has no Windows wheels.
Motion planning features are unavailable; all benchmark tasks still work.
"""

class Planner:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "mplib is not available on Windows. "
            "Motion planning features require Linux. "
            "Benchmark evaluation without the FK planner still works."
        )
"@
```

### 3.2 Patch the ManiSkill Vulkan device parser

The default parser breaks on PCI bus address strings like `pci:0a:00.0` (they contain colons). Replace the function:

Open the file:
```
.venv\Lib\site-packages\mani_skill\envs\utils\system\backend.py
```

Find the function `parse_backend_device_id` and replace its body:

```python
def parse_backend_device_id(backend: str) -> tuple[str, int]:
    if backend.startswith("pci:"):
        return backend, None  # PCI bus address — keep whole string as backend name
    if ":" in backend:
        return backend.split(":")
    return backend, None
```

> **Why:** SAPIEN identifies specific GPUs for Vulkan rendering via their PCI bus address (e.g., `pci:0a:00.0`). The original parser splits on `:` which corrupts multi-segment PCI addresses.

### 3.3 Verify your GPU's PCI bus address

Run this once to see what device string your GPU will use:

```powershell
nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader
# Example output: 00000000:0A:00.0
# The render_backend string will be: pci:0a:00.0  (lowercase, drop leading domain)
```

The `episode_config_resolver.py` detects this automatically at runtime — no manual configuration needed.

---

## Part 4: Verify the Environment

Run a quick end-to-end test to confirm everything works before training:

```powershell
.venv\Scripts\python.exe -c "
from train.envs.rl_env import RobommeRLEnv
import numpy as np

env = RobommeRLEnv('BinFill', seed=0)
obs, _ = env.reset()
print('Reset OK | obs shapes:', {k: v.shape for k, v in obs.items()})

obs, rew, term, trunc, info = env.step(env.action_space.sample())
print(f'Step OK  | reward={rew:.4f}  terminated={term}')

env.close()
print('PASS')
"
```

Expected output (warnings about pynvml/pinocchio are harmless):
```
Reset OK | obs shapes: {'front_rgb': (128, 128, 3), 'wrist_rgb': (128, 128, 3), 'joint_state': (7,), 'eef_state': (6,), 'gripper': (2,)}
Step OK  | reward=0.0000  terminated=False
PASS
```

If this fails, check the troubleshooting section at the end.

---

## Part 5: Running the Three PPO Baselines

All three scripts share common arguments. Run from the project root with the venv Python.

### Common Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `BinFill` | Task name (see list below) |
| `--timesteps` | `1_000_000` | Total environment steps |
| `--n_envs` | `4` | Parallel environments |
| `--n_steps` | `512` | Steps per rollout per env |
| `--batch_size` | `256` | SGD minibatch size |
| `--lr` | `3e-4` | PPO learning rate |
| `--outdir` | varies | Output directory for checkpoints + TB logs |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |

### Baseline 1: Vanilla PPO

No memory, no curiosity. The myopic baseline.

```powershell
.venv\Scripts\python.exe -m train.train_ppo `
  --task BinFill `
  --timesteps 1000000 `
  --n_envs 2 `
  --outdir runs/ppo
```

Output:
- `runs/ppo/ppo_BinFill_final.zip` — saved model
- `runs/ppo/ckpts/` — periodic checkpoints every 50k steps
- `runs/ppo/tb/` — TensorBoard logs

### Baseline 2: PPO + ICM (Curiosity)

Adds an Intrinsic Curiosity Module. Forward and inverse models predict state transitions; the forward error drives exploration.

```powershell
.venv\Scripts\python.exe -m train.train_ppo_icm `
  --task BinFill `
  --timesteps 1000000 `
  --n_envs 2 `
  --eta 0.01 `
  --icm_beta 0.2 `
  --outdir runs/ppo_icm
```

Additional arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--eta` | `0.01` | Intrinsic reward scale |
| `--icm_lr` | `3e-4` | ICM optimizer learning rate |
| `--icm_beta` | `0.2` | Loss weight: `β*L_fwd + (1-β)*L_inv` |

Output:
- `runs/ppo_icm/ppo_icm_BinFill_final.zip` — PPO model
- `runs/ppo_icm/ppo_icm_BinFill_final_icm.pt` — ICM weights

### Baseline 3: PPO + PTP Memory

Adds a 2-layer transformer over a K-step state/action history (K=8 by default). A Past-Token Prediction auxiliary loss trains the transformer to predict past and future action tokens.

Uses `RobommeRLEnvWithMemory` which adds `history_state (K,15)` and `history_action (K,8)` to the observation dict.

```powershell
.venv\Scripts\python.exe -m train.train_ppo_ptp `
  --task BinFill `
  --timesteps 1000000 `
  --n_envs 2 `
  --K 8 `
  --ptp_weight 0.1 `
  --outdir runs/ppo_ptp
```

Additional arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--K` | `8` | History length (steps) |
| `--ptp_lr` | `3e-4` | PTP head optimizer learning rate |
| `--ptp_weight` | `0.1` | PTP loss multiplier |

Output:
- `runs/ppo_ptp/ppo_ptp_BinFill_final.zip` — PPO model
- `runs/ppo_ptp/ppo_ptp_BinFill_final_ptphead.pt` — PTP head weights

### Running All Three Sequentially

Use the provided batch script to run all three back-to-back:

```powershell
# Edit run_all_baselines.bat if needed, then:
.\run_all_baselines.bat
```

---

## Part 6: Monitoring Training

### TensorBoard

```powershell
.venv\Scripts\python.exe -m tensorboard.main --logdir runs/
```

Then open http://localhost:6006 in a browser. Key metrics to watch:

- `train/value_loss` — should decrease
- `train/entropy_loss` — starts ~-11 (random), should rise toward -9 to -8 as policy sharpens
- `train/approx_kl` — keep below 0.05; spikes mean LR is too high
- `icm/intr_reward` — (Baseline 2 only) curiosity reward magnitude
- `ptp/loss` — (Baseline 3 only) auxiliary prediction loss

### Expected Performance (RTX 4090, physx_cpu, n_envs=1)

~25 FPS → 500k steps ≈ 5.5 hours

With `n_envs=2`: ~40 FPS → 500k steps ≈ 3.5 hours

---

## Part 7: Evaluation

After training, evaluate any model on the benchmark test split:

```powershell
# Evaluate Baseline 1 (vanilla PPO or ICM):
.venv\Scripts\python.exe -m train.evaluate_trained `
  --model runs/ppo/ppo_BinFill_final.zip `
  --baseline ppo `
  --tasks BinFill `
  --n_eval 20

# Evaluate Baseline 3 (PTP Memory):
.venv\Scripts\python.exe -m train.evaluate_trained `
  --model runs/ppo_ptp/ppo_ptp_BinFill_final.zip `
  --baseline ptp `
  --K 8 `
  --tasks BinFill `
  --n_eval 20

# Evaluate across all 16 tasks:
.venv\Scripts\python.exe -m train.evaluate_trained `
  --model runs/ppo/ppo_BinFill_final.zip `
  --baseline ppo `
  --tasks all `
  --n_eval 10 `
  --outfile results/ppo_all_tasks.json
```

---

## Available Tasks

The 16 RoboMME benchmark tasks (grouped by cognitive suite):

| Suite | Tasks |
|-------|-------|
| Counting | `PickXtimes`, `StopCube`, `SwingXtimes`, `BinFill` |
| Permanence | `VideoUnmaskSwap`, `VideoUnmask`, `ButtonUnmaskSwap`, `ButtonUnmask` |
| Reference | `VideoRepick`, `VideoPlaceButton`, `VideoPlaceOrder`, `PickHighlight` |
| Imitation | `InsertPeg`, `MoveCube`, `PatternLock`, `RouteStick` |

Start with `BinFill` — it has dense reward and is the most forgiving for initial experiments.

---

## Project Structure

```
robomme_benchmark/
├── src/robomme/              # Benchmark environment package
│   └── env_record_wrapper/
│       ├── episode_config_resolver.py   # Backend auto-detection (Windows/Linux)
│       ├── DemonstrationWrapper.py      # mplib graceful fallback
│       └── ...
├── train/
│   ├── envs/
│   │   └── rl_env.py         # gym.Env wrappers (memoryless + memory-augmented)
│   ├── models/
│   │   ├── encoder.py        # ResNet18 CNN feature extractors (SB3 compatible)
│   │   ├── icm.py            # Intrinsic Curiosity Module
│   │   └── ptp_memory.py     # MemoryTransformer + PTPHead + ptp_loss
│   ├── callbacks/
│   │   ├── icm_callback.py   # SB3 callback: adds ICM reward, trains ICM
│   │   └── ptp_callback.py   # SB3 callback: runs PTP auxiliary loss
│   ├── train_ppo.py          # Baseline 1: Vanilla PPO
│   ├── train_ppo_icm.py      # Baseline 2: PPO + ICM
│   ├── train_ppo_ptp.py      # Baseline 3: PPO + PTP Memory
│   └── evaluate_trained.py   # Evaluation script
├── pyproject.toml            # Dependencies (with CUDA torch sources)
├── uv.lock                   # Pinned lockfile
└── run_all_baselines.bat     # Run all 3 baselines sequentially
```

### Architecture Overview

**Observation space** (all baselines):
- `front_rgb`: `(128, 128, 3)` uint8 — front camera
- `wrist_rgb`: `(128, 128, 3)` uint8 — wrist camera
- `joint_state`: `(7,)` float32 — joint angles
- `eef_state`: `(6,)` float32 — end-effector pose
- `gripper`: `(2,)` float32 — gripper state

**Baseline 3 adds:**
- `history_state`: `(K, 15)` float32 — last K state vectors
- `history_action`: `(K, 8)` float32 — last K actions (zero-padded at episode start)

**Feature extractor outputs:**
- Baselines 1 & 2: 576-d = front(256) + wrist(256) + state(64)
- Baseline 3: 832-d = front(256) + wrist(256) + state(64) + memory(256)

---

## Troubleshooting

### `RuntimeError: vk::createInstanceUnique: ErrorIncompatibleDriver`

SAPIEN cannot find a working Vulkan driver. Causes:
- NVIDIA driver too old (need 527+)
- Running inside WSL2 (which lacks the Linux NVIDIA Vulkan ICD) — must run on native Windows
- Missing the ManiSkill `parse_backend_device_id` patch (see Part 3.2)

### `OSError: libcuda.so / cuda.dll not found`

SAPIEN is trying to enable GPU physics (`physx_cuda`). On Windows this crashes. The `episode_config_resolver.py` should automatically detect Windows and use `physx_cpu`. Confirm the file contains the platform detection block — if not, manually set:

```powershell
$env:ROBOMME_SIM_BACKEND = "physx_cpu"
.venv\Scripts\python.exe -m train.train_ppo ...
```

### `ModuleNotFoundError: No module named 'mplib'`

The mplib stub was not created. Re-run the PowerShell commands in Part 3.1.

### `RuntimeError: The size of tensor a (128) must match tensor b (3)`

The encoder is receiving images in the wrong channel order. Ensure you have the latest `train/models/encoder.py` — it contains the `_to_chw()` helper that handles both `(H,W,C)` and `(C,H,W)` input formats (SB3's `VecTransposeImage` converts to channel-first before passing to the extractor).

### `torch.cuda.is_available()` returns `False`

- Confirm `torch==2.9.1+cu128` is installed (the `+cu128` suffix is required)
- If `uv sync` installed CPU torch, confirm `pyproject.toml` has the `[[tool.uv.index]]` and `[tool.uv.sources]` blocks from Part 2.1, then re-run `uv sync`

### Training is very slow (<10 FPS)

- `n_envs=1` with `physx_cpu` is the bottleneck. Physics simulation is CPU-bound.
- Increase to `--n_envs 2` or `--n_envs 4` for roughly linear speedup up to ~4 envs
- GPU utilization will be low (20-40%) — this is normal; the sim is the bottleneck, not the GPU

### `uv sync` re-installs CPU torch every time

`uv sync` reads the lockfile. If the lockfile was generated without the CUDA index, it pins CPU torch. Fix: ensure `pyproject.toml` has the CUDA index blocks, then run `uv lock` to regenerate, then `uv sync`.

---

## Notes on Platform Differences

| Feature | Windows | Linux |
|---------|---------|-------|
| Physics backend | `physx_cpu` | `physx_cuda` (GPU) |
| Render backend | Vulkan via PCI address | `gpu` (CUDA/Vulkan) |
| mplib | Stub (no motion planning) | Full (FK planner available) |
| Training speed | ~25 FPS (1 env) | ~60+ FPS (1 env, GPU physics) |

The physics backend difference means training is slower on Windows. For large-scale experiments, a Linux GPU server is preferable. The model weights and evaluation results are fully portable between platforms.
