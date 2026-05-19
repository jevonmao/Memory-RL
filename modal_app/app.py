"""Modal entrypoint for RoboMME training/eval.

Why this exists
---------------
SAPIEN/ManiSkill require a Vulkan ICD that WSL2 doesn't expose. Modal's
containers run on real Linux with proper NVIDIA Vulkan drivers, so we ship
training there.

Usage (from the host, after `modal token new` and `modal secret create wandb ...`):

    # Smoke test — does RoboMME import + step on Modal?
    modal run modal_app/app.py::smoke --task BinFill

    # Random-policy baseline eval
    modal run modal_app/app.py::random_baseline --task BinFill --episodes 20

    # Single PPO run
    modal run modal_app/app.py::train_ppo --task BinFill --seed 0 --steps 100000

    # Full baseline sweep (4 tasks x 3 seeds)
    modal run modal_app/app.py::sweep

Layout
------
We mirror RoboMME's Dockerfile recipe (CUDA 12.8 base + libvulkan1 + the
pinned mani-skill rev) so SAPIEN renders with Vulkan. The repo and the
robomme checkout are added as Mounts so we can iterate on training code
locally without rebuilding the image.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Image: matches robomme_benchmark/Dockerfile
# ---------------------------------------------------------------------------
ROBOMME_REV = "07be6fbc66350ddca200abfb0a11b692f078f7fd"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04",
        add_python="3.11",
    )
    .apt_install(
        "build-essential",
        "clang",  # toppra (mani-skill dep) hardcodes clang for its Cython ext.
        "ca-certificates",
        "curl",
        "ffmpeg",
        "git",
        "libegl1",
        "libgl1",
        "libglib2.0-0",
        "libvulkan1",
        "vulkan-tools",
        "libxext6",
        "libxrender1",
    )
    .pip_install(
        # Pinned per RoboMME pyproject — these versions are picked to be
        # mutually compatible with mani-skill 3.0.0b21.
        "torch==2.9.1",
        "torchvision==0.24.1",
        f"mani-skill @ git+https://github.com/YinpeiDai/ManiSkill.git@{ROBOMME_REV}",
        "h5py",
        "opencv-python>=4.11.0.86",
        "stable-baselines3==2.8.0",
        "tensorboard",
        "pyyaml",
        "wandb>=0.18",
    )
    .env(
        {
            "NVIDIA_DRIVER_CAPABILITIES": "compute,graphics,utility,video",
            "SAPIEN_RENDER_DEVICE": "cuda",
            "PYTHONUNBUFFERED": "1",
            "WANDB_PROJECT": "memory-rl",
        }
    )
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOMME_SRC = Path("/home/jevon/projects/robomme_benchmark")

# Modal 1.x: bake source dirs into the image (rebuild-free across runs because
# Modal hashes contents — only changed files trigger a layer rebuild).
image = image.add_local_dir(str(REPO_ROOT), remote_path="/workspace")
image = image.add_local_dir(str(ROBOMME_SRC), remote_path="/robomme_src")

app = modal.App("memory-rl")

GPU = "A10G"  # cheapest GPU on Modal that comfortably runs SAPIEN; bump to A100 for big sweeps.
WANDB_SECRET = modal.Secret.from_name("wandb")

COMMON_ENV = {
    "ROBOMME_PATH": "/robomme_src",
    "PYTHONPATH": "/workspace:/robomme_src/src",
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
@app.function(image=image, gpu=GPU, timeout=600)
def smoke(task: str = "BinFill") -> str:
    """Confirm RoboMME imports + steps on Modal."""
    os.environ.update(COMMON_ENV)
    sys.path.insert(0, "/workspace")
    from scripts.inspect_env import main as inspect_main  # type: ignore

    sys.argv = ["inspect_env", "--task", task]
    rc = inspect_main()
    return f"inspect_env exit={rc}"


@app.function(image=image, gpu=GPU, timeout=3 * 60 * 60, secrets=[WANDB_SECRET])
def train_ppo(
    task: str = "BinFill",
    seed: int = 0,
    steps: int = 100_000,
    tag: str | None = None,
):
    """Run a single PPO baseline on Modal, logging to W&B."""
    os.environ.update(COMMON_ENV)
    sys.path.insert(0, "/workspace")

    import wandb  # noqa: F401 — ensures secret is wired before training imports

    from training.train_ppo import main as train_main  # type: ignore

    argv = [
        "train_ppo",
        "--config", "/workspace/configs/ppo.yaml",
        "--task", task,
        "--seed", str(seed),
        "--total_steps", str(steps),
        "--output_dir", "/workspace/logs",
    ]
    if tag:
        argv += ["--tag", tag]
    sys.argv = argv
    return train_main()


@app.function(image=image, gpu=GPU, timeout=60 * 60)
def random_baseline(task: str = "BinFill", episodes: int = 20, seed: int = 0):
    """Random-policy eval baseline. Useful for sanity-checking the metrics
    pipeline on RoboMME before any learned policy gets involved."""
    os.environ.update(COMMON_ENV)
    sys.path.insert(0, "/workspace")
    from env.robomme_env import make_env  # type: ignore
    from metrics.evaluation import EpisodeRecord, summarize  # type: ignore

    env = make_env(task, seed=seed)
    records = []
    for ep_i in range(episodes):
        obs, info = env.reset(seed=seed + ep_i)
        observations, rewards, infos = [obs], [], []
        term = trunc = False
        while not (term or trunc):
            action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            observations.append(obs)
            rewards.append(float(r))
            infos.append(info)
        records.append(EpisodeRecord(observations, rewards, term, trunc, infos))
    return summarize(records)


@app.local_entrypoint()
def sweep(steps: int = 100_000):
    """4 tasks x 3 seeds. Runs in parallel on Modal."""
    tasks = ["BinFill", "PickXtimes", "StopCube", "VideoUnmask"]
    seeds = [0, 1, 2]
    jobs = [(t, s) for t in tasks for s in seeds]
    print(f"[sweep] launching {len(jobs)} runs on Modal ({GPU})")
    results = list(
        train_ppo.starmap([(t, s, steps, "baseline") for t, s in jobs])
    )
    for (t, s), r in zip(jobs, results):
        print(f"[sweep] {t} seed={s} -> {r}")
