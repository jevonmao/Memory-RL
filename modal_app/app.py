"""Modal entrypoint for RoboMME training/eval.

Why this exists
---------------
SAPIEN/ManiSkill require a Vulkan ICD that WSL2 doesn't expose. Modal's
containers run on real Linux with proper NVIDIA Vulkan drivers, so we ship
training there.

BC → PPO workflow (primary research pipeline)
---------------------------------------------
Prerequisites: `modal token new`, data volume already contains
    memory-rl-data/robomme_data/record_dataset_BinFill.h5

Step 0 — verify the H5 dataset layout (run once):
    modal run modal_app/app.py::inspect_h5

Step 1 — train BC on Modal (no SAPIEN needed; GPU used for the MLP only):
    modal run modal_app/app.py::train_bc
    # with overrides:
    modal run modal_app/app.py::train_bc --task BinFill --seed 0 \
        --bc-epochs 100 --vf-pretrain-epochs 20

    Outputs saved to the memory-rl-data volume:
        models/BinFill_bc_seed0/checkpoints/bc_best.zip   (best NLL during training)
        models/BinFill_bc_seed0/checkpoints/bc_final.zip  (after value-head pretraining)
        models/BinFill_bc_seed0/bc_train_log.json

Step 2 — download the trained BC model to the VM:
    modal volume get memory-rl-data \
        models/BinFill_bc_seed0/checkpoints/bc_final.zip \
        ./bc_BinFill_seed0.zip

Step 3 — warm-start PPO from the BC checkpoint on the VM (needs SAPIEN):
    ROBOMME_PATH=[path to robomme_benchmark] \
    python training/train_ppo.py \
        --config configs/ppo.yaml \
        --bc_checkpoint bc_BinFill_seed0.zip \
        --task BinFill --seed 0

    The BC actor + value-head weights are loaded into the fresh PPO model
    before model.learn() is called. PPO then fine-tunes with curiosity/memory
    rewards on top of the BC-initialised policy.

Other Modal commands
--------------------
    # Smoke test — does RoboMME import + step on Modal?
    modal run modal_app/app.py::smoke --task BinFill

    # Random-policy baseline eval
    modal run modal_app/app.py::random_baseline --task BinFill --episodes 20

    # Single PPO run (no BC warm-start)
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
ROBOMME_SRC = Path("/Users/sunfl/Documents/My/Stanford/cs224R/group_project/robomme_benchmark")

# Modal 1.x: bake source dirs into the image (rebuild-free across runs because
# Modal hashes contents — only changed files trigger a layer rebuild).
image = image.add_local_dir(str(REPO_ROOT), remote_path="/workspace")
image = image.add_local_dir(str(ROBOMME_SRC), remote_path="/robomme_src")

app = modal.App("memory-rl")

GPU = "A10G"  # cheapest GPU on Modal that comfortably runs SAPIEN; bump to A100 for big sweeps.
WANDB_SECRET = modal.Secret.from_name("wandb")

# Persistent volume that holds the RoboMME HDF5 datasets and saved models.
# Data layout inside the volume:
#   robomme_data/record_dataset_BinFill.h5   (already downloaded)
#   models/<task>_bc_seed<N>/bc_final.zip    (written by train_bc)
#   models/<task>_bc_seed<N>/bc_best.zip
DATA_VOLUME = modal.Volume.from_name("memory-rl-data")
DATA_MOUNT = "/data"

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


@app.function(
    image=image,
    timeout=120,
    volumes={DATA_MOUNT: DATA_VOLUME},
)
def inspect_h5(h5_path: str = "robomme_data/record_dataset_BinFill.h5") -> dict:
    """Print the HDF5 structure so you can verify the dataset layout before training.

    Usage:
        modal run modal_app/app.py::inspect_h5
        modal run modal_app/app.py::inspect_h5 --h5-path robomme_data/record_dataset_BinFill.h5
    """
    import json
    import os
    import sys

    os.environ.update(COMMON_ENV)
    sys.path.insert(0, "/workspace")

    from training.train_bc import inspect_h5 as _inspect  # type: ignore

    full_path = f"{DATA_MOUNT}/{h5_path}"
    structure = _inspect(full_path, max_depth=5)
    print(json.dumps(structure, indent=2, default=str))
    return structure


@app.function(
    image=image,
    gpu=GPU,
    timeout=4 * 60 * 60,
    volumes={DATA_MOUNT: DATA_VOLUME},
)
def train_bc(
    task: str = "BinFill",
    seed: int = 0,
    bc_epochs: int = 100,
    bc_batch_size: int = 256,
    bc_lr: float = 3e-4,
    l2_coef: float = 1e-4,
    vf_pretrain_epochs: int = 20,
    max_episodes: int | None = None,
    h5_filename: str | None = None,
) -> str:
    """Behavioral Cloning from the RoboMME HDF5 dataset on Modal.

    Reads the record HDF5 from the data volume, trains an MlpPolicy with
    supervised NLL loss (no environment rollouts needed), and saves the model
    back to the volume. No SAPIEN / GPU physics required — GPU is used only
    for the neural network forward/backward passes.

    Usage:
        modal run modal_app/app.py::train_bc
        modal run modal_app/app.py::train_bc --task BinFill --seed 0 --bc-epochs 100

    Download the result:
        modal volume get memory-rl-data models/BinFill_bc_seed0/bc_final.zip ./bc_BinFill_seed0.zip

    Load for PPO fine-tuning on the VM:
        python training/train_ppo.py --config configs/ppo.yaml \\
            --bc_checkpoint bc_BinFill_seed0.zip
    """
    import json
    import os
    import sys
    from pathlib import Path

    os.environ.update(COMMON_ENV)
    sys.path.insert(0, "/workspace")

    from training.train_bc import (  # type: ignore
        H5BCDataset,
        _build_policy_from_spaces,
        _load_vf_episodes_from_h5,
        pretrain_value_head,
        train_bc as _train_bc,
    )
    from training.utils import set_global_seed  # type: ignore

    # ---- locate H5 file -------------------------------------------------------
    if h5_filename is None:
        h5_filename = f"record_dataset_{task}.h5"
    h5_path = f"{DATA_MOUNT}/robomme_data/{h5_filename}"
    print(f"[train_bc] loading dataset from {h5_path}")

    # ---- dataset --------------------------------------------------------------
    dataset = H5BCDataset(h5_path, max_episodes=max_episodes)

    # ---- config ---------------------------------------------------------------
    cfg = {
        "policy": "MlpPolicy",
        "policy_kwargs": {},
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 8,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "seed": seed,
        "device": "auto",
        # BC-specific
        "bc_epochs": bc_epochs,
        "bc_batch_size": bc_batch_size,
        "bc_lr": bc_lr,
        "bc_l2_coef": l2_coef,
        "bc_log_every": 10,
        "vf_pretrain_epochs": vf_pretrain_epochs,
        "vf_pretrain_lr": bc_lr,
    }

    # ---- policy (no SAPIEN / env rollouts needed) -----------------------------
    set_global_seed(seed)
    model = _build_policy_from_spaces(cfg, dataset.obs_dim, dataset.act_dim)

    # ---- train ----------------------------------------------------------------
    run_dir = Path(f"{DATA_MOUNT}/models/{task}_bc_seed{seed}")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    _train_bc(dataset, model, cfg, run_dir)

    # ---- value head pretraining -----------------------------------------------
    vf_episodes = _load_vf_episodes_from_h5(
        h5_path, gamma=cfg["gamma"], max_episodes=max_episodes
    )
    pretrain_value_head(vf_episodes, model, cfg, run_dir)

    # ---- save final & commit to volume ----------------------------------------
    final_path = run_dir / "checkpoints" / "bc_final.zip"
    model.save(str(final_path))
    DATA_VOLUME.commit()

    log_path = run_dir / "bc_train_log.json"
    bc_log = {}
    if log_path.exists():
        bc_log = json.loads(log_path.read_text())
    bc_entries = bc_log.get("bc", [])
    last_entry = bc_entries[-1] if bc_entries else {}

    nll_val = last_entry.get("nll")
    nll_str = f"{nll_val:.4f}" if nll_val is not None else "?"
    vf_entries = bc_log.get("vf_pretrain", [])
    vf_mse_val = vf_entries[-1].get("vf_mse") if vf_entries else None
    vf_mse_str = f"{vf_mse_val:.4f}" if vf_mse_val is not None else "?"
    print(f"\n[train_bc] model saved to volume: {final_path}")
    print(f"[train_bc]   bc nll={nll_str}  vf mse={vf_mse_str}  "
          f"transitions={len(dataset)}")
    print(f"\nDownload:")
    print(f"  modal volume get memory-rl-data "
          f"models/{task}_bc_seed{seed}/checkpoints/bc_final.zip "
          f"./bc_{task}_seed{seed}.zip")
    print(f"\nWarm-start PPO:")
    print(f"  python training/train_ppo.py --config configs/ppo.yaml \\")
    print(f"      --bc_checkpoint bc_{task}_seed{seed}.zip")

    return str(final_path)


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
