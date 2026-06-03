from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

# --------------------------------------------------
# App
# --------------------------------------------------
app = modal.App("memory-rl-bc-ptp")

REPO_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------
# Persistent checkpoint volume
# --------------------------------------------------
checkpoint_volume = modal.Volume.from_name(
    "memory-rl-checkpoints",
    create_if_missing=True,
)

# --------------------------------------------------
# Image
# --------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04",
        add_python="3.11",
    )
    .apt_install(
        "build-essential",
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "h5py",
        "wandb",
        "tqdm",
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/workspace",
    )
)

COMMON_ENV = {
    "PYTHONPATH": "/workspace",
}

GPU = "A10G"

CHECKPOINT_DIR = Path("/checkpoints")

# --------------------------------------------------
# Training function
# --------------------------------------------------
@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60 * 4,  # 4 hours
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/checkpoints": checkpoint_volume},
    memory=32768,  # 32GB RAM
)
def train_bc_ptp():
    os.environ.update(COMMON_ENV)
    sys.path.insert(0, "/workspace")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("[Modal] Starting BC + PTP training...")
    print(f"[Modal] Checkpoint directory: {CHECKPOINT_DIR}")

    existing = list(CHECKPOINT_DIR.glob("*"))
    print(f"[Modal] Existing checkpoints: {len(existing)}")

    from memory.train_bc_ptp import train

    train()

    # Persist volume changes
    checkpoint_volume.commit()

    print("\n[Modal] Final checkpoint contents:")
    for f in sorted(CHECKPOINT_DIR.glob("*")):
        print(f"  {f}")

    return "training finished"


# --------------------------------------------------
# Smoke test
# --------------------------------------------------
@app.function(
    image=image,
    gpu=GPU,
    timeout=10 * 60,
)
def smoke():
    os.environ.update(COMMON_ENV)
    sys.path.insert(0, "/workspace")

    from memory.dataset import RoboVLAPTPDataset
    from memory.model import SimpleVLA

    DATA_PATH = (
        "/workspace/data/data/robomme_data_h5/"
        "record_dataset_BinFill.h5"
    )

    dataset = RoboVLAPTPDataset(
        [DATA_PATH],
        history_len=8,
    )

    sample = dataset[0]

    model = SimpleVLA()

    print("Dataset OK")
    print(
        {
            k: getattr(v, "shape", None)
            for k, v in sample.items()
        }
    )

    print("Model OK")

    return "smoke passed"


# --------------------------------------------------
# List checkpoints
# --------------------------------------------------
@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
)
def list_checkpoints():
    files = sorted(str(f) for f in Path("/checkpoints").glob("*"))
    print(files)
    return files


# --------------------------------------------------
# Entry
# --------------------------------------------------
@app.local_entrypoint()
def main_entry():
    print("[Modal] Launching BC + PTP training...")

    result = train_bc_ptp.remote()

    print(result)