"""Modal entrypoint for RoboMME training/eval.

Why this exists
---------------
SAPIEN/ManiSkill require a Vulkan ICD that WSL2 doesn't expose and Modal
does not support SAPIEN/Vulkan. Offline IQL training reads directly from H5
demonstration files — no environment stepping, no Vulkan dependency — and
runs fully on Modal. RoboMME source is cloned into the image at build time;
datasets and run outputs are persisted on a Modal Volume.

Offline IQL workflow (primary — no SAPIEN needed)
-------------------------------------------------
    # 1. One-time: download H5 episode datasets onto the Modal volume.
    modal run modal_app/app.py::download_data
    modal run modal_app/app.py::download_data --tasks BinFill,PickXtimes,StopCube,VideoUnmask

    # 2. One-time: verify H5 structure matches expected obs/action keys.
    #    Check printed key names; update obs_keys in configs/iql.yaml if needed.
    modal run modal_app/app.py::inspect_h5 --task BinFill

    # 3. Train IQL offline on a single task/seed.
    modal run modal_app/app.py::train_iql --task BinFill --seed 0
    modal run modal_app/app.py::train_iql --task BinFill --seed 0 --steps 1000000

    # 4. Train 4 tasks x 3 seeds in parallel.
    modal run modal_app/app.py::iql_sweep
    modal run modal_app/app.py::iql_sweep --steps 1000000

Online PPO workflow (blocked — requires SAPIEN/Vulkan)
------------------------------------------------------
    # These commands require a machine with full SAPIEN/Vulkan support.
    # They are kept here for reference but will not run on Modal.

    # Verify RoboMME env steps correctly (needs Vulkan).
    modal run modal_app/app.py::smoke --task BinFill

    # Single PPO training run.
    modal run modal_app/app.py::train_ppo --task BinFill --seed 0 --steps 100000

    # Evaluate a PPO checkpoint.
    modal run modal_app/app.py::evaluate --task BinFill --seed 0

    # Random-policy sanity check.
    modal run modal_app/app.py::random_baseline --task BinFill --episodes 20

    # Full PPO sweep: 4 tasks x 3 seeds in parallel.
    modal run modal_app/app.py::sweep
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Persistent volume — survives between function calls and across modal runs.
#   /vol/robomme_data  — downloaded RoboMME episode datasets
#   /vol/logs          — PPO checkpoints, TensorBoard, eval metrics
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("memory-rl-data", create_if_missing=True)
VOLUME_PATH = "/vol"
DATA_DIR = f"{VOLUME_PATH}/robomme_data"
LOGS_DIR = f"{VOLUME_PATH}/logs"

# ---------------------------------------------------------------------------
# Image — CUDA 12.8 base, Vulkan ICD, all Python deps, RoboMME source.
# RoboMME is cloned at image-build time so no local checkout is needed.
# ---------------------------------------------------------------------------
ROBOMME_REV = "07be6fbc66350ddca200abfb0a11b692f078f7fd"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04",
        add_python="3.11",
    )
    .apt_install(
        "build-essential",
        "clang",              # toppra (mani-skill dep) hardcodes clang for its Cython ext
        "ca-certificates",
        "curl",
        "ffmpeg",
        "git",
        "libegl1",
        "libgl1",
        "libglib2.0-0",
        "libglvnd0",
        "libvulkan1",         # Vulkan loader
        "mesa-vulkan-drivers", # lavapipe CPU Vulkan — Modal GPU containers don't expose NVIDIA Vulkan graphics;
                              # lavapipe lets SAPIEN's render system initialise without a GPU graphics stack.
                              # Physics still runs on GPU via CUDA.
        "vulkan-tools",       # vulkaninfo for diagnostics
        "libxext6",
        "libxrender1",
    )
    .run_commands(
        # Pre-create dirs that SAPIEN's _vulkan_tricks.py tries to write ICD patches into at
        # runtime. Without them the silent write fails and SAPIEN falls back to nothing.
        "mkdir -p /etc/vulkan/icd.d /etc/vulkan/implicit_layer.d /etc/glvnd/egl_vendor.d",
    )
    .pip_install(
        # Pinned per RoboMME pyproject — mutually compatible with mani-skill 3.0.0b21.
        "torch==2.9.1",
        "torchvision==0.24.1",
        f"mani-skill @ git+https://github.com/YinpeiDai/ManiSkill.git@{ROBOMME_REV}",
        "h5py",
        "opencv-python>=4.11.0.86",
        "stable-baselines3==2.8.0",
        "tensorboard",
        "pyyaml",
        "wandb>=0.18",
        "huggingface_hub>=0.26",  # for snapshot_download of robomme_data_h5
    )
    .run_commands(
        "git clone https://github.com/RoboMME/robomme_benchmark /robomme_src",
        "cd /robomme_src && pip install -e .",
    )
    .env(
        {
            # graphics/video capabilities are not needed — lavapipe handles Vulkan in software.
            # compute is needed for CUDA physics (PhysX); utility for nvidia-smi / diagnostics.
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            # Point both env-var spellings at lavapipe so the Vulkan loader never tries
            # libGLX_nvidia.so.0, which fails in Modal's container environment.
            "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json",
            "VK_DRIVER_FILES":  "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json",
            "SAPIEN_RENDER_DEVICE": "0",
            "PYTHONUNBUFFERED": "1",
            "WANDB_PROJECT": "memory-rl",
        }
    )
)

# This repo's training code is uploaded from your local working copy each run.
# Only this directory needs to exist locally — no other local paths required.
REPO_ROOT = Path(__file__).resolve().parents[1]
image = image.add_local_dir(str(REPO_ROOT), remote_path="/workspace")

app = modal.App("memory-rl")

GPU = "A10G"  # cheapest Modal GPU that comfortably runs SAPIEN; bump to A100 for sweeps
WANDB_SECRET = modal.Secret.from_name("wandb")

# Environment variables set inside every Modal function.
COMMON_ENV = {
    "ROBOMME_PATH": "/robomme_src",
    "ROBOMME_DATA_DIR": DATA_DIR,   # picked up by env/robomme_env.py → BenchmarkEnvBuilder
    "MANI_SKILL_DATA": DATA_DIR,    # ManiSkill 3 standard data path
    "MS_ASSET_DIR": DATA_DIR,       # ManiSkill 3 asset path (demos + meshes)
    "PYTHONPATH": "/workspace:/robomme_src/src",
}


def _setup():
    """Bootstrap sys.path inside a Modal container. Call at the top of every function."""
    os.environ.update(COMMON_ENV)
    sys.path.insert(0, "/workspace")
    sys.path.insert(0, "/robomme_src/src")


# ---------------------------------------------------------------------------
# download_data — run once to populate the volume before training
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=None,           # no GPU needed for a plain file download
    timeout=60 * 60,
    volumes={VOLUME_PATH: volume},
)
def download_data(tasks: str = "BinFill,PickXtimes,SwingXtimes,StopCube"):
    """Download RoboMME episode datasets from HuggingFace to the Modal volume.

    Source: https://huggingface.co/datasets/Yinpei/robomme_data_h5

    Must be run once before train_ppo, evaluate, or smoke. Data persists on
    the volume across all future runs so this only needs to be run once (or
    again if you want to refresh the data).

    Args:
        tasks: Comma-separated task names to download. Defaults to the four
               Counting suite tasks (BinFill, PickXtimes, SwingXtimes,
               StopCube). Pass "all" to download the full 16-task dataset.
               Partial downloads use allow_patterns so only matching
               archives are fetched.

    If the HuggingFace repo is private, create a Modal secret first:
        modal secret create huggingface HF_TOKEN=<your_token>
    and add secrets=[modal.Secret.from_name("huggingface")] to this function.
    """
    import tarfile

    from huggingface_hub import snapshot_download

    _setup()
    volume.reload()

    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    # Build allow_patterns to fetch only requested tasks.
    # HF repo structure: record_dataset_<TaskName>.h5.tar.xz (flat at repo root)
    if tasks.strip().lower() == "all":
        allow_patterns = None   # download everything
    else:
        task_list = [t.strip() for t in tasks.split(",") if t.strip()]
        allow_patterns = [f"record_dataset_{t}.h5.tar.xz" for t in task_list]
        print(f"[download_data] filtering to tasks: {task_list}")

    print(f"[download_data] downloading Yinpei/robomme_data_h5 → {DATA_DIR} ...")
    snapshot_download(
        repo_id="Yinpei/robomme_data_h5",
        repo_type="dataset",
        local_dir=DATA_DIR,
        local_dir_use_symlinks=False,  # write real files to volume; symlinks break after container exit
        allow_patterns=allow_patterns,
        ignore_patterns=["*.gitattributes", ".gitattributes", "README.md"],
        token=os.environ.get("HF_TOKEN"),   # None is fine for public repos
    )

    # Extract all downloaded .tar.xz archives in-place.
    archives = sorted(Path(DATA_DIR).glob("*.tar.xz"))
    print(f"[download_data] extracting {len(archives)} archive(s) ...")
    for archive in archives:
        print(f"  extracting {archive.name} ...")
        with tarfile.open(archive, "r:xz") as tf:
            tf.extractall(DATA_DIR)

    # List what was extracted so the caller can verify.
    downloaded = sorted(str(p) for p in Path(DATA_DIR).rglob("*.h5"))
    print(f"[download_data] done — {len(downloaded)} .h5 files in {DATA_DIR}")
    for p in downloaded[:20]:
        print(f"  {p}")
    if len(downloaded) > 20:
        print(f"  ... and {len(downloaded) - 20} more")

    volume.commit()
    return {"data_dir": DATA_DIR, "num_h5_files": len(downloaded)}


# ---------------------------------------------------------------------------
# smoke — verify RoboMME imports and can step the env
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU,
    timeout=600,
    volumes={VOLUME_PATH: volume},
)
def smoke(task: str = "BinFill") -> dict:
    """Confirm RoboMME imports + env steps correctly on Modal."""
    import subprocess

    _setup()
    volume.reload()

    # Vulkan diagnostic — runs before SAPIEN import so we see raw driver state.
    # Expected: lavapipe (CPU Vulkan) reported as the active device.
    print("[smoke] vulkaninfo --summary:")
    r = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True)
    print(r.stdout or "(no stdout)")
    if r.returncode != 0:
        print(f"[smoke] vulkaninfo FAILED (exit {r.returncode}):\n{r.stderr}")
    else:
        print("[smoke] vulkaninfo OK")

    from scripts.inspect_env import main as inspect_main  # type: ignore

    sys.argv = ["inspect_env", "--task", task]
    rc = inspect_main()
    return {"task": task, "exit_code": rc}


# ---------------------------------------------------------------------------
# train_ppo — single PPO run, outputs saved to the volume
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU,
    timeout=3 * 60 * 60,
    secrets=[WANDB_SECRET],
    volumes={VOLUME_PATH: volume},
)
def train_ppo(
    task: str = "BinFill",
    seed: int = 0,
    steps: int = 100_000,
    tag: str | None = None,
) -> dict:
    """Train a vanilla PPO baseline and save checkpoints to the volume.

    Returns a dict with the run directory path so evaluate() can find it.
    """
    _setup()
    volume.reload()

    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

    import wandb  # noqa: F401 — ensures WANDB_API_KEY secret is wired before SB3 imports

    from training.train_ppo import main as train_main  # type: ignore

    sys.argv = [
        "train_ppo",
        "--config", "/workspace/configs/ppo.yaml",
        "--task", task,
        "--seed", str(seed),
        "--total_steps", str(steps),
        "--output_dir", LOGS_DIR,
    ]
    if tag:
        sys.argv += ["--tag", tag]

    train_main()
    volume.commit()

    # Find the run dir by modification time — train_main creates it with a live timestamp.
    run_dirs = sorted(Path(LOGS_DIR).glob(f"{task}_seed{seed}_*"), key=lambda p: p.stat().st_mtime)
    run_name = run_dirs[-1].name if run_dirs else None
    return {"task": task, "seed": seed, "run_name": run_name, "logs_dir": LOGS_DIR}


# ---------------------------------------------------------------------------
# evaluate — load a saved checkpoint from the volume and compute metrics
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,
    volumes={VOLUME_PATH: volume},
)
def evaluate(
    task: str = "BinFill",
    seed: int = 0,
    episodes: int = 20,
    run_name: str | None = None,
    random: bool = False,
) -> dict:
    """Evaluate a PPO checkpoint (or random policy) from the volume.

    Args:
        task:      RoboMME task name.
        seed:      Seed used during training (used to locate the run dir).
        episodes:  Number of evaluation episodes.
        run_name:  Exact run directory name inside the volume logs dir.
                   If None, the most recently modified run for (task, seed)
                   is used automatically.
        random:    If True, run a random policy instead of loading a checkpoint.

    Returns metrics dict (success_rate, average_return, etc.).
    """
    import json

    _setup()
    volume.reload()

    from training.evaluate import main as eval_main  # type: ignore

    if random:
        checkpoint_args = ["--random"]
    else:
        logs_path = Path(LOGS_DIR)
        if run_name:
            run_dir = logs_path / run_name
        else:
            candidates = sorted(
                logs_path.glob(f"{task}_seed{seed}_*"),
                key=lambda p: p.stat().st_mtime,
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No run dirs found for task={task} seed={seed} in {LOGS_DIR}. "
                    "Run train_ppo first."
                )
            run_dir = candidates[-1]

        checkpoint = run_dir / "checkpoints" / "ppo_final.zip"
        if not checkpoint.exists():
            checkpoint = run_dir / "checkpoints" / "best" / "best_model.zip"
        if not checkpoint.exists():
            raise FileNotFoundError(f"No checkpoint found in {run_dir / 'checkpoints'}")

        print(f"[evaluate] checkpoint: {checkpoint}")
        checkpoint_args = ["--checkpoint", str(checkpoint)]

    out_dir = Path(LOGS_DIR) / f"eval_{task}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.argv = [
        "evaluate",
        *checkpoint_args,
        "--task", task,
        "--seed", str(seed),
        "--episodes", str(episodes),
        "--save_trajectories",
        "--out_dir", str(out_dir),
    ]
    eval_main()
    volume.commit()

    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# random_baseline — sanity-check metrics before any training
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,
    volumes={VOLUME_PATH: volume},
)
def random_baseline(task: str = "BinFill", episodes: int = 20, seed: int = 0) -> dict:
    """Run a random policy and return metrics. No training or data download needed."""
    _setup()
    volume.reload()

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


# ---------------------------------------------------------------------------
# inspect_h5 — print H5 file structure so we can verify obs/action keys
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=None,
    timeout=900,
    volumes={VOLUME_PATH: volume},
)
def inspect_h5(task: str = "PickXtimes", n_timesteps: int = 3) -> dict:
    """Print a safe summary of the H5 structure without walking all timestep groups.

    Shows top-level key counts, root obs/info structure, and the first
    n_timesteps timestep groups. Safe on large files (no full tree walk).

    Example:
        modal run modal_app/app.py::inspect_h5 --task PickXtimes
    """
    _setup()
    volume.reload()

    from data.h5_dataset import find_h5_files, inspect_h5 as _inspect  # type: ignore

    paths = find_h5_files(DATA_DIR, task)
    if not paths:
        raise FileNotFoundError(
            f"No H5 files for task '{task}' in {DATA_DIR}. Run download_data first."
        )

    for path in paths[:1]:
        _inspect(path, n_timesteps=n_timesteps)

    return {"task": task, "files": [str(p) for p in paths]}


# ---------------------------------------------------------------------------
# train_iql — offline IQL from H5 data (no SAPIEN / Vulkan needed)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU,
    timeout=6 * 60 * 60,
    secrets=[WANDB_SECRET],
    volumes={VOLUME_PATH: volume},
)
def train_iql(
    task: str = "BinFill",
    seed: int = 0,
    steps: int = 500_000,
    tag: str | None = None,
) -> dict:
    """Train an IQL agent offline from H5 demonstrations — no SAPIEN/Vulkan needed.

    H5 data must already be on the volume (run download_data first).
    Checkpoints are written to the volume under logs/<run_name>/checkpoints/.

    Example:
        modal run modal_app/app.py::train_iql --task BinFill --seed 0
        modal run modal_app/app.py::train_iql --task BinFill --seed 0 --steps 1000000

    Returns the run directory name (useful for scripted pipelines).
    """
    import signal

    _setup()
    volume.reload()

    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

    # Commit the volume on SIGTERM (Modal cancellation) so any checkpoints
    # already written to /vol are persisted before the container exits.
    def _sigterm_handler(_signum, _frame):
        print("[train_iql] SIGTERM received — committing volume before exit")
        volume.commit()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    import wandb  # noqa: F401 — wire WANDB_API_KEY before training imports

    from training.train_iql import main as iql_main  # type: ignore

    sys.argv = [
        "train_iql",
        "--config",      "/workspace/configs/iql.yaml",
        "--task",        task,
        "--seed",        str(seed),
        "--total_steps", str(steps),
        "--output_dir",  LOGS_DIR,
        "--data_dir",    DATA_DIR,
    ]
    if tag:
        sys.argv += ["--tag", tag]

    def _on_checkpoint(step, path):
        print(f"[train_iql] committing checkpoint step={step} → volume")
        volume.commit()

    try:
        iql_main(on_checkpoint=_on_checkpoint)
    finally:
        volume.commit()

    run_dirs = sorted(
        Path(LOGS_DIR).glob(f"{task}_seed{seed}_*"),
        key=lambda p: p.stat().st_mtime,
    )
    run_name = run_dirs[-1].name if run_dirs else None
    return {"task": task, "seed": seed, "run_name": run_name, "logs_dir": LOGS_DIR}


# ---------------------------------------------------------------------------
# eval_iql_offline — offline IQL evaluation from H5 dataset (no SAPIEN needed)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,
    volumes={VOLUME_PATH: volume},
)
def eval_iql_offline(
    task: str = "PickXtimes",
    seed: int = 0,
    checkpoint: str | None = None,
    val_fraction: float = 0.2,
) -> dict:
    """Evaluate a trained IQL checkpoint against the offline H5 dataset.

    No SAPIEN / Vulkan required — runs fully offline on Modal.

    Checkpoint resolution order:
      1. Explicit --checkpoint path (e.g. /workspace/iql_final.pt for the
         repo-root file uploaded from your local machine, or an absolute path
         inside the volume).
      2. Most recently modified iql_final.pt for (task, seed) in the volume
         logs directory (i.e. the output of a prior train_iql run).

    Examples:
        modal run modal_app/app.py::eval_iql_offline --task PickXtimes
        modal run modal_app/app.py::eval_iql_offline --task PickXtimes --seed 0 \\
            --checkpoint /workspace/iql_final.pt

    Returns the metrics dict (action_mse, action_rmse, v_mean, q_mean,
    adv_mean, adv_positive_fraction, dataset_success_rate, num_transitions).
    """
    import json

    _setup()
    volume.reload()

    if checkpoint is None:
        candidates = sorted(
            Path(LOGS_DIR).glob(f"{task}_seed{seed}_*/checkpoints/iql_final.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No iql_final.pt found for task={task} seed={seed} in {LOGS_DIR}. "
                "Pass --checkpoint explicitly or run train_iql first."
            )
        ckpt_path = str(candidates[-1])
    else:
        ckpt_path = checkpoint

    print(f"[eval_iql_offline] checkpoint: {ckpt_path}")

    out_dir = f"{LOGS_DIR}/eval_offline_{task}_seed{seed}"

    from training.eval_iql_offline import main as offline_eval_main  # type: ignore

    sys.argv = [
        "eval_iql_offline",
        "--checkpoint",   ckpt_path,
        "--task",         task,
        "--seed",         str(seed),
        "--data_dir",     DATA_DIR,
        "--val_fraction", str(val_fraction),
        "--out_dir",      out_dir,
    ]
    offline_eval_main()
    volume.commit()

    metrics_path = Path(out_dir) / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# train_eval_iql — train IQL then evaluate on the held-out val split
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def train_eval_iql(
    task: str = "PickXtimes",
    seed: int = 0,
    steps: int = 500_000,
    val_fraction: float = 0.2,
    tag: str | None = None,
):
    """Train IQL offline, then evaluate on the held-out val split.

    Episodes are split deterministically: the last val_fraction of sorted
    episode keys are withheld from training and used exclusively for eval.
    Both steps use the same val_fraction so the split is consistent.

    Example:
        modal run modal_app/app.py::train_eval_iql --task PickXtimes --seed 0
        modal run modal_app/app.py::train_eval_iql --task PickXtimes --seed 0 \\
            --steps 1000000 --val-fraction 0.2
    """
    import json

    print(f"[train_eval_iql] task={task}  seed={seed}  steps={steps}  "
          f"val_fraction={val_fraction}")

    print("[train_eval_iql] === phase 1: training ===")
    train_result = train_iql.remote(task=task, seed=seed, steps=steps, tag=tag)
    run_name = train_result.get("run_name")
    print(f"[train_eval_iql] training complete — run={run_name}")

    # Use the exact checkpoint from this run to avoid ambiguity.
    ckpt = f"{LOGS_DIR}/{run_name}/checkpoints/iql_final.pt" if run_name else None

    print("[train_eval_iql] === phase 2: offline eval (val split) ===")
    metrics = eval_iql_offline.remote(
        task=task,
        seed=seed,
        checkpoint=ckpt,
        val_fraction=val_fraction,
    )

    print("\n[train_eval_iql] === results ===")
    print(json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------------
# iql_sweep — 4 tasks x 3 seeds offline IQL in parallel
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def iql_sweep(steps: int = 500_000):
    """Train IQL on 4 tasks x 3 seeds in parallel (no SAPIEN needed)."""
    tasks = ["BinFill", "PickXtimes", "StopCube", "VideoUnmask"]
    seeds = [0, 1, 2]
    jobs  = [(t, s) for t in tasks for s in seeds]

    print(f"[iql_sweep] launching {len(jobs)} IQL runs on Modal ({GPU}) ...")
    results = list(
        train_iql.starmap([(t, s, steps, "iql-baseline") for t, s in jobs])
    )
    print("\n[iql_sweep] all runs complete:")
    for r in results:
        print(f"  {r['task']:20s} seed={r['seed']}  run={r['run_name']}")


# ---------------------------------------------------------------------------
# sweep — 4 tasks x 3 seeds in parallel, then evaluate each
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def sweep(steps: int = 100_000):
    """Train 4 tasks x 3 seeds in parallel on Modal, then evaluate each."""
    tasks = ["BinFill", "PickXtimes", "StopCube", "VideoUnmask"]
    seeds = [0, 1, 2]
    jobs = [(t, s) for t in tasks for s in seeds]

    print(f"[sweep] launching {len(jobs)} training runs on Modal ({GPU}) ...")
    train_results = list(
        train_ppo.starmap([(t, s, steps, "baseline") for t, s in jobs])
    )

    print("[sweep] training done — running evaluation ...")
    eval_results = list(
        evaluate.starmap(
            [(r["task"], r["seed"], 20, r["run_name"]) for r in train_results]
        )
    )

    print("\n[sweep] results:")
    for (t, s), metrics in zip(jobs, eval_results):
        sr = metrics.get("success_rate", float("nan"))
        ret = metrics.get("average_return", float("nan"))
        print(f"  {t:20s} seed={s}  success_rate={sr:.3f}  avg_return={ret:.3f}")
