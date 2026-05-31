"""Offline IQL training — no environment stepping, no SAPIEN required.

Loads pre-recorded H5 demonstrations from ROBOMME_DATA_DIR (or --data_dir)
and trains an IQL agent entirely from the static dataset. Works on Modal
because it never calls BenchmarkEnvBuilder or any Vulkan-dependent code.

Local example (after downloading H5 data):
    ROBOMME_DATA_DIR=/vol/robomme_data \\
    python training/train_iql.py --config configs/iql.yaml --task BinFill --seed 0

On Modal (no Vulkan/SAPIEN needed):
    modal run modal_app/app.py::train_iql --task BinFill --seed 0 --steps 500000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.h5_dataset import make_dataloader          # noqa: E402
from training.iql import IQL                          # noqa: E402
from training.utils import (                          # noqa: E402
    build_run_dir,
    load_yaml,
    merge_overrides,
    save_run_config,
    set_global_seed,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      default="configs/iql.yaml")
    ap.add_argument("--task",        dest="task_name", default=None)
    ap.add_argument("--seed",        type=int,         default=None)
    ap.add_argument("--total_steps", type=int,         default=None)
    ap.add_argument("--output_dir",  default=None)
    ap.add_argument("--data_dir",     default=None, help="Path to directory containing H5 files")
    ap.add_argument("--val_fraction", type=float,   default=None,
                    help="Fraction of episodes held out for val (default: 0.2)")
    ap.add_argument("--tag",          default=None)
    return ap.parse_args()


def main(on_checkpoint=None) -> int:
    """Run offline IQL training.

    on_checkpoint: optional callable(step, path) called after each checkpoint
    is written — used by the Modal entrypoint to commit the volume so files
    survive cancellation.
    """
    args = parse_args()
    cfg  = load_yaml(args.config)
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("config", "tag") and v is not None}
    cfg = merge_overrides(cfg, overrides)

    set_global_seed(cfg["seed"])

    data_dir = (
        cfg.get("data_dir")
        or os.environ.get("ROBOMME_DATA_DIR")
        or os.environ.get("MANI_SKILL_DATA")
    )
    if not data_dir:
        raise SystemExit(
            "Data directory not set. Pass --data_dir or export ROBOMME_DATA_DIR."
        )

    run_dir  = build_run_dir(cfg["output_dir"], cfg["task_name"], cfg["seed"], tag=args.tag)
    save_run_config(run_dir, cfg)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_iql] run_dir={run_dir}")

    loader, obs_dim, action_dim = make_dataloader(
        data_dir=data_dir,
        task=cfg["task_name"],
        obs_keys=cfg.get("obs_keys", ("eef_state", "joint_state", "gripper_state")),
        action_key=cfg.get("action_key", "joint_action"),
        batch_size=cfg.get("batch_size", 256),
        num_workers=cfg.get("num_workers", 4),
        max_transitions=cfg.get("max_transitions"),
        reward_scale=cfg.get("reward_scale", 1.0),
        split="train",
        val_fraction=cfg.get("val_fraction", 0.2),
    )
    action_mean = loader.dataset.action_mean
    action_std  = loader.dataset.action_std

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_iql] device={device}  obs_dim={obs_dim}  action_dim={action_dim}")

    agent = IQL(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        hidden=cfg.get("hidden", 256),
        lr=cfg.get("learning_rate", 3e-4),
        gamma=cfg.get("gamma", 0.99),
        tau=cfg.get("tau", 0.005),
        expectile=cfg.get("expectile", 0.7),
        temperature=cfg.get("temperature", 3.0),
        adv_clip=cfg.get("adv_clip", cfg.get("advantage_clip", 100.0)),
        weight_clip=cfg.get("weight_clip", 100.0),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        action_mean=action_mean,
        action_std=action_std,
    )

    # TensorBoard
    tb = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=str(run_dir / "tb"))
    except ImportError:
        print("[train_iql] tensorboard not available — skipping TB logging")

    # WandB (optional — only if API key is set)
    wb = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wb = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "memory-rl"),
                name=run_dir.name,
                config=cfg,
                save_code=False,
            )
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
        except Exception as e:
            print(f"[train_iql] wandb init failed (continuing without it): {e}")

    total_steps   = cfg.get("total_steps", 500_000)
    log_interval  = cfg.get("log_interval", 1_000)
    save_interval = cfg.get("save_interval", 50_000)

    step      = 0
    t0        = time.time()
    data_iter = iter(loader)

    while step < total_steps:
        # Cycle through the dataset indefinitely
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch     = next(data_iter)

        metrics = agent.update(batch)
        step += 1

        if step % log_interval == 0:
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            print(
                f"[train_iql] step={step:>7}/{total_steps} | "
                f"v={metrics['v_loss']:.4f} q={metrics['q_loss']:.4f} "
                f"π={metrics['actor_loss']:.4f} | "
                f"{steps_per_sec:.0f} steps/s"
            )
            if tb:
                for k, v in metrics.items():
                    tb.add_scalar(f"train/{k}", v, step)
            if wb:
                wb.log({"train/step": step, "train/steps_per_sec": steps_per_sec,
                        **{f"train/{k}": v for k, v in metrics.items()}})

        if step % save_interval == 0 or step == total_steps:
            ckpt_path = ckpt_dir / f"iql_step{step}.pt"
            agent.save(ckpt_path)
            if tb:
                tb.flush()
            print(f"[train_iql] saved {ckpt_path}")
            if on_checkpoint:
                on_checkpoint(step, str(ckpt_path))

    final_path = ckpt_dir / "iql_final.pt"
    agent.save(final_path)
    print(f"[train_iql] training complete — final model: {final_path}")
    if on_checkpoint:
        on_checkpoint(total_steps, str(final_path))

    if wb:
        wb.finish()
    if tb:
        tb.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
