"""Vanilla PPO baseline training entrypoint.

Uses Stable-Baselines3 PPO. Reads defaults from configs/ppo.yaml and accepts
CLI overrides for the things you usually want to vary per-run.

Example:
    python training/train_ppo.py --config configs/ppo.yaml \\
        --task spatial_memory --seed 0 --total_steps 100000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.utils import (  # noqa: E402
    build_run_dir,
    load_yaml,
    merge_overrides,
    save_run_config,
    set_global_seed,
)


def _require_sb3():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        return PPO, CheckpointCallback, EvalCallback, Monitor, DummyVecEnv
    except ImportError as e:
        raise SystemExit(
            "stable-baselines3 is required for PPO training.\n"
            "Install with: pip install -r requirements.txt"
        ) from e


def _make_vec_env(task_name, seed, n_envs, allow_gym_fallback, env_kwargs, Monitor, DummyVecEnv):
    from env.robomme_env import make_env

    def _thunk(rank: int):
        def _f():
            env = make_env(
                task_name, seed=seed + rank,
                allow_gym_fallback=allow_gym_fallback,
                env_kwargs=env_kwargs,
            )
            return Monitor(env)
        return _f

    return DummyVecEnv([_thunk(i) for i in range(n_envs)])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--task", dest="task_name", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--total_steps", dest="total_timesteps", type=int, default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--allow-gym-fallback", dest="allow_gym_fallback", action="store_true", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    overrides = {k: v for k, v in vars(args).items() if k not in ("config", "tag") and v is not None}
    cfg = merge_overrides(cfg, overrides)

    PPO, CheckpointCallback, EvalCallback, Monitor, DummyVecEnv = _require_sb3()

    set_global_seed(cfg["seed"], deterministic=cfg.get("deterministic_torch", False))
    run_dir = build_run_dir(cfg["output_dir"], cfg["task_name"], cfg["seed"], tag=args.tag)
    save_run_config(run_dir, cfg)
    print(f"[train_ppo] run_dir={run_dir}")

    env_kwargs = cfg.get("env_kwargs") or {}
    env = _make_vec_env(
        cfg["task_name"], cfg["seed"], cfg.get("n_envs", 1),
        cfg.get("allow_gym_fallback", False), env_kwargs, Monitor, DummyVecEnv,
    )
    eval_env_kwargs = dict(env_kwargs)
    if "dataset" in eval_env_kwargs:
        eval_env_kwargs["dataset"] = "val"  # standard practice: evaluate on val split
    eval_env = _make_vec_env(
        cfg["task_name"], cfg["seed"] + 10_000, 1,
        cfg.get("allow_gym_fallback", False), eval_env_kwargs, Monitor, DummyVecEnv,
    )

    model = PPO(
        policy=cfg.get("policy", "MlpPolicy"),
        env=env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg.get("n_epochs", 10),
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        tensorboard_log=str(run_dir / "tb"),
        seed=cfg["seed"],
        device=cfg.get("device", "auto"),
        verbose=1,
    )

    wandb_run = None
    if os.environ.get("WANDB_API_KEY"):
        import wandb
        from wandb.integration.sb3 import WandbCallback
        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "memory-rl"),
            name=run_dir.name,
            config=cfg,
            sync_tensorboard=True,
            save_code=False,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=max(1, cfg["checkpoint_interval"] // max(1, cfg.get("n_envs", 1))),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "checkpoints" / "best"),
            log_path=str(run_dir / "eval"),
            eval_freq=max(1, cfg["eval_interval"] // max(1, cfg.get("n_envs", 1))),
            n_eval_episodes=cfg.get("eval_episodes", 5),
            deterministic=True,
            render=False,
        ),
    ]

    if wandb_run is not None:
        callbacks.append(WandbCallback(
            model_save_path=str(run_dir / "wandb_models"),
            verbose=1,
        ))

    model.learn(total_timesteps=cfg["total_timesteps"], callback=callbacks, progress_bar=False)
    final_path = run_dir / "checkpoints" / "ppo_final.zip"
    model.save(str(final_path))
    print(f"[train_ppo] saved final model to {final_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
