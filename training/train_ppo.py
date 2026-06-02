"""Vanilla PPO baseline training entrypoint.

Uses Stable-Baselines3 PPO. Reads defaults from configs/ppo.yaml and accepts
CLI overrides for the things you usually want to vary per-run.

Example:
    python training/train_ppo.py --config configs/ppo.yaml \\
        --task spatial_memory --seed 0 --total_steps 100000
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
from pathlib import Path

# Print a Python+C stack trace on segfault so SAPIEN crashes are diagnosable.
faulthandler.enable()

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
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        return PPO, CheckpointCallback, EvalCallback, Monitor, DummyVecEnv, SubprocVecEnv
    except ImportError as e:
        raise SystemExit(
            "stable-baselines3 is required for PPO training.\n"
            "Install with: pip install -r requirements.txt"
        ) from e


def _make_vec_env(task_name, seed, n_envs, allow_gym_fallback, env_kwargs,
                  Monitor, DummyVecEnv, SubprocVecEnv, *, force_dummy=False):
    # Capture repo root in the parent so subprocesses can add it to sys.path.
    # With spawn, sys.path is NOT inherited (each subprocess is a fresh Python).
    repo_root = str(Path(__file__).resolve().parents[1])

    def _thunk(rank: int):
        def _f():
            # Print any crash to stderr immediately so it isn't lost when the
            # subprocess dies before the parent can read from the pipe.
            import sys as _sys
            try:
                if repo_root not in _sys.path:
                    _sys.path.insert(0, repo_root)
                from env.robomme_env import make_env as _make_env       # type: ignore
                from stable_baselines3.common.monitor import Monitor as _Monitor
                return _Monitor(
                    _make_env(
                        task_name, seed=seed + rank,
                        allow_gym_fallback=allow_gym_fallback,
                        env_kwargs=env_kwargs,
                    )
                )
            except Exception:
                import traceback
                _sys.stderr.write(
                    f"\n[vec_env worker rank={rank}] env creation failed:\n"
                    + traceback.format_exc() + "\n"
                )
                _sys.stderr.flush()
                raise
        return _f

    fns = [_thunk(i) for i in range(n_envs)]
    if force_dummy or n_envs == 1:
        return DummyVecEnv(fns)

    # SubprocVecEnv + spawn is the only CUDA-safe multiprocessing strategy:
    # fork inherits torch.cuda._original_pid from the parent and any CUDA call
    # in the child raises "Cannot re-initialize CUDA in forked subprocess".
    #
    # However, spawning n_envs SAPIEN/ManiSkill processes simultaneously on a
    # single GPU is often fatal: each fresh process imports torch + SAPIEN and
    # initialises a GPU physics context. With n_envs=16 this can exceed GPU
    # memory or hit kernel resource limits and the process is OOM-killed.
    #
    # Fallback: if SubprocVecEnv fails, drop to a single DummyVecEnv env.
    # Training continues — just slower. Use --n_envs 1 to avoid the attempt.
    try:
        return SubprocVecEnv(fns, start_method="spawn")
    except Exception as exc:
        import sys as _sys
        print(
            f"\n[train_ppo] SubprocVecEnv(n_envs={n_envs}) failed: {exc}\n"
            f"           Falling back to DummyVecEnv(n_envs=1).\n"
            f"           Pass --n_envs 1 to use a single env from the start.",
            file=_sys.stderr, flush=True,
        )
        return DummyVecEnv([_thunk(0)])


def _run_post_eval(model, cfg, env_kwargs, run_dir):
    """Run deterministic evaluation on the val split and print + save metrics."""
    from env.robomme_env import make_env
    from metrics.evaluation import EpisodeRecord, summarize

    print("[train_ppo] running post-training evaluation ...")
    post_env_kwargs = dict(env_kwargs)
    post_env_kwargs["dataset"] = "val"
    env = make_env(
        cfg["task_name"],
        seed=cfg["seed"] + 20_000,
        allow_gym_fallback=cfg.get("allow_gym_fallback", False),
        env_kwargs=post_env_kwargs,
    )
    n_eval = cfg.get("eval_episodes", 5)
    episodes = []
    for i in range(n_eval):
        obs, _ = env.reset()
        observations, rewards, infos = [obs], [], []
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            infos.append(info if isinstance(info, dict) else {})
            observations.append(obs)
        ep = EpisodeRecord(
            observations=observations, rewards=rewards,
            terminated=bool(terminated), truncated=bool(truncated), infos=infos,
        )
        episodes.append(ep)
        print(f"[eval] episode {i+1}/{n_eval}  "
              f"return={sum(ep.rewards):.3f}  len={len(ep.rewards)}  "
              f"term={ep.terminated}  trunc={ep.truncated}")
    env.close()

    metrics = summarize(episodes)
    print(json.dumps(metrics, indent=2))
    metrics_path = run_dir / "eval_final.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train_ppo] final eval saved to {metrics_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--task", dest="task_name", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--total_steps", dest="total_timesteps", type=int, default=None)
    ap.add_argument("--n_envs", type=int, default=None,
                    help="Override n_envs from config (e.g. 4 for local CPU runs)")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--allow-gym-fallback", dest="allow_gym_fallback", action="store_true", default=None)
    ap.add_argument(
        "--bc_checkpoint", default=None,
        help="Path to a BC-pretrained .zip to warm-start the PPO policy weights",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    overrides = {
        k: v for k, v in vars(args).items()
        if k not in ("config", "tag", "bc_checkpoint") and v is not None
    }
    cfg = merge_overrides(cfg, overrides)

    PPO, CheckpointCallback, EvalCallback, Monitor, DummyVecEnv, SubprocVecEnv = _require_sb3()

    set_global_seed(cfg["seed"], deterministic=cfg.get("deterministic_torch", False))
    run_dir = build_run_dir(cfg["output_dir"], cfg["task_name"], cfg["seed"], tag=args.tag)
    save_run_config(run_dir, cfg)

    n_envs = cfg.get("n_envs", 1)

    # batch_size must divide n_envs × n_steps exactly (SB3 hard requirement).
    # Auto-adjust downward to the largest valid divisor so the run never errors.
    rollout_size = n_envs * cfg["n_steps"]
    batch_size = cfg["batch_size"]
    if rollout_size % batch_size != 0:
        adjusted = next(bs for bs in range(batch_size, 0, -1) if rollout_size % bs == 0)
        print(f"[train_ppo] WARNING: batch_size={batch_size} does not divide "
              f"n_envs({n_envs}) × n_steps({cfg['n_steps']}) = {rollout_size}. "
              f"Auto-adjusted to {adjusted}.")
        cfg["batch_size"] = adjusted

    print(f"[train_ppo] run_dir={run_dir}  n_envs={n_envs}  "
          f"rollout={rollout_size}  batch_size={cfg['batch_size']}")

    env_kwargs = cfg.get("env_kwargs") or {}
    env = _make_vec_env(
        cfg["task_name"], cfg["seed"], n_envs,
        cfg.get("allow_gym_fallback", False), env_kwargs,
        Monitor, DummyVecEnv, SubprocVecEnv,
    )

    # Eval always uses a single DummyVecEnv — no need for SubprocVecEnv overhead.
    eval_env_kwargs = dict(env_kwargs)
    eval_env_kwargs["dataset"] = "val"
    eval_env = _make_vec_env(
        cfg["task_name"], cfg["seed"] + 10_000, 1,
        cfg.get("allow_gym_fallback", False), eval_env_kwargs,
        Monitor, DummyVecEnv, SubprocVecEnv, force_dummy=True,
    )

    model = PPO(
        policy=cfg.get("policy", "MlpPolicy"),
        env=env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg.get("n_epochs", 4),
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        tensorboard_log=str(run_dir / "tb"),
        seed=cfg["seed"],
        device=cfg.get("device", "auto"),
        policy_kwargs=cfg.get("policy_kwargs") or {},
        verbose=1,
    )

    bc_checkpoint = args.bc_checkpoint or cfg.get("bc_checkpoint")
    if bc_checkpoint:
        bc_path = Path(bc_checkpoint)
        if not bc_path.exists():
            raise FileNotFoundError(f"BC checkpoint not found: {bc_path}")
        print(f"[train_ppo] warm-starting policy from BC checkpoint: {bc_path}")
        bc_model = PPO.load(str(bc_path), device=cfg.get("device", "auto"))
        bc_obs_shape = bc_model.policy.observation_space.shape
        ppo_obs_shape = model.policy.observation_space.shape
        if bc_obs_shape != ppo_obs_shape:
            print(
                f"[train_ppo] WARNING: BC obs shape {bc_obs_shape} ≠ "
                f"env obs shape {ppo_obs_shape}.\n"
                f"           This usually means the live env obs extraction is\n"
                f"           incomplete (e.g. EEF state missing from extra dict).\n"
                f"           Check the stderr output from robomme_env.py for the\n"
                f"           exact extra keys and update _extract_native_maniskill_obs.\n"
                f"           Skipping BC warm-start — training from random init."
            )
        else:
            model.policy.load_state_dict(bc_model.policy.state_dict())
            print("[train_ppo] BC policy weights loaded")
        del bc_model

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
            save_freq=max(1, cfg["checkpoint_interval"] // max(1, n_envs)),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "checkpoints" / "best"),
            log_path=str(run_dir / "eval"),
            eval_freq=max(1, cfg["eval_interval"] // max(1, n_envs)),
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

    _run_post_eval(model, cfg, env_kwargs, run_dir)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
