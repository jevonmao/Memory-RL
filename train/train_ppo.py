"""
Baseline 1: Vanilla PPO — no memory, no curiosity.

Usage:
    python -m train.train_ppo --task BinFill --timesteps 1_000_000 --wandb
"""

from __future__ import annotations

import argparse
import os
import platform

# Fix CUDA allocator fragmentation (SAPIEN Vulkan holds VRAM outside PyTorch's view)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from train.envs.rl_env import RobommeRLEnv
from train.models.encoder import RobommeCNNExtractor
from train.wandb_utils import add_wandb_args, init_wandb, finish_wandb


VECNORM_FILE = "vecnormalize.pkl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",       default="BinFill")
    p.add_argument("--timesteps",  type=int,   default=1_000_000)
    p.add_argument("--n_envs",     type=int,   default=4)
    p.add_argument("--n_steps",    type=int,   default=2048)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--n_epochs",   type=int,   default=4)
    p.add_argument("--target_kl",  type=float, default=0.02)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--gamma",      type=float, default=0.997)
    p.add_argument("--ent_coef",   type=float, default=0.01)
    p.add_argument("--outdir",     default="runs/ppo_v3")
    p.add_argument("--device",     default="auto")
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--resume",     default=None,
                   help="Path to a checkpoint .zip to resume from. "
                        "Use 'latest' to auto-pick the newest ckpt in <outdir>/ckpts.")
    add_wandb_args(p)
    return p.parse_args()


def _find_latest_ckpt(ckpt_dir):
    import glob
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.zip"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    # name pattern: ppo_<task>_<steps>_steps.zip
    return max(ckpts, key=lambda p: int(p.split("_")[-2]))


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    def make_env(rank):
        def _init():
            # Monitor injects the per-episode `episode` info dict that SB3
            # needs to log rollout/ep_rew_mean & ep_len_mean.
            return Monitor(RobommeRLEnv(env_id=args.task, seed=args.seed + rank))
        return _init

    # SubprocVecEnv crashes on Windows (ERROR_COMMITMENT_LIMIT shared mapping bug)
    VecEnvCls = DummyVecEnv if platform.system() == "Windows" else SubprocVecEnv
    vec_env = VecEnvCls([make_env(i) for i in range(args.n_envs)])

    # Reward normalization stabilizes PPO when reward magnitude is unknown.
    # Obs is left raw (already in fixed dtypes, images normalized by SB3).
    vecnorm_path = os.path.join(args.outdir, VECNORM_FILE)
    if args.resume and os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize stats from {vecnorm_path}")
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
    else:
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True,
                               clip_reward=10.0, gamma=args.gamma)

    policy_kwargs = dict(
        features_extractor_class=RobommeCNNExtractor,
        features_extractor_kwargs={},
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        # Tanh-squashed Gaussian: avoids the unbounded-Gaussian + env-clip
        # mismatch that drove std → 2.4 in v1/v2 runs. Requires use_sde=True
        # under SB3's policy code (see policies.py:519 assertion).
        squash_output=True,
    )

    if args.resume:
        ckpt = (_find_latest_ckpt(os.path.join(args.outdir, "ckpts"))
                if args.resume == "latest" else args.resume)
        print(f"Resuming from checkpoint: {ckpt}")
        model = PPO.load(
            ckpt,
            env=vec_env,
            device=args.device,
            tensorboard_log=os.path.join(args.outdir, "tb"),
        )
    else:
        model = PPO(
            policy="MultiInputPolicy",
            env=vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            target_kl=args.target_kl,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            use_sde=True,
            sde_sample_freq=4,
            policy_kwargs=policy_kwargs,
            device=args.device,
            verbose=1,
            tensorboard_log=os.path.join(args.outdir, "tb"),
            seed=args.seed,
        )

    ckpt_cb = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=os.path.join(args.outdir, "ckpts"),
        name_prefix=f"ppo_{args.task}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    callbacks = [ckpt_cb]
    wandb_cb = init_wandb(args, baseline="ppo", config=vars(args))
    if wandb_cb is not None:
        callbacks.append(wandb_cb)

    # When resuming, --timesteps is the absolute target; train only the remainder.
    remaining = args.timesteps
    if args.resume:
        remaining = args.timesteps - model.num_timesteps
        if remaining <= 0:
            print(f"Already at {model.num_timesteps} >= target {args.timesteps}; nothing to do.")
            vec_env.close()
            return
        print(f"Resuming at {model.num_timesteps} steps; training {remaining} more "
              f"to reach {args.timesteps}.")

    model.learn(
        total_timesteps=remaining,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=not args.resume,
    )

    save_path = os.path.join(args.outdir, f"ppo_{args.task}_final")
    model.save(save_path)
    vec_env.save(vecnorm_path)
    print(f"Model saved to {save_path}")
    print(f"VecNormalize stats saved to {vecnorm_path}")
    finish_wandb(args)
    vec_env.close()


if __name__ == "__main__":
    main()
