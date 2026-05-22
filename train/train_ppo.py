"""
Baseline 1: Vanilla PPO — no memory, no curiosity.

Usage:
    python -m train.train_ppo --task BinFill --timesteps 1_000_000
"""

from __future__ import annotations

import argparse
import os
import platform

# Fix CUDA allocator fragmentation (SAPIEN Vulkan holds VRAM outside PyTorch's view)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from train.envs.rl_env import RobommeRLEnv
from train.models.encoder import RobommeCNNExtractor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",       default="BinFill")
    p.add_argument("--timesteps",  type=int,   default=1_000_000)
    p.add_argument("--n_envs",     type=int,   default=4)
    p.add_argument("--n_steps",    type=int,   default=512)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--gamma",      type=float, default=0.99)
    p.add_argument("--ent_coef",   type=float, default=0.01)
    p.add_argument("--outdir",     default="runs/ppo")
    p.add_argument("--device",     default="auto")
    p.add_argument("--seed",       type=int,   default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    def make_env(rank):
        def _init():
            return RobommeRLEnv(env_id=args.task, seed=args.seed + rank)
        return _init

    # SubprocVecEnv crashes on Windows (ERROR_COMMITMENT_LIMIT shared mapping bug)
    VecEnvCls = DummyVecEnv if platform.system() == "Windows" else SubprocVecEnv
    vec_env = VecEnvCls([make_env(i) for i in range(args.n_envs)])

    policy_kwargs = dict(
        features_extractor_class=RobommeCNNExtractor,
        features_extractor_kwargs={},
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
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

    model.learn(
        total_timesteps=args.timesteps,
        callback=[ckpt_cb],
        progress_bar=True,
    )

    save_path = os.path.join(args.outdir, f"ppo_{args.task}_final")
    model.save(save_path)
    print(f"Model saved to {save_path}")
    vec_env.close()


if __name__ == "__main__":
    main()
