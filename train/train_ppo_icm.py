"""
Baseline 2: PPO + Intrinsic Curiosity Module (ICM).

ICM runs as an SB3 callback that:
  - Augments extrinsic rewards with η * forward-prediction-error
  - Trains forward + inverse models on (s_t, a_t, s_{t+1}) transitions

Usage:
    python -m train.train_ppo_icm --task BinFill --timesteps 1_000_000
"""

from __future__ import annotations

import argparse
import os
import platform

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from train.envs.rl_env import RobommeRLEnv
from train.models.encoder import RobommeCNNExtractor
from train.models.icm import ICM
from train.callbacks.icm_callback import ICMCallback


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",       default="BinFill")
    p.add_argument("--timesteps",  type=int,   default=1_000_000)
    p.add_argument("--n_envs",     type=int,   default=4)
    p.add_argument("--n_steps",    type=int,   default=512)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--icm_lr",     type=float, default=3e-4)
    p.add_argument("--eta",        type=float, default=0.01,
                   help="Intrinsic reward scale")
    p.add_argument("--icm_beta",   type=float, default=0.2,
                   help="ICM loss weighting (β*L_fwd + (1-β)*L_inv)")
    p.add_argument("--gamma",      type=float, default=0.99)
    p.add_argument("--ent_coef",   type=float, default=0.01)
    p.add_argument("--outdir",     default="runs/ppo_icm")
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

    # ICM shares the same feature extractor as the policy
    extractor = model.policy.features_extractor
    icm       = ICM(feat_dim=576, action_dim=8, hidden=256)

    icm_cb = ICMCallback(
        icm=icm,
        extractor=extractor,
        eta=args.eta,
        lr=args.icm_lr,
        beta=args.icm_beta,
        device=args.device,
        verbose=1,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=os.path.join(args.outdir, "ckpts"),
        name_prefix=f"ppo_icm_{args.task}",
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[icm_cb, ckpt_cb],
        progress_bar=True,
    )

    save_path = os.path.join(args.outdir, f"ppo_icm_{args.task}_final")
    model.save(save_path)
    # Save ICM separately so it can be inspected / reloaded
    torch.save(icm.state_dict(), save_path + "_icm.pt")
    print(f"Model saved to {save_path}")
    vec_env.close()


if __name__ == "__main__":
    main()
