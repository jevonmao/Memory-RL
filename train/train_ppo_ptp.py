"""
Baseline 3: PPO + PTP Memory.

Architecture:
  - RobommeMemoryExtractor: CNN features + 2-layer transformer over K-step history
  - PTPHead: predicts past + future action tokens from the memory embedding
  - PTPCallback: runs auxiliary PTP loss after each rollout

The env provides history_state and history_action in the observation dict.
Memory resets at episode boundaries.

Usage:
    python -m train.train_ppo_ptp --task BinFill --timesteps 1_000_000
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

from train.envs.rl_env import RobommeRLEnvWithMemory
from train.models.encoder import RobommeMemoryExtractor
from train.models.ptp_memory import PTPHead
from train.callbacks.ptp_callback import PTPCallback


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",        default="BinFill")
    p.add_argument("--timesteps",   type=int,   default=1_000_000)
    p.add_argument("--n_envs",      type=int,   default=4)
    p.add_argument("--n_steps",     type=int,   default=512)
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--ptp_lr",      type=float, default=3e-4)
    p.add_argument("--ptp_weight",  type=float, default=0.1,
                   help="PTP loss multiplier")
    p.add_argument("--K",           type=int,   default=8,
                   help="History length for memory transformer")
    p.add_argument("--gamma",       type=float, default=0.99)
    p.add_argument("--ent_coef",    type=float, default=0.01)
    p.add_argument("--outdir",      default="runs/ppo_ptp")
    p.add_argument("--device",      default="auto")
    p.add_argument("--seed",        type=int,   default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    def make_env(rank):
        def _init():
            return RobommeRLEnvWithMemory(
                env_id=args.task, seed=args.seed + rank, K=args.K
            )
        return _init

    VecEnvCls = DummyVecEnv if platform.system() == "Windows" else SubprocVecEnv
    vec_env = VecEnvCls([make_env(i) for i in range(args.n_envs)])

    policy_kwargs = dict(
        features_extractor_class=RobommeMemoryExtractor,
        features_extractor_kwargs=dict(
            memory_feat_dim=256,
            n_heads=4,
            n_layers=2,
        ),
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

    # PTP head: same memory_dim as MemoryTransformer out_dim (256)
    ptp_head = PTPHead(memory_dim=256, action_dim=8, K=args.K)

    ptp_cb = PTPCallback(
        memory_transformer=model.policy.features_extractor.memory_transformer,
        ptp_head=ptp_head,
        action_dim=8,
        K=args.K,
        lr=args.ptp_lr,
        ptp_weight=args.ptp_weight,
        device=args.device,
        verbose=1,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=os.path.join(args.outdir, "ckpts"),
        name_prefix=f"ppo_ptp_{args.task}",
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[ptp_cb, ckpt_cb],
        progress_bar=True,
    )

    save_path = os.path.join(args.outdir, f"ppo_ptp_{args.task}_final")
    model.save(save_path)
    torch.save(ptp_head.state_dict(), save_path + "_ptphead.pt")
    print(f"Model saved to {save_path}")
    vec_env.close()


if __name__ == "__main__":
    main()
