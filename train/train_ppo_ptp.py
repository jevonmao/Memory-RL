"""
Baseline 3: PPO + PTP Memory.

Architecture:
  - RobommeMemoryExtractor: CNN features + 2-layer transformer over K-step
    state-only history (action portion dropped from input — see plan D#4)
  - PTPHead: predicts past + future action tokens from the memory embedding
  - PTPCallback: runs auxiliary PTP loss after each rollout

The env provides history_state and history_action in the observation dict.
Memory resets at episode boundaries.

Usage:
    python -m train.train_ppo_ptp --task BinFill --timesteps 1_000_000 --wandb
"""

from __future__ import annotations

import argparse
import os
import platform

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from train.envs.rl_env import RobommeRLEnvWithMemory
from train.models.encoder import RobommeMemoryExtractor
from train.models.ptp_memory import PTPHead
from train.callbacks.ptp_callback import PTPCallback
from train.wandb_utils import add_wandb_args, init_wandb, finish_wandb


VECNORM_FILE = "vecnormalize.pkl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",        default="BinFill")
    p.add_argument("--timesteps",   type=int,   default=1_000_000)
    p.add_argument("--n_envs",      type=int,   default=4)
    p.add_argument("--n_steps",     type=int,   default=2048)
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--n_epochs",    type=int,   default=4)
    p.add_argument("--target_kl",   type=float, default=0.02)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--ptp_lr",      type=float, default=3e-4)
    p.add_argument("--ptp_weight",  type=float, default=0.1,
                   help="PTP loss multiplier")
    p.add_argument("--K",           type=int,   default=8,
                   help="History length for memory transformer")
    p.add_argument("--gamma",       type=float, default=0.997)
    p.add_argument("--ent_coef",    type=float, default=0.01)
    p.add_argument("--outdir",      default="runs/ppo_ptp_v3")
    p.add_argument("--device",      default="auto")
    p.add_argument("--seed",        type=int,   default=0)
    add_wandb_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    def make_env(rank):
        def _init():
            return Monitor(RobommeRLEnvWithMemory(
                env_id=args.task, seed=args.seed + rank, K=args.K
            ))
        return _init

    VecEnvCls = DummyVecEnv if platform.system() == "Windows" else SubprocVecEnv
    vec_env = VecEnvCls([make_env(i) for i in range(args.n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True,
                           clip_reward=10.0, gamma=args.gamma)

    policy_kwargs = dict(
        features_extractor_class=RobommeMemoryExtractor,
        features_extractor_kwargs=dict(
            memory_feat_dim=256,
            n_heads=4,
            n_layers=2,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        squash_output=True,
    )

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
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=os.path.join(args.outdir, "ckpts"),
        name_prefix=f"ppo_ptp_{args.task}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    callbacks = [ptp_cb, ckpt_cb]
    wandb_cb = init_wandb(args, baseline="ppo_ptp", config=vars(args))
    if wandb_cb is not None:
        callbacks.append(wandb_cb)

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    save_path = os.path.join(args.outdir, f"ppo_ptp_{args.task}_final")
    model.save(save_path)
    torch.save(ptp_head.state_dict(), save_path + "_ptphead.pt")
    vec_env.save(os.path.join(args.outdir, VECNORM_FILE))
    print(f"Model saved to {save_path}")
    finish_wandb(args)
    vec_env.close()


if __name__ == "__main__":
    main()
