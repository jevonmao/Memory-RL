"""
Baseline 2: PPO + Intrinsic Curiosity Module (ICM).

The ICM is owned by the algorithm itself (`PPOWithICM`) rather than a callback,
because SB3 computes advantages before any callback fires; a callback-based
intrinsic injection is silently a no-op for the policy. See
train/algos/ppo_with_icm.py for the full explanation.

Usage:
    python -m train.train_ppo_icm --task BinFill --timesteps 1_000_000 --wandb
"""

from __future__ import annotations

import argparse
import os
import platform

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from train.algos.ppo_with_icm import PPOWithICM
from train.envs.rl_env import RobommeRLEnv
from train.models.encoder import RobommeCNNExtractor
from train.models.icm import ICM
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
    p.add_argument("--target_kl",  type=float, default=0.05)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--icm_lr",     type=float, default=3e-4)
    p.add_argument("--eta",        type=float, default=0.01,
                   help="Intrinsic reward scale")
    p.add_argument("--icm_beta",   type=float, default=0.2,
                   help="ICM loss weighting (β*L_fwd + (1-β)*L_inv)")
    p.add_argument("--gamma",      type=float, default=0.997)
    p.add_argument("--ent_coef",   type=float, default=0.01)
    p.add_argument("--outdir",     default="runs/ppo_icm_v3")
    p.add_argument("--device",     default="auto")
    p.add_argument("--seed",       type=int,   default=0)
    add_wandb_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    def make_env(rank):
        def _init():
            return Monitor(RobommeRLEnv(env_id=args.task, seed=args.seed + rank))
        return _init

    VecEnvCls = DummyVecEnv if platform.system() == "Windows" else SubprocVecEnv
    vec_env = VecEnvCls([make_env(i) for i in range(args.n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True,
                           clip_reward=10.0, gamma=args.gamma)

    policy_kwargs = dict(
        features_extractor_class=RobommeCNNExtractor,
        features_extractor_kwargs={},
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        squash_output=True,
    )

    icm = ICM(feat_dim=576, action_dim=8, hidden=256)

    model = PPOWithICM(
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
        # ICM-specific
        icm=icm,
        eta=args.eta,
        icm_lr=args.icm_lr,
        icm_beta=args.icm_beta,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=os.path.join(args.outdir, "ckpts"),
        name_prefix=f"ppo_icm_{args.task}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    callbacks = [ckpt_cb]
    wandb_cb = init_wandb(args, baseline="ppo_icm", config=vars(args))
    if wandb_cb is not None:
        callbacks.append(wandb_cb)

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    save_path = os.path.join(args.outdir, f"ppo_icm_{args.task}_final")
    model.save(save_path)
    # Save ICM separately so it can be inspected / reloaded
    torch.save(model.icm.state_dict(), save_path + "_icm.pt")
    vec_env.save(os.path.join(args.outdir, VECNORM_FILE))
    print(f"Model saved to {save_path}")
    finish_wandb(args)
    vec_env.close()


if __name__ == "__main__":
    main()
