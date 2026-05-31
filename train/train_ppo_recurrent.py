"""
Baseline B5: PPO with an LSTM (sb3-contrib RecurrentPPO).

The MIKASA-Robo paper (ICLR 2025) showed PPO-LSTM outperforms vanilla PPO
on memory-demanding manipulation tasks; this replaces our home-grown PTP
transformer (Baseline 3) with a battle-tested LSTM that SB3-contrib threads
hidden states through correctly across the rollout.

Policy: MultiInputLstmPolicy — a CNN feature-extractor (ours: 576-d shared
RobommeCNNExtractor) feeds into an LSTM whose hidden state is propagated
across consecutive env.step calls per env.

Usage:
    python -m train.train_ppo_recurrent --task PickXtimes --timesteps 1_000_000 --wandb
    python -m train.train_ppo_recurrent --resume latest --outdir runs/ppo_recurrent_v1 ...
"""

from __future__ import annotations

import argparse
import glob
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from train.envs.rl_env import RobommeRLEnv
from train.models.encoder import RobommeCNNExtractor
from train.wandb_utils import add_wandb_args, init_wandb, finish_wandb


VECNORM_FILE = "vecnormalize.pkl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",         default="PickXtimes")
    p.add_argument("--timesteps",    type=int,   default=1_000_000)
    p.add_argument("--n_envs",       type=int,   default=4)
    p.add_argument("--vec_env",      choices=["auto", "dummy", "subproc"], default="auto")
    # RecurrentPPO default n_steps=128 is tiny; bigger rollouts let the LSTM
    # propagate further before resets. Keep memory bounded.
    p.add_argument("--n_steps",      type=int,   default=512)
    # batch_size MUST be a multiple of (n_envs) for RecurrentPPO's sequence
    # batching. We choose 64 sequences (each n_steps long) by default.
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--n_epochs",     type=int,   default=4)
    p.add_argument("--target_kl",    type=float, default=0.05)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--gamma",        type=float, default=0.997)
    p.add_argument("--ent_coef",     type=float, default=0.01)
    p.add_argument("--lstm_hidden",  type=int,   default=256)
    p.add_argument("--lstm_layers",  type=int,   default=1)
    p.add_argument("--outdir",       default="runs/ppo_recurrent_v1")
    p.add_argument("--device",       default="auto")
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--resume",       default=None)
    add_wandb_args(p)
    return p.parse_args()


def _find_latest_ckpt(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, "*_steps.zip"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return max(ckpts, key=lambda p: int(p.rsplit("_", 2)[-2]))


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    def make_env(rank):
        def _init():
            return Monitor(RobommeRLEnv(env_id=args.task, seed=args.seed + rank))
        return _init

    if args.vec_env == "dummy" or args.n_envs == 1:
        vec_env = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
    else:
        vec_env = SubprocVecEnv(
            [make_env(i) for i in range(args.n_envs)],
            start_method="spawn",
        )
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True,
                           clip_reward=10.0, gamma=args.gamma)

    # Note: RecurrentPPO's MultiInputLstmPolicy does NOT accept squash_output
    # (it composes a categorical / DiagGaussian distribution itself). For our
    # continuous control we let the policy emit unbounded Gaussians and the
    # env wrapper clips to [-1, 1] (RobommeRLEnv.step does np.clip).
    policy_kwargs = dict(
        features_extractor_class=RobommeCNNExtractor,
        features_extractor_kwargs={},
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        lstm_hidden_size=args.lstm_hidden,
        n_lstm_layers=args.lstm_layers,
        # Share the LSTM between policy and value head for parameter efficiency
        # (we have 12M+ params already in the encoder; doubling LSTM is wasteful).
        shared_lstm=True,
        enable_critic_lstm=False,
    )

    if args.resume:
        ckpt = (_find_latest_ckpt(os.path.join(args.outdir, "ckpts"))
                if args.resume == "latest" else args.resume)
        print(f"Resuming from RecurrentPPO checkpoint: {ckpt}")
        model = RecurrentPPO.load(
            ckpt,
            env=vec_env,
            device=args.device,
            tensorboard_log=os.path.join(args.outdir, "tb"),
        )
    else:
        model = RecurrentPPO(
            policy="MultiInputLstmPolicy",
            env=vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            target_kl=args.target_kl,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            device=args.device,
            verbose=1,
            tensorboard_log=os.path.join(args.outdir, "tb"),
            seed=args.seed,
        )

    save_freq = max(100_000 // args.n_envs, 1)
    callbacks = [
        CheckpointCallback(
            save_freq=save_freq,
            save_path=os.path.join(args.outdir, "ckpts"),
            name_prefix=f"ppo_recurrent_{args.task}",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
    ]
    wandb_cb = init_wandb(args, baseline="ppo_recurrent", config=vars(args))
    if wandb_cb is not None:
        callbacks.append(wandb_cb)

    remaining = args.timesteps
    if args.resume:
        remaining = args.timesteps - model.num_timesteps
        if remaining <= 0:
            print(f"Already at {model.num_timesteps} >= target {args.timesteps}; nothing to do.")
            vec_env.close()
            return
        print(f"Resuming at {model.num_timesteps} steps; training {remaining} more.")

    model.learn(
        total_timesteps=remaining,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=not args.resume,
    )

    save_path = os.path.join(args.outdir, f"ppo_recurrent_{args.task}_final")
    model.save(save_path)
    vec_env.save(os.path.join(args.outdir, VECNORM_FILE))
    print(f"Model saved to {save_path}")
    finish_wandb(args)
    vec_env.close()


if __name__ == "__main__":
    main()
