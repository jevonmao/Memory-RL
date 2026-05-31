"""
Baseline B4: PPO + Random Network Distillation (RND).

Same wiring as train_ppo_icm.py but swaps the curiosity module:
  * ICM saturates fast (forward-model error decays as the model fits)
  * RND novelty is bounded only by what the predictor has seen, so it does
    not saturate on the *unseen* states we need exploration to reach.

Usage:
    python -m train.train_ppo_rnd --task PickXtimes --timesteps 1_000_000 --wandb
    python -m train.train_ppo_rnd --resume latest --outdir runs/ppo_rnd_v1 ...
"""

from __future__ import annotations

import argparse
import glob
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from train.algos.ppo_with_rnd import PPOWithRND
from train.envs.rl_env import RobommeRLEnv
from train.models.encoder import RobommeCNNExtractor
from train.models.rnd import RND
from train.wandb_utils import add_wandb_args, init_wandb, finish_wandb


VECNORM_FILE = "vecnormalize.pkl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",       default="PickXtimes")
    p.add_argument("--timesteps",  type=int,   default=1_000_000)
    p.add_argument("--n_envs",     type=int,   default=4)
    p.add_argument("--vec_env",    choices=["auto", "dummy", "subproc"], default="auto")
    p.add_argument("--n_steps",    type=int,   default=2048)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--n_epochs",   type=int,   default=4)
    p.add_argument("--target_kl",  type=float, default=0.05)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--rnd_lr",     type=float, default=1e-4)
    p.add_argument("--eta",        type=float, default=1.0,
                   help="Intrinsic reward scale (RND auto-normalises so ~1 works).")
    p.add_argument("--rnd_gamma",  type=float, default=0.99,
                   help="Discount for intrinsic-return RMS (decoupled from policy γ).")
    p.add_argument("--gamma",      type=float, default=0.997)
    p.add_argument("--ent_coef",   type=float, default=0.01)
    p.add_argument("--outdir",     default="runs/ppo_rnd_v1")
    p.add_argument("--device",     default="auto")
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--resume",     default=None,
                   help="Path to a PPO checkpoint .zip, or 'latest'.")
    add_wandb_args(p)
    return p.parse_args()


def _find_latest_ckpt(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, "*_steps.zip"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return max(ckpts, key=lambda p: int(p.rsplit("_", 2)[-2]))


class RNDCheckpointCallback(BaseCallback):
    """Snapshot the RND module alongside the PPO .zip checkpoints."""

    def __init__(self, save_freq: int, save_path: str, name_prefix: str):
        super().__init__(verbose=0)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True
        path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps_rnd.pt")
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.model.rnd.state_dict(), path)
        return True


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

    policy_kwargs = dict(
        features_extractor_class=RobommeCNNExtractor,
        features_extractor_kwargs={},
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        squash_output=True,
    )

    rnd = RND(feat_dim=576, out_dim=128, hidden=256)

    model = PPOWithRND(
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
        rnd=rnd,
        eta=args.eta,
        rnd_lr=args.rnd_lr,
        rnd_gamma=args.rnd_gamma,
    )

    if args.resume:
        ckpt = (_find_latest_ckpt(os.path.join(args.outdir, "ckpts"))
                if args.resume == "latest" else args.resume)
        print(f"Resuming from PPO checkpoint: {ckpt}")
        model.set_parameters(ckpt, exact_match=True, device=args.device)
        try:
            resumed_steps = int(os.path.basename(ckpt).rsplit("_", 2)[-2])
            model.num_timesteps = resumed_steps
            print(f"  resumed at num_timesteps={resumed_steps}")
        except (ValueError, IndexError):
            print("  warning: could not parse step count from filename")
        rnd_pt = ckpt.replace(".zip", "_rnd.pt")
        if not os.path.exists(rnd_pt):
            rnd_pt = ckpt[:-len("_steps.zip")] + "_rnd.pt"
        if os.path.exists(rnd_pt):
            print(f"  loading RND weights from {rnd_pt}")
            model.rnd.load_state_dict(torch.load(rnd_pt, map_location=args.device))
        else:
            print(f"  WARNING: no RND checkpoint at {rnd_pt} — RND starts from scratch")

    save_freq = max(100_000 // args.n_envs, 1)
    callbacks = [
        CheckpointCallback(
            save_freq=save_freq,
            save_path=os.path.join(args.outdir, "ckpts"),
            name_prefix=f"ppo_rnd_{args.task}",
            save_replay_buffer=False,
            save_vecnormalize=False,
        ),
        RNDCheckpointCallback(
            save_freq=save_freq,
            save_path=os.path.join(args.outdir, "ckpts"),
            name_prefix=f"ppo_rnd_{args.task}",
        ),
    ]
    wandb_cb = init_wandb(args, baseline="ppo_rnd", config=vars(args))
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

    save_path = os.path.join(args.outdir, f"ppo_rnd_{args.task}_final")
    model.save(save_path)
    torch.save(model.rnd.state_dict(), save_path + "_rnd.pt")
    vec_env.save(os.path.join(args.outdir, VECNORM_FILE))
    print(f"Model saved to {save_path}")
    finish_wandb(args)
    vec_env.close()


if __name__ == "__main__":
    main()
