"""
Baseline B6 (HEADLINE): RecurrentPPO + RND.

Combines:
  * sb3-contrib RecurrentPPO  — LSTM memory across the rollout
  * RND                        — non-saturating intrinsic curiosity

This is the stack the research plan identifies as the most promising for
PickXtimes: counting requires *memory* (reactive policies cap at 1/N) and
sparse extrinsic reward requires *curiosity* to find the first successful
trajectory.

Usage:
    python -m train.train_ppo_recurrent_rnd --task PickXtimes \\
        --timesteps 2_000_000 --n_envs 8 --wandb
"""

from __future__ import annotations

import argparse
import glob
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from train.algos.recurrent_ppo_with_rnd import RecurrentPPOWithRND
from train.envs.rl_env import RobommeRLEnv
from train.models.encoder import RobommeCNNExtractor
from train.models.rnd import RND
from train.wandb_utils import add_wandb_args, init_wandb, finish_wandb


VECNORM_FILE = "vecnormalize.pkl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",         default="PickXtimes")
    p.add_argument("--timesteps",    type=int,   default=2_000_000)
    p.add_argument("--n_envs",       type=int,   default=4)
    p.add_argument("--vec_env",      choices=["auto", "dummy", "subproc"], default="auto")
    p.add_argument("--n_steps",      type=int,   default=512)
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--n_epochs",     type=int,   default=4)
    p.add_argument("--target_kl",    type=float, default=0.05)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--rnd_lr",       type=float, default=1e-4)
    p.add_argument("--eta",          type=float, default=1.0)
    p.add_argument("--rnd_gamma",    type=float, default=0.99)
    p.add_argument("--gamma",        type=float, default=0.997)
    p.add_argument("--ent_coef",     type=float, default=0.01)
    p.add_argument("--lstm_hidden",  type=int,   default=256)
    p.add_argument("--lstm_layers",  type=int,   default=1)
    p.add_argument("--outdir",       default="runs/ppo_recurrent_rnd_v1")
    p.add_argument("--device",       default="auto")
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--resume",       default=None)
    p.add_argument("--bc_warmstart", default=None,
                   help="Optional .pt produced by scripts/bc_pretrain.py — "
                        "loads into the policy's state_dict before training.")
    p.add_argument("--init_log_std", type=float, default=None,
                   help="If set, overrides policy.log_std after weight load. "
                        "Default policy init is log_std=0 (std=1.0); pass "
                        "-1.0 (std=0.37) when warm-starting from BC so the "
                        "policy actually executes BC-like actions early on.")
    p.add_argument("--bc_aux_dataset", default=None,
                   help="Optional path to bc_dataset_pickxtimes.pt. Adds an "
                        "auxiliary BC loss term to each PPO update.")
    p.add_argument("--bc_aux_coef", type=float, default=0.0,
                   help="Weight of the auxiliary BC loss. Pass ~0.5 to "
                        "balance against PPO's policy gradient signal.")
    p.add_argument("--bc_aux_n_transitions", type=int, default=20000,
                   help="Subset size of the BC dataset to load into memory. "
                        "Full 80-episode dataset is 5.9 GB; 20k transitions "
                        "≈ 1.5 GB and is enough to regularise the policy.")
    add_wandb_args(p)
    return p.parse_args()


def _find_latest_ckpt(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, "*_steps.zip"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return max(ckpts, key=lambda p: int(p.rsplit("_", 2)[-2]))


class RNDCheckpointCallback(BaseCallback):
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
        lstm_hidden_size=args.lstm_hidden,
        n_lstm_layers=args.lstm_layers,
        shared_lstm=True,
        enable_critic_lstm=False,
    )

    rnd = RND(feat_dim=576, out_dim=128, hidden=256)

    # Optionally load a slice of the BC dataset into memory for the aux loss.
    # Sample whole episodes from per-episode .npz files to avoid loading the
    # full 5.9 GB .pt blob (which OOMs at 48G slurm allocation).
    bc_obs = bc_actions = None
    if args.bc_aux_dataset and args.bc_aux_coef > 0:
        import glob
        ep_dir = os.path.join(os.path.dirname(args.bc_aux_dataset), "episodes")
        if os.path.isdir(ep_dir):
            print(f"Loading BC aux dataset from per-episode .npz files in {ep_dir}")
            files = sorted(glob.glob(os.path.join(ep_dir, "ep_*.npz")))
            rng = np.random.default_rng(0)
            rng.shuffle(files)
            acc_obs: dict[str, list] = {}
            acc_act: list = []
            total = 0
            for f in files:
                d = np.load(f)
                n = d["actions"].shape[0]
                if total + n > args.bc_aux_n_transitions and acc_act:
                    d.close()
                    break
                acc_act.append(d["actions"].astype(np.float32))
                for k in d.files:
                    if k == "actions":
                        continue
                    acc_obs.setdefault(k, []).append(d[k])
                total += n
                d.close()
            n_episodes_loaded = len(acc_act)
            bc_actions = np.concatenate(acc_act, axis=0)
            bc_obs     = {k: np.concatenate(v, axis=0) for k, v in acc_obs.items()}
            del acc_act, acc_obs
            print(f"  loaded {bc_actions.shape[0]} transitions from {n_episodes_loaded} episodes "
                  f"({sum(v.nbytes for v in bc_obs.values()) / 1e6:.0f} MB obs)")
        else:
            print(f"  WARNING: no per-episode dir at {ep_dir}; falling back to full .pt load (will OOM if >32 GB allocation)")
            bc_data = torch.load(args.bc_aux_dataset, map_location="cpu", weights_only=False)
            n_avail = bc_data["actions"].shape[0]
            n_take  = min(args.bc_aux_n_transitions, n_avail)
            rng = np.random.default_rng(0)
            keep = rng.choice(n_avail, n_take, replace=False)
            bc_obs = {k: v[keep] for k, v in bc_data["obs"].items()}
            bc_actions = bc_data["actions"][keep].astype(np.float32)
            del bc_data

    model = RecurrentPPOWithRND(
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
        rnd=rnd,
        eta=args.eta,
        rnd_lr=args.rnd_lr,
        rnd_gamma=args.rnd_gamma,
        bc_obs=bc_obs,
        bc_actions=bc_actions,
        bc_aux_coef=args.bc_aux_coef,
    )

    if args.bc_warmstart and not args.resume:
        print(f"Loading BC warm-start weights from {args.bc_warmstart}")
        bc_sd = torch.load(args.bc_warmstart, map_location="cpu", weights_only=True)
        missing, unexpected = model.policy.load_state_dict(bc_sd, strict=False)
        print(f"  loaded; missing={len(missing)}  unexpected={len(unexpected)}")

    if args.init_log_std is not None and not args.resume:
        # Manually override the policy's log_std parameter. Default is 0
        # (std=1.0); with BC warm-start we want a smaller exploration noise
        # so the policy actually executes the near-expert mean.
        with torch.no_grad():
            new_log_std = torch.full_like(model.policy.log_std, args.init_log_std)
            model.policy.log_std.data.copy_(new_log_std)
        print(f"  overrode policy.log_std = {args.init_log_std}  (std = {np.exp(args.init_log_std):.3f})")

    if args.resume:
        ckpt = (_find_latest_ckpt(os.path.join(args.outdir, "ckpts"))
                if args.resume == "latest" else args.resume)
        print(f"Resuming from RecurrentPPO checkpoint: {ckpt}")
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

    save_freq = max(100_000 // args.n_envs, 1)
    callbacks = [
        CheckpointCallback(
            save_freq=save_freq,
            save_path=os.path.join(args.outdir, "ckpts"),
            name_prefix=f"ppo_recurrent_rnd_{args.task}",
            save_replay_buffer=False,
            save_vecnormalize=False,
        ),
        RNDCheckpointCallback(
            save_freq=save_freq,
            save_path=os.path.join(args.outdir, "ckpts"),
            name_prefix=f"ppo_recurrent_rnd_{args.task}",
        ),
    ]
    wandb_cb = init_wandb(args, baseline="ppo_recurrent_rnd", config=vars(args))
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

    save_path = os.path.join(args.outdir, f"ppo_recurrent_rnd_{args.task}_final")
    model.save(save_path)
    torch.save(model.rnd.state_dict(), save_path + "_rnd.pt")
    vec_env.save(os.path.join(args.outdir, VECNORM_FILE))
    print(f"Model saved to {save_path}")
    finish_wandb(args)
    vec_env.close()


if __name__ == "__main__":
    main()
