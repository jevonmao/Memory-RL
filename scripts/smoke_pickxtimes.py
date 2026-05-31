"""Smoke test: build PickXtimes env, instantiate every new training stack,
run one PPO update, save/load each checkpoint type. Intended for the local
CPU/low-VRAM machine — exercises the wiring without needing real throughput.

Usage:
    python -m scripts.smoke_pickxtimes [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

# Avoid wandb during the smoke test
os.environ["WANDB_MODE"] = "disabled"


def _make_vec_env(task: str, n_envs: int = 1):
    from train.envs.rl_env import RobommeRLEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def make_env(rank):
        def _init():
            return Monitor(RobommeRLEnv(env_id=task, seed=rank, max_steps=150))
        return _init

    vec = DummyVecEnv([make_env(i) for i in range(n_envs)])
    vec = VecNormalize(vec, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99)
    return vec


def smoke_reward(task: str = "PickXtimes") -> None:
    print(f"\n[smoke reward] {task} ...")
    from train.rewards import make_reward
    from train.envs.rl_env import RobommeRLEnv

    rfn = make_reward(task)
    assert rfn is not None, f"no shaped reward registered for {task}"

    env = RobommeRLEnv(env_id=task, seed=0, max_steps=60)
    obs, info = env.reset()
    rfn.reset(env._env.unwrapped)
    total = 0.0
    for _ in range(20):
        a = env.action_space.sample().astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    env.close()
    print(f"  20 random steps; sum reward = {total:.3f}  (nonzero+finite expected)")
    assert np.isfinite(total)


def smoke_ppo_rnd(device: str, outdir: str) -> None:
    print(f"\n[smoke PPOWithRND] device={device} ...")
    from train.algos.ppo_with_rnd import PPOWithRND
    from train.models.encoder import RobommeCNNExtractor
    from train.models.rnd import RND

    vec_env = _make_vec_env("PickXtimes", n_envs=1)
    rnd = RND(feat_dim=576, out_dim=64, hidden=128)  # smaller for smoke
    policy_kwargs = dict(
        features_extractor_class=RobommeCNNExtractor,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        squash_output=True,
    )
    model = PPOWithRND(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        gamma=0.99,
        ent_coef=0.0,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=0,
        seed=0,
        rnd=rnd,
        eta=1.0,
        rnd_lr=1e-4,
    )
    print("  learn(64) ...")
    model.learn(total_timesteps=64, progress_bar=False)
    save_path = os.path.join(outdir, "smoke_ppo_rnd")
    model.save(save_path)
    torch.save(model.rnd.state_dict(), save_path + "_rnd.pt")
    vec_env.close()
    print(f"  saved -> {save_path}.zip + {save_path}_rnd.pt")


def smoke_recurrent_ppo_rnd(device: str, outdir: str) -> None:
    print(f"\n[smoke RecurrentPPOWithRND] device={device} ...")
    from train.algos.recurrent_ppo_with_rnd import RecurrentPPOWithRND
    from train.models.encoder import RobommeCNNExtractor
    from train.models.rnd import RND

    vec_env = _make_vec_env("PickXtimes", n_envs=1)
    rnd = RND(feat_dim=576, out_dim=64, hidden=128)
    policy_kwargs = dict(
        features_extractor_class=RobommeCNNExtractor,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        lstm_hidden_size=64,
        n_lstm_layers=1,
        shared_lstm=True,
        enable_critic_lstm=False,
    )
    model = RecurrentPPOWithRND(
        policy="MultiInputLstmPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        gamma=0.99,
        ent_coef=0.0,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=0,
        seed=0,
        rnd=rnd,
        eta=1.0,
        rnd_lr=1e-4,
    )
    print("  learn(64) ...")
    model.learn(total_timesteps=64, progress_bar=False)
    save_path = os.path.join(outdir, "smoke_recurrent_rnd")
    model.save(save_path)
    torch.save(model.rnd.state_dict(), save_path + "_rnd.pt")
    vec_env.close()
    print(f"  saved -> {save_path}.zip")


def smoke_recurrent_vanilla(device: str, outdir: str) -> None:
    print(f"\n[smoke RecurrentPPO vanilla] device={device} ...")
    from sb3_contrib import RecurrentPPO
    from train.models.encoder import RobommeCNNExtractor

    vec_env = _make_vec_env("PickXtimes", n_envs=1)
    policy_kwargs = dict(
        features_extractor_class=RobommeCNNExtractor,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        lstm_hidden_size=64,
        n_lstm_layers=1,
        shared_lstm=True,
        enable_critic_lstm=False,
    )
    model = RecurrentPPO(
        policy="MultiInputLstmPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        gamma=0.99,
        ent_coef=0.0,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=0,
        seed=0,
    )
    print("  learn(64) ...")
    model.learn(total_timesteps=64, progress_bar=False)
    save_path = os.path.join(outdir, "smoke_recurrent")
    model.save(save_path)
    vec_env.close()
    print(f"  saved -> {save_path}.zip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda. Default cpu — local 4090 has other workloads.")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Skip individual checks: reward, ppo_rnd, recurrent, recurrent_rnd")
    args = ap.parse_args()

    outdir = tempfile.mkdtemp(prefix="robomme_smoke_")
    print(f"smoke artifacts dir: {outdir}")

    try:
        if "reward" not in args.skip:
            smoke_reward()
        if "ppo_rnd" not in args.skip:
            smoke_ppo_rnd(args.device, outdir)
        if "recurrent" not in args.skip:
            smoke_recurrent_vanilla(args.device, outdir)
        if "recurrent_rnd" not in args.skip:
            smoke_recurrent_ppo_rnd(args.device, outdir)
        print("\nALL SMOKE CHECKS PASSED.")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


if __name__ == "__main__":
    main()
