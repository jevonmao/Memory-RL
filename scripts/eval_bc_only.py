"""Eval a BC-only policy (no further RL training) on PickXtimes.

Builds a fresh RecurrentPPO/PPO model, loads the BC state_dict, then runs
evaluate_trained logic on the test split.

Usage:
    python -m scripts.eval_bc_only \\
        --bc runs/bc/pickxtimes/bc_policy_recurrent.pt \\
        --recurrent --n_eval 20
"""

from __future__ import annotations

import argparse
import os
os.environ.setdefault("ROBOMME_OBS_MODE", "rgb")
os.environ.setdefault("ROBOMME_SIM_BACKEND", "physx_cpu")

import json
import numpy as np
import torch

from gymnasium import spaces
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from train.envs.rl_env import IMG_H, IMG_W, ACTION_DIM
from train.models.encoder import RobommeCNNExtractor


def make_obs_action_space():
    obs_space = spaces.Dict({
        "front_rgb":   spaces.Box(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8),
        "wrist_rgb":   spaces.Box(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8),
        "joint_state": spaces.Box(-np.inf, np.inf, (7,),  dtype=np.float32),
        "eef_state":   spaces.Box(-np.inf, np.inf, (6,),  dtype=np.float32),
        "gripper":     spaces.Box(-1., 1.,           (2,),  dtype=np.float32),
    })
    act_space = spaces.Box(-1., 1., (ACTION_DIM,), dtype=np.float32)
    return obs_space, act_space


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bc", required=True)
    p.add_argument("--recurrent", action="store_true")
    p.add_argument("--lstm_hidden", type=int, default=256)
    p.add_argument("--lstm_layers", type=int, default=1)
    p.add_argument("--n_eval", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=1500)
    p.add_argument("--task", default="PickXtimes")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--outfile", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    obs_space, act_space = make_obs_action_space()

    class _FakeEnv(gym.Env):
        def __init__(self): self.observation_space = obs_space; self.action_space = act_space
        def reset(self, *, seed=None, options=None): return obs_space.sample(), {}
        def step(self, a): return obs_space.sample(), 0.0, True, False, {}

    vec = DummyVecEnv([lambda: _FakeEnv()])

    if args.recurrent:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO(
            "MultiInputLstmPolicy", vec,
            policy_kwargs=dict(
                features_extractor_class=RobommeCNNExtractor,
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                lstm_hidden_size=args.lstm_hidden, n_lstm_layers=args.lstm_layers,
                shared_lstm=True, enable_critic_lstm=False,
            ),
            device=args.device, verbose=0, seed=0,
            n_steps=16, batch_size=16, n_epochs=1,
        )
    else:
        model = PPO(
            "MultiInputPolicy", vec,
            policy_kwargs=dict(
                features_extractor_class=RobommeCNNExtractor,
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                squash_output=True,
            ),
            device=args.device, verbose=0, seed=0,
            use_sde=True, sde_sample_freq=4,
            n_steps=16, batch_size=16, n_epochs=1,
        )
    bc_sd = torch.load(args.bc, map_location="cpu", weights_only=True)
    missing, unexpected = model.policy.load_state_dict(bc_sd, strict=False)
    print(f"[bc-eval] loaded BC weights from {args.bc}; missing={len(missing)} unexpected={len(unexpected)}")
    model.policy.set_training_mode(False)

    # Run the same eval logic as evaluate_trained.py but inline.
    from train.evaluate_trained import evaluate_task
    baseline = "recurrent" if args.recurrent else "ppo"
    result = evaluate_task(model, args.task, args.n_eval, baseline, K=8, max_steps=args.max_steps)
    print(f"\n=== {args.task} BC-only eval ===")
    print(json.dumps(result, indent=2))
    if args.outfile:
        os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
        with open(args.outfile, "w") as f:
            json.dump({"results": [result], "aggregate_sr": result["success_rate"]}, f, indent=2)
        print(f"saved -> {args.outfile}")


if __name__ == "__main__":
    main()
