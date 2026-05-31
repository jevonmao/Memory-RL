"""Eval BC policy on the TRAIN split (same distribution as BC training data).

If BC alone produces >0 SR here, the model learned the expert. If 0 SR even
on train split, BC compounding errors are killing it and we need DAgger or
sequential BC training.

Usage:
    python -m scripts.eval_bc_only_train --bc runs/bc/pickxtimes/bc_policy_recurrent.pt --recurrent --n_eval 20
"""

from __future__ import annotations

import argparse, os, json
os.environ.setdefault("ROBOMME_OBS_MODE", "rgb")
os.environ.setdefault("ROBOMME_SIM_BACKEND", "physx_cpu")

import numpy as np
import torch
from gymnasium import spaces
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from robomme.env_record_wrapper import BenchmarkEnvBuilder

from train.envs.rl_env import IMG_H, IMG_W, ACTION_DIM
from train.models.encoder import RobommeCNNExtractor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bc", required=True)
    p.add_argument("--recurrent", action="store_true")
    p.add_argument("--n_eval", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=1500)
    p.add_argument("--task", default="PickXtimes")
    p.add_argument("--dataset", default="train")
    p.add_argument("--outfile", default=None)
    return p.parse_args()


def make_spaces():
    obs_space = spaces.Dict({
        "front_rgb":   spaces.Box(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8),
        "wrist_rgb":   spaces.Box(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8),
        "joint_state": spaces.Box(-np.inf, np.inf, (7,),  dtype=np.float32),
        "eef_state":   spaces.Box(-np.inf, np.inf, (6,),  dtype=np.float32),
        "gripper":     spaces.Box(-1., 1.,           (2,),  dtype=np.float32),
    })
    act_space = spaces.Box(-1., 1., (ACTION_DIM,), dtype=np.float32)
    return obs_space, act_space


def main():
    args = parse_args()
    obs_space, act_space = make_spaces()
    class FE(gym.Env):
        def __init__(self): self.observation_space = obs_space; self.action_space = act_space
        def reset(self, *, seed=None, options=None): return obs_space.sample(), {}
        def step(self, a): return obs_space.sample(), 0.0, True, False, {}
    vec = DummyVecEnv([lambda: FE()])

    if args.recurrent:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO("MultiInputLstmPolicy", vec,
            policy_kwargs=dict(features_extractor_class=RobommeCNNExtractor,
                               net_arch=dict(pi=[256,256], vf=[256,256]),
                               lstm_hidden_size=256, n_lstm_layers=1,
                               shared_lstm=True, enable_critic_lstm=False),
            device="cuda" if torch.cuda.is_available() else "cpu", verbose=0,
            n_steps=16, batch_size=16, n_epochs=1)
    else:
        model = PPO("MultiInputPolicy", vec,
            policy_kwargs=dict(features_extractor_class=RobommeCNNExtractor,
                               net_arch=dict(pi=[256,256], vf=[256,256]),
                               squash_output=True),
            device="cuda" if torch.cuda.is_available() else "cpu", verbose=0,
            use_sde=True, sde_sample_freq=4,
            n_steps=16, batch_size=16, n_epochs=1)
    sd = torch.load(args.bc, map_location="cpu", weights_only=True)
    miss, unexp = model.policy.load_state_dict(sd, strict=False)
    print(f"[bc-eval] loaded; missing={len(miss)} unexpected={len(unexp)}", flush=True)
    model.policy.set_training_mode(False)

    # Use BenchmarkEnvBuilder directly with the requested dataset (train or test).
    builder = BenchmarkEnvBuilder(args.task, dataset=args.dataset,
                                  action_space="joint_angle", max_steps=args.max_steps)
    n_ep = builder.get_episode_num()
    n_run = min(args.n_eval, n_ep) if n_ep > 0 else args.n_eval

    import cv2
    def _extract_obs(raw_obs):
        def np_(x): return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
        front = cv2.resize(np_(raw_obs["front_rgb_list"][-1]).astype(np.uint8), (IMG_W, IMG_H))
        wrist = cv2.resize(np_(raw_obs["wrist_rgb_list"][-1]).astype(np.uint8), (IMG_W, IMG_H))
        joint = np_(raw_obs["joint_state_list"][-1]).astype(np.float32).flatten()
        eef   = np_(raw_obs["eef_state_list"][-1]).astype(np.float32).flatten()
        grip  = np_(raw_obs["gripper_state_list"][-1]).astype(np.float32).flatten()
        return {"front_rgb": front, "wrist_rgb": wrist,
                "joint_state": joint[:7], "eef_state": eef[:6], "gripper": grip[:2]}

    successes = []
    for ep in range(n_run):
        env = builder.make_env_for_episode(ep)
        raw_obs, _ = env.reset()
        lstm = None; ep_start = True
        steps = 0
        while True:
            obs = _extract_obs(raw_obs)
            obs_b = {k: np.expand_dims(v, 0) for k, v in obs.items()}
            if args.recurrent:
                action, lstm = model.predict(obs_b, state=lstm,
                                              episode_start=np.array([ep_start], dtype=bool),
                                              deterministic=True)
                ep_start = False
            else:
                action, _ = model.predict(obs_b, deterministic=True)
            action = np.clip(action[0], -1.0, 1.0).astype(np.float32)
            raw_obs, _, term, trunc, info = env.step(action)
            steps += 1
            if bool(term) or bool(trunc): break
        success = bool(info.get("success", False))
        successes.append(success)
        env.close()
        print(f"  [{args.task}/{args.dataset}] ep {ep+1}/{n_run}  success={success}  steps={steps}", flush=True)

    sr = float(np.mean(successes)) if successes else 0.0
    print(f"\nBC-only on {args.dataset} split: SR = {sr*100:.1f}% ({int(sum(successes))}/{len(successes)})")
    if args.outfile:
        os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
        with open(args.outfile, "w") as f:
            json.dump({"split": args.dataset, "success_rate": sr,
                       "n_episodes": len(successes),
                       "successes": [int(s) for s in successes]}, f, indent=2)
        print(f"saved -> {args.outfile}")


if __name__ == "__main__":
    main()
