"""Eval that counts subgoals achieved per episode rather than just terminal
success. Useful when the policy makes partial progress (e.g. completes 1
pickup but not the full chain) — binary success would call this 0/N but
this diagnostic shows the actual completion rate per subgoal.

Usage:
    python -m scripts.eval_subgoal <model_path> <baseline> <n_eval>
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("ROBOMME_OBS_MODE", "rgb")
os.environ.setdefault("ROBOMME_SIM_BACKEND", "physx_cpu")

import numpy as np
import torch
import cv2

from robomme.env_record_wrapper import BenchmarkEnvBuilder
from train.envs.rl_env import IMG_H, IMG_W


def main():
    model_path = sys.argv[1]
    baseline   = sys.argv[2]
    n_eval     = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    recurrent  = baseline == "recurrent"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Auto-detect file type: SB3 .zip vs BC/DAgger .pt state_dict.
    is_state_dict = model_path.endswith(".pt") or not model_path.endswith(".zip")

    if recurrent:
        from sb3_contrib import RecurrentPPO
        if is_state_dict:
            # Build fresh model, load state_dict.
            import gymnasium as gym
            from gymnasium import spaces
            from stable_baselines3.common.vec_env import DummyVecEnv
            from train.envs.rl_env import IMG_H as _H, IMG_W as _W, ACTION_DIM as _A
            from train.models.encoder import RobommeCNNExtractor as _Enc
            obs_space = spaces.Dict({
                "front_rgb":   spaces.Box(0, 255, (_H, _W, 3), dtype=np.uint8),
                "wrist_rgb":   spaces.Box(0, 255, (_H, _W, 3), dtype=np.uint8),
                "joint_state": spaces.Box(-np.inf, np.inf, (7,),  dtype=np.float32),
                "eef_state":   spaces.Box(-np.inf, np.inf, (6,),  dtype=np.float32),
                "gripper":     spaces.Box(-1., 1., (2,),          dtype=np.float32),
            })
            act_space = spaces.Box(-1., 1., (_A,), dtype=np.float32)
            class _FE(gym.Env):
                def __init__(self): self.observation_space = obs_space; self.action_space = act_space
                def reset(self, *, seed=None, options=None): return obs_space.sample(), {}
                def step(self, a): return obs_space.sample(), 0.0, True, False, {}
            vec = DummyVecEnv([lambda: _FE()])
            model = RecurrentPPO(
                "MultiInputLstmPolicy", vec,
                policy_kwargs=dict(
                    features_extractor_class=_Enc,
                    net_arch=dict(pi=[256, 256], vf=[256, 256]),
                    lstm_hidden_size=256, n_lstm_layers=1,
                    shared_lstm=True, enable_critic_lstm=False,
                ),
                device=device, verbose=0, seed=0,
                n_steps=16, batch_size=16, n_epochs=1,
            )
            sd = torch.load(model_path, map_location="cpu", weights_only=True)
            miss, unexp = model.policy.load_state_dict(sd, strict=False)
            print(f"  loaded BC/DAgger state_dict; missing={len(miss)} unexpected={len(unexp)}", flush=True)
        else:
            model = RecurrentPPO.load(model_path, device=device)
    else:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device=device)
    model.policy.set_training_mode(False)

    builder = BenchmarkEnvBuilder("PickXtimes", dataset="test",
                                  action_space="joint_angle", max_steps=1500)
    n_run = min(n_eval, builder.get_episode_num())

    def _extract(raw):
        def np_(x): return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
        front = cv2.resize(np_(raw["front_rgb_list"][-1]).astype(np.uint8), (IMG_W, IMG_H))
        wrist = cv2.resize(np_(raw["wrist_rgb_list"][-1]).astype(np.uint8), (IMG_W, IMG_H))
        j = np_(raw["joint_state_list"][-1]).astype(np.float32).flatten()
        e = np_(raw["eef_state_list"][-1]).astype(np.float32).flatten()
        g = np_(raw["gripper_state_list"][-1]).astype(np.float32).flatten()
        return {"front_rgb": front, "wrist_rgb": wrist,
                "joint_state": j[:7], "eef_state": e[:6], "gripper": g[:2]}

    results = []
    for ep in range(n_run):
        env = builder.make_env_for_episode(ep)
        raw, _ = env.reset()
        lstm = None
        ep_start = True
        max_ts = 0
        while True:
            ob = _extract(raw)
            ob_b = {k: np.expand_dims(v, 0) for k, v in ob.items()}
            if recurrent:
                a, lstm = model.predict(ob_b, state=lstm,
                                         episode_start=np.array([ep_start], dtype=bool),
                                         deterministic=True)
                ep_start = False
            else:
                a, _ = model.predict(ob_b, deterministic=True)
            a = np.clip(a[0], -1.0, 1.0).astype(np.float32)
            raw, _, term, trunc, info = env.step(a)
            ts = int(getattr(env.unwrapped, "timestep", 0))
            max_ts = max(max_ts, ts)
            if bool(term) or bool(trunc):
                break
        n = int(getattr(env.unwrapped, "num_repeats", 1))
        total_subgoals = 2 * n + 1
        success = bool(info.get("success", False))
        results.append({"ep": ep, "max_subgoal": max_ts,
                        "total_subgoals": total_subgoals,
                        "success": success, "num_repeats": n})
        env.close()
        print(f"  ep {ep+1}/{n_run}: subgoal {max_ts}/{total_subgoals}  success={success}  num_repeats={n}", flush=True)

    print()
    print("=== Subgoal progress summary ===")
    avg_progress = np.mean([r["max_subgoal"] / r["total_subgoals"] for r in results])
    n_any_pickup = sum(1 for r in results if r["max_subgoal"] >= 1)
    n_any_drop   = sum(1 for r in results if r["max_subgoal"] >= 2)
    n_success    = sum(1 for r in results if r["success"])
    print(f"  avg progress: {avg_progress*100:.1f}%")
    print(f"  >=1 pickup:   {n_any_pickup}/{n_run}  ({n_any_pickup/n_run*100:.0f}%)")
    print(f"  >=1 drop:     {n_any_drop}/{n_run}  ({n_any_drop/n_run*100:.0f}%)")
    print(f"  success:      {n_success}/{n_run}  ({n_success/n_run*100:.0f}%)")


if __name__ == "__main__":
    main()
