"""Quick stochastic-action eval for diagnosing deterministic-vs-stochastic gap.

When training shaped-reward climbs but deterministic eval is 0, it's often
because the policy's mean action is degenerate while its sampled actions
sometimes complete subgoals.

Usage:
    python -m scripts.eval_stochastic <model_path> <baseline> <n_eval> [dataset]
"""
from __future__ import annotations

import os, sys

os.environ.setdefault("ROBOMME_OBS_MODE", "rgb")
os.environ.setdefault("ROBOMME_SIM_BACKEND", "physx_cpu")

import numpy as np
import torch
import cv2

from robomme.env_record_wrapper import BenchmarkEnvBuilder
from train.envs.rl_env import IMG_H, IMG_W


def main():
    model_path = sys.argv[1]
    baseline = sys.argv[2]              # 'ppo' | 'recurrent'
    n_eval = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    dataset = sys.argv[4] if len(sys.argv) > 4 else "test"
    recurrent = baseline == "recurrent"

    print(f"[stoch] model={model_path} baseline={baseline} n_eval={n_eval} dataset={dataset}", flush=True)

    if recurrent:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    model.policy.set_training_mode(False)

    builder = BenchmarkEnvBuilder("PickXtimes", dataset=dataset,
                                  action_space="joint_angle", max_steps=1500)
    n_ep = builder.get_episode_num()
    n_run = min(n_eval, n_ep) if n_ep > 0 else n_eval

    def _extract(raw):
        def np_(x): return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
        front = cv2.resize(np_(raw["front_rgb_list"][-1]).astype(np.uint8), (IMG_W, IMG_H))
        wrist = cv2.resize(np_(raw["wrist_rgb_list"][-1]).astype(np.uint8), (IMG_W, IMG_H))
        j = np_(raw["joint_state_list"][-1]).astype(np.float32).flatten()
        e = np_(raw["eef_state_list"][-1]).astype(np.float32).flatten()
        g = np_(raw["gripper_state_list"][-1]).astype(np.float32).flatten()
        return {"front_rgb": front, "wrist_rgb": wrist,
                "joint_state": j[:7], "eef_state": e[:6], "gripper": g[:2]}

    succ = []
    for ep in range(n_run):
        env = builder.make_env_for_episode(ep)
        raw, _ = env.reset()
        lstm = None
        ep_start = True
        while True:
            ob = _extract(raw)
            ob_b = {k: np.expand_dims(v, 0) for k, v in ob.items()}
            if recurrent:
                a, lstm = model.predict(ob_b, state=lstm,
                                         episode_start=np.array([ep_start], dtype=bool),
                                         deterministic=False)
                ep_start = False
            else:
                a, _ = model.predict(ob_b, deterministic=False)
            a = np.clip(a[0], -1.0, 1.0).astype(np.float32)
            raw, _, term, trunc, info = env.step(a)
            if bool(term) or bool(trunc):
                break
        s = bool(info.get("success", False))
        succ.append(s)
        env.close()
        print(f"  [{dataset}] ep {ep+1}/{n_run}  success={s}", flush=True)
    print(f"\nSTOCHASTIC SR ({dataset}) = {np.mean(succ)*100:.1f}% ({int(sum(succ))}/{len(succ)})")


if __name__ == "__main__":
    main()
