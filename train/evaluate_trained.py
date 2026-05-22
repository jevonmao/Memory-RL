"""
Evaluate a trained SB3 model on the RoboMME benchmark (test split).

Runs N episodes per task and reports:
  - Per-task success rate
  - Per-task mean return
  - Aggregate mean across all tasks

Usage:
    # Vanilla PPO or PPO+ICM model (no memory obs):
    python -m train.evaluate_trained \\
        --model runs/ppo/ppo_BinFill_final.zip \\
        --baseline ppo

    # PPO+PTP Memory model:
    python -m train.evaluate_trained \\
        --model runs/ppo_ptp/ppo_ptp_BinFill_final.zip \\
        --baseline ptp --K 8

    # Evaluate on all 16 tasks:
    python -m train.evaluate_trained \\
        --model runs/ppo/ppo_BinFill_final.zip \\
        --baseline ppo --tasks all
"""

from __future__ import annotations

import argparse
import json
import os
import numpy as np
from stable_baselines3 import PPO

from robomme.env_record_wrapper import BenchmarkEnvBuilder

ALL_TASKS = [
    "PickXtimes", "StopCube", "SwingXtimes", "BinFill",
    "VideoUnmaskSwap", "VideoUnmask", "ButtonUnmaskSwap", "ButtonUnmask",
    "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder", "PickHighlight",
    "InsertPeg", "MoveCube", "PatternLock", "RouteStick",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    required=True,
                   help="Path to a .zip SB3 model file")
    p.add_argument("--baseline", choices=["ppo", "icm", "ptp"], default="ppo",
                   help="Which baseline (determines obs format)")
    p.add_argument("--tasks",    nargs="+", default=["BinFill"],
                   help="Task name(s) or 'all'")
    p.add_argument("--n_eval",   type=int, default=20,
                   help="Number of test episodes per task")
    p.add_argument("--K",        type=int, default=8,
                   help="History length for PTP baseline")
    p.add_argument("--device",   default="auto")
    p.add_argument("--outfile",  default=None,
                   help="Optional JSON file to write results")
    return p.parse_args()


def _extract_obs_memoryless(raw_obs: dict) -> dict:
    """Current-step obs for Baseline 1 & 2."""
    import numpy as np, cv2
    from train.envs.rl_env import IMG_H, IMG_W

    def _np(x):
        return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

    front = cv2.resize(_np(raw_obs["front_rgb_list"][-1]).astype(np.uint8), (IMG_W, IMG_H))
    wrist = cv2.resize(_np(raw_obs["wrist_rgb_list"][-1]).astype(np.uint8), (IMG_W, IMG_H))
    joint = _np(raw_obs["joint_state_list"][-1]).astype(np.float32).flatten()
    eef   = _np(raw_obs["eef_state_list"][-1]).astype(np.float32).flatten()
    grip  = _np(raw_obs["gripper_state_list"][-1]).astype(np.float32).flatten()
    return {
        "front_rgb":   front,
        "wrist_rgb":   wrist,
        "joint_state": joint[:7],
        "eef_state":   eef[:6],
        "gripper":     grip[:2],
    }


def _add_batch(obs: dict) -> dict:
    """Add batch dimension for model.predict."""
    import numpy as np
    return {k: np.expand_dims(v, 0) for k, v in obs.items()}


def evaluate_task(model, task: str, n_eval: int, baseline: str, K: int) -> dict:
    from collections import deque
    import numpy as np
    from train.envs.rl_env import STATE_DIM, ACTION_DIM

    builder   = BenchmarkEnvBuilder(task, dataset="test",
                                    action_space="joint_angle", max_steps=500)
    n_ep      = builder.get_episode_num()
    n_run     = min(n_eval, n_ep) if n_ep > 0 else n_eval

    successes, returns = [], []

    # History buffers for PTP baseline
    state_buf:  deque = deque(maxlen=K)
    action_buf: deque = deque(maxlen=K)

    def _get_history():
        pad_s = np.zeros((K, STATE_DIM),  dtype=np.float32)
        pad_a = np.zeros((K, ACTION_DIM), dtype=np.float32)
        for i, s in enumerate(state_buf):
            pad_s[K - len(state_buf) + i] = s
        for i, a in enumerate(action_buf):
            pad_a[K - len(action_buf) + i] = a
        return {"history_state": pad_s, "history_action": pad_a}

    for ep in range(n_run):
        env = builder.make_env_for_episode(ep % max(n_ep, 1))
        raw_obs, _ = env.reset()

        if baseline == "ptp":
            state_buf.clear()
            action_buf.clear()

        ep_return = 0.0
        done = False

        while not done:
            obs = _extract_obs_memoryless(raw_obs)
            if baseline == "ptp":
                state = np.concatenate([obs["joint_state"], obs["eef_state"], obs["gripper"]])
                state_buf.append(state)
                obs = {**obs, **_get_history()}

            action, _ = model.predict(_add_batch(obs), deterministic=True)
            action = action[0]

            raw_obs, reward, terminated, truncated, info = env.step(action)
            rew = float(reward.cpu().item() if hasattr(reward, "cpu") else reward)
            ep_return += rew

            if baseline == "ptp":
                action_buf.append(np.clip(action, -1., 1.).astype(np.float32))

            done = bool(terminated) or bool(truncated)

        success = bool(info.get("success", False))
        successes.append(float(success))
        returns.append(ep_return)
        env.close()
        print(f"  [{task}] ep {ep+1}/{n_run}  success={success}  return={ep_return:.2f}")

    return {
        "task":         task,
        "n_episodes":   n_run,
        "success_rate": float(np.mean(successes)),
        "mean_return":  float(np.mean(returns)),
    }


def main():
    args = parse_args()

    tasks = ALL_TASKS if (len(args.tasks) == 1 and args.tasks[0] == "all") else args.tasks

    model = PPO.load(args.model, device=args.device)
    model.policy.set_training_mode(False)

    all_results = []
    for task in tasks:
        print(f"\n=== Evaluating {task} ===")
        result = evaluate_task(model, task, args.n_eval, args.baseline, args.K)
        all_results.append(result)
        print(f"  success_rate={result['success_rate']:.3f}  mean_return={result['mean_return']:.2f}")

    print("\n=== Summary ===")
    for r in all_results:
        print(f"  {r['task']:25s}  SR={r['success_rate']:.3f}  R={r['mean_return']:.2f}")
    agg_sr = np.mean([r["success_rate"] for r in all_results])
    print(f"\n  Aggregate SR = {agg_sr:.3f}")

    if args.outfile:
        os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
        with open(args.outfile, "w") as f:
            json.dump({"results": all_results, "aggregate_sr": float(agg_sr)}, f, indent=2)
        print(f"Results written to {args.outfile}")


if __name__ == "__main__":
    main()
