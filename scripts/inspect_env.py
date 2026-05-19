"""Print high-level information about a RoboMME task.

Usage:
    python scripts/inspect_env.py --task spatial_memory
    python scripts/inspect_env.py --task CartPole-v1 --allow-gym-fallback
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.robomme_env import describe_space, list_tasks, make_env  # noqa: E402


def _summarize_obs(obs):
    if isinstance(obs, dict):
        return {k: {"shape": tuple(np.asarray(v).shape), "dtype": str(np.asarray(v).dtype)} for k, v in obs.items()}
    arr = np.asarray(obs)
    return {"shape": tuple(arr.shape), "dtype": str(arr.dtype)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-gym-fallback", action="store_true")
    ap.add_argument("--list-tasks", action="store_true")
    ap.add_argument("--dataset", default="train")
    ap.add_argument("--action_space", default="joint_angle")
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--episode_idx", type=int, default=None)
    ap.add_argument("--no-flatten", action="store_true",
                    help="Forward raw dict observation (default flattens to a Box vector)")
    args = ap.parse_args()

    if args.list_tasks:
        tasks = list_tasks()
        print("Available RoboMME tasks:" if tasks else "No RoboMME tasks discovered.")
        for t in tasks:
            print(f"  - {t}")
        return 0

    env_kwargs = {
        "dataset": args.dataset,
        "action_space": args.action_space,
        "max_steps": args.max_steps,
        "flatten_obs": not args.no_flatten,
    }
    if args.episode_idx is not None:
        env_kwargs["episode_idx"] = args.episode_idx
    env = make_env(
        args.task, seed=args.seed,
        allow_gym_fallback=args.allow_gym_fallback,
        env_kwargs=env_kwargs,
    )
    obs, info = env.reset(seed=args.seed)
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    report = {
        "task_name": env.task_name,
        "backend": env.metadata_info.backend,
        "observation_space": describe_space(env.observation_space),
        "action_space": describe_space(env.action_space),
        "sample_observation": _summarize_obs(obs),
        "sample_action": np.asarray(action).tolist() if hasattr(action, "__iter__") or isinstance(action, (int, float, np.integer, np.floating)) else repr(action),
        "reward_after_random_step": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "step_info_keys": sorted(list(step_info.keys())) if isinstance(step_info, dict) else None,
        "reset_info_keys": sorted(list(info.keys())) if isinstance(info, dict) else None,
        "max_episode_steps": env.metadata_info.max_episode_steps,
        "task_metadata": env.metadata_info.extra,
    }
    print(json.dumps(report, indent=2, default=str))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
