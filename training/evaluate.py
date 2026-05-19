"""Evaluate a trained PPO checkpoint (or random policy) on a RoboMME task.

Computes the metrics in `metrics/evaluation.py` and optionally writes raw
trajectories via `data/trajectory_logger.py` for later memory/curiosity
training and redundancy analysis.

Example:
    python training/evaluate.py --checkpoint logs/.../ppo_final.zip \\
        --task spatial_memory --episodes 20 --save_trajectories
    python training/evaluate.py --random --task CartPole-v1 \\
        --allow-gym-fallback --episodes 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.trajectory_logger import TrajectoryLogger  # noqa: E402
from env.robomme_env import make_env  # noqa: E402
from metrics.evaluation import EpisodeRecord, summarize  # noqa: E402
from training.utils import set_global_seed  # noqa: E402


def _load_policy(checkpoint: str, env):
    try:
        from stable_baselines3 import PPO
    except ImportError as e:
        raise SystemExit("stable-baselines3 is required to load a PPO checkpoint.") from e
    return PPO.load(checkpoint, env=env, device="auto")


def _run_episode(env, policy, deterministic: bool, logger):
    obs, info = env.reset()
    if logger is not None:
        logger.start_episode(obs)
    observations = [obs]
    rewards, infos = [], []
    terminated = truncated = False
    while not (terminated or truncated):
        if policy is None:
            action = env.action_space.sample()
        else:
            action, _ = policy.predict(obs, deterministic=deterministic)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        rewards.append(float(reward))
        infos.append(step_info if isinstance(step_info, dict) else {})
        observations.append(next_obs)
        if logger is not None:
            logger.record(action, reward, terminated, truncated, step_info, next_obs)
        obs = next_obs
    if logger is not None:
        logger.end_episode()
    return EpisodeRecord(
        observations=observations, rewards=rewards,
        terminated=bool(terminated), truncated=bool(truncated), infos=infos,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="Path to SB3 PPO .zip; omit with --random")
    ap.add_argument("--random", action="store_true", help="Use uniform-random policy")
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--allow-gym-fallback", action="store_true")
    ap.add_argument("--save_trajectories", action="store_true")
    ap.add_argument("--dataset", default=None, help="RoboMME split: train|val|test")
    ap.add_argument("--action_space", default=None, help="RoboMME action space")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--episode_idx", type=int, default=None,
                    help="Pin to a single RoboMME episode (default: cycle)")
    ap.add_argument("--out_dir", default=None,
                    help="Where to save metrics.json (and trajectories). Defaults next to the checkpoint.")
    args = ap.parse_args()

    if not args.random and args.checkpoint is None:
        ap.error("Provide --checkpoint or --random")

    set_global_seed(args.seed)
    env_kwargs = {k: v for k, v in {
        "dataset": args.dataset,
        "action_space": args.action_space,
        "max_steps": args.max_steps,
        "episode_idx": args.episode_idx,
    }.items() if v is not None}
    env = make_env(
        args.task, seed=args.seed,
        allow_gym_fallback=args.allow_gym_fallback,
        env_kwargs=env_kwargs or None,
    )

    policy = None if args.random else _load_policy(args.checkpoint, env)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif args.checkpoint:
        out_dir = Path(args.checkpoint).resolve().parent.parent / "eval_runs"
    else:
        out_dir = Path("logs") / f"random_{args.task}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = None
    if args.save_trajectories:
        logger = TrajectoryLogger(out_dir / "trajectories", task_name=args.task, seed=args.seed)

    episodes = []
    for i in range(args.episodes):
        ep = _run_episode(env, policy, args.deterministic, logger)
        episodes.append(ep)
        print(f"[eval] episode {i+1}/{args.episodes} return={sum(ep.rewards):.3f} "
              f"len={len(ep.rewards)} term={ep.terminated} trunc={ep.truncated}")

    if logger is not None:
        logger.close()
    env.close()

    metrics = summarize(episodes)
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"[eval] metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
