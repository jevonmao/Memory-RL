"""Online evaluation of a trained IQL agent on a RoboMME task.

Loads a .pt checkpoint saved by train_iql, instantiates the RoboMME
environment (requires a local machine with Vulkan/SAPIEN support), and
rolls out the IQL actor for N episodes.

Checkpoint obs_dim / action_dim are inferred from the saved weight shapes
automatically — no need to pass them as flags.

Local example (after downloading checkpoint from Modal):
    ROBOMME_PATH=/home/jevon/projects/robomme_benchmark \\
    ROBOMME_DATA_DIR=/data/robomme_data \\
    python training/eval_iql.py \\
        --checkpoint logs/PickXtimes_seed0_20240101-120000/checkpoints/iql_final.pt \\
        --task PickXtimes --seed 0 --episodes 20

See README for how to download the checkpoint from the Modal volume.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.robomme_env import make_env              # noqa: E402
from metrics.evaluation import EpisodeRecord, summarize  # noqa: E402
from training.iql import IQL                      # noqa: E402
from training.utils import set_global_seed        # noqa: E402


def _infer_dims(ckpt_path: str | Path) -> tuple[int, int]:
    """Read obs_dim and action_dim from saved weight tensor shapes."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # value.net is [Linear(obs_dim, hidden), ReLU, Linear(hidden, hidden), ReLU, Linear(hidden, 1)]
    obs_dim = ckpt["value"]["net.0.weight"].shape[1]
    # actor.mu is Linear(hidden, action_dim)
    action_dim = ckpt["actor"]["mu.weight"].shape[0]
    return obs_dim, action_dim


def _run_episode(
    env,
    agent: IQL,
    deterministic: bool,
) -> EpisodeRecord:
    obs, info = env.reset()
    observations = [obs]
    rewards: list[float] = []
    infos: list[dict] = []
    terminated = truncated = False

    while not (terminated or truncated):
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        action = agent.predict(obs_t, deterministic=deterministic).squeeze(0).numpy()

        obs, reward, terminated, truncated, step_info = env.step(action)
        rewards.append(float(reward))
        infos.append(step_info if isinstance(step_info, dict) else {})
        observations.append(obs)

    return EpisodeRecord(
        observations=observations,
        rewards=rewards,
        terminated=bool(terminated),
        truncated=bool(truncated),
        infos=infos,
    )


def parse_args():
    ap = argparse.ArgumentParser(description="Online IQL evaluation on RoboMME")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to iql_final.pt (or iql_stepN.pt) checkpoint")
    ap.add_argument("--task", required=True,
                    help="RoboMME task name, e.g. PickXtimes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=20,
                    help="Number of evaluation episodes")
    ap.add_argument("--dataset", default="val",
                    choices=["train", "val", "test"],
                    help="RoboMME episode split to evaluate on (default: val)")
    ap.add_argument("--max_steps", type=int, default=300,
                    help="Max steps per episode before truncation")
    ap.add_argument("--stochastic", action="store_true",
                    help="Sample from the actor distribution instead of using the mode")
    ap.add_argument("--out_dir", default=None,
                    help="Directory to save metrics.json. Defaults to <checkpoint_dir>/eval_runs/")
    ap.add_argument("--device", default=None,
                    help="torch device override (default: cuda if available, else cpu)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    set_global_seed(args.seed)

    obs_dim, action_dim = _infer_dims(ckpt_path)
    print(f"[eval_iql] checkpoint: {ckpt_path}")
    print(f"[eval_iql] inferred  obs_dim={obs_dim}  action_dim={action_dim}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    agent = IQL(obs_dim=obs_dim, action_dim=action_dim, device=device)
    agent.load(ckpt_path)
    agent.value.eval()
    agent.qnet.eval()
    agent.actor.eval()
    print(f"[eval_iql] loaded IQL weights  device={device}")

    env = make_env(
        task_name=args.task,
        seed=args.seed,
        env_kwargs={
            "dataset": args.dataset,
            "action_space": "joint_angle",
            "max_steps": args.max_steps,
        },
    )
    print(f"[eval_iql] env ready  task={args.task}  dataset={args.dataset}")

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent.parent / "eval_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    deterministic = not args.stochastic
    episodes: list[EpisodeRecord] = []
    for i in range(args.episodes):
        ep = _run_episode(env, agent, deterministic)
        ret = sum(ep.rewards)
        success = ep.terminated and not ep.truncated and ret > 0
        print(
            f"[eval_iql] ep {i+1:>3}/{args.episodes}  "
            f"return={ret:.3f}  steps={len(ep.rewards):>4}  "
            f"success={'YES' if success else 'no '}"
        )
        episodes.append(ep)

    env.close()

    metrics = summarize(episodes)
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print()
    print(json.dumps(metrics, indent=2))
    print(f"\n[eval_iql] metrics saved → {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
