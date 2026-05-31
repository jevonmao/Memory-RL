"""Offline IQL evaluation from H5 dataset — no environment needed.

Loads a .pt checkpoint and the H5 demonstration dataset, runs the IQL actor
over all observations, and reports offline metrics: action reconstruction
error, value estimates, and advantage statistics.

No SAPIEN / Vulkan required — works locally or on Modal.

Local example:
    ROBOMME_DATA_DIR=/vol/robomme_data \\
    python training/eval_iql_offline.py \\
        --checkpoint iql_final.pt \\
        --task PickXtimes --seed 0

On Modal:
    modal run modal_app/app.py::eval_iql_offline --task PickXtimes --seed 0
    modal run modal_app/app.py::eval_iql_offline --task PickXtimes \\
        --checkpoint /workspace/iql_final.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.h5_dataset import make_dataloader           # noqa: E402
from training.iql import IQL, infer_dims_from_checkpoint  # noqa: E402
from training.utils import set_global_seed            # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Offline IQL evaluation on H5 dataset")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to iql_final.pt (or iql_stepN.pt) checkpoint")
    ap.add_argument("--task", required=True,
                    help="RoboMME task name, e.g. PickXtimes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", default=None,
                    help="Path to H5 data directory. Falls back to ROBOMME_DATA_DIR env var.")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--val_fraction", type=float, default=0.2,
                    help="Episode-level val fraction; must match the value used in training (default: 0.2)")
    ap.add_argument("--max_transitions", type=int, default=None,
                    help="Cap dataset size (default: use all transitions)")
    ap.add_argument("--out_dir", default=None,
                    help="Directory to save metrics.json. "
                         "Defaults to <checkpoint_dir>/eval_offline/")
    ap.add_argument("--device", default=None,
                    help="torch device override (default: cuda if available, else cpu)")
    return ap.parse_args()


@torch.no_grad()
def evaluate_offline(agent: IQL, loader: DataLoader) -> dict:
    """Run the IQL actor and critics over the entire dataset, return aggregate metrics."""
    action_sq_errors: list[float] = []
    v_vals:   list[float] = []
    q_vals:   list[float] = []
    adv_vals: list[float] = []
    rewards:  list[float] = []

    for batch in loader:
        obs     = batch["obs"].to(agent.device)
        actions = batch["actions"].to(agent.device)
        rews    = batch["rewards"]

        pred_actions = agent.actor.mode(obs)
        mse = ((pred_actions - actions) ** 2).mean(dim=-1)
        action_sq_errors.extend(mse.cpu().tolist())

        v   = agent.value(obs)
        q   = agent.q_tgt.min(obs, actions)
        adv = q - v

        v_vals.extend(v.cpu().tolist())
        q_vals.extend(q.cpu().tolist())
        adv_vals.extend(adv.cpu().tolist())
        rewards.extend(rews.tolist())

    rewards_arr = np.array(rewards)
    adv_arr     = np.array(adv_vals)
    n_success   = int((rewards_arr > 0).sum())

    return {
        "action_mse":            float(np.mean(action_sq_errors)),
        "action_rmse":           float(np.sqrt(np.mean(action_sq_errors))),
        "v_mean":                float(np.mean(v_vals)),
        "v_std":                 float(np.std(v_vals)),
        "q_mean":                float(np.mean(q_vals)),
        "q_std":                 float(np.std(q_vals)),
        "adv_mean":              float(np.mean(adv_arr)),
        "adv_std":               float(np.std(adv_arr)),
        "adv_positive_fraction": float((adv_arr > 0).mean()),
        "dataset_success_rate":  float(n_success / max(1, len(rewards))),
        "num_transitions":       len(rewards),
    }


def main() -> int:
    args = parse_args()
    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    set_global_seed(args.seed)

    data_dir = (
        args.data_dir
        or os.environ.get("ROBOMME_DATA_DIR")
        or os.environ.get("MANI_SKILL_DATA")
    )
    if not data_dir:
        raise SystemExit(
            "Data directory not set. Pass --data_dir or export ROBOMME_DATA_DIR."
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_iql_offline] checkpoint: {ckpt_path}")
    print(f"[eval_iql_offline] data_dir:   {data_dir}")
    print(f"[eval_iql_offline] device:     {device}")

    # Load the agent first so we can recover the training-split action stats
    # and apply the same normalization to the val dataset.
    obs_dim, action_dim = infer_dims_from_checkpoint(ckpt_path)
    agent = IQL(obs_dim=obs_dim, action_dim=action_dim, device=device)
    agent.load(ckpt_path)
    agent.value.eval()
    agent.qnet.eval()
    agent.q_tgt.eval()
    agent.actor.eval()
    print(f"[eval_iql_offline] loaded IQL weights  obs_dim={obs_dim}  action_dim={action_dim}")

    action_mean = agent.action_mean.numpy() if agent.action_mean is not None else None
    action_std  = agent.action_std.numpy()  if agent.action_std  is not None else None

    loader, *_ = make_dataloader(
        data_dir=data_dir,
        task=args.task,
        batch_size=args.batch_size,
        num_workers=0,
        max_transitions=args.max_transitions,
        pin_memory=(device == "cuda"),
        split="val",
        val_fraction=args.val_fraction,
        action_mean=action_mean,
        action_std=action_std,
    )

    print("[eval_iql_offline] evaluating ...")
    metrics = evaluate_offline(agent, loader)

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent.parent / "eval_offline"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print()
    print(json.dumps(metrics, indent=2))
    print(f"\n[eval_iql_offline] metrics saved → {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
