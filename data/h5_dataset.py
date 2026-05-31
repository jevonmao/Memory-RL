"""Offline replay buffer built from RoboMME H5 episode files.

Confirmed H5 format (verified via inspect_h5 on record_dataset_PickXtimes.h5):

    HIERARCHICAL layout — episode groups at file root, timesteps nested inside:

    /episode_ID/                  <- root-level group, one per episode
        obs/                      <- initial observation (before first action)
            eef_state:    (6,)    float32
            joint_state:  (7,)    float32
            gripper_state:(2,)    float32
            (also: front_rgb, wrist_rgb, depth, extrinsics — ignored)
        info/
            is_completed: ()      bool
        timestep_K/               <- depth=1, one per step; K is a global index
            obs/
                eef_state, joint_state, gripper_state
            action/
                joint_action:    (8,)   float64  <- default
                eef_action:      (7,)   float64
                waypoint_action: (7,)   float64
            info/
                is_completed: () bool

Transitions built by _read_episode:
    obs      = timestep_K / obs   (observation the policy sees before taking action K)
    action   = timestep_K / action / joint_action
    next_obs = timestep_{K+1} / obs  (or self-loop on the last / success step)
    reward   = float(timestep_K / info / is_completed)
    done     = reward > 0 or last timestep in episode

Note: episode_root/obs is the initial observation and equals timestep_0/obs;
it is not used in transition construction.

obs_dim = 6 + 7 + 2 = 15   |   action_dim = 8
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_DEFAULT_OBS_KEYS   = ("eef_state", "joint_state", "gripper_state")
_DEFAULT_ACTION_KEY = "joint_action"


# ---------------------------------------------------------------------------
# H5 inspection — safe, never walks the full file
# ---------------------------------------------------------------------------

def inspect_h5(path: str | Path, n_timesteps: int = 2) -> None:
    """Print a concise H5 summary without walking all groups.

    Shows top-level key counts, root obs/info structure, and the first
    n_timesteps timestep groups from the first episode.
    """
    with h5py.File(path, "r") as f:
        root_keys = list(f.keys())
        ep_keys   = [k for k in root_keys if isinstance(f[k], h5py.Group)]
        print(f"=== {path} ===")
        print(f"Root-level groups : {len(ep_keys)}  (first few: {ep_keys[:5]})")

        if not ep_keys:
            print("  (no groups at root — unexpected format)")
            return

        # Inspect the first episode group
        ep = f[ep_keys[0]]
        print(f"\nFirst episode group: '{ep_keys[0]}'")
        print(f"  Direct children: {list(ep.keys())}")

        def _show(grp, indent=2):
            for k, v in grp.items():
                pad = " " * indent
                if isinstance(v, h5py.Group):
                    print(f"{pad}[group] {k}/")
                    _show(v, indent + 2)
                else:
                    print(f"{pad}{k}: shape={v.shape} dtype={v.dtype}")

        for child in ("obs", "info"):
            if child in ep:
                print(f"\n  {child}/")
                _show(ep[child], indent=4)

        # Show first n_timesteps timestep sub-groups
        ts_names = sorted(
            [k for k in ep.keys() if k.startswith("timestep_") and isinstance(ep[k], h5py.Group)],
            key=_timestep_index,
        )
        print(f"\n  timestep sub-groups: {len(ts_names)} total"
              + (f" ({ts_names[0]} … {ts_names[-1]})" if ts_names else ""))
        for ts_name in ts_names[:n_timesteps]:
            print(f"\n  [group] {ts_name}/")
            _show(ep[ts_name], indent=4)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_obs(obs_grp: h5py.Group, obs_keys: Sequence[str]) -> np.ndarray:
    parts: List[np.ndarray] = []
    for k in obs_keys:
        if k in obs_grp and not isinstance(obs_grp[k], h5py.Group):
            parts.append(np.asarray(obs_grp[k], dtype=np.float32).flatten())
    if not parts:
        raise ValueError(
            f"None of {list(obs_keys)} found. "
            f"Available scalar keys: "
            f"{[k for k in obs_grp.keys() if not isinstance(obs_grp[k], h5py.Group)]}"
        )
    return np.concatenate(parts)


def _timestep_index(name: str) -> int:
    tail = name.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


# ---------------------------------------------------------------------------
# Per-episode parser
# ---------------------------------------------------------------------------

def _read_episode(
    ep_grp: h5py.Group,
    obs_keys: Sequence[str],
    action_key: str,
) -> Optional[Dict[str, np.ndarray]]:
    """Build transition arrays from one episode group.

    Each episode group contains only timestep_K/ sub-groups (no episode-level
    obs/). Transitions are (obs_K, action_K, obs_{K+1}) where obs_K comes from
    timestep_K/obs/ — the observation the policy sees at step K.
    """
    ts_names = sorted(
        [k for k in ep_grp.keys()
         if k.startswith("timestep_") and isinstance(ep_grp[k], h5py.Group)],
        key=_timestep_index,
    )
    if not ts_names:
        return None

    # Pre-read every timestep so we can look ahead for next_obs
    steps: List[Dict] = []
    for ts_name in ts_names:
        ts = ep_grp[ts_name]

        obs = None
        if "obs" in ts and isinstance(ts["obs"], h5py.Group):
            try:
                obs = _read_obs(ts["obs"], obs_keys)
            except ValueError:
                pass

        action = None
        if "action" in ts and isinstance(ts["action"], h5py.Group):
            if action_key in ts["action"]:
                action = np.asarray(ts["action"][action_key], dtype=np.float32).flatten()

        reward = 0.0
        if "info" in ts and "is_completed" in ts["info"]:
            reward = float(np.asarray(ts["info"]["is_completed"]))

        steps.append({"obs": obs, "action": action, "reward": reward})

    obs_list:      List[np.ndarray] = []
    next_obs_list: List[np.ndarray] = []
    action_list:   List[np.ndarray] = []
    reward_list:   List[float]      = []
    done_list:     List[float]      = []

    for i, step in enumerate(steps):
        if step["obs"] is None or step["action"] is None:
            continue

        is_last = (i == len(steps) - 1)
        done    = float(is_last or step["reward"] > 0.0)

        # next_obs: next step's obs, or absorbing (self) if done/last
        if not done and i + 1 < len(steps) and steps[i + 1]["obs"] is not None:
            next_obs = steps[i + 1]["obs"]
        else:
            next_obs = step["obs"]

        obs_list.append(step["obs"])
        next_obs_list.append(next_obs)
        action_list.append(step["action"])
        reward_list.append(step["reward"])
        done_list.append(done)

    if not obs_list:
        return None

    return {
        "obs":      np.stack(obs_list).astype(np.float32),
        "next_obs": np.stack(next_obs_list).astype(np.float32),
        "actions":  np.stack(action_list).astype(np.float32),
        "rewards":  np.array(reward_list, dtype=np.float32),
        "dones":    np.array(done_list,   dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class H5ReplayBuffer(Dataset):
    """Flat PyTorch Dataset of offline (obs, action, reward, next_obs, done)."""

    def __init__(
        self,
        h5_paths: Sequence[str | Path],
        obs_keys: Sequence[str] = _DEFAULT_OBS_KEYS,
        action_key: str = _DEFAULT_ACTION_KEY,
        max_transitions: Optional[int] = None,
        reward_scale: float = 1.0,
        split: str = "all",
        val_fraction: float = 0.2,
        action_mean: Optional[np.ndarray] = None,
        action_std: Optional[np.ndarray] = None,
    ):
        if split not in ("all", "train", "val"):
            raise ValueError(f"split must be 'all', 'train', or 'val', got '{split}'")

        bufs: Dict[str, List[np.ndarray]] = {
            k: [] for k in ("obs", "next_obs", "actions", "rewards", "dones")
        }
        total = 0
        n_eps = 0

        for path in h5_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"H5 file not found: {path}")
            with h5py.File(path, "r") as f:
                # Collect and sort episode keys for a deterministic, reproducible split.
                all_ep_keys = sorted(
                    (k for k in f.keys() if isinstance(f[k], h5py.Group)),
                    key=_timestep_index,
                )

                if split == "all" or not all_ep_keys:
                    ep_keys = all_ep_keys
                else:
                    n_val   = max(1, int(len(all_ep_keys) * val_fraction))
                    n_train = len(all_ep_keys) - n_val
                    ep_keys = all_ep_keys[:n_train] if split == "train" else all_ep_keys[n_train:]
                    print(
                        f"[H5ReplayBuffer] split={split}  "
                        f"episodes: train={n_train} val={n_val}  "
                        f"using {len(ep_keys)}"
                    )

                for ep_key in ep_keys:
                    if max_transitions and total >= max_transitions:
                        break
                    ep = _read_episode(f[ep_key], obs_keys, action_key)
                    if ep is None:
                        continue
                    for k in bufs:
                        bufs[k].append(ep[k])
                    total += len(ep["obs"])
                    n_eps += 1

        if not bufs["obs"]:
            raise RuntimeError(
                f"No valid episodes found in {[str(p) for p in h5_paths]}.\n"
                "Run inspect_h5 to verify obs_keys / action_key match the file."
            )

        def _cat(key: str) -> torch.Tensor:
            arr = np.concatenate(bufs[key], axis=0)
            if max_transitions:
                arr = arr[:max_transitions]
            return torch.from_numpy(arr)

        self.obs      = _cat("obs")
        self.next_obs = _cat("next_obs")
        self.actions  = _cat("actions")
        self.rewards  = _cat("rewards") * reward_scale
        self.dones    = _cat("dones")

        self.obs_dim    = self.obs.shape[1]
        self.action_dim = self.actions.shape[1]

        # Normalize actions to zero mean / unit variance.
        # If stats are provided (e.g. from the train split checkpoint) use them
        # directly so train and val are on the same scale.
        if action_mean is not None and action_std is not None:
            self.action_mean: np.ndarray = action_mean.astype(np.float32)
            self.action_std:  np.ndarray = action_std.astype(np.float32)
        else:
            self.action_mean = self.actions.mean(dim=0).numpy().astype(np.float32)
            self.action_std  = self.actions.std(dim=0).numpy().astype(np.float32)
            self.action_std  = np.clip(self.action_std, 1e-6, None)

        mean_t = torch.from_numpy(self.action_mean)
        std_t  = torch.from_numpy(self.action_std)
        self.actions = (self.actions - mean_t) / std_t

        n_success = int((self.rewards > 0).sum().item())
        print(
            f"[H5ReplayBuffer] {n_eps} episodes | {len(self):,} transitions | "
            f"obs_dim={self.obs_dim} action_dim={self.action_dim} | "
            f"success steps={n_success} ({100*n_success/max(1,len(self)):.1f}%)"
        )

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx):
        return {
            "obs":      self.obs[idx],
            "next_obs": self.next_obs[idx],
            "actions":  self.actions[idx],
            "rewards":  self.rewards[idx],
            "dones":    self.dones[idx],
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def find_h5_files(data_dir: str | Path, task: str) -> List[Path]:
    data_dir = Path(data_dir)
    hits = list(data_dir.glob(f"*{task}*.h5")) + list(data_dir.glob(f"*{task}*/*.h5"))
    if not hits:
        low = task.lower()
        hits = [p for p in data_dir.rglob("*.h5") if low in p.name.lower()]
    return sorted(set(hits))


def make_dataloader(
    data_dir: str | Path,
    task: str,
    obs_keys: Sequence[str] = _DEFAULT_OBS_KEYS,
    action_key: str = _DEFAULT_ACTION_KEY,
    batch_size: int = 256,
    num_workers: int = 4,
    max_transitions: Optional[int] = None,
    reward_scale: float = 1.0,
    pin_memory: bool = True,
    split: str = "all",
    val_fraction: float = 0.2,
    action_mean: Optional[np.ndarray] = None,
    action_std: Optional[np.ndarray] = None,
) -> Tuple["DataLoader", int, int]:
    """Build a DataLoader for IQL training/eval. Returns (loader, obs_dim, action_dim).

    split: "all" uses every episode; "train" / "val" applies an 80/20 episode-level
    split deterministically by sorted episode key. val_fraction controls the ratio.
    """
    paths = find_h5_files(data_dir, task)
    if not paths:
        raise FileNotFoundError(
            f"No H5 files found for task '{task}' in {data_dir}.\n"
            "Run: modal run modal_app/app.py::download_data"
        )
    print(f"[make_dataloader] task='{task}'  split='{split}'  {len(paths)} file(s): "
          f"{[p.name for p in paths]}")

    dataset = H5ReplayBuffer(
        paths,
        obs_keys=obs_keys,
        action_key=action_key,
        max_transitions=max_transitions,
        reward_scale=reward_scale,
        split=split,
        val_fraction=val_fraction,
        action_mean=action_mean,
        action_std=action_std,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    return loader, dataset.obs_dim, dataset.action_dim
