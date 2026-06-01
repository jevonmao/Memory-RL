from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# -----------------------------
# Config
# -----------------------------

OBS_KEYS = ("eef_state", "joint_state", "gripper_state")
ACTION_KEY = "joint_action"
IMAGE_KEYS = ("front_rgb", "wrist_rgb")


# -----------------------------
# Helpers
# -----------------------------

def _timestep_index(name: str) -> int:
    return int(name.split("_")[-1])


def _read_state(obs_grp) -> np.ndarray:
    parts = []
    for k in OBS_KEYS:
        if k in obs_grp:
            parts.append(np.asarray(obs_grp[k], dtype=np.float32).flatten())
    return np.concatenate(parts, axis=0)


def _read_image(obs_grp):
    for k in IMAGE_KEYS:
        if k in obs_grp:
            img = np.asarray(obs_grp[k])
            if img.ndim == 3:
                return img
    return None


def _load_episode(ep_grp) -> List[Dict[str, Any]]:
    timesteps = sorted(
        [k for k in ep_grp.keys() if k.startswith("timestep_")],
        key=_timestep_index
    )

    traj = []

    for ts_name in timesteps:
        ts = ep_grp[ts_name]

        if "obs" not in ts or "action" not in ts:
            continue

        obs = ts["obs"]
        act = ts["action"]

        if ACTION_KEY not in act:
            continue

        state = _read_state(obs)
        image = _read_image(obs)
        action = np.asarray(act[ACTION_KEY], dtype=np.float32)

        traj.append({
            "image": image,
            "state": state,
            "action": action,
        })

    return traj


def load_all_episodes(h5_path: str):
    episodes = []

    with h5py.File(h5_path, "r") as f:
        for ep_name in f.keys():
            if not isinstance(f[ep_name], h5py.Group):
                continue

            ep = _load_episode(f[ep_name])

            if len(ep) > 10:
                episodes.append(ep)

    return episodes


# -----------------------------
# Dataset
# -----------------------------

class RoboMMEMemoryDataset(Dataset):

    def __init__(
        self,
        episodes,
        history_len: int = 8,
        use_ptp: bool = True,
    ):
        self.episodes = episodes
        self.history_len = history_len
        self.use_ptp = use_ptp

        self.index = []
        for eid, ep in enumerate(episodes):
            for t in range(history_len, len(ep) - 1):
                self.index.append((eid, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        ep_id, t = self.index[idx]
        ep = self.episodes[ep_id]

        hist = ep[t - self.history_len:t]

        # -----------------------------
        # History inputs
        # -----------------------------

        history_images = [
            h["image"] if h["image"] is not None else np.zeros((1,))
            for h in hist
        ]

        history_states = np.stack([
            h["state"] for h in hist
        ])

        # -----------------------------
        # Current / next / past signals
        # -----------------------------

        current = ep[t]
        future = ep[t + 1]

        past_actions = np.stack([
            h["action"] for h in hist
        ])

        sample = {
            "history_images": history_images,
            "history_states": history_states,

            "current_state": current["state"],
            "current_image": current["image"],

            "action": current["action"],              # BC target
            "future_action": future["action"],        # optional planning signal
            "past_actions": past_actions,             # PTP target
        }

        return sample


# -----------------------------
# Collate function
# -----------------------------

def collate_fn(batch):
    return {
        "history_states": torch.tensor(
            np.array([b["history_states"] for b in batch]),
            dtype=torch.float32
        ),

        "current_state": torch.tensor(
            np.array([b["current_state"] for b in batch]),
            dtype=torch.float32
        ),

        "action": torch.tensor(
            np.array([b["action"] for b in batch]),
            dtype=torch.float32
        ),

        "future_action": torch.tensor(
            np.array([b["future_action"] for b in batch]),
            dtype=torch.float32
        ),

        "past_actions": torch.tensor(
            np.array([b["past_actions"] for b in batch]),
            dtype=torch.float32
        ),
    }


# -----------------------------
# Builder
# -----------------------------

def build_dataset(data_dir: str, task: str, history_len: int = 8):

    data_dir = Path(data_dir)

    h5_files = list(data_dir.glob(f"*{task}*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No H5 found for task {task}")

    episodes = []
    for f in h5_files:
        print("Loading:", f)
        episodes.extend(load_all_episodes(str(f)))

    print("Total episodes:", len(episodes))

    return RoboMMEMemoryDataset(
        episodes,
        history_len=history_len,
        use_ptp=True,
    )


# -----------------------------
# Debug
# -----------------------------

if __name__ == "__main__":

    dataset = build_dataset(
        data_dir="data/robomme_data_h5",
        task="PickXtimes",
        history_len=8,
    )

    print("Dataset size:", len(dataset))

    sample = dataset[0]

    print("states:", sample["history_states"].shape)
    print("action:", sample["action"].shape)
    print("past_actions:", sample["past_actions"].shape)
    print("future_action:", sample["future_action"].shape)