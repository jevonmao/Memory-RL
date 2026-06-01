from typing import List, Dict, Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# --------------------------------------------------
# Config
# --------------------------------------------------

OBS_KEYS = ("eef_state", "joint_state", "gripper_state")
ACTION_KEY = "joint_action"

IMAGE_KEYS = (
    "front_rgb",
    "wrist_rgb",
)

# --------------------------------------------------
# Helpers
# --------------------------------------------------


def _timestep_index(name: str) -> int:
    return int(name.split("_")[-1])


def _read_state(obs_grp) -> np.ndarray:
    parts = []

    for k in OBS_KEYS:
        if k in obs_grp:
            parts.append(
                np.asarray(
                    obs_grp[k],
                    dtype=np.float32
                ).flatten()
            )

    return np.concatenate(parts, axis=0)


def _read_image(obs_grp):
    """
    Returns:
        image: (3,H,W) float32 in [0,1]
    """

    for k in IMAGE_KEYS:

        if k not in obs_grp:
            continue

        img = np.asarray(obs_grp[k])

        if img.ndim != 3:
            continue

        # HWC -> CHW
        if img.shape[-1] in (1, 3):
            img = np.transpose(img, (2, 0, 1))

        img = img.astype(np.float32)

        if img.max() > 1:
            img = img / 255.0

        return img

    return None


def load_episode(ep_grp):

    timesteps = sorted(
        [k for k in ep_grp.keys()
         if k.startswith("timestep_")],
        key=_timestep_index
    )

    episode = []

    for ts_name in timesteps:

        ts = ep_grp[ts_name]

        if "obs" not in ts:
            continue

        if "action" not in ts:
            continue

        obs = ts["obs"]
        act = ts["action"]

        if ACTION_KEY not in act:
            continue

        state = _read_state(obs)
        image = _read_image(obs)

        action = np.asarray(
            act[ACTION_KEY],
            dtype=np.float32
        )

        episode.append({
            "state": state,
            "image": image,
            "action": action,
        })

    return episode


# --------------------------------------------------
# Dataset
# --------------------------------------------------


class RoboVLAPTPDataset(Dataset):

    def __init__(
        self,
        h5_files,
        history_len: int = 8,
        debug: bool = False,
        max_episodes: int = 5,
        max_samples: int = 5000,
    ):
        self.history_len = history_len
        self.episodes = []

        self.debug = debug
        self.max_samples = max_samples

        print("Loading episodes...")

        for h5_file in h5_files:

            print(f"  {h5_file}")

            with h5py.File(h5_file, "r") as hf:

                for ep_name in hf.keys():

                    if not isinstance(
                        hf[ep_name],
                        h5py.Group
                    ):
                        continue

                    ep = load_episode(hf[ep_name])

                    if len(ep) > history_len:
                        self.episodes.append(ep)

        print(f"Loaded {len(self.episodes)} episodes")

        # --------------------------------------------------
        # DEBUG MODE: limit episodes
        # --------------------------------------------------
        if self.debug:
            print("DEBUG MODE: limiting episodes")
            self.episodes = self.episodes[:max_episodes]

        # --------------------------------------------------
        # build index safely (with optional cap)
        # --------------------------------------------------
        self.index = []
        count = 0

        for ep_id, ep in enumerate(self.episodes):

            for t in range(history_len, len(ep)):

                self.index.append((ep_id, t))
                count += 1

                if self.debug and count >= max_samples:
                    break

            if self.debug and count >= max_samples:
                break

        print(f"Created {len(self.index)} training samples")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        ep_id, t = self.index[idx]

        ep = self.episodes[ep_id]

        history = ep[
            t - self.history_len:t
        ]

        # --------------------------------------------------
        # History inputs
        # --------------------------------------------------

        history_states = np.stack(
            [x["state"] for x in history]
        )

        history_images = []

        for x in history:

            if x["image"] is None:
                raise ValueError(
                    "Missing image in history"
                )

            history_images.append(
                x["image"]
            )

        history_images = np.stack(
            history_images
        )

        # --------------------------------------------------
        # Current observation
        # --------------------------------------------------

        current_state = ep[t]["state"]

        if ep[t]["image"] is None:
            raise ValueError(
                "Missing current image"
            )

        current_image = ep[t]["image"]

        # --------------------------------------------------
        # Targets
        # --------------------------------------------------

        current_action = ep[t]["action"]

        past_actions = np.stack(
            [x["action"] for x in history]
        )

        return {

            # history observations
            "history_states": torch.tensor(
                history_states,
                dtype=torch.float32
            ),

            "history_images": torch.tensor(
                history_images,
                dtype=torch.float32
            ),

            # current observation
            "current_state": torch.tensor(
                current_state,
                dtype=torch.float32
            ),

            "current_image": torch.tensor(
                current_image,
                dtype=torch.float32
            ),

            # BC target
            "current_action": torch.tensor(
                current_action,
                dtype=torch.float32
            ),

            # PTP target
            "past_actions": torch.tensor(
                past_actions,
                dtype=torch.float32
            ),
        }


# --------------------------------------------------
# Collate
# --------------------------------------------------


def collate_fn(batch):

    return {

        "history_states": torch.stack(
            [b["history_states"] for b in batch]
        ),

        "history_images": torch.stack(
            [b["history_images"] for b in batch]
        ),

        "current_state": torch.stack(
            [b["current_state"] for b in batch]
        ),

        "current_image": torch.stack(
            [b["current_image"] for b in batch]
        ),

        "current_action": torch.stack(
            [b["current_action"] for b in batch]
        ),

        "past_actions": torch.stack(
            [b["past_actions"] for b in batch]
        ),
    }


# --------------------------------------------------
# Debug test
# --------------------------------------------------

if __name__ == "__main__":

    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]

    DATA_PATH = (
        ROOT
        / "data"
        / "data"
        / "robomme_data_h5"
        / "record_dataset_PickXtimes.h5"
    )

    dataset = RoboVLAPTPDataset(
        [str(DATA_PATH)],
        history_len=8,
        debug=True,              # ✔ ENABLE DEBUG
        max_episodes=5,
        max_samples=2000
    )

    sample = dataset[0]

    print("history_states:",
          sample["history_states"].shape)

    print("history_images:",
          sample["history_images"].shape)

    print("current_state:",
          sample["current_state"].shape)

    print("current_image:",
          sample["current_image"].shape)

    print("current_action:",
          sample["current_action"].shape)

    print("past_actions:",
          sample["past_actions"].shape)