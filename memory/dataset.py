from typing import Optional, Tuple
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# --------------------------------------------------
# Config
# --------------------------------------------------

OBS_KEYS = ("eef_state", "joint_state", "gripper_state")
ACTION_KEY = "joint_action"

IMAGE_KEYS = ("front_rgb", "wrist_rgb")

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _as_float_array(x) -> Optional[np.ndarray]:
    """Convert an HDF5 dataset/value to a finite float32 numpy array."""
    try:
        # h5py.Dataset is a handle; read the actual array before validation.
        if isinstance(x, h5py.Dataset):
            x = x[()]
        arr = np.asarray(x, dtype=np.float32)
    except (TypeError, ValueError, OSError):
        return None

    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None

    return arr


def _timestep_index(name: str) -> int:
    return int(name.split("_")[-1])


def _read_state(obs_grp) -> Optional[np.ndarray]:
    parts = []
    for k in OBS_KEYS:
        if k not in obs_grp:
            continue

        arr = _as_float_array(obs_grp[k])
        if arr is None:
            continue

        parts.append(arr.flatten())

    if not parts:
        return None

    return np.concatenate(parts, axis=0).astype(np.float32)


def _read_image(obs_grp) -> Optional[torch.Tensor]:
    for k in IMAGE_KEYS:
        if k not in obs_grp:
            continue

        img = _as_float_array(obs_grp[k])
        if img is None or img.ndim != 3:
            continue

        # HWC -> CHW. RoboMME RGB observations are usually stored as HWC.
        if img.shape[-1] in (1, 3):
            img = np.transpose(img, (2, 0, 1))

        # Require CHW with 1 or 3 channels after possible transpose.
        if img.shape[0] not in (1, 3):
            continue

        # Convert grayscale to RGB for CLIP.
        if img.shape[0] == 1:
            img = np.repeat(img, repeats=3, axis=0)

        # normalize to [0, 1]
        if img.max() > 1.0:
            img = img / 255.0

        # torch resize requires BCHW
        img_t = torch.from_numpy(img).float().unsqueeze(0)  # (1,C,H,W)
        img_t = F.interpolate(
            img_t,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).contiguous()

        # CLIP image normalization.
        img_t = (img_t - CLIP_MEAN) / CLIP_STD

        return img_t  # (3,224,224)

    return None


def _read_instruction(ep_grp) -> str:
    if "setup" not in ep_grp or "task_goal" not in ep_grp["setup"]:
        return "no_instruction"

    try:
        goal = ep_grp["setup"]["task_goal"][()]
    except (TypeError, OSError):
        return "no_instruction"

    goal = np.asarray(goal)
    if goal.size == 0:
        return "no_instruction"

    inst = goal.flatten()[0]

    if isinstance(inst, bytes):
        return inst.decode("utf-8")

    return str(inst)


def _read_action(act_grp) -> Optional[np.ndarray]:
    if ACTION_KEY not in act_grp:
        return None

    action = _as_float_array(act_grp[ACTION_KEY])
    if action is None:
        return None

    return action.flatten().astype(np.float32)


def _same_shape(a, expected_shape: Optional[Tuple[int, ...]]) -> bool:
    return expected_shape is None or tuple(a.shape) == expected_shape


def load_episode(ep_grp):
    timesteps = sorted(
        [k for k in ep_grp.keys() if k.startswith("timestep_")],
        key=_timestep_index,
    )

    instruction = _read_instruction(ep_grp)
    episode = []

    expected_state_shape = None
    expected_image_shape = None
    expected_action_shape = None

    for ts_name in timesteps:
        ts = ep_grp[ts_name]

        if "obs" not in ts or "action" not in ts:
            continue

        state = _read_state(ts["obs"])
        image = _read_image(ts["obs"])
        action = _read_action(ts["action"])

        # Skip incomplete rows. Otherwise __getitem__ can fail later in stack.
        if state is None or image is None or action is None:
            continue

        # Keep tensor dimensions fixed within an episode.
        if expected_state_shape is None:
            expected_state_shape = tuple(state.shape)
            expected_image_shape = tuple(image.shape)
            expected_action_shape = tuple(action.shape)

        if not _same_shape(state, expected_state_shape):
            continue
        if not _same_shape(image, expected_image_shape):
            continue
        if not _same_shape(action, expected_action_shape):
            continue

        episode.append({
            "state": state,
            "image": image,
            "action": action,
            "instruction": instruction,
        })

    return episode


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class RoboVLAPTPDataset(Dataset):

    def __init__(self, h5_files, history_len=8, debug=False, max_episodes=5, max_samples=5000):
        self.history_len = history_len
        self.episodes = []

        print("Loading episodes...")

        for h5_file in h5_files:
            print(f"  {h5_file}")

            with h5py.File(h5_file, "r") as hf:
                for ep_name in hf.keys():
                    if not isinstance(hf[ep_name], h5py.Group):
                        continue

                    ep = load_episode(hf[ep_name])

                    if len(ep) > history_len:
                        self.episodes.append(ep)

        print(f"Loaded {len(self.episodes)} episodes")

        if debug:
            self.episodes = self.episodes[:max_episodes]

        self.index = []
        count = 0

        for ep_id, ep in enumerate(self.episodes):
            for t in range(history_len, len(ep)):
                self.index.append((ep_id, t))
                count += 1

                if debug and count >= max_samples:
                    break

            if debug and count >= max_samples:
                break

        print(f"Created {len(self.index)} training samples")

    def __len__(self):
        return len(self.index)

    def num_episodes(self):
        return len(self.episodes)

    def episode_length(self, ep_id):
        return len(self.episodes[ep_id])

    def get_episode_window(self, ep_id, t):
        """Return a sample addressed by episode/time.

        t is the target timestep. The input history is [t-history_len, ..., t-1].
        """
        ep = self.episodes[ep_id]
        if t < self.history_len or t >= len(ep):
            raise IndexError(
                f"t={t} is invalid for episode length {len(ep)} and history_len {self.history_len}"
            )

        history = ep[t - self.history_len:t]

        history_states = np.stack([x["state"] for x in history]).astype(np.float32)
        history_images = torch.stack([x["image"] for x in history]).float()

        current_state = ep[t]["state"].astype(np.float32)
        current_image = ep[t]["image"].float()
        current_action = ep[t]["action"].astype(np.float32)
        past_actions = np.stack([x["action"] for x in history]).astype(np.float32)
        instruction = ep[t]["instruction"]

        return {
            "episode_id": ep_id,
            "timestep": t,
            "instruction": instruction,
            "history_states": torch.tensor(history_states, dtype=torch.float32),
            "history_images": history_images,
            "current_state": torch.tensor(current_state, dtype=torch.float32),
            "current_image": current_image,
            "current_action": torch.as_tensor(current_action, dtype=torch.float32),
            "past_actions": torch.as_tensor(past_actions, dtype=torch.float32),
        }

    def __getitem__(self, idx):
        ep_id, t = self.index[idx]
        return self.get_episode_window(ep_id, t)


# --------------------------------------------------
# Collate
# --------------------------------------------------

def collate_fn(batch):
    return {
        "episode_id": torch.as_tensor([b["episode_id"] for b in batch], dtype=torch.long),
        "timestep": torch.as_tensor([b["timestep"] for b in batch], dtype=torch.long),
        "instruction": [b["instruction"] for b in batch],

        "history_states": torch.stack([b["history_states"] for b in batch]),
        "history_images": torch.stack([b["history_images"] for b in batch]),

        "current_state": torch.stack([b["current_state"] for b in batch]),
        "current_image": torch.stack([b["current_image"] for b in batch]),

        "current_action": torch.stack([b["current_action"] for b in batch]),
        "past_actions": torch.stack([b["past_actions"] for b in batch]),
    }


def collate_episode_windows(samples, device):
    """Collate episode-ordered windows for parallel episode training."""
    batch = collate_fn(samples)
    return {
        "episode_id": batch["episode_id"],
        "timestep": batch["timestep"],
        "instruction": batch["instruction"],
        "history_states": batch["history_states"].to(device),
        "history_images": batch["history_images"].to(device),
        "current_state": batch["current_state"].to(device),
        "current_image": batch["current_image"].to(device),
        "current_action": batch["current_action"].to(device),
        "past_actions": batch["past_actions"].to(device),
    }

# --------------------------------------------------
# Debug test
# --------------------------------------------------

if __name__ == "__main__":

    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]

    DATA_PATH = (
        ROOT / "data" / "data" /
        "robomme_data_h5" /
        "record_dataset_BinFill.h5"
    )

    dataset = RoboVLAPTPDataset(
        [str(DATA_PATH)],
        history_len=8,
        debug=True,
        max_episodes=5,
        max_samples=2000
    )

    sample = dataset[0]

    print("\n=== SAMPLE DEBUG ===")
    print("instruction:", sample["instruction"])
    print("episode_id:", sample["episode_id"])
    print("timestep:", sample["timestep"])

    print("history_states:", sample["history_states"].shape)
    print("history_images:", sample["history_images"].shape)
    print("current_state:", sample["current_state"].shape)
    print("current_image:", sample["current_image"].shape)

    print("current_action:", sample["current_action"].shape)
    print("current_action dtype:", sample["current_action"].dtype)

    print("past_actions:", sample["past_actions"].shape)
    print("past_actions dtype:", sample["past_actions"].dtype)

    print("\n=== RAW ACTION CHECK ===")
    ep0 = dataset.episodes[0]
    print("raw action type:", type(ep0[0]["action"]))
    print("raw action dtype:", ep0[0]["action"].dtype)
    print("raw action shape:", ep0[0]["action"].shape)
    print("raw action sample:", ep0[0]["action"][: min(5, len(ep0[0]["action"]))])