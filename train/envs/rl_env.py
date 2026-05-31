"""
gym.Env wrapper around BenchmarkEnvBuilder for RL training.

Baseline 1 & 2 use RobommeRLEnv (current obs only).
Baseline 3 uses RobommeRLEnvWithMemory (current obs + K-step state/action history).

The benchmark's underlying compute_dense_reward is a zero stub, so this wrapper
substitutes a shaped reward from train.rewards.<task>.<TaskReward> when one is
registered. See plan: F1-F3, train/rewards/binfill.py.
"""

from __future__ import annotations

import gc
import os
import platform
import subprocess
from collections import deque
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

# Default to obs_mode="rgb" for RL: we only read RGB + proprio, and skipping
# the depth+segmentation render passes roughly halves SAPIEN's per-step GPU
# work. This is a pure throughput optimization — the wrapper only ever reads
# RGB anyway, so the trained policy sees identical observations regardless.
# Override (e.g. ROBOMME_OBS_MODE=rgb+depth+segmentation) if a downstream
# consumer needs the extra modalities.
os.environ.setdefault("ROBOMME_OBS_MODE", "rgb")
# NOTE: we intentionally do NOT default ROBOMME_CAMERA_RES or ROBOMME_SIM_FREQ.
#   * ROBOMME_CAMERA_RES would change SAPIEN's native render resolution
#     (default 256 → resize to 128 in wrapper). Setting it to 128 native
#     gives a small (~5%) GPU speedup but introduces a distribution shift
#     vs the official eval, which renders at 256 native.
#   * ROBOMME_SIM_FREQ would change physx_cpu substeps per env.step
#     (default 100 → 40 gives ~14% CPU speedup) but changes the physics
#     fidelity the policy is trained against. Official eval uses 100.
# Both env vars are still honored by episode_config_resolver if explicitly
# set — they are useful for fast smoke tests or for runs that don't need
# to compare against the official leaderboard.

from robomme.env_record_wrapper import BenchmarkEnvBuilder
from train.rewards import make_reward


def _vulkan_render_backend() -> str:
    if platform.system() != "Windows":
        return "gpu"
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        pci = r.stdout.strip().lower()
        parts = pci.split(":")
        return f"pci:{parts[1]}:{parts[2]}"
    except Exception:
        return "cpu"


_RENDER_BACKEND = _vulkan_render_backend()

IMG_H, IMG_W = 128, 128   # resize obs images to save memory
STATE_DIM = 15             # joint(7) + eef(6) + gripper(2)
ACTION_DIM = 8


def _to_scalar(x: Any) -> Any:
    """Coerce torch / numpy / batched scalars down to a Python scalar."""
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    arr = np.asarray(x)
    if arr.size == 1:
        return arr.item()
    return arr.flat[0]


def _extract_obs(raw_obs: dict) -> dict[str, np.ndarray]:
    """Pull current-step arrays out of the list-based benchmark obs."""
    def _to_np(x):
        return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

    front = _to_np(raw_obs["front_rgb_list"][-1]).astype(np.uint8)
    wrist = _to_np(raw_obs["wrist_rgb_list"][-1]).astype(np.uint8)
    joint = _to_np(raw_obs["joint_state_list"][-1]).astype(np.float32).flatten()
    eef   = _to_np(raw_obs["eef_state_list"][-1]).astype(np.float32).flatten()
    grip  = _to_np(raw_obs["gripper_state_list"][-1]).astype(np.float32).flatten()

    import cv2
    front = cv2.resize(front, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    wrist = cv2.resize(wrist, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

    return {
        "front_rgb":   front,
        "wrist_rgb":   wrist,
        "joint_state": joint[:7],
        "eef_state":   eef[:6],
        "gripper":     grip[:2],
    }


class RobommeRLEnv(gym.Env):
    """
    Standard (memoryless) gym wrapper for RL.

    Cycles through train-split episodes of a single task in a shuffled order
    that reshuffles every pass. The shaped reward (if registered for this
    task) replaces the underlying env's reward; otherwise the env reward is
    passed through unchanged.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self,
                 env_id: str = "BinFill",
                 seed: int = 0,
                 shape_reward: bool = True,
                 max_steps: int = 1500):
        """
        max_steps defaults to 1500 to match the official RoboMME Challenge
        evaluation horizon (challenge_interface/scripts/phase1_eval.py).
        Override for fast smoke/iteration runs where you knowingly don't
        need policies that can complete full-length episodes.
        """
        super().__init__()
        self.env_id = env_id
        self._rng = np.random.default_rng(seed)
        self._builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="train",
            action_space="joint_angle",
            max_steps=max_steps,
        )
        self._num_episodes = self._builder.get_episode_num()
        if self._num_episodes <= 0:
            raise RuntimeError(
                f"BenchmarkEnvBuilder reports 0 episodes for {env_id} "
                f"(train split); cannot train."
            )
        # Shuffled episode pointer (reshuffled every pass).
        self._episode_order = self._rng.permutation(self._num_episodes)
        self._pos = 0
        self._env = None

        # Per-task shaped reward (None if not registered → pass through env reward).
        self._reward_fn = make_reward(env_id) if shape_reward else None

        self.observation_space = spaces.Dict({
            "front_rgb":   spaces.Box(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8),
            "wrist_rgb":   spaces.Box(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8),
            "joint_state": spaces.Box(-np.inf, np.inf, (7,),  dtype=np.float32),
            "eef_state":   spaces.Box(-np.inf, np.inf, (6,),  dtype=np.float32),
            "gripper":     spaces.Box(-1., 1.,           (2,),  dtype=np.float32),
        })
        self.action_space = spaces.Box(-1., 1., (ACTION_DIM,), dtype=np.float32)

    # ------------------------------------------------------------------
    def _next_episode_idx(self) -> int:
        if self._pos >= len(self._episode_order):
            self._episode_order = self._rng.permutation(self._num_episodes)
            self._pos = 0
        ep = int(self._episode_order[self._pos])
        self._pos += 1
        return ep

    # Reset counter for amortized VRAM cleanup.
    _resets_since_gc: int = 0
    _GC_EVERY: int = 32

    def _release_env(self) -> None:
        """Tear down the current inner env without calling its .close().

        ManiSkill's close() path appears to leak VRAM when paired with
        gym.make() of a fresh env; dropping the reference is enough to
        keep VRAM growth bounded. gc.collect() + cuda.empty_cache() are
        expensive (10–30 ms each) so we rate-limit them — running every
        reset is wasted work when episodes are 500 steps long.
        """
        if self._env is not None:
            self._env = None
            self._resets_since_gc += 1
            if self._resets_since_gc >= self._GC_EVERY:
                self._resets_since_gc = 0
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def reset(self, *, seed=None, options=None):
        # Reseed the episode-order RNG if SB3 passes one through.
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._episode_order = self._rng.permutation(self._num_episodes)
            self._pos = 0

        self._release_env()

        ep = self._next_episode_idx()
        self._env = self._builder.make_env_for_episode(ep)
        # Per-reset inner-env seed: derived from our RNG so different --seed
        # values produce different trajectories (the resolver's metadata seed
        # is fixed per episode_idx).
        inner_seed = int(self._rng.integers(0, 2**31 - 1))
        raw_obs, info = self._env.reset(seed=inner_seed)

        if self._reward_fn is not None:
            self._reward_fn.reset(self._env.unwrapped)

        return _extract_obs(raw_obs), info if isinstance(info, dict) else {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1., 1.).astype(np.float32)
        raw_obs, reward, terminated, truncated, info = self._env.step(action)

        obs   = _extract_obs(raw_obs)
        term  = bool(_to_scalar(terminated))
        trunc = bool(_to_scalar(truncated))

        if self._reward_fn is not None:
            rew = self._reward_fn.step(self._env.unwrapped, info if isinstance(info, dict) else {})
        else:
            rew = float(_to_scalar(reward))

        return obs, rew, term, trunc, info

    def close(self):
        self._release_env()

    def render(self):
        pass


class RobommeRLEnvWithMemory(RobommeRLEnv):
    """
    Memory-augmented variant for Baseline 3 (PPO + PTP Memory).

    Adds two extra observation keys:
      history_state  : (K, STATE_DIM)  — last K [joint‖eef‖gripper] vectors
      history_action : (K, ACTION_DIM) — last K actions taken

    Both are zero-padded at episode start. Memory resets at every episode boundary.
    """

    def __init__(self,
                 env_id: str = "BinFill",
                 seed: int = 0,
                 K: int = 8,
                 shape_reward: bool = True,
                 max_steps: int = 1500):
        super().__init__(env_id=env_id, seed=seed, shape_reward=shape_reward,
                         max_steps=max_steps)
        self.K = K
        self._state_buf:  deque[np.ndarray] = deque(maxlen=K)
        self._action_buf: deque[np.ndarray] = deque(maxlen=K)

        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            "history_state":  spaces.Box(-np.inf, np.inf, (K, STATE_DIM),  dtype=np.float32),
            "history_action": spaces.Box(-1.,     1.,     (K, ACTION_DIM), dtype=np.float32),
        })

    def _get_history(self) -> dict[str, np.ndarray]:
        pad_s = np.zeros((self.K, STATE_DIM),  dtype=np.float32)
        pad_a = np.zeros((self.K, ACTION_DIM), dtype=np.float32)
        for i, s in enumerate(self._state_buf):
            pad_s[self.K - len(self._state_buf) + i] = s
        for i, a in enumerate(self._action_buf):
            pad_a[self.K - len(self._action_buf) + i] = a
        return {"history_state": pad_s, "history_action": pad_a}

    def reset(self, *, seed=None, options=None):
        self._state_buf.clear()
        self._action_buf.clear()
        obs, info = super().reset(seed=seed, options=options)
        state = np.concatenate([obs["joint_state"], obs["eef_state"], obs["gripper"]])
        self._state_buf.append(state)
        return {**obs, **self._get_history()}, info

    def step(self, action: np.ndarray):
        obs, rew, term, trunc, info = super().step(action)
        state = np.concatenate([obs["joint_state"], obs["eef_state"], obs["gripper"]])
        self._state_buf.append(state)
        self._action_buf.append(np.clip(action, -1., 1.).astype(np.float32))
        return {**obs, **self._get_history()}, rew, term, trunc, info
