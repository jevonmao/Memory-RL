"""
gym.Env wrapper around BenchmarkEnvBuilder for RL training.

Baseline 1 & 2 use RobommeRLEnv (current obs only).
Baseline 3 uses RobommeRLEnvWithMemory (current obs + K-step state/action history).
"""

from __future__ import annotations

import platform
import subprocess
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from robomme.env_record_wrapper import BenchmarkEnvBuilder


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
    Cycles through train-split episodes of a single task.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env_id: str = "BinFill", seed: int = 0):
        super().__init__()
        self.env_id = env_id
        self._rng = np.random.default_rng(seed)
        self._builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="train",
            action_space="joint_angle",
            max_steps=500,
        )
        self._num_episodes = self._builder.get_episode_num()
        self._episode_idx = 0
        self._env = None

        self.observation_space = spaces.Dict({
            "front_rgb":   spaces.Box(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8),
            "wrist_rgb":   spaces.Box(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8),
            "joint_state": spaces.Box(-np.inf, np.inf, (7,),  dtype=np.float32),
            "eef_state":   spaces.Box(-np.inf, np.inf, (6,),  dtype=np.float32),
            "gripper":     spaces.Box(-1., 1.,           (2,),  dtype=np.float32),
        })
        self.action_space = spaces.Box(-1., 1., (ACTION_DIM,), dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if self._env is not None:
            self._env.close()

        ep = self._episode_idx % self._num_episodes
        self._episode_idx += 1
        self._env = self._builder.make_env_for_episode(ep)
        raw_obs, info = self._env.reset()
        return _extract_obs(raw_obs), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1., 1.).astype(np.float32)
        raw_obs, reward, terminated, truncated, info = self._env.step(action)

        obs = _extract_obs(raw_obs)
        rew = float(reward.cpu().item() if hasattr(reward, "cpu") else reward)
        term = bool(terminated.cpu().item() if hasattr(terminated, "cpu") else terminated)
        trunc = bool(truncated.cpu().item() if hasattr(truncated, "cpu") else truncated)
        return obs, rew, term, trunc, info

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def render(self):
        pass


class RobommeRLEnvWithMemory(RobommeRLEnv):
    """
    Memory-augmented variant for Baseline 3 (PPO + PTP Memory).

    Adds two extra observation keys:
      history_state  : (K, STATE_DIM)  — last K [joint‖eef‖gripper] vectors
      history_action : (K, ACTION_DIM) — last K actions taken

    Both are zero-padded at episode start.
    Memory resets at every episode boundary.
    """

    def __init__(self, env_id: str = "BinFill", seed: int = 0, K: int = 8):
        super().__init__(env_id=env_id, seed=seed)
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
