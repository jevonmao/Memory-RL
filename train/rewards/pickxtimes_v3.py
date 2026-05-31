"""V3 reward: V1 proximity gradient + V2 task-completion bonuses.

V1 (proximity=0.05) gave smooth learning signal but was exploitable.
V2 (proximity=0.005) removed the exploit but left no exploration gradient.
V3 keeps V1's proximity gradient AND boosts subgoal+terminal so they
dominate at the per-episode level — best of both worlds.

Budget over a 1500-step episode:
  proximity (always near target):  ~50–70   (from V1's 0.05/(1+d))
  step penalty:                    -15
  subgoal (full 3-cycle):         +30        (was +6 in V1)
  pickup (3 cycles):              +15        (was +3)
  success terminal:               +50        (was +10)
  grasp_hint:                     up to +50  (was +5)
=> task-completion bonus (>+95) easily dominates proximity (50-70) over
   an episode of >=2 subgoals, while proximity still pulls the policy
   toward objects (vs. V2 where the policy has no incentive to approach).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .pickxtimes_v2 import _to_np_xyz, _scalar, _flag  # noqa: F401  (shared helpers)


class PickXtimesRewardV3:
    """V3 — V1 proximity + V2 task-completion bonuses."""

    STEP_PENALTY     = -0.01
    SUBGOAL_BONUS    = 10.0    # like V2
    PICKUP_BONUS     =  5.0    # like V2
    REACH_COEF       =  0.5    # potential-based shaping (unchanged)
    PROXIMITY_COEF   =  0.05   # V1 level — keep the gradient
    GRASP_HINT       =  0.5    # V2's bigger grasp nudge
    GRASP_HINT_DIST  =  0.06
    TERMINAL_SUCCESS = 50.0    # V2 level — dominates proximity
    TERMINAL_FAIL    = -10.0

    def __init__(self) -> None:
        self._prev_timestep:    Optional[int]   = None
        self._prev_dist:        Optional[float] = None
        self._prev_pickup_done: int             = 0

    def reset(self, unwrapped) -> None:
        self._prev_timestep    = getattr(unwrapped, "timestep", 0)
        self._prev_dist        = self._cur_target_dist(unwrapped)
        self._prev_pickup_done = self._pickups_completed(unwrapped)

    def step(self, unwrapped, info: dict) -> float:
        r = self.STEP_PENALTY

        cur_ts = int(getattr(unwrapped, "timestep", 0))
        if self._prev_timestep is None:
            self._prev_timestep = cur_ts
        r += self.SUBGOAL_BONUS * max(0, cur_ts - self._prev_timestep)
        self._prev_timestep = cur_ts

        cur_pickups = self._pickups_completed(unwrapped)
        r += self.PICKUP_BONUS * max(0, cur_pickups - self._prev_pickup_done)
        self._prev_pickup_done = cur_pickups

        cur_dist = self._cur_target_dist(unwrapped)
        if cur_dist is not None:
            r += self.PROXIMITY_COEF / (1.0 + cur_dist)
            if self._prev_dist is not None:
                r += self.REACH_COEF * (self._prev_dist - cur_dist)
        self._prev_dist = cur_dist

        if self._is_pickup_phase(unwrapped) and cur_dist is not None and cur_dist < self.GRASP_HINT_DIST:
            if self._gripper_closing(unwrapped):
                r += self.GRASP_HINT

        if _flag(info, "success"):
            r += self.TERMINAL_SUCCESS
        if _flag(info, "fail"):
            r += self.TERMINAL_FAIL

        return float(r)

    @staticmethod
    def _is_pickup_phase(unwrapped) -> bool:
        n = int(getattr(unwrapped, "num_repeats", 1))
        ts = int(getattr(unwrapped, "timestep", 0))
        return (ts < 2 * n) and (ts % 2 == 0)

    @staticmethod
    def _is_drop_phase(unwrapped) -> bool:
        n = int(getattr(unwrapped, "num_repeats", 1))
        ts = int(getattr(unwrapped, "timestep", 0))
        return (ts < 2 * n) and (ts % 2 == 1)

    @staticmethod
    def _is_button_phase(unwrapped) -> bool:
        n = int(getattr(unwrapped, "num_repeats", 1))
        ts = int(getattr(unwrapped, "timestep", 0))
        return ts >= 2 * n

    def _pickups_completed(self, unwrapped) -> int:
        n  = int(getattr(unwrapped, "num_repeats", 1))
        ts = int(getattr(unwrapped, "timestep", 0))
        if ts >= 2 * n:
            return n
        return (ts + 1) // 2

    @staticmethod
    def _gripper_closing(unwrapped) -> bool:
        try:
            qpos = unwrapped.agent.robot.get_qpos()
            if hasattr(qpos, "cpu"):
                qpos = qpos.cpu().numpy()
            arr = np.asarray(qpos).reshape(-1)
            return bool(arr[-1] < 0.02 and arr[-2] < 0.02)
        except Exception:
            return False

    def _cur_target_dist(self, unwrapped) -> Optional[float]:
        try:
            tcp = _to_np_xyz(unwrapped.agent.tcp_pose.p)
        except Exception:
            return None

        if self._is_button_phase(unwrapped):
            try:
                return float(np.linalg.norm(tcp - _to_np_xyz(unwrapped.button.pose.p)))
            except Exception:
                return None

        target_cube = getattr(unwrapped, "target_cube", None)
        target_pad  = getattr(unwrapped, "target",      None)
        if target_cube is None:
            return None

        if self._is_drop_phase(unwrapped) and target_pad is not None:
            try:
                cube_p = _to_np_xyz(target_cube.pose.p)
                pad_p  = _to_np_xyz(target_pad.pose.p)
                return float(np.linalg.norm(cube_p - pad_p))
            except Exception:
                return None

        try:
            return float(np.linalg.norm(tcp - _to_np_xyz(target_cube.pose.p)))
        except Exception:
            return None
