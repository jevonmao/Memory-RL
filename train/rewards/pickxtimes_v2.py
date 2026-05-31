"""Shaped reward V2 for PickXtimes — fixes the proximity-dominance issue
in V1.

V1 result on cluster (~300k steps, 4 baselines × 3 seeds): training reward
climbed smoothly 25→43 but eval SR stayed 0/10 across all baselines AND
stochastic-eval on train split also 0/10. Reward trajectory was monotonic +
spike-free, consistent with reward-hacking the proximity term rather than
actually completing subgoals.

V1 budget over a 1500-step episode:
  proximity:   0.05 / (1+d) per step  → up to ~50–70 from hovering alone
  step:        -0.01 per step          → -15
  subgoal:     +2 per advance          → max +14 (for N=3 cycles)
  pickup:      +1 per pickup           → max +3
  success:     +10
  →  proximity alone could equal subgoal+terminal contributions.

V2 rebalance: shrink proximity 10×, scale subgoal/pickup/terminal up 5–10×,
so the dominant signal is "did the agent advance the task" not "is the gripper
close to something."
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
def _to_np_xyz(p: Any) -> np.ndarray:
    if hasattr(p, "cpu"):
        p = p.cpu().numpy()
    arr = np.asarray(p).reshape(-1)
    return arr[:3].astype(np.float32)


def _scalar(x: Any) -> float:
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    arr = np.asarray(x)
    return float(arr.item() if arr.size == 1 else arr.flat[0])


def _flag(info: dict, key: str) -> bool:
    v = info.get(key, False)
    if hasattr(v, "any"):
        try:
            return bool(v.any())
        except Exception:
            pass
    return bool(v)


# ---------------------------------------------------------------------------
class PickXtimesRewardV2:
    """V2 — subgoal-dominated reward shape."""

    STEP_PENALTY     = -0.01
    SUBGOAL_BONUS    = 10.0      # was 2.0 — 5× boost
    PICKUP_BONUS     =  5.0      # was 1.0 — 5× boost
    REACH_COEF       =  0.5      # potential-based shaping (telescopes)
    PROXIMITY_COEF   =  0.005    # was 0.05 — 10× smaller, breaks the "hover for reward" exploit
    GRASP_HINT       =  0.5      # was 0.05 — bigger nudge to actually close the gripper
    GRASP_HINT_DIST  =  0.06
    TERMINAL_SUCCESS = 50.0      # was 10 — 5× boost
    TERMINAL_FAIL    = -10.0     # was -5

    def __init__(self) -> None:
        self._prev_timestep:    Optional[int]   = None
        self._prev_dist:        Optional[float] = None
        self._prev_pickup_done: int             = 0

    # ----------------------------------------------------------------
    def reset(self, unwrapped) -> None:
        self._prev_timestep    = getattr(unwrapped, "timestep", 0)
        self._prev_dist        = self._cur_target_dist(unwrapped)
        self._prev_pickup_done = self._pickups_completed(unwrapped)

    # ----------------------------------------------------------------
    def step(self, unwrapped, info: dict) -> float:
        r = self.STEP_PENALTY

        cur_ts = int(getattr(unwrapped, "timestep", 0))
        if self._prev_timestep is None:
            self._prev_timestep = cur_ts
        delta_ts = max(0, cur_ts - self._prev_timestep)
        r += self.SUBGOAL_BONUS * delta_ts
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

    # ----------------------------------------------------------------
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
