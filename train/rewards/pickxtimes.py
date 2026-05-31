"""Shaped reward for PickXtimes.

Task structure (from src/robomme/robomme_env/PickXtimes.py):
  For N ∈ {1..5} pickup–drop cycles (set per episode by seed/difficulty):
    subgoal 2i     : is_obj_pickup(target_cube)             — pick up the target cube
    subgoal 2i+1   : is_obj_dropped_onto(target_cube,target)— place it on the target
  subgoal 2N       : is_button_pressed(button)              — press the button

`self.timestep` (set inside `sequential_task_check`) advances by 1 each time
the current subgoal's predicate goes True; that gives us a clean per-subgoal
progress signal that does NOT depend on env stepping logic.

Failure modes (from the task's failure_funcs):
  - During any pickup/drop subgoal: picking up any non-target cube OR
    pressing the button.
  - During the button subgoal: picking up any cube.
These already terminate the episode with info["fail"]=True; we add −5 there.

Reward shape (per step):
  step penalty                : -0.01    (encourages finishing)
  subgoal completion          : +2.0     per `timestep` advance (large + sparse
                                          → the dominant learning signal)
  pickup-count anchor         : +1.0 each time the pickup counter advances
                                (matters at the count axis the task tests)
  TCP→current-target distance : potential shaping
                                  +0.5 * (prev_d - cur_d)
                                plus per-step proximity
                                  +0.05 / (1 + cur_d)
                                ("current target" = target_cube during
                                pickup/drop, then the button)
  gripper-close-when-near     : +0.05 if within 6cm of target_cube AND closing
                                (only during a pickup subgoal — cheap hint
                                that the agent should grasp once close)
  terminal                    : +10 on success, -5 on fail

These coefficients are chosen so that:
  * subgoal_completion ≫ proximity   (final goal > intermediate shaping)
  * per-step proximity is non-zero even when the agent is far away
    (avoids the early-training plateau seen on BinFill v3 where shaped reward
     telescoped to ~0 and the agent only ever saw the -0.01 floor)

The reward is stateful (tracks previous distances/timestep) so the wrapper
instantiates one per episode via reset(unwrapped).
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
class PickXtimesReward:
    """Per-step shaped reward for PickXtimes (stateful across an episode)."""

    STEP_PENALTY     = -0.01
    SUBGOAL_BONUS    =  2.0
    PICKUP_BONUS     =  1.0
    REACH_COEF       =  0.5
    PROXIMITY_COEF   =  0.05
    GRASP_HINT       =  0.05
    GRASP_HINT_DIST  =  0.06
    TERMINAL_SUCCESS = 10.0
    TERMINAL_FAIL    = -5.0

    def __init__(self) -> None:
        self._prev_timestep:    Optional[int]   = None
        self._prev_dist:        Optional[float] = None
        self._prev_pickup_done: int             = 0

    # ----------------------------------------------------------------
    def reset(self, unwrapped) -> None:
        # `timestep` is created by sequential_task_check on the first
        # evaluate(); before any step it may not exist yet. Treat None.
        self._prev_timestep    = getattr(unwrapped, "timestep", 0)
        self._prev_dist        = self._cur_target_dist(unwrapped)
        self._prev_pickup_done = self._pickups_completed(unwrapped)

    # ----------------------------------------------------------------
    def step(self, unwrapped, info: dict) -> float:
        r = self.STEP_PENALTY

        # ---- subgoal completion ----
        cur_ts = int(getattr(unwrapped, "timestep", 0))
        if self._prev_timestep is None:
            self._prev_timestep = cur_ts
        delta_ts = max(0, cur_ts - self._prev_timestep)
        r += self.SUBGOAL_BONUS * delta_ts
        self._prev_timestep = cur_ts

        # ---- pickup-count anchor (separate from subgoal so that even
        # partially completing the pickup half of a cycle still rewards) ----
        cur_pickups = self._pickups_completed(unwrapped)
        r += self.PICKUP_BONUS * max(0, cur_pickups - self._prev_pickup_done)
        self._prev_pickup_done = cur_pickups

        # ---- distance shaping toward the *current* target ----
        cur_dist = self._cur_target_dist(unwrapped)
        if cur_dist is not None:
            r += self.PROXIMITY_COEF / (1.0 + cur_dist)
            if self._prev_dist is not None:
                r += self.REACH_COEF * (self._prev_dist - cur_dist)
        self._prev_dist = cur_dist

        # ---- grasp hint: small bonus when close + gripper closing AND we
        # are in a pickup phase (even subgoal index, not the final button) ----
        if self._is_pickup_phase(unwrapped) and cur_dist is not None and cur_dist < self.GRASP_HINT_DIST:
            if self._gripper_closing(unwrapped):
                r += self.GRASP_HINT

        # ---- terminal bonuses ----
        if _flag(info, "success"):
            r += self.TERMINAL_SUCCESS
        if _flag(info, "fail"):
            r += self.TERMINAL_FAIL

        return float(r)

    # ----------------------------------------------------------------
    # Phase helpers.
    # ----------------------------------------------------------------
    @staticmethod
    def _is_pickup_phase(unwrapped) -> bool:
        """True if the current subgoal is a pickup (even-indexed, < 2N)."""
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

    # ----------------------------------------------------------------
    def _pickups_completed(self, unwrapped) -> int:
        """How many of the N pickups have already been recorded as done.

        The task advances `timestep` by 1 each completed subgoal; subgoals
        alternate pickup/drop. So pickups completed = ceil(timestep / 2)
        clipped to N. Equivalently:
            (timestep + 1) // 2  if timestep < 2N else N.
        """
        n  = int(getattr(unwrapped, "num_repeats", 1))
        ts = int(getattr(unwrapped, "timestep", 0))
        if ts >= 2 * n:
            return n
        return (ts + 1) // 2

    # ----------------------------------------------------------------
    @staticmethod
    def _gripper_closing(unwrapped) -> bool:
        """Return True if both finger joints are within their closed range.

        Panda gripper qpos[-2:] are the two finger positions; ≈0.04 fully
        open, ≈0.00 fully closed.
        """
        try:
            qpos = unwrapped.agent.robot.get_qpos()
            if hasattr(qpos, "cpu"):
                qpos = qpos.cpu().numpy()
            arr = np.asarray(qpos).reshape(-1)
            return bool(arr[-1] < 0.02 and arr[-2] < 0.02)
        except Exception:
            return False

    # ----------------------------------------------------------------
    def _cur_target_dist(self, unwrapped) -> Optional[float]:
        """Distance from TCP (or cube) to the currently relevant target.

        - During pickup phase: TCP → target_cube (must grasp)
        - During drop  phase: target_cube → self.target (drop on pad)
        - During button phase: TCP → button.pose
        """
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

        # pickup phase (or fallback)
        try:
            return float(np.linalg.norm(tcp - _to_np_xyz(target_cube.pose.p)))
        except Exception:
            return None
