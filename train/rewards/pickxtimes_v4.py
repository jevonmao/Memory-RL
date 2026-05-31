"""V4 reward for PickXtimes — pure terminal / task-completion only.

Removes ALL proximity shaping and potential-based reach signals. The agent
only earns reward when it actually completes a subgoal, pickup, or the
terminal success. Avoids the V1/V3 reward-hacking failure (where the policy
exploited proximity to ~0.05/step ≈ +50/episode without grasping anything)
AND the V2 problem (no exploration gradient at all).

V4 budget over a 1500-step episode:
  step penalty:          -15           (constant — punishes hovering)
  grasp hint:            +0.5/step IF close + closing (max ~+50, but only fires
                                                       when actually attempting)
  subgoal advance:       +10 × delta   (every 'timestep' advance)
  pickup-count anchor:   +5 each
  TERMINAL_SUCCESS:      +100
  TERMINAL_FAIL:         -10

Crucial design choice: there is NO distance-based reward at all. The agent
cannot earn reward by being near things. The only way to escape the -15
step-penalty floor is to:
  - close the gripper near a cube (grasp_hint), OR
  - complete a subgoal (+10), OR
  - complete an episode successfully (+100)

Intended pairing: BC warm-start (so the policy starts near-expert in action
space) + lower initial log_std (so it actually executes BC-near actions) +
auxiliary BC loss during PPO (so it doesn't drift far from expert during
RL fine-tuning).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .pickxtimes_v2 import _to_np_xyz, _flag  # noqa: F401  (helpers)


class PickXtimesRewardV4:
    """V4 — terminal-only reward; no proximity / potential shaping."""

    STEP_PENALTY     = -0.01
    SUBGOAL_BONUS    = 10.0     # per-subgoal completion (timestep advance)
    PICKUP_BONUS     =  5.0     # per pickup completed
    GRASP_HINT       =  0.5     # per-step when close to cube AND gripper closing
    GRASP_HINT_DIST  =  0.06
    TERMINAL_SUCCESS = 100.0    # 2× V2/V3 — make full-task completion the dominant signal
    TERMINAL_FAIL    = -10.0

    def __init__(self) -> None:
        self._prev_timestep:    Optional[int]   = None
        self._prev_pickup_done: int             = 0

    # ----------------------------------------------------------------
    def reset(self, unwrapped) -> None:
        self._prev_timestep    = getattr(unwrapped, "timestep", 0)
        self._prev_pickup_done = self._pickups_completed(unwrapped)

    # ----------------------------------------------------------------
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

        # Grasp hint — encourage closing the gripper near the target cube.
        # NOT a proximity reward (no distance gradient); only fires when close
        # AND gripper is closed. Small bonus to seed the grasp behavior.
        if self._is_pickup_phase(unwrapped):
            dist = self._tcp_to_target_cube(unwrapped)
            if dist is not None and dist < self.GRASP_HINT_DIST:
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

    def _tcp_to_target_cube(self, unwrapped) -> Optional[float]:
        try:
            tcp = _to_np_xyz(unwrapped.agent.tcp_pose.p)
            tc  = getattr(unwrapped, "target_cube", None)
            if tc is None:
                return None
            return float(np.linalg.norm(tcp - _to_np_xyz(tc.pose.p)))
        except Exception:
            return None
