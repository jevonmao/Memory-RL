"""Shaped reward for the BinFill task.

Reward shape:
  - step penalty            : -0.01     (encourages finishing)
  - in-bin progress         : +1.0      per newly-binned target-color cube
  - potential-based reach   : +0.5 * (prev_dist - cur_dist), dist = TCP→nearest needed cube
                              (switches to TCP→button once all target cubes are placed)
  - terminal                : +10.0 on info["success"], -5.0 on info["fail"]

Runtime attributes used (all set by BinFill._load_scene / _initialize_episode):
  unwrapped.agent.tcp_pose.p
  unwrapped.{red,blue,green}_cubes           — lists of cube actors
  unwrapped.{red,blue,green}_cubes_target_number
  unwrapped.{red,blue,green}_cubes_in_bin    — incremented during step by
      is_any_obj_dropped_onto_delete in subgoal_evaluate_func.py
  unwrapped.button.pose.p
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _to_np_xyz(p: Any) -> np.ndarray:
    """Pull an (x, y, z) numpy vector out of a torch/sapien position field."""
    if hasattr(p, "cpu"):
        p = p.cpu().numpy()
    arr = np.asarray(p).reshape(-1)
    return arr[:3].astype(np.float32)


class BinFillReward:
    """Per-step shaped reward for BinFill (stateful across an episode)."""

    STEP_PENALTY     = -0.01
    BIN_PROGRESS     =  1.0
    REACH_COEF       =  0.5
    TERMINAL_SUCCESS = 10.0
    TERMINAL_FAIL    = -5.0

    def __init__(self) -> None:
        self._prev_in_bin: tuple[int, int, int] = (0, 0, 0)
        self._prev_dist:   Optional[float]      = None

    # ------------------------------------------------------------------
    def reset(self, unwrapped) -> None:
        self._prev_in_bin = self._current_in_bin(unwrapped)
        self._prev_dist   = self._min_target_dist(unwrapped)

    def step(self, unwrapped, info: dict) -> float:
        r = self.STEP_PENALTY

        # In-bin progress (diff against last step's counts; per-cube increments
        # are monotonic so max(0, ...) is a safety net only).
        cur = self._current_in_bin(unwrapped)
        delta = sum(max(0, c - p) for c, p in zip(cur, self._prev_in_bin))
        r += self.BIN_PROGRESS * delta
        self._prev_in_bin = cur

        # Potential-based reaching reward.
        cur_dist = self._min_target_dist(unwrapped)
        if cur_dist is not None and self._prev_dist is not None:
            r += self.REACH_COEF * (self._prev_dist - cur_dist)
        self._prev_dist = cur_dist

        # Terminal bonuses (info flags come from BinFill.evaluate()).
        if _flag(info, "success"):
            r += self.TERMINAL_SUCCESS
        if _flag(info, "fail"):
            r += self.TERMINAL_FAIL

        return float(r)

    # ------------------------------------------------------------------
    @staticmethod
    def _current_in_bin(unwrapped) -> tuple[int, int, int]:
        return (
            int(_scalar(unwrapped.red_cubes_in_bin)),
            int(_scalar(unwrapped.blue_cubes_in_bin)),
            int(_scalar(unwrapped.green_cubes_in_bin)),
        )

    @staticmethod
    def _min_target_dist(unwrapped) -> Optional[float]:
        """Distance from TCP to the nearest still-needed target-color cube.

        Once every target cube is placed, the target becomes the button —
        which carries the agent into the final "press the button" subgoal.
        Returns None only if the scene has no actors at all (shouldn't happen).
        """
        try:
            tcp = _to_np_xyz(unwrapped.agent.tcp_pose.p)
        except Exception:
            return None

        candidates = []
        triples = [
            (unwrapped.red_cubes,   unwrapped.red_cubes_target_number,   unwrapped.red_cubes_in_bin),
            (unwrapped.blue_cubes,  unwrapped.blue_cubes_target_number,  unwrapped.blue_cubes_in_bin),
            (unwrapped.green_cubes, unwrapped.green_cubes_target_number, unwrapped.green_cubes_in_bin),
        ]
        for cube_list, target, in_bin in triples:
            remaining = max(0, int(_scalar(target)) - int(_scalar(in_bin)))
            if remaining <= 0:
                continue
            # The first `remaining` cubes are the still-needed ones; the rest
            # are already binned (the env removes them on drop).
            candidates.extend(cube_list[:remaining])

        if not candidates:
            # All target cubes placed — guide TCP toward the button.
            try:
                return float(np.linalg.norm(tcp - _to_np_xyz(unwrapped.button.pose.p)))
            except Exception:
                return None

        dists = []
        for c in candidates:
            try:
                dists.append(float(np.linalg.norm(tcp - _to_np_xyz(c.pose.p))))
            except Exception:
                continue
        if not dists:
            return None
        return min(dists)


# ----------------------------------------------------------------------
def _scalar(x: Any) -> float:
    """Coerce torch/numpy scalars (possibly shape (1,)) to Python float."""
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
