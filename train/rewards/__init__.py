"""Per-task shaped reward functions for RL training.

The RoboMME benchmark's `compute_dense_reward` methods are zero stubs (the
benchmark grades via success/fail flags from `evaluate()`). For RL training
from scratch we need shaped rewards; those live here, one module per task,
and are pulled in by env_id via `make_reward()`.

Tasks without a shaped reward fall back to whatever the benchmark returns
(after F1, that's `info["success"] - info["fail"]`, i.e. sparse).
"""

from __future__ import annotations

from typing import Optional


def make_reward(env_id: str):
    """Return a shaped reward instance for env_id, or None if none registered.

    A shaped reward instance must expose:
        reset(unwrapped) -> None
        step(unwrapped, info: dict) -> float
    """
    if env_id == "BinFill":
        from .binfill import BinFillReward
        return BinFillReward()
    return None
