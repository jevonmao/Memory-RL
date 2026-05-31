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
    if env_id == "PickXtimes":
        # Reward version is selected by the ROBOMME_REWARD_VERSION env var
        # to allow training the v1 and v2 reward schedules in parallel
        # under different OUTDIR roots without code edits.
        import os
        v = os.environ.get("ROBOMME_REWARD_VERSION", "v1")
        if v == "v2":
            from .pickxtimes_v2 import PickXtimesRewardV2
            return PickXtimesRewardV2()
        if v == "v3":
            from .pickxtimes_v3 import PickXtimesRewardV3
            return PickXtimesRewardV3()
        if v == "v4":
            from .pickxtimes_v4 import PickXtimesRewardV4
            return PickXtimesRewardV4()
        from .pickxtimes import PickXtimesReward
        return PickXtimesReward()
    return None
