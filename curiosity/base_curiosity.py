"""Interface for curiosity / intrinsic-reward modules (e.g. ICM, M-ICM).

Memory-conditioned variants can read the optional `memory_state` argument.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional


class BaseCuriosity(ABC):
    @abstractmethod
    def compute_intrinsic_reward(
        self,
        obs: Any,
        action: Any,
        next_obs: Any,
        memory_state: Optional[Any] = None,
    ) -> float:
        """Return the per-step intrinsic reward signal."""

    @abstractmethod
    def update(self, batch: Mapping[str, Any]) -> Mapping[str, float]:
        """Update internal parameters from a batch of transitions.

        Returns a dict of scalar training stats for logging.
        """
