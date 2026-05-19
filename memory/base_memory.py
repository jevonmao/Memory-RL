"""Interface for memory modules (e.g. Past-Token Prediction memory).

Concrete memory implementations live in sibling files (added later by the
teammate owning the memory module). Training/eval code should depend only on
this abstract surface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMemory(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Clear episodic state. Called at the start of every episode."""

    @abstractmethod
    def update(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        done: bool,
    ) -> None:
        """Consume a single transition."""

    @abstractmethod
    def get_state(self) -> Any:
        """Return the current memory representation (e.g. tensor/dict)."""
