"""Evaluation metrics used by training/evaluate.py.

Conventions:
- "success" is read from `info["is_success"]` (Gymnasium standard). If absent
  we fall back to `info["success"]`, else treat the episode as successful iff
  it terminated (not truncated) with positive return.
- "steps_to_completion" is the step index of the first success signal in the
  episode, or the episode length if no success signal was emitted.
- "redundancy_score" measures repeated state visits. Definition:
      redundancy = (visits - unique_states) / max(visits, 1)
  which is equivalent to "fraction of steps spent re-visiting an already-seen
  state cluster". State identity is taken from `info["state_id"]` when the
  env exposes one; otherwise we hash a coarsely-quantized flattened
  observation (3 decimal places) so near-duplicate continuous states cluster.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class EpisodeRecord:
    observations: Sequence[Any]           # length T+1 (incl. initial)
    rewards: Sequence[float]              # length T
    terminated: bool
    truncated: bool
    infos: Sequence[Dict[str, Any]]       # length T
    state_ids: Optional[Sequence[Any]] = None  # length T+1 if available


def _episode_success(ep: EpisodeRecord) -> bool:
    for info in ep.infos:
        if isinstance(info, dict):
            if info.get("is_success"):
                return True
            if info.get("success"):
                return True
    if ep.terminated and not ep.truncated and sum(ep.rewards) > 0:
        return True
    return False


def _steps_to_completion(ep: EpisodeRecord) -> int:
    for i, info in enumerate(ep.infos):
        if isinstance(info, dict) and (info.get("is_success") or info.get("success")):
            return i + 1
    return len(ep.rewards)


def success_rate(eps: Iterable[EpisodeRecord]) -> float:
    eps = list(eps)
    if not eps:
        return 0.0
    return float(np.mean([_episode_success(e) for e in eps]))


def average_return(eps: Iterable[EpisodeRecord]) -> float:
    eps = list(eps)
    if not eps:
        return 0.0
    return float(np.mean([float(np.sum(e.rewards)) for e in eps]))


def average_episode_length(eps: Iterable[EpisodeRecord]) -> float:
    eps = list(eps)
    if not eps:
        return 0.0
    return float(np.mean([len(e.rewards) for e in eps]))


def average_steps_to_completion(eps: Iterable[EpisodeRecord]) -> float:
    eps = list(eps)
    succ = [_steps_to_completion(e) for e in eps if _episode_success(e)]
    if not succ:
        return float("nan")
    return float(np.mean(succ))


def _quantize_obs(o: Any, decimals: int = 3) -> bytes:
    if isinstance(o, dict):
        parts = []
        for k in sorted(o.keys()):
            arr = np.asarray(o[k]).flatten()
            parts.append(k.encode())
            parts.append(np.round(arr.astype(np.float64), decimals).tobytes())
        return b"|".join(parts)
    arr = np.asarray(o).flatten()
    return np.round(arr.astype(np.float64), decimals).tobytes()


def redundancy_score(ep: EpisodeRecord, decimals: int = 3) -> float:
    """Fraction of steps that revisit an already-seen state cluster.

    Uses `state_ids` if provided, else `info["state_id"]` per step, else a
    hash of the rounded flattened observation. Returns value in [0, 1).
    """
    if ep.state_ids is not None and len(ep.state_ids) > 0:
        ids = [str(s) for s in ep.state_ids]
    else:
        ids: List[str] = []
        for info in ep.infos:
            if isinstance(info, dict) and "state_id" in info:
                ids.append(str(info["state_id"]))
                break
        if len(ids) == len(ep.infos) and ids:
            ids = [str(i.get("state_id")) for i in ep.infos]
        else:
            ids = [_quantize_obs(o, decimals).hex() for o in ep.observations]
    visits = len(ids)
    if visits <= 1:
        return 0.0
    unique = len(set(ids))
    return float((visits - unique) / visits)


def summarize(eps: Sequence[EpisodeRecord]) -> Dict[str, float]:
    return {
        "success_rate": success_rate(eps),
        "average_return": average_return(eps),
        "average_episode_length": average_episode_length(eps),
        "average_steps_to_completion": average_steps_to_completion(eps),
        "average_redundancy_score": float(np.mean([redundancy_score(e) for e in eps])) if eps else 0.0,
        "num_episodes": len(eps),
    }
