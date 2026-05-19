"""Per-step trajectory logger.

Saves episodes as compressed NPZ files plus a JSONL index. NPZ is chosen
because observations are typically numeric arrays (possibly dict-of-arrays);
it is fast to load for PTP memory training and ICM transition fitting.

File layout under `out_dir`:
    index.jsonl                  one row per saved episode (metadata only)
    ep_000000_<task>_<seed>.npz  per-episode arrays

Each NPZ contains arrays of shape (T, ...) for observations, actions,
rewards, dones, truncateds, plus a JSON-encoded `info_list` for per-step
info dicts (kept as JSON because info schemas can be heterogeneous).

Loading: see `load_trajectories`.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np


def _to_array(x):
    return np.asarray(x)


def _stack_obs(seq: List[Any]):
    """Stack a sequence of observations, handling dict observations."""
    if len(seq) == 0:
        return np.empty((0,))
    first = seq[0]
    if isinstance(first, dict):
        return {k: np.stack([_to_array(o[k]) for o in seq]) for k in first.keys()}
    return np.stack([_to_array(o) for o in seq])


@dataclass
class _EpisodeBuffer:
    episode_id: int
    task_name: str
    seed: Optional[int]
    obs: List[Any] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    truncateds: List[bool] = field(default_factory=list)
    infos: List[Dict[str, Any]] = field(default_factory=list)


class TrajectoryLogger:
    """Streaming trajectory writer.

    Usage:
        logger = TrajectoryLogger(out_dir, task_name="...", seed=0)
        obs, _ = env.reset()
        logger.start_episode(obs)
        while not done:
            action = policy(obs)
            next_obs, r, term, trunc, info = env.step(action)
            logger.record(action, r, term, trunc, info, next_obs)
            obs = next_obs
        logger.end_episode()
    """

    def __init__(self, out_dir: str | Path, task_name: str, seed: Optional[int] = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.task_name = task_name
        self.seed = seed
        self._index_path = self.out_dir / "index.jsonl"
        self._index_fp = open(self._index_path, "a")
        self._next_id = self._scan_next_id()
        self._cur: Optional[_EpisodeBuffer] = None

    def _scan_next_id(self) -> int:
        if not self._index_path.exists():
            return 0
        n = 0
        with open(self._index_path) as f:
            for _ in f:
                n += 1
        return n

    def start_episode(self, initial_obs: Any) -> None:
        if self._cur is not None:
            raise RuntimeError("start_episode called while an episode is open")
        self._cur = _EpisodeBuffer(
            episode_id=self._next_id, task_name=self.task_name, seed=self.seed
        )
        self._cur.obs.append(initial_obs)

    def record(
        self,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
        next_obs: Any,
    ) -> None:
        if self._cur is None:
            raise RuntimeError("record called without start_episode")
        self._cur.actions.append(action)
        self._cur.rewards.append(float(reward))
        self._cur.dones.append(bool(terminated))
        self._cur.truncateds.append(bool(truncated))
        self._cur.infos.append(_sanitize_info(info))
        self._cur.obs.append(next_obs)

    def end_episode(self) -> Optional[Path]:
        if self._cur is None:
            return None
        ep = self._cur
        self._cur = None
        path = self.out_dir / f"ep_{ep.episode_id:06d}_{ep.task_name}_seed{ep.seed}.npz"
        T = len(ep.actions)
        arrays: Dict[str, Any] = {}
        obs_stacked = _stack_obs(ep.obs)
        if isinstance(obs_stacked, dict):
            for k, v in obs_stacked.items():
                arrays[f"obs__{k}"] = v
            arrays["__obs_keys__"] = np.array(list(obs_stacked.keys()))
        else:
            arrays["obs"] = obs_stacked
        arrays["actions"] = _to_array(ep.actions) if T else np.empty((0,))
        arrays["rewards"] = np.asarray(ep.rewards, dtype=np.float32)
        arrays["dones"] = np.asarray(ep.dones, dtype=bool)
        arrays["truncateds"] = np.asarray(ep.truncateds, dtype=bool)
        arrays["info_json"] = np.array(json.dumps(ep.infos))
        np.savez_compressed(path, **arrays)

        meta = {
            "episode_id": ep.episode_id,
            "path": str(path.name),
            "task_name": ep.task_name,
            "seed": ep.seed,
            "length": T,
            "return": float(sum(ep.rewards)),
            "terminated": bool(ep.dones[-1]) if ep.dones else False,
            "truncated": bool(ep.truncateds[-1]) if ep.truncateds else False,
            "saved_at": time.time(),
        }
        self._index_fp.write(json.dumps(meta) + "\n")
        self._index_fp.flush()
        self._next_id += 1
        return path

    def close(self) -> None:
        if self._cur is not None:
            self.end_episode()
        self._index_fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _sanitize_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """Drop values that aren't JSON-serializable; coerce numpy scalars."""
    out: Dict[str, Any] = {}
    for k, v in (info or {}).items():
        if isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        elif isinstance(v, np.ndarray):
            if v.size <= 64:
                out[k] = v.tolist()
        elif isinstance(v, (int, float, str, bool, list, dict)) or v is None:
            try:
                json.dumps(v)
                out[k] = v
            except TypeError:
                continue
    return out


def load_trajectories(out_dir: str | Path) -> Iterator[Dict[str, Any]]:
    """Yield dicts for each saved episode, lazily reading NPZs."""
    out_dir = Path(out_dir)
    idx = out_dir / "index.jsonl"
    if not idx.exists():
        return
    with open(idx) as f:
        for line in f:
            meta = json.loads(line)
            data = np.load(out_dir / meta["path"], allow_pickle=False)
            entry = dict(meta)
            entry["data"] = {k: data[k] for k in data.files}
            yield entry
