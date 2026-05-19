"""Reproducibility + bookkeeping utilities shared by training/eval scripts."""
from __future__ import annotations

import datetime as _dt
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_run_dir(
    base: str | os.PathLike,
    task_name: str,
    seed: int,
    tag: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Path:
    timestamp = timestamp or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [task_name, f"seed{seed}", timestamp]
    if tag:
        parts.append(tag)
    run = Path(base) / "_".join(parts)
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "trajectories").mkdir(parents=True, exist_ok=True)
    (run / "tb").mkdir(parents=True, exist_ok=True)
    return run


def save_run_config(run_dir: Path, cfg: Dict[str, Any]) -> None:
    import yaml

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def merge_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out
