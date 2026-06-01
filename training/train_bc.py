"""Behavioral Cloning from RoboMME expert demonstrations.

Trains an SB3 MlpPolicy on (obs, action) pairs via supervised NLL loss,
then pretrains the value head on Monte Carlo returns, and saves as a PPO
.zip for warm-starting PPO fine-tuning.

RoboMME is NOT imported unless --collect is used. Data loading and all
training (BC + value-head pretraining) are pure numpy/torch/SB3 operations.

Data sources (pick one):
    --h5 <file>       RoboMME record HDF5 file (e.g. record_dataset_BinFill.h5).
                      Primary path for the Modal workflow.
    --dataset <dir>   Directory of NPZ trajectories from TrajectoryLogger.
    --collect         Collect demos by running the RoboMME env (needs SAPIEN).

Loading the saved model for PPO fine-tuning:
    python training/train_ppo.py --config configs/ppo.yaml \\
        --bc_checkpoint <run_dir>/checkpoints/bc_final.zip

Example (Modal H5 workflow):
    python training/train_bc.py --config configs/bc.yaml \\
        --h5 data/record_dataset_BinFill.h5 --task BinFill

Example (NPZ dataset):
    python training/train_bc.py --config configs/bc.yaml \\
        --dataset logs/my_demos/trajectories
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.trajectory_logger import TrajectoryLogger, load_trajectories
from training.utils import (
    build_run_dir,
    load_yaml,
    merge_overrides,
    save_run_config,
    set_global_seed,
)


# ---------------------------------------------------------------------------
# NPZ dataset
# ---------------------------------------------------------------------------

class BCDataset(Dataset):
    """(obs, action) pairs from a directory of NPZ trajectory files."""

    def __init__(self, trajectory_dir: str | Path):
        trajectory_dir = Path(trajectory_dir)
        obs_list: List[np.ndarray] = []
        act_list: List[np.ndarray] = []
        n_eps = 0

        for ep in load_trajectories(trajectory_dir):
            data = ep["data"]
            actions = data.get("actions")
            if actions is None or len(actions) == 0:
                continue
            obs_arr = data.get("obs")
            if obs_arr is None:
                continue  # dict-obs format not supported for MlpPolicy
            T = len(actions)
            obs_list.append(obs_arr[:T].astype(np.float32))
            act_list.append(actions.astype(np.float32))
            n_eps += 1

        if n_eps == 0:
            raise ValueError(
                f"No usable trajectories in {trajectory_dir}. "
                "Ensure flatten_obs=True and non-empty actions arrays."
            )

        self._obs = np.concatenate(obs_list, axis=0)
        self._actions = np.concatenate(act_list, axis=0)
        print(f"[BCDataset] {n_eps} episodes  {len(self._obs)} transitions  "
              f"obs={self._obs.shape[1:]}  act={self._actions.shape[1:]}")

    @property
    def obs_dim(self) -> int:
        return self._obs.shape[1]

    @property
    def act_dim(self) -> int:
        return self._actions.shape[1]

    def __len__(self) -> int:
        return len(self._obs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(self._obs[idx], dtype=torch.float32),
            torch.as_tensor(self._actions[idx], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# HDF5 dataset
# ---------------------------------------------------------------------------

# Observation keys used when loading from the RoboMME record HDF5 file.
# The H5 stores single-frame arrays ("eef_state"), whereas the Gymnasium env
# wrapper returns windowed lists ("eef_state_list"). These are different names
# for the same physical signal; do not change one without the other.
_FLATTEN_KEYS = ("eef_state", "joint_state", "gripper_state")


def inspect_h5(h5_path: str | Path, max_depth: int = 3) -> Dict[str, Any]:
    """Return a nested dict describing an HDF5 file's structure.

    Run this on Modal before BC training to verify the obs key layout:
        modal run modal_app/app.py::inspect_h5
    """
    import h5py

    def _walk(node, depth: int) -> Any:
        if depth == 0:
            return "..."
        if isinstance(node, h5py.Dataset):
            return {"shape": tuple(node.shape), "dtype": str(node.dtype)}
        return {k: _walk(node[k], depth - 1) for k in list(node.keys())[:20]}

    with h5py.File(h5_path, "r") as f:
        return _walk(f, max_depth)


class H5BCDataset(Dataset):
    """(obs, action) pairs from a RoboMME record HDF5 file.

    Actual H5 layout (two-level: episode → timestep):
        episode_0/
          setup/                           ← metadata, skipped
          timestep_0/
            action/joint_action  (8,) f64  ← expert action
            obs/eef_state        (6,) f32
            obs/joint_state      (7,) f32
            obs/gripper_state    (2,) f32
            info/is_completed    ()   bool
          timestep_1/
          ...
        episode_1/
          ...

    Iterates all episodes → all timesteps, extracts one (15,)/(8,) pair per
    timestep, and stacks everything into a flat (N, obs_dim)/(N, act_dim) array.
    max_episodes limits the number of episode_N groups processed.
    """

    def __init__(
        self,
        h5_path: str | Path,
        flatten_keys: Sequence[str] = _FLATTEN_KEYS,
        max_episodes: Optional[int] = None,
        h5_obs_prefix: Optional[str] = None,
    ):
        import h5py

        self._obs_keys = tuple(flatten_keys)
        obs_list: List[np.ndarray] = []
        act_list: List[np.ndarray] = []
        n_steps = n_skipped = n_episodes = 0
        first_skip_reason: Optional[str] = None

        with h5py.File(h5_path, "r") as f:
            episode_groups = _find_traj_groups(f)
            for ep_name, ep_grp in episode_groups:
                if max_episodes is not None and n_episodes >= max_episodes:
                    break
                timesteps = _find_timestep_groups(ep_grp)
                if not timesteps:
                    continue
                for ts_name, ts_grp in timesteps:
                    obs, act, skip_reason = _load_h5_timestep(
                        ts_grp, self._obs_keys, h5_obs_prefix
                    )
                    if obs is None:
                        n_skipped += 1
                        if first_skip_reason is None:
                            first_skip_reason = f"'{ep_name}/{ts_name}': {skip_reason}"
                        continue
                    obs_list.append(obs)
                    act_list.append(act)
                    n_steps += 1
                n_episodes += 1

        if n_steps == 0:
            hint = f"\nFirst skip reason: {first_skip_reason}" if first_skip_reason else ""
            raise ValueError(
                f"No usable timesteps in {h5_path}. "
                f"Run inspect_h5() with max_depth=5 to check shapes.{hint}"
            )

        self._obs = np.stack(obs_list, axis=0).astype(np.float32)      # (N, obs_dim)
        self._actions = np.stack(act_list, axis=0).astype(np.float32)  # (N, act_dim)
        print(f"[H5BCDataset] episodes={n_episodes}  steps={n_steps}  skipped={n_skipped}  "
              f"obs={self._obs.shape[1:]}  act={self._actions.shape[1:]}")

    @property
    def obs_dim(self) -> int:
        return self._obs.shape[1]

    @property
    def act_dim(self) -> int:
        return self._actions.shape[1]

    def __len__(self) -> int:
        return len(self._obs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(self._obs[idx], dtype=torch.float32),
            torch.as_tensor(self._actions[idx], dtype=torch.float32),
        )


def _find_traj_groups(f) -> List[Tuple[str, Any]]:
    """Return (name, group) pairs for top-level episode groups, sorted numerically."""
    import h5py
    import re

    root = f["data"] if "data" in f else f
    ep_pat = re.compile(r"^(traj|demo|episode|ep)_?\d+$", re.IGNORECASE)
    groups = [
        (k, root[k])
        for k in sorted(
            root.keys(),
            key=lambda s: int(re.search(r"\d+", s).group()) if re.search(r"\d+", s) else 0,
        )
        if isinstance(root[k], h5py.Group) and ep_pat.match(k)
    ]
    if not groups:
        groups = [(k, root[k]) for k in sorted(root.keys()) if isinstance(root[k], h5py.Group)]
    return groups


def _find_timestep_groups(ep_grp) -> List[Tuple[str, Any]]:
    """Return (name, group) pairs for timestep_N subgroups within one episode."""
    import h5py
    import re

    ts_pat = re.compile(r"^timestep_?\d+$", re.IGNORECASE)
    return [
        (k, ep_grp[k])
        for k in sorted(
            ep_grp.keys(),
            key=lambda s: int(re.search(r"\d+", s).group()) if re.search(r"\d+", s) else 0,
        )
        if isinstance(ep_grp[k], h5py.Group) and ts_pat.match(k)
    ]


def _load_h5_timestep(
    grp,
    flatten_keys: Sequence[str],
    obs_prefix: Optional[str],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """Read one single-step group → (obs_1D [obs_dim], action_1D [act_dim], None).

    RoboMME stores each recorded timestep as its own H5 group where every
    array is 1-D (one vector per signal). Returns (None, None, reason) if
    the group cannot be parsed.
    """
    import h5py

    # ---- action (1-D vector) ------------------------------------------------
    act = None
    act_source = None
    for key in ("actions", "action", "act"):
        if key not in grp:
            continue
        val = grp[key]
        if isinstance(val, h5py.Dataset):
            a = np.asarray(val, dtype=np.float32)
            if a.ndim == 1:
                act = a
                act_source = key
        elif isinstance(val, h5py.Group):
            # RoboMME wraps action types in a subgroup; use joint_action.
            for akey in ("joint_action", "joint_actions", "actions"):
                if akey in val and isinstance(val[akey], h5py.Dataset):
                    a = np.asarray(val[akey], dtype=np.float32)
                    if a.ndim == 1:
                        act = a
                        act_source = f"{key}/{akey}"
                        break
        if act is not None:
            break

    if act is None:
        return None, None, "no 1-D action array found under 'action/joint_action'"

    # ---- observations (1-D vectors, concatenated) ---------------------------
    obs_containers = []
    if obs_prefix and obs_prefix in grp:
        obs_containers.append(grp[obs_prefix])
    for candidate in ("obs", "observations", "observation"):
        if candidate in grp:
            obs_containers.append(grp[candidate])
            break
    obs_containers.append(grp)  # fallback: keys at group root

    parts: List[np.ndarray] = []
    missing: List[str] = []
    for key in flatten_keys:
        arr = None
        for container in obs_containers:
            if key in container and isinstance(container[key], h5py.Dataset):
                arr = np.asarray(container[key], dtype=np.float32).flatten()
                break
        if arr is None:
            missing.append(key)
        else:
            parts.append(arr)

    if not parts:
        return None, None, (
            f"none of the obs keys {list(flatten_keys)} found "
            f"(missing: {missing})"
        )

    obs = np.concatenate(parts, axis=0)  # (obs_dim,)
    return obs, act, None


# ---------------------------------------------------------------------------
# Policy construction (no RoboMME needed)
# ---------------------------------------------------------------------------

def _build_policy_from_spaces(
    cfg: Dict[str, Any],
    obs_dim: int,
    act_dim: int,
) -> Any:
    """Build an SB3 PPO model using a DummyEnv with fixed obs/act dims.

    No SAPIEN or RoboMME required. The saved .zip is architecturally identical
    to what train_ppo.py produces, so PPO.load() + load_state_dict() work
    without any shape mismatch.
    """
    import gymnasium as gym
    from gymnasium import spaces as gym_spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    # SB3 2.3+ asserts that action spaces have finite bounds.
    # Joint angles fit comfortably within [-10, 10] rad; obs range is similar.
    obs_space = gym_spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
    act_space = gym_spaces.Box(low=-10.0, high=10.0, shape=(act_dim,), dtype=np.float32)

    class _DummyEnv(gym.Env):
        observation_space = obs_space
        action_space = act_space

        def reset(self, **kwargs):
            return obs_space.sample(), {}

        def step(self, action):
            return obs_space.sample(), 0.0, False, False, {}

    n_steps = cfg.get("n_steps", 512)
    batch_size = cfg.get("batch_size", 256)
    if n_steps % batch_size != 0:
        batch_size = next(bs for bs in range(batch_size, 0, -1) if n_steps % bs == 0)

    vec_env = DummyVecEnv([_DummyEnv])
    model = PPO(
        policy=cfg.get("policy", "MlpPolicy"),
        env=vec_env,
        learning_rate=cfg.get("learning_rate", 3e-4),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=cfg.get("n_epochs", 8),
        gamma=cfg.get("gamma", 0.99),
        gae_lambda=cfg.get("gae_lambda", 0.95),
        clip_range=cfg.get("clip_range", 0.2),
        ent_coef=cfg.get("ent_coef", 0.0),
        vf_coef=cfg.get("vf_coef", 0.5),
        max_grad_norm=cfg.get("max_grad_norm", 0.5),
        seed=cfg.get("seed", 0),
        device=cfg.get("device", "auto"),
        policy_kwargs=cfg.get("policy_kwargs") or {},
        verbose=0,
    )
    vec_env.close()
    return model


# ---------------------------------------------------------------------------
# BC training loop
# ---------------------------------------------------------------------------

def train_bc(
    dataset: "BCDataset | H5BCDataset",
    model: Any,
    cfg: Dict[str, Any],
    run_dir: Path,
) -> None:
    """Train model.policy actor branch with supervised NLL loss.

    Only mlp_extractor.policy_net + action_net + log_std are meaningfully
    updated. The value branch (mlp_extractor.value_net + value_net) receives
    gradient through shared input preprocessing, but its parameters will be
    properly initialized by pretrain_value_head() afterwards.
    """
    device = model.device
    policy = model.policy
    policy.set_training_mode(True)

    batch_size = cfg.get("bc_batch_size", 256)
    n_epochs = cfg.get("bc_epochs", 100)
    lr = cfg.get("bc_lr", 3e-4)
    l2_coef = cfg.get("bc_l2_coef", 1e-4)
    max_grad_norm = cfg.get("max_grad_norm", 0.5)
    log_every = cfg.get("bc_log_every", 10)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=False, num_workers=0)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=l2_coef)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_loss = float("inf")
    best_path = run_dir / "checkpoints" / "bc_best.zip"
    log_entries: List[Dict[str, Any]] = []

    print(f"[train_bc] transitions={len(dataset)}  batch={batch_size}  "
          f"epochs={n_epochs}  lr={lr}  device={device}")

    for epoch in range(1, n_epochs + 1):
        total_nll = total_ent = 0.0
        n_batches = 0

        for obs_b, act_b in loader:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)

            # evaluate_actions → (values, log_probs, entropy)
            # log_prob = Σ_i log N(a_i | μ_i(s), σ_i)  — sum over action dims
            _, log_probs, entropy = policy.evaluate_actions(obs_b, act_b)
            loss = -log_probs.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            total_nll += loss.item()
            total_ent += entropy.mean().item()
            n_batches += 1

        scheduler.step()
        avg_nll = total_nll / max(n_batches, 1)
        avg_ent = total_ent / max(n_batches, 1)
        current_lr = scheduler.get_last_lr()[0]
        log_entries.append({"epoch": epoch, "nll": avg_nll,
                            "entropy": avg_ent, "lr": current_lr})

        if epoch == 1 or epoch % log_every == 0 or epoch == n_epochs:
            print(f"[train_bc] epoch {epoch:4d}/{n_epochs}  "
                  f"nll={avg_nll:.4f}  ent={avg_ent:.4f}  lr={current_lr:.2e}")

        if avg_nll < best_loss:
            best_loss = avg_nll
            model.save(str(best_path))

    with open(run_dir / "bc_train_log.json", "w") as f:
        json.dump({"bc": log_entries}, f, indent=2)
    print(f"[train_bc] best nll={best_loss:.4f}  best model → {best_path}")


# ---------------------------------------------------------------------------
# Value head pretraining
# ---------------------------------------------------------------------------

def _compute_mc_returns(rewards: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """Backward-accumulate discounted returns G_t = r_t + γ·G_{t+1}."""
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    G = 0.0
    for t in reversed(range(T)):
        G = float(rewards[t]) + gamma * G
        returns[t] = G
    return returns


def _flush_vf_episode(
    episodes: list,
    obs_buf: List[np.ndarray],
    gamma: float,
    success: bool,
) -> None:
    """Stack buffered obs, assign terminal reward if success, append episode."""
    if not obs_buf:
        return
    obs_arr = np.stack(obs_buf, axis=0)       # (T, obs_dim)
    T = len(obs_arr)
    rewards = np.zeros(T, dtype=np.float32)
    if success:
        rewards[-1] = 1.0                      # terminal reward on task completion
    episodes.append((obs_arr, _compute_mc_returns(rewards, gamma)))


def _load_vf_episodes_from_h5(
    h5_path: str | Path,
    flatten_keys: Sequence[str] = _FLATTEN_KEYS,
    gamma: float = 0.99,
    max_episodes: Optional[int] = None,
    h5_obs_prefix: Optional[str] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return [(obs [T, D], mc_returns [T]), ...] for value head pretraining.

    Uses the two-level episode/timestep structure of the H5 file: each
    episode_N group is one demonstration. Within each episode the last
    timestep's info/is_completed flag determines whether the terminal
    reward is 1.0 (success) or 0 (truncated/failed).
    """
    import h5py

    episodes: List[Tuple[np.ndarray, np.ndarray]] = []

    with h5py.File(h5_path, "r") as f:
        episode_groups = _find_traj_groups(f)

        for ep_name, ep_grp in episode_groups:
            if max_episodes is not None and len(episodes) >= max_episodes:
                break

            timesteps = _find_timestep_groups(ep_grp)
            if not timesteps:
                continue

            obs_buf: List[np.ndarray] = []
            episode_success = False

            for ts_name, ts_grp in timesteps:
                obs, _, _ = _load_h5_timestep(ts_grp, flatten_keys, h5_obs_prefix)
                if obs is None:
                    continue
                obs_buf.append(obs)
                # Update success from each step; the last step's value is kept
                try:
                    episode_success = bool(np.asarray(ts_grp["info"]["is_completed"]))
                except Exception:
                    pass

            _flush_vf_episode(episodes, obs_buf, gamma, success=episode_success)

    return episodes


def _load_vf_episodes_from_npz(
    trajectory_dir: str | Path,
    gamma: float = 0.99,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return [(obs [T,D], mc_returns [T]), ...] from NPZ trajectory files."""
    episodes = []
    for ep in load_trajectories(trajectory_dir):
        data = ep["data"]
        obs_arr = data.get("obs")
        rewards = data.get("rewards")
        if obs_arr is None or rewards is None:
            continue
        T = len(rewards)
        obs = obs_arr[:T].astype(np.float32)
        mc_returns = _compute_mc_returns(rewards.astype(np.float32), gamma)
        episodes.append((obs, mc_returns))
    return episodes


def pretrain_value_head(
    episodes: List[Tuple[np.ndarray, np.ndarray]],
    model: Any,
    cfg: Dict[str, Any],
    run_dir: Path,
) -> None:
    """Pretrain the value branch on Monte Carlo returns from expert demos.

    SB3's MlpPolicy has SEPARATE MLPs for actor and critic:
      actor path:  mlp_extractor.policy_net → action_net + log_std
      value path:  mlp_extractor.value_net  → value_net

    This function only optimizes the value path parameters, so BC-trained
    actor weights are completely unaffected.
    """
    if not episodes:
        print("[vf_pretrain] no episodes — skipping")
        return

    device = model.device
    policy = model.policy
    policy.set_training_mode(True)

    obs_all = np.concatenate([obs for obs, _ in episodes], axis=0)
    ret_all = np.concatenate([ret for _, ret in episodes], axis=0).reshape(-1, 1)

    vf_ds = TensorDataset(
        torch.as_tensor(obs_all, dtype=torch.float32),
        torch.as_tensor(ret_all, dtype=torch.float32),
    )
    loader = DataLoader(vf_ds, batch_size=cfg.get("bc_batch_size", 256),
                        shuffle=True, drop_last=False, num_workers=0)

    n_epochs = cfg.get("vf_pretrain_epochs", 20)
    lr = cfg.get("vf_pretrain_lr", cfg.get("bc_lr", 3e-4))
    max_grad_norm = cfg.get("max_grad_norm", 0.5)

    # Only the value branch — actor weights stay frozen from BC
    vf_params = (
        list(policy.mlp_extractor.value_net.parameters())
        + list(policy.value_net.parameters())
    )
    optimizer = torch.optim.Adam(vf_params, lr=lr)

    print(f"[vf_pretrain] {len(obs_all)} transitions  "
          f"epochs={n_epochs}  lr={lr}  "
          f"return_range=[{ret_all.min():.3f}, {ret_all.max():.3f}]")

    log_entries: List[Dict[str, Any]] = []
    for epoch in range(1, n_epochs + 1):
        total_mse = 0.0
        n_batches = 0

        for obs_b, ret_b in loader:
            obs_b = obs_b.to(device)
            ret_b = ret_b.to(device)

            values = policy.predict_values(obs_b)   # (B, 1)
            loss = F.mse_loss(values, ret_b)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(vf_params, max_grad_norm)
            optimizer.step()

            total_mse += loss.item()
            n_batches += 1

        avg_mse = total_mse / max(n_batches, 1)
        log_entries.append({"epoch": epoch, "vf_mse": avg_mse})
        if epoch == 1 or epoch % 5 == 0 or epoch == n_epochs:
            print(f"[vf_pretrain] epoch {epoch:3d}/{n_epochs}  mse={avg_mse:.4f}")

    # Merge into existing train log
    log_path = run_dir / "bc_train_log.json"
    existing = json.loads(log_path.read_text()) if log_path.exists() else {}
    existing["vf_pretrain"] = log_entries
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[vf_pretrain] done  final mse={log_entries[-1]['vf_mse']:.4f}")


# ---------------------------------------------------------------------------
# Expert demo collection from RoboMME (needs SAPIEN — lazy import)
# ---------------------------------------------------------------------------

def _read_episode_actions(env) -> Optional[np.ndarray]:
    candidates = [env]
    if hasattr(env, "_inner") and env._inner is not None:
        candidates.append(env._inner)
    for obj in candidates:
        while True:
            for attr in ("demo_actions", "expert_actions", "actions",
                         "episode_actions", "recorded_actions", "traj_actions"):
                val = getattr(obj, attr, None)
                if val is not None:
                    try:
                        arr = np.asarray(val, dtype=np.float32)
                        if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
                            return arr
                    except Exception:
                        pass
            for ep_attr in ("episode", "demo", "recording", "task"):
                ep_obj = getattr(obj, ep_attr, None)
                if ep_obj is None:
                    continue
                for act_attr in ("actions", "demo_actions", "expert_actions"):
                    val = getattr(ep_obj, act_attr, None)
                    if val is not None:
                        try:
                            arr = np.asarray(val, dtype=np.float32)
                            if arr.ndim == 2 and arr.shape[0] > 0:
                                return arr
                        except Exception:
                            pass
            inner = getattr(obj, "unwrapped", None) or getattr(obj, "env", None)
            if inner is None or inner is obj:
                break
            obj = inner
    return None


def _parse_demo_action_from_info(
    info: Dict[str, Any], action_shape: tuple
) -> Optional[np.ndarray]:
    for key in ("demo_action", "expert_action", "action", "gt_action"):
        val = info.get(key)
        if val is not None:
            try:
                arr = np.asarray(val, dtype=np.float32).flatten()
                if arr.shape == action_shape:
                    return arr
            except Exception:
                continue
    return None


def _collect_demos(
    task_name: str,
    seed: int,
    out_dir: Path,
    n_episodes: Optional[int],
    env_kwargs: Dict[str, Any],
) -> Path:
    """Collect expert demos by running the RoboMME environment.

    RoboMME / SAPIEN is imported lazily here — it is NOT imported anywhere
    else in this file. If you have the H5 dataset already, use --h5 instead.
    """
    from env.robomme_env import make_env  # lazy: only when --collect is used

    demo_dir = out_dir / "demos"
    demo_dir.mkdir(parents=True, exist_ok=True)

    kw = dict(env_kwargs or {})
    kw.setdefault("dataset", "train")
    kw["subtask_reward"] = 0.0
    kw["step_penalty"] = 0.0

    probe = make_env(task_name, seed=seed, env_kwargs=kw)
    total_episodes = getattr(probe, "_episode_num", None) or 1
    probe.close()

    if n_episodes is not None:
        total_episodes = min(n_episodes, total_episodes)

    print(f"[bc_collect] collecting {total_episodes} expert episodes from {task_name}")

    logger = TrajectoryLogger(demo_dir, task_name=task_name, seed=seed)
    n_collected = n_success = 0

    for ep_idx in range(total_episodes):
        ep_kw = dict(kw)
        ep_kw["episode_idx"] = ep_idx
        env = make_env(task_name, seed=seed + ep_idx, env_kwargs=ep_kw)

        episode_actions = _read_episode_actions(env)
        if episode_actions is not None:
            obs, info = env.reset()
            logger.start_episode(obs)
            terminated = truncated = False
            for action in episode_actions:
                if terminated or truncated:
                    break
                action = np.asarray(action, dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                logger.record(action, reward, terminated, truncated, info, obs)
            if info.get("success"):
                n_success += 1
            logger.end_episode()
        else:
            obs, info = env.reset()
            logger.start_episode(obs)
            terminated = truncated = False
            while not (terminated or truncated):
                zero = np.zeros(env.action_space.shape, dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(zero)
                expert = _parse_demo_action_from_info(info, env.action_space.shape)
                if expert is None:
                    logger.end_episode()
                    env.close()
                    logger.close()
                    raise RuntimeError(
                        "Cannot extract expert actions from RoboMME.\n"
                        "Provide --dataset with pre-collected NPZ trajectories,\n"
                        "or --h5 with the RoboMME record HDF5 file."
                    )
                logger.record(expert, reward, terminated, truncated, info, obs)
            if info.get("success"):
                n_success += 1
            logger.end_episode()

        env.close()
        n_collected += 1
        if n_collected % 20 == 0 or n_collected == total_episodes:
            print(f"[bc_collect] {n_collected}/{total_episodes}  "
                  f"success_rate={n_success/n_collected:.2f}")

    logger.close()
    print(f"[bc_collect] saved {n_collected} episodes to {demo_dir}")
    return demo_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Behavioral Cloning from RoboMME expert demonstrations"
    )
    ap.add_argument("--config", default="configs/bc.yaml")
    ap.add_argument("--task", dest="task_name", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--h5", dest="h5_path", default=None,
                    help="RoboMME record HDF5 file")
    ap.add_argument("--dataset", default=None,
                    help="Directory of pre-saved NPZ trajectory files")
    ap.add_argument("--collect", action="store_true",
                    help="Collect expert demos by running RoboMME (needs SAPIEN)")
    ap.add_argument("--max_episodes", type=int, default=None,
                    help="Max episodes to load from H5 or collect")
    ap.add_argument("--inspect", action="store_true",
                    help="Print H5 structure and exit (requires --h5)")
    ap.add_argument("--n_demos", type=int, default=None,
                    help="Alias for --max_episodes when using --collect")
    ap.add_argument("--bc_epochs", type=int, default=None)
    ap.add_argument("--bc_lr", type=float, default=None)
    ap.add_argument("--bc_batch_size", type=int, default=None)
    ap.add_argument("--vf_pretrain_epochs", type=int, default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--allow-gym-fallback", dest="allow_gym_fallback",
                    action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    override_keys = (
        "task_name", "seed", "bc_epochs", "bc_lr", "bc_batch_size",
        "vf_pretrain_epochs", "output_dir", "allow_gym_fallback",
    )
    overrides = {k: v for k, v in vars(args).items()
                 if k in override_keys and v is not None}
    cfg = merge_overrides(cfg, overrides)

    if args.inspect:
        if not args.h5_path:
            raise SystemExit("--inspect requires --h5 <file>")
        import json as _json
        print(_json.dumps(inspect_h5(args.h5_path), indent=2, default=str))
        return

    set_global_seed(cfg["seed"], deterministic=cfg.get("deterministic_torch", False))
    run_dir = build_run_dir(cfg["output_dir"], cfg["task_name"], cfg["seed"], tag=args.tag)
    save_run_config(run_dir, cfg)

    env_kwargs = cfg.get("env_kwargs") or {}
    gamma = cfg.get("gamma", 0.99)
    max_ep = args.max_episodes or args.n_demos or cfg.get("max_episodes")

    # ---- data ---------------------------------------------------------------
    if args.h5_path:
        print(f"[train_bc] H5 dataset: {args.h5_path}")
        dataset = H5BCDataset(args.h5_path, max_episodes=max_ep)
        vf_episodes = _load_vf_episodes_from_h5(
            args.h5_path, gamma=gamma, max_episodes=max_ep
        )
    elif args.dataset:
        data_dir = Path(args.dataset)
        print(f"[train_bc] NPZ dataset: {data_dir}")
        dataset = BCDataset(data_dir)
        vf_episodes = _load_vf_episodes_from_npz(data_dir, gamma=gamma)
    elif args.collect or cfg.get("collect_demos", False):
        data_dir = _collect_demos(
            task_name=cfg["task_name"],
            seed=cfg["seed"],
            out_dir=run_dir,
            n_episodes=max_ep,
            env_kwargs=env_kwargs,
        )
        dataset = BCDataset(data_dir)
        vf_episodes = _load_vf_episodes_from_npz(data_dir, gamma=gamma)
    else:
        raise SystemExit(
            "Specify a data source:\n"
            "  --h5 <file>      RoboMME record HDF5 file (recommended)\n"
            "  --dataset <dir>  directory of pre-saved NPZ trajectories\n"
            "  --collect        collect from RoboMME env (needs SAPIEN)\n"
        )

    # ---- build policy (no RoboMME needed) -----------------------------------
    model = _build_policy_from_spaces(cfg, dataset.obs_dim, dataset.act_dim)

    # ---- BC actor training --------------------------------------------------
    train_bc(dataset, model, cfg, run_dir)

    # ---- value head pretraining ---------------------------------------------
    pretrain_value_head(vf_episodes, model, cfg, run_dir)

    # ---- save ---------------------------------------------------------------
    final_path = run_dir / "checkpoints" / "bc_final.zip"
    model.save(str(final_path))
    print(f"[train_bc] saved → {final_path}")
    print(f"[train_bc] warm-start PPO:")
    print(f"    python training/train_ppo.py --config configs/ppo.yaml \\")
    print(f"        --bc_checkpoint {final_path}")


if __name__ == "__main__":
    main()
