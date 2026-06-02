"""RoboMME environment adapter — PPO edition.

For PPO, RoboMME is used purely as a physics simulator:
  reset() → initialise a BinFill scene → return flat obs
  step()  → apply joint-angle action → return (obs, reward, done, info)

The H5 demo dataset and DemonstrationWrapper are NOT used here.
BenchmarkEnvBuilder is called only to enumerate the recorded episode
configurations (so each PPO episode starts from a diverse, reproducible
initial scene matching one of the expert demos).  The raw ManiSkill/SAPIEN
env is unwrapped from the gymnasium wrapper stack immediately, so mplib and
the demonstration-replay code are never reached.

Observations are extracted directly from the live SAPIEN physics simulation:
  eef_state   (6,)  TCP [x,y,z, roll,pitch,yaw] — SAPIEN FK
  joint_state (7,)  arm joint positions          — qpos[:7]
  gripper_state(2,) finger positions             — qpos[7:9]
  Total obs_dim = 15, matching the BC training data.

Reward engineering (same as original design):
  +1.0   on task success
  +1.0 × subtasks_completed  per sub-task step
  −0.005 per non-terminal step (step_penalty)
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "gymnasium is required. Install with: pip install -r requirements.txt"
    ) from e


SETUP_HINT = (
    "Could not locate the RoboMME benchmark. Options:\n"
    "  1. Clone https://github.com/RoboMME/robomme_benchmark and install it.\n"
    "  2. Set ROBOMME_PATH=/abs/path/to/robomme_benchmark — this adapter\n"
    "     prepends <ROBOMME_PATH>/src to sys.path so `import robomme` works.\n"
    "  3. For pipeline smoke tests only, pass allow_gym_fallback=True."
)

ROBOMME_TASKS: List[str] = [
    "BinFill", "PickXtimes", "SwingXtimes", "StopCube",
    "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap",
    "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
]

_ACTION_DIMS = {
    "joint_angle": 8,   # 7 arm joints + 1 gripper
    "ee_pose": 7,
    "waypoint": 7,
}

class _ManiSkillNoiseFilter(logging.Filter):
    """Block known informational warnings that survive setLevel resets.

    Attached to both the 'mani_skill' logger and the root logger.
    The root attachment is the reliable one: make_env_for_episode() clears
    the child logger's filter list on every call, but mani_skill records
    still propagate to root (propagate=True is the default), where this
    filter catches them before any handler emits them.
    """
    _PATTERNS = ("panda_wristcam",)

    def filter(self, record: logging.LogRecord) -> bool:
        if not record.name.startswith("mani_skill"):
            return True
        return not any(p in record.getMessage() for p in self._PATTERNS)


_MANI_SKILL_NOISE_FILTER = _ManiSkillNoiseFilter()


def _suppress_mani_skill_noise() -> None:
    """Suppress mani_skill noise on both the child logger and the root logger.

    Called at import time and before every make_env_for_episode() call.
    Idempotent — each filter is added at most once.
    """
    child = logging.getLogger("mani_skill")
    child.setLevel(logging.ERROR)
    if _MANI_SKILL_NOISE_FILTER not in child.filters:
        child.addFilter(_MANI_SKILL_NOISE_FILTER)

    root = logging.getLogger()
    if _MANI_SKILL_NOISE_FILTER not in root.filters:
        root.addFilter(_MANI_SKILL_NOISE_FILTER)


_suppress_mani_skill_noise()

# Reconstruct the ManiSkill physics context every N episodes rather than every
# reset. Each reconstruction loads a new episode config (object layout) from
# the demo dataset. N=10 gives ~200 distinct configs over a 10 M-step run
# while cutting reconstruction overhead by 10x.
_EPISODE_RELOAD_FREQ = 10


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _try_import_robomme():
    try:
        return importlib.import_module("robomme")
    except ImportError:
        pass
    path = os.environ.get("ROBOMME_PATH")
    if path:
        for candidate in (os.path.join(path, "src"), path):
            if os.path.isdir(candidate) and candidate not in sys.path:
                sys.path.insert(0, candidate)
        try:
            return importlib.import_module("robomme")
        except ImportError:
            return None
    return None


def _quat_to_rpy(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert [qw, qx, qy, qz] quaternion to [roll, pitch, yaw] in radians."""
    roll  = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw   = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return np.array([roll, pitch, yaw], dtype=np.float32)


def _action_space_for(name: str) -> spaces.Space:
    if name in _ACTION_DIMS:
        d = _ACTION_DIMS[name]
        # SB3 2.3+ requires finite bounds.
        return spaces.Box(low=-10.0, high=10.0, shape=(d,), dtype=np.float32)
    raise ValueError(
        f"action_space '{name}' not supported (supported: {sorted(_ACTION_DIMS)})"
    )


@dataclass
class EnvMetadata:
    task_name: str
    backend: str
    max_episode_steps: Optional[int]
    extra: Dict[str, Any]


def _sanitize_info(info: Any) -> Dict[str, Any]:
    """Convert a ManiSkill info dict to plain Python types.

    SubprocVecEnv pickles info dicts across processes; PyTorch tensors and
    numpy scalars with non-standard dtypes cause pickling or stacking errors.
    We convert every value to a Python scalar, list, or numpy array.
    """
    if not isinstance(info, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in info.items():
        try:
            if hasattr(v, "item"):          # 0-d tensor or numpy scalar
                out[k] = v.item()
            elif hasattr(v, "tolist"):      # tensor / ndarray with shape
                out[k] = v.tolist()
            else:
                out[k] = v
        except Exception:
            out[k] = v
    return out


def list_tasks() -> List[str]:
    robomme = _try_import_robomme()
    if robomme is not None:
        try:
            from robomme.env_record_wrapper import BenchmarkEnvBuilder
            return list(BenchmarkEnvBuilder.get_task_list())
        except Exception:
            pass
    return list(ROBOMME_TASKS)


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------

class RoboMMEEnv(gym.Env):
    """Gymnasium wrapper around RoboMME / ManiSkill BinFill for PPO training.

    Architecture
    ------------
    BenchmarkEnvBuilder is used ONLY to enumerate the recorded episode configs
    (object placements, task goals) so every PPO episode starts from a diverse,
    reproducible initial scene.  The env returned by make_env_for_episode() is
    immediately .unwrapped to get the raw ManiSkill env, bypassing every
    gymnasium wrapper — in particular DemonstrationWrapper and its mplib
    motion-planning dependency.

    Observations are read directly from the SAPIEN physics simulation after
    every reset() and step():
      eef_state    (6,)  TCP position [x,y,z] + orientation [roll,pitch,yaw]
      joint_state  (7,)  arm qpos[:7]
      gripper_state(2,)  finger qpos[7:9]
    Total obs_dim = 15, matching the BC training data from the H5 dataset.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        task_name: str,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        allow_gym_fallback: bool = False,
        dataset: str = "train",
        action_space: str = "joint_angle",
        max_steps: int = 300,
        episode_idx: Optional[int] = None,
        flatten_obs: bool = True,           # kept for API compat; always True
        flatten_keys: Sequence[str] = (),   # kept for API compat; unused
        builder_kwargs: Optional[Dict[str, Any]] = None,
        episode_kwargs: Optional[Dict[str, Any]] = None,
        subtask_reward: float = 0.0,
        step_penalty: float = 0.0,
    ):
        super().__init__()
        self.task_name = task_name
        self._render_mode = render_mode
        self._dataset = dataset
        self._action_space_name = action_space
        self._max_steps = max_steps
        self._fixed_episode_idx = episode_idx
        self._builder_kwargs = dict(builder_kwargs or {})
        self._episode_kwargs = dict(episode_kwargs or {})
        self._subtask_reward = float(subtask_reward)
        self._step_penalty = float(step_penalty)
        self._next_ep_cursor = 0
        self._episode_num: Optional[int] = None
        self._inner = None          # raw ManiSkill env (unwrapped)
        self._prev_task_index = 0
        self._step_count = 0
        self._reset_count = 0

        self._backend, self._builder, _gym_env = self._init_backend(allow_gym_fallback)

        if self._backend == "gymnasium-fallback":
            self._inner = _gym_env
            self.observation_space = _gym_env.observation_space
            self.action_space = _gym_env.action_space
        else:
            # Open one episode to determine obs/act spaces.
            self._inner = self._open_episode_env(self._select_episode(seed))
            self._inner.reset()
            sample_obs = self._obs_from_sapien()
            self.observation_space = spaces.Box(
                low=-10.0, high=10.0, shape=sample_obs.shape, dtype=np.float32
            )
            self.action_space = _action_space_for(action_space)

        self.metadata_info = EnvMetadata(
            task_name=task_name,
            backend=self._backend,
            max_episode_steps=max_steps if self._backend == "robomme" else None,
            extra={"dataset": dataset, "episode_num": self._episode_num},
        )

    # -----------------------------------------------------------------------
    # Backend init
    # -----------------------------------------------------------------------

    def _init_backend(self, allow_gym_fallback: bool) -> Tuple[str, Any, Any]:
        # Set GPU rendering env vars so SAPIEN can use Vulkan on headless VMs.
        os.environ.setdefault("SAPIEN_RENDER_DEVICE", "cuda")
        if not os.environ.get("VK_ICD_FILENAMES"):
            try:
                import sapien as _sapien
                from pathlib import Path as _P
                _icd = _P(_sapien.__file__).parent / "vulkan_library" / "nvidia_icd.json"
                if _icd.exists():
                    os.environ["VK_ICD_FILENAMES"] = str(_icd)
            except Exception:
                pass

        robomme = _try_import_robomme()
        if robomme is not None:
            # Re-apply after robomme/mani_skill import, which resets the level.
            _suppress_mani_skill_noise()
            from robomme.env_record_wrapper import BenchmarkEnvBuilder
            builder = BenchmarkEnvBuilder(
                env_id=self.task_name,
                dataset=self._dataset,
                action_space=self._action_space_name,
                gui_render=(self._render_mode == "human"),
                max_steps=self._max_steps,
                **self._builder_kwargs,
            )
            self._episode_num = int(builder.get_episode_num())
            return "robomme", builder, None

        if allow_gym_fallback:
            try:
                env = gym.make(self.task_name, render_mode=self._render_mode)
                return "gymnasium-fallback", None, env
            except Exception as e:
                raise ImportError(
                    f"Gymnasium fallback for '{self.task_name}' failed: {e}.\n"
                    + SETUP_HINT
                ) from e
        raise ImportError(SETUP_HINT)

    # -----------------------------------------------------------------------
    # Episode management
    # -----------------------------------------------------------------------

    def _select_episode(self, seed: Optional[int]) -> int:
        if self._fixed_episode_idx is not None:
            return int(self._fixed_episode_idx) % max(1, self._episode_num or 1)
        if seed is not None and self._episode_num:
            ep = int(seed) % self._episode_num
            self._next_ep_cursor = (ep + 1) % self._episode_num
            return ep
        ep = self._next_ep_cursor
        if self._episode_num:
            self._next_ep_cursor = (self._next_ep_cursor + 1) % self._episode_num
        return ep

    def _open_episode_env(self, episode_idx: int):
        """Return the raw ManiSkill BinFill env for one episode.

        BenchmarkEnvBuilder.make_env_for_episode() returns:
            FailAwareWrapper → DemonstrationWrapper → BinFill (ManiSkill)
        .unwrapped drills through every wrapper to the raw BinFill env,
        bypassing DemonstrationWrapper (mplib / demo replay entirely).

        Critical: BinFill.evaluate() checks self.use_demonstrationwrapper to
        decide whether allow_subgoal_change_this_timestep.  With the default
        False, tasks NEVER advance — current_task_index stays 0 and success
        is never set, so PPO only ever sees step penalties.  Setting it to
        True (with demonstration_record_traj=False, which is also the default)
        enables task advancement on every step, giving correct subtask rewards
        and success signals for free PPO exploration.
        """
        if self._inner is not None:
            try:
                self._inner.close()
            except Exception:
                pass
        _suppress_mani_skill_noise()
        wrapped = self._builder.make_env_for_episode(episode_idx, **self._episode_kwargs)
        env = wrapped.unwrapped   # raw ManiSkill BinFill env

        # Enable task progression for PPO (see BinFill.evaluate() lines 429-438).
        # demonstration_record_traj stays False (not recording), so the
        # allow_subgoal_change_this_timestep branch evaluates to True.
        if hasattr(env, "use_demonstrationwrapper"):
            env.use_demonstrationwrapper = True

        return env

    # -----------------------------------------------------------------------
    # Observation extraction from SAPIEN
    # -----------------------------------------------------------------------

    def _obs_from_sapien(self) -> np.ndarray:
        """Extract [eef(6) + joint(7) + gripper(2)] from the live SAPIEN sim.

        After reset() or step(), SAPIEN has already solved FK for the current
        joint configuration.  We read:
          agent.tcp.pose         → TCP position + quaternion → pos + rpy
          agent.robot.get_qpos() → all joint positions (9-D for Panda)

        Note: _inner.reset() / _inner.step() return values are deliberately
        ignored for obs.  We read SAPIEN state directly so obs is always in
        the same format regardless of which gymnasium wrappers are in the stack.
        """
        import sys as _sys
        env = self._inner

        # ---- EEF from TCP pose (FK result after every step/reset) -----------
        # BinFill uses agent.tcp_pose directly (see BinFill.py line 464).
        # That property returns a Pose object with .p (position) and .q (quat).
        # Quaternion convention in ManiSkill/SAPIEN: [qw, qx, qy, qz].
        eef_state = np.zeros(6, dtype=np.float32)
        try:
            tcp_pose = env.agent.tcp_pose    # BinFill's native API
            p = np.asarray(tcp_pose.p).flatten().astype(np.float32)[:3]
            q = np.asarray(tcp_pose.q).flatten().astype(np.float32)[:4]
            rpy = _quat_to_rpy(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            eef_state = np.concatenate([p, rpy])
        except Exception as _e:
            try:
                # Fallback: some ManiSkill versions expose tcp via agent.tcp.pose
                tcp_pose = env.agent.tcp.pose
                p = np.asarray(tcp_pose.p).flatten().astype(np.float32)[:3]
                q = np.asarray(tcp_pose.q).flatten().astype(np.float32)[:4]
                rpy = _quat_to_rpy(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
                eef_state = np.concatenate([p, rpy])
            except Exception as _e2:
                if not getattr(self, "_eef_warn_printed", False):
                    _sys.stderr.write(
                        f"[robomme_env] WARNING: could not read EEF pose: {_e2}\n"
                        f"  eef_state will be zeros. Tried agent.tcp_pose and agent.tcp.pose.\n"
                    )
                    _sys.stderr.flush()
                    self._eef_warn_printed = True

        # ---- Joint + gripper from qpos -------------------------------------
        joint_state = np.zeros(7, dtype=np.float32)
        gripper_state = np.zeros(2, dtype=np.float32)
        try:
            qpos = np.asarray(env.agent.robot.get_qpos()).flatten().astype(np.float32)
            joint_state   = qpos[:7]  if len(qpos) >= 7 else joint_state
            gripper_state = qpos[7:9] if len(qpos) >= 9 else gripper_state
        except Exception as _e:
            if not getattr(self, "_qpos_warn_printed", False):
                _sys.stderr.write(
                    f"[robomme_env] WARNING: could not read qpos from SAPIEN: {_e}\n"
                    f"  joint_state and gripper_state will be zeros.\n"
                )
                _sys.stderr.flush()
                self._qpos_warn_printed = True

        return np.concatenate([eef_state, joint_state, gripper_state])  # (15,)

    # -----------------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._prev_task_index = 0
        self._step_count = 0
        self._reset_count += 1

        if self._backend != "robomme":
            return self._inner.reset(seed=seed, options=options)

        if self._reset_count % _EPISODE_RELOAD_FREQ == 0:
            ep = self._select_episode(seed)
            self._inner = self._open_episode_env(ep)
            info = {"episode_idx": ep}
        else:
            info = {}

        self._inner.reset()         # initialise physics + FK
        obs = self._obs_from_sapien()
        return obs, info

    def step(self, action):
        if self._backend != "robomme":
            return self._inner.step(action)

        _, _reward_env, terminated, truncated, info = self._inner.step(action)

        # Convert ManiSkill tensor types → plain Python
        terminated = bool(
            np.asarray(terminated).item() if hasattr(terminated, "item") else terminated
        )
        truncated = bool(
            np.asarray(truncated).item() if hasattr(truncated, "item") else truncated
        )
        info = _sanitize_info(info)

        # BinFill.evaluate() sets info["fail"] when a task fails (e.g. wrong
        # cube dropped, or button pressed before bin is filled).
        # ManiSkill's BaseEnv.step() only terminates on success; we must
        # handle fail → terminated ourselves to avoid wasting rollout steps.
        fail_val = info.get("fail")
        if fail_val is not None:
            try:
                if bool(fail_val.item() if hasattr(fail_val, "item") else fail_val):
                    terminated = True
            except Exception:
                pass

        # Enforce max_steps: .unwrapped strips the TimeLimit gymnasium wrapper,
        # so we must track the step count ourselves.
        self._step_count = getattr(self, "_step_count", 0) + 1
        if self._max_steps > 0 and self._step_count >= self._max_steps:
            truncated = True

        # Read obs from SAPIEN AFTER physics step
        obs = self._obs_from_sapien()

        # Reward engineering
        reward = self._compute_reward(info, terminated, truncated)

        return obs, reward, terminated, truncated, info

    def _compute_reward(
        self, info: Dict[str, Any], terminated: bool, truncated: bool
    ) -> float:
        reward = 0.0

        # +1.0 on full task success
        success_val = info.get("success")
        if success_val is not None:
            try:
                if bool(success_val.item() if hasattr(success_val, "item") else success_val):
                    reward = 1.0
            except Exception:
                pass

        # +subtask_reward per completed sub-task within the episode
        if self._subtask_reward > 0.0:
            curr_idx = self._get_task_index(info)
            if curr_idx is not None and curr_idx > self._prev_task_index:
                reward += self._subtask_reward * (curr_idx - self._prev_task_index)
                self._prev_task_index = curr_idx

        # −step_penalty on every step that isn't a true termination (success/fail).
        # Truncation (time limit) still incurs the penalty so that the episode
        # total matches: n_subtasks × subtask_reward − max_steps × step_penalty.
        if self._step_penalty > 0.0 and not terminated:
            reward -= self._step_penalty

        return reward

    def _get_task_index(self, info: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """Read the sequential sub-task counter for dense subtask rewards.

        ManiSkill's raw BinFill env tracks completed sub-tasks in
        info['current_task_index'] (set by sequential_task_check inside the
        task).  We try the info dict first (reliable), then fall back to an
        env attribute (may not be present on the unwrapped env).
        """
        # Primary: read from step info dict
        if info:
            for key in ("current_task_index", "task_index", "num_tasks_done"):
                val = info.get(key)
                if val is not None:
                    try:
                        return int(val.item() if hasattr(val, "item") else val)
                    except (TypeError, ValueError):
                        pass

        # Fallback: attribute set by sequential_task_check on the BinFill env.
        # Only look for current_task_index — NOT elapsed_steps / timestep
        # which is the physics step counter, not the sub-task counter.
        env = self._inner
        val = getattr(env, "current_task_index", None)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
        return None

    def render(self):
        if self._inner is not None and hasattr(self._inner, "render"):
            return self._inner.render()
        return None

    def close(self):
        if self._inner is not None and hasattr(self._inner, "close"):
            try:
                self._inner.close()
            except Exception:
                pass

    @property
    def unwrapped(self):
        return self._inner


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def make_env(
    task_name: str,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    allow_gym_fallback: bool = False,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> RoboMMEEnv:
    env_kwargs = dict(env_kwargs or {})
    return RoboMMEEnv(
        task_name=task_name,
        seed=seed,
        render_mode=render_mode,
        allow_gym_fallback=allow_gym_fallback,
        **env_kwargs,
    )


def describe_space(space) -> Dict[str, Any]:
    if isinstance(space, spaces.Box):
        return {
            "type": "Box",
            "shape": tuple(space.shape),
            "dtype": str(space.dtype),
        }
    if isinstance(space, spaces.Discrete):
        return {"type": "Discrete", "n": int(space.n)}
    return {"type": type(space).__name__, "repr": repr(space)}
