"""RoboMME environment adapter.

Wraps the official benchmark from https://github.com/RoboMME/robomme_benchmark
(`robomme.env_record_wrapper.BenchmarkEnvBuilder`) behind a Gymnasium-shaped
interface so the rest of this repo (training, eval, trajectory logger) can
talk to it uniformly.

Backends, tried in order:
  1. The installed `robomme` package.
  2. `$ROBOMME_PATH` — a local checkout (we prepend `<path>/src` to sys.path
     so the `robomme` package import resolves).
  3. A plain Gymnasium env id — only if `allow_gym_fallback=True`. Used for
     pipeline smoke tests (e.g. `CartPole-v1`); never silently substituted.

Important RoboMME-specific behavior (per doc/env_format.md):
  * `BenchmarkEnvBuilder.make_env_for_episode(ep)` creates an env bound to
    a single fixed episode. To support multi-episode rollouts (PPO needs
    many resets), this wrapper rebuilds the inner env each `reset()`,
    cycling through `episode_idx` mod `episode_num`.
  * Observations are `dict[str, list]` — every value is a list of frames
    over the last sub-step window. With `flatten_obs=True` (default) we
    take the latest entry of each requested key and concatenate the
    numeric ones into a single 1-D float32 vector for use with MlpPolicy.
    With `flatten_obs=False` the raw dict is forwarded (you must use a
    custom policy that understands it).
  * Per the RoboMME docs the scalar `reward` is currently unused (the
    benchmark targets imitation learning). For PPO baselines you'll
    therefore see ~zero extrinsic reward; this is exactly the motivation
    for the curiosity/memory modules. See README "Known issues".
"""
from __future__ import annotations

import importlib
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
    "  1. Clone https://github.com/RoboMME/robomme_benchmark and `uv pip install -e .`\n"
    "     (or use the Dockerfile in that repo).\n"
    "  2. Set ROBOMME_PATH=/abs/path/to/robomme_benchmark — this adapter\n"
    "     prepends <ROBOMME_PATH>/src to sys.path so `import robomme` works.\n"
    "  3. For pipeline smoke tests only, pass allow_gym_fallback=True and use a\n"
    "     Gymnasium env id as task_name (e.g. CartPole-v1)."
)

ROBOMME_TASKS: List[str] = [
    "BinFill", "PickXtimes", "SwingXtimes", "StopCube",
    "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap",
    "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
]

# Numeric observation keys we use when `flatten_obs=True`.
_DEFAULT_FLATTEN_KEYS = ("eef_state_list", "joint_state_list", "gripper_state_list")


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


def list_tasks() -> List[str]:
    robomme = _try_import_robomme()
    if robomme is not None:
        try:
            from robomme.env_record_wrapper import BenchmarkEnvBuilder
            return list(BenchmarkEnvBuilder.get_task_list())
        except Exception:
            pass
    return list(ROBOMME_TASKS)


@dataclass
class EnvMetadata:
    task_name: str
    backend: str
    max_episode_steps: Optional[int]
    extra: Dict[str, Any]


class _FlattenLatestObs:
    """Take the last item of each list-valued obs key and concatenate numeric ones.

    Produces a single float32 vector observation + a Box space. Image keys
    (`*_rgb_list`, `*_depth_list`, `maniskill_obs`) are skipped; if you need
    pixels, set `flatten_obs=False` and provide a custom policy.
    """

    def __init__(self, keys: Sequence[str] = _DEFAULT_FLATTEN_KEYS):
        self.keys = tuple(keys)
        self._dim: Optional[int] = None

    def _vec(self, obs: Dict[str, Any]) -> np.ndarray:
        parts: List[np.ndarray] = []
        for k in self.keys:
            v = obs.get(k)
            if v is None:
                continue
            arr = np.asarray(v[-1] if isinstance(v, (list, tuple)) else v).astype(np.float32).flatten()
            parts.append(arr)
        if not parts:
            raise RuntimeError(
                f"None of the requested keys {self.keys} were present in obs."
            )
        return np.concatenate(parts, axis=0)

    def transform(self, obs: Dict[str, Any]) -> np.ndarray:
        v = self._vec(obs)
        if self._dim is None:
            self._dim = v.shape[0]
        return v

    def space(self, sample_obs: Dict[str, Any]) -> spaces.Box:
        v = self._vec(sample_obs)
        self._dim = v.shape[0]
        # SB3 2.3+ requires finite observation space bounds for on-policy algorithms.
        return spaces.Box(low=-10.0, high=10.0, shape=(v.shape[0],), dtype=np.float32)


_ACTION_DIMS = {
    "joint_angle": 8,   # 7 joints + gripper
    "ee_pose": 7,       # xyz + rpy + gripper
    "waypoint": 7,      # same as ee_pose, discrete keyframes
}


def _action_space_for(name: str) -> spaces.Space:
    if name in _ACTION_DIMS:
        d = _ACTION_DIMS[name]
        # SB3 2.3+ requires finite bounds; joint angles stay well within [-10, 10] rad.
        return spaces.Box(low=-10.0, high=10.0, shape=(d,), dtype=np.float32)
    raise ValueError(
        f"action_space '{name}' is not supported by the Gym-compatible wrapper "
        f"(supported: {sorted(_ACTION_DIMS)}). For multi_choice use the raw env."
    )


def _patch_demonstration_wrapper() -> None:
    """Bypass DemonstrationWrapper.get_demonstration_trajectory() for PPO.

    get_demonstration_trajectory() creates PandaMotionPlanner which calls
    mplib.Planner → C++ ArticulatedModel → segfault on this machine.

    PPO does not need demonstration trajectories — only the underlying
    BinFill env's reset (valid physical scene, correct obs) is required.
    This replaces DemonstrationWrapper.reset() with a direct passthrough
    to self.env.reset(), completely skipping the mplib call.

    Must be called after _try_import_robomme() has added robomme to sys.path.
    """
    import sys as _sys
    try:
        from robomme.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper

        if getattr(DemonstrationWrapper, "_ppo_patch_applied", False):
            return

        def _direct_reset(self, seed=None, options=None):
            return self.env.reset(seed=seed, options=options)

        DemonstrationWrapper.reset = _direct_reset
        DemonstrationWrapper._ppo_patch_applied = True
        _sys.stderr.write(
            "[robomme patch] DemonstrationWrapper.reset → direct env.reset() "
            "(skipping get_demonstration_trajectory / mplib)\n"
        )
        _sys.stderr.flush()
    except Exception as exc:
        _sys.stderr.write(
            f"[robomme patch] DemonstrationWrapper patch failed: {exc}\n"
        )
        _sys.stderr.flush()


def _apply_robomme_patches() -> None:
    """One-time patches for panda_wristcam → panda URDF substitution.

    DemonstrationWrapper.reset() calls get_demonstration_trajectory() which
    creates PandaMotionPlanner → mplib.Planner(urdf=robot.urdf_path, ...).
    panda_wristcam's URDF makes mplib's C++ ArticulatedModel segfault.

    mplib.Planner.__init__ is pure Python so we can intercept it before the
    C++ call and swap the wristcam URDF for the standard panda URDF (same
    kinematic chain, valid mplib asset).
    """
    import os as _os, sys as _sys

    # --- gym.make: log what robot_uids is actually passed -----------------
    if not getattr(gym, "_robomme_patch_applied", False):
        _orig_make = gym.make

        def _patched_gym_make(env_id, **kwargs):
            robot = kwargs.get("robot_uids", "<not in kwargs>")
            _sys.stderr.write(
                f"[robomme patch] gym.make({env_id!r}, robot_uids={robot!r})\n"
            )
            _sys.stderr.flush()
            if "wristcam" in str(robot):
                kwargs["robot_uids"] = "panda"
            return _orig_make(env_id, **kwargs)

        gym.make = _patched_gym_make
        gym._robomme_patch_applied = True

    # --- mplib.Planner: replace wristcam URDF path with panda path --------
    try:
        import mplib as _mplib

        if getattr(_mplib.Planner, "_robomme_patch_applied", False):
            return

        _orig_planner_init = _mplib.Planner.__init__

        def _patched_planner_init(self, *args, **kwargs):
            # Resolve urdf from first positional arg or keyword arg.
            urdf = args[0] if args else kwargs.get("urdf", "")
            if "wristcam" in str(urdf):
                panda_urdf = str(urdf).replace("panda_wristcam", "panda")
                if _os.path.exists(panda_urdf):
                    _sys.stderr.write(
                        f"[mplib patch] wristcam→panda: {panda_urdf}\n"
                    )
                    _sys.stderr.flush()
                    if args:
                        args = (panda_urdf,) + args[1:]
                    else:
                        kwargs["urdf"] = panda_urdf
            # Forward all args/kwargs exactly as received — avoids "multiple
            # values" errors that arise when explicit param names conflict with
            # the kwargs dict.
            return _orig_planner_init(self, *args, **kwargs)

        _patched_planner_init._robomme_patch_applied = True
        _mplib.Planner.__init__ = _patched_planner_init
        _mplib.Planner._robomme_patch_applied = True
        _sys.stderr.write("[robomme patch] mplib.Planner.__init__ patched\n")
        _sys.stderr.flush()

    except Exception as exc:
        _sys.stderr.write(f"[robomme patch] mplib patch failed: {exc}\n")
        _sys.stderr.flush()


class RoboMMEEnv(gym.Env):
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
        flatten_obs: bool = True,
        flatten_keys: Sequence[str] = _DEFAULT_FLATTEN_KEYS,
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
        self._flatten_obs = flatten_obs
        self._flattener = _FlattenLatestObs(flatten_keys) if flatten_obs else None
        self._builder_kwargs = dict(builder_kwargs or {})
        self._episode_kwargs = dict(episode_kwargs or {})
        self._next_ep_cursor = 0
        self._inner = None
        self._builder = None
        self._episode_num: Optional[int] = None
        self._subtask_reward = float(subtask_reward)
        self._step_penalty = float(step_penalty)
        self._prev_task_index: int = 0
        self._backend, self._builder, _gym_env = self._init_backend(allow_gym_fallback)

        if self._backend == "gymnasium-fallback":
            self._inner = _gym_env
            self.observation_space = self._inner.observation_space
            self.action_space = self._inner.action_space
        else:
            self._inner = self._open_new_episode(self._select_episode(seed))
            sample, _ = self._inner.reset()
            if self._flattener is not None:
                self.observation_space = self._flattener.space(sample)
            else:
                # Best-effort: expose a Dict space with float Boxes for known numeric keys
                self.observation_space = self._infer_dict_space(sample)
            self.action_space = _action_space_for(action_space)

        self.metadata_info = EnvMetadata(
            task_name=task_name,
            backend=self._backend,
            max_episode_steps=max_steps if self._backend == "robomme" else
                (getattr(self._inner.spec, "max_episode_steps", None)
                 if getattr(self._inner, "spec", None) is not None else None),
            extra={
                "dataset": dataset,
                "action_space": action_space,
                "episode_num": self._episode_num,
            },
        )

    def _init_backend(self, allow_gym_fallback: bool) -> Tuple[str, Any, Any]:
        # SAPIEN needs two env vars for headless GPU rendering on compute VMs:
        #
        #   SAPIEN_RENDER_DEVICE=cuda  — tells SAPIEN to use the CUDA/GPU renderer
        #   VK_ICD_FILENAMES           — points Vulkan at SAPIEN's bundled NVIDIA
        #                               ICD so it doesn't rely on the system-wide
        #                               Vulkan installation (which may be absent).
        #
        # SAPIEN ships nvidia_icd.json inside its Python package under
        # sapien/vulkan_library/. We auto-detect that path here so neither var
        # needs to be exported manually before running the script.
        import os as _os
        _os.environ.setdefault("SAPIEN_RENDER_DEVICE", "cuda")
        if not _os.environ.get("VK_ICD_FILENAMES"):
            try:
                import sapien as _sapien
                from pathlib import Path as _P
                _icd = _P(_sapien.__file__).parent / "vulkan_library" / "nvidia_icd.json"
                if _icd.exists():
                    _os.environ["VK_ICD_FILENAMES"] = str(_icd)
            except Exception:
                pass

        # DemonstrationWrapper.reset() creates PandaMotionPlanner(env), which
        # calls mplib.Planner(urdf=self.robot.urdf_path, ...). For panda_wristcam
        # the URDF path resolves to the wristcam variant which mplib's C++
        # ArticulatedModel cannot load → segfault.
        #
        # mplib.Planner.__init__ is pure Python so we CAN monkey-patch it.
        # We replace any panda_wristcam URDF path with the standard panda path
        # (same kinematic chain, valid mplib URDF). Both patches log to stderr
        # so we can see which paths are actually being used.
        _apply_robomme_patches()

        robomme = _try_import_robomme()
        if robomme is not None:
            _patch_demonstration_wrapper()
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
                    f"Gymnasium fallback for task '{self.task_name}' failed: {e}.\n"
                    + SETUP_HINT
                ) from e
        raise ImportError(SETUP_HINT)

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

    def _open_new_episode(self, episode_idx: int):
        if hasattr(self._inner, "close"):
            try:
                self._inner.close()
            except Exception:
                pass
        return self._builder.make_env_for_episode(episode_idx, **self._episode_kwargs)

    def _infer_dict_space(self, sample: Dict[str, Any]) -> spaces.Dict:
        out: Dict[str, spaces.Space] = {}
        for k, v in sample.items():
            try:
                arr = np.asarray(v[-1] if isinstance(v, (list, tuple)) else v)
                if arr.dtype.kind in "fiu" and arr.ndim <= 3:
                    out[k] = spaces.Box(low=-np.inf, high=np.inf, shape=arr.shape, dtype=arr.dtype)
            except Exception:
                continue
        return spaces.Dict(out)

    def _post(self, obs):
        if self._flattener is not None and isinstance(obs, dict):
            return self._flattener.transform(obs)
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._prev_task_index = 0
        if self._backend == "robomme":
            ep = self._select_episode(seed)
            self._inner = self._open_new_episode(ep)
            obs, info = self._inner.reset()
            info = dict(info) if isinstance(info, dict) else {}
            info["episode_idx"] = ep
            return self._post(obs), info
        return self._inner.reset(seed=seed, options=options)

    def _get_task_index(self) -> Optional[int]:
        """Read current_task_index from the underlying task env (set by sequential_task_check)."""
        base = getattr(self._inner, "unwrapped", self._inner)
        idx = getattr(base, "current_task_index", None)
        if idx is None:
            idx = getattr(base, "timestep", None)
        try:
            return int(idx) if idx is not None else None
        except (TypeError, ValueError):
            return None

    def step(self, action):
        obs, reward, terminated, truncated, info = self._inner.step(action)
        if self._backend == "robomme":
            # Use explicit None check — `reward or 0.0` would silently zero small negatives.
            raw = reward.item() if hasattr(reward, "item") else reward
            reward = float(raw if raw is not None else 0.0)
            terminated = bool(np.asarray(terminated).item() if hasattr(terminated, "item") else terminated)
            truncated = bool(np.asarray(truncated).item() if hasattr(truncated, "item") else truncated)
            # RoboMME zeroes reward intentionally (IL benchmark). Emit +1 on success
            # using info["success"] (a torch.Tensor) which is always populated by the env.
            success_val = info.get("success") if isinstance(info, dict) else None
            if success_val is not None:
                try:
                    if bool(success_val.item() if hasattr(success_val, "item") else success_val):
                        reward = 1.0
                except Exception:
                    pass
            # Emit subtask_reward for each sub-task the agent completes within an episode.
            # current_task_index (set by sequential_task_check) increments each time one
            # entry in task_list is satisfied; rewarding its advance gives a dense signal
            # without waiting for full episode success.
            if self._subtask_reward > 0.0:
                curr_idx = self._get_task_index()
                if curr_idx is not None and curr_idx > self._prev_task_index:
                    reward += self._subtask_reward * (curr_idx - self._prev_task_index)
                    self._prev_task_index = curr_idx
            # Penalise every non-terminal step to encourage shorter solutions.
            # Applied last so the success/subtask bonuses are not eroded on the winning step.
            if self._step_penalty > 0.0 and not (terminated or truncated):
                reward -= self._step_penalty
        return self._post(obs), reward, terminated, truncated, info

    def render(self):
        return self._inner.render() if hasattr(self._inner, "render") else None

    def close(self):
        if hasattr(self._inner, "close"):
            self._inner.close()

    @property
    def unwrapped(self):
        return self._inner.unwrapped if hasattr(self._inner, "unwrapped") else self._inner


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
            "low": float(np.min(space.low)) if np.isfinite(space.low).all() else None,
            "high": float(np.max(space.high)) if np.isfinite(space.high).all() else None,
        }
    if isinstance(space, spaces.Discrete):
        return {"type": "Discrete", "n": int(space.n)}
    if isinstance(space, spaces.Dict):
        return {"type": "Dict", "spaces": {k: describe_space(v) for k, v in space.spaces.items()}}
    if isinstance(space, spaces.Tuple):
        return {"type": "Tuple", "spaces": [describe_space(s) for s in space.spaces]}
    return {"type": type(space).__name__, "repr": repr(space)}
