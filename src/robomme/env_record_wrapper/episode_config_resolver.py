"""
Episode configuration resolver: Parse episode seed and difficulty from metadata, and build wrapped environment.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym

from ..logging_utils import logger

DATASET_ROOT = Path(__file__).resolve().parents[1] / "env_metadata"

_ALLOWED_DATASETS = {"train", "test", "val"}
_ALLOWED_ACTION_SPACES = {"joint_angle", "ee_pose", "waypoint", "multi_choice"}
_DEFAULT_TASK_LIST = [
    "PickXtimes",
    "StopCube",
    "SwingXtimes",
    "BinFill",
    "VideoUnmaskSwap",
    "VideoUnmask",
    "ButtonUnmaskSwap",
    "ButtonUnmask",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "PickHighlight",
    "InsertPeg",
    "MoveCube",
    "PatternLock",
    "RouteStick",
]


def load_episode_metadata(metadata_path: Union[str, Path, None]) -> Dict[Tuple[str, int], Dict[str, object]]:
    """
    Read episode metadata from JSON file; return empty dictionary if missing or invalid.
    Used to restore specific episode configuration (e.g., seed, difficulty).
    """

    metadata_index: Dict[Tuple[str, int], Dict[str, object]] = {}
    if not metadata_path:
        return metadata_index

    path = Path(metadata_path)
    if not path.exists():
        logger.debug(f"Metadata file not found, skipping: {path}")
        return metadata_index

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # pragma: no cover - informational logging only
        logger.debug(f"Failed to read metadata {path}: {exc}")
        return metadata_index

    default_task = str(payload.get("env_id") or "").strip()
    for record in payload.get("records", []):
        task_name = str(record.get("task") or default_task or "").strip()
        episode = record.get("episode")
        if not task_name or episode is None:
            continue
        try:
            episode_idx = int(episode)
        except (TypeError, ValueError):
            continue
        metadata_index[(task_name, episode_idx)] = record

    if metadata_index:
        logger.debug(f"Loaded {len(metadata_index)} metadata records from {path}")
    else:
        logger.debug(f"No valid metadata entries found in {path}")
    return metadata_index


def get_episode_metadata(
    metadata_index: Dict[Tuple[str, int], Dict[str, object]],
    task: str,
    episode: int,
) -> Optional[Dict[str, object]]:
    """Find metadata entry for specific (task, episode) pair."""

    if not metadata_index:
        return None
    return metadata_index.get((task, episode))


class BenchmarkEnvBuilder:
    """
    Episode environment builder.

    Automatically parse metadata based on dataset and env_id, and build wrapped environment according to action_space.
    """

    def __init__(
        self,
        env_id: str,
        dataset: str = "test",
        action_space: str = "joint_angle",
        gui_render: bool = False,
        override_metadata_path: Optional[Union[str, Path]] = None,
        max_steps: int = 10000,
    ):
        if dataset not in _ALLOWED_DATASETS:
            raise ValueError(f"Unsupported dataset '{dataset}'. Allowed datasets: {sorted(_ALLOWED_DATASETS)}")
        if action_space not in _ALLOWED_ACTION_SPACES:
            raise ValueError(
                f"Unsupported action_space '{action_space}'. "
                f"Allowed action spaces: {sorted(_ALLOWED_ACTION_SPACES)}"
            )

        self.env_id = env_id
        self.dataset = dataset
        self.action_space = action_space
        self.gui_render = gui_render
        self.override_metadata_path = (
            Path(override_metadata_path) if override_metadata_path is not None else None
        )
        self.render_mode = "human" if gui_render else "rgb_array"
        self.max_steps_without_demonstration = max_steps+2

        metadata_path = self._resolve_metadata_path()
        self.metadata_index = load_episode_metadata(metadata_path)

    @classmethod
    def get_task_list(cls) -> List[str]:
        """
        Return list of evaluatable tasks.
        Task list is fixed to built-in default order, not automatically discovered from metadata.
        """

        return list(_DEFAULT_TASK_LIST)

    def _resolve_metadata_path(self) -> str:
        if self.override_metadata_path is not None:
            return str(
                self.override_metadata_path / f"record_dataset_{self.env_id}_metadata.json"
            )
        if self.dataset in _ALLOWED_DATASETS:
            return os.path.join(str(DATASET_ROOT), self.dataset, f"record_dataset_{self.env_id}_metadata.json")
        raise ValueError(f"Unsupported dataset '{self.dataset}'.")

    def resolve_episode(self, episode: int):
        """Parse episode configuration based on metadata."""
        seed = None
        difficulty_hint = None

        metadata = get_episode_metadata(self.metadata_index, self.env_id, episode)
        if metadata:
            metadata_seed = metadata.get("seed")
            if metadata_seed is not None:
                try:
                    seed = int(metadata_seed)
                except (TypeError, ValueError):
                    logger.debug(f"[{self.env_id}] Invalid metadata seed for episode {episode}: {metadata_seed}")
            difficulty_hint = metadata.get("difficulty")

        return seed, difficulty_hint

    def get_episode_num(self) -> int:
        """
        Return number of episodes for current env_id in metadata.
        Note: By convention, this method returns count (int) instead of list.
        """
        if not self.metadata_index:
            return 0
        episode_set = {episode for (task, episode) in self.metadata_index if task == self.env_id}
        return len(episode_set)

    def make_env_for_episode(
        self,
        episode_idx: int,
        max_steps: Optional[int] = None,
        include_maniskill_obs: bool = False,
        include_front_depth: bool = False,
        include_wrist_depth: bool = False,
        include_front_camera_extrinsic: bool = False,
        include_wrist_camera_extrinsic: bool = False,
        include_available_multi_choices: bool = False,
        include_front_camera_intrinsic: bool = False,
        include_wrist_camera_intrinsic: bool = False,
    ):
        """Create and configure environment for specific episode. Wrap EndeffectorDemonstrationWrapper for action_space=ee_pose, MultiStepDemonstrationWrapper for waypoint, OraclePlannerDemonstrationWrapper for multi_choice."""
        from .DemonstrationWrapper import DemonstrationWrapper

        max_steps_without_demo = (
            max_steps + 2 if max_steps is not None else self.max_steps_without_demonstration
        )

        seed, difficulty_hint = self.resolve_episode(episode_idx)
        import os, platform, torch
        _sim_backend = os.environ.get("ROBOMME_SIM_BACKEND")
        _render_backend = os.environ.get("ROBOMME_RENDER_BACKEND")
        if _sim_backend is None or _render_backend is None:
            if platform.system() == "Windows":
                def _win_vulkan():
                    try:
                        import subprocess
                        r = subprocess.run(
                            ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=5,
                        )
                        pci = r.stdout.strip().lower()
                        if pci:
                            parts = pci.split(":")
                            return f"pci:{parts[1]}:{parts[2]}"
                    except Exception:
                        pass
                    return "cpu"
                _sim_backend = _sim_backend or "physx_cpu"
                _render_backend = _render_backend or _win_vulkan()
            else:
                _gpu = torch.cuda.is_available()
                _sim_backend = _sim_backend or ("physx_cuda" if _gpu else "physx_cpu")
                _render_backend = _render_backend or ("gpu" if _gpu else "cpu")
        # ROBOMME_OBS_MODE lets the RL training path drop the depth and
        # segmentation render passes (set to "rgb"). SAPIEN does ~2x the
        # per-step GPU rendering work for the full "rgb+depth+segmentation"
        # mode; video recording wrappers still want the full mode.
        _obs_mode = os.environ.get("ROBOMME_OBS_MODE", "rgb+depth+segmentation")
        env_kwargs = dict(
            obs_mode=_obs_mode,
            control_mode="pd_joint_pos",
            render_mode=self.render_mode,
            # Default to "sparse" because every task's compute_dense_reward is a
            # zero stub (the benchmark grades via success/fail flags). Sparse mode
            # returns info["success"] - info["fail"] from BaseEnv.compute_sparse_reward,
            # which is well-defined. RL training paths that need a shaped reward
            # build it in the wrapper layer (see train/rewards/).
            # The two test fixtures that need the dense path override this explicitly.
            reward_mode="sparse",
            sim_backend=_sim_backend,
            render_backend=_render_backend,
        )
        # ROBOMME_CAMERA_RES sets a global width/height override for every camera
        # in the env. Tasks ship 256x256 base + 256x256 wrist cameras; the RL
        # wrapper resizes both down to 128x128 with cv2 anyway, so rendering at
        # 256 then resizing is ~4x wasted GPU work per camera. Recording paths
        # that want full-res should leave this unset.
        _cam_res = os.environ.get("ROBOMME_CAMERA_RES")
        if _cam_res:
            try:
                _res = int(_cam_res)
                env_kwargs["sensor_configs"] = dict(width=_res, height=_res)
            except ValueError:
                logger.warning(f"Ignoring invalid ROBOMME_CAMERA_RES={_cam_res!r}")

        # ROBOMME_SIM_FREQ overrides the physics simulation rate (Hz). Default
        # SimConfig uses sim_freq=100 / control_freq=20 → 5 physx substeps per
        # env.step. With physx_cpu running on multi-threaded SAPIEN, those
        # substeps dominate the CPU budget; lowering sim_freq to 50 halves
        # per-step CPU work and is fine for tasks like BinFill where contact
        # dynamics don't require fine timesteps. control_freq stays at 20.
        _sim_freq = os.environ.get("ROBOMME_SIM_FREQ")
        if _sim_freq:
            try:
                env_kwargs["sim_config"] = dict(sim_freq=int(_sim_freq))
            except ValueError:
                logger.warning(f"Ignoring invalid ROBOMME_SIM_FREQ={_sim_freq!r}")
        if seed is not None:
            env_kwargs["seed"] = seed
        if difficulty_hint:
            env_kwargs["difficulty"] = difficulty_hint
        seed_desc = seed if seed is not None else "default"
        difficulty_str = f", difficulty={difficulty_hint}" if difficulty_hint else ""
        logger.debug(f"[{self.env_id}] Episode {episode_idx}: seed={seed_desc}{difficulty_str}")

        env = gym.make(self.env_id, **env_kwargs)
        force_front_camera_params = self.action_space == "multi_choice"
        include_front_camera_extrinsic_effective = (
            include_front_camera_extrinsic or force_front_camera_params
        )
        include_front_camera_intrinsic_effective = (
            include_front_camera_intrinsic or force_front_camera_params
        )
        env = DemonstrationWrapper(
            env,
            max_steps_without_demonstration=max_steps_without_demo,
            gui_render=self.gui_render,
            include_maniskill_obs=include_maniskill_obs,
            include_front_depth=include_front_depth,
            include_wrist_depth=include_wrist_depth,
            include_front_camera_extrinsic=include_front_camera_extrinsic_effective,
            include_wrist_camera_extrinsic=include_wrist_camera_extrinsic,
            include_available_multi_choices=include_available_multi_choices,
            include_front_camera_intrinsic=include_front_camera_intrinsic_effective,
            include_wrist_camera_intrinsic=include_wrist_camera_intrinsic,
        )
        if self.action_space == "joint_angle":
            pass
        elif self.action_space == "ee_pose":
            from .EndeffectorDemonstrationWrapper import EndeffectorDemonstrationWrapper

            env = EndeffectorDemonstrationWrapper(env, action_repr="rpy")

        elif self.action_space == "waypoint":
            from .MultiStepDemonstrationWrapper import MultiStepDemonstrationWrapper

            env = MultiStepDemonstrationWrapper(env, gui_render=self.gui_render, vis=self.gui_render)
        elif self.action_space == "multi_choice":
            from .OraclePlannerDemonstrationWrapper import OraclePlannerDemonstrationWrapper
            env = OraclePlannerDemonstrationWrapper(env, env_id=self.env_id, gui_render=self.gui_render)

        # ====== After all wrappers are packaged, add a fallback Try...Except ErrorWrapper =====
        from .FailAwareWrapper import FailAwareWrapper
        env = FailAwareWrapper(env)

        return env
