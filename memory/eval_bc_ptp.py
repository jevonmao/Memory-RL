import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import imageio.v2 as imageio
from transformers import CLIPModel, CLIPTokenizer

from .model import CLIPMemoryVLA
from env.robomme_env import make_env


# CLIP normalization constants
CLIP_MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(3, 1, 1)
CLIP_STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(3, 1, 1)


# -----------------------------
# Load model
# -----------------------------
def load_model(
    checkpoint_path,
    device,
    clip_name="openai/clip-vit-base-patch32",
    strict=True,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "If running on Modal, make sure the checkpoint volume is mounted at /checkpoints."
        )

    print(f"[INFO] Loading checkpoint: {checkpoint_path}", flush=True)
    print(f"[INFO] Checkpoint size_bytes: {checkpoint_path.stat().st_size}", flush=True)
    print(f"[INFO] Loading CLIP backbone: {clip_name}", flush=True)

    clip_model = CLIPModel.from_pretrained(clip_name)
    tokenizer = CLIPTokenizer.from_pretrained(clip_name)

    model = CLIPMemoryVLA(clip_model=clip_model)
    ckpt = torch.load(checkpoint_path, map_location=device)

    print(f"[INFO] Checkpoint keys: {sorted(list(ckpt.keys()))}", flush=True)
    if "epoch" in ckpt:
        print(f"[INFO] Checkpoint epoch: {ckpt['epoch']}", flush=True)
    if "val_loss" in ckpt:
        print(f"[INFO] Checkpoint val_loss: {ckpt['val_loss']}", flush=True)

    try:
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    except RuntimeError:
        if strict:
            print("[WARN] Strict checkpoint loading failed. Retrying with strict=False.", flush=True)
            missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            raise

    if missing:
        print(f"[WARN] Missing checkpoint parameters: {missing}", flush=True)
    if unexpected:
        print(f"[WARN] Unexpected checkpoint parameters: {unexpected}", flush=True)

    model.to(device)
    model.eval()
    return model, tokenizer


# -----------------------------
# Action selection
# -----------------------------
@torch.no_grad()
def select_action(
    model,
    tokenizer,
    states,
    images,
    instruction,
    device,
    memory=None,
    action_clip=None,
):
    states = torch.as_tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
    images = torch.as_tensor(images, dtype=torch.float32, device=device).unsqueeze(0)

    # CLIP ViT-B/32 expects 224x224 images. Offline data may already be 224, but
    # online env images can differ, so resize here for safe rollout evaluation.
    b, t, c, h, w = images.shape
    if h != 224 or w != 224:
        images = images.view(b * t, c, h, w)
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        images = images.view(b, t, c, 224, 224)

    text_tokens = tokenizer(
        [instruction],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    text_tokens = {k: v.to(device) for k, v in text_tokens.items()}

    out = model(
        images=images,
        states=states,
        text_tokens=text_tokens,
        memory=memory,
        detach_memory=True,
        memory_update="all",
    )

    action = out["pred_action"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    next_memory = out["memory"]

    if action_clip is not None:
        action = np.clip(action, -float(action_clip), float(action_clip))

    return action, next_memory


# -----------------------------
# STATE EXTRACTION
# -----------------------------

def _to_numpy(x):
    """Convert tensors / arrays / scalars to a CPU numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _squeeze_batch(x):
    """Remove leading singleton vector-env batch dimensions."""
    x = _to_numpy(x)
    while x.ndim > 1 and x.shape[0] == 1:
        x = x[0]
    return x



def _flatten_numeric(x):
    return _squeeze_batch(x).astype(np.float32).reshape(-1)


# -----------------------------
# Quaternion and live state helpers
# -----------------------------
def _quat_wxyz_to_rpy(q):
    """Convert quaternion [w, x, y, z] to roll/pitch/yaw."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"Expected quaternion shape (4,), got {q.shape}")
    w, x, y, z = q

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.sign(sinp) * (np.pi / 2.0)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.asarray([roll, pitch, yaw], dtype=np.float32)


def _extract_live_eef_state(env):
    """Extract live TCP pose as [x, y, z, roll, pitch, yaw]."""
    if env is None:
        return None
    try:
        agent = env._inner.unwrapped.agent
        raw_pose = agent.tcp_pose.raw_pose
        raw_pose = _squeeze_batch(raw_pose).reshape(-1)
        if raw_pose.shape[0] < 7:
            return None
        pos = raw_pose[:3].astype(np.float32)
        quat = raw_pose[3:7]
        rpy = _quat_wxyz_to_rpy(quat)
        return np.concatenate([pos, rpy], axis=0).astype(np.float32)
    except Exception as exc:
        print(f"[WARN] Failed to extract live TCP pose: {exc}", flush=True)
        return None


# -----------------------------
# STATE EXTRACTION
# -----------------------------
def extract_state(obs, env=None):
    """
    Extract a 15D state vector.

    Supports both the recorded RoboMME observation format used during offline
    training and the native ManiSkill/RoboMME format returned after bypassing
    demonstration wrappers. For native live observations, reconstructs the
    training state as eef_state(6) + joint_state(7) + gripper_state(2).
    """

    # flattened env already returns ndarray
    if not isinstance(obs, dict):
        state = np.asarray(obs, dtype=np.float32).reshape(-1)
        if state.shape[0] != 15:
            raise ValueError(f"[State] Expected state dim 15, got {state.shape[0]}")
        return state

    # Offline / recorded RoboMME format.
    keys = ["eef_state_list", "joint_state_list", "gripper_state_list"]
    parts = []
    for k in keys:
        if k in obs:
            v = obs[k]
            v = v[-1] if isinstance(v, (list, tuple)) else v
            parts.append(np.asarray(v).flatten())

    if parts:
        state = np.concatenate(parts).astype(np.float32)
        if state.shape[0] != 15:
            raise ValueError(f"[State] Expected state dim 15, got {state.shape[0]}")
        return state

    # Native ManiSkill/RoboMME live format. Reconstruct the same state layout
    # used during H5 training: eef_state(6) + joint_state(7) + gripper_state(2).
    if "agent" in obs and isinstance(obs["agent"], dict):
        agent_obs = obs["agent"]
        qpos = _flatten_numeric(agent_obs["qpos"]) if "qpos" in agent_obs else None
        eef_state = _extract_live_eef_state(env)
        if qpos is not None and eef_state is not None and qpos.shape[0] >= 9:
            joint_state = qpos[:7].astype(np.float32)
            gripper_state = qpos[7:9].astype(np.float32)
            state = np.concatenate([eef_state, joint_state, gripper_state], axis=0).astype(np.float32)
            if state.shape[0] != 15:
                raise ValueError(f"[State] Expected reconstructed state dim 15, got {state.shape[0]}")
            return state

        # Last-resort smoke-test fallback, not faithful to training.
        native_parts = []
        for k in ("qpos", "qvel"):
            if k in agent_obs:
                native_parts.append(_flatten_numeric(agent_obs[k]))
        if native_parts:
            print("[WARN] Using non-training qpos/qvel state fallback", flush=True)
            state = np.concatenate(native_parts).astype(np.float32)
            if state.shape[0] < 15:
                state = np.pad(state, (0, 15 - state.shape[0]))
            elif state.shape[0] > 15:
                state = state[:15]
            return state

    raise KeyError(f"[State] No valid keys found. Available: {list(obs.keys())}")


# -----------------------------
# IMAGE EXTRACTION
# -----------------------------

def _normalize_image(img):
    img = _squeeze_batch(img)

    # Some ManiSkill camera tensors have shape HWC, BHWC, CHW, or BCHW. After
    # squeezing singleton batch dims above, convert HWC to CHW.
    if img.dtype == np.uint8:
        img = img.astype(np.float32)
    else:
        img = img.astype(np.float32)

    if img.max(initial=0) > 1.5:
        img = img / 255.0

    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]

    if img.ndim == 3 and img.shape[-1] == 3:
        img = np.transpose(img, (2, 0, 1))

    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"[Image] Expected CHW RGB image, got shape {img.shape}")

    img = img.astype(np.float32)
    img = (img - CLIP_MEAN) / CLIP_STD
    return img.astype(np.float32)


# -----------------------------
# IMAGE EXTRACTION
# -----------------------------
def extract_image(obs):
    """
    Extract RGB image as CHW float32 in [0, 1].
    """

    fallback = np.zeros((3, 224, 224), dtype=np.float32)

    if not isinstance(obs, dict):
        return fallback

    # Offline / recorded RoboMME format.
    image_keys = [
        "front_rgb_list",
        "rgb_list",
        "camera_rgb_list",
        "image",
        "rgb",
    ]

    for k in image_keys:
        if k in obs:
            img = obs[k]
            img = img[-1] if isinstance(img, (list, tuple)) else img
            return _normalize_image(img)

    # Native ManiSkill format: obs['sensor_data'][camera_name]['rgb'].
    sensor_data = obs.get("sensor_data")
    if isinstance(sensor_data, dict):
        preferred_cameras = [
            "base_camera",
            "front_camera",
            "camera",
            "hand_camera",
            "hand_camera_rgb",
        ]
        camera_names = preferred_cameras + [k for k in sensor_data.keys() if k not in preferred_cameras]
        for camera_name in camera_names:
            camera_obs = sensor_data.get(camera_name)
            if not isinstance(camera_obs, dict):
                continue
            for image_key in ("rgb", "Color", "color", "image"):
                if image_key in camera_obs:
                    return _normalize_image(camera_obs[image_key])

    print("[WARN] No image key found. Available keys:", list(obs.keys()), flush=True)
    return fallback


def _raw_rgb_from_obs(obs, camera="base_camera"):
    """Extract raw uint8 HWC RGB frames for visualization videos."""
    try:
        if not isinstance(obs, dict):
            return None
        sensor_data = obs.get("sensor_data")
        if not isinstance(sensor_data, dict):
            return None
        camera_obs = sensor_data.get(camera)
        if not isinstance(camera_obs, dict) or "rgb" not in camera_obs:
            return None
        img = _squeeze_batch(camera_obs["rgb"])
        if img.ndim != 3:
            return None
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.shape[-1] != 3:
            return None
        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            if img.max(initial=0) <= 1.5:
                img = img * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img[..., :3]
    except Exception:
        return None


def _concat_video_views(base_frame, hand_frame):
    """Concatenate base and hand camera frames side by side."""
    if base_frame is None:
        return hand_frame
    if hand_frame is None:
        return base_frame

    if hand_frame.shape[0] != base_frame.shape[0]:
        pad_h = base_frame.shape[0] - hand_frame.shape[0]
        if pad_h > 0:
            hand_frame = np.pad(hand_frame, ((0, pad_h), (0, 0), (0, 0)), mode="constant")
        elif pad_h < 0:
            base_frame = np.pad(base_frame, ((0, -pad_h), (0, 0), (0, 0)), mode="constant")
    return np.concatenate([base_frame, hand_frame], axis=1)


# -----------------------------
# Video rendering helpers
# -----------------------------
def _render_frame_from_env(env):
    """Render a fresh RGB frame from the live env for visualization videos."""
    render_targets = []
    try:
        render_targets.append(env._inner)
    except Exception:
        pass
    try:
        render_targets.append(env._inner.unwrapped)
    except Exception:
        pass
    try:
        render_targets.append(env)
    except Exception:
        pass

    for target in render_targets:
        try:
            frame = target.render()
        except TypeError:
            try:
                frame = target.render(mode="rgb_array")
            except Exception:
                frame = None
        except Exception:
            frame = None

        if frame is None:
            continue

        frame = np.asarray(frame)
        frame = _squeeze_batch(frame)
        if frame.ndim != 3:
            continue
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.shape[-1] != 3:
            continue
        if frame.dtype != np.uint8:
            frame = frame.astype(np.float32)
            if frame.max(initial=0) <= 1.5:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame[..., :3]

    return None


def _video_frame_from_env_or_obs(env, obs):
    """Prefer fresh render frames; fallback to obs sensor frames if rendering is unavailable."""
    frame = _render_frame_from_env(env)
    if frame is not None:
        return frame

    base_frame = _raw_rgb_from_obs(obs, "base_camera")
    hand_frame = _raw_rgb_from_obs(obs, "hand_camera")
    return _concat_video_views(base_frame, hand_frame)


def parse_obs(obs, env=None):
    return extract_state(obs, env=env), extract_image(obs)


# -----------------------------
# PROGRESS METRICS HELPERS
# -----------------------------
def _safe_scalar_bool(x):
    return bool(np.asarray(_to_numpy(x)).any())


def _safe_scalar_float(x, default=np.nan):
    try:
        arr = np.asarray(_to_numpy(x)).reshape(-1)
        if arr.size == 0:
            return float(default)
        return float(arr[0])
    except Exception:
        return float(default)


def _pose_position_from_obj(obj):
    """Best-effort extraction of a 3D position from SAPIEN/ManiSkill objects."""
    if obj is None:
        return None
    try:
        if hasattr(obj, "pose"):
            pose = obj.pose
            if hasattr(pose, "p"):
                return _squeeze_batch(pose.p).reshape(-1)[:3].astype(np.float32)
            if hasattr(pose, "raw_pose"):
                return _squeeze_batch(pose.raw_pose).reshape(-1)[:3].astype(np.float32)
        if hasattr(obj, "p"):
            return _squeeze_batch(obj.p).reshape(-1)[:3].astype(np.float32)
        if hasattr(obj, "raw_pose"):
            return _squeeze_batch(obj.raw_pose).reshape(-1)[:3].astype(np.float32)
    except Exception:
        return None
    return None


# -----------------------------
# RoboMME symbolic task/subgoal helpers
# -----------------------------

def _extract_unwrapped_env(env):
    try:
        return env._inner.unwrapped
    except Exception:
        return None


def _extract_current_task_info(env):
    """Extract RoboMME symbolic task/subgoal information when available."""
    unwrapped = _extract_unwrapped_env(env)
    if unwrapped is None:
        return {
            "task_index": None,
            "task_name": None,
            "subgoal": None,
            "segment_name": None,
            "segment_pos": None,
            "num_tasks": None,
        }

    task_index = getattr(unwrapped, "current_task_index", None)
    task_name = getattr(unwrapped, "current_task_name_online", None)
    if task_name is None:
        task_name = getattr(unwrapped, "current_task_name", None)

    subgoal = getattr(unwrapped, "current_subgoal_segment_online", None)
    if subgoal is None:
        subgoal = getattr(unwrapped, "current_subgoal_segment", None)

    task_list = getattr(unwrapped, "task_list", None)
    num_tasks = len(task_list) if isinstance(task_list, list) else None
    segment_name = None
    segment_pos = None

    if isinstance(task_list, list) and task_index is not None:
        try:
            task_entry = task_list[int(task_index)]
            if isinstance(task_entry, dict):
                segment = task_entry.get("segment")
                if isinstance(segment, (list, tuple)):
                    segment_items = segment
                elif segment is None:
                    segment_items = []
                else:
                    segment_items = [segment]
                for obj in segment_items:
                    pos = _pose_position_from_obj(obj)
                    if pos is not None:
                        segment_name = getattr(obj, "name", None)
                        if segment_name is None:
                            segment_name = getattr(obj, "uid", None)
                        if segment_name is None:
                            segment_name = obj.__class__.__name__
                        segment_pos = pos
                        break
        except Exception:
            pass

    return {
        "task_index": int(task_index) if task_index is not None else None,
        "task_name": task_name,
        "subgoal": subgoal,
        "segment_name": segment_name,
        "segment_pos": segment_pos,
        "num_tasks": num_tasks,
    }



def _select_instruction(base_instruction, env, use_env_subgoal=False):
    """Choose the language fed to the policy at the current step."""
    if not use_env_subgoal:
        return base_instruction
    task_info = _extract_current_task_info(env)
    return task_info.get("subgoal") or task_info.get("task_name") or base_instruction


# -----------------------------
# ENV EPISODE INSTRUCTION HELPERS
# -----------------------------
def _pluralize_cube(n):
    return "cube" if int(n) == 1 else "cubes"


def _instruction_from_binfill_sequence(sequence):
    """Build a full BinFill instruction from env.binfill_language_sequence."""
    if not sequence:
        return None

    parts = []
    for color, count in sequence:
        count = int(count)
        if count == 1:
            parts.append(f"one {color} cube")
        elif count == 2:
            parts.append(f"two {color} cubes")
        elif count == 3:
            parts.append(f"three {color} cubes")
        else:
            parts.append(f"{count} {color} cubes")

    if len(parts) == 1:
        object_phrase = parts[0]
    elif len(parts) == 2:
        object_phrase = f"{parts[0]} and {parts[1]}"
    else:
        object_phrase = ", ".join(parts[:-1]) + f", and {parts[-1]}"

    return f"put {object_phrase} into the bin, then press the button to stop"


def _extract_env_episode_instruction(env):
    """Extract or reconstruct the full per-episode language instruction."""
    unwrapped = _extract_unwrapped_env(env)
    if unwrapped is None:
        return None

    for attr in ("task_goal", "language_goal", "goal"):
        value = getattr(unwrapped, attr, None)
        if value is None:
            continue
        try:
            arr = np.asarray(value).reshape(-1)
            if arr.size > 0:
                value = arr[0]
        except Exception:
            pass
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, str):
            return value

    sequence = getattr(unwrapped, "binfill_language_sequence", None)
    return _instruction_from_binfill_sequence(sequence)


def _find_named_position(unwrapped_env, name_patterns):
    """Find the first env attribute whose name contains one of name_patterns and has a pose."""
    if unwrapped_env is None:
        return None, None
    for attr in dir(unwrapped_env):
        lower = attr.lower()
        if not any(pat in lower for pat in name_patterns):
            continue
        if attr.startswith("__"):
            continue
        try:
            value = getattr(unwrapped_env, attr)
        except Exception:
            continue
        pos = _pose_position_from_obj(value)
        if pos is not None:
            return attr, pos
    return None, None


def _extract_progress_positions(env):
    """Best-effort task progress positions for sparse-reward RoboMME envs.

    Returns a dict containing tcp/object/goal positions when they are available.
    This is intentionally heuristic because RoboMME task object names differ
    across tasks.
    """
    out = {}
    try:
        unwrapped = env._inner.unwrapped
    except Exception:
        return out

    eef = _extract_live_eef_state(env)
    if eef is not None:
        out["tcp_pos"] = eef[:3]

    obj_name, obj_pos = _find_named_position(
        unwrapped,
        ["cube", "obj", "object", "ball", "peg", "button"],
    )
    if obj_pos is not None:
        out["object_name"] = obj_name
        out["object_pos"] = obj_pos

    goal_name, goal_pos = _find_named_position(
        unwrapped,
        ["goal", "target", "bin", "basket", "container"],
    )
    if goal_pos is not None:
        out["goal_name"] = goal_name
        out["goal_pos"] = goal_pos

    return out


def _summarize_distance_series(values):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float32)
    if arr.size == 0:
        return None
    return {
        "initial": float(arr[0]),
        "final": float(arr[-1]),
        "min": float(np.min(arr)),
        "delta_final_minus_initial": float(arr[-1] - arr[0]),
        "improvement_initial_minus_min": float(arr[0] - np.min(arr)),
    }


# -----------------------------
# EVALUATION
# -----------------------------
def evaluate(
    task="BinFill",
    checkpoint=None,
    episodes=20,
    history_len=8,
    instruction=None,
    clip_name="openai/clip-vit-base-patch32",
    action_clip=None,
    render=False,
    max_eval_steps=1000,
    debug_rollout=False,
    use_env_subgoal_instruction=False,
    use_env_episode_instruction=True,
    save_video_dir=None,
    video_every=5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint is None:
        checkpoint = os.environ.get("CHECKPOINT_PATH", f"/checkpoints/{task}_bc_best.pt")

    if instruction is None:
        # Keep this as None unless the user explicitly provides TASK_INSTRUCTION.
        # This allows per-episode instructions to be reconstructed from the env
        # after reset, which is important because BinFill goals vary by episode.
        instruction = os.environ.get("TASK_INSTRUCTION", None)

    print(f"[INFO] Using device: {device}", flush=True)
    print(f"[INFO] Task: {task}", flush=True)
    print(f"[INFO] Checkpoint: {checkpoint}", flush=True)
    print(f"[INFO] Episodes: {episodes}", flush=True)
    print(f"[INFO] History len: {history_len}", flush=True)
    print(f"[INFO] Instruction override: {instruction}", flush=True)
    print(f"[INFO] CLIP name: {clip_name}", flush=True)
    print(f"[INFO] Action clip: {action_clip}", flush=True)
    print(f"[INFO] Max eval steps: {max_eval_steps}", flush=True)
    print(f"[INFO] Debug rollout: {debug_rollout}", flush=True)
    print(f"[INFO] Use env subgoal instruction: {use_env_subgoal_instruction}", flush=True)
    print(f"[INFO] Use env episode instruction: {use_env_episode_instruction}", flush=True)
    print(f"[INFO] Save video dir: {save_video_dir}", flush=True)
    print(f"[INFO] Video every: {video_every}", flush=True)

    model, tokenizer = load_model(checkpoint, device, clip_name=clip_name)

    # IMPORTANT: keep structured obs
    env = make_env(task, env_kwargs={"flatten_obs": False})

    success_count = 0
    returns = []
    episode_progress_summaries = []

    for ep in range(episodes):
        print(f"[Episode {ep}] reset start", flush=True)
        obs, info = env.reset(seed=ep)
        print(f"[Episode {ep}] reset done info={info}", flush=True)

        episode_instruction = instruction
        if use_env_episode_instruction and instruction is None:
            episode_instruction = _extract_env_episode_instruction(env)
        if episode_instruction is None:
            episode_instruction = task
        print(f"[Episode {ep}] instruction: {episode_instruction}", flush=True)

        state, image = parse_obs(obs, env=env)
        if debug_rollout:
            print(
                f"[Episode {ep}] parsed initial obs: state_shape={state.shape} image_shape={image.shape}",
                flush=True,
            )
            print(
                f"[Episode {ep}] initial state vector: "
                f"{np.array2string(state, precision=4, suppress_small=True)}",
                flush=True,
            )
            print(
                f"[Episode {ep}] initial state stats: min={float(np.min(state)):.4f} "
                f"max={float(np.max(state)):.4f} mean={float(np.mean(state)):.4f}",
                flush=True,
            )
            print(
                f"[Episode {ep}] initial image stats: min={float(np.min(image)):.4f} "
                f"max={float(np.max(image)):.4f} mean={float(np.mean(image)):.4f}",
                flush=True,
            )

        # IMPORTANT: avoid aliasing bug
        history_states = [state.copy() for _ in range(history_len)]
        history_images = [image.copy() for _ in range(history_len)]

        memory = None
        done = False
        total_reward = 0.0
        step = 0
        video_frames = []
        if save_video_dir is not None:
            frame = _video_frame_from_env_or_obs(env, obs)
            if frame is not None:
                video_frames.append(frame)
        eef_positions = [state[:3].copy()]
        joint_positions = [state[6:13].copy()]
        action_norms = []
        gripper_actions = []
        gripper_qpos_values = [state[13:15].copy()]
        gripper_qpos_deltas = []
        memory_norms = []
        memory_delta_norms = []
        prev_memory_for_delta = None
        tcp_to_object_dists = []
        object_to_goal_dists = []
        object_heights = []
        current_segment_dists = []
        current_segment_heights = []
        task_indices = []
        task_transitions = []
        prev_task_index = None
        prev_task_name = None
        progress_positions = _extract_progress_positions(env)
        object_name = progress_positions.get("object_name")
        goal_name = progress_positions.get("goal_name")
        if "tcp_pos" in progress_positions and "object_pos" in progress_positions:
            tcp_to_object_dists.append(float(np.linalg.norm(progress_positions["tcp_pos"] - progress_positions["object_pos"])))
        if "object_pos" in progress_positions and "goal_pos" in progress_positions:
            object_to_goal_dists.append(float(np.linalg.norm(progress_positions["object_pos"] - progress_positions["goal_pos"])))
        if "object_pos" in progress_positions:
            object_heights.append(float(progress_positions["object_pos"][2]))
        task_info = _extract_current_task_info(env)
        if task_info["task_index"] is not None:
            task_indices.append(task_info["task_index"])
            prev_task_index = task_info["task_index"]
            prev_task_name = task_info["task_name"]
            task_transitions.append((0, task_info["task_index"], task_info["task_name"], task_info["subgoal"]))
        if "tcp_pos" in progress_positions and task_info.get("segment_pos") is not None:
            current_segment_dists.append(float(np.linalg.norm(progress_positions["tcp_pos"] - task_info["segment_pos"])))
            current_segment_heights.append(float(task_info["segment_pos"][2]))
        if debug_rollout:
            print(
                f"[Episode {ep}] progress object={object_name} goal={goal_name} "
                f"initial_tcp_to_object={tcp_to_object_dists[-1] if tcp_to_object_dists else 'NA'} "
                f"initial_object_to_goal={object_to_goal_dists[-1] if object_to_goal_dists else 'NA'} "
                f"task_index={task_info.get('task_index')} task_name={task_info.get('task_name')} "
                f"subgoal={task_info.get('subgoal')}",
                flush=True,
            )

        while not done and step < max_eval_steps:
            hs = np.stack(history_states[-history_len:])
            hi = np.stack(history_images[-history_len:])

            step_instruction = _select_instruction(
                episode_instruction,
                env,
                use_env_subgoal=use_env_subgoal_instruction,
            )

            if debug_rollout or step == 0 or (step + 1) % 100 == 0:
                print(f"[Episode {ep}] step {step}: selecting action", flush=True)
            action, memory = select_action(
                model=model,
                tokenizer=tokenizer,
                states=hs,
                images=hi,
                instruction=step_instruction,
                device=device,
                memory=memory,
                action_clip=action_clip,
            )
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            action_norms.append(float(np.linalg.norm(action)))
            if action.shape[0] >= 8:
                gripper_actions.append(float(action[-1]))
            if memory is not None:
                try:
                    memory_detached = memory.detach()
                    memory_norms.append(float(torch.norm(memory_detached).cpu().item()))
                    if prev_memory_for_delta is not None:
                        memory_delta_norms.append(float(torch.norm(memory_detached.cpu() - prev_memory_for_delta).item()))
                    prev_memory_for_delta = memory_detached.cpu().clone()
                except Exception:
                    try:
                        memory_detached = memory.detach()
                        memory_norms.append(float(torch.norm(memory_detached).cpu().item()))
                    except Exception:
                        pass
            if debug_rollout or step == 0 or (step + 1) % 100 == 0:
                print(
                    f"[Episode {ep}] step {step}: action_shape={action.shape} "
                    f"min={float(np.min(action)):.4f} max={float(np.max(action)):.4f}",
                    flush=True,
                )
                print(f"[Episode {ep}] step {step}: env.step start", flush=True)

            obs, reward, term, trunc, info = env.step(action)
            if debug_rollout or step == 0 or (step + 1) % 100 == 0:
                print(
                    f"[Episode {ep}] step {step}: env.step done "
                    f"reward={reward} term={term} trunc={trunc} info={info}",
                    flush=True,
                )
            done = _safe_scalar_bool(term) or _safe_scalar_bool(trunc)

            if render and hasattr(env, "render"):
                env.render()

            total_reward += _safe_scalar_float(reward, default=0.0)

            state, image = parse_obs(obs, env=env)
            history_states.append(state.copy())
            history_images.append(image.copy())
            eef_positions.append(state[:3].copy())
            joint_positions.append(state[6:13].copy())
            gripper_qpos_values.append(state[13:15].copy())
            if len(gripper_qpos_values) >= 2:
                gripper_qpos_deltas.append(float(np.linalg.norm(gripper_qpos_values[-1] - gripper_qpos_values[-2])))
            progress_positions = _extract_progress_positions(env)
            if "tcp_pos" in progress_positions and "object_pos" in progress_positions:
                tcp_to_object_dists.append(float(np.linalg.norm(progress_positions["tcp_pos"] - progress_positions["object_pos"])))
            if "object_pos" in progress_positions and "goal_pos" in progress_positions:
                object_to_goal_dists.append(float(np.linalg.norm(progress_positions["object_pos"] - progress_positions["goal_pos"])))
            if "object_pos" in progress_positions:
                object_heights.append(float(progress_positions["object_pos"][2]))
            task_info = _extract_current_task_info(env)
            if task_info["task_index"] is not None:
                task_indices.append(task_info["task_index"])
                if task_info["task_index"] != prev_task_index or task_info["task_name"] != prev_task_name:
                    task_transitions.append((step + 1, task_info["task_index"], task_info["task_name"], task_info["subgoal"]))
                    print(
                        f"[Episode {ep}] task transition at step {step + 1}: "
                        f"idx={task_info['task_index']} name={task_info['task_name']} subgoal={task_info['subgoal']}",
                        flush=True,
                    )
                    prev_task_index = task_info["task_index"]
                    prev_task_name = task_info["task_name"]
            if "tcp_pos" in progress_positions and task_info.get("segment_pos") is not None:
                current_segment_dists.append(float(np.linalg.norm(progress_positions["tcp_pos"] - task_info["segment_pos"])))
                current_segment_heights.append(float(task_info["segment_pos"][2]))
            if save_video_dir is not None and video_every > 0 and (step + 1) % video_every == 0:
                frame = _video_frame_from_env_or_obs(env, obs)
                if frame is not None:
                    video_frames.append(frame)

            step += 1

        if step >= max_eval_steps and not done:
            print(
                f"[Episode {ep}] reached max_eval_steps={max_eval_steps}; forcing episode stop",
                flush=True,
            )
        print(f"episode length: {step}", flush=True)

        success = info.get("success", False) if isinstance(info, dict) else False
        success_bool = _safe_scalar_bool(success)
        success_count += int(success_bool)
        returns.append(total_reward)

        eef_positions_arr = np.asarray(eef_positions, dtype=np.float32)
        joint_positions_arr = np.asarray(joint_positions, dtype=np.float32)
        tcp_path_length = float(np.sum(np.linalg.norm(np.diff(eef_positions_arr, axis=0), axis=1))) if len(eef_positions_arr) > 1 else 0.0
        tcp_net_displacement = float(np.linalg.norm(eef_positions_arr[-1] - eef_positions_arr[0])) if len(eef_positions_arr) > 1 else 0.0
        joint_path_length = float(np.sum(np.linalg.norm(np.diff(joint_positions_arr, axis=0), axis=1))) if len(joint_positions_arr) > 1 else 0.0
        gripper_qpos_arr = np.asarray(gripper_qpos_values, dtype=np.float32)
        gripper_opening = np.mean(gripper_qpos_arr, axis=1) if gripper_qpos_arr.size else np.asarray([], dtype=np.float32)
        gripper_initial_opening = float(gripper_opening[0]) if gripper_opening.size else float("nan")
        gripper_min_opening = float(np.min(gripper_opening)) if gripper_opening.size else float("nan")
        gripper_final_opening = float(gripper_opening[-1]) if gripper_opening.size else float("nan")
        gripper_closure = float(gripper_initial_opening - gripper_min_opening) if gripper_opening.size else float("nan")
        demo_normalized_horizon = float(step / 603.0)
        tcp_to_object_summary = _summarize_distance_series(tcp_to_object_dists)
        object_to_goal_summary = _summarize_distance_series(object_to_goal_dists)
        object_height_summary = _summarize_distance_series(object_heights)
        current_segment_summary = _summarize_distance_series(current_segment_dists)
        current_segment_height_summary = _summarize_distance_series(current_segment_heights)
        max_task_index_reached = max(task_indices) if task_indices else -1
        final_task_index = task_indices[-1] if task_indices else -1

        progress_summary = {
            "success": success_bool,
            "return": total_reward,
            "steps": step,
            "demo_normalized_horizon": demo_normalized_horizon,
            "tcp_path_length": tcp_path_length,
            "tcp_net_displacement": tcp_net_displacement,
            "joint_path_length": joint_path_length,
            "mean_action_norm": float(np.mean(action_norms)) if action_norms else 0.0,
            "max_action_norm": float(np.max(action_norms)) if action_norms else 0.0,
            "mean_gripper_action": float(np.mean(gripper_actions)) if gripper_actions else float("nan"),
            "min_gripper_action": float(np.min(gripper_actions)) if gripper_actions else float("nan"),
            "max_gripper_action": float(np.max(gripper_actions)) if gripper_actions else float("nan"),
            "gripper_initial_opening": gripper_initial_opening,
            "gripper_min_opening": gripper_min_opening,
            "gripper_final_opening": gripper_final_opening,
            "gripper_closure": gripper_closure,
            "mean_gripper_qpos_delta": float(np.mean(gripper_qpos_deltas)) if gripper_qpos_deltas else float("nan"),
            "mean_memory_norm": float(np.mean(memory_norms)) if memory_norms else float("nan"),
            "max_memory_norm": float(np.max(memory_norms)) if memory_norms else float("nan"),
            "mean_memory_delta_norm": float(np.mean(memory_delta_norms)) if memory_delta_norms else float("nan"),
            "max_memory_delta_norm": float(np.max(memory_delta_norms)) if memory_delta_norms else float("nan"),
            "tcp_to_object": tcp_to_object_summary,
            "object_to_goal": object_to_goal_summary,
            "object_height": object_height_summary,
            "current_segment": current_segment_summary,
            "current_segment_height": current_segment_height_summary,
            "max_task_index_reached": float(max_task_index_reached),
            "final_task_index": float(final_task_index),
            "num_task_transitions": float(max(0, len(task_transitions) - 1)),
            "object_name": object_name,
            "goal_name": goal_name,
        }
        episode_progress_summaries.append(progress_summary)

        print(f"[Episode {ep}] success={success_bool} return={total_reward:.3f}", flush=True)
        print(
            f"[Episode {ep}] progress: steps={step} demo_horizon_x={demo_normalized_horizon:.2f} "
            f"tcp_path={tcp_path_length:.4f} tcp_net={tcp_net_displacement:.4f} "
            f"joint_path={joint_path_length:.4f} mean_action_norm={progress_summary['mean_action_norm']:.4f} "
            f"max_action_norm={progress_summary['max_action_norm']:.4f} "
            f"grip_action_mean={progress_summary['mean_gripper_action']:.4f} "
            f"grip_action_min={progress_summary['min_gripper_action']:.4f} "
            f"grip_action_max={progress_summary['max_gripper_action']:.4f} "
            f"mean_memory_norm={progress_summary['mean_memory_norm']:.4f} "
            f"mean_memory_delta={progress_summary['mean_memory_delta_norm']:.6f} "
            f"max_task_idx={max_task_index_reached} final_task_idx={final_task_index} "
            f"task_transitions={max(0, len(task_transitions) - 1)}",
            flush=True,
        )
        print(
            f"[Episode {ep}] gripper: "
            f"opening_initial={gripper_initial_opening:.4f} "
            f"opening_min={gripper_min_opening:.4f} "
            f"opening_final={gripper_final_opening:.4f} "
            f"closure={gripper_closure:.4f} "
            f"mean_qpos_delta={progress_summary['mean_gripper_qpos_delta']:.6f}",
            flush=True,
        )
        if tcp_to_object_summary is not None:
            print(
                f"[Episode {ep}] tcp_to_object({object_name}): "
                f"initial={tcp_to_object_summary['initial']:.4f} "
                f"min={tcp_to_object_summary['min']:.4f} "
                f"final={tcp_to_object_summary['final']:.4f} "
                f"improvement={tcp_to_object_summary['improvement_initial_minus_min']:.4f}",
                flush=True,
            )
        if object_to_goal_summary is not None:
            print(
                f"[Episode {ep}] object_to_goal({object_name}->{goal_name}): "
                f"initial={object_to_goal_summary['initial']:.4f} "
                f"min={object_to_goal_summary['min']:.4f} "
                f"final={object_to_goal_summary['final']:.4f} "
                f"improvement={object_to_goal_summary['improvement_initial_minus_min']:.4f}",
                flush=True,
            )
        if object_height_summary is not None:
            print(
                f"[Episode {ep}] object_height({object_name}): "
                f"initial={object_height_summary['initial']:.4f} "
                f"max={max(object_heights):.4f} "
                f"final={object_height_summary['final']:.4f} "
                f"lift={max(object_heights) - object_height_summary['initial']:.4f}",
                flush=True,
            )
        if current_segment_summary is not None:
            print(
                f"[Episode {ep}] tcp_to_current_segment: "
                f"initial={current_segment_summary['initial']:.4f} "
                f"min={current_segment_summary['min']:.4f} "
                f"final={current_segment_summary['final']:.4f} "
                f"improvement={current_segment_summary['improvement_initial_minus_min']:.4f}",
                flush=True,
            )
        if current_segment_height_summary is not None:
            print(
                f"[Episode {ep}] current_segment_height: "
                f"initial={current_segment_height_summary['initial']:.4f} "
                f"max={max(current_segment_heights):.4f} "
                f"final={current_segment_height_summary['final']:.4f} "
                f"lift={max(current_segment_heights) - current_segment_height_summary['initial']:.4f}",
                flush=True,
            )
        if task_transitions:
            print(f"[Episode {ep}] task transitions:", flush=True)
            for transition_step, transition_idx, transition_name, transition_subgoal in task_transitions:
                print(
                    f"  step={transition_step} idx={transition_idx} name={transition_name} subgoal={transition_subgoal}",
                    flush=True,
                )
        if save_video_dir is not None and video_frames:
            os.makedirs(os.path.expanduser(save_video_dir), exist_ok=True)
            video_path = os.path.join(os.path.expanduser(save_video_dir), f"{task}_episode_{ep}.mp4")
            imageio.mimsave(video_path, video_frames, fps=10)
            print(f"[Episode {ep}] saved video: {video_path}", flush=True)

    print("\n===== FINAL RESULTS =====", flush=True)
    print(f"Success rate: {success_count / episodes:.3f}", flush=True)
    print(f"Avg return: {np.mean(returns):.3f}", flush=True)
    if episode_progress_summaries:
        print("\n===== PROGRESS METRICS =====", flush=True)
        for key in [
            "steps",
            "demo_normalized_horizon",
            "tcp_path_length",
            "tcp_net_displacement",
            "joint_path_length",
            "mean_action_norm",
            "max_action_norm",
            "mean_gripper_action",
            "min_gripper_action",
            "max_gripper_action",
            "gripper_initial_opening",
            "gripper_min_opening",
            "gripper_final_opening",
            "gripper_closure",
            "mean_gripper_qpos_delta",
            "mean_memory_norm",
            "max_memory_norm",
            "mean_memory_delta_norm",
            "max_memory_delta_norm",
            "max_task_index_reached",
            "final_task_index",
            "num_task_transitions",
        ]:
            vals = np.asarray([s[key] for s in episode_progress_summaries], dtype=np.float32)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                print(f"{key}: mean={float(np.mean(vals)):.4f} min={float(np.min(vals)):.4f} max={float(np.max(vals)):.4f}", flush=True)

        for dist_key in ["tcp_to_object", "object_to_goal", "object_height", "current_segment", "current_segment_height"]:
            summaries = [s[dist_key] for s in episode_progress_summaries if s[dist_key] is not None]
            if summaries:
                for subkey in ["initial", "min", "final", "improvement_initial_minus_min"]:
                    vals = np.asarray([d[subkey] for d in summaries], dtype=np.float32)
                    print(f"{dist_key}.{subkey}: mean={float(np.mean(vals)):.4f} min={float(np.min(vals)):.4f} max={float(np.max(vals)):.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online rollout evaluation for CLIPMemoryVLA BC + PTP policy.")
    parser.add_argument("--task", type=str, default="BinFill")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--history-len", type=int, default=8)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--clip-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--action-clip", type=float, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-eval-steps", type=int, default=1000)
    parser.add_argument("--debug-rollout", action="store_true")
    parser.add_argument(
        "--use-env-subgoal-instruction",
        action="store_true",
        help="Use the online RoboMME current subgoal/current task text as the policy instruction at each step.",
    )
    parser.add_argument(
        "--no-env-episode-instruction",
        action="store_true",
        help="Disable automatic reconstruction of the per-episode instruction from the online RoboMME env.",
    )
    parser.add_argument(
        "--save-video-dir",
        type=str,
        default=None,
        help="Directory for saving rollout MP4 videos. Each frame is base camera plus hand camera.",
    )
    parser.add_argument(
        "--video-every",
        type=int,
        default=5,
        help="Save one video frame every N env steps.",
    )
    args = parser.parse_args()

    print("=== Starting evaluation ===", flush=True)
    evaluate(
        task=args.task,
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        history_len=args.history_len,
        instruction=args.instruction,
        clip_name=args.clip_name,
        action_clip=args.action_clip,
        render=args.render,
        max_eval_steps=args.max_eval_steps,
        debug_rollout=args.debug_rollout,
        use_env_subgoal_instruction=args.use_env_subgoal_instruction,
        use_env_episode_instruction=not args.no_env_episode_instruction,
        save_video_dir=args.save_video_dir,
        video_every=args.video_every,
    )
    print("=== Evaluation finished ===", flush=True)