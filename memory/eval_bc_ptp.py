import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
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


def parse_obs(obs, env=None):
    return extract_state(obs, env=env), extract_image(obs)


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
    max_eval_steps=200,
    debug_rollout=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint is None:
        checkpoint = os.environ.get("CHECKPOINT_PATH", f"/checkpoints/{task}_bc_best.pt")

    if instruction is None:
        instruction = os.environ.get("TASK_INSTRUCTION", task)

    print(f"[INFO] Using device: {device}", flush=True)
    print(f"[INFO] Task: {task}", flush=True)
    print(f"[INFO] Checkpoint: {checkpoint}", flush=True)
    print(f"[INFO] Episodes: {episodes}", flush=True)
    print(f"[INFO] History len: {history_len}", flush=True)
    print(f"[INFO] Instruction: {instruction}", flush=True)
    print(f"[INFO] CLIP name: {clip_name}", flush=True)
    print(f"[INFO] Action clip: {action_clip}", flush=True)
    print(f"[INFO] Max eval steps: {max_eval_steps}", flush=True)
    print(f"[INFO] Debug rollout: {debug_rollout}", flush=True)

    model, tokenizer = load_model(checkpoint, device, clip_name=clip_name)

    # IMPORTANT: keep structured obs
    env = make_env(task, env_kwargs={"flatten_obs": False})

    success_count = 0
    returns = []

    for ep in range(episodes):
        print(f"[Episode {ep}] reset start", flush=True)
        obs, info = env.reset(seed=ep)
        print(f"[Episode {ep}] reset done info={info}", flush=True)

        state, image = parse_obs(obs, env=env)
        if debug_rollout:
            print(
                f"[Episode {ep}] parsed initial obs: state_shape={state.shape} image_shape={image.shape}",
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

        while not done and step < max_eval_steps:
            hs = np.stack(history_states[-history_len:])
            hi = np.stack(history_images[-history_len:])

            if debug_rollout or step == 0 or (step + 1) % 25 == 0:
                print(f"[Episode {ep}] step {step}: selecting action", flush=True)
            action, memory = select_action(
                model=model,
                tokenizer=tokenizer,
                states=hs,
                images=hi,
                instruction=instruction,
                device=device,
                memory=memory,
                action_clip=action_clip,
            )
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if debug_rollout or step == 0 or (step + 1) % 25 == 0:
                print(
                    f"[Episode {ep}] step {step}: action_shape={action.shape} "
                    f"min={float(np.min(action)):.4f} max={float(np.max(action)):.4f}",
                    flush=True,
                )
                print(f"[Episode {ep}] step {step}: env.step start", flush=True)

            obs, reward, term, trunc, info = env.step(action)
            if debug_rollout or step == 0 or (step + 1) % 25 == 0:
                print(
                    f"[Episode {ep}] step {step}: env.step done "
                    f"reward={reward} term={term} trunc={trunc} info={info}",
                    flush=True,
                )
            done = bool(np.asarray(_to_numpy(term)).any()) or bool(np.asarray(_to_numpy(trunc)).any())

            if render and hasattr(env, "render"):
                env.render()

            total_reward += float(np.asarray(_to_numpy(reward)).reshape(-1)[0])

            state, image = parse_obs(obs, env=env)
            history_states.append(state.copy())
            history_images.append(image.copy())

            step += 1

        if step >= max_eval_steps and not done:
            print(
                f"[Episode {ep}] reached max_eval_steps={max_eval_steps}; forcing episode stop",
                flush=True,
            )
        print(f"episode length: {step}", flush=True)

        success = info.get("success", False) if isinstance(info, dict) else False
        success_bool = bool(np.asarray(_to_numpy(success)).any())
        success_count += int(success_bool)
        returns.append(total_reward)

        print(f"[Episode {ep}] success={success_bool} return={total_reward:.3f}", flush=True)

    print("\n===== FINAL RESULTS =====", flush=True)
    print(f"Success rate: {success_count / episodes:.3f}", flush=True)
    print(f"Avg return: {np.mean(returns):.3f}", flush=True)


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
    parser.add_argument("--max-eval-steps", type=int, default=200)
    parser.add_argument("--debug-rollout", action="store_true")
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
    )
    print("=== Evaluation finished ===", flush=True)