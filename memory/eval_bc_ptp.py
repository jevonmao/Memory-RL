import torch
import numpy as np

from .model import SimpleVLA
from env.robomme_env import make_env


# -----------------------------
# Load model
# -----------------------------
def load_model(checkpoint_path, device):
    model = SimpleVLA()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# -----------------------------
# Action selection
# -----------------------------
@torch.no_grad()
def select_action(model, states, images, device):
    states = torch.tensor(states, dtype=torch.float32).unsqueeze(0).to(device)
    images = torch.tensor(images, dtype=torch.float32).unsqueeze(0).to(device)

    action, _, _ = model(states, images)
    return action.squeeze(0).cpu().numpy()


# -----------------------------
# OBS INSPECTION (SAFE)
# -----------------------------
def debug_obs(obs, once=False):
    if not once:
        return

    print("\n===== OBS DEBUG =====")
    print("type:", type(obs))

    if isinstance(obs, dict):
        print("keys:", list(obs.keys()))
        for k in list(obs.keys())[:5]:
            v = obs[k]
            print(f"{k}: type={type(v)}, len={len(v) if hasattr(v,'__len__') else 'NA'}")

    else:
        arr = np.asarray(obs)
        print("shape:", arr.shape)
        print("dtype:", arr.dtype)
        print("first 10:", arr[:10])


# -----------------------------
# STATE EXTRACTION
# -----------------------------
def extract_state(obs):
    """
    RoboMME state extractor (robust across configs)
    """

    # flattened env already returns ndarray
    if not isinstance(obs, dict):
        return np.asarray(obs, dtype=np.float32)

    keys = ["eef_state_list", "joint_state_list", "gripper_state_list"]
    parts = []

    for k in keys:
        if k in obs:
            v = obs[k]
            v = v[-1] if isinstance(v, (list, tuple)) else v
            parts.append(np.asarray(v).flatten())

    if len(parts) == 0:
        raise KeyError(f"[State] No valid keys found. Available: {list(obs.keys())}")

    return np.concatenate(parts).astype(np.float32)


# -----------------------------
# IMAGE EXTRACTION
# -----------------------------
def extract_image(obs):
    """
    Extract RGB image (CHW)
    """

    # fallback dummy
    fallback = np.zeros((3, 64, 64), dtype=np.float32)

    if not isinstance(obs, dict):
        return fallback

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
            img = np.asarray(img)

            # handle uint8 images
            if img.dtype == np.uint8:
                img = img.astype(np.float32)

            # normalize if needed
            if img.max() > 1.5:
                img = img / 255.0

            # HWC → CHW
            if img.ndim == 3 and img.shape[-1] == 3:
                img = np.transpose(img, (2, 0, 1))

            return img.astype(np.float32)

    print("[WARN] No image key found. Available keys:", list(obs.keys()))
    return fallback


def parse_obs(obs):
    return extract_state(obs), extract_image(obs)


# -----------------------------
# EVALUATION
# -----------------------------
def evaluate(
    task="BinFill",
    checkpoint="memory/BinFill_bc_best.pt",
    episodes=20,
    history_len=8,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = load_model(checkpoint, device)

    # IMPORTANT: keep structured obs
    env = make_env(task, env_kwargs={"flatten_obs": False})

    success_count = 0
    returns = []

    for ep in range(episodes):

        obs, info = env.reset(seed=ep)

        # debug only first episode
        debug_obs(obs, once=(ep == 0))

        state, image = parse_obs(obs)

        print("\ninitial state:", state[:10])

        # IMPORTANT: avoid aliasing bug
        history_states = [state.copy() for _ in range(history_len)]
        history_images = [image.copy() for _ in range(history_len)]

        done = False
        total_reward = 0
        step = 0

        while not done:

            hs = np.stack(history_states[-history_len:])
            hi = np.stack(history_images[-history_len:])

            if step == 0:
                print(f"\n[Episode {ep}]")
                print("state shape:", state.shape)
                print("image shape:", image.shape)
                print("history state shape:", hs.shape)
                print("history image shape:", hi.shape)

            action = select_action(model, hs, hi, device)

            if step == 0:
                print("action sample:", action)
                print("image min/max:", image.min(), image.max())

            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

            total_reward += float(reward)

            state, image = parse_obs(obs)

            if step < 5:
                print(f"step={step}")
                print("state[:3] =", state[:3])
                print("action =", action)

            history_states.append(state)
            history_images.append(image)

            step += 1

        print("episode length:", step)

        success = info.get("success", False)
        success_count += int(bool(success))
        returns.append(total_reward)

        print(f"[Episode {ep}] success={success} return={total_reward:.3f}")

    print("\n===== FINAL RESULTS =====")
    print(f"Success rate: {success_count / episodes:.3f}")
    print(f"Avg return: {np.mean(returns):.3f}")


if __name__ == "__main__":
    print("=== Starting evaluation ===")
    evaluate()
    print("=== Evaluation finished ===")