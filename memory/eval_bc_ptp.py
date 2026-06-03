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
# 🔥 ROBUST OBS PARSING (FIXED)
# -----------------------------
def extract_state(obs):
    """
    Build 1D state vector from RoboMME dict.
    """

    if not isinstance(obs, dict):
        return np.asarray(obs, dtype=np.float32)

    parts = []

    for k in ["eef_state_list", "joint_state_list", "gripper_state_list"]:
        if k in obs:
            v = obs[k]
            v = v[-1] if isinstance(v, (list, tuple)) else v
            parts.append(np.asarray(v).flatten())

    if len(parts) == 0:
        raise KeyError(f"[State] No valid keys in obs: {list(obs.keys())}")

    return np.concatenate(parts).astype(np.float32)


def extract_image(obs):
    """
    Extract RGB image in CHW format.
    """

    if not isinstance(obs, dict):
        return np.zeros((3, 64, 64), dtype=np.float32)

    # try common RoboMME camera keys
    for k in [
        "front_rgb_list",
        "rgb_list",
        "camera_rgb_list",
        "image",
    ]:
        if k in obs:
            img = obs[k]
            img = img[-1] if isinstance(img, (list, tuple)) else img
            img = np.asarray(img)

            # HWC → CHW
            if img.ndim == 3 and img.shape[-1] == 3:
                img = np.transpose(img, (2, 0, 1))

            img = img.astype(np.float32)

            return img

    # fallback (IMPORTANT: avoid silent failure)
    print("[WARN] No image found in obs keys:", obs.keys())
    return np.zeros((3, 64, 64), dtype=np.float32)


def parse_obs(obs):
    return extract_state(obs), extract_image(obs)


# -----------------------------
# Evaluation
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
    env = make_env(task)

    success_count = 0
    returns = []

    for ep in range(episodes):

        obs, info = env.reset(seed=ep)

        # 🔥 DEBUG: dump obs structure ONCE
        if ep == 0:
            print("\n===== OBS DEBUG =====")
            print("type:", type(obs))
            print("shape:", getattr(obs, "shape", None))
            print("dtype:", getattr(obs, "dtype", None))
            print("first 10 values:", obs[:10])

        state, image = parse_obs(obs)

        print("\ninitial state:", state[:10])

        history_states = [state] * history_len
        history_images = [image] * history_len

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

            total_reward += reward

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
        success_count += int(success)
        returns.append(total_reward)

        print(f"[Episode {ep}] success={success} return={total_reward:.3f}")

    print("\n===== FINAL RESULTS =====")
    print(f"Success rate: {success_count / episodes:.3f}")
    print(f"Avg return: {np.mean(returns):.3f}")


if __name__ == "__main__":
    print("=== Starting evaluation ===")
    evaluate()
    print("=== Evaluation finished ===")