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
# Observation parsing (ROBUST)
# -----------------------------
def parse_obs(obs):
    """
    Tries to robustly extract:
    - state vector (D,)
    - image (3,H,W)
    """

    if isinstance(obs, dict):
        # Try common keys
        if "state" in obs:
            state = obs["state"]
        elif "agent" in obs:
            state = obs["agent"]
        elif "proprio" in obs:
            state = obs["proprio"]
        else:
            raise KeyError(f"Unknown state keys: {obs.keys()}")

        if "image" in obs:
            image = obs["image"]
        elif "rgb" in obs:
            image = obs["rgb"]
        else:
            image = np.zeros((64, 64, 3), dtype=np.float32)

    else:
        state = obs
        image = np.zeros((64, 64, 3), dtype=np.float32)

    # -------------------------
    # Fix image format
    # -------------------------
    image = np.asarray(image)

    if image.ndim == 3 and image.shape[-1] == 3:
        # HWC → CHW
        image = np.transpose(image, (2, 0, 1))
    elif image.ndim != 3:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    return np.asarray(state), image


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
        state, image = parse_obs(obs)
        print("initial state:", state[:10])

        # -------------------------
        # FIXED HISTORY INIT
        # -------------------------
        history_states = [state] * history_len
        history_images = [image] * history_len

        done = False
        total_reward = 0

        step = 0

        while not done:

            hs = np.stack(history_states[-history_len:])
            hi = np.stack(history_images[-history_len:])

            # -------------------------
            # DEBUG (lightweight)
            # -------------------------
            if step == 0:
                print(f"\n[Episode {ep}]")
                print("state shape:", state.shape)
                print("image shape:", image.shape)
                print("history state shape:", hs.shape)
                print("history image shape:", hi.shape)

            action = select_action(model, hs, hi, device)

            # DEBUG: action sanity check
            if step == 0:
                print("action sample:", action)
                print("image min/max:", image.min(), image.max())

            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

            total_reward += reward

            # update obs
            state, image = parse_obs(obs)

            if step < 10:
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