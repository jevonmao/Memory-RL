import torch
import numpy as np
from pathlib import Path

from .model import SimpleVLA
from .dataset import RoboVLAPTPDataset  # only for preprocessing helpers if needed

# IMPORTANT: use your real env
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
# Policy wrapper
# -----------------------------
@torch.no_grad()
def select_action(model, history_states, history_images, device):
    """
    history_states: (T, state_dim)
    history_images: (T, C, H, W)
    """

    states = torch.tensor(history_states, dtype=torch.float32).unsqueeze(0).to(device)
    images = torch.tensor(history_images, dtype=torch.float32).unsqueeze(0).to(device)

    action, _, _ = model(states, images)
    return action.squeeze(0).cpu().numpy()


# -----------------------------
# Evaluation loop
# -----------------------------
def evaluate(
    task="PickXtimes",
    checkpoint="memory/BinFill_bc_best.pt",
    episodes=20,
    history_len=8,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(checkpoint, device)

    env = make_env(task)

    success_count = 0
    returns = []

    for ep in range(episodes):

        obs, info = env.reset(seed=ep)

        history_states = []
        history_images = []

        done = False
        total_reward = 0

        while not done:

            # ---------------------------------
            # build history window
            # ---------------------------------
            if len(history_states) < history_len:
                pad_state = np.zeros_like(history_states[-1]) if history_states else np.zeros(15)
                pad_img = np.zeros((3, 64, 64))

                while len(history_states) < history_len:
                    history_states.insert(0, pad_state)
                    history_images.insert(0, pad_img)

            hs = np.array(history_states[-history_len:])
            hi = np.array(history_images[-history_len:])

            # ---------------------------------
            # policy action
            # ---------------------------------
            action = select_action(model, hs, hi, device)

            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

            total_reward += reward

            # ---------------------------------
            # update history
            # ---------------------------------
            state = obs["state"] if isinstance(obs, dict) else obs
            image = obs["image"] if isinstance(obs, dict) else np.zeros((3, 64, 64))

            history_states.append(state)
            history_images.append(image)

        success = info.get("success", False)

        success_count += int(success)
        returns.append(total_reward)

        print(f"[Episode {ep}] success={success} return={total_reward:.3f}")

    print("\n===== FINAL RESULTS =====")
    print(f"Success rate: {success_count / episodes:.3f}")
    print(f"Avg return: {np.mean(returns):.3f}")


if __name__ == "__main__":
    evaluate()