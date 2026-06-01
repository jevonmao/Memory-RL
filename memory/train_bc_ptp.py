import torch
from torch.utils.data import DataLoader

from dataset import RoboVLAPTPDataset, collate_fn
from model import SimpleVLA
from losses import action_loss, ptp_loss

from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "data" / "robomme_data_h5" / "record_dataset_PickXtimes.h5"

NUM_EPOCHS = 2
BATCH_SIZE = 64
LR = 3e-4

# -----------------------------
# Train
# -----------------------------
def train():

    # -------------------------
    # Dataset
    # -------------------------
    dataset = RoboVLAPTPDataset(
        h5_files=[str(DATA_PATH)],
        history_len=8,
        debug=True,
        max_episodes=5, 
        max_samples=2000
    )

    dataset.episodes = dataset.episodes[:5]
    
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    # -------------------------
    # Model
    # -------------------------
    model = SimpleVLA()
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(NUM_EPOCHS):

        total_loss = 0.0

        for batch in loader:

            # -------------------------
            # Data
            # -------------------------
            states = batch["history_states"].to(device)     # (B,T,15)
            images = batch["history_images"].to(device)     # (B,T,3,H,W)

            gt_action = batch["current_action"].to(device)  # (B,8)
            gt_past_actions = batch["past_actions"].to(device)  # (B,T,8)

            # -------------------------
            # Forward
            # -------------------------
            pred_action, pred_ptp, _ = model(states, images)

            # -------------------------
            # Losses
            # -------------------------
            loss_action = action_loss(pred_action, gt_action)

            loss_ptp = ptp_loss(
                pred_ptp,
                gt_past_actions
            )

            loss = loss_action + 0.2 * loss_ptp

            # -------------------------
            # Backprop
            # -------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(
            f"[epoch {epoch}] "
            f"loss = {avg_loss:.4f}"
        )


# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    train()