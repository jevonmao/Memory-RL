import sys
sys.path.append("/workspace")

from pathlib import Path
import os

import torch
from torch.utils.data import DataLoader, random_split

import wandb

from .dataset import RoboVLAPTPDataset, collate_fn
from .model import SimpleVLA
from .losses import action_loss, ptp_loss

# -----------------------------
# Paths / Hyperparameters
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "data"
    / "robomme_data_h5"
    / "record_dataset_BinFill.h5"
)

TASK_NAME = "BinFill"

NUM_EPOCHS = 30
BATCH_SIZE = 64
LR = 3e-4

HISTORY_LEN = 8
PTP_WEIGHT = 0.2

VAL_RATIO = 0.1
SAVE_EVERY = 5


# -----------------------------
# Train
# -----------------------------
def train():

    # =========================================================
    # CHECKPOINT SETUP (IMPORTANT FOR MODAL VOLUME)
    # =========================================================
    CHECKPOINT_DIR = Path("/checkpoints")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Train] checkpoint dir = {CHECKPOINT_DIR}")
    print(f"[Train] exists = {CHECKPOINT_DIR.exists()}")

    # =========================================================
    # WANDB (SAFE CHECK)
    # =========================================================
    use_wandb = os.environ.get("WANDB_API_KEY") is not None

    if use_wandb:
        wandb.init(
            project="memory-rl",
            name=f"bc-ptp-{TASK_NAME}",
            config={
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "history_len": HISTORY_LEN,
                "ptp_weight": PTP_WEIGHT,
            },
        )

    # =========================================================
    # DATASET
    # =========================================================
    dataset = RoboVLAPTPDataset(
        h5_files=[str(DATA_PATH)],
        history_len=HISTORY_LEN,
    )

    print(f"Dataset size: {len(dataset)}")

    # =========================================================
    # SPLIT
    # =========================================================
    n_total = len(dataset)
    n_val = int(n_total * VAL_RATIO)
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    # =========================================================
    # MODEL
    # =========================================================
    model = SimpleVLA()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    # =========================================================
    # TRAIN LOOP
    # =========================================================
    for epoch in range(NUM_EPOCHS):

        # ---------------------
        # TRAIN
        # ---------------------
        model.train()

        train_loss = train_a = train_p = 0.0

        for batch in train_loader:

            states = batch["history_states"].to(device)
            images = batch["history_images"].to(device)

            gt_action = batch["current_action"].to(device)
            gt_past_actions = batch["past_actions"].to(device)

            pred_action, pred_ptp, _ = model(states, images)

            loss_a = action_loss(pred_action, gt_action)
            loss_p = ptp_loss(pred_ptp, gt_past_actions)

            loss = loss_a + PTP_WEIGHT * loss_p

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_a += loss_a.item()
            train_p += loss_p.item()

        train_loss /= len(train_loader)
        train_a /= len(train_loader)
        train_p /= len(train_loader)

        # ---------------------
        # VALIDATION
        # ---------------------
        model.eval()

        val_loss = val_a = val_p = 0.0

        with torch.no_grad():
            for batch in val_loader:

                states = batch["history_states"].to(device)
                images = batch["history_images"].to(device)

                gt_action = batch["current_action"].to(device)
                gt_past_actions = batch["past_actions"].to(device)

                pred_action, pred_ptp, _ = model(states, images)

                loss_a = action_loss(pred_action, gt_action)
                loss_p = ptp_loss(pred_ptp, gt_past_actions)

                loss = loss_a + PTP_WEIGHT * loss_p

                val_loss += loss.item()
                val_a += loss_a.item()
                val_p += loss_p.item()

        val_loss /= len(val_loader)
        val_a /= len(val_loader)
        val_p /= len(val_loader)

        # ---------------------
        # LOGGING
        # ---------------------
        print(
            f"[Epoch {epoch}] "
            f"train={train_loss:.4f} val={val_loss:.4f} | "
            f"train_a={train_a:.4f} val_a={val_a:.4f} | "
            f"train_p={train_p:.4f} val_p={val_p:.4f}"
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/action_loss": train_a,
                "train/ptp_loss": train_p,
                "val/loss": val_loss,
                "val/action_loss": val_a,
                "val/ptp_loss": val_p,
            })

        # =========================================================
        # BEST CHECKPOINT (CRITICAL)
        # =========================================================
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_path = CHECKPOINT_DIR / f"{TASK_NAME}_bc_best.pt"

            print(f"[Checkpoint] Saving BEST -> {best_path}")

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_path,
            )

            if use_wandb:
                wandb.run.summary["best_val_loss"] = best_val_loss

        # =========================================================
        # PERIODIC CHECKPOINT
        # =========================================================
        if epoch % SAVE_EVERY == 0:

            ckpt_path = CHECKPOINT_DIR / f"{TASK_NAME}_epoch_{epoch}.pt"

            print(f"[Checkpoint] Saving EPOCH -> {ckpt_path}")

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                },
                ckpt_path,
            )

    # =========================================================
    # FINAL CHECK
    # =========================================================
    print("\nTraining complete.")
    print("Final checkpoint directory contents:")

    for f in sorted(CHECKPOINT_DIR.glob("*.pt")):
        print(" -", f)

    if use_wandb:
        wandb.finish()


# -----------------------------
# ENTRY
# -----------------------------
def main():
    train()


if __name__ == "__main__":
    main()