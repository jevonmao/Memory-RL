import sys
sys.path.append("/workspace")

from pathlib import Path
import os
import random

import torch
import wandb
from transformers import CLIPModel, CLIPTokenizer

from .dataset import RoboVLAPTPDataset, collate_episode_windows
from .model import CLIPMemoryVLA
from .losses import action_loss, ptp_loss


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "data" / "robomme_data_h5" / "record_dataset_BinFill.h5"

TASK_NAME = "BinFill"
NUM_EPOCHS = 20
BATCH_SIZE = 8          # parallel episodes
LR = 1e-4
HISTORY_LEN = 8
PTP_WEIGHT = 0.2
VAL_RATIO = 0.1
SAVE_EVERY = 5
GRAD_CLIP_NORM = 1.0

DETACH_MEMORY_EACH_STEP = True
RESUME_FROM_BEST = True


def commit_checkpoint_volume_if_available():
    """Commit Modal checkpoint volume after each save.

    This is a no-op outside Modal. Inside Modal, train_bc_ptp_modal.py should set
    CHECKPOINT_VOLUME_NAME to the mounted volume name.
    """
    volume_name = os.environ.get("CHECKPOINT_VOLUME_NAME")
    if not volume_name:
        return

    try:
        import modal

        volume = modal.Volume.from_name(volume_name)
        volume.commit()
        print(f"[Checkpoint] committed Modal volume {volume_name}", flush=True)
    except Exception as exc:
        print(f"[Checkpoint][Warning] failed to commit Modal volume {volume_name}: {exc}", flush=True)


def load_clip_model():
    return CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


def unpack_model_outputs(outputs):
    if not isinstance(outputs, dict):
        raise TypeError(f"Expected model output to be a dict, got {type(outputs)}")

    missing = [k for k in ("pred_action", "pred_ptp", "memory") if k not in outputs]
    if missing:
        raise KeyError(f"Missing model output keys {missing}; available keys: {list(outputs.keys())}")

    pred_action = outputs["pred_action"]
    pred_ptp = outputs["pred_ptp"]
    memory = outputs["memory"]

    if not isinstance(pred_action, torch.Tensor):
        raise TypeError(f"pred_action must be a tensor, got {type(pred_action)}")
    if not isinstance(pred_ptp, torch.Tensor):
        raise TypeError(f"pred_ptp must be a tensor, got {type(pred_ptp)}")
    if not isinstance(memory, torch.Tensor):
        raise TypeError(f"memory must be a tensor, got {type(memory)}")

    return pred_action, pred_ptp, memory


def validate_prediction_shapes(pred_action, pred_ptp, gt_action, gt_past_actions):
    if pred_action.shape != gt_action.shape:
        raise ValueError(f"{pred_action.shape=} != {gt_action.shape=}")
    if pred_ptp.shape != gt_past_actions.shape:
        raise ValueError(f"{pred_ptp.shape=} != {gt_past_actions.shape=}")


def chunked(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def run_episode_batch(model, dataset, tokenizer, episode_ids, device, optimizer=None, debug=False):
    is_train = optimizer is not None
    memory = None
    active_episode_ids = list(episode_ids)

    if not active_episode_ids:
        return 0.0, 0.0, 0.0, 0

    total_loss = total_a = total_p = 0.0
    steps = 0
    max_len = max(dataset.episode_length(ep_id) for ep_id in episode_ids)

    for t in range(HISTORY_LEN, max_len):
        samples = []
        keep_positions = []
        next_active = []

        for pos, ep_id in enumerate(active_episode_ids):
            if t < dataset.episode_length(ep_id):
                samples.append(dataset.get_episode_window(ep_id, t))
                keep_positions.append(pos)
                next_active.append(ep_id)

        if not samples:
            continue

        if memory is not None and len(keep_positions) != memory.shape[0]:
            idx = torch.as_tensor(keep_positions, device=memory.device)
            memory = memory.index_select(0, idx)

        batch = collate_episode_windows(samples, device)

        text_tokens = tokenizer(
            batch["instruction"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        outputs = model(
            batch["history_images"],
            batch["history_states"],
            text_tokens,
            memory=memory,
            detach_memory=DETACH_MEMORY_EACH_STEP,
            memory_update="all" if memory is None else "last",
        )

        pred_action, pred_ptp, memory = unpack_model_outputs(outputs)
        validate_prediction_shapes(
            pred_action,
            pred_ptp,
            batch["current_action"],
            batch["past_actions"],
        )

        if debug and steps == 0:
            print("[Debug] parallel episode memory mode")
            print("[Debug] episode batch size:", len(episode_ids))
            print("[Debug] active episodes:", len(samples))
            print("[Debug] model output keys:", list(outputs.keys()))
            print("[Debug] pred_action:", pred_action.shape, pred_action.dtype)
            print("[Debug] gt_action:", batch["current_action"].shape, batch["current_action"].dtype)
            print("[Debug] pred_ptp:", pred_ptp.shape, pred_ptp.dtype)
            print("[Debug] gt_past_actions:", batch["past_actions"].shape, batch["past_actions"].dtype)
            print("[Debug] memory:", memory.shape, memory.dtype)

        loss_a = action_loss(pred_action, batch["current_action"])
        loss_p = ptp_loss(pred_ptp, batch["past_actions"])
        loss = loss_a + PTP_WEIGHT * loss_p

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        total_loss += loss.item()
        total_a += loss_a.item()
        total_p += loss_p.item()
        steps += 1

        memory = memory.detach()
        active_episode_ids = next_active

    return total_loss, total_a, total_p, steps


def train():
    # Save checkpoints to the Modal-mounted persistent volume when running on Modal.
    # This keeps checkpoints available for later online evaluation jobs.
    ckpt_dir = Path(os.environ.get("CHECKPOINT_DIR", "/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Train] checkpoint dir = {ckpt_dir}")
    print(f"[Train] checkpoint dir exists = {ckpt_dir.exists()}")
    print(f"[Train] checkpoint dir resolved = {ckpt_dir.resolve()}")

    use_wandb = os.environ.get("WANDB_API_KEY") is not None
    if use_wandb:
        wandb.init(
            project="memory-rl",
            name=f"bc-ptp-{TASK_NAME}",
            config={
                "epochs": NUM_EPOCHS,
                "parallel_episodes": BATCH_SIZE,
                "lr": LR,
                "history_len": HISTORY_LEN,
                "ptp_weight": PTP_WEIGHT,
                "memory_mode": "episode_persistent_last_update",
            },
        )

    dataset = RoboVLAPTPDataset([str(DATA_PATH)], history_len=HISTORY_LEN)

    if dataset.num_episodes() < 2:
        raise ValueError(f"Need at least 2 episodes for train/val split, got {dataset.num_episodes()}")

    episode_ids = list(range(dataset.num_episodes()))
    rng = random.Random(42)
    rng.shuffle(episode_ids)

    n_val = max(1, int(len(episode_ids) * VAL_RATIO))
    n_val = min(n_val, len(episode_ids) - 1)
    train_episode_ids = episode_ids[:-n_val]
    val_episode_ids = episode_ids[-n_val:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device={device}")
    print(f"Train episodes={len(train_episode_ids)} Val episodes={len(val_episode_ids)}")

    clip_model = load_clip_model()
    model = CLIPMemoryVLA(
        clip_model=clip_model,
        state_dim=15,
        action_dim=8,
        d_model=512,
        num_slots=8,
    ).to(device)

    model.vision.eval()
    model.text.eval()

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )

    start_epoch = 0

    best_path = ckpt_dir / f"{TASK_NAME}_bc_best.pt"
    if RESUME_FROM_BEST and best_path.exists():
        print(f"[Resume] Loading checkpoint from {best_path}")
        checkpoint = torch.load(best_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint.get("epoch", -1) + 1
        best_val = checkpoint.get("val_loss", float("inf"))

        print(
            f"[Resume] Resuming from epoch {start_epoch} "
            f"with best_val={best_val:.4f}"
        )
    else:
        best_val = float("inf")

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        # Keep frozen CLIP encoders in eval mode even after model.train().
        model.vision.eval()
        model.text.eval()
        rng.shuffle(train_episode_ids)

        train_loss = train_a = train_p = 0.0
        train_steps = 0

        for i, ep_batch in enumerate(chunked(train_episode_ids, BATCH_SIZE)):
            loss, la, lp, steps = run_episode_batch(
                model, dataset, tokenizer, ep_batch, device,
                optimizer=optimizer,
                debug=(epoch == 0 and i == 0),
            )
            train_loss += loss
            train_a += la
            train_p += lp
            train_steps += steps

        train_loss /= max(1, train_steps)
        train_a /= max(1, train_steps)
        train_p /= max(1, train_steps)

        model.eval()
        val_loss = val_a = val_p = 0.0
        val_steps = 0

        with torch.no_grad():
            for ep_batch in chunked(val_episode_ids, BATCH_SIZE):
                loss, la, lp, steps = run_episode_batch(
                    model, dataset, tokenizer, ep_batch, device,
                    optimizer=None,
                )
                val_loss += loss
                val_a += la
                val_p += lp
                val_steps += steps

        val_loss /= max(1, val_steps)
        val_a /= max(1, val_steps)
        val_p /= max(1, val_steps)

        print(
            f"[Epoch {epoch}] train={train_loss:.4f} val={val_loss:.4f} | "
            f"train_a={train_a:.4f} val_a={val_a:.4f} | "
            f"train_p={train_p:.4f} val_p={val_p:.4f} | "
            f"steps={train_steps}/{val_steps}"
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
                "train/steps": train_steps,
                "val/steps": val_steps,
            })

        if val_loss < best_val:
            best_val = val_loss
            best_path = ckpt_dir / f"{TASK_NAME}_bc_best.pt"
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
            print(f"[Checkpoint] BEST saved exists={best_path.exists()} size_bytes={best_path.stat().st_size if best_path.exists() else 'missing'}")
            commit_checkpoint_volume_if_available()

        if epoch % SAVE_EVERY == 0:
            ckpt_path = ckpt_dir / f"{TASK_NAME}_epoch_{epoch}.pt"
            print(f"[Checkpoint] Saving EPOCH -> {ckpt_path}")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"[Checkpoint] EPOCH saved exists={ckpt_path.exists()} size_bytes={ckpt_path.stat().st_size if ckpt_path.exists() else 'missing'}")
            commit_checkpoint_volume_if_available()

    print("\nTraining complete.")
    print(f"Final checkpoint directory contents in {ckpt_dir}:")
    for f in sorted(ckpt_dir.glob("*.pt")):
        print(f" - {f} size_bytes={f.stat().st_size}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()