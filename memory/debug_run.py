import torch

from .dataset import RoboVLAPTPDataset, collate_fn
from .model import CLIPMemoryVLA
from .losses import action_loss, ptp_loss
from transformers import CLIPModel

DATA_PATH = "data/data/robomme_data_h5/record_dataset_BinFill.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HISTORY_LEN = 4
BATCH_SIZE = 2


def load_clip():
    return CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


def main():

    print("\n============================")
    print("🚀 DEBUG TRAINING HARNESS")
    print("============================\n")

    dataset = RoboVLAPTPDataset(
        h5_files=[DATA_PATH],
        history_len=HISTORY_LEN,
        debug=True,
        max_episodes=2,
        max_samples=10,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    batch = next(iter(loader))

    print("\n[CHECK] Batch shapes")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k:20s} {tuple(v.shape)}")

    clip = load_clip()

    model = CLIPMemoryVLA(
        clip_model=clip,
    ).to(DEVICE)

    states = batch["history_states"].to(DEVICE)
    images = batch["history_images"].to(DEVICE)

    text_tokens = torch.zeros(
        (states.shape[0], 10),
        dtype=torch.long,
        device=DEVICE
    )

    print("\n[CHECK] Running forward pass...")

    # ❌ DO NOT use torch.no_grad()
    out = model(images, states, text_tokens)

    print("\n[CHECK] Output shapes")
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"{k:20s} {tuple(v.shape)}")

    print("\n[CHECK] Loss computation")

    pred_action = out["action_pred"]        # (B, A)
    pred_ptp = out["ptp_pred"]              # (B, T, A)

    gt_action = batch["current_action"].to(DEVICE)
    gt_ptp = batch["past_actions"].to(DEVICE)

    print("pred_action:", pred_action.shape)
    print("gt_action  :", gt_action.shape)

    loss_a = action_loss(pred_action, gt_action)
    loss_p = ptp_loss(pred_ptp, gt_ptp)

    loss = loss_a + 0.2 * loss_p

    print("\n[CHECK] Loss values")
    print("action loss:", float(loss_a))
    print("ptp loss   :", float(loss_p))
    print("total loss :", float(loss))

    print("\n[CHECK] Backprop test")

    loss.backward()

    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item()

    print("grad norm:", grad_norm)

    print("\n🎉 DEBUG RUN SUCCESSFUL\n")


if __name__ == "__main__":
    main()