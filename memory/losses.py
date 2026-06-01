import torch
import torch.nn.functional as F

PTP_LAMBDA = 0.2  # keep consistent with training script


# -----------------------------
# Behavior Cloning Loss
# -----------------------------
def action_loss(pred_action, gt_action):
    """
    pred_action: (B, A)
    gt_action:   (B, A)
    """
    return F.mse_loss(pred_action, gt_action)


# -----------------------------
# Past-Token Prediction Loss
# -----------------------------
def ptp_loss(pred_ptp, gt_past_actions):
    """
    pred_ptp:        (B, T, A)
    gt_past_actions: (B, T, A)
    """
    return F.mse_loss(pred_ptp, gt_past_actions)


# -----------------------------
# Full objective (optional helper)
# -----------------------------
def total_loss(pred_action, pred_ptp, gt_action, gt_past_actions):
    loss_bc = action_loss(pred_action, gt_action)
    loss_p = ptp_loss(pred_ptp, gt_past_actions)

    return loss_bc + PTP_LAMBDA * loss_p