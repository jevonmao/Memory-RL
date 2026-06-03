from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.utils import (
    build_run_dir,
    load_yaml,
    merge_overrides,
    save_run_config,
    set_global_seed,
)

from memory.model import SimpleVLA


# --------------------------------------------------
# Value head wrapper
# --------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, base_model: SimpleVLA, action_dim: int):
        super().__init__()
        self.base = base_model

        # assumes hidden comes out of model
        self.value_head = nn.Linear(base_model.hidden_dim, 1)

        # log std (learnable)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, states, images):
        action_mean, ptp, hidden = self.base(states, images)
        value = self.value_head(hidden)
        return action_mean, value


# --------------------------------------------------
# PPO utilities
# --------------------------------------------------
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    adv = []
    gae = 0
    values = values + [0]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv.insert(0, gae)

    returns = [a + v for a, v in zip(adv, values[:-1])]
    return adv, returns


def log_prob_gaussian(x, mean, log_std):
    std = torch.exp(log_std)
    var = std ** 2
    return -0.5 * (((x - mean) ** 2) / var + 2 * log_std + np.log(2 * np.pi))


# --------------------------------------------------
# PPO training
# --------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--bc_ckpt", required=True)
    return ap.parse_args()


def load_bc(model, path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.base.load_state_dict(state, strict=False)
    print("[BC] loaded checkpoint")


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    set_global_seed(cfg["seed"], False)

    run_dir = build_run_dir(cfg["output_dir"], cfg["task_name"], cfg["seed"])
    save_run_config(run_dir, cfg)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # --------------------------------------------------
    # Env
    # --------------------------------------------------
    from env.robomme_env import make_env

    env = make_env(cfg["task_name"], seed=cfg["seed"], env_kwargs=cfg.get("env_kwargs", {}))

    # infer dims
    obs = env.reset()
    action_dim = env.action_space.shape[0]

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    base = SimpleVLA().to(device)
    model = ActorCritic(base, action_dim).to(device)

    load_bc(model, args.bc_ckpt)

    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    # --------------------------------------------------
    # PPO hyperparams
    # --------------------------------------------------
    clip_eps = cfg["clip_range"]
    gamma = cfg["gamma"]
    lam = cfg["gae_lambda"]

    batch_size = cfg["batch_size"]
    update_epochs = cfg.get("n_epochs", 10)

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for iteration in range(cfg["total_timesteps"] // cfg["n_steps"]):

        states_buf, images_buf = [], []
        actions_buf, rewards_buf, dones_buf = [], [], []
        values_buf, logp_buf = [], []

        state = env.reset()

        # --------------------------------------------------
        # rollout
        # --------------------------------------------------
        for t in range(cfg["n_steps"]):

            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            # assume env gives image + state (adapt if needed)
            images = torch.zeros_like(state_t).to(device)  # replace if real images exist

            with torch.no_grad():
                action_mean, value = model(state_t, images)

                std = torch.exp(model.log_std)
                dist = torch.distributions.Normal(action_mean, std)
                action = dist.sample()
                logp = dist.log_prob(action).sum(dim=-1)

            next_state, reward, done, info = env.step(action.cpu().numpy()[0])

            # store
            states_buf.append(state)
            actions_buf.append(action.cpu().numpy()[0])
            rewards_buf.append(reward)
            dones_buf.append(done)
            values_buf.append(value.item())
            logp_buf.append(logp.item())

            state = next_state

            if done:
                state = env.reset()

        # --------------------------------------------------
        # compute returns
        # --------------------------------------------------
        _, returns = compute_gae(rewards_buf, values_buf, dones_buf, gamma, lam)

        # convert to tensors
        states = torch.tensor(np.array(states_buf), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions_buf), dtype=torch.float32).to(device)
        old_logp = torch.tensor(logp_buf, dtype=torch.float32).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # --------------------------------------------------
        # PPO update
        # --------------------------------------------------
        for _ in range(update_epochs):

            action_mean, values = model(states, torch.zeros_like(states))
            std = torch.exp(model.log_std)

            dist = torch.distributions.Normal(action_mean, std)
            logp = dist.log_prob(actions).sum(dim=-1)

            ratio = torch.exp(logp - old_logp)

            advantages = returns - values.squeeze().detach()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (values.squeeze() - returns).pow(2).mean()
            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[iter {iteration}] loss={loss.item():.4f}")


if __name__ == "__main__":
    main()