"""Behaviour-clone a policy on the PickXtimes expert dataset produced by
`collect_bc_pickxtimes.py`. Outputs a state_dict loadable via the
`--bc_warmstart` flag in train_ppo_recurrent_rnd.py / train_ppo_rnd.py.

Approach (deliberately minimal):
  * Build the same MultiInputPolicy / MultiInputLstmPolicy that the RL
    trainer will use, with the same CNN feature extractor.
  * Train the policy's action-net + extractor heads to regress (MSE) the
    expert action — squashed-Gaussian mean head, tanh applied implicitly
    by squash_output=True at PPO time.
  * For the recurrent variant we BC the non-recurrent path (ignoring LSTM
    state) since BC is per-step; the LSTM weights are still initialised
    in the resulting state_dict so loading via load_state_dict(strict=False)
    just leaves them at the LSTM's random init.

Usage:
    python -m scripts.bc_pretrain_pickxtimes \\
        --dataset runs/bc/pickxtimes/bc_dataset_pickxtimes.pt \\
        --out runs/bc/pickxtimes/bc_policy.pt \\
        --epochs 30 --batch_size 256 --recurrent
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

from train.envs.rl_env import IMG_H, IMG_W, ACTION_DIM
from train.models.encoder import RobommeCNNExtractor


# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  required=True)
    p.add_argument("--out",      required=True)
    p.add_argument("--recurrent", action="store_true",
                   help="Use a MultiInputLstmPolicy (sb3-contrib) instead of MultiInputPolicy.")
    p.add_argument("--lstm_hidden", type=int, default=256)
    p.add_argument("--lstm_layers", type=int, default=1)
    p.add_argument("--epochs",   type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr",       type=float, default=3e-4)
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",     type=int, default=0)
    return p.parse_args()


def make_obs_action_space():
    obs_space = spaces.Dict({
        "front_rgb":   spaces.Box(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8),
        "wrist_rgb":   spaces.Box(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8),
        "joint_state": spaces.Box(-np.inf, np.inf, (7,),  dtype=np.float32),
        "eef_state":   spaces.Box(-np.inf, np.inf, (6,),  dtype=np.float32),
        "gripper":     spaces.Box(-1., 1.,           (2,),  dtype=np.float32),
    })
    act_space = spaces.Box(-1., 1., (ACTION_DIM,), dtype=np.float32)
    return obs_space, act_space


def build_policy(recurrent: bool, lstm_hidden: int, lstm_layers: int, device: str):
    """Construct the SB3 policy *only* (no env, no algorithm) so we can BC it.

    Trick: build a thin fake env so SB3 will construct the policy normally,
    then yank the policy out.
    """
    obs_space, act_space = make_obs_action_space()

    class _FakeEnv(gym.Env):
        def __init__(self): self.observation_space = obs_space; self.action_space = act_space
        def reset(self, *, seed=None, options=None): return obs_space.sample(), {}
        def step(self, a): return obs_space.sample(), 0.0, True, False, {}

    vec = DummyVecEnv([lambda: _FakeEnv()])

    if recurrent:
        from sb3_contrib import RecurrentPPO
        pk = dict(
            features_extractor_class=RobommeCNNExtractor,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            lstm_hidden_size=lstm_hidden,
            n_lstm_layers=lstm_layers,
            shared_lstm=True,
            enable_critic_lstm=False,
        )
        algo = RecurrentPPO(
            "MultiInputLstmPolicy", vec, policy_kwargs=pk, device=device, verbose=0, seed=0,
            n_steps=16, batch_size=16, n_epochs=1,
        )
    else:
        pk = dict(
            features_extractor_class=RobommeCNNExtractor,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            squash_output=True,
        )
        algo = PPO(
            "MultiInputPolicy", vec, policy_kwargs=pk, device=device, verbose=0, seed=0,
            use_sde=True, sde_sample_freq=4,
            n_steps=16, batch_size=16, n_epochs=1,
        )
    return algo.policy


# ----------------------------------------------------------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"[bc] loading {args.dataset}")
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    obs = data["obs"]
    actions = data["actions"].astype(np.float32)
    N = actions.shape[0]
    print(f"[bc] N={N}  ep_lengths={data['ep_lengths'].tolist()[:5]}...  "
          f"actions shape={actions.shape}")

    policy = build_policy(args.recurrent, args.lstm_hidden, args.lstm_layers, args.device)
    policy.set_training_mode(True)
    print(f"[bc] policy={'MultiInputLstmPolicy' if args.recurrent else 'MultiInputPolicy'} "
          f"params={sum(p.numel() for p in policy.parameters())}")

    opt = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=args.lr)

    device = args.device
    a_t = torch.as_tensor(actions, dtype=torch.float32)

    def get_minibatch(idx):
        ob = {k: torch.as_tensor(v[idx]).to(device) for k, v in obs.items()}
        ac = a_t[idx].to(device)
        return ob, ac

    indices = np.arange(N)
    losses = []
    for epoch in range(args.epochs):
        np.random.shuffle(indices)
        ep_loss = 0.0; ep_n = 0
        t0 = time.time()
        for start in range(0, N, args.batch_size):
            mb_idx = indices[start:start + args.batch_size]
            ob, ac = get_minibatch(mb_idx)

            # Forward through the policy's action mean head. For SB3
            # ActorCriticPolicy / RecurrentActorCriticPolicy, we can use
            # `policy._predict(obs, deterministic=True)` to get the mean.
            # But that swallows the gradient. Use forward instead:
            #   features = policy.extract_features(obs)
            #   latent_pi = policy.mlp_extractor.policy_net(features)  [non-recurrent]
            #   mean_actions = policy.action_net(latent_pi)
            # For recurrent: features → LSTM → mlp_extractor → action_net.
            features = policy.extract_features(ob)
            if args.recurrent:
                # Run a 1-step LSTM with episode_starts=True so hidden state resets
                # per sample — i.e. treat this BC as memoryless to keep the loss simple.
                from sb3_contrib.common.recurrent.type_aliases import RNNStates
                B = features.shape[0]
                init = (torch.zeros(args.lstm_layers, B, args.lstm_hidden, device=device),
                        torch.zeros(args.lstm_layers, B, args.lstm_hidden, device=device))
                episode_starts = torch.ones(B, device=device)
                latent, _ = policy._process_sequence(features, init, episode_starts, policy.lstm_actor)
                latent_pi = policy.mlp_extractor.forward_actor(latent)
            else:
                latent_pi = policy.mlp_extractor.forward_actor(features)
            mean_actions = policy.action_net(latent_pi)

            # With squash_output=True (non-recurrent), the action distribution
            # is Tanh(N(mean, std)). The "deterministic" action is tanh(mean).
            # Match the target action which is already in [-1, 1].
            if args.recurrent:
                # Recurrent uses an unsquashed Gaussian; the env wrapper clips to
                # [-1, 1]. Loss against target works fine without tanh.
                pred = mean_actions
            else:
                pred = torch.tanh(mean_actions)

            loss = ((pred - ac) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach().cpu()) * mb_idx.size
            ep_n += mb_idx.size

        losses.append(ep_loss / max(1, ep_n))
        dt = time.time() - t0
        print(f"[bc] epoch {epoch+1:3d}/{args.epochs}  loss={losses[-1]:.5f}  ({dt:.1f}s)")

    # Save the policy state_dict (on CPU so torch.load with map_location works
    # regardless of the training device).
    state_dict_cpu = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    torch.save(state_dict_cpu, args.out)
    print(f"\n[bc] saved policy state_dict -> {args.out}  ({len(state_dict_cpu)} keys)")


if __name__ == "__main__":
    main()
