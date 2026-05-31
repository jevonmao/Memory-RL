"""
PPOWithRND: PPO subclass that owns a Random Network Distillation curiosity head.

Mirror of PPOWithICM (see neighbouring module for the SB3 hook explanation).
The intrinsic reward is injected into the rewards array BEFORE the rollout
buffer's compute_returns_and_advantage runs, so PPO's advantages reflect
the augmented reward signal.

Differences from PPOWithICM:
  * No action input to the curiosity head (RND's input is φ(s) only) — this
    is what kills the noisy-TV failure mode.
  * Predictor gradient does NOT flow into the policy's extractor. The RND
    target is a *random* network, so dragging the encoder toward it would
    regularise the policy toward noise. The predictor MLP is the only thing
    that needs to learn.
  * Running normalisation of (a) the input features φ and (b) the discounted
    intrinsic returns — see RND module docstring.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch as th
import torch.optim as optim
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv


class PPOWithRND(PPO):
    """PPO + RND intrinsic curiosity, with the same external API as PPO."""

    def __init__(
        self,
        *args,
        rnd,
        eta: float = 1.0,
        rnd_lr: float = 1e-4,
        rnd_batch_size: int = 256,
        rnd_gamma: float = 0.99,
        **kwargs,
    ):
        """
        Args:
            rnd: an RND module (train.models.rnd.RND).
            eta: intrinsic-reward coefficient. RND already normalises by
                 return-std so eta ~ 1.0 is the canonical scale; tune only
                 if you see intrinsic dominate / vanish in logs.
            rnd_lr: Adam LR for the predictor.
            rnd_batch_size: minibatch size for the offline predictor update.
            rnd_gamma: discount used to estimate the intrinsic return for the
                       ret_rms tracker. Decoupled from the policy γ.
        """
        super().__init__(*args, **kwargs)
        self.rnd = rnd.to(self.device)
        self.eta = float(eta)
        self.rnd_batch_size = int(rnd_batch_size)
        self.rnd_gamma = float(rnd_gamma)

        # Predictor-only optimiser (rnd.parameters() returns predictor only).
        self.rnd_opt = optim.Adam(self.rnd.parameters(), lr=float(rnd_lr))

    # ------------------------------------------------------------------
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        # Buffers for offline RND update + intrinsic return estimation.
        # We store next-state features (because RND's input is φ(s_{t+1})
        # → r_i depends on novelty of the *arrived* state).
        feat_tp1_buf: list[th.Tensor] = []
        intr_buf:     list[np.ndarray] = []

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions_np = actions.cpu().numpy()

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(actions_np)
                else:
                    clipped_actions = np.clip(
                        actions_np, self.action_space.low, self.action_space.high
                    )
            else:
                clipped_actions = actions_np

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            # ---- Online RND intrinsic reward ----
            with th.no_grad():
                feat_tp1 = self.policy.extract_features(
                    obs_as_tensor(new_obs, self.device)
                )
                # Update obs_rms incrementally so normalisation tracks the
                # distribution of features the policy actually visits.
                self.rnd.update_obs_rms(feat_tp1)
                intr = self.rnd.intrinsic(feat_tp1)   # (n_envs,)
            intr_np = intr.cpu().numpy().astype(rewards.dtype, copy=False)
            rewards = rewards + self.eta * intr_np

            feat_tp1_buf.append(feat_tp1.detach().cpu())   # keep off-GPU between iters
            intr_buf.append(intr_np)

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions_np = actions_np.reshape(-1, 1)

            # Timeout bootstrap — verbatim from SB3.
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions_np,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        # ---- Update intrinsic-return RMS using the rollout's intrinsic reward ----
        # Estimate discounted intrinsic returns per env, then update ret_rms.
        intr_arr = np.asarray(intr_buf, dtype=np.float32)        # (T, n_envs)
        disc_ret = np.zeros_like(intr_arr)
        running  = np.zeros(intr_arr.shape[1], dtype=np.float32)
        for t in reversed(range(intr_arr.shape[0])):
            running = intr_arr[t] + self.rnd_gamma * running
            disc_ret[t] = running
        self.rnd.update_ret_rms(th.as_tensor(disc_ret.flatten()))

        # ---- Offline RND predictor update ----
        rnd_metrics = self._rnd_update(feat_tp1_buf)
        self.logger.record("rnd/loss",        rnd_metrics["loss"])
        self.logger.record("rnd/intr_reward", float(intr_arr.mean()) if intr_arr.size else 0.0)
        self.logger.record("rnd/intr_std",    float(intr_arr.std())  if intr_arr.size else 0.0)
        self.logger.record("rnd/ret_std",     float(self.rnd.ret_rms.std.item()))

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    # ------------------------------------------------------------------
    def _rnd_update(self, feat_tp1_buf: list[th.Tensor]) -> dict[str, float]:
        """Train the predictor on the rollout's φ(s_{t+1}) batch."""
        if not feat_tp1_buf:
            return {"loss": 0.0}
        feats = th.cat(feat_tp1_buf, dim=0)                     # (T*n_envs, feat_dim)
        N = feats.shape[0]
        idx = np.arange(N)
        np.random.shuffle(idx)
        bs = self.rnd_batch_size

        total = 0.0
        n_batches = 0
        # No need to set extractor train mode — predictor doesn't touch it.
        for start in range(0, N, bs):
            mb = idx[start:start + bs]
            f  = feats[mb].to(self.device)
            loss = self.rnd.loss(f)
            self.rnd_opt.zero_grad()
            loss.backward()
            self.rnd_opt.step()
            total += float(loss.detach().cpu())
            n_batches += 1

        return {"loss": total / max(1, n_batches)}
