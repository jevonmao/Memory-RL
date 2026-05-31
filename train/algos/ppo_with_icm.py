"""
PPOWithICM: PPO subclass that owns an Intrinsic Curiosity Module (ICM).

Why a subclass and not a callback:
  SB3's OnPolicyAlgorithm.collect_rollouts computes advantages BEFORE firing
  callback.on_rollout_end (see stable_baselines3/common/on_policy_algorithm.py
  lines 262 vs 266). So a callback that modifies buf.rewards in on_rollout_end
  has no effect — PPO already trained on the pre-augmentation advantages.

  Instead we override collect_rollouts and inject the intrinsic reward into
  the rewards array BEFORE rollout_buffer.add(...), so the augmented reward
  enters compute_returns_and_advantage normally.

Reward scaling note:
  When the env is wrapped in VecNormalize(norm_reward=True), the rewards
  returned by env.step() are already normalised against a running estimate.
  We add eta * intrinsic on top — pick eta small enough to be comparable.

ICM gradient ownership (decision #5 of the plan):
  The ICM optimiser owns BOTH icm.parameters() and the policy's extractor
  parameters. PPO's own optimiser also touches the extractor, but each Adam
  maintains independent moment state on a different objective. This is the
  canonical Pathak-ICM-with-shared-encoder pattern, not a double-optimisation
  bug — removing the extractor from this optimiser would prevent the
  inverse-dynamics signal from ever shaping the encoder (PPO.train zeros all
  policy grads before its own backward pass).
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


class PPOWithICM(PPO):
    """PPO with an online ICM intrinsic reward + offline ICM training step."""

    def __init__(
        self,
        *args,
        icm,
        eta: float = 0.01,
        icm_lr: float = 3e-4,
        icm_beta: float = 0.2,
        icm_batch_size: int = 256,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.icm = icm.to(self.device)
        self.eta = float(eta)
        self.icm_beta = float(icm_beta)
        self.icm_batch_size = int(icm_batch_size)

        # ICM optimiser owns icm + extractor (see module docstring).
        extractor = self.policy.features_extractor
        self.icm_opt = optim.Adam(
            list(self.icm.parameters()) + list(extractor.parameters()),
            lr=float(icm_lr),
        )

    # ------------------------------------------------------------------
    # Rollout collection: SB3's logic with intrinsic-reward injection.
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

        # Buffers for the offline ICM update at the end of the rollout.
        icm_obs_t:   list[Any]        = []
        icm_obs_tp1: list[Any]        = []
        icm_actions: list[np.ndarray] = []
        intr_log:    list[float]      = []

        # Cache feat_t across the rollout: feat_tp1 of step N becomes feat_t
        # of step N+1, so we only need ONE extract_features per env step
        # instead of two. Seeded with a fresh extraction on the very first
        # _last_obs of this rollout. (Don't reuse across rollouts because
        # the policy may have just been updated.)
        with th.no_grad():
            cached_feat_t = self.policy.extract_features(
                obs_as_tensor(self._last_obs, self.device)
            )

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

            # ---- Online ICM intrinsic reward (eval-only — no grads here) ----
            with th.no_grad():
                feat_t   = cached_feat_t
                feat_tp1 = self.policy.extract_features(
                    obs_as_tensor(new_obs, self.device)
                )
                # ICM expects a tensor of actions matching feat batch dim.
                actions_t = actions if isinstance(actions, th.Tensor) else th.as_tensor(
                    actions_np, dtype=th.float32, device=self.device
                )
                _, _, intr = self.icm(feat_t, actions_t, feat_tp1)
            cached_feat_t = feat_tp1   # reuse on the next iteration
            intr_np = intr.cpu().numpy().astype(rewards.dtype, copy=False)
            rewards = rewards + self.eta * intr_np
            intr_log.append(float(intr_np.mean()))

            # Stash for the offline grad step.
            icm_obs_t.append(self._last_obs)
            icm_obs_tp1.append(new_obs)
            icm_actions.append(actions_np)

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

        # Offline ICM grad step on the just-collected transitions.
        icm_metrics = self._icm_update(icm_obs_t, icm_obs_tp1, icm_actions)
        self.logger.record("icm/loss",         icm_metrics["loss"])
        self.logger.record("icm/fwd_loss",     icm_metrics["fwd_loss"])
        self.logger.record("icm/inv_loss",     icm_metrics["inv_loss"])
        self.logger.record("icm/intr_reward",  float(np.mean(intr_log)) if intr_log else 0.0)

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    # ------------------------------------------------------------------
    # Offline ICM training step.
    # ------------------------------------------------------------------
    def _icm_update(self,
                    obs_t_list:   list[Any],
                    obs_tp1_list: list[Any],
                    action_list:  list[np.ndarray]) -> dict[str, float]:
        """One pass of minibatch ICM training on the rollout's transitions.

        Gradients flow through the policy's features extractor here — that is
        the whole point of ICM-shaping-the-encoder. PPO.train will independently
        compute its own gradients on the policy/value objective; the two
        optimisers keep separate Adam moment buffers.
        """
        # Concatenate the per-step minibatches (each is shaped (n_envs, ...))
        # into a single (T*n_envs, ...) batch. For Dict obs we concat per key.
        obs_t   = _concat_obs_batches(obs_t_list)
        obs_tp1 = _concat_obs_batches(obs_tp1_list)
        actions = np.concatenate(action_list, axis=0)

        N = actions.shape[0]
        if N == 0:
            return {"loss": 0.0, "fwd_loss": 0.0, "inv_loss": 0.0}

        device = self.device
        actions_t = th.as_tensor(actions, dtype=th.float32, device=device)

        # Minibatch over the rollout to stay within memory.
        idx = np.arange(N)
        np.random.shuffle(idx)
        bs = self.icm_batch_size

        total_loss = 0.0
        total_fwd  = 0.0
        total_inv  = 0.0
        n_batches  = 0

        self.policy.set_training_mode(True)
        for start in range(0, N, bs):
            mb = idx[start:start + bs]
            obs_t_mb   = _slice_obs(obs_t,   mb, device)
            obs_tp1_mb = _slice_obs(obs_tp1, mb, device)
            a_mb       = actions_t[mb]

            # Forward through extractor WITH grad (the C3 fix).
            feat_t   = self.policy.extract_features(obs_t_mb)
            feat_tp1 = self.policy.extract_features(obs_tp1_mb)

            phi_hat, act_hat, _ = self.icm(feat_t, a_mb, feat_tp1)
            loss = self.icm.loss(
                phi_hat, feat_tp1.detach(),   # detach target for forward loss
                act_hat, a_mb,
                beta=self.icm_beta,
            )

            self.icm_opt.zero_grad()
            loss.backward()
            self.icm_opt.step()

            total_loss += float(loss.detach().cpu())
            # Recompute the two loss components for logging (cheap; same tensors).
            with th.no_grad():
                fwd = th.nn.functional.mse_loss(phi_hat, feat_tp1).item()
                inv = th.nn.functional.mse_loss(act_hat, a_mb).item()
            total_fwd += fwd
            total_inv += inv
            n_batches += 1
        self.policy.set_training_mode(False)

        return {
            "loss":     total_loss / max(1, n_batches),
            "fwd_loss": total_fwd  / max(1, n_batches),
            "inv_loss": total_inv  / max(1, n_batches),
        }


# ----------------------------------------------------------------------
# Helpers for Dict / Box obs batching.
# ----------------------------------------------------------------------
def _concat_obs_batches(batches: list[Any]) -> Any:
    """Stack a list of per-step obs batches into one tall batch.

    Each entry may be either a numpy array (Box obs) or a dict of numpy arrays
    (Dict obs). We assume all entries share the same structure.
    """
    if not batches:
        return batches
    first = batches[0]
    if isinstance(first, dict):
        return {k: np.concatenate([b[k] for b in batches], axis=0) for k in first}
    return np.concatenate(batches, axis=0)


def _slice_obs(obs: Any, idx: np.ndarray, device: th.device) -> Any:
    """Index into a concatenated obs (dict or array) and move to device tensor."""
    if isinstance(obs, dict):
        return {k: th.as_tensor(v[idx], device=device) for k, v in obs.items()}
    return th.as_tensor(obs[idx], device=device)
