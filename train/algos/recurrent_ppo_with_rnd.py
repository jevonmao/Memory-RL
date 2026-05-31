"""
RecurrentPPOWithRND — RecurrentPPO (sb3-contrib) + RND intrinsic curiosity.

sb3_contrib.RecurrentPPO has a similar collect_rollouts to PPO's but it also
threads LSTM hidden states through the rollout and stores per-step hidden
states + episode_starts in its RecurrentRolloutBuffer. We override
collect_rollouts in the same spirit as PPOWithICM/RND: inject the intrinsic
reward into the rewards array BEFORE buffer.add(...) so PPO's advantages
reflect the augmented signal.

This file copies the relevant logic from sb3-contrib==2.8.0 / RecurrentPPO
(see sb3_contrib/ppo_recurrent/ppo_recurrent.py for the upstream impl).
The hidden-state / done-mask handling matches the upstream exactly; only
the reward-augmentation block is added.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import spaces

from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv


class RecurrentPPOWithRND(RecurrentPPO):
    """RecurrentPPO with an RND intrinsic-reward head bolted on.

    Optional auxiliary BC regularisation (Set bc_obs / bc_actions at
    construction time). After each PPO update we sample a minibatch from
    the BC dataset and add a `bc_aux_coef * MSE(policy_mean, expert_action)`
    gradient step. This keeps the policy near the expert distribution
    even as RL fine-tunes — closes the "policy drifts away from BC mean
    once std starts to rise" failure we observed without the aux loss.
    """

    def __init__(
        self,
        *args,
        rnd,
        eta: float = 1.0,
        rnd_lr: float = 1e-4,
        rnd_batch_size: int = 256,
        rnd_gamma: float = 0.99,
        bc_obs: Optional[dict] = None,
        bc_actions: Optional[np.ndarray] = None,
        bc_aux_coef: float = 0.0,
        bc_aux_batch_size: int = 256,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rnd = rnd.to(self.device)
        self.eta = float(eta)
        self.rnd_batch_size = int(rnd_batch_size)
        self.rnd_gamma = float(rnd_gamma)
        self.rnd_opt = optim.Adam(self.rnd.parameters(), lr=float(rnd_lr))

        # Auxiliary BC loss state (optional).
        self.bc_obs = bc_obs
        self.bc_actions = bc_actions
        self.bc_aux_coef = float(bc_aux_coef)
        self.bc_aux_batch_size = int(bc_aux_batch_size)

    # ------------------------------------------------------------------
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RecurrentRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None
        assert self._last_lstm_states is not None
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        feat_tp1_buf: list[th.Tensor] = []
        intr_buf:     list[np.ndarray] = []

        lstm_states = self._last_lstm_states

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.as_tensor(
                    self._last_episode_starts, dtype=th.float32, device=self.device
                )
                actions, values, log_probs, lstm_states = self.policy.forward(
                    obs_tensor, lstm_states, episode_starts
                )
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

            # ---- RND intrinsic on the *arrived* state ----
            with th.no_grad():
                feat_tp1 = self.policy.extract_features(
                    obs_as_tensor(new_obs, self.device)
                )
                self.rnd.update_obs_rms(feat_tp1)
                intr = self.rnd.intrinsic(feat_tp1)
            intr_np = intr.cpu().numpy().astype(rewards.dtype, copy=False)
            rewards = rewards + self.eta * intr_np

            feat_tp1_buf.append(feat_tp1.detach().cpu())
            intr_buf.append(intr_np)

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions_np = actions_np.reshape(-1, 1)

            # Timeout bootstrap (same form as PPO but predict_values needs hidden state).
            # NOTE: predict_values takes a SINGLE (h, c) tuple, not RNNStates —
            # see sb3_contrib.common.recurrent.policies.predict_values. When
            # shared_lstm=True the value head reuses lstm_states.pi.
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_lstm = (
                            lstm_states.pi[0][:, idx:idx + 1, :].contiguous(),
                            lstm_states.pi[1][:, idx:idx + 1, :].contiguous(),
                        )
                        terminal_episode_starts = th.zeros(1, dtype=th.float32, device=self.device)
                        terminal_value = self.policy.predict_values(
                            terminal_obs,
                            terminal_lstm,
                            terminal_episode_starts,
                        )[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions_np,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        with th.no_grad():
            episode_starts = th.as_tensor(dones, dtype=th.float32, device=self.device)
            # predict_values expects a single (h, c) tuple, not RNNStates.
            values = self.policy.predict_values(
                obs_as_tensor(new_obs, self.device), lstm_states.pi, episode_starts
            )
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        # ---- Update intrinsic-return RMS ----
        intr_arr = np.asarray(intr_buf, dtype=np.float32)        # (T, n_envs)
        disc_ret = np.zeros_like(intr_arr)
        running  = np.zeros(intr_arr.shape[1], dtype=np.float32)
        for t in reversed(range(intr_arr.shape[0])):
            running = intr_arr[t] + self.rnd_gamma * running
            disc_ret[t] = running
        self.rnd.update_ret_rms(th.as_tensor(disc_ret.flatten()))

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
        if not feat_tp1_buf:
            return {"loss": 0.0}
        feats = th.cat(feat_tp1_buf, dim=0)
        N = feats.shape[0]
        idx = np.arange(N)
        np.random.shuffle(idx)
        bs = self.rnd_batch_size

        total = 0.0
        n_batches = 0
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

    # ------------------------------------------------------------------
    def train(self) -> None:
        """Run normal PPO update + optional auxiliary BC step.

        sb3-contrib's RecurrentPPO.train() runs the PPO clipped-objective
        update. We chain a single BC gradient step after, so the policy
        sees both the on-policy RL signal AND the off-policy expert signal
        every update cycle.
        """
        super().train()
        if self.bc_aux_coef > 0.0 and self.bc_actions is not None:
            metrics = self._bc_aux_update()
            self.logger.record("bc_aux/loss",  metrics["loss"])
            self.logger.record("bc_aux/coef",  self.bc_aux_coef)

    def _bc_aux_update(self) -> dict[str, float]:
        """One BC gradient step on a sample of saved expert transitions.

        Treats each sample memorylessly (LSTM init-zero per sample), matching
        how the recurrent BC was trained originally — keeps loss surface
        consistent across the two training phases.
        """
        N = self.bc_actions.shape[0]
        bs = min(self.bc_aux_batch_size, N)
        idx = np.random.choice(N, bs, replace=False)

        # Build minibatch on device.
        obs_mb = {k: th.as_tensor(v[idx], device=self.device) for k, v in self.bc_obs.items()}
        act_mb = th.as_tensor(self.bc_actions[idx], dtype=th.float32, device=self.device)

        self.policy.set_training_mode(True)
        features = self.policy.extract_features(obs_mb)
        B = features.shape[0]

        # LSTM hidden state init-zero per sample (matches BC training).
        # lstm_hidden_state_shape is (n_layers, 1, lstm_hidden_size).
        shp = self.policy.lstm_hidden_state_shape
        n_layers, _, lstm_hidden = shp[0], shp[1], shp[2]
        init = (
            th.zeros(n_layers, B, lstm_hidden, device=self.device),
            th.zeros(n_layers, B, lstm_hidden, device=self.device),
        )
        episode_starts = th.ones(B, device=self.device)
        latent, _ = self.policy._process_sequence(
            features, init, episode_starts, self.policy.lstm_actor
        )
        latent_pi = self.policy.mlp_extractor.forward_actor(latent)
        mean_actions = self.policy.action_net(latent_pi)

        # Recurrent policy outputs unbounded Gaussian-mean; BC targets are
        # in [-1, 1] (clipped during collection) and the env wrapper clips
        # at step time. MSE in raw mean space is the right loss.
        bc_loss = self.bc_aux_coef * F.mse_loss(mean_actions, act_mb)

        self.policy.optimizer.zero_grad()
        bc_loss.backward()
        # SB3 clips grad norm at self.max_grad_norm during PPO; do the same here.
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()
        self.policy.set_training_mode(False)

        return {"loss": float(bc_loss.detach().cpu()) / max(self.bc_aux_coef, 1e-8)}
