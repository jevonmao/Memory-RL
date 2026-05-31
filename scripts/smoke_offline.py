"""Offline smoke test: validates the new components without spinning up SAPIEN
(Vulkan is unavailable on the local WSL host). Runs in any env that has
torch + sb3 + sb3-contrib.

Covers:
  * train.rewards.pickxtimes — reward computation against a mocked unwrapped
  * train.models.rnd — forward / loss / RMS updates / intrinsic
  * train.algos.ppo_with_rnd — collect_rollouts on a fake dict-obs env
  * train.algos.recurrent_ppo_with_rnd — same, against RecurrentPPO

For the actual SAPIEN-backed env smoke, run on the cluster:
    python -m scripts.smoke_pickxtimes --device cuda
"""

from __future__ import annotations

import os, sys, traceback
import numpy as np

os.environ["WANDB_MODE"] = "disabled"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv


# ----------------------------------------------------------------------
# A toy dict-obs gym env with the same shape as RobommeRLEnv but no SAPIEN.
# ----------------------------------------------------------------------
import gymnasium as gym


class FakeRobommeEnv(gym.Env):
    """Mimics RobommeRLEnv's observation/action space for offline tests."""
    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0, max_steps: int = 32):
        super().__init__()
        self._t = 0
        self._max = max_steps
        self._rng = np.random.default_rng(seed)
        self.observation_space = spaces.Dict({
            "front_rgb":   spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
            "wrist_rgb":   spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
            "joint_state": spaces.Box(-np.inf, np.inf, (7,),  dtype=np.float32),
            "eef_state":   spaces.Box(-np.inf, np.inf, (6,),  dtype=np.float32),
            "gripper":     spaces.Box(-1., 1., (2,),          dtype=np.float32),
        })
        self.action_space = spaces.Box(-1., 1., (8,), dtype=np.float32)

    def _obs(self):
        return {
            "front_rgb":   self._rng.integers(0, 255, (128, 128, 3), dtype=np.uint8),
            "wrist_rgb":   self._rng.integers(0, 255, (128, 128, 3), dtype=np.uint8),
            "joint_state": self._rng.standard_normal(7).astype(np.float32),
            "eef_state":   self._rng.standard_normal(6).astype(np.float32),
            "gripper":     self._rng.standard_normal(2).astype(np.float32),
        }

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        rew = float(self._rng.standard_normal())
        term = False
        trunc = self._t >= self._max
        return self._obs(), rew, term, trunc, {}

    def close(self):
        pass


# ----------------------------------------------------------------------
# Mocked PickXtimes "unwrapped" state for reward unit test.
# ----------------------------------------------------------------------
class _Pose:
    def __init__(self, xyz):
        self.p = np.asarray(xyz, dtype=np.float32).reshape(1, 3)


class _Body:
    def __init__(self, xyz):
        self.pose = _Pose(xyz)


class _Agent:
    def __init__(self, tcp_xyz, qpos):
        self.tcp_pose = _Pose(tcp_xyz)
        class R:
            def __init__(self, q): self._q = q
            def get_qpos(self):    return np.asarray(self._q, dtype=np.float32).reshape(1, -1)
        self.robot = R(qpos)


class MockUnwrapped:
    """Just enough surface for PickXtimesReward to call into."""
    def __init__(self, num_repeats: int = 2,
                 tcp_xyz=(0.0, 0.0, 0.10),
                 cube_xyz=(0.10, 0.10, 0.02),
                 target_xyz=(0.20, 0.0, 0.02),
                 button_xyz=(-0.20, 0.0, 0.02),
                 gripper_qpos=(0.04, 0.04)):
        self.num_repeats = num_repeats
        self.timestep    = 0
        self.target_cube = _Body(cube_xyz)
        self.target      = _Body(target_xyz)
        self.button      = _Body(button_xyz)
        self.agent       = _Agent(tcp_xyz, qpos=[0.0]*7 + list(gripper_qpos))


# ----------------------------------------------------------------------
def t_reward():
    print("[reward] PickXtimesReward ...")
    from train.rewards.pickxtimes import PickXtimesReward
    rfn = PickXtimesReward()
    u = MockUnwrapped(num_repeats=2)
    rfn.reset(u)

    rewards = []
    # 5 random steps, then advance timestep by 1 (simulating subgoal completion)
    for k in range(5):
        rewards.append(rfn.step(u, info={}))
    u.timestep = 1                    # pickup subgoal completed
    rewards.append(rfn.step(u, info={}))
    u.timestep = 2                    # drop subgoal completed → entering pickup #2
    rewards.append(rfn.step(u, info={}))
    rewards.append(rfn.step(u, info={"success": True}))
    print(f"  rewards seq: {[round(r,3) for r in rewards]}")
    assert all(np.isfinite(rewards))
    # Subgoal-completion bonus shows up between steps 4 and 5 (+~2 jump).
    # Hard to assert exact numbers (other shaping terms move), but verify
    # the success step contains the +10 bonus contribution.
    assert rewards[-1] > 5.0, f"terminal success should swing reward positive: got {rewards[-1]}"
    print("  PASS")


def t_rnd():
    print("[rnd] forward / loss / intrinsic / RMS ...")
    from train.models.rnd import RND
    rnd = RND(feat_dim=576, out_dim=64, hidden=128)
    feats = th.randn(32, 576)
    rnd.update_obs_rms(feats)
    intr = rnd.intrinsic(feats)
    assert intr.shape == (32,), intr.shape
    assert th.isfinite(intr).all()

    # One predictor optimisation step reduces loss
    opt = th.optim.Adam(rnd.parameters(), lr=1e-3)
    pre = rnd.loss(feats).item()
    for _ in range(5):
        opt.zero_grad()
        l = rnd.loss(feats)
        l.backward()
        opt.step()
    post = rnd.loss(feats).item()
    print(f"  loss before={pre:.4f}  after 5 steps={post:.4f}")
    assert post < pre, "RND predictor failed to fit a fixed batch"
    print("  PASS")


def t_ppo_with_rnd():
    print("[algo] PPOWithRND.collect_rollouts ...")
    from train.algos.ppo_with_rnd import PPOWithRND
    from train.models.encoder import RobommeCNNExtractor
    from train.models.rnd import RND

    vec = DummyVecEnv([lambda: FakeRobommeEnv(seed=0, max_steps=32) for _ in range(2)])
    rnd = RND(feat_dim=576, out_dim=64, hidden=128)
    model = PPOWithRND(
        policy="MultiInputPolicy",
        env=vec,
        learning_rate=3e-4,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        gamma=0.99,
        ent_coef=0.0,
        use_sde=False,
        policy_kwargs=dict(
            features_extractor_class=RobommeCNNExtractor,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            squash_output=False,
        ),
        device="cpu",
        verbose=0,
        seed=0,
        rnd=rnd,
        eta=1.0,
        rnd_lr=1e-4,
    )
    model.learn(total_timesteps=64, progress_bar=False)
    vec.close()
    print("  PASS")


def t_recurrent_ppo_with_rnd():
    print("[algo] RecurrentPPOWithRND.collect_rollouts ...")
    from train.algos.recurrent_ppo_with_rnd import RecurrentPPOWithRND
    from train.models.encoder import RobommeCNNExtractor
    from train.models.rnd import RND

    vec = DummyVecEnv([lambda: FakeRobommeEnv(seed=0, max_steps=16) for _ in range(2)])
    rnd = RND(feat_dim=576, out_dim=64, hidden=128)
    model = RecurrentPPOWithRND(
        policy="MultiInputLstmPolicy",
        env=vec,
        learning_rate=3e-4,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        gamma=0.99,
        ent_coef=0.0,
        policy_kwargs=dict(
            features_extractor_class=RobommeCNNExtractor,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            lstm_hidden_size=32,
            n_lstm_layers=1,
            shared_lstm=True,
            enable_critic_lstm=False,
        ),
        device="cpu",
        verbose=0,
        seed=0,
        rnd=rnd,
        eta=1.0,
        rnd_lr=1e-4,
    )
    model.learn(total_timesteps=64, progress_bar=False)
    vec.close()
    print("  PASS")


def t_save_load_rnd():
    print("[ckpt] PPOWithRND save/load ...")
    import tempfile
    from train.algos.ppo_with_rnd import PPOWithRND
    from train.models.encoder import RobommeCNNExtractor
    from train.models.rnd import RND
    vec = DummyVecEnv([lambda: FakeRobommeEnv(seed=0, max_steps=16) for _ in range(1)])
    rnd = RND(feat_dim=576, out_dim=64, hidden=128)
    model = PPOWithRND(
        policy="MultiInputPolicy",
        env=vec,
        n_steps=16, batch_size=16, n_epochs=1, gamma=0.99, ent_coef=0.0,
        policy_kwargs=dict(
            features_extractor_class=RobommeCNNExtractor,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            squash_output=False,
        ),
        device="cpu", verbose=0, seed=0,
        rnd=rnd, eta=1.0,
    )
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "smoke")
        model.save(p)
        th.save(model.rnd.state_dict(), p + "_rnd.pt")

        # Reload PPO weights into a fresh model + restore RND weights manually.
        rnd2 = RND(feat_dim=576, out_dim=64, hidden=128)
        model2 = PPOWithRND(
            policy="MultiInputPolicy",
            env=vec,
            n_steps=16, batch_size=16, n_epochs=1, gamma=0.99, ent_coef=0.0,
            policy_kwargs=dict(
                features_extractor_class=RobommeCNNExtractor,
                net_arch=dict(pi=[64, 64], vf=[64, 64]),
                squash_output=False,
            ),
            device="cpu", verbose=0, seed=0,
            rnd=rnd2, eta=1.0,
        )
        model2.set_parameters(p + ".zip", exact_match=True, device="cpu")
        model2.rnd.load_state_dict(th.load(p + "_rnd.pt", map_location="cpu"))
    vec.close()
    print("  PASS")


# ----------------------------------------------------------------------
def main():
    tests = [t_reward, t_rnd, t_ppo_with_rnd, t_recurrent_ppo_with_rnd, t_save_load_rnd]
    failed = []
    for t in tests:
        try:
            t()
        except Exception:
            traceback.print_exc()
            failed.append(t.__name__)
    print("=" * 60)
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    print("ALL OFFLINE SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()
