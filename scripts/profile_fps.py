"""
Microbenchmark: where does training time go?

Measures, in isolation:
  1. env.step()                    — SAPIEN physics + render
  2. env.reset()                   — gym.make rebuild + first render
  3. extractor.forward(obs)        — ResNet18 x2 + state MLP
  4. PPO minibatch update          — full forward + backward + optimizer.step

For (1) we sweep obs_mode in {"rgb", "rgb+depth", "rgb+depth+segmentation"} so
we can quantify what dropping depth/segmentation saves.

Usage:
    python scripts/profile_fps.py --task BinFill --steps 200 --warmup 20
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from contextlib import contextmanager

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import gymnasium as gym


@contextmanager
def cuda_timer(label, results):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    results.setdefault(label, []).append(dt)


def summarize(samples_ms, n_per_call=1):
    """Return (mean_ms, p50_ms, p95_ms, fps_per_call)."""
    if not samples_ms:
        return (0, 0, 0, 0)
    mean = statistics.mean(samples_ms)
    p50  = statistics.median(samples_ms)
    p95  = sorted(samples_ms)[int(0.95 * (len(samples_ms) - 1))]
    fps  = 1000.0 * n_per_call / mean if mean > 0 else 0.0
    return mean, p50, p95, fps


def print_table(title, rows):
    print(f"\n=== {title} ===")
    print(f"{'op':<40} {'mean_ms':>10} {'p50_ms':>10} {'p95_ms':>10} {'fps':>10}")
    for op, mean, p50, p95, fps in rows:
        print(f"{op:<40} {mean:>10.2f} {p50:>10.2f} {p95:>10.2f} {fps:>10.1f}")


# ---------------------------------------------------------------------------
# Env construction helpers
# ---------------------------------------------------------------------------

def _detect_render_backend():
    """Mirror episode_config_resolver._win_vulkan logic for WSL/Linux."""
    if torch.cuda.is_available():
        return "gpu"
    return "cpu"


def build_env_with_obs_mode(task: str, obs_mode: str, episode_idx: int = 0):
    """Bypass BenchmarkEnvBuilder so we can override obs_mode."""
    from robomme.env_record_wrapper.episode_config_resolver import BenchmarkEnvBuilder
    from robomme.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper

    builder = BenchmarkEnvBuilder(
        env_id=task, dataset="train", action_space="joint_angle", max_steps=500,
    )
    seed, difficulty_hint = builder.resolve_episode(episode_idx)

    sim_backend = "physx_cpu" if platform.system() == "Windows" else (
        "physx_cuda" if torch.cuda.is_available() else "physx_cpu"
    )
    render_backend = _detect_render_backend()

    env_kwargs = dict(
        obs_mode=obs_mode,
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="sparse",
        sim_backend=sim_backend,
        render_backend=render_backend,
    )
    if seed is not None:
        env_kwargs["seed"] = seed
    if difficulty_hint:
        env_kwargs["difficulty"] = difficulty_hint

    env = gym.make(task, **env_kwargs)
    env = DemonstrationWrapper(
        env,
        max_steps_without_demonstration=502,
        gui_render=False,
        include_maniskill_obs=False,
        include_front_depth=False,
        include_wrist_depth=False,
        include_front_camera_extrinsic=False,
        include_wrist_camera_extrinsic=False,
        include_available_multi_choices=False,
        include_front_camera_intrinsic=False,
        include_wrist_camera_intrinsic=False,
    )
    return env, builder, seed


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_env_step(task: str, obs_mode: str, steps: int, warmup: int):
    print(f"\n[bench_env_step] task={task} obs_mode={obs_mode}")
    env, _, seed = build_env_with_obs_mode(task, obs_mode)
    env.reset(seed=int(seed or 0))
    action_space = env.action_space

    # warmup
    for _ in range(warmup):
        a = action_space.sample()
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            env.reset()

    samples_ms = []
    t_total = time.perf_counter()
    for _ in range(steps):
        a = action_space.sample()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        obs, _, term, trunc, _ = env.step(a)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples_ms.append(1000.0 * (time.perf_counter() - t0))
        if term or trunc:
            env.reset()
    wall = time.perf_counter() - t_total
    env.close()

    mean, p50, p95, fps = summarize(samples_ms)
    print(f"  wall={wall:.2f}s  step_mean={mean:.1f}ms  fps_single={fps:.1f}")
    return ("env.step "+obs_mode, mean, p50, p95, fps)


def bench_env_reset(task: str, resets: int, warmup: int = 1):
    print(f"\n[bench_env_reset] task={task} resets={resets}")
    # Use the production wrapper because that's the actual reset path.
    from train.envs.rl_env import RobommeRLEnv
    env = RobommeRLEnv(env_id=task, seed=0, shape_reward=False)

    # warmup
    for _ in range(warmup):
        env.reset()

    samples_ms = []
    for _ in range(resets):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        env.reset()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples_ms.append(1000.0 * (time.perf_counter() - t0))
    env.close()

    mean, p50, p95, fps = summarize(samples_ms)
    print(f"  reset_mean={mean:.1f}ms  resets/s={fps:.2f}")
    return ("env.reset (production)", mean, p50, p95, fps)


def bench_extractor(batch_sizes, img_h=128, img_w=128, device="cuda"):
    print(f"\n[bench_extractor] sizes={batch_sizes}")
    from train.models.encoder import RobommeCNNExtractor
    import gymnasium.spaces as spaces

    obs_space = spaces.Dict({
        "front_rgb":   spaces.Box(0, 255, (img_h, img_w, 3), dtype=np.uint8),
        "wrist_rgb":   spaces.Box(0, 255, (img_h, img_w, 3), dtype=np.uint8),
        "joint_state": spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
        "eef_state":   spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
        "gripper":     spaces.Box(-1., 1., (2,), dtype=np.float32),
    })
    ext = RobommeCNNExtractor(obs_space).to(device).eval()

    rows = []
    for B in batch_sizes:
        obs = {
            "front_rgb":   torch.randint(0, 255, (B, img_h, img_w, 3), dtype=torch.uint8, device=device),
            "wrist_rgb":   torch.randint(0, 255, (B, img_h, img_w, 3), dtype=torch.uint8, device=device),
            "joint_state": torch.zeros((B, 7), device=device),
            "eef_state":   torch.zeros((B, 6), device=device),
            "gripper":     torch.zeros((B, 2), device=device),
        }
        # warmup
        for _ in range(5):
            with torch.no_grad():
                _ = ext(obs)

        samples_ms = []
        for _ in range(30):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = ext(obs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            samples_ms.append(1000.0 * (time.perf_counter() - t0))

        mean, p50, p95, fps = summarize(samples_ms)
        rows.append((f"extractor B={B}", mean, p50, p95, fps))
        print(f"  B={B:>4}  mean={mean:.2f}ms  imgs/s={B/mean*1000:.0f}")
    return rows


def bench_ppo_minibatch(batch_sizes, n_steps=2048, n_envs=1, img_h=128, img_w=128, device="cuda"):
    """Estimate PPO update cost: forward + backward + optimizer.step on a minibatch."""
    print(f"\n[bench_ppo_minibatch] sizes={batch_sizes}")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from train.models.encoder import RobommeCNNExtractor
    import gymnasium as gym
    import gymnasium.spaces as spaces

    class FakeEnv(gym.Env):
        observation_space = spaces.Dict({
            "front_rgb":   spaces.Box(0, 255, (img_h, img_w, 3), dtype=np.uint8),
            "wrist_rgb":   spaces.Box(0, 255, (img_h, img_w, 3), dtype=np.uint8),
            "joint_state": spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
            "eef_state":   spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
            "gripper":     spaces.Box(-1., 1., (2,), dtype=np.float32),
        })
        action_space = spaces.Box(-1., 1., (8,), dtype=np.float32)
        def reset(self, *, seed=None, options=None):
            return self._obs(), {}
        def step(self, a):
            return self._obs(), 0.0, False, False, {}
        def _obs(self):
            return {
                "front_rgb":   np.zeros((img_h, img_w, 3), dtype=np.uint8),
                "wrist_rgb":   np.zeros((img_h, img_w, 3), dtype=np.uint8),
                "joint_state": np.zeros((7,), dtype=np.float32),
                "eef_state":   np.zeros((6,), dtype=np.float32),
                "gripper":     np.zeros((2,), dtype=np.float32),
            }

    vec = DummyVecEnv([lambda: FakeEnv() for _ in range(n_envs)])
    policy_kwargs = dict(
        features_extractor_class=RobommeCNNExtractor,
        features_extractor_kwargs={},
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        squash_output=True,
    )
    rows = []
    for B in batch_sizes:
        model = PPO(
            policy="MultiInputPolicy", env=vec, n_steps=n_steps, batch_size=B,
            n_epochs=1, policy_kwargs=policy_kwargs, device=device,
            use_sde=True, sde_sample_freq=4, verbose=0,
        )
        # Fill the rollout buffer with garbage so train() will run.
        model.collect_rollouts(model.env, callback=None, rollout_buffer=model.rollout_buffer,
                               n_rollout_steps=n_steps) if False else None  # skip, fill manually

        rb = model.rollout_buffer
        rb.reset()
        for _ in range(n_steps):
            o = vec.reset()
            a = np.zeros((n_envs, 8), dtype=np.float32)
            rb.add(
                o, a,
                reward=np.zeros((n_envs,), dtype=np.float32),
                episode_start=np.zeros((n_envs,), dtype=np.float32),
                value=torch.zeros((n_envs,), device=device),
                log_prob=torch.zeros((n_envs,), device=device),
            )
        rb.compute_returns_and_advantage(
            last_values=torch.zeros((n_envs,), device=device),
            dones=np.zeros((n_envs,), dtype=np.float32),
        )

        # Warmup
        model._n_updates = 0
        model.train()

        # Time several train() calls (each does n_epochs * (n_steps*n_envs / B) minibatches)
        samples_ms = []
        for _ in range(3):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.train()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            samples_ms.append(1000.0 * (time.perf_counter() - t0))

        mean = statistics.mean(samples_ms)
        n_minibatches = (n_steps * n_envs) // B
        per_mb = mean / max(n_minibatches, 1)
        rows.append((f"ppo.train B={B} (n_mb={n_minibatches})", mean, mean, mean, 1000.0/per_mb))
        print(f"  B={B:>4}  full_train={mean:.0f}ms  per_mb={per_mb:.1f}ms")
    return rows


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="BinFill")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--resets", type=int, default=5)
    ap.add_argument("--skip-env", action="store_true", help="Skip SAPIEN env benches")
    ap.add_argument("--skip-ppo", action="store_true", help="Skip PPO update bench")
    ap.add_argument("--obs-modes", nargs="+",
                    default=["rgb+depth+segmentation", "rgb+depth", "rgb"])
    args = ap.parse_args()

    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device={torch.cuda.get_device_name(0)}")

    rows = []

    if not args.skip_env:
        for om in args.obs_modes:
            try:
                rows.append(bench_env_step(args.task, om, args.steps, args.warmup))
            except Exception as e:
                print(f"  FAIL obs_mode={om}: {e}")

        try:
            rows.append(bench_env_reset(args.task, args.resets))
        except Exception as e:
            print(f"  FAIL reset: {e}")

    rows += bench_extractor([1, 4, 16, 64, 256])

    if not args.skip_ppo:
        try:
            rows += bench_ppo_minibatch([64, 128, 256])
        except Exception as e:
            print(f"  FAIL ppo: {e}")

    print_table("SUMMARY", rows)


if __name__ == "__main__":
    main()
