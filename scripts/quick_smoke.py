"""
30-step smoke + microbench. Runs a single RobommeRLEnv, times .step().

Designed to be safe to run alongside an active training job: completes in
~30 s, holds one extra SAPIEN context briefly. Reports per-step latency
so we can see whether the obs_mode=rgb + shared-backbone changes moved
the needle.

Usage (Windows):
    .venv\\Scripts\\python.exe -m scripts.quick_smoke --task BinFill
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from train.envs.rl_env import RobommeRLEnv
from train.models.encoder import RobommeCNNExtractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="BinFill")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    print(f"ROBOMME_OBS_MODE={os.environ.get('ROBOMME_OBS_MODE')}", flush=True)

    print("constructing env...", flush=True)
    t0 = time.perf_counter()
    env = RobommeRLEnv(env_id=args.task, seed=0, shape_reward=False)
    print(f"  ctor:  {time.perf_counter()-t0:.2f}s", flush=True)

    t0 = time.perf_counter()
    obs, _ = env.reset()
    print(f"  reset: {time.perf_counter()-t0:.2f}s   obs keys: {list(obs.keys())}", flush=True)
    print(f"  obs.front_rgb shape: {obs['front_rgb'].shape}", flush=True)

    # warmup
    for _ in range(args.warmup):
        env.step(env.action_space.sample())

    # timed steps
    samples_ms = []
    t_all = time.perf_counter()
    for _ in range(args.steps):
        a = env.action_space.sample()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        obs, r, term, trunc, info = env.step(a)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples_ms.append(1000.0 * (time.perf_counter() - t0))
        if term or trunc:
            env.reset()
    wall = time.perf_counter() - t_all

    mean = sum(samples_ms) / len(samples_ms)
    p50  = sorted(samples_ms)[len(samples_ms) // 2]
    p95  = sorted(samples_ms)[int(0.95 * (len(samples_ms) - 1))]
    fps  = 1000.0 / mean
    print(f"\nstep_mean={mean:.1f}ms  p50={p50:.1f}ms  p95={p95:.1f}ms  fps_single={fps:.1f}", flush=True)
    print(f"wall: {wall:.2f}s for {args.steps} steps  ({args.steps/wall:.1f} fps)", flush=True)

    # Extractor B=1 microbench (no env in path)
    import gymnasium.spaces as spaces
    obs_space = spaces.Dict({
        "front_rgb":   spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
        "wrist_rgb":   spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
        "joint_state": spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
        "eef_state":   spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
        "gripper":     spaces.Box(-1., 1., (2,), dtype=np.float32),
    })
    ext = RobommeCNNExtractor(obs_space).cuda().eval()
    fake = {
        "front_rgb":   torch.randint(0, 255, (1, 128, 128, 3), dtype=torch.uint8, device="cuda"),
        "wrist_rgb":   torch.randint(0, 255, (1, 128, 128, 3), dtype=torch.uint8, device="cuda"),
        "joint_state": torch.zeros((1, 7), device="cuda"),
        "eef_state":   torch.zeros((1, 6), device="cuda"),
        "gripper":     torch.zeros((1, 2), device="cuda"),
    }
    for _ in range(5):
        with torch.no_grad(): ext(fake)
    samples = []
    for _ in range(30):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(): ext(fake)
        torch.cuda.synchronize()
        samples.append(1000.0 * (time.perf_counter() - t0))
    print(f"\nextractor B=1: mean={sum(samples)/len(samples):.2f}ms", flush=True)

    env.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
