"""
Verify SubprocVecEnv with start_method=spawn works on Windows. Builds a
small (n_envs=2 by default) vec_env, takes a handful of steps, reports
per-vec-step latency. Run alongside an active training only with --n_envs 2
or fewer — SAPIEN contexts are heavy.
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def make_env(rank):
    from train.envs.rl_env import RobommeRLEnv
    from stable_baselines3.common.monitor import Monitor
    def _init():
        return Monitor(RobommeRLEnv(env_id="BinFill", seed=42 + rank, shape_reward=False))
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_envs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--backend", choices=["subproc", "dummy"], default="subproc")
    args = ap.parse_args()

    print(f"backend={args.backend} n_envs={args.n_envs} steps={args.steps}", flush=True)

    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    if args.backend == "subproc":
        vec = SubprocVecEnv([make_env(i) for i in range(args.n_envs)], start_method="spawn")
    else:
        vec = DummyVecEnv([make_env(i) for i in range(args.n_envs)])

    print("reset...", flush=True)
    t0 = time.perf_counter()
    obs = vec.reset()
    print(f"  reset: {time.perf_counter()-t0:.2f}s", flush=True)

    import numpy as np
    a = np.zeros((args.n_envs, vec.action_space.shape[0]), dtype=np.float32)

    # warmup
    for _ in range(3):
        vec.step(a)

    samples = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        vec.step(a)
        samples.append(1000.0 * (time.perf_counter() - t0))
    mean = sum(samples) / len(samples)
    p95 = sorted(samples)[int(0.95 * (len(samples) - 1))]
    print(f"\nvec_step_mean={mean:.1f}ms  p95={p95:.1f}ms  "
          f"sample_fps={args.n_envs * 1000.0 / mean:.1f}  "
          f"vec_fps={1000.0/mean:.1f}", flush=True)
    vec.close()


if __name__ == "__main__":
    # spawn is the Windows default; set explicitly so behavior matches when
    # launched via cmd.exe or from a non-main module.
    try:
        multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    main()
