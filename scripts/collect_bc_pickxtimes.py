"""Collect planner-based expert trajectories for PickXtimes for BC pretraining.

For each episode in the train split:
  1. Build env via BenchmarkEnvBuilder (matches RobommeRLEnv setup).
  2. Initialise a PandaArmMotionPlanningSolver against the inner env.
  3. Walk the env's task_list, calling each task's `solve(env, planner)`
     in sequence. The planner internally calls env.step(action) many times.
  4. Hook env.step to record (raw_obs, action) at every step.
  5. Convert raw_obs to our wrapper's flat dict via the same _extract_obs
     used by RobommeRLEnv.
  6. Clip actions to [-1, 1] (our policy's action space; Panda controllers
     use normalize_action=False so [-1, 1] is a sensible subset of full
     joint range, and is exactly the space the policy will emit).

Saves to: <out_dir>/bc_dataset_pickxtimes.pt
  {"obs": dict[str, np.ndarray],  # batched along axis 0
   "actions": np.ndarray (N, 8),
   "ep_lengths": np.ndarray (E,),
   "success_mask": np.ndarray (E,) bool}

Only successful episodes go into the dataset by default (--include_failures
to keep failures too).

Usage:
    python -m scripts.collect_bc_pickxtimes --n_episodes 50 \\
        --out runs/bc/pickxtimes
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Force CPU sim: matches our training stack on cluster.
os.environ.setdefault("ROBOMME_SIM_BACKEND", "physx_cpu")
os.environ.setdefault("ROBOMME_OBS_MODE", "rgb")

import numpy as np
import torch

from robomme.env_record_wrapper import BenchmarkEnvBuilder

print("[bc] importing train.envs.rl_env ...", flush=True)
from train.envs.rl_env import _extract_obs, ACTION_DIM
print("[bc] imports done", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--out", default="runs/bc/pickxtimes")
    p.add_argument("--task", default="PickXtimes")
    p.add_argument("--max_steps", type=int, default=1500)
    p.add_argument("--include_failures", action="store_true",
                   help="Keep episodes even if the planner failed.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def collect_one_episode(env, ep_idx: int, max_steps: int):
    """Run planner against env's task_list, recording (obs, action) pairs.

    Returns: list[(obs_dict_wrapped, action_clipped_8d)], bool success.
    """
    from mani_skill.examples.motionplanning.panda.motionplanner import (
        PandaArmMotionPlanningSolver,
    )

    raw_obs, _ = env.reset()
    unwrapped = env.unwrapped

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )

    # Hook env.step to record (obs_before, action). The planner stepping path
    # is `planner.follow_path → self.env.step(action)` — we wrap THIS env.step.
    transitions: list[tuple[dict, np.ndarray]] = []

    last_raw_obs = {"_obs": raw_obs}
    orig_step = env.step

    def hooked_step(action, *a, **kw):
        # Record (obs_before, action) BEFORE stepping.
        try:
            wrapped_obs = _extract_obs(last_raw_obs["_obs"])
            arr = np.asarray(action, dtype=np.float32).reshape(-1)[:ACTION_DIM]
            if arr.shape[0] < ACTION_DIM:
                arr = np.concatenate([arr, np.zeros(ACTION_DIM - arr.shape[0], dtype=np.float32)])
            arr = np.clip(arr, -1.0, 1.0)
            transitions.append((wrapped_obs, arr))
        except Exception as exc:
            print(f"  warning: failed to record transition: {exc}")

        result = orig_step(action, *a, **kw)
        # update last_raw_obs to the post-step obs
        new_raw_obs = result[0]
        last_raw_obs["_obs"] = new_raw_obs
        return result

    env.step = hooked_step

    task_list = getattr(unwrapped, "task_list", None)
    if task_list is None:
        env.step = orig_step
        return [], False

    # Walk every task entry. Each entry's "solve" is a callable (env, planner) -> ...
    success = True
    step_budget = max_steps
    for i, entry in enumerate(task_list):
        solve = entry.get("solve")
        if solve is None:
            continue
        try:
            solve(env, planner)
        except Exception as exc:
            print(f"  ep {ep_idx}, task {i}/{len(task_list)} failed: {exc}")
            success = False
            break
        if len(transitions) >= step_budget:
            print(f"  ep {ep_idx}: exceeded step budget {step_budget}; truncating")
            success = False
            break

    env.step = orig_step

    # Check the env's success flag after the planner completes.
    try:
        eval_info = unwrapped.evaluate(solve_complete_eval=True)
        env_success = bool(eval_info.get("success", False)) and not bool(eval_info.get("fail", False))
        # Treat env's verdict as authoritative.
        success = success and env_success
    except Exception as exc:
        print(f"  ep {ep_idx}: post-solve evaluate failed: {exc}")

    return transitions, success


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"[main] constructing BenchmarkEnvBuilder for {args.task} ...", flush=True)
    builder = BenchmarkEnvBuilder(
        env_id=args.task,
        dataset="train",
        action_space="joint_angle",
        max_steps=args.max_steps,
    )
    print(f"[main] BenchmarkEnvBuilder constructed", flush=True)
    n_ep_avail = builder.get_episode_num()
    n_run = min(args.n_episodes, n_ep_avail) if n_ep_avail > 0 else args.n_episodes
    print(f"[collect] task={args.task}  episodes_available={n_ep_avail}  collecting={n_run}")

    rng = np.random.default_rng(args.seed)
    ep_order = rng.permutation(n_ep_avail if n_ep_avail > 0 else n_run)[:n_run]

    # Stream-to-disk: save each successful episode as its own .npz file under
    # <out>/episodes/, then concat into one .pt at the end. This caps peak
    # memory to a single episode (~50 MB) instead of the full dataset (~4 GB).
    ep_dir = os.path.join(args.out, "episodes")
    os.makedirs(ep_dir, exist_ok=True)

    ep_lengths: list[int] = []
    successes:  list[bool] = []
    saved_ep_files: list[str] = []

    t0 = time.time()
    for k, ep in enumerate(ep_order):
        print(f"\n[ep {k+1}/{n_run}] idx={int(ep)} elapsed={time.time()-t0:.1f}s", flush=True)
        env = builder.make_env_for_episode(int(ep))
        try:
            transitions, success = collect_one_episode(env, int(ep), args.max_steps)
        except Exception:
            traceback.print_exc()
            transitions, success = [], False
        env.close()

        if not transitions:
            print(f"  no transitions — skipping")
            continue
        if not success and not args.include_failures:
            print(f"  failed planner — skipping (use --include_failures to keep)")
            continue

        obs_keys = transitions[0][0].keys()
        ep_obs = {key: np.stack([t[0][key] for t in transitions], axis=0) for key in obs_keys}
        ep_act = np.stack([t[1] for t in transitions], axis=0)

        ep_file = os.path.join(ep_dir, f"ep_{k:04d}.npz")
        np.savez_compressed(ep_file, actions=ep_act.astype(np.float32), **ep_obs)
        saved_ep_files.append(ep_file)
        ep_lengths.append(len(transitions))
        successes.append(success)

        # Free per-episode arrays immediately.
        del transitions, ep_obs, ep_act
        print(f"  ✓ recorded {ep_lengths[-1]} steps  (success={success})  -> {os.path.basename(ep_file)}", flush=True)

    if not saved_ep_files:
        print("\nNo data collected — exiting.")
        sys.exit(1)

    # Final pass: concat all episode files into one .pt. Read each .npz once.
    print(f"\n[main] concatenating {len(saved_ep_files)} episode files ...", flush=True)
    sample = np.load(saved_ep_files[0])
    obs_keys = [k for k in sample.files if k != "actions"]
    total_n  = sum(ep_lengths)
    print(f"[main] total transitions: {total_n}", flush=True)

    obs_concat: dict[str, np.ndarray] = {}
    for key in obs_keys:
        shape = (total_n,) + tuple(sample[key].shape[1:])
        obs_concat[key] = np.empty(shape, dtype=sample[key].dtype)
    act_concat = np.empty((total_n, ACTION_DIM), dtype=np.float32)

    off = 0
    for f in saved_ep_files:
        d = np.load(f)
        n = d["actions"].shape[0]
        act_concat[off:off + n] = d["actions"]
        for key in obs_keys:
            obs_concat[key][off:off + n] = d[key]
        off += n
        d.close()

    out_path = os.path.join(args.out, f"bc_dataset_{args.task.lower()}.pt")
    torch.save({
        "obs":         obs_concat,
        "actions":     act_concat,
        "ep_lengths":  np.asarray(ep_lengths, dtype=np.int32),
        "success_mask": np.asarray(successes, dtype=bool),
        "task":        args.task,
    }, out_path)
    print(f"\n=== {len(successes)} episodes ({sum(successes)} successful), "
          f"{act_concat.shape[0]} transitions ===")
    print(f"saved -> {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
