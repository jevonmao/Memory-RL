"""One DAgger iteration on PickXtimes.

Procedure:
  1. Load current policy (BC or DAgger-improved).
  2. Roll out policy in real env for N episodes.
  3. Every K env-steps during rollout, query the planner-expert for the
     first action it would take from the *currently-visited state*. Save
     (wrapped_obs, expert_action) as an additional training transition.
  4. Save the new transitions as .npz files alongside the original BC episodes.
  5. (Caller then runs scripts.bc_pretrain_pickxtimes on the aggregated set.)

The expert-query trick: each task subgoal carries a `solve(env, planner)`
callback. We monkey-patch `env.step` to capture the FIRST action the solver
emits and raise an exception to abort the rest of the solver's planning.
The planner is fully re-entrant — its internal mplib call is pure, so
re-querying from arbitrary visited states is safe.

Usage:
    python -m scripts.dagger_iter \\
        --policy runs/bc/pickxtimes/bc_policy_recurrent.pt \\
        --recurrent --n_episodes 20 --k_query 10 \\
        --out runs/dagger/iter1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

os.environ.setdefault("ROBOMME_OBS_MODE", "rgb")
os.environ.setdefault("ROBOMME_SIM_BACKEND", "physx_cpu")

import numpy as np
import torch
import cv2

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from robomme.env_record_wrapper import BenchmarkEnvBuilder
from train.envs.rl_env import IMG_H, IMG_W, ACTION_DIM
from train.models.encoder import RobommeCNNExtractor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy",      required=True,
                   help="Path to BC/DAgger policy state_dict (CPU-saved).")
    p.add_argument("--recurrent",   action="store_true")
    p.add_argument("--lstm_hidden", type=int, default=256)
    p.add_argument("--lstm_layers", type=int, default=1)
    p.add_argument("--n_episodes",  type=int, default=20)
    p.add_argument("--k_query",     type=int, default=10,
                   help="Query expert every K env steps (lower = more data, higher cost).")
    p.add_argument("--max_steps",   type=int, default=1500)
    p.add_argument("--out",         required=True,
                   help="Output dir for new ep_*.npz files.")
    p.add_argument("--start_idx",   type=int, default=0,
                   help="Starting episode-index for output filenames (avoid collisions across iters).")
    p.add_argument("--seed",        type=int, default=0)
    return p.parse_args()


# ----------------------------------------------------------------------
def _extract_obs(raw_obs: dict) -> dict[str, np.ndarray]:
    def np_(x): return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    front = cv2.resize(np_(raw_obs["front_rgb_list"][-1]).astype(np.uint8), (IMG_W, IMG_H))
    wrist = cv2.resize(np_(raw_obs["wrist_rgb_list"][-1]).astype(np.uint8), (IMG_W, IMG_H))
    j = np_(raw_obs["joint_state_list"][-1]).astype(np.float32).flatten()
    e = np_(raw_obs["eef_state_list"][-1]).astype(np.float32).flatten()
    g = np_(raw_obs["gripper_state_list"][-1]).astype(np.float32).flatten()
    return {"front_rgb": front, "wrist_rgb": wrist,
            "joint_state": j[:7], "eef_state": e[:6], "gripper": g[:2]}


# ----------------------------------------------------------------------
class _StopAfterFirstStep(Exception):
    """Sentinel raised inside hooked env.step to abort planner-solve early."""


def get_first_expert_action(env, planner) -> np.ndarray | None:
    """Run the env's current-subgoal `solve` callback in dry-run mode.

    Bug we hit in iter 1: `solve_pickup` starts with `planner.open_gripper(t=6)`
    which emits 6 "stay-put + open-gripper" actions BEFORE any motion. Our
    naive hook captured the FIRST of these — useless as an expert correction.

    Fix: patch `planner.open_gripper` and `close_gripper` to no-ops so the
    next call (typically `planner.move_to_pose_with_screw`) emits the FIRST
    real movement action. Also patch env.step so each "skipped" call returns
    a fake tuple WITHOUT mutating env state — the planner doesn't re-read
    state inside open_gripper's loop, so this is safe.
    """
    try:
        ts = int(getattr(env.unwrapped, "timestep", 0))
        task_list = getattr(env.unwrapped, "task_list", None)
        if task_list is None or ts >= len(task_list):
            return None
        task = task_list[ts]
        solve = task.get("solve")
        if solve is None:
            return None
    except Exception:
        return None

    captured: list[np.ndarray] = []
    orig_step = env.step
    orig_open_gripper  = getattr(planner, "open_gripper",  None)
    orig_close_gripper = getattr(planner, "close_gripper", None)

    # Stash the most recent obs so we can return a plausible fake response.
    # We only need *shape* — planner doesn't read these fields during loops.
    try:
        fake_obs_template = None  # if first hooked call needs an obs, return {}
    except Exception:
        fake_obs_template = None

    def hooked_step(action, *a, **kw):
        arr = np.asarray(action, dtype=np.float32).reshape(-1)[:ACTION_DIM]
        if arr.shape[0] < ACTION_DIM:
            arr = np.concatenate([arr, np.zeros(ACTION_DIM - arr.shape[0], dtype=np.float32)])
        captured.append(arr)
        raise _StopAfterFirstStep()

    # No-op gripper toggles: just update the planner's internal gripper_state
    # flag (some downstream logic reads it) but don't step env.
    def noop_open(t=6, gripper_state=None):
        planner.gripper_state = getattr(planner, "OPEN", 1)

    def noop_close(t=6, gripper_state=None):
        planner.gripper_state = getattr(planner, "CLOSED", -1)

    env.step = hooked_step
    if orig_open_gripper is not None:
        planner.open_gripper = noop_open
    if orig_close_gripper is not None:
        planner.close_gripper = noop_close

    try:
        solve(env, planner)
    except _StopAfterFirstStep:
        pass
    except Exception:
        # mplib path failure, fingers unreachable, etc. — just skip this query.
        pass
    finally:
        env.step = orig_step
        if orig_open_gripper is not None:
            planner.open_gripper = orig_open_gripper
        if orig_close_gripper is not None:
            planner.close_gripper = orig_close_gripper

    if not captured:
        return None
    return np.clip(captured[0], -1.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------
def make_spaces():
    obs_space = spaces.Dict({
        "front_rgb":   spaces.Box(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8),
        "wrist_rgb":   spaces.Box(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8),
        "joint_state": spaces.Box(-np.inf, np.inf, (7,),  dtype=np.float32),
        "eef_state":   spaces.Box(-np.inf, np.inf, (6,),  dtype=np.float32),
        "gripper":     spaces.Box(-1., 1.,           (2,),  dtype=np.float32),
    })
    act_space = spaces.Box(-1., 1., (ACTION_DIM,), dtype=np.float32)
    return obs_space, act_space


def load_policy(path: str, recurrent: bool, lstm_hidden: int, lstm_layers: int):
    """Construct a policy and load BC/DAgger weights."""
    obs_space, act_space = make_spaces()

    class FE(gym.Env):
        def __init__(self): self.observation_space = obs_space; self.action_space = act_space
        def reset(self, *, seed=None, options=None): return obs_space.sample(), {}
        def step(self, a): return obs_space.sample(), 0.0, True, False, {}

    vec = DummyVecEnv([lambda: FE()])

    if recurrent:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO(
            "MultiInputLstmPolicy", vec,
            policy_kwargs=dict(
                features_extractor_class=RobommeCNNExtractor,
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                lstm_hidden_size=lstm_hidden, n_lstm_layers=lstm_layers,
                shared_lstm=True, enable_critic_lstm=False,
            ),
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=0, seed=0,
            n_steps=16, batch_size=16, n_epochs=1,
        )
    else:
        from stable_baselines3 import PPO
        model = PPO(
            "MultiInputPolicy", vec,
            policy_kwargs=dict(
                features_extractor_class=RobommeCNNExtractor,
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                squash_output=True,
            ),
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=0, seed=0,
            use_sde=True, sde_sample_freq=4,
            n_steps=16, batch_size=16, n_epochs=1,
        )

    sd = torch.load(path, map_location="cpu", weights_only=True)
    miss, unexp = model.policy.load_state_dict(sd, strict=False)
    print(f"  loaded policy from {path}  (missing={len(miss)}, unexpected={len(unexp)})", flush=True)
    model.policy.set_training_mode(False)
    return model


# ----------------------------------------------------------------------
def collect_one_episode(model, env_builder, ep_idx, recurrent, k_query, max_steps):
    """Roll out policy, query expert every k_query steps, return transitions."""
    from mani_skill.examples.motionplanning.panda.motionplanner import (
        PandaArmMotionPlanningSolver,
    )

    env = env_builder.make_env_for_episode(ep_idx)
    raw_obs, _ = env.reset()
    unwrapped = env.unwrapped

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False, vis=False,
        base_pose=unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )

    transitions: list[tuple[dict, np.ndarray]] = []
    lstm = None
    ep_start = True
    step = 0
    n_queried = 0
    n_query_success = 0

    while step < max_steps:
        ob = _extract_obs(raw_obs)
        ob_b = {k: np.expand_dims(v, 0) for k, v in ob.items()}

        # Expert query at this state.
        if step % k_query == 0:
            n_queried += 1
            exp = get_first_expert_action(env, planner)
            if exp is not None:
                transitions.append((ob, exp))
                n_query_success += 1

        # Policy action (stochastic for diversity; could also use deterministic).
        if recurrent:
            a, lstm = model.predict(ob_b, state=lstm,
                                     episode_start=np.array([ep_start], dtype=bool),
                                     deterministic=False)
            ep_start = False
        else:
            a, _ = model.predict(ob_b, deterministic=False)
        a = np.clip(a[0], -1.0, 1.0).astype(np.float32)

        raw_obs, _, term, trunc, info = env.step(a)
        step += 1
        if bool(term) or bool(trunc):
            break

    success = bool(info.get("success", False))
    max_ts = int(getattr(env.unwrapped, "timestep", 0))
    env.close()
    return transitions, success, max_ts, step, n_queried, n_query_success


# ----------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"[dagger] loading policy from {args.policy}", flush=True)
    model = load_policy(args.policy, args.recurrent, args.lstm_hidden, args.lstm_layers)

    print(f"[dagger] building env (PickXtimes train split)", flush=True)
    env_builder = BenchmarkEnvBuilder(
        "PickXtimes", dataset="train",
        action_space="joint_angle", max_steps=args.max_steps,
    )
    n_ep_avail = env_builder.get_episode_num()
    n_run = min(args.n_episodes, n_ep_avail)
    rng = np.random.default_rng(args.seed)
    ep_order = rng.permutation(n_ep_avail)[:n_run]

    print(f"[dagger] collecting {n_run} episodes with k_query={args.k_query}", flush=True)
    t0 = time.time()
    total_transitions = 0
    total_queries = 0
    total_success = 0
    total_max_ts = 0

    for k, ep in enumerate(ep_order):
        print(f"\n[ep {k+1}/{n_run}] idx={int(ep)} elapsed={time.time()-t0:.1f}s", flush=True)
        try:
            transitions, success, max_ts, n_steps, n_queried, n_query_success = \
                collect_one_episode(model, env_builder, int(ep),
                                    args.recurrent, args.k_query, args.max_steps)
        except Exception:
            traceback.print_exc()
            continue

        if not transitions:
            print(f"  no expert-query transitions; success={success}, max_subgoal={max_ts}, steps={n_steps}")
            continue

        # Save this episode's expert-corrected transitions as .npz
        obs_keys = transitions[0][0].keys()
        ep_obs = {key: np.stack([t[0][key] for t in transitions], axis=0) for key in obs_keys}
        ep_act = np.stack([t[1] for t in transitions], axis=0).astype(np.float32)
        out_path = os.path.join(args.out, f"ep_{args.start_idx + k:04d}.npz")
        np.savez_compressed(out_path, actions=ep_act, **ep_obs)

        total_transitions += len(transitions)
        total_queries     += n_queried
        total_success     += int(success)
        total_max_ts      += max_ts
        print(f"  saved {len(transitions)} expert transitions ({n_query_success}/{n_queried} queries succeeded); "
              f"policy max_subgoal={max_ts}, success={success}", flush=True)

    print()
    print(f"=== DAgger iter summary ===", flush=True)
    print(f"  episodes:           {n_run}")
    print(f"  policy successes:   {total_success}/{n_run}  ({total_success/max(n_run,1)*100:.0f}%)")
    print(f"  avg max_subgoal:    {total_max_ts/max(n_run,1):.2f}")
    print(f"  expert queries:     {total_queries}")
    print(f"  new transitions:    {total_transitions}")
    print(f"  elapsed:            {time.time()-t0:.1f}s")
    print(f"  saved to:           {args.out}")


if __name__ == "__main__":
    main()
