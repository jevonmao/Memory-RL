# PickXtimes RL — Final Findings (2026-05-24)

## Headline

**0% success rate across 6 reward variants × 4 baseline cells × 3 seeds.** The deterministic policy never completes even the first pickup of any test episode (max_subgoal = 0/N for 100% of 20-episode runs at every checkpoint we evaluated, from 100k → 1.34M steps).

Memory (LSTM) > vanilla PPO at training reward, but neither produces task progress. RND, BC warmstart, and reward-shape tuning all fail to lift the deterministic policy above the "hover near object" failure mode.

## Experimental matrix

| Variant | Memory (LSTM) | Curiosity (RND) | BC warm-start | Reward | Seeds | Final eval SR |
|---|---|---|---|---|---|---|
| b1 V1 (PPO, control) | ✗ | ✗ | ✗ | V1 | killed | 0/10 @ 900k |
| b4 V1 (PPO+RND) | ✗ | ✓ | ✗ | V1 | killed | inferred 0% |
| b5 V1 (RecurrentPPO) | ✓ | ✗ | ✗ | V1 | 1 seed → 1.34M | 0/20 @ 700k, 0/20 @ 900k, 0/20 subgoal |
| b6 V1 | ✓ | ✓ | ✗ | V1 | 3 seeds → 1.17M+ | 0/20 @ 900k |
| b6 V1 + BC | ✓ | ✓ | ✓ | V1 | 3 seeds → ~1M | 0/20 @ 400k, 0/20 @ 600k |
| b1–b6 V2 | (all 4 cells) | | ✗ | V2 | 3 seeds @ ~700k | 0/20 @ 600k V2-b5 |
| b6 V2 + BC | ✓ | ✓ | ✓ | V2 | 3 seeds | training |
| b6 V3 + BC | ✓ | ✓ | ✓ | V3 | 3 seeds @ ~560k | 0/20 @ 500k |
| b6 V3 + BC + low ent_coef | ✓ | ✓ | ✓ | V3 + ent=0.001 | 3 seeds @ ~640k | **all 3 seeds: 0/10 subgoals @ 600k** |

Total cluster compute: ~33 RL jobs × ~5h ≈ 165 Titan RTX-hours.

## Reward shapes tested

| | proximity coef | subgoal | pickup | terminal-succ |
|---|---|---|---|---|
| V1 | 0.05 | 2 | 1 | 10 |
| V2 | **0.005** | **10** | **5** | **50** |
| V3 | 0.05 | 10 | 5 | 50 |

## Findings (with evidence)

### F1 — V1 reward is exploitable; agent hovers without grasping

V1 training reward smoothly climbs 25→47 over 1.3M steps **without spikes**, characteristic of proximity-hacking rather than subgoal completion. Diagnostic eval at 900k: 0/20 episodes reach even subgoal 1 (first pickup). Reward growth is entirely from "hover-near-target" + telescoping potential-based shaping, NOT from any actual task progress.

### F2 — V2 (proximity-attenuated) doesn't enable exploration

V2 kept reward at floor ~−10 throughout ~700k steps for vanilla PPO/PPO+RND (no memory). Memory variants (b5, b6) crept up to ~−9 (suggesting ~10% of training episodes get a single pickup via stochastic sampling), but deterministic eval at 600k still gives 0/20 subgoals. The strong task bonus signal exists but the policy can't translate stochastic-noise pickups into a deterministic-mean strategy.

### F3 — V3 (hybrid) plateaus at proximity ceiling

V3 keeps V1's proximity gradient + V2's strong task bonuses. Training reward reaches ~41 (proximity ceiling) but no spikes appear. Eval at 500k: 0/20 subgoals.

### F4 — BC alone is insufficient AND BC warm-start doesn't unlock pickup

- BC policy fits the planner perfectly (loss = 0.006 on 42 821 expert transitions from 80 planner episodes, 100% success in collection).
- BC eval on test split: 0/20. On TRAIN split: 0/20.
- BC + RL (V1 / V2 / V3 / V3+lowent): all 0/20 subgoals at 300–600k.

Compounding-error catastrophic forgetting kills naive BC. Warm-start improves training-time reward by ~+2 (b6+BC ≈ 43 vs b6 ≈ 41) but does NOT lift deterministic-action subgoal completion above 0.

### F5 — Memory > Curiosity (in training reward), but neither produces task success

At ~600k steps (V1 reward):
- b5 (RecurrentPPO):  rew 46.6
- b6 (RecurrentPPO+RND): rew 40.4
- b1 (PPO):  rew 35.9
- b4 (PPO+RND): rew 31.4

LSTM clearly helps reward-shaping signal. RND on top of LSTM slightly hurts (probably adding intrinsic noise to a policy already converging on hover-strategy). Neither memory nor curiosity, with or without each other, produces non-zero eval SR.

### F6 — Lower entropy coefficient marginally tightens policy but still 0%

V3 + BC + ent_coef = 0.001 brings std from 1.02 (default) down to 0.83 over 600k steps. Reward profile and subgoal progress essentially identical to ent_coef = 0.01 → 0/10 subgoals.

## Diagnosis

The deterministic policy never completes the first pickup at ANY checkpoint we evaluated, across ALL reward variants, ALL baseline cells, and ALL seeds. Three contributing factors:

1. **Action-space mismatch under deterministic eval**: RecurrentPPO uses an unbounded Gaussian policy; our wrapper clips to [-1, 1]. With std≈1 throughout training, the policy mean (used at deterministic eval) sits at very different action regions than the sampled actions that occasionally got rewarded.
2. **Reward hacking**: V1's proximity reward gives generous credit for hovering — gradient pulls toward "be close" not "grasp + lift". V2 kills this signal but leaves no gradient for exploration. V3 hybrid still proximity-dominated until the policy stumbles onto a subgoal (which it never does).
3. **Sparse-reward, long-horizon, high-dim action**: 1500-step episodes, 8-D continuous actions, sparse subgoal signal — the random-exploration probability of a coordinated grasp sequence is astronomical. PPO's std=0.97 stays high throughout because ent_coef=0.01 and target_kl=0.05 prevent fast convergence; even ent_coef=0.001 only drops std to 0.83.

## What would likely unlock >0% SR (future work)

1. **DAgger or iterative BC** — interactively query the expert at policy-visited states to fix compounding errors.
2. **Action space redesign** — delta-action or per-joint scaled action; OR use squash_output=True for recurrent (sb3-contrib doesn't expose this; need a custom policy).
3. **5–10× more training** — MIKASA-Robo used 5–25M steps; we have 1.5M.
4. **Shorter max_steps** (600 instead of 1500) — forces task completion faster, more episodes per gradient step.
5. **Reward only at terminal events** — remove all proximity shaping; rely on +50 success or BC bootstrap to seed exploration.
6. **Curriculum** — force training on `seed % 3 == 0` (easy, N=1 cycle only) until first success, then add medium/hard.

## Code shipped this iteration

- `train/rewards/pickxtimes.py` (V1), `pickxtimes_v2.py` (V2), `pickxtimes_v3.py` (V3) — switched via `ROBOMME_REWARD_VERSION`.
- `train/models/rnd.py` — RND module with running-stats normalisation.
- `train/algos/ppo_with_rnd.py`, `recurrent_ppo_with_rnd.py` — PPO + RND, RecurrentPPO + RND.
- `train/train_ppo_rnd.py`, `train_ppo_recurrent.py`, `train_ppo_recurrent_rnd.py` — three new training drivers.
- `scripts/collect_bc_pickxtimes.py`, `bc_pretrain_pickxtimes.py`, `cluster/bc_pretrain_job.sbatch` — BC pipeline (data collection + training).
- `scripts/eval_subgoal.py`, `eval_stochastic.py`, `eval_bc_only.py`, `eval_bc_only_train.py` — diagnostic evals.
- `scripts/cluster/submit_{pickxtimes_sweep,pickxtimes_v2,bc_warmstart_sweep,b6_bc_v2,b6_bc_v3,b6_bc_v3_lowent}.sh` — sweep submission scripts.
- `train/evaluate_trained.py` — added `recurrent` baseline path + action-clip fix.

## Cluster engineering notes (replication)

- conda env at `/vision/u/jevon/miniconda3/envs/robomme/`: torch 2.9.1+cu121, torchvision 0.24.1, sb3-contrib 2.8.0, stable_baselines3 2.8.0, mani_skill 3.0.0b21 (YinpeiDai fork, git rev 07be6fbc), sapien 3.0.2.
- **MUST** `export ROBOMME_SIM_BACKEND=physx_cpu`: sapien 3.0.3 + torch 2.9 trip a `__cuda_array_interface__` typestr parse error in `articulation_joint.drive_target` (used during scene load for buttons). CPU sim is fast enough — 60–75 FPS aggregate at n_envs=4.
- **MUST** clip eval-time actions to [-1, 1]: `train/evaluate_trained.py` was passing unclipped policy means to `env.step`, causing recurrent-policy distribution mismatch vs training.
- `scripts/cluster/sync_to_cluster.sh` MUST exclude `logs/` — `--delete` was wiping cluster's slurm log dir and causing exit-120 startup failures.

---

_Generated 2026-05-24 by Claude Opus 4.7 (CS224R project, RoboMME PickXtimes RL._
