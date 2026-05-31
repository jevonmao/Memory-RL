# Initial Eval Report — RoboMME PPO / PPO+ICM on BinFill

_Generated 2026-05-23. Single task (BinFill), single seed (seed=0). Pre-fix snapshot — read all caveats before citing._

## Test-split evaluation (`train/evaluate_trained.py`)

### Training-matched eval (legacy 500-step horizon)

| Run | Baseline | Task | Episodes | **Success rate** | Mean return | Loaded? |
|---|---|---|---|---|---|---|
| `ppo_icm_v7` | PPO + ICM | BinFill | 20 | **0.0 %** | 0.00 | ✓ |
| `ppo_v4`     | Vanilla PPO (v4 reward) | BinFill | — | **n/a** | — | ✗ — encoder schema mismatch (fixed retroactively — see below) |

### Official-comparable eval (1500-step horizon, default physics, default render)

After adding the legacy-encoder loader and bumping the eval horizon to match `challenge_interface/scripts/phase1_eval.py`:

| Run | Baseline | Task | Episodes | **Success rate** | Mean return | Notes |
|---|---|---|---|---|---|---|
| `ppo_v4`     | Vanilla PPO | BinFill | 20 | **0.0 %** | 0.00 | sim_freq=100, camera=256 (both match training) |
| `ppo_icm_v7` | PPO + ICM   | BinFill | 20 | **0.0 %** | 0.00 | sim_freq=100, camera=256 (both ≠ training: v7 trained at 40 / 128 — distribution shift) |

Raw JSON: [`ppo_v4_BinFill_official.json`](ppo_v4_BinFill_official.json), [`ppo_icm_v7_BinFill_official.json`](ppo_icm_v7_BinFill_official.json).

**Headline result: neither baseline completes a single BinFill test episode**, even with the official 3× longer time budget (1500 vs 500 steps). The +12% shaped-reward gap that ICM showed during training (10.3 → 11.5) does not translate into any task completion.

## Training progression (shaped reward — not the benchmark metric)

| Run | Baseline | Steps | Final `ep_rew_mean` | Final `ep_len_mean` | Wall-clock | Train FPS |
|---|---|---|---|---|---|---|
| `ppo_v2`     | Vanilla PPO (orig reward stub) | 500k | 0.0  | 501 | 4.8 h | 28 |
| `ppo_v3`     | Vanilla PPO (v3 shaping)       | 252k | −5.1 | 501 | 2.4 h | 29 |
| `ppo_v4`     | Vanilla PPO (v4 shaping)       | 252k | 10.3 | 501 | 2.4 h | 29 |
| `ppo_icm_v4` | PPO + ICM (v4 reward)          | 88k (killed) | 10.4 | 501 | 1.5 h | 16 |
| `ppo_icm_v7` | PPO + ICM (v4 reward + opts)   | 254k | 11.5 | 501 | 1.0 h | 75 |

`ep_len_mean = 501` across every run = **no episode ever terminated via success during training**. The policy is climbing the shaped reward (approach cubes, partial grasps) but never closes out a full BinFill within the training horizon.

## Reference numbers from the RoboMME paper (imitation learning, not RL)

| Method | BinFill success rate |
|---|---|
| Human | 96.0 % |
| GroundSG + Oracle (uses ground-truth info) | 85.8 % |
| FrameSamp + Modul (best non-oracle perceptual) | 39.6 % |

No RL baseline (PPO, SAC, etc.) is reported in the RoboMME paper — the benchmark ships with imitation-learning baselines only.

## Caveats that must accompany any number above

1. **Eval used `max_steps = 500`; official RoboMME Challenge horizon is `max_steps = 1500`.** Multi-stage tasks like BinFill (place several cubes + press button) may simply not be completable in 500 steps regardless of policy quality. _Fixed in code, not yet retrained._
2. **`ppo_icm_v7` was trained at `sim_freq = 40` (40 Hz physics, 2 substeps/env.step).** Official physics is `sim_freq = 100` (5 substeps). Trained-at-coarse, eval-at-coarse is internally consistent but introduces a physics distribution shift vs the official eval. _Default removed; future runs use 100._
3. **`ppo_icm_v7` was trained with SAPIEN cameras rendered natively at 128×128.** Official renders at 256 → downsamples in the wrapper. Minor distribution shift. _Default removed; future runs render native 256._
4. **Single seed (seed=0).** Official submission template (`doc/submission/model_example.md`) requires seeds 7, 42, 0 averaged.
5. **One task (BinFill) out of 16.** Per the paper, BinFill is one of the harder tasks because of cluttered scenes.
6. **Evaluation bypassed `challenge_interface/scripts/phase1_eval.py`** (the official harness). Used a custom in-process `train/evaluate_trained.py` loop. Numbers are not directly comparable to the leaderboard. _An `SB3Policy` wrapper for the official harness has now been added; future eval will run through it._
7. **Closest comparable RL prior work:** MIKASA-Robo (Memory, Benchmark & Robots, ICLR 2025) — different benchmark (32 ManiSkill memory tasks), but evaluated PPO-MLP, PPO-LSTM, SAC, TD-MPC2. Their finding: "none of the models — including those explicitly designed for sequence modeling — were able to successfully solve the majority of MIKASA-Robo tasks." PPO and PPO-LSTM both collapsed to ~0 % on harder memory tasks. **Suggests vanilla-PPO results at 250k on RoboMME BinFill are not anomalous — they are the expected order of magnitude.**

## Honest bottom-line for the writeup

At 250k training steps on BinFill, **both** the vanilla PPO baseline (`ppo_v4`) and the PPO+ICM baseline (`ppo_icm_v7`) fail to complete a single test-split episode on the official 1500-step horizon (0/20 each). The +12% shaped-reward gap that ICM showed during training (10.3 → 11.5) does not translate into any task completion. Training curves (`ep_len_mean = 501` throughout) corroborate that no episode was ever completed during rollouts either — the policy is climbing shaped-reward proxies (proximity, partial grasps) but cannot close out the multi-cube pick + bin + button sequence.

This matches the MIKASA-Robo result (PPO and PPO-LSTM collapse to ~0% on hard memory tasks) and the order of magnitude expected for 250k-step RL on contact-rich manipulation. The next planned run (`v8`) will train under the official eval-compatible configuration (sim_freq=100, native 256 cameras, max_steps=1500, evaluated via `challenge_interface/scripts/phase1_eval.py`) and target ≥1M steps so the resulting numbers have a chance of being non-zero.

## Methodology note — single-task vs multi-task

The official RoboMME submission template (`doc/submission/model_example.md`) prescribes:

> "We evaluate in a **multi-task setting, using a single model checkpoint for all tasks**, and require at least three runs with different random seeds to reduce performance variance."

**Our setup trains a separate policy per `(task, seed)` pair**, not a single multi-task policy. For the planned cluster sweep that's 16 tasks × 2 baselines × 3 seeds = 96 individual checkpoints, vs the template's 2 baselines × 3 seeds = 6 multi-task checkpoints. This is a deliberate deviation, made for two reasons:

1. **Multi-task PPO on 16 contact-rich manipulation tasks is an open research problem**, well beyond the project's scope. The MIKASA-Robo result already showed that single-task PPO on memory tasks of comparable difficulty lands at ~0% SR; multi-task RL almost certainly does worse, not better.
2. **Single-task results are a cleaner upper bound for the RL paradigm.** If single-task RL at the chosen step budget can't solve task X, multi-task RL definitely can't, so the single-task setting is the most generous evaluation for the methods being benchmarked.

**Implication for the writeup:** numbers should be presented as "per-task single-task RL upper bounds" and explicitly distinguished from the leaderboard's multi-task IL submissions. Direct comparison to the paper's Human (96%) / Oracle (85.8%) / IL (39.6%) BinFill numbers is fair *in absolute SR* but the underlying training-cost / generalization story is different — the IL methods produce one policy that handles all 16 tasks.

## File pointers

- Training logs: `runs/<run>/train.log`
- This eval (v7): `runs/eval_results/ppo_icm_v7_BinFill.json`, `.log`
- Failed eval (v4): `runs/eval_results/ppo_v4_BinFill.log`
- Memory file with full bottleneck analysis: `~/.claude/projects/-home-jevon-projects/memory/robomme-rl-fps-bottlenecks.md`
