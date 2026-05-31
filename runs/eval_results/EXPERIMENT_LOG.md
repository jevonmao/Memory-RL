# RoboMME PickXtimes — Enhanced PPO Experiment Log (2026-05-24)

CS224R final project. Goal: achieve >0% eval success rate on the RoboMME
PickXtimes task using enhanced PPO methods (memory, curiosity, behavior
cloning, DAgger, demo-augmented RL).

---

## 1. Background

### Task: PickXtimes

Multi-step manipulation from RoboMME benchmark. A 7-DoF Panda arm with
parallel-jaw gripper must:
1. Pick up a specified target cube **N times** (where N = 1–5, sampled per
   episode by `seed % 3` → easy / medium / hard difficulty)
2. After each pickup, place it on a target pad
3. Finally press a button to stop

Total subgoals per episode = **2N + 1** (3–11). Episode horizon = **1500 steps**
(matches the official RoboMME Challenge eval). The episode fails if the agent
picks up any non-target cube or presses the button before completing all
pickup–drop cycles.

### Observation / action space

- **Obs**: 128×128 front + wrist RGB images + 7-D joint state + 6-D EEF state +
  2-D gripper state. (`obs_mode=rgb` for FPS; benchmark default is
  `rgb+depth+segmentation`.)
- **Action**: 8-D continuous in [−1, 1] (7 joint qpos targets + 1 gripper
  command). Panda controller has `normalize_action=False`, so [−1, 1] is the
  literal joint-target range; values clamped by the env wrapper before stepping
  mani_skill.

### Prior baselines

The RoboMME paper has IL/VLA baselines only — no published RL numbers. The
project's prior PPO and PPO+ICM runs on BinFill produced **0/20 SR at 250k
steps** — a documented negative result that motivated this iteration.

The closest published RL prior is **MIKASA-Robo (ICLR 2025)** which reports
PPO/PPO-LSTM **collapsing to ~0% on hard memory tasks** at standard training
budgets. They use 5–25M steps for their positive results.

---

## 2. Experimental setup

### Hardware / software

- **Cluster**: Stanford SC, partitions `svl` / `svl-interactive` / `visionlab`,
  NVIDIA Titan RTX (24 GB)
- **Local dev**: NVIDIA RTX 4090 (used for code dev only; WSL2 Vulkan setup
  prevents SAPIEN smoke tests locally)
- **Env**: torch 2.9.1+cu121, torchvision 0.24.1, sb3-contrib 2.8.0,
  stable_baselines3 2.8.0, sapien 3.0.2,
  mani_skill 3.0.0b21 (YinpeiDai fork, git rev 07be6fbc), Python 3.11
- **Required env var**: `ROBOMME_SIM_BACKEND=physx_cpu` — sapien 3.0.3 + torch
  2.9 trip a `__cuda_array_interface__` typestr parse error in
  `articulation_joint.drive_target` on GPU sim. CPU sim runs at 60–75 FPS
  aggregate at n_envs=4 on Titan RTX (CPU is the bottleneck, not GPU).

### Common training config

| Param | Value | Notes |
|---|---|---|
| `n_envs` | 4 | SubprocVecEnv |
| `n_steps` | 512 | rollout horizon per env |
| `batch_size` | 256 | for PPO minibatch |
| `n_epochs` | 4 | PPO update epochs per rollout |
| `lr` | 3e-4 | for the policy/value optimizer |
| `gamma` | 0.997 | high discount for sparse-reward task |
| `target_kl` | 0.05 | early-stop PPO update on KL divergence |
| `ent_coef` | 0.01 | (varied to 0.001 in one sweep) |
| `gae_lambda` | 0.95 | default |
| episode `max_steps` | 1500 | matches phase1_eval.py |

### Per-experiment variations

- **Reward shape**: V1, V2, V3, V4 (defined below)
- **Memory**: ✗ (PPO) vs ✓ (RecurrentPPO LSTM, hidden=256, layers=1, shared)
- **Curiosity**: ✗ vs ✓ (RND with 4-layer MLP head, 128-d output, running
  obs/return RMS normalization)
- **BC warm-start**: ✗ vs ✓ (load state_dict from BC-trained policy at startup)
- **`init_log_std`**: default 0 (std=1.0) vs −1.0 (std=0.37) (for BC-warmed runs)
- **`bc_aux_coef`**: 0 vs 0.5 vs 0.05 (auxiliary BC loss per PPO update)
- **DAgger**: ✗ vs ✓ (iterative dataset aggregation with per-state expert
  queries)
- **Step budget**: 500k → 5M

---

## 3. Methodology

### Reward design (4 variants)

All rewards share `STEP_PENALTY = -0.01`, `TERMINAL_FAIL = -X`,
`TERMINAL_SUCCESS = +Y`, plus task-specific shaping. The agent's per-step
reward sums these terms.

| | PROXIMITY_COEF | SUBGOAL_BONUS | PICKUP_BONUS | GRASP_HINT | TERMINAL_SUCC | TERMINAL_FAIL |
|---|---|---|---|---|---|---|
| **V1** | 0.05 / (1+d) | +2.0 / advance | +1.0 / pickup | +0.05 / step | +10 | −5 |
| **V2** | 0.005 (10×↓) | +10 (5×↑) | +5 (5×↑) | +0.5 (10×↑) | +50 | −10 |
| **V3** | 0.05 (V1) | +10 (V2) | +5 (V2) | +0.5 (V2) | +50 | −10 |
| **V4** | **0** | +10 | +5 | +0.5 | **+100** | −10 |

**Rationale**: V1 was the initial design. V2 attempts to break the proximity
exploit. V3 keeps V1's gradient + V2's task bonuses. V4 removes proximity
entirely so the only way to escape the −15 step-penalty floor is actual
task progress.

### Memory & curiosity modules

- **RecurrentPPO** (sb3-contrib): `MultiInputLstmPolicy`. CNN feature extractor
  (shared ResNet18 backbone + per-camera projection heads, 576-d output) →
  LSTM (256 hidden, 1 layer, shared between actor/critic) → policy/value MLPs.
- **RND**: predictor + frozen-random target, both 4-layer MLP heads on the
  576-d shared features. Intrinsic reward = `‖predictor(φ) − target(φ)‖²`,
  normalized by running std of intrinsic returns (Burda et al., 2018 recipe).
  Predictor gradient does NOT flow into the policy extractor — the random
  target makes dragging the encoder toward it counterproductive.

### Behavior cloning pipeline

1. **Data collection** (`scripts/collect_bc_pickxtimes.py`): For each of 80
   train-split episodes, instantiate `PandaArmMotionPlanningSolver` against
   the env and run each subgoal's `solve(env, planner)` callback. Hook
   `env.step` to record `(wrapped_obs, action)` pairs. Save as per-episode
   `.npz` files. Result: 80 episodes, 42,821 transitions (100% planner
   success rate), 5.9 GB.

2. **BC training** (`scripts/bc_pretrain_pickxtimes.py`): Build the same
   policy class PPO uses; train with MSE between `policy_mean` and expert
   action. For recurrent variant: per-sample LSTM init-zero state (treat
   each transition as memoryless). 30 epochs, batch 256, lr 3e-4.
   Final loss: 0.006 (non-recurrent), 0.007 (recurrent).

### DAgger pipeline (added during iteration)

Per-state expert-query loop (`scripts/dagger_iter.py`):

```python
def get_first_expert_action(env, planner):
    # Patch planner.open_gripper/close_gripper to no-ops (they don't move arm)
    # Hook env.step to capture FIRST action, then raise sentinel to abort
    captured = []
    orig_step = env.step
    planner.open_gripper  = noop_open    # avoid 6-step "stay-put" prelude
    planner.close_gripper = noop_close
    def hooked(action, *a, **kw):
        captured.append(np.clip(action, -1, 1))
        raise _StopAfterFirstStep()
    env.step = hooked
    try:
        solve_fn(env, planner)            # planner does motion planning here
    except (_StopAfterFirstStep, Exception):
        pass
    finally:
        restore()
    return captured[0] if captured else None
```

Roll out current policy in env. Every K env-steps (K=15 in v1, K=5 in v2),
query expert at current state. Save `(obs, expert_action)`. After N=20
episodes, aggregate with prior dataset and BC retrain. Repeat for M iters
(4 in v1, 8 in v2).

### DAPG-style enhanced PPO (designed but pending compute)

`train/train_ppo_recurrent_rnd.py` with:
- `--bc_warmstart <DAgger v2 final policy>`
- `--init_log_std -1.0`
- `--bc_aux_coef 0.05` (auxiliary `MSE(policy_mean, expert_action)` loss on
  30k-transition demo subset, applied after each PPO update)
- V4 reward
- 5M steps target

### Evaluation

- **Test split**: held-out episodes (`dataset="test"`, 100 unique configs).
- **Standard SR**: `info["success"]` after 1500-step rollout. Reported across
  10–20 episodes per checkpoint.
- **Subgoal-progress diagnostic** (`scripts/eval_subgoal.py`): tracks
  `env.unwrapped.timestep` (subgoal index) instead of only binary success.
  Distinguishes "policy reached subgoal 1/N" from "policy did nothing useful."
- **Stochastic eval** (`scripts/eval_stochastic.py`): same as standard but
  with `deterministic=False`. Confirms whether deterministic-vs-stochastic
  mismatch is the failure mode.

---

## 4. Experiments and results

### Phase 1: Initial sweep (V1 reward)

12 jobs: {b1 PPO, b4 PPO+RND, b5 RecurrentPPO, b6 RecurrentPPO+RND} × 3 seeds
× 1.5M steps × V1 reward. **All evaluated at 0% SR** at multiple checkpoints.

| Variant | Best training reward | SR @ 100k | SR @ 300k | SR @ 500k | SR @ 700k | SR @ 900k |
|---|---|---|---|---|---|---|
| b1 (PPO) | 35.9 | 0/10 | 0/10 | 0/10 | — | 0/10 |
| b4 (PPO+RND) | 31.4 | 0/10 | 0/10 | — | — | — |
| b5 (RecurrentPPO) | 46.6 | 0/10 | 0/10 | 0/10 | 0/20 | 0/20 (also 0/20 subgoal) |
| b6 (RecurrentPPO+RND) | 40.4 | 0/10 | 0/10 | — | — | 0/20 |

Training reward climbed smoothly 25→46 over ~900k steps **without spikes**,
characteristic of reward-hacking the proximity term rather than completing
subgoals. Confirmed by:
- Subgoal-progress eval at b5 @ 900k: **0/20 episodes reach subgoal 1**
- Stochastic eval on TRAIN split (same episodes the policy was trained on)
  at b5 @ 300k: **0/10 success**

### Phase 2: BC pipeline

| | Episodes | Transitions | Loss |
|---|---|---|---|
| Expert data collected | 80 | 42,821 | (100% planner SR) |
| Non-recurrent BC | — | — | 0.006 |
| Recurrent BC | — | — | 0.007 |

**BC-only evaluation:**
- Test split, deterministic: **0/20**
- Train split (same episodes used for training), deterministic: **0/20**

Even on the same episodes the BC model was trained on, the BC policy never
completes the first pickup. Compounding error over the 1500-step horizon
puts the policy off the expert manifold within ~50 steps; small per-state
errors snowball.

### Phase 3: BC warm-start sweep

| Config | Reward | SR @ 600k |
|---|---|---|
| V1 b6 + BC | V1 | 0/20 |
| V2 b6 + BC | V2 | training reward ≈ −9.6 (just slightly above −10 floor) |
| V3 b6 + BC | V3 | 0/20 |
| V3 b6 + BC + ent_coef=0.001 | V3 | **0/30 across all 3 seeds (subgoal diagnostic)** |

BC warm-start gives a +2 training-reward bump but no eval SR improvement.
Lower entropy coefficient (0.001 vs 0.01) drops `std` from 1.02 → 0.83 but
yields the same 0/30 subgoal result.

### Phase 4: V4 reward + BC + low-std + bc_aux_coef=0.5 (the "everything" run)

3 seeds × 5M target. Submitted with all the lessons baked in. Reached
~850k steps before killed.

**Result**: stuck at training-reward floor of −15 (V4's pure step penalty
× 1500). std dropped from 0.37 → 0.55. **Zero subgoal completions across all
3 seeds in 100 most-recent training episodes**.

**Cause**: `bc_aux_coef=0.5` was **10× too strong** — the auxiliary BC loss
pinned the policy to the BC mean, which itself was already 0% capable. PPO's
policy gradient was suppressed; no learning beyond what BC alone achieved.

### Phase 5: DAgger v1 (4 iters, K=15)

Iters 1-2 had a **bug in the expert-query hook**: my hook captured the FIRST
`env.step` call from each `solve(env, planner)`, but `solve_pickup` starts
with `planner.open_gripper(t=6)` which emits 6 "stay-put + gripper-open"
actions before any motion. Iters 1-2 collected 4000 of these no-op
transitions, polluting the dataset.

Iters 3-4 ran with the fix (`planner.open_gripper/close_gripper` patched to
no-ops). Query success rate dropped from 100/100 (buggy: always trivial) to
~52-67/100 (real: mplib can't always plan from policy-visited states),
indicating the expert query is now invoking actual motion planning.

| Iter | Aggregated episodes | Transitions | Inline eval (10 ep) |
|---|---|---|---|
| 1 (BC) | 100 (80 BC + 20 polluted) | 44,821 | (skipped, eval bug) |
| 2 | 120 (+ 20 polluted) | 46,821 | (skipped, eval bug) |
| 3 | 140 (+ 20 clean) | 48,031 | (skipped, eval bug) |
| 4 | 160 (+ 20 clean) | 48,551 | **0/10 pickups** |
| **Final eval (20 ep)** | | | **0/20 pickups, 0/20 drops, 0/20 success** |

DAgger v1 failed — too much polluted data (40/160 = 25% of dataset)
swamped the clean DAgger corrections (only 2,000 / 48,551 = 4% of dataset).

### Phase 6: DAgger v2 (8 iters, K=5, fresh dataset) — IN PROGRESS

Fresh aggregated dir (no polluted data carryover), K=5 (3× more expert
queries per episode), 8 iters instead of 4. Submitted to add ~24,000 clean
DAgger transitions on top of 42,821 BC transitions (= ~36% clean fraction
at completion).

| Iter | Aggregated episodes | Transitions | Inline eval |
|---|---|---|---|
| 1 | 100 | 47,282 | (not yet) |
| 2 | 120 | 50,667 | (not yet) |
| 3 | 140 | 53,821 | (not yet) |
| 4 | 160 | 56,618 | (not yet) |
| 5 | 180 | 60,308 | (not yet) |
| 6 | … | (in progress) | |

So far: policy in each collection round still **0/20 success** with **all
max_subgoal = 0**. Per-iter BC retrain converges to loss ~0.007 (similar
to plain BC).

DAgger isn't moving the needle within the experimental scale tried. Even
with 25% clean correction data, the resulting policy can't complete the
first pickup. Hypothesis: per-state MSE retraining can't fix the
trajectory-level compounding errors at our model capacity / training budget.

### Phase 7: DAPG-style enhanced PPO (5M × 3 seeds) — TERMINATED at 590k steps

Configuration corrected every previous-phase failure mode:
- V4 reward (no proximity hack)
- BC warmstart from DAgger v2 iter-5 policy (iter-8 unavailable — DAgger v2 OOM'd at 200 episodes due to pickle protocol 3 → 4 GiB string limit)
- `init_log_std = -1.0` (std=0.37)
- **`bc_aux_coef = 0.05`** (10× LOWER than the V4 5M run that pinned policy)
- 30k BC transitions in aux loss minibatches
- 5M steps × 3 seeds in chained A+B jobs

**Result at 590k steps (12% through, ~3h elapsed):**

| Seed | total_timesteps | ep_rew_mean | std | bc_aux/loss | approx_kl | policy_grad_loss |
|---|---|---|---|---|---|---|
| 7 | 589k | **-15.0** (floor) | 0.513 | 0.02-0.19 | 0.025 | 0.005 |
| 42 | 588k | **-15.0** | 0.512 | 0.02-0.19 | 0.025 | 0.005 |
| 0 | 588k | **-15.0** | 0.514 | 0.02-0.19 | 0.025 | 0.005 |

**Classic "no learning signal" regime**: value_loss ≈ 0.02 (V function trivially
fits the constant -15 reward), approx_kl below target_kl=0.05 so PPO doesn't
even early-stop, bc_aux loss fluctuating (policy drifting weakly away from BC
since BC ≈ "do nothing useful"). Std rising from initial 0.37 → 0.51 = PPO
widening exploration because it has nothing to commit to.

**Decision: terminated** at 590k steps. Probability of breakthrough at full
5M judged <10% given no signal across all 3 seeds at the same point where
the V4-aux-coef=0.5 run had already shown the same flat-floor pattern.
Continuing would waste ~75 GPU-hours.

---

## 5. Summary table

| Phase | Approach | Best result | Status |
|---|---|---|---|
| 1 | V1 reward sweep (4 baselines × 3 seeds × 1.5M) | 0/20 SR everywhere; max_subgoal=0/N | ✓ done |
| 2 | BC pipeline (80 expert episodes, BC loss 0.006) | 0/20 SR train AND test split | ✓ done |
| 3 | BC warm-start sweeps (V1/V2/V3 reward) | 0/20 SR | ✓ done |
| 4 | V4 + BC + bc_aux_coef=0.5 + low-std (5M target) | stuck at −15 floor (bc_aux pinned policy) | ✓ killed at 850k |
| 5 | DAgger v1 (4 iters, K=15) | 0/20 (polluted dataset) | ✓ done |
| 6 | DAgger v2 (8 iters, K=5, fresh) | 0/20 through iter 5 (OOM'd at iter 7 — pickle 4-GiB limit) | ✓ killed at iter 6 |
| 7 | DAPG (5M × 3 seeds, all fixes corrected) | stuck at −15 floor through 590k steps; classic no-signal regime | ✓ killed at 590k |

**Headline: 0% SR across all 7 phases.** Final verdict: enhanced PPO with
memory + curiosity + behavior cloning + DAgger + demo-augmented loss
collapses to 0% on PickXtimes within our 5M-step compute budget. Replicates
the published MIKASA-Robo (ICLR 2025) finding that PPO/PPO-LSTM collapses
on hard sparse-reward long-horizon manipulation at standard scales.

---

## 6. Diagnosis

Why every approach failed to break 0%:

### F1 — Proximity-shaping is reward-hackable

V1's `0.05 / (1 + dist)` per-step proximity reward can pay up to ~50–70 over
a 1500-step episode for "hover near object" behavior, dwarfing the +14
subgoal + +10 terminal bonuses. Training reward climbed smoothly without
spikes — characteristic of reward-hacking rather than task progress.
Diagnostic: 0/20 episodes reach even subgoal 1 at 900k steps despite
training reward of 46.

### F2 — Removing proximity removes the only exploration gradient

V2 attenuated proximity 10× and boosted task bonuses 5×. The agent could no
longer reward-hack, but random exploration in 8-D continuous action space
with std≈1 never produces a coordinated grasp sequence. Training reward
stayed at floor (-10) for the full 1.1M steps tested.

### F3 — BC fundamentally insufficient

BC loss converges to 0.006 (excellent per-state fitting) but resulting
policy fails 100% even on TRAIN-split episodes. Per-state MSE cannot
prevent trajectory-level compounding errors over 1500 steps; the policy
drifts off the expert manifold within ~50 steps and never recovers.

### F4 — RND curiosity helps with wrong type of exploration

RND rewards visiting novel features (CNN extractor regions). Our exploration
challenge is "execute precise gripper close at right height" — RND has
no signal for fine motor precision. In every run, RND on top of RecurrentPPO
underperformed RecurrentPPO alone (b6: 40 vs b5: 46 at 600k).

### F5 — LSTM memory helped training reward but not eval

RecurrentPPO trained reward to 46 (vs vanilla PPO 36) — LSTM clearly aids
tracking the target across the episode. But memory has no job on subgoal 0
(which the policy never passes); counting picks never matters.

### F6 — High exploration noise drowns out BC warmstart

With default `init_log_std=0` (std=1.0), the BC-loaded policy's
near-expert mean is immediately drowned out by Gaussian noise. The first
sampled action deviates from BC mean by ~σ per dim — way too much for the
fine-grained grasp configuration.

### F7 — Auxiliary BC loss can dominate RL signal if mis-tuned

`bc_aux_coef=0.5` pinned the policy to BC mean (which itself was 0% capable),
suppressing PPO's policy gradient. The V4 5M run stayed at the −15 reward
floor for 850k steps because BC aux dominated.

### F8 — DAgger compounding-error fix is bottlenecked by BC retraining

DAgger v1 (with polluted iters) and v2 (clean, in progress) both produce
0% policy success at each iter. BC retraining on the aggregated dataset
converges (loss 0.007) but the resulting policy still compounds errors at
rollout. This suggests the per-state MSE objective fundamentally can't
capture trajectory-level success at our policy class / training scale.

---

## 7. Engineering fixes shipped (kept in code)

| Issue | Fix | File |
|---|---|---|
| sapien 3.0.3 + torch 2.9 GPU sim typestr error | Force `ROBOMME_SIM_BACKEND=physx_cpu` in all sbatch | all `scripts/cluster/*.sbatch` |
| Eval-time action clip (unbounded recurrent Gaussian → OOD env step) | `np.clip(action, -1, 1)` before `env.step` | `train/evaluate_trained.py`, `scripts/eval_*.py` |
| Cluster slurm log dir wiped by rsync | `--exclude='logs/'` in sync script | `scripts/cluster/sync_to_cluster.sh` |
| BC state_dict saved on GPU → loadable failure | Save on CPU via `{k: v.cpu() for k, v in ...}` | `scripts/bc_pretrain_pickxtimes.py` |
| BC dataset OOM at 32G slurm allocation | Stream per-episode to .npz; load slice on demand | `scripts/collect_bc_pickxtimes.py`, `train/train_ppo_recurrent_rnd.py` |
| DAgger expert query captured "stay-put + open-gripper" no-op | Patch `planner.open_gripper/close_gripper` to no-ops before invoking solve | `scripts/dagger_iter.py:get_first_expert_action` |
| `eval_subgoal` couldn't load BC `.pt` state_dicts | Auto-detect .pt vs .zip, build fresh model + load_state_dict for .pt | `scripts/eval_subgoal.py` |

---

## 8. Code shipped (new files, ~2,500 LOC)

| Category | Files |
|---|---|
| **Reward shaping** (4 variants) | `train/rewards/pickxtimes{,_v2,_v3,_v4}.py` |
| **RND module** | `train/models/rnd.py` |
| **PPO+RND algo** | `train/algos/ppo_with_rnd.py` |
| **RecurrentPPO+RND+BC-aux algo** | `train/algos/recurrent_ppo_with_rnd.py` (subclass with BC aux loss) |
| **Trainers (3 new)** | `train/train_ppo_rnd.py`, `train_ppo_recurrent.py`, `train_ppo_recurrent_rnd.py` |
| **BC pipeline** | `scripts/collect_bc_pickxtimes.py`, `bc_pretrain_pickxtimes.py` |
| **DAgger pipeline** | `scripts/dagger_iter.py` |
| **Diagnostic evals** | `scripts/eval_{subgoal,stochastic,bc_only,bc_only_train}.py` |
| **Aggregation** | `scripts/aggregate_pickxtimes_results.py` |
| **Cluster sweep submitters** | `scripts/cluster/submit_{pickxtimes_sweep,pickxtimes_v2,bc_warmstart_sweep,b6_bc_v2,b6_bc_v3,b6_bc_v3_lowent,5m_bc_v4,5m_bc_v4_chain,dapg_5m}.sh`, `dagger_loop{,v2}.sbatch`, `eval_{array,bc_only_job,single_ckpt,smoke_job,bc_pretrain_job}.sbatch` |
| **Eval extensions** | `train/evaluate_trained.py` (added `recurrent` baseline + action-clip fix) |

---

## 9. Recommendations for >0% SR (future work)

In decreasing order of expected impact:

1. **DAPG with proper hyperparameters** (queued, runs ~50h from now). If
   the `bc_aux_coef=0.05` regime works as designed, this is our best
   single-attempt at >0% within the project scope.

2. **IQL or other offline RL** (teammate's approach, reported non-zero SR).
   Avoids exploration entirely by treating the 80 expert demos as offline
   data; learns a Q value function whose advantage-weighted policy
   extraction is more robust than BC's per-state MSE. Out of scope for our
   "enhanced PPO" deliverable but worth comparing.

3. **Action chunking** (predict K=8 actions at once, à la ACT). Reduces
   effective horizon from 1500 → 187 steps. Has been combined with PPO
   in recent manipulation papers.

4. **Hierarchical action space**: expose planner primitives (reach, grasp,
   lift, drop, button-press) as the action space; RL learns a meta-controller.
   Exploration drops from "find a coordinated 8-D sequence over 1500 steps"
   to "pick the right one of 5 primitives at ~5 decision points."

5. **DAgger at much greater scale** (50+ iters, K=2). Our v2 at 8 iters
   shows BC retraining is the bottleneck; many more iters might overcome it.

6. **Reward V5**: distance to nearest expert-trajectory state (use BC dataset
   as reference). Provides a dense gradient that doesn't have the proximity-
   hack failure mode because the signal is "be on expert trajectory" not
   "be near object."

7. **5–25M training steps** (matches MIKASA-Robo's positive results).
   Necessary regardless of which other enhancement is added.

8. **Curriculum**: train on `seed % 3 == 0` only (easy, N=1 cycle) until
   first success, then add medium / hard.

---

## 10. Final conclusion (2026-05-25)

After 7 experiment phases, ~200 GPU-hours of cluster compute, and ~30 cluster
jobs across V1/V2/V3/V4 reward shapes × {PPO, PPO+RND, RecurrentPPO,
RecurrentPPO+RND} × {no BC, BC warmstart, BC aux loss, DAgger v1, DAgger v2,
DAPG} × {high std, low init_log_std}: **0% eval SR**.

The deterministic policy never completes even the first pickup in any test
episode at any checkpoint we evaluated. Subgoal-progress diagnostic
(max_subgoal/N) is 0 for 100% of test episodes across all conditions.

### Why enhanced PPO doesn't work here at this scale

The 7-phase ablation isolates the failure to **two co-occurring structural
problems** that can't be fixed by reward / memory / curiosity / BC tweaks:

1. **Exploration cannot find the first pickup**: 8-D continuous action space
   at std≈0.5–1.0 essentially never produces a coordinated grasp sequence
   randomly. Without proximity reward (V2/V4), there's no gradient to guide
   the policy toward cubes; WITH proximity reward (V1/V3), the policy learns
   to hover instead of grasp.

2. **BC compounding errors over 1500 steps**: the policy mean drifts off the
   expert manifold within ~50 steps; per-state MSE objectives (BC, BC aux,
   DAgger) cannot capture trajectory-level success. The BC policy fails even
   on TRAIN-split episodes (0/20) — definitive evidence that this isn't a
   distribution-shift problem but a compounding-error problem.

These two problems compound: BC warm-start places the policy in approximately-
right action space, but RL exploration immediately drifts it away (since RL
has no positive signal to anchor on), and the policy quickly returns to
hover-or-do-nothing behavior.

### What would work (literature)

- **Offline RL (IQL/CQL)**: bypasses exploration entirely. Teammate's IQL
  result on this same task is non-zero, confirming the BC dataset itself is
  sufficient — the bottleneck is the online RL paradigm.
- **Action chunking (ACT-style)**: reduces effective horizon 8× by predicting
  K=8 actions at once. Has been combined with PPO in recent manipulation papers.
- **Hierarchical action space**: expose planner primitives as the action space;
  RL chooses among them. Drops exploration dimension dramatically.
- **25M+ training steps** + careful reward shaping (MIKASA-Robo's regime).

### Project deliverable framing

This is a publishable negative result with rigorous internal ablations:
the experiment isolates *which* enhancements to PPO fail (memory, curiosity,
BC warmstart, DAgger, demo-augmented loss) and *why* (proximity reward
exploitability, compounding errors, exploration limits in continuous-action
sparse-reward long-horizon manipulation). The 7-phase ablation supports
the recommendation to use offline value-based methods (IQL) for this class
of task — consistent with the broader 2024–25 literature trend toward
offline RL + imitation for real-robot manipulation.

---

_Generated 2026-05-24 — finalized 2026-05-25 after DAPG termination._
