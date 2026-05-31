# Next Steps — Research Plan after 0% Baseline

_Created 2026-05-24. Working doc; supersedes any earlier "what to run next" notes._

## Where we are

- Baselines completed: PPO (`ppo_v4`, 252k steps) and PPO+ICM (`ppo_icm_v7`, 254k steps) on **BinFill**.
- **Both eval to 0/20 success on the test split** at the official 1500-step horizon (see `REPORT.md`).
- Training showed real learning signal (shaped reward 8.6 → 11.5 for ICM, ICM intrinsic decaying 0.49 → 0.016) — the *pipeline* is correct; the *policy* never closes out an episode.
- Pre-existing context: no RL baseline exists in the RoboMME paper (all baselines are IL/VLA). Closest comparable RL prior work — MIKASA-Robo (ICLR 2025) — reports PPO/PPO-LSTM collapse to ~0% on hard memory tasks even at scale.

## Direction chosen: B (actually solve it)

Among the three directions considered:

- (A) Establish negative result rigorously — defer / pair with B.
- **(B) Solve it via better memory + better curiosity — chosen.**
- (C) Pivot to a different benchmark — not now.

## Algorithmic upgrades

### Memory (replaces the unfinished PTP-transformer B3)

Current B3 (state-only K=8 transformer with PTP auxiliary loss) has three weaknesses: state-only tokens, K=8 ≈ 0.4 s @ 20 Hz, fragile custom training.

| Option | What it is | Cost | Risk |
|---|---|---|---|
| **M1: RecurrentPPO (LSTM)** via `sb3-contrib` | LSTM after the visual extractor; SB3-contrib handles BPTT + hidden-state propagation across the rollout. | ~60 LOC, one new file | low — well-tested, MIKASA showed PPO-LSTM > PPO-MLP on memory tasks |
| M2: Visual+state transformer, K=32-64 | Replace PTP's state-only tokens with `[vis_t, state_t]` pairs. | ~150 LOC | medium — closer to the field's 2026 frontier but more unstable |
| M3: Dreamer-style RSSM | Latent world model + imagined rollouts. | Weeks of engineering | high — out of scope this term |

**Plan: start with M1.**

### Curiosity (replaces ICM)

Current B2 (Pathak ICM): forward-dynamics MSE as intrinsic. Saturates fast (in v7: 0.49 → 0.016 in 250k steps — the curiosity exhausted *before* the policy succeeded once). See `icm_dissociation_binfill.png`.

| Option | What it is | Why better than ICM for our setting | Cost |
|---|---|---|---|
| **C1: RND** (Random Network Distillation) | Predictor matches a frozen random target; novelty reward never saturates because predictor only fits *seen* states. | First-ever solver of Montezuma's Revenge; no action dependency → no noisy-TV problem; predictor saturation can't happen on the *unseen* states we actually need to find | ~120 LOC, mirrors `ppo_with_icm.py` |
| C2: NovelD | RND + episodic-novelty multiplier. | SOTA on hard-exploration episodic tasks (MiniGrid). Particularly relevant to BinFill-style "have I picked *this particular* cube this episode?" | +50 LOC on top of RND |
| **C3: RIDE** | Intrinsic reward = state-feature change between $t$ and $t+1$, inverse-visitation scaled. | **Most BinFill / PickXtimes-relevant intrinsic motivation in the literature**: picking up a cube *changes* state, so the act of manipulating gets rewarded regardless of whether the agent reaches the goal yet | ~60 LOC |
| C4: Disagreement ensemble | Variance across K forward models. | Robust to stochastic dynamics — but RoboMME physics is deterministic, so probably overkill. | ~80 LOC, K× compute |

**Plan: RND as the new default; layer RIDE on top for the BinFill/manipulation-heavy tasks.**

## Task selection — what to validate on

BinFill was the wrong starter — it was the *default* in the existing scripts (inertia), not a careful pick. Per the RoboMME paper: "policies struggle in cluttered scenes like BinFill…" — its motor difficulty (multi-cube + bin + button + clutter) confounds the contribution of memory/curiosity.

### What "right difficulty" means for this project

The task needs:

- **Sparse extrinsic reward** — otherwise curiosity is wasted.
- **Information distributed across time** — otherwise memory is wasted.
- **Manageable motor complexity** — otherwise control failures dominate.
- **Multi-step structure** — otherwise exploration / planning don't matter.

### Estimated difficulty matrix (no published per-task RL numbers — best-guess from paper task descriptions)

| Task | Motor | Memory demand | Curiosity demand | Verdict |
|---|---|---|---|---|
| MoveCube | low | none | low | pipeline sanity check only |
| PickHighlight | low-med | low | low | sanity check |
| **ButtonUnmask** | **low** | high (pre-mask position) | medium | clean *single-axis memory* demo |
| VideoUnmask | medium | high | medium | memory demo, harder motor |
| **PickXtimes** | medium | **high (count)** | **high (find repeat-pick pattern)** | **best both-axes demo** |
| StopCube | medium | high (timing) | medium | humans fail; deprioritize |
| ButtonUnmaskSwap | low | very high (swap) | high | harder ButtonUnmask variant |
| BinFill | **high** | high | high | what we tried — motor too hard |
| PatternLock / RouteStick | high | very high (long horizon) | medium | humans fail; too hard for first RL |
| InsertPeg | very high | low | low | pure control; doesn't fit story |

### Recommendation

**Single-task pick: PickXtimes.**

- Memory: must internally track "I've picked N times" — *reactive policies fundamentally cap at 1/N*. Clean ablation story.
- Curiosity: sparse extrinsic reward; agent must discover "repeat the pick primitive". RND + RIDE both directly applicable.
- Motor: one object type, one primitive — moderate, not crippling.
- Ablation table writes itself:
  - PPO (no memory, no curiosity) — should cap near 1/N
  - PPO + curiosity only — finds first pick but no memory of count
  - PPO + memory only — can count *if* it stumbles into first success
  - PPO + memory + curiosity — both axes contribute

**If we have budget for a per-task ablation** (recommended — ~1 day extra on 20 GPUs):

> **{MoveCube, ButtonUnmask, PickXtimes}** — climbing the demand ladder.
> - MoveCube: "RL works without memory/curiosity" (control upper bound)
> - ButtonUnmask: "memory matters" (isolated)
> - PickXtimes: "memory + curiosity both matter together" (joint)

Same RecurrentPPO + RND stack against all three. 3 tasks × 4 method cells × 3 seeds = 36 jobs ≈ 1-2 days wall-clock at 20 GPU cap.

## Concrete implementation plan

1. **Write `train/rewards/pickxtimes.py`** — shaped reward so 1-2M steps is enough to learn. Approx 50 LOC: TCP→cube distance + per-pick bonus + count-progress bonus + terminal +10/-5. Register in `train/rewards/__init__.py`.
2. **Write `train/algos/ppo_with_rnd.py`** — RND analogue of `ppo_with_icm.py`. Use the `cached_feat_t` reuse trick we already added. Predictor + frozen target on the shared 576-d extractor features. One Adam optimizer for the predictor.
3. **Write `train/models/rnd.py`** — predictor + target networks (4-layer MLPs on top of 576-d features).
4. **Write `train/train_ppo_recurrent.py`** — RecurrentPPO scaffold using `sb3-contrib`. Multi-input LSTM policy. Reuse our `RobommeCNNExtractor`.
5. **Write `train/train_ppo_recurrent_rnd.py`** — combines (2) + (4). Headline model.
6. **(Optional) `train/algos/ppo_with_ride.py`** — RIDE intrinsic, plug-compatible with the RND code paths.
7. **Update cluster sbatch** — add `b4=RecurrentPPO`, `b5=RecurrentPPO+RND` as `BASELINE` choices in `train_array.sbatch`.
8. **Add `train/rewards/{movecube,buttonunmask}.py`** if doing the 3-task ablation.

Total: ~600 LOC of new training code, ~2 days focused implementation.

## What we are explicitly NOT doing

- Continuing to optimize BinFill with the current method stack (already shown 0%; diminishing returns).
- More algorithm permutations (SAC, TD-MPC2 — MIKASA already shows they're not the bottleneck).
- Engineering further FPS (eval-compatibility ceiling is ~30-50 FPS on Titan RTX; sufficient).
- Multi-task RL training (open research problem; out of scope this term).
- Behavior-cloning warm-start (would lose the "pure RL" framing; revisit only if the new stack also stalls).
- VLM backbone (π0.5 / OpenVLA RL fine-tune) — out of scope; documented as future work.

## Open questions to revisit after first results

- Does the new stack solve **MoveCube**? If not, the bottleneck is control, not memory/curiosity.
- Does it solve **ButtonUnmask**? If not, the recurrent memory is insufficient.
- Does it solve **PickXtimes**? If not, even the most generous memory+curiosity demonstration on RoboMME requires more than current off-the-shelf methods can deliver — *itself a publishable finding*.

## Pointers

- `runs/eval_results/REPORT.md` — current 0% numbers + caveats.
- `runs/eval_results/icm_dissociation_binfill.png` — the "curiosity exhausted before solving" chart that motivates moving to RND.
- `runs/eval_results/policy_contraction_binfill.png` — "policy still exploring at 250k" chart that motivates RecurrentPPO + longer training.
- `~/.claude/projects/-home-jevon-projects/memory/robomme-rl-research-plan.md` — distilled version of this doc that persists across sessions.
