# CLAUDE.md — memory-RL

Context for future Claude Code sessions in this repo.

## Project

CS224R project: **Curiosity-Driven Memory-Augmented RL for Adaptive Robot Tasks.**
Benchmark is RoboMME (16 tasks across counting / permanence / reference / imitation memory).
The vanilla PPO baseline is scaffolding only — RoboMME's extrinsic reward is reserved,
so the research contribution is intrinsic reward via **ICM** and memory-conditioned
**M-ICM** layered on top of a **PTP**-style memory module.

See `README.md` for layout, commands, and known caveats — don't duplicate that here.

## Environment

- Conda env `memrl` at `/home/jevon/miniforge3/envs/memrl/` — use this Python, **not**
  system Python (user feedback: never `pip install` into system Python).
- Installed: sb3 2.8.0, torch 2.12.0+cu130, gymnasium 1.2.3.
- RoboMME is cloned at `/home/jevon/projects/robomme_benchmark` but **not** installed
  into `memrl`. Use `ROBOMME_PATH=/home/jevon/projects/robomme_benchmark` so the
  adapter prepends `$ROBOMME_PATH/src` to `sys.path`.
- WSL2 Linux. No local GPU usable for ManiSkill/SAPIEN — GPU runs go to **Modal**.

## Run commands

CPU-only (works locally):
```
ROBOMME_PATH=/home/jevon/projects/robomme_benchmark \
/home/jevon/miniforge3/envs/memrl/bin/python scripts/inspect_env.py \
    --task CartPole-v1 --allow-gym-fallback
```

GPU (must run on Modal — do not attempt locally):
```
python scripts/inspect_env.py --task BinFill
python training/train_ppo.py --config configs/ppo.yaml --task <robomme_task> ...
```

## Status

| Phase | Status |
| --- | --- |
| 0. Scaffolding (env adapter, train, eval, logger, metrics, configs) | done |
| 0a. CartPole smoke (inspect_env, PPO, evaluate) | in progress, local CPU |
| 0b. RoboMME smoke (`inspect_env --task BinFill`) | blocked on Modal |
| 1. Vanilla PPO baseline (4 tasks × 3 seeds × 100k) | blocked on Modal |
| 2. ICM curiosity (`curiosity/icm.py` + VecEnvWrapper) | not started |
| 3. PTP memory + M-ICM (`memory/ptp.py`, `curiosity/m_icm.py`) | not started |
| 4. Full sweep (16 × 3 × {baseline, ICM, M-ICM}) + ablations | not started |

`memory/base_memory.py` and `curiosity/base_curiosity.py` are interface stubs only —
no implementations yet.

## Conventions

- Run dirs: `logs/<task>_seed_<seed>_<timestamp>[_<tag>]/`. Smoke runs use `--tag smoke`.
- Aggregated results land in `results/<phase>.json` (not yet created).
- Don't silently substitute Gym for RoboMME — `--allow-gym-fallback` is opt-in.
