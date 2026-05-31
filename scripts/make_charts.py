"""
Produce charts for the initial report:
  1. Training learning curves: shaped ep_rew_mean vs total_timesteps
     for the completed BinFill runs (ppo_v3, ppo_v4, ppo_icm_v7).
  2. Landscape bar chart: BinFill test-split success rate, ours vs
     the RoboMME paper's imitation-learning baselines and human reference.
  3. Reward/curiosity dissociation: two-panel plot showing shaped reward
     rising while ICM intrinsic reward decays — explains why training
     "progress" did not translate to task completion.
  4. Policy contraction: action-std and entropy_loss vs training steps,
     confirming the policy concentrated but slowly.

PNGs saved under runs/eval_results/.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# Pull metrics out of SB3 train.log box-format lines.
ITER_RE = re.compile(
    r"\|\s*(ep_rew_mean|total_timesteps|fps|intr_reward|fwd_loss|inv_loss"
    r"|entropy_loss|std|approx_kl|ep_len_mean)\s*\|\s*([-\d\.eE+]+)\s*\|"
)


def extract_iters(log_path: Path) -> list[dict[str, float]]:
    """Parse SB3 train.log into a list of per-iter metric dicts.

    SB3 prints metrics in fixed-order blocks (icm/, rollout/, time/, train/)
    per iter. We detect iter boundaries by seeing the same metric appear
    twice — at that point we flush the previous iter's dict and restart.
    """
    iters: list[dict[str, float]] = []
    cur: dict[str, float] = {}
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = ITER_RE.search(line)
            if not m:
                continue
            k, v = m.group(1), float(m.group(2))
            if k in cur:
                # repeat-metric signals iter boundary
                if "total_timesteps" in cur:
                    iters.append(cur)
                cur = {}
            cur[k] = v
    if "total_timesteps" in cur:
        iters.append(cur)
    return iters


def series(iters: list[dict[str, float]], key: str) -> tuple[list[int], list[float]]:
    """Return (steps, values) for `key`, filtered to iters where it's present."""
    xs, ys = [], []
    for it in iters:
        if key in it and "total_timesteps" in it:
            xs.append(int(it["total_timesteps"]))
            ys.append(it[key])
    return xs, ys


def extract_curve(log_path: Path) -> tuple[list[int], list[float]]:
    """Backwards-compat shim used by Chart 1."""
    return series(extract_iters(log_path), "ep_rew_mean")


RUNS = [
    ("ppo_v3",     "PPO (reward shaping v3)", "tab:gray",   "--"),
    ("ppo_v4",     "PPO (reward shaping v4)", "tab:blue",   "-"),
    ("ppo_icm_v7", "PPO + ICM",               "tab:orange", "-"),
]

RUNS_ROOT = Path("/mnt/c/Users/Jingwen Mao/projects/robomme_benchmark/runs")
OUT_DIR   = RUNS_ROOT / "eval_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Chart 1: training learning curves
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
for slug, label, color, ls in RUNS:
    log = RUNS_ROOT / slug / "train.log"
    if not log.exists():
        print(f"skip {slug}: no train.log")
        continue
    xs, ys = extract_curve(log)
    if not xs:
        print(f"skip {slug}: no parseable iters")
        continue
    ax.plot(xs, ys, label=label, color=color, linestyle=ls, linewidth=2)

ax.set_xlabel("Environment steps ($\\times 10^6$)")
ax.set_ylabel("Average shaped return")
ax.set_title("BinFill: training learning curves")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}"))
ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
chart1 = OUT_DIR / "training_curves_binfill.png"
fig.savefig(chart1)
print(f"wrote {chart1}")
plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 2: BinFill SR landscape
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
methods = [
    "Human",
    "GroundSG + Oracle",
    "FrameSamp + Modular",
    "PPO (Ours)",
    "PPO + ICM (Ours)",
]
scores = [96.0, 85.8, 39.6, 0.0, 0.0]
colors = ["tab:green", "tab:olive", "tab:cyan", "tab:blue", "tab:orange"]

bars = ax.bar(methods, scores, color=colors, edgecolor="black", linewidth=0.6)
for bar, s in zip(bars, scores):
    ax.annotate(f"{s:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, s),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=10,
                fontweight="bold" if s == 0 else "normal")

ax.set_ylabel("Success rate (%)")
ax.set_ylim(0, 105)
ax.set_title("BinFill: test-set success rate")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
chart2 = OUT_DIR / "binfill_sr_landscape.png"
fig.savefig(chart2)
print(f"wrote {chart2}")
plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 3: shaped reward ↑ vs ICM intrinsic reward ↓  (the dissociation plot)
# ---------------------------------------------------------------------------
v7_iters = extract_iters(RUNS_ROOT / "ppo_icm_v7" / "train.log")

xs_rew,  ys_rew  = series(v7_iters, "ep_rew_mean")
xs_intr, ys_intr = series(v7_iters, "intr_reward")

if xs_rew and xs_intr:
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 5.2), dpi=140, sharex=True,
        gridspec_kw=dict(height_ratios=[1.0, 1.0], hspace=0.12),
    )

    ax_top.plot(xs_rew, ys_rew, color="tab:orange", linewidth=2,
                label="Shaped extrinsic return  (training signal)")
    ax_top.set_ylabel("Average shaped return")
    ax_top.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="lower right")
    ax_top.set_title("PPO + ICM on BinFill: training signal rises while "
                     "curiosity exhausts (test SR = 0\\%)",
                     fontsize=11)

    ax_bot.plot(xs_intr, ys_intr, color="tab:red", linewidth=2,
                label="ICM intrinsic reward  (forward-prediction error)")
    ax_bot.set_ylabel("Intrinsic reward")
    ax_bot.set_yscale("log")
    ax_bot.grid(True, alpha=0.3, which="both")
    ax_bot.legend(loc="upper right")
    ax_bot.set_xlabel("Environment steps ($\\times 10^6$)")
    ax_bot.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e6:.2f}"))

    fig.tight_layout()
    chart3 = OUT_DIR / "icm_dissociation_binfill.png"
    fig.savefig(chart3)
    print(f"wrote {chart3}")
    plt.close(fig)
else:
    print("skip chart 3: missing intr_reward or ep_rew_mean in ppo_icm_v7")


# ---------------------------------------------------------------------------
# Chart 4: policy contraction — action std and entropy_loss over training
# ---------------------------------------------------------------------------
contraction_runs = [
    ("ppo_v4",     "PPO",       "tab:blue"),
    ("ppo_icm_v7", "PPO + ICM", "tab:orange"),
]

fig, (ax_std, ax_ent) = plt.subplots(
    2, 1, figsize=(8, 5.2), dpi=140, sharex=True,
    gridspec_kw=dict(height_ratios=[1.0, 1.0], hspace=0.12),
)
plotted = 0
for slug, label, color in contraction_runs:
    log = RUNS_ROOT / slug / "train.log"
    if not log.exists():
        print(f"skip {slug}: no train.log")
        continue
    iters = extract_iters(log)
    xs_s, ys_s = series(iters, "std")
    xs_e, ys_e = series(iters, "entropy_loss")
    if xs_s:
        ax_std.plot(xs_s, ys_s, label=label, color=color, linewidth=2)
        plotted += 1
    if xs_e:
        ax_ent.plot(xs_e, ys_e, label=label, color=color, linewidth=2)

if plotted == 0:
    print("skip chart 4: no std/entropy_loss data parseable")
    plt.close(fig)
else:
    ax_std.set_ylabel("Policy action std")
    ax_std.grid(True, alpha=0.3)
    ax_std.legend(loc="upper right")
    ax_std.set_title("Policy contraction on BinFill: slow std + entropy decay "
                     "→ policy is still exploring at 250k steps",
                     fontsize=11)

    ax_ent.set_ylabel("Entropy loss (negative entropy)")
    ax_ent.grid(True, alpha=0.3)
    ax_ent.legend(loc="upper right")
    ax_ent.set_xlabel("Environment steps ($\\times 10^6$)")
    ax_ent.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e6:.2f}"))

    fig.tight_layout()
    chart4 = OUT_DIR / "policy_contraction_binfill.png"
    fig.savefig(chart4)
    print(f"wrote {chart4}")
    plt.close(fig)
