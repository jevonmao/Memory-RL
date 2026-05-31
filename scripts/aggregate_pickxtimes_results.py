"""Aggregate eval JSONs for PickXtimes across baselines + seeds.

Reads each runs/cluster/eval_results/ppo_b{1,4,5,6}_*_PickXtimes_s*.json
and prints a markdown table with mean ± std SR per baseline.

Usage:
    python -m scripts.aggregate_pickxtimes_results
    python -m scripts.aggregate_pickxtimes_results --root runs/cluster --out runs/cluster/eval_results/PICKXTIMES_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np


BASELINE_LABELS = {
    "ppo_b1":             "PPO (control, v1 reward)",
    "ppo_b4_rnd":         "PPO + RND (v1 reward)",
    "ppo_b5_recurrent":   "RecurrentPPO (v1 reward)",
    "ppo_b6_recurrent_rnd": "RecurrentPPO + RND (v1 reward, headline)",
    "ppo_b6_recurrent_rnd_bc": "RecurrentPPO + RND + BC (v1 reward)",
    # v2 reward variants live under runs/cluster/v2/<same-prefix> — same JSON names
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs/cluster",
                   help="Cluster runs root containing eval_results/.")
    p.add_argument("--out",  default=None,
                   help="Optional markdown output file.")
    return p.parse_args()


def load_evals(root: Path) -> Dict[str, List[dict]]:
    """Return {prefix: [json blob, ...]} grouped by baseline prefix.

    Eval json files live at <root>/eval_results/<prefix>_<task>_s<seed>.json,
    OR as siblings of training ckpts at <root>/<prefix>_<task>_s<seed>/ckpts/*eval*.json.
    Searches both layouts.
    """
    groups: Dict[str, List[dict]] = {p: [] for p in BASELINE_LABELS}

    # Layout A: <root>/eval_results/<prefix>_<task>_s<seed>.json (aggregated dir)
    eval_dir = root / "eval_results"
    if eval_dir.is_dir():
        for f in sorted(eval_dir.glob("*PickXtimes_s*.json")):
            m = re.match(r"^(.+)_PickXtimes_s(\d+)\.json$", f.name)
            if not m:
                continue
            prefix, seed = m.group(1), int(m.group(2))
            with open(f) as fh:
                blob = json.load(fh)
            blob["__seed"] = seed
            blob["__file"] = str(f)
            if prefix in groups:
                groups[prefix].append(blob)

    # Layout B: <root>/<run-name>/ckpts/<ckpt-name>.eval.json (single-ckpt evals)
    for run_dir in sorted(root.glob("ppo_b*_PickXtimes_s*")):
        m = re.match(r"^(.+)_PickXtimes_s(\d+)$", run_dir.name)
        if not m:
            continue
        prefix, seed = m.group(1), int(m.group(2))
        if prefix not in groups:
            continue
        ckpt_dir = run_dir / "ckpts"
        if not ckpt_dir.is_dir():
            continue
        evals = sorted(ckpt_dir.glob("*.eval.json"))
        if not evals:
            continue
        latest_eval = evals[-1]
        with open(latest_eval) as fh:
            blob = json.load(fh)
        blob["__seed"] = seed
        blob["__file"] = str(latest_eval)
        # Avoid duplicating if Layout A already grabbed this seed.
        if not any(e["__seed"] == seed for e in groups[prefix]):
            groups[prefix].append(blob)

    return groups


def summarise(groups: Dict[str, List[dict]]) -> str:
    lines = ["# PickXtimes — RL baseline comparison",
             "",
             "| Baseline | seeds | SR (mean ± std) | mean return | best seed SR |",
             "|---|---|---|---|---|"]
    for prefix in BASELINE_LABELS:
        entries = groups.get(prefix, [])
        if not entries:
            lines.append(f"| {BASELINE_LABELS[prefix]} | 0 | — | — | — |")
            continue
        srs   = [e["aggregate_sr"] for e in entries]
        rets  = [np.mean([r["mean_return"] for r in e["results"]]) for e in entries]
        seeds = [e["__seed"] for e in entries]
        best  = max(zip(srs, seeds))
        lines.append(
            f"| {BASELINE_LABELS[prefix]} | {len(srs)} ({','.join(map(str, sorted(seeds)))}) "
            f"| {np.mean(srs)*100:.1f} ± {np.std(srs)*100:.1f}% "
            f"| {np.mean(rets):.2f} "
            f"| {best[0]*100:.0f}% (s{best[1]}) |"
        )

    lines.append("")
    lines.append("## Per-seed detail")
    lines.append("")
    for prefix, entries in groups.items():
        if not entries:
            continue
        lines.append(f"### {BASELINE_LABELS[prefix]}")
        lines.append("| seed | SR | mean return |")
        lines.append("|---|---|---|")
        for e in sorted(entries, key=lambda x: x["__seed"]):
            r = e["results"][0]
            lines.append(f"| {e['__seed']} | {r['success_rate']*100:.1f}% | {r['mean_return']:.2f} |")
        lines.append("")

    return "\n".join(lines)


def main():
    args = parse_args()
    root = Path(args.root)
    groups = load_evals(root)
    report = summarise(groups)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write(report)
        print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
