#!/usr/bin/env python3
"""A/B evaluation for old vs new MP2 policy rollouts.

Input format: directory containing the 7 CSV files produced by
`misc/test_policy_openloop.py --compare`:
  openloop_scheduled_forward_0.15.csv
  openloop_scheduled_backward_0.15.csv
  openloop_scheduled_left_0.15.csv
  openloop_scheduled_right_0.15.csv
  openloop_scheduled_yaw_left_0.2.csv
  openloop_scheduled_yaw_right_0.2.csv
  openloop_scheduled_stop.csv

This script compares two directories (old vs new) and prints robust metrics:
- command separability (lower cross-command action correlation is better),
- joint-limit hit rate (lower is better),
- action smoothness (lower jerk is better).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

COMMAND_FILES = [
    "openloop_scheduled_forward_0.15.csv",
    "openloop_scheduled_backward_0.15.csv",
    "openloop_scheduled_left_0.15.csv",
    "openloop_scheduled_right_0.15.csv",
    "openloop_scheduled_yaw_left_0.2.csv",
    "openloop_scheduled_yaw_right_0.2.csv",
    "openloop_scheduled_stop.csv",
]

JOINT_LOWER = np.array([-0.524, 0.0, -2.356] * 4, dtype=np.float64)
JOINT_UPPER = np.array([0.524, 1.396, 0.0] * 4, dtype=np.float64)


def load_suite(run_dir: Path) -> Dict[str, pd.DataFrame]:
    suite: Dict[str, pd.DataFrame] = {}
    for name in COMMAND_FILES:
        p = run_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Missing required CSV: {p}")
        suite[name] = pd.read_csv(p)
    return suite


def settled_actions(df: pd.DataFrame, settle_s: float) -> np.ndarray:
    s = df[df["time"] >= settle_s]
    cols = [f"action_{i}" for i in range(12)]
    return s[cols].to_numpy(dtype=np.float64)


def settled_joint_pos(df: pd.DataFrame, settle_s: float) -> np.ndarray:
    s = df[df["time"] >= settle_s]
    cols = [f"joint_pos_{i}" for i in range(12)]
    return s[cols].to_numpy(dtype=np.float64)


def separability_score(suite: Dict[str, pd.DataFrame], settle_s: float) -> float:
    keys = list(suite.keys())
    flats: List[np.ndarray] = []
    for k in keys:
        a = settled_actions(suite[k], settle_s)
        flats.append(a.reshape(-1))

    corrs = []
    for i in range(len(flats)):
        for j in range(i + 1, len(flats)):
            n = min(flats[i].shape[0], flats[j].shape[0])
            if n < 10:
                continue
            c = np.corrcoef(flats[i][:n], flats[j][:n])[0, 1]
            if np.isnan(c):
                continue
            corrs.append(c)
    if not corrs:
        return 0.0
    # Lower avg corr is better. Convert to [0,1] where 1 is best.
    return float(np.clip(1.0 - np.mean(corrs), 0.0, 1.0))


def limit_hit_rate(suite: Dict[str, pd.DataFrame], settle_s: float) -> float:
    hits = 0
    total = 0
    eps = 1e-3
    for df in suite.values():
        p = settled_joint_pos(df, settle_s)
        low_hit = p <= (JOINT_LOWER + eps)
        high_hit = p >= (JOINT_UPPER - eps)
        hits += int(np.count_nonzero(low_hit | high_hit))
        total += p.size
    return float(hits / max(1, total))


def smoothness_jerk(suite: Dict[str, pd.DataFrame], settle_s: float) -> float:
    jerks = []
    for df in suite.values():
        a = settled_actions(df, settle_s)
        if len(a) < 3:
            continue
        d2 = np.diff(a, n=2, axis=0)
        jerks.append(np.mean(np.abs(d2)))
    if not jerks:
        return float("inf")
    return float(np.mean(jerks))


def evaluate(run_dir: Path, settle_s: float) -> Dict[str, float]:
    suite = load_suite(run_dir)
    sep = separability_score(suite, settle_s)
    hit = limit_hit_rate(suite, settle_s)
    jerk = smoothness_jerk(suite, settle_s)

    # Normalize components into [0,1] heuristically.
    # hit: lower is better, 0.0 perfect. Penalize strongly after 2%.
    hit_score = float(np.clip(1.0 - (hit / 0.02), 0.0, 1.0))
    # jerk: lower better. 0.02 is very smooth, 0.20 rough.
    jerk_score = float(np.clip(1.0 - ((jerk - 0.02) / 0.18), 0.0, 1.0))

    total = 0.45 * sep + 0.35 * hit_score + 0.20 * jerk_score
    return {
        "separability": sep,
        "limit_hit_rate": hit,
        "smoothness_jerk": jerk,
        "hit_score": hit_score,
        "jerk_score": jerk_score,
        "total_score": total,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B compare old vs new policy rollout suites")
    ap.add_argument("--old", type=Path, required=True, help="Dir with old model compare CSVs")
    ap.add_argument("--new", type=Path, required=True, help="Dir with new model compare CSVs")
    ap.add_argument("--settle", type=float, default=2.0, help="Seconds to ignore at start")
    args = ap.parse_args()

    old_m = evaluate(args.old, args.settle)
    new_m = evaluate(args.new, args.settle)

    print("\n=== OLD ===")
    for k, v in old_m.items():
        print(f"{k:16s}: {v:.6f}")

    print("\n=== NEW ===")
    for k, v in new_m.items():
        print(f"{k:16s}: {v:.6f}")

    print("\n=== DELTA (NEW - OLD) ===")
    for k in old_m.keys():
        print(f"{k:16s}: {new_m[k] - old_m[k]:+.6f}")

    if new_m["total_score"] > old_m["total_score"]:
        print("\nVERDICT: NEW policy is better under this rollout benchmark.")
    else:
        print("\nVERDICT: OLD policy is better (or new not yet improved) under this benchmark.")


if __name__ == "__main__":
    main()
