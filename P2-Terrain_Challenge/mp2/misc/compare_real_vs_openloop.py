#!/usr/bin/env python3
"""
Compare real robot walk CSV against open-loop policy rollout.
Run this on any machine (no robot needed).

Usage:
    python3 compare_real_vs_openloop.py --real real_walk_vx0p15_vy0p00_*.csv
    python3 compare_real_vs_openloop.py --real real_walk.csv --openloop openloop.csv

If --openloop is not provided, generates one automatically from the policy.
"""

import argparse
import numpy as np
import pandas as pd
import sys
import os

JOINT_NAMES = [
    'LF_hip', 'LF_thigh', 'LF_calf',
    'RF_hip', 'RF_thigh', 'RF_calf',
    'LB_hip', 'LB_thigh', 'LB_calf',
    'RB_hip', 'RB_thigh', 'RB_calf',
]

OBS_LABELS = {
    (0, 3):   'base_lin_vel',
    (3, 6):   'base_ang_vel',
    (6, 9):   'projected_gravity',
    (9, 12):  'velocity_cmd',
    (12, 24): 'joint_pos_rel',
    (24, 36): 'joint_vel',
    (36, 48): 'joint_effort',
    (48, 60): 'prev_actions',
}


def estimate_gait_freq(signal, dt=0.02):
    """Estimate dominant frequency from zero crossings."""
    centered = signal - signal.mean()
    crossings = np.where(np.diff(np.sign(centered)))[0]
    if len(crossings) > 4:
        avg_half = np.mean(np.diff(crossings)) * dt
        return 1.0 / (2 * avg_half)
    return float('nan')


def obs_stats(df, label, settle_time=2.0):
    """Extract observation statistics from a dataframe."""
    settled = df[df['time'] >= settle_time]
    stats = {}

    # Per-obs-dim stats
    for i in range(60):
        col = f'obs_{i}'
        if col in settled.columns:
            vals = settled[col].values
            stats[f'obs_{i}'] = {
                'mean': vals.mean(), 'std': vals.std(),
                'min': vals.min(), 'max': vals.max(),
            }

    # Per-joint stats
    for j in range(12):
        # Position relative
        pos_col = f'joint_pos_rel_{j}'
        if pos_col in settled.columns:
            pos = settled[pos_col].values
        else:
            pos = settled[f'obs_{12+j}'].values

        # Action
        act_col = f'action_{j}'
        act = settled[act_col].values if act_col in settled.columns else None

        stats[f'joint_{j}'] = {
            'pos_min': pos.min(), 'pos_max': pos.max(),
            'pos_mean': pos.mean(), 'pos_std': pos.std(),
            'pos_range': pos.max() - pos.min(),
            'freq': estimate_gait_freq(pos),
        }
        if act is not None:
            stats[f'joint_{j}']['act_min'] = act.min()
            stats[f'joint_{j}']['act_max'] = act.max()
            stats[f'joint_{j}']['act_std'] = act.std()

    return stats


def print_comparison(real_stats, openloop_stats):
    """Side-by-side comparison."""
    print("\n" + "=" * 90)
    print("JOINT POSITION COMPARISON (degrees, after 2s settle)")
    print("=" * 90)
    print(f"{'Joint':12s} | {'REAL ROBOT':^38s} | {'OPEN-LOOP MODEL':^38s}")
    print(f"{'':12s} | {'Min':>7s} {'Max':>7s} {'Range':>7s} {'Freq':>6s} | {'Min':>7s} {'Max':>7s} {'Range':>7s} {'Freq':>6s}")
    print("-" * 90)

    for j in range(12):
        r = real_stats[f'joint_{j}']
        o = openloop_stats[f'joint_{j}']
        print(f"{JOINT_NAMES[j]:12s} | "
              f"{np.degrees(r['pos_min']):+7.2f} {np.degrees(r['pos_max']):+7.2f} "
              f"{np.degrees(r['pos_range']):7.2f} {r['freq']:6.2f} | "
              f"{np.degrees(o['pos_min']):+7.2f} {np.degrees(o['pos_max']):+7.2f} "
              f"{np.degrees(o['pos_range']):7.2f} {o['freq']:6.2f}")

    # Actions
    print("\n" + "=" * 90)
    print("ACTION COMPARISON (raw policy output)")
    print("=" * 90)
    print(f"{'Joint':12s} | {'REAL ROBOT':^26s} | {'OPEN-LOOP':^26s}")
    print(f"{'':12s} | {'Min':>8s} {'Max':>8s} {'Std':>8s} | {'Min':>8s} {'Max':>8s} {'Std':>8s}")
    print("-" * 90)

    for j in range(12):
        r = real_stats[f'joint_{j}']
        o = openloop_stats[f'joint_{j}']
        if 'act_min' in r and 'act_min' in o:
            print(f"{JOINT_NAMES[j]:12s} | "
                  f"{r['act_min']:+8.4f} {r['act_max']:+8.4f} {r['act_std']:8.4f} | "
                  f"{o['act_min']:+8.4f} {o['act_max']:+8.4f} {o['act_std']:8.4f}")

    # Observation channel comparison
    print("\n" + "=" * 90)
    print("OBSERVATION CHANNEL COMPARISON (mean ± std)")
    print("=" * 90)

    for (start, end), label in OBS_LABELS.items():
        print(f"\n  {label} [obs_{start}:{end}]:")
        for i in range(start, end):
            idx = i - start
            sub_label = ""
            if label == 'joint_pos_rel' or label == 'joint_vel' or label == 'joint_effort' or label == 'prev_actions':
                sub_label = f" ({JOINT_NAMES[idx]})" if idx < 12 else ""
            elif label == 'base_lin_vel' or label == 'velocity_cmd':
                sub_label = f" ({'xyz'[idx]})"
            elif label == 'base_ang_vel':
                sub_label = f" ({'xyz'[idx]})"
            elif label == 'projected_gravity':
                sub_label = f" ({'xyz'[idx]})"

            r = real_stats.get(f'obs_{i}', {})
            o = openloop_stats.get(f'obs_{i}', {})
            if r and o:
                print(f"    [{i:2d}]{sub_label:16s}: "
                      f"real={r['mean']:+8.4f}±{r['std']:.4f}  "
                      f"openloop={o['mean']:+8.4f}±{o['std']:.4f}  "
                      f"delta_mean={r['mean']-o['mean']:+8.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=str, required=True, help="CSV from real robot walk")
    parser.add_argument("--openloop", type=str, default=None, help="CSV from open-loop rollout (auto-generated if not provided)")
    parser.add_argument("--settle", type=float, default=2.0, help="Settle time to skip (seconds)")
    args = parser.parse_args()

    # Load real data
    print(f"Loading real data: {args.real}")
    real_df = pd.read_csv(args.real)
    print(f"  {len(real_df)} samples, {real_df['time'].max():.1f}s")

    # Load or generate open-loop data
    if args.openloop:
        print(f"Loading open-loop data: {args.openloop}")
        ol_df = pd.read_csv(args.openloop)
    else:
        # Try to find a matching open-loop CSV in the same directory
        ol_path = os.path.join(os.path.dirname(args.real) or '.', 'openloop_cmd_0.15_0.0_0.0.csv')
        if os.path.exists(ol_path):
            print(f"Loading open-loop data: {ol_path}")
            ol_df = pd.read_csv(ol_path)
        else:
            print("No open-loop CSV found. Run test_policy_openloop.py first or pass --openloop.")
            sys.exit(1)
    print(f"  {len(ol_df)} samples, {ol_df['time'].max():.1f}s")

    # Compute stats
    real_stats = obs_stats(real_df, "real", settle_time=args.settle)
    ol_stats = obs_stats(ol_df, "openloop", settle_time=args.settle)

    print_comparison(real_stats, ol_stats)

    # Summary verdict
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Compute overall observation divergence
    real_settled = real_df[real_df['time'] >= args.settle]
    ol_settled = ol_df[ol_df['time'] >= args.settle]

    obs_cols = [f'obs_{i}' for i in range(60)]
    real_mean = real_settled[obs_cols].mean().values
    ol_mean = ol_settled[obs_cols].mean().values
    real_std = real_settled[obs_cols].std().values
    ol_std = ol_settled[obs_cols].std().values

    # Normalized divergence per channel
    combined_std = np.sqrt(real_std**2 + ol_std**2)
    combined_std[combined_std < 1e-6] = 1.0  # avoid division by zero
    norm_diff = np.abs(real_mean - ol_mean) / combined_std

    print(f"\nMean normalized divergence (|delta_mean| / combined_std):")
    for (start, end), label in OBS_LABELS.items():
        channel_div = norm_diff[start:end].mean()
        bar = "#" * int(min(channel_div * 10, 40))
        print(f"  {label:20s}: {channel_div:.3f}  {bar}")

    print(f"\n  Overall: {norm_diff.mean():.3f}")
    if norm_diff.mean() < 0.5:
        print("  → Open-loop model is a REASONABLE proxy for real observations")
    elif norm_diff.mean() < 1.5:
        print("  → MODERATE divergence — open-loop model captures trends but magnitudes differ")
    else:
        print("  → HIGH divergence — open-loop dynamics model needs improvement")

    # Gait frequency match
    print(f"\nGait frequency (LF_thigh):")
    r_freq = real_stats['joint_1']['freq']
    o_freq = openloop_stats['joint_1']['freq'] if 'joint_1' in ol_stats else float('nan')
    print(f"  Real: {r_freq:.2f} Hz")
    print(f"  Open-loop: {o_freq:.2f} Hz")
    if not np.isnan(r_freq) and not np.isnan(o_freq):
        print(f"  Ratio: {r_freq / o_freq:.2f}x")


if __name__ == "__main__":
    main()
