"""
Convert hardware CSV logs (from mlp_controller_v3.py) → MesNet training CSV.

Hardware logs contain raw IMU columns (imu_gx..imu_az) since the latest update.
Uses self-supervised mode: detect stationary periods from low joint velocity.

Usage:
    python hw_log_to_mesnet.py --hw-log ~/mp2_mlp/mlp_pd_cf_hw_log_*.csv --output data/hw_mlp_imu.csv
    python hw_log_to_mesnet.py --hw-log ~/mp2_mlp/lstm_nopd_cf_hw_log_*.csv --output data/hw_lstm_imu.csv
"""

import argparse
import csv
import glob
import os
import numpy as np


def convert_hw_log(hw_log_path, output_path):
    """Convert a single hardware log CSV to MesNet format."""

    with open(hw_log_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"  [SKIP] Empty: {hw_log_path}")
        return None

    # Check for IMU columns
    has_imu = 'imu_gx' in rows[0]

    if has_imu:
        # New format: raw IMU columns present
        print(f"  [OK] {hw_log_path}: {len(rows)} steps (has raw IMU)")
        output_rows = []
        for r in rows:
            output_rows.append({
                'timestamp': float(r['time']),
                'gx': float(r['imu_gx']),
                'gy': float(r['imu_gy']),
                'gz': float(r['imu_gz']),
                'ax': float(r['imu_ax']),
                'ay': float(r['imu_ay']),
                'az': float(r['imu_az']),
            })
        return output_rows
    else:
        # Old format: reconstruct from observations
        # obs_3:5 = base_ang_vel (gyro), obs_6:8 = projected_gravity
        print(f"  [OK] {hw_log_path}: {len(rows)} steps (no raw IMU, reconstructing)")
        output_rows = []
        for r in rows:
            # Gyro from obs (already in rad/s)
            gx = float(r['obs_3'])
            gy = float(r['obs_4'])
            gz = float(r['obs_5'])

            # Reconstruct accel from gravity obs (obs_6:8 = projected_gravity, unit vector)
            # When comp_filter was not active, gravity is hardcoded [0,0,-1]
            grav_x = float(r['obs_6'])
            grav_y = float(r['obs_7'])
            grav_z = float(r['obs_8'])

            # Accelerometer ≈ -gravity * 9.81 when stationary
            ax = -grav_x * 9.81
            ay = -grav_y * 9.81
            az = -grav_z * 9.81

            output_rows.append({
                'timestamp': float(r['time']),
                'gx': gx, 'gy': gy, 'gz': gz,
                'ax': ax, 'ay': ay, 'az': az,
            })
        return output_rows


def detect_stationary(rows, joint_vel_threshold=0.5):
    """Add contact columns based on low joint velocity (proxy for stationary).

    For self-supervised MesNet training: stationary = all joints nearly still.
    """
    for r in rows:
        # We don't have joint vel in the MesNet CSV, so this is handled
        # by train_mesnet.py's low-acceleration proxy.
        pass
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hw-log', required=True, nargs='+', help='Hardware log CSV(s)')
    parser.add_argument('--output', required=True, help='Output MesNet CSV')
    args = parser.parse_args()

    # Expand globs
    files = []
    for pattern in args.hw_log:
        files.extend(glob.glob(pattern))
    files.sort()

    if not files:
        print(f"[ERROR] No files matched: {args.hw_log}")
        return

    all_rows = []
    for f in files:
        rows = convert_hw_log(f, args.output)
        if rows:
            all_rows.extend(rows)

    if not all_rows:
        print("[ERROR] No data extracted")
        return

    # Normalize timestamps to start at 0
    t0 = all_rows[0]['timestamp']
    for r in all_rows:
        r['timestamp'] -= t0

    # Write output
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fieldnames = ['timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az']
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: f'{v:.6f}' for k, v in r.items()})

    # Stats
    gyro = np.array([[r['gx'], r['gy'], r['gz']] for r in all_rows])
    accel = np.array([[r['ax'], r['ay'], r['az']] for r in all_rows])

    print(f"\n[STATS] Total: {len(all_rows)} samples")
    print(f"  Gyro: mean={gyro.mean(axis=0)}, std={gyro.std(axis=0)}")
    print(f"  Accel: mean={accel.mean(axis=0)}, std={accel.std(axis=0)}")
    print(f"\n[SAVED] {args.output}")
    print(f"\nTo train (self-supervised, no ground truth velocity):")
    print(f"  python train_mesnet.py --data {args.output} --mode self-supervised --output mesnet_hw.pt")


if __name__ == '__main__':
    main()
