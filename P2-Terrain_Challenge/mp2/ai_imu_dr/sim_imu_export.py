"""
Convert IsaacSim observation logs → MesNet training CSV.

Reconstructs raw accelerometer from sim's projected_gravity + finite-difference
linear acceleration. Ground truth velocity is directly available.

Usage:
    python sim_imu_export.py --sim-dir ~/mlp_obs_action_logs --output data/sim_mlp_imu.csv
    python sim_imu_export.py --sim-dir ~/lstm_obs_action_logs --output data/sim_lstm_imu.csv
"""

import argparse
import csv
import glob
import os
import numpy as np


# Observation indices (60-dim obs from training)
# Order matches CSV header: base_lin_vel, base_ang_vel, projected_gravity, ...
OBS_GROUPS = {
    'base_lin_vel': (0, 3),       # ground truth velocity in body frame (m/s)
    'base_ang_vel': (3, 6),       # gyro in body frame (rad/s)
    'projected_gravity': (6, 9),  # gravity direction in body frame (unit vec)
    'commands': (9, 12),
    'joint_pos': (12, 24),
    'joint_vel': (24, 36),
    'joint_effort': (36, 48),
    'prev_actions': (48, 60),
}

# Sim physics dt (each row = 1 physics step at 50 Hz control)
CONTROL_DT = 0.02  # 50 Hz


def load_sim_envs(sim_dir):
    """Load all env observation CSVs from a sim log directory."""
    obs_files = sorted(glob.glob(os.path.join(sim_dir, 'env_*_observations.csv')))
    all_obs = []
    for f in obs_files:
        data = np.genfromtxt(f, delimiter=',', skip_header=1)
        # Drop the time_step column (index 0)
        obs = data[:, 1:]  # (T, 60)
        all_obs.append(obs)
        print(f"  Loaded {f}: {obs.shape[0]} steps")
    return all_obs


def reconstruct_imu(obs_sequence, dt=CONTROL_DT):
    """Reconstruct raw IMU (gyro + accel) from sim observations.

    Accelerometer model:
        a_measured = linear_accel_body - gravity_body
        where gravity_body = projected_gravity * 9.81

    Linear acceleration is computed via finite differences on velocity.

    Returns:
        timestamps: (T-1,) array
        gyro: (T-1, 3) - angular velocity in body frame (rad/s)
        accel: (T-1, 3) - reconstructed accelerometer reading (m/s^2)
        vel_gt: (T-1, 3) - ground truth velocity in body frame (m/s)
    """
    T = obs_sequence.shape[0]

    # Extract fields
    ang_vel = obs_sequence[:, OBS_GROUPS['base_ang_vel'][0]:OBS_GROUPS['base_ang_vel'][1]]
    gravity = obs_sequence[:, OBS_GROUPS['projected_gravity'][0]:OBS_GROUPS['projected_gravity'][1]]
    lin_vel = obs_sequence[:, OBS_GROUPS['base_lin_vel'][0]:OBS_GROUPS['base_lin_vel'][1]]

    # Finite-difference linear acceleration (body frame)
    lin_accel = np.diff(lin_vel, axis=0) / dt  # (T-1, 3)

    # Reconstructed accelerometer in HW IMU convention.
    # HW IMU reports az ≈ -9.81 when level (gravity in sensor frame).
    # Sim projected_gravity points [0,0,-1] when level.
    # HW convention: accel_measured = gravity_body * 9.81 + lin_accel
    # → level+stationary: [0,0,-1]*9.81 + 0 = [0,0,-9.81] ✓ matches HW
    gravity_ms2 = gravity[:-1] * 9.81  # (T-1, 3), e.g. [0, 0, -9.81] when level
    accel_reading = gravity_ms2 + lin_accel  # (T-1, 3)

    timestamps = np.arange(T - 1) * dt
    gyro = ang_vel[:-1]  # (T-1, 3)
    vel_gt = lin_vel[:-1]  # (T-1, 3) - use current step as GT

    return timestamps, gyro, accel_reading, vel_gt


def export_csv(output_path, timestamps, gyro, accel, vel_gt):
    """Write MesNet-compatible training CSV."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fieldnames = ['timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az',
                  'vx_gt', 'vy_gt', 'vz_gt']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(timestamps)):
            writer.writerow({
                'timestamp': f'{timestamps[i]:.6f}',
                'gx': f'{gyro[i, 0]:.6f}',
                'gy': f'{gyro[i, 1]:.6f}',
                'gz': f'{gyro[i, 2]:.6f}',
                'ax': f'{accel[i, 0]:.6f}',
                'ay': f'{accel[i, 1]:.6f}',
                'az': f'{accel[i, 2]:.6f}',
                'vx_gt': f'{vel_gt[i, 0]:.6f}',
                'vy_gt': f'{vel_gt[i, 1]:.6f}',
                'vz_gt': f'{vel_gt[i, 2]:.6f}',
            })


def main():
    parser = argparse.ArgumentParser(description='Convert sim obs logs to MesNet training CSV')
    parser.add_argument('--sim-dir', required=True, help='Directory with env_*_observations.csv')
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--dt', type=float, default=CONTROL_DT, help='Control timestep')
    args = parser.parse_args()

    print(f"[SIM→IMU] Loading from {args.sim_dir}")
    env_obs = load_sim_envs(args.sim_dir)

    # Concatenate all environments
    all_ts, all_gyro, all_accel, all_vel = [], [], [], []
    time_offset = 0.0

    for i, obs in enumerate(env_obs):
        ts, gyro, accel, vel_gt = reconstruct_imu(obs, args.dt)
        all_ts.append(ts + time_offset)
        all_gyro.append(gyro)
        all_accel.append(accel)
        all_vel.append(vel_gt)
        time_offset = all_ts[-1][-1] + args.dt
        print(f"  Env {i}: {len(ts)} samples, vel range [{vel_gt.min():.3f}, {vel_gt.max():.3f}]")

    timestamps = np.concatenate(all_ts)
    gyro = np.concatenate(all_gyro)
    accel = np.concatenate(all_accel)
    vel_gt = np.concatenate(all_vel)

    print(f"\n[STATS] Total: {len(timestamps)} samples")
    print(f"  Gyro: mean={gyro.mean(axis=0)}, std={gyro.std(axis=0)}")
    print(f"  Accel: mean={accel.mean(axis=0)}, std={accel.std(axis=0)}")
    print(f"  Vel GT: mean={vel_gt.mean(axis=0)}, std={vel_gt.std(axis=0)}")

    export_csv(args.output, timestamps, gyro, accel, vel_gt)
    print(f"\n[SAVED] {args.output}")


if __name__ == '__main__':
    main()
