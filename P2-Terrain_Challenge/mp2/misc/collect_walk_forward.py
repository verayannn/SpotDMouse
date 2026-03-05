#!/usr/bin/env python3
"""
Collect real robot observations while walking forward with the RL policy.
Run this ON THE ROBOT. Saves a CSV with 60-dim obs + 12-dim actions at 50Hz.

Usage:
    python3 collect_walk_forward.py --policy /path/to/policy.pt --vx 0.15 --duration 10
"""

import time
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime

try:
    from MangDang.mini_pupper.HardwareInterface import HardwareInterface
    from MangDang.mini_pupper.Config import Configuration
except ImportError:
    print("ERROR: MangDang library not found. This script must run on the robot.")
    exit(1)


# ─── Constants ────────────────────────────────────────────────────────────────

SIM_DEFAULTS = np.array([
    0.0, 0.785, -1.57,  # LF
    0.0, 0.785, -1.57,  # RF
    0.0, 0.785, -1.57,  # LB
    0.0, 0.785, -1.57,  # RB
], dtype=np.float32)

REAL_DEFAULTS = np.array([0.0, 0.785, -0.785] * 4, dtype=np.float32)
CALF_OFFSET = 0.785  # sim calf (-1.57) vs real calf (-0.785)

HW_TO_ISAAC = {0: 1, 1: 0, 2: 3, 3: 2}  # hw_col -> isaac_leg

EFFORT_DIRECTION = np.array([
    +1.0, +1.0, +1.0,  # LF
    -1.0, -1.0, -1.0,  # RF
    +1.0, +1.0, +1.0,  # LB
    -1.0, -1.0, -1.0,  # RB
], dtype=np.float32)

LOAD_SCALE = 5000.0
GYRO_SCALE = np.pi / 180.0
ACCEL_SCALE = 9.81

JOINT_NAMES = [
    'LF_hip', 'LF_thigh', 'LF_calf',
    'RF_hip', 'RF_thigh', 'RF_calf',
    'LB_hip', 'LB_thigh', 'LB_calf',
    'RB_hip', 'RB_thigh', 'RB_calf',
]


class WalkRecorder:
    def __init__(self):
        self.config = Configuration()
        self.hw = HardwareInterface()
        self.pwm_params = self.hw.pwm_params
        self.servo_params = self.hw.servo_params
        self.esp32 = self.pwm_params.esp32
        time.sleep(0.3)

        self.prev_actions = np.zeros(12, dtype=np.float32)
        self.prev_joint_angles = SIM_DEFAULTS.copy()
        self.prev_time = time.time()
        self.gyro_offset = np.zeros(3)
        self.load_offset = np.zeros(12)

    def calibrate(self, samples=50):
        """Calibrate gyro and load offsets at rest."""
        print("Calibrating (keep robot still)...")
        gyro_buf, load_buf = [], []
        for _ in range(samples):
            imu = self.esp32.imu_get_data()
            if imu:
                gyro_buf.append([imu['gx'], imu['gy'], imu['gz']])
            load = self.esp32.servos_get_load()
            if load:
                load_buf.append(load)
            time.sleep(0.02)
        if gyro_buf:
            self.gyro_offset = np.mean(gyro_buf, axis=0)
        if load_buf:
            self.load_offset = np.mean(load_buf, axis=0)
        print(f"  gyro_offset: {self.gyro_offset}")
        print(f"  load_offset mean: {np.mean(np.abs(self.load_offset)):.1f}")

    def read_joint_positions(self):
        """Read servo positions, return in Isaac order (12,)."""
        raw = self.esp32.servos_get_position()
        if raw is None:
            return self.prev_joint_angles.copy()

        angles_hw = np.zeros((3, 4))
        for leg in range(4):
            for axis in range(3):
                sid = self.pwm_params.servo_ids[axis, leg]
                pos = raw[sid - 1]
                dev = (self.servo_params.neutral_position - pos) / self.servo_params.micros_per_rad
                angles_hw[axis, leg] = dev / self.servo_params.servo_multipliers[axis, leg] + self.servo_params.neutral_angles[axis, leg]

        angles = np.zeros(12, dtype=np.float32)
        for hw_col, isaac_leg in HW_TO_ISAAC.items():
            for axis in range(3):
                angles[isaac_leg * 3 + axis] = angles_hw[axis, hw_col]
        return angles

    def read_joint_efforts(self):
        """Read servo load, return normalized in Isaac order (12,)."""
        raw = self.esp32.servos_get_load()
        if raw is None:
            return np.zeros(12, dtype=np.float32)

        raw = np.array(raw, dtype=np.float32)
        effort = np.zeros(12, dtype=np.float32)
        offset = np.zeros(12, dtype=np.float32)
        for hw_col, isaac_leg in HW_TO_ISAAC.items():
            for axis in range(3):
                sid = self.pwm_params.servo_ids[axis, hw_col]
                effort[isaac_leg * 3 + axis] = raw[sid - 1]
                offset[isaac_leg * 3 + axis] = self.load_offset[sid - 1]
        return (effort - offset) / LOAD_SCALE * EFFORT_DIRECTION

    def build_observation(self, velocity_cmd):
        """
        Build 60-dim observation matching the sim's SpotObservationsCfg:
          [0:3]   base_lin_vel
          [3:6]   base_ang_vel
          [6:9]   projected_gravity
          [9:12]  velocity_cmd
          [12:24] joint_pos_rel
          [24:36] joint_vel
          [36:48] joint_effort
          [48:60] prev_actions
        """
        now = time.time()
        dt = now - self.prev_time

        imu = self.esp32.imu_get_data()
        angles = self.read_joint_positions()
        effort = self.read_joint_efforts()

        # Base velocities
        base_lin_vel = velocity_cmd * 0.7  # rough proxy (no odometry)

        gyro = np.array([imu['gx'], imu['gy'], imu['gz']], dtype=np.float32)
        base_ang_vel = (gyro - self.gyro_offset) * GYRO_SCALE

        # Gravity from accelerometer
        accel = np.array([imu['ax'], imu['ay'], imu['az']], dtype=np.float32) * ACCEL_SCALE
        norm = np.linalg.norm(accel)
        gravity = accel / norm if norm > 0.1 else np.array([0, 0, -1], dtype=np.float32)

        # Joint states
        joint_pos_rel = angles - SIM_DEFAULTS
        joint_vel = (angles - self.prev_joint_angles) / dt if dt > 0.001 else np.zeros(12, dtype=np.float32)

        self.prev_joint_angles = angles.copy()
        self.prev_time = now

        obs = np.concatenate([
            base_lin_vel,    # [0:3]
            base_ang_vel,    # [3:6]
            gravity,         # [6:9]
            velocity_cmd,    # [9:12]
            joint_pos_rel,   # [12:24]
            joint_vel,       # [24:36]
            effort,          # [36:48]
            self.prev_actions,  # [48:60]
        ]).astype(np.float32)

        return obs, angles, effort

    def isaac_to_hw_matrix(self, angles):
        """Convert Isaac-order flat array to hardware (3,4) matrix."""
        m = np.zeros((3, 4))
        m[:, 1] = angles[0:3]   # LF
        m[:, 0] = angles[3:6]   # RF
        m[:, 3] = angles[6:9]   # LB
        m[:, 2] = angles[9:12]  # RB
        return m

    def record(self, policy_path, velocity_cmd, duration=10.0, action_scale=0.5, fade_steps=50):
        """
        Walk forward with policy, recording all observations and actions.
        """
        print(f"\nLoading policy: {policy_path}")
        policy = torch.jit.load(policy_path, map_location='cpu')
        policy.eval()

        cmd = np.array(velocity_cmd, dtype=np.float32)
        print(f"Command: {cmd.tolist()}")
        print(f"Duration: {duration}s, action_scale: {action_scale}")

        # Move to default pose first
        print("Moving to default pose...")
        target_real = REAL_DEFAULTS.copy()
        self.hw.set_actuator_postions(self.isaac_to_hw_matrix(
            np.array([0.0, 0.785, -0.785] * 4)  # real defaults
        ))
        time.sleep(2.0)

        self.calibrate()

        print("\nStarting walk in 3 seconds...")
        time.sleep(3.0)

        self.prev_actions = np.zeros(12, dtype=np.float32)
        self.prev_joint_angles = self.read_joint_positions()
        self.prev_time = time.time()

        rows = []
        start = time.time()
        step = 0

        try:
            while time.time() - start < duration:
                loop_start = time.time()

                obs, raw_angles, effort = self.build_observation(cmd)

                with torch.no_grad():
                    obs_t = torch.tensor(obs).unsqueeze(0)
                    raw_actions = policy(obs_t).squeeze().numpy()

                # Fade in
                fade = min(step / float(fade_steps), 1.0)
                actions = raw_actions * fade

                self.prev_actions = actions.copy()

                # Convert to real servo targets
                target_sim = SIM_DEFAULTS + actions * action_scale
                target_real = target_sim.copy()
                target_real[2::3] += CALF_OFFSET  # sim→real calf offset

                self.hw.set_actuator_postions(self.isaac_to_hw_matrix(target_real))

                # Record everything
                row = {
                    'step': step,
                    'time': time.time() - start,
                    'fade': fade,
                }
                # Full 60-dim observation
                for i in range(60):
                    row[f'obs_{i}'] = obs[i]
                # Raw and faded actions
                for i in range(12):
                    row[f'raw_action_{i}'] = raw_actions[i]
                    row[f'action_{i}'] = actions[i]
                    row[f'joint_pos_{i}'] = raw_angles[i]
                    row[f'joint_pos_rel_{i}'] = obs[12 + i]
                    row[f'joint_vel_{i}'] = obs[24 + i]
                    row[f'effort_{i}'] = obs[36 + i]
                rows.append(row)

                step += 1

                # Maintain 50Hz
                elapsed = time.time() - loop_start
                if elapsed < 0.02:
                    time.sleep(0.02 - elapsed)

        except KeyboardInterrupt:
            print("\nInterrupted!")

        # Return to default
        print("Returning to default pose...")
        self.hw.set_actuator_postions(self.isaac_to_hw_matrix(
            np.array([0.0, 0.785, -0.785] * 4)
        ))

        df = pd.DataFrame(rows)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vx_str = f"{cmd[0]:.2f}".replace('.', 'p')
        vy_str = f"{cmd[1]:.2f}".replace('.', 'p')
        fname = f"real_walk_vx{vx_str}_vy{vy_str}_{timestamp}.csv"
        df.to_csv(fname, index=False)
        print(f"\nRecorded {step} steps ({step * 0.02:.1f}s) → {fname}")
        return fname


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record real robot walking with RL policy")
    parser.add_argument("--policy", type=str, default="/home/ubuntu/mp2_mlp/policy_joyboy_delayedpdactuator.pt")
    parser.add_argument("--vx", type=float, default=0.15)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--vyaw", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--scale", type=float, default=0.5)
    args = parser.parse_args()

    recorder = WalkRecorder()
    recorder.record(
        policy_path=args.policy,
        velocity_cmd=[args.vx, args.vy, args.vyaw],
        duration=args.duration,
        action_scale=args.scale,
    )
