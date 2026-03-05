#!/usr/bin/env python3
"""
Open-loop policy rollout — runs the trained policy WITHOUT a simulator or robot.
Uses a simple first-order joint dynamics model to feed observations back.

Purpose: Verify the policy produces a reasonable forward gait at cmd=[0.15, 0, 0],
then compare against the rosbag CHAMP data.
"""

import torch
import numpy as np
import pandas as pd
import argparse

# ─── Config ───────────────────────────────────────────────────────────────────
POLICY_PATH = "/Users/javierweddington/misc/policy_joyboy_delayedpdactuator.pt"

# Sim defaults (Isaac frame)
SIM_DEFAULTS = np.array([
    0.0, 0.785, -1.57,  # LF
    0.0, 0.785, -1.57,  # RF
    0.0, 0.785, -1.57,  # LB
    0.0, 0.785, -1.57,  # RB
], dtype=np.float32)

# URDF joint limits (from your URDF / IsaacLab config)
JOINT_LOWER = np.array([
    -0.524, 0.0, -2.356,   # LF: hip, thigh, calf
    -0.524, 0.0, -2.356,   # RF
    -0.524, 0.0, -2.356,   # LB
    -0.524, 0.0, -2.356,   # RB
], dtype=np.float32)

JOINT_UPPER = np.array([
    0.524, 1.396, 0.0,     # LF
    0.524, 1.396, 0.0,     # RF
    0.524, 1.396, 0.0,     # LB
    0.524, 1.396, 0.0,     # RB
], dtype=np.float32)

JOINT_NAMES = [
    'LF_hip', 'LF_thigh', 'LF_calf',
    'RF_hip', 'RF_thigh', 'RF_calf',
    'LB_hip', 'LB_thigh', 'LB_calf',
    'RB_hip', 'RB_thigh', 'RB_calf',
]

# Actuator params (from your DelayedPDActuatorCfg)
STIFFNESS = 70.0
DAMPING = 1.2
ACTION_SCALE = 0.5  # from SpotActionsCfg


def run_openloop(cmd_vel, duration=10.0, dt=0.02, policy_path=POLICY_PATH):
    """
    Run the policy in open loop with a simple PD joint model.

    Observation layout (60 dims):
      [0:3]   base_lin_vel
      [3:6]   base_ang_vel
      [6:9]   projected_gravity
      [9:12]  velocity_cmd
      [12:24] joint_pos_rel
      [24:36] joint_vel
      [36:48] joint_effort
      [48:60] prev_actions
    """
    print(f"Loading policy: {policy_path}")
    policy = torch.jit.load(policy_path, map_location='cpu')
    policy.eval()

    cmd = np.array(cmd_vel, dtype=np.float32)
    n_steps = int(duration / dt)

    # State
    joint_pos = SIM_DEFAULTS.copy()
    joint_vel = np.zeros(12, dtype=np.float32)
    prev_actions = np.zeros(12, dtype=np.float32)

    rows = []

    print(f"Running {n_steps} steps ({duration}s) at cmd={cmd.tolist()}")
    print(f"Action scale={ACTION_SCALE}, stiffness={STIFFNESS}, damping={DAMPING}")

    for step in range(n_steps):
        t = step * dt

        # Build observation
        joint_pos_rel = joint_pos - SIM_DEFAULTS

        # Simple effort model: tau = Kp * (target - pos) - Kd * vel
        # where target = default + prev_action * scale
        target = SIM_DEFAULTS + prev_actions * ACTION_SCALE
        effort = STIFFNESS * (target - joint_pos) - DAMPING * joint_vel
        # Normalize effort roughly (sim effort is in N·m, servos are ~0-5 N·m range)
        effort_normalized = effort / 5.0

        obs = np.concatenate([
            cmd * 0.7,          # base_lin_vel (rough estimate, matching db_1.py)
            np.array([0, 0, 0], dtype=np.float32),  # base_ang_vel (standing steady)
            np.array([0, 0, -1], dtype=np.float32),  # projected_gravity (upright)
            cmd,                 # velocity_cmd
            joint_pos_rel,       # joint_pos_rel
            joint_vel,           # joint_vel
            effort_normalized,   # joint_effort
            prev_actions,        # prev_actions
        ]).astype(np.float32)

        # Policy inference
        with torch.no_grad():
            obs_t = torch.tensor(obs).unsqueeze(0)
            actions = policy(obs_t).squeeze().numpy()

        # Update state with simple PD dynamics
        # Target position = default + action * scale
        new_target = SIM_DEFAULTS + actions * ACTION_SCALE

        # PD acceleration: a = Kp*(target - pos) - Kd*vel  (divided by inertia ~1)
        accel = STIFFNESS * (new_target - joint_pos) - DAMPING * joint_vel

        # Integrate (semi-implicit Euler)
        joint_vel = joint_vel + accel * dt
        joint_vel = np.clip(joint_vel, -10.5, 10.5)  # velocity limit
        joint_pos = joint_pos + joint_vel * dt

        # Clamp to joint limits
        joint_pos = np.clip(joint_pos, JOINT_LOWER, JOINT_UPPER)

        prev_actions = actions.copy()

        # Record
        row = {'step': step, 'time': t}
        for i in range(60):
            row[f'obs_{i}'] = obs[i]
        for i in range(12):
            row[f'action_{i}'] = actions[i]
            row[f'joint_pos_{i}'] = joint_pos[i]
            row[f'joint_pos_rel_{i}'] = joint_pos[i] - SIM_DEFAULTS[i]
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def summarize(df):
    """Print gait summary for LF leg."""
    print("\n" + "=" * 70)
    print("LEFT FRONT LEG — OPEN-LOOP GAIT ANALYSIS")
    print("=" * 70)

    # Skip first 2 seconds (settle time)
    settled = df[df['time'] >= 2.0]

    for j, name in enumerate(['LF_hip', 'LF_thigh', 'LF_calf']):
        pos = settled[f'joint_pos_rel_{j}'].values
        act = settled[f'action_{j}'].values

        mn, mx = pos.min(), pos.max()
        print(f"\n{name}:")
        print(f"  Position range: [{mn:+.4f}, {mx:+.4f}] rad  "
              f"([{np.degrees(mn):+.2f}, {np.degrees(mx):+.2f}] deg)")
        print(f"  Action range:   [{act.min():+.4f}, {act.max():+.4f}]")

        # Estimate frequency from zero crossings
        centered = pos - pos.mean()
        crossings = np.where(np.diff(np.sign(centered)))[0]
        if len(crossings) > 2:
            avg_half_period = np.mean(np.diff(crossings)) * 0.02  # dt
            freq = 1.0 / (2 * avg_half_period)
            print(f"  Estimated freq: {freq:.2f} Hz")

    print("\n" + "=" * 70)
    print("ALL JOINTS — POSITION RANGE (degrees, after 2s settle)")
    print("=" * 70)
    for j in range(12):
        pos = settled[f'joint_pos_rel_{j}'].values
        print(f"  {JOINT_NAMES[j]:12s}: [{np.degrees(pos.min()):+7.2f}, {np.degrees(pos.max()):+7.2f}]  "
              f"range={np.degrees(pos.max()-pos.min()):5.2f} deg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd_x", type=float, default=0.15)
    parser.add_argument("--cmd_y", type=float, default=0.0)
    parser.add_argument("--cmd_yaw", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cmd = [args.cmd_x, args.cmd_y, args.cmd_yaw]
    df = run_openloop(cmd, duration=args.duration)

    out_path = args.output or f"/Users/javierweddington/SpotDMouse/P2-Terrain_Challenge/mp2/misc/openloop_cmd_{args.cmd_x}_{args.cmd_y}_{args.cmd_yaw}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} steps to: {out_path}")

    summarize(df)
