#!/usr/bin/env python3
"""
Open-loop policy rollout — runs the trained policy WITHOUT a simulator or robot.
Uses a simple first-order joint dynamics model to feed observations back.
Includes a delay buffer to model DelayedPDActuator behavior.

Purpose: Verify the policy produces a reasonable gait for multiple commands,
check direction-dependent behavior, and validate joint ranges.
"""

import torch
import numpy as np
import pandas as pd
import argparse
from collections import deque

# ─── Config ───────────────────────────────────────────────────────────────────
POLICY_PATH = "/Users/javierweddington/policy_joyboy_delayedpdactuator_scheduled.pt"

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

# Actuator params (from current DelayedPDActuatorCfg)
STIFFNESS = 80.0
DAMPING = 2.5
ACTION_SCALE = 0.5  # from SpotActionsCfg

# Delay params: sim runs at 500Hz (dt=0.002), policy at 50Hz (decimation=10)
# min_delay=26, max_delay=31 physics steps → ~3 policy steps avg
DELAY_POLICY_STEPS = 3  # average delay in policy steps


def run_openloop(cmd_vel, duration=10.0, dt=0.02, policy_path=POLICY_PATH, use_delay=True):
    """
    Run the policy in open loop with a simple PD joint model and optional delay buffer.

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

    # Delay buffer — actions are delayed by DELAY_POLICY_STEPS before being applied
    delay_steps = DELAY_POLICY_STEPS if use_delay else 0
    action_buffer = deque([np.zeros(12, dtype=np.float32)] * (delay_steps + 1), maxlen=delay_steps + 1)

    rows = []

    print(f"Running {n_steps} steps ({duration}s) at cmd={cmd.tolist()}")
    print(f"Action scale={ACTION_SCALE}, stiffness={STIFFNESS}, damping={DAMPING}")
    print(f"Delay: {delay_steps} policy steps ({'enabled' if use_delay else 'disabled'})")

    for step in range(n_steps):
        t = step * dt

        # Build observation
        joint_pos_rel = joint_pos - SIM_DEFAULTS

        # Simple effort model: tau = Kp * (target - pos) - Kd * vel
        # where target = default + prev_action * scale (using the DELAYED action)
        target = SIM_DEFAULTS + prev_actions * ACTION_SCALE
        effort = STIFFNESS * (target - joint_pos) - DAMPING * joint_vel
        effort_normalized = effort / 5.0

        obs = np.concatenate([
            cmd * 0.7,          # base_lin_vel (rough estimate)
            np.array([0, 0, 0], dtype=np.float32),  # base_ang_vel
            np.array([0, 0, -1], dtype=np.float32),  # projected_gravity (upright)
            cmd,                 # velocity_cmd
            joint_pos_rel,       # joint_pos_rel
            joint_vel,           # joint_vel
            effort_normalized,   # joint_effort
            prev_actions,        # prev_actions (the delayed action the actuator actually applied)
        ]).astype(np.float32)

        # Policy inference
        with torch.no_grad():
            obs_t = torch.tensor(obs).unsqueeze(0)
            raw_actions = policy(obs_t).squeeze().numpy()

        # Push new action into delay buffer, pop delayed action
        action_buffer.append(raw_actions.copy())
        delayed_actions = action_buffer[0]

        # The delayed action is what actually gets applied
        prev_actions = delayed_actions.copy()

        # Update state with simple PD dynamics using DELAYED action
        new_target = SIM_DEFAULTS + delayed_actions * ACTION_SCALE

        accel = STIFFNESS * (new_target - joint_pos) - DAMPING * joint_vel

        # Integrate (semi-implicit Euler)
        joint_vel = joint_vel + accel * dt
        joint_vel = np.clip(joint_vel, -15.0, 15.0)  # velocity limit matches actuator config
        joint_pos = joint_pos + joint_vel * dt

        # Clamp to joint limits
        joint_pos = np.clip(joint_pos, JOINT_LOWER, JOINT_UPPER)

        # Record
        row = {'step': step, 'time': t}
        for i in range(60):
            row[f'obs_{i}'] = obs[i]
        for i in range(12):
            row[f'action_{i}'] = raw_actions[i]
            row[f'delayed_action_{i}'] = delayed_actions[i]
            row[f'joint_pos_{i}'] = joint_pos[i]
            row[f'joint_pos_rel_{i}'] = joint_pos[i] - SIM_DEFAULTS[i]
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def summarize(df, cmd_label=""):
    """Print gait summary for all joints."""
    print("\n" + "=" * 70)
    print(f"GAIT ANALYSIS — {cmd_label}")
    print("=" * 70)

    # Skip first 2 seconds (settle time)
    settled = df[df['time'] >= 2.0]

    print("\nALL JOINTS — POSITION RANGE (degrees, after 2s settle)")
    print("-" * 70)
    for j in range(12):
        pos = settled[f'joint_pos_rel_{j}'].values
        act = settled[f'action_{j}'].values

        # Estimate frequency from zero crossings
        centered = pos - pos.mean()
        crossings = np.where(np.diff(np.sign(centered)))[0]
        freq = float('nan')
        if len(crossings) > 2:
            avg_half_period = np.mean(np.diff(crossings)) * 0.02
            freq = 1.0 / (2 * avg_half_period)

        print(f"  {JOINT_NAMES[j]:12s}: pos=[{np.degrees(pos.min()):+7.2f}, {np.degrees(pos.max()):+7.2f}] "
              f"range={np.degrees(pos.max()-pos.min()):5.2f}°  "
              f"act=[{act.min():+.3f}, {act.max():+.3f}]  "
              f"freq={freq:.2f}Hz")

    # Check if actions differ from previous policy (same gait for all cmds?)
    print(f"\n  Action std (mean across joints): {settled[[f'action_{j}' for j in range(12)]].std().mean():.4f}")


def run_multi_command_comparison(policy_path, duration=10.0):
    """Run the policy with multiple commands to check direction-dependent behavior."""
    commands = {
        "forward_0.15":  [0.15, 0.0, 0.0],
        "backward_0.15": [-0.15, 0.0, 0.0],
        "left_0.15":     [0.0, 0.15, 0.0],
        "right_0.15":    [0.0, -0.15, 0.0],
        "yaw_left_0.2":  [0.0, 0.0, 0.2],
        "yaw_right_0.2": [0.0, 0.0, -0.2],
        "stop":          [0.0, 0.0, 0.0],
    }

    results = {}
    for name, cmd in commands.items():
        print(f"\n{'='*70}")
        print(f"COMMAND: {name} → {cmd}")
        print(f"{'='*70}")
        df = run_openloop(cmd, duration=duration, policy_path=policy_path)
        results[name] = df
        summarize(df, cmd_label=name)

        # Save CSV
        out_dir = "/Users/javierweddington/SpotDMouse/P2-Terrain_Challenge/mp2/misc"
        out_path = f"{out_dir}/openloop_scheduled_{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}")

    # Cross-command comparison
    print("\n" + "=" * 70)
    print("CROSS-COMMAND COMPARISON (action correlation after 2s settle)")
    print("=" * 70)
    cmd_names = list(commands.keys())
    for i in range(len(cmd_names)):
        for j in range(i+1, len(cmd_names)):
            df_a = results[cmd_names[i]]
            df_b = results[cmd_names[j]]
            settled_a = df_a[df_a['time'] >= 2.0]
            settled_b = df_b[df_b['time'] >= 2.0]
            min_len = min(len(settled_a), len(settled_b))
            acts_a = settled_a[[f'action_{k}' for k in range(12)]].values[:min_len].flatten()
            acts_b = settled_b[[f'action_{k}' for k in range(12)]].values[:min_len].flatten()
            corr = np.corrcoef(acts_a, acts_b)[0, 1]
            print(f"  {cmd_names[i]:18s} vs {cmd_names[j]:18s}: corr={corr:+.3f}")

    print("\n  Low correlation (<0.5) = direction-dependent gait (GOOD)")
    print("  High correlation (>0.8) = same gait for all commands (BAD)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd_x", type=float, default=None)
    parser.add_argument("--cmd_y", type=float, default=None)
    parser.add_argument("--cmd_yaw", type=float, default=None)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--policy", type=str, default=POLICY_PATH)
    parser.add_argument("--no-delay", action="store_true", help="Disable delay buffer")
    parser.add_argument("--compare", action="store_true", help="Run all directions and compare")
    args = parser.parse_args()

    if args.compare:
        run_multi_command_comparison(args.policy, duration=args.duration)
    else:
        cmd_x = args.cmd_x if args.cmd_x is not None else 0.15
        cmd_y = args.cmd_y if args.cmd_y is not None else 0.0
        cmd_yaw = args.cmd_yaw if args.cmd_yaw is not None else 0.0
        cmd = [cmd_x, cmd_y, cmd_yaw]

        df = run_openloop(cmd, duration=args.duration, policy_path=args.policy, use_delay=not args.no_delay)

        out_path = args.output or f"/Users/javierweddington/SpotDMouse/P2-Terrain_Challenge/mp2/misc/openloop_cmd_{cmd_x}_{cmd_y}_{cmd_yaw}.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(df)} steps to: {out_path}")

        summarize(df, cmd_label=f"cmd=[{cmd_x}, {cmd_y}, {cmd_yaw}]")
