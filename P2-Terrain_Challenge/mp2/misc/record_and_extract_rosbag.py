#!/usr/bin/env python3
"""
Two-part script:
  Part 1 (--record): Prints the ros2 commands to run ON THE ROBOT
  Part 2 (--extract): Reads a rosbag .db3 and extracts 60-dim observations to CSV
                       (runs on any machine, no ROS2 needed)

The 60-dim observation matches mlp_controller_node.py / SpotObservationsCfg:
  [0:3]   base_lin_vel       ← /odom/local twist (or estimated from cmd_vel)
  [3:6]   base_ang_vel       ← /imu/data angular_velocity
  [6:9]   projected_gravity  ← /imu/data orientation → gravity projection
  [9:12]  velocity_cmd       ← /cmd_vel
  [12:24] joint_pos_rel      ← /joint_states position - defaults
  [24:36] joint_vel          ← /joint_states velocity (or finite diff)
  [36:48] joint_effort       ← /joint_states effort (or zeros)
  [48:60] prev_actions       ← /joint_group_effort_controller/joint_trajectory

Usage:
  # On robot — see what to run:
  python3 record_and_extract_rosbag.py --record

  # After copying bag to this machine — extract:
  python3 record_and_extract_rosbag.py --extract /path/to/rosbag_dir

  # Compare against open-loop:
  python3 record_and_extract_rosbag.py --extract /path/to/rosbag_dir --compare openloop_cmd_0.15_0.0_0.0.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation


# ─── Constants ────────────────────────────────────────────────────────────────

SIM_DEFAULTS = np.array([
    0.0, 0.785, -1.57,  # LF
    0.0, 0.785, -1.57,  # RF
    0.0, 0.785, -1.57,  # LB
    0.0, 0.785, -1.57,  # RB
], dtype=np.float64)

JOINT_NAMES_ISAAC = [
    'base_lf1', 'lf1_lf2', 'lf2_lf3',
    'base_rf1', 'rf1_rf2', 'rf2_rf3',
    'base_lb1', 'lb1_lb2', 'lb2_lb3',
    'base_rb1', 'rb1_rb2', 'rb2_rb3',
]

JOINT_LABELS = [
    'LF_hip', 'LF_thigh', 'LF_calf',
    'RF_hip', 'RF_thigh', 'RF_calf',
    'LB_hip', 'LB_thigh', 'LB_calf',
    'RB_hip', 'RB_thigh', 'RB_calf',
]

TOPICS_TO_RECORD = [
    '/joint_states',
    '/imu/data',
    '/odom/local',
    '/cmd_vel',
    '/joint_group_effort_controller/joint_trajectory',
    '/foot_contacts',
]


# ─── Part 1: Print recording instructions ────────────────────────────────────

def print_record_instructions():
    topics_str = ' '.join(TOPICS_TO_RECORD)
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║              ROSBAG RECORDING INSTRUCTIONS                         ║
║              Run these commands ON THE ROBOT                       ║
╚══════════════════════════════════════════════════════════════════════╝

STEP 1: Launch the robot bringup (Terminal 1)
─────────────────────────────────────────────
  ros2 launch mini_pupper_bringup bringup.launch.py \\
      hardware_connected:=true \\
      has_imu:=true

STEP 2: Start rosbag recording (Terminal 2)
───────────────────────────────────────────
  ros2 bag record -o walk_forward_$(date +%Y%m%d_%H%M%S) {topics_str} --compression-mode message --compression-format zstd

STEP 3: Teleop walk forward (Terminal 3)
────────────────────────────────────────
  # Option A: keyboard teleop
  ros2 run teleop_twist_keyboard teleop_twist_keyboard

  # Option B: publish fixed cmd_vel for ~10 seconds
  ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \\
      "{linear: {x: 0.15, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" \\
      -r 50

  # Walk for 10-20 seconds, then Ctrl+C the teleop

STEP 4: Stop recording
──────────────────────
  Ctrl+C the rosbag recording in Terminal 2

STEP 5: Copy to this machine
─────────────────────────────
  scp -r ubuntu@<robot_ip>:~/walk_forward_* .

STEP 6: Extract observations
────────────────────────────
  python3 record_and_extract_rosbag.py --extract walk_forward_*/

NOTE: The bag MUST contain /imu/data for gravity and angular velocity.
      If /odom/local is missing, base_lin_vel will be estimated from cmd_vel.
      If /joint_states has no velocity/effort, they'll be computed/zeroed.
""")


# ─── Part 2: Extract observations from rosbag ────────────────────────────────

def extract_rosbag(bag_path, output_csv=None):
    """Read rosbag and construct 60-dim observations at ~50Hz."""
    from rosbags.rosbag2 import Reader
    from rosbags.serde import deserialize_cdr

    bag_path = Path(bag_path)
    print(f"Reading rosbag: {bag_path}")

    # ── Read all messages into time-sorted lists ──
    joint_states = []    # (t_ns, positions, velocities, efforts, names)
    imu_data = []        # (t_ns, ang_vel, orientation_quat)
    cmd_vels = []        # (t_ns, [vx, vy, vyaw])
    odom_data = []       # (t_ns, [vx, vy, vz])
    joint_cmds = []      # (t_ns, positions)  — from trajectory commands

    with Reader(bag_path) as reader:
        print("Topics in bag:")
        for conn in reader.connections:
            print(f"  {conn.topic}: {conn.msgtype} ({conn.msgcount} msgs)")

        for conn, timestamp, rawdata in reader.messages():
            topic = conn.topic
            try:
                if topic == '/joint_states':
                    msg = deserialize_cdr(rawdata, conn.msgtype)
                    names = list(msg.name)
                    pos = np.array(msg.position, dtype=np.float64)
                    vel = np.array(msg.velocity, dtype=np.float64) if len(msg.velocity) > 0 else None
                    eff = np.array(msg.effort, dtype=np.float64) if len(msg.effort) > 0 else None
                    joint_states.append((timestamp, pos, vel, eff, names))

                elif topic == '/imu/data':
                    msg = deserialize_cdr(rawdata, conn.msgtype)
                    ang_vel = np.array([
                        msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z,
                    ])
                    quat = np.array([
                        msg.orientation.x, msg.orientation.y,
                        msg.orientation.z, msg.orientation.w,
                    ])
                    imu_data.append((timestamp, ang_vel, quat))

                elif topic == '/cmd_vel':
                    msg = deserialize_cdr(rawdata, conn.msgtype)
                    cmd_vels.append((timestamp, np.array([
                        msg.linear.x, msg.linear.y, msg.angular.z,
                    ])))

                elif topic == '/odom/local' or topic == '/odom':
                    msg = deserialize_cdr(rawdata, conn.msgtype)
                    odom_data.append((timestamp, np.array([
                        msg.twist.twist.linear.x,
                        msg.twist.twist.linear.y,
                        msg.twist.twist.linear.z,
                    ])))

                elif topic == '/joint_group_effort_controller/joint_trajectory':
                    msg = deserialize_cdr(rawdata, conn.msgtype)
                    if msg.points:
                        cmd_pos = np.array(msg.points[0].positions, dtype=np.float64)
                        joint_cmds.append((timestamp, cmd_pos))

            except Exception as e:
                pass  # Skip messages that can't be deserialized (e.g. champ_msgs)

    print(f"\nMessages read:")
    print(f"  joint_states:  {len(joint_states)}")
    print(f"  imu_data:      {len(imu_data)}")
    print(f"  cmd_vel:       {len(cmd_vels)}")
    print(f"  odom:          {len(odom_data)}")
    print(f"  joint_cmds:    {len(joint_cmds)}")

    if not joint_states:
        print("ERROR: No /joint_states messages found!")
        return None

    # ── Build joint name → index mapping ──
    sample_names = joint_states[0][4]
    name_to_idx = {n: i for i, n in enumerate(sample_names)}
    # Reorder map: isaac_idx → bag_idx
    reorder = []
    for isaac_name in JOINT_NAMES_ISAAC:
        if isaac_name in name_to_idx:
            reorder.append(name_to_idx[isaac_name])
        else:
            print(f"WARNING: Joint '{isaac_name}' not found in bag. Available: {sample_names}")
            reorder.append(0)
    reorder = np.array(reorder)

    # Check if joints are already in Isaac order
    already_ordered = all(sample_names[i] == JOINT_NAMES_ISAAC[i] for i in range(min(12, len(sample_names))))
    if already_ordered:
        print("Joint names already in Isaac order — no reordering needed")

    has_imu = len(imu_data) > 0
    has_odom = len(odom_data) > 0
    has_vel = joint_states[0][2] is not None
    has_effort = joint_states[0][3] is not None
    has_cmds = len(joint_cmds) > 0

    print(f"\nData availability:")
    print(f"  IMU (gravity/ang_vel): {'YES' if has_imu else 'NO — will use defaults'}")
    print(f"  Odom (base_lin_vel):   {'YES' if has_odom else 'NO — will estimate from cmd_vel'}")
    print(f"  Joint velocities:      {'YES' if has_vel else 'NO — will compute from finite diff'}")
    print(f"  Joint efforts:         {'YES' if has_effort else 'NO — will be zeros'}")
    print(f"  Joint commands:        {'YES' if has_cmds else 'NO — will use joint positions as prev_actions'}")

    # ── Helper: find nearest message before timestamp ──
    def find_nearest(msg_list, t_ns):
        """Binary search for latest message at or before t_ns."""
        if not msg_list:
            return None
        lo, hi = 0, len(msg_list) - 1
        if t_ns < msg_list[0][0]:
            return msg_list[0]
        if t_ns >= msg_list[-1][0]:
            return msg_list[-1]
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if msg_list[mid][0] <= t_ns:
                lo = mid
            else:
                hi = mid - 1
        return msg_list[lo]

    # ── Build observations at each joint_states timestamp ──
    t0 = joint_states[0][0]
    prev_positions = None
    prev_t_ns = None
    prev_actions = np.zeros(12, dtype=np.float64)

    rows = []
    for step, (t_ns, raw_pos, raw_vel, raw_eff, _names) in enumerate(joint_states):
        t_sec = (t_ns - t0) / 1e9

        # Reorder positions to Isaac order
        pos = raw_pos[reorder] if not already_ordered else raw_pos[:12].copy()

        # Joint pos relative to defaults
        joint_pos_rel = pos - SIM_DEFAULTS

        # Joint velocity
        if raw_vel is not None and len(raw_vel) >= 12:
            joint_vel = (raw_vel[reorder] if not already_ordered else raw_vel[:12]).copy()
        elif prev_positions is not None and prev_t_ns is not None:
            dt = (t_ns - prev_t_ns) / 1e9
            if dt > 0.001:
                joint_vel = (pos - prev_positions) / dt
            else:
                joint_vel = np.zeros(12)
        else:
            joint_vel = np.zeros(12)

        # Joint effort
        if raw_eff is not None and len(raw_eff) >= 12:
            joint_effort = (raw_eff[reorder] if not already_ordered else raw_eff[:12]).copy()
        else:
            joint_effort = np.zeros(12)

        # IMU: angular velocity and gravity
        if has_imu:
            imu_msg = find_nearest(imu_data, t_ns)
            base_ang_vel = imu_msg[1].copy()
            quat_xyzw = imu_msg[2]
            rot = Rotation.from_quat(quat_xyzw)  # scipy uses [x,y,z,w]
            gravity_world = np.array([0.0, 0.0, -1.0])
            projected_gravity = rot.inv().apply(gravity_world)
        else:
            base_ang_vel = np.zeros(3)
            projected_gravity = np.array([0.0, 0.0, -1.0])

        # Base linear velocity
        if has_odom:
            odom_msg = find_nearest(odom_data, t_ns)
            base_lin_vel = odom_msg[1].copy()
        else:
            # Estimate from cmd_vel (rough proxy)
            cmd_msg = find_nearest(cmd_vels, t_ns) if cmd_vels else None
            if cmd_msg is not None:
                base_lin_vel = cmd_msg[1] * 0.7  # same heuristic as db_1.py
            else:
                base_lin_vel = np.zeros(3)

        # Velocity command
        cmd_msg = find_nearest(cmd_vels, t_ns) if cmd_vels else None
        velocity_cmd = cmd_msg[1].copy() if cmd_msg is not None else np.zeros(3)

        # Previous actions (commanded joint positions relative to default)
        if has_cmds:
            cmd_nearest = find_nearest(joint_cmds, t_ns)
            if cmd_nearest is not None:
                cmd_pos = cmd_nearest[1][:12]
                if not already_ordered:
                    cmd_pos = cmd_pos[reorder]
                prev_actions = (cmd_pos - SIM_DEFAULTS)
        else:
            # Use previous joint positions as proxy
            if prev_positions is not None:
                prev_actions = prev_positions - SIM_DEFAULTS
            else:
                prev_actions = np.zeros(12)

        # Assemble 60-dim observation
        obs = np.concatenate([
            base_lin_vel,       # [0:3]
            base_ang_vel,       # [3:6]
            projected_gravity,  # [6:9]
            velocity_cmd,       # [9:12]
            joint_pos_rel,      # [12:24]
            joint_vel,          # [24:36]
            joint_effort,       # [36:48]
            prev_actions,       # [48:60]
        ])

        row = {'step': step, 'time': t_sec}
        for i in range(60):
            row[f'obs_{i}'] = obs[i]
        for i in range(12):
            row[f'joint_pos_rel_{i}'] = joint_pos_rel[i]
            row[f'joint_vel_{i}'] = joint_vel[i]
            row[f'action_{i}'] = prev_actions[i]
        rows.append(row)

        prev_positions = pos.copy()
        prev_t_ns = t_ns

    df = pd.DataFrame(rows)

    # Save
    if output_csv is None:
        output_csv = str(bag_path).rstrip('/') + '_obs60.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nExtracted {len(df)} observations → {output_csv}")
    print(f"Duration: {df['time'].max():.1f}s at ~{len(df)/df['time'].max():.0f} Hz")

    return df


# ─── Part 3: Compare ─────────────────────────────────────────────────────────

def compare(real_df, openloop_df, settle=2.0):
    """Compare real rosbag observations against open-loop policy rollout."""

    real = real_df[real_df['time'] >= settle].copy()
    ol = openloop_df[openloop_df['time'] >= settle].copy()

    print(f"\n{'='*85}")
    print(f"COMPARISON: Rosbag Real Walk vs Open-Loop Policy Rollout")
    print(f"{'='*85}")
    print(f"Real: {len(real)} samples ({real['time'].min():.1f}–{real['time'].max():.1f}s)")
    print(f"Open-loop: {len(ol)} samples ({ol['time'].min():.1f}–{ol['time'].max():.1f}s)")

    # Joint position comparison
    print(f"\n{'─'*85}")
    print(f"{'Joint':12s} | {'REAL (rosbag)':^36s} | {'OPEN-LOOP':^36s}")
    print(f"{'':12s} | {'Min°':>7s} {'Max°':>7s} {'Range°':>7s} {'Std°':>7s} {'Hz':>5s} | "
          f"{'Min°':>7s} {'Max°':>7s} {'Range°':>7s} {'Std°':>7s} {'Hz':>5s}")
    print(f"{'─'*85}")

    for j in range(12):
        r_pos = real[f'joint_pos_rel_{j}'].values
        o_pos = ol[f'joint_pos_rel_{j}'].values

        def freq(sig, dt=0.02):
            c = sig - sig.mean()
            x = np.where(np.diff(np.sign(c)))[0]
            return 1.0 / (2 * np.mean(np.diff(x)) * dt) if len(x) > 4 else float('nan')

        print(f"{JOINT_LABELS[j]:12s} | "
              f"{np.degrees(r_pos.min()):+7.2f} {np.degrees(r_pos.max()):+7.2f} "
              f"{np.degrees(r_pos.ptp()):7.2f} {np.degrees(r_pos.std()):7.2f} {freq(r_pos):5.2f} | "
              f"{np.degrees(o_pos.min()):+7.2f} {np.degrees(o_pos.max()):+7.2f} "
              f"{np.degrees(o_pos.ptp()):7.2f} {np.degrees(o_pos.std()):7.2f} {freq(o_pos):5.2f}")

    # Observation channel divergence
    obs_labels = [
        ('base_lin_vel', 0, 3),
        ('base_ang_vel', 3, 6),
        ('proj_gravity', 6, 9),
        ('velocity_cmd', 9, 12),
        ('joint_pos_rel', 12, 24),
        ('joint_vel', 24, 36),
        ('joint_effort', 36, 48),
        ('prev_actions', 48, 60),
    ]

    print(f"\n{'─'*85}")
    print("OBSERVATION CHANNEL DIVERGENCE (mean ± std)")
    print(f"{'─'*85}")

    for label, start, end in obs_labels:
        r_cols = [f'obs_{i}' for i in range(start, end)]
        o_cols = [f'obs_{i}' for i in range(start, end)]

        r_mean = real[r_cols].mean().values
        o_mean = ol[o_cols].mean().values
        r_std = real[r_cols].std().values
        o_std = ol[o_cols].std().values

        combined_std = np.sqrt(r_std**2 + o_std**2)
        combined_std[combined_std < 1e-6] = 1.0
        norm_div = np.abs(r_mean - o_mean) / combined_std

        bar = "█" * int(min(norm_div.mean() * 10, 40))
        print(f"  {label:16s}: div={norm_div.mean():.3f}  "
              f"real_mean={r_mean.mean():+.4f}  ol_mean={o_mean.mean():+.4f}  {bar}")

    # Overall
    all_r = real[[f'obs_{i}' for i in range(60)]].mean().values
    all_o = ol[[f'obs_{i}' for i in range(60)]].mean().values
    all_rs = real[[f'obs_{i}' for i in range(60)]].std().values
    all_os = ol[[f'obs_{i}' for i in range(60)]].std().values
    cs = np.sqrt(all_rs**2 + all_os**2)
    cs[cs < 1e-6] = 1.0
    overall = np.abs(all_r - all_o) / cs

    print(f"\n  Overall normalized divergence: {overall.mean():.3f}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--record", action="store_true", help="Print recording instructions")
    group.add_argument("--extract", type=str, metavar="BAG_PATH", help="Extract 60-dim obs from rosbag")

    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--compare", type=str, default=None, help="Open-loop CSV to compare against")
    parser.add_argument("--settle", type=float, default=2.0, help="Settle time (seconds)")
    args = parser.parse_args()

    if args.record:
        print_record_instructions()
    else:
        real_df = extract_rosbag(args.extract, output_csv=args.output)

        if real_df is not None and args.compare:
            ol_df = pd.read_csv(args.compare)
            compare(real_df, ol_df, settle=args.settle)
