"""
Compare Real vs Simulation Observations
========================================
Replays simulation actions while comparing:
1. Real robot observations (from sensors)
2. Simulation observations (from CSV)

This identifies which observations are mismatched and by how much.
"""

import numpy as np
import pandas as pd
import time
from mlp_controller_v3 import FixedMappingControllerV3


def compare_observations(controller, actions_csv, observations_csv, max_steps=150):
    """
    Replay actions from CSV while comparing observations.

    Args:
        controller: Controller instance
        actions_csv: Path to simulation actions CSV
        observations_csv: Path to simulation observations CSV
        max_steps: Number of steps to replay
    """
    print("\n" + "="*70)
    print("OBSERVATION COMPARISON TEST")
    print("="*70)

    # Load CSVs
    try:
        sim_actions = pd.read_csv(actions_csv)
        sim_obs = pd.read_csv(observations_csv)
        print(f"[OK] Loaded {len(sim_actions)} action steps")
        print(f"[OK] Loaded {len(sim_obs)} observation steps")
    except Exception as e:
        print(f"[ERROR] Could not load CSVs: {e}")
        return

    print("\n[INFO] Starting comparison in 3 seconds...")
    time.sleep(3)

    # Storage for comparison
    mismatches = {
        'base_lin_vel': [],
        'base_ang_vel': [],
        'gravity': [],
        'joint_pos_rel': [],
        'joint_vel': [],
    }

    # Replay and compare
    for i in range(min(max_steps, len(sim_actions), len(sim_obs))):
        # Apply action
        action_row = sim_actions.iloc[i]
        actions = np.array([
            action_row['action_base_lf1'], action_row['action_lf1_lf2'], action_row['action_lf2_lf3'],
            action_row['action_base_rf1'], action_row['action_rf1_rf2'], action_row['action_rf2_rf3'],
            action_row['action_base_lb1'], action_row['action_lb1_lb2'], action_row['action_lb2_lb3'],
            action_row['action_base_rb1'], action_row['action_rb1_rb2'], action_row['action_rb2_rb3'],
        ])

        target_sim = controller.sim_default_positions + actions * 0.5
        target_sim = np.clip(target_sim, controller.joint_lower_limits,
                            controller.joint_upper_limits)
        target_matrix = controller._isaac_to_hardware_matrix(target_sim)
        controller.hardware.set_actuator_postions(target_matrix)

        time.sleep(0.02)  # 50Hz

        # Get real observation
        real_obs = controller.get_observation()

        # Get sim observation
        obs_row = sim_obs.iloc[i]
        sim_obs_vec = np.array([
            obs_row['base_lin_vel_x'], obs_row['base_lin_vel_y'], obs_row['base_lin_vel_z'],
            obs_row['base_ang_vel_x'], obs_row['base_ang_vel_y'], obs_row['base_ang_vel_z'],
            obs_row['projected_gravity_x'], obs_row['projected_gravity_y'], obs_row['projected_gravity_z'],
            obs_row['velocity_command_x'], obs_row['velocity_command_y'], obs_row['velocity_command_yaw'],
            obs_row['joint_pos_base_lf1'], obs_row['joint_pos_lf1_lf2'], obs_row['joint_pos_lf2_lf3'],
            obs_row['joint_pos_base_rf1'], obs_row['joint_pos_rf1_rf2'], obs_row['joint_pos_rf2_rf3'],
            obs_row['joint_pos_base_lb1'], obs_row['joint_pos_lb1_lb2'], obs_row['joint_pos_lb2_lb3'],
            obs_row['joint_pos_base_rb1'], obs_row['joint_pos_rb1_rb2'], obs_row['joint_pos_rb2_rb3'],
            obs_row['joint_vel_base_lf1'], obs_row['joint_vel_lf1_lf2'], obs_row['joint_vel_lf2_lf3'],
            obs_row['joint_vel_base_rf1'], obs_row['joint_vel_rf1_rf2'], obs_row['joint_vel_rf2_rf3'],
            obs_row['joint_vel_base_lb1'], obs_row['joint_vel_lb1_lb2'], obs_row['joint_vel_lb2_lb3'],
            obs_row['joint_vel_base_rb1'], obs_row['joint_vel_rb1_rb2'], obs_row['joint_vel_rb2_rb3'],
            # Skip effort and prev_actions for now
        ], dtype=np.float32)

        # Compare key observations
        real_base_lin_vel = real_obs[0:3]
        sim_base_lin_vel = sim_obs_vec[0:3]

        real_base_ang_vel = real_obs[3:6]
        sim_base_ang_vel = sim_obs_vec[3:6]

        real_gravity = real_obs[6:9]
        sim_gravity = sim_obs_vec[6:9]

        real_joint_pos = real_obs[12:24]
        sim_joint_pos = sim_obs_vec[12:24]

        real_joint_vel = real_obs[24:36]
        sim_joint_vel = sim_obs_vec[24:36]

        # Compute errors
        lin_vel_error = np.abs(real_base_lin_vel - sim_base_lin_vel).max()
        ang_vel_error = np.abs(real_base_ang_vel - sim_base_ang_vel).max()
        gravity_error = np.abs(real_gravity - sim_gravity).max()
        joint_pos_error = np.abs(real_joint_pos - sim_joint_pos).max()
        joint_vel_error = np.abs(real_joint_vel - sim_joint_vel).max()

        mismatches['base_lin_vel'].append(lin_vel_error)
        mismatches['base_ang_vel'].append(ang_vel_error)
        mismatches['gravity'].append(gravity_error)
        mismatches['joint_pos_rel'].append(joint_pos_error)
        mismatches['joint_vel'].append(joint_vel_error)

        # Print every 25 steps
        if i % 25 == 0:
            print(f"\nStep {i:3d}:")
            print(f"  Lin Vel Error:  {lin_vel_error:.4f}")
            print(f"  Ang Vel Error:  {ang_vel_error:.4f}")
            print(f"  Gravity Error:  {gravity_error:.4f}")
            print(f"  Joint Pos Err:  {joint_pos_error:.4f}")
            print(f"  Joint Vel Err:  {joint_vel_error:.4f}")

            if i == 0:
                print(f"\n  Real Lin Vel:   [{real_base_lin_vel[0]:+.3f}, {real_base_lin_vel[1]:+.3f}, {real_base_lin_vel[2]:+.3f}]")
                print(f"  Sim  Lin Vel:   [{sim_base_lin_vel[0]:+.3f}, {sim_base_lin_vel[1]:+.3f}, {sim_base_lin_vel[2]:+.3f}]")
                print(f"\n  Real Ang Vel:   [{real_base_ang_vel[0]:+.3f}, {real_base_ang_vel[1]:+.3f}, {real_base_ang_vel[2]:+.3f}]")
                print(f"  Sim  Ang Vel:   [{sim_base_ang_vel[0]:+.3f}, {sim_base_ang_vel[1]:+.3f}, {sim_base_ang_vel[2]:+.3f}]")
                print(f"\n  Real Gravity:   [{real_gravity[0]:+.3f}, {real_gravity[1]:+.3f}, {real_gravity[2]:+.3f}]")
                print(f"  Sim  Gravity:   [{sim_gravity[0]:+.3f}, {sim_gravity[1]:+.3f}, {sim_gravity[2]:+.3f}]")

    # Summary statistics
    print("\n" + "="*70)
    print("OBSERVATION ERROR SUMMARY")
    print("="*70)

    for key, errors in mismatches.items():
        errors = np.array(errors)
        print(f"\n{key}:")
        print(f"  Mean Error:   {errors.mean():.4f}")
        print(f"  Max Error:    {errors.max():.4f}")
        print(f"  Min Error:    {errors.min():.4f}")
        print(f"  Std Dev:      {errors.std():.4f}")

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    # Identify problem observations
    problems = []

    lin_vel_mean = np.mean(mismatches['base_lin_vel'])
    ang_vel_mean = np.mean(mismatches['base_ang_vel'])
    gravity_mean = np.mean(mismatches['gravity'])
    joint_pos_mean = np.mean(mismatches['joint_pos_rel'])
    joint_vel_mean = np.mean(mismatches['joint_vel'])

    # Thresholds for "acceptable" error
    if lin_vel_mean > 0.1:
        problems.append(f"❌ base_lin_vel: High error ({lin_vel_mean:.3f}) - Check velocity estimation")
    else:
        print(f"✓ base_lin_vel: Low error ({lin_vel_mean:.3f})")

    if ang_vel_mean > 0.2:
        problems.append(f"❌ base_ang_vel: High error ({ang_vel_mean:.3f}) - Check IMU calibration/frame")
    else:
        print(f"✓ base_ang_vel: Low error ({ang_vel_mean:.3f})")

    if gravity_mean > 0.05:
        problems.append(f"❌ gravity: High error ({gravity_mean:.3f}) - Check IMU frame orientation")
    else:
        print(f"✓ gravity: Low error ({gravity_mean:.3f})")

    if joint_pos_mean > 0.1:
        problems.append(f"❌ joint_pos_rel: High error ({joint_pos_mean:.3f}) - Check joint transformations")
    else:
        print(f"✓ joint_pos_rel: Low error ({joint_pos_mean:.3f})")

    if joint_vel_mean > 0.5:
        problems.append(f"❌ joint_vel: High error ({joint_vel_mean:.3f}) - Check velocity computation")
    else:
        print(f"✓ joint_vel: Low error ({joint_vel_mean:.3f})")

    if problems:
        print("\n" + "="*70)
        print("PROBLEMS DETECTED:")
        print("="*70)
        for problem in problems:
            print(problem)
        print("\nThese observations are significantly different from simulation!")
        print("The policy receives different inputs than it was trained on.")
    else:
        print("\n✓ All observations within acceptable error ranges!")

    # Return to default
    print("\n[INFO] Returning to default stance...")
    target_matrix = controller._isaac_to_hardware_matrix(controller.sim_default_positions)
    controller.hardware.set_actuator_postions(target_matrix)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("OBSERVATION COMPARISON TOOL")
    print("="*70)
    print("""
This tool replays simulation actions on the real robot while
comparing the real sensor observations to the simulation observations.

This helps identify which observations are mismatched.
""")

    input("Press Enter to start...")

    controller = FixedMappingControllerV3("/home/ubuntu/mp2_mlp/policy_joyboy.pt")

    # Use forward walking test
    actions_csv = "/home/ubuntu/debug/obs_action_logs_x_030/env_0_actions.csv"
    observations_csv = "/home/ubuntu/debug/obs_action_logs_x_030/env_0_observations.csv"

    try:
        compare_observations(controller, actions_csv, observations_csv, max_steps=150)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

    print("\n[DONE]")
