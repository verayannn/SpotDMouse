"""
Verify Transformation Pipeline
================================
This script tests the ENTIRE transformation pipeline step-by-step to identify
where the bug is located.

Pipeline stages:
1. CSV action → Sim joint positions (with ACTION_SCALE)
2. Sim joint positions → Real joint positions (joint_direction flip)
3. Real joint positions (Isaac order) → Hardware matrix
4. Hardware matrix → Physical robot

We'll test ONE joint at a time and print out the transformation at each stage.
"""

import numpy as np
import time
from mlp_controller_v3 import FixedMappingControllerV3


def print_joint_transformation(controller, joint_idx, action_value, joint_name):
    """
    Print the transformation pipeline for a single joint.

    Shows:
    - Input: CSV action value
    - Stage 1: Sim joint position
    - Stage 2: Real joint position
    - Stage 3: Hardware matrix position
    - Expected: Which servo should move and in what direction
    """
    print(f"\n{'='*70}")
    print(f"JOINT {joint_idx}: {joint_name}")
    print(f"{'='*70}")

    # Input
    print(f"\n[INPUT] CSV Action Value: {action_value:+.4f}")

    # Stage 1: Apply action to sim default position with ACTION_SCALE
    actions = np.zeros(12)
    actions[joint_idx] = action_value

    ACTION_SCALE = 0.5  # Same as CSV replay
    target_sim = controller.sim_default_positions + actions * ACTION_SCALE

    print(f"\n[STAGE 1] Sim Joint Position:")
    print(f"  Default:  {controller.sim_default_positions[joint_idx]:+.4f}")
    print(f"  + Action: {actions[joint_idx] * ACTION_SCALE:+.4f}")
    print(f"  = Target: {target_sim[joint_idx]:+.4f}")

    # Stage 2: Transform sim to real (using joint_direction)
    deviation_sim = target_sim - controller.sim_default_positions
    deviation_real = deviation_sim * controller.joint_direction
    angles_real = controller.real_default_positions + deviation_real

    print(f"\n[STAGE 2] Real Joint Position (after joint_direction flip):")
    print(f"  Deviation (sim):  {deviation_sim[joint_idx]:+.4f}")
    print(f"  joint_direction:  {controller.joint_direction[joint_idx]:+.1f}")
    print(f"  Deviation (real): {deviation_real[joint_idx]:+.4f}")
    print(f"  Real position:    {angles_real[joint_idx]:+.4f}")

    # Stage 3: Map to hardware matrix
    # Isaac order: LF(0-2), RF(3-5), LB(6-8), RB(9-11)
    # Hardware matrix: [3 axes, 4 legs]
    # Matrix columns: 0=RF, 1=LF, 2=RB, 3=LB (based on _isaac_to_hardware_matrix)

    isaac_leg = joint_idx // 3  # 0=LF, 1=RF, 2=LB, 3=RB
    joint_in_leg = joint_idx % 3  # 0=hip, 1=thigh, 2=calf

    leg_names = ['LF', 'RF', 'LB', 'RB']
    joint_names_in_leg = ['hip', 'thigh', 'calf']

    print(f"\n[STAGE 3] Hardware Matrix Mapping:")
    print(f"  Isaac leg: {isaac_leg} ({leg_names[isaac_leg]})")
    print(f"  Joint in leg: {joint_in_leg} ({joint_names_in_leg[joint_in_leg]})")

    # Based on _isaac_to_hardware_matrix:
    # matrix[:, 1] = angles_real[0:3]   # LF → hw_col 1
    # matrix[:, 0] = angles_real[3:6]   # RF → hw_col 0
    # matrix[:, 3] = angles_real[6:9]   # LB → hw_col 3
    # matrix[:, 2] = angles_real[9:12]  # RB → hw_col 2

    isaac_to_hw_col = {0: 1, 1: 0, 2: 3, 3: 2}  # LF→1, RF→0, LB→3, RB→2
    hw_col = isaac_to_hw_col[isaac_leg]
    hw_row = joint_in_leg

    print(f"  Maps to: matrix[{hw_row}, {hw_col}]")
    print(f"  Value: {angles_real[joint_idx]:+.4f}")

    # Also check hw_to_isaac_leg mapping (used for reading positions)
    print(f"\n[INFO] hw_to_isaac_leg mapping: {controller.hw_to_isaac_leg}")
    print(f"  hw_col {hw_col} → isaac_leg {isaac_leg}")

    # Verify reverse mapping
    for hw_c, isaac_l in controller.hw_to_isaac_leg.items():
        if isaac_l == isaac_leg:
            if hw_c != hw_col:
                print(f"  ⚠️  MISMATCH! hw_to_isaac_leg[{hw_c}]={isaac_l} but we calculated hw_col={hw_col}")

    print(f"\n[EXPECTED PHYSICAL BEHAVIOR]:")
    print(f"  Physical leg: {leg_names[isaac_leg]} (column {hw_col} in hardware)")
    print(f"  Physical joint: {joint_names_in_leg[joint_in_leg]} (row {hw_row} in hardware)")

    if angles_real[joint_idx] > controller.real_default_positions[joint_idx]:
        print(f"  Direction: POSITIVE movement (angle increases)")
    elif angles_real[joint_idx] < controller.real_default_positions[joint_idx]:
        print(f"  Direction: NEGATIVE movement (angle decreases)")
    else:
        print(f"  Direction: NO MOVEMENT (angle unchanged)")

    print(f"\n{'='*70}")


def main():
    print("="*70)
    print("TRANSFORMATION PIPELINE VERIFICATION")
    print("="*70)
    print("""
This script analyzes the transformation pipeline for forward motion joints.

For each joint with significant action in the forward motion CSV, we'll:
1. Show the transformation at each stage
2. Predict which physical joint should move
3. Predict the direction of movement

You can then compare this with what actually happens on the robot.
""")

    input("Press Enter to begin analysis...")

    # Initialize controller
    print("\n[INFO] Initializing controller...")
    controller = FixedMappingControllerV3("/home/ubuntu/mp2_mlp/policy_joyboy.pt")

    # Load forward motion CSV
    csv_path = "/home/ubuntu/debug/obs_action_logs_x_010/env_1_actions.csv"
    print(f"\n[INFO] Analyzing forward motion CSV: {csv_path}")

    import pandas as pd
    try:
        df = pd.read_csv(csv_path)

        joint_names_list = [
            'action_base_lf1', 'action_lf1_lf2', 'action_lf2_lf3',  # LF
            'action_base_rf1', 'action_rf1_rf2', 'action_rf2_rf3',  # RF
            'action_base_lb1', 'action_lb1_lb2', 'action_lb2_lb3',  # LB
            'action_base_rb1', 'action_rb1_rb2', 'action_rb2_rb3',  # RB
        ]

        joint_labels = [
            'LF_hip', 'LF_thigh', 'LF_calf',
            'RF_hip', 'RF_thigh', 'RF_calf',
            'LB_hip', 'LB_thigh', 'LB_calf',
            'RB_hip', 'RB_thigh', 'RB_calf',
        ]

        # Get average actions
        avg_actions = []
        for joint_name in joint_names_list:
            if joint_name in df.columns:
                avg_actions.append(df[joint_name].iloc[10:].mean())
            else:
                avg_actions.append(0.0)

        print("\n[INFO] Average actions from CSV:")
        for i, (label, action) in enumerate(zip(joint_labels, avg_actions)):
            if abs(action) > 0.01:
                print(f"  Joint {i:2d} ({label:12s}): {action:+.4f}")

        # Analyze each joint with significant action
        print("\n" + "="*70)
        print("DETAILED PIPELINE ANALYSIS")
        print("="*70)

        for joint_idx, (label, action) in enumerate(zip(joint_labels, avg_actions)):
            if abs(action) > 0.01:  # Only analyze joints with significant action
                print_joint_transformation(controller, joint_idx, action, label)

                cont = input("\n[Press Enter to continue to next joint, or 'q' to quit]: ").strip().lower()
                if cont == 'q':
                    break

        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("""
Compare the predicted behavior above with what you observe on the robot.

If predictions don't match reality, the issue is likely in:
1. hw_to_isaac_leg mapping (if wrong leg moves)
2. Matrix row/column assignment in _isaac_to_hardware_matrix (if wrong joint moves)
3. joint_direction signs (if correct joint moves but wrong direction)

Your manual sim vs real comparison should align with the joint_direction array.
If it doesn't, there might be additional transformations happening in hardware.
""")

    except Exception as e:
        print(f"[ERROR] Could not load CSV: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
