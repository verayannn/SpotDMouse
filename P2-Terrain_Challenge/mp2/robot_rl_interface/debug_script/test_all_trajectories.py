"""
Test All Simulation Trajectories
=================================
Replays all simulation action logs to validate joint transformations
across different velocity commands (x, y, z directions, positive and negative)

This helps identify:
1. If transformations work correctly for all movement directions
2. If certain directions have transformation errors
3. Baseline for comparing live RL policy performance
"""

import numpy as np
import time
import os
from mlp_controller_v3 import FixedMappingControllerV3


def test_trajectory(controller, csv_path, description, max_steps=150):
    """Test a single trajectory from CSV."""
    import pandas as pd

    print("\n" + "="*70)
    print(f"TESTING: {description}")
    print(f"CSV: {csv_path}")
    print("="*70)

    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return False

    try:
        sim_actions = pd.read_csv(csv_path)
        print(f"[INFO] Loaded {len(sim_actions)} steps from CSV")
    except Exception as e:
        print(f"[ERROR] Could not load CSV: {e}")
        return False

    print("[INFO] Starting replay in 3 seconds...")
    print("        Watch the robot's movement direction!")
    time.sleep(3)

    start_time = time.time()

    for i in range(min(max_steps, len(sim_actions))):
        row = sim_actions.iloc[i]
        actions = np.array([
            row['action_base_lf1'], row['action_lf1_lf2'], row['action_lf2_lf3'],
            row['action_base_rf1'], row['action_rf1_rf2'], row['action_rf2_rf3'],
            row['action_base_lb1'], row['action_lb1_lb2'], row['action_lb2_lb3'],
            row['action_base_rb1'], row['action_rb1_rb2'], row['action_rb2_rb3'],
        ])

        # Apply with sim ACTION_SCALE = 0.5
        target_sim = controller.sim_default_positions + actions * 0.5
        target_sim = np.clip(target_sim, controller.joint_lower_limits,
                            controller.joint_upper_limits)

        target_matrix = controller._isaac_to_hardware_matrix(target_sim)
        controller.hardware.set_actuator_postions(target_matrix)

        time.sleep(0.02)  # 50Hz

        if i % 25 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {i:3d}/{max_steps}: "
                  f"actions=[{actions.min():+.2f}, {actions.max():+.2f}] "
                  f"t={elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"[DONE] Completed {max_steps} steps in {total_time:.1f}s")
    print(f"[INFO] Robot should have moved in expected direction")
    print(f"[INFO] Returning to default stance...")
    time.sleep(2)

    # Return to default
    target_matrix = controller._isaac_to_hardware_matrix(controller.sim_default_positions)
    controller.hardware.set_actuator_postions(target_matrix)
    time.sleep(1)

    return True


def main():
    print("\n" + "="*70)
    print("SIMULATION TRAJECTORY REPLAY TEST SUITE")
    print("="*70)
    print("""
This script will replay simulation actions from all 6 directions:
  1. X+  (forward)
  2. X-  (backward)
  3. Y+  (strafe left)
  4. Y-  (strafe right)
  5. Z+  (turn left)
  6. Z-  (turn right)

Expected behavior:
  - Robot should move in the labeled direction
  - Movement should be smooth and coordinated
  - If direction is wrong, there may be a frame transformation error

Press Enter after each test to continue to the next one.
""")

    input("Press Enter to begin tests...")

    # Initialize controller
    controller = FixedMappingControllerV3("/home/ubuntu/mp2_mlp/policy_joyboy.pt")

    # Define test cases
    base_dir = "/home/ubuntu/debug"
    tests = [
        {
            "csv": f"{base_dir}/obs_action_logs_x_030/env_0_actions.csv",
            "desc": "Forward (X+, vx=+0.3)",
            "expected": "Robot should walk FORWARD"
        },
        {
            "csv": f"{base_dir}/obs_action_logs_x_n030/env_0_actions.csv",
            "desc": "Backward (X-, vx=-0.3)",
            "expected": "Robot should walk BACKWARD"
        },
        {
            "csv": f"{base_dir}/obs_action_logs_y_030/env_0_actions.csv",
            "desc": "Strafe Left (Y+, vy=+0.3)",
            "expected": "Robot should strafe LEFT"
        },
        {
            "csv": f"{base_dir}/obs_action_logs_y_n030/env_0_actions.csv",
            "desc": "Strafe Right (Y-, vy=-0.3)",
            "expected": "Robot should strafe RIGHT"
        },
        {
            "csv": f"{base_dir}/obs_action_logs_z_030/env_0_actions.csv",
            "desc": "Turn Left (Z+, vyaw=+0.3)",
            "expected": "Robot should turn LEFT (CCW)"
        },
        {
            "csv": f"{base_dir}/obs_action_logs_z_n030/env_0_actions.csv",
            "desc": "Turn Right (Z-, vyaw=-0.3)",
            "expected": "Robot should turn RIGHT (CW)"
        },
    ]

    results = []

    # Run all tests
    for i, test in enumerate(tests, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/6: {test['desc']}")
        print(f"EXPECTED: {test['expected']}")
        print(f"{'='*70}")

        success = test_trajectory(controller, test['csv'], test['desc'])
        results.append({
            "test": test['desc'],
            "success": success,
            "expected": test['expected']
        })

        if i < len(tests):
            print("\n[PAUSE] Observe the robot's movement.")
            input("Did the robot move as expected? Press Enter to continue...")

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for i, result in enumerate(results, 1):
        status = "✓ LOADED" if result['success'] else "✗ FAILED"
        print(f"{i}. {result['test']:<30} {status}")
        print(f"   Expected: {result['expected']}")

    print("\n" + "="*70)
    print("ANALYSIS QUESTIONS:")
    print("="*70)
    print("""
1. Did ALL tests move the robot smoothly and coordinated?
   - If YES: Joint transformations are CORRECT
   - If NO: Which test failed?

2. Did the robot move in the EXPECTED direction for each test?
   - If YES: Frame orientations are CORRECT
   - If NO: Which directions were wrong?
     * If X+/X- are swapped: X axis is flipped
     * If Y+/Y- are swapped: Y axis is flipped
     * If turns are swapped: Yaw axis is flipped
     * If forward goes left: X/Y axes are swapped

3. Compare to live RL policy performance:
   - Sim replay: smooth, coordinated, directional
   - Live policy: thrashing, tilting, non-directional
   - This suggests OBSERVATION mismatch, not action transformation!

Next steps:
- If sim replay works perfectly → Problem is in observations
- If sim replay has issues → Problem is in transformations
""")

    print("\n[DONE] All tests complete")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Test suite cancelled")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
