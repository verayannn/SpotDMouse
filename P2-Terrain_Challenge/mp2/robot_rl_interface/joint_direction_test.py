# from MangDang.mini_pupper.ESP32Interface import ESP32Interface
# import torch
# import numpy as np
# import time
# from mp2_mlp_controller import FinalMLPController as MP2IsaacRLController

# class JointDirectionTester:
#     def __init__(self):
#         self.esp32 = ESP32Interface()
#         time.sleep(0.5)
        
#         # Load the trained policy
#         self.controller = MP2IsaacRLController()
        
#         # Servo calibration
#         self.servo_offset = 512 
#         self.servo_scale = 1024 / (2 * np.pi)
        
#         # Mapping arrays
#         self.esp32_servo_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
#         self.isaac_to_esp32 = np.zeros(12, dtype=int)
#         for i, esp_idx in enumerate(self.esp32_servo_order):
#             self.isaac_to_esp32[esp_idx] = i
        
#         # Isaac default positions (standing pose)
#         self.isaac_defaults = np.array([
#             -0.1, 0.8, -1.5,  # LF
#              0.1, 0.8, -1.5,  # RF
#             -0.1, 0.8, -1.5,  # LB
#              0.1, 0.8, -1.5   # RB
#         ])
        
#         # Store initial positions
#         self.initial_positions = np.array(self.esp32.servos_get_position())
        
#         # Joint names for clarity
#         self.isaac_joint_names = [
#             "LF hip", "LF thigh", "LF calf",
#             "RF hip", "RF thigh", "RF calf",
#             "LB hip", "LB thigh", "LB calf",
#             "RB hip", "RB thigh", "RB calf"
#         ]
    
#     def test_individual_joint_directions(self):
#         """Test each joint's positive/negative direction"""
#         print("\n" + "="*60)
#         print("TESTING INDIVIDUAL JOINT DIRECTIONS")
#         print("="*60)
#         print("This will move each joint in positive and negative directions")
#         print("Verify the movements match Isaac Sim conventions:")
#         print("- Hip: Positive = abduction (outward), Negative = adduction (inward)")
#         print("- Thigh: Positive = flexion (forward), Negative = extension (backward)")
#         print("- Calf: Positive = extension (straighten), Negative = flexion (bend)")
        
#         for isaac_idx in range(12):
#             joint_name = self.isaac_joint_names[isaac_idx]
#             esp32_idx = self.isaac_to_esp32[isaac_idx]
            
#             print(f"\n{'-'*40}")
#             print(f"Testing {joint_name} (Isaac idx {isaac_idx} -> ESP32 idx {esp32_idx})")
            
#             # Get current position in radians
#             current_servo = self.initial_positions[esp32_idx]
#             current_rad = (current_servo - self.servo_offset) / self.servo_scale
            
#             # Test positive direction
#             input(f"Press Enter to test POSITIVE direction (+0.3 rad)...")
#             test_rad_pos = current_rad + 0.3
#             test_servo_pos = int(test_rad_pos * self.servo_scale + self.servo_offset)
#             test_positions = self.initial_positions.copy()
#             test_positions[esp32_idx] = np.clip(test_servo_pos, 100, 924)
#             self.esp32.servos_set_position(test_positions)
#             time.sleep(1.0)
            
#             # Test negative direction
#             input(f"Press Enter to test NEGATIVE direction (-0.3 rad)...")
#             test_rad_neg = current_rad - 0.3
#             test_servo_neg = int(test_rad_neg * self.servo_scale + self.servo_offset)
#             test_positions = self.initial_positions.copy()
#             test_positions[esp32_idx] = np.clip(test_servo_neg, 100, 924)
#             self.esp32.servos_set_position(test_positions)
#             time.sleep(1.0)
            
#             # Return to initial
#             input("Press Enter to return to initial position...")
#             self.esp32.servos_set_position(self.initial_positions)
#             time.sleep(0.3)
    
#     def test_policy_actions(self):
#         """Test how policy actions affect joints"""
#         print("\n" + "="*60)
#         print("TESTING POLICY ACTION DIRECTIONS")
#         print("="*60)
#         print("This will test how the policy's output actions move joints")
        
#         # Create a zero observation (standing still)
#         obs = self.controller._get_safe_observation()
        
#         # Test different velocity commands
#         test_commands = [
#             ([0.3, 0.0, 0.0], "Forward walk"),
#             ([-0.3, 0.0, 0.0], "Backward walk"),
#             ([0.0, 0.2, 0.0], "Strafe right"),
#             ([0.0, -0.2, 0.0], "Strafe left"),
#             ([0.0, 0.0, 0.2], "Turn right"),
#             ([0.0, 0.0, -0.2], "Turn left")
#         ]
        
#         for cmd, description in test_commands:
#             print(f"\n{'-'*40}")
#             print(f"Testing: {description}")
#             print(f"Command: vx={cmd[0]}, vy={cmd[1]}, vyaw={cmd[2]}")
            
#             # Set velocity command in observation
#             obs[9:12] = cmd  # velocity_command indices
            
#             # Run through policy
#             with torch.no_grad():
#                 obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
#                 raw_actions = self.controller.policy(obs_tensor).squeeze().numpy()
            
#             # Scale actions
#             scaled_actions = raw_actions * 0.25
            
#             # Show which joints are being activated
#             print("\nPolicy action outputs (scaled):")
#             for i in range(12):
#                 if abs(scaled_actions[i]) > 0.05:  # Only show significant actions
#                     direction = "+" if scaled_actions[i] > 0 else "-"
#                     print(f"  {self.isaac_joint_names[i]}: {direction}{abs(scaled_actions[i]):.3f}")
            
#             input(f"\nPress Enter to apply these actions...")
            
#             # Apply actions
#             absolute_positions = scaled_actions + self.isaac_defaults
#             servo_positions = absolute_positions * self.servo_scale + self.servo_offset
#             servo_positions = np.clip(servo_positions, 100, 924)
            
#             # Reorder for ESP32
#             esp32_positions = servo_positions[self.isaac_to_esp32]
#             self.esp32.servos_set_position([int(pos) for pos in esp32_positions])
            
#             time.sleep(2.0)
            
#             # Return to initial
#             print("Returning to initial position...")
#             self.esp32.servos_set_position(self.initial_positions)
#             time.sleep(0.5)
    
#     def test_standing_position(self):
#         """Test if Isaac's default standing position matches robot's standing"""
#         print("\n" + "="*60)
#         print("TESTING STANDING POSITION")
#         print("="*60)
        
#         input("Press Enter to move to Isaac Sim's default standing position...")
        
#         # Convert Isaac defaults to servo positions
#         servo_positions = self.isaac_defaults * self.servo_scale + self.servo_offset
#         servo_positions = np.clip(servo_positions, 100, 924)
        
#         # Reorder for ESP32
#         esp32_positions = servo_positions[self.isaac_to_esp32]
        
#         print("\nIsaac default angles (rad):", self.isaac_defaults)
#         print("ESP32 servo positions:", [int(pos) for pos in esp32_positions])
        
#         self.esp32.servos_set_position([int(pos) for pos in esp32_positions])
        
#         input("\nObserve if robot is in a proper standing position. Press Enter to continue...")
        
#         # Return to initial
#         self.esp32.servos_set_position(self.initial_positions)

# def main():
#     tester = JointDirectionTester()
    
#     while True:
#         print("\n" + "="*60)
#         print("JOINT DIRECTION TEST MENU")
#         print("="*60)
#         print("1. Test individual joint directions")
#         print("2. Test policy action directions")
#         print("3. Test standing position")
#         print("4. Return to home position")
#         print("q. Quit")
        
#         choice = input("\nEnter choice: ").strip().lower()
        
#         if choice == '1':
#             tester.test_individual_joint_directions()
#         elif choice == '2':
#             tester.test_policy_actions()
#         elif choice == '3':
#             tester.test_standing_position()
#         elif choice == '4':
#             print("Returning to initial position...")
#             tester.esp32.servos_set_position(tester.initial_positions)
#         elif choice == 'q':
#             break
#         else:
#             print("Invalid choice")
    
#     print("\nReturning to initial position...")
#     tester.esp32.servos_set_position(tester.initial_positions)
#     print("Done!")

# if __name__ == "__main__":
#     main()


import numpy as np
import torch
import time
from MangDang.mini_pupper.ESP32Interface import ESP32Interface

class VerifiedMLPController:
    def __init__(self, policy_path="/home/ubuntu/mp2_mlp/policy_only.pt"):
        self.esp32 = ESP32Interface()
        time.sleep(0.5)
        
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        
        # ESP32 servo index for each Isaac joint index
        # Isaac: [LF0,LF1,LF2, RF0,RF1,RF2, LB0,LB1,LB2, RB0,RB1,RB2]
        # ESP32: [RF0,RF1,RF2, LF0,LF1,LF2, RB0,RB1,RB2, LB0,LB1,LB2]
        self.esp32_servo_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        
        # Direction multipliers: to convert real angles to sim angles (and vice versa)
        self.direction_multipliers = np.array([
            -1.0, -1.0, -1.0,  # LF
            -1.0,  1.0,  1.0,  # RF
             1.0, -1.0, -1.0,  # LB
             1.0,  1.0,  1.0,  # RB
        ])
        
        self.isaac_defaults = np.array([
             0.0, 0.785, -1.57,
             0.0, 0.785, -1.57,
             0.0, 0.785, -1.57,
             0.0, 0.785, -1.57
        ])
        
        self.servo_center = 512
        self.servo_scale = 1024 / (2 * np.pi)
        
        self.ACTION_SCALE = 0.1  # Start small!
        
        self.prev_actions = np.zeros(12)
        self.prev_joint_pos_rel = np.zeros(12)
        self.prev_joint_vel = np.zeros(12)
        self.prev_time = time.time()
        self.velocity_command = np.zeros(3)
        
        self._calibrate_imu()
        
    def _calibrate_imu(self, samples=30):
        print("Calibrating IMU (keep still)...")
        gyro_sum = np.zeros(3)
        for _ in range(samples):
            data = self.esp32.imu_get_data()
            gyro_sum += np.array([data['gx'], data['gy'], data['gz']])
            time.sleep(0.02)
        self.gyro_offset = gyro_sum / samples
        print(f"Gyro offset: {self.gyro_offset}")
        
    def read_joints(self):
        """Read servo positions, return joint_pos_rel in sim frame."""
        raw = np.array(self.esp32.servos_get_position())
        
        # Reorder to Isaac order
        isaac_ordered = raw[self.esp32_servo_order]
        
        # Convert to radians (hardware frame)
        hw_radians = (isaac_ordered - self.servo_center) / self.servo_scale
        
        # Convert to sim frame relative position
        # At standing, hw_radians ≈ 0, and we want joint_pos_rel = 0
        joint_pos_rel = hw_radians * self.direction_multipliers
        
        return joint_pos_rel
    
    def write_joints(self, actions):
        """Write actions to servos."""
        # Actions -> relative target positions
        target_rel = actions * self.ACTION_SCALE
        
        # Convert to hardware frame
        hw_radians = target_rel * self.direction_multipliers
        
        # Convert to servo units
        servo_pos = hw_radians * self.servo_scale + self.servo_center
        servo_pos = np.clip(servo_pos, 150, 874)
        
        # Create ESP32-ordered output
        esp32_out = np.zeros(12)
        esp32_out[self.esp32_servo_order] = servo_pos
        
        self.esp32.servos_set_position([int(p) for p in esp32_out])
        
    def get_observation(self):
        imu = self.esp32.imu_get_data()
        
        base_ang_vel = np.array([imu['gx'], imu['gy'], imu['gz']]) - self.gyro_offset
        
        accel = np.array([imu['ax'], imu['ay'], imu['az']])
        norm = np.linalg.norm(accel)
        projected_gravity = -accel / norm if norm > 0.1 else np.array([0, 0, -1])
        
        joint_pos_rel = self.read_joints()
        
        dt = time.time() - self.prev_time
        if dt > 0.001:
            vel_raw = (joint_pos_rel - self.prev_joint_pos_rel) / dt
            joint_vel = 0.4 * vel_raw + 0.6 * self.prev_joint_vel
            joint_vel = np.clip(joint_vel, -1.5, 1.5)
        else:
            joint_vel = self.prev_joint_vel
            
        self.prev_joint_pos_rel = joint_pos_rel.copy()
        self.prev_joint_vel = joint_vel.copy()
        self.prev_time = time.time()
        
        obs = np.concatenate([
            np.zeros(3),            # base_lin_vel
            base_ang_vel,           # base_ang_vel
            projected_gravity,      # projected_gravity
            self.velocity_command,  # velocity_cmd
            joint_pos_rel,          # joint_pos
            joint_vel,              # joint_vel
            np.zeros(12),           # joint_effort
            self.prev_actions       # prev_actions
        ]).astype(np.float32)
        
        return obs, joint_pos_rel
    
    def step(self):
        obs, pos_rel = self.get_observation()
        
        with torch.no_grad():
            actions = self.policy(torch.tensor(obs).unsqueeze(0)).squeeze().numpy()
        
        self.prev_actions = actions.copy()
        self.write_joints(actions)
        
        return actions, pos_rel
    
    def test_single_joint(self, joint_idx, amplitude=0.2):
        """Test a single joint to verify direction."""
        print(f"\nTesting joint {joint_idx} with amplitude {amplitude}")
        print("Watch the robot and verify the motion direction...")
        
        for t in range(100):
            # Sine wave on one joint
            actions = np.zeros(12)
            actions[joint_idx] = amplitude * np.sin(t * 0.1)
            
            self.write_joints(actions)
            time.sleep(0.02)
            
            if t % 25 == 0:
                print(f"  t={t}: action[{joint_idx}] = {actions[joint_idx]:.3f}")
        
        # Return to neutral
        self.write_joints(np.zeros(12))
        print("Done. Did the joint move in the expected direction?")


if __name__ == "__main__":
    ctrl = VerifiedMLPController("/home/ubuntu/mp2_mlp/policy_only.pt")
    
    print("\n" + "="*60)
    print("JOINT DIRECTION VERIFICATION")
    print("="*60)
    print("We'll test each joint one at a time.")
    print("For each, verify it moves in the SIMULATION direction.")
    print("="*60)
    
    joint_names = [
        "LF hip (base_lf1) - should: +action = INWARD (adduction)",
        "LF thigh (lf1_lf2) - should: +action = BACKWARD (extension)", 
        "LF calf (lf2_lf3) - should: +action = EXTENDS (straightens)",
        "RF hip (base_rf1) - should: +action = OUTWARD (abduction)",
        "RF thigh (rf1_rf2) - should: +action = BACKWARD (extension)",
        "RF calf (rf2_rf3) - should: +action = EXTENDS (straightens)",
        "LB hip (base_lb1) - should: +action = INWARD (adduction)",
        "LB thigh (lb1_lb2) - should: +action = BACKWARD (extension)",
        "LB calf (lb2_lb3) - should: +action = EXTENDS (straightens)",
        "RB hip (base_rb1) - should: +action = OUTWARD (abduction)",
        "RB thigh (rb1_rb2) - should: +action = BACKWARD (extension)",
        "RB calf (rb2_rb3) - should: +action = EXTENDS (straightens)",
    ]
    
    for idx, name in enumerate(joint_names):
        print(f"\n--- Joint {idx}: {name} ---")
        input("Press Enter to test this joint...")
        ctrl.test_single_joint(idx, amplitude=0.3)
        
        result = input("Did it move correctly? (y/n): ").strip().lower()
        if result != 'y':
            print(f"  *** Joint {idx} FAILED - multiplier may be wrong ***")
    
    print("\n" + "="*60)
    print("Verification complete!")