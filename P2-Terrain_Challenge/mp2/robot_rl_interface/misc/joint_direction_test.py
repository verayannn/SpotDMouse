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
        ctrl.test_single_joint(idx, amplitude=0.9)
        
        result = input("Did it move correctly? (y/n): ").strip().lower()
        if result != 'y':
            print(f"  *** Joint {idx} FAILED - multiplier may be wrong ***")
    
    print("\n" + "="*60)
    print("Verification complete!")
