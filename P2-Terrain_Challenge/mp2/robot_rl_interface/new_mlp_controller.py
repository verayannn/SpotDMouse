import numpy as np
import torch
import time
from collections import deque
from MangDang.mini_pupper.ESP32Interface import ESP32Interface

class WorkingMLPController:
    def __init__(self, policy_path="/home/ubuntu/mp2_mlp/policy_only.pt"):
        self.esp32 = ESP32Interface()
        time.sleep(0.5)
        
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        print(f"Loaded policy from {policy_path}")
        
        # Hardware mapping: Isaac joint index -> ESP32 servo index
        # Isaac order: LF(0,1,2), RF(3,4,5), LB(6,7,8), RB(9,10,11)
        # ESP32 order: RF(0,1,2), LF(3,4,5), RB(6,7,8), LB(9,10,11)
        self.esp32_servo_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        
        # Inverse mapping for writing
        self.isaac_to_esp32 = np.argsort(self.esp32_servo_order)

        # Direction multipliers: real_angle * multiplier = sim_angle
        self.direction_multipliers = np.array([
            -1.0, -1.0, -1.0,  # LF: all inverted
            -1.0,  1.0,  1.0,  # RF: hip inverted, thigh/calf same
             1.0, -1.0, -1.0,  # LB: hip same, thigh/calf inverted
             1.0,  1.0,  1.0,  # RB: all same
        ])
        
        # Isaac default pose (what policy considers "zero" for joint_pos_rel)
        self.isaac_defaults = np.array([
             0.0, 0.785, -1.57,  # LF
             0.0, 0.785, -1.57,  # RF
             0.0, 0.785, -1.57,  # LB
             0.0, 0.785, -1.57   # RB
        ])
        
        # Servo constants
        self.servo_center = 512
        self.servo_scale = 1024 / (2 * np.pi)  # ~163 counts/radian
        
        # From training config
<<<<<<< HEAD
        self.ACTION_SCALE = 0.1
=======
        self.ACTION_SCALE = 0.05
>>>>>>> cb213efbee9a3eb0fa1c574867502ea8483c0e1a
        
        # Calibrate: find servo positions that correspond to Isaac defaults
        print("Calibrating to Isaac default pose...")
        self._calibrate()
        
        # State
        self.prev_actions = np.zeros(12)
        self.prev_joint_pos_rel = np.zeros(12)
        self.prev_joint_vel = np.zeros(12)
        self.prev_time = time.time()
        self.velocity_command = np.zeros(3)
        self.control_active = False
        self.shutdown = False
        
        # IMU calibration
        print("Calibrating IMU...")
        self._calibrate_imu()
        
        self.CONTROL_FREQUENCY = 50
        self.debug_counter = 0
        
    def _calibrate(self):
        """
        Determine the servo positions that correspond to the Isaac default pose.
        When robot is in standing pose, servos should produce isaac_defaults.
        """
        # Read current servo positions (robot should be in standing pose)
        raw_servos = np.array(self.esp32.servos_get_position())
        isaac_ordered = raw_servos[self.esp32_servo_order]
        
        # Convert to radians (hardware frame)
        hardware_radians = (isaac_ordered - self.servo_center) / self.servo_scale
        
        # Convert to sim frame
        sim_radians = hardware_radians * self.direction_multipliers
        
        print(f"Current servo positions (Isaac order): {isaac_ordered}")
        print(f"Hardware radians: {hardware_radians}")
        print(f"Sim frame angles: {sim_radians}")
        print(f"Isaac defaults:   {self.isaac_defaults}")
        
        # The offset between current sim angles and Isaac defaults
        # At standing, sim_radians should equal isaac_defaults
        # If not, we need to account for this offset
        self.calibration_offset = sim_radians - self.isaac_defaults
        print(f"Calibration offset: {self.calibration_offset}")
        print(f"(This should be near zero if robot is in proper standing pose)")
        
    def _calibrate_imu(self, samples=50):
        print("Keep robot still...")
        gyro_sum = np.zeros(3)
        
        for i in range(samples):
            data = self.esp32.imu_get_data()
            gyro_sum += np.array([data['gx'], data['gy'], data['gz']])
            time.sleep(0.02)
            
        self.gyro_offset = gyro_sum / samples
        print(f"Gyro offset: {self.gyro_offset}")
    
    def read_joint_positions(self):
        """Read servo positions and convert to joint_pos_rel (sim frame, relative to defaults)."""
        raw_servos = np.array(self.esp32.servos_get_position())
        isaac_ordered = raw_servos[self.esp32_servo_order]
        
        # To radians (hardware frame)
        hardware_radians = (isaac_ordered - self.servo_center) / self.servo_scale
        
        # To sim frame
        sim_radians = hardware_radians * self.direction_multipliers
        
        # Relative to Isaac defaults (this is what joint_pos_rel means)
        joint_pos_rel = sim_radians - self.isaac_defaults - self.calibration_offset
        
        return joint_pos_rel
    
    def write_joint_positions(self, actions):
        """Convert policy actions to servo commands and send."""
        # Actions are offsets from default pose, scaled
        target_pos_rel = actions * self.ACTION_SCALE
        
        # Absolute position in sim frame
        target_sim = self.isaac_defaults + target_pos_rel
        
        # To hardware frame (multiply by same direction multipliers - they're self-inverse for ±1)
        target_hardware = target_sim * self.direction_multipliers
        
        # To servo units
        servo_positions = target_hardware * self.servo_scale + self.servo_center
        servo_positions = np.clip(servo_positions, 150, 874)  # Safe range
        
        # Reorder for ESP32
        esp32_positions = np.zeros(12)
        for isaac_idx in range(12):
            esp32_idx = self.esp32_servo_order[isaac_idx]
            esp32_positions[esp32_idx] = servo_positions[isaac_idx]
        
        self.esp32.servos_set_position([int(pos) for pos in esp32_positions])
    
    def get_observation(self):
        """Build observation vector for policy."""
        # IMU
        imu_data = self.esp32.imu_get_data()
        
        # Angular velocity (calibrated gyro)
        gyro_raw = np.array([imu_data['gx'], imu_data['gy'], imu_data['gz']])
        base_ang_vel = gyro_raw - self.gyro_offset
        
        # Projected gravity from accelerometer
        accel = np.array([imu_data['ax'], imu_data['ay'], imu_data['az']])
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0.1:
            # Accelerometer measures reaction to gravity, so negate
            projected_gravity = -accel / accel_norm
        else:
            projected_gravity = np.array([0, 0, -1])
        
        # Base linear velocity - zeros (policy should handle this via domain randomization)
        base_lin_vel = np.zeros(3)
        
        # Joint positions (relative to defaults)
        joint_pos_rel = self.read_joint_positions()
        
        # Joint velocities
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt > 0.001:
            joint_vel_raw = (joint_pos_rel - self.prev_joint_pos_rel) / dt
            # Low-pass filter
            alpha = 0.4
            joint_vel = alpha * joint_vel_raw + (1 - alpha) * self.prev_joint_vel
            joint_vel = np.clip(joint_vel, -1.5, 1.5)
        else:
            joint_vel = self.prev_joint_vel
        
        self.prev_joint_pos_rel = joint_pos_rel.copy()
        self.prev_joint_vel = joint_vel.copy()
        self.prev_time = current_time
        
        # Joint efforts - zeros
        joint_effort = np.zeros(12)
        
        # Build observation
        obs = np.concatenate([
            base_lin_vel,           # 3
            base_ang_vel,           # 3
            projected_gravity,      # 3
            self.velocity_command,  # 3
            joint_pos_rel,          # 12
            joint_vel,              # 12
            joint_effort,           # 12
            self.prev_actions       # 12
        ]).astype(np.float32)
        
        return obs, joint_pos_rel, joint_vel
    
    def control_step(self):
        """Single control loop iteration."""
        obs, joint_pos_rel, joint_vel = self.get_observation()
        
        # Run policy
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            actions = self.policy(obs_tensor).squeeze().numpy()
        
        # Store for next observation
        self.prev_actions = actions.copy()
        
        # Send to servos
        self.write_joint_positions(actions)
        
        # Debug output
        self.debug_counter += 1
        if self.debug_counter % 50 == 0:  # Every second at 50Hz
            print(f"\n--- Step {self.debug_counter} ---")
            print(f"Velocity cmd: {self.velocity_command}")
            print(f"Joint pos rel: [{joint_pos_rel.min():.3f}, {joint_pos_rel.max():.3f}]")
            print(f"Joint vel:     [{joint_vel.min():.3f}, {joint_vel.max():.3f}]")
            print(f"Actions:       [{actions.min():.3f}, {actions.max():.3f}]")
            print(f"Sample actions: [{actions[0]:.2f}, {actions[1]:.2f}, {actions[2]:.2f}, {actions[3]:.2f}]")
        
        return actions
    
    def control_loop(self):
        """Main control loop."""
        dt_target = 1.0 / self.CONTROL_FREQUENCY
        
        print(f"\nControl loop started at {self.CONTROL_FREQUENCY}Hz")
        print("Use set_velocity_command(vx, vy, vyaw) to move")
        
        while not self.shutdown:
            loop_start = time.time()
            
            if self.control_active:
                try:
                    self.control_step()
                except Exception as e:
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                    self.control_active = False
            
            elapsed = time.time() - loop_start
            if elapsed < dt_target:
                time.sleep(dt_target - elapsed)
    
    def set_velocity_command(self, vx, vy, vyaw):
        self.velocity_command = np.array([
            np.clip(vx, -0.35, 0.40),
            np.clip(vy, -0.35, 0.35),
            np.clip(vyaw, -0.30, 0.30)
        ])
        if np.any(np.abs(self.velocity_command) > 0.01):
            if not self.control_active:
                # Reset state when starting
                self.prev_actions = np.zeros(12)
                self.prev_joint_pos_rel = self.read_joint_positions()
                self.prev_joint_vel = np.zeros(12)
                self.prev_time = time.time()
            self.control_active = True
            print(f"Active: cmd={self.velocity_command}")
        else:
            self.control_active = False
            print("Stopped")
    
    def stop(self):
        self.control_active = False
        self.shutdown = True


if __name__ == "__main__":
    import threading
    
    controller = WorkingMLPController("/home/ubuntu/mp2_mlp/policy_only.pt")
    
    thread = threading.Thread(target=controller.control_loop)
    thread.start()
    
    print("\nCommands: w=forward, s=back, a=left, d=right, q/e=turn, space=stop, x=exit")
    
    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == 'w':
                controller.set_velocity_command(0.15, 0, 0)
            elif cmd == 's':
                controller.set_velocity_command(-0.15, 0, 0)
            elif cmd == 'a':
                controller.set_velocity_command(0, 0.15, 0)
            elif cmd == 'd':
                controller.set_velocity_command(0, -0.15, 0)
            elif cmd == 'q':
                controller.set_velocity_command(0, 0, 0.15)
            elif cmd == 'e':
                controller.set_velocity_command(0, 0, -0.15)
            elif cmd == ' ' or cmd == '':
                controller.set_velocity_command(0, 0, 0)
            elif cmd == 'x':
                break
    except KeyboardInterrupt:
        pass
    
    controller.stop()
    thread.join()
    print("Done")
