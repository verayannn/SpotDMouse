import numpy as np
import torch
import time
from collections import deque
from MangDang.mini_pupper.ESP32Interface import ESP32Interface

class FinalMLPController:
    def __init__(self, policy_path="/home/ubuntu/mp2_mlp/policy_only.pt"):
        """Initialize the complete MLP controller with all corrections"""
        self.esp32 = ESP32Interface()
        time.sleep(0.5)
        
        # Test IMU connection
        imu_test = self.esp32.imu_get_data()
        print(f"IMU test data: {imu_test}")
        
        # Load trained policy
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        print(f"Loaded policy from {policy_path}")
        
        # === Hardware Mapping Layer ===
        
        # Joint order mapping (Isaac [LF,RF,LB,RB] -> ESP32 [FR,FL,BR,BL])
        self.esp32_servo_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        self.isaac_to_esp32 = np.zeros(12, dtype=int)
        for i, esp_idx in enumerate(self.esp32_servo_order):
            self.isaac_to_esp32[esp_idx] = i
        
        # Direction multipliers (from hardware testing)
        # self.joint_direction_multipliers = np.array([
        #     -1.0,  1.0,  1.0,  # LF: hip inward, thigh/calf forward positive
        #      1.0, -1.0, -1.0,  # RF: hip outward, thigh/calf forward negative
        #     -1.0,  1.0,  1.0,  # LB: hip inward, thigh/calf forward positive
        #     -1.0, -1.0, -1.0,  # RB: hip outward, thigh/calf forward negative
        # ])

                # Direction multipliers (based on real hardware vs simulation mapping)
        # If real and sim have same positive direction: 1.0, if opposite: -1.0
        self.joint_direction_multipliers = np.array([
            -1.0, -1.0, -1.0,  # LF: hip (real outward+ vs sim inward+), thigh (real forward+ vs sim backward+), calf (real flex+ vs sim extend+)
            -1.0,  1.0,  1.0,  # RF: hip (real inward+ vs sim outward+), thigh (both backward+), calf (both extend+)  
             1.0, -1.0, -1.0,  # LB: hip (both inward+), thigh (real forward+ vs sim backward+), calf (real flex+ vs sim extend+)
             1.0,  1.0,  1.0,  # RB: hip (both outward+), thigh (both backward+), calf (both extend+)
        ])
        
        # === Reference Frame Translation ===
        
        # Isaac Sim training defaults (what the policy expects)
        self.isaac_training_defaults = np.array([
            -0.1, 0.8, -1.5,  # LF
             0.1, 0.8, -1.5,  # RF
            -0.1, 0.8, -1.5,  # LB
             0.1, 0.8, -1.5   # RB
        ])
        
        # Get hardware's actual standing position
        print("Reading hardware standing position...")
        standing_servos = np.array(self.esp32.servos_get_position())
        isaac_ordered_standing = standing_servos[self.esp32_servo_order]
        standing_radians = (isaac_ordered_standing - 512) / (1024 / (2 * np.pi))
        self.hardware_standing_angles = standing_radians / self.joint_direction_multipliers
        print(f"Hardware standing angles: {self.hardware_standing_angles}")
        
        # === Servo Calibration ===
        self.servo_offset = 512
        self.servo_scale = 1024 / (2 * np.pi)
        
        # === Action Processing ===
        self.ACTION_SCALE = 0.1
        self.MAX_ACTION_CHANGE = 0.05
        self.prev_actions = np.zeros(12)
        
        # === State Tracking ===
        self.prev_positions = self.hardware_standing_angles.copy()
        self.prev_time = time.time()
        self.velocity_command = np.zeros(3)
        self.control_active = False
        self.shutdown = False
        
        # === IMU Calibration ===
        print("Calibrating IMU...")
        self.calibrate_imu()
        
        # === Control Loop Timing ===
        self.CONTROL_FREQUENCY = 50  # Hz
        self.loop_times = deque(maxlen=50)
        
    def calibrate_imu(self, samples=50):
        """Calibrate IMU offsets when stationary"""
        print("Keep robot still for IMU calibration...")
        
        accel_sum = np.zeros(3)
        gyro_sum = np.zeros(3)
        
        for i in range(samples):
            data = self.esp32.imu_get_data()
            accel_sum[0] += data['ax']
            accel_sum[1] += data['ay']
            accel_sum[2] += data['az']
            gyro_sum[0] += data['gx']
            gyro_sum[1] += data['gy']
            gyro_sum[2] += data['gz']
            time.sleep(0.02)
            
            if i % 10 == 0:
                print(f"  Calibration {i}/{samples}")
        
        # Calculate offsets
        self.accel_offset = accel_sum / samples
        self.gyro_offset = gyro_sum / samples
        
        # Gravity should be ~9.81 on z-axis when upright
        expected_gravity = 9.81
        actual_gravity = np.linalg.norm(self.accel_offset)
        self.gravity_scale = expected_gravity / actual_gravity if actual_gravity > 0 else 1.0
        
        print(f"IMU calibration complete:")
        print(f"  Accel offset: {self.accel_offset}")
        print(f"  Gyro offset: {self.gyro_offset}")
        print(f"  Gravity scale: {self.gravity_scale}")
    
    def get_observation(self):
        """Get complete observation with real sensor data"""
        try:
            # === IMU Data ===
            imu_data = self.esp32.imu_get_data()
            
            # Apply calibration to accelerometer
            accel_raw = np.array([imu_data['ax'], imu_data['ay'], imu_data['az']])
            accel_calibrated = (accel_raw - self.accel_offset) * self.gravity_scale
            
            # Apply calibration to gyroscope
            gyro_raw = np.array([imu_data['gx'], imu_data['gy'], imu_data['gz']])
            gyro_calibrated = gyro_raw - self.gyro_offset
            
            # Calculate projected gravity (normalized acceleration vector)
            accel_norm = np.linalg.norm(accel_calibrated)
            if accel_norm > 0.1:
                projected_gravity = accel_calibrated / accel_norm
            else:
                projected_gravity = np.array([0, 0, -1])
            
            # === Joint Data ===
            raw_positions = np.array(self.esp32.servos_get_position())
            raw_loads = np.array(self.esp32.servos_get_load())
            
            # Reorder to Isaac format
            isaac_positions = raw_positions[self.esp32_servo_order]
            isaac_loads = raw_loads[self.esp32_servo_order]
            
            # Convert positions to radians
            hardware_radians = (isaac_positions - self.servo_offset) / self.servo_scale
            
            # Apply direction corrections to get true angles
            true_angles = hardware_radians / self.joint_direction_multipliers
            
            # Transform to Isaac's training reference frame
            # This is critical: MLP expects positions relative to its training defaults
            isaac_relative_positions = (true_angles - self.hardware_standing_angles + 
                                       self.isaac_training_defaults)
            
            # Calculate joint velocities
            current_time = time.time()
            dt = current_time - self.prev_time
            if dt > 0.001:  # Avoid division by very small numbers
                joint_velocities = (isaac_relative_positions - self.prev_positions) / dt
                # Apply smoothing to reduce noise
                joint_velocities = np.clip(joint_velocities, -10, 10)
            else:
                joint_velocities = np.zeros(12)
            
            # Update state
            self.prev_positions = isaac_relative_positions.copy()
            self.prev_time = current_time
            
            # Normalize joint efforts
            joint_efforts = np.clip(isaac_loads / 500.0, -10, 10)
            
            # === Build Complete Observation (60 dims) ===
            obs = np.concatenate([
                accel_calibrated,           # 0:3 - base linear acceleration (from IMU)
                gyro_calibrated,            # 3:6 - base angular velocity (from IMU)
                projected_gravity,          # 6:9 - gravity direction (from IMU)
                self.velocity_command,      # 9:12 - commanded velocities
                isaac_relative_positions,   # 12:24 - joint positions (Isaac frame)
                joint_velocities,           # 24:36 - joint velocities
                joint_efforts,              # 36:48 - joint efforts/torques
                self.prev_actions           # 48:60 - previous actions
            ])
            
            return obs
            
        except Exception as e:
            print(f"Error getting observation: {e}")
            # Return safe observation on error
            return self._get_safe_observation()
    
    def _get_safe_observation(self):
        """Return a safe default observation if sensors fail"""
        return np.concatenate([
            np.zeros(3),                # base_lin_vel
            np.zeros(3),                # base_ang_vel
            np.array([0, 0, -1]),       # projected_gravity
            self.velocity_command,       # commands
            self.isaac_training_defaults,# joint_pos (at defaults)
            np.zeros(12),               # joint_vel
            np.zeros(12),               # joint_effort
            self.prev_actions           # prev_actions
        ])
    
    def process_actions(self, mlp_actions):
        """Convert MLP actions to hardware servo commands"""
        # Scale actions (MLP outputs are typically normalized)
        scaled = mlp_actions * self.ACTION_SCALE
        
        # Apply per-joint limits based on training
        limited = scaled.copy()
        for i in range(4):  # 4 legs
            base_idx = i * 3
            # Hip: smaller range
            limited[base_idx] = np.clip(limited[base_idx], -0.5, 0.5)
            # Thigh: medium range
            limited[base_idx + 1] = np.clip(limited[base_idx + 1], -1.0, 1.0)
            # Calf: medium range
            limited[base_idx + 2] = np.clip(limited[base_idx + 2], -1.0, 1.0)
        
        # Smooth actions to prevent jerky movements
        action_delta = limited - self.prev_actions
        action_delta = np.clip(action_delta, -self.MAX_ACTION_CHANGE, self.MAX_ACTION_CHANGE)
        smoothed = self.prev_actions + action_delta
        
        # Update for next iteration
        self.prev_actions = smoothed.copy()
        
        # === Reference Frame Translation ===
        # MLP outputs are relative to Isaac training defaults
        isaac_absolute = smoothed + self.isaac_training_defaults
        
        # Transform from Isaac frame to hardware frame
        # 1. Remove Isaac's training standing pose
        # 2. Add hardware's actual standing pose
        hardware_angles = (isaac_absolute - self.isaac_training_defaults + 
                          self.hardware_standing_angles)
        
        # Apply direction corrections for hardware
        hardware_corrected = hardware_angles * self.joint_direction_multipliers
        
        # Convert to servo positions
        servo_positions = hardware_corrected * self.servo_scale + self.servo_offset
        servo_positions = np.clip(servo_positions, 100, 924)
        
        # Reorder for ESP32 interface
        esp32_positions = servo_positions[self.isaac_to_esp32]
        
        return [int(pos) for pos in esp32_positions]
    
    def control_loop(self):
        """Main control loop running at specified frequency"""
        dt_target = 1.0 / self.CONTROL_FREQUENCY
        
        print(f"\nControl loop started at {self.CONTROL_FREQUENCY}Hz")
        print("Waiting for velocity commands...")
        print("Use set_velocity_command(vx, vy, vyaw) to control robot")
        
        while not self.shutdown:
            loop_start = time.time()
            
            try:
                if self.control_active:
                    # Get observation with all sensor data
                    obs = self.get_observation()
                    
                    # Run policy inference
                    with torch.no_grad():
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                        raw_actions = self.policy(obs_tensor).squeeze().numpy()
                    
                    # Process actions and send to hardware
                    servo_commands = self.process_actions(raw_actions)
                    self.esp32.servos_set_position_torque(servo_commands, [1]*12)
                    
                    # Performance monitoring
                    if len(self.loop_times) == self.loop_times.maxlen:
                        avg_time = np.mean(self.loop_times)
                        if int(loop_start) % 10 == 0:  # Print every 10 seconds
                            print(f"Loop: {1/avg_time:.1f}Hz | "
                                  f"Cmd: [{self.velocity_command[0]:.2f}, "
                                  f"{self.velocity_command[1]:.2f}, "
                                  f"{self.velocity_command[2]:.2f}] | "
                                  f"Actions: [{np.min(raw_actions):.2f}, "
                                  f"{np.max(raw_actions):.2f}]")
                
                else:
                    # When not active, just maintain standing position
                    standing_servos = self.hardware_standing_angles * self.joint_direction_multipliers
                    servo_positions = standing_servos * self.servo_scale + self.servo_offset
                    servo_positions = np.clip(servo_positions, 100, 924)
                    esp32_positions = servo_positions[self.isaac_to_esp32]
                    self.esp32.servos_set_position([int(pos) for pos in esp32_positions])
                    
            except Exception as e:
                print(f"Control loop error: {e}")
                import traceback
                traceback.print_exc()
                self.control_active = False
            
            # Maintain loop frequency
            elapsed = time.time() - loop_start
            self.loop_times.append(elapsed)
            
            if elapsed < dt_target:
                time.sleep(dt_target - elapsed)
        
        print("Control loop stopped")
    
    def set_velocity_command(self, vx, vy, vyaw):
        """Set velocity command and activate control"""
        self.velocity_command = np.array([
            np.clip(vx, -0.35, 0.40),   # Forward/backward
            np.clip(vy, -0.35, 0.35),   # Left/right
            np.clip(vyaw, -0.30, 0.30)  # Rotation
        ])
        
        # Activate control on non-zero command
        if np.any(self.velocity_command != 0):
            self.control_active = True
            print(f"Control activated with command: {self.velocity_command}")
        else:
            self.control_active = False
            print("Control deactivated (zero command)")
    
    def stop(self):
        """Stop the controller"""
        print("Stopping controller...")
        self.control_active = False
        self.shutdown = True
        time.sleep(0.1)
        
        # Return to standing position
        print("Returning to standing position...")
        standing_servos = self.hardware_standing_angles * self.joint_direction_multipliers
        servo_positions = standing_servos * self.servo_scale + self.servo_offset
        servo_positions = np.clip(servo_positions, 100, 924)
        esp32_positions = servo_positions[self.isaac_to_esp32]
        self.esp32.servos_set_position([int(pos) for pos in esp32_positions])


# Example usage and test
if __name__ == "__main__":
    import threading
    
    # Initialize controller
    controller = FinalMLPController("/home/ubuntu/mp2_mlp/policy_only.pt")
    
    # Start control loop in background
    control_thread = threading.Thread(target=controller.control_loop)
    control_thread.start()
    
    # Interactive control
    try:
        while True:
            print("\n" + "="*60)
            print("ROBOT CONTROL INTERFACE")
            print("="*60)
            print("Commands:")
            print("  w/s: forward/backward")
            print("  a/d: strafe left/right")
            print("  q/e: turn left/right")
            print("  space: stop")
            print("  x: exit")
            print("="*60)
            
            cmd = input("Enter command: ").strip().lower()
            
            if cmd == 'w':
                controller.set_velocity_command(0.3, 0.0, 0.0)
            elif cmd == 's':
                controller.set_velocity_command(-0.3, 0.0, 0.0)
            elif cmd == 'a':
                controller.set_velocity_command(0.0, 0.2, 0.0)
            elif cmd == 'd':
                controller.set_velocity_command(0.0, -0.2, 0.0)
            elif cmd == 'q':
                controller.set_velocity_command(0.0, 0.0, 0.2)
            elif cmd == 'e':
                controller.set_velocity_command(0.0, 0.0, -0.2)
            elif cmd == ' ':
                controller.set_velocity_command(0.0, 0.0, 0.0)
            elif cmd == 'x':
                break
            else:
                print("Unknown command")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    # Cleanup
    controller.stop()
    control_thread.join()
    print("Controller stopped successfully")
