import numpy as np
import torch
import time
from collections import deque
from MangDang.mini_pupper.ESP32Interface import ESP32Interface

class FinalMLPController:
    def __init__(self, policy_path="/home/ubuntu/mp2_mlp/policy_only.pt"):
        self.esp32 = ESP32Interface()
        time.sleep(0.5)
        
        imu_test = self.esp32.imu_get_data()
        print(f"IMU test data: {imu_test}")
        
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        print(f"Loaded policy from {policy_path}")
        
        # Hardware mapping
        self.esp32_servo_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        self.isaac_to_esp32 = np.zeros(12, dtype=int)
        for i, esp_idx in enumerate(self.esp32_servo_order):
            self.isaac_to_esp32[esp_idx] = i

        self.joint_direction_multipliers = np.array([
            -1.0, -1.0, -1.0,  # LF
            -1.0,  1.0,  1.0,  # RF
             1.0, -1.0, -1.0,  # LB
             1.0,  1.0,  1.0,  # RB
        ])
        
        self.isaac_training_defaults = np.array([
             0.0, 0.785, -1.57,  # LF
             0.0, 0.785, -1.57,  # RF
             0.0, 0.785, -1.57,  # LB
             0.0, 0.785, -1.57   # RB
        ])
        
        self.servo_offset = 512
        self.servo_scale = 1024 / (2 * np.pi)
        
        # Start VERY conservative
        self.ACTION_SCALE = 0.01  # Much smaller!
        self.MAX_ACTION_CHANGE = 0.01  # Much smaller!
        self.prev_actions = np.zeros(12)
        
        # Initialize with training defaults for safety
        self.prev_positions = np.zeros(12)  # Will store relative positions
        self.prev_time = time.time()
        self.velocity_command = np.zeros(3)
        self.control_active = False
        self.shutdown = False
        
        print("Calibrating IMU...")
        self.calibrate_imu()
        
        self.CONTROL_FREQUENCY = 50
        self.loop_times = deque(maxlen=50)
        
        # Debug timing
        self.last_debug_time = time.time()
        self.debug_interval = 2.0
        
    def calibrate_imu(self, samples=50):
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
        
        self.accel_offset = accel_sum / samples
        self.gyro_offset = gyro_sum / samples
        
        expected_gravity = 9.81
        actual_gravity = np.linalg.norm(self.accel_offset)
        self.gravity_scale = expected_gravity / actual_gravity if actual_gravity > 0 else 1.0
        
        print(f"IMU calibration complete:")
        print(f"  Accel offset: {self.accel_offset}")
        print(f"  Gyro offset: {self.gyro_offset}")
        print(f"  Gravity scale: {self.gravity_scale}")
    
    def get_observation(self):
        try:
            imu_data = self.esp32.imu_get_data()
            
            accel_raw = np.array([imu_data['ax'], imu_data['ay'], imu_data['az']])
            accel_calibrated = (accel_raw - self.accel_offset) * self.gravity_scale
            
            gyro_raw = np.array([imu_data['gx'], imu_data['gy'], imu_data['gz']])
            gyro_calibrated = gyro_raw - self.gyro_offset
            
            # Convert gyro from degrees/s to rad/s
            gyro_calibrated = gyro_calibrated * (np.pi / 180.0)
            
            accel_norm = np.linalg.norm(accel_calibrated)
            if accel_norm > 0.1:
                projected_gravity = accel_calibrated / accel_norm
            else:
                projected_gravity = np.array([0, 0, -1])
            
            raw_positions = np.array(self.esp32.servos_get_position())
            raw_loads = np.array(self.esp32.servos_get_load())
            
            isaac_positions = raw_positions[self.esp32_servo_order]
            isaac_loads = raw_loads[self.esp32_servo_order]
            
            hardware_radians = (isaac_positions - self.servo_offset) / self.servo_scale
            joint_positions_actual = hardware_radians / self.joint_direction_multipliers
            
            # Calculate positions relative to training defaults (what the policy expects)
            joint_positions_relative = joint_positions_actual - self.isaac_training_defaults
            
            current_time = time.time()
            dt = current_time - self.prev_time
            if dt > 0.001:
                joint_velocities = (joint_positions_relative - self.prev_positions) / dt
                joint_velocities = np.clip(joint_velocities, -10, 10)
            else:
                joint_velocities = np.zeros(12)
            
            self.prev_positions = joint_positions_relative.copy()
            self.prev_time = current_time
            
            joint_efforts = np.clip(isaac_loads / 500.0, -10, 10)
            
            obs = np.concatenate([
                accel_calibrated,
                gyro_calibrated,
                projected_gravity,
                self.velocity_command,
                joint_positions_relative,  # Relative positions
                joint_velocities,
                joint_efforts,
                self.prev_actions
            ])
            
            # Debug output every 2 seconds
            if time.time() - self.last_debug_time >= self.debug_interval:
                self.last_debug_time = time.time()
                print("\n" + "="*80)
                print("OBSERVATION DEBUG:")
                print(f"Total observation size: {len(obs)}")
                print("-"*80)
                print(f"Linear Acceleration (calibrated): {accel_calibrated}")
                print(f"Angular Velocity (calibrated): {gyro_calibrated}")
                print(f"Projected Gravity: {projected_gravity}")
                print(f"Velocity Command: {self.velocity_command}")
                print("-"*40)
                print("Joint Positions (relative to training default):")
                for i in range(4):
                    leg = ["LF", "RF", "LB", "RB"][i]
                    idx = i * 3
                    print(f"  {leg}: Hip={joint_positions_relative[idx]:.3f}, "
                          f"Thigh={joint_positions_relative[idx+1]:.3f}, "
                          f"Calf={joint_positions_relative[idx+2]:.3f}")
                print("-"*40)
                print(f"Joint Velocities (min/max): {np.min(joint_velocities):.3f} / {np.max(joint_velocities):.3f}")
                print(f"Joint Efforts (min/max): {np.min(joint_efforts):.3f} / {np.max(joint_efforts):.3f}")
                print(f"Previous Actions: {self.prev_actions}")
                print("="*80)
            
            return obs
            
        except Exception as e:
            print(f"Error getting observation: {e}")
            import traceback
            traceback.print_exc()
            return self._get_safe_observation()
    
    def _get_safe_observation(self):
        return np.concatenate([
            np.array([0, 0, -9.81]),     # Gravity
            np.zeros(3),                 # No rotation
            np.array([0, 0, -1]),        # Gravity direction
            self.velocity_command,       # Current command
            np.zeros(12),                # Zero relative position
            np.zeros(12),                # No velocity
            np.zeros(12),                # No effort
            self.prev_actions            # Previous actions
        ])
    
    def process_actions(self, mlp_actions):
        # Debug actions every 2 seconds
        if time.time() - self.last_debug_time < 0.1:
            print("\nACTION DEBUG:")
            print(f"Raw MLP outputs: {mlp_actions}")
            print(f"Action range: [{np.min(mlp_actions):.3f}, {np.max(mlp_actions):.3f}]")
        
        scaled = mlp_actions * self.ACTION_SCALE
        
        limited = scaled.copy()
        for i in range(4):
            base_idx = i * 3
            limited[base_idx] = np.clip(limited[base_idx], -0.1, 0.1)      # Hip
            limited[base_idx + 1] = np.clip(limited[base_idx + 1], -0.2, 0.2)  # Thigh  
            limited[base_idx + 2] = np.clip(limited[base_idx + 2], -0.2, 0.2)  # Calf
        
        action_delta = limited - self.prev_actions
        action_delta = np.clip(action_delta, -self.MAX_ACTION_CHANGE, self.MAX_ACTION_CHANGE)
        smoothed = self.prev_actions + action_delta
        
        self.prev_actions = smoothed.copy()
        
        # Actions are deltas from training defaults
        target_positions = self.isaac_training_defaults + smoothed
        
        # Convert to hardware
        hardware_corrected = target_positions * self.joint_direction_multipliers
        servo_positions = hardware_corrected * self.servo_scale + self.servo_offset
        servo_positions = np.clip(servo_positions, 100, 924)
        
        esp32_positions = servo_positions[self.isaac_to_esp32]
        
        # More debug info
        if time.time() - self.last_debug_time < 0.1:
            print(f"Scaled actions: {scaled[:6]}...")  # First 6 for brevity
            print(f"Smoothed actions: {smoothed[:6]}...")
            print(f"Target angles: {target_positions[:6]}...")
            print(f"Servo positions: {servo_positions[:6]}...")
            print("-"*80)
        
        return [int(pos) for pos in esp32_positions]
    
    def go_to_default_stance(self):
        """Safely move to default stance"""
        print("Moving to default stance...")
        target_positions = self.isaac_training_defaults
        hardware_corrected = target_positions * self.joint_direction_multipliers
        servo_positions = hardware_corrected * self.servo_scale + self.servo_offset
        servo_positions = np.clip(servo_positions, 100, 924)
        esp32_positions = servo_positions[self.isaac_to_esp32]
        
        self.esp32.servos_set_position([int(pos) for pos in esp32_positions])
        time.sleep(1.0)
        print("At default stance")
    
    def control_loop(self):
        dt_target = 1.0 / self.CONTROL_FREQUENCY
        
        print(f"\nControl loop started at {self.CONTROL_FREQUENCY}Hz")
        print("Moving to safe default stance first...")
        self.go_to_default_stance()
        
        while not self.shutdown:
            loop_start = time.time()
            
            try:
                if self.control_active:
                    obs = self.get_observation()
                    
                    with torch.no_grad():
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                        raw_actions = self.policy(obs_tensor).squeeze().numpy()
                    
                    servo_commands = self.process_actions(raw_actions)
                    self.esp32.servos_set_position_torque(servo_commands, [1]*12)
                    
                    if len(self.loop_times) == self.loop_times.maxlen:
                        avg_time = np.mean(self.loop_times)
                        # Regular status update
                        if int(loop_start) % 50 == 0:  # Every second at 50Hz
                            print(f"Status - Loop: {1/avg_time:.1f}Hz | "
                                  f"Cmd: [{self.velocity_command[0]:.2f}, "
                                  f"{self.velocity_command[1]:.2f}, "
                                  f"{self.velocity_command[2]:.2f}] | "
                                  f"Action scale: {self.ACTION_SCALE:.3f}")
                                  
                else:
                    # Hold default stance when not active
                    pass
                    
            except Exception as e:
                print(f"Control loop error: {e}")
                import traceback
                traceback.print_exc()
                self.control_active = False
                # Go to safe stance on error
                self.go_to_default_stance()
            
            elapsed = time.time() - loop_start
            self.loop_times.append(elapsed)
            
            if elapsed < dt_target:
                time.sleep(dt_target - elapsed)
        
        print("Control loop stopped")
    
    def set_velocity_command(self, vx, vy, vyaw):
        self.velocity_command = np.array([
            np.clip(vx, -0.35, 0.40),
            np.clip(vy, -0.35, 0.35),
            np.clip(vyaw, -0.30, 0.30)
        ])
        
        if np.any(self.velocity_command != 0):
            self.control_active = True
            print(f"Control activated with command: {self.velocity_command}")
        else:
            self.control_active = False
            print("Control deactivated (zero command)")
    
    def stop(self):
        print("Stopping controller...")
        self.control_active = False
        self.shutdown = True
        time.sleep(0.1)
        self.go_to_default_stance()


if __name__ == "__main__":
    import threading
    
    controller = FinalMLPController("/home/ubuntu/mp2_mlp/policy_only.pt")
    
    control_thread = threading.Thread(target=controller.control_loop)
    control_thread.start()
    
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
            print("  t: test stand (very small movements)")
            print("  scale <value>: set action scale (e.g., scale 0.02)")
            print("  x: exit")
            print("="*60)
            
            cmd = input("Enter command: ").strip().lower()
            
            if cmd == 'w':
                controller.set_velocity_command(0.2, 0.0, 0.0)
            elif cmd == 's':
                controller.set_velocity_command(-0.2, 0.0, 0.0)
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
            elif cmd == 't':
                print("Test mode - very small movements")
                controller.ACTION_SCALE = 0.005  # Even smaller!
                controller.MAX_ACTION_CHANGE = 0.005
                controller.set_velocity_command(0.0, 0.0, 0.0)
            elif cmd.startswith('scale'):
                try:
                    scale = float(cmd.split()[1])
                    controller.ACTION_SCALE = np.clip(scale, 0.001, 0.1)  # Max 0.1 for safety
                    controller.MAX_ACTION_CHANGE = controller.ACTION_SCALE
                    print(f"Action scale set to: {controller.ACTION_SCALE}")
                except:
                    print("Usage: scale <value>  (e.g., scale 0.02)")
            elif cmd == 'x':
                break
            else:
                print("Unknown command")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    controller.stop()
    control_thread.join()
    print("Controller stopped successfully")