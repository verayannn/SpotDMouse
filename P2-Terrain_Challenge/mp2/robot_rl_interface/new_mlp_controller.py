import numpy as np
import torch
import time
from collections import deque
from MangDang.mini_pupper.ESP32Interface import ESP32Interface

class CorrectedMLPController:
    def __init__(self, policy_path="/home/ubuntu/mp2_mlp/policy_only.pt"):
        self.esp32 = ESP32Interface()
        time.sleep(0.5)
        
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        print(f"Loaded policy from {policy_path}")
        
        # Hardware mapping: Isaac index -> ESP32 servo index
        self.esp32_servo_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        
        # Inverse mapping: ESP32 index -> Isaac index
        self.isaac_to_esp32 = np.zeros(12, dtype=int)
        for isaac_idx, esp_idx in enumerate(self.esp32_servo_order):
            self.isaac_to_esp32[esp_idx] = isaac_idx

        # Direction multipliers: multiply real angles by these to get sim angles
        # (and multiply sim angles by these to get real angles - same operation)
        self.real_to_sim_multipliers = np.array([
            -1.0, -1.0, -1.0,  # LF
            -1.0,  1.0,  1.0,  # RF
             1.0, -1.0, -1.0,  # LB
             1.0,  1.0,  1.0,  # RB
        ])
        
        # Isaac training default pose (in sim coordinates)
        self.isaac_defaults = np.array([
             0.0, 0.785, -1.57,  # LF
             0.0, 0.785, -1.57,  # RF
             0.0, 0.785, -1.57,  # LB
             0.0, 0.785, -1.57   # RB
        ])
        
        # Servo conversion constants
        self.servo_center = 512
        self.servo_scale = 1024 / (2 * np.pi)  # ~163 counts per radian
        
        # Calibrate standing pose
        print("Reading hardware standing position...")
        self._calibrate_standing_pose()
        
        # Action scaling (from training config)
        self.ACTION_SCALE = 0.25  # Check your training config for this value
        
        # State variables
        self.prev_actions = np.zeros(12)
        self.prev_joint_pos_rel = np.zeros(12)
        self.prev_joint_vel = np.zeros(12)
        self.prev_time = time.time()
        self.velocity_command = np.zeros(3)
        self.control_active = False
        self.shutdown = False
        
        # Calibrate IMU
        print("Calibrating IMU...")
        self._calibrate_imu()
        
        self.CONTROL_FREQUENCY = 50
        self.loop_times = deque(maxlen=50)
        
    def _calibrate_standing_pose(self):
        """
        Read current servo positions and establish the mapping between
        hardware standing pose and Isaac default pose.
        """
        raw_servos = np.array(self.esp32.servos_get_position())
        isaac_ordered_servos = raw_servos[self.esp32_servo_order]
        
        # Convert to radians (in hardware frame, centered at servo_center)
        hardware_radians = (isaac_ordered_servos - self.servo_center) / self.servo_scale
        
        # Convert to sim frame
        sim_frame_standing = hardware_radians * self.real_to_sim_multipliers
        
        # This is what the sim would call "absolute joint positions" at standing
        # For joint_pos_rel, we need: current_pos - default_pos
        # At standing, joint_pos_rel should be 0
        # So we store the offset needed
        self.standing_offset_sim = sim_frame_standing  # Should be close to isaac_defaults
        
        print(f"Hardware servos (Isaac order): {isaac_ordered_servos}")
        print(f"Hardware radians: {hardware_radians}")
        print(f"Sim frame standing: {sim_frame_standing}")
        print(f"Expected Isaac defaults: {self.isaac_defaults}")
        print(f"Difference: {sim_frame_standing - self.isaac_defaults}")
        
    def _calibrate_imu(self, samples=50):
        print("Keep robot still for IMU calibration...")
        
        gyro_sum = np.zeros(3)
        accel_sum = np.zeros(3)
        
        for i in range(samples):
            data = self.esp32.imu_get_data()
            accel_sum += np.array([data['ax'], data['ay'], data['az']])
            gyro_sum += np.array([data['gx'], data['gy'], data['gz']])
            time.sleep(0.02)
            
        self.gyro_offset = gyro_sum / samples
        self.accel_offset = accel_sum / samples
        
        print(f"Gyro offset: {self.gyro_offset}")
        print(f"Accel offset: {self.accel_offset}")
    
    def _read_joint_positions_sim_frame(self):
        """Read servo positions and convert to sim frame."""
        raw_servos = np.array(self.esp32.servos_get_position())
        isaac_ordered = raw_servos[self.esp32_servo_order]
        
        # Convert to radians in hardware frame
        hardware_radians = (isaac_ordered - self.servo_center) / self.servo_scale
        
        # Convert to sim frame
        sim_frame = hardware_radians * self.real_to_sim_multipliers
        
        return sim_frame
    
    def _sim_actions_to_servo_commands(self, sim_actions):
        """Convert sim-frame action deltas to servo commands."""
        # Actions are deltas from default pose (scaled)
        scaled_actions = sim_actions * self.ACTION_SCALE
        
        # Target position in sim frame = default + action
        target_sim = self.isaac_defaults + scaled_actions
        
        # Convert to hardware frame
        target_hardware = target_sim * self.real_to_sim_multipliers  # Same operation for inverse
        
        # Convert to servo units
        servo_positions = target_hardware * self.servo_scale + self.servo_center
        servo_positions = np.clip(servo_positions, 100, 924)
        
        # Reorder for ESP32
        esp32_positions = servo_positions[self.isaac_to_esp32]
        
        return [int(pos) for pos in esp32_positions]
    
    def get_observation(self):
        try:
            # IMU data
            imu_data = self.esp32.imu_get_data()
            
            # Base angular velocity (from gyro, calibrated)
            gyro_raw = np.array([imu_data['gx'], imu_data['gy'], imu_data['gz']])
            base_ang_vel = gyro_raw - self.gyro_offset
            
            # Projected gravity (from accelerometer)
            accel_raw = np.array([imu_data['ax'], imu_data['ay'], imu_data['az']])
            accel_calibrated = accel_raw  # Don't subtract offset for gravity direction
            accel_norm = np.linalg.norm(accel_calibrated)
            if accel_norm > 0.1:
                projected_gravity = -accel_calibrated / accel_norm  # Negative because accel measures reaction to gravity
            else:
                projected_gravity = np.array([0, 0, -1])
            
            # Base linear velocity - zeros (policy should cope with noise in training)
            base_lin_vel = np.zeros(3)
            
            # Joint positions in sim frame
            joint_pos_sim = self._read_joint_positions_sim_frame()
            
            # Relative to default (this is what joint_pos_rel computes in Isaac)
            joint_pos_rel = joint_pos_sim - self.isaac_defaults
            
            # Joint velocities
            current_time = time.time()
            dt = current_time - self.prev_time
            if dt > 0.001:
                joint_vel_raw = (joint_pos_rel - self.prev_joint_pos_rel) / dt
                # Low-pass filter
                alpha = 0.3
                joint_vel = alpha * joint_vel_raw + (1 - alpha) * self.prev_joint_vel
                # Clip to match sim range
                joint_vel = np.clip(joint_vel, -1.5, 1.5)
            else:
                joint_vel = self.prev_joint_vel
            
            self.prev_joint_pos_rel = joint_pos_rel.copy()
            self.prev_joint_vel = joint_vel.copy()
            self.prev_time = current_time
            
            # Joint efforts - zeros for now
            joint_efforts = np.zeros(12)
            
            # Build observation in EXACT order from training config
            obs = np.concatenate([
                base_lin_vel,           # 3: base_lin_vel
                base_ang_vel,           # 3: base_ang_vel
                projected_gravity,      # 3: projected_gravity
                self.velocity_command,  # 3: velocity_commands
                joint_pos_rel,          # 12: joint_pos_rel
                joint_vel,              # 12: joint_vel_rel
                joint_efforts,          # 12: joint_effort
                self.prev_actions       # 12: last_action
            ])
            
            return obs, joint_pos_rel  # Return joint_pos_rel for debugging
            
        except Exception as e:
            print(f"Error getting observation: {e}")
            import traceback
            traceback.print_exc()
            return self._get_safe_observation(), np.zeros(12)
    
    def _get_safe_observation(self):
        return np.concatenate([
            np.zeros(3),  # base_lin_vel
            np.zeros(3),  # base_ang_vel
            np.array([0, 0, -1]),  # projected_gravity
            self.velocity_command,
            np.zeros(12),  # joint_pos_rel
            np.zeros(12),  # joint_vel
            np.zeros(12),  # joint_effort
            self.prev_actions
        ])
    
    def control_loop(self):
        dt_target = 1.0 / self.CONTROL_FREQUENCY
        
        print(f"\nControl loop started at {self.CONTROL_FREQUENCY}Hz")
        print("Commands: set_velocity_command(vx, vy, vyaw)")
        
        debug_counter = 0
        
        while not self.shutdown:
            loop_start = time.time()
            
            try:
                if self.control_active:
                    obs, joint_pos_rel = self.get_observation()
                    
                    # Debug print every 2 seconds
                    debug_counter += 1
                    if debug_counter % 100 == 0:
                        print(f"\n--- Debug (step {debug_counter}) ---")
                        print(f"joint_pos_rel: {joint_pos_rel}")
                        print(f"Range: [{joint_pos_rel.min():.3f}, {joint_pos_rel.max():.3f}]")
                        print(f"Velocity cmd: {self.velocity_command}")
                    
                    with torch.no_grad():
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                        raw_actions = self.policy(obs_tensor).squeeze().numpy()
                    
                    # Store for next observation
                    self.prev_actions = raw_actions.copy()
                    
                    # Convert to servo commands
                    servo_commands = self._sim_actions_to_servo_commands(raw_actions)
                    self.esp32.servos_set_position(servo_commands)
                    
            except Exception as e:
                print(f"Control loop error: {e}")
                import traceback
                traceback.print_exc()
                self.control_active = False
            
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
        self.control_active = np.any(self.velocity_command != 0)
        print(f"Command: {self.velocity_command}, Active: {self.control_active}")
    
    def stop(self):
        self.control_active = False
        self.shutdown = True


if __name__ == "__main__":
    import threading
    
    controller = CorrectedMLPController("/home/ubuntu/mp2_mlp/policy_only.pt")
    
    control_thread = threading.Thread(target=controller.control_loop)
    control_thread.start()
    
    try:
        while True:
            cmd = input("Command (w/s/a/d/q/e/space/x): ").strip().lower()
            
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
            elif cmd == ' ' or cmd == '':
                controller.set_velocity_command(0.0, 0.0, 0.0)
            elif cmd == 'x':
                break
                
    except KeyboardInterrupt:
        pass
    
    controller.stop()
    control_thread.join()