import numpy as np
import torch
import time
import threading
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration

class HybridController:
    def __init__(self, policy_path="/home/ubuntu/mp2_mlp/policy_joyboy.pt"):
        print("=" * 60)
        print("Initializing Hybrid Controller (Final Version)")
        print("   - Writing: Uses HardwareInterface (Calibrated)")
        print("   - Reading: Uses Reverse-Math on Raw ESP32 Data")
        print("   - Fixes: Height Bias (Anti-Sag), Blind Velocity, Smoothers")
        print("=" * 60)
        
        # 1. Initialize the MangDang Hardware Stack
        self.config = Configuration()
        self.hardware = HardwareInterface()
        
        # 2. Access the hidden ESP32 object and Calibration Params
        self.esp32 = self.hardware.pwm_params.esp32
        self.servo_params = self.hardware.servo_params
        self.pwm_params = self.hardware.pwm_params
        
        time.sleep(0.5)
        
        # 3. Load Policy
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        
        # 4. Simulation Defaults (Standing Pose in Radians)
        self.sim_default_positions = np.array([
            0.0,  0.785, -1.57,  # LF
            0.0,  0.785, -1.57,  # RF
            0.0,  0.785, -1.57,  # LB
            0.0,  0.785, -1.57,  # RB
        ])
        
        # Limits
        self.joint_lower_limits = np.array([-0.5, 0.0, -2.5] * 4)
        self.joint_upper_limits = np.array([0.5, 1.5, -0.5] * 4)
        
        # Tuning
        self.ACTION_SCALE = 0.50
        self.action_smoothing = 0.5   # Smooths the brain
        self.target_smoothing = 0.3   # Smooths the servos
        
        # --- GRAVITY COMPENSATION (HEIGHT BIAS) ---
        # This fixes the "Crouch" issue verified by your wiggle test.
        # 0.20 rads = ~11 degrees higher standing height.
        self.height_bias = 0.20 
        
        # IMU Config
        self.gyro_z_sign = -1.0 
        self.gyro_scale = np.pi / 180.0
        self.accel_scale = 9.81

        # State Variables
        self.prev_actions = np.zeros(12)
        self.prev_joint_angles = self.sim_default_positions.copy()
        self.prev_target_positions = self.sim_default_positions.copy()
        self.prev_time = time.time()
        self.velocity_command = np.zeros(3)
        
        self.control_active = False
        self.shutdown = False
        self.debug_counter = 0
        self.startup_steps = 0
        self.startup_duration = 25
        self.CONTROL_FREQUENCY = 50

        # Calibrate IMU
        print("\n[CALIBRATION] Calibrating IMU...")
        self._calibrate_imu()

    def _calibrate_imu(self, samples=50):
        # We access the ESP32 directly since HardwareInterface doesn't wrap IMU
        gyro_samples = []
        for _ in range(samples):
            data = self.esp32.imu_get_data()
            if data:
                gyro_samples.append([data['gx'], data['gy'], data['gz']])
            time.sleep(0.02)
        
        if len(gyro_samples) > 0:
            self.gyro_offset = np.mean(gyro_samples, axis=0)
        else:
            self.gyro_offset = np.zeros(3)
        print(f"Gyro Offset: {np.round(self.gyro_offset, 3)}")

    def _isaac_to_hardware_matrix(self, flat_angles_radians):
        """Converts Isaac Sim flat array to MangDang 3x4 Matrix"""
        matrix = np.zeros((3, 4))
        # LF (Isaac 0-2) -> FL (Matrix Col 1)
        matrix[:, 1] = flat_angles_radians[0:3]
        # RF (Isaac 3-5) -> FR (Matrix Col 0)
        matrix[:, 0] = flat_angles_radians[3:6]
        # LB (Isaac 6-8) -> BL (Matrix Col 3)
        matrix[:, 3] = flat_angles_radians[6:9]
        # RB (Isaac 9-11) -> BR (Matrix Col 2)
        matrix[:, 2] = flat_angles_radians[9:12]
        return matrix

    def _hardware_read_to_isaac(self):
        """Reads raw servo ticks, uses ServoParams to convert to Radians"""
        raw_positions = self.esp32.servos_get_position()
        current_angles_isaac = np.zeros(12)
        
        legs_map = [(0, 1), (1, 0), (2, 3), (3, 2)]
        
        for isaac_leg_idx, hw_col_idx in legs_map:
            for axis in range(3):
                servo_id = self.pwm_params.servo_ids[axis, hw_col_idx]
                raw_val = raw_positions[servo_id - 1]
                
                neutral_pos = self.servo_params.neutral_position
                micros_per_rad = self.servo_params.micros_per_rad
                neutral_angle = self.servo_params.neutral_angles[axis, hw_col_idx]
                multiplier = self.servo_params.servo_multipliers[axis, hw_col_idx]
                
                if raw_val == 0 or raw_val > 1024: 
                    angle = self.sim_default_positions[isaac_leg_idx*3 + axis]
                else:
                    deviation = (neutral_pos - raw_val) / micros_per_rad
                    angle_dev = deviation / multiplier
                    angle = angle_dev + neutral_angle
                
                current_angles_isaac[isaac_leg_idx*3 + axis] = angle
        return current_angles_isaac

    def get_observation(self):
        current_time = time.time()
        
        # 1. READ SENSORS
        imu_data = self.esp32.imu_get_data()
        current_angles = self._hardware_read_to_isaac()
        
        # 2. Build Obs
        base_lin_vel = self.velocity_command * 0.7
        
        gyro_raw = np.array([imu_data['gx'], imu_data['gy'], imu_data['gz']])
        base_ang_vel = (gyro_raw - self.gyro_offset) * self.gyro_scale
        base_ang_vel[2] *= self.gyro_z_sign
        
        accel_raw = np.array([imu_data['ax'], imu_data['ay'], imu_data['az']])
        accel = accel_raw * self.accel_scale
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0.1: projected_gravity = accel / accel_norm
        else: projected_gravity = np.array([0.0, 0.0, -1.0])
        
        velocity_commands = self.velocity_command.copy()
        joint_pos_rel = current_angles - self.sim_default_positions
        
        # BLIND VELOCITY (Best for Sim2Real)
        joint_vel = np.zeros(12, dtype=np.float32)
        joint_effort = np.zeros(12, dtype=np.float32)
        prev_actions = self.prev_actions.copy()
        
        self.prev_joint_angles = current_angles.copy()
        self.prev_time = current_time
        
        obs = np.concatenate([
            base_lin_vel, base_ang_vel, projected_gravity, velocity_commands,
            joint_pos_rel, joint_vel, joint_effort, prev_actions
        ]).astype(np.float32)
        
        return obs

    def control_step(self):
        obs = self.get_observation()
        
        # Policy
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            raw_actions = self.policy(obs_tensor).squeeze().numpy()
            
        clipped_actions = np.clip(raw_actions, -1.0, 1.0)
        
        # Fade In
        if self.startup_steps < self.startup_duration:
            fade_in = self.startup_steps / self.startup_duration
            self.startup_steps += 1
        else:
            fade_in = 1.0
        faded_actions = clipped_actions * fade_in
        
        # 1. Action Smoothing
        alpha = self.action_smoothing
        smoothed_actions = (alpha * faded_actions) + ((1.0 - alpha) * self.prev_actions)
        self.prev_actions = smoothed_actions.copy()
        
        # 2. Compute Target (Radians)
        target_positions = self.sim_default_positions + smoothed_actions * self.ACTION_SCALE
        
        # --- FIX: HEIGHT BIAS (Gravity Compensation) ---
        # Add bias to Thighs (Indices 1,4,7,10) and Calves (2,5,8,11)
        # to lift the robot out of the "Sim Squat".
        target_positions[1::3] += self.height_bias * 0.5  # Thighs
        target_positions[2::3] += self.height_bias        # Calves
        # -----------------------------------------------
        
        target_positions = np.clip(target_positions, self.joint_lower_limits, self.joint_upper_limits)
        
        # 3. Target Smoothing (Slew Rate)
        beta = self.target_smoothing
        final_target = (beta * target_positions) + ((1.0 - beta) * self.prev_target_positions)
        self.prev_target_positions = final_target.copy()
        
        # 4. CONVERT TO HARDWARE MATRIX AND SEND
        target_matrix = self._isaac_to_hardware_matrix(final_target)
        self.hardware.set_actuator_postions(target_matrix)
        
        # Debug
        self.debug_counter += 1
        if self.debug_counter % 25 == 0:
            print(f"Act: [{smoothed_actions.min():+.2f}, {smoothed_actions.max():+.2f}] | "
                  f"Pos: [{final_target.min():+.2f}, {final_target.max():+.2f}]")

    def control_loop(self):
        dt_target = 1.0 / self.CONTROL_FREQUENCY
        print(f"\nRunning Hybrid Controller at {self.CONTROL_FREQUENCY}Hz")
        
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
        self.velocity_command = np.array([vx, vy, vyaw])
        if np.any(np.abs(self.velocity_command) > 0.01):
            self.control_active = True
            print(f"CMD: {self.velocity_command}")
        else:
            self.control_active = False
            print("STOPPED")

    def stop(self):
        self.control_active = False
        self.shutdown = True

if __name__ == "__main__":
    controller = HybridController("/home/ubuntu/mp2_mlp/policy_joyboy.pt")
    
    thread = threading.Thread(target=controller.control_loop)
    thread.start()
    
    print("Commands: w/s/a/d/q/e, space=stop, x=exit")
    cur_speed = 0.2
    
    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == 'w': controller.set_velocity_command(cur_speed, 0, 0)
            elif cmd == 's': controller.set_velocity_command(-cur_speed, 0, 0)
            elif cmd == 'a': controller.set_velocity_command(0, cur_speed, 0)
            elif cmd == 'd': controller.set_velocity_command(0, -cur_speed, 0)
            elif cmd == 'q': controller.set_velocity_command(0, 0, cur_speed)
            elif cmd == 'e': controller.set_velocity_command(0, 0, -cur_speed)
            elif cmd in [' ','']: controller.set_velocity_command(0, 0, 0)
            elif cmd == 'x': break
    except KeyboardInterrupt:
        pass
    
    controller.stop()
    thread.join()