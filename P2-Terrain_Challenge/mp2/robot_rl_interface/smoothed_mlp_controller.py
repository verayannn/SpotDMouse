import numpy as np
import torch
import time
import threading
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration

class RetargetingController:
    def __init__(self, policy_path="/home/ubuntu/mp2_mlp/policy_joyboy.pt"):
        print("=" * 60)
        print("Initializing Retargeting Controller (The Translation Layer)")
        print("   - Strategy: Bidirectional Mapping (Sim World <-> Real World)")
        print("   - Reading: REAL SENSORS (Translated to Sim Frame)")
        print("   - Writing: REAL COMMANDS (Translated from Sim Frame)")
        print("=" * 60)
        
        # 1. Hardware
        self.config = Configuration()
        self.hardware = HardwareInterface()
        self.esp32 = self.hardware.pwm_params.esp32
        self.servo_params = self.hardware.servo_params
        self.pwm_params = self.hardware.pwm_params
        
        time.sleep(0.5)
        
        # 2. Policy
        try:
            self.policy = torch.jit.load(policy_path)
            self.policy.eval()
            print(f"[OK] Loaded policy from {policy_path}")
        except Exception as e:
            print(f"[ERROR] Could not load policy: {e}")
            self.shutdown = True
            return
        
        # 3. Simulation Defaults
        self.sim_default_positions = np.array([
            0.0,  0.785, -1.57,  # LF
            0.0,  0.785, -1.57,  # RF
            0.0,  0.785, -1.57,  # LB
            0.0,  0.785, -1.57,  # RB
        ])
        
        self.joint_lower_limits = np.array([-0.5, 0.0, -2.5] * 4)
        self.joint_upper_limits = np.array([0.5, 1.5, -0.5] * 4)
        
        # 4. Tuning
        self.ACTION_SCALE = 0.50
        self.action_smoothing = 0.5   
        self.target_smoothing = 0.2   # Lower = Snappier servos
        
        # --- THE TRANSLATION OFFSET ---
        # This is the "Magic Number" that converts Squat <-> Stand
        self.calf_offset = 0.60
        
        # IMU Config
        self.gyro_z_sign = -1.0 
        self.gyro_scale = np.pi / 180.0
        self.accel_scale = 9.81

        # 5. INITIALIZATION
        # Read the real robot, then convert it to Sim Frame for the history
        print("[STARTUP] Syncing State...")
        real_angles = self._hardware_read_raw_angles()
        sim_frame_angles = self._real_to_sim(real_angles)
        
        self.prev_joint_angles = sim_frame_angles.copy()
        self.prev_target_positions = real_angles.copy() # Target is in Real Frame
        self.prev_actions = np.zeros(12)
        self.prev_joint_vel = np.zeros(12)
        self.prev_time = time.time()
        
        self.velocity_command = np.zeros(3)
        self.control_active = False
        self.shutdown = False
        self.debug_counter = 0
        self.startup_steps = 0
        self.startup_duration = 50 
        self.CONTROL_FREQUENCY = 50

        print("[CALIBRATION] Calibrating IMU...")
        self._calibrate_imu()

    def _calibrate_imu(self, samples=50):
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

    # --- TRANSLATION LAYER METHODS ---

    def _real_to_sim(self, real_angles):
        """Converts REAL Robot angles to SIMULATION angles"""
        sim_angles = real_angles.copy()
        # Robot is Standing (-0.8), Sim expects Squat (-1.57)
        # So we SUBTRACT the offset (-0.8 - 0.77 = -1.57)
        sim_angles[2::3] -= self.calf_offset
        return sim_angles

    def _sim_to_real(self, sim_angles):
        """Converts SIMULATION targets to REAL Robot targets"""
        real_angles = sim_angles.copy()
        # Sim commands Squat (-1.57), Robot needs Stand (-0.8)
        # So we ADD the offset (-1.57 + 0.77 = -0.8)
        real_angles[2::3] += self.calf_offset
        return real_angles

    def _hardware_read_raw_angles(self):
        """Reads ESP32 ticks and converts to Radians (No offsets applied yet)"""
        raw_positions = self.esp32.servos_get_position()
        angles = np.zeros(12)
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
                    # Fallback to previous target if sensor fails
                    if hasattr(self, 'prev_target_positions'):
                        angle = self.prev_target_positions[isaac_leg_idx*3 + axis]
                    else:
                        angle = self._sim_to_real(self.sim_default_positions)[isaac_leg_idx*3 + axis]
                else:
                    deviation = (neutral_pos - raw_val) / micros_per_rad
                    angle_dev = deviation / multiplier
                    angle = angle_dev + neutral_angle
                
                angles[isaac_leg_idx*3 + axis] = angle
        return angles

    def _isaac_to_hardware_matrix(self, flat_angles_radians):
        matrix = np.zeros((3, 4))
        matrix[:, 1] = flat_angles_radians[0:3] # LF
        matrix[:, 0] = flat_angles_radians[3:6] # RF
        matrix[:, 3] = flat_angles_radians[6:9] # LB
        matrix[:, 2] = flat_angles_radians[9:12] # RB
        return matrix

    def get_observation(self):
        current_time = time.time()
        dt = current_time - self.prev_time
        
        # 1. READ REAL SENSORS
        imu_data = self.esp32.imu_get_data()
        real_angles = self._hardware_read_raw_angles()
        
        # 2. CONVERT TO SIM FRAME
        # The policy thinks it is controlling the squatted simulation robot
        sim_compatible_angles = self._real_to_sim(real_angles)
        
        # 3. Build Obs
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
        joint_pos_rel = sim_compatible_angles - self.sim_default_positions
        
        # 4. REAL VELOCITY (Calculated in Sim Frame)
        # Since we are using real sensors now, we can calculate real velocity
        if dt > 0.001:
            raw_vel = (sim_compatible_angles - self.prev_joint_angles) / dt
            alpha_vel = 0.1 # Heavy filter
            joint_vel = alpha_vel * raw_vel + (1 - alpha_vel) * self.prev_joint_vel
        else:
            joint_vel = self.prev_joint_vel.copy()
            
        joint_effort = np.zeros(12, dtype=np.float32)
        prev_actions = self.prev_actions.copy()
        
        self.prev_joint_angles = sim_compatible_angles.copy()
        self.prev_joint_vel = joint_vel.copy()
        self.prev_time = current_time
        
        obs = np.concatenate([
            base_lin_vel, base_ang_vel, projected_gravity, velocity_commands,
            joint_pos_rel, joint_vel, joint_effort, prev_actions
        ]).astype(np.float32)
        
        return obs

    def control_step(self):
        obs = self.get_observation()
        
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            raw_actions = self.policy(obs_tensor).squeeze().numpy()
            
        clipped_actions = np.clip(raw_actions, -1.0, 1.0)
        
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
        
        # 2. Compute Target (SIM FRAME)
        policy_target_sim = self.sim_default_positions + smoothed_actions * self.ACTION_SCALE
        
        # 3. TRANSLATE TO REAL FRAME (Add Offset)
        policy_target_real = self._sim_to_real(policy_target_sim)
        policy_target_real = np.clip(policy_target_real, -2.5, 2.5)
        
        # 4. Target Smoothing (Slew Rate)
        beta = self.target_smoothing
        final_target = (beta * policy_target_real) + ((1.0 - beta) * self.prev_target_positions)
        self.prev_target_positions = final_target.copy()
        
        # 5. Write
        target_matrix = self._isaac_to_hardware_matrix(final_target)
        self.hardware.set_actuator_postions(target_matrix)
        
        self.debug_counter += 1
        if self.debug_counter % 25 == 0:
            print(f"Act: [{smoothed_actions.min():+.2f}, {smoothed_actions.max():+.2f}]")

    def control_loop(self):
        dt_target = 1.0 / self.CONTROL_FREQUENCY
        print(f"\nRunning Retargeting Controller at {self.CONTROL_FREQUENCY}Hz")
        
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
    controller = RetargetingController("/home/ubuntu/mp2_mlp/policy_joyboy.pt")
    
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