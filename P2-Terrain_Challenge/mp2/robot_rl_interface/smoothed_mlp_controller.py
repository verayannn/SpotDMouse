import numpy as np
import torch
import time
import threading
from MangDang.mini_pupper.ESP32Interface import ESP32Interface

class SimMatchedMLPController:
    """
    Controller that exactly matches simulation's observation space and action processing.
    Includes fixes for:
    1. IMU Unit Scaling (Degrees vs Radians)
    2. Action Smoothing (Software Damping)
    3. Loop Frequency Matching
    """
    
    def __init__(self, policy_path="/home/ubuntu/mp2_mlp/policy_joyboy.pt"):
        print("=" * 60)
        print("Initializing SimMatchedMLPController")
        print("=" * 60)
        
        # Connect to ESP32
        self.esp32 = ESP32Interface()
        time.sleep(0.5)
        
        # Load policy
        try:
            self.policy = torch.jit.load(policy_path)
            self.policy.eval()
            print(f"[OK] Loaded policy from {policy_path}")
        except Exception as e:
            print(f"[ERROR] Could not load policy: {e}")
            self.shutdown = True
            return

        # ====== HARDWARE CONFIGURATION ======
        
        # 1. Servo Mapping (Correct per your tests)
        # Isaac order: LF(0,1,2), RF(3,4,5), LB(6,7,8), RB(9,10,11)
        self.esp32_servo_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        
        # 2. Direction Multipliers (Verified Correct)
        self.direction_multipliers = np.array([
            -1.0, -1.0, -1.0,  # LF
            -1.0,  1.0,  1.0,  # RF
             1.0, -1.0, -1.0,  # LB
             1.0,  1.0,  1.0,  # RB
        ])
        
        # 3. IMU Axis Configuration
        # Change to -1.0 if "Turning Left" gives negative Gyro Z in the test
        self.gyro_z_sign = - 1.0 
        
        # 4. Simulation Defaults (Standing Pose)
        self.sim_default_positions = np.array([
            0.0,  0.785, -1.57,  # LF
            0.0,  0.785, -1.57,  # RF
            0.0,  0.785, -1.57,  # LB
            0.0,  0.785, -1.57,  # RB
        ])

        self.joint_lower_limits = np.array([-0.5, 0.0, -2.5] * 4)
        self.joint_upper_limits = np.array([0.5, 1.5, -0.5] * 4)
        
        self.servo_center = 512
        self.servo_scale = 1024 / (2 * np.pi)
        
        # ====== TUNING PARAMETERS ======
        
        # ACTION SCALE: 
        # 0.30 is standard. Since we fixed the IMU units (which were 57x too big),
        # 0.30 should now feel "correct" and not weak.
        self.ACTION_SCALE = 0.40
        
        # SMOOTHING (Software Damping):
        # 0.5 = Balance between smooth and responsive.
        # If still jittery, LOWER this to 0.3 (trust old value more).
        self.action_smoothing = 0.5 
        
        # FREQUENCY:
        # 50Hz matches standard decimation (500Hz sim / 10 repeats)
        self.CONTROL_FREQUENCY = 50  
        self.startup_duration = 25   # 0.5 seconds at 50Hz
        
        # ====== CALIBRATION ======
        
        print("\n[CALIBRATION] Recording hardware standing pose...")
        raw = np.array(self.esp32.servos_get_position())
        self.standing_servos = raw[self.esp32_servo_order].astype(float)
        
        # Calculate Offset
        self.use_offset = False
        self.hw_standing_angles = self._servos_to_angles(self.standing_servos)
        self.hw_to_sim_offset = self.sim_default_positions - self.hw_standing_angles
        self.use_offset = True
        
        self._print_calibration_info()
        
        # Calibrate Sensors
        print("\n[CALIBRATION] Calibrating IMU (keep robot still)...")
        self._calibrate_imu()
        
        print("\n[CALIBRATION] Calibrating servo load...")
        self._calibrate_load()
        
        # ====== STATE VARIABLES ======
        self.prev_actions = np.zeros(12)
        self.prev_joint_angles = self.sim_default_positions.copy()
        self.prev_joint_vel = np.zeros(12)
        self.prev_time = time.time()
        self.velocity_command = np.zeros(3)
        self.control_active = False
        self.shutdown = False
        self.debug_counter = 0
        self.startup_steps = 0
        
        print("\n" + "=" * 60)
        print("Initialization complete!")
        print("=" * 60)
    
    # ====== CALIBRATION METHODS ======
    
    def _print_calibration_info(self):
        print(f"\nHardware standing servos: {self.standing_servos.astype(int)}")
        print(f"HW to Sim offset (rad): {np.round(self.hw_to_sim_offset, 3)}")
    
    def _calibrate_imu(self, samples=50):
        """Calibrate IMU with FORCED units (Fixing the overactive behavior)."""
        gyro_samples = []
        accel_samples = []
        
        for _ in range(samples):
            data = self.esp32.imu_get_data()
            if data:
                gyro_samples.append([data['gx'], data['gy'], data['gz']])
                accel_samples.append([data['ax'], data['ay'], data['az']])
            time.sleep(0.02)
        
        gyro_samples = np.array(gyro_samples)
        self.gyro_offset = np.mean(gyro_samples, axis=0)
        
        # --- FIX: Force Unit Conversion ---
        # Mini Pupper ESP32 firmware sends Degrees/s and G's.
        # We force the conversion to Rad/s and m/s².
        self.gyro_scale = np.pi / 180.0 
        self.accel_scale = 9.81         
        
        print(f"  Gyro offset: {np.round(self.gyro_offset, 3)}")
        print(f"  Gyro scale: {self.gyro_scale:.4f} (Forced Deg->Rad)")
        print(f"  Accel scale: {self.accel_scale:.2f} (Forced G->m/s²)")
    
    def _calibrate_load(self, samples=30):
        load_samples = []
        for _ in range(samples):
            load = self.esp32.servos_get_load()
            if load is not None: load_samples.append(load)
            time.sleep(0.02)
        
        if not load_samples:
            self.load_scale = 1000.0; self.load_offset = np.zeros(12)
            return
            
        load_samples = np.array(load_samples)
        self.load_offset = np.mean(load_samples, axis=0)
        self.load_scale = 5000.0 # Heuristic scale
        print(f"  Load scale: {self.load_scale}")

    # ====== SERVO HELPERS ======
    
    def _servos_to_angles(self, servo_positions):
        servo_delta = servo_positions - self.servo_center
        angles_raw = servo_delta / self.servo_scale
        angles_hw = angles_raw * self.direction_multipliers
        if self.use_offset: return angles_hw + self.hw_to_sim_offset
        return angles_hw
    
    def _angles_to_servos(self, angles_sim):
        angles_hw = angles_sim - self.hw_to_sim_offset
        angles_raw = angles_hw * self.direction_multipliers
        servo_delta = angles_raw * self.servo_scale
        return self.servo_center + servo_delta

    # ====== SENSORS ======
    
    def read_joint_positions(self):
        raw = np.array(self.esp32.servos_get_position())
        isaac_ordered = raw[self.esp32_servo_order].astype(float)
        return self._servos_to_angles(isaac_ordered)
    
    def read_joint_efforts(self):
        raw_load = self.esp32.servos_get_load()
        if raw_load is None: return np.zeros(12)
        raw_load = np.array(raw_load, dtype=float)
        isaac_ordered = raw_load[self.esp32_servo_order]
        centered = isaac_ordered - self.load_offset[self.esp32_servo_order]
        normalized = (centered / self.load_scale) * self.direction_multipliers
        return np.clip(normalized, -1.0, 1.0)
    
    def write_joint_positions(self, target_angles):
        target_servos = self._angles_to_servos(target_angles)
        target_servos = np.clip(target_servos, 180, 844)
        esp32_out = np.zeros(12)
        esp32_out[self.esp32_servo_order] = target_servos
        self.esp32.servos_set_position([int(p) for p in esp32_out])

    # ====== OBSERVATION ======
    
    def get_observation(self):
            current_time = time.time()
            dt = current_time - self.prev_time
            
            imu_data = self.esp32.imu_get_data()
            current_angles = self.read_joint_positions()
            joint_effort = self.read_joint_efforts()
            
            # 1. Base Linear Velocity 
            base_lin_vel = self.velocity_command * 0.7
            
            # 2. Base Angular Velocity (With Corrected Units & Sign Flip)
            gyro_raw = np.array([imu_data['gx'], imu_data['gy'], imu_data['gz']])
            base_ang_vel = (gyro_raw - self.gyro_offset) * self.gyro_scale
            base_ang_vel[2] *= self.gyro_z_sign 
            
            # 3. Projected Gravity
            accel_raw = np.array([imu_data['ax'], imu_data['ay'], imu_data['az']])
            accel = accel_raw * self.accel_scale
            accel_norm = np.linalg.norm(accel)
            if accel_norm > 0.1:
                projected_gravity = accel / accel_norm
            else:
                projected_gravity = np.array([0.0, 0.0, -1.0])
            
            # 4. Commands
            velocity_commands = self.velocity_command.copy()
            
            # 5. Joint Positions Rel
            joint_pos_rel = current_angles - self.sim_default_positions
            
            # 6. Joint Velocity (WITH SAFETY CLIPPING)
            if dt > 0.001:
                joint_vel_raw = (current_angles - self.prev_joint_angles) / dt
                alpha_vel = 0.2
                joint_vel_est = alpha_vel * joint_vel_raw + (1 - alpha_vel) * self.prev_joint_vel
            else:
                joint_vel_est = self.prev_joint_vel.copy()

            
            # <--- FIX 1: Clip Velocity Input
            # If a servo twitches, this prevents the policy from seeing "100 rad/s" and panicking.
            joint_vel = np.clip(joint_vel_est, -3.5, 3.5)
            joint_vel = np.zeros(12, dtype=np.float32)
                
            # 8. Previous Actions
            prev_actions = self.prev_actions.copy()
            
            # Update State
            self.prev_joint_angles = current_angles.copy()
            self.prev_joint_vel = joint_vel.copy() # Store the clipped version
            self.prev_time = current_time
            
            obs = np.concatenate([
                base_lin_vel, base_ang_vel, projected_gravity, velocity_commands,
                joint_pos_rel, joint_vel, joint_effort, prev_actions
            ]).astype(np.float32)
            
            return obs, joint_pos_rel, joint_vel, joint_effort

    # ====== CONTROL STEP (WITH SMOOTHING) ======
    
    def control_step(self):
            obs, joint_pos_rel, joint_vel, joint_effort = self.get_observation()
            
            # Policy Inference
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                raw_actions = self.policy(obs_tensor).squeeze().numpy()
            
            # <--- FIX 2: Clip Raw Actions
            # This is the most critical fix. It prevents the "-4.61" explosions.
            clipped_actions = np.clip(raw_actions, -1.0, 1.0)
            
            # Startup Fade-in
            if self.startup_steps < self.startup_duration:
                fade_in = self.startup_steps / self.startup_duration
                self.startup_steps += 1
            else:
                fade_in = 1.0
            
            faded_actions = clipped_actions * fade_in
            
            # Smoothing (Software Damping)
            alpha = self.action_smoothing
            smoothed_actions = (alpha * faded_actions) + ((1.0 - alpha) * self.prev_actions)
            
            self.prev_actions = smoothed_actions.copy()
            
            # Compute Target
            target_positions = self.sim_default_positions + smoothed_actions * self.ACTION_SCALE
            target_positions = np.clip(target_positions, self.joint_lower_limits, self.joint_upper_limits)
            
            self.write_joint_positions(target_positions)
            
            # Debug
            self.debug_counter += 1
            if self.debug_counter % 25 == 0:
                print(f"Act: [{smoothed_actions.min():+.2f}, {smoothed_actions.max():+.2f}] | "
                    f"Pos: [{target_positions.min():+.2f}, {target_positions.max():+.2f}]")
            
            return smoothed_actions

    # ====== MAIN LOOPS ======

    def control_loop(self):
        dt_target = 1.0 / self.CONTROL_FREQUENCY
        print(f"\nRunning Control Loop at {self.CONTROL_FREQUENCY}Hz")
        print(f"Smoothing: {self.action_smoothing}")
        
        while not self.shutdown:
            loop_start = time.time()
            if self.control_active:
                try:
                    self.control_step()
                except Exception as e:
                    print(f"Error: {e}")
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
            if not self.control_active: self._reset_control_state()
            self.control_active = True
            print(f"CMD: {self.velocity_command}")
        else:
            self.control_active = False
            print("STOPPED")

    def _reset_control_state(self):
        self.prev_joint_angles = self.read_joint_positions()
        self.prev_actions = np.zeros(12)
        self.prev_joint_vel = np.zeros(12)
        self.startup_steps = 0

    def stop(self):
        self.control_active = False
        self.shutdown = True


# ====== DIAGNOSTIC TOOLS ======

def check_imu_orientation(controller):
    print("\n" + "="*60)
    print("IMU ORIENTATION CHECK")
    print("="*60)
    print("CRITICAL: Rotate robot LEFT (Counter-Clockwise).")
    print("Look at 'GyrZ'. It MUST be POSITIVE.")
    print("If it is NEGATIVE, change self.gyro_z_sign to -1.0 in code.")
    print("Press Ctrl+C to exit check.\n")
    
    try:
        while True:
            data = controller.esp32.imu_get_data()
            # Apply offsets and scaling
            gx = (data['gx'] - controller.gyro_offset[0]) * controller.gyro_scale
            gy = (data['gy'] - controller.gyro_offset[1]) * controller.gyro_scale
            gz = (data['gz'] - controller.gyro_offset[2]) * controller.gyro_scale * controller.gyro_z_sign
            
            print(f"\rGyrZ (Turn Left -> +): {gz:+.2f} | GyrX: {gx:+.2f} | GyrY: {gy:+.2f}", end="")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nCheck Complete.")

# ====== ENTRY POINT ======

if __name__ == "__main__":
    controller = SimMatchedMLPController("/home/ubuntu/mp2_mlp/policy_joyboy.pt")
    
    print("\n[MENU]")
    print("1. (c)ontroller - Drive the robot")
    print("2. (i)mu check  - Verify Left Turn = Positive Z")
    choice = input("Enter choice: ").strip().lower()
    
    if choice == 'i':
        check_imu_orientation(controller)
    else:
        # Start Control Thread
        thread = threading.Thread(target=controller.control_loop)
        thread.start()
        
        speeds = {'1': 0.15, '2': 0.25, '3': 0.35}
        cur_speed = 0.2
        
        print("Commands: w/s/a/d/q/e, space=stop, x=exit")
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
                elif cmd in speeds: cur_speed = speeds[cmd]; print(f"Speed: {cur_speed}")
                elif cmd == 'x': break
        except KeyboardInterrupt:
            pass
        
        controller.stop()
        thread.join()
        print("Done.")
