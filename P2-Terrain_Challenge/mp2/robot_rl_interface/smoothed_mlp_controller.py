import numpy as np
import torch
import time
from MangDang.mini_pupper.ESP32Interface import ESP32Interface
import threading

class SimMatchedMLPController:
    """
    Controller that exactly matches simulation's observation space and action processing.
    """
    
    def __init__(self, policy_path="/home/ubuntu/mp2_mlp/policy_joyboy.pt"):
        print("=" * 60)
        print("Initializing SimMatchedMLPController")
        print("=" * 60)
        
        # Connect to ESP32
        self.esp32 = ESP32Interface()
        time.sleep(0.5)
        
        # Load policy
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        print(f"[OK] Loaded policy from {policy_path}")
        
        # ====== HARDWARE CONFIGURATION ======
        
        # Servo order: Isaac Sim index -> ESP32 servo index
        # Isaac order: LF(0,1,2), RF(3,4,5), LB(6,7,8), RB(9,10,11)
        self.esp32_servo_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        
        # Direction multipliers: convert hardware motion direction to sim direction
        self.direction_multipliers = np.array([
            -1.0, -1.0, -1.0,  # LF
            -1.0,  1.0,  1.0,  # RF
             1.0, -1.0, -1.0,  # LB
             1.0,  1.0,  1.0,  # RB
        ])
        
        # Simulation's default joint positions (standing pose in radians)
        self.sim_default_positions = np.array([
            0.0,  0.785, -1.57,  # LF
            0.0,  0.785, -1.57,  # RF
            0.0,  0.785, -1.57,  # LB
            0.0,  0.785, -1.57,  # RB
        ])

        # Joint limits in radians
        self.joint_lower_limits = np.array([-0.5, 0.0, -2.5] * 4)
        self.joint_upper_limits = np.array([0.5, 1.5, -0.5] * 4)
        
        # Servo constants
        self.servo_center = 512
        self.servo_scale = 1024 / (2 * np.pi)
        
        # Action scale from training config
        self.ACTION_SCALE = 0.50
        
        # ====== CALIBRATION ======
        
        # Record hardware standing pose
        print("\n[CALIBRATION] Recording hardware standing pose...")
        raw = np.array(self.esp32.servos_get_position())
        self.standing_servos = raw[self.esp32_servo_order].astype(float)
        
        # Convert to angles
        self.use_offset = False
        self.hw_standing_angles = self._servos_to_angles(self.standing_servos)
        
        # Calculate offset between hardware and simulation frames
        self.hw_to_sim_offset = self.sim_default_positions - self.hw_standing_angles
        self.use_offset = True
        
        self._print_calibration_info()
        
        # Calibrate IMU
        print("\n[CALIBRATION] Calibrating IMU (keep robot still)...")
        self._calibrate_imu()
        
        # Calibrate servo load scaling
        print("\n[CALIBRATION] Calibrating servo load range...")
        self._calibrate_load()
        
        # ====== STATE VARIABLES ======
        
        self.prev_actions = np.zeros(12)
        self.prev_joint_angles = self.sim_default_positions.copy()
        self.prev_joint_vel = np.zeros(12)
        self.prev_time = time.time()
        
        # Velocity command and estimation
        self.velocity_command = np.zeros(3)
        
        # Control state
        self.control_active = False
        self.shutdown = False
        self.debug_counter = 0
        self.startup_steps = 0
        
        # ====== CONTROL PARAMETERS ======
        
        # <--- FIX 1: LOWERED FREQUENCY
        # Changed from 100Hz to 50Hz.
        # This matches standard Policy decimation (500Hz sim / 10 repeat = 50Hz)
        # And reduces serial bottleneck issues.
        self.CONTROL_FREQUENCY = 50  
        
        self.startup_duration = 25 
        
        # Action processing
        self.action_clip = True
        
        # <--- FIX 2: SMOOTHING FACTOR
        # 0.5 means "New action is 50% new command + 50% old command"
        # This acts as the "software damper" (KD)
        self.action_smoothing = 0.70 
        
        print("\n" + "=" * 60)
        print("Initialization complete!")
        print("=" * 60)
    
    # ====== CALIBRATION METHODS ======
    
    def _print_calibration_info(self):
        """Print detailed calibration information."""
        print(f"\nHardware standing servos: {self.standing_servos.astype(int)}")
        print(f"HW to Sim offset (rad): {np.round(self.hw_to_sim_offset, 3)}")
    
    def _calibrate_imu(self, samples=50):
        """Calibrate IMU gyroscope offset and detect units."""
        gyro_samples = []
        accel_samples = []
        
        for _ in range(samples):
            data = self.esp32.imu_get_data()
            if data:
                gyro_samples.append([data['gx'], data['gy'], data['gz']])
                accel_samples.append([data['ax'], data['ay'], data['az']])
            time.sleep(0.02)
        
        gyro_samples = np.array(gyro_samples)
        accel_samples = np.array(accel_samples)
        
        self.gyro_offset = np.mean(gyro_samples, axis=0)
        
        # Detect units
        if np.max(np.abs(self.gyro_offset)) > 5:
            self.gyro_scale = np.pi / 180.0
            print(f"  Gyro: deg/s detected -> converting to rad/s")
        else:
            self.gyro_scale = 1.0
            print(f"  Gyro: rad/s detected")
            
        accel_mean = np.mean(accel_samples, axis=0)
        accel_magnitude = np.linalg.norm(accel_mean)
        
        if accel_magnitude > 5:
            self.accel_scale = 1.0
            print(f"  Accel: m/s² detected")
        else:
            self.accel_scale = 9.81
            print(f"  Accel: g's detected -> converting to m/s²")
    
    def _calibrate_load(self, samples=30):
        """Calibrate servo load scaling."""
        load_samples = []
        for _ in range(samples):
            load = self.esp32.servos_get_load()
            if load is not None:
                load_samples.append(load)
            time.sleep(0.02)
        
        if not load_samples:
            self.load_scale = 1000.0
            self.load_offset = np.zeros(12)
            return
        
        load_samples = np.array(load_samples)
        self.load_offset = np.mean(load_samples, axis=0)
        self.load_scale = 5000.0 
        print(f"  Load scale set to: {self.load_scale}")
    
    # ====== SERVO CONVERSION METHODS ======
    
    def _servos_to_angles(self, servo_positions):
        servo_delta = servo_positions - self.servo_center
        angles_raw = servo_delta / self.servo_scale
        angles_hw = angles_raw * self.direction_multipliers
        if self.use_offset:
            return angles_hw + self.hw_to_sim_offset
        return angles_hw
    
    def _angles_to_servos(self, angles_sim):
        angles_hw = angles_sim - self.hw_to_sim_offset
        angles_raw = angles_hw * self.direction_multipliers
        servo_delta = angles_raw * self.servo_scale
        return self.servo_center + servo_delta
    
    # ====== SENSOR READING METHODS ======
    
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
    
    # ====== OBSERVATION BUILDING ======
    
    def get_observation(self):
        current_time = time.time()
        dt = current_time - self.prev_time
        
        imu_data = self.esp32.imu_get_data()
        current_angles = self.read_joint_positions()
        joint_effort = self.read_joint_efforts()
        
        # 1. Base Linear Velocity (Estimated)
        base_lin_vel = self.velocity_command * 0.7
        
        # 2. Base Angular Velocity
        gyro_raw = np.array([imu_data['gx'], imu_data['gy'], imu_data['gz']])
        base_ang_vel = (gyro_raw - self.gyro_offset) * self.gyro_scale
        
        # 3. Projected Gravity
        accel_raw = np.array([imu_data['ax'], imu_data['ay'], imu_data['az']])
        accel = accel_raw * self.accel_scale
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0.1:
            projected_gravity = accel / accel_norm
        else:
            projected_gravity = np.array([0.0, 0.0, -1.0])
        
        # 4. Velocity Commands
        velocity_commands = self.velocity_command.copy()
        
        # 5. Joint Positions Relative
        joint_pos_rel = current_angles - self.sim_default_positions
        
        # 6. Joint Velocities
        if dt > 0.001:
            joint_vel_raw = (current_angles - self.prev_joint_angles) / dt
            # <--- FIX 3: MORE AGGRESSIVE FILTERING
            # Changed alpha from 0.4 to 0.2. 
            # This ignores more "potentiometer noise" so the robot doesn't twitch.
            alpha_vel = 0.2 
            joint_vel = alpha_vel * joint_vel_raw + (1 - alpha_vel) * self.prev_joint_vel
        else:
            joint_vel = self.prev_joint_vel.copy()
        
        # 7. Joint Efforts (Already read)
        
        # 8. Previous Actions
        prev_actions = self.prev_actions.copy()
        
        # Update state
        self.prev_joint_angles = current_angles.copy()
        self.prev_joint_vel = joint_vel.copy()
        self.prev_time = current_time
        
        obs = np.concatenate([
            base_lin_vel, base_ang_vel, projected_gravity, velocity_commands,
            joint_pos_rel, joint_vel, joint_effort, prev_actions
        ]).astype(np.float32)
        
        return obs, joint_pos_rel, joint_vel, joint_effort

    # ====== CONTROL LOOP ======
        
    def control_step(self):
        """Matches C++ neural_controller behavior with software smoothing."""
        obs, joint_pos_rel, joint_vel, joint_effort = self.get_observation()
        
        # Policy inference
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            raw_actions = self.policy(obs_tensor).squeeze().numpy()
        
        # Fade-in multiplier
        if self.startup_steps < self.startup_duration:
            fade_in = self.startup_steps / self.startup_duration
            self.startup_steps += 1
        else:
            fade_in = 1.0
        
        # Apply fade-in
        faded_actions = raw_actions * fade_in
        
        # <--- FIX 4: APPLY ACTION SMOOTHING (The Software Shock Absorber)
        # This acts as the "Kd" (Damping) term that hardware PD controllers have.
        # Without this, the raw neural network output is too jittery for Python + PWM Servos.
        alpha = self.action_smoothing
        smoothed_actions = (alpha * faded_actions) + ((1.0 - alpha) * self.prev_actions)
        
        # Update state for next step
        self.prev_actions = smoothed_actions.copy()
        
        # Compute target using SMOOTHED actions
        target_positions = self.sim_default_positions + smoothed_actions * self.ACTION_SCALE
        
        # Clamp to joint limits
        target_positions = np.clip(target_positions, 
                                    self.joint_lower_limits, 
                                    self.joint_upper_limits)
        
        # Send to servos
        self.write_joint_positions(target_positions)
        
        # Debug
        self.debug_counter += 1
        if self.debug_counter % 25 == 0: # Adjusted for 50Hz
            print(f"Target pos: [{target_positions.min():+.2f}, {target_positions.max():+.2f}]")
        
        return smoothed_actions
    
    def control_loop(self):
        """Main control loop."""
        dt_target = 1.0 / self.CONTROL_FREQUENCY
        
        print(f"\n{'='*60}")
        print(f"Control loop running at {self.CONTROL_FREQUENCY}Hz")
        print(f"Action Smoothing: {self.action_smoothing} (Software Damping)")
        print(f"{'='*60}")
        print("\nCommands: w/s/a/d/q/e, space to stop, x to exit")
        
        self.speed_multiplier = 1.0
        
        while not self.shutdown:
            loop_start = time.time()
            
            if self.control_active:
                try:
                    self.control_step()
                except Exception as e:
                    print(f"Error in control step: {e}")
                    import traceback
                    traceback.print_exc()
                    self.control_active = False
            
            # Maintain control frequency
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
                self._reset_control_state()
            self.control_active = True
            print(f"Active: cmd=[{self.velocity_command[0]:+.2f}, {self.velocity_command[1]:+.2f}, {self.velocity_command[2]:+.2f}]")
        else:
            self.control_active = False
            self.startup_steps = 0
            self.estimated_lin_vel = np.zeros(3)
            print("Stopped")
    
    def _reset_control_state(self):
        current_angles = self.read_joint_positions()
        self.prev_joint_angles = current_angles.copy()
        self.prev_actions = np.zeros(12)
        self.prev_joint_vel = np.zeros(12)
        self.prev_time = time.time()
        self.startup_steps = 0
        self.debug_counter = 0

    def stop(self):
        self.control_active = False
        self.shutdown = True

# ====== MAIN ======

if __name__ == "__main__":
    
    # Initialize controller
    controller = SimMatchedMLPController("/home/ubuntu/mp2_mlp/policy_joyboy.pt")
    
    # Start control loop in background thread
    print("\nStarting control loop...")
    thread = threading.Thread(target=controller.control_loop)
    thread.start()
    
    speeds = {'1': 0.1, '2': 0.2, '3': 0.3}
    current_speed = 0.2
    
    try:
        while True:
            cmd = input("> ").strip().lower()
            
            if cmd == 'w': controller.set_velocity_command(current_speed, 0, 0)
            elif cmd == 's': controller.set_velocity_command(-current_speed, 0, 0)
            elif cmd == 'a': controller.set_velocity_command(0, current_speed, 0)
            elif cmd == 'd': controller.set_velocity_command(0, -current_speed, 0)
            elif cmd == 'q': controller.set_velocity_command(0, 0, current_speed)
            elif cmd == 'e': controller.set_velocity_command(0, 0, -current_speed)
            elif cmd in ['', ' ']: controller.set_velocity_command(0, 0, 0)
            elif cmd in speeds: 
                current_speed = speeds[cmd]
                print(f"Speed: {current_speed}")
            elif cmd == 'x': break
                
    except KeyboardInterrupt:
        pass
    
    controller.stop()
    thread.join()
    print("Done!")
