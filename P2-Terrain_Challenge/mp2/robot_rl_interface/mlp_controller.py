import numpy as np
import torch
import time
from MangDang.mini_pupper.ESP32Interface import ESP32Interface


class SimMatchedMLPController:
    """
    Controller that exactly matches simulation's observation space and action processing.
    
    Observation vector (60 dims):
        [0:3]   base_lin_vel      - Estimated from smoothed velocity command
        [3:6]   base_ang_vel      - From IMU gyroscope
        [6:9]   projected_gravity - From IMU accelerometer (normalized)
        [9:12]  velocity_commands - User input
        [12:24] joint_pos_rel     - Current joint pos minus default
        [24:36] joint_vel         - Joint velocities (computed from positions)
        [36:48] joint_effort      - From servo load feedback
        [48:60] prev_actions      - Previous policy output
    
    Action processing (matches Isaac Sim's JointPositionAction):
        target_position = default_joint_pos + (clipped_action * action_scale)
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
        # Order: LF(hip, thigh, calf), RF, LB, RB
        self.sim_default_positions = np.array([
            0.0,  0.785, -1.57,  # LF: hip=0°, thigh=45°, calf=-90°
            0.0,  0.785, -1.57,  # RF
            0.0,  0.785, -1.57,  # LB
            0.0,  0.785, -1.57,  # RB
        ])
        
        # Servo constants
        self.servo_center = 512
        self.servo_scale = 1024 / (2 * np.pi)  # servo units per radian
        
        # Action scale from training config (SpotActionsCfg.joint_pos.scale)
        self.ACTION_SCALE = 0.5
        
        # ====== CALIBRATION ======
        
        # Record hardware standing pose
        print("\n[CALIBRATION] Recording hardware standing pose...")
        raw = np.array(self.esp32.servos_get_position())
        self.standing_servos = raw[self.esp32_servo_order].astype(float)
        
        # Convert to angles (without offset initially)
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
        self.estimated_lin_vel = np.zeros(3)  # Smoothed estimate for base_lin_vel
        
        # Control state
        self.control_active = False
        self.shutdown = False
        self.debug_counter = 0
        self.startup_steps = 0
        
        # ====== CONTROL PARAMETERS ======
        
        self.CONTROL_FREQUENCY = 50  # Hz (matches sim: 500Hz physics / 10 decimation)
        self.startup_duration = 40 #25   # Steps to ramp up (0.5 sec at 50Hz)
        
        # Action processing
        self.action_clip = True
        self.action_smoothing = 0.5  # HOW JITTERY IS THE ROBOT?
        self.max_action_delta = 0.15  # Radians per step
        
        # Observation clipping (should match training noise ranges)
        self.obs_clip = {
            'lin_vel': 1.0,      # m/s
            'ang_vel': 2.0,      # rad/s
            'joint_pos': 0.8,    # rad
            'joint_vel': 1.5,    # rad/s (training noise was ±1.5)
            'joint_effort': 1.0, # normalized
        }
        
        # Base linear velocity estimation
        self.lin_vel_smoothing = 0.4  # How fast estimate follows command
        
        print("\n" + "=" * 60)
        print("Initialization complete!")
        print("=" * 60)
    
    # ====== CALIBRATION METHODS ======
    
    def _print_calibration_info(self):
        """Print detailed calibration information."""
        print(f"\nHardware standing servos: {self.standing_servos.astype(int)}")
        print(f"Hardware standing angles (rad): {np.round(self.hw_standing_angles, 3)}")
        print(f"Simulation default angles (rad): {np.round(self.sim_default_positions, 3)}")
        print(f"HW to Sim offset (rad): {np.round(self.hw_to_sim_offset, 3)}")
        
        print("\nPer-leg breakdown (HW -> Sim):")
        legs = ['LF', 'RF', 'LB', 'RB']
        joints = ['hip', 'thigh', 'calf']
        for i, leg in enumerate(legs):
            idx = i * 3
            hw = self.hw_standing_angles[idx:idx+3]
            sim = self.sim_default_positions[idx:idx+3]
            print(f"  {leg}: HW={np.round(hw, 2)} -> Sim={np.round(sim, 2)}")
    
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
        
        # Gyro offset (should be near zero when still)
        self.gyro_offset = np.mean(gyro_samples, axis=0)
        gyro_std = np.std(gyro_samples, axis=0)
        
        # Detect if gyro is in deg/s or rad/s
        # If values are > 10 when still, probably deg/s
        if np.max(np.abs(self.gyro_offset)) > 5:
            self.gyro_scale = np.pi / 180.0  # Convert deg/s to rad/s
            print(f"  Gyro appears to be in deg/s, will convert to rad/s")
        else:
            self.gyro_scale = 1.0  # Already in rad/s
            print(f"  Gyro appears to be in rad/s")
        
        print(f"  Gyro offset: {np.round(self.gyro_offset, 3)}")
        print(f"  Gyro noise (std): {np.round(gyro_std, 3)}")
        
        # Check accelerometer (should show ~9.81 or ~1g on one axis)
        accel_mean = np.mean(accel_samples, axis=0)
        accel_magnitude = np.linalg.norm(accel_mean)
        print(f"  Accel mean: {np.round(accel_mean, 3)}, magnitude: {accel_magnitude:.2f}")
        
        # Detect accel units (m/s² vs g's)
        if accel_magnitude > 5:
            self.accel_scale = 1.0  # Already in m/s²
            print(f"  Accel appears to be in m/s²")
        else:
            self.accel_scale = 9.81  # Convert g's to m/s²
            print(f"  Accel appears to be in g's, will convert to m/s²")
    
    def _calibrate_load(self, samples=30):
        """Calibrate servo load scaling by reading values at rest."""
        load_samples = []
        
        for _ in range(samples):
            load = self.esp32.servos_get_load()
            if load is not None:
                load_samples.append(load)
            time.sleep(0.02)
        
        if not load_samples:
            print("  WARNING: Could not read servo load, using default scale")
            self.load_scale = 1000.0
            self.load_offset = np.zeros(12)
            return
        
        load_samples = np.array(load_samples)
        
        # Calculate offset (resting load) and range
        self.load_offset = np.mean(load_samples, axis=0)
        load_std = np.std(load_samples, axis=0)
        load_range = np.max(np.abs(load_samples - self.load_offset))
        
        # Estimate scale (we'll normalize to roughly [-1, 1])
        # Use a conservative estimate; can tune later
        self.load_scale = max(load_range * 2, 100.0)  # At least 100 to avoid division issues
        
        print(f"  Load offset (resting): {np.round(self.load_offset, 1)}")
        print(f"  Load noise (std): {np.round(load_std, 1)}")
        print(f"  Load scale: {self.load_scale:.1f}")
        print(f"  TIP: Apply force to legs and check if joint_effort values look reasonable")
    
    # ====== SERVO CONVERSION METHODS ======
    
    def _servos_to_angles(self, servo_positions):
        """Convert servo positions to joint angles in simulation frame (radians)."""
        servo_delta = servo_positions - self.servo_center
        angles_raw = servo_delta / self.servo_scale
        angles_hw = angles_raw * self.direction_multipliers
        
        if self.use_offset and hasattr(self, 'hw_to_sim_offset'):
            return angles_hw + self.hw_to_sim_offset
        return angles_hw
    
    def _angles_to_servos(self, angles_sim):
        """Convert joint angles (simulation frame, radians) to servo positions."""
        angles_hw = angles_sim - self.hw_to_sim_offset
        angles_raw = angles_hw * self.direction_multipliers
        servo_delta = angles_raw * self.servo_scale
        return self.servo_center + servo_delta
    
    # ====== SENSOR READING METHODS ======
    
    def read_joint_positions(self):
        """Read current joint angles in simulation frame."""
        raw = np.array(self.esp32.servos_get_position())
        isaac_ordered = raw[self.esp32_servo_order].astype(float)
        return self._servos_to_angles(isaac_ordered)
    
    def read_joint_efforts(self):
        """Read servo load/torque feedback, normalized to ~[-1, 1]."""
        raw_load = self.esp32.servos_get_load()
        if raw_load is None:
            return np.zeros(12)
        
        raw_load = np.array(raw_load, dtype=float)
        
        # Reorder to Isaac ordering
        isaac_ordered = raw_load[self.esp32_servo_order]
        
        # Remove offset and normalize
        centered = isaac_ordered - self.load_offset[self.esp32_servo_order]
        normalized = centered / self.load_scale
        
        # Apply direction multipliers (torque direction matches joint direction)
        normalized = normalized * self.direction_multipliers
        
        return np.clip(normalized, -1.0, 1.0)
    
    def write_joint_positions(self, target_angles):
        """Write joint positions (simulation frame) to servos."""
        target_servos = self._angles_to_servos(target_angles)
        target_servos = np.clip(target_servos, 180, 844)  # Servo limits
        
        # Reorder for ESP32
        esp32_out = np.zeros(12)
        esp32_out[self.esp32_servo_order] = target_servos
        
        self.esp32.servos_set_position([int(p) for p in esp32_out])
    
    # ====== OBSERVATION BUILDING ======
    
    def get_observation(self):
        """
        Build 60-dim observation vector matching simulation exactly.
        
        Returns:
            obs: (60,) numpy array
            joint_pos_rel: (12,) for debugging
            joint_vel: (12,) for debugging
            joint_effort: (12,) for debugging
        """
        current_time = time.time()
        dt = current_time - self.prev_time
        
        # Read sensors
        imu_data = self.esp32.imu_get_data()
        current_angles = self.read_joint_positions()
        joint_effort = self.read_joint_efforts()
        
        # ---- 1. Base Linear Velocity (3 dims) ----
        # Estimated by smoothing toward velocity command
        # This approximates the policy's expectation that base_lin_vel tracks velocity_commands
        alpha = self.lin_vel_smoothing
        self.estimated_lin_vel = (alpha * self.velocity_command + 
                                   (1 - alpha) * self.estimated_lin_vel)
        base_lin_vel = self.estimated_lin_vel.copy()
        base_lin_vel = np.clip(base_lin_vel, -self.obs_clip['lin_vel'], self.obs_clip['lin_vel'])
        
        # ---- 2. Base Angular Velocity (3 dims) ----
        # From IMU gyroscope
        gyro_raw = np.array([imu_data['gx'], imu_data['gy'], imu_data['gz']])
        base_ang_vel = (gyro_raw - self.gyro_offset) * self.gyro_scale
        base_ang_vel = np.clip(base_ang_vel, -self.obs_clip['ang_vel'], self.obs_clip['ang_vel'])
        
        # ---- 3. Projected Gravity (3 dims) ----
        # Normalized accelerometer reading
        # In Isaac Sim with z-down: standing upright -> projected_gravity ≈ [0, 0, -1]
        accel_raw = np.array([imu_data['ax'], imu_data['ay'], imu_data['az']])
        accel = accel_raw * self.accel_scale
        accel_norm = np.linalg.norm(accel)
        
        if accel_norm > 0.1:
            # Normalize to unit vector
            projected_gravity = accel / accel_norm
        else:
            # Fallback if accel reading is invalid
            projected_gravity = np.array([0.0, 0.0, -1.0])
        
        # ---- 4. Velocity Commands (3 dims) ----
        velocity_commands = self.velocity_command.copy()
        
        # ---- 5. Joint Positions Relative to Default (12 dims) ----
        joint_pos_rel = current_angles - self.sim_default_positions
        joint_pos_rel = np.clip(joint_pos_rel, -self.obs_clip['joint_pos'], self.obs_clip['joint_pos'])
        
        # ---- 6. Joint Velocities (12 dims) ----
        if dt > 0.001:
            joint_vel_raw = (current_angles - self.prev_joint_angles) / dt
            # Low-pass filter to reduce noise
            alpha_vel = 0.4
            joint_vel = alpha_vel * joint_vel_raw + (1 - alpha_vel) * self.prev_joint_vel
        else:
            joint_vel = self.prev_joint_vel.copy()
        
        joint_vel = np.clip(joint_vel, -self.obs_clip['joint_vel'], self.obs_clip['joint_vel'])
        
        # ---- 7. Joint Efforts (12 dims) ----
        joint_effort = np.clip(joint_effort, -self.obs_clip['joint_effort'], self.obs_clip['joint_effort'])
        
        # ---- 8. Previous Actions (12 dims) ----
        prev_actions = self.prev_actions.copy()
        
        # Update state for next iteration
        self.prev_joint_angles = current_angles.copy()
        self.prev_joint_vel = joint_vel.copy()
        self.prev_time = current_time
        
        # ---- Concatenate in correct order ----
        obs = np.concatenate([
            base_lin_vel,       # 0:3
            base_ang_vel,       # 3:6
            projected_gravity,  # 6:9
            velocity_commands,  # 9:12
            joint_pos_rel,      # 12:24
            joint_vel,          # 24:36
            joint_effort,       # 36:48
            prev_actions,       # 48:60
        ]).astype(np.float32)
        
        return obs, joint_pos_rel, joint_vel, joint_effort
    
    # ====== ACTION PROCESSING ======
    
    def process_actions(self, raw_actions):
        """
        Process policy output exactly as simulation's JointPositionAction.
        
        Steps:
            1. Clip to [-1, 1]
            2. Scale by ACTION_SCALE (0.5)
            3. Add to default positions
        
        Returns:
            target_positions: Absolute joint positions in radians
            clipped_actions: The clipped actions (for storing as prev_actions)
        """
        if self.action_clip:
            clipped = np.clip(raw_actions, -1.0, 1.0)
        else:
            clipped = raw_actions.copy()
        
        scaled = clipped * self.ACTION_SCALE
        target_positions = self.sim_default_positions + scaled
        
        return target_positions, clipped
    
    # ====== CONTROL LOOP ======
    
    def control_step(self):
        """Single control loop iteration."""
        # Get observation
        obs, joint_pos_rel, joint_vel, joint_effort = self.get_observation()
        
        # Policy inference
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            raw_actions = self.policy(obs_tensor).squeeze().numpy()
        
        # Process actions (NO gait modulation - let the policy do its job!)
        target_positions, clipped_actions = self.process_actions(raw_actions)
        
        # Startup ramp (gradually blend from default to policy output)
        if self.startup_steps < self.startup_duration:
            ramp = self.startup_steps / self.startup_duration
            target_positions = (self.sim_default_positions + 
                              (target_positions - self.sim_default_positions) * ramp)
            clipped_actions = clipped_actions * ramp
            self.startup_steps += 1
        
        # Optional smoothing
        if self.action_smoothing > 0 and hasattr(self, 'prev_target_positions'):
            # Blend with previous targets
            alpha = self.action_smoothing
            smoothed = alpha * target_positions + (1 - alpha) * self.prev_target_positions
            
            # Rate limit the change
            delta = smoothed - self.prev_target_positions
            delta = np.clip(delta, -self.max_action_delta, self.max_action_delta)
            target_positions = self.prev_target_positions + delta
        
        self.prev_target_positions = target_positions.copy()
        
        # Send to robot
        self.write_joint_positions(target_positions)
        
        # Update state
        self.prev_actions = clipped_actions.copy()
        
        # Debug output
        self.debug_counter += 1
        if self.debug_counter % 50 == 0:
            self._print_debug(obs, raw_actions, target_positions, joint_effort)
        
        return clipped_actions
    
    def _print_debug(self, obs, raw_actions, target_positions, joint_effort):
        """Print debug information."""
        print(f"\n--- Step {self.debug_counter} ---")
        print(f"Cmd: [{self.velocity_command[0]:+.2f}, {self.velocity_command[1]:+.2f}, {self.velocity_command[2]:+.2f}]")
        print(f"Est vel: [{self.estimated_lin_vel[0]:+.2f}, {self.estimated_lin_vel[1]:+.2f}, {self.estimated_lin_vel[2]:+.2f}]")
        print(f"Gravity: [{obs[6]:+.2f}, {obs[7]:+.2f}, {obs[8]:+.2f}]")
        print(f"Joint pos rel: [{obs[12:24].min():+.3f}, {obs[12:24].max():+.3f}]")
        print(f"Joint vel: [{obs[24:36].min():+.2f}, {obs[24:36].max():+.2f}]")
        print(f"Joint effort: [{joint_effort.min():+.2f}, {joint_effort.max():+.2f}]")
        print(f"Raw actions: [{raw_actions.min():+.2f}, {raw_actions.max():+.2f}]")
        print(f"Target pos: [{target_positions.min():+.2f}, {target_positions.max():+.2f}]")
        
        if self.startup_steps < self.startup_duration:
            print(f"Startup: {self.startup_steps}/{self.startup_duration}")
    
    def control_loop(self):
        """Main control loop."""
        dt_target = 1.0 / self.CONTROL_FREQUENCY
        
        print(f"\n{'='*60}")
        print(f"Control loop running at {self.CONTROL_FREQUENCY}Hz")
        print(f"Action scale: {self.ACTION_SCALE}")
        print(f"Smoothing: {self.action_smoothing}")
        print(f"{'='*60}")
        print("\nCommands:")
        print("  w/s     - Forward/Backward")
        print("  a/d     - Strafe Left/Right")
        print("  q/e     - Turn Left/Right")
        print("  space   - Stop")
        print("  x       - Exit")
        print("  1/2/3   - Speed: Slow/Medium/Fast")
        print()
        
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
        """Set velocity command with proper clipping."""
        self.velocity_command = np.array([
            np.clip(vx, -0.35, 0.40),
            np.clip(vy, -0.35, 0.35),
            np.clip(vyaw, -0.30, 0.30)
        ])
        
        if np.any(np.abs(self.velocity_command) > 0.01):
            if not self.control_active:
                # Reset state on activation
                self._reset_control_state()
            self.control_active = True
            print(f"Active: cmd=[{self.velocity_command[0]:+.2f}, "
                  f"{self.velocity_command[1]:+.2f}, {self.velocity_command[2]:+.2f}]")
        else:
            self.control_active = False
            self.startup_steps = 0
            self.estimated_lin_vel = np.zeros(3)
            print("Stopped")
    
    def _reset_control_state(self):
        """Reset control state when starting movement."""
        current_angles = self.read_joint_positions()
        self.prev_joint_angles = current_angles.copy()
        self.prev_target_positions = current_angles.copy()
        self.prev_actions = np.zeros(12)
        self.prev_joint_vel = np.zeros(12)
        self.prev_time = time.time()
        self.startup_steps = 0
        self.debug_counter = 0
        self.estimated_lin_vel = np.zeros(3)
    
    def stop(self):
        """Stop the controller."""
        self.control_active = False
        self.shutdown = True


# ====== TEST FUNCTIONS ======

def test_sensors(controller):
    """Test sensor readings."""
    print("\n" + "="*60)
    print("SENSOR TEST")
    print("="*60)
    
    print("\nReading sensors 5 times...")
    for i in range(5):
        obs, pos_rel, vel, effort = controller.get_observation()
        print(f"\nReading {i+1}:")
        print(f"  base_lin_vel:  {obs[0:3]}")
        print(f"  base_ang_vel:  {obs[3:6]}")
        print(f"  proj_gravity:  {obs[6:9]}")
        print(f"  velocity_cmd:  {obs[9:12]}")
        print(f"  joint_pos_rel: min={pos_rel.min():.3f}, max={pos_rel.max():.3f}")
        print(f"  joint_vel:     min={vel.min():.3f}, max={vel.max():.3f}")
        print(f"  joint_effort:  min={effort.min():.3f}, max={effort.max():.3f}")
        time.sleep(0.2)


def test_actions(controller):
    """Test action processing."""
    print("\n" + "="*60)
    print("ACTION PROCESSING TEST")
    print("="*60)
    
    # Test zero actions
    test_actions = np.zeros(12)
    target, clipped = controller.process_actions(test_actions)
    print(f"\nZero actions:")
    print(f"  Target positions: {np.round(target, 3)}")
    print(f"  Sim defaults:     {np.round(controller.sim_default_positions, 3)}")
    print(f"  Match: {np.allclose(target, controller.sim_default_positions)}")
    
    # Test +0.5 actions (should give max positive offset)
    test_actions = np.ones(12) * 0.5
    target, clipped = controller.process_actions(test_actions)
    expected = controller.sim_default_positions + 0.5 * controller.ACTION_SCALE
    print(f"\n+0.5 actions:")
    print(f"  Target positions: {np.round(target, 3)}")
    print(f"  Expected:         {np.round(expected, 3)}")
    print(f"  Match: {np.allclose(target, expected)}")


def test_load_response(controller):
    """Interactive test to verify load sensing works."""
    print("\n" + "="*60)
    print("LOAD SENSOR TEST")
    print("="*60)
    print("\nPush on each leg and watch the joint_effort values change.")
    print("Press Ctrl+C to stop.\n")
    
    try:
        while True:
            effort = controller.read_joint_efforts()
            # Format nicely
            legs = ['LF', 'RF', 'LB', 'RB']
            output = "Effort: "
            for i, leg in enumerate(legs):
                idx = i * 3
                e = effort[idx:idx+3]
                output += f"{leg}=[{e[0]:+.2f},{e[1]:+.2f},{e[2]:+.2f}] "
            print(f"\r{output}", end='', flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n")


# ====== MAIN ======

if __name__ == "__main__":
    import threading
    
    # Initialize controller
    controller = SimMatchedMLPController("/home/ubuntu/mp2_mlp/policy_joyboy.pt")
    
    # Run tests
    test_sensors(controller)
    test_actions(controller)
    
    # Ask if user wants to test load
    print("\nTest load sensors interactively? (y/n): ", end='')
    if input().strip().lower() == 'y':
        test_load_response(controller)
    
    # Start control loop in background thread
    print("\nStarting control loop...")
    thread = threading.Thread(target=controller.control_loop)
    thread.start()
    
    # Speed settings
    speeds = {
        '1': 0.1,   # Slow
        '2': 0.2,   # Medium  
        '3': 0.3,   # Fast
    }
    current_speed = 0.2
    
    try:
        while True:
            cmd = input("> ").strip().lower()
            
            if cmd == 'w':
                controller.set_velocity_command(current_speed, 0, 0)
            elif cmd == 's':
                controller.set_velocity_command(-current_speed, 0, 0)
            elif cmd == 'a':
                controller.set_velocity_command(0, current_speed, 0)
            elif cmd == 'd':
                controller.set_velocity_command(0, -current_speed, 0)
            elif cmd == 'q':
                controller.set_velocity_command(0, 0, current_speed)
            elif cmd == 'e':
                controller.set_velocity_command(0, 0, -current_speed)
            elif cmd in ['', ' ']:
                controller.set_velocity_command(0, 0, 0)
            elif cmd in speeds:
                current_speed = speeds[cmd]
                print(f"Speed set to {current_speed}")
            elif cmd == 'x':
                break
            elif cmd == 't':
                # Quick test of current observation
                obs, _, _, effort = controller.get_observation()
                print(f"Obs shape: {obs.shape}")
                print(f"Effort: {np.round(effort, 2)}")
            else:
                print("Unknown command. Use w/s/a/d/q/e, space, 1/2/3, or x to exit")
                
    except KeyboardInterrupt:
        pass
    
    controller.stop()
    thread.join()
    print("Done!")