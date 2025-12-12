"""
Improved Mini Pupper 2 RL Controller
=====================================
Addresses all sim-to-real gaps identified from data analysis in simulation adn realoty:

1. FRAME HANDLING: Sim uses -1.57 rad calf default, real uses -0.785 rad
   - Brain lives in sim frame (thinks robot is in "squat")
   - Hardware offset applied only at output
   - Position feedback uses actual servo readings, transformed to sim frame

2. DYNAMIC BASE_LIN_VEL: Estimated from IMU acceleration integration
   - Not constant, varies like in simulation

3. SMOOTHED JOINT VELOCITIES: Low-pass filter to reduce noise
   - Real servo feedback is noisy, sim is clean

4. CORRECTED EFFORT SIGNS: Based on data analysis
   - Flip signs for joints that showed opposite polarity

5. PROPER SCALING: Action scale tuned for real servos
"""

import numpy as np
import torch
import time
import threading
from collections import deque
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration


class ImprovedRLController:
    """
    Controller that properly bridges simulation and real robot frames.
    """
    
    def __init__(self, policy_path="/home/ubuntu/mp2_mlp/policy_joyboy.pt"):
        print("=" * 70)
        print("IMPROVED RL CONTROLLER - Sim2Real Gap Fixes")
        print("=" * 70)
        
        # ==================== HARDWARE ====================
        self.config = Configuration()
        self.hardware = HardwareInterface()
        self.pwm_params = self.hardware.pwm_params
        self.servo_params = self.hardware.servo_params
        self.esp32 = self.pwm_params.esp32
        time.sleep(0.3)
        
        # ==================== POLICY ====================
        try:
            self.policy = torch.jit.load(policy_path)
            self.policy.eval()
            print(f"[OK] Policy loaded: {policy_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load policy: {e}")
            raise
        
        # ==================== FRAME CONFIGURATION ====================
        # Simulation default pose (what the policy was trained with)
        # Thigh at 45°, Calf at -90° relative to thigh
        self.sim_default_positions = np.array([
            0.0,  0.785, -1.57,   # LF: hip, thigh, calf
            0.0,  0.785, -1.57,   # RF
            0.0,  0.785, -1.57,   # LB
            0.0,  0.785, -1.57,   # RB
        ])
        
        # The offset between sim frame and real frame for calf joints
        # Real robot: calf neutral = -0.785 rad (45° from vertical)
        # Sim robot:  calf neutral = -1.57 rad (90° from thigh)
        # Same physical pose, different reference frames
        # When reading real positions: real_calf - 0.785 = sim_calf
        # When writing to real hardware: sim_calf + 0.785 = real_calf
        self.CALF_FRAME_OFFSET = 0.785
        
        # Joint limits (in sim frame)
        self.joint_lower_limits = np.array([-0.5, 0.0, -2.5] * 4)
        self.joint_upper_limits = np.array([0.5, 1.5, -0.5] * 4)
        
        # Hardware mapping: Isaac [LF, RF, LB, RB] -> Hardware columns
        self.hw_to_isaac_leg = {0: 1, 1: 0, 2: 3, 3: 2}  # HW col -> Isaac leg
        self.isaac_to_hw_col = {1: 0, 0: 1, 3: 2, 2: 3}  # Isaac leg -> HW col
        
        # ==================== ACTION TUNING ====================
        self.ACTION_SCALE = 0.35           # Conservative for real servos
        self.action_smoothing_alpha = 0.4  # 0=no smoothing, 1=instant
        
        # ==================== VELOCITY SMOOTHING ====================
        self.vel_filter_alpha = 0.3        # Low-pass filter for joint velocities
        self.vel_history_size = 3          # Median filter window
        self.vel_histories = [deque(maxlen=self.vel_history_size) for _ in range(12)]
        
        # ==================== DYNAMIC BASE_LIN_VEL ====================
        # Estimate linear velocity from IMU acceleration integration
        self.estimated_lin_vel = np.zeros(3)
        self.lin_vel_decay = 0.95          # Decay factor (drift correction)
        self.lin_vel_accel_gain = 0.02     # Integration gain
        
        # ==================== IMU CONFIGURATION ====================
        self.gyro_scale = np.pi / 180.0    # deg/s to rad/s
        self.accel_scale = 9.81            # g to m/s²
        self.gyro_offset = np.zeros(3)
        self.accel_offset = np.zeros(3)
        
        # ==================== EFFORT DIRECTION ====================
        # Based on data analysis: flip signs for joints where real != sim
        # From analysis: lf1_lf2, lf2_lf3, base_rf1, base_lb1, lb2_lb3, rb2_rb3 need flip
        self.effort_direction = np.array([
            +1.0, -1.0, -1.0,   # LF: hip OK, thigh FLIP, calf FLIP
            -1.0, +1.0, +1.0,   # RF: hip FLIP, thigh OK, calf OK
            -1.0, +1.0, -1.0,   # LB: hip FLIP, thigh OK, calf FLIP
            +1.0, +1.0, -1.0,   # RB: hip OK, thigh OK, calf FLIP
        ])
        self.effort_scale = 5000.0
        self.effort_offset = np.zeros(12)
        
        # ==================== STATE ====================
        self.prev_actions = np.zeros(12)
        self.prev_joint_angles_sim = self.sim_default_positions.copy()
        self.prev_joint_vel = np.zeros(12)
        self.prev_time = time.time()
        
        self.velocity_command = np.zeros(3)
        self.control_active = False
        self.shutdown = False
        self.startup_steps = 0
        self.startup_duration = 50
        self.debug_counter = 0
        
        self.CONTROL_FREQUENCY = 50
        
        # ==================== CALIBRATION ====================
        print("[CALIBRATING] IMU and effort sensors...")
        self._calibrate_sensors()
        print("[READY] Controller initialized")
        print("=" * 70)
    
    def _calibrate_sensors(self, samples=50):
        """Calibrate IMU and effort sensor offsets."""
        gyro_samples = []
        accel_samples = []
        effort_samples = []
        
        for _ in range(samples):
            imu = self.esp32.imu_get_data()
            if imu:
                gyro_samples.append([imu['gx'], imu['gy'], imu['gz']])
                accel_samples.append([imu['ax'], imu['ay'], imu['az']])
            
            effort = self.esp32.servos_get_load()
            if effort:
                effort_samples.append(effort)
            
            time.sleep(0.02)
        
        if gyro_samples:
            self.gyro_offset = np.mean(gyro_samples, axis=0)
        if accel_samples:
            # Gravity should be [0, 0, -1g] when level
            self.accel_offset = np.mean(accel_samples, axis=0)
            self.accel_offset[2] -= 1.0  # Remove gravity
        if effort_samples:
            self.effort_offset = np.mean(effort_samples, axis=0)
        
        print(f"   Gyro offset: {self.gyro_offset}")
        print(f"   Accel offset: {self.accel_offset}")
    
    def _read_joint_positions_sim_frame(self):
        """
        Read actual servo positions and convert to simulation frame.
        """
        raw = self.esp32.servos_get_position()
        if raw is None:
            return self.prev_joint_angles_sim.copy()
        
        # Convert raw PWM to angles in hardware frame
        angles_hw = np.zeros((3, 4))
        for leg in range(4):
            for axis in range(3):
                servo_id = self.pwm_params.servo_ids[axis, leg]
                pos = raw[servo_id - 1]
                dev = (self.servo_params.neutral_position - pos) / self.servo_params.micros_per_rad
                angle = dev / self.servo_params.servo_multipliers[axis, leg] + \
                        self.servo_params.neutral_angles[axis, leg]
                angles_hw[axis, leg] = angle
        
        # Reorder to Isaac format [LF, RF, LB, RB]
        angles_isaac = np.zeros(12)
        for hw_col, isaac_leg in self.hw_to_isaac_leg.items():
            for axis in range(3):
                angles_isaac[isaac_leg * 3 + axis] = angles_hw[axis, hw_col]
        
        # Transform calf angles from real frame to sim frame
        # Real: -0.785 neutral -> Sim: -1.57 neutral
        # So: sim_calf = real_calf - OFFSET
        angles_isaac[2::3] -= self.CALF_FRAME_OFFSET
        
        return angles_isaac
    
    def _read_joint_efforts(self):
        """Read joint efforts and apply direction corrections."""
        raw = self.esp32.servos_get_load()
        if raw is None:
            return np.zeros(12)
        
        raw = np.array(raw, dtype=float)
        
        # Reorder to Isaac format
        effort_isaac = np.zeros(12)
        for hw_col, isaac_leg in self.hw_to_isaac_leg.items():
            for axis in range(3):
                servo_id = self.pwm_params.servo_ids[axis, hw_col]
                effort_isaac[isaac_leg * 3 + axis] = raw[servo_id - 1]
        
        # Remove offset and normalize
        centered = effort_isaac - self._reorder_effort_offset()
        normalized = (centered / self.effort_scale) * self.effort_direction
        
        return normalized
    
    def _reorder_effort_offset(self):
        """Reorder effort offset to Isaac format."""
        offset_isaac = np.zeros(12)
        for hw_col, isaac_leg in self.hw_to_isaac_leg.items():
            for axis in range(3):
                servo_id = self.pwm_params.servo_ids[axis, hw_col]
                offset_isaac[isaac_leg * 3 + axis] = self.effort_offset[servo_id - 1]
        return offset_isaac
    
    def _compute_smoothed_velocity(self, current_angles, dt):
        """
        Compute joint velocities with smoothing to reduce noise.
        Uses combination of low-pass filter and median filter.
        """
        if dt < 0.001:
            return self.prev_joint_vel.copy()
        
        # Raw velocity
        raw_vel = (current_angles - self.prev_joint_angles_sim) / dt
        
        # Clip extreme values (servo glitches)
        raw_vel = np.clip(raw_vel, -15.0, 15.0)
        
        # Add to history and compute median
        smoothed_vel = np.zeros(12)
        for i in range(12):
            self.vel_histories[i].append(raw_vel[i])
            if len(self.vel_histories[i]) > 0:
                smoothed_vel[i] = np.median(list(self.vel_histories[i]))
        
        # Low-pass filter
        filtered_vel = (self.vel_filter_alpha * smoothed_vel + 
                       (1 - self.vel_filter_alpha) * self.prev_joint_vel)
        
        return filtered_vel
    
    def _estimate_base_lin_vel(self, accel_body, dt):
        """
        Estimate base linear velocity from IMU acceleration.
        This provides dynamic velocity estimate similar to simulation.
        """
        # Remove gravity component (assuming robot is roughly level)
        # In body frame, gravity is approximately [0, 0, -9.81]
        accel_corrected = accel_body.copy()
        accel_corrected[2] += 9.81  # Remove gravity
        
        # Integrate acceleration to get velocity
        self.estimated_lin_vel += accel_corrected * self.lin_vel_accel_gain * dt
        
        # Apply decay to prevent drift
        self.estimated_lin_vel *= self.lin_vel_decay
        
        # Blend with command-based estimate for stability
        cmd_estimate = self.velocity_command * 0.7
        blended = 0.7 * self.estimated_lin_vel + 0.3 * cmd_estimate
        
        # Clip to reasonable range
        blended = np.clip(blended, -0.5, 0.5)
        
        return blended
    
    def _isaac_to_hardware_matrix(self, flat_angles_sim_frame):
        """
        Convert Isaac-order angles (sim frame) to hardware matrix.
        Applies calf frame offset for hardware.
        """
        # Apply calf offset: sim -> real frame
        angles_real_frame = flat_angles_sim_frame.copy()
        angles_real_frame[2::3] += self.CALF_FRAME_OFFSET
        
        # Reorder to hardware format
        matrix = np.zeros((3, 4))
        matrix[:, 1] = angles_real_frame[0:3]   # LF -> hw col 1
        matrix[:, 0] = angles_real_frame[3:6]   # RF -> hw col 0
        matrix[:, 3] = angles_real_frame[6:9]   # LB -> hw col 3
        matrix[:, 2] = angles_real_frame[9:12]  # RB -> hw col 2
        
        return matrix
    
    def get_observation(self):
        """
        Build 60-dim observation matching training format.
        All values computed to match simulation distributions.
        """
        current_time = time.time()
        dt = current_time - self.prev_time
        
        # Read sensors
        imu = self.esp32.imu_get_data()
        current_angles_sim = self._read_joint_positions_sim_frame()
        joint_effort = self._read_joint_efforts()
        
        # ===== BASE LINEAR VELOCITY (DYNAMIC) =====
        accel_raw = np.array([imu['ax'], imu['ay'], imu['az']])
        accel = (accel_raw - self.accel_offset) * self.accel_scale
        base_lin_vel = self._estimate_base_lin_vel(accel, dt)
        
        # ===== BASE ANGULAR VELOCITY =====
        gyro_raw = np.array([imu['gx'], imu['gy'], imu['gz']])
        base_ang_vel = (gyro_raw - self.gyro_offset) * self.gyro_scale
        
        # ===== PROJECTED GRAVITY =====
        accel_for_gravity = accel_raw * self.accel_scale
        accel_norm = np.linalg.norm(accel_for_gravity)
        if accel_norm > 0.1:
            projected_gravity = accel_for_gravity / accel_norm
        else:
            projected_gravity = np.array([0.0, 0.0, -1.0])
        
        # ===== VELOCITY COMMANDS =====
        velocity_commands = self.velocity_command.copy()
        
        # ===== JOINT POSITIONS (relative to default, sim frame) =====
        joint_pos_rel = current_angles_sim - self.sim_default_positions
        
        # ===== JOINT VELOCITIES (smoothed) =====
        joint_vel = self._compute_smoothed_velocity(current_angles_sim, dt)
        
        # ===== UPDATE STATE =====
        self.prev_joint_angles_sim = current_angles_sim.copy()
        self.prev_joint_vel = joint_vel.copy()
        self.prev_time = current_time
        
        # ===== PREVIOUS ACTIONS =====
        prev_actions = self.prev_actions.copy()
        
        # ===== BUILD OBSERVATION (60 dims) =====
        obs = np.concatenate([
            base_lin_vel,         # 0:3
            base_ang_vel,         # 3:6
            projected_gravity,    # 6:9
            velocity_commands,    # 9:12
            joint_pos_rel,        # 12:24
            joint_vel,            # 24:36
            joint_effort,         # 36:48
            prev_actions,         # 48:60
        ]).astype(np.float32)
        
        return obs
    
    def control_step(self):
        """Execute one control step."""
        obs = self.get_observation()
        
        # Run policy
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            raw_actions = self.policy(obs_tensor).squeeze().numpy()
        
        # Clip actions to valid range
        clipped_actions = np.clip(raw_actions, -1.0, 1.0)
        
        # Startup fade-in
        if self.startup_steps < self.startup_duration:
            fade = self.startup_steps / self.startup_duration
            self.startup_steps += 1
        else:
            fade = 1.0
        
        faded_actions = clipped_actions * fade
        
        # Action smoothing
        smoothed_actions = (self.action_smoothing_alpha * faded_actions + 
                          (1 - self.action_smoothing_alpha) * self.prev_actions)
        self.prev_actions = smoothed_actions.copy()
        
        # Compute target positions (sim frame)
        target_sim = self.sim_default_positions + smoothed_actions * self.ACTION_SCALE
        target_sim = np.clip(target_sim, self.joint_lower_limits, self.joint_upper_limits)
        
        # Send to hardware (converts to real frame internally)
        target_matrix = self._isaac_to_hardware_matrix(target_sim)
        self.hardware.set_actuator_postions(target_matrix)
        
        # Debug output
        self.debug_counter += 1
        if self.debug_counter % 50 == 0:
            obs_ranges = {
                'lin_vel': obs[0:3],
                'ang_vel': obs[3:6],
                'gravity': obs[6:9],
                'pos_rel': obs[12:24],
            }
            print(f"Step {self.debug_counter}: "
                  f"lin_vel=[{obs[0]:.2f},{obs[1]:.2f},{obs[2]:.2f}] "
                  f"act=[{smoothed_actions.min():+.2f},{smoothed_actions.max():+.2f}]")
    
    def control_loop(self):
        """Main control loop."""
        dt_target = 1.0 / self.CONTROL_FREQUENCY
        print(f"\n[RUNNING] Control loop at {self.CONTROL_FREQUENCY}Hz")
        
        while not self.shutdown:
            loop_start = time.time()
            
            if self.control_active:
                try:
                    self.control_step()
                except Exception as e:
                    print(f"[ERROR] Control step failed: {e}")
                    import traceback
                    traceback.print_exc()
                    self.control_active = False
            
            elapsed = time.time() - loop_start
            if elapsed < dt_target:
                time.sleep(dt_target - elapsed)
    
    def set_velocity_command(self, vx, vy, vyaw):
        """Set velocity command."""
        self.velocity_command = np.array([vx, vy, vyaw])
        
        if np.any(np.abs(self.velocity_command) > 0.01):
            if not self.control_active:
                self.startup_steps = 0  # Reset fade-in
                self.estimated_lin_vel = np.zeros(3)  # Reset velocity estimate
            self.control_active = True
            print(f"[CMD] vx={vx:.2f}, vy={vy:.2f}, vyaw={vyaw:.2f}")
        else:
            self.control_active = False
            print("[CMD] STOPPED")
    
    def stop(self):
        """Shutdown controller."""
        self.control_active = False
        self.shutdown = True
        print("[SHUTDOWN] Controller stopped")


# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("MINI PUPPER 2 - IMPROVED RL CONTROLLER")
    print("="*70)
    print("""
Controls:
  w/s  - Forward/Backward
  a/d  - Strafe Left/Right  
  q/e  - Turn Left/Right
  SPACE or ENTER - Stop
  +/-  - Increase/Decrease speed
  x    - Exit
""")
    
    controller = ImprovedRLController("/home/ubuntu/mp2_mlp/policy_joyboy.pt")
    
    # Start control thread
    control_thread = threading.Thread(target=controller.control_loop, daemon=True)
    control_thread.start()
    
    speed = 0.15  # Start conservative
    
    try:
        while True:
            cmd = input("> ").strip().lower()
            
            if cmd == 'w':
                controller.set_velocity_command(speed, 0, 0)
            elif cmd == 's':
                controller.set_velocity_command(-speed, 0, 0)
            elif cmd == 'a':
                controller.set_velocity_command(0, speed, 0)
            elif cmd == 'd':
                controller.set_velocity_command(0, -speed, 0)
            elif cmd == 'q':
                controller.set_velocity_command(0, 0, speed * 2)
            elif cmd == 'e':
                controller.set_velocity_command(0, 0, -speed * 2)
            elif cmd in ['', ' ']:
                controller.set_velocity_command(0, 0, 0)
            elif cmd == '+':
                speed = min(speed + 0.05, 0.5)
                print(f"[SPEED] {speed:.2f} m/s")
            elif cmd == '-':
                speed = max(speed - 0.05, 0.05)
                print(f"[SPEED] {speed:.2f} m/s")
            elif cmd == 'x':
                break
            elif cmd == 'debug':
                # Print current state
                obs = controller.get_observation()
                print(f"Observation shape: {obs.shape}")
                print(f"  base_lin_vel: {obs[0:3]}")
                print(f"  base_ang_vel: {obs[3:6]}")
                print(f"  gravity: {obs[6:9]}")
                print(f"  commands: {obs[9:12]}")
                print(f"  joint_pos_rel: {obs[12:24]}")
                print(f"  joint_vel: {obs[24:36]}")
                print(f"  joint_effort: {obs[36:48]}")
                print(f"  prev_actions: {obs[48:60]}")
            else:
                print("Unknown command. Use w/s/a/d/q/e, space=stop, x=exit")
    
    except KeyboardInterrupt:
        pass
    
    controller.stop()
    control_thread.join(timeout=1.0)
    print("[DONE]")