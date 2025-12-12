"""
Mini Pupper 2 RL Controller - v2 Fixes
======================================
Fixes from testing:
1. base_lin_vel Z was stuck at +0.5 (bug in estimation)
2. Actions too large/oscillating
3. Robot turning instead of walking forward

Changes:
- Fixed base_lin_vel estimation (Z should be near 0)
- Option to use simpler velocity estimate
- Reduced action scale
- Added action rate limiting
"""

import numpy as np
import torch
import time
import threading
from collections import deque
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration


class FixedMappingControllerV2:
    """
    Controller with corrected frame transformations - Version 2
    """
    
    def __init__(self, policy_path="/home/ubuntu/mp2_mlp/policy_joyboy.pt"):
        print("=" * 70)
        print("FIXED MAPPING RL CONTROLLER v2")
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
            print(f"[OK] Policy: {policy_path}")
        except Exception as e:
            print(f"[ERROR] {e}")
            raise
        
        # ==================== DEFAULT POSITIONS ====================
        self.real_default_positions = np.array([
            0.0,  0.785, -0.785,   # LF
            0.0,  0.785, -0.785,   # RF
            0.0,  0.785, -0.785,   # LB
            0.0,  0.785, -0.785,   # RB
        ])
        
        self.sim_default_positions = np.array([
            0.0,  0.785, -1.57,   # LF
            0.0,  0.785, -1.57,   # RF
            0.0,  0.785, -1.57,   # LB
            0.0,  0.785, -1.57,   # RB
        ])
        
        # ==================== JOINT DIRECTION MAPPING ====================
        self.joint_direction = np.array([
            -1.0, -1.0, -1.0,   # LF: all FLIP
            -1.0, +1.0, +1.0,   # RF: hip FLIP
            +1.0, -1.0, -1.0,   # LB: thigh/calf FLIP
            +1.0, +1.0, +1.0,   # RB: none
        ])
        
        # ==================== EFFORT DIRECTION MAPPING ====================
        self.effort_direction = np.array([
            +1.0, -1.0, -1.0,   # LF
            -1.0, +1.0, +1.0,   # RF
            -1.0, +1.0, -1.0,   # LB
            +1.0, +1.0, -1.0,   # RB
        ])
        
        # ==================== JOINT LIMITS (sim frame) ====================
        self.joint_lower_limits = np.array([-0.5, 0.0, -2.5] * 4)
        self.joint_upper_limits = np.array([0.5, 1.5, -0.5] * 4)
        
        # Hardware mapping
        self.hw_to_isaac_leg = {0: 1, 1: 0, 2: 3, 3: 2}
        
        # ==================== TUNING (ADJUSTED) ====================
        self.ACTION_SCALE = 0.30          # Reduced from 0.35
        self.action_smoothing = 0.5       # Increased smoothing
        self.action_rate_limit = 0.15     # Max change per step (NEW)
        
        # Velocity smoothing
        self.vel_filter_alpha = 0.2       # More smoothing
        self.vel_histories = [deque(maxlen=5) for _ in range(12)]  # Longer history
        
        # ==================== BASE LINEAR VELOCITY ====================
        # Option 1: Simple command-based estimate (more stable)
        # Option 2: IMU integration (more dynamic but noisier)
        self.use_simple_lin_vel = True    # Use simpler estimate
        
        self.estimated_lin_vel = np.zeros(3)
        self.lin_vel_decay = 0.9
        self.lin_vel_gain = 0.01          # Reduced gain
        
        # ==================== IMU ====================
        self.gyro_scale = np.pi / 180.0
        self.accel_scale = 9.81
        self.gyro_offset = np.zeros(3)
        self.accel_offset = np.zeros(3)
        self.gravity_ref = np.array([0.0, 0.0, -1.0])
        
        # ==================== EFFORT ====================
        self.effort_scale = 5000.0
        self.effort_offset = np.zeros(12)
        
        # ==================== STATE ====================
        self.prev_actions = np.zeros(12)
        self.prev_smoothed_actions = np.zeros(12)
        self.prev_joint_angles_sim = self.sim_default_positions.copy()
        self.prev_joint_vel = np.zeros(12)
        self.prev_time = time.time()
        
        self.velocity_command = np.zeros(3)
        self.control_active = False
        self.shutdown = False
        self.startup_steps = 0
        self.startup_duration = 75        # Longer fade-in
        self.debug_counter = 0
        
        self.CONTROL_FREQUENCY = 50
        
        # ==================== CALIBRATION ====================
        print("\n[CALIBRATING] Sensors...")
        self._calibrate_sensors()
        self._print_config()
        print("[READY]")
        print("=" * 70)
    
    def _print_config(self):
        """Print current configuration."""
        print(f"\n[CONFIG]")
        print(f"  ACTION_SCALE:      {self.ACTION_SCALE}")
        print(f"  action_smoothing:  {self.action_smoothing}")
        print(f"  action_rate_limit: {self.action_rate_limit}")
        print(f"  use_simple_lin_vel: {self.use_simple_lin_vel}")
        print(f"  startup_duration:  {self.startup_duration} steps")
    
    def _calibrate_sensors(self, samples=50):
        """Calibrate IMU and effort offsets."""
        gyro_samples, accel_samples, effort_samples = [], [], []
        
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
            self.accel_offset = np.mean(accel_samples, axis=0)
            # Don't subtract gravity here - handle it properly later
        if effort_samples:
            self.effort_offset = np.mean(effort_samples, axis=0)
    
    # ==============================================================
    # POSITION TRANSFORMATIONS
    # ==============================================================
    
    def _read_joint_positions_raw(self):
        """Read raw positions from hardware in Isaac order."""
        raw = self.esp32.servos_get_position()
        if raw is None:
            return None
        
        angles_hw = np.zeros((3, 4))
        for leg in range(4):
            for axis in range(3):
                servo_id = self.pwm_params.servo_ids[axis, leg]
                pos = raw[servo_id - 1]
                dev = (self.servo_params.neutral_position - pos) / self.servo_params.micros_per_rad
                angle = dev / self.servo_params.servo_multipliers[axis, leg] + \
                        self.servo_params.neutral_angles[axis, leg]
                angles_hw[axis, leg] = angle
        
        angles_isaac = np.zeros(12)
        for hw_col, isaac_leg in self.hw_to_isaac_leg.items():
            for axis in range(3):
                angles_isaac[isaac_leg * 3 + axis] = angles_hw[axis, hw_col]
        
        return angles_isaac
    
    def _real_to_sim_positions(self, angles_real):
        """Transform real positions to sim frame using deviation method."""
        deviation_real = angles_real - self.real_default_positions
        deviation_sim = deviation_real * self.joint_direction
        angles_sim = self.sim_default_positions + deviation_sim
        return angles_sim
    
    def _sim_to_real_positions(self, angles_sim):
        """Transform sim positions to real frame."""
        deviation_sim = angles_sim - self.sim_default_positions
        deviation_real = deviation_sim * self.joint_direction
        angles_real = self.real_default_positions + deviation_real
        return angles_real
    
    def _read_joint_positions_sim_frame(self):
        """Read positions and convert to sim frame."""
        angles_real = self._read_joint_positions_raw()
        if angles_real is None:
            return self.prev_joint_angles_sim.copy()
        return self._real_to_sim_positions(angles_real)
    
    # ==============================================================
    # EFFORT
    # ==============================================================
    
    def _read_joint_efforts(self):
        """Read efforts with effort direction correction."""
        raw = self.esp32.servos_get_load()
        if raw is None:
            return np.zeros(12)
        
        raw = np.array(raw, dtype=float)
        
        effort_isaac = np.zeros(12)
        offset_isaac = np.zeros(12)
        for hw_col, isaac_leg in self.hw_to_isaac_leg.items():
            for axis in range(3):
                servo_id = self.pwm_params.servo_ids[axis, hw_col]
                effort_isaac[isaac_leg * 3 + axis] = raw[servo_id - 1]
                offset_isaac[isaac_leg * 3 + axis] = self.effort_offset[servo_id - 1]
        
        centered = effort_isaac - offset_isaac
        normalized = centered / self.effort_scale
        normalized = normalized * self.effort_direction
        
        return normalized
    
    # ==============================================================
    # VELOCITY
    # ==============================================================
    
    def _compute_smoothed_velocity(self, current_angles, dt):
        """Compute smoothed joint velocities."""
        if dt < 0.001:
            return self.prev_joint_vel.copy()
        
        raw_vel = (current_angles - self.prev_joint_angles_sim) / dt
        raw_vel = np.clip(raw_vel, -10.0, 10.0)  # Tighter clipping
        
        smoothed_vel = np.zeros(12)
        for i in range(12):
            self.vel_histories[i].append(raw_vel[i])
            if len(self.vel_histories[i]) >= 3:
                # Use median for robustness
                smoothed_vel[i] = np.median(list(self.vel_histories[i]))
            else:
                smoothed_vel[i] = raw_vel[i]
        
        # Low-pass filter
        filtered_vel = (self.vel_filter_alpha * smoothed_vel + 
                       (1 - self.vel_filter_alpha) * self.prev_joint_vel)
        
        return filtered_vel
    
    def _estimate_base_lin_vel(self, accel_body, dt):
        """
        Estimate base linear velocity.
        
        FIXED: Z component should be near 0 for a walking robot!
        """
        if self.use_simple_lin_vel:
            # Simple approach: assume robot achieves ~70% of commanded velocity
            # Z velocity should be ~0 for ground locomotion
            lin_vel = np.array([
                self.velocity_command[0] * 0.7,  # Forward
                self.velocity_command[1] * 0.7,  # Lateral
                0.0  # Vertical - should be ~0!
            ])
            return lin_vel
        
        else:
            # IMU integration approach (experimental)
            # Remove gravity from acceleration
            # Gravity in body frame when level: [0, 0, -9.81]
            accel_corrected = accel_body.copy()
            accel_corrected[2] += 9.81  # Remove gravity
            
            # Only integrate X and Y, keep Z at 0
            self.estimated_lin_vel[0] += accel_corrected[0] * self.lin_vel_gain * dt
            self.estimated_lin_vel[1] += accel_corrected[1] * self.lin_vel_gain * dt
            self.estimated_lin_vel[2] = 0.0  # Force Z to 0
            
            # Decay
            self.estimated_lin_vel *= self.lin_vel_decay
            
            # Blend with command
            cmd_estimate = np.array([
                self.velocity_command[0] * 0.7,
                self.velocity_command[1] * 0.7,
                0.0
            ])
            blended = 0.5 * self.estimated_lin_vel + 0.5 * cmd_estimate
            
            return np.clip(blended, -0.5, 0.5)
    
    # ==============================================================
    # HARDWARE OUTPUT
    # ==============================================================
    
    def _isaac_to_hardware_matrix(self, flat_angles_sim):
        """Convert sim frame angles to hardware matrix."""
        angles_real = self._sim_to_real_positions(flat_angles_sim)
        
        matrix = np.zeros((3, 4))
        matrix[:, 1] = angles_real[0:3]   # LF
        matrix[:, 0] = angles_real[3:6]   # RF
        matrix[:, 3] = angles_real[6:9]   # LB
        matrix[:, 2] = angles_real[9:12]  # RB
        
        return matrix
    
    # ==============================================================
    # OBSERVATION
    # ==============================================================
    
    def get_observation(self):
        """Build 60-dim observation in sim frame."""
        current_time = time.time()
        dt = current_time - self.prev_time
        
        imu = self.esp32.imu_get_data()
        current_angles_sim = self._read_joint_positions_sim_frame()
        joint_effort = self._read_joint_efforts()
        
        # Base linear velocity (FIXED)
        accel_raw = np.array([imu['ax'], imu['ay'], imu['az']])
        accel = accel_raw * self.accel_scale
        base_lin_vel = self._estimate_base_lin_vel(accel, dt)
        
        # Base angular velocity
        gyro_raw = np.array([imu['gx'], imu['gy'], imu['gz']])
        base_ang_vel = (gyro_raw - self.gyro_offset) * self.gyro_scale
        
        # Projected gravity (normalized acceleration)
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0.1:
            projected_gravity = accel / accel_norm
        else:
            projected_gravity = np.array([0.0, 0.0, -1.0])
        
        velocity_commands = self.velocity_command.copy()
        
        joint_pos_rel = current_angles_sim - self.sim_default_positions
        joint_vel = self._compute_smoothed_velocity(current_angles_sim, dt)
        
        # Update state
        self.prev_joint_angles_sim = current_angles_sim.copy()
        self.prev_joint_vel = joint_vel.copy()
        self.prev_time = current_time
        
        prev_actions = self.prev_actions.copy()
        
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
    
    # ==============================================================
    # CONTROL
    # ==============================================================
    
    def control_step(self):
        """Execute one control step with rate limiting."""
        obs = self.get_observation()
        
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            raw_actions = self.policy(obs_tensor).squeeze().numpy()
        
        # Clip actions
        clipped_actions = np.clip(raw_actions, -1.0, 1.0)
        
        # Startup fade-in
        if self.startup_steps < self.startup_duration:
            fade = self.startup_steps / self.startup_duration
            self.startup_steps += 1
        else:
            fade = 1.0
        
        faded_actions = clipped_actions * fade
        
        # Action smoothing (exponential moving average)
        smoothed_actions = (self.action_smoothing * faded_actions + 
                          (1 - self.action_smoothing) * self.prev_smoothed_actions)
        
        # Rate limiting: limit how fast actions can change
        action_delta = smoothed_actions - self.prev_smoothed_actions
        action_delta = np.clip(action_delta, -self.action_rate_limit, self.action_rate_limit)
        smoothed_actions = self.prev_smoothed_actions + action_delta
        
        # Store for next iteration
        self.prev_actions = faded_actions.copy()  # Raw faded actions for observation
        self.prev_smoothed_actions = smoothed_actions.copy()
        
        # Compute target (sim frame)
        target_sim = self.sim_default_positions + smoothed_actions * self.ACTION_SCALE
        target_sim = np.clip(target_sim, self.joint_lower_limits, self.joint_upper_limits)
        
        # Send to hardware
        target_matrix = self._isaac_to_hardware_matrix(target_sim)
        self.hardware.set_actuator_postions(target_matrix)
        
        self.debug_counter += 1
        if self.debug_counter % 50 == 0:
            pos_rel = obs[12:24]
            lin_vel = obs[0:3]
            print(f"Step {self.debug_counter}: "
                  f"pos=[{pos_rel.min():+.2f},{pos_rel.max():+.2f}] "
                  f"act=[{smoothed_actions.min():+.2f},{smoothed_actions.max():+.2f}] "
                  f"vel=[{lin_vel[0]:+.2f},{lin_vel[1]:+.2f},{lin_vel[2]:+.2f}]")
    
    def control_loop(self):
        """Main control loop."""
        dt_target = 1.0 / self.CONTROL_FREQUENCY
        print(f"\n[RUNNING] {self.CONTROL_FREQUENCY}Hz")
        
        while not self.shutdown:
            loop_start = time.time()
            
            if self.control_active:
                try:
                    self.control_step()
                except Exception as e:
                    print(f"[ERROR] {e}")
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
                self.startup_steps = 0
                self.estimated_lin_vel = np.zeros(3)
                self.prev_smoothed_actions = np.zeros(12)
            self.control_active = True
            print(f"[CMD] vx={vx:.2f}, vy={vy:.2f}, vyaw={vyaw:.2f}")
        else:
            self.control_active = False
            print("[STOP]")
    
    def stop(self):
        """Shutdown."""
        self.control_active = False
        self.shutdown = True
    
    def print_debug_info(self):
        """Print detailed debug information."""
        angles_real = self._read_joint_positions_raw()
        angles_sim = self._read_joint_positions_sim_frame()
        effort = self._read_joint_efforts()
        obs = self.get_observation()
        
        print("\n" + "="*70)
        print("DEBUG INFO")
        print("="*70)
        
        names = ['LF_hip', 'LF_thigh', 'LF_calf',
                 'RF_hip', 'RF_thigh', 'RF_calf',
                 'LB_hip', 'LB_thigh', 'LB_calf',
                 'RB_hip', 'RB_thigh', 'RB_calf']
        
        print("\n[JOINT POSITIONS]")
        print(f"{'Joint':<12} {'Real':>8} {'Sim':>8} {'Rel':>8} {'JntDir':>8}")
        print("-" * 48)
        for i, name in enumerate(names):
            real = angles_real[i] if angles_real is not None else 0
            sim = angles_sim[i]
            rel = sim - self.sim_default_positions[i]
            jd = "FLIP" if self.joint_direction[i] < 0 else "same"
            print(f"{name:<12} {real:>+8.3f} {sim:>+8.3f} {rel:>+8.3f} {jd:>8}")
        
        print("\n[OBSERVATION SUMMARY]")
        print(f"  base_lin_vel:    [{obs[0]:+.3f}, {obs[1]:+.3f}, {obs[2]:+.3f}]")
        print(f"  base_ang_vel:    [{obs[3]:+.3f}, {obs[4]:+.3f}, {obs[5]:+.3f}]")
        print(f"  gravity:         [{obs[6]:+.3f}, {obs[7]:+.3f}, {obs[8]:+.3f}]")
        print(f"  commands:        [{obs[9]:+.3f}, {obs[10]:+.3f}, {obs[11]:+.3f}]")
        print(f"  joint_pos_rel:   range [{obs[12:24].min():+.3f}, {obs[12:24].max():+.3f}]")
        print(f"  joint_vel:       range [{obs[24:36].min():+.3f}, {obs[24:36].max():+.3f}]")
        print(f"  joint_effort:    range [{obs[36:48].min():+.3f}, {obs[36:48].max():+.3f}]")
        print(f"  prev_actions:    range [{obs[48:60].min():+.3f}, {obs[48:60].max():+.3f}]")
        print("="*70)
    
    def set_param(self, param, value):
        """Adjust parameters on the fly."""
        if param == 'scale':
            self.ACTION_SCALE = float(value)
            print(f"[PARAM] ACTION_SCALE = {self.ACTION_SCALE}")
        elif param == 'smooth':
            self.action_smoothing = float(value)
            print(f"[PARAM] action_smoothing = {self.action_smoothing}")
        elif param == 'rate':
            self.action_rate_limit = float(value)
            print(f"[PARAM] action_rate_limit = {self.action_rate_limit}")
        elif param == 'simple':
            self.use_simple_lin_vel = value.lower() in ['true', '1', 'yes']
            print(f"[PARAM] use_simple_lin_vel = {self.use_simple_lin_vel}")
        else:
            print(f"Unknown param: {param}")
            print("Available: scale, smooth, rate, simple")


# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("MINI PUPPER 2 - FIXED MAPPING CONTROLLER v2")
    print("="*70)
    print("""
Controls:
  w/s      - Forward/Backward
  a/d      - Strafe Left/Right  
  q/e      - Turn Left/Right
  SPACE    - Stop
  +/-      - Adjust speed
  debug    - Print debug info
  set X Y  - Set parameter (scale/smooth/rate) to value Y
  x        - Exit
""")
    
    controller = FixedMappingControllerV2("/home/ubuntu/mp2_mlp/policy_joyboy.pt")
    
    control_thread = threading.Thread(target=controller.control_loop, daemon=True)
    control_thread.start()
    
    speed = 0.15
    
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
                print(f"[SPEED] {speed:.2f}")
            elif cmd == '-':
                speed = max(speed - 0.05, 0.05)
                print(f"[SPEED] {speed:.2f}")
            elif cmd == 'debug':
                controller.print_debug_info()
            elif cmd.startswith('set '):
                parts = cmd.split()
                if len(parts) == 3:
                    controller.set_param(parts[1], parts[2])
                else:
                    print("Usage: set <param> <value>")
            elif cmd == 'x':
                break
            else:
                print("Unknown command")
    
    except KeyboardInterrupt:
        pass
    
    controller.stop()
    control_thread.join(timeout=1.0)
    print("[DONE]")