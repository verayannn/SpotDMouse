"""
NEW APPROACH:
  1. Compute deviation from REAL default
  2. Flip the deviation if needed
  3. Add back to SIM default
"""

import numpy as np
import torch
import time
import threading
from collections import deque
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration


class FixedMappingController:
    """
    Controller with corrected frame transformations.
    
    Key insight: Joint direction flip applies to DEVIATIONS from neutral,
    not to absolute positions!
    """
    
    def __init__(self, policy_path="/home/ubuntu/mp2_mlp/policy_joyboy.pt"):
        print("=" * 70)
        print("FIXED MAPPING RL CONTROLLER")
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
        # Real robot default positions (from servo calibration)
        self.real_default_positions = np.array([
            0.0,  0.785, -0.785,   # LF: hip=0, thigh=45°, calf=-45°
            0.0,  0.785, -0.785,   # RF
            0.0,  0.785, -0.785,   # LB
            0.0,  0.785, -0.785,   # RB
        ])
        
        # Simulation default positions
        self.sim_default_positions = np.array([
            0.0,  0.785, -1.57,   # LF
            0.0,  0.785, -1.57,   # RF
            0.0,  0.785, -1.57,   # LB
            0.0,  0.785, -1.57,   # RB
        ])
        
        # ==============================================================
        # JOINT DIRECTION MAPPING
        # ==============================================================
        # This flips the DEVIATION from neutral, not the absolute position
        # +1 = same direction, -1 = opposite direction
        self.joint_direction = np.array([
            # LF
            -1.0,  # hip:   sim INWARD+ / real OUTWARD+  → FLIP
            -1.0,  # thigh: sim BACK+   / real FORWARD+  → FLIP
            -1.0,  # calf:  sim EXTEND+ / real FLEX+     → FLIP
            
            # RF
            -1.0,  # hip:   sim OUT+    / real IN+       → FLIP
            +1.0,  # thigh: sim BACK+   / real BACK+     → SAME
            +1.0,  # calf:  sim EXTEND+ / real EXTEND+   → SAME
            
            # LB
            +1.0,  # hip:   sim IN+     / real IN+       → SAME
            -1.0,  # thigh: sim BACK+   / real FORWARD+  → FLIP
            -1.0,  # calf:  sim EXTEND+ / real FLEX+     → FLIP
            
            # RB
            +1.0,  # hip:   sim OUT+    / real OUT+      → SAME
            +1.0,  # thigh: sim BACK+   / real BACK+     → SAME
            +1.0,  # calf:  sim EXTEND+ / real EXTEND+   → SAME
        ])
        
        # ==============================================================
        # EFFORT DIRECTION MAPPING (separate from joint direction)
        # ==============================================================
        self.effort_direction = np.array([
            +1.0, -1.0, -1.0,   # LF
            -1.0, +1.0, +1.0,   # RF
            -1.0, +1.0, -1.0,   # LB
            +1.0, +1.0, -1.0,   # RB
        ])
        
        self._print_mappings()
        
        # ==================== JOINT LIMITS (sim frame) ====================
        self.joint_lower_limits = np.array([-0.5, 0.0, -2.5] * 4)
        self.joint_upper_limits = np.array([0.5, 1.5, -0.5] * 4)
        
        # Hardware mapping
        self.hw_to_isaac_leg = {0: 1, 1: 0, 2: 3, 3: 2}
        
        # ==================== TUNING ====================
        self.ACTION_SCALE = 0.35
        self.action_smoothing = 0.4
        self.vel_filter_alpha = 0.3
        self.vel_histories = [deque(maxlen=3) for _ in range(12)]
        
        # ==================== VELOCITY ESTIMATION ====================
        self.estimated_lin_vel = np.zeros(3)
        self.lin_vel_decay = 0.95
        self.lin_vel_gain = 0.02
        
        # ==================== IMU ====================
        self.gyro_scale = np.pi / 180.0
        self.accel_scale = 9.81
        self.gyro_offset = np.zeros(3)
        self.accel_offset = np.zeros(3)
        
        # ==================== EFFORT ====================
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
        print("\n[CALIBRATING] Sensors...")
        self._calibrate_sensors()
        print("[READY]")
        print("=" * 70)
    
    def _print_mappings(self):
        """Print mappings."""
        names = ['LF_hip', 'LF_thigh', 'LF_calf',
                 'RF_hip', 'RF_thigh', 'RF_calf',
                 'LB_hip', 'LB_thigh', 'LB_calf',
                 'RB_hip', 'RB_thigh', 'RB_calf']
        
        print("\n[MAPPINGS]")
        print(f"{'Joint':<12} {'RealDef':>8} {'SimDef':>8} {'JntDir':>8} {'EffDir':>8}")
        print("-" * 48)
        for i, name in enumerate(names):
            rd = self.real_default_positions[i]
            sd = self.sim_default_positions[i]
            jd = "FLIP" if self.joint_direction[i] < 0 else "same"
            ed = "FLIP" if self.effort_direction[i] < 0 else "same"
            print(f"{name:<12} {rd:>+8.3f} {sd:>+8.3f} {jd:>8} {ed:>8}")
    
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
            self.accel_offset[2] -= 1.0
        if effort_samples:
            self.effort_offset = np.mean(effort_samples, axis=0)
    
    # ==============================================================
    # POSITION TRANSFORMATIONS - FIXED
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
        
        # Reorder to Isaac [LF, RF, LB, RB]
        angles_isaac = np.zeros(12)
        for hw_col, isaac_leg in self.hw_to_isaac_leg.items():
            for axis in range(3):
                angles_isaac[isaac_leg * 3 + axis] = angles_hw[axis, hw_col]
        
        return angles_isaac
    
    def _real_to_sim_positions(self, angles_real):
        """
        Transform real robot positions to simulation frame.
        
        Method: Transform DEVIATIONS, not absolute positions
        1. Compute deviation from real default
        2. Flip deviation if joint direction is -1
        3. Add to sim default
        
        sim_pos = sim_default + joint_direction * (real_pos - real_default)
        """
        # Deviation from real default
        deviation_real = angles_real - self.real_default_positions
        
        # Flip deviation where needed
        deviation_sim = deviation_real * self.joint_direction
        
        # Add to sim default
        angles_sim = self.sim_default_positions + deviation_sim
        
        return angles_sim
    
    def _sim_to_real_positions(self, angles_sim):
        """
        Transform simulation frame positions to real robot frame.
        
        Method: Inverse of _real_to_sim_positions
        1. Compute deviation from sim default
        2. Flip deviation if joint direction is -1 (same operation, since ±1)
        3. Add to real default
        
        real_pos = real_default + joint_direction * (sim_pos - sim_default)
        """
        # Deviation from sim default
        deviation_sim = angles_sim - self.sim_default_positions
        
        # Flip deviation where needed (inverse of ±1 is same value)
        deviation_real = deviation_sim * self.joint_direction
        
        # Add to real default
        angles_real = self.real_default_positions + deviation_real
        
        return angles_real
    
    def _read_joint_positions_sim_frame(self):
        """Read positions and convert to sim frame."""
        angles_real = self._read_joint_positions_raw()
        if angles_real is None:
            return self.prev_joint_angles_sim.copy()
        
        return self._real_to_sim_positions(angles_real)
    
    # ==============================================================
    # EFFORT TRANSFORMATIONS
    # ==============================================================
    
    def _read_joint_efforts(self):
        """Read efforts with effort direction correction."""
        raw = self.esp32.servos_get_load()
        if raw is None:
            return np.zeros(12)
        
        raw = np.array(raw, dtype=float)
        
        # Reorder to Isaac format
        effort_isaac = np.zeros(12)
        offset_isaac = np.zeros(12)
        for hw_col, isaac_leg in self.hw_to_isaac_leg.items():
            for axis in range(3):
                servo_id = self.pwm_params.servo_ids[axis, hw_col]
                effort_isaac[isaac_leg * 3 + axis] = raw[servo_id - 1]
                offset_isaac[isaac_leg * 3 + axis] = self.effort_offset[servo_id - 1]
        
        centered = effort_isaac - offset_isaac
        normalized = centered / self.effort_scale
        
        # Apply EFFORT direction (separate from joint direction)
        normalized = normalized * self.effort_direction
        
        return normalized
    
    # ==============================================================
    # VELOCITY & IMU
    # ==============================================================
    
    def _compute_smoothed_velocity(self, current_angles, dt):
        """Compute smoothed joint velocities."""
        if dt < 0.001:
            return self.prev_joint_vel.copy()
        
        raw_vel = (current_angles - self.prev_joint_angles_sim) / dt
        raw_vel = np.clip(raw_vel, -15.0, 15.0)
        
        smoothed_vel = np.zeros(12)
        for i in range(12):
            self.vel_histories[i].append(raw_vel[i])
            if len(self.vel_histories[i]) > 0:
                smoothed_vel[i] = np.median(list(self.vel_histories[i]))
        
        filtered_vel = (self.vel_filter_alpha * smoothed_vel + 
                       (1 - self.vel_filter_alpha) * self.prev_joint_vel)
        
        return filtered_vel
    
    def _estimate_base_lin_vel(self, accel, dt):
        """Estimate linear velocity from IMU."""
        accel_corrected = accel.copy()
        accel_corrected[2] += 9.81
        
        self.estimated_lin_vel += accel_corrected * self.lin_vel_gain * dt
        self.estimated_lin_vel *= self.lin_vel_decay
        
        cmd_estimate = self.velocity_command * 0.7
        blended = 0.7 * self.estimated_lin_vel + 0.3 * cmd_estimate
        
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
    # OBSERVATION & CONTROL
    # ==============================================================
    
    def get_observation(self):
        """Build 60-dim observation in sim frame."""
        current_time = time.time()
        dt = current_time - self.prev_time
        
        imu = self.esp32.imu_get_data()
        current_angles_sim = self._read_joint_positions_sim_frame()
        joint_effort = self._read_joint_efforts()
        
        # Base linear velocity
        accel_raw = np.array([imu['ax'], imu['ay'], imu['az']])
        accel = (accel_raw - self.accel_offset) * self.accel_scale
        base_lin_vel = self._estimate_base_lin_vel(accel, dt)
        
        # Base angular velocity
        gyro_raw = np.array([imu['gx'], imu['gy'], imu['gz']])
        base_ang_vel = (gyro_raw - self.gyro_offset) * self.gyro_scale
        
        # Projected gravity
        accel_for_gravity = accel_raw * self.accel_scale
        accel_norm = np.linalg.norm(accel_for_gravity)
        if accel_norm > 0.1:
            projected_gravity = accel_for_gravity / accel_norm
        else:
            projected_gravity = np.array([0.0, 0.0, -1.0])
        
        velocity_commands = self.velocity_command.copy()
        
        # Joint positions relative to sim default
        joint_pos_rel = current_angles_sim - self.sim_default_positions
        
        # Joint velocities
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
    
    def control_step(self):
        """Execute one control step."""
        obs = self.get_observation()
        
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            raw_actions = self.policy(obs_tensor).squeeze().numpy()
        
        clipped_actions = np.clip(raw_actions, -1.0, 1.0)
        
        if self.startup_steps < self.startup_duration:
            fade = self.startup_steps / self.startup_duration
            self.startup_steps += 1
        else:
            fade = 1.0
        
        faded_actions = clipped_actions * fade
        
        smoothed_actions = (self.action_smoothing * faded_actions + 
                          (1 - self.action_smoothing) * self.prev_actions)
        self.prev_actions = smoothed_actions.copy()
        
        target_sim = self.sim_default_positions + smoothed_actions * self.ACTION_SCALE
        target_sim = np.clip(target_sim, self.joint_lower_limits, self.joint_upper_limits)
        
        target_matrix = self._isaac_to_hardware_matrix(target_sim)
        self.hardware.set_actuator_postions(target_matrix)
        
        self.debug_counter += 1
        if self.debug_counter % 50 == 0:
            pos_rel = obs[12:24]
            print(f"Step {self.debug_counter}: "
                  f"pos_rel=[{pos_rel.min():+.2f},{pos_rel.max():+.2f}] "
                  f"act=[{smoothed_actions.min():+.2f},{smoothed_actions.max():+.2f}]")
    
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
        print(f"{'Joint':<12} {'Real':>8} {'RealDef':>8} {'Sim':>8} {'SimDef':>8} {'Rel':>8}")
        print("-" * 60)
        for i, name in enumerate(names):
            real = angles_real[i] if angles_real is not None else 0
            real_def = self.real_default_positions[i]
            sim = angles_sim[i]
            sim_def = self.sim_default_positions[i]
            rel = sim - sim_def
            print(f"{name:<12} {real:>+8.3f} {real_def:>+8.3f} {sim:>+8.3f} {sim_def:>+8.3f} {rel:>+8.3f}")
        
        print("\n[JOINT EFFORTS]")
        print(f"{'Joint':<12} {'Effort':>10} {'EffDir':>8}")
        print("-" * 32)
        for i, name in enumerate(names):
            ed = "FLIP" if self.effort_direction[i] < 0 else "same"
            print(f"{name:<12} {effort[i]:>+10.4f} {ed:>8}")
        
        print("\n[OBSERVATION SUMMARY]")
        print(f"  base_lin_vel:    [{obs[0]:+.3f}, {obs[1]:+.3f}, {obs[2]:+.3f}]")
        print(f"  base_ang_vel:    [{obs[3]:+.3f}, {obs[4]:+.3f}, {obs[5]:+.3f}]")
        print(f"  gravity:         [{obs[6]:+.3f}, {obs[7]:+.3f}, {obs[8]:+.3f}]")
        print(f"  commands:        [{obs[9]:+.3f}, {obs[10]:+.3f}, {obs[11]:+.3f}]")
        print(f"  joint_pos_rel:   range [{obs[12:24].min():+.3f}, {obs[12:24].max():+.3f}]")
        print(f"  joint_vel:       range [{obs[24:36].min():+.3f}, {obs[24:36].max():+.3f}]")
        print(f"  joint_effort:    range [{obs[36:48].min():+.3f}, {obs[36:48].max():+.3f}]")
        print(f"  prev_actions:    range [{obs[48:60].min():+.3f}, {obs[48:60].max():+.3f}]")
        
        # Show what the transformation is doing
        print("\n[TRANSFORMATION CHECK]")
        if angles_real is not None:
            dev_real = angles_real - self.real_default_positions
            dev_sim = dev_real * self.joint_direction
            print(f"  Real deviation from real_default: [{dev_real.min():+.3f}, {dev_real.max():+.3f}]")
            print(f"  After direction flip:             [{dev_sim.min():+.3f}, {dev_sim.max():+.3f}]")
            print(f"  Sim position (sim_def + dev_sim): should match 'Sim' column above")
        
        print("="*70)


# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("MINI PUPPER 2 - FIXED MAPPING CONTROLLER")
    print("="*70)
    print("""
Controls:
  w/s    - Forward/Backward
  a/d    - Strafe Left/Right  
  q/e    - Turn Left/Right
  SPACE  - Stop
  +/-    - Adjust speed
  debug  - Print debug info
  x      - Exit
""")
    
    controller = FixedMappingController("/home/ubuntu/mp2_mlp/policy_joyboy.pt")
    
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
            elif cmd == 'x':
                break
            else:
                print("Unknown command")
    
    except KeyboardInterrupt:
        pass
    
    controller.stop()
    control_thread.join(timeout=1.0)
    print("[DONE]")