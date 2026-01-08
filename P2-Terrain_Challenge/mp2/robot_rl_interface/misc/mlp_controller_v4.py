"""
Mini Pupper 2 RL Controller - v4 COORDINATE FRAME FIX
======================================================
CRITICAL FIX in v4:
Based on trajectory tests, there is a coordinate frame mismatch between
simulation and hardware. This version adds proper frame transformations.

Test Results Analysis:
- Sim X+ (forward) → Real Yaw+ (rotate CW)
- Sim X- (backward) → Real Y+ (strafe left)
- Sim Y+ (strafe left) → Real Y- (strafe right) [FLIPPED]
- Sim Y- (strafe right) → Real Yaw- (rotate CCW)
- Sim Z+ (turn left) → Real mostly still
- Sim Z- (turn right) → Real Yaw- (turn CCW) [FLIPPED]

Pattern: 90-degree rotation + axis flips

Hypothesis for correction:
Hardware frame appears to be rotated relative to simulation frame.
We need to remap velocity commands AND IMU readings.

Changes from v3:
- Added velocity command frame transformation
- Added IMU frame transformation (gyro + accel)
- Experimental remapping options (can be toggled)
"""

import numpy as np
import torch
import time
import threading
from collections import deque
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration


class FixedMappingControllerV4:
    """
    Controller with coordinate frame corrections - Version 4
    """

    def __init__(self, policy_path="/home/ubuntu/mp2_mlp/policy_joyboy.pt"):
        print("=" * 70)
        print("FIXED MAPPING RL CONTROLLER v4 - COORDINATE FRAME FIX")
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

        # ==================== COORDINATE FRAME REMAPPING ====================
        # Based on trajectory test results, we need to remap coordinates

        # Option 1: Y-flip + 90-degree rotation
        self.cmd_remap_mode = 1

        # Option 2: Alternative remapping (can be changed via set_param)
        # self.cmd_remap_mode = 2

        # IMU frame remapping (try different options)
        self.imu_remap_mode = 1

        # ==================== TUNING ====================
        self.ACTION_SCALE = 0.5          # Back to sim scale now that we fix coords
        self.ema_alpha = 0.6#0.3
        self.action_rate_limit = 0.05#0.05

        # Velocity smoothing
        self.vel_filter_alpha = 0.2
        self.vel_histories = [deque(maxlen=5) for _ in range(12)]

        # ==================== BASE LINEAR VELOCITY ====================
        self.use_simple_lin_vel = True

        self.estimated_lin_vel = np.zeros(3)
        self.lin_vel_decay = 0.9
        self.lin_vel_gain = 0.01

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

        self.velocity_command_sim = np.zeros(3)  # In sim frame
        self.velocity_command_hw = np.zeros(3)   # In hardware frame
        self.control_active = False
        self.shutdown = False
        self.startup_steps = 0
        self.startup_duration = 75
        self.debug_counter = 0

        self.CONTROL_FREQUENCY = 50

        # ==================== OBSERVATION HEALTH MONITORING ====================
        self.health_check_enabled = True

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
        print(f"  ema_alpha:         {self.ema_alpha}")
        print(f"  action_rate_limit: {self.action_rate_limit}")
        print(f"  cmd_remap_mode:    {self.cmd_remap_mode}")
        print(f"  imu_remap_mode:    {self.imu_remap_mode}")
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
        if effort_samples:
            self.effort_offset = np.mean(effort_samples, axis=0)

    # ==============================================================
    # COORDINATE FRAME TRANSFORMATIONS
    # ==============================================================

    def _sim_velocity_to_hw_velocity(self, vx_sim, vy_sim, vyaw_sim):
        """
        Transform velocity command from simulation frame to hardware frame.

        Based on trajectory tests:
        - Sim X+ (forward) → HW causes clockwise rotation
        - Sim Y+ (left) → HW causes right strafe (flipped)
        - Sim Yaw seems mixed up

        Trying different remapping strategies.
        """
        if self.cmd_remap_mode == 1:
            # Hypothesis 1: Y-axis flip + 90-degree rotation
            # Sim X → HW -Y (forward maps to right, rotated CW)
            # Sim Y → HW -X (left maps to backward)
            # Sim Yaw → HW -Yaw (flip yaw)
            vx_hw = -vy_sim
            vy_hw = -vx_sim
            vyaw_hw = -vyaw_sim

        elif self.cmd_remap_mode == 2:
            # Hypothesis 2: Just Y flip
            vx_hw = vx_sim
            vy_hw = -vy_sim
            vyaw_hw = vyaw_sim

        elif self.cmd_remap_mode == 3:
            # Hypothesis 3: 90-degree rotation only
            vx_hw = -vy_sim
            vy_hw = vx_sim
            vyaw_hw = vyaw_sim

        else:
            # No remapping
            vx_hw = vx_sim
            vy_hw = vy_sim
            vyaw_hw = vyaw_sim

        return vx_hw, vy_hw, vyaw_hw

    def _hw_imu_to_sim_imu(self, gyro_hw, accel_hw):
        """
        Transform IMU readings from hardware frame to simulation frame.

        Hardware IMU frame may be rotated relative to simulation expectations.
        """
        if self.imu_remap_mode == 1:
            # Match velocity remapping
            gyro_sim = np.array([-gyro_hw[1], -gyro_hw[0], -gyro_hw[2]])
            accel_sim = np.array([-accel_hw[1], -accel_hw[0], accel_hw[2]])

        elif self.imu_remap_mode == 2:
            # Just Y flip
            gyro_sim = np.array([gyro_hw[0], -gyro_hw[1], gyro_hw[2]])
            accel_sim = np.array([accel_hw[0], -accel_hw[1], accel_hw[2]])

        elif self.imu_remap_mode == 3:
            # 90-degree rotation
            gyro_sim = np.array([-gyro_hw[1], gyro_hw[0], gyro_hw[2]])
            accel_sim = np.array([-accel_hw[1], accel_hw[0], accel_hw[2]])

        else:
            # No remapping
            gyro_sim = gyro_hw.copy()
            accel_sim = accel_hw.copy()

        return gyro_sim, accel_sim

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
        raw_vel = np.clip(raw_vel, -10.0, 10.0)

        smoothed_vel = np.zeros(12)
        for i in range(12):
            self.vel_histories[i].append(raw_vel[i])
            if len(self.vel_histories[i]) >= 3:
                smoothed_vel[i] = np.median(list(self.vel_histories[i]))
            else:
                smoothed_vel[i] = raw_vel[i]

        filtered_vel = (self.vel_filter_alpha * smoothed_vel +
                       (1 - self.vel_filter_alpha) * self.prev_joint_vel)

        return filtered_vel

    def _estimate_base_lin_vel(self, accel_body, dt):
        """Estimate base linear velocity (Z should be near 0)."""
        if self.use_simple_lin_vel:
            # Use velocity command in SIM frame (already transformed)
            lin_vel = np.array([
                self.velocity_command_sim[0] * 0.7,
                self.velocity_command_sim[1] * 0.7,
                0.0
            ])
            return lin_vel
        else:
            accel_corrected = accel_body.copy()
            accel_corrected[2] += 9.81

            self.estimated_lin_vel[0] += accel_corrected[0] * self.lin_vel_gain * dt
            self.estimated_lin_vel[1] += accel_corrected[1] * self.lin_vel_gain * dt
            self.estimated_lin_vel[2] = 0.0

            self.estimated_lin_vel *= self.lin_vel_decay

            cmd_estimate = np.array([
                self.velocity_command_sim[0] * 0.7,
                self.velocity_command_sim[1] * 0.7,
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
        """Build 60-dim observation in sim frame with coordinate transforms."""
        current_time = time.time()
        dt = current_time - self.prev_time

        imu = self.esp32.imu_get_data()
        current_angles_sim = self._read_joint_positions_sim_frame()
        joint_effort = self._read_joint_efforts()

        # Read IMU in hardware frame
        gyro_raw_hw = np.array([imu['gx'], imu['gy'], imu['gz']])
        accel_raw_hw = np.array([imu['ax'], imu['ay'], imu['az']])

        # Remove offsets
        gyro_hw = (gyro_raw_hw - self.gyro_offset) * self.gyro_scale
        accel_hw = accel_raw_hw * self.accel_scale

        # Transform to sim frame
        gyro_sim, accel_sim = self._hw_imu_to_sim_imu(gyro_hw, accel_hw)

        base_ang_vel = gyro_sim

        # Base linear velocity (uses sim frame velocity command)
        base_lin_vel = self._estimate_base_lin_vel(accel_sim, dt)

        # Projected gravity from sim-frame acceleration
        accel_norm = np.linalg.norm(accel_sim)
        if accel_norm > 0.1:
            projected_gravity = accel_sim / accel_norm
        else:
            projected_gravity = np.array([0.0, 0.0, -1.0])

        velocity_commands = self.velocity_command_sim.copy()

        joint_pos_rel = current_angles_sim - self.sim_default_positions
        joint_vel = self._compute_smoothed_velocity(current_angles_sim, dt)

        # Update state
        self.prev_joint_angles_sim = current_angles_sim.copy()
        self.prev_joint_vel = joint_vel.copy()
        self.prev_time = current_time

        prev_actions = self.prev_actions.copy()

        # Clamp observations to simulation ranges
        base_lin_vel = np.clip(base_lin_vel, -0.5, 0.5)
        base_ang_vel = np.clip(base_ang_vel, -2.0, 2.0)
        projected_gravity = np.clip(projected_gravity, -1.0, 1.0)
        joint_pos_rel = np.clip(joint_pos_rel, -0.9, 0.9)
        joint_vel = np.clip(joint_vel, -10.0, 10.0)
        joint_effort = np.clip(joint_effort, -1.0, 1.0)

        obs = np.concatenate([
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            velocity_commands,
            joint_pos_rel,
            joint_vel,
            joint_effort,
            prev_actions,
        ]).astype(np.float32)

        return obs

    def check_observation_health(self, obs):
        """Check if observations are in healthy ranges."""
        if not self.health_check_enabled:
            return True

        if self.debug_counter % 50 != 0:
            return True

        base_ang_vel = obs[3:6]
        gravity = obs[6:9]
        joint_vel = obs[24:36]

        warnings = []

        max_ang_vel = np.max(np.abs(base_ang_vel))
        if max_ang_vel > 1.5:
            warnings.append(f"HIGH_ANG_VEL:{max_ang_vel:.2f}")

        if gravity[2] > -0.95:
            warnings.append(f"TILTED:gz={gravity[2]:.2f}")

        max_jvel = np.max(np.abs(joint_vel))
        if max_jvel > 2.0:
            warnings.append(f"HIGH_JVEL:{max_jvel:.2f}")

        if warnings:
            print(f"[HEALTH] Step {self.debug_counter}: {', '.join(warnings)}")
            return False
        return True

    # ==============================================================
    # CONTROL
    # ==============================================================

    def control_step(self):
        """Execute one control step with EMA smoothing and rate limiting."""
        obs = self.get_observation()

        self.check_observation_health(obs)

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

        smoothed_actions = (self.ema_alpha * faded_actions +
                           (1 - self.ema_alpha) * self.prev_smoothed_actions)

        action_delta = smoothed_actions - self.prev_smoothed_actions
        action_delta = np.clip(action_delta, -self.action_rate_limit, self.action_rate_limit)
        final_actions = self.prev_smoothed_actions + action_delta

        self.prev_actions = clipped_actions.copy()
        self.prev_smoothed_actions = final_actions.copy()

        target_sim = self.sim_default_positions + final_actions * self.ACTION_SCALE
        target_sim = np.clip(target_sim, self.joint_lower_limits, self.joint_upper_limits)

        target_matrix = self._isaac_to_hardware_matrix(target_sim)
        self.hardware.set_actuator_postions(target_matrix)

        self.debug_counter += 1
        if self.debug_counter % 50 == 0:
            pos_rel = obs[12:24]
            ang_vel = obs[3:6]
            gravity = obs[6:9]
            raw_range = f"[{clipped_actions.min():+.2f},{clipped_actions.max():+.2f}]"
            final_range = f"[{final_actions.min():+.2f},{final_actions.max():+.2f}]"
            print(f"Step {self.debug_counter}: "
                  f"pos=[{pos_rel.min():+.2f},{pos_rel.max():+.2f}] "
                  f"raw={raw_range} final={final_range} "
                  f"ang=[{ang_vel[0]:+.2f},{ang_vel[1]:+.2f},{ang_vel[2]:+.2f}] "
                  f"gz={gravity[2]:+.2f}")

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

    def set_velocity_command(self, vx_sim, vy_sim, vyaw_sim):
        """
        Set velocity command in SIMULATION frame.
        Will be transformed to hardware frame internally.
        """
        self.velocity_command_sim = np.array([vx_sim, vy_sim, vyaw_sim])

        # Transform to hardware frame for hardware execution
        vx_hw, vy_hw, vyaw_hw = self._sim_velocity_to_hw_velocity(vx_sim, vy_sim, vyaw_sim)
        self.velocity_command_hw = np.array([vx_hw, vy_hw, vyaw_hw])

        if np.any(np.abs(self.velocity_command_sim) > 0.01):
            if not self.control_active:
                self.startup_steps = 0
                self.estimated_lin_vel = np.zeros(3)
                self.prev_smoothed_actions = np.zeros(12)
            self.control_active = True
            print(f"[CMD] sim: vx={vx_sim:.2f}, vy={vy_sim:.2f}, vyaw={vyaw_sim:.2f}")
            print(f"      hw:  vx={vx_hw:.2f}, vy={vy_hw:.2f}, vyaw={vyaw_hw:.2f}")
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
        elif param == 'ema':
            self.ema_alpha = float(value)
            print(f"[PARAM] ema_alpha = {self.ema_alpha}")
        elif param == 'rate':
            self.action_rate_limit = float(value)
            print(f"[PARAM] action_rate_limit = {self.action_rate_limit}")
        elif param == 'simple':
            self.use_simple_lin_vel = value.lower() in ['true', '1', 'yes']
            print(f"[PARAM] use_simple_lin_vel = {self.use_simple_lin_vel}")
        elif param == 'health':
            self.health_check_enabled = value.lower() in ['true', '1', 'yes']
            print(f"[PARAM] health_check_enabled = {self.health_check_enabled}")
        elif param == 'cmd_remap':
            self.cmd_remap_mode = int(value)
            print(f"[PARAM] cmd_remap_mode = {self.cmd_remap_mode}")
        elif param == 'imu_remap':
            self.imu_remap_mode = int(value)
            print(f"[PARAM] imu_remap_mode = {self.imu_remap_mode}")
        else:
            print(f"Unknown param: {param}")
            print("Available: scale, ema, rate, simple, health, cmd_remap, imu_remap")

    # ==============================================================
    # TESTING & VALIDATION
    # ==============================================================

    def test_with_sim_actions(self, csv_path="/home/ubuntu/debug/obs_action_logs_x_030/env_0_actions.csv", max_steps=150):
        """Replay actions from simulation to test transformations."""
        try:
            import pandas as pd
            sim_actions = pd.read_csv(csv_path)
            print(f"\n[TEST] Replaying simulation actions from {csv_path}")
            print(f"[TEST] Total steps in CSV: {len(sim_actions)}, replaying {min(max_steps, len(sim_actions))}")
        except Exception as e:
            print(f"[ERROR] Could not load CSV: {e}")
            return

        print("[TEST] Starting in 3 seconds...")
        time.sleep(3)

        for i in range(min(max_steps, len(sim_actions))):
            row = sim_actions.iloc[i]
            actions = np.array([
                row['action_base_lf1'], row['action_lf1_lf2'], row['action_lf2_lf3'],
                row['action_base_rf1'], row['action_rf1_rf2'], row['action_rf2_rf3'],
                row['action_base_lb1'], row['action_lb1_lb2'], row['action_lb2_lb3'],
                row['action_base_rb1'], row['action_rb1_rb2'], row['action_rb2_rb3'],
            ])

            target_sim = self.sim_default_positions + actions * 0.5
            target_sim = np.clip(target_sim, self.joint_lower_limits, self.joint_upper_limits)

            target_matrix = self._isaac_to_hardware_matrix(target_sim)
            self.hardware.set_actuator_postions(target_matrix)

            time.sleep(0.02)

            if i % 25 == 0:
                print(f"  Step {i}/{min(max_steps, len(sim_actions))}: "
                      f"actions=[{actions.min():+.2f}, {actions.max():+.2f}]")

        print("[TEST] Replay complete.")
        print("[TEST] Returning to default stance in 2 seconds...")
        time.sleep(2)

        target_matrix = self._isaac_to_hardware_matrix(self.sim_default_positions)
        self.hardware.set_actuator_postions(target_matrix)


# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("MINI PUPPER 2 - COORDINATE FRAME FIX v4")
    print("="*70)
    print("""
Controls:
  w/s      - Forward/Backward (sim frame)
  a/d      - Strafe Left/Right (sim frame)
  q/e      - Turn Left/Right (sim frame)
  SPACE    - Stop
  +/-      - Adjust speed
  debug    - Print debug info
  test     - Replay simulation actions from CSV
  set cmd_remap N  - Change velocity remap mode (1/2/3/0)
  set imu_remap N  - Change IMU remap mode (1/2/3/0)
  x        - Exit

Note: Commands are now in SIMULATION frame and will be
      automatically transformed to hardware frame.
""")

    controller = FixedMappingControllerV4("/home/ubuntu/mp2_mlp/policy_joyboy.pt")

    control_thread = threading.Thread(target=controller.control_loop, daemon=True)
    control_thread.start()

    speed = 0.10

    try:
        while True:
            cmd = input("> ").strip().lower()

            if cmd == 'w':
                controller.set_velocity_command(speed, 0, 0)  # Forward in sim frame
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
            elif cmd == 'test':
                controller.set_velocity_command(0, 0, 0)
                time.sleep(0.5)
                controller.test_with_sim_actions()
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
