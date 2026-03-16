"""
LSTM Controller v4 — Three observation modes:

  1. "openloop"  — No feedback at all. PD sim driven by policy actions,
                   but the LSTM just sees its own actions echoed back.
                   Closest to the CSV playback that actually works.

  2. "legacy"    — Open-loop PD ODE simulation. Policy actions feed a
                   delayed mass-spring-damper that produces oscillating
                   pos/vel matching training. Servos never feed back.
                   pd_inertia=0.20 (TRAINING VALUE, not 2.0).

  3. "hw_observer" — Real servo positions fed back. Only use if you've
                     verified the dynamics match training (they don't
                     currently — servos don't oscillate like PD sim).

All three: effort=0, lin_vel=0 (can't estimate on HW).
"""

import numpy as np
import torch
import time
import threading
from collections import deque
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration


class ButterworthLPF:
    def __init__(self, cutoff_hz=3.0, fs=50.0, n_channels=12):
        self.n = n_channels
        wc = 2.0 * np.pi * cutoff_hz
        wc_d = 2.0 * fs * np.tan(wc / (2.0 * fs))
        K = wc_d / (2.0 * fs)
        K2 = K * K
        sqrt2_K = np.sqrt(2.0) * K
        norm = 1.0 + sqrt2_K + K2
        self.b0 = K2 / norm
        self.b1 = 2.0 * K2 / norm
        self.b2 = K2 / norm
        self.a1 = 2.0 * (K2 - 1.0) / norm
        self.a2 = (1.0 - sqrt2_K + K2) / norm
        self.w1 = np.zeros(n_channels)
        self.w2 = np.zeros(n_channels)

    def filter(self, x):
        w0 = x - self.a1 * self.w1 - self.a2 * self.w2
        y = self.b0 * w0 + self.b1 * self.w1 + self.b2 * self.w2
        self.w2 = self.w1.copy()
        self.w1 = w0.copy()
        return y

    def reset(self, value=None):
        if value is not None:
            self.w1 = value.copy()
            self.w2 = value.copy()
        else:
            self.w1 = np.zeros(self.n)
            self.w2 = np.zeros(self.n)


class ComplementaryFilter:
    def __init__(self, alpha_gravity=0.02, alpha_vel=0.95, dt=0.02):
        self.alpha_g = alpha_gravity
        self.alpha_v = alpha_vel
        self.dt = dt
        self.gravity_est = np.array([0.0, 0.0, -1.0])
        self.velocity_est = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self._cal_gyro = []
        self._cal_accel = []

    def calibrate(self, gyro, accel):
        self._cal_gyro.append(gyro.copy())
        self._cal_accel.append(accel.copy())

    def finish_calibration(self):
        if self._cal_gyro:
            self.gyro_bias = np.mean(self._cal_gyro, axis=0)
        if self._cal_accel:
            a = np.mean(self._cal_accel, axis=0)
            n = np.linalg.norm(a)
            if n > 0.1:
                self.gravity_est = a / n
        self._cal_gyro = []
        self._cal_accel = []

    def update(self, gyro, accel, dt=None):
        if dt is None:
            dt = self.dt
        omega = gyro - self.gyro_bias
        g_gyro = self.gravity_est - np.cross(omega, self.gravity_est) * dt
        an = np.linalg.norm(accel)
        g_accel = accel / an if an > 0.1 else self.gravity_est
        self.gravity_est = (1 - self.alpha_g) * g_gyro + self.alpha_g * g_accel
        gn = np.linalg.norm(self.gravity_est)
        if gn > 0.1:
            self.gravity_est /= gn
        lin_accel = accel - self.gravity_est * 9.81
        self.velocity_est = self.alpha_v * (self.velocity_est + lin_accel * dt)
        return self.velocity_est.copy(), omega.copy(), self.gravity_est.copy()


class LSTMController:

    PRESETS = {
        # === RECOMMENDED: Open-loop — same as CSV playback but with live policy ===
        "openloop_25hz": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_LSTM.pt",
            "obs_mode": "openloop",
            "HW_SCALE": 1.0,
            "ACTION_SCALE": 0.5,
            "ema_alpha": 1.0,            # Butterworth only
            "control_frequency": 25,
            "use_butterworth": True,
            "butterworth_cutoff": 8.0,
            "bias_scale": 0.0,
            "use_comp_filter": True,
            # PD params for the internal sim that builds obs
            "pd_delay_steps": 5,         # 5 * 40ms = 200ms (training: 9 * 20ms = 180ms)
            "pd_substeps": 4,
            "pd_stiffness": 70.0,
            "pd_damping": 1.2,
            "pd_inertia": 0.20,          # ← TRAINING VALUE
            "pd_friction": 0.03,
            "pd_effort_limit": 5.0,
            "pd_effort_clamp": 0.0,
            "lin_vel_mode": "zero",
            "description": "OPEN-LOOP @ 25Hz — PD sim obs, no servo feedback",
        },
        "openloop_50hz": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_LSTM.pt",
            "obs_mode": "openloop",
            "HW_SCALE": 1.5,
            "ACTION_SCALE": 0.5,
            "ema_alpha": 1.0,
            "control_frequency": 50,
            "use_butterworth": True,
            "butterworth_cutoff": 8.0,
            "bias_scale": 0.0,
            "use_comp_filter": True,
            "pd_delay_steps": 9,         # 9 * 20ms = 180ms (exact training value)
            "pd_substeps": 4,
            "pd_stiffness": 70.0,
            "pd_damping": 1.2,
            "pd_inertia": 0.20,          # ← TRAINING VALUE
            "pd_friction": 0.03,
            "pd_effort_limit": 5.0,
            "pd_effort_clamp": 0.0,
            "lin_vel_mode": "zero",
            "description": "OPEN-LOOP @ 50Hz — exact training PD timing",
        },
        # === Legacy PD (same as openloop but obs come from PD ODE) ===
        "legacy_25hz": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_LSTM.pt",
            "obs_mode": "legacy",
            "HW_SCALE": 1.0,
            "ACTION_SCALE": 0.5,
            "ema_alpha": 1.0,
            "control_frequency": 25,
            "use_butterworth": True,
            "butterworth_cutoff": 8.0,
            "bias_scale": 0.0,
            "use_comp_filter": True,
            "pd_delay_steps": 5,
            "pd_substeps": 4,
            "pd_stiffness": 70.0,
            "pd_damping": 1.2,
            "pd_inertia": 0.20,          # ← TRAINING VALUE
            "pd_friction": 0.03,
            "pd_effort_limit": 5.0,
            "pd_effort_clamp": 0.0,
            "lin_vel_mode": "zero",
            "description": "LEGACY PD @ 25Hz — PD ODE obs, correct I=0.20",
        },
        "legacy_50hz": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_LSTM.pt",
            "obs_mode": "legacy",
            "HW_SCALE": 1.5,
            "ACTION_SCALE": 0.5,
            "ema_alpha": 1.0,
            "control_frequency": 50,
            "use_butterworth": True,
            "butterworth_cutoff": 8.0,
            "bias_scale": 0.0,
            "use_comp_filter": True,
            "pd_delay_steps": 9,
            "pd_substeps": 4,
            "pd_stiffness": 70.0,
            "pd_damping": 1.2,
            "pd_inertia": 0.20,
            "pd_friction": 0.03,
            "pd_effort_limit": 5.0,
            "pd_effort_clamp": 0.0,
            "lin_vel_mode": "zero",
            "description": "LEGACY PD @ 50Hz — exact training dynamics",
        },
    }

    def __init__(self, preset="openloop_50hz"):
        print("=" * 60)
        print("LSTM CONTROLLER v4")
        print("=" * 60)

        self.config = Configuration()
        self.hardware = HardwareInterface()
        self.pwm_params = self.hardware.pwm_params
        self.servo_params = self.hardware.servo_params
        self.esp32 = self.pwm_params.esp32
        time.sleep(0.3)

        self.current_preset = preset
        cfg = self.PRESETS[preset]
        self._apply_config(cfg)

        self.policy = torch.jit.load(cfg["policy"])
        self.policy.eval()
        print(f"[OK] Policy: {cfg['policy']}")

        # Constants
        self.real_default_positions = np.array([
            0.0, 0.785, -0.785,
            0.0, 0.785, -0.785,
            0.0, 0.785, -0.785,
            0.0, 0.785, -0.785,
        ])
        self.sim_default_positions = np.array([
            0.0, 0.785, -1.57,
            0.0, 0.785, -1.57,
            0.0, 0.785, -1.57,
            0.0, 0.785, -1.57,
        ])
        self.joint_direction = np.ones(12)
        self.joint_lower_limits = np.array([-0.5, 0.0, -2.5] * 4)
        self.joint_upper_limits = np.array([0.5, 1.5, -0.5] * 4)
        self.hw_to_isaac_leg = {0: 1, 1: 0, 2: 3, 3: 2}
        self.gyro_scale = np.pi / 180.0
        self.accel_scale = 9.81
        self.gyro_offset = np.zeros(3)

        # State
        self.prev_actions = np.zeros(12)
        self.prev_smoothed_actions = np.zeros(12)
        self.prev_joint_angles_sim = self.sim_default_positions.copy()
        self.prev_time = time.time()
        self.velocity_command = np.zeros(3)
        self.control_active = False
        self.shutdown = False
        self._last_hw_pos_sim = self.sim_default_positions.copy()
        self._last_dt = 1.0 / self.CONTROL_FREQUENCY
        self.startup_steps = 0
        self.debug_counter = 0

        self._reset_pd_state()
        self._reset_hw_observer()

        self.log_enabled = False
        self.log_rows = []
        self.log_max_steps = 600

        print("[CALIBRATING]...")
        self._calibrate_sensors()
        self._print_config()
        print("[READY]")

    # ------------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------------

    def _apply_config(self, cfg):
        self.CONTROL_FREQUENCY = cfg.get("control_frequency", 50)
        self.ACTION_SCALE = cfg.get("ACTION_SCALE", 0.5)
        self.HW_SCALE = cfg.get("HW_SCALE", 1.0)
        self.ema_alpha = cfg.get("ema_alpha", 1.0)
        self.bias_scale = cfg.get("bias_scale", 0.0)
        self.action_rate_limit = 100.0
        self.obs_mode = cfg.get("obs_mode", "openloop")
        self.lin_vel_mode = cfg.get("lin_vel_mode", "zero")

        # PD dynamics (TRAINING VALUES by default)
        self.pd_delay_steps = cfg.get("pd_delay_steps", 9)
        self.pd_substeps = cfg.get("pd_substeps", 4)
        self.pd_stiffness = cfg.get("pd_stiffness", 70.0)
        self.pd_damping = cfg.get("pd_damping", 1.2)
        self.pd_inertia = cfg.get("pd_inertia", 0.20)   # ← 0.20 not 2.0!
        self.pd_friction = cfg.get("pd_friction", 0.03)
        self.pd_effort_limit = cfg.get("pd_effort_limit", 5.0)
        self.pd_effort_clamp = cfg.get("pd_effort_clamp", 0.0)

        self.obs_vel_alpha = cfg.get("vel_alpha", 0.15)

        bw = cfg.get("use_butterworth", True)
        self.butterworth_cutoff = cfg.get("butterworth_cutoff", 8.0)
        self.action_lpf = ButterworthLPF(self.butterworth_cutoff, self.CONTROL_FREQUENCY, 12) if bw else None

        self.use_comp_filter = cfg.get("use_comp_filter", True)
        self.comp_filter = ComplementaryFilter(dt=1.0/self.CONTROL_FREQUENCY) if self.use_comp_filter else None

        self.startup_duration = int(4.0 * self.CONTROL_FREQUENCY)

    def load_preset(self, name):
        if name not in self.PRESETS:
            print(f"[ERROR] Unknown: {name}. Available: {', '.join(self.PRESETS.keys())}")
            return
        self.control_active = False
        time.sleep(0.1)
        cfg = self.PRESETS[name]
        self.current_preset = name
        try:
            self.policy = torch.jit.load(cfg["policy"])
            self.policy.eval()
        except Exception as e:
            print(f"[ERROR] {e}"); return
        self._apply_config(cfg)
        self.prev_actions = np.zeros(12)
        self.prev_smoothed_actions = np.zeros(12)
        self.startup_steps = 0
        self._reset_pd_state()
        self._reset_hw_observer()
        if self.comp_filter:
            self._calibrate_sensors()
        self._print_config()

    def _print_config(self):
        wn = np.sqrt(self.pd_stiffness / self.pd_inertia)
        zeta = self.pd_damping / (2.0 * np.sqrt(self.pd_stiffness * self.pd_inertia))
        print(f"\n[CONFIG] {self.current_preset}")
        print(f"  obs_mode={self.obs_mode}  freq={self.CONTROL_FREQUENCY}Hz  HW_SCALE={self.HW_SCALE}")
        print(f"  lin_vel={self.lin_vel_mode}  effort_clamp={self.pd_effort_clamp}  ema={self.ema_alpha}")
        print(f"  PD: Kp={self.pd_stiffness} Kd={self.pd_damping} I={self.pd_inertia} delay={self.pd_delay_steps}")
        print(f"  PD: wn={wn:.1f}rad/s ({wn/(2*np.pi):.1f}Hz) zeta={zeta:.2f}")
        if self.action_lpf:
            print(f"  Butterworth: {self.butterworth_cutoff}Hz")
        print()

    # ------------------------------------------------------------------
    # HARDWARE I/O
    # ------------------------------------------------------------------

    def _read_joint_positions_raw(self):
        raw = self.esp32.servos_get_position()
        if raw is None:
            return None
        angles_hw = np.zeros((3, 4))
        for leg in range(4):
            for axis in range(3):
                sid = self.pwm_params.servo_ids[axis, leg]
                pos = raw[sid - 1]
                dev = (self.servo_params.neutral_position - pos) / self.servo_params.micros_per_rad
                angles_hw[axis, leg] = (
                    dev / self.servo_params.servo_multipliers[axis, leg]
                    + self.servo_params.neutral_angles[axis, leg]
                )
        angles_isaac = np.zeros(12)
        for hw_col, isaac_leg in self.hw_to_isaac_leg.items():
            for axis in range(3):
                angles_isaac[isaac_leg * 3 + axis] = angles_hw[axis, hw_col]
        return angles_isaac

    def _real_to_sim(self, angles_real):
        return self.sim_default_positions + (angles_real - self.real_default_positions) * self.joint_direction

    def _sim_to_real(self, angles_sim):
        return self.real_default_positions + (angles_sim - self.sim_default_positions) * self.joint_direction

    def _read_sim_frame(self):
        raw = self._read_joint_positions_raw()
        return self._real_to_sim(raw) if raw is not None else self.prev_joint_angles_sim.copy()

    def _send_to_hardware(self, flat_sim):
        real = self._sim_to_real(flat_sim)
        m = np.zeros((3, 4))
        m[:, 1] = real[0:3]
        m[:, 0] = real[3:6]
        m[:, 3] = real[6:9]
        m[:, 2] = real[9:12]
        self.hardware.set_actuator_postions(m)

    def _calibrate_sensors(self, samples=50):
        gyro_samples = []
        for _ in range(samples):
            imu = self.esp32.imu_get_data()
            if imu:
                gyro_samples.append([imu['gx'], imu['gy'], imu['gz']])
                if self.comp_filter:
                    self.comp_filter.calibrate(
                        np.array([imu['gx'], imu['gy'], imu['gz']]) * self.gyro_scale,
                        np.array([imu['ax'], imu['ay'], imu['az']]) * self.accel_scale)
            time.sleep(0.02)
        if gyro_samples:
            self.gyro_offset = np.mean(gyro_samples, axis=0)
        if self.comp_filter:
            self.comp_filter.finish_calibration()

    # ------------------------------------------------------------------
    # PD SIMULATION (open-loop, matches training)
    # ------------------------------------------------------------------

    def _reset_pd_state(self):
        self.pd_position = self.sim_default_positions.copy()
        self.pd_velocity = np.zeros(12)
        self.pd_action_buffer = deque(maxlen=self.pd_delay_steps + 1)
        for _ in range(self.pd_delay_steps + 1):
            self.pd_action_buffer.append(np.zeros(12))
        self.syn_pos_rel = np.zeros(12)
        self.syn_vel = np.zeros(12)
        self.syn_effort = np.zeros(12)

    def _step_pd_dynamics(self, current_action):
        """Open-loop PD ODE — matches training DelayedPDActuator exactly."""
        self.pd_action_buffer.append(current_action.copy())
        delayed = self.pd_action_buffer[0]

        target = self.sim_default_positions + delayed * self.ACTION_SCALE
        target = np.clip(target, self.joint_lower_limits, self.joint_upper_limits)

        dt_sub = (1.0 / self.CONTROL_FREQUENCY) / self.pd_substeps
        for _ in range(self.pd_substeps):
            error = target - self.pd_position
            torque = self.pd_stiffness * error - self.pd_damping * self.pd_velocity
            torque -= self.pd_friction * np.sign(self.pd_velocity)
            accel = torque / self.pd_inertia
            self.pd_velocity += accel * dt_sub
            self.pd_position += self.pd_velocity * dt_sub

        # Enforce limits
        for j in range(12):
            if self.pd_position[j] < self.joint_lower_limits[j]:
                self.pd_position[j] = self.joint_lower_limits[j]
                self.pd_velocity[j] = 0.0
            elif self.pd_position[j] > self.joint_upper_limits[j]:
                self.pd_position[j] = self.joint_upper_limits[j]
                self.pd_velocity[j] = 0.0

        self.syn_pos_rel = self.pd_position - self.sim_default_positions
        self.syn_vel = self.pd_velocity.copy()

        # Effort from current PD state
        final_err = target - self.pd_position
        final_torque = self.pd_stiffness * final_err - self.pd_damping * self.pd_velocity
        self.syn_effort = np.clip(final_torque / self.pd_effort_limit, -1.0, 1.0)
        if self.pd_effort_clamp is not None:
            self.syn_effort = np.clip(self.syn_effort, -self.pd_effort_clamp, self.pd_effort_clamp)

    # ------------------------------------------------------------------
    # HW OBSERVER (for reference — not recommended for this policy)
    # ------------------------------------------------------------------

    def _reset_hw_observer(self):
        self.obs_prev_hw_pos = self.sim_default_positions.copy()
        self.obs_hw_vel = np.zeros(12)

    def _step_hw_observer(self, action, hw_pos, dt):
        self.pd_action_buffer.append(action.copy())
        delayed = self.pd_action_buffer[0]
        target = self.sim_default_positions + delayed * self.ACTION_SCALE
        target = np.clip(target, self.joint_lower_limits, self.joint_upper_limits)
        if dt > 0.001:
            raw_vel = np.clip((hw_pos - self.obs_prev_hw_pos) / dt, -10.5, 10.5)
            self.obs_hw_vel = self.obs_vel_alpha * raw_vel + (1 - self.obs_vel_alpha) * self.obs_hw_vel
        error = target - hw_pos
        torque = self.pd_stiffness * error - self.pd_damping * self.obs_hw_vel
        effort = np.clip(torque / self.pd_effort_limit, -1.0, 1.0)
        if self.pd_effort_clamp is not None:
            effort = np.clip(effort, -self.pd_effort_clamp, self.pd_effort_clamp)
        self.syn_pos_rel = hw_pos - self.sim_default_positions
        self.syn_vel = self.obs_hw_vel.copy()
        self.syn_effort = effort
        self.obs_prev_hw_pos = hw_pos.copy()

    # ------------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------------

    def get_observation(self):
        now = time.time()
        dt = now - self.prev_time
        current_sim = self._read_sim_frame()
        self._last_hw_pos_sim = current_sim.copy()
        self._last_dt = dt

        # IMU → ang_vel + gravity
        imu = self.esp32.imu_get_data()
        gyro_si = np.array([imu['gx'], imu['gy'], imu['gz']]) * self.gyro_scale
        accel_si = np.array([imu['ax'], imu['ay'], imu['az']]) * self.accel_scale

        if self.comp_filter:
            _, cf_ang, cf_grav = self.comp_filter.update(gyro_si, accel_si, max(dt, 0.001))
            base_ang_vel = cf_ang
            projected_gravity = cf_grav
        else:
            base_ang_vel = (np.array([imu['gx'], imu['gy'], imu['gz']]) - self.gyro_offset) * self.gyro_scale
            projected_gravity = np.array([0.0, 0.0, -1.0])

        # lin_vel
        base_lin_vel = np.zeros(3) if self.lin_vel_mode == "zero" else self.velocity_command * 0.7

        # Joint obs — from PD sim (openloop/legacy) or HW observer
        joint_pos_rel = self.syn_pos_rel.copy()
        joint_vel = self.syn_vel.copy()
        joint_effort = self.syn_effort.copy()

        self.prev_joint_angles_sim = current_sim.copy()
        self.prev_time = now

        return np.concatenate([
            np.clip(base_lin_vel, -0.5, 0.5),
            np.clip(base_ang_vel, -2.0, 2.0),
            np.clip(projected_gravity, -1.0, 1.0),
            self.velocity_command.copy(),
            np.clip(joint_pos_rel, -0.9, 0.9),
            np.clip(joint_vel, -10.0, 10.0),
            np.clip(joint_effort, -1.0, 1.0),
            self.prev_actions.copy(),
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    # CONTROL
    # ------------------------------------------------------------------

    def control_step(self):
        obs = self.get_observation()

        with torch.no_grad():
            raw = self.policy(torch.tensor(obs).float().unsqueeze(0)).squeeze().numpy()

        clipped = np.clip(raw, -1.0, 1.0)

        # Fade
        if self.startup_steps < self.startup_duration:
            fade = self.startup_steps / self.startup_duration
            self.startup_steps += 1
        else:
            fade = 1.0

        faded = clipped * fade

        # Smooth
        if self.action_lpf:
            smoothed = self.action_lpf.filter(faded)
        else:
            smoothed = self.ema_alpha * faded + (1 - self.ema_alpha) * self.prev_smoothed_actions

        delta = np.clip(smoothed - self.prev_smoothed_actions, -self.action_rate_limit, self.action_rate_limit)
        final = self.prev_smoothed_actions + delta

        self.prev_actions = clipped.copy()
        self.prev_smoothed_actions = final.copy()

        # Update obs source
        if self.obs_mode == "hw_observer":
            self._step_hw_observer(clipped, self._last_hw_pos_sim, self._last_dt)
        else:
            # Both "openloop" and "legacy" use the PD ODE
            self._step_pd_dynamics(clipped)

        # Send to servos
        target = self.sim_default_positions + final * self.HW_SCALE
        target = np.clip(target, self.joint_lower_limits, self.joint_upper_limits)
        self._send_to_hardware(target)

        # Log
        if self.log_enabled and len(self.log_rows) < self.log_max_steps:
            row = {'step': self.debug_counter, 'time': time.time(),
                   'cmd_x': self.velocity_command[0], 'fade': fade,
                   'obs_mode': self.obs_mode}
            for i in range(60):
                row[f'obs_{i}'] = float(obs[i])
            for i in range(12):
                row[f'raw_action_{i}'] = float(raw[i])
                row[f'final_action_{i}'] = float(final[i])
            self.log_rows.append(row)
            if len(self.log_rows) == self.log_max_steps:
                self._save_log()

        self.debug_counter += 1
        if self.debug_counter % 50 == 0:
            p = obs[12:24]
            v = obs[24:36]
            e = obs[36:48]
            print(f"Step {self.debug_counter} [{self.obs_mode}]: "
                  f"pos=[{p.min():+.2f},{p.max():+.2f}] "
                  f"vel=[{v.min():+.2f},{v.max():+.2f}] "
                  f"eff=[{e.min():+.2f},{e.max():+.2f}] "
                  f"act=[{final.min():+.2f},{final.max():+.2f}]")

    def control_loop(self):
        print(f"[RUNNING] {self.CONTROL_FREQUENCY}Hz | obs_mode={self.obs_mode}")
        while not self.shutdown:
            dt_target = 1.0 / self.CONTROL_FREQUENCY
            t0 = time.time()
            if self.control_active:
                try:
                    self.control_step()
                except Exception as e:
                    print(f"[ERROR] {e}")
                    import traceback; traceback.print_exc()
                    self.control_active = False
            elapsed = time.time() - t0
            if elapsed < dt_target:
                time.sleep(dt_target - elapsed)

    # ------------------------------------------------------------------
    # COMMANDS
    # ------------------------------------------------------------------

    def set_velocity_command(self, vx, vy, vyaw):
        self.velocity_command = np.array([vx, vy, vyaw])
        if np.any(np.abs(self.velocity_command) > 0.01):
            if not self.control_active:
                self.startup_steps = 0
                self.prev_smoothed_actions = np.zeros(12)
                self.prev_actions = np.zeros(12)
                if self.action_lpf:
                    self.action_lpf.reset()
                self._reset_pd_state()
                self._reset_hw_observer()
            self.control_active = True
            print(f"[CMD] vx={vx:.2f} vy={vy:.2f} vyaw={vyaw:.2f}")
        else:
            self.control_active = False
            print("[STOP]")

    def stop(self):
        self.control_active = False
        self.shutdown = True

    def goto_stance(self, duration=3.0):
        print(f"[STANCE] {duration}s...")
        cur = self._read_joint_positions_raw()
        if cur is None:
            print("[STANCE] ERROR"); return
        tgt = self.real_default_positions.copy()
        dt = 1.0 / self.CONTROL_FREQUENCY
        n = int(duration * self.CONTROL_FREQUENCY)
        for s in range(n):
            a = 0.5 * (1.0 - np.cos(np.pi * min(1.0, (s+1)/n)))
            interp = cur * (1-a) + tgt * a
            m = np.zeros((3,4))
            m[:,1]=interp[0:3]; m[:,0]=interp[3:6]; m[:,3]=interp[6:9]; m[:,2]=interp[9:12]
            self.hardware.set_actuator_postions(m)
            time.sleep(dt)
        time.sleep(0.5)
        self.prev_actions = np.zeros(12)
        self.prev_smoothed_actions = np.zeros(12)
        if self.action_lpf: self.action_lpf.reset()
        self._reset_pd_state()
        self._reset_hw_observer()
        self.startup_steps = 0
        print("[STANCE] Ready.")

    def _save_log(self):
        import csv
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn = f"/home/ubuntu/mp2_mlp/{self.current_preset}_{ts}.csv"
        if not self.log_rows: return
        with open(fn, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(self.log_rows[0].keys()))
            w.writeheader()
            w.writerows(self.log_rows)
        print(f"[LOG] {len(self.log_rows)} rows -> {fn}")
        self.log_enabled = False

    def set_param(self, param, value):
        pmap = {
            'hw_scale':    ('HW_SCALE', float),
            'ema':         ('ema_alpha', float),
            'vel_alpha':   ('obs_vel_alpha', float),
            'pd_delay':    ('pd_delay_steps', int),
            'pd_inertia':  ('pd_inertia', float),
            'pd_damping':  ('pd_damping', float),
        }
        if param in pmap:
            attr, typ = pmap[param]
            setattr(self, attr, typ(value))
            if param == 'pd_delay':
                self._reset_pd_state()
            print(f"[PARAM] {attr} = {getattr(self, attr)}")
        elif param == 'effort_clamp':
            v = float(value)
            self.pd_effort_clamp = v if v >= 0 else None
            print(f"[PARAM] pd_effort_clamp = {self.pd_effort_clamp}")
        elif param == 'lin_vel':
            self.lin_vel_mode = value
            print(f"[PARAM] lin_vel_mode = {value}")
        elif param == 'obs_mode':
            if value in ('openloop', 'legacy', 'hw_observer'):
                self.obs_mode = value
                self._reset_pd_state()
                self._reset_hw_observer()
                print(f"[PARAM] obs_mode = {value}")
            else:
                print("Must be: openloop, legacy, hw_observer")
        elif param == 'butterworth':
            c = float(value)
            self.action_lpf = ButterworthLPF(c, self.CONTROL_FREQUENCY, 12) if c > 0 else None
            print(f"[PARAM] butterworth = {c}")
        else:
            print(f"Unknown: {param}")
            print("  Try: hw_scale, ema, vel_alpha, pd_delay, pd_inertia, pd_damping, effort_clamp, lin_vel, obs_mode, butterworth")


# ==================== MAIN ====================

if __name__ == "__main__":
    import sys
    preset = sys.argv[1] if len(sys.argv) > 1 else "openloop_50hz"

    controller = LSTMController(preset=preset)
    thread = threading.Thread(target=controller.control_loop, daemon=True)
    thread.start()

    speed = 0.15
    print("""
Commands:
  w/s/a/d/q/e  = move         SPACE = stop       +/- = speed
  stance       = default pose
  log          = capture 600 steps
  set <p> <v>  = adjust param (obs_mode, hw_scale, ema, pd_inertia, ...)
  debug        = print observations
  preset <n>   = switch          presets = list
  x            = exit

Recommended test order:
  1. preset openloop_50hz  → stance → log → w   (best match to training)
  2. preset openloop_25hz  → stance → log → w   (if 50hz too fast for servos)
  3. preset legacy_50hz    → stance → log → w   (PD ODE obs)
  4. preset legacy_25hz    → stance → log → w   (PD ODE obs, slower)
""")

    try:
        while True:
            cmd = input("> ").strip()
            cl = cmd.lower()
            if cl == 'w':   controller.set_velocity_command(speed, 0, 0)
            elif cl == 's': controller.set_velocity_command(-speed, 0, 0)
            elif cl == 'a': controller.set_velocity_command(0, speed, 0)
            elif cl == 'd': controller.set_velocity_command(0, -speed, 0)
            elif cl == 'q': controller.set_velocity_command(0, 0, speed*2)
            elif cl == 'e': controller.set_velocity_command(0, 0, -speed*2)
            elif cl in ('', ' '): controller.set_velocity_command(0, 0, 0)
            elif cl == '+': speed = min(speed+0.05, 0.5); print(f"[SPEED] {speed:.2f}")
            elif cl == '-': speed = max(speed-0.05, 0.05); print(f"[SPEED] {speed:.2f}")
            elif cl == 'log':
                controller.log_rows = []; controller.log_enabled = True
                print("[LOG] Started (600 steps)")
            elif cl == 'stance':
                controller.set_velocity_command(0,0,0); time.sleep(0.3)
                controller.goto_stance()
            elif cl == 'debug':
                obs = controller.get_observation()
                labels = ['lin_vel','ang_vel','gravity','cmd','pos_rel','jnt_vel','effort','prev_act']
                slices = [(0,3),(3,6),(6,9),(9,12),(12,24),(24,36),(36,48),(48,60)]
                print(f"\n[OBS] mode={controller.obs_mode}")
                for lbl,(s,e) in zip(labels, slices):
                    v = obs[s:e]
                    if e-s <= 3:
                        print(f"  {lbl:10s}: [{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}]")
                    else:
                        print(f"  {lbl:10s}: |m|={np.abs(v).mean():.3f}  [{v.min():+.3f},{v.max():+.3f}]")
                print()
            elif cl.startswith('preset '):
                controller.load_preset(cl.split(None,1)[1].strip())
            elif cl.startswith('set '):
                parts = cl.split()
                if len(parts) == 3: controller.set_param(parts[1], parts[2])
            elif cl == 'presets':
                for n,c in LSTMController.PRESETS.items():
                    mark = " <-" if n == controller.current_preset else ""
                    print(f"  {n:20s} {c['description']}{mark}")
            elif cl == 'x': break
            else: print("Try: w, stance, log, debug, set, preset, presets, x")
    except KeyboardInterrupt:
        pass

    controller.stop()
    thread.join(timeout=1.0)
    print("[DONE]")