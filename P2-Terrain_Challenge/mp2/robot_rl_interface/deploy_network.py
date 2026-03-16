"""
Deployment controllers for Mini Pupper 2.

Two policies, one principle: open-loop PD sim for observations, no servo feedback.

  LSTM: Works at HW_SCALE 0.40–0.50 (default 0.45). Subtle, accurate directional control.
  MLP:  Bigger action amplitudes. Start at HW_SCALE 0.25, tune up.
        MLP has no hidden state so it's more forgiving of obs mismatch,
        but still trained with DelayedPDActuator so needs PD sim obs.

Both: effort=0, lin_vel=0, open-loop PD with I=0.20 (training value).

Usage:
  python deploy.py lstm          # LSTM open-loop (recommended)
  python deploy.py mlp           # MLP open-loop
  python deploy.py lstm_25hz     # LSTM at 25Hz (if servos struggle at 50Hz)
  python deploy.py mlp_25hz      # MLP at 25Hz
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
            self.w1 = value.copy(); self.w2 = value.copy()
        else:
            self.w1 = np.zeros(self.n); self.w2 = np.zeros(self.n)


class ComplementaryFilter:
    def __init__(self, dt=0.02):
        self.alpha_g = 0.02
        self.alpha_v = 0.95
        self.dt = dt
        self.gravity_est = np.array([0.0, 0.0, -1.0])
        self.velocity_est = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self._cal_g = []; self._cal_a = []

    def calibrate(self, gyro, accel):
        self._cal_g.append(gyro.copy()); self._cal_a.append(accel.copy())

    def finish_calibration(self):
        if self._cal_g: self.gyro_bias = np.mean(self._cal_g, axis=0)
        if self._cal_a:
            a = np.mean(self._cal_a, axis=0); n = np.linalg.norm(a)
            if n > 0.1: self.gravity_est = a / n
        self._cal_g = []; self._cal_a = []

    def update(self, gyro, accel, dt=None):
        dt = dt or self.dt
        omega = gyro - self.gyro_bias
        g = self.gravity_est - np.cross(omega, self.gravity_est) * dt
        an = np.linalg.norm(accel)
        ga = accel / an if an > 0.1 else self.gravity_est
        self.gravity_est = (1-self.alpha_g)*g + self.alpha_g*ga
        gn = np.linalg.norm(self.gravity_est)
        if gn > 0.1: self.gravity_est /= gn
        la = accel - self.gravity_est * 9.81
        self.velocity_est = self.alpha_v * (self.velocity_est + la * dt)
        return self.velocity_est.copy(), omega.copy(), self.gravity_est.copy()


class DeployController:

    PRESETS = {
        # ==================== LSTM ====================
        "lstm": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_LSTM.pt",
            "HW_SCALE": 0.45,
            "ACTION_SCALE": 0.5,
            "ema_alpha": 1.0,             # Butterworth only
            "control_frequency": 50,
            "butterworth_cutoff": 8.0,
            "pd_delay_steps": 9,          # 9 * 20ms = 180ms (exact training)
            "pd_substeps": 4,
            "pd_stiffness": 70.0,
            "pd_damping": 1.2,
            "pd_inertia": 0.20,           # TRAINING VALUE
            "pd_friction": 0.03,
            "pd_effort_limit": 5.0,
            "description": "LSTM open-loop @ 50Hz, HW_SCALE=0.45",
        },
        "lstm_25hz": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_LSTM.pt",
            "HW_SCALE": 0.45,
            "ACTION_SCALE": 0.5,
            "ema_alpha": 1.0,
            "control_frequency": 25,
            "butterworth_cutoff": 8.0,
            "pd_delay_steps": 5,          # 5 * 40ms = 200ms (~training)
            "pd_substeps": 4,
            "pd_stiffness": 70.0,
            "pd_damping": 1.2,
            "pd_inertia": 0.20,
            "pd_friction": 0.03,
            "pd_effort_limit": 5.0,
            "description": "LSTM open-loop @ 25Hz, HW_SCALE=0.45",
        },
        # ==================== MLP ====================
        "mlp": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_hippy.pt",
            "HW_SCALE": 0.25,             # MLP has 2-3× larger actions — start small
            "ACTION_SCALE": 0.5,
            "ema_alpha": 1.0,
            "control_frequency": 50,
            "butterworth_cutoff": 10.0,    # MLP gait ~7.7Hz, need higher cutoff
            "pd_delay_steps": 9,
            "pd_substeps": 4,
            "pd_stiffness": 70.0,
            "pd_damping": 1.2,
            "pd_inertia": 0.20,
            "pd_friction": 0.03,
            "pd_effort_limit": 5.0,
            "description": "MLP open-loop @ 50Hz, HW_SCALE=0.25",
        },
        "mlp_25hz": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_hippy.pt",
            "HW_SCALE": 0.25,
            "ACTION_SCALE": 0.5,
            "ema_alpha": 1.0,
            "control_frequency": 25,
            "butterworth_cutoff": 10.0,
            "pd_delay_steps": 5,
            "pd_substeps": 4,
            "pd_stiffness": 70.0,
            "pd_damping": 1.2,
            "pd_inertia": 0.20,
            "pd_friction": 0.03,
            "pd_effort_limit": 5.0,
            "description": "MLP open-loop @ 25Hz, HW_SCALE=0.25",
        },
    }

    def __init__(self, preset="lstm"):
        print("=" * 60)
        print(f"DEPLOY CONTROLLER — {preset}")
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

        self.real_default_positions = np.array([
            0.0, 0.785, -0.785,   # LF
            0.0, 0.785, -0.785,   # RF
            0.0, 0.785, -0.785,   # LB
            0.0, 0.785, -0.785,   # RB
        ])
        self.sim_default_positions = np.array([
            0.0, 0.785, -1.57,    # LF
            0.0, 0.785, -1.57,    # RF
            0.0, 0.785, -1.57,    # LB
            0.0, 0.785, -1.57,    # RB
        ])
        self.joint_direction = np.ones(12)
        self.joint_lower = np.array([-0.5, 0.0, -2.5] * 4)
        self.joint_upper = np.array([0.5, 1.5, -0.5] * 4)
        self.hw_to_isaac = {0: 1, 1: 0, 2: 3, 3: 2}
        self.gyro_scale = np.pi / 180.0
        self.accel_scale = 9.81
        self.gyro_offset = np.zeros(3)

        # State
        self.prev_actions = np.zeros(12)
        self.prev_smoothed = np.zeros(12)
        self.prev_time = time.time()
        self.velocity_command = np.zeros(3)
        self.control_active = False
        self.shutdown = False
        self.startup_steps = 0
        self.step_count = 0

        # PD sim state
        self.pd_pos = self.sim_default_positions.copy()
        self.pd_vel = np.zeros(12)
        self.pd_buf = deque(maxlen=self.pd_delay_steps + 1)
        for _ in range(self.pd_delay_steps + 1):
            self.pd_buf.append(np.zeros(12))
        self.syn_pos_rel = np.zeros(12)
        self.syn_vel = np.zeros(12)
        self.syn_effort = np.zeros(12)

        # Logging
        self.log_enabled = False
        self.log_rows = []
        self.log_max = 600

        # Comp filter for gravity/ang_vel (the two real IMU signals we use)
        self.comp_filter = ComplementaryFilter(dt=1.0/self.CONTROL_FREQUENCY)

        print("[CALIBRATING]...")
        self._calibrate(50)
        self._print_config()
        print("[READY]\n")

    def _apply_config(self, cfg):
        self.CONTROL_FREQUENCY = cfg.get("control_frequency", 50)
        self.ACTION_SCALE = cfg.get("ACTION_SCALE", 0.5)
        self.HW_SCALE = cfg.get("HW_SCALE", 0.45)
        self.ema_alpha = cfg.get("ema_alpha", 1.0)
        self.pd_delay_steps = cfg.get("pd_delay_steps", 9)
        self.pd_substeps = cfg.get("pd_substeps", 4)
        self.pd_Kp = cfg.get("pd_stiffness", 70.0)
        self.pd_Kd = cfg.get("pd_damping", 1.2)
        self.pd_I = cfg.get("pd_inertia", 0.20)
        self.pd_friction = cfg.get("pd_friction", 0.03)
        self.pd_effort_limit = cfg.get("pd_effort_limit", 5.0)
        bw_cutoff = cfg.get("butterworth_cutoff", 8.0)
        self.action_lpf = ButterworthLPF(bw_cutoff, self.CONTROL_FREQUENCY, 12)
        self.bw_cutoff = bw_cutoff
        self.startup_dur = int(4.0 * self.CONTROL_FREQUENCY)

    def _print_config(self):
        wn = np.sqrt(self.pd_Kp / self.pd_I)
        z = self.pd_Kd / (2.0 * np.sqrt(self.pd_Kp * self.pd_I))
        print(f"\n[CONFIG] {self.current_preset}")
        print(f"  {self.CONTROL_FREQUENCY}Hz | HW_SCALE={self.HW_SCALE} | Butterworth={self.bw_cutoff}Hz")
        print(f"  PD: Kp={self.pd_Kp} Kd={self.pd_Kd} I={self.pd_I} delay={self.pd_delay_steps}")
        print(f"  PD: wn={wn:.1f}rad/s ({wn/(2*np.pi):.1f}Hz) zeta={z:.2f}")
        print(f"  Open-loop: effort=0, lin_vel=0, joint obs from PD sim")

    def _calibrate(self, n=50):
        for _ in range(n):
            imu = self.esp32.imu_get_data()
            if imu:
                self.comp_filter.calibrate(
                    np.array([imu['gx'], imu['gy'], imu['gz']]) * self.gyro_scale,
                    np.array([imu['ax'], imu['ay'], imu['az']]) * self.accel_scale)
            time.sleep(0.02)
        self.comp_filter.finish_calibration()

    # ------------------------------------------------------------------
    # Hardware I/O
    # ------------------------------------------------------------------

    def _send(self, sim_angles):
        real = self.real_default_positions + (sim_angles - self.sim_default_positions) * self.joint_direction
        m = np.zeros((3, 4))
        m[:, 1] = real[0:3]; m[:, 0] = real[3:6]
        m[:, 3] = real[6:9]; m[:, 2] = real[9:12]
        self.hardware.set_actuator_postions(m)

    def _read_imu(self):
        imu = self.esp32.imu_get_data()
        g = np.array([imu['gx'], imu['gy'], imu['gz']]) * self.gyro_scale
        a = np.array([imu['ax'], imu['ay'], imu['az']]) * self.accel_scale
        return g, a

    # ------------------------------------------------------------------
    # Open-loop PD simulation (matches training exactly)
    # ------------------------------------------------------------------

    def _step_pd(self, action):
        self.pd_buf.append(action.copy())
        delayed = self.pd_buf[0]
        target = self.sim_default_positions + delayed * self.ACTION_SCALE
        target = np.clip(target, self.joint_lower, self.joint_upper)

        dt_sub = (1.0 / self.CONTROL_FREQUENCY) / self.pd_substeps
        for _ in range(self.pd_substeps):
            err = target - self.pd_pos
            tau = self.pd_Kp * err - self.pd_Kd * self.pd_vel - self.pd_friction * np.sign(self.pd_vel)
            self.pd_vel += (tau / self.pd_I) * dt_sub
            self.pd_pos += self.pd_vel * dt_sub

        for j in range(12):
            if self.pd_pos[j] < self.joint_lower[j]:
                self.pd_pos[j] = self.joint_lower[j]; self.pd_vel[j] = 0
            elif self.pd_pos[j] > self.joint_upper[j]:
                self.pd_pos[j] = self.joint_upper[j]; self.pd_vel[j] = 0

        self.syn_pos_rel = self.pd_pos - self.sim_default_positions
        self.syn_vel = self.pd_vel.copy()

        final_err = target - self.pd_pos
        tau = self.pd_Kp * final_err - self.pd_Kd * self.pd_vel
        self.syn_effort = np.zeros(12)  # effort = 0 always

    def _reset_pd(self):
        self.pd_pos = self.sim_default_positions.copy()
        self.pd_vel = np.zeros(12)
        self.pd_buf = deque(maxlen=self.pd_delay_steps + 1)
        for _ in range(self.pd_delay_steps + 1):
            self.pd_buf.append(np.zeros(12))
        self.syn_pos_rel = np.zeros(12)
        self.syn_vel = np.zeros(12)
        self.syn_effort = np.zeros(12)

    # ------------------------------------------------------------------
    # Observation (open-loop: IMU real, joints from PD sim)
    # ------------------------------------------------------------------

    def _get_obs(self):
        now = time.time()
        dt = now - self.prev_time

        gyro, accel = self._read_imu()
        _, ang_vel, gravity = self.comp_filter.update(gyro, accel, max(dt, 0.001))

        self.prev_time = now

        return np.concatenate([
            np.zeros(3),                                   # lin_vel = 0
            np.clip(ang_vel, -2.0, 2.0),                  # real IMU
            np.clip(gravity, -1.0, 1.0),                  # real IMU
            self.velocity_command.copy(),                   # commanded
            np.clip(self.syn_pos_rel, -0.9, 0.9),         # PD sim
            np.clip(self.syn_vel, -10.0, 10.0),           # PD sim
            np.zeros(12),                                   # effort = 0
            self.prev_actions.copy(),                       # prev actions
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def control_step(self):
        obs = self._get_obs()

        with torch.no_grad():
            raw = self.policy(torch.tensor(obs).float().unsqueeze(0)).squeeze().numpy()

        clipped = np.clip(raw, -1.0, 1.0)

        # Fade in
        if self.startup_steps < self.startup_dur:
            fade = self.startup_steps / self.startup_dur
            self.startup_steps += 1
        else:
            fade = 1.0

        faded = clipped * fade

        # Butterworth filter (ema_alpha=1.0 means Butterworth is the only smoother)
        smoothed = self.action_lpf.filter(faded)

        # Rate limit
        delta = np.clip(smoothed - self.prev_smoothed, -100.0, 100.0)
        final = self.prev_smoothed + delta

        self.prev_actions = clipped.copy()
        self.prev_smoothed = final.copy()

        # PD sim for next obs
        self._step_pd(clipped)

        # Servo output
        target = self.sim_default_positions + final * self.HW_SCALE
        target = np.clip(target, self.joint_lower, self.joint_upper)
        self._send(target)

        # Log
        if self.log_enabled and len(self.log_rows) < self.log_max:
            row = {'step': self.step_count, 'time': time.time(),
                   'cmd_x': self.velocity_command[0], 'fade': fade,
                   'preset': self.current_preset}
            for i in range(60): row[f'obs_{i}'] = float(obs[i])
            for i in range(12):
                row[f'raw_{i}'] = float(raw[i])
                row[f'final_{i}'] = float(final[i])
            self.log_rows.append(row)
            if len(self.log_rows) == self.log_max:
                self._save_log()

        self.step_count += 1
        if self.step_count % 50 == 0:
            p = obs[12:24]; v = obs[24:36]
            print(f"Step {self.step_count}: "
                  f"pos=[{p.min():+.2f},{p.max():+.2f}] "
                  f"vel=[{v.min():+.2f},{v.max():+.2f}] "
                  f"act=[{final.min():+.2f},{final.max():+.2f}]")

    def control_loop(self):
        print(f"[RUNNING] {self.CONTROL_FREQUENCY}Hz | {self.current_preset}")
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
    # Commands
    # ------------------------------------------------------------------

    def set_velocity(self, vx, vy, vyaw):
        self.velocity_command = np.array([vx, vy, vyaw])
        if np.any(np.abs(self.velocity_command) > 0.01):
            if not self.control_active:
                self.startup_steps = 0
                self.prev_smoothed = np.zeros(12)
                self.prev_actions = np.zeros(12)
                self.action_lpf.reset()
                self._reset_pd()
            self.control_active = True
            print(f"[CMD] vx={vx:.2f} vy={vy:.2f} vyaw={vyaw:.2f}")
        else:
            self.control_active = False
            print("[STOP]")

    def stop(self):
        self.control_active = False
        self.shutdown = True

    def goto_stance(self, dur=3.0):
        print(f"[STANCE] {dur}s...")
        raw = self.esp32.servos_get_position()
        if raw is None:
            print("[STANCE] ERROR: no read"); return

        # Read current real positions
        angles_hw = np.zeros((3, 4))
        for leg in range(4):
            for axis in range(3):
                sid = self.pwm_params.servo_ids[axis, leg]
                pos = raw[sid - 1]
                dev = (self.servo_params.neutral_position - pos) / self.servo_params.micros_per_rad
                angles_hw[axis, leg] = (
                    dev / self.servo_params.servo_multipliers[axis, leg]
                    + self.servo_params.neutral_angles[axis, leg])
        current = np.zeros(12)
        for hw_col, il in self.hw_to_isaac.items():
            for ax in range(3):
                current[il*3+ax] = angles_hw[ax, hw_col]

        tgt = self.real_default_positions.copy()
        dt = 1.0 / self.CONTROL_FREQUENCY
        n = int(dur * self.CONTROL_FREQUENCY)
        for s in range(n):
            a = 0.5 * (1.0 - np.cos(np.pi * min(1.0, (s+1)/n)))
            interp = current*(1-a) + tgt*a
            m = np.zeros((3,4))
            m[:,1]=interp[0:3]; m[:,0]=interp[3:6]; m[:,3]=interp[6:9]; m[:,2]=interp[9:12]
            self.hardware.set_actuator_postions(m)
            time.sleep(dt)
        time.sleep(0.5)
        self.prev_actions = np.zeros(12)
        self.prev_smoothed = np.zeros(12)
        self.action_lpf.reset()
        self._reset_pd()
        self.startup_steps = 0
        print("[STANCE] Ready.")

    def load_preset(self, name):
        if name not in self.PRESETS:
            print(f"[ERROR] Unknown: {name}. Options: {', '.join(self.PRESETS.keys())}")
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
        self.comp_filter = ComplementaryFilter(dt=1.0/self.CONTROL_FREQUENCY)
        self._calibrate(50)
        self.prev_actions = np.zeros(12)
        self.prev_smoothed = np.zeros(12)
        self.startup_steps = 0
        self._reset_pd()
        self._print_config()

    def _save_log(self):
        import csv
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn = f"/home/ubuntu/mp2_mlp/{self.current_preset}_{ts}.csv"
        if not self.log_rows: return
        with open(fn, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(self.log_rows[0].keys()))
            w.writeheader(); w.writerows(self.log_rows)
        print(f"[LOG] {len(self.log_rows)} rows -> {fn}")
        self.log_enabled = False

    def set_param(self, p, v):
        if p == 'hw_scale':
            self.HW_SCALE = float(v); print(f"[PARAM] HW_SCALE={self.HW_SCALE}")
        elif p == 'ema':
            self.ema_alpha = float(v); print(f"[PARAM] ema_alpha={self.ema_alpha}")
        elif p == 'butterworth':
            c = float(v)
            self.action_lpf = ButterworthLPF(c, self.CONTROL_FREQUENCY, 12)
            self.bw_cutoff = c; print(f"[PARAM] butterworth={c}Hz")
        elif p == 'pd_delay':
            self.pd_delay_steps = int(v); self._reset_pd()
            print(f"[PARAM] pd_delay={self.pd_delay_steps}")
        elif p == 'pd_inertia':
            self.pd_I = float(v); print(f"[PARAM] pd_inertia={self.pd_I}")
        else:
            print(f"Unknown: {p}. Try: hw_scale, ema, butterworth, pd_delay, pd_inertia")


# ==================== MAIN ====================

if __name__ == "__main__":
    import sys
    preset = sys.argv[1] if len(sys.argv) > 1 else "lstm"

    ctrl = DeployController(preset=preset)
    thread = threading.Thread(target=ctrl.control_loop, daemon=True)
    thread.start()

    speed = 0.15
    print("""
  w/s/a/d/q/e = move       SPACE = stop       +/- = speed
  stance      = default pose
  log         = capture 600 steps
  set <p> <v> = hw_scale, ema, butterworth, pd_delay, pd_inertia
  debug       = print obs
  preset <n>  = switch (lstm, lstm_25hz, mlp, mlp_25hz)
  presets     = list
  x           = exit

Quick start:  stance → log → w
MLP tuning:   preset mlp → set hw_scale 0.30 → stance → w
""")

    try:
        while True:
            cl = input("> ").strip().lower()
            if cl == 'w':   ctrl.set_velocity(speed, 0, 0)
            elif cl == 's': ctrl.set_velocity(-speed, 0, 0)
            elif cl == 'a': ctrl.set_velocity(0, speed, 0)
            elif cl == 'd': ctrl.set_velocity(0, -speed, 0)
            elif cl == 'q': ctrl.set_velocity(0, 0, speed*2)
            elif cl == 'e': ctrl.set_velocity(0, 0, -speed*2)
            elif cl in ('', ' '): ctrl.set_velocity(0, 0, 0)
            elif cl == '+': speed = min(speed+0.05, 0.5); print(f"[SPEED] {speed:.2f}")
            elif cl == '-': speed = max(speed-0.05, 0.05); print(f"[SPEED] {speed:.2f}")
            elif cl == 'log':
                ctrl.log_rows = []; ctrl.log_enabled = True
                print("[LOG] Started (600 steps)")
            elif cl == 'stance':
                ctrl.set_velocity(0,0,0); time.sleep(0.3); ctrl.goto_stance()
            elif cl == 'debug':
                obs = ctrl._get_obs()
                for lbl, s, e in [('lin_vel',0,3),('ang_vel',3,6),('gravity',6,9),('cmd',9,12)]:
                    v = obs[s:e]; print(f"  {lbl:10s}: [{v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f}]")
                for lbl, s, e in [('pos_rel',12,24),('vel',24,36),('effort',36,48),('prev_act',48,60)]:
                    v = obs[s:e]; print(f"  {lbl:10s}: |m|={np.abs(v).mean():.3f} [{v.min():+.3f},{v.max():+.3f}]")
            elif cl.startswith('preset '):
                ctrl.load_preset(cl.split(None,1)[1].strip())
            elif cl.startswith('set '):
                parts = cl.split()
                if len(parts)==3: ctrl.set_param(parts[1], parts[2])
            elif cl == 'presets':
                for n,c in DeployController.PRESETS.items():
                    mark = " <-" if n == ctrl.current_preset else ""
                    print(f"  {n:15s} {c['description']}{mark}")
            elif cl == 'x': break
            else: print("Try: w, stance, log, debug, set, preset, presets, x")
    except KeyboardInterrupt:
        pass

    ctrl.stop()
    thread.join(timeout=1.0)
    print("[DONE]")
    