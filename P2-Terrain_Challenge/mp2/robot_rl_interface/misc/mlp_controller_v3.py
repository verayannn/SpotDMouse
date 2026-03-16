import numpy as np
import torch
import time
import threading
from collections import deque
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration

# ==================== BUTTERWORTH LOW-PASS FILTER ====================
# Replaces EMA smoothing for action output. 2nd-order Butterworth at 3Hz cutoff
# cleanly passes 1-3Hz gait content while rejecting 5-8Hz policy oscillations.
# Unlike EMA, flat passband preserves gait shape; sharp rolloff matches servo BW.

class ButterworthLPF:
    """Per-channel 2nd-order Butterworth low-pass filter (Direct Form II)."""

    def __init__(self, cutoff_hz=3.0, fs=50.0, n_channels=12):
        self.n = n_channels
        # Precompute coefficients (bilinear transform)
        wc = 2.0 * np.pi * cutoff_hz
        wc_d = 2.0 * fs * np.tan(wc / (2.0 * fs))  # pre-warp
        K = wc_d / (2.0 * fs)
        K2 = K * K
        sqrt2_K = np.sqrt(2.0) * K
        norm = 1.0 + sqrt2_K + K2
        self.b0 = K2 / norm
        self.b1 = 2.0 * K2 / norm
        self.b2 = K2 / norm
        self.a1 = 2.0 * (K2 - 1.0) / norm
        self.a2 = (1.0 - sqrt2_K + K2) / norm
        # State: 2 delay elements per channel
        self.w1 = np.zeros(n_channels)
        self.w2 = np.zeros(n_channels)

    def filter(self, x):
        """Filter one sample (array of n_channels)."""
        w0 = x - self.a1 * self.w1 - self.a2 * self.w2
        y = self.b0 * w0 + self.b1 * self.w1 + self.b2 * self.w2
        self.w2 = self.w1.copy()
        self.w1 = w0.copy()
        return y

    def reset(self, value=None):
        """Reset filter state. If value given, initialize to steady-state."""
        if value is not None:
            # Steady-state: output = input, so w1 = w2 = x / (1 + a1 + a2) ... simplified:
            # For DC input x, w_ss = x / (1 - a1 - a2) ... but just warm-start
            self.w1 = value.copy()
            self.w2 = value.copy()
        else:
            self.w1 = np.zeros(self.n)
            self.w2 = np.zeros(self.n)


# ==================== EMBEDDED COMPLEMENTARY FILTER ====================
# Inlined from ai_imu_dr/ai_imu_filter.py to avoid deployment dependency.

class ComplementaryFilter:
    """Fuses gyro (fast, drifts) + accel (slow, noisy) for gravity + velocity."""

    def __init__(self, alpha_gravity=0.02, alpha_vel=0.95, dt=0.02):
        self.alpha_g = alpha_gravity
        self.alpha_v = alpha_vel
        self.dt = dt
        self.gravity_est = np.array([0.0, 0.0, -1.0])
        self.velocity_est = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.calibrated = False
        self._cal_gyro = []
        self._cal_accel = []

    def calibrate(self, gyro, accel):
        self._cal_gyro.append(gyro.copy())
        self._cal_accel.append(accel.copy())

    def finish_calibration(self):
        if self._cal_gyro:
            self.gyro_bias = np.mean(self._cal_gyro, axis=0)
        if self._cal_accel:
            accel_mean = np.mean(self._cal_accel, axis=0)
            norm = np.linalg.norm(accel_mean)
            if norm > 0.1:
                self.gravity_est = accel_mean / norm
        self.calibrated = True
        self._cal_gyro = []
        self._cal_accel = []

    def update(self, gyro, accel, dt=None):
        if dt is None:
            dt = self.dt
        omega = gyro - self.gyro_bias
        g_gyro = self.gravity_est - np.cross(omega, self.gravity_est) * dt
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0.1:
            g_accel = accel / accel_norm
        else:
            g_accel = self.gravity_est
        self.gravity_est = (1 - self.alpha_g) * g_gyro + self.alpha_g * g_accel
        gn = np.linalg.norm(self.gravity_est)
        if gn > 0.1:
            self.gravity_est = self.gravity_est / gn
        lin_accel = accel - self.gravity_est * 9.81
        self.velocity_est = self.alpha_v * (self.velocity_est + lin_accel * dt)
        return self.velocity_est.copy(), omega.copy(), self.gravity_est.copy()

HAS_COMP_FILTER = True


class FixedMappingControllerV3:
    """
    Controller with corrected frame transformations - Version 3
    """

    # ==================== TEST PRESETS ====================
    # Each preset configures policy, PD, EMA, HW_SCALE, and effort handling.
    # effort_mode: "synthetic" (from PD), "real" (raw servo load), "zero", "boost5x"
    PRESETS = {
        # --- MLP presets ---
        "mlp_pd": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_hippy.pt",
            "use_synthetic_obs": True,
            "ema_alpha": 1.0,
            "HW_SCALE": 0.55,
            "ACTION_SCALE": 0.5,
            "pd_delay_steps": 9,
            "effort_mode": "synthetic",
            "description": "MLP + PD wrapper (baseline, no EMA)",
        },
        "mlp_pd_ema": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_hippy.pt",
            "use_synthetic_obs": True,
            "ema_alpha": 0.3,
            "HW_SCALE": 0.55,
            "ACTION_SCALE": 0.5,
            "pd_delay_steps": 9,
            "effort_mode": "synthetic",
            "description": "MLP + PD wrapper + EMA (target ~2Hz gait)",
        },
        "mlp_nopd": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_hippy.pt",
            "use_synthetic_obs": False,
            "ema_alpha": 1.0,
            "HW_SCALE": 0.55,
            "ACTION_SCALE": 0.5,
            "pd_delay_steps": 9,
            "effort_mode": "real",
            "description": "MLP + real sensors (no PD — expect failure)",
        },
        # --- LSTM presets ---
        "lstm_pd": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_LSTM.pt",
            "use_synthetic_obs": True,
            "ema_alpha": 0.3,
            "HW_SCALE": 1.5,
            "ACTION_SCALE": 0.5,
            "pd_delay_steps": 9,
            "effort_mode": "synthetic",
            "description": "LSTM + PD wrapper + EMA (freq-limited)",
        },
        "lstm_nopd": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_LSTM.pt",
            "use_synthetic_obs": False,
            "ema_alpha": 0.3,
            "HW_SCALE": 1.5,
            "ACTION_SCALE": 0.5,
            "pd_delay_steps": 9,
            "effort_mode": "real",
            "description": "LSTM + real sensors + EMA (no PD, raw effort)",
        },
        "lstm_nopd_boost": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_LSTM.pt",
            "use_synthetic_obs": False,
            "ema_alpha": 0.3,
            "HW_SCALE": 1.5,
            "ACTION_SCALE": 0.5,
            "pd_delay_steps": 9,
            "effort_mode": "boost5x",
            "description": "LSTM + real sensors + effort x5 (match training magnitude)",
        },
        "lstm_nopd_zero": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_LSTM.pt",
            "use_synthetic_obs": False,
            "ema_alpha": 0.3,
            "HW_SCALE": 1.5,
            "ACTION_SCALE": 0.5,
            "pd_delay_steps": 9,
            "effort_mode": "zero",
            "description": "LSTM + real sensors + zero effort (remove OOD signal)",
        },
        # --- ComplementaryFilter presets (AI-IMU fallback) ---
        "mlp_pd_cf": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_hippy.pt",
            "use_synthetic_obs": True,
            "ema_alpha": 1.0,
            "HW_SCALE": 0.55,
            "ACTION_SCALE": 0.5,
            "pd_delay_steps": 9,
            "effort_mode": "synthetic",
            "use_comp_filter": True,
            "description": "MLP + PD + ComplementaryFilter (real base_lin_vel + gravity)",
        },
        "lstm_nopd_cf": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_LSTM.pt",
            "use_synthetic_obs": False,
            "ema_alpha": 0.3,
            "HW_SCALE": 1.5,
            "ACTION_SCALE": 0.5,
            "pd_delay_steps": 9,
            "effort_mode": "real",
            "use_comp_filter": True,
            "description": "LSTM + no PD + ComplementaryFilter (real base_lin_vel + gravity)",
        },
        # --- Hardware-driven PD observer presets ---
        # Uses real joint positions + PD-computed velocity/effort (closed-loop observer)
        # MLP gait=7.7Hz, LSTM gait=5.3Hz — cutoff must be ABOVE gait freq
        # Butterworth removes noise/aliasing above gait while preserving the signal
        "mlp_pd_hw": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_hippy.pt",
            "use_synthetic_obs": True,
            "obs_mode": "hw_observer",
            "ema_alpha": 1.0,           # no EMA (Butterworth handles smoothing)
            "HW_SCALE": 0.55,
            "ACTION_SCALE": 0.5,
            "pd_delay_steps": 9,
            "effort_mode": "synthetic",
            "use_comp_filter": True,
            "use_butterworth": True,
            "butterworth_cutoff": 10.0,  # above 7.7Hz gait, removes >10Hz noise
            "bias_scale": 0.0,          # disable stale action bias
            "vel_alpha": 0.15,          # smoother velocity obs (quantization noise)
            "description": "MLP + HW observer + Butterworth 10Hz",
        },
        "lstm_pd_hw": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_LSTM.pt",
            "use_synthetic_obs": True,
            "obs_mode": "hw_observer",
            "ema_alpha": 0.3,           # fallback if Butterworth disabled
            "HW_SCALE": 1.5,
            "ACTION_SCALE": 0.5,
            "pd_delay_steps": 9,
            "effort_mode": "synthetic",
            "use_comp_filter": True,
            "use_butterworth": True,
            "butterworth_cutoff": 8.0,   # above 5.3Hz gait, removes >8Hz noise
            "bias_scale": 0.0,          # disable stale action bias
            "vel_alpha": 0.15,          # smoother velocity obs (quantization noise)
            "description": "LSTM + HW observer + Butterworth 8Hz",
        },
        # --- 25Hz deployment presets ---
        # Instead of retraining at lower frequency, deploy at 25Hz.
        # Each action held 40ms (vs 20ms at 50Hz) → servos have 2x longer to track.
        # MLP 7.7Hz gait well below 12.5Hz Nyquist. LSTM 5.3Hz gait comfortable.
        # Key adjustments: delay_steps halved (keep ~160-200ms effective delay),
        # pd_substeps=2 (100Hz physics / 25Hz policy), HW_SCALE reduced (~0.7x).
        "mlp_25hz": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_hippy.pt",
            "use_synthetic_obs": True,
            "obs_mode": "hw_observer",
            "ema_alpha": 1.0,
            "HW_SCALE": 0.40,           # reduced from 0.55 (actions held 2x longer)
            "ACTION_SCALE": 0.5,
            "pd_delay_steps": 5,         # 5 * 40ms = 200ms (was 9 * 20ms = 180ms)
            "pd_substeps": 2,            # 2 substeps * 25Hz = 50Hz physics (was 4 * 50Hz = 200Hz)
            "effort_mode": "synthetic",
            "use_comp_filter": True,
            "use_butterworth": True,
            "butterworth_cutoff": 10.0,
            "bias_scale": 0.0,
            "vel_alpha": 0.15,
            "control_frequency": 25,
            "description": "MLP + HW observer @ 25Hz (2x servo tracking time)",
        },
        "lstm_25hz": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_LSTM.pt",
            "use_synthetic_obs": True,
            "obs_mode": "hw_observer",
            "ema_alpha": 0.3,
            "HW_SCALE": 1.0,            # reduced from 1.5 (actions held 2x longer)
            "ACTION_SCALE": 0.5,
            "pd_delay_steps": 5,         # 5 * 40ms = 200ms
            "pd_substeps": 2,
            "effort_mode": "synthetic",
            "use_comp_filter": True,
            "use_butterworth": True,
            "butterworth_cutoff": 8.0,
            "bias_scale": 0.0,
            "vel_alpha": 0.15,
            "control_frequency": 25,
            "description": "LSTM + HW observer @ 25Hz (2x servo tracking time)",
        },
        # --- CSV-warmup presets ---
        # Open-loop CSV playback walks forward; closed-loop doesn't.
        # Root cause: effort obs always saturated (Kp=70 × servo_tracking_error > 5.0).
        # Fix: zero effort obs + zero lin_vel (neither can be estimated on hardware).
        # hw_observer gives real joint_pos_rel, filtered joint_vel.
        # CSV warmup: play 3-4 loops of known-good actions to prime LSTM hidden state,
        # then hand off to live policy with primed observations.
        "lstm_25hz_crit": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_LSTM.pt",
            "use_synthetic_obs": True,
            "obs_mode": "hw_observer",
            "ema_alpha": 0.3,
            "HW_SCALE": 1.0,
            "ACTION_SCALE": 0.5,
            "pd_delay_steps": 2,
            "pd_substeps": 2,
            "pd_stiffness": 70.0,
            "pd_damping": 1.2,
            "pd_inertia": 2.0,
            "pd_friction": 0.03,
            "pd_effort_limit": 5.0,
            "pd_effort_clamp": 0.0,      # ZERO effort obs — can't estimate on HW
            "effort_mode": "synthetic",
            "use_comp_filter": True,
            "use_butterworth": True,
            "butterworth_cutoff": 8.0,
            "bias_scale": 0.0,
            "vel_alpha": 0.15,
            "lin_vel_mode": "zero",
            "control_frequency": 25,
            "description": "LSTM @ 25Hz + zero effort/lin_vel + HW observer",
        },
        "lstm_50hz_crit": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_LSTM.pt",
            "use_synthetic_obs": True,
            "obs_mode": "hw_observer",
            "ema_alpha": 0.3,
            "HW_SCALE": 1.5,
            "ACTION_SCALE": 0.5,
            "pd_delay_steps": 4,
            "pd_substeps": 4,
            "pd_stiffness": 70.0,
            "pd_damping": 1.2,
            "pd_inertia": 2.0,
            "pd_friction": 0.03,
            "pd_effort_limit": 5.0,
            "pd_effort_clamp": 0.0,
            "effort_mode": "synthetic",
            "use_comp_filter": True,
            "use_butterworth": True,
            "butterworth_cutoff": 8.0,
            "bias_scale": 0.0,
            "vel_alpha": 0.15,
            "lin_vel_mode": "zero",
            "control_frequency": 50,
            "description": "LSTM @ 50Hz + zero effort/lin_vel + HW observer",
        },
        "mlp_25hz_crit": {
            "policy": "/home/ubuntu/policy_joyboy_delayedpdactuator_hippy.pt",
            "use_synthetic_obs": True,
            "obs_mode": "hw_observer",
            "ema_alpha": 1.0,
            "HW_SCALE": 0.40,
            "ACTION_SCALE": 0.5,
            "pd_delay_steps": 2,
            "pd_substeps": 2,
            "pd_stiffness": 70.0,
            "pd_damping": 1.2,
            "pd_inertia": 2.0,
            "pd_friction": 0.03,
            "pd_effort_limit": 5.0,
            "pd_effort_clamp": 0.0,
            "effort_mode": "synthetic",
            "use_comp_filter": True,
            "use_butterworth": True,
            "butterworth_cutoff": 10.0,
            "bias_scale": 0.0,
            "vel_alpha": 0.15,
            "lin_vel_mode": "zero",
            "control_frequency": 25,
            "description": "MLP @ 25Hz + zero effort/lin_vel + HW observer",
        },
    }

    def __init__(self, policy_path=None, preset="mlp_pd"):
        print("=" * 70)
        print("FIXED MAPPING RL CONTROLLER v3")
        print("=" * 70)

        # ==================== HARDWARE ====================
        self.config = Configuration()
        self.hardware = HardwareInterface()
        self.pwm_params = self.hardware.pwm_params
        self.servo_params = self.hardware.servo_params
        self.esp32 = self.pwm_params.esp32
        time.sleep(0.3)

        # ==================== PRESET / POLICY ====================
        self.current_preset = preset
        cfg = self.PRESETS.get(preset, self.PRESETS["mlp_pd"])
        actual_policy = policy_path if policy_path else cfg["policy"]
        try:
            self.policy = torch.jit.load(actual_policy)
            self.policy.eval()
            print(f"[OK] Policy: {actual_policy}")
            print(f"[OK] Preset: {preset} — {cfg['description']}")
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
            +1.0, +1.0, +1.0,   # LF: HIP(C)/THIGH/CALF
            +1.0, +1.0, +1.0,   # RF: HIP(C)/THIGH/CALF
            +1.0, +1.0, +1.0,   # LB: HIP(C)/THIGH/CALF
            +1.0, +1.0, +1.0,   # RB: HIP(C)/THIGH/CALF
        ])

        # ==================== EFFORT DIRECTION MAPPING ====================
        self.effort_direction = np.array([
            +1.0, +1.0, +1.0,   # LF
            +1.0, +1.0, +1.0,   # RF
            +1.0, +1.0, +1.0,   # LB
            +1.0, +1.0, +1.0,   # RB
        ])

        # ==================== JOINT LIMITS (sim frame) ====================
        self.joint_lower_limits = np.array([-0.5, 0.0, -2.5] * 4)
        self.joint_upper_limits = np.array([0.5, 1.5, -0.5] * 4)

        # Hardware mapping
        self.hw_to_isaac_leg = {0: 1, 1: 0, 2: 3, 3: 2}

        # ==================== TUNING (from preset) ====================
        self.ACTION_SCALE = cfg.get("ACTION_SCALE", 0.5)
        self.HW_SCALE = cfg.get("HW_SCALE", 0.55)
        self.ema_alpha = cfg.get("ema_alpha", 1.0)
        self.effort_mode = cfg.get("effort_mode", "synthetic")
        self.obs_mode = cfg.get("obs_mode", "legacy")  # "legacy" or "hw_observer"
        self.action_rate_limit = 100.0     # Max change per step
        self.bias_scale = cfg.get("bias_scale", 1.0)   # 0.0 = disable action bias
        self.lin_vel_mode = cfg.get("lin_vel_mode", "filter")  # 'filter', 'zero', or 'command'

        # ==================== BUTTERWORTH ACTION FILTER ====================
        # (initialized after CONTROL_FREQUENCY is set, see below)
        self.use_butterworth = cfg.get("use_butterworth", False)
        self.butterworth_cutoff = cfg.get("butterworth_cutoff", 3.0)
        self.action_lpf = None

        # Velocity smoothing
        self.vel_filter_alpha = 0.2       # Joint velocity smoothing
        self.vel_histories = [deque(maxlen=5) for _ in range(12)]

        # ==================== BASE LINEAR VELOCITY ====================
        self.use_simple_lin_vel = True    # Use simpler command-based estimate
        self.use_comp_filter = cfg.get("use_comp_filter", False)

        self.estimated_lin_vel = np.zeros(3)
        self.lin_vel_decay = 0.9
        self.lin_vel_gain = 0.01

        # ==================== COMPLEMENTARY FILTER ====================
        # (initialized after CONTROL_FREQUENCY is set, see below)
        self.comp_filter = None

        # ==================== IMU ====================
        self.gyro_scale = np.pi / 180.0
        self.accel_scale = 9.81
        self.gyro_offset = np.zeros(3)
        self.accel_offset = np.zeros(3)
        self.gravity_ref = np.array([0.0, 0.0, -1.0])

        # Raw IMU storage for logging (populated each step)
        self._raw_imu = {'gx': 0, 'gy': 0, 'gz': 0, 'ax': 0, 'ay': 0, 'az': 0}

        # ==================== EFFORT ====================
        self.effort_scale = 5000.0
        self.effort_offset = np.zeros(12)

        # ==================== ACTION BIAS CORRECTION ====================
        # Measured from HW logs: policy outputs are biased vs sim, causing circle walking.
        # hw_bias = mean(hw_actions) - mean(sim_actions) per joint.
        # Subtract this from raw actions to center them where sim expects.
        # Set to zeros to disable. Tune with: set bias_scale 1.0
        self.action_bias = np.array([
            +0.13, -0.18, -0.13,   # LF: hip pulls outward on HW
            -0.57, -0.35, +0.17,   # RF: hip pulls inward on HW (main asymmetry)
            +0.08, -0.01, -0.17,   # LB
            -0.18, +0.10, +0.18,   # RB
        ])
        self.bias_scale = 1.0  # 0.0 = disabled, 1.0 = full correction

        # ==================== STATE ====================
        self.prev_actions = np.zeros(12)
        self.prev_smoothed_actions = np.zeros(12)
        self.prev_joint_angles_sim = self.sim_default_positions.copy()
        self.prev_joint_vel = np.zeros(12)
        self.prev_time = time.time()

        self.velocity_command = np.zeros(3)
        self.control_active = False
        self.shutdown = False
        self._last_hw_pos_sim = self.sim_default_positions.copy()
        self._last_dt = 1.0 / cfg.get("control_frequency", 50)
        self.startup_steps = 0
        self.debug_counter = 0

        self.CONTROL_FREQUENCY = cfg.get("control_frequency", 50)
        self.startup_duration = int(4.0 * self.CONTROL_FREQUENCY)  # ~4s warmup at any frequency

        # Finish Butterworth init (needs CONTROL_FREQUENCY)
        if self.use_butterworth:
            self.action_lpf = ButterworthLPF(
                cutoff_hz=self.butterworth_cutoff,
                fs=self.CONTROL_FREQUENCY,
                n_channels=12
            )
            print(f"[FILTER] Butterworth LPF: {self.butterworth_cutoff}Hz cutoff at {self.CONTROL_FREQUENCY}Hz")

        # Finish ComplementaryFilter init (needs CONTROL_FREQUENCY)
        if self.use_comp_filter and HAS_COMP_FILTER:
            self.comp_filter = ComplementaryFilter(dt=1.0/self.CONTROL_FREQUENCY)
            print(f"[AI-IMU] ComplementaryFilter enabled (dt={1.0/self.CONTROL_FREQUENCY:.3f}s)")
        elif self.use_comp_filter and not HAS_COMP_FILTER:
            print("[AI-IMU] WARNING: ComplementaryFilter not available, falling back to simple")
            self.use_comp_filter = False

        # ==================== SYNTHETIC PD DYNAMICS ====================
        # Simulates the DelayedPDActuator dynamics the MLP was trained with.
        # Hardware position servos don't produce PD oscillations, so these
        # synthetic observations provide the feedback loop the policy needs.
        self.use_synthetic_obs = cfg.get("use_synthetic_obs", True)
        self.pd_delay_steps = cfg.get("pd_delay_steps", 9)
        self.pd_stiffness = cfg.get("pd_stiffness", 70.0)
        self.pd_damping = cfg.get("pd_damping", 1.2)
        self.pd_inertia = cfg.get("pd_inertia", 0.20)
        self.pd_substeps = cfg.get("pd_substeps", 4)
        self.pd_friction = cfg.get("pd_friction", 0.03)
        self.pd_effort_limit = cfg.get("pd_effort_limit", 5.0)
        self.pd_effort_clamp = cfg.get("pd_effort_clamp", None)  # clamp normalized effort obs

        # PD simulation state
        self.pd_position = self.sim_default_positions.copy()
        self.pd_velocity = np.zeros(12)
        self.pd_action_buffer = deque(maxlen=self.pd_delay_steps + 1)
        for _ in range(self.pd_delay_steps + 1):
            self.pd_action_buffer.append(np.zeros(12))

        # Cached synthetic observations (read by get_observation, updated by _step_pd_dynamics)
        self.syn_pos_rel = np.zeros(12)
        self.syn_vel = np.zeros(12)
        self.syn_effort = np.zeros(12)

        # ==================== HW-DRIVEN PD OBSERVER STATE ====================
        # Instead of open-loop PD simulation, uses real joint positions and
        # computes PD-consistent velocity/effort from the actual trajectory.
        self.obs_prev_hw_pos = self.sim_default_positions.copy()
        self.obs_hw_vel = np.zeros(12)           # filtered velocity from hw positions
        self.obs_vel_alpha = cfg.get("vel_alpha", 0.3)  # EMA alpha for velocity filtering
        self.obs_hw_effort = np.zeros(12)         # PD effort from real position error

        # ==================== CSV LOGGING ====================
        self.log_enabled = False
        self.log_rows = []
        self.log_max_steps = 600  # 12 seconds at 50Hz

        # ==================== OBSERVATION HEALTH MONITORING ====================
        self.health_check_enabled = True

        # ==================== CALIBRATION ====================
        print("\n[CALIBRATING] Sensors...")
        self._calibrate_sensors()
        self._measure_stance_offset()
        self._print_config()
        print("[READY]")
        print("=" * 70)

    def load_preset(self, preset_name):
        """Switch to a different test preset (reloads policy if needed)."""
        if preset_name not in self.PRESETS:
            print(f"[ERROR] Unknown preset: {preset_name}")
            print(f"  Available: {', '.join(self.PRESETS.keys())}")
            return

        was_active = self.control_active
        self.control_active = False
        time.sleep(0.1)

        cfg = self.PRESETS[preset_name]
        self.current_preset = preset_name

        # Reload policy if it changed
        new_policy_path = cfg["policy"]
        try:
            self.policy = torch.jit.load(new_policy_path)
            self.policy.eval()
            print(f"[OK] Policy: {new_policy_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load policy: {e}")
            return

        # Apply all preset params
        self.CONTROL_FREQUENCY = cfg.get("control_frequency", 50)
        self.ACTION_SCALE = cfg.get("ACTION_SCALE", 0.5)
        self.HW_SCALE = cfg.get("HW_SCALE", 0.55)
        self.ema_alpha = cfg.get("ema_alpha", 1.0)
        self.effort_mode = cfg.get("effort_mode", "synthetic")
        self.use_synthetic_obs = cfg.get("use_synthetic_obs", True)
        self.obs_mode = cfg.get("obs_mode", "legacy")
        self.pd_delay_steps = cfg.get("pd_delay_steps", 9)
        self.pd_substeps = cfg.get("pd_substeps", 4)
        self.pd_stiffness = cfg.get("pd_stiffness", 70.0)
        self.pd_damping = cfg.get("pd_damping", 1.2)
        self.pd_inertia = cfg.get("pd_inertia", 0.20)
        self.pd_friction = cfg.get("pd_friction", 0.03)
        self.pd_effort_limit = cfg.get("pd_effort_limit", 5.0)
        self.pd_effort_clamp = cfg.get("pd_effort_clamp", None)
        self.bias_scale = cfg.get("bias_scale", 1.0)

        # Butterworth filter
        self.use_butterworth = cfg.get("use_butterworth", False)
        self.butterworth_cutoff = cfg.get("butterworth_cutoff", 3.0)
        if self.use_butterworth:
            self.action_lpf = ButterworthLPF(
                cutoff_hz=self.butterworth_cutoff,
                fs=self.CONTROL_FREQUENCY,
                n_channels=12
            )
            print(f"[FILTER] Butterworth LPF: {self.butterworth_cutoff}Hz cutoff")
        else:
            self.action_lpf = None

        # HW observer velocity smoothing
        self.obs_vel_alpha = cfg.get("vel_alpha", 0.3)

        # ComplementaryFilter — always create fresh to ensure calibration
        self.use_comp_filter = cfg.get("use_comp_filter", False)
        if self.use_comp_filter and HAS_COMP_FILTER:
            self.comp_filter = ComplementaryFilter(dt=1.0/self.CONTROL_FREQUENCY)
            self._calibrate_sensors()
            print("[AI-IMU] ComplementaryFilter enabled (freshly calibrated)")
        else:
            self.comp_filter = None

        # Reset state
        self.prev_actions = np.zeros(12)
        self.prev_smoothed_actions = np.zeros(12)
        self.startup_steps = 0
        self.startup_duration = int(4.0 * self.CONTROL_FREQUENCY)
        self.pd_action_buffer = deque(maxlen=self.pd_delay_steps + 1)
        self._reset_pd_dynamics()
        self._reset_hw_observer()

        print(f"[PRESET] {preset_name} — {cfg['description']}")
        self._print_config()

    def _print_config(self):
        """Print current configuration."""
        print(f"\n[CONFIG] Preset: {self.current_preset}")
        print(f"  CONTROL_FREQ:      {self.CONTROL_FREQUENCY}Hz ({1000/self.CONTROL_FREQUENCY:.0f}ms per step)")
        print(f"  ACTION_SCALE:      {self.ACTION_SCALE} (synthetic PD)")
        print(f"  HW_SCALE:          {self.HW_SCALE} (servo output)")
        print(f"  ema_alpha:         {self.ema_alpha}{' (bypassed by Butterworth)' if self.action_lpf else ''}")
        print(f"  butterworth:       {self.butterworth_cutoff}Hz" if self.action_lpf else "  butterworth:       off")
        print(f"  action_rate_limit: {self.action_rate_limit}")
        print(f"  bias_scale:        {self.bias_scale}")
        print(f"  startup_duration:  {self.startup_duration} steps")
        print(f"  use_synthetic_obs: {self.use_synthetic_obs}")
        print(f"  obs_mode:          {self.obs_mode}")
        print(f"  effort_mode:       {self.effort_mode}")
        print(f"  comp_filter:       {self.comp_filter is not None}")
        print(f"  lin_vel_mode:      {self.lin_vel_mode}")
        if self.use_synthetic_obs:
            wn = np.sqrt(self.pd_stiffness / self.pd_inertia)
            zeta = self.pd_damping / (2.0 * np.sqrt(self.pd_stiffness * self.pd_inertia))
            print(f"    pd_delay:  {self.pd_delay_steps} steps ({self.pd_delay_steps * 1000 / self.CONTROL_FREQUENCY:.0f}ms)")
            print(f"    pd_Kp:     {self.pd_stiffness}  Kd: {self.pd_damping}  I: {self.pd_inertia}")
            print(f"    pd_ωn:     {wn:.1f} rad/s ({wn/(2*np.pi):.2f}Hz)  ζ: {zeta:.2f} ({'crit' if abs(zeta-1.0)<0.05 else 'under' if zeta<1.0 else 'over'}-damped)")

    def _calibrate_sensors(self, samples=50):
        """Calibrate IMU and effort offsets."""
        gyro_samples, accel_samples, effort_samples = [], [], []

        for _ in range(samples):
            imu = self.esp32.imu_get_data()
            if imu:
                gyro_samples.append([imu['gx'], imu['gy'], imu['gz']])
                accel_samples.append([imu['ax'], imu['ay'], imu['az']])
                # Feed raw samples to comp filter for calibration
                if self.comp_filter is not None:
                    gyro_si = np.array([imu['gx'], imu['gy'], imu['gz']]) * self.gyro_scale
                    accel_si = np.array([imu['ax'], imu['ay'], imu['az']]) * self.accel_scale
                    self.comp_filter.calibrate(gyro_si, accel_si)

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

        # Finalize comp filter calibration
        if self.comp_filter is not None:
            self.comp_filter.finish_calibration()
            print(f"[AI-IMU] Calibrated: gyro_bias={self.comp_filter.gyro_bias}, gravity={self.comp_filter.gravity_est}")

    def _measure_stance_offset(self, samples=10):
        """Measure and report calibration offset between real stance and sim defaults.

        Reads servo positions at startup (before any RL control) and computes
        the offset from sim_default_positions. Large offsets indicate either
        servo miscalibration or that the robot is not in the expected stance.
        """
        readings = []
        for _ in range(samples):
            angles_real = self._read_joint_positions_raw()
            if angles_real is not None:
                angles_sim = self._real_to_sim_positions(angles_real)
                offset = angles_sim - self.sim_default_positions
                readings.append(offset)
            time.sleep(0.02)

        if not readings:
            print("[STANCE] WARNING: Could not read servo positions")
            return

        mean_offset = np.mean(readings, axis=0)
        names = ['LF_hip', 'LF_thigh', 'LF_calf',
                 'RF_hip', 'RF_thigh', 'RF_calf',
                 'LB_hip', 'LB_thigh', 'LB_calf',
                 'RB_hip', 'RB_thigh', 'RB_calf']

        large_offsets = []
        print("\n[STANCE] Calibration offset (real vs sim default):")
        for i, name in enumerate(names):
            flag = " *** CHECK" if abs(mean_offset[i]) > 0.1 else ""
            print(f"  {name:<12}: {mean_offset[i]:+.4f} rad ({np.degrees(mean_offset[i]):+.1f}°){flag}")
            if abs(mean_offset[i]) > 0.1:
                large_offsets.append((name, mean_offset[i]))

        if large_offsets:
            print(f"\n[STANCE] WARNING: {len(large_offsets)} joints have >0.1 rad offset!")
            print("  This may cause OOD observations. Check servo calibration")
            print("  or adjust real_default_positions to match actual neutral.")

        self._stance_offset = mean_offset

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
            lin_vel = np.array([
                self.velocity_command[0] * 0.7,
                self.velocity_command[1] * 0.7,
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
                self.velocity_command[0] * 0.7,
                self.velocity_command[1] * 0.7,
                0.0
            ])
            blended = 0.5 * self.estimated_lin_vel + 0.5 * cmd_estimate

            return np.clip(blended, -0.5, 0.5)

    # ==============================================================
    # SYNTHETIC PD DYNAMICS
    # ==============================================================

    def _step_pd_dynamics(self, current_action):
        """Advance synthetic PD dynamics by one policy step.

        Simulates: action → delay buffer → PD torque → mass-spring-damper integration.
        Called AFTER the policy produces an action each step.
        """
        # Push current action into delay buffer
        self.pd_action_buffer.append(current_action.copy())

        # Get delayed action (from pd_delay_steps ago)
        delayed_action = self.pd_action_buffer[0]

        # Target position from delayed action (same formula as sim)
        target = self.sim_default_positions + delayed_action * self.ACTION_SCALE
        target = np.clip(target, self.joint_lower_limits, self.joint_upper_limits)

        dt_sub = (1.0 / self.CONTROL_FREQUENCY) / self.pd_substeps

        for _ in range(self.pd_substeps):
            # PD torque: τ = Kp*(target - q) - Kd*q̇
            error = target - self.pd_position
            torque = self.pd_stiffness * error - self.pd_damping * self.pd_velocity

            # Coulomb friction
            torque -= self.pd_friction * np.sign(self.pd_velocity)

            # Acceleration: a = τ / I
            accel = torque / self.pd_inertia

            # Semi-implicit Euler integration
            self.pd_velocity += accel * dt_sub
            self.pd_position += self.pd_velocity * dt_sub

        # Enforce joint limits (bounce off limits)
        for j in range(12):
            if self.pd_position[j] < self.joint_lower_limits[j]:
                self.pd_position[j] = self.joint_lower_limits[j]
                self.pd_velocity[j] = 0.0
            elif self.pd_position[j] > self.joint_upper_limits[j]:
                self.pd_position[j] = self.joint_upper_limits[j]
                self.pd_velocity[j] = 0.0

        # Cache synthetic observations for get_observation() to use
        self.syn_pos_rel = self.pd_position - self.sim_default_positions
        self.syn_vel = self.pd_velocity.copy()

        # Synthetic effort = current PD torque, normalized by effort limit
        final_error = target - self.pd_position
        final_torque = self.pd_stiffness * final_error - self.pd_damping * self.pd_velocity
        self.syn_effort = np.clip(final_torque / self.pd_effort_limit, -1.0, 1.0)
        if self.pd_effort_clamp is not None:
            self.syn_effort = np.clip(self.syn_effort, -self.pd_effort_clamp, self.pd_effort_clamp)

    def _reset_pd_dynamics(self):
        """Reset PD simulation to default stance."""
        self.pd_position = self.sim_default_positions.copy()
        self.pd_velocity = np.zeros(12)
        self.pd_action_buffer.clear()
        for _ in range(self.pd_delay_steps + 1):
            self.pd_action_buffer.append(np.zeros(12))
        self.syn_pos_rel = np.zeros(12)
        self.syn_vel = np.zeros(12)
        self.syn_effort = np.zeros(12)

    # ==============================================================
    # HARDWARE-DRIVEN PD OBSERVER
    # ==============================================================

    def _step_hw_observer(self, current_action, current_hw_pos_sim, dt):
        """Compute PD-consistent velocity and effort from real hardware positions.

        Instead of running a standalone PD ODE (open-loop), this uses:
          - Real joint positions from hardware as ground truth
          - Delayed action target from the action buffer (same delay model)
          - PD effort = Kp*(delayed_target - hw_pos) - Kd*hw_vel
          - Velocity from filtered differentiation of real positions

        This closes the loop: the policy sees observations that reflect
        what the robot actually did, with PD-consistent effort signals.
        """
        # Push current action into delay buffer (same as open-loop PD)
        self.pd_action_buffer.append(current_action.copy())

        # Get delayed action target
        delayed_action = self.pd_action_buffer[0]
        target = self.sim_default_positions + delayed_action * self.ACTION_SCALE
        target = np.clip(target, self.joint_lower_limits, self.joint_upper_limits)

        # Compute velocity from real position change (filtered)
        if dt > 0.001:
            raw_vel = (current_hw_pos_sim - self.obs_prev_hw_pos) / dt
            raw_vel = np.clip(raw_vel, -10.5, 10.5)
            # EMA filter to smooth quantization noise from position servos
            self.obs_hw_vel = (self.obs_vel_alpha * raw_vel +
                               (1 - self.obs_vel_alpha) * self.obs_hw_vel)

        # PD effort from real position error (what the PD controller would command)
        error = target - current_hw_pos_sim
        torque = self.pd_stiffness * error - self.pd_damping * self.obs_hw_vel
        torque -= self.pd_friction * np.sign(self.obs_hw_vel)
        self.obs_hw_effort = np.clip(torque / self.pd_effort_limit, -1.0, 1.0)
        if self.pd_effort_clamp is not None:
            self.obs_hw_effort = np.clip(self.obs_hw_effort, -self.pd_effort_clamp, self.pd_effort_clamp)

        # Cache for get_observation()
        self.syn_pos_rel = current_hw_pos_sim - self.sim_default_positions
        self.syn_vel = self.obs_hw_vel.copy()
        self.syn_effort = self.obs_hw_effort.copy()

        # Update previous position
        self.obs_prev_hw_pos = current_hw_pos_sim.copy()

    def _reset_hw_observer(self):
        """Reset HW observer state."""
        self.obs_prev_hw_pos = self.sim_default_positions.copy()
        self.obs_hw_vel = np.zeros(12)
        self.obs_hw_effort = np.zeros(12)

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

        # Handle ESP32 communication failure (Invalid Ack → None)
        if imu is None:
            imu = {'gx': 0, 'gy': 0, 'gz': 0, 'ax': 0, 'ay': 0, 'az': -1.0/self.accel_scale}

        # Store for hw_observer (control_step reads this after get_observation)
        self._last_hw_pos_sim = current_angles_sim.copy()
        self._last_dt = dt

        # Raw IMU in SI units (for logging + comp filter)
        gyro_raw = np.array([imu['gx'], imu['gy'], imu['gz']])
        accel_raw = np.array([imu['ax'], imu['ay'], imu['az']])
        gyro_si = gyro_raw * self.gyro_scale     # rad/s
        accel_si = accel_raw * self.accel_scale   # m/s^2

        # Store raw IMU for CSV logging
        self._raw_imu = {
            'gx': gyro_si[0], 'gy': gyro_si[1], 'gz': gyro_si[2],
            'ax': accel_si[0], 'ay': accel_si[1], 'az': accel_si[2],
        }

        if self.comp_filter is not None:
            # AI-IMU: ComplementaryFilter produces real estimates
            cf_vel, cf_ang_vel, cf_gravity = self.comp_filter.update(gyro_si, accel_si, dt if dt > 0.001 else self.comp_filter.dt)
            base_lin_vel = cf_vel
            base_ang_vel = cf_ang_vel
            projected_gravity = cf_gravity
        else:
            # Legacy: fake base_lin_vel + raw gyro + raw gravity
            accel = accel_raw * self.accel_scale
            base_lin_vel = self._estimate_base_lin_vel(accel, dt)
            base_ang_vel = (gyro_raw - self.gyro_offset) * self.gyro_scale
            accel_norm = np.linalg.norm(accel)
            if accel_norm > 0.1:
                projected_gravity = accel / accel_norm
            else:
                projected_gravity = np.array([0.0, 0.0, -1.0])

        velocity_commands = self.velocity_command.copy()

        # Joint observations: synthetic PD or hardware
        if self.use_synthetic_obs:
            joint_pos_rel = self.syn_pos_rel.copy()
            joint_vel = self.syn_vel.copy()
            joint_effort = self.syn_effort.copy()
        else:
            joint_pos_rel = current_angles_sim - self.sim_default_positions
            joint_vel = self._compute_smoothed_velocity(current_angles_sim, dt)

        # Effort mode (only matters when use_synthetic_obs=False)
        if not self.use_synthetic_obs:
            if self.effort_mode == "zero":
                joint_effort = np.zeros(12)
            elif self.effort_mode == "boost5x":
                joint_effort = joint_effort * 5.0
            # else "real" — use raw servo load as-is

        # Update state (always track hardware for other uses)
        self.prev_joint_angles_sim = current_angles_sim.copy()
        if not self.use_synthetic_obs:
            self.prev_joint_vel = joint_vel.copy()
        self.prev_time = current_time

        prev_actions = self.prev_actions.copy()

        # Base linear velocity mode
        lin_vel_mode = getattr(self, 'lin_vel_mode', 'filter')
        if lin_vel_mode == 'zero':
            base_lin_vel = np.zeros(3)
        elif lin_vel_mode == 'command':
            # Proxy: assume robot tracks ~70% of commanded velocity
            base_lin_vel = self.velocity_command * 0.7
        # else 'filter': use comp filter / estimator output

        # Clamp observations to simulation ranges
        base_lin_vel = np.clip(base_lin_vel, -0.5, 0.5)
        base_ang_vel = np.clip(base_ang_vel, -2.0, 2.0)
        projected_gravity = np.clip(projected_gravity, -1.0, 1.0)
        joint_pos_rel = np.clip(joint_pos_rel, -0.9, 0.9)
        joint_vel = np.clip(joint_vel, -10.0, 10.0)
        joint_effort = np.clip(joint_effort, -1.0, 1.0)

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

        # Check angular velocity (sim range: -2.0 to +2.0)
        max_ang_vel = np.max(np.abs(base_ang_vel))
        if max_ang_vel > 1.5:
            warnings.append(f"HIGH_ANG_VEL:{max_ang_vel:.2f}")

        # Check gravity tilt (Z should be < -0.95)
        if gravity[2] > -0.95:
            warnings.append(f"TILTED:gz={gravity[2]:.2f}")

        # Check joint velocities (sim range: -1.5 to +1.5 typically)
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

        obs = self.get_observation()

        if self.comp_filter is None:
            obs[6:9] = [0.0, 0.0, -1.0]  # gravity override for noisy raw IMU
        # When comp_filter is active, gravity comes from the filter (no override)
        # obs[3:6] = 0.0 basically still
        # obs[24:36] = 0.0 jittery
        # obs[36:48] = 0.0 jittery

        # Check observation health
        self.check_observation_health(obs)

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            raw_actions = self.policy(obs_tensor).squeeze().numpy()

        # Clip actions
        clipped_actions = np.clip(raw_actions, -1.0, 1.0)

        # Action bias correction (center HW actions where sim expects them)
        if self.bias_scale > 0:
            clipped_actions = clipped_actions - self.action_bias * self.bias_scale
            clipped_actions = np.clip(clipped_actions, -1.0, 1.0)

        # Startup fade-in
        if self.startup_steps < self.startup_duration:
            fade = self.startup_steps / self.startup_duration
            self.startup_steps += 1
        else:
            fade = 1.0

        faded_actions = clipped_actions * fade

        # === Action Smoothing ===
        if self.action_lpf is not None:
            # Butterworth LPF: flat passband at 1-3Hz, sharp rolloff above cutoff
            smoothed_actions = self.action_lpf.filter(faded_actions)
        else:
            # Legacy EMA: lower ema_alpha = more smoothing
            smoothed_actions = (self.ema_alpha * faded_actions +
                               (1 - self.ema_alpha) * self.prev_smoothed_actions)

        # === Rate Limiting ===
        # Limit how fast the smoothed actions can change
        action_delta = smoothed_actions - self.prev_smoothed_actions
        action_delta = np.clip(action_delta, -self.action_rate_limit, self.action_rate_limit)
        final_actions = self.prev_smoothed_actions + action_delta

        # Store for next iteration
        self.prev_actions = clipped_actions.copy()  # Store raw for observation
        self.prev_smoothed_actions = final_actions.copy()

        # Step PD dynamics: hw_observer uses real positions, legacy uses open-loop sim
        if self.use_synthetic_obs:
            if self.obs_mode == "hw_observer":
                self._step_hw_observer(clipped_actions, self._last_hw_pos_sim, self._last_dt)
            else:
                self._step_pd_dynamics(clipped_actions)

        # Compute target (sim frame) — per-joint HW_SCALE amplifies physical movements
        target_sim = self.sim_default_positions + final_actions * self.HW_SCALE
        target_sim = np.clip(target_sim, self.joint_lower_limits, self.joint_upper_limits)

        # Send to hardware
        target_matrix = self._isaac_to_hardware_matrix(target_sim)
        self.hardware.set_actuator_postions(target_matrix)

        # CSV logging (includes raw IMU for MesNet training)
        if self.log_enabled and len(self.log_rows) < self.log_max_steps:
            row = {
                'step': self.debug_counter,
                'time': time.time(),
                'cmd_x': self.velocity_command[0],
                'cmd_y': self.velocity_command[1],
                'cmd_yaw': self.velocity_command[2],
                'fade': fade,
                # Raw IMU in SI units (rad/s, m/s^2) for MesNet training
                'imu_gx': self._raw_imu['gx'],
                'imu_gy': self._raw_imu['gy'],
                'imu_gz': self._raw_imu['gz'],
                'imu_ax': self._raw_imu['ax'],
                'imu_ay': self._raw_imu['ay'],
                'imu_az': self._raw_imu['az'],
            }
            for i in range(60):
                row[f'obs_{i}'] = float(obs[i])
            for i in range(12):
                row[f'raw_action_{i}'] = float(raw_actions[i])
                row[f'final_action_{i}'] = float(final_actions[i])
            self.log_rows.append(row)
            if len(self.log_rows) == self.log_max_steps:
                self._save_log()

        # Enhanced debug output
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

    def _save_log(self):
        """Save collected log rows to CSV."""
        import csv
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"/home/ubuntu/mp2_mlp/{self.current_preset}_hw_log_{timestamp}.csv"
        if not self.log_rows:
            print("[LOG] No data to save.")
            return
        fieldnames = list(self.log_rows[0].keys())
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.log_rows)
        print(f"[LOG] Saved {len(self.log_rows)} rows to {filename}")
        self.log_enabled = False

    def goto_stance(self, duration=3.0, tolerance=0.05):
        """Move robot to sim default stance and verify position before RL control.

        Call this before giving a walking command. Smoothly interpolates
        from current position to sim default over `duration` seconds,
        then measures and reports the final calibration offset.

        Usage: controller.goto_stance()  # then controller.set_velocity_command(...)
        """
        print(f"\n[STANCE] Moving to sim default stance over {duration}s...")

        # Read current position
        current_real = self._read_joint_positions_raw()
        if current_real is None:
            print("[STANCE] ERROR: Cannot read servo positions")
            return False

        target_real = self.real_default_positions.copy()
        dt = 1.0 / self.CONTROL_FREQUENCY
        n_steps = int(duration * self.CONTROL_FREQUENCY)

        for step in range(n_steps):
            alpha = min(1.0, (step + 1) / n_steps)
            # Smooth cosine interpolation (less jerky than linear)
            alpha_smooth = 0.5 * (1.0 - np.cos(np.pi * alpha))
            interp = current_real * (1 - alpha_smooth) + target_real * alpha_smooth

            # Convert to hardware matrix and send
            # Build matrix directly from real-frame angles
            matrix = np.zeros((3, 4))
            matrix[:, 1] = interp[0:3]    # LF (isaac leg 0 → hw col 1)
            matrix[:, 0] = interp[3:6]    # RF (isaac leg 1 → hw col 0)
            matrix[:, 3] = interp[6:9]    # LB (isaac leg 2 → hw col 3)
            matrix[:, 2] = interp[9:12]   # RB (isaac leg 3 → hw col 2)

            self.hardware.set_actuator_postions(matrix)
            time.sleep(dt)

        # Hold for a moment to let servos settle
        time.sleep(0.5)

        # Measure final position
        names = ['LF_hip', 'LF_thigh', 'LF_calf',
                 'RF_hip', 'RF_thigh', 'RF_calf',
                 'LB_hip', 'LB_thigh', 'LB_calf',
                 'RB_hip', 'RB_thigh', 'RB_calf']

        readings = []
        for _ in range(20):
            angles_real = self._read_joint_positions_raw()
            if angles_real is not None:
                angles_sim = self._real_to_sim_positions(angles_real)
                readings.append(angles_sim - self.sim_default_positions)
            time.sleep(0.02)

        if readings:
            offset = np.mean(readings, axis=0)
            max_err = np.max(np.abs(offset))
            print(f"\n[STANCE] Final offset from sim default (max: {max_err:.4f} rad):")
            for i, name in enumerate(names):
                flag = " *** LARGE" if abs(offset[i]) > tolerance else ""
                print(f"  {name:<12}: {offset[i]:+.4f} rad ({np.degrees(offset[i]):+.1f}°){flag}")

            if max_err < tolerance:
                print(f"\n[STANCE] OK — all joints within {tolerance} rad of sim default")
            else:
                print(f"\n[STANCE] WARNING — some joints are off. Consider adjusting real_default_positions")
                print(f"  To auto-correct, run: controller.auto_calibrate_stance()")

            self._stance_offset = offset
        else:
            print("[STANCE] WARNING: Could not verify final position")

        # Reset all state for clean RL start
        self.prev_actions = np.zeros(12)
        self.prev_smoothed_actions = np.zeros(12)
        if self.action_lpf is not None:
            self.action_lpf.reset()
        self._reset_pd_dynamics()
        self._reset_hw_observer()
        self.startup_steps = 0
        self.prev_joint_angles_sim = self.sim_default_positions.copy()
        self._last_hw_pos_sim = self.sim_default_positions.copy()

        print("[STANCE] Ready for walking command.")
        return True

    def auto_calibrate_stance(self):
        """Measure actual servo positions and update real_default_positions to match.

        Run this with the robot in the physical pose that corresponds to
        sim_default_positions. Adjusts real_default_positions so that
        joint_pos_rel reads ~0 when the robot is in this stance.
        """
        print("[CAL] Reading current servo positions (hold robot in desired stance)...")
        readings = []
        for _ in range(50):
            angles_real = self._read_joint_positions_raw()
            if angles_real is not None:
                readings.append(angles_real)
            time.sleep(0.02)

        if not readings:
            print("[CAL] ERROR: No servo readings")
            return

        measured_real = np.mean(readings, axis=0)
        old_defaults = self.real_default_positions.copy()
        names = ['LF_hip', 'LF_thigh', 'LF_calf',
                 'RF_hip', 'RF_thigh', 'RF_calf',
                 'LB_hip', 'LB_thigh', 'LB_calf',
                 'RB_hip', 'RB_thigh', 'RB_calf']

        print(f"\n[CAL] Measured real positions vs current defaults:")
        for i, name in enumerate(names):
            delta = measured_real[i] - old_defaults[i]
            print(f"  {name:<12}: measured={measured_real[i]:+.4f}  default={old_defaults[i]:+.4f}  delta={delta:+.4f}")

        self.real_default_positions = measured_real.copy()
        print(f"\n[CAL] Updated real_default_positions. Verify with goto_stance().")
        print(f"[CAL] To make permanent, update the array in __init__:")
        print(f"  self.real_default_positions = np.array([")
        for leg in range(4):
            vals = measured_real[leg*3:(leg+1)*3]
            leg_name = ['LF', 'RF', 'LB', 'RB'][leg]
            print(f"      {vals[0]:+.4f}, {vals[1]:+.4f}, {vals[2]:+.4f},   # {leg_name}")
        print(f"  ])")

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
                if self.action_lpf is not None:
                    self.action_lpf.reset()
                if self.use_synthetic_obs:
                    self._reset_pd_dynamics()
                    self._reset_hw_observer()
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
        elif param == 'hw_scale':
            self.HW_SCALE = float(value)
            print(f"[PARAM] HW_SCALE = {self.HW_SCALE}")
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
        elif param == 'synthetic':
            self.use_synthetic_obs = value.lower() in ['true', '1', 'yes']
            if self.use_synthetic_obs:
                self._reset_pd_dynamics()
            print(f"[PARAM] use_synthetic_obs = {self.use_synthetic_obs}")
        elif param == 'pd_inertia':
            self.pd_inertia = float(value)
            print(f"[PARAM] pd_inertia = {self.pd_inertia}")
        elif param == 'pd_damping':
            self.pd_damping = float(value)
            print(f"[PARAM] pd_damping = {self.pd_damping}")
        elif param == 'pd_delay':
            self.pd_delay_steps = int(value)
            self.pd_action_buffer = deque(maxlen=self.pd_delay_steps + 1)
            self._reset_pd_dynamics()
            print(f"[PARAM] pd_delay_steps = {self.pd_delay_steps}")
        elif param == 'bias_scale':
            self.bias_scale = float(value)
            print(f"[PARAM] bias_scale = {self.bias_scale} (0=off, 1=full correction)")
        elif param == 'obs_mode':
            if value in ('legacy', 'hw_observer'):
                self.obs_mode = value
                if value == 'hw_observer':
                    self._reset_hw_observer()
                    self.use_synthetic_obs = True
                print(f"[PARAM] obs_mode = {self.obs_mode}")
            else:
                print(f"[ERROR] obs_mode must be 'legacy' or 'hw_observer'")
        elif param == 'vel_alpha':
            self.obs_vel_alpha = float(value)
            print(f"[PARAM] obs_vel_alpha = {self.obs_vel_alpha}")
        elif param == 'lin_vel':
            if value in ('filter', 'zero', 'command'):
                self.lin_vel_mode = value
                print(f"[PARAM] lin_vel_mode = {self.lin_vel_mode}")
            else:
                print(f"[ERROR] lin_vel must be 'filter', 'zero', or 'command'")
        elif param == 'butterworth':
            cutoff = float(value)
            if cutoff <= 0:
                self.action_lpf = None
                self.use_butterworth = False
                print(f"[PARAM] Butterworth disabled, using EMA")
            else:
                self.butterworth_cutoff = cutoff
                self.action_lpf = ButterworthLPF(
                    cutoff_hz=cutoff, fs=self.CONTROL_FREQUENCY, n_channels=12)
                self.use_butterworth = True
                print(f"[PARAM] Butterworth cutoff = {cutoff}Hz")
        else:
            print(f"Unknown param: {param}")
            print("Available: scale, hw_scale, ema, rate, simple, health, synthetic, pd_inertia, pd_damping, pd_delay, bias_scale, obs_mode, vel_alpha, butterworth")

    # ==============================================================
    # TESTING & VALIDATION
    # ==============================================================

    def test_with_sim_actions(self, csv_path="/home/ubuntu/env_2_actions.csv",
                              max_steps=150, playback_hz=None, loops=1,
                              hw_scale=None, log=True, warmup_skip=30):
        """Replay raw actions from simulation CSV — open-loop, no policy.

        Args:
            csv_path: Path to actions CSV (12 columns, one row per sim step)
            max_steps: Max steps per loop (0 = all)
            playback_hz: Control rate (default: self.CONTROL_FREQUENCY)
            loops: Number of times to loop the trajectory
            hw_scale: Override HW_SCALE for this playback (default: self.HW_SCALE)
            log: If True, log commanded + real joint positions to CSV
            warmup_skip: Skip first N steps on loops 2+ (avoids backward startup transient)
        """
        try:
            import pandas as pd
            sim_actions = pd.read_csv(csv_path)
            print(f"\n[PLAY] Replaying actions from {csv_path}")
            print(f"[PLAY] CSV has {len(sim_actions)} steps")
        except Exception as e:
            print(f"[ERROR] Could not load CSV: {e}")
            return

        hz = playback_hz or self.CONTROL_FREQUENCY
        scale = hw_scale if hw_scale is not None else self.HW_SCALE
        dt = 1.0 / hz
        n_steps = min(max_steps, len(sim_actions)) if max_steps > 0 else len(sim_actions)

        print(f"[PLAY] Rate: {hz}Hz ({dt*1000:.0f}ms/step), HW_SCALE: {scale}")
        print(f"[PLAY] Steps: {n_steps}, Loops: {loops}, Warmup skip: {warmup_skip} (loops 2+)")
        print(f"[PLAY] Logging: {log}")
        print(f"[PLAY] Starting in 3 seconds...")
        time.sleep(3)

        # Drop non-action columns (e.g. time_step, index)
        action_cols = [c for c in sim_actions.columns if c.startswith('action_')]
        if len(action_cols) == 12:
            action_data = sim_actions[action_cols].iloc[:n_steps].values
        else:
            action_data = sim_actions.iloc[:n_steps, -12:].values
        print(f"[PLAY] Using {action_data.shape[1]} action columns")

        joint_names = ['LF_hip', 'LF_thigh', 'LF_calf',
                       'RF_hip', 'RF_thigh', 'RF_calf',
                       'LB_hip', 'LB_thigh', 'LB_calf',
                       'RB_hip', 'RB_thigh', 'RB_calf']

        log_rows = []
        global_step = 0
        t0 = time.time()

        for loop_i in range(loops):
            # Skip startup transient on subsequent loops
            start_idx = warmup_skip if loop_i > 0 else 0
            if loops > 1:
                print(f"  Loop {loop_i + 1}/{loops} (from step {start_idx})")

            for i in range(start_idx, n_steps):
                step_start = time.time()
                actions = action_data[i]

                target_sim = self.sim_default_positions + actions * scale
                target_sim = np.clip(target_sim, self.joint_lower_limits, self.joint_upper_limits)

                target_matrix = self._isaac_to_hardware_matrix(target_sim)
                self.hardware.set_actuator_postions(target_matrix)

                # Read back actual joint positions for logging
                if log:
                    real_pos = self._read_joint_positions_sim_frame()
                    # Read IMU if available
                    imu_data = self._read_imu() if hasattr(self, '_read_imu') else None

                    row = {
                        'global_step': global_step,
                        'loop': loop_i,
                        'csv_step': i,
                        'time': time.time() - t0,
                        'playback_hz': hz,
                        'hw_scale': scale,
                    }
                    for j in range(12):
                        row[f'sim_action_{j}'] = float(actions[j])
                        row[f'cmd_pos_{j}'] = float(target_sim[j])
                        if real_pos is not None:
                            row[f'real_pos_{j}'] = float(real_pos[j])
                    log_rows.append(row)

                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

                if i % max(1, int(hz)) == 0:
                    tracking = ""
                    if log and real_pos is not None:
                        err = np.abs(target_sim - real_pos)
                        tracking = f"  track_err={err.mean():.4f}"
                    print(f"  Step {global_step} (csv:{i}): "
                          f"actions=[{actions.min():+.3f}, {actions.max():+.3f}]{tracking}")

                global_step += 1

        # Save log
        if log and log_rows:
            import csv
            source = csv_path.split('/')[-1].replace('.csv', '')
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_file = f"/home/ubuntu/mp2_mlp/playback_{source}_{hz}hz_{timestamp}.csv"
            fieldnames = list(log_rows[0].keys())
            with open(log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(log_rows)
            print(f"[LOG] Saved {len(log_rows)} rows to {log_file}")

        print("[PLAY] Replay complete.")
        print("[PLAY] Returning to stance in 2 seconds...")
        time.sleep(2)
        target_matrix = self._isaac_to_hardware_matrix(self.sim_default_positions)
        self.hardware.set_actuator_postions(target_matrix)

    def csv_warmup_then_live(self, csv_path="/home/ubuntu/env_2_actions_lstm.csv",
                              warmup_loops=3, warmup_skip=30, live_duration=10.0):
        """Prime LSTM hidden state with known-good CSV actions, then hand off to live policy.

        Phase 1 (warmup): Play CSV actions open-loop while feeding observations through
        the policy network (but IGNORING its output). This lets the LSTM build up
        internal state from real observations + known-good actions as prev_actions.

        Phase 2 (live): Seamlessly switch to using the policy's own output.
        The LSTM hidden state carries over — it should continue the gait pattern.

        Args:
            csv_path: Path to known-good sim actions CSV
            warmup_loops: Number of CSV replay loops for priming
            warmup_skip: Skip first N steps on loops 2+
            live_duration: Seconds to run live policy after warmup
        """
        try:
            import pandas as pd
            sim_actions_df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[ERROR] Could not load CSV: {e}")
            return

        action_cols = [c for c in sim_actions_df.columns if c.startswith('action_')]
        if len(action_cols) == 12:
            action_data = sim_actions_df[action_cols].values
        else:
            action_data = sim_actions_df.iloc[:, -12:].values
        n_csv_steps = len(action_data)

        hz = self.CONTROL_FREQUENCY
        dt = 1.0 / hz
        scale = self.HW_SCALE

        print(f"\n[WARMUP] CSV: {csv_path} ({n_csv_steps} steps)")
        print(f"[WARMUP] Phase 1: {warmup_loops} loops open-loop @ {hz}Hz (priming LSTM)")
        print(f"[WARMUP] Phase 2: {live_duration}s live policy")
        print(f"[WARMUP] Starting in 3 seconds...")
        time.sleep(3)

        # Reset policy state
        self.startup_steps = 0
        self.prev_actions = np.zeros(12)
        self.prev_smoothed_actions = np.zeros(12)
        if self.action_lpf is not None:
            self.action_lpf.reset()
        self._reset_pd_dynamics()
        self._reset_hw_observer()

        # Logging
        log_rows = []
        t0 = time.time()
        global_step = 0

        # ===== PHASE 1: CSV warmup =====
        print("\n--- PHASE 1: CSV Warmup (open-loop actions, LSTM observing) ---")
        for loop_i in range(warmup_loops):
            start_idx = warmup_skip  # skip startup transient on ALL loops
            print(f"  Warmup loop {loop_i + 1}/{warmup_loops} (from step {start_idx})")

            for i in range(start_idx, n_csv_steps):
                step_start = time.time()

                # Use CSV actions for the servos (open-loop)
                csv_actions = action_data[i]
                target_sim = self.sim_default_positions + csv_actions * scale
                target_sim = np.clip(target_sim, self.joint_lower_limits, self.joint_upper_limits)
                target_matrix = self._isaac_to_hardware_matrix(target_sim)
                self.hardware.set_actuator_postions(target_matrix)

                # Update hw_observer with the CSV actions (builds obs from real servo positions)
                current_hw_pos = self._read_joint_positions_sim_frame()
                if current_hw_pos is not None and self.obs_mode == "hw_observer":
                    self._last_hw_pos_sim = current_hw_pos
                    actual_dt = time.time() - (self.prev_time if self.prev_time else time.time())
                    if actual_dt < 0.001:
                        actual_dt = dt
                    self._step_hw_observer(csv_actions, current_hw_pos, actual_dt)
                    self._last_dt = actual_dt
                    self.prev_time = time.time()

                # Build observation — use POLICY's own prev_actions (not CSV)
                # so the LSTM sees its own output distribution during priming
                obs = self.get_observation()
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                with torch.no_grad():
                    policy_output = self.policy(obs_tensor).squeeze().numpy()

                # Store POLICY output as prev_actions for next step's observation
                # This is key: the LSTM must be conditioned on its OWN action
                # distribution, not the CSV actions, to avoid shock at handoff
                self.prev_actions = np.clip(policy_output, -1.0, 1.0)

                # Also feed policy output through the PD dynamics so
                # joint_pos_rel/vel/effort reflect what the policy would create
                if self.use_synthetic_obs:
                    if self.obs_mode == "hw_observer":
                        # hw_observer already updated above with csv_actions for servo tracking
                        # but also step PD with policy output for effort/vel consistency
                        pass  # hw_observer uses real positions, already stepped
                    else:
                        self._step_pd_dynamics(self.prev_actions)

                # Log
                row = {
                    'step': global_step, 'time': time.time() - t0,
                    'phase': 'warmup', 'loop': loop_i, 'csv_step': i,
                }
                for j in range(12):
                    row[f'csv_action_{j}'] = float(csv_actions[j])
                    row[f'policy_action_{j}'] = float(policy_output[j])
                for j in range(60):
                    row[f'obs_{j}'] = float(obs[j])
                log_rows.append(row)

                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                global_step += 1

        print(f"  Warmup complete ({global_step} steps). LSTM primed.")
        print(f"  Policy prev_actions at handoff: [{self.prev_actions.min():+.3f}, {self.prev_actions.max():+.3f}]")

        # ===== PHASE 2: Live policy =====
        print(f"\n--- PHASE 2: Live Policy ({live_duration}s) ---")
        n_live_steps = int(live_duration * hz)
        self.control_active = True
        self.startup_steps = self.startup_duration  # skip fade-in (already warmed up)
        # Seed smoothed actions from policy's current output to avoid jump
        self.prev_smoothed_actions = self.prev_actions.copy()

        for i in range(n_live_steps):
            step_start = time.time()

            try:
                self.control_step()
            except Exception as e:
                print(f"[ERROR] {e}")
                import traceback
                traceback.print_exc()
                break

            # Log
            obs = self.get_observation()
            row = {
                'step': global_step, 'time': time.time() - t0,
                'phase': 'live', 'loop': -1, 'csv_step': -1,
            }
            for j in range(12):
                row[f'csv_action_{j}'] = float(self.prev_actions[j])
            for j in range(60):
                row[f'obs_{j}'] = float(obs[j])
            log_rows.append(row)

            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

            if i % max(1, int(hz)) == 0:
                act_range = f"[{self.prev_actions.min():+.3f}, {self.prev_actions.max():+.3f}]"
                print(f"  Live step {i}/{n_live_steps}: actions={act_range}")

            global_step += 1

        self.control_active = False

        # Save log
        if log_rows:
            import csv
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_file = f"/home/ubuntu/mp2_mlp/csv_warmup_{timestamp}.csv"
            fieldnames = list(log_rows[0].keys())
            with open(log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(log_rows)
            print(f"[LOG] Saved {len(log_rows)} rows to {log_file}")

        print("[WARMUP] Done. Returning to stance...")
        time.sleep(2)
        target_matrix = self._isaac_to_hardware_matrix(self.sim_default_positions)
        self.hardware.set_actuator_postions(target_matrix)


# ==================== MAIN ====================
if __name__ == "__main__":
    import sys

    # Allow preset from command line: python mlp_controller_v3.py lstm_pd
    initial_preset = sys.argv[1] if len(sys.argv) > 1 else "mlp_pd"

    print("\n" + "="*70)
    print("MINI PUPPER 2 - FIXED MAPPING CONTROLLER v3")
    print("="*70)
    print(f"""
Controls:
  w/s      - Forward/Backward
  a/d      - Strafe Left/Right
  q/e      - Turn Left/Right
  SPACE    - Stop
  +/-      - Adjust speed
  debug    - Print debug info
  log      - Start CSV logging (12s capture)
  set X Y  - Set parameter to value Y
  play mlp [hz] [scale] [loops] - Open-loop playback of best MLP sim actions
  play lstm [hz] [scale] [loops] - Open-loop playback of best LSTM sim actions
  warmup [loops] [live_secs]    - CSV warmup → live policy handoff
  test     - Legacy test replay
  x        - Exit

Presets (switch with 'preset <name>', list with 'presets'):
  mlp_pd        - MLP + PD (baseline, no EMA)
  mlp_pd_ema    - MLP + PD + EMA (target ~2Hz gait)
  mlp_nopd      - MLP no PD (expect failure)
  lstm_pd       - LSTM + PD + EMA
  lstm_nopd     - LSTM no PD, raw effort
  lstm_nopd_boost - LSTM no PD, effort x5
  lstm_nopd_zero  - LSTM no PD, zero effort
  mlp_pd_cf       - MLP + PD + ComplementaryFilter (real vel/gravity)
  lstm_nopd_cf    - LSTM + no PD + ComplementaryFilter

Test order for paper:
  1. preset mlp_pd       → log → w → (12s) → MLP baseline
  2. preset mlp_pd_ema   → log → w → (12s) → MLP with 2Hz filter
  3. preset lstm_pd      → log → w → (12s) → LSTM with PD
  4. preset lstm_nopd_boost → log → w → LSTM no PD, effort x5
  5. preset lstm_nopd_zero  → log → w → LSTM no PD, zero effort
  6. preset mlp_nopd     → log → w → (12s) → MLP no PD (failure control)
""")

    controller = FixedMappingControllerV3(preset=initial_preset)

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
            elif cmd == 'log':
                controller.log_rows = []
                controller.log_enabled = True
                print(f"[LOG] Logging started ({controller.current_preset}) — 600 steps (12s)")
                print("[LOG] Give a movement command (e.g. 'w') to capture walking data")
            elif cmd == 'test':
                controller.set_velocity_command(0, 0, 0)
                time.sleep(0.5)
                controller.test_with_sim_actions()
            elif cmd.startswith('play '):
                # play mlp|lstm [hz] [hw_scale] [loops]
                # e.g. "play mlp", "play lstm 25", "play mlp 25 0.5 3"
                controller.set_velocity_command(0, 0, 0)
                time.sleep(0.5)
                parts = cmd.split()
                policy_type = parts[1] if len(parts) > 1 else 'mlp'
                play_hz = int(parts[2]) if len(parts) > 2 else None
                play_scale = float(parts[3]) if len(parts) > 3 else None
                play_loops = int(parts[4]) if len(parts) > 4 else 3
                csv_map = {
                    'mlp': '/home/ubuntu/env_2_actions_mlp.csv',
                    'lstm': '/home/ubuntu/env_2_actions_lstm.csv',
                }
                csv_path = csv_map.get(policy_type, csv_map['mlp'])
                controller.test_with_sim_actions(
                    csv_path=csv_path, max_steps=0,
                    playback_hz=play_hz, hw_scale=play_scale, loops=play_loops
                )
            elif cmd.startswith('warmup'):
                # warmup [loops] [live_seconds]
                # e.g. "warmup", "warmup 4", "warmup 3 15"
                controller.set_velocity_command(speed, 0, 0)  # set command for obs
                time.sleep(0.1)
                controller.set_velocity_command(0, 0, 0)
                parts = cmd.split()
                w_loops = int(parts[1]) if len(parts) > 1 else 3
                w_live = float(parts[2]) if len(parts) > 2 else 10.0
                controller.velocity_command = np.array([speed, 0.0, 0.0])
                controller.csv_warmup_then_live(
                    csv_path='/home/ubuntu/env_2_actions_lstm.csv',
                    warmup_loops=w_loops, live_duration=w_live
                )
            elif cmd.startswith('preset '):
                preset_name = cmd.split(None, 1)[1].strip()
                controller.load_preset(preset_name)
            elif cmd == 'presets':
                print("\nAvailable presets:")
                for name, cfg in FixedMappingControllerV3.PRESETS.items():
                    marker = " <-- active" if name == controller.current_preset else ""
                    print(f"  {name:12s} {cfg['description']}{marker}")
                print()
            elif cmd.startswith('set '):
                parts = cmd.split()
                if len(parts) == 3:
                    controller.set_param(parts[1], parts[2])
                else:
                    print("Usage: set <param> <value>")
            elif cmd == 'x':
                break
            else:
                print("Unknown command. Type 'presets' to see test configurations.")

    except KeyboardInterrupt:
        pass

    controller.stop()
    control_thread.join(timeout=1.0)
    print("[DONE]")
