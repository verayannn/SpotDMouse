"""
HW-Observer closed-loop simulation analysis.

Runs MLP and LSTM policies through THREE different observation pipelines
in closed-loop simulation, then compares obs/action distributions:

  1. Sim (IsaacLab logged data — ground truth)
  2. Open-Loop PD (legacy _step_pd_dynamics — current HW wrapper)
  3. HW-Observer PD (NEW _step_hw_observer — proposed fix)

The HW-Observer simulation models what happens on real hardware:
  - Policy action → position servo instantly tracks to target (with noise/quantization)
  - HW-Observer reads actual position, differentiates for velocity, computes PD effort
  - This is a CLOSED LOOP: policy(obs) → action → servo model → hw_observer(pos) → obs

The key difference from open-loop PD:
  - Open-loop: PD ODE integrates its own mass-spring-damper (disconnected from servo reality)
  - HW-Observer: takes actual servo position as ground truth, computes PD vel/effort from it

If the hw_observer closes the gap, the green distributions should overlap with blue (sim).
"""

import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from scipy import stats
import os

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────
MLP_PT  = os.path.expanduser("~/policy_joyboy_delayedpdactuator_hippy.pt")
LSTM_PT = os.path.expanduser("~/policy_joyboy_delayedpdactuator_LSTM.pt")

SIM_MLP_DIR  = os.path.expanduser("~/obs_action_logs_mlp")
SIM_LSTM_DIR = os.path.expanduser("~/obs_action_logs_lstm")

HW_MLP_CSV  = os.path.expanduser("~/mp2_mlp/mlp_pd_cf_hw_log_20260314_004932.csv")
HW_LSTM_CSV = os.path.expanduser("~/mp2_mlp/lstm_nopd_cf_hw_log_20260314_005221.csv")

OUTPUT_PDF = os.path.expanduser(
    "~/SpotDMouse/P2-Terrain_Challenge/mp2/analysis/hw_observer_projection.pdf"
)

# ─── Config ───────────────────────────────────────────────────────────────────
DT = 0.02
SIM_SECONDS = 20.0
N_STEPS = int(SIM_SECONDS / DT)
KP, KD, INERTIA, FRICTION = 70.0, 1.2, 0.20, 0.03
EFFORT_LIMIT = 5.0
DELAY_STEPS = 9
PD_SUBSTEPS = 4
ACTION_SCALE = 1.5   # Training action scale (for PD target computation)
HW_SCALE = 0.55      # MLP servo output scale
LSTM_HW_SCALE = 1.5  # LSTM servo output scale
VEL_ALPHA = 0.3      # EMA alpha for hw_observer velocity filtering

# Servo model parameters
SERVO_QUANTIZATION = 0.012   # ~0.012 rad position quantization
SERVO_BANDWIDTH_HZ = 8.0     # Position servo tracking bandwidth
SERVO_TAU = 1.0 / (2 * np.pi * SERVO_BANDWIDTH_HZ)  # First-order lag time constant

DEFAULT_JOINT_POS = np.array([
    0.0, 0.785, -1.57,  0.0, 0.785, -1.57,
    0.0, 0.785, -1.57,  0.0, 0.785, -1.57,
])
JOINT_LOWER = np.array([
    -0.5, 0.0, -2.5, -0.5, 0.0, -2.5,
    -0.5, 0.0, -2.5, -0.5, 0.0, -2.5,
])
JOINT_UPPER = np.array([
    0.5, 1.5, -0.5, 0.5, 1.5, -0.5,
    0.5, 1.5, -0.5, 0.5, 1.5, -0.5,
])

OBS_DIM_NAMES = [
    'lin_vel_x', 'lin_vel_y', 'lin_vel_z',
    'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
    'grav_x', 'grav_y', 'grav_z',
    'cmd_x', 'cmd_y', 'cmd_yaw',
    'jpos_LF_hip', 'jpos_LF_thigh', 'jpos_LF_calf',
    'jpos_RF_hip', 'jpos_RF_thigh', 'jpos_RF_calf',
    'jpos_LB_hip', 'jpos_LB_thigh', 'jpos_LB_calf',
    'jpos_RB_hip', 'jpos_RB_thigh', 'jpos_RB_calf',
    'jvel_LF_hip', 'jvel_LF_thigh', 'jvel_LF_calf',
    'jvel_RF_hip', 'jvel_RF_thigh', 'jvel_RF_calf',
    'jvel_LB_hip', 'jvel_LB_thigh', 'jvel_LB_calf',
    'jvel_RB_hip', 'jvel_RB_thigh', 'jvel_RB_calf',
    'jeff_LF_hip', 'jeff_LF_thigh', 'jeff_LF_calf',
    'jeff_RF_hip', 'jeff_RF_thigh', 'jeff_RF_calf',
    'jeff_LB_hip', 'jeff_LB_thigh', 'jeff_LB_calf',
    'jeff_RB_hip', 'jeff_RB_thigh', 'jeff_RB_calf',
    'pact_LF_hip', 'pact_LF_thigh', 'pact_LF_calf',
    'pact_RF_hip', 'pact_RF_thigh', 'pact_RF_calf',
    'pact_LB_hip', 'pact_LB_thigh', 'pact_LB_calf',
    'pact_RB_hip', 'pact_RB_thigh', 'pact_RB_calf',
]
JOINT_NAMES = [
    'LF_hip', 'LF_thigh', 'LF_calf',
    'RF_hip', 'RF_thigh', 'RF_calf',
    'LB_hip', 'LB_thigh', 'LB_calf',
    'RB_hip', 'RB_thigh', 'RB_calf',
]

OBS_GROUPS = {
    'lin_vel':   (slice(0, 3),   3),
    'ang_vel':   (slice(3, 6),   3),
    'gravity':   (slice(6, 9),   3),
    'commands':  (slice(9, 12),  3),
    'joint_pos': (slice(12, 24), 12),
    'joint_vel': (slice(24, 36), 12),
    'joint_eff': (slice(36, 48), 12),
    'prev_act':  (slice(48, 60), 12),
}

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'figure.titlesize': 14, 'legend.fontsize': 9,
    'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 1.4,
})


# ─── Data Loaders ─────────────────────────────────────────────────────────────

def load_sim_envs(directory, n_envs=6):
    all_obs, all_act = [], []
    for i in range(n_envs):
        obs_path = os.path.join(directory, f'env_{i}_observations.csv')
        act_path = os.path.join(directory, f'env_{i}_actions.csv')
        if not os.path.exists(obs_path):
            continue
        obs_df = pd.read_csv(obs_path)
        act_df = pd.read_csv(act_path)
        all_obs.append(obs_df.iloc[:, 1:].values)
        all_act.append(act_df.iloc[:, 1:].values)
    return np.concatenate(all_obs), np.concatenate(all_act)


def load_hw_csv(path):
    df = pd.read_csv(path)
    obs_cols = [f'obs_{i}' for i in range(60)]
    act_cols = [f'raw_action_{i}' for i in range(12)]
    obs = df[obs_cols].values
    act = df[act_cols].values
    fade = df['fade'].values
    mask = fade >= 1.0
    return obs[mask], act[mask]


# ─── Simulation Models ───────────────────────────────────────────────────────

class OpenLoopPD:
    """Legacy open-loop PD wrapper (current _step_pd_dynamics)."""

    def __init__(self):
        self.position = DEFAULT_JOINT_POS.copy()
        self.velocity = np.zeros(12)
        self.effort = np.zeros(12)
        self.action_buffer = []

    def step(self, action_target):
        self.action_buffer.append(action_target.copy())
        idx = max(0, len(self.action_buffer) - DELAY_STEPS - 1)
        delayed = self.action_buffer[idx]
        dt_sub = DT / PD_SUBSTEPS
        for _ in range(PD_SUBSTEPS):
            err = delayed - self.position
            torque = KP * err - KD * self.velocity - FRICTION * np.sign(self.velocity)
            torque = np.clip(torque, -EFFORT_LIMIT, EFFORT_LIMIT)
            self.velocity += (torque / INERTIA) * dt_sub
            self.position += self.velocity * dt_sub
            hit = (self.position < JOINT_LOWER) | (self.position > JOINT_UPPER)
            self.position = np.clip(self.position, JOINT_LOWER, JOINT_UPPER)
            self.velocity[hit] = 0.0
        self.effort = torque
        return self.position.copy(), self.velocity.copy(), self.effort.copy()


class ServoModel:
    """Models the real Mini Pupper position servo behavior.

    Position servos track commanded targets with:
    - First-order lag (~8Hz bandwidth)
    - Position quantization (~0.012 rad)
    - Joint limit enforcement
    """

    def __init__(self):
        self.position = DEFAULT_JOINT_POS.copy()

    def step(self, target_position):
        """Move servo toward target with first-order dynamics + quantization."""
        # First-order lag: pos += (target - pos) * (1 - exp(-dt/tau))
        alpha = 1.0 - np.exp(-DT / SERVO_TAU)
        self.position = self.position + alpha * (target_position - self.position)
        # Quantization noise
        self.position += np.random.normal(0, SERVO_QUANTIZATION * 0.5, 12)
        # Joint limits
        self.position = np.clip(self.position, JOINT_LOWER, JOINT_UPPER)
        return self.position.copy()


class HWObserverPD:
    """New hardware-driven PD observer (proposed _step_hw_observer).

    Takes actual servo positions as input (not its own ODE state),
    computes PD-consistent velocity and effort from real trajectory.
    """

    def __init__(self):
        self.action_buffer = []
        self.prev_pos = DEFAULT_JOINT_POS.copy()
        self.hw_vel = np.zeros(12)

    def step(self, raw_action, actual_servo_pos):
        """Compute obs from actual position + delayed action target.

        Args:
            raw_action: current policy output (clipped to [-1,1])
            actual_servo_pos: real servo position in sim frame (12,)

        Returns:
            pos_rel, velocity, effort — the 36 dims of obs[12:48]
        """
        # Push action into delay buffer
        self.action_buffer.append(raw_action.copy())
        idx = max(0, len(self.action_buffer) - DELAY_STEPS - 1)
        delayed_action = self.action_buffer[idx]

        # Delayed target (same formula as sim training)
        target = DEFAULT_JOINT_POS + delayed_action * ACTION_SCALE
        target = np.clip(target, JOINT_LOWER, JOINT_UPPER)

        # Velocity from real position differentiation (EMA filtered)
        raw_vel = (actual_servo_pos - self.prev_pos) / DT
        raw_vel = np.clip(raw_vel, -10.5, 10.5)
        self.hw_vel = VEL_ALPHA * raw_vel + (1 - VEL_ALPHA) * self.hw_vel

        # PD effort from real position error
        error = target - actual_servo_pos
        torque = KP * error - KD * self.hw_vel - FRICTION * np.sign(self.hw_vel)
        effort = np.clip(torque / EFFORT_LIMIT, -1.0, 1.0)

        # Position relative to default
        pos_rel = actual_servo_pos - DEFAULT_JOINT_POS

        self.prev_pos = actual_servo_pos.copy()
        return pos_rel, self.hw_vel.copy(), effort


# ─── Closed-Loop Simulations ─────────────────────────────────────────────────

def run_openloop_pd(model, is_lstm=False):
    """Run policy with legacy open-loop PD wrapper (current approach)."""
    pda = OpenLoopPD()
    obs_all = np.zeros((N_STEPS, 60))
    act_all = np.zeros((N_STEPS, 12))
    prev_action = np.zeros(12)
    grav = np.array([0.0, 0.0, -1.0])
    cmd = np.array([0.15, 0.0, 0.0])
    if is_lstm:
        model.hidden_state.zero_()
        model.cell_state.zero_()
    for t in range(N_STEPS):
        obs = np.zeros(60)
        obs[0:3] = np.random.normal(0, 0.02, 3)
        obs[3:6] = np.random.normal(0, 0.1, 3)
        obs[6:9] = grav + np.random.normal(0, 0.01, 3)
        obs[9:12] = cmd
        obs[12:24] = pda.position - DEFAULT_JOINT_POS
        obs[24:36] = pda.velocity
        obs[36:48] = pda.effort / EFFORT_LIMIT
        obs[48:60] = prev_action
        obs_all[t] = obs
        with torch.no_grad():
            action = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        action = np.clip(action, -1.0, 1.0)
        act_all[t] = action
        pda.step(DEFAULT_JOINT_POS + action * ACTION_SCALE)
        prev_action = action
    return obs_all, act_all


def run_hw_observer_pd(model, hw_scale, is_lstm=False):
    """Run policy with HW-Observer PD (proposed approach).

    Closed loop:
      1. Build obs using hw_observer (real servo pos → PD vel/effort)
      2. Policy produces action from obs
      3. Compute servo target = default + action * hw_scale
      4. Servo model tracks target (with lag + quantization)
      5. HW-Observer reads new servo position → back to step 1
    """
    servo = ServoModel()
    observer = HWObserverPD()
    obs_all = np.zeros((N_STEPS, 60))
    act_all = np.zeros((N_STEPS, 12))
    prev_action = np.zeros(12)
    grav = np.array([0.0, 0.0, -1.0])
    cmd = np.array([0.15, 0.0, 0.0])
    if is_lstm:
        model.hidden_state.zero_()
        model.cell_state.zero_()

    # Warm up: servo at default for a few steps
    for _ in range(DELAY_STEPS + 2):
        observer.step(np.zeros(12), DEFAULT_JOINT_POS.copy())

    for t in range(N_STEPS):
        # Read current servo position
        servo_pos = servo.position.copy()

        # HW-Observer computes obs from real position
        pos_rel, vel, effort = observer.step(prev_action, servo_pos)

        # Build full observation
        obs = np.zeros(60)
        obs[0:3] = np.random.normal(0, 0.02, 3)      # lin_vel (noisy, like CF on HW)
        obs[3:6] = np.random.normal(0, 0.1, 3)        # ang_vel (noisy gyro)
        obs[6:9] = grav + np.random.normal(0, 0.01, 3) # gravity
        obs[9:12] = cmd
        obs[12:24] = np.clip(pos_rel, -0.9, 0.9)
        obs[24:36] = np.clip(vel, -10.0, 10.0)
        obs[36:48] = np.clip(effort, -1.0, 1.0)
        obs[48:60] = prev_action
        obs_all[t] = obs

        # Policy forward pass
        with torch.no_grad():
            action = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        action = np.clip(action, -1.0, 1.0)
        act_all[t] = action

        # Compute servo target (this is what the real controller sends to servos)
        servo_target = DEFAULT_JOINT_POS + action * hw_scale
        servo_target = np.clip(servo_target, JOINT_LOWER, JOINT_UPPER)

        # Servo tracks target
        servo.step(servo_target)

        prev_action = action

    return obs_all, act_all


# ─── Stats ────────────────────────────────────────────────────────────────────

def compute_dim_stats(data_a, data_b, names):
    results = []
    n = min(len(data_a), len(data_b))
    for i, name in enumerate(names):
        a = data_a[:n, i]
        b = data_b[:n, i]
        mean_a, std_a = a.mean(), a.std()
        mean_b, std_b = b.mean(), b.std()
        if std_a > 1e-8 and std_b > 1e-8:
            corr = np.corrcoef(a, b)[0, 1]
            slope, intercept, r_val, _, _ = stats.linregress(a, b)
            r2 = r_val ** 2
        else:
            corr, slope, intercept, r2 = 0.0, 0.0, 0.0, 0.0
        results.append({
            'name': name, 'idx': i,
            'mean_a': mean_a, 'std_a': std_a,
            'mean_b': mean_b, 'std_b': std_b,
            'corr': corr, 'slope': slope, 'intercept': intercept, 'r2': r2,
        })
    return results


# ─── Plotting ─────────────────────────────────────────────────────────────────

GROUP_COLORS = {
    'lin_vel': '#e41a1c', 'ang_vel': '#377eb8', 'gravity': '#4daf4a',
    'commands': '#984ea3', 'joint_pos': '#ff7f00', 'joint_vel': '#a65628',
    'joint_eff': '#f781bf', 'prev_act': '#999999',
}


def page_correlation_comparison(fig, stats_sim_ol, stats_sim_hw, stats_sim_hwobs,
                                 dim_names, arch, title_prefix="Obs"):
    """Triple bar chart: Sim↔HW, Sim↔OpenLoop, Sim↔HW-Observer correlation."""
    gs = GridSpec(2, 1, hspace=0.5, height_ratios=[2, 1])
    n = len(dim_names)
    x = np.arange(n)
    w = 0.25

    corr_hw = [s['corr'] for s in stats_sim_hw]
    corr_ol = [s['corr'] for s in stats_sim_ol]
    corr_hwobs = [s['corr'] for s in stats_sim_hwobs]

    r2_hw = [s['r2'] for s in stats_sim_hw]
    r2_ol = [s['r2'] for s in stats_sim_ol]
    r2_hwobs = [s['r2'] for s in stats_sim_hwobs]

    # Top: correlation
    ax = fig.add_subplot(gs[0])
    ax.bar(x - w, corr_hw, w, label='HW (logged)', color='#d62728', alpha=0.6)
    ax.bar(x, corr_ol, w, label='Open-Loop PD', color='#ff7f0e', alpha=0.6)
    ax.bar(x + w, corr_hwobs, w, label='HW-Observer PD', color='#2ca02c', alpha=0.8)
    ax.set_ylabel('Pearson Correlation with Sim')
    ax.set_title(f'{arch} — {title_prefix} Correlation with Sim (3 approaches)',
                 fontweight='bold')
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axhline(0.5, color='gray', ls='--', alpha=0.3)
    ax.axhline(-0.5, color='gray', ls='--', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(dim_names, rotation=90, fontsize=5 if n > 20 else 8)
    ax.legend(fontsize=9, loc='lower right')

    # Bottom: R²
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(x - w, r2_hw, w, label='HW (logged)', color='#d62728', alpha=0.6)
    ax2.bar(x, r2_ol, w, label='Open-Loop PD', color='#ff7f0e', alpha=0.6)
    ax2.bar(x + w, r2_hwobs, w, label='HW-Observer PD', color='#2ca02c', alpha=0.8)
    ax2.set_ylabel('R² (linear fit)')
    ax2.set_xlabel(f'{title_prefix} Dimension')
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(x)
    ax2.set_xticklabels(dim_names, rotation=90, fontsize=5 if n > 20 else 8)
    ax2.legend(fontsize=9, loc='upper right')


def page_distribution_overlay(fig, data_dict, dim_names, title, indices):
    """Overlaid histograms for selected dims across all domains."""
    n = len(indices)
    cols = 4
    rows = (n + cols - 1) // cols
    gs = GridSpec(rows, cols, hspace=0.6, wspace=0.35)

    domain_colors = {
        'Sim': '#1f77b4',
        'Hardware': '#d62728',
        'Open-Loop PD': '#ff7f0e',
        'HW-Observer PD': '#2ca02c',
    }

    for i, dim_idx in enumerate(indices):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        for dom_name, data in data_dict.items():
            vals = data[:, dim_idx]
            ax.hist(vals, bins=50, alpha=0.35,
                    color=domain_colors.get(dom_name, 'gray'),
                    label=f'{dom_name} (μ={vals.mean():.3f}, σ={vals.std():.3f})',
                    density=True)
        ax.set_title(dim_names[dim_idx], fontsize=8, fontweight='bold')
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=4.5)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)


def page_scatter_3way(fig, sim_data, ol_data, hwobs_data, hw_data,
                      stats_ol, stats_hwobs, stats_hw,
                      dim_names, arch, indices):
    """3 columns of scatter: HW vs Sim, Open-Loop vs Sim, HW-Observer vs Sim."""
    n = len(indices)
    gs = GridSpec(n, 3, hspace=0.8, wspace=0.35)
    n_pts_ol = min(len(sim_data), len(ol_data))
    n_pts_hw = min(len(sim_data), len(hw_data))
    n_pts_hwobs = min(len(sim_data), len(hwobs_data))

    for row, dim_idx in enumerate(indices):
        # Col 0: HW (logged) vs Sim
        ax0 = fig.add_subplot(gs[row, 0])
        s = stats_hw[dim_idx]
        a = sim_data[:n_pts_hw, dim_idx]
        b = hw_data[:n_pts_hw, dim_idx]
        ax0.scatter(a, b, s=2, alpha=0.15, color='#d62728')
        if s['std_a'] > 1e-8:
            xl = np.array([a.min(), a.max()])
            ax0.plot(xl, s['slope'] * xl + s['intercept'], 'k-', lw=1.5, alpha=0.7)
        ax0.set_title(f'{dim_names[dim_idx]} — HW\n'
                      f'r={s["corr"]:.3f} R²={s["r2"]:.3f} slope={s["slope"]:.2f}',
                      fontsize=6, fontweight='bold', color='#d62728')
        ax0.set_xlabel('Sim', fontsize=6)
        ax0.set_ylabel('HW', fontsize=6)
        ax0.tick_params(labelsize=5)

        # Col 1: Open-Loop vs Sim
        ax1 = fig.add_subplot(gs[row, 1])
        s = stats_ol[dim_idx]
        a = sim_data[:n_pts_ol, dim_idx]
        b = ol_data[:n_pts_ol, dim_idx]
        ax1.scatter(a, b, s=2, alpha=0.15, color='#ff7f0e')
        if s['std_a'] > 1e-8:
            xl = np.array([a.min(), a.max()])
            ax1.plot(xl, s['slope'] * xl + s['intercept'], 'k-', lw=1.5, alpha=0.7)
        ax1.set_title(f'{dim_names[dim_idx]} — Open-Loop\n'
                      f'r={s["corr"]:.3f} R²={s["r2"]:.3f} slope={s["slope"]:.2f}',
                      fontsize=6, fontweight='bold', color='#ff7f0e')
        ax1.set_xlabel('Sim', fontsize=6)
        ax1.set_ylabel('Open-Loop', fontsize=6)
        ax1.tick_params(labelsize=5)

        # Col 2: HW-Observer vs Sim
        ax2 = fig.add_subplot(gs[row, 2])
        s = stats_hwobs[dim_idx]
        a = sim_data[:n_pts_hwobs, dim_idx]
        b = hwobs_data[:n_pts_hwobs, dim_idx]
        ax2.scatter(a, b, s=2, alpha=0.15, color='#2ca02c')
        if s['std_a'] > 1e-8:
            xl = np.array([a.min(), a.max()])
            ax2.plot(xl, s['slope'] * xl + s['intercept'], 'k-', lw=1.5, alpha=0.7)
        ax2.set_title(f'{dim_names[dim_idx]} — HW-Observer\n'
                      f'r={s["corr"]:.3f} R²={s["r2"]:.3f} slope={s["slope"]:.2f}',
                      fontsize=6, fontweight='bold', color='#2ca02c')
        ax2.set_xlabel('Sim', fontsize=6)
        ax2.set_ylabel('HW-Observer', fontsize=6)
        ax2.tick_params(labelsize=5)

    fig.suptitle(f'{arch} — Scatter: Sim vs HW / Open-Loop / HW-Observer',
                 fontsize=12, fontweight='bold', y=1.0)


def page_time_series(fig, sim_obs, ol_obs, hwobs_obs, dim_names, arch, indices,
                     t_range=(0, 200)):
    """Time series for selected dims: overlay sim, open-loop, hw-observer."""
    n = len(indices)
    gs = GridSpec(n, 1, hspace=0.5)
    t0, t1 = t_range
    t = np.arange(t0, t1) * DT

    colors = {'Sim': '#1f77b4', 'Open-Loop PD': '#ff7f0e', 'HW-Observer PD': '#2ca02c'}

    for i, dim_idx in enumerate(indices):
        ax = fig.add_subplot(gs[i])
        ax.plot(t, sim_obs[t0:t1, dim_idx], color=colors['Sim'],
                alpha=0.8, label='Sim', linewidth=1.2)
        ax.plot(t, ol_obs[t0:t1, dim_idx], color=colors['Open-Loop PD'],
                alpha=0.7, label='Open-Loop PD', linewidth=1.0, linestyle='--')
        ax.plot(t, hwobs_obs[t0:t1, dim_idx], color=colors['HW-Observer PD'],
                alpha=0.8, label='HW-Observer PD', linewidth=1.0, linestyle='-.')
        ax.set_ylabel(dim_names[dim_idx], fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')
        if i == n - 1:
            ax.set_xlabel('Time (s)', fontsize=9)

    fig.suptitle(f'{arch} — Time Series Comparison (t={t0*DT:.1f}-{t1*DT:.1f}s)',
                 fontsize=12, fontweight='bold')


def page_summary_table(fig, obs_stats_hw, obs_stats_ol, obs_stats_hwobs,
                       act_stats_hw, act_stats_ol, act_stats_hwobs, arch):
    """Full dimension-by-dimension comparison table."""
    ax = fig.add_subplot(111)
    ax.axis('off')

    lines = []
    lines.append(f"{arch} — FULL COMPARISON: Sim ↔ {{HW, Open-Loop, HW-Observer}}")
    lines.append("=" * 110)
    lines.append(f"  {'Dimension':25s} {'HW Corr':>8s} {'OL Corr':>8s} {'Obs Corr':>8s}  "
                 f"{'HW R²':>6s} {'OL R²':>6s} {'Obs R²':>6s}  "
                 f"{'HW μ':>7s} {'OL μ':>7s} {'Obs μ':>7s} {'Sim μ':>7s}")
    lines.append(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8}  {'─'*6} {'─'*6} {'─'*6}  "
                 f"{'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    lines.append("  OBSERVATIONS:")
    for sh, so, sn in zip(obs_stats_hw, obs_stats_ol, obs_stats_hwobs):
        best = ''
        corrs = [abs(sh['corr']), abs(so['corr']), abs(sn['corr'])]
        if corrs[2] == max(corrs) and corrs[2] > 0.01:
            best = ' *'
        lines.append(f"  {sh['name']:25s} {sh['corr']:+8.3f} {so['corr']:+8.3f} {sn['corr']:+8.3f}  "
                     f"{sh['r2']:6.3f} {so['r2']:6.3f} {sn['r2']:6.3f}  "
                     f"{sh['mean_b']:7.3f} {so['mean_b']:7.3f} {sn['mean_b']:7.3f} {sh['mean_a']:7.3f}{best}")

    lines.append("\n  ACTIONS:")
    for sh, so, sn in zip(act_stats_hw, act_stats_ol, act_stats_hwobs):
        best = ''
        corrs = [abs(sh['corr']), abs(so['corr']), abs(sn['corr'])]
        if corrs[2] == max(corrs) and corrs[2] > 0.01:
            best = ' *'
        lines.append(f"  {sh['name']:25s} {sh['corr']:+8.3f} {so['corr']:+8.3f} {sn['corr']:+8.3f}  "
                     f"{sh['r2']:6.3f} {so['r2']:6.3f} {sn['r2']:6.3f}  "
                     f"{sh['mean_b']:7.3f} {so['mean_b']:7.3f} {sn['mean_b']:7.3f} {sh['mean_a']:7.3f}{best}")

    # Summary stats
    hw_mean_r2 = np.mean([s['r2'] for s in obs_stats_hw])
    ol_mean_r2 = np.mean([s['r2'] for s in obs_stats_ol])
    hwobs_mean_r2 = np.mean([s['r2'] for s in obs_stats_hwobs])
    hw_mean_corr = np.mean([abs(s['corr']) for s in obs_stats_hw])
    ol_mean_corr = np.mean([abs(s['corr']) for s in obs_stats_ol])
    hwobs_mean_corr = np.mean([abs(s['corr']) for s in obs_stats_hwobs])

    lines.append(f"\n  SUMMARY (obs only):")
    lines.append(f"    Mean |corr|:  HW={hw_mean_corr:.4f}  OL={ol_mean_corr:.4f}  HW-Obs={hwobs_mean_corr:.4f}")
    lines.append(f"    Mean R²:      HW={hw_mean_r2:.4f}  OL={ol_mean_r2:.4f}  HW-Obs={hwobs_mean_r2:.4f}")

    # Count wins
    hwobs_wins = sum(1 for sh, so, sn in zip(obs_stats_hw, obs_stats_ol, obs_stats_hwobs)
                     if abs(sn['corr']) > abs(so['corr']) and abs(sn['corr']) > abs(sh['corr']))
    lines.append(f"    HW-Observer wins (highest |corr|): {hwobs_wins}/{len(obs_stats_hw)} dims")

    text = "\n".join(lines)
    ax.text(0.01, 0.99, text, transform=ax.transAxes,
            fontsize=5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.6))
    fig.suptitle(f'{arch} — Full Comparison Table (* = HW-Observer best)',
                 fontsize=12, fontweight='bold')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("[1/5] Loading sim data...")
    sim_data = {}
    for arch, sim_dir in [('MLP', SIM_MLP_DIR), ('LSTM', SIM_LSTM_DIR)]:
        obs, act = load_sim_envs(sim_dir)
        sim_data[arch] = (obs, act)
        print(f"  {arch} sim: {obs.shape[0]} steps")

    print("[2/5] Loading hardware logs...")
    hw_data = {}
    for arch, hw_path in [('MLP', HW_MLP_CSV), ('LSTM', HW_LSTM_CSV)]:
        obs, act = load_hw_csv(hw_path)
        hw_data[arch] = (obs, act)
        print(f"  {arch} HW: {obs.shape[0]} steps")

    print("[3/5] Running Open-Loop PD simulation (legacy)...")
    model_mlp = torch.jit.load(MLP_PT, map_location='cpu'); model_mlp.eval()
    model_lstm = torch.jit.load(LSTM_PT, map_location='cpu'); model_lstm.eval()
    skip = int(5.0 / DT)  # Skip first 5s warmup

    np.random.seed(42)
    ol_mlp_obs, ol_mlp_act = run_openloop_pd(model_mlp, is_lstm=False)
    ol_mlp_obs, ol_mlp_act = ol_mlp_obs[skip:], ol_mlp_act[skip:]
    print(f"  MLP open-loop: {ol_mlp_obs.shape[0]} steps")

    np.random.seed(42)
    ol_lstm_obs, ol_lstm_act = run_openloop_pd(model_lstm, is_lstm=True)
    ol_lstm_obs, ol_lstm_act = ol_lstm_obs[skip:], ol_lstm_act[skip:]
    print(f"  LSTM open-loop: {ol_lstm_obs.shape[0]} steps")

    print("[4/5] Running HW-Observer PD simulation (proposed)...")
    np.random.seed(42)
    hwobs_mlp_obs, hwobs_mlp_act = run_hw_observer_pd(
        model_mlp, hw_scale=HW_SCALE, is_lstm=False)
    hwobs_mlp_obs, hwobs_mlp_act = hwobs_mlp_obs[skip:], hwobs_mlp_act[skip:]
    print(f"  MLP hw-observer: {hwobs_mlp_obs.shape[0]} steps")

    # Reload LSTM to reset hidden state
    model_lstm = torch.jit.load(LSTM_PT, map_location='cpu'); model_lstm.eval()
    np.random.seed(42)
    hwobs_lstm_obs, hwobs_lstm_act = run_hw_observer_pd(
        model_lstm, hw_scale=LSTM_HW_SCALE, is_lstm=True)
    hwobs_lstm_obs, hwobs_lstm_act = hwobs_lstm_obs[skip:], hwobs_lstm_act[skip:]
    print(f"  LSTM hw-observer: {hwobs_lstm_obs.shape[0]} steps")

    all_data = {
        'MLP': {
            'sim': sim_data['MLP'],
            'hw': hw_data['MLP'],
            'ol': (ol_mlp_obs, ol_mlp_act),
            'hwobs': (hwobs_mlp_obs, hwobs_mlp_act),
        },
        'LSTM': {
            'sim': sim_data['LSTM'],
            'hw': hw_data['LSTM'],
            'ol': (ol_lstm_obs, ol_lstm_act),
            'hwobs': (hwobs_lstm_obs, hwobs_lstm_act),
        },
    }

    print("[5/5] Generating comparison PDF...")

    with PdfPages(OUTPUT_PDF) as pdf:
        for arch in ['MLP', 'LSTM']:
            d = all_data[arch]
            sim_obs, sim_act = d['sim']
            hw_obs, hw_act = d['hw']
            ol_obs, ol_act = d['ol']
            hwobs_obs, hwobs_act = d['hwobs']

            # Compute stats: each vs Sim
            obs_stats_hw = compute_dim_stats(sim_obs, hw_obs, OBS_DIM_NAMES)
            obs_stats_ol = compute_dim_stats(sim_obs, ol_obs, OBS_DIM_NAMES)
            obs_stats_hwobs = compute_dim_stats(sim_obs, hwobs_obs, OBS_DIM_NAMES)

            act_stats_hw = compute_dim_stats(sim_act, hw_act, JOINT_NAMES)
            act_stats_ol = compute_dim_stats(sim_act, ol_act, JOINT_NAMES)
            act_stats_hwobs = compute_dim_stats(sim_act, hwobs_act, JOINT_NAMES)

            # ── PAGE 1: Obs correlation comparison (3 bars) ──
            fig = plt.figure(figsize=(18, 8))
            page_correlation_comparison(fig, obs_stats_ol, obs_stats_hw, obs_stats_hwobs,
                                        OBS_DIM_NAMES, arch, "Observation")
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            # ── PAGE 2: Action correlation comparison (3 bars) ──
            fig = plt.figure(figsize=(14, 8))
            page_correlation_comparison(fig, act_stats_ol, act_stats_hw, act_stats_hwobs,
                                        JOINT_NAMES, arch, "Action")
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            # ── PAGE 3: Distribution overlays — thigh positions ──
            thigh_pos_idx = [13, 16, 19, 22]
            obs_dict = {
                'Sim': sim_obs, 'Hardware': hw_obs,
                'Open-Loop PD': ol_obs, 'HW-Observer PD': hwobs_obs,
            }
            fig = plt.figure(figsize=(16, 5))
            page_distribution_overlay(fig, obs_dict, OBS_DIM_NAMES,
                f'{arch} — Thigh Position Distributions', thigh_pos_idx)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            # ── PAGE 4: Distribution overlays — thigh velocities ──
            thigh_vel_idx = [25, 28, 31, 34]
            fig = plt.figure(figsize=(16, 5))
            page_distribution_overlay(fig, obs_dict, OBS_DIM_NAMES,
                f'{arch} — Thigh Velocity Distributions', thigh_vel_idx)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            # ── PAGE 5: Distribution overlays — thigh efforts ──
            thigh_eff_idx = [37, 40, 43, 46]
            fig = plt.figure(figsize=(16, 5))
            page_distribution_overlay(fig, obs_dict, OBS_DIM_NAMES,
                f'{arch} — Thigh Effort Distributions', thigh_eff_idx)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            # ── PAGE 6: Distribution overlays — actions ──
            act_dict = {
                'Sim': sim_act, 'Hardware': hw_act,
                'Open-Loop PD': ol_act, 'HW-Observer PD': hwobs_act,
            }
            fig = plt.figure(figsize=(16, 8))
            page_distribution_overlay(fig, act_dict, JOINT_NAMES,
                f'{arch} — Action Distributions', list(range(12)))
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            # ── PAGES 7-8: Scatter 3-way for thigh dims ──
            fig = plt.figure(figsize=(12, 4 * len(thigh_pos_idx)))
            page_scatter_3way(fig, sim_obs, ol_obs, hwobs_obs, hw_obs,
                             obs_stats_ol, obs_stats_hwobs, obs_stats_hw,
                             OBS_DIM_NAMES, arch, thigh_pos_idx)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            fig = plt.figure(figsize=(12, 4 * len(thigh_vel_idx)))
            page_scatter_3way(fig, sim_obs, ol_obs, hwobs_obs, hw_obs,
                             obs_stats_ol, obs_stats_hwobs, obs_stats_hw,
                             OBS_DIM_NAMES, arch, thigh_vel_idx)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            # ── PAGE 9: Time series for thigh dims ──
            fig = plt.figure(figsize=(14, 3 * 4))
            page_time_series(fig, sim_obs, ol_obs, hwobs_obs,
                            OBS_DIM_NAMES, arch,
                            [13, 25, 37, 49],  # LF thigh: pos, vel, effort, prev_act
                            t_range=(0, min(300, len(ol_obs))))
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            # ── PAGE 10: Full summary table ──
            fig = plt.figure(figsize=(16, 22))
            page_summary_table(fig, obs_stats_hw, obs_stats_ol, obs_stats_hwobs,
                              act_stats_hw, act_stats_ol, act_stats_hwobs, arch)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            print(f"  {arch}: done")

    print(f"\n[DONE] PDF saved to: {OUTPUT_PDF}")


if __name__ == '__main__':
    main()
