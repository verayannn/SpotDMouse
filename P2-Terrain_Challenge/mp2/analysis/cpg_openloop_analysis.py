"""
CPG Open-Loop Analysis: MLP vs LSTM — Thigh Joints Only.

Runs the actual policy networks forward in open-loop with a synthetic PD
actuator model. Injects delay perturbation at t=2.0s (mid-stride, after
gait stabilizes). Computes % recovery metrics and input ablation to prove
the MLP is a CPG oscillator.

Usage:
    python cpg_openloop_analysis.py
"""

import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import os

warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero')

# ─── Config ───────────────────────────────────────────────────────────────────
MLP_PATH = os.path.expanduser("~/policy_joyboy_delayedpdactuator_mlp_retrain.pt")
LSTM_PATH = os.path.expanduser("~/policy_joyboy_delayedpdactuator_LSTM_retrain.pt")
OUTPUT_PDF = os.path.expanduser(
    "~/SpotDMouse/P2-Terrain_Challenge/mp2/analysis/cpg_openloop_analysis.pdf"
)

DT = 0.02            # 50 Hz control loop
SIM_SECONDS = 20.0   # 20s total: 10s stabilization + 10s post-perturbation
N_STEPS = int(SIM_SECONDS / DT)
PERTURB_TIME = 10.0  # Inject perturbation at 10.0s (after gait fully stable)
PERTURB_STEP = int(PERTURB_TIME / DT)

# PD actuator params (matches DelayedPDActuatorCfg)
KP = 70.0
KD = 1.2
INERTIA = 0.20
FRICTION = 0.03
EFFORT_LIMIT = 5.0
DELAY_STEPS = 9    # Nominal ~76ms
PD_SUBSTEPS = 4    # Integration substeps per control step
ACTION_SCALE = 1.5  # Training action scale

# Default joint positions (from URDF)
DEFAULT_JOINT_POS = np.array([
    0.0, 0.55, -1.0,   # LF: hip, thigh, calf
    0.0, 0.55, -1.0,   # RF
    0.0, 0.55, -1.0,   # LB
    0.0, 0.55, -1.0,   # RB
])

# URDF joint position limits (PhysX hard clamps these)
JOINT_LOWER = np.array([
    -0.524, 0.0, -2.356,   # LF: hip, thigh, calf
    -0.524, 0.0, -2.356,   # RF
    -0.524, 0.0, -2.356,   # LB
    -0.524, 0.0, -2.356,   # RB
])
JOINT_UPPER = np.array([
    0.524, 1.396, 0.0,     # LF
    0.524, 1.396, 0.0,     # RF
    0.524, 1.396, 0.0,     # LB
    0.524, 1.396, 0.0,     # RB
])

# Obs indices
IDX_LIN_VEL  = slice(0, 3)
IDX_ANG_VEL  = slice(3, 6)
IDX_GRAVITY  = slice(6, 9)
IDX_CMD      = slice(9, 12)
IDX_JPOS     = slice(12, 24)
IDX_JVEL     = slice(24, 36)
IDX_JEFFORT  = slice(36, 48)
IDX_PREV_ACT = slice(48, 60)

# Thigh indices within 12 joints
THIGH_IDX = [1, 4, 7, 10]  # LF, RF, LB, RB thigh
LEG_NAMES = ['LF', 'RF', 'LB', 'RB']
LEG_COLORS = {'LF': '#d62728', 'RF': '#1f77b4', 'LB': '#8c564b', 'RB': '#2ca02c'}
LEG_FULL = {'LF': 'Left Front', 'RF': 'Right Front', 'LB': 'Left Back', 'RB': 'Right Back'}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.titlesize': 14,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.4,
})


# ─── Synthetic PD Actuator ────────────────────────────────────────────────────

class SyntheticPDActuator:
    """Simulates delayed PD actuator dynamics matching DelayedPDActuatorCfg."""

    def __init__(self, n_joints=12, delay_steps=DELAY_STEPS):
        self.n_joints = n_joints
        self.delay_steps = delay_steps
        self.position = np.zeros(n_joints)
        self.velocity = np.zeros(n_joints)
        self.effort = np.zeros(n_joints)
        self.action_buffer = []

    def reset(self, init_pos=None):
        self.position = init_pos.copy() if init_pos is not None else np.zeros(self.n_joints)
        self.velocity = np.zeros(self.n_joints)
        self.effort = np.zeros(self.n_joints)
        self.action_buffer = []

    def step(self, action_target):
        """Apply one control step with delay and PD dynamics."""
        # Add to delay buffer
        self.action_buffer.append(action_target.copy())

        # Get delayed target
        if len(self.action_buffer) > self.delay_steps:
            delayed_target = self.action_buffer[-self.delay_steps - 1]
        else:
            delayed_target = self.action_buffer[0]

        # PD dynamics with substeps
        dt_sub = DT / PD_SUBSTEPS
        for _ in range(PD_SUBSTEPS):
            error = delayed_target - self.position
            torque = KP * error - KD * self.velocity
            torque -= FRICTION * np.sign(self.velocity)
            torque = np.clip(torque, -EFFORT_LIMIT, EFFORT_LIMIT)

            accel = torque / INERTIA
            self.velocity += accel * dt_sub
            self.position += self.velocity * dt_sub

            # Enforce URDF joint position limits (PhysX does this)
            hit_lower = self.position < JOINT_LOWER
            hit_upper = self.position > JOINT_UPPER
            self.position = np.clip(self.position, JOINT_LOWER, JOINT_UPPER)
            # Zero velocity on limit contact
            self.velocity[hit_lower | hit_upper] = 0.0

        self.effort = torque
        return self.position.copy(), self.velocity.copy(), self.effort.copy()


# ─── Open-Loop Sim ────────────────────────────────────────────────────────────

def run_openloop(model, is_lstm=False, cmd=np.array([0.15, 0.0, 0.0]),
                 extra_delay=0, perturb_step=None):
    """Run policy open-loop with synthetic PD actuator.

    Args:
        extra_delay: Additional delay steps injected at perturb_step.
        perturb_step: Step at which to inject extra delay (None = no perturbation).
    """
    pd = SyntheticPDActuator(delay_steps=DELAY_STEPS)
    pd.reset(init_pos=DEFAULT_JOINT_POS.copy())

    # Storage
    all_obs = np.zeros((N_STEPS, 60))
    all_actions = np.zeros((N_STEPS, 12))
    all_thigh_pos = np.zeros((N_STEPS, 4))
    all_thigh_vel = np.zeros((N_STEPS, 4))

    # Initial state
    prev_action = np.zeros(12)
    gravity = np.array([0.0, 0.0, -1.0])

    if is_lstm:
        # Reset hidden state
        model.hidden_state.zero_()
        model.cell_state.zero_()

    for t in range(N_STEPS):
        # If perturbation active, increase delay
        if perturb_step is not None and t >= perturb_step:
            pd.delay_steps = DELAY_STEPS + extra_delay
        else:
            pd.delay_steps = DELAY_STEPS

        # Build observation
        obs = np.zeros(60)
        obs[IDX_LIN_VEL] = np.random.normal(0, 0.02, 3)  # Small noise (open-loop, no real vel)
        obs[IDX_ANG_VEL] = np.random.normal(0, 0.1, 3)
        obs[IDX_GRAVITY] = gravity + np.random.normal(0, 0.01, 3)
        obs[IDX_CMD] = cmd
        obs[IDX_JPOS] = pd.position - DEFAULT_JOINT_POS  # Relative to default
        obs[IDX_JVEL] = pd.velocity
        obs[IDX_JEFFORT] = pd.effort / EFFORT_LIMIT  # Normalized
        obs[IDX_PREV_ACT] = prev_action

        all_obs[t] = obs

        # Forward pass
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = model(obs_t).squeeze(0).numpy()

        all_actions[t] = action

        # Scale action and compute target position
        target_pos = DEFAULT_JOINT_POS + action * ACTION_SCALE

        # Step PD actuator
        pos, vel, eff = pd.step(target_pos)

        # Record thigh joints
        all_thigh_pos[t] = pos[THIGH_IDX]
        all_thigh_vel[t] = vel[THIGH_IDX]

        prev_action = action

    return all_obs, all_actions, all_thigh_pos, all_thigh_vel


# ─── Analysis Functions ───────────────────────────────────────────────────────

def compute_mlp_recovery(nominal_pos, perturbed_pos, perturb_step, window=10):
    """MLP metric: % return to original gait (RMSE between nominal and perturbed).

    Appropriate for MLP because it returns to the same gait.
    """
    n_post = len(nominal_pos) - perturb_step
    if n_post < window * 2:
        return np.zeros(4), np.full(4, -1), np.zeros((1, 4))

    rmse_over_time = np.zeros((n_post, 4))
    for i in range(n_post):
        start = perturb_step + i
        end = min(start + window, len(nominal_pos))
        diff = nominal_pos[start:end] - perturbed_pos[start:end]
        rmse_over_time[i] = np.sqrt(np.mean(diff**2, axis=0))

    initial_rmse = rmse_over_time[:5].mean(axis=0)
    final_rmse = rmse_over_time[-10:].mean(axis=0)

    recovery_pct = np.where(
        initial_rmse > 1e-8,
        np.clip((1 - final_rmse / initial_rmse) * 100, 0, 100),
        100.0
    )

    # Convergence time: when RMSE drops below 20% of initial
    threshold = initial_rmse * 0.2
    convergence_steps = np.full(4, -1, dtype=int)
    for j in range(4):
        if initial_rmse[j] < 1e-8:
            convergence_steps[j] = 0
            continue
        for i in range(len(rmse_over_time)):
            if rmse_over_time[i, j] < threshold[j]:
                convergence_steps[j] = i
                break

    return recovery_pct, convergence_steps, rmse_over_time


def _get_period_from_signal(sig, dt=DT):
    """Estimate period from autocorrelation of a 1D signal."""
    sig = sig - sig.mean()
    if np.std(sig) < 1e-8:
        return 0.0
    acf = np.correlate(sig, sig, mode='full')[len(sig)-1:]
    acf /= acf[0] if acf[0] != 0 else 1
    crossed = False
    for k in range(1, len(acf) - 1):
        if acf[k] < 0:
            crossed = True
        if crossed and acf[k] > acf[k-1] and acf[k] > acf[k+1] and acf[k] > 0.1:
            return k * dt
    return 0.0


def compute_lstm_gait_adaptation(nominal_pos, perturbed_pos, perturb_step, window=25):
    """LSTM metric: characterize gait CHANGE rather than recovery.

    Measures:
    - Pre-perturbation gait properties (freq, amplitude per leg)
    - Post-perturbation gait properties (after settling)
    - Rolling gait stability (how quickly new gait becomes periodic)
    - Time to new stable gait
    """
    n_pre = perturb_step
    n_post = len(perturbed_pos) - perturb_step

    # Pre-perturbation gait properties (from nominal, last 2s before perturbation)
    pre_start = max(0, perturb_step - int(2.0 / DT))
    pre_data = nominal_pos[pre_start:perturb_step]
    pre_amp = np.std(pre_data, axis=0)
    pre_mean = np.mean(pre_data, axis=0)
    pre_periods = np.array([_get_period_from_signal(pre_data[:, j]) for j in range(4)])

    # Post-perturbation: use the last 3s of the perturbed run
    settle_start = max(perturb_step, len(perturbed_pos) - int(3.0 / DT))
    post_data = perturbed_pos[settle_start:]
    post_amp = np.std(post_data, axis=0)
    post_mean = np.mean(post_data, axis=0)
    post_periods = np.array([_get_period_from_signal(post_data[:, j]) for j in range(4)])

    # Rolling gait stability: std-of-std in sliding windows
    # Low value = periodic/stable, high value = transient/adapting
    roll_stability = np.zeros((n_post, 4))
    for i in range(n_post):
        start = perturb_step + i
        end = min(start + window, len(perturbed_pos))
        chunk = perturbed_pos[start:end]
        # Stability = how consistent the amplitude is within this window
        if len(chunk) >= 4:
            # Split into halves, compare amplitude
            half = len(chunk) // 2
            amp1 = np.std(chunk[:half], axis=0)
            amp2 = np.std(chunk[half:], axis=0)
            roll_stability[i] = np.abs(amp1 - amp2) / (np.maximum(amp1, amp2) + 1e-8)
        else:
            roll_stability[i] = 1.0  # Unstable by default

    # Time to new stable gait: when rolling stability drops below threshold
    stability_threshold = 0.15  # 15% amplitude variation = "stable"
    settle_steps = np.full(4, -1, dtype=int)
    sustained = 10  # Need 10 consecutive stable windows
    for j in range(4):
        for i in range(len(roll_stability) - sustained):
            if np.all(roll_stability[i:i+sustained, j] < stability_threshold):
                settle_steps[j] = i
                break

    return {
        'pre_amp': pre_amp,
        'pre_mean': pre_mean,
        'pre_period': pre_periods,
        'post_amp': post_amp,
        'post_mean': post_mean,
        'post_period': post_periods,
        'roll_stability': roll_stability,
        'settle_steps': settle_steps,
        'amp_change_pct': np.where(pre_amp > 1e-8,
                                    (post_amp - pre_amp) / pre_amp * 100,
                                    0.0),
        'freq_change_pct': np.where(pre_periods > 1e-8,
                                     (post_periods - pre_periods) / pre_periods * 100,
                                     0.0),
    }


def compute_input_sensitivity(model, baseline_obs, is_lstm=False):
    """Ablation study: zero out each observation group and measure action change.

    Proves whether the network relies on feedback (joint state) or just
    generates open-loop patterns (CPG behavior).
    """
    groups = {
        'Base Lin Vel': IDX_LIN_VEL,
        'Base Ang Vel': IDX_ANG_VEL,
        'Gravity':      IDX_GRAVITY,
        'Commands':     IDX_CMD,
        'Joint Pos':    IDX_JPOS,
        'Joint Vel':    IDX_JVEL,
        'Joint Effort': IDX_JEFFORT,
        'Prev Actions': IDX_PREV_ACT,
    }

    obs_t = torch.tensor(baseline_obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        baseline_action = model(obs_t).squeeze(0).numpy()

    sensitivities = {}
    thigh_sensitivities = {}

    for name, idx in groups.items():
        ablated = baseline_obs.copy()
        ablated[idx] = 0.0
        ablated_t = torch.tensor(ablated, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            ablated_action = model(ablated_t).squeeze(0).numpy()

        diff = np.abs(baseline_action - ablated_action)
        sensitivities[name] = np.mean(diff)
        thigh_sensitivities[name] = np.mean(diff[THIGH_IDX])

    return sensitivities, thigh_sensitivities, baseline_action


def compute_multi_step_sensitivity(model, all_obs, is_lstm=False):
    """Average sensitivity across many timesteps for robustness."""
    groups = {
        'Base Lin Vel': IDX_LIN_VEL,
        'Base Ang Vel': IDX_ANG_VEL,
        'Gravity':      IDX_GRAVITY,
        'Commands':     IDX_CMD,
        'Joint Pos':    IDX_JPOS,
        'Joint Vel':    IDX_JVEL,
        'Joint Effort': IDX_JEFFORT,
        'Prev Actions': IDX_PREV_ACT,
    }

    # Sample 50 timesteps from stable region
    stable_start = int(N_STEPS * 0.3)
    sample_idx = np.linspace(stable_start, N_STEPS - 1, 50, dtype=int)

    avg_sens = {k: 0.0 for k in groups}
    avg_thigh_sens = {k: 0.0 for k in groups}

    for idx in sample_idx:
        obs = all_obs[idx]
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            baseline = model(obs_t).squeeze(0).numpy()

        for name, slc in groups.items():
            ablated = obs.copy()
            ablated[slc] = 0.0
            abl_t = torch.tensor(ablated, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                abl_action = model(abl_t).squeeze(0).numpy()
            diff = np.abs(baseline - abl_action)
            avg_sens[name] += np.mean(diff)
            avg_thigh_sens[name] += np.mean(diff[THIGH_IDX])

    for k in groups:
        avg_sens[k] /= len(sample_idx)
        avg_thigh_sens[k] /= len(sample_idx)

    return avg_sens, avg_thigh_sens


# ─── Plotting ─────────────────────────────────────────────────────────────────

def normalize_thigh(data):
    """Zero-mean, unit-std normalization per joint for visual comparison."""
    out = np.zeros_like(data)
    for j in range(data.shape[1]):
        mu = data[:, j].mean()
        sigma = data[:, j].std()
        out[:, j] = (data[:, j] - mu) / sigma if sigma > 1e-8 else data[:, j] - mu
    return out


def page1_normalized_gaits(fig, tp_mlp, tp_lstm):
    """Normalized thigh trajectories — steady-state only."""
    # Skip first 5s for onset transients (gait stabilizes ~5s)
    skip = int(5.0 / DT)
    stable_mlp = normalize_thigh(tp_mlp[skip:])
    stable_lstm = normalize_thigh(tp_lstm[skip:])
    t_mlp = np.arange(len(stable_mlp)) * DT
    t_lstm = np.arange(len(stable_lstm)) * DT

    gs = GridSpec(2, 1, hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    for i, leg in enumerate(LEG_NAMES):
        ax1.plot(t_mlp, stable_mlp[:, i], color=LEG_COLORS[leg],
                 label=LEG_FULL[leg], alpha=0.85)
    ax1.set_title('MLP — Normalized Thigh Trajectories (Steady-State)', fontweight='bold')
    ax1.set_ylabel('Normalized Position (z-score)')
    ax1.legend(loc='upper right', ncol=4, framealpha=0.9)
    ax1.set_xlim(0, t_mlp[-1])

    ax2 = fig.add_subplot(gs[1])
    for i, leg in enumerate(LEG_NAMES):
        ax2.plot(t_lstm, stable_lstm[:, i], color=LEG_COLORS[leg],
                 label=LEG_FULL[leg], alpha=0.85)
    ax2.set_title('LSTM — Normalized Thigh Trajectories (Steady-State)', fontweight='bold')
    ax2.set_ylabel('Normalized Position (z-score)')
    ax2.set_xlabel('Time (s)')
    ax2.legend(loc='upper right', ncol=4, framealpha=0.9)
    ax2.set_xlim(0, t_lstm[-1])

    fig.suptitle('Steady-State CPG: Normalized Thigh Oscillations (Open-Loop, 20s)',
                 fontsize=13, fontweight='bold', y=0.98)


def page2_per_leg_overlay(fig, tp_mlp, tp_lstm):
    """Per-leg normalized overlay MLP vs LSTM."""
    skip = int(1.5 / DT)
    stable_mlp = normalize_thigh(tp_mlp[skip:])
    stable_lstm = normalize_thigh(tp_lstm[skip:])
    T = min(len(stable_mlp), len(stable_lstm))
    t = np.arange(T) * DT

    gs = GridSpec(4, 1, hspace=0.4)

    for i, leg in enumerate(LEG_NAMES):
        ax = fig.add_subplot(gs[i])
        ax.plot(t, stable_mlp[:T, i], color='#1f77b4', label='MLP', alpha=0.85)
        ax.plot(t, stable_lstm[:T, i], color='#d62728', label='LSTM', alpha=0.85, ls='--')
        ax.set_title(f'{LEG_FULL[leg]} Thigh (Normalized)', fontweight='bold', fontsize=11)
        ax.set_ylabel('z-score')
        if i == 0:
            ax.legend(loc='upper right', ncol=2)
        if i == 3:
            ax.set_xlabel('Time (s)')

    fig.suptitle('MLP vs LSTM: Per-Leg Normalized Comparison',
                 fontsize=13, fontweight='bold', y=0.98)


def page3_perturbation(fig, tp_nom_mlp, tp_pert_mlp, tp_nom_lstm, tp_pert_lstm):
    """Mid-stride perturbation at t=2.0s — all 4 thighs."""
    t = np.arange(N_STEPS) * DT

    gs = GridSpec(4, 2, hspace=0.45, wspace=0.3)

    for i, leg in enumerate(LEG_NAMES):
        # MLP
        ax_m = fig.add_subplot(gs[i, 0])
        ax_m.plot(t, tp_nom_mlp[:, i], color='#1f77b4', alpha=0.8, label='Nominal')
        ax_m.plot(t, tp_pert_mlp[:, i], color='#d62728', alpha=0.7, ls='--', label='+160ms delay')
        ax_m.axvline(PERTURB_TIME, color='gray', ls=':', alpha=0.6)
        ax_m.set_title(f'MLP — {LEG_FULL[leg]}', fontweight='bold', fontsize=10)
        ax_m.set_ylabel('Pos (rad)', fontsize=8)
        if i == 0:
            ax_m.legend(fontsize=7, ncol=2)
        if i == 3:
            ax_m.set_xlabel('Time (s)')

        # LSTM
        ax_l = fig.add_subplot(gs[i, 1])
        ax_l.plot(t, tp_nom_lstm[:, i], color='#1f77b4', alpha=0.8, label='Nominal')
        ax_l.plot(t, tp_pert_lstm[:, i], color='#d62728', alpha=0.7, ls='--', label='+160ms delay')
        ax_l.axvline(PERTURB_TIME, color='gray', ls=':', alpha=0.6)
        ax_l.set_title(f'LSTM — {LEG_FULL[leg]}', fontweight='bold', fontsize=10)
        if i == 0:
            ax_l.legend(fontsize=7, ncol=2)
        if i == 3:
            ax_l.set_xlabel('Time (s)')

    fig.suptitle(f'Mid-Stride Perturbation at t={PERTURB_TIME}s: +160ms Delay Injection',
                 fontsize=13, fontweight='bold', y=0.98)


def page4_mlp_recovery(fig, tp_nom_mlp, tp_pert_mlp):
    """MLP: returns to original gait — show RMSE convergence back to nominal."""
    rec_pct, conv_steps, rmse_time = compute_mlp_recovery(
        tp_nom_mlp, tp_pert_mlp, PERTURB_STEP)
    t_post = np.arange(len(rmse_time)) * DT

    gs = GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # Rolling RMSE over time
    ax1 = fig.add_subplot(gs[0, :])
    for j, leg in enumerate(LEG_NAMES):
        ax1.plot(t_post, rmse_time[:, j], color=LEG_COLORS[leg], alpha=0.8, label=leg)
        if conv_steps[j] >= 0:
            ax1.axvline(conv_steps[j] * DT, color=LEG_COLORS[leg], ls=':', alpha=0.4)
    ax1.set_title('MLP — RMSE vs Nominal After Perturbation (Returns to Same Gait)',
                  fontweight='bold')
    ax1.set_ylabel('RMSE (rad)')
    ax1.set_xlabel('Time After Perturbation (s)')
    ax1.legend(fontsize=8, ncol=4)

    # Recovery bar chart
    ax2 = fig.add_subplot(gs[1, 0])
    x = np.arange(4)
    bars = ax2.bar(x, rec_pct, color=[LEG_COLORS[l] for l in LEG_NAMES], alpha=0.7)
    ax2.set_title('MLP: % Return to Original Gait', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(LEG_NAMES)
    ax2.set_ylabel('Recovery %')
    ax2.set_ylim(0, 110)
    for bar, pct in zip(bars, rec_pct):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{pct:.0f}%', ha='center', fontsize=9, fontweight='bold')

    # Summary
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    conv_ms = conv_steps * DT * 1000
    summary = (
        "MLP Recovery Summary\n"
        "───────────────────────\n\n"
        f"Perturbation: +160ms delay at t={PERTURB_TIME}s\n\n"
        f"Avg recovery: {np.mean(rec_pct):.1f}%\n\n"
    )
    for j, leg in enumerate(LEG_NAMES):
        ct = f"{conv_ms[j]:.0f}ms" if conv_steps[j] >= 0 else "never"
        summary += f"  {leg}: {rec_pct[j]:.1f}%  converge: {ct}\n"
    summary += (
        "\nMLP returns to its original gait\n"
        "because it has no internal state —\n"
        "same PD feedback → same limit cycle."
    )
    ax3.text(0.02, 0.98, summary, transform=ax3.transAxes,
             fontsize=9.5, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#d4e6f1', alpha=0.6))

    fig.suptitle('MLP Perturbation Response: Return to Original Gait',
                 fontsize=13, fontweight='bold', y=0.98)


def _compute_periodicity_trace(tp_data, start_step, n_steps, total_steps):
    """Compute rolling periodicity (autocorrelation peak) for all 4 thigh joints."""
    window = int(1.0 / DT)  # 1s windows
    periodicity = np.zeros((n_steps, 4))
    for j in range(4):
        for k in range(n_steps):
            s = start_step + k
            e = min(s + window, total_steps)
            chunk = tp_data[s:e, j]
            if len(chunk) < 10:
                continue
            sig = chunk - chunk.mean()
            if np.std(sig) < 1e-8:
                continue
            acf = np.correlate(sig, sig, mode='full')[len(sig)-1:]
            acf /= acf[0] if acf[0] > 0 else 1
            crossed = False
            for m in range(1, len(acf) - 1):
                if acf[m] < 0:
                    crossed = True
                if crossed and acf[m] > acf[m-1] and acf[m] > acf[m+1]:
                    periodicity[k, j] = max(0, acf[m])
                    break
    return periodicity


def _gait_quality_metrics(tp_data, start_step, end_step):
    """Compute gait quality metrics over a window.

    Returns dict with:
    - amplitude: oscillation amplitude per leg (std)
    - periodicity: mean autocorrelation peak per leg
    - symmetry: L/R amplitude ratio (1.0 = perfect)
    - mean_pos: mean position per leg
    """
    data = tp_data[start_step:end_step]
    amp = np.std(data, axis=0)

    # Periodicity from autocorrelation
    period_scores = np.zeros(4)
    for j in range(4):
        sig = data[:, j] - data[:, j].mean()
        if np.std(sig) < 1e-8:
            continue
        acf = np.correlate(sig, sig, mode='full')[len(sig)-1:]
        acf /= acf[0] if acf[0] > 0 else 1
        crossed = False
        for k in range(1, len(acf) - 1):
            if acf[k] < 0:
                crossed = True
            if crossed and acf[k] > acf[k-1] and acf[k] > acf[k+1]:
                period_scores[j] = max(0, acf[k])
                break

    # Symmetry: front L/R and back L/R
    front_sym = min(amp[0], amp[1]) / (max(amp[0], amp[1]) + 1e-8)
    back_sym = min(amp[2], amp[3]) / (max(amp[2], amp[3]) + 1e-8)

    return {
        'amplitude': amp,
        'periodicity': period_scores,
        'front_symmetry': front_sym,
        'back_symmetry': back_sym,
        'mean_pos': np.mean(data, axis=0),
    }


def page4_adaptation_quality(fig, tp_nom_mlp, tp_pert_mlp, tp_nom_lstm, tp_pert_lstm):
    """Unified adaptation analysis: both MLP and LSTM gait quality before vs after."""
    gs = GridSpec(2, 2, hspace=0.55, wspace=0.35, height_ratios=[1, 1.2])

    # Define windows: last 3s before perturbation, last 3s of sim
    pre_start = PERTURB_STEP - int(3.0 / DT)
    post_start = N_STEPS - int(3.0 / DT)
    pre_len = int(3.0 / DT)

    # Compute metrics for all four conditions
    mlp_pre = _gait_quality_metrics(tp_nom_mlp, pre_start, PERTURB_STEP)
    mlp_post = _gait_quality_metrics(tp_pert_mlp, post_start, N_STEPS)
    lstm_pre = _gait_quality_metrics(tp_nom_lstm, pre_start, PERTURB_STEP)
    lstm_post = _gait_quality_metrics(tp_pert_lstm, post_start, N_STEPS)

    # Row 1: Old vs New gait overlay (last 3s each), MLP left, LSTM right
    t_win = np.arange(pre_len) * DT
    for col, (nom, pert, pre_m, post_m, name) in enumerate([
        (tp_nom_mlp, tp_pert_mlp, mlp_pre, mlp_post, 'MLP'),
        (tp_nom_lstm, tp_pert_lstm, lstm_pre, lstm_post, 'LSTM'),
    ]):
        ax = fig.add_subplot(gs[0, col])
        # Show LF thigh as representative
        pre_data = nom[pre_start:PERTURB_STEP, 0]
        post_data = pert[post_start:N_STEPS, 0]
        ax.plot(t_win, pre_data, color='#1f77b4', alpha=0.85, label='Pre (LF)')
        ax.plot(t_win, post_data, color='#d62728', alpha=0.85, ls='--', label='Post (LF)')
        ax.set_title(f'{name} — LF Thigh: Old vs New Gait (3s windows)',
                     fontweight='bold', fontsize=10)
        ax.set_ylabel('Position (rad)', fontsize=8)
        ax.set_xlabel('Time (s)')
        ax.legend(fontsize=7)

    # Row 2: Adaptation quality scorecard
    ax_txt = fig.add_subplot(gs[1, :])
    ax_txt.axis('off')

    def _score_line(name, pre_m, post_m):
        lines = f"  {name}:\n"
        for j, leg in enumerate(LEG_NAMES):
            amp_pre = pre_m['amplitude'][j]
            amp_post = post_m['amplitude'][j]
            amp_delta = (amp_post - amp_pre) / (amp_pre + 1e-8) * 100
            per_pre = pre_m['periodicity'][j]
            per_post = post_m['periodicity'][j]
            mean_shift = abs(post_m['mean_pos'][j] - pre_m['mean_pos'][j])
            lines += (f"    {leg}: amp {amp_pre:.3f}→{amp_post:.3f} ({amp_delta:+.0f}%)  "
                      f"period {per_pre:.2f}→{per_post:.2f}  "
                      f"drift {mean_shift:.3f}r\n")
        fsym_pre = pre_m['front_symmetry']
        fsym_post = post_m['front_symmetry']
        bsym_pre = pre_m['back_symmetry']
        bsym_post = post_m['back_symmetry']
        lines += (f"    Symmetry: front {fsym_pre:.2f}→{fsym_post:.2f}  "
                  f"back {bsym_pre:.2f}→{bsym_post:.2f}\n")

        # Overall quality score (higher = better adaptation)
        # Weighted: periodicity (most important), symmetry, amplitude preservation
        per_score = np.mean(post_m['periodicity'])
        sym_score = (fsym_post + bsym_post) / 2
        amp_score = 1.0 - min(1.0, np.mean(np.abs(
            post_m['amplitude'] - pre_m['amplitude'])) / (np.mean(pre_m['amplitude']) + 1e-8))
        overall = per_score * 0.5 + sym_score * 0.3 + amp_score * 0.2
        lines += f"    QUALITY SCORE: {overall:.2f} (periodicity={per_score:.2f} sym={sym_score:.2f} amp={amp_score:.2f})\n"
        return lines, overall

    summary = (
        "Adaptation Quality Scorecard\n"
        "════════════════════════════════════════════════════════════════════\n"
        f"Perturbation: +160ms delay at t={PERTURB_TIME}s\n"
        f"Comparing last 3s pre-perturbation vs last 3s post-perturbation\n\n"
    )

    mlp_lines, mlp_score = _score_line('MLP', mlp_pre, mlp_post)
    lstm_lines, lstm_score = _score_line('LSTM', lstm_pre, lstm_post)
    summary += mlp_lines + "\n" + lstm_lines

    winner = 'LSTM' if lstm_score > mlp_score else 'MLP'
    summary += (
        f"\n  VERDICT: {winner} adapts better "
        f"(MLP={mlp_score:.2f} vs LSTM={lstm_score:.2f})\n"
        f"  Both networks settle into new gaits — "
        f"the question is which new gait is better."
    )

    ax_txt.text(0.02, 0.98, summary, transform=ax_txt.transAxes,
                fontsize=8.5, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.6))

    fig.suptitle('Adaptation Quality: MLP vs LSTM Post-Perturbation Gait Assessment',
                 fontsize=13, fontweight='bold', y=0.98)


def page_attractor_trajectories(fig, tp_nom_mlp, tv_nom_mlp, tp_pert_mlp, tv_pert_mlp,
                                  tp_nom_lstm, tv_nom_lstm, tp_pert_lstm, tv_pert_lstm):
    """Phase portraits (position vs velocity) for the most oscillatory thigh per network.

    Shows limit-cycle attractors: stable pre-perturbation orbit vs post-perturbation orbit.
    The most oscillatory leg is selected by highest amplitude (std) in the pre-perturbation window.
    """
    gs = GridSpec(2, 3, hspace=0.40, wspace=0.35,
                  width_ratios=[1.2, 1.2, 0.8])

    # Windows: last 4s before perturbation, last 4s of sim (well within stable region)
    pre_start = PERTURB_STEP - int(4.0 / DT)
    pre_end = PERTURB_STEP
    post_start = N_STEPS - int(4.0 / DT)
    post_end = N_STEPS

    datasets = [
        ('MLP', tp_nom_mlp, tv_nom_mlp, tp_pert_mlp, tv_pert_mlp, 0),
        ('LSTM', tp_nom_lstm, tv_nom_lstm, tp_pert_lstm, tv_pert_lstm, 1),
    ]

    for label, tp_nom, tv_nom, tp_pert, tv_pert, row in datasets:
        # Find most oscillatory thigh (highest amplitude in pre-perturbation nominal)
        amp_pre = np.std(tp_nom[pre_start:pre_end], axis=0)
        best_leg = int(np.argmax(amp_pre))
        leg_name = LEG_NAMES[best_leg]
        leg_color = LEG_COLORS[leg_name]

        # Extract data
        pos_pre = tp_nom[pre_start:pre_end, best_leg]
        vel_pre = tv_nom[pre_start:pre_end, best_leg]
        pos_post_nom = tp_nom[post_start:post_end, best_leg]
        vel_post_nom = tv_nom[post_start:post_end, best_leg]
        pos_post_pert = tp_pert[post_start:post_end, best_leg]
        vel_post_pert = tv_pert[post_start:post_end, best_leg]

        # Plot 1: Nominal pre vs post (should be identical for MLP, may drift for LSTM)
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.plot(pos_pre, vel_pre, color=leg_color, alpha=0.6, linewidth=0.8,
                 label=f'Pre ({PERTURB_TIME-4:.0f}–{PERTURB_TIME:.0f}s)')
        ax1.plot(pos_post_nom, vel_post_nom, color='gray', alpha=0.5, linewidth=0.8,
                 linestyle='--', label=f'Post nominal ({SIM_SECONDS-4:.0f}–{SIM_SECONDS:.0f}s)')
        ax1.set_xlabel('Thigh Position (rad)')
        ax1.set_ylabel('Thigh Velocity (rad/s)')
        ax1.set_title(f'{label} — {leg_name} Thigh Nominal Orbit', fontweight='bold')
        ax1.legend(fontsize=7, loc='upper right')
        # No equal aspect — velocity range >> position range

        # Plot 2: Pre-perturbation vs Post-perturbation
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.plot(pos_pre, vel_pre, color=leg_color, alpha=0.6, linewidth=0.8,
                 label=f'Pre-perturbation')
        ax2.plot(pos_post_pert, vel_post_pert, color='#ff7f0e', alpha=0.7, linewidth=0.8,
                 label=f'Post-perturbation (+160ms)')
        # Mark start points
        ax2.scatter(pos_pre[0], vel_pre[0], color=leg_color, s=30, zorder=5, marker='o')
        ax2.scatter(pos_post_pert[0], vel_post_pert[0], color='#ff7f0e', s=30, zorder=5, marker='o')
        ax2.set_xlabel('Thigh Position (rad)')
        ax2.set_ylabel('Thigh Velocity (rad/s)')
        ax2.set_title(f'{label} — {leg_name} Thigh: Pre vs Perturbed', fontweight='bold')
        ax2.legend(fontsize=7, loc='upper right')
        # No equal aspect — velocity range >> position range

        # Plot 3: Metrics text
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.axis('off')

        # Compute orbit metrics
        amp_pre_val = np.std(pos_pre)
        amp_post_val = np.std(pos_post_pert)
        vel_amp_pre = np.std(vel_pre)
        vel_amp_post = np.std(vel_post_pert)

        # Orbit area (bounding box proxy)
        area_pre = (pos_pre.max() - pos_pre.min()) * (vel_pre.max() - vel_pre.min())
        area_post = (pos_post_pert.max() - pos_post_pert.min()) * (vel_post_pert.max() - vel_post_pert.min())

        # Center shift
        center_pre = (np.mean(pos_pre), np.mean(vel_pre))
        center_post = (np.mean(pos_post_pert), np.mean(vel_post_pert))
        center_shift = np.sqrt((center_post[0] - center_pre[0])**2 +
                                (center_post[1] - center_pre[1])**2)

        metrics_text = (
            f"Most oscillatory: {LEG_FULL[leg_name]}\n"
            f"(amplitude = {amp_pre[best_leg]:.4f} rad)\n\n"
            f"{'Metric':<18s} {'Pre':>7s} {'Post':>7s}\n"
            f"{'─'*34}\n"
            f"{'Pos amp (std)':.<18s} {amp_pre_val:.4f}  {amp_post_val:.4f}\n"
            f"{'Vel amp (std)':.<18s} {vel_amp_pre:.3f}  {vel_amp_post:.3f}\n"
            f"{'Orbit area':.<18s} {area_pre:.4f}  {area_post:.4f}\n"
            f"{'Center shift':.<18s} {center_shift:.4f} rad\n\n"
            f"{'Amp change':.<18s} {(amp_post_val/amp_pre_val - 1)*100:+.1f}%\n"
            f"{'Area change':.<18s} {(area_post/area_pre - 1)*100:+.1f}%"
        )
        ax3.text(0.05, 0.95, metrics_text, transform=ax3.transAxes,
                 fontsize=8.5, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.6))

    fig.suptitle('Attractor Trajectories: Limit-Cycle Phase Portraits (Most Oscillatory Thigh)',
                 fontsize=13, fontweight='bold', y=0.99)


def page5_input_ablation(fig, sens_mlp, sens_lstm, thigh_sens_mlp, thigh_sens_lstm):
    """Input ablation: which observation groups drive thigh actions?"""
    groups = list(sens_mlp.keys())
    n = len(groups)
    x = np.arange(n)
    w = 0.35

    gs = GridSpec(2, 1, hspace=0.45)

    # All joints
    ax1 = fig.add_subplot(gs[0])
    vals_mlp = [sens_mlp[g] for g in groups]
    vals_lstm = [sens_lstm[g] for g in groups]
    ax1.barh(x - w/2, vals_mlp, w, label='MLP', color='#1f77b4', alpha=0.7)
    ax1.barh(x + w/2, vals_lstm, w, label='LSTM', color='#d62728', alpha=0.7)
    ax1.set_yticks(x)
    ax1.set_yticklabels(groups)
    ax1.set_xlabel('Mean |Action Change| When Group Zeroed')
    ax1.set_title('Input Ablation: All Joints', fontweight='bold')
    ax1.legend()
    ax1.invert_yaxis()

    # Thigh joints only
    ax2 = fig.add_subplot(gs[1])
    tvals_mlp = [thigh_sens_mlp[g] for g in groups]
    tvals_lstm = [thigh_sens_lstm[g] for g in groups]
    ax2.barh(x - w/2, tvals_mlp, w, label='MLP', color='#1f77b4', alpha=0.7)
    ax2.barh(x + w/2, tvals_lstm, w, label='LSTM', color='#d62728', alpha=0.7)
    ax2.set_yticks(x)
    ax2.set_yticklabels(groups)
    ax2.set_xlabel('Mean |Thigh Action Change| When Group Zeroed')
    ax2.set_title('Input Ablation: Thigh Joints Only', fontweight='bold')
    ax2.legend()
    ax2.invert_yaxis()

    fig.suptitle('CPG Proof: Input Sensitivity Analysis\n'
                 '(High sensitivity to Prev Actions + low to Joint State = open-loop CPG)',
                 fontsize=12, fontweight='bold', y=0.99)


def page6_cpg_proof(fig, all_obs_mlp, all_actions_mlp, all_obs_lstm, all_actions_lstm,
                     model_mlp, model_lstm):
    """The smoking gun: run MLP with FROZEN observations (constant input).

    If the output still oscillates → the network IS a CPG (oscillation is
    internal, not feedback-driven).
    """
    skip = int(10.0 / DT)  # Use a mid-stride observation from stable region

    gs = GridSpec(2, 2, hspace=0.45, wspace=0.3)

    # Freeze MLP at a single observation
    frozen_obs = all_obs_mlp[skip].copy()
    frozen_actions = np.zeros((100, 12))  # Run 100 steps with same input
    for t in range(100):
        obs_t = torch.tensor(frozen_obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = model_mlp(obs_t).squeeze(0).numpy()
        frozen_actions[t] = action

    t_frozen = np.arange(100) * DT

    ax1 = fig.add_subplot(gs[0, 0])
    for i, leg in enumerate(LEG_NAMES):
        ax1.plot(t_frozen, frozen_actions[:, THIGH_IDX[i]], color=LEG_COLORS[leg],
                 alpha=0.8, label=leg)
    ax1.set_title('MLP — Frozen Input (Same Obs Every Step)', fontweight='bold')
    ax1.set_ylabel('Thigh Action')
    ax1.set_xlabel('Time (s)')
    ax1.legend(fontsize=7, ncol=4)

    # MLP with evolving PD feedback
    ax2 = fig.add_subplot(gs[0, 1])
    t_full = np.arange(N_STEPS) * DT
    for i, leg in enumerate(LEG_NAMES):
        ax2.plot(t_full[skip:], all_actions_mlp[skip:, THIGH_IDX[i]],
                 color=LEG_COLORS[leg], alpha=0.8, label=leg)
    ax2.set_title('MLP — Normal (PD Feedback Loop)', fontweight='bold')
    ax2.set_ylabel('Thigh Action')
    ax2.set_xlabel('Time (s)')
    ax2.legend(fontsize=7, ncol=4)

    # Freeze LSTM
    frozen_obs_lstm = all_obs_lstm[skip].copy()
    model_lstm.hidden_state.zero_()
    model_lstm.cell_state.zero_()
    frozen_actions_lstm = np.zeros((100, 12))
    for t in range(100):
        obs_t = torch.tensor(frozen_obs_lstm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = model_lstm(obs_t).squeeze(0).numpy()
        frozen_actions_lstm[t] = action

    ax3 = fig.add_subplot(gs[1, 0])
    for i, leg in enumerate(LEG_NAMES):
        ax3.plot(t_frozen, frozen_actions_lstm[:, THIGH_IDX[i]], color=LEG_COLORS[leg],
                 alpha=0.8, label=leg)
    ax3.set_title('LSTM — Frozen Input (Same Obs Every Step)', fontweight='bold')
    ax3.set_ylabel('Thigh Action')
    ax3.set_xlabel('Time (s)')
    ax3.legend(fontsize=7, ncol=4)

    ax4 = fig.add_subplot(gs[1, 1])
    for i, leg in enumerate(LEG_NAMES):
        ax4.plot(t_full[skip:], all_actions_lstm[skip:, THIGH_IDX[i]],
                 color=LEG_COLORS[leg], alpha=0.8, label=leg)
    ax4.set_title('LSTM — Normal (PD Feedback Loop)', fontweight='bold')
    ax4.set_ylabel('Thigh Action')
    ax4.set_xlabel('Time (s)')
    ax4.legend(fontsize=7, ncol=4)

    fig.suptitle('CPG Proof: Frozen Input Test\n'
                 'MLP constant output = open-loop CPG (oscillation from PD feedback loop)\n'
                 'LSTM drifting output = recurrent dynamics generate internal oscillation',
                 fontsize=11, fontweight='bold', y=1.01)


def page7_mlp_diagram(fig):
    """Visual network diagram showing input→output mapping."""
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 11)
    ax.axis('off')

    # Input groups with importance coloring
    input_groups = [
        ('Base Lin Vel [3]',    0.5, 9.5, '#cccccc'),
        ('Base Ang Vel [3]',    0.5, 8.5, '#cccccc'),
        ('Gravity [3]',         0.5, 7.5, '#aaaaaa'),
        ('Commands [3]',        0.5, 6.5, '#cccccc'),
        ('Joint Pos [12]',      0.5, 5.5, '#ff9999'),
        ('Joint Vel [12]',      0.5, 4.5, '#ff9999'),
        ('Joint Effort [12]',   0.5, 3.5, '#ff9999'),
        ('Prev Actions [12]',   0.5, 2.5, '#ff4444'),
    ]

    # Draw input boxes
    for name, x, y, color in input_groups:
        rect = plt.Rectangle((x-0.4, y-0.3), 2.5, 0.6,
                              facecolor=color, edgecolor='black', alpha=0.6, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + 0.85, y, name, ha='center', va='center', fontsize=8, fontweight='bold')

    # Hidden layers
    hidden = [
        ('512 (ELU)', 4.5, 6.0),
        ('256 (ELU)', 6.0, 6.0),
        ('128 (ELU)', 7.5, 6.0),
    ]
    for name, x, y in hidden:
        circle = plt.Circle((x, y), 0.5, facecolor='#4488ff', edgecolor='black',
                            alpha=0.5, linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=7, fontweight='bold')

    # Output
    rect = plt.Rectangle((8.6, 5.0), 1.8, 2.0,
                          facecolor='#44cc44', edgecolor='black', alpha=0.5, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(9.5, 6.0, 'Actions\n[12]', ha='center', va='center', fontsize=10, fontweight='bold')

    # LSTM branch (for comparison)
    lstm_box = plt.Rectangle((3.5, 0.5), 1.8, 1.0,
                              facecolor='#ff8844', edgecolor='black', alpha=0.5, linewidth=1.5)
    ax.add_patch(lstm_box)
    ax.text(4.4, 1.0, 'LSTM\n(128)', ha='center', va='center', fontsize=8, fontweight='bold')

    # Arrows: inputs → hidden
    for _, ix, iy, _ in input_groups:
        ax.annotate('', xy=(4.0, 6.0), xytext=(3.0, iy),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3, lw=0.8))

    # Arrows: hidden → hidden → output
    ax.annotate('', xy=(5.5, 6.0), xytext=(5.0, 6.0),
                arrowprops=dict(arrowstyle='->', color='#4488ff', lw=2))
    ax.annotate('', xy=(7.0, 6.0), xytext=(6.5, 6.0),
                arrowprops=dict(arrowstyle='->', color='#4488ff', lw=2))
    ax.annotate('', xy=(8.6, 6.0), xytext=(8.0, 6.0),
                arrowprops=dict(arrowstyle='->', color='#4488ff', lw=2))

    # Feedback loop arrow (prev_actions → output)
    ax.annotate('', xy=(0.5, 2.5), xytext=(9.5, 4.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5,
                                connectionstyle='arc3,rad=0.3', ls='--'))
    ax.text(5.0, 1.8, 'Prev Actions feedback\n(t-1 output → t input)',
            ha='center', fontsize=8, color='red', fontstyle='italic')

    # PD actuator box
    pd_box = plt.Rectangle((8.6, 2.5), 1.8, 1.5,
                            facecolor='#ffcc44', edgecolor='black', alpha=0.5, linewidth=1.5)
    ax.add_patch(pd_box)
    ax.text(9.5, 3.25, 'Synthetic\nPD Actuator\n(Kp=70, Kd=1.2)',
            ha='center', va='center', fontsize=7, fontweight='bold')

    ax.annotate('', xy=(9.5, 4.8), xytext=(9.5, 4.0),
                arrowprops=dict(arrowstyle='->', color='#cc8800', lw=2))

    # Joint state feedback
    ax.annotate('', xy=(0.5, 5.5), xytext=(8.6, 3.25),
                arrowprops=dict(arrowstyle='->', color='#cc8800', lw=1.5,
                                connectionstyle='arc3,rad=-0.2', ls='--'))
    ax.text(4.5, 3.8, 'Joint state feedback\n(pos, vel, effort from PD)',
            ha='center', fontsize=7, color='#886600', fontstyle='italic')

    # Title and key insight
    ax.text(5.25, 10.5,
            'MLP + PD Actuator System Diagram\n'
            'The MLP outputs are CONSTANT for constant input → oscillation emerges\n'
            'entirely from the PD feedback loop (prev_actions + joint state)',
            ha='center', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))

def page8_lstm_diagram(fig):
    """LSTM system diagram — shows recurrent state as internal oscillator."""
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 11)
    ax.axis('off')

    # Input groups
    input_groups = [
        ('Base Lin Vel [3]',    0.5, 9.5, '#cccccc'),
        ('Base Ang Vel [3]',    0.5, 8.5, '#cccccc'),
        ('Gravity [3]',         0.5, 7.5, '#aaaaaa'),
        ('Commands [3]',        0.5, 6.5, '#cccccc'),
        ('Joint Pos [12]',      0.5, 5.5, '#ff9999'),
        ('Joint Vel [12]',      0.5, 4.5, '#ff9999'),
        ('Joint Effort [12]',   0.5, 3.5, '#ff9999'),
        ('Prev Actions [12]',   0.5, 2.5, '#ff4444'),
    ]
    for name, x, y, color in input_groups:
        rect = plt.Rectangle((x-0.4, y-0.3), 2.5, 0.6,
                              facecolor=color, edgecolor='black', alpha=0.6, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + 0.85, y, name, ha='center', va='center', fontsize=8, fontweight='bold')

    # LSTM cell (the key difference)
    lstm_box = plt.Rectangle((3.3, 4.5), 2.0, 3.0,
                              facecolor='#ff8844', edgecolor='black', alpha=0.5, linewidth=2.0)
    ax.add_patch(lstm_box)
    ax.text(4.3, 6.0, 'LSTM\nCell\n(128)', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # Recurrent self-loop (the internal oscillator)
    ax.annotate('', xy=(3.3, 5.0), xytext=(3.3, 7.0),
                arrowprops=dict(arrowstyle='->', color='#cc4400', lw=3.0,
                                connectionstyle='arc3,rad=-1.2'))
    ax.text(2.0, 6.0, 'h(t-1)\nc(t-1)', ha='center', va='center',
            fontsize=8, fontweight='bold', color='#cc4400',
            bbox=dict(boxstyle='round', facecolor='#ffe0c0', edgecolor='#cc4400', alpha=0.8))

    # Hidden layers after LSTM
    hidden = [
        ('512 (ELU)', 6.0, 6.0),
        ('256 (ELU)', 7.2, 6.0),
        ('128 (ELU)', 8.4, 6.0),
    ]
    for name, x, y in hidden:
        circle = plt.Circle((x, y), 0.45, facecolor='#4488ff', edgecolor='black',
                            alpha=0.5, linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=7, fontweight='bold')

    # Output
    rect = plt.Rectangle((9.0, 5.2), 1.3, 1.6,
                          facecolor='#44cc44', edgecolor='black', alpha=0.5, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(9.65, 6.0, 'Actions\n[12]', ha='center', va='center',
            fontsize=9, fontweight='bold')

    # Arrows: inputs → LSTM
    for _, ix, iy, _ in input_groups:
        ax.annotate('', xy=(3.3, 6.0), xytext=(3.0, iy),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3, lw=0.8))

    # LSTM → hidden → output
    ax.annotate('', xy=(5.55, 6.0), xytext=(5.3, 6.0),
                arrowprops=dict(arrowstyle='->', color='#ff8844', lw=2))
    ax.annotate('', xy=(6.75, 6.0), xytext=(6.45, 6.0),
                arrowprops=dict(arrowstyle='->', color='#4488ff', lw=2))
    ax.annotate('', xy=(7.95, 6.0), xytext=(7.65, 6.0),
                arrowprops=dict(arrowstyle='->', color='#4488ff', lw=2))
    ax.annotate('', xy=(9.0, 6.0), xytext=(8.85, 6.0),
                arrowprops=dict(arrowstyle='->', color='#4488ff', lw=2))

    # PD actuator box
    pd_box = plt.Rectangle((8.5, 2.5), 1.8, 1.5,
                            facecolor='#ffcc44', edgecolor='black', alpha=0.5, linewidth=1.5)
    ax.add_patch(pd_box)
    ax.text(9.4, 3.25, 'Synthetic\nPD Actuator\n(Kp=70, Kd=1.2)',
            ha='center', va='center', fontsize=7, fontweight='bold')

    ax.annotate('', xy=(9.65, 5.0), xytext=(9.4, 4.0),
                arrowprops=dict(arrowstyle='->', color='#cc8800', lw=2))

    # Feedback loops
    ax.annotate('', xy=(0.5, 2.5), xytext=(9.65, 5.0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.0,
                                connectionstyle='arc3,rad=0.3', ls='--'))
    ax.text(5.0, 1.5, 'Prev Actions feedback (t-1 → t)',
            ha='center', fontsize=8, color='red', fontstyle='italic')

    ax.annotate('', xy=(0.5, 5.5), xytext=(8.5, 3.25),
                arrowprops=dict(arrowstyle='->', color='#cc8800', lw=1.5,
                                connectionstyle='arc3,rad=-0.2', ls='--'))
    ax.text(4.0, 3.5, 'Joint state feedback (pos, vel, effort)',
            ha='center', fontsize=7, color='#886600', fontstyle='italic')

    # Title
    ax.text(5.25, 10.5,
            'LSTM + PD Actuator System Diagram\n'
            'LSTM hidden state h(t), c(t) carries temporal memory across steps\n'
            '→ can generate internal oscillation AND adapt gait after perturbation',
            ha='center', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))

    # Key difference callout
    ax.text(1.8, 0.3,
            'KEY DIFFERENCE vs MLP:\n'
            'Recurrent h(t)/c(t) = internal state\n'
            '→ Output CHANGES even with frozen input\n'
            '→ Perturbation shifts hidden state\n'
            '→ New stable gait (adaptation, not recovery)',
            ha='left', fontsize=8.5, color='#882200',
            bbox=dict(boxstyle='round', facecolor='#fff0e0', edgecolor='#cc4400',
                      alpha=0.8, linewidth=1.5))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("[1/5] Loading models...")
    model_mlp = torch.jit.load(MLP_PATH, map_location='cpu')
    model_lstm = torch.jit.load(LSTM_PATH, map_location='cpu')
    model_mlp.eval()
    model_lstm.eval()

    print(f"[2/5] Running open-loop simulation ({SIM_SECONDS}s, cmd=[0.15, 0, 0])...")

    # Nominal runs
    np.random.seed(42)
    obs_mlp, act_mlp, tp_mlp, tv_mlp = run_openloop(model_mlp, is_lstm=False)

    np.random.seed(42)
    model_lstm.hidden_state.zero_()
    model_lstm.cell_state.zero_()
    obs_lstm, act_lstm, tp_lstm, tv_lstm = run_openloop(model_lstm, is_lstm=True)

    print(f"[3/5] Running perturbed simulation (+160ms delay at t={PERTURB_TIME}s)...")

    # Perturbed runs (8 extra delay steps = 160ms)
    np.random.seed(42)
    _, _, tp_pert_mlp, tv_pert_mlp = run_openloop(model_mlp, is_lstm=False,
                                                    extra_delay=8, perturb_step=PERTURB_STEP)

    np.random.seed(42)
    model_lstm.hidden_state.zero_()
    model_lstm.cell_state.zero_()
    _, _, tp_pert_lstm, tv_pert_lstm = run_openloop(model_lstm, is_lstm=True,
                                                      extra_delay=8, perturb_step=PERTURB_STEP)

    print("[4/5] Computing input sensitivity (ablation)...")
    # For MLP: no hidden state, ablation is clean
    sens_mlp, tsens_mlp = compute_multi_step_sensitivity(model_mlp, obs_mlp)

    # For LSTM: need fresh hidden state per evaluation (approximate)
    model_lstm.hidden_state.zero_()
    model_lstm.cell_state.zero_()
    sens_lstm, tsens_lstm = compute_multi_step_sensitivity(model_lstm, obs_lstm)

    print("[5/5] Generating PDF...")
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)

    with PdfPages(OUTPUT_PDF) as pdf:
        # Page 1: Adaptation quality comparison
        fig = plt.figure(figsize=(11, 9))
        page4_adaptation_quality(fig, tp_mlp, tp_pert_mlp, tp_lstm, tp_pert_lstm)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print("  Page 1: Adaptation quality")

        # Page 2: Attractor trajectories
        fig = plt.figure(figsize=(14, 9))
        page_attractor_trajectories(fig, tp_mlp, tv_mlp, tp_pert_mlp, tv_pert_mlp,
                                     tp_lstm, tv_lstm, tp_pert_lstm, tv_pert_lstm)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print("  Page 2: Attractor trajectories")

        # Page 3: Input ablation
        fig = plt.figure(figsize=(11, 9))
        page5_input_ablation(fig, sens_mlp, sens_lstm, tsens_mlp, tsens_lstm)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print("  Page 3: Input ablation")

        # Page 4: MLP system diagram
        fig = plt.figure(figsize=(11, 8))
        page7_mlp_diagram(fig)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print("  Page 4: MLP diagram")

        # Page 5: LSTM system diagram
        fig = plt.figure(figsize=(11, 8))
        page8_lstm_diagram(fig)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print("  Page 5: LSTM diagram")

    print(f"\n[DONE] PDF saved to: {OUTPUT_PDF}")


if __name__ == '__main__':
    main()
