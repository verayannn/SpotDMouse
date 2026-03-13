"""
CPG Attractor Analysis: MLP vs LSTM — Thigh Joints Only.

Focus: clean sinusoidal gait patterns, mid-stride delay perturbation,
and attractor recovery dynamics.

Usage:
    python cpg_attractor_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch
import glob
import os
from scipy.signal import hilbert, butter, filtfilt

# ─── Config ───────────────────────────────────────────────────────────────────
MLP_DIR = os.path.expanduser("~/obs_action_logs_mlp_retrain")
LSTM_DIR = os.path.expanduser("~/obs_action_logs_LSTM_retrain")
OUTPUT_PDF = os.path.expanduser("~/SpotDMouse/P2-Terrain_Challenge/mp2/analysis/cpg_attractor_analysis.pdf")
DT = 0.02  # 50 Hz control

# Observation column indices (0-indexed, after time_step column is dropped)
OBS = {
    'base_lin_vel': (0, 3),
    'base_ang_vel': (3, 6),
    'gravity':      (6, 9),
    'commands':     (9, 12),
    'joint_pos':    (12, 24),
    'joint_vel':    (24, 36),
    'joint_effort': (36, 48),
    'prev_actions': (48, 60),
}

LEGS = {
    'LF': [0, 1, 2],
    'RF': [3, 4, 5],
    'LB': [6, 7, 8],
    'RB': [9, 10, 11],
}

LEG_COLORS = {'LF': '#d62728', 'RF': '#1f77b4', 'LB': '#8c564b', 'RB': '#2ca02c'}
LEG_LABELS = {'LF': 'Left Front', 'RF': 'Right Front', 'LB': 'Left Back', 'RB': 'Right Back'}

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


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_single_env(log_dir, env_idx=0):
    obs_file = os.path.join(log_dir, f'env_{env_idx}_observations.csv')
    act_file = os.path.join(log_dir, f'env_{env_idx}_actions.csv')
    obs = np.genfromtxt(obs_file, delimiter=',', skip_header=1)[:, 1:]
    act = np.genfromtxt(act_file, delimiter=',', skip_header=1)[:, 1:]
    return obs, act


def load_all_envs_separate(log_dir):
    obs_files = sorted(glob.glob(os.path.join(log_dir, 'env_*_observations.csv')))
    act_files = sorted(glob.glob(os.path.join(log_dir, 'env_*_actions.csv')))
    envs = []
    for of, af in zip(obs_files, act_files):
        obs = np.genfromtxt(of, delimiter=',', skip_header=1)[:, 1:]
        act = np.genfromtxt(af, delimiter=',', skip_header=1)[:, 1:]
        envs.append((obs, act))
    return envs


def get_thigh_pos(obs):
    """Extract 4 thigh joint positions."""
    jp = obs[:, OBS['joint_pos'][0]:OBS['joint_pos'][1]]
    return np.column_stack([jp[:, LEGS[l][1]] for l in LEGS])


def get_thigh_vel(obs):
    """Extract 4 thigh joint velocities."""
    jv = obs[:, OBS['joint_vel'][0]:OBS['joint_vel'][1]]
    return np.column_stack([jv[:, LEGS[l][1]] for l in LEGS])


def get_thigh_actions(act):
    """Extract 4 thigh actions."""
    return np.column_stack([act[:, LEGS[l][1]] for l in LEGS])


def find_stable_onset(thigh_pos, window=10):
    """Find the timestep where gait becomes stable.

    Uses rolling variance — stable when variance stops changing rapidly.
    Returns the index where steady-state begins.
    """
    T = len(thigh_pos)
    if T < window * 3:
        return window

    # Compute rolling std of each thigh
    roll_std = np.zeros(T - window)
    for i in range(T - window):
        roll_std[i] = np.mean(np.std(thigh_pos[i:i+window], axis=0))

    # Find where the derivative of rolling std settles (< threshold)
    d_std = np.abs(np.diff(roll_std))
    threshold = np.median(d_std) * 0.5

    # Look for the first sustained low-derivative region
    sustained = 5  # need 5 consecutive low-derivative steps
    for i in range(len(d_std) - sustained):
        if np.all(d_std[i:i+sustained] < threshold):
            return i + window

    # Fallback: use 40% of data as transient
    return int(T * 0.4)


def find_best_env(log_dir):
    """Find env with most periodic steady-state gait."""
    envs = load_all_envs_separate(log_dir)
    best_idx, best_score = 0, -np.inf
    for i, (obs, act) in enumerate(envs):
        tp = get_thigh_pos(obs)
        onset = find_stable_onset(tp)
        stable = tp[onset:]
        if len(stable) < 20:
            continue
        # Score by oscillation amplitude in steady state
        score = np.mean(np.std(stable, axis=0))
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def apply_mid_stride_delay(signal, delay_steps, onset_step):
    """Apply delay perturbation starting at onset_step (mid-stride).

    Before onset_step: signal is unchanged (nominal).
    After onset_step: signal is delayed by delay_steps.
    This creates a visible disruption at the perturbation point.
    """
    T = len(signal)
    result = signal.copy()
    for i in range(onset_step, T):
        src = i - delay_steps
        if src >= 0:
            result[i] = signal[src]
        else:
            result[i] = signal[0]
    return result


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_steady_state_gaits(fig, obs_mlp, obs_lstm, onset_mlp, onset_lstm):
    """Page 1: Clean sinusoidal thigh trajectories in steady-state."""
    tp_mlp = get_thigh_pos(obs_mlp)[onset_mlp:]
    tp_lstm = get_thigh_pos(obs_lstm)[onset_lstm:]
    t_mlp = np.arange(len(tp_mlp)) * DT
    t_lstm = np.arange(len(tp_lstm)) * DT

    gs = GridSpec(2, 1, hspace=0.35)
    legs = list(LEGS.keys())

    ax1 = fig.add_subplot(gs[0])
    for i, leg in enumerate(legs):
        ax1.plot(t_mlp, tp_mlp[:, i], color=LEG_COLORS[leg],
                 label=LEG_LABELS[leg], alpha=0.85)
    ax1.set_title('MLP — Thigh Joint Trajectories (Fixed CPG, Steady-State)', fontweight='bold')
    ax1.set_ylabel('Joint Position (rad)')
    ax1.legend(loc='upper right', ncol=4, framealpha=0.9)

    ax2 = fig.add_subplot(gs[1])
    for i, leg in enumerate(legs):
        ax2.plot(t_lstm, tp_lstm[:, i], color=LEG_COLORS[leg],
                 label=LEG_LABELS[leg], alpha=0.85)
    ax2.set_title('LSTM — Thigh Joint Trajectories (Adaptive CPG, Steady-State)', fontweight='bold')
    ax2.set_ylabel('Joint Position (rad)')
    ax2.set_xlabel('Time (s)')
    ax2.legend(loc='upper right', ncol=4, framealpha=0.9)

    fig.suptitle('Steady-State Central Pattern Generator: Thigh Joints Only',
                 fontsize=13, fontweight='bold', y=0.98)


def plot_individual_legs(fig, obs_mlp, obs_lstm, onset_mlp, onset_lstm):
    """Page 2: Each leg's thigh trajectory side-by-side MLP vs LSTM."""
    tp_mlp = get_thigh_pos(obs_mlp)[onset_mlp:]
    tp_lstm = get_thigh_pos(obs_lstm)[onset_lstm:]
    T = min(len(tp_mlp), len(tp_lstm))
    t = np.arange(T) * DT
    legs = list(LEGS.keys())

    gs = GridSpec(4, 1, hspace=0.4)

    for i, leg in enumerate(legs):
        ax = fig.add_subplot(gs[i])
        ax.plot(t, tp_mlp[:T, i], color='#1f77b4', label='MLP', alpha=0.85)
        ax.plot(t, tp_lstm[:T, i], color='#d62728', label='LSTM', alpha=0.85, ls='--')
        ax.set_title(f'{LEG_LABELS[leg]} Thigh', fontweight='bold', fontsize=11)
        ax.set_ylabel('Position (rad)')
        if i == 0:
            ax.legend(loc='upper right', ncol=2)
        if i == 3:
            ax.set_xlabel('Time (s)')

    fig.suptitle('MLP vs LSTM: Per-Leg Thigh Comparison (Steady-State)',
                 fontsize=13, fontweight='bold', y=0.98)


def plot_phase_portraits_steady(fig, obs_mlp, obs_lstm, onset_mlp, onset_lstm):
    """Page 3: Phase portraits (pos vs vel) for each leg thigh — steady-state only."""
    tp_mlp = get_thigh_pos(obs_mlp)[onset_mlp:]
    tv_mlp = get_thigh_vel(obs_mlp)[onset_mlp:]
    tp_lstm = get_thigh_pos(obs_lstm)[onset_lstm:]
    tv_lstm = get_thigh_vel(obs_lstm)[onset_lstm:]
    legs = list(LEGS.keys())

    gs = GridSpec(2, 4, hspace=0.4, wspace=0.35)

    for i, leg in enumerate(legs):
        # MLP
        ax_m = fig.add_subplot(gs[0, i])
        ax_m.plot(tp_mlp[:, i], tv_mlp[:, i], color=LEG_COLORS[leg],
                  alpha=0.5, linewidth=0.9)
        # Add direction arrows at a few points
        n = len(tp_mlp)
        for k in [n//4, n//2, 3*n//4]:
            if k + 1 < n:
                ax_m.annotate('', xy=(tp_mlp[k+1, i], tv_mlp[k+1, i]),
                             xytext=(tp_mlp[k, i], tv_mlp[k, i]),
                             arrowprops=dict(arrowstyle='->', color=LEG_COLORS[leg], lw=1.5))
        ax_m.set_title(f'MLP {leg}', fontweight='bold', fontsize=10)
        if i == 0:
            ax_m.set_ylabel('Velocity (rad/s)')
        ax_m.set_xlabel('Position', fontsize=8)
        ax_m.tick_params(labelsize=7)

        # LSTM
        ax_l = fig.add_subplot(gs[1, i])
        ax_l.plot(tp_lstm[:, i], tv_lstm[:, i], color=LEG_COLORS[leg],
                  alpha=0.5, linewidth=0.9)
        n = len(tp_lstm)
        for k in [n//4, n//2, 3*n//4]:
            if k + 1 < n:
                ax_l.annotate('', xy=(tp_lstm[k+1, i], tv_lstm[k+1, i]),
                             xytext=(tp_lstm[k, i], tv_lstm[k, i]),
                             arrowprops=dict(arrowstyle='->', color=LEG_COLORS[leg], lw=1.5))
        ax_l.set_title(f'LSTM {leg}', fontweight='bold', fontsize=10)
        if i == 0:
            ax_l.set_ylabel('Velocity (rad/s)')
        ax_l.set_xlabel('Position', fontsize=8)
        ax_l.tick_params(labelsize=7)

    fig.suptitle('Phase-Space Limit Cycles: Thigh (Position vs Velocity), Steady-State',
                 fontsize=13, fontweight='bold', y=0.98)


def plot_mid_stride_perturbation(fig, obs_mlp, obs_lstm, act_mlp, act_lstm,
                                  onset_mlp, onset_lstm):
    """Page 4: Mid-stride delay injection — show gait before, during, and after perturbation."""
    # Use full stable region; inject perturbation at 40% of stable region
    ta_mlp = get_thigh_actions(act_mlp)[onset_mlp:]
    ta_lstm = get_thigh_actions(act_lstm)[onset_lstm:]
    T = min(len(ta_mlp), len(ta_lstm))
    t = np.arange(T) * DT

    perturb_onset = int(T * 0.35)  # Inject delay at 35% through stable region
    perturb_time = perturb_onset * DT
    delay_steps = 8  # 160ms (2× nominal)

    gs = GridSpec(2, 1, hspace=0.35)

    # LF thigh action: nominal vs perturbed
    ax1 = fig.add_subplot(gs[0])
    nominal = ta_mlp[:T, 0]
    perturbed = apply_mid_stride_delay(nominal, delay_steps, perturb_onset)
    ax1.plot(t, nominal, color='#1f77b4', label='Nominal', alpha=0.85)
    ax1.plot(t, perturbed, color='#d62728', label=f'+ {delay_steps*DT*1000:.0f}ms delay @ t={perturb_time:.1f}s',
             alpha=0.75, ls='--')
    ax1.axvline(perturb_time, color='gray', ls=':', alpha=0.6, label='Perturbation onset')
    ax1.fill_betweenx(ax1.get_ylim(), perturb_time, t[-1], alpha=0.05, color='red')
    ax1.set_title('MLP — LF Thigh Action: Mid-Stride Delay Injection', fontweight='bold')
    ax1.set_ylabel('Action (rad)')
    ax1.legend(loc='upper right', fontsize=8)

    ax2 = fig.add_subplot(gs[1])
    nominal_l = ta_lstm[:T, 0]
    perturbed_l = apply_mid_stride_delay(nominal_l, delay_steps, perturb_onset)
    ax2.plot(t, nominal_l, color='#1f77b4', label='Nominal', alpha=0.85)
    ax2.plot(t, perturbed_l, color='#d62728', label=f'+ {delay_steps*DT*1000:.0f}ms delay @ t={perturb_time:.1f}s',
             alpha=0.75, ls='--')
    ax2.axvline(perturb_time, color='gray', ls=':', alpha=0.6, label='Perturbation onset')
    ax2.set_title('LSTM — LF Thigh Action: Mid-Stride Delay Injection', fontweight='bold')
    ax2.set_ylabel('Action (rad)')
    ax2.set_xlabel('Time (s)')
    ax2.legend(loc='upper right', fontsize=8)

    fig.suptitle('Mid-Stride Perturbation: 160ms Delay Injected After Gait Stabilization',
                 fontsize=13, fontweight='bold', y=0.98)


def plot_all_legs_perturbation(fig, obs_mlp, obs_lstm, act_mlp, act_lstm,
                                onset_mlp, onset_lstm):
    """Page 5: All 4 legs under mid-stride perturbation for both policies."""
    ta_mlp = get_thigh_actions(act_mlp)[onset_mlp:]
    ta_lstm = get_thigh_actions(act_lstm)[onset_lstm:]
    T = min(len(ta_mlp), len(ta_lstm))
    t = np.arange(T) * DT

    perturb_onset = int(T * 0.35)
    perturb_time = perturb_onset * DT
    delay_steps = 8

    legs = list(LEGS.keys())
    gs = GridSpec(4, 2, hspace=0.45, wspace=0.3)

    for i, leg in enumerate(legs):
        # MLP
        ax_m = fig.add_subplot(gs[i, 0])
        nom = ta_mlp[:T, i]
        pert = apply_mid_stride_delay(nom, delay_steps, perturb_onset)
        ax_m.plot(t, nom, color='#1f77b4', alpha=0.8, label='Nominal')
        ax_m.plot(t, pert, color='#d62728', alpha=0.65, ls='--', label='Delayed')
        ax_m.axvline(perturb_time, color='gray', ls=':', alpha=0.5)
        ax_m.set_title(f'MLP — {LEG_LABELS[leg]}', fontweight='bold', fontsize=10)
        ax_m.set_ylabel('Action', fontsize=8)
        if i == 0:
            ax_m.legend(fontsize=7, ncol=2)
        if i == 3:
            ax_m.set_xlabel('Time (s)')

        # LSTM
        ax_l = fig.add_subplot(gs[i, 1])
        nom_l = ta_lstm[:T, i]
        pert_l = apply_mid_stride_delay(nom_l, delay_steps, perturb_onset)
        ax_l.plot(t, nom_l, color='#1f77b4', alpha=0.8, label='Nominal')
        ax_l.plot(t, pert_l, color='#d62728', alpha=0.65, ls='--', label='Delayed')
        ax_l.axvline(perturb_time, color='gray', ls=':', alpha=0.5)
        ax_l.set_title(f'LSTM — {LEG_LABELS[leg]}', fontweight='bold', fontsize=10)
        if i == 0:
            ax_l.legend(fontsize=7, ncol=2)
        if i == 3:
            ax_l.set_xlabel('Time (s)')

    fig.suptitle('All Legs: Mid-Stride 160ms Delay Perturbation',
                 fontsize=13, fontweight='bold', y=0.98)


def plot_attractor_before_after(fig, obs_mlp, obs_lstm, act_mlp, act_lstm,
                                 onset_mlp, onset_lstm):
    """Page 6: Phase portraits split into pre-perturbation and post-perturbation.

    Shows attractor shape before delay injection vs after — does the orbit
    deform or recover?
    """
    tp_mlp = get_thigh_pos(obs_mlp)[onset_mlp:]
    tv_mlp = get_thigh_vel(obs_mlp)[onset_mlp:]
    tp_lstm = get_thigh_pos(obs_lstm)[onset_lstm:]
    tv_lstm = get_thigh_vel(obs_lstm)[onset_lstm:]

    T = min(len(tp_mlp), len(tp_lstm))
    perturb_onset = int(T * 0.35)

    # LF thigh only for clarity
    gs = GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # Pre-perturbation
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(tp_mlp[:perturb_onset, 0], tv_mlp[:perturb_onset, 0],
             color='#1f77b4', alpha=0.6, linewidth=1.0)
    ax1.set_title('MLP — Pre-Perturbation', fontweight='bold')
    ax1.set_ylabel('Velocity (rad/s)')
    ax1.set_xlabel('Position (rad)')

    ax2 = fig.add_subplot(gs[0, 1])
    # Post-perturbation: use delayed position
    delay_steps = 8
    delayed_pos = apply_mid_stride_delay(tp_mlp[:T, 0], delay_steps, perturb_onset)
    ax2.plot(delayed_pos[perturb_onset:], tv_mlp[perturb_onset:T, 0],
             color='#d62728', alpha=0.6, linewidth=1.0)
    # Overlay nominal for reference
    ax2.plot(tp_mlp[perturb_onset:T, 0], tv_mlp[perturb_onset:T, 0],
             color='#1f77b4', alpha=0.2, linewidth=0.7, ls='--', label='Nominal')
    ax2.set_title('MLP — Post-Perturbation (160ms)', fontweight='bold')
    ax2.set_xlabel('Position (rad)')
    ax2.legend(fontsize=7)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(tp_lstm[:perturb_onset, 0], tv_lstm[:perturb_onset, 0],
             color='#1f77b4', alpha=0.6, linewidth=1.0)
    ax3.set_title('LSTM — Pre-Perturbation', fontweight='bold')
    ax3.set_ylabel('Velocity (rad/s)')
    ax3.set_xlabel('Position (rad)')

    ax4 = fig.add_subplot(gs[1, 1])
    delayed_pos_l = apply_mid_stride_delay(tp_lstm[:T, 0], delay_steps, perturb_onset)
    ax4.plot(delayed_pos_l[perturb_onset:], tv_lstm[perturb_onset:T, 0],
             color='#d62728', alpha=0.6, linewidth=1.0)
    ax4.plot(tp_lstm[perturb_onset:T, 0], tv_lstm[perturb_onset:T, 0],
             color='#1f77b4', alpha=0.2, linewidth=0.7, ls='--', label='Nominal')
    ax4.set_title('LSTM — Post-Perturbation (160ms)', fontweight='bold')
    ax4.set_xlabel('Position (rad)')
    ax4.legend(fontsize=7)

    fig.suptitle('LF Thigh Attractor: Before vs After Mid-Stride Delay Injection',
                 fontsize=13, fontweight='bold', y=0.98)


def plot_3d_attractor_thighs(fig, obs_mlp, obs_lstm, onset_mlp, onset_lstm):
    """Page 7: 3D attractor using LF, RF, LB thigh positions (steady-state)."""
    tp_mlp = get_thigh_pos(obs_mlp)[onset_mlp:]
    tp_lstm = get_thigh_pos(obs_lstm)[onset_lstm:]

    gs = GridSpec(1, 2, wspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.plot(tp_mlp[:, 0], tp_mlp[:, 1], tp_mlp[:, 2],
             color='#1f77b4', alpha=0.6, linewidth=0.9)
    ax1.scatter(tp_mlp[0, 0], tp_mlp[0, 1], tp_mlp[0, 2],
                color='green', s=60, zorder=5, label='Start')
    ax1.scatter(tp_mlp[-1, 0], tp_mlp[-1, 1], tp_mlp[-1, 2],
                color='red', s=60, zorder=5, marker='x', label='End')
    ax1.set_xlabel('LF Thigh', fontsize=8)
    ax1.set_ylabel('RF Thigh', fontsize=8)
    ax1.set_zlabel('LB Thigh', fontsize=8)
    ax1.set_title('MLP Attractor', fontweight='bold')
    ax1.tick_params(labelsize=7)
    ax1.legend(fontsize=7)

    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax2.plot(tp_lstm[:, 0], tp_lstm[:, 1], tp_lstm[:, 2],
             color='#d62728', alpha=0.6, linewidth=0.9)
    ax2.scatter(tp_lstm[0, 0], tp_lstm[0, 1], tp_lstm[0, 2],
                color='green', s=60, zorder=5, label='Start')
    ax2.scatter(tp_lstm[-1, 0], tp_lstm[-1, 1], tp_lstm[-1, 2],
                color='red', s=60, zorder=5, marker='x', label='End')
    ax2.set_xlabel('LF Thigh', fontsize=8)
    ax2.set_ylabel('RF Thigh', fontsize=8)
    ax2.set_zlabel('LB Thigh', fontsize=8)
    ax2.set_title('LSTM Attractor', fontweight='bold')
    ax2.tick_params(labelsize=7)
    ax2.legend(fontsize=7)

    fig.suptitle('3D Attractor: Inter-Leg Thigh Coordination (Steady-State)',
                 fontsize=13, fontweight='bold', y=0.98)


def plot_autocorrelation_and_metrics(fig, obs_mlp, obs_lstm, act_mlp, act_lstm,
                                      onset_mlp, onset_lstm):
    """Page 8: Autocorrelation, gait frequency, amplitude, action-pos coherence."""
    tp_mlp = get_thigh_pos(obs_mlp)[onset_mlp:]
    tp_lstm = get_thigh_pos(obs_lstm)[onset_lstm:]
    ta_mlp = get_thigh_actions(act_mlp)[onset_mlp:]
    ta_lstm = get_thigh_actions(act_lstm)[onset_lstm:]

    legs = list(LEGS.keys())
    gs = GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # Autocorrelation for all 4 thigh joints
    ax1 = fig.add_subplot(gs[0, 0])
    max_lag = min(int(1.0 / DT), len(tp_mlp) - 1)
    lags = np.arange(max_lag) * DT
    for i, leg in enumerate(legs):
        sig = tp_mlp[:, i] - tp_mlp[:, i].mean()
        acf = np.correlate(sig, sig, mode='full')[len(sig)-1:]
        acf /= acf[0] if acf[0] != 0 else 1
        ax1.plot(lags, acf[:max_lag], color=LEG_COLORS[leg], alpha=0.7, label=leg)
    ax1.axhline(0, color='k', ls='--', alpha=0.3)
    ax1.set_title('MLP — Thigh Autocorrelation', fontweight='bold')
    ax1.set_xlabel('Lag (s)')
    ax1.set_ylabel('Autocorrelation')
    ax1.legend(fontsize=7, ncol=4)

    ax2 = fig.add_subplot(gs[0, 1])
    max_lag_l = min(int(1.0 / DT), len(tp_lstm) - 1)
    lags_l = np.arange(max_lag_l) * DT
    for i, leg in enumerate(legs):
        sig = tp_lstm[:, i] - tp_lstm[:, i].mean()
        acf = np.correlate(sig, sig, mode='full')[len(sig)-1:]
        acf /= acf[0] if acf[0] != 0 else 1
        ax2.plot(lags_l, acf[:max_lag_l], color=LEG_COLORS[leg], alpha=0.7, label=leg)
    ax2.axhline(0, color='k', ls='--', alpha=0.3)
    ax2.set_title('LSTM — Thigh Autocorrelation', fontweight='bold')
    ax2.set_xlabel('Lag (s)')
    ax2.legend(fontsize=7, ncol=4)

    # Gait period and amplitude
    def get_period(jp_col):
        sig = jp_col - jp_col.mean()
        if np.std(sig) < 1e-6:
            return 0
        acf = np.correlate(sig, sig, mode='full')[len(sig)-1:]
        acf /= acf[0] if acf[0] != 0 else 1
        crossed = False
        for k in range(1, len(acf) - 1):
            if acf[k] < 0:
                crossed = True
            if crossed and acf[k] > acf[k-1] and acf[k] > acf[k+1] and acf[k] > 0.15:
                return k * DT
        return 0

    periods_mlp = [get_period(tp_mlp[:, i]) for i in range(4)]
    periods_lstm = [get_period(tp_lstm[:, i]) for i in range(4)]
    amps_mlp = [np.std(tp_mlp[:, i]) for i in range(4)]
    amps_lstm = [np.std(tp_lstm[:, i]) for i in range(4)]

    # Action→Position correlation
    T_m = min(len(ta_mlp), len(tp_mlp))
    T_l = min(len(ta_lstm), len(tp_lstm))
    corrs_mlp = [np.abs(np.corrcoef(ta_mlp[:T_m, i], tp_mlp[:T_m, i])[0, 1]) for i in range(4)]
    corrs_lstm = [np.abs(np.corrcoef(ta_lstm[:T_l, i], tp_lstm[:T_l, i])[0, 1]) for i in range(4)]

    # Summary bar chart
    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(4)
    w = 0.35
    ax3.bar(x - w/2, amps_mlp, w, label='MLP', color='#1f77b4', alpha=0.7)
    ax3.bar(x + w/2, amps_lstm, w, label='LSTM', color='#d62728', alpha=0.7)
    ax3.set_title('Thigh Oscillation Amplitude', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(legs)
    ax3.set_ylabel('Amplitude (rad)')
    ax3.legend()

    # Summary text box
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    mlp_p = np.mean([p for p in periods_mlp if p > 0]) if any(p > 0 for p in periods_mlp) else 0
    lstm_p = np.mean([p for p in periods_lstm if p > 0]) if any(p > 0 for p in periods_lstm) else 0
    mlp_f = 1.0/mlp_p if mlp_p > 0 else 0
    lstm_f = 1.0/lstm_p if lstm_p > 0 else 0

    # Delay vulnerability: fraction of gait cycle lost to 76ms delay
    mlp_vuln = mlp_f * 0.076 * 100 if mlp_f > 0 else 0
    lstm_vuln = lstm_f * 0.076 * 100 if lstm_f > 0 else 0

    summary = (
        "CPG Characterization (Thigh Only)\n"
        "───────────────────────────────────\n\n"
        f"MLP gait freq:   {mlp_f:.1f} Hz  (T={mlp_p:.3f}s)\n"
        f"LSTM gait freq:  {lstm_f:.1f} Hz  (T={lstm_p:.3f}s)\n\n"
        f"MLP amplitude:   {np.mean(amps_mlp):.4f} rad\n"
        f"LSTM amplitude:  {np.mean(amps_lstm):.4f} rad\n\n"
        f"MLP act→pos:     {np.mean(corrs_mlp):.3f}\n"
        f"LSTM act→pos:    {np.mean(corrs_lstm):.3f}\n\n"
        f"76ms delay vulnerability:\n"
        f"  MLP: {mlp_vuln:.0f}% of gait cycle\n"
        f"  LSTM: {lstm_vuln:.0f}% of gait cycle\n\n"
        "LSTM: lower freq → less delay-sensitive\n"
        "       higher correlation → feedback-driven\n"
        "MLP:  higher freq → more delay-sensitive\n"
        "       lower correlation → open-loop CPG"
    )
    ax4.text(0.02, 0.98, summary, transform=ax4.transAxes,
             fontsize=9.5, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Quantitative CPG Characterization: Thigh Joints',
                 fontsize=13, fontweight='bold', y=0.98)


def plot_delay_sweep_attractors(fig, obs_mlp, obs_lstm, onset_mlp, onset_lstm):
    """Page 9: Attractor under increasing delay — injected mid-stride, showing only post-perturbation."""
    tp_mlp = get_thigh_pos(obs_mlp)[onset_mlp:]
    tv_mlp = get_thigh_vel(obs_mlp)[onset_mlp:]
    tp_lstm = get_thigh_pos(obs_lstm)[onset_lstm:]
    tv_lstm = get_thigh_vel(obs_lstm)[onset_lstm:]

    T = min(len(tp_mlp), len(tp_lstm))
    perturb_onset = int(T * 0.35)

    delays = [0, 4, 8, 12]
    delay_labels = ['Nominal', '+76ms', '+160ms', '+240ms']
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd']

    gs = GridSpec(2, 4, hspace=0.4, wspace=0.35)

    for j, (d, dl, c) in enumerate(zip(delays, delay_labels, colors)):
        # MLP — show post-perturbation orbit only (or full for nominal)
        ax_m = fig.add_subplot(gs[0, j])
        if d == 0:
            ax_m.plot(tp_mlp[perturb_onset:T, 0], tv_mlp[perturb_onset:T, 0],
                      color=c, alpha=0.5, linewidth=0.9)
        else:
            delayed = apply_mid_stride_delay(tp_mlp[:T, 0], d, perturb_onset)
            ax_m.plot(delayed[perturb_onset:], tv_mlp[perturb_onset:T, 0],
                      color=c, alpha=0.5, linewidth=0.9)
        ax_m.set_title(f'MLP {dl}', fontweight='bold', fontsize=9)
        if j == 0:
            ax_m.set_ylabel('Vel (rad/s)')
        ax_m.set_xlabel('Pos (rad)', fontsize=8)
        ax_m.tick_params(labelsize=7)

        # LSTM
        ax_l = fig.add_subplot(gs[1, j])
        if d == 0:
            ax_l.plot(tp_lstm[perturb_onset:T, 0], tv_lstm[perturb_onset:T, 0],
                      color=c, alpha=0.5, linewidth=0.9)
        else:
            delayed_l = apply_mid_stride_delay(tp_lstm[:T, 0], d, perturb_onset)
            ax_l.plot(delayed_l[perturb_onset:], tv_lstm[perturb_onset:T, 0],
                      color=c, alpha=0.5, linewidth=0.9)
        ax_l.set_title(f'LSTM {dl}', fontweight='bold', fontsize=9)
        if j == 0:
            ax_l.set_ylabel('Vel (rad/s)')
        ax_l.set_xlabel('Pos (rad)', fontsize=8)
        ax_l.tick_params(labelsize=7)

    fig.suptitle('LF Thigh Attractor Under Mid-Stride Delay Sweep (Post-Perturbation Only)',
                 fontsize=12, fontweight='bold', y=0.98)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("[1/3] Finding best envs and stable onsets...")
    best_mlp = find_best_env(MLP_DIR)
    best_lstm = find_best_env(LSTM_DIR)

    obs_mlp, act_mlp = load_single_env(MLP_DIR, best_mlp)
    obs_lstm, act_lstm = load_single_env(LSTM_DIR, best_lstm)

    onset_mlp = find_stable_onset(get_thigh_pos(obs_mlp))
    onset_lstm = find_stable_onset(get_thigh_pos(obs_lstm))

    stable_mlp = len(obs_mlp) - onset_mlp
    stable_lstm = len(obs_lstm) - onset_lstm

    print(f"  MLP env {best_mlp}: {len(obs_mlp)} steps, stable from step {onset_mlp} "
          f"({onset_mlp*DT:.2f}s) → {stable_mlp} stable steps ({stable_mlp*DT:.1f}s)")
    print(f"  LSTM env {best_lstm}: {len(obs_lstm)} steps, stable from step {onset_lstm} "
          f"({onset_lstm*DT:.2f}s) → {stable_lstm} stable steps ({stable_lstm*DT:.1f}s)")

    print("[2/3] Generating plots...")
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)

    with PdfPages(OUTPUT_PDF) as pdf:
        # Page 1: Steady-state gait
        fig = plt.figure(figsize=(11, 7))
        plot_steady_state_gaits(fig, obs_mlp, obs_lstm, onset_mlp, onset_lstm)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print("  Page 1: Steady-state gait")

        # Page 2: Per-leg comparison
        fig = plt.figure(figsize=(11, 11))
        plot_individual_legs(fig, obs_mlp, obs_lstm, onset_mlp, onset_lstm)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print("  Page 2: Per-leg comparison")

        # Page 3: Phase portraits
        fig = plt.figure(figsize=(14, 7))
        plot_phase_portraits_steady(fig, obs_mlp, obs_lstm, onset_mlp, onset_lstm)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print("  Page 3: Phase portraits")

        # Page 4: Mid-stride perturbation (LF thigh)
        fig = plt.figure(figsize=(11, 7))
        plot_mid_stride_perturbation(fig, obs_mlp, obs_lstm, act_mlp, act_lstm,
                                      onset_mlp, onset_lstm)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print("  Page 4: Mid-stride perturbation")

        # Page 5: All legs perturbation
        fig = plt.figure(figsize=(11, 12))
        plot_all_legs_perturbation(fig, obs_mlp, obs_lstm, act_mlp, act_lstm,
                                    onset_mlp, onset_lstm)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print("  Page 5: All legs perturbation")

        # Page 6: Attractor before vs after
        fig = plt.figure(figsize=(11, 9))
        plot_attractor_before_after(fig, obs_mlp, obs_lstm, act_mlp, act_lstm,
                                     onset_mlp, onset_lstm)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print("  Page 6: Attractor before/after")

        # Page 7: 3D attractors
        fig = plt.figure(figsize=(11, 6))
        plot_3d_attractor_thighs(fig, obs_mlp, obs_lstm, onset_mlp, onset_lstm)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print("  Page 7: 3D attractors")

        # Page 8: Metrics
        fig = plt.figure(figsize=(11, 9))
        plot_autocorrelation_and_metrics(fig, obs_mlp, obs_lstm, act_mlp, act_lstm,
                                          onset_mlp, onset_lstm)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print("  Page 8: Metrics")

        # Page 9: Delay sweep attractors
        fig = plt.figure(figsize=(14, 7))
        plot_delay_sweep_attractors(fig, obs_mlp, obs_lstm, onset_mlp, onset_lstm)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print("  Page 9: Delay sweep attractors")

    print(f"\n[DONE] PDF saved to: {OUTPUT_PDF}")


if __name__ == '__main__':
    main()
