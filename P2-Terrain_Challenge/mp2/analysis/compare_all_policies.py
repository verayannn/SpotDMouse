"""
Compare all policy variants open-loop: MLP (hippy, scheduled, retrain) + LSTM (original, retrain).
Runs each through synthetic PD actuator for 20s and overlays thigh joint trajectories.
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
POLICIES = {
    'MLP hippy':     (os.path.expanduser("~/policy_joyboy_delayedpdactuator_hippy.pt"), False),
    'MLP scheduled': (os.path.expanduser("~/policy_joyboy_delayedpdactuator_scheduled.pt"), False),
    'MLP retrain':   (os.path.expanduser("~/policy_joyboy_delayedpdactuator_mlp_retrain.pt"), False),
    'LSTM original': (os.path.expanduser("~/policy_joyboy_delayedpdactuator_LSTM.pt"), True),
    'LSTM retrain':  (os.path.expanduser("~/policy_joyboy_delayedpdactuator_LSTM_retrain.pt"), True),
}

OUTPUT_PDF = os.path.expanduser(
    "~/SpotDMouse/P2-Terrain_Challenge/mp2/analysis/compare_all_policies.pdf"
)

DT = 0.02
SIM_SECONDS = 20.0
N_STEPS = int(SIM_SECONDS / DT)

# PD actuator params
KP = 70.0
KD = 1.2
INERTIA = 0.20
FRICTION = 0.03
EFFORT_LIMIT = 5.0
DELAY_STEPS = 9
PD_SUBSTEPS = 4
ACTION_SCALE = 1.5

DEFAULT_JOINT_POS = np.array([
    0.0, 0.55, -1.0,  # LF
    0.0, 0.55, -1.0,  # RF
    0.0, 0.55, -1.0,  # LB
    0.0, 0.55, -1.0,  # RB
])

JOINT_LOWER = np.array([
    -0.524, 0.0, -2.356,
    -0.524, 0.0, -2.356,
    -0.524, 0.0, -2.356,
    -0.524, 0.0, -2.356,
])
JOINT_UPPER = np.array([
    0.524, 1.396, 0.0,
    0.524, 1.396, 0.0,
    0.524, 1.396, 0.0,
    0.524, 1.396, 0.0,
])

IDX_LIN_VEL  = slice(0, 3)
IDX_ANG_VEL  = slice(3, 6)
IDX_GRAVITY  = slice(6, 9)
IDX_CMD      = slice(9, 12)
IDX_JPOS     = slice(12, 24)
IDX_JVEL     = slice(24, 36)
IDX_JEFFORT  = slice(36, 48)
IDX_PREV_ACT = slice(48, 60)

THIGH_IDX = [1, 4, 7, 10]
LEG_NAMES = ['LF', 'RF', 'LB', 'RB']

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


class SyntheticPDActuator:
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
        self.action_buffer.append(action_target.copy())
        if len(self.action_buffer) > self.delay_steps:
            delayed_target = self.action_buffer[-self.delay_steps - 1]
        else:
            delayed_target = self.action_buffer[0]

        dt_sub = DT / PD_SUBSTEPS
        for _ in range(PD_SUBSTEPS):
            error = delayed_target - self.position
            torque = KP * error - KD * self.velocity
            torque -= FRICTION * np.sign(self.velocity)
            torque = np.clip(torque, -EFFORT_LIMIT, EFFORT_LIMIT)
            accel = torque / INERTIA
            self.velocity += accel * dt_sub
            self.position += self.velocity * dt_sub
            hit_lower = self.position < JOINT_LOWER
            hit_upper = self.position > JOINT_UPPER
            self.position = np.clip(self.position, JOINT_LOWER, JOINT_UPPER)
            self.velocity[hit_lower | hit_upper] = 0.0

        self.effort = torque
        return self.position.copy(), self.velocity.copy(), self.effort.copy()


def run_openloop(model, is_lstm=False, cmd=np.array([0.15, 0.0, 0.0])):
    pd = SyntheticPDActuator(delay_steps=DELAY_STEPS)
    pd.reset(init_pos=DEFAULT_JOINT_POS.copy())

    all_thigh_pos = np.zeros((N_STEPS, 4))
    all_thigh_vel = np.zeros((N_STEPS, 4))
    all_actions = np.zeros((N_STEPS, 12))
    prev_action = np.zeros(12)
    gravity = np.array([0.0, 0.0, -1.0])

    if is_lstm:
        model.hidden_state.zero_()
        model.cell_state.zero_()

    for t in range(N_STEPS):
        obs = np.zeros(60)
        obs[IDX_LIN_VEL] = np.random.normal(0, 0.02, 3)
        obs[IDX_ANG_VEL] = np.random.normal(0, 0.1, 3)
        obs[IDX_GRAVITY] = gravity + np.random.normal(0, 0.01, 3)
        obs[IDX_CMD] = cmd
        obs[IDX_JPOS] = pd.position - DEFAULT_JOINT_POS
        obs[IDX_JVEL] = pd.velocity
        obs[IDX_JEFFORT] = pd.effort / EFFORT_LIMIT
        obs[IDX_PREV_ACT] = prev_action

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = model(obs_t).squeeze(0).numpy()

        all_actions[t] = action
        target_pos = DEFAULT_JOINT_POS + action * ACTION_SCALE
        pos, vel, eff = pd.step(target_pos)
        all_thigh_pos[t] = pos[THIGH_IDX]
        all_thigh_vel[t] = vel[THIGH_IDX]
        prev_action = action

    return all_thigh_pos, all_thigh_vel, all_actions


def compute_gait_metrics(thigh_pos, start_step, end_step):
    """Compute amplitude, frequency, and mean position for each thigh."""
    data = thigh_pos[start_step:end_step]
    n = len(data)
    metrics = {}
    amps = []
    freqs = []
    means = []
    for j in range(4):
        sig = data[:, j]
        amps.append((sig.max() - sig.min()) / 2)
        means.append(sig.mean())
        # FFT for dominant frequency
        sig_centered = sig - sig.mean()
        if np.std(sig_centered) > 1e-6:
            fft = np.abs(np.fft.rfft(sig_centered))
            fft[0] = 0  # remove DC
            freq_axis = np.fft.rfftfreq(n, d=DT)
            peak_idx = np.argmax(fft[1:]) + 1
            freqs.append(freq_axis[peak_idx])
        else:
            freqs.append(0.0)
    metrics['amplitude'] = np.array(amps)
    metrics['frequency'] = np.array(freqs)
    metrics['mean_pos'] = np.array(means)
    return metrics


def main():
    # Run all policies
    results = {}
    for name, (path, is_lstm) in POLICIES.items():
        print(f"  Running {name}...")
        model = torch.jit.load(path, map_location='cpu')
        model.eval()
        np.random.seed(42)
        if is_lstm:
            model.hidden_state.zero_()
            model.cell_state.zero_()
        tp, tv, act = run_openloop(model, is_lstm=is_lstm)
        results[name] = {'thigh_pos': tp, 'thigh_vel': tv, 'actions': act}

    # Stable window: 5s-20s
    skip = int(5.0 / DT)
    t_stable = np.arange(skip, N_STEPS) * DT

    # Metric window: last 10s (stable gait)
    metric_start = int(10.0 / DT)

    # Colors per policy
    policy_colors = {
        'MLP hippy':     '#1f77b4',
        'MLP scheduled': '#ff7f0e',
        'MLP retrain':   '#d62728',
        'LSTM original': '#2ca02c',
        'LSTM retrain':  '#9467bd',
    }
    policy_ls = {
        'MLP hippy':     '-',
        'MLP scheduled': '--',
        'MLP retrain':   '-.',
        'LSTM original': '-',
        'LSTM retrain':  '--',
    }

    print("Generating PDF...")

    with PdfPages(OUTPUT_PDF) as pdf:
        # ─── Page 1: All 5 policies, one subplot per thigh, overlaid ─────────
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle('All Policies — Thigh Joint Trajectories (after 5s stabilization)',
                     fontsize=14, fontweight='bold', y=0.98)
        for j, leg in enumerate(LEG_NAMES):
            ax = axes[j]
            for name in POLICIES:
                tp = results[name]['thigh_pos']
                ax.plot(t_stable, tp[skip:, j],
                        color=policy_colors[name], ls=policy_ls[name],
                        alpha=0.8, label=name)
            ax.set_ylabel(f'{leg} Thigh (rad)')
            ax.legend(fontsize=7, ncol=5, loc='upper right')
        axes[-1].set_xlabel('Time (s)')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print("  Page 1: All policies overlay")

        # ─── Page 2: MLP variants only (hippy vs scheduled vs retrain) ───────
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle('MLP Variants — Thigh Joint Comparison',
                     fontsize=14, fontweight='bold', y=0.98)
        mlp_names = ['MLP hippy', 'MLP scheduled', 'MLP retrain']
        for j, leg in enumerate(LEG_NAMES):
            ax = axes[j]
            for name in mlp_names:
                tp = results[name]['thigh_pos']
                ax.plot(t_stable, tp[skip:, j],
                        color=policy_colors[name], ls=policy_ls[name],
                        alpha=0.85, linewidth=1.6, label=name)
            ax.set_ylabel(f'{leg} Thigh (rad)')
            ax.legend(fontsize=8, ncol=3)
        axes[-1].set_xlabel('Time (s)')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print("  Page 2: MLP variants")

        # ─── Page 3: LSTM variants only ──────────────────────────────────────
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle('LSTM Variants — Thigh Joint Comparison',
                     fontsize=14, fontweight='bold', y=0.98)
        lstm_names = ['LSTM original', 'LSTM retrain']
        lstm_colors = {'LSTM original': '#2ca02c', 'LSTM retrain': '#9467bd'}
        for j, leg in enumerate(LEG_NAMES):
            ax = axes[j]
            for name in lstm_names:
                tp = results[name]['thigh_pos']
                ax.plot(t_stable, tp[skip:, j],
                        color=lstm_colors[name], linewidth=1.6,
                        ls='-' if 'original' in name else '--',
                        alpha=0.85, label=name)
            ax.set_ylabel(f'{leg} Thigh (rad)')
            ax.legend(fontsize=8)
        axes[-1].set_xlabel('Time (s)')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print("  Page 3: LSTM variants")

        # ─── Page 4: Zoomed steady-state (last 3s) for fine detail ───────────
        zoom_start = N_STEPS - int(3.0 / DT)
        t_zoom = np.arange(zoom_start, N_STEPS) * DT

        fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex='col')
        fig.suptitle('Steady-State Detail (last 3s) — MLP vs LSTM variants',
                     fontsize=14, fontweight='bold', y=0.98)

        for j, leg in enumerate(LEG_NAMES):
            # Left col: MLPs
            ax = axes[j, 0]
            for name in mlp_names:
                tp = results[name]['thigh_pos']
                ax.plot(t_zoom, tp[zoom_start:, j],
                        color=policy_colors[name], ls=policy_ls[name],
                        linewidth=1.6, alpha=0.85, label=name)
            ax.set_ylabel(f'{leg} (rad)')
            if j == 0:
                ax.set_title('MLP Variants', fontweight='bold')
                ax.legend(fontsize=7, ncol=3)

            # Right col: LSTMs
            ax = axes[j, 1]
            for name in lstm_names:
                tp = results[name]['thigh_pos']
                ax.plot(t_zoom, tp[zoom_start:, j],
                        color=lstm_colors[name],
                        ls='-' if 'original' in name else '--',
                        linewidth=1.6, alpha=0.85, label=name)
            if j == 0:
                ax.set_title('LSTM Variants', fontweight='bold')
                ax.legend(fontsize=7)

        axes[-1, 0].set_xlabel('Time (s)')
        axes[-1, 1].set_xlabel('Time (s)')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print("  Page 4: Zoomed steady-state")

        # ─── Page 5: Gait metrics table ──────────────────────────────────────
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111)
        ax.axis('off')

        header = f"{'Policy':20s} | {'Leg':4s} | {'Amplitude':10s} | {'Freq (Hz)':10s} | {'Mean Pos':10s}"
        lines = [header, "─" * len(header)]

        for name in POLICIES:
            tp = results[name]['thigh_pos']
            m = compute_gait_metrics(tp, metric_start, N_STEPS)
            for j, leg in enumerate(LEG_NAMES):
                lines.append(
                    f"{name:20s} | {leg:4s} | {m['amplitude'][j]:10.4f} | "
                    f"{m['frequency'][j]:10.2f} | {m['mean_pos'][j]:10.4f}"
                )
            lines.append("")

        # Add cross-policy comparison
        lines.append("")
        lines.append("Cross-Policy Comparison (last 10s):")
        lines.append("─" * 60)

        # MLP retrain vs hippy
        tp_h = results['MLP hippy']['thigh_pos']
        tp_r = results['MLP retrain']['thigh_pos']
        rmse = np.sqrt(np.mean((tp_h[metric_start:] - tp_r[metric_start:])**2, axis=0))
        corr = [np.corrcoef(tp_h[metric_start:, j], tp_r[metric_start:, j])[0, 1] for j in range(4)]
        lines.append(f"  MLP hippy↔retrain:     RMSE={rmse.mean():.4f}  corr={np.mean(corr):.3f}")

        tp_s = results['MLP scheduled']['thigh_pos']
        rmse = np.sqrt(np.mean((tp_s[metric_start:] - tp_r[metric_start:])**2, axis=0))
        corr = [np.corrcoef(tp_s[metric_start:, j], tp_r[metric_start:, j])[0, 1] for j in range(4)]
        lines.append(f"  MLP scheduled↔retrain: RMSE={rmse.mean():.4f}  corr={np.mean(corr):.3f}")

        tp_lo = results['LSTM original']['thigh_pos']
        tp_lr = results['LSTM retrain']['thigh_pos']
        rmse = np.sqrt(np.mean((tp_lo[metric_start:] - tp_lr[metric_start:])**2, axis=0))
        corr = [np.corrcoef(tp_lo[metric_start:, j], tp_lr[metric_start:, j])[0, 1] for j in range(4)]
        lines.append(f"  LSTM original↔retrain: RMSE={rmse.mean():.4f}  corr={np.mean(corr):.3f}")

        # MLP retrain vs LSTM retrain
        rmse = np.sqrt(np.mean((tp_r[metric_start:] - tp_lr[metric_start:])**2, axis=0))
        corr = [np.corrcoef(tp_r[metric_start:, j], tp_lr[metric_start:, j])[0, 1] for j in range(4)]
        lines.append(f"  MLP retrain↔LSTM retrain: RMSE={rmse.mean():.4f}  corr={np.mean(corr):.3f}")

        text = "\n".join(lines)
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.6))
        fig.suptitle('Gait Metrics: All Policy Variants (last 10s)',
                     fontsize=14, fontweight='bold')
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print("  Page 5: Gait metrics")

        # ─── Page 6: Phase portraits (position vs velocity) ──────────────────
        fig, axes = plt.subplots(4, 2, figsize=(14, 12))
        fig.suptitle('Phase Portraits (Position vs Velocity) — Steady State (last 5s)',
                     fontsize=14, fontweight='bold', y=0.98)
        phase_start = N_STEPS - int(5.0 / DT)

        for j, leg in enumerate(LEG_NAMES):
            # Left: MLPs
            ax = axes[j, 0]
            for name in mlp_names:
                tp = results[name]['thigh_pos']
                tv = results[name]['thigh_vel']
                ax.plot(tp[phase_start:, j], tv[phase_start:, j],
                        color=policy_colors[name], ls=policy_ls[name],
                        alpha=0.6, linewidth=0.8, label=name)
            ax.set_ylabel(f'{leg} Vel (rad/s)')
            if j == 0:
                ax.set_title('MLP Variants', fontweight='bold')
                ax.legend(fontsize=7, ncol=3)
            if j == 3:
                ax.set_xlabel('Position (rad)')

            # Right: LSTMs
            ax = axes[j, 1]
            for name in lstm_names:
                tp = results[name]['thigh_pos']
                tv = results[name]['thigh_vel']
                ax.plot(tp[phase_start:, j], tv[phase_start:, j],
                        color=lstm_colors[name],
                        ls='-' if 'original' in name else '--',
                        alpha=0.6, linewidth=0.8, label=name)
            if j == 0:
                ax.set_title('LSTM Variants', fontweight='bold')
                ax.legend(fontsize=7)
            if j == 3:
                ax.set_xlabel('Position (rad)')

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print("  Page 6: Phase portraits")

    print(f"\n[DONE] PDF saved to: {OUTPUT_PDF}")


if __name__ == '__main__':
    main()
