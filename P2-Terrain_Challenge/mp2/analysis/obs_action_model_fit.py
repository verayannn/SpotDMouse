"""
Forward-direction sim-to-real linear relationship analysis.

Three data sources (all cmd_x=0.15):
  1. IsaacLab sim (6 envs, MLP & LSTM)
  2. Hardware logs (MLP pd + LSTM nopd)
  3. Open-loop (synthetic PD actuator)

Key questions answered:
  - What is the linear mapping between sim obs and HW obs?
  - What is the linear mapping between sim actions and HW actions?
  - Which obs/action dimensions are well-correlated across domains?
  - Where does the sim-to-real gap live?
"""

import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
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
    "~/SpotDMouse/P2-Terrain_Challenge/mp2/analysis/obs_action_model_fit.pdf"
)

# ─── Sim config ───────────────────────────────────────────────────────────────
DT = 0.02
SIM_SECONDS = 20.0
N_STEPS = int(SIM_SECONDS / DT)
KP, KD, INERTIA, FRICTION = 70.0, 1.2, 0.20, 0.03
EFFORT_LIMIT = 5.0
DELAY_STEPS = 9
PD_SUBSTEPS = 4
ACTION_SCALE = 1.5

DEFAULT_JOINT_POS = np.array([
    0.0, 0.55, -1.0,  0.0, 0.55, -1.0,
    0.0, 0.55, -1.0,  0.0, 0.55, -1.0,
])
JOINT_LOWER = np.array([
    -0.524, 0.0, -2.356, -0.524, 0.0, -2.356,
    -0.524, 0.0, -2.356, -0.524, 0.0, -2.356,
])
JOINT_UPPER = np.array([
    0.524, 1.396, 0.0, 0.524, 1.396, 0.0,
    0.524, 1.396, 0.0, 0.524, 1.396, 0.0,
])

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
OBS_GROUP_NAMES = list(OBS_GROUPS.keys())

OBS_DIM_NAMES = [
    # lin_vel
    'lin_vel_x', 'lin_vel_y', 'lin_vel_z',
    # ang_vel
    'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
    # gravity
    'grav_x', 'grav_y', 'grav_z',
    # commands
    'cmd_x', 'cmd_y', 'cmd_yaw',
    # joint_pos (12)
    'jpos_LF_hip', 'jpos_LF_thigh', 'jpos_LF_calf',
    'jpos_RF_hip', 'jpos_RF_thigh', 'jpos_RF_calf',
    'jpos_LB_hip', 'jpos_LB_thigh', 'jpos_LB_calf',
    'jpos_RB_hip', 'jpos_RB_thigh', 'jpos_RB_calf',
    # joint_vel (12)
    'jvel_LF_hip', 'jvel_LF_thigh', 'jvel_LF_calf',
    'jvel_RF_hip', 'jvel_RF_thigh', 'jvel_RF_calf',
    'jvel_LB_hip', 'jvel_LB_thigh', 'jvel_LB_calf',
    'jvel_RB_hip', 'jvel_RB_thigh', 'jvel_RB_calf',
    # joint_eff (12)
    'jeff_LF_hip', 'jeff_LF_thigh', 'jeff_LF_calf',
    'jeff_RF_hip', 'jeff_RF_thigh', 'jeff_RF_calf',
    'jeff_LB_hip', 'jeff_LB_thigh', 'jeff_LB_calf',
    'jeff_RB_hip', 'jeff_RB_thigh', 'jeff_RB_calf',
    # prev_act (12)
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
THIGH_IDX = [1, 4, 7, 10]
LEG_NAMES = ['LF', 'RF', 'LB', 'RB']

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


class SyntheticPDActuator:
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


def run_openloop(model, is_lstm=False):
    pda = SyntheticPDActuator()
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
        act_all[t] = action
        pda.step(DEFAULT_JOINT_POS + action * ACTION_SCALE)
        prev_action = action
    return obs_all, act_all


# ─── Per-Dimension Statistics ─────────────────────────────────���───────────────

def compute_dim_stats(data_a, data_b, names):
    """For each dimension: mean, std, correlation, linear fit (slope, intercept, R²).
    Uses the marginal distribution comparison (not time-aligned).
    """
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


# ─── PDF Pages ────────────────────────────────────────────────────────────────

def page_obs_correlation(fig, obs_stats, arch, src_name, tgt_name):
    """Bar chart: per-obs-dimension correlation between two domains."""
    gs = GridSpec(2, 1, hspace=0.5, height_ratios=[2, 1])

    names = [s['name'] for s in obs_stats]
    corrs = [s['corr'] for s in obs_stats]
    r2s = [s['r2'] for s in obs_stats]
    n = len(names)
    x = np.arange(n)

    # Color by obs group
    colors = []
    group_color_map = {
        'lin_vel': '#e41a1c', 'ang_vel': '#377eb8', 'gravity': '#4daf4a',
        'commands': '#984ea3', 'joint_pos': '#ff7f00', 'joint_vel': '#a65628',
        'joint_eff': '#f781bf', 'prev_act': '#999999',
    }
    for i in range(60):
        for g, (slc, _) in OBS_GROUPS.items():
            if slc.start <= i < slc.stop:
                colors.append(group_color_map[g])
                break

    # Top: correlation
    ax = fig.add_subplot(gs[0])
    bars = ax.bar(x, corrs, color=colors, alpha=0.8)
    ax.set_ylabel('Pearson Correlation')
    ax.set_title(f'{arch} — Observation Correlation: {src_name} vs {tgt_name}',
                 fontweight='bold')
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axhline(0.5, color='gray', ls='--', alpha=0.3)
    ax.axhline(-0.5, color='gray', ls='--', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=5)

    # Legend for groups
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, label=g) for g, c in group_color_map.items()]
    ax.legend(handles=legend_patches, fontsize=6, ncol=4, loc='lower right')

    # Bottom: R² of linear fit
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(x, r2s, color=colors, alpha=0.8)
    ax2.set_ylabel('R² (linear fit)')
    ax2.set_xlabel('Observation Dimension')
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=90, fontsize=5)


def page_action_correlation(fig, act_stats, arch, src_name, tgt_name):
    """Per-action-joint: correlation, slope, intercept between two domains."""
    gs = GridSpec(1, 2, wspace=0.35)
    names = [s['name'] for s in act_stats]
    n = len(names)
    x = np.arange(n)

    # Left: correlation and R²
    ax = fig.add_subplot(gs[0])
    corrs = [s['corr'] for s in act_stats]
    r2s = [s['r2'] for s in act_stats]
    w = 0.35
    ax.bar(x - w/2, corrs, w, label='Correlation', color='#1f77b4', alpha=0.8)
    ax.bar(x + w/2, r2s, w, label='R²', color='#ff7f0e', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Value')
    ax.set_title(f'{arch} — Action Correlation\n{src_name} vs {tgt_name}',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)
    ax.axhline(0, color='black', linewidth=0.5)

    # Right: slope and intercept (the actual linear relationship)
    ax2 = fig.add_subplot(gs[1])
    slopes = [s['slope'] for s in act_stats]
    intercepts = [s['intercept'] for s in act_stats]
    ax2.bar(x - w/2, slopes, w, label='Slope', color='#2ca02c', alpha=0.8)
    ax2.bar(x + w/2, intercepts, w, label='Intercept', color='#d62728', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Value')
    ax2.set_title(f'Linear Fit: {tgt_name} ≈ slope·{src_name} + intercept',
                 fontweight='bold', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.axhline(1.0, color='gray', ls='--', alpha=0.3, label='ideal slope=1')
    ax2.axhline(0.0, color='black', linewidth=0.5)


def page_scatter_key_dims(fig, data_a, data_b, stats_list, dim_names,
                          arch, src_name, tgt_name, indices=None):
    """Scatter plots for selected dimensions showing the linear relationship."""
    if indices is None:
        # Pick dims with highest and lowest correlation
        sorted_by_corr = sorted(enumerate(stats_list), key=lambda x: abs(x[1]['corr']), reverse=True)
        # Top 6 most correlated + bottom 2 least
        top = [s[0] for s in sorted_by_corr[:6]]
        bot = [s[0] for s in sorted_by_corr[-2:]]
        indices = top + bot

    n = len(indices)
    cols = 4
    rows = (n + cols - 1) // cols
    gs = GridSpec(rows, cols, hspace=0.6, wspace=0.35)

    n_pts = min(len(data_a), len(data_b))

    for i, dim_idx in enumerate(indices):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        s = stats_list[dim_idx]
        a = data_a[:n_pts, dim_idx]
        b = data_b[:n_pts, dim_idx]
        ax.scatter(a, b, s=3, alpha=0.3, color='#1f77b4')
        # Plot fit line
        if s['std_a'] > 1e-8:
            x_line = np.array([a.min(), a.max()])
            y_line = s['slope'] * x_line + s['intercept']
            ax.plot(x_line, y_line, 'r-', linewidth=1.5, alpha=0.8)
        ax.set_xlabel(src_name, fontsize=7)
        ax.set_ylabel(tgt_name, fontsize=7)
        ax.set_title(f'{dim_names[dim_idx]}\nr={s["corr"]:.2f} slope={s["slope"]:.2f}',
                    fontsize=7, fontweight='bold')
        ax.tick_params(labelsize=6)

    fig.suptitle(f'{arch} — Key Dimensions: {src_name} vs {tgt_name}',
                 fontsize=13, fontweight='bold', y=0.98)


def page_distribution_comparison(fig, data_dict, dim_names, title, indices=None):
    """Overlaid histograms for selected dims across all 3 domains."""
    if indices is None:
        indices = list(range(min(12, len(dim_names))))

    n = len(indices)
    cols = 4
    rows = (n + cols - 1) // cols
    gs = GridSpec(rows, cols, hspace=0.6, wspace=0.35)

    domain_colors = {'Sim': '#1f77b4', 'Hardware': '#d62728', 'Open-Loop': '#2ca02c'}

    for i, dim_idx in enumerate(indices):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        for dom_name, data in data_dict.items():
            vals = data[:, dim_idx]
            ax.hist(vals, bins=40, alpha=0.4, color=domain_colors.get(dom_name, 'gray'),
                    label=f'{dom_name} (μ={vals.mean():.2f})', density=True)
        ax.set_title(dim_names[dim_idx], fontsize=8, fontweight='bold')
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=5)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)


def page_summary_table(fig, all_stats, arch):
    """Text table of all per-dimension linear relationships."""
    ax = fig.add_subplot(111)
    ax.axis('off')

    lines = []
    lines.append(f"{arch} — LINEAR RELATIONSHIPS: Sim/Open-Loop → Hardware")
    lines.append("=" * 90)

    for comparison_name, obs_s, act_s in all_stats:
        lines.append(f"\n{comparison_name}:")
        lines.append("-" * 90)

        lines.append(f"  {'Dimension':25s} {'Corr':>7s} {'Slope':>8s} {'Intcpt':>8s} "
                     f"{'R²':>6s} {'μ_src':>8s} {'μ_tgt':>8s} {'σ_src':>8s} {'σ_tgt':>8s}")
        lines.append(f"  {'─'*25} {'─'*7} {'─'*8} {'─'*8} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

        lines.append("  OBSERVATIONS:")
        for s in obs_s:
            lines.append(f"  {s['name']:25s} {s['corr']:+7.3f} {s['slope']:8.3f} "
                        f"{s['intercept']:8.3f} {s['r2']:6.3f} "
                        f"{s['mean_a']:8.3f} {s['mean_b']:8.3f} "
                        f"{s['std_a']:8.3f} {s['std_b']:8.3f}")

        lines.append("\n  ACTIONS:")
        for s in act_s:
            lines.append(f"  {s['name']:25s} {s['corr']:+7.3f} {s['slope']:8.3f} "
                        f"{s['intercept']:8.3f} {s['r2']:6.3f} "
                        f"{s['mean_a']:8.3f} {s['mean_b']:8.3f} "
                        f"{s['std_a']:8.3f} {s['std_b']:8.3f}")

    text = "\n".join(lines)
    ax.text(0.01, 0.99, text, transform=ax.transAxes,
            fontsize=5.5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.6))
    fig.suptitle(f'{arch} — Full Dimension-by-Dimension Linear Relationships',
                 fontsize=12, fontweight='bold')


def main():
    data = {}  # {arch: {domain: (obs, act)}}

    # ─── Load all data ────────────────────────────────────────────────────
    print("[1/4] Loading sim CSVs...")
    for arch, sim_dir in [('MLP', SIM_MLP_DIR), ('LSTM', SIM_LSTM_DIR)]:
        obs, act = load_sim_envs(sim_dir)
        data.setdefault(arch, {})['Sim'] = (obs, act)
        print(f"  {arch} sim: {obs.shape[0]} steps")

    print("[2/4] Loading hardware CSVs...")
    for arch, hw_path in [('MLP', HW_MLP_CSV), ('LSTM', HW_LSTM_CSV)]:
        obs, act = load_hw_csv(hw_path)
        data.setdefault(arch, {})['Hardware'] = (obs, act)
        print(f"  {arch} HW: {obs.shape[0]} steps")

    print("[3/4] Running open-loop...")
    model_mlp = torch.jit.load(MLP_PT, map_location='cpu'); model_mlp.eval()
    model_lstm = torch.jit.load(LSTM_PT, map_location='cpu'); model_lstm.eval()
    skip = int(5.0 / DT)

    np.random.seed(42)
    obs_m, act_m = run_openloop(model_mlp, is_lstm=False)
    data.setdefault('MLP', {})['Open-Loop'] = (obs_m[skip:], act_m[skip:])

    np.random.seed(42)
    obs_l, act_l = run_openloop(model_lstm, is_lstm=True)
    data.setdefault('LSTM', {})['Open-Loop'] = (obs_l[skip:], act_l[skip:])

    # ─── Compute per-dimension stats ──────────────────────────────────────
    print("[4/4] Computing relationships and generating PDF...")
    comparisons = [
        ('Sim', 'Hardware'),
        ('Open-Loop', 'Hardware'),
        ('Sim', 'Open-Loop'),
    ]

    with PdfPages(OUTPUT_PDF) as pdf:
        for arch in ['MLP', 'LSTM']:
            all_table_stats = []

            for src_name, tgt_name in comparisons:
                obs_a, act_a = data[arch][src_name]
                obs_b, act_b = data[arch][tgt_name]

                obs_stats = compute_dim_stats(obs_a, obs_b, OBS_DIM_NAMES)
                act_stats = compute_dim_stats(act_a, act_b, JOINT_NAMES)

                all_table_stats.append(
                    (f'{src_name} → {tgt_name}', obs_stats, act_stats))

                # Obs correlation bar chart
                fig = plt.figure(figsize=(16, 8))
                page_obs_correlation(fig, obs_stats, arch, src_name, tgt_name)
                pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

                # Action correlation + slope/intercept
                fig = plt.figure(figsize=(14, 6))
                page_action_correlation(fig, act_stats, arch, src_name, tgt_name)
                pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

                # Scatter plots for key obs dims
                fig = plt.figure(figsize=(16, 10))
                page_scatter_key_dims(fig, obs_a, obs_b, obs_stats, OBS_DIM_NAMES,
                                      arch, src_name, tgt_name)
                pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

                # Scatter plots for all 12 action dims
                fig = plt.figure(figsize=(16, 8))
                page_scatter_key_dims(fig, act_a, act_b, act_stats, JOINT_NAMES,
                                      arch, src_name, tgt_name,
                                      indices=list(range(12)))
                pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

                print(f"  {arch} {src_name}→{tgt_name}: done")

            # Distribution histograms for obs (thigh-related dims)
            thigh_obs_idx = [13, 16, 19, 22,   # jpos thighs
                             25, 28, 31, 34,   # jvel thighs
                             37, 40, 43, 46]   # jeff thighs
            obs_dict = {d: data[arch][d][0] for d in ['Sim', 'Hardware', 'Open-Loop']}
            fig = plt.figure(figsize=(16, 8))
            page_distribution_comparison(fig, obs_dict, OBS_DIM_NAMES,
                f'{arch} — Thigh Obs Distributions Across Domains',
                indices=thigh_obs_idx)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            # Distribution histograms for actions (thigh joints)
            act_dict = {d: data[arch][d][1] for d in ['Sim', 'Hardware', 'Open-Loop']}
            fig = plt.figure(figsize=(16, 6))
            page_distribution_comparison(fig, act_dict, JOINT_NAMES,
                f'{arch} — Action Distributions Across Domains',
                indices=list(range(12)))
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            # Summary table
            fig = plt.figure(figsize=(14, 18))
            page_summary_table(fig, all_table_stats, arch)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            print(f"  {arch} summary table: done")

    print(f"\n[DONE] PDF saved to: {OUTPUT_PDF}")


if __name__ == '__main__':
    main()
