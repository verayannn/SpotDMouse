"""
Sensorimotor Mapping Analysis: Sim vs Real

Compares the observation → action relationship between simulation
and hardware for both MLP and LSTM policies. Fits a simple linear
model (obs_t → action_t) to quantify:
  1. How much of the action variance each observation explains
  2. Where sim and real diverge in the feedback loop
  3. Which observations are most informative vs dead/noisy

Usage:
    python sensorimotor_analysis.py
    python sensorimotor_analysis.py --save-plots

Inputs:
    - Sim logs:  ~/mlp_obs_action_logs/, ~/lstm_obs_action_logs/
    - HW logs:   ~/mp2_mlp/*.csv (preset-named logs)
"""

import csv
import glob
import os
import argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ============================================================
# Data loading
# ============================================================

OBS_NAMES = [
    'base_lin_vel_x', 'base_lin_vel_y', 'base_lin_vel_z',
    'base_ang_vel_x', 'base_ang_vel_y', 'base_ang_vel_z',
    'gravity_x', 'gravity_y', 'gravity_z',
    'cmd_x', 'cmd_y', 'cmd_yaw',
    'jpos_lf_hip', 'jpos_lf_thigh', 'jpos_lf_calf',
    'jpos_rf_hip', 'jpos_rf_thigh', 'jpos_rf_calf',
    'jpos_lb_hip', 'jpos_lb_thigh', 'jpos_lb_calf',
    'jpos_rb_hip', 'jpos_rb_thigh', 'jpos_rb_calf',
    'jvel_lf_hip', 'jvel_lf_thigh', 'jvel_lf_calf',
    'jvel_rf_hip', 'jvel_rf_thigh', 'jvel_rf_calf',
    'jvel_lb_hip', 'jvel_lb_thigh', 'jvel_lb_calf',
    'jvel_rb_hip', 'jvel_rb_thigh', 'jvel_rb_calf',
    'jeff_lf_hip', 'jeff_lf_thigh', 'jeff_lf_calf',
    'jeff_rf_hip', 'jeff_rf_thigh', 'jeff_rf_calf',
    'jeff_lb_hip', 'jeff_lb_thigh', 'jeff_lb_calf',
    'jeff_rb_hip', 'jeff_rb_thigh', 'jeff_rb_calf',
    'prev_act_lf_hip', 'prev_act_lf_thigh', 'prev_act_lf_calf',
    'prev_act_rf_hip', 'prev_act_rf_thigh', 'prev_act_rf_calf',
    'prev_act_lb_hip', 'prev_act_lb_thigh', 'prev_act_lb_calf',
    'prev_act_rb_hip', 'prev_act_rb_thigh', 'prev_act_rb_calf',
]

OBS_GROUPS = {
    'base_lin_vel': (0, 3),
    'base_ang_vel': (3, 6),
    'gravity': (6, 9),
    'commands': (9, 12),
    'joint_pos': (12, 24),
    'joint_vel': (24, 36),
    'joint_effort': (36, 48),
    'prev_actions': (48, 60),
}

ACTION_NAMES = [
    'act_lf_hip', 'act_lf_thigh', 'act_lf_calf',
    'act_rf_hip', 'act_rf_thigh', 'act_rf_calf',
    'act_lb_hip', 'act_lb_thigh', 'act_lb_calf',
    'act_rb_hip', 'act_rb_thigh', 'act_rb_calf',
]


def load_sim_data(log_dir, n_envs=6):
    """Load sim obs+action logs, return (obs_array, action_array)."""
    all_obs, all_act = [], []

    for env_id in range(n_envs):
        obs_path = os.path.join(log_dir, f'env_{env_id}_observations.csv')
        act_path = os.path.join(log_dir, f'env_{env_id}_actions.csv')

        if not os.path.exists(obs_path):
            continue

        with open(obs_path) as f:
            obs_rows = list(csv.DictReader(f))
        with open(act_path) as f:
            act_rows = list(csv.DictReader(f))

        # Extract obs: 60 dims (skip time_step column)
        obs_keys = [k for k in obs_rows[0].keys() if k != 'time_step']
        obs = np.array([[float(r[k]) for k in obs_keys] for r in obs_rows])

        # Extract actions: 12 dims
        act_keys = [k for k in act_rows[0].keys() if k != 'time_step']
        act = np.array([[float(r[k]) for k in act_keys] for r in act_rows])

        # Align lengths
        n = min(len(obs), len(act))
        all_obs.append(obs[:n])
        all_act.append(act[:n])

    return np.concatenate(all_obs), np.concatenate(all_act)


def load_hw_data(csv_path):
    """Load hardware log, return (obs_array, action_array)."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    obs = np.array([[float(r[f'obs_{i}']) for i in range(60)] for r in rows])
    act = np.array([[float(r[f'raw_action_{i}']) for i in range(12)] for r in rows])

    return obs, act


# ============================================================
# Linear sensorimotor model: obs_t → action_t
# ============================================================

def fit_linear_model(obs, actions):
    """Fit obs → action linear model. Returns weights, R^2 per action, R^2 per obs group."""
    # Add bias column
    X = np.column_stack([obs, np.ones(len(obs))])
    n_actions = actions.shape[1]

    # Least squares: W = (X^T X)^{-1} X^T Y
    try:
        W = np.linalg.lstsq(X, actions, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None, None, None

    predictions = X @ W
    residuals = actions - predictions

    # R^2 per action dimension
    ss_res = np.sum(residuals ** 2, axis=0)
    ss_tot = np.sum((actions - actions.mean(axis=0)) ** 2, axis=0)
    r2_per_action = 1 - ss_res / np.maximum(ss_tot, 1e-10)

    # R^2 contribution per observation group
    # Ablation: zero out each group, measure R^2 drop
    r2_importance = {}
    for group_name, (start, end) in OBS_GROUPS.items():
        X_ablated = X.copy()
        X_ablated[:, start:end] = 0.0
        pred_ablated = X_ablated @ W
        ss_res_ablated = np.sum((actions - pred_ablated) ** 2, axis=0)
        r2_ablated = 1 - ss_res_ablated / np.maximum(ss_tot, 1e-10)
        # Importance = how much R^2 drops when this group is removed
        r2_importance[group_name] = np.mean(r2_per_action) - np.mean(r2_ablated)

    return W[:-1], r2_per_action, r2_importance  # Exclude bias row from weights


# ============================================================
# Distribution comparison
# ============================================================

def compare_distributions(sim_obs, hw_obs, sim_act, hw_act):
    """Compare observation and action distributions between sim and hardware."""
    print("\n" + "=" * 90)
    print("OBSERVATION DISTRIBUTION COMPARISON (Sim vs Hardware)")
    print("=" * 90)
    print(f"{'Group':>16s}  {'Sim mean':>10s} {'Sim std':>10s}  {'HW mean':>10s} {'HW std':>10s}  {'Mean gap':>10s} {'Std ratio':>10s}")
    print("-" * 90)

    for group_name, (start, end) in OBS_GROUPS.items():
        sim_vals = sim_obs[:, start:end]
        hw_vals = hw_obs[:, start:end]
        gap = np.abs(sim_vals.mean() - hw_vals.mean())
        std_ratio = hw_vals.std() / max(sim_vals.std(), 1e-6)
        print(f"{group_name:>16s}  {sim_vals.mean():>+10.4f} {sim_vals.std():>10.4f}  "
              f"{hw_vals.mean():>+10.4f} {hw_vals.std():>10.4f}  "
              f"{gap:>10.4f} {std_ratio:>10.2f}x")

    print(f"\n{'Actions':>16s}  {sim_act.mean():>+10.4f} {sim_act.std():>10.4f}  "
          f"{hw_act.mean():>+10.4f} {hw_act.std():>10.4f}  "
          f"{abs(sim_act.mean() - hw_act.mean()):>10.4f} {hw_act.std() / max(sim_act.std(), 1e-6):>10.2f}x")


# ============================================================
# One-step transition analysis: (obs_t, act_t) → obs_{t+1}
# ============================================================

def fit_transition_model(obs, actions):
    """Fit (obs_t, action_t) → obs_{t+1} linear model.

    This captures one timestep of the feedback loop:
    how does the current state + action predict the next state?
    """
    # Align: X = [obs_t, act_t], Y = obs_{t+1}
    X = np.column_stack([obs[:-1], actions[:-1]])
    Y = obs[1:]

    try:
        W = np.linalg.lstsq(X, Y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None, None

    predictions = X @ W
    residuals = Y - predictions

    ss_res = np.sum(residuals ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
    r2_per_obs = 1 - ss_res / np.maximum(ss_tot, 1e-10)

    return W, r2_per_obs


# ============================================================
# Main analysis
# ============================================================

def run_analysis(save_plots=False):
    home = os.path.expanduser("~")

    datasets = {}

    # Load sim data
    for name, log_dir in [
        ('mlp_sim', os.path.join(home, 'mlp_obs_action_logs')),
        ('lstm_sim', os.path.join(home, 'lstm_obs_action_logs')),
    ]:
        if os.path.exists(log_dir):
            obs, act = load_sim_data(log_dir)
            datasets[name] = (obs, act)
            print(f"[LOADED] {name}: {obs.shape[0]} steps")

    # Load hardware data
    hw_dir = os.path.join(home, 'mp2_mlp')
    if os.path.exists(hw_dir):
        for csv_file in sorted(glob.glob(os.path.join(hw_dir, '*.csv'))):
            basename = os.path.basename(csv_file).replace('_hw_log_', '_').rsplit('_', 1)[0]
            # Extract preset name from filename
            preset = basename.split('_20')[0] if '_20' in basename else basename
            obs, act = load_hw_data(csv_file)
            # Use steady state only (skip first 200 steps for fade-in)
            obs, act = obs[200:], act[200:]
            if len(obs) > 10:
                datasets[f'{preset}_hw'] = (obs, act)
                print(f"[LOADED] {preset}_hw: {obs.shape[0]} steps")

    if not datasets:
        print("[ERROR] No data found!")
        return

    # ---- Sensorimotor model: obs → action ----
    print("\n" + "=" * 90)
    print("SENSORIMOTOR MODEL: obs_t → action_t (linear regression)")
    print("=" * 90)

    for name, (obs, act) in sorted(datasets.items()):
        W, r2_action, r2_importance = fit_linear_model(obs, act)
        if r2_action is None:
            continue

        print(f"\n--- {name} ({obs.shape[0]} steps) ---")
        print(f"  Overall R²: {np.mean(r2_action):.4f}")
        print(f"  Per-action R²: min={r2_action.min():.3f} max={r2_action.max():.3f}")
        print(f"  Observation group importance (R² drop when ablated):")
        for group, importance in sorted(r2_importance.items(), key=lambda x: -x[1]):
            bar = '#' * int(importance * 100)
            print(f"    {group:>16s}: {importance:+.4f}  {bar}")

    # ---- Transition model: (obs_t, act_t) → obs_{t+1} ----
    print("\n" + "=" * 90)
    print("TRANSITION MODEL: (obs_t, action_t) → obs_{t+1} (linear, one-step)")
    print("=" * 90)

    for name, (obs, act) in sorted(datasets.items()):
        if len(obs) < 10:
            continue
        W, r2_obs = fit_transition_model(obs, act)
        if r2_obs is None:
            continue

        print(f"\n--- {name} ---")
        print(f"  Overall R²: {np.mean(r2_obs):.4f}")
        print(f"  Per obs group predictability (R² of next-step prediction):")
        for group_name, (start, end) in OBS_GROUPS.items():
            group_r2 = np.mean(r2_obs[start:end])
            bar = '#' * int(max(0, group_r2) * 50)
            print(f"    {group_name:>16s}: {group_r2:.4f}  {bar}")

    # ---- Sim vs Hardware comparison ----
    if 'mlp_sim' in datasets and 'mlp_pd_hw' in datasets:
        sim_obs, sim_act = datasets['mlp_sim']
        hw_obs, hw_act = datasets['mlp_pd_hw']
        print("\n\n>>> MLP: Sim vs Hardware <<<")
        compare_distributions(sim_obs, hw_obs, sim_act, hw_act)

    if 'lstm_sim' in datasets and 'lstm_pd_hw' in datasets:
        sim_obs, sim_act = datasets['lstm_sim']
        hw_obs, hw_act = datasets['lstm_pd_hw']
        print("\n\n>>> LSTM: Sim vs Hardware <<<")
        compare_distributions(sim_obs, hw_obs, sim_act, hw_act)

    # ---- Plots ----
    if save_plots and HAS_PLT:
        _generate_plots(datasets)
        print("\n[PLOTS] Saved to sensorimotor_plots/")


def _generate_plots(datasets):
    """Generate comparison plots."""
    os.makedirs('sensorimotor_plots', exist_ok=True)

    # Plot 1: Observation group importance across all datasets
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 6), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, (name, (obs, act)) in zip(axes, sorted(datasets.items())):
        _, _, r2_imp = fit_linear_model(obs, act)
        if r2_imp is None:
            continue
        groups = sorted(r2_imp.keys(), key=lambda x: -r2_imp[x])
        vals = [r2_imp[g] for g in groups]
        ax.barh(groups, vals)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('R² importance')

    plt.tight_layout()
    plt.savefig('sensorimotor_plots/obs_importance.png', dpi=150)
    plt.close()

    # Plot 2: Obs distribution comparison (sim vs hw) for each group
    sim_hw_pairs = []
    if 'mlp_sim' in datasets and 'mlp_pd_hw' in datasets:
        sim_hw_pairs.append(('MLP', datasets['mlp_sim'], datasets['mlp_pd_hw']))
    if 'lstm_sim' in datasets and 'lstm_pd_hw' in datasets:
        sim_hw_pairs.append(('LSTM', datasets['lstm_sim'], datasets['lstm_pd_hw']))

    for label, (sim_obs, sim_act), (hw_obs, hw_act) in sim_hw_pairs:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ax, (group_name, (start, end)) in zip(axes.flat, OBS_GROUPS.items()):
            sim_flat = sim_obs[:, start:end].flatten()
            hw_flat = hw_obs[:, start:end].flatten()
            ax.hist(sim_flat, bins=50, alpha=0.5, label='Sim', density=True)
            ax.hist(hw_flat, bins=50, alpha=0.5, label='HW', density=True)
            ax.set_title(group_name, fontsize=10)
            ax.legend(fontsize=8)

        fig.suptitle(f'{label}: Sim vs Hardware Observation Distributions', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'sensorimotor_plots/{label.lower()}_obs_distributions.png', dpi=150)
        plt.close()

    print(f"[PLOTS] Generated {2 + len(sim_hw_pairs)} plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-plots", action="store_true", help="Generate PNG plots")
    args = parser.parse_args()

    run_analysis(save_plots=args.save_plots)
