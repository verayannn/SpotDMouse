import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import copy
from matplotlib.patches import Patch

# Model paths
# IL_MLP_FILE = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models_rsl_format/best_model_rsl_format.pt"
RL_MLP_FILE = "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/birthdayrun/2025-08-07_19-17-44/model_9999_with_stats.pt"
IL_MLP_FILE = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models_rsl_format/best_model_rsl_format_scaled_precise.pt"
#might need to make 
device = torch.device('cuda:0')

# Load models
IL_MODEL = torch.load(IL_MLP_FILE, map_location=device)
RL_MODEL = torch.load(RL_MLP_FILE, map_location=device)

print("=== COMPREHENSIVE JOINT ANALYSIS ===")

# Load all demo data
all_demo_obs = []
all_demo_actions = []

with h5py.File('/workspace/rosbag_recordings/hdf5_datasets/mini_pupper_demos_20250914_233847.hdf5', 'r') as f:
    for demo_name in ['demo_1', 'demo_2', 'demo_3', 'demo_4']:
        if f'data/{demo_name}/obs' in f:
            obs = f[f'data/{demo_name}/obs'][:]
            actions = f[f'data/{demo_name}/actions'][:]
            all_demo_obs.append(obs)
            all_demo_actions.append(actions)
            print(f"\n{demo_name}:")
            print(f"  Obs shape: {obs.shape}")
            print(f"  Actions shape: {actions.shape}")

all_demo_obs = np.vstack(all_demo_obs)
all_demo_actions = np.vstack(all_demo_actions)

# Joint names
joint_names = ['LF-Hip', 'LF-Thigh', 'LF-Knee', 'RF-Hip', 'RF-Thigh', 'RF-Knee',
               'LB-Hip', 'LB-Thigh', 'LB-Knee', 'RB-Hip', 'RB-Thigh', 'RB-Knee']


# Forward pass function
def forward_actor(model_dict, obs):
    x = obs.float()
    x = torch.nn.functional.elu(x @ model_dict['actor.0.weight'].T + model_dict['actor.0.bias'])
    x = torch.nn.functional.elu(x @ model_dict['actor.2.weight'].T + model_dict['actor.2.bias'])
    x = torch.nn.functional.elu(x @ model_dict['actor.4.weight'].T + model_dict['actor.4.bias'])
    x = x @ model_dict['actor.6.weight'].T + model_dict['actor.6.bias']
    return x

# First, analyze scaling relationships between IL and RL
print("\n=== ANALYZING IL vs RL SCALING RELATIONSHIPS ===")
il_outputs = []
rl_outputs = []

# Sample observations for analysis
sample_indices = range(0, min(len(all_demo_obs), 1000), 2)
for idx in sample_indices:
    obs = torch.tensor(all_demo_obs[idx], dtype=torch.float32, device=device).unsqueeze(0)
    
    # Normalize for each model
    il_obs_norm = (obs - IL_MODEL['obs_rms_mean'].to(device)) / torch.sqrt(IL_MODEL['obs_rms_var'].to(device) + 1e-8)
    rl_obs_norm = (obs - torch.tensor(RL_MODEL['obs_rms_mean'], device=device)) / torch.sqrt(torch.tensor(RL_MODEL['obs_rms_var'], device=device) + 1e-8)
    
    # Forward pass
    il_out = forward_actor(IL_MODEL['model_state_dict'], il_obs_norm)
    rl_out = forward_actor(RL_MODEL['model_state_dict'], rl_obs_norm)
    
    il_outputs.append(il_out[0].cpu().numpy())
    rl_outputs.append(rl_out[0].cpu().numpy())

il_outputs = np.array(il_outputs)
rl_outputs = np.array(rl_outputs)

# Determine scaling factors based on variance matching
print("\n=== PER-JOINT SCALING FACTORS (VARIANCE MATCHING) ===")
print(f"{'Joint':<12} {'IL std':>8} {'RL std':>8} {'Scale':>8} {'Method':>15}")
print("-" * 50)

scaling_factors = []
for joint_idx in range(12):
    il_joint = il_outputs[:, joint_idx]
    rl_joint = rl_outputs[:, joint_idx]
    
    # Calculate standard deviations
    il_std = np.std(il_joint)
    rl_std = np.std(rl_joint)
    
    # Scale factor to match variance
    if il_std > 1e-6:  # Avoid division by zero
        scale = rl_std / il_std
        method = "Variance match"
    else:
        # If IL has no variance, use a large default scale
        scale = 100.0
        method = "Default (no var)"
    
    scaling_factors.append(scale)
    print(f"{joint_names[joint_idx]:<12} {il_std:>8.4f} {rl_std:>8.1f} {scale:>8.1f} {method:>15}")

# Now analyze centering shifts
print("\n=== COMPREHENSIVE JOINT CENTERING ANALYSIS ===")

# Analyze all joints
joint_shifts = {}
shift_threshold = 5.0  # Only shift if centers are more than 5 units apart

# Create visualization for all joints
fig_all, axes_all = plt.subplots(4, 3, figsize=(18, 16))
fig_all.suptitle('All Joints Centering Analysis', fontsize=16)

print("\nDetailed Joint Analysis:")
print(f"{'Joint':<12} {'RL Center':>10} {'IL Scaled':>10} {'Difference':>10} {'Action':>15}")
print("-" * 60)

for joint_idx in range(12):
    # Get RL values
    rl_joint = rl_outputs[:, joint_idx]
    rl_center = np.mean(rl_joint)
    rl_std = np.std(rl_joint)
    rl_min = np.min(rl_joint)
    rl_max = np.max(rl_joint)
    
    # Get IL values and scale them
    il_joint = il_outputs[:, joint_idx]
    scale_factor = scaling_factors[joint_idx]
    scaled_il_joint = il_joint * scale_factor
    scaled_il_center = np.mean(scaled_il_joint)
    scaled_il_std = np.std(scaled_il_joint)
    scaled_il_min = np.min(scaled_il_joint)
    scaled_il_max = np.max(scaled_il_joint)
    
    # Calculate the shift needed
    shift_needed = rl_center - scaled_il_center
    
    # Determine action
    if abs(shift_needed) > shift_threshold:
        joint_shifts[joint_idx] = shift_needed
        if shift_needed > 0:
            action = f"Shift UP {abs(shift_needed):.1f}"
        else:
            action = f"Shift DOWN {abs(shift_needed):.1f}"
    else:
        joint_shifts[joint_idx] = 0.0
        action = "In range"
    
    print(f"{joint_names[joint_idx]:<12} {rl_center:>10.1f} {scaled_il_center:>10.1f} {shift_needed:>10.1f} {action:>15}")
    
    # Plot distributions
    row = joint_idx // 3
    col = joint_idx % 3
    ax = axes_all[row, col]
    
    # Create bins that cover both distributions
    all_values = np.concatenate([rl_joint, scaled_il_joint])
    bins = np.linspace(np.percentile(all_values, 1), np.percentile(all_values, 99), 50)
    
    # Plot histograms
    ax.hist(rl_joint, bins=bins, alpha=0.5, label='RL', color='blue', density=True)
    ax.hist(scaled_il_joint, bins=bins, alpha=0.5, label='Scaled IL', color='red', density=True)
    
    # If shift is needed, show the shifted distribution
    if abs(joint_shifts[joint_idx]) > shift_threshold:
        shifted_il = scaled_il_joint + joint_shifts[joint_idx]
        ax.hist(shifted_il, bins=bins, alpha=0.5, label=f'Shifted IL', 
                color='green', density=True, linestyle='--', histtype='step', linewidth=2)
    
    # Mark centers
    ax.axvline(rl_center, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(scaled_il_center, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add text info
    ax.set_title(f'{joint_names[joint_idx]} (scale: {scale_factor:.1f}x)')
    ax.set_xlabel('Joint angle')
    ax.set_ylabel('Density')
    
    # Add shift info to plot
    if abs(joint_shifts[joint_idx]) > shift_threshold:
        shift_text = f"Shift: {joint_shifts[joint_idx]:+.1f}"
        ax.text(0.95, 0.95, shift_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor="yellow" if abs(joint_shifts[joint_idx]) > 10 else "lightgreen", 
                         alpha=0.7))
    
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/all_joints_centering_validation.png', dpi=150)
print("\nSaved all joints centering analysis to: /workspace/all_joints_centering_validation.png")

# Now let's visualize the comparison
print("\n=== COMPARING ORIGINAL IL vs SCALED IL vs RL OUTPUTS ===")

# Use forward walking segment for visualization
forward_start = 300
forward_end = 500
viz_obs = all_demo_obs[forward_start:forward_end]

il_outputs_orig = []
il_outputs_scaled = []
rl_outputs_compare = []

# Run all three models on the same observations
for idx in range(len(viz_obs)):
    obs = torch.tensor(viz_obs[idx], dtype=torch.float32, device=device).unsqueeze(0)
    
    # IL original
    il_obs_norm = (obs - IL_MODEL['obs_rms_mean'].to(device)) / torch.sqrt(IL_MODEL['obs_rms_var'].to(device) + 1e-8)
    il_out_orig = forward_actor(IL_MODEL['model_state_dict'], il_obs_norm)
    il_outputs_orig.append(il_out_orig[0].cpu().numpy())
    
    # IL scaled
    # il_out_scaled = forward_actor(scaled_IL_MODEL['model_state_dict'], il_obs_norm)
    # il_outputs_scaled.append(il_out_scaled[0].cpu().numpy())
    
    # RL for comparison
    rl_obs_norm = (obs - torch.tensor(RL_MODEL['obs_rms_mean'], device=device)) / torch.sqrt(torch.tensor(RL_MODEL['obs_rms_var'], device=device) + 1e-8)
    rl_out = forward_actor(RL_MODEL['model_state_dict'], rl_obs_norm)
    rl_outputs_compare.append(rl_out[0].cpu().numpy())

il_outputs_orig = np.array(il_outputs_orig)
il_outputs_scaled = np.array(il_outputs_scaled)
rl_outputs_compare = np.array(rl_outputs_compare)

# Create comparison plots
fig, axes = plt.subplots(4, 3, figsize=(18, 16))
fig.suptitle('Model Output Comparison: Original IL vs Scaled IL vs RL (Precise Joint-Specific Transforms)', fontsize=16)

time_steps = np.arange(len(il_outputs_orig)) * 0.01  # 100Hz

for joint_idx in range(12):
    row = joint_idx // 3
    col = joint_idx % 3
    ax = axes[row, col]
    
    # Plot all three
    ax.plot(time_steps, il_outputs_orig[:, joint_idx], 'b-', label='IL Original', alpha=0.7, linewidth=1)
    # ax.plot(time_steps, il_outputs_scaled[:, joint_idx], 'r-', label='IL Scaled', alpha=0.7, linewidth=2)
    ax.plot(time_steps, rl_outputs_compare[:, joint_idx], 'g--', label='RL Reference', alpha=0.7, linewidth=1.5)
    
    # Get transform info
    scale = scaling_factors[joint_idx]
    shift = joint_shifts.get(joint_idx, 0.0)
    
    ax.set_title(f'{joint_names[joint_idx]} (×{scale:.1f}, shift: {shift:+.1f})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Action Value')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    orig_mean = np.mean(il_outputs_orig[:, joint_idx])
    orig_std = np.std(il_outputs_orig[:, joint_idx])
    # scaled_mean = np.mean(il_outputs_scaled[:, joint_idx])
    # scaled_std = np.std(il_outputs_scaled[:, joint_idx])
    rl_mean = np.mean(rl_outputs_compare[:, joint_idx])
    rl_std = np.std(rl_outputs_compare[:, joint_idx])
    # stats_text = f"Means: {orig_mean:.2f}, {scaled_mean:.1f}, {rl_mean:.1f}\n"
    # stats_text += f"Stds: {orig_std:.2f}, {scaled_std:.1f}, {rl_std:.2f}"
    
    # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=7, 
    #         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

plt.tight_layout()
plt.savefig('/workspace/model_comparison_precise_validation.png', dpi=150)
print("\nSaved comparison plot to: /workspace/model_comparison_precise_validation.png")
