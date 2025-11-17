import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import optimize, signal
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import sys

# Model paths
IL_MLP_FILE = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models_rsl_format/best_model_rsl_format.pt"
RL_MLP_FILE = "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/harvardrun_45/2025-09-21_21-11-46/model_19999.pt"

device = torch.device('cuda:0')

# Load models
IL_MODEL = torch.load(IL_MLP_FILE, map_location=device)
RL_MODEL = torch.load(RL_MLP_FILE, map_location=device)

print("=== ANALYZING LEG ACTION PATTERNS ===")

# First, let's analyze the actual demo actions to understand the patterns
all_demo_actions = []
all_demo_obs = []

with h5py.File('/workspace/rosbag_recordings/hdf5_datasets/mini_pupper_demos_20250914_233847.hdf5', 'r') as f:
    for demo_name in ['demo_1', 'demo_2', 'demo_3', 'demo_4']:
        if f'data/{demo_name}/actions' in f:
            actions = f[f'data/{demo_name}/actions'][:]
            obs = f[f'data/{demo_name}/obs'][:]
            all_demo_actions.append(actions)
            all_demo_obs.append(obs)
            print(f"\n{demo_name}:")
            print(f"  Actions shape: {actions.shape}")
            print(f"  Action range: [{actions.min():.3f}, {actions.max():.3f}] rad")

all_demo_actions = np.vstack(all_demo_actions)
all_demo_obs = np.vstack(all_demo_obs)

import IPython
IPython.embed()

print(IL_MODEL.keys())
print(RL_MODEL.keys())

#make the models the same, 
#put them back in the robot with the propper scaling
#think about a task or just put the models in the robot

sys.exit()

joint_names = ['LF-Hip', 'LF-Thigh', 'LF-Knee', 'RF-Hip', 'RF-Thigh', 'RF-Knee',
               'LB-Hip', 'LB-Thigh', 'LB-Knee', 'RB-Hip', 'RB-Thigh', 'RB-Knee']

# Analyze per-leg patterns
print("\n=== PER-LEG ACTION PATTERNS ===")
leg_names = ['LF (Left Front)', 'RF (Right Front)', 'LB (Left Back)', 'RB (Right Back)']

fig, axes = plt.subplots(4, 3, figsize=(15, 12))
fig.suptitle('Demo Action Patterns by Leg', fontsize=16)

for leg_idx, leg_name in enumerate(leg_names):
    print(f"\n{leg_name}:")
    base_idx = leg_idx * 3
    
    # Get joint actions for this leg
    hip = all_demo_actions[:, base_idx]
    thigh = all_demo_actions[:, base_idx + 1]
    knee = all_demo_actions[:, base_idx + 2]
    
    # Statistics
    print(f"  Hip:   mean={hip.mean():6.3f}, std={hip.std():6.3f}, range=[{hip.min():6.3f}, {hip.max():6.3f}]")
    print(f"  Thigh: mean={thigh.mean():6.3f}, std={thigh.std():6.3f}, range=[{thigh.min():6.3f}, {thigh.max():6.3f}]")
    print(f"  Knee:  mean={knee.mean():6.3f}, std={knee.std():6.3f}, range=[{knee.min():6.3f}, {knee.max():6.3f}]")
    
    # Plot histograms
    axes[leg_idx, 0].hist(hip, bins=50, alpha=0.7, color='blue')
    axes[leg_idx, 0].set_title(f'{leg_name} Hip')
    axes[leg_idx, 0].set_xlabel('Angle (rad)')
    
    axes[leg_idx, 1].hist(thigh, bins=50, alpha=0.7, color='green')
    axes[leg_idx, 1].set_title(f'{leg_name} Thigh')
    axes[leg_idx, 1].set_xlabel('Angle (rad)')
    
    axes[leg_idx, 2].hist(knee, bins=50, alpha=0.7, color='red')
    axes[leg_idx, 2].set_title(f'{leg_name} Knee')
    axes[leg_idx, 2].set_xlabel('Angle (rad)')

plt.tight_layout()
plt.savefig('/workspace/demo_action_patterns.png', dpi=150)
print("\nSaved demo action patterns to: /workspace/demo_action_patterns.png")

# Now let's see how the IL model reproduces these patterns
print("\n=== COMPARING IL MODEL vs DEMO ACTIONS ===")

il_actions = []
demo_actions_subset = []

# Run IL model on demo observations
for idx in range(0, min(len(all_demo_obs), 1000), 10):  # Sample every 10th
    obs = torch.tensor(all_demo_obs[idx], dtype=torch.float32, device=device).unsqueeze(0)
    
    # Normalize for IL model
    il_obs_norm = (obs - IL_MODEL['obs_rms_mean'].to(device)) / torch.sqrt(IL_MODEL['obs_rms_var'].to(device) + 1e-8)
    
    # Forward pass
    def forward_actor(model_dict, obs):
        x = obs.float()
        x = torch.nn.functional.elu(x @ model_dict['actor.0.weight'].T + model_dict['actor.0.bias'])
        x = torch.nn.functional.elu(x @ model_dict['actor.2.weight'].T + model_dict['actor.2.bias'])
        x = torch.nn.functional.elu(x @ model_dict['actor.4.weight'].T + model_dict['actor.4.bias'])
        x = x @ model_dict['actor.6.weight'].T + model_dict['actor.6.bias']
        return x
    
    il_out = forward_actor(IL_MODEL['model_state_dict'], il_obs_norm)
    il_actions.append(il_out[0].cpu().numpy())
    demo_actions_subset.append(all_demo_actions[idx])

il_actions = np.array(il_actions)
demo_actions_subset = np.array(demo_actions_subset)

# Compare action ranges
print("\nAction Range Comparison:")
print(f"{'Joint':<12} {'Demo Min':>8} {'Demo Max':>8} {'IL Min':>8} {'IL Max':>8}")
print("-" * 50)
for i, name in enumerate(joint_names):
    demo_min = demo_actions_subset[:, i].min()
    demo_max = demo_actions_subset[:, i].max()
    il_min = il_actions[:, i].min()
    il_max = il_actions[:, i].max()
    print(f"{name:<12} {demo_min:>8.3f} {demo_max:>8.3f} {il_min:>8.3f} {il_max:>8.3f}")

# Analyze gait patterns
print("\n=== ANALYZING GAIT PATTERNS ===")

# Focus on forward walking segments
forward_mask = (all_demo_obs[:, 0] > 0.2) & (all_demo_obs[:, 0] < 0.3)  # Forward velocity command
forward_actions = all_demo_actions[forward_mask]
forward_obs = all_demo_obs[forward_mask]

if len(forward_actions) > 100:
    print(f"\nFound {len(forward_actions)} forward walking samples")
    
    # Analyze phase relationships between legs
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('Gait Phase Analysis (Forward Walking)', fontsize=16)
    
    # Plot hip joint trajectories for each leg
    time_window = min(200, len(forward_actions))
    t = np.arange(time_window)
    
    for leg_idx, leg_name in enumerate(['LF', 'RF', 'LB', 'RB']):
        ax = axes2[leg_idx // 2, leg_idx % 2]
        
        hip_idx = leg_idx * 3
        hip_trajectory = forward_actions[:time_window, hip_idx]
        
        # Plot raw trajectory
        ax.plot(t, hip_trajectory, 'b-', alpha=0.7, label='Hip')
        
        # Find peaks (stance phase)
        peaks, _ = signal.find_peaks(hip_trajectory, distance=20)
        ax.plot(peaks, hip_trajectory[peaks], 'ro', markersize=8, label='Stance peaks')
        
        # Calculate gait frequency
        if len(peaks) > 1:
            period = np.mean(np.diff(peaks))
            freq = 1.0 / period * 100  # Assuming 100Hz sampling
            ax.text(0.02, 0.98, f'Gait freq: {freq:.2f} Hz', 
                   transform=ax.transAxes, verticalalignment='top')
        
        ax.set_title(f'{leg_name} Hip Trajectory')
        ax.set_xlabel('Time steps')
        ax.set_ylabel('Angle (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/gait_phase_analysis.png')
    print("Saved gait analysis to: /workspace/gait_phase_analysis.png")

# Now create the scaling function based on actual action patterns
print("\n=== CREATING ACTION-BASED SCALING FUNCTION ===")

# The key insight: IL model already outputs correct actions for the real robot
# So we need to find what transformation the RL model applies

# Let's run both models on the same observations and find the relationship
il_outputs = []
rl_outputs = []

for idx in range(0, min(len(all_demo_obs), 500)):
    obs = torch.tensor(all_demo_obs[idx], dtype=torch.float32, device=device).unsqueeze(0)
    
    # Normalize
    il_obs_norm = (obs - IL_MODEL['obs_rms_mean'].to(device)) / torch.sqrt(IL_MODEL['obs_rms_var'].to(device) + 1e-8)
    rl_obs_norm = (obs - torch.tensor(RL_MODEL['obs_rms_mean'], device=device)) / torch.sqrt(torch.tensor(RL_MODEL['obs_rms_var'], device=device) + 1e-8)
    
    il_out = forward_actor(IL_MODEL['model_state_dict'], il_obs_norm)
    rl_out = forward_actor(RL_MODEL['model_state_dict'], rl_obs_norm)
    
    il_outputs.append(il_out[0].cpu().numpy())
    rl_outputs.append(rl_out[0].cpu().numpy())

il_outputs = np.array(il_outputs)

il_outputs = il_outputs
rl_outputs = np.array(rl_outputs)


# Print first few samples as examples
print("\n=== FIRST 5 SAMPLES OF IL OUTPUTS (rad) ===")
print("Sample | LF-Hip  LF-Thigh LF-Knee | RF-Hip  RF-Thigh RF-Knee | LB-Hip  LB-Thigh LB-Knee | RB-Hip  RB-Thigh RB-Knee")
print("-" * 120)
for i in range(min(5, len(il_outputs))):
    print(f"{i:6d} |", end="")
    for leg_idx in range(4):
        base_idx = leg_idx * 3
        print(f" {il_outputs[i, base_idx]:7.4f} {il_outputs[i, base_idx+1]:8.4f} {il_outputs[i, base_idx+2]:7.4f} |", end="")
    print()

# Plot comparison for all joints
fig3, axes3 = plt.subplots(4, 3, figsize=(15, 12))
fig3.suptitle('IL vs RL Model Outputs Comparison', fontsize=16)

for joint_idx in range(12):
    leg_idx = joint_idx // 3
    joint_type_idx = joint_idx % 3
    ax = axes3[leg_idx, joint_type_idx]
    
    ax.plot(il_outputs[:, joint_idx], label='IL', alpha=0.7)
    ax.plot(rl_outputs[:, joint_idx], label='RL', alpha=0.7)
    
    ax.set_title(f'{joint_names[joint_idx]}')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Output')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/il_vs_rl_outputs.png', dpi=150)
print("Saved IL vs RL comparison to: /workspace/il_vs_rl_outputs.png")


# Find per-joint scaling that preserves gait patterns
print("\nPer-Joint Scaling Analysis:")
print(f"{'Joint':<12} {'Scale':>8} {'Offset':>8} {'Type':>15}")
print("-" * 45)

scaling_functions = {}

for joint_idx in range(12):
    il_joint = il_outputs[:, joint_idx]
    rl_joint = rl_outputs[:, joint_idx]
    
    # Remove outliers
    mask = np.abs(rl_joint) < 100
    il_joint = il_joint[mask]
    rl_joint = rl_joint[mask]
    
    if len(il_joint) < 10:
        continue
    
    # Check if relationship is approximately linear
    corr = np.corrcoef(il_joint, rl_joint)[0, 1]
    
    if abs(corr) > 0.7:  # Strong linear relationship
        # Linear fit
        A = np.vstack([il_joint, np.ones(len(il_joint))]).T
        scale, offset = np.linalg.lstsq(A, rl_joint, rcond=None)[0]
        scaling_type = "Linear"
    else:
        # Non-linear - use polynomial
        poly = PolynomialFeatures(degree=2)
        il_poly = poly.fit_transform(il_joint.reshape(-1, 1))
        poly_model = LinearRegression()
        poly_model.fit(il_poly, rl_joint)
        scale = poly_model.coef_[1]  # Linear term
        offset = poly_model.intercept_
        scaling_type = "Polynomial"
    
    print(f"{joint_names[joint_idx]:<12} {scale:>8.2f} {offset:>8.2f} {scaling_type:>15}")
    
    scaling_functions[joint_idx] = {
        'type': scaling_type,
        'scale': scale,
        'offset': offset,
        'corr': corr
    }

# Generate final scaling function
print("\n\n=== FINAL SCALING FUNCTION ===")
print("""
def scale_il_to_sim(il_actions):
    \"\"\"
    Scale IL model outputs to simulation action space.
    Preserves gait patterns while adjusting magnitudes.
    \"\"\"
    scaled_actions = torch.zeros_like(il_actions)
    
    # Per-joint scaling based on action analysis
    scaling = [
        # Scale factors derived from comparing IL and RL outputs on same observations
""")

for i in range(12):
    if i in scaling_functions:
        s = scaling_functions[i]
        print(f"        {s['scale']:.1f},  # {joint_names[i]} ({s['type'].lower()}, corr={s['corr']:.2f})")
    else:
        print(f"        22.0,  # {joint_names[i]} (default)")

print("""    ]
    
    # Apply scaling
    for i in range(12):
        scaled_actions[..., i] = il_actions[..., i] * scaling[i]
    
    # Clip to joint limits (important for simulation stability)
    joint_limits = torch.tensor([
        1.57, 1.57, 2.36,  # LF: hip(±90°), thigh(±90°), knee(±135°)
        1.57, 1.57, 2.36,  # RF
        1.57, 1.57, 2.36,  # LB
        1.57, 1.57, 2.36,  # RB
    ], device=il_actions.device)
    
    scaled_actions = torch.clamp(scaled_actions, -joint_limits, joint_limits)
    
    return scaled_actions
""")

print("\nTo view the analysis plots:")
print('$BROWSER /workspace/demo_action_patterns.png')
print('$BROWSER /workspace/gait_phase_analysis.png')

# Add this after you've computed the scaled actions

# Visualize motion comparison with proper scaling
print("\n=== CREATING MOTION VISUALIZATION WITH PROPER SCALING ===")

# Select a short sequence of forward walking
demo_idx = 100  # Starting point
sequence_length = 200  # 2 seconds at 100Hz

# Get the observation sequence
obs_sequence = all_demo_obs[demo_idx:demo_idx+sequence_length]
demo_actions_sequence = all_demo_actions[demo_idx:demo_idx+sequence_length]

# Run IL model on this sequence
il_actions_sequence = []
il_actions_scaled_sequence = []

for i in range(sequence_length):
    obs = torch.tensor(obs_sequence[i], dtype=torch.float32, device=device).unsqueeze(0)
    
    # Normalize for IL model
    il_obs_norm = (obs - IL_MODEL['obs_rms_mean'].to(device)) / torch.sqrt(IL_MODEL['obs_rms_var'].to(device) + 1e-8)
    
    # Get IL output
    il_out = forward_actor(IL_MODEL['model_state_dict'], il_obs_norm)
    il_actions_sequence.append(il_out[0].cpu().numpy())
    
    # Apply proper scaling based on the analysis
    # Using the scaling factors derived from comparing IL to RL outputs
    scaling_factors = [
        -38.7, 18.2, -13.7,  # LF
        -15.5, 20.0, 0.8,    # RF
        -84.2, 13.8, -11.9,  # LB
        29.1, 28.4, -3.5     # RB
    ]
    
    il_scaled = il_out.clone()
    for j in range(12):
        il_scaled[0, j] = il_out[0, j] * scaling_factors[j]
    
    il_actions_scaled_sequence.append(il_scaled[0].cpu().numpy())

il_actions_sequence = np.array(il_actions_sequence)
il_actions_scaled_sequence = np.array(il_actions_scaled_sequence)

# Create motion comparison plot with separate y-axes
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Motion Comparison: Original IL vs Scaled IL Actions (Note different Y scales!)', fontsize=16)

joint_names = ['LF-Hip', 'LF-Thigh', 'LF-Knee', 'RF-Hip', 'RF-Thigh', 'RF-Knee',
               'LB-Hip', 'LB-Thigh', 'LB-Knee', 'RB-Hip', 'RB-Thigh', 'RB-Knee']

time_steps = np.arange(sequence_length) * 0.01  # Convert to seconds (100Hz)

for joint_idx in range(12):
    row = joint_idx // 4
    col = joint_idx % 4
    ax1 = axes[row, col]
    
    # Create twin axis for scaled version
    ax2 = ax1.twinx()
    
    # Plot original IL output on left y-axis
    line1 = ax1.plot(time_steps, il_actions_sequence[:, joint_idx], 'b-', 
                     label='Original IL', linewidth=2, alpha=0.8)
    ax1.set_ylabel('Original (rad)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot scaled IL output on right y-axis
    line2 = ax2.plot(time_steps, il_actions_scaled_sequence[:, joint_idx], 'r-', 
                     label='Scaled IL', linewidth=2, alpha=0.8)
    ax2.set_ylabel('Scaled (rad)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    ax1.set_title(joint_names[joint_idx])
    ax1.set_xlabel('Time (s)')
    ax1.grid(True, alpha=0.3)
    
    # Add scaling factor info
    scale_factor = scaling_factors[joint_idx]
    ax1.text(0.02, 0.98, f'Scale: {scale_factor:.1f}x', 
            transform=ax1.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig('/workspace/motion_comparison_dual_axes.png', dpi=150)
print("Saved motion comparison (dual axes) to: /workspace/motion_comparison_dual_axes.png")

# Create a comparison showing the dramatic scale difference
fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))
fig2.suptitle('Scale Comparison: IL Original vs Scaled (Left Front Hip)', fontsize=14)

# Focus on one joint to show the scale difference clearly
joint_idx = 0  # LF-Hip

# Top plot: Original scale
ax_top = axes2[0]
ax_top.plot(time_steps, il_actions_sequence[:, joint_idx], 'b-', linewidth=2)
ax_top.set_title('Original IL Output')
ax_top.set_ylabel('Angle (rad)')
ax_top.set_ylim([-0.5, 0.5])
ax_top.grid(True, alpha=0.3)
orig_range = il_actions_sequence[:, joint_idx].max() - il_actions_sequence[:, joint_idx].min()
ax_top.text(0.02, 0.98, f'Range: {orig_range:.3f} rad', 
           transform=ax_top.transAxes, verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

# Bottom plot: Scaled
ax_bottom = axes2[1]
ax_bottom.plot(time_steps, il_actions_scaled_sequence[:, joint_idx], 'r-', linewidth=2)
ax_bottom.set_title(f'Scaled IL Output (×{scaling_factors[joint_idx]:.1f})')
ax_bottom.set_ylabel('Angle (rad)')
ax_bottom.set_xlabel('Time (s)')
ax_bottom.grid(True, alpha=0.3)
scaled_range = il_actions_scaled_sequence[:, joint_idx].max() - il_actions_scaled_sequence[:, joint_idx].min()
ax_bottom.text(0.02, 0.98, f'Range: {scaled_range:.1f} rad (!)', 
              transform=ax_bottom.transAxes, verticalalignment='top',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))

plt.tight_layout()
plt.savefig('/workspace/scale_comparison_dramatic.png', dpi=150)
print("Saved dramatic scale comparison to: /workspace/scale_comparison_dramatic.png")

# Print summary of scaling effect
print("\n=== SCALING EFFECT SUMMARY ===")
print(f"{'Joint':<12} {'Orig Min':>8} {'Orig Max':>8} {'Scaled Min':>10} {'Scaled Max':>10} {'Factor':>8}")
print("-" * 70)
for i, name in enumerate(joint_names):
    orig_min = il_actions_sequence[:, i].min()
    orig_max = il_actions_sequence[:, i].max()
    scaled_min = il_actions_scaled_sequence[:, i].min()
    scaled_max = il_actions_scaled_sequence[:, i].max()
    print(f"{name:<12} {orig_min:>8.3f} {orig_max:>8.3f} {scaled_min:>10.1f} {scaled_max:>10.1f} {scaling_factors[i]:>8.1f}x")

print("\nTo view the visualizations:")
print('$BROWSER /workspace/motion_comparison_dual_axes.png')
print('$BROWSER /workspace/scale_comparison_dramatic.png')