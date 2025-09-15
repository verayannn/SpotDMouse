#!/usr/bin/env python3
"""
Simple comparison: Model output vs Demo 2/4 for forward walking
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

sys.path.append('/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL')

# Load model
print("Loading model...")
checkpoint = torch.load('/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model.pt', map_location='cpu')

from train_il import MLPPolicy
model = MLPPolicy()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get normalization stats (use defaults if missing)
obs_mean = checkpoint.get('obs_mean')
obs_std = checkpoint.get('obs_std')
action_mean = checkpoint.get('action_mean')
action_std = checkpoint.get('action_std')

# Ensure stats are not None
if obs_mean is None:
    print("Warning: obs_mean not found, using zeros")
    obs_mean = torch.zeros(48)
if obs_std is None:
    print("Warning: obs_std not found, using ones")
    obs_std = torch.ones(48)
if action_mean is None:
    print("Warning: action_mean not found, using zeros")
    action_mean = torch.zeros(12)
if action_std is None:
    print("Warning: action_std not found, using ones")
    action_std = torch.ones(12)

# Load demo 2 data (forward walk)
print("\nLoading demo 3 data...")
with h5py.File('/workspace/rosbag_recordings/hdf5_datasets/mini_pupper_demos_20250914_233847.hdf5', 'r') as f:
    demo_obs = f['data/demo_3/obs'][:]
    demo_actions = f['data/demo_3/actions'][:]
    demo_commands = demo_obs[:, :3]

# Find a good forward walking segment (where vx=0.24)
forward_mask = np.abs(demo_commands[:, 0] - 0.24) < 0.05
forward_indices = np.where(forward_mask)[0]

if len(forward_indices) > 100:
    # Take 100 steps from the middle of forward walking
    start_idx = forward_indices[len(forward_indices)//2 - 50]
    end_idx = start_idx + 100
    
    demo_segment_obs = demo_obs[start_idx:end_idx]
    demo_segment_actions = demo_actions[start_idx:end_idx]
    
    print(f"Using forward walking segment: steps {start_idx} to {end_idx}")
else:
    print("Warning: Short forward segment, using first 100 steps")
    demo_segment_obs = demo_obs[:100]
    demo_segment_actions = demo_actions[:100]

# Run model on the same observations
print("Running model on demo observations...")
model_actions = []

with torch.no_grad():
    for obs in demo_segment_obs:
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        obs_norm = (obs_tensor - obs_mean) / obs_std
        
        action = model(obs_norm.unsqueeze(0))
        action = action.squeeze(0) * action_std + action_mean
        
        model_actions.append(action.numpy())

model_actions = np.array(model_actions)

# Create simple comparison plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('Simple Forward Walk Comparison: Demo 2 vs IL Model', fontsize=16)

# Plot 1: Hip joints comparison (shows gait)
ax1 = axes[0]
leg_names = ['FR', 'FL', 'RR', 'RL']
colors = ['red', 'blue', 'green', 'orange']

for i, (leg, color) in enumerate(zip(leg_names, colors)):
    hip_idx = i * 3
    ax1.plot(demo_segment_actions[:, hip_idx], color=color, linestyle='-', 
             label=f'{leg} Demo', linewidth=2, alpha=0.8)
    ax1.plot(model_actions[:, hip_idx], color=color, linestyle='--', 
             label=f'{leg} Model', linewidth=2, alpha=0.6)

ax1.set_xlabel('Time Step')
ax1.set_ylabel('Hip Position (rad)')
ax1.set_title('Hip Joint Comparison (solid=demo, dashed=model)')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot 2: Single leg detail (Front Right)
ax2 = axes[1]
joint_names = ['Hip', 'Thigh', 'Knee']

for j in range(3):
    ax2.plot(demo_segment_actions[:, j], label=f'{joint_names[j]} Demo', linewidth=2)
    ax2.plot(model_actions[:, j], '--', label=f'{joint_names[j]} Model', linewidth=2)

ax2.set_xlabel('Time Step')
ax2.set_ylabel('Joint Position (rad)')
ax2.set_title('Front Right Leg - All Joints')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Error
ax3 = axes[2]
errors = np.abs(demo_segment_actions - model_actions)
mean_error_per_joint = np.mean(errors, axis=0)

x = np.arange(12)
bars = ax3.bar(x, mean_error_per_joint)

# Color by leg
for i in range(12):
    bars[i].set_color(colors[i // 3])

ax3.set_xticks(x)
ax3.set_xticklabels([f'{leg_names[i//3]}-{joint_names[i%3]}' for i in range(12)], rotation=45)
ax3.set_ylabel('Mean Absolute Error (rad)')
ax3.set_title(f'Error Per Joint (Total MAE: {np.mean(errors):.3f} rad)')
ax3.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/simple_forward_comparison.png', dpi=150)
print("\nSaved comparison to /workspace/simple_forward_comparison.png")

# Print summary
print(f"\nSummary:")
print(f"  Average error: {np.mean(errors):.4f} rad")
print(f"  Max error: {np.max(errors):.4f} rad")
print(f"  Command in segment: vx={np.mean(demo_segment_obs[:, 0]):.3f}")

print('\nView with: "$BROWSER" /workspace/simple_forward_comparison.png')