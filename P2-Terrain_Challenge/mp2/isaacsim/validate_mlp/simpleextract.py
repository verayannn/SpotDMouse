import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Direct path to the data
specific_path = "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/harvardrun_45/2025-11-18_00-57-12/obs_action_logs_x_015/"

# Load simulation data
print(f"Loading data from: {specific_path}")

# Load env_0 (forward movement) for detailed analysis
try:
    df_obs = pd.read_csv(f"{specific_path}env_0_observations.csv")
    df_act = pd.read_csv(f"{specific_path}env_0_actions.csv")
    print(f"Successfully loaded env_0 data")
    print(f"Observations shape: {df_obs.shape}")
    print(f"Actions shape: {df_act.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Define leg structure
legs = {
    'LF': ['action_base_lf1', 'action_lf1_lf2', 'action_lf2_lf3'],  # Left Front
    'RF': ['action_base_rf1', 'action_rf1_rf2', 'action_rf2_rf3'],  # Right Front
    'LB': ['action_base_lb1', 'action_lb1_lb2', 'action_lb2_lb3'],  # Left Back
    'RB': ['action_base_rb1', 'action_rb1_rb2', 'action_rb2_rb3']   # Right Back
}

# Joint names for display
joint_names = ['Hip', 'Thigh', 'Calf']

# Create per-limb analysis figure
fig, axes = plt.subplots(4, 3, figsize=(18, 16))
fig.suptitle('Per-Limb Action Analysis - Forward Movement (vx=0.15)', fontsize=16)

# Colors for each joint
colors = ['red', 'green', 'blue']

# Plot each leg's actions
for leg_idx, (leg_name, joint_cols) in enumerate(legs.items()):
    for joint_idx, col in enumerate(joint_cols):
        ax = axes[leg_idx, joint_idx]
        
        # Plot action values
        ax.plot(df_act['time_step'], df_act[col], color=colors[joint_idx], linewidth=1.5)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Formatting
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Action Value')
        ax.set_title(f'{leg_name} - {joint_names[joint_idx]}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2, 2)  # Adjust based on your action range
        
        # Add statistics
        mean_val = df_act[col].mean()
        std_val = df_act[col].std()
        ax.text(0.05, 0.95, f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save the per-limb plot
output_path = '/workspace/SpotDMouse/P2-Terrain_Challenge/mp2/isaacsim/validate_mlp/per_limb_actions.png'
plt.savefig(output_path, dpi=150)
print(f"\nPer-limb plot saved to: {output_path}")
print(f"Use: $BROWSER file://{output_path}")

# Create a phase analysis figure
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('Leg Phase Analysis - Forward Movement', fontsize=16)

# Plot hip joint comparisons
ax = axes2[0, 0]
ax.plot(df_act['time_step'], df_act['action_base_lf1'], label='LF', color='red')
ax.plot(df_act['time_step'], df_act['action_base_rf1'], label='RF', color='green')
ax.plot(df_act['time_step'], df_act['action_base_lb1'], label='LB', color='blue')
ax.plot(df_act['time_step'], df_act['action_base_rb1'], label='RB', color='orange')
ax.set_xlabel('Time Step')
ax.set_ylabel('Hip Action')
ax.set_title('All Hip Joints Comparison')
ax.legend()
ax.grid(True)

# Plot diagonal pairs
ax = axes2[0, 1]
ax.plot(df_act['time_step'], df_act['action_base_lf1'], label='LF Hip', color='red', linestyle='-')
ax.plot(df_act['time_step'], df_act['action_base_rb1'], label='RB Hip', color='red', linestyle='--')
ax.plot(df_act['time_step'], df_act['action_base_rf1'], label='RF Hip', color='blue', linestyle='-')
ax.plot(df_act['time_step'], df_act['action_base_lb1'], label='LB Hip', color='blue', linestyle='--')
ax.set_xlabel('Time Step')
ax.set_ylabel('Hip Action')
ax.set_title('Diagonal Pairs (Trot Pattern)')
ax.legend()
ax.grid(True)

# Plot gait cycle analysis (zoom in on 100 timesteps)
start_idx = 100
end_idx = 200
ax = axes2[1, 0]
ax.plot(df_act['time_step'][start_idx:end_idx], df_act['action_base_lf1'][start_idx:end_idx], label='LF', color='red', marker='.')
ax.plot(df_act['time_step'][start_idx:end_idx], df_act['action_base_rf1'][start_idx:end_idx], label='RF', color='green', marker='.')
ax.plot(df_act['time_step'][start_idx:end_idx], df_act['action_base_lb1'][start_idx:end_idx], label='LB', color='blue', marker='.')
ax.plot(df_act['time_step'][start_idx:end_idx], df_act['action_base_rb1'][start_idx:end_idx], label='RB', color='orange', marker='.')
ax.set_xlabel('Time Step')
ax.set_ylabel('Hip Action')
ax.set_title(f'Gait Detail (Steps {start_idx}-{end_idx})')
ax.legend()
ax.grid(True)

# Plot action distribution per leg
ax = axes2[1, 1]
for leg_name, joint_cols in legs.items():
    hip_actions = df_act[joint_cols[0]].values
    ax.hist(hip_actions, bins=30, alpha=0.5, label=leg_name, density=True)
ax.set_xlabel('Hip Action Value')
ax.set_ylabel('Density')
ax.set_title('Hip Action Distribution by Leg')
ax.legend()
ax.grid(True)

plt.tight_layout()

# Save the phase analysis plot
output_path2 = '/workspace/SpotDMouse/P2-Terrain_Challenge/mp2/isaacsim/validate_mlp/phase_analysis.png'
plt.savefig(output_path2, dpi=150)
print(f"\nPhase analysis plot saved to: {output_path2}")
print(f"Use: $BROWSER file://{output_path2}")

# Create a summary statistics table
print("\n" + "="*60)
print("ACTION SUMMARY STATISTICS")
print("="*60)
print(f"{'Joint':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("-"*60)

for leg_name, joint_cols in legs.items():
    print(f"\n{leg_name} Leg:")
    for i, col in enumerate(joint_cols):
        stats = df_act[col].describe()
        print(f"  {joint_names[i]:<23} {stats['mean']:>10.3f} {stats['std']:>10.3f} {stats['min']:>10.3f} {stats['max']:>10.3f}")

# Calculate phase relationships
print("\n" + "="*60)
print("PHASE RELATIONSHIPS (Hip joints)")
print("="*60)

# Calculate correlations between hip joints
hip_joints = ['action_base_lf1', 'action_base_rf1', 'action_base_lb1', 'action_base_rb1']
hip_data = df_act[hip_joints]
corr_matrix = hip_data.corr()

print("\nCorrelation Matrix:")
print("       LF     RF     LB     RB")
for i, leg in enumerate(['LF', 'RF', 'LB', 'RB']):
    print(f"{leg}: ", end="")
    for j in range(4):
        print(f"{corr_matrix.iloc[i,j]:>6.3f} ", end="")
    print()

print("\nExpected for trot gait: LF-RB and RF-LB should be positively correlated")
print(f"LF-RB correlation: {corr_matrix.loc['action_base_lf1', 'action_base_rb1']:.3f}")
print(f"RF-LB correlation: {corr_matrix.loc['action_base_rf1', 'action_base_lb1']:.3f}")