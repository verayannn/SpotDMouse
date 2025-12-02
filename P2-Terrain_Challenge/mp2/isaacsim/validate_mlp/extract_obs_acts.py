import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define CORRECT paths based on your file system
realbot_path = "/Users/javierweddington/isaaclabstock/realbot_compare/realbot_obs_action_logs/"
sim_path = '/Users/javierweddington/isaaclabstock/directional_obs_action_logs/obs_action_logs_x_015/'#"/Users/javierweddington/isaaclabstock/isaaclabstock/obs_action_logs/"

# Load all 4 movements for real robot
movements = ['Forward', 'Backward', 'Left', 'Right']
real_data = {}
sim_data = {}

# Load real robot data (4 environments)
for i in range(4):
    try:
        real_data[f'obs_{i}'] = pd.read_csv(f"{realbot_path}env_{i}_observations.csv")
        real_data[f'act_{i}'] = pd.read_csv(f"{realbot_path}env_{i}_actions.csv")
        print(f"Loaded real robot env_{i} successfully")
    except Exception as e:
        print(f"Error loading real robot env_{i}: {e}")

# Load simulation data - we'll use the first 4 for comparison
for i in range(4):
    try:
        sim_data[f'obs_{i}'] = pd.read_csv(f"{sim_path}env_{i}_observations.csv")
        sim_data[f'act_{i}'] = pd.read_csv(f"{sim_path}env_{i}_actions.csv")
        print(f"Loaded simulation env_{i} successfully")
    except Exception as e:
        print(f"Error loading simulation env_{i}: {e}")

# For detailed analysis, let's focus on forward movement (env_0)
df_obs = real_data['obs_0']
df_act = real_data['act_0']
df_obs_sim = sim_data['obs_0']
df_act_sim = sim_data['act_0']

print(f"\nReal robot data shape: {df_obs.shape}")
print(f"Simulation data shape: {df_obs_sim.shape}")

# Create figure with subplots for comprehensive analysis
fig = plt.figure(figsize=(20, 16))

# 1. Base velocities (Note: real robot has 0 for linear velocities)
ax1 = plt.subplot(4, 3, 1)
ax1.plot(df_obs['time_step'], df_obs['base_lin_vel_x'], label='Linear X', color='red')
ax1.plot(df_obs['time_step'], df_obs['base_lin_vel_y'], label='Linear Y', color='green')
ax1.plot(df_obs['time_step'], df_obs['base_lin_vel_z'], label='Linear Z', color='blue')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Velocity (m/s)')
ax1.set_title('Real Robot: Base Linear Velocities')
ax1.legend()
ax1.grid(True)

# 2. Base angular velocities (from IMU)
ax2 = plt.subplot(4, 3, 2)
ax2.plot(df_obs['time_step'], df_obs['base_ang_vel_x'], label='Angular X', color='red')
ax2.plot(df_obs['time_step'], df_obs['base_ang_vel_y'], label='Angular Y', color='green')
ax2.plot(df_obs['time_step'], df_obs['base_ang_vel_z'], label='Angular Z', color='blue')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Angular Velocity (rad/s)')
ax2.set_title('Real Robot: Base Angular Velocities (IMU)')
ax2.legend()
ax2.grid(True)

# 3. Velocity commands
ax3 = plt.subplot(4, 3, 3)
ax3.plot(df_obs['time_step'], df_obs['velocity_command_x'], label='Command X', linestyle='--', color='red')
ax3.plot(df_obs['time_step'], df_obs['velocity_command_y'], label='Command Y', linestyle='--', color='green')
ax3.plot(df_obs['time_step'], df_obs['velocity_command_yaw'], label='Command Yaw', linestyle='--', color='blue')
ax3.set_xlabel('Time Step')
ax3.set_ylabel('Command')
ax3.set_title('Real Robot: Velocity Commands')
ax3.legend()
ax3.grid(True)

# 4. Joint positions for all legs
ax4 = plt.subplot(4, 3, 4)
for leg in ['lf', 'rf', 'lb', 'rb']:
    ax4.plot(df_obs['time_step'], df_obs[f'joint_pos_base_{leg}1'], label=f'{leg.upper()} Base')
ax4.set_xlabel('Time Step')
ax4.set_ylabel('Position (rad)')
ax4.set_title('Real Robot: Base Joint Positions (All Legs)')
ax4.legend()
ax4.grid(True)

# 5. Actions for all legs (base joints)
ax5 = plt.subplot(4, 3, 5)
for leg in ['lf', 'rf', 'lb', 'rb']:
    ax5.plot(df_act['time_step'], df_act[f'action_base_{leg}1'], label=f'{leg.upper()} Base')
ax5.set_xlabel('Time Step')
ax5.set_ylabel('Action')
ax5.set_title('Real Robot: Base Joint Actions (All Legs)')
ax5.legend()
ax5.grid(True)

# 6. Joint efforts analysis
ax6 = plt.subplot(4, 3, 6)
for leg in ['lf', 'rf', 'lb', 'rb']:
    leg_effort = df_obs[[f'joint_effort_base_{leg}1', f'joint_effort_{leg}1_{leg}2', 
                        f'joint_effort_{leg}2_{leg}3']].mean(axis=1)
    ax6.plot(df_obs['time_step'], leg_effort, label=f'{leg.upper()}')
ax6.set_xlabel('Time Step')
ax6.set_ylabel('Average Effort')
ax6.set_title('Real Robot: Average Joint Efforts per Leg')
ax6.legend()
ax6.grid(True)

# 7. Projected gravity
ax7 = plt.subplot(4, 3, 7)
ax7.plot(df_obs['time_step'], df_obs['projected_gravity_x'], label='X', color='red')
ax7.plot(df_obs['time_step'], df_obs['projected_gravity_y'], label='Y', color='green')
ax7.plot(df_obs['time_step'], df_obs['projected_gravity_z'], label='Z', color='blue')
ax7.set_xlabel('Time Step')
ax7.set_ylabel('Projected Gravity')
ax7.set_title('Real Robot: Projected Gravity Components')
ax7.legend()
ax7.grid(True)

# 8. Action vs Previous Action (example for LF leg)
ax8 = plt.subplot(4, 3, 8)
ax8.plot(df_act['time_step'], df_act['action_base_lf1'], label='Current Action', alpha=0.7)
ax8.plot(df_obs['time_step'], df_obs['prev_action_base_lf1'], label='Previous Action', alpha=0.7)
ax8.set_xlabel('Time Step')
ax8.set_ylabel('Action')
ax8.set_title('Real Robot LF Base Joint: Action vs Previous Action')
ax8.legend()
ax8.grid(True)

# 9. Joint velocities distribution
ax9 = plt.subplot(4, 3, 9)
vel_cols = [col for col in df_obs.columns if 'joint_vel' in col]
vel_data = df_obs[vel_cols].values.flatten()
ax9.hist(vel_data, bins=50, alpha=0.7, edgecolor='black', color='blue', label='Real Robot')
# Add simulation for comparison
sim_vel_data = df_obs_sim[vel_cols].values.flatten()
ax9.hist(sim_vel_data, bins=50, alpha=0.5, edgecolor='black', color='red', label='Simulation')
ax9.set_xlabel('Joint Velocity (rad/s)')
ax9.set_ylabel('Frequency')
ax9.set_title('Joint Velocities Distribution: Real vs Sim')
ax9.legend()
ax9.grid(True)

# 10. Action distribution
ax10 = plt.subplot(4, 3, 10)
action_cols = [col for col in df_act.columns if col != 'time_step']
action_data = df_act[action_cols].values.flatten()
ax10.hist(action_data, bins=50, alpha=0.7, edgecolor='black', color='blue', label='Real Robot')
# Add simulation for comparison
sim_action_data = df_act_sim[action_cols].values.flatten()
ax10.hist(sim_action_data, bins=50, alpha=0.5, edgecolor='black', color='red', label='Simulation')
ax10.set_xlabel('Action Value')
ax10.set_ylabel('Frequency')
ax10.set_title('Actions Distribution: Real vs Sim')
ax10.legend()
ax10.grid(True)

# 11. Comparison across movements
ax11 = plt.subplot(4, 3, 11)
for i, movement in enumerate(movements):
    if f'obs_{i}' in real_data:
        obs_data = real_data[f'obs_{i}']
        act_data = real_data[f'act_{i}']
        mean_action = act_data[action_cols].abs().mean(axis=1)
        ax11.plot(act_data['time_step'], mean_action, label=movement)
ax11.set_xlabel('Time Step')
ax11.set_ylabel('Mean Absolute Action')
ax11.set_title('Real Robot: Movement Comparison')
ax11.legend()
ax11.grid(True)

# 12. Summary statistics
ax12 = plt.subplot(4, 3, 12)
ax12.axis('off')
stats_text = f"""Real Robot Summary Statistics (Forward Movement):
Total timesteps: {len(df_obs)}
Avg angular vel X: {df_obs['base_ang_vel_x'].mean():.3f} ± {df_obs['base_ang_vel_x'].std():.3f}
Avg angular vel Y: {df_obs['base_ang_vel_y'].mean():.3f} ± {df_obs['base_ang_vel_y'].std():.3f}
Avg angular vel Z: {df_obs['base_ang_vel_z'].mean():.3f} ± {df_obs['base_ang_vel_z'].std():.3f}
Velocity cmd X: {df_obs['velocity_command_x'].mean():.3f}
Velocity cmd Y: {df_obs['velocity_command_y'].mean():.3f}
Action range: [{action_data.min():.3f}, {action_data.max():.3f}]
Joint vel range: [{vel_data.min():.3f}, {vel_data.max():.3f}]

Movement samples:
Forward: {len(real_data['obs_0'])} | Backward: {len(real_data['obs_1'])}
Left: {len(real_data['obs_2'])} | Right: {len(real_data['obs_3'])}
"""
ax12.text(0.1, 0.5, stats_text, transform=ax12.transAxes, fontsize=10, 
          verticalalignment='center', fontfamily='monospace')
ax12.set_title('Summary Statistics')

plt.suptitle('Real Robot Data Analysis (CSV Format)', fontsize=14)
plt.tight_layout()
plt.show()

# Create a second figure for direct real vs simulation comparison
fig2 = plt.figure(figsize=(20, 12))

# 1. Joint position comparison
ax1 = plt.subplot(3, 3, 1)
ax1.plot(df_obs['time_step'], df_obs['joint_pos_base_lf1'], label='Real LF', color='blue')
ax1.plot(df_obs_sim['time_step'], df_obs_sim['joint_pos_base_lf1'], label='Sim LF', color='red', alpha=0.7)
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Position (rad)')
ax1.set_title('LF Base Joint Position: Real vs Simulation')
ax1.legend()
ax1.grid(True)

# 2. Action comparison
ax2 = plt.subplot(3, 3, 2)
ax2.plot(df_act['time_step'], df_act['action_base_lf1'], label='Real', color='blue')
ax2.plot(df_act_sim['time_step'], df_act_sim['action_base_lf1'], label='Sim', color='red', alpha=0.7)
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Action')
ax2.set_title('LF Base Joint Action: Real vs Simulation')
ax2.legend()
ax2.grid(True)

# 3. Phase portrait comparison
ax3 = plt.subplot(3, 3, 3)
ax3.plot(df_obs['joint_pos_base_lf1'], df_obs['joint_vel_base_lf1'], 
         'b-', alpha=0.7, label='Real Robot')
ax3.plot(df_obs_sim['joint_pos_base_lf1'], df_obs_sim['joint_vel_base_lf1'], 
         'r-', alpha=0.7, label='Simulation')
ax3.set_xlabel('Position (rad)')
ax3.set_ylabel('Velocity (rad/s)')
ax3.set_title('Phase Portrait: LF Base Joint')
ax3.legend()
ax3.grid(True)

# 4. Angular velocity comparison
ax4 = plt.subplot(3, 3, 4)
ax4.plot(df_obs['time_step'], df_obs['base_ang_vel_z'], label='Real', color='blue')
ax4.plot(df_obs_sim['time_step'], df_obs_sim['base_ang_vel_z'], label='Sim', color='red', alpha=0.7)
ax4.set_xlabel('Time Step')
ax4.set_ylabel('Angular Velocity Z (rad/s)')
ax4.set_title('Base Angular Velocity Z: Real vs Simulation')
ax4.legend()
ax4.grid(True)

# 5. All movements action magnitude
ax5 = plt.subplot(3, 3, 5)
for i, movement in enumerate(movements):
    if f'act_{i}' in real_data:
        act_data = real_data[f'act_{i}']
        action_magnitude = np.linalg.norm(act_data[action_cols].values, axis=1)
        ax5.plot(act_data['time_step'], action_magnitude, label=f'{movement}')
ax5.set_xlabel('Time Step')
ax5.set_ylabel('Action Magnitude')
ax5.set_title('Real Robot: Action Magnitude by Movement')
ax5.legend()
ax5.grid(True)

# 6. Power spectral density
ax6 = plt.subplot(3, 3, 6)
from scipy import signal
# Calculate sampling frequency
dt = np.mean(np.diff(df_obs['time_step'].values))
fs = 1/dt if dt > 0 else 1
f_real, psd_real = signal.periodogram(df_obs['joint_pos_base_lf1'], fs)
f_sim, psd_sim = signal.periodogram(df_obs_sim['joint_pos_base_lf1'], fs)
ax6.semilogy(f_real, psd_real, label='Real Robot', color='blue')
ax6.semilogy(f_sim, psd_sim, label='Simulation', color='red', alpha=0.7)
ax6.set_xlabel('Frequency (Hz)')
ax6.set_ylabel('PSD')
ax6.set_title('Joint Position PSD: Real vs Simulation')
ax6.legend()
ax6.grid(True)

# 7. Action smoothness
ax7 = plt.subplot(3, 3, 7)
real_action_diff = np.diff(df_act[action_cols].values, axis=0)
real_smoothness = np.linalg.norm(real_action_diff, axis=1)
sim_action_diff = np.diff(df_act_sim[action_cols].values, axis=0)
sim_smoothness = np.linalg.norm(sim_action_diff, axis=1)
ax7.plot(df_act['time_step'].values[1:], real_smoothness, label='Real', color='blue')
ax7.plot(df_act_sim['time_step'].values[1:], sim_smoothness, label='Sim', color='red', alpha=0.7)
ax7.set_xlabel('Time Step')
ax7.set_ylabel('Action Change Magnitude')
ax7.set_title('Action Smoothness: Real vs Simulation')
ax7.legend()
ax7.grid(True)

# 8. Projected gravity magnitude
ax8 = plt.subplot(3, 3, 8)
real_grav_mag = np.sqrt(df_obs['projected_gravity_x']**2 + 
                       df_obs['projected_gravity_y']**2 + 
                       df_obs['projected_gravity_z']**2)
sim_grav_mag = np.sqrt(df_obs_sim['projected_gravity_x']**2 + 
                      df_obs_sim['projected_gravity_y']**2 + 
                      df_obs_sim['projected_gravity_z']**2)
ax8.plot(df_obs['time_step'], real_grav_mag, label='Real', color='blue')
ax8.plot(df_obs_sim['time_step'], sim_grav_mag, label='Sim', color='red', alpha=0.7)
ax8.set_xlabel('Time Step')
ax8.set_ylabel('Magnitude')
ax8.set_title('Projected Gravity Magnitude')
ax8.legend()
ax8.grid(True)

# 9. Statistical comparison
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
# Calculate statistics for comparison
real_action_mean = action_data.mean()
real_action_std = action_data.std()
sim_action_mean = sim_action_data.mean()
sim_action_std = sim_action_data.std()

real_vel_mean = vel_data.mean()
real_vel_std = vel_data.std()
sim_vel_mean = sim_vel_data.mean()
sim_vel_std = sim_vel_data.std()

# Safe calculation of dt for real and sim
dt_real = np.mean(np.diff(df_obs['time_step'].values)) if len(df_obs) > 1 else 1.0
dt_sim = np.mean(np.diff(df_obs_sim['time_step'].values)) if len(df_obs_sim) > 1 else 1.0

comparison_text = f"""Real vs Simulation Comparison:

Action Statistics:
Real: {real_action_mean:.3f} ± {real_action_std:.3f}
Sim:  {sim_action_mean:.3f} ± {sim_action_std:.3f}

Joint Velocity Statistics:
Real: {real_vel_mean:.3f} ± {real_vel_std:.3f}
Sim:  {sim_vel_mean:.3f} ± {sim_vel_std:.3f}

Sample Rates:
Real: {1/dt_real:.1f} Hz
Sim:  {1/dt_sim:.1f} Hz
"""
ax9.text(0.1, 0.5, comparison_text, transform=ax9.transAxes, fontsize=11, 
         verticalalignment='center', fontfamily='monospace')
ax9.set_title('Statistical Comparison')

plt.suptitle('Real Robot vs Simulation Comparison', fontsize=14)
plt.tight_layout()
plt.show()

# Print summary for all movements
print("\nSummary for all movements:")
print("-" * 50)
for i, movement in enumerate(movements):
    if f'obs_{i}' in real_data and f'act_{i}' in real_data:
        obs = real_data[f'obs_{i}']
        act = real_data[f'act_{i}']
        print(f"\n{movement} Movement:")
        print(f"  Samples: {len(obs)}")
        print(f"  Duration: {obs['time_step'].iloc[-1]:.2f} time steps")
        print(f"  Avg Cmd X: {obs['velocity_command_x'].mean():.3f}")
        print(f"  Avg Cmd Y: {obs['velocity_command_y'].mean():.3f}")
        print(f"  Action range: [{act[action_cols].min().min():.3f}, {act[action_cols].max().max():.3f}]")

# Additional info about simulation environments 4 and 5 if needed
print("\n\nNote: Simulation has environments 0-5, real robot has environments 0-3")

# ...existing code...

# Additional info about simulation environments 4 and 5 if needed
print("\n\nNote: Simulation has environments 0-5, real robot has environments 0-3")

# Add detailed observation and action range comparison
print("\n\n" + "="*80)
print("DETAILED OBSERVATION AND ACTION RANGE COMPARISON (Forward Movement - env_0)")
print("="*80)

# Observation columns to compare
obs_columns_to_compare = [
    # Base velocities
    'base_lin_vel_x', 'base_lin_vel_y', 'base_lin_vel_z',
    'base_ang_vel_x', 'base_ang_vel_y', 'base_ang_vel_z',
    # Projected gravity
    'projected_gravity_x', 'projected_gravity_y', 'projected_gravity_z',
    # Velocity commands
    'velocity_command_x', 'velocity_command_y', 'velocity_command_yaw',
    # Joint positions
    'joint_pos_base_lf1', 'joint_pos_lf1_lf2', 'joint_pos_lf2_lf3',
    'joint_pos_base_rf1', 'joint_pos_rf1_rf2', 'joint_pos_rf2_rf3',
    'joint_pos_base_lb1', 'joint_pos_lb1_lb2', 'joint_pos_lb2_lb3',
    'joint_pos_base_rb1', 'joint_pos_rb1_rb2', 'joint_pos_rb2_rb3',
    # Joint velocities
    'joint_vel_base_lf1', 'joint_vel_lf1_lf2', 'joint_vel_lf2_lf3',
    'joint_vel_base_rf1', 'joint_vel_rf1_rf2', 'joint_vel_rf2_rf3',
    'joint_vel_base_lb1', 'joint_vel_lb1_lb2', 'joint_vel_lb2_lb3',
    'joint_vel_base_rb1', 'joint_vel_rb1_rb2', 'joint_vel_rb2_rb3',
]

print("\nOBSERVATION VALUE RANGES:")
print("-"*80)
print(f"{'Observation':<30} {'Real Min':>12} {'Real Max':>12} {'Sim Min':>12} {'Sim Max':>12} {'Difference':>12}")
print("-"*80)

for col in obs_columns_to_compare:
    if col in df_obs.columns and col in df_obs_sim.columns:
        real_min = df_obs[col].min()
        real_max = df_obs[col].max()
        sim_min = df_obs_sim[col].min()
        sim_max = df_obs_sim[col].max()
        
        # Calculate the difference in ranges
        range_diff = abs((real_max - real_min) - (sim_max - sim_min))
        
        print(f"{col:<30} {real_min:>12.4f} {real_max:>12.4f} {sim_min:>12.4f} {sim_max:>12.4f} {range_diff:>12.4f}")

# Action columns to compare
action_columns_to_compare = [
    'action_base_lf1', 'action_lf1_lf2', 'action_lf2_lf3',
    'action_base_rf1', 'action_rf1_rf2', 'action_rf2_rf3',
    'action_base_lb1', 'action_lb1_lb2', 'action_lb2_lb3',
    'action_base_rb1', 'action_rb1_rb2', 'action_rb2_rb3',
]

print("\n\nACTION VALUE RANGES:")
print("-"*80)
print(f"{'Action':<30} {'Real Min':>12} {'Real Max':>12} {'Sim Min':>12} {'Sim Max':>12} {'Difference':>12}")
print("-"*80)

for col in action_columns_to_compare:
    if col in df_act.columns and col in df_act_sim.columns:
        real_min = df_act[col].min()
        real_max = df_act[col].max()
        sim_min = df_act_sim[col].min()
        sim_max = df_act_sim[col].max()
        
        # Calculate the difference in ranges
        range_diff = abs((real_max - real_min) - (sim_max - sim_min))
        
        print(f"{col:<30} {real_min:>12.4f} {real_max:>12.4f} {sim_min:>12.4f} {sim_max:>12.4f} {range_diff:>12.4f}")

# Create visualization for range comparison
fig3 = plt.figure(figsize=(20, 12))

# 1. Observation range comparison - Base velocities
ax1 = plt.subplot(3, 3, 1)
obs_names = ['lin_vel_x', 'lin_vel_y', 'lin_vel_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z']
real_ranges = []
sim_ranges = []
for i, col in enumerate(['base_lin_vel_x', 'base_lin_vel_y', 'base_lin_vel_z', 
                         'base_ang_vel_x', 'base_ang_vel_y', 'base_ang_vel_z']):
    if col in df_obs.columns and col in df_obs_sim.columns:
        real_ranges.append(df_obs[col].max() - df_obs[col].min())
        sim_ranges.append(df_obs_sim[col].max() - df_obs_sim[col].min())

x = np.arange(len(obs_names))
width = 0.35
ax1.bar(x - width/2, real_ranges, width, label='Real', alpha=0.8)
ax1.bar(x + width/2, sim_ranges, width, label='Sim', alpha=0.8)
ax1.set_xlabel('Observation')
ax1.set_ylabel('Range')
ax1.set_title('Base Velocity Range Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(obs_names, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Joint position range comparison
ax2 = plt.subplot(3, 3, 2)
joint_names = ['LF_base', 'LF_mid', 'LF_tip', 'RF_base', 'RF_mid', 'RF_tip']
real_pos_ranges = []
sim_pos_ranges = []
for col in ['joint_pos_base_lf1', 'joint_pos_lf1_lf2', 'joint_pos_lf2_lf3',
            'joint_pos_base_rf1', 'joint_pos_rf1_rf2', 'joint_pos_rf2_rf3']:
    if col in df_obs.columns and col in df_obs_sim.columns:
        real_pos_ranges.append(df_obs[col].max() - df_obs[col].min())
        sim_pos_ranges.append(df_obs_sim[col].max() - df_obs_sim[col].min())

x = np.arange(len(joint_names))
ax2.bar(x - width/2, real_pos_ranges, width, label='Real', alpha=0.8)
ax2.bar(x + width/2, sim_pos_ranges, width, label='Sim', alpha=0.8)
ax2.set_xlabel('Joint')
ax2.set_ylabel('Position Range (rad)')
ax2.set_title('Joint Position Range Comparison (Front Legs)')
ax2.set_xticks(x)
ax2.set_xticklabels(joint_names, rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Action range comparison
ax3 = plt.subplot(3, 3, 3)
action_names = ['LF_base', 'LF_mid', 'LF_tip', 'RF_base', 'RF_mid', 'RF_tip']
real_act_ranges = []
sim_act_ranges = []
for col in ['action_base_lf1', 'action_lf1_lf2', 'action_lf2_lf3',
            'action_base_rf1', 'action_rf1_rf2', 'action_rf2_rf3']:
    if col in df_act.columns and col in df_act_sim.columns:
        real_act_ranges.append(df_act[col].max() - df_act[col].min())
        sim_act_ranges.append(df_act_sim[col].max() - df_act_sim[col].min())

x = np.arange(len(action_names))
ax3.bar(x - width/2, real_act_ranges, width, label='Real', alpha=0.8)
ax3.bar(x + width/2, sim_act_ranges, width, label='Sim', alpha=0.8)
ax3.set_xlabel('Joint')
ax3.set_ylabel('Action Range')
ax3.set_title('Action Range Comparison (Front Legs)')
ax3.set_xticks(x)
ax3.set_xticklabels(action_names, rotation=45)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Box plot comparison for all joint positions
ax4 = plt.subplot(3, 3, 4)
real_joint_data = []
sim_joint_data = []
labels = []
for col in ['joint_pos_base_lf1', 'joint_pos_base_rf1', 'joint_pos_base_lb1', 'joint_pos_base_rb1']:
    if col in df_obs.columns and col in df_obs_sim.columns:
        real_joint_data.append(df_obs[col].values)
        sim_joint_data.append(df_obs_sim[col].values)
        labels.append(col.split('_')[-2].upper() + col.split('_')[-1])

positions = np.arange(len(labels))
bp1 = ax4.boxplot(real_joint_data, positions=positions - 0.2, widths=0.3, patch_artist=True,
                  boxprops=dict(facecolor='blue', alpha=0.5), label='Real')
bp2 = ax4.boxplot(sim_joint_data, positions=positions + 0.2, widths=0.3, patch_artist=True,
                  boxprops=dict(facecolor='red', alpha=0.5), label='Sim')
ax4.set_xticks(positions)
ax4.set_xticklabels(labels)
ax4.set_ylabel('Position (rad)')
ax4.set_title('Base Joint Position Distribution')
ax4.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Real', 'Sim'])
ax4.grid(True, alpha=0.3)

# 5. Box plot comparison for all actions
ax5 = plt.subplot(3, 3, 5)
real_action_data = []
sim_action_data = []
labels = []
for col in ['action_base_lf1', 'action_base_rf1', 'action_base_lb1', 'action_base_rb1']:
    if col in df_act.columns and col in df_act_sim.columns:
        real_action_data.append(df_act[col].values)
        sim_action_data.append(df_act_sim[col].values)
        labels.append(col.split('_')[-2].upper() + col.split('_')[-1])

positions = np.arange(len(labels))
bp1 = ax5.boxplot(real_action_data, positions=positions - 0.2, widths=0.3, patch_artist=True,
                  boxprops=dict(facecolor='blue', alpha=0.5))
bp2 = ax5.boxplot(sim_action_data, positions=positions + 0.2, widths=0.3, patch_artist=True,
                  boxprops=dict(facecolor='red', alpha=0.5))
ax5.set_xticks(positions)
ax5.set_xticklabels(labels)
ax5.set_ylabel('Action Value')
ax5.set_title('Base Joint Action Distribution')
ax5.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Real', 'Sim'])
ax5.grid(True, alpha=0.3)

# 6. Projected gravity comparison
ax6 = plt.subplot(3, 3, 6)
grav_cols = ['projected_gravity_x', 'projected_gravity_y', 'projected_gravity_z']
real_grav_means = [df_obs[col].mean() for col in grav_cols]
sim_grav_means = [df_obs_sim[col].mean() for col in grav_cols]
real_grav_stds = [df_obs[col].std() for col in grav_cols]
sim_grav_stds = [df_obs_sim[col].std() for col in grav_cols]

x = np.arange(len(grav_cols))
ax6.bar(x - width/2, real_grav_means, width, yerr=real_grav_stds, label='Real', alpha=0.8, capsize=5)
ax6.bar(x + width/2, sim_grav_means, width, yerr=sim_grav_stds, label='Sim', alpha=0.8, capsize=5)
ax6.set_xlabel('Component')
ax6.set_ylabel('Value')
ax6.set_title('Projected Gravity Mean ± Std')
ax6.set_xticks(x)
ax6.set_xticklabels(['X', 'Y', 'Z'])
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Velocity command comparison
ax7 = plt.subplot(3, 3, 7)
cmd_cols = ['velocity_command_x', 'velocity_command_y', 'velocity_command_yaw']
real_cmd_ranges = []
sim_cmd_ranges = []
for col in cmd_cols:
    if col in df_obs.columns and col in df_obs_sim.columns:
        real_cmd_ranges.append([df_obs[col].min(), df_obs[col].max()])
        sim_cmd_ranges.append([df_obs_sim[col].min(), df_obs_sim[col].max()])

for i, (col, real_range, sim_range) in enumerate(zip(cmd_cols, real_cmd_ranges, sim_cmd_ranges)):
    ax7.plot([i-0.2, i-0.2], real_range, 'b-', linewidth=3, label='Real' if i == 0 else '')
    ax7.plot([i+0.2, i+0.2], sim_range, 'r-', linewidth=3, label='Sim' if i == 0 else '')
    ax7.plot(i-0.2, real_range[0], 'bo', markersize=8)
    ax7.plot(i-0.2, real_range[1], 'bo', markersize=8)
    ax7.plot(i+0.2, sim_range[0], 'ro', markersize=8)
    ax7.plot(i+0.2, sim_range[1], 'ro', markersize=8)

ax7.set_xticks(range(len(cmd_cols)))
ax7.set_xticklabels(['Cmd X', 'Cmd Y', 'Cmd Yaw'])
ax7.set_ylabel('Command Value')
ax7.set_title('Velocity Command Ranges')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Joint velocity range heatmap
ax8 = plt.subplot(3, 3, 8)
joint_vel_cols = [col for col in df_obs.columns if 'joint_vel_' in col and col in df_obs_sim.columns][:12]
range_diff_matrix = []
for col in joint_vel_cols:
    real_range = df_obs[col].max() - df_obs[col].min()
    sim_range = df_obs_sim[col].max() - df_obs_sim[col].min()
    range_diff_matrix.append([real_range, sim_range])

range_diff_matrix = np.array(range_diff_matrix).T
im = ax8.imshow(range_diff_matrix, aspect='auto', cmap='viridis')
ax8.set_yticks([0, 1])
ax8.set_yticklabels(['Real', 'Sim'])
ax8.set_xticks(range(len(joint_vel_cols)))
ax8.set_xticklabels([col.split('_')[-2] + '_' + col.split('_')[-1] for col in joint_vel_cols], rotation=90)
ax8.set_title('Joint Velocity Ranges Heatmap')
plt.colorbar(im, ax=ax8, label='Range (rad/s)')

# 9. Summary statistics comparison
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

# Calculate key metrics
real_obs_zero_count = sum((df_obs[col] == 0).sum() for col in obs_columns_to_compare if col in df_obs.columns)
sim_obs_zero_count = sum((df_obs_sim[col] == 0).sum() for col in obs_columns_to_compare if col in df_obs_sim.columns)

summary_text = f"""Range Comparison Summary:

Observations with constant zero (Real vs Sim):
Real: {real_obs_zero_count} zero values
Sim: {sim_obs_zero_count} zero values

Largest range differences:
"""

# Find top 5 largest differences
range_diffs = []
for col in obs_columns_to_compare:
    if col in df_obs.columns and col in df_obs_sim.columns:
        real_range = df_obs[col].max() - df_obs[col].min()
        sim_range = df_obs_sim[col].max() - df_obs_sim[col].min()
        diff = abs(real_range - sim_range)
        range_diffs.append((col, diff, real_range, sim_range))

range_diffs.sort(key=lambda x: x[1], reverse=True)
for col, diff, real_range, sim_range in range_diffs[:5]:
    summary_text += f"\n{col}: Δ={diff:.3f}"
    summary_text += f"\n  Real: {real_range:.3f}, Sim: {sim_range:.3f}"

ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=9, 
         verticalalignment='top', fontfamily='monospace')
ax9.set_title('Range Difference Summary')

plt.suptitle('Observation and Action Range Comparison', fontsize=14)
plt.tight_layout()
plt.show()

# Print comparison for all movements
print("\n\n" + "="*80)
print("RANGE COMPARISON ACROSS ALL MOVEMENTS")
print("="*80)

for movement_idx, movement_name in enumerate(movements):
    if f'obs_{movement_idx}' in real_data and f'act_{movement_idx}' in real_data:
        print(f"\n{movement_name.upper()} MOVEMENT (env_{movement_idx}):")
        print("-"*60)
        
        real_obs = real_data[f'obs_{movement_idx}']
        real_act = real_data[f'act_{movement_idx}']
        sim_obs = sim_data[f'obs_{movement_idx}']
        sim_act = sim_data[f'act_{movement_idx}']
        
        # Key observations
        key_obs = ['velocity_command_x', 'velocity_command_y', 'base_ang_vel_z']
        print("Key Observations:")
        for obs in key_obs:
            if obs in real_obs.columns and obs in sim_obs.columns:
                print(f"  {obs}:")
                print(f"    Real: [{real_obs[obs].min():.3f}, {real_obs[obs].max():.3f}]")
                print(f"    Sim:  [{sim_obs[obs].min():.3f}, {sim_obs[obs].max():.3f}]")
        
        # Key actions
        key_acts = ['action_base_lf1', 'action_base_rf1', 'action_base_lb1', 'action_base_rb1']
        print("\nKey Actions (base joints):")
        for act in key_acts:
            if act in real_act.columns and act in sim_act.columns:
                print(f"  {act}:")
                print(f"    Real: [{real_act[act].min():.3f}, {real_act[act].max():.3f}]")
                print(f"    Sim:  [{sim_act[act].min():.3f}, {sim_act[act].max():.3f}]")