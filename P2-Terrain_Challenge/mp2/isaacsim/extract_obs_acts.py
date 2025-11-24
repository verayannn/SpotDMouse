import pandas as pd
import IPython
import matlotlib.pyplot as plt

df_obs = pd.read_csv("/Users/javierweddington/isaaclabstock/isaaclabstock/obs_action_logs/env_0_observations.csv")
df_act = pd.read_csv("/Users/javierweddington/isaaclabstock/isaaclabstock/obs_action_logs/env_0_actions.csv")

IPython.embed()

# time_step,action_base_lf1,action_lf1_lf2,action_lf2_lf3,action_base_rf1,action_rf1_rf2,action_rf2_rf3,action_base_lb1,action_lb1_lb2,action_lb2_lb3,action_base_rb1,action_rb1_rb2,action_rb2_rb3
# time_step,base_lin_vel_x,base_lin_vel_y,base_lin_vel_z,base_ang_vel_x,base_ang_vel_y,base_ang_vel_z,projected_gravity_x,projected_gravity_y,projected_gravity_z,velocity_command_x,velocity_command_y,velocity_command_yaw,joint_pos_base_lf1,joint_pos_lf1_lf2,joint_pos_lf2_lf3,joint_pos_base_rf1,joint_pos_rf1_rf2,joint_pos_rf2_rf3,joint_pos_base_lb1,joint_pos_lb1_lb2,joint_pos_lb2_lb3,joint_pos_base_rb1,joint_pos_rb1_rb2,joint_pos_rb2_rb3,joint_vel_base_lf1,joint_vel_lf1_lf2,joint_vel_lf2_lf3,joint_vel_base_rf1,joint_vel_rf1_rf2,joint_vel_rf2_rf3,joint_vel_base_lb1,joint_vel_lb1_lb2,joint_vel_lb2_lb3,joint_vel_base_rb1,joint_vel_rb1_rb2,joint_vel_rb2_rb3,joint_effort_base_lf1,joint_effort_lf1_lf2,joint_effort_lf2_lf3,joint_effort_base_rf1,joint_effort_rf1_rf2,joint_effort_rf2_rf3,joint_effort_base_lb1,joint_effort_lb1_lb2,joint_effort_lb2_lb3,joint_effort_base_rb1,joint_effort_rb1_rb2,joint_effort_rb2_rb3,prev_action_base_lf1,prev_action_lf1_lf2,prev_action_lf2_lf3,prev_action_base_rf1,prev_action_rf1_rf2,prev_action_rf2_rf3,prev_action_base_lb1,prev_action_lb1_lb2,prev_action_lb2_lb3,prev_action_base_rb1,prev_action_rb1_rb2,prev_action_rb2_rb3
# Create figure with subplots for comprehensive analysis
fig = plt.figure(figsize=(20, 16))

# 1. Base velocities
ax1 = plt.subplot(4, 3, 1)
ax1.plot(df_obs['time_step'], df_obs['base_lin_vel_x'], label='Linear X')
ax1.plot(df_obs['time_step'], df_obs['base_lin_vel_y'], label='Linear Y')
ax1.plot(df_obs['time_step'], df_obs['base_lin_vel_z'], label='Linear Z')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Velocity (m/s)')
ax1.set_title('Base Linear Velocities')
ax1.legend()
ax1.grid(True)

# 2. Base angular velocities
ax2 = plt.subplot(4, 3, 2)
ax2.plot(df_obs['time_step'], df_obs['base_ang_vel_x'], label='Angular X')
ax2.plot(df_obs['time_step'], df_obs['base_ang_vel_y'], label='Angular Y')
ax2.plot(df_obs['time_step'], df_obs['base_ang_vel_z'], label='Angular Z')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Angular Velocity (rad/s)')
ax2.set_title('Base Angular Velocities')
ax2.legend()
ax2.grid(True)

# 3. Velocity commands vs actual
ax3 = plt.subplot(4, 3, 3)
ax3.plot(df_obs['time_step'], df_obs['velocity_command_x'], label='Command X', linestyle='--')
ax3.plot(df_obs['time_step'], df_obs['base_lin_vel_x'], label='Actual X')
ax3.plot(df_obs['time_step'], df_obs['velocity_command_y'], label='Command Y', linestyle='--')
ax3.plot(df_obs['time_step'], df_obs['base_lin_vel_y'], label='Actual Y')
ax3.set_xlabel('Time Step')
ax3.set_ylabel('Velocity (m/s)')
ax3.set_title('Velocity Command vs Actual')
ax3.legend()
ax3.grid(True)

# 4. Joint positions for all legs
ax4 = plt.subplot(4, 3, 4)
for leg in ['lf', 'rf', 'lb', 'rb']:
    ax4.plot(df_obs['time_step'], df_obs[f'joint_pos_base_{leg}1'], label=f'{leg.upper()} Base')
ax4.set_xlabel('Time Step')
ax4.set_ylabel('Position (rad)')
ax4.set_title('Base Joint Positions (All Legs)')
ax4.legend()
ax4.grid(True)

# 5. Actions for all legs (base joints)
ax5 = plt.subplot(4, 3, 5)
for leg in ['lf', 'rf', 'lb', 'rb']:
    ax5.plot(df_act['time_step'], df_act[f'action_base_{leg}1'], label=f'{leg.upper()} Base')
ax5.set_xlabel('Time Step')
ax5.set_ylabel('Action')
ax5.set_title('Base Joint Actions (All Legs)')
ax5.legend()
ax5.grid(True)

# 6. Joint efforts analysis
ax6 = plt.subplot(4, 3, 6)
avg_efforts = []
for leg in ['lf', 'rf', 'lb', 'rb']:
    leg_effort = df_obs[[f'joint_effort_base_{leg}1', f'joint_effort_{leg}1_{leg}2', 
                        f'joint_effort_{leg}2_{leg}3']].mean(axis=1)
    ax6.plot(df_obs['time_step'], leg_effort, label=f'{leg.upper()}')
    avg_efforts.append(leg_effort)
ax6.set_xlabel('Time Step')
ax6.set_ylabel('Average Effort')
ax6.set_title('Average Joint Efforts per Leg')
ax6.legend()
ax6.grid(True)

# 7. Projected gravity
ax7 = plt.subplot(4, 3, 7)
ax7.plot(df_obs['time_step'], df_obs['projected_gravity_x'], label='X')
ax7.plot(df_obs['time_step'], df_obs['projected_gravity_y'], label='Y')
ax7.plot(df_obs['time_step'], df_obs['projected_gravity_z'], label='Z')
ax7.set_xlabel('Time Step')
ax7.set_ylabel('Projected Gravity')
ax7.set_title('Projected Gravity Components')
ax7.legend()
ax7.grid(True)

# 8. Action vs Previous Action (example for LF leg)
ax8 = plt.subplot(4, 3, 8)
ax8.plot(df_act['time_step'], df_act['action_base_lf1'], label='Current Action', alpha=0.7)
ax8.plot(df_obs['time_step'], df_obs['prev_action_base_lf1'], label='Previous Action', alpha=0.7)
ax8.set_xlabel('Time Step')
ax8.set_ylabel('Action')
ax8.set_title('LF Base Joint: Action vs Previous Action')
ax8.legend()
ax8.grid(True)

# 9. Joint velocities distribution
ax9 = plt.subplot(4, 3, 9)
vel_cols = [col for col in df_obs.columns if 'joint_vel' in col]
vel_data = df_obs[vel_cols].values.flatten()
ax9.hist(vel_data, bins=50, alpha=0.7, edgecolor='black')
ax9.set_xlabel('Joint Velocity (rad/s)')
ax9.set_ylabel('Frequency')
ax9.set_title('Joint Velocities Distribution')
ax9.grid(True)

# 10. Action distribution
ax10 = plt.subplot(4, 3, 10)
action_cols = [col for col in df_act.columns if col != 'time_step']
action_data = df_act[action_cols].values.flatten()
ax10.hist(action_data, bins=50, alpha=0.7, edgecolor='black')
ax10.set_xlabel('Action Value')
ax10.set_ylabel('Frequency')
ax10.set_title('Actions Distribution')
ax10.grid(True)

# 11. Correlation heatmap for key variables
ax11 = plt.subplot(4, 3, 11)
corr_vars = ['base_lin_vel_x', 'base_lin_vel_y', 'base_ang_vel_z', 
             'velocity_command_x', 'velocity_command_y']
corr_matrix = df_obs[corr_vars].corr()
im = ax11.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax11.set_xticks(range(len(corr_vars)))
ax11.set_yticks(range(len(corr_vars)))
ax11.set_xticklabels([v.split('_')[-1] for v in corr_vars], rotation=45)
ax11.set_yticklabels([v.split('_')[-1] for v in corr_vars])
ax11.set_title('Velocity Correlations')
plt.colorbar(im, ax=ax11)

# 12. Summary statistics
ax12 = plt.subplot(4, 3, 12)
ax12.axis('off')
stats_text = f"""Summary Statistics:
Total timesteps: {len(df_obs)}
Avg linear vel X: {df_obs['base_lin_vel_x'].mean():.3f} ± {df_obs['base_lin_vel_x'].std():.3f}
Avg linear vel Y: {df_obs['base_lin_vel_y'].mean():.3f} ± {df_obs['base_lin_vel_y'].std():.3f}
Avg angular vel Z: {df_obs['base_ang_vel_z'].mean():.3f} ± {df_obs['base_ang_vel_z'].std():.3f}
Max joint effort: {df_obs[vel_cols].max().max():.3f}
Action range: [{action_data.min():.3f}, {action_data.max():.3f}]
"""
ax12.text(0.1, 0.5, stats_text, transform=ax12.transAxes, fontsize=10, 
          verticalalignment='center', fontfamily='monospace')
ax12.set_title('Summary Statistics')

plt.tight_layout()
plt.show()

