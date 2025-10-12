import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
import os

hdf5_path = "/Users/javierweddington/mini_pupper_demos_20250914_233847.hdf5"

# Define joint labels for Mini Pupper
JOINT_LABELS = [
    "base_lf1", "lf1_lf2", "lf2_lf3",  # Left Front leg
    "base_rf1", "rf1_rf2", "rf2_rf3",  # Right Front leg
    "base_lb1", "lb1_lb2", "lb2_lb3",  # Left Back leg
    "base_rb1", "rb1_rb2", "rb2_rb3"   # Right Back leg
]

# Check if file exists
if not os.path.exists(hdf5_path):
    print(f"ERROR: HDF5 file not found at {hdf5_path}")
    exit(1)
else:
    print(f"Found HDF5 file at {hdf5_path}")

# Create a directory for saving plots
plot_dir = "/Users/javierweddington/h5plots"
os.makedirs(plot_dir, exist_ok=True)
print(f"Plot directory: {plot_dir}")

print("=== Analyzing Demo 1 ===")
with h5py.File(hdf5_path, 'r') as f:
    # Focus on demo_1
    demo_key = 'demo_1'
    
    if f'data/{demo_key}' not in f:
        print(f"ERROR: {demo_key} not found in HDF5 file")
        exit(1)
    
    demo_group = f[f'data/{demo_key}']
    
    print(f"\n=== Demo 1 Structure ===")
    print(f"Available data: {list(demo_group.keys())}")
    
    # Get all data
    actions = demo_group['actions'][:] if 'actions' in demo_group else None
    obs = demo_group['obs'][:] if 'obs' in demo_group else None
    rewards = demo_group['rewards'][:] if 'rewards' in demo_group else None
    dones = demo_group['dones'][:] if 'dones' in demo_group else None
    timestamps = demo_group['timestamps'][:] if 'timestamps' in demo_group else None
    
    # Print detailed information
    if actions is not None:
        print(f"\nActions shape: {actions.shape}")
        print(f"Actions range: [{np.min(actions):.4f}, {np.max(actions):.4f}]")
        print(f"Actions mean: {np.mean(actions):.4f}, std: {np.std(actions):.4f}")
    
    if obs is not None:
        print(f"\nObservations shape: {obs.shape}")
        print(f"Observations range: [{np.min(obs):.4f}, {np.max(obs):.4f}]")
        print(f"Observations mean: {np.mean(obs):.4f}, std: {np.std(obs):.4f}")
        print(f"Number of observation dimensions: {obs.shape[1] if obs.ndim > 1 else 1}")
    
    if rewards is not None:
        print(f"\nRewards shape: {rewards.shape}")
        print(f"Rewards range: [{np.min(rewards):.4f}, {np.max(rewards):.4f}]")
        print(f"Total reward: {np.sum(rewards):.4f}")
    
    if timestamps is not None:
        print(f"\nTimestamps shape: {timestamps.shape}")
        print(f"Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
        print(f"Average timestep: {np.mean(np.diff(timestamps)):.4f} seconds")
    
    # Create comprehensive visualization for demo_1
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Actions by leg
    ax1 = plt.subplot(4, 2, 1)
    if actions is not None and actions.size > 0:
        leg_names = ['Left Front', 'Right Front', 'Left Back', 'Right Back']
        leg_colors = ['red', 'blue', 'green', 'purple']
        leg_joint_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        
        for leg_idx, (leg_name, color, joint_indices) in enumerate(zip(leg_names, leg_colors, leg_joint_indices)):
            for i, joint_idx in enumerate(joint_indices):
                if joint_idx < actions.shape[1]:
                    linestyle = ['-', '--', ':'][i]  # Different line styles for each joint
                    ax1.plot(actions[:, joint_idx], 
                            color=color, 
                            linestyle=linestyle,
                            linewidth=1.5,
                            label=JOINT_LABELS[joint_idx])
        
        ax1.set_title('All Joint Actions')
        ax1.set_ylabel('Joint Angle')
        ax1.set_xlabel('Time step')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 2: Actions heatmap
    ax2 = plt.subplot(4, 2, 2)
    if actions is not None and actions.ndim > 1:
        im = ax2.imshow(actions.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
        ax2.set_yticks(range(len(JOINT_LABELS)))
        ax2.set_yticklabels(JOINT_LABELS, fontsize=8)
        ax2.set_xlabel('Time step')
        ax2.set_title('Actions Heatmap')
        plt.colorbar(im, ax=ax2)
    
    # Plot 3-6: Individual leg actions
    for leg_idx in range(4):
        ax = plt.subplot(4, 2, 3 + leg_idx)
        leg_names = ['Left Front', 'Right Front', 'Left Back', 'Right Back']
        leg_joint_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        
        if actions is not None and actions.ndim > 1:
            for joint_idx in leg_joint_indices[leg_idx]:
                if joint_idx < actions.shape[1]:
                    ax.plot(actions[:, joint_idx], linewidth=1.5, label=JOINT_LABELS[joint_idx])
            
            ax.set_title(f'{leg_names[leg_idx]} Leg Actions')
            ax.set_ylabel('Joint Angle')
            ax.set_xlabel('Time step')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    # Plot 7: Observations overview
    ax7 = plt.subplot(4, 2, 7)
    if obs is not None and obs.size > 0:
        n_obs = obs.shape[1] if obs.ndim > 1 else 1
        
        # Plot first 20 observations with different colors
        for i in range(min(20, n_obs)):
            ax7.plot(obs[:, i] if obs.ndim > 1 else obs, 
                    linewidth=1.0, 
                    label=f'Obs {i}', 
                    alpha=0.7)
        
        ax7.set_title(f'First 20 Observations (of {n_obs} total)')
        ax7.set_ylabel('Value')
        ax7.set_xlabel('Time step')
        ax7.grid(True, alpha=0.3)
        if n_obs <= 10:
            ax7.legend(fontsize=8)
    
    # Plot 8: Rewards and dones
    ax8 = plt.subplot(4, 2, 8)
    if rewards is not None:
        ax8_twin = ax8.twinx()
        ax8.plot(rewards, 'g-', linewidth=1.5, label='Rewards')
        ax8.set_ylabel('Rewards', color='g')
        ax8.tick_params(axis='y', labelcolor='g')
        
        if dones is not None:
            ax8_twin.plot(dones, 'r--', linewidth=1.5, label='Dones', alpha=0.7)
            ax8_twin.set_ylabel('Done Flag', color='r')
            ax8_twin.tick_params(axis='y', labelcolor='r')
        
        ax8.set_xlabel('Time step')
        ax8.set_title('Rewards and Episode Termination')
        ax8.grid(True, alpha=0.3)
    
    plt.suptitle('Demo 1 - Comprehensive Analysis', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    plot_path = f"{plot_dir}/demo_1_comprehensive_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comprehensive plot to: {plot_path}")
    
    # Create a separate figure for observation analysis
    if obs is not None and obs.ndim > 1:
        fig2, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Observation heatmap
        im = axes[0].imshow(obs.T, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[0].set_ylabel('Observation Dimension')
        axes[0].set_xlabel('Time step')
        axes[0].set_title(f'All {obs.shape[1]} Observations Heatmap')
        plt.colorbar(im, ax=axes[0])
        
        # Observation statistics
        obs_means = np.mean(obs, axis=0)
        obs_stds = np.std(obs, axis=0)
        x = range(len(obs_means))
        
        axes[1].bar(x, obs_means, yerr=obs_stds, capsize=3, alpha=0.7)
        axes[1].set_xlabel('Observation Dimension')
        axes[1].set_ylabel('Mean ± Std')
        axes[1].set_title('Observation Statistics (Mean and Std Dev)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        obs_plot_path = f"{plot_dir}/demo_1_observations_analysis.png"
        plt.savefig(obs_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved observation analysis to: {obs_plot_path}")

print("\nAnalysis complete!")
print(f"Generated plots in: {plot_dir}")
