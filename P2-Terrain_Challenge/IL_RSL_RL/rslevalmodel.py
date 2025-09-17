#!/usr/bin/env python3
"""
Simple forward walk comparison for RSL RL format model vs demonstration data
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

class ActorCriticMLP(nn.Module):
    """Actor-Critic MLP matching RSL RL's expected structure exactly"""
    def __init__(self, obs_dim=48, action_dim=12, hidden_dims=[512, 256, 128]):
        super().__init__()
        
        # Actor network
        actor_layers = []
        in_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ELU(),
            ])
            in_dim = hidden_dim
            
        actor_layers.append(nn.Linear(in_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic network
        critic_layers = []
        in_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ELU(),
            ])
            in_dim = hidden_dim
            
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # Action standard deviation - will be loaded from checkpoint
        self.std = nn.Parameter(torch.ones(action_dim) * np.log(0.1))
                
    def get_action(self, obs, deterministic=True):
        """Get action with optional stochasticity"""
        mean = self.actor(obs)
        return mean

def create_simple_forward_comparison():
    """Create a simple comparison plot like the original"""
    
    # Load RSL RL format model
    checkpoint_path = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models_rsl_format/best_model_rsl_format.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    model = ActorCriticMLP()
    
    # Load state dict
    state_dict = checkpoint['model_state_dict']
    actor_state = {k.replace('actor.', ''): v for k, v in state_dict.items() if k.startswith('actor.')}
    critic_state = {k.replace('critic.', ''): v for k, v in state_dict.items() if k.startswith('critic.')}
    
    model.actor.load_state_dict(actor_state)
    model.critic.load_state_dict(critic_state)
    
    if 'std' in state_dict:
        model.std.data = state_dict['std']
    
    model.eval()
    
    # Load demonstration data - using demo_3 (forward walk)
    dataset_path = '/workspace/rosbag_recordings/hdf5_datasets/mini_pupper_demos_20250914_233847.hdf5'
    with h5py.File(dataset_path, 'r') as f:
        # Use first 100 timesteps for clarity
        demo_obs = f['data/demo_2/obs'][200:300]
        demo_actions = f['data/demo_2/actions'][200:300]
        demo_type = f['data/demo_2'].attrs.get('demo_type', 'unknown')
    
    # Get normalization parameters
    obs_mean = checkpoint.get('obs_rms_mean', torch.zeros(48))
    obs_var = checkpoint.get('obs_rms_var', torch.ones(48))
    
    # Run inference
    with torch.no_grad():
        obs_tensor = torch.tensor(demo_obs, dtype=torch.float32)
        obs_norm = (obs_tensor - obs_mean) / torch.sqrt(obs_var + 1e-8)
        pred_actions = model.get_action(obs_norm, deterministic=True).numpy()
    
    # Joint groupings for FR leg (as example)
    joint_groups = {
        'Hip': {
            'FL': 0, 'FR': 3, 'RL': 6, 'RR': 9
        },
        'Thigh': {
            'FL': 1, 'FR': 4, 'RL': 7, 'RR': 10
        },
        'Knee': {
            'FL': 2, 'FR': 5, 'RL': 8, 'RR': 11
        }
    }
    
    # Create figure with similar layout to original
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle('Simple Forward Walk Comparison: Demo 3 vs RSL RL Model', fontsize=14)
    
    # Plot 1: Hip joints comparison
    ax1 = plt.subplot(3, 1, 1)
    for leg, idx in joint_groups['Hip'].items():
        # Plot demonstration
        ax1.plot(demo_actions[:, idx], label=f'{leg} Demo', linestyle='-', linewidth=2, alpha=0.2)
        # Plot model prediction
        ax1.plot(pred_actions[:, idx], label=f'{leg} Model', linestyle='--', linewidth=2)
    ax1.set_ylabel('Hip Position (rad)')
    ax1.set_title('Hip Joint Comparison (solid=demo, dashed=model)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', ncol=4, fontsize=8)
    ax1.set_xlim(0, 100)
    
    # Plot 2: Front Right Leg - All Joints
    ax2 = plt.subplot(3, 1, 2)
    leg_name = 'FR'
    joint_names = ['Hip', 'Thigh', 'Knee']
    colors = ['blue', 'green', 'red']
    
    for joint, color in zip(joint_names, colors):
        idx = joint_groups[joint][leg_name]
        # Demonstration
        ax2.plot(demo_actions[:, idx], color=color, label=f'{joint} Demo', 
                linestyle='-', linewidth=2, alpha=0.2)
        # Model prediction
        ax2.plot(pred_actions[:, idx], color=color, label=f'{joint} Model', 
                linestyle='--', linewidth=2, alpha=1.0)
    
    ax2.set_ylabel('Joint Position (rad)')
    ax2.set_xlabel('Time Step')
    ax2.set_title('Front Right Leg - All Joints')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', ncol=3, fontsize=8)
    ax2.set_xlim(0, 100)
    
    # Plot 3: Error per joint
    ax3 = plt.subplot(3, 1, 3)
    
    # Calculate errors
    errors = np.abs(pred_actions - demo_actions).mean(axis=0)
    joint_names_full = ['FR-Hip', 'FR-Thigh', 'FR-Knee', 
                        'FL-Hip', 'FL-Thigh', 'FL-Knee',
                        'RR-Hip', 'RR-Thigh', 'RR-Knee',
                        'RL-Hip', 'RL-Thigh', 'RL-Knee']
    
    # Color code by joint type
    colors = []
    for name in joint_names_full:
        if 'Hip' in name:
            colors.append('blue')
        elif 'Thigh' in name:
            colors.append('green')
        else:  # Knee
            colors.append('red')
    
    bars = ax3.bar(range(12), errors, color=colors)
    ax3.set_xticks(range(12))
    ax3.set_xticklabels([name.split('-')[0] + '\n' + name.split('-')[1] for name in joint_names_full], 
                        rotation=0, fontsize=8)
    ax3.set_ylabel('Mean Absolute Error (rad)')
    ax3.set_title(f'Error Per Joint (Total MAE: {errors.mean():.3f} rad)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{err:.3f}', ha='center', va='bottom', fontsize=7)
    
    # Add legend for joint types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Hip'),
                      Patch(facecolor='green', label='Thigh'),
                      Patch(facecolor='red', label='Knee')]
    ax3.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save the plot
    save_path = '/workspace/simple_forward_comparison_rsl.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved comparison plot to {save_path}")
    
    # Print summary statistics
    print(f"\nModel Performance Summary:")
    print(f"Overall MAE: {errors.mean():.4f} rad")
    print(f"\nPer-joint errors:")
    for name, err in zip(joint_names_full, errors):
        print(f"  {name:12s}: {err:.4f} rad")
    
    # Check training info
    if 'infos' in checkpoint:
        infos = checkpoint['infos']
        if 'epoch' in infos:
            print(f"\nModel trained for {infos['epoch']} epochs")
        if 'val_actor_loss' in infos:
            print(f"Final validation actor loss: {infos['val_actor_loss']:.6f}")
    
    return save_path

def create_knee_focused_comparison():
    """Create a comparison focused on knee joints"""
    
    # Load model and data (same as above)
    checkpoint_path = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models_rsl_format/best_model_rsl_format.pt"
    
    if not os.path.exists(checkpoint_path):
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = ActorCriticMLP()
    
    state_dict = checkpoint['model_state_dict']
    actor_state = {k.replace('actor.', ''): v for k, v in state_dict.items() if k.startswith('actor.')}
    model.actor.load_state_dict(actor_state)
    model.eval()
    
    # Load data
    dataset_path = '/workspace/rosbag_recordings/hdf5_datasets/mini_pupper_demos_20250914_233847.hdf5'
    with h5py.File(dataset_path, 'r') as f:
        demo_obs = f['data/demo_2/obs'][300:500]
        demo_actions = f['data/demo_2/actions'][300:500]
    
    # Get predictions
    obs_mean = checkpoint.get('obs_rms_mean', torch.zeros(48))
    obs_var = checkpoint.get('obs_rms_var', torch.ones(48))
    
    with torch.no_grad():
        obs_tensor = torch.tensor(demo_obs, dtype=torch.float32)
        obs_norm = (obs_tensor - obs_mean) / torch.sqrt(obs_var + 1e-8)
        pred_actions = model.get_action(obs_norm, deterministic=True).numpy()
    
    # Create knee-focused plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Knee Joint Comparison - Demo vs Model', fontsize=14)
    
    knee_indices = [2, 5, 8, 11]
    knee_names = ['FR-Knee', 'FL-Knee', 'RR-Knee', 'RL-Knee']
    
    for ax, idx, name in zip(axes.flat, knee_indices, knee_names):
        # Plot demonstration
        ax.plot(demo_actions[:, idx], 'k-', label='Demo', linewidth=2.5)
        # Plot prediction
        ax.plot(pred_actions[:, idx], 'r--', label='Model', linewidth=2)
        
        # Calculate error
        error = np.abs(pred_actions[:, idx] - demo_actions[:, idx]).mean()
        
        ax.set_title(f'{name} (MAE: {error:.4f} rad)')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Position (rad)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    save_path = '/workspace/knee_comparison_rsl.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved knee comparison to {save_path}")
    
    return save_path

if __name__ == "__main__":
    # Create both comparisons
    plot1 = create_simple_forward_comparison()
    plot2 = create_knee_focused_comparison()
    
    print("\nTo view the plots:")
    if plot1:
        print(f'  "$BROWSER" {plot1}')
    if plot2:
        print(f'  "$BROWSER" {plot2}')