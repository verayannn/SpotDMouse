#!/usr/bin/env python3
"""
Quick analysis script for IL model - simplified version
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

class MLPPolicy(nn.Module):
    """MLP policy matching your deployed controller architecture"""
    def __init__(self, obs_dim=48, action_dim=12, hidden_dims=[512, 256, 128]):
        super().__init__()
        
        layers = []
        in_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ELU(),
            ])
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, action_dim))
        
        self.actor = nn.Sequential(*layers)
        
    def forward(self, obs):
        return self.actor(obs)


def create_command_observation(cmd_linear_x, cmd_linear_y, cmd_angular_z):
    """Create a synthetic observation with given command velocity"""
    obs = np.zeros(48)
    
    # Command velocities (first 3 dims)
    obs[0] = cmd_linear_x
    obs[1] = cmd_linear_y
    obs[2] = cmd_angular_z
    
    # Joint positions (12 dims) - standing position with small variations
    obs[3:15] = np.array([0.0, 0.5, -1.0,   # FL
                          0.0, 0.5, -1.0,   # FR
                          0.0, 0.5, -1.0,   # RL
                          0.0, 0.5, -1.0])  # RR
    obs[3:15] += np.random.normal(0, 0.05, 12)
    
    # Joint velocities (12 dims) - small velocities
    obs[15:27] = np.random.normal(0, 0.1, 12)
    
    # Previous actions (12 dims) - similar to current positions
    obs[27:39] = obs[3:15] + np.random.normal(0, 0.02, 12)
    
    # Gravity vector (3 dims) - mostly pointing down
    obs[39:42] = np.array([0.1, 0.1, -9.8]) + np.random.normal(0, 0.05, 3)
    
    # Angular velocity (3 dims) - related to turning command
    obs[42:45] = np.array([0.0, 0.0, cmd_angular_z * 0.5]) + np.random.normal(0, 0.05, 3)
    
    # Gait phase (2 dims) - random phase
    phase = np.random.uniform(0, 2*np.pi)
    obs[45:47] = np.array([np.sin(phase), np.cos(phase)])
    
    # Foot contact (1 dim) - all feet on ground
    obs[47] = 1.0
    
    return obs


def test_command_responses(model, device='cuda'):
    """Test model responses to different commands"""
    
    # Define test commands (similar to teleop_record demos)
    test_commands = [
        # Name, linear_x, linear_y, angular_z
        ("Standing", 0.0, 0.0, 0.0),
        ("Forward Slow", 0.2, 0.0, 0.0),
        ("Forward Fast", 0.4, 0.0, 0.0),
        ("Backward", -0.2, 0.0, 0.0),
        ("Sideways Left", 0.0, 0.2, 0.0),
        ("Sideways Right", 0.0, -0.2, 0.0),
        ("Turn Left", 0.0, 0.0, 0.3),
        ("Turn Right", 0.0, 0.0, -0.3),
        ("Forward + Turn", 0.2, 0.0, 0.2),
        ("Diagonal", 0.2, 0.15, 0.0),
    ]
    
    results = []
    
    with torch.no_grad():
        for name, lin_x, lin_y, ang_z in test_commands:
            # Create observation
            obs = create_command_observation(lin_x, lin_y, ang_z)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Get model prediction
            action = model(obs_tensor).squeeze(0).cpu().numpy()
            
            results.append({
                'name': name,
                'command': np.array([lin_x, lin_y, ang_z]),
                'action': action
            })
    
    return results


def plot_command_responses(results, save_path='command_response_analysis.png'):
    """Plot model responses to different commands"""
    
    joint_names = [
        'FL_hip', 'FL_thigh', 'FL_calf',
        'FR_hip', 'FR_thigh', 'FR_calf', 
        'RL_hip', 'RL_thigh', 'RL_calf',
        'RR_hip', 'RR_thigh', 'RR_calf'
    ]
    
    num_commands = len(results)
    
    # Create subplots
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Model Response to Different Commands', fontsize=16)
    
    # Plot each command's response
    for cmd_idx, result in enumerate(results[:10]):  # Limit to 10 commands
        row = cmd_idx // 4
        col = cmd_idx % 4
        
        if row < 3:
            ax = axes[row, col] if cmd_idx < 12 else None
            if ax:
                # Plot joint positions
                x = np.arange(12)
                bars = ax.bar(x, result['action'], alpha=0.7)
                
                # Color code by leg
                colors = ['blue']*3 + ['green']*3 + ['red']*3 + ['orange']*3
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                # Add command info
                cmd = result['command']
                ax.set_title(f"{result['name']}\n[{cmd[0]:.1f}, {cmd[1]:.1f}, {cmd[2]:.1f}]", 
                           fontsize=10)
                ax.set_ylim(-2, 2)
                ax.grid(True, alpha=0.3)
                
                if row == 2:
                    ax.set_xticks(x)
                    ax.set_xticklabels(joint_names, rotation=45, ha='right', fontsize=8)
                else:
                    ax.set_xticks([])
                
                if col == 0:
                    ax.set_ylabel('Joint Position (rad)')
    
    # Remove empty subplots
    for i in range(cmd_idx + 1, 12):
        row = i // 4
        col = i % 4
        if row < 3:
            fig.delaxes(axes[row, col])
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, fc='blue', label='Front Left'),
        plt.Rectangle((0,0),1,1, fc='green', label='Front Right'),
        plt.Rectangle((0,0),1,1, fc='red', label='Rear Left'),
        plt.Rectangle((0,0),1,1, fc='orange', label='Rear Right')
    ]
    axes[0, 3].legend(handles=legend_elements, loc='center')
    axes[0, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Command response plot saved to {save_path}")


def plot_action_patterns(results, save_path='action_patterns.png'):
    """Plot action patterns for different motion types"""
    
    # Group commands by type
    forward_actions = []
    turn_actions = []
    sideways_actions = []
    
    for result in results:
        cmd = result['command']
        action = result['action']
        
        if abs(cmd[0]) > 0.1 and abs(cmd[1]) < 0.1 and abs(cmd[2]) < 0.1:
            forward_actions.append(action)
        elif abs(cmd[2]) > 0.1 and abs(cmd[0]) < 0.1 and abs(cmd[1]) < 0.1:
            turn_actions.append(action)
        elif abs(cmd[1]) > 0.1 and abs(cmd[0]) < 0.1 and abs(cmd[2]) < 0.1:
            sideways_actions.append(action)
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Average Action Patterns by Motion Type', fontsize=16)
    
    joint_names = [
        'FL_hip', 'FL_thigh', 'FL_calf',
        'FR_hip', 'FR_thigh', 'FR_calf', 
        'RL_hip', 'RL_thigh', 'RL_calf',
        'RR_hip', 'RR_thigh', 'RR_calf'
    ]
    
    motion_types = [
        ('Forward/Backward', forward_actions),
        ('Turning', turn_actions),
        ('Sideways', sideways_actions)
    ]
    
    for ax, (motion_name, actions) in zip(axes, motion_types):
        if actions:
            actions = np.array(actions)
            mean_action = np.mean(actions, axis=0)
            std_action = np.std(actions, axis=0)
            
            x = np.arange(12)
            bars = ax.bar(x, mean_action, yerr=std_action, capsize=5, alpha=0.7)
            
            # Color code by leg
            colors = ['blue']*3 + ['green']*3 + ['red']*3 + ['orange']*3
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_title(f'{motion_name} (n={len(actions)})')
            ax.set_xticks(x)
            ax.set_xticklabels(joint_names, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Joint Position (rad)')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-2, 2)
        else:
            ax.text(0.5, 0.5, 'No samples', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(motion_name)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Action patterns plot saved to {save_path}")


def test_random_commands(model, device='cuda', num_tests=20):
    """Test with random commands"""
    
    results = []
    
    with torch.no_grad():
        for i in range(num_tests):
            # Generate random command
            lin_x = np.random.uniform(-0.4, 0.4)
            lin_y = np.random.uniform(-0.3, 0.3)
            ang_z = np.random.uniform(-0.4, 0.4)
            
            # Sometimes zero out some components
            if np.random.rand() < 0.3:
                lin_y = 0.0
            if np.random.rand() < 0.3:
                ang_z = 0.0
            if np.random.rand() < 0.1:  # Standing still
                lin_x = lin_y = ang_z = 0.0
            
            # Create observation
            obs = create_command_observation(lin_x, lin_y, ang_z)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Get model prediction
            action = model(obs_tensor).squeeze(0).cpu().numpy()
            
            results.append({
                'command': np.array([lin_x, lin_y, ang_z]),
                'action': action
            })
    
    return results


def plot_command_action_correlation(results, save_path='command_action_correlation.png'):
    """Plot correlation between commands and joint actions"""
    
    commands = np.array([r['command'] for r in results])
    actions = np.array([r['action'] for r in results])
    
    # Calculate correlations
    correlations = np.zeros((3, 12))
    for i in range(3):  # For each command dimension
        for j in range(12):  # For each joint
            correlations[i, j] = np.corrcoef(commands[:, i], actions[:, j])[0, 1]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 4))
    
    im = ax.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    # Labels
    cmd_labels = ['Linear X', 'Linear Y', 'Angular Z']
    joint_names = [
        'FL_hip', 'FL_thigh', 'FL_calf',
        'FR_hip', 'FR_thigh', 'FR_calf', 
        'RL_hip', 'RL_thigh', 'RL_calf',
        'RR_hip', 'RR_thigh', 'RR_calf'
    ]
    
    ax.set_xticks(range(12))
    ax.set_xticklabels(joint_names, rotation=45, ha='right')
    ax.set_yticks(range(3))
    ax.set_yticklabels(cmd_labels)
    
    # Add correlation values
    for i in range(3):
        for j in range(12):
            text = ax.text(j, i, f'{correlations[i, j]:.2f}',
                         ha='center', va='center', 
                         color='white' if abs(correlations[i, j]) > 0.5 else 'black')
    
    ax.set_title('Command-Action Correlation Matrix')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Command-action correlation plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Quick IL model analysis")
    parser.add_argument("--checkpoint", 
                       default="/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--output-dir", default="quick_analysis", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = MLPPolicy().to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Determine state dict
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Print keys for debugging
    print(f"Checkpoint keys: {list(state_dict.keys())[:8]}")
    
    # Try loading with and without 'actor.' prefix
    try:
        model.load_state_dict(state_dict)
        print("Loaded model state dict directly.")
    except RuntimeError as e:
        print(f"Direct load failed: {e}\nTrying with 'actor.' prefix...")
        # Add 'actor.' prefix if missing
        new_state_dict = {f'actor.{k}': v for k, v in state_dict.items()}
        try:
            model.load_state_dict(new_state_dict)
            print("Loaded model state dict with 'actor.' prefix.")
        except RuntimeError as e2:
            print(f"Failed to load model state dict: {e2}")
            raise
    
    model.eval()
    print("Model loaded successfully!")
    
    # 1. Test specific commands
    print("\nTesting specific commands...")
    command_results = test_command_responses(model, args.device)
    plot_command_responses(command_results, os.path.join(args.output_dir, 'command_responses.png'))
    plot_action_patterns(command_results, os.path.join(args.output_dir, 'action_patterns.png'))
    
    # 2. Test random commands
    print("\nTesting random commands...")
    random_results = test_random_commands(model, args.device, num_tests=50)
    plot_command_action_correlation(random_results, os.path.join(args.output_dir, 'command_action_correlation.png'))
    
    # Print summary
    print(f"\nAnalysis complete! Results saved to {args.output_dir}/")
    print("\nGenerated plots:")
    print("1. command_responses.png - Model responses to specific commands")
    print("2. action_patterns.png - Average patterns for different motion types")
    print("3. command_action_correlation.png - How commands correlate with joint actions")
    
    # Quick insights
    print("\nQuick insights:")
    print("- Check command_responses.png to see if the model responds appropriately to different commands")
    print("- Look at action_patterns.png to verify distinct gaits for forward, turning, and sideways motion")
    print("- The correlation matrix shows which joints are most affected by each command dimension")


if __name__ == "__main__":
    main()