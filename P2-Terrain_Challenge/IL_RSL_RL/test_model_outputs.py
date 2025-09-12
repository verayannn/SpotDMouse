#!/usr/bin/env python3
# filepath: /workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/test_model_outputs.py
"""
Test and compare outputs from different models with simple commands
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from train_il import MLPPolicy

def create_test_observation(vx=0.3, vy=0.0, wz=0.0):
    """Create a 48-dim observation vector with simple command"""
    obs = torch.zeros(1, 48)
    
    # Set velocities (first 6 elements)
    obs[0, 0] = vx  # Linear velocity X (forward)
    obs[0, 1] = vy  # Linear velocity Y (sideways)
    obs[0, 2] = 0.0  # Linear velocity Z (up)
    obs[0, 3] = 0.0  # Angular velocity X (roll)
    obs[0, 4] = 0.0  # Angular velocity Y (pitch)
    obs[0, 5] = wz  # Angular velocity Z (yaw)
    
    # Set some reasonable default values for other components
    # Joint positions (elements 6-17) - standing pose
    default_joint_pos = [0.0, -0.8, 1.6] * 4  # Hip, thigh, calf for each leg
    for i, pos in enumerate(default_joint_pos):
        obs[0, 6 + i] = pos
    
    # Joint velocities (elements 18-29) - all zero
    # Quaternion (elements 30-33) - upright orientation
    obs[0, 30] = 0.0  # qx
    obs[0, 31] = 0.0  # qy
    obs[0, 32] = 0.0  # qz
    obs[0, 33] = 1.0  # qw (upright)
    
    # Previous actions (elements 34-45) - copy joint positions
    for i in range(12):
        obs[0, 34 + i] = obs[0, 6 + i]
    
    # Clock phase (elements 46-47)
    obs[0, 46] = 0.0  # sin(phase)
    obs[0, 47] = 1.0  # cos(phase)
    
    return obs

def test_model(model_path, obs, model_type='unknown'):
    """Test a single model with given observation"""
    print(f"\n=== Testing {model_type} model: {model_path.split('/')[-1]} ===")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract observation normalization
    if 'obs_mean' in checkpoint and checkpoint['obs_mean'] is not None:
        obs_mean = checkpoint['obs_mean']
        obs_std = checkpoint['obs_std']
        # Convert to torch tensors if needed
        if isinstance(obs_mean, np.ndarray):
            obs_mean = torch.from_numpy(obs_mean).float()
            obs_std = torch.from_numpy(obs_std).float()
        print(f"Using saved normalization stats")
    elif 'obs_rms_mean' in checkpoint:
        obs_mean = checkpoint['obs_rms_mean']
        obs_var = checkpoint['obs_rms_var']
        # Convert to torch tensors if needed
        if isinstance(obs_mean, np.ndarray):
            obs_mean = torch.from_numpy(obs_mean).float()
            obs_var = torch.from_numpy(obs_var).float()
        obs_std = torch.sqrt(obs_var + 1e-8)
        print(f"Using RMS normalization stats")
    else:
        obs_mean = torch.zeros(48)
        obs_std = torch.ones(48)
        print(f"WARNING: No normalization stats found, using defaults")
    
    # Normalize observation
    obs_norm = (obs - obs_mean) / (obs_std + 1e-8)
    
    # Load model based on type
    if model_type == 'IL':
        model = MLPPolicy(obs_dim=48, action_dim=12)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Try to load directly if it's just state dict
            model.load_state_dict(checkpoint)
    elif model_type == 'RSL_RL' or model_type == 'RSL_RL_Original':
        # For RSL_RL, we need to extract just the actor
        from collections import OrderedDict
        model = MLPPolicy(obs_dim=48, action_dim=12)
        
        # Map RSL_RL actor weights to our model
        actor_state_dict = OrderedDict()
        model_state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        for key, value in model_state_dict.items():
            if key.startswith('actor.'):
                new_key = key.replace('actor.', 'net.')
                actor_state_dict[new_key] = value
        
        if actor_state_dict:
            model.load_state_dict(actor_state_dict)
            print(f"Loaded actor weights from RSL_RL model")
        else:
            print(f"ERROR: Could not find actor weights in RSL_RL model")
            return None
    
    model.eval()
    
    # Get raw output
    with torch.no_grad():
        output = model(obs_norm)
    
    # Denormalize if action stats are available
    if 'action_mean' in checkpoint and checkpoint['action_mean'] is not None:
        action_mean = checkpoint['action_mean']
        action_std = checkpoint['action_std']
        # Convert to torch tensors if needed
        if isinstance(action_mean, np.ndarray):
            action_mean = torch.from_numpy(action_mean).float()
            action_std = torch.from_numpy(action_std).float()
        output_denorm = output * action_std + action_mean
        print(f"Applied action denormalization")
    else:
        output_denorm = output
        print(f"No action normalization applied")
    
    # Print results
    print(f"\nInput command: vx={obs[0, 0]:.2f}, vy={obs[0, 1]:.2f}, wz={obs[0, 5]:.2f}")
    print(f"Raw observation (first 6): {obs[0, :6].numpy()}")
    print(f"Normalized obs (first 6): {obs_norm[0, :6].numpy()}")
    
    print(f"\nOutput joint positions (radians):")
    joint_names = [
        "FR_hip", "FR_thigh", "FR_calf",
        "FL_hip", "FL_thigh", "FL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
        "RL_hip", "RL_thigh", "RL_calf"
    ]
    
    for i, name in enumerate(joint_names):
        print(f"  {name:10s}: {output_denorm[0, i]:7.3f} rad ({np.degrees(output_denorm[0, i]):6.1f} deg)")
    
    # Check if outputs seem reasonable
    output_np = output_denorm[0].numpy()
    print(f"\nOutput statistics:")
    print(f"  Range: [{output_np.min():.3f}, {output_np.max():.3f}] rad")
    print(f"  Mean: {output_np.mean():.3f} rad")
    print(f"  Std: {output_np.std():.3f} rad")
    
    # Check for common issues
    if np.all(np.abs(output_np) < 0.01):
        print("⚠️  WARNING: All outputs near zero - model might be outputting zeros")
    elif np.all(np.abs(output_np - output_np[0]) < 0.01):
        print("⚠️  WARNING: All outputs similar - model might be outputting constant")
    elif np.any(np.abs(output_np) > 3.0):
        print("⚠️  WARNING: Some outputs > 3 rad - might be out of joint limits")
    else:
        print("✓ Outputs appear reasonable")
    
    return output_denorm[0].numpy()

def compare_models():
    """Compare outputs from all models"""
    
    # Test different commands
    test_commands = [
        (0.3, 0.0, 0.0, "Forward walk"),
        (0.0, 0.0, 0.5, "Turn in place"),
        (0.0, 0.0, 0.0, "Stand still"),
        (-0.2, 0.0, 0.0, "Backward walk")
    ]
    
    models = [
        ("/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model.pt", "IL"),
        ("/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model_rsl_rl.pt", "RSL_RL"),
        ("/workspace/SpotDMouse/P2-Terrain_Challenge/sim2real/model_9999_with_stats.pt", "RSL_RL_Original")
    ]
    
    # Store results for comparison
    all_results = {}
    
    for vx, vy, wz, desc in test_commands:
        print(f"\n{'='*60}")
        print(f"Testing command: {desc} (vx={vx}, vy={vy}, wz={wz})")
        print(f"{'='*60}")
        
        obs = create_test_observation(vx, vy, wz)
        results = {}
        
        for model_path, model_type in models:
            if os.path.exists(model_path):
                output = test_model(model_path, obs, model_type)
                if output is not None:
                    results[model_type] = output
            else:
                print(f"\nModel not found: {model_path}")
        
        all_results[desc] = results
    
    # Visualize comparison
    visualize_comparison(all_results)

def visualize_comparison(all_results):
    """Create visualization comparing model outputs"""
    
    n_commands = len(all_results)
    n_models = 3
    
    fig, axes = plt.subplots(n_commands, 2, figsize=(14, 4 * n_commands))
    if n_commands == 1:
        axes = axes.reshape(1, -1)
    
    joint_names = [
        "FR_hip", "FR_thigh", "FR_calf",
        "FL_hip", "FL_thigh", "FL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
        "RL_hip", "RL_thigh", "RL_calf"
    ]
    
    colors = {'IL': 'blue', 'RSL_RL': 'green', 'RSL_RL_Original': 'red'}
    
    for i, (command_desc, results) in enumerate(all_results.items()):
        # Plot joint positions
        ax = axes[i, 0]
        x = np.arange(12)
        width = 0.25
        
        for j, (model_type, output) in enumerate(results.items()):
            ax.bar(x + j * width, np.degrees(output), width, 
                   label=model_type, alpha=0.7, color=colors.get(model_type, 'gray'))
        
        ax.set_xlabel('Joint')
        ax.set_ylabel('Position (degrees)')
        ax.set_title(f'{command_desc} - Joint Positions')
        ax.set_xticks(x + width)
        ax.set_xticklabels([name.replace('_', '\n') for name in joint_names], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot differences from original model
        ax = axes[i, 1]
        if 'RSL_RL_Original' in results:
            ref_output = results['RSL_RL_Original']
            
            for model_type, output in results.items():
                if model_type != 'RSL_RL_Original':
                    diff = np.degrees(output - ref_output)
                    ax.plot(x, diff, 'o-', label=f'{model_type} vs Original', 
                           color=colors.get(model_type, 'gray'), markersize=6)
            
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Joint')
            ax.set_ylabel('Difference (degrees)')
            ax.set_title(f'{command_desc} - Difference from Original Model')
            ax.set_xticks(x)
            ax.set_xticklabels([f'J{i}' for i in range(12)])
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/model_comparison.png', dpi=150)
    print(f"\nSaved comparison plot to model_comparison.png")
    print(f"Open with: $BROWSER /workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/model_comparison.png")

def debug_normalization():
    """Debug normalization issues"""
    print("\n=== Debugging Normalization ===")
    
    # Load IL model checkpoint
    il_checkpoint = torch.load("/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model.pt", map_location='cpu')
    print("\nIL Model normalization:")
    print(f"  obs_mean: {il_checkpoint.get('obs_mean', 'Not found')}")
    print(f"  obs_std: {il_checkpoint.get('obs_std', 'Not found')}")
    
    # Load RSL_RL converted model
    rsl_checkpoint = torch.load("/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model_rsl_rl.pt", map_location='cpu')
    print("\nRSL_RL Converted normalization:")
    if 'obs_rms_mean' in rsl_checkpoint:
        print(f"  obs_rms_mean shape: {rsl_checkpoint['obs_rms_mean'].shape}")
        print(f"  obs_rms_var shape: {rsl_checkpoint['obs_rms_var'].shape}")
        print(f"  First 6 mean values: {rsl_checkpoint['obs_rms_mean'][:6]}")
        print(f"  First 6 var values: {rsl_checkpoint['obs_rms_var'][:6]}")
    
    # Load original model
    orig_checkpoint = torch.load("/workspace/SpotDMouse/P2-Terrain_Challenge/sim2real/model_9999_with_stats.pt", map_location='cpu')
    print("\nOriginal model normalization:")
    if 'obs_rms_mean' in orig_checkpoint:
        obs_mean = orig_checkpoint['obs_rms_mean']
        obs_var = orig_checkpoint['obs_rms_var']
        if isinstance(obs_mean, np.ndarray):
            obs_mean = torch.from_numpy(obs_mean).float()
            obs_var = torch.from_numpy(obs_var).float()
        print(f"  obs_rms_mean shape: {obs_mean.shape}")
        print(f"  obs_rms_var shape: {obs_var.shape}")
        print(f"  First 6 mean values: {obs_mean[:6]}")
        print(f"  First 6 var values: {obs_var[:6]}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test model outputs")
    parser.add_argument("--simple", action="store_true", 
                        help="Just test forward command on IL model")
    parser.add_argument("--debug", action="store_true",
                        help="Debug normalization issues")
    
    args = parser.parse_args()
    
    if args.simple:
        # Simple test with just forward command
        print("=== Simple forward command test ===")
        obs = create_test_observation(vx=0.3)
        test_model("/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model.pt", 
                   obs, model_type='IL')
    elif args.debug:
        debug_normalization()
    else:
        # Full comparison
        compare_models()

if __name__ == "__main__":
    main()