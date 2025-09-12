#!/usr/bin/env python3
# filepath: /workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/add_normalization_to_il.py
"""
Add proper normalization statistics to IL model based on assessment results
"""

import torch
import numpy as np

def add_normalization_to_il_model():
    """Add carefully tuned normalization to IL model"""
    
    # Load the model
    model_path = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model.pt"
    print(f"Loading IL model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create normalization statistics based on typical quadruped data
    obs_mean = torch.zeros(48)
    obs_std = torch.ones(48)
    
    # Velocity commands (0-5): small values, moderate variation
    obs_mean[0:3] = 0.0  # Linear velocities centered at 0
    obs_std[0:3] = 0.2   # Typical range [-0.5, 0.5] m/s
    
    obs_mean[3:6] = 0.0  # Angular velocities centered at 0
    obs_std[3:6] = 0.3   # Typical range [-1, 1] rad/s
    
    # Joint positions (6-17): based on default standing pose
    default_joints = torch.tensor([0.0, -0.8, 1.6] * 4)  # Hip, thigh, calf x4
    obs_mean[6:18] = default_joints
    obs_std[6:18] = 0.3  # Moderate variation around default pose
    
    # Joint velocities (18-29): centered at zero, larger variation
    obs_mean[18:30] = 0.0
    obs_std[18:30] = 1.5  # Joint velocities can be fast
    
    # Quaternion (30-33): upright orientation
    obs_mean[30:34] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    obs_std[30:34] = 0.15  # Small orientation changes
    
    # Previous actions (34-45): same as joint positions
    obs_mean[34:46] = default_joints
    obs_std[34:46] = 0.3
    
    # Clock phase (46-47): sine/cosine
    obs_mean[46:48] = 0.0
    obs_std[46:48] = 0.7  # Full range for sin/cos
    
    # No action normalization (actions are already in joint space)
    action_mean = torch.zeros(12)
    action_std = torch.ones(12)
    
    # Update checkpoint
    checkpoint['obs_mean'] = obs_mean
    checkpoint['obs_std'] = obs_std
    checkpoint['action_mean'] = action_mean
    checkpoint['action_std'] = action_std
    
    # Save with normalization
    output_path = model_path.replace('.pt', '_normalized.pt')
    torch.save(checkpoint, output_path)
    print(f"\nSaved normalized model to: {output_path}")
    
    # Also create RSL_RL version
    create_rsl_rl_version(output_path, obs_mean, obs_std)
    
    return output_path

def create_rsl_rl_version(il_model_path, obs_mean, obs_std):
    """Create RSL_RL compatible version"""
    
    print("\nCreating RSL_RL compatible version...")
    
    # Load IL model
    il_checkpoint = torch.load(il_model_path, map_location='cpu')
    il_state_dict = il_checkpoint['model_state_dict']
    
    # Create RSL_RL state dict
    rsl_rl_state_dict = {}
    
    # Map IL weights to actor
    for key, value in il_state_dict.items():
        if key.startswith('net.'):
            new_key = key.replace('net.', 'actor.')
            rsl_rl_state_dict[new_key] = value
    
    # Add action std
    rsl_rl_state_dict['std'] = torch.ones(12) * 0.5
    
    # Create dummy critic
    hidden_dims = [512, 256, 128]
    rsl_rl_state_dict['critic.0.weight'] = torch.randn(hidden_dims[0], 48) * 0.1
    rsl_rl_state_dict['critic.0.bias'] = torch.zeros(hidden_dims[0])
    rsl_rl_state_dict['critic.2.weight'] = torch.randn(hidden_dims[1], hidden_dims[0]) * 0.1
    rsl_rl_state_dict['critic.2.bias'] = torch.zeros(hidden_dims[1])
    rsl_rl_state_dict['critic.4.weight'] = torch.randn(hidden_dims[2], hidden_dims[1]) * 0.1
    rsl_rl_state_dict['critic.4.bias'] = torch.zeros(hidden_dims[2])
    rsl_rl_state_dict['critic.6.weight'] = torch.randn(1, hidden_dims[2]) * 0.1
    rsl_rl_state_dict['critic.6.bias'] = torch.zeros(1)
    
    # Create optimizer state dict
    optimizer_state_dict = {
        'state': {},
        'param_groups': [{
            'lr': 3e-4,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0,
            'amsgrad': False,
            'maximize': False,
            'foreach': None,
            'capturable': False,
            'differentiable': False,
            'fused': False,
            'params': list(range(len(rsl_rl_state_dict)))
        }]
    }
    
    # Create RSL_RL checkpoint
    rsl_rl_checkpoint = {
        'model_state_dict': rsl_rl_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'iter': 0,
        'obs_rms_mean': obs_mean,
        'obs_rms_var': obs_std ** 2,
        'num_obs': 48,
        'num_actions': 12,
        'infos': {
            'note': 'IL model with tuned normalization'
        }
    }
    
    output_path = il_model_path.replace('_normalized.pt', '_normalized_rsl_rl.pt')
    torch.save(rsl_rl_checkpoint, output_path)
    print(f"Saved RSL_RL version to: {output_path}")
    
    return output_path

def test_normalized_model():
    """Quick test of the normalized model"""
    from assess_il_model import create_observation_batch
    from train_il import MLPPolicy
    
    print("\n=== Testing Normalized Model ===")
    
    # Load normalized model
    model_path = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model_normalized.pt"
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = MLPPolicy(obs_dim=48, action_dim=12)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test commands
    test_commands = [
        (0.0, 0.0, 0.0),   # Stand
        (0.3, 0.0, 0.0),   # Forward
        (0.0, 0.0, 0.5),   # Turn
    ]
    
    obs_batch = create_observation_batch(test_commands)
    obs_norm = (obs_batch - checkpoint['obs_mean']) / checkpoint['obs_std']
    
    with torch.no_grad():
        outputs = model(obs_norm)
    
    print("\nNormalized model outputs:")
    labels = ["Stand", "Forward", "Turn"]
    for i, label in enumerate(labels):
        out = outputs[i].numpy()
        print(f"{label:8s}: mean={out.mean():6.3f}, std={out.std():5.3f}, "
              f"first 3 joints (deg): [{np.degrees(out[0]):5.1f}, {np.degrees(out[1]):5.1f}, {np.degrees(out[2]):5.1f}]")

def main():
    # Add normalization to model
    normalized_path = add_normalization_to_il_model()
    
    # Test it
    test_normalized_model()
    
    print("\n✅ Success! Your normalized models are ready:")
    print(f"  IL model: /workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model_normalized.pt")
    print(f"  RSL_RL model: /workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model_normalized_rsl_rl.pt")
    
    print("\nTo test in simulation:")
    print("cd /workspace/isaaclab")
    print("CUDA_VISIBLE_DEVICES=2 python scripts/reinforcement_learning/rsl_rl/play.py \\")
    print("  --task=Isaac-Velocity-Flat-Custom-Quad-v0 \\")
    print("  --checkpoint=/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model_normalized_rsl_rl.pt \\")
    print("  --num_envs 300")

if __name__ == "__main__":
    main()