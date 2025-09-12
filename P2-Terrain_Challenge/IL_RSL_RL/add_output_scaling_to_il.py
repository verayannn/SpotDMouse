#!/usr/bin/env python3
# filepath: /workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/add_output_scaling_to_il.py
"""
Add output scaling to IL model for proper motion in Isaac Sim
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import os

# Default values
DEFAULT_STANDING_POSE = [0.0, 0.52, -1.05] * 4  # Hip, thigh, calf for each leg
HIDDEN_DIMS = [512, 256, 128]


def test_scaling_factors(model_path: str, scaling_factors: List[float]) -> Dict:
    """Test different scaling factors to find optimal value"""
    from train_il import MLPPolicy
    
    print("Testing different scaling factors...")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = MLPPolicy(obs_dim=48, action_dim=12)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get normalization stats
    obs_mean = checkpoint.get('obs_mean', torch.zeros(48))
    obs_std = checkpoint.get('obs_std', torch.ones(48))
    
    # Test commands
    test_commands = [
        ("Stand", 0.0, 0.0, 0.0),
        ("Forward", 0.3, 0.0, 0.0),
        ("Turn", 0.0, 0.0, 0.5),
    ]
    
    results = {}
    
    for scale in scaling_factors:
        print(f"\n  Testing scale factor: {scale}x")
        
        scale_results = []
        for cmd_name, vx, vy, wz in test_commands:
            # Create observation
            obs = torch.zeros(1, 48)
            obs[0, 0] = vx
            obs[0, 1] = vy
            obs[0, 5] = wz
            obs[0, 6:18] = torch.tensor(DEFAULT_STANDING_POSE)
            obs[0, 30:34] = torch.tensor([0.0, 0.0, 0.0, 1.0])
            obs[0, 34:46] = obs[0, 6:18]
            obs[0, 46:48] = torch.tensor([0.0, 1.0])
            
            # Normalize and get output
            obs_norm = (obs - obs_mean) / obs_std
            with torch.no_grad():
                output = model(obs_norm)
                scaled_output = output * scale
            
            scaled_np = scaled_output[0].numpy()
            print(f"    {cmd_name}: range=[{scaled_np.min():.2f}, {scaled_np.max():.2f}], "
                  f"magnitude={np.linalg.norm(scaled_np):.2f}")
            
            scale_results.append({
                'command': cmd_name,
                'output': scaled_np,
                'magnitude': np.linalg.norm(scaled_np)
            })
        
        results[scale] = scale_results
    
    return results


def create_scaled_il_model(scale_factor: float = 1.0) -> str:
    """Create IL model with output scaling wrapper"""
    
    class ScaledILPolicy(torch.nn.Module):
        """Wrapper that scales IL model outputs"""
        def __init__(self, base_model, scale_factor):
            super().__init__()
            self.base_model = base_model
            self.scale_factor = scale_factor
            
        def forward(self, obs):
            base_output = self.base_model(obs)
            return base_output * self.scale_factor
    
    # Load original model
    model_path = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model_normalized.pt"
    print(f"\nLoading IL model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Add scaling factor to checkpoint
    checkpoint['action_scale_factor'] = scale_factor
    
    # Save scaled version
    output_path = model_path.replace('_normalized.pt', f'_scaled_{scale_factor}x.pt')
    torch.save(checkpoint, output_path)
    print(f"Saved scaled IL model to: {output_path}")
    
    return output_path


def create_scaled_rsl_rl_version(il_model_path: str, scale_factor: float) -> str:
    """Create RSL_RL compatible version with output scaling"""
    print(f"\nCreating RSL_RL version with {scale_factor}x output scaling...")
    
    # Load IL model
    il_checkpoint = torch.load(il_model_path, map_location='cpu')
    il_state_dict = il_checkpoint['model_state_dict']
    
    # Scale the output layer weights and biases
    scaled_state_dict = {}
    
    for key, value in il_state_dict.items():
        if key == 'actor.6.weight':  # Output layer weights
            scaled_state_dict[key] = value * scale_factor
        elif key == 'actor.6.bias':  # Output layer biases
            scaled_state_dict[key] = value * scale_factor
        else:
            scaled_state_dict[key] = value
    
    # Build RSL_RL state dict
    rsl_rl_state_dict = {}
    
    # Copy scaled actor weights
    for key, value in scaled_state_dict.items():
        if key.startswith('actor.'):
            rsl_rl_state_dict[key] = value
    
    # Add action std parameter
    rsl_rl_state_dict['std'] = torch.ones(12) * 0.5
    
    # Add dummy critic
    critic_weights = create_dummy_critic_weights()
    rsl_rl_state_dict.update(critic_weights)
    
    # Get normalization stats
    obs_mean = il_checkpoint.get('obs_mean', torch.zeros(48))
    obs_std = il_checkpoint.get('obs_std', torch.ones(48))
    
    # Create full checkpoint
    rsl_rl_checkpoint = {
        'model_state_dict': rsl_rl_state_dict,
        'optimizer_state_dict': create_optimizer_state_dict(len(rsl_rl_state_dict)),
        'iter': 0,
        'obs_rms_mean': obs_mean,
        'obs_rms_var': obs_std ** 2,
        'num_obs': 48,
        'num_actions': 12,
        'action_scale_factor': scale_factor,
        'infos': {
            'note': f'IL model with {scale_factor}x output scaling'
        }
    }
    
    # Save RSL_RL version
    output_path = il_model_path.replace('.pt', '_rsl_rl.pt')
    torch.save(rsl_rl_checkpoint, output_path)
    print(f"Saved RSL_RL version to: {output_path}")
    
    return output_path


def create_dummy_critic_weights() -> Dict:
    """Create dummy critic weights"""
    critic_weights = {}
    
    layer_specs = [
        (0, 48, HIDDEN_DIMS[0]),
        (2, HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
        (4, HIDDEN_DIMS[1], HIDDEN_DIMS[2]),
        (6, HIDDEN_DIMS[2], 1)
    ]
    
    for layer_idx, in_dim, out_dim in layer_specs:
        critic_weights[f'critic.{layer_idx}.weight'] = torch.randn(out_dim, in_dim) * 0.1
        critic_weights[f'critic.{layer_idx}.bias'] = torch.zeros(out_dim)
    
    return critic_weights


def create_optimizer_state_dict(num_params: int) -> Dict:
    """Create optimizer state dict"""
    return {
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
            'params': list(range(num_params))
        }]
    }


def test_scaled_model(model_path: str, scale_factor: float):
    """Test the scaled model outputs"""
    from train_il import MLPPolicy
    
    print(f"\n=== Testing Model with {scale_factor}x Scaling ===")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Create model
    model = MLPPolicy(obs_dim=48, action_dim=12)
    
    # Load actor weights
    actor_state_dict = {k.replace('actor.', ''): v for k, v in state_dict.items() if k.startswith('actor.')}
    model.actor.load_state_dict(actor_state_dict)
    model.eval()
    
    # Get normalization
    obs_mean = checkpoint.get('obs_rms_mean', torch.zeros(48))
    obs_var = checkpoint.get('obs_rms_var', torch.ones(48))
    
    # Convert numpy to torch if needed
    if isinstance(obs_mean, np.ndarray):
        obs_mean = torch.from_numpy(obs_mean).float()
    if isinstance(obs_var, np.ndarray):
        obs_var = torch.from_numpy(obs_var).float()
    
    obs_std = torch.sqrt(obs_var + 1e-8)
    
    # Test commands
    test_commands = [
        ("Stand", 0.0, 0.0, 0.0),
        ("Forward", 0.3, 0.0, 0.0),
        ("Turn", 0.0, 0.0, 0.5),
        ("Forward+Turn", 0.3, 0.0, 0.3),
    ]
    
    print("\nScaled outputs:")
    for cmd_name, vx, vy, wz in test_commands:
        obs = torch.zeros(1, 48)
        obs[0, 0] = vx
        obs[0, 1] = vy
        obs[0, 5] = wz
        obs[0, 6:18] = torch.tensor(DEFAULT_STANDING_POSE)
        obs[0, 30:34] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        obs[0, 34:46] = obs[0, 6:18]
        obs[0, 46:48] = torch.tensor([0.0, 1.0])
        
        obs_norm = (obs - obs_mean.unsqueeze(0)) / obs_std.unsqueeze(0)
        
        with torch.no_grad():
            output = model(obs_norm)
        
        output_np = output[0].numpy()
        print(f"{cmd_name:12s}: range=[{output_np.min():6.2f}, {output_np.max():6.2f}], "
              f"magnitude={np.linalg.norm(output_np):6.2f}, "
              f"first 3 joints (deg): [{np.degrees(output_np[0]):6.1f}, "
              f"{np.degrees(output_np[1]):6.1f}, {np.degrees(output_np[2]):6.1f}]")


def main():
    """Main function to add scaling to IL model"""
    print("="*80)
    print("Adding Output Scaling to IL Model")
    print("="*80)
    
    # Test different scaling factors
    model_path = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model_normalized.pt"
    
    if os.path.exists(model_path):
        scaling_factors = [1.0, 2.0, 3.0, 4.0, 5.0]
        test_results = test_scaling_factors(model_path, scaling_factors)
        
        # Let user choose or use default
        print("\n" + "="*80)
        print("Based on typical joint ranges for quadrupeds:")
        print("  - Joint angles should be roughly in [-3, 3] radians")
        print("  - Total magnitude should be around 5-10 for movement")
        print("\nRecommended scaling factor: 3.0x")
        scale_factor = 3.0
    else:
        print(f"⚠️  Model not found at {model_path}")
        print("Using default scaling factor: 3.0x")
        scale_factor = 3.0
    
    # Create scaled versions
    print(f"\nCreating scaled models with {scale_factor}x factor...")
    
    # Create scaled IL model
    scaled_il_path = create_scaled_il_model(scale_factor)
    
    # Create RSL_RL version
    rsl_rl_path = create_scaled_rsl_rl_version(scaled_il_path, scale_factor)
    
    # Test the scaled RSL_RL model
    test_scaled_model(rsl_rl_path, scale_factor)
    
    print("\n" + "="*80)
    print("✅ SUCCESS! Scaled models created")
    print("="*80)
    
    print(f"\nScaled IL model: {scaled_il_path}")
    print(f"Scaled RSL_RL model: {rsl_rl_path}")
    
    print("\nTo test in Isaac Sim:")
    print("cd /workspace/isaaclab")
    print("CUDA_VISIBLE_DEVICES=2 python scripts/reinforcement_learning/rsl_rl/play.py \\")
    print(f"  --task=Isaac-Velocity-Flat-Custom-Quad-v0 \\")
    print(f"  --checkpoint={rsl_rl_path} \\")
    print("  --num_envs 1")
    
    print("\nIf movement is still too small/large, create new versions with different scaling:")
    print(f"  - Too small: Try 5.0x or 10.0x")
    print(f"  - Too large: Try 2.0x or 1.5x")


if __name__ == "__main__":
    main()