#!/usr/bin/env python3
"""
Simple script to create IL models with different output scaling factors
"""

import torch
import numpy as np
import os

def create_scaled_model(scale_factor: float):
    """Create RSL_RL checkpoint with scaled IL model outputs"""
    
    print(f"\nCreating model with {scale_factor}x output scaling...")
    
    # Load the normalized IL model
    base_path = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model_normalized.pt"
    checkpoint = torch.load(base_path, map_location='cpu')
    
    # Get the model state dict
    state_dict = checkpoint['model_state_dict'].copy()
    
    # Scale the output layer (layer 6 is the output layer)
    state_dict['actor.6.weight'] = state_dict['actor.6.weight'] * scale_factor
    state_dict['actor.6.bias'] = state_dict['actor.6.bias'] * scale_factor
    
    # Add std parameter for RSL_RL
    state_dict['std'] = torch.ones(12) * 0.5
    
    # Add dummy critic (required by RSL_RL)
    hidden_dims = [512, 256, 128]
    state_dict['critic.0.weight'] = torch.randn(hidden_dims[0], 48) * 0.1
    state_dict['critic.0.bias'] = torch.zeros(hidden_dims[0])
    state_dict['critic.2.weight'] = torch.randn(hidden_dims[1], hidden_dims[0]) * 0.1
    state_dict['critic.2.bias'] = torch.zeros(hidden_dims[1])
    state_dict['critic.4.weight'] = torch.randn(hidden_dims[2], hidden_dims[1]) * 0.1
    state_dict['critic.4.bias'] = torch.zeros(hidden_dims[2])
    state_dict['critic.6.weight'] = torch.randn(1, hidden_dims[2]) * 0.1
    state_dict['critic.6.bias'] = torch.zeros(1)
    
    # Create RSL_RL checkpoint with all required fields
    rsl_checkpoint = {
        'model_state_dict': state_dict,
        'optimizer_state_dict': {
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
                'params': list(range(len(state_dict)))
            }]
        },
        'iter': 0,
        'obs_rms_mean': checkpoint.get('obs_mean', torch.zeros(48)),
        'obs_rms_var': checkpoint.get('obs_std', torch.ones(48)) ** 2,
        'num_obs': 48,
        'num_actions': 12,
        'infos': {
            'episode': 0,
            'ep_info': {},
            'action_scale_factor': scale_factor,
            'source': 'IL model with output scaling'
        }
    }
    
    # Save the scaled model
    output_path = f"/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/il_scaled_{scale_factor}x.pt"
    torch.save(rsl_checkpoint, output_path)
    print(f"Saved: {output_path}")
    
    return output_path

def main():
    """Create models with different scaling factors"""
    
    print("Creating IL models with different output scaling factors...")
    print("="*60)
    
    # Scaling factors to test
    scale_factors = [1.5, 1.61, 1.72, 1.83, 1.94, 2.06, 2.17, 2.28, 2.39, 2.5]
    
    created_models = []
    for scale in scale_factors:
        path = create_scaled_model(scale)
        created_models.append((scale, path))
    
    print("\n" + "="*60)
    print("✅ Created scaled models:")
    print("="*60)
    
    for scale, path in created_models:
        print(f"\n{scale}x scaling:")
        print(f"CUDA_VISIBLE_DEVICES=2 python scripts/reinforcement_learning/rsl_rl/play.py \\")
        print(f"  --task=Isaac-Velocity-Flat-Custom-Quad-v0 \\")
        print(f"  --checkpoint={path} \\")
        print(f"  --num_envs 1")
    
    print("\n💡 Test each model and see which scaling factor produces proper movement!")

if __name__ == "__main__":
    main()