#!/usr/bin/env python3
# filepath: /workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/convert_il_to_rsl_rl.py
#!/usr/bin/env python3
# filepath: /workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/convert_il_to_rsl_rl.py
"""
Convert IL-trained model to RSL_RL ActorCritic format
"""

import torch
import argparse
import os

def convert_il_to_rsl_rl(il_checkpoint_path, output_path):
    """Convert IL model to RSL_RL ActorCritic format"""
    
    # Load IL checkpoint
    print(f"Loading IL model from: {il_checkpoint_path}")
    il_checkpoint = torch.load(il_checkpoint_path, map_location='cpu')
    
    # Extract IL model state dict
    il_state_dict = il_checkpoint.get('model_state_dict', il_checkpoint)
    
    # Create RSL_RL compatible state dict
    rsl_rl_state_dict = {}
    
    # Map IL model weights to actor weights
    # IL uses: net.0.weight, net.2.weight, etc.
    # RSL_RL uses: actor.0.weight, actor.2.weight, etc.
    for key, value in il_state_dict.items():
        if key.startswith('net.'):
            # Replace 'net.' with 'actor.'
            new_key = key.replace('net.', 'actor.')
            rsl_rl_state_dict[new_key] = value
            print(f"Mapped {key} -> {new_key}")
    
    # Add action std parameter (required by RSL_RL)
    # This controls exploration noise during training
    num_actions = 12
    rsl_rl_state_dict['std'] = torch.ones(num_actions) * 0.5  # Moderate initial std
    
    # Create dummy critic weights (same architecture as actor)
    # The critic will be trained from scratch in RL
    hidden_dims = [512, 256, 128]
    
    # Layer 0: Input to first hidden
    rsl_rl_state_dict['critic.0.weight'] = torch.randn(hidden_dims[0], 48) * 0.1
    rsl_rl_state_dict['critic.0.bias'] = torch.zeros(hidden_dims[0])
    
    # Layer 2: First hidden to second hidden
    rsl_rl_state_dict['critic.2.weight'] = torch.randn(hidden_dims[1], hidden_dims[0]) * 0.1
    rsl_rl_state_dict['critic.2.bias'] = torch.zeros(hidden_dims[1])
    
    # Layer 4: Second hidden to third hidden
    rsl_rl_state_dict['critic.4.weight'] = torch.randn(hidden_dims[2], hidden_dims[1]) * 0.1
    rsl_rl_state_dict['critic.4.bias'] = torch.zeros(hidden_dims[2])
    
    # Layer 6: Third hidden to output (value)
    rsl_rl_state_dict['critic.6.weight'] = torch.randn(1, hidden_dims[2]) * 0.1
    rsl_rl_state_dict['critic.6.bias'] = torch.zeros(1)
    
    print("\nCreated critic network with random initialization")
    
    # Create dummy optimizer state dict for Adam optimizer
    # RSL_RL expects this to exist even if we're just playing
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
            'params': list(range(len(rsl_rl_state_dict)))  # Parameter indices
        }]
    }
    
    # Create RSL_RL checkpoint format
    rsl_rl_checkpoint = {
        'model_state_dict': rsl_rl_state_dict,
        'optimizer_state_dict': optimizer_state_dict,  # Now properly initialized
        'iter': 0,
        'infos': {
            'pretrained_from_il': True,
            'il_checkpoint': il_checkpoint_path,
            'note': 'Actor weights from IL, critic initialized randomly'
        }
    }
    
    # Add normalization stats if available
    if 'obs_mean' in il_checkpoint and il_checkpoint['obs_mean'] is not None:
        rsl_rl_checkpoint['obs_rms_mean'] = il_checkpoint['obs_mean']
        rsl_rl_checkpoint['obs_rms_var'] = il_checkpoint['obs_std'] ** 2
        print("\nAdded observation normalization stats from IL model")
    else:
        # Use default normalization
        rsl_rl_checkpoint['obs_rms_mean'] = torch.zeros(48)
        rsl_rl_checkpoint['obs_rms_var'] = torch.ones(48)
        print("\nUsing default observation normalization (mean=0, var=1)")
    
    # Add dimensions
    rsl_rl_checkpoint['num_obs'] = 48
    rsl_rl_checkpoint['num_actions'] = 12
    
    # Save converted checkpoint
    torch.save(rsl_rl_checkpoint, output_path)
    print(f"\nSaved RSL_RL compatible checkpoint to: {output_path}")
    
    # Print summary
    print("\n=== Conversion Summary ===")
    print(f"Actor weights: Loaded from IL model")
    print(f"Critic weights: Randomly initialized")
    print(f"Action std: {rsl_rl_state_dict['std'].mean().item():.3f}")
    print(f"Optimizer: Adam with default parameters")
    print(f"Total parameters: {sum(p.numel() for p in rsl_rl_state_dict.values()):,}")
    
    return output_path

def verify_checkpoint(checkpoint_path):
    """Verify the converted checkpoint can be loaded"""
    print(f"\n=== Verifying checkpoint: {checkpoint_path} ===")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Check for required keys
    required_keys = ['std', 'actor.0.weight', 'critic.0.weight']
    missing_keys = [k for k in required_keys if k not in state_dict]
    
    if missing_keys:
        print(f"ERROR: Missing required keys: {missing_keys}")
        return False
    
    print("✓ All required keys present")
    
    # Check optimizer state dict
    if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
        print("✓ Optimizer state dict present")
    else:
        print("✗ Optimizer state dict missing or None")
        return False
    
    # Print layer shapes
    print("\nActor layers:")
    for key in sorted([k for k in state_dict.keys() if k.startswith('actor.')]):
        print(f"  {key}: {state_dict[key].shape}")
    
    print("\nCritic layers:")
    for key in sorted([k for k in state_dict.keys() if k.startswith('critic.')]):
        print(f"  {key}: {state_dict[key].shape}")
    
    print(f"\nAction std: {state_dict['std'].shape}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert IL model to RSL_RL format")
    parser.add_argument("--il-checkpoint", 
                        default="/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model.pt",
                        help="Path to IL model checkpoint")
    parser.add_argument("--output", 
                        default="/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model_rsl_rl.pt",
                        help="Output path for RSL_RL checkpoint")
    parser.add_argument("--verify", action="store_true",
                        help="Verify the converted checkpoint")
    
    args = parser.parse_args()
    
    # Convert the model
    output_path = convert_il_to_rsl_rl(args.il_checkpoint, args.output)
    
    # Verify if requested
    if args.verify:
        verify_checkpoint(output_path)
    
    print(f"\nNow you can use this model with RSL_RL:")
    print(f"cd /workspace/isaaclab")
    print(f"CUDA_VISIBLE_DEVICES=2 python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Flat-Custom-Quad-v0 --checkpoint={output_path} --num_envs 300")

if __name__ == "__main__":
    main()