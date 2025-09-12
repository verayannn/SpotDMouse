#!/usr/bin/env python3
# filepath: /workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/add_normalization_to_il.py
"""
Add proper normalization statistics to IL model based on empirical testing
"""

import torch
import numpy as np
from typing import Dict, Tuple, List
import os
from tqdm import tqdm

# Default values for initialization
DEFAULT_STANDING_POSE = [0.0, 0.52, -1.05] * 4  # Hip, thigh, calf for each leg
HIDDEN_DIMS = [512, 256, 128]


def create_test_observations() -> torch.Tensor:
    """Create a diverse set of test observations to determine normalization statistics"""
    observations = []
    
    # Generate various velocity commands
    velocity_ranges = {
        'vx': np.linspace(-0.5, 0.5, 11),  # Forward/backward
        'vy': np.linspace(-0.3, 0.3, 7),   # Sideways
        'wz': np.linspace(-1.0, 1.0, 9),   # Turning
    }
    
    # Generate combinations of commands
    for vx in velocity_ranges['vx']:
        for vy in velocity_ranges['vy']:
            for wz in velocity_ranges['wz']:
                obs = torch.zeros(48)
                
                # Velocity commands
                obs[0] = vx
                obs[1] = vy
                obs[5] = wz
                
                # Joint positions - vary around default
                joint_noise = np.random.normal(0, 0.1, 12)
                for j, (pos, noise) in enumerate(zip(DEFAULT_STANDING_POSE, joint_noise)):
                    obs[6 + j] = pos + noise
                
                # Joint velocities - small random values
                obs[18:30] = torch.randn(12) * 0.5
                
                # Quaternion - slight variations from upright
                quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
                quat[:3] += torch.randn(3) * 0.05
                quat = quat / torch.norm(quat)  # Normalize
                obs[30:34] = quat
                
                # Previous actions = current joint positions with small variation
                obs[34:46] = obs[6:18] + torch.randn(12) * 0.05
                
                # Clock phase - various phases
                phase = np.random.uniform(0, 2 * np.pi)
                obs[46] = np.sin(phase)
                obs[47] = np.cos(phase)
                
                observations.append(obs)
    
    return torch.stack(observations)


def compute_empirical_statistics(model_path: str) -> Dict[str, torch.Tensor]:
    """Run test observations through model to determine normalization statistics"""
    print("Computing empirical normalization statistics...")
    
    # Load the trained IL model
    from train_il import MLPPolicy
    checkpoint = torch.load(model_path, map_location='cpu')
    model = MLPPolicy(obs_dim=48, action_dim=12)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Generate test observations
    test_obs = create_test_observations()
    print(f"Generated {len(test_obs)} test observations")
    
    # Try different normalization schemes and measure output stability
    best_score = float('inf')
    best_stats = None
    
    # Define search ranges for normalization parameters
    std_ranges = {
        'velocity': [0.1, 0.2, 0.3, 0.4, 0.5],
        'angular_velocity': [0.2, 0.3, 0.4, 0.5, 0.6],
        'joint_position': [0.2, 0.3, 0.4, 0.5],
        'joint_velocity': [1.0, 1.5, 2.0, 2.5],
        'orientation': [0.1, 0.15, 0.2, 0.25],
        'clock': [0.5, 0.7, 0.9, 1.0],
    }
    
    print("\nSearching for optimal normalization parameters...")
    
    for vel_std in std_ranges['velocity']:
        for ang_vel_std in std_ranges['angular_velocity']:
            for joint_pos_std in std_ranges['joint_position']:
                # Create normalization stats with current parameters
                obs_mean = torch.zeros(48)
                obs_std = torch.ones(48)
                
                # Apply means
                obs_mean[6:18] = torch.tensor(DEFAULT_STANDING_POSE)
                obs_mean[30:34] = torch.tensor([0.0, 0.0, 0.0, 1.0])
                obs_mean[34:46] = torch.tensor(DEFAULT_STANDING_POSE)
                
                # Apply stds
                obs_std[0:3] = vel_std
                obs_std[3:6] = ang_vel_std
                obs_std[6:18] = joint_pos_std
                obs_std[18:30] = 1.5  # Fixed for now
                obs_std[30:34] = 0.15  # Fixed for now
                obs_std[34:46] = joint_pos_std
                obs_std[46:48] = 0.7  # Fixed for now
                
                # Normalize observations
                test_obs_norm = (test_obs - obs_mean) / obs_std
                
                # Get model outputs
                with torch.no_grad():
                    outputs = model(test_obs_norm)
                
                # Compute stability metrics
                output_std = outputs.std()
                output_range = outputs.max() - outputs.min()
                
                # Score: we want reasonable variation but not extreme
                score = abs(output_std - 1.0) + 0.1 * abs(output_range - 4.0)
                
                if score < best_score:
                    best_score = score
                    best_stats = {
                        'obs_mean': obs_mean.clone(),
                        'obs_std': obs_std.clone(),
                        'metrics': {
                            'output_std': output_std.item(),
                            'output_range': output_range.item(),
                            'score': score.item()
                        }
                    }
    
    print(f"\nBest normalization found:")
    print(f"  Output std: {best_stats['metrics']['output_std']:.3f}")
    print(f"  Output range: {best_stats['metrics']['output_range']:.3f}")
    print(f"  Score: {best_stats['metrics']['score']:.3f}")
    
    # Refine the best stats with full grid search
    print("\nRefining normalization parameters...")
    refined_stats = refine_normalization_stats(model, test_obs, best_stats)
    
    return refined_stats


def refine_normalization_stats(model, test_obs: torch.Tensor, initial_stats: Dict) -> Dict:
    """Refine normalization statistics with finer grid search"""
    obs_mean = initial_stats['obs_mean'].clone()
    obs_std = initial_stats['obs_std'].clone()
    
    # Fine-tune each segment
    segments = [
        ('velocity', 0, 3),
        ('angular_velocity', 3, 6),
        ('joint_position', 6, 18),
        ('joint_velocity', 18, 30),
        ('orientation', 30, 34),
        ('prev_actions', 34, 46),
        ('clock', 46, 48),
    ]
    
    for segment_name, start_idx, end_idx in segments:
        print(f"  Optimizing {segment_name}...")
        
        current_std = obs_std[start_idx:end_idx].mean().item()
        best_std = current_std
        best_metric = float('inf')
        
        # Try variations around current value
        for std_mult in np.linspace(0.7, 1.3, 13):
            test_std = obs_std.clone()
            test_std[start_idx:end_idx] = current_std * std_mult
            
            # Normalize and test
            test_obs_norm = (test_obs - obs_mean) / test_std
            with torch.no_grad():
                outputs = model(test_obs_norm)
            
            # Measure output quality
            output_std = outputs.std()
            metric = abs(output_std - 1.0)  # Target std of 1.0
            
            if metric < best_metric:
                best_metric = metric
                best_std = current_std * std_mult
        
        obs_std[start_idx:end_idx] = best_std
        print(f"    Best std: {best_std:.3f}")
    
    # Test final normalization
    print("\nTesting final normalization...")
    test_obs_norm = (test_obs - obs_mean) / obs_std
    with torch.no_grad():
        outputs = model(test_obs_norm)
    
    print(f"Final output statistics:")
    print(f"  Mean: {outputs.mean():.3f}")
    print(f"  Std: {outputs.std():.3f}")
    print(f"  Range: [{outputs.min():.3f}, {outputs.max():.3f}]")
    
    # Return complete stats
    return {
        'obs_mean': obs_mean,
        'obs_std': obs_std,
        'action_mean': torch.zeros(12),
        'action_std': torch.ones(12)
    }


def create_observation_batch(commands: List[Tuple[float, float, float]]) -> torch.Tensor:
    """Create a batch of observations with different velocity commands"""
    batch_size = len(commands)
    obs = torch.zeros(batch_size, 48)
    
    for i, (vx, vy, wz) in enumerate(commands):
        # Velocity commands
        obs[i, 0] = vx
        obs[i, 1] = vy
        obs[i, 5] = wz
        
        # Default standing pose for joints
        for j, pos in enumerate(DEFAULT_STANDING_POSE):
            obs[i, 6 + j] = pos
        
        # Quaternion (upright)
        obs[i, 30:34] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        
        # Previous actions = current joint positions
        obs[i, 34:46] = obs[i, 6:18]
        
        # Clock phase
        obs[i, 46] = 0.0
        obs[i, 47] = 1.0
    
    return obs


def create_dummy_critic_weights() -> Dict:
    """Create randomly initialized critic weights for RSL_RL"""
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
    """Create Adam optimizer state dict for RSL_RL"""
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


def add_normalization_to_il_model() -> str:
    """Add empirically determined normalization to IL model"""
    model_path = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model.pt"
    print(f"Loading IL model from: {model_path}")
    
    # Compute empirical normalization statistics
    stats = compute_empirical_statistics(model_path)
    
    # Load checkpoint and add normalization
    checkpoint = torch.load(model_path, map_location='cpu')
    checkpoint.update(stats)
    
    # Save normalized model
    output_path = model_path.replace('.pt', '_normalized.pt')
    torch.save(checkpoint, output_path)
    print(f"\nSaved normalized model to: {output_path}")
    
    # Create RSL_RL version
    create_rsl_rl_version(output_path, stats['obs_mean'], stats['obs_std'])
    
    return output_path


def create_rsl_rl_version(il_model_path: str, obs_mean: torch.Tensor, obs_std: torch.Tensor) -> str:
    """Create RSL_RL compatible version of IL model"""
    print("\nCreating RSL_RL compatible version...")
    
    # Load IL model
    il_checkpoint = torch.load(il_model_path, map_location='cpu')
    il_state_dict = il_checkpoint['model_state_dict']
    
    # Build RSL_RL state dict
    rsl_rl_state_dict = {}
    
    # The IL model already uses 'actor.*' naming, so just copy it
    for key, value in il_state_dict.items():
        if key.startswith('actor.'):
            rsl_rl_state_dict[key] = value
    
    # Add action std parameter
    rsl_rl_state_dict['std'] = torch.ones(12) * 0.5
    
    # Add dummy critic
    rsl_rl_state_dict.update(create_dummy_critic_weights())
    
    # Create full checkpoint
    rsl_rl_checkpoint = {
        'model_state_dict': rsl_rl_state_dict,
        'optimizer_state_dict': create_optimizer_state_dict(len(rsl_rl_state_dict)),
        'iter': 0,
        'obs_rms_mean': obs_mean,
        'obs_rms_var': obs_std ** 2,
        'num_obs': 48,
        'num_actions': 12,
        'infos': {
            'note': 'IL model with empirically determined normalization'
        }
    }
    
    # Save RSL_RL version
    output_path = il_model_path.replace('_normalized.pt', '_normalized_rsl_rl.pt')
    torch.save(rsl_rl_checkpoint, output_path)
    print(f"Saved RSL_RL version to: {output_path}")
    
    return output_path


def test_normalized_model() -> None:
    """Test the normalized model with various commands"""
    try:
        from train_il import MLPPolicy
    except ImportError:
        print("\n⚠️  Skipping test - train_il module not found")
        return
    
    print("\n=== Testing Normalized Model ===")
    
    model_path = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model_normalized.pt"
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found at {model_path}")
        return
    
    # Load model and normalization stats
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = MLPPolicy(obs_dim=48, action_dim=12)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test with various commands
    test_commands = [
        (0.0, 0.0, 0.0),   # Stand
        (0.3, 0.0, 0.0),   # Forward
        (0.0, 0.2, 0.0),   # Sideways
        (0.0, 0.0, 0.5),   # Turn
        (-0.2, 0.0, 0.0),  # Backward
    ]
    
    obs_batch = create_observation_batch(test_commands)
    obs_norm = (obs_batch - checkpoint['obs_mean']) / checkpoint['obs_std']
    
    with torch.no_grad():
        outputs = model(obs_norm)
    
    print("\nNormalized model outputs:")
    labels = ["Stand", "Forward", "Sideways", "Turn", "Backward"]
    for i, label in enumerate(labels):
        out = outputs[i].numpy()
        joint_degs = np.degrees(out[:3])
        print(f"{label:10s}: mean={out.mean():6.3f}, std={out.std():5.3f}, "
              f"first 3 joints (deg): [{joint_degs[0]:5.1f}, {joint_degs[1]:5.1f}, {joint_degs[2]:5.1f}]")
    
    # Print normalization parameters used
    print("\nNormalization parameters:")
    print(f"  Velocity std: {checkpoint['obs_std'][0:3].mean():.3f}")
    print(f"  Angular velocity std: {checkpoint['obs_std'][3:6].mean():.3f}")
    print(f"  Joint position std: {checkpoint['obs_std'][6:18].mean():.3f}")
    print(f"  Joint velocity std: {checkpoint['obs_std'][18:30].mean():.3f}")


def print_deployment_instructions() -> None:
    """Print instructions for using the normalized models"""
    print("\n✅ Success! Your normalized models are ready:")
    print(f"  IL model: /workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model_normalized.pt")
    print(f"  RSL_RL model: /workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model_normalized_rsl_rl.pt")
    
    print("\nTo test in simulation:")
    print("cd /workspace/isaaclab")
    print("CUDA_VISIBLE_DEVICES=2 python scripts/reinforcement_learning/rsl_rl/play.py \\")
    print("  --task=Isaac-Velocity-Flat-Custom-Quad-v0 \\")
    print("  --checkpoint=/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model_normalized_rsl_rl.pt \\")
    print("  --num_envs 300")


def main():
    """Main function to add empirical normalization to IL model"""
    # Add normalization to model
    normalized_path = add_normalization_to_il_model()
    
    # Test the normalized model
    test_normalized_model()
    
    # Print deployment instructions
    print_deployment_instructions()


if __name__ == "__main__":
    main()