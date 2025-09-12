#!/usr/bin/env python3
# filepath: /workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/debug_model_in_sim.py
"""
Debug why the IL model doesn't produce movement in Isaac Sim
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def load_and_inspect_checkpoint(checkpoint_path: str) -> Dict:
    """Load and inspect checkpoint contents"""
    print(f"\n=== Inspecting: {checkpoint_path} ===")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Print all top-level keys
    print("\nCheckpoint keys:")
    for key in checkpoint.keys():
        if key == 'model_state_dict':
            print(f"  {key}: {len(checkpoint[key])} parameters")
        elif isinstance(checkpoint[key], torch.Tensor):
            print(f"  {key}: shape {checkpoint[key].shape}")
        else:
            print(f"  {key}: {type(checkpoint[key])}")
    
    # Check normalization
    if 'obs_rms_mean' in checkpoint:
        mean = checkpoint['obs_rms_mean']
        var = checkpoint['obs_rms_var']
        std = torch.sqrt(var + 1e-8)
        
        print("\nNormalization stats:")
        print(f"  Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
        print(f"  Std range: [{std.min():.3f}, {std.max():.3f}]")
        
        # Print per-component stats for velocity commands
        print("\nVelocity command normalization:")
        print("  Index | Component | Mean    | Std")
        print("  ------|-----------|---------|-------")
        labels = ["Lin vel X", "Lin vel Y", "Lin vel Z", "Ang vel X", "Ang vel Y", "Ang vel Z"]
        for i in range(6):
            print(f"  {i:5d} | {labels[i]:9s} | {mean[i]:7.3f} | {std[i]:6.3f}")
    
    return checkpoint

def simulate_model_behavior(checkpoint_path: str):
    """Simulate what the model should output for various commands"""
    from train_il import MLPPolicy
    
    print(f"\n=== Simulating Model Behavior ===")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Create model
    model = MLPPolicy(obs_dim=48, action_dim=12)
    
    # Load only actor weights
    actor_state_dict = {k: v for k, v in state_dict.items() if k.startswith('actor.')}
    model.actor.load_state_dict({k.replace('actor.', ''): v for k, v in actor_state_dict.items()})
    model.eval()
    
    # Get normalization stats
    obs_mean = checkpoint.get('obs_rms_mean', torch.zeros(48))
    obs_var = checkpoint.get('obs_rms_var', torch.ones(48))
    obs_std = torch.sqrt(obs_var + 1e-8)
    
    # Test various commands
    test_scenarios = [
        ("Zero command", 0.0, 0.0, 0.0),
        ("Slow forward", 0.1, 0.0, 0.0),
        ("Normal forward", 0.3, 0.0, 0.0),
        ("Fast forward", 0.5, 0.0, 0.0),
        ("Turn right", 0.0, 0.0, 0.5),
        ("Turn left", 0.0, 0.0, -0.5),
    ]
    
    results = []
    
    for scenario_name, vx, vy, wz in test_scenarios:
        # Create observation
        obs = torch.zeros(1, 48)
        obs[0, 0] = vx
        obs[0, 1] = vy
        obs[0, 5] = wz
        
        # Default joint positions
        default_joints = torch.tensor([0.0, -0.8, 1.6] * 4)
        obs[0, 6:18] = default_joints
        
        # Quaternion
        obs[0, 30:34] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        
        # Previous actions
        obs[0, 34:46] = default_joints
        
        # Clock
        obs[0, 46] = 0.0
        obs[0, 47] = 1.0
        
        # Normalize
        obs_norm = (obs - obs_mean) / obs_std
        
        # Get output
        with torch.no_grad():
            action = model(obs_norm)
        
        # Compute action statistics
        action_np = action[0].numpy()
        action_magnitude = np.linalg.norm(action_np)
        action_diff_from_default = action_np - default_joints.numpy()
        diff_magnitude = np.linalg.norm(action_diff_from_default)
        
        results.append({
            'scenario': scenario_name,
            'command': (vx, vy, wz),
            'action': action_np,
            'magnitude': action_magnitude,
            'diff_from_default': diff_magnitude
        })
        
        print(f"\n{scenario_name} (vx={vx}, vy={vy}, wz={wz}):")
        print(f"  Action magnitude: {action_magnitude:.3f}")
        print(f"  Diff from default: {diff_magnitude:.3f}")
        print(f"  First 3 joints (deg): [{np.degrees(action_np[0]):6.1f}, {np.degrees(action_np[1]):6.1f}, {np.degrees(action_np[2]):6.1f}]")
        print(f"  Action range: [{action_np.min():.3f}, {action_np.max():.3f}]")
    
    return results

def compare_with_working_model():
    """Compare IL model with a known working RSL_RL model"""
    print("\n=== Comparing Models ===")
    
    models_to_test = [
        ("/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model_normalized_rsl_rl.pt", "IL Model"),
        ("/workspace/SpotDMouse/P2-Terrain_Challenge/sim2real/model_9999_with_stats.pt", "Original RL Model")
    ]
    
    for model_path, model_name in models_to_test:
        try:
            print(f"\n--- {model_name} ---")
            checkpoint = load_and_inspect_checkpoint(model_path)
            
            if 'IL' in model_name:
                results = simulate_model_behavior(model_path)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")

def check_action_scaling():
    """Check if actions need scaling or clipping"""
    print("\n=== Checking Action Scaling ===")
    
    # Load the IL model checkpoint
    checkpoint_path = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model_normalized_rsl_rl.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if there's action normalization
    if 'action_mean' in checkpoint:
        print("Action normalization found:")
        print(f"  Mean: {checkpoint['action_mean']}")
        print(f"  Std: {checkpoint['action_std']}")
    else:
        print("No action normalization found")
    
    # Check joint limits (typical for quadrupeds)
    print("\nTypical joint limits for quadrupeds:")
    print("  Hip: [-0.5, 0.5] rad")
    print("  Thigh: [-2.0, 0.5] rad") 
    print("  Calf: [0.0, 2.5] rad")
    
    # Check if model outputs are within reasonable ranges
    from train_il import MLPPolicy
    model = MLPPolicy()
    state_dict = checkpoint['model_state_dict']
    actor_state_dict = {k: v for k, v in state_dict.items() if k.startswith('actor.')}
    model.actor.load_state_dict({k.replace('actor.', ''): v for k, v in actor_state_dict.items()})
    model.eval()
    
    # Generate random inputs and check outputs
    random_inputs = torch.randn(100, 48)
    with torch.no_grad():
        outputs = model(random_inputs)
    
    print(f"\nModel output statistics (100 random inputs):")
    print(f"  Mean: {outputs.mean():.3f}")
    print(f"  Std: {outputs.std():.3f}")
    print(f"  Min: {outputs.min():.3f}")
    print(f"  Max: {outputs.max():.3f}")
    print(f"  % in [-3, 3]: {((outputs >= -3) & (outputs <= 3)).float().mean() * 100:.1f}%")

def create_minimal_test_checkpoint():
    """Create a minimal checkpoint that should definitely move"""
    print("\n=== Creating Minimal Test Checkpoint ===")
    
    # Create a simple policy that outputs oscillating joint positions
    class SimpleOscillatingPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.counter = 0
            
        def forward(self, obs):
            batch_size = obs.shape[0]
            actions = torch.zeros(batch_size, 12)
            
            # Extract velocity command
            vx = obs[:, 0]
            
            # Simple oscillating pattern scaled by velocity
            phase = self.counter * 0.1
            for i in range(batch_size):
                # Hip joints - small oscillation
                actions[i, [0, 3, 6, 9]] = 0.1 * vx[i] * torch.sin(torch.tensor(phase))
                
                # Thigh joints - larger oscillation
                actions[i, [1, 4, 7, 10]] = -0.8 + 0.3 * vx[i] * torch.sin(torch.tensor(phase))
                
                # Calf joints - opposite phase
                actions[i, [2, 5, 8, 11]] = 1.6 + 0.3 * vx[i] * torch.cos(torch.tensor(phase))
            
            self.counter += 1
            return actions
    
    # Create state dict in RSL_RL format
    from train_il import MLPPolicy
    dummy_model = MLPPolicy()
    state_dict = dummy_model.state_dict()
    
    # Create minimal checkpoint
    checkpoint = {
        'model_state_dict': state_dict,
        'optimizer_state_dict': {'state': {}, 'param_groups': [{}]},
        'iter': 0,
        'obs_rms_mean': torch.zeros(48),
        'obs_rms_var': torch.ones(48),
        'num_obs': 48,
        'num_actions': 12,
    }
    
    # Add dummy critic
    for i in range(0, 7, 2):
        checkpoint['model_state_dict'][f'critic.{i}.weight'] = torch.randn_like(state_dict[f'actor.{i}.weight'])
        checkpoint['model_state_dict'][f'critic.{i}.bias'] = torch.zeros_like(state_dict[f'actor.{i}.bias'])
    checkpoint['model_state_dict']['std'] = torch.ones(12) * 0.5
    
    # Save
    test_path = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/minimal_test.pt"
    torch.save(checkpoint, test_path)
    print(f"Saved minimal test checkpoint to: {test_path}")
    
    return test_path

def visualize_model_outputs():
    """Create visualization of model outputs"""
    from train_il import MLPPolicy
    import matplotlib.pyplot as plt
    
    print("\n=== Visualizing Model Outputs ===")
    
    # Load model
    checkpoint_path = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model_normalized_rsl_rl.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model = MLPPolicy()
    state_dict = checkpoint['model_state_dict']
    actor_state_dict = {k: v for k, v in state_dict.items() if k.startswith('actor.')}
    model.actor.load_state_dict({k.replace('actor.', ''): v for k, v in actor_state_dict.items()})
    model.eval()
    
    # Get normalization
    obs_mean = checkpoint.get('obs_rms_mean', torch.zeros(48))
    obs_var = checkpoint.get('obs_rms_var', torch.ones(48))
    obs_std = torch.sqrt(obs_var + 1e-8)
    
    # Test forward velocity sweep
    velocities = np.linspace(-0.5, 0.5, 21)
    outputs = []
    
    for vx in velocities:
        obs = torch.zeros(1, 48)
        obs[0, 0] = vx
        obs[0, 6:18] = torch.tensor([0.0, -0.8, 1.6] * 4)
        obs[0, 30:34] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        obs[0, 34:46] = obs[0, 6:18]
        obs[0, 46:48] = torch.tensor([0.0, 1.0])
        
        obs_norm = (obs - obs_mean) / obs_std
        
        with torch.no_grad():
            action = model(obs_norm)
        
        outputs.append(action[0].numpy())
    
    outputs = np.array(outputs)
    
    # Plot
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    joint_names = ['FR_hip', 'FR_thigh', 'FR_calf', 'FL_hip', 'FL_thigh', 'FL_calf',
                   'RR_hip', 'RR_thigh', 'RR_calf', 'RL_hip', 'RL_thigh', 'RL_calf']
    
    for i, (ax, joint_name) in enumerate(zip(axes.flat, joint_names)):
        ax.plot(velocities, np.degrees(outputs[:, i]), 'b-', linewidth=2)
        ax.axhline(y=np.degrees([0.0, -0.8, 1.6][i % 3]), color='r', linestyle='--', alpha=0.5, label='Default')
        ax.set_xlabel('Forward Velocity Command (m/s)')
        ax.set_ylabel('Joint Angle (deg)')
        ax.set_title(joint_name)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/model_output_visualization.png', dpi=150)
    print("Saved visualization to model_output_visualization.png")

def main():
    print("="*60)
    print("IL Model Debugging for Isaac Sim")
    print("="*60)
    
    # 1. Compare models
    compare_with_working_model()
    
    # 2. Check action scaling
    check_action_scaling()
    
    # 3. Visualize outputs
    visualize_model_outputs()
    
    # 4. Create test checkpoint
    test_checkpoint = create_minimal_test_checkpoint()
    
    print("\n" + "="*60)
    print("DEBUGGING SUMMARY")
    print("="*60)
    
    print("\n1. First, test with the minimal checkpoint to verify sim setup:")
    print(f"   cd /workspace/isaaclab")
    print(f"   CUDA_VISIBLE_DEVICES=2 python scripts/reinforcement_learning/rsl_rl/play.py \\")
    print(f"     --task=Isaac-Velocity-Flat-Custom-Quad-v0 \\")
    print(f"     --checkpoint={test_checkpoint} \\")
    print(f"     --num_envs 1")
    
    print("\n2. Check the visualization:")
    print(f"   $BROWSER /workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/model_output_visualization.png")
    
    print("\n3. If the minimal checkpoint moves but IL doesn't, the issue is likely:")
    print("   - Normalization mismatch")
    print("   - Action scaling")
    print("   - Model outputs are too small")
    
    print("\n4. Try running the IL model with different commands:")
    print("   - Use keyboard: W/A/S/D for movement")
    print("   - Check if ANY command produces movement")

if __name__ == "__main__":
    main()