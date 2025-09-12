#!/usr/bin/env python3
# filepath: /workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/assess_il_model.py
"""
Assess IL model by testing various forward commands through the MLP
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from train_il import MLPPolicy

def create_observation_batch(commands):
    """Create a batch of observations with different velocity commands"""
    batch_size = len(commands)
    obs = torch.zeros(batch_size, 48)
    
    for i, (vx, vy, wz) in enumerate(commands):
        # Velocity commands
        obs[i, 0] = vx  # Linear velocity X
        obs[i, 1] = vy  # Linear velocity Y
        obs[i, 2] = 0.0  # Linear velocity Z
        obs[i, 3] = 0.0  # Angular velocity X
        obs[i, 4] = 0.0  # Angular velocity Y
        obs[i, 5] = wz  # Angular velocity Z
        
        # Default standing pose for joints
        default_joint_pos = [0.0, -0.8, 1.6] * 4
        for j, pos in enumerate(default_joint_pos):
            obs[i, 6 + j] = pos
        
        # Quaternion (upright)
        obs[i, 30:34] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        
        # Previous actions = current joint positions
        obs[i, 34:46] = obs[i, 6:18]
        
        # Clock phase
        obs[i, 46] = 0.0  # sin
        obs[i, 47] = 1.0  # cos
    
    return obs

def assess_il_model(model_path):
    """Assess IL model outputs for various commands"""
    
    print(f"Loading IL model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Load model
    model = MLPPolicy(obs_dim=48, action_dim=12)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test commands: (vx, vy, wz)
    test_commands = [
        (0.0, 0.0, 0.0),   # Stand still
        (0.1, 0.0, 0.0),   # Slow forward
        (0.3, 0.0, 0.0),   # Normal forward
        (0.5, 0.0, 0.0),   # Fast forward
        (-0.2, 0.0, 0.0),  # Backward
        (0.0, 0.2, 0.0),   # Sideways right
        (0.0, -0.2, 0.0),  # Sideways left
        (0.0, 0.0, 0.5),   # Turn right
        (0.0, 0.0, -0.5),  # Turn left
        (0.3, 0.0, 0.3),   # Forward + turn
    ]
    
    labels = [
        "Stand", "Slow fwd", "Normal fwd", "Fast fwd", "Backward",
        "Right", "Left", "Turn R", "Turn L", "Fwd+Turn"
    ]
    
    # Create observations
    obs_batch = create_observation_batch(test_commands)
    
    # Test without normalization first
    print("\n=== Testing WITHOUT normalization ===")
    with torch.no_grad():
        outputs_raw = model(obs_batch)
    
    print_outputs(test_commands, labels, outputs_raw, "Raw")
    
    # Test with different normalization approaches
    normalizations = []
    
    # 1. No normalization (identity)
    normalizations.append(("No norm", torch.zeros(48), torch.ones(48)))
    
    # 2. Simple normalization (reasonable defaults)
    simple_mean = torch.zeros(48)
    simple_std = torch.ones(48)
    simple_std[0:6] = 0.3    # velocities
    simple_std[6:18] = 0.5   # joint positions
    simple_std[18:30] = 2.0  # joint velocities
    normalizations.append(("Simple norm", simple_mean, simple_std))
    
    # 3. From checkpoint if available
    if 'obs_mean' in checkpoint and checkpoint['obs_mean'] is not None:
        normalizations.append(("Checkpoint norm", checkpoint['obs_mean'], checkpoint['obs_std']))
    
    # Test each normalization
    all_outputs = {}
    for norm_name, obs_mean, obs_std in normalizations:
        print(f"\n=== Testing with {norm_name} ===")
        obs_norm = (obs_batch - obs_mean) / (obs_std + 1e-8)
        
        with torch.no_grad():
            outputs = model(obs_norm)
        
        all_outputs[norm_name] = outputs
        print_outputs(test_commands, labels, outputs, norm_name)
    
    # Visualize comparison
    visualize_assessment(test_commands, labels, all_outputs)
    
    # Analyze patterns
    analyze_patterns(test_commands, labels, all_outputs)

def print_outputs(commands, labels, outputs, norm_type):
    """Print output summary"""
    print(f"\n{norm_type} outputs:")
    print("Command         | Output stats            | Sample joints (deg)")
    print("----------------|-------------------------|--------------------")
    
    for i, (label, (vx, vy, wz)) in enumerate(zip(labels, commands)):
        out = outputs[i].numpy()
        # Convert first 3 joints to degrees for readability
        joints_deg = np.degrees(out[:3])
        print(f"{label:15s} | "
              f"mean={out.mean():6.3f} std={out.std():5.3f} | "
              f"[{joints_deg[0]:5.1f}, {joints_deg[1]:5.1f}, {joints_deg[2]:5.1f}]")

def analyze_patterns(commands, labels, all_outputs):
    """Analyze patterns in model outputs"""
    print("\n=== Pattern Analysis ===")
    
    # Check if outputs change with velocity
    for norm_name, outputs in all_outputs.items():
        print(f"\n{norm_name}:")
        
        # Compare stand vs forward
        stand_out = outputs[0].numpy()
        fwd_out = outputs[2].numpy()
        diff = np.abs(fwd_out - stand_out)
        
        print(f"  Stand vs Forward: max diff = {diff.max():.3f}, mean diff = {diff.mean():.3f}")
        
        # Check output range
        all_out = outputs.numpy()
        print(f"  Output range: [{all_out.min():.3f}, {all_out.max():.3f}]")
        
        # Check if outputs are stuck
        if diff.max() < 0.01:
            print("  ⚠️  WARNING: Outputs barely change with velocity!")
        elif all_out.std() < 0.01:
            print("  ⚠️  WARNING: All outputs very similar!")
        else:
            print("  ✓ Outputs vary with commands")

def visualize_assessment(commands, labels, all_outputs):
    """Create visualization of model behavior"""
    
    n_commands = len(commands)
    n_norms = len(all_outputs)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Plot 1: Output magnitude vs command
    ax = axes[0]
    for norm_name, outputs in all_outputs.items():
        magnitudes = [np.linalg.norm(out.numpy()) for out in outputs]
        ax.plot(range(n_commands), magnitudes, 'o-', label=norm_name, markersize=8)
    ax.set_xlabel('Command')
    ax.set_ylabel('Output Magnitude')
    ax.set_title('Output Magnitude vs Command')
    ax.set_xticks(range(n_commands))
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Joint variation
    ax = axes[1]
    for norm_name, outputs in all_outputs.items():
        variations = outputs.std(dim=0).numpy()
        ax.plot(variations, 'o-', label=norm_name, markersize=6)
    ax.set_xlabel('Joint Index')
    ax.set_ylabel('Std Dev Across Commands')
    ax.set_title('Joint Output Variation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Heatmap of outputs for one normalization
    ax = axes[2]
    norm_name = list(all_outputs.keys())[0]
    outputs = all_outputs[norm_name].numpy()
    im = ax.imshow(outputs.T, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_xlabel('Command')
    ax.set_ylabel('Joint')
    ax.set_title(f'Output Heatmap ({norm_name})')
    ax.set_xticks(range(n_commands))
    ax.set_xticklabels(labels, rotation=45)
    plt.colorbar(im, ax=ax)
    
    # Plot 4: Forward velocity response
    ax = axes[3]
    fwd_velocities = [cmd[0] for cmd in commands]
    for norm_name, outputs in all_outputs.items():
        # Use hip joint average as indicator
        hip_responses = outputs[:, [0, 3, 6, 9]].mean(dim=1).numpy()
        ax.plot(fwd_velocities[:5], hip_responses[:5], 'o-', label=norm_name, markersize=8)
    ax.set_xlabel('Forward Velocity Command')
    ax.set_ylabel('Average Hip Joint Output')
    ax.set_title('Velocity Response')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/il_model_assessment.png', dpi=150)
    print(f"\nSaved assessment plot to il_model_assessment.png")
    print(f"View with: $BROWSER /workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/il_model_assessment.png")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Assess IL model behavior")
    parser.add_argument("--model",
                        default="/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model.pt",
                        help="Path to IL model")
    
    args = parser.parse_args()
    
    assess_il_model(args.model)

if __name__ == "__main__":
    main()

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